"""
Integer reference model of one on-chip CNN training step (the mnist_cnn arch:
conv 1->8 s1 / conv 8->16 s2 / conv 16->16 s2, all 3x3 p1 relu, flatten HWC 784,
FC 784->16 bias-free, softmax over 10). Batch 4.

Mirrors the EXACT hardware arithmetic (GEMM ints, relu/bypass shifts, fixed-point
softmax, sub/mask, promote/accumulate/requant) so the bench asserts bit-exact.
numpy int64 keeps it exact AND fast; numpy >> on negatives is arithmetic (floor),
same as the RTL's >>>. im2col/kernel layouts match tb_mnist_cnn.py exactly.

col2im (the backward scatter-add) is HOST work on the chip too — the model and
the bench host share this function, so it is not part of the bit-exact contract
with the hardware; everything else is.
"""

import math

import numpy as np

# ── softmax fixed-point primitives (identical to exp_lut.sv / reciprocal.sv) ──
LUT     = [int(2 ** (-i / 256.0) * 32768 + 0.5) for i in range(256)]
SEEDLUT = [int(65536 / (256 + i) + 0.5) for i in range(256)]
SCALE_SH = 16


def exp_fixed(v):
    return LUT[v & 0xFF] >> ((v >> 8) & 0x1F)


def recip_fixed(D):
    e = D.bit_length() - 1
    x0 = SEEDLUT[(D >> (e - 8)) & 0xFF] << (24 - e)
    return ((x0 * ((1 << 33) - D * x0)) >> 32) & 0xFFFFFFFF


def ref_softmax_row(x, scale):
    m = max(x)
    es = []
    for xn in x:
        vfull = ((m - xn) * scale) >> SCALE_SH
        es.append(exp_fixed(0x1FFF if vfull > 0x1FFF else vfull))
    recip = recip_fixed(sum(es))
    out = []
    for e in es:
        qv = (127 * e * recip + (1 << 31)) >> 32
        out.append(127 if qv > 127 else qv)
    return out


def softmax_rows(L, n_valid, scale):
    """HW softmax N=n_valid on INT32 logit rows; padded lanes -> 0."""
    out = np.zeros(L.shape, dtype=np.int64)
    for r in range(L.shape[0]):
        out[r, :n_valid] = ref_softmax_row([int(v) for v in L[r, :n_valid]], scale)
    return out


# ── HW op mirrors (numpy int64, exact) ──────────────────────────────────────
def sat8(x):
    return np.clip(x, -128, 127)


def gemm_i32(A, W):
    """out_fmt=1: raw INT32 accumulator (values never overflow 32b here)."""
    return A.astype(np.int64) @ W.astype(np.int64)


def gemm_relu(A, W, shift):
    z = gemm_i32(A, W)
    return np.minimum(np.where(z > 0, z, 0) >> shift, 127)


def gemm_bypass(A, W, shift):
    return sat8(gemm_i32(A, W) >> shift)


def ew_sub(A, B):
    return sat8(A - B)


def ew_mask(A, B):
    return np.where(B > 0, A, 0)


def opt_accumulate(Mst, dW, lr, sha):
    d = (dW.astype(np.int64) * lr) >> sha
    return np.clip(Mst - d, -(1 << 31), (1 << 31) - 1)


def opt_requant(Mst, shp):
    return sat8(Mst >> shp)


# ── host-side data shaping (same functions drive the bench's host code) ─────
def im2col(FM, KH, KW, stride, pad):
    """FM (B,H,W,C) -> A ((B*Ho*Wo) x KH*KW*C); k = (kh*KW+kw)*C+ci, rows
    b-major then (oh,ow) raster — matches tb_mnist_cnn's im2col exactly."""
    B, H, W, C = FM.shape
    Ho = (H + 2 * pad - KH) // stride + 1
    Wo = (W + 2 * pad - KW) // stride + 1
    P = np.zeros((B, H + 2 * pad, W + 2 * pad, C), dtype=np.int64)
    P[:, pad:pad + H, pad:pad + W, :] = FM
    A = np.zeros((B, Ho, Wo, KH * KW * C), dtype=np.int64)
    for kh in range(KH):
        for kw in range(KW):
            patch = P[:, kh:kh + Ho * stride:stride, kw:kw + Wo * stride:stride, :]
            A[:, :, :, (kh * KW + kw) * C:(kh * KW + kw + 1) * C] = patch
    return A.reshape(B * Ho * Wo, KH * KW * C)


def col2im(dA, B, H, W, C, KH, KW, stride, pad):
    """Inverse of im2col: scatter-ADD patch gradients back to (B,H,W,C).
    Overlaps sum — this is the host's arithmetic in CNN training."""
    Ho = (H + 2 * pad - KH) // stride + 1
    Wo = (W + 2 * pad - KW) // stride + 1
    dP = np.zeros((B, H + 2 * pad, W + 2 * pad, C), dtype=np.int64)
    dA4 = dA.reshape(B, Ho, Wo, KH * KW * C)
    for kh in range(KH):
        for kw in range(KW):
            dP[:, kh:kh + Ho * stride:stride, kw:kw + Wo * stride:stride, :] += \
                dA4[:, :, :, (kh * KW + kw) * C:(kh * KW + kw + 1) * C]
    return dP[:, pad:pad + H, pad:pad + W, :]


def kernel_to_W(kernel):
    """kernel (Cout,KH,KW,Cin) -> GEMM W (KH*KW*Cin x Cout), k-order as im2col."""
    Cout, KH, KW, Cin = kernel.shape
    return kernel.transpose(1, 2, 3, 0).reshape(KH * KW * Cin, Cout).astype(np.int64)


# ── the architecture ────────────────────────────────────────────────────────
#  (Cin, Cout, KH, KW, stride, pad, H_in, W_in) — mirrors tb_mnist_cnn ARCH
ARCH = [
    (1,  8, 3, 3, 1, 1, 28, 28),   # -> (B,28,28,8)
    (8, 16, 3, 3, 2, 1, 28, 28),   # -> (B,14,14,16)
    (16, 16, 3, 3, 2, 1, 14, 14),  # -> (B,7,7,16)
]
FLAT, NCLS, B = 784, 10, 4


def init_state(rng, C):
    """Random per-layer INT8 init (W2 pad-class cols zeroed); masters promoted."""
    st = {}
    for l, (ci, co, kh, kw, *_r) in enumerate(ARCH):
        k = kh * kw * ci
        st[f"W{l}"] = np.array([[rng.randint(-C["wi"][l], C["wi"][l])
                                 for _ in range(co)] for _ in range(k)], dtype=np.int64)
    st["Wfc"] = np.array([[rng.randint(-C["wi"][3], C["wi"][3]) if c < NCLS else 0
                           for c in range(16)] for _ in range(FLAT)], dtype=np.int64)
    for name in ("W0", "W1", "W2", "Wfc"):
        st["M" + name] = st[name] << C["shp"]
    return st


def forward(st, X, C):
    """X (B,28,28,1) int. Returns A-matrices + activations (all HW tensors)."""
    t = {}
    fm = X
    for l, (ci, co, kh, kw, s, p, h, w) in enumerate(ARCH):
        t[f"A{l}"] = im2col(fm, kh, kw, s, p)
        t[f"a{l}"] = gemm_relu(t[f"A{l}"], st[f"W{l}"], C["s"][l])
        ho = (h + 2 * p - kh) // s + 1
        wo = (w + 2 * p - kw) // s + 1
        fm = t[f"a{l}"].reshape(B, ho, wo, co)
    t["flat"] = fm.reshape(B, FLAT)                        # HWC flatten = reshape
    t["logits"] = gemm_i32(t["flat"], st["Wfc"])           # INT32, raw
    t["p"] = softmax_rows(t["logits"], NCLS, C["sm"])
    return t


def train_step(st, X, onehot, C):
    """One full step; mutates st; returns all intermediates for assertions."""
    t = forward(st, X, C)
    t["dz"] = ew_sub(t["p"], onehot)                                    # 4x16
    t["dWfc"] = gemm_i32(t["flat"].T, t["dz"])                          # 784x16
    t["dflat"] = gemm_bypass(t["dz"], st["Wfc"].T, C["sg"][2])          # 4x784
    up = t["dflat"].reshape(B * 49, 16)                                 # -> a2 rows
    for l in (2, 1, 0):
        ci, co, kh, kw, s, p, h, w = ARCH[l]
        t[f"dz{l}"] = ew_mask(up, t[f"a{l}"])                           # relu' gate (HW)
        t[f"dW{l}"] = gemm_i32(t[f"A{l}"].T, t[f"dz{l}"])               # INT32 (HW)
        if l > 0:
            t[f"dA{l}"] = gemm_bypass(t[f"dz{l}"], st[f"W{l}"].T, C["sg"][l - 1])
            dFM = col2im(t[f"dA{l}"], B, h, w, ci, kh, kw, s, p)        # HOST adds
            up = sat8(dFM >> C["sc"][l - 1]).reshape(-1, ci)            # host requant
    for i, (name, d) in enumerate((("W0", "dW0"), ("W1", "dW1"),
                                   ("W2", "dW2"), ("Wfc", "dWfc"))):
        st["M" + name] = opt_accumulate(st["M" + name], t[d], C["lr"], C["sha"][i])
        st[name] = opt_requant(st["M" + name], C["shp"])
    return t


def batch_loss(p, labels):
    return sum(-math.log(max(int(p[r, y]) / 127.0, 1e-4))
               for r, y in enumerate(labels)) / len(labels)


def predict(st, X, C):
    """Forward-only argmax over the 10 real classes."""
    lg = forward(st, X, C)["logits"]
    return np.argmax(lg[:, :NCLS], axis=1)
