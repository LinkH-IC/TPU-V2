"""
attn_int_model.py — exact-arithmetic reference for the attention / encoder bench.

Integer side mirrors the hardware bit-for-bit: GEMM (bias-free) with bypass /
inline-GELU requant, INT32 raw scores, the fixed-point softmax and layernorm
(exp/reciprocal/rsqrt LUT models identical to tb_top_softmax), and the sat8
elementwise add. This is the bit-exact contract the bench asserts against.

Float side mirrors the SAME graph in real arithmetic, carrying each integer
tensor's exact quantization scale q (value ≈ int/q), with every sat8/clamp
mirrored as a clip — so max|int − float·q| measures pure quantization error
in output LSBs. Reported as a metric, not the bit-exact contract.

Run standalone for the tuning report:  python3 attn_int_model.py
"""

import math
import random

import numpy as np

# ── Sealed constants ────────────────────────────────────────────────────────
# Attention block (raw INT8 tokens, qX=64): qQ=qK=64·128/2^sp=32, qV=32,
# qO=127·qV/2^sav, qY=qO·128/2^so. sm_scale ⇒ τ_eff≈1/√D (base-2 exp:
# τ_eff = sm_scale·ln2·qQ·qK/2^24).
CA = dict(S=24, D=16, sp=8, sv=8, sav=7, so=6, sm_scale=5909, qX=64, qW=128)

# Encoder layer (pre-LN): LN out Q4 (qU=16) feeds attention ⇒ sp/sv drop to 6
# (qQ=qK=qV=32, same τ product). Residual alignment: qT=127·32·128/2^(7+6)=63.5
# vs qX=64 (ρ=127/128, mirrored in float); FFN qH=16 (GELU F=4, 2^sg=16·128/16),
# qF=16·128/2^sf=64=qX.
CE = dict(S=24, D=16, F=64, sp=6, sv=6, sav=7, so=6, sm_scale=5909,
          sh_ln=8, sg=7, sf=5, qX=64, qW=128)

# ── Fixed-point primitive models (identical to the RTL LUTs) ────────────────
LUT     = [int(2 ** (-i / 256.0) * 32768 + 0.5) for i in range(256)]
SEEDLUT = [int(65536 / (256 + i) + 0.5) for i in range(256)]
RSQ_SEED = [int(2048 / math.sqrt(i + 0.5) + 0.5) if i >= 64 else 0 for i in range(256)]
SCALE_SH = 16
GELU_FBITS = 4


def _gelu_rom(u):
    s = u - 256 if u >= 128 else u
    real = s / (1 << GELU_FBITS)
    v = math.floor(0.5 * real * (1 + math.erf(real / math.sqrt(2))) * (1 << GELU_FBITS) + 0.5)
    return max(-128, min(127, v))


GELU_ROM = [_gelu_rom(u) for u in range(256)]


def sat8(x):
    return 127 if x > 127 else (-128 if x < -128 else x)


def to_signed(v, w):
    return v - (1 << w) if v & (1 << (w - 1)) else v


def transpose(M):
    return [list(r) for r in zip(*M)]


def exp_fixed(v):
    return LUT[v & 0xFF] >> ((v >> 8) & 0x1F)


def recip_fixed(D):
    e = D.bit_length() - 1
    x0 = SEEDLUT[(D >> (e - 8)) & 0xFF] << (24 - e)
    return ((x0 * ((1 << 33) - D * x0)) >> 32) & 0xFFFFFFFF


def rsqrt_fixed(D):
    e = D.bit_length() - 1
    p = e >> 1
    m = (D >> (2 * p - 16)) if 2 * p >= 16 else (D << (16 - 2 * p))
    y0 = RSQ_SEED[m >> 10]
    t = m * y0 * y0
    r = 0 if t > (3 << 32) else (3 << 32) - t
    y1 = min((y0 * r) >> 25, 1 << 16)
    return (y1 << (8 - p)) if p <= 8 else (y1 >> (p - 8))


# ── Integer ops (the hardware, exactly) ─────────────────────────────────────
def gemm8(A, W, shift, gelu=False):
    """Bias-free GEMM + drain requant: bypass sat8(acc>>>sh) or inline GELU."""
    M, K, N = len(A), len(W), len(W[0])
    out = [[0] * N for _ in range(M)]
    for m in range(M):
        for n in range(N):
            acc = sum(A[m][k] * W[k][n] for k in range(K))
            s = acc >> shift                       # Python >> floors == Verilog >>>
            if gelu:
                s = 127 if s > 127 else (-128 if s < -128 else s)
                out[m][n] = GELU_ROM[s & 0xFF]
            else:
                out[m][n] = sat8(s)
    return out


def gemm_i32(A, W):
    """GEMM out_fmt=INT32, bias off: raw 32-bit two's-complement accumulators."""
    M, K, N = len(A), len(W), len(W[0])
    mask = (1 << 32) - 1
    return [[to_signed(sum(A[m][k] * W[k][n] for k in range(K)) & mask, 32)
             for n in range(N)] for m in range(M)]


def softmax_row(x, scale):
    m = max(x)
    es = []
    for xn in x:
        v = ((m - xn) * scale) >> SCALE_SH
        es.append(exp_fixed(0x1FFF if v > 0x1FFF else v))
    recip = recip_fixed(sum(es))
    out = []
    for e in es:
        qv = (127 * e * recip + (1 << 31)) >> 32
        out.append(127 if qv > 127 else qv)
    return out


def softmax_mat(SC, scale):
    return [softmax_row(r, scale) for r in SC]


def ln_row(x, gam, bet, sh):
    N = len(x)
    rn = recip_fixed(N << 15)                      # 2^17/N
    mu = (sum(x) * rn) >> 9                        # Q8 mean
    dd = [(xi << 8) - mu for xi in x]
    sig2 = ((sum(d * d for d in dd) * rn) >> 17) + 1
    rs = rsqrt_fixed(sig2)                         # 2^16/sigma
    rnd = (1 << (sh - 1)) if sh else 0
    return [sat8(((max(-32768, min(32767, (d * rs) >> 18)) * g + rnd) >> sh) + b)
            for d, g, b in zip(dd, gam, bet)]


def ln_mat(X, gam, bet, sh):
    return [ln_row(r, gam, bet, sh) for r in X]


def add8(A, B):
    return [[sat8(a + b) for a, b in zip(ra, rb)] for ra, rb in zip(A, B)]


# ── Integer graphs ──────────────────────────────────────────────────────────
def attention_int(X, W, C):
    """Single-head attention block. W = dict(Wq,Wk,Wv,Wo). Returns all tensors."""
    Q = gemm8(X, W["Wq"], C["sp"])
    K = gemm8(X, W["Wk"], C["sp"])
    V = gemm8(X, W["Wv"], C["sv"])
    SC = gemm_i32(Q, transpose(K))
    A = softmax_mat(SC, C["sm_scale"])
    O = gemm8(A, V, C["sav"])
    Y = gemm8(O, W["Wo"], C["so"])
    return dict(Q=Q, K=K, V=V, SC=SC, A=A, O=O, Y=Y)


def encoder_int(X, W, C):
    """Pre-LN encoder layer: r = x + Attn(LN1(x)); y = r + FFN(LN2(r)),
    FFN = GELU_inline(v·W1)·W2. W adds W1, W2, g1, b1, g2, b2."""
    U = ln_mat(X, W["g1"], W["b1"], C["sh_ln"])
    at = attention_int(U, W, C)
    R = add8(X, at["Y"])
    VL = ln_mat(R, W["g2"], W["b2"], C["sh_ln"])
    H = gemm8(VL, W["W1"], C["sg"], gelu=True)
    F = gemm8(H, W["W2"], C["sf"])
    Y = add8(R, F)
    out = dict(at, T=at["Y"])
    out.update(U=U, R=R, VL=VL, H=H, F=F, Y=Y)
    return out


# ── Float mirror (same graph, real arithmetic, clips mirrored) ──────────────
def _clipq(x, q):
    return np.clip(x, -128.0 / q, 127.0 / q)


def _softmax_f(S, tau):
    Z = tau * S
    Z = Z - Z.max(axis=1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)


def _ln_f(X, gam, bet):
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True) + 1e-12
    y = (np.array(gam) / 64.0) * (X - mu) / sd + np.array(bet) / 16.0
    return _clipq(y, 16)


def _gelu_f(x):
    return 0.5 * x * (1 + np.vectorize(math.erf)(x / math.sqrt(2)))


def attention_float(Xf, W, C, qX):
    """Xf = float tokens (value = int/qX). Returns {name: (float mat, q)}."""
    qW = C["qW"]
    Wq, Wk = np.array(W["Wq"]) / qW, np.array(W["Wk"]) / qW
    Wv, Wo = np.array(W["Wv"]) / qW, np.array(W["Wo"]) / qW
    qQ = qX * qW / (1 << C["sp"])
    qV = qX * qW / (1 << C["sv"])
    qO = 127 * qV / (1 << C["sav"])
    qY = qO * qW / (1 << C["so"])
    tau = C["sm_scale"] * math.log(2) * qQ * qQ / (1 << 24)
    Q = _clipq(Xf @ Wq, qQ)
    K = _clipq(Xf @ Wk, qQ)
    V = _clipq(Xf @ Wv, qV)
    SC = Q @ K.T
    A = _softmax_f(SC, tau)                        # exp arg = τ·(q̂·k̂ᵀ) exactly
    O = _clipq(A @ V, qO)
    Y = _clipq(O @ Wo, qY)
    return dict(Q=(Q, qQ), K=(K, qQ), V=(V, qV), SC=(SC, qQ * qQ),
                A=(A, 127), O=(O, qO), Y=(Y, qY))


def encoder_float(Xf, W, C):
    qX = C["qX"]
    U = _ln_f(Xf, W["g1"], W["b1"])
    at = attention_float(U, W, C, 16)
    T, qT = at["Y"]
    R = _clipq(Xf + (qT / qX) * T, qX)             # sat8 add; ρ=qT/qX scale skew
    VL = _ln_f(R, W["g2"], W["b2"])
    W1, W2 = np.array(W["W1"]) / C["qW"], np.array(W["W2"]) / C["qW"]
    pre = np.clip(VL @ W1, -8.0, 127.0 / 16)       # GELU input clamp8 at Q4
    H = _clipq(_gelu_f(pre), 16)
    qF = 16 * C["qW"] / (1 << C["sf"])
    F = _clipq(H @ W2, qF)
    Y = _clipq(R + (qF / qX) * F, qX)
    out = dict(at, T=(T, qT))
    out.update(U=(U, 16), R=(R, qX), VL=(VL, 16), H=(H, 16),
               F=(F, qF), Y=(Y, qX))
    return out


# ── Case generators (single source of draws for model AND bench) ────────────
def _draw(rng, R, Cc, mag):
    return [[rng.randint(-mag, mag) for _ in range(Cc)] for _ in range(R)]


def make_attention_case(seed=0xA77):
    rng = random.Random(seed)
    C = CA
    X = _draw(rng, C["S"], C["D"], 100)
    W = {k: _draw(rng, C["D"], C["D"], 64) for k in ("Wq", "Wk", "Wv", "Wo")}
    return X, W


def make_encoder_case(seed=0xE9C):
    rng = random.Random(seed)
    C = CE
    X = _draw(rng, C["S"], C["D"], 64)
    W = {k: _draw(rng, C["D"], C["D"], m)
         for k, m in (("Wq", 48), ("Wk", 48), ("Wv", 48), ("Wo", 24))}
    W["W1"] = _draw(rng, C["D"], C["F"], 64)
    W["W2"] = _draw(rng, C["F"], C["D"], 16)
    W["g1"] = [rng.randint(40, 88) for _ in range(C["D"])]
    W["b1"] = [rng.randint(-16, 16) for _ in range(C["D"])]
    W["g2"] = [rng.randint(40, 88) for _ in range(C["D"])]
    W["b2"] = [rng.randint(-16, 16) for _ in range(C["D"])]
    return X, W


# ── Metrics / tuning report ─────────────────────────────────────────────────
def report(tag, ints, floats, order):
    print(f"\n── {tag} ──")
    print(f"{'tensor':>6} {'amax':>6} {'sat':>4} {'errLSB':>7}")
    for name in order:
        I = np.array(ints[name], dtype=float)
        Ff, q = floats[name]
        Ff = np.asarray(Ff)
        I = I[:Ff.shape[0], :Ff.shape[1]]
        amax = int(np.abs(I).max())
        lim = (1 << 31) - 1 if name == "SC" else 127
        nsat = int((np.abs(I) >= lim).sum())
        err = np.abs(I - Ff * q).max()
        print(f"{name:>6} {amax:>6} {nsat:>4} {err:>7.2f}")
    Iy = np.array(ints["Y"], dtype=float).ravel()
    Fy = (np.asarray(floats["Y"][0]) * floats["Y"][1]).ravel()
    cos = float(Iy @ Fy / (np.linalg.norm(Iy) * np.linalg.norm(Fy)))
    print(f"cos(Y int, Y float) = {cos:.5f}")
    return cos


if __name__ == "__main__":
    X, W = make_attention_case()
    ints = attention_int(X, W, CA)
    floats = attention_float(np.array(X) / CA["qX"], W, CA, CA["qX"])
    print(f"attention τ_eff = {CA['sm_scale'] * math.log(2) * 32 * 32 / (1 << 24):.4f}"
          f"  (ideal 1/√D = {1 / math.sqrt(CA['D']):.4f})")
    report("attention block", ints, floats, ["Q", "K", "V", "SC", "A", "O", "Y"])

    X, W = make_encoder_case()
    ints = encoder_int(X, W, CE)
    floats = encoder_float(np.array(X) / CE["qX"], W, CE)
    report("encoder layer", ints, floats,
           ["U", "Q", "K", "V", "SC", "A", "O", "T", "R", "VL", "H", "F", "Y"])
