"""
vit_int_model.py — exact-arithmetic reference (oracle) for the MNIST ViT.

The full trained ViT (train/mnist_vit) folded into the chip's integer
arithmetic. Multi-head aware (n_heads from the npz meta): attention runs
per head on d_k = D/NH column slices — bit-identical to the chip's per-head
N=16 dispatches at column-tile weight bases, with the head concat as a host
merge. τ_eff targets 1/√d_k. Op models are IMPORTED from tb/attention/
attn_int_model.py — the S27-proven single source of truth (GEMM requant,
INT32 scores, LUT softmax/layernorm, GELU ROM, sat8 add). New here, per the
RTL bias_add.sv semantics proven since M1: GEMM bias (per-column INT32 added
to the accumulator BEFORE the requant/activation), self-tested at import
against the bias-free proven ops.

Scale algebra (S27 discipline — every tensor carries an exact rational q,
value ≈ int/q; shifts are constrained, not free):
  qU = 16 fixed (LN LUT), A q=127 (softmax), GELU in/out q=16 (ROM Q4)
  residual stream at ONE scale qX: embed, Wo and W2 scales are CHOSEN so
  qT == qF == qX exactly (sat8 add needs equal scales); W1 scale must be a
  power of two (GELU input lands exactly on q=16); Wq/Wk/Wv scales are free
  (sm_scale absorbs qQ·qK into τ_eff = 1/√D); head is scale-free (argmax).

Calibration + accuracy: the bundled 100 test images inside trained_vit.npz
(user-scoped: this chapter proves the transformer RUNS on the chip; the float
reference on these 100 is 99/100). The bench rung imports this module and
asserts the chip equals vit_int() bit-for-bit, tensor by tensor.

Run standalone for the fold summary + health + accuracy:
    python3 vit_int_model.py
"""

import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "attention"))
from attn_int_model import (GELU_ROM, add8, gemm8, gemm_i32, ln_mat,
                            softmax_mat, transpose)

NPZ = HERE / "trained_vit.npz"
D, FF, S, NPX, NL = 32, 128, 49, 16, 2
META = json.loads(str(np.load(NPZ)["meta"]))
NH = int(META.get("n_heads", 1))                   # heads (d_k = D/NH)
DK = D // NH
TAU = 1.0 / math.sqrt(DK)

# ── numpy GEMM with bias (bias_add.sv: acc + bias[col], pre-requant) ─────────
_ROM = np.array(GELU_ROM, dtype=np.int64)


def gemm8b(A, W, bias, shift, gelu=False):
    """sat8((ΣA·W + bias) >>> shift) or GELU ROM on the clamped Q4 sum."""
    acc = np.asarray(A, np.int64) @ np.asarray(W, np.int64) \
        + np.asarray(bias, np.int64)[None, :]
    assert (np.abs(acc) < 1 << 31).all(), "INT32 accumulator overflow"
    s = acc >> shift
    if gelu:
        s = np.clip(s, -128, 127)
        return _ROM[(s & 0xFF).astype(np.int64)].tolist()
    return np.clip(s, -128, 127).tolist()


def gemm_i32b(A, W, bias):
    """out_fmt=INT32 with bias: raw 32-bit two's-complement accumulators."""
    acc = np.asarray(A, np.int64) @ np.asarray(W, np.int64) \
        + np.asarray(bias, np.int64)[None, :]
    return (((acc + (1 << 31)) & ((1 << 32) - 1)) - (1 << 31)).tolist()


def _selftest():
    rng = np.random.default_rng(7)
    A = rng.integers(-127, 128, (5, 9)).tolist()
    W = rng.integers(-127, 128, (9, 6)).tolist()
    b = rng.integers(-40000, 40000, 6).tolist()
    z = [0] * 6
    assert gemm8b(A, W, z, 3) == gemm8(A, W, 3)
    assert gemm8b(A, W, z, 2, gelu=True) == gemm8(A, W, 2, gelu=True)
    assert gemm_i32b(A, W, z) == gemm_i32(A, W)
    # bias spec: data_out = data_in + bias_sel, before shift/sat (bias_add.sv)
    ref = [[max(-128, min(127, (sum(A[m][k] * W[k][n] for k in range(9))
                                + b[n]) >> 3)) for n in range(6)] for m in range(5)]
    assert gemm8b(A, W, b, 3) == ref
    ref32 = [[sum(A[m][k] * W[k][n] for k in range(9)) + b[n]
              for n in range(6)] for m in range(5)]
    assert gemm_i32b(A, W, b) == ref32


_selftest()


# ── data ─────────────────────────────────────────────────────────────────────
def patchify(x784):
    """int8/float (N,784) → (N,49,16): 7x7 grid of 4x4 patches, row-major."""
    n = x784.shape[0]
    return x784.reshape(n, 7, 4, 7, 4).transpose(0, 1, 3, 2, 4).reshape(n, 49, 16)


def load():
    d = np.load(NPZ)
    P = {k: d[k].astype(np.float64) for k in d.files
         if k not in ("meta", "recipe", "X_test", "y_test")}
    return P, d["X_test"].astype(np.int64), d["y_test"].astype(np.int64)


# ── float forward (exact torch mirror) — calibration amax + reference ───────
def _lnf_exact(x, g, b, eps=1e-5):
    m = x.mean(-1, keepdims=True)
    v = x.var(-1, keepdims=True)
    return (x - m) / np.sqrt(v + eps) * g + b


_erf = np.vectorize(math.erf)


def _geluf(x):
    return 0.5 * x * (1.0 + _erf(x / math.sqrt(2.0)))


def _softf(x):
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def vit_float(P, X100):
    """Exact float forward on int8 images. Returns (pred, amax dict)."""
    am = {}
    x = patchify(X100.astype(np.float64) / 127.0)
    x = x @ P["embed_weight"].T + P["embed_bias"] + P["pos"]
    am["X0"] = np.abs(x).max()
    for i in range(NL):
        L = f"layers_{i}_"
        u = _lnf_exact(x, P[L + "ln1_weight"], P[L + "ln1_bias"])
        q = u @ P[L + "wq_weight"].T + P[L + "wq_bias"]
        k = u @ P[L + "wk_weight"].T + P[L + "wk_bias"]
        v = u @ P[L + "wv_weight"].T + P[L + "wv_bias"]
        am[f"Q{i}"], am[f"K{i}"], am[f"V{i}"] = (np.abs(t).max() for t in (q, k, v))
        n = q.shape[0]                             # split heads: (n,NH,S,DK)
        qh, kh, vh = (t.reshape(n, S, NH, DK).transpose(0, 2, 1, 3)
                      for t in (q, k, v))
        a = _softf(qh @ kh.transpose(0, 1, 3, 2) * TAU)
        o = (a @ vh).transpose(0, 2, 1, 3).reshape(n, S, D)   # concat heads
        am[f"O{i}"] = np.abs(o).max()
        t = o @ P[L + "wo_weight"].T + P[L + "wo_bias"]
        am[f"T{i}"] = np.abs(t).max()
        x = x + t
        am[f"R{i}"] = np.abs(x).max()
        vl = _lnf_exact(x, P[L + "ln2_weight"], P[L + "ln2_bias"])
        f = _geluf(vl @ P[L + "w1_weight"].T + P[L + "w1_bias"]) \
            @ P[L + "w2_weight"].T + P[L + "w2_bias"]
        am[f"F{i}"] = np.abs(f).max()
        x = x + f
        am[f"Y{i}"] = np.abs(x).max()
    p = _lnf_exact(x, P["lnf_weight"], P["lnf_bias"]).mean(axis=1)
    am["P"] = np.abs(p).max()
    logits = p @ P["head_weight"].T + P["head_bias"]
    return logits.argmax(-1), am


# ── the fold: float weights + calib amax → int weights + sealed constants ───
def _q8(Wf, s):
    W = np.rint(Wf * s).astype(np.int64)
    assert np.abs(W).max() <= 127, "int8 weight overflow"
    return W


def fold(P, am):
    """Returns (Wq dict of int lists, C dict of constants)."""
    C, Wq = {}, {}
    stream = max(am[k] for k in am if k[0] in "XRY")
    C["qX"] = float(1 << int(math.floor(math.log2(120.0 / stream))))
    qX = C["qX"]

    def shift_for(q_in, sW, amax):                 # smallest sh: q_out·amax ≤ 120
        return max(0, math.ceil(math.log2(q_in * sW * amax / 120.0)))

    # patch embed (+pos): qT target = qX exactly ⇒ sWe = qX·2^se/127
    se = int(math.floor(math.log2(127.0 * 127.0 / (np.abs(P["embed_weight"]).max() * qX))))
    sWe = qX * (1 << se) / 127.0
    Wq["We"] = _q8(P["embed_weight"].T, sWe)
    Wq["be"] = np.rint(P["embed_bias"] * 127.0 * sWe).astype(np.int64)
    Wq["pos"] = np.rint(P["pos"] * qX).astype(np.int64)
    assert np.abs(Wq["pos"]).max() <= 127
    C["se"] = se

    for i in range(NL):
        L, c = f"layers_{i}_", {}
        for nm in ("ln1", "ln2"):
            g = np.rint(P[L + nm + "_weight"] * 64).astype(np.int64)
            b = np.rint(P[L + nm + "_bias"] * 16).astype(np.int64)
            assert np.abs(g).max() <= 127 and np.abs(b).max() <= 127
            Wq[f"g{nm[-1]}_{i}"], Wq[f"b{nm[-1]}_{i}"] = g, b
        for nm, ak in (("wq", f"Q{i}"), ("wk", f"K{i}"), ("wv", f"V{i}")):
            sW = 127.0 / np.abs(P[L + nm + "_weight"]).max()
            sh = shift_for(16, sW, am[ak])
            Wq[nm[1] + str(i)] = _q8(P[L + nm + "_weight"].T, sW)
            Wq["b" + nm[1] + str(i)] = np.rint(P[L + nm + "_bias"] * 16 * sW).astype(np.int64)
            c["s" + nm[1]] = sh
            c["q" + nm[1].upper()] = 16 * sW / (1 << sh)
        c["sav"] = shift_for(1, 127 * c["qV"], am[f"O{i}"])   # qO = 127·qV/2^sav
        qO = 127 * c["qV"] / (1 << c["sav"])
        so = int(math.floor(math.log2(127.0 * qO / (np.abs(P[L + "wo_weight"]).max() * qX))))
        sWo = qX * (1 << so) / qO                              # qT == qX exactly
        Wq[f"o{i}"] = _q8(P[L + "wo_weight"].T, sWo)
        Wq[f"bo{i}"] = np.rint(P[L + "wo_bias"] * qO * sWo).astype(np.int64)
        c["so"] = so
        sg = int(math.floor(math.log2(127.0 / np.abs(P[L + "w1_weight"]).max())))
        Wq[f"w1_{i}"] = _q8(P[L + "w1_weight"].T, float(1 << sg))   # sW1 = 2^sg ⇒ GELU in q=16
        Wq[f"bw1_{i}"] = np.rint(P[L + "w1_bias"] * 16 * (1 << sg)).astype(np.int64)
        c["sg"] = sg
        sf = int(math.floor(math.log2(127.0 * 16.0 / (np.abs(P[L + "w2_weight"]).max() * qX))))
        sW2 = qX * (1 << sf) / 16.0                            # qF == qX exactly
        Wq[f"w2_{i}"] = _q8(P[L + "w2_weight"].T, sW2)
        Wq[f"bw2_{i}"] = np.rint(P[L + "w2_bias"] * 16 * sW2).astype(np.int64)
        c["sf"] = sf
        c["sm_scale"] = int(round(TAU * (1 << 24) / (math.log(2) * c["qQ"] * c["qK"])))
        assert 0 < c["sm_scale"] < (1 << 16)
        C[f"L{i}"] = c

    Wq["gf"] = np.rint(P["lnf_weight"] * 64).astype(np.int64)
    Wq["bf"] = np.rint(P["lnf_bias"] * 16).astype(np.int64)
    assert np.abs(Wq["gf"]).max() <= 127 and np.abs(Wq["bf"]).max() <= 127
    C["sp2"] = max(0, math.ceil(math.log2(784.0 * am["P"] / 120.0)))  # qP = 784/2^sp2
    C["qP"] = 784.0 / (1 << C["sp2"])
    sWh = 127.0 / np.abs(P["head_weight"]).max()
    C["sWh"] = sWh
    Wh = _q8(P["head_weight"].T, sWh)                          # 32×10 → pad to 16
    Wq["wh"] = np.pad(Wh, ((0, 0), (0, 6)))
    Wq["bh"] = np.pad(np.rint(P["head_bias"] * C["qP"] * sWh).astype(np.int64), (0, 6))
    for c in [C["L0"], C["L1"]]:
        assert all(0 <= c[s] <= 31 for s in ("sq", "sk", "sv", "sav", "so", "sg", "sf"))
    assert 0 <= C["se"] <= 31 and 0 <= C["sp2"] <= 31
    assert all(np.abs(Wq[k]).max() < (1 << 31) for k in Wq)
    return Wq, C


# ── integer forward (the chip, exactly) — one image ──────────────────────────
def vit_int(Wq, C, x784):
    """x784: 784 int8 pixels. Returns dict of every named int tensor."""
    ts = {}
    A = patchify(np.asarray(x784, np.int64)[None, :])[0].tolist()
    E = gemm8b(A, Wq["We"].tolist(), Wq["be"].tolist(), C["se"])
    X = add8(E, Wq["pos"].tolist())
    ts["X0"] = X
    for i in range(NL):
        c = C[f"L{i}"]
        U = ln_mat(X, Wq[f"g1_{i}"].tolist(), Wq[f"b1_{i}"].tolist(), 8)
        Q = gemm8b(U, Wq[f"q{i}"].tolist(), Wq[f"bq{i}"].tolist(), c["sq"])
        K = gemm8b(U, Wq[f"k{i}"].tolist(), Wq[f"bk{i}"].tolist(), c["sk"])
        V = gemm8b(U, Wq[f"v{i}"].tolist(), Wq[f"bv{i}"].tolist(), c["sv"])
        heads = []                                 # per-head attention (d_k cols)
        for h in range(NH):
            Qh = [r[h * DK:(h + 1) * DK] for r in Q]
            Kh = [r[h * DK:(h + 1) * DK] for r in K]
            Vh = [r[h * DK:(h + 1) * DK] for r in V]
            SCh = gemm_i32(Qh, transpose(Kh))
            Ah = softmax_mat(SCh, c["sm_scale"])
            Oh = gemm8(Ah, Vh, c["sav"])
            ts.update({f"SC{i}_{h}": SCh, f"A{i}_{h}": Ah, f"O{i}_{h}": Oh})
            heads.append(Oh)
        O = [sum((heads[h][r] for h in range(NH)), []) for r in range(S)]
        T = gemm8b(O, Wq[f"o{i}"].tolist(), Wq[f"bo{i}"].tolist(), c["so"])
        R = add8(X, T)
        VL = ln_mat(R, Wq[f"g2_{i}"].tolist(), Wq[f"b2_{i}"].tolist(), 8)
        H = gemm8b(VL, Wq[f"w1_{i}"].tolist(), Wq[f"bw1_{i}"].tolist(), c["sg"], gelu=True)
        F = gemm8b(H, Wq[f"w2_{i}"].tolist(), Wq[f"bw2_{i}"].tolist(), c["sf"])
        X = add8(R, F)
        ts.update({f"U{i}": U, f"Q{i}": Q, f"K{i}": K, f"V{i}": V, f"O{i}": O,
                   f"T{i}": T, f"R{i}": R, f"VL{i}": VL,
                   f"H{i}": H, f"F{i}": F, f"Y{i}": X})
    LNF = ln_mat(X, Wq["gf"].tolist(), Wq["bf"].tolist(), 8)
    Pl = gemm8([[1] * S], LNF, C["sp2"])
    logits = gemm_i32b(Pl, Wq["wh"].tolist(), Wq["bh"].tolist())
    ts.update(LNF=LNF, P=Pl, logits=logits,
              pred=int(np.argmax(logits[0][:10])))
    return ts


# ── quantized-float mirror (clips mirrored, exact q per tensor) — health ────
def _clipq(x, q):
    return np.clip(x, -128.0 / q, 127.0 / q)


def _lnq(x, g, b):
    m = x.mean(-1, keepdims=True)
    sd = x.std(-1, keepdims=True) + 1e-12
    return _clipq((g / 64.0) * (x - m) / sd + b / 16.0, 16.0)


def vit_qfloat(Wq, C, X100):
    """Batched quantized-float mirror. Returns {name: (float tensor, q)}."""
    qX = C["qX"]
    ts = {}
    x = patchify(X100.astype(np.float64) / 127.0)
    se, sWe = C["se"], qX * (1 << C["se"]) / 127.0
    x = _clipq(x @ (Wq["We"] / sWe) + Wq["be"] / (127.0 * sWe), qX)
    x = _clipq(x + Wq["pos"] / qX, qX)
    ts["X0"] = (x, qX)
    for i in range(NL):
        c = C[f"L{i}"]
        u = _lnq(x, Wq[f"g1_{i}"], Wq[f"b1_{i}"])
        sWq = c["qQ"] * (1 << c["sq"]) / 16.0
        sWk = c["qK"] * (1 << c["sk"]) / 16.0
        sWv = c["qV"] * (1 << c["sv"]) / 16.0
        q_ = _clipq(u @ (Wq[f"q{i}"] / sWq) + Wq[f"bq{i}"] / (16.0 * sWq), c["qQ"])
        k_ = _clipq(u @ (Wq[f"k{i}"] / sWk) + Wq[f"bk{i}"] / (16.0 * sWk), c["qK"])
        v_ = _clipq(u @ (Wq[f"v{i}"] / sWv) + Wq[f"bv{i}"] / (16.0 * sWv), c["qV"])
        tau = c["sm_scale"] * math.log(2) * c["qQ"] * c["qK"] / (1 << 24)
        n = u.shape[0]                             # split heads: (n,NH,S,DK)
        qh, kh, vh = (t.reshape(n, S, NH, DK).transpose(0, 2, 1, 3)
                      for t in (q_, k_, v_))
        a = _softf(qh @ kh.transpose(0, 1, 3, 2) * tau)
        qO = 127 * c["qV"] / (1 << c["sav"])
        o = _clipq(a @ vh, qO)                     # (n,NH,S,DK)
        om = o.transpose(0, 2, 1, 3).reshape(n, S, D)         # concat heads
        for h in range(NH):
            ts[f"A{i}_{h}"] = (a[:, h], 127.0)
            ts[f"O{i}_{h}"] = (o[:, h], qO)
        so, sWo = c["so"], qX * (1 << c["so"]) / qO
        t = _clipq(om @ (Wq[f"o{i}"] / sWo) + Wq[f"bo{i}"] / (qO * sWo), qX)
        r = _clipq(x + t, qX)
        vl = _lnq(r, Wq[f"g2_{i}"], Wq[f"b2_{i}"])
        pre = np.clip(vl @ (Wq[f"w1_{i}"] / (1 << c["sg"]))
                      + Wq[f"bw1_{i}"] / (16.0 * (1 << c["sg"])), -8.0, 127.0 / 16)
        h = _clipq(_geluf(pre), 16.0)
        sW2 = qX * (1 << c["sf"]) / 16.0
        f = _clipq(h @ (Wq[f"w2_{i}"] / sW2) + Wq[f"bw2_{i}"] / (16.0 * sW2), qX)
        y = _clipq(r + f, qX)
        ts.update({f"U{i}": (u, 16.0), f"Q{i}": (q_, c["qQ"]), f"K{i}": (k_, c["qK"]),
                   f"V{i}": (v_, c["qV"]), f"O{i}": (om, qO),
                   f"T{i}": (t, qX), f"R{i}": (r, qX), f"VL{i}": (vl, 16.0),
                   f"H{i}": (h, 16.0), f"F{i}": (f, qX), f"Y{i}": (y, qX)})
        x = y
    lnf = _lnq(x, Wq["gf"], Wq["bf"])
    ts["LNF"] = (lnf, 16.0)
    p = _clipq(lnf.mean(axis=1), C["qP"])
    ts["P"] = (p, C["qP"])
    sWh = C["sWh"]
    logits = p @ (Wq["wh"][:, :10] / sWh) + Wq["bh"][:10] / (C["qP"] * sWh)
    ts["logits"] = (logits, C["qP"] * sWh)
    return ts


# ── main: calibrate → fold → run the 100 → health + accuracy ────────────────
ORDER = (["X0"]
         + [name for i in range(NL) for name in
            [f"U{i}", f"Q{i}", f"K{i}", f"V{i}"]
            + [f"SC{i}_{h}" for h in range(NH)]
            + [f"A{i}_{h}" for h in range(NH)]
            + [f"O{i}_{h}" for h in range(NH)]
            + [f"T{i}", f"R{i}", f"VL{i}", f"H{i}", f"F{i}", f"Y{i}"]]
         + ["LNF", "P", "logits"])


def run_oracle(verbose=False):
    """Fold + run the bundled 100. Returns (Wq, C, per-image tensor dicts)."""
    P, X100, y100 = load()
    _, am = vit_float(P, X100)
    Wq, C = fold(P, am)
    outs = [vit_int(Wq, C, X100[n]) for n in range(len(X100))]
    return Wq, C, outs


if __name__ == "__main__":
    P, X100, y100 = load()
    pred_f, am = vit_float(P, X100)
    acc_f = float((pred_f == y100).mean())
    Wq, C = fold(P, am)

    print(f"qX = {C['qX']:.0f}   se = {C['se']}   sp2 = {C['sp2']} (qP = {C['qP']:.2f})")
    for i in range(NL):
        c = C[f"L{i}"]
        print(f"L{i}: sq={c['sq']} sk={c['sk']} sv={c['sv']} sav={c['sav']} "
              f"so={c['so']} sg={c['sg']} sf={c['sf']} sm_scale={c['sm_scale']} "
              f"(τ_eff={c['sm_scale'] * math.log(2) * c['qQ'] * c['qK'] / (1 << 24):.4f}"
              f" ideal {TAU:.4f})")

    outs = [vit_int(Wq, C, X100[n]) for n in range(len(X100))]
    qf = vit_qfloat(Wq, C, X100)

    print(f"\n{'tensor':>7} {'amax':>10} {'sat':>5} {'errLSB':>8}")
    for name in ORDER:
        In = np.array([o[name] for o in outs], dtype=np.float64)
        if name.startswith("SC"):
            print(f"{name:>7} {int(np.abs(In).max()):>10} {'-':>5} {'-':>8}")
            continue
        Ff, q = qf[name]
        if Ff.ndim == 2:
            Ff = Ff[:, None, :]
        if name == "logits":
            In = In[:, :, :10]                     # drop the 6 pad columns
        lim = 127
        err = np.abs(In - Ff * q).max()
        nsat = int((np.abs(In) >= lim).sum()) if name != "logits" else 0
        print(f"{name:>7} {int(np.abs(In).max()):>10} {nsat:>5} {err:>8.2f}")

    pred_i = np.array([o["pred"] for o in outs])
    acc_i = float((pred_i == y100).mean())
    agree = float((pred_i == pred_f).mean())
    print(f"\nfloat exact  : {acc_f:.2f}  ({int(acc_f * 100)}/100)")
    print(f"INT8 oracle  : {acc_i:.2f}  ({int(acc_i * 100)}/100)")
    print(f"int-vs-float agreement: {agree:.2f}")
