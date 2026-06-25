"""
gpt_int_model.py — exact-arithmetic reference (oracle) for the LINK TPU GPT.

The trained char-GPT (train/gpt) folded into the chip's integer arithmetic,
executing the SEALED decode graph: incremental KV-cached generation. Each
step runs ONE new row through both layers (M=1 GEMMs); k/v rows append to
per-layer caches that stay resident, and the new row's scores GEMM sees
exactly the rows that exist — causality is structural, no masks anywhere.
Op models are IMPORTED single-source: attn_int_model (LUT softmax/layernorm,
sat8 add, transpose) and vit_int_model (bias-GEMM per bias_add.sv, self-
tested at import).

Scale algebra (the S27/ViT discipline — every tensor carries an exact q):
  qU = 16 fixed (LN LUT), A q=127 (softmax), GELU in/out q=16 (ROM Q4)
  residual stream at ONE scale qX: Wo and W2 scales CHOSEN so qT == qF == qX
  exactly; sW1 = 2^sg (GELU input lands on the Q4 grid); Wq/Wk/Wv free
  (sm_scale absorbs qQ·qK into τ_eff = 1/√d_k = 1/4); head scale-free.
  Host side folds ONCE: XE[c][p] = rint((embed[c]+pos[p])·qX) — the whole
  host embedding job at inference is this int8 table lookup.

Calibration: the bundled held-out val stream inside trained_gpt.npz (float
batched-causal forward supplies the activation stream), percentile-99.9 amax
— matching the trainer's QAT calibration exactly (outlier-driven full-amax
costs a whole bit of stream resolution: qX 32 → 16). Verification here:
teacher-forced next-char accuracy (int vs float vs truth) on the full val
stream + greedy generation int-vs-float side by side. The bench rung imports
this module and asserts the chip equals decode_step() bit-for-bit.

Run standalone for the fold summary + health + accuracy:
    python3 gpt_int_model.py
"""

import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "vit"))
sys.path.insert(0, str(HERE.parent / "attention"))
from attn_int_model import add8, ln_row, softmax_row, transpose
from vit_int_model import gemm8b, gemm_i32b

NPZ = HERE / "trained_gpt.npz"
_d = np.load(NPZ)
META = json.loads(str(_d["meta"]))
VOCAB = str(_d["vocab"])
D, FF, NL, NH = META["d"], META["d_ff"], META["n_layers"], META["n_heads"]
CTX, NV = META["ctx"], META["n_vocab"]
DK = D // NH
TAU = 1.0 / math.sqrt(DK)
NVP = 48                                           # head cols padded to 3 tiles
PCT = 99.9                                         # calibration percentile


def load():
    P = {k: _d[k].astype(np.float64) for k in _d.files
         if k not in ("meta", "vocab", "val_ids")}
    return P, _d["val_ids"].astype(np.int64)


# ── float forward (exact torch mirror, batched causal) — calib + reference ──
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


def _am(t):
    return float(np.percentile(np.abs(t), PCT))


def gpt_float(P, ids):
    """ids: (B,S) int. Causal batched forward. Returns (logits (B,S,NV),
    calibration dict: percentile-PCT amax per tensor)."""
    am = {}
    B, S = ids.shape
    x = P["embed_weight"][ids] + P["pos"][:S]
    am["X0"] = _am(x)
    mask = np.triu(np.full((S, S), -np.inf), 1)
    for i in range(NL):
        L = f"layers_{i}_"
        u = _lnf_exact(x, P[L + "ln1_weight"], P[L + "ln1_bias"])
        q = u @ P[L + "wq_weight"].T + P[L + "wq_bias"]
        k = u @ P[L + "wk_weight"].T + P[L + "wk_bias"]
        v = u @ P[L + "wv_weight"].T + P[L + "wv_bias"]
        am[f"Q{i}"], am[f"K{i}"], am[f"V{i}"] = (_am(t) for t in (q, k, v))
        qh, kh, vh = (t.reshape(B, S, NH, DK).transpose(0, 2, 1, 3)
                      for t in (q, k, v))
        a = _softf(qh @ kh.transpose(0, 1, 3, 2) * TAU + mask)
        o = (a @ vh).transpose(0, 2, 1, 3).reshape(B, S, D)   # concat heads
        am[f"O{i}"] = _am(o)
        t = o @ P[L + "wo_weight"].T + P[L + "wo_bias"]
        am[f"T{i}"] = _am(t)
        x = x + t
        am[f"R{i}"] = _am(x)
        f = _geluf(_lnf_exact(x, P[L + "ln2_weight"], P[L + "ln2_bias"])
                   @ P[L + "w1_weight"].T + P[L + "w1_bias"]) \
            @ P[L + "w2_weight"].T + P[L + "w2_bias"]
        am[f"F{i}"] = _am(f)
        x = x + f
        am[f"Y{i}"] = _am(x)
    lnf = _lnf_exact(x, P["lnf_weight"], P["lnf_bias"])
    return lnf @ P["head_weight"].T + P["head_bias"], am


def float_reply(P, question, max_new=60):
    """Greedy float generation (full-prefix causal forward each step)."""
    stoi = {c: i for i, c in enumerate(VOCAB)}
    ids = [stoi[c] for c in f"u: {question}\nt: "]
    out = ""
    for _ in range(max_new):
        logits, _ = gpt_float(P, np.array([ids[-CTX:]]))
        c = int(logits[0, -1].argmax())
        if VOCAB[c] == "\n":
            break
        out += VOCAB[c]
        ids.append(c)
    return out


# ── the fold: float weights + calib amax → int weights + sealed constants ───
def _q8(Wf, s):
    W = np.rint(Wf * s).astype(np.int64)
    assert np.abs(W).max() <= 127, "int8 weight overflow"
    return W


def fold(P, am):
    """Returns (Wq dict of int arrays, C dict of constants)."""
    C, Wq = {}, {}
    stream = max(am[k] for k in am if k[0] in "XRY")
    C["qX"] = float(1 << int(math.floor(math.log2(120.0 / stream))))
    qX = C["qX"]

    def shift_for(q_in, sW, amax):                 # smallest sh: q_out·amax ≤ 120
        return max(0, math.ceil(math.log2(q_in * sW * amax / 120.0)))

    # host embed table: XE[c][p] = sat8(rint((embed[c]+pos[p])·qX)) — the sat8
    # clip IS the quantizer here (QAT trained with this exact clamp)
    XE = np.rint((P["embed_weight"][:, None, :] + P["pos"][None, :, :]) * qX)
    Wq["XE"] = np.clip(XE, -128, 127).astype(np.int64)     # (NV, CTX, D)

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
    sWh = 127.0 / np.abs(P["head_weight"]).max()
    C["sWh"] = sWh
    Wq["wh"] = np.pad(_q8(P["head_weight"].T, sWh), ((0, 0), (0, NVP - NV)))
    Wq["bh"] = np.pad(np.rint(P["head_bias"] * 16 * sWh).astype(np.int64),
                      (0, NVP - NV))
    for i in range(NL):
        assert all(0 <= C[f"L{i}"][s] <= 31
                   for s in ("sq", "sk", "sv", "sav", "so", "sg", "sf"))
    assert all(np.abs(Wq[k]).max() < (1 << 31) for k in Wq)
    return Wq, C


# ── integer decode (the chip, exactly) — one char in, next-char logits out ──
def decode_init():
    """Fresh exchange: empty per-layer K/V caches (the chip's resident state)."""
    return dict(n=0, K=[[] for _ in range(NL)], V=[[] for _ in range(NL)])


def decode_step(Wq, C, st, cid):
    """Feed char id `cid` at position st['n']. M=1 row through both layers;
    k/v append to the caches. Returns every named int tensor (bench contract),
    incl. 'logits' = next-char head row (first NV of NVP columns count)."""
    p = st["n"]
    assert p < CTX, "context overflow — exchanges start fresh"
    X = [Wq["XE"][cid][p].tolist()]
    ts = {"X0": X}
    for i in range(NL):
        c = C[f"L{i}"]
        U = [ln_row(X[0], Wq[f"g1_{i}"].tolist(), Wq[f"b1_{i}"].tolist(), 8)]
        Q = gemm8b(U, Wq[f"q{i}"], Wq[f"bq{i}"], c["sq"])
        K = gemm8b(U, Wq[f"k{i}"], Wq[f"bk{i}"], c["sk"])
        V = gemm8b(U, Wq[f"v{i}"], Wq[f"bv{i}"], c["sv"])
        st["K"][i].append(K[0])                    # cache append: k_p, v_p
        st["V"][i].append(V[0])
        heads = []                                 # per-head attention (d_k cols)
        for h in range(NH):
            sl = slice(h * DK, (h + 1) * DK)
            Qh = [Q[0][sl]]
            KhT = transpose([r[sl] for r in st["K"][i]])
            Vh = [r[sl] for r in st["V"][i]]
            SCh = gemm_i32b(Qh, KhT, [0] * (p + 1))
            Ah = [softmax_row(SCh[0], c["sm_scale"])]
            Oh = gemm8b(Ah, Vh, [0] * DK, c["sav"])
            ts.update({f"SC{i}_{h}": SCh, f"A{i}_{h}": Ah, f"O{i}_{h}": Oh})
            heads.append(Oh[0])
        O = [[x for hd in heads for x in hd]]      # concat heads
        T = gemm8b(O, Wq[f"o{i}"], Wq[f"bo{i}"], c["so"])
        R = add8(X, T)
        VL = [ln_row(R[0], Wq[f"g2_{i}"].tolist(), Wq[f"b2_{i}"].tolist(), 8)]
        H = gemm8b(VL, Wq[f"w1_{i}"], Wq[f"bw1_{i}"], c["sg"], gelu=True)
        F = gemm8b(H, Wq[f"w2_{i}"], Wq[f"bw2_{i}"], c["sf"])
        X = add8(R, F)
        ts.update({f"U{i}": U, f"Q{i}": Q, f"K{i}": K, f"V{i}": V, f"O{i}": O,
                   f"T{i}": T, f"R{i}": R, f"VL{i}": VL,
                   f"H{i}": H, f"F{i}": F, f"Y{i}": X})
    LNF = [ln_row(X[0], Wq["gf"].tolist(), Wq["bf"].tolist(), 8)]
    logits = gemm_i32b(LNF, Wq["wh"], Wq["bh"])
    ts.update(LNF=LNF, logits=logits)
    st["n"] += 1
    return ts


def int_reply(Wq, C, question, max_new=60):
    """Greedy integer generation — the sealed chip graph end to end."""
    stoi = {c: i for i, c in enumerate(VOCAB)}
    st = decode_init()
    for ch in f"u: {question}\nt: ":
        logits = decode_step(Wq, C, st, stoi[ch])["logits"]
    out = ""
    for _ in range(max_new):
        c = int(np.argmax(logits[0][:NV]))
        if VOCAB[c] == "\n" or st["n"] >= CTX:
            break
        out += VOCAB[c]
        logits = decode_step(Wq, C, st, c)["logits"]
    return out


def run_oracle():
    """Fold from the bundled calibration. Returns (Wq, C) for the bench."""
    P, val = load()
    starts = np.arange(0, len(val) - CTX - 1, CTX)
    win = np.stack([val[j:j + CTX] for j in starts])
    _, am = gpt_float(P, win)
    return fold(P, am)


def export_chip(path):
    """Bundle everything the HOST script needs (fpga_scripts/gpt_chip.npz):
    folded int weights/biases/γβ + the XE table + sealed constants + a proof
    pack (two greedy exchanges: fed char ids + per-step logits, bit-exact) —
    so run_gpt.py is self-contained with zero model code."""
    Wq, C = run_oracle()
    arrays = {}
    for k, v in Wq.items():
        arrays[k] = v.astype(np.int32 if k[0] == "b" and k not in
                             ("b1_0", "b1_1", "b2_0", "b2_1", "bf") else np.int8)
    stoi = {c: i for i, c in enumerate(VOCAB)}
    proofs = []
    for pi, q in enumerate(["who are you?", "what is 6 times 7?"]):
        st = decode_init()
        ids, logits = [], []
        for ch in f"u: {q}\nt: ":
            ids.append(stoi[ch])
            logits.append(decode_step(Wq, C, st, ids[-1])["logits"][0])
        out = ""
        for _ in range(60):
            c = int(np.argmax(logits[-1][:NV]))
            if VOCAB[c] == "\n" or st["n"] >= CTX:
                break
            out += VOCAB[c]
            ids.append(c)
            logits.append(decode_step(Wq, C, st, c)["logits"][0])
        arrays[f"proof_ids_{pi}"] = np.array(ids, dtype=np.uint8)
        arrays[f"proof_logits_{pi}"] = np.array(logits, dtype=np.int64)
        proofs.append(dict(q=q, a=out))
    meta = dict(d=D, d_ff=FF, n_layers=NL, n_heads=NH, d_k=DK, ctx=CTX,
                n_vocab=NV, n_vocab_pad=NVP, qX=C["qX"],
                q_logits=16.0 * C["sWh"],
                layers=[{k: C[f"L{i}"][k] for k in
                         ("sq", "sk", "sv", "sav", "so", "sg", "sf", "sm_scale")}
                        for i in range(NL)],
                proofs=proofs)
    np.savez(path, meta=json.dumps(meta), vocab=VOCAB, **arrays)
    print(f"exported {path}: chip-ready fold + proof pack "
          f"({', '.join(repr(p['a']) for p in proofs)})")


# ── main: calibrate → fold → teacher-forced val + generation match ──────────
if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--export":
        export_chip(sys.argv[2])
        sys.exit(0)
    P, val = load()
    starts = np.arange(0, len(val) - CTX - 1, CTX)
    win = np.stack([val[j:j + CTX] for j in starts])
    tgt = np.stack([val[j + 1:j + 1 + CTX] for j in starts])
    logits_f, am = gpt_float(P, win)
    Wq, C = fold(P, am)

    print(f"qX = {C['qX']:.0f}   vocab {NV} (head padded {NVP})   ctx {CTX}")
    for i in range(NL):
        c = C[f"L{i}"]
        print(f"L{i}: sq={c['sq']} sk={c['sk']} sv={c['sv']} sav={c['sav']} "
              f"so={c['so']} sg={c['sg']} sf={c['sf']} sm_scale={c['sm_scale']} "
              f"(τ_eff={c['sm_scale'] * math.log(2) * c['qQ'] * c['qK'] / (1 << 24):.4f}"
              f" ideal {TAU:.4f})")

    # teacher-forced: incremental int decode over every val window
    pf = logits_f.argmax(-1)
    pi = np.zeros_like(pf)
    amax = {}
    nsat = {}
    for w in range(len(win)):
        st = decode_init()
        for t in range(CTX):
            ts = decode_step(Wq, C, st, int(win[w, t]))
            pi[w, t] = int(np.argmax(ts["logits"][0][:NV]))
            if w == 0:                             # health stats, first window
                for k, v in ts.items():
                    a = np.abs(np.asarray(v))
                    amax[k] = max(amax.get(k, 0), int(a.max()))
                    if not k.startswith(("SC", "logits")):
                        nsat[k] = nsat.get(k, 0) + int((a >= 127).sum())
    acc_f = float((pf == tgt).mean())
    acc_i = float((pi == tgt).mean())
    agree = float((pi == pf).mean())
    print(f"\nteacher-forced next-char acc ({tgt.size} positions):")
    print(f"  float exact : {acc_f:.4f}")
    print(f"  INT8 oracle : {acc_i:.4f}")
    print(f"  int-vs-float agreement: {agree:.4f}")

    print(f"\n{'tensor':>7} {'amax':>9} {'sat':>5}   (window 0, all {CTX} steps)")
    for k in sorted(amax):
        s = "-" if k.startswith(("SC", "logits")) else str(nsat[k])
        print(f"{k:>7} {amax[k]:>9} {s:>5}")

    print("\ngeneration (greedy): INT8 == float ?")
    match = 0
    prompts = ["who are you?", "hello", "what is 6 times 7?", "do you like cats?",
               "how fast are you?", "tell me your name", "what can you do?",
               "where do you live?", "thanks", "can you learn?"]
    for q in prompts:
        ri = int_reply(Wq, C, q)
        rf = float_reply(P, q)
        ok = ri == rf
        match += ok
        print(f"  u: {q}\n    int  : {ri}" + ("" if ok else f"\n    float: {rf}"))
    print(f"\ngeneration match: {match}/{len(prompts)}")
