"""
cocotb END-TO-END GPT bench — the chip TALKS: incremental KV-cache decode.

The blessed char-GPT (train/gpt, QAT) generates text on the chip via the
SEALED decode graph: each character is ONE M=1 row through both layers —
LN → biased Q/K/V (per-head A-base slices of one 1×64 row) → k/v rows append
to RESIDENT caches → per-head 1×n INT32 scores vs the whole prefix → softmax
→ A·V → biased out-proj → residual → LN → biased FFN (inline GELU) →
residual — then final LN → head (INT32 + bias) → next-char logits. Causality
is structural: the scores GEMM sees exactly the rows that exist.

New chip mechanisms proven beyond the ViT bench: persistent KV caches with
per-step column-append relayouts (host shadows, one 16-word tile rewrite per
head), M=1 GEMMs everywhere, per-head A-operand slicing by base arithmetic
(head h of a 1×64 row = word base+16h — no per-head GEMMs for Q/K/V), and
the FLIPPED A·V: region 0 cannot hold weights + Kᵀ + V (8484 > 8192 words),
so V lives TRANSPOSED in region 1 as the A operand of O_hᵀ = V_hᵀ·A_hᵀ with
the softmax row as a transient n×1 W operand — same dot products, same
requant, bit-identical to the oracle's straight A·V. Region 0 tops out at
7704/8192.

Verification: every tensor bit-exact vs gpt_int_model.decode_step (which
imports the S27/ViT-proven op models), oracle and chip stepping in lockstep;
then a full greedy exchange where the chip's answer must equal int_reply's.
"""

import math

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import First, RisingEdge, Timer

import numpy as np

from gpt_int_model import (CTX, D, DK, FF, NH, NL, NV, NVP, VOCAB,
                           decode_init, decode_step, int_reply, run_oracle)

# ── Geometry / ISA ──────────────────────────────────────────────────────────
TILE, ADDR_W, DIM_W = 16, 13, 12
REGION_W, REGION_A = 0, 1
OP_WEIGHT, OP_BIAS, OP_ACT, OP_RESULT, OP_CONFIG, OP_GO = 1, 2, 3, 4, 5, 6
GO_GEMM, GO_SOFTMAX, GO_ADD, GO_LN = 0, 1, 2, 9
ACT_BYPASS, ACT_GELU = 2, 3

CFG_I32 = (1 << 11) | (1 << 10)                    # raw INT32 out, bias OFF
CFG_I32_B = (1 << 10)                              # raw INT32 out, bias ON


def cfg_byp(sh):                                   # bias OFF
    return (1 << 11) | (ACT_BYPASS << 5) | sh


def cfg_byp_b(sh):                                 # bias ON
    return (ACT_BYPASS << 5) | sh


def cfg_gelu_b(sh):                                # inline GELU, bias ON
    return (ACT_GELU << 5) | sh


NT = CTX // TILE                                   # 7 cache tiles per head

# ── Memory map — region 0 (weights + biases + resident Kᵀ caches) ────────────
WQ_B = [0, 3072]                                   # per layer: 4x4-tile projs
WK_B = [256, 3328]
WV_B = [512, 3584]
WO_B = [768, 4096 - 256]                           # 768, 3840
W1_B = [1024, 4096]                                # 64x256 (1024w)
W2_B = [2048, 5120]                                # 256x64 (1024w)
WH_B = 6144                                        # head 64x48 (192w)
KT_B = [[6336 + (i * NH + h) * NT * TILE for h in range(NH)] for i in range(NL)]
ATW_B = 7232                                       # transient A_hᵀ n×1 (112w)
BQ_B, BK_B = [7344, 7488], [7360, 7504]            # 16w each (N=64)
BV_B, BO_B = [7376, 7520], [7392, 7536]
BW1_B, BW2_B = [7408, 7552], [7472, 7616]          # 64w / 16w
BH_B = 7632                                        # head bias (12w, N=48)
G1_B, G2_B = [7648, 7664], [7656, 7672]            # gamma(4w);beta(4w) pairs
GF_B = 7680
BIAS0 = 7688                                       # zero scratch for bias-off

# ── Memory map — region 1 (M=1 rows + scores + resident Vᵀ caches) ──────────
X_B, U_B = 0, 64                                   # current row · LN1 out
Q_B, K_B, V_B = 128, 192, 256                      # 1x64 projection rows
SC_B = 320                                         # scores 1xn INT32 (448w)
A_B = 768                                          # softmax row 1xn (112w)
OT_B, OM_B = 896, 912                              # O_hᵀ 16x1 · head concat
T_B, R_B, VL_B = 976, 1040, 1104
H_B = 1168                                         # FFN hidden 1x256 (256w)
F_B, Y_B, LNF_B = 1424, 1488, 1552
LG_B = 1616                                        # logits 1x48 INT32 (192w)
VT_B = [[2048 + (i * NH + h) * NT * TILE for h in range(NH)] for i in range(NL)]


# ── Host port / command helpers (S27 patterns verbatim) ─────────────────────
def _idle(dut):
    dut.host_addr.value = 0
    dut.host_en.value = 0
    dut.host_we.value = 0
    dut.host_wdata.value = 0


async def reset(dut):
    dut.rst_n.value = 0
    _idle(dut)
    dut.cmd_valid.value = 0
    dut.cmd_word.value = 0
    for _ in range(3):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


async def mem_write(dut, region, word, data128):
    dut.host_addr.value = (region << ADDR_W) | word
    dut.host_en.value = 1
    dut.host_we.value = 1
    dut.host_wdata.value = data128
    await RisingEdge(dut.clk)
    _idle(dut)


async def mem_read(dut, region, word):
    dut.host_addr.value = (region << ADDR_W) | word
    dut.host_en.value = 1
    dut.host_we.value = 0
    await RisingEdge(dut.clk)
    _idle(dut)
    await RisingEdge(dut.clk)
    val = dut.host_rdata.value
    assert val.is_resolvable, f"host_rdata X (region {region} word {word})"
    return val.to_unsigned()


async def send_cmd(dut, op, base=0, row=0, col=0):
    dut.cmd_word.value = (op << (2 * DIM_W + ADDR_W)) | (base << (2 * DIM_W)) \
                         | (row << DIM_W) | col
    dut.cmd_valid.value = 1
    await RisingEdge(dut.clk)
    dut.cmd_valid.value = 0
    dut.cmd_word.value = 0


async def wait_done(dut, timeout=500_000):
    await First(RisingEdge(dut.done), Timer(timeout * 10, unit="ns"))
    assert int(dut.done.value) == 1, "timeout waiting for done"


def to_signed(v, w):
    return v - (1 << w) if v & (1 << (w - 1)) else v


def pack8(vals):
    return sum((v & 0xFF) << (i * 8) for i, v in enumerate(vals))


def pack32(vals):
    return sum((v & 0xFFFFFFFF) << (i * 32) for i, v in enumerate(vals))


def pad2d(M2, rows, cols, R, Cc):
    return [[(M2[r][c] if (r < rows and c < cols) else 0) for c in range(Cc)]
            for r in range(R)]


# ── Tiled layout packers (ViT bench verbatim) ───────────────────────────────
async def write_A(dut, A, M, K, base):
    nMt, nKt = math.ceil(M / TILE), math.ceil(K / TILE)
    P = pad2d(A, M, K, nMt * TILE, nKt * TILE)
    for i in range(nMt):
        for k in range(nKt):
            for w in range(TILE):
                await mem_write(dut, REGION_A, base + (i * nKt + k) * TILE + w,
                                pack8([P[i * TILE + w][k * TILE + c] for c in range(TILE)]))


async def write_W(dut, W, K, N, base):
    nKt, nNt = math.ceil(K / TILE), math.ceil(N / TILE)
    P = pad2d(W, K, N, nKt * TILE, nNt * TILE)
    for j in range(nNt):
        for k in range(nKt):
            for w in range(TILE):
                await mem_write(dut, REGION_W, base + (j * nKt + k) * TILE + w,
                                pack8([P[k * TILE + w][j * TILE + c] for c in range(TILE)]))


async def write_vec_int8(dut, v, base):
    for j in range(math.ceil(len(v) / TILE)):
        vals = [(v[j * TILE + c] if j * TILE + c < len(v) else 0) for c in range(TILE)]
        await mem_write(dut, REGION_W, base + j, pack8(vals))


async def write_bias(dut, b, N, base):
    """INT32 bias words: word j·4+p = cols j·16+p·4..+3 (4 lanes/word)."""
    nNt = math.ceil(N / TILE)
    bp = [(b[n] if n < len(b) else 0) for n in range(nNt * TILE)]
    for j in range(nNt):
        for p in range(TILE // 4):
            await mem_write(dut, REGION_W, base + j * 4 + p,
                            pack32([bp[j * TILE + p * 4 + q] for q in range(4)]))


async def read_R8(dut, M, N, base, region=REGION_A):
    nMt, nNt = math.ceil(M / TILE), math.ceil(N / TILE)
    out = [[0] * (nNt * TILE) for _ in range(M)]
    for i in range(nMt):
        for j in range(nNt):
            for lr in range(TILE):
                r = i * TILE + lr
                if r >= M:
                    continue
                w = await mem_read(dut, region, base + (i * nNt + j) * TILE + lr)
                for c in range(TILE):
                    out[r][j * TILE + c] = to_signed((w >> (c * 8)) & 0xFF, 8)
    return out


async def read_R32(dut, M, N, base):
    nMt, nNt = math.ceil(M / TILE), math.ceil(N / TILE)
    out = [[0] * (nNt * TILE) for _ in range(M)]
    for i in range(nMt):
        for j in range(nNt):
            for lr in range(TILE):
                r = i * TILE + lr
                if r >= M:
                    continue
                for sub in range(4):
                    w = await mem_read(dut, REGION_A,
                                       base + (i * nNt + j) * 64 + lr * 4 + sub)
                    for p in range(4):
                        out[r][j * TILE + sub * 4 + p] = \
                            to_signed((w >> (p * 32)) & 0xFFFFFFFF, 32)
    return out


# ── Dispatch helpers ────────────────────────────────────────────────────────
async def gemm(dut, w_base, b_base, a_base, r_base, M, K, N, cfg):
    await send_cmd(dut, OP_WEIGHT, base=w_base, col=N)
    await send_cmd(dut, OP_BIAS,   base=b_base)
    await send_cmd(dut, OP_ACT,    base=a_base, row=M, col=K)
    await send_cmd(dut, OP_RESULT, base=r_base)
    await send_cmd(dut, OP_CONFIG, base=cfg)
    await send_cmd(dut, OP_GO,     base=GO_GEMM)
    await wait_done(dut)


async def vpu_op(dut, target, a_base, w_base, r_base, M, N, scale=0):
    await send_cmd(dut, OP_ACT,    base=a_base, row=M, col=N)
    await send_cmd(dut, OP_WEIGHT, base=w_base)
    await send_cmd(dut, OP_RESULT, base=r_base)
    await send_cmd(dut, OP_CONFIG, base=0, row=(scale >> 12), col=(scale & 0xFFF))
    await send_cmd(dut, OP_GO,     base=target)
    await wait_done(dut)


# ── Oracle (folded once, shared by both tests) ──────────────────────────────
_ORACLE = None


def oracle():
    global _ORACLE
    if _ORACLE is None:
        _ORACLE = run_oracle()
    return _ORACLE


async def setup(dut):
    """Reset + upload everything static: weights, biases, γβ, zero scratch."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)
    Wq, C = oracle()
    for i in range(NL):
        await write_W(dut, Wq[f"q{i}"].tolist(), D, D, WQ_B[i])
        await write_W(dut, Wq[f"k{i}"].tolist(), D, D, WK_B[i])
        await write_W(dut, Wq[f"v{i}"].tolist(), D, D, WV_B[i])
        await write_W(dut, Wq[f"o{i}"].tolist(), D, D, WO_B[i])
        await write_W(dut, Wq[f"w1_{i}"].tolist(), D, FF, W1_B[i])
        await write_W(dut, Wq[f"w2_{i}"].tolist(), FF, D, W2_B[i])
        await write_vec_int8(dut, Wq[f"g1_{i}"].tolist(), G1_B[i])
        await write_vec_int8(dut, Wq[f"b1_{i}"].tolist(), G1_B[i] + 4)
        await write_vec_int8(dut, Wq[f"g2_{i}"].tolist(), G2_B[i])
        await write_vec_int8(dut, Wq[f"b2_{i}"].tolist(), G2_B[i] + 4)
        await write_bias(dut, Wq[f"bq{i}"].tolist(), D, BQ_B[i])
        await write_bias(dut, Wq[f"bk{i}"].tolist(), D, BK_B[i])
        await write_bias(dut, Wq[f"bv{i}"].tolist(), D, BV_B[i])
        await write_bias(dut, Wq[f"bo{i}"].tolist(), D, BO_B[i])
        await write_bias(dut, Wq[f"bw1_{i}"].tolist(), FF, BW1_B[i])
        await write_bias(dut, Wq[f"bw2_{i}"].tolist(), D, BW2_B[i])
    await write_W(dut, Wq["wh"].tolist(), D, NVP, WH_B)
    await write_bias(dut, Wq["bh"].tolist(), NVP, BH_B)
    await write_vec_int8(dut, Wq["gf"].tolist(), GF_B)
    await write_vec_int8(dut, Wq["bf"].tolist(), GF_B + 4)
    for w in range(16):
        await mem_write(dut, REGION_W, BIAS0 + w, 0)


def check(tag, got, exp, rows, cols):
    for r in range(rows):
        for c in range(cols):
            assert got[r][c] == exp[r][c], \
                f"{tag}[{r}][{c}]: hw {got[r][c]} != model {exp[r][c]}"


# ── The chip-side decoder ───────────────────────────────────────────────────
async def dec_reset(dut):
    """Fresh exchange: zero both resident caches + host shadows, n=0."""
    for i in range(NL):
        for h in range(NH):
            for w in range(NT * TILE):
                await mem_write(dut, REGION_W, KT_B[i][h] + w, 0)
                await mem_write(dut, REGION_A, VT_B[i][h] + w, 0)
    z = lambda: [[[0] * CTX for _ in range(DK)] for _ in range(NL * NH)]
    return dict(n=0, kt=z(), vt=z())


async def kv_append(dut, st, i, K8, V8):
    """Append k/v row at position n: column n of Kᵀ_h (region 0 W layout) and
    of Vᵀ_h (region 1 A layout) — one 16-word tile rewrite per head from the
    host shadow (zero-padded beyond n by construction)."""
    p = st["n"]
    jt = p // TILE
    for h in range(NH):
        kt, vt = st["kt"][i * NH + h], st["vt"][i * NH + h]
        for r in range(DK):
            kt[r][p] = K8[h * DK + r]
            vt[r][p] = V8[h * DK + r]
        for w in range(TILE):
            await mem_write(dut, REGION_W, KT_B[i][h] + jt * TILE + w,
                            pack8([kt[w][jt * TILE + c] for c in range(TILE)]))
            await mem_write(dut, REGION_A, VT_B[i][h] + jt * TILE + w,
                            pack8([vt[w][jt * TILE + c] for c in range(TILE)]))


async def dec_step(dut, Wq, C, st, cid, ref=None):
    """One char through the chip. ref = oracle decode_step tensors → assert
    everything bit-exact along the way. Returns the 1×48 INT32 logits row."""
    p = st["n"]
    n = p + 1

    async def grab8(tag, base, cols):
        if ref is None:
            return
        hw = await read_R8(dut, 1, cols, base)
        check(tag, [hw[0][:cols]], [ref[tag][0][:cols]], 1, cols)

    await write_A(dut, [Wq["XE"][cid][p].tolist()], 1, D, X_B)
    await grab8("X0", X_B, D)
    x_base = X_B
    for i in range(NL):
        c = C[f"L{i}"]
        await vpu_op(dut, GO_LN, x_base, G1_B[i], U_B, 1, D, scale=8)
        await gemm(dut, WQ_B[i], BQ_B[i], U_B, Q_B, 1, D, D, cfg_byp_b(c["sq"]))
        await gemm(dut, WK_B[i], BK_B[i], U_B, K_B, 1, D, D, cfg_byp_b(c["sk"]))
        await gemm(dut, WV_B[i], BV_B[i], U_B, V_B, 1, D, D, cfg_byp_b(c["sv"]))
        K8 = (await read_R8(dut, 1, D, K_B))[0][:D]            # host: cache append
        V8 = (await read_R8(dut, 1, D, V_B))[0][:D]
        await kv_append(dut, st, i, K8, V8)
        await grab8(f"U{i}", U_B, D)
        await grab8(f"Q{i}", Q_B, D)
        if ref is not None:
            check(f"K{i}", [K8], ref[f"K{i}"], 1, D)
            check(f"V{i}", [V8], ref[f"V{i}"], 1, D)
        Om = []
        for h in range(NH):                        # per-head attention core
            await gemm(dut, KT_B[i][h], BIAS0, Q_B + h * TILE, SC_B,
                       1, DK, n, CFG_I32)
            await vpu_op(dut, GO_SOFTMAX, SC_B, 0, A_B, 1, n, scale=c["sm_scale"])
            A8 = (await read_R8(dut, 1, n, A_B))[0][:n]        # host: A_hᵀ as W op
            await write_W(dut, [[a] for a in A8], n, 1, ATW_B)
            await gemm(dut, ATW_B, BIAS0, VT_B[i][h], OT_B,    # O_hᵀ = V_hᵀ·A_hᵀ
                       DK, n, 1, cfg_byp(c["sav"]))
            OT = await read_R8(dut, DK, 1, OT_B)               # host: head concat
            Oh = [OT[r][0] for r in range(DK)]
            if ref is not None:
                SCh = await read_R32(dut, 1, n, SC_B)
                check(f"SC{i}_{h}", [SCh[0][:n]], ref[f"SC{i}_{h}"], 1, n)
                check(f"A{i}_{h}", [A8], ref[f"A{i}_{h}"], 1, n)
                check(f"O{i}_{h}", [Oh], ref[f"O{i}_{h}"], 1, DK)
            Om += Oh
        await write_A(dut, [Om], 1, D, OM_B)
        await gemm(dut, WO_B[i], BO_B[i], OM_B, T_B, 1, D, D, cfg_byp_b(c["so"]))
        await vpu_op(dut, GO_ADD, x_base, T_B, R_B, 1, D)
        await vpu_op(dut, GO_LN, R_B, G2_B[i], VL_B, 1, D, scale=8)
        await gemm(dut, W1_B[i], BW1_B[i], VL_B, H_B, 1, D, FF, cfg_gelu_b(c["sg"]))
        await gemm(dut, W2_B[i], BW2_B[i], H_B, F_B, 1, FF, D, cfg_byp_b(c["sf"]))
        await vpu_op(dut, GO_ADD, R_B, F_B, Y_B, 1, D)
        await grab8(f"T{i}", T_B, D)
        await grab8(f"R{i}", R_B, D)
        await grab8(f"VL{i}", VL_B, D)
        await grab8(f"H{i}", H_B, FF)
        await grab8(f"F{i}", F_B, D)
        await grab8(f"Y{i}", Y_B, D)
        x_base = Y_B
    await vpu_op(dut, GO_LN, Y_B, GF_B, LNF_B, 1, D, scale=8)
    await gemm(dut, WH_B, BH_B, LNF_B, LG_B, 1, D, NVP, CFG_I32_B)
    await grab8("LNF", LNF_B, D)
    lg = (await read_R32(dut, 1, NVP, LG_B))[0][:NVP]
    if ref is not None:
        check("logits", [lg], ref["logits"], 1, NVP)
    st["n"] = n
    return lg


STOI = {c: i for i, c in enumerate(VOCAB)}


# ── tests ───────────────────────────────────────────────────────────────────
@cocotb.test()
async def test_gpt_decode_tensors(dut):
    """A prompt + 4 generated chars, oracle and chip in lockstep, every
    intermediate bit-exact at every position — crosses the first cache-tile
    boundary (n > 16), so the column-append relayouts and the growing-n
    scores/softmax/flipped-A·V are all exercised."""
    await setup(dut)
    log = dut._log.info
    Wq, C = oracle()
    st_hw = await dec_reset(dut)
    st_or = decode_init()
    ids = [STOI[ch] for ch in "u: how are you?\nt: "]          # 19 chars
    out = ""
    for step in range(len(ids) + 4):
        cid = ids[step] if step < len(ids) else cid_next
        ref = decode_step(Wq, C, st_or, cid)
        lg = await dec_step(dut, Wq, C, st_hw, cid, ref=ref)
        cid_next = int(np.argmax(lg[:NV]))
        if step >= len(ids) - 1:
            out += VOCAB[cid_next]
    log(f"all tensors bit-exact over {len(ids) + 4} positions "
        f"(n reaches {st_hw['n']}); chip continues: {out!r}")


@cocotb.test()
async def test_gpt_generate(dut):
    """The payoff: a full greedy exchange on the chip. Logits bit-exact vs the
    oracle at every step; the chip's answer must equal int_reply's."""
    await setup(dut)
    log = dut._log.info
    Wq, C = oracle()
    q = "who are you?"
    expect = int_reply(Wq, C, q)
    st_hw = await dec_reset(dut)
    st_or = decode_init()
    lg = None
    for ch in f"u: {q}\nt: ":
        ref = decode_step(Wq, C, st_or, STOI[ch])
        lg = await dec_step(dut, Wq, C, st_hw, STOI[ch])
        assert lg == ref["logits"][0], "prompt logits mismatch"
    out = ""
    for _ in range(60):
        cid = int(np.argmax(lg[:NV]))
        if VOCAB[cid] == "\n" or st_hw["n"] >= CTX:
            break
        out += VOCAB[cid]
        ref = decode_step(Wq, C, st_or, cid)
        lg = await dec_step(dut, Wq, C, st_hw, cid)
        assert lg == ref["logits"][0], f"logits mismatch at {out!r}"
    assert out == expect, f"chip {out!r} != oracle {expect!r}"
    log(f'chip answers "u: {q}" → "{out}"  (== oracle, logits bit-exact '
        f"every step, {st_hw['n']} positions)")
