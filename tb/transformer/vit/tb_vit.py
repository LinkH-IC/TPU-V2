"""
cocotb END-TO-END MNIST ViT bench — the full trained transformer on top.

The whole network from train/mnist_vit (97.81% float, 98/100 INT8 oracle) runs
on the chip, real weights, real digits: patch embed (GEMM + bias) → +pos (EW
add) → 2 × pre-LN encoder layer (LN → biased Q/K/V proj → INT32 scores →
softmax → A·V → biased out-proj → residual → LN → biased FFN with inline GELU
→ residual) → final LN → mean-pool (ones-GEMM) → head (GEMM, INT32 out + bias
— first use of bias_add on the raw-INT32 drain).

Extends the S27 attention bench with: bias-enabled GEMMs (cfg bit 11 clear,
OP_BIAS → real INT32 vectors), full write_W relayouts for the as-W-operand
hops (multi-tile — S27's verbatim-copy shortcut was single-tile only), and
MULTI-HEAD attention (NH from the npz meta): per-head Q/K/V are N=d_k GEMMs
whose weight base points at column tile h of the SAME resident matrix (bias
base at word tile h·4) — no weight duplication; each head gets its own
scores/softmax/A·V on shared scratch; the head concat is a host merge
(read O_h, write the 49×32 A-layout tensor) feeding one out-projection.

Verification: every tensor bit-exact vs vit_int_model.py (which imports the
S27-proven op models); the fold (weights + sealed shift constants) comes from
the same module, so bench and oracle share one arithmetic source.
"""

import math

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import First, RisingEdge, Timer

import numpy as np

from vit_int_model import (DK, NH, NL, S, fold, load, vit_float, vit_int,
                           patchify)
from attn_int_model import transpose

# ── Geometry / ISA ──────────────────────────────────────────────────────────
TILE, ADDR_W, DIM_W = 16, 13, 12
REGION_W, REGION_A = 0, 1
OP_WEIGHT, OP_BIAS, OP_ACT, OP_RESULT, OP_CONFIG, OP_GO = 1, 2, 3, 4, 5, 6
GO_GEMM, GO_SOFTMAX, GO_ADD, GO_LN = 0, 1, 2, 9
ACT_BYPASS, ACT_GELU = 2, 3
D, FF = 32, 128

CFG_I32 = (1 << 11) | (1 << 10)                    # raw INT32 out, bias OFF


def cfg_byp(sh):                                   # bias OFF
    return (1 << 11) | (ACT_BYPASS << 5) | sh


def cfg_byp_b(sh):                                 # bias ON
    return (ACT_BYPASS << 5) | sh


def cfg_gelu_b(sh):                                # inline GELU, bias ON
    return (ACT_GELU << 5) | sh


CFG_I32_B = (1 << 10)                              # raw INT32 out, bias ON


# ── Memory map — region 0 (weights + biases + host-shaped operands) ─────────
WE_B = 0                                           # embed 16x32 (32w)
WQ_B = [32, 800]                                   # per layer: 4x 64w projections
WK_B = [96, 864]
WV_B = [160, 928]
WO_B = [224, 992]
W1_B = [288, 1056]                                 # 32x128 (256w)
W2_B = [544, 1312]                                 # 128x32 (256w)
KT_B, VW_B, LNW_B = 1568, 1696, 1824               # per-image relayouts (128w each)
WH_B = 1952                                        # head 32x16 (32w)
G1_B, G2_B = [1984, 1992], [1988, 1996]            # gamma(2w);beta(2w) pairs
GF_B = 2000
BE_B = 2004                                        # embed bias (8w, N=32)
BQ_B, BK_B = [2012, 2084], [2020, 2092]            # per layer (8w each)
BV_B, BO_B = [2028, 2100], [2036, 2108]
BW1_B, BW2_B = [2044, 2116], [2076, 2148]          # FFN biases (32w / 8w)
BH_B = 2156                                        # head bias (4w, N=16)
BIAS0 = 2160                                       # zero scratch for bias-off (16w)

# ── Memory map — region 1 (activations, reused across images) ───────────────
XP_B = 0                                           # patches 49x16 (64w)
E_B, POS_B, X0_B = 64, 192, 320                    # embed out · pos · tokens (128w)
U_B = 448
QH_B = [576 + h * 4 * DK for h in range(NH)]       # per-head Q/K/V (4·DK w each)
KH_B = [704 + h * 4 * DK for h in range(NH)]
VH_B = [832 + h * 4 * DK for h in range(NH)]
SC_B = 960                                         # scores 49x49 INT32 (1024w, per-head scratch)
A_B = 1984                                         # softmax 49x49 (256w, per-head scratch)
OH_B = [2240 + h * 4 * DK for h in range(NH)]      # per-head A·V outputs
T_B, R_B, VL_B = 2368, 2496, 2624
H_B = 2752                                         # FFN hidden 49x128 (512w)
F_B, Y_B, LNF_B = 3264, 3392, 3520
P_B, LG_B, ONES_B = 3648, 3680, 3744               # pool · logits INT32 · ones
OM_B = 3808                                        # host-merged head concat (128w)


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


# ── Tiled layout packers ────────────────────────────────────────────────────
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
        P, X100, y100 = load()
        _, am = vit_float(P, X100)
        Wq, C = fold(P, am)
        _ORACLE = (Wq, C, X100, y100)
    return _ORACLE


async def setup(dut):
    """Reset + upload everything static: all weights, biases, γβ, pos, ones."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)
    Wq, C, _, _ = oracle()
    await write_W(dut, Wq["We"].tolist(), 16, D, WE_B)
    await write_bias(dut, Wq["be"].tolist(), D, BE_B)
    for i in range(NL):
        await write_W(dut, Wq[f"q{i}"].tolist(), D, D, WQ_B[i])
        await write_W(dut, Wq[f"k{i}"].tolist(), D, D, WK_B[i])
        await write_W(dut, Wq[f"v{i}"].tolist(), D, D, WV_B[i])
        await write_W(dut, Wq[f"o{i}"].tolist(), D, D, WO_B[i])
        await write_W(dut, Wq[f"w1_{i}"].tolist(), D, FF, W1_B[i])
        await write_W(dut, Wq[f"w2_{i}"].tolist(), FF, D, W2_B[i])
        await write_vec_int8(dut, Wq[f"g1_{i}"].tolist(), G1_B[i])
        await write_vec_int8(dut, Wq[f"b1_{i}"].tolist(), G1_B[i] + 2)
        await write_vec_int8(dut, Wq[f"g2_{i}"].tolist(), G2_B[i])
        await write_vec_int8(dut, Wq[f"b2_{i}"].tolist(), G2_B[i] + 2)
        await write_bias(dut, Wq[f"bq{i}"].tolist(), D, BQ_B[i])
        await write_bias(dut, Wq[f"bk{i}"].tolist(), D, BK_B[i])
        await write_bias(dut, Wq[f"bv{i}"].tolist(), D, BV_B[i])
        await write_bias(dut, Wq[f"bo{i}"].tolist(), D, BO_B[i])
        await write_bias(dut, Wq[f"bw1_{i}"].tolist(), FF, BW1_B[i])
        await write_bias(dut, Wq[f"bw2_{i}"].tolist(), D, BW2_B[i])
    await write_W(dut, Wq["wh"].tolist(), D, TILE, WH_B)
    await write_bias(dut, Wq["bh"].tolist(), TILE, BH_B)
    await write_vec_int8(dut, Wq["gf"].tolist(), GF_B)
    await write_vec_int8(dut, Wq["bf"].tolist(), GF_B + 2)
    for w in range(16):
        await mem_write(dut, REGION_W, BIAS0 + w, 0)
    await write_A(dut, Wq["pos"].tolist(), S, D, POS_B)
    await write_A(dut, [[1] * S], 1, S, ONES_B)


def check(tag, got, exp, rows, cols):
    for r in range(rows):
        for c in range(cols):
            assert got[r][c] == exp[r][c], \
                f"{tag}[{r}][{c}]: hw {got[r][c]} != model {exp[r][c]}"


async def vit_image(dut, x784, ref=None, log=None):
    """Run one digit through the chip. ref = oracle tensors → assert everything
    bit-exact along the way; otherwise only the logits are read back."""
    Wq, C, _, _ = oracle()

    async def grab8(tag, base, cols):
        if ref is None:
            return None
        hw = await read_R8(dut, S, cols, base)
        check(tag, hw, ref[tag], S, cols)
        return hw

    patches = patchify(np.asarray(x784, np.int64)[None, :])[0].tolist()
    await write_A(dut, patches, S, 16, XP_B)
    await gemm(dut, WE_B, BE_B, XP_B, E_B, S, 16, D, cfg_byp_b(C["se"]))
    await vpu_op(dut, GO_ADD, E_B, POS_B, X0_B, S, D)
    await grab8("X0", X0_B, D)

    x_base = X0_B
    for i in range(NL):
        c = C[f"L{i}"]
        await vpu_op(dut, GO_LN, x_base, G1_B[i], U_B, S, D, scale=8)
        await grab8(f"U{i}", U_B, D)
        O8s = []
        for h in range(NH):                        # per-head attention core
            wof = h * (D // TILE) * TILE           # W column-tile block offset
            bof = h * 4 * (DK // TILE)             # bias words for tile h
            await gemm(dut, WQ_B[i] + wof, BQ_B[i] + bof, U_B, QH_B[h],
                       S, D, DK, cfg_byp_b(c["sq"]))
            await gemm(dut, WK_B[i] + wof, BK_B[i] + bof, U_B, KH_B[h],
                       S, D, DK, cfg_byp_b(c["sk"]))
            await gemm(dut, WV_B[i] + wof, BV_B[i] + bof, U_B, VH_B[h],
                       S, D, DK, cfg_byp_b(c["sv"]))
            K8 = await read_R8(dut, S, DK, KH_B[h])           # host: K_h^T
            V8 = await read_R8(dut, S, DK, VH_B[h])           # host: V_h as W op
            if ref is not None:
                sl = slice(h * DK, (h + 1) * DK)
                Q8 = await read_R8(dut, S, DK, QH_B[h])
                check(f"Q{i}h{h}", Q8, [r[sl] for r in ref[f"Q{i}"]], S, DK)
                check(f"K{i}h{h}", K8, [r[sl] for r in ref[f"K{i}"]], S, DK)
                check(f"V{i}h{h}", V8, [r[sl] for r in ref[f"V{i}"]], S, DK)
            await write_W(dut, transpose(K8), DK, S, KT_B)
            await write_W(dut, V8, S, DK, VW_B)
            await gemm(dut, KT_B, BIAS0, QH_B[h], SC_B, S, DK, S, CFG_I32)
            await vpu_op(dut, GO_SOFTMAX, SC_B, 0, A_B, S, S, scale=c["sm_scale"])
            await gemm(dut, VW_B, BIAS0, A_B, OH_B[h], S, S, DK, cfg_byp(c["sav"]))
            if ref is not None:
                SCh = await read_R32(dut, S, S, SC_B)
                check(f"SC{i}_{h}", SCh, ref[f"SC{i}_{h}"], S, S)
                await grab8(f"A{i}_{h}", A_B, S)
            O8 = await read_R8(dut, S, DK, OH_B[h])           # host: concat input
            if ref is not None:
                check(f"O{i}_{h}", O8, ref[f"O{i}_{h}"], S, DK)
            O8s.append(O8)
        Om = [sum((O8s[h][r] for h in range(NH)), []) for r in range(S)]
        await write_A(dut, Om, S, D, OM_B)                    # host: head concat
        await gemm(dut, WO_B[i], BO_B[i], OM_B, T_B, S, D, D, cfg_byp_b(c["so"]))
        await vpu_op(dut, GO_ADD, x_base, T_B, R_B, S, D)
        await vpu_op(dut, GO_LN, R_B, G2_B[i], VL_B, S, D, scale=8)
        await gemm(dut, W1_B[i], BW1_B[i], VL_B, H_B, S, D, FF, cfg_gelu_b(c["sg"]))
        await gemm(dut, W2_B[i], BW2_B[i], H_B, F_B, S, FF, D, cfg_byp_b(c["sf"]))
        await vpu_op(dut, GO_ADD, R_B, F_B, Y_B, S, D)
        if ref is not None:
            await grab8(f"T{i}", T_B, D)
            await grab8(f"R{i}", R_B, D)
            await grab8(f"VL{i}", VL_B, D)
            await grab8(f"H{i}", H_B, FF)
            await grab8(f"F{i}", F_B, D)
            await grab8(f"Y{i}", Y_B, D)
        x_base = Y_B

    await vpu_op(dut, GO_LN, Y_B, GF_B, LNF_B, S, D, scale=8)
    LNF8 = await read_R8(dut, S, D, LNF_B)                    # host: LNF as W operand
    if ref is not None:
        check("LNF", LNF8, ref["LNF"], S, D)
    await write_W(dut, LNF8, S, D, LNW_B)
    await gemm(dut, LNW_B, BIAS0, ONES_B, P_B, 1, S, D, cfg_byp(C["sp2"]))
    await gemm(dut, WH_B, BH_B, P_B, LG_B, 1, D, TILE, CFG_I32_B)
    if ref is not None:
        Ph = await read_R8(dut, 1, D, P_B)
        check("P", Ph, ref["P"], 1, D)
    lg = await read_R32(dut, 1, TILE, LG_B)
    if ref is not None:
        check("logits", lg, ref["logits"], 1, TILE)
    pred = int(np.argmax(lg[0][:10]))
    if log:
        log(f"  logits[:10] = {lg[0][:10]}  →  pred {pred}")
    return lg, pred


# ── tests ───────────────────────────────────────────────────────────────────
@cocotb.test()
async def test_vit_tensors(dut):
    """Two digits, every intermediate bit-exact — embed, pos-add, 2 encoder
    layers with per-head Q/K/V/scores/softmax/A·V (NH from the npz meta),
    head concat, final LN, pool, INT32+bias logits."""
    await setup(dut)
    log = dut._log.info
    Wq, C, X100, y100 = oracle()
    for n in (0, 1):
        ref = vit_int(Wq, C, X100[n])
        _, pred = await vit_image(dut, X100[n], ref=ref, log=log)
        assert pred == ref["pred"]
        log(f"image {n} (label {y100[n]}): all tensors bit-exact, pred {pred}")


@cocotb.test()
async def test_vit_batch(dut):
    """First dozen digits: logits + prediction bit-exact per image, accuracy
    reported (the oracle's own 98/100 truth on these 12)."""
    await setup(dut)
    log = dut._log.info
    Wq, C, X100, y100 = oracle()
    hits = 0
    for n in range(12):
        ref = vit_int(Wq, C, X100[n])
        lg, pred = await vit_image(dut, X100[n])
        assert lg == ref["logits"], f"image {n}: logits mismatch"
        assert pred == ref["pred"], f"image {n}: pred {pred} != oracle {ref['pred']}"
        hits += int(pred == y100[n])
        log(f"  image {n}: pred {pred} label {y100[n]}")
    log(f"batch of 12: all logits bit-exact vs oracle, accuracy {hits}/12")
