"""
cocotb END-TO-END CNN TRAINING bench — the mnist_cnn architecture learns MNIST
on the TPU.

Trains conv(1->8,s1) -> conv(8->16,s2) -> conv(16->16,s2) -> FC 784->16 (all
3x3 p1 relu, bias-free) from scratch, batch 4. On-chip: every GEMM (4 forward,
7 backward), softmax, sub, the 3 relu' masks, and the optimizer (accumulate x4,
requant x4, masters in region 2). Host: im2col/col2im (col2im scatter-ADDS —
the one host-arithmetic concession), transposes, packing, batching.

Every chip output is asserted bit-exact per step against cnn_int_model (whose
im2col/col2im also drive the host side, so host data equals model data by
construction). Loss/accuracy printed are the hardware's own numbers.

Env: TRAIN_NPZ (default mnist_train.npz), STEPS (default 1024),
EVAL_EVERY (default 256),
WEIGHTS_OUT (default trained_cnn.npz).
"""

import math
import os
import random

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import First, RisingEdge, Timer

import numpy as np

import cnn_int_model as M
from cnn_int_model import (ARCH, NCLS, B, FLAT, init_state, train_step,
                           forward, batch_loss, im2col, col2im, kernel_to_W)

# ── Geometry / ISA ──────────────────────────────────────────────────────────
TILE, ADDR_W, DIM_W = 16, 13, 12
REGION_W, REGION_A = 0, 1
OP_WEIGHT, OP_BIAS, OP_ACT, OP_RESULT, OP_CONFIG, OP_GO = 1, 2, 3, 4, 5, 6
GO_GEMM, GO_SOFTMAX, GO_SUB, GO_MASK = 0, 1, 3, 5
GO_ACCUM, GO_REQUANT = 7, 8
GO_PROMOTE = 6
ACT_RELU, ACT_BYPASS = 0, 2

# ── Training constants (tuned in the integer model; sealed until the run) ───
C = dict(wi=(24, 24, 24, 12), s=(7, 7, 7), sg=(6, 6, 7), sc=(2, 2),
         sm=512, lr=512, sha=(18, 16, 15, 12), shp=8)
SEED = 1

# Layer GEMM dims: (M, K, N) per conv (batch-4 im2col) + FC
LDIM = [(B * 28 * 28, 9, 8), (B * 14 * 14, 72, 16), (B * 7 * 7, 144, 16)]

# ── Memory map ──────────────────────────────────────────────────────────────
# region 0 — resident weights + W-operand temps:
K_B   = (0, 32, 128)          # K1(16) K2(80) K3(144)
WFC_B = 288                   # 784
WFCT_B, K3T_B, K2T_B = 1088, 1888, 2048
DZFC_W, DZ3_W, DZ2_W, DZ1_W = 2144, 2176, 2400, 3200
BIAS0 = 6400
# region 1 — TOP static zone + rotating workspace @0..6799 (see phase comments):
A2_ST, A3_ST = 6800, 7600     # a2(784) a3(208)
LG_ST, P_ST, OH_ST, DZ_ST = 7812, 7880, 7900, 7920
WS = 0                        # workspace base
A1_ACT = 3200                 # a1 lives here during fwd + is re-written for mask1
# region 2 — masters:
MK_B = (0, 64, 384)           # 64 / 320 / 576
MFC_B = 960                   # 3136

CFG_RELU = [(1 << 11) | (ACT_RELU << 5) | C["s"][l] for l in range(3)]
CFG_I32 = (1 << 11) | (1 << 10)
CFG_BYP = [(1 << 11) | (ACT_BYPASS << 5) | g for g in C["sg"]]


# ── Host port / command helpers (as tb_train_mlp) ───────────────────────────
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


async def wait_done(dut, timeout=2_000_000):
    # Edge-wait (no per-cycle Python polling; measured ~25% wall-time win).
    # Every wait_done follows a GO, which clears done on its accept edge ->
    # done always makes a fresh 0->1 for RisingEdge to catch.
    await First(RisingEdge(dut.done), Timer(timeout * 10, unit="ns"))
    assert int(dut.done.value) == 1, "timeout waiting for done"


def to_signed(v, w):
    return v - (1 << w) if v & (1 << (w - 1)) else v


def pack8(vals):
    return sum((int(v) & 0xFF) << (i * 8) for i, v in enumerate(vals))


# ── Tiled packers (numpy in, chip layout out) ───────────────────────────────
async def write_A(dut, A, base):
    """A-layout (also R-layout for single-N-tile tensors): (i·nKt+k)·16+w."""
    Mr, Kc = A.shape
    nMt, nKt = math.ceil(Mr / TILE), math.ceil(Kc / TILE)
    P = np.zeros((nMt * TILE, nKt * TILE), dtype=np.int64)
    P[:Mr, :Kc] = A
    for i in range(nMt):
        for k in range(nKt):
            for w in range(TILE):
                await mem_write(dut, REGION_A, base + (i * nKt + k) * TILE + w,
                                pack8(P[i * TILE + w, k * TILE:k * TILE + TILE]))


async def write_W(dut, W, base):
    """W-layout: (j·nKt+k)·16+w."""
    Kc, Nc = W.shape
    nKt, nNt = math.ceil(Kc / TILE), math.ceil(Nc / TILE)
    P = np.zeros((nKt * TILE, nNt * TILE), dtype=np.int64)
    P[:Kc, :Nc] = W
    for j in range(nNt):
        for k in range(nKt):
            for w in range(TILE):
                await mem_write(dut, REGION_W, base + (j * nKt + k) * TILE + w,
                                pack8(P[k * TILE + w, j * TILE:j * TILE + TILE]))


async def read_R8(dut, Mr, Nc, base, region=REGION_A):
    nMt, nNt = math.ceil(Mr / TILE), math.ceil(Nc / TILE)
    out = np.zeros((Mr, nNt * TILE), dtype=np.int64)
    for i in range(nMt):
        for j in range(nNt):
            for lr in range(TILE):
                r = i * TILE + lr
                if r >= Mr:
                    continue
                w = await mem_read(dut, region, base + (i * nNt + j) * TILE + lr)
                for c in range(TILE):
                    out[r, j * TILE + c] = to_signed((w >> (c * 8)) & 0xFF, 8)
    return out[:, :Nc]


async def read_W8(dut, Kc, Nc, base):
    nKt, nNt = math.ceil(Kc / TILE), math.ceil(Nc / TILE)
    out = np.zeros((Kc, nNt * TILE), dtype=np.int64)
    for j in range(nNt):
        for k in range(nKt):
            for w in range(TILE):
                r = k * TILE + w
                if r >= Kc:
                    continue
                wd = await mem_read(dut, REGION_W, base + (j * nKt + k) * TILE + w)
                for c in range(TILE):
                    out[r, j * TILE + c] = to_signed((wd >> (c * 8)) & 0xFF, 8)
    return out[:, :Nc]


async def read_R32(dut, Mr, Nc, base):
    nMt, nNt = math.ceil(Mr / TILE), math.ceil(Nc / TILE)
    out = np.zeros((Mr, nNt * TILE), dtype=np.int64)
    for i in range(nMt):
        for j in range(nNt):
            for lr in range(TILE):
                r = i * TILE + lr
                if r >= Mr:
                    continue
                for sub in range(4):
                    w = await mem_read(dut, REGION_A,
                                       base + (i * nNt + j) * 64 + lr * 4 + sub)
                    for p in range(4):
                        out[r, j * TILE + sub * 4 + p] = \
                            to_signed((w >> (p * 32)) & 0xFFFFFFFF, 32)
    return out[:, :Nc]


async def copy_r1_to_r0(dut, src, dst, nwords):
    for w in range(nwords):
        v = await mem_read(dut, REGION_A, src + w)
        await mem_write(dut, REGION_W, dst + w, v)


# ── Dispatch helpers ────────────────────────────────────────────────────────
async def gemm(dut, w_base, a_base, r_base, Mr, Kc, Nc, cfg):
    await send_cmd(dut, OP_WEIGHT, base=w_base, col=Nc)
    await send_cmd(dut, OP_BIAS,   base=BIAS0)
    await send_cmd(dut, OP_ACT,    base=a_base, row=Mr, col=Kc)
    await send_cmd(dut, OP_RESULT, base=r_base)
    await send_cmd(dut, OP_CONFIG, base=cfg)
    await send_cmd(dut, OP_GO,     base=GO_GEMM)
    await wait_done(dut)


async def vpu_op(dut, target, a_base, w_base, r_base, Mr, Nc, scale=0):
    await send_cmd(dut, OP_ACT,    base=a_base, row=Mr, col=Nc)
    await send_cmd(dut, OP_WEIGHT, base=w_base)
    await send_cmd(dut, OP_RESULT, base=r_base)
    await send_cmd(dut, OP_CONFIG, base=0, row=(scale >> 12), col=(scale & 0xFFF))
    await send_cmd(dut, OP_GO,     base=target)
    await wait_done(dut)


def chk(tag, step, hw, ref):
    assert np.array_equal(hw, ref), \
        f"step {step} {tag}: first diff at {np.argwhere(hw != ref)[0]}"


# ── HW forward pass (shared by training + eval + the handwriting finale) ────
async def hw_forward(dut, X4):
    """X4 (4,28,28,1) -> uploads im2col per layer, returns a1,a2,a3,logits (np)."""
    acts = []
    fm = X4
    for l, (ci, co, kh, kw, s, p, h, w) in enumerate(ARCH):
        A = im2col(fm, kh, kw, s, p)
        await write_A(dut, A, WS)
        dst = A1_ACT if l == 0 else (A2_ST if l == 1 else A3_ST)
        Mr, Kc, Nc = LDIM[l]
        await gemm(dut, K_B[l], WS, dst, Mr, Kc, Nc, CFG_RELU[l])
        a = await read_R8(dut, Mr, Nc, dst)
        acts.append(a)
        ho = (h + 2 * p - kh) // s + 1
        wo = (w + 2 * p - kw) // s + 1
        fm = a.reshape(B, ho, wo, co)
    # FC: flat = a3 rows reshaped; A operand = a3 already on chip? layout differs
    # (FC reads batch rows x 784) -> upload flat explicitly.
    flat = fm.reshape(B, FLAT)
    await write_A(dut, flat, WS)
    await gemm(dut, WFC_B, WS, LG_ST, B, FLAT, 16, CFG_I32)
    lg = await read_R32(dut, B, 16, LG_ST)
    return acts, flat, lg


# ── The training run ────────────────────────────────────────────────────────
@cocotb.test()
async def test_train_cnn(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)
    log = dut._log.info
    here = os.path.dirname(__file__)

    d = np.load(os.environ.get("TRAIN_NPZ",
                               os.path.join(here, "mnist_train.npz")))
    Xtr = d["X_train"].astype(np.int64).reshape(-1, 28, 28, 1)
    ytr = d["y_train"].astype(int)
    Xte = d["X_test"].astype(np.int64).reshape(-1, 28, 28, 1)
    yte = d["y_test"].astype(int)
    steps = int(os.environ.get("STEPS", "1024"))
    eval_every = int(os.environ.get("EVAL_EVERY", "256"))
    log(f"CNN TRAIN: {len(Xtr)} imgs, holdout {len(Xte)}, {steps} steps, batch {B}")

    rng = random.Random(SEED)
    st = init_state(rng, C)

    # resident weights + masters (promote once) + bias scratch
    for l in range(3):
        await write_W(dut, st[f"W{l}"], K_B[l])
        k = st[f"W{l}"].shape[0]
        await vpu_op(dut, GO_PROMOTE, 0, K_B[l], MK_B[l], k, st[f"W{l}"].shape[1],
                     scale=C["shp"])
    await write_W(dut, st["Wfc"], WFC_B)
    await vpu_op(dut, GO_PROMOTE, 0, WFC_B, MFC_B, FLAT, 16, scale=C["shp"])
    for w in range(4):
        await mem_write(dut, REGION_W, BIAS0 + w, 0)
    await write_W(dut, st["Wfc"].T.copy(), WFCT_B)      # transposed temps, step 0
    await write_W(dut, st["W2"].T.copy(), K3T_B)
    await write_W(dut, st["W1"].T.copy(), K2T_B)

    async def hw_eval(X, y, limit=None):
        n = limit or (len(X) // B * B)
        hits = 0
        for b in range(0, n, B):
            _, _, lg = await hw_forward(dut, X[b:b + B])
            hits += int((np.argmax(lg[:, :NCLS], 1) == np.array(y[b:b + B])).sum())
        return hits / n

    order = list(range(len(Xtr)))
    nb = len(Xtr) // B
    sca = [(C["lr"] << 5) | C["sha"][i] for i in range(4)]

    for s in range(steps):
        if s % nb == 0:
            rng.shuffle(order)
        idx = order[(s % nb) * B:(s % nb) * B + B]
        X = Xtr[idx]
        lab = [ytr[i] for i in idx]
        oh = np.array([[127 if c == y else 0 for c in range(16)] for y in lab],
                      dtype=np.int64)
        t = train_step(st, X, oh, C)            # model step (same batch)

        # ── forward (a1 stays at A1_ACT until mid-backward) ──
        acts, flat, lg = await hw_forward(dut, X)
        for l in range(3):
            chk(f"a{l}", s, acts[l], t[f"a{l}"])
        chk("logits", s, lg, t["logits"])

        await write_A(dut, oh, OH_ST)
        await vpu_op(dut, GO_SOFTMAX, LG_ST, 0, P_ST, B, NCLS, scale=C["sm"])
        await vpu_op(dut, GO_SUB, P_ST, OH_ST, DZ_ST, B, 16)
        p = await read_R8(dut, B, 16, P_ST)
        dz = await read_R8(dut, B, 16, DZ_ST)
        chk("p", s, p, t["p"])
        chk("dz", s, dz, t["dz"])

        # ── backward: dWfc (accumulate frees the INT32 buffer immediately) ──
        await copy_r1_to_r0(dut, DZ_ST, DZFC_W, 16)
        await write_A(dut, flat.T.copy(), WS)                     # flat^T @0 (784)
        await gemm(dut, DZFC_W, WS, WS + 1024, FLAT, B, 16, CFG_I32)
        await vpu_op(dut, GO_ACCUM, MFC_B, WS + 1024, MFC_B, FLAT, 16, scale=sca[3])

        # dflat -> dY3 -> mask(a3) -> dz3
        await gemm(dut, WFCT_B, DZ_ST, WS, B, 16, FLAT, CFG_BYP[2])
        dflat = await read_R8(dut, B, FLAT, WS)
        chk("dflat", s, dflat, t["dflat"])
        await write_A(dut, dflat.reshape(B * 49, 16), WS + 1024)
        await vpu_op(dut, GO_MASK, WS + 1024, A3_ST, WS + 1024, B * 49, 16)
        dz3 = await read_R8(dut, B * 49, 16, WS + 1024)
        chk("dz3", s, dz3, t["dz2"])                              # model dz2 = layer idx 2
        await copy_r1_to_r0(dut, WS + 1024, DZ3_W, 208)

        # dK3 + dA3
        A3 = im2col(t["a1"].reshape(B, 14, 14, 16), 3, 3, 2, 1)   # = t["A2"]
        await write_A(dut, A3.T.copy(), WS + 2048)                # A3^T (1872)
        await gemm(dut, DZ3_W, WS + 2048, WS, 144, B * 49, 16, CFG_I32)
        await vpu_op(dut, GO_ACCUM, MK_B[2], WS, MK_B[2], 144, 16, scale=sca[2])
        await gemm(dut, K3T_B, WS + 1024, WS + 2048, B * 49, 16, 144, CFG_BYP[1])
        dA3 = await read_R8(dut, B * 49, 144, WS + 2048)
        chk("dA3", s, dA3, t["dA2"])

        # col2im (HOST adds) -> dFM2 -> mask(a2) -> dz2
        dFM2 = M.sat8(col2im(dA3, B, 14, 14, 16, 3, 3, 2, 1) >> C["sc"][1])
        await write_A(dut, dFM2.reshape(B * 196, 16), WS)
        await vpu_op(dut, GO_MASK, WS, A2_ST, WS, B * 196, 16)
        dz2 = await read_R8(dut, B * 196, 16, WS)
        chk("dz2", s, dz2, t["dz1"])
        await copy_r1_to_r0(dut, WS, DZ2_W, 784)

        # dA2 then dK2 (both read dz2; dA2 first, then A2^T overwrites its space)
        await gemm(dut, K2T_B, WS, WS + 1024, B * 196, 16, 72, CFG_BYP[0])
        dA2 = await read_R8(dut, B * 196, 72, WS + 1024)
        chk("dA2", s, dA2, t["dA1"])
        A2 = im2col(t["a0"].reshape(B, 28, 28, 8), 3, 3, 2, 1)    # = t["A1"]
        await write_A(dut, A2.T.copy(), WS + 1024)
        await gemm(dut, DZ2_W, WS + 1024, WS, 72, B * 196, 16, CFG_I32)
        await vpu_op(dut, GO_ACCUM, MK_B[1], WS, MK_B[1], 72, 16, scale=sca[1])

        # col2im -> dFM1 -> mask(a1, re-uploaded) -> dz1 -> dK1
        dFM1 = M.sat8(col2im(dA2, B, 28, 28, 8, 3, 3, 2, 1) >> C["sc"][0])
        await write_A(dut, dFM1.reshape(B * 784, 8), WS)
        await write_A(dut, t["a0"], A1_ACT)                       # a1 back for the mask
        await vpu_op(dut, GO_MASK, WS, A1_ACT, WS, B * 784, 8)
        dz1 = await read_R8(dut, B * 784, 8, WS)
        chk("dz1", s, dz1, t["dz0"])
        await copy_r1_to_r0(dut, WS, DZ1_W, 3136)
        A1 = im2col(X, 3, 3, 1, 1)                                # = t["A0"]
        await write_A(dut, A1.T.copy(), A1_ACT)
        await gemm(dut, DZ1_W, A1_ACT, WS, 9, B * 784, 8, CFG_I32)
        await vpu_op(dut, GO_ACCUM, MK_B[0], WS, MK_B[0], 9, 8, scale=sca[0])

        # ── requant all four, refresh transposed temps, assert weights ──
        for l, (kk, nn) in enumerate(((9, 8), (72, 16), (144, 16))):
            await vpu_op(dut, GO_REQUANT, MK_B[l], 0, K_B[l], kk, nn, scale=C["shp"])
        await vpu_op(dut, GO_REQUANT, MFC_B, 0, WFC_B, FLAT, 16, scale=C["shp"])
        Ks = [await read_W8(dut, *kn, K_B[l]) for l, kn in
              enumerate(((9, 8), (72, 16), (144, 16)))]
        for l in range(3):
            chk(f"K{l}", s, Ks[l], st[f"W{l}"])
        await write_W(dut, Ks[1].T.copy(), K2T_B)
        await write_W(dut, Ks[2].T.copy(), K3T_B)
        if s % 16 == 0 or s == steps - 1:
            Wfc = await read_W8(dut, FLAT, 16, WFC_B)
            chk("Wfc", s, Wfc, st["Wfc"])
            await write_W(dut, Wfc.T.copy(), WFCT_B)
        else:
            await write_W(dut, st["Wfc"].T.copy(), WFCT_B)        # model == chip (asserted)

        if s % 16 == 0:
            log(f"step {s:4d}: loss {batch_loss(p, lab):.3f}  (bit-exact)")
        if s and s % eval_every == 0:
            acc = await hw_eval(Xte, yte, limit=64)               # quick 64-img probe
            log(f"step {s:4d}: holdout(64) accuracy {acc:.2f}")

    # ── final: full evals + weight export ──
    acc_te = await hw_eval(Xte, yte)
    acc_tr = await hw_eval(Xtr, ytr, limit=256)
    out = os.environ.get("WEIGHTS_OUT", os.path.join(here, "trained_cnn.npz"))
    np.savez_compressed(out,
                        K1=st["W0"].astype(np.int8), K2=st["W1"].astype(np.int8),
                        K3=st["W2"].astype(np.int8), Wfc=st["Wfc"].astype(np.int8),
                        shifts=np.array(C["s"], dtype=np.int32),
                        sm_scale=np.int32(C["sm"]),
                        train_acc=np.float32(acc_tr), holdout_acc=np.float32(acc_te))
    log("=" * 64)
    log(f"HW-TRAINED CNN  ({steps} steps, batch {B}, GEMM/softmax/mask/opt on-chip)")
    log(f"  train accuracy (256-img sample) : {acc_tr:.2%}")
    log(f"  holdout accuracy ({len(Xte)} never-seen) : {acc_te:.2%}")
    log(f"  weights saved -> {out}")
    log("=" * 64)
    if steps >= 512:
        assert acc_te > 0.5, "training did not learn"
