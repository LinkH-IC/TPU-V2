"""
Trained pooling-free MNIST CNN on the v2 TPU — the Phase-3 payoff run.

Runs the pooling-free CNN on the UNCHANGED `top` with REAL trained INT8 weights
(mnist_cnn_int8.npz from tarin/train_mnist_cnn.py). Supersedes the earlier
plumbing-first version of this bench (seeded-random weights, pipeline-only): the
plumbing is proven, so this is the conv analogue of the Session-6 MLP run —
bit-exactness AND real classification accuracy on hardware.

Self-contained: weights, per-layer requant shifts, AND the INT8 test images +
labels all come from the .npz (no dependency on the old MLP artifact). Drivers /
run_gemm / im2col / direct-conv reference are the proven helpers from tb/top.

Network (pooling-free, strided-conv downsampling):
    28x28x1 -[conv 3x3 1->8  s1 p1]-> ReLU -> 28x28x8     (M=784 K=9   N=8)
            -[conv 3x3 8->16 s2 p1]-> ReLU -> 14x14x16    (M=196 K=72  N=16)
            -[conv 3x3 16->16 s2 p1]-> ReLU -> 7x7x16     (M=49  K=144 N=16)
            - flatten 7*7*16=784 -[FC 784->16, 10 real]-> bypass -> argmax

Verification (two independent references, like the rest of the suite):
  * test_layerwise_one_image — one image, every layer's feature map + logits
    checked bit-exact vs an INDEPENDENT direct nested-loop conv (no im2col); the
    fast numpy model is anchored to that same direct conv.
  * test_accuracy — all bundled images: hardware logits checked bit-exact vs the
    numpy integer model for every image, then argmax vs the true labels for the
    real accuracy + the autonomy / cycles-per-image metrics.
"""

import math
import os
import numpy as np

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge
from cocotb.utils import get_sim_time


# ── Fixed geometry (top params) ────────────────────────────────────────────
TILE   = 16
DATA_W = 8
ACC_W  = 32
ADDR_W = 13
DIM_W  = 12

REGION_W = 0
REGION_A = 1

OP_WEIGHT, OP_BIAS, OP_ACT, OP_RESULT, OP_CONFIG, OP_GO = 1, 2, 3, 4, 5, 6
ACT_RELU, ACT_LEAKY, ACT_BYPASS = 0, 1, 2


# ── Pure-Python integer reference (datapath tail) ──────────────────────────

def to_signed(value, width):
    return value - (1 << width) if value & (1 << (width - 1)) else value


def bit(sig):
    try:
        return int(sig.value)
    except (ValueError, TypeError):
        return None


def sat8(x):
    if x > 127:
        return 127
    if x < -128:
        return -128
    return x


def activation(x, shift, act_sel, leak_shift=0):
    """Bit-exact mirror of the RTL activation modules (input x = acc + bias)."""
    if act_sel == ACT_RELU:
        clipped = x if x >= 0 else 0
        shifted = clipped >> shift
        return 127 if shifted > 127 else shifted
    if act_sel == ACT_BYPASS:
        return sat8(x >> shift)
    if act_sel == ACT_LEAKY:
        shifted = x >> shift
        return sat8((shifted >> leak_shift) if shifted < 0 else shifted)
    raise ValueError(f"unsupported act_sel {act_sel}")


# ── Bit packing ────────────────────────────────────────────────────────────

def pack_lanes(vals, width=DATA_W):
    mask = (1 << width) - 1
    out = 0
    for i, v in enumerate(vals):
        out |= (int(v) & mask) << (i * width)
    return out


def pack_bias(vals):
    mask = (1 << ACC_W) - 1
    out = 0
    for p, v in enumerate(vals):
        out |= (int(v) & mask) << (p * ACC_W)
    return out


def pad2d(M2, rows, cols, R, C):
    return [[(M2[r][c] if (r < rows and c < cols) else 0) for c in range(C)]
            for r in range(R)]


# ── DUT host-side drivers (from tb/top) ──

def _host_idle(dut):
    dut.host_addr.value = 0
    dut.host_en.value = 0
    dut.host_we.value = 0
    dut.host_wdata.value = 0


async def reset(dut):
    dut.rst_n.value = 0
    _host_idle(dut)
    dut.cmd_valid.value = 0
    dut.cmd_word.value = 0
    for _ in range(3):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


async def mem_write(dut, region, word_addr, data128):
    dut.host_addr.value = (region << ADDR_W) | word_addr
    dut.host_en.value = 1
    dut.host_we.value = 1
    dut.host_wdata.value = data128
    await RisingEdge(dut.clk)
    _host_idle(dut)


async def mem_read(dut, region, word_addr):
    dut.host_addr.value = (region << ADDR_W) | word_addr
    dut.host_en.value = 1
    dut.host_we.value = 0
    await RisingEdge(dut.clk)
    _host_idle(dut)
    await RisingEdge(dut.clk)
    val = dut.host_rdata.value
    assert val.is_resolvable, f"host_rdata X on read of region {region} addr {word_addr}"
    return val.to_unsigned()


def pack_cmd(op, base=0, row=0, col=0):
    return (op << (2 * DIM_W + ADDR_W)) | (base << (2 * DIM_W)) | (row << DIM_W) | col


async def send_cmd(dut, op, base=0, row=0, col=0):
    dut.cmd_word.value = pack_cmd(op, base, row, col)
    dut.cmd_valid.value = 1
    await RisingEdge(dut.clk)
    dut.cmd_valid.value = 0
    dut.cmd_word.value = 0


async def wait_done(dut, timeout=400_000):
    for _ in range(timeout):
        await RisingEdge(dut.clk)
        if bit(dut.done) == 1:
            return
    raise AssertionError("timeout waiting for cmd_proc done")


# ── End-to-end GEMM driver ─────────────────────────────────────────────────

async def run_gemm(dut, A, W, bias, M, K, N,
                   shift=0, act_sel=ACT_BYPASS, leak_shift=0):
    """Load operands, run one instruction, read R back, slice to M×N."""
    nMt = math.ceil(M / TILE)
    nKt = math.ceil(K / TILE)
    nNt = math.ceil(N / TILE)

    W_base    = 0
    bias_base = nNt * nKt * TILE
    A_base    = 0
    R_base    = nMt * nKt * TILE

    Apad = pad2d(A, M, K, nMt * TILE, nKt * TILE)
    Wpad = pad2d(W, K, N, nKt * TILE, nNt * TILE)
    biaspad = [bias[n] if n < N else 0 for n in range(nNt * TILE)]

    for j in range(nNt):
        for k in range(nKt):
            for w in range(TILE):
                vals = [Wpad[k * TILE + w][j * TILE + c] for c in range(TILE)]
                await mem_write(dut, REGION_W, W_base + (j * nKt + k) * TILE + w,
                                pack_lanes(vals))

    for j in range(nNt):
        for p in range(TILE // 4):
            vals = [biaspad[j * TILE + p * 4 + q] for q in range(4)]
            await mem_write(dut, REGION_W, bias_base + j * 4 + p, pack_bias(vals))

    for i in range(nMt):
        for k in range(nKt):
            for w in range(TILE):
                vals = [Apad[i * TILE + w][k * TILE + c] for c in range(TILE)]
                await mem_write(dut, REGION_A, A_base + (i * nKt + k) * TILE + w,
                                pack_lanes(vals))

    cfg = (leak_shift << 7) | (act_sel << 5) | shift
    await send_cmd(dut, OP_WEIGHT, base=W_base, col=N)
    await send_cmd(dut, OP_BIAS,   base=bias_base)
    await send_cmd(dut, OP_ACT,    base=A_base, row=M, col=K)
    await send_cmd(dut, OP_RESULT, base=R_base)
    await send_cmd(dut, OP_CONFIG, base=cfg)
    await send_cmd(dut, OP_GO)

    await wait_done(dut)

    C = [[0] * N for _ in range(M)]
    for i in range(nMt):
        for j in range(nNt):
            for w in range(TILE):
                word = await mem_read(dut, REGION_A, R_base + (i * nNt + j) * TILE + w)
                for c in range(TILE):
                    gm, gn = i * TILE + w, j * TILE + c
                    if gm < M and gn < N:
                        C[gm][gn] = to_signed((word >> (c * DATA_W)) & 0xFF, DATA_W)
    return C


# ── Convolution layer (software im2col → GEMM, + independent direct ref) ───

def conv_out(in_dim, k, stride, pad):
    return (in_dim + 2 * pad - k) // stride + 1


def ifm_get(ifm, ih, iw, ci, H_in, W_in):
    if 0 <= ih < H_in and 0 <= iw < W_in:
        return ifm[ih][iw][ci]
    return 0


def im2col(ifm, H_in, W_in, C_in, KH, KW, H_out, W_out, stride, pad):
    K = KH * KW * C_in
    A = []
    for oh in range(H_out):
        for ow in range(W_out):
            row = [0] * K
            for kh in range(KH):
                for kw in range(KW):
                    for ci in range(C_in):
                        ih = oh * stride + kh - pad
                        iw = ow * stride + kw - pad
                        k = (kh * KW + kw) * C_in + ci
                        row[k] = ifm_get(ifm, ih, iw, ci, H_in, W_in)
            A.append(row)
    return A


def kernel_to_W(kernel, C_out, KH, KW, C_in):
    K = KH * KW * C_in
    W = [[0] * C_out for _ in range(K)]
    for oc in range(C_out):
        for kh in range(KH):
            for kw in range(KW):
                for ci in range(C_in):
                    k = (kh * KW + kw) * C_in + ci
                    W[k][oc] = kernel[oc][kh][kw][ci]
    return W


def ref_conv_direct(ifm, kernel, bias, H_in, W_in, C_in, KH, KW, C_out,
                    stride, pad, shift, act_sel):
    """Independent ground truth: direct nested-loop conv (no im2col) + tail."""
    H_out = conv_out(H_in, KH, stride, pad)
    W_out = conv_out(W_in, KW, stride, pad)
    OFM = [[[0] * C_out for _ in range(W_out)] for _ in range(H_out)]
    for oh in range(H_out):
        for ow in range(W_out):
            for oc in range(C_out):
                acc = 0
                for kh in range(KH):
                    for kw in range(KW):
                        for ci in range(C_in):
                            ih = oh * stride + kh - pad
                            iw = ow * stride + kw - pad
                            acc += ifm_get(ifm, ih, iw, ci, H_in, W_in) \
                                * kernel[oc][kh][kw][ci]
                OFM[oh][ow][oc] = activation(acc + bias[oc], shift, act_sel)
    return OFM


async def run_conv(dut, ifm, kernel, bias, H_in, W_in, C_in, KH, KW, C_out,
                   stride, pad, shift, act_sel):
    """im2col in software -> conv as ONE GEMM on `top` -> reshape to OFM[h][w][c]."""
    H_out = conv_out(H_in, KH, stride, pad)
    W_out = conv_out(W_in, KW, stride, pad)
    M, K, N = H_out * W_out, KH * KW * C_in, C_out

    A = im2col(ifm, H_in, W_in, C_in, KH, KW, H_out, W_out, stride, pad)
    Wmat = kernel_to_W(kernel, C_out, KH, KW, C_in)
    Cflat = await run_gemm(dut, A, Wmat, bias, M, K, N, shift, act_sel)

    OFM = [[[Cflat[oh * W_out + ow][oc] for oc in range(C_out)]
            for ow in range(W_out)] for oh in range(H_out)]
    return OFM


# ── the trained CNN: architecture, weights, forward passes ─────────────────

#  (Cin, Cout, KH, KW, stride, pad, act) — shifts come from the .npz
ARCH = [
    (1,  8, 3, 3, 1, 1, ACT_RELU),
    (8, 16, 3, 3, 2, 1, ACT_RELU),
    (16, 16, 3, 3, 2, 1, ACT_RELU),
]
FC_OUT, N_CLASSES, FLAT_DIM, FC_ACT = 16, 10, 7 * 7 * 16, ACT_BYPASS


def load_npz():
    """Trained weights/shifts + bundled INT8 test set. Weights kept in BOTH forms:
    python-int nested lists (for the hardware drivers + direct-conv reference, no
    overflow) and numpy arrays (for the fast vectorised integer model)."""
    npz = os.environ.get("MNIST_NPZ", "mnist_cnn_int8.npz")   # swap inputs w/o editing
    d = np.load(os.path.join(os.path.dirname(__file__), npz))
    kernels_np = [d[f"k{L}"] for L in range(3)]
    biases_np  = [d[f"cb{L}"] for L in range(3)]
    return dict(
        kernels=[k.astype(np.int64).tolist() for k in kernels_np],
        biases=[b.astype(np.int64).tolist() for b in biases_np],
        Wfc=d["Wfc"].astype(np.int64).tolist(),
        bfc=d["bfc"].astype(np.int64).tolist(),
        kernels_np=kernels_np, biases_np=biases_np,
        Wfc_np=d["Wfc"], bfc_np=d["bfc"],
        shifts=d["shifts"].tolist(),
        X=d["X_test"], y=d["y_test"],
        int_acc=float(d["int_acc"]), float_acc=float(d["float_acc"]),
    )


def image_to_hwc(row784):
    """Flat INT8 image (784) -> 28x28x1 HWC python-int feature map."""
    return [[[int(row784[h * 28 + w])] for w in range(28)] for h in range(28)]


def flatten_hwc(fm):
    return [fm[h][w][c] for h in range(len(fm))
            for w in range(len(fm[0])) for c in range(len(fm[0][0]))]


# ── fast vectorised numpy integer model (bit-exact mirror; the 100-img ref) ─

def _np_conv(x, k, b, stride, pad, shift, act):
    Cout, KH, KW, Cin = k.shape
    N, H, W, _ = x.shape
    Ho = (H + 2 * pad - KH) // stride + 1
    Wo = (W + 2 * pad - KW) // stride + 1
    xp = np.pad(x, ((0, 0), (pad, pad), (pad, pad), (0, 0)))
    cols = np.empty((N, Ho, Wo, KH, KW, Cin), dtype=np.int64)
    for kh in range(KH):
        for kw in range(KW):
            cols[:, :, :, kh, kw, :] = \
                xp[:, kh:kh + stride * Ho:stride, kw:kw + stride * Wo:stride, :]
    A = cols.reshape(N, Ho * Wo, KH * KW * Cin)
    Wm = k.transpose(1, 2, 3, 0).reshape(KH * KW * Cin, Cout).astype(np.int64)
    acc = A @ Wm + b.astype(np.int64)[None, None, :]
    y = np.clip(np.where(acc < 0, 0, acc) >> shift, 0, 127)   # ACT_RELU
    return y.reshape(N, Ho, Wo, Cout)


def np_forward(images, D):
    """images[N,28,28,1] -> logits[N,16], bit-exact with the RTL datapath."""
    x = images.astype(np.int64)
    for L, (Cin, Cout, KH, KW, s, p, act) in enumerate(ARCH):
        x = _np_conv(x, D["kernels_np"][L], D["biases_np"][L], s, p, D["shifts"][L], act)
    vec = x.reshape(x.shape[0], -1)                           # HWC flatten
    logits = vec @ D["Wfc_np"].astype(np.int64) + D["bfc_np"].astype(np.int64)[None, :]
    return np.clip(logits >> D["shifts"][3], -128, 127)


async def hw_forward(dut, image, D):
    """Full trained CNN on `top`: chained run_conv (feature map bounces through
    the host for each im2col), flatten, then the FC GEMM."""
    fm, H, W, fmaps = image, 28, 28, []
    for L, (Cin, Cout, KH, KW, s, p, act) in enumerate(ARCH):
        fm = await run_conv(dut, fm, D["kernels"][L], D["biases"][L], H, W,
                            Cin, KH, KW, Cout, s, p, D["shifts"][L], act)
        H, W = conv_out(H, KH, s, p), conv_out(W, KW, s, p)
        fmaps.append(fm)
    vec = flatten_hwc(fm)
    logits_row = await run_gemm(dut, [vec], D["Wfc"], D["bfc"], 1, FLAT_DIM, FC_OUT,
                                D["shifts"][3], FC_ACT)
    return fmaps, logits_row[0]


def ref_forward(image, D):
    """Independent reference: direct nested-loop conv chain + FC."""
    fm, H, W, fmaps = image, 28, 28, []
    for L, (Cin, Cout, KH, KW, s, p, act) in enumerate(ARCH):
        fm = ref_conv_direct(fm, D["kernels"][L], D["biases"][L], H, W,
                             Cin, KH, KW, Cout, s, p, D["shifts"][L], act)
        H, W = conv_out(H, KH, s, p), conv_out(W, KW, s, p)
        fmaps.append(fm)
    vec = flatten_hwc(fm)
    logits = [activation(sum(vec[k] * D["Wfc"][k][n] for k in range(FLAT_DIM)) + D["bfc"][n],
                         D["shifts"][3], FC_ACT) for n in range(FC_OUT)]
    return fmaps, logits


def check_fmap(got, exp, tag):
    H, W, C = len(exp), len(exp[0]), len(exp[0][0])
    for h in range(H):
        for w in range(W):
            for c in range(C):
                assert got[h][w][c] == exp[h][w][c], (
                    f"{tag}: fmap[{h}][{w}][{c}] got {got[h][w][c]} exp {exp[h][w][c]}")


async def setup(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)


# ── resident weights + image batching (the optimized path) ─────────────────
DEPTH = 8192   # words per memory (mem.sv); A + R of one GEMM must fit together


def gemm_words(M, K, N):
    """(A_words, R_words) a GEMM of these dims occupies in activation memory."""
    nMt, nKt, nNt = math.ceil(M / TILE), math.ceil(K / TILE), math.ceil(N / TILE)
    return nMt * nKt * TILE, nMt * nNt * TILE


def max_batch(M_img, K, N):
    """Largest #images whose stacked GEMM (A+R) still fits one activation memory."""
    B = 1
    while True:
        a, r = gemm_words((B + 1) * M_img, K, N)
        if a + r > DEPTH:
            return B
        B += 1


def weight_layout():
    """Resident weight-memory map: (W_base, bias_base) per layer (3 conv + FC).
    All layers fit at once (separate weight memory), so weights load ONCE."""
    specs = [(KH * KW * Cin, Cout) for (Cin, Cout, KH, KW, s, p, act) in ARCH]
    specs.append((FLAT_DIM, FC_OUT))
    bases, addr = [], 0
    for K, N in specs:
        nKt, nNt = math.ceil(K / TILE), math.ceil(N / TILE)
        Wb = addr; addr += nNt * nKt * TILE
        Bb = addr; addr += nNt * (TILE // 4)
        bases.append((Wb, Bb))
    assert addr <= DEPTH, f"weight memory overflow: {addr} > {DEPTH}"
    return bases


async def load_weights_resident(dut, Wmat, bias, K, N, W_base, bias_base):
    """Write one layer's weight + bias tiles to fixed bases (done once, up front)."""
    nKt, nNt = math.ceil(K / TILE), math.ceil(N / TILE)
    Wpad = pad2d(Wmat, K, N, nKt * TILE, nNt * TILE)
    biaspad = [bias[n] if n < N else 0 for n in range(nNt * TILE)]
    for j in range(nNt):
        for k in range(nKt):
            for w in range(TILE):
                vals = [Wpad[k * TILE + w][j * TILE + c] for c in range(TILE)]
                await mem_write(dut, REGION_W, W_base + (j * nKt + k) * TILE + w,
                                pack_lanes(vals))
    for j in range(nNt):
        for p in range(TILE // 4):
            vals = [biaspad[j * TILE + p * 4 + q] for q in range(4)]
            await mem_write(dut, REGION_W, bias_base + j * 4 + p, pack_bias(vals))


async def preload_weights(dut, D):
    """Upload every layer's weights to its resident base — once for the whole run."""
    bases = weight_layout()
    for L, (Cin, Cout, KH, KW, s, p, act) in enumerate(ARCH):
        Wmat = kernel_to_W(D["kernels"][L], Cout, KH, KW, Cin)
        await load_weights_resident(dut, Wmat, D["biases"][L], KH * KW * Cin, Cout, *bases[L])
    await load_weights_resident(dut, D["Wfc"], D["bfc"], FLAT_DIM, FC_OUT, *bases[3])
    return bases


async def run_gemm_resident(dut, A, M, K, N, W_base, bias_base, shift, act_sel):
    """Like run_gemm but weights are ALREADY resident at W_base/bias_base — only
    the activations are written, then one descriptor stream + GO + read-back."""
    nMt, nKt, nNt = math.ceil(M / TILE), math.ceil(K / TILE), math.ceil(N / TILE)
    A_base, R_base = 0, nMt * nKt * TILE
    assert R_base + nMt * nNt * TILE <= DEPTH, f"act-mem overflow M={M} K={K} N={N}"

    Apad = pad2d(A, M, K, nMt * TILE, nKt * TILE)
    for i in range(nMt):
        for k in range(nKt):
            for w in range(TILE):
                vals = [Apad[i * TILE + w][k * TILE + c] for c in range(TILE)]
                await mem_write(dut, REGION_A, A_base + (i * nKt + k) * TILE + w,
                                pack_lanes(vals))

    await send_cmd(dut, OP_WEIGHT, base=W_base, col=N)
    await send_cmd(dut, OP_BIAS,   base=bias_base)
    await send_cmd(dut, OP_ACT,    base=A_base, row=M, col=K)
    await send_cmd(dut, OP_RESULT, base=R_base)
    await send_cmd(dut, OP_CONFIG, base=(act_sel << 5) | shift)
    await send_cmd(dut, OP_GO)
    await wait_done(dut)

    C = [[0] * N for _ in range(M)]
    for i in range(nMt):
        for j in range(nNt):
            for w in range(TILE):
                word = await mem_read(dut, REGION_A, R_base + (i * nNt + j) * TILE + w)
                for c in range(TILE):
                    gm, gn = i * TILE + w, j * TILE + c
                    if gm < M and gn < N:
                        C[gm][gn] = to_signed((word >> (c * DATA_W)) & 0xFF, DATA_W)
    return C


async def run_layer_batched(dut, fmaps_in, L, W_base, bias_base, shift):
    """One conv layer over MANY images at once: stack each image's im2col patches
    into a single tall GEMM (weights resident, loaded into the array ONCE per
    chunk). Returns (output feature maps, batch size used)."""
    Cin, Cout, KH, KW, s, p, act = ARCH[L]
    H, W = len(fmaps_in[0]), len(fmaps_in[0][0])
    Ho, Wo = conv_out(H, KH, s, p), conv_out(W, KW, s, p)
    M_img, K, N = Ho * Wo, KH * KW * Cin, Cout
    B = max_batch(M_img, K, N)

    out = []
    for start in range(0, len(fmaps_in), B):
        chunk = fmaps_in[start:start + B]
        A = []
        for fm in chunk:
            A += im2col(fm, H, W, Cin, KH, KW, Ho, Wo, s, p)
        Cflat = await run_gemm_resident(dut, A, len(A), K, N, W_base, bias_base, shift, act)
        for bi in range(len(chunk)):
            rows = Cflat[bi * M_img:(bi + 1) * M_img]
            out.append([[[rows[oh * Wo + ow][oc] for oc in range(N)]
                         for ow in range(Wo)] for oh in range(Ho)])
    return out, B


async def run_fc_batched(dut, vecs, W_base, bias_base, shift):
    """The FC over MANY images at once (M=1 per image -> stack into one GEMM)."""
    B = max_batch(1, FLAT_DIM, FC_OUT)
    out = []
    for start in range(0, len(vecs), B):
        chunk = vecs[start:start + B]
        out += await run_gemm_resident(dut, chunk, len(chunk), FLAT_DIM, FC_OUT,
                                       W_base, bias_base, shift, FC_ACT)
    return out, B


# ── tests ───────────────────────────────────────────────────────────────────

@cocotb.test()
async def test_layerwise_one_image(dut):
    """One image: every layer's feature map + logits bit-exact vs the independent
    direct-conv reference; the fast numpy model is anchored to that same ref."""
    await setup(dut)
    D = load_npz()
    dut._log.info(f"trained shifts={D['shifts']}  reported int_acc(10k)={D['int_acc']:.4f}")
    img = image_to_hwc(D["X"][0])

    fmaps_hw, logits_hw = await hw_forward(dut, img, D)
    fmaps_ref, logits_ref = ref_forward(img, D)

    for L, name in enumerate(["conv1 28x28x8", "conv2 14x14x16", "conv3 7x7x16"]):
        check_fmap(fmaps_hw[L], fmaps_ref[L], f"img0 {name}")
        flat = np.array(fmaps_ref[L]).reshape(-1)
        dut._log.info(f"img0 {name}: bit-exact OK  (min{flat.min()} "
                      f"max{flat.max()} nz{(flat != 0).mean():.2f})")

    assert logits_hw == logits_ref, f"img0 logits hw {logits_hw} != ref {logits_ref}"
    np_logits = np_forward(np.array([img]), D)[0].tolist()
    assert np_logits == logits_ref, f"numpy ref {np_logits} != direct-conv ref {logits_ref}"
    pred, true = int(np.argmax(logits_ref[:N_CLASSES])), int(D["y"][0])
    dut._log.info(f"img0 logits bit-exact (hw == direct-conv == numpy)  "
                  f"pred={pred} true={true}")


@cocotb.test()
async def test_accuracy(dut):
    """All bundled images through the OPTIMIZED path — weights preloaded ONCE
    (resident) and images batched per layer to fill the M stream (batch sized to
    fit the 4096-word activation memory). Hardware logits checked bit-exact vs the
    numpy integer model for every image, then argmax vs labels for the real
    accuracy + the cycle metrics."""
    await setup(dut)
    D = load_npz()
    X, y, n = D["X"], D["y"], len(D["y"])
    ref_logits = np_forward(X.reshape(n, 28, 28, 1), D)        # oracle

    t_begin = get_sim_time("ns")
    bases = await preload_weights(dut, D)                      # ONCE for all images
    t_pre = get_sim_time("ns")

    fmaps = [image_to_hwc(X[i]) for i in range(n)]
    batch = []
    for L in range(3):
        fmaps, B = await run_layer_batched(dut, fmaps, L, *bases[L], D["shifts"][L])
        batch.append(B)
    vecs = [flatten_hwc(fm) for fm in fmaps]
    logits, Bfc = await run_fc_batched(dut, vecs, *bases[3], D["shifts"][3])
    batch.append(Bfc)
    t_end = get_sim_time("ns")

    correct, preds = 0, []
    for i in range(n):
        assert logits[i] == ref_logits[i].tolist(), (
            f"img{i}: hw logits {logits[i]} != model {ref_logits[i].tolist()}")
        p = int(np.argmax(logits[i][:N_CLASSES]))
        preds.append(p)
        correct += (p == int(y[i]))
    acc = correct / n
    preload_cyc = int((t_pre - t_begin) // 10)
    proc_cyc = int((t_end - t_pre) // 10)
    n_gemm = sum(math.ceil(n / b) for b in batch)
    custom = os.environ.get("MNIST_NPZ") not in (None, "mnist_cnn_int8.npz")

    dut._log.info("══════════ trained MNIST CNN on v2 — optimized payoff ══════════")
    dut._log.info(f"bit-exact  hw == integer model : {n}/{n}")
    dut._log.info(f"accuracy   ({n} images)         : {correct}/{n} = {acc:.4f}")
    if n <= 20:
        dut._log.info("per-image label→pred : " +
                      "  ".join(f"{int(y[i])}→{preds[i]}{'' if preds[i]==int(y[i]) else '✗'}"
                               for i in range(n)))
    dut._log.info(f"per-layer image batch          : conv{batch[:3]}  fc {batch[3]}")
    dut._log.info(f"GEMM dispatches (whole run)    : {n_gemm} for {n} images "
                  f"({n_gemm / n:.2f}/image)")
    dut._log.info(f"weight preload (one-time)      : {preload_cyc} cyc")
    dut._log.info(f"~cycles / image (compute+I/O)  : {proc_cyc // n}  (weights resident)")
    dut._log.info("════════════════════════════════════════════════════════════════")
    if custom:
        dut._log.info("(custom inputs — accuracy is informational, not asserted)")
    else:
        assert acc > 0.90, f"hardware accuracy {acc:.4f} unexpectedly low"
