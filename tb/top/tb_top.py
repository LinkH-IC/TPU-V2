"""
cocotb integration test for the v2 top — the instruction-driven 16x16 TPU.

This is the real Phase-2 verification floor: the first time cmd_proc + the four
rebuilt buffers + the two memories run together. It exercises the whole host
contract end to end:

    1. host pre-loads weight mem (W tiles + bias) and activation mem (A tiles)
       through the ONE region-decoded memory port (Port A);
    2. host streams the typed command descriptors (WEIGHT/BIAS/ACT/RESULT/CONFIG)
       then GO;
    3. cmd_proc sweeps every output tile by itself  ->  R = act((A·W + bias) >> shift);
    4. host polls `done`, reads R back from memory, slices the real M×N region.

Everything is checked bit-exact against a pure-Python integer model of the exact
datapath (matmul -> +bias -> arithmetic right shift -> activation -> saturate INT8).

Memory layout (pre-tiled, contiguous, k-inner — matches cmd_proc's AGUs and the
buffer staging indexing):

    W    = W_base    + (j*nKt + k)*16 + w     word w = W-row (k*16+w), cols j*16..+15
    A    = A_base    + (i*nKt + k)*16 + w     word w = A-row (i*16+w), cols k*16..+15
    R    = R_base    + (i*nNt + j)*16 + w     word w = C-row (i*16+w), cols j*16..+15
    bias = bias_base + j*4 + p                word p = 4 packed INT32, col j*16+p*4..+3

Weight mem and bias live in region 0; A and R live in region 1 (the unified
activation/result memory). Host picks non-overlapping bases.
"""

import math
import random

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge


# ── Fixed geometry (top params) ────────────────────────────────────────────
TILE   = 16            # 16x16 array / 16 lanes per word
DATA_W = 8             # INT8 operands
ACC_W  = 32            # accumulator / bias width
ADDR_W = 13            # word-address width (DEPTH=8192)
DIM_W  = 12            # M/K/N descriptor field width

REGION_W = 0           # host_addr[ADDR_W] = 0 -> weight memory
REGION_A = 1           # host_addr[ADDR_W] = 1 -> activation/result memory

# Opcodes (green card)
OP_WEIGHT = 1
OP_BIAS   = 2
OP_ACT    = 3
OP_RESULT = 4
OP_CONFIG = 5
OP_GO     = 6

# act_sel encodings
ACT_RELU   = 0
ACT_LEAKY  = 1
ACT_BYPASS = 2
ACT_GELU   = 3

# GELU ROM — identical formula to rtl/gelu_lut.sv (F=4): ROM[u] = sat8(round(K·gelu(s/K)))
GELU_FBITS = 4


def _gelu_rom(u):
    s = u - 256 if u >= 128 else u                       # signed 8-bit
    real = s / (1 << GELU_FBITS)
    v = math.floor(0.5 * real * (1 + math.erf(real / math.sqrt(2))) * (1 << GELU_FBITS) + 0.5)
    return max(-128, min(127, v))


GELU_ROM = [_gelu_rom(u) for u in range(256)]


# ── Pure-Python integer reference ──────────────────────────────────────────

def to_signed(value, width):
    return value - (1 << width) if value & (1 << (width - 1)) else value


def bit(sig):
    """Read a 1-bit signal as 0/1, or None if unresolvable (X/Z)."""
    try:
        return int(sig.value)
    except (ValueError, TypeError):
        return None


def sat8(x):
    """Clamp a signed value to INT8 [-128, 127]."""
    if x > 127:
        return 127
    if x < -128:
        return -128
    return x


def activation(x, shift, act_sel, leak_shift):
    """Bit-exact mirror of the RTL activation modules (input x = acc + bias)."""
    if act_sel == ACT_RELU:
        clipped = x if x >= 0 else 0           # ReLU
        shifted = clipped >> shift             # value >= 0
        return 127 if shifted > 127 else shifted
    if act_sel == ACT_LEAKY:
        shifted = x >> shift
        leaked = (shifted >> leak_shift) if shifted < 0 else shifted
        return sat8(leaked)
    if act_sel == ACT_BYPASS:
        return sat8(x >> shift)
    # ACT_GELU: requant like the others, clamp to 8b, ROM lookup
    s = x >> shift
    s = 127 if s > 127 else (-128 if s < -128 else s)
    return GELU_ROM[s & 0xFF]


def ref_gemm(A, W, bias, M, K, N, shift, act_sel, leak_shift):
    """Exact C[m][n] over the real M×N region. Python ints are unbounded."""
    C = [[0] * N for _ in range(M)]
    for m in range(M):
        for n in range(N):
            acc = sum(A[m][k] * W[k][n] for k in range(K))
            C[m][n] = activation(acc + bias[n], shift, act_sel, leak_shift)
    return C


# ── Bit packing ────────────────────────────────────────────────────────────

def pack_lanes(vals, width=DATA_W):
    """Pack TILE signed values into one 128b word; lane 0 at the LSB."""
    mask = (1 << width) - 1
    out = 0
    for i, v in enumerate(vals):
        out |= (v & mask) << (i * width)
    return out


def pack_bias(vals):
    """Pack 4 signed INT32 biases into one 128b word; bias 0 at the LSB."""
    mask = (1 << ACC_W) - 1
    out = 0
    for p, v in enumerate(vals):
        out |= (v & mask) << (p * ACC_W)
    return out


def pad2d(M2, rows, cols, R, C):
    """Zero-pad a rows×cols matrix up to R×C."""
    return [[(M2[r][c] if (r < rows and c < cols) else 0) for c in range(C)]
            for r in range(R)]


# ── DUT host-side drivers ──────────────────────────────────────────────────

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
    """Single full-word write through host Port A (commits on the clock edge)."""
    dut.host_addr.value = (region << ADDR_W) | word_addr
    dut.host_en.value = 1
    dut.host_we.value = 1
    dut.host_wdata.value = data128
    await RisingEdge(dut.clk)
    _host_idle(dut)


async def mem_read(dut, region, word_addr):
    """Synchronous (1-cycle) read: present addr+en, then sample host_rdata the
    cycle after the address edge (Port A read is registered)."""
    dut.host_addr.value = (region << ADDR_W) | word_addr
    dut.host_en.value = 1
    dut.host_we.value = 0
    await RisingEdge(dut.clk)            # edge: ram[addr] -> a_rdata, region -> region_q
    _host_idle(dut)
    await RisingEdge(dut.clk)            # registered rdata is stable to sample now
    val = dut.host_rdata.value
    assert val.is_resolvable, f"host_rdata X on read of region {region} addr {word_addr}"
    return val.to_unsigned()


def pack_cmd(op, base=0, row=0, col=0):
    """[ op(3) : base(ADDR_W) : row(DIM_W) : col(DIM_W) ], op at the MSB."""
    return (op << (2 * DIM_W + ADDR_W)) | (base << (2 * DIM_W)) | (row << DIM_W) | col


async def send_cmd(dut, op, base=0, row=0, col=0):
    dut.cmd_word.value = pack_cmd(op, base, row, col)
    dut.cmd_valid.value = 1
    await RisingEdge(dut.clk)
    dut.cmd_valid.value = 0
    dut.cmd_word.value = 0


async def wait_done(dut, timeout=200_000):
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

    # Non-overlapping bases. Weight mem (region 0): W tiles then bias.
    # Act mem (region 1): A tiles then R tiles.
    W_base    = 0
    bias_base = nNt * nKt * TILE
    A_base    = 0
    R_base    = nMt * nKt * TILE

    Apad = pad2d(A, M, K, nMt * TILE, nKt * TILE)
    Wpad = pad2d(W, K, N, nKt * TILE, nNt * TILE)
    biaspad = [bias[n] if n < N else 0 for n in range(nNt * TILE)]

    # Weights -> region 0 (word w = W-row k*16+w, cols j*16..+15)
    for j in range(nNt):
        for k in range(nKt):
            for w in range(TILE):
                vals = [Wpad[k * TILE + w][j * TILE + c] for c in range(TILE)]
                await mem_write(dut, REGION_W, W_base + (j * nKt + k) * TILE + w,
                                pack_lanes(vals))

    # Bias -> region 0 (4 packed INT32 per word)
    for j in range(nNt):
        for p in range(TILE // 4):
            vals = [biaspad[j * TILE + p * 4 + q] for q in range(4)]
            await mem_write(dut, REGION_W, bias_base + j * 4 + p, pack_bias(vals))

    # Activations -> region 1 (word w = A-row i*16+w, cols k*16..+15)
    for i in range(nMt):
        for k in range(nKt):
            for w in range(TILE):
                vals = [Apad[i * TILE + w][k * TILE + c] for c in range(TILE)]
                await mem_write(dut, REGION_A, A_base + (i * nKt + k) * TILE + w,
                                pack_lanes(vals))

    # Command stream (any order before GO), then GO.
    cfg = (leak_shift << 7) | (act_sel << 5) | shift
    await send_cmd(dut, OP_WEIGHT, base=W_base, col=N)
    await send_cmd(dut, OP_BIAS,   base=bias_base)
    await send_cmd(dut, OP_ACT,    base=A_base, row=M, col=K)
    await send_cmd(dut, OP_RESULT, base=R_base)
    await send_cmd(dut, OP_CONFIG, base=cfg)
    await send_cmd(dut, OP_GO)

    await wait_done(dut)

    # Read R back (full tiles), slice the real M×N region.
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


async def setup(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)


def check(got, exp, M, N, tag):
    for m in range(M):
        for n in range(N):
            assert got[m][n] == exp[m][n], \
                f"{tag}: C[{m}][{n}] got {got[m][n]} expected {exp[m][n]}"


async def do_case(dut, A, W, bias, M, K, N, shift, act_sel, leak_shift, tag):
    got = await run_gemm(dut, A, W, bias, M, K, N, shift, act_sel, leak_shift)
    exp = ref_gemm(A, W, bias, M, K, N, shift, act_sel, leak_shift)
    check(got, exp, M, N, tag)
    dut._log.info(f"{tag}: {M}x{K}x{N} bit-exact OK")


# ── tests (simplest first) ─────────────────────────────────────────────────

@cocotb.test()
async def test_single_identity(dut):
    """One tile, W = I, bypass, no shift/bias  ->  C == A. Smoke test of the whole path."""
    await setup(dut)
    M = K = N = TILE
    A = [[random.Random(1).randint(-128, 127) for _ in range(K)] for _ in range(M)]
    W = [[1 if r == c else 0 for c in range(N)] for r in range(K)]
    bias = [0] * N
    await do_case(dut, A, W, bias, M, K, N, 0, ACT_BYPASS, 0, "identity")


@cocotb.test()
async def test_single_matmul(dut):
    """One tile, real signed matmul, shift=6 bypass, nonzero bias."""
    await setup(dut)
    rng = random.Random(0xA5)
    M = K = N = TILE
    A = [[rng.randint(-128, 127) for _ in range(K)] for _ in range(M)]
    W = [[rng.randint(-128, 127) for _ in range(N)] for _ in range(K)]
    bias = [rng.randint(-4096, 4096) for _ in range(N)]
    await do_case(dut, A, W, bias, M, K, N, 6, ACT_BYPASS, 0, "matmul")


@cocotb.test()
async def test_single_relu(dut):
    """One tile, ReLU activation, shift=5."""
    await setup(dut)
    rng = random.Random(0x1234)
    M = K = N = TILE
    A = [[rng.randint(-128, 127) for _ in range(K)] for _ in range(M)]
    W = [[rng.randint(-128, 127) for _ in range(N)] for _ in range(K)]
    bias = [rng.randint(-2000, 2000) for _ in range(N)]
    await do_case(dut, A, W, bias, M, K, N, 5, ACT_RELU, 0, "relu")


@cocotb.test()
async def test_multi_tile(dut):
    """K-reduction + 2-D output sweep: 32x32x32 = 2x2x2 tiles. Exercises the odometer."""
    await setup(dut)
    rng = random.Random(0xBEEF)
    M = K = N = 32
    A = [[rng.randint(-128, 127) for _ in range(K)] for _ in range(M)]
    W = [[rng.randint(-128, 127) for _ in range(N)] for _ in range(K)]
    bias = [rng.randint(-3000, 3000) for _ in range(N)]
    await do_case(dut, A, W, bias, M, K, N, 8, ACT_BYPASS, 0, "multi_tile")


@cocotb.test()
async def test_partial(dut):
    """Non-multiple-of-16 dims: 20x40x12. Host zero-pads K; writes full tiles, slices on read."""
    await setup(dut)
    rng = random.Random(0x0FAD)
    M, K, N = 20, 40, 12
    A = [[rng.randint(-128, 127) for _ in range(K)] for _ in range(M)]
    W = [[rng.randint(-128, 127) for _ in range(N)] for _ in range(K)]
    bias = [rng.randint(-3000, 3000) for _ in range(N)]
    await do_case(dut, A, W, bias, M, K, N, 7, ACT_RELU, 0, "partial")


@cocotb.test()
async def test_gelu(dut):
    """Inline GELU on the GEMM drain: shift lands the accumulator in gelu's active
    region so the curve (incl. the negative dip) is exercised, bit-exact vs the ROM."""
    await setup(dut)
    rng = random.Random(0x6E1)
    M, K, N = 16, 16, 16
    A = [[rng.randint(-64, 64) for _ in range(K)] for _ in range(M)]
    W = [[rng.randint(-64, 64) for _ in range(N)] for _ in range(K)]
    bias = [rng.randint(-3000, 3000) for _ in range(N)]
    await do_case(dut, A, W, bias, M, K, N, 8, ACT_GELU, 0, "gelu")


@cocotb.test()
async def test_random_sweep(dut):
    """Seeded random GEMMs over assorted shapes / activations / shifts."""
    await setup(dut)
    rng = random.Random(0xC0FFEE)
    shapes = [(16, 16, 16), (16, 32, 16), (24, 16, 24), (8, 24, 40)]
    sels = [ACT_BYPASS, ACT_RELU, ACT_LEAKY, ACT_GELU]
    for idx, (M, K, N) in enumerate(shapes):
        await reset(dut)
        A = [[rng.randint(-128, 127) for _ in range(K)] for _ in range(M)]
        W = [[rng.randint(-128, 127) for _ in range(N)] for _ in range(K)]
        bias = [rng.randint(-5000, 5000) for _ in range(N)]
        shift = rng.randint(0, 10)
        sel = sels[idx % len(sels)]
        leak = rng.randint(1, 4)
        await do_case(dut, A, W, bias, M, K, N, shift, sel, leak, f"rand{idx}")
    dut._log.info("random sweep OK")
