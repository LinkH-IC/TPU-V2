// =============================================================================
// vpu.sv  (Vector Processing Unit — the non-linear domain)
// One engine for everything the systolic array can't do. Two modes:
//
//   INLINE (combinational, on the GEMM drain): act_sel picks relu / leaky /
//     bypass / gelu per drained accumulator element. No FSM involved.
//
//   STANDALONE (the FSM, dispatched by cmd_proc via GO(func3) = op):
//     1     softmax    : per row  q = round(127·e/Σe),  e = 2^((x−max)·scale)
//     2..5  add/sub/mul/mask : lanewise INT8, sat8
//     6..8  promote/accumulate/requant : optimizer — INT32 master weights in
//           region 2;  W32=W8<<sh  /  W32−=(dW·lr)>>sh  /  W8=sat8(W32>>sh)
//     9     layernorm  : per row  y = sat8((γ·(x−μ)/σ)>>sh + β)
//
//   Memory: ONE port, region-routed by `mem_region` in top (0=weight 1=act
//   2=master). Layouts (nNt = ceil(N/16)):
//     INT8  tiled : tile(i,j) at base+(i·nNt+j)·16, row lr = 1 word (16 lanes)
//     INT32 tiled : tile(i,j) at base+(i·nNt+j)·64, row lr = 4 subwords (4 lanes)
//     γ/β vectors : linear, word j = elements 16j..16j+15; β follows γ at +nNt
// =============================================================================

`default_nettype none

module vpu #(
    parameter int ROWS     = 16,               // inline lanes (MXU drain column)
    parameter int DATA_W   = 8,                // INT8 output
    parameter int ACC_W    = 32,               // INT32 score / exponent magnitude
    parameter int WORD_W   = 128,              // memory word
    parameter int ADDR_W   = 13,               // word address (DEPTH=8192)
    parameter int DIM_W    = 12,               // M/N dimension width
    parameter int MAX_N    = 512,              // longest row supported (row-buffer depth)
    parameter int SCALE_W  = 16,               // exp scale-constant width
    parameter int V_W      = 13,               // exp arg (Q5.8)
    parameter int E_W      = 16,               // exp value (Q1.15)
    parameter int SCALE_SH   = 16,             // v = (max−x)·scale >> SCALE_SH
    parameter int GELU_FBITS = 4               // inline GELU LUT fixed-point scale (K = 2^F)
)(
    input  logic                clk,
    input  logic                rst_n,

    // ── Control / descriptor (from cmd_proc dispatch) ────────────
    input  logic                start,         // pulse: run the op
    input  logic [3:0]          op,            // 1=softmax 2..5=elementwise 6..8=optimizer 9=layernorm
    input  logic [DIM_W-1:0]    M,             // rows
    input  logic [DIM_W-1:0]    N,             // row length / cols (≤ MAX_N)
    input  logic [ADDR_W-1:0]   src_base,      // operand-1 base (word addr)
    input  logic [ADDR_W-1:0]   src2_base,     // operand-2 base (EW b / optimizer W8·dW / LN γ)
    input  logic [ADDR_W-1:0]   dst_base,      // result base
    input  logic [SCALE_W-1:0]  scale,         // per-op constant: softmax exp scale / sh / {lr,sh}
    output logic                busy,
    output logic                done,

    // ── Inline activation (MXU drain) — combinational ────────────
    input  logic [ROWS-1:0][ACC_W-1:0]  act_in,    // = biased_reg (accumulator drain)
    input  logic [4:0]                  act_shift, // requant right-shift
    input  logic [1:0]                  act_sel,   // 0=relu 1=leaky 2=bypass 3=gelu
    input  logic [2:0]                  act_leak,  // leaky-relu negative-path shift
    output logic [ROWS-1:0][DATA_W-1:0] act_out,

    // ── Data-memory port (Port B in top; bench models it) ────────
    output logic [1:0]          mem_region,    // 0=weight(R0) 1=act(R1) 2=master(R2); top routes
    output logic [ADDR_W-1:0]   mem_addr,
    output logic                mem_en,
    output logic                mem_we,
    output logic [WORD_W-1:0]   mem_wdata,
    input  logic [WORD_W-1:0]   mem_rdata      // valid 1 cycle after a read is issued
);

    // ── Derived ──────────────────────────────────────────────────
    localparam int LANES = WORD_W / ACC_W;          // INT32 lanes per word (=4)
    localparam int MAXW  = MAX_N / LANES;           // row-buffer depth (words/row max)
    localparam int IDXW  = $clog2(MAXW);            // row-buffer index width
    localparam int WC_W  = $clog2(MAX_N + 1);       // word-stream counter width
    localparam int NT_W  = DIM_W - 3;               // nNt width (ceil(N/16))
    localparam int WADDR = 24;                      // wide intermediate for address math
    localparam int WW_W  = WORD_W - DATA_W*LANES;   // assembled-word reg (subs 0..2) = 96b
    localparam int EP_W  = ACC_W + SCALE_W;         // exp product d·scale = 48b
    localparam int P1_W  = E_W + ACC_W;             // norm product e·recip = 48b
    localparam logic [ACC_W-1:0] NEG_INF = {1'b1, {(ACC_W-1){1'b0}}}; // INT32 min

    // ── Op codes (= GO(func3) target from cmd_proc) ──────────────
    localparam logic [3:0] OP_SOFTMAX = 4'd1,
                           OP_SUB     = 4'd3,   // 2=add is the default elementwise op
                           OP_MUL     = 4'd4,   // Hadamard, requantized: sat8((a·b) >>> scale[4:0])
                           OP_MASK    = 4'd5,   // relu' gate: (op2>0)?op1:0 — one-pass backward
                           OP_PROMOTE = 4'd6,   // W32 = sext(W8) << sh   (region0 → region2, init master)
                           OP_ACCUM   = 4'd7,   // W32 -= (dW·lr) >> sh   (master r2, dW r1 → r2)
                           OP_REQUANT = 4'd8,   // W8  = sat8(W32 >> sh)  (region2 → region0, refresh)
                           OP_LNORM   = 4'd9;   // y = sat8((γ·(x−μ)/σ) >> sh + β), per row (INT8→INT8)

    // ── Memory regions (mem_region output; top routes to wmem/amem/mmem) ─
    localparam logic [1:0] R_W0 = 2'd0,          // region 0 = weight mem (INT8)
                           R_A1 = 2'd1,          // region 1 = act/result mem
                           R_M2 = 2'd2;          // region 2 = master-weight mem (INT32)

    // ── Optimizer scale-field split: sh = shift (all 3), lr = accumulate multiplier ─
    localparam int WACC = ACC_W + 12;            // 44b accumulate intermediate (dW·lr headroom + sat)
    localparam signed [WACC-1:0] I32_MAX =  44'sd2147483647;
    localparam signed [WACC-1:0] I32_MIN = -44'sd2147483648;
    // scale is static for the whole op but lives in cmd_proc — latched locally
    // at start so the barrel shifters / DSP operands don't eat a cross-module
    // route (5th synth: scale_r → q_acc was the −2.75 peak).
    logic [SCALE_W-1:0] scale_q;
    wire [4:0]  cvt_sh = scale_q[4:0];           // INT8↔INT32 shift (promote «, requant/accum »)
    wire [10:0] acc_lr = scale_q[15:5];          // accumulate fixed-point lr numerator

    // ── FSM ──────────────────────────────────────────────────────
    typedef enum logic [4:0] {
        IDLE,
        ROWSET,     // softmax: row setup (INT32-tiled bases, clear max/Σ)
        MAX_P,      // softmax: stream row, running max     (4 beats/word, staged tree fold)
        EXP_P,      // softmax: e = 2^((x−max)·scale) → ebuf, Σe  (6 beats/word)
        RECIP_P,    // softmax: recip = 2^32/Σ  (reciprocal, once per row)
        NORM_P,     // softmax: q = sat8(round(127·e·recip/2^32)) → write (4 beats/sub)
        ROWEND,     // softmax: row++ / done
        EW_A,       // elementwise: drive operand-1 read
        EW_B,       // elementwise: capture op1, drive operand-2 read
        EW_C,       // elementwise: capture op2
        EW_D,       // elementwise: stage the op math (pre-requant 16b)
        EW_E,       // elementwise: requant + sat8 → staged write data
        EW_W,       // elementwise: write result (register-direct)
        P_RD,       // promote:   read region0 INT8 word    (2 beats)
        P_WR,       // promote:   register «sh / write master word (2 beats/sub)
        Q_RD,       // requant:   read region2 master word  (4 beats/sub, staged »sh)
        Q_WR,       // requant:   write region0 INT8 word
        AC_A,       // accumulate: drive master read (region2)
        AC_B,       // accumulate: capture master, drive dW read (region1)
        AC_W,       // accumulate: capture dW / ×lr / register / write (4 beats)
        LN_RN,      // layernorm: recipn = 2^17/N via reciprocal (once per op)
        LN_RS,      // layernorm: row setup (INT8-tiled bases, clear accums)
        LN_MEAN,    // layernorm: stream x, Σx              (3 beats/word)
        LN_MU,      // layernorm: μ = (Σx·recipn) >>> 9  (Q8, 2 beats — staged multiply)
        LN_VAR,     // layernorm: stream x, Σ(x·2^8−μ)²     (2 + 2 beats/sub)
        LN_SG,      // layernorm: σ² = (Σd²·recipn) >> 17 + ε  (Q16, 2 beats — staged multiply)
        LN_RQ,      // layernorm: rs = 2^24/√σ² via rsqrt (once per row)
        LN_NORM,    // layernorm: y = sat8((n·γ)>>sh + β) → write (4 + 5·4 + 1 beats/word)
        LN_RE       // layernorm: row++ / done
    } state_t;
    state_t state;

    // ── Row-loop registers (softmax; row/bases/wc/beat shared with LN) ──
    logic [DIM_W-1:0]        row;               // current row r
    logic [NT_W-1:0]         nNt;               // ceil(N/16)
    logic [ADDR_W-1:0]       rd_base, wr_base;  // per-row memory bases
    logic [WC_W-1:0]         wc;                // MAX/EXP word counter
    logic [4:0]              beat;              // per-word beat counter (pipeline stages)
    logic [WC_W-1:0]         nc;                // NORM word counter
    logic signed [ACC_W-1:0] runmax;            // running row max
    logic [ACC_W-1:0]        sumacc;            // Σ e_j
    logic [ACC_W-1:0]        recip_r;           // 2^32 / Σ
    logic                    rstarted;          // reciprocal kicked off
    logic [WW_W-1:0]         wword;             // NORM output word: subs 0..2
    logic                    done_r;

    // ── Elementwise + optimizer — linear INT8-word stream (layout-agnostic) ──
    logic [ADDR_W-1:0]       ewc;               // word counter (w8 = INT8 word index)
    logic [ADDR_W-1:0]       ew_words;          // total INT8 words = ceil(M/16)·ceil(N/16)·16
    logic [WORD_W-1:0]       a_reg;             // captured operand-1 word (EW)
    logic [WORD_W-1:0]       b_reg;             // captured operand-2 word (EW) / dW word (accum)
    logic [WORD_W-1:0]       ew_res;            // lanewise elementwise result (combinational)
    // Write-data staging (S28c): mem_wdata fans to ALL THREE memories' DIADI
    // pins (~10 ns of route in the 3rd synth) — deep computes must not ride
    // that net combinationally. Registered here, Vivado replicates per region.
    logic [WORD_W-1:0]       ew_q;              // staged ew_res
    logic [WORD_W-1:0]       acw_q;             // staged accum_word
    logic [WORD_W-1:0]       pw_q;              // staged promote_word
    // 7th synth: mul+barrel+sat in one beat (a_reg→ew_q) and the requant
    // barrel (scale_q→q_acc) were the last −2.8 families — both split:
    logic signed [2*DATA_W-1:0] ewi_q [WORD_W/DATA_W];  // EW op math, pre-requant (16b)
    logic signed [ACC_W-1:0]    rq_q [LANES];           // requant: staged m >>> sh

    // ── Optimizer (promote/accumulate/requant) — INT8 word ewc × 4 INT32 subwords ──
    logic [1:0]              sub;               // subword 0..3 (4 INT32 masters per INT8 word)
    logic [WORD_W-1:0]       cvt_word;          // captured word (softmax x / promote / LN x / accum master)
    logic [WORD_W-1:0]       q_acc;             // requant/LN: assembled 16-INT8 output word
    // master/dW subword offset: keystone layout (i·nNt+j)·64 + lr·4 + sub = 4·w8 + sub
    wire [ADDR_W-1:0] r2_off = ADDR_W'((WADDR'(ewc) << 2) + WADDR'(sub));

    // ── LayerNorm — per-row 3-pass reduction (mean / var / normalize) ──
    logic [17:0]             recipn;            // 2^17/N  (reciprocal of N<<15, once per op)
    logic signed [17:0]      sumx;              // Σx  (|Σ| ≤ 512·128 = 2^16)
    logic signed [17:0]      mu;                // row mean, Q8  (|μ| ≤ 2^15)
    logic [41:0]             sumsq;             // Σd²  (d Q8 ≤ 2^16 → d² ≤ 2^32, ×512)
    logic [31:0]             sig2;              // σ² + ε, Q16
    logic [24:0]             rs_r;              // 2^24/√σ²  (= 2^16/σ)
    logic [WORD_W-1:0]       gword, bword;      // captured γ / β words (region 0)

    // ── Pipeline registers (one small array per cut in the old cones) ──
    logic                    vld_q [LANES];     // lane validity, captured with the x word
    logic [ACC_W-1:0]        d_q [LANES];       // softmax: max − x (≥ 0 for valid lanes)
    logic [EP_W-1:0]         eprod_q [LANES];   // softmax: d·scale (32×16 → DSP)
    logic [E_W-1:0]          e_q [LANES];       // softmax: masked exp_lut output
    logic [E_W-1:0]          enorm_q [LANES];   // norm: e_j read out of the row buffer
    logic [P1_W-1:0]         p1_q [LANES];      // norm: e·recip (16×32 → DSP)
    logic [DATA_W-1:0]       qb_q [LANES];      // norm: quantized softmax byte
    logic signed [WACC-1:0]  ap_q [LANES];      // accumulate: dW·lr (32×11 → DSP)
    logic signed [17:0]      ds4_q [LANES];     // LN var: d = x·2^8 − μ (masked at load)
    logic signed [17:0]      lnds_q [LANES];    // LN norm: d = x·2^8 − μ (mask at output)
    logic [15:0]             lnvld_q;           // LN: INT8-word lane validity, captured with x
                                                //   (5th synth: the comb wc→compare cone into
                                                //    the mean adder tree was a −2.7 family)
    logic signed [43:0]      np_q [LANES];      // LN norm: d·rs (18×25 → DSP), Q24
    logic signed [23:0]      gy_q [LANES];      // LN norm: n_q6·γ (16×8 → DSP)
    logic signed [36:0]      mprod_q;           // LN mean: Σx·recipn (19×18 → DSP), staged
    logic [59:0]             vprod_q;           // LN var: Σd²·recipn (42×18 → DSP), staged
    logic signed [31:0]      yr_q [LANES];      // LN norm: (n·γ + rnd) >>> sh, staged
                                                //   (6th synth: rnd-shifter → DSP → shifter →
                                                //    sat all rode one beat into q_acc, −2.7)
    logic signed [31:0]      rnd32_q;           // rounding constant — static per op (f(sh) only)
    logic [2:0]              ph;                // LN norm sub-phase counter (5 phases/sub)

    // ── Row buffer: MAXW slots × one packed 4-lane e word (single write port,
    //   single read port → infers distributed LUTRAM; the old [LANES][MAXW]
    //   FF array cost ~2.5k LUTs of 128:1 read-mux forest) ──
    logic [E_W*LANES-1:0]    ebuf [MAXW];

    // ── Words per row (INT32 read = nNt·4, INT8 write = nNt) ─────
    wire [WC_W-1:0] nwords = WC_W'(nNt) << 2;       // = nNt·4

    // ── Elementwise tile counts (ceil(M/16), ceil(N/16)) ─────────
    wire [DIM_W-1:0] ew_nMt = (M + DIM_W'(15)) >> 4;
    wire [DIM_W-1:0] ew_nNt = (N + DIM_W'(15)) >> 4;

    // ── Read / write addresses (wide intermediates, truncate once) ─
    wire [WC_W-1:0]   tilew  = wc >> 2;             // which N-tile in the row
    wire [1:0]        subw   = wc[1:0];             // subword 0..3
    wire [ADDR_W-1:0] rd_addr = ADDR_W'(WADDR'(rd_base) + (WADDR'(tilew) << 6) + WADDR'(subw));
    wire [ADDR_W-1:0] wr_addr = ADDR_W'(WADDR'(wr_base) + (WADDR'(nc >> 2) << 4));

    // ── Lane unpack (from the CAPTURED word) + per-lane validity ──
    logic signed [ACC_W-1:0] xlane [LANES];
    logic                    vlane [LANES];
    always_comb begin
        for (int l = 0; l < LANES; l++) begin
            xlane[l] = $signed(cvt_word[l*ACC_W +: ACC_W]);
            vlane[l] = (WC_W'(wc << 2) + WC_W'(l)) < WC_W'(N);
        end
    end

    // ── Valid-masked max of the captured word — staged tree fold ─
    //   (Replaces the old per-lane conditional NBA loop, which was simulator-
    //   dependent: the LRM makes it "last lane to beat the OLD max wins" —
    //   wrong, and what synthesis builds; Verilator happened to optimize it
    //   into the intended fold. The first fix, a single-beat priority chain,
    //   synthesized as 4 chained 32-bit compares — 24 logic levels, WNS −5.4
    //   at 69% route share. Staged tree instead: beat 2 registers the two
    //   pairwise maxima (invalid lanes → NEG_INF, so they can never win),
    //   beat 3 folds max(m01,m23) into runmax — every cycle of the recurrence
    //   is now ≤ 2 chained compares.)
    logic signed [ACC_W-1:0] mlane [LANES];        // capture-masked lanes
    always_comb begin
        for (int l = 0; l < LANES; l++)
            mlane[l] = vld_q[l] ? xlane[l] : $signed(NEG_INF);
    end
    logic signed [ACC_W-1:0] m01_q, m23_q;         // staged pairwise maxima
    wire signed  [ACC_W-1:0] w4 = (m01_q > m23_q) ? m01_q : m23_q;

    // ── exp datapath: arg = sat((max−x)·scale >> SCALE_SH) to Q5.8, from eprod_q ─
    logic [V_W-1:0]  earg [LANES];
    logic [E_W-1:0]  ecomb [LANES];                // exp_lut outputs
    logic [E_W-1:0]  estore [LANES];               // 0 for invalid lanes
    always_comb begin
        for (int l = 0; l < LANES; l++) begin
            logic [ACC_W-1:0] vfull;
            vfull   = ACC_W'(eprod_q[l] >> SCALE_SH);
            earg[l] = (vfull > ACC_W'(13'h1FFF)) ? 13'h1FFF : vfull[V_W-1:0]; // clamp → underflow
            estore[l] = vld_q[l] ? ecomb[l] : '0;
        end
    end

    genvar gl;
    generate
        for (gl = 0; gl < LANES; gl++) begin : g_exp
            exp_lut #(.V_W(V_W), .E_W(E_W)) u_exp (.v(earg[gl]), .e(ecomb[gl]));
        end
    endgenerate

    // ── reciprocal (once per row) ────────────────────────────────
    logic              recip_start;
    logic [ACC_W-1:0]  recip_q;
    logic              recip_done;
    /* verilator lint_off PINCONNECTEMPTY */
    reciprocal #(.SUM_W(ACC_W), .RECIP_W(ACC_W)) u_recip (
        .clk   (clk),
        .rst_n (rst_n),
        .start (recip_start),
        .sum   (sumacc),
        .recip (recip_q),
        .busy  (),
        .done  (recip_done)
    );
    /* verilator lint_on PINCONNECTEMPTY */
    assign recip_start = ((state == RECIP_P) || (state == LN_RN)) && !rstarted;  // softmax 1/Σ, LN 1/N

    // ── rsqrt (once per LayerNorm row): rs = 2^24/√σ² ─────────────
    logic              rsq_start;
    logic [24:0]       rsq_q;
    logic              rsq_done;
    /* verilator lint_off PINCONNECTEMPTY */
    rsqrt u_rsqrt (
        .clk   (clk),
        .rst_n (rst_n),
        .start (rsq_start),
        .d     (sig2),
        .rs    (rsq_q),
        .busy  (),
        .done  (rsq_done)
    );
    /* verilator lint_on PINCONNECTEMPTY */
    assign rsq_start = (state == LN_RQ) && !rstarted;

    // ── NORM quantize: q = sat8(round(127·e·recip / 2^32)), from p1_q ───
    //   127·p1 = (p1<<7) − p1 — same integer as the old 127·e·recip in one go.
    logic [DATA_W-1:0]   qb_c [LANES];
    always_comb begin
        for (int l = 0; l < LANES; l++) begin
            logic [63:0] num;
            logic [8:0]  q;
            num = (64'(p1_q[l]) << 7) - 64'(p1_q[l]) + 64'h8000_0000;  // round, /2^32 below
            q   = 9'(num >> 32);
            qb_c[l] = (q > 9'd127) ? 8'd127 : q[DATA_W-1:0];           // ≥0, sat to 127
        end
    end
    wire [DATA_W*LANES-1:0] chunk     = {qb_q[3], qb_q[2], qb_q[1], qb_q[0]};
    wire [WORD_W-1:0]       norm_word = {chunk, wword};      // sub3 (now) ++ subs 0..2

    // ── Elementwise lanewise op (sat8): op1 = a_reg, op2 = b_reg (both captured) ──
    //   2=add  3=sub  4=mul (>>> scale[4:0])  5=mask (relu' gate: op2>0 ? op1 : 0)
    localparam int                 EWW    = 2*DATA_W;           // 16b op width (room for av·bv)
    localparam signed [EWW-1:0]    EW_MAX =  16'sd127;          // INT8 saturation bounds
    localparam signed [EWW-1:0]    EW_MIN = -16'sd128;
    wire [4:0] ew_shift = scale_q[4:0];                         // mul requant right-shift
    // stage 1: the op math, pre-requant (mul product |·| ≤ 16384 fits 16b)
    logic signed [EWW-1:0] ewi_c [WORD_W/DATA_W];
    always_comb begin
        for (int l = 0; l < WORD_W/DATA_W; l++) begin
            logic signed [EWW-1:0] av, bv;                      // 16b sign-extended operands
            av = EWW'($signed(a_reg[l*DATA_W +: DATA_W]));
            bv = EWW'($signed(b_reg[l*DATA_W +: DATA_W]));
            case (op)
                OP_SUB:  ewi_c[l] = av - bv;
                OP_MUL:  ewi_c[l] = av * bv;
                OP_MASK: ewi_c[l] = (bv > 0) ? av : EWW'(0);    // relu'(op2)·op1, gate strictly >0
                default: ewi_c[l] = av + bv;                    // OP_ADD (2)
            endcase
        end
    end
    // stage 2: requant (mul only) + sat8, from the staged intermediates
    always_comb begin
        for (int l = 0; l < WORD_W/DATA_W; l++) begin
            logic signed [EWW-1:0] r;
            r = (op == OP_MUL) ? (ewi_q[l] >>> ew_shift) : ewi_q[l];  // signed → arithmetic shift
            ew_res[l*DATA_W +: DATA_W] = (r > EW_MAX) ? 8'h7F :
                                         (r < EW_MIN) ? 8'h80 : r[DATA_W-1:0];
        end
    end

    // ── promote: 4 INT8 (cvt_word bytes for subword `sub`) → 4 INT32 masters «sh ─
    logic [WORD_W-1:0] promote_word;
    always_comb begin
        promote_word = '0;
        for (int l = 0; l < LANES; l++) begin
            logic signed [DATA_W-1:0] pb;
            pb = $signed(cvt_word[(int'(sub)*LANES + l)*DATA_W +: DATA_W]);
            promote_word[l*ACC_W +: ACC_W] = ACC_W'($signed(pb)) << cvt_sh;
        end
    end

    // ── requant: sat8 of the staged m »sh (barrel runs its own beat into rq_q) ──
    logic [DATA_W*LANES-1:0] requant_chunk;
    always_comb begin
        requant_chunk = '0;
        for (int l = 0; l < LANES; l++)
            requant_chunk[l*DATA_W +: DATA_W] = (rq_q[l] > A_MAX) ? DATA_W'(A_MAX) :
                                                (rq_q[l] < A_MIN) ? DATA_W'(A_MIN) : DATA_W'(rq_q[l]);
    end

    // ── accumulate: master (cvt_word) − ap_q »sh, INT32-saturated ─
    //   ap_q = dW·lr staged one beat earlier (the 32×11 multiply → DSP).
    logic [WORD_W-1:0] accum_word;
    always_comb begin
        accum_word = '0;
        for (int l = 0; l < LANES; l++) begin
            logic signed [ACC_W-1:0] am;
            logic signed [WACC-1:0]  anv;
            am  = $signed(cvt_word[l*ACC_W +: ACC_W]);          // master
            anv = WACC'(am) - (ap_q[l] >>> cvt_sh);             // master − delta
            accum_word[l*ACC_W +: ACC_W] = (anv > I32_MAX) ? ACC_W'(I32_MAX) :
                                           (anv < I32_MIN) ? ACC_W'(I32_MIN) : anv[ACC_W-1:0];
        end
    end

    // ── LayerNorm datapaths (comb) — x is INT8 tiled, γ/β linear vectors ──
    //   LN word addresses: x/y word j at {rd,wr}_base + j·16 (tile stride along j);
    //   γ at src2_base + j, β right after γ at src2_base + nNt + j (locked layout).
    wire [ADDR_W-1:0] ln_x_addr = ADDR_W'(WADDR'(rd_base)   + (WADDR'(wc) << 4));
    wire [ADDR_W-1:0] ln_y_addr = ADDR_W'(WADDR'(wr_base)   + (WADDR'(wc) << 4));
    wire [ADDR_W-1:0] ln_g_addr = ADDR_W'(WADDR'(src2_base) + WADDR'(wc));
    wire [ADDR_W-1:0] ln_b_addr = ADDR_W'(WADDR'(src2_base) + WADDR'(nNt) + WADDR'(wc));

    wire [DIM_W-1:0] ln_el0 = DIM_W'(wc) << 4;              // first element index of word wc

    // MEAN: 16-lane masked sum of the captured x word (mask = registered lnvld_q)
    logic signed [17:0] mean16;
    always_comb begin
        mean16 = '0;
        for (int l = 0; l < 16; l++)
            if (lnvld_q[l])
                mean16 = mean16 + 18'($signed(cvt_word[l*DATA_W +: DATA_W]));
    end

    // VAR beat map: 0 drive x / 1 capture / then per sub: even = stage d, odd = Σd²
    wire [1:0] var_sub = 2'((beat - 5'd2) >> 1);            // beats 2/4/6/8 → subs 0..3
    // NORM: beats 0..3 = x/γ/β reads; beat 4 = the (sub, ph) 5-phase engine; 5 = write
    wire [1:0] ds_sel  = (state == LN_VAR) ? var_sub : sub;

    // d = x·2^8 − μ for the 4 lanes of subword ds_sel (shared by VAR and NORM)
    logic signed [17:0] ds_c [LANES];
    always_comb begin
        for (int l = 0; l < LANES; l++)
            ds_c[l] = (18'($signed(cvt_word[(int'(ds_sel)*4 + l)*DATA_W +: DATA_W])) <<< 8) - mu;
    end

    // VAR squares (from the staged d; masked lanes were loaded as 0 → contribute 0)
    logic [35:0] dsq [LANES];
    always_comb begin
        for (int l = 0; l < LANES; l++)
            dsq[l] = $unsigned(36'(ds4_q[l] * ds4_q[l]));
    end

    // NORM: n_q6 = clamp16((d·rs) >>> 18) from the staged product np_q
    logic signed [15:0] nq_c [LANES];
    always_comb begin
        for (int l = 0; l < LANES; l++) begin
            logic signed [25:0] n26;
            n26 = 26'(np_q[l] >>> 18);                      // → Q6 normalized
            nq_c[l] = (n26 > 26'sd32767)  ? 16'sd32767  :
                      (n26 < -26'sd32768) ? -16'sd32768 : 16'(n26);
        end
    end

    // NORM output: y = sat8(yr + β) from the staged rounded-requant yr_q
    logic [DATA_W*LANES-1:0] ln_chunk;
    always_comb begin
        for (int l = 0; l < LANES; l++) begin
            logic signed [31:0] ys;
            ys = yr_q[l] + 32'($signed(bword[(int'(sub)*4 + l)*DATA_W +: DATA_W]));
            ln_chunk[l*DATA_W +: DATA_W] =
                (!lnvld_q[int'(sub)*4 + l]) ? 8'h00 :                    // padded lanes → 0
                (ys > 32'sd127)  ? 8'h7F :
                (ys < -32'sd128) ? 8'h80 : ys[DATA_W-1:0];
        end
    end

    // ── FSM ──────────────────────────────────────────────────────
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state    <= IDLE;
            row      <= '0;
            nNt      <= '0;
            rd_base  <= '0;
            wr_base  <= '0;
            wc       <= '0;
            beat     <= '0;
            nc       <= '0;
            runmax   <= $signed(NEG_INF);
            sumacc   <= '0;
            recip_r  <= '0;
            rstarted <= 1'b0;
            wword    <= '0;
            done_r   <= 1'b0;
            sub      <= '0;
            cvt_word <= '0;
            q_acc    <= '0;
            recipn   <= '0;
            sumx     <= '0;
            mu       <= '0;
            sumsq    <= '0;
            sig2     <= '0;
            rs_r     <= '0;
            gword    <= '0;
            bword    <= '0;
            ph       <= '0;
        end else begin
            case (state)
                IDLE: begin
                    done_r <= 1'b0;                               // done is a 1-cycle pulse
                    if (start) begin
                        nNt      <= NT_W'((N + DIM_W'(15)) >> 4);  // ceil(N/16)
                        scale_q  <= scale;                         // descriptor latch (static per op)
                        rnd32_q  <= (scale[4:0] == 5'd0) ? 32'sd0  // rounding constant: f(sh) only,
                                  : (32'sd1 <<< (scale[4:0] - 5'd1)); // shifter runs ONCE per op
                        ewc      <= '0;
                        sub      <= '0;
                        beat     <= '0;
                        ew_words <= ADDR_W'((WADDR'(ew_nMt) * WADDR'(ew_nNt)) << 4); // INT8 words
                        case (op)
                            OP_SOFTMAX: begin row <= '0; state <= ROWSET; end
                            OP_PROMOTE: state <= P_RD;
                            OP_ACCUM:   state <= AC_A;
                            OP_REQUANT: state <= Q_RD;
                            OP_LNORM: begin
                                sumacc   <= ACC_W'(N) << 15;   // recip(N·2^15) = 2^17/N (in contract)
                                rstarted <= 1'b0;
                                state    <= LN_RN;
                            end
                            default:                    state <= EW_A;   // add/sub/mul/mask
                        endcase
                    end
                end

                ROWSET: begin
                    // per-row bases: tile(i,j) at (i·nNt+j)·{64,16}; row lr offset
                    rd_base <= ADDR_W'(WADDR'(src_base)
                             + ((WADDR'(row >> 4) * WADDR'(nNt)) << 6)   // i·nNt·64
                             + (WADDR'(row[3:0]) << 2));                 // lr·4
                    wr_base <= ADDR_W'(WADDR'(dst_base)
                             + ((WADDR'(row >> 4) * WADDR'(nNt)) << 4)   // i·nNt·16
                             + WADDR'(row[3:0]));                        // lr
                    runmax  <= $signed(NEG_INF);
                    sumacc  <= '0;
                    wc      <= '0;
                    beat    <= '0;
                    state   <= MAX_P;
                end

                MAX_P: begin
                    case (beat)
                        5'd0: beat <= 5'd1;                       // read address driven (comb)
                        5'd1: begin
                            cvt_word <= mem_rdata;                // x word captured
                            for (int l = 0; l < LANES; l++)
                                vld_q[l] <= vlane[l];
                            beat <= 5'd2;
                        end
                        5'd2: begin                               // pairwise maxima (1 compare each)
                            m01_q <= (mlane[0] > mlane[1]) ? mlane[0] : mlane[1];
                            m23_q <= (mlane[2] > mlane[3]) ? mlane[2] : mlane[3];
                            beat  <= 5'd3;
                        end
                        default: begin                            // fold into runmax (2 compares)
                            if (w4 > runmax)
                                runmax <= w4;
                            beat <= 5'd0;
                            if (wc == nwords - WC_W'(1)) begin
                                wc    <= '0;
                                state <= EXP_P;
                            end else begin
                                wc <= wc + WC_W'(1);
                            end
                        end
                    endcase
                end

                EXP_P: begin
                    case (beat)
                        5'd0: beat <= 5'd1;                       // read address driven (comb)
                        5'd1: begin
                            cvt_word <= mem_rdata;                // x word captured
                            for (int l = 0; l < LANES; l++)
                                vld_q[l] <= vlane[l];
                            beat <= 5'd2;
                        end
                        5'd2: begin                               // d = max − x (≥ 0 for valid lanes)
                            for (int l = 0; l < LANES; l++)
                                d_q[l] <= ACC_W'(runmax - xlane[l]);
                            beat <= 5'd3;
                        end
                        5'd3: begin                               // ×scale — isolated multiply → DSP
                            for (int l = 0; l < LANES; l++)
                                eprod_q[l] <= EP_W'(d_q[l]) * EP_W'(scale_q);
                            beat <= 5'd4;
                        end
                        5'd4: begin                               // clamp → exp LUT → lane mask
                            for (int l = 0; l < LANES; l++)
                                e_q[l] <= estore[l];
                            beat <= 5'd5;
                        end
                        default: begin                            // Σe + row-buffer store
                            sumacc <= sumacc + ACC_W'(e_q[0]) + ACC_W'(e_q[1])
                                             + ACC_W'(e_q[2]) + ACC_W'(e_q[3]);
                            ebuf[wc[IDXW-1:0]] <= {e_q[3], e_q[2], e_q[1], e_q[0]};
                            beat <= 5'd0;
                            if (wc == nwords - WC_W'(1)) begin
                                rstarted <= 1'b0;
                                state    <= RECIP_P;
                            end else begin
                                wc <= wc + WC_W'(1);
                            end
                        end
                    endcase
                end

                RECIP_P: begin
                    rstarted <= 1'b1;                             // start pulses 1 cyc (comb)
                    if (recip_done) begin
                        recip_r <= recip_q;
                        nc      <= '0;
                        wword   <= '0;
                        beat    <= '0;
                        state   <= NORM_P;
                    end
                end

                NORM_P: begin
                    case (beat)
                        5'd0: begin                               // row-buffer read → register
                            for (int l = 0; l < LANES; l++)
                                enorm_q[l] <= ebuf[nc[IDXW-1:0]][l*E_W +: E_W];
                            beat <= 5'd1;
                        end
                        5'd1: begin                               // e·recip — isolated → DSP
                            for (int l = 0; l < LANES; l++)
                                p1_q[l] <= P1_W'(enorm_q[l]) * P1_W'(recip_r);
                            beat <= 5'd2;
                        end
                        5'd2: begin                               // ·127 + round + sat (adds only)
                            for (int l = 0; l < LANES; l++)
                                qb_q[l] <= qb_c[l];
                            beat <= 5'd3;
                        end
                        default: begin
                            // assemble subs 0..2 into wword; sub3 written combinationally
                            case (nc[1:0])
                                2'd0:    wword[31:0]  <= chunk;
                                2'd1:    wword[63:32] <= chunk;
                                2'd2:    wword[95:64] <= chunk;
                                default: ;                        // sub3 -> direct write below
                            endcase
                            beat <= 5'd0;
                            if (nc == nwords - WC_W'(1))
                                state <= ROWEND;
                            else
                                nc <= nc + WC_W'(1);
                        end
                    endcase
                end

                ROWEND: begin
                    if (row == M - DIM_W'(1)) begin
                        done_r <= 1'b1;
                        state  <= IDLE;
                    end else begin
                        row   <= row + DIM_W'(1);
                        state <= ROWSET;
                    end
                end

                // ── Elementwise add/sub/mul/mask (5 cyc/word: rd op1, rd op2, capture, stage, write) ──
                EW_A: state <= EW_B;                              // op1 read addr driven (comb)
                EW_B: begin
                    a_reg <= mem_rdata;                           // op1 word captured; op2 read driven (comb)
                    state <= EW_C;
                end
                EW_C: begin
                    b_reg <= mem_rdata;                           // op2 word captured
                    state <= EW_D;
                end
                EW_D: begin
                    for (int l = 0; l < WORD_W/DATA_W; l++)
                        ewi_q[l] <= ewi_c[l];                     // op math staged (mul off the shifter)
                    state <= EW_E;
                end
                EW_E: begin
                    ew_q  <= ew_res;                              // requant+sat staged off the wdata route
                    state <= EW_W;
                end
                EW_W: begin                                       // ew_q written (register-direct)
                    if (ewc == ew_words - ADDR_W'(1)) begin
                        done_r <= 1'b1;
                        state  <= IDLE;
                    end else begin
                        ewc   <= ewc + ADDR_W'(1);
                        state <= EW_A;
                    end
                end

                // ── promote: read 1 region0 word, write 4 region2 master words ──
                P_RD: begin
                    case (beat)
                        5'd0: beat <= 5'd1;                       // region0 read addr driven (comb)
                        default: begin
                            cvt_word <= mem_rdata;                // INT8 word captured
                            beat     <= 5'd0;
                            sub      <= '0;
                            state    <= P_WR;
                        end
                    endcase
                end
                P_WR: begin
                    case (beat)
                        5'd0: begin                                // stage «sh for `sub` off the route
                            pw_q <= promote_word;
                            beat <= 5'd1;
                        end
                        default: begin                             // write pw_q (register-direct)
                            beat <= 5'd0;
                            if (sub == 2'd3) begin
                                sub <= '0;
                                if (ewc == ew_words - ADDR_W'(1)) begin
                                    done_r <= 1'b1;
                                    state  <= IDLE;
                                end else begin
                                    ewc   <= ewc + ADDR_W'(1);
                                    state <= P_RD;
                                end
                            end else begin
                                sub <= sub + 2'd1;
                            end
                        end
                    endcase
                end

                // ── requant: read 4 region2 master words, write 1 region0 word ──
                Q_RD: begin
                    case (beat)
                        5'd0: beat <= 5'd1;                       // region2 read addr driven (comb)
                        5'd1: begin
                            cvt_word <= mem_rdata;                // master word captured
                            beat     <= 5'd2;
                        end
                        5'd2: begin                               // »sh — the barrel gets its own beat
                            for (int l = 0; l < LANES; l++)
                                rq_q[l] <= $signed(cvt_word[l*ACC_W +: ACC_W]) >>> cvt_sh;
                            beat <= 5'd3;
                        end
                        default: begin                            // sat8 → q_acc
                            q_acc[{sub, 5'd0} +: DATA_W*LANES] <= requant_chunk;
                            beat <= 5'd0;
                            if (sub == 2'd3) begin
                                sub   <= '0;
                                state <= Q_WR;
                            end else begin
                                sub <= sub + 2'd1;
                            end
                        end
                    endcase
                end
                Q_WR: begin                                        // write q_acc to region0 (comb)
                    if (ewc == ew_words - ADDR_W'(1)) begin
                        done_r <= 1'b1;
                        state  <= IDLE;
                    end else begin
                        ewc   <= ewc + ADDR_W'(1);
                        sub   <= '0;
                        state <= Q_RD;
                    end
                end

                // ── accumulate: read master + read dW → ×lr → write master (per subword) ──
                AC_A: state <= AC_B;                               // master read addr driven (comb)
                AC_B: begin
                    cvt_word <= mem_rdata;                         // master captured; dW read driven (comb)
                    beat     <= '0;
                    state    <= AC_W;
                end
                AC_W: begin
                    case (beat)
                        5'd0: begin
                            b_reg <= mem_rdata;                    // dW word captured
                            beat  <= 5'd1;
                        end
                        5'd1: begin                                // dW·lr — isolated multiply → DSP
                            for (int l = 0; l < LANES; l++)
                                ap_q[l] <= $signed(b_reg[l*ACC_W +: ACC_W]) * $signed({1'b0, acc_lr});
                            beat <= 5'd2;
                        end
                        5'd2: begin                                // subtract+sat staged off the route
                            acw_q <= accum_word;
                            beat  <= 5'd3;
                        end
                        default: begin                             // acw_q written (register-direct)
                            beat <= 5'd0;
                            if (sub == 2'd3) begin
                                sub <= '0;
                                if (ewc == ew_words - ADDR_W'(1)) begin
                                    done_r <= 1'b1;
                                    state  <= IDLE;
                                end else begin
                                    ewc   <= ewc + ADDR_W'(1);
                                    state <= AC_A;
                                end
                            end else begin
                                sub   <= sub + 2'd1;
                                state <= AC_A;
                            end
                        end
                    endcase
                end

                // ── LayerNorm: 1/N once, then per row MEAN → VAR → RSQRT → NORM ──
                LN_RN: begin
                    rstarted <= 1'b1;                              // recip start pulses 1 cyc (comb)
                    if (recip_done) begin
                        recipn <= recip_q[17:0];                   // 2^17/N  (≤ 2^17)
                        row    <= '0;
                        state  <= LN_RS;
                    end
                end

                LN_RS: begin
                    // INT8-tiled row bases both sides: tile(i,j) at base+(i·nNt+j)·16, row lr at +lr
                    rd_base <= ADDR_W'(WADDR'(src_base)
                             + ((WADDR'(row >> 4) * WADDR'(nNt)) << 4)
                             + WADDR'(row[3:0]));
                    wr_base <= ADDR_W'(WADDR'(dst_base)
                             + ((WADDR'(row >> 4) * WADDR'(nNt)) << 4)
                             + WADDR'(row[3:0]));
                    sumx    <= '0;
                    sumsq   <= '0;
                    wc      <= '0;
                    beat    <= '0;
                    state   <= LN_MEAN;
                end

                LN_MEAN: begin
                    case (beat)
                        5'd0: beat <= 5'd1;                        // x read addr driven (comb)
                        5'd1: begin
                            cvt_word <= mem_rdata;                 // x word captured + lane masks
                            for (int l = 0; l < 16; l++)
                                lnvld_q[l] <= (ln_el0 + DIM_W'(l)) < N;
                            beat     <= 5'd2;
                        end
                        default: begin
                            sumx <= sumx + mean16;                 // masked 16-lane sum
                            beat <= 5'd0;
                            if (wc == WC_W'(nNt) - WC_W'(1)) begin
                                state <= LN_MU;
                            end else begin
                                wc <= wc + WC_W'(1);
                            end
                        end
                    endcase
                end

                LN_MU: begin
                    case (beat)
                        5'd0: begin                                // Σx·recipn — isolated → DSP
                            mprod_q <= 37'(sumx) * 37'($signed({1'b0, recipn}));
                            beat    <= 5'd1;
                        end
                        default: begin                             // μ = product >>> 9 (Q8)
                            mu    <= 18'(mprod_q >>> 9);
                            wc    <= '0;
                            beat  <= '0;
                            state <= LN_VAR;
                        end
                    endcase
                end

                LN_VAR: begin
                    // per word: 0 drive x / 1 capture / even beats stage d, odd beats Σd²
                    case (beat)
                        5'd0: beat <= 5'd1;
                        5'd1: begin
                            cvt_word <= mem_rdata;                 // x word captured + lane masks
                            for (int l = 0; l < 16; l++)
                                lnvld_q[l] <= (ln_el0 + DIM_W'(l)) < N;
                            beat     <= 5'd2;
                        end
                        default: begin
                            if (!beat[0]) begin                    // beats 2/4/6/8: stage d for sub
                                for (int l = 0; l < LANES; l++)
                                    ds4_q[l] <= lnvld_q[int'(var_sub)*4 + l]
                                              ? ds_c[l] : 18'sd0;  // masked lanes → 0 → d² = 0
                                beat <= beat + 5'd1;
                            end else begin                         // beats 3/5/7/9: Σd² (squares → DSP)
                                sumsq <= sumsq + 42'(dsq[0]) + 42'(dsq[1])
                                               + 42'(dsq[2]) + 42'(dsq[3]);
                                if (beat == 5'd9) begin
                                    beat <= 5'd0;
                                    if (wc == WC_W'(nNt) - WC_W'(1))
                                        state <= LN_SG;
                                    else
                                        wc <= wc + WC_W'(1);
                                end else begin
                                    beat <= beat + 5'd1;
                                end
                            end
                        end
                    endcase
                end

                LN_SG: begin
                    case (beat)
                        5'd0: begin                                // Σd²·recipn — isolated → DSP
                            vprod_q <= 60'(sumsq) * 60'(recipn);   // (4th synth: single-cycle version
                            beat    <= 5'd1;                       //  got 1 DSP + 16-CARRY4 completion)
                        end
                        default: begin                             // σ² = product >> 17 + ε (Q16)
                            sig2     <= 32'(vprod_q >> 17) + 32'd1;
                            rstarted <= 1'b0;
                            beat     <= '0;
                            state    <= LN_RQ;
                        end
                    endcase
                end

                LN_RQ: begin
                    rstarted <= 1'b1;                              // rsqrt start pulses 1 cyc (comb)
                    if (rsq_done) begin
                        rs_r  <= rsq_q;                            // 2^16/σ
                        wc    <= '0;
                        beat  <= '0;
                        state <= LN_NORM;
                    end
                end

                LN_NORM: begin
                    // per word beats: 0 drive x / 1 capture x, drive γ / 2 capture γ, drive β /
                    //   3 capture β / 4 = the (sub, ph) engine: 5 phases × subs 0..3 / 5 write y
                    case (beat)
                        5'd0: beat <= 5'd1;
                        5'd1: begin
                            cvt_word <= mem_rdata;                 // x word captured + lane masks
                            for (int l = 0; l < 16; l++)
                                lnvld_q[l] <= (ln_el0 + DIM_W'(l)) < N;
                            beat     <= 5'd2;
                        end
                        5'd2: begin
                            gword <= mem_rdata;
                            beat  <= 5'd3;
                        end
                        5'd3: begin
                            bword <= mem_rdata;
                            sub   <= '0;
                            ph    <= '0;
                            beat  <= 5'd4;
                        end
                        5'd4: begin                                // sub-phase engine
                            case (ph)
                                3'd0: begin                        // d = x·2^8 − μ
                                    for (int l = 0; l < LANES; l++)
                                        lnds_q[l] <= ds_c[l];
                                    ph <= 3'd1;
                                end
                                3'd1: begin                        // ×rs — isolated → DSP (Q24)
                                    for (int l = 0; l < LANES; l++)
                                        np_q[l] <= 44'(lnds_q[l]) * 44'($signed({1'b0, rs_r}));
                                    ph <= 3'd2;
                                end
                                3'd2: begin                        // clamp Q6 → ×γ — → DSP
                                    for (int l = 0; l < LANES; l++)
                                        gy_q[l] <= 24'(nq_c[l])
                                                 * 24'($signed(gword[(int'(sub)*4 + l)*DATA_W +: DATA_W]));
                                    ph <= 3'd3;
                                end
                                3'd3: begin                        // rounded requant (rnd32_q static)
                                    for (int l = 0; l < LANES; l++)
                                        yr_q[l] <= (32'(gy_q[l]) + rnd32_q) >>> ew_shift;
                                    ph <= 3'd4;
                                end
                                default: begin                     // +β, sat, pad-mask → q_acc
                                    q_acc[{sub, 5'd0} +: DATA_W*LANES] <= ln_chunk;
                                    ph <= 3'd0;
                                    if (sub == 2'd3) begin
                                        sub  <= '0;
                                        beat <= 5'd5;
                                    end else begin
                                        sub <= sub + 2'd1;
                                    end
                                end
                            endcase
                        end
                        default: begin                             // beat 5: y word written (comb)
                            beat <= 5'd0;
                            if (wc == WC_W'(nNt) - WC_W'(1))
                                state <= LN_RE;
                            else
                                wc <= wc + WC_W'(1);
                        end
                    endcase
                end

                LN_RE: begin
                    if (row == M - DIM_W'(1)) begin
                        done_r <= 1'b1;
                        state  <= IDLE;
                    end else begin
                        row   <= row + DIM_W'(1);
                        state <= LN_RS;
                    end
                end

                default: state <= IDLE;
            endcase
        end
    end

    // ── Memory drive (combinational) ─────────────────────────────
    always_comb begin
        mem_region = R_A1;                                       // default: act/result mem (softmax/EW)
        mem_addr   = '0;
        mem_en     = 1'b0;
        mem_we     = 1'b0;
        mem_wdata  = '0;
        case (state)
            MAX_P, EXP_P: begin
                if (beat == 5'd0) begin                          // drive read address
                    mem_addr = rd_addr;
                    mem_en   = 1'b1;
                end
            end
            NORM_P: begin
                if ((nc[1:0] == 2'd3) && (beat == 5'd3)) begin   // tile word complete → write
                    mem_addr  = wr_addr;
                    mem_en    = 1'b1;
                    mem_we    = 1'b1;
                    mem_wdata = norm_word;
                end
            end
            EW_A: begin                                          // read operand 1
                mem_addr = src_base  + ewc;
                mem_en   = 1'b1;
            end
            EW_B: begin                                          // read operand 2
                mem_addr = src2_base + ewc;
                mem_en   = 1'b1;
            end
            EW_W: begin                                          // write result (staged)
                mem_addr  = dst_base + ewc;
                mem_en    = 1'b1;
                mem_we    = 1'b1;
                mem_wdata = ew_q;
            end

            // ── promote (region0 → region2) ──
            P_RD: begin
                if (beat == 5'd0) begin                          // read INT8 word (region0)
                    mem_region = R_W0;
                    mem_addr   = src2_base + ewc;
                    mem_en     = 1'b1;
                end
            end
            P_WR: begin
                if (beat == 5'd1) begin                          // write master subword (staged)
                    mem_region = R_M2;
                    mem_addr   = dst_base + r2_off;
                    mem_en     = 1'b1;
                    mem_we     = 1'b1;
                    mem_wdata  = pw_q;
                end
            end

            // ── requant (region2 → region0) ──
            Q_RD: begin
                if (beat == 5'd0) begin                          // read master subword (region2)
                    mem_region = R_M2;
                    mem_addr   = src_base + r2_off;
                    mem_en     = 1'b1;
                end
            end
            Q_WR: begin                                          // write INT8 word (region0)
                mem_region = R_W0;
                mem_addr   = dst_base + ewc;
                mem_en     = 1'b1;
                mem_we     = 1'b1;
                mem_wdata  = q_acc;
            end

            // ── accumulate (master r2, dW r1 → master r2) ──
            AC_A: begin                                          // read master (region2)
                mem_region = R_M2;
                mem_addr   = src_base + r2_off;
                mem_en     = 1'b1;
            end
            AC_B: begin                                          // read dW (region1)
                mem_region = R_A1;
                mem_addr   = src2_base + r2_off;
                mem_en     = 1'b1;
            end
            AC_W: begin
                if (beat == 5'd3) begin                          // write master (staged)
                    mem_region = R_M2;
                    mem_addr   = dst_base + r2_off;
                    mem_en     = 1'b1;
                    mem_we     = 1'b1;
                    mem_wdata  = acw_q;
                end
            end

            // ── layernorm (x/y region1; γ/β region0) ──
            LN_MEAN: begin
                if (beat == 5'd0) begin                          // read x word
                    mem_addr = ln_x_addr;
                    mem_en   = 1'b1;
                end
            end
            LN_VAR: begin
                if (beat == 5'd0) begin                          // read x word
                    mem_addr = ln_x_addr;
                    mem_en   = 1'b1;
                end
            end
            LN_NORM: begin
                case (beat)
                    5'd0: begin                                  // read x word
                        mem_addr = ln_x_addr;
                        mem_en   = 1'b1;
                    end
                    5'd1: begin                                  // read γ word
                        mem_region = R_W0;
                        mem_addr   = ln_g_addr;
                        mem_en     = 1'b1;
                    end
                    5'd2: begin                                  // read β word
                        mem_region = R_W0;
                        mem_addr   = ln_b_addr;
                        mem_en     = 1'b1;
                    end
                    5'd5: begin                                  // write y word
                        mem_addr  = ln_y_addr;
                        mem_en    = 1'b1;
                        mem_we    = 1'b1;
                        mem_wdata = q_acc;
                    end
                    default: ;
                endcase
            end

            default: ;
        endcase
    end

    assign busy = (state != IDLE);
    assign done = done_r;

    // ── Inline activation bank (combinational, independent of the FSM) ──
    //   relu/leaky/bypass = the v1 modules bit-identical; gelu = fixed-point LUT.
    localparam logic [1:0] ACT_RELU   = 2'd0,
                           ACT_LEAKY  = 2'd1,
                           ACT_BYPASS = 2'd2,
                           ACT_GELU   = 2'd3;
    localparam signed [ACC_W-1:0] A_MAX =  ACC_W'(127);
    localparam signed [ACC_W-1:0] A_MIN = -ACC_W'(128);

    // ── GELU: per-lane requant → clamp to 8b → ROM lookup ────────
    logic [DATA_W-1:0]        gelu_idx [ROWS];   // clamped post-shift activation (raw 8b)
    logic signed [DATA_W-1:0] gelu_out [ROWS];   // sat8(K·gelu(idx/K))
    always_comb begin
        for (int r = 0; r < ROWS; r++) begin
            logic signed [ACC_W-1:0] gs;
            gs = $signed(act_in[r]) >>> act_shift;                // requant like the other acts
            gelu_idx[r] = (gs > A_MAX) ? 8'h7F :                  // clamp to [-128,127] → ROM addr
                          (gs < A_MIN) ? 8'h80 : gs[DATA_W-1:0];
        end
    end
    genvar gr;
    generate
        for (gr = 0; gr < ROWS; gr++) begin : g_gelu
            gelu_lut #(.GELU_FBITS(GELU_FBITS), .DATA_W(DATA_W)) u_gelu (
                .idx (gelu_idx[gr]),
                .g   (gelu_out[gr])
            );
        end
    endgenerate

    always_comb begin
        for (int r = 0; r < ROWS; r++) begin
            logic signed [ACC_W-1:0] x, sh, val;
            x          = $signed(act_in[r]);
            sh         = '0;
            val        = '0;
            act_out[r] = '0;
            case (act_sel)
                ACT_RELU: begin
                    val = (x < 0) ? '0 : x;
                    sh  = val >>> act_shift;
                    act_out[r] = (sh > A_MAX) ? DATA_W'(A_MAX) : DATA_W'(sh);
                end
                ACT_LEAKY: begin
                    sh  = x >>> act_shift;
                    val = (sh < 0) ? (sh >>> act_leak) : sh;
                    act_out[r] = (val > A_MAX) ? DATA_W'(A_MAX) :
                                 (val < A_MIN) ? DATA_W'(A_MIN) : DATA_W'(val);
                end
                ACT_BYPASS: begin
                    sh = x >>> act_shift;
                    act_out[r] = (sh > A_MAX) ? DATA_W'(A_MAX) :
                                 (sh < A_MIN) ? DATA_W'(A_MIN) : DATA_W'(sh);
                end
                ACT_GELU: begin
                    act_out[r] = gelu_out[r];                     // fixed-point GELU ROM
                end
                default: act_out[r] = '0;
            endcase
        end
    end

endmodule

`default_nettype wire
