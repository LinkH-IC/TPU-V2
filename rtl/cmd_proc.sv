// =============================================================================
// cmd_proc.sv  (v2 — single-layer command processor / "the brain")
// Decodes the host command stream into registers, then on GO runs ONE op:
//   GO func3 = 0 : the tiled GEMM  C = act((A·W + bias) >> shift)  — sweep every
//     tile, driving the unchanged control_fsm plus the memory-Port-B AGUs.
//   GO func3 != 0: a VPU op — latch func3 as vpu_op, pulse vpu_go, park in S_VPU
//     until vpu_done (the VPU drives the memories itself; bases/dims reuse the
//     same descriptor registers).
//
//   R(i,j) = Σ_k A(i,k)·W(k,j)     i=result row, j=result col, k=contraction,
//                                  w=word-in-tile (0..15)
//   One 16×16 accumulator tile; writes full tiles (host slices partial M/N on
//   read-back). Single layer per `go` — a future layer-loop wraps this engine.
//
//   Per output tile (i,j):
//     CLEAR → [for k: fill W+A, COMPUTE] → fill bias → DRAIN → write back R(i,j)
//
//   Addresses (pre-tiled storage, contraction k inner):
//     W = W_base   + (j*nKt + k)*16 + w      A = A_base + (i*nKt + k)*16 + w
//     R = R_base   + (i*nNt + j)*16 + w      bias = bias_base + j*4 + w
//
//   Pure control: drives addresses / enables / strobes only — data never flows
//   through it (mem rdata -> buffers, result_buffer -> mem wdata are wired at
//   top). Addresses use small multiplies; strength-reduction to incremental
//   adders is a later optimization.
// =============================================================================

`default_nettype none

module cmd_proc #(
    parameter ADDR_W     = 12,    // memory word-address width (DEPTH=4096)
    parameter DIM_W      = 12,    // M/K/N dimension width
    parameter TILE_WORDS = 16,    // words per 16x16 tile
    parameter BIAS_WORDS = 4,     // bias words per output-col tile (4x INT32/word 128b)
    // packed instruction width = opcode(3) + base(ADDR_W) + row(DIM_W) + col(DIM_W)
    localparam int INSTR_W = 3 + ADDR_W + 2*DIM_W
)(
    input  logic                clk,
    input  logic                rst_n,

    // ── Command stream (host -> cmd_proc) ────────────────────────
    //   One packed instruction word; sliced into op/base/row/col below.
    input  logic                cmd_valid,
    input  logic [INSTR_W-1:0]  cmd_word,

    // ── Status (cmd_proc -> host) ────────────────────────────────
    output logic                busy,
    output logic                done,

    // ── Decoded datapath config (cmd_proc -> datapath) ───────────
    output logic [4:0]          cfg_shift,
    output logic [1:0]          cfg_act_sel,
    output logic [2:0]          cfg_leak_shift,
    output logic                cfg_out_fmt,    // 0 = INT8 result, 1 = INT32 raw writeback
    output logic                cfg_bias_en,    // 1 = add bias, 0 = pure A·B (scores / gradients)

    // ── control_fsm handshake ────────────────────────────────────
    output logic [1:0]          cf_cmd,
    output logic                cf_start,
    input  logic                cf_done,

    // ── Weight memory Port B (read: weights + bias) ──────────────
    output logic [ADDR_W-1:0]   wmem_addr,
    output logic                wmem_en,

    // ── Activation memory Port B (read A / write R) ──────────────
    output logic [ADDR_W-1:0]   amem_addr,
    output logic                amem_en,
    output logic                amem_we,

    // ── Buffer fill / writeback strobes ──────────────────────────
    output logic                wbuf_fill_en,   // weight_buffer  (fill from W mem)
    output logic                abuf_fill_en,   // activation_buffer (fill from A mem)
    output logic                bias_fill_en,   // bias_add       (fill from W mem)
    output logic                rbuf_wb_en,     // result_buffer  (write back to A mem)

    // ── VPU dispatch (GO func3 != 0) ─────────────────────────────
    output logic                vpu_go,         // pulse: start the VPU op
    output logic                vpu_sel_amem,   // VPU owns memory Port B (region-routed in top)
    output logic [3:0]          vpu_op,         // GO func3 target (1..9 — vpu.sv decodes)
    output logic [DIM_W-1:0]    vpu_M,          // rows
    output logic [DIM_W-1:0]    vpu_N,          // row length / cols (= K reg, reused)
    output logic [ADDR_W-1:0]   vpu_src,        // operand-1 base (= A_base)
    output logic [ADDR_W-1:0]   vpu_src2,       // operand-2 base, elementwise (= W_base)
    output logic [ADDR_W-1:0]   vpu_dst,        // result base (= R_base)
    output logic [15:0]         vpu_scale,
    input  logic                vpu_done
);

    // ── Opcodes ──────────────────────────────────────────────────
    localparam logic [2:0] OP_WEIGHT = 3'd1,   // base=W_base, row=K, col=N
                           OP_BIAS   = 3'd2,   // base=bias_base
                           OP_ACT    = 3'd3,   // base=A_base, row=M, col=K
                           OP_RESULT = 3'd4,   // base=R_base
                           OP_CONFIG = 3'd5,   // base=packed config
                           OP_GO     = 3'd6;   // start

    // ── Instruction word field map (LSB-aligned slices of cmd_word) ──
    //   [ op : base : row : col ]  MSB -> LSB
    localparam int COL_LSB  = 0;                // col  at [DIM_W-1 : 0]
    localparam int ROW_LSB  = DIM_W;            // row  at [2*DIM_W-1 : DIM_W]
    localparam int BASE_LSB = 2*DIM_W;          // base at [2*DIM_W+ADDR_W-1 : 2*DIM_W]
    localparam int OP_LSB   = 2*DIM_W + ADDR_W; // op   at the top 3 bits

    wire [2:0]        cmd_op   = cmd_word[OP_LSB   +: 3];      // opcode
    wire [ADDR_W-1:0] cmd_base = cmd_word[BASE_LSB +: ADDR_W]; // mem base / packed cfg
    wire [DIM_W-1:0]  cmd_row  = cmd_word[ROW_LSB  +: DIM_W];  // M (act) — unused for others
    wire [DIM_W-1:0]  cmd_col  = cmd_word[COL_LSB  +: DIM_W];  // K (act) / N (weight)

    // ── control_fsm command encoding (matches control_fsm.sv) ────
    localparam logic [1:0] CF_COMPUTE = 2'b00,
                           CF_DRAIN   = 2'b01,
                           CF_CLEAR   = 2'b11;

    // ── Word-phase terminals ─────────────────────────────────────
    localparam int FILL_LAST = TILE_WORDS;       // addr 0..15, fill_en 1..16 (1-cyc read latency)
    localparam int BIAS_LAST = BIAS_WORDS;       // addr 0..3,  fill_en 1..4
    localparam int WB_I8     = TILE_WORDS;       // INT8  writeback: 16 words/tile (one C row each)
    localparam int WB_I32    = TILE_WORDS * 4;   // INT32 writeback: 64 words/tile (32b -> 4 words/row)

    localparam int TILE_W = DIM_W - 3;           // tile-count width (ceil(4095/16)=256 -> 9b)
    localparam int WCNT_W = $clog2(WB_I32);      // word counter holds 0..63 (INT32 WB is the longest phase)

    // ── FSM ──────────────────────────────────────────────────────
    typedef enum logic [3:0] {
        IDLE,
        S_CLEAR,
        S_FILL,
        S_COMPUTE,
        S_BIAS,
        S_DRAIN,
        S_WB,
        S_DONE,
        S_VPU       // GO func3 != 0: park while the VPU runs the op
    } state_t;
    state_t state, state_next, state_q;

    // ── Descriptor registers (latched from the command stream) ───
    logic [ADDR_W-1:0] W_base, A_base, R_base, bias_base;
    logic [DIM_W-1:0]  M, K, N;
    logic [11:0]       cfg;   // {bias_off, out_fmt, leak_shift[2:0], act_sel[1:0], shift[4:0]}
    logic [15:0]       vpu_scale_r;   // VPU per-op constant (CONFIG row/col fields)
    logic [3:0]        vpu_op_r;      // VPU op latched on GO (GO target)

    // ── Loop counters + word counter ─────────────────────────────
    logic [TILE_W-1:0] i_cnt, j_cnt, k_cnt;
    logic [WCNT_W-1:0] wcnt;
    logic              done_reg;

    // ── Tile counts (ceil(dim/16)) and their -1 forms ────────────
    logic [TILE_W-1:0] nMt, nKt, nNt, nMt_m1, nKt_m1, nNt_m1;   // number of M,K,N tiles
    assign nMt = {1'b0, M[DIM_W-1:4]} + TILE_W'(|M[3:0]);       // nMt = floor(M/16) + (1 if M isn't a multiple of 16) = ceil(M/16)
    assign nKt = {1'b0, K[DIM_W-1:4]} + TILE_W'(|K[3:0]);
    assign nNt = {1'b0, N[DIM_W-1:4]} + TILE_W'(|N[3:0]);
    assign nMt_m1 = nMt - TILE_W'(1);
    assign nKt_m1 = nKt - TILE_W'(1);
    assign nNt_m1 = nNt - TILE_W'(1);

    // ── Descriptor decode (only while idle) ──────────────────────
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            W_base <= '0;
            A_base <= '0;
            R_base <= '0;
            bias_base <= '0;
            M <= '0;
            K <= '0;
            N <= '0;
            cfg <= '0;
            vpu_scale_r <= '0;
            vpu_op_r <= '0;
        end else if (state == IDLE && cmd_valid) begin
            case (cmd_op)
                OP_WEIGHT: begin W_base    <= cmd_base; N <= cmd_col; end
                OP_BIAS:         bias_base <= cmd_base;
                OP_ACT:    begin A_base    <= cmd_base; M <= cmd_row; K <= cmd_col; end
                OP_RESULT:       R_base    <= cmd_base;
                OP_CONFIG: begin cfg       <= cmd_base[11:0];
                                 vpu_scale_r <= {cmd_row[3:0], cmd_col}; end  // 16b VPU constant
                OP_GO:           vpu_op_r  <= cmd_base[3:0];                   // GO target -> VPU op
                default: ;
            endcase
        end
    end

    assign cfg_shift      = cfg[4:0];
    assign cfg_act_sel    = cfg[6:5];
    assign cfg_leak_shift = cfg[9:7];
    assign cfg_out_fmt    = cfg[10];        // 0 = INT8 (default), 1 = INT32 raw
    assign cfg_bias_en    = ~cfg[11];       // cfg[11]=1 suppresses bias; default 0 -> bias on (backward-compatible)

    // Active writeback terminal: INT32 emits 4× the words (64 vs 16 per tile).
    wire [WCNT_W-1:0] wb_last = cfg_out_fmt ? WCNT_W'(WB_I32 - 1)    // 63
                                            : WCNT_W'(WB_I8  - 1);   // 15

    // ── AGU addresses (combinational) ────────────────────────────
    // Tile index (j*nKt+k etc.), then ×16 + word. Valid configs keep the total
    // < DEPTH, so the low ADDR_W bits are the true word address.
    //
    // Strength reduction: the j*nKt / i*nKt / i*nNt products are NOT multiplied
    // here — a runtime 9×9 multiply feeding straight into the BRAM address was
    // the 100 MHz critical path. Instead they live in the registered accumulators
    // j_nKt/i_nKt/i_nNt below, advanced by += stride on the SAME odometer edges as
    // i/j/k (see the counter block). Invariant j_nKt==j_cnt*nKt holds every cycle,
    // so the address sequence is bit-identical — just multiply-free (k has unit
    // stride, so it is added live, no accumulator needed).
    localparam int IDX_W = 2*TILE_W;
    logic [IDX_W-1:0]  j_nKt, i_nKt, i_nNt;          // registered: j*nKt, i*nKt, i*nNt
    logic [IDX_W-1:0]  w_idx, a_idx, r_idx;
    logic [ADDR_W-1:0] w_addr, a_addr, r_addr, b_addr;
    assign w_idx  = j_nKt + IDX_W'(k_cnt);
    assign a_idx  = i_nKt + IDX_W'(k_cnt);
    assign r_idx  = i_nNt + IDX_W'(j_cnt);
    assign w_addr = W_base    + ADDR_W'(w_idx << 4) + ADDR_W'(wcnt);
    assign a_addr = A_base    + ADDR_W'(a_idx << 4) + ADDR_W'(wcnt);
    assign r_addr = R_base    + ADDR_W'(cfg_out_fmt ? (r_idx << 6) : (r_idx << 4)) + ADDR_W'(wcnt);  // INT32 tile = 64 words, INT8 = 16
    assign b_addr = bias_base + ADDR_W'(j_cnt << 2) + ADDR_W'(wcnt);

    // ── Previous state (cf_start entry pulse) ────────────────────
    always_ff @(posedge clk) begin
        if (!rst_n)
            state_q <= IDLE;
        else
            state_q <= state;
    end
    wire entry = (state != state_q);

    // ── State register ───────────────────────────────────────────
    always_ff @(posedge clk) begin
        if (!rst_n)
            state <= IDLE;
        else
            state <= state_next;
    end

    // ── Next-state logic ─────────────────────────────────────────
    always_comb begin
        state_next = state;
        case (state)
            IDLE:      if (cmd_valid && cmd_op == OP_GO)
                           state_next = (|cmd_base[3:0]) ? S_VPU : S_CLEAR;  // target 0=GEMM, else VPU op
            S_CLEAR:   if (cf_done)                        state_next = S_FILL;
            S_FILL:    if (wcnt == WCNT_W'(FILL_LAST))     state_next = S_COMPUTE;
            S_COMPUTE: if (cf_done)                        state_next = (k_cnt == nKt_m1) ? S_BIAS : S_FILL;
            S_BIAS:    if (wcnt == WCNT_W'(BIAS_LAST))     state_next = S_DRAIN;
            S_DRAIN:   if (cf_done)                        state_next = S_WB;
            S_WB:      if (wcnt == wb_last)
                           state_next = (j_cnt != nNt_m1) ? S_CLEAR :
                                        (i_cnt != nMt_m1) ? S_CLEAR : S_DONE;
            S_VPU:     if (vpu_done)                       state_next = S_DONE;
            S_DONE:                                        state_next = IDLE;
            default:                                       state_next = IDLE;
        endcase
    end

    // ── Loop counters (odometer: k inner, then j, then i) ────────
    //   The tile-base accumulators j_nKt/i_nKt/i_nNt advance here in lockstep
    //   with j/i, keeping the AGU multiply-free (see the AGU block above).
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            i_cnt <= '0;
            j_cnt <= '0;
            k_cnt <= '0;
            j_nKt <= '0;
            i_nKt <= '0;
            i_nNt <= '0;
        end else begin
            case (state)
                IDLE: if (cmd_valid && cmd_op == OP_GO) begin
                        i_cnt <= '0;
                        j_cnt <= '0;
                        k_cnt <= '0;
                        j_nKt <= '0;
                        i_nKt <= '0;
                        i_nNt <= '0;
                      end
                S_COMPUTE: if (cf_done && k_cnt != nKt_m1)
                               k_cnt <= k_cnt + TILE_W'(1);
                S_WB: if (wcnt == wb_last) begin
                          k_cnt <= '0;
                          if (j_cnt != nNt_m1) begin
                                j_cnt <= j_cnt + TILE_W'(1);
                                j_nKt <= j_nKt + IDX_W'(nKt);    // j++       -> j*nKt += nKt
                          end
                          else begin
                              j_cnt <= '0;
                              j_nKt <= '0;                       // j wraps   -> j*nKt = 0
                              if (i_cnt != nMt_m1) begin
                                i_cnt <= i_cnt + TILE_W'(1);
                                i_nKt <= i_nKt + IDX_W'(nKt);    // i++       -> i*nKt += nKt
                                i_nNt <= i_nNt + IDX_W'(nNt);    // i++       -> i*nNt += nNt
                              end
                          end
                      end
                default: ;
            endcase
        end
    end

    // ── Word counter (runs only during the word phases) ──────────
    always_ff @(posedge clk) begin
        if (!rst_n)
            wcnt <= '0;
        else if (state == S_FILL || state == S_BIAS || state == S_WB)
            wcnt <= wcnt + WCNT_W'(1);
        else
            wcnt <= '0;
    end

    // ── Done flag (held until next go) ───────────────────────────
    always_ff @(posedge clk) begin
        if (!rst_n)
            done_reg <= 1'b0;
        else if (state == IDLE && cmd_valid && cmd_op == OP_GO)
            done_reg <= 1'b0;
        else if (state == S_DONE)
            done_reg <= 1'b1;
    end

    assign busy = (state != IDLE);
    assign done = done_reg;

    // ── VPU dispatch outputs ─────────────────────────────────────
    assign vpu_go       = (state == S_VPU) && entry;   // 1-cyc start pulse
    assign vpu_sel_amem = (state == S_VPU);            // VPU owns memory Port B (region-routed in top)
    assign vpu_op       = vpu_op_r;
    assign vpu_M        = M;
    assign vpu_N        = K;                            // K reg carries the VPU op's N / cols
    assign vpu_src      = A_base;
    assign vpu_src2     = W_base;                       // elementwise 2nd operand
    assign vpu_dst      = R_base;
    assign vpu_scale    = vpu_scale_r;

    // ── Output / AGU drive (combinational) ───────────────────────
    always_comb begin
        cf_cmd          = CF_COMPUTE;
        cf_start        = 1'b0;
        wmem_addr       = '0;
        wmem_en         = 1'b0;
        amem_addr       = '0;
        amem_en         = 1'b0;
        amem_we         = 1'b0;
        wbuf_fill_en    = 1'b0;
        abuf_fill_en    = 1'b0;
        bias_fill_en    = 1'b0;
        rbuf_wb_en      = 1'b0;

        case (state)
            S_CLEAR: begin
                cf_cmd   = CF_CLEAR;
                cf_start = entry;
            end

            S_FILL: begin
                // drive W-tile + A-tile read addresses (words 0..15)
                if (wcnt <= WCNT_W'(FILL_LAST-1)) begin
                    wmem_addr = w_addr;
                    wmem_en = 1'b1;
                    amem_addr = a_addr;
                    amem_en = 1'b1;
                end
                // capture returning words (1-cycle read latency): words 1..16
                if (wcnt >= WCNT_W'(1) && wcnt <= WCNT_W'(FILL_LAST)) begin
                    wbuf_fill_en = 1'b1;
                    abuf_fill_en = 1'b1;
                end
            end

            S_COMPUTE: begin
                cf_cmd   = CF_COMPUTE;
                cf_start = entry;
            end

            S_BIAS: begin
                if (wcnt <= WCNT_W'(BIAS_LAST-1)) begin
                    wmem_addr = b_addr;
                    wmem_en = 1'b1;
                end
                if (wcnt >= WCNT_W'(1) && wcnt <= WCNT_W'(BIAS_LAST))
                    bias_fill_en = 1'b1;
            end

            S_DRAIN: begin
                cf_cmd   = CF_DRAIN;
                cf_start = entry;
            end

            S_WB: begin
                if (wcnt <= wb_last) begin
                    amem_addr   = r_addr;
                    amem_en     = 1'b1;
                    amem_we     = 1'b1;
                    rbuf_wb_en  = 1'b1;
                end
            end

            default: ;
        endcase
    end

endmodule

`default_nettype wire
