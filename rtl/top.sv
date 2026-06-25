// =============================================================================
// top.sv  (v2 — instruction-driven integration)
// The whole v2 accelerator: on-chip memory + the cmd_proc brain + the inner
// control_fsm + the systolic datapath, behind a single host contract.
//
//   Host boundary (v1's many direct buffer ports are gone):
//     - ONE region-decoded memory port (load weights/bias/A, read R back)
//     - a command stream (cmd_valid + packed cmd_word)
//     - busy / done status
//
//   Three on-chip memories. The host sees TWO regions behind one port:
//     host_addr[ADDR_W]   = region   (0 = weight mem, 1 = activation/result mem)
//     host_addr[ADDR_W-1:0] = word address within that memory
//   Port A read is synchronous (1-cyc) -> host_rdata selects on the REGISTERED
//   region bit. Region 2 (INT32 master weights) is VPU-internal — no host port.
//   Port B is region-routed: cmd_proc owns wmem+amem during a GEMM; a VPU op
//   hands ALL Port Bs to the VPU, which steers its single access by mem_region.
//   Sequential ownership -> no arbiter.
//
//   GO(func3) selects the engine: func3=0 runs the tiled GEMM — per output tile
//     CLEAR -> [for k: fill W+A, COMPUTE] -> fill bias -> DRAIN -> write back R
//   (cmd_proc fills the staging buffers, the unchanged control_fsm emits them
//   into the array; data never flows through cmd_proc). func3!=0 parks cmd_proc
//   in S_VPU and the VPU runs the op (softmax / elementwise / optimizer / LN).
//
//   Drain path keeps v1's two pipeline registers (biased_reg, act_reg) for
//   100 MHz timing; the drain_done handshake to control_fsm is delayed 2 cycles
//   to match, so cf_done lands only after the last result column reaches the
//   result_buffer.
// =============================================================================

`default_nettype none

module top #(
    parameter ROWS   = 16,
    parameter COLS   = 16,
    parameter DATA_W = 8,
    parameter ACC_W  = 32,
    parameter DEPTH  = 8192,                        // words per memory (= mem DEPTH)
    parameter DIM_W  = 12,                          // M/K/N dimension width (host contract)
    localparam int WORD_W  = COLS*DATA_W,           // 128b memory word = one 16-byte column
    localparam int ADDR_W  = $clog2(DEPTH),         // 13b word address
    localparam int INSTR_W = 3 + ADDR_W + 2*DIM_W   // packed command word (= 40b)
)(
    input  logic                            clk,
    input  logic                            rst_n,

    // ── Host memory port (unified, region-decoded) ───────────
    input  logic [ADDR_W:0]                 host_addr,    // [ADDR_W]=region, [ADDR_W-1:0]=word
    input  logic                            host_en,
    input  logic                            host_we,
    input  logic [WORD_W-1:0]               host_wdata,
    output logic [WORD_W-1:0]               host_rdata,   // registered, valid 1 cycle after host_en

    // ── Host command stream ──────────────────────────────────
    input  logic                            cmd_valid,
    input  logic [INSTR_W-1:0]              cmd_word,

    // ── Status (cmd_proc -> host) ────────────────────────────
    output logic                            busy,
    output logic                            done
);

    // ── cmd_proc -> control_fsm handshake ────────────────────
    logic [1:0]                     cp_cf_cmd;
    logic                           cp_cf_start;
    logic                           cf_done;

    // ── cmd_proc -> datapath config ──────────────────────────
    logic [4:0]                     cfg_shift;
    logic [1:0]                     cfg_act_sel;
    logic [2:0]                     cfg_leak_shift;
    logic                           cfg_out_fmt;    // 0 = INT8 result, 1 = INT32 raw
    logic                           cfg_bias_en;    // 1 = add bias, 0 = pure A·B

    // ── Memory Port B — sources + region-routed mux ──────────
    //   During a VPU op the VPU owns Port B of ALL three mems; its mem_region
    //   picks which one takes the (single) access. cmd_proc drives wmem/amem
    //   during GEMM. Master mem (region 2) is VPU-only. No arbiter.
    logic [ADDR_W-1:0]              cp_wmem_addr;    // cmd_proc -> weight mem (GEMM read)
    logic                           cp_wmem_en;
    logic [ADDR_W-1:0]              cp_amem_addr;    // cmd_proc -> act/result mem (GEMM)
    logic                           cp_amem_en;
    logic                           cp_amem_we;
    logic [1:0]                     vpu_region;      // VPU access region (0=W 1=A 2=master)
    logic [1:0]                     vpu_region_q;    // registered (1-cyc rdata latency)
    logic [ADDR_W-1:0]              vpu_mem_addr;    // VPU -> selected mem Port B
    logic                           vpu_mem_en;
    logic                           vpu_mem_we;
    logic [WORD_W-1:0]              vpu_mem_wdata;
    logic [WORD_W-1:0]              vpu_mem_rdata;   // muxed from the region's b_rdata
    logic                           vpu_r0, vpu_r1, vpu_r2;
    logic [ADDR_W-1:0]              wmem_b_addr;     // muxed -> u_wmem Port B
    logic                           wmem_b_en;
    logic                           wmem_b_we;
    logic [WORD_W-1:0]              wmem_b_wdata;
    logic [ADDR_W-1:0]              amem_b_addr;     // muxed -> u_amem Port B
    logic                           amem_b_en;
    logic                           amem_b_we;
    logic [ADDR_W-1:0]              mmem_b_addr;     // -> u_mmem Port B (VPU only)
    logic                           mmem_b_en;
    logic                           mmem_b_we;
    logic [WORD_W-1:0]              mmem_b_wdata;

    // ── VPU dispatch (cmd_proc <-> vpu) ──────────────────────
    logic                           vpu_go;
    logic                           vpu_sel_amem;
    logic [3:0]                     vpu_op;
    logic [DIM_W-1:0]               vpu_M, vpu_N;
    logic [ADDR_W-1:0]              vpu_src, vpu_src2, vpu_dst;
    logic [15:0]                    vpu_scale;
    logic                           vpu_done;

    // ── cmd_proc -> buffer fill / writeback strobes ──────────
    logic                           wbuf_fill_en;
    logic                           abuf_fill_en;
    logic                           bias_fill_en;
    logic                           rbuf_wb_en;

    // ── Intentionally-unused submodule status outputs ────────
    //   Named (not left as empty pins) so they're observable in the integration
    //   bench. The serialized cmd_proc doesn't consume the buffers' fill_ready /
    //   collect_done status.
    /* verilator lint_off UNUSEDSIGNAL */
    logic                           wbuf_fill_ready;
    logic                           abuf_fill_ready;
    logic                           rbuf_collect_done;
    logic                           vpu_busy;          // top.busy comes from cmd_proc
    logic [WORD_W-1:0]              mmem_a_rdata_unused; // master mem Port A (host has no access)
    /* verilator lint_on UNUSEDSIGNAL */

    // ── Memory data buses (128b words) ───────────────────────
    logic [WORD_W-1:0]              wmem_b_rdata;   // weight mem -> weight_buffer + bias_add
    logic [WORD_W-1:0]              amem_b_rdata;   // act mem    -> activation_buffer + vpu
    logic [WORD_W-1:0]              mmem_b_rdata;   // master mem -> vpu (requant/accumulate reads)
    logic [WORD_W-1:0]              rbuf_wb_data;   // result_buffer -> act mem (GEMM writeback)
    logic [WORD_W-1:0]              amem_b_wdata;   // muxed -> u_amem Port B
    logic [WORD_W-1:0]              wmem_a_rdata;   // host read, region 0
    logic [WORD_W-1:0]              amem_a_rdata;   // host read, region 1

    // ── Internal Wires — Weight Buffer to Systolic Array ─────
    logic [ROWS-1:0][DATA_W-1:0]    wb_weight_out;
    logic [COLS-1:0]                wb_weight_load;

    // ── Internal Wires — Activation Buffer to Systolic Array ─
    logic [ROWS-1:0][DATA_W-1:0]    ub_act_out;
    logic [ROWS-1:0]                ub_valid;

    // ── Internal Wires — Systolic Array to Accumulator ───────
    logic [COLS-1:0][ACC_W-1:0]     sa_psum_out;
    logic [COLS-1:0]                sa_valid_out;

    // ── Internal Wires — Accumulator to Bias Add ─────────────
    logic [ROWS-1:0][ACC_W-1:0]     acc_out;
    logic [$clog2(COLS)-1:0]        acc_col_idx;
    logic                           acc_valid;

    // ── Internal Wires — Bias Add to Activation Functions ────
    logic [ROWS-1:0][ACC_W-1:0]     biased_out;

    // ── Internal Wires — VPU inline activation output ────────
    logic [ROWS-1:0][DATA_W-1:0]    act_out;        // vpu inline act (relu/leaky/bypass/gelu)

    // ── Pipeline Registers — Drain Path (R1, R2) ─────────────
    logic [ROWS-1:0][ACC_W-1:0]     biased_reg;     // R1: post-bias_add
    logic [ROWS-1:0][DATA_W-1:0]    act_reg;        // R2: post-activation
    logic [ROWS-1:0][ACC_W-1:0]     biased_reg_q2;  // R2-aligned raw INT32 (out_fmt=1 writeback source)
    logic [ROWS-1:0][ACC_W-1:0]     result_collect; // result_buffer collect data (INT8 sext / INT32 raw)

    // ── Drain Pipeline Aligners — valid + col_idx ────────────
    logic                           acc_valid_q1, acc_valid_q2;
    logic [$clog2(COLS)-1:0]        acc_col_idx_q1, acc_col_idx_q2;

    // ── Drain-done Delay Line (2 cycles to match data) ───────
    logic                           acc_drain_done_raw;
    logic                           drain_done_q1;
    logic                           fsm_acc_drain_done;

    // ── control_fsm handshakes (emit side + accumulator) ─────
    logic                           fsm_wb_load_trigger;
    logic                           fsm_wb_done;
    logic                           fsm_ab_load_trigger;
    logic                           fsm_ab_load_done;
    logic                           fsm_acc_clear;
    logic                           fsm_acc_drain_trigger;
    logic                           fsm_acc_pass_done;

    // ── Host memory decoder (one host port -> two memories) ──
    //   region = top address bit; word address = low ADDR_W bits.
    logic                           host_region;
    logic                           host_region_q;
    logic [ADDR_W-1:0]              host_word_addr;
    logic                           wmem_a_en;
    logic                           amem_a_en;

    assign host_region    = host_addr[ADDR_W];
    assign host_word_addr = host_addr[ADDR_W-1:0];
    assign wmem_a_en      = host_en && !host_region;    // region 0 = weight mem
    assign amem_a_en      = host_en &&  host_region;    // region 1 = act/result mem

    // Register the region of the issued access so the 1-cycle-late rdata mux
    // points at the memory that was actually read.
    always_ff @(posedge clk) begin
        if (!rst_n)
            host_region_q <= 1'b0;
        else if (host_en)
            host_region_q <= host_region;
    end

    assign host_rdata = host_region_q ? amem_a_rdata : wmem_a_rdata;

    // ── Drain Pipeline Registers ─────────────────────────────
    //   R1: biased_reg  (acc_out -> bias_add -> R1)
    //   R2: act_reg      (R1 -> activation -> R2)
    //   acc_valid + acc_col_idx pipelined alongside; drain_done delayed 2 cycles
    //   so control_fsm holds DRAIN until the final R2 column lands in result_buffer.
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            biased_reg         <= '0;
            act_reg            <= '0;
            biased_reg_q2      <= '0;

            acc_valid_q1       <= 1'b0;
            acc_valid_q2       <= 1'b0;

            acc_col_idx_q1     <= '0;
            acc_col_idx_q2     <= '0;

            drain_done_q1      <= 1'b0;
            fsm_acc_drain_done <= 1'b0;
        end
        else begin
            biased_reg         <= biased_out;
            act_reg            <= act_out;
            biased_reg_q2      <= biased_reg;          // align raw INT32 with act_reg (R2)

            acc_valid_q1       <= acc_valid;
            acc_valid_q2       <= acc_valid_q1;

            acc_col_idx_q1     <= acc_col_idx;
            acc_col_idx_q2     <= acc_col_idx_q1;

            drain_done_q1      <= acc_drain_done_raw;   // +1: matches biased_reg
            fsm_acc_drain_done <= drain_done_q1;        // +2: matches act_reg (result_buffer fill)
        end
    end

    // ── Weight Memory (Port A = host region 0, Port B = cmd_proc read / VPU r+w) ──
    mem #(
        .DATA_W (DATA_W),
        .LANES  (COLS),
        .DEPTH  (DEPTH)
    ) u_wmem (
        .clk     (clk),
        .a_addr  (host_word_addr),
        .a_en    (wmem_a_en),
        .a_we    (host_we),
        .a_wdata (host_wdata),
        .a_rdata (wmem_a_rdata),
        .b_addr  (wmem_b_addr),
        .b_en    (wmem_b_en),
        .b_we    (wmem_b_we),            // VPU requant writes refreshed INT8 weights here
        .b_wdata (wmem_b_wdata),
        .b_rdata (wmem_b_rdata)
    );

    // ── Activation / Result Memory (Port A = host region 1, Port B = muxed engine R/W) ──
    mem #(
        .DATA_W (DATA_W),
        .LANES  (COLS),
        .DEPTH  (DEPTH)
    ) u_amem (
        .clk     (clk),
        .a_addr  (host_word_addr),
        .a_en    (amem_a_en),
        .a_we    (host_we),
        .a_wdata (host_wdata),
        .a_rdata (amem_a_rdata),
        .b_addr  (amem_b_addr),
        .b_en    (amem_b_en),
        .b_we    (amem_b_we),
        .b_wdata (amem_b_wdata),
        .b_rdata (amem_b_rdata)
    );

    // ── Command Processor (the brain) ────────────────────────
    cmd_proc #(
        .ADDR_W     (ADDR_W),
        .DIM_W      (DIM_W),
        .TILE_WORDS (COLS),
        .BIAS_WORDS ((COLS*ACC_W)/WORD_W)   // biases per tile = COLS / (WORD_W/ACC_W) = 4
    ) u_cmd_proc (
        .clk            (clk),
        .rst_n          (rst_n),
        .cmd_valid      (cmd_valid),
        .cmd_word       (cmd_word),
        .busy           (busy),
        .done           (done),
        .cfg_shift      (cfg_shift),
        .cfg_act_sel    (cfg_act_sel),
        .cfg_leak_shift (cfg_leak_shift),
        .cfg_out_fmt    (cfg_out_fmt),
        .cfg_bias_en    (cfg_bias_en),
        .cf_cmd         (cp_cf_cmd),
        .cf_start       (cp_cf_start),
        .cf_done        (cf_done),
        .wmem_addr      (cp_wmem_addr),
        .wmem_en        (cp_wmem_en),
        .amem_addr      (cp_amem_addr),
        .amem_en        (cp_amem_en),
        .amem_we        (cp_amem_we),
        .wbuf_fill_en   (wbuf_fill_en),
        .abuf_fill_en   (abuf_fill_en),
        .bias_fill_en   (bias_fill_en),
        .rbuf_wb_en     (rbuf_wb_en),
        .vpu_go         (vpu_go),
        .vpu_sel_amem   (vpu_sel_amem),
        .vpu_op         (vpu_op),
        .vpu_M          (vpu_M),
        .vpu_N          (vpu_N),
        .vpu_src        (vpu_src),
        .vpu_src2       (vpu_src2),
        .vpu_dst        (vpu_dst),
        .vpu_scale      (vpu_scale),
        .vpu_done       (vpu_done)
    );

    // ── Port B region routing ────────────────────────────────
    //   vpu_sel_amem = VPU owns Port B (cmd_proc idle in S_VPU). The VPU's
    //   mem_region then selects which physical mem takes its single access;
    //   region defaults to 1 for softmax/EW so their behavior is unchanged.
    assign vpu_r0 = vpu_sel_amem && (vpu_region == 2'd0);   // weight mem
    assign vpu_r1 = vpu_sel_amem && (vpu_region == 2'd1);   // act/result mem
    assign vpu_r2 = vpu_sel_amem && (vpu_region == 2'd2);   // master mem

    // weight mem Port B: cmd_proc read (GEMM) OR VPU (promote read / requant write)
    assign wmem_b_addr  = vpu_r0 ? vpu_mem_addr  : cp_wmem_addr;
    assign wmem_b_en    = vpu_r0 ? vpu_mem_en    : cp_wmem_en;
    assign wmem_b_we    = vpu_r0 && vpu_mem_we;
    assign wmem_b_wdata = vpu_mem_wdata;

    // act/result mem Port B: VPU (region 1) OR cmd_proc + result_buffer (GEMM)
    assign amem_b_addr  = vpu_r1 ? vpu_mem_addr  : cp_amem_addr;
    assign amem_b_en    = vpu_r1 ? vpu_mem_en    : cp_amem_en;
    assign amem_b_we    = vpu_r1 ? vpu_mem_we    : cp_amem_we;
    assign amem_b_wdata = vpu_r1 ? vpu_mem_wdata : rbuf_wb_data;

    // master mem Port B: VPU only
    assign mmem_b_addr  = vpu_mem_addr;
    assign mmem_b_en    = vpu_r2;
    assign mmem_b_we    = vpu_r2 && vpu_mem_we;
    assign mmem_b_wdata = vpu_mem_wdata;

    // VPU read-data: register the access region (1-cyc mem latency), mux b_rdata.
    always_ff @(posedge clk) begin
        if (!rst_n)
            vpu_region_q <= 2'd1;
        else
            vpu_region_q <= vpu_region;
    end
    assign vpu_mem_rdata = (vpu_region_q == 2'd0) ? wmem_b_rdata :
                           (vpu_region_q == 2'd2) ? mmem_b_rdata : amem_b_rdata;

    // ── Master-Weight Memory (region 2, INT32) — VPU-only, no host port ──
    mem #(
        .DATA_W (DATA_W),
        .LANES  (COLS),
        .DEPTH  (DEPTH)
    ) u_mmem (
        .clk     (clk),
        .a_addr  ('0),
        .a_en    (1'b0),
        .a_we    (1'b0),
        .a_wdata ('0),
        .a_rdata (mmem_a_rdata_unused),
        .b_addr  (mmem_b_addr),
        .b_en    (mmem_b_en),
        .b_we    (mmem_b_we),
        .b_wdata (mmem_b_wdata),
        .b_rdata (mmem_b_rdata)
    );

    // ── Weight Buffer (fill from weight mem, emit columns to array) ──
    weight_buffer #(
        .ROWS   (ROWS),
        .COLS   (COLS),
        .DATA_W (DATA_W)
    ) u_weight_buffer (
        .clk          (clk),
        .rst_n        (rst_n),
        .fill_data    (wmem_b_rdata),
        .fill_en      (wbuf_fill_en),
        .fill_ready   (wbuf_fill_ready),       // single-buffer; cmd_proc serializes
        .load_trigger (fsm_wb_load_trigger),
        .done         (fsm_wb_done),
        .weight_out   (wb_weight_out),
        .weight_load  (wb_weight_load)
    );

    // ── Activation Buffer (fill from act mem, stream rows to array) ──
    activation_buffer #(
        .ROWS   (ROWS),
        .COLS   (COLS),
        .DATA_W (DATA_W)
    ) u_activation_buffer (
        .clk          (clk),
        .rst_n        (rst_n),
        .fill_data    (amem_b_rdata),
        .fill_en      (abuf_fill_en),
        .fill_ready   (abuf_fill_ready),       // single-buffer; cmd_proc serializes
        .load_trigger (fsm_ab_load_trigger),
        .load_done    (fsm_ab_load_done),
        .act_out      (ub_act_out),
        .valid        (ub_valid)
    );

    // ── Systolic Array ───────────────────────────────────────
    systolic_array #(
        .ROWS   (ROWS),
        .COLS   (COLS),
        .DATA_W (DATA_W),
        .ACC_W  (ACC_W)
    ) u_systolic_array (
        .clk         (clk),
        .rst_n       (rst_n),
        .weight_in   (wb_weight_out),
        .weight_load (wb_weight_load),
        .act_in      (ub_act_out),
        .valid_in    (ub_valid),
        .psum_out    (sa_psum_out),
        .valid_out   (sa_valid_out)
    );

    // ── Accumulator ──────────────────────────────────────────
    accumulator #(
        .ROWS  (ROWS),
        .COLS  (COLS),
        .ACC_W (ACC_W)
    ) u_accumulator (
        .clk           (clk),
        .rst_n         (rst_n),
        .psum_in       (sa_psum_out),
        .valid_in      (sa_valid_out),
        .clear         (fsm_acc_clear),
        .drain_trigger (fsm_acc_drain_trigger),
        .acc_out       (acc_out),
        .col_idx       (acc_col_idx),
        .acc_valid     (acc_valid),
        .pass_done     (fsm_acc_pass_done),
        .drain_done    (acc_drain_done_raw)
    );

    // ── Bias Addition (fill from weight mem, add on drain) ───
    bias_add #(
        .ROWS   (ROWS),
        .COLS   (COLS),
        .ACC_W  (ACC_W),
        .WORD_W (WORD_W)
    ) u_bias_add (
        .clk       (clk),
        .rst_n     (rst_n),
        .fill_data (wmem_b_rdata),
        .fill_en   (bias_fill_en),
        .bias_en   (cfg_bias_en),
        .data_in   (acc_out),
        .col_idx   (acc_col_idx),
        .data_out  (biased_out)
    );

    // ── VPU (the non-linear domain) ──────────────────────────
    //   Inline: combinational activation on the MXU drain (biased_reg → act_out).
    //   Standalone: cmd_proc dispatches GO(func3) → the VPU streams the memories
    //     itself through the region-routed Port-B access above.
    vpu #(
        .ROWS   (ROWS),
        .DATA_W (DATA_W),
        .ACC_W  (ACC_W),
        .WORD_W (WORD_W),
        .ADDR_W (ADDR_W),
        .DIM_W  (DIM_W)
    ) u_vpu (
        .clk       (clk),
        .rst_n     (rst_n),
        // standalone ops (dispatched by cmd_proc)
        .start     (vpu_go),
        .op        (vpu_op),
        .M         (vpu_M),
        .N         (vpu_N),
        .src_base  (vpu_src),
        .src2_base (vpu_src2),
        .dst_base  (vpu_dst),
        .scale     (vpu_scale),
        .busy      (vpu_busy),
        .done      (vpu_done),
        // inline activation (MXU drain)
        .act_in    (biased_reg),
        .act_shift (cfg_shift),
        .act_sel   (cfg_act_sel),
        .act_leak  (cfg_leak_shift),
        .act_out   (act_out),
        // region-routed mem Port B (owned during a VPU op; top routes by region)
        .mem_region (vpu_region),
        .mem_addr   (vpu_mem_addr),
        .mem_en     (vpu_mem_en),
        .mem_we     (vpu_mem_we),
        .mem_wdata  (vpu_mem_wdata),
        .mem_rdata  (vpu_mem_rdata)
    );

    // ── Result Collect Source Mux (INT8 sign-extended byte vs INT32 raw) ──
    //   INT8 : sign-extend the activated byte to ACC_W (result_buffer writes the low byte).
    //   INT32: the R2-aligned raw biased accumulator (no shift / activation / saturate).
    always_comb begin
        for (int r = 0; r < ROWS; r++)
            result_collect[r] = cfg_out_fmt ? biased_reg_q2[r]
                                            : {{(ACC_W-DATA_W){act_reg[r][DATA_W-1]}}, act_reg[r]};
    end

    // ── Result Buffer (collect drain columns, write back words) ──
    result_buffer #(
        .ROWS   (ROWS),
        .COLS   (COLS),
        .DATA_W (DATA_W),
        .ACC_W  (ACC_W),
        .WORD_W (WORD_W)
    ) u_result_buffer (
        .clk            (clk),
        .rst_n          (rst_n),
        .result_in      (result_collect),
        .result_col_idx (acc_col_idx_q2),
        .result_valid   (acc_valid_q2),
        .out_fmt        (cfg_out_fmt),
        .collect_done   (rbuf_collect_done), // observed in sim; cmd_proc times S_WB
        .wb_en          (rbuf_wb_en),
        .wb_data        (rbuf_wb_data)
    );

    // ── Control FSM (inner per-tile engine) ──────────────────
    //   Driven by cmd_proc (cf_cmd/cf_start/cf_done): emits the weight + activation
    //   staging buffers into the array and runs the accumulator per tile.
    control_fsm u_control_fsm (
        .clk               (clk),
        .rst_n             (rst_n),
        .cmd               (cp_cf_cmd),
        .start             (cp_cf_start),
        .done              (cf_done),
        .wb_load_trigger   (fsm_wb_load_trigger),
        .wb_done           (fsm_wb_done),
        .ab_load_trigger   (fsm_ab_load_trigger),
        .ab_load_done      (fsm_ab_load_done),
        .acc_clear         (fsm_acc_clear),
        .acc_drain_trigger (fsm_acc_drain_trigger),
        .acc_pass_done     (fsm_acc_pass_done),
        .acc_drain_done    (fsm_acc_drain_done)
    );

endmodule

`default_nettype wire
