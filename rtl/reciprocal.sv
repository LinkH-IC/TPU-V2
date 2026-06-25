// =============================================================================
// reciprocal.sv  (VPU primitive — fixed-point reciprocal of the softmax sum)
// Computes  recip ≈ 2^32 / sum  via a seed LUT + one Newton-Raphson step, so it
// finishes in ~5 cycles (was a 34-cycle sequential divide). Used once per softmax
// row; the result feeds  q = round(127·e·recip / 2^32) = round(127·e/Σ).
// S28: the Newton step is cut in two (TSUB registers the saturating subtract)
// so each big multiply (→ DSP48) sits reg-to-reg — same values, +1 cycle.
//
//   sum = 2^e · (1+f),  e = MSB position (Σ ≥ 2^15 ⇒ e ≥ 15).
//   seed:  rm = SEEDLUT[idx] ≈ 2^8/(1+f)  (idx = 8 frac bits below MSB),
//          x0 = rm << (24−e) ≈ 2^32/sum         (≈ 8-bit accurate)
//   Newton (quadratic): x1 = x0·(2^33 − sum·x0) >> 32  ≈ 2^32/sum (~16-bit)
//
//   start pulses a compute; done pulses one cycle when recip is valid.
//   (Assumes sum ≥ 2^15 — true in softmax: the max element's exp = 1.0 = 0x8000.)
// =============================================================================

`default_nettype none

module reciprocal #(
    parameter int SUM_W   = 32,    // divisor width
    parameter int RECIP_W = 32     // quotient output width
)(
    input  logic               clk,
    input  logic               rst_n,
    input  logic               start,
    input  logic [SUM_W-1:0]   sum,      // Σ e_j  (≥ 2^15)
    output logic [RECIP_W-1:0] recip,    // ≈ 2^32 / sum
    output logic               busy,
    output logic               done
);

    // ── seed LUT: rm[idx] = round(2^16 / (256+idx)) ≈ 2^8/(1+f), 9-bit [128,256] ─
    localparam logic [8:0] SEEDLUT [0:255] = '{
        9'd256, 9'd255, 9'd254, 9'd253, 9'd252, 9'd251, 9'd250, 9'd249,
        9'd248, 9'd247, 9'd246, 9'd245, 9'd245, 9'd244, 9'd243, 9'd242,
        9'd241, 9'd240, 9'd239, 9'd238, 9'd237, 9'd237, 9'd236, 9'd235,
        9'd234, 9'd233, 9'd232, 9'd232, 9'd231, 9'd230, 9'd229, 9'd228,
        9'd228, 9'd227, 9'd226, 9'd225, 9'd224, 9'd224, 9'd223, 9'd222,
        9'd221, 9'd221, 9'd220, 9'd219, 9'd218, 9'd218, 9'd217, 9'd216,
        9'd216, 9'd215, 9'd214, 9'd213, 9'd213, 9'd212, 9'd211, 9'd211,
        9'd210, 9'd209, 9'd209, 9'd208, 9'd207, 9'd207, 9'd206, 9'd205,
        9'd205, 9'd204, 9'd204, 9'd203, 9'd202, 9'd202, 9'd201, 9'd200,
        9'd200, 9'd199, 9'd199, 9'd198, 9'd197, 9'd197, 9'd196, 9'd196,
        9'd195, 9'd194, 9'd194, 9'd193, 9'd193, 9'd192, 9'd192, 9'd191,
        9'd191, 9'd190, 9'd189, 9'd189, 9'd188, 9'd188, 9'd187, 9'd187,
        9'd186, 9'd186, 9'd185, 9'd185, 9'd184, 9'd184, 9'd183, 9'd183,
        9'd182, 9'd182, 9'd181, 9'd181, 9'd180, 9'd180, 9'd179, 9'd179,
        9'd178, 9'd178, 9'd177, 9'd177, 9'd176, 9'd176, 9'd175, 9'd175,
        9'd174, 9'd174, 9'd173, 9'd173, 9'd172, 9'd172, 9'd172, 9'd171,
        9'd171, 9'd170, 9'd170, 9'd169, 9'd169, 9'd168, 9'd168, 9'd168,
        9'd167, 9'd167, 9'd166, 9'd166, 9'd165, 9'd165, 9'd165, 9'd164,
        9'd164, 9'd163, 9'd163, 9'd163, 9'd162, 9'd162, 9'd161, 9'd161,
        9'd161, 9'd160, 9'd160, 9'd159, 9'd159, 9'd159, 9'd158, 9'd158,
        9'd158, 9'd157, 9'd157, 9'd156, 9'd156, 9'd156, 9'd155, 9'd155,
        9'd155, 9'd154, 9'd154, 9'd153, 9'd153, 9'd153, 9'd152, 9'd152,
        9'd152, 9'd151, 9'd151, 9'd151, 9'd150, 9'd150, 9'd150, 9'd149,
        9'd149, 9'd149, 9'd148, 9'd148, 9'd148, 9'd147, 9'd147, 9'd147,
        9'd146, 9'd146, 9'd146, 9'd145, 9'd145, 9'd145, 9'd144, 9'd144,
        9'd144, 9'd143, 9'd143, 9'd143, 9'd142, 9'd142, 9'd142, 9'd142,
        9'd141, 9'd141, 9'd141, 9'd140, 9'd140, 9'd140, 9'd139, 9'd139,
        9'd139, 9'd139, 9'd138, 9'd138, 9'd138, 9'd137, 9'd137, 9'd137,
        9'd137, 9'd136, 9'd136, 9'd136, 9'd135, 9'd135, 9'd135, 9'd135,
        9'd134, 9'd134, 9'd134, 9'd133, 9'd133, 9'd133, 9'd133, 9'd132,
        9'd132, 9'd132, 9'd132, 9'd131, 9'd131, 9'd131, 9'd131, 9'd130,
        9'd130, 9'd130, 9'd130, 9'd129, 9'd129, 9'd129, 9'd129, 9'd128
    };

    typedef enum logic [2:0] {
        IDLE,
        SEED,
        MUL1,
        TSUB,
        NEWT
    } state_t;
    state_t state;

    logic [SUM_W-1:0]  d;          // latched divisor
    logic [17:0]       x0;         // seed ≈ 2^32/d  (≤ 2^17)
    logic [63:0]       dx0;        // d · x0  (≈ 2^32) — 32×18 multiply → DSP
    logic [33:0]       t_q;        // staged 2^33 − d·x0 (value ≤ 2^33 → 34 bits;
                                   //  keeps the Newton multiply 18×34, not 18×64)

    // ── seed (combinational from the latched divisor d) ──────────
    logic [5:0] msb;                                   // MSB position e
    always_comb begin
        msb = 6'd0;
        for (int b = 0; b < SUM_W; b++)
            if (d[b])
                msb = b[5:0];
    end
    wire [4:0]  sh_idx = 5'(msb - 6'd8);                // e−8 (≥7)
    wire [7:0]  idx    = 8'(d >> sh_idx);                // 8 frac bits below MSB
    wire [8:0]  rm     = SEEDLUT[idx];                  // ≈ 2^8/(1+f)
    wire [4:0]  sh_x0  = 5'd24 - msb[4:0];               // 24−e ∈ [0,9]
    wire [17:0] x0_c   = 18'(rm) << sh_x0;               // ≈ 2^32/d

    // ── Newton: x1 = x0·(2^33 − d·x0) >> 32 ──────────────────────
    //   Insurance: dx0 ≈ 2^32 in-contract (sum∈[2^15,2^24], ~2× margin to 2^33),
    //   but a SATURATING subtract keeps an out-of-contract input from wrapping the
    //   64-bit unsigned t to ~2^64 (which would blow past the NORM sat8). Clamp →
    //   t=0 → recip=0 (graceful all-zeros), never corruption. Dead in normal use.
    wire [33:0] t  = (dx0 > 64'h2_0000_0000) ? 34'd0 : 34'(64'h2_0000_0000 - dx0);
    wire [63:0] xt = 64'(x0) * 64'(t_q);                // Newton product (18×34 → DSP)
    wire [31:0] x1 = 32'(xt >> 32);                     // ≈ 2^32/d  (≤ 2^18)

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state <= IDLE;
            d     <= '0;
            x0    <= '0;
            dx0   <= '0;
            t_q   <= '0;
            recip <= '0;
            done  <= 1'b0;
        end else begin
            done <= 1'b0;
            case (state)
                IDLE: begin
                    if (start) begin
                        d     <= sum;
                        state <= SEED;
                    end
                end
                SEED: begin
                    x0    <= x0_c;
                    state <= MUL1;
                end
                MUL1: begin
                    dx0   <= 64'(d) * 64'(x0);
                    state <= TSUB;
                end
                TSUB: begin
                    t_q   <= t;                        // stage the subtract off the multiply path
                    state <= NEWT;
                end
                NEWT: begin
                    recip <= RECIP_W'(x1);
                    done  <= 1'b1;
                    state <= IDLE;
                end
                default: state <= IDLE;
            endcase
        end
    end

    assign busy = (state != IDLE);

endmodule

`default_nettype wire
