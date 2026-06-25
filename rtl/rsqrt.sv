// =============================================================================
// rsqrt.sv  (VPU primitive — fixed-point reciprocal square root, for LayerNorm)
// Computes  rs ≈ 2^24 / √d  via a seed LUT + one Newton-Raphson step (~6 cycles).
// S28: the Newton step is cut in two (RSUB registers the saturating subtract)
// so each big multiply (→ DSP48) sits reg-to-reg — same values, +1 cycle.
// Used once per LayerNorm row on the Q16 variance:  rs = 2^24/(σ·2^8) = 2^16/σ,
// so the normalized value is  n_q6 = (d_q8 · rs) >>> 18 = (x−μ)/σ · 2^6.
//
//   d = m · 4^p  with  m ∈ [1,4) as Q16  (p = e>>1, e = MSB position), so
//   1/√d = (1/√m) · 2^−p  and only 1/√m over one octave-pair needs the LUT.
//   seed:  y0 = SEED[m>>10] ≈ 2^8/√m          (idx ∈ [64,255] ⇔ m ∈ [1,4))
//   Newton (quadratic): y1 = y0·(3 − m·y0²)/2  → Q16, ~13-bit accurate
//   scale: rs = y1 << (8−p)   (>> (p−8) when p > 8)
//
//   start pulses a compute; done pulses one cycle when rs is valid.
//   (Assumes d ≥ 1 — guaranteed by LayerNorm's +1 ε on the variance.)
// =============================================================================

`default_nettype none

module rsqrt #(
    parameter int D_W  = 32,    // radicand width (Q16 variance)
    parameter int RS_W = 25     // result width (rs ≤ 2^24 when d ≥ 1)
)(
    input  logic            clk,
    input  logic            rst_n,
    input  logic            start,
    input  logic [D_W-1:0]  d,     // σ² as Q16  (≥ 1)
    output logic [RS_W-1:0] rs,    // ≈ 2^24 / √d
    output logic            busy,
    output logic            done
);

    // ── seed LUT: SEED[i] = round(2048/√(i+0.5)) ≈ 2^8/√m at the bin center ──
    //   i = m[17:10] ∈ [64,255] ⇔ m ∈ [1,4); entries < 64 unreachable (m ≥ 2^16).
    localparam logic [7:0] SEED [0:255] = '{
        8'd0, 8'd0, 8'd0, 8'd0, 8'd0, 8'd0, 8'd0, 8'd0,
        8'd0, 8'd0, 8'd0, 8'd0, 8'd0, 8'd0, 8'd0, 8'd0,
        8'd0, 8'd0, 8'd0, 8'd0, 8'd0, 8'd0, 8'd0, 8'd0,
        8'd0, 8'd0, 8'd0, 8'd0, 8'd0, 8'd0, 8'd0, 8'd0,
        8'd0, 8'd0, 8'd0, 8'd0, 8'd0, 8'd0, 8'd0, 8'd0,
        8'd0, 8'd0, 8'd0, 8'd0, 8'd0, 8'd0, 8'd0, 8'd0,
        8'd0, 8'd0, 8'd0, 8'd0, 8'd0, 8'd0, 8'd0, 8'd0,
        8'd0, 8'd0, 8'd0, 8'd0, 8'd0, 8'd0, 8'd0, 8'd0,
        8'd255, 8'd253, 8'd251, 8'd249, 8'd247, 8'd246, 8'd244, 8'd242,
        8'd241, 8'd239, 8'd237, 8'd236, 8'd234, 8'd233, 8'd231, 8'd230,
        8'd228, 8'd227, 8'd225, 8'd224, 8'd223, 8'd221, 8'd220, 8'd219,
        8'd218, 8'd216, 8'd215, 8'd214, 8'd213, 8'd212, 8'd211, 8'd210,
        8'd208, 8'd207, 8'd206, 8'd205, 8'd204, 8'd203, 8'd202, 8'd201,
        8'd200, 8'd199, 8'd198, 8'd198, 8'd197, 8'd196, 8'd195, 8'd194,
        8'd193, 8'd192, 8'd191, 8'd191, 8'd190, 8'd189, 8'd188, 8'd187,
        8'd187, 8'd186, 8'd185, 8'd184, 8'd184, 8'd183, 8'd182, 8'd181,
        8'd181, 8'd180, 8'd179, 8'd179, 8'd178, 8'd177, 8'd177, 8'd176,
        8'd175, 8'd175, 8'd174, 8'd173, 8'd173, 8'd172, 8'd172, 8'd171,
        8'd170, 8'd170, 8'd169, 8'd169, 8'd168, 8'd167, 8'd167, 8'd166,
        8'd166, 8'd165, 8'd165, 8'd164, 8'd164, 8'd163, 8'd163, 8'd162,
        8'd162, 8'd161, 8'd161, 8'd160, 8'd160, 8'd159, 8'd159, 8'd158,
        8'd158, 8'd157, 8'd157, 8'd156, 8'd156, 8'd155, 8'd155, 8'd155,
        8'd154, 8'd154, 8'd153, 8'd153, 8'd152, 8'd152, 8'd152, 8'd151,
        8'd151, 8'd150, 8'd150, 8'd150, 8'd149, 8'd149, 8'd148, 8'd148,
        8'd148, 8'd147, 8'd147, 8'd146, 8'd146, 8'd146, 8'd145, 8'd145,
        8'd145, 8'd144, 8'd144, 8'd144, 8'd143, 8'd143, 8'd143, 8'd142,
        8'd142, 8'd141, 8'd141, 8'd141, 8'd140, 8'd140, 8'd140, 8'd140,
        8'd139, 8'd139, 8'd139, 8'd138, 8'd138, 8'd138, 8'd137, 8'd137,
        8'd137, 8'd136, 8'd136, 8'd136, 8'd135, 8'd135, 8'd135, 8'd135,
        8'd134, 8'd134, 8'd134, 8'd133, 8'd133, 8'd133, 8'd133, 8'd132,
        8'd132, 8'd132, 8'd132, 8'd131, 8'd131, 8'd131, 8'd130, 8'd130,
        8'd130, 8'd130, 8'd129, 8'd129, 8'd129, 8'd129, 8'd128, 8'd128
    };

    typedef enum logic [2:0] {
        IDLE,
        NORM,
        SEEDST,
        SQ,
        TMUL,
        RSUB,
        NEWT
    } state_t;
    state_t state;

    logic [D_W-1:0] d_r;           // latched radicand
    logic [3:0]     p;             // exponent-pair count (e>>1, ≤ 15)
    logic [17:0]    m;             // normalized mantissa, Q16 ∈ [2^16, 2^18)
    logic [7:0]     y0;            // seed ≈ 2^8/√m
    logic [15:0]    yy;            // y0²  (8×8 → DSP)
    logic [33:0]    t;             // m·y0²  (Q32 ≈ 1.0; 18×16 → DSP)
    logic [33:0]    r_q;           // staged 3·2^32 − t (saturating subtract)

    // ── normalize (combinational from the latched radicand) ──────
    logic [3:0] p_c;                                   // exponent-pair count p = MSB>>1
    always_comb begin
        p_c = 4'd0;
        for (int b = 0; b < D_W; b++)
            if (d_r[b])
                p_c = 4'(b >> 1);
    end
    wire [17:0] m_c = (p_c >= 4'd8) ? 18'(d_r >> (5'(p_c) << 1) - 5'd16)
                                    : 18'(d_r << 5'd16 - (5'(p_c) << 1));

    // ── Newton: y1 = y0·(3·2^32 − t) >> 25 → Q16, clamp to 1.0 ───
    //   Insurance: t ≈ 2^32 in-contract; the saturating subtract keeps a
    //   pathological t (> 3·2^32) from wrapping — r=0 → rs=0, never corruption.
    wire [33:0] r  = (t > 34'h3_0000_0000) ? 34'd0 : (34'h3_0000_0000 - t);
    wire [41:0] yr = 42'(y0) * 42'(r_q);                            // Newton product (8×34 → DSP)
    wire [16:0] y1_raw = 17'(yr >> 25);
    wire [16:0] y1 = (y1_raw > 17'h1_0000) ? 17'h1_0000 : y1_raw;   // ≤ 1.0 Q16

    // ── scale: rs = y1 · 2^(8−p)  (1/√d = (1/√m)·2^−p) ───────────
    wire [24:0] rs_c = (p <= 4'd8) ? 25'(y1) << (4'd8 - p)
                                   : 25'(y1  >> (p - 4'd8));

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state <= IDLE;
            d_r   <= '0;
            p     <= '0;
            m     <= '0;
            y0    <= '0;
            yy    <= '0;
            t     <= '0;
            r_q   <= '0;
            rs    <= '0;
            done  <= 1'b0;
        end else begin
            done <= 1'b0;
            case (state)
                IDLE: begin
                    if (start) begin
                        d_r   <= d;
                        state <= NORM;
                    end
                end
                NORM: begin
                    p     <= p_c;
                    m     <= m_c;
                    state <= SEEDST;
                end
                SEEDST: begin
                    y0    <= SEED[m[17:10]];
                    state <= SQ;
                end
                SQ: begin
                    yy    <= 16'(y0) * 16'(y0);
                    state <= TMUL;
                end
                TMUL: begin
                    t     <= 34'(m) * 34'(yy);
                    state <= RSUB;
                end
                RSUB: begin
                    r_q   <= r;                        // stage the subtract off the multiply path
                    state <= NEWT;
                end
                NEWT: begin
                    rs    <= rs_c;
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
