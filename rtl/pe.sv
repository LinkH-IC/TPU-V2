// =============================================================================
// pe.sv  (Processing Element — one MAC of the weight-stationary systolic array)
// Weight preloaded once (weight_load) and held stationary in weight_reg.
// Activations flow left→right (registered); partial sums flow top→bottom:
//     psum_out = psum_in + weight_reg · act_in,   gated by valid_in.
// INT8 × INT8 → 16-bit product, 32-bit accumulator. Latency: 1 cycle.
// =============================================================================

`default_nettype none

module pe #(
    parameter int DATA_W = 8,   // activation / weight width
    parameter int ACC_W  = 32   // partial-sum width
)(
    input  logic                clk,
    input  logic                rst_n,       // active-low synchronous reset

    // ── Weight preload ───────────────────────────────────────────
    input  logic signed [DATA_W-1:0] weight_in,
    input  logic                     weight_load,  // 1-cycle strobe: latch weight

    // ── Activation datapath (left → right) ───────────────────────
    input  logic signed [DATA_W-1:0] act_in,
    output logic signed [DATA_W-1:0] act_out,      // registered

    // ── Partial-sum datapath (top → bottom) ──────────────────────
    input  logic signed [ACC_W-1:0]  psum_in,
    output logic signed [ACC_W-1:0]  psum_out,

    // ── Flow control ─────────────────────────────────────────────
    input  logic valid_in,                          // qualifies act_in / psum_in
    output logic valid_out                          // registered, matches act_out
);

    logic signed [DATA_W-1:0]       weight_reg;  // stationary weight
    logic signed [2*DATA_W-1:0]     product;     // 16-bit multiply result
    logic signed [ACC_W-1:0]        psum_next;   // combinational accumulation

    // ── Weight register — latched once per tile ──────────────────
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            weight_reg <= '0;
        end else if (weight_load) begin
            weight_reg <= weight_in;
        end
    end

    // ── MAC — combinational multiply, registered into psum_out ───
    assign product   = weight_reg * act_in;        // signed × signed
    assign psum_next = psum_in + ACC_W'(product);  // sign-extend to 32-bit

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            psum_out <= '0;
        end else if (valid_in) begin
            psum_out <= psum_next;   // holds last value when valid_in low
        end
    end

    // ── Activation pipeline register (1-cycle skew → systolic wave) ──
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            act_out <= '0;
        end else begin
            act_out <= act_in;
        end
    end

    // ── Valid pipeline register (stays in phase with act_out) ────
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            valid_out <= 1'b0;
        end else begin
            valid_out <= valid_in;
        end
    end

endmodule

`default_nettype wire
