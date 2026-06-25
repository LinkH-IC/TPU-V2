// =============================================================================
// bias_add.sv  (v2 — memory-fed)
// Per-column bias add on the drain path: data_out[r] = data_in[r] + bias[col].
// Input side rebuilt: v1's host write port -> AGU fill from WEIGHT memory.
//   Bias lives in weight mem (stationary like weights), packed BPW=WORD_W/ACC_W
//   biases per word. cmd_proc reads NWORDS = COLS/BPW words for the current
//   j-tile (before drain, when Port B is free) — each fill_en latches one word
//   (BPW packed biases, bias[base+p] in bits [p*ACC_W +: ACC_W]) into bias_reg.
// Datapath unchanged from v1.
// =============================================================================

`default_nettype none

module bias_add #(
    parameter ROWS   = 16,
    parameter COLS   = 16,
    parameter ACC_W  = 32,
    parameter WORD_W = 128            // memory word width (= LANES*DATA_W)
)(
    input  logic                        clk,
    input  logic                        rst_n,

    // ── Fill interface — from weight memory Port B (driven by cmd_proc) ──
    input  logic [WORD_W-1:0]           fill_data,   // one word = BPW packed biases
    input  logic                        fill_en,

    // ── Datapath — combinational add ────────────────────────────
    input  logic                        bias_en,     // 1 = add bias, 0 = add 0 (pure A·B)
    input  logic [ROWS-1:0][ACC_W-1:0]  data_in,
    input  logic [$clog2(COLS)-1:0]     col_idx,
    output logic [ROWS-1:0][ACC_W-1:0]  data_out
);

    // ── Derived ──────────────────────────────────────────────────
    localparam BPW      = WORD_W / ACC_W;          // biases per word (=4)
    localparam NWORDS   = COLS / BPW;              // words per tile (=4)
    localparam FCNT_W   = $clog2(NWORDS);
    localparam FCNT_END = FCNT_W'(NWORDS - 1);

    logic [FCNT_W-1:0]        fill_cnt;
    logic signed [ACC_W-1:0]  bias_reg [COLS];

    // ── Fill — latch BPW biases per fill_en ──────────────────────
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            fill_cnt <= '0;
            for (int c = 0; c < COLS; c++)
                bias_reg[c] <= '0;
        end else if (fill_en) begin
            for (int p = 0; p < BPW; p++)
                bias_reg[int'(fill_cnt)*BPW + p] <= fill_data[p*ACC_W +: ACC_W];
            fill_cnt <= (fill_cnt == FCNT_END) ? '0 : fill_cnt + FCNT_W'(1);
        end
    end

    // ── Combinational Bias Addition ──────────────────────────────
    //   Every row in the current drain column gets the same bias.
    //   bias_en=0 selects 0 -> data_out = data_in (pure matmul: scores / gradients).
    logic signed [ACC_W-1:0]  bias_sel;
    always_comb begin
        bias_sel = bias_en ? bias_reg[col_idx] : '0;
        for (int r = 0; r < ROWS; r++)
            data_out[r] = data_in[r] + bias_sel;
    end

endmodule

`default_nettype wire
