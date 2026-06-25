// =============================================================================
// result_buffer.sv  (v2 — INT8 / INT32 dual-format writeback)
// Transpose buffer: systolic-array drain -> activation/result memory.
//   The accumulator drains one COLUMN per cycle (result_in = a result column,
//   result_col_idx = which N-column), but C is stored row-major (word = a C
//   row). So collect the 16 columns, then write back row by row.
//   COLLECT: result_staging[r][result_col_idx] <= result_in[r] on result_valid.
//     result_in is ACC_W wide: the INT8 path feeds sign-extended act bytes, the
//     INT32 path feeds the raw biased accumulator (top selects on out_fmt).
//   WRITEBACK depends on out_fmt:
//     INT8  — one 128b word per C row  (16 lanes × 8b)            -> 16 words/tile
//     INT32 — four 128b words per C row (4 lanes × 32b each)      -> 64 words/tile
//     The AGU drives the matching word address (×16 vs ×64 tile stride).
//   Single-buffer: collect (drain) and writeback are serialized by cmd_proc.
// =============================================================================

`default_nettype none

module result_buffer #(
    parameter ROWS   = 16,
    parameter COLS   = 16,
    parameter DATA_W = 8,
    parameter ACC_W  = 32,            // raw INT32 element width
    parameter WORD_W = 128            // memory word width (= COLS*DATA_W)
)(
    input  logic                            clk,
    input  logic                            rst_n,

    // ── Collect interface — from the drain / activation pipeline ────────────
    input  logic [ROWS-1:0][ACC_W-1:0]      result_in,      // INT8: sign-extended byte; INT32: raw accumulator
    input  logic [$clog2(COLS)-1:0]         result_col_idx,
    input  logic                            result_valid,
    input  logic                            out_fmt,        // 0 = INT8 (1 word/row), 1 = INT32 raw (4 words/row)
    output logic                            collect_done,   // pulse: last column captured

    // ── Writeback interface — to activation/result memory Port B (cmd_proc) ──
    input  logic                            wb_en,          // advance to next word
    output logic [WORD_W-1:0]               wb_data         // current word -> memory wdata
);

    // ── Internal Parameters ──────────────────────────────────────
    localparam int SUBW   = WORD_W / ACC_W;       // INT32 lanes per word (=4)
    localparam int WB_I8  = COLS;                 // INT8  words/tile (one C row each) = 16
    localparam int WB_I32 = ROWS * SUBW;          // INT32 words/tile (4 subwords × 16 rows) = 64
    localparam int CNT_W  = $clog2(WB_I32);       // writeback word counter (0..63)
    localparam int ROW_W  = $clog2(ROWS);         // C-row index width
    localparam int SUB_W  = $clog2(SUBW);         // subword index width (=2)
    localparam int COL_W  = $clog2(COLS);         // collect column index width

    logic [CNT_W-1:0]    wb_cnt;
    wire  [CNT_W-1:0]    wb_end = out_fmt ? CNT_W'(WB_I32 - 1)    // INT32: 64 words/tile
                                          : CNT_W'(WB_I8  - 1);   // INT8 : 16 words/tile

    // ── Storage — one tile: result_staging[m][n] = C[i*16+m][j*16+n] ──
    logic [ACC_W-1:0]    result_staging [ROWS][COLS];

    // ── Collect — capture one result column per result_valid ─────
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            for (int r = 0; r < ROWS; r++)
                for (int c = 0; c < COLS; c++)
                    result_staging[r][c] <= '0;
        end else if (result_valid) begin
            for (int r = 0; r < ROWS; r++)
                result_staging[r][result_col_idx] <= result_in[r];
        end
    end

    // ── Writeback Word Counter ───────────────────────────────────
    always_ff @(posedge clk) begin
        if (!rst_n)
            wb_cnt <= '0;
        else if (wb_en)
            wb_cnt <= (wb_cnt == wb_end) ? '0 : wb_cnt + CNT_W'(1);
    end

    // ── Word -> (row, subword) decode ────────────────────────────
    //   INT8 : the whole word is C-row wb_cnt (0..15).
    //   INT32: 4 words per row -> row = wb_cnt>>2, subword = wb_cnt[1:0].
    wire [ROW_W-1:0] wb_row = out_fmt ? wb_cnt[SUB_W +: ROW_W]   // INT32: wb_cnt / 4
                                      : wb_cnt[0     +: ROW_W];  // INT8 : wb_cnt
    wire [SUB_W-1:0] wb_sub = wb_cnt[SUB_W-1:0];                 // INT32: which group of 4 columns

    // ── Writeback Data — present one memory word (combinational) ─
    always_comb begin
        wb_data = '0;
        if (out_fmt) begin
            // INT32: 4 raw 32b lanes, columns [wb_sub*4 .. wb_sub*4+3]
            for (int p = 0; p < SUBW; p++)
                wb_data[p*ACC_W +: ACC_W] = result_staging[wb_row][int'(wb_sub)*SUBW + p];
        end
        else begin
            // INT8: low byte of all 16 lanes of one C row
            for (int c = 0; c < COLS; c++)
                wb_data[c*DATA_W +: DATA_W] = result_staging[wb_row][c][DATA_W-1:0];
        end
    end

    // ── Handshake Output ─────────────────────────────────────────
    assign collect_done = result_valid && (result_col_idx == COL_W'(COLS - 1));

endmodule

`default_nettype wire
