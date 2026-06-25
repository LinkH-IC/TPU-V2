// =============================================================================
// weight_buffer.sv  (v2 — memory-fed)
// One-tile staging buffer between the weight memory and the systolic array.
//   FILL side (NEW): cmd_proc's AGU streams 16 words from weight mem (Port B);
//     each fill_en latches one word (one tile row = COLS N-values) into staging.
//   EMIT side (unchanged from v1): on load_trigger, sequence the 16 per-column
//     array loads (weight_out + weight_load[c]).
//   staging[r][c] = B[k*16+r][j*16+c]; emit reads it by column (fixed seq_cnt,
//     varying r) — the array's column-load orientation falls out of indexing.
//   Single-buffer: cannot refill while emitting (fill_ready low during emit).
//   Double-buffering (fill next tile under compute) is a later optimization.
// =============================================================================

`default_nettype none

module weight_buffer #(
    parameter ROWS   = 16,
    parameter COLS   = 16,
    parameter DATA_W = 8
)(
    input  logic                            clk,
    input  logic                            rst_n,

    // ── Fill interface — from weight memory Port B (driven by cmd_proc) ──
    input  logic [COLS-1:0][DATA_W-1:0]     fill_data,   // one word = one tile row (COLS values)
    input  logic                            fill_en,     // latch this word into the next staging slot
    output logic                            fill_ready,  // staging free to accept a tile

    // ── Control interface (emit) — unchanged from v1 ────────────────────
    input  logic                            load_trigger,
    output logic                            done,         // pulse: array loaded

    // ── Systolic array interface — unchanged ────────────────────────────
    output logic [ROWS-1:0][DATA_W-1:0]     weight_out,   // weight bus to array
    output logic [COLS-1:0]                 weight_load   // per-column load strobe
);

    // ── Internal Parameters ──────────────────────────────────────
    localparam CNT_W   = $clog2(COLS);
    localparam CNT_END = CNT_W'(COLS - 1);

    // ── FSM States ───────────────────────────────────────────────
    typedef enum logic {
        IDLE     = 1'b0,
        SEQUENCE = 1'b1
    } state_t;

    state_t              state, state_next;
    logic [CNT_W-1:0]    seq_cnt, seq_cnt_next;
    logic [CNT_W-1:0]    fill_cnt;

    // ── Storage — one tile: staging[r][c] = B[k*16+r][j*16+c] ─────
    logic [DATA_W-1:0]   staging [ROWS][COLS];

    // ── State Register ───────────────────────────────────────────
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state   <= IDLE;
            seq_cnt <= '0;
        end else begin
            state   <= state_next;
            seq_cnt <= seq_cnt_next;
        end
    end

    // ── Next-State Logic ─────────────────────────────────────────
    always_comb begin
        state_next   = state;
        seq_cnt_next = '0;

        case (state)
            IDLE: begin
                if (load_trigger)
                    state_next = SEQUENCE;
            end

            SEQUENCE: begin
                if (seq_cnt == CNT_END)
                    state_next = IDLE;
                else
                    seq_cnt_next = seq_cnt + CNT_W'(1);
            end

            default: state_next = IDLE;
        endcase
    end

    // ── Fill — latch one tile row per fill_en (word w -> staging row w) ──
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            fill_cnt <= '0;
            for (int r = 0; r < ROWS; r++)
                for (int c = 0; c < COLS; c++)
                    staging[r][c] <= '0;
        end else if (fill_en) begin
            for (int c = 0; c < COLS; c++)
                staging[fill_cnt][c] <= fill_data[c];
            fill_cnt <= (fill_cnt == CNT_END) ? '0 : fill_cnt + CNT_W'(1);
        end
    end

    // ── Emit — drive one array column per cycle ──────────────────
    always_comb begin
        weight_out  = '0;
        weight_load = '0;

        if (state == SEQUENCE) begin
            for (int r = 0; r < ROWS; r++)
                weight_out[r] = staging[r][seq_cnt];
            weight_load[seq_cnt] = 1'b1;
        end
    end

    // ── Handshake Outputs ────────────────────────────────────────
    assign fill_ready = (state == IDLE);
    assign done       = (state == SEQUENCE) && (seq_cnt == CNT_END);

endmodule

`default_nettype wire
