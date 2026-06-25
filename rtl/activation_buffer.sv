// =============================================================================
// activation_buffer.sv  (v2 — memory-fed)
// One-tile staging buffer: activation memory -> systolic array.
//   FILL: cmd_proc's AGU streams 16 words from activation mem (Port B); each
//     fill_en latches one word (one tile row = COLS K-values) into staging.
//   EMIT: on load_trigger, stream the tile to the array one ROW per cycle
//     (act_out[r]=staging[seq_cnt][r]); the array self-skews. (Contrast the
//     weight buffer, which emits by COLUMN — that's the only difference.)
//   staging[m][c] = A[i*16+m][k*16+c]; emit reads it by row.
//   Single-buffer: fill_ready low while emitting.
// =============================================================================

`default_nettype none

module activation_buffer #(
    parameter ROWS   = 16,
    parameter COLS   = 16,
    parameter DATA_W = 8
)(
    input  logic                            clk,
    input  logic                            rst_n,

    // ── Fill interface — from activation memory Port B (driven by cmd_proc) ──
    input  logic [COLS-1:0][DATA_W-1:0]     fill_data,   // one word = one tile row (COLS values)
    input  logic                            fill_en,
    output logic                            fill_ready,

    // ── Control interface (emit) — unchanged role from v1 ───────────────────
    input  logic                            load_trigger,
    output logic                            load_done,

    // ── Systolic array interface ────────────────────────────────────────────
    output logic [ROWS-1:0][DATA_W-1:0]     act_out,
    output logic [ROWS-1:0]                 valid
);

    // ── Internal Parameters ──────────────────────────────────────
    localparam CNT_W   = $clog2(COLS);
    localparam CNT_END = CNT_W'(COLS - 1);

    // ── FSM States ───────────────────────────────────────────────
    typedef enum logic {
        IDLE   = 1'b0,
        STREAM = 1'b1
    } state_t;

    state_t              state, state_next;
    logic [CNT_W-1:0]    seq_cnt, seq_cnt_next;
    logic [CNT_W-1:0]    fill_cnt;

    // ── Storage — one tile: staging[m][c] = A[i*16+m][k*16+c] ─────
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
                    state_next = STREAM;
            end

            STREAM: begin
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

    // ── Emit — drive one tile row per cycle to the array ─────────
    always_comb begin
        act_out = '0;
        valid   = '0;

        if (state == STREAM) begin
            for (int r = 0; r < ROWS; r++)
                act_out[r] = staging[seq_cnt][r];
            valid = '1;
        end
    end

    // ── Handshake Outputs ────────────────────────────────────────
    assign fill_ready = (state == IDLE);
    assign load_done  = (state == STREAM) && (seq_cnt == CNT_END);

endmodule

`default_nettype wire
