// =============================================================================
// control_fsm.sv  (the inner per-tile engine)
// Runs ONE primitive per start pulse, selected by cmd (from cmd_proc):
//   COMPUTE: emit the staged weight tile into the array (LOAD_W → WAIT_W), then
//            stream the staged activation tile (LOAD_A), then wait until the
//            accumulator has received the whole pass (COMPUTE_WAIT).
//   DRAIN  : run the accumulator drain (one column/cycle down the drain path).
//   CLEAR  : zero the accumulators (start of an output tile).
// done pulses one cycle as the FSM returns to IDLE. cmd_proc serializes around
// it: buffer fills / result writeback happen between starts (Port B free then).
// =============================================================================

`default_nettype none

module control_fsm (
    input  logic        clk,
    input  logic        rst_n,

    // ── Command interface (driven by cmd_proc) ───────────────────
    input  logic [1:0]  cmd,
    input  logic        start,
    output logic        done,

    // ── Weight buffer handshake (emit) ───────────────────────────
    output logic        wb_load_trigger,
    input  logic        wb_done,

    // ── Activation buffer handshake (emit) ───────────────────────
    output logic        ab_load_trigger,
    input  logic        ab_load_done,

    // ── Accumulator handshake ────────────────────────────────────
    output logic        acc_clear,
    output logic        acc_drain_trigger,
    input  logic        acc_pass_done,
    input  logic        acc_drain_done
);

    // ── Command Encoding ───────────────────────────────────────
    //   2'b10 is reserved (v1's STORE — unused in v2; cmd_proc's AGU
    //   does writeback, so it only ever issues COMPUTE / DRAIN / CLEAR).
    localparam logic [1:0] CMD_COMPUTE = 2'b00,
                           CMD_DRAIN   = 2'b01,
                           CMD_CLEAR   = 2'b11;

    // ── FSM States ─────────────────────────────────────────────
    typedef enum logic [2:0] {
        IDLE         = 3'b000,
        LOAD_W       = 3'b001,   // trigger the weight-buffer emit
        WAIT_W       = 3'b010,   // wait wb_done (16 per-column array loads)
        LOAD_A       = 3'b011,   // stream the activation tile into the array
        COMPUTE_WAIT = 3'b100,   // wait acc_pass_done (pass fully accumulated)
        DRAIN_RUN    = 3'b101,   // accumulator drain running
        CLEAR_RUN    = 3'b111    // 1-cycle accumulator clear
    } state_t;

    state_t state, state_next;

    // ── State Register ─────────────────────────────────────────
    always_ff @(posedge clk) begin
        if (!rst_n)
            state <= IDLE;
        else
            state <= state_next;
    end

    // ── Next-State Logic ───────────────────────────────────────
    always_comb begin
        state_next = state;

        case (state)
            IDLE: begin
                if (start) begin
                    case (cmd)
                        CMD_COMPUTE: state_next = LOAD_W;
                        CMD_DRAIN:   state_next = DRAIN_RUN;
                        CMD_CLEAR:   state_next = CLEAR_RUN;
                        default:     state_next = IDLE;
                    endcase
                end
            end

            LOAD_W: begin
                state_next = WAIT_W;
            end

            WAIT_W: begin
                if (wb_done)
                    state_next = LOAD_A;
            end

            LOAD_A: begin
                if (ab_load_done)
                    state_next = COMPUTE_WAIT;
            end

            COMPUTE_WAIT: begin
                if (acc_pass_done)
                    state_next = IDLE;
            end

            DRAIN_RUN: begin
                if (acc_drain_done)
                    state_next = IDLE;
            end

            CLEAR_RUN: begin
                state_next = IDLE;
            end

            default: state_next = IDLE;
        endcase
    end

    // ── Output Logic ───────────────────────────────────────────
    assign wb_load_trigger   = (state == LOAD_W);
    assign ab_load_trigger   = (state == LOAD_A);
    assign acc_clear         = (state == CLEAR_RUN);
    assign acc_drain_trigger = (state == DRAIN_RUN);

    assign done = (state != IDLE) && (state_next == IDLE);

endmodule

`default_nettype wire
