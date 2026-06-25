// =============================================================================
// mem.sv
// True dual-port on-chip memory (infers block RAM).
// - 128b word = one 16-byte column (one read = one lane-parallel column)
// - Synchronous read -> 1-cycle latency (infers BRAM; AGU runs one step ahead)
// - Read-first (no bypass), full-word writes, no byte-enables
// - No reset: memory contents aren't resettable (bench writes before reading)
// - Port A = host R/W, Port B = engine R/W (cmd_proc / VPU, muxed in top);
//   each owns a port -> no arbiter
// - Never write the same address from both ports the same cycle (undefined)
// =============================================================================

`default_nettype none

module mem #(
    parameter  int DATA_W = 8,            // byte width of one lane
    parameter  int LANES  = 16,           // lanes per word (= one column)
    parameter  int DEPTH  = 4096,         // words; capacity = DEPTH * WORD_W/8 B
    localparam int WORD_W = LANES*DATA_W, // 128b word
    localparam int ADDR_W = $clog2(DEPTH)
)(
    input  logic                clk,

    // ── Port A — host read/write ─────────────────────────────────
    input  logic [ADDR_W-1:0]   a_addr,
    input  logic                a_en,     // port active (clock-enable on read reg)
    input  logic                a_we,     // write enable (whole word)
    input  logic [WORD_W-1:0]   a_wdata,
    output logic [WORD_W-1:0]   a_rdata,  // registered, valid 1 cycle after a_en

    // ── Port B — engine read/write (cmd_proc / VPU) ──────────────
    input  logic [ADDR_W-1:0]   b_addr,
    input  logic                b_en,
    input  logic                b_we,
    input  logic [WORD_W-1:0]   b_wdata,
    output logic [WORD_W-1:0]   b_rdata
);

    // ── Storage ──────────────────────────────────────────────────
    logic [WORD_W-1:0] ram [DEPTH];

    // ── Port A ───────────────────────────────────────────────────
    // Read-first: a_rdata samples ram[a_addr] before any same-cycle write
    // commits (both are nonblocking -> RHS reads the old value).
    always_ff @(posedge clk) begin
        if (a_en) begin
            if (a_we)
                ram[a_addr] <= a_wdata;
            a_rdata <= ram[a_addr];
        end
    end

    // ── Port B ───────────────────────────────────────────────────
    always_ff @(posedge clk) begin
        if (b_en) begin
            if (b_we)
                ram[b_addr] <= b_wdata;
            b_rdata <= ram[b_addr];
        end
    end

endmodule

`default_nettype wire
