// =============================================================================
// gelu_lut.sv  (inline GELU activation — 256-entry INT8 ROM)
// Combinational lookup for the VPU inline-activation bank (act_sel = GELU).
//
//   Unlike ReLU, GELU is NOT scale-invariant (gelu(k·x) ≠ k·gelu(x)), so the
//   table is fixed-point: the 8-bit input `idx` is a signed activation with
//   GELU_FBITS fractional bits (K = 2^F), and
//       ROM[u] = sat8( round( K · gelu(s/K) ) ),   s = signed(u),
//   with exact GELU  gelu(x) = 0.5·x·(1 + erf(x/√2)).
//
//   `idx` is the raw 8-bit two's-complement of the post-shift activation
//   (clamped to [-128,127] by the caller), used directly as the ROM address:
//   u = 0..127 → s = 0..127 (real 0..7.9375);  u = 128..255 → s = -128..-1.
//
//   The ROM literals below are generated for GELU_FBITS = 4. Because $erf is a
//   simulation-only function (not synthesizable), the table is hard-coded — to
//   retune F, regenerate the array (Python: floor(K·gelu(s/K)+0.5), K=2^F). The
//   parameter documents the assumed scale; the caller's requant `act_shift` must
//   land activations at the same Q·F scale.
// =============================================================================

`default_nettype none

module gelu_lut #(
    parameter int GELU_FBITS = 4,      // fractional bits of the INT8 activation (K = 2^F); ROM tuned for 4
    parameter int DATA_W     = 8
)(
    input  logic [DATA_W-1:0]        idx,   // signed post-shift activation, raw 8b (unsigned ROM addr)
    output logic signed [DATA_W-1:0] g      // sat8(round(K·gelu(idx/K)))
);

    // GELU ROM (generated, GELU_FBITS = 4)
    localparam logic signed [DATA_W-1:0] ROM [0:255] = '{
           0,    1,    1,    2,    2,    3,    4,    5,    6,    6,    7,    8,    9,   10,   11,   12,   // u=  0..15
          13,   15,   16,   17,   18,   19,   20,   21,   22,   24,   25,   26,   27,   28,   29,   30,   // u= 16..31
          31,   32,   33,   34,   36,   37,   38,   39,   40,   41,   42,   43,   44,   45,   46,   47,   // u= 32..47
          48,   49,   50,   51,   52,   53,   54,   55,   56,   57,   58,   59,   60,   61,   62,   63,   // u= 48..63
          64,   65,   66,   67,   68,   69,   70,   71,   72,   73,   74,   75,   76,   77,   78,   79,   // u= 64..79
          80,   81,   82,   83,   84,   85,   86,   87,   88,   89,   90,   91,   92,   93,   94,   95,   // u= 80..95
          96,   97,   98,   99,  100,  101,  102,  103,  104,  105,  106,  107,  108,  109,  110,  111,   // u= 96..111
         112,  113,  114,  115,  116,  117,  118,  119,  120,  121,  122,  123,  124,  125,  126,  127,   // u=112..127
           0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,   // u=128..143
           0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,   // u=144..159
           0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,   // u=160..175
           0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,   // u=176..191
           0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,   // u=192..207
           0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,   -1,   -1,   -1,   // u=208..223
          -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -2,   -2,   -2,   -2,   -2,   -2,   -2,   -2,   // u=224..239
          -3,   -3,   -3,   -3,   -3,   -3,   -3,   -3,   -2,   -2,   -2,   -2,   -2,   -1,   -1,    0    // u=240..255
    };

    assign g = ROM[idx];

    // The ROM literals are hard-coded for F=4; guard against a mismatched override.
    if (GELU_FBITS != 4) begin : g_fcheck
        $error("gelu_lut: ROM is generated for GELU_FBITS=4; regenerate the table for other F");
    end

endmodule

`default_nettype wire
