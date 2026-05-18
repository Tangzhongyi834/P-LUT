// ============================================================================
// mul_fp16.v — IEEE 754 Half-Precision (FP16) Pipelined Multiplier
//
// Format : 1-bit sign | 5-bit exponent (bias 15) | 10-bit fraction
// Pipeline : 3 stages  (latency = 3 cycles, throughput = 1 result / cycle)
// Rounding : Round to Nearest Even (RNE)
// Denormals: Flush-to-Zero on both input and output (FTZ)
// Special  : Full NaN / ±Inf / ±Zero propagation
// ============================================================================

module mul_fp16 (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        valid_in,
    input  wire [15:0] a,
    input  wire [15:0] b,
    output reg         valid_out,
    output reg  [15:0] result
);

// =========================================================================
// Stage 1 — Decode, classify, prepare exponent & mantissa
// =========================================================================

wire        a_sign = a[15];
wire [4:0]  a_exp  = a[14:10];
wire [9:0]  a_frac = a[9:0];
wire        b_sign = b[15];
wire [4:0]  b_exp  = b[14:10];
wire [9:0]  b_frac = b[9:0];

wire a_is_zero = (a_exp == 5'd0);
wire a_is_inf  = (a_exp == 5'd31) && (a_frac == 10'd0);
wire a_is_nan  = (a_exp == 5'd31) && (a_frac != 10'd0);
wire b_is_zero = (b_exp == 5'd0);
wire b_is_inf  = (b_exp == 5'd31) && (b_frac == 10'd0);
wire b_is_nan  = (b_exp == 5'd31) && (b_frac != 10'd0);

wire        p_sign = a_sign ^ b_sign;
wire [10:0] a_mant = a_is_zero ? 11'd0 : {1'b1, a_frac};
wire [10:0] b_mant = b_is_zero ? 11'd0 : {1'b1, b_frac};

// Biased product exponent: (Ea-15)+(Eb-15)+15 = Ea+Eb-15
wire signed [7:0] p_exp_raw =
    $signed({3'b0, a_exp}) + $signed({3'b0, b_exp}) - 8'sd15;

wire sp_nan  = a_is_nan | b_is_nan
             | (a_is_inf & b_is_zero) | (b_is_inf & a_is_zero);
wire sp_inf  = ~sp_nan & (a_is_inf | b_is_inf);
wire sp_zero = ~sp_nan & ~sp_inf & (a_is_zero | b_is_zero);

reg               s1_sign;
reg signed  [7:0] s1_exp;
reg         [10:0] s1_mant_a, s1_mant_b;
reg               s1_nan, s1_inf, s1_zero;
reg               s1_valid;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        s1_valid <= 1'b0;
    end else begin
        s1_sign   <= p_sign;
        s1_exp    <= p_exp_raw;
        s1_mant_a <= a_mant;
        s1_mant_b <= b_mant;
        s1_nan    <= sp_nan;
        s1_inf    <= sp_inf;
        s1_zero   <= sp_zero;
        s1_valid  <= valid_in;
    end
end

// =========================================================================
// Stage 2 — 11×11 mantissa multiplication
// =========================================================================

// 1.10 × 1.10 → product in [1.0, 4.0), represented as 2.20 fixed-point
wire [21:0] product = s1_mant_a * s1_mant_b;

reg               s2_sign;
reg signed  [7:0] s2_exp;
reg         [21:0] s2_product;
reg               s2_nan, s2_inf, s2_zero;
reg               s2_valid;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        s2_valid <= 1'b0;
    end else begin
        s2_sign   <= s1_sign;
        s2_exp    <= s1_exp;
        s2_product <= product;
        s2_nan    <= s1_nan;
        s2_inf    <= s1_inf;
        s2_zero   <= s1_zero;
        s2_valid  <= s1_valid;
    end
end

// =========================================================================
// Stage 3 — Normalize, round-to-nearest-even, pack
// =========================================================================

// If product[21]=1  →  1x.xxxx…  →  right-shift 1, exp+1
//   frac = product[20:11],  G = [10],  R = [9],  S = |[8:0]
// If product[21]=0  →  01.xxxx…  →  no shift
//   frac = product[19:10],  G = [9],   R = [8],  S = |[7:0]

wire        do_shift  = s2_product[21];
wire [9:0]  norm_frac = do_shift ? s2_product[20:11] : s2_product[19:10];
wire        norm_G    = do_shift ? s2_product[10]     : s2_product[9];
wire        norm_R    = do_shift ? s2_product[9]      : s2_product[8];
wire        norm_S    = do_shift ? (|s2_product[8:0]) : (|s2_product[7:0]);

wire signed [7:0] norm_exp = do_shift ? (s2_exp + 8'sd1) : s2_exp;

// RNE: round up when G=1 AND (R|S|LSB)
wire rnd_up = norm_G & (norm_R | norm_S | norm_frac[0]);

wire [10:0] frac_rnd = {1'b0, norm_frac} + {10'd0, rnd_up};
wire        rnd_ovf  = frac_rnd[10];

wire signed [7:0] final_exp  = rnd_ovf ? (norm_exp + 8'sd1) : norm_exp;
wire [9:0]        final_frac = rnd_ovf ? frac_rnd[10:1]     : frac_rnd[9:0];

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        result    <= 16'd0;
        valid_out <= 1'b0;
    end else begin
        valid_out <= s2_valid;
        if (s2_nan)
            result <= {1'b0, 5'h1f, 10'h200};
        else if (s2_inf)
            result <= {s2_sign, 5'h1f, 10'h000};
        else if (s2_zero)
            result <= {s2_sign, 5'd0, 10'd0};
        else if (final_exp >= 8'sd31)
            result <= {s2_sign, 5'h1f, 10'h000};
        else if (final_exp <= 8'sd0)
            result <= {s2_sign, 5'd0, 10'd0};
        else
            result <= {s2_sign, final_exp[4:0], final_frac};
    end
end

endmodule
