// ============================================================================
// adder_fp16.v — IEEE 754 Half-Precision (FP16) Pipelined Adder
//
// Format : 1-bit sign | 5-bit exponent (bias 15) | 10-bit fraction
// Pipeline : 3 stages  (latency = 3 cycles, throughput = 1 result / cycle)
// Rounding : Round to Nearest Even (RNE)
// Denormals: Flush-to-Zero on both input and output (FTZ)
// Special  : Full NaN / ±Inf / ±Zero propagation
// ============================================================================

module adder_fp16 (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        valid_in,
    input  wire [15:0] a,
    input  wire [15:0] b,
    output reg         valid_out,
    output reg  [15:0] result
);

// =========================================================================
// Stage 1 — Decode, compare magnitudes, swap so "large" ≥ "small"
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

wire sp_nan = a_is_nan | b_is_nan
            | (a_is_inf & b_is_inf & (a_sign != b_sign));
wire sp_inf = ~sp_nan & (a_is_inf | b_is_inf);
wire sp_inf_sign = a_is_inf ? a_sign : b_sign;

wire [10:0] a_mant = a_is_zero ? 11'd0 : {1'b1, a_frac};
wire [10:0] b_mant = b_is_zero ? 11'd0 : {1'b1, b_frac};

// |a| ≥ |b|?
wire a_ge_b = (a_exp > b_exp)
            | ((a_exp == b_exp) & (a_frac >= b_frac));

wire        lg_sign = a_ge_b ? a_sign : b_sign;
wire [4:0]  lg_exp  = a_ge_b ? a_exp  : b_exp;
wire [10:0] lg_mant = a_ge_b ? a_mant : b_mant;
wire [10:0] sm_mant = a_ge_b ? b_mant : a_mant;
wire        sm_sign = a_ge_b ? b_sign : a_sign;

wire        eff_sub = lg_sign ^ sm_sign;
wire [4:0]  shamt   = lg_exp - (a_ge_b ? b_exp : a_exp);

reg               s1_lg_sign;
reg [4:0]         s1_lg_exp;
reg [10:0]        s1_lg_mant;
reg [10:0]        s1_sm_mant;
reg [4:0]         s1_shamt;
reg               s1_eff_sub;
reg               s1_sp_nan, s1_sp_inf;
reg               s1_sp_inf_sign;
reg               s1_valid;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        s1_valid <= 1'b0;
    end else begin
        s1_lg_sign     <= lg_sign;
        s1_lg_exp      <= lg_exp;
        s1_lg_mant     <= lg_mant;
        s1_sm_mant     <= sm_mant;
        s1_shamt       <= shamt;
        s1_eff_sub     <= eff_sub;
        s1_sp_nan      <= sp_nan;
        s1_sp_inf      <= sp_inf;
        s1_sp_inf_sign <= sp_inf_sign;
        s1_valid       <= valid_in;
    end
end

// =========================================================================
// Stage 2 — Align the smaller mantissa, then add / subtract
// =========================================================================

// Shift small mantissa in a 36-bit field to capture Guard, Round, Sticky
// {mant[10:0], 25'b0}  >>  shamt
//   [35:25] = aligned mantissa (11 bits)
//   [24]    = guard
//   [23]    = round
//   |[22:0] = sticky

wire [35:0] sm_wide    = {s1_sm_mant, 25'b0};
wire [35:0] sm_shifted = sm_wide >> s1_shamt;
wire [10:0] sm_aligned = sm_shifted[35:25];
wire        guard_s    = sm_shifted[24];
wire        round_s    = sm_shifted[23];
wire        sticky_s   = |sm_shifted[22:0];

// Working format: {hidden, frac[9:0], G, R, S} = 14 bits
wire [13:0] lg_ext = {s1_lg_mant, 3'b0};
wire [13:0] sm_ext = {sm_aligned, guard_s, round_s, sticky_s};

// 15-bit result (extra bit for carry on addition)
wire [14:0] sum_raw = s1_eff_sub
                    ? ({1'b0, lg_ext} - {1'b0, sm_ext})
                    : ({1'b0, lg_ext} + {1'b0, sm_ext});

reg               s2_lg_sign;
reg [4:0]         s2_lg_exp;
reg [14:0]        s2_sum;
reg               s2_eff_sub;
reg               s2_sp_nan, s2_sp_inf;
reg               s2_sp_inf_sign;
reg               s2_valid;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        s2_valid <= 1'b0;
    end else begin
        s2_lg_sign     <= s1_lg_sign;
        s2_lg_exp      <= s1_lg_exp;
        s2_sum         <= sum_raw;
        s2_eff_sub     <= s1_eff_sub;
        s2_sp_nan      <= s1_sp_nan;
        s2_sp_inf      <= s1_sp_inf;
        s2_sp_inf_sign <= s1_sp_inf_sign;
        s2_valid       <= s1_valid;
    end
end

// =========================================================================
// Stage 3 — Normalize (carry / leading-zero shift), round, pack
// =========================================================================

wire        carry  = s2_sum[14] & ~s2_eff_sub;
wire [13:0] sum14  = s2_sum[13:0];

// --- Addition with carry: right-shift 1, fold R|S into new sticky ---------
wire [13:0] carry_norm = {s2_sum[14:2], s2_sum[1] | s2_sum[0]};
wire signed [6:0] carry_exp = $signed({2'b0, s2_lg_exp}) + 7'sd1;

// --- Subtraction: count leading zeros then left-shift to normalise --------
reg [3:0] lzc;
always @(*) begin
    casez (sum14)
        14'b1?????????????:  lzc = 4'd0;
        14'b01????????????:  lzc = 4'd1;
        14'b001???????????:  lzc = 4'd2;
        14'b0001??????????:  lzc = 4'd3;
        14'b00001?????????:  lzc = 4'd4;
        14'b000001????????:  lzc = 4'd5;
        14'b0000001???????:  lzc = 4'd6;
        14'b00000001??????:  lzc = 4'd7;
        14'b000000001?????:  lzc = 4'd8;
        14'b0000000001????:  lzc = 4'd9;
        14'b00000000001???:  lzc = 4'd10;
        14'b000000000001??:  lzc = 4'd11;
        14'b0000000000001?:  lzc = 4'd12;
        14'b00000000000001:  lzc = 4'd13;
        default:             lzc = 4'd0;
    endcase
end

wire [13:0]       sub_norm = sum14 << lzc;
wire signed [6:0] sub_exp  = $signed({2'b0, s2_lg_exp}) - $signed({3'b0, lzc});

// --- Select normalised mantissa & exponent --------------------------------
wire [13:0]       norm_mant = carry     ? carry_norm :
                              s2_eff_sub ? sub_norm   : sum14;
wire signed [6:0] norm_exp  = carry     ? carry_exp  :
                              s2_eff_sub ? sub_exp    :
                              $signed({2'b0, s2_lg_exp});

// {hidden, frac[9:0], G, R, S}
wire [9:0] final_frac = norm_mant[12:3];
wire       final_G    = norm_mant[2];
wire       final_R    = norm_mant[1];
wire       final_S    = norm_mant[0];

// RNE: round up when G=1 AND (R|S|LSB)
wire rnd_up = final_G & (final_R | final_S | final_frac[0]);

wire [10:0] frac_rnd = {1'b0, final_frac} + {10'd0, rnd_up};
wire        rnd_ovf  = frac_rnd[10];

wire signed [6:0] exp_rnd  = rnd_ovf ? (norm_exp + 7'sd1) : norm_exp;
wire [9:0]        frac_out = rnd_ovf ? frac_rnd[10:1]     : frac_rnd[9:0];

wire result_is_zero = (s2_sum == 15'd0);

// IEEE 754: +0 on exact cancellation (RNE), preserve −0 + −0 = −0
wire zero_sign = s2_eff_sub ? 1'b0 : s2_lg_sign;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        result    <= 16'd0;
        valid_out <= 1'b0;
    end else begin
        valid_out <= s2_valid;
        if (s2_sp_nan)
            result <= {1'b0, 5'h1f, 10'h200};
        else if (s2_sp_inf)
            result <= {s2_sp_inf_sign, 5'h1f, 10'h000};
        else if (result_is_zero)
            result <= {zero_sign, 5'd0, 10'd0};
        else if (exp_rnd >= 7'sd31)
            result <= {s2_lg_sign, 5'h1f, 10'h000};
        else if (exp_rnd <= 7'sd0)
            result <= {s2_lg_sign, 5'd0, 10'd0};
        else
            result <= {s2_lg_sign, exp_rnd[4:0], frac_out};
    end
end

endmodule
