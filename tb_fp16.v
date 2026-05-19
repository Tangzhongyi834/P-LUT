`timescale 1ns/1ps

module tb_fp16;

reg         clk, rst_n, valid_in;
reg  [15:0] a, b;
wire        mul_valid, add_valid;
wire [15:0] mul_result, add_result;

mul_fp16 u_mul (
    .clk(clk), .rst_n(rst_n), .valid_in(valid_in),
    .a(a), .b(b), .valid_out(mul_valid), .result(mul_result)
);

adder_fp16 u_add (
    .clk(clk), .rst_n(rst_n), .valid_in(valid_in),
    .a(a), .b(b), .valid_out(add_valid), .result(add_result)
);

initial clk = 0;
always #5 clk = ~clk;

integer pass_cnt, fail_cnt;

task automatic run_test(
    input [127:0] label,
    input [15:0]  ia, ib, exp_m, exp_a
);
begin
    @(posedge clk); #1;
    a = ia; b = ib; valid_in = 1;
    @(posedge clk); #1;
    valid_in = 0; a = 0; b = 0;
    // Wait 3 pipeline stages (already 1 posedge consumed above)
    @(posedge clk); #1;
    @(posedge clk); #1;
    // Now mul_valid / add_valid should be high
    if (!mul_valid) begin
        $display("  MUL ERROR [%0s] : valid_out not asserted", label);
        fail_cnt = fail_cnt + 1;
    end else if (mul_result === exp_m) begin
        $display("  MUL PASS  [%0s] : got %h", label, mul_result);
        pass_cnt = pass_cnt + 1;
    end else begin
        $display("  MUL FAIL  [%0s] : expected %h, got %h", label, exp_m, mul_result);
        fail_cnt = fail_cnt + 1;
    end
    if (!add_valid) begin
        $display("  ADD ERROR [%0s] : valid_out not asserted", label);
        fail_cnt = fail_cnt + 1;
    end else if (add_result === exp_a) begin
        $display("  ADD PASS  [%0s] : got %h", label, add_result);
        pass_cnt = pass_cnt + 1;
    end else begin
        $display("  ADD FAIL  [%0s] : expected %h, got %h", label, exp_a, add_result);
        fail_cnt = fail_cnt + 1;
    end
end
endtask

initial begin
    $dumpfile("tb_fp16.vcd");
    $dumpvars(0, tb_fp16);

    pass_cnt = 0; fail_cnt = 0;
    rst_n = 0; valid_in = 0; a = 0; b = 0;
    repeat (4) @(posedge clk);
    rst_n = 1;
    @(posedge clk); #1;

    //                  label             a         b        mul_exp   add_exp
    run_test("1.0 op 1.0       ", 16'h3C00, 16'h3C00, 16'h3C00, 16'h4000);
    run_test("1.5 op 1.0       ", 16'h3E00, 16'h3C00, 16'h3E00, 16'h4100);
    run_test("1.5 op 1.5       ", 16'h3E00, 16'h3E00, 16'h4080, 16'h4200);
    run_test("1.5 op -1.0      ", 16'h3E00, 16'hBC00, 16'hBE00, 16'h3800);
    run_test("2.0 op 0.5       ", 16'h4000, 16'h3800, 16'h3C00, 16'h4100);
    run_test("0 op 1.0         ", 16'h0000, 16'h3C00, 16'h0000, 16'h3C00);
    run_test("Inf op 1.0       ", 16'h7C00, 16'h3C00, 16'h7C00, 16'h7C00);
    run_test("Inf op -Inf      ", 16'h7C00, 16'hFC00, 16'hFC00, 16'h7E00);
    run_test("1.0 op -1.0      ", 16'h3C00, 16'hBC00, 16'hBC00, 16'h0000);
    run_test("-0 op -0         ", 16'h8000, 16'h8000, 16'h0000, 16'h8000);
    run_test("-Inf op 1.0      ", 16'hFC00, 16'h3C00, 16'hFC00, 16'hFC00);
    run_test("NaN op 1.0       ", 16'h7E01, 16'h3C00, 16'h7E00, 16'h7E00);
    run_test("Inf op 0         ", 16'h7C00, 16'h0000, 16'h7E00, 16'h7C00);

    $display("\n========================================");
    $display("  PASS: %0d   FAIL: %0d   (of %0d checks)", pass_cnt, fail_cnt, pass_cnt+fail_cnt);
    $display("========================================");
    if (fail_cnt > 0)
        $display("*** SOME TESTS FAILED ***\n");
    else
        $display("*** ALL TESTS PASSED ***\n");

    $finish;
end

endmodule
