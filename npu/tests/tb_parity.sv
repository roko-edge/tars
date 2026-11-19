`timescale 1ns / 1ps
module tb_parity;
  import npu_pkg::*;

  localparam int IN = 784;
  localparam int H = 64;
  localparam int OUT = 10;
  localparam int SAMPLES = 100;

  logic clk, rst, start;
  logic [1:0] l0_w[IN*H], l1_w[H*OUT];
  logic signed [31:0] a[IN], l0_b[H], l1_b[OUT], l0_w0, l1_w0;
  logic done;
  npu_pkg::activation_t l0_act_fn, l1_act_fn;
  logic signed [31:0] result[OUT];

  npu #(
      .IN (IN),
      .H  (H),
      .OUT(OUT)
  ) dut (
      .clk,
      .rst,
      .start,
      .l0_w,
      .l1_w,
      .a,
      .l0_b,
      .l1_b,
      .l0_w0,
      .l1_w0,
      .l0_act_fn,
      .l1_act_fn,
      .done,
      .result
  );

  logic        [ 1:0] mem_wt  [ IN*H + H*OUT];
  logic signed [31:0] mem_w0  [            2];
  logic signed [31:0] mem_bias[      H + OUT];
  logic signed [31:0] mem_act [ SAMPLES * IN];
  logic signed [31:0] mem_exp [SAMPLES * OUT];

  task automatic check(input string nome, input logic signed [31:0] got,
                       input logic signed [31:0] exp);
    logic signed [31:0] diff;
    diff = (got > exp) ? (got - exp) : (exp - got);

    if (diff > 32'sh1) begin
      $fatal(1, "%s: ERROR in parity! got %h, expected %h (diff: %d)", nome, got, exp, diff);
    end else begin
      $display("PASS %s (got %h, expected %h)", nome, got, exp);
    end
  endtask

  always #5 clk = ~clk;

  initial begin
    #100_000_000 $fatal(1, "timeout: NPU is too slow in inference");
  end

  initial begin
    clk = 0;
    rst = 1;
    start = 0;
    l0_act_fn = npu_pkg::RELU;
    l1_act_fn = npu_pkg::IDENTITY;

    $readmemh("data/wt.mem", mem_wt);
    $readmemh("data/w0.mem", mem_w0);
    $readmemh("data/bias.mem", mem_bias);
    $readmemh("data/activations.mem", mem_act);
    $readmemh("data/target.mem", mem_exp);

    for (int i = 0; i < IN * H; i++) l0_w[i] = mem_wt[i];
    for (int i = 0; i < H * OUT; i++) l1_w[i] = mem_wt[IN*H+i];

    l0_w0 = mem_w0[0];
    l1_w0 = mem_w0[1];

    for (int i = 0; i < H; i++) l0_b[i] = mem_bias[i];
    for (int i = 0; i < OUT; i++) l1_b[i] = mem_bias[H+i];

    repeat (2) @(posedge clk);
    rst = 0;
    @(posedge clk);

    $display("=== [TIER 2 PARITY] MNIST inference in NPU is running (50.816 weights) ===");

    for (int s = 0; s < SAMPLES; s++) begin
      for (int i = 0; i < IN; i++) begin
        a[i] = mem_act[s*IN+i];
      end

      @(posedge clk);
      start = 1;
      @(posedge clk);
      start = 0;

      do @(posedge clk); while (!done);

      for (int o = 0; o < OUT; o++) begin
        check($sformatf("mnist_sample_%0d_digit_%0d", s, o), result[o], mem_exp[s*OUT+o]);
      end
    end

    $display("=== tb_parity: SUCCESS! NPU parity with the rust results verified ! ===");
    $finish;
  end

endmodule
