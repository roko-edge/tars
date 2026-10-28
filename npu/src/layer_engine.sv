module layer_engine
  import npu_pkg::*;
#(
    parameter int IN  = 2,
    parameter int OUT = 2

) (
    input logic clk,
    rst,
    start,
    input npu_pkg::activation_t act_fn,
    input logic [1:0] w[IN*OUT],
    input logic signed [31:0] a[IN],
    b[OUT],
    w0,
    output logic signed [31:0] result[OUT],
    output logic done
);

  npu_pkg::state_t state;
  function automatic [63:0] safe_log;
    input [63:0] x;
    safe_log = x < 2 ? 64'(1) : 64'($clog2(x));
  endfunction

  logic unsigned [safe_log(OUT)-1:0] j;
  logic unsigned [safe_log(IN)-1:0] i;
  logic unsigned [safe_log(OUT*IN)-1:0] w_ptr;

  logic signed [63:0] pe_acc;
  pe u_pe (
      .clk        (clk),
      .rst        (rst),
      .init       (state == RUN && i == 0),
      .en         (state == RUN),
      .a          (a[(i)]),
      .w          (w[(w_ptr)]),
      .accumulator(pe_acc)
  );

  logic signed [31:0] ep_out;
  epilogue u_epilogue (
      .pe_acc(pe_acc),
      .b(b[j]),
      .w0(w0),
      .act_fn(act_fn),
      .out(ep_out)
  );

  always_ff @(posedge clk or posedge rst) begin
    if (rst) begin
      j <= 0;
      i <= 0;
      w_ptr <= 0;
      for (int k = 0; k < OUT; ++k) begin
        result[k] <= 0;
      end
      done  <= 0;

      state <= IDLE;
    end else begin
      unique case (state)
        IDLE: begin
          if (start) begin
            done <= 0;
            state <= RUN;
            i <= 0;
            j <= 0;
            w_ptr <= 0;
          end
        end
        RUN: begin
          if (32'(i) == IN - 1) begin
            state <= WRITE;
          end else begin
            i <= i + 1;
            w_ptr <= w_ptr + 1;
          end

        end
        WRITE: begin
          result[j] <= ep_out;
          i <= 0;
          if (32'(j) == OUT - 1) begin
            state <= IDLE;
            done <= 1;
            j <= 0;
            w_ptr <= 0;
          end else begin
            j <= j + 1;
            w_ptr <= w_ptr + 1;
            state <= RUN;
          end

        end
      endcase

    end
  end
endmodule
