module epilogue
  import npu_pkg::*;
(
    input logic signed [63:0] pe_acc,
    input logic signed [31:0] b,
    w0,
    input npu_pkg::activation_t act_fn,
    output logic signed [31:0] out

);
  function automatic signed [31:0] safe_64_to_32;
    input [63:0] x;
    logic signed [31:0] y;
    if (x == 64'(signed'(x[31:0]))) begin
      y = x[31:0];
    end else begin
      y = x[63] ? npu_pkg::Q8_24_MIN : npu_pkg::Q8_24_MAX;
    end
    safe_64_to_32 = y;
  endfunction

  function automatic signed [31:0] safe_96_to_32;
    input [95:0] x;
    logic signed [31:0] y;
    if (x == 96'(signed'(x[31:0]))) begin
      y = x[31:0];
    end else begin
      y = x[95] ? npu_pkg::Q8_24_MIN : npu_pkg::Q8_24_MAX;
    end
    safe_96_to_32 = y;
  endfunction

  logic signed [31:0] z;
  assign z = safe_64_to_32(64'(safe_96_to_32(pe_acc * w0 >>> 24)) + 64'(b));

  always_comb begin
    unique case (act_fn)
      RELU: begin
        out = z[31] ? 0 : z;
      end
      default: out = z;
    endcase
  end
endmodule
