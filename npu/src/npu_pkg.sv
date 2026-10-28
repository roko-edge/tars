package npu_pkg;
  typedef enum logic [1:0] {
    IDENTITY = 0,
    RELU     = 1,
    SIGMOID  = 2
  } activation_t;
  typedef enum logic [1:0] {
    IDLE  = 0,
    RUN   = 1,
    WRITE = 2
  } state_t;
  localparam logic signed [31:0] Q8_24_MAX = 32'sh7FFF_FFFF;  // +127.999…
  localparam logic signed [31:0] Q8_24_MIN = 32'sh8000_0000;  // -128.0
endpackage
