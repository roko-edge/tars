# Board Integration and Local Synthesis Lab

This document specifies the separation between the vendor-agnostic NPU core and the
maintainer-local FPGA board integration environment, and defines the command surface
used to synthesize and flash designs on the RealDigital Urbana board.

---

## 1. Architectural Boundaries

| Layer | Location | Version Control | Content |
|---|---|---|---|
| NPU core | `npu/src/` | Versioned | Generic, synthesizable SystemVerilog; no board primitives or pin mappings |
| Board lab | `synthesis/` | Ignored (`.gitignore`) | Lab Makefile, board constraints, Vivado scripts, board wrapper tops |
| Lab experiments | `synthesis/*.sv` | Ignored (`.gitignore`) | Loose synthesis test modules (LED, switch, register experiments) |

Board-specific content never enters version control. The repository ships only the
generic core and its simulation flow.

---

## 2. NPU Interface Contract

A board wrapper instantiates `npu` and connects it to physical pins. The contract is:

| Port | Direction | Type | Description |
|---|---|---|---|
| `clk` | Input | `logic` | System clock |
| `rst` | Input | `logic` | Asynchronous, active-high reset |
| `start` | Input | `logic` | Starts an inference pass |
| `l0_act_fn`, `l1_act_fn` | Input | `npu_pkg::activation_t` | Per-layer activation selectors. Identity and ReLU are implemented. Sigmoid is reserved. |
| `done` | Output | `logic` | Asserted when `result` is committed |
| `result` | Output | `logic signed [31:0] [OUT]` | Output vector in Q8.24 |

---

## 3. Local Lab Layout

```text
synthesis/                (git-ignored)
  Makefile                synt/flash/sim/lint for loose files; synt-npu/flash-npu
  urbana.xdc              Official Urbana constraints (rev. V2I1, realdigital.org)
  scripts/
    build_bitstream.tcl   Non-project Vivado flow (synthesis, P&R, bitstream)
    flash_board.tcl       JTAG programming via hw_server
  <wrapper>.sv            Board wrapper modules for the NPU (maintainer-authored)
  <experiment>.sv         Loose lab test modules
```

The wrapper module for the NPU is authored by the maintainers and kept outside version
control together with the rest of the board lab.

---

## 4. Lab Commands

Run from `synthesis/`:

| Command | Effect |
|---|---|
| `make synt <file.sv>` | Synthesizes a loose module into `build_vivado/<top>.bit` |
| `make flash <file.sv>` | Synthesizes when needed and programs the board via JTAG |
| `make sim <file.sv>` | Quick Verilator simulation of a loose module |
| `make lint <file.sv>` | Verilator lint of a loose module |
| `make synt-npu` | Synthesizes the NPU with the local board wrapper |
| `make flash-npu` | Programs the NPU bitstream |
| `make clean` | Removes Vivado and simulation artifacts |

The top module name defaults to the file basename and can be overridden with
`TOP=<name>`. Part and constraints default to the Urbana values and can be overridden
with `PART=<part>` and `XDC=<file>`.

---

## 5. Board Reference

- Board: RealDigital Urbana, Xilinx Spartan-7 XC7S50-CSGA324.
- Vivado part: `xc7s50csga324-1`.
- Constraint source: official Urbana constraints revision V2I1 (realdigital.org),
  kept locally under `synthesis/urbana.xdc`.
- Canonical port names honored by the lab constraints: `CLK_100MHZ`, `SW[15:0]`,
  `BTN[3:0]`, `LED[15:0]`, `RGB0`, `RGB1`, `UART_TXD`, `UART_RXD`, `BLE_UART_*`.
- Connectors: programming and UART use the micro-USB Type-B `PROG UART` port, which
  also powers the board with jumper `J16` set to `USB`; the USB Type-C `EXTPWR`
  port is power-only (no data signals) and cannot be used for JTAG or UART.

---

## 6. Environment and Board Bring-Up Requirements

The lab invokes Vivado through a local wrapper (`vivado-run`) that runs inside a
container whose working directory differs from the lab directory. The Makefile
passes absolute paths for scripts, sources, constraints, outputs, and bitstreams
for this reason.

Vivado applies constraint files through a managed parser that accepts only
constraint commands. The lab constraints file uses `get_ports -quiet` and
`set_property -quiet` per pin, so a single file serves any top module that
exposes a subset of the canonical port names; the conditional 100 MHz
`create_clock` is issued by the build script instead of the constraints file.

Flashing requires the board connected over USB and host udev rules granting the
user access to the FTDI JTAG bridge (the `lsusb` command shows the vendor and
product identifiers). The container shares the host device nodes, so no extra
configuration is needed inside the Vivado environment.

---

## 7. Known Limitations

The simulation benches populate arrays and drive the RTL through explicit
ports. A board wrapper must supply the selected weights, scales, biases, and
input activations through those ports. The versioned flow provides simulation
and lint commands. Board inference requires a matching local integration and
observed results for that hardware build.
[Board evidence requirements](../project/V0_DELIVERY_PLAN.md#33-what-should-roko-be-able-to-consume-proposed)

The activation unit implements ReLU only. Selecting sigmoid (`activation = 2`) routes
the saturated sum unchanged, so the sigmoid output layer of the built-in experiments
has no hardware equivalent under the current sources.
