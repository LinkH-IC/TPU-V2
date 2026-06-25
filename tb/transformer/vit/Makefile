# cocotb end-to-end MNIST ViT bench — full trained transformer on top (Verilator)
#   make            # run
#   make WAVES=1    # + dump.vcd, then:  gtkwave dump.vcd
#
# One directory per testbench: all sim artifacts (sim_build/, results.xml,
# dump.vcd) land here, keeping tb/ clean.

TOPLEVEL_LANG = verilog
SIM           = verilator

RTL := $(PWD)/../../../rtl
VERILOG_SOURCES = \
    $(RTL)/pe.sv \
    $(RTL)/systolic_array.sv \
    $(RTL)/accumulator.sv \
    $(RTL)/weight_buffer.sv \
    $(RTL)/activation_buffer.sv \
    $(RTL)/result_buffer.sv \
    $(RTL)/bias_add.sv \
    $(RTL)/exp_lut.sv \
    $(RTL)/reciprocal.sv \
    $(RTL)/rsqrt.sv \
    $(RTL)/gelu_lut.sv \
    $(RTL)/vpu.sv \
    $(RTL)/control_fsm.sv \
    $(RTL)/mem.sv \
    $(RTL)/cmd_proc.sv \
    $(RTL)/top.sv

TOPLEVEL            = top
COCOTB_TEST_MODULES = tb_vit

EXTRA_ARGS += -Wno-fatal

# Waveform: WAVES=1 -> dump.vcd in this dir (open with: gtkwave dump.vcd).
# Use a separate build dir so toggling WAVES never reuses a non-traced binary
# (Verilator errors if --trace runs against a build without trace support).
ifeq ($(WAVES),1)
    COMPILE_ARGS += --trace --trace-structs
    SIM_ARGS     += --trace
    SIM_BUILD    := sim_build_waves
endif

include $(shell cocotb-config --makefiles)/Makefile.sim
