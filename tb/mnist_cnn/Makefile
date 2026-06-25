# cocotb pooling-free MNIST CNN test — v2 top (instruction-driven 16x16 TPU)
#   make            # run
#   make WAVES=1    # + dump.vcd
#
# Phase-3 milestone 1, plumbing-first: a full multi-layer pooling-free CNN
# (conv -> ReLU -> strided conv -> strided conv -> flatten -> FC) runs entirely
# on the UNCHANGED `top`, with software im2col between layers. Same RTL sources
# as tb/top and tb/synthetic_conv — zero new RTL.

TOPLEVEL_LANG = verilog
SIM           = verilator

RTL := $(PWD)/../../rtl
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
COCOTB_TEST_MODULES = tb_mnist_cnn

EXTRA_ARGS += -Wno-fatal

ifeq ($(WAVES),1)
    COMPILE_ARGS += --trace --trace-structs
    SIM_ARGS     += --trace
    SIM_BUILD    := sim_build_waves
endif

include $(shell cocotb-config --makefiles)/Makefile.sim
