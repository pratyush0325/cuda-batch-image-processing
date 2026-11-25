NVCC ?= nvcc
CXX = g++
CPPFLAGS = -Iinclude
CXXFLAGS = -std=c++17 -O3 -Wall -Wextra -Werror -ffp-contract=off
NVCCFLAGS = -std=c++17 -O3 --fmad=false -Xcompiler=-ffp-contract=off
LDLIBS ?=

.PHONY: all build test run clean
all: build

build: build/batch_image

build/batch_image: src/main.cc src/image.cc src/cpu_reference.cc src/gpu_pipeline.cu include/image.h include/gpu_pipeline.h
	mkdir -p build
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) src/main.cc src/image.cc src/cpu_reference.cc src/gpu_pipeline.cu -o $@ $(LDLIBS)

build/cpu_tests: tests/cpu_tests.cc src/image.cc src/cpu_reference.cc include/image.h
	mkdir -p build
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) tests/cpu_tests.cc src/image.cc src/cpu_reference.cc -o $@

test: build/cpu_tests
	./build/cpu_tests
	python3 tests/test_tools.py

run:
	bash run.sh

clean:
	rm -rf build
