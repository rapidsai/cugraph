# Copyright (c) 2015, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

CXX=g++
CXXFLAGS=-Wall -Ofast -march=native -pipe

all: print_info plodg rmatg

plodg: plod.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

rmatg: rmat.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

clean:
	rm -f rmatg plodg

print_info:
	$(info The Boost Graph Library is required)
