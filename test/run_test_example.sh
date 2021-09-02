#!/bin/sh

set -e

# Command-line arguments:
# op_name: the test operation, value should be same with the interface in cnnl_example.h
# input_shape: the shape of input tensor, each number means a dim value of input tensor and split with '-', the max supported len is 8
# output_shape: the shape of output tensor, output shape should be same with input shape of element-wise operations
# data_type: the data type of tensor, support values: half, float
# prefer: the chosen algorithm used for implementation of activation and accumulation operations, support values: fast, accuracy
# log_base: the base of log algorithm, support values: 2, 10, e

# Examples:
./test_example --op_name="cnnlAbs" --input_shape="{128}" --output_shape="{128}" --data_type=half
./test_example --op_name="cnnlSqrt" --prefer=fast --input_shape="{12-224-64}" --output_shape="{12-224-64}" --data_type=float
./test_example --op_name="cnnlDiv" --prefer=accuracy --input_shape="{1-112-112-3}" --output_shape="{1-112-112-3}" --data_type=float
./test_example --op_name="cnnlSqrtBackward" --input_shape="{2-3-4-5-6-7}" --output_shape="{2-3-4-5-6-7}" --data_type=half
./test_example --op_name="cnnlLog" --prefer=fast --log_base=2 --input_shape="{3-4-5-6-7-8-9-10}" --output_shape="{3-4-5-6-7-8-9-10}" --data_type=half
