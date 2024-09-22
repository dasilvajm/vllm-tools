#!/bin/bash

# Usage:  tune_create.sh $1 $2 $3
# $1 = model
# $2 = BS
# $3 = input/output length


modelfolder=/dockerx/$1
echo "Tuning $1 for batch size $2 and input/output length $3"
echo ""
echo ""

# First generate shape files and run gemm_tuner.py to select the best perf model for a particular batchsize and input/output length
 
echo "Creating file /app/vllm/benchmarks/vllm-tools/gradlib_data/$1/gemms_$2_$3_$3.csv"
echo ""


HIP_FORCE_DEV_KERNARG=1 VLLM_TUNE_GEMM=1 VLLM_UNTUNE_FILE=/app/vllm/benchmarks/vllm-tools/gradlib_data/$1/gemms_$2_$3_$3.csv VLLM_USE_TRITON_FLASH_ATTN=1 python /app/vllm/benchmarks/benchmark_latency.py --model $modelfolder/ --batch-size $2 --input-len $3 --output-len $3 --num-iters-warmup 1 --num-iters 1 --dtype float16 --trust-remote-code

CACHE_INVALIDATE_BUFFERS=7 python3 gemm_tuner.py --input_file /app/vllm/benchmarks/vllm-tools/gradlib_data/$1/gemms_$2_$3_$3.csv --tuned_file /app/vllm/benchmarks/vllm-tools/gradlib_data/$1/gemms_tuned_$2_$3_$3.csv --indtype f16 --outdtype f16
    
echo "Created file /app/vllm/benchmarks/vllm-tools/gradlib_data/$1/gemms_tuned_$2_$3_$3.csv"
