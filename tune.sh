#!/bin/bash

# Usage:  tune.sh $1 $2 $3 $4
# $1 = model
# $2 = BS
# $3 = input length
# $4 = output length


modelfolder=/dockerx/$1
echo "Tuning $1 for batch size $2, input length $3 and output length $4"
echo ""
echo ""

# First generate shape files and run gemm_tuner.py to select the best perf model for a particular batchsize and input/output length

echo "Creating file /app/vllm/benchmarks/vllm-tools/gradlib_data/$1/gemms_$2_$3_$4.csv"
echo ""


HIP_FORCE_DEV_KERNARG=1 VLLM_TUNE_GEMM=1 VLLM_UNTUNE_FILE=/app/vllm/benchmarks/vllm-tools/gradlib_data/$1/gemms_$2_$3_$4.csv VLLM_USE_TRITON_FLASH_ATTN=1 python /app/vllm/benchmarks/benchmark_latency.py --model $modelfolder/ --batch-size $2 --input-len $3 --output-len $4 --num-iters-warmup 1 --num-iters 1 --dtype float16 --trust-remote-code --max_model_len=8192

CACHE_INVALIDATE_BUFFERS=7 python3 gemm_tuner.py --input_file /app/vllm/benchmarks/vllm-tools/gradlib_data/$1/gemms_$2_$3_$4.csv --tuned_file /app/vllm/benchmarks/vllm-tools/gradlib_data/$1/gemms_tuned_$2_$3_$4.csv --indtype f16 --outdtype f16

echo "Created file /app/vllm/benchmarks/vllm-tools/gradlib_data/$1/gemms_tuned_$2_$3_$4.csv"

sleep 5
echo "Running $1 at batch size $2, input length $3 and output length $4 with tuned results . . ."
sleep 5

VLLM_TUNE_FILE=/app/vllm/benchmarks/vllm-tools/gradlib_data/$1/gemms_tuned_$2_$3_$4.csv HIP_FORCE_DEV_KERNARG=1 VLLM_USE_TRITON_FLASH_ATTN=1 python /app/vllm/benchmarks/benchmark_latency.py --model $modelfolder/ --batch-size $2 --input-len $3 --output-len $4 --dtype float16 --trust-remote-code  --num-iters=10  --num-iters-warmup=1 --max_model_len=8192
