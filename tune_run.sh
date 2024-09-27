#!/bin/bash

# Use the generated gemms_tuned CSV file to run the model

modelfolder=/dockerx/$1

echo $2, $3, $3

VLLM_TUNE_FILE=/app/vllm/benchmarks/vllm-tools/gradlib_data/$1/gemms_tuned_$2_$3_$3.csv HIP_FORCE_DEV_KERNARG=1 VLLM_USE_TRITON_FLASH_ATTN=1 python /app/vllm/benchmarks/benchmark_latency.py --model $modelfolder/ --batch-size $2 --input-len $3 --output-len $3 --dtype float16 --trust-remote-code  --num-iters=10  --num-iters-warmup=1 
