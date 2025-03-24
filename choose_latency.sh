#!/bin/bash

# latency_run.sh

# Exit on any error
set -e

# Get GPU part number
ifwi=$(amd-smi static | grep "PART_NUMBER:" | sed 's/PART_NUMBER: *//' | sed 's/^[ \t]*//;s/[ \t]*$//')
[ -z "$ifwi" ] && { echo "Error: Could not retrieve GPU part number"; exit 1; }

# List available models
echo "Available models:"
models=($(ls -d /dockerx/*/ | grep -v "ShareGPT_Vicuna_unfiltered" | sed 's/\/dockerx\///g' | sed 's/\///g'))
[ ${#models[@]} -eq 0 ] && { echo "Error: No models found in /dockerx"; exit 1; }
for i in "${!models[@]}"; do
    echo "$((i+1)). ${models[$i]}"
done

# Prompt user to select multiple models
echo "Please enter the model numbers to benchmark (space-separated, e.g., '1 2 3'):"
read -r selections

# Convert selections to an array and validate
selected_indices=()
for sel in $selections; do
    # Check if input is a valid number
    if ! [[ "$sel" =~ ^[0-9]+$ ]] || [ "$sel" -lt 1 ] || [ "$sel" -gt "${#models[@]}" ]; then
        echo "Invalid selection: $sel"
        exit 1
    fi
    selected_indices+=("$((sel-1))")
done

# Check if at least one model was selected
[ ${#selected_indices[@]} -eq 0 ] && { echo "Error: No valid models selected"; exit 1; }

# Function to run benchmark and extract latency
run_benchmark() {
    local batch_size=$1
    local input_len=$2
    local output_len=$3
    local model=$4

    cmd="python benchmark_latency.py --input-len=$input_len --output-len=$output_len --batch-size=$batch_size --num-iters=5 --model=/dockerx/$model/ --trust-remote-code --num-iters-warmup 1 --dtype=float16 --max_model_len 4096"

    if ! output=$($cmd 2>&1); then
        echo "Error: Benchmark failed for model=$model, BS=$batch_size, Input=$input_len, Output=$output_len"
        return 1
    fi

    latency=$(echo "$output" | grep "Avg latency" | grep -o '[0-9]\+\.[0-9]\+' || echo "N/A")
    echo "$latency"
}

# Benchmark parameters
benchmark_params=("1 1024 1" "1 1024 1024" "8 1024 1024")

# Output file
output_file="$ifwi-benchmarks.txt"
echo "IFWI: $ifwi" > "$output_file"

# Process each selected model
for idx in "${selected_indices[@]}"; do
    model="${models[$idx]}"
    echo -e "\nBenchmarking model: $model"

    # Initialize output for this model
    output="Results for $model
BS      Input   Output  Latency"
    echo -e "$output"
    echo -e "\n$output" >> "$output_file"

    # Run benchmarks for this model
    for params in "${benchmark_params[@]}"; do
        read batch_size input_len output_len <<< "$params"
        echo -n "Running BS=$batch_size, Input=$input_len, Output=$output_len... "
        latency=$(run_benchmark "$batch_size" "$input_len" "$output_len" "$model")
        echo "$latency"
        output="$batch_size     $input_len      $output_len     $latency"
        echo -e "$output" >> "$output_file"
    done
done

echo -e "\nAll benchmarks completed! Results saved to $output_file"
