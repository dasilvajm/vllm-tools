#!/bin/bash

# throughput_run.sh

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

# Function to run throughput benchmark and extract throughput
run_benchmark() {
    local model=$1

    cmd="python benchmark_throughput.py --model /dockerx/$model --max_model_len 4096 --trust-remote-code --dataset /dockerx/ShareGPT_Vicuna_unfiltered/ShareGPT_V3_unfiltered_cleaned_split.json --num_prompts 1000"

    if ! output=$($cmd 2>&1); then
        echo "Error: Benchmark failed for model=$model"
        return 1
    fi

    # Extract the number before "requests/s" from "Throughput: 6.40 requests/s, ..."
    throughput=$(echo "$output" | grep "Throughput:" | sed 's/Throughput: \([0-9]\+\.[0-9]\+\) requests\/s.*/\1/' || echo "N/A")
    echo "$throughput"
}

# Output file
output_file="$ifwi-throughput-benchmarks.txt"
echo "IFWI: $ifwi" > "$output_file"
echo "Dataset: /dockerx/ShareGPT_Vicuna_unfiltered/ShareGPT_V3_unfiltered_cleaned_split.json" >> "$output_file"
echo "Num Prompts: 1000" >> "$output_file"

# Process each selected model
for idx in "${selected_indices[@]}"; do
    model="${models[$idx]}"
    echo -e "\nBenchmarking model: $model"

    # Initialize output for this model
    output="Results for $model
Throughput (requests/s)"
    echo -e "$output"
    echo -e "\n$output" >> "$output_file"

    # Run benchmark for this model
    echo -n "Running throughput benchmark... "
    throughput=$(run_benchmark "$model")
    echo "$throughput"
    echo -e "$throughput" >> "$output_file"
done

echo -e "\nAll benchmarks completed! Results saved to $output_file"
