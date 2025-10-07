#!/bin/bash

# --- Script Configuration ---
# This script automates the process of creating curriculum files
# by running the Python script for all specified metric keys.

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Variables ---
# The name of the Python script to run.
# IMPORTANT: Make sure this matches the filename of your Python script.
PYTHON_SCRIPT="BulkMerge.py"

# The model name, used for defining input and output paths.
MODEL="Gemma4B"  # Example: Llama8B, Gemma4B, Mistral7B, General

# The path to the large input JSONL file.
# The ${MODEL} variable is used here to make it easy to switch models.

# INPUT_FILE="Data/reasoning_steps.jsonl"
# INPUT_FILE="Data/comprehension_difficulty.jsonl"
# INPUT_FILE="Data/symbol_complexity.jsonl"


INPUT_FILE="Data/${MODEL}_entropy.jsonl"
# INPUT_FILE="Data/${MODEL}_pass_k.jsonl"

# The directory where all curriculum files will be saved.
OUTPUT_DIR="Processed_data/BulkMerge/${MODEL}"

# An array containing all the metric keys you want to process.
# This list is for the entropy metrics.

METRICS_TO_SORT=(
    "mean_nll"
    "variance_nll"
    "mean_nll_per_token"
    "variance_nll_per_token"
    "mean_top5_shannon_entropy"
    "variance_top5_shannon_entropy"
    "mean_sequence_entropy"
    "variance_sequence_entropy"
    "mean_token_entropy"
    "variance_token_entropy"
    "mean_logit_gap"
    "variance_logit_gap"
)

# METRICS_TO_SORT=(  # "Data/Llama8B__pass_k.jsonl"
#     "pass@k"  # "Data/Mistral7B_pass_k.jsonl"
#     "pass@k_variance"  # "Data/Gemma4B_pass_k.jsonl"       
# )

# METRICS_TO_SORT=(     # "Data/reasoning_steps.jsonl"
#     "reasoning_steps"  
# )

# METRICS_TO_SORT=(    # "Data/comprehension_difficulty.jsonl"
#     "comprehension_difficulty"  
# )

# METRICS_TO_SORT=(   # # ".Data/symbol_complexity.jsonl"
#     "symbol_complexity"  
# )
# --- Main Logic ---

echo "🚀 Starting batch curriculum creation process..."
echo "Input file: ${INPUT_FILE}"
echo "Output directory: ${OUTPUT_DIR}"

# Loop through each metric in the array.
for metric in "${METRICS_TO_SORT[@]}"; do
    echo ""
    echo "----------------------------------------------------"
    echo "🔄 Processing metric: ${metric}"
    echo "----------------------------------------------------"
    
    # Construct and execute the command to run the Python script.
    # The quotes around variables are a good practice to handle paths with spaces.
    python "${PYTHON_SCRIPT}" \
        --input_file "${INPUT_FILE}" \
        --metric "${metric}" \
        --output_dir "${OUTPUT_DIR}"
done

echo ""
echo "🎉 All curriculum creation tasks complete!"
echo "✅ Curriculum files have been saved to the '${OUTPUT_DIR}' directory."
