#!/bin/bash

# --- Script Configuration ---
# This script automates the process of categorizing the metrics JSONL file
# into three shuffled terciles (low, medium, high) for all specified metrics.

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Variables ---
# The name of the Python script to run.
# Make sure the Python code is saved with this filename.
PYTHON_SCRIPT="BulkByMetric.py"
# The path to the large input JSONL file.

MODEL="Llama8B"  ## Llama8B Gemma4B Mistral7B General


# INPUT_FILE="Data/reasoning_steps.jsonl"
# INPUT_FILE="Data/symbol_complexity.jsonl"
# INPUT_FILE="Data/comprehension_difficulty.jsonl"

INPUT_FILE="Data/${MODEL}_entropy.jsonl"
# INPUT_FILE="Data/${MODEL}_pass_k.jsonl"

# The directory where all categorized files will be saved.
OUTPUT_DIR="Processed_data/BulkByMetric/${MODEL}"

# An array containing all the metric keys you want to sort by.
METRICS_TO_SORT=(        # "Data/Llama8B_entropy.jsonl"
    "mean_nll"
    "variance_nll"       # "Data/Mistral7B_entropy.jsonl"
    "mean_nll_per_token"
    "variance_nll_per_token"  # "Data/Gemma4B_entropy.jsonl"
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

echo "🚀 Starting batch categorization process..."
echo "Input file: ${INPUT_FILE}"
echo "Output directory: ${OUTPUT_DIR}"

# Loop through each metric in the array.
for metric in "${METRICS_TO_SORT[@]}"; do
    echo ""
    echo "----------------------------------------------------"
    echo "🔄 Processing metric: ${metric}"
    echo "----------------------------------------------------"
    
    # Construct and execute the command to run the Python script.
    python "${PYTHON_SCRIPT}" \
        --input_file "${INPUT_FILE}" \
        --metric "${metric}" \
        --output_dir "${OUTPUT_DIR}"
done

echo ""
echo "🎉 All categorization tasks complete!"
echo "✅ All categorized files have been saved to the '${OUTPUT_DIR}' directory."