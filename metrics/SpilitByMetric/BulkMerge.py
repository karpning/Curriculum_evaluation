import json
import os
import argparse
import sys
from tqdm import tqdm
import random

def check_metric_exists(input_file: str, metric_key: str):
    """
    Efficiently checks if the metric_key exists in the first valid JSON object of the file.
    Raises ValueError if the key is not found or the file is empty/corrupt.
    """
    print(f"🔎 Verifying if metric '{metric_key}' exists in the first record of '{input_file}'...")
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            
            try:
                data = json.loads(line)
                if metric_key not in data:
                    available_keys = list(data.keys())
                    error_message = (
                        f"Metric key '{metric_key}' not found.\n"
                        f"Available keys in the first record are: {available_keys}"
                    )
                    raise ValueError(error_message)
                
                print(f"✅ Metric key '{metric_key}' found.")
                return
            except json.JSONDecodeError:
                raise ValueError(f"Could not decode the first line. File might be corrupted: {line.strip()}")

    raise ValueError("Input file is empty or contains no valid JSON objects.")


def create_curriculum_files(input_file: str, metric_key: str, output_dir: str):
    """
    Loads a JSONL file, sorts it by a metric, divides it into three shuffled 
    terciles, and creates two curriculum files: easy-to-hard and hard-to-easy.
    """
    # --- 1. Create base output directory ---
    os.makedirs(output_dir, exist_ok=True)
    print(f"📂 Base output directory: '{output_dir}'")

    # --- 2. Load all data from the JSONL file into memory ---
    print(f"\n⏳ Loading data from '{input_file}'...")
    data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"⚠️ Warning: Skipping corrupted line: {line.strip()}")
    
    if not data:
        print("❌ Error: No data loaded from file.")
        return
        
    print(f"✅ Loaded {len(data)} records successfully.")

    # --- 3. Sort the data by the specified metric ---
    print(f"\n🔄 Sorting all data by '{metric_key}' in ascending order...")
    sort_key = lambda item: item.get(metric_key, 0.0)
    sorted_data = sorted(data, key=sort_key)
    print("✅ Sorting complete.")

    # --- 4. Split the data into three terciles (low, medium, high) ---
    print("\n🔪 Splitting data into three terciles...")
    total_count = len(sorted_data)
    split_1_index = total_count // 3
    split_2_index = 2 * (total_count // 3)

    low_tercile = sorted_data[:split_1_index]
    medium_tercile = sorted_data[split_1_index:split_2_index]
    high_tercile = sorted_data[split_2_index:]

    print(f"✅ Data split into: Low ({len(low_tercile)}), Medium ({len(medium_tercile)}), High ({len(high_tercile)})")

    # --- 5. Shuffle each difficulty tercile internally ---
    print("\n🔀 Shuffling data within each difficulty tercile...")
    random.shuffle(low_tercile)
    random.shuffle(medium_tercile)
    random.shuffle(high_tercile)
    print("✅ Shuffling complete.")

    # --- 6. Filter the data to keep only specified keys ---
    print("\n📝 Filtering records to keep only essential keys...")
    base_keys_to_keep = ['query', 'response', 'type', 'id']
    keys_to_keep = list(set(base_keys_to_keep + [metric_key]))
    
    def filter_item_keys(item):
        return {key: item.get(key) for key in keys_to_keep if key in item}

    filtered_low = [filter_item_keys(item) for item in low_tercile]
    filtered_medium = [filter_item_keys(item) for item in medium_tercile]
    filtered_high = [filter_item_keys(item) for item in high_tercile]
    print("✅ Filtering complete.")

    # --- 7. Determine output dir and write both curriculum files ---
    final_output_dir = output_dir
    if "mean" in metric_key:
        final_output_dir = os.path.join(output_dir, "mean")
    elif "variance" in metric_key:
        final_output_dir = os.path.join(output_dir, "variance")
    
    os.makedirs(final_output_dir, exist_ok=True)
    base_filename = os.path.splitext(os.path.basename(input_file))[0]

    # --- Write easy-to-hard file ---
    easy_to_hard_data = filtered_low + filtered_medium + filtered_high
    easy_to_hard_path = os.path.join(final_output_dir, f"{base_filename}_{metric_key}_easy_to_hard.jsonl")
    
    print(f"\n💾 Writing {len(easy_to_hard_data)} records to '{easy_to_hard_path}'...")
    with open(easy_to_hard_path, 'w', encoding='utf-8') as f:
        for item in tqdm(easy_to_hard_data, desc="Writing easy-to-hard"):
            f.write(json.dumps(item) + '\n')

    # --- Write hard-to-easy file ---
    hard_to_easy_data = filtered_high + filtered_medium + filtered_low
    hard_to_easy_path = os.path.join(final_output_dir, f"{base_filename}_{metric_key}_hard_to_easy.jsonl")

    print(f"\n💾 Writing {len(hard_to_easy_data)} records to '{hard_to_easy_path}'...")
    with open(hard_to_easy_path, 'w', encoding='utf-8') as f:
        for item in tqdm(hard_to_easy_data, desc="Writing hard-to-easy"):
            f.write(json.dumps(item) + '\n')
            
    print("\n🎉 All done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create easy-to-hard and hard-to-easy curriculum files from a JSONL file based on a metric.")
    
    parser.add_argument(
        "--input_file",
        type=str,
        default="metrics/outputs/completed/Qwen3_8B_entropy_20K_metrics.jsonl",
        help="Path to the input JSONL file."
    )
    parser.add_argument(
        "--metric",
        type=str,
        required=True,
        help="The metric key to build the curriculum from."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./curriculum_datasets",
        help="Directory to save the curriculum files."
    )
    
    args = parser.parse_args()
    
    try:
        if not os.path.exists(args.input_file):
            raise FileNotFoundError(f"Input file not found at '{args.input_file}'")
        check_metric_exists(args.input_file, args.metric)
        create_curriculum_files(args.input_file, args.metric, args.output_dir)
    except (ValueError, FileNotFoundError) as e:
        print(f"\n❌ Operation failed: {e}", file=sys.stderr)
        sys.exit(1)
