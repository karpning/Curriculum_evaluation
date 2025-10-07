import json
import os
import argparse
import sys
from tqdm import tqdm

def check_metric_exists(input_file: str, metric_key: str):
    """
    Efficiently checks if the metric_key exists in the first valid JSON object of the file.
    Raises ValueError if the key is not found or the file is empty/corrupt.
    """
    print(f"🔎 Verifying if metric '{metric_key}' exists in the first record of '{input_file}'...")
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            # Skip empty lines
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
                
                # Metric found, no need to check further
                print(f"✅ Metric key '{metric_key}' found.")
                return
            except json.JSONDecodeError:
                # If the first line is corrupt, we can't verify, so we raise an error.
                raise ValueError(f"Could not decode the first line. File might be corrupted: {line.strip()}")

    # If the loop finishes, the file was empty or contained only empty lines.
    raise ValueError("Input file is empty or contains no valid JSON objects.")


def sort_jsonl_by_metric(input_file: str, metric_key: str, output_dir: str):
    """
    Loads a JSONL file, sorts it by a specific metric, filters to keep only essential keys,
    and saves two new files: one sorted ascending and one descending.
    The output directory is determined by the metric name.
    """
    # --- 1. Create base output directory ---
    # Input file existence is checked by check_metric_exists
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
        print("❌ Error: No data was loaded from the file. Exiting.")
        return
        
    print(f"✅ Loaded {len(data)} records successfully.")

    # --- 3. Sort the data ---
    # Use a default value of 0.0 for items that might be missing the metric key
    sort_key = lambda item: item.get(metric_key, 0.0)

    print(f"\n🔄 Sorting by '{metric_key}' in ascending order...")
    sorted_asc = sorted(data, key=sort_key)
    
    print(f"🔄 Sorting by '{metric_key}' in descending order...")
    sorted_desc = sorted(data, key=sort_key, reverse=True)
    print("✅ Sorting complete.")

    # --- 4. Filter the data to keep only specified keys ---
    print("\n📝 Filtering records to keep only essential keys...")
    
    base_keys_to_keep = ['query', 'response', 'type', 'id']
    keys_to_keep = list(set(base_keys_to_keep + [metric_key]))
    
    def filter_item_keys(item):
        return {key: item.get(key) for key in keys_to_keep if key in item}

    filtered_asc = [filter_item_keys(item) for item in sorted_asc]
    filtered_desc = [filter_item_keys(item) for item in sorted_desc]
    print("✅ Filtering complete.")

    # --- 5. Determine output directory and write the filtered/sorted data ---
    final_output_dir = output_dir
    if "mean" in metric_key:
        final_output_dir = os.path.join(output_dir, "mean")
    elif "variance" in metric_key:
        final_output_dir = os.path.join(output_dir, "variance")
    
    os.makedirs(final_output_dir, exist_ok=True)
    
    base_filename = os.path.splitext(os.path.basename(input_file))[0]
    
    output_asc_path = os.path.join(final_output_dir, f"{base_filename}_{metric_key}_asc.jsonl")
    print(f"\n💾 Writing ascending results to '{output_asc_path}'...")
    with open(output_asc_path, 'w', encoding='utf-8') as f:
        for item in tqdm(filtered_asc, desc="Writing ASC"):
            f.write(json.dumps(item) + '\n')

    output_desc_path = os.path.join(final_output_dir, f"{base_filename}_{metric_key}_desc.jsonl")
    print(f"💾 Writing descending results to '{output_desc_path}'...")
    with open(output_desc_path, 'w', encoding='utf-8') as f:
        for item in tqdm(filtered_desc, desc="Writing DESC"):
            f.write(json.dumps(item) + '\n')
            
    print("\n🎉 All done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sort a JSONL file by a specified numerical metric and filter its content.")
    
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
        help="The metric key to sort the file by."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./sorted_outputs",
        help="Directory to save the sorted output files."
    )
    
    args = parser.parse_args()
    
    try:
        # [MODIFIED] Validate inputs before starting the main process
        if not os.path.exists(args.input_file):
            raise FileNotFoundError(f"Input file not found at '{args.input_file}'")
        check_metric_exists(args.input_file, args.metric)
        sort_jsonl_by_metric(args.input_file, args.metric, args.output_dir)
    except (ValueError, FileNotFoundError) as e:
        print(f"\n❌ Operation failed: {e}", file=sys.stderr)
        sys.exit(1)
