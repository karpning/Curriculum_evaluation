import json
import os
import argparse
from tqdm import tqdm
import random

def create_baseline_datasets(input_file: str, output_dir: str):
    """
    Loads a JSONL file, filters out metric keys, and creates two baseline datasets:
    1. A fully shuffled version of the entire dataset.
    2. A 1/3-sized, shuffled random subset.
    """
    # --- 1. Validate inputs and create output directory ---
    if not os.path.exists(input_file):
        print(f"❌ Error: Input file not found at '{input_file}'")
        return

    os.makedirs(output_dir, exist_ok=True)
    print(f"📂 Output directory: '{output_dir}'")

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

    # --- 3. Filter the data to keep only essential keys ---
    print("\n📝 Filtering records to remove metric keys...")
    base_keys_to_keep = ['query', 'response', 'type', 'id']
    
    def filter_item_keys(item):
        return {key: item.get(key) for key in base_keys_to_keep if key in item}

    filtered_data = [filter_item_keys(item) for item in data]
    print("✅ Filtering complete.")

    # --- 4. Create the shuffled datasets ---
    print("\n🔀 Shuffling data to create baselines...")
    
    # Create a shuffled copy for the full baseline
    full_shuffled_data = filtered_data.copy()
    random.shuffle(full_shuffled_data)
    
    # Create the 1/3 subset from the shuffled data
    subset_size = len(full_shuffled_data) // 3
    subset_shuffled_data = full_shuffled_data[:subset_size]
    
    print(f"✅ Created full shuffled baseline ({len(full_shuffled_data)} items) and 1/3 subset ({len(subset_shuffled_data)} items).")

    # --- 5. Write the baseline data to new files ---
    base_filename = os.path.splitext(os.path.basename(input_file))[0]
    
    def write_to_jsonl(filepath, data_list, desc):
        print(f"\n💾 Writing {len(data_list)} records to '{filepath}'...")
        with open(filepath, 'w', encoding='utf-8') as f:
            for item in tqdm(data_list, desc=desc):
                f.write(json.dumps(item) + '\n')

    # Define paths and write each file
    output_full_path = os.path.join(output_dir, f"{base_filename}_baseline_full_shuffled.jsonl")
    write_to_jsonl(output_full_path, full_shuffled_data, "Writing FULL baseline")
    
    output_subset_path = os.path.join(output_dir, f"{base_filename}_baseline_subset_shuffled.jsonl")
    write_to_jsonl(output_subset_path, subset_shuffled_data, "Writing SUBSET baseline")
            
    print("\n🎉 All done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create baseline (control group) datasets from a JSONL file.")
    
    parser.add_argument(
        "--input_file",
        type=str,
        default="Processed_data/Baseline_data/MetaMathQA_20K.jsonl",
        help="Path to the input JSONL file containing all metrics."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="Processed_data/Baseline_data",
        help="Directory to save the baseline dataset files."
    )
    
    args = parser.parse_args()
    
    create_baseline_datasets(args.input_file, args.output_dir)