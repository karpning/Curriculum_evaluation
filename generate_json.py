from datasets import load_dataset
import os

# --- Step 1: Load the dataset from Hugging Face Hub ---
print("Loading dataset from Hugging Face Hub...")
dataset = load_dataset("meta-math/MetaMathQA-40K")


train_dataset = dataset['train']

# Display the first example to confirm the original format.
print("\nFirst example from the original dataset:")
print(train_dataset[0])

# --- Step 2: Save the FULL original data (40k) to a JSON Lines file ---

# Define the output directory and filename.
output_dir = "dataset"
full_output_filename = os.path.join(output_dir, "MetaMathQA-40K_original.jsonl")

# Create the directory if it doesn't exist.
os.makedirs(output_dir, exist_ok=True)

print(f"\nSaving the full original data to {full_output_filename}...")

# Save the original train_dataset directly.
train_dataset.to_json(full_output_filename, orient="records", lines=True)

print(f"Successfully saved {len(train_dataset)} records in their original format.")


# --- Step 3: Randomly sample 20k records and save to a new file ---

num_samples_to_use = 20000
print(f"\nRandomly sampling {num_samples_to_use} records...")

# Shuffle the dataset with a fixed seed for reproducibility.
shuffled_dataset = train_dataset.shuffle(seed=42)

# Select the first 20,000 records from the shuffled dataset.
subset_dataset = shuffled_dataset.select(range(num_samples_to_use))

# Define the new filename for the 20k subset.
subset_output_filename = os.path.join(output_dir, "MetaMathQA-20K_train.jsonl")

print(f"Saving the {num_samples_to_use}-record subset to {subset_output_filename}...")

# Save the new smaller dataset to the specified file.
subset_dataset.to_json(subset_output_filename, orient="records", lines=True)

print(f"Successfully saved {len(subset_dataset)} random records.")