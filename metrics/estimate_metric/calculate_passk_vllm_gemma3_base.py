import json
import re
import asyncio
import numpy as np
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio
import os
import time
import math
from typing import List, Dict, Any, Tuple

# --- Configuration ---
# NOTE: Update these paths and parameters for your setup.
INPUT_FILE = '../dataset/MetaMathQA-20K_train.jsonl'
OUTPUT_FILE = 'outputs/Gemma_4B_base_pass_k_20K_metrics_1024.jsonl'
FAILURE_LOG_FILE = 'outputs/Gemma_4B_base_pass_k_20K_failed_1024.log'
K = 20
VLLM_API_BASE = "http://localhost:8002/v1"
MODEL_NAME="google/gemma-3-4b-pt"

# --- ACCELERATION & STABILITY ---
MAX_CONCURRENT_REQUESTS = 20
CHUNK_SIZE = 100
API_TIMEOUT = 120.0

# --- Helper Functions ---

def parse_and_clean_value(s: str) -> str | None:
    """
    Helper function to clean the string extracted from a marker.
    It handles LaTeX fractions, extracts the last number, and removes commas.
    """
    if not s:
        return None
    
    # Handle LaTeX fractions like \frac{a}{b}
    frac_match = re.search(r'\\frac{(\d+)}{(\d+)}', s)
    if frac_match:
        num = int(frac_match.group(1))
        den = int(frac_match.group(2))
        if den != 0:
            return str(num / den)

    # Remove commas from numbers (e.g., "1,000" -> "1000")
    s = s.replace(',', '')

    # Use a more robust regex to match all number formats, including ".5"
    numbers = re.findall(r'-?(?:\d+\.?\d*|\d*\.\d+)', s)
    if numbers:
        return numbers[-1]
    
    return None

def extract_final_answer(text: str) -> str | None:
    """
    Improved final answer extraction, prioritizing "The answer is..." format.
    Handles answers with or without brackets.
    """
    if not text:
        return None

    # Tier 1: Look for "The answer is..." on the final line. This is the most reliable pattern.
    final_answer_match = re.search(
        r'The answer is[:\s]*\[?(.+?)\]?\.?\s*$', 
        text, 
        re.IGNORECASE | re.DOTALL
    )
    if final_answer_match:
        content = final_answer_match.group(1).strip()
        candidate = parse_and_clean_value(content)
        if candidate:
            return candidate

    # Tier 2: Fallback to \boxed{} marker (if the primary pattern fails)
    boxed_match = re.search(r'\\boxed{([\s\S]*?)}', text)
    if boxed_match:
        content = boxed_match.group(1)
        candidate = parse_and_clean_value(content)
        if candidate:
            return candidate

    # Tier 3: Fallback to #### marker
    hash_match = re.search(r'####[^\S\r\n]*(.*)', text)
    if hash_match:
        line_content = hash_match.group(1) or ""
        candidate = parse_and_clean_value(line_content)
        if candidate:
            return candidate
        
        text_after_marker = text[hash_match.end():]
        if text_after_marker.lstrip():
            next_line = text_after_marker.lstrip().splitlines()[0]
            candidate_next_line = parse_and_clean_value(next_line)
            if candidate_next_line:
                return candidate_next_line

    # Tier 4: Fallback to the last valid number in the entire text as a last resort
    numbers = re.findall(r'-?(?:\d+\.?\d*|\d*\.\d+)', text)
    if numbers:
        valid_numbers = [num for num in numbers if len(num) <= 15]
        if valid_numbers:
            return valid_numbers[-1]
    
    return None


def is_correct(model_response: str, ground_truth: str) -> bool:
    """
    Checks correctness by applying the SAME robust extraction logic to both model and ground truth.
    """
    model_answer_str = extract_final_answer(model_response)
    correct_answer_str = extract_final_answer(ground_truth)
    # print("--------------------------------------------------")
    # print(f"Model Output: {model_response.strip()[:250]}...")
    # print(f"Extracted Model Answer: {model_answer_str}")
    # print(f"Ground Truth Answer: {correct_answer_str}")
    # print("--------------------------------------------------")

    if model_answer_str is None or correct_answer_str is None:
        return False

    try:
        model_num = float(model_answer_str.strip())
        correct_num = float(correct_answer_str.strip())
        
        is_match = math.isclose(model_num, correct_num, rel_tol=1e-9, abs_tol=1e-9)
        return is_match
    
    except (ValueError, TypeError):
        normalized_model_ans = model_answer_str.strip().lower()
        normalized_correct_ans = correct_answer_str.strip().lower()
        is_match = normalized_model_ans == normalized_correct_ans
        return is_match

# def format_prompt(item: Dict[str, Any]):
#     """
#     Formats the prompt with clear instructions for the model.
#     """
#     problem_query = item['query']

#     base_instruction = "Solve the following problem by showing your work in clear, concise steps. Your response should be direct and to the point."
#     format_instruction = "Conclude your response with a final line in the format: The answer is <numerical value>."
#     stop_instruction = "After providing the final answer, stop generating."

#     full_instruction = f"{base_instruction} {format_instruction} {stop_instruction}"

#     return f"""{full_instruction}
# Problem: {problem_query}
# Solution:"""

def format_prompt(item: Dict[str, Any]):
    """
    Formats a simple completion prompt for a base model
    with a Chain-of-Thought (CoT) trigger.
    """
    problem_query = item['query']
    
    # Provide the problem and cue the model to start the solution
    # by thinking step-by-step.
    return (
        f"Problem: {problem_query}\n"
        f"Solution: Let's think step by step.\n"
    )


async def get_completions(item: Dict[str, Any], client: AsyncOpenAI, n: int) -> List[str]:
    """Generates `n` completions in a single, more efficient API call."""
    formatted_prompt = format_prompt(item)
    try:
        response = await client.completions.create(
            model=MODEL_NAME,
            prompt=formatted_prompt,
            max_tokens=1024,
            temperature=0.7,
            top_p=0.95,
            n=n,
            stop=["<end_of_turn>"] 
        )
        return [choice.text for choice in response.choices]
    except Exception as e:
        print(f"\n⚠️ Warning: An internal error occurred during an API call. Error: {e}. Returning an empty list.")
        return []

def log_failure(item_id: str):
    """Logs the ID of a failed item to the failure log file."""
    with open(FAILURE_LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{item_id}\n")

# --- Main Execution Logic ---

async def process_chunk(chunk: List[Dict[str, Any]], client: AsyncOpenAI, semaphore: asyncio.Semaphore):
    """Creates, runs, and processes concurrent tasks for a single chunk of data."""
    async def process_item(item: Dict[str, Any]):
        item_id = item.get('id', 'N/A')
        async with semaphore:
            try:
                completions = await asyncio.wait_for(get_completions(item, client, K), timeout=API_TIMEOUT)
                if not completions:
                    raise ValueError("API returned an empty list of completions")
                
                ground_truth_response = item['response']
                successes = []
                
                for i, completion_text in enumerate(completions):
                    cleaned_text = re.sub(r'<start_of_turn>(?:user|model)\s*|<end_of_turn>\s*', '', completion_text, flags=re.IGNORECASE).strip()
                    is_match = is_correct(cleaned_text, ground_truth_response)
                    successes.append(is_match)

                item['pass@k'] = round(np.mean(successes), 9)
                item['pass@k_variance'] = round(np.var(successes), 9)
                item['status'] = 'success'
            except (asyncio.TimeoutError, Exception) as e:
                print(f"\n❌ Error processing item {item_id}: {e}")
                log_failure(item_id)
                item['pass@k'] = -1.0
                item['pass@k_variance'] = -1.0
                item['status'] = 'failed_timeout' if isinstance(e, asyncio.TimeoutError) else 'failed_exception'
            return item
            
    tasks = [process_item(item) for item in chunk]
    processed_chunk = await tqdm_asyncio.gather(*tasks, desc="⚙️  Processing Chunk")
    return processed_chunk

# --- Orchestration Functions (Unchanged) ---

async def run_processing(data_to_process: List[Dict[str, Any]]):
    """A wrapper to run the processing logic for a given list of items."""
    if not data_to_process:
        return
    
    print(f"Starting processing for {len(data_to_process)} items...")
    client = AsyncOpenAI(base_url=VLLM_API_BASE, api_key="NOT_USED")
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    data_chunks = [data_to_process[i:i + CHUNK_SIZE] for i in range(0, len(data_to_process), CHUNK_SIZE)]
    
    with open(OUTPUT_FILE, 'a', encoding='utf-8') as f_out:
        for i, chunk in enumerate(data_chunks):
            print(f"\n--- Processing Chunk {i+1}/{len(data_chunks)} ({len(chunk)} items) ---")
            processed_chunk = await process_chunk(chunk, client, semaphore)
            print(f"--- Appending results for Chunk {i+1}... ---")
            for item in processed_chunk:
                f_out.write(json.dumps(item, ensure_ascii=False) + '\n')

def cleanup_and_sort_output():
    """Reads the output file, de-duplicates, sorts by ID, and overwrites."""
    if not os.path.exists(OUTPUT_FILE):
        return
    
    print("\n--- Starting Final Cleanup and Sorting of Output File ---")
    latest_results = {}
    with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line)
                item_id = item.get('id')
                if item_id:
                    latest_results[item_id] = item
            except json.JSONDecodeError:
                print(f"Skipping corrupted line during cleanup: {line.strip()}")

    try:
        sorted_results = sorted(latest_results.values(), key=lambda x: int(re.search(r'\d+', x.get('id', 'item_-1')).group()))
    except (ValueError, IndexError, AttributeError):
        print("Warning: Could not sort by ID numerically. Saving in collected order.")
        sorted_results = list(latest_results.values())
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for item in sorted_results:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"✅ Cleanup complete. {len(sorted_results)} unique and sorted results saved to {OUTPUT_FILE}.")

async def main():
    """Main function to orchestrate the entire run-retry-cleanup workflow."""
    start_time = time.time()
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    if os.path.exists(FAILURE_LOG_FILE): os.remove(FAILURE_LOG_FILE)
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Error: Input file not found at '{INPUT_FILE}'")
        return

    # --- Phase 1: Initial Run ---
    print("🚀 Starting Initial Run ---")
    all_source_data = []
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            item = json.loads(line)
            if 'id' not in item: item['id'] = f"item_{i}"
            all_source_data.append(item)
    print(f"✅ Loaded {len(all_source_data)} total problems from '{INPUT_FILE}'.")

    completed_ids = set()
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f_done:
            for line in f_done:
                try:
                    obj = json.loads(line)
                    if obj.get('status') == 'success': completed_ids.add(obj['id'])
                except json.JSONDecodeError: print(f"Skipping corrupted line: {line.strip()}")
    
    initial_data_to_process = [item for item in all_source_data if item['id'] not in completed_ids]
    
    if initial_data_to_process:
        print(f"✅ Processing {len(initial_data_to_process)} new or previously failed items.")
        await run_processing(initial_data_to_process)
    else:
        print("✅ No new items to process for the initial run.")

    # --- Phase 2: Failure Retry ---
    print("\n--- Checking for Failed Items to Retry ---")
    if os.path.exists(FAILURE_LOG_FILE) and os.path.getsize(FAILURE_LOG_FILE) > 0:
        failed_ids = set()
        with open(FAILURE_LOG_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                failed_ids.add(line.strip())
        
        items_to_retry = [item for item in all_source_data if item['id'] in failed_ids]
        
        if items_to_retry:
            print(f"Found {len(items_to_retry)} items in the fail log. Retrying once...")
            open(FAILURE_LOG_FILE, 'w').close() # Clear the log before retry
            await run_processing(items_to_retry)
        else:
            print("✅ Fail log found, but no matching items in source data.")
    else:
        print("✅ No failures recorded. Skipping retry phase.")

    # --- Phase 3: Final Cleanup and Sorting ---
    cleanup_and_sort_output()

    total_time = time.time() - start_time
    print(f"\n\n🎉 All operations complete! It took {total_time:.2f} seconds.")
    print(f"Final results are in: {OUTPUT_FILE}")
    if os.path.exists(FAILURE_LOG_FILE) and os.path.getsize(FAILURE_LOG_FILE) > 0: 
        print(f"⚠️ Some items still failed after retry. See {FAILURE_LOG_FILE} for details.")
    
if __name__ == "__main__":
    asyncio.run(main())