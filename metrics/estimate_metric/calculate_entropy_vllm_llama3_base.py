import json
import re
import asyncio
import numpy as np
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio
import os
import time
from typing import List, Dict, Any
import math

# --- Configuration ---
# NOTE: Update these paths and parameters for your setup.
INPUT_FILE = '../dataset/MetaMathQA-20K_train.jsonl'
# CHANGED: Updated output file names for the new model to avoid overwriting.
OUTPUT_FILE = 'outputs/Llama3_1_8B_base_entropy_20K_metrics_512.jsonl' 
FAILURE_LOG_FILE = 'outputs/Llama3.1-8B_base_entropy_20K_failed_512.log' 
K = 20
VLLM_API_BASE = "http://localhost:8003/v1"

# CHANGED: Updated the model name to Llama 3.1 8B.
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B"

# --- ACCELERATION & STABILITY ---
# NOTE: Reduced max concurrent requests as Llama-3.1-8B is larger and more VRAM-intensive than Gemma-4B.
# You may need to adjust this based on your hardware capacity.
MAX_CONCURRENT_REQUESTS = 20
CHUNK_SIZE = 100
API_TIMEOUT = 120.0
MAX_RETRIES = 1 # One retry pass for failed items


# # CHANGED: Rewrote the prompt formatting function for the Llama 3.1 chat template.
# def format_prompt(item: Dict[str, Any]):
#     """
#     Formats the prompt using the official Llama 3.1 chat template.
#     """
#     problem_query = item['query']
#     item_type = item.get('type', '')

#     # Construct the system prompt
#     base_instruction = "Solve the following problem by showing your work in clear, concise steps. Your response should be direct and to the point."
#     # if 'GSM' in item_type:
#     #     format_instruction = "Conclude by writing your final answer inside ####."
#     # else:
#     #     format_instruction = "Conclude by writing your final answer inside \\boxed{}."
#     format_instruction = "Conclude your response with a final line in the format: The answer is <numerical value>."

#     stop_instruction = "After providing the final answer, stop generating."
#     system_prompt = f"{base_instruction} {format_instruction} {stop_instruction}"

#     # Construct the user prompt
#     user_prompt = f"Problem: {problem_query}"

#     # Assemble the final prompt using the Llama 3.1 template
#     return (
#         f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
#         f"{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
#         f"{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
#         f"Solution:"
#     )

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

def calculate_shannon_entropy(logprobs: List[float]) -> float:
    """
    Calculates Shannon entropy from a list of log probabilities using log-space
    for numerical stability.
    """
    if not logprobs:
        return 0.0

    try:
        max_lp = max(logprobs)
        log_sum_exp = max_lp + math.log(sum(math.exp(lp - max_lp) for lp in logprobs))
    except ValueError:
        return 0.0

    entropy = 0.0
    for lp in logprobs:
        if lp > -float('inf'):
            p = math.exp(lp - log_sum_exp)
            if p > 0:
                entropy -= p * (lp - log_sum_exp) / math.log(2)
    return entropy
    
async def get_completions_with_logprobs(item: Dict[str, Any], client: AsyncOpenAI, n: int):
    """Generates `n` completions in a single API call, requesting logprobs, and returns a list of response-like objects."""
    formatted_prompt = format_prompt(item)
    try:
        response = await client.completions.create(
            model=MODEL_NAME,
            prompt=formatted_prompt,
            # CHANGED: Increased max_tokens to give the larger model more space for solutions.
            max_tokens=512,
            temperature=0.7,
            top_p=0.95,
            n=n,
            logprobs=5
            # stop=["<|eot_id|>"]
        )
        return [type("FakeResponse", (), {"choices": [choice]}) for choice in response.choices]
    except Exception as e:
        print(f"\n⚠️ Warning: An internal error occurred during an API call. Error: {e}. Returning empty list.")
        return []


def log_failure(item_id: str):
    with open(FAILURE_LOG_FILE, 'a', encoding='utf-8') as f: f.write(f"{item_id}\n")

# --- Main Execution Logic ---

async def process_chunk(chunk: List[Dict[str, Any]], client: AsyncOpenAI, semaphore: asyncio.Semaphore):
    """Creates, runs, and processes concurrent tasks for a single chunk of data."""
    async def process_item(item: Dict[str, Any]):
        """Processes a single problem to calculate all defined metrics."""
        item_id = item.get('id', 'N/A')
        async with semaphore:
            try:
                responses = await asyncio.wait_for(get_completions_with_logprobs(item, client, K), timeout=API_TIMEOUT)
                if not responses: raise ValueError("API returned an empty list of responses")
                
                nll_scores, nll_per_token_scores, logit_gap_means = [], [], []
                total_entropy_scores, mean_entropy_per_token_scores = [], []
                all_per_token_entropies = []
                
                for response in responses:
                    if not (response.choices and response.choices[0].logprobs): continue
                    
                    logprobs_obj = response.choices[0].logprobs
                    if not (logprobs_obj and logprobs_obj.token_logprobs): continue

                    valid_token_logprobs = [lp for lp in logprobs_obj.token_logprobs if lp is not None]
                    if not valid_token_logprobs:
                        print(f"Warning: No valid token logprobs found for a sample in item {item_id}")
                        continue

                    total_log_prob = sum(valid_token_logprobs)
                    num_tokens = len(valid_token_logprobs)
                    per_token_top5_entropies, per_token_logit_gaps = [], []
                    
                    if logprobs_obj.top_logprobs:
                        valid_indices = [i for i, lp in enumerate(logprobs_obj.token_logprobs) if lp is not None]
                        for i in valid_indices:
                            if i < len(logprobs_obj.top_logprobs) and logprobs_obj.top_logprobs[i]:
                                potential_dict = logprobs_obj.top_logprobs[i]
                                
                                if isinstance(potential_dict, list):
                                    top_logprobs_dict = dict(potential_dict)
                                else:
                                    top_logprobs_dict = potential_dict

                                top_logprobs_values = list(top_logprobs_dict.values())
                                per_token_top5_entropies.append(calculate_shannon_entropy(top_logprobs_values))
                                
                                if len(top_logprobs_values) >= 2:
                                    top_logprobs_values.sort(reverse=True)
                                    per_token_logit_gaps.append(top_logprobs_values[0] - top_logprobs_values[1])

                    if num_tokens == 0: continue
                    
                    nll_scores.append(-total_log_prob)
                    nll_per_token_scores.append(-total_log_prob / num_tokens)
                    
                    if per_token_top5_entropies:
                        total_entropy_scores.append(np.sum(per_token_top5_entropies))
                        mean_entropy_per_token_scores.append(np.mean(per_token_top5_entropies))
                        all_per_token_entropies.extend(per_token_top5_entropies)

                    if per_token_logit_gaps: logit_gap_means.append(np.mean(per_token_logit_gaps))

                if not nll_scores: raise ValueError("Could not calculate NLL for any sample.")

                item['mean_nll'] = round(np.mean(nll_scores), 9)
                item['variance_nll'] = round(np.var(nll_scores), 9)
                item['mean_nll_per_token'] = round(np.mean(nll_per_token_scores), 9)
                item['variance_nll_per_token'] = round(np.var(nll_per_token_scores), 9)
                item['mean_top5_shannon_entropy'] = round(np.mean(all_per_token_entropies) if all_per_token_entropies else -1.0, 9)
                item['variance_top5_shannon_entropy'] = round(np.var(all_per_token_entropies) if all_per_token_entropies else -1.0, 9)
                item['mean_sequence_entropy'] = round(np.mean(total_entropy_scores) if total_entropy_scores else -1.0, 9)
                item['variance_sequence_entropy'] = round(np.var(total_entropy_scores) if total_entropy_scores else -1.0, 9)
                item['mean_token_entropy'] = round(np.mean(mean_entropy_per_token_scores) if mean_entropy_per_token_scores else -1.0, 9)
                item['variance_token_entropy'] = round(np.var(mean_entropy_per_token_scores) if mean_entropy_per_token_scores else -1.0, 9)
                item['mean_logit_gap'] = round(np.mean(logit_gap_means) if logit_gap_means else -1.0, 9)
                item['variance_logit_gap'] = round(np.var(logit_gap_means) if logit_gap_means else -1.0, 9)
                
                item['status'] = 'success'
            except (asyncio.TimeoutError, Exception) as e:
                print(f"\n❌ Error processing item {item_id}: {e}")
                log_failure(item_id)
                all_metrics = [
                    'mean_nll', 'variance_nll', 'mean_nll_per_token', 'variance_nll_per_token',
                    'mean_top5_shannon_entropy', 'variance_top5_shannon_entropy',
                    'mean_sequence_entropy', 'variance_sequence_entropy', 
                    'mean_token_entropy', 'variance_token_entropy',
                    'mean_logit_gap', 'variance_logit_gap'
                ]
                item.update({k: -1.0 for k in all_metrics})
                item['status'] = 'failed_timeout' if isinstance(e, asyncio.TimeoutError) else 'failed_exception'
            return item
    tasks = [process_item(item) for item in chunk]
    processed_chunk = await tqdm_asyncio.gather(*tasks, desc="⚙️  Processing Chunk")
    return processed_chunk

async def run_processing(data_to_process: List[Dict[str, Any]]):
    """A wrapper to run the processing logic for a given list of items."""
    if not data_to_process: return
    print(f"✅ Processing {len(data_to_process)} items...")
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
    if not os.path.exists(OUTPUT_FILE): return
    print("\n--- Starting Final Cleanup and Sorting of Output File ---")
    latest_results = {}
    with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line)
                item_id = item.get('id')
                if item_id: latest_results[item_id] = item
            except json.JSONDecodeError: print(f"Skipping corrupted line during cleanup: {line.strip()}")
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
            for line in f: failed_ids.add(line.strip())
        
        items_to_retry = [item for item in all_source_data if item['id'] in failed_ids]
        
        if items_to_retry:
            print(f"Found {len(items_to_retry)} items in the fail log. Retrying once...")
            open(FAILURE_LOG_FILE, 'w').close()
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