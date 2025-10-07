#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
import time
import asyncio
from typing import List, Dict, Any
import openai
from tqdm.asyncio import tqdm as async_tqdm

# --- Configuration ---
VLLM_HOST = "localhost"
VLLM_PORT = "8001"
BASE_URL = f"http://{VLLM_HOST}:{VLLM_PORT}/v1"

# MODIFIED: Updated filenames for JSONL format
INPUT_FILE = "../dataset/MetaMathQA-40K_original.jsonl" 
OUTPUT_FILE = "outputs/MetaMathQA_reasoning_steps_40K.jsonl"
MODEL_NAME = "Qwen/Qwen2.5-Math-72B-Instruct" 

# --- ACCELERATION & STABILITY ---
MAX_CONCURRENT_REQUESTS = 10
CHUNK_SIZE = 100

# === Updated and consolidated prompt for reasoning steps ===
REASONING_STEPS_PROMPT_TEMPLATE = """You are a strict and precise math curriculum analyzer. Your task is to count the exact number of distinct logical reasoning steps in the provided solution text.

Solution: {solution_text}

---
### RULES FOR COUNTING STEPS:

**1. A single logical step is ONE of the following:**
- **Applying a stated theorem or formula** (e.g., "Using Vieta's formulas...").
- **Factoring an expression** (e.g., factoring a quadratic).
- **Substituting values** into a formula or expression.
- **A major algebraic transformation** (e.g., completing the square, cross-multiplication).
- **A clear logical deduction** to reach a new conclusion (e.g., "Since the discriminant is negative...").
- **Grouping similar, repetitive calculations** into a single conceptual step (e.g. finding multiple values from a list of pairs is one step).


**2. DO NOT COUNT the following:**
- Basic arithmetic operations (e.g., `16 * 2 = 32`, `10/5=2`).
- Solving a simple one-step linear equation (e.g., `6x = 12` which leads to `x=2`).
- Simply restating the problem or the final boxed answer.
- Steps that are implied but not explicitly written in the solution text.

---
### EXAMPLES:

#### Example 1:
- **Solution:** `x²-9=0` -> `(x-3)(x+3)=0` -> `x=3` or `x=-3`.
- **Analysis:** Step 1 is factoring. Step 2 is deducing the roots from the factors.
- **Correct Count:** {{"steps": 2}}

#### Example 2:
- **Solution:** "$x^2 - y^2$ factors into $(x+y)(x-y)$, so we multiply $16 \\cdot 2$ to get 32."
- **Analysis:** Step 1 is factoring. Step 2 is substituting. The final multiplication is simple arithmetic and is **not** counted as a step.
- **Correct Count:** {{"steps": 2}}

---
Based on these strict rules, count the total number of logical steps (from 1 to 15) and return it in a JSON object with a single key "steps".
"""


class ReasoningStepExtractor:
    """
    Extracts the number of reasoning steps from math solutions concurrently
    by calling a vLLM OpenAI-compatible API using asyncio.
    """

    def __init__(self):
        self.client = openai.AsyncOpenAI(
            base_url=BASE_URL,
            api_key="EMPTY",
        )
        self.semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
        print(f"✅ OpenAI client initialized for vLLM server at: {BASE_URL}")
        print(f"Concurrency limit set to {MAX_CONCURRENT_REQUESTS}. Chunk size set to {CHUNK_SIZE}.")

    async def _call_vllm_api(self, item_id: str, prompt: str) -> str:
        """Calls the vLLM API under the semaphore's control with retry logic."""
        async with self.semaphore:
            for attempt in range(3):
                try:
                    response = await self.client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=[
                            {"role": "system", "content": "You are a math analysis assistant that returns responses in JSON format."},
                            {"role": "user", "content": prompt}
                        ],
                        response_format={"type": "json_object"},
                        max_tokens=500,
                        temperature=0.0
                    )
                    return response.choices[0].message.content
                except Exception as e:
                    print(f"\n❌ API Error for item {item_id} (Attempt {attempt + 1}/3): {type(e).__name__} - {e}")
                    if "response_format" in str(e):
                        print("Error: The vLLM server may not support the 'response_format' parameter.")
                        return "{}"
                    if attempt < 2:
                        print(f"Retrying in 2 seconds...")
                        await asyncio.sleep(2)
            
            print(f"❌ All retries failed for item {item_id}. Returning empty response.")
            return "{}"

    async def process_chunk(self, chunk: List[Dict[str, Any]]):
        """
        Creates, runs, and processes concurrent tasks for a single chunk of data.
        """
        async def process_item(item: Dict[str, Any]):
            # MODIFIED: Use the generated id for logging
            item_id = item.get('id', 'N/A')
            
            # MODIFIED: Use the 'response' field for the solution text
            solution_text = item.get('response', '')
            if not solution_text:
                print(f"\n⚠️ Warning: Item {item_id} has an empty 'response' field. Skipping.")
                item['reasoning_steps'] = -1
                return item

            prompt = REASONING_STEPS_PROMPT_TEMPLATE.format(solution_text=solution_text)
            
            response_str = await self._call_vllm_api(item_id, prompt)
            try:
                result = json.loads(response_str)
                steps = result.get('steps')
                if steps is None:
                    raise KeyError("'steps' key not found")
                
                # Add the new key to the original item dictionary
                item['reasoning_steps'] = max(1, min(15, int(steps)))
            except Exception as e:
                print(f"\n⚠️ Warning: Failed to parse steps for item {item_id}. Error: {e}. Response: '{response_str}'")
                item['reasoning_steps'] = -1
            return item

        tasks = [process_item(item) for item in chunk]
        processed_chunk = await async_tqdm.gather(*tasks, desc="⚙️ Processing Chunk")
        return processed_chunk

# NEW: Simplified function to save results to a JSONL file
def save_results_to_jsonl(data: List[Dict[str, Any]], filepath: str):
    """Appends a list of dictionaries to a JSONL file."""
    try:
        with open(filepath, 'a', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')
    except IOError as e:
        print(f"❌ Error saving to file {filepath}: {e}")


async def main():
    """Main asynchronous execution function."""
    print("🚀 Starting CHUNKED reasoning step extraction using vLLM...")

    # Ensure output directory exists
    output_dir = os.path.dirname(OUTPUT_FILE)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"📁 Created output directory: {output_dir}")

    if not os.path.exists(INPUT_FILE):
        print(f"❌ Error: Input file not found at {INPUT_FILE}")
        return

    # MODIFIED: Load data from JSONL file line by line
    source_data = []
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                item = json.loads(line)
                # Add a unique ID for tracking purposes
                item['id'] = f"train_{i}" 
                source_data.append(item)
        print(f"✅ Loaded {len(source_data)} samples from {INPUT_FILE}")
    except Exception as e:
        print(f"❌ Error loading input file: {e}")
        return

    # Overwrite the output file at the start of the run
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)

    extractor = ReasoningStepExtractor()
    start_time = time.time()
    
    data_chunks = [source_data[i:i + CHUNK_SIZE] for i in range(0, len(source_data), CHUNK_SIZE)]
    
    all_processed_data = []
    for i, chunk in enumerate(data_chunks):
        print(f"\n--- Processing Chunk {i+1}/{len(data_chunks)} ---")
        processed_chunk = await extractor.process_chunk(chunk)
        all_processed_data.extend(processed_chunk)
        
        # Save results for the current chunk
        save_results_to_jsonl(processed_chunk, OUTPUT_FILE)
    
    print(f"\n--- All chunks processed and saved incrementally. ---")
    end_time = time.time()

    # --- Final Statistics ---
    print("\n\n=== Overall Processing Statistics ===")
    valid_scores = [
        item.get('reasoning_steps') for item in all_processed_data
        if item.get('reasoning_steps') not in (None, -1)
    ]
    total = len(all_processed_data)
    success = len(valid_scores)
    if success:
        print(f"📊 Reasoning Steps ({success}/{total} successful):")
        print(f"   - Min: {min(valid_scores)}")
        print(f"   - Max: {max(valid_scores)}")
        print(f"   - Average: {sum(valid_scores)/success:.2f}")
    else:
        print("No reasoning steps were successfully extracted.")

    total_time = end_time - start_time
    avg_time_per_sample = total_time / len(source_data) if source_data else 0

    print(f"\n🎉 Feature extraction complete!")
    print(f"📁 Results saved to: {OUTPUT_FILE}")
    print(f"⏱️  Total time: {total_time:.2f} seconds")
    print(f"📈 Average time per sample: {avg_time_per_sample:.2f} seconds")

if __name__ == "__main__":
    asyncio.run(main())