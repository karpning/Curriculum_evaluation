#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
import time
import asyncio
from typing import List, Dict, Any
import openai
from tqdm.asyncio import tqdm as async_tqdm

# --- Configuration for Local vLLM ---
VLLM_HOST = "localhost"
VLLM_PORT = "8002"
API_BASE = f"http://{VLLM_HOST}:{VLLM_PORT}/v1" 
API_KEY = "EMPTY"  # API key is not needed for local vLLM

# The model name must match the one loaded by your vLLM server.
MODEL_NAME = "Qwen/Qwen2.5-Math-72B-Instruct"

# MODIFIED: Updated file paths for MetaMathQA JSONL processing
INPUT_FILE = "../dataset/MetaMathQA-40K_original.jsonl"
OUTPUT_FILE = "outputs/MetaMathQA_symbol_complexity_40K.jsonl"

# --- ACCELERATION & STABILITY ---
MAX_CONCURRENT_REQUESTS = 10
CHUNK_SIZE = 100 # Process data in chunks of this size

# === Prompt for symbol complexity (Unchanged) ===
# === Updated prompt for symbol complexity with new clarifications ===
SYMBOL_COMPLEXITY_PROMPT_TEMPLATE = """You are a precise math curriculum analyzer. Your task is to rate the **mathematical notation complexity** of the provided problem text on a strict 1-5 scale.

Problem: {problem_text}

---
### SCORING GUIDE (1-5 Scale):

**Score 1: Basic Arithmetic**
- **Contains:** Only numbers and basic operators (+, -, =, ×, ÷).

**Score 2: Standard Algebra**
- **Contains:** Basic variables (x, y), parentheses, simple fractions, single exponents (x²).

**Score 3: Intermediate Symbols**
- **Contains:** Inequalities (<, ≤, >, ≥), roots (√, ³√), absolute value |x|, function notation f(x), factorial (!), pi (π).

**Score 4: Advanced Functions & Notation**
- **Contains:** Trigonometry (sin, cos), logarithms (log, ln), summation (∑), product (∏), derivatives (dy/dx, f'(x)), piecewise functions.

**Score 5: Highly Complex & Nested Notation**
- **Contains:** Integrals (∫), limits (lim), complex numbers (i), advanced set theory (∈, ∪, ∩), or deeply nested expressions.

---
### Final Instructions:

1.  **Highest Complexity Wins:** If a problem contains multiple symbols, the final score must be based on the **single most complex symbol** found. For example, a problem with both variables (Score 2) and a summation symbol (Score 4) must be rated as 4.
2.  **Generalize for Unlisted Symbols:** If you encounter a symbol not explicitly listed in the guide, classify it into the most appropriate category (e.g., "Standard Algebra," "Advanced Functions") and assign that category's score.

---
Based on these strict rules, return a JSON object with a single key "complexity_score".
"""

class SymbolComplexityExtractor:
    """
    Extracts a symbol complexity score from math problems concurrently
    by calling an OpenAI-compatible API using asyncio and processing in chunks.
    """

    def __init__(self):
        self.client = openai.AsyncOpenAI(
            api_key=API_KEY,
            base_url=API_BASE,
            timeout=120.0
        )
        self.semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
        print(f"✅ OpenAI client initialized for vLLM server at: {API_BASE}")
        print(f"Concurrency limit set to {MAX_CONCURRENT_REQUESTS}. Chunk size set to {CHUNK_SIZE}.")

    async def _call_api_with_retry(self, item_id: str, prompt: str) -> str:
        """Calls the API under the semaphore's control with retry logic."""
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

    async def process_chunk(self, chunk: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Creates, runs, and processes concurrent tasks for a single chunk of data.
        """
        async def process_item(item: Dict[str, Any]):
            # MODIFIED: Use the generated 'id' for logging
            item_id = item.get('id', 'N/A')
            
            # MODIFIED: Use the 'query' field for the problem text from MetaMathQA
            problem_text = item.get('query', '')
            if not problem_text:
                print(f"\n⚠️ Warning: Item {item_id} has an empty 'query' field. Skipping.")
                item['symbol_complexity'] = -1
                return item

            prompt = SYMBOL_COMPLEXITY_PROMPT_TEMPLATE.format(problem_text=problem_text)
            
            response_str = await self._call_api_with_retry(item_id, prompt)
            try:
                result = json.loads(response_str)
                score = result.get('complexity_score')
                if score is None:
                    raise KeyError("'complexity_score' key not found")
                
                item['symbol_complexity'] = max(1, min(5, int(score)))
            except Exception as e:
                print(f"\n⚠️ Warning: Failed to parse complexity for item {item_id}. Error: {e}. Response: '{response_str}'")
                item['symbol_complexity'] = -1
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
    """Main asynchronous execution function with chunking."""
    print("🚀 Starting CHUNKED symbol complexity extraction using local vLLM...")

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

    # MODIFIED: Overwrite the output file at the start of the run
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)
        print(f"🧹 Removed old output file: {OUTPUT_FILE}")

    extractor = SymbolComplexityExtractor()
    start_time = time.time()
    
    data_chunks = [source_data[i:i + CHUNK_SIZE] for i in range(0, len(source_data), CHUNK_SIZE)]
    
    all_processed_data = []
    for i, chunk in enumerate(data_chunks):
        print(f"\n--- Processing Chunk {i+1}/{len(data_chunks)} ---")
        processed_chunk = await extractor.process_chunk(chunk)
        all_processed_data.extend(processed_chunk)
        
        # MODIFIED: Save results for the current chunk to JSONL
        save_results_to_jsonl(processed_chunk, OUTPUT_FILE)
    
    print(f"\n--- All chunks processed and saved incrementally. ---")
    end_time = time.time()

    # --- Final Statistics ---
    print("\n\n=== Overall Processing Statistics ===")
    valid_scores = [
        item.get('symbol_complexity') for item in all_processed_data
        if item.get('symbol_complexity') not in (None, -1)
    ]
    total = len(all_processed_data)
    success = len(valid_scores)
    if success:
        print(f"📊 Symbol Complexity ({success}/{total} successful):")
        print(f"   - Min: {min(valid_scores)}")
        print(f"   - Max: {max(valid_scores)}")
        print(f"   - Average: {sum(valid_scores)/success:.2f}")
    else:
        print("No symbol complexity scores were successfully extracted.")

    total_time = end_time - start_time
    avg_time_per_sample = total_time / len(source_data) if source_data else 0

    print(f"\n🎉 Feature extraction complete!")
    print(f"📁 Results saved to: {OUTPUT_FILE}")
    print(f"⏱️  Total time: {total_time:.2f} seconds")
    print(f"📈 Average time per sample: {avg_time_per_sample:.2f} seconds")

if __name__ == "__main__":
    asyncio.run(main())