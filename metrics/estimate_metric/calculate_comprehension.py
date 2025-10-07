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
VLLM_PORT = "8002" # MODIFIED: Matched port from previous script
BASE_URL = f"http://{VLLM_HOST}:{VLLM_PORT}/v1"

# MODIFIED: Updated filenames for JSONL format and new task
INPUT_FILE = "../dataset/MetaMathQA-40K_original.jsonl" 
OUTPUT_FILE = "outputs/MetaMathQA_comprehension_difficulty_40K.jsonl"
MODEL_NAME = "Qwen/Qwen2.5-Math-72B-Instruct" 

# --- ACCELERATION & STABILITY ---
MAX_CONCURRENT_REQUESTS = 10
CHUNK_SIZE = 100

# === Prompt for Comprehension Difficulty ===
COMPREHENSION_DIFFICULTY_PROMPT_TEMPLATE = """You are a strict and precise math curriculum analyzer. Your task is to rate the comprehension difficulty of the provided math problem on a scale of 1-10.

This score must reflect how hard it is to understand WHAT is being asked. Consider the language, concepts, and required background knowledge. This is NOT about how hard the problem is to SOLVE. You must use the full 1-10 scoring range.

Problem: {query_text}

---
### SCORING GUIDE (1-5 Scale):

**Score 1: Direct Calculation**
- **Definition:** A straightforward command with no context or story.
- **Example:** "Evaluate $(2^2)^3$."

**Score 2: Simple Context or Notation**
- **Definition:** A direct question that involves minimal context, a simple word problem, or basic function notation.
- **Example:** "If $g(x) = 3x + 7$ and $f(x) = 5x - 9$, what is $f(g(8))$?"

**Score 3: Standard Word Problem**
- **Definition:** Requires translating a multi-sentence scenario into a mathematical setup.
- **Example:** "A rectangle's length is twice its width. If its perimeter is 18, what is its area?"

**Score 4: Complex Notation or Visuals**
- **Definition:** The problem involves specialized mathematical notation (calculus, logarithms, advanced functions) or requires interpreting a visual diagram/code (`[asy]`).
- **Example:** "Let f(x) = sin(x²). Find f'(x)." or problems with diagrams.

**Score 5: Abstract & Multi-Constraint**
- **Definition:** The problem involves abstract concepts (like invertible functions, group theory) or has multiple, layered constraints that must be understood together.
- **Example:** "Prove that for any cyclic group G of prime order p..."

---
Based on these strict rules, return a JSON object with a single key "difficulty".
"""

# COMPREHENSION_DIFFICULTY_PROMPT_TEMPLATE = """You are a strict and precise math curriculum analyzer. Your task is to rate the comprehension difficulty of the provided math problem on a scale of 1-10.

# This score must reflect how hard it is to understand WHAT is being asked. Consider the language, concepts, and required background knowledge. This is NOT about how hard the problem is to SOLVE. You must use the full 1-10 scoring range.

# Problem: {query_text}

# ---
# ### SCORING GUIDE (1-10 Scale):

# **Score 1: Pure Symbolic Calculation**
# - **Definition:** A direct, symbolic command with no context.
# - **Example:** "Evaluate $(2^2)^3$."

# **Score 2: Direct Verbal Calculation**
# - **Definition:** A direct command phrased as a simple question.
# - **Example:** "What is 20% of 50?"

# **Score 3: Simple Notation**
# - **Definition:** A problem using basic, standard mathematical notation like f(x).
# - **Example:** "If $f(x) = 3x + 7$, what is $f(2)$?"

# **Score 4: Simple Word Problem**
# - **Definition:** A one-sentence word problem requiring minimal translation.
# - **Example:** "A car travels 100 miles in 2 hours. What is its average speed?"

# **Score 5: Standard Word Problem**
# - **Definition:** Requires translating a multi-sentence scenario into a standard mathematical setup.
# - **Example:** "A rectangle's length is twice its width. If its perimeter is 18, what is its area?"

# **Score 6: Complex Word Problem**
# - **Definition:** A word problem with multiple entities, steps, or irrelevant information that must be filtered out.
# - **Example:** "Jane starts with $500, invests 30% in stocks, and 50% of the remainder in bonds. How much money is not invested?"

# **Score 7: Specialized Notation**
# - **Definition:** The problem involves specialized, subject-specific notation (e.g., calculus, logarithms, summation).
# - **Example:** "Let $f(x) = \sin(x^2)$. Find $f'(x)$."

# **Score 8: Visual Interpretation**
# - **Definition:** Requires understanding and interpreting a visual diagram, geometric figure, or code block (`[asy]`).
# - **Example:** A problem asking for the area of a shaded region in a provided geometric diagram.

# **Score 9: Abstract Concepts**
# - **Definition:** The problem is based on abstract or theoretical concepts (e.g., invertible functions, group theory, proofs).
# - **Example:** "Prove that for any cyclic group G of prime order p, G is isomorphic to Z_p."

# **Score 10: Abstract with Multi-Constraint**
# - **Definition:** Combines abstract concepts with multiple, layered constraints that must be understood together to formulate the problem.
# - **Example:** "Find all functions $f: \mathbb{{R}} \to \mathbb{{R}}$ such that $f(x+y) = f(x)f(y)$ and $f$ is continuous at 0."

# ---
# Based on these strict rules, return a JSON object with a single key "difficulty".
# """


class ComprehensionDifficultyExtractor:
    """
    Extracts a comprehension difficulty score from math problems concurrently
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
            item_id = item.get('id', 'N/A')
            
            # MODIFIED: Use the 'query' field for the problem text
            query_text = item.get('query', '')
            if not query_text:
                print(f"\n⚠️ Warning: Item {item_id} has an empty 'query' field. Skipping.")
                item['comprehension_difficulty'] = -1
                return item

            prompt = COMPREHENSION_DIFFICULTY_PROMPT_TEMPLATE.format(query_text=query_text)
            
            response_str = await self._call_vllm_api(item_id, prompt)
            try:
                result = json.loads(response_str)
                difficulty = result.get('difficulty')
                if difficulty is None:
                    raise KeyError("'difficulty' key not found")
                
                # Add the new key to the original item dictionary
                item['comprehension_difficulty'] = max(1, min(5, int(difficulty))) # Clamp score to 1-5 range
            except Exception as e:
                print(f"\n⚠️ Warning: Failed to parse difficulty for item {item_id}. Error: {e}. Response: '{response_str}'")
                item['comprehension_difficulty'] = -1
            return item

        tasks = [process_item(item) for item in chunk]
        processed_chunk = await async_tqdm.gather(*tasks, desc="⚙️ Processing Chunk")
        return processed_chunk

# MODIFIED: Simplified function to save results to a JSONL file
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
    print("🚀 Starting CHUNKED comprehension difficulty extraction using vLLM...")

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

    # MODIFIED: Overwrite the output file at the start of the run
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)
        print(f"🗑️ Cleared existing output file: {OUTPUT_FILE}")

    extractor = ComprehensionDifficultyExtractor()
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
        item.get('comprehension_difficulty') for item in all_processed_data
        if item.get('comprehension_difficulty') not in (None, -1)
    ]
    total = len(all_processed_data)
    success = len(valid_scores)
    if success:
        print(f"📊 Comprehension Difficulty ({success}/{total} successful):")
        print(f"   - Min: {min(valid_scores)}")
        print(f"   - Max: {max(valid_scores)}")
        print(f"   - Average: {sum(valid_scores)/success:.2f}")
    else:
        print("No comprehension difficulty scores were successfully extracted.")

    total_time = end_time - start_time
    avg_time_per_sample = total_time / len(source_data) if source_data else 0

    print(f"\n🎉 Feature extraction complete!")
    print(f"📁 Results saved to: {OUTPUT_FILE}")
    print(f"⏱️  Total time: {total_time:.2f} seconds")
    print(f"📈 Average time per sample: {avg_time_per_sample:.2f} seconds")

if __name__ == "__main__":
    asyncio.run(main())