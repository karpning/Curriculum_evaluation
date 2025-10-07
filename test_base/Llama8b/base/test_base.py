import argparse
import json
import re
import os
import math
import csv
import random
from pathlib import Path
from typing import List, Dict, Any

import torch
import numpy as np
# [MODIFIED] Added 'load_dataset' to be more explicit
from datasets import load_dataset, Dataset 
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader
import re
import math
from typing import Union, List

def set_seed(seed: int = 42):
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in cuDNN (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def normalize_answer(text: str) -> str:
    if text is None:
        return ""
    s = str(text)

    s = re.sub(r'\\(text|mathrm|mathbf|left|right|cdot|times)', '', s)
    s = re.sub(r'\\(frac|dfrac){([^}]+)}{([^}]+)}', r'(\2)/(\3)', s)
    s = re.sub(r'\\sqrt{([^}]+)}', r'sqrt(\1)', s)
    s = s.replace('$','').replace('%','')

    s = re.sub(r'\\begin{(?:p|b)matrix}.*?\\end{(?:p|b)matrix}', ' ', s, flags=re.DOTALL)
    s = s.replace('&', ' ').replace('\\\\', ' ')

    s = re.sub(r'(?<=\d),(?=\d{3}\b)', '', s)

    s = re.sub(r'[^0-9\-\./()]', '', s)
    s = re.sub(r'\s+', '', s)
    return s.strip().lower()

_NUM_0_19 = {
    "zero":0,"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,"eight":8,"nine":9,
    "ten":10,"eleven":11,"twelve":12,"thirteen":13,"fourteen":14,"fifteen":15,"sixteen":16,
    "seventeen":17,"eighteen":18,"nineteen":19
}
_TENS = {"twenty":20,"thirty":30,"forty":40,"fifty":50,"sixty":60,"seventy":70,"eighty":80,"ninety":90}
_WORD_RE = r"(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)"

def _words_to_int(phrase: str) -> int | None:
    tokens = re.sub(r"[^a-z\- ]", " ", phrase.lower()).replace("-", " ").split()
    val, used = 0, False
    for w in tokens[::-1]:  
        if w in _NUM_0_19:
            val += _NUM_0_19[w]; used = True
        elif w in _TENS:
            val += _TENS[w]; used = True
        else:
            if used: break
            val = 0
    return val if used else None

def extract_final_answer(text: str) -> str | None:
    """
    Enhanced final answer extractor:
    1) Prioritize explicit wrappers (\\boxed{} / ####).
    2) Scan text for candidates (fractions, decimals, integers).
       Score them based on:
         - Being in conclusion context (keywords, "=", final line).
         - Appearing near the end of the text.
         - Being the only content on the line.
         - Penalty if inside a question line.
    3) Return the highest scoring candidate (prefer later ones).
    """
    if text is None:
        return None
    t = str(text)

    # 1) Explicit delimiters take priority
    for pat in (r'\\boxed{([^}]*)}', r'####\s*([^\n]*)'):
        ms = re.findall(pat, t)
        if ms:
            return ms[-1].strip()

    # 2) Scan line by line
    lines = [ln.strip() for ln in t.strip().splitlines() if ln.strip()]
    if not lines:
        return None
    n = len(lines)

    num_re = re.compile(r'(?:-?\d+/\d+|-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+(?:\.\d+)?)')
    kw_re = re.compile(r'(?i)\b(answer|final|result|thus|therefore|so|hence|equals?)\b')

    candidates = []
    for li, ln in enumerate(lines):
        is_question = '?' in ln

        # Features for scoring
        near_end_bonus = 3 if li >= n - 2 else (2 if li >= n - 5 else (1 if li >= n - 10 else 0))
        kw_bonus = 3 if kw_re.search(ln) else 0
        solo_number_bonus = 2 if num_re.fullmatch(ln) else 0  # line contains only the number

        for m in num_re.finditer(ln):
            val = m.group(0).strip().rstrip('.')

            # Check if preceded by '='
            eq_bonus = 0
            left = ln[max(0, m.start() - 8):m.start()]
            if re.search(r'=\s*$', left):
                eq_bonus = 3

            score = near_end_bonus + kw_bonus + eq_bonus + solo_number_bonus
            if is_question:
                score -= 2  # penalize numbers inside question lines

            candidates.append((score, li, m.start(), val))

    if candidates:
        # Sort by (score, line idx, position)
        candidates.sort(key=lambda x: (x[0], x[1], x[2]))
        return candidates[-1][3]

    return None


def _clean_and_validate_answer(text: str) -> str | None:
    """
    Clean and validate extracted fragment:
    - Preserve structures (matrix/tuple).
    - Normalize LaTeX fractions into "a/b".
    - Pick the last valid numeric candidate.
    """
    if text is None:
        return None
    s = str(text).strip()

    # Structural answers (matrices, tuples, coordinates)
    struct_pat = re.compile(
        r'(\\begin{(?:p|b)matrix}.*?\\end{(?:p|b)matrix})|'
        r'([\[(]-?\s*\d+(?:\.\d+)?(?:,\s*-?\d+(?:\.\d+)?)+\s*[\])])',
        re.DOTALL
    )
    sm = struct_pat.search(s)
    if sm:
        return sm.group(0).strip().rstrip('.')

    # Normalize LaTeX fractions and sqrt
    s = re.sub(r'\\(?:frac|dfrac){([^}]+)}{([^}]+)}', r'\1/\2', s)
    s = re.sub(r'\\sqrt{([^}]+)}', r'sqrt(\1)', s)
    s = s.replace('$', '')

    num_re = re.compile(r'(?:-?\d+/\d+|-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+(?:\.\d+)?)')
    ms = num_re.findall(s)
    if ms:
        return ms[-1].strip().rstrip('.')

    return s if s else None

def is_correct(generated_solution: str, ground_truth_solution: str) -> bool:
    """
    Robust comparison between predicted and ground truth answers.
    Steps:
      1) Extract final answers from both prediction and ground truth.
      2) Clean and normalize the fragments.
      3) Try exact string match first (strict, safe).
      4) Try numeric match with tolerance (covers int/float/fractions).
      5) Try structural match for tuples/matrices.
      6) As a last fallback, compare sets of numbers extracted.
    """
    # --- Step 1: Extract raw answers ---
    pred_raw = extract_final_answer(generated_solution)
    gt_raw = extract_final_answer(ground_truth_solution)

    if pred_raw is None or gt_raw is None:
        return False

    # --- Step 2: Clean and normalize ---
    pred_clean = _clean_and_validate_answer(pred_raw)
    gt_clean = _clean_and_validate_answer(gt_raw)

    if pred_clean is None or gt_clean is None:
        return False

    # --- Step 3: Exact normalized string match ---
    if pred_clean.strip().lower() == gt_clean.strip().lower():
        return True

    # --- Step 4: Numeric comparison (floats & fractions) ---
    def to_float(s: str) -> float | None:
        s = s.replace(',', '')
        try:
            if '/' in s and not re.search(r'[a-zA-Z]', s):
                num, den = s.split('/')
                return float(num) / float(den)
            return float(s)
        except Exception:
            return None

    pred_val, gt_val = to_float(pred_clean), to_float(gt_clean)
    if pred_val is not None and gt_val is not None:
        if math.isclose(pred_val, gt_val, rel_tol=1e-4, abs_tol=1e-4):
            return True
        # Extra: integer rounding match (e.g., 23.0 ≈ 23)
        if abs(gt_val - round(gt_val)) < 1e-6 and int(round(pred_val)) == int(round(gt_val)):
            return True

    # --- Step 5: Structural match (tuples, matrices) ---
    struct_pat = re.compile(r'[\(\[].*?[\)\]]|\\begin{(?:p|b)matrix}.*?\\end{(?:p|b)matrix}', re.DOTALL)
    if struct_pat.search(pred_clean) and struct_pat.search(gt_clean):
        pred_nums = re.findall(r'-?\d+\.?\d*', pred_clean)
        gt_nums = re.findall(r'-?\d+\.?\d*', gt_clean)
        if pred_nums and gt_nums and pred_nums == gt_nums:
            return True

    # --- Step 6: Fallback — compare sets of extracted numbers ---
    pred_nums = re.findall(r'-?\d+\.?\d*', pred_clean)
    gt_nums = re.findall(r'-?\d+\.?\d*', gt_clean)
    if pred_nums and gt_nums and sorted(pred_nums) == sorted(gt_nums):
        return True

    return False



def collate_fn(batch):
    """
    Custom collate function. Your dataset's columns ('question', 'answer') match this perfectly.
    """
    return {
        'question': [item['question'] for item in batch],
        'answer': [item['answer'] for item in batch],
    }

def format_prompt(question: str) -> str:
    """
    Formats the prompt for the model.
    """
    prompt = f"Question: {question}\nLet's solve this question."
    # prompt = f"Question: {question}\nLet's think step by step."
    return prompt

def main(args):
    
    set_seed(42)
    
    print(f"🚀 Starting evaluation on local dataset: {args.dataset_path}...")
    
    # --- 1. Load Tokenizer and Model ---
    # (This section remains unchanged)
    print(f"--- Loading tokenizer from '{args.base_model_path}'... ---")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    print(f"--- Loading base model '{args.base_model_path}' with device_map='auto'... ---")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa"
    )

    if args.lora_adapter_path:
        print(f"--- Applying LoRA adapter from '{args.lora_adapter_path}'... ---")
        model = PeftModel.from_pretrained(base_model, args.lora_adapter_path)
        model_name = os.path.basename(os.path.normpath(args.lora_adapter_path))
        print("Evaluating LoRA-finetuned model.")
    else:
        print("--- No LoRA adapter path provided. Evaluating the base model. ---")
        model = base_model
        model_name = os.path.basename(os.path.normpath(args.base_model_path))

    model.eval()
    print("✅ Model loaded and configured for evaluation.")

    # --- 2. Load and Prepare Dataset ---
    print(f"\n--- Loading local dataset from '{args.dataset_path}'... ---")
    # [MODIFIED] Changed dataset loading to use the local file path
    if not os.path.exists(args.dataset_path):
        raise FileNotFoundError(f"Dataset file not found at: {args.dataset_path}")
    
    # Use the 'json' loader for .jsonl files. The default split name is 'train'.
    dataset = load_dataset('json', data_files=args.dataset_path, split='train')
    
    dataset = dataset.shuffle(seed=42) 
    if args.limit > 0:
        dataset = dataset.select(range(args.limit))

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    print(f"✅ Testing on {len(dataset)} samples with batch size {args.batch_size}")

    # --- 3. Run Inference and Evaluation Loop ---
    # (This section remains unchanged as the logic is the same)
    all_results = []
    processed_count = 0
    print("\n--- Starting inference loop... ---")
    for batch in tqdm(dataloader, desc="Evaluating Batches"):
        prompts = [format_prompt(p) for p in batch['question']]
        
        # Determine max input length allowed by the model
        context_length = getattr(model.config, "max_position_embeddings", 8192)
        max_input_len = context_length - args.max_new_tokens
        if max_input_len <= 0:
            raise ValueError(
                f"max_new_tokens ({args.max_new_tokens}) is too large for the model's context window ({context_length})."
            )

        # Tokenize inputs
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_len
        )
        inputs = {key: tensor.to(model.device) for key, tensor in inputs.items()}

        # Generate outputs
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
            )

        # Decode model outputs
        generated_texts = tokenizer.batch_decode(
            outputs[:, inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

        # Process each sample in the batch
        for i, generated_text in enumerate(generated_texts):
            
            ## choice 1
            # input_length = inputs['input_ids'].shape[1]

            # # Slice out only the generated tokens
            # generated_part = outputs[i, input_length:]

            # # Remove padding tokens
            # non_padding_tokens = generated_part[generated_part != tokenizer.pad_token_id]

            # # Count non-padding tokens
            # output_token_length = len(non_padding_tokens)

            # # Exclude EOS if the last non-padding token is EOS
            # if output_token_length > 0 and non_padding_tokens[-1].item() == tokenizer.eos_token_id:
            #     output_token_length -= 1
            
            ## choice 2
            generated_tokens = tokenizer.encode(
                generated_text.strip(),
                add_special_tokens=False
            ) 
            output_token_length = len(generated_tokens)
                
            # Compare correctness
            correct = is_correct(generated_text, batch['answer'][i])

            # Save result
            result = {
                'question': batch['question'][i],
                'ground_truth': str(batch['answer'][i]),
                'generated_solution': generated_text.strip(),
                'is_correct': correct,
                'output_token_length': output_token_length
            }
            all_results.append(result)

            # Print detailed results for the first N samples
            should_print = (args.print_first_n == -1) or (processed_count < args.print_first_n)
            if should_print:
                print("\n" + "="*80)
                print(f"SAMPLE #{processed_count + 1}")
                print(f"QUESTION: {result['question']}")
                print("-" * 40)
                print(f"MODEL SOLUTION:\n{result['generated_solution']}")
                print("-" * 40)
                print(f"GROUND TRUTH: {result['ground_truth']}")
                print(f"CORRECT: {result['is_correct']}")
                print(f"OUTPUT TOKEN LENGTH: {result['output_token_length']}")
                print("="*80)
            
            processed_count += 1

    # --- 4. Calculate Final Metrics and Save ---
    # (This section remains unchanged)
    correct_count = sum(1 for r in all_results if r['is_correct'])
    total_samples = len(all_results)
    accuracy = (correct_count / total_samples) * 100 if total_samples > 0 else 0
    total_output_tokens = sum(r['output_token_length'] for r in all_results)
    avg_output_token_length = total_output_tokens / total_samples if total_samples > 0 else 0

    print(f"\n--- ✅ Evaluation Complete ---")
    print(f"Total samples evaluated: {total_samples}")
    print(f"Correct answers: {correct_count}")
    print(f"Final Accuracy: {accuracy:.2f}%")
    print(f"Average Output Token Length: {avg_output_token_length:.2f}")

    # --- Save results (independent of LoRA adapter) ---
    dataset_name_noext = os.path.splitext(os.path.basename(args.dataset_path))[0]

    # Save accuracy summary (CSV)
    results_dir = os.path.join("acc_results")
    os.makedirs(results_dir, exist_ok=True)
    summary_file = os.path.join(results_dir, f"{dataset_name_noext}.csv")

    file_exists = os.path.isfile(summary_file)
    with open(summary_file, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['accuracy', 'avg_output_token_length'])
        writer.writerow([f"{accuracy:.2f}", f"{avg_output_token_length:.2f}"])
    print(f"📊 Accuracy metrics saved to {summary_file}")

    # Save detailed answer results (JSONL)
    answer_dir = os.path.join("answer_results")
    os.makedirs(answer_dir, exist_ok=True)
    answer_file = os.path.join(answer_dir, f"{dataset_name_noext}.jsonl")

    with open(answer_file, 'w', encoding='utf-8') as f:
        for item in all_results:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"📂 Answer results saved to {answer_file}")




if __name__ == "__main__":
    # [MODIFIED] Updated description and added the new dataset_path argument
    parser = argparse.ArgumentParser(description="Evaluation script for base or LoRA-tuned models on a specified math dataset.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the local dataset file (in .jsonl format).")
    parser.add_argument("--base_model_path", type=str, default="meta-llama/Meta-Llama-3.1-8B", help="Path or Hub ID of the base model.")
    parser.add_argument("--lora_adapter_path", type=str, default=None, help="Optional path to the trained LoRA adapter.")
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="Maximum number of new tokens for the model to generate.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference.")
    parser.add_argument("--limit", type=int, default=-1, help="Limit the number of test samples for a quick run. -1 means use all.")
    parser.add_argument("--print_first_n", type=int, default=0, help="Prints the detailed solution for the first N problems.")
    
    args = parser.parse_args()
    main(args)