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
    text = str(text).strip()
    text = re.sub(r'\\begin{(?:p|b)matrix}', ' ', text)
    text = re.sub(r'\\end{(?:p|b)matrix}', ' ', text)
    text = text.replace('&', ' ').replace('\\\\', ' ')
    text = re.sub(r'\\(text|mathrm|mathbf){[^}]+}', '', text)
    text = re.sub(r'\\sqrt{([^}]+)}', r'sqrt(\1)', text)
    text = re.sub(r'\\(frac|dfrac){([^}]+)}{([^}]+)}', r'\2/\3', text)
    text = re.sub(r'\\pi', 'pi', text)
    text = text.replace('$', '').replace('%', '')
    text = re.sub(r'[\[\]()]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.replace(' ', ',')
    text = re.sub(r'\\', '', text)
    text = re.sub(r'(?<=\d),(?=\d{3})', '', text)
    text = re.sub(r'[,]+', ',', text).strip(',')
    return text.lower()


def extract_final_answer(text: str) -> str | None:
    """
    [V5] Extracts the most likely final answer string using a multi-stage process.
    This version is simplified to just find the *most promising line or block* containing the answer.
    The final cleaning is handled by `_clean_and_validate_answer`.
    """
    if text is None:
        return None

    # 1. Highest priority: explicit delimiters
    for pattern in [r'\\boxed{([^}]*)}', r'####\s*([^\n]*)']:
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()

    # 2. Search from the end for explicit phrases
    lines = text.strip().split('\n')
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        # Phrases like "the final answer is", "the value of k is"
        match = re.search(r'(?:answer|value|result)\s*(?:is|:)\s*(.*)', line, re.IGNORECASE)
        if match:
            return match.group(1).strip()

    # 3. Fallback: return the last non-empty line
    if lines:
        return lines[-1]

    return None


def _clean_and_validate_answer(text: str) -> str | None:
    """
    [NEW HELPER V5] Takes a raw extracted string and pulls out the cleanest possible answer.
    This is the key to ignoring trailing text like "(C)" or "is the answer".
    """
    if text is None:
        return None

    # This pattern is the heart of the new logic. It finds many valid answer formats.
    # We will return the *last* thing in the string that matches this pattern.
    answer_pattern = re.compile(
        r"""
        # LaTeX matrix or cases environments (handles multi-line)
        (?:\\begin{(?:p|b)matrix}.*?\\end{(?:p|b)matrix})|
        # Tuples, coordinates, or lists of numbers
        (?:[\[(]-?\s*\d+(?:\.\d+)?(?:,\s*-?\d+(?:\.\d+)?)+\s*[\])])|
        # LaTeX fractions
        (?:\\(?:frac|dfrac){[^}]+}{[^}]+})|
        # Numbers with decimals and/or commas
        (?:-?\d{1,3}(?:,\d{3})*\.\d+)|
        # Simple fractions
        (?:-?\d+/\d+)|
        # Integers
        (?:-?\d+)
        """,
        re.VERBOSE | re.DOTALL
    )
    
    matches = answer_pattern.findall(text)
    if matches:
        # Return the last valid match found in the string
        return matches[-1].strip().rstrip('.')

    return text # Return original if no specific pattern is found

def is_correct(generated_solution: str, ground_truth_solution: str) -> bool:
    """
    [IMPROVED V5] Compares answers using the new cleaning logic.
    """
    # Step 1: Extract the most promising string from the model output.
    pred_raw = extract_final_answer(generated_solution)
    gt_raw = str(ground_truth_solution)

    if pred_raw is None:
        return False
        
    # Step 2: Clean BOTH the prediction and the ground truth to isolate the answer.
    pred_clean = _clean_and_validate_answer(pred_raw)
    gt_clean = _clean_and_validate_answer(gt_raw)

    if pred_clean is None:
        return False

    # Step 3: First, try a structural comparison for matrices/tuples.
    structural_pattern = r'[\(\[].*?[\)\]]|\\begin{(?:p|b)matrix}.*?\\end{(?:p|b)matrix}'
    is_pred_structural = re.search(structural_pattern, pred_clean, re.DOTALL)
    is_gt_structural = re.search(structural_pattern, gt_clean, re.DOTALL)
    
    if is_pred_structural and is_gt_structural:
        pred_nums = re.findall(r'-?\d+\.?\d*', pred_clean)
        gt_nums = re.findall(r'-?\d+\.?\d*', gt_clean)
        if pred_nums and gt_nums and pred_nums == gt_nums:
            return True

    # Step 4: Fallback to general normalization and comparison.
    pred_norm = normalize_answer(pred_clean)
    gt_norm = normalize_answer(gt_clean)

    if not pred_norm or not gt_norm: return False
    if pred_norm == gt_norm: return True

    # Step 5: Final check for numerical proximity.
    try:
        if ',' in pred_norm or re.search(r'[a-z]', pred_norm): return False
            
        def to_float(s: str) -> float:
            if '/' in s:
                num, den = s.split('/'); return float(num) / float(den)
            return float(s)

        pred_val, gt_val = to_float(pred_norm), to_float(gt_norm)

        if math.isclose(pred_val, gt_val, rel_tol=1e-4, abs_tol=1e-4): return True
        if abs(gt_val - round(gt_val)) < 1e-5 and int(round(pred_val)) == int(gt_val): return True
    except (ValueError, TypeError, ZeroDivisionError, OverflowError):
        pass

    return False

def format_prompt(question: str) -> str:
    """
    Formats the prompt for the model.
    """
    prompt = f"Question: {question}\nLet's solve this question."
    # prompt = f"Question: {question}\nLet's think step by step."
    return prompt

def collate_fn(batch):
    """
    Custom collate function. Your dataset's columns ('question', 'answer') match this perfectly.
    """
    return {
        'question': [item['question'] for item in batch],
        'answer': [item['answer'] for item in batch],
    }

# ======================================================================================
# 主函数 (Main Function)
# ======================================================================================

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
                eos_token_id=tokenizer.eos_token_id,
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
    parser.add_argument("--base_model_path", type=str, default="google/gemma-3-4b-pt", help="Path or Hub ID of the base model.")
    parser.add_argument("--lora_adapter_path", type=str, default=None, help="Optional path to the trained LoRA adapter.")
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="Maximum number of new tokens for the model to generate.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference.")
    parser.add_argument("--limit", type=int, default=-1, help="Limit the number of test samples for a quick run. -1 means use all.")
    parser.add_argument("--print_first_n", type=int, default=0, help="Prints the detailed solution for the first N problems.")
    
    args = parser.parse_args()
    main(args)