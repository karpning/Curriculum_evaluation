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
    """
    [FIXED & COMPLETE] Normalizes math answers for robust comparison.
    - Keeps all original functionality
    - Fixes bug where \sqrt inside \frac was lost
    """
    if text is None:
        return ""
    text = str(text).strip()

    # remove commas (both half-width , and full-width ，)
    text = re.sub(r'(?<=\d)[,，](?=\d)', '', text)

    # remove LaTeX formatting wrappers
    text = re.sub(r'\\(text|mathrm|mathbf){[^}]+}', '', text)

    # remove LaTeX operators/symbols
    text = re.sub(r'\\(circ|degree|cdot|times|!)', '', text)
    text = re.sub(r'\^\\circ', '', text)  # e.g., 87^\circ → 87

    # --- sqrt first (fixes \frac{\sqrt{3}}{2} issue) ---
    text = re.sub(r'\\sqrt{([^}]+)}', r'sqrt(\1)', text)
    text = re.sub(r'sqrt{([^}]+)}', r'sqrt(\1)', text)

    # --- fractions ---
    text = re.sub(r'\\(frac|dfrac){-([^}]+)}{([^}]+)}', r'-\2/\3', text)   # \frac{-a}{b}
    text = re.sub(r'-\\(frac|dfrac){([^}]+)}{([^}]+)}', r'-\2/\3', text)   # -\frac{a}{b}
    text = re.sub(r'\\(frac|dfrac){([^}]+)}{([^}]+)}', r'\2/\3', text)     # \frac{a}{b}

    # pi
    text = re.sub(r'\\pi', 'pi', text)

    # cleanup braces, $, parens, slashes
    text = re.sub(r'[{}$()]', '', text)
    text = re.sub(r'\\', '', text)

    # remove extra whitespace
    text = re.sub(r'\s+', '', text)

    # strip trailing units or words (e.g., "108m" -> "108", "25 years" -> "25")
    text = re.sub(r'[a-zA-Z]+$', '', text)

    return text.lower()


def extract_final_answer(text: str) -> str | None:
    """
    [MODIFIED] Extract the final answer from model output with improved regex.
    Priority: \boxed{} > #### > 'the answer is' > last non-empty line.
    """
    # 1. Boxed
    boxed_matches = re.findall(r'\\boxed{([^}]*)}', text)
    if boxed_matches:
        return boxed_matches[-1].strip()

    # 2. ####
    hash_matches = re.findall(r'####\s*([^\n]*)', text)
    if hash_matches:
        return hash_matches[-1].strip()

    # 3. Common answer phrases with a flexible regex to capture non-numeric answers
    answer_phrases = [
        r'(?:the\s+final\s+answer\s+is|the\s+answer\s+is|so\s+the\s+answer\s+is)[:\s]*([^\n]+)',
        r'(?:answer\s*:)[:\s]*([^\n]+)'
    ]
    for phrase in answer_phrases:
        phrase_matches = re.findall(phrase, text, re.IGNORECASE)
        if phrase_matches:
            return phrase_matches[-1].strip()

    # 4. Fallback: last non-empty line of the output
    lines = text.strip().splitlines()
    if lines:
        last_line = lines[-1].strip()
        # Avoid returning a long sentence as the answer
        if len(last_line.split()) < 10:
            return last_line

    return None

def is_correct(generated_solution: str, ground_truth_solution: str) -> bool:
    """
    Compares the generated solution with the ground truth.
    """
    # The ground truth is a simple number, so we only need to extract from the model's output
    pred_answer_str = extract_final_answer(generated_solution)
    gt_answer_str = str(ground_truth_solution) # Ensure GT is a string

    if pred_answer_str is None or gt_answer_str is None:
        return False

    pred_norm = normalize_answer(pred_answer_str)
    gt_norm = normalize_answer(gt_answer_str)

    if pred_norm == gt_norm:
        return True

    try:
        def to_float(s):
            s = s.replace(',', '')
            if '/' in s:
                parts = s.split('/')
                if len(parts) == 2 and parts[1]:
                    return float(parts[0]) / float(parts[1])
            return float(s)

        pred_val = to_float(pred_norm)
        gt_val = to_float(gt_norm)
        
        if math.isclose(pred_val, gt_val, rel_tol=1e-12):
            return True
    except (ValueError, ZeroDivisionError, TypeError):
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
    # if tokenizer.pad_token is None:
    #     tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token is None:   ## align with training setting
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.padding_side = 'left'

    print(f"--- Loading base model '{args.base_model_path}' with device_map='auto'... ---")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa"
    )
    
    base_model.resize_token_embeddings(len(tokenizer))
    base_model.config.vocab_size = len(tokenizer)
    base_model.config.pad_token_id = tokenizer.pad_token_id
     
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
                # pad_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
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

    if args.lora_adapter_path:
        adapter_full = os.path.normpath(args.lora_adapter_path)
        adapter_last = os.path.basename(adapter_full)  # last dir name, e.g. Gemma4B_entropy_mean_token_entropy_desc

        # Extract path after "model_save" and before the last dir
        parts = adapter_full.split(os.sep)
        if "model_save" in parts:
            idx = parts.index("model_save")
            rel_parts = parts[idx+1:-1]  # e.g. ["SortByMetric", "Gemma4B", "mean"]
            rel_path = os.path.join(*rel_parts)
        else:
            rel_path = "default"

        # Dataset name (without extension)
        dataset_name_noext = os.path.splitext(os.path.basename(args.dataset_path))[0]

        # --- Save accuracy summary (CSV) ---
        results_dir = os.path.join("acc_results", rel_path, dataset_name_noext)
        os.makedirs(results_dir, exist_ok=True)
        summary_file = os.path.join(results_dir, f"{dataset_name_noext}.csv")

        file_exists = os.path.isfile(summary_file)
        with open(summary_file, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['model_name', 'accuracy', 'avg_output_token_length'])
            writer.writerow([adapter_last, f"{accuracy:.2f}", f"{avg_output_token_length:.2f}"])
        print(f"📊 Accuracy metrics saved to {summary_file}")

        # --- Save detailed answer results (JSONL) ---
        answer_dir = os.path.join("answer_results", rel_path, dataset_name_noext)
        os.makedirs(answer_dir, exist_ok=True)
        answer_file = os.path.join(answer_dir, f"{adapter_last}.jsonl")

        with open(answer_file, 'w', encoding='utf-8') as f:
            for item in all_results:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"📂 Answer results saved to {answer_file}")
    else:
        print("⚠️ No lora_adapter_path provided, skipping results save.")



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