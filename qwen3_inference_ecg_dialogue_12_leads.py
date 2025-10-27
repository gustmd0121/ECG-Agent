#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
End-to-end inference for Qwen3 + LoRA (unmerged) on ECG dialogue evaluation.

- Loads base Qwen3 model
- Attaches LoRA adapters (no merge required)
- Applies Unsloth Qwen3 chat template
- Runs turn-by-turn inference over the test set
- Uses precomputed tool outputs (CSV) when tools are "called"
- SUPPORTS both 'with_gt' and 'without_gt' inference modes.
- MODIFIED to allow filtering for dialogues containing a specific action.
"""

import os
import sys
import re
import json
import argparse
from datetime import datetime

import torch
import pandas as pd

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
)
from peft import PeftModel
from tqdm import tqdm

# ---- Unsloth chat template for Qwen3 ----
from unsloth.chat_templates import get_chat_template
from prompts import ECG_EVALUATION_PROMPT


def load_ground_truth_data():
    """Loads all ground-truth tool data from pre-computed CSV files."""
    print("Loading ground-truth data from CSV files...")
    try:
        measurement_df = pd.read_csv(MEASUREMENT_SUMMARY_PATH)
        classification_df = pd.read_csv(CLASSIFICATION_SUMMARY_PATH)
        print("✅ Successfully loaded all ground-truth summary CSVs.")
        return {
            "measurement": measurement_df,
            "classification": classification_df
        }
    except FileNotFoundError as e:
        print(f"🛑 Error: Could not find a CSV file. Please check your paths.\nDetails: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"🛑 An error occurred while loading CSV files: {e}")
        sys.exit(1)


def get_precomputed_tool_output(action, ecg_filename, gt_data):
    """Retrieves pre-computed tool output from loaded dataframes."""
    if not ecg_filename:
        return "[Error: ECG filename not provided]"
    try:
        if action == "call_classification_tool":
            df = gt_data["classification"]
            record = df[df['ecg_file_path'] == ecg_filename]
            if record.empty: return "[]"
            top_classes_val = record['top_classes'].iloc[0]
            return str([c.strip() for c in top_classes_val.split(',')]) if pd.notna(top_classes_val) else "[]"

        elif action == "call_measurement_tool":
            df = gt_data["measurement"]
            record = df[df['ecg_file_path'] == ecg_filename]
            if record.empty: return "{}"
            rec = record.iloc[0]
            measurements = {
                "heart_rate": f"{rec.get('Heart_Rate'):.2f}" if pd.notna(rec.get('Heart_Rate')) else None,
                "pr_interval": f"{rec.get('PR_Interval_ms'):.0f}" if pd.notna(rec.get('PR_Interval_ms')) else None,
                "qrs_duration": f"{rec.get('QRS_Duration_ms'):.0f}" if pd.notna(rec.get('QRS_Duration_ms')) else None,
                "qtc_interval": f"{rec.get('QTc_ms'):.2f}" if pd.notna(rec.get('QTc_ms')) else None,
            }
            return json.dumps(measurements)
        
    except Exception:
        return f"[Error: Failed to retrieve data for {ecg_filename}]"
    return "[Error: Unknown tool action]"


# ------------------------------------
# Qwen3 + LoRA (UNMERGED) initialiser
# ------------------------------------
def load_model_and_tokenizer(base_model_path, lora_path, max_seq_len=4096):
    """
    Loads base Qwen3 model, attaches LoRA adapters (unmerged), and applies Qwen3 chat template.
    """
    print(f"Loading base model from: {base_model_path}")
    print(f"Attaching LoRA adapters from: {lora_path}")

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=True)
    # Ensure Qwen3 chat template is applied
    tokenizer = get_chat_template(tokenizer, chat_template="qwen-3")

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=dtype,
        device_map="auto",
    )
    # Attach LoRA adapters
    print("Attaching LoRA adapters...")
    model = PeftModel.from_pretrained(model, lora_path)
    model.eval()
    print("✅ Model and tokenizer loaded successfully.")

    return model, tokenizer


# -------------------------
# Prompting / Generation
# -------------------------
def generate_full_response(model, tokenizer, messages, generation_config):
    """
    Generates a response given chat 'messages' using the tokenizer's chat template.
    """
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(**inputs, generation_config=generation_config)

    # Decode only newly generated tokens
    new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
    decoded = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return decoded.strip()


# -------------------------
# Parsing & Formatting
# -------------------------
def parse_generated_response(generated_text):
    """
    Parses the full text output from the model to extract action, thought, and content.
    """
    text = generated_text.strip()
    action, thought, content = '', '', ''

    action_pattern = r"(?im)^\s*\[?\s*Action\s*[:\-]\s*([^\n\r\]]+)"
    thought_pattern = r"^\s*\[?Thought:\s*(.*?)(?=\n\s*\[?(?:Action|Content|Tool_Output):|$)"
    content_pattern = r"^\s*\[?Content:\s*(.*)"
    tool_output_pattern = r"^\s*\[?Tool_Output:\s*(.*)"

    single_line_flags = re.IGNORECASE | re.MULTILINE
    multi_line_flags = re.IGNORECASE | re.MULTILINE | re.DOTALL

    action_match = re.search(action_pattern, text, single_line_flags)
    thought_match = re.search(thought_pattern, text, multi_line_flags)
    content_match = re.search(content_pattern, text, multi_line_flags)
    tool_output_match = re.search(tool_output_pattern, text, multi_line_flags)

    if action_match:
        action = action_match.group(1).strip()
    
    if thought_match:
        thought = thought_match.group(1).strip()

    if content_match:
        content = content_match.group(1).strip()
    elif not (action_match or thought_match or tool_output_match):
        content = text
        
    return {
        'role': 'assistant',
        'action': action,
        'thought': thought,
        'content': content
    }


def format_assistant_turn_for_messages(turn):
    """
    Formats an assistant turn into the string content for the message list,
    using the non-bracket tags from the training prompt.
    """
    parts = []
    if 'action' in turn and turn['action']: parts.append(f"Action: {turn['action']}")
    if 'thought' in turn and turn['thought']: parts.append(f"Thought: {turn['thought']}")
    if 'tool_output' in turn and turn['tool_output']: parts.append(f"Tool_Output: {turn['tool_output']}")
    elif 'content' in turn and turn['content']: parts.append(f"Content: {turn.get('content', '')}")
    return "\n".join(parts).strip()


# ---------- Alignment Helper ----------
def build_aligned_turns(gt_dialogue, gen_dialogue):
    """Aligns ground-truth and generated assistant turns for comparison."""
    gt_assistant_turns = [t for t in gt_dialogue if t.get("role") == "assistant"]
    gen_assistant_turns = [t for t in gen_dialogue if t.get("role") == "assistant"]

    user_contexts = [t.get("content", "") for t in gt_dialogue if t.get("role") == "user"]
    
    n_max = max(len(gt_assistant_turns), len(gen_assistant_turns))
    aligned = []
    for i in range(n_max):
        gt_t = gt_assistant_turns[i] if i < len(gt_assistant_turns) else {}
        gen_t = gen_assistant_turns[i] if i < len(gen_assistant_turns) else {}
        
        gt_action = gt_t.get("action")
        gen_action = gen_t.get("action")
        gt_content = gt_t.get("content")
        gen_content = gen_t.get("content")

        aligned.append({
            "turn_index": i,
            "user_content": user_contexts[i] if i < len(user_contexts) else None,
            "gt_action": gt_action,
            "gt_thought": gt_t.get("thought"),
            "gt_content": gt_content,
            "gt_tool_output": gt_t.get("tool_output"),
            "gen_action": gen_action,
            "gen_thought": gen_t.get("thought"),
            "gen_content": gen_content,
            "gen_tool_output": gen_t.get("tool_output"),
            "action_correct": str(gt_action).strip() == str(gen_action).strip(),
            "response_exact_match": str(gt_content).strip() == str(gen_content).strip(),
        })

    return aligned, len(gt_assistant_turns), len(gen_assistant_turns)


# -------------------------
# Main Inference Loop
# -------------------------
def run_inference_on_test_set(base_model_path, lora_path, output_file=None, max_samples=None, inference_mode='with_gt', filter_action=None):
    model, tokenizer = load_model_and_tokenizer(base_model_path, lora_path)
    gt_data = load_ground_truth_data()

    # Generation config for deterministic output
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    eos_ids = [t for t in [tokenizer.eos_token_id, eot_id] if t is not None]
    
    generation_config = GenerationConfig(
        max_new_tokens=512,
        temperature=0.6,
        top_p=0.95,
        do_sample=False,
        eos_token_id=eos_ids,
        pad_token_id=tokenizer.pad_token_id,
    )

    dataset = load_dataset("gustmd0121/12-lead-ecg-mtd-dataset")['test']
    
    # <<< NEW: Filtering logic added here >>>
    if filter_action:
        print(f"Filtering dataset to only include samples with the action: '{filter_action}'")
        
        def contains_action(example):
            try:
                dialogue = json.loads(example['dialogue'])
                for turn in dialogue:
                    if turn.get('action') == filter_action:
                        return True
                return False
            except (json.JSONDecodeError, TypeError):
                return False

        original_size = len(dataset)
        dataset = dataset.filter(contains_action, num_proc=4)
        print(f"✅ Filtering complete. Found {len(dataset)} matching samples out of {original_size}.")

    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    print(f"Running inference on {len(dataset)} samples in '{inference_mode}' mode...")

    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_tag = os.path.basename(lora_path.rstrip("/"))
        output_file = f"inference_{base_tag}_{timestamp}_{inference_mode}.jsonl"

    with open(output_file, 'w', encoding='utf-8') as f_out:
        for i, example in enumerate(tqdm(dataset, desc="Generating Dialogues")):
            try:
                gt_dialogue = json.loads(example['dialogue'])
                ecg_files_str = example.get('ecg_files')
                file_list = json.loads(ecg_files_str) if ecg_files_str and ecg_files_str != '[]' else []
                ecg_filename = file_list[0] if file_list else None

                generated_dialogue = []
                messages = [{"role": "system", "content": ECG_EVALUATION_PROMPT}]
                
                gt_dialogue_idx = 0
                while gt_dialogue_idx < len(gt_dialogue):
                    turn = gt_dialogue[gt_dialogue_idx]
                    
                    if turn['role'] == 'user':
                        messages.append(turn)
                        generated_dialogue.append(turn)

                        model_output_str = generate_full_response(model, tokenizer, messages, generation_config)
                        parsed_turn = parse_generated_response(model_output_str)
                        
                        model_generated_turns = []

                        if parsed_turn['action'] in ["call_classification_tool", "call_measurement_tool"]:
                            tool_call_turn = {
                                "role": "assistant",
                                "action": parsed_turn['action'],
                                "thought": parsed_turn['thought'],
                                "tool_output": get_precomputed_tool_output(parsed_turn['action'], ecg_filename, gt_data)
                            }
                            model_generated_turns.append(tool_call_turn)
                            
                            temp_messages = messages + [{"role": "assistant", "content": format_assistant_turn_for_messages(tool_call_turn)}]
                            
                            final_content_str = generate_full_response(model, tokenizer, temp_messages, generation_config)
                            parsed_final_turn = parse_generated_response(final_content_str)

                            response_turn = {
                                "role": "assistant",
                                "action": parsed_final_turn.get("action", "response"),
                                "thought": parsed_final_turn.get("thought", ""),
                                "content": parsed_final_turn.get("content", "")
                            }
                            model_generated_turns.append(response_turn)
                        else:
                            direct_response_turn = parsed_turn
                            model_generated_turns.append(direct_response_turn)
                        
                        generated_dialogue.extend(model_generated_turns)

                        if inference_mode == 'without_gt':
                            for gen_turn in model_generated_turns:
                                messages.append({"role": "assistant", "content": format_assistant_turn_for_messages(gen_turn)})
                        
                        elif inference_mode == 'with_gt':
                            temp_idx = gt_dialogue_idx + 1
                            while temp_idx < len(gt_dialogue) and gt_dialogue[temp_idx]['role'] == 'assistant':
                                gt_assistant_turn = gt_dialogue[temp_idx]
                                messages.append({"role": "assistant", "content": format_assistant_turn_for_messages(gt_assistant_turn)})
                                temp_idx += 1
                    
                    gt_dialogue_idx += 1


                aligned_turns, n_gt_asst, n_gen_asst = build_aligned_turns(gt_dialogue, generated_dialogue)

                record = {
                    "sample_id": i,
                    "ecg_file": ecg_filename,
                    "source_category": example.get("source_category"),
                    "turns": aligned_turns,
                    "generated_dialogue": generated_dialogue,
                    "ground_truth_dialogue": gt_dialogue,
                    "summary": {
                        "num_gt_assistant_turns": n_gt_asst,
                        "num_generated_assistant_turns": n_gen_asst,
                    }
                }
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                f_out.flush()
                os.fsync(f_out.fileno())


            except Exception as e:
                import traceback
                print(f"🛑 Error processing sample {i} (ECG: {ecg_filename}): {e}\n{traceback.format_exc()}")

    print(f"✅ Inference complete. Results saved to {output_file}")


# ----------------------------
# CLI Interface
# ----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with Qwen3 + LoRA adapters (unmerged).")
    parser.add_argument(
        "--base-model-path",
        type=str,
        default="unsloth/Qwen3-32B-unsloth-bnb-4bit",
        help="Base Qwen3 model (Hugging Face ID or local path)."
    )
    parser.add_argument(
        "--lora-path",
        type=str,
        default="/home/hschung/unsloth/ecg-dialogue-finetuned-turn-by-turn-32b-Instruct-0817-4bit",
        help="Path to the LoRA adapter folder saved by trainer.save_model()."
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="/home/hschung/unsloth/ecg_dialogue_icassp/results_qwen3_32b_4bit_0909_with_gt_response_fail_final.jsonl",
        help="Output .jsonl path. If not provided, a timestamped name will be generated."
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit the number of test samples to process for a quick test."
    )
    parser.add_argument(
        "--inference-mode",
        type=str,
        choices=["without_gt", "with_gt"],
        default="with_gt",
        help="Use model-generated history ('without_gt') or ground-truth history ('with_gt')."
    )
    # <<< NEW: Argument to specify the action to filter on >>>
    parser.add_argument(
        "--filter-action",
        type=str,
        default="response_fail",
        help="Optional. Only run inference on samples that contain this action in their ground-truth dialogue (e.g., 'response_fail')."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # <<< MODIFIED: Pass the new filter_action argument >>>
    run_inference_on_test_set(
        base_model_path=args.base_model_path,
        lora_path=args.lora_path,
        output_file=args.output_file,
        max_samples=args.max_samples,
        inference_mode=args.inference_mode,
        filter_action=args.filter_action,
    )