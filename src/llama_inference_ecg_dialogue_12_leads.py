# Place this at the top with your other imports
from transformers import LlamaTokenizerFast
import os
import sys
import json
import torch
import argparse
import re
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from peft import PeftModel 
from prompts import ECG_EVALUATION_PROMPT

def load_model_and_tokenizer(base_model_path, adapter_path):
    """Loads the base model and applies the PEFT adapter."""
    print(f"Loading base model from: {base_model_path}")
    print(f"Applying adapter from: {adapter_path}")

    # Use bfloat16 if supported for better performance, otherwise float16
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    # Load the base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=dtype,
        device_map="auto",  # Automatically distributes model across available GPUs
    )

    # Load the tokenizer from the adapter path (it's often saved there during fine-tuning)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=True)
    
    # Load the PEFT model (adapter) and apply it to the base model
    model = PeftModel.from_pretrained(model, adapter_path)
    
    # Set to evaluation mode
    model.eval()

    # Llama models often don't have a pad token; set it to the EOS token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer

def load_ground_truth_data(measurement_path, classification_path):
    """Loads pre-computed tool outputs (ground truth) from CSV files."""
    print("Loading ground-truth data from CSV files...")
    try:
        measurement_df = pd.read_csv(measurement_path)
        classification_df = pd.read_csv(classification_path)
        print("✅ Successfully loaded all ground-truth summary CSVs.")
        return {
            "measurement": measurement_df,
            "classification": classification_df
        }
    except FileNotFoundError as e:
        print(f"🛑 Error: Could not find a CSV file. Please check your paths.")
        print(f"Details: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"🛑 An error occurred while loading CSV files: {e}")
        sys.exit(1)

def get_precomputed_tool_output(action, ecg_filename, gt_data):
    """Retrieves a pre-computed tool output from the loaded dataframes."""
    if not ecg_filename:
        return "[Error: ECG filename not provided]"
    try:
        if action == "call_classification_tool":
            df = gt_data["classification"]
            record = df[df['ecg_file_path'] == ecg_filename]
            if record.empty:
                return "[]"
            top_classes_val = record['top_classes'].iloc[0]
            return str([c.strip() for c in top_classes_val.split(',')]) if pd.notna(top_classes_val) else "[]"
        elif action == "call_measurement_tool":
            df = gt_data["measurement"]
            record = df[df['ecg_file_path'] == ecg_filename]
            if record.empty:
                return "{}"
            rec = record.iloc[0]
            measurements = {
                "heart_rate": f"{rec.get('Heart_Rate'):.2f}" if pd.notna(rec.get('Heart_Rate')) else None,
                "pr_interval": f"{rec.get('PR_Interval_ms'):.0f}" if pd.notna(rec.get('PR_Interval_ms')) else None,
                "qrs_duration": f"{rec.get('QRS_Duration_ms'):.0f}" if pd.notna(rec.get('QRS_Duration_ms')) else None,
                "qtc_interval": f"{rec.get('QTc_ms'):.2f}" if pd.notna(rec.get('QTc_ms')) else None,
            }
            return json.dumps(measurements)
            return "[]"
    except Exception:
        return f"[Error: Failed to retrieve data for {ecg_filename}]"
    return "[Error: Unknown tool action]"

def parse_generated_response(generated_text):
    """
    Parses the full text output from the model to extract action, thought, and content.
    This version is highly robust and handles:
    - Tags with or without square brackets [].
    - Case-insensitivity (Action:, action:, etc.).
    - Multi-line thoughts and content.
    - Tool outputs containing JSON-like structures with {}.
    """
    text = generated_text.strip()
    action, thought, content = '', '', ''

    # Define robust patterns with optional brackets and careful multiline handling
    # The action is expected to be a single line.
    action_pattern = r"(?im)^\s*\[?\s*Action\s*[:\-]\s*([^\n\r\]]+)"
    
    # The thought can be multiline. We use a "positive lookahead" (?=...) to make it
    # capture everything until the *next* tag is found or the text ends.
    thought_pattern = r"^\s*\[?Thought:\s*(.*?)(?=\n\s*\[?(?:Action|Content|Tool_Output):|$)"
    
    # Content and Tool_Output can be multiline and capture everything to the end.
    content_pattern = r"^\s*\[?Content:\s*(.*)"
    tool_output_pattern = r"^\s*\[?Tool_Output:\s*(.*)"

    # Define the regex flags to be used
    single_line_flags = re.IGNORECASE | re.MULTILINE
    multi_line_flags = re.IGNORECASE | re.MULTILINE | re.DOTALL

    # Find matches for all parts
    action_match = re.search(action_pattern, text, single_line_flags)
    thought_match = re.search(thought_pattern, text, multi_line_flags)
    content_match = re.search(content_pattern, text, multi_line_flags)
    tool_output_match = re.search(tool_output_pattern, text, multi_line_flags)

    # Extract data from matches
    if action_match:
        action = action_match.group(1).strip()
    
    if thought_match:
        thought = thought_match.group(1).strip()

    # Intelligently determine the final content
    if content_match:
        # If there is an explicit "Content:" tag, that's our content.
        content = content_match.group(1).strip()
    elif not (action_match or thought_match or tool_output_match):
        # If NO tags were found at all, the entire text is the user-facing content.
        content = text
    # Otherwise, `content` remains a blank string (''), which is the correct behavior
    # for a tool call turn that doesn't have an explicit "Content:" section.
        
    return {
        'role': 'assistant',
        'action': action,
        'thought': thought,
        'content': content
    }

def generate_full_response(model, tokenizer, messages, generation_config):
    """
    Generates a response using the provided message history.
    It now takes a list of message dicts and uses the chat template.
    """
    # Apply the chat template
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(**inputs, generation_config=generation_config)

    # Decode only the newly generated tokens
    new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
    decoded_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

    return decoded_text.strip()

def format_assistant_turn_for_messages(turn):
    """
    Formats an assistant turn into the string content for the message list.
    """
    action = turn.get('action', '')
    thought = turn.get('thought', '')

    assistant_content = f"Action: {action}\nThought: {thought}\n"

    if 'tool_output' in turn:
        assistant_content += f"Tool_Output: {turn['tool_output']}"
    elif 'content' in turn:
        assistant_content += f"Content: {turn.get('content', '')}"

    return assistant_content.strip()

def build_aligned_turns(gt_dialogue, gen_dialogue):
    """
    Create a per-turn alignment of assistant turns between GT and Generated dialogues.
    Also attaches the most recent user message before each assistant turn (from GT).
    Returns: (aligned_turns, num_gt_assistant, num_gen_assistant)
    """
    gt_assistant_turns = [t for t in gt_dialogue if t.get("role") == "assistant"]
    gen_assistant_turns = [t for t in gen_dialogue if t.get("role") == "assistant"]

    # Track the latest user message before each assistant turn (from GT)
    user_contexts = []
    last_user_content = None
    for t in gt_dialogue:
        if t.get("role") == "user":
            last_user_content = t.get("content", "")
        elif t.get("role") == "assistant":
            user_contexts.append(last_user_content)

    n_max = max(len(gt_assistant_turns), len(gen_assistant_turns))
    aligned = []
    for i in range(n_max):
        gt_t = gt_assistant_turns[i] if i < len(gt_assistant_turns) else {}
        gen_t = gen_assistant_turns[i] if i < len(gen_assistant_turns) else {}

        gt_action = gt_t.get("action")
        gen_action = gen_t.get("action")
        gt_content = gt_t.get("content")
        gen_content = gen_t.get("content")

        # Convenience flags for quick metric scripts
        action_correct = (gt_action is not None and gen_action is not None and str(gt_action).strip() == str(gen_action).strip())
        response_exact_match = (gt_content is not None and gen_content is not None and str(gt_content).strip() == str(gen_content).strip())

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

            "action_correct": action_correct,
            "response_exact_match": response_exact_match,
        })

    return aligned, len(gt_assistant_turns), len(gen_assistant_turns)

# ### MAIN ###
# <<< CHANGED: Main function signature updated >>>
def run_inference_on_test_set(
    base_model_path, 
    adapter_path,
    measurement_path,
    classification_path, 
    output_file=None, 
    max_samples=None, 
    inference_mode='without_gt', 
    filter_action=None):

    model, tokenizer = load_model_and_tokenizer(base_model_path, adapter_path)
    gt_data = load_ground_truth_data(measurement_path, classification_path)

    # Deterministic generation for eval
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    eos_ids = [t for t in [tokenizer.eos_token_id, eot_id] if t is not None]
    generation_config = GenerationConfig(
        max_new_tokens=512,
        temperature=0.0,
        top_p=1.0,
        do_sample=False,
        eos_token_id=eos_ids or None,
        pad_token_id=tokenizer.pad_token_id,
    )

    dataset = load_dataset("gustmd0121/12-lead-ecg-mtd-dataset")['test']
    if filter_action:
        print(f"Filtering dataset to only include samples with the action: '{filter_action}'")
        
        def contains_action(example):
            try:
                # Load the dialogue from the JSON string
                dialogue = json.loads(example['dialogue'])
                # Check if any turn in the dialogue has the target action
                for turn in dialogue:
                    if turn.get('action') == filter_action:
                        return True
                return False
            except (json.JSONDecodeError, TypeError):
                # If the dialogue string is invalid, exclude the sample
                return False

        original_size = len(dataset)
        dataset = dataset.filter(contains_action, num_proc=4) # Use num_proc for faster filtering
        print(f"✅ Filtering complete. Found {len(dataset)} matching samples out of {original_size}.")


    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    print(f"Running inference on {len(dataset)} samples in '{inference_mode}' mode...")

    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name_tag = os.path.basename(adapter_path) # Use adapter path for a unique name
        output_file = f"inference_{model_name_tag}_{timestamp}_{inference_mode}.jsonl"

    # Open once in append mode; append one JSONL line per sample and persist immediately
    with open(output_file, 'a', encoding='utf-8') as f_out:
        for i, example in enumerate(tqdm(dataset, desc="Generating Dialogues")):
            try:
                gt_dialogue = json.loads(example['dialogue'])
                ecg_files_str = example.get('ecg_files')
                file_list = json.loads(ecg_files_str) if ecg_files_str else []
                ecg_filename = file_list[0] if file_list else None

                generated_dialogue = []
                messages = [{"role": "system", "content": ECG_EVALUATION_PROMPT}]

                for turn in gt_dialogue:
                    if turn['role'] == 'user':
                        # Add user turn
                        user_turn = {"role": "user", "content": turn.get('content', '')}
                        messages.append(user_turn)
                        generated_dialogue.append(turn)

                        # First assistant step
                        model_output_str = generate_full_response(model, tokenizer, messages, generation_config)
                        parsed_turn = parse_generated_response(model_output_str)

                        model_generated_turns_for_this_user_prompt = []

                        # Tool call or direct response?
                        if parsed_turn['action'] in ["call_classification_tool", "call_measurement_tool"]:
                            tool_call_turn = {
                                "role": "assistant",
                                "action": parsed_turn['action'],
                                "thought": parsed_turn['thought'],
                                "tool_output": get_precomputed_tool_output(parsed_turn['action'], ecg_filename, gt_data)
                            }
                            model_generated_turns_for_this_user_prompt.append(tool_call_turn)

                            # Final response using tool output
                            messages.append({"role": "assistant", "content": format_assistant_turn_for_messages(tool_call_turn)})
                            final_content_str = generate_full_response(model, tokenizer, messages, generation_config)
                            parsed_final_turn = parse_generated_response(final_content_str)

                            response_turn = {
                                "role": "assistant",
                                "action": parsed_final_turn.get("action", "response"), 
                                "thought": parsed_final_turn.get("thought", "No thought generated."), # Provide a descriptive default thought.
                                "content": parsed_final_turn.get("content", "") # The only part we truly need from the model's second output.
                            }
                            model_generated_turns_for_this_user_prompt.append(response_turn)
                        else:
                            direct_response_turn = {
                                "role": "assistant",
                                "action": parsed_turn['action'],
                                "thought": parsed_turn['thought'],
                                "content": parsed_turn.get("content", "")
                            }
                            model_generated_turns_for_this_user_prompt.append(direct_response_turn)

                        generated_dialogue.extend(model_generated_turns_for_this_user_prompt)

                        # Update history
                        if inference_mode == 'without_gt':
                            if len(model_generated_turns_for_this_user_prompt) == 1:
                                messages.append({"role": "assistant", "content": format_assistant_turn_for_messages(model_generated_turns_for_this_user_prompt[0])})
                            elif len(model_generated_turns_for_this_user_prompt) == 2:
                                messages.append({"role": "assistant", "content": format_assistant_turn_for_messages(model_generated_turns_for_this_user_prompt[1])})
                        elif inference_mode == 'with_gt':
                            messages.pop()  # remove last user
                            if len(messages) > 1 and messages[-1]['role'] == 'assistant':
                                messages.pop()
                            current_gt_index = len(generated_dialogue)
                            messages = [{"role": "system", "content": ECG_EVALUATION_PROMPT}]
                            for gt_turn in gt_dialogue[:current_gt_index]:
                                role = gt_turn['role']
                                content = format_assistant_turn_for_messages(gt_turn) if role == 'assistant' else gt_turn.get('content', '')
                                messages.append({'role': role, 'content': content})

                # Build aligned per-turn view
                aligned_turns, n_gt_asst, n_gen_asst = build_aligned_turns(gt_dialogue, generated_dialogue)

                record = {
                    "sample_id": i,
                    "ecg_file": example.get("ecg_files"),
                    "source_category": example.get("source_category"),
                    "turns": aligned_turns,
                    "generated_dialogue": generated_dialogue,
                    "ground_truth_dialogue": gt_dialogue,
                    "summary": {
                        "num_gt_assistant_turns": n_gt_asst,
                        "num_generated_assistant_turns": n_gen_asst,
                    }
                }

                # Append one JSON line per sample and fsync
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                f_out.flush()
                os.fsync(f_out.fileno())

            except Exception as e:
                import traceback
                print(f"Error processing sample {i} (ECG: {ecg_filename}): {e}\n{traceback.format_exc()}")

    print(f"Inference complete. Results appended to {output_file}")

# ----------------------------
# <<< CHANGED: Argparse updated for base and adapter model paths >>>
# ----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with an unmerged (base + adapter) ECG dialogue model.")
    parser.add_argument(
        "--base-model-path",
        type=str,
        default="unsloth/Llama-3.1-8B-Instruct", # Make this required
        help="Path to the base model (e.g., 'meta-llama/Meta-Llama-3-8B-Instruct')."
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default="./src/model_epoch_overall_Llama-3.1-8B-Instruct_20251015_220107_hf", # Make this required
        help="Path to the fine-tuned adapter folder (containing adapter_config.json)."
    )

    parser.add_argument(
        "--measurement-summary-path",
        type=str,
        default="/nfs_edlab/hschung/MedRAX/tool_output_classes/ecg_analysis_measurements_intervals_lead_ii.csv",
        help="Path to ECG measurement summary CSV file."
    )

    parser.add_argument(
        "--classification-summary-path",
        type=str,
        default="/nfs_edlab/hschung/MedRAX/tool_output_classes/ecg_analysis_summary_top_classes_0_5.csv",
        help="Path to ECG classification summary CSV file."
    )
    
    parser.add_argument(
        "--output-file",
        type=str,
        default="./inference/llama_8b_results_8B_Instruct_20251015_220107.jsonl",
        help="Optional output .jsonl path. If omitted, a timestamped file is created."
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit the number of test samples to run."
    )
    parser.add_argument(
        "--inference-mode",
        type=str,
        choices=["without_gt", "with_gt"],
        default="with_gt",
        help="Use model-generated history (without_gt) or ground-truth history between turns (with_gt)."
    )
    
    parser.add_argument(
        "--filter-action",
        type=str,
        default=None,
        help="Optional. Only run inference on samples that contain this action in their ground-truth dialogue (e.g., 'response_fail')."
    )
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    run_inference_on_test_set(
        base_model_path=args.base_model_path,
        adapter_path=args.adapter_path,
        measurement_path=args.measurement_summary_path,
        classification_path=args.classification_summary_path,
        output_file=args.output_file,
        max_samples=args.max_samples,
        inference_mode=args.inference_mode,
        filter_action=args.filter_action,
    )