#!/usr/bin/env python3
"""
ECG Dialogue Generation Script for LLaVA-based Models

This script processes an ECG dialogue dataset, generates corresponding ECG images,
and runs a LLaVA-style model to generate dialogue responses.

MODIFIED to handle different lead configurations based on the dataset name:
- If 'dataset-name' contains 'single-lead-I', it processes only lead I (index 0).
- If 'dataset-name' contains 'single-lead-II', it processes only lead II (index 1).
- Otherwise, it defaults to processing all 12 leads.
"""

import os
import json
import argparse
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import wfdb
from datasets import load_dataset
from PIL import Image
import pdb 

try:
    import ecg_plot
    from llava.constants import (
        IMAGE_TOKEN_INDEX,
        DEFAULT_IMAGE_TOKEN,
        DEFAULT_IM_START_TOKEN,
        DEFAULT_IM_END_TOKEN,
    )
    from llava.conversation import conv_templates
    from llava.model.builder import load_pretrained_model
    from llava.utils import disable_torch_init
    from llava.mm_utils import (
        process_images,
        tokenizer_image_token,
        get_model_name_from_path,
    )
except ImportError as e:
    print(f"Error: A required local module could not be imported: {e}")
    print("Please ensure that 'ecg_plot' and the 'llava' package are correctly installed and in your PYTHONPATH.")
    sys.exit(1)

# MODIFICATION: Added lead_config argument to handle different lead setups
def load_ecg_and_create_image(ecg_path, output_img_path, lead_config="12-lead"):
    """Load ECG data and create an image based on the specified lead configuration."""
    output_img_path_with_ext = output_img_path + '.png'
    if os.path.exists(output_img_path_with_ext):
        return output_img_path_with_ext
    
    try:
        try:
            record = wfdb.rdrecord(ecg_path)
        except FileNotFoundError:
            ecg_path_no_ext = os.path.splitext(ecg_path)[0]
            record = wfdb.rdrecord(ecg_path_no_ext)
            
        signal = record.p_signal.T  # Transpose to (12, 5000)
        
        # --- MODIFICATION: Conditional plotting based on lead_config ---
        # pdb.set_trace()
        if lead_config == "single-lead-I":
            print(f"Plotting single lead I (index 0) for {os.path.basename(ecg_path)}")
            # The ecg-plot library expects a 1D array for plot_1
            ecg_plot.plot_1(signal[0], sample_rate=record.fs, title='Lead I')
        elif lead_config == "single-lead-II":
            print(f"Plotting single lead II (index 1) for {os.path.basename(ecg_path)}")
            ecg_plot.plot_1(signal[1], sample_rate=record.fs, title='Lead II')
        else: # Default to 12-lead
            ecg_plot.plot(signal, sample_rate=record.fs)
        # --- End of Modification ---

        output_dir = os.path.dirname(output_img_path)
        output_filename = os.path.basename(output_img_path)
        
        ecg_plot.save_as_png(output_filename, output_dir + '/')
        plt.close('all')
        
        return output_img_path_with_ext
        
    except Exception as e:
        print(f"Error creating ECG image from {ecg_path}: {e}")
        return None

# MODIFICATION: Added lead_config argument to mask signals
def load_ecg_signal(ecg_path, lead_config="12-lead", seq_length=5000):
    """Load ECG signal data, masking leads based on the configuration."""
    try:
        ecg_data, _ = wfdb.rdsamp(ecg_path)
        ecg_data[np.isnan(ecg_data)] = 0
        ecg_data[np.isinf(ecg_data)] = 0
        ecg = torch.Tensor(np.transpose(ecg_data, (1, 0)).astype(np.float32))
        
        # --- MODIFICATION: Conditional signal masking ---
        if lead_config == "single-lead-I":
            print(f"Masking signal for Lead I (index 0) for {os.path.basename(ecg_path)}")
            masked_ecg = torch.zeros_like(ecg)
            masked_ecg[0, :] = ecg[0, :] # Keep only lead I
            ecg = masked_ecg
        elif lead_config == "single-lead-II":
            print(f"Masking signal for Lead II (index 1) for {os.path.basename(ecg_path)}")
            masked_ecg = torch.zeros_like(ecg)
            masked_ecg[1, :] = ecg[1, :] # Keep only lead II
            ecg = masked_ecg
        # --- End of Modification ---

        c, length = ecg.shape
        if length < seq_length:
            new_ecg = torch.zeros((c, seq_length))
            new_ecg[:, 0:length] = ecg
            ecg = new_ecg
        elif length > seq_length:
            ecg = ecg[:, 0:seq_length]
            
        return ecg
        
    except Exception as e:
        print(f"Error loading ECG signal from {ecg_path}: {e}")
        return None

def run_llava_model_multiturn_with_ecg(model, tokenizer, image_processor, image_path, ecg_signal, dialogue_turns, conv_mode="llava_v1", history_mode="without_gt", gt_responses=None):
    """Run the LLaVA model with multi-turn dialogue flow and ECG signal input."""
    try:
        image = Image.open(image_path).convert("RGB")
        images_tensor = process_images([image], image_processor, model.config)[0]
        
        def wrap_first_query(qs):
            return f"{DEFAULT_IM_START_TOKEN}{DEFAULT_IMAGE_TOKEN}{DEFAULT_IM_END_TOKEN}\n{qs}" if model.config.mm_use_im_start_end else f"{DEFAULT_IMAGE_TOKEN}\n{qs}"
        
        def model_generate(input_ids, image_tensor, ecg_tensor):
            with torch.inference_mode():
                device = next(model.parameters()).device
                input_ids = input_ids.to(device)
                
                if ecg_tensor is not None:
                    ecg_tensor = ecg_tensor.half().to(device)
                
                if len(image_tensor.shape) == 3:
                    image_tensor = image_tensor.unsqueeze(0)
                image_tensor = image_tensor.half().to(device)
                
                output_ids = model.generate(
                    input_ids,
                    ecgs=None,
                    images=image_tensor,
                    image_sizes=[image.size],
                    do_sample=True, temperature=0.2, top_p=None,
                    num_beams=1, max_new_tokens=512, use_cache=True,
                )
            return tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        
        conv_history = conv_templates[conv_mode].copy()
        model_responses = []
        
        for turn_idx, turn in enumerate(dialogue_turns):
            if turn["role"] == "user":
                cur_prompt = turn["content"]
                
                if turn_idx == 0:
                    qs = wrap_first_query(cur_prompt)
                else:
                    qs = cur_prompt

                conv_history.append_message(conv_history.roles[0], qs)
                conv_history.append_message(conv_history.roles[1], None)
                prompt = conv_history.get_prompt()
                input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0)
                
                # For subsequent turns, we don't need to pass the image again if the model architecture supports it
                # but for simplicity and robustness, we pass it every time.
                model_response = model_generate(input_ids, images_tensor, ecg_signal)
                model_responses.append(model_response)
                
                if history_mode == "with_gt" and gt_responses and turn_idx < len(gt_responses):
                    conv_history.messages[-1][1] = gt_responses[turn_idx]
                else:
                    conv_history.messages[-1][1] = model_response
        
        return model_responses
        
    except Exception as e:
        print(f"Error running LLaVA model: {e}")
        return [f"Error generating response: {str(e)}"]

# MODIFICATION: Added lead_config argument
def process_ecg_dialogue_dataset_with_ecg_signal(dataset_name, ecg_base_dir, output_dir, model_path, conv_mode, history_mode, split, max_samples, lead_config):
    """Process ECG dialogue dataset with ECG signal input based on lead configuration."""
    os.makedirs(output_dir, exist_ok=True)
    ecg_images_dir = os.path.join(output_dir, "ecg_images")
    os.makedirs(ecg_images_dir, exist_ok=True)
    
    print("Loading model...")
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path, None, "llava_llama", device_map="cuda:0"
    )
    # pdb.set_trace()
    print("Model loaded successfully!")

    print(f"Loading dataset {dataset_name}...")
    dataset = load_dataset(dataset_name, split=f"{split}[:{max_samples}]" if max_samples else split)
    print(f"Found {len(dataset)} samples to process")
    
    all_results = []
    
    try:
        for idx, sample in enumerate(tqdm(dataset, desc="Processing dialogues")):
            try:
                metadata = json.loads(sample["metadata"])
                dialogue = json.loads(sample["dialogue"])
                ecg_files = json.loads(sample["ecg_files"])
                
                ecg_id = metadata.get("ecg_id")
                if not ecg_id or not ecg_files:
                    print(f"Skipping sample {idx} due to missing ECG ID or file info.")
                    continue

                ecg_file = ecg_files[0]
                ecg_num = ecg_file[2:-4]
                ecg_num_int = int(ecg_num)
                dir_num = (ecg_num_int // 1000) * 1000
                ecg_signal_path = os.path.join(ecg_base_dir, "records500", f"{dir_num:05d}", f"{ecg_num_int:05d}_hr")
                
                ecg_image_path = os.path.join(ecg_images_dir, str(ecg_id))
                # pdb.set_trace()
                # Pass lead_config to image creation and signal loading
                ecg_image_path_with_ext = load_ecg_and_create_image(ecg_signal_path, ecg_image_path, lead_config)
                if not ecg_image_path_with_ext:
                    print(f"Failed to create ECG image for {ecg_id}, skipping...")
                    continue
                
                ecg_signal = load_ecg_signal(ecg_signal_path, lead_config)
                if ecg_signal is None:
                    print(f"Failed to load ECG signal for {ecg_id}, skipping...")
                    continue
                pdb.set_trace()
                tool_actions = ["call_classification_tool", "call_measurement_tool", "call_explanation_tool"]
                user_turns = [msg for msg in dialogue if msg["role"] == "user"]
                gt_responses = [msg["content"] for msg in dialogue if msg["role"] == "assistant" and msg.get("action") not in tool_actions and "content" in msg]
                
                if not user_turns:
                    print(f"No valid user turns found for {ecg_id}, skipping...")
                    continue
                    
                model_responses = run_llava_model_multiturn_with_ecg(
                    model, tokenizer, image_processor,
                    ecg_image_path_with_ext, ecg_signal, user_turns, conv_mode,
                    history_mode=history_mode, gt_responses=gt_responses
                )
                
                result_dialogue = []
                action_sequence = [a for a in metadata.get("action_sequence_prompted", []) if a not in tool_actions]
                action_idx = 0
                for i, user_turn in enumerate(user_turns):
                    user_action = action_sequence[action_idx] if action_idx < len(action_sequence) else "ECG_Inquiry"
                    result_dialogue.append({"role": "user", "action": user_action, "content": user_turn["content"]})
                    action_idx += 1
                    
                    if i < len(model_responses):
                        assistant_action = action_sequence[action_idx] if action_idx < len(action_sequence) else "response"
                        result_dialogue.append({"role": "assistant", "action": assistant_action, "content": model_responses[i]})
                        action_idx += 1
                
                all_results.append({
                    "metadata": json.dumps(metadata),
                    "dialogue": json.dumps(result_dialogue),
                    "ecg_files": json.dumps(ecg_files)
                })
                
            except Exception as e:
                print(f"Error processing sample {idx} (ECG ID: {ecg_id if 'ecg_id' in locals() else 'unknown'}): {e}")
                continue
    
    finally:
        del model, tokenizer, image_processor
        torch.cuda.empty_cache()
    
    output_file = os.path.join(output_dir, f"GEM_ecg_dialogue_model_responses_{split}.json")
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"Saved {len(all_results)} processed dialogues to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Process ECG dialogue dataset with GEM model")
    parser.add_argument("--dataset-name", type=str, default="gustmd0121/single-lead-II-ecg-mtd-dataset", help="HuggingFace dataset name")
    parser.add_argument("--ecg-base-dir", type=str, default="./ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3", help="Base directory for ECG signal files")
    parser.add_argument("--output-dir", type=str, default="./results/GEM/single_lead_ii/with_gt", help="Base directory to save results and images")
    parser.add_argument("--model-path", type=str, default="LANSG/GEM", help="Path to the GEM model")
    parser.add_argument("--conv-mode", type=str, default="llava_v1", help="Conversation mode for the model")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to process")
    parser.add_argument("--max-samples", type=int, default=None, help="Maximum number of samples to process")
    parser.add_argument("--history-mode", type=str, default="with_gt", choices=["without_gt", "with_gt"], help="Conversation history mode")
    
    args = parser.parse_args()
    
    # --- MODIFICATION: Determine lead configuration from dataset name ---
    lead_config = "12-lead" # Default
    if "single-lead-ii" in args.dataset_name.lower():
        lead_config = "single-lead-II"
    elif "single-lead-i" in args.dataset_name.lower():
        lead_config = "single-lead-I"
    # --- End of Modification ---
    
    print(f"Starting ECG dialogue processing with model: {args.model_path}")
    print(f"Processing dataset: {args.dataset_name}")
    print(f"History mode: {args.history_mode}")
    print(f"LEAD CONFIGURATION DETECTED: {lead_config}") # Info message
    
    process_ecg_dialogue_dataset_with_ecg_signal(
        args.dataset_name,
        args.ecg_base_dir,
        args.output_dir,
        args.model_path,
        args.conv_mode,
        args.history_mode,
        args.split,
        args.max_samples,
        lead_config # Pass the determined configuration
    )
    
    print("ECG dialogue processing complete!")

if __name__ == "__main__":
    main()