#!/usr/bin/env python3
"""
Multimodal Benchmark Framework for ECG Dialogue Processing
Supports Gemini with with_gt and without_gt history modes.

MODIFIED to handle different lead configurations based on the dataset name:
- If 'dataset-name' contains 'single-lead-I', it processes only lead I (index 0).
- If 'dataset-name' contains 'single-lead-II', it processes only lead II (index 1).
- Otherwise, it defaults to processing all 12 leads.
"""

import os
import sys
import json
import argparse
import base64
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import matplotlib.pyplot as plt
import wfdb
from datasets import load_dataset
from PIL import Image
import ecg_plot


class MultimodalModel(ABC):
    """Abstract base class for multimodal models."""
    
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.model_config = kwargs
    
    @abstractmethod
    def generate_response(self, image_path: str, prompt: str, conversation_history: List[Dict] = None) -> str:
        """Generate response for given image and prompt with optional conversation history."""
        pass
    
    @abstractmethod
    def setup_model(self):
        """Initialize the model and any required configurations."""
        pass
    
    def encode_image_base64(self, image_path: str) -> str:
        """Encode image to base64 string."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

class GeminiModel(MultimodalModel):
    """Gemini model implementation."""
    
    def __init__(self, model_name: str = "gemini-2.5-pro", **kwargs):
        super().__init__(model_name, **kwargs)
        self.client = None
        
    def setup_model(self):
        """Initialize Gemini client."""
        try:
            import google.generativeai as genai
            
            api_key = os.getenv('GEMINI_API_KEY') or self.model_config.get('api_key')
            if not api_key:
                raise ValueError("GEMINI_API_KEY environment variable or api_key parameter required")
            
            genai.configure(api_key=api_key)
            self.client = genai.GenerativeModel(self.model_name)
            print(f"Gemini model {self.model_name} initialized successfully")
            
        except ImportError:
            raise ImportError("google-generativeai package required. Install with: pip install google-generativeai")
    
    def generate_response(self, image_path: str, prompt: str, conversation_history: List[Dict] = None) -> str:
        """Generate response using Gemini."""
        try:
            image = Image.open(image_path)
            
            full_prompt = self._build_conversation_prompt(prompt, conversation_history) if conversation_history else prompt
            
            response = self.client.generate_content([full_prompt, image])
            return response.text
            
        except Exception as e:
            error_str = str(e).lower()
            if "429" in error_str and "quota" in error_str:
                print(f"\nFATAL ERROR: Gemini API quota exceeded. Stopping the script.\nDetails: {e}")
                sys.exit(1)
            else:
                print(f"Error generating Gemini response: {e}")
                return f"Error generating response: {str(e)}"
    
    def _build_conversation_prompt(self, current_prompt: str, history: List[Dict]) -> str:
        """Build conversation prompt from history."""
        conversation = "Previous conversation:\n"
        for turn in history:
            role = "User" if turn["role"] == "user" else "Assistant"
            conversation += f"{role}: {turn['content']}\n"
        conversation += f"\nCurrent question: {current_prompt}"
        return conversation

def load_ecg_and_create_image(ecg_path: str, output_img_path: str, lead_config: str = "12-lead") -> Optional[str]:
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
            
        signal = record.p_signal.T
        
        if lead_config == "single-lead-I":
            ecg_plot.plot_1(signal[0], sample_rate=record.fs, title='Lead I')
        elif lead_config == "single-lead-II":
            ecg_plot.plot_1(signal[1], sample_rate=record.fs, title='Lead II')
        else:
            ecg_plot.plot(signal, sample_rate=record.fs)

        output_dir = os.path.dirname(output_img_path)
        output_filename = os.path.basename(output_img_path)
        
        ecg_plot.save_as_png(output_filename, output_dir + '/')
        plt.close('all')
        
        return output_img_path_with_ext
        
    except Exception as e:
        print(f"Error creating ECG image from {ecg_path}: {e}")
        return None

class MultimodalBenchmarkFramework:
    """Main framework for running multimodal ECG dialogue benchmarks."""
    
    def __init__(self, model: MultimodalModel):
        self.model = model
        self.model.setup_model()
    
    def run_multiturn_dialogue(self, image_path: str, user_turns: List[Dict], 
                               history_mode: str = "without_gt", gt_responses: List[str] = None) -> List[str]:
        """Run multi-turn dialogue with specified history mode."""
        model_responses = []
        conversation_history = []
        
        for turn_idx, user_turn in enumerate(user_turns):
            prompt = user_turn["content"]
            
            history_for_this_turn = conversation_history if turn_idx > 0 else None
            model_response = self.model.generate_response(image_path, prompt, history_for_this_turn)
            
            model_responses.append(model_response)
            
            conversation_history.append({"role": "user", "content": prompt})
            if history_mode == "with_gt" and gt_responses and turn_idx < len(gt_responses):
                conversation_history.append({"role": "assistant", "content": gt_responses[turn_idx]})
            else:
                conversation_history.append({"role": "assistant", "content": model_response})
        
        return model_responses
    
    def process_dataset(self, dataset_name: str, ecg_base_dir: str, output_dir: str,
                        split: str = "test", max_samples: Optional[int] = None, 
                        history_mode: str = "without_gt", lead_config: str = "12-lead") -> str:
        """Process ECG dialogue dataset and save results incrementally."""
        os.makedirs(output_dir, exist_ok=True)
        ecg_images_dir = os.path.join(output_dir, "ecg_images")
        os.makedirs(ecg_images_dir, exist_ok=True)
        
        # --- MODIFICATION: Use .jsonl for robust, incremental saving ---
        output_file = os.path.join(output_dir, f"ecg_dialogue_{self.model.model_name.replace('/', '_')}_{history_mode}_responses.jsonl")

        # --- MODIFICATION: Load existing ECG IDs from the .jsonl file to avoid reprocessing ---
        existing_ecg_ids = set()
        if os.path.exists(output_file):
            print(f"Loading existing results from {output_file}")
            try:
                with open(output_file, 'r') as f:
                    for line in f:
                        if line.strip(): # Ensure line is not empty
                            entry = json.loads(line)
                            if "metadata" in entry:
                                meta = json.loads(entry["metadata"])
                                if "ecg_id" in meta:
                                    existing_ecg_ids.add(str(meta["ecg_id"]))
                print(f"Loaded and will skip {len(existing_ecg_ids)} existing ECG IDs.")
            except (json.JSONDecodeError, FileNotFoundError) as e:
                print(f"Warning: Could not parse {output_file}: {e}. Starting fresh.")
                existing_ecg_ids = set()
        
        dataset = load_dataset(dataset_name, split=f"{split}[:{max_samples}]" if max_samples else split)
        print(f"Found {len(dataset)} total samples in dataset")
        
        new_results_count = 0
        # --- MODIFICATION: Open file in append mode to save results one by one ---
        with open(output_file, 'a') as f_out:
            for idx, sample in enumerate(tqdm(dataset, desc="Processing dialogues")):
                ecg_id = "unknown"
                try:
                    metadata = json.loads(sample["metadata"])
                    dialogue = json.loads(sample["dialogue"])
                    ecg_files = json.loads(sample["ecg_files"])
                    
                    ecg_id = metadata.get("ecg_id")
                    if not ecg_id: continue
                    
                    if str(ecg_id) in existing_ecg_ids:
                        continue
                    
                    if not ecg_files: continue
                    
                    ecg_file = ecg_files[0]
                    ecg_num = ecg_file[2:-4]
                    ecg_num_int = int(ecg_num)
                    dir_num = (ecg_num_int // 1000) * 1000
                    ecg_signal_path = os.path.join(ecg_base_dir, "records500", f"{dir_num:05d}", f"{ecg_num}_hr")
                    
                    ecg_image_path = os.path.join(ecg_images_dir, str(ecg_id))
                    
                    ecg_image_path_with_ext = load_ecg_and_create_image(ecg_signal_path, ecg_image_path, lead_config)
                    if not ecg_image_path_with_ext: continue
                    
                    tool_actions = ["call_classification_tool", "call_measurement_tool", "call_explanation_tool"]
                    user_turns = [msg for msg in dialogue if msg["role"] == "user"]
                    gt_responses = [msg["content"] for msg in dialogue if msg["role"] == "assistant" and msg.get("action") not in tool_actions and "content" in msg]
                    
                    if not user_turns: continue
                        
                    model_responses = self.run_multiturn_dialogue(
                        ecg_image_path_with_ext, user_turns, history_mode, gt_responses
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
                    
                    result_entry = {
                        "metadata": json.dumps(metadata),
                        "dialogue": json.dumps(result_dialogue),
                        "ecg_files": json.dumps(ecg_files)
                    }

                    # --- MODIFICATION: Write the processed result directly to the file ---
                    f_out.write(json.dumps(result_entry) + '\n')
                    new_results_count += 1
                    
                except Exception as e:
                    print(f"Error processing sample {idx} (ECG ID: {ecg_id}): {e}")
                    continue
        
        # --- MODIFICATION: Final summary message ---
        total_in_file = len(existing_ecg_ids) + new_results_count
        print(f"Processing complete. Saved {new_results_count} new dialogues.")
        print(f"Total dialogues in {output_file}: {total_in_file}")
        return output_file

def create_model(model_type: str, model_name: str, **kwargs) -> MultimodalModel:
    """Factory function to create model instances."""
    model_map = {
        "gemini": GeminiModel
    }
    model_class = model_map.get(model_type.lower())
    if model_class:
        return model_class(model_name, **kwargs)
    raise ValueError(f"Unsupported model type: {model_type}")

def main():
    parser = argparse.ArgumentParser(description="Multimodal ECG dialogue benchmark framework")
    parser.add_argument("--model-type", type=str, default="gemini")
    parser.add_argument("--model-name", type=str, default="gemini-2.5-flash", help="Specific model name/path")
    parser.add_argument("--dataset-name", type=str, default="gustmd0121/single-lead-II-ecg-mtd-dataset", help="HuggingFace dataset name")
    parser.add_argument("--ecg-base-dir", type=str, default="./ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3", help="Base directory for ECG signal files (PTB-XL dataset)")
    parser.add_argument("--output-dir", type=str, default="./results/Gemini/single_lead_ii/with_gt", help="Base directory to save results")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to process")
    parser.add_argument("--max-samples", type=int, default=None, help="Maximum number of samples to process")
    parser.add_argument("--history-mode", type=str, default="with_gt", choices=["without_gt", "with_gt"], help="Conversation history mode")
    
    args = parser.parse_args()
    
    lead_config = "12-lead"
    if "single-lead-II" in args.dataset_name:
        lead_config = "single-lead-II"
    elif "single-lead-I" in args.dataset_name:
        lead_config = "single-lead-I"
    
    print("--- Starting Multimodal ECG Dialogue Processing ---")
    print(f"Model: {args.model_type} ({args.model_name})")
    print(f"Dataset: {args.dataset_name}")
    print(f"History Mode: {args.history_mode}")
    print(f"Lead Configuration: {lead_config}")
    print("-" * 50)
    
    model = create_model(args.model_type, args.model_name)
    framework = MultimodalBenchmarkFramework(model)
    
    output_file = framework.process_dataset(
        args.dataset_name,
        args.ecg_base_dir,
        args.output_dir,
        split=args.split,
        max_samples=args.max_samples,
        history_mode=args.history_mode,
        lead_config=lead_config
    )
    
    print(f"\n✅ All done! Results saved to: {output_file}")

if __name__ == "__main__":
    main()