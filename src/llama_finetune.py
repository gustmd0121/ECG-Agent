import json,random,re,yaml,os,datetime
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Dataset, Subset
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
import bitsandbytes
from prompts import ecg_evaluation_prompt
import traceback
from copy import deepcopy
from torch.optim.lr_scheduler import StepLR
import pdb 
import sys
import wandb 
from medrax.tools.classification import ECGClassifierTool, ECGAnalysisTool, ECGExplainTool
import json,random,re,yaml,os,datetime
import numpy as np 
import pdb
import argparse

def set_seed(seed=1):
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

def replace_with_flash_attention(model):
    for name, module in model.named_modules():
        if isinstance(module, nn.MultiheadAttention):  # Check for standard attention layers
            print("Self attn")
            setattr(model, name, FlashSelfAttention(causal=True))  # Causal=True for autoregressive models
            
def detect_model_type(model_id):
    """Detect model type based on model ID for format-specific handling"""
    if model_id is None:
        return "llama"  # Default fallback
    
    model_id_lower = str(model_id).lower()
    if "llama" in model_id_lower:
        return "llama"
    else:
        # Default to llama format for unknown models
        print(f"Warning: Unknown model type for {model_id}, defaulting to LLaMA format")
        return "llama"

def overall_collate_fn(batches):
    x_texts,y,obs,dialogue_idx,ecg_files,original_dialogues = [],[],[],[],[],[]
    for item in batches:
        x_texts.append(item['X'])
        y.append(item['y'])
        obs.append(item['obs'])
        dialogue_idx.append(item['dialogue_idx'])
        ecg_files.append(item.get('ecg_files', []))
        original_dialogues.append(item.get('original_dialogue', []))
    return {"input":x_texts,"output":y,"obs":obs,"dialogue_idx":dialogue_idx,"ecg_files":ecg_files,"original_dialogues":original_dialogues}
################

def normalize_action(text):
    lower_text = text.lower()
    normalized_text = re.sub(r'[^a-z]', '', lower_text)
    return normalized_text.replace("action","").replace("api","")

def ecg_form(dialogue_idx, dialogue_list, tokenizer_inf, is_val=False):
    """
    Corrected version: Formats the dialogue string with the exact field order
    (Action, Thought, Output/Content) and ensures all assistant turns are
    properly separated to create clean training and validation examples.
    """
    import json
    if isinstance(dialogue_list, str):
        try:
            dialogue_list = json.loads(dialogue_list)
        except json.JSONDecodeError as e:
            print(f"Error parsing dialogue_list: {e}")
            return []

    # This function now directly creates training examples turn-by-turn
    # without creating a single, flawed 'dialogue_string' first.
    training_examples = []
    current_history_str = ""

    for i, turn in enumerate(dialogue_list):
        if turn['role'] == 'user':
            if 'content' not in turn:
                print(turn) 
            current_history_str += f"User: {turn['content']}\n"
            
        elif turn['role'] == 'assistant':
            # This is an assistant turn, so it will be our 'label'.
            # The 'prompt' is everything that came before it.
            prompt_for_this_turn = current_history_str + "Assistant:"
            
            # --- Construct the label with the correct structure and order ---
            assistant_turn_label = ""
            if 'action' in turn:
                assistant_turn_label += f"Action: {turn['action']}\n"
            if 'thought' in turn:
                assistant_turn_label += f"Thought: {turn['thought']}\n"
            
            if 'tool_output' in turn:
                assistant_turn_label += f"Tool_Output: {turn['tool_output']}\n"
            if 'content' in turn:
                assistant_turn_label += f"Content: {turn['content']}\n"
            
            assistant_turn_label = assistant_turn_label.strip()
            # --- Label construction finished ---

            # Now, create the final training/validation item
            # The prompt content must include the persistent instructions
            final_prompt_content = f"{ecg_evaluation_prompt}\n{current_history_str}"
            
            if is_val:
                # For validation, we create a single user turn containing the history and instructions.
                chat_eval = [{"role": "user", "content": final_prompt_content}]
                template_chat = tokenizer_inf.apply_chat_template(chat_eval, tokenize=False, add_generation_prompt=True)
                training_examples.append({
                    "dialogue_idx": dialogue_idx,
                    "dial": template_chat,
                    "label": assistant_turn_label,
                    "obs": extract_ecg_observation(assistant_turn_label)
                })
            else: # For training
                # Create a user-assistant pair for the chat template
                chat = [{"role": "user", "content": final_prompt_content}, {"role": "assistant", "content": assistant_turn_label}]
                template_chat = tokenizer_inf.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)
                
                # Create a history prompt for loss masking (user turn + empty assistant prompt)
                history_chat = [{"role": "user", "content": final_prompt_content}, {"role": "assistant", "content": ""}]
                template_history = tokenizer_inf.apply_chat_template(history_chat, tokenize=False, add_generation_prompt=False)

                training_examples.append({
                    "dialogue_idx": dialogue_idx,
                    "history": template_history,
                    "dial": template_chat,
                    "label": assistant_turn_label
                })

            # CRUCIAL: After processing the assistant turn, add it to the history
            # for the *next* turn in the dialogue.
            current_history_str += f"Assistant:\n{assistant_turn_label}\n"

    return training_examples

def extract_ecg_observation(label):
    """Extract tool outputs from ECG dialogue format"""
    lines = label.split("\n")
    return_label = {}
    
    for idx, sentence in enumerate(lines):
        if "Action:" in sentence:
            action = sentence.replace("Action: ", "").strip()
            # Look for the next observation
            if idx + 1 < len(lines) and "Tool_Output:" in lines[idx + 1]:
                current_obs = lines[idx + 1].replace("Tool_Output: ", "")
                try:
                    # Try to parse as JSON if it's a dict/list
                    import ast
                    parsed_obs = ast.literal_eval(current_obs)
                    return_label[action] = parsed_obs
                except:
                    return_label[action] = current_obs
    
    return return_label

from transformers import StoppingCriteria, StoppingCriteriaList

class MultiTokenStoppingCriteria(StoppingCriteria):
    """
    Custom stopping criteria that stops generation when any of the specified 
    multi-token sequences are generated.
    """
    def __init__(self, tokenizer, stop_sequences):
        self.tokenizer = tokenizer
        self.stop_sequences = []
        
        for seq in stop_sequences:
            # Encode the sequence, making sure not to add special tokens
            # that might not be part of the sequence itself.
            tokens = tokenizer.encode(seq, add_special_tokens=False)
            self.stop_sequences.append({
                'tokens': torch.tensor(tokens),
                'length': len(tokens),
                'text': seq
            })
    
    def __call__(self, input_ids, scores, **kwargs):
        # Check each stop sequence
        for stop_seq in self.stop_sequences:
            seq_len = stop_seq['length']
            # Ensure the generated sequence is long enough to contain the stop sequence
            if input_ids.shape[1] >= seq_len:
                # Get the last `seq_len` tokens from the generated sequence
                last_tokens = input_ids[0, -seq_len:]
                # Compare with the stop sequence's tokens
                if torch.equal(last_tokens, stop_seq['tokens'].to(input_ids.device)):
                    return True # Stop generation
        return False # Continue generation

class StopOnToken(StoppingCriteria):
    def __init__(self, stop_token_id):
        self.stop_token_id = stop_token_id

    def __call__(self, input_ids, scores, **kwargs):
        return input_ids[0, -1] == self.stop_token_id

# =========================================================================================
# FINAL HELPER FUNCTION 1: The "Smart" Parser
# This is the most critical fix for the error you are seeing.
# =========================================================================================
def parse_chat_template(prompt_str, tokenizer, model_type="llama"):
    """
    Parses a string formatted by a chat template back into a list of dictionaries.
    This version is specifically enhanced to handle the validation case where the
    entire dialogue history is nested inside a single user turn's content,
    and correctly preserves the instructions as a system message.
    """
    history = []
    
    # Default LLaMA format
    pattern = re.compile(
        r"<\|start_header_id\|>(?P<role>user|assistant)<\|end_header_id\|>\n\n(?P<content>.*?)(?=<\|eot_id\|>)",
        re.DOTALL
    )
    
    prompt_str_no_bos = prompt_str.replace(tokenizer.bos_token, "")

    # Standard parse first
    for match in pattern.finditer(prompt_str_no_bos):
        role = match.group("role")

        history.append({"role": role, "content": match.group("content").strip()})

    # This handles both single-turn validation format and multi-turn formats
    for i, message in enumerate(history):
        if (message['role'] == 'user' and 
            "Dialogue History:" in message['content'] and
            "Instruction:" in message['content']):
            
            main_content = message['content']
            
            # Separate the instructions from the actual dialogue
            try:
                parts = main_content.split("Dialogue History:")
                instruction_part = parts[0].strip()
                dialogue_part = parts[1].strip()
            except IndexError:
                # Failsafe if the format is unexpected
                continue

            # Start the new history with the instructions as a system message
            new_history = [{"role": "system", "content": instruction_part}]
            
            # Parse the plain-text inner dialogue
            # This regex splits the text into turns, keeping the "User:" and "Assistant:" markers.
            turns = re.split(r'\n(?=User:|Assistant:)', dialogue_part)
            
            for turn_str in turns:
                turn_str = turn_str.strip()
                if not turn_str:
                    continue

                if turn_str.startswith("User:"):
                    role = "user"
                    content = turn_str.replace("User:", "").strip()
                elif turn_str.startswith("Assistant:"):
                    role = "assistant"
                    # Handle multi-line assistant content correctly
                    content = turn_str.replace("Assistant:", "").strip()
                else:
                    continue
                
                new_history.append({"role": role, "content": content})
            
            # If we successfully parsed the inner dialogue, return this new, correct history.
            if len(new_history) > 1: # Ensure we parsed something beyond just the system prompt
                return new_history
            break  # Only process the first matching message

    # Fallback for any other cases
    if not history:
        history.append({"role": "user", "content": prompt_str_no_bos})
    return history

# =========================================================================================
# FINAL HELPER FUNCTION 2: To Extract Next User Turn
# =========================================================================================
def clean_generation_artifacts(generated_text, model_type):
    """Clean model-specific generation artifacts from text"""
    # Clean LLaMA artifacts (existing logic)
    generated_text = generated_text.replace("<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n", "")
    generated_text = generated_text.replace("<|begin_of_text|>", "")
    generated_text = generated_text.replace("<|eot_id|>", "")
    generated_text = generated_text.replace("<|start_header_id|>assistant<|end_header_id|>\n\n", "")
    
    return generated_text.strip()

def get_stopping_criteria(tokenizer, model_type):
    """Get model-specific stopping criteria"""
    # LLaMA format
    end_of_turn_strings = ["\nUser:", "\n\nUser:", "user_bye", "system_bye", tokenizer.eos_token]
    action_stopping_strings = ["Tool_Output:"]
    
    end_criteria = StoppingCriteriaList([
        MultiTokenStoppingCriteria(tokenizer, end_of_turn_strings)
    ])
    action_criteria = StoppingCriteriaList([
        MultiTokenStoppingCriteria(tokenizer, action_stopping_strings)
    ])
    
    return end_criteria, action_criteria

def get_model_config(model_type):
    """Get model-specific generation configuration - optimized for memory usage"""

    # LLaMA default
    return {
        "max_new_tokens": 150,  # Reduced for memory efficiency
        "do_sample": False
    }

def get_lora_target_modules(model_type):
    """Get model-specific LoRA target modules"""
    # LLaMA default (works for most transformer architectures)
    return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

def extract_next_user_turn(dialogue_context, ground_truth_label, turn_count, original_dialogue=None):
    """
    Extracts the next user turn from the ground truth label string. This version is
    compatible with the final overall_generate function.
    Now also extracts the user action from the original dialogue structure.
    """
    # This helper reconstructs a plain context to find the right turn.
    # It's kept this way for compatibility with your data format.
    gt_lines = ground_truth_label.strip().split('\n')
    current_user_count = dialogue_context.count("User:")
    
    user_turn_index = 0
    for i, line in enumerate(gt_lines):
        if line.startswith("User:"):
            if user_turn_index == current_user_count:
                # Extract user action from original dialogue
                user_action = None
                if original_dialogue and isinstance(original_dialogue, list):
                    # Count user turns in original dialogue to find the matching turn
                    user_turns_in_original = [turn for turn in original_dialogue if turn.get('role') == 'user']
                    if current_user_count < len(user_turns_in_original):
                        user_action = user_turns_in_original[current_user_count].get('action')
                
                return {'content': line, 'action': user_action}
            user_turn_index += 1
    # pdb.set_trace()
    return None

# =========================================================================================
# FINAL `overall_generate` FUNCTION
# This version calls the correct helpers and handles all logic correctly.
# =========================================================================================
def overall_generate(current_dial, model, rank, tokenizer, tools=None, ecg_base_path="./preprocess_classification/ptbxl_10s", use_gt=False, tool_output_cache=None):
    """
    Robust generation function that correctly handles chat history and special tokens,
    and correctly separates tool calls and responses into two distinct assistant turns.
    """
    import re
    
    # Detect model type for model-specific handling
    try:
        if hasattr(tokenizer, 'name_or_path'):
            model_type = detect_model_type(tokenizer.name_or_path)
        elif hasattr(tokenizer, 'model_name'):
            model_type = detect_model_type(tokenizer.model_name)
        else:
            model_type = detect_model_type(str(tokenizer))
    except Exception as e:
        print(f"Warning: Could not detect model type, defaulting to llama: {e}")
        model_type = "llama"
    
    # --- Setup Model-Specific Stopping Criteria ---
    end_criteria, action_criteria = get_stopping_criteria(tokenizer, model_type)
    
    # Get model-specific generation config
    gen_config = get_model_config(model_type)

    # --- Initial Dialogue Setup ---
    initial_prompt_str = deepcopy(current_dial['input'][0])
    initial_obs = deepcopy(current_dial['obs'][0])
    label = current_dial['output'][0]
    dialogue_idx = current_dial['dialogue_idx'][0]
    ecg_files = current_dial['ecg_files'][0]
    original_dialogue = current_dial.get('original_dialogues', [None])[0]
    
    # Use persistent tool output cache if provided
    if tool_output_cache is not None and dialogue_idx in tool_output_cache:
        # Update initial_obs with cached real-time results
        initial_obs.update(tool_output_cache[dialogue_idx])
    elif tool_output_cache is not None:
        # Initialize cache for this dialogue
        tool_output_cache[dialogue_idx] = {}
    
    # ECG file path logic
    if isinstance(ecg_files, str):
        import ast
        try:
            ecg_files = ast.literal_eval(ecg_files)
        except (ValueError, SyntaxError):
            ecg_files = [ecg_files]
            
    ecg_path = None
    ecg_id = None
    if ecg_files and ecg_base_path:
        if isinstance(ecg_files, list) and ecg_files:
            ecg_filename = ecg_files[0]
            ecg_path = os.path.join(ecg_base_path, ecg_filename)
            # Extract ecg_id from filename (e.g., "HR06828.mat" -> "06828")
            import re
            match = re.search(r'HR(\d+)\.mat', ecg_filename)
            if match:
                ecg_id = match.group(1)
        else:
            ecg_filename = str(ecg_files)
            ecg_path = os.path.join(ecg_base_path, ecg_filename)
            # Extract ecg_id from filename
            import re
            match = re.search(r'HR(\d+)\.mat', ecg_filename)
            if match:
                ecg_id = match.group(1)
            
    # Ground truth turn extraction for `use_gt` mode
    gt_assistant_turns = []
    if use_gt:
        gt_lines = label.strip().split('\n')
        current_turn = []
        for line in gt_lines:
            if line.strip() and not line.startswith("User:"):
                current_turn.append(line)
            elif line.startswith("User:") and current_turn:
                gt_assistant_turns.append('\n'.join(current_turn))
                current_turn = []
        if current_turn:
            gt_assistant_turns.append('\n'.join(current_turn))

    # --- Core Logic: Initialize history using the SMART parser ---
    dialogue_history = parse_chat_template(initial_prompt_str, tokenizer, model_type)
    
    # Enrich initial dialogue history with user actions from original dialogue
    if original_dialogue and isinstance(original_dialogue, list):
        user_turns_in_original = [turn for turn in original_dialogue if turn.get('role') == 'user']
        user_turn_index = 0
        for turn in dialogue_history:
            if turn['role'] == 'user' and user_turn_index < len(user_turns_in_original):
                original_action = user_turns_in_original[user_turn_index].get('action')
                turn['action'] = original_action
                user_turn_index += 1
    
    # Replace pre-computed tool outputs with real-time ones if tools are available
    if tools and ecg_path and os.path.exists(ecg_path):
        # Track which tool outputs we've already computed to avoid duplicates
        computed_tools = set()
        
        for turn in dialogue_history:
            if turn['role'] == 'assistant' and 'Tool_Output:' in turn['content']:
                # Check if this turn contains a tool call that we can replace
                for tool_name in ['call_classification_tool', 'call_measurement_tool', 'call_explanation_tool']:
                    if f'Action: {tool_name}' in turn['content'] and tool_name not in computed_tools:
                        # Execute the real-time tool call
                        real_time_output = execute_tool_call(tool_name, tools, ecg_path, dialogue_history[:dialogue_history.index(turn)])
                        computed_tools.add(tool_name)
                        
                        # Replace the Tool_Output in the content with multiple patterns to handle different formats
                        content = turn['content']
                        
                        # Pattern 1: Tool_Output: [content] (list format)
                        pattern1 = r'Tool_Output: \[.*?\]'
                        if re.search(pattern1, content):
                            content = re.sub(pattern1, f'Tool_Output: {real_time_output}', content)
                        
                        # Pattern 2: Tool_Output: 'content' (string format)  
                        pattern2 = r'Tool_Output: \'.*?\''
                        if re.search(pattern2, content):
                            content = re.sub(pattern2, f'Tool_Output: {real_time_output}', content)
                        
                        # Pattern 3: Tool_Output: content (no quotes/brackets)
                        pattern3 = r'Tool_Output: .*?(?=\n(?:Action|Thought|Content|Assistant|User|$))'
                        if re.search(pattern3, content, re.DOTALL):
                            content = re.sub(pattern3, f'Tool_Output: {real_time_output}', content, flags=re.DOTALL)
                        
                        turn['content'] = content
                        
                        # Also replace in the initial_prompt_str that gets fed to the model
                        initial_prompt_str = re.sub(pattern1, f'Tool_Output: {real_time_output}', initial_prompt_str)
                        initial_prompt_str = re.sub(pattern2, f'Tool_Output: {real_time_output}', initial_prompt_str)
                        initial_prompt_str = re.sub(pattern3, f'Tool_Output: {real_time_output}', initial_prompt_str, flags=re.DOTALL)
                        
                        # Cache the result
                        if tool_output_cache is not None:
                            if tool_name not in tool_output_cache[dialogue_idx]:
                                tool_output_cache[dialogue_idx][tool_name] = []
                            tool_output_cache[dialogue_idx][tool_name].append(real_time_output)
                        break
    
    generated_assistant_turns = []
    turn_count = 0
    max_turns = 10 

    while turn_count < max_turns:
        # 1. Prepare Input by Applying Chat Template
        current_input_str = tokenizer.apply_chat_template(
            dialogue_history, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        input_ids = tokenizer(current_input_str, return_tensors="pt").input_ids.to(rank)
        attention_mask = tokenizer(current_input_str, return_tensors="pt").attention_mask.to(rank)
        
        # 2. Generate Assistant's Response
        # Use model.generate directly; it works for both DDP-wrapped and standalone models.
        generate_func = model.module.generate if isinstance(model, DDP) else model.generate
        outputs = generate_func(
            input_ids,
            stopping_criteria=action_criteria,
            attention_mask=attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            **gen_config
        )

        assistant_turn_content = tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=True)
        assistant_turn_content = clean_generation_artifacts(assistant_turn_content, model_type)
        
        # 3. Handle Tool Calls as a Distinct Turn
        if "Tool_Output:" in assistant_turn_content:
            action_match = re.search(r"Action:\s*([^\n]+)", assistant_turn_content)
            if action_match:
                action_text = action_match.group(1).strip()
                
                tool_output = "nan"
                if action_text in ['call_classification_tool', 'call_measurement_tool', 'call_explanation_tool']:
                    if tools and ecg_path and os.path.exists(ecg_path):
                        tool_output = execute_tool_call(action_text, tools, ecg_path, dialogue_history)
                        # Cache the real-time result for subsequent turns
                        if tool_output_cache is not None:
                            if action_text not in tool_output_cache[dialogue_idx]:
                                tool_output_cache[dialogue_idx][action_text] = []
                            tool_output_cache[dialogue_idx][action_text].append(tool_output)
                    elif action_text in initial_obs and initial_obs.get(action_text):
                        if isinstance(initial_obs[action_text], list) and initial_obs[action_text]:
                            tool_output = initial_obs[action_text].pop(0)
                        else:
                            tool_output = initial_obs[action_text]
                
                # Finalize and store the FIRST turn (the tool call).
                tool_call_turn_content = assistant_turn_content + f" {tool_output}"
                generated_assistant_turns.append(tool_call_turn_content)
                dialogue_history.append({"role": "assistant", "content": tool_call_turn_content})
                
                # Immediately continue to the next loop to generate the SECOND turn.
                turn_count += 1
                continue

        # 4. Update History for a normal turn (or the second part of a tool call)
        generated_assistant_turns.append(assistant_turn_content)
        
        if use_gt and turn_count < len(gt_assistant_turns):
            dialogue_history.append({"role": "assistant", "content": gt_assistant_turns[turn_count]})
        else:
            dialogue_history.append({"role": "assistant", "content": assistant_turn_content})
        
        if any(end_marker in assistant_turn_content.lower() for end_marker in ["system_bye", "goodbye"]):
            break
            
        # 5. Add Next User Turn
        plain_text_context = "\n".join(turn['content'] for turn in dialogue_history if turn['role'] == 'user')
        plain_text_context += "\n" + "\n".join(turn['content'] for turn in dialogue_history if turn['role'] == 'assistant')

        next_user_turn_data = extract_next_user_turn(plain_text_context, label, turn_count, original_dialogue)
        
        if next_user_turn_data:
            user_content = next_user_turn_data['content'].replace("User:", "").strip()
            user_action = next_user_turn_data.get('action')
            dialogue_history.append({"role": "user", "content": user_content, "action": user_action})
            if "user_bye" in user_content.lower():
                turn_count += 1
                continue
        else:
            break
        
        turn_count += 1
        
    # 6. Prepare Final Output
    final_dialogue_str = tokenizer.apply_chat_template(dialogue_history, tokenize=False, add_generation_prompt=False)
    
    # Convert dialogue_history to HuggingFace dataset format
    formatted_dialogue = []
    user_turn_counter = 0
    for turn in dialogue_history:
        if turn['role'] == 'system':
            continue  # Skip system messages in final output
        
        formatted_turn = {"role": turn['role']}
        
        if turn['role'] == 'user':
            # For user turns, use the action already stored in dialogue_history
            content = turn['content']
            user_action = turn.get('action')  # This was extracted by extract_next_user_turn
            
            formatted_turn["action"] = user_action
            formatted_turn["content"] = content
            user_turn_counter += 1
            
        elif turn['role'] == 'assistant':
            # Parse assistant content to extract structured fields
            content = turn['content']
            
            # Extract action
            action_match = re.search(r'Action:\s*([^\n]+)', content)
            if action_match:
                formatted_turn["action"] = action_match.group(1).strip()
            else:
                formatted_turn["action"] = "response"
            
            # Extract thought
            thought_match = re.search(r'Thought:\s*([^\n]+)', content)
            if thought_match:
                formatted_turn["thought"] = thought_match.group(1).strip()
            
            # Extract tool output
            tool_output_match = re.search(r'Tool_Output:\s*(.+?)(?=\n(?:Action|Thought|Content|$)|<\|eot_id\||$)', content, re.DOTALL)
            if tool_output_match:
                tool_output_str = tool_output_match.group(1).strip()
                try:
                    # Try to parse as JSON/list
                    import ast
                    formatted_turn["tool_output"] = ast.literal_eval(tool_output_str)
                except:
                    formatted_turn["tool_output"] = tool_output_str
            
            # Extract content (response text)
            content_match = re.search(r'Content:\s*(.+?)(?=\n(?:Action|Thought|Tool_Output|$)|<\|eot_id\||$)', content, re.DOTALL)
            if content_match:
                formatted_turn["content"] = content_match.group(1).strip()
        
        formatted_dialogue.append(formatted_turn)
    
    result = {
        "ecg_id": ecg_id,
        "dialogue": formatted_dialogue
    }
    
    return result

def validate_with_modes(rank,
                       model, 
                       val_dataset, 
                       world_size, 
                       tokenizer_inf,
                       file_name,
                       result_save_path,
                       tools=None,
                       validation_mode="without_gt",
                       ecg_base_path="./preprocess_classification/ptbxl_10s"):
    """
    Enhanced validation function that processes every turn but only saves the final
    result for each complete dialogue to a JSON Lines file.
    """
    model.eval()
    
    # Get unique dialogue IDs and split by dialogues, not individual turns
    dialogue_ids = list(set(sample['dialogue_idx'] for sample in val_dataset.data))
    dialogue_ids.sort()  # Ensure consistent ordering
    
    total_dialogues = len(dialogue_ids)
    dialogues_per_gpu = total_dialogues // world_size
    remainder = total_dialogues % world_size
    start_dialogue_idx = rank * dialogues_per_gpu + min(rank, remainder)
    end_dialogue_idx = start_dialogue_idx + dialogues_per_gpu + (1 if rank < remainder else 0)
    
    # Get dialogue IDs for this GPU
    gpu_dialogue_ids = set(dialogue_ids[start_dialogue_idx:end_dialogue_idx])
    
    # Filter samples to only include those from assigned dialogues
    gpu_sample_indices = [i for i, sample in enumerate(val_dataset.data) 
                         if sample['dialogue_idx'] in gpu_dialogue_ids]
    
    val_subset = Subset(val_dataset, gpu_sample_indices)
    val_loader = DataLoader(val_subset, batch_size=1, collate_fn=overall_collate_fn)

    print(f"Rank {rank} processing dialogues {start_dialogue_idx} to {end_dialogue_idx}")
    print(f"Rank {rank} processing {len(gpu_sample_indices)} samples from {len(gpu_dialogue_ids)} dialogues")
    
    modes = []
    if validation_mode == 'without_gt':
        modes = [("without_gt", False)]
    elif validation_mode == 'with_gt':
        modes = [("with_gt", True)]
    else:
        # It's good practice to handle unexpected values
        raise ValueError(f"Invalid validation_mode: '{validation_mode}'. Choose from 'without_gt', 'with_gt'.")
    
    for mode_name, use_gt in modes:
        print(f"\n{'='*50}")
        print(f"Running inference mode: {mode_name}")
        print(f"{ '='*50}\n")

        results_file_path = os.path.join(result_save_path, f"results_rank{rank}_{mode_name}.jsonl")
        with open(results_file_path, 'w') as f:
            pass # Clear file at start
        dist.barrier()

        successful_generations = 0
        failed_generations = 0
        last_processed_result = None
        current_dialogue_idx = -1
        # Initialize tool output cache for persistent storage across samples
        tool_output_cache = {}

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(val_loader, dynamic_ncols=False, ascii=True, desc=f"Rank {rank} - {mode_name}")):
                dialogue_idx = batch['dialogue_idx'][0]

                # If this is the first sample, initialize current_dialogue_idx
                if current_dialogue_idx == -1:
                    current_dialogue_idx = dialogue_idx

                # If the dialogue_idx changes, it means the previous dialogue is complete.
                # Write the last processed result of the completed dialogue to the file.
                if dialogue_idx != current_dialogue_idx and last_processed_result:
                    with open(results_file_path, 'a') as f:
                        f.write(json.dumps(last_processed_result) + '\n')
                    current_dialogue_idx = dialogue_idx
                    last_processed_result = None # Reset for the new dialogue
                    # Clear cache for the completed dialogue to save memory
                    if current_dialogue_idx in tool_output_cache:
                        del tool_output_cache[current_dialogue_idx]

                try:
                    result = overall_generate(
                        batch, model, rank, tokenizer_inf, tools, ecg_base_path, use_gt, tool_output_cache
                    )
                    result['batch_idx'] = batch_idx

                    # Remove the prompt from the output fields before saving
                    if 'input' in result and isinstance(result.get('input'), str):
                        result['input'] = result['input'].replace(ecg_evaluation_prompt, '').strip()
                    if 'full_dialogue' in result and isinstance(result.get('full_dialogue'), str):
                        result['full_dialogue'] = result['full_dialogue'].replace(ecg_evaluation_prompt, '').strip()

                    last_processed_result = result # Hold the latest result
                    successful_generations += 1

                except Exception as e:
                    # Extract ecg_id for error reporting
                    ecg_files = batch.get('ecg_files', [[]])[0]
                    ecg_id = None
                    if ecg_files:
                        ecg_filename = ecg_files[0] if isinstance(ecg_files, list) else str(ecg_files)
                        import re
                        match = re.search(r'HR(\d+)\.mat', ecg_filename)
                        if match:
                            ecg_id = match.group(1)
                    
                    print(f"\nError processing batch {batch_idx} (ECG ID: {ecg_id}): {str(e)}")
                    traceback.print_exc()
                    error_result = {
                        "ecg_id": ecg_id,
                        "error": str(e),
                        "mode": mode_name,
                        "batch_idx": batch_idx,
                        "traceback": traceback.format_exc()
                    }
                    # If an error occurs, write it immediately and reset.
                    with open(results_file_path, 'a') as f:
                        f.write(json.dumps(error_result) + '\n')
                    failed_generations += 1
                    last_processed_result = None # Don't carry over failed results
                    current_dialogue_idx = -1 # Reset to handle next dialogue correctly

            # After the loop, there will be one last result held in memory.
            # Write the final completed dialogue to the file.
            if last_processed_result:
                with open(results_file_path, 'a') as f:
                    f.write(json.dumps(last_processed_result) + '\n')

        dist.barrier()

        # Aggregate stats (counting dialogues, not turns)
        stats_tensor = torch.tensor([successful_generations, failed_generations], dtype=torch.float32).to(rank)
        dist.reduce(stats_tensor, dst=0, op=dist.ReduceOp.SUM)

        if rank == 0:
            total_successful = int(stats_tensor[0].item())
            total_failed = int(stats_tensor[1].item())
            total_dialogues = total_successful + total_failed

            print(f"\n--- {mode_name} Mode Summary ---")
            print(f"Final results saved to: {result_save_path}")
            print(f"Total dialogues processed: {total_dialogues}")
            print(f"Successful generations: {total_successful}")
            print(f"Failed generations: {total_failed}")
            print("------------------------------------")

    dist.barrier()
    print(f"\nRank {rank} completed all inference modes.")

class JSONStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.brace_count = 0
        self.in_json = False
        
    def __call__(self, input_ids, scores, **kwargs):
        # Decode the last few tokens to check for JSON completion
        last_tokens = input_ids[0, -10:] if input_ids.shape[1] >= 10 else input_ids[0]
        decoded_text = self.tokenizer.decode(last_tokens, skip_special_tokens=True)
        
        # Count braces to determine if JSON is complete
        for char in decoded_text:
            if char == '{':
                self.brace_count += 1
                self.in_json = True
            elif char == '}':
                self.brace_count -= 1
                
        # Stop if we've completed a JSON object (brace count returns to 0 after opening)
        return self.in_json and self.brace_count == 0

def ecg_generate(current_dial, model, rank, tokenizer):
    initial_prompt = deepcopy(current_dial['input'][0])
    initial_obs = deepcopy(current_dial['obs'][0])
    input_prompt = current_dial['input'][0]
    input_obs = current_dial['obs'][0]
    label = current_dial['output'][0]
    dialogue_idx = current_dial['dialogue_idx'][0]
    
    # Detect model type for model-specific handling
    try:
        if hasattr(tokenizer, 'name_or_path'):
            model_type = detect_model_type(tokenizer.name_or_path)
        elif hasattr(tokenizer, 'model_name'):
            model_type = detect_model_type(tokenizer.model_name)
        else:
            model_type = detect_model_type(str(tokenizer))
    except Exception as e:
        print(f"Warning: Could not detect model type, defaulting to llama: {e}")
        model_type = "llama"
    
    # Create stopping criteria for JSON completion
    json_stopping_criteria = StoppingCriteriaList([JSONStoppingCriteria(tokenizer)])
    
    cnt = 0
    while True:
        if cnt > 5:
            return {"dialogue_idx": dialogue_idx,
                    "input": initial_prompt,
                    "predict": None,
                    "full_context": input_prompt,
                    "label": label,
                    "obs": initial_obs}
        
        # Check if we've reached a response action (end condition)
        last_lines = input_prompt.strip().split('\n')[-3:]
        if any('"action": "response"' in line or '"action": "response_followup"' in line or 
               '"action": "response_fail"' in line or '"action": "system_bye"' in line 
               for line in last_lines):
            break
        if any("Content:" in line for line in last_lines):
            break
            
        input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids.to(rank)
        attention_mask = tokenizer(input_prompt, return_tensors="pt")["attention_mask"].to(rank)
        
        outputs = model.module.generate(
            input_ids,
            max_new_tokens=500,
            do_sample=True,
            temperature=0.1,
            attention_mask=attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            stopping_criteria=json_stopping_criteria
        )
        
        generated = tokenizer.decode(outputs[0])
        
        # Clean up model-specific chat template artifacts
        generated = clean_generation_artifacts(generated, model_type)
        
        # More aggressive JSON extraction - find the last complete JSON object
        import re
        import json
        
        # Look for JSON patterns and extract only valid JSON
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        json_matches = list(re.finditer(json_pattern, generated, re.DOTALL))
        
        if json_matches:
            # Take the last complete JSON match
            last_match = json_matches[-1]
            json_text = last_match.group()
            
            # Validate it's actually valid JSON
            try:
                json.loads(json_text)
                generated = generated[:last_match.end()]
            except json.JSONDecodeError:
                # If not valid JSON, trim at the last closing brace
                last_brace = generated.rfind('}')
                if last_brace != -1:
                    generated = generated[:last_brace + 1]
        else:
            # Fallback: trim at last closing brace
            last_brace = generated.rfind('}')
            if last_brace != -1:
                generated = generated[:last_brace + 1]
        
        # Extract action from JSON-like structure in ECG format
        action_match = re.search(r'"action":\s*"(call_[^"]+)"', generated)
        if action_match:
            action_text = action_match.group(1)
            
            # Only use observation if it exists
            if action_text in input_obs and input_obs[action_text]:
                if isinstance(input_obs[action_text], list) and len(input_obs[action_text]) > 0:
                    obs_output = input_obs[action_text].pop(-1)
                else:
                    obs_output = input_obs[action_text]
                
                # Insert or replace tool_output in the JSON structure
                if '"tool_output":' not in generated:
                    # Find the end of the JSON object and insert tool_output
                    brace_pattern = r'(\{[^}]*)\s*\}'
                    match = re.search(brace_pattern, generated, re.DOTALL)
                    if match:
                        json_content = match.group(1)
                        new_json = json_content + f',\n  "tool_output": {json.dumps(obs_output)}\n}}'
                        generated = re.sub(brace_pattern, new_json, generated, flags=re.DOTALL)
                else:
                    # Replace existing tool_output
                    tool_output_pattern = r'"tool_output":\s*"[^"]*"'
                    generated = re.sub(tool_output_pattern, f'"tool_output": {json.dumps(obs_output)}', generated)
        
        input_prompt = generated
        cnt += 1
    
    predict = "\n".join(generated.split("\n")[len(initial_prompt.strip().split("\n")):])
    return {
        "dialogue_idx": dialogue_idx,
        "input": initial_prompt,
        "predict": predict,
        "full_context": generated,
        "label": label,
        "obs": initial_obs
    }

################

def collate_function(batches,tokenizer):
    history_length = []
    for item in batches:
        history_length.append(len(tokenizer.encode(item['history'])))

    x_texts = [item['X'] for item in batches]
    x_encoded = tokenizer(x_texts, return_tensors='pt', padding='longest')
    return {"history":torch.tensor(history_length),"input_ids":x_encoded['input_ids'],"attention_mask":x_encoded['attention_mask']}

def normalize_json(data):
    def traverse_and_normalize(value):
        if isinstance(value, dict):
            return {traverse_and_normalize(k): traverse_and_normalize(v) for k, v in value.items()}
        elif isinstance(value, list):
            value = str(value).lower()
            return re.sub(r'[^a-z0-9]', '', value.lower())
        elif isinstance(value, str):
            return re.sub(r'[^a-z0-9]', '', value.lower())
        elif isinstance(value,int) or isinstance(value,float):
            value = str(value)
            return re.sub(r'[^a-z0-9]', '', value.lower())
        else:
            return value
    return traverse_and_normalize(data)

def extract_and_convert_dict(text):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        dict_str = match.group(0)  # Extract the matched string (the dictionary)
        dict_str = dict_str.replace("'", '"')
        try:
            extracted_dict = json.loads(dict_str)
            return extracted_dict
        except json.JSONDecodeError as e:
            try:
                extracted_dict=eval(dict_str)
                return extracted_dict
            except:
                raise ValueError
    else:
        print("No dictionary found in the text.")
        return None
    
##########################
# Setup function for DDP
def setup(rank, world_size):
    # dist.init_process_group(backend="nccl")
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29517'

    dist.init_process_group("nccl", rank=rank, world_size=world_size,timeout=datetime.timedelta(seconds=36000))
    torch.cuda.set_device(rank)

def cleanup():
    try:
        if dist.is_initialized():
            dist.destroy_process_group()
    except Exception as e:
        print(f"Warning: Error during cleanup: {e}")


##########################
# Check GPU availability
def check_gpu_availibility():
    if torch.cuda.is_available():
        num_cuda_devices = torch.cuda.device_count()
        print(f'Number of CUDA Device available: {num_cuda_devices}')
        for i in range(num_cuda_devices):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")
            print(f"Is available: {torch.cuda.is_available()}")
    else:
        print("No cuda devices available")

def split_data_among_gpus(data, world_size):
    chunk_size = len(data) // world_size
    chunks = [data[i * chunk_size: (i + 1) * chunk_size] for i in range(world_size - 1)]
    chunks.append(data[(world_size - 1) * chunk_size:])  # Handle the last chunk
    return chunks

#tool calling functions 
def initialize_tools(lead_config='single-lead-I'):
    """Initialize the actual ECG tools"""
    try:
        classifier_tool = ECGClassifierTool(
            simulated_lead_context=lead_config,
            device="cuda"
        )
        analysis_tool = ECGAnalysisTool(
            simulated_lead_context=lead_config, 
            device="cuda"
        )
        explain_tool = ECGExplainTool(
            simulated_lead_context=lead_config,
            device="cuda"
        )
        
        tools = {
            'call_classification_tool': classifier_tool,
            'call_measurement_tool': analysis_tool,
            'call_explanation_tool': explain_tool
        }
        
        print("Tools initialized successfully")
        return tools
    except Exception as e:
        print(f"Error initializing tools: {e}")
        return None


def execute_tool_call(action, tools, ecg_path, dialogue_history=None):
    """Execute the actual tool call, now with dialogue context for explanation"""
    
    try:
        tool_name = action
        
        if tool_name == "call_classification_tool":
            if tools.get("call_classification_tool"):
                result = tools["call_classification_tool"]._run(ecg_path)
                if isinstance(result, tuple) and len(result) >= 1:
                    classifications = result[0]
                    formatted_result = []
                    for condition, prob in classifications.items():
                        if prob > 0.5:
                            formatted_result.append(f"{condition} ({prob:.2%})")
                    return str(formatted_result)
                return str(result)
            
        elif tool_name == "call_measurement_tool":
            if tools.get("call_measurement_tool"):
                result = tools["call_measurement_tool"]._run(ecg_path)
                measurements = {
                    "heart_rate": result.get("Heart_Rate", np.nan),
                    "pr_interval": result.get("PR_Interval_ms", np.nan),
                    "qrs_duration": result.get("QRS_Duration_ms", np.nan),
                    "qtc_interval": result.get("QTc_ms", np.nan)
                }
                clean_measurements = {k: v for k, v in measurements.items() if not np.isnan(v)}
                return str(clean_measurements)
                
        elif tool_name == "call_explanation_tool":
            if tools.get("call_explanation_tool"):
                target_class = None
                
                # 1. Look back in the dialogue history for context
                if dialogue_history:
                    # Get the last user turn
                    last_user_turn = next((turn['content'] for turn in reversed(dialogue_history) if turn['role'] == 'user'), None)
                    
                    # Get the last classification result from the assistant
                    last_classification_output = None
                    for turn in reversed(dialogue_history):
                        if turn['role'] == 'assistant' and 'call_classification_tool' in turn['content']:
                            # Extract the tool output part
                            match = re.search(r"Tool_Output:\s*(\[.*?])", turn['content'])
                            if match:
                                try:
                                    # Safely evaluate the list string
                                    import ast
                                    output_list = ast.literal_eval(match.group(1))
                                    last_classification_output = [re.match(r"([^\(]+)", item).group(1).strip() for item in output_list]
                                except (ValueError, SyntaxError):
                                    pass # Could not parse the output
                            break
                    
                    # 2. Determine the target class
                    if last_user_turn and last_classification_output:
                        # Check if the user explicitly mentions a class from the last classification
                        for class_name in last_classification_output:
                            if class_name.lower() in last_user_turn.lower():
                                target_class = class_name
                                print(f"Found target class from user query: {target_class}")
                                break
                    
                    # 3. Fallback logic
                    if not target_class:
                        # If user didn't specify, or we couldn't find it, use the highest-prob class from the last output
                        if last_classification_output:
                            target_class = last_classification_output[0] # The first one is the highest probability
                            print(f"Using fallback (highest probability) class: {target_class}")
                        else:
                            # If there's no classification history, we can't proceed.
                            return "Explanation error: Could not determine which class to explain. No prior classification found."

                # 4. Execute the tool with the determined target class
                if target_class:
                    tool_input = {"ecg_path": ecg_path, "target_class": target_class}
                    result = tools["call_explanation_tool"].run(tool_input)
                    if isinstance(result, dict) and "error" in result:
                        return f"Explanation error: {result['error']}"
                    return str(result)
                else:
                    return "Explanation error: Failed to identify a target class for explanation."

        return f"Unknown tool: {tool_name}"
        
    except Exception as e:
        import traceback
        return f"Tool execution error: {str(e)}\n{traceback.format_exc()}"

def get_ecg_path_from_dialogue_idx(dialogue_idx, dataset):
    """Get ECG file path from dialogue index"""
    try:
        if dialogue_idx < len(dataset):
            sample = dataset[dialogue_idx]
            ecg_files = sample.get("ecg_files", [])
            if ecg_files:
                # Assuming ECG files are in a specific directory
                ecg_base_path = "./data/ecg_files"
                return os.path.join(ecg_base_path, ecg_files[0])
        return None
    except Exception as e:
        print(f"Error getting ECG path: {e}")
        return None

##########################
# Load data only on rank 0 and broadcast to other ranks
def load_data(rank, world_size, tokenizer, train_file, test_file, lead_config='single-lead-I'):
    if rank == 0:
        # ===================================================================
        # <<< MODIFICATION START >>>
        # By setting this environment variable, you are telling the Hugging Face
        # library to NOT attempt to connect to the Hub and to rely solely
        # on local files and the cache.
        # print("Forcing offline mode to use cached dataset...")
        # os.environ["HF_DATASETS_OFFLINE"] = "1"
        # <<< MODIFICATION END >>>
        # ===================================================================

        print("Train data loading....")
        from datasets import load_dataset
        dataset = load_dataset(f"gustmd0121/{lead_config}-ecg-mtd-dataset-cleaned-final")
        train_dialogue_list = dataset["train"]["dialogue"]
        val_dialogue_list = dataset["test"]["dialogue"]  # or validation split
        
        # Store the full dataset for ECG file access
        train_dataset = dataset["train"]
        val_dataset = dataset["test"]
        
        train_dst = []
        for idx, dialogue in enumerate(train_dialogue_list):
            ecg_files = train_dataset[idx].get("ecg_files", [])
            # Ensure dialogue is properly parsed if it's a string
            if isinstance(dialogue, str):
                try:
                    dialogue_parsed = json.loads(dialogue)
                except json.JSONDecodeError:
                    dialogue_parsed = dialogue
            else:
                dialogue_parsed = dialogue
            
            forms = ecg_form(idx, dialogue_parsed, tokenizer)
            # Add ECG file info and original dialogue to each form
            for form in forms:
                form['ecg_files'] = ecg_files
                form['original_dialogue'] = dialogue_parsed
            train_dst += forms
            
        val_dst = []
        for idx, dialogue in enumerate(val_dialogue_list):
            ecg_files = val_dataset[idx].get("ecg_files", [])
            # Ensure dialogue is properly parsed if it's a string
            if isinstance(dialogue, str):
                try:
                    dialogue_parsed = json.loads(dialogue)
                except json.JSONDecodeError:
                    dialogue_parsed = dialogue
            else:
                dialogue_parsed = dialogue
            
            forms = ecg_form(idx, dialogue_parsed, tokenizer, is_val=True)
            # Add ECG file info and original dialogue to each form
            for form in forms:
                form['ecg_files'] = ecg_files
                form['original_dialogue'] = dialogue_parsed
            val_dst += forms
    
        print(f"Length of train: {len(train_dst)}")
        print(f"Length of test: {len(val_dst)}")
        
        # Package dataset info for broadcasting
        data_package = {
            'train_dst': train_dst,
            'val_dst': val_dst,
            'train_dataset': train_dataset,
            'val_dataset': val_dataset
        }
    else:
        data_package = None
    
    # Broadcast the data from rank 0 to all other ranks
    data_package = [data_package]
    dist.broadcast_object_list(data_package, src=0)
    data_package = data_package[0]

    return data_package['train_dst'], data_package['val_dst'], data_package.get('train_dataset'), data_package.get('val_dataset')

##########################
# Dataset Class
class CustomDataset(Dataset):
    def __init__(self, full_data):
        self.data = []
        self.length = []
        for entity in tqdm(full_data):
            if len(entity['dial']) > 4096:
                continue
            new_dial = {"X": entity['dial'], "history": entity['history']}
            self.data.append(new_dial)
            self.length.append(len(new_dial['X']))
        # random.shuffle(self.data)
        print(f"Avg length: {sum(self.length)/len(self.length)}")
        print(f"MAx len: {max(self.length)}")
        print("Total used ratio:", len(self.data)/len(full_data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class OverallDataset(Dataset):
    def __init__(self, full_data):
        self.data = []
        for entity in tqdm(full_data):
            new_dial = {
                "X": entity['dial'], 
                "y": entity['label'],
                "obs": entity['obs'],
                "dialogue_idx": entity['dialogue_idx'],
                "ecg_files": entity.get('ecg_files', []),
                "original_dialogue": entity.get('original_dialogue', [])
            }
            self.data.append(new_dial)
        print(f"Length of testdata: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
    
##########################
# Data Loading and Distributed Training Setup
def train(rank, world_size, lead_config, device="cuda"):

    setup(rank, world_size)
    with open("./src/experiments/overall_train.yaml") as f:
        config = yaml.load(f,Loader=yaml.SafeLoader)
        
    train_file,test_file = config['train_file'],config['test_file']
    is_van = config['is_van']
    
    model_id = config['model_id']
    model_sum = model_id.split("/")[-1] if "/" in model_id else model_id.replace("./", "")
    
    # Detect model type for configuration
    model_type = detect_model_type(model_id)
    print(f"Detected model type: {model_type} for model: {model_id}")
    
    # Get model-specific LoRA target modules early for wandb config
    target_modules = get_lora_target_modules(model_type)
    
    # Load inference-specific settings
    run_real_time_tools = config.get('run_real_time_tools', True)
    validation_mode = config.get('validation_mode', 'without_gt')
    ecg_base_path = config.get('ecg_base_path', "./hschung/preprocess_classification/ptbxl_10s")
    
    # Initialize wandb only on rank 0
    if rank == 0:
        # Determine mode string for logging
        if is_van:
            # Use the new validation_mode string directly for logging
            mode_str = validation_mode
        else:
            mode_str = "training"
            
        # Create model name for wandb run name
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        wandb_run_name = f"{model_sum}_{current_time}"
        if is_van:
            wandb_run_name += "_inference"
        else:
            wandb_run_name += "_training"
            
        wandb.init(
            project="ECG_Agent",
            entity="gustmd0121",
            name=wandb_run_name,
            config={
                "model_id": model_id,
                "model_type": model_type,
                "is_van": is_van,
                "mode": mode_str,
                "run_real_time_tools": run_real_time_tools,
                "validation_mode": validation_mode,
                "ecg_base_path": ecg_base_path,
                "lead_config": lead_config,
                "epochs": 1,
                "learning_rate": 0.00001,
                "batch_size": 1,
                # LoRA configuration
                "lora_alpha": 16,
                "lora_dropout": 0.1,
                "lora_r": 8,
                "lora_bias": "none",
                "lora_task_type": "CAUSAL_LM",
                "lora_target_modules": target_modules,
                # Quantization configuration
                "load_in_4bit": True,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_compute_dtype": "torch.bfloat16",
                "bnb_4bit_use_double_quant": True,
            }
        )
    
    tools = None
    if rank == 0 and (is_van and run_real_time_tools):
        print("Initializing real-time tools for inference...")
        tools = initialize_tools(lead_config)
    elif rank == 0:
        print("Real-time tools disabled - using pre-computed observations")
    
    if not is_van: #is_van False means training 
        os.makedirs("./experiments/overall_results", exist_ok=True)
        if "td_ecg_llama_final" not in os.listdir("./experiments/overall_results"):
            os.mkdir("./experiments/overall_results/td_ecg_llama_final")
            result_save_path = "./experiments/overall_results/td_ecg_llama_final"
        else:
            result_save_path = "./experiments/overall_results/td_ecg_llama_final"
    else:
        if model_sum not in os.listdir("./experiments/overall_results"):
            os.mkdir(f"./experiments/overall_results/{model_sum}")
            result_save_path = f"./experiments/overall_results/{model_sum}"
        else:
            result_save_path = f"./experiments/overall_results/{model_sum}"

    # Log the result save path in wandb
    if rank == 0:
        wandb.config.update({"result_save_path": result_save_path}) 

    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        devices = [d for d in os.environ['CUDA_VISIBLE_DEVICES'].split(',') if d.strip()]
        world_size = len(devices) if devices else torch.cuda.device_count()
    else:
        world_size = torch.cuda.device_count()
        devices = [str(i) for i in range(world_size)]
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # tokenizer.add_special_tokens({"pad_token": "<pad>","unk_token":"<unk>"})
    tokenizer.pad_token_id = tokenizer.eos_token_id #add special pad token 

    tokenizer_inf = AutoTokenizer.from_pretrained(model_id,padding_side='left')
    # tokenizer_inf.add_special_tokens({"pad_token": "<pad>","unk_token":"<unk>"})
    ##
    tokenizer_inf.pad_token_id = tokenizer_inf.eos_token_id
    ##

    train_sft_form_list, val_sft_form_list, train_dataset, val_dataset = load_data(rank, world_size, tokenizer, train_file, test_file, lead_config)

    # Convert loaded data to CustomDataset
    dst_train_dataset = CustomDataset(train_sft_form_list)
    dst_metric_dataset = OverallDataset(val_sft_form_list)

    dst_metric_dataset = OverallDataset(val_sft_form_list)

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    lora_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=8,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules
    )
    
    # --- Corrected Model Loading ---
    if is_van:
        # INFERENCE MODE: Manually load base model, resize, then attach adapter
        from peft import PeftModel, PeftConfig
        
        # 1. Load the config from the adapter directory to find the base model name.
        peft_config = PeftConfig.from_pretrained(model_id)

        # 2. Load the base model itself.
        model_kwargs = {
            "quantization_config": quant_config,
            "device_map": {"": rank}
        }
            
        model = AutoModelForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path,
            **model_kwargs
        )

        # 3. Check and handle vocabulary size compatibility
        base_vocab_size = model.get_input_embeddings().weight.shape[0]
        fine_tuned_vocab_size = len(tokenizer)
        
        print(f"Base model vocab size: {base_vocab_size}")
        print(f"Fine-tuned tokenizer vocab size: {fine_tuned_vocab_size}")
        
        # Always resize to match the fine-tuned tokenizer regardless of base model size
        # This ensures compatibility with the PEFT adapter which was trained with this vocab size
        if base_vocab_size != fine_tuned_vocab_size:
            print(f"Vocabulary size mismatch detected. Resizing embeddings from {base_vocab_size} to {fine_tuned_vocab_size}")
        else:
            print("Vocabulary sizes match, but ensuring consistency for PEFT loading")
        
        # Resize embeddings to match fine-tuned tokenizer
        model.resize_token_embeddings(fine_tuned_vocab_size)
        print(f"Resized model embeddings to {model.get_input_embeddings().weight.shape[0]} tokens")

        # 4. Now, load the adapter weights onto the correctly-sized base model.
        try:
            model = PeftModel.from_pretrained(model, model_id)
            print("Successfully loaded PEFT adapter")
        except Exception as e:
            print(f"Error loading PEFT adapter: {e}")
            print("Attempting alternative loading strategy...")
            
            # Alternative: Clear CUDA cache and try again with memory management
            try:
                import gc
                
                # Clear CUDA cache and Python garbage collection
                torch.cuda.empty_cache()
                gc.collect()
                
                print(f"Attempting to load PEFT adapter again after memory cleanup...")
                model = PeftModel.from_pretrained(model, model_id)
                print("Successfully loaded PEFT adapter after memory cleanup")
            except Exception as e2:
                print(f"Alternative loading also failed: {e2}")
                print("This indicates a fundamental compatibility issue between base model and adapter")
                
                # Final attempt: Try to provide more specific error information
                print(f"Adapter config base model: {peft_config.base_model_name_or_path}")
                print(f"Current model type: {type(model)}")
                print(f"Current embedding size: {model.get_input_embeddings().weight.shape}")
                print(f"Model config vocab size: {getattr(model.config, 'vocab_size', 'Not found')}")
                
                raise e2
        
        model.config.use_cache = False
        replace_with_flash_attention(model)

    else:
        # TRAINING MODE: Load base model and prepare it for new training with PEFT
        model_kwargs = {
            "quantization_config": quant_config,
            "ignore_mismatched_sizes": True  # Important for resuming training
        }
        
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            **model_kwargs
        )
        model.resize_token_embeddings(len(tokenizer))
        model.config.use_cache = False
        replace_with_flash_attention(model)
        model = get_peft_model(model, lora_config)

    model.config.pad_token_id = tokenizer.pad_token_id

    # Move the model to the correct device
    model = model.to(rank)

    # For training, wrap with DDP and create optimizer. For inference, the model is used directly.
    if not is_van:
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)
        model.module.gradient_checkpointing_enable()
        optim = bitsandbytes.optim.adam.Adam8bit(model.parameters(), lr=0.00001, betas=(0.9, 0.995))
        scheduler = StepLR(optim, step_size=1, gamma=0.1)
    else:
        optim = None
        scheduler = None

    # Use DistributedSampler for both train and validation datasets
    train_sampler = DistributedSampler(dst_train_dataset, num_replicas=world_size, rank=rank)

    # Create DataLoaders using DistributedSampler
    n_batch = len(devices)
    train_loader = DataLoader(dst_train_dataset, batch_size=1, sampler=train_sampler, collate_fn=lambda batches: collate_function(batches, tokenizer))
    
    # Debug: Print data distribution info
    print(f"GPU {rank}: Total dataset size: {len(dst_train_dataset)}")
    print(f"GPU {rank}: DataLoader length: {len(train_loader)}")
    print(f"GPU {rank}: Sampler num_samples: {train_sampler.num_samples}")
    
    epochs = 1
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        print(f"Training on GPU {rank}... Epoch {epoch + 1}/{epochs}")

        # Set the epoch for the sampler to shuffle the data differently at each epoch
        train_sampler.set_epoch(epoch)
        # pdb.set_trace()
        if not is_van:
            # Add more detailed progress tracking per GPU
            if rank == 0:
                pbar = tqdm(train_loader, dynamic_ncols=False, ascii=True, desc=f"GPU {rank}")
            else:
                pbar = tqdm(train_loader, dynamic_ncols=False, ascii=True, desc=f"GPU {rank}", position=rank)
            
            batch_count = 0
            for entity in pbar:
                batch_count += 1
                if batch_count % 100 == 0:  # Print every 100 batches
                    print(f"GPU {rank}: Processing batch {batch_count}")
                
                X = entity['input_ids'].to(rank)
                a = entity['attention_mask'].to(rank)
                history_length = entity['history']
                # print("Hist:",tokenizer.decode(entity['input_ids'].tolist()[0][:entity['history'].item()]))
                # print("--------------------")
                # print("Dial:",tokenizer.decode(entity['input_ids'].tolist()[0]))
                optim.zero_grad()
                X_masked = X.clone()
                rows = torch.arange(X.size(1)).unsqueeze(0)  # (1, seq_length)
                mask = rows < history_length.unsqueeze(1)  # (batch_size, seq_length)
                X_masked[mask] = -100  # Apply -100 to positions to be ignored in loss calculation
                outputs = model(X, attention_mask=a, labels=X_masked)
                loss = outputs.loss
                # Fix for PEFT/transformers compatibility
                from torch.autograd import Variable
                loss = Variable(loss, requires_grad=True)
                # print(f"Current loss:{loss.detach().item()}")
                train_loss += loss.detach().item()
                loss.backward()
                optim.step()
            
            print(f"GPU {rank}: Completed epoch with {batch_count} batches")

            # Reduce train_loss across all GPUs
            train_loss_tensor = torch.tensor(train_loss).to(rank)
            dist.reduce(train_loss_tensor, dst=0, op=dist.ReduceOp.SUM)

            # Only rank 0 prints the average train loss
            if rank == 0:
                avg_train_loss = train_loss_tensor.item() / len(train_loader) / world_size
                print(f"Train Loss (Average) on all GPUs: {avg_train_loss}")
                
                # Log training loss to wandb
                wandb.log({
                    "epoch": epoch + 1,
                    "train_loss": avg_train_loss,
                })

            scheduler.step()
        
        file_name = f"overall_result_{test_file}_{model_sum}_with_instruction"
        if is_van:
            file_name+="_van"
        file_name+=".json"

        if rank == 0:
            # Create model name with model_id and current date/time
            current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = f"model_epoch_overall_{model_sum}_{lead_config}_{current_time}"
            if is_van:
                model_name+="_van"
            
            if not is_van:
                # Save .pt format (existing)
                torch.save(model.module.state_dict(), model_name + ".pt")
                
                # Save HuggingFace format (new)
                model.module.save_pretrained(model_name + "_hf")
                tokenizer.save_pretrained(model_name + "_hf")
                
                print(f"Model saved in both formats:")
                print(f"  PyTorch: {model_name}.pt")
                print(f"  HuggingFace: {model_name}_hf/")
                
                # Log model save information to wandb
                wandb.config.update({
                    "model_save_path_pt": f"{model_name}.pt",
                    "model_save_path_hf": f"{model_name}_hf/",
                })
        
        # Run validation
        if is_van:
            # Inference mode: run both with_gt and without_gt
            print("Running inference with both modes (with_gt and without_gt)...")
            validate_with_modes(rank, model, dst_metric_dataset, world_size, tokenizer_inf, file_name, result_save_path, tools, validation_mode, ecg_base_path)
        else:
            # Training mode: just run without_gt for validation
            print("Running validation in training mode...")
            validate_with_modes(rank, model, dst_metric_dataset, world_size, tokenizer_inf, file_name, result_save_path, tools, False, ecg_base_path)

    # Close wandb run
    if rank == 0:
        wandb.finish()
        
    cleanup()


##########################
def main():
    # Set up the argument parser
    parser = argparse.ArgumentParser(description="Train or run inference for ECG-Agent.")
    parser.add_argument(
        '--lead_config', 
        type=str, 
        default='12-lead', 
        choices=['single-lead-I', 'single-lead-II', '12-lead'],
        help='The ECG lead configuration to use for dataset loading and tool simulation.'
    )
    args = parser.parse_args()

    # Use the number of GPUs specified by CUDA_VISIBLE_DEVICES
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        visible_devices = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
        world_size = len([d for d in visible_devices if d.strip()])
    else:
        world_size = torch.cuda.device_count()
    
    print(f"Using {world_size} GPUs with lead configuration: {args.lead_config}")
    
    try:
        # Pass the lead_config argument to the train function
        mp.spawn(train, args=(world_size, args.lead_config), nprocs=world_size, join=True)
    except Exception as e:
        error_message = traceback.format_exc()
        print(f"Error: {error_message}")  # Print to console
        with open("error_train_overall.log", 'w') as error_file:
            error_file.write(f"An error occurred during training:\n{error_message}")


if __name__ == "__main__":
    # check_gpu_availibility()
    # set_seed(2)
    # main()
    # check_gpu_availibility() # Optional for debugging
    set_seed(2)

    print("--- RUNNING IN MULTI-PROCESS TRAIN MODE ---")
    main()
