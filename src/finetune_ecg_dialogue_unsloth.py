#!/usr/bin/env python3
"""
Unsloth Fine-tuning Script for ECG Dialogue Dataset
MODIFIED to use a turn-by-turn training approach for better inference performance.
"""

import os
os.environ['UNSLOTH_RETURN_LOGITS'] = '1'
import json
import torch
import argparse
from datasets import load_dataset, Dataset
from transformers import TrainingArguments, EarlyStoppingCallback
from trl import SFTTrainer, SFTConfig
from unsloth import FastLanguageModel
import wandb

# --- COPIED FROM SCRIPT 1: The instruction prompt is now part of the training data ---
# This prompt will be treated as the "system" message for every training example.
ECG_EVALUATION_PROMPT = """
Instruction:
You will be provided with part of a dialogue between the user and the system. The dialogue is conducted with a ECG wearable device user and the system, therefore the system should provide direct answers to the user's inquries without acting like a medical professional.
In this dialogue, the user is inquiring about an ECG reading, and the system will either respond directly or make appropriate tool calls to retrieve the necessary information to respond.
Your task is to generate appropriate thoughts, actions, and responses based on the dialogue history and the user's most recent utterance.

<Action list>
- call_classification_tool: Use this to identify arrhythmias, abnormalities, and findings from the provided ECG data
- call_measurement_tool: Use this to measure and output the heart rate, PR interval, QRS duration, and QTc interval.
- response: Provide comprehensive answer using results from tool outputs, combining technical findings with clinical interpretation
- response_fail: Indicate that the requested analysis cannot be performed due to tool limitations or because the question is not within the scope of the System and requires medical professional consultation.
- response_followup: Responds to the user's followup questions by providing additional information, clarification, or related insights about previous discussed ECG findings without requiring new tool calls.
- system_bye: Acknowledges the user's gratitude end the conversation politely

<General Rules>
1. Each generated dialogue (except for system_bye) is during the conversation with the user, therefore the responses should only address the user's last utterance and should not be too long.
2. Generated dialogue should be like a natural conversation between a user and ECG wearable assistant.

Based on the rules above, you will get an input as below:
Input:
<Dialogue history>
User: <User's last utterance> or Assistant: <Assistant's last Tool_Output>

And here is the format for what you should return in two different cases:

Case1. When a tool must be called based on the user's inquiry:
Action: <Assistant's chosen action, one of the tools>
Thought: <Assistant's reasoning process>
Tool_Output: <This will be provided externally.>


Case2. When providing a response based on previous tool or any other action not requiring a tool call:
Action: <Assistant's chosen action>
Thought: <Assistant's reasoning process>
Content: <Assistant's response>

Now, generate appropriate reasoning trace and responding message to user. Only generate in the proper format above, without indicating the chosen case.
"""

def load_and_preprocess_dataset(tokenizer):
    """
    Load and preprocess the ECG dialogue dataset using a TURN-BY-TURN approach.
    This function breaks each dialogue into multiple training examples.
    """
    print("Loading ECG dialogue dataset...")
    original_dataset = load_dataset("gustmd0121/12-lead-ecg-mtd-dataset")
    
    def create_turn_by_turn_examples(dialogues):
        all_examples = []
        total_dialogues = 0
        total_turns = 0
        skipped_turns = 0
        skipped_dialogues = 0
        
        for dialogue_json in dialogues:
            total_dialogues += 1
            dialogue = json.loads(dialogue_json)
            messages = [{"role": "system", "content": ECG_EVALUATION_PROMPT}]
            dialogue_has_valid_turns = False
            
            for turn in dialogue:
                total_turns += 1
                
                # Skip empty turns or turns missing 'role' key
                if not turn or 'role' not in turn:
                    skipped_turns += 1
                    print(f"Skipping problematic turn in dialogue {total_dialogues-1}: {turn}")
                    continue
                
                # Skip user turns missing content
                if turn['role'] == 'user' and 'content' not in turn:
                    skipped_turns += 1
                    print(f"Skipping user turn missing content in dialogue {total_dialogues-1}: {turn}")
                    continue
                
                # Skip assistant turns that have neither content nor tool_output
                if turn['role'] == 'assistant' and 'content' not in turn and 'tool_output' not in turn:
                    skipped_turns += 1
                    print(f"Skipping assistant turn missing content/tool_output in dialogue {total_dialogues-1}: {turn}")
                    continue
                
                dialogue_has_valid_turns = True
                
                if turn['role'] == 'user':
                    messages.append({"role": "user", "content": turn['content']})
                elif turn['role'] == 'assistant':
                    assistant_full_content = ""
                    if 'action' in turn:
                        assistant_full_content += f"Action: {turn['action']}\n"
                    if 'thought' in turn:
                        assistant_full_content += f"Thought: {turn['thought']}\n"
                    if 'tool_output' in turn:
                        assistant_full_content += f"Tool_Output: {turn['tool_output']}"
                    elif 'content' in turn:
                        assistant_full_content += f"Content: {turn['content']}"
                    
                    # Only add if we have some content
                    if assistant_full_content.strip():
                        messages.append({"role": "assistant", "content": assistant_full_content.strip()})
                        all_examples.append({"text": tokenizer.apply_chat_template(messages, tokenize=False)})
            
            if not dialogue_has_valid_turns:
                skipped_dialogues += 1
        
        print(f"  Total dialogues processed: {total_dialogues}")
        print(f"  Total turns processed: {total_turns}")
        print(f"  Skipped turns (empty/missing role): {skipped_turns}")
        print(f"  Skipped dialogues (no valid turns): {skipped_dialogues}")
        print(f"  Valid training examples created: {len(all_examples)}")
        
        return all_examples

    print("Processing training data into turn-by-turn examples...")
    train_examples = create_turn_by_turn_examples(original_dataset['train']['dialogue'])

    print("Processing validation data into turn-by-turn examples...")
    validation_examples = create_turn_by_turn_examples(original_dataset['validation']['dialogue'])
    
    print("Processing test data into turn-by-turn examples...")
    test_examples = create_turn_by_turn_examples(original_dataset['test']['dialogue'])

    # Convert lists to Dataset objects
    train_dataset = Dataset.from_dict({"text": [e["text"] for e in train_examples]})
    validation_dataset = Dataset.from_dict({"text": [e["text"] for e in validation_examples]})
    test_dataset = Dataset.from_dict({"text": [e["text"] for e in test_examples]})

    # --- CHANGE THIS SECTION ---
    # Split the full training dataset into a reproducible 8:1 train/validation split
    # An 8:1 ratio means the validation set is 1/9 of the total.

    print(f"Training samples: {len(train_dataset)}")
    print(f"Created validation samples: {len(validation_dataset)}")
    print(f"Created test samples: {len(test_dataset)}")
    
    # Return all three datasets
    return train_dataset, validation_dataset, test_dataset


def setup_model_and_tokenizer(model_name="unsloth/llama-3-8b-Instruct-bnb-4bit", max_seq_length=4096):
    """Initialize model and tokenizer with Unsloth"""
    print(f"Loading model: {model_name}")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=False,
    )
    
    FastLanguageModel.for_inference(model)
    
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    
    return model, tokenizer


def setup_training_args(output_dir="./ecg-dialogue-finetuned"):
    """Configure training arguments"""
    return TrainingArguments(
        per_device_train_batch_size=16,
        gradient_accumulation_steps=8,
        warmup_steps=10,
        num_train_epochs=3,  # Increase epochs since we'll stop early
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=42,
        output_dir=output_dir,
        eval_strategy="steps",      # Check validation loss during training
        eval_steps=1000,                    # How often to check (e.g., every 20 steps)
        save_strategy="steps",            # Save strategy should match evaluation strategy
        save_steps=1000,
        load_best_model_at_end=True,      # Load the best model when training ends
        metric_for_best_model="loss",     # Use validation loss to determine the best model
        save_total_limit=2,               # Save the best and the latest checkpoints
        report_to="wandb" if 'WANDB_API_KEY' in os.environ else None,
        run_name="ecg-dialogue-finetune-turn-by-turn")


def main(model_name=None, max_seq_length=4096, output_dir=None):
    """Main training function"""
    os.environ['UNSLOTH_RETURN_LOGITS'] = '1'
    
    if model_name is None: model_name = "unsloth/Llama-3.2-1B-Instruct"
    if output_dir is None: output_dir = "./ecg-dialogue-finetuned-turn-by-turn-1b-Instruct-0811"
    
    if 'WANDB_API_KEY' in os.environ:
        model_short_name = model_name.split('/')[-1]
        wandb.init(project="ecg-dialogue-finetune", name=f"unsloth-{model_short_name}-turn-by-turn")
    
    model, tokenizer = setup_model_and_tokenizer(model_name, max_seq_length)
    
    # --- CHANGE THIS LINE ---
    # Capture all three datasets returned by the function
    train_dataset, validation_dataset, test_dataset = load_and_preprocess_dataset(tokenizer)
    
    training_args = setup_training_args(output_dir)
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset, # Pass the validation set here
        args=training_args,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)], # Add callback
    )
    
    print("Starting training...")
    trainer.train()
    
    print("Saving final LoRA adapters (best model based on validation loss)...")
    trainer.save_model()
    
    print("Starting training...")
    trainer.train()
    
    print("Saving final LoRA adapters...")
    trainer.save_model()
    
    # merged_dir = f"{output_dir}-merged"
    # print(f"Saving merged 16-bit model to {merged_dir}...")
    # model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_16bit")
    
    print("Training completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune models on ECG dialogue dataset using Unsloth")
    parser.add_argument("--model", type=str, default="unsloth/Llama-3.2-3B-Instruct", help="Model to fine-tune.")
    parser.add_argument("--max-seq-length", type=int, default=4096, help="Maximum sequence length")
    parser.add_argument("--output-dir", type=str, default="./ecg-dialogue-finetune/Llama-3.2-3B-Instruct", help="Output directory for the fine-tuned model")

    args = parser.parse_args()
    main(args.model, args.max_seq_length, args.output_dir)