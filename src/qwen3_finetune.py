#!/usr/bin/env python3
"""
Unsloth Fine-tuning Script for ECG Dialogue Dataset
MODIFIED to use a turn-by-turn training approach for better inference performance.
REVISED for unsloth/Phi-4 based on the official Unsloth Colab notebook.
"""

import os
import json
import torch
import argparse
from datasets import load_dataset, Dataset
from transformers import TrainingArguments, DataCollatorForSeq2Seq, EarlyStoppingCallback
from trl import SFTTrainer
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only
import wandb
from prompts import ECG_EVALUATION_PROMPT


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

    print(f"Training samples: {len(train_dataset)}")
    print(f"Created validation samples: {len(validation_dataset)}")
    print(f"Created test samples: {len(test_dataset)}")
    
    # Return all three datasets
    return train_dataset, validation_dataset, test_dataset


def setup_model_and_tokenizer(model_name="unsloth/Phi-4-unsloth-bnb-4bit", max_seq_length=4096):
    """Initialize model and tokenizer with Unsloth for Phi-4."""
    print(f"Loading model: {model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )

    print("Applying Qwen-3 chat template to tokenizer.")
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="qwen-3",
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        # --- REVISED: Using target_modules from the official Phi-4 notebook ---
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    return model, tokenizer


def setup_training_args(output_dir="./ecg-dialogue-finetuned-phi-4"):
    """Configure training arguments."""
    return TrainingArguments(
        per_device_train_batch_size=8,
        gradient_accumulation_steps=16,
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
        run_name="ecg-dialogue-finetune-turn-by-turn",
    )


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

    print("Saving final LoRA adapters...")
    trainer.save_model()

    # merged_dir = f"{output_dir}-merged"
    # print(f"Saving merged 16-bit model to {merged_dir}...")
    # model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_16bit")
    
    print("Training completed! 🚀")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Qwen3 on ECG dialogue dataset using Unsloth")
    parser.add_argument("--model", type=str, default="unsloth/Qwen3-32B-unsloth-bnb-4bit", help="Qwen3 model to fine-tune.")
    parser.add_argument("--max-seq-length", type=int, default=4096, help="Maximum sequence length")
    parser.add_argument("--output-dir", type=str, default="./ecg-dialogue-finetuned-turn-by-turn-32b-Instruct-0817-4bit", help="Output directory for the fine-tuned model")
    
    args = parser.parse_args()
    main(args.model, args.max_seq_length, args.output_dir)