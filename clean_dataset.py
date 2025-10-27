# remove_known_bad_samples_and_push.py
import json
from datasets import load_dataset, DatasetDict
from huggingface_hub import login
import os

def is_dialogue_valid(dialogue_str: str) -> bool:
    """
    Validates a single dialogue to ensure all its turns adhere to the required format.

    A dialogue is considered VALID if and only if:
    1. It's a valid JSON string representing a list of turns.
    2. Every turn is a dictionary with a 'role'.
    3. User turns have non-empty 'role', 'action', and 'content' keys.
    4. Assistant turns have 'role', 'action', 'thought', and EITHER 'content' OR 'tool_output'.
    """
    try:
        dialogue = json.loads(dialogue_str)
    except (json.JSONDecodeError, TypeError):
        # The dialogue string is not valid JSON.
        return False

    if not isinstance(dialogue, list) or not dialogue:
        # The dialogue must be a non-empty list of turns.
        return False

    for turn in dialogue:
        if not isinstance(turn, dict) or 'role' not in turn:
            # Each turn must be a dictionary with a 'role'.
            return False

        keys = set(turn.keys())
        role = turn['role']

        if role == 'user':
            # A user turn must have 'role', 'action', and a non-empty 'content'.
            required_keys = {'role', 'action', 'content'}
            if not required_keys.issubset(keys) or not turn.get('content'):
                return False
        elif role == 'assistant':
            # An assistant turn must have base keys plus one exclusive key.
            required_base_keys = {'role', 'action', 'thought'}
            if not required_base_keys.issubset(keys):
                return False
            
            # It must have 'content' or 'tool_output', but not both (XOR check).
            has_content = 'content' in keys
            has_tool_output = 'tool_output' in keys
            if not (has_content ^ has_tool_output):
                return False
        else:
            # Any role other than 'user' or 'assistant' is invalid.
            return False
            
    # If all turns in the dialogue are valid, the dialogue is valid.
    return True


def main():
    """
    Loads the dataset, filters out invalid dialogues from all splits,
    and pushes the cleaned dataset to the Hugging Face Hub.
    """
    # 🔐 Authenticate (use HF_TOKEN env var or login interactively)
    token = os.getenv("HF_TOKEN")
    if token:
        login(token=token)
    else:
        login()

    # 1️⃣ Load the original dataset
    original_repo = "gustmd0121/single-lead-II-ecg-mtd-dataset"
    ds = load_dataset(original_repo)
    print(f"--- Loading dataset from: {original_repo} ---")
    print("Original dataset sizes:")
    print({split: len(subset) for split, subset in ds.items()})

    # 2️⃣ Filter each split in the dataset using the validation function
    print("\n--- Applying validation filter to all splits... ---")
    cleaned_ds = ds.filter(lambda example: is_dialogue_valid(example['dialogue']))
    
    print("\nCleaned dataset sizes:")
    print({split: len(subset) for split, subset in cleaned_ds.items()})

    # 3️⃣ Push the fully cleaned dataset to a new repository on the Hub
    repo_id = "gustmd0121/single-lead-II-ecg-mtd-dataset-cleaned-final"
    commit_msg = "Filter out samples with invalid turn structures from all splits"
    
    print(f"\n--- Pushing cleaned dataset to: {repo_id} ---")
    cleaned_ds.push_to_hub(
        repo_id=repo_id,
        private=False, # Set to True if you want the repo to be private
        commit_message=commit_msg
    )

    print(f"\n✅ Cleaned dataset pushed successfully!")
    print(f"Repo URL: https://huggingface.co/datasets/{repo_id}")

if __name__ == "__main__":
    main()