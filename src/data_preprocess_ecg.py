import json
from datasets import load_dataset, Dataset, DatasetDict
from huggingface_hub import HfApi
import os

def check_missing_content(dialogue_idx, dialogue_list):
    """
    Check if a dialogue has missing content based on the same logic as ecg_form
    Returns True if content is missing, False otherwise
    """
    # Parse dialogue_list if it's a string
    if isinstance(dialogue_list, str):
        try:
            dialogue_list = json.loads(dialogue_list)
        except json.JSONDecodeError as e:
            print(f"Error parsing dialogue_list for idx {dialogue_idx}: {e}")
            return True  # Consider malformed JSON as missing content
    
    for turn in dialogue_list:
        if turn['role'] == 'user':
            if 'content' not in turn or 'action' not in turn:
                return True
        elif turn['role'] == 'assistant':
            if 'action' not in turn:
                return True
            if turn['action'] in ['response', 'response_followup', 'response_fail', 'system_bye']:
                if 'content' not in turn:
                    return True
    
    return False

def process_single_split(dataset, split_name):
    """
    Process a single split and return cleaned dataset and removed indices
    """
    print(f"\n=== Processing {split_name} split ===")
    print(f"Original {split_name} size: {len(dataset)}")
    
    # Track indices to remove
    indices_to_remove = []
    missing_content_indices = []
    
    # Check each entry for missing content
    for idx, example in enumerate(dataset):
        dialogue_idx = example.get('dialogue_idx', idx)  # Use idx as fallback
        dialogue_list = example.get('dialogue_list', example.get('dialogue', []))
        
        if check_missing_content(dialogue_idx, dialogue_list):
            indices_to_remove.append(idx)
            missing_content_indices.append(dialogue_idx)
            print(f"MISSING CONTENT - ECG_IDX: {dialogue_idx} (dataset index: {idx}) [{split_name}]")
    
    print(f"\nFound {len(indices_to_remove)} entries with missing content in {split_name}")
    print(f"ECG indices with missing content in {split_name}: {missing_content_indices}")
    
    if len(indices_to_remove) == 0:
        print(f"No entries to remove from {split_name}!")
        return dataset, missing_content_indices
    
    # Create filtered dataset by selecting all indices except the ones to remove
    valid_indices = [i for i in range(len(dataset)) if i not in indices_to_remove]
    filtered_dataset = dataset.select(valid_indices)
    
    print(f"Filtered {split_name} size: {len(filtered_dataset)}")
    print(f"Removed {len(dataset) - len(filtered_dataset)} entries from {split_name}")
    
    return filtered_dataset, missing_content_indices

def remove_missing_content_multiple_splits(dataset_name, splits=["train", "test"], push_to_hub=False):
    """
    Remove entries with missing content from multiple splits of the dataset
    
    Args:
        dataset_name: Name of the HuggingFace dataset
        splits: List of dataset splits to process (e.g., ["train", "test", "validation"])
        push_to_hub: Whether to push the cleaned dataset back to HuggingFace Hub
    """
    print(f"Loading dataset: {dataset_name}")
    
    # Load the full dataset
    try:
        full_dataset = load_dataset(dataset_name)
        available_splits = list(full_dataset.keys())
        print(f"Available splits: {available_splits}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None
    
    # Filter splits to only process those that exist
    splits_to_process = [split for split in splits if split in available_splits]
    if not splits_to_process:
        print(f"None of the requested splits {splits} are available in the dataset")
        return None
    
    print(f"Processing splits: {splits_to_process}")
    
    cleaned_splits = {}
    all_removed_indices = {}
    
    # Process each split
    for split in splits_to_process:
        cleaned_dataset, removed_indices = process_single_split(full_dataset[split], split)
        cleaned_splits[split] = cleaned_dataset
        all_removed_indices[split] = removed_indices
    
    # Create DatasetDict with cleaned splits
    cleaned_dataset_dict = DatasetDict(cleaned_splits)
    
    # Print summary
    print(f"\n=== SUMMARY ===")
    total_original = sum(len(full_dataset[split]) for split in splits_to_process)
    total_cleaned = sum(len(cleaned_splits[split]) for split in splits_to_process)
    total_removed = total_original - total_cleaned
    
    print(f"Total original entries: {total_original}")
    print(f"Total cleaned entries: {total_cleaned}")
    print(f"Total removed entries: {total_removed}")
    
    for split in splits_to_process:
        original_size = len(full_dataset[split])
        cleaned_size = len(cleaned_splits[split])
        removed_count = len(all_removed_indices[split])
        print(f"  {split}: {original_size} -> {cleaned_size} (removed {removed_count})")
    
    # Save locally first
    output_dir = f"./cleaned_{dataset_name.replace('/', '_')}"
    cleaned_dataset_dict.save_to_disk(output_dir)
    print(f"\nCleaned dataset saved locally to: {output_dir}")
    
    # Save removed indices to a file for reference
    with open(f"{output_dir}/removed_indices.json", "w") as f:
        json.dump(all_removed_indices, f, indent=2)
    print(f"Removed indices saved to: {output_dir}/removed_indices.json")
    
    # Optionally push to hub
    if push_to_hub:
        try:
            # Create new dataset name with suffix
            new_dataset_name = f"{dataset_name}_cleaned"
            cleaned_dataset_dict.push_to_hub(new_dataset_name)
            print(f"Cleaned dataset pushed to: {new_dataset_name}")
        except Exception as e:
            print(f"Error pushing to hub: {e}")
            print("You may need to authenticate with `huggingface-cli login` first")
    
    return cleaned_dataset_dict, all_removed_indices

def remove_specific_indices_multiple_splits(dataset_name, indices_to_remove_dict, splits=["train", "test"], push_to_hub=False):
    """
    Remove specific dialogue indices from multiple splits of the dataset
    
    Args:
        dataset_name: Name of the HuggingFace dataset
        indices_to_remove_dict: Dict with split names as keys and lists of dialogue_idx values to remove
                               e.g., {"train": [123, 456], "test": [789, 101]}
        splits: List of dataset splits to process
        push_to_hub: Whether to push the cleaned dataset back to HuggingFace Hub
    """
    print(f"Loading dataset: {dataset_name}")
    full_dataset = load_dataset(dataset_name)
    available_splits = list(full_dataset.keys())
    
    # Filter splits to only process those that exist
    splits_to_process = [split for split in splits if split in available_splits]
    print(f"Processing splits: {splits_to_process}")
    
    cleaned_splits = {}
    
    for split in splits_to_process:
        indices_to_remove = indices_to_remove_dict.get(split, [])
        if not indices_to_remove:
            print(f"\nNo indices to remove for {split} split")
            cleaned_splits[split] = full_dataset[split]
            continue
        
        print(f"\n=== Processing {split} split ===")
        print(f"Original {split} size: {len(full_dataset[split])}")
        print(f"Removing dialogue indices: {indices_to_remove}")
        
        # Find dataset indices that correspond to the dialogue indices to remove
        dataset_indices_to_remove = []
        for idx, example in enumerate(full_dataset[split]):
            dialogue_idx = example.get('dialogue_idx', idx)
            if dialogue_idx in indices_to_remove:
                dataset_indices_to_remove.append(idx)
                print(f"Found dialogue_idx {dialogue_idx} at dataset index {idx} [{split}]")
        
        if len(dataset_indices_to_remove) == 0:
            print(f"No matching entries found to remove from {split}!")
            cleaned_splits[split] = full_dataset[split]
            continue
        
        # Create filtered dataset
        valid_indices = [i for i in range(len(full_dataset[split])) if i not in dataset_indices_to_remove]
        filtered_dataset = full_dataset[split].select(valid_indices)
        
        print(f"Filtered {split} size: {len(filtered_dataset)}")
        print(f"Removed {len(dataset_indices_to_remove)} entries from {split}")
        
        cleaned_splits[split] = filtered_dataset
    
    # Create DatasetDict with cleaned splits
    cleaned_dataset_dict = DatasetDict(cleaned_splits)
    
    # Save locally
    output_dir = f"./cleaned_{dataset_name.replace('/', '_')}"
    cleaned_dataset_dict.save_to_disk(output_dir)
    print(f"\nCleaned dataset saved locally to: {output_dir}")
    
    # Optionally push to hub
    if push_to_hub:
        try:
            new_dataset_name = f"{dataset_name}_cleaned"
            cleaned_dataset_dict.push_to_hub(new_dataset_name)
            print(f"Cleaned dataset pushed to: {new_dataset_name}")
        except Exception as e:
            print(f"Error pushing to hub: {e}")
    
    return cleaned_dataset_dict

if __name__ == "__main__":
    # Example usage - replace with your actual dataset name
    DATASET_NAME = "gustmd0121/ecg-dialogue-final"
    
    # Method 1: Automatically detect and remove entries with missing content from both splits
    print("=== Method 1: Auto-detect missing content for multiple splits ===")
    cleaned_dataset, removed_indices = remove_missing_content_multiple_splits(
        DATASET_NAME, 
        splits=["train", "test"],  # Can also include "validation" if it exists
        push_to_hub=True  # Set to True when ready to upload
    )
    
    # Method 2: Remove specific dialogue indices from multiple splits
    # print("=== Method 2: Remove specific indices from multiple splits ===")
    # specific_indices_dict = {
    #     "train": [123, 456, 789],  # Replace with actual train indices
    #     "test": [101, 202, 303]    # Replace with actual test indices
    # }
    # cleaned_dataset = remove_specific_indices_multiple_splits(
    #     DATASET_NAME,
    #     specific_indices_dict,
    #     splits=["train", "test"],
    #     push_to_hub=False
    # )
    
    # Method 3: If you want to process all available splits automatically
    # print("=== Method 3: Process all available splits ===")
    # full_dataset = load_dataset(DATASET_NAME)
    # all_splits = list(full_dataset.keys())
    # cleaned_dataset, removed_indices = remove_missing_content_multiple_splits(
    #     DATASET_NAME,
    #     splits=all_splits,
    #     push_to_hub=False
    # )