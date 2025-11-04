import json
import os
import pandas as pd
from collections import defaultdict
import argparse  # Added import

def analyze_action_accuracy(file_path: str) -> dict:
    """
    Analyzes a JSONL file to calculate the next action prediction accuracy,
    categorized by each type of action.

    Args:
        file_path (str): The path to the JSONL evaluation file.

    Returns:
        dict: A dictionary where keys are action types and values are
              dicts containing total and correct counts.
    """
    # Use defaultdict to easily initialize new action types
    action_stats = defaultdict(lambda: {"total": 0, "correct": 0})

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if 'turns' in data and isinstance(data['turns'], list):
                        for turn in data['turns']:
                            # Use the ground truth action as the category
                            action_type = turn.get('gt_action')
                            
                            # Ensure the turn is an assistant action and has the necessary keys
                            if action_type and 'action_correct' in turn:
                                action_stats[action_type]["total"] += 1
                                if turn['action_correct'] is True:
                                    action_stats[action_type]["correct"] += 1
                except json.JSONDecodeError:
                    print(f"Warning: Could not decode JSON from a line in {file_path}. Skipping line.")
                    continue
    except FileNotFoundError:
        print(f"Warning: File not found at {file_path}. Skipping.")
        return None

    return dict(action_stats)

if __name__ == "__main__":
    # --- Configuration ---
    # MODIFICATION: Replaced hardcoded paths with argparse, following the example
    parser = argparse.ArgumentParser(description="Analyze next action prediction accuracy from JSONL files.")
    
    # Arguments based on the example script, excluding 'pulse_file' and 'gem_file'
    parser.add_argument('--llama_1b_file', type=str, help='Path to LLaMA-1B results file')
    parser.add_argument('--llama_3b_file', type=str, help='Path to LLaMA-3B results file')
    parser.add_argument('--llama_8b_file', type=str, help='Path to LLaMA-8B results file')
    parser.add_argument('--qwen3_32b_file', type=str, help='Path to Qwen3_32b results file')


    args = parser.parse_args()

    # MODIFICATION: Dynamically build the files_to_analyze dictionary
    files_to_analyze = {}
    if args.llama_1b_file:
        files_to_analyze["llama_1b"] = args.llama_1b_file
    if args.llama_3b_file:
        files_to_analyze["llama_3b"] = args.llama_3b_file
    if args.llama_8b_file:
        files_to_analyze["llama_8b"] = args.llama_8b_file
    if args.qwen3_32b_file:
        files_to_analyze["Qwen3_32b"] = args.qwen3_32b_file
    
    all_results = {}

    print("Analyzing files...")
    # The rest of the script logic remains unchanged and works with the new dict
    for model_name, filename in files_to_analyze.items():
        if os.path.exists(filename):
            analysis = analyze_action_accuracy(filename)
            if analysis:
                all_results[model_name] = analysis
        else:
            print(f"Warning: File '{filename}' for model '{model_name}' not found. Skipping.")

    if not all_results:
        print("\nNo valid results to display. Please check file paths and content.")
    else:
        print("\n" + "="*50)
        print("====== Next Action Prediction Accuracy Report ======")
        print("="*50)

        for model_name, action_stats in all_results.items():
            print(f"\n--- Results for: {model_name} ---\n")
            
            if not action_stats:
                print("No actions with 'action_correct' flag found in the file.")
                continue

            report_data = []
            total_all_actions = 0
            correct_all_actions = 0

            # Prepare data for the DataFrame and calculate totals
            for action, stats in sorted(action_stats.items()):
                total = stats['total']
                correct = stats['correct']
                
                total_all_actions += total
                correct_all_actions += correct
                
                accuracy = (correct / total * 100) if total > 0 else 0
                report_data.append({
                    "Action Type": action,
                    "Accuracy": f"{accuracy:.2f}%"
                })
            
            # Calculate overall accuracy
            overall_accuracy = (correct_all_actions / total_all_actions * 100) if total_all_actions > 0 else 0
            
            # Add a separator and the overall accuracy row
            report_data.append({"Action Type": "---", "Accuracy": "---"})
            report_data.append({
                "Action Type": "Overall Accuracy",
                "Accuracy": f"{overall_accuracy:.2f}%"
            })
            
            # Create and display the DataFrame for the current model
            df = pd.DataFrame(report_data)
            print(df.to_string(index=False))
            print("\n" + "-"*50)