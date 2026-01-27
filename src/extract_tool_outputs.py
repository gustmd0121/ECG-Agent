import os
import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm
from medrax.tools.classification import ECGClassifierTool, ECGAnalysisTool

def process_ecg_files_to_csv(
    ecg_file_paths: list,
    output_csv_path: str,
    tool_choice: str,
    model_path: str = None,
    probability_threshold: float = 0.8
):
    """
    Processes a list of ECG files to extract classifications or interval measurements,
    then saves the results to a CSV file.

    Args:
        ecg_file_paths: List of paths to ECG .mat files
        output_csv_path: Path to save the output CSV
        tool_choice: 'classification' or 'measurements'
        model_path: Path to the fine-tuned model checkpoint (required for classification)
        probability_threshold: Threshold for filtering classification results
    """
    print(f"Initializing tool: {tool_choice}...")
    tool = None
    try:
        if tool_choice == 'classification':
            if not model_path:
                print("Error: --model_path is required for classification tool.")
                return
            tool = ECGClassifierTool(model_path=model_path)
        elif tool_choice == 'measurements':
            tool = ECGAnalysisTool()
        else:
            print("Invalid tool choice. Please choose 'classification' or 'measurements'.")
            return
    except Exception as e:
        print(f"Error initializing tool: {e}")
        return

    results_list = []
    # CHANGED: Wrapped the main loop with tqdm to create a progress bar.
    # The `desc` argument provides a descriptive label for the bar.
    print(f"Processing {len(ecg_file_paths)} files...")
    for ecg_path in tqdm(ecg_file_paths, desc=f"Analyzing ECGs ({tool_choice})"):
        if not os.path.exists(ecg_path):
            # Note: printing inside a fast tqdm loop can clutter the output.
            # It's often better to log errors to a separate file.
            results_list.append({"ecg_file_path": ecg_path, "error": "File Not Found"})
            continue

        # The print statement for each file was removed to keep the tqdm bar clean.
        current_file_results = {"ecg_file_path": os.path.basename(ecg_path)}

        try:
            if tool_choice == 'classification':
                outputs, _ = tool._run(ecg_path=ecg_path)
                if outputs and "error" not in outputs:
                    filtered_classes = {label: prob for label, prob in outputs.items() if prob > probability_threshold}
                    sorted_classes = sorted(filtered_classes.items(), key=lambda item: item[1], reverse=True)
                    if sorted_classes:
                        current_file_results["top_classes"] = ", ".join([f"{label} ({prob:.2%})" for label, prob in sorted_classes])
                    else:
                        current_file_results["top_classes"] = f"None > {probability_threshold:.0%}"
                else:
                    current_file_results["top_classes"] = "Classification Error"
            
            elif tool_choice == 'measurements':
                outputs = tool._run(ecg_path=ecg_path)
                if outputs and "error" not in outputs:
                    current_file_results["PR_Interval_ms"] = outputs.get("PR_Interval_ms", np.nan)
                    current_file_results["QTc_ms"] = outputs.get("QTc_ms", np.nan)
                    current_file_results["QRS_Duration_ms"] = outputs.get("QRS_Duration_ms", np.nan)
                    current_file_results["Heart_Rate"] = outputs.get("Heart_Rate", np.nan)
                else:
                    current_file_results["error"] = "Analysis Error"

        except Exception as e:
            # Capturing the error is good, but printing it in the loop can be messy.
            current_file_results["error"] = "Processing Exception"
            
        results_list.append(current_file_results)

    if not results_list:
        print("No files were processed.")
        return

    print("Processing complete. Saving results to CSV...")
    df = pd.DataFrame(results_list)
    try:
        df.to_csv(output_csv_path, index=False)
        print(f"\nResults successfully saved to {output_csv_path}")
    except Exception as e:
        print(f"\nError saving CSV to {output_csv_path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze 12-lead ECG files using classification or measurement tools.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run classification with a fine-tuned model
  python extract_tool_outputs.py --tool classification --ecg_dir ./ecg_data --model_path ./checkpoints/checkpoint_best.pt --output_dir ./results

  # Run measurements (no model needed)
  python extract_tool_outputs.py --tool measurements --ecg_dir ./ecg_data --output_dir ./results
        """
    )
    parser.add_argument(
        "--tool",
        type=str,
        default="classification",
        choices=['classification', 'measurements'],
        help="The analysis tool to use: 'classification' (requires --model_path) or 'measurements'."
    )
    parser.add_argument(
        "--ecg_dir",
        type=str,
        required=True,
        help="Directory containing ECG .mat files."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to the fine-tuned model checkpoint (required for classification tool)."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Directory to save the output CSV file. Defaults to current directory."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Full path to save the output CSV file. Overrides --output_dir if specified."
    )
    parser.add_argument(
        "--prob_threshold",
        type=float,
        default=0.5,
        help="Probability threshold for filtering classification results (default: 0.5)."
    )
    args = parser.parse_args()

    # Validate arguments
    if args.tool == 'classification' and not args.model_path:
        parser.error("--model_path is required when using --tool classification")

    # Determine output file path
    if args.output_file:
        output_file = args.output_file
    else:
        os.makedirs(args.output_dir, exist_ok=True)
        output_file = os.path.join(args.output_dir, f"ecg_analysis_{args.tool}.csv")

    # Collect ECG files
    ecg_files_to_process = []
    if os.path.exists(args.ecg_dir):
        for filename in sorted(os.listdir(args.ecg_dir)):
            if filename.lower().endswith(".mat"):
                ecg_files_to_process.append(os.path.join(args.ecg_dir, filename))
    else:
        print(f"Error: The specified ECG directory does not exist: {args.ecg_dir}")
        exit(1)

    if not ecg_files_to_process:
        print("No .mat files found in the specified directory.")
        exit(1)
    else:
        print(f"Found {len(ecg_files_to_process)} ECG files to process.")
        process_ecg_files_to_csv(
            ecg_files_to_process,
            output_file,
            tool_choice=args.tool,
            model_path=args.model_path,
            probability_threshold=args.prob_threshold
        )