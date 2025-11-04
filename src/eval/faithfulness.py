import json
import os
import time
import argparse  # Using argparse
# You will need to install the google-generativeai and tqdm libraries
# pip install google-generativeai tqdm
import google.generativeai as genai
from tqdm import tqdm

def evaluate_with_gemini(tool_output: str, response_content: str) -> int:
    """
    Evaluates the faithfulness of a response to a tool_output using Gemini.
    Includes retry logic and response validation.

    Args:
        tool_output: The JSON string from the tool.
        response_content: The assistant's response to be evaluated.

    Returns:
        1 if the response is faithful, 0 otherwise.
    """
    # 1. Configure the Gemini API using the environment variable
    try:
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            print("Error: GOOGLE_API_KEY environment variable not set.")
            return 0
        genai.configure(api_key=api_key)
        # Using a current, robust model name
        model = genai.GenerativeModel('gemini-2.5-pro')
    except Exception as e:
        print(f"Error configuring the Gemini API: {e}")
        return 0

    prompt = f"""
    As an evaluator, your task is to determine if the assistant's response is a faithful representation of the provided tool output.
    You must score the response as either 1 (aligned) or 0 (not aligned).

    **Key Principle for Evaluation:**
    The primary goal is to ensure the response does not misrepresent the tool's output. **It is acceptable for the response to omit some information from the tool_output, as long as the information that *is* presented is accurate and not misleading.** A score of 0 should be given if the response contains incorrect or fabricated details.

    **Evaluation Criteria:**
    - **For classification tools:** The classes mentioned in the response **must be present in the tool_output. The response must not wrongfully represent or invent classes.** It is acceptable if the response does not mention all classes from the tool_output.
    - **For measurement tools:** Any measurements and values mentioned in the response **must be correctly and accurately cited from the tool_output.** It is acceptable to only use a subset of the measurements.
    - **For explanation tools:** The portion of the explanation conveyed in the response **must accurately reflect the core points of the tool_output without contradiction.** The response can be a partial summary and does not need to cover every detail.

    **Tool Output:**
    ```json
    {tool_output}
    ```

    **Assistant Response:**
    ```
    {response_content}
    ```

    **Instructions:**
    Based on the criteria above, is the "Assistant Response" aligned with the "Tool Output"?
    Respond with only a single integer: 1 for aligned, or 0 for not aligned.
    """

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            if response.parts:
                return int(response.text.strip())
            else:
                print(f"Warning: Received an empty response. Finish reason: {response.candidates[0].finish_reason}. Scoring as 0.")
                return 0
        except Exception as e:
            print(f"An error occurred during API call (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(1)
            else:
                print("Max retries reached. Scoring this turn as 0.")
                return 0
    return 0

def analyze_file(file_path: str) -> tuple[str, float] | None:
    """
    Analyzes a single JSONL file to calculate the faithfulness score.
    CORRECTED to parse the specific 'generated_dialogue' structure.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at '{file_path}'")
        return None
        
    scores = []
    # Model name is inferred from the filename
    model_name = os.path.basename(file_path).split('_results')[0]

    with open(file_path, 'r') as f:
        for line in tqdm(f, desc=f"Processing {model_name}"):
            try:
                data = json.loads(line)
                dialogue = data.get("generated_dialogue", [])

                for i, turn in enumerate(dialogue):
                    # --- CORRECTED LOGIC ---
                    # 1. Check if the turn is an assistant action that contains a tool_output
                    if turn.get("role") == "assistant" and "tool_output" in turn:
                        # 2. Check if there is a next turn and it's the assistant's response
                        if i + 1 < len(dialogue) and dialogue[i+1].get("role") == "assistant":
                            if dialogue[i+1].get("action") == "response_fail":
                                continue
                            # 3. Extract the correct fields for evaluation
                            tool_output_content = turn["tool_output"]
                            response_content = dialogue[i+1].get("content")

                            if response_content:
                                score = evaluate_with_gemini(tool_output_content, response_content)
                                scores.append(score)
            except json.JSONDecodeError:
                print(f"Warning: Skipping a malformed line in {file_path}")

    if scores:
        average_score = sum(scores) / len(scores)
        print(f"Analysis complete for '{model_name}'. Average score: {average_score:.2f}")
        return model_name, average_score
    else:
        print(f"No tool outputs found for model '{model_name}' in the file.")
        return model_name, 0.0

if __name__ == "__main__":
    
    # --- MODIFICATION: Changed argparse to use separate arguments ---
    parser = argparse.ArgumentParser(description="Evaluate model response faithfulness using the Gemini API.")
    
    # Add separate arguments for each model file
    parser.add_argument('--llama_1b_file', type=str, help='Path to LLaMA-1B results file')
    parser.add_argument('--llama_3b_file', type=str, help='Path to LLaMA-3B results file')
    parser.add_argument('--llama_8b_file', type=str, help='Path to LLaMA-8B results file')
    parser.add_argument('--qwen3_32b_file', type=str, help='Path to Qwen3_32b results file')
    # You can add more arguments for other models here if needed

    parser.add_argument(
        '--output_filename',
        type=str,
        required=True,
        help='The path to the output .json file to save scores (e.g., /path/to/scores.json).'
    )
    
    args = parser.parse_args()

    # --- MODIFICATION: Build the list of files to process ---
    model_files = []
    if args.llama_1b_file:
        model_files.append(args.llama_1b_file)
    if args.llama_3b_file:
        model_files.append(args.llama_3b_file)
    if args.llama_8b_file:
        model_files.append(args.llama_8b_file)
    if args.qwen3_32b_file:
        model_files.append(args.qwen3_32b_file)
    
    # Use the output filename from the parser
    output_filename = args.output_filename
    
    # --- All hardcoded paths are now replaced by the parser ---
    
    model_scores = {}

    # The rest of the script works as before
    for file_path in model_files:
        result = analyze_file(file_path)
        if result:
            model_name, score = result
            model_scores[model_name] = score
    
    # --- Print Summary to Console ---
    print("\n" + "="*30)
    print("      Faithfulness Score Summary      ")
    print("="*30)
    if model_scores:
        for model, score in model_scores.items():
            print(f"- {model}: {score:.2f}")
    else:
        print("No results to display. Check if any file paths were provided.")
    print("="*30)

    # --- Save Results to a JSON File ---
    try:
        with open(output_filename, 'w') as outfile:
            json.dump(model_scores, outfile, indent=4)
        print(f"\nSuccessfully saved scores to '{output_filename}'")
    except Exception as e:
        print(f"\nError saving scores to file: {e}")