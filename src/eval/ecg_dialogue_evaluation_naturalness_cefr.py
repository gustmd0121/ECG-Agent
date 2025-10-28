"""
ECG Dialogue Quality Evaluation Script

This script evaluates the dialogue-level naturalness and CEFR level adherence
of generated ECG conversations from multiple models. It uses an LLM-as-a-judge
(Gemini 1.5 Flash) to score each complete dialogue.
"""
import json
import os
import sys
import re
from typing import Dict, List, Optional
import logging
from datetime import datetime
import argparse
import random
from datasets import load_dataset
from tqdm import tqdm
import time# --- Basic Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
class DialogueQualityEvaluator:
    MODELS = ['gemini', 'pulse', 'gem', 'llama_1b', 'llama_3b', 'llama_8b', 'Qwen3_32b']

    def __init__(self, api_key: Optional[str] = None, output_dir: str = 'evaluation_results', samples_to_eval: Optional[int] = None):
        """
        Initializes the dialogue quality evaluator.
        """
        self.api_key = api_key or os.getenv('GOOGLE_API_KEY')
        if not self.api_key:
            logger.error("A Google API key is required. Please set the GOOGLE_API_KEY environment variable.")
            sys.exit(1)
        
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.samples_to_eval = samples_to_eval

        # Data storage
        self.model_data = {model: [] for model in self.MODELS}
        self.cefr_levels = {}
        self.common_ecg_ids = set()

    def load_data(self, **kwargs):
        """Loads all model output files and ground truth CEFR levels."""
        logger.info("Loading model output files and ground truth data...")

        def _load_file(filepath):
            if not filepath or not os.path.exists(filepath): return []
            try:
                if filepath.endswith('.jsonl'):
                    with open(filepath, 'r') as f: return [json.loads(line) for line in f if line.strip()]
                else:
                    with open(filepath, 'r') as f: return json.load(f)
            except Exception as e:
                logger.error(f"Error loading file {filepath}: {e}")
                return []

        self.model_data['gemini'] = _load_file(kwargs.get('gemini_file'))
        self.model_data['pulse'] = _load_file(kwargs.get('pulse_file'))
        self.model_data['gem'] = _load_file(kwargs.get('gem_file'))
        self.model_data['llama_1b'] = _load_file(kwargs.get('llama_1b_file'))
        self.model_data['llama_3b'] = _load_file(kwargs.get('llama_3b_file'))
        self.model_data['llama_8b'] = _load_file(kwargs.get('llama_8b_file'))
        self.model_data['Qwen3_32b'] = _load_file(kwargs.get('qwen3_32b_file')) # Name mapping

        for model, data in self.model_data.items():
            if data:
                logger.info(f"Loaded {len(data)} responses for {model.upper()}")

        self._load_cefr_levels_from_hf()
        return True

    def _load_cefr_levels_from_hf(self):
        """Loads CEFR levels from the Hugging Face dataset."""
        logger.info("Loading CEFR levels from Hugging Face dataset...")
        try:
            dataset = load_dataset('gustmd0121/single-lead-II-ecg-mtd-dataset-gt-gemini-pro', split='train')
            for row in dataset:
                try:
                    metadata = json.loads(row['metadata'])
                    ecg_id = str(metadata['ecg_id'])
                    self.cefr_levels[ecg_id] = metadata.get('cefr_level', 'UNKNOWN')
                except (json.JSONDecodeError, KeyError):
                    continue
            logger.info(f"Loaded CEFR levels for {len(self.cefr_levels)} unique ECG IDs.")
        except Exception as e:
            logger.error(f"Failed to load Hugging Face dataset: {e}")
            sys.exit(1)

    def filter_to_common_samples(self):
        """Finds common ECG IDs across all loaded models and randomly samples if needed."""
        logger.info("Filtering to a common set of ECG IDs...")
        random.seed(42)  # For reproducibility

        id_extractors = {
            'gemini': lambda r: str(json.loads(r['metadata'])['ecg_id']),
            'pulse': lambda r: str(json.loads(r['metadata'])['ecg_id']),
            'gem': lambda r: str(json.loads(r['metadata'])['ecg_id']),
            'llama_1b': lambda r: re.search(r'(\d+)', r.get('ecg_file', '')).group(0),
            'llama_3b': lambda r: re.search(r'(\d+)', r.get('ecg_file', '')).group(0),
            'llama_8b': lambda r: re.search(r'(\d+)', r.get('ecg_file', '')).group(0),
            'Qwen3_32b': lambda r: re.search(r'(\d+)', r.get('ecg_file', '')).group(0),
        }

        id_sets = []
        for model_name, data in self.model_data.items():
            if not data: continue
            extractor = id_extractors[model_name]
            current_ids = set()
            for item in data:
                try:
                    ecg_id = extractor(item)
                    if ecg_id:
                        current_ids.add(ecg_id)
                except:
                    continue
            id_sets.append(current_ids)
            logger.info(f"Found {len(current_ids)} valid IDs for {model_name.upper()}")
        
        if not id_sets:
            logger.error("No valid data loaded for any model. Exiting.")
            sys.exit(1)

        self.common_ecg_ids = set.intersection(*id_sets)
        logger.info(f"Found {len(self.common_ecg_ids)} common ECG IDs across all models.")

        if self.samples_to_eval and self.samples_to_eval < len(self.common_ecg_ids):
            logger.info(f"Randomly sampling {self.samples_to_eval} ECG IDs for evaluation.")
            self.common_ecg_ids = set(random.sample(list(self.common_ecg_ids), self.samples_to_eval))
        
        logger.info(f"Final evaluation set contains {len(self.common_ecg_ids)} ECG IDs.")

    def _extract_clean_dialogue_text(self, raw_dialogue: List[Dict]) -> str:
        """
        Formats a dialogue list into a clean "User: ... \nAssistant: ..." string,
        skipping non-conversational turns like tool calls.
        """
        if not raw_dialogue:
            return "No dialogue available."
            
        clean_turns = []
        response_actions = {"response", "response_followup", "system_bye"} # Actions indicating a conversational reply
        
        for turn in raw_dialogue:
            role = turn.get('role')
            content = turn.get('content', '').strip()
            action = turn.get('action')

            if not content:
                continue

            if role == 'user':
                clean_turns.append(f"User: {content}")
            elif role == 'assistant':
                # For agents, only include turns that are direct responses to the user
                if action in response_actions:
                    clean_turns.append(f"Assistant: {content}")
        
        return "\n".join(clean_turns)

    def _create_dialogue_quality_prompt(self, dialogue_text: str, cefr_level: str) -> str:
        """Creates the LLM-as-a-judge prompt for naturalness and CEFR evaluation."""
        return f"""
**Role:** You are an expert evaluator of conversational AI, specializing in linguistics and user experience.

**Task:** Evaluate the provided AI-generated dialogue on two criteria: **Dialogue Naturalness** and **CEFR Level Adherence**.

---

**Dialogue to Evaluate:**
{dialogue_text}


---

**Context for Evaluation:**
- The dialogue is between a user and a medical AI assistant discussing an ECG.
- **The user's language proficiency is CEFR Level: {cefr_level}**

---

**Evaluation Criteria & Scales:**

**1. Dialogue Naturalness (Score 1-5):**
Assess how fluid, coherent, and human-like the conversation feels.
- **5 (Very Natural):** Indistinguishable from a conversation between two humans. Flows logically and has a natural tone.
- **4 (Natural):** Mostly fluid and easy to follow, with only minor instances of stiffness.
- **3 (Acceptable):** Understandable, but contains some awkward phrasing or unnatural transitions that reveal its AI origin.
- **2 (Unnatural):** Stilted, difficult to follow, and clearly sounds robotic or scripted.
- **1 (Very Unnatural):** Incoherent, nonsensical, or fails to resemble a real conversation.

**2. CEFR Adherence (Score 1-5):**
Assess if the assistant's language (vocabulary, sentence structure, jargon) is appropriate for the user's specified CEFR level ({cefr_level}).
- **5 (Excellent Adherence):** Language is perfectly tailored to the user's level. Complex medical concepts are explained simply and appropriately.
- **4 (Good Adherence):** Language is mostly appropriate, with only a few words or phrases being slightly too simple or complex.
- **3 (Moderate Adherence):** A noticeable mismatch. The assistant frequently uses overly complex jargon or, conversely, is condescendingly simple.
- **2 (Poor Adherence):** Consistently mismatched language that would likely hinder the user's understanding or feel patronizing.
- **1 (No Adherence):** The assistant completely ignores the user's language level.

---

CEFR Levels Reference: 
- Level A: Describes a user with basic language skills who communicates using short, simple sentences, common vocabulary, and direct questions to express immediate worry and seek simple clarification.
- Level B: Characterizes an intermediate user who employs moderately complex sentences and some medical vocabulary to ask explanatory follow-up questions about the causes, seriousness, and implications of their condition.
- Level C: Defines an advanced user who confidently uses complex sentence structures and precise medical terminology to ask sophisticated, analytical questions about clinical implications, correlations, and prognosis.

**Instructions:** Provide a score and a brief justification for each criterion based on the scales above.

**Required Output Format:**
Naturalness Score: [score]
Naturalness Justification: [one-sentence justification]
CEFR Adherence Score: [score]
CEFR Adherence Justification: [one-sentence justification]
"""

    def _parse_llm_response(self, response_text: str) -> Dict:
        """Parses the structured response from the LLM evaluator."""
        parsed = {}
        try:
            nat_score = re.search(r"Naturalness Score:\s*(\d)", response_text, re.IGNORECASE)
            nat_just = re.search(r"Naturalness Justification:\s*(.*)", response_text, re.IGNORECASE)
            cefr_score = re.search(r"CEFR Adherence Score:\s*(\d)", response_text, re.IGNORECASE)
            cefr_just = re.search(r"CEFR Adherence Justification:\s*(.*)", response_text, re.IGNORECASE)

            if nat_score: parsed['naturalness'] = {'score': int(nat_score.group(1))}
            if nat_just: parsed.setdefault('naturalness', {})['justification'] = nat_just.group(1).strip()
            if cefr_score: parsed['cefr_adherence'] = {'score': int(cefr_score.group(1))}
            if cefr_just: parsed.setdefault('cefr_adherence', {})['justification'] = cefr_just.group(1).strip()
            
            if 'naturalness' not in parsed or 'cefr_adherence' not in parsed:
                raise ValueError("Could not parse all required fields.")

        except Exception as e:
            logger.warning(f"Failed to parse LLM response. Error: {e}. Raw text: {response_text[:200]}")
            return {'error': str(e), 'raw_response': response_text}
        return parsed

    def run_evaluation(self):
        """Main function to run the dialogue quality evaluation across all models."""
        logger.info("🚀 Starting Dialogue Quality Evaluation...")
        
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self.evaluator_model = genai.GenerativeModel('gemini-2.5-pro')
        except Exception as e:
            logger.error(f"Failed to initialize Gemini model: {e}")
            return None

        results_filename = 'llm_eval_dialogue_quality.json'
        results_filepath = os.path.join(self.output_dir, results_filename)
        
        if os.path.exists(results_filepath):
            with open(results_filepath, 'r') as f:
                evaluation_results = json.load(f)
            logger.info(f"Resuming from existing results file: {results_filepath}")
        else:
            evaluation_results = {model: {} for model in self.MODELS}

        # --- Data Preparation ---
        # Create a lookup map for easy access: ecg_id -> model -> raw_dialogue
        dialogue_map = {ecg_id: {model: None for model in self.MODELS} for ecg_id in self.common_ecg_ids}

        dialogue_keys = {
            'gemini': lambda r: json.loads(r.get('dialogue', '{}')),
            'pulse': lambda r: json.loads(r.get('dialogue', '{}')),
            'gem': lambda r: json.loads(r.get('dialogue', '{}')),
            'llama_1b': lambda r: r.get('generated_dialogue'),
            'llama_3b': lambda r: r.get('generated_dialogue'),
            'llama_8b': lambda r: r.get('generated_dialogue'),
            'Qwen3_32b': lambda r: r.get('generated_dialogue'),
        }
        id_extractors = {
            'gemini': lambda r: str(json.loads(r['metadata'])['ecg_id']),
            'pulse': lambda r: str(json.loads(r['metadata'])['ecg_id']),
            'gem': lambda r: str(json.loads(r['metadata'])['ecg_id']),
            'llama_1b': lambda r: re.search(r'(\d+)', r.get('ecg_file', '')).group(0),
            'llama_3b': lambda r: re.search(r'(\d+)', r.get('ecg_file', '')).group(0),
            'llama_8b': lambda r: re.search(r'(\d+)', r.get('ecg_file', '')).group(0),
            'Qwen3_32b': lambda r: re.search(r'(\d+)', r.get('ecg_file', '')).group(0),
        }
        
        for model_name, data in self.model_data.items():
            if not data: continue
            id_extractor = id_extractors[model_name]
            dialogue_extractor = dialogue_keys[model_name]
            for item in data:
                try:
                    ecg_id = id_extractor(item)
                    if ecg_id in self.common_ecg_ids:
                        dialogue_map[ecg_id][model_name] = dialogue_extractor(item)
                except:
                    continue

        # --- Evaluation Loop ---
        safety_settings = {
            "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE", "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE", "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
        }

        tasks = [(ecg_id, model_name) for ecg_id in self.common_ecg_ids for model_name in self.MODELS if self.model_data.get(model_name)]

        for ecg_id, model_name in tqdm(tasks, desc="Evaluating Dialogues"):
            if evaluation_results.get(model_name, {}).get(ecg_id):
                continue # Skip if already evaluated

            raw_dialogue = dialogue_map.get(ecg_id, {}).get(model_name)
            cefr_level = self.cefr_levels.get(ecg_id, 'B') # Default to 'B' if not found

            if not raw_dialogue:
                logger.warning(f"No dialogue found for {model_name} on ECG ID {ecg_id}. Skipping.")
                continue

            clean_dialogue_text = self._extract_clean_dialogue_text(raw_dialogue)
            prompt = self._create_dialogue_quality_prompt(clean_dialogue_text, cefr_level)

            try:
                response = self.evaluator_model.generate_content(prompt, safety_settings=safety_settings)
                score_data = self._parse_llm_response(response.text)
                time.sleep(1) # Rate limit to avoid API errors
            except Exception as e:
                # Check for specific quota-related errors to halt the script
                error_text = str(e).lower()
                if 'quota' in error_text or 'resource has been exhausted' in error_text:
                    logger.critical(f"🛑 Quota limit reached. Halting evaluation. Error: {e}")
                    
                    # Save progress before exiting to not lose completed work
                    with open(results_filepath, 'w') as f:
                        json.dump(evaluation_results, f, indent=2)
                    logger.info(f"Progress saved to {results_filepath} before exiting.")
                    
                    sys.exit(1) # Exit the script with an error code
                else:
                    # Handle other, non-fatal API errors as before
                    logger.error(f"API Error for {model_name} on {ecg_id}: {e}")
                    score_data = {'error': str(e)}
                    time.sleep(5) # Longer backoff on error

            evaluation_results[model_name][ecg_id] = {
                'clean_dialogue': clean_dialogue_text,
                'cefr_level': cefr_level,
                'scores': score_data
            }
            
            # Incremental save
            with open(results_filepath, 'w') as f:
                json.dump(evaluation_results, f, indent=2)
                
        logger.info(f"Evaluation complete. Results saved to {results_filepath}")
        return evaluation_results

    def generate_report(self, results: Dict):
        """Generates and saves a markdown summary report."""
        logger.info("Generating final report...")
        report_path = os.path.join(self.output_dir, 'dialogue_quality_report.md')
        
        summary = {model: {'naturalness': [], 'cefr_adherence': [], 'count': 0, 'errors': 0} for model in self.MODELS}
        
        for model_name, model_results in results.items():
            if not model_results: continue
            for ecg_id, data in model_results.items():
                scores = data.get('scores', {})
                if 'error' in scores:
                    summary[model_name]['errors'] += 1
                    continue
                
                if 'naturalness' in scores and 'cefr_adherence' in scores:
                    summary[model_name]['naturalness'].append(scores['naturalness']['score'])
                    summary[model_name]['cefr_adherence'].append(scores['cefr_adherence']['score'])
                    summary[model_name]['count'] += 1
        
        report_lines = [
            "# Dialogue Quality Evaluation Report",
            f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
            "This report summarizes the average scores for dialogue naturalness and CEFR level adherence.",
            f"Total dialogues evaluated per model: {summary[self.MODELS[0]]['count']}\n",
            "| Model         | Avg Naturalness | Avg CEFR Adherence | Valid Evals | Errors |",
            "|---------------|-----------------|--------------------|-------------|--------|"
        ]

        for model_name in self.MODELS:
            data = summary[model_name]
            if data['count'] == 0 and data['errors'] == 0: continue
            
            avg_nat = sum(data['naturalness']) / len(data['naturalness']) if data['naturalness'] else 0
            avg_cefr = sum(data['cefr_adherence']) / len(data['cefr_adherence']) if data['cefr_adherence'] else 0
            
            report_lines.append(
                f"| {model_name.upper():<13} | {avg_nat:<15.2f} | {avg_cefr:<18.2f} | {data['count']:<11} | {data['errors']:<6} |"
            )
            
        final_report = "\n".join(report_lines)
        with open(report_path, 'w') as f:
            f.write(final_report)
            
        logger.info(f"Report saved to {report_path}")
        print("\n" + final_report)


def main():
    parser = argparse.ArgumentParser(description="Evaluate ECG dialogue naturalness and CEFR adherence.")
    parser.add_argument(
        '--samples_to_eval', type=int, default=200,
        help='The total number of common dialogues to evaluate. Set to 0 for all.'
    )
    # --- FILE PATHS (Update these to match your system) ---
    # Note: Using commented-out single-lead-ii paths as a template
    parser.add_argument('--gemini_file', type=str, help='Path to Gemini response file (.json or .jsonl)')
    parser.add_argument('--pulse_file', type=str, help='Path to PULSE response file (.json)')
    parser.add_argument('--gem_file', type=str, help='Path to GEM response file (.json)')
    parser.add_argument('--llama_1b_file', type=str, help='Path to LLaMA-1B response file (.jsonl)')
    parser.add_argument('--llama_3b_file', type=str, help='Path to LLaMA-3B response file (.jsonl)')
    parser.add_argument('--llama_8b_file', type=str, help='Path to LLaMA-8B response file (.jsonl)')
    parser.add_argument('--qwen3_32b_file', type=str, help='Path to Qwen3_32b response file (.jsonl)')
    parser.add_argument('--output_dir', type=str, default='lead_ii_naturalness_cefr_evaluation_results_0905')
    
    args = parser.parse_args()

    # Create an absolute path for the output directory
    output_dir = os.path.abspath(args.output_dir)
    
    evaluator = DialogueQualityEvaluator(
        output_dir=output_dir,
        samples_to_eval=args.samples_to_eval if args.samples_to_eval > 0 else None
    )

    model_files = {
        'gemini_file': args.gemini_file, 'pulse_file': args.pulse_file, 'gem_file': args.gem_file,
        'llama_1b_file': args.llama_1b_file, 'llama_3b_file': args.llama_3b_file, 'llama_8b_file': args.llama_8b_file,
        'qwen3_32b_file': args.qwen3_32b_file
    }

    evaluator.load_data(**model_files)
    evaluator.filter_to_common_samples()
    
    results = evaluator.run_evaluation()
    if results:
        evaluator.generate_report(results)
    
    logger.info("✅ Evaluation finished successfully!")

if __name__ == "__main__":
    main()
