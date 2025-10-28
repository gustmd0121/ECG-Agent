#!/usr/bin/env python3
"""
ECG Dialogue Response Evaluation Script (Modified for Query-Based Matching & Output)

This script evaluates the accuracy, completeness, and coherence of ECG dialogue
responses from multiple models.

MODIFICATION: This version uses the preceding user query for matching and now INCLUDES
the ground truth response in the final output JSON for easier analysis.
MODIFICATION: This version loads the ground truth data directly from the Hugging Face Hub
dataset 'gustmd0121/12-lead-ecg-mtd-dataset'.
MODIFICATION: Implemented reproducible, stratified sampling to ensure a minimum number of
samples for each response category.
"""

import json
import os
import sys
import re
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime
import argparse
import random
from datasets import load_dataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ECGDialogueEvaluator:
    MODELS = ['gemini', 'pulse', 'gem', 'llama_1b', 'llama_3b', 'llama_8b', 'Qwen3_32b']
    RESPONSE_CATEGORIES = ['post_classification', 'post_measurement', 'direct_response']

    def __init__(self, api_key: Optional[str] = None, output_dir: str = 'evaluation_results', max_samples_per_category: Optional[int] = None):
        """
        Initialize the evaluator.
        
        Args:
            api_key: Google API key for Gemini-pro. If None, will try to get from environment.
            output_dir: Output directory for saving results.
            max_samples_per_category: The maximum number of samples to evaluate PER CATEGORY.
        """
        self.api_key = api_key or os.getenv('GOOGLE_API_KEY')
        if not self.api_key:
            logger.warning("No Google API key provided or found in environment. LLM-as-Judge evaluation will be skipped.")
        
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # MODIFICATION: Changed attribute name for clarity
        self.max_samples_per_category = max_samples_per_category
        
        self.gemini_responses = []
        self.pulse_responses = []
        self.gem_responses = []
        self.llama_1b_responses = []
        self.llama_3b_responses = []
        self.llama_8b_responses = []
        self.qwen3_32b_responses = []
        self.ground_truth = {}
        
    def load_data(self, gemini_file: str = None, pulse_file: str = None, gem_file: str = None, llama_1b_file: str = None, llama_3b_file: str = None, llama_8b_file: str = None, qwen3_32b_file: str = None):
        """Load response files and ground truth data. Files are optional."""
        logger.info("Loading response data files...")
        
        if gemini_file and os.path.exists(gemini_file):
            try:
                # MODIFICATION: Handle both .json and .jsonl files
                if gemini_file.endswith('.jsonl'):
                    with open(gemini_file, 'r') as f: self.gemini_responses = [json.loads(line) for line in f if line.strip()]
                else: # Assumes .json
                    with open(gemini_file, 'r') as f: self.gemini_responses = json.load(f)
                
                logger.info(f"Loaded {len(self.gemini_responses)} Gemini responses")
            except Exception as e: logger.error(f"Error loading Gemini file {gemini_file}: {e}")
            
        if pulse_file and os.path.exists(pulse_file):
            try:
                with open(pulse_file, 'r') as f: self.pulse_responses = json.load(f)
                logger.info(f"Loaded {len(self.pulse_responses)} PULSE responses")
            except Exception as e: logger.error(f"Error loading PULSE file {pulse_file}: {e}")

        if gem_file and os.path.exists(gem_file):
            try:
                with open(gem_file, 'r') as f: self.gem_responses = json.load(f)
                logger.info(f"Loaded {len(self.gem_responses)} GEM responses")
            except Exception as e: logger.error(f"Error loading GEM file {gem_file}: {e}")
            
        if llama_1b_file and os.path.exists(llama_1b_file):
            try:
                with open(llama_1b_file, 'r') as f: self.llama_1b_responses = [json.loads(line) for line in f if line.strip()]
                logger.info(f"Loaded {len(self.llama_1b_responses)} LLaMA-1B responses")
            except Exception as e: logger.error(f"Error loading LLaMA-1B file {llama_1b_file}: {e}")
        
        if llama_3b_file and os.path.exists(llama_3b_file):
            try:
                with open(llama_3b_file, 'r') as f: self.llama_3b_responses = [json.loads(line) for line in f if line.strip()]
                logger.info(f"Loaded {len(self.llama_3b_responses)} LLaMA-3B responses")
            except Exception as e: logger.error(f"Error loading LLaMA-3B file {llama_3b_file}: {e}")

        if llama_8b_file and os.path.exists(llama_8b_file):
            try:
                with open(llama_8b_file, 'r') as f: self.llama_8b_responses = [json.loads(line) for line in f if line.strip()]
                logger.info(f"Loaded {len(self.llama_8b_responses)} LLaMA-8B responses")
            except Exception as e: logger.error(f"Error loading LLaMA-8B file {llama_8b_file}: {e}")

        if qwen3_32b_file and os.path.exists(qwen3_32b_file):
            try:
                with open(qwen3_32b_file, 'r') as f: self.qwen3_32b_responses = [json.loads(line) for line in f if line.strip()]
                logger.info(f"Loaded {len(self.qwen3_32b_responses)} Qwen3_32b responses")
            except Exception as e: logger.error(f"Error loading Qwen3_32b file {qwen3_32b_file}: {e}")

        # Load ground truth from Hugging Face dataset
        self._load_ground_truth_from_hf()

        return True

    def _filter_to_common_ecg_ids(self):
        """
        MODIFIED:
        - Identifies common ECG IDs across all loaded models AFTER filtering out IDs with empty responses.
        - Performs stratified sampling to ensure a specific number of samples PER CATEGORY.
        - Uses a fixed random seed for reproducible sampling.
        - Filters all model data to this final, combined set of IDs.
        """
        logger.info("Finding common ECG IDs with non-empty responses and performing stratified sampling...")
        random.seed(42)  # Ensure reproducible random sampling

        gemini_pulse_id_extractor = lambda r: str(json.loads(r['metadata'])['ecg_id'])
        common_format_id_extractor = lambda r: re.search(r'(\d+)', r.get('ecg_file', '')).group(0) if re.search(r'(\d+)', r.get('ecg_file', '')) else None

        all_model_sources = [
            (self.gemini_responses, gemini_pulse_id_extractor, 'Gemini', lambda r: json.loads(r.get('dialogue', '{}'))),
            (self.pulse_responses, gemini_pulse_id_extractor, 'PULSE', lambda r: json.loads(r.get('dialogue', '{}'))),
            (self.gem_responses, gemini_pulse_id_extractor, 'GEM', lambda r: json.loads(r.get('dialogue', '{}'))),
            (self.llama_1b_responses, common_format_id_extractor, 'LLaMA-1B', lambda r: r.get('generated_dialogue')),
            (self.llama_3b_responses, common_format_id_extractor, 'LLaMA-3B', lambda r: r.get('generated_dialogue')),
            (self.llama_8b_responses, common_format_id_extractor, 'LLaMA-8B', lambda r: r.get('generated_dialogue')),
            (self.qwen3_32b_responses, common_format_id_extractor, 'Qwen3_32b', lambda r: r.get('generated_dialogue'))
        ]

        loaded_sources = [(responses, id_ext, name, dlg_ext) for responses, id_ext, name, dlg_ext in all_model_sources if responses]

        if len(loaded_sources) < 2:
            logger.warning("Fewer than 2 model files loaded. Skipping common ID filtering.")
            return

        id_sets = []
        for responses, id_extractor, name, dialogue_extractor in loaded_sources:
            valid_ids = set()
            for r in responses:
                dialogue = dialogue_extractor(r)
                if self._has_assistant_response(dialogue):
                    ecg_id = id_extractor(r)
                    if ecg_id:
                        valid_ids.add(ecg_id)
            id_sets.append(valid_ids)
            logger.info(f"Found {len(valid_ids)} ECG IDs with valid responses for model '{name}'.")

        common_ids = set.intersection(*id_sets)
        logger.info(f"Found {len(common_ids)} total common ECG IDs with responses across all models.")

        final_selected_ids = set()

        if not self.max_samples_per_category:
            logger.info("No max_samples_per_category specified. Using all common ECG IDs.")
            final_selected_ids = common_ids
        else:
            logger.info(f"Performing stratified sampling to get up to {self.max_samples_per_category} samples per category.")
            
            category_to_ids = {cat: [] for cat in self.RESPONSE_CATEGORIES}
            for ecg_id in common_ids:
                if ecg_id in self.ground_truth:
                    gt_categories = self.ground_truth[ecg_id]
                    for category in self.RESPONSE_CATEGORIES:
                        if gt_categories.get(category):
                            category_to_ids[category].append(ecg_id)
            
            for category, ids in category_to_ids.items():
                logger.info(f"Found {len(ids)} common IDs for category '{category}'.")
                
                if len(ids) > self.max_samples_per_category:
                    sampled_ids = random.sample(ids, self.max_samples_per_category)
                    logger.info(f" -> Randomly sampling {self.max_samples_per_category} IDs for this category.")
                else:
                    sampled_ids = ids
                    logger.info(f" -> Taking all {len(ids)} available IDs for this category.")
                    
                final_selected_ids.update(sampled_ids)

            logger.info(f"Total unique ECG IDs selected after stratified sampling: {len(final_selected_ids)}")

        if not final_selected_ids:
            logger.error("No ECG IDs were selected for evaluation. Cannot proceed.")
            sys.exit(1)
            
        self.gemini_responses = [r for r in self.gemini_responses if gemini_pulse_id_extractor(r) in final_selected_ids]
        self.pulse_responses = [r for r in self.pulse_responses if gemini_pulse_id_extractor(r) in final_selected_ids]
        self.gem_responses = [r for r in self.gem_responses if gemini_pulse_id_extractor(r) in final_selected_ids]
        self.llama_1b_responses = [r for r in self.llama_1b_responses if common_format_id_extractor(r) in final_selected_ids]
        self.llama_3b_responses = [r for r in self.llama_3b_responses if common_format_id_extractor(r) in final_selected_ids]
        self.llama_8b_responses = [r for r in self.llama_8b_responses if common_format_id_extractor(r) in final_selected_ids]
        self.qwen3_32b_responses = [r for r in self.qwen3_32b_responses if common_format_id_extractor(r) in final_selected_ids]
        
        logger.info("Filtered all model data to the final sampled set. New counts:")
        logger.info(f"  - Gemini: {len(self.gemini_responses)} responses")
        logger.info(f"  - PULSE: {len(self.pulse_responses)} responses")
        logger.info(f"  - GEM: {len(self.gem_responses)} responses")
        logger.info(f"  - LLaMA-1B: {len(self.llama_1b_responses)} responses")
        logger.info(f"  - LLaMA-3B: {len(self.llama_3b_responses)} responses")
        logger.info(f"  - LLaMA-8B: {len(self.llama_8b_responses)} responses")
        logger.info(f"  - Qwen3_32b: {len(self.qwen3_32b_responses)} responses")

    def _load_existing_results(self, filename: str) -> Dict:
        """Load existing evaluation results from a JSON file to allow resuming."""
        filepath = os.path.join(self.output_dir, filename)
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error loading existing results from {filepath}: {e}")
        return {}

    def _save_incremental_results(self, results: Dict, filename: str):
        """Save results incrementally to a JSON file."""
        filepath = os.path.join(self.output_dir, filename)
        try:
            with open(filepath, 'w') as f: json.dump(results, f, indent=2)
        except Exception as e: logger.warning(f"Error saving incremental results to {filepath}: {e}")

    def _load_ground_truth_from_hf(self):
        """Load and categorize ground truth responses from the Hugging Face dataset."""
        logger.info("Loading ground truth from Hugging Face dataset")
        try:
            dataset = load_dataset('gustmd0121/single-lead-I-ecg-mtd-dataset-gt-gemini-pro', split='train')
            self.ground_truth = {}
            for row in dataset:
                try:
                    metadata = json.loads(row['metadata'])
                    ecg_id = str(metadata['ecg_id'])
                    dialogue = json.loads(row['dialogue'])
                    self.ground_truth[ecg_id] = self._categorize_assistant_responses(dialogue)
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"Skipping ground truth row due to parsing error: {e}")
            
            logger.info(f"Processed ground truth for {len(self.ground_truth)} unique ECG IDs.")
        except Exception as e:
            logger.error(f"Failed to load or process Hugging Face dataset: {e}")
            self.ground_truth = {}

    def extract_and_categorize_responses(self) -> Dict:
        """Extract and categorize all assistant responses for all loaded models."""
        logger.info("Extracting and categorizing responses...")
        
        results = {}

        def process_model_data(responses: List[Dict], model_name: str, dialogue_key: str, id_extractor):
            for res in responses:
                try:
                    ecg_id = id_extractor(res)
                    if not ecg_id: continue
                    dialogue = res.get(dialogue_key) if dialogue_key else json.loads(res.get('dialogue'))
                    if not dialogue: continue

                    if ecg_id not in results:
                        results[ecg_id] = {m: {cat: {} for cat in self.RESPONSE_CATEGORIES} for m in self.MODELS}
                    
                    categorized_responses = self._categorize_assistant_responses(dialogue)
                    results[ecg_id][model_name] = categorized_responses
                except Exception as e:
                    logger.warning(f"Error processing {model_name} response for ECG ID {ecg_id if 'ecg_id' in locals() else 'unknown'}: {e}")

        common_format_id_extractor = lambda r: re.search(r'(\d+)', r.get('ecg_file', '')).group(0) if re.search(r'(\d+)', r.get('ecg_file', '')) else None

        process_model_data(self.gemini_responses, 'gemini', None, lambda r: str(json.loads(r['metadata'])['ecg_id']))
        process_model_data(self.pulse_responses, 'pulse', None, lambda r: str(json.loads(r['metadata'])['ecg_id']))
        process_model_data(self.gem_responses, 'gem', None, lambda r: str(json.loads(r['metadata'])['ecg_id']))
        process_model_data(self.llama_1b_responses, 'llama_1b', 'generated_dialogue', common_format_id_extractor)
        process_model_data(self.llama_3b_responses, 'llama_3b', 'generated_dialogue', common_format_id_extractor)
        process_model_data(self.llama_8b_responses, 'llama_8b', 'generated_dialogue', common_format_id_extractor)
        process_model_data(self.qwen3_32b_responses, 'Qwen3_32b', 'generated_dialogue', common_format_id_extractor)

        logger.info(f"Extracted and categorized responses for {len(results)} ECG IDs")
        return results

    def _has_assistant_response(self, dialogue: List[Dict]) -> bool:
        """Checks if a dialogue contains at least one valid assistant response."""
        if not dialogue:
            return False
        for turn in dialogue:
            if (turn.get('role') == 'assistant' and
                turn.get('action') == 'response' and
                turn.get('content', '').strip()):
                return True
        return False

    def _categorize_assistant_responses(self, dialogue: List[Dict]) -> Dict:
        """
        Iterates through a dialogue and sorts assistant 'response' turns into
        dictionaries keyed by the preceding user query.
        """
        categorized = {cat: {} for cat in self.RESPONSE_CATEGORIES}
        
        for i, turn in enumerate(dialogue):
            if turn.get('role') == 'assistant' and turn.get('action') == 'response':
                prev_turn = dialogue[i - 1] if i > 0 else {}
                prev_action = prev_turn.get('action')
                prev_role = prev_turn.get('role')

                user_query = None
                # Find the most recent user turn to use as the key
                for j in range(i - 1, -1, -1):
                    if dialogue[j].get('role') == 'user':
                        user_query = dialogue[j].get('content')
                        break
                
                if not user_query:
                    continue
                
                context = {'user_query': user_query, 'turn_number': i + 1}
                
                response_data = {
                    'response': turn.get('content'),
                    'context': context
                }

                category = None
                if prev_action == 'call_classification_tool':
                    response_data['tool_output'] = prev_turn.get('tool_output')
                    category = 'post_classification'
                elif prev_action == 'call_measurement_tool':
                    response_data['tool_output'] = prev_turn.get('tool_output')
                    category = 'post_measurement'
                elif prev_role == 'user' or prev_action == 'response_followup':
                    category = 'direct_response'
                
                if category:
                    categorized[category][user_query] = response_data

        return categorized

    def evaluate_tool_responses(self, categorized_responses: Dict, category: str) -> Dict:
        """
        Evaluate tool-based responses, including the GT response in the output.
        """
        if not self.api_key or not self.ground_truth: return {}
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            model = genai.GenerativeModel('gemini-2.5-pro') # Updated model name
        except Exception as e:
            logger.error(f"Error initializing Gemini model: {e}. Skipping tool response evaluation.")
            return {}

        results_filename = f'llm_eval_{category}.json'
        evaluation_results = self._load_existing_results(results_filename)
        logger.info(f"Starting LLM-as-Judge evaluation for category: '{category}'...")

        safety_settings = {
            "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
            "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
            "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
        }

        for ecg_id, gt_data in self.ground_truth.items():
            if ecg_id not in evaluation_results: evaluation_results[ecg_id] = {}
            
            gt_category_responses = gt_data.get(category, {})
            for user_query, gt_response_data in gt_category_responses.items():
                
                for model_name in self.MODELS:
                    # Logic to check if re-evaluation is needed
                    should_skip = False
                    if model_name in evaluation_results.get(ecg_id, {}):
                        for res in evaluation_results[ecg_id][model_name]:
                            res_context = res.get('response_data', {}).get('context', {})
                            if res_context and res_context.get('user_query') == user_query:
                                score_data = res.get('score', {})
                                # If the score exists AND it is NOT the specific error we want to retry, then we skip.
                                if score_data.get('error') != 'Model did not provide a response for this query.':
                                    should_skip = True
                                break # Found the relevant entry, no need to check further.
                    
                    if should_skip:
                        continue
                    
                    model_dialogue_data = categorized_responses.get(ecg_id, {}).get(model_name)
                    if not model_dialogue_data: continue

                    model_response_data = None
                    for search_category in self.RESPONSE_CATEGORIES:
                        if user_query in model_dialogue_data.get(search_category, {}):
                            model_response_data = model_dialogue_data[search_category][user_query]
                            break
                    
                    logger.info(f"Evaluating {model_name.upper()} for ECG ID: {ecg_id} (Query: '{user_query[:30]}...', Category: {category})")
                    
                    if model_name not in evaluation_results[ecg_id]:
                        evaluation_results[ecg_id][model_name] = []

                    score = {}
                    if not model_response_data:
                        score = {'error': 'Model did not provide a response for this query.'}
                    else:
                        try:
                            prompt = self._create_tool_evaluation_prompt(ecg_id, model_response_data, model_name, gt_response_data['response'], category)
                            response = model.generate_content(prompt, safety_settings=safety_settings)
                            
                            if not response.parts:
                                raise ValueError("The model returned an empty response, likely due to a content filter.")
                                
                            score = self._parse_structured_response(response.text, ['accuracy', 'completeness'])
                        except Exception as e:
                            logger.error(f"Error evaluating {model_name} on {ecg_id} for {category}: {e}")
                            raw_response_text = str(response) if 'response' in locals() else "Response object not created."
                            score = {'error': str(e), 'raw_response_for_debug': raw_response_text}

                    # Overwrite existing entry if it was a "no response" error, otherwise append
                    entry_found_and_updated = False
                    if model_name in evaluation_results.get(ecg_id, {}):
                        for i, res in enumerate(evaluation_results[ecg_id][model_name]):
                            res_context = res.get('response_data', {}).get('context', {})
                            if res_context and res_context.get('user_query') == user_query:
                                evaluation_results[ecg_id][model_name][i] = {
                                    'response_data': model_response_data or {'context': {'user_query': user_query}},
                                    'score': score,
                                    'ground_truth_response': gt_response_data.get('response') 
                                }
                                entry_found_and_updated = True
                                break
                    
                    if not entry_found_and_updated:
                        evaluation_results[ecg_id][model_name].append({
                            'response_data': model_response_data or {'context': {'user_query': user_query}},
                            'score': score,
                            'ground_truth_response': gt_response_data.get('response') 
                        })

            self._save_incremental_results(evaluation_results, results_filename)
        
        logger.info(f"LLM-as-Judge evaluation for '{category}' completed.")
        return evaluation_results

    def evaluate_direct_responses(self, categorized_responses: Dict) -> Dict:
        """
        MODIFIED: Evaluate direct responses for ACCURACY and COMPLETENESS.
        """
        if not self.api_key or not self.ground_truth: return {}
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            model = genai.GenerativeModel('gemini-2.5-pro') # Updated model name
        except Exception as e:
            logger.error(f"Error initializing Gemini model: {e}. Skipping direct response evaluation.")
            return {}

        results_filename = 'llm_eval_direct_response.json'
        evaluation_results = self._load_existing_results(results_filename)
        logger.info("Starting LLM-as-Judge evaluation for category: 'direct_response'...")
        category = 'direct_response'

        safety_settings = {
            "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
            "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
            "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
        }

        for ecg_id, gt_data in self.ground_truth.items():
            if ecg_id not in evaluation_results: evaluation_results[ecg_id] = {}

            gt_category_responses = gt_data.get(category, {})
            for user_query, gt_response_data in gt_category_responses.items():
                
                for model_name in self.MODELS:
                    # Logic to check if re-evaluation is needed
                    should_skip = False
                    if model_name in evaluation_results.get(ecg_id, {}):
                        for res in evaluation_results[ecg_id][model_name]:
                            res_context = res.get('response_data', {}).get('context', {})
                            if res_context and res_context.get('user_query') == user_query:
                                score_data = res.get('score', {})
                                # If the score exists AND it is NOT the specific error we want to retry, then we skip.
                                if score_data.get('error') != 'Model did not provide a response for this query.':
                                    should_skip = True
                                break # Found the relevant entry, no need to check further.
                    
                    if should_skip:
                        continue

                    model_dialogue_data = categorized_responses.get(ecg_id, {}).get(model_name)
                    if not model_dialogue_data: continue

                    model_response_data = None
                    for search_category in self.RESPONSE_CATEGORIES:
                        if user_query in model_dialogue_data.get(search_category, {}):
                            model_response_data = model_dialogue_data[search_category][user_query]
                            break
                    
                    logger.info(f"Evaluating {model_name.upper()} for ECG ID: {ecg_id} (Query: '{user_query[:30]}...', Category: {category})")

                    if model_name not in evaluation_results[ecg_id]:
                        evaluation_results[ecg_id][model_name] = []

                    score = {}
                    if not model_response_data:
                        score = {'error': 'Model did not provide a response for this query.'}
                    else:
                        try:
                            prompt = self._create_direct_evaluation_prompt(ecg_id, model_response_data, model_name, gt_response_data['response'])
                            response = model.generate_content(prompt, safety_settings=safety_settings)

                            if not response.parts:
                                raise ValueError("The model returned an empty response, likely due to a content filter.")

                            score = self._parse_structured_response(response.text, ['accuracy', 'completeness'])
                        except Exception as e:
                            logger.error(f"Error evaluating {model_name} on {ecg_id} for direct_response: {e}")
                            raw_response_text = str(response) if 'response' in locals() else "Response object not created."
                            score = {'error': str(e), 'raw_response_for_debug': raw_response_text}

                    # Overwrite existing entry if it was a "no response" error, otherwise append
                    entry_found_and_updated = False
                    if model_name in evaluation_results.get(ecg_id, {}):
                        for i, res in enumerate(evaluation_results[ecg_id][model_name]):
                            res_context = res.get('response_data', {}).get('context', {})
                            if res_context and res_context.get('user_query') == user_query:
                                evaluation_results[ecg_id][model_name][i] = {
                                    'response_data': model_response_data or {'context': {'user_query': user_query}},
                                    'score': score,
                                    'ground_truth_response': gt_response_data.get('response')
                                }
                                entry_found_and_updated = True
                                break
                    
                    if not entry_found_and_updated:
                        evaluation_results[ecg_id][model_name].append({
                            'response_data': model_response_data or {'context': {'user_query': user_query}},
                            'score': score,
                            'ground_truth_response': gt_response_data.get('response')
                        })

            self._save_incremental_results(evaluation_results, results_filename)
        
        logger.info("LLM-as-Judge evaluation for 'direct_response' completed.")
        return evaluation_results

    def _create_tool_evaluation_prompt(self, ecg_id: str, response_data: Dict, model_name: str, gt_response: str, category: str) -> str:
        """Create the evaluation prompt for tool-based responses."""
        return f"""
You are a medical expert evaluating an AI's ECG dialogue response. Compare the assistant's response to the ground truth.

**Context:**
- ECG ID: {ecg_id}
- Model: {model_name}
- Response Category: {category}
- Tool Output (provided to assistant): {response_data.get('tool_output', 'N/A')}
- User's Question: {response_data.get('context', {}).get('user_query', '')}

**Ground Truth Response:**
{gt_response}

**Assistant Response to Evaluate:**
{response_data.get('response', '')}

**Evaluation Criteria:**
1. Accuracy (1-5): How well does the response match the ground truth? Key information to look for are representative diagnosis classes (e.g., sinus rhythm, myocardial infarction) or measurement interval (e.g., Heart rate, PR interval, QRS duration, QTc interval) for post tool calls and representative key information for direct responses. 
    - 5: Fully accurate. If the diagnosis classes mentioned by the model's response is fully accurate, as in all representative diagnosis classes are identified and correct, this score should be applied. For measurement interval, if for example the heart rate, pr interval, qrs duration, and qtc interval is accurate within the normal range for a normal ECG, this score should be applied. 
    - 3: Partially accurate. If the diagnosis classes mentioned by the model's response is partially inaccurate, as in one representative diagnosis class is identified but another one is inaccurate (50% accuracy), this score should be applied. For measurement interval, if for example the measurements of heart rate and pr interval is accurate within the normal range for a normal ECG but the QRS duration and QTC interval are inaccurate, this score should be applied.  
    - 1: Largely inaccurate. If the diagnosis classes mentioned by the model's response is entirely inaccurate, as in not a single relevant class is identified, this score should be applied. For measurement interval, if for example abnormal heart rate, pr interval, qrs duration, and qtc interval is identified for a completely normal ECG, this score should be applied.
2. Completeness (1-5): Does the response cover essential information mentioned in the ground truth? This score focuses on whether the answer is comprehensive and includes as much essential information as possible. 
    - 5: Comprehensive. If the response includes all key information from the ground truth, this score should be applied. 
    - 3: Partially complete. If the response includes some key information but is missing others, this score should be applied.
    - 1: Incomplete. If the response is missing all key information, this score should be applied.

**Instructions:** Rate each criterion and provide a brief justification.

**Response Format:**
Accuracy: [score]
Accuracy Explanation: [justification]
Completeness: [score]
Completeness Explanation: [justification]
"""

    def _create_direct_evaluation_prompt(self, ecg_id: str, response_data: Dict, model_name: str, gt_response: str) -> str:
        """
        MODIFIED: Create the evaluation prompt for direct (non-tool) responses,
        now evaluating for Accuracy and Completeness.
        """
        return f"""
You are a medical expert evaluating an AI's ECG dialogue response. Compare the assistant's response to the ground truth.

**Context:**
- ECG ID: {ecg_id}
- Model: {model_name}
- Response Category: direct_response
- User's Question/Previous Turn: {response_data.get('context', {}).get('user_query', '')}

**Ground Truth Response:**
{gt_response}

**Assistant Response to Evaluate:**
{response_data.get('response', '')}

**Evaluation Criteria:**
1. Accuracy (1-5): How well does the response match the ground truth? Key information to look for are representative diagnosis classes (e.g., sinus rhythm, myocardial infarction) or measurement interval (e.g., Heart rate, PR interval, QRS duration, QTc interval) for post tool calls and representative key information for direct responses. 
    - 5: Fully accurate. If the diagnosis classes mentioned by the model's response is fully accurate, as in all representative diagnosis classes are identified and correct, this score should be applied. For measurement interval, if for example the heart rate, pr interval, qrs duration, and qtc interval is accurate within the normal range for a normal ECG, this score should be applied.  
    - 3: Partially accurate. If the diagnosis classes mentioned by the model's response is partially inaccurate, as in one representative diagnosis class is identified but another one is inaccurate (50% accuracy), this score should be applied. For measurement interval, if for example the measurements of heart rate and pr interval is accurate within the normal range for a normal ECG but the QRS duration and QTC interval are inaccurate, this score should be applied.  
    - 1: Largely inaccurate. If the diagnosis classes mentioned by the model's response is entirely inaccurate, as in not a single relevant class is identified, this score should be applied. For measurement interval, if for example abnormal heart rate, pr interval, qrs duration, and qtc interval is identified for a completely normal ECG, this score should be applied.
2. Completeness (1-5): Does the response cover essential information mentioned in the ground truth? This score focuses on whether the answer is comprehensive and includes as much essential information as possible. 
    - 5: Comprehensive. If the response includes all key information from the ground truth, this score should be applied. 
    - 3: Partially complete. If the response includes some key information but is missing others, this score should be applied.
    - 1: Incomplete. If the response is missing all key information, this score should be applied.
    

**Instructions:** Rate each criterion and provide a brief justification.

**Response Format:**
Accuracy: [score]
Accuracy Explanation: [justification]
Completeness: [score]
Completeness Explanation: [justification]
"""

    def _parse_structured_response(self, response_text: str, criteria: List[str]) -> Dict:
        """Parse a structured LLM response for multiple criteria."""
        parsed_data = {}
        try:
            for criterion in criteria:
                score_match = re.search(rf"^{criterion}:\s*(\d+)", response_text, re.IGNORECASE | re.MULTILINE)
                explanation_match = re.search(rf"^{criterion} Explanation:\s*(.*)", response_text, re.IGNORECASE | re.MULTILINE)
                
                score = int(score_match.group(1)) if score_match else None
                justification = explanation_match.group(1).strip() if explanation_match else "No justification found."
                
                if score is not None:
                    parsed_data[criterion] = {'score': score, 'justification': justification}
            
            numeric_scores = [v['score'] for k, v in parsed_data.items() if isinstance(v, dict) and 'score' in v]
            if numeric_scores:
                parsed_data['average_score'] = sum(numeric_scores) / len(numeric_scores)
            
            if not parsed_data:
                return {'error': 'Could not parse any criteria', 'raw_response': response_text}

        except Exception as e:
            return {'error': str(e), 'raw_response': response_text}
        
        return parsed_data

    def _summarize_results(self, evaluation_results: Dict, score_key: str, criteria: List[str]) -> Dict:
        """Generic summarizer for evaluation results."""
        summary = {model: {'scores': {c: [] for c in criteria}, 'count': 0, 'errors': 0} for model in self.MODELS}
        
        for ecg_id, results in evaluation_results.items():
            for model, model_results in results.items():
                if model not in summary: continue
                for result in model_results:
                    score_data = result.get(score_key, {})
                    if score_data and 'error' not in score_data:
                        has_valid_score = False
                        for key in criteria:
                            if key in score_data and isinstance(score_data[key], dict) and 'score' in score_data[key]:
                                summary[model]['scores'][key].append(score_data[key]['score'])
                                has_valid_score = True
                        if has_valid_score:
                            summary[model]['count'] += 1
                    else:
                        summary[model]['errors'] +=1
        
        for model, data in summary.items():
            data['avg_scores'] = {}
            for key, scores in data['scores'].items():
                data['avg_scores'][key] = sum(scores) / len(scores) if scores else 0
        
        return summary
    
    def generate_report(self, all_eval_results: Dict) -> str:
        """Generate a comprehensive evaluation report with categorized results."""
        logger.info("Generating evaluation report...")
        report = ["# ECG Dialogue Response Evaluation Report", f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ""]

        if 'post_classification' in all_eval_results and all_eval_results['post_classification']:
            report.append("## Post-Classification Response Evaluation (Accuracy & Completeness)")
            summary = self._summarize_results(all_eval_results['post_classification'], 'score', ['accuracy', 'completeness'])
            report.append("| Model           | Valid Responses | Avg Accuracy | Avg Completeness |")
            report.append("|---------------|-----------------|--------------|------------------|")
            for model in self.MODELS:
                s = summary.get(model, {})
                if s.get('count', 0) > 0 or s.get('errors', 0) > 0:
                    report.append(f"| {model.upper():<13} | {s['count']:<15} | {s['avg_scores'].get('accuracy', 0):<12.2f} | {s['avg_scores'].get('completeness', 0):<16.2f} |")
            report.append("")

        if 'post_measurement' in all_eval_results and all_eval_results['post_measurement']:
            report.append("## Post-Measurement Response Evaluation (Accuracy & Completeness)")
            summary = self._summarize_results(all_eval_results['post_measurement'], 'score', ['accuracy', 'completeness'])
            report.append("| Model           | Valid Responses | Avg Accuracy | Avg Completeness |")
            report.append("|---------------|-----------------|--------------|------------------|")
            for model in self.MODELS:
                s = summary.get(model, {})
                if s.get('count', 0) > 0 or s.get('errors', 0) > 0:
                    report.append(f"| {model.upper():<13} | {s['count']:<15} | {s['avg_scores'].get('accuracy', 0):<12.2f} | {s['avg_scores'].get('completeness', 0):<16.2f} |")
            report.append("")

        if 'direct_response' in all_eval_results and all_eval_results['direct_response']:
            report.append("## Direct Response Evaluation (Accuracy & Completeness)")
            summary = self._summarize_results(all_eval_results['direct_response'], 'score', ['accuracy', 'completeness'])
            report.append("| Model           | Valid Responses | Avg Accuracy | Avg Completeness |")
            report.append("|---------------|-----------------|--------------|------------------|")
            for model in self.MODELS:
                s = summary.get(model, {})
                if s.get('count', 0) > 0 or s.get('errors', 0) > 0:
                    report.append(f"| {model.upper():<13} | {s['count']:<15} | {s['avg_scores'].get('accuracy', 0):<12.2f} | {s['avg_scores'].get('completeness', 0):<16.2f} |")
            report.append("")
        
        return '\n'.join(report)
    
    def save_results(self, categorized_responses: Dict, all_eval_results: Dict):
        """Save all categorized results to files in the output directory."""
        with open(os.path.join(self.output_dir, 'categorized_responses_by_query.json'), 'w') as f:
            json.dump(categorized_responses, f, indent=2)
        
        for category, results in all_eval_results.items():
            if results:
                with open(os.path.join(self.output_dir, f'llm_eval_{category}.json'), 'w') as f:
                    json.dump(results, f, indent=2)
        
        report = self.generate_report(all_eval_results)
        with open(os.path.join(self.output_dir, 'evaluation_report.md'), 'w') as f:
            f.write(report)
        
        logger.info(f"All results and report saved to directory: {self.output_dir}/")

def main():
    """Main evaluation function."""
    
    parser = argparse.ArgumentParser(description="Evaluate ECG dialogue responses from multiple models.")
    
    # --- MODIFIED: Added arguments for file paths ---
    parser.add_argument('--gemini_file', type=str, help='Path to Gemini response file (.json or .jsonl)')
    parser.add_argument('--pulse_file', type=str, help='Path to PULSE response file (.json)')
    parser.add_argument('--gem_file', type=str, help='Path to GEM response file (.json)')
    parser.add_argument('--llama_1b_file', type=str, help='Path to LLaMA-1B response file (.jsonl)')
    parser.add_argument('--llama_3b_file', type=str, help='Path to LLaMA-3B response file (.jsonl)')
    parser.add_argument('--llama_8b_file', type=str, help='Path to LLaMA-8B response file (.jsonl)')
    parser.add_argument('--qwen3_32b_file', type=str, help='Path to Qwen3_32b response file (.jsonl)')

    # --- MODIFIED: Added argument for output directory ---
    parser.add_argument(
        '--output_dir',
        type=str,
        default='evaluation_results', # Uses the class default
        help='Directory to save evaluation results and reports.'
    )
    
    # --- Existing Arguments ---
    parser.add_argument(
        '--max_samples_per_category',
        type=int,
        default=100,
        help='The maximum number of randomly selected common samples to evaluate for EACH category.'
    )

    parser.add_argument('--skip_post_classification', action='store_true',
                        help='Skip evaluating post-classification responses.')   

    args = parser.parse_args()

    # --- MODIFIED: Removed hardcoded paths ---
    # The hardcoded path variables have been removed.

    # --- MODIFIED: Populate model_files from args ---
    model_files = {
        'gemini_file': args.gemini_file, 
        'pulse_file': args.pulse_file,
        'gem_file' : args.gem_file, 
        'llama_1b_file': args.llama_1b_file, 
        'llama_3b_file': args.llama_3b_file, 
        'llama_8b_file': args.llama_8b_file,
        'qwen3_32b_file': args.qwen3_32b_file
    }
    
    # This loop correctly checks only the paths that were provided (are not None)
    for name, path in model_files.items():
        if path and not os.path.exists(path):
            logger.error(f"Provided file path for '{name}' is invalid: {path}")
            sys.exit(1)

    # --- MODIFIED: Use args.output_dir from parser ---
    evaluator = ECGDialogueEvaluator(output_dir=args.output_dir, max_samples_per_category=args.max_samples_per_category)
    
    if not evaluator.load_data(**model_files):
        logger.error("Failed to load data files. Exiting.")
        sys.exit(1)
    
    # This function now performs stratified sampling
    evaluator._filter_to_common_ecg_ids()

    categorized_responses = evaluator.extract_and_categorize_responses()

    # Evaluate the categories you want
    post_measurement_eval = evaluator.evaluate_tool_responses(categorized_responses, 'post_measurement')
    direct_response_eval = evaluator.evaluate_direct_responses(categorized_responses)

    # Only run post_classification if not skipped
    post_classification_eval = {}
    if not args.skip_post_classification:
        post_classification_eval = evaluator.evaluate_tool_responses(
            categorized_responses, 'post_classification'
        )

    # Build results without re-triggering evaluations
    all_eval_results = {
        'post_measurement': post_measurement_eval,
        'direct_response': direct_response_eval,
    }
    if not args.skip_post_classification:
        all_eval_results['post_classification'] = post_classification_eval

    evaluator.save_results(categorized_responses, all_eval_results)
    logger.info("Evaluation with stratified sampling completed successfully!")

if __name__ == "__main__":
    main()