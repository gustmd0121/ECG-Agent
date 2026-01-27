from typing import Dict, Optional, Tuple, Type
from pydantic import BaseModel, Field, PrivateAttr
import pdb
import skimage.io
import numpy as np 
import torch
import os 
import torchvision
import neurokit2 as nk 
import sys

import pdb 
import pandas as pd 
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool

from fairseq_signals.utils import checkpoint_utils
import scipy.io
from typing import List, Optional, Union, Any
import traceback
import math 

class ECGInput(BaseModel):
    """Input for ECG analysis tools. Only supports MAT files."""

    ecg_path: str = Field(
        ..., description="Path to the ECG signal file, only supports MAT files"
    )


class ECGAnalysisTool(BaseTool):
    """Tool that analyzes raw Electrocardiogram (ECG) signals to extract PQRST wave features and intervals.

    This tool uses the neurokit2 package to process ECG signals and extract medically relevant features including:
    - Heart Rate
    - RR Interval
    - PR Interval
    - QRS Duration
    - QTc (Corrected QT Interval)
    
    The analysis is performed specifically on lead II (lead I if lead II not available) of the ECG, which is the standard lead used for 
    rhythm analysis and interval measurements in clinical practice. 
    """
    name: str = "ecg_analysis"
    description: str = (
        "A tool that analyzes ECG signals and extracts key features like RR intervals, PR intervals, QRS duration"
        "QTc intervals, and heart rate. Input should be the path to an ECG signal file. "
        "Output is a dictionary of ECG features and their measurements in seconds, milliseconds, or beats per minute. "
        "The values are measured from lead II (lead I if lead II not available), which is the standard lead used for interval measurements in clinical practice. "
        "Use this tool to get precise measurements of ECG components when you need to assess "
        "specific intervals or identify potential abnormalities in the cardiac cycle."
    )
    args_schema: Type[BaseModel] = ECGInput
    device: Optional[str] = "cuda"
    
    def __init__(self, device: Optional[str] = "cuda"):
        super().__init__()
        self._device = device

    def _process_ecg_mat(self, ecg_path: str) -> np.ndarray:
        """Load ECG data from a MAT file and return the signal array."""
        try:
            import scipy.io
            ecg_data = scipy.io.loadmat(ecg_path)
            
            # Extract and return the ECG signal
            if "feats" in ecg_data:
                return ecg_data["feats"]  # Shape: (12, time_samples)
            else:
                # Try alternative field names that might contain the ECG data
                for key in ["data", "ECG", "signal", "val"]:
                    if key in ecg_data:
                        return ecg_data[key]
                        
                raise ValueError(f"Could not find ECG data in MAT file: {ecg_path}")
        except Exception as e:
            print(f"Error loading MAT file: {e}")
            raise
    
    # Add this missing helper method that you're using in the code
    def _safe_median(self, values, min_val=None, max_val=None):
        """Calculate median with NaN handling and optional range filtering."""
        import numpy as np 
        
        if not values or len(values) == 0:
            return np.nan
            
        # Convert to numpy array if it's not already
        arr = np.array(values, dtype=np.float64)
        
        # Filter out NaN values
        arr = arr[~np.isnan(arr)]
        
        # Apply range filters if specified
        if min_val is not None:
            arr = arr[arr >= min_val]
        if max_val is not None:
            arr = arr[arr <= max_val]
        
        # Check if we have any valid values left
        if arr.size == 0:
            return np.nan
            
        # Calculate and return the median
        return np.median(arr)

    def _extract_features(self, ecg_signal: np.ndarray, sampling_rate: int = 500) -> Dict:
        """Extract ECG features using neurokit2 from Lead II (standard for interval measurements)."""
        import numpy as np

        # Use Lead II (index 1) - standard lead for rhythm analysis and interval measurements
        lead_idx = 1
        lead_name_for_report = "Lead II"
        print(f"ECGAnalysisTool: Using {lead_name_for_report} (index {lead_idx}) for analysis.")

        all_features = {
            "Heart_Rate": np.nan,
            "RR_Interval_ms": np.nan,
            "P_Duration_ms": np.nan,
            "PR_Interval_ms": np.nan,
            "QRS_Duration_ms": np.nan, # Changed calculation method
            # "QT_Interval_ms": np.nan,
            "QTc_ms": np.nan,          # Changed calculation method
        }

        analysis_status = "failed" # Default status

        try:
            # Check if the signal has enough leads
            if ecg_signal.shape[0] <= lead_idx:
                print(f"ECG signal does not have {lead_name_for_report} (index {lead_idx})")
                return {
                    **all_features,
                    "analysis_status": "failed",
                    "note": f"ECG signal does not have {lead_name_for_report} data"
                }

            # Get lead II data
            lead_data = ecg_signal[lead_idx, :]

            # Check for invalid data
            if np.all(lead_data == 0) or np.all(np.isnan(lead_data)):
                print(f"{lead_name_for_report} contains invalid data - analysis failed")
                return {
                    **all_features,
                    "analysis_status": "failed",
                    "note": f"{lead_name_for_report} contains invalid data (all zeros or NaN)"
                }

            # --- Process signal using separate nk steps like extract_numeric_features ---
            try:
                cleaned_signal = nk.ecg_clean(lead_data, sampling_rate=sampling_rate, method="neurokit")
                _, r_peaks_dict = nk.ecg_peaks(cleaned_signal, sampling_rate=sampling_rate)

                # Check for sufficient peaks
                if len(r_peaks_dict['ECG_R_Peaks']) < 7: # Match check from extract_numeric_features
                    print(f"Too few peaks detected in {lead_name_for_report}: {len(r_peaks_dict['ECG_R_Peaks'])}. Cannot calculate numeric features.")
                    # Return NaNs but indicate partial success if cleaning worked
                    return {
                        **all_features,
                        "analysis_status": "partially_completed",
                        "note": f"Too few R-peaks ({len(r_peaks_dict['ECG_R_Peaks'])}) detected in {lead_name_for_report} for reliable interval calculation."
                    }

                _, waves_dict = nk.ecg_delineate(cleaned_signal, r_peaks_dict, sampling_rate=sampling_rate, method="dwt") # Using dwt method as implied by waves_dwt name

            except Exception as e:
                print(f"Error processing {lead_name_for_report} with neurokit2 (separate steps): {e}")
                traceback.print_exc()
                return {
                    **all_features,
                    "analysis_status": "failed",
                    "note": f"Error processing {lead_name_for_report}: {str(e)}"
                }

            # Extract delineation points
            r_peaks = np.array(r_peaks_dict.get("ECG_R_Peaks", []), dtype=np.float64)
            p_onsets = np.array(waves_dict.get("ECG_P_Onsets", []), dtype=np.float64)
            p_offsets = np.array(waves_dict.get("ECG_P_Offsets", []), dtype=np.float64)
            q_peaks = np.array(waves_dict.get("ECG_Q_Peaks", []), dtype=np.float64) # For QRS, QT
            r_onsets = np.array(waves_dict.get("ECG_R_Onsets", []), dtype=np.float64) # For PR, QRS
            r_offsets = np.array(waves_dict.get("ECG_R_Offsets", []), dtype=np.float64) # For QRS
            s_peaks = np.array(waves_dict.get("ECG_S_Peaks", []), dtype=np.float64) # For QRS
            t_offsets = np.array(waves_dict.get("ECG_T_Offsets", []), dtype=np.float64) # For QT

            # --- Calculate features based on extract_numeric_features logic ---

            # RR Intervals (ms)
            rr_intervals_ms = []
            for j in range(len(r_peaks) - 1):
                rr_interval_sec = (r_peaks[j+1] - r_peaks[j]) / sampling_rate
                if rr_interval_sec < 2.0: # Filter from extract_numeric_features
                     # Convert to ms
                    rr_intervals_ms.append(rr_interval_sec * 1000)

            # P Wave Duration (ms)
            # p_duration_ms = []
            # for p_onset, p_offset in zip(p_onsets, p_offsets):
            #     if not (np.isnan(p_onset) or np.isnan(p_offset)):
            #         # Add +1 like extract_numeric_features, convert to ms
            #         duration = ((p_offset - p_onset + 1) / sampling_rate) * 1000
            #         p_duration_ms.append(duration)

            # PR Interval (ms)
            pr_interval_ms = []
            # Ensure alignment: Use the length of the shorter array if necessary
            min_len_pr = min(len(p_onsets), len(r_onsets))
            for i in range(min_len_pr):
                p_onset = p_onsets[i]
                r_onset = r_onsets[i] # Use corresponding r_onset
                if not (np.isnan(p_onset) or np.isnan(r_onset)):
                     # Add +1 like extract_numeric_features, convert to ms
                    interval = ((r_onset - p_onset + 1) / sampling_rate) * 1000
                    pr_interval_ms.append(interval)

            # QRS Duration (ms) - Using Q-peak to S-peak
            # pdb.set_trace()
            qrs_duration_ms = []
            # Ensure alignment: Use the length of the shorter array if necessary
            min_len_qrs = min(len(q_peaks), len(s_peaks))
            for i in range(min_len_qrs):
                q_peak = q_peaks[i]
                s_peak = s_peaks[i]
                if not (np.isnan(q_peak) or np.isnan(s_peak)):
                    # Add +1 like extract_numeric_features, convert to ms
                    if s_peak > q_peak:
                        duration = ((s_peak - q_peak + 1) / sampling_rate) * 1000
                        qrs_duration_ms.append(duration)

            # QT Interval (ms) - Using Q-peak to T-offset
            qt_intervals_ms = []
            min_len_qt = min(len(q_peaks), len(t_offsets))
            for i in range(min_len_qt):
                q_peak = q_peaks[i]
                t_offset = t_offsets[i]
                if not (np.isnan(q_peak) or np.isnan(t_offset)):
                     # Add +1 like extract_numeric_features, convert to ms
                    interval = ((t_offset - q_peak + 1) / sampling_rate) * 1000
                    qt_intervals_ms.append(interval)

            # QTc calculation (ms) - Using per-beat RR
            qtc_ms = []
            # Need aligned QT intervals and RR intervals
            min_len_qtc = min(len(qt_intervals_ms), len(r_peaks) - 1) # QT needs preceding RR
            for j in range(min_len_qtc):
                 # Use the j-th calculated QT interval
                qt_ms = qt_intervals_ms[j]
                # Calculate the preceding RR interval in seconds
                rr_sec = (r_peaks[j+1] - r_peaks[j]) / sampling_rate
                if rr_sec > 0 and rr_sec < 2.0: # Avoid division by zero and use RR filter
                    try:
                        # Bazett's formula: QTc = QT / sqrt(RR_sec)
                        qtc = qt_ms / math.sqrt(rr_sec)
                        qtc_ms.append(qtc)
                    except (ZeroDivisionError, ValueError, RuntimeWarning):
                        continue # Skip if calculation fails

            # --- Calculate final median features with physiological filtering ---
            # (Keeping filtering before median as good practice, though extract_numeric_features didn't show it)

            # Calculate Heart Rate from median RR
            median_rr_ms = self._safe_median(rr_intervals_ms, 300, 2000)
            heart_rate = (60 / (median_rr_ms / 1000)) if not np.isnan(median_rr_ms) and median_rr_ms > 0 else np.nan

            all_features = {
                "Heart_Rate": round(float(heart_rate), 2) if not np.isnan(heart_rate) else np.nan,
                "RR_Interval_ms": round(float(median_rr_ms), 2) if not np.isnan(median_rr_ms) else np.nan,
                # "P_Duration_ms": round(float(self._safe_median(p_duration_ms, 20, 120)), 2) if p_duration_ms else np.nan,
                "PR_Interval_ms": round(float(self._safe_median(pr_interval_ms, 80, 400)), 2) if pr_interval_ms else np.nan,
                "QRS_Duration_ms": round(float(self._safe_median(qrs_duration_ms, 80, 240)), 2) if qrs_duration_ms else np.nan, # Using new QRS list
                # "QT_Interval_ms": round(float(self._safe_median(qt_intervals_ms, 100, 500)), 2) if qt_intervals_ms else np.nan,
                "QTc_ms": round(float(self._safe_median(qtc_ms, 300, 600)), 2) if qtc_ms else np.nan # Using new QTc list
            }
            analysis_status = "completed"

        except Exception as e:
            print(f"Error in feature extraction for {lead_name_for_report}: {str(e)}")
            traceback.print_exc()
            return {
                **all_features,
                "analysis_status": "failed",
                "note": f"Error in feature extraction: {str(e)}"
            }

        # --- Add reference ranges and interpretations (remain the same) ---
        reference_ranges = {
            "Heart_Rate_Normal_Range": "Roughly 60-100 BPM",
            "PR_Interval_Normal_Range": "Roughly 120-200 ms",
            "QRS_Duration_Normal_Range": "Roughly 80-120 ms", # Standard range, even if calculated differently
            "QTc_Normal_Range": "≤ 460 ms",
            "Lead_Used": lead_name_for_report
        }

        interpretations = {}
        # Heart Rate interpretation
        hr = all_features["Heart_Rate"]
        if np.isnan(hr):
            interpretations["Heart_Rate_Interpretation"] = "Unable to determine"
        elif hr < 60:
            interpretations["Heart_Rate_Interpretation"] = "Bradycardia"
        elif hr > 100:
            interpretations["Heart_Rate_Interpretation"] = "Tachycardia"
        else:
            interpretations["Heart_Rate_Interpretation"] = "Normal"

        # PR Interval interpretation
        pr = all_features["PR_Interval_ms"]
        if np.isnan(pr):
            interpretations["PR_Interval_Interpretation"] = "Unable to determine"
        elif pr < 120:
            interpretations["PR_Interval_Interpretation"] = "Short PR interval"
        elif pr > 200:
            interpretations["PR_Interval_Interpretation"] = "Prolonged PR interval (First-degree AV block)"
        else:
            interpretations["PR_Interval_Interpretation"] = "Normal"

        # QTc interpretation
        qtc = all_features["QTc_ms"]
        if np.isnan(qtc):
            interpretations["QTc_Interpretation"] = "Unable to determine"
        elif qtc > 460: # Using 460ms as a common threshold for prolonged QTc in women, slightly lower for men often used
            interpretations["QTc_Interpretation"] = "Prolonged (Increased risk of arrhythmias)"
        else:
            interpretations["QTc_Interpretation"] = "Normal"

        # QRS Duration interpretation
        qrs = all_features["QRS_Duration_ms"]
        if np.isnan(qrs):
            interpretations["QRS_Duration_Interpretation"] = "Unable to determine"
        elif qrs > 120: # Standard threshold for wide QRS
            interpretations["QRS_Duration_Interpretation"] = "Wide QRS (potential bundle branch block or ventricular origin)"
        elif qrs < 80: # Adjusted lower bound based on common ranges
             interpretations["QRS_Duration_Interpretation"] = "Narrow QRS (typically normal)"
        else:
            interpretations["QRS_Duration_Interpretation"] = "Normal"

        # Add analysis status
        interpretations["Analysis_Status"] = f"{lead_name_for_report} analysis {analysis_status}"

        # Combine everything
        result = {**all_features, **reference_ranges, **interpretations}

        return result

    
    def _run(
        self,
        ecg_path: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
        """Process the ECG signal and extract features with robust error handling.

        Args:
            ecg_path (str): The absolute path to the ECG signal file (MAT format).
            run_manager (Optional[CallbackManagerForToolRun]): The callback manager for the tool run.

        Returns:
            Dict[str, Any]: A dictionary containing ECG features and their measurements.

        Raises:
            Exception: If there's an error processing the ECG or during feature extraction.
        """
        print(f"Using ECG file for analysis: {ecg_path}")
        
        if not os.path.exists(ecg_path):
            return {
                "error": f"ECG file not found: {ecg_path}",
                "analysis_status": "failed",
                "note": "Could not locate the specified ECG file"
            }
        
        try:
            # Load and process the ECG signal
            ecg_signal = self._process_ecg_mat(ecg_path)
            
            # Extract features with robust NaN handling
            features = self._extract_features(ecg_signal)
            
            # Add metadata
            result = {
                **features,
                "ecg_path": ecg_path,
                "analysis_status": "completed",
                "note": "All interval measurements are in milliseconds (ms), heart rate in beats per minute (BPM)."
            }
            
            return result
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {
                "error": str(e),
                "ecg_path": ecg_path,
                "analysis_status": "failed",
                "note": f"Analysis failed due to: {str(e)}"
            }
    
    async def _arun(
        self,
        ecg_path: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
        """Asynchronously process the ECG signal and extract features.
        
        This method currently calls the synchronous version, as the processing
        is not inherently asynchronous.
        
        Args:
            ecg_path (str): The path to the ECG signal file.
            run_manager (Optional[AsyncCallbackManagerForToolRun]): The async callback manager for the tool run.
        
        Returns:
            Dict[str, Any]: A dictionary containing ECG features and their measurements.
        """
        return self._run(ecg_path)

# SCP-ECG class labels for 12-lead ECG classification
CLASS_LABELS = [
    '1AVB', '2AVB', '3AVB', 'ABQRS', 'AFIB', 'AFLT', 'ALMI', 'AMI', 'ANEUR', 'ASMI',
    'BIGU', 'CLBBB', 'CRBBB', 'DIG', 'EL', 'HVOLT', 'ILBBB', 'ILMI', 'IMI', 'INJAL',
    'INJAS', 'INJIL', 'INJIN', 'INJLA', 'INVT', 'IPLMI', 'IPMI', 'IRBBB', 'ISCAL',
    'ISCAN', 'ISCAS', 'ISCIL', 'ISCIN', 'ISCLA', 'ISC_', 'IVCD', 'LAFB', 'LAO/LAE',
    'LMI', 'LNGQT', 'LOWT', 'LPFB', 'LPR', 'LVH', 'LVOLT', 'NDT', 'NORM', 'NST_',
    'NT_', 'PAC', 'PACE', 'PMI', 'PRC(S)', 'PSVT', 'PVC', 'QWAVE', 'RAO/RAE', 'RVH',
    'SARRH', 'SBRAD', 'SEHYP', 'SR', 'STACH', 'STD_', 'STE_', 'SVARR', 'SVTAC',
    'TAB_', 'TRIGU', 'VCLVH', 'WPW'
]
   
class ECGClassifierTool(BaseTool):
    """Tool that classifies raw 12-lead ECG signals for multiple SCP-ECG statements.

    This tool uses a fine-tuned model to analyze 12-lead ECG signals and
    predict the likelihood of various SCP-ECG statements.
    """
    name: str = "ecg_classifier"
    description: str = (
        "A tool that analyzes 12-lead ECG signals and classifies them for various cardiac conditions. "
        "Input should be the path to an ECG signal file (.mat format). "
        "Output is a dictionary of SCP-ECG statements and their predicted probabilities (0 to 1). "
        "Higher values indicate a higher likelihood of the condition being present."
    )
    args_schema: Type[BaseModel] = ECGInput
    _model: Any = PrivateAttr()
    _device: Optional[str] = PrivateAttr(default="cuda")
    _class_labels: List[str] = PrivateAttr()

    def __init__(self, model_path: str, device: Optional[str] = "cuda"):
        """Initialize the ECG classifier tool.

        Args:
            model_path: Path to the fine-tuned model checkpoint (required).
            device: Device to run inference on ('cuda' or 'cpu').
        """
        super().__init__()

        if not model_path:
            raise ValueError("model_path is required for ECGClassifierTool")

        print(f"ECGClassifierTool: Loading model from '{model_path}'")

        # Store the class labels
        self._class_labels = CLASS_LABELS
        
        # Initialize the model
        self._device = torch.device(device) if device and torch.cuda.is_available() else torch.device("cpu")
        try:
            model, cfg, task = checkpoint_utils.load_model_and_task(model_path)
            self._model = model
            self._model.eval()
            self._model = self._model.to(self._device)
        except Exception as e:
            print(f"Error loading model from {model_path}: {e}")
            traceback.print_exc()
            self._model = None


    def load_specific_leads(self, feats, leads_to_load):
        feats = feats[leads_to_load]
        padded = torch.zeros((12, feats.size(-1)))
        padded[leads_to_load] = feats
        feats = padded

        return feats

    def get_lead_index(self, lead: Union[int, str]) -> int:
        if isinstance(lead, int):
            return lead
        lead = lead.lower()
        order = ['i', 'ii', 'iii', 'avr', 'avl', 'avf', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6']
        try:
            index = order.index(lead)
        except ValueError:
            raise ValueError(
                "Please make sure that the lead indicator is correct"
            )
        return index

    def postprocess(self, feats, curr_sample_rate=None, leads_to_load=None):
        """
        Process ECG features by selecting specific leads.
        
        Args:
            feats: The ECG features tensor.
            curr_sample_rate: The current sample rate of the ECG.
            leads_to_load: The leads to load, either as a comma-separated string or a list of indices.
        
        Returns:
            torch.Tensor: Processed ECG features.
        """
        if leads_to_load is not None:
            # Check if leads_to_load is a string (comma-separated values)
            if isinstance(leads_to_load, str):
                leads_to_load = leads_to_load.split(',')
                leads_to_load = list(map(self.get_lead_index, leads_to_load))
        else:
            # Default to all 12 leads
            leads_to_load = list(range(12))
        
        feats = feats.float()
        feats = self.load_specific_leads(feats, leads_to_load=leads_to_load)

        return feats

    def _process_ecg(self, ecg_path: str, lead_context: str = "12-lead") -> dict:
        """Process the input 12-lead ECG signal for model inference.

        Args:
            ecg_path: The file path to the ECG signal (.mat file).
            lead_context: Lead configuration (default: "12-lead").

        Returns:
            dict: A sample dictionary with net_input ready for model inference.
        """
        # Load all 12 leads
        leads_to_load = list(range(12))

        ecg = scipy.io.loadmat(ecg_path)
        curr_sample_rate = ecg['curr_sample_rate']
        feats = torch.from_numpy(ecg['feats'])

        final_ecg = self.postprocess(feats, curr_sample_rate, leads_to_load=leads_to_load)

        final_ecg = final_ecg.unsqueeze(0).to(self._device)
        ecg_padding_mask = torch.zeros(1, 12, 5000, dtype=torch.bool).to(self._device)

        sample = {
            "net_input": {
                "source": final_ecg,
                "padding_mask": ecg_padding_mask
            }
        }

        return sample

    def _run(
        self,
        ecg_path: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Tuple[Dict[str, float], Dict]:
        """Classify the 12-lead ECG signal for multiple pathologies.

        Args:
            ecg_path: Absolute path to the ECG .mat file.
            run_manager: Optional callback manager.

        Returns:
            Tuple of (classification_results, metadata).
        """
        # Check if model loaded successfully
        if self._model is None:
            return {"error": "Model failed to load"}, {
                "ecg_path": ecg_path,
                "analysis_status": "failed"
            }

        print(f"Using ECG file: {ecg_path}")

        if not os.path.exists(ecg_path):
            return {"error": f"ECG file not found: {ecg_path}"}, {
                "ecg_path": ecg_path,
                "analysis_status": "failed"
            }

        try:
            # Process the ECG and run the model (12-lead)
            ecg = self._process_ecg(ecg_path, "12-lead")

            with torch.inference_mode():
                net_output = self._model(**ecg["net_input"])
                probs = torch.sigmoid(net_output['out']).squeeze(0)

            # Get all outputs for 12-lead classification
            outputs = {label: round(prob.item(), 4) for label, prob in zip(self._class_labels, probs)}

            metadata = {
                "ecg_path": ecg_path,
                "analysis_status": "completed",
                "note": "12-lead ECG classification. Higher values indicate higher likelihood of the condition."
            }
            return outputs, metadata
        except Exception as e:
            print(f"Error classifying ECG: {e}")
            traceback.print_exc()
            return {"error": str(e)}, {
                "ecg_path": ecg_path,
                "analysis_status": "failed"
            }

    async def _arun(
        self,
        ecg_path: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Tuple[Dict[str, float], Dict]:
        """Asynchronously classify the 12-lead ECG signal."""
        return self._run(ecg_path)

