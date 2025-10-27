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
# sys.path.append("/nfs_edlab/hschung/fairseq-signals")
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
    
    def __init__(self, 
                 simulated_lead_context: str = "12-lead", 
                 device: Optional[str] = "cuda"):
        super().__init__()
        self._simulated_lead_context = simulated_lead_context
        self._device = device # Store device if it were to be used by this tool's methods

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
        import numpy as np 
        """Extract ECG features using neurokit2, selecting lead based on context."""
        # Define lead names
        lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

        # Determine which lead to use based on simulated_lead_context
        lead_idx = 1  # Default to Lead II
        lead_name_for_report = "Lead II"
        if hasattr(self, '_simulated_lead_context') and self._simulated_lead_context == "single-lead-I":
            lead_idx = 0  # Use Lead I
            lead_name_for_report = "Lead I"
            print(f"ECGAnalysisTool: Using Lead I (index 0) for analysis due to context: {self._simulated_lead_context}")
        else:
            print(f"ECGAnalysisTool: Using {lead_name_for_report} (index {lead_idx}) for analysis. Context: {getattr(self, '_simulated_lead_context', 'N/A')}")

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
            ecg_path (str): The path to the ECG signal file (MAT format).
            run_manager (Optional[CallbackManagerForToolRun]): The callback manager for the tool run.
        
        Returns:
            Dict[str, Any]: A dictionary containing ECG features and their measurements.
        
        Raises:
            Exception: If there's an error processing the ECG or during feature extraction.
        """
        # Check if the file exists
        PROCESSED_ECG_DIR = "/nfs_edlab/hschung/preprocess_classification/ptbxl_10s"
        
        # First check if the ecg_path is a full path and exists
        if not os.path.exists(ecg_path):
            # If it's just a filename or partial path, try different methods to find the full path
            
            # Method 1: Check if the path exists in the thread state
            if hasattr(self, 'agent'):
                try:
                    thread_state = self.agent.workflow.get_state()
                    if thread_state and "configurable" in thread_state and "ecg_files" in thread_state["configurable"]:
                        for file_path in thread_state["configurable"]["ecg_files"]:
                            if os.path.basename(file_path) == os.path.basename(ecg_path) or ecg_path in file_path:
                                ecg_path = file_path
                                break
                except Exception as e:
                    print(f"Warning: Unable to locate ECG file from thread state: {e}")
            
            # Method 2: Try to find the file in the processed directory
            if not os.path.exists(ecg_path):
                # Try different filename patterns
                basename = os.path.basename(ecg_path)
                
                # Pattern 1: Direct filename match
                direct_path = os.path.join(PROCESSED_ECG_DIR, basename)
                if os.path.exists(direct_path):
                    ecg_path = direct_path
                
                # Pattern 2: HR00123.mat format (if input is 00123_hr or similar)
                elif "_hr" in basename:
                    hr_number = basename.replace("_hr", "").replace(".mat", "")
                    hr_path = os.path.join(PROCESSED_ECG_DIR, f"HR{hr_number}.mat")
                    if os.path.exists(hr_path):
                        ecg_path = hr_path
                
                # Pattern 3: Handle numeric formats (just the patient ID number)
                else:
                    # Remove non-numeric characters and try to form HR filename
                    numeric_only = ''.join(filter(str.isdigit, basename))
                    if numeric_only:
                        hr_path = os.path.join(PROCESSED_ECG_DIR, f"HR{numeric_only.zfill(5)}.mat")
                        if os.path.exists(hr_path):
                            ecg_path = hr_path
        
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

# Define which abnormalities can be detected with different lead configurations
TWELVE_LEAD_ABNORMALITIES = ['1AVB', '2AVB', '3AVB', 'ABQRS', 'AFIB', 'AFLT', 'ALMI', 'AMI', 'ANEUR', 'ASMI', 'BIGU', 'CLBBB', 'CRBBB', 'DIG', 'EL', 'HVOLT', 'ILBBB', 'ILMI', 'IMI', 'INJAL', 'INJAS', 'INJIL', 'INJIN', 'INJLA', 'INVT', 'IPLMI', 'IPMI', 'IRBBB', 'ISCAL', 'ISCAN', 'ISCAS', 'ISCIL', 'ISCIN', 'ISCLA', 'ISC_', 'IVCD', 'LAFB', 'LAO/LAE', 'LMI', 'LNGQT', 'LOWT', 'LPFB', 'LPR', 'LVH', 'LVOLT', 'NDT', 'NORM', 'NST_', 'NT_', 'PAC', 'PACE', 'PMI', 'PRC(S)', 'PSVT', 'PVC', 'QWAVE', 'RAO/RAE', 'RVH', 'SARRH', 'SBRAD', 'SEHYP', 'SR', 'STACH', 'STD_', 'STE_', 'SVARR', 'SVTAC', 'TAB_', 'TRIGU', 'VCLVH', 'WPW']

# Common abnormalities detectable across different lead configurations
SHARED_ABNORMALITIES = ['SR', 'PAC', 'ABQRS', 'SVARR', 'LVH', '1AVB', '2AVB', '3AVB', 'AFIB', 'AFLT', 'AMI', 'BIGU', 'CLBBB', 'CRBBB', 'DIG', 'LPR', 'PACE', 'PRC(S)', 'PSVT', 'RVH', 'SBRAD', 'STACH', 'SVTAC', 'TRIGU', 'WPW']

# Lead-specific detectable abnormalities
DETECTABLE_ABNORMALITIES = {
    "single-lead-I": ['SR', 'PAC', 'ABQRS', '1AVB', '2AVB', '3AVB', 'AFIB', 'AFLT', 'BIGU', 'CLBBB', 'CRBBB', 'LAFB', 'LPR', 'PACE', 'PSVT', 'RVH', 'SBRAD', 'STACH', 'SVTAC', 'TRIGU', 'WPW', 'STD_', 'STE_', 'PVC'],
    "single-lead-II": ['SR', 'LAFB', 'LNGQT', '1AVB', '2AVB', '3AVB', 'AFIB', 'AFLT', 'BIGU', 'CLBBB', 'CRBBB', 'DIG', 'ILBBB', 'LPR', 'PACE', 'PRC(S)', 'PSVT', 'RVH', 'SBRAD', 'STACH', 'SVARR', 'SVTAC', 'TRIGU', 'WPW', 'PAC', 'STD_', 'STE_', 'PVC'],
    "two-leads": SHARED_ABNORMALITIES,
    "three-leads": SHARED_ABNORMALITIES,
    "six-leads": SHARED_ABNORMALITIES,
    "12-lead": TWELVE_LEAD_ABNORMALITIES
}

# Map from lead contexts to model directory names
LEAD_CONTEXT_TO_MODEL_SEGMENT = {
    "single-lead-I": "lead-i",
    "single-lead-II": "lead-ii",
    "two-leads": "two-leads",
    "three-leads": "three-leads",
    "six-leads": "six-leads",
    "12-lead": "twelve-leads"
}
DEFAULT_MODEL_SEGMENT = "twelve-leads"  # Default model if context not found
   
class ECGClassifierTool(BaseTool):
    """Tool that classifies raw Electrocardiogram (ECG) signals for multiple SCP-ECG statements.

    This tool uses a pre-trained W2V 2.0 + CMSC + RLM model to analyze ECG signals and
    predict the likelihood of various SCP-ECG statements based on the available lead context.
    """
    name: str = "ecg_classifier"
    description: str = (
        "A tool that analyzes ECG signals and classifies them based on the available lead configuration. "
        "Input should be the path to an ECG signal file. "
        "Output is a dictionary of SCP-ECG statements and their predicted probabilities (0 to 1), "
        "filtered to only include abnormalities that can be reliably detected with the available leads. "
        "Higher values indicate a higher likelihood of the condition being present."
    )
    args_schema: Type[BaseModel] = ECGInput
    _model: Any = PrivateAttr()
    _device: Optional[str] = PrivateAttr(default="cuda")
    _simulated_lead_context: str = PrivateAttr(default="12-lead")
    _class_labels: List[str] = PrivateAttr()

    def __init__(self, 
                 simulated_lead_context: str = "12-lead", 
                 model_path: str = None, 
                 device: Optional[str] = "cuda"):
        super().__init__()
        
        # Store the lead context
        self._simulated_lead_context = simulated_lead_context
        
        # Determine model path based on lead context if not provided
        if not model_path:
            model_segment = LEAD_CONTEXT_TO_MODEL_SEGMENT.get(
                self._simulated_lead_context, 
                DEFAULT_MODEL_SEGMENT
            )
            model_path = f"/nfs_edlab/hschung/fairseq-signals/outputs/2025-02-26/{model_segment}-ptbxl-10s/checkpoints/checkpoint_best.pt"
        
        print(f"ECGClassifierTool: Using lead context '{self._simulated_lead_context}', loading model from '{model_path}'")
        
        # Store the class labels
        self._class_labels = ['1AVB', '2AVB', '3AVB', 'ABQRS', 'AFIB', 'AFLT', 'ALMI', 'AMI', 'ANEUR', 'ASMI', 'BIGU', 'CLBBB', 'CRBBB', 'DIG', 'EL', 'HVOLT', 'ILBBB', 'ILMI', 'IMI', 'INJAL', 'INJAS', 'INJIL', 'INJIN', 'INJLA', 'INVT', 'IPLMI', 'IPMI', 'IRBBB', 'ISCAL', 'ISCAN', 'ISCAS', 'ISCIL', 'ISCIN', 'ISCLA', 'ISC_', 'IVCD', 'LAFB', 'LAO/LAE', 'LMI', 'LNGQT', 'LOWT', 'LPFB', 'LPR', 'LVH', 'LVOLT', 'NDT', 'NORM', 'NST_', 'NT_', 'PAC', 'PACE', 'PMI', 'PRC(S)', 'PSVT', 'PVC', 'QWAVE', 'RAO/RAE', 'RVH', 'SARRH', 'SBRAD', 'SEHYP', 'SR', 'STACH', 'STD_', 'STE_', 'SVARR', 'SVTAC', 'TAB_', 'TRIGU', 'VCLVH', 'WPW']
        
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

    def _process_ecg(self, ecg_path: str, _simulated_lead_context: str) -> torch.Tensor:
        """
        Process the input ECG signal for model inference.

        This method loads the ECG and prepares it as a torch.Tensor for model input.

        Args:
            ECG_path (str): The file path to the ECG signal.

        Returns:
            torch.Tensor: A processed image tensor ready for model inference.

        Raises:
            FileNotFoundError: If the specified image file does not exist.
            ValueError: If the image cannot be properly loaded or processed.
        """
        lead_indices = {
            "single-lead-I": [0],
            "single-lead-II": [1],
            "two-leads": [0, 1],
            "three-leads": [0, 1, 2],
            "six-leads": [0, 1, 2, 3, 4, 5],
            "12-lead": list(range(12))  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        }

        # Get the appropriate leads for the current context
        leads_to_load = lead_indices.get(_simulated_lead_context, list(range(12)))

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
        """Classify the ECG signal for multiple pathologies based on lead context."""
        
        # Check if model loaded successfully
        if self._model is None:
            return {"error": f"Model for lead context '{self._simulated_lead_context}' failed to load"}, {
                "ecg_path": ecg_path,
                "analysis_status": "failed",
                "lead_context": self._simulated_lead_context
            }
        
        PROCESSED_ECG_DIR = "/nfs_edlab/hschung/preprocess_classification/ptbxl_10s"
        
        # First check if the ecg_path is a full path and exists
        if not os.path.exists(ecg_path):
            # If it's just a filename or partial path, try different methods to find the full path
            
            # Method 1: Check if the path exists in the thread state
            if hasattr(self, 'agent'):
                try:
                    thread_state = self.agent.workflow.get_state()
                    if thread_state and "configurable" in thread_state and "ecg_files" in thread_state["configurable"]:
                        for file_path in thread_state["configurable"]["ecg_files"]:
                            if os.path.basename(file_path) == os.path.basename(ecg_path) or ecg_path in file_path:
                                ecg_path = file_path
                                break
                except Exception as e:
                    print(f"Warning: Unable to locate ECG file from thread state: {e}")
            
            # Method 2: Try to find the file in the processed directory
            if not os.path.exists(ecg_path):
                # Try different filename patterns
                basename = os.path.basename(ecg_path)
                
                # Pattern 1: Direct filename match
                direct_path = os.path.join(PROCESSED_ECG_DIR, basename)
                if os.path.exists(direct_path):
                    ecg_path = direct_path
                
                # Pattern 2: HR00123.mat format (if input is 00123_hr or similar)
                elif "_hr" in basename:
                    hr_number = basename.replace("_hr", "").replace(".mat", "")
                    hr_path = os.path.join(PROCESSED_ECG_DIR, f"HR{hr_number}.mat")
                    if os.path.exists(hr_path):
                        ecg_path = hr_path
                
                # Pattern 3: Handle numeric formats (just the patient ID number)
                else:
                    # Remove non-numeric characters and try to form HR filename
                    numeric_only = ''.join(filter(str.isdigit, basename))
                    if numeric_only:
                        hr_path = os.path.join(PROCESSED_ECG_DIR, f"HR{numeric_only.zfill(5)}.mat")
                        if os.path.exists(hr_path):
                            ecg_path = hr_path
        
        print(f"Using ECG file: {ecg_path} with lead context: {self._simulated_lead_context}")
        
        if not os.path.exists(ecg_path):
            return {"error": f"ECG file not found: {ecg_path}"}, {
                "ecg_path": ecg_path,
                "analysis_status": "failed",
                "lead_context": self._simulated_lead_context
            }
        
        try:
            # Process the ECG and run the model
            ecg = self._process_ecg(ecg_path, self._simulated_lead_context)

            with torch.inference_mode():
                net_output = self._model(**ecg["net_input"])
                probs = torch.sigmoid(net_output['out']).squeeze(0)

            # First, get all outputs
            all_outputs = {label: round(prob.item(), 4) for label, prob in zip(self._class_labels, probs)}
            
            # Then, filter based on lead context
            detectable_abnormalities = DETECTABLE_ABNORMALITIES.get(
                self._simulated_lead_context, 
                TWELVE_LEAD_ABNORMALITIES
            )
            
            # Filter to only include abnormalities that can be detected with this lead configuration
            filtered_outputs = {k: v for k, v in all_outputs.items() if k in detectable_abnormalities}
            
            metadata = {
                "ecg_path": ecg_path,
                "analysis_status": "completed",
                "lead_context": self._simulated_lead_context,
                "note": f"Results filtered for {self._simulated_lead_context} lead configuration. Higher values indicate higher likelihood of the condition.",
                "detectable_abnormalities_count": len(detectable_abnormalities)
            }
            return filtered_outputs, metadata
        except Exception as e:
            print(f"Error classifying ECG: {e}")
            traceback.print_exc()
            return {"error": str(e)}, {
                "ecg_path": ecg_path,
                "analysis_status": "failed",
                "lead_context": self._simulated_lead_context
            }

    async def _arun(
        self,
        ecg_path: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Tuple[Dict[str, float], Dict]:
        """Asynchronously classify the ecg signal for multiple SCP-ECG statements.

        This method currently calls the synchronous version, as the model inference
        is not inherently asynchronous. For true asynchronous behavior, consider
        using a separate thread or process.

        Args:
            ecg_path (str): The path to the ecg signal file.
            run_manager (Optional[AsyncCallbackManagerForToolRun]): The async callback manager for the tool run.

        Returns:
            Tuple[Dict[str, float], Dict]: A tuple containing the classification results
                                           (pathologies and their probabilities from 0 to 1)
                                           and any additional metadata.

        Raises:
            Exception: If there's an error processing the image or during classification.
        """
        return self._run(ecg_path)

class ECGExplainInput(ECGInput):
    """Input for the ECG Explanation tool."""
    target_class: Optional[str] = Field(
        None, description="The specific class to explain. If None, the highest probability class will be used."
    )

class ECGExplainTool(BaseTool):
    """Tool that provides explanations for ECG classifications using SpectralX method.
    
    This tool uses FIA-combined method to explain ECG predictions by identifying
    important time-frequency regions in the spectrogram that contribute to the classification.
    """
    name: str = "ecg_explanation"
    description: str = (
        "A tool that explains ECG classification results by identifying important time-frequency regions "
        "in the spectrogram that contribute to the prediction. Input should be the path to an ECG signal file. "
        "Output includes the predicted class, probability, and explanation of which time and frequency ranges "
        "are most important for the classification decision. Use this tool when you need to understand "
        "why a specific classification was made or which parts of the ECG signal are most relevant."
    )
    args_schema: Type[BaseModel] = ECGExplainInput
    _classifier_tool: Any = PrivateAttr()
    _device: Optional[str] = PrivateAttr(default="cuda")
    _simulated_lead_context: str = PrivateAttr(default="12-lead")
    
    def __init__(self, 
                 simulated_lead_context: str = "12-lead", 
                 device: Optional[str] = "cuda"):
        super().__init__()
        
        # Import required modules
        import sys
        sys.path.insert(0, "/nfs_edlab/hschung/MedRAX/Time_is_not_Enough")
        
        try:
            import data_loader
            from models.spectralx import XAITrainer
            from scipy.signal import stft
            
            self._data_loader = data_loader
            self._XAITrainer = XAITrainer
            self._stft = stft
            
        except ImportError as e:
            print(f"Warning: SpectralX dependencies not available: {e}")
            self._data_loader = None
            self._XAITrainer = None
            self._stft = None
        
        self._simulated_lead_context = simulated_lead_context
        self._device = device
        
        # Initialize the ECG classifier tool for getting predictions
        self._classifier_tool = ECGClassifierTool(
            simulated_lead_context=simulated_lead_context,
            device=device
        )
    
    def _get_lead_for_explanation(self) -> int:
        """Get the appropriate lead index for explanation based on context."""
        lead_context_to_idx = {
            "single-lead-I": 0,    # Lead I
            "single-lead-II": 1,   # Lead II
            "12-lead": 1          # Default to lead-II for 12-lead
        }
        return lead_context_to_idx.get(self._simulated_lead_context, 1)
    
    def _explain_ecg_with_fia_combined(self, ecg_path: str, target_label_idx: int, topk: int = 1, num_perturbations: int = 50):
        """Use FIA-combined method to explain ECG predictions for a specific label."""
        if not self._XAITrainer or not self._data_loader:
            raise ImportError("SpectralX dependencies not available")
        
        # Set seed for reproducible explanations
        import random
        import numpy as np
        import torch
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        
        # Get prediction
        predictions, metadata = self._classifier_tool._run(ecg_path)
        class_labels = self._classifier_tool._class_labels
        
        if "error" in predictions:
            raise ValueError(f"Error in ECG classification: {predictions['error']}")
        
        target_label_name = class_labels[target_label_idx]
        target_prob = predictions.get(target_label_name, 0.0)
        
        lead_for_explanation_idx = self._get_lead_for_explanation()
        
        # Set up arguments for XAI
        class Args:
            def __init__(self, simulated_lead_context="12-lead"):
                self.model_framework = 'fairseq'
                self.classification_model = None
                self.dataset = "custom_single_ptbxl_mat_for_explain"
                self.batch_size = 1
                self.label = target_label_idx
                self.lead_for_explanation_idx = lead_for_explanation_idx
                self.num_perturbations = num_perturbations
                self.selected_regions = 1
                self.topk = topk
                self.method = 'combined'
                self.insertion_weight = 0.8
                self.deletion_weight = 0.2
                self.fs = 500
                self.nperseg = 512
                self.noverlap = 384
                self.original_mat_fs = 500
                self.simulated_lead_context = simulated_lead_context
        
        args = Args(simulated_lead_context=self._simulated_lead_context)
        
        # Set global variable for data loader
        self._data_loader.SINGLE_MAT_FILE_PATH_FOR_LOADER = ecg_path
        
        # Initialize XAI trainer
        trainer = self._XAITrainer(args, self._classifier_tool._model)
        
        # Run FIA-combined explanation
        selected_positions = []
        positions_consider = []
        
        for i in range(topk + 1):
            selected_positions, positions_consider = trainer.combined(selected_positions, positions_consider)
            if len(selected_positions) >= topk:
                break
        
        return {
            'important_regions': selected_positions,
            'target_label': target_label_name,
            'target_label_idx': target_label_idx,
            'target_probability': target_prob,
            'lead_explained': lead_for_explanation_idx,
            'explanation_method': 'fia-combined'
        }
    
    def _extract_spectrogram_pixel_info(self, explanation_result, ecg_path, lead_idx):
        """Extract spectrogram pixel information for the important regions."""
        try:
            # Load ECG data
            mat_data = scipy.io.loadmat(ecg_path)
            
            # Find ECG data
            ecg_data = None
            possible_keys = ['val', 'data', 'feats', 'signal', 'ecg', 'x', 'y']
            
            for key in possible_keys:
                if key in mat_data:
                    ecg_data = mat_data[key]
                    break
            
            if ecg_data is None:
                keys = [k for k in mat_data.keys() if not k.startswith('__')]
                if keys:
                    ecg_data = mat_data[keys[0]]
                else:
                    raise ValueError("No data found in .mat file")
            
            # Process ECG data to get single lead
            ecg_data = np.squeeze(ecg_data)
            
            if ecg_data.ndim == 1:
                original_signal = ecg_data
            elif ecg_data.ndim == 2:
                if ecg_data.shape[0] <= 12:  # Leads are rows
                    if lead_idx >= ecg_data.shape[0]:
                        lead_idx = 0
                    original_signal = ecg_data[lead_idx, :]
                else:  # Leads are columns or time x leads
                    if ecg_data.shape[1] <= 12:
                        if lead_idx >= ecg_data.shape[1]:
                            lead_idx = 0
                        original_signal = ecg_data[:, lead_idx]
                    else:
                        original_signal = ecg_data[0, :] if ecg_data.shape[0] < ecg_data.shape[1] else ecg_data[:, 0]
            else:
                original_signal = ecg_data.flatten()
            
            # Compute STFT with appropriate parameters
            signal_length = len(original_signal)
            if signal_length < 512:
                nperseg = min(64, signal_length // 4)
            elif signal_length < 2048:
                nperseg = min(128, signal_length // 4)
            else:
                nperseg = 256
            
            noverlap = nperseg // 2
            nperseg = max(8, nperseg)
            noverlap = min(noverlap, nperseg - 1)
            
            frequencies, times, Zxx = self._stft(original_signal, fs=500, nperseg=nperseg, noverlap=noverlap)
            
            # Get spectrogram dimensions
            height, width = Zxx.shape
            
            # Process important regions to get pixel coordinates and time-frequency values
            pixel_regions = []
            
            for region_idx, region_group in enumerate(explanation_result['important_regions']):
                region_pixels = []
                
                for flat_idx in region_group:
                    # Convert flat index back to (freq_bin, time_bin)
                    freq_bin = flat_idx // width
                    time_bin = flat_idx % width
                    
                    # Get actual time and frequency values
                    freq_value = frequencies[freq_bin] if freq_bin < len(frequencies) else frequencies[-1]
                    time_value = times[time_bin] if time_bin < len(times) else times[-1]
                    
                    region_pixels.append({
                        'flat_index': flat_idx,
                        'freq_bin': freq_bin,
                        'time_bin': time_bin,
                        'frequency_hz': freq_value,
                        'time_seconds': time_value
                    })
                
                pixel_regions.append(region_pixels)
            
            return {
                'pixel_regions': pixel_regions,
                'spectrogram_shape': (height, width),
                'frequencies': frequencies,
                'times': times
            }
            
        except Exception as e:
            print(f"Error extracting spectrogram pixel info: {e}")
            return None
    
    def _format_explanation_for_output(self, explanation_result, pixel_info):
        """Format explanation information into a readable string."""
        target_label = explanation_result['target_label']
        target_prob = explanation_result['target_probability']
        lead_name = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'][explanation_result['lead_explained']]
        
        explanation_parts = []
        
        if pixel_info and pixel_info['pixel_regions']:
            
            for region_idx, region_pixels in enumerate(pixel_info['pixel_regions']):
                if region_pixels:  # Check if region has pixels
                    # Get ranges for this region
                    times = [pixel['time_seconds'] for pixel in region_pixels]
                    freqs = [pixel['frequency_hz'] for pixel in region_pixels]
                    
                    time_min, time_max = min(times), max(times)
                    freq_min, freq_max = min(freqs), max(freqs)
                    
                    region_str = (f"{target_label} ({target_prob:.2%}) | "
                                 f"time={time_min:.3f}-{time_max:.3f}s, "
                                 f"freq={freq_min:.1f}-{freq_max:.1f}Hz")
                    
                    explanation_parts.append(region_str)
        else:
            # Fallback if no pixel info
            explanation_parts.append(f"{target_label} ({target_prob:.2%}) | explanation unavailable")
        
        result = "\n".join(explanation_parts)
        return result
    
    def _run(
        self,
        ecg_path: str,
        target_class: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
        """Explain ECG classification results using SpectralX method."""
        
        if not ecg_path:
            return {"error": "ECG path was not provided."}
        
        # Check if SpectralX dependencies are available
        if not self._XAITrainer or not self._data_loader:
            return {
                "error": "SpectralX explanation dependencies not available",
                "ecg_path": ecg_path,
                "analysis_status": "failed"
            }
        
        # Handle ECG file path resolution (same as other tools)
        PROCESSED_ECG_DIR = "/nfs_edlab/hschung/preprocess_classification/ptbxl_10s"
        
        if not os.path.exists(ecg_path):
            # Try to find the file in the processed directory
            basename = os.path.basename(ecg_path)
            
            # Pattern 1: Direct filename match
            direct_path = os.path.join(PROCESSED_ECG_DIR, basename)
            if os.path.exists(direct_path):
                ecg_path = direct_path
            
            # Pattern 2: HR00123.mat format
            elif "_hr" in basename:
                hr_number = basename.replace("_hr", "").replace(".mat", "")
                hr_path = os.path.join(PROCESSED_ECG_DIR, f"HR{hr_number}.mat")
                if os.path.exists(hr_path):
                    ecg_path = hr_path
            
            # Pattern 3: Handle numeric formats
            else:
                numeric_only = ''.join(filter(str.isdigit, basename))
                if numeric_only:
                    hr_path = os.path.join(PROCESSED_ECG_DIR, f"HR{numeric_only.zfill(5)}.mat")
                    if os.path.exists(hr_path):
                        ecg_path = hr_path
        
        if not os.path.exists(ecg_path):
            return {
                "error": f"ECG file not found: {ecg_path}",
                "ecg_path": ecg_path,
                "analysis_status": "failed"
            }
        
        try:
            # Get predictions to find highest probability label
            predictions, metadata = self._classifier_tool._run(ecg_path)
            
            if "error" in predictions:
                return {
                    "error": f"Error in ECG classification: {predictions['error']}",
                    "ecg_path": ecg_path,
                    "analysis_status": "failed"
                }
            
            # --- THIS IS THE FIX ---
            # Prioritize the user-specified target_class if it exists and is valid.
            if target_class and target_class in predictions:
                target_label = target_class
                target_prob = predictions[target_label]
                print(f"Explaining user-specified class: {target_label} ({target_prob:.4f})")
            else:
                # Fallback to the highest probability class if target_class is not provided or invalid
                high_prob_labels = [(label, prob) for label, prob in predictions.items() 
                                   if isinstance(prob, float) and label != "error"]
                
                if not high_prob_labels:
                    return {
                        "error": "No valid predictions found",
                        "ecg_path": ecg_path,
                        "analysis_status": "failed"
                    }
                
                high_prob_labels.sort(key=lambda x: x[1], reverse=True)
                target_label, target_prob = high_prob_labels[0]
                print(f"Fallback: Explaining highest probability class: {target_label} ({target_prob:.4f})")
            # --- END OF FIX ---

            # Get class index
            class_labels = self._classifier_tool._class_labels
            if target_label not in class_labels:
                return {
                    "error": f"Target class '{target_label}' not found in model's class labels.",
                    "ecg_path": ecg_path,
                    "analysis_status": "failed"
                }
            target_label_idx = class_labels.index(target_label)
            
            # Get explanation
            explanation_result = self._explain_ecg_with_fia_combined(
                ecg_path,
                target_label_idx,
                topk=1,
                num_perturbations=50
            )
            
            # Extract spectrogram pixel information
            pixel_info = self._extract_spectrogram_pixel_info(
                explanation_result, 
                ecg_path, 
                self._get_lead_for_explanation()
            )
            
            # Format explanation string
            explanation_str = self._format_explanation_for_output(explanation_result, pixel_info)
            
            return explanation_str

            
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {
                "error": str(e),
                "ecg_path": ecg_path,
                "analysis_status": "failed"
            }
    
    async def _arun(
        self,
        ecg_path: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
        """Asynchronously explain ECG classification results."""
        return self._run(ecg_path)
