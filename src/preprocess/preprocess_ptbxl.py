import argparse
import os
import pandas as pd
import wfdb
import scipy.io
import numpy as np
import pdb 

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root", metavar="DIR",
        help="root directory containing data files to pre-process"
    )
    parser.add_argument(
        "--dest", type=str, metavar="DIR",
        help="output directory"
    )
    parser.add_argument(
        "--leads",
        default="0,1,2,3,4,5,6,7,8,9,10,11",
        type=str,
        help="comma separated list of lead numbers. (e.g. 0,1 loads only lead I and lead II) "
        "note that the order is following: [I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6]"
    )
    parser.add_argument(
        "--sample-rate",
        default=500,
        type=int,
        help="if set, data must be sampled by this sampling rate to be processed"
    )
    parser.add_argument("--seed", default=42, type=int, metavar="N", help="random seed")

    return parser

def main(args):
    dir_path = os.path.realpath(args.root)
    dest_path = os.path.realpath(args.dest)

    if not os.path.exists(dest_path):
        os.makedirs(dest_path)

    leads = args.leads.replace(' ','').split(',')
    leads_to_load = [int(lead) for lead in leads]

    np.random.seed(args.seed)

    # Load the CSV and create a mapping from ecg_id to patient_id
    csv = pd.read_csv('./ptbxl_files/ptbxl_database.csv')
    ecg_id_to_patient = dict(zip(csv['ecg_id'].astype(str), csv['patient_id'].astype(str)))

    for root, subdirs, _ in os.walk(dir_path):
        for subdir in subdirs:
            subdir_path = os.path.join(root, subdir)
            for file in os.listdir(subdir_path):
                if file.endswith(".hea"):  # Process only .hea files (WFDB records)
                    ecg_id = str(int(file[:-7]))  # Extract ECG ID from filename
                    record_path = os.path.join(subdir_path, file[:-4])  # Remove .hea extension

                    if ecg_id not in ecg_id_to_patient:
                        print(f"Warning: ECG ID {ecg_id} not found in CSV, skipping.")
                        continue

                    patient_id = ecg_id_to_patient[ecg_id]  # Retrieve patient_id

                    try:
                        record, metadata = wfdb.rdsamp(record_path)  # Read WFDB format
                        sample_rate = metadata['fs']
                        
                        record = record.T  # Transpose to match expected format

                        if args.sample_rate and sample_rate != args.sample_rate:
                            continue

                        if np.isnan(record).any():
                            print(f"Detected NaN values in: {record_path}, skipping.")
                            continue

                        data = {
                            "curr_sample_rate": sample_rate,
                            "feats": record[leads_to_load, :],
                            "patient_id": patient_id  # Include patient ID
                        }

                        output_filename = f"HR{ecg_id}.mat"
                        scipy.io.savemat(os.path.join(dest_path, output_filename), data)
                        print(f"Saved {output_filename}")

                    except Exception as e:
                        print(f"Error processing {record_path}: {e}")

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
