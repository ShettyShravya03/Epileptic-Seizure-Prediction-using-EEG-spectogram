import os
import re
import pandas as pd
from collections import defaultdict

NUM_CHANNELS = 22
STANDARD_CHANNELS = [
    'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1',
    'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
    'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
    'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2',
    'FZ-CZ', 'CZ-PZ', 'P7-T7', 'T7-FT9',
    'FT9-FT10', 'FT10-T8'
]
CHANNEL_PREFIX = "channel_"  # All images are named like channel_FZ-CZ.png
# PREICTAL_WINDOW = 180  # Placeholder for future use

def parse_seizure_summaries(summary_dir):
    seizure_times = defaultdict(dict)

    for filename in os.listdir(summary_dir):
        if filename.endswith('-summary.txt'):
            patient_id = filename.split('-')[0]
            with open(os.path.join(summary_dir, filename), 'r') as f:
                lines = f.readlines()

            current_file = None
            seizures = []
            start_time = None  # Init outside loop

            for line in lines:
                file_match = re.match(r'File Name:\s*(\S+\.edf)', line)

                # Matches both "Seizure Start Time: 1234" and "Seizure 1 Start Time: 1234"
                start_match = re.match(r'Seizure(?: \d+)? Start Time:\s*(\d+)\s*seconds', line)
                end_match = re.match(r'Seizure(?: \d+)? End Time:\s*(\d+)\s*seconds', line)

                if file_match:
                    if current_file and seizures:
                        seizure_times[patient_id][current_file] = seizures
                    current_file = file_match.group(1)
                    seizures = []

                if start_match:
                    start_time = int(start_match.group(1))
                if end_match and start_time is not None:
                    end_time = int(end_match.group(1))
                    seizures.append((start_time, end_time))
                    start_time = None  # Reset for next seizure

            if current_file and seizures:
                seizure_times[patient_id][current_file] = seizures

    return dict(seizure_times)


def is_preictal(seizure_windows):
    # print("Seizure windows ",seizure_windows)
    return 1 if seizure_windows else 0  # Binary classification: preictal or not

def prepare_dataset(root_dir, summary_dir, output_csv):
    seizure_times = parse_seizure_summaries(summary_dir)
    rows = []

    for patient_id in os.listdir(root_dir):
        patient_path = os.path.join(root_dir, patient_id)
        if not os.path.isdir(patient_path):
            continue

        for edf_folder in os.listdir(patient_path):
            edf_path = os.path.join(patient_path, edf_folder)
            if not os.path.isdir(edf_path):
                continue

            edf_basename = edf_folder
            edf_basename_with_ext = edf_basename + '.edf'
            # print("EDF basename ",edf_basename)
            # print("Patient ID ",patient_id)
            # print("Seizure times ",seizure_times)
            seizure_windows = seizure_times.get(patient_id, {}).get(edf_basename_with_ext, [])

            label = is_preictal(seizure_windows)

            png_files = [f for f in os.listdir(edf_path) if f.endswith('.png')]
            if len(png_files) < NUM_CHANNELS:
                print(f"⚠️ Skipping {edf_path}: only {len(png_files)} channels found.")
                continue

            # Get all PNGs with valid format
            png_files = [f for f in os.listdir(edf_path) if f.endswith('.png') and f.startswith('channel_')]
            png_dict = {
                re.match(r'channel_(.*?)\.png', f).group(1): f
                for f in png_files if re.match(r'channel_(.*?)\.png', f)
            }

            # Only use channels that match the standard list
            if all(ch in png_dict for ch in STANDARD_CHANNELS):
                row = {
                    'patient_id': patient_id,
                    'edf_file': edf_basename,
                    'label': label
                }

                for ch in STANDARD_CHANNELS:
                    row[f'channel_{ch}'] = os.path.join(edf_path, png_dict[ch])

                rows.append(row)
            else:
                missing = [ch for ch in STANDARD_CHANNELS if ch not in png_dict]
                print(f"⚠️ Skipping {edf_path}: missing channels {missing}")

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"\n✅ Final dataset saved to {output_csv} with {len(df)} valid samples.")

# Example usage
prepare_dataset(
    root_dir="C:/Users/ranji/OneDrive/Desktop/Train first",
    summary_dir="C:/Users/ranji/OneDrive/Desktop/summary_text_seizure",
    output_csv="C:/Users/ranji/OneDrive/Desktop/preprocessed_dataset.csv"
)
