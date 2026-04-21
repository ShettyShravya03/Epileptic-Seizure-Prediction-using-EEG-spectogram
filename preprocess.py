"""
preprocess.py
-------------
Parses CHB-MIT seizure summary text files and builds a structured CSV dataset
where each row represents one EDF segment with 22 channel spectrogram paths
and a binary label (0 = interictal, 1 = preictal).

Usage:
    python preprocess.py \
        --root_dir   /path/to/spectrogram/root \
        --summary_dir /path/to/summary_text_files \
        --output_csv  preprocessed_dataset.csv
"""

import os
import re
import argparse
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


def parse_seizure_summaries(summary_dir):
    """
    Parse all *-summary.txt files in summary_dir.

    Returns:
        dict: { patient_id: { edf_filename: [(start_sec, end_sec), ...] } }
    """
    seizure_times = defaultdict(dict)

    for filename in os.listdir(summary_dir):
        if not filename.endswith('-summary.txt'):
            continue

        patient_id = filename.split('-')[0]
        with open(os.path.join(summary_dir, filename), 'r') as f:
            lines = f.readlines()

        current_file = None
        seizures = []
        start_time = None

        for line in lines:
            file_match  = re.match(r'File Name:\s*(\S+\.edf)', line)
            start_match = re.match(r'Seizure(?: \d+)? Start Time:\s*(\d+)\s*seconds', line)
            end_match   = re.match(r'Seizure(?: \d+)? End Time:\s*(\d+)\s*seconds', line)

            if file_match:
                if current_file and seizures:
                    seizure_times[patient_id][current_file] = seizures
                current_file = file_match.group(1)
                seizures = []

            if start_match:
                start_time = int(start_match.group(1))
            if end_match and start_time is not None:
                seizures.append((start_time, int(end_match.group(1))))
                start_time = None

        if current_file and seizures:
            seizure_times[patient_id][current_file] = seizures

    return dict(seizure_times)


def prepare_dataset(root_dir, summary_dir, output_csv):
    """
    Walk root_dir, match each EDF folder to seizure labels, and build a CSV
    with one row per valid segment (all 22 standard channels present).
    """
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

            edf_basename_ext = edf_folder + '.edf'
            seizure_windows = seizure_times.get(patient_id, {}).get(edf_basename_ext, [])
            label = 1 if seizure_windows else 0  # binary: preictal vs interictal

            png_files = [
                f for f in os.listdir(edf_path)
                if f.endswith('.png') and f.startswith('channel_')
            ]
            if len(png_files) < NUM_CHANNELS:
                print(f"⚠️  Skipping {edf_path}: only {len(png_files)} channels found.")
                continue

            png_dict = {
                re.match(r'channel_(.*?)\.png', f).group(1): f
                for f in png_files if re.match(r'channel_(.*?)\.png', f)
            }

            missing = [ch for ch in STANDARD_CHANNELS if ch not in png_dict]
            if missing:
                print(f"⚠️  Skipping {edf_path}: missing channels {missing}")
                continue

            row = {
                'patient_id': patient_id,
                'edf_file':   edf_folder,
                'label':      label,
            }
            for ch in STANDARD_CHANNELS:
                row[f'channel_{ch}'] = os.path.join(edf_path, png_dict[ch])

            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"\n✅ Dataset saved → {output_csv}  ({len(df)} valid segments)")
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build dataset CSV from CHB-MIT spectrograms')
    parser.add_argument('--root_dir',    required=True, help='Root folder of spectrogram images')
    parser.add_argument('--summary_dir', required=True, help='Folder containing *-summary.txt files')
    parser.add_argument('--output_csv',  default='preprocessed_dataset.csv')
    args = parser.parse_args()

    prepare_dataset(args.root_dir, args.summary_dir, args.output_csv)