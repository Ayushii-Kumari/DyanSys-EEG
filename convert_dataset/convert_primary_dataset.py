#!/usr/bin/env python3
"""
Convert ds004504 BIDS .set files → .npy files for DynaSys-EEG pipeline.

Dataset: https://openneuro.org/datasets/ds004504
- 88 subjects: A=AD(36), F=FTD(23), C=HC(29)
- 19 channels, 500 Hz, resting-state eyes-closed EEG
- Uses preprocessed derivatives (bandpass + ICA cleaned)

Output structure:
    data/primary/
        AD/sub-001.npy   (shape: 19 × n_times)
        FTD/sub-037.npy
        HC/sub-060.npy
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

RAW_DIR   = Path("data/primary_raw")
OUT_DIR   = Path("data/primary")
PARTICIPANTS = RAW_DIR / "participants.tsv"

GROUP_MAP = {"A": "AD", "F": "FTD", "C": "HC"}

def convert():
    # Read participants file
    df = pd.read_csv(PARTICIPANTS, sep="\t")
    # Strip CR from values (Windows line endings in the file)
    df = df.apply(lambda col: col.str.strip() if col.dtype == "object" else col)

    print(f"Found {len(df)} subjects in participants.tsv")
    print(df["Group"].value_counts().to_dict())

    # Create output directories
    for label in ["AD", "FTD", "HC"]:
        (OUT_DIR / label).mkdir(parents=True, exist_ok=True)

    converted = 0
    failed = 0

    for _, row in df.iterrows():
        sub_id = row["participant_id"].strip()
        group_code = row["Group"].strip()
        label = GROUP_MAP.get(group_code)
        if label is None:
            print(f"  Unknown group '{group_code}' for {sub_id}, skipping")
            continue

        # Prefer preprocessed derivatives (ICA cleaned)
        set_path = RAW_DIR / "derivatives" / sub_id / "eeg" / f"{sub_id}_task-eyesclosed_eeg.set"
        if not set_path.exists():
            # Fall back to raw
            set_path = RAW_DIR / sub_id / "eeg" / f"{sub_id}_task-eyesclosed_eeg.set"

        if not set_path.exists():
            print(f"  ⚠ File not found: {set_path}")
            failed += 1
            continue

        out_path = OUT_DIR / label / f"{sub_id}.npy"
        if out_path.exists():
            print(f"  ↷ Already exists: {out_path.name}")
            converted += 1
            continue

        try:
            import mne
            mne.set_log_level("WARNING")
            raw = mne.io.read_raw_eeglab(str(set_path), preload=True, verbose=False)
            data = raw.get_data()  # (n_channels, n_times) in Volts
            data = (data * 1e6).astype(np.float32)  # Convert V → µV
            np.save(str(out_path), data)
            print(f"  ✓ {sub_id} ({label}): shape={data.shape}, sfreq={raw.info['sfreq']} Hz")
            converted += 1
        except Exception as e:
            print(f"  ✗ {sub_id}: {e}")
            failed += 1

    print(f"\nDone! Converted: {converted}, Failed: {failed}")
    print(f"Output: {OUT_DIR.resolve()}/")

    # Show class counts
    for label in ["AD", "FTD", "HC"]:
        n = len(list((OUT_DIR / label).glob("*.npy")))
        print(f"  {label}: {n} subjects")


if __name__ == "__main__":
    convert()
