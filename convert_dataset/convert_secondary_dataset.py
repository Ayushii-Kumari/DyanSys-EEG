#!/usr/bin/env python3
"""
Convert secondary olfactory EEG dataset (.mat) → .npy files
for DynaSys-EEG pipeline.

Dataset:
- AD patients
- aMCI patients
- Healthy controls
- 4 EEG channels
- 200 Hz

Output structure:
    data/secondary/
        AD/
        aMCI/
        HC/
"""

#!/usr/bin/env python3

#!/usr/bin/env python3

import numpy as np
import scipy.io as sio
from pathlib import Path

RAW_DIR = Path("data/secondary_raw")
OUT_DIR = Path("data/secondary")

FILES = {
    "AD.mat": "AD",
    "MCI.mat": "MCI",
    "normal.mat": "HC"
}

# create folders
for label in ["AD", "MCI", "HC"]:
    (OUT_DIR / label).mkdir(parents=True, exist_ok=True)

converted = 0

for filename, label in FILES.items():

    file_path = RAW_DIR / filename

    print(f"\nLoading {filename}...")

    mat = sio.loadmat(file_path)

    # get variable name
    variable_name = [k for k in mat.keys() if not k.startswith("__")][0]

    data = mat[variable_name]

    print(f"Variable: {variable_name}")
    print(f"Subjects: {data.shape[1]}")

    subjects = data[0]

    for idx, subject in enumerate(subjects):

        try:
            # extract EEG epochs
            eeg = subject["epoch"]

            # convert object array → float32
            eeg = np.array(eeg, dtype=np.float32)

            # reshape:
            # current = (4, 600, trials)
            # convert to = (channels, time)
            eeg = eeg.reshape(4, -1)

            out_path = OUT_DIR / label / f"{label}_{idx+1:03d}.npy"

            np.save(out_path, eeg)

            print(f"  Saved {out_path.name} shape={eeg.shape}")

            converted += 1

        except Exception as e:
            print(f"  Failed subject {idx+1}: {e}")

print(f"\nDone! Converted {converted} subjects.")
print(f"Saved to: {OUT_DIR.resolve()}")