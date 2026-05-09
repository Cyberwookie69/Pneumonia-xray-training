"""Download (and extract) the Kaggle chest X-ray pneumonia dataset.

Looks for kaggle.json next to this file first (where the original Windows
setup put it), then falls back to ~/.kaggle/kaggle.json (the Linux/Colab
default). Either works.
"""
import os
import sys
import zipfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DATASET = "paultimothymooney/chest-xray-pneumonia"
DOWNLOAD_DIR = SCRIPT_DIR / "data"
ZIP_PATH = DOWNLOAD_DIR / "chest-xray-pneumonia.zip"
EXTRACT_MARKER = DOWNLOAD_DIR / "chest_xray" / "train" / "NORMAL"

# Resolve credentials. SCRIPT_DIR first (Windows habit), then ~/.kaggle/ (Colab).
if (SCRIPT_DIR / "kaggle.json").exists():
    os.environ["KAGGLE_CONFIG_DIR"] = str(SCRIPT_DIR)
elif (Path.home() / ".kaggle" / "kaggle.json").exists():
    os.environ["KAGGLE_CONFIG_DIR"] = str(Path.home() / ".kaggle")
else:
    sys.exit(
        "kaggle.json not found in "
        f"{SCRIPT_DIR} or {Path.home() / '.kaggle'} — go grab one from "
        "kaggle.com/settings → Create New API Token."
    )

# Kaggle insists on 0600 perms on Linux; on Windows this is theatre but harmless.
try:
    os.chmod(Path(os.environ["KAGGLE_CONFIG_DIR"]) / "kaggle.json", 0o600)
except OSError:
    pass

from kaggle.api.kaggle_api_extended import KaggleApi  # noqa: E402

api = KaggleApi()
api.authenticate()

DOWNLOAD_DIR.mkdir(exist_ok=True)

# Step 1: download zip if missing.
if ZIP_PATH.exists():
    print(f"Zip already here: {ZIP_PATH} — skipping download.")
else:
    print(f"Pulling {DATASET} into {DOWNLOAD_DIR} ...")
    api.dataset_download_files(DATASET, path=str(DOWNLOAD_DIR), unzip=False, quiet=False)

# Step 2: extract if not already done. Idempotent — skips when the marker dir exists.
if EXTRACT_MARKER.exists() and any(EXTRACT_MARKER.glob("*.jpeg")):
    print(f"Dataset already extracted at {EXTRACT_MARKER.parent.parent}.")
else:
    print(f"Extracting {ZIP_PATH.name} ...")
    with zipfile.ZipFile(ZIP_PATH) as z:
        z.extractall(DOWNLOAD_DIR)
    # Some Kaggle dumps nest the dataset (chest_xray/chest_xray/train/...). Flatten.
    nested = DOWNLOAD_DIR / "chest_xray" / "chest_xray"
    if nested.is_dir() and (nested / "train").exists():
        for child in nested.iterdir():
            child.rename(DOWNLOAD_DIR / "chest_xray" / child.name)
        nested.rmdir()
    print(f"Extracted to {DOWNLOAD_DIR / 'chest_xray'}.")

print("Done.")
