"""Download the Kaggle chest X-ray pneumonia dataset.

Reads the kaggle.json sitting next to this file (because keeping API tokens
in your home dir is for people who like organization).
"""
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
KAGGLE_JSON = SCRIPT_DIR / "kaggle.json"
DATASET = "paultimothymooney/chest-xray-pneumonia"
DOWNLOAD_DIR = SCRIPT_DIR / "data"
ZIP_PATH = DOWNLOAD_DIR / "chest-xray-pneumonia.zip"

# Tell the kaggle SDK where to look. Otherwise it will hunt for ~/.kaggle/kaggle.json
# and complain loudly when it doesn't find it.
os.environ["KAGGLE_CONFIG_DIR"] = str(SCRIPT_DIR)

if not KAGGLE_JSON.exists():
    sys.exit(f"kaggle.json not found in {SCRIPT_DIR} — go get one from kaggle.com/account.")

# Kaggle insists on 0600 perms on Linux; on Windows this is theatre but harmless.
try:
    os.chmod(KAGGLE_JSON, 0o600)
except OSError:
    pass

from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

DOWNLOAD_DIR.mkdir(exist_ok=True)
if ZIP_PATH.exists():
    print(f"Zip already here: {ZIP_PATH} — skipping download.")
else:
    print(f"Pulling {DATASET} into {DOWNLOAD_DIR} ...")
    api.dataset_download_files(DATASET, path=str(DOWNLOAD_DIR), unzip=False, quiet=False)
    print("Done. Now go forget about this for three weeks.")
