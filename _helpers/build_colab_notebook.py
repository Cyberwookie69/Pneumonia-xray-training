"""Rebuild pneumonia_colab.ipynb from scratch.

The notebook is structured as an increasing-complexity ladder:
  Step 1 — ResNet50 from scratch
  Step 2 — + ImageNet pretraining
  Step 3 — + ConvNeXt-Tiny (architectural diversity)
  Step 4 — + SNR-AdamW (theory-driven optimizer)
  Eval (per rung) → 15-model multi-arch ensemble headline.

Run:
    python _helpers/build_colab_notebook.py
"""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "pneumonia_colab.ipynb"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text,
    }


CELLS = [
    md(
        "# Pneumonia detection — Colab pipeline\n"
        "\n"
        "A single-pass walkthrough of the project, framed as an "
        "**increasing-complexity ladder**.\n"
        "Each rung adds one ingredient on top of the previous one and "
        "reports the held-out test accuracy.\n"
        "\n"
        "| Rung | Setup | Adds vs. previous |\n"
        "|------|-------|-------------------|\n"
        "| 1 | ResNet50 from scratch (5-fold) | baseline — no transfer learning |\n"
        "| 2 | ResNet50 + ImageNet pretrained (5-fold) | + transfer learning |\n"
        "| 3 | + ConvNeXt-Tiny (10 models) | + architectural diversity |\n"
        "| 4 | + SNR-AdamW ResNet50 (15 models) | + theory-driven optimizer (Litman & Guo 2026) |\n"
        "\n"
        "The headline number is rung 4: the 15-model multi-architecture ensemble.\n"
        "\n"
        "## Hardware\n"
        "*Runtime → Change runtime type → T4 GPU* (or better).\n"
        "\n"
        "## Time estimate (free T4)\n"
        "- Setup + dataset download: ~3 min\n"
        "- Step 1 — 5-fold ResNet50 from scratch: ~25 min\n"
        "- Step 2 — 5-fold ResNet50 pretrained: ~25 min\n"
        "- Step 3 — 5-fold ConvNeXt-Tiny: ~25 min\n"
        "- Step 4 — 5-fold SNR-AdamW ResNet50: ~25 min\n"
        "- Eval + plots: ~5 min\n"
        "- **Total: ~110 min on a free T4** (well inside the ~12 h Colab session limit)\n"
        "\n"
        "(The same pipeline takes ~30+ hours on AMD Vega 64 + DirectML on Windows.)\n"
        "\n"
        "## Setup flow\n"
        "Run sections 1-2, then **either** section 3 (recommended — uses Drive "
        "for credentials and persistent runs) **or** section 4 (manual upload). "
        "After that, run sections 5-12 in order."
    ),
    md("## 1. Clone repo + verify GPU"),
    code(
        "!nvidia-smi --query-gpu=name,memory.total --format=csv,noheader\n"
        "%cd /content\n"
        "!git clone https://github.com/Cyberwookie69/Pneumonia-xray-training.git\n"
        "%cd /content/Pneumonia-xray-training"
    ),
    md(
        "## 2. Install dependencies\n"
        "\n"
        "Colab already has PyTorch with CUDA. We just need the project's extras."
    ),
    code("!pip install -q timm grad-cam opencv-python-headless kaggle"),
    md(
        "## 3. Mount Drive + load Kaggle credentials (recommended)\n"
        "\n"
        "Mounts Google Drive, then:\n"
        "- Copies `kaggle.json` from `My Drive/kaggle.json` to `~/.kaggle/` "
        "if you've put it there (one-time setup — skip the upload widget every run).\n"
        "- Sets `PNEUMONIA_RUNS` to a Drive folder so checkpoints survive "
        "the ~12 h Colab session timeout.\n"
        "\n"
        "If `kaggle.json` is **not** on Drive, this cell still works (sets up "
        "persistence) and you fall through to section 4 to upload it manually."
    ),
    code(
        "from google.colab import drive\n"
        "import os, shutil\n"
        "\n"
        "drive.mount('/content/drive')\n"
        "\n"
        "# Pull kaggle.json from Drive if you've placed it there.\n"
        "src = '/content/drive/MyDrive/kaggle.json'\n"
        "kaggle_dir = os.path.expanduser('~/.kaggle')\n"
        "if os.path.exists(src):\n"
        "    os.makedirs(kaggle_dir, exist_ok=True)\n"
        "    shutil.copy(src, os.path.join(kaggle_dir, 'kaggle.json'))\n"
        "    os.chmod(os.path.join(kaggle_dir, 'kaggle.json'), 0o600)\n"
        "    print('✓ kaggle.json copied from Drive — section 4 can be skipped')\n"
        "else:\n"
        "    print(f'⚠ {src} not found — run section 4 to upload it manually,\\n'\n"
        "          f'  or place kaggle.json at that Drive path for next time.')\n"
        "\n"
        "# Persist runs across Colab sessions.\n"
        "os.makedirs('/content/drive/MyDrive/pneumonia_runs', exist_ok=True)\n"
        "os.environ['PNEUMONIA_RUNS'] = '/content/drive/MyDrive/pneumonia_runs'\n"
        "print(f\"Runs will be saved to: {os.environ['PNEUMONIA_RUNS']}\")"
    ),
    md(
        "## 4. Kaggle authentication via upload widget (fallback)\n"
        "\n"
        "Only needed if section 3 didn't find `kaggle.json` on your Drive. "
        "Click *Choose Files* and pick the `kaggle.json` you downloaded from "
        "https://www.kaggle.com/settings → \"Create New API Token\"."
    ),
    code(
        "import os\n"
        "if os.path.exists(os.path.expanduser('~/.kaggle/kaggle.json')):\n"
        "    print('✓ kaggle.json already in place (loaded by section 3) — '\n"
        "          'no need to upload')\n"
        "else:\n"
        "    from google.colab import files\n"
        "    uploaded = files.upload()  # select kaggle.json\n"
        "    !mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json"
    ),
    md("## 5. Download dataset (~2.3 GB, ~1 min)"),
    code("!python pneumonia.py"),
    md(
        "## 6. Step 1 — ResNet50 from scratch (baseline)\n"
        "\n"
        "5-fold ResNet50 trained from random init (no ImageNet weights). "
        "This is the weakest rung — it isolates everything that pretraining "
        "gives us at Step 2. Run names: `scratch_f0..f4`."
    ),
    code(
        "# No --pretrained flag → tag defaults to \"scratch\", produces scratch_f0..f4\n"
        "!python pneumonia_run_folds.py --extra=\"--img_size 288 --num_workers 4\""
    ),
    md(
        "## 7. Step 2 — + ImageNet pretraining (transfer learning)\n"
        "\n"
        "Same architecture (ResNet50 @ 288), now initialised from ImageNet "
        "weights and fine-tuned with focal loss. The Step 1 → Step 2 delta "
        "is the contribution of transfer learning. Run names: `ens_f0..f4`."
    ),
    code(
        "# --pretrained → tag defaults to \"ens\", produces ens_f0..f4\n"
        "!python pneumonia_run_folds.py --pretrained --extra=\"--img_size 288 --num_workers 4\""
    ),
    md(
        "## 8. Step 3 — + ConvNeXt-Tiny (architectural diversity)\n"
        "\n"
        "A second pretrained backbone, 5-fold @ 224. Adding it to the "
        "ResNet50 ensemble brings the count to 10 models and tests whether "
        "*architectural* diversity helps on top of *fold* diversity. Run "
        "names: `cnx224_f0..f4`."
    ),
    code(
        "# ConvNeXt-Tiny @ 224 — ~25 min on T4\n"
        "# Paper: https://arxiv.org/abs/2201.03545\n"
        "# A 2022 pure-CNN that matches Vision Transformers on ImageNet by borrowing\n"
        "# their design choices (large kernels, LayerNorm, GELU, inverted bottleneck).\n"
        "# Chosen here for architectural diversity vs. ResNet50 in the multi-arch ensemble.\n"
        "!python pneumonia_run_folds.py --pretrained "
        "--extra=\"--model convnext_tiny.fb_in22k_ft_in1k\" --tag cnx224"
    ),
    md(
        "## 9. Step 4 — + SNR-AdamW ResNet50 (theory-driven optimizer)\n"
        "\n"
        "A third 5-fold run, same backbone as Step 2 but trained with the "
        "SNR-gated AdamW optimizer (Litman & Guo 2026). Adding it brings "
        "the multi-arch ensemble to 15 models. Run names: `snr_r50_f0..f4`."
    ),
    code(
        "# SNR-AdamW ResNet50 — ~25 min on T4\n"
        "# Paper: https://arxiv.org/abs/2605.01172\n"
        "# Adds a per-parameter signal-to-noise gate to AdamW: each step, updates are\n"
        "# scaled by max(0, (μ²−σ²/(b−1))/(σ²+ε)) using EMA gradient mean μ and variance σ².\n"
        "# Suppresses parameters whose minibatch SNR is below threshold; same wall-clock cost.\n"
        "!python pneumonia_run_folds.py --pretrained --extra=\"--snr_optimizer\" --tag snr_r50"
    ),
    md(
        "## 10. Eval — the complexity ladder\n"
        "\n"
        "Each rung is one row in the eventual report table. The 15-model "
        "multi-arch ensemble at the bottom is the headline number. "
        "Per-architecture references (ConvNeXt alone, SNR alone) are also "
        "printed for completeness."
    ),
    code(
        "# === The complexity ladder (4 rungs) ===\n"
        "print(\"\\n=== Step 1 — ResNet50 from scratch (5-fold) ===\")\n"
        "!python pneumonia_eval.py --ensemble scratch_f0,scratch_f1,scratch_f2,scratch_f3,scratch_f4 --use_best --num_workers 0\n"
        "\n"
        "print(\"\\n=== Step 2 — ResNet50 pretrained (5-fold) ===\")\n"
        "!python pneumonia_eval.py --ensemble ens_f0,ens_f1,ens_f2,ens_f3,ens_f4 --use_best --num_workers 0\n"
        "\n"
        "print(\"\\n=== Step 3 — + ConvNeXt-Tiny (10 models) ===\")\n"
        "!python pneumonia_eval.py --ensemble ens_f0,ens_f1,ens_f2,ens_f3,ens_f4,cnx224_f0,cnx224_f1,cnx224_f2,cnx224_f3,cnx224_f4 --use_best --num_workers 0\n"
        "\n"
        "print(\"\\n=== Step 4 — + SNR-AdamW (15 models, multi-arch) — HEADLINE ===\")\n"
        "!python pneumonia_eval.py --ensemble ens_f0,ens_f1,ens_f2,ens_f3,ens_f4,cnx224_f0,cnx224_f1,cnx224_f2,cnx224_f3,cnx224_f4,snr_r50_f0,snr_r50_f1,snr_r50_f2,snr_r50_f3,snr_r50_f4 --use_best --num_workers 0\n"
        "\n"
        "# === Per-architecture (reference, not on the ladder) ===\n"
        "print(\"\\n=== Reference: ConvNeXt-Tiny alone (5-fold) ===\")\n"
        "!python pneumonia_eval.py --ensemble cnx224_f0,cnx224_f1,cnx224_f2,cnx224_f3,cnx224_f4 --use_best --num_workers 0\n"
        "\n"
        "print(\"\\n=== Reference: SNR-AdamW ResNet50 alone (5-fold) ===\")\n"
        "!python pneumonia_eval.py --ensemble snr_r50_f0,snr_r50_f1,snr_r50_f2,snr_r50_f3,snr_r50_f4 --use_best --num_workers 0"
    ),
    md("## 11. Plots"),
    code(
        "# Learning curves — one plot per rung\n"
        "!python pneumonia_plots.py curves --runs scratch_f0,scratch_f1,scratch_f2,scratch_f3,scratch_f4\n"
        "!python pneumonia_plots.py curves --runs ens_f0,ens_f1,ens_f2,ens_f3,ens_f4\n"
        "!python pneumonia_plots.py curves --runs cnx224_f0,cnx224_f1,cnx224_f2,cnx224_f3,cnx224_f4\n"
        "!python pneumonia_plots.py curves --runs snr_r50_f0,snr_r50_f1,snr_r50_f2,snr_r50_f3,snr_r50_f4\n"
        "\n"
        "# t-SNE feature embedding + Grad-CAM (use the strongest single fold = ens_f0)\n"
        "!python pneumonia_plots.py features --run ens_f0 --use_best\n"
        "!python pneumonia_gradcam.py --run_name ens_f0 --use_best --n_samples 8"
    ),
    md("## 12. Display the figures inline"),
    code(
        "from IPython.display import Image, display\n"
        "import os\n"
        "\n"
        "runs_root = os.environ.get('PNEUMONIA_RUNS', 'runs')\n"
        "for relpath in [\n"
        "    f'{runs_root}/plots/learning_curves_scratch.png',\n"
        "    f'{runs_root}/plots/learning_curves_ens.png',\n"
        "    f'{runs_root}/plots/learning_curves_cnx224.png',\n"
        "    f'{runs_root}/plots/learning_curves_snr_r50.png',\n"
        "    f'{runs_root}/plots/fold_best_val_ens.png',\n"
        "    f'{runs_root}/ens_f0/plots/tsne_best.png',\n"
        "]:\n"
        "    if os.path.exists(relpath):\n"
        "        print(relpath)\n"
        "        display(Image(relpath))"
    ),
]


NOTEBOOK = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.10"},
        "colab": {"provenance": []},
        "accelerator": "GPU",
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}


if __name__ == "__main__":
    OUT.write_text(json.dumps(NOTEBOOK, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {OUT} ({len(CELLS)} cells)")
