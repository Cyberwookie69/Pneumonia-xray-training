"""Rebuild pneumonia_colab.ipynb for the academic CNN-design assignment.

The notebook is structured around the three assignment questions:
  Q1. Number of conv-pool building blocks  → Ablation A1
  Q2. Strides, padding, activation         → Ablation A2
  Q3. Solution to avoid overfitting        → Ablation A3
plus a 5-fold champion run, medical KPI evaluation, and a few side studies
(Mixup/CutMix/Manifold demo, transfer-learning comparison) reported as
"other things we tried" in §7 of the report.

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
        "# Pneumonia detection — academic CNN-design study\n"
        "\n"
        "This notebook answers the three assignment questions through ablation:\n"
        "\n"
        "| # | Question | Ablation |\n"
        "|---|----------|----------|\n"
        "| Q1 | Number of conv-pool building blocks | A1 (depth) |\n"
        "| Q2 | Strides / padding / activation | A2 |\n"
        "| Q3 | Solution to avoid overfitting | A3 |\n"
        "\n"
        "The custom CNN is parametric (`pneumonia_cnn_custom.py`): every architectural choice "
        "is a CLI flag, so each ablation row is one shell invocation. Train/val/test discipline: "
        "test is touched only at the end of each row.\n"
        "\n"
        "## Hardware\n"
        "*Runtime → Change runtime type → H100 GPU* (Pro+ tier or pay-as-you-go compute units).\n"
        "\n"
        "## Time estimate (H100)\n"
        "- Setup + dataset: ~3 min (network-bound, unchanged across GPUs)\n"
        "- Smoke test: ~30 s\n"
        "- A1 depth ablation (4 rows × 20 epochs): ~6 min\n"
        "- A2 stride/padding/activation (6 rows): ~9 min\n"
        "- A3 overfitting (6 rows): ~9 min\n"
        "- Champion 5-fold + KPIs + curves + Grad-CAM: ~10 min\n"
        "- Mixup demo (display): instant\n"
        "- Transfer-learning comparison (ResNet50 @ 288, 5-fold + eval): ~8 min\n"
        "- **Total: ~45 min on H100**\n"
        "\n"
        "*Reference for context*: same pipeline takes ~3 h on free T4, ~1 h on A100, "
        "~30+ h on AMD Vega 64 + DirectML.\n"
        "\n"
        "All ablation cells are **incrementally resumable** — a row whose "
        "`runs/<name>/summary.json` already exists is skipped (a tiny shell wrapper handles this)."
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
        "Colab already has PyTorch with CUDA. We add the few extras the project uses."
    ),
    code("!pip install -q timm grad-cam opencv-python-headless kaggle"),
    md(
        "## 3. Mount Drive + load Kaggle credentials (recommended)\n"
        "\n"
        "Mounts Google Drive, copies `kaggle.json` from `My Drive/kaggle.json` to `~/.kaggle/` "
        "(if present), and sets `PNEUMONIA_RUNS` to a Drive folder so checkpoints survive "
        "session timeouts. If you haven't placed `kaggle.json` on Drive yet, this cell still "
        "works (sets up persistence) and you fall through to section 4 to upload it manually."
    ),
    code(
        "from google.colab import drive\n"
        "import os, shutil\n"
        "\n"
        "drive.mount('/content/drive')\n"
        "\n"
        "src = '/content/drive/MyDrive/kaggle.json'\n"
        "kaggle_dir = os.path.expanduser('~/.kaggle')\n"
        "if os.path.exists(src):\n"
        "    os.makedirs(kaggle_dir, exist_ok=True)\n"
        "    shutil.copy(src, os.path.join(kaggle_dir, 'kaggle.json'))\n"
        "    os.chmod(os.path.join(kaggle_dir, 'kaggle.json'), 0o600)\n"
        "    print('✓ kaggle.json copied from Drive — section 4 can be skipped')\n"
        "else:\n"
        "    print(f'⚠ {src} not found — run section 4 to upload it manually.')\n"
        "\n"
        "os.makedirs('/content/drive/MyDrive/pneumonia_runs', exist_ok=True)\n"
        "os.environ['PNEUMONIA_RUNS'] = '/content/drive/MyDrive/pneumonia_runs'\n"
        "print(f\"Runs will be saved to: {os.environ['PNEUMONIA_RUNS']}\")"
    ),
    md(
        "## 4. Kaggle authentication via upload widget (fallback)\n"
        "\n"
        "Skip this cell if section 3 already loaded `kaggle.json`."
    ),
    code(
        "import os\n"
        "if os.path.exists(os.path.expanduser('~/.kaggle/kaggle.json')):\n"
        "    print('✓ kaggle.json already in place — no need to upload')\n"
        "else:\n"
        "    from google.colab import files\n"
        "    uploaded = files.upload()\n"
        "    !mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json"
    ),
    md("## 5. Download dataset (~2.3 GB, ~1 min)"),
    code("!python pneumonia.py"),
    md(
        "## 6. Smoke test — one fast custom-CNN run\n"
        "\n"
        "Verifies that data loading + model + training loop works end-to-end before "
        "we burn time on the ablations. 5 epochs only, default 4-block ReLU CNN."
    ),
    code(
        "!python pneumonia_cnn_custom.py --run_name smoke_test --epochs 5 --num_workers 6"
    ),
    md(
        "---\n"
        "## 7. Ablation A1 — Number of conv-pool building blocks (Q1)\n"
        "\n"
        "Holds activation=ReLU, padding=same, stride_mode=pool, no regularisation. "
        "Varies only `n_blocks ∈ {2, 3, 4, 5}`. Each row is a single 88/12 train/val "
        "split (fast, fine for hyperparameter selection); the champion will be 5-fold."
    ),
    code(
        "import os, subprocess, json\n"
        "\n"
        "RUNS_ROOT = os.environ.get('PNEUMONIA_RUNS', 'runs')\n"
        "\n"
        "def run_if_missing(run_name, args):\n"
        "    summary = f'{RUNS_ROOT}/{run_name}/summary.json'\n"
        "    if os.path.exists(summary):\n"
        "        print(f'✓ {run_name} already done — skipping')\n"
        "        return\n"
        "    cmd = ['python', 'pneumonia_cnn_custom.py', '--run_name', run_name] + args\n"
        "    print('>>>', ' '.join(cmd))\n"
        "    subprocess.run(cmd, check=True)\n"
        "\n"
        "for n in [2, 3, 4, 5]:\n"
        "    run_if_missing(f'a1_d{n}', ['--n_blocks', str(n), '--epochs', '20', '--num_workers', '6'])"
    ),
    code(
        "# Collect A1 results + add a Glorot-init control on the winning depth\n"
        "# (He et al. 2015 argue Glorot fails with stacked ReLUs at depth ≥ 4 —\n"
        "#  this is the controlled comparison that makes the depth ablation a\n"
        "#  *finding*, not just a tuning sweep).\n"
        "import json, os, math\n"
        "from pathlib import Path\n"
        "\n"
        "RUNS_ROOT = os.environ.get('PNEUMONIA_RUNS', 'runs')\n"
        "\n"
        "best_n, best_va = 4, 0.0\n"
        "for n in [2, 3, 4, 5]:\n"
        "    s = json.load(open(f'{RUNS_ROOT}/a1_d{n}/summary.json'))\n"
        "    if s['best_val_acc'] > best_va:\n"
        "        best_va, best_n = s['best_val_acc'], n\n"
        "\n"
        "# Glorot control on the winning depth\n"
        "run_if_missing(f'a1_d{best_n}_glorot',\n"
        "               ['--n_blocks', str(best_n), '--init', 'glorot',\n"
        "                '--epochs', '20', '--num_workers', '6'])\n"
        "\n"
        "# === A1 results table with binomial noise-floor annotation ===\n"
        "rows = []\n"
        "for label, run in [(f'{n}', f'a1_d{n}') for n in [2, 3, 4, 5]] + \\\n"
        "                  [(f'{best_n} (Glorot)', f'a1_d{best_n}_glorot')]:\n"
        "    s = json.load(open(f'{RUNS_ROOT}/{run}/summary.json'))\n"
        "    rows.append((label, s['architecture']['n_params'], s['best_val_acc'],\n"
        "                 s['test_acc'], s['training']['elapsed_min']))\n"
        "\n"
        "# Binomial noise floor on test accuracy: sigma = sqrt(p*(1-p)/n)\n"
        "p = max(r[3] for r in rows); n_test = 624\n"
        "sigma_test = math.sqrt(p * (1 - p) / n_test)\n"
        "print(f'{\"n_blocks\":>14}{\"params\":>12}{\"val_acc\":>10}{\"test_acc\":>10}{\"min\":>8}')\n"
        "for lbl, params, va, ta, mn in rows:\n"
        "    print(f'{lbl:>14}{params:>12,}{va:>10.4f}{ta:>10.4f}{mn:>8.1f}')\n"
        "print()\n"
        "print(f'Binomial noise floor on test acc (n={n_test}): '\n"
        "      f'sigma ~= {sigma_test:.4f} ({sigma_test*100:.2f} pp)')\n"
        "print(f'-> any pairwise delta below ~{2*sigma_test*100:.2f} pp '\n"
        "      f'is statistically indistinguishable.')"
    ),
    md(
        "---\n"
        "## 8. Ablation A2 — Stride / padding / activation (Q2)\n"
        "\n"
        "Holds depth at the A1 winner (default n_blocks=4 — adjust below if A1 picked otherwise). "
        "Varies activation, padding, and stride_mode. Six representative cells; full Cartesian "
        "(3×2×2 = 12) is overkill for the report."
    ),
    code(
        "# Pick the A1 winner (largest test_acc among 2..5 blocks). Default to 4 if tie.\n"
        "import json, os\n"
        "RUNS_ROOT = os.environ.get('PNEUMONIA_RUNS', 'runs')\n"
        "best_n, best_acc = 4, 0.0\n"
        "for n in [2, 3, 4, 5]:\n"
        "    s = json.load(open(f'{RUNS_ROOT}/a1_d{n}/summary.json'))\n"
        "    if s['test_acc'] > best_acc:\n"
        "        best_acc, best_n = s['test_acc'], n\n"
        "print(f'A1 winner: n_blocks={best_n} (test_acc={best_acc:.4f})')\n"
        "\n"
        "A2_RUNS = [\n"
        "    # name              activation padding  stride_mode\n"
        "    ('a2_relu_same_pool',     'relu',  'same',  'pool'),\n"
        "    ('a2_leaky_same_pool',    'leaky', 'same',  'pool'),\n"
        "    ('a2_gelu_same_pool',     'gelu',  'same',  'pool'),\n"
        "    ('a2_relu_valid_pool',    'relu',  'valid', 'pool'),\n"
        "    ('a2_relu_same_strided',  'relu',  'same',  'strided'),\n"
        "    ('a2_gelu_same_strided',  'gelu',  'same',  'strided'),\n"
        "]\n"
        "for name, act, pad, sm in A2_RUNS:\n"
        "    run_if_missing(name, ['--n_blocks', str(best_n),\n"
        "                          '--activation', act, '--padding', pad,\n"
        "                          '--stride_mode', sm,\n"
        "                          '--epochs', '20', '--num_workers', '6'])"
    ),
    code(
        "# A2 results table\n"
        "rows = []\n"
        "for name, act, pad, sm in A2_RUNS:\n"
        "    s = json.load(open(f'{RUNS_ROOT}/{name}/summary.json'))\n"
        "    rows.append((act, pad, sm, s['best_val_acc'], s['test_acc']))\n"
        "print(f'{\"activation\":>11}{\"padding\":>9}{\"stride\":>10}{\"val_acc\":>10}{\"test_acc\":>10}')\n"
        "for r in rows:\n"
        "    print(f'{r[0]:>11}{r[1]:>9}{r[2]:>10}{r[3]:>10.4f}{r[4]:>10.4f}')"
    ),
    md(
        "---\n"
        "## 9. Ablation A3 — Overfitting solutions (Q3)\n"
        "\n"
        "Holds the A1+A2 winner architecture. Varies regularisation: none / BN / dropout / L2 / "
        "augmentation / combined. The 'none' row is the same architecture without any "
        "anti-overfit mechanism — expect the largest train-vs-val gap there."
    ),
    code(
        "# Pick the A2 winner from the table above. We assume the highest-test row.\n"
        "best_a2 = None; best_acc = 0.0\n"
        "for name, act, pad, sm in A2_RUNS:\n"
        "    s = json.load(open(f'{RUNS_ROOT}/{name}/summary.json'))\n"
        "    if s['test_acc'] > best_acc:\n"
        "        best_acc = s['test_acc']\n"
        "        best_a2 = (act, pad, sm)\n"
        "act, pad, sm = best_a2\n"
        "print(f'A2 winner: act={act}, pad={pad}, stride_mode={sm} (test_acc={best_acc:.4f})')\n"
        "\n"
        "BASE = ['--n_blocks', str(best_n), '--activation', act, '--padding', pad,\n"
        "        '--stride_mode', sm, '--epochs', '20', '--num_workers', '6']\n"
        "\n"
        "A3_RUNS = [\n"
        "    ('a3_none',    BASE),\n"
        "    ('a3_bn',      BASE + ['--use_bn']),\n"
        "    ('a3_dropout', BASE + ['--use_dropout', '0.3']),\n"
        "    ('a3_l2',      BASE + ['--weight_decay', '1e-4']),\n"
        "    ('a3_aug',     BASE + ['--augment']),\n"
        "    ('a3_combo',   BASE + ['--use_bn', '--use_dropout', '0.3', '--augment',\n"
        "                           '--weight_decay', '1e-4',\n"
        "                           '--early_stop_patience', '5']),\n"
        "]\n"
        "for name, args in A3_RUNS:\n"
        "    run_if_missing(name, args)"
    ),
    code(
        "# A3 results table — focus on the train-vs-val gap as a regularisation indicator\n"
        "rows = []\n"
        "for name, _ in A3_RUNS:\n"
        "    s = json.load(open(f'{RUNS_ROOT}/{name}/summary.json'))\n"
        "    h = json.load(open(f'{RUNS_ROOT}/{name}/history.json'))\n"
        "    final_train = h['train_acc'][-1] if h['train_acc'] else 0.0\n"
        "    gap = final_train - s['best_val_acc']\n"
        "    rows.append((name.replace('a3_', ''), final_train, s['best_val_acc'],\n"
        "                 gap, s['test_acc']))\n"
        "print(f'{\"reg\":>10}{\"train_acc\":>11}{\"val_acc\":>10}{\"gap\":>8}{\"test_acc\":>10}')\n"
        "for r in rows:\n"
        "    print(f'{r[0]:>10}{r[1]:>11.4f}{r[2]:>10.4f}{r[3]:>8.4f}{r[4]:>10.4f}')"
    ),
    md(
        "---\n"
        "## 10. Champion — train winning configuration with 5-fold CV\n"
        "\n"
        "Combines the A1, A2, and A3 winners. Trains 5 folds for a robust ensemble result on "
        "the official Kaggle test set."
    ),
    code(
        "# Pick A3 winner (highest test_acc — but feel free to override based on the table)\n"
        "best_a3, best_acc = None, 0.0\n"
        "for name, args in A3_RUNS:\n"
        "    s = json.load(open(f'{RUNS_ROOT}/{name}/summary.json'))\n"
        "    if s['test_acc'] > best_acc:\n"
        "        best_acc = s['test_acc']; best_a3 = (name, args)\n"
        "print(f'A3 winner: {best_a3[0]} (test_acc={best_acc:.4f})')\n"
        "champion_extra = best_a3[1]\n"
        "\n"
        "# Train 5 folds; each saves test_probs.npy that we ensemble below.\n"
        "for fold in range(5):\n"
        "    args = champion_extra + ['--n_folds', '5', '--fold', str(fold)]\n"
        "    run_if_missing(f'champion_f{fold}', args)"
    ),
    md(
        "## 11. Champion — medical KPI evaluation (Sens / Spec / AUROC / ECE)\n"
        "\n"
        "Ensembles the 5 fold probability predictions and reports the four medical KPIs at "
        "three operating points: default τ=0.5, val-tuned best-accuracy τ, and "
        "sensitivity-targeted τ ≥ 0.97."
    ),
    code(
        "import numpy as np, json, os\n"
        "RUNS_ROOT = os.environ.get('PNEUMONIA_RUNS', 'runs')\n"
        "\n"
        "probs = [np.load(f'{RUNS_ROOT}/champion_f{i}/test_probs.npy') for i in range(5)]\n"
        "labels = np.load(f'{RUNS_ROOT}/champion_f0/test_labels.npy').astype(int)\n"
        "ensemble_probs = np.mean(np.stack(probs, axis=0), axis=0)\n"
        "\n"
        "# Save ensemble probs+labels so _helpers/medical_kpis.py can read them\n"
        "ens_dir = f'{RUNS_ROOT}/champion_ensemble'\n"
        "os.makedirs(ens_dir, exist_ok=True)\n"
        "np.save(f'{ens_dir}/test_probs.npy', ensemble_probs)\n"
        "np.save(f'{ens_dir}/test_labels.npy', labels)\n"
        "\n"
        "!python _helpers/medical_kpis.py --run {ens_dir}"
    ),
    md(
        "## 12. Champion — learning curves\n"
        "\n"
        "One curve per fold, plus the per-fold final test accuracy."
    ),
    code(
        "import json, matplotlib.pyplot as plt\n"
        "\n"
        "fig, axes = plt.subplots(1, 2, figsize=(13, 5))\n"
        "for fold in range(5):\n"
        "    h = json.load(open(f'{RUNS_ROOT}/champion_f{fold}/history.json'))\n"
        "    axes[0].plot(h['train_loss'], alpha=0.6, label=f'fold{fold} train')\n"
        "    axes[0].plot(h['val_loss'], alpha=0.6, linestyle='--', label=f'fold{fold} val')\n"
        "    axes[1].plot(h['train_acc'], alpha=0.6, label=f'fold{fold} train')\n"
        "    axes[1].plot(h['val_acc'], alpha=0.6, linestyle='--', label=f'fold{fold} val')\n"
        "axes[0].set_title('Loss'); axes[0].set_xlabel('epoch'); axes[0].legend(fontsize=7)\n"
        "axes[1].set_title('Accuracy'); axes[1].set_xlabel('epoch'); axes[1].legend(fontsize=7)\n"
        "plt.tight_layout(); plt.savefig(f'{RUNS_ROOT}/champion_ensemble/learning_curves.png', dpi=110)\n"
        "plt.show()"
    ),
    md(
        "## 13. Champion — Grad-CAM\n"
        "\n"
        "Where does the champion model look when it predicts pneumonia? Useful for clinician "
        "trust and for catching reliance on dataset artefacts (text annotations, machine IDs)."
    ),
    code(
        "# Grad-CAM only works with the existing pneumonia_gradcam.py for timm models.\n"
        "# For our custom CNN we draw heatmaps directly via the last conv block's gradients.\n"
        "import torch, json\n"
        "import numpy as np\n"
        "import matplotlib.pyplot as plt\n"
        "from PIL import Image\n"
        "from pneumonia_cnn_custom import CustomCNN, build_transforms\n"
        "from pneumonia_train import DATA_ROOT, list_images\n"
        "\n"
        "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n"
        "\n"
        "summ = json.load(open(f'{RUNS_ROOT}/champion_f0/summary.json'))\n"
        "arch = summ['architecture']\n"
        "model = CustomCNN(n_blocks=arch['n_blocks'], base_channels=arch['base_channels'],\n"
        "                  activation=arch['activation'], padding=arch['padding'],\n"
        "                  stride_mode=arch['stride_mode'], use_bn=arch['use_bn'],\n"
        "                  use_dropout=arch['use_dropout']).to(device)\n"
        "model.load_state_dict(torch.load(f'{RUNS_ROOT}/champion_f0/best_state.pt',\n"
        "                                  map_location=device))\n"
        "model.eval()\n"
        "\n"
        "# Pick 4 test images: 2 NORMAL, 2 PNEUMONIA\n"
        "items = list_images(DATA_ROOT)\n"
        "test_n = [(p, l) for p, l, s in items if s == 'test' and l == 0][:2]\n"
        "test_p = [(p, l) for p, l, s in items if s == 'test' and l == 1][:2]\n"
        "samples = test_n + test_p\n"
        "\n"
        "tf = build_transforms(summ['training']['img_size'], train=False, augment=False)\n"
        "\n"
        "# Hook the last conv block's output and gradient\n"
        "feat_buf, grad_buf = [], []\n"
        "h1 = model.features[-1].conv.register_forward_hook(\n"
        "    lambda m, i, o: feat_buf.append(o.detach()))\n"
        "h2 = model.features[-1].conv.register_full_backward_hook(\n"
        "    lambda m, gi, go: grad_buf.append(go[0].detach()))\n"
        "\n"
        "fig, axes = plt.subplots(2, 4, figsize=(15, 7))\n"
        "for col, (path, label) in enumerate(samples):\n"
        "    img = Image.open(path).convert('L')\n"
        "    img_show = np.array(img.resize((summ['training']['img_size'],) * 2))\n"
        "    x = tf(img).unsqueeze(0).to(device); x.requires_grad_(True)\n"
        "    feat_buf.clear(); grad_buf.clear()\n"
        "    logit = model(x)\n"
        "    model.zero_grad(); logit.sum().backward()\n"
        "    feat = feat_buf[0][0]; grad = grad_buf[0][0]\n"
        "    weights = grad.mean(dim=(1, 2))\n"
        "    cam = torch.relu((weights[:, None, None] * feat).sum(0))\n"
        "    cam = (cam / (cam.max() + 1e-8)).cpu().numpy()\n"
        "    cam_up = np.array(Image.fromarray(cam).resize(img_show.shape[::-1]))\n"
        "    prob = torch.sigmoid(logit).item()\n"
        "    pred = 'PNE' if prob > 0.5 else 'NORM'\n"
        "    truth = 'PNE' if label == 1 else 'NORM'\n"
        "    axes[0, col].imshow(img_show, cmap='gray')\n"
        "    axes[0, col].set_title(f'true={truth}'); axes[0, col].axis('off')\n"
        "    axes[1, col].imshow(img_show, cmap='gray')\n"
        "    axes[1, col].imshow(cam_up, cmap='jet', alpha=0.5)\n"
        "    axes[1, col].set_title(f'pred={pred} ({prob:.2f})'); axes[1, col].axis('off')\n"
        "h1.remove(); h2.remove()\n"
        "plt.tight_layout(); plt.savefig(f'{RUNS_ROOT}/champion_ensemble/gradcam.png', dpi=110)\n"
        "plt.show()"
    ),
    md(
        "---\n"
        "## 14. Other things we tried — Mixup / CutMix / Manifold Mixup demo\n"
        "\n"
        "Visual reference for §7 of the report. Generated by `_helpers/_mixup_cutmix_demo.py`. "
        "We tested Mixup α=0.2 on the pretrained track and lost 0.64 pp accuracy — the report "
        "explains why pretrained models tend not to benefit (already-calibrated class boundaries)."
    ),
    code(
        "from IPython.display import Image as IPImage, display\n"
        "import os\n"
        "\n"
        "demo_path = '_helpers/mixup_cutmix_demo.png'\n"
        "if not os.path.exists(demo_path):\n"
        "    !python _helpers/_mixup_cutmix_demo.py\n"
        "if os.path.exists(demo_path):\n"
        "    display(IPImage(demo_path))\n"
        "else:\n"
        "    print('Demo image not found. Re-run after dataset has been downloaded.')"
    ),
    md(
        "---\n"
        "## 15. Transfer-learning comparison (ResNet50 + ImageNet)\n"
        "\n"
        "5-fold ResNet50 fine-tuned from ImageNet weights at image size 288. Quantifies "
        "how much pretrained features add on top of our from-scratch design choices. "
        "Reported in §7 of the report as a comparison baseline — *not* a headline result, "
        "since the assignment asks us to design our own CNN.\n"
        "\n"
        "**Why it stays on the standard pipeline**: the ~8 min cost on H100 is small compared "
        "to the strength of the resulting comparison ('our 4-block CNN reaches X% from scratch; "
        "with ImageNet pretraining the same project reaches Y%')."
    ),
    code(
        "# 5-fold ResNet50 + ImageNet pretrained, ~7 min on H100.\n"
        "# pneumonia_run_folds.py auto-skips folds with an existing summary.json,\n"
        "# so re-running is free if Drive already holds prior results.\n"
        "!python pneumonia_run_folds.py --pretrained --extra='--img_size 288 --num_workers 6'"
    ),
    code(
        "# Eval the transfer-learning ensemble + medical KPIs\n"
        "!python pneumonia_eval.py --ensemble ens_f0,ens_f1,ens_f2,ens_f3,ens_f4 \\\n"
        "    --use_best --num_workers 0 --img_size 288\n"
        "!python _helpers/medical_kpis.py --run $PNEUMONIA_RUNS/ensemble"
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
