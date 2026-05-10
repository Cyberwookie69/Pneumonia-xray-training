# Architecture Log

Running journal of architectural choices, changes, and the reasoning behind them.
Append a new entry every time a non-trivial decision is made or reversed.

Times before approximately 2026-05-07 09:00 are reconstructed from chat history
and may be off by a few minutes.

---

## 2026-05-06

### 22:50 — Project bootstrapped
- Placed `kaggle.json` in `c:\temp\pneumonia\` (alongside the script, not in `~/.kaggle/`).
- Created empty `pneumonia.py`.

### 22:53 — `pneumonia.py` v1 — dataset downloader
- Uses `kaggle` Python SDK (not the CLI).
- Sets `KAGGLE_CONFIG_DIR` to the script directory so the local `kaggle.json` is picked up.
- Downloads `paultimothymooney/chest-xray-pneumonia` (~2.3 GB).

### 23:00 — `pneumonia.py` v2 — skip if zip exists
- Added `ZIP_PATH.exists()` check to avoid re-downloading.
- Switched `unzip=True` → `unzip=False` (the Kaggle API still unzipped because
  the zip from a prior run was already extracted).

### 23:10 — `_top_kernels.py` — list public notebooks for the dataset
- Used `api.kernels_list(..., sort_by="voteCount", page_size=100)`.
- Pulled 1000 notebooks, sorted by `total_votes` (snake_case attribute, not camelCase).
- Discovered `sort_by="scoreDescending"` does NOT order by votes — sort manually.

### 23:12 — `_pull_top10.py` — download top 10 notebooks
- Saved each as `<rank>_<votes>_<author>__<slug>/` for easy ranking visibility.
- Skips folders that already contain files (re-runnable).

### 23:30 — Analyzed top-10 notebooks (subagent task)
- Findings drove our architecture choices below.
- Key takeaways: pretrained backbone almost always beats from-scratch; class
  imbalance handling matters; TTA and ensembling are common but rarely combined.

---

## 2026-05-08

### 14:00 — Cross-platform support: scripts now run on Colab/Linux/CUDA, not just Vega + DirectML
- All `pneumonia_*.py` scripts now use `get_device()` which tries DirectML →
  CUDA → CPU in that order. The `import torch_directml` is wrapped in
  `try/except` so the absence of the (Windows-only) package is harmless.
- Hardcoded `c:\temp\pneumonia\...` paths replaced with
  `Path(__file__).resolve().parent` defaults plus `PNEUMONIA_DATA` /
  `PNEUMONIA_RUNS` environment-variable overrides. Lets the project run
  unchanged from `/content/Pneumonia-xray-training/` on Colab or
  `~/projects/...` on Linux.
- `pneumonia_run_folds.py` now uses `sys.executable` instead of a hardcoded
  venv path, so it works in any Python environment.
- New `pneumonia_colab.ipynb` notebook gives a one-click pipeline: clone,
  install deps, upload `kaggle.json`, download dataset, train, eval, plot.
- **Speed reality check:** the same 5-fold ResNet50 @ 288 run takes
  ~25 minutes on a free Colab T4 vs. ~7 hours on Vega 64 + DirectML —
  roughly a 17× speedup, consistent with our earlier estimate.

### 09:30 — Implemented SNRAdamW optimizer (Litman & Guo 2026)
- Source: "A Theory of Generalization in Deep Learning", arXiv:2605.01172,
  published May 2026.
- The paper introduces a per-parameter SNR gate over the standard Adam update.
  Each step, the gate evaluates `q = max(0, (μ² − σ²/(b−1)) / (σ² + ε))`
  using the EMA mean `μ` and EMA variance `σ²` of the gradient. Updates are
  multiplied by this gate, suppressing parameters whose minibatch
  signal-to-noise ratio sits below threshold.
- Cost: one extra state vector per parameter (the variance EMA). Same
  wall-clock per step.
- **Implementation**: `SNRAdamW` class in `pneumonia_train.py` subclasses
  `torch.optim.AdamW`. New flag `--snr_optimizer` swaps it in for both
  Phase 1 and Phase 2 of training. Saved in checkpoint summary as
  `optimizer: "snr_adamw"`.
- **Smoke**: 10 steps on ResNet50 + real chest X-rays — no crash, loss falls,
  gate distribution healthy (mean 0.024, 20% of params get non-zero update
  per step, max 0.74). DirectML compatible.
- **Why we care**: zero notebooks in our top-100 audit use this technique
  (it's literally a-week-old from the paper). Even if the empirical gain on
  this dataset is modest (~+0.3-1%, since the paper targets noisier
  domains), it's a uniqueness signal for the AI/ML report — a
  theoretically-motivated, brand-new ablation row that no competitor has.
- **Trade-off**: theory's strongest claims (5× grokking, 2.4× PINN) are on
  synthetic / structured-noise domains, not standard supervised vision. The
  Kaggle pneumonia dataset has known label noise (~5-10% per literature),
  so we expect *some* benefit, but it won't be dramatic.

---

## 2026-05-10

### Default seed bumped 72 → 78 (no additional invalidation)
- Same set of files as the 42 → 72 change below; no runs were trained
  under seed=72 in between, so this is effectively a single 42 → 78
  change as far as cached results are concerned.

### Default seed flipped 42 → 72 — INVALIDATES every prior summary.json
- Changed default `--seed` from 42 to 72 in: `pneumonia_train.py`,
  `pneumonia_cnn_custom.py`, `pneumonia_biomedclip.py`, `pneumonia_rad_dino.py`,
  `pneumonia_gradcam.py`, `pneumonia_plots.py`, and the hardcoded seeds in
  `_helpers/_mixup_cutmix_demo.py`.
- **Why this is breaking**: every random operation downstream of the seed
  changes. K-fold splits compose differently, train_test_split picks a
  different val set, model weights initialise from different random values,
  WeightedRandomSampler shuffles differently, t-SNE init differs.
  Every `summary.json`, `test_probs.npy`, `best_state.pt`, `history.json`,
  and `medical_kpis.json` produced under seed=42 is now stale relative
  to the codebase defaults.
- **What is NOT invalidated**: cached frozen features in
  `runs/rad_dino_features/` and `runs/biomedclip_features/` (depend only
  on model + image bytes, not on seed). The 5-fold linear classifiers
  trained on top of those features ARE invalidated though, since the
  classifier training seeds the K-fold split.
- **What you must do to actually re-run with seed=72**: delete the
  affected `runs/<name>/` folders or pass an explicit `--run_name`
  with a new tag (e.g., `champion_f0_s72`) so the auto-skip in
  `run_if_missing` doesn't keep the seed=42 results around. The current
  `--run_name` defaults all stay the same, so without manual cleanup
  the notebook will skip every previously-completed row.

### Academic refocus: project scope cut to match the assignment
- The assignment asks three specific CNN-design questions: (Q1) number of
  conv-pool blocks, (Q2) strides/padding/activation, (Q3) overfitting solution.
  The 4-rung complexity ladder we built (ResNet50 from scratch → +pretrained →
  +ConvNeXt → +SNR-AdamW + 15-model multi-arch ensemble) didn't directly answer
  any of these — it was a SOTA-tactics study, not a CNN-design study.
- **Decision**: refactor to a custom-CNN-with-ablations pipeline as the main
  story, with the previous transfer-learning work demoted to one comparison
  baseline in §7 of the report.
- New core file: `pneumonia_cnn_custom.py`, a parametric from-scratch CNN with
  flags for every architectural choice (`--n_blocks`, `--activation`,
  `--padding`, `--stride_mode`, `--use_bn`, `--use_dropout`, `--weight_decay`,
  `--augment`, `--early_stop_patience`). One script generates every ablation
  row across A1, A2, A3.
- New notebook structure (`pneumonia_colab.ipynb`, regenerated by
  `_helpers/build_colab_notebook.py`): smoke test → A1 (depth) → A2
  (stride/padding/activation) → A3 (overfitting) → champion 5-fold →
  medical KPIs → learning curves → Grad-CAM → Mixup demo (kept as visual
  in §7) → optional transfer-learning baseline.
- Pre-refactor state preserved at git tag `pre-academic-refocus` and heavy
  run artefacts moved to `archief2/` (gitignored). The transfer-learning
  ensemble (`runs/ensemble`) and per-fold ResNet50/ConvNeXt/SNR-AdamW runs
  all live there now; reachable for the §7 comparison via README.

### Medical KPI helper (`_helpers/medical_kpis.py`)
- Standalone post-hoc KPI computation from `test_probs.npy` + `test_labels.npy`.
- Reports the four assignment-relevant medical KPIs:
  Sensitivity (PNE recall), Specificity (NORM recall), AUROC, ECE.
- Reports three operating points: default τ=0.5, val-tuned best-accuracy τ,
  and sensitivity-targeted τ ≥ 0.97 (the clinical screening default).
- 15-model ensemble (gearchiveerd in `archief2/runs/ensemble`) measured:
  AUROC 0.984, sens 0.977, spec 0.880, ECE 0.164. The high ECE indicates
  the ensemble is *under-confident* — temperature scaling on val would
  fix it but is intentionally NOT added (out of scope for the new academic
  focus; documented as a future-work item).

## 2026-05-09

### Colab notebook: Drive-based Kaggle credentials by default
- Section 3 of `pneumonia_colab.ipynb` now mounts Drive **and** copies
  `kaggle.json` from `My Drive/kaggle.json` into `~/.kaggle/` if present.
  The upload widget in section 4 is now a fallback that detects the file
  is already in place and prints a "no need to upload" notice.
- Why: the `files.upload()` widget hangs indefinitely waiting for a click
  every Colab session. With the JSON on Drive, "run all cells" works
  end-to-end without manual intervention.
- Trade-off: kaggle.json now lives on the user's Drive. Anyone with
  share-access to that Drive folder can use the API key. Mitigated by
  documenting it explicitly + recommending a private folder.
- Persistence (`PNEUMONIA_RUNS=/content/drive/MyDrive/pneumonia_runs`)
  is now part of the same cell — no longer a separate optional step.

### Colab notebook restructured as 4-rung complexity ladder
- `pneumonia_colab.ipynb` rewritten so each training section is one rung
  with explicit framing of what it adds vs. the previous rung:
  1. ResNet50 from scratch (baseline) → `scratch_f0..f4`
  2. + ImageNet pretraining → `ens_f0..f4`
  3. + ConvNeXt-Tiny → `cnx224_f0..f4` (10-model ensemble at this rung)
  4. + SNR-AdamW ResNet50 → `snr_r50_f0..f4` (15-model multi-arch headline)
- Eval cell prints scores per rung in order, so the "report table" is just
  a copy from the cell output. Per-architecture references (ConvNeXt alone,
  SNR alone) are also printed.
- Why this structure: per user request, the project should *show* increasing
  complexity and the corresponding accuracy gains, not just hand the reader
  a peak number. Each rung-to-rung delta isolates one design choice.
- Notebook is now generated by `_helpers/build_colab_notebook.py` (single
  source of truth, easier to maintain than editing JSON cell-by-cell).
- Total wall-clock on a free T4: ~110 min (4× 25 min training + setup +
  eval/plots), still inside the ~12 h Colab session limit.

### Colab notebook: 3-architecture default pipeline (superseded same day)
- Earlier in the day the notebook gained ConvNeXt-Tiny and SNR-AdamW as
  default training tracks (previously optional cells) and a 15-model
  multi-arch eval. Replaced a few hours later by the 4-rung structure
  above when the user asked for an explicit complexity ladder including
  the from-scratch baseline.

### Explicit `--device` selection in `pneumonia_train.py`
- Added `--device {auto,dml,cuda,cpu}` CLI flag. Default `auto` keeps the
  existing DirectML → CUDA → CPU fallback chain unchanged.
- `get_device()` now takes a `prefer` argument; non-`auto` choices fall back
  to CPU (with a clear message) if the requested backend is unavailable.
- Motivation: lets a user pin a specific backend for benchmarking or debugging
  without editing code, complementing the cross-platform work from 2026-05-08.
  Notebook (`pneumonia_colab.ipynb`) needs no change — Colab still hits the
  `auto` path and lands on CUDA.

---

## 2026-05-07

### 00:00 — Defined the goal
- Target: ≥97.7% test accuracy on the official Kaggle test set (624 images).
- Stretch goal: beat the top-10 notebooks (best published ~98.55%).
- Hardware: AMD Vega 64 (8 GB) on Windows 11.

### 00:05 — Software stack decision
- Python **3.11.9** (installed via `winget`, alongside existing 3.13).
  Reason: `torch-directml` only ships wheels for Python ≤ 3.11.
- venv at `c:\temp\pneumonia\.venv311\`.
- Packages: `torch-directml 0.2.5`, `torch 2.4.1`, `timm 1.0.26`,
  `scikit-learn 1.8.0`, `kaggle 2.1.2`.
- Confirmed DirectML detects "Radeon RX Vega" as device 0.

### 00:10 — Initial model choice: EfficientNetV2-B0 (`tf_efficientnetv2_b0.in1k`)
- Smoke test forward pass: OK (0.56 s for batch=8 at 260×260).
- Smoke test backward pass: **FAIL** — `RuntimeError: ensure_in_bounds`. The
  TF-style "same" padding produces strided tensors that DirectML's autograd
  cannot handle.

### 00:15 — Backbone changed: ConvNeXt-Tiny (`convnext_tiny.fb_in22k_ft_in1k`)
- Smoke test forward+backward at batch=16, 224×224: OK (~2 s/step).
- Reason: PyTorch-native architecture (no TF padding tricks), strong transfer
  learning, IN22k pretraining typically transfers well to medical imaging.

### 00:30 — First real training run — **CRASH**
- `RuntimeError: The GPU device instance has been suspended.` Windows TDR.
- Hypothesis: `binary_cross_entropy_with_logits` falls back to CPU because
  `log_sigmoid` is not implemented on DirectML, causing CPU↔GPU memory ping-pong.

### 00:40 — Fix: hand-rolled `FocalLoss` and `BCEManual`
- Implements focal loss using only `torch.sigmoid`, `clamp`, `where`, `pow`,
  `log` — all GPU-resident on DirectML.
- Avoids the CPU fallback during loss computation entirely.

### 00:45 — Discovered freeze-logic bug
- Code froze the backbone via `'fc' in name` substring match.
- For ConvNeXt this matched **every MLP block** (`stages.X.blocks.Y.mlp.fc1/fc2`),
  resulting in 25.9 M "trainable" params instead of ~2 K.
- Fix: use `model.get_classifier()` and compare param IDs.

### 00:50 — Backbone changed: ResNet50 (`resnet50.a1_in1k`)
- ConvNeXt-Tiny still crashed during longer training despite the fixes above.
- ResNet50 is a known-stable baseline on DirectML, lighter (~92 MB weights),
  and well-published on this dataset (~95-97% range).
- Trade-off: ConvNeXt likely a few % higher ceiling, but reliability trumps.

### 01:00 — Crash-resilience hardening (after exit-5 silent crash mid-Phase 2)
- Per-epoch checkpoint saved as `<run_dir>/last_state.pt` (model + progress + best_state).
- New `--resume` flag picks up at the last completed epoch.
- `eval_batch_size` (default 4) decoupled from `batch_size` (default 8) — smaller
  eval batches reduce DirectML memory pressure.
- TTA disabled during per-epoch validation; only used on the final test eval.

### 01:30 — Throughput tuning
- Bumped `num_workers` 0 → 4. GPU utilization jumped from ~35% to ~94% — the
  bottleneck was single-threaded JPEG decoding, not compute.
- Batch tried at 12 (3.7× higher items/s than batch=8 + workers=0), but went back
  to 8 to be safe with respect to OOM.

### 02:30 — First complete training (15 epochs, single fold, ResNet50)
- Best val_acc: 0.9485 (at P2 epoch 13).
- TEST eval crashed silently at the end (TTA-related).
- Recovered the test-acc later via standalone eval: **0.9311** without TTA.
- Confusion: TP=378, TN=203, FP=31, FN=12. Model over-predicts pneumonia.

### 09:00 — Created `pneumonia_eval.py` (separate from training)
- Reason: training and eval crash at different points; coupling them lost training
  progress on every eval crash. Separation lets us re-eval saved checkpoints
  freely with different settings (TTA on/off, threshold, ensemble).
- Loads `last_state.pt`, can use either `last` or `--use_best` state.
- Reports confusion matrix + per-class metrics.

### 10:00 — Decided on the "volledige route" to push past ResNet50 single-fold ceiling
- Plan: Mixup + EMA + 5-fold ensemble + TTA + threshold tuning.
- Expected impact (additive, but with diminishing returns):
  - Mixup: +1-1.5%
  - EMA: +0.5-1%
  - 5-fold ensemble: +1-2%
  - TTA: +0.5-1%
  - Threshold tuning: +0.5-1.5%
- Realistic landing zone: 96-98% test accuracy.

### 10:30 — Added Mixup
- Beta(α, α) blend of two random images per batch, applied with probability `mixup_prob` (default 0.5).
- Default α = 0.2 (typical for image classification).
- Loss for mixup batch: `λ * loss(logits, y_a) + (1-λ) * loss(logits, y_b)` —
  works with the existing FocalLoss without needing soft-target support.

### 10:45 — Added EMA (Exponential Moving Average) of weights
- Class `ModelEMA` keeps shadow weights on **CPU** (saves VRAM on the 8 GB Vega).
- Updated after every optimizer step.
- Validation evaluates the EMA shadow (temporarily swapped into the live model),
  not the live weights — this is what gets saved as `best_state` when EMA is on.
- Default decay: 0.999. For runs shorter than ~1000 steps this barely moves;
  for our real run (~7800 updates per fold) it converges well.

### 11:00 — Added `pneumonia_run_folds.py` — k-fold runner
- Each fold runs in its own subprocess. Reason: a fresh Python process means a
  fresh DirectML context, which means leaked GPU state from fold N can't take
  down fold N+1.
- `--start_fold N` resumes after a crash (skips already-completed folds).
- Train script always passes `--resume`, so a partially-completed fold continues
  rather than restarting.

### 11:15 — Updated `pneumonia_eval.py` for ensembles + advanced metrics
- `--ensemble run0,run1,...` averages probabilities across N checkpoints.
- Threshold sweep over [0.30, 0.70] to find the optimum (rarely 0.5).
- Cohen's kappa with Landis-Koch interpretation (so we can compare to
  inter-radiologist agreement of ~0.6-0.85).
- ROC-AUC, calibration-by-confidence bins, borderline-case count, confidently-wrong count.

### 11:30 — Removed final test-eval from training script (made optional via `--final_test`)
- Reason: every silent crash we ever saw was during the final test-eval. Splitting
  it out means training reliably exits with code 0 and `pneumonia_eval.py`
  handles inference where it can be retried independently.

### 12:00 — TTA fix attempt 1 (failed)
- Refactored `evaluate()` to do all original batches first, then all flipped
  batches (instead of alternating per batch). Theory: alternating flip pattern
  was the trigger.
- Result: still crashes silently. Theory was wrong. The crash is `torch.flip`
  on the GPU itself, regardless of when it's called.

### 12:30 — TTA fix attempt 2 (worked)
- Moved horizontal flip to **CPU**, inside the `torchvision.transforms` pipeline
  (`build_transforms(..., hflip_eval=True)`).
- `evaluate()` now takes an optional second `loader_flip` whose dataset
  pre-flips images in worker processes.
- The model never sees a `torch.flip` op — it just sees normal forward passes.
- Validated: smoke ensemble eval with TTA now completes; +1.1% accuracy boost
  vs no-TTA (on a deliberately undertrained 2-epoch smoke test).

### 13:00 — Comments rewritten to English with dry humor
- All four scripts (`pneumonia.py`, `pneumonia_train.py`, `pneumonia_eval.py`,
  `pneumonia_run_folds.py`) updated.
- Comments now document the *why* — usually involving a DirectML workaround.

### 13:30 — Smoke results validated full pipeline
- 2 folds × 2 epochs (with mixup, no EMA): per-fold 0.875, ensemble 0.8846.
- Threshold tuning (best=0.44): 0.9071.
- Cohen's kappa 0.8017 (almost perfect range).
- Ready for the real 5×15 run.

### 14:00 — Architecture constraint: must be a CNN
- The assignment brief is explicit: "train the model with **CNN**" and "choose
  the **number of convolution-pooling building blocks**, the **strides,
  padding and activation function**".
- This excludes pure Vision Transformers (ViT, DINOv2, Swin, BEiT, MAE, …)
  regardless of their pretraining data — they have no convolutions, no
  pooling blocks, no strides/padding to choose.
- **Allowed:** ResNet, DenseNet, VGG, Inception, Xception, MobileNet,
  EfficientNet, ConvNeXt (the paper's title is literally "A ConvNet for the
  2020s", uses depthwise + pointwise convolutions throughout).
- **Forbidden:** any backbone whose forward pass is dominated by self-attention
  rather than convolutions.
- **Impact on planned work:** F5 (DINOv2) is dropped. Replacement candidates
  for the "modern backbone" ablation row: `convnext_tiny.fb_in22k_ft_in1k`
  (modern CNN with IN22k pretraining) or `efficientnet_b3` (proven CNN family).

### 14:30 — Constraint locked in: no medical-domain external data
- **Allowed:** generic pretraining (ImageNet, LVD-142M and similar non-medical
  corpora), since none of those contain X-rays.
- **Forbidden:** any pretraining or augmentation that uses medical images
  (CXR-Foundation, BiomedCLIP, MedCLIP, RoentGen-generated synthetic X-rays).
- **Forbidden:** evaluating on or mixing in any other X-ray dataset (RSNA
  Pneumonia, NIH ChestX-ray14, etc.).
- **Reason:** assignment rule — model must be based on the original Kaggle
  pneumonia dataset only.
- **Impact on current pipeline:** none. ResNet50 with `in1k` weights is fine
  (zero medical images in ImageNet). The 5-fold ensemble run can continue
  unchanged.
- **Impact on planned future work:** F3, F4, F9 dropped; F5 still OK.

### 16:00 — Project forks into two parallel variants: pretrained vs. from-scratch
- **Variant A (pretrained):** ResNet50 with ImageNet `in1k` weights (current
  approach). Full pipeline: 2-phase training, focal loss, mixup, EMA, k-fold.
- **Variant B (from-scratch):** Same architecture, random init. Phase 1 skipped
  (no pretrained backbone to thaw). Everything else shared with Variant A.
- **Reason:** the assignment may forbid pretraining in stricter readings; we
  also want a clean A/B comparison to quantify the value of transfer learning
  on this dataset.
- **Implementation:** single codebase with a `--from_scratch` flag, which
  toggles `timm.create_model(pretrained=...)`. Auto-forwarded by
  `pneumonia_run_folds.py --from_scratch`, which switches the run-name prefix
  from `ens_` to `scratch_`. Checkpoints record `model_tag` so the eval
  rehydrates the right architecture without operator input.
- **Recommended pairing:** for from-scratch use a small `--model` (e.g.
  `resnet18`, ~11M params) — `resnet50` from random init on 5K images
  overfits hard.
- **Maintenance rule (carried by the user):** any future architectural change
  that is not pretraining-related must be applied to both variants — i.e.
  changes flow through the shared training/eval/runner code, not through one
  variant's branch.

### 17:30 — First 5-fold ensemble result: ensemble (0.9247) underperforms single-fold baseline (0.9311)
- Setup: ResNet50 + Mixup α=0.2 + EMA decay 0.999, 5-fold, full TTA at eval.
- Per-fold test acc: 0.877, 0.918, 0.896, 0.928, 0.886 → mean 0.901.
- Ensemble (mean of probabilities): 0.9247. ROC-AUC 0.9733. Cohen's κ 0.84.
- Threshold sweep didn't help (best=0.5).
- **Diagnosis:** Mixup at α=0.2 is likely too aggressive for a pretrained
  backbone — adds noise on top of already-strong priors. EMA at 0.999 had a
  half-life (~700 steps) longer than the effective length of most folds,
  which early-stopped at epoch 8-13. Folds also lost ~500 train images each
  (80/20 vs the baseline's 90/10), which hurt the per-fold ceiling.
- **Decision:** roll back the runner defaults to mixup=0.0 and ema=0.0 (i.e.
  the recipe that gave 93.11%) and re-run. Ensemble averaging alone should
  push past baseline. Mixup/EMA stay available behind explicit flags so the
  from-scratch variant (where they're more likely to help) can still use them.

### 17:45 — Codebase simplification: dropped `FromScratchCNN` class
- Removed the ~60-line custom CNN class. Replaced with `timm.create_model(
  pretrained=not args.from_scratch)` — same architecture, just different
  initialisation.
- Eval lost its arch-detection branch; now it always goes through timm and
  strips the `_scratch` suffix from the saved tag.
- **Reason:** less code to maintain. Reduces surface for bugs and keeps
  comparisons cleaner (same architecture, only init differs).
- **Trade-off:** the from-scratch variant no longer uses a tiny CNN matched
  to the competing solution. User is expected to pair `--from_scratch` with
  a small `--model` flag for a sensible from-scratch run.

### 18:30 — Realistic targets per variant + primary/alternative role split
- **Variant A (pretrained, transfer learning):** target **97-98% test acc**.
  Realistic given the literature ceiling on this dataset (~98-98.5% with
  serious tricks) and our pipeline (5-fold ensemble + TTA + threshold tuning,
  optionally with stronger augmentation or a DINOv2 backbone).
- **Variant B (from-scratch):** target **85-90% test acc**. Hard ceiling
  imposed by 5K training images and ~1-25M random-init params — even with our
  full pipeline (mixup+EMA *do* help here unlike for transfer learning),
  ensemble, TTA, threshold tuning. Beats the competing solution's 71.79% by
  a wide margin.
- **Strategic role:**
  - **A is the primary submission** if the assignment permits any pretraining
    on non-medical data (likely interpretation).
  - **B is the alternative** if the assignment forbids pretraining entirely.
  - Both are maintained as parallel tracks in the same codebase via a single
    `--pretrained` flag (see entry below).

### 18:45 — Default flipped: `--pretrained` opt-in (was `--from_scratch` opt-in)
- Replaced `--from_scratch` flag with `--pretrained` (inverted logic).
  Default behaviour is now **from-scratch** (no ImageNet weights loaded).
- Pass `--pretrained` to opt into the transfer-learning variant.
- **Reason:** the user wants the assignment-constrained baseline as the
  default, since the strict reading of the assignment may forbid pretraining.
  Pretraining stays one explicit flag away.
- **Implementation:** `pneumonia_train.py` and `pneumonia_run_folds.py` both
  switched. Run-name prefix flips automatically: default is `scratch_`,
  `--pretrained` switches to `ens_`. Checkpoint stores `pretrained: bool`
  for eval-time arch reconstruction.

### 19:30 — F1 implemented: Grad-CAM heatmaps on trained checkpoints
- New script `pneumonia_gradcam.py`. Loads any of our checkpoints, runs Grad-CAM
  on N random test images (default 6) or a single user-supplied image, and
  saves a side-by-side `original | heatmap | overlay` PNG per sample plus a
  `summary.json` with the predictions.
- Library: `pytorch-grad-cam` (`pip install grad-cam`). Auto-detects the final
  conv stage of timm models by walking common attribute names
  (`layer4`, `stages`, `blocks`, `features`).
- Custom single-logit target wrapper because pytorch-grad-cam's default
  `ClassifierOutputTarget` assumes multi-class softmax — our binary head is a
  single sigmoid logit.
- **First impression:** on `ens_f3` test samples, the model focuses on the
  correct lung field for PNEUMONIA cases (consolidations are highlighted),
  but on NORMAL cases it sometimes attends to the diaphragm/abdomen region
  rather than the lung fields — i.e. predicts "normal" via *absence* of
  features, not via positively recognising healthy anatomy. Useful diagnostic
  signal, exactly the value Grad-CAM is supposed to add.

### 20:00 — Critical finding: most "99%" claims in the top-100 use a re-split test set, not the official Kaggle test
- Inspected the data pipelines of notebooks claiming 99-100% test accuracy
  (#32 abdallahwagih EfficientNetB0, #60 fathyfathysahlool EfficientNetB7,
  #67 minawagdy EfficientNetB3, #73 yasserlatreche DenseNet121).
- Three of them (#32, #60, #67 — clearly forks of the same template) build a
  dataframe from the **`/train` folder only** (5216 images), then
  `train_test_split` it into 80/12/8 and call the 8% slice "test". They
  **completely ignore** the official `/test` folder.
- This means their reported "test acc" is effectively a *train-distribution*
  metric: same 74% pneumonia prior as training, possible patient overlap,
  no held-out distribution shift. 99% on that is roughly the same task
  difficulty as 99% on the train set itself.
- **Implication:** our 93.11% on the official Kaggle test (624 images, 62.5%
  PNE prior, real held-out) is a much stronger result than their headline
  numbers, even though it looks lower on paper. Same lesson as the Liverpool
  team's 97.6% val / 71.79% test gap.
- **Decision:** continue evaluating on the official `/test` folder only. Do
  not adopt the re-split-train approach even though it would inflate our
  numbers. Document this in the eventual report so reviewers don't compare
  apples to oranges with the literature.

### 18:50 — Top-100 community audit: how do peers split on pretraining?
- Pulled 100 highest-voted notebooks for this dataset; deduped to 94 unique.
- **Split:** 45 PRETRAINED (48%), 42 FROM-SCRATCH (45%), 7 AMBIGUOUS-but-
  probably-pretrained (7%). Net: ~55% pretraining, ~45% from-scratch.
- **Backbone popularity** (among pretrained notebooks):

  | Rank | Backbone | Count |
  |---:|---|---:|
  | 1 | VGG16 | 18 (40% of pretrained!) |
  | 2 | DenseNet121 | 6 |
  | 2 | ResNet50V2 | 6 |
  | 4 | ResNet50 | 5 |
  | 5 | InceptionV3 | 4 |
  | 5 | VGG19 | 4 |
  | 7 | Xception | 3 |
  | 7 | MobileNetV2 | 3 |
  | — | EfficientNetB0/B3/B7 | 3 (combined) |
  | — | ResNet34/101/152V2 | 3 (combined) |
  | — | DenseNet161 | 1 |

- **Notable absence:** zero ConvNeXt, zero ViT/DINOv2, zero EfficientNetV2,
  zero EVA. The community uses 2014-2019 architectures. Our `resnet50.a1_in1k`
  recipe (2021) and our DINOv2 plan (F5) are both more modern than the
  consensus.
- **Misleading-title finding:** notebook #50 "pneumonia-detection-from-scratch"
  is in fact pretrained (loads `vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5`
  from `/kaggle/input/`). Several "from-scratch" claims in the audit
  similarly use pretrained weights via custom file paths — title vs. code
  mismatch is common.

---

## Open decisions / not yet tried

> Rescoped 2026-05-07 ~21:00 once the project's audience shifted from a
> medical-relevance frame to an AI/ML coursework frame. Items that no longer
> fit (clinical safety framing, niche techniques) are dropped; items that
> demonstrate ML methodology and yield strong report figures are added.

### Tactical (existing-architecture tweaks)

| # | Idea | Reason to try | Reason to skip |
|---|---|---|---|
| T1 | ConvNeXt-Tiny again with crash-resilient setup | Higher ceiling, modern arch (would be a unique ablation row) | DirectML risk on `tf_*` variants |
| T2 | CutMix in addition to / instead of Mixup | Better for from-scratch variant where regularisation matters | Modest effect on pretrained |
| T3 | Random Erasing | Cheap regularisation | Largely covered by Mixup |
| T4 | RandAugment / TrivialAugment | Stronger augmentation, especially for from-scratch | Existing aug already reasonable for pretrained |
| ~~T5~~ | ~~Mixing in RSNA / NIH ChestX-ray14~~ | — | **BLOCKED** by no-other-data rule |
| T6 | Higher resolution (288 / 320) | Marginal gain | ~50% slower per epoch on Vega 64 |
| T7 | Label smoothing | Cheap, decent ablation row to add | Mixup already supplies soft targets |

### Strategic — directions still queued (post-rescoping)

| # | Direction | Why kept | Effort | Notes |
|---|---|---|---|---|
| ~~F1~~ | ~~Grad-CAM heatmaps~~ | DONE 19:30 | — | Implemented in `pneumonia_gradcam.py` |
| ~~F2~~ | ~~Conformal prediction~~ | DROPPED — clinical-safety framing, not central to AI report | — | Cool but not curriculum-aligned |
| ~~F3~~ | ~~CXR-Foundation backbone~~ | — | — | **BLOCKED** by no-medical-data rule |
| ~~F4~~ | ~~External validation on RSNA / NIH~~ | — | — | **BLOCKED** by no-other-data rule |
| ~~F5~~ | ~~DINOv2 backbone~~ | — | — | **BLOCKED** by assignment "must be a CNN" rule. DINOv2 is a Vision Transformer (no convolutions, no pooling blocks, no strides/padding choices to make). Even though LVD-142M pretraining is allowed, the architecture itself disqualifies. |
| ~~F6~~ | ~~Test-Time Adaptation (Tent / EATA)~~ | DROPPED — niche technique, low pedagogical value for an AI report | — | Interesting research direction but off-curriculum |
| F7 | **CutMix** alongside Mixup | Useful as ablation row in Variant B (from-scratch where regularisation matters more) | Low | Implements as new branch in `train_one_epoch` |
| F8 | **Knowledge distillation** ensemble → single student | Classic ML topic, demonstrable result ("5-fold @ 95% → 1 model @ 94%") | Medium | Separate training run after ensemble; nice "story" element |
| ~~F9~~ | ~~Synthetic data via RoentGen~~ | — | — | **BLOCKED** by no-medical-data rule |

### New AI/ML report-focused additions

These were added after the audience rescope. Most are figure/analysis tasks
rather than training experiments — they demonstrate ML methodology and
produce material directly usable in the report.

| # | Direction | Why it earns marks | Effort |
|---|---|---|---|
| N1 | **Multi-seed runs** (3 seeds × 1 fold) | Real error bars on accuracy. Demonstrates variance-awareness rather than single-run reporting. | ~3× one fold ≈ 1.5h |
| N2 | **Learning-curve plots** (per fold + mean) reading `history.json` | Standard ML report figure — shows convergence + overfitting. | Low (script in progress) |
| N3 | **t-SNE / UMAP of penultimate features**, coloured by true label | Shows you can analyse representations, not only metrics. Often the strongest "I understand my model" signal in a report. | Low (script in progress) |
| N4 | **Ablation matrix** (pretrained × mixup × EMA × ensemble) | Methodology centrepiece. Combine our completed runs into one table. | Low (analysis, no new training) |
| N5 | **Reliability diagram** for calibration | Turns our existing confidence-bin data into a publication-quality figure. | Low |
| N6 | **Confusion-matrix heatmap** at the chosen threshold | Standard rapport figure | Low |
| N7 | **Theoretical appendix** on focal loss + mixup math | Demonstrates derivational understanding, not just usage. | Writing only |

### Composition of items in the current pipeline

- **F5** is a backbone swap → `--model vit_small_patch14_dinov2.lvd142m --pretrained` and a separate run-name prefix.
- **F7** plugs into the existing Mixup branch in `train_one_epoch`.
- **F8** is a separate post-ensemble training run; load saved `test_probs.npy` from each fold, train a student against the soft-mean target.
- **N1** is a runner change (loop seeds), not a code change.
- **N2-N6** are figure-producing scripts that read existing artefacts (`history.json`, `summary.json`, saved checkpoints).
- **N7** is purely written content for the report appendix.

---

## Hardware reality check

| Component | Value |
|---|---|
| GPU | AMD Radeon RX Vega 64, 8 GB HBM2 (GCN5, 2017) |
| Backend | DirectML (ROCm doesn't support GCN5 anymore) |
| Effective throughput | ~10-15 it/s P1 (head-only), ~6 it/s P2 (full backward), batch 8-12 |
| One full fold (15 ep, ResNet50, mixup+EMA) | ~45 min |
| Five-fold ensemble run estimate | ~3.5-4 hours |

