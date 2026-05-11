# Changelog

Retro-assigned semantic versions for `pneumonia_colab.ipynb` and the builder
`_helpers/build_colab_notebook.py`, derived from the git history of those
two files. Convention:

- **MAJOR** — breaks the run sequence, required env-vars, or GPU expectations
- **MINOR** — new pipeline cells, new ablation/approach, new appendix section
- **PATCH** — polish, clarifications, parameter bumps, bugfixes

Only the **1.0.0** stamp is materialised in the notebook itself
(`NOTEBOOK_VERSION` constant in the builder). Earlier versions are
retrospective labels for the historical state — the commit hash is the
authoritative anchor.

---

## [1.0.1] — 2026-05-11
**Drop duplicate §14 Mixup demo cell + renumber §15–§28 → §14–§27.**
Old §14 was a verbatim duplicate of §23 (same image, near-identical
text). Removed §14, moved its `IPython.display` imports into the
surviving demo cell (now §22), and shifted all subsequent section
numbers down by 1. Notebook now 73 cells (was 75). Updated all
internal cross-references (HF token notice in §3, §16 in BiomedCLIP
note, §15–§16 in caveats footer).

## [1.0.0] — 2026-05-11 · `35430f1`
**Versioning system introduced.** First explicit `NOTEBOOK_VERSION`
constant. Title cell and clone-cell both display `v1.0.0 · 2026-05-11`.
From here on, the constant is the single source of truth.

## [0.13.0] — 2026-05-11 · `b804531`
**A4 variant-stacking sweep + Future Work section.** New §9b cell runs
one-by-one tests of label-smoothing, SWA, TrivialAugment, CutMix, and
Lion on top of the A3 winner. §10 champion now applies the validated
stacked config (smoothing + CutMix + SWA). `SELF_TRAINING_VARIANTS.md`
companion document added.

## [0.12.1] — 2026-05-11 · `457d263`
Two Claude API cells replaced with Gemini equivalents (vision critique +
conclusion drafter) — keeps the optional AI-assist cells working without
Anthropic API access.

## [0.12.0] — 2026-05-11 · `33724bc`
**Semantic-zone bullet chart for §26 headline.** Replaces the prior bar
chart with a per-metric bullet visualisation using red / green / grey
zones for clinically-unusable / sweet-zone / suspect ranges.

## [0.11.1] — 2026-05-11 · `3cb12b5`
§24 headline chart: per-metric reference lines (was a single horizontal
line for all metrics).

## [0.11.0] — 2026-05-11 · `49c293f`
**Appendix B (clinical usability thresholds) + reference lines on every
chart.** Use-case-specific minima, prevalence-PPV trade-off, improvement
roadmap.

## [0.10.0] — 2026-05-11 · `c640664`
**Appendix A — theoretical performance ceiling.** Label noise, Bayes
error, binomial noise floor, ROC trade-off derivation.

## [0.9.2] — 2026-05-11 · `a467641`
HF token now read from Colab Secrets in §3 (Drive + credentials setup) —
no more interactive prompt every session for RAD-DINO.

## [0.9.1] — 2026-05-10 · `51d37a5`
Two optional Claude cells added (vision critique + conclusion drafter).
*Superseded in 0.12.1 by Gemini variants.*

## [0.9.0] — 2026-05-10 · `ba2011e`
**BiomedCLIP added as second transfer-learning baseline** + 5-stage
methodology flow diagram + metric glossary + section renumbering.

## [0.8.2] — 2026-05-10 · `dde1b9a`
Optional Gemini text-model cell for AI-assisted chart drafting.

## [0.8.1] — 2026-05-10 · `accec35`
Optional Nano Banana renderer for the methodology flow diagram.

## [0.8.0] — 2026-05-10 · `948cd1c`
**RAD-DINO added as third transfer-learning baseline** (medical-domain
pretrained, chest-X-ray-specific).

## [0.7.2] — 2026-05-10 · `ce5d980`
§13 clarification: the per-heatmap number is *P(Pneumonia) certainty*,
not raw logits.

## [0.7.1] — 2026-05-10 · `47b1090`
Notebook polish — integer epochs, clearer Grad-CAM titles.

## [0.7.0] — 2026-05-10 · `6610120`
**Mini-report synthesis sections §16–§25 added to the notebook.** Lifts
the notebook from "experiment runner" to "end-to-end study deliverable".

## [0.6.1] — 2026-05-10 · `3f6bb9b`
Strip report/assignment references from scripts and notebook
(neutral academic framing).

## [0.6.0] — 2026-05-10 · `a2eaadf`
**§16 synthesis + He/Glorot ablation + noise-floor calculation +
methodology flow diagram.**

## [0.5.0] — 2026-05-10 · `5ee852c`
**§15 transfer-learning comparison promoted to a default pipeline
step** (was previously optional).

## [0.4.2] — 2026-05-10 · `57cc027`
Bump `--num_workers 2 → 6` in all ablation rows for H100 throughput.

## [0.4.1] — 2026-05-10 · `ec2a0a7`
Time-estimate documentation updated to H100 baseline (was T4).

## [0.4.0] — 2026-05-10 · `711132f`
**Refocus to academic CNN-design study.** Drop the ensemble-centric
narrative, foreground the three ablation questions (depth /
stride·padding·activation / regularisation).

## [0.3.1] — 2026-05-09 · `8eb24d1`
Load `kaggle.json` from Drive instead of the upload widget — cuts one
manual step from the setup flow.

## [0.3.0] — 2026-05-09 · `796cf81`
**4-rung complexity ladder restructure** (custom CNN → ResNet50 →
BiomedCLIP-equivalent → ensemble).

## [0.2.0] — 2026-05-09 · `41dbe69`
**Train all 3 architectures by default + multi-arch ensemble at the
end.** First version with a coherent end-to-end run sequence.

## [0.1.0] — 2026-05-08 · `1a3c581`
**Initial Colab support.** Cross-platform device detection
(Colab / CUDA / DirectML / CPU) and the first version of
`pneumonia_colab.ipynb` — clone, install, upload Kaggle creds, train,
evaluate. Birth of the notebook.

---

## Mapping summary

| Version | Date | Commit | One-line |
|---|---|---|---|
| 1.0.1 | 2026-05-11 | _this commit_ | Drop duplicate §14, renumber §15–§28 |
| 1.0.0 | 2026-05-11 | `35430f1` | Version stamp introduced |
| 0.13.0 | 2026-05-11 | `b804531` | A4 sweep + Future Work |
| 0.12.1 | 2026-05-11 | `457d263` | Claude → Gemini |
| 0.12.0 | 2026-05-11 | `33724bc` | Semantic bullet charts |
| 0.11.1 | 2026-05-11 | `3cb12b5` | Per-metric reference lines |
| 0.11.0 | 2026-05-11 | `49c293f` | Appendix B (clinical usability) |
| 0.10.0 | 2026-05-11 | `c640664` | Appendix A (theoretical ceiling) |
| 0.9.2  | 2026-05-11 | `a467641` | HF token from Colab Secrets |
| 0.9.1  | 2026-05-10 | `51d37a5` | Two optional Claude cells |
| 0.9.0  | 2026-05-10 | `ba2011e` | BiomedCLIP + methodology flow |
| 0.8.2  | 2026-05-10 | `dde1b9a` | Gemini text-assist cell |
| 0.8.1  | 2026-05-10 | `accec35` | Nano Banana renderer |
| 0.8.0  | 2026-05-10 | `948cd1c` | RAD-DINO added |
| 0.7.2  | 2026-05-10 | `ce5d980` | §13 clarification |
| 0.7.1  | 2026-05-10 | `47b1090` | Notebook polish |
| 0.7.0  | 2026-05-10 | `6610120` | Mini-report §16–§25 |
| 0.6.1  | 2026-05-10 | `3f6bb9b` | Strip report references |
| 0.6.0  | 2026-05-10 | `a2eaadf` | §16 synthesis + noise floor |
| 0.5.0  | 2026-05-10 | `5ee852c` | Transfer-learning standardised |
| 0.4.2  | 2026-05-10 | `57cc027` | `--num_workers 6` |
| 0.4.1  | 2026-05-10 | `ec2a0a7` | H100 time estimates |
| 0.4.0  | 2026-05-10 | `711132f` | Academic CNN-design refocus |
| 0.3.1  | 2026-05-09 | `8eb24d1` | Kaggle creds from Drive |
| 0.3.0  | 2026-05-09 | `796cf81` | 4-rung complexity ladder |
| 0.2.0  | 2026-05-09 | `41dbe69` | Multi-arch ensemble default |
| 0.1.0  | 2026-05-08 | `1a3c581` | First Colab notebook |
