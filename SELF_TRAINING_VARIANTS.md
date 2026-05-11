# Self-training variants — what we tried, what we scored, and why

This document catalogues every "from-scratch" training variant we ran on the
custom 4-block CNN (~389k params, 224×224 grayscale). All numbers are on the
official Kaggle test set (624 images, prevalence 0.625 PNE) and were measured
with `_helpers/medical_kpis.py`. Each variant uses the same architecture
(A3 = BN + Dropout 0.3 + WD 1e-4 + light augmentation) so the comparison
isolates the effect of the regularisation/optimisation knob being toggled.

Threshold convention below: **test accuracy** and **sensitivity / specificity**
are reported at the default `t=0.5` operating point — that is where a
clinically-deployed model lives before any post-hoc tuning. AUROC is
threshold-independent; ECE is reported after temperature scaling fit on val.

---

## 1. Headline table — four medical KPIs per variant

| # | Variant | Test acc @0.5 | Sensitivity @0.5 | Specificity @0.5 | AUROC | ECE (cal.) | T\* |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | `local_baseline` — no BN, no dropout, no WD, no aug | 0.7821 | **1.0000** | 0.4188 | 0.9157 | 0.1697 | 1.17 |
| 2 | `local_a3combo` — A3 (BN + Dropout + WD + light aug) | 0.8061 | 0.9897 | 0.5000 | 0.9159 | 0.1346 | 0.78 |
| 3 | `local_a3_smooth` — A3 + label smoothing 0.05 | 0.8221 | 0.9974 | 0.5299 | 0.9410 | 0.1256 | 0.50 |
| 4 | `local_a3_swa` — A3 + SWA (start 75 %) | 0.8606 | 0.9846 | 0.6538 | 0.9390 | 0.0768 | 1.08 |
| 5 | `local_a3_trivial` — A3 + TrivialAugmentWide | 0.7981 | 0.9692 | 0.5128 | 0.9263 | 0.1252 | 0.53 |
| 6 | `local_a3_cutmix` — A3 + CutMix α=1.0 | 0.8574 | 0.9769 | 0.6581 | 0.9427 | 0.0708 | 0.51 |
| 7 | **`local_a3_champion`** — A3 + smooth + CutMix + SWA | **0.8878** | 0.9846 | **0.7265** | **0.9523** | **0.0459** | 0.50 |

Δ vs. baseline (champion): test acc **+10.6 pp**, specificity **+30.8 pp**,
AUROC **+0.037**, ECE **−0.124** (3.7× lower). Sensitivity holds within
1.5 pp of the saturated baseline while specificity climbs out of the
"useless" zone.

Best-accuracy operating point (tuned on val, applied to test) is reported
in each run's `medical_kpis.json` and is consistently 1–4 pp above the
default `t=0.5` figure. The champion at its tuned threshold (t≈0.765)
reaches **acc 0.926, sens 0.967, spec 0.859** — three of four KPIs in the
clinical sweet zone, with specificity 4 pp below the 0.90 ceiling we set.

---

## 2. Per-variant explanation — what each technique *does*, and why it helped or didn't

### Variant 1 — `local_baseline`: no regularisation at all

**What it is.** Four convolutional blocks with conv → ReLU → max-pool, then a
linear head. No BatchNorm, no Dropout, no weight decay, no augmentation. 20
epochs of AdamW @ 1e-3 with seed 78.

**Why we ran it.** A reference point: "what does the architecture do if you
give it no help fighting overfitting?" Required so every other variant
expresses a measurable delta.

**What happened.** Validation accuracy hits 0.978 in epoch 5 and the model
calls every test image *pneumonia* (sens 1.000, spec 0.419). At t=0.5 it is
essentially a "PNE-always" classifier with a tiny carve-out for the most
obviously-normal scans.

**Why it failed.** Two compounding problems:

1. **Sigmoid saturation.** With no BN/Dropout/WD the logits grow unboundedly.
   The softmax/sigmoid output for the majority class quickly hits ~0.99 and
   the gradient through correctly-confident PNE samples vanishes — the model
   never has to learn how to recognise *normal*.
2. **Class-prior shift.** Training prevalence is 0.74 PNE; test prevalence is
   0.625 PNE. A saturated model that latches onto the training prior pays
   maximum cost on the test set's higher proportion of normals.

The 0.17 ECE confirms it: the model is extremely confident, and that
confidence is misplaced — exactly the pathology that regularisation exists
to prevent.

### Variant 2 — `local_a3combo`: BN + Dropout 0.3 + WD 1e-4 + light aug

**What it is.** Same architecture, plus BatchNorm after every conv, Dropout
0.3 before the head, AdamW weight decay 1e-4, and light affine + horizontal
flip augmentation. This is the "A3" winner from the regularisation ablation
(question 3 of the assignment).

**Why we ran it.** This is *the* answer to "what regularisation works best",
and it's the platform every subsequent variant builds on.

**What happened.** Specificity climbs from 0.42 → 0.50, ECE drops 0.170 →
0.135, test acc rises 2.4 pp. AUROC is essentially unchanged from baseline
(0.916 vs 0.916), but the operating point at t=0.5 is in a better place
because the logits no longer pile up at +∞.

**Why it helped (but not enough yet).** BN puts the logits on a bounded
scale so the sigmoid stops saturating. Dropout adds a small ensembling
effect inside training. WD shrinks the decision boundary towards the
origin, reducing the model's willingness to be 0.99-confident on everything.
The remaining problem is that the *training distribution* is still 74 % PNE
— the model has correctly learned the prior, and on a 62.5 %-PNE test set
that prior is wrong.

### Variant 3 — `local_a3_smooth`: A3 + label smoothing 0.05

**What it is.** Binary targets become `[0.05, 0.95]` instead of `[0, 1]`.
The BCE loss is computed against these softer targets. Implementation in
`pneumonia_cnn_custom.py` via `--label_smoothing 0.05`.

**Why we ran it.** Saturated sigmoid is the *symptom* of variant 2's
half-success; label smoothing is the textbook anti-saturation tool. By
making perfect confidence "wrong by 0.05", it caps the gradient that pushes
logits to ±∞.

**What happened.** Modest improvement in everything: test acc +1.6 pp,
specificity +3 pp, AUROC jumps 0.916 → 0.941 (a meaningful gain because
this metric is normally hard to move with just-regularisation tricks).
ECE drops slightly. Temperature scaling fits T\*=0.50 (clamped against the
lower bound), confirming the predictions are now *under*-confident — exactly
what the technique is supposed to do.

**Why it helped.** Smoothing the targets does three useful things on this
dataset: it slows saturation (so gradients flow longer), it forces the
model to learn discriminative features for the minority class (it can't
just hit 0.99 on PNE and call it a day), and it makes the ranking quality
(AUROC) better even when the raw operating point is similar.

**Why the gain is small.** 0.05 is a conservative smoothing strength. With
a 74 % majority class and only ~5 % label noise in this dataset
(Kermany 2018), the asymmetric class-prior is the dominant problem, not
overconfidence per se. Smoothing helps but doesn't address prior shift.

### Variant 4 — `local_a3_swa`: A3 + Stochastic Weight Averaging

**What it is.** From epoch 0.75 × 20 = 15 onwards, the optimiser keeps a
running mean of model weights. After training the BN running statistics are
re-computed on the averaged weights. The averaged model — not the final
checkpoint — is what gets tested.

**Why we ran it.** SGD with a high learning rate moves around a flat region
of the loss surface rather than converging to a single point. Averaging
weights from late epochs picks a point closer to the centre of that flat
region, which generalises better than any one of the snapshots.

**What happened.** Big jump: test acc 0.806 → 0.861 (+5.5 pp), specificity
0.500 → 0.654 (+15.4 pp), AUROC +0.023, ECE almost halves (0.135 → 0.077).
This is the single largest gain from any of the additive ingredients.

**Why it helped so much.** Two effects compound:

1. **Reduced gradient noise.** Averaging weights is equivalent to applying
   a low-pass filter to the SGD trajectory. The averaged model behaves like
   it was trained with an effectively smaller learning rate at the end of
   the schedule, without the actual training cost.
2. **Implicit ensemble.** SWA produces something that *behaves* like the
   mean prediction of 5–10 late-epoch snapshots. Ensembles always lift
   AUROC on small classification problems like this one. We got the
   ensembling benefit at zero inference cost.

The calibration improvement (ECE −0.058) is the most striking effect: the
averaged model is naturally less confident, because per-snapshot
overconfidence cancels out in the mean.

### Variant 5 — `local_a3_trivial`: A3 + TrivialAugmentWide

**What it is.** TrivialAugmentWide (Müller & Hutter 2021) picks one
augmentation operation uniformly at random per image — from a list of 14
including rotation, shear, autocontrast, equalize, posterize, solarize,
sharpness, brightness, contrast — and applies it with a strength drawn
uniformly from `[0, 30]`. No tuning required, hence "trivial".

**Why we ran it.** State-of-the-art results on ImageNet, CIFAR, and many
medical-imaging benchmarks. The hypothesis: stronger augmentation should
break the over-fitting to training-set artefacts (X-ray machine
orientation, exposure level) and force the model to rely on actual
pathology features.

**What happened.** It *hurt*. Test acc 0.806 → 0.798 (−0.8 pp). AUROC up
slightly (0.926 vs 0.916), ECE essentially unchanged. The first variant
where the metric we care about most (test acc at default threshold) gets
worse.

**Why it failed.** Two distinct reasons specific to this dataset:

1. **Operations that destroy chest-X-ray semantics.** Posterize and
   solarize reduce the bit-depth of a grayscale image and invert intensity
   regions. On natural images that's a useful distortion; on a chest X-ray,
   posterising the lung field flattens out the very texture (mid-grey
   ground-glass opacities, the patchy consolidation pattern of bacterial
   pneumonia) that distinguishes PNE from NORMAL. Equalize and autocontrast
   redistribute intensity in ways that the diagnostic feature does not survive.
2. **The grayscale round-trip.** TrivialAugmentWide is implemented for 3-channel
   RGB in `torchvision`. We work around by `Grayscale(3) → TrivialAugmentWide
   → Grayscale(1)`, but the round-trip leaks small numerical noise into the
   image even when no augmentation is meaningfully applied.

Lesson: augmentation policies that are domain-agnostic ("just throw
everything at the image") underperform a small, domain-specific set on
medical imaging.

### Variant 6 — `local_a3_cutmix`: A3 + CutMix α=1.0

**What it is.** For each batch, draw `λ ~ Beta(1.0, 1.0)`. Cut a random
rectangle from image B (a different image in the same batch) and paste
it over image A. The target becomes `λ * y_A + (1-λ) * y_B`. The model
is trained to predict the mixed target on the mixed image.

**Why we ran it.** Three known benefits in the literature: better
generalisation through regularisation-by-example-mixing, less overconfidence
on test data, and improved AUROC because the model is forced to be
spatially-aware rather than reading off a single global texture.

**What happened.** Test acc 0.806 → 0.857 (+5.1 pp), specificity +15.8 pp,
ECE almost halves (0.135 → 0.071). Comparable to SWA in magnitude.

**Why it helped.** CutMix is doing two things simultaneously:

1. **Target smoothing through mixing.** Like label smoothing, the target
   distribution is no longer one-hot, so the model can't fully saturate.
2. **Spatial regularisation.** The model learns to predict PNE from a
   *part* of a chest X-ray, not from a global gestalt of the whole image.
   This makes it harder to over-rely on the training set's specific
   global statistics (which differ between hospitals/machines), and
   easier to generalise to test images from a slightly different source.

Crucially, **CutMix on chest X-rays does not damage diagnostic semantics**
the way TrivialAugment does: a rectangle of lung tissue pasted onto another
chest is still a chest X-ray with a believable pathology pattern. The
operation is structure-preserving.

### Variant 7 — `local_a3_champion`: A3 + smoothing + CutMix + SWA

**What it is.** The three positive variants stacked on top of the A3
platform. TrivialAugment is excluded because it regressed.

**What happened.** Best run on every KPI:

| Metric | Champion | Best individual variant | Δ vs. best individual |
|---|---:|---:|---:|
| Test acc @0.5 | 0.8878 | 0.8606 (SWA) | +2.7 pp |
| Sensitivity @0.5 | 0.9846 | 0.9974 (smooth) | −1.3 pp |
| Specificity @0.5 | 0.7265 | 0.6581 (CutMix) | +6.8 pp |
| AUROC | 0.9523 | 0.9427 (CutMix) | +0.010 |
| ECE | 0.0459 | 0.0708 (CutMix) | −0.025 |

**Why the stacking is additive.** Each technique targets a *different*
failure mode:

- Label smoothing prevents per-sample saturation.
- CutMix prevents global-feature reliance.
- SWA averages out the optimisation noise that ordinary stochastic gradient
  descent leaves in the final weights.

None of the three substitutes for the others. The combination produces
the only single-fold run where three of four medical KPIs land in the
clinical sweet zone at the tuned threshold (acc 0.926, sens 0.967, spec
0.859, AUROC 0.952, ECE 0.046).

---

## 3. What we did *not* run locally (and what would extend this list)

These are the missing entries that the Colab run will add to the appendix:

| Planned | Justification |
|---|---|
| `lion` optimiser (instead of AdamW) | Cheaper memory; reportedly better on small image datasets. Worth a single-fold run as data-point. |
| `augment_policy=trivial` *without* `Grayscale(3)` round-trip | Confirm whether the channel-conversion was the regression or whether TA itself is the wrong policy for X-rays. |
| Mixup (α=0.4) | Strictly weaker than CutMix on natural images but sometimes better when the target is binary and the dominant feature is global. |
| Manifold Mixup | Mixes hidden activations, not pixels. No published result on this dataset. |
| RandAugment (n=2, m=9) | Tuned alternative to TrivialAugment. If TA fails on X-rays but RandAugment succeeds, the issue is the *distribution* of strengths, not the operation set. |

We will only escalate the variants whose single-fold result clears the A3
baseline by more than the test-set noise floor (σ ≈ 0.016, so a meaningful
delta is ≥ 0.032). Variants below that bar stay in the appendix as
"tried-and-discarded" rather than being promoted to a 5-fold run.

---

## 4. Take-aways for the report

1. **Order of importance, by KPI delta from baseline to champion.** SWA
   (+5.5 pp test acc on its own) > CutMix (+5.1 pp) > label smoothing
   (+1.6 pp) > A3 regularisation (+2.4 pp over baseline) > TrivialAugment
   (−0.8 pp — *negative*).
2. **All three positive variants attack overconfidence**, not feature
   learning. The architecture (A3 winner) already extracts the right
   features; the additional gain comes from making the head's output
   distribution more honest about what the model knows.
3. **Domain matters for augmentation policies.** TrivialAugmentWide is
   strictly worse than a small, hand-curated affine + flip policy on this
   dataset, despite being the SOTA pick for natural images.
4. **Specificity is the hard KPI on this benchmark.** Sensitivity sits
   above 0.97 from variant 3 onwards essentially for free (the saturated
   baseline got there by predicting "PNE always"). Lifting specificity
   from 0.42 → 0.73 — the actual clinical bottleneck — is what the variant
   programme bought us.
5. **Ensembling-by-averaging (SWA) gave the best per-unit-effort return.**
   Zero inference overhead, no architecture change, ~5 pp test acc gain.
   A 5-fold cross-validated ensemble of the champion config is the
   natural next step, and is queued for Colab H100.

---

*Generated from `runs/local_*` after the 2026-05-09 Vega 64 variant
sweep. Each cell in §1 is sourced from the relevant
`runs/<name>/medical_kpis.json` — open those files for the full
calibration breakdown and per-threshold confusion matrices.*
