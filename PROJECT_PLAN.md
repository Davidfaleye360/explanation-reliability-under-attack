# Project Plan: AI Explanation Reliability Under Attack

## 1. Project Overview

**Goal:** Determine whether Grad-CAM explanations for an image classifier remain visually convincing ("deceptive") or visibly degrade when an adversarial attack successfully causes a misclassification.

**Primary research question:** How does adversarial attack strength affect the visual similarity between an image's Grad-CAM explanation before and after an attack, when the attack successfully changes the classifier's prediction?

**Hypotheses:**

- H1: Increasing attack strength (epsilon) will generally decrease Grad-CAM similarity between the original and adversarial explanations.
- H2: PGD (iterative attack) will produce different explanation-preservation behavior than FGSM (single-step attack) at comparable epsilon values.
- H3: A subset of adversarial examples will maintain high Grad-CAM similarity despite producing an incorrect prediction — a "deceptive explanation," demonstrating a potential failure mode of explanation-based safeguards. This is the central finding of the project.

**Dataset:** CIFAR-10 (60,000 32x32 color images, 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck; 50,000 train / 10,000 test).

**Model:** ResNet-18 (adapted for 32x32 input), trained from scratch, Adam optimizer, standard augmentation (random crop, horizontal flip). Target accuracy: 85–93%.

**Core libraries:**

- `torch`, `torchvision` — model + data
- `torchattacks` — FGSM and PGD attack implementations
- `pytorch-grad-cam` — Grad-CAM heatmap generation
- `scikit-image` — SSIM computation
- `numpy` / `scipy` — IoU, Pearson/Spearman correlation
- `matplotlib` — charts and visual galleries

**Attack configurations (6 total):** {FGSM, PGD} × {epsilon = 0.01, 0.03, 0.05}, normalized 0–1 pixel scale.

**Similarity metrics:**

- SSIM — PRIMARY metric. Used to define and threshold "deceptive" explanations. A single, holistic structural similarity score that does not require an extra thresholding step.
- IoU (top salient regions) — SECONDARY / supporting metric only. Binarization rule: for each heatmap, keep the top 20% of pixels by intensity (i.e., threshold at the 80th percentile of that heatmap's own pixel values) to define its "salient region"; IoU is then computed between the two binary masks. Use the same fixed 20% rule for every heatmap — do not tune this per-image.
- Pearson or Spearman correlation (full heatmaps) — SECONDARY / supporting metric only.
- IoU and correlation should never be used to independently decide "deceptive" status — only to corroborate the SSIM-based classification.

**Deceptive-explanation threshold (must be fixed BEFORE evaluating any adversarial examples):**

1. Build a held-out calibration set of clean, correctly classified images.
2. Pair each with a lightly perturbed version of itself using small random (non-adversarial) noise at the same magnitude as the smallest epsilon (0.01).
3. Compute SSIM between each pair's Grad-CAM heatmaps → this is the "noise-floor" distribution.
4. Fix the deceptive-explanation threshold at the 25th percentile of this distribution (i.e., an adversarial SSIM at or above this value is "as similar as clean image pairs typically are, or more similar" — the bottom 25% of the noise-floor distribution is excluded to avoid flagging borderline/noisy cases as deceptive). This specific percentile is a project decision made now, before any attack data exists, precisely so it cannot be tuned to fit later results.
5. Do not adjust this threshold after seeing adversarial results.

**Success criteria:**

- A working, reproducible end-to-end pipeline.
- Sample size target: at least 500 misclassified test images evaluated per attack configuration (out of the 10,000-image test set), or all successfully-attacked images if fewer than 500 exist at a given configuration/epsilon.
- Attack success rate and adversarial accuracy reported per configuration.
- Deceptive-explanation rate reported per configuration.
- A clear quantitative comparison across attack types/strengths supporting or refuting H1–H3 (or a clearly documented absence of a trend — that is still a valid result).
- A curated gallery of visual before/after examples, including highlighted deceptive cases.

**Scope discipline:** Do not add additional datasets, models, attack types, or metrics beyond what is listed above. The 6-configuration × 3-metric design is intentionally sized for a single semester.

## 2. Repository Structure

```
explanation-reliability-under-attack/
├── README.md
├── PROJECT_PLAN.md              (this file)
├── requirements.txt
├── data/
│   └── (CIFAR-10 downloaded here at runtime, not committed)
├── src/
│   ├── model.py                 ResNet-18 definition + training loop
│   ├── train.py                 script: trains + saves checkpoint
│   ├── gradcam_utils.py          Grad-CAM generation wrapper
│   ├── attacks.py               FGSM/PGD generation wrapper (torchattacks)
│   ├── calibration.py           builds clean-noise calibration set, computes threshold
│   ├── metrics.py               SSIM, IoU, Pearson/Spearman implementations
│   ├── pipeline.py              end-to-end: attack → explain → compare → log results
│   └── analyze.py               aggregates results, produces charts/tables
├── notebooks/
│   └── exploration.ipynb         ad hoc exploration, sanity checks
├── results/
│   ├── checkpoints/              saved model weights
│   ├── calibration/               calibration distribution + threshold value
│   ├── tables/                    CSV/JSON result tables per configuration
│   ├── charts/                    similarity-vs-epsilon plots, deceptive-rate plots
│   └── gallery/                   curated before/after image examples
└── reports/
    ├── report1.pdf
    ├── midterm_report.pdf
    ├── report3.pdf
    ├── report4.pdf
    ├── report5.pdf
    └── final_report.pdf
```

## 3. Checkpoint Breakdown

### Checkpoint 1 — Report 1: Foundation
**Goal:** Repo set up, data pipeline working, classifier training loop running.

Steps:
1. Initialize GitHub repo with the structure above; commit `README.md` and `PROJECT_PLAN.md`.
2. Write `src/model.py`: ResNet-18 adapted for 32x32 CIFAR-10 input (adjust first conv layer / remove initial maxpool, standard adaptation for CIFAR-sized ResNets).
3. Write `src/train.py`: CIFAR-10 loading via `torchvision.datasets.CIFAR10`, augmentation (`RandomCrop(32, padding=4)`, `RandomHorizontalFlip`), Adam optimizer, training loop with per-epoch train/test accuracy logging.
4. Run a short training job (even a few epochs) to confirm the pipeline works end-to-end (data loads, loss decreases, checkpoint saves).

**Done when:** Repo is live on GitHub with initial commits; `train.py` runs without errors and produces a loss curve; a checkpoint file is saved.
**Evidence for report:** GitHub repo link, first commits, loss curve plot, early accuracy number.

### Checkpoint 2 — Midterm Report: Working Classifier + Explanations
**Goal:** Classifier reaches target accuracy; Grad-CAM integrated and sanity-checked.

Steps:
1. Complete full training run; tune epochs/learning rate as needed to reach 85–93% test accuracy.
2. Write `src/gradcam_utils.py` using `pytorch-grad-cam`, targeting the final convolutional layer of the ResNet-18.
3. Generate Grad-CAM heatmaps for a handful of correctly classified test images per class; visually confirm heatmaps focus on plausible regions (e.g., the animal/object, not background).
4. Save example heatmap images to `results/gallery/`.

**Done when:** Final test accuracy is recorded and within target range; at least 10 sample Grad-CAM heatmaps look sensible on visual inspection.
**Evidence for report:** Final accuracy number, training curve, sample heatmap images (original photo + heatmap overlay, several classes).

### Checkpoint 3 — Report 3: Attacks + Calibration Baseline
**Goal:** Adversarial attacks working; deceptive-explanation threshold locked in.

Steps:
1. Write `src/attacks.py` wrapping `torchattacks.FGSM` and `torchattacks.PGD` at epsilon = {0.01, 0.03, 0.05}.
2. Run each of the 6 attack configurations on the test set; record attack success rate (% of initially-correct images that become misclassified) and adversarial accuracy per configuration.
3. Write `src/calibration.py`: build the clean-noise calibration set (small random non-adversarial noise at epsilon = 0.01 magnitude), generate Grad-CAM heatmaps for clean/perturbed pairs, compute SSIM for each pair.
4. Choose and fix the deceptive-explanation threshold as a percentile of this SSIM distribution. Document the exact percentile and resulting threshold value — do not revisit this later.

**Done when:** A results table exists with attack success rate + adversarial accuracy for all 6 configurations; the calibration SSIM distribution is plotted; the threshold value is fixed and recorded in `results/calibration/`.
**Evidence for report:** Attack success rate table (6 rows), calibration distribution histogram, stated threshold value and the percentile rule used to derive it.

### Checkpoint 4 — Report 4: Full Comparison Pipeline
**Goal:** End-to-end pipeline runs on a meaningful subset: attack → explain → compare → classify as deceptive or not.

Steps:
1. Write `src/metrics.py`: SSIM (primary), IoU-of-top-salient-region, Pearson/Spearman correlation.
2. Write `src/pipeline.py`: for each of the 6 configurations, for every misclassified image, generate the post-attack Grad-CAM heatmap and compute all three metrics against the pre-attack heatmap.
3. Apply the fixed SSIM threshold from Checkpoint 3 to flag each case as "deceptive" or not.
4. Run this on a meaningful subset (e.g., 500–1000 test images) to validate the pipeline works correctly before scaling to the full test set.
5. Save a handful of side-by-side visual examples (clean image + heatmap, attacked image + heatmap) to `results/gallery/`, including at least one clear deceptive example if found.

**Done when:** Pipeline runs without errors on the subset; a partial results table (per configuration: n images, SSIM/IoU/correlation stats, deceptive count) exists; at least a few side-by-side visual examples are saved.
**Evidence for report:** Partial results table, 3–5 side-by-side before/after visual examples, code snippet or link to `pipeline.py`.

### Checkpoint 5 — Report 5: Full-Scale Results + Analysis
**Goal:** Full test set processed for all 6 configurations; hypotheses evaluated against real data.

Steps:
1. Run `pipeline.py` across the full test set (or at minimum 500 misclassified images per configuration, per the sample size target in Section 1) for all 6 configurations.
2. Write `src/analyze.py`: aggregate per-configuration stats (mean/median SSIM, IoU, correlation; deceptive-explanation rate) into `results/tables/`.
3. Produce charts: SSIM (and deceptive rate) vs. epsilon, split by attack type (FGSM vs. PGD).
4. Write a first-draft interpretation: does the data support or refute H1, H2, H3?

**Done when:** Full result tables and charts exist for all 6 configurations; a written first-draft interpretation of H1–H3 exists.
**Evidence for report:** Full results table (6 rows, all metrics), charts (similarity vs. epsilon, deceptive rate vs. epsilon/attack type), draft interpretation paragraph.

### Checkpoint 6 — Final Report: Polish + Conclusions
**Goal:** Finalize analysis, polish repository and deliverables, prepare final write-up and video.

Steps:
1. Finalize written conclusions on H1, H2, and H3 (supported / refuted / mixed), explicitly scoped to the CIFAR-10 + ResNet-18 controlled setting (not a universal claim about Grad-CAM).
2. Curate the strongest gallery examples (clear deceptive cases, clear non-deceptive/degraded cases) for the final report and demo video.
3. Clean up the repository: consistent naming, a complete `README.md` with setup + run instructions, remove dead code/notebooks.
4. Record the final demo video showing: the trained classifier, a live or pre-recorded example of an attack fooling the model, the before/after Grad-CAM comparison, and the summary charts.

**Done when:** Repo is clean and fully reproducible from `README.md` instructions; final report and video are complete.
**Evidence for report:** Final charts/tables, full gallery, complete written analysis, polished GitHub repo link, video.
