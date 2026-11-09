# Explanation Reliability Under Attack

When an adversarial attack makes an image classifier give a wrong answer, does the model's Grad-CAM explanation change visibly, or does it stay almost the same as the explanation for the clean image? This project measures that on CIFAR-10 with a ResNet-18, using FGSM and PGD attacks at three strengths.

**What is and is not measured.** The outcome is *explanation similarity*: how similar the Grad-CAM heatmap of an attacked image is to the heatmap of its clean original, measured with SSIM. An explanation is called "deceptive" only in this operational sense (similarity at or above a pre-fixed threshold). The project does **not** measure whether a human finds an explanation plausible, nor whether Grad-CAM faithfully reflects the model's reasoning. There is no user study, so conclusions about human trust are out of scope.

The full plan, hypotheses and checkpoint schedule are in [PROJECT_PLAN.md](PROJECT_PLAN.md).

## Status

| Checkpoint | Status |
|---|---|
| 1. Pipeline + short training test | Done |
| 2. Full training (91.03% test accuracy) + Grad-CAM sanity gallery | Done |
| 3. Attack evaluation + locked deceptive-explanation threshold | Done |
| 4. Attack, explain, compare pipeline on a 1,000-image subset | Done |
| 5. Full-scale results, analysis and first-draft interpretation | Done |
| 6. Final polish, conclusions, video | Not started |

## Setup

Tested on macOS (Apple M2, GPU via PyTorch `mps`) with Python 3.13.1. Do not keep the project or its virtual environment inside an iCloud-synced folder such as Desktop or Documents; sync contention made `import torch` hang for hours during development.

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt   # exact pinned versions
```

CIFAR-10 is downloaded automatically (from the University of Toronto) into `data/` the first time any script needs it.

## Running the pipeline

Run in this order. The calibration step must come before the attacks are evaluated, because it fixes the threshold before any attack result exists.

```bash
python -m src.train --epochs 30 --device mps        # train, writes results/checkpoints/ and results/logs/
python -m src.gradcam_utils --device mps            # Grad-CAM sanity gallery -> results/gallery/
python -m src.calibration --device mps              # noise-floor SSIM + locked threshold -> results/calibration/
python -m src.attacks --device mps                  # six attack configurations -> results/tables/attack_results.csv
python -m src.pipeline --subset-size 9000 --device mps   # attack, explain, compare (use 1000 for a quick subset) -> results/tables/, results/gallery/
python -m src.analyze --n 9000                      # tables with intervals, hypothesis tests, charts -> results/tables/, results/charts/
```

`--device` can be `cuda`, `mps` or `cpu`. The checkpoint (`results/checkpoints/resnet18_cifar10.pt`, ~45 MB) is not committed; train it with the first command.

## Experimental parameters

| Item | Setting |
|---|---|
| Dataset | CIFAR-10, 50,000 train / 10,000 test, 32x32 RGB |
| Model | torchvision ResNet-18, 3x3 stride-1 stem conv, no initial max-pool, trained from scratch |
| Training | Adam, lr 1e-3 (no schedule), batch size 128, 30 epochs, RandomCrop(32, padding 4) + horizontal flip, channel mean/std normalization |
| Checkpoint used | Epoch 28 (best test accuracy, 0.9103). Logs: `results/logs/` |
| Test-set split | Seeded random split (seed 0) of the 10,000 test images: 1,000 calibration pool / 9,000 evaluation. Indices in `results/calibration/split_indices.json` |
| Pixel scale | Attacks and noise operate on [0, 1] pixels; normalization happens inside the model |
| FGSM | Untargeted, single step `eps * sign(grad)` of the cross-entropy loss, clipped to [0, 1] |
| PGD | Untargeted, L-infinity, 10 steps, step size `2.5 * eps / 10`, random start in the eps-ball, cross-entropy loss, projected to the eps-ball and [0, 1] |
| Epsilons | 0.01, 0.03, 0.05 (so six attack configurations) |
| Attack run | True labels, model in eval mode, batch size 250, `torch.manual_seed(0)` before each configuration. A check aborts the run if any perturbation exceeds its epsilon |
| Grad-CAM | `pytorch-grad-cam` 1.5.7, target layer `layer4[-1]` (native resolution 4x4, 512 channels). Channel weights are the spatial mean of the target-class gradients, activations are weighted and summed, then ReLU. Each heatmap is min-max normalized to [0, 1] and bilinearly resized to 32x32 |
| Explained class | The model's own top-1 prediction for the image being explained |
| SSIM | `skimage.metrics.structural_similarity`, `data_range=1.0`, default 7x7 uniform window, on the 32x32 heatmaps (primary metric) |
| Noise-floor calibration | 905 correctly classified calibration images (of 1,000), each paired with a copy plus random +/-0.01 sign noise per pixel channel (same per-pixel size as an FGSM step at eps 0.01, random direction), clipped to [0, 1]. Noise seed 0 |
| Pipeline subset | The first 1,000 evaluation images (never a calibration image). Only images classified correctly before the attack and misclassified after it (successful attacks) are compared |
| Heatmaps compared | Clean image: heatmap for its (correct) top-1 class. Attacked image: heatmap for the model's top-1 prediction on the attacked image, i.e. the wrong class the model now reports. The same rule was used in calibration |
| IoU (supporting) | Top 20% of pixels of each heatmap: pixels at or above the 80th percentile of that heatmap's own values (same fixed rule for every heatmap). If that percentile is 0, only pixels with positive intensity are kept. IoU = intersection / union of the two masks; NaN if both masks are empty |
| Correlation (supporting) | Pearson correlation between the two full flattened 32x32 heatmaps; NaN if either heatmap is constant. NaN cases are counted in the summary table, not hidden |
| Role of the metrics | Only SSIM against the locked threshold decides "deceptive". IoU and correlation only corroborate it |
| Deceptive threshold | 25th percentile of the noise-floor SSIM distribution = **0.9802**. An attacked example's explanation counts as deceptive if its SSIM to the clean heatmap is >= 0.9802. Fixed before any attack was run; the script will not overwrite `results/calibration/threshold.json` |

## Results so far

**Classifier.** Best test accuracy 91.03% (epoch 28 of 30); see `results/charts/training_curve.png` and `results/logs/`. Grad-CAM examples for all 10 classes: `results/gallery/gradcam_samples.png`.

**Noise floor** (`results/calibration/`). Across the 905 clean/noisy pairs, SSIM between heatmaps has mean 0.969, median 0.994 and minimum -0.176. Random noise changed the predicted class for 13 of the 905 images (those pairs were kept). The heavy concentration near 1.0 is why the threshold is high (0.9802).

**Attacks** on the 9,000-image evaluation split (8,198 images were classified correctly before attack; clean accuracy 91.09%). Success rate is the share of those 8,198 that become misclassified.

| Attack | Epsilon | Success rate | Adversarial accuracy | Successfully attacked images |
|---|---|---|---|---|
| FGSM | 0.01 | 77.6% | 20.4% | 6,362 |
| FGSM | 0.03 | 91.2% | 8.0% | 7,479 |
| FGSM | 0.05 | 91.9% | 7.5% | 7,531 |
| PGD | 0.01 | 97.7% | 2.1% | 8,011 |
| PGD | 0.03 | 100% | 0.0% | 8,198 |
| PGD | 0.05 | 100% | 0.0% | 8,198 |

Every configuration has far more than the 500 successfully attacked images the plan requires.

**Explanation similarity at full scale** (all 9,000 evaluation images; per-image values in `results/tables/pipeline_n9000_per_image.csv`, statistics in `results/tables/analysis_n9000_*.csv`, charts in `results/charts/`). For each successfully attacked image, the clean heatmap is compared with the heatmap of the attacked image. 8,198 images were classified correctly before attack; 45,779 attacked image-configuration pairs were compared across the six configurations. Brackets are 95% bootstrap intervals (10,000 resamples). "Deceptive" means SSIM >= 0.9802 (the locked threshold), with exact 95% intervals on the rate.

| Attack | Epsilon | Attacked images | SSIM mean [95% CI] | SSIM median | IoU mean | Pearson mean | Deceptive (rate; exact 95% interval) |
|---|---|---|---|---|---|---|---|
| FGSM | 0.01 | 6,362 | 0.512 [0.505, 0.518] | 0.568 | 0.383 | 0.542 | 0 (0.000%; 0.000% to 0.058%) |
| FGSM | 0.03 | 7,479 | 0.493 [0.486, 0.499] | 0.546 | 0.372 | 0.522 | 0 (0.000%; 0.000% to 0.049%) |
| FGSM | 0.05 | 7,531 | 0.440 [0.433, 0.446] | 0.488 | 0.342 | 0.463 | 1 (0.013%; 0.000% to 0.074%) |
| PGD | 0.01 | 8,011 | 0.628 [0.623, 0.633] | 0.682 | 0.444 | 0.655 | 5 (0.062%; 0.020% to 0.146%) |
| PGD | 0.03 | 8,198 | 0.655 [0.651, 0.660] | 0.700 | 0.461 | 0.686 | 3 (0.037%; 0.008% to 0.107%) |
| PGD | 0.05 | 8,198 | 0.648 [0.643, 0.652] | 0.690 | 0.454 | 0.680 | 4 (0.049%; 0.013% to 0.125%) |

Paired tests use the same images across conditions (images successfully attacked in both). The difference is in SSIM; p-values are Wilcoxon signed-rank, Holm-adjusted within each hypothesis. With thousands of images almost any difference is "significant", so the intervals matter more than the p-values.

| Hypothesis | Comparison | Paired images | Mean difference [95% CI] | Holm p |
|---|---|---|---|---|
| H1 strength | FGSM: SSIM at eps 0.03 minus eps 0.01 | 6,287 | -0.0351 [-0.0406, -0.0297] | < 0.001 |
| H1 strength | FGSM: SSIM at eps 0.05 minus eps 0.03 | 7,292 | -0.0499 [-0.0554, -0.0444] | < 0.001 |
| H1 strength | FGSM: SSIM at eps 0.05 minus eps 0.01 | 6,167 | -0.0824 [-0.0899, -0.0750] | < 0.001 |
| H1 strength | PGD: SSIM at eps 0.03 minus eps 0.01 | 8,011 | +0.0246 [+0.0215, +0.0277] | < 0.001 |
| H1 strength | PGD: SSIM at eps 0.05 minus eps 0.03 | 8,198 | -0.0073 [-0.0096, -0.0051] | < 0.001 |
| H1 strength | PGD: SSIM at eps 0.05 minus eps 0.01 | 8,011 | +0.0173 [+0.0136, +0.0211] | < 0.001 |
| H2 attack type | eps 0.01: SSIM under PGD minus FGSM | 6,362 | +0.1041 [+0.0999, +0.1083] | < 0.001 |
| H2 attack type | eps 0.03: SSIM under PGD minus FGSM | 7,479 | +0.1541 [+0.1484, +0.1597] | < 0.001 |
| H2 attack type | eps 0.05: SSIM under PGD minus FGSM | 7,531 | +0.2015 [+0.1950, +0.2078] | < 0.001 |

What the data show (this controlled setting only):

- **H1 (similarity falls as strength rises): supported for FGSM, not for PGD.** FGSM mean SSIM falls at every step (0.512, 0.493, 0.440). PGD does not fall: it rises from epsilon 0.01 to 0.03, dips slightly from 0.03 to 0.05, and ends higher than it started.
- **H2 (PGD and FGSM differ): supported.** At every epsilon PGD explanations stay more similar to the clean ones than FGSM explanations, and the gap widens with strength (+0.104, +0.154, +0.202).
- **H3 (some attacks keep a highly similar explanation): supported, but the effect is very rare.** 13 of 45,779 attacked pairs (0.028%), from 10 distinct images, reach the threshold; all but one are PGD. Their IoU is at least 0.78 and Pearson correlation at least 0.987, so the supporting metrics agree. Per-case details: `results/tables/analysis_n9000_deceptive_cases.csv`; examples: `results/gallery/attack_examples_n9000.png`.
- The attacked explanations are far less similar to the clean ones (median SSIM 0.49 to 0.70) than explanations under random noise of the same per-pixel size (median 0.994).
- IoU and correlation order the six configurations the same way SSIM does.
- **Exploratory, found after seeing the results (not a pre-set test):** truck<->automobile confusions are 12.8% of all successful attacks but 11 of the 13 deceptive cases. The deceptive rate is 0.188% within those confusions versus 0.005% for all other attacks (intervals do not overlap). A plausible reading is that explanations for two visually similar classes naturally highlight the same object region, so a similar heatmap may say little about the attack itself. This rests on 13 cases and was not tested in advance.

## Limitations

- Heatmap similarity is not human-perceived plausibility and not explanation faithfulness. "Deceptive" is defined only by the SSIM threshold above.
- Grad-CAM at `layer4[-1]` is natively 4x4, upsampled to 32x32, so heatmaps are coarse and smooth. This affects how sensitive SSIM is.
- One dataset, one model, two attacks, three strengths. Conclusions apply to this controlled setting only.
- The deployed checkpoint came from an unseeded training run (the `--seed` option was added afterwards), so retraining will give slightly different numbers.
- The checkpoint epoch was selected by test accuracy on the same test set that supplies the calibration and evaluation images, so the reported test accuracy is slightly optimistic.
- The threshold depends on the chosen noise model (random sign noise at 0.01).
- Each configuration was run with a single attack seed (PGD uses a random start), so run-to-run variation from the seed is not measured. The intervals above reflect sampling over images only.
- The same images are re-used across the six configurations, so results across configurations are not independent samples (the paired tests account for this by comparing the same images).
- The explained class for an attacked image is the model's (wrong) top-1 prediction. Heatmaps for the original class under attack were not examined. A different noise model would give a different threshold.

## Repository layout

```
src/model.py          ResNet-18 for CIFAR, checkpoint loading, input-normalizing wrapper
src/train.py          training script (writes checkpoint, chart, log)
src/gradcam_utils.py  Grad-CAM wrapper and sanity gallery
src/metrics.py        SSIM (primary), IoU of top regions and Pearson correlation (supporting)
src/calibration.py    held-out split, noise-floor distribution, locked threshold
src/attacks.py        FGSM/PGD evaluation
src/pipeline.py       attack -> explain -> compare, deceptive flag against the locked threshold
src/analyze.py        per-configuration statistics, hypothesis tests, charts
results/              logs, charts, gallery, calibration, tables (checkpoint weights are not committed)
reports/              progress reports
```
