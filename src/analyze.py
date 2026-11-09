"""Aggregate pipeline results into tables, hypothesis tests and charts.

Reads results/tables/pipeline_n{N}_per_image.csv (one row per successfully attacked image) and writes:
  results/tables/analysis_n{N}_summary.csv           per-configuration statistics with 95% confidence intervals
  results/tables/analysis_n{N}_hypothesis_tests.csv  paired comparisons for H1 (strength) and H2 (attack type)
  results/tables/analysis_n{N}_deceptive_cases.csv   every attacked explanation at or above the locked threshold
  results/tables/analysis_n{N}_exploratory_*.csv     post-hoc class-pair tables (see below)
  results/charts/*_n{N}.png                          charts

Statistical choices (fixed in code before the full-scale results existed):
  - 95% intervals for means use a percentile bootstrap (10,000 resamples, seeded).
  - H1 compares the same images across epsilon levels (images successfully attacked at both), H2 compares the
    same images under FGSM and PGD (images successfully attacked by both). Each comparison reports the mean
    paired difference in SSIM with a bootstrap interval and a Wilcoxon signed-rank p-value, Holm-adjusted within
    each hypothesis. Effect sizes and intervals matter more than p-values: with thousands of images, tiny
    differences become "significant".
  - Deceptive rates carry exact (Clopper-Pearson) 95% intervals.
  - EXPLORATORY (added after the full-scale results were seen, not part of the pre-set plan): per class-pair tables, and
    a comparison of truck<->automobile confusions against all other successful attacks.
  - The locked SSIM threshold is the only rule that decides "deceptive". Shares of pairs above other SSIM values
    are reported as descriptive context only.

Usage:
    python -m src.analyze --n 9000
"""

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src.train import ROOT

TABLES_DIR = ROOT / "results" / "tables"
CHARTS_DIR = ROOT / "results" / "charts"
CALIBRATION_DIR = ROOT / "results" / "calibration"

BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 0
ALPHA = 0.05

# Chart styling: one color per attack, used in every chart (color follows the entity)
COLORS = {"FGSM": "#2a78d6", "PGD": "#eb6834"}
SURFACE, INK, INK_SECONDARY, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
EPSILONS = (0.01, 0.03, 0.05)
CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def bootstrap_mean_ci(values, rng):
    values = np.asarray(values, dtype=np.float64)
    means = np.empty(BOOTSTRAP_RESAMPLES)
    chunk = 500
    for start in range(0, BOOTSTRAP_RESAMPLES, chunk):
        size = min(chunk, BOOTSTRAP_RESAMPLES - start)
        means[start : start + size] = values[rng.integers(0, len(values), (size, len(values)))].mean(axis=1)
    low, high = np.percentile(means, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
    return float(low), float(high)


def clopper_pearson(successes, trials):
    low = 0.0 if successes == 0 else stats.beta.ppf(ALPHA / 2, successes, trials - successes + 1)
    high = 1.0 if successes == trials else stats.beta.ppf(1 - ALPHA / 2, successes + 1, trials - successes)
    return float(low), float(high)


def holm_adjust(p_values):
    p = np.asarray(p_values, dtype=np.float64)
    adjusted = np.empty(len(p))
    running = 0.0
    for rank, i in enumerate(np.argsort(p)):
        running = max(running, (len(p) - rank) * p[i])
        adjusted[i] = min(1.0, running)
    return adjusted


def summarize_config(frame, clean_correct, threshold, rng):
    row = {"attacked_images": len(frame), "attack_success_rate_on_subset": len(frame) / clean_correct}
    for name, column in (("ssim", "ssim"), ("iou", "iou_top20"), ("pearson", "pearson")):
        values = frame[column].to_numpy()
        low, high = bootstrap_mean_ci(values, rng)
        row.update({
            f"{name}_mean": values.mean(), f"{name}_ci_low": low, f"{name}_ci_high": high,
            f"{name}_median": np.median(values), f"{name}_std": values.std(ddof=1),
        })
    ssim = frame["ssim"].to_numpy()
    deceptive = int((ssim >= threshold).sum())
    ci_low, ci_high = clopper_pearson(deceptive, len(ssim))
    row.update({
        "deceptive_count": deceptive, "deceptive_rate": deceptive / len(ssim),
        "deceptive_ci_low": ci_low, "deceptive_ci_high": ci_high,
        "share_ssim_ge_0.90": float((ssim >= 0.90).mean()), "share_ssim_ge_0.95": float((ssim >= 0.95).mean()),
    })
    return row


def paired_test(frame_a, frame_b, rng):
    """b minus a on the same images (matched by subset_position). Returns summary of the paired SSIM difference."""
    merged = frame_a[["subset_position", "ssim"]].merge(frame_b[["subset_position", "ssim"]], on="subset_position", suffixes=("_a", "_b"))
    diff = (merged["ssim_b"] - merged["ssim_a"]).to_numpy()
    low, high = bootstrap_mean_ci(diff, rng)
    return {
        "paired_images": len(diff), "mean_difference": float(diff.mean()), "ci_low": low, "ci_high": high,
        "wilcoxon_p": float(stats.wilcoxon(diff).pvalue),
    }


def hypothesis_tests(per_image, rng):
    groups = {key: g for key, g in per_image.groupby(["attack", "epsilon"])}
    rows = []
    for attack in ("FGSM", "PGD"):
        for low_eps, high_eps in ((0.01, 0.03), (0.03, 0.05), (0.01, 0.05)):
            result = paired_test(groups[(attack, low_eps)], groups[(attack, high_eps)], rng)
            rows.append({"hypothesis": "H1 strength", "comparison": f"{attack}: SSIM at eps {high_eps} minus eps {low_eps}", **result})
    h1_adjusted = holm_adjust([r["wilcoxon_p"] for r in rows])
    for r, adj in zip(rows, h1_adjusted):
        r["holm_p"] = float(adj)
    h2_rows = []
    for eps in EPSILONS:
        result = paired_test(groups[("FGSM", eps)], groups[("PGD", eps)], rng)
        h2_rows.append({"hypothesis": "H2 attack type", "comparison": f"eps {eps}: SSIM under PGD minus FGSM", **result})
    for r, adj in zip(h2_rows, holm_adjust([r["wilcoxon_p"] for r in h2_rows])):
        r["holm_p"] = float(adj)
    table = pd.DataFrame(rows + h2_rows)
    table["interval_excludes_zero"] = (table["ci_low"] > 0) | (table["ci_high"] < 0)
    return table


def class_pair_table(per_image, threshold):
    """Exploratory: for every (true class, attacked class) pair, how many attacks and how similar the explanations stay."""
    frame = per_image.assign(true_class=per_image["label"].map(dict(enumerate(CLASS_NAMES))),
                             attacked_class=per_image["adv_pred"].map(dict(enumerate(CLASS_NAMES))))
    table = frame.groupby(["true_class", "attacked_class"]).agg(
        attacked_images=("ssim", "size"), median_ssim=("ssim", "median"),
        deceptive_count=("ssim", lambda v: int((v >= threshold).sum())),
    ).reset_index()
    return table.sort_values("attacked_images", ascending=False)


def vehicle_pair_comparison(per_image, threshold):
    """Exploratory: truck<->automobile confusions versus every other successful attack."""
    is_pair = ((per_image["label"] == 9) & (per_image["adv_pred"] == 1)) | ((per_image["label"] == 1) & (per_image["adv_pred"] == 9))
    rows = []
    for name, frame in (("truck<->automobile", per_image[is_pair]), ("all other attacks", per_image[~is_pair])):
        deceptive = int((frame["ssim"] >= threshold).sum())
        low, high = clopper_pearson(deceptive, len(frame))
        rows.append({"group": name, "attacked_images": len(frame), "share_of_all_attacks": len(frame) / len(per_image),
                     "median_ssim": frame["ssim"].median(), "deceptive_count": deceptive, "deceptive_rate": deceptive / len(frame),
                     "deceptive_ci_low": low, "deceptive_ci_high": high})
    return pd.DataFrame(rows)


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_SECONDARY)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    ax.title.set_color(INK)


def new_figure(width=6.6, height=4.2, **kwargs):
    fig, axes = plt.subplots(figsize=(width, height), facecolor=SURFACE, **kwargs)
    for ax in np.atleast_1d(axes).ravel():
        style_axes(ax)
    return fig, axes


def line_with_intervals(ax, summary, metric, attack):
    rows = summary[summary["attack"] == attack].sort_values("epsilon")
    y = rows[f"{metric}_mean"].to_numpy()
    yerr = np.vstack([y - rows[f"{metric}_ci_low"].to_numpy(), rows[f"{metric}_ci_high"].to_numpy() - y])
    ax.errorbar(rows["epsilon"], y, yerr=yerr, color=COLORS[attack], marker="o", markersize=7, linewidth=2,
                elinewidth=1.5, capsize=3, markeredgecolor=SURFACE, markeredgewidth=1.5, label=attack)
    ax.annotate(attack, (rows["epsilon"].iloc[-1], y[-1]), xytext=(10, 0), textcoords="offset points",
                va="center", fontsize=9, color=INK)


def chart_ssim_vs_epsilon(summary, threshold, path):
    fig, ax = new_figure()
    for attack in ("FGSM", "PGD"):
        line_with_intervals(ax, summary, "ssim", attack)
    ax.axhline(threshold, color=INK_SECONDARY, linestyle="--", linewidth=1.2)
    ax.text(0.0098, threshold + 0.012, f"locked deceptive threshold ({threshold:.4f})", fontsize=8.5, color=INK_SECONDARY, va="bottom")
    ax.set_xticks(EPSILONS)
    ax.set_xlim(0.0085, 0.0545)
    ax.set_ylim(0.3, 1.04)
    ax.set_xlabel("Attack strength (epsilon)")
    ax.set_ylabel("Mean SSIM, clean vs attacked heatmap")
    ax.set_title("Attacked explanations stay far below the deceptive threshold", fontsize=11, loc="left")
    ax.legend(frameon=False, loc="lower left", fontsize=9, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def chart_distributions(per_image, threshold, path):
    fig, ax = new_figure(width=7.2)
    handles = []
    for i, eps in enumerate(EPSILONS):
        for j, attack in enumerate(("FGSM", "PGD")):
            values = per_image[(per_image["attack"] == attack) & (per_image["epsilon"] == eps)]["ssim"].to_numpy()
            box = ax.boxplot(values, positions=[i * 3 + j], widths=0.8, whis=(5, 95), showfliers=False, patch_artist=True,
                             medianprops={"color": INK, "linewidth": 1.5}, whiskerprops={"color": INK_SECONDARY, "linewidth": 1.2},
                             capprops={"color": INK_SECONDARY, "linewidth": 1.2}, boxprops={"edgecolor": SURFACE, "linewidth": 1.5})
            box["boxes"][0].set_facecolor(COLORS[attack])
            if i == 0:
                handles.append(box["boxes"][0])
    ax.axhline(threshold, color=INK_SECONDARY, linestyle="--", linewidth=1.2)
    ax.text(5.55, threshold + 0.008, f"locked threshold {threshold:.4f}", fontsize=8.5, color=INK_SECONDARY, ha="right", va="bottom")
    ax.set_xticks([0.5, 3.5, 6.5])
    ax.set_xticklabels([f"epsilon {e}" for e in EPSILONS])
    ax.set_xlim(-0.8, 7.3)
    ax.set_ylim(-0.3, 1.05)
    ax.set_ylabel("SSIM, clean vs attacked heatmap")
    ax.set_title("Spread of explanation similarity (box: middle 50%, whiskers: 5th to 95th percentile)", fontsize=10, loc="left")
    ax.legend(handles, ["FGSM", "PGD"], frameon=False, loc="lower right", fontsize=9, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def chart_deceptive_rate(summary, path):
    """One row per configuration: the observed rate and its exact 95% interval (no lines between configurations)."""
    from matplotlib.lines import Line2D

    fig, ax = new_figure(width=7.4, height=3.9)
    ordered = summary.sort_values(["attack", "epsilon"]).reset_index(drop=True)
    positions = [0, 1, 2, 3.7, 4.7, 5.7]
    max_high = 100 * ordered["deceptive_ci_high"].max()
    for y, (_, row) in zip(positions, ordered.iterrows()):
        color = COLORS[row["attack"]]
        ax.hlines(y, 100 * row["deceptive_ci_low"], 100 * row["deceptive_ci_high"], color=color, linewidth=2)
        ax.plot(100 * row["deceptive_rate"], y, "o", color=color, markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5)
        ax.text(100 * row["deceptive_ci_high"] + 0.02 * max_high, y, f"{row['deceptive_count']} of {row['attacked_images']}",
                va="center", fontsize=8.5, color=INK_SECONDARY)
    ax.set_yticks(positions)
    ax.set_yticklabels([f"{r['attack']}  eps {r['epsilon']}" for _, r in ordered.iterrows()])
    ax.invert_yaxis()
    ax.set_xlim(-0.03 * max_high, 1.3 * max_high)
    ax.yaxis.grid(False)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_xlabel("Deceptive explanations, % of attacked images (bar: exact 95% interval)")
    ax.set_title("Deceptive explanations are rare under the locked threshold", fontsize=11, loc="left")
    ax.legend([Line2D([0], [0], color=COLORS[a], marker="o", markersize=7, linewidth=2) for a in ("FGSM", "PGD")],
              ["FGSM", "PGD"], frameon=False, loc="lower right", fontsize=9, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def chart_supporting_metrics(summary, path):
    fig, axes = new_figure(width=8.4, height=3.9, ncols=2, sharex=True)
    for ax, (metric, label) in zip(axes, (("iou", "Mean IoU of top-20% regions"), ("pearson", "Mean Pearson correlation"))):
        for attack in ("FGSM", "PGD"):
            line_with_intervals(ax, summary, metric, attack)
        ax.set_xticks(EPSILONS)
        ax.set_xlim(0.0085, 0.0555)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Attack strength (epsilon)")
        ax.set_ylabel(label)
    axes[0].legend(frameon=False, loc="lower left", fontsize=9, labelcolor=INK)
    fig.suptitle("Supporting metrics order the configurations the same way SSIM does", fontsize=11, x=0.01, ha="left", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=9000, help="subset size the pipeline was run with")
    args = parser.parse_args()
    tag = f"n{args.n}"

    threshold = json.loads((CALIBRATION_DIR / "threshold.json").read_text())["threshold"]
    per_image = pd.read_csv(TABLES_DIR / f"pipeline_{tag}_per_image.csv")
    pipeline_summary = pd.read_csv(TABLES_DIR / f"pipeline_{tag}_summary.csv")
    clean_correct = int(pipeline_summary["clean_correct"].iloc[0])
    rng = np.random.default_rng(BOOTSTRAP_SEED)

    rows = []
    for (attack, eps), frame in per_image.groupby(["attack", "epsilon"]):
        rows.append({"attack": attack, "epsilon": eps, **summarize_config(frame, clean_correct, threshold, rng)})
    summary = pd.DataFrame(rows).sort_values(["attack", "epsilon"]).reset_index(drop=True)
    tests = hypothesis_tests(per_image, rng)

    evaluation_idx = json.loads((CALIBRATION_DIR / "split_indices.json").read_text())["evaluation_indices"]
    cases = per_image[per_image["ssim"] >= threshold].copy()
    cases.insert(2, "test_index", [evaluation_idx[p] for p in cases["subset_position"]])

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(TABLES_DIR / f"analysis_{tag}_summary.csv", index=False)
    tests.to_csv(TABLES_DIR / f"analysis_{tag}_hypothesis_tests.csv", index=False)
    cases.to_csv(TABLES_DIR / f"analysis_{tag}_deceptive_cases.csv", index=False)
    class_pair_table(per_image, threshold).to_csv(TABLES_DIR / f"analysis_{tag}_exploratory_class_pairs.csv", index=False)
    vehicles = vehicle_pair_comparison(per_image, threshold)
    vehicles.to_csv(TABLES_DIR / f"analysis_{tag}_exploratory_vehicle_pair.csv", index=False)

    chart_ssim_vs_epsilon(summary, threshold, CHARTS_DIR / f"ssim_vs_epsilon_{tag}.png")
    chart_distributions(per_image, threshold, CHARTS_DIR / f"ssim_distributions_{tag}.png")
    chart_deceptive_rate(summary, CHARTS_DIR / f"deceptive_rate_{tag}.png")
    chart_supporting_metrics(summary, CHARTS_DIR / f"supporting_metrics_{tag}.png")

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 40)
    print(summary[["attack", "epsilon", "attacked_images", "ssim_mean", "ssim_ci_low", "ssim_ci_high", "ssim_median",
                   "iou_mean", "pearson_mean", "deceptive_count", "deceptive_rate", "share_ssim_ge_0.90"]].round(4).to_string(index=False))
    print()
    print(tests[["hypothesis", "comparison", "paired_images", "mean_difference", "ci_low", "ci_high", "holm_p"]].round(4).to_string(index=False))
    print(f"\nDeceptive cases: {len(cases)} rows, {cases['subset_position'].nunique()} distinct images")
    print("\nExploratory (post hoc):")
    print(vehicles.round(5).to_string(index=False))


if __name__ == "__main__":
    main()
