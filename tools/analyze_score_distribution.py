import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_condition(condition):
    if condition == "clean":
        return "clean", 0
    if condition.endswith("_s3"):
        return condition[:-3], 3
    return condition, 0


def collect_sample_scores(root):
    rows = []
    for path in sorted(root.glob("*/*/*/*/sample_scores_*.csv")):
        model, dataset, class_name, condition = path.relative_to(root).parts[:4]
        corruption, severity = parse_condition(condition)
        df = pd.read_csv(path)
        df.insert(0, "severity", severity)
        df.insert(0, "corruption", corruption)
        df.insert(0, "condition", condition)
        df.insert(0, "target_class", class_name)
        df.insert(0, "model", model)
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def summarize_distribution(scores):
    if scores.empty:
        return pd.DataFrame()

    summary = (
        scores.groupby(["model", "dataset", "target_class", "condition", "corruption", "severity", "label"], dropna=False)
        .agg(
            n=("image_score", "size"),
            mean_score=("image_score", "mean"),
            median_score=("image_score", "median"),
            std_score=("image_score", "std"),
            q25_score=("image_score", lambda x: x.quantile(0.25)),
            q75_score=("image_score", lambda x: x.quantile(0.75)),
        )
        .reset_index()
    )

    pivot = summary.pivot_table(
        index=["model", "dataset", "target_class", "condition", "corruption", "severity"],
        columns="label",
        values=["n", "mean_score", "median_score", "std_score", "q25_score", "q75_score"],
        aggfunc="first",
    )
    pivot.columns = [f"{name}_{'normal' if label == 0 else 'anomaly'}" for name, label in pivot.columns]
    pivot = pivot.reset_index()

    for col in ["mean_score_normal", "mean_score_anomaly", "median_score_normal", "median_score_anomaly"]:
        if col not in pivot:
            pivot[col] = np.nan
    pivot["mean_score_gap"] = pivot["mean_score_anomaly"] - pivot["mean_score_normal"]
    pivot["median_score_gap"] = pivot["median_score_anomaly"] - pivot["median_score_normal"]
    return pivot


def compute_clean_shifts(distribution):
    if distribution.empty:
        return pd.DataFrame()

    keys = ["model", "dataset", "target_class"]
    clean_cols = [
        "mean_score_normal",
        "mean_score_anomaly",
        "mean_score_gap",
        "median_score_normal",
        "median_score_anomaly",
        "median_score_gap",
    ]
    clean = distribution[distribution["condition"] == "clean"][keys + clean_cols].copy()
    clean = clean.rename(columns={col: f"clean_{col}" for col in clean_cols})

    corrupt = distribution[distribution["condition"] != "clean"].copy()
    merged = corrupt.merge(clean, on=keys, how="left")
    for col in clean_cols:
        merged[f"delta_{col}"] = merged[col] - merged[f"clean_{col}"]

    merged["fp_shift"] = merged["delta_mean_score_normal"]
    merged["fn_shift"] = -merged["delta_mean_score_anomaly"]
    merged["gap_collapse"] = -merged["delta_mean_score_gap"]
    return merged


def save_distribution_plots(scores, output_dir):
    if scores.empty:
        return

    import matplotlib.pyplot as plt

    plot_dir = output_dir / "score_distribution_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    for (model, dataset, target_class), group in scores.groupby(["model", "dataset", "target_class"]):
        conditions = ["clean"] + sorted([c for c in group["condition"].unique() if c != "clean"])
        fig, axes = plt.subplots(len(conditions), 1, figsize=(7, max(2.4 * len(conditions), 3)), sharex=True)
        if len(conditions) == 1:
            axes = [axes]

        for ax, condition in zip(axes, conditions):
            item = group[group["condition"] == condition]
            normal = item[item["label"] == 0]["image_score"]
            anomaly = item[item["label"] == 1]["image_score"]
            bins = np.linspace(
                float(item["image_score"].min()) if len(item) else 0.0,
                float(item["image_score"].max()) if len(item) else 1.0,
                30,
            )
            if len(normal):
                ax.hist(normal, bins=bins, alpha=0.55, label="normal", density=True)
            if len(anomaly):
                ax.hist(anomaly, bins=bins, alpha=0.55, label="anomaly", density=True)
            ax.set_title(condition)
            ax.set_ylabel("density")
            ax.legend(loc="best")

        axes[-1].set_xlabel("image anomaly score")
        fig.suptitle(f"{model} / {dataset} / {target_class}")
        fig.tight_layout()
        fig.savefig(plot_dir / f"{model}_{dataset}_{target_class}.png", dpi=160)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser("Analyze target-class anomaly score distributions")
    parser.add_argument("--root", default="results/target_failure_analysis")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--plots", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    root = Path(args.root)
    output_dir = Path(args.output_dir) if args.output_dir else root / "summaries"
    output_dir.mkdir(parents=True, exist_ok=True)

    scores = collect_sample_scores(root)
    scores.to_csv(output_dir / "target_sample_scores_raw.csv", index=False)

    distribution = summarize_distribution(scores)
    distribution.to_csv(output_dir / "target_score_distribution_summary.csv", index=False)

    shifts = compute_clean_shifts(distribution)
    shifts.to_csv(output_dir / "target_clean_to_corruption_score_shift.csv", index=False)

    if args.plots:
        save_distribution_plots(scores, output_dir)

    print(f"wrote target score analysis to: {output_dir}")


if __name__ == "__main__":
    main()
