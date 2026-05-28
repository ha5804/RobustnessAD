import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


METRIC_COLUMNS = ["I-AUROC", "I-AP", "I-F1max", "P-AUROC", "P-AP", "P-F1max", "P-AUPRO"]
CONDITION_RE = re.compile(r"^(?P<corruption>.+)_s(?P<severity>\d+)$")


def parse_condition(condition):
    if condition == "clean":
        return "clean", 0
    match = CONDITION_RE.match(condition)
    if not match:
        return condition, 0
    return match.group("corruption"), int(match.group("severity"))


def collect_class_metrics(root):
    rows = []
    for path in sorted(root.glob("*/*/*/*/class_metrics_*.csv")):
        rel = path.relative_to(root).parts
        if len(rel) < 5:
            continue
        model, dataset, split, condition = rel[:4]
        corruption, severity = parse_condition(condition)
        df = pd.read_csv(path)
        df.insert(0, "severity", severity)
        df.insert(0, "corruption", corruption)
        df.insert(0, "split", split)
        df.insert(0, "dataset", dataset)
        df.insert(0, "model", model)
        rows.append(df)
    if not rows:
        return pd.DataFrame(columns=["model", "dataset", "split", "corruption", "severity", "class"] + METRIC_COLUMNS)
    return pd.concat(rows, ignore_index=True)


def metric_columns(df):
    return [col for col in METRIC_COLUMNS if col in df.columns]


def build_drop_table(raw):
    metrics = metric_columns(raw)
    if not metrics:
        return pd.DataFrame()

    id_cols = ["model", "dataset", "split", "class"]
    clean = raw[raw["corruption"] == "clean"][id_cols + metrics].copy()
    corrupt = raw[raw["corruption"] != "clean"][id_cols + ["corruption", "severity"] + metrics].copy()
    if clean.empty or corrupt.empty:
        return pd.DataFrame()

    long_rows = []
    merged = corrupt.merge(clean, on=id_cols, suffixes=("_corrupt", "_clean"))
    for metric in metrics:
        item = merged[id_cols + ["corruption", "severity", f"{metric}_clean", f"{metric}_corrupt"]].copy()
        item = item.rename(columns={f"{metric}_clean": "clean_metric", f"{metric}_corrupt": "corrupt_metric"})
        item["metric"] = metric
        item["absolute_drop"] = item["clean_metric"] - item["corrupt_metric"]
        item["relative_drop"] = np.where(
            item["clean_metric"].abs() > 1e-12,
            item["absolute_drop"] / item["clean_metric"],
            np.nan,
        )
        long_rows.append(item)
    return pd.concat(long_rows, ignore_index=True)


def build_macro_summary(raw):
    metrics = metric_columns(raw)
    if not metrics:
        return pd.DataFrame()

    long = raw.melt(
        id_vars=["model", "dataset", "split", "corruption", "severity", "class"],
        value_vars=metrics,
        var_name="metric",
        value_name="value",
    )
    dataset_macro = (
        long.groupby(["model", "dataset", "split", "corruption", "severity", "metric"], dropna=False)["value"]
        .mean()
        .reset_index()
    )
    unified_macro = (
        long.groupby(["model", "split", "corruption", "severity", "metric"], dropna=False)["value"]
        .mean()
        .reset_index()
    )
    unified_macro.insert(1, "dataset", "unified")
    return pd.concat([dataset_macro, unified_macro], ignore_index=True)


def build_robustness_summary(summary):
    if summary.empty:
        return pd.DataFrame()

    keys = ["model", "dataset", "split", "metric"]
    clean = summary[summary["corruption"] == "clean"][keys + ["value"]].rename(columns={"value": "clean_metric"})
    corrupt = (
        summary[summary["corruption"] != "clean"]
        .groupby(keys, dropna=False)["value"]
        .mean()
        .reset_index()
        .rename(columns={"value": "mean_corrupt_metric"})
    )
    merged = clean.merge(corrupt, on=keys, how="inner")
    merged["mean_drop"] = merged["clean_metric"] - merged["mean_corrupt_metric"]
    merged["robustness_ratio"] = np.where(
        merged["clean_metric"].abs() > 1e-12,
        merged["mean_corrupt_metric"] / merged["clean_metric"],
        np.nan,
    )
    return merged


def build_micro_summary(raw, samples):
    metrics = metric_columns(raw)
    if not metrics or samples.empty:
        return pd.DataFrame()

    counts = (
        samples.groupby(["model", "dataset", "split", "corruption", "severity", "class"], dropna=False)
        .size()
        .reset_index(name="sample_count")
    )
    merged = raw.merge(counts, on=["model", "dataset", "split", "corruption", "severity", "class"], how="left")
    merged["sample_count"] = merged["sample_count"].fillna(0.0)
    long = merged.melt(
        id_vars=["model", "dataset", "split", "corruption", "severity", "class", "sample_count"],
        value_vars=metrics,
        var_name="metric",
        value_name="value",
    )
    long = long.dropna(subset=["value"])

    def weighted_mean(group):
        weights = group["sample_count"].to_numpy(dtype=np.float64)
        values = group["value"].to_numpy(dtype=np.float64)
        if weights.sum() <= 0:
            return float(np.nanmean(values))
        return float(np.average(values, weights=weights))

    dataset_micro = (
        long.groupby(["model", "dataset", "split", "corruption", "severity", "metric"], dropna=False)
        .apply(weighted_mean, include_groups=False)
        .reset_index(name="value")
    )
    unified_micro = (
        long.groupby(["model", "split", "corruption", "severity", "metric"], dropna=False)
        .apply(weighted_mean, include_groups=False)
        .reset_index(name="value")
    )
    unified_micro.insert(1, "dataset", "unified")
    return pd.concat([dataset_micro, unified_micro], ignore_index=True)


def sample_key_from_path(path):
    parts = Path(str(path)).parts
    if "test" in parts:
        test_idx = parts.index("test")
        return "/".join(parts[max(test_idx - 1, 0) :])
    for marker in ["MVTec", "Visa", "BTAD", "mvtec", "visa", "btad"]:
        if marker in parts:
            return "/".join(parts[parts.index(marker) + 1 :])
    return str(path)


def minmax(x):
    x = np.asarray(x, dtype=np.float32)
    lo, hi = float(np.min(x)), float(np.max(x))
    if hi - lo < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return (x - lo) / (hi - lo)


def pixel_quality(pred_map, gt_mask):
    gt = gt_mask.astype(np.uint8).reshape(-1)
    pred = pred_map.astype(np.float32).reshape(-1)
    if gt.max() == gt.min():
        return np.nan
    try:
        return float(roc_auc_score(gt, pred))
    except ValueError:
        return np.nan


def collect_sample_scores(root):
    rows = []
    for path in sorted(root.glob("*/*/*/*/difficulty_inputs/*/all_predictions.npz")):
        rel = path.relative_to(root).parts
        if len(rel) < 6:
            continue
        model, dataset, split, condition = rel[:4]
        corruption, severity = parse_condition(condition)
        data = np.load(path, allow_pickle=True)
        image_scores = minmax(data["pr_anomalys"])
        for idx, query_path in enumerate(data["query_paths"]):
            label = int(data["gt_anomalys"][idx])
            p_auc = pixel_quality(data["pr_masks"][idx], data["gt_masks"][idx])
            rows.append(
                {
                    "model": model,
                    "dataset": dataset,
                    "split": split,
                    "corruption": corruption,
                    "severity": severity,
                    "class": str(data["cls_names"][idx]),
                    "sample_key": sample_key_from_path(str(query_path)),
                    "query_path": str(query_path),
                    "label": label,
                    "image_score": float(image_scores[idx]),
                    "pixel_auroc_sample": p_auc,
                }
            )
    return pd.DataFrame(rows)


def build_sample_drop_table(samples):
    if samples.empty:
        return pd.DataFrame()
    keys = ["model", "dataset", "split", "class", "sample_key", "label"]
    clean = samples[samples["corruption"] == "clean"][keys + ["image_score", "pixel_auroc_sample"]]
    corrupt = samples[samples["corruption"] != "clean"][keys + ["corruption", "severity", "image_score", "pixel_auroc_sample", "query_path"]]
    merged = corrupt.merge(clean, on=keys, suffixes=("_corrupt", "_clean"))
    if merged.empty:
        return merged
    merged["image_score_delta"] = merged["image_score_corrupt"] - merged["image_score_clean"]
    merged["pixel_auroc_drop"] = merged["pixel_auroc_sample_clean"] - merged["pixel_auroc_sample_corrupt"]
    merged["sensitivity_score"] = np.where(
        merged["label"] == 1,
        merged["pixel_auroc_drop"].fillna(0.0) + np.maximum(-merged["image_score_delta"], 0.0),
        np.maximum(merged["image_score_delta"], 0.0),
    )
    return merged.sort_values("sensitivity_score", ascending=False)


def main():
    parser = argparse.ArgumentParser("Summarize unified corruption robustness benchmark")
    parser.add_argument("--root", default="results/corruption_benchmark")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--topk", type=int, default=100)
    args = parser.parse_args()

    root = Path(args.root)
    output_dir = Path(args.output_dir) if args.output_dir else root / "summaries"
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = collect_class_metrics(root)
    raw.to_csv(output_dir / "class_metrics_raw.csv", index=False)

    drops = build_drop_table(raw)
    drops.to_csv(output_dir / "clean_vs_corruption_drop.csv", index=False)

    macro = build_macro_summary(raw)
    macro.to_csv(output_dir / "unified_macro_summary.csv", index=False)

    robust = build_robustness_summary(macro)
    robust.to_csv(output_dir / "robustness_summary.csv", index=False)

    if not drops.empty:
        top_classes = (
            drops[drops["metric"].isin(["P-AUROC", "P-AP", "P-F1max"])]
            .sort_values(["absolute_drop", "relative_drop"], ascending=False)
            .head(args.topk)
        )
        top_classes.to_csv(output_dir / "top_corruption_sensitive_classes.csv", index=False)

    samples = collect_sample_scores(root)
    samples.to_csv(output_dir / "sample_scores_raw.csv", index=False)

    micro = build_micro_summary(raw, samples)
    micro.to_csv(output_dir / "unified_micro_summary.csv", index=False)

    sample_drops = build_sample_drop_table(samples)
    sample_drops.to_csv(output_dir / "sample_clean_vs_corruption_drop.csv", index=False)
    if not sample_drops.empty:
        sample_drops.head(args.topk).to_csv(output_dir / "top_corruption_sensitive_samples.csv", index=False)

    print(f"wrote summaries to: {output_dir}")


if __name__ == "__main__":
    main()
