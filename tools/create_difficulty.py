import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


def minmax(x):
    x = np.asarray(x, dtype=np.float32)
    lo = float(np.min(x))
    hi = float(np.max(x))
    if hi - lo < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return (x - lo) / (hi - lo)


def category_from_path(path):
    parts = Path(path).parts
    if "test" in parts:
        test_idx = parts.index("test")
        if test_idx + 1 < len(parts):
            return parts[test_idx + 1]
    return "unknown"


def sample_key_from_path(path):
    parts = Path(path).parts
    if "test" in parts:
        test_idx = parts.index("test")
        start_idx = max(test_idx - 1, 0)
        return "/".join(parts[start_idx:])
    for marker in ["MVTec", "Visa", "MPDD", "BTAD", "mvtec", "visa", "mpdd", "btad"]:
        if marker in parts:
            return "/".join(parts[parts.index(marker) + 1 :])
    return str(path)


def pixel_quality_for_anomaly(pred_map, gt_mask):
    gt = gt_mask.astype(np.uint8).reshape(-1)
    pred = pred_map.astype(np.float32).reshape(-1)
    if gt.max() == gt.min():
        return 0.5
    try:
        return float(roc_auc_score(gt, pred))
    except ValueError:
        return 0.5


def compute_difficulty(npz_path, dataset, method, seed, shot):
    data = np.load(npz_path, allow_pickle=True)

    query_paths = data["query_paths"]
    cls_names = data["cls_names"]
    labels = data["gt_anomalys"].astype(np.int32)
    image_scores = minmax(data["pr_anomalys"])
    pred_maps = data["pr_masks"].astype(np.float32)
    gt_masks = data["gt_masks"].astype(np.uint8)

    rows = []

    for i, path in enumerate(query_paths):
        label = int(labels[i])
        image_score = float(image_scores[i])
        pred_map = pred_maps[i]
        gt_mask = gt_masks[i]

        if label == 1:
            # anomaly image:
            # high image score is good, high pixel AUROC is good
            image_difficulty = 1.0 - image_score
            pixel_quality = pixel_quality_for_anomaly(pred_map, gt_mask)
            pixel_difficulty = 1.0 - pixel_quality
        else:
            # normal image:
            # high image score or strong heatmap means false positive
            image_difficulty = image_score
            pixel_difficulty = float(np.mean(minmax(pred_map)))

        difficulty_score = 0.5 * image_difficulty + 0.5 * pixel_difficulty

        rows.append(
            {
                "image_path": str(path),
                "sample_key": sample_key_from_path(str(path)),
                "dataset": dataset,
                "method": method,
                "seed": seed,
                "shot": shot,
                "class": str(cls_names[i]),
                "category": category_from_path(str(path)),
                "label": label,
                "image_score": image_score,
                "difficulty_score": float(difficulty_score),
            }
        )

    return rows


def split_30_40_30(rows):
    rows = sorted(rows, key=lambda x: x["difficulty_score"])

    n = len(rows)
    easy_end = int(round(n * 0.3))
    normal_end = int(round(n * 0.7))

    easy = rows[:easy_end]
    normal = rows[easy_end:normal_end]
    hard = rows[normal_end:]

    denom = max(n - 1, 1)
    for rank, item in enumerate(rows, start=1):
        item["rank"] = rank
        item["percentile"] = float((rank - 1) / denom)

    for item in easy:
        item["difficulty"] = "easy"
    for item in normal:
        item["difficulty"] = "normal"
    for item in hard:
        item["difficulty"] = "hard"

    return easy, normal, hard


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz_path", required=True)
    parser.add_argument("--output_dir", default="./results/difficulty_splits/mvtec")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--shot", type=int, required=True)
    args = parser.parse_args()

    rows = compute_difficulty(args.npz_path, args.dataset, args.method, args.seed, args.shot)

    os.makedirs(args.output_dir, exist_ok=True)

    summary = {
        "_meta": {
            "dataset": args.dataset,
            "method": args.method,
            "seed": args.seed,
            "shot": args.shot,
            "npz_path": args.npz_path,
            "split": "global 30/40/30, sorted by difficulty_score ascending",
        }
    }
    easy, normal, hard = split_30_40_30(rows)
    all_split_rows = easy + normal + hard

    summary["global"] = {
        "easy": len(easy),
        "normal": len(normal),
        "hard": len(hard),
        "total": len(all_split_rows),
    }

    by_class = {}
    for row in all_split_rows:
        by_class.setdefault(row["class"], []).append(row)

    for cls_name, cls_rows in sorted(by_class.items()):
        final = {
            split_name: [row for row in cls_rows if row["difficulty"] == split_name]
            for split_name in ["easy", "normal", "hard"]
        }

        class_dir = os.path.join(args.output_dir, cls_name)
        os.makedirs(class_dir, exist_ok=True)

        for split_name, split_rows in final.items():
            out_path = os.path.join(class_dir, f"{split_name}.json")
            with open(out_path, "w") as f:
                json.dump(split_rows, f, indent=4)

        summary[cls_name] = {
            "easy": len(final["easy"]),
            "normal": len(final["normal"]),
            "hard": len(final["hard"]),
            "total": sum(len(v) for v in final.values()),
        }

        print(
            f"{cls_name}: total={summary[cls_name]['total']}, "
            f"easy={summary[cls_name]['easy']}, "
            f"normal={summary[cls_name]['normal']}, "
            f"hard={summary[cls_name]['hard']}"
        )

    print(
        f"global: total={summary['global']['total']}, "
        f"easy={summary['global']['easy']}, "
        f"normal={summary['global']['normal']}, "
        f"hard={summary['global']['hard']}"
    )

    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)

    all_split_rows = sorted(all_split_rows, key=lambda x: x["rank"])
    with open(os.path.join(args.output_dir, "all.json"), "w") as f:
        json.dump(all_split_rows, f, indent=4)

    with open(os.path.join(args.output_dir, "all.jsonl"), "w") as f:
        for row in all_split_rows:
            f.write(json.dumps(row) + "\n")

    csv_fields = [
        "dataset",
        "method",
        "seed",
        "shot",
        "class",
        "category",
        "difficulty",
        "rank",
        "percentile",
        "difficulty_score",
        "image_score",
        "label",
        "sample_key",
        "image_path",
    ]
    with open(os.path.join(args.output_dir, "all.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(all_split_rows)

    for split_name, split_rows in [("easy", easy), ("normal", normal), ("hard", hard)]:
        split_rows = sorted(split_rows, key=lambda x: x["rank"])
        with open(os.path.join(args.output_dir, f"{split_name}.csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields)
            writer.writeheader()
            writer.writerows(split_rows)

    print(f"Saved difficulty split to: {args.output_dir}")


if __name__ == "__main__":
    main()
