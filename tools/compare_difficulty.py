import argparse
import csv
import json
from pathlib import Path


DIFFICULTY_ORDER = {"easy": 0, "normal": 1, "hard": 2}


def read_rows(path):
    with Path(path).open(newline="") as f:
        return {row["sample_key"]: row for row in csv.DictReader(f)}


def as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def main():
    parser = argparse.ArgumentParser("Compare difficulty splits from two models")
    parser.add_argument("--left_csv", required=True)
    parser.add_argument("--right_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    left_rows = read_rows(args.left_csv)
    right_rows = read_rows(args.right_csv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    changed = []
    common_keys = sorted(set(left_rows) & set(right_rows))
    for key in common_keys:
        left = left_rows[key]
        right = right_rows[key]
        if left["difficulty"] == right["difficulty"]:
            continue

        left_level = DIFFICULTY_ORDER[left["difficulty"]]
        right_level = DIFFICULTY_ORDER[right["difficulty"]]
        changed.append(
            {
                "sample_key": key,
                "dataset": left["dataset"],
                "class": left["class"],
                "category": left["category"],
                "label": int(left["label"]),
                "left_method": left["method"],
                "left_difficulty": left["difficulty"],
                "left_score": as_float(left["difficulty_score"]),
                "left_percentile": as_float(left["percentile"]),
                "right_method": right["method"],
                "right_difficulty": right["difficulty"],
                "right_score": as_float(right["difficulty_score"]),
                "right_percentile": as_float(right["percentile"]),
                "difficulty_delta": right_level - left_level,
                "score_delta": as_float(right["difficulty_score"]) - as_float(left["difficulty_score"]),
                "image_path": left["image_path"],
            }
        )

    changed.sort(key=lambda row: (abs(row["difficulty_delta"]), abs(row["score_delta"])), reverse=True)

    fields = [
        "sample_key",
        "dataset",
        "class",
        "category",
        "label",
        "left_method",
        "left_difficulty",
        "left_score",
        "left_percentile",
        "right_method",
        "right_difficulty",
        "right_score",
        "right_percentile",
        "difficulty_delta",
        "score_delta",
        "image_path",
    ]
    with (output_dir / "changed.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(changed)

    with (output_dir / "changed.json").open("w") as f:
        json.dump(changed, f, indent=4)

    summary = {
        "left_csv": args.left_csv,
        "right_csv": args.right_csv,
        "left_total": len(left_rows),
        "right_total": len(right_rows),
        "common_total": len(common_keys),
        "changed_total": len(changed),
    }
    with (output_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=4)

    print(
        f"Compared {len(common_keys)} common samples, "
        f"changed={len(changed)}. Saved to: {output_dir}"
    )


if __name__ == "__main__":
    main()
