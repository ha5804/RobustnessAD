import argparse
import csv
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from tools.create_difficulty import compute_difficulty, split_30_40_30


CSV_FIELDS = [
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


def parse_input(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError("--input must be formatted as dataset=/path/to/predictions.npz")
    dataset, path = value.split("=", 1)
    dataset = dataset.strip()
    path = path.strip()
    if not dataset or not path:
        raise argparse.ArgumentTypeError("--input must be formatted as dataset=/path/to/predictions.npz")
    return dataset, path


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser("Create unified clean difficulty split across datasets")
    parser.add_argument("--input", action="append", required=True, type=parse_input, help="dataset=npz_path; repeat for each dataset")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--shot", type=int, required=True)
    args = parser.parse_args()

    rows = []
    inputs = {}
    for dataset, npz_path in args.input:
        inputs[dataset] = npz_path
        rows.extend(compute_difficulty(npz_path, dataset, args.method, args.seed, args.shot))

    os.makedirs(args.output_dir, exist_ok=True)
    easy, normal, hard = split_30_40_30(rows)
    all_rows = sorted(easy + normal + hard, key=lambda x: x["rank"])

    write_csv(os.path.join(args.output_dir, "all.csv"), all_rows)
    write_csv(os.path.join(args.output_dir, "easy.csv"), sorted(easy, key=lambda x: x["rank"]))
    write_csv(os.path.join(args.output_dir, "normal.csv"), sorted(normal, key=lambda x: x["rank"]))
    write_csv(os.path.join(args.output_dir, "hard.csv"), sorted(hard, key=lambda x: x["rank"]))

    summary = {
        "_meta": {
            "method": args.method,
            "seed": args.seed,
            "shot": args.shot,
            "inputs": inputs,
            "split": "unified 30/40/30 over all input datasets, sorted by clean difficulty_score ascending",
        },
        "global": {
            "easy": len(easy),
            "normal": len(normal),
            "hard": len(hard),
            "total": len(all_rows),
        },
        "datasets": {},
    }

    for dataset in sorted(inputs):
        dataset_rows = [row for row in all_rows if row["dataset"] == dataset]
        summary["datasets"][dataset] = {
            "easy": sum(row["difficulty"] == "easy" for row in dataset_rows),
            "normal": sum(row["difficulty"] == "normal" for row in dataset_rows),
            "hard": sum(row["difficulty"] == "hard" for row in dataset_rows),
            "total": len(dataset_rows),
        }

    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=4)

    with open(os.path.join(args.output_dir, "all.jsonl"), "w") as f:
        for row in all_rows:
            f.write(json.dumps(row) + "\n")

    print(
        f"unified: total={summary['global']['total']}, "
        f"easy={summary['global']['easy']}, "
        f"normal={summary['global']['normal']}, "
        f"hard={summary['global']['hard']}"
    )
    for dataset, item in summary["datasets"].items():
        print(
            f"{dataset}: total={item['total']}, easy={item['easy']}, "
            f"normal={item['normal']}, hard={item['hard']}"
        )
    print(f"Saved unified difficulty split to: {args.output_dir}")


if __name__ == "__main__":
    main()
