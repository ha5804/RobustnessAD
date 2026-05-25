import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def normalize(x):
    x = np.asarray(x, dtype=np.float32)
    lo = float(np.min(x))
    hi = float(np.max(x))
    if hi - lo < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return (x - lo) / (hi - lo)


def colorize_jet(scoremap):
    scoremap = normalize(scoremap)
    x = scoremap[..., None]
    red = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0.0, 1.0)
    green = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0.0, 1.0)
    blue = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0.0, 1.0)
    return np.concatenate([red, green, blue], axis=-1)


def overlay_heatmap(image_path, pred_map, gt_mask, alpha=0.45):
    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    pred = Image.fromarray((normalize(pred_map) * 255).astype(np.uint8)).resize((width, height), Image.Resampling.BILINEAR)
    heatmap = Image.fromarray((colorize_jet(np.asarray(pred) / 255.0) * 255).astype(np.uint8))

    overlay = Image.blend(image, heatmap, alpha=alpha)

    if gt_mask is not None:
        mask = Image.fromarray((np.asarray(gt_mask) > 0).astype(np.uint8) * 255).resize((width, height), Image.Resampling.NEAREST)
        draw = ImageDraw.Draw(overlay)
        mask_np = np.asarray(mask)
        ys, xs = np.where(mask_np > 0)
        if len(xs) > 0 and len(ys) > 0:
            draw.rectangle([int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())], outline=(0, 255, 0), width=3)

    return overlay


def load_split_rows(split_dir, cls_name, difficulty):
    path = Path(split_dir) / cls_name / f"{difficulty}.json"
    if not path.exists():
        return []
    with path.open() as f:
        return json.load(f)


def selected_rows(rows, topk):
    if len(rows) <= topk * 2:
        return [("low", idx + 1, row) for idx, row in enumerate(rows)]

    lows = [("low", idx + 1, row) for idx, row in enumerate(rows[:topk])]
    highs = [("high", idx + 1, row) for idx, row in enumerate(rows[-topk:])]
    return lows + highs


def main():
    parser = argparse.ArgumentParser("Render split heatmaps from saved npz predictions")
    parser.add_argument("--npz_path", required=True)
    parser.add_argument("--split_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--classes", nargs="*", default=None)
    parser.add_argument("--difficulties", nargs="+", default=["easy", "normal", "hard"])
    args = parser.parse_args()

    data = np.load(args.npz_path, allow_pickle=True)
    query_paths = [str(path) for path in data["query_paths"]]
    pred_maps = data["pr_masks"].astype(np.float32)
    gt_masks = data["gt_masks"].astype(np.uint8)
    cls_names = [str(cls_name) for cls_name in data["cls_names"]]

    by_path = {
        path: {
            "pred_map": pred_maps[idx],
            "gt_mask": gt_masks[idx],
            "class": cls_names[idx],
        }
        for idx, path in enumerate(query_paths)
    }

    classes = args.classes or sorted(set(cls_names))
    output_root = Path(args.output_dir)
    saved = 0

    for cls_name in classes:
        for difficulty in args.difficulties:
            rows = load_split_rows(args.split_dir, cls_name, difficulty)
            rows = sorted(rows, key=lambda row: float(row["difficulty_score"]))

            for side, rank, row in selected_rows(rows, args.topk):
                image_path = row["image_path"]
                if image_path not in by_path:
                    continue

                item = by_path[image_path]
                overlay = overlay_heatmap(image_path, item["pred_map"], item["gt_mask"])

                out_dir = output_root / cls_name / difficulty / side
                out_dir.mkdir(parents=True, exist_ok=True)

                src_name = Path(image_path).stem
                out_name = f"{rank:02d}_score_{float(row['difficulty_score']):.4f}_{src_name}.png"
                overlay.save(out_dir / out_name)
                saved += 1

    print(f"Saved {saved} heatmaps to: {output_root}")


if __name__ == "__main__":
    main()
