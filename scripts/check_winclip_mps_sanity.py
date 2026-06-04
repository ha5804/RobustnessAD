"""Quick WinCLIP sanity check before rerunning full corruption benchmarks.

This script is intentionally small:
- runs only a few MVTec classes;
- limits samples per class;
- uses MPS for inference when available;
- computes metrics on CPU;
- reports heatmap variance so constant-map failures are easy to catch.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset import Dataset, PromptDataset
from models.wincliplib.winclip import WinCLIP
from test_winclip import ensure_meta_json, limit_test_samples_per_class, prepare_class_model, resolve_dataset_path
from tools import get_transform, setup_seed
from tools.metric import Evaluator


def select_device(device_arg):
    if device_arg == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device_arg)


def as_numpy_results(results):
    return {
        key: value.detach().cpu().numpy() if hasattr(value, "detach") else value
        for key, value in results.items()
    }


def summarize_heatmaps(pr_masks):
    flat = pr_masks.flatten(1)
    per_sample_std = flat.std(dim=1)
    per_sample_range = flat.max(dim=1).values - flat.min(dim=1).values
    return {
        "std_mean": float(per_sample_std.mean().item()),
        "std_min": float(per_sample_std.min().item()),
        "range_mean": float(per_sample_range.mean().item()),
        "range_min": float(per_sample_range.min().item()),
        "constant_count": int((per_sample_range <= 1e-8).sum().item()),
    }


def run_one(args, device, dataset_dir, cls_name, corruption):
    preprocess, target_transform = get_transform(image_size=args.image_size)
    save_dir = Path(args.save_path) / "winclip_mps_sanity"
    save_dir.mkdir(parents=True, exist_ok=True)

    test_data = Dataset(
        root=dataset_dir,
        transform=preprocess,
        target_transform=target_transform,
        dataset_name=args.dataset,
        k_shots=args.k_shots,
        save_dir=str(save_dir),
        mode="test",
        seed=args.seed,
        class_name=cls_name,
        corruption=corruption,
        corruption_severity=args.corruption_severity if corruption else 0,
    )
    limit_test_samples_per_class(test_data, args.max_samples)

    prompt_data = PromptDataset(
        root=dataset_dir,
        transform=preprocess,
        target_transform=target_transform,
        dataset_name=args.dataset,
        k_shots=args.k_shots,
        save_dir=str(save_dir),
        mode="test",
        seed=args.seed,
        class_name=cls_name,
    )

    winclip = WinCLIP(
        device=device,
        backbone=args.backbone,
        pretrained_dataset=args.pretrained_dataset,
        scales=args.scales,
        out_size_h=args.image_size,
        out_size_w=args.image_size,
        img_resize=args.image_size,
        img_cropsize=args.image_size,
        precision=args.precision,
        use_visual_gallery=args.use_visual_gallery,
        batch_size=args.batch_size,
        fusion_version=args.fusion_version,
        image_score_mode="global",
    )
    prompt_loader = DataLoader(prompt_data, batch_size=args.batch_size, shuffle=False, num_workers=0)
    prepare_class_model(winclip, cls_name, prompt_loader, args)

    loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    gt_masks, pr_masks, gt_anomalys, pr_anomalys, cls_names = [], [], [], [], []

    for items in loader:
        images = items["img"]
        gt_mask = items["img_mask"][:, 0]
        gt_mask[gt_mask > 0.5], gt_mask[gt_mask <= 0.5] = 1, 0

        with torch.no_grad():
            image_scores, heatmaps = winclip.predict_batch(images)

        pixel_maps = torch.stack(heatmaps, dim=0).cpu()
        if args.sigma > 0:
            pixel_maps = torch.stack(
                [torch.from_numpy(gaussian_filter(h.numpy(), sigma=args.sigma)).float() for h in pixel_maps],
                dim=0,
            )
        pixel_maps = torch.nan_to_num(pixel_maps, nan=0.0, posinf=0.0, neginf=0.0)

        gt_masks.append(gt_mask.int().cpu())
        pr_masks.append(pixel_maps)
        gt_anomalys.append(items["anomaly"].int().cpu())
        pr_anomalys.append(torch.nan_to_num(image_scores.detach().cpu(), nan=0.0, posinf=0.0, neginf=0.0))
        cls_names.append(np.array(items["cls_name"]))

    results = {
        "gt_masks": torch.cat(gt_masks, dim=0),
        "pr_masks": torch.cat(pr_masks, dim=0),
        "gt_anomalys": torch.cat(gt_anomalys, dim=0),
        "pr_anomalys": torch.cat(pr_anomalys, dim=0),
        "cls_names": np.concatenate(cls_names, axis=0),
    }

    metrics = Evaluator(metrics=["I-AUROC", "P-AUROC", "P-AP", "P-F1max"]).run(
        as_numpy_results(results),
        cls_name,
    )
    heatmap_stats = summarize_heatmaps(results["pr_masks"])
    n_samples = int(results["gt_anomalys"].numel())
    n_anom = int(results["gt_anomalys"].sum().item())

    return {
        "class": cls_name,
        "corruption": corruption or "clean",
        "n": n_samples,
        "anom": n_anom,
        **metrics,
        **heatmap_stats,
    }


def print_table(rows):
    headers = [
        "corruption",
        "class",
        "n",
        "anom",
        "I-AUROC",
        "P-AUROC",
        "P-AP",
        "P-F1max",
        "std_mean",
        "range_min",
        "constant_count",
        "status",
    ]
    print("\t".join(headers))
    for row in rows:
        status = "OK"
        if row["constant_count"] > 0 or row["range_min"] <= 1e-8:
            status = "CHECK_CONSTANT_MAP"
        elif row["P-AUROC"] < 0.6:
            status = "CHECK_LOW_PAUROC"

        values = []
        for key in headers[:-1]:
            value = row[key]
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        values.append(status)
        print("\t".join(values))


def main():
    parser = argparse.ArgumentParser("WinCLIP MPS sanity check")
    parser.add_argument("--dataset", default="mvtec")
    parser.add_argument("--test_data_path", default="./dataset/MVTec")
    parser.add_argument("--save_path", default="./results")
    parser.add_argument("--classes", nargs="+", default=["bottle", "screw", "transistor"])
    parser.add_argument("--corruptions", nargs="+", default=["clean", "gaussian_noise"])
    parser.add_argument("--corruption_severity", type=int, default=1)
    parser.add_argument("--max_samples", type=int, default=24)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--image_size", type=int, default=240)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--sigma", type=int, default=4)
    parser.add_argument("--k_shots", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    parser.add_argument("--backbone", default="ViT-B-16-plus-240")
    parser.add_argument("--pretrained_dataset", default="laion400m_e32")
    parser.add_argument("--scales", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--precision", default=None, choices=[None, "fp32", "fp16"])
    parser.add_argument("--fusion_version", default="textual", choices=["textual", "visual", "textual_visual"])
    parser.add_argument("--use_visual_gallery", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--replace_underscore", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    setup_seed(args.seed)
    device = select_device(args.device)
    dataset_dir = ensure_meta_json(args.dataset, resolve_dataset_path(args.dataset, args.test_data_path))

    print(f"device={device} dataset={args.dataset} max_samples={args.max_samples} k_shots={args.k_shots}")
    print("A healthy run should have constant_count=0 and non-zero heatmap std/range.")

    rows = []
    for corruption in args.corruptions:
        corruption_arg = None if corruption == "clean" else corruption
        for cls_name in args.classes:
            print(f"\nRunning {cls_name} / {corruption} ...")
            rows.append(run_one(args, device, dataset_dir, cls_name, corruption_arg))

    print("\nSummary")
    print_table(rows)


if __name__ == "__main__":
    main()
