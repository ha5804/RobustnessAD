"""Testing script for WinCLIP in the AdaptCLIP evaluation pipeline."""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from tabulate import tabulate
from tqdm import tqdm

from dataset import Dataset, PromptDataset
from tools import (
    Evaluator,
    SelectedHeatmapSaver,
    get_logger,
    get_transform,
    resolve_corruption_save_path,
    save_class_metrics,
    save_sample_scores,
    setup_seed,
    visualizer,
)
from models.wincliplib.winclip import WinCLIP


def select_device(device_arg):
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but CUDA is not available.")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested, but MPS is not available.")
    return device


def resolve_dataset_path(dataset_name, requested_path):
    #인자로 dataset_name과 사용자 경로를 받는다.
    candidates = []
    if requested_path:
        #사용자가 경로를 입력한 경우 해당 경로를 후보에 추가한다.
        candidates.append(Path(requested_path))

    repo_dataset = Path(__file__).resolve().parent / "dataset"
    dataset_dirs = {
        "mvtec": ["MVTec", "mvtec"],
        "mvtec3d": ["MVTec-3D", "MVTec3D", "mvtec3d"],
        "visa": ["Visa", "visa"],
        "mpdd": ["MPDD", "mpdd"],
        "btad": ["BTAD", "btad"],
    }
    for dirname in dataset_dirs.get(dataset_name, [dataset_name]):
        candidates.append(repo_dataset / dirname)

    for candidate in candidates:
        if candidate.exists():
            #.exists메서드는 파이썬 Path 객체에서 해당 경로가 실제로 존재하는지 확인하는 메서드입니다.
            return str(candidate)

    tried = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Dataset root not found for {dataset_name}. Tried: {tried}")


def ensure_meta_json(dataset_name, dataset_dir):
    meta_path = Path(dataset_dir) / "meta.json"
    if meta_path.exists():
        return dataset_dir

    if dataset_name == "mvtec":
        from dataset.mvtec import MVTecSolver

        MVTecSolver(root=dataset_dir).run()
        return dataset_dir

    if dataset_name == "mvtec3d":
        from dataset.mvtec3d import MVTec3DSolver

        MVTec3DSolver(root=dataset_dir).run()
        return dataset_dir

    if dataset_name == "visa":
        from dataset.visa import VisASolver

        VisASolver(root=dataset_dir).run()
        return dataset_dir

    if dataset_name in ["mpdd", "btad"]:
        from dataset.generic_mvtec import MVTecStyleSolver

        solver = MVTecStyleSolver(root=dataset_dir, dataset_name=dataset_name)
        solver.run()
        return str(solver.root)

    raise FileNotFoundError(f"{meta_path} does not exist. Generate meta.json for {dataset_name} first.")


def format_category_name(cls_name, replace_underscore=True):
    if replace_underscore:
        return cls_name.replace("_", " ")
    return cls_name
#이건 clip전용으로 자연어 인식위해 언더스코어 제외하는것.

@torch.no_grad()
def build_visual_gallery(winclip, prompt_loader):
    galleries = None

    for items in tqdm(prompt_loader, desc="Building visual gallery", leave=False):
        imgs = winclip._to_clip_input(items["img"])
        visual_features = winclip.model.encode_image(imgs)

        if galleries is None:
            galleries = [[] for _ in range(len(winclip.model.scale_begin_indx))]

        for scale_index in range(len(winclip.model.scale_begin_indx)):
            begin = winclip.model.scale_begin_indx[scale_index]
            if scale_index == len(winclip.model.scale_begin_indx) - 1:
                scale_features = visual_features[begin:]
            else:
                end = winclip.model.scale_begin_indx[scale_index + 1]
                scale_features = visual_features[begin:end]
            galleries[scale_index].append(torch.cat(scale_features, dim=0))

    if galleries is None:
        winclip.model.visual_gallery = None
        return

    winclip.model.visual_gallery = [torch.cat(scale_gallery, dim=0) for scale_gallery in galleries]


def prepare_class_model(winclip, cls_name, prompt_loader, args):
    category = format_category_name(cls_name, args.replace_underscore)
    winclip.model.build_text_feature_gallery(category)

    if args.k_shots > 0 and args.use_visual_gallery:
        build_visual_gallery(winclip, prompt_loader)
    else:
        winclip.model.visual_gallery = None

    winclip.category = category
    winclip._is_fit = True


def limit_test_samples_per_class(dataset, max_samples_per_class):
    if max_samples_per_class is None or max_samples_per_class <= 0:
        return

    limited = []
    for cls_name in dataset.obj_list:
        cls_items = [item for item in dataset.data_all if item["cls_name"] == cls_name]
        normal_items = [item for item in cls_items if item["anomaly"] == 0]
        anomaly_items = [item for item in cls_items if item["anomaly"] != 0]

        selected = []
        if normal_items:
            selected.append(normal_items[0])
        if anomaly_items and len(selected) < max_samples_per_class:
            selected.append(anomaly_items[0])

        for item in cls_items:
            if len(selected) >= max_samples_per_class:
                break
            if item not in selected:
                selected.append(item)

        limited.extend(selected)

    dataset.data_all = limited
    dataset.length = len(limited)


def normalize_class_names(class_name):
    if class_name is None:
        return None
    if isinstance(class_name, str):
        raw_names = [class_name]
    else:
        raw_names = class_name

    class_names = []
    for raw_name in raw_names:
        for name in str(raw_name).split(","):
            name = name.strip()
            if name:
                class_names.append(name)
    return class_names or None


def filter_results_by_class(results_eval, cls_name):
    cls_names = results_eval["cls_names"]
    mask = cls_names == cls_name
    filtered = {}
    for key, value in results_eval.items():
        if key in ["cls_names", "query_paths", "sample_ids"]:
            filtered[key] = value[mask]
        else:
            filtered[key] = value[torch.from_numpy(mask).to(value.device)]
    return filtered


def test(args):
    args.class_name = normalize_class_names(args.class_name)
    dataset_dir = resolve_dataset_path(args.dataset, args.test_data_path)
    dataset_dir = ensure_meta_json(args.dataset, dataset_dir)

    save_path = resolve_corruption_save_path(
        args.save_path,
        args.dataset,
        args.class_name,
        args.corruption,
        args.corruption_severity,
    )
    Path(save_path).mkdir(parents=True, exist_ok=True)

    log_file = f"{args.dataset}_{args.seed}seed_{args.k_shots}shot_winclip_test_log.txt"
    logger = get_logger(save_path, log_file)
    logger.info("\n")
    logger.info(args)

    device = select_device(args.device)
    logger.info(f"Using device: {device}")

    preprocess, target_transform = get_transform(image_size=args.image_size)

    all_test_data = Dataset(
        root=dataset_dir,
        transform=preprocess,
        target_transform=target_transform,
        dataset_name=args.dataset,
        k_shots=args.k_shots,
        save_dir=save_path,
        mode="test",
        seed=args.seed,
        class_name=args.class_name,
        corruption=args.corruption,
        corruption_severity=args.corruption_severity,
        sample_csv=args.sample_csv,
    )
    obj_list = all_test_data.obj_list
    sample_level = args.dataset in ["Real-IAD-Variety", "RealIAD"]

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
        image_score_mode=args.image_score_mode,
        image_score_topk_ratio=args.image_score_topk_ratio,
    )

    if args.evaluator_device == "auto":
        evaluator_device = "cpu" if device.type in ["cpu", "mps"] else device
    else:
        evaluator_device = args.evaluator_device
    cpu_eva = evaluator_device == "cpu"
    evaluator = Evaluator(evaluator_device, metrics=args.eval_metrics, sample_level=sample_level)
    selected_heatmaps = SelectedHeatmapSaver(save_path, args.dataset, args.image_size, args.heatmap_topk) if args.save_selected_heatmaps else None

    sample_ids, gt_masks, pr_masks, cls_names, gt_anomalys, pr_anomalys, query_paths = [], [], [], [], [], [], []

    for cls_name in tqdm(obj_list, desc="Classes"):
        prompt_data = PromptDataset(
            root=dataset_dir,
            transform=preprocess,
            target_transform=target_transform,
            dataset_name=args.dataset,
            k_shots=args.k_shots,
            save_dir=save_path,
            mode="test",
            seed=args.seed,
            class_name=cls_name,
            corruption=args.corruption if args.corrupt_prompts else None,
            corruption_severity=args.corruption_severity if args.corrupt_prompts else 0,
        )
        test_data = Dataset(
            root=dataset_dir,
            transform=preprocess,
            target_transform=target_transform,
            dataset_name=args.dataset,
            k_shots=args.k_shots,
            save_dir=save_path,
            mode="test",
            seed=args.seed,
            class_name=cls_name,
            corruption=args.corruption,
            corruption_severity=args.corruption_severity,
            sample_csv=args.sample_csv,
        )
        limit_test_samples_per_class(test_data, args.max_test_samples_per_class)

        prompt_loader = torch.utils.data.DataLoader(prompt_data, batch_size=args.batch_size, shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

        prepare_class_model(winclip, cls_name, prompt_loader, args)

        for items in tqdm(test_loader, desc=f"Testing {cls_name}", leave=False):
            query_image = items["img"]
            query_path = items["img_path"]
            batch_cls_names = items["cls_name"]
            sample_id = items["sample_id"]

            gt_anomaly = items["anomaly"].to(device)
            gt_mask = items["img_mask"][:, 0]
            gt_mask[gt_mask > 0.5], gt_mask[gt_mask <= 0.5] = 1, 0
            gt_mask = gt_mask.to(device)

            with torch.no_grad():
                image_anomaly_pred, heatmaps = winclip.predict_batch(query_image)

            pixel_anomaly_map = torch.stack(heatmaps, dim=0).to(device)
            if args.sigma > 0:
                pixel_anomaly_map = torch.stack(
                    [torch.from_numpy(gaussian_filter(i, sigma=args.sigma)).float() for i in pixel_anomaly_map.cpu()],
                    dim=0,
                ).to(device)

            pixel_anomaly_map = torch.nan_to_num(pixel_anomaly_map, nan=0.0, posinf=0.0, neginf=0.0)
            image_anomaly_pred = torch.nan_to_num(image_anomaly_pred.to(device), nan=0.0, posinf=0.0, neginf=0.0)

            if selected_heatmaps is not None:
                selected_heatmaps.update(
                    query_path,
                    query_image.detach().cpu(),
                    pixel_anomaly_map.detach().cpu(),
                    gt_mask.detach().cpu(),
                    batch_cls_names,
                )

            if args.save_heatmap:
                visualizer(
                    query_path,
                    query_image.detach().cpu(),
                    pixel_anomaly_map.detach().cpu(),
                    (args.image_size, args.image_size),
                    save_path,
                    batch_cls_names,
                    gt_mask.detach().cpu(),
                )

            if args.dataset in ["Real-IAD-Variety", "RealIAD", "bmad-medical"]:
                resize_mask = 256
                pixel_anomaly_map = F.interpolate(
                    pixel_anomaly_map[:, None],
                    size=(resize_mask, resize_mask),
                    mode="bilinear",
                    align_corners=False,
                )[:, 0]
                gt_mask = F.interpolate(gt_mask[:, None], size=(resize_mask, resize_mask), mode="nearest")
                gt_mask = gt_mask.bool().int()

            sample_ids.append(np.array(sample_id))
            cls_names.append(np.array(batch_cls_names))
            query_paths.append(np.array(query_path))

            if cpu_eva:
                gt_masks.append(gt_mask.int().cpu())
                pr_masks.append(pixel_anomaly_map.cpu())
                gt_anomalys.append(gt_anomaly.int().cpu())
                pr_anomalys.append(image_anomaly_pred.cpu())
            else:
                gt_masks.append(gt_mask.int())
                pr_masks.append(pixel_anomaly_map)
                gt_anomalys.append(gt_anomaly.int())
                pr_anomalys.append(image_anomaly_pred)

    results_eval = {
        "sample_ids": sample_ids,
        "gt_masks": gt_masks,
        "pr_masks": pr_masks,
        "cls_names": cls_names,
        "gt_anomalys": gt_anomalys,
        "pr_anomalys": pr_anomalys,
        "query_paths": query_paths,
    }
    results_eval = {
        key: np.concatenate(value, axis=0) if key in ["cls_names", "query_paths", "sample_ids"] else torch.cat(value, dim=0)
        for key, value in results_eval.items()
    }
    if args.save_sample_scores:
        sample_score_root = Path(save_path)
        score_paths = []
        for cls_name in obj_list:
            safe_class_name = str(cls_name).replace("/", "_").replace("\\", "_")
            class_results_eval = filter_results_by_class(results_eval, cls_name)
            scores_path = save_sample_scores(sample_score_root / f"{safe_class_name}.csv", args.dataset, args.seed, args.k_shots, class_results_eval)
            score_paths.append(scores_path)
        logger.info(f"Saved sample scores to: {', '.join(str(path) for path in score_paths)}")

    if args.save_difficulty_inputs:
        difficulty_dir = Path(save_path) / "difficulty_inputs" / args.dataset
        difficulty_dir.mkdir(parents=True, exist_ok=True)

        target_class = "_".join(args.class_name) if args.class_name is not None else "all"
        difficulty_path = difficulty_dir / f"{target_class}_predictions.npz"

        np.savez_compressed(
            difficulty_path,
            sample_ids=results_eval["sample_ids"],
            cls_names=results_eval["cls_names"],
            query_paths=results_eval["query_paths"],
            gt_anomalys=results_eval["gt_anomalys"].detach().cpu().numpy(),
            pr_anomalys=results_eval["pr_anomalys"].detach().cpu().numpy(),
            gt_masks=results_eval["gt_masks"].detach().cpu().numpy(),
            pr_masks=results_eval["pr_masks"].detach().cpu().numpy(),
        )

        logger.info(f"Saved difficulty inputs to: {difficulty_path}")
        if args.predictions_only:
            logger.info("Predictions-only mode enabled; skipping metric evaluation.")
            return

    msg = {}
    class_metric_rows = []
    for idx, cls_name in enumerate(tqdm(obj_list, desc="Evaluating")):
        metric_results = evaluator.run(results_eval, cls_name, logger)
        class_metric_rows.append({"class": cls_name, **metric_results})
        msg["Name"] = msg.get("Name", [])
        msg["Name"].append(cls_name)
        avg_act = len(obj_list) > 1 and idx == len(obj_list) - 1
        if avg_act:
            msg["Name"].append("Avg")

        for metric in args.eval_metrics:
            metric_result = metric_results[metric] * 100
            msg[metric] = msg.get(metric, [])
            msg[metric].append(metric_result)

            if avg_act:
                msg[metric].append(sum(msg[metric]) / len(msg[metric]))

    tab = tabulate(msg, headers="keys", tablefmt="pipe", floatfmt=".1f", numalign="center", stralign="center")
    logger.info("\n" + tab)
    metrics_path = save_class_metrics(save_path, args.dataset, args.seed, args.k_shots, class_metric_rows)
    logger.info(f"Saved class metrics to: {metrics_path}")

    if selected_heatmaps is not None:
        saved_count = selected_heatmaps.finalize()
        logger.info(f"Saved selected heatmaps ({saved_count} files) to: {selected_heatmaps.root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("WinCLIP", add_help=True)
    parser.add_argument("--test_data_path", type=str, default=None, help="path to test dataset")
    parser.add_argument("--save_path", type=str, default="./results/winclip", help="path to save results")
    parser.add_argument("--dataset", type=str, default="mvtec")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="test dataloader workers")
    parser.add_argument("--image_size", type=int, default=240, help="WinCLIP input and output size")
    parser.add_argument("--seed", type=int, default=10, help="random seed")
    parser.add_argument("--sigma", type=int, default=4, help="Gaussian smoothing sigma for anomaly maps")
    parser.add_argument("--k_shots", type=int, default=1, help="how many normal samples")
    parser.add_argument("--class_name", type=str, nargs="+", help="class name for a special dataset, for example, bottle in MVTec")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"], help="device to run inference on")
    parser.add_argument("--eval_metrics", type=str, nargs="+", default=["I-AUROC", "I-AP", "I-F1max", "P-AUROC", "P-AP", "P-F1max", "P-AUPRO"], help="evaluation metrics")
    parser.add_argument("--evaluator_device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="device to run evaluation metrics on")
    parser.add_argument("--backbone", type=str, default="ViT-B-16-plus-240", help="open_clip backbone")
    parser.add_argument("--pretrained_dataset", type=str, default="laion400m_e32", help="open_clip pretrained tag")
    parser.add_argument("--scales", type=int, nargs="+", default=[2, 3], help="WinCLIP window scales")
    parser.add_argument("--precision", type=str, default=None, choices=[None, "fp32", "fp16"], help="model precision")
    parser.add_argument("--fusion_version", type=str, default="textual_visual", choices=["textual", "visual", "textual_visual"], help="WinCLIP fusion mode")
    parser.add_argument("--image_score_mode", type=str, default="global", choices=["global", "max", "mean", "topk_mean"], help="image-level anomaly score mode")
    parser.add_argument("--image_score_topk_ratio", type=float, default=0.01, help="top-k ratio for image-level score")
    parser.add_argument("--use_visual_gallery", action=argparse.BooleanOptionalAction, default=True, help="enable few-shot visual gallery when k_shots > 0")
    parser.add_argument("--replace_underscore", action=argparse.BooleanOptionalAction, default=True, help="replace underscores in class names for text prompts")
    parser.add_argument("--save_heatmap", action="store_true", help="save anomaly heatmap overlays during testing")
    parser.add_argument("--save_difficulty_inputs", action="store_true", help="save per-sample predictions for difficulty split")
    parser.add_argument("--save_sample_scores", action="store_true", help="save per-image anomaly scores for distribution analysis")
    parser.add_argument("--predictions_only", "--predictions-only", action="store_true", help="stop after saving per-sample predictions")
    parser.add_argument("--save_selected_heatmaps", "--save-selected-heatmaps", action=argparse.BooleanOptionalAction, default=True, help="save top/bottom heatmap examples by per-image pixel AUROC")
    parser.add_argument("--heatmap_topk", type=int, default=5, help="number of high/low heatmaps to save per class")
    parser.add_argument("--max_test_samples_per_class", type=int, default=None, help="limit test samples per class for quick debugging")
    parser.add_argument("--corruption", type=str, default=None, choices=[None, "gaussian_noise", "motion_blur", "brightness", "rotation", "translation", "contrast", "jpeg_compression", "downsample_upsample"], help="optional corruption applied to test images")
    parser.add_argument("--corruption_severity", type=int, default=0, choices=[0, 1, 2, 3], help="corruption severity; 0 disables corruption")
    parser.add_argument("--corrupt_prompts", action="store_true", help="also apply corruption to few-shot prompt images")
    parser.add_argument("--sample_csv", type=str, default=None, help="optional CSV with dataset and sample_key columns to filter test samples")
    args = parser.parse_args()

    if args.corruption is not None and args.corruption_severity == 0:
        args.corruption_severity = 1

    print(args)
    setup_seed(args.seed)
    test(args)
