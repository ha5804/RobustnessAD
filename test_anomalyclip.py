"""Testing script for AnomalyCLIP in the shared difficulty pipeline."""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from tabulate import tabulate
from tqdm import tqdm

from dataset import Dataset
from models import anomalycliplib
from models.adaptcliplib.adaptclip import tokenize
from tools import (
    Evaluator,
    SelectedHeatmapSaver,
    get_logger,
    get_transform,
    resolve_corruption_save_path,
    save_class_metrics,
    setup_seed,
    visualizer,
)


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
    candidates = []
    if requested_path:
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
            return str(candidate)

    tried = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Dataset root not found for {dataset_name}. Tried: {tried}")


def ensure_meta_json(dataset_name, dataset_dir):
    meta_path = Path(dataset_dir) / "meta.json"
    if meta_path.exists():
        return

    if dataset_name == "mvtec":
        from dataset.mvtec import MVTecSolver

        MVTecSolver(root=dataset_dir).run()
        return

    if dataset_name == "mvtec3d":
        from dataset.mvtec3d import MVTec3DSolver

        MVTec3DSolver(root=dataset_dir).run()
        return

    if dataset_name == "visa":
        from dataset.visa import VisASolver

        VisASolver(root=dataset_dir).run()
        return

    if dataset_name in ["mpdd", "btad"]:
        from dataset.generic_mvtec import MVTecStyleSolver

        MVTecStyleSolver(root=dataset_dir, dataset_name=dataset_name).run()
        return

    raise FileNotFoundError(f"{meta_path} does not exist. Generate meta.json for {dataset_name} first.")


class AnomalyCLIPPromptLearner(nn.Module):
    def __init__(self, clip_model, prompt_length=12, depth=9, text_embedding_length=4):
        super().__init__()
        ctx_dim = clip_model.ln_final.weight.shape[0]
        suffix_len = clip_model.context_length - prompt_length - 1

        self.ctx_pos = nn.Parameter(torch.empty(1, 1, prompt_length, ctx_dim))
        self.ctx_neg = nn.Parameter(torch.empty(1, 1, prompt_length, ctx_dim))
        nn.init.normal_(self.ctx_pos, std=0.02)
        nn.init.normal_(self.ctx_neg, std=0.02)

        self.register_buffer("token_prefix_pos", torch.zeros(1, 1, 1, ctx_dim))
        self.register_buffer("token_suffix_pos", torch.zeros(1, 1, suffix_len, ctx_dim))
        self.register_buffer("token_prefix_neg", torch.zeros(1, 1, 1, ctx_dim))
        self.register_buffer("token_suffix_neg", torch.zeros(1, 1, suffix_len, ctx_dim))
        self.register_buffer("tokenized_prompts_pos", torch.zeros(1, 1, clip_model.context_length, dtype=torch.int))
        self.register_buffer("tokenized_prompts_neg", torch.zeros(1, 1, clip_model.context_length, dtype=torch.int))

        self.compound_prompts_text = nn.ParameterList(
            [nn.Parameter(torch.empty(text_embedding_length, ctx_dim)) for _ in range(depth - 1)]
        )
        for prompt in self.compound_prompts_text:
            nn.init.normal_(prompt, std=0.02)

        self.compound_prompt_projections = nn.ModuleList(
            [nn.Linear(ctx_dim, 896) for _ in range(depth - 1)]
        )

    def forward(self):
        prompts_pos = torch.cat([self.token_prefix_pos, self.ctx_pos, self.token_suffix_pos], dim=2)
        prompts_neg = torch.cat([self.token_prefix_neg, self.ctx_neg, self.token_suffix_neg], dim=2)
        prompts = torch.cat([prompts_pos.reshape(-1, prompts_pos.shape[-2], prompts_pos.shape[-1]),
                             prompts_neg.reshape(-1, prompts_neg.shape[-2], prompts_neg.shape[-1])], dim=0)

        tokenized_prompts = torch.cat(
            [
                self.tokenized_prompts_pos.reshape(-1, self.tokenized_prompts_pos.shape[-1]),
                self.tokenized_prompts_neg.reshape(-1, self.tokenized_prompts_neg.shape[-1]),
            ],
            dim=0,
        )
        return prompts, tokenized_prompts, list(self.compound_prompts_text)


def load_optional_checkpoint(model, checkpoint_path, logger):
    if checkpoint_path is None:
        return

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict):
        for key in ["state_dict", "model", "model_state_dict"]:
            if key in checkpoint:
                checkpoint = checkpoint[key]
                break

    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"Unsupported checkpoint format: {checkpoint_path}")

    state_dict = {k[7:] if k.startswith("module.") else k: v for k, v in checkpoint.items()}
    incompatible = model.load_state_dict(state_dict, strict=False)
    logger.info(f"Loaded optional AnomalyCLIP checkpoint: {checkpoint_path}")
    logger.info(f"Missing keys: {len(incompatible.missing_keys)}, unexpected keys: {len(incompatible.unexpected_keys)}")


def build_learned_text_features(model, checkpoint_path, args, device, logger):
    if checkpoint_path is None:
        return None

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "prompt_learner" not in checkpoint:
        load_optional_checkpoint(model, checkpoint_path, logger)
        return None

    prompt_learner = AnomalyCLIPPromptLearner(
        model,
        prompt_length=args.n_ctx,
        depth=args.depth,
        text_embedding_length=args.t_n_ctx,
    )
    prompt_learner.load_state_dict(checkpoint["prompt_learner"], strict=True)
    prompt_learner.to(device)
    prompt_learner.eval()

    prompts, tokenized_prompts, compound_prompts_text = prompt_learner()
    text_features = model.encode_text_learn(prompts, tokenized_prompts, compound_prompts_text).float()
    text_features = torch.stack(torch.chunk(text_features, dim=0, chunks=2), dim=1)
    text_features = F.normalize(text_features, dim=-1)
    logger.info(f"Loaded AnomalyCLIP prompt learner from: {checkpoint_path}")
    return text_features


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


def prompt_phrases(category):
    normal_states = [
        "normal {}",
        "flawless {}",
        "perfect {}",
        "undamaged {}",
    ]
    anomaly_states = [
        "damaged {}",
        "defective {}",
        "anomalous {}",
        "abnormal {}",
    ]
    templates = [
        "a photo of a {}.",
        "a close-up photo of a {}.",
        "a photo of the {} for anomaly detection.",
    ]

    normal = [template.format(state.format(category)) for template in templates for state in normal_states]
    anomaly = [template.format(state.format(category)) for template in templates for state in anomaly_states]
    return normal, anomaly


@torch.no_grad()
def build_text_features(model, category, device):
    normal_prompts, anomaly_prompts = prompt_phrases(category)
    normal_tokens = tokenize(normal_prompts).to(device)
    anomaly_tokens = tokenize(anomaly_prompts).to(device)

    normal_features = model.encode_text(normal_tokens).float()
    anomaly_features = model.encode_text(anomaly_tokens).float()
    normal_features = F.normalize(normal_features, dim=-1).mean(dim=0, keepdim=True)
    anomaly_features = F.normalize(anomaly_features, dim=-1).mean(dim=0, keepdim=True)
    return F.normalize(torch.cat([normal_features, anomaly_features], dim=0), dim=-1)


def category_for_class(cls_name, object_agnostic):
    if object_agnostic:
        return "object"
    return str(cls_name).replace("_", " ")


def local_map_from_patches(patch_features, text_features, image_size, temperature, feature_map_layer):
    maps = []
    start_layer = feature_map_layer[0] if feature_map_layer else 0
    for idx, patch in enumerate(patch_features):
        if idx < start_layer:
            continue
        patch = patch[:, 1:, :]
        patch = F.normalize(patch.float(), dim=-1)
        similarity = (patch @ text_features.t() / temperature).softmax(dim=-1)[..., 1]
        side = int(similarity.shape[1] ** 0.5)
        similarity = similarity[:, : side * side].reshape(similarity.shape[0], 1, side, side)
        similarity = F.interpolate(similarity, size=(image_size, image_size), mode="bilinear", align_corners=False)
        maps.append(similarity[:, 0])
    if not maps:
        raise RuntimeError("No patch feature maps selected. Check --feature_map_layer.")
    return torch.stack(maps, dim=0).sum(dim=0)


@torch.no_grad()
def predict_batch(model, images, cls_names, text_cache, learned_text_features, args, device):
    image_features, patch_features = model.encode_image(
        images,
        args.features_list,
        DPAM_layer=args.dpam_layer,
        ffn=args.ffn,
    )
    image_features = F.normalize(image_features.float(), dim=-1)

    if learned_text_features is not None:
        text_features = learned_text_features[0]
        logits = (image_features @ text_features.t() / args.temperature).softmax(dim=-1)
        image_scores = logits[:, 1]
        pixel_maps = local_map_from_patches(patch_features, text_features, args.image_size, args.temperature, args.feature_map_layer)
        return image_scores, pixel_maps

    if args.object_agnostic or len(set(cls_names)) == 1:
        category = category_for_class(cls_names[0], args.object_agnostic)
        text_features = text_cache.setdefault(category, build_text_features(model, category, device))
        logits = (image_features @ text_features.t() * args.logit_scale).softmax(dim=-1)
        image_scores = logits[:, 1]
        pixel_maps = local_map_from_patches(patch_features, text_features, args.image_size, args.temperature, args.feature_map_layer)
        return image_scores, pixel_maps

    image_scores = []
    pixel_maps = []
    for idx, cls_name in enumerate(cls_names):
        category = category_for_class(cls_name, args.object_agnostic)
        text_features = text_cache.setdefault(category, build_text_features(model, category, device))
        one_image_feature = image_features[idx : idx + 1]
        one_patch_features = [patch[idx : idx + 1] for patch in patch_features]
        logits = (one_image_feature @ text_features.t() * args.logit_scale).softmax(dim=-1)
        image_scores.append(logits[:, 1])
        pixel_maps.append(local_map_from_patches(one_patch_features, text_features, args.image_size, args.temperature, args.feature_map_layer))

    return torch.cat(image_scores, dim=0), torch.cat(pixel_maps, dim=0)


def test(args):
    dataset_dir = resolve_dataset_path(args.dataset, args.test_data_path)
    ensure_meta_json(args.dataset, dataset_dir)

    save_path = resolve_corruption_save_path(
        args.save_path,
        args.dataset,
        args.class_name,
        args.corruption,
        args.corruption_severity,
    )
    Path(save_path).mkdir(parents=True, exist_ok=True)

    log_file = f"{args.dataset}_{args.seed}seed_{args.k_shots}shot_anomalyclip_test_log.txt"
    logger = get_logger(save_path, log_file)
    logger.info("\n")
    logger.info(args)

    device = select_device(args.device)
    logger.info(f"Using device: {device}")

    preprocess, target_transform = get_transform(image_size=args.image_size)
    test_data = Dataset(
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
    )
    limit_test_samples_per_class(test_data, args.max_test_samples_per_class)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    sample_level = args.dataset in ["Real-IAD-Variety", "RealIAD"]

    design_details = None
    if args.checkpoint_path is not None:
        design_details = {
            "Prompt_length": args.n_ctx,
            "learnabel_text_embedding_depth": args.depth,
            "learnabel_text_embedding_length": args.t_n_ctx,
        }

    model, _ = anomalycliplib.load(args.pretrained_model, device=device, design_details=design_details)
    if hasattr(model.visual, "DAPM_replace") and args.dpam_layer is not None:
        model.visual.DAPM_replace(DPAM_layer=args.dpam_layer)
    model.to(device)
    model.eval()
    learned_text_features = build_learned_text_features(model, args.checkpoint_path, args, device, logger)

    if args.evaluator_device == "auto":
        evaluator_device = "cpu" if device.type in ["cpu", "mps"] else device
    else:
        evaluator_device = args.evaluator_device
    cpu_eva = evaluator_device == "cpu"
    evaluator = Evaluator(evaluator_device, metrics=args.eval_metrics, sample_level=sample_level)
    selected_heatmaps = SelectedHeatmapSaver(save_path, args.dataset, args.image_size, args.heatmap_topk) if args.save_selected_heatmaps else None

    text_cache = {}
    sample_ids, gt_masks, pr_masks, cls_names, gt_anomalys, pr_anomalys, query_paths = [], [], [], [], [], [], []

    for items in tqdm(test_loader, desc="Testing"):
        query_image = items["img"].to(device)
        query_path = items["img_path"]
        batch_cls_names = list(items["cls_name"])
        sample_id = items["sample_id"]

        gt_anomaly = items["anomaly"].to(device)
        gt_mask = items["img_mask"][:, 0]
        gt_mask[gt_mask > 0.5], gt_mask[gt_mask <= 0.5] = 1, 0
        gt_mask = gt_mask.to(device)

        image_anomaly_pred, pixel_anomaly_map = predict_batch(model, query_image, batch_cls_names, text_cache, learned_text_features, args, device)

        if args.sigma > 0:
            pixel_anomaly_map = torch.stack(
                [torch.from_numpy(gaussian_filter(i, sigma=args.sigma)).float() for i in pixel_anomaly_map.cpu()],
                dim=0,
            ).to(device)

        pixel_anomaly_map = torch.nan_to_num(pixel_anomaly_map, nan=0.0, posinf=0.0, neginf=0.0)
        image_anomaly_pred = torch.nan_to_num(image_anomaly_pred, nan=0.0, posinf=0.0, neginf=0.0)

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

    if args.save_difficulty_inputs:
        difficulty_dir = Path(save_path) / "difficulty_inputs" / args.dataset
        difficulty_dir.mkdir(parents=True, exist_ok=True)
        target_class = args.class_name if args.class_name is not None else "all"
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
    for idx, cls_name in enumerate(tqdm(test_data.obj_list, desc="Evaluating")):
        metric_results = evaluator.run(results_eval, cls_name, logger)
        class_metric_rows.append(
            {
                "class": cls_name,
                "image_auroc": metric_results.get("I-AUROC", ""),
                "pixel_auroc": metric_results.get("P-AUROC", ""),
                "p_aupr": metric_results.get("P-AP", ""),
            }
        )
        msg["Name"] = msg.get("Name", [])
        msg["Name"].append(cls_name)
        avg_act = len(test_data.obj_list) > 1 and idx == len(test_data.obj_list) - 1
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
    parser = argparse.ArgumentParser("AnomalyCLIP", add_help=True)
    parser.add_argument("--test_data_path", type=str, default=None)
    parser.add_argument("--save_path", type=str, default="./results/anomalyclip")
    parser.add_argument("--dataset", type=str, default="mvtec")
    parser.add_argument("--pretrained_model", type=str, default="ViT-L/14@336px")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="optional AnomalyCLIP checkpoint")
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24])
    parser.add_argument("--feature_map_layer", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=518)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--k_shots", type=int, default=0)
    parser.add_argument("--class_name", type=str)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--eval_metrics", type=str, nargs="+", default=["I-AUROC", "I-AP", "I-F1max", "P-AUROC", "P-AP", "P-F1max", "P-AUPRO"])
    parser.add_argument("--evaluator_device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="device to run evaluation metrics on")
    parser.add_argument("--dpam_layer", type=int, default=20)
    parser.add_argument("--depth", type=int, default=9)
    parser.add_argument("--n_ctx", type=int, default=12)
    parser.add_argument("--t_n_ctx", type=int, default=4)
    parser.add_argument("--ffn", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--logit_scale", type=float, default=100.0)
    parser.add_argument("--sigma", type=int, default=4)
    parser.add_argument("--object_agnostic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save_heatmap", action="store_true")
    parser.add_argument("--save_difficulty_inputs", action="store_true")
    parser.add_argument("--predictions_only", "--predictions-only", action="store_true", help="stop after saving per-sample predictions")
    parser.add_argument("--save_selected_heatmaps", "--save-selected-heatmaps", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--heatmap_topk", type=int, default=5)
    parser.add_argument("--max_test_samples_per_class", type=int, default=None)
    parser.add_argument("--corruption", type=str, default=None, choices=[None, "gaussian_noise", "motion_blur", "brightness", "contrast", "jpeg_compression", "downsample_upsample"])
    parser.add_argument("--corruption_severity", type=int, default=0, choices=[0, 1, 2, 3])
    args = parser.parse_args()

    if args.corruption is not None and args.corruption_severity == 0:
        args.corruption_severity = 1

    print(args)
    setup_seed(args.seed)
    test(args)
