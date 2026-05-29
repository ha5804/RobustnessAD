import csv
import heapq
from itertools import count
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from .visualization import apply_ad_scoremap, _denormalize_image, _to_numpy
from .utils import normalize


def resolve_corruption_save_path(save_path, dataset_name, class_name, corruption, severity):
    if corruption is None or severity == 0:
        return str(save_path)

    save_path = Path(save_path)
    class_label = class_name or "all_classes"
    class_label = str(class_label).replace("/", "_")
    corruption_label = f"{corruption}_s{severity}"
    if save_path.name == corruption_label:
        return str(save_path)
    return str(save_path / "corruption" / dataset_name / class_label / corruption_label)


def save_class_metrics(save_path, dataset_name, seed, k_shots, rows):
    output_path = Path(save_path) / f"class_metrics_{dataset_name}_{seed}seed_{k_shots}shot.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["class"]
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def save_sample_scores(save_path, dataset_name, seed, k_shots, results_eval):
    output_path = Path(save_path) / f"sample_scores_{dataset_name}_{seed}seed_{k_shots}shot.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sample_ids = results_eval["sample_ids"]
    cls_names = results_eval["cls_names"]
    query_paths = results_eval["query_paths"]
    gt_anomalys = results_eval["gt_anomalys"].detach().cpu().numpy()
    pr_anomalys = results_eval["pr_anomalys"].detach().cpu().numpy()

    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "sample_id",
                "class",
                "label",
                "image_score",
                "query_path",
            ],
        )
        writer.writeheader()
        for idx in range(len(sample_ids)):
            writer.writerow(
                {
                    "dataset": dataset_name,
                    "sample_id": sample_ids[idx],
                    "class": cls_names[idx],
                    "label": int(gt_anomalys[idx]),
                    "image_score": float(pr_anomalys[idx]),
                    "query_path": query_paths[idx],
                }
            )
    return output_path


class SelectedHeatmapSaver:
    def __init__(self, save_path, dataset_name, image_size, topk=5):
        self.root = Path(save_path) / "heatmap" / dataset_name
        self.image_size = (image_size, image_size) if isinstance(image_size, int) else tuple(image_size)
        self.topk = topk
        self._counter = count()
        self._high = {}
        self._low = {}

    def update(self, paths, images, anomaly_maps, masks, cls_names):
        images = _to_numpy(images)
        anomaly_maps = _to_numpy(anomaly_maps)
        masks = _to_numpy(masks)

        for idx, path in enumerate(paths):
            gt = masks[idx].squeeze().astype(np.uint8)
            if gt.max() == gt.min():
                continue

            score_map = anomaly_maps[idx].squeeze()
            try:
                score = float(roc_auc_score(gt.ravel(), score_map.ravel()))
            except ValueError:
                continue

            cls_name = str(cls_names[idx])
            item = {
                "score": score,
                "path": str(path),
                "image": images[idx].copy(),
                "map": score_map.copy(),
                "mask": gt.copy(),
            }
            serial = next(self._counter)

            high_heap = self._high.setdefault(cls_name, [])
            heapq.heappush(high_heap, (score, serial, item))
            if len(high_heap) > self.topk:
                heapq.heappop(high_heap)

            low_heap = self._low.setdefault(cls_name, [])
            heapq.heappush(low_heap, (-score, serial, item))
            if len(low_heap) > self.topk:
                heapq.heappop(low_heap)

    def finalize(self):
        saved = 0
        classes = sorted(set(self._high) | set(self._low))
        for cls_name in classes:
            high_items = [entry[2] for entry in self._high.get(cls_name, [])]
            low_items = [entry[2] for entry in self._low.get(cls_name, [])]
            high_items.sort(key=lambda item: item["score"], reverse=True)
            low_items.sort(key=lambda item: item["score"])
            saved += self._save_group(cls_name, "high", high_items)
            saved += self._save_group(cls_name, "low", low_items)
        return saved

    def _save_group(self, cls_name, group, items):
        group_dir = self.root / cls_name / group
        group_dir.mkdir(parents=True, exist_ok=True)
        for rank, item in enumerate(items, start=1):
            filename = Path(item["path"]).name
            output_name = f"{rank:02d}_auroc_{item['score']:.4f}_{filename}"
            self._write_heatmap(group_dir / output_name, item)
        return len(items)

    def _write_heatmap(self, output_path, item):
        import cv2

        width, height = self.image_size
        image = _denormalize_image(item["image"])
        image = cv2.resize(image, (width, height))
        score_map = normalize(item["map"])
        if score_map.shape[:2] != (height, width):
            score_map = cv2.resize(score_map, (width, height), interpolation=cv2.INTER_LINEAR)
        vis = apply_ad_scoremap(image, score_map)

        gt_mask = item["mask"].astype(np.uint8)
        if gt_mask.shape[:2] != (height, width):
            gt_mask = cv2.resize(gt_mask, (width, height), interpolation=cv2.INTER_NEAREST)
        contours, _ = cv2.findContours(gt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.polylines(vis, contours, isClosed=True, color=(0, 255, 0), thickness=2)

        vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_path), vis)
