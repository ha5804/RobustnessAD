import os

import cv2
import numpy as np

from models.adaptcliplib.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
from .utils import normalize


def _to_numpy(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu()
    if hasattr(x, "numpy"):
        x = x.numpy()
    return x


def _denormalize_image(image):
    image = _to_numpy(image).transpose(1, 2, 0)
    mean = np.asarray(OPENAI_DATASET_MEAN, dtype=np.float32)
    std = np.asarray(OPENAI_DATASET_STD, dtype=np.float32)
    image = image * std + mean
    image = np.clip(image, 0, 1)
    return (image * 255).astype(np.uint8)


def visualizer(pathes, ori_img, anomaly_map, img_size, save_path, cls_name, img_mask=None, max=None, min=None):
    for idx, path in enumerate(pathes):
        cls = path.split('/')[-2]
        filename = path.split('/')[-1]
        ori = _denormalize_image(ori_img[idx])
        vis = cv2.resize(ori, (img_size[0], img_size[1]))
        mask = normalize(_to_numpy(anomaly_map[idx]), max_value=max, min_value=min)
        if mask.shape[:2] != (img_size[1], img_size[0]):
            mask = cv2.resize(mask, (img_size[0], img_size[1]), interpolation=cv2.INTER_LINEAR)
        vis = apply_ad_scoremap(vis, mask)

        if img_mask is not None:
            gt_mask = _to_numpy(img_mask[idx]).squeeze().astype(np.uint8)
            if gt_mask.shape[:2] != (img_size[1], img_size[0]):
                gt_mask = cv2.resize(gt_mask, (img_size[0], img_size[1]), interpolation=cv2.INTER_NEAREST)
            contours, _ = cv2.findContours(gt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.polylines(vis, contours, isClosed=True, color=(0, 255, 0), thickness=2)

        vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)  # BGR
        save_vis = os.path.join(save_path, 'imgs', cls_name[idx], cls)
        if not os.path.exists(save_vis):
            os.makedirs(save_vis)
        cv2.imwrite(os.path.join(save_vis, filename), vis)


def apply_ad_scoremap(image, scoremap, alpha=0.5):
    np_image = np.asarray(image, dtype=float)
    scoremap = (scoremap * 255).astype(np.uint8)
    scoremap = cv2.applyColorMap(scoremap, cv2.COLORMAP_JET)
    scoremap = cv2.cvtColor(scoremap, cv2.COLOR_BGR2RGB)
    return (alpha * np_image + (1 - alpha) * scoremap).astype(np.uint8)
