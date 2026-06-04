# AdaptCLIP Codebase Map

## 0. How To Read This Repo

This repo is easiest to understand from the experiment entrypoints, not from the model files first.

Recommended reading order:

1. `scripts/run_full_corruption_benchmark.sh`
2. `test_adpatclip.py`, `test_anomalyclip.py`, `test_winclip.py`
3. `dataset/dataset.py`
4. `tools/result_saver.py`, `tools/metric.py`, `tools/effecient_metric.py`
5. `models/adaptcliplib/`, `models/anomalycliplib/`, `models/wincliplib/`
6. `tools/summarize_corruption_benchmark.py`, `tools/analyze_score_distribution.py`
7. `analysis/corruption_failure_analysis.ipynb`

Core question while reading:

```text
script -> dataset -> model -> anomaly score / heatmap -> metric -> saved results -> summary
```

---

## 1. Top-Level Structure

| Path | Role | Notes |
|---|---|---|
| `scripts/` | Shell entrypoints for experiments | Full benchmark, target hard-class analysis, rerun utilities |
| `test_adpatclip.py` | AdaptCLIP inference/evaluation entrypoint | Main official-ish AdaptCLIP evaluation pipeline |
| `test_anomalyclip.py` | AnomalyCLIP inference/evaluation entrypoint | Custom inference wrapper using AnomalyCLIP components |
| `test_winclip.py` | WinCLIP inference/evaluation entrypoint | Uses `models/wincliplib/winclip.py` |
| `train.py` | AdaptCLIP training script | Initializes CLIP + adapters and trains with anomaly losses |
| `dataset/` | Dataset loading and metadata generation | MVTec, VisA, BTAD, MPDD style loaders |
| `models/` | Model implementations | AdaptCLIP, AnomalyCLIP, WinCLIP |
| `tools/` | Shared utilities | Metrics, corruption, saving, visualization, summaries |
| `metrics/` | Alternative metric implementations | Lower-level metric modules |
| `analysis/` | Jupyter analysis notebooks | Failure analysis, corruption summaries, plotting |
| `results/` | Experiment outputs | Metrics, sample predictions, heatmaps, summaries |
| `overleaf_tables/` | Exported paper tables | CSV/TeX tables |
| `checkpoints/` | Model checkpoints | AdaptCLIP / AnomalyCLIP checkpoints |

---

## 2. Main Experiment Flows

### 2.1 Full Corruption Benchmark

Entrypoint:

```bash
scripts/run_full_corruption_benchmark.sh
```

Purpose:

```text
Run AdaptCLIP / WinCLIP / AnomalyCLIP on:
datasets = mvtec, visa, btad
splits = all, easy, normal, hard
conditions = clean, gaussian_noise_s3, motion_blur_s3, brightness_s3
```

Dependency flow:

```text
scripts/run_full_corruption_benchmark.sh
  -> test_adpatclip.py
  -> test_winclip.py
  -> test_anomalyclip.py
  -> tools/create_unified_difficulty.py
  -> tools/summarize_corruption_benchmark.py
```

Outputs:

```text
results/corruption_benchmark/{model}/{dataset}/{split}/{condition}/class_metrics_*.csv
results/corruption_benchmark/{model}/{dataset}/all/clean/difficulty_inputs/{dataset}/all_predictions.npz
results/corruption_benchmark/splits/{model}/unified/{seed}seed_{shot}shot/*.csv
results/corruption_benchmark/summaries/*.csv
```

Important notes:

```text
SKIP_EXISTING=1 skips already computed outputs.
Set SKIP_EXISTING=0 to overwrite/recompute.
Only split=all clean saves difficulty_inputs by default.
```

---

### 2.2 Target Hard-Class Failure Analysis

Entrypoint:

```bash
scripts/run_target_failure_analysis.sh
```

Purpose:

```text
Run selected hard target classes only.
Useful for score distribution, hard-class failure mode, selected heatmap visualization.
```

Target classes:

```text
mvtec: cable, pill, screw, transistor
visa: cashew, macaroni1, macaroni2, pcb2, pcb3
btad: 02
```

Dependency flow:

```text
scripts/run_target_failure_analysis.sh
  -> test_adpatclip.py / test_winclip.py / test_anomalyclip.py
  -> tools/analyze_score_distribution.py
```

Outputs:

```text
results/target_failure_analysis/{model}/{dataset}/{class}/{condition}/sample_scores_*.csv
results/target_failure_analysis/{model}/{dataset}/{class}/{condition}/class_metrics_*.csv
results/target_failure_analysis/{model}/{dataset}/{class}/{condition}/heatmap/{dataset}/...
results/target_failure_analysis/summaries/*.csv
results/target_failure_analysis/summaries/score_distribution_plots/*.png
```

Important environment variables:

```text
SAVE_HEATMAPS=1       # save selected high/low heatmaps
HEATMAP_TOPK=2        # high 2 + low 2 per class
SAVE_ALL_HEATMAPS=1   # save every sample heatmap overlay; storage-heavy
```

---

### 2.3 WinCLIP Fixed Rerun

Entrypoint:

```bash
scripts/rerun_all_winclip_results_runpod.sh
```

Purpose:

```text
Rerun only WinCLIP results after fixing window embedding.
Keeps AdaptCLIP / AnomalyCLIP outputs intact.
```

Default behavior:

```text
Deletes old results/corruption_benchmark/winclip
Deletes old results/corruption_benchmark/splits/winclip
Deletes old results/target_failure_analysis/winclip
Regenerates WinCLIP benchmark and target analysis
Regenerates summary CSVs
```

Typical RunPod command:

```bash
SAVE_HEATMAPS=1 HEATMAP_TOPK=2 CUDA_DEVICE=0 BATCH_SIZE=8 NUM_WORKERS=8 bash scripts/rerun_all_winclip_results_runpod.sh
```

---

## 3. Inference Entrypoints

### 3.1 `test_adpatclip.py`

Role:

```text
AdaptCLIP inference/evaluation entrypoint.
Loads AdaptCLIP checkpoint, builds prompt memory, predicts image anomaly score and pixel anomaly map, evaluates metrics, saves outputs.
```

Main dependencies:

```text
dataset.Dataset
dataset.PromptDataset
models.adaptcliplib
tools.get_transform
tools.Evaluator
tools.visualizer
tools.save_class_metrics
tools.save_sample_scores
tools.SelectedHeatmapSaver
```

Important functions:

| Function | Role |
|---|---|
| `select_device` | Resolve `auto/cuda/mps/cpu` device |
| `resolve_dataset_path` | Find dataset root |
| `ensure_meta_json` | Generate `meta.json` if missing |
| `limit_test_samples_per_class` | Quick debug sample limiter |
| `prompt_association` | Match image/patch features to class prompt memory |
| `build_prompt_memory` | Build few-shot prompt visual memory |
| `test` | Main evaluation loop |

Main output variables:

```text
image_anomaly_pred -> image-level anomaly score
pixel_anomaly_map  -> pixel-level heatmap
gt_mask            -> ground-truth mask
gt_anomaly         -> image-level label
```

---

### 3.2 `test_anomalyclip.py`

Role:

```text
AnomalyCLIP inference/evaluation entrypoint.
Loads AnomalyCLIP-like model, builds text features, computes image score and local patch map.
```

Main dependencies:

```text
dataset.Dataset
models.anomalycliplib
tools.get_transform
tools.Evaluator
tools.result_saver
```

Important functions/classes:

| Function / Class | Role |
|---|---|
| `AnomalyCLIPPromptLearner` | Learnable prompt context module |
| `load_optional_checkpoint` | Load checkpoint if provided |
| `build_learned_text_features` | Build learned text features from checkpoint |
| `prompt_phrases` | Normal/anomaly text prompts |
| `build_text_features` | Encode prompt phrases |
| `local_map_from_patches` | Convert patch features to anomaly map |
| `predict_batch` | Batch image score + heatmap prediction |
| `test` | Main evaluation loop |

---

### 3.3 `test_winclip.py`

Role:

```text
WinCLIP inference/evaluation entrypoint.
Builds WinCLIP text prompts, computes global image score and window-based pixel heatmap.
```

Main dependencies:

```text
dataset.Dataset
dataset.PromptDataset
models.wincliplib.winclip.WinCLIP
tools.get_transform
tools.Evaluator
tools.result_saver
```

Important functions:

| Function | Role |
|---|---|
| `select_device` | Resolve device |
| `resolve_dataset_path` | Find dataset root |
| `ensure_meta_json` | Generate metadata if missing |
| `format_category_name` | Convert class name to prompt object name |
| `build_visual_gallery` | Few-shot visual gallery for k-shot WinCLIP |
| `prepare_class_model` | Build text features and gallery per class |
| `limit_test_samples_per_class` | Small debug run helper |
| `test` | Main evaluation loop |

Important CLI args:

```text
--image_score_mode global      # image score from global CLIP two-class score
--fusion_version textual       # textual / visual / textual_visual map fusion
--scales 2 3                  # WinCLIP window scales
--image_size 240              # OpenCLIP ViT-B-16-plus-240 input
```

---

## 4. Dataset Layer

### 4.1 `dataset/dataset.py`

Role:

```text
Main runtime dataset loader.
Reads meta.json, loads image/mask, applies corruption, applies transforms, returns dict consumed by test scripts.
```

Important functions/classes:

| Function / Class | Role |
|---|---|
| `generate_class_info` | Static class lists per dataset |
| `_load_sample_keys` | Filter samples using split CSV |
| `Dataset` | Test/train sample loader |
| `PromptDataset` | Few-shot normal prompt image loader |

`Dataset.__getitem__` returns:

```text
img          transformed image tensor
img_mask     transformed GT mask
cls_name     class name
anomaly      0 normal / 1 anomaly
view_id      view id for multi-view datasets
sample_id    sample identifier
prompt_img   normal prompt images for train mode
img_path     absolute image path
cls_id       class id
```

Corruption is applied here:

```text
dataset.Dataset.__getitem__
  -> tools.corruption.apply_corruption
```

---

### 4.2 Metadata Generators

| File | Role |
|---|---|
| `dataset/mvtec.py` | Generate `meta.json` for MVTec |
| `dataset/visa.py` | Generate `meta.json` for VisA |
| `dataset/mvtec3d.py` | Generate `meta.json` for MVTec 3D-AD |
| `dataset/generic_mvtec.py` | Generate MVTec-style metadata for BTAD / MPDD |

Why metadata matters:

```text
All test scripts read dataset samples through meta.json.
If meta.json is missing, ensure_meta_json() tries to generate it.
```

---

## 5. Model Layer

### 5.1 `models/adaptcliplib/`

Role:

```text
AdaptCLIP implementation and CLIP loading utilities.
This is the most official-like model path in this repo.
```

Important files:

| File | Role |
|---|---|
| `adaptclip.py` | Main AdaptCLIP model, adapters, similarity maps |
| `clip.py` | CLIP backbone implementation |
| `model_load.py` | Load CLIP/AdaptCLIP weights |
| `build_model.py` | Build model from checkpoint state dict |
| `loss.py` | Focal loss, Dice loss, regularizers |
| `simple_tokenizer.py` | CLIP tokenizer |
| `transform.py` | Image transform helpers |

Important classes/functions in `adaptclip.py`:

| Name | Role |
|---|---|
| `AdaptCLIP` | Main CLIP model with anomaly-specific components |
| `VisionTransformer` | Visual encoder |
| `ResidualAttentionBlock_learnable_token` | Transformer block with learnable token behavior |
| `VisualAdapter` | Visual adapter module |
| `TextualAdapter` | Text prompt adapter |
| `PQAdapter` | Prompt-query adapter |
| `compute_similarity` | Image/text similarity |
| `get_similarity_map` | Patch similarity to spatial map |
| `calculate_visual_anomaly_score` | Visual-memory anomaly score |
| `fusion_fun` | Combine multi-layer maps |

---

### 5.2 `models/anomalycliplib/`

Role:

```text
AnomalyCLIP model implementation and CLIP-style loading utilities.
Used by test_anomalyclip.py.
```

Important files:

| File | Role |
|---|---|
| `anomalyclip.py` | Main AnomalyCLIP model |
| `clip.py` | CLIP backbone implementation |
| `model_load.py` | Model loading + similarity helpers |
| `build_model.py` | Build model from state dict |
| `simple_tokenizer.py` | CLIP tokenizer |
| `transform.py` | Image transform helpers |

Important classes/functions:

| Name | Role |
|---|---|
| `AnomalyCLIP` | Main model |
| `VisionTransformer` | Visual encoder |
| `ResidualAttentionBlock_learnable_token` | Prompt/context-aware transformer block |
| `compute_similarity` | Image/text similarity |
| `get_similarity_map` | Convert patch similarity to spatial heatmap |

---

### 5.3 `models/wincliplib/winclip.py`

Role:

```text
WinCLIP implementation using open_clip.
Builds CPE text prompts and window-level visual embeddings for pixel localization.
```

Main class dependency:

```text
WinCLIP
  -> WinClipAD
    -> OpenClipWinModel
      -> OpenClipWinVisual
```

Important classes/functions:

| Name | Role |
|---|---|
| `OpenClipWinVisual` | Extract window embeddings from OpenCLIP ViT |
| `OpenClipWinVisual.forward` | Build all window features across scales |
| `encode_pre_transformer_tokens` | Convert image to raw patch tokens + positional embeddings |
| `encode_window_features` | Select window tokens, run CLIP transformer, use class token output |
| `OpenClipWinModel` | Wrapper around OpenCLIP model |
| `WinClipAD` | WinCLIP anomaly detector module |
| `build_text_feature_gallery` | CPE normal/anomaly text prompt ensemble |
| `encode_image` | Window-level feature extraction |
| `encode_global_image` | Global image feature for image-level score |
| `calculate_textual_anomaly_score` | Text-window anomaly heatmap |
| `calculate_visual_anomaly_score` | Few-shot visual-gallery anomaly heatmap |
| `WinCLIP._predict_batch_map` | Heatmap prediction |
| `WinCLIP._score_images` | Image anomaly score, default global two-class score |
| `WinCLIP.predict_batch` | Public batch prediction API |

Important implementation note:

```text
WinCLIP localization should use:
selected window patch tokens
  + selected positional embeddings
  + class token
  -> CLIP visual transformer
  -> class token output as window embedding
```

---

## 6. Tools Layer

### 6.1 Shared Utilities

| File | Role |
|---|---|
| `tools/utils.py` | Seed, normalization, image/mask transforms |
| `tools/corruption.py` | Gaussian noise, motion blur, brightness, contrast, JPEG, downsample/upsample |
| `tools/logger.py` | File + stdout logger |
| `tools/visualization.py` | Heatmap overlay visualization |
| `tools/result_saver.py` | Metrics CSV, sample scores CSV, selected heatmaps |

Important functions:

| Function | Role |
|---|---|
| `setup_seed` | Reproducibility |
| `get_transform` | Image and mask transform |
| `apply_corruption` | Apply corruption to PIL image |
| `save_class_metrics` | Save class-level metric CSV |
| `save_sample_scores` | Save per-sample image anomaly score |
| `SelectedHeatmapSaver` | Save high/low pixel-AUROC heatmap examples |
| `visualizer` | Save all heatmap overlays if enabled |

---

### 6.2 Metrics

| File | Role |
|---|---|
| `tools/metric.py` | Sklearn/numpy evaluator |
| `tools/effecient_metric.py` | Faster evaluator used when import succeeds |
| `metrics/` | More modular metric implementations |

Main metrics:

```text
I-AUROC   image-level AUROC
I-AP      image-level average precision
I-F1max   best image-level F1
P-AUROC   pixel-level AUROC
P-AP      pixel-level average precision
P-F1max   best pixel-level F1
P-AUPRO   per-region overlap score
```

Important caution:

```text
P-AP and P-F1max are very sensitive to pixel imbalance and thresholding.
P-AUROC mainly measures pixel ranking quality.
```

---

### 6.3 Summary / Analysis Tools

| File | Role |
|---|---|
| `tools/create_difficulty.py` | Create difficulty splits from one model/dataset prediction npz |
| `tools/create_unified_difficulty.py` | Create unified easy/normal/hard split across datasets |
| `tools/summarize_corruption_benchmark.py` | Collect benchmark metrics and compute drops/summaries |
| `tools/analyze_score_distribution.py` | Target class image score distribution summaries/plots |
| `tools/render_split_heatmaps.py` | Render heatmaps from saved npz predictions |
| `tools/compare_difficulty.py` | Compare two difficulty split CSVs |
| `tools/parse_results_log.py` | Parse markdown metric tables from logs |
| `tools/compute_fs_mean_std.py` | Aggregate metric CSV mean/std |

---

## 7. Results Structure

### 7.1 Class Metrics

Saved by:

```text
tools.result_saver.save_class_metrics
```

Pattern:

```text
class_metrics_{dataset}_{seed}seed_{shot}shot.csv
```

Contains:

```text
class, I-AUROC, I-AP, I-F1max, P-AUROC, P-AP, P-F1max, P-AUPRO
```

---

### 7.2 Per-Sample Scores

Saved by:

```text
tools.result_saver.save_sample_scores
```

Pattern:

```text
sample_scores_{dataset}_{seed}seed_{shot}shot.csv
```

Contains:

```text
dataset
sample_id
class
label
image_score
query_path
```

Used by:

```text
tools/analyze_score_distribution.py
```

---

### 7.3 Difficulty Inputs

Saved when:

```text
--save_difficulty_inputs
```

Pattern:

```text
difficulty_inputs/{dataset}/all_predictions.npz
```

Contains:

```text
sample_ids
cls_names
query_paths
gt_anomalys
pr_anomalys
gt_masks
pr_masks
```

Meaning:

```text
pr_anomalys = image anomaly scores
pr_masks    = pixel anomaly maps / heatmaps
```

Used by:

```text
tools/create_unified_difficulty.py
tools/summarize_corruption_benchmark.py
tools/render_split_heatmaps.py
```

---

### 7.4 Selected Heatmaps

Saved by:

```text
SelectedHeatmapSaver
```

Enabled with:

```text
--save-selected-heatmaps
```

Path:

```text
{save_path}/heatmap/{dataset}/{class}/high/*.png
{save_path}/heatmap/{dataset}/{class}/low/*.png
```

Meaning:

```text
high = samples with high per-sample pixel AUROC
low  = samples with low per-sample pixel AUROC
```

---

## 8. Core Function Call Graphs

### 8.1 Generic Test Script Flow

```text
parse args
setup_seed
resolve_dataset_path
ensure_meta_json
get_transform
Dataset(...)
PromptDataset(...)
model init
for class:
  prepare class model / text features
  for batch:
    predict image anomaly score
    predict pixel anomaly map
    optional Gaussian smoothing
    optional save heatmap
    accumulate predictions
Evaluator.run
save_class_metrics
optional save_sample_scores
optional save_difficulty_inputs
optional selected_heatmaps.finalize
```

---

### 8.2 Corruption Flow

```text
test script receives:
  --corruption gaussian_noise
  --corruption_severity 3

Dataset.__getitem__
  -> PIL image load
  -> apply_corruption(image, corruption, severity)
  -> transform image
  -> return tensor
```

---

### 8.3 Difficulty Split Flow

```text
full benchmark clean all split
  -> save difficulty_inputs/*.npz
  -> tools/create_unified_difficulty.py
  -> easy.csv / normal.csv / hard.csv

corrupted split runs
  -> pass --sample_csv easy.csv / normal.csv / hard.csv
  -> evaluate same sample subset under corruption
```

---

## 9. Notion Study Template

Use this table for every file:

| Path | Role | Called By | Calls | Key Functions / Classes | Inputs | Outputs | Notes |
|---|---|---|---|---|---|---|---|
|  |  |  |  |  |  |  |  |

Use this table for every important function:

| Function | File | Inputs | Output | Side Effects | Why It Exists | Questions |
|---|---|---|---|---|---|---|
|  |  |  |  |  |  |  |

Use this table for experiments:

| Experiment | Script | Model | Dataset | Split | Corruption | Saves Metrics | Saves Scores | Saves Heatmaps | Notes |
|---|---|---|---|---|---|---|---|---|---|
|  |  |  |  |  |  |  |  |  |  |

---

## 10. First Review Checklist

Start with these checkboxes:

- [ ] Understand `scripts/run_full_corruption_benchmark.sh`
- [ ] Understand one complete test script, preferably `test_winclip.py`
- [ ] Understand `dataset.Dataset.__getitem__`
- [ ] Understand how corruption is applied
- [ ] Understand where image anomaly score is produced
- [ ] Understand where pixel anomaly map is produced
- [ ] Understand `Evaluator.run`
- [ ] Understand `save_class_metrics`, `save_sample_scores`, `difficulty_inputs`
- [ ] Understand selected heatmap saving
- [ ] Understand summary CSV generation
- [ ] Then study AdaptCLIP model internals
- [ ] Then study AnomalyCLIP model internals
- [ ] Then study WinCLIP model internals

---

## 11. Questions To Keep While Reading

Good code review questions:

```text
1. Is this file an entrypoint, library module, or analysis helper?
2. Does this function compute something, save something, or mutate model state?
3. Is this path used in training, inference, or post-analysis?
4. Which tensors are image-level and which are pixel-level?
5. Are scores normalized globally, per image, or not normalized?
6. Is this result class-level, sample-level, or pixel-level?
7. Does this function depend on clean-only predictions?
8. Does this function skip existing files?
9. Does this path save enough information to reproduce later analysis?
10. Is this implementation official, adapted, or custom?
```

