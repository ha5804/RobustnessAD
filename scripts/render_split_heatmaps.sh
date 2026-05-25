#!/usr/bin/env bash
set -euo pipefail

seed="${SEED:-10}"
shot="${SHOT:-0}"
topk="${TOPK:-3}"
results_root="${RESULTS_ROOT:-./results}"
split_root="${SPLIT_ROOT:-${results_root}/difficulty_splits}"
heatmap_root="${HEATMAP_ROOT:-${results_root}/split_heatmaps}"
models="${MODELS:-adaptclip winclip anomalyclip}"
datasets="${DATASETS:-mvtec visa mpdd btad}"

for model in ${models}; do
    for dataset in ${datasets}; do
        npz_path="${results_root}/${model}/difficulty_inputs/${dataset}/all_predictions.npz"
        split_dir="${split_root}/${model}/${dataset}/${seed}seed_${shot}shot"
        output_dir="${heatmap_root}/${model}/${dataset}/${seed}seed_${shot}shot"

        if [[ ! -f "${npz_path}" ]]; then
            echo "Skip missing npz: ${npz_path}"
            continue
        fi

        if [[ ! -d "${split_dir}" ]]; then
            echo "Skip missing split dir: ${split_dir}"
            continue
        fi

        echo "==> Render split heatmaps: model=${model}, dataset=${dataset}, topk=${topk}"
        python tools/render_split_heatmaps.py \
            --npz_path "${npz_path}" \
            --split_dir "${split_dir}" \
            --output_dir "${output_dir}" \
            --topk "${topk}"
    done
done

echo "Split heatmap rendering done."
