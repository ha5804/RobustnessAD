#!/usr/bin/env bash
set -euo pipefail

seed="${SEED:-10}"
shot="${SHOT:-0}"
results_root="${RESULTS_ROOT:-./results}"
split_root="${SPLIT_ROOT:-${results_root}/difficulty_splits}"
models="${MODELS:-adaptclip winclip anomalyclip}"
datasets="${DATASETS:-mvtec visa mpdd btad}"

for model in ${models}; do
    for dataset in ${datasets}; do
        npz_path="${results_root}/${model}/difficulty_inputs/${dataset}/all_predictions.npz"
        output_dir="${split_root}/${model}/${dataset}/${seed}seed_${shot}shot"

        if [[ ! -f "${npz_path}" ]]; then
            echo "Skip missing npz: ${npz_path}"
            continue
        fi

        echo "==> Difficulty split: model=${model}, dataset=${dataset}"
        python tools/create_difficulty.py \
            --npz_path "${npz_path}" \
            --output_dir "${output_dir}" \
            --dataset "${dataset}" \
            --method "${model}" \
            --seed "${seed}" \
            --shot "${shot}"
    done
done

echo "Difficulty split creation done."
