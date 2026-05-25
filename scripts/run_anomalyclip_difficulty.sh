#!/usr/bin/env bash
set -euo pipefail

device="${CUDA_DEVICE:-0}"
shot="${SHOT:-0}"
seed="${SEED:-10}"
batch_size="${BATCH_SIZE:-8}"
num_workers="${NUM_WORKERS:-4}"
max_test_samples="${MAX_TEST_SAMPLES_PER_CLASS:-}"
save_selected_heatmaps="${SAVE_SELECTED_HEATMAPS:-0}"

mvtec_root="${MVTEC_ROOT:-./dataset/MVTec}"
visa_root="${VISA_ROOT:-./dataset/Visa}"
mpdd_root="${MPDD_ROOT:-./dataset/MPDD}"
btad_root="${BTAD_ROOT:-./dataset/BTAD}"
save_root="${SAVE_ROOT:-./results/anomalyclip}"
split_root="${SPLIT_ROOT:-./results/difficulty_splits}"
method="${METHOD:-anomalyclip}"
datasets="${DATASETS:-mvtec visa mpdd btad}"
checkpoint_root="${ANOMALYCLIP_CHECKPOINT_ROOT:-./checkpoints/anomalyclip}"
checkpoint_path="${ANOMALYCLIP_CHECKPOINT:-}"

checkpoint_for() {
    local dataset="$1"

    if [[ -n "${checkpoint_path}" ]]; then
        printf '%s\n' "${checkpoint_path}"
        return
    fi

    local candidate
    if [[ "${dataset}" = "mvtec" ]]; then
        candidate="${checkpoint_root}/9_12_4_multiscale_visa_epoch_15.pth"
    else
        candidate="${checkpoint_root}/9_12_4_multiscale_epoch_15.pth"
    fi

    if [[ -f "${candidate}" ]]; then
        printf '%s\n' "${candidate}"
        return
    fi

    printf 'Missing AnomalyCLIP checkpoint for %s. Tried:\n  %s\n' "${dataset}" "${candidate}" >&2
    exit 1
}

run_dataset() {
    local dataset="$1"
    local data_path="$2"
    local dataset_checkpoint
    local heatmap_args=(--no-save-selected-heatmaps)
    local sample_args=()
    dataset_checkpoint="$(checkpoint_for "${dataset}")"

    if [[ "${save_selected_heatmaps}" = "1" ]]; then
        heatmap_args=(--save-selected-heatmaps)
    fi

    if [[ -n "${max_test_samples}" ]]; then
        sample_args=(--max_test_samples_per_class "${max_test_samples}")
    fi

    echo "==> AnomalyCLIP inference: dataset=${dataset}, shot=${shot}, seed=${seed}"
    CUDA_VISIBLE_DEVICES="${device}" python test_anomalyclip.py \
        --dataset "${dataset}" \
        --test_data_path "${data_path}" \
        --seed "${seed}" \
        --k_shots "${shot}" \
        --save_path "${save_root}" \
        --batch_size "${batch_size}" \
        --num_workers "${num_workers}" \
        --save_difficulty_inputs \
        --checkpoint_path "${dataset_checkpoint}" \
        "${heatmap_args[@]}" \
        "${sample_args[@]}"

    local npz_path="${save_root}/difficulty_inputs/${dataset}/all_predictions.npz"
    local output_dir="${split_root}/${method}/${dataset}/${seed}seed_${shot}shot"

    echo "==> Difficulty split: ${dataset}"
    python tools/create_difficulty.py \
        --npz_path "${npz_path}" \
        --output_dir "${output_dir}" \
        --dataset "${dataset}" \
        --method "${method}" \
        --seed "${seed}" \
        --shot "${shot}"
}

for dataset in ${datasets}; do
    case "${dataset}" in
        mvtec)
            run_dataset mvtec "${mvtec_root}"
            ;;
        visa)
            run_dataset visa "${visa_root}"
            ;;
        mpdd)
            run_dataset mpdd "${mpdd_root}"
            ;;
        btad)
            run_dataset btad "${btad_root}"
            ;;
        *)
            echo "Unknown dataset: ${dataset}" >&2
            exit 2
            ;;
    esac
done

echo "Done."
