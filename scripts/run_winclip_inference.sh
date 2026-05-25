#!/usr/bin/env bash
set -euo pipefail

device="${CUDA_DEVICE:-0}"
shot="${SHOT:-0}"
seed="${SEED:-10}"
batch_size="${BATCH_SIZE:-8}"
num_workers="${NUM_WORKERS:-4}"
image_size="${IMAGE_SIZE:-240}"
max_test_samples="${MAX_TEST_SAMPLES_PER_CLASS:-}"
evaluator_device="${EVALUATOR_DEVICE:-cpu}"
eval_metrics="${EVAL_METRICS:-I-AUROC P-AUROC P-AP}"
predictions_only="${PREDICTIONS_ONLY:-0}"

mvtec_root="${MVTEC_ROOT:-./dataset/MVTec}"
visa_root="${VISA_ROOT:-./dataset/Visa}"
mpdd_root="${MPDD_ROOT:-./dataset/MPDD}"
btad_root="${BTAD_ROOT:-./dataset/BTAD}"
save_root="${SAVE_ROOT:-./results/winclip}"
datasets="${DATASETS:-mvtec visa mpdd btad}"

run_dataset() {
    local dataset="$1"
    local data_path="$2"
    local sample_args=()

    if [[ -n "${max_test_samples}" ]]; then
        sample_args=(--max_test_samples_per_class "${max_test_samples}")
    fi
    if [[ "${predictions_only}" == "1" ]]; then
        sample_args+=(--predictions_only)
    fi

    echo "==> WinCLIP inference only: dataset=${dataset}, shot=${shot}, seed=${seed}"
    CUDA_VISIBLE_DEVICES="${device}" python test_winclip.py \
        --dataset "${dataset}" \
        --test_data_path "${data_path}" \
        --seed "${seed}" \
        --k_shots "${shot}" \
        --save_path "${save_root}" \
        --image_size "${image_size}" \
        --batch_size "${batch_size}" \
        --num_workers "${num_workers}" \
        --evaluator_device "${evaluator_device}" \
        --eval_metrics ${eval_metrics} \
        --save_difficulty_inputs \
        --no-save-selected-heatmaps \
        "${sample_args[@]}"
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

echo "Inference done. Run scripts/create_all_difficulty_splits.sh locally for splits."
