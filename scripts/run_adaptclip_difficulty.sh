#!/usr/bin/env bash
set -euo pipefail

device="${CUDA_DEVICE:-0}"
shot="${SHOT:-0}"
seed="${SEED:-10}"
batch_size="${BATCH_SIZE:-8}"
num_workers="${NUM_WORKERS:-4}"

mvtec_root="${MVTEC_ROOT:-./dataset/MVTec}"
visa_root="${VISA_ROOT:-./dataset/Visa}"
mpdd_root="${MPDD_ROOT:-./dataset/MPDD}"
btad_root="${BTAD_ROOT:-./dataset/BTAD}"
save_root="${SAVE_ROOT:-./results/adaptclip}"
split_root="${SPLIT_ROOT:-./results/difficulty_splits}"
method="${METHOD:-adaptclip}"
datasets="${DATASETS:-mvtec visa mpdd btad}"

n_ctx=12
vl_reduction=4
pq_mid_dim=128

checkpoint_root="${CHECKPOINT_ROOT:-./adaptclip_checkpoints}"

checkpoint_for() {
    local train_dataset="$1"
    local base_dir="${n_ctx}_${vl_reduction}_${pq_mid_dim}_train_on_${train_dataset}_3adapters_batch8"
    local direct="${checkpoint_root}/${base_dir}/epoch_15.pth"
    local nested="${checkpoint_root}/adaptclip_checkpoints/${base_dir}/epoch_15.pth"

    if [[ -f "${direct}" ]]; then
        printf '%s\n' "${direct}"
        return
    fi

    if [[ -f "${nested}" ]]; then
        printf '%s\n' "${nested}"
        return
    fi

    printf 'Missing checkpoint for %s. Tried:\n  %s\n  %s\n' "${train_dataset}" "${direct}" "${nested}" >&2
    exit 1
}

run_dataset() {
    local dataset="$1"
    local data_path="$2"
    local train_dataset="$3"
    local checkpoint_path
    checkpoint_path="$(checkpoint_for "${train_dataset}")"

    echo "==> AdaptCLIP inference: dataset=${dataset}, shot=${shot}, seed=${seed}"
    CUDA_VISIBLE_DEVICES="${device}" python test.py \
        --dataset "${dataset}" \
        --test_data_path "${data_path}" \
        --seed "${seed}" \
        --k_shots "${shot}" \
        --checkpoint_path "${checkpoint_path}" \
        --save_path "${save_root}" \
        --features_list 6 12 18 24 \
        --image_size 518 \
        --batch_size "${batch_size}" \
        --num_workers "${num_workers}" \
        --n_ctx "${n_ctx}" \
        --vl_reduction "${vl_reduction}" \
        --pq_mid_dim "${pq_mid_dim}" \
        --visual_learner \
        --textual_learner \
        --pq_learner \
        --pq_context \
        --save_difficulty_inputs

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
            run_dataset mvtec "${mvtec_root}" mvtec
            ;;
        visa)
            run_dataset visa "${visa_root}" visa
            ;;
        mpdd)
            run_dataset mpdd "${mpdd_root}" mvtec
            ;;
        btad)
            run_dataset btad "${btad_root}" mvtec
            ;;
        *)
            echo "Unknown dataset: ${dataset}" >&2
            exit 2
            ;;
    esac
done

echo "Done."
