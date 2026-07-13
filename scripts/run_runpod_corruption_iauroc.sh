#!/usr/bin/env bash
set -euo pipefail

# Evaluate already-generated MVTec_corruption and VisA_corruption datasets.
# Output: results/<model>/<corruption>/<dataset>/{class_metrics_*.csv,<class>.csv,...}

device="${CUDA_DEVICE:-0}"
seed="${SEED:-10}"
shot="${SHOT:-0}"
batch_size="${BATCH_SIZE:-8}"
num_workers="${NUM_WORKERS:-4}"
evaluator_device="${EVALUATOR_DEVICE:-cpu}"
results_root="${RESULTS_ROOT:-./results}"
models="${MODELS:-adaptclip anomalyclip winclip}"
datasets="${DATASETS:-mvtec visa}"
corruptions="${CORRUPTIONS:-gaussian_noise motion_blur brightness}"
skip_existing="${SKIP_EXISTING:-1}"

mvtec_corruption_root="${MVTEC_CORRUPTION_ROOT:-./dataset/MVTec_corruption}"
visa_corruption_root="${VISA_CORRUPTION_ROOT:-./dataset/VisA_corruption}"
adaptclip_checkpoint_root="${CHECKPOINT_ROOT:-./checkpoints/adaptclip}"
anomalyclip_checkpoint_root="${ANOMALYCLIP_CHECKPOINT_ROOT:-./checkpoints/anomalyclip}"
anomalyclip_checkpoint_path="${ANOMALYCLIP_CHECKPOINT:-}"

n_ctx="${N_CTX:-12}"
vl_reduction="${VL_REDUCTION:-4}"
pq_mid_dim="${PQ_MID_DIM:-128}"

dataset_root() {
    local dataset="$1" corruption="$2"
    case "${dataset}" in
        mvtec) printf '%s/%s\n' "${mvtec_corruption_root}" "${corruption}" ;;
        visa) printf '%s/%s\n' "${visa_corruption_root}" "${corruption}" ;;
        *) echo "Unknown dataset: ${dataset}" >&2; exit 2 ;;
    esac
}

adaptclip_checkpoint_for() {
    local dataset="$1"
    local train_dataset="mvtec"
    if [[ "${dataset}" == "mvtec" ]]; then train_dataset="visa"; fi
    local base="${n_ctx}_${vl_reduction}_${pq_mid_dim}_train_on_${train_dataset}_3adapters_batch8"
    local candidates=(
        "${adaptclip_checkpoint_root}/${base}/epoch_15.pth"
        "${adaptclip_checkpoint_root}/adaptclip/${base}/epoch_15.pth"
        "${adaptclip_checkpoint_root}/adaptclip_checkpoints/${base}/epoch_15.pth"
        "./adaptclip_checkpoints/adaptclip_checkpoints/${base}/epoch_15.pth"
        "./adaptclip_checkpoints/${base}/epoch_15.pth"
    )
    local path
    for path in "${candidates[@]}"; do
        if [[ -f "${path}" ]]; then printf '%s\n' "${path}"; return; fi
    done
    printf 'Missing AdaptCLIP checkpoint for %s\n' "${dataset}" >&2
    printf '  %s\n' "${candidates[@]}" >&2
    exit 1
}

anomalyclip_checkpoint_for() {
    local dataset="$1"
    if [[ -n "${anomalyclip_checkpoint_path}" ]]; then
        printf '%s\n' "${anomalyclip_checkpoint_path}"
        return
    fi
    local path="${anomalyclip_checkpoint_root}/9_12_4_multiscale_visa_epoch_15.pth"
    if [[ "${dataset}" == "mvtec" ]]; then
        path="${anomalyclip_checkpoint_root}/9_12_4_multiscale_epoch_15.pth"
    fi
    if [[ ! -f "${path}" ]]; then
        echo "Missing AnomalyCLIP checkpoint: ${path}" >&2
        exit 1
    fi
    printf '%s\n' "${path}"
}

run_one() {
    local model="$1" dataset="$2" corruption="$3"
    local root save_dir metric_file checkpoint
    root="$(dataset_root "${dataset}" "${corruption}")"
    save_dir="${results_root}/${model}/${corruption}/${dataset}"
    metric_file="${save_dir}/class_metrics_${dataset}_${seed}seed_${shot}shot.csv"

    if [[ ! -d "${root}" ]]; then
        echo "Missing dataset directory: ${root}" >&2
        exit 1
    fi
    if [[ "${skip_existing}" == "1" && -s "${metric_file}" ]]; then
        echo "==> Skip existing: ${model}/${corruption}/${dataset}"
        return
    fi
    mkdir -p "${save_dir}"
    echo "==> ${model} | ${dataset} | ${corruption} | data=${root}"

    case "${model}" in
        adaptclip)
            checkpoint="$(adaptclip_checkpoint_for "${dataset}")"
            CUDA_VISIBLE_DEVICES="${device}" python test_adpatclip.py \
                --dataset "${dataset}" --test_data_path "${root}" \
                --seed "${seed}" --k_shots "${shot}" --checkpoint_path "${checkpoint}" \
                --save_path "${save_dir}" --features_list 6 12 18 24 --image_size 518 \
                --batch_size "${batch_size}" --num_workers "${num_workers}" \
                --evaluator_device "${evaluator_device}" --eval_metrics I-AUROC \
                --n_ctx "${n_ctx}" --vl_reduction "${vl_reduction}" --pq_mid_dim "${pq_mid_dim}" \
                --visual_learner --textual_learner --pq_learner --pq_context \
                --save_sample_scores --no-save-selected-heatmaps
            ;;
        anomalyclip)
            checkpoint="$(anomalyclip_checkpoint_for "${dataset}")"
            CUDA_VISIBLE_DEVICES="${device}" python test_anomalyclip.py \
                --dataset "${dataset}" --test_data_path "${root}" \
                --seed "${seed}" --k_shots "${shot}" --checkpoint_path "${checkpoint}" \
                --save_path "${save_dir}" --batch_size "${batch_size}" \
                --num_workers "${num_workers}" --evaluator_device "${evaluator_device}" \
                --eval_metrics I-AUROC --save_sample_scores --no-save-selected-heatmaps
            ;;
        winclip)
            CUDA_VISIBLE_DEVICES="${device}" python test_winclip.py \
                --dataset "${dataset}" --test_data_path "${root}" \
                --seed "${seed}" --k_shots "${shot}" --save_path "${save_dir}" \
                --image_size 240 --batch_size "${batch_size}" --num_workers "${num_workers}" \
                --evaluator_device "${evaluator_device}" --eval_metrics I-AUROC \
                --save_sample_scores --no-save-selected-heatmaps
            ;;
        *) echo "Unknown model: ${model}" >&2; exit 2 ;;
    esac
}

echo "models=${models}"
echo "datasets=${datasets}"
echo "corruptions=${corruptions}"
echo "results_root=${results_root}"

for model in ${models}; do
    for corruption in ${corruptions}; do
        for dataset in ${datasets}; do
            run_one "${model}" "${dataset}" "${corruption}"
        done
    done
done

echo "Done: ${results_root}/{adaptclip,anomalyclip,winclip}/<corruption>/<dataset>"
