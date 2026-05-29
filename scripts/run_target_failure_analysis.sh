#!/usr/bin/env bash
set -euo pipefail

device="${CUDA_DEVICE:-0}"
seed="${SEED:-10}"
shot="${SHOT:-0}"
batch_size="${BATCH_SIZE:-2}"
num_workers="${NUM_WORKERS:-4}"
evaluator_device="${EVALUATOR_DEVICE:-cuda}"
results_root="${RESULTS_ROOT:-./results/target_failure_analysis}"

models="${MODELS:-adaptclip winclip anomalyclip}"
corruptions="${CORRUPTIONS:-gaussian_noise motion_blur brightness}"
eval_metrics="${EVAL_METRICS:-I-AUROC P-AUROC P-AP P-F1max}"
skip_existing="${SKIP_EXISTING:-1}"
save_heatmaps="${SAVE_HEATMAPS:-0}"

mvtec_root="${MVTEC_ROOT:-./dataset/MVTec}"
visa_root="${VISA_ROOT:-./dataset/Visa}"
btad_root="${BTAD_ROOT:-./dataset/BTAD}"

n_ctx="${N_CTX:-12}"
vl_reduction="${VL_REDUCTION:-4}"
pq_mid_dim="${PQ_MID_DIM:-128}"
adaptclip_checkpoint_root="${CHECKPOINT_ROOT:-./checkpoints/adaptclip}"
anomalyclip_checkpoint_root="${ANOMALYCLIP_CHECKPOINT_ROOT:-./checkpoints/anomalyclip}"
anomalyclip_checkpoint_path="${ANOMALYCLIP_CHECKPOINT:-}"

dataset_root() {
    case "$1" in
        mvtec) printf '%s\n' "${mvtec_root}" ;;
        visa) printf '%s\n' "${visa_root}" ;;
        btad) printf '%s\n' "${btad_root}" ;;
        *) echo "Unknown dataset: $1" >&2; exit 2 ;;
    esac
}

classes_for_dataset() {
    case "$1" in
        mvtec) printf '%s\n' cable pill screw transistor ;;
        visa) printf '%s\n' cashew macaroni1 macaroni2 pcb2 pcb3 ;;
        btad) printf '%s\n' 02 ;;
        *) echo "Unknown dataset: $1" >&2; exit 2 ;;
    esac
}

adaptclip_checkpoint_for() {
    local train_dataset="$1"
    local base_dir="${n_ctx}_${vl_reduction}_${pq_mid_dim}_train_on_${train_dataset}_3adapters_batch8"
    local candidates=(
        "${adaptclip_checkpoint_root}/${base_dir}/epoch_15.pth"
        "${adaptclip_checkpoint_root}/adaptclip_checkpoints/${base_dir}/epoch_15.pth"
        "./adaptclip_checkpoints/adaptclip_checkpoints/${base_dir}/epoch_15.pth"
        "./adaptclip_checkpoints/${base_dir}/epoch_15.pth"
    )
    for candidate in "${candidates[@]}"; do
        if [[ -f "${candidate}" ]]; then
            printf '%s\n' "${candidate}"
            return
        fi
    done
    printf 'Missing AdaptCLIP checkpoint for %s. Tried:\n' "${train_dataset}" >&2
    printf '  %s\n' "${candidates[@]}" >&2
    exit 1
}

anomalyclip_checkpoint_for() {
    local dataset="$1"
    if [[ -n "${anomalyclip_checkpoint_path}" ]]; then
        printf '%s\n' "${anomalyclip_checkpoint_path}"
        return
    fi

    local candidate
    if [[ "${dataset}" = "mvtec" ]]; then
        candidate="${anomalyclip_checkpoint_root}/9_12_4_multiscale_visa_epoch_15.pth"
    else
        candidate="${anomalyclip_checkpoint_root}/9_12_4_multiscale_epoch_15.pth"
    fi

    if [[ -f "${candidate}" ]]; then
        printf '%s\n' "${candidate}"
        return
    fi
    printf 'Missing AnomalyCLIP checkpoint for %s. Tried:\n  %s\n' "${dataset}" "${candidate}" >&2
    exit 1
}

train_dataset_for_adaptclip() {
    case "$1" in
        mvtec) printf 'visa\n' ;;
        *) printf 'mvtec\n' ;;
    esac
}

conditions() {
    printf 'clean\n'
    for corruption in ${corruptions}; do
        printf '%s_s3\n' "${corruption}"
    done
}

condition_corruption() {
    local condition="$1"
    if [[ "${condition}" == "clean" ]]; then
        printf '\n'
    else
        printf '%s\n' "${condition%_s3}"
    fi
}

run_target() {
    local model="$1"
    local dataset="$2"
    local class_name="$3"
    local condition="$4"
    local root
    root="$(dataset_root "${dataset}")"

    local save_dir="${results_root}/${model}/${dataset}/${class_name}/${condition}"
    local score_file="${save_dir}/sample_scores_${dataset}_${seed}seed_${shot}shot.csv"
    if [[ "${skip_existing}" == "1" && -f "${score_file}" ]]; then
        echo "==> Skip existing target: model=${model}, dataset=${dataset}, class=${class_name}, condition=${condition}"
        return
    fi

    local corruption
    corruption="$(condition_corruption "${condition}")"
    local corruption_args=()
    if [[ -n "${corruption}" ]]; then
        corruption_args=(--corruption "${corruption}" --corruption_severity 3)
    fi

    local heatmap_args=(--no-save-selected-heatmaps)
    if [[ "${save_heatmaps}" == "1" ]]; then
        heatmap_args=(--save_heatmap --save-selected-heatmaps --heatmap_topk 5)
    fi

    echo "==> Target failure: model=${model}, dataset=${dataset}, class=${class_name}, condition=${condition}"
    case "${model}" in
        adaptclip)
            local train_dataset
            local checkpoint_path
            train_dataset="$(train_dataset_for_adaptclip "${dataset}")"
            checkpoint_path="$(adaptclip_checkpoint_for "${train_dataset}")"
            CUDA_VISIBLE_DEVICES="${device}" python test_adpatclip.py \
                --dataset "${dataset}" \
                --test_data_path "${root}" \
                --class_name "${class_name}" \
                --seed "${seed}" \
                --k_shots "${shot}" \
                --checkpoint_path "${checkpoint_path}" \
                --save_path "${save_dir}" \
                --features_list 6 12 18 24 \
                --image_size 518 \
                --batch_size "${batch_size}" \
                --num_workers "${num_workers}" \
                --evaluator_device "${evaluator_device}" \
                --eval_metrics ${eval_metrics} \
                --n_ctx "${n_ctx}" \
                --vl_reduction "${vl_reduction}" \
                --pq_mid_dim "${pq_mid_dim}" \
                --visual_learner \
                --textual_learner \
                --pq_learner \
                --pq_context \
                --save_sample_scores \
                "${heatmap_args[@]}" \
                "${corruption_args[@]}"
            ;;
        winclip)
            CUDA_VISIBLE_DEVICES="${device}" python test_winclip.py \
                --dataset "${dataset}" \
                --test_data_path "${root}" \
                --class_name "${class_name}" \
                --seed "${seed}" \
                --k_shots "${shot}" \
                --save_path "${save_dir}" \
                --image_size 240 \
                --batch_size "${batch_size}" \
                --num_workers "${num_workers}" \
                --evaluator_device "${evaluator_device}" \
                --eval_metrics ${eval_metrics} \
                --save_sample_scores \
                "${heatmap_args[@]}" \
                "${corruption_args[@]}"
            ;;
        anomalyclip)
            local checkpoint_path
            checkpoint_path="$(anomalyclip_checkpoint_for "${dataset}")"
            CUDA_VISIBLE_DEVICES="${device}" python test_anomalyclip.py \
                --dataset "${dataset}" \
                --test_data_path "${root}" \
                --class_name "${class_name}" \
                --seed "${seed}" \
                --k_shots "${shot}" \
                --save_path "${save_dir}" \
                --batch_size "${batch_size}" \
                --num_workers "${num_workers}" \
                --evaluator_device "${evaluator_device}" \
                --eval_metrics ${eval_metrics} \
                --checkpoint_path "${checkpoint_path}" \
                --save_sample_scores \
                "${heatmap_args[@]}" \
                "${corruption_args[@]}"
            ;;
        *)
            echo "Unknown model: ${model}" >&2
            exit 2
            ;;
    esac
}

for model in ${models}; do
    for dataset in mvtec visa btad; do
        while IFS= read -r class_name; do
            while IFS= read -r condition; do
                run_target "${model}" "${dataset}" "${class_name}" "${condition}"
            done < <(conditions)
        done < <(classes_for_dataset "${dataset}")
    done
done

python tools/analyze_score_distribution.py --root "${results_root}"

echo "Target failure analysis done: ${results_root}"
