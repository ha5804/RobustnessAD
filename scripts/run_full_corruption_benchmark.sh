#!/usr/bin/env bash
set -euo pipefail

device="${CUDA_DEVICE:-0}"
seed="${SEED:-10}"
shot="${SHOT:-0}"
batch_size="${BATCH_SIZE:-8}"
num_workers="${NUM_WORKERS:-4}"
evaluator_device="${EVALUATOR_DEVICE:-cpu}"
results_root="${RESULTS_ROOT:-./results/corruption_benchmark}"

models="${MODELS:-adaptclip winclip anomalyclip}"
datasets="${DATASETS:-mvtec visa btad}"
splits="${SPLITS:-all easy normal hard}"
base_corruptions="${CORRUPTIONS:-gaussian_noise motion_blur brightness rotation translation}"
include_mvtec_extra="${INCLUDE_MVTEC_EXTRA:-0}"
skip_existing="${SKIP_EXISTING:-1}"

eval_metrics="${EVAL_METRICS:-I-AUROC I-AP I-F1max P-AUROC P-AP P-F1max P-AUPRO}"

mvtec_root="${MVTEC_ROOT:-./dataset/MVTec}"
visa_root="${VISA_ROOT:-./dataset/Visa}"
btad_root="${BTAD_ROOT:-./dataset/BTAD}"

n_ctx="${N_CTX:-12}"
vl_reduction="${VL_REDUCTION:-4}"
pq_mid_dim="${PQ_MID_DIM:-128}"
adaptclip_checkpoint_root="${CHECKPOINT_ROOT:-./checkpoints/adaptclip}"
anomalyclip_checkpoint_root="${ANOMALYCLIP_CHECKPOINT_ROOT:-./checkpoints/anomalyclip}"
anomalyclip_checkpoint_path="${ANOMALYCLIP_CHECKPOINT:-}"

mkdir -p "${results_root}/summaries"

dataset_root() {
    case "$1" in
        mvtec) printf '%s\n' "${mvtec_root}" ;;
        visa) printf '%s\n' "${visa_root}" ;;
        btad) printf '%s\n' "${btad_root}" ;;
        *) echo "Unknown dataset: $1" >&2; exit 2 ;;
    esac
}

adaptclip_checkpoint_for() {
    local train_dataset="$1"
    local base_dir="${n_ctx}_${vl_reduction}_${pq_mid_dim}_train_on_${train_dataset}_3adapters_batch8"
    local candidates=(
        "${adaptclip_checkpoint_root}/${base_dir}/epoch_15.pth"
        "${adaptclip_checkpoint_root}/adaptclip/${base_dir}/epoch_15.pth"
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
        candidate="${anomalyclip_checkpoint_root}/9_12_4_multiscale_epoch_15.pth"
    else
        candidate="${anomalyclip_checkpoint_root}/9_12_4_multiscale_visa_epoch_15.pth"
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

condition_corruption() {
    local condition="$1"
    if [[ "${condition}" == "clean" ]]; then
        printf '\n'
    else
        printf '%s\n' "${condition%_s3}"
    fi
}

metric_file_for() {
    local save_dir="$1"
    local dataset="$2"
    printf '%s/class_metrics_%s_%sseed_%sshot.csv\n' "${save_dir}" "${dataset}" "${seed}" "${shot}"
}

prediction_file_for() {
    local save_dir="$1"
    local dataset="$2"
    printf '%s/difficulty_inputs/%s/all_predictions.npz\n' "${save_dir}" "${dataset}"
}

run_complete() {
    local split="$1"
    local metric_file="$2"
    local prediction_file="$3"

    if [[ ! -s "${metric_file}" ]]; then
        return 1
    fi
    if [[ "${split}" == "all" && ! -s "${prediction_file}" ]]; then
        return 1
    fi
    return 0
}

run_inference() {
    local model="$1"
    local dataset="$2"
    local split="$3"
    local condition="$4"
    local sample_csv="${5:-}"

    local root
    root="$(dataset_root "${dataset}")"

    local save_dir="${results_root}/${model}/${dataset}/${split}/${condition}"
    local metric_file
    local prediction_file
    metric_file="$(metric_file_for "${save_dir}" "${dataset}")"
    prediction_file="$(prediction_file_for "${save_dir}" "${dataset}")"

    if [[ "${skip_existing}" == "1" ]] && run_complete "${split}" "${metric_file}" "${prediction_file}"; then
        echo "==> Skip existing: model=${model}, dataset=${dataset}, split=${split}, condition=${condition}"
        return
    fi

    local corruption
    corruption="$(condition_corruption "${condition}")"
    local corruption_args=()
    if [[ -n "${corruption}" ]]; then
        corruption_args=(--corruption "${corruption}" --corruption_severity 3)
    fi

    local sample_args=()
    if [[ -n "${sample_csv}" ]]; then
        sample_args=(--sample_csv "${sample_csv}")
    fi

    local difficulty_args=()
    if [[ "${split}" == "all" ]]; then
        difficulty_args=(--save_difficulty_inputs)
    fi

    echo "==> Run: model=${model}, dataset=${dataset}, split=${split}, condition=${condition}, shot=${shot}, seed=${seed}"

    case "${model}" in
        adaptclip)
            local train_dataset
            local checkpoint_path
            train_dataset="$(train_dataset_for_adaptclip "${dataset}")"
            checkpoint_path="$(adaptclip_checkpoint_for "${train_dataset}")"
            CUDA_VISIBLE_DEVICES="${device}" python test_adpatclip.py \
                --dataset "${dataset}" \
                --test_data_path "${root}" \
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
                ${difficulty_args[@]+"${difficulty_args[@]}"} \
                --no-save-selected-heatmaps \
                "${corruption_args[@]}" \
                "${sample_args[@]}"
            ;;
        winclip)
            CUDA_VISIBLE_DEVICES="${device}" python test_winclip.py \
                --dataset "${dataset}" \
                --test_data_path "${root}" \
                --seed "${seed}" \
                --k_shots "${shot}" \
                --save_path "${save_dir}" \
                --image_size 240 \
                --batch_size "${batch_size}" \
                --num_workers "${num_workers}" \
                --evaluator_device "${evaluator_device}" \
                --eval_metrics ${eval_metrics} \
                ${difficulty_args[@]+"${difficulty_args[@]}"} \
                --no-save-selected-heatmaps \
                "${corruption_args[@]}" \
                "${sample_args[@]}"
            ;;
        anomalyclip)
            local checkpoint_path
            checkpoint_path="$(anomalyclip_checkpoint_for "${dataset}")"
            CUDA_VISIBLE_DEVICES="${device}" python test_anomalyclip.py \
                --dataset "${dataset}" \
                --test_data_path "${root}" \
                --seed "${seed}" \
                --k_shots "${shot}" \
                --save_path "${save_dir}" \
                --batch_size "${batch_size}" \
                --num_workers "${num_workers}" \
                --evaluator_device "${evaluator_device}" \
                --eval_metrics ${eval_metrics} \
                --checkpoint_path "${checkpoint_path}" \
                ${difficulty_args[@]+"${difficulty_args[@]}"} \
                --no-save-selected-heatmaps \
                "${corruption_args[@]}" \
                "${sample_args[@]}"
            ;;
        *)
            echo "Unknown model: ${model}" >&2
            exit 2
            ;;
    esac
}

create_unified_split() {
    local model="$1"
    local output_dir="${results_root}/splits/${model}/unified/${seed}seed_${shot}shot"

    if [[ "${skip_existing}" == "1" && -f "${output_dir}/all.csv" ]]; then
        echo "==> Skip existing unified split: model=${model}"
        return
    fi

    local input_args=()
    for dataset in ${datasets}; do
        local npz_path="${results_root}/${model}/${dataset}/all/clean/difficulty_inputs/${dataset}/all_predictions.npz"
        if [[ ! -f "${npz_path}" ]]; then
            echo "Missing clean prediction npz for unified split creation: ${npz_path}" >&2
            exit 1
        fi
        input_args+=(--input "${dataset}=${npz_path}")
    done

    echo "==> Create unified split: model=${model}, datasets=${datasets}"
    python tools/create_unified_difficulty.py \
        "${input_args[@]}" \
        --output_dir "${output_dir}" \
        --method "${model}" \
        --seed "${seed}" \
        --shot "${shot}"
}

conditions_for_dataset() {
    local dataset="$1"
    printf 'clean\n'
    for corruption in ${base_corruptions}; do
        printf '%s_s3\n' "${corruption}"
    done
    if [[ "${dataset}" == "mvtec" && "${include_mvtec_extra}" == "1" ]]; then
        printf 'contrast_s3\njpeg_compression_s3\ndownsample_upsample_s3\n'
    fi
}

for model in ${models}; do
    for dataset in ${datasets}; do
        run_inference "${model}" "${dataset}" "all" "clean" ""
    done
    create_unified_split "${model}"
done

for model in ${models}; do
    for dataset in ${datasets}; do
        split_dir="${results_root}/splits/${model}/unified/${seed}seed_${shot}shot"
        while IFS= read -r condition; do
            for split in ${splits}; do
                sample_csv=""
                if [[ "${split}" != "all" ]]; then
                    sample_csv="${split_dir}/${split}.csv"
                    if [[ ! -f "${sample_csv}" ]]; then
                        echo "Missing split csv: ${sample_csv}" >&2
                        exit 1
                    fi
                fi
                run_inference "${model}" "${dataset}" "${split}" "${condition}" "${sample_csv}"
            done
        done < <(conditions_for_dataset "${dataset}")
    done
done

python tools/summarize_corruption_benchmark.py --root "${results_root}"

echo "Unified corruption benchmark done: ${results_root}"
