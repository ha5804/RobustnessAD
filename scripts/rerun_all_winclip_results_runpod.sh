#!/usr/bin/env bash
set -euo pipefail

# Rerun every WinCLIP result family currently used in this repo.
#
# Default behavior:
# - resumes missing WinCLIP outputs under results/corruption_benchmark
# - resumes missing WinCLIP outputs under results/target_failure_analysis
# - optionally refreshes the simple results/winclip inference folder
# - keeps AdaptCLIP / AnomalyCLIP outputs intact
# - does not delete existing results unless DELETE_OLD_WINCLIP=1 is explicitly set
#
# RunPod example:
#   bash scripts/rerun_all_winclip_results_runpod.sh
#
# Useful overrides:
#   CUDA_DEVICE=0 BATCH_SIZE=8 NUM_WORKERS=8 bash scripts/rerun_all_winclip_results_runpod.sh
#   FORCE_RERUN=1 CUDA_DEVICE=0 BATCH_SIZE=8 NUM_WORKERS=8 bash scripts/rerun_all_winclip_results_runpod.sh
#   RUN_TARGET=0 bash scripts/rerun_all_winclip_results_runpod.sh
#   RUN_CORRUPTION=0 RUN_STANDALONE=1 bash scripts/rerun_all_winclip_results_runpod.sh

device="${CUDA_DEVICE:-0}"
seed="${SEED:-10}"
shot="${SHOT:-0}"
batch_size="${BATCH_SIZE:-8}"
target_batch_size="${TARGET_BATCH_SIZE:-${batch_size}}"
num_workers="${NUM_WORKERS:-4}"
evaluator_device="${EVALUATOR_DEVICE:-cpu}"

run_corruption="${RUN_CORRUPTION:-1}"
run_target="${RUN_TARGET:-1}"
run_standalone="${RUN_STANDALONE:-0}"
delete_old="${DELETE_OLD_WINCLIP:-0}"
force_rerun="${FORCE_RERUN:-0}"
skip_existing="${SKIP_EXISTING:-1}"
if [[ "${force_rerun}" == "1" ]]; then
    skip_existing=0
fi

results_root="${RESULTS_ROOT:-./results/corruption_benchmark}"
target_results_root="${TARGET_RESULTS_ROOT:-./results/target_failure_analysis}"
standalone_save_root="${STANDALONE_SAVE_ROOT:-./results/winclip}"

datasets="${DATASETS:-mvtec visa btad}"
splits="${SPLITS:-all easy normal hard}"
corruptions="${CORRUPTIONS:-gaussian_noise motion_blur brightness}"
eval_metrics="${EVAL_METRICS:-I-AUROC I-AP I-F1max P-AUROC P-AP P-F1max P-AUPRO}"
target_eval_metrics="${TARGET_EVAL_METRICS:-I-AUROC P-AUROC P-AP P-F1max}"

mvtec_root="${MVTEC_ROOT:-./dataset/MVTec}"
visa_root="${VISA_ROOT:-./dataset/Visa}"
btad_root="${BTAD_ROOT:-./dataset/BTAD}"

export CUDA_DEVICE="${device}"
export SEED="${seed}"
export SHOT="${shot}"
export NUM_WORKERS="${num_workers}"
export EVALUATOR_DEVICE="${evaluator_device}"
export MVTEC_ROOT="${mvtec_root}"
export VISA_ROOT="${visa_root}"
export BTAD_ROOT="${btad_root}"
export SKIP_EXISTING="${skip_existing}"

echo "==> Rerun WinCLIP results"
echo "    CUDA_DEVICE=${device}"
echo "    SEED=${seed}, SHOT=${shot}"
echo "    DATASETS=${datasets}"
echo "    CORRUPTIONS=${corruptions}"
echo "    DELETE_OLD_WINCLIP=${delete_old}"
echo "    SKIP_EXISTING=${SKIP_EXISTING}"
echo

if [[ -d "${btad_root}" ]]; then
    echo "==> Cleaning macOS resource-fork files from BTAD, if any"
    find "${btad_root}" -name '._*' -delete
    if [[ -f "${btad_root}/meta.json" ]]; then
        if grep -qE '(^|/)\._|\.DS_Store' "${btad_root}/meta.json"; then
            echo "==> Regenerating BTAD meta.json because hidden files were recorded"
            rm -f "${btad_root}/meta.json"
            python dataset/generic_mvtec.py --root "${btad_root}" --dataset btad
        fi
    else
        echo "==> Generating BTAD meta.json"
        python dataset/generic_mvtec.py --root "${btad_root}" --dataset btad
    fi
fi

if [[ "${delete_old}" == "1" ]]; then
    backup_path="./winclip_results_backup_before_delete_$(date +%Y%m%d_%H%M%S).tar.gz"
    echo "==> DELETE_OLD_WINCLIP=1 requested. Backing up existing WinCLIP outputs to ${backup_path}"
    tar -czf "${backup_path}" \
        "${results_root}/winclip" \
        "${results_root}/splits/winclip" \
        "${target_results_root}/winclip" \
        "${standalone_save_root}" \
        2>/dev/null || true

    echo "==> Removing old WinCLIP-only outputs after backup"
    if [[ "${run_corruption}" == "1" ]]; then
        rm -rf \
            "${results_root}/winclip" \
            "${results_root}/splits/winclip"
        rm -f \
            "${results_root}/summaries/"*winclip* \
            "${results_root}/summaries/class_metrics_raw.csv" \
            "${results_root}/summaries/clean_vs_corruption_drop.csv" \
            "${results_root}/summaries/macro_summary.csv" \
            "${results_root}/summaries/micro_summary.csv" \
            "${results_root}/summaries/robustness_summary.csv" \
            "${results_root}/summaries/sample_scores_raw.csv" \
            "${results_root}/summaries/sample_clean_vs_corruption_drop.csv"
    fi

    if [[ "${run_target}" == "1" ]]; then
        rm -rf "${target_results_root}/winclip"
        rm -f "${target_results_root}/summaries/score_distribution_plots/"winclip_*
        rm -f \
            "${target_results_root}/summaries/target_sample_scores_raw.csv" \
            "${target_results_root}/summaries/target_score_distribution_summary.csv" \
            "${target_results_root}/summaries/target_clean_to_corruption_score_shift.csv"
    fi

    if [[ "${run_standalone}" == "1" ]]; then
        rm -rf "${standalone_save_root}"
    fi
fi

if [[ "${run_corruption}" == "1" ]]; then
    echo
    echo "==> Rebuilding corruption benchmark WinCLIP outputs in ${results_root}"
    MODELS="winclip" \
    RESULTS_ROOT="${results_root}" \
    DATASETS="${datasets}" \
    SPLITS="${splits}" \
    CORRUPTIONS="${corruptions}" \
    BATCH_SIZE="${batch_size}" \
    EVAL_METRICS="${eval_metrics}" \
    bash scripts/run_full_corruption_benchmark.sh
fi

if [[ "${run_target}" == "1" ]]; then
    echo
    echo "==> Rebuilding target failure WinCLIP outputs in ${target_results_root}"
    MODELS="winclip" \
    RESULTS_ROOT="${target_results_root}" \
    CORRUPTIONS="${corruptions}" \
    BATCH_SIZE="${target_batch_size}" \
    EVAL_METRICS="${target_eval_metrics}" \
    SAVE_HEATMAPS="${SAVE_HEATMAPS:-0}" \
    SAVE_ALL_HEATMAPS="${SAVE_ALL_HEATMAPS:-0}" \
    bash scripts/run_target_failure_analysis.sh
fi

if [[ "${run_standalone}" == "1" ]]; then
    echo
    echo "==> Rebuilding standalone WinCLIP inference outputs in ${standalone_save_root}"
    SAVE_ROOT="${standalone_save_root}" \
    DATASETS="${datasets}" \
    BATCH_SIZE="${batch_size}" \
    EVAL_METRICS="${eval_metrics}" \
    bash scripts/run_winclip_inference.sh
fi

echo
echo "==> Done rerunning requested WinCLIP outputs."
