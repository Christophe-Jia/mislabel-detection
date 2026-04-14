#!/bin/bash
set -euo pipefail

# ===========================================================================
# Unified experiment runner for mislabel-detection
# Usage: ./run.sh <command> [args...]
# ===========================================================================

RESULT_SAVE_PATH="replication"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

check_required() {
    local name="$1" value="$2"
    if [ -z "$value" ]; then
        echo "Error: missing required argument <${name}>"
        echo ""
        usage
        exit 1
    fi
}

run_experiment() {
    local savedir="$1"; shift
    local cmd="$*"
    echo "$cmd"
    if [ -z "${TESTRUN:-}" ]; then
        mkdir -p "$savedir"
        echo "$cmd" > "$savedir/cmd.txt"
        eval "$cmd"
    fi
}

# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

cmd_generate_td() {
    local datadir="${1:-}"; check_required "datadir" "$datadir"
    local dataset="${2:-}"; check_required "dataset" "$dataset"
    local seed="${3:-}";    check_required "seed" "$seed"
    local noise_ratio="${4:-}"; check_required "noise_ratio" "$noise_ratio"
    local noise_type="${5:-}";  check_required "noise_type" "$noise_type"
    local net_type="${6:-}";    check_required "net_type" "$net_type"
    local depth="${7:-}";       check_required "depth" "$depth"

    # Build save directory
    local savedir
    if [ "$dataset" = "webvision50" ] || [ "$dataset" = "clothing100k" ]; then
        savedir="${RESULT_SAVE_PATH}/${dataset}_${net_type}${depth}/computation4td_seed${seed}"
    else
        savedir="${RESULT_SAVE_PATH}/${dataset}_${net_type}${depth}_percmislabeled${noise_ratio}_${noise_type}/computation4td_seed${seed}"
    fi

    local args="--data ${datadir}/${dataset} --dataset ${dataset} --net_type ${net_type} --depth ${depth}"
    args="${args} --noise_ratio ${noise_ratio} --noise_type ${noise_type} --seed ${seed} --num_valid 0"
    local train_args="--num_epochs 200 --lr 0.1 --wd 1e-4 --batch_size 128 --num_workers 4"

    run_experiment "$savedir" "python runner.py ${args} --save ${savedir} - train_for_td_computation ${train_args} - done"
}

cmd_denoise_small() {
    local datadir="${1:-}";       check_required "datadir" "$datadir"
    local dataset="${2:-}";       check_required "dataset" "$dataset"
    local seed="${3:-}";          check_required "seed" "$seed"
    local noise_ratio="${4:-}";   check_required "noise_ratio" "$noise_ratio"
    local noise_type="${5:-}";    check_required "noise_type" "$noise_type"
    local detector_file="${6:-}"; check_required "detector_file" "$detector_file"
    local remove_ratio="${7:-}";  check_required "remove_ratio" "$remove_ratio"

    local net_type="resnet"
    local depth=32

    local td_file="${RESULT_SAVE_PATH}/${dataset}_${net_type}${depth}_percmislabeled${noise_ratio}_${noise_type}/computation4td_seed${seed}"

    local savedir="${RESULT_SAVE_PATH}/${dataset}_${net_type}${depth}"
    savedir="${savedir}_percmislabeled${noise_ratio}_${noise_type}/prune4retrain_seed${seed}/detector_${detector_file}/percremove${remove_ratio}"

    local args="--data ${datadir}/${dataset} --save ${savedir} --dataset ${dataset} --net_type ${net_type} --depth ${depth}"
    args="${args} --noise_ratio ${noise_ratio} --noise_type ${noise_type} --seed ${seed} --num_valid 0"
    local train_args="--num_epochs 300 --lr 0.1 --lr_drops 0.5 --wd 1e-4 --batch_size 256 --num_workers 4"
    train_args="${train_args} --td_files ${td_file} --remove_ratio ${remove_ratio} --detector_files ${detector_file}"

    run_experiment "$savedir" "python runner.py ${args} - train ${train_args} - done"
}

cmd_denoise_large() {
    local datadir="${1:-}";       check_required "datadir" "$datadir"
    local dataset="${2:-}";       check_required "dataset" "$dataset"
    local seed="${3:-}";          check_required "seed" "$seed"
    local detector_file="${4:-}"; check_required "detector_file" "$detector_file"
    local remove_ratio="${5:-}";  check_required "remove_ratio" "$remove_ratio"

    local net_type="resnet"
    local depth=50

    local td_file="${RESULT_SAVE_PATH}/${dataset}_${net_type}${depth}/computation4td_seed${seed}"

    local savedir="${RESULT_SAVE_PATH}/${dataset}_${net_type}${depth}"
    savedir="${savedir}/prune4retrain_seed${seed}/detector_${detector_file}/percremove${remove_ratio}"

    local args="--data ${datadir}/${dataset} --save ${savedir} --dataset ${dataset} --net_type ${net_type} --depth ${depth}"
    args="${args} --seed ${seed} --num_valid 0"
    local train_args="--num_epochs 300 --lr 0.1 --lr_drops 0.33,0.67 --wd 1e-4 --batch_size 256 --num_workers 4"
    train_args="${train_args} --td_files ${td_file} --remove_ratio ${remove_ratio} --detector_files ${detector_file}"

    run_experiment "$savedir" "python runner.py ${args} - train ${train_args} - done"
}

cmd_train_cub() {
    echo "Running CUB-200-2011 data debugging experiment..."
    python "${SCRIPT_DIR}/data_debug_dm/Train_cub.py" "$@"
}

cmd_train_webvision() {
    echo "Running WebVision data debugging experiment..."
    python "${SCRIPT_DIR}/data_debug_dm/Train_webvision.py" "$@"
}

# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------

usage() {
    cat <<'HELP'
Usage: ./run.sh <command> [args...]

Commands:
  generate_td       Generate training dynamics (Step 1)
  denoise_small     Denoise small datasets and retrain (Step 3a)
  denoise_large     Denoise large datasets and retrain (Step 3b)
  train_cub         CUB-200 data debugging (Chapter 4.3)
  train_webvision   WebVision data debugging (Chapter 4.3)
  help              Show this help message

Arguments for generate_td:
  <datadir> <dataset> <seed> <noise_ratio> <noise_type> <net_type> <depth>

  Example:
    CUDA_VISIBLE_DEVICES=0 ./run.sh generate_td /data cifar10 1 0.2 uniform resnet 32
    CUDA_VISIBLE_DEVICES=0 ./run.sh generate_td /data webvision50 1 0. uniform resnet 50

Arguments for denoise_small:
  <datadir> <dataset> <seed> <noise_ratio> <noise_type> <detector_file> <remove_ratio>

  Example:
    CUDA_VISIBLE_DEVICES=0 ./run.sh denoise_small /data cifar10 1 0.2 uniform cifar10_0.2_lstm_detector.pth.tar 0.2

Arguments for denoise_large:
  <datadir> <dataset> <seed> <detector_file> <remove_ratio>

  Example:
    CUDA_VISIBLE_DEVICES=0 ./run.sh denoise_large /data webvision50 1 cifar100_0.3_lstm_detector.pth.tar 0.2

Arguments for train_cub (passed directly to Train_cub.py):
  --r 0.2  --noise_mode sym  --repair_ratio 0.05  --num_epochs 300  ...

  Example:
    CUDA_VISIBLE_DEVICES=0 ./run.sh train_cub --r 0.2 --noise_mode sym --num_epochs 300

Arguments for train_webvision (passed directly to Train_webvision.py):
  --repair_ratio 0.05  --num_epochs 100  --num_class 50  ...

  Example:
    CUDA_VISIBLE_DEVICES=0 ./run.sh train_webvision --repair_ratio 0.05 --num_epochs 100

Environment variables:
  CUDA_VISIBLE_DEVICES  GPU device IDs (e.g. 0, 0,1)
  TESTRUN               Set to any value for dry run (print commands only)
HELP
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

case "${1:-help}" in
    generate_td)     shift; cmd_generate_td "$@" ;;
    denoise_small)   shift; cmd_denoise_small "$@" ;;
    denoise_large)   shift; cmd_denoise_large "$@" ;;
    train_cub)       shift; cmd_train_cub "$@" ;;
    train_webvision) shift; cmd_train_webvision "$@" ;;
    help|*)          usage ;;
esac
