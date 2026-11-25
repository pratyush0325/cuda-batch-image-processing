#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

bash tools/check_environment.sh
make test
make build
python3 tools/generate_dataset.py --output data/input --count 256 \
    --width 256 --height 256 --seed 2026

run_dir="artifacts/run_$(date -u +%Y%m%dT%H%M%SZ)_$$"
mkdir -p "$run_dir"
{
    date -u
    uname -a
    python3 --version
    g++ --version
    "${NVCC:-nvcc}" --version
    nvidia-smi || true
    git rev-parse HEAD 2>/dev/null || true
    printf 'Command: build/batch_image --input data/input --output %s --batch %s --threshold 80 --samples 8\n' \
        "$run_dir" "${BATCH_SIZE:-16}"
} > "$run_dir/environment.txt" 2>&1

./build/batch_image --input data/input --output "$run_dir" \
    --batch "${BATCH_SIZE:-16}" --threshold 80 --samples 8 \
    | tee "$run_dir/console.log"
python3 tools/build_report.py --run "$run_dir" --input data/input
printf '\nGPU run complete. Keep and commit the entire evidence directory:\n%s\n' "$run_dir"
