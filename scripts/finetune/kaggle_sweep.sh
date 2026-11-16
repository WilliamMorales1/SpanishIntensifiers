#!/bin/bash
# sweep.sh on Kaggle instead of Colab: uploads the job as a private dataset, runs scripts/finetune/kaggle_job.py as a
# private GPU kernel (T4 x2, one config per GPU) and copies each finished config's metrics into models/sweep/<name>/.
# Configs already in models/sweep are skipped. Start it in tmux; WATCH=1 only follows a kernel already pushed.
# usage: scripts/finetune/kaggle_sweep.sh configs.txt
set -e
repo=$(cd "$(dirname "$0")/../.." && pwd)
configs=$(realpath "$1")
stage=$repo/data/kaggle_sweep
user=$(kaggle config view | sed -n 's/^- username: //p')
data=$user/intensifier-sweep-data
kernel=$user/intensifier-sweep
mkdir -p "$stage/data" "$stage/kernel" "$repo/models/sweep"
cd "$repo"

push() {
  cp "$configs" "$stage/configs.txt"
  tar czf "$stage/data/job.tgz" pipeline scripts/finetune data/finetune/*.json data/finetune/*.jsonl.gz models/sweep \
    -C "$stage" configs.txt
  cat > "$stage/data/dataset-metadata.json" <<JSON
{"title": "intensifier-sweep-data", "id": "$data", "licenses": [{"name": "unknown"}]}
JSON
  if kaggle datasets status "$data" > /dev/null 2>&1; then
    kaggle datasets version -p "$stage/data" -m "sweep $(date +%F_%T)" -q
  else
    kaggle datasets create -p "$stage/data" -q
  fi
  until kaggle datasets status "$data" 2>/dev/null | grep -q ready; do sleep 20; done
  cp scripts/finetune/kaggle_job.py "$stage/kernel/"
  cat > "$stage/kernel/kernel-metadata.json" <<JSON
{"id": "$kernel", "title": "intensifier-sweep", "code_file": "kaggle_job.py", "language": "python",
 "kernel_type": "script", "is_private": true, "enable_gpu": true, "enable_internet": true,
 "dataset_sources": ["$data"], "competition_sources": [], "kernel_sources": []}
JSON
  kaggle kernels push -p "$stage/kernel" --accelerator NvidiaTeslaT4 -t 43200
}

pull() {
  rm -rf "$stage/out" && mkdir -p "$stage/out"
  kaggle kernels output "$kernel" -p "$stage/out" -q > /dev/null 2>&1 || return 0
  for d in "$stage"/out/sweep/*/; do
    [ -f "$d/metrics.json" ] && mkdir -p "models/sweep/$(basename "$d")" && cp "$d"/*.json "models/sweep/$(basename "$d")/"
  done
  true
}

[ -n "$WATCH" ] || push
while true; do
  sleep 300
  s=$(kaggle kernels status "$kernel" 2>&1 | grep -o 'status "[^"]*"' || echo unknown)
  echo "$(date +%T) $s"
  case $s in
    *COMPLETE*) pull; echo "done: models/sweep ($(ls models/sweep | tr '\n' ' ')), log $stage/out/"; exit 0 ;;
    *ERROR*|*CANCEL*) pull; echo "kernel ended: $s; logs: kaggle kernels output $kernel -p $stage/out"; exit 1 ;;
  esac
done
