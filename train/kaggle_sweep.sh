#!/bin/bash
# sweep.sh on Kaggle instead of Colab: uploads the job as a private dataset, runs train/sweep.py in a
# private GPU kernel (T4 x2, one config per GPU) and copies each finished config's metrics into models/sweep/<name>/.
# Configs already in models/sweep are skipped. Start it in tmux; WATCH=1 only follows a kernel already pushed.
# usage: train/kaggle_sweep.sh configs.txt
set -e
repo=$(cd "$(dirname "$0")/.." && pwd)
configs=$(realpath "$1")
stage=$repo/data/kaggle_sweep
user=$(kaggle config view | sed -n 's/^- username: //p')
mkdir -p "$stage/data" "$repo/models/sweep"
cd "$repo"
. train/lib.sh

if [ -z "$WATCH" ]; then
  cp "$configs" "$stage/configs.txt"
  pack "$stage/data/job.tgz" "python -u train/sweep.py configs.txt --done models/sweep --out \"\$1/sweep\"" \
    pipeline train data/finetune/*.json data/finetune/*.jsonl.gz models/sweep -C "$stage" configs.txt
  kaggle_push intensifier-sweep
fi
kaggle_wait "$user/intensifier-sweep"
for d in "$stage"/out/sweep/*/; do
  [ -f "$d/metrics.json" ] && mkdir -p "models/sweep/$(basename "$d")" && cp "$d"/*.json "models/sweep/$(basename "$d")/"
done
case $status in
  *COMPLETE*) echo "done: models/sweep ($(ls models/sweep | tr '\n' ' ')), log $stage/out/" ;;
  *) echo "kernel ended: $status; logs in $stage/out/"; exit 1 ;;
esac
