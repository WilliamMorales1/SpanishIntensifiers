#!/bin/bash
# trains the released tagger config on any machine with a CUDA GPU (or CPU for a smoke test) in the current Python env.
# Extra arguments go to train.py and override the defaults below, e.g. train/run.sh --epochs 2 --limit 5
# usage: [OUT=models/intensifier] [DATA=data/finetune] train/run.sh [train.py args]
set -e
repo=$(cd "$(dirname "$0")/.." && pwd)
cd "$repo"
[ -f "${DATA:-data/finetune}/train.jsonl.gz" ] || { echo "no ${DATA:-data/finetune}/train.jsonl.gz: build it with pipeline/dataset.py"; exit 1; }
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
exec python -u train/train.py \
  --data "${DATA:-data/finetune}" --out "${OUT:-models/intensifier}" \
  --model FacebookAI/xlm-roberta-large --batch 8 --lr 2e-5 --epochs 8 "$@"
