#!/bin/bash
# colab.sh on Kaggle: trains on a Kaggle T4 (private dataset + private kernel), runs the model over
# data/text and downloads models/intensifier (old one kept as models/intensifier.old) and data/output/.
# Extra arguments go to train.py. Runs ~1 h in the foreground: start it in tmux. WATCH=1 follows a pushed kernel.
set -e
repo=$(cd "$(dirname "$0")/.." && pwd)
stage=$repo/data/kaggle_finetune
user=$(kaggle config view | sed -n 's/^- username: //p')
mkdir -p "$stage/data"
cd "$repo"
. train/lib.sh

if [ -z "$WATCH" ]; then
  need_data
  echo "data/output/intensifier_data_model_$(date +%F)" > "$stage/out_name.txt"
  pack "$stage/data/job.tgz" "bash train/job.sh \"\$1\" $(cat "$stage/out_name.txt") $(quote_args "$@")" \
    pipeline train data/finetune data/text
  kaggle_push intensifier-ft
fi
kaggle_wait "$user/intensifier-ft"
install_model "$stage/out/model.tgz" || { echo "no model; kernel $status, log $stage/out/job.log"; exit 1; }
[ -f "$stage/out/results.tgz" ] && tar xzf "$stage/out/results.tgz" -C "$repo"
echo "done ($status): models/intensifier, output $(cat "$stage/out_name.txt").sqlite, log $stage/out/job.log"
