#!/bin/bash
# colab.sh on Kaggle: trains on a Kaggle T4 (private dataset + private kernel), runs the model over
# data/stanza_corrected and downloads models/intensifier (old one kept as models/intensifier.old) and output/.
# Extra arguments go to train.py. Runs ~1 h in the foreground: start it in tmux. WATCH=1 follows a pushed kernel.
set -e
repo=$(cd "$(dirname "$0")/../.." && pwd)
stage=$repo/data/kaggle_finetune
user=$(kaggle config view | sed -n 's/^- username: //p')
data=$user/intensifier-ft-data
kernel=$user/intensifier-ft
mkdir -p "$stage/data" "$stage/kernel"
cd "$repo"

push() {
  [ -f data/finetune/train.jsonl.gz ] || python pipeline/silver.py
  : > "$stage/train_args.txt"; [ $# -gt 0 ] && printf '%q ' "$@" > "$stage/train_args.txt"
  echo "output/intensifier_data_model_$(date +%F)" > "$stage/out_name.txt"
  tar czf "$stage/data/job.tgz" pipeline scripts/finetune data/finetune data/stanza_corrected \
    -C "$stage" train_args.txt out_name.txt
  cat > "$stage/data/dataset-metadata.json" <<JSON
{"title": "intensifier-ft-data", "id": "$data", "licenses": [{"name": "unknown"}]}
JSON
  if kaggle datasets status "$data" > /dev/null 2>&1; then
    kaggle datasets version -p "$stage/data" -m "ft $(date +%F_%T)" -q
  else
    kaggle datasets create -p "$stage/data" -q
  fi
  until kaggle datasets status "$data" 2>/dev/null | grep -q ready; do sleep 20; done
  cp scripts/finetune/kaggle_train.py "$stage/kernel/"
  cat > "$stage/kernel/kernel-metadata.json" <<JSON
{"id": "$kernel", "title": "intensifier-ft", "code_file": "kaggle_train.py", "language": "python",
 "kernel_type": "script", "is_private": true, "enable_gpu": true, "enable_internet": true,
 "dataset_sources": ["$data"], "competition_sources": [], "kernel_sources": []}
JSON
  kaggle kernels push -p "$stage/kernel" --accelerator NvidiaTeslaT4 -t 43200
}

[ -n "$WATCH" ] || push "$@"
while true; do
  sleep 300
  s=$(kaggle kernels status "$kernel" 2>&1 | grep -o 'status "[^"]*"' || echo unknown)
  echo "$(date +%T) $s"
  case $s in
    *COMPLETE*|*ERROR*|*CANCEL*) break ;;
  esac
done
rm -rf "$stage/out" && mkdir -p "$stage/out"
kaggle kernels output "$kernel" -p "$stage/out" -q
[ -f "$stage/out/model.tgz" ] || { echo "no model; kernel $s, log $stage/out/job.log"; exit 1; }
if [ -d models/intensifier ]; then
  rm -rf models/intensifier.old
  mv models/intensifier models/intensifier.old
fi
tar xzf "$stage/out/model.tgz" -C "$repo"
[ -d "$stage/out/output" ] && cp -r "$stage/out/output/." output/
echo "done ($s): models/intensifier ($(du -sh models/intensifier | cut -f1)), output $(cat "$stage/out_name.txt").sqlite, log $stage/out/job.log"
