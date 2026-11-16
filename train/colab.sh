#!/bin/bash
# trains the intensifier tagger on a Colab GPU, runs it over data/text, downloads the model into
# models/intensifier (as soon as training ends, before predict) and the table into data/output/, then stops the VM.
# Extra arguments go to train.py.
# Runs in the foreground for ~30-60 min: start it in tmux. A lost VM is replaced up to 3 times (training restarts).
set -e
repo=$(cd "$(dirname "$0")/.." && pwd)
stage=$repo/data/colab_finetune
session=${COLAB_SESSION:-intensifier-ft}
gpu=${COLAB_GPU:-T4}
out=data/output/intensifier_data_model_$(date +%F)
mkdir -p "$stage"
cd "$repo"
. train/lib.sh
need_data
pack "$stage/job.tgz" "bash train/job.sh \"\$1\" $out $(quote_args "$@")" pipeline train data/finetune data/text

have_model=0
fetch_model() {
  colab download -s "$session" /content/model.tgz "$stage/model.tgz" && install_model "$stage/model.tgz"
}
start() { colab_start; }
on_tick() {
  echo "$(date +%T) $(grep -a -v -e '^\s*$' "$stage/job.log" | tail -n 1)"
  if [ $have_model = 0 ] && grep -aqx TRAINED "$stage/job.log"; then
    fetch_model && have_model=1
  fi
  true
}
on_lost() {
  # training already finished and the model is local: only predict is left
  [ $have_model = 1 ] && { echo "VM lost after training; model is in models/intensifier, run predict.py"; return 1; }
  true
}
on_failed() {
  echo "job failed, log in $stage/job.log; model saved: $have_model; VM left running for inspection: colab stop -s $session"
}
on_done() {
  [ $have_model = 1 ] || fetch_model || { echo "model download failed; VM left running: colab stop -s $session"; exit 1; }
  colab download -s "$session" /content/results.tgz "$stage/results.tgz"
  tar xzf "$stage/results.tgz" -C "$repo"
  colab stop -s "$session"
  echo "done: models/intensifier ($(du -sh models/intensifier | cut -f1)), $out.sqlite, log $stage/job.log"
}

start
colab_watch
