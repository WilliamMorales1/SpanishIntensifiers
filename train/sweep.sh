#!/bin/bash
# trains several train.py configs one after another on one Colab GPU and keeps only their metrics.json
# (models/sweep/<name>/), to pick settings on the dev split. Configs come from a file of "name args..." lines.
# Finished configs are skipped, so rerunning after a lost VM resumes. Start it in tmux.
# usage: train/sweep.sh configs.txt; WATCH=1 only follows a job already running in the session
set -e
repo=$(cd "$(dirname "$0")/.." && pwd)
configs=$(realpath "$1")
stage=$repo/data/colab_sweep
session=${COLAB_SESSION:-intensifier-sweep}
gpu=${COLAB_GPU:-T4}
mkdir -p "$stage" "$repo/models/sweep"
cd "$repo"
. train/lib.sh

pull() {
  colab download -s "$session" /content/sweep.tgz "$stage/sweep.tgz" > /dev/null 2>&1 && tar xzf "$stage/sweep.tgz" -C "$repo" || true
}
start() {
  cp "$configs" "$stage/configs.txt"
  pack "$stage/job.tgz" "python -u train/sweep.py configs.txt --tar \"\$1/sweep.tgz\"" \
    pipeline train data/finetune/*.json data/finetune/*.jsonl.gz models/sweep -C "$stage" configs.txt
  colab_start
}
on_tick() {
  echo "$(date +%T) $(grep -a -e '^===' -e FAILED -e 'dev host' "$stage/job.log" | tail -n 1)"
  pull
}
on_lost() { echo "done so far: $(ls models/sweep | tr '\n' ' ')"; }
on_failed() { :; }
on_done() {
  pull
  colab stop -s "$session"
  echo "done: models/sweep ($(ls models/sweep | tr '\n' ' ')), log $stage/job.log"
}

[ -n "$WATCH" ] || start
colab_watch
