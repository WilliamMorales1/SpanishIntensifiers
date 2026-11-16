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
[ -f data/finetune/train.jsonl.gz ] || { echo "no data/finetune/train.jsonl.gz: build it with pipeline/dataset.py"; exit 1; }
tar czf "$stage/job.tgz" pipeline train data/finetune data/text

# printf with no arguments would emit '' as an empty argument
train_args=
[ $# -gt 0 ] && train_args=$(printf '%q ' "$@")
cat > "$stage/job.sh" <<SH
cd /content/proj
python -u train/train.py --out models/intensifier $train_args || { echo FAILED; touch /content/FAILED; exit 1; }
tar czf /content/model.tgz models/intensifier && touch /content/TRAINED
python -u pipeline/predict.py --model models/intensifier --text data/text --output $out || { echo FAILED; touch /content/FAILED; exit 1; }
tar czf /content/results.tgz data/output
echo DONE
touch /content/DONE
SH
cat > "$stage/setup_vm.py" <<'PY'
import subprocess
cmd = """cd /content && mkdir -p proj && tar xzf job.tgz -C proj
pip install -q stanza==1.15.0
nohup bash job.sh > job.log 2>&1 &"""
print(subprocess.run(cmd, shell=True, capture_output=True, text=True, executable="/bin/bash").stdout[-500:])
PY
cat > "$stage/status.py" <<'PY'
import os
import subprocess
print(subprocess.run("tail -n 3 /content/job.log", shell=True, capture_output=True, text=True).stdout)
# marker files, since the log tail scrolls past TRAINED during predict
print("MARKERS", *[m for m in ("TRAINED", "DONE", "FAILED") if os.path.exists("/content/" + m)])
PY

start() {
  colab stop -s "$session" 2>/dev/null || true
  colab new -s "$session" --gpu "$gpu"
  colab upload -s "$session" "$stage/job.tgz" /content/job.tgz
  colab upload -s "$session" "$stage/job.sh" /content/job.sh
  # the client can time out on setup while the VM keeps going
  colab exec -s "$session" -f "$stage/setup_vm.py" || echo "setup exec ended early"
}

# pulls the trained model to models/intensifier; a previous model is kept as models/intensifier.old
fetch_model() {
  colab download -s "$session" /content/model.tgz "$stage/model.tgz" || return 1
  tar tzf "$stage/model.tgz" models/intensifier/model.pt > /dev/null || return 1
  if [ -d models/intensifier ]; then
    rm -rf models/intensifier.old
    mv models/intensifier models/intensifier.old
  fi
  tar xzf "$stage/model.tgz" -C "$repo"
  echo "$(date +%T) model saved locally: models/intensifier ($(du -sh models/intensifier | cut -f1))"
}

start
restarts=0
fails=0
have_model=0
while true; do
  sleep 300
  if timeout 300 colab exec -s "$session" -f "$stage/status.py" > "$stage/status.txt" 2>&1; then
    fails=0
    echo "$(date +%T) $(grep -a -v -e '^\s*$' -e '^MARKERS' "$stage/status.txt" | tail -n 1)"
  else
    fails=$((fails + 1))
    echo "$(date +%T) status failed ($fails)"
    if [ $fails -ge 3 ] && ! colab sessions 2>&1 | grep -q "$session"; then
      restarts=$((restarts + 1))
      [ $restarts -gt 3 ] && { echo "VM lost $restarts times, giving up"; exit 1; }
      # training already finished and the model is local: only predict is left
      [ $have_model = 1 ] && { echo "VM lost after training; model is in models/intensifier, run predict.py"; exit 1; }
      echo "VM lost, restarting ($restarts)"
      start
      fails=0
    fi
    continue
  fi
  markers=$(grep -a '^MARKERS' "$stage/status.txt" || true)
  if [ $have_model = 0 ] && [[ $markers == *TRAINED* ]]; then
    fetch_model && have_model=1
  fi
  if [[ $markers == *FAILED* ]]; then
    colab download -s "$session" /content/job.log "$stage/job.log" || true
    echo "job failed, log in $stage/job.log; model saved: $have_model; VM left running for inspection: colab stop -s $session"
    exit 1
  fi
  if [[ $markers == *DONE* ]]; then
    [ $have_model = 1 ] || fetch_model || { echo "model download failed; VM left running: colab stop -s $session"; exit 1; }
    colab download -s "$session" /content/results.tgz "$stage/results.tgz"
    colab download -s "$session" /content/job.log "$stage/job.log"
    tar xzf "$stage/results.tgz" -C "$repo"
    colab stop -s "$session"
    echo "done: models/intensifier ($(du -sh models/intensifier | cut -f1)), $out.sqlite, log $stage/job.log"
    exit 0
  fi
done
