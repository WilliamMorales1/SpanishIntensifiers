#!/bin/bash
# trains several train.py configs one after another on one Colab GPU and keeps only their metrics.json
# (models/sweep/<name>/), to pick settings on the dev split. Configs come from a file of "name args..." lines.
# Finished configs are skipped, so rerunning after a lost VM resumes. Start it in tmux.
# usage: scripts/finetune/sweep.sh configs.txt; WATCH=1 only follows a job already running in the session
set -e
repo=$(cd "$(dirname "$0")/../.." && pwd)
configs=$(realpath "$1")
stage=$repo/data/colab_sweep
session=${COLAB_SESSION:-intensifier-sweep}
gpu=${COLAB_GPU:-T4}
mkdir -p "$stage" "$repo/models/sweep"
cd "$repo"

pack() {
  cp "$configs" "$stage/configs.txt"
  tar czf "$stage/job.tgz" pipeline scripts/finetune data/finetune/*.json data/finetune/*.jsonl.gz models/sweep
  cat > "$stage/job.sh" <<'SH'
cd /content/proj
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while read -r name args; do
  [ -z "$name" ] && continue
  [[ $name == \#* ]] && continue
  [ -f "models/sweep/$name/metrics.json" ] && continue
  echo "=== $name $args"
  if python -u scripts/finetune/train.py --out "tmp/$name" $args; then
    mkdir -p "models/sweep/$name"
    cp "tmp/$name/metrics.json" "tmp/$name/labels.json" "models/sweep/$name/"
  else
    echo "$name FAILED"
  fi
  rm -rf "tmp/$name"
  tar czf /content/sweep.tgz models/sweep
done < configs.txt
echo DONE
touch /content/DONE
SH
}

start() {
  pack
  colab stop -s "$session" 2>/dev/null || true
  colab new -s "$session" --gpu "$gpu"
  colab upload -s "$session" "$stage/job.tgz" /content/job.tgz
  colab upload -s "$session" "$stage/job.sh" /content/job.sh
  colab upload -s "$session" "$stage/configs.txt" /content/configs.txt
  cat > "$stage/setup_vm.py" <<'PY'
import subprocess
# the background job must not hold this cell's stdout, or the kernel stays busy
cmd = """cd /content && mkdir -p proj && tar xzf job.tgz -C proj && cp configs.txt proj/
nohup bash job.sh > job.log 2>&1 &"""
print(subprocess.run(cmd, shell=True, capture_output=True, text=True, executable="/bin/bash").stdout[-500:])
PY
  colab exec -s "$session" -f "$stage/setup_vm.py" || echo "setup exec ended early"
}

pull() {
  colab download -s "$session" /content/sweep.tgz "$stage/sweep.tgz" 2>/dev/null && tar xzf "$stage/sweep.tgz" -C "$repo" || true
}

[ -n "$WATCH" ] || start
restarts=0
fails=0
while true; do
  sleep 300
  # file downloads work even when the kernel is busy, unlike exec
  if timeout 300 colab download -s "$session" /content/job.log "$stage/job.log" > /dev/null 2>&1; then
    fails=0
    echo "$(date +%T) $(grep -a -e '^===' -e FAILED -e 'dev host' "$stage/job.log" | tail -n 1)"
    pull
  else
    fails=$((fails + 1))
    echo "$(date +%T) status failed ($fails)"
    if [ $fails -ge 3 ] && ! colab sessions 2>&1 | grep -q "$session"; then
      restarts=$((restarts + 1))
      [ $restarts -gt 3 ] && { echo "VM lost $restarts times, giving up"; exit 1; }
      echo "VM lost, restarting ($restarts); done so far: $(ls models/sweep | tr '\n' ' ')"
      start
      fails=0
    fi
    continue
  fi
  if grep -aqx DONE "$stage/job.log"; then
    pull
    colab stop -s "$session"
    echo "done: models/sweep ($(ls models/sweep | tr '\n' ' ')), log $stage/job.log"
    exit 0
  fi
done
