# shared helpers of the train/ runners; source after setting repo, stage (and session, gpu or user) and cd "$repo"

need_data() {
  [ -f "${1:-data/finetune}/train.jsonl.gz" ] || { echo "no ${1:-data/finetune}/train.jsonl.gz: build it with pipeline/dataset.py"; exit 1; }
}

# quote_args prints its arguments shell-quoted; printf with no arguments would emit '' as an empty argument
quote_args() {
  [ $# -gt 0 ] && printf '%q ' "$@"
  true
}

# pack <tgz> <cmd.sh body> <paths...> tars the paths plus $stage/cmd.sh, which the VM runs as `bash cmd.sh <work dir>`
pack() {
  local tgz=$1
  printf '%s\n' "$2" > "$stage/cmd.sh"
  shift 2
  tar czf "$tgz" "$@" -C "$stage" cmd.sh
}

# install_model <tgz> unpacks a model.tgz into models/intensifier; a previous model is kept as models/intensifier.old
install_model() {
  tar tzf "$1" models/intensifier/model.pt > /dev/null || return 1
  if [ -d models/intensifier ]; then
    rm -rf models/intensifier.old
    mv models/intensifier models/intensifier.old
  fi
  tar xzf "$1" -C "$repo"
  echo "$(date +%T) model saved locally: models/intensifier ($(du -sh models/intensifier | cut -f1))"
}

# colab_start opens a fresh $session VM, uploads $stage/job.tgz and starts its cmd.sh in the background
colab_start() {
  colab stop -s "$session" 2>/dev/null || true
  colab new -s "$session" --gpu "$gpu"
  colab upload -s "$session" "$stage/job.tgz" /content/job.tgz
  # the client can time out on setup while the VM keeps going
  colab exec -s "$session" -f "$repo/train/colab_setup.py" || echo "setup exec ended early"
}

# colab_watch polls the job log into $stage/job.log every 5 min and calls the caller's hooks: on_tick each poll,
# then on_done or on_failed on a DONE or FAILED line (exits with their status). A lost VM is replaced up to
# 3 times by the caller's start, unless on_lost returns non-zero.
colab_watch() {
  local restarts=0 fails=0
  while true; do
    sleep 300
    # file downloads work even when the kernel is busy, unlike exec
    if timeout 300 colab download -s "$session" /content/job.log "$stage/job.log" > /dev/null 2>&1; then
      fails=0
      on_tick
      if grep -aqx FAILED "$stage/job.log"; then
        on_failed
        exit 1
      fi
      if grep -aqx DONE "$stage/job.log"; then
        on_done
        exit 0
      fi
      continue
    fi
    fails=$((fails + 1))
    echo "$(date +%T) status failed ($fails)"
    if [ $fails -ge 3 ] && ! colab sessions 2>&1 | grep -q "$session"; then
      restarts=$((restarts + 1))
      [ $restarts -gt 3 ] && { echo "VM lost $restarts times, giving up"; exit 1; }
      on_lost || exit 1
      echo "VM lost, restarting ($restarts)"
      start
      fails=0
    fi
  done
}

# kaggle_push <name> uploads $stage/data (job.tgz) as private dataset $user/<name>-data and runs
# train/kaggle_kernel.py on it as private GPU kernel $user/<name>
kaggle_push() {
  local name=$1 data=$user/$1-data
  mkdir -p "$stage/kernel"
  cat > "$stage/data/dataset-metadata.json" <<JSON
{"title": "$name-data", "id": "$data", "licenses": [{"name": "unknown"}]}
JSON
  if kaggle datasets status "$data" > /dev/null 2>&1; then
    kaggle datasets version -p "$stage/data" -m "$name $(date +%F_%T)" -q
  else
    kaggle datasets create -p "$stage/data" -q
  fi
  until kaggle datasets status "$data" 2>/dev/null | grep -q ready; do sleep 20; done
  cp "$repo/train/kaggle_kernel.py" "$stage/kernel/"
  cat > "$stage/kernel/kernel-metadata.json" <<JSON
{"id": "$user/$name", "title": "$name", "code_file": "kaggle_kernel.py", "language": "python",
 "kernel_type": "script", "is_private": true, "enable_gpu": true, "enable_internet": true,
 "dataset_sources": ["$data"], "competition_sources": [], "kernel_sources": []}
JSON
  kaggle kernels push -p "$stage/kernel" --accelerator NvidiaTeslaT4 -t 43200
}

# kaggle_wait <kernel> polls every 5 min until the kernel ends, leaves its status in $status
# and its output in $stage/out
kaggle_wait() {
  while true; do
    sleep 300
    status=$(kaggle kernels status "$1" 2>&1 | grep -o 'status "[^"]*"' || echo unknown)
    echo "$(date +%T) $status"
    case $status in
      *COMPLETE*|*ERROR*|*CANCEL*) break ;;
    esac
  done
  rm -rf "$stage/out" && mkdir -p "$stage/out"
  kaggle kernels output "$1" -p "$stage/out" -q > /dev/null 2>&1 || true
}
