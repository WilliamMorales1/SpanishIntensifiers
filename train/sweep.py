# trains each train.py config of a "name args..." file, one per GPU at a time, and keeps only its metrics.json and
# labels.json in <out>/<name>/; configs finished in <out> or --done are skipped, so a rerun resumes.
# usage: python train/sweep.py configs.txt [--out models/sweep] [--done dir] [--tar sweep.tgz]
import argparse
import contextlib
import os
import queue
import shutil
import subprocess
import sys
import tarfile
import tempfile
import threading

import torch

parser = argparse.ArgumentParser()
parser.add_argument("configs")
parser.add_argument("--out", default="models/sweep")
parser.add_argument("--done", help="another folder of finished configs")
parser.add_argument("--tar", help="tar of --out, rewritten after each config")
args = parser.parse_args()
train = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train.py")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def finished(name):
    return any(d and os.path.exists(os.path.join(d, name, "metrics.json")) for d in (args.out, args.done))


todo = queue.Queue()
with open(args.configs, encoding="utf-8") as f:
    for line in f:
        if not line.strip() or line.startswith("#"):
            continue
        name, *config = line.split()
        if not finished(name):
            todo.put((name, config))
gpus = max(1, torch.cuda.device_count())
tar_lock = threading.Lock()


def worker(gpu):
    while True:
        try:
            name, config = todo.get_nowait()
        except queue.Empty:
            return
        out = os.path.join(args.out, name)
        tmp = os.path.join(tempfile.gettempdir(), "sweep", name)
        os.makedirs(out, exist_ok=True)
        print(f"=== gpu{gpu} {name} {' '.join(config)}", flush=True)
        # one GPU streams its log; parallel logs would interleave
        with open(os.path.join(out, "train.log"), "w") if gpus > 1 else contextlib.nullcontext() as log:
            code = subprocess.call(
                [sys.executable, "-u", train, "--out", tmp, *config],
                stdout=log,
                stderr=subprocess.STDOUT if log else None,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)},
            )
        if code == 0:
            for f in ("metrics.json", "labels.json"):
                shutil.copy(os.path.join(tmp, f), out)
        print(f"{name} {'done' if code == 0 else f'FAILED ({code})'}", flush=True)
        shutil.rmtree(tmp, ignore_errors=True)
        if args.tar:
            with tar_lock, tarfile.open(args.tar, "w:gz") as t:
                t.add(args.out)


print("gpus", gpus, "configs", todo.qsize(), flush=True)
threads = [threading.Thread(target=worker, args=(g,)) for g in range(gpus)]
for t in threads:
    t.start()
for t in threads:
    t.join()
print("DONE", flush=True)
