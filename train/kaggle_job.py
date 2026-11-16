# Kaggle kernel body for kaggle_sweep.sh: unpacks the job tarball from the attached dataset and trains each config
# of configs.txt, one per T4 at a time; only metrics.json, labels.json and the log are kept in /kaggle/working/sweep.
import glob
import os
import queue
import shutil
import subprocess
import tarfile
import threading
from pathlib import Path

import torch

work, proj = "/kaggle/working", "/tmp/proj"
os.makedirs(proj, exist_ok=True)
tgz = glob.glob("/kaggle/input/**/job.tgz", recursive=True)
if tgz:
    with tarfile.open(tgz[0]) as t:
        t.extractall(proj, filter="data")
else:  # Kaggle unpacked the archive on upload
    src = os.path.dirname(glob.glob("/kaggle/input/**/configs.txt", recursive=True)[0])
    shutil.copytree(src, proj, dirs_exist_ok=True)
os.chdir(proj)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

todo = queue.Queue()
for line in Path("configs.txt").read_text().splitlines():
    if not line.strip():
        continue
    name, *args = line.split()
    if name.startswith("#") or os.path.exists(f"models/sweep/{name}/metrics.json"):
        continue
    todo.put((name, args))


def worker(gpu):
    while True:
        try:
            name, args = todo.get_nowait()
        except queue.Empty:
            return
        out = f"{work}/sweep/{name}"
        os.makedirs(out, exist_ok=True)
        tmp = f"/tmp/{name}"
        print(f"=== gpu{gpu} {name} {' '.join(args)}", flush=True)
        with open(f"{out}/train.log", "w") as log:
            code = subprocess.call(
                ["python", "-u", "train/train.py", "--out", tmp, *args],
                stdout=log,
                stderr=subprocess.STDOUT,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)},
            )
        if code == 0:
            for f in ("metrics.json", "labels.json"):
                shutil.copy(f"{tmp}/{f}", out)
        print(f"{name} {'done' if code == 0 else f'FAILED ({code})'}", flush=True)
        shutil.rmtree(tmp, ignore_errors=True)


print("gpus", torch.cuda.device_count(), "configs", todo.qsize(), flush=True)
threads = [threading.Thread(target=worker, args=(g,)) for g in range(max(1, torch.cuda.device_count()))]
for t in threads:
    t.start()
for t in threads:
    t.join()
print("DONE", flush=True)
