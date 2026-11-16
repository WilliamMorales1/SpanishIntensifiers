# Kaggle kernel body for kaggle.sh: trains with the arguments in train_args.txt, then runs predict.py over
# data/text; /kaggle/working gets model.tgz, output/ (from data/output) and job.log.
import glob
import os
import shlex
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

work, proj = "/kaggle/working", "/tmp/proj"
os.makedirs(proj, exist_ok=True)
tgz = glob.glob("/kaggle/input/**/job.tgz", recursive=True)
if tgz:
    with tarfile.open(tgz[0]) as t:
        t.extractall(proj, filter="data")
else:  # Kaggle unpacked the archive on upload
    src = os.path.dirname(glob.glob("/kaggle/input/**/train_args.txt", recursive=True)[0])
    shutil.copytree(src, proj, dirs_exist_ok=True)
os.chdir(proj)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
train_args = shlex.split(Path("train_args.txt").read_text())
out = Path("out_name.txt").read_text().strip()


def run(cmd):
    print("===", *cmd, flush=True)
    with open(f"{work}/job.log", "a") as log:
        code = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT)
    if code:
        print("FAILED", code, flush=True)
        sys.exit(1)


run(["python", "-u", "train/train.py", "--out", "models/intensifier", *train_args])
with tarfile.open(f"{work}/model.tgz", "w:gz") as t:
    t.add("models/intensifier")
print("TRAINED", flush=True)
run(["pip", "install", "-q", "stanza==1.15.0"])
run(
    [
        "python",
        "-u",
        "pipeline/predict.py",
        "--model",
        "models/intensifier",
        "--text",
        "data/text",
        "--output",
        out,
    ]
)
shutil.copytree("data/output", f"{work}/output", dirs_exist_ok=True)
print("DONE", flush=True)
