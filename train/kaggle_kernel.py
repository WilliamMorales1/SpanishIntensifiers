# Kaggle kernel body of kaggle.sh and kaggle_sweep.sh: unpacks the job tarball from the attached dataset and runs
# its cmd.sh with /kaggle/working as work dir; the output is also kept in /kaggle/working/job.log.
import glob
import os
import shutil
import subprocess
import sys
import tarfile

work, proj = "/kaggle/working", "/tmp/proj"
os.makedirs(proj, exist_ok=True)
tgz = glob.glob("/kaggle/input/**/job.tgz", recursive=True)
if tgz:
    with tarfile.open(tgz[0]) as t:
        t.extractall(proj, filter="data")
else:  # Kaggle unpacked the archive on upload
    src = os.path.dirname(glob.glob("/kaggle/input/**/cmd.sh", recursive=True)[0])
    shutil.copytree(src, proj, dirs_exist_ok=True)
os.chdir(proj)
proc = subprocess.Popen(["bash", "cmd.sh", work], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
with open(f"{work}/job.log", "w") as log:
    for line in proc.stdout:
        log.write(line)
        log.flush()
        print(line, end="", flush=True)
sys.exit(proc.wait())
