# Colab VM setup of colab.sh and sweep.sh: unpacks /content/job.tgz into /content/proj and starts its cmd.sh.
import subprocess

# the background job must not hold this cell's stdout, or the kernel stays busy
cmd = """cd /content && mkdir -p proj && tar xzf job.tgz -C proj && cd proj
nohup bash cmd.sh /content > /content/job.log 2>&1 &"""
print(
    subprocess.run(cmd, check=False, shell=True, capture_output=True, text=True, executable="/bin/bash").stdout[-500:]
)
