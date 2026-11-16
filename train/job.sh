#!/bin/bash
# runs on the Colab or Kaggle VM from the unpacked project: trains, packs the model, predicts over data/text.
# Prints TRAINED, then DONE or FAILED, as whole lines for colab.sh and kaggle.sh to watch.
# usage: train/job.sh <work dir> <output name> [train.py args]
work=$1
out=$2
shift 2
fail() {
  echo FAILED
  exit 1
}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
python -u train/train.py --out models/intensifier "$@" || fail
tar czf "$work/model.tgz" models/intensifier || fail
echo TRAINED
pip install -q stanza==1.15.0 || fail
python -u pipeline/predict.py --model models/intensifier --text data/text --output "$out" || fail
tar czf "$work/results.tgz" data/output || fail
echo DONE
