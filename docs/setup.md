# Setup

Pick one. Every option installs `requirements.txt` (dependencies plus pytest and ruff). Nothing is installed as a package; the scripts run from the repo root. Needs Python 3.10+ and [LibreOffice](https://www.libreoffice.org/) for old `.doc` transcripts.

**conda** ([Miniforge](https://github.com/conda-forge/miniforge)):

```fish
conda create -n intensifier-research python=3.12
conda activate intensifier-research
pip install -r requirements.txt
```

`conda run -n intensifier-research ...` works too; add `--no-capture-output` when the command reads stdin or streams a log.

**uv**:

```fish
uv venv
uv pip install -r requirements.txt
uv run python -m pytest -q
```

**venv + pip**:

```fish
python -m venv .venv
source .venv/bin/activate.fish     # bash/zsh: source .venv/bin/activate
pip install -r requirements.txt
```

## CPU-only PyTorch

PyTorch's default Linux wheel bundles CUDA (~3 GB). Training needs a GPU anyway (Kaggle or Colab, [finetune.md](finetune.md)), and inference runs on a CPU, so on a machine without an NVIDIA GPU take the CPU wheel (~200 MB), as CI does:

```fish
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
uv pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

## Hardware

Training runs on a Kaggle T4 (`train/kaggle.sh`, ~70 min) or Colab (`train/colab.sh`); start either in tmux, they run in the foreground. Inference on a laptop CPU: the example interview takes ~40 s at a 5.6 GB peak, the whole corpus hours (`kaggle.sh` predicts over it after training). The model (`models/intensifier/`, 2.2 GB) is gitignored; `kaggle.sh` keeps the previous one as `models/intensifier.old/`.

## Changing dependencies

Edit `requirements.txt` only.

## Tests

```fish
python -m pytest -q
ruff check . && ruff format --check .
```

`data/examples/` doubles as an end-to-end check: rerunning the example commands in the README with the current model should reproduce `data/examples/intensifier_data.csv`.

Tests: transcript cleaning (speaker labels, interviewer turns, minute cut, normalization) in `test_clean_transcripts.py`; label records, decoding, the correction merge and row scoring, plus model chunking and lexicon, in `test_dataset.py`; speaker merge, syllables and sqlite/csv writers in `test_output.py`. CI runs both commands on every push and pull request.
