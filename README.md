# UNO-VC

Author: Huang, S.

This folder keeps the original flat Python structure and procedural training
logic. It contains base k-fold training, transfer learning, and a small
prediction demonstration for both directions.

The full training datasets are not included. The two example files contain one
held-out GVDA record each: input, observation, and the prediction saved by the
original workflow.

## Files

- `VC_uno1d.py`: original UNO-VC model and loss classes.
- `data_loader.py`: MATLAB v5/v7.3 reader.
- `train_VC.py`: original base-training loop.
- `transfer_VC.py`: original transfer-learning loop.
- `train_base.py`: ten-fold base-training entry script.
- `train_transfer.py`: transfer-learning entry script.
- `prediction_demo.ipynb`: load an example and model, predict, compare, and plot.
- `verify_prediction.py`: repeatable prediction regression check.

## Environment

The validated environment is Python 3.9, PyTorch 1.11.0, and CUDA 11.3.

```bash
conda env create -f environment.yml
conda activate uno-vc
```

The running WSL2 Docker used for validation has an RTX 3090 with 24 GiB VRAM.

## Prediction demonstration

Open `prediction_demo.ipynb` and set:

```python
DIRECTION = "upward"  # or "downward"
```

The direction selects both the example data and its matching trained model.

Run the same check without a notebook:

```bash
python verify_prediction.py --device cuda
```

A small CPU check is also possible when memory is available:

```bash
python verify_prediction.py --device cpu --direction upward --skip-hash
```

## Base training

Edit the settings at the top of `train_base.py`, place the MATLAB file under
`data/`, and run:

```bash
python train_base.py
```

The default is ten-fold training. The legacy trainer expects a test loader, so
the base script passes the current validation fold in that position; its
reported test value is therefore not an independent test result.

## Transfer learning

Edit `DIRECTION` and `DATA_FILES` at the top of `train_transfer.py`, then run:

```bash
python train_transfer.py
```

Defaults preserve the established workflow: decoder/output fine-tuning,
trainable normalization affine parameters, layerwise learning rates, source to
target amplitude-bias correction, waveform augmentation, and 150 epochs.

## GitHub model storage

Each model is about 1.46 GiB. Install Git LFS before the first `git add`:

```bash
git lfs install
git lfs track "models/*.pt"
```

Check your GitHub LFS quota before pushing. No software license has been chosen
yet, so this folder intentionally does not contain a `LICENSE` file.
