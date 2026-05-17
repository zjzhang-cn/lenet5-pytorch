# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

This is a small educational PyTorch project implementing LeNet-5 for MNIST handwritten digit classification. It targets Apple Silicon Macs (MPS) but also supports CUDA and CPU.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Unified CLI (recommended)
python main.py train              # Train LeNet-5
python main.py evaluate           # Evaluate on test set
python main.py demo               # Demo predictions on random samples
python main.py predict IMAGE_PATH # Predict custom image
python main.py train-cnn          # Train modern CNN

# Or run individual scripts directly
python lenet5_train.py
python lenet5_inference.py
python cnn.py
```

There is no test framework or linting setup in this project. `requirements.txt` includes `torchaudio` but the project doesn't use it — only `torch`, `torchvision`, `numpy`, `matplotlib`, and `Pillow` are needed.

## Architecture

**`lenet5_model.py`** — LeNet-5 model definition. Classic architecture: Conv1(1→6) → AvgPool → Conv2(6→16) → AvgPool → FC(400→120) → FC(120→84) → FC(84→10). Uses `tanh` activation per the original 1998 paper. Input expects 32×32 grayscale images.

**`lenet5_train.py`** — Training script. Loads MNIST via torchvision (auto-downloads to `./data/`), resizes from 28×28 to 32×32, normalizes with mean=0.1307/std=0.3081. Trains for 10 epochs with Adam (lr=0.001) and CrossEntropyLoss. Saves model as a dict with `model_state_dict` and `model_architecture` keys to `lenet5_mnist.pth`. Generates `training_history.png` with loss/accuracy/overfitting plots.

**`lenet5_inference.py`** — Interactive inference script. Loads the saved `.pth` checkpoint. Three modes: random MNIST test samples demo, custom image prediction (accepts file paths or numpy arrays, auto-converts RGB→grayscale), and full test-set evaluation with per-class accuracy.

**`cnn.py`** — A separate, unrelated modern CNN (`Net` class) with ReLU, MaxPool, Dropout, LogSoftmax+NLLLoss, and Adadelta optimizer. Uses argparse for configuration. Unlike the LeNet-5 scripts, it does not use the `get_device()` pattern — it defaults to CPU and requires `--use_gpu` to enable MPS. Accepts 28×28 input (no resize).

**`main.py`** — Unified CLI entry point with subcommands: `train`, `evaluate`, `demo`, `predict`, `train-cnn`. Dispatch-only, all logic lives in the existing modules.

**`test.py`** — Simple one-off script to check MPS device availability.

## Device selection pattern

The LeNet-5 scripts (`lenet5_train.py`, `lenet5_inference.py`) use a `get_device()` helper with priority: MPS → CUDA → CPU. Reuse this pattern for any new LeNet-5-related code.

`cnn.py` does NOT follow this pattern — it defaults to CPU and only uses MPS when `--use_gpu` is explicitly passed. This inconsistency means `cnn.py` won't auto-detect Apple Silicon.

## Data conventions

- MNIST data auto-downloaded to `./data/` by torchvision
- Input images must be 32×32 grayscale (MNIST's native 28×28 gets resized)
- Normalization: mean=0.1307, std=0.3081 (MNIST standard)
- Model checkpoint format: `{'model_state_dict': ..., 'model_architecture': 'LeNet5'}`
- Loading a checkpoint requires the `LeNet5` class to be importable (the checkpoint stores weights, not the model definition)

## Potential pitfalls

- `DataLoader` uses `num_workers=2` — on some macOS versions this can cause crashes. If training hangs or crashes, set `num_workers=0`.
- `cnn.py` uses LogSoftmax + NLLLoss, while LeNet-5 scripts use raw logits + CrossEntropyLoss. Don't mix the two approaches when moving code between files.
