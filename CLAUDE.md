# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

This is a small educational PyTorch project implementing LeNet-5 for MNIST handwritten digit classification. It targets Apple Silicon Macs (MPS) but also supports CUDA and CPU.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Unified CLI (recommended) — use -m to select model
python main.py train                  # Train LeNet-5 (default)
python main.py train -m cnn           # Train modern CNN
python main.py evaluate               # Evaluate on test set
python main.py evaluate -m cnn        # Evaluate CNN
python main.py demo --samples 5       # Demo predictions
python main.py predict IMAGE_PATH     # Predict custom image
python main.py predict -m cnn IMAGE   # Predict with CNN

# Or run individual scripts directly
python train.py
python inference.py
python cnn.py
```

There is no test framework or linting setup in this project. `requirements.txt` includes `torchaudio` but the project doesn't use it — only `torch`, `torchvision`, `numpy`, `matplotlib`, and `Pillow` are needed.

## Architecture

**`lenet5_model.py`** — LeNet-5 model definition. Classic architecture: Conv1(1→6) → AvgPool → Conv2(6→16) → AvgPool → FC(400→120) → FC(120→84) → FC(84→10). Uses `tanh` activation per the original 1998 paper. Input expects 32×32 grayscale images.

**`train.py`** — Unified model-agnostic training module. `load_data()` accepts `input_size` (32 for LeNet-5, 28 for CNN). `train_model()` supports `optimizer_name` ('adam' or 'adadelta') and optional StepLR scheduler. `save_model()` accepts `architecture_name` for unified dict-format checkpoint (`{'model_state_dict': ..., 'model_architecture': ...}`). Generates `training_history.png`.

**`inference.py`** — Unified model-agnostic inference module. `load_trained_model()` accepts `model_class` to load either LeNet-5 or CNN checkpoints. `preprocess_image()` accepts `input_size`. Also contains interactive `main()` for standalone use.

**`cnn.py`** — Modern CNN model definition (`Net` class). Conv(1→32,3x3)→ReLU→Conv(32→64,3x3)→ReLU→MaxPool2d→Dropout(0.25)→FC(9216→128)→ReLU→Dropout(0.5)→FC(128→10). Accepts 28×28 input. Outputs raw logits (unified with LeNet-5 to use CrossEntropyLoss).

**`main.py`** — Unified CLI entry point with subcommands: `train`, `evaluate`, `demo`, `predict`. All accept `--model lenet5|cnn` to switch architecture. Uses `MODEL_CONFIGS` dict for per-model defaults (input size, optimizer, save path).

**`test.py`** — Simple one-off script to check MPS device availability.

## Device selection pattern

All scripts use the `get_device()` helper with priority: MPS → CUDA → CPU. The helper is defined in both `train.py` and `inference.py`, and separately in `main.py` (since main.py dispatches without importing those modules at top level).

## Data conventions

- MNIST data auto-downloaded to `./data/` by torchvision
- LeNet-5 expects 32×32 input (MNIST's native 28×28 gets resized); CNN expects 28×28
- Normalization: mean=0.1307, std=0.3081 (MNIST standard)
- Model checkpoint format (unified): `{'model_state_dict': ..., 'model_architecture': 'LeNet5'|'CNN'}`
- Both LeNet5 and Net accept `num_classes=10` constructor parameter

## Potential pitfalls

- `DataLoader` uses `num_workers=2` — on some macOS versions this can cause crashes. If training hangs or crashes, set `num_workers=0`.
- Both models now output raw logits + use `CrossEntropyLoss`. Do not use `LogSoftmax` + `NLLLoss`.
