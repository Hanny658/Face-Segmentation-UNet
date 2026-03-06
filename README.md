# Lightweight Face Parsing (CelebAMask-style) - PyTorch Prototype

This project is a clean prototype for semantic face parsing with strict constraints:

- train only on provided `data/train`
- no external data, no pretrained weights, random initialization
- no ensemble
- lightweight model under `1,821,085` trainable parameters
- default input resolution: `512 x 512`

## Project Structure

```text
.
├─ train.py
├─ validate.py
├─ infer.py
├─ count_params.py
├─ requirements.txt
├─ README.md
├─ config.yaml
├─ src/
│  ├─ datasets/
│  │  ├─ celebamask_dataset.py
│  │  └─ transforms.py
│  ├─ models/
│  │  ├─ blocks.py
│  │  ├─ lightweight_unet.py
│  │  └─ attention.py
│  ├─ losses/
│  │  ├─ dice.py
│  │  └─ segmentation_loss.py
│  ├─ engine/
│  │  ├─ trainer.py
│  │  ├─ evaluator.py
│  │  └─ inference.py
│  └─ utils/
│     ├─ metrics.py
│     ├─ checkpoint.py
│     ├─ plotting.py
│     ├─ seed.py
│     └─ param_count.py
└─ experiments/
```

## Data Layout

Expected layout:

```text
data/
├─ train/
│  ├─ images/
│  └─ masks/
└─ val/
   └─ images/
```

Optional labeled validation is supported with:

```text
data/val/masks/
```

## Setup

```bash
pip install -r requirements.txt
```

## Default Training Recipe

- Loss: `CrossEntropy + 0.5 * Dice`
- Optimizer: `AdamW(lr=3e-4, weight_decay=5e-4)`
- Scheduler: cosine annealing
- Epochs: `100`
- Batch size: `10`
- AMP: enabled
- Augmentations:
  - random resized crop
  - random horizontal flip
  - random rotation (+/- 15 deg)
  - color jitter
  - gaussian blur
- Internal validation split from train set by default (`val_split=0.1`)
- Validation F-measure curve is saved to `experiments/<run>/val_f1_curve.png` when validation is enabled

## Run

1. Count parameters

```bash
python count_params.py --config config.yaml
```

2. Train

```bash
python train.py --config config.yaml
```

3. Evaluate checkpoint

```bash
python validate.py --config config.yaml --checkpoint experiments/baseline/best.pt --source internal
```

If `data/val/masks` exists:

```bash
python validate.py --config config.yaml --checkpoint experiments/baseline/best.pt --source val
```

4. Infer on unlabeled val images

```bash
python infer.py --config config.yaml --checkpoint experiments/baseline/best.pt --output-dir outputs --tta-flip
```

Predictions are saved as indexed masks (default `.png`) in `outputs/`.

## Slurm Job Scripts

Two submit-ready scripts are included under `experiments/`:

- `experiments/env_setup.batch`: loads modules, creates virtualenv, installs requirements
- `experiments/train.batch`: trains and stores metrics/plot/checkpoints in one output folder

### 1) Environment setup

```bash
sbatch experiments/env_setup.batch
```

Optional overrides:

```bash
sbatch --export=ALL,VENV_DIR=/path/to/venv experiments/env_setup.batch
```

### 2) Training job

```bash
sbatch experiments/train.batch config.yaml
```

Optional overrides:

```bash
sbatch --export=ALL,RUN_NAME=my_run,TRAIN_ARGS="--epochs 50 --batch-size 8 --num-workers 4" experiments/train.batch config.yaml
```

All key outputs are written into the same run directory, default:

- `experiments/run_<SLURM_JOB_ID>/best.pt`
- `experiments/run_<SLURM_JOB_ID>/last.pt`
- `experiments/run_<SLURM_JOB_ID>/history.json`
- `experiments/run_<SLURM_JOB_ID>/val_f1_curve.png` (if validation is enabled)
- `experiments/run_<SLURM_JOB_ID>/val_metrics.txt`
- `experiments/run_<SLURM_JOB_ID>/param_count.txt`
