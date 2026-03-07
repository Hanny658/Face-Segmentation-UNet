# Lightweight Face Parsing (CelebAMask-style) - PyTorch Prototype

This project is a clean prototype for semantic face parsing with strict constraints:

- train only on provided `data/train`
- no external data, no pretrained weights, random initialization
- no ensemble
- lightweight model under `1,821,085` trainable parameters
- default input resolution: `512 x 512`
- model: MobileNetV2-style encoder + UNet decoder (FPN decoder also supported via config)

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
│  │  ├─ bisenet.py
│  │  ├─ pidnet.py
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
│     ├─ class_weights.py
│     ├─ class_names.py
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
- Dice defaults to `present-only` averaging (only GT-present classes in a batch)
- Optional boundary regularization is enabled by default:
  - total = `CE + 0.5*Dice + 0.05*Boundary` (after warmup)
  - warmup: first `8` epochs without boundary term
- Weighted CE (class-aware) is supported from `experiments/mask_stats.json`
- Optimizer: `AdamW(lr=3e-4, weight_decay=5e-4)`
- Scheduler: cosine annealing
- Epochs: `100`
- Batch size: `10`
- AMP: enabled
- Augmentations:
  - random resized crop
  - random horizontal flip (with left/right label swap for paired classes)
  - random rotation (+/- 15 deg)
  - color jitter
  - gaussian blur
- Internal validation split from train set by default (`val_split=0.1`)
- Validation F-score curve is saved to `experiments/<run>/val_fscore_curve.png` when validation is enabled

### Encoder/Decoder Options

`model.encoder_type` supports:

- `mobilenetv2` (default)
- `resnet` (light residual backbone)

`model.decoder_type` supports:

- `unet` (default)
- `fpn`

For a submit-ready residual+FPN variant under the parameter cap, use:

```bash
python train.py --config experiments/config_residual_fpn.yaml
```

For a BiSeNetV2-style variant under the parameter cap, use:

```bash
python train.py --config experiments/config_bisenet.yaml
```

For a PIDNet-style variant under the parameter cap, use:

```bash
python train.py --config experiments/config_pid.yaml
```

`config_pid.yaml` uses a closer-to-paper PIDNet setup:

- `pidnet.m / pidnet.n / pidnet.planes / pidnet.ppm_planes / pidnet.head_planes`
- `pidnet.augment: true` (enables auxiliary P and D heads)
- `loss.pid_aux` controls auxiliary loss weights for training:
  - `p_weight`: segmentation aux head weight
  - `d_weight`: boundary aux head weight (BCE on boundary map)

## Run

1. Count parameters

```bash
python count_params.py --config config.yaml
```

2. Train

```bash
python train.py --config config.yaml
```

If `train.run_val_data: true`, after training finishes the script will:

- load `best.pt`
- run inference on `data/val/images`
- write predictions to `data/val/masks`
- also write palette masks to `data/val/masks-palette` (when `inference.save_palette: true`)

3. Evaluate checkpoint

```bash
python validate.py --config config.yaml --checkpoint experiments/baseline/best.pt --source internal
```

`validate.py` prints:

- overall loss / pixel accuracy / mean F-score
- per-class F-score with class names from `data.class_names` in `config.yaml`

If `data/val/masks` exists:

```bash
python validate.py --config config.yaml --checkpoint experiments/baseline/best.pt --source val
```

4. Infer on unlabeled val images

```bash
python infer.py --config config.yaml --checkpoint experiments/baseline/best.pt --output-dir outputs --tta-flip
```

Predictions are saved as indexed masks (default `.png`) in `outputs/`.
When `--tta-flip` is used, left/right class channels are swapped back before averaging logits.
If `inference.save_palette: true`, palette PNGs are additionally saved to `inference.palette_output_dir`.

## Mask Label Statistics

To compute exact label IDs and class proportions from `data/train/masks`:

```bash
python analyze_masks.py --config config.yaml --masks-dir data/train/masks
```

Outputs:

- `experiments/mask_stats.json`
- `experiments/mask_stats.csv`

Recommended weights for weighted CE are in:

- `weights.median_freq_norm_mean1` inside `mask_stats.json`

Current training reads weighted CE settings from `config.yaml`:

- `loss.ce_weighting.enabled`
- `loss.ce_weighting.stats_json`
- `loss.dice_present_only`
- `loss.boundary.enabled / weight / pos_weight / warmup_epochs / pred_scale`
- `loss.ce_weighting.key` (for example `recommended_weighted_ce.weights`)

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
- `experiments/run_<SLURM_JOB_ID>/val_fscore_curve.png` (if validation is enabled)
- `experiments/run_<SLURM_JOB_ID>/val_metrics.txt`
- `experiments/run_<SLURM_JOB_ID>/param_count.txt`
