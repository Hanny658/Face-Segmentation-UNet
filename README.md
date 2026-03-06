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
