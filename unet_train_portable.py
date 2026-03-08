from __future__ import annotations

"""
Usage: 
python unet_train_portable.py --mode train
python unet_train_portable.py --mode validate --checkpoint experiments/portable_unet/best.pt
python unet_train_portable.py --mode infer --checkpoint experiments/portable_unet/best.pt
python unet_train_portable.py --mode count-params
"""

import argparse
import json
import random
from copy import deepcopy
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFilter
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ColorJitter, RandomResizedCrop
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm


CFG = {
    "data": {
        "val_split": 0.1,
        "pin_memory": True,
    },
    "model": {"encoder_channels": [33, 64, 97, 126, 170], "expand_ratio": 4, "max_trainable_params": 1821085},
    "train": {
        "epochs": 100, "batch_size": 20, "drop_last": True, "use_amp": True, "use_internal_val": True,
        "run_val_data": True, "lr": 5e-4, "weight_decay": 5e-4, "min_lr": 3e-6,
    },
    "loss": {
        "dice_weight": 0.8, "dice_present_only": True, "ignore_index": None,
        "boundary": {"enabled": True, "weight": 0.1, "pos_weight": 4.0, "warmup_epochs": 8, "pred_scale": 4.0},
        "ce_weighting": {
            "enabled": True, "key": "recommended_weighted_ce.weights",
            "min_weight": 0.0, "max_weight": 6.0, "normalize_mean_one": False,
        },
    },
    "augmentation": {
        "hflip_prob": 0.5, "rotation_deg": 15, "resize_scale": [0.75, 1.05], "resize_ratio": [0.9, 1.1],
        "color_jitter": {"brightness": 0.3, "contrast": 0.2, "saturation": 0.2, "hue": 0.05, "prob": 0.75},
        "gaussian_blur": {"prob": 0.1, "sigma": [0.1, 1.5]},
    },
    "inference": {"batch_size": 20},
}

DATA_ROOT = Path("data")
TRAIN_IMAGES_DIR = DATA_ROOT / "train" / "images"
TRAIN_MASKS_DIR = DATA_ROOT / "train" / "masks"
VAL_IMAGES_DIR = DATA_ROOT / "val" / "images"
VAL_MASKS_DIR = DATA_ROOT / "val" / "masks"

INPUT_SIZE = 512
NUM_CLASSES = 19
CLASS_NAMES = [
    "background", "skin", "nose", "eye_g", "l_eye", "r_eye", "l_brow", "r_brow", "l_ear", "r_ear",
    "mouth", "u_lip", "l_lip", "hair", "hat", "ear_r", "neck_l", "neck", "cloth",
]
FLIP_PAIRS = ((4, 5), (6, 7), (8, 9))
NUM_WORKERS = 2
WEIGHT_STATS_JSON = Path("experiments") / "mask_stats.json"
SAVE_DIR = Path("experiments") / "portable_unet"
INFER_OUTPUT_DIR = Path("outputs") / "portable_unet"


def list_files(d, exts):
    d = Path(d)
    if not d.exists():
        raise FileNotFoundError(d)
    return sorted([p for p in d.iterdir() if p.is_file() and p.suffix.lower() in exts])


def pair_samples(img_dir, msk_dir, strict=True):
    imgs = list_files(img_dir, ".jpg")
    masks = {p.stem: p for p in list_files(msk_dir, ".png")}
    pairs, miss = [], []
    for p in imgs:
        m = masks.get(p.stem)
        if m is None:
            miss.append(p.name)
        else:
            pairs.append((p, m))
    if strict and miss:
        raise FileNotFoundError(f"missing masks: {len(miss)} examples: {', '.join(miss[:5])}")
    if not pairs:
        raise RuntimeError("no image-mask pairs")
    return pairs


def split_samples(samples, val_split, use_internal_val):
    s = list(samples)
    if not use_internal_val or val_split <= 0:
        return s, []
    n = len(s)
    nv = max(1, int(n * val_split))
    if nv >= n:
        nv = n - 1
    idx = list(range(n))
    random.shuffle(idx)
    val_idx = set(idx[:nv])
    tr = [s[i] for i in range(n) if i not in val_idx]
    va = [s[i] for i in range(n) if i in val_idx]
    return tr, va


def maybe_weights(cfg, num_classes):
    c = cfg["loss"]["ce_weighting"]
    if not c.get("enabled", False):
        return None
    p = WEIGHT_STATS_JSON
    if not p.exists():
        raise FileNotFoundError(p)
    j = json.loads(p.read_text(encoding="utf-8"))
    cur = j
    for k in str(c["key"]).split("."):
        cur = cur[k]
    w = np.asarray(cur, dtype=np.float32)
    if w.ndim != 1 or len(w) != num_classes:
        raise ValueError("weight shape mismatch")
    if c.get("min_weight", 0.0) > 0:
        w = np.maximum(w, float(c["min_weight"]))
    if c.get("max_weight", None) is not None:
        w = np.minimum(w, float(c["max_weight"]))
    if c.get("normalize_mean_one", False):
        m = float(w.mean())
        if m > 0:
            w = w / m
    return torch.tensor(w, dtype=torch.float32)


def normalize_palette(p):
    p = [int(x) & 255 for x in p]
    if len(p) < 768:
        p += [0] * (768 - len(p))
    elif len(p) > 768:
        p = p[:768]
    return p


def make_pascal_palette(n=256):
    pal = [0] * (n * 3)
    for j in range(n):
        lab, r, g, b, i = j, 0, 0, 0, 0
        while lab:
            r |= (((lab >> 0) & 1) << (7 - i))
            g |= (((lab >> 1) & 1) << (7 - i))
            b |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
        pal[3 * j + 0], pal[3 * j + 1], pal[3 * j + 2] = r, g, b
    return normalize_palette(pal)


def load_palette(masks_dir):
    d = Path(masks_dir)
    if not d.exists():
        return None
    for p in list_files(d, ".png"):
        try:
            with Image.open(p) as im:
                pal = im.getpalette()
                if pal:
                    return normalize_palette(pal)
        except OSError:
            pass
    return None


def swap_pairs(mask_t, pairs):
    if not pairs:
        return mask_t
    out = mask_t.clone()
    for l, r in pairs:
        lm, rm = mask_t == l, mask_t == r
        out[lm], out[rm] = r, l
    return out


class TrainTransform:
    def __init__(self, cfg):
        a = cfg["augmentation"]
        self.size = INPUT_SIZE
        self.scale = tuple(a["resize_scale"])
        self.ratio = tuple(a["resize_ratio"])
        self.hflip_prob = float(a["hflip_prob"])
        self.rotation_deg = float(a["rotation_deg"])
        self.flip_pairs = FLIP_PAIRS
        cj = a["color_jitter"]
        self.cj = ColorJitter(float(cj["brightness"]), float(cj["contrast"]), float(cj["saturation"]), float(cj["hue"]))
        self.cj_prob = float(cj["prob"])
        gb = a["gaussian_blur"]
        self.gb_prob = float(gb["prob"])
        self.gb_sigma = tuple(float(x) for x in gb["sigma"])

    def __call__(self, img, msk):
        i, j, h, w = RandomResizedCrop.get_params(img, self.scale, self.ratio)
        img = TF.resized_crop(img, i, j, h, w, [self.size, self.size], InterpolationMode.BILINEAR)
        msk = TF.resized_crop(msk, i, j, h, w, [self.size, self.size], InterpolationMode.NEAREST)
        did = False
        if random.random() < self.hflip_prob:
            img, msk, did = TF.hflip(img), TF.hflip(msk), True
        if self.rotation_deg > 0:
            a = random.uniform(-self.rotation_deg, self.rotation_deg)
            img = TF.rotate(img, a, interpolation=InterpolationMode.BILINEAR, fill=0)
            msk = TF.rotate(msk, a, interpolation=InterpolationMode.NEAREST, fill=0)
        if random.random() < self.cj_prob:
            img = self.cj(img)
        if random.random() < self.gb_prob:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(*self.gb_sigma)))
        x = TF.normalize(TF.to_tensor(img), (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        y = torch.from_numpy(np.array(msk, dtype=np.int64))
        if did:
            y = swap_pairs(y, self.flip_pairs)
        return x, y


class EvalTf:
    def __init__(self, cfg):
        self.size = INPUT_SIZE

    def __call__(self, img, msk):
        img = TF.resize(img, [self.size, self.size], interpolation=InterpolationMode.BILINEAR)
        msk = TF.resize(msk, [self.size, self.size], interpolation=InterpolationMode.NEAREST)
        x = TF.normalize(TF.to_tensor(img), (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        y = torch.from_numpy(np.array(msk, dtype=np.int64))
        return x, y


class InferTf:
    def __init__(self, cfg):
        self.size = INPUT_SIZE

    def __call__(self, img):
        img = TF.resize(img, [self.size, self.size], interpolation=InterpolationMode.BILINEAR)
        return TF.normalize(TF.to_tensor(img), (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))


class SegDS(Dataset):
    def __init__(self, samples, tf):
        self.samples, self.tf = list(samples), tf
        if not self.samples:
            raise ValueError("empty dataset")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        ip, mp = self.samples[i]
        img = Image.open(ip).convert("RGB")
        msk = Image.open(mp)
        if msk.mode not in {"L", "P", "I"}:
            msk = msk.convert("L")
        x, y = self.tf(img, msk)
        return {"image": x, "mask": y, "name": ip.stem}


class InfDS(Dataset):
    def __init__(self, img_dir, tf):
        self.imgs, self.tf = list_files(img_dir, ".jpg"), tf
        if not self.imgs:
            raise RuntimeError("no images")

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        p = self.imgs[i]
        img = Image.open(p).convert("RGB")
        w, h = img.size
        return {"image": self.tf(img), "name": p.stem, "orig_size": torch.tensor([w, h], dtype=torch.int64)}


class CBA(nn.Sequential):
    def __init__(self, ic, oc, k=3, s=1, p=1, g=1, act=True):
        layers = [nn.Conv2d(ic, oc, k, s, p, groups=g, bias=False), nn.BatchNorm2d(oc)]
        if act:
            layers.append(nn.ReLU6(inplace=True))
        super().__init__(*layers)


class IR(nn.Module):
    def __init__(self, ic, oc, st, er=4):
        super().__init__()
        hid = int(ic * er)
        self.res = st == 1 and ic == oc
        layers = []
        if er != 1:
            layers.append(CBA(ic, hid, k=1, s=1, p=0))
        else:
            hid = ic
        layers.append(CBA(hid, hid, k=3, s=st, p=1, g=hid))
        layers.append(CBA(hid, oc, k=1, s=1, p=0, act=False))
        self.b = nn.Sequential(*layers)

    def forward(self, x):
        y = self.b(x)
        return y + x if self.res else y


class Dec(nn.Module):
    def __init__(self, ic, sc, oc):
        super().__init__()
        self.up, self.rd = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False), CBA(ic, oc, k=1, s=1, p=0)
        self.c1, self.c2 = CBA(oc + sc, oc), CBA(oc, oc)

    def forward(self, x, s):
        x = self.rd(self.up(x))
        if s is not None:
            if s.shape[-2:] != x.shape[-2:]:
                s = F.interpolate(s, size=x.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, s], dim=1)
        return self.c2(self.c1(x))


class Net(nn.Module):
    def __init__(self, ncls, chs, er=4):
        super().__init__()
        c1, c2, c3, c4, c5 = [int(x) for x in chs]
        self.s = CBA(3, c1, 3, 2, 1)
        self.s2 = nn.Sequential(IR(c1, c2, 2, er), IR(c2, c2, 1, er))
        self.s3 = nn.Sequential(IR(c2, c3, 2, er), IR(c3, c3, 1, er))
        self.s4 = nn.Sequential(IR(c3, c4, 2, er), IR(c4, c4, 1, er), IR(c4, c4, 1, er))
        self.s5 = nn.Sequential(IR(c4, c5, 2, er), IR(c5, c5, 1, er))
        self.d4, self.d3, self.d2, self.d1, self.d0 = Dec(c5, c4, c4), Dec(c4, c3, c3), Dec(c3, c2, c2), Dec(c2, c1, c1), Dec(c1, 0, c1)
        self.h = nn.Conv2d(c1, ncls, 1)
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        sh = x.shape[-2:]
        f1 = self.s(x)
        f2, f3, f4, f5 = self.s2(f1), None, None, None
        f3 = self.s3(f2)
        f4 = self.s4(f3)
        f5 = self.s5(f4)
        d = self.d0(self.d1(self.d2(self.d3(self.d4(f5, f4), f3), f2), f1), None)
        y = self.h(d)
        if y.shape[-2:] != sh:
            y = F.interpolate(y, size=sh, mode="bilinear", align_corners=False)
        return y


class Dice(nn.Module):
    def __init__(self, ncls, ignore=None, eps=1e-6, present_only=True):
        super().__init__()
        self.ncls, self.ignore, self.eps, self.present_only = ncls, ignore, eps, present_only

    def forward(self, logits, target):
        p = F.softmax(logits, dim=1)
        t = target.clamp(min=0, max=self.ncls - 1)
        oh = F.one_hot(t, num_classes=self.ncls).permute(0, 3, 1, 2).float()
        if self.ignore is not None:
            v = (target != self.ignore).unsqueeze(1)
            p, oh = p * v, oh * v
        dims = (0, 2, 3)
        inter = (p * oh).sum(dims)
        den = p.sum(dims) + oh.sum(dims)
        d = (2 * inter + self.eps) / (den + self.eps)
        if self.present_only:
            pr = oh.sum(dims) > 0
            if pr.any():
                return 1.0 - d[pr].mean()
        return 1.0 - d.mean()


class SegLoss(nn.Module):
    def __init__(self, ncls, dw=0.5, ignore=None, cw=None, d_present=True):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=cw) if ignore is None else nn.CrossEntropyLoss(weight=cw, ignore_index=ignore)
        self.dice = Dice(ncls, ignore=ignore, present_only=d_present)
        self.dw = float(dw)

    def forward(self, logits, target):
        ce = self.ce(logits, target)
        di = self.dice(logits, target)
        tt = ce + self.dw * di
        return tt, {"ce": ce.detach(), "dice": di.detach(), "total": tt.detach()}


def boundary_target(mask):
    e = torch.zeros_like(mask, dtype=torch.bool)
    e[:, 1:, :] |= mask[:, 1:, :] != mask[:, :-1, :]
    e[:, :-1, :] |= mask[:, :-1, :] != mask[:, 1:, :]
    e[:, :, 1:] |= mask[:, :, 1:] != mask[:, :, :-1]
    e[:, :, :-1] |= mask[:, :, :-1] != mask[:, :, 1:]
    return e.unsqueeze(1).float()


def boundary_bce(logits, bgt, pos_w=4.0, ps=4.0, eps=1e-6):
    p = F.softmax(logits, dim=1)
    dv = (p[:, :, 1:, :] - p[:, :, :-1, :]).abs().mean(dim=1, keepdim=True)
    dh = (p[:, :, :, 1:] - p[:, :, :, :-1]).abs().mean(dim=1, keepdim=True)
    dvu, dvd = F.pad(dv, (0, 0, 1, 0)), F.pad(dv, (0, 0, 0, 1))
    dhl, dhr = F.pad(dh, (1, 0, 0, 0)), F.pad(dh, (0, 1, 0, 0))
    pr = torch.maximum(torch.maximum(dvu, dvd), torch.maximum(dhl, dhr))
    pr = (1.0 - torch.exp(-ps * pr)).clamp(eps, 1.0 - eps)
    pl = -bgt * torch.log(pr)
    nl = -(1.0 - bgt) * torch.log(1.0 - pr)
    return (float(pos_w) * pl + nl).mean()


def fscore_img(gt, pd, beta=1.0):
    eps, b2, fs = 1e-7, beta * beta, []
    for cid in torch.unique(gt):
        gm, pm = gt == cid, pd == cid
        tp = (gm & pm).sum().double()
        fp = ((~gm) & pm).sum().double()
        fn = (gm & (~pm)).sum().double()
        pr = tp / (tp + fp + eps)
        rc = tp / (tp + fn + eps)
        fs.append(((1 + b2) * (pr * rc)) / ((b2 * pr) + rc + eps))
    return float(torch.stack(fs).mean().item()) if fs else 0.0


class Meter:
    def __init__(self, ncls):
        self.ncls = ncls
        self.cm = torch.zeros((ncls, ncls), dtype=torch.float64)
        self.fsum = 0.0
        self.n = 0

    def update(self, logits, target):
        pred = logits.argmax(dim=1)
        p, t = pred.view(-1).cpu(), target.view(-1).cpu()
        v = (t >= 0) & (t < self.ncls)
        h = torch.bincount(t[v] * self.ncls + p[v], minlength=self.ncls ** 2).view(self.ncls, self.ncls)
        self.cm += h.double()
        for g, d in zip(target.cpu(), pred.cpu()):
            self.fsum += fscore_img(g, d)
            self.n += 1

    def compute(self):
        tp = torch.diag(self.cm)
        pa = (tp.sum() / self.cm.sum()).item() if self.cm.sum() > 0 else 0.0
        fp = self.cm.sum(dim=0) - tp
        fn = self.cm.sum(dim=1) - tp
        pr = tp / (tp + fp + 1e-7)
        rc = tp / (tp + fn + 1e-7)
        fpc = (2 * pr * rc) / (pr + rc + 1e-7)
        gp = self.cm.sum(dim=1) > 0
        return {
            "pixel_accuracy": float(pa),
            "f_score": float(self.fsum / self.n if self.n > 0 else 0.0),
            "f_score_per_class": [float(x) for x in fpc.tolist()],
            "gt_present": [bool(x) for x in gp.tolist()],
        }


def eval_epoch(model, loader, crit, device, ncls, use_amp, desc):
    model.eval()
    m, ls = Meter(ncls), []
    amp = bool(use_amp) and device.type == "cuda"
    with torch.no_grad():
        for b in tqdm(loader, desc=desc, leave=False):
            x, y = b["image"].to(device), b["mask"].to(device)
            with torch.amp.autocast(device_type=device.type, enabled=amp):
                lo = model(x)
                l, _ = crit(lo, y)
            ls.append(float(l.item()))
            m.update(lo, y)
    out = m.compute()
    out["loss"] = float(np.mean(ls)) if ls else float("nan")
    return out


def train_epoch(model, loader, crit, opt, scaler, device, cfg, ep, eps):
    model.train()
    amp = bool(cfg["train"]["use_amp"]) and device.type == "cuda"
    bc = cfg["loss"]["boundary"]
    ben, bw, bpw, bws, bps = bool(bc["enabled"]), float(bc["weight"]), float(bc["pos_weight"]), int(bc["warmup_epochs"]), float(bc["pred_scale"])
    bactive = ben and ep >= bws
    tl, cel, dil, bl = [], [], [], []
    for b in tqdm(loader, desc=f"train {ep + 1}/{eps}", leave=False):
        x, y = b["image"].to(device), b["mask"].to(device)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp):
            lo = model(x)
            l, c = crit(lo, y)
            bdl = torch.zeros((), device=device)
            if ben:
                bdl = boundary_bce(lo, boundary_target(y), pos_w=bpw, ps=bps)
                if bactive:
                    l = l + bw * bdl
        scaler.scale(l).backward()
        scaler.step(opt)
        scaler.update()
        tl.append(float(l.detach().item()))
        cel.append(float(c["ce"].item()))
        dil.append(float(c["dice"].item()))
        bl.append(float(bdl.detach().item()))
    return {"loss": float(np.mean(tl)), "ce": float(np.mean(cel)), "dice": float(np.mean(dil)), "boundary": float(np.mean(bl))}


def run_infer(model, loader, device, out_dir, use_amp, palette):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pal = [int(x) & 255 for x in palette] if palette is not None else None
    amp = bool(use_amp) and device.type == "cuda"
    model.eval()
    with torch.no_grad():
        for b in tqdm(loader, desc="infer", leave=False):
            x, nms, szs = b["image"].to(device), b["name"], b["orig_size"].cpu().tolist()
            with torch.amp.autocast(device_type=device.type, enabled=amp):
                p = model(x).argmax(dim=1).cpu().numpy().astype(np.uint8)
            for arr, n, sz in zip(p, nms, szs):
                w, h = int(sz[0]), int(sz[1])
                im = Image.fromarray(arr, mode="P")
                if pal is not None:
                    im.putpalette(pal)
                if im.size != (w, h):
                    im = im.resize((w, h), resample=Image.NEAREST)
                im.save(out_dir / f"{n}.png")


def make_model(cfg):
    return Net(NUM_CLASSES, cfg["model"]["encoder_channels"], cfg["model"]["expand_ratio"])


def train_mode(cfg):
    pairs = pair_samples(TRAIN_IMAGES_DIR, TRAIN_MASKS_DIR, strict=True)
    tr, va = split_samples(pairs, float(cfg["data"]["val_split"]), bool(cfg["train"]["use_internal_val"]))
    tr_ds = SegDS(tr, TrainTransform(cfg))
    va_ds = SegDS(va, EvalTf(cfg)) if va else None
    tr_ld = DataLoader(tr_ds, batch_size=int(cfg["train"]["batch_size"]), shuffle=True, num_workers=NUM_WORKERS, pin_memory=bool(cfg["data"]["pin_memory"]), drop_last=bool(cfg["train"]["drop_last"]))
    va_ld = DataLoader(va_ds, batch_size=int(cfg["train"]["batch_size"]), shuffle=False, num_workers=NUM_WORKERS, pin_memory=bool(cfg["data"]["pin_memory"]), drop_last=False) if va_ds is not None else None
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = make_model(cfg).to(dev)
    npm = sum(p.numel() for p in m.parameters() if p.requires_grad)
    if npm >= int(cfg["model"]["max_trainable_params"]):
        raise ValueError(f"params {npm} >= limit {cfg['model']['max_trainable_params']}")
    cw = maybe_weights(cfg, NUM_CLASSES)
    crit = SegLoss(NUM_CLASSES, cfg["loss"]["dice_weight"], cfg["loss"]["ignore_index"], cw, cfg["loss"]["dice_present_only"]).to(dev)
    opt = AdamW(m.parameters(), lr=float(cfg["train"]["lr"]), weight_decay=float(cfg["train"]["weight_decay"]))
    sch = CosineAnnealingLR(opt, T_max=int(cfg["train"]["epochs"]), eta_min=float(cfg["train"]["min_lr"]))
    scaler = torch.amp.GradScaler("cuda", enabled=bool(cfg["train"]["use_amp"]) and dev.type == "cuda")
    sdir = SAVE_DIR
    sdir.mkdir(parents=True, exist_ok=True)
    (sdir / "portable_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print(f"Device: {dev}")
    print(f"Train samples: {len(tr_ds)}")
    print(f"Val samples: {0 if va_ds is None else len(va_ds)}")
    print(f"Trainable params: {npm:,}")
    best, hist = -float("inf"), []
    for ep in range(int(cfg["train"]["epochs"])):
        tm = train_epoch(m, tr_ld, crit, opt, scaler, dev, cfg, ep, int(cfg["train"]["epochs"]))
        sch.step()
        if va_ld is not None:
            vm = eval_epoch(m, va_ld, crit, dev, NUM_CLASSES, cfg["train"]["use_amp"], f"val {ep + 1}/{cfg['train']['epochs']}")
            score = float(vm["f_score"])
        else:
            vm, score = {}, -float(tm["loss"])
        ck = {
            "epoch": ep + 1, "model_state": m.state_dict(), "optimizer_state": opt.state_dict(),
            "scheduler_state": sch.state_dict(), "scaler_state": scaler.state_dict(),
            "train_metrics": tm, "val_metrics": vm, "config": cfg,
        }
        torch.save(ck, sdir / "last.pt")
        if score > best:
            best = score
            torch.save(ck, sdir / "best.pt")
        hist.append({"epoch": ep + 1, "lr": float(opt.param_groups[0]["lr"]), "train": tm, "val": vm, "best_score": best})
        print(f"Epoch {ep + 1:03d} | train_loss={tm['loss']:.4f} val_f_score={vm.get('f_score', float('nan')):.4f} best={'yes' if score == best else 'no'}")
    (sdir / "history.json").write_text(json.dumps(hist, indent=2), encoding="utf-8")
    if bool(cfg["train"]["run_val_data"]):
        bpt = sdir / "best.pt"
        if not bpt.exists():
            raise FileNotFoundError(bpt)
        ck = torch.load(bpt, map_location=dev)
        m.load_state_dict(ck["model_state"], strict=True)
        ids = InfDS(VAL_IMAGES_DIR, InferTf(cfg))
        ild = DataLoader(ids, batch_size=int(cfg["inference"]["batch_size"]), shuffle=False, num_workers=NUM_WORKERS, pin_memory=bool(cfg["data"]["pin_memory"]), drop_last=False)
        pal = load_palette(TRAIN_MASKS_DIR) or make_pascal_palette()
        run_infer(m, ild, dev, VAL_MASKS_DIR, cfg["train"]["use_amp"], pal)
        print(f"[post-train] Saved val predictions to {VAL_MASKS_DIR.resolve()}")


def validate_mode(cfg, ckpt, source):
    if source == "internal":
        pairs = pair_samples(TRAIN_IMAGES_DIR, TRAIN_MASKS_DIR, strict=True)
        _, ev = split_samples(pairs, float(cfg["data"]["val_split"]), True)
        if not ev:
            raise ValueError("empty internal val split")
    else:
        if not VAL_MASKS_DIR.exists():
            raise FileNotFoundError(VAL_MASKS_DIR)
        ev = pair_samples(VAL_IMAGES_DIR, VAL_MASKS_DIR, strict=True)
    ds = SegDS(ev, EvalTf(cfg))
    ld = DataLoader(ds, batch_size=int(cfg["train"]["batch_size"]), shuffle=False, num_workers=NUM_WORKERS, pin_memory=bool(cfg["data"]["pin_memory"]), drop_last=False)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = make_model(cfg).to(dev)
    ck = torch.load(Path(ckpt), map_location=dev)
    m.load_state_dict(ck["model_state"] if "model_state" in ck else ck, strict=True)
    cw = maybe_weights(cfg, NUM_CLASSES)
    crit = SegLoss(NUM_CLASSES, cfg["loss"]["dice_weight"], cfg["loss"]["ignore_index"], cw, cfg["loss"]["dice_present_only"]).to(dev)
    met = eval_epoch(m, ld, crit, dev, NUM_CLASSES, cfg["train"]["use_amp"], f"validate:{source}")
    print(f"Samples: {len(ds)}")
    print(f"Loss: {met['loss']:.6f}")
    print(f"Pixel Accuracy: {met['pixel_accuracy']:.6f}")
    print(f"F-score: {met['f_score']:.6f}")
    print("Per-class F-score:")
    for i, (n, s, p) in enumerate(zip(CLASS_NAMES, met["f_score_per_class"], met["gt_present"])):
        print(f"  [{i:02d}] {n:<12} f_score={float(s):.6f} ({'present' if p else 'absent'})")


def infer_mode(cfg, ckpt):
    ds = InfDS(VAL_IMAGES_DIR, InferTf(cfg))
    ld = DataLoader(ds, batch_size=int(cfg["inference"]["batch_size"]), shuffle=False, num_workers=NUM_WORKERS, pin_memory=bool(cfg["data"]["pin_memory"]), drop_last=False)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = make_model(cfg).to(dev)
    ck = torch.load(Path(ckpt), map_location=dev)
    m.load_state_dict(ck["model_state"] if "model_state" in ck else ck, strict=True)
    pal = load_palette(TRAIN_MASKS_DIR) or make_pascal_palette()
    run_infer(m, ld, dev, INFER_OUTPUT_DIR, cfg["train"]["use_amp"], pal)
    print(f"Saved predictions to: {INFER_OUTPUT_DIR.resolve()}")


def count_params_mode(cfg):
    n = sum(p.numel() for p in make_model(cfg).parameters() if p.requires_grad)
    lim = int(cfg["model"]["max_trainable_params"])
    print(f"Trainable params: {n:,}")
    print(f"Limit:            {lim:,}")
    print(f"Under limit:      {n < lim}")


def parse_args():
    p = argparse.ArgumentParser(description="Single-file MobileNet+UNet (no local module dependency)")
    p.add_argument("--mode", default="train", choices=["train", "validate", "infer", "count-params"])
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--source", default="internal", choices=["internal", "val"])
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--full-train", action="store_true")
    return p.parse_args()


def main():
    a = parse_args()
    cfg = deepcopy(CFG)
    if a.epochs is not None:
        cfg["train"]["epochs"] = int(a.epochs)
    if a.batch_size is not None:
        cfg["train"]["batch_size"] = int(a.batch_size)
        cfg["inference"]["batch_size"] = int(a.batch_size)
    if a.full_train:
        cfg["train"]["use_internal_val"] = False

    if a.mode == "count-params":
        count_params_mode(cfg)
    elif a.mode == "train":
        train_mode(cfg)
    elif a.mode == "validate":
        if a.checkpoint is None:
            raise ValueError("--checkpoint is required for validate mode")
        validate_mode(cfg, a.checkpoint, a.source)
    elif a.mode == "infer":
        if a.checkpoint is None:
            raise ValueError("--checkpoint is required for infer mode")
        infer_mode(cfg, a.checkpoint)
    else:
        raise ValueError(a.mode)


if __name__ == "__main__":
    main()
