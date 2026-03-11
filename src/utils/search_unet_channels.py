from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

if __package__ is None or __package__ == "":
    import sys

    ROOT = Path(__file__).resolve().parents[2]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

# import my model and counting function
from src.models.lightweight_unet import LightweightUNet
from src.utils.param_count import count_trainable_parameters

# search settings
NUM_CLASSES = 19
EXPAND_RATIO = 4
USE_SE = False
BASE_CHANNELS = (32, 64, 96, 128, 160)
N = 10
MIN_CHANNEL = 4
EARLY_STOP_GAP = 128


# Construct the candidate list for each lvl
def ordered_values_from_base(base: int, n: int, low: int, high: int) -> List[int]:
    vals: List[int] = []
    if low <= base <= high:
        vals.append(base)
    for d in range(1, n + 1):
        up = base + d
        down = base - d
        if low <= up <= high:
            vals.append(up)
        if low <= down <= high:
            vals.append(down)
    return vals


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Brute-force search in [base-N, base+N], expanding from base outward."
    )
    parser.add_argument("--max-param", type=int, required=True)
    args = parser.parse_args()
    budget = int(args.max_param)

    levels = len(BASE_CHANNELS)
    value_lists: List[List[int]] = []
    for i in range(levels):
        b = int(BASE_CHANNELS[i])
        low = max(MIN_CHANNEL, b - N)
        high = b + N
        value_lists.append(ordered_values_from_base(b, N, low, high))

    seen: Dict[Tuple[int, ...], int] = {}
    attempt = 0

    best_under_channels: Optional[Tuple[int, ...]] = None
    best_under_params = -1
    early_stop_hit: Optional[Tuple[Tuple[int, ...], int]] = None

    def count(channels: Tuple[int, ...]) -> int:
        nonlocal attempt
        if channels in seen:
            return seen[channels]

        model = LightweightUNet(
            num_classes=NUM_CLASSES,
            encoder_channels=channels,
            expand_ratio=EXPAND_RATIO,
            use_se=USE_SE,
            encoder_type="mobilenetv2",
            decoder_type="unet",
        )
        params = int(count_trainable_parameters(model))
        seen[channels] = params
        attempt += 1
        sign = "<=" if params <= budget else ">"
        print(f"[try {attempt:04d}] channels={list(channels)} params={params:,} ({sign} {budget:,})")
        return params

    def update_best(channels: Tuple[int, ...], params: int) -> None:
        nonlocal best_under_channels, best_under_params
        if params <= budget:
            if params > best_under_params:
                best_under_params = params
                best_under_channels = channels

    # BRUTEEEEEE FORCE in the bound, shaking from base to outward
    for c0 in value_lists[0]:
        if early_stop_hit is not None:
            break
        for c1 in value_lists[1]:
            if early_stop_hit is not None:
                break
            if c1 <= c0:
                continue
            for c2 in value_lists[2]:
                if early_stop_hit is not None:
                    break
                if c2 <= c1:
                    continue
                for c3 in value_lists[3]:
                    if early_stop_hit is not None:
                        break
                    if c3 <= c2:
                        continue
                    for c4 in value_lists[4]:
                        if c4 <= c3:
                            continue
                        ch = (c0, c1, c2, c3, c4)
                        p = count(ch)
                        update_best(ch, p)
                        if p <= budget and (budget - p) < EARLY_STOP_GAP:  # if close enough!
                            early_stop_hit = (ch, p)
                            # Removed BnB since my previvous implementation of BnB cutted some useful branches..
                            break

    print("\n=== result ===")
    if early_stop_hit is not None:
        stop_ch, stop_p = early_stop_hit
        print( # when found eaarly stoppin
            f"early stop: budget-params={budget - stop_p} < {EARLY_STOP_GAP}, "
            f"channels={list(stop_ch)}, params={stop_p:,}"
        )
    print(f"searched candidates: {attempt:,}")
    if best_under_channels is None:
        print("best <= limit: none")
    else:
        print(
            f"best <= limit: channels={list(best_under_channels)}, "
            f"params={best_under_params:,}, margin={budget - best_under_params}"
        )


if __name__ == "__main__":
    main()

# Algorithms are FUN !!
