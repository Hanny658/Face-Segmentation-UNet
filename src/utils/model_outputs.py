from __future__ import annotations

from typing import Optional, Tuple
import torch


def split_model_outputs(
    outputs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Return (main_logits, aux_p_logits, aux_d_logits)."""
    if isinstance(outputs, (list, tuple)):
        if len(outputs) >= 3:
            aux_p, main, aux_d = outputs[0], outputs[1], outputs[2]
            return main, aux_p, aux_d
        if len(outputs) == 2:
            aux_p, main = outputs[0], outputs[1]
            return main, aux_p, None
        if len(outputs) == 1:
            return outputs[0], None, None
        raise ValueError("Model returned an empty list/tuple.")
    return outputs, None, None
