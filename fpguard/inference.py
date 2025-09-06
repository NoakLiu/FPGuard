from typing import Dict, List, Tuple

import numpy as np
import torch

from .config import FPGuardConfig
from .fingerprints import cosine_similarity


def contamination_scores(
    test_fps: torch.Tensor,
    bank_vectors: np.ndarray,
    alpha: float,
    top_k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # naive cosine similarity against dense bank for MVP; can be swapped with FAISS results
    bank = torch.from_numpy(bank_vectors).to(test_fps.device)
    test_norm = torch.nn.functional.normalize(test_fps, dim=-1)
    bank_norm = torch.nn.functional.normalize(bank, dim=-1)
    sims = test_norm @ bank_norm.T
    top_vals, top_idx = torch.topk(sims, k=min(top_k, sims.shape[-1]), dim=-1)
    gamma = top_vals.max(dim=-1).values
    return gamma, top_idx


def dynamic_threshold(base: float, beta: float, history_contam: float, context_factor: float = 1.0) -> float:
    return base * (1.0 + beta * history_contam) * context_factor


