from typing import Callable, Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def precision_at_k(y_true: np.ndarray, scores: np.ndarray, k: int = 10) -> float:
    idx = np.argsort(-scores)[:k]
    return float(y_true[idx].mean()) if k > 0 else 0.0


@torch.no_grad()
def likelihood_scores(model_name: str, texts: List[str]) -> np.ndarray:
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    scores: List[float] = []
    for t in texts:
        inputs = tok(t, return_tensors="pt").to(device)
        with torch.cuda.amp.autocast(enabled=(device == "cuda")):
            out = model(**inputs, labels=inputs["input_ids"])  # uses LM loss
        # lower loss (perplexity) -> higher contamination suspicion, so invert
        scores.append(float(-out.loss.detach().cpu().item()))
    return np.asarray(scores, dtype=np.float32)


@torch.no_grad()
def embedding_knn_scores(model_name: str, 
                         texts: List[str], 
                         bank_vectors: np.ndarray, 
                         top_k: int = 32) -> np.ndarray:
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    bank = torch.from_numpy(bank_vectors).to(device)
    bank = torch.nn.functional.normalize(bank, dim=-1)

    scores: List[float] = []
    for t in texts:
        inputs = tok(t, return_tensors="pt").to(device)
        with torch.cuda.amp.autocast(enabled=(device == "cuda")):
            out = model(**inputs)
        last = out.hidden_states[-1][0]  # [T, D]
        q = last.mean(dim=0, keepdim=True)  # pooled
        q = torch.nn.functional.normalize(q, dim=-1)
        sims = (q @ bank.T).squeeze(0)
        topv = torch.topk(sims, k=min(top_k, sims.numel()), dim=-1).values
        scores.append(float(topv.max().item()))
    return np.asarray(scores, dtype=np.float32)


def run_benchmarks(model_name: str,
                   test_texts: List[str],
                   y_true: np.ndarray,
                   fpguard_scores_fn: Callable[[List[str]], np.ndarray] = None,
                   bank_vectors: np.ndarray = None,
                   report_topk: int = 10) -> Dict[str, float]:
    metrics: Dict[str, float] = {}

    # Baseline 1: Likelihood
    lik_scores = likelihood_scores(model_name, test_texts)
    metrics[f"likelihood_p@{report_topk}"] = precision_at_k(y_true, lik_scores, report_topk)

    # Baseline 2: Embedding kNN
    if bank_vectors is not None:
        emb_scores = embedding_knn_scores(model_name, test_texts, bank_vectors, top_k=32)
        metrics[f"embknn_p@{report_topk}"] = precision_at_k(y_true, emb_scores, report_topk)

    # FPGuard (if provided)
    if fpguard_scores_fn is not None:
        fp_scores = fpguard_scores_fn(test_texts)
        metrics[f"fpguard_p@{report_topk}"] = precision_at_k(y_true, fp_scores, report_topk)

    return metrics


