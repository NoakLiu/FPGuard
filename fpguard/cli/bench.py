import json
from typing import Optional

import numpy as np
import typer

from fpguard.benchmarks import run_benchmarks

app = typer.Typer()


@app.command()
def main(
    model: str = typer.Option("meta-llama/Llama-2-7b-hf", help="HF model name"),
    test_file: str = typer.Option(..., help="Path to JSONL with fields: text, label (0/1)"),
    bank_npy: Optional[str] = typer.Option(None, help="Optional .npy for baseline embedding bank vectors (K, D)"),
    topk: int = typer.Option(10, help="Precision@k"),
):
    texts = []
    labels = []
    with open(test_file, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            texts.append(obj["text"])  # type: ignore
            labels.append(int(obj.get("label", 0)))
    y_true = np.asarray(labels, dtype=np.int32)

    bank = None
    if bank_npy:
        bank = np.load(bank_npy)

    metrics = run_benchmarks(
        model_name=model,
        test_texts=texts,
        y_true=y_true,
        fpguard_scores_fn=None,
        bank_vectors=bank,
        report_topk=topk,
    )

    for k, v in metrics.items():
        typer.echo(f"{k}: {v:.4f}")


if __name__ == "__main__":
    app()


