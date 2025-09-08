from typing import Optional

import faiss


def tune_ivf_hnsw(index, nprobe: int = 32, ef_search: int = 64):
    if isinstance(index, faiss.IndexIVF):
        index.nprobe = nprobe
    try:
        # HNSW specific parameter
        index.hnsw.efSearch = ef_search  # type: ignore[attr-defined]
    except Exception:
        pass


