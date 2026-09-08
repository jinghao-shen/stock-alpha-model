"""
embed.py — Qwen embedding with disk caching.

Embeddings are expensive to compute (~1 min on GPU for 15k prompts).
We cache to .npz keyed by a hash of the prompt list so re-runs skip encoding.
"""

import hashlib
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

EMB_MODEL = "Qwen/Qwen3-Embedding-0.6B"
ENCODE_BATCH = 128   # chunks for progress display; inner batch_size=32


def _prompt_hash(prompts: list[str]) -> str:
    h = hashlib.md5("\n".join(prompts).encode()).hexdigest()[:12]
    return h


def get_embeddings(
    prompts: list[str],
    cache_dir: Path,
    device: str = "cpu",
    model_name: str = EMB_MODEL,
) -> np.ndarray:
    """
    Return normalized float32 embeddings for `prompts`.
    Loads from cache if available; encodes and saves otherwise.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    tag = _prompt_hash(prompts)
    cache_path = cache_dir / f"emb_{model_name.replace('/', '_')}_{tag}.npz"

    if cache_path.exists():
        blob = np.load(cache_path, allow_pickle=True)
        print(f"Loaded cached embeddings {blob['embeddings'].shape} from {cache_path.name}")
        return blob["embeddings"]

    print(f"Encoding {len(prompts)} prompts with {model_name} on {device}")
    model = SentenceTransformer(model_name, device=device)

    chunks = [prompts[i: i + ENCODE_BATCH] for i in range(0, len(prompts), ENCODE_BATCH)]
    parts, t0 = [], time.time()
    for i, chunk in enumerate(chunks):
        parts.append(
            model.encode(chunk, batch_size=32, normalize_embeddings=True,
                         convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
        )
        done = min((i + 1) * ENCODE_BATCH, len(prompts))
        elapsed = time.time() - t0
        eta = (elapsed / done) * (len(prompts) - done) if done else 0
        print(f"  [{done:>5}/{len(prompts)}]  {elapsed:.0f}s elapsed  ETA ~{eta:.0f}s")

    embeddings = np.concatenate(parts, axis=0)
    del model

    np.savez_compressed(cache_path, embeddings=embeddings,
                        prompts=np.array(prompts, dtype=object))
    print(f"Saved embeddings {embeddings.shape} → {cache_path.name}")
    return embeddings


def align_embeddings(embeddings: np.ndarray, prompts_original: list[str],
                     prompts_sorted: list[str]) -> np.ndarray:
    """Re-index embeddings to match a re-sorted sample order."""
    idx_map = {p: i for i, p in enumerate(prompts_original)}
    order = [idx_map[p] for p in prompts_sorted]
    return embeddings[order]
