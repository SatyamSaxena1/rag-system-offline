import os
import json
from typing import Optional, List, Dict, Any
import numpy as np


class SemanticCache:
    """
    Lightweight semantic cache for query -> {docs, answer} with cosine similarity matching.

    - Stores normalized query embeddings and associated payloads.
    - Lookup returns the best match if similarity >= threshold.
    - Optional persistence to a folder (embeddings.npy + entries.json).
    """

    def __init__(
        self,
        max_items: int = 1000,
        threshold: float = 0.92,
        reuse_answer: bool = True,
        persist_dir: Optional[str] = None,
        eviction_mode: str = "fifo",        # "fifo" or "lru"
        ttl_seconds: int = 0,                # 0 disables TTL
    ) -> None:
        self.max_items = int(max(1, max_items))
        self.threshold = float(threshold)
        self.reuse_answer = bool(reuse_answer)
        self.persist_dir = persist_dir
        self.eviction_mode = (eviction_mode or "fifo").lower()
        if self.eviction_mode not in ("fifo", "lru"):
            self.eviction_mode = "fifo"
        self.ttl_seconds = int(max(0, ttl_seconds))
        self._emb = np.zeros((0, 0), dtype=np.float32)  # shape (n, d)
        # each entry: {"q": str, "docs": List[str], "answer": Optional[str], "ts": float, "last": float, "hits": int}
        self._entries: List[Dict[str, Any]] = []
        # stats
        self.lookups = 0
        self.hits = 0
        self.misses = 0
        if self.persist_dir:
            os.makedirs(self.persist_dir, exist_ok=True)
            self._load()

    def _save(self) -> None:
        if not self.persist_dir:
            return
        entries_path = os.path.join(self.persist_dir, "entries.json")
        emb_path = os.path.join(self.persist_dir, "embeddings.npy")
        with open(entries_path, "w", encoding="utf-8") as f:
            json.dump(self._entries, f, ensure_ascii=False)
        try:
            np.save(emb_path, self._emb)
        except Exception:
            pass

    def _load(self) -> None:
        entries_path = os.path.join(self.persist_dir, "entries.json")
        emb_path = os.path.join(self.persist_dir, "embeddings.npy")
        try:
            if os.path.exists(entries_path) and os.path.exists(emb_path):
                with open(entries_path, "r", encoding="utf-8") as f:
                    self._entries = json.load(f)
                self._emb = np.load(emb_path).astype(np.float32)
        except Exception:
            # Corrupt cache; reset
            self._entries = []
            self._emb = np.zeros((0, 0), dtype=np.float32)

    @staticmethod
    def _l2n(a: np.ndarray) -> np.ndarray:
        return a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)

    def lookup(self, q_emb: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Return a dict {docs, answer, score} if a similar query is cached, else None.
        q_emb: shape (1, d) normalized or will be normalized here.
        """
        self.lookups += 1
        if self._emb.size == 0:
            self.misses += 1
            return None
        if q_emb.ndim == 1:
            q_emb = q_emb.reshape(1, -1)
        # Ensure normalized
        qn = self._l2n(q_emb.astype(np.float32))
        # Filter by TTL if enabled
        valid_idx = list(range(self._emb.shape[0]))
        if self.ttl_seconds > 0:
            import time
            now = time.time()
            valid_idx = [i for i, e in enumerate(self._entries) if (now - float(e.get("ts", now))) <= self.ttl_seconds]
            if len(valid_idx) < self._emb.shape[0]:
                # prune expired
                self._entries = [self._entries[i] for i in valid_idx]
                self._emb = self._emb[valid_idx, :]
                if self._emb.size == 0:
                    self.misses += 1
                    return None
        # Cosine sim for normalized vectors is dot product
        sims = (qn @ self._emb.T)[0]
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])
        if best_sim >= self.threshold:
            import time
            self.hits += 1
            # update LRU stats
            self._entries[best_idx]["last"] = time.time()
            self._entries[best_idx]["hits"] = int(self._entries[best_idx].get("hits", 0)) + 1
            entry = self._entries[best_idx].copy()
            entry["score"] = best_sim
            return entry
        self.misses += 1
        return None

    def add(self, q_text: str, q_emb: np.ndarray, docs: List[str], answer: Optional[str] = None, sources: Optional[List[str]] = None) -> None:
        if q_emb.ndim == 1:
            q_emb = q_emb.reshape(1, -1)
        qn = self._l2n(q_emb.astype(np.float32))
        # Initialize emb matrix dim if empty
        if self._emb.size == 0:
            self._emb = qn
        else:
            # Align dims if needed
            if qn.shape[1] != self._emb.shape[1]:
                # Reset cache if dims mismatch (embedding model changed)
                self._entries = []
                self._emb = qn
            else:
                self._emb = np.vstack([self._emb, qn])
        import time
        now = time.time()
        entry = {"q": q_text, "docs": docs, "answer": answer, "ts": now, "last": now, "hits": 0}
        if sources is not None:
            entry["sources"] = sources
        self._entries.append(entry)
        # Evict oldest if over capacity
        if len(self._entries) > self.max_items:
            self._evict()
        self._save()

    def _evict(self) -> None:
        if self.eviction_mode == "lru":
            # remove entry with smallest last-access time
            if not self._entries:
                return
            idx = int(np.argmin([e.get("last", 0.0) for e in self._entries]))
        else:
            # fifo: pop the first
            idx = 0
        self._entries.pop(idx)
        self._emb = np.delete(self._emb, idx, axis=0)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "lookups": int(self.lookups),
            "hits": int(self.hits),
            "misses": int(self.misses),
            "size": int(len(self._entries)),
            "capacity": int(self.max_items),
            "mode": self.eviction_mode,
            "ttl_seconds": int(self.ttl_seconds),
        }
