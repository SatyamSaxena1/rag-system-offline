from typing import List, Any, Tuple
import os
import json
import numpy as np


class Retriever:
    """Embeds queries and retrieves documents from the saved local index.

    Expects data/indices/meta.json (or indices_store) to exist, created by scripts/index_corpus.py
    """

    def __init__(self, indices_dir: str | None = None):
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        default_paths = [
            os.path.join(root, 'data', 'indices'),
            os.path.join(root, 'data', 'indices_store'),
        ]
        self.indices_dir = indices_dir or next((p for p in default_paths if os.path.exists(os.path.join(p, 'meta.json'))), None)
        if not self.indices_dir:
            raise FileNotFoundError('No meta.json found in data/indices or data/indices_store')
        with open(os.path.join(self.indices_dir, 'meta.json'), 'r', encoding='utf-8') as f:
            self.meta = json.load(f)
        with open(os.path.join(self.indices_dir, 'docs.json'), 'r', encoding='utf-8') as f:
            self.docs = json.load(f)
        self.backend = self.meta.get('backend')
        self.index_type = self.meta.get('index_type')
        self._setup_embedder()
        self._setup_search()
        # Optional sources list
        sources_path = os.path.join(self.indices_dir, 'sources.json')
        self.sources = []
        if os.path.exists(sources_path):
            try:
                import json as _json
                with open(sources_path, 'r', encoding='utf-8') as f:
                    self.sources = _json.load(f)
            except Exception:
                self.sources = []

    def _setup_embedder(self):
        def l2n(a: np.ndarray) -> np.ndarray:
            return a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
        self.l2n = l2n
        model_id = self.meta.get('model')
        if self.backend == 'sentence-transformers':
            from sentence_transformers import SentenceTransformer
            self.embedder = SentenceTransformer(model_id)
            self.embed = lambda texts: self.embedder.encode(texts, batch_size=64, convert_to_numpy=True, normalize_embeddings=True)
        elif self.backend == 'transformers-mean':
            from transformers import AutoTokenizer, AutoModel
            import torch
            tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True, use_fast=True)
            mdl = AutoModel.from_pretrained(model_id, local_files_only=True)
            if torch.cuda.is_available():
                mdl = mdl.to('cuda').eval()
            else:
                mdl = mdl.eval()
            def _embed(texts):
                with torch.no_grad():
                    batch = tok(texts, padding=True, truncation=True, return_tensors='pt', max_length=256)
                    if torch.cuda.is_available():
                        batch = {k: v.to('cuda') for k, v in batch.items()}
                    out = mdl(**batch)
                    last_hidden = out.last_hidden_state
                    mask = batch['attention_mask'].unsqueeze(-1)
                    summed = (last_hidden * mask).sum(dim=1)
                    counts = mask.sum(dim=1).clamp(min=1)
                    emb = summed / counts
                    x = emb.detach().cpu().numpy().astype(np.float32)
                    return self.l2n(x)
            self.embed = _embed
        else:
            # TF-IDF fallback: re-fit on docs
            from sklearn.feature_extraction.text import TfidfVectorizer
            vec = TfidfVectorizer()
            vec.fit(self.docs)
            self.embed = lambda texts: self.l2n(vec.transform(texts).astype(np.float32).toarray())

    def _setup_search(self):
        if self.index_type in ('faiss_ip', 'faiss_ivf'):
            import faiss
            self.index = faiss.read_index(os.path.join(self.indices_dir, 'faiss.index'))
            if self.index_type == 'faiss_ivf':
                try:
                    self.index.nprobe = int(self.meta.get('nprobe', 10))
                except Exception:
                    pass
            self.search = lambda q, k: self.index.search(q.astype(np.float32), k)
            self.search_full = lambda q: self.index.search(q.astype(np.float32), len(self.docs))
        else:
            emb_path = self.meta.get('embeddings_path')
            self.emb = np.load(emb_path)
            def _search(q, k):
                sims = q @ self.emb.T
                I = np.argsort(-sims, axis=1)[:, :k]
                D = np.take_along_axis(sims, I, axis=1)
                return D, I
            self.search = _search
            def _search_full(q):
                sims = q @ self.emb.T
                I = np.argsort(-sims, axis=1)
                D = np.take_along_axis(sims, I, axis=1)
                return D, I
            self.search_full = _search_full

    def retrieve(self, query: str, top_k: int = 5) -> List[str]:
        q = self.embed([query]).astype(np.float32)
        _, I = self.search(q, top_k)
        return [self.docs[i] for i in I[0]]

    # New: expose embedding for a single query (normalized float32)
    def embed_query(self, query: str) -> np.ndarray:
        return self.embed([query]).astype(np.float32)

    # New: retrieve using a precomputed embedding
    def retrieve_with_embedding(self, q_emb: np.ndarray, top_k: int = 5) -> List[str]:
        if q_emb.ndim == 1:
            q_emb = q_emb.reshape(1, -1)
        _, I = self.search(q_emb.astype(np.float32), top_k)
        return [self.docs[i] for i in I[0]]

    # New: retrieve docs and scores; apply optional min_score and source_filter
    def retrieve_scored(self, q_emb: np.ndarray, top_k: int = 5, min_score: float = 0.0, source_filter: str | None = None):
        if q_emb.ndim == 1:
            q_emb = q_emb.reshape(1, -1)
        # Default search pool
        pool_k = max(top_k * 50, 2000) if source_filter else top_k
        D, I = self.search(q_emb.astype(np.float32), min(pool_k, len(self.docs)))
        pairs = [(int(i), float(d)) for i, d in zip(I[0].tolist(), D[0].tolist())]
        # Apply source filter
        if source_filter and self.sources:
            sf = source_filter.lower()
            allowed = [idx for idx, s in enumerate(self.sources) if sf in str(s).lower()]
            if allowed:
                allowed_set = set(allowed)
                pairs = [(i, d) for i, d in pairs if i in allowed_set]
                # If empty after initial pool, do a full search and filter
                if not pairs:
                    D, I = self.search_full(q_emb.astype(np.float32))
                    pairs = [(int(i), float(d)) for i, d in zip(I[0].tolist(), D[0].tolist()) if i in allowed_set]
        # Sort by score and apply threshold
        pairs.sort(key=lambda x: -x[1])
        if min_score > 0:
            pairs = [(i, d) for i, d in pairs if d >= min_score]
        pairs = pairs[:top_k]
        return [self.docs[i] for i, _ in pairs], [d for _, d in pairs]

    # New: same as retrieve_scored but also returns source strings when available
    def retrieve_scored_with_sources(self, q_emb: np.ndarray, top_k: int = 5, min_score: float = 0.0, source_filter: str | None = None):
        docs, scores = self.retrieve_scored(q_emb, top_k=top_k, min_score=min_score, source_filter=source_filter)
        srcs = []
        if self.sources and docs:
            # Map doc text back to index by direct identity via search over docs list
            # Build a map from doc text to indices (handle duplicates conservatively)
            text_to_idxs = {}
            for idx, text in enumerate(self.docs):
                text_to_idxs.setdefault(text, []).append(idx)
            for d in docs:
                cand = text_to_idxs.get(d, [])
                srcs.append(self.sources[cand[0]] if cand else None)
        else:
            srcs = [None] * len(docs)
        return docs, scores, srcs

    # New: compute only top score for gating
    def top_score(self, q_emb: np.ndarray) -> float:
        if q_emb.ndim == 1:
            q_emb = q_emb.reshape(1, -1)
        D, _ = self.search(q_emb.astype(np.float32), 1)
        try:
            return float(D[0][0])
        except Exception:
            return 0.0

    def retrieve_with_history(self, query: str, history: List[str], top_k: int = 5) -> List[str]:
        augmented_query = self._augment_query_with_history(query, history)
        return self.retrieve(augmented_query, top_k)

    def _augment_query_with_history(self, query: str, history: List[str]) -> str:
        return " ".join(history + [query])

    # Placeholder metrics
    def evaluate_retrieval(self, retrieved_docs: List[str], ground_truth: List[Any]) -> dict:
        return {"precision": 0.0, "recall": 0.0}
