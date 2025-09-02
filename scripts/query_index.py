import os, sys, json, argparse
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.config import Config
from src.generation.generator import Generator

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
INDICES_DIR = os.path.join(ROOT, 'data', 'indices')
meta_paths = [
    os.path.join(ROOT, 'data', 'indices', 'meta.json'),
    os.path.join(ROOT, 'data', 'indices_store', 'meta.json')
]
META = None
for mp in meta_paths:
    if os.path.exists(mp) and os.path.isfile(mp):
        with open(mp, 'r', encoding='utf-8') as f:
            META = json.load(f)
        INDICES_DIR = os.path.dirname(mp)
        break
if META is None:
    raise FileNotFoundError('meta.json not found in data/indices or data/indices_store')
# If meta specifies alternate dirs/paths, use them
if META.get('indices_dir'):
    INDICES_DIR = META['indices_dir']

with open(os.path.join(INDICES_DIR, 'docs.json'), 'r', encoding='utf-8') as f:
    DOCS = json.load(f)
sources_path = os.path.join(INDICES_DIR, 'sources.json')
SOURCES = []
if os.path.exists(sources_path):
    with open(sources_path, 'r', encoding='utf-8') as f:
        SOURCES = json.load(f)
    

conf = Config('configs/default.yaml')
emb_model_name = conf.get('embedding_model') or 'sentence-transformers/all-MiniLM-L6-v2'
emb_model_path = conf.get('embedding_model_path')

print('Loaded index meta:', META)

# Embed function matching index backend
backend = META.get('backend')
model_id = META.get('model')

def l2n(a: np.ndarray) -> np.ndarray:
    return a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)

if backend == 'sentence-transformers':
    from sentence_transformers import SentenceTransformer
    import torch
    load_id = model_id if (model_id and os.path.exists(model_id)) else emb_model_path or emb_model_name
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    st = SentenceTransformer(load_id, device=device)
    def embed(texts):
        bs = int(os.environ.get('RAG_EMB_BATCH', '64'))
        return st.encode(texts, batch_size=bs, convert_to_numpy=True, normalize_embeddings=True)
elif backend == 'transformers-mean':
    from transformers import AutoTokenizer, AutoModel
    import torch
    load_id = model_id if (model_id and os.path.exists(model_id)) else emb_model_path or emb_model_name
    tok = AutoTokenizer.from_pretrained(load_id, local_files_only=True, use_fast=True)
    mdl = AutoModel.from_pretrained(load_id, local_files_only=True)
    if torch.cuda.is_available():
        mdl = mdl.to('cuda').eval()
    else:
        mdl = mdl.eval()
    def embed(texts):
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
            return l2n(x)
else:
    from sklearn.feature_extraction.text import TfidfVectorizer
    # Fit the same vectorizer is not possible without saving vocabulary; re-fit on docs
    print('Re-fitting TF-IDF on docs to embed query (approximate retrieval).')
    vec = TfidfVectorizer()
    vec.fit(DOCS)
    def embed(texts):
        X = vec.transform(texts).astype(np.float32).toarray()
        return l2n(X)

index_type = META.get('index_type')

if index_type in ('faiss_ip', 'faiss_ivf'):
    import faiss
    index = faiss.read_index(os.path.join(INDICES_DIR, 'faiss.index'))
    if index_type == 'faiss_ivf':
        # Set nprobe if provided
        try:
            nprobe = int(META.get('nprobe', 10))
            index.nprobe = nprobe
        except Exception:
            pass
    def search(q_emb: np.ndarray, k: int):
        D, I = index.search(q_emb.astype(np.float32), k)
        return D, I
else:
    # Load NumPy embeddings
    emb_path = META.get('embeddings_path') or os.path.join(ROOT, 'data', 'embeddings', 'embeddings.npy')
    EMB = np.load(emb_path)
    def search(q_emb: np.ndarray, k: int):
        # cosine since vectors are normalized
        sims = q_emb @ EMB.T
        I = np.argsort(-sims, axis=1)[:, :k]
        D = np.take_along_axis(sims, I, axis=1)
        return D, I


def retrieve(query: str, k: int = 4, source_filter: str | None = None, min_score: float = 0.0):
    q_emb = embed([query]).astype(np.float32)
    if not source_filter:
        D, I = search(q_emb, k)
        pairs = list(zip([DOCS[i] for i in I[0]], D[0].tolist()))
        if min_score > 0:
            pairs = [(d, s) for d, s in pairs if s >= min_score]
        return pairs
    # Filter by source substring: build a mask of indices to allow
    allowed = [i for i, s in enumerate(SOURCES or ["" for _ in DOCS]) if source_filter.lower() in str(s).lower()]
    if not allowed:
        # No restriction possible; return normal
        D, I = search(q_emb, k)
        docs = [DOCS[i] for i in I[0]]
        scores = D[0].tolist()
        return list(zip(docs, scores))
    # If any allowed chunks contain the query as a direct substring, surface them immediately
    try:
        q_lower = query.lower()
        direct_hits = [i for i in allowed if q_lower in (DOCS[i] or "").lower()]
        if direct_hits:
            # Keep order by first occurrence index to preserve document order; cap at k
            top_idxs = direct_hits[:k]
            return [(DOCS[i], 1.0) for i in top_idxs]
    except Exception:
        pass
    # Search a bigger pool then filter
    # Start with an expanded pool, since filtering can drop many hits
    pool_k = min(len(DOCS), max(2000, k * 50))
    D, I = search(q_emb, pool_k)
    pairs = [(int(i), float(d)) for i, d in zip(I[0].tolist(), D[0].tolist()) if int(i) in allowed]
    # If nothing matched in the expanded pool, fall back to a full scan of the index
    if not pairs and pool_k < len(DOCS):
        D, I = search(q_emb, len(DOCS))
        pairs = [(int(i), float(d)) for i, d in zip(I[0].tolist(), D[0].tolist()) if int(i) in allowed]
    pairs.sort(key=lambda x: -x[1])
    pairs = pairs[:k]
    out = [(DOCS[i], score) for i, score in pairs]
    if min_score > 0:
        out = [(d, s) for d, s in out if s >= min_score]
    return out


def answer(query: str, history=None, k: int = 4, source_filter: str | None = None):
    ctx_pairs = retrieve(query, k, source_filter)
    # Include the user question alongside retrieved docs
    context = f"Question: {query}\n\n" + "\n\n".join([t for t, _ in ctx_pairs])
    gen = Generator()
    out = gen.generate(context, history or [])
    return out, ctx_pairs

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Query the saved index and generate grounded answers.')
    ap.add_argument('--k', type=int, default=4, help='Top-k documents to retrieve')
    ap.add_argument('--show-context', action='store_true', help='Print retrieved context block')
    ap.add_argument('--query', type=str, nargs='*', help='Query string(s). If omitted, runs a small demo suite.')
    ap.add_argument('--source-filter', type=str, default=None, help='Restrict retrieval to sources whose path contains this text (e.g., a pdf filename).')
    ap.add_argument('--no-gen', action='store_true', help='Retrieval-only mode: skip LLM generation and just print retrieved docs and scores.')
    ap.add_argument('--min-score', type=float, default=0.0, help='Similarity floor (cosine/IP) to accept retrieved docs; below this returns fewer/no docs.')
    args = ap.parse_args()

    queries = args.query or [
        'What is the capital of France?',
        'What is the capital of Canada?',
        'What is the capital of Australia?',
        'What is the capital of the United States?'
    ]
    for q in queries:
        if args.no_gen:
            pairs = retrieve(q, k=args.k, source_filter=args.source_filter, min_score=args.min_score)
            print('\nQ:', q)
            print('Top-ctx:', [round(s,3) for _, s in pairs])
            if args.show_context:
                print('Context:')
                print('\n\n'.join([t for t,_ in pairs]))
        else:
            pairs = retrieve(q, k=args.k, source_filter=args.source_filter, min_score=args.min_score)
            if not pairs:
                print('\nQ:', q)
                print('A: I don\'t know.')
                print('Top-ctx: []')
                continue
            # Use the filtered pairs for generation
            ctx_docs = [t for t, _ in pairs]
            context = f"Question: {q}\n\n" + "\n\n".join(ctx_docs)
            gen = Generator()
            ans = gen.generate(context, [])
            ctx = pairs
            print('\nQ:', q)
            print('A:', ans)
            print('Top-ctx:', [round(s,3) for _, s in ctx])
            if args.show_context:
                print('Context:')
                print('\n\n'.join([t for t,_ in ctx]))
