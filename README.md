# Offline RAG System (Windows)

GPU-accelerated, fully offline RAG pipeline for Windows featuring:

- Local LLM via llama.cpp (GGUF)
- FAISS retrieval (Flat/IP, IVF optional)
- Sentence Transformers embeddings (or fallback)
- Semantic cache with persistence
- Behavior modes (strict | balanced | loose)
- Conversation history persistence
- Optional citations

This repository is sanitized: no personal data, indices, or raw documents are included. Point configs to your local model/data dirs.

## Quick start

- Create environment and install requirements from `requirements.txt`.
- Edit `configs/default.yaml`:
  - `generation.model_path`: set to your local GGUF path
  - `embedding_model_path`: set to your local embedding model (optional)
- Place your corpus under `data/raw/` (excluded from git) and run indexer scripts.

## Ask a question

Use the pipeline CLI to ask questions and optionally export responses with citations:

```powershell
python scripts/ask_pipeline.py --mode balanced --show-citations --export out/results.jsonl --queries "Who is Sardanapale?"
```

## Behavior mode

Control retrieval strictness and context size via environment or config:

```powershell
$env:RAG_BEHAVIOR_MODE = "strict"
python scripts/ask_pipeline.py --queries "When did Sardanapale race?"
```

## Folders

- `src/` core code (pipeline, retrieval, generation, cache, conversation)
- `scripts/` utilities (indexing, query, ask/export)
- `configs/` YAML config files
- `data/` (ignored): put your raw docs and indices here
- `logs/` (ignored)

## License
MIT
