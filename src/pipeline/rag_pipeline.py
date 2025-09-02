import logging
import os
from src.retrieval.retriever import Retriever
from src.generation.generator import Generator
from src.conversation.history import ConversationHistory
from src.utils.config import Config
from src.cache.semantic_cache import SemanticCache


class RAGPipeline:
    def __init__(self, config_path: str = 'configs/default.yaml'):
        self.config = Config(config_file=config_path).settings
        self.retriever = Retriever()
        self.generator = Generator(self.config.get('generation'))
        conv_conf = (self.config.get('conversation') or {})
        self.history = ConversationHistory(persist_path=conv_conf.get('persist_path'))
        self._last_sources: list[str] = []
        # Semantic cache settings with safe defaults
        sc_conf = (self.config.get('semantic_cache') or {})
        self.cache = SemanticCache(
            max_items=int(sc_conf.get('max_items', 1000)),
            threshold=float(sc_conf.get('threshold', 0.92)),
            reuse_answer=bool(sc_conf.get('reuse_answer', True)),
            persist_dir=sc_conf.get('persist_dir') or None,
            eviction_mode=(sc_conf.get('eviction_mode') or 'fifo'),
            ttl_seconds=int(sc_conf.get('ttl_seconds', 0)),
        )
        # Logger
        log_conf = (self.config.get('logging') or {})
        level = getattr(logging, str(log_conf.get('level', 'INFO')).upper(), logging.INFO)
        logging.basicConfig(level=level, filename=log_conf.get('log_file'), format='%(asctime)s %(levelname)s %(message)s')
        self.log = logging.getLogger(__name__)
        # Citations toggle
        self.show_citations = bool((self.config.get('retrieval') or {}).get('show_citations', False))
        # Behavior mode (strict | balanced | loose) with env override
        self.behavior_mode = (os.environ.get('RAG_BEHAVIOR_MODE') or (self.config.get('retrieval') or {}).get('behavior_mode') or 'balanced').lower()
        if self.behavior_mode not in ('strict', 'balanced', 'loose'):
            self.behavior_mode = 'balanced'
        self._retr_conf = self._compute_effective_retrieval()

    def process_query(self, query: str) -> str:
        # Cache-aware retrieval/generation
        # 1) Build both original and history-augmented queries and embeddings
        hist_list = [t for t in self._format_history_simple()]
        aug_text = (" ".join(hist_list + [query])).strip() if hist_list else query
        q_emb = self.retriever.embed_query(query)
        aug_emb = self.retriever.embed_query(aug_text) if aug_text != query else q_emb

        # 2) Semantic cache lookup: prefer augmented query hit, then original
        hit = self.cache.lookup(aug_emb) or self.cache.lookup(q_emb)
        if hit is not None:
            cached_docs = hit.get('docs') or []
            cached_answer = hit.get('answer') if self.cache.reuse_answer else None
            cached_sources = hit.get('sources') or []
            # Update conversation history
            self.history.add_turn(query, cached_answer or "")
            if cached_answer:
                self._log_cache_stats()
                self._last_sources = list(cached_sources or [])
                return cached_answer
            # No cached answer, only docs: proceed to generation with limited context
            q_context, sel_docs, sel_sources = self._build_context(query, cached_docs, None, cached_sources)
            # Guard: if context doesn't contain any meaningful query terms, abstain
            if not self._passes_term_overlap(query, sel_docs):
                response = "I don't know."
            else:
                response = self.generator.generate(q_context, hist_list)
            self._last_sources = list(sel_sources or [])
            if self.show_citations and sel_sources:
                cites = [s for s in sel_sources if s]
                if cites:
                    response = response.rstrip() + "\n\nSources:\n" + "\n".join(cites)
            self.history.add_turn(query, response)
            # Add/refresh cache with answer
            self.cache.add(query, q_emb, sel_docs, response, sources=sel_sources)
            if aug_text:
                self.cache.add(aug_text, aug_emb, sel_docs, response, sources=sel_sources)
            self._log_cache_stats()
            return response

        # 3) Regular retrieval with similarity floor, then generation
        retr_conf = self._retr_conf
        top_k = int(retr_conf.get('top_k', 5))
        min_score = float(retr_conf.get('min_score', 0.28))
        # Retrieve docs and optionally sources for citations
        try:
            docs, scores, sources = self.retriever.retrieve_scored_with_sources(aug_emb, top_k=top_k, min_score=min_score)
        except Exception:
            docs, scores = self.retriever.retrieve_scored(aug_emb, top_k=top_k, min_score=min_score)
            sources = []
        if not docs:
            # Below similarity floor -> abstain
            abstain = "I don't know."
            self.history.add_turn(query, abstain)
            # Cache empty docs with abstain so we don't recompute repeatedly
            self.cache.add(query, q_emb, [], abstain, sources=[])
            if aug_text:
                self.cache.add(aug_text, aug_emb, [], abstain, sources=[])
            self._log_cache_stats()
            self._last_sources = []
            return abstain

        # Build limited, focused context
        q_context, retrieved_docs, sources = self._build_context(query, docs, scores, sources)
        # Guard: if context doesn't contain any meaningful query terms, abstain
        if not self._passes_term_overlap(query, retrieved_docs):
            response = "I don't know."
        else:
            response = self.generator.generate(q_context, hist_list)
        self._last_sources = list(sources or [])
        if self.show_citations and sources:
            cites = [s for s in sources if s]
            if cites:
                response = response.rstrip() + "\n\nSources:\n" + "\n".join(cites)

        # Optional pipeline-level post-generation guard
        if bool(retr_conf.get('pipeline_guard', False)):
            try:
                def _first_sentence_span(txt: str):
                    t = txt.strip()
                    if not t:
                        return 0
                    ends = []
                    for p in ['. ', '? ', '! ', '\n']:
                        i = t.find(p)
                        if i != -1:
                            ends.append(i + 1)
                    end = min(ends) if ends else len(t)
                    return end
                fs_end = _first_sentence_span(response)
                first_sentence = response.strip()[:fs_end].strip()
                low = first_sentence.lower()
                hedges = (
                    "i don't know",
                    "i do not know",
                    "cannot determine",
                    "can't determine",
                    "insufficient information",
                    "not enough information",
                    "no sufficient context",
                    "unknown based on the context",
                )
                strong_threshold = float(retr_conf.get('strong_threshold', max(min_score + 0.1, 0.35)))
                if any(h in low for h in hedges):
                    max_score = max(scores) if scores else 0.0
                    # If retrieval is confident, try to drop the hedge; else keep abstain
                    if max_score >= strong_threshold:
                        # Remove the first sentence span and any leading punctuation/whitespace
                        tail = response.strip()[fs_end:].lstrip(" .!\n\t")
                        if not tail:
                            # Second pass: include the explicit question to elicit an answer
                            q_context2 = q_context  # reuse same focused context
                            second = self.generator.generate(q_context2, hist_list).strip()
                            if second:
                                fs_end2 = _first_sentence_span(second)
                                fs2 = second[:fs_end2].strip().lower()
                                if any(h in fs2 for h in hedges):
                                    tail2 = second[fs_end2:].lstrip(" .!\n\t")
                                    response = tail2 or "I don't know."
                                else:
                                    response = second
                            else:
                                response = "I don't know."
                        else:
                            response = tail
                    else:
                        response = "I don't know."
            except Exception:
                pass

        # Final clean again for safety before storing
        response = self._clean_answer(response)
        # Update history and cache
        self.history.add_turn(query, response)
        self.cache.add(query, q_emb, retrieved_docs, response, sources=sources)
        if aug_text:
            self.cache.add(aug_text, aug_emb, retrieved_docs, response, sources=sources)
        self._log_cache_stats()
        return response

    def _clean_answer(self, text: str) -> str:
        try:
            if not text:
                return text
            t = str(text).strip()
            # Drop anything after our own Sources: marker if present
            if "\n\nSources:" in t:
                t = t.split("\n\nSources:", 1)[0].strip()
            # If the first sentence is a hedge, collapse to that
            def _first_span(s: str):
                ends = []
                for p in ['. ', '? ', '! ', '\n']:
                    i = s.find(p)
                    if i != -1:
                        ends.append(i + 1)
                return (min(ends) if ends else len(s))
            fs_end = _first_span(t)
            first = t[:fs_end].strip()
            low_first = first.lower()
            hedges = (
                "i don't know",
                "i do not know",
                "cannot determine",
                "can't determine",
                "insufficient information",
                "not enough information",
                "no sufficient context",
                "unknown based on the context",
            )
            if any(h in low_first for h in hedges):
                return "I don't know."
            # Remove common prefaces
            for lead in ("Explanation:", "Answer to the original question:", "Answer:", "Response:"):
                if t.startswith(lead):
                    t = t[len(lead):].lstrip()
            # Trim to first 2 sentences max
            ends = []
            for p in ['. ', '? ', '! ', '\n']:
                i1 = t.find(p)
                if i1 != -1:
                    ends.append(i1 + 1)
                    i2 = t.find(p, i1 + 1)
                    if i2 != -1:
                        ends.append(i2 + 1)
                        break
            if ends:
                t = t[: max(ends)].strip()
            # Compact excessive whitespace
            return ' '.join(t.split())
        except Exception:
            return text

    def _passes_term_overlap(self, query: str, docs) -> bool:
        try:
            if not query or not docs:
                return False
            if isinstance(docs, str):
                body = docs
            else:
                body = "\n".join(str(d) for d in docs)
            q = query.lower()
            body_low = body.lower()
            # Extract simple tokens >3 chars
            import re
            tokens = set(t for t in re.findall(r"[a-zA-Z0-9_]+", q) if len(t) >= 4)
            if not tokens:
                return True  # fallback, don't block
            hits = sum(1 for t in tokens if t in body_low)
            return hits >= 1
        except Exception:
            return True

    def _build_context(self, query, docs, scores=None, sources=None):
        """Select a limited set of docs and clip per-doc text to reduce drift.

        Returns (context_text, selected_docs, selected_sources)
        """
        retr_conf = self._retr_conf
        max_docs = int(retr_conf.get('max_context_docs', 3))
        max_chars = int(retr_conf.get('max_chars_per_doc', 1200))
        focus_top = bool(retr_conf.get('focus_top_when_strong', True))
        strong_thr = float(retr_conf.get('strong_threshold', max(float(retr_conf.get('min_score', 0.28)) + 0.1, 0.35)))

        docs = docs or []
        sources = sources or []
        # If retrieval is strong, narrow down to very top doc(s)
        if focus_top and scores:
            try:
                mx = max(scores)
            except Exception:
                mx = 0.0
            if mx >= strong_thr:
                max_docs = min(max_docs, 2)

        sel_docs = [str(d)[:max_chars] for d in (docs[:max_docs])]
        # Clean cut at whitespace if possible
        clipped = []
        for d in sel_docs:
            if len(d) >= max_chars:
                cut = d.rfind(' ', 0, max_chars)
                clipped.append(d[: cut if cut > 0 else max_chars].rstrip())
            else:
                clipped.append(d)
        sel_docs = clipped
        sel_sources = sources[: len(sel_docs)] if sources else []

        # Assemble final context with clear separators to avoid bleed
        separator = "\n\n----\n\n"
        body = separator.join(sel_docs)
        context_text = f"Question: {query}\n\n" + body
        return context_text, sel_docs, sel_sources

    def _compute_effective_retrieval(self):
        base = (self.config.get('retrieval') or {}).copy()
        mode = self.behavior_mode
        # Start from base defaults to keep user choices, then tweak core knobs
        min_score = float(base.get('min_score', 0.28))
        max_docs = int(base.get('max_context_docs', 3))
        max_chars = int(base.get('max_chars_per_doc', 1200))
        pipeline_guard = bool(base.get('pipeline_guard', True))
        focus_top = bool(base.get('focus_top_when_strong', True))
        top_k = int(base.get('top_k', 5))
        # Mode deltas
        if mode == 'strict':
            min_score = max(min_score, 0.32)
            max_docs = min(max_docs, 2)
            max_chars = min(max_chars, 900)
            pipeline_guard = True
            focus_top = True
        elif mode == 'loose':
            min_score = min(min_score, 0.24)
            max_docs = max(max_docs, 4)
            max_chars = max(max_chars, 1600)
            pipeline_guard = True
            focus_top = False
            top_k = max(top_k, 6)
        else:  # balanced
            # keep as provided; ensure guard on
            pipeline_guard = True if base.get('pipeline_guard') is None else pipeline_guard
        # Strong threshold tracks min_score + margin
        strong_thr = float(base.get('strong_threshold', max(min_score + 0.1, 0.35)))
        eff = base
        eff.update({
            'min_score': float(min_score),
            'max_context_docs': int(max_docs),
            'max_chars_per_doc': int(max_chars),
            'pipeline_guard': bool(pipeline_guard),
            'focus_top_when_strong': bool(focus_top),
            'strong_threshold': float(strong_thr),
            'top_k': int(top_k),
        })
        return eff

    def reset_history(self):
        # Clear persisted conversation history
        try:
            self.history.clear_history()
        except Exception:
            pass

    def _format_history_simple(self):
        # Convert ConversationHistory turns into simple strings
        try:
            turns = self.history.get_history()
            # Limit to last N turns from config
            n = int((self.config.get('conversation') or {}).get('history_length', 5))
            if n > 0:
                turns = turns[-n:]
            return [
                f"User: {t.get('user_input','')}\nBot: {t.get('bot_response','')}"
                for t in turns
            ]
        except Exception:
            return []

    def _log_cache_stats(self):
        try:
            stats = self.cache.get_stats()
            self.log.info(f"semantic_cache stats: {stats}")
        except Exception:
            pass

    # Compatibility shim for tests and scripts expecting .run()
    def run(self, query_or_history):
        """
        Accepts either a single query string or a list of utterances where the last
        item is the current user question and prior items are treated as history.
        """
        if isinstance(query_or_history, (list, tuple)) and query_or_history:
            # Seed prior user-only turns into history for context
            prior = list(query_or_history[:-1])
            for u in prior:
                try:
                    self.history.add_turn(str(u), "")
                except Exception:
                    pass
            query = str(query_or_history[-1])
        else:
            query = str(query_or_history)
        return self.process_query(query)
