from typing import List, Dict, Optional
import os
import numpy as np

class Generator:
    """Local text generator supporting Transformers or llama.cpp backends.

    Contract:
    - init(gen_config): dict with keys depending on backend
        transformers: {model_name_or_path, torch_dtype?, device_map?}
        llama-cpp: {model_path, n_ctx?, n_threads?, n_gpu_layers?}
    - generate(context_docs | str, history?) -> str
    - generate_response(context: str, history: List[str], max_new_tokens?) -> str
    """

    def __init__(self, gen_config: Optional[Dict] = None):
        # Accept None to auto-load from default config for backward compatibility
        self.config = gen_config or {}
        if not self.config:
            try:
                from src.utils.config import Config as _Config
                self.config = (_Config().get("generation") or {})
            except Exception:
                # Proceed with empty dict; will raise clearer error below if required keys missing
                self.config = {}
        self.backend = self.config.get("backend", "transformers")
        # Enforce offline mode by default to avoid external API calls
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        if self.backend == "transformers":
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            model_path = self.config.get("model_name_or_path")
            if not model_path:
                raise ValueError("generation.model_name_or_path must be set for transformers backend")
            if not os.path.exists(model_path):
                raise FileNotFoundError(
                    f"Transformers model path not found: {model_path}. "
                    "Update configs/default.yaml generation.model_name_or_path to your actual folder."
                )
            torch_dtype = self.config.get("torch_dtype", None)
            device_map = self.config.get("device_map", "auto")
            # Prefer a fast single-device CUDA load when possible, fall back to accelerate sharding if OOM
            kwargs = {"low_cpu_mem_usage": True}

            # Prefer fast tokenizer; if it fails, fall back to slow
            tok = None
            try:
                tok = AutoTokenizer.from_pretrained(
                    model_path,
                    local_files_only=True,
                    use_fast=True,
                    legacy=False,
                )
            except Exception:
                try:
                    tok = AutoTokenizer.from_pretrained(
                        model_path,
                        local_files_only=True,
                        use_fast=True,
                    )
                except Exception:
                    tok = AutoTokenizer.from_pretrained(
                        model_path,
                        local_files_only=True,
                        use_fast=False,
                    )
            self.tokenizer = tok
            # Load model: try full-GPU first for speed, else fall back to accelerate device_map
            try:
                mdl = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, **kwargs)
                if torch.cuda.is_available():
                    mdl = mdl.to("cuda")
                    if torch_dtype == "float16":
                        mdl = mdl.half()
                    elif torch_dtype == "bfloat16":
                        mdl = mdl.to(dtype=torch.bfloat16)
                self.model = mdl
            except torch.cuda.OutOfMemoryError:
                # Fallback to accelerate sharded load
                try:
                    from accelerate import init_empty_weights, load_checkpoint_and_dispatch  # noqa: F401
                    shard_kwargs = {"low_cpu_mem_usage": True, "device_map": device_map or "auto"}
                    if torch_dtype == "float16":
                        shard_kwargs["torch_dtype"] = torch.float16
                    elif torch_dtype == "bfloat16":
                        shard_kwargs["torch_dtype"] = torch.bfloat16
                    self.model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, **shard_kwargs)
                except Exception:
                    # As a last resort, load on CPU
                    self.model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, low_cpu_mem_usage=True)
            # Ensure pad/eos are consistent
            try:
                if getattr(self.tokenizer, "pad_token", None) is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                if getattr(self.model.config, "pad_token_id", None) is None and getattr(self.tokenizer, "eos_token_id", None) is not None:
                    self.model.config.pad_token_id = self.tokenizer.eos_token_id
            except Exception:
                pass
            try:
                self.model.eval()
            except Exception:
                pass

        elif self.backend == "llama-cpp":
            try:
                from llama_cpp import Llama
            except Exception as e:
                raise RuntimeError("llama-cpp-python not installed. Add it to requirements and install.") from e
            model_path = self.config.get("model_path")
            if not model_path:
                raise ValueError("generation.model_path must be set for llama-cpp backend")
            self.llm = Llama(
                model_path=model_path,
                n_ctx=int(self.config.get("n_ctx", 4096)),
                n_threads=int(self.config.get("n_threads", os.cpu_count() or 8)),
                n_gpu_layers=int(self.config.get("n_gpu_layers", 0)),
                verbose=False,
            )
            self.tokenizer = None
            self.model = None
        else:
            raise ValueError(f"Unsupported generation backend: {self.backend}")

    def _prepare_input(self, context: str, history: Optional[List[str]]) -> str:
        history = history or []
        history_text = "\n".join(history[-self.config.get("history_length", 5):])
        if history_text:
            history_text += "\n"
        instruction = (
            "You are a concise, grounded assistant. Answer ONLY using the Context below. "
            "If the Context does not contain the exact answer, but contains partial/approximate information (e.g., a year range or summary), provide the best-supported concise answer from the Context and state if the exact value is not specified. "
            "If the Context is irrelevant or truly insufficient, reply exactly: I don't know. "
            "Do not add external facts, opinions, or fillers. Keep answers short and factual.\n"
        )
        return f"{instruction}{history_text}Context:\n{context}\n\nResponse:"

    def _ensure_context_text(self, context_docs) -> str:
        if isinstance(context_docs, str):
            return context_docs
        if isinstance(context_docs, list):
            # Concatenate top documents
            return "\n\n".join(str(d) for d in context_docs[: self.config.get("concat_docs", 5)])
        return str(context_docs)

    def generate(self, context_docs, history: Optional[List[str]] = None) -> str:
        context = self._ensure_context_text(context_docs)
        return self.generate_response(context, history)

    def generate_response(self, context: str, history: Optional[List[str]] = None, max_new_tokens: Optional[int] = None) -> str:
        prompt = self._prepare_input(context, history)
        # Guarantee a minimum continuation length
        max_new_tokens = int(max_new_tokens or self.config.get("max_new_tokens", 256))
        if max_new_tokens < 64:
            max_new_tokens = 64
        temperature = float(self.config.get("temperature", 0.7))
        do_sample = bool(self.config.get("do_sample", True))
        min_new_tokens = int(self.config.get("min_new_tokens", max(16, min(64, max_new_tokens // 4))))
        # Output guard toggle
        guard_enabled = bool(self.config.get("output_guard", True))

        if self.backend == "transformers":
            import torch
            device = next(self.model.parameters()).device
            enc = self.tokenizer(prompt, return_tensors='pt')
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "min_new_tokens": min_new_tokens,
                "do_sample": do_sample,
                "temperature": temperature,
                "repetition_penalty": float(self.config.get("repetition_penalty", 1.05)),
                "no_repeat_ngram_size": int(self.config.get("no_repeat_ngram_size", 3)),
                "eos_token_id": getattr(self.tokenizer, "eos_token_id", None),
                "pad_token_id": getattr(self.model.config, "pad_token_id", None),
            }
            gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}
            with torch.no_grad():
                output = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **gen_kwargs,
                )
            # Decode only the continuation after the prompt tokens
            try:
                gen_ids = output[0][input_ids.shape[1]:]
                gen_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                if gen_text:
                    return self._apply_output_guard(gen_text) if guard_enabled else gen_text
            except Exception:
                pass
            # Fallback: decode full and split on anchor
            text = self.tokenizer.decode(output[0], skip_special_tokens=True)
            if "Response:" in text:
                after = text.split("Response:", 1)[-1].strip()
                if after:
                    return self._apply_output_guard(after) if guard_enabled else after
            # If still empty, try a second pass with mild sampling
            try:
                with torch.no_grad():
                    output2 = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=max_new_tokens,
                        min_new_tokens=min_new_tokens,
                        do_sample=True,
                        temperature=max(0.3, float(self.config.get("temperature", 0.7))),
                        repetition_penalty=float(self.config.get("repetition_penalty", 1.05)),
                        no_repeat_ngram_size=int(self.config.get("no_repeat_ngram_size", 3)),
                        eos_token_id=getattr(self.tokenizer, "eos_token_id", None),
                        pad_token_id=getattr(self.model.config, "pad_token_id", None),
                    )
                gen_ids2 = output2[0][input_ids.shape[1]:]
                gen_text2 = self.tokenizer.decode(gen_ids2, skip_special_tokens=True).strip()
                if gen_text2:
                    return self._apply_output_guard(gen_text2) if guard_enabled else gen_text2
            except Exception:
                pass
            return self._apply_output_guard(text.strip()) if guard_enabled else text.strip()

        elif self.backend == "llama-cpp":
            # Sampling controls
            top_p = float(self.config.get("top_p", 0.9))
            top_k = int(self.config.get("top_k", 40))
            repeat_penalty = float(self.config.get("repeat_penalty", 1.1))
            stop = self.config.get("stop") or None
            res = self.llm(
                prompt=prompt,
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repeat_penalty=repeat_penalty,
                stop=stop,
            )
            text = res.get("choices", [{}])[0].get("text", "").strip()
            return self._apply_output_guard(text) if guard_enabled else text

        else:
            raise ValueError(f"Unsupported generation backend: {self.backend}")

    def evaluate_factuality(self, generated_response: str, context: str) -> bool:
        # Placeholder hook for a proper factuality module
        return True

    def generate_multi_turn_response(self, context: str, history: List[str], num_turns: int) -> List[str]:
        responses = []
        for _ in range(num_turns):
            response = self.generate_response(context, history)
            responses.append(response)
            history.append(response)
        return responses

    def _apply_output_guard(self, text: str) -> str:
        """If the first sentence contains an explicit "I don't know"-style admission, return only that."""
        if not text:
            return text
        t = text.strip()
        # Find end of first sentence conservatively
        end = len(t)
        for p in ['. ', '? ', '! ', '\n']:
            idx = t.find(p)
            if idx != -1:
                end = min(end, idx + 1)
        first = t[:end].strip()
        low = first.lower()
        hedges = [
            "i don't know",
            "i do not know",
            "cannot determine",
            "can't determine",
            "insufficient information",
            "not enough information",
            "no sufficient context",
            "unknown based on the context",
        ]
        if any(h in low for h in hedges):
            return first
        return t
