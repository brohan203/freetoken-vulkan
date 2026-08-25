"""Incremental token-ID session over fully resident gpt-oss decode."""
from __future__ import annotations

import torch

from .resident_decode import ResidentDecodeWorkspace, resident_decode_model_step


class ResidentDecodeSession:
    """Persistent KV context with caller-controlled token-ID appends.

    No chat protocol is invented here. Callers own formatting and append exact
    token IDs between generations.
    """

    def __init__(self, model, tokenizer, capacity: int = 384):
        if model.resident_projections is None or model.h_lm_head is None:
            raise RuntimeError("resident projections and LM head must be pinned")
        self.model = model
        self.tokenizer = tokenizer
        self.capacity = capacity
        self.workspace = ResidentDecodeWorkspace(
            model.ext, model.cfg, model.cfg.num_hidden_layers, capacity
        )
        self.position = 0
        self.pending_token: int | None = None
        self.token_ids: list[int] = []
        self._pending_emitted = False
        self._started = False
        self._freed = False

    @torch.no_grad()
    def prefill_token_ids(self, token_ids: list[int]) -> int:
        self._require_live()
        if self._started:
            raise RuntimeError("session has already been initialized")
        if not token_ids:
            raise ValueError("prefill requires at least one token")
        if len(token_ids) >= self.capacity:
            raise ValueError("prefill exceeds resident session capacity")
        input_ids = torch.tensor([token_ids], dtype=torch.long)
        cache = self.model.make_kv_cache(self.capacity)
        logits = self.model.forward(
            input_ids, past_kv=cache, past_len=0, only_last_logits=True
        )
        cache.advance(len(token_ids))
        self.workspace.load_kv_cache(cache)
        self.position = len(token_ids)
        self.token_ids = list(token_ids)
        self.pending_token = int(logits[0, -1].argmax())
        self._pending_emitted = False
        self._started = True
        return self.pending_token

    def prefill_text(self, text: str) -> int:
        return self.prefill_token_ids(self.tokenizer.encode(text))

    @torch.no_grad()
    def _commit_token(self, token_id: int) -> torch.Tensor:
        if self.position >= self.capacity:
            raise ValueError("resident session capacity exhausted")
        logits, _ = resident_decode_model_step(
            self.model, self.workspace, int(token_id), self.position
        )
        assert logits is not None
        self.token_ids.append(int(token_id))
        self.position += 1
        return logits

    @torch.no_grad()
    def generate(self, max_new_tokens: int) -> list[int]:
        """Emit tokens and retain the final emitted token as pending context."""
        self._require_started()
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        generated: list[int] = []
        for _ in range(max_new_tokens):
            if self._pending_emitted:
                assert self.pending_token is not None
                logits = self._commit_token(self.pending_token)
                self.pending_token = int(logits[0].argmax())
                self._pending_emitted = False
            assert self.pending_token is not None
            generated.append(self.pending_token)
            self._pending_emitted = True
        return generated

    @torch.no_grad()
    def append_token_ids(self, token_ids: list[int]) -> int:
        """Commit pending output, append exact tokens, and predict next ID."""
        self._require_started()
        if not token_ids:
            raise ValueError("append_token_ids requires at least one token")
        if self._pending_emitted:
            assert self.pending_token is not None
            self._commit_token(self.pending_token)
        self._pending_emitted = False
        self.pending_token = None
        logits = None
        for token_id in token_ids:
            logits = self._commit_token(int(token_id))
        assert logits is not None
        self.pending_token = int(logits[0].argmax())
        return self.pending_token

    def decode(self) -> str:
        ids = list(self.token_ids)
        if self._pending_emitted and self.pending_token is not None:
            ids.append(self.pending_token)
        return self.tokenizer.decode(ids, skip_special_tokens=False)

    def free(self) -> None:
        if self._freed:
            return
        self.workspace.free()
        self._freed = True

    def _require_live(self) -> None:
        if self._freed:
            raise RuntimeError("ResidentDecodeSession has been freed")

    def _require_started(self) -> None:
        self._require_live()
        if not self._started:
            raise RuntimeError("prefill the session before generation")

    def __enter__(self) -> "ResidentDecodeSession":
        self._require_live()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.free()

    def __del__(self) -> None:
        try:
            self.free()
        except Exception:
            pass
