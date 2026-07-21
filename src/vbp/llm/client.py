from __future__ import annotations

from typing import Protocol

from .schemas import CorrectionBatch


class LLMClient(Protocol):
    def correct(self, prompt: str) -> CorrectionBatch: ...
    def reflect(self, prompt: str) -> str: ...


class FakeLLM:
    """Deterministic stand-in for tests. Returns scripted outputs in order."""
    def __init__(self, corrections: list[CorrectionBatch], reflections: list[str]):
        self._corrections = list(corrections)
        self._reflections = list(reflections)
        self.calls: list[dict] = []

    def correct(self, prompt: str) -> CorrectionBatch:
        self.calls.append({"kind": "correct", "prompt": prompt})
        return self._corrections.pop(0) if self._corrections else CorrectionBatch(corrections=[])

    def reflect(self, prompt: str) -> str:
        self.calls.append({"kind": "reflect", "prompt": prompt})
        return self._reflections.pop(0) if self._reflections else ""


class AnthropicClient:
    """Real Claude client. NOT exercised by the test suite (needs ANTHROPIC_API_KEY).
    Correction = Haiku 4.5 (accepts temperature, structured output, NO effort/thinking).
    Reflection = Sonnet 5 (adaptive thinking, NO temperature - it 400s)."""
    def __init__(self, correct_model="claude-haiku-4-5", reflect_model="claude-sonnet-5",
                 temp_correct=0.0, reflect_effort="medium", log=None):
        import anthropic
        self._client = anthropic.Anthropic()
        self.correct_model = correct_model
        self.reflect_model = reflect_model
        self.temp_correct = temp_correct
        self.reflect_effort = reflect_effort
        self.log = log  # callable(dict) for audit, optional

    def correct(self, prompt: str) -> CorrectionBatch:
        import hashlib
        resp = self._client.messages.parse(
            model=self.correct_model, max_tokens=4000,
            temperature=self.temp_correct,          # OK on Haiku 4.5
            output_format=CorrectionBatch,
            messages=[{"role": "user", "content": prompt}],
        )
        if self.log:
            self.log({"kind": "correct", "model": self.correct_model,
                      "prompt_sha": hashlib.sha256(prompt.encode()).hexdigest(),
                      "request_id": resp._request_id, "raw": resp.to_dict(),
                      "corrections": resp.parsed_output.model_dump()})   # for replay
        return resp.parsed_output

    def reflect(self, prompt: str) -> str:
        import hashlib
        resp = self._client.messages.create(
            model=self.reflect_model, max_tokens=4000,
            output_config={"effort": self.reflect_effort},  # NO temperature on Sonnet 5
            thinking={"type": "adaptive"},
            messages=[{"role": "user", "content": prompt}],
        )
        text = next((b.text for b in resp.content if b.type == "text"), "")
        if self.log:
            self.log({"kind": "reflect", "model": self.reflect_model,
                      "prompt_sha": hashlib.sha256(prompt.encode()).hexdigest(),
                      "request_id": resp._request_id, "raw": resp.to_dict(),
                      "text": text})                                     # for replay
        return text


class ReplayLLM:
    """Deterministic replay from an audit log (list of dicts written by AnthropicClient.log).
    Matches each incoming prompt by sha256 to the logged response - so --replay reproduces a
    run bit-for-bit without calling Anthropic. Falls back to call-order if a hash is missing."""
    def __init__(self, log_entries: list[dict]):
        import hashlib
        self._hashlib = hashlib
        self._by_sha = {e["prompt_sha"]: e for e in log_entries if "prompt_sha" in e}
        self._corr_order = [e for e in log_entries if e["kind"] == "correct"]
        self._refl_order = [e for e in log_entries if e["kind"] == "reflect"]
        self._ci = self._ri = 0

    def _lookup(self, prompt, order, idx_attr):
        sha = self._hashlib.sha256(prompt.encode()).hexdigest()
        if sha in self._by_sha:
            return self._by_sha[sha]
        i = getattr(self, idx_attr)
        setattr(self, idx_attr, i + 1)
        return order[i]

    def correct(self, prompt: str) -> CorrectionBatch:
        return CorrectionBatch(**self._lookup(prompt, self._corr_order, "_ci")["corrections"])

    def reflect(self, prompt: str) -> str:
        return self._lookup(prompt, self._refl_order, "_ri")["text"]
