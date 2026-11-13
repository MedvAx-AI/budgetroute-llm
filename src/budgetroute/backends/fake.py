"""Deterministic, offline fake generation backend for tests, CI, and demos."""

from __future__ import annotations

import hashlib
import re
import time
from typing import Any

from budgetroute.config import BackendConfig
from budgetroute.exceptions import BackendError
from budgetroute.schemas import BackendGeneration, BackendName, GenerationRequest


def _stable_fraction(text: str) -> float:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64 - 1)


class FakeBackend:
    """Simulate backend quality and latency without pretending to be a real model."""

    def __init__(self, name: BackendName, config: BackendConfig) -> None:
        self.name = name
        self.config = config
        self._initialized = False

    def initialize(self) -> None:
        self._initialized = True

    def health(self) -> dict[str, Any]:
        return {
            "name": self.name.value,
            "type": "fake",
            "initialized": self._initialized,
            "ready": self._initialized,
        }

    def token_count(self, text: str) -> int:
        return len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE))

    def _known_answer(self, prompt: str) -> str:
        lowered = prompt.lower()
        if "capital of france" in lowered:
            return "Paris"
        if "2 + 2" in lowered or "2+2" in lowered:
            return "4"
        if "12 items" in lowered and "3" in lowered and "removed" in lowered:
            return "9"
        if "red planet" in lowered:
            return "Mars"
        if "budgetroute" in lowered and "orchid" in lowered:
            return "The Orchid release introduced retrieval-aware routing."
        if "project codename" in lowered and "reference context" in lowered:
            match = re.search(r"codename(?: is|:)\s+([A-Za-z-]+)", prompt, re.IGNORECASE)
            if match:
                return match.group(1)
        if "sentiment" in lowered:
            if any(word in lowered for word in ("excellent", "love", "helpful", "great")):
                return "positive"
            if any(word in lowered for word in ("awful", "hate", "broken", "poor")):
                return "negative"
            return "neutral"
        return "A deterministic fake response."

    def _answer_and_confidence(self, request: GenerationRequest) -> tuple[str, float]:
        metadata = request.metadata
        if metadata.get("must_abstain"):
            return "ABSTAIN", 0.98
        reference = metadata.get("fake_reference_answer")
        explicit_success = metadata.get("fake_small_success")
        if self.name == BackendName.LARGE:
            answer = str(reference) if reference is not None else self._known_answer(request.prompt)
            return answer, min(0.99, max(0.75, self.config.quality))
        success = (
            True
            if reference is not None and "reference context:" in request.prompt.lower()
            else (
                bool(explicit_success)
                if explicit_success is not None
                else _stable_fraction(request.prompt) <= self.config.quality
            )
        )
        if success:
            answer = str(reference) if reference is not None else self._known_answer(request.prompt)
            return answer, min(0.95, max(0.66, self.config.quality))
        return "I am not sure.", min(0.49, max(0.05, self.config.quality / 2))

    def generate(self, request: GenerationRequest) -> BackendGeneration:
        if not self._initialized:
            raise BackendError(f"fake {self.name.value} backend is not initialized")
        for marker in self.config.fail_on_substrings:
            if marker.lower() in request.prompt.lower():
                raise BackendError(f"fake {self.name.value} backend simulated a configured failure")
        start = time.perf_counter_ns()
        if self.config.artificial_latency_ms:
            time.sleep(self.config.artificial_latency_ms / 1000.0)
        answer, confidence = self._answer_and_confidence(request)
        elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000
        return BackendGeneration(
            text=answer,
            backend=self.name,
            input_tokens=self.token_count(request.prompt),
            output_tokens=self.token_count(answer),
            confidence=confidence,
            generation_ms=elapsed_ms,
            usage_is_measured=True,
            time_to_first_token_ms=elapsed_ms,
            metadata={"fake": True, "configured_quality": self.config.quality},
        )

    def generate_batch(self, requests: list[GenerationRequest]) -> list[BackendGeneration]:
        return [self.generate(request) for request in requests]

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name.value,
            "type": "fake",
            "fake": True,
            "configured_quality": self.config.quality,
            "artificial_latency_ms": self.config.artificial_latency_ms,
            "max_concurrency": self.config.max_concurrency,
            "input_cost_units_per_1k_tokens": self.config.input_cost_units_per_1k_tokens,
            "output_cost_units_per_1k_tokens": self.config.output_cost_units_per_1k_tokens,
        }

    def cleanup(self) -> None:
        self._initialized = False
