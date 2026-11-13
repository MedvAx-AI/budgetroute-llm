"""Deterministic evaluator registry for benchmark answer types."""

from __future__ import annotations

import math
import re
import string
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass

from budgetroute.schemas import BenchmarkRecord, EvaluationType


@dataclass(frozen=True)
class EvaluationResult:
    score: float
    metric_name: str
    normalized_answer: str


def normalize_text(text: str) -> str:
    lowered = text.lower().strip()
    without_punctuation = "".join(
        character if character not in string.punctuation else " " for character in lowered
    )
    return " ".join(without_punctuation.split())


def exact_match(answer: str, reference: str) -> EvaluationResult:
    normalized = normalize_text(answer)
    return EvaluationResult(
        float(normalized == normalize_text(reference)), "exact_match", normalized
    )


def token_f1(answer: str, reference: str) -> EvaluationResult:
    normalized = normalize_text(answer)
    prediction_tokens = normalized.split()
    reference_tokens = normalize_text(reference).split()
    if not prediction_tokens and not reference_tokens:
        score = 1.0
    elif not prediction_tokens or not reference_tokens:
        score = 0.0
    else:
        common = Counter(prediction_tokens) & Counter(reference_tokens)
        overlap = sum(common.values())
        precision = overlap / len(prediction_tokens)
        recall = overlap / len(reference_tokens)
        score = 2 * precision * recall / (precision + recall) if overlap else 0.0
    return EvaluationResult(score, "token_f1", normalized)


def _extract_number(text: str) -> float | None:
    boxed = re.findall(r"\\boxed\{([^{}]+)\}", text)
    candidate = boxed[-1] if boxed else text
    final_markers = re.split(r"(?:final answer|answer is|therefore)\s*[:=]?", candidate, flags=re.I)
    candidate = final_markers[-1]
    pattern = (
        r"[-+]?(?:\d+(?:,\d{3})*\.?\d*|\.\d+)(?:[eE][-+]?\d+)?(?:\s*/\s*[-+]?\d+(?:\.\d+)?)?%?"
    )
    matches = re.findall(pattern, candidate)
    if not matches:
        return None
    raw = matches[-1].replace(",", "").replace(" ", "")
    try:
        percentage = raw.endswith("%")
        raw = raw.removesuffix("%")
        if "/" in raw:
            numerator, denominator = raw.split("/", 1)
            value = float(numerator) / float(denominator)
        else:
            value = float(raw)
        return value / 100.0 if percentage else value
    except (ValueError, ZeroDivisionError):
        return None


def numeric(answer: str, reference: str, tolerance: float = 1e-9) -> EvaluationResult:
    normalized = normalize_text(answer)
    predicted = _extract_number(answer)
    expected = _extract_number(reference)
    score = float(
        predicted is not None
        and expected is not None
        and math.isclose(predicted, expected, rel_tol=tolerance, abs_tol=tolerance)
    )
    return EvaluationResult(score, "numeric_correctness", normalized)


def classification(answer: str, reference: str) -> EvaluationResult:
    normalized = normalize_text(answer)
    expected = normalize_text(reference)
    first_label = normalized.split()[0] if normalized else ""
    return EvaluationResult(
        float(normalized == expected or first_label == expected),
        "classification_accuracy",
        normalized,
    )


def strict_mcq(answer: str, reference: str, choice_count: int) -> EvaluationResult:
    text = answer.strip().upper()
    match = re.fullmatch(r"([A-Z])|\(([A-Z])\)", text)
    label = (match.group(1) or match.group(2)) if match else ""
    valid = 1 <= choice_count <= 26 and label in [chr(65 + i) for i in range(choice_count)]
    return EvaluationResult(
        float(valid and label == reference.strip().upper()), "strict_mcq_accuracy", label.lower()
    )


def abstention(answer: str, must_abstain: bool) -> EvaluationResult:
    normalized = normalize_text(answer)
    did_abstain = normalized.startswith("abstain") or "i cannot answer" in normalized
    return EvaluationResult(
        float(did_abstain == must_abstain), "abstention_correctness", normalized
    )


def keyword(answer: str, keywords: list[str]) -> EvaluationResult:
    normalized = normalize_text(answer)
    normalized_keywords = [normalize_text(item) for item in keywords]
    if not normalized_keywords:
        score = 0.0
    else:
        padded = f" {normalized} "
        score = sum(f" {item} " in padded for item in normalized_keywords) / len(
            normalized_keywords
        )
    return EvaluationResult(score, "keyword_coverage", normalized)


Evaluator = Callable[[str, BenchmarkRecord], EvaluationResult]


def _exact_adapter(answer: str, record: BenchmarkRecord) -> EvaluationResult:
    return exact_match(answer, record.reference_answer)


def _f1_adapter(answer: str, record: BenchmarkRecord) -> EvaluationResult:
    return token_f1(answer, record.reference_answer)


def _numeric_adapter(answer: str, record: BenchmarkRecord) -> EvaluationResult:
    tolerance = float(record.metadata.get("tolerance", 1e-9))
    return numeric(answer, record.reference_answer, tolerance)


def _classification_adapter(answer: str, record: BenchmarkRecord) -> EvaluationResult:
    if record.metadata.get("prompt_version") == "direct-mcq-v2":
        return strict_mcq(answer, record.reference_answer, int(record.metadata["choice_count"]))
    return classification(answer, record.reference_answer)


def _abstention_adapter(answer: str, record: BenchmarkRecord) -> EvaluationResult:
    return abstention(answer, record.must_abstain)


def _keyword_adapter(answer: str, record: BenchmarkRecord) -> EvaluationResult:
    raw = record.metadata.get("keywords", [record.reference_answer])
    keywords = [str(item) for item in raw] if isinstance(raw, list) else [str(raw)]
    return keyword(answer, keywords)


EVALUATORS: dict[EvaluationType, Evaluator] = {
    EvaluationType.EXACT_MATCH: _exact_adapter,
    EvaluationType.TOKEN_F1: _f1_adapter,
    EvaluationType.NUMERIC: _numeric_adapter,
    EvaluationType.CLASSIFICATION: _classification_adapter,
    EvaluationType.ABSTENTION: _abstention_adapter,
    EvaluationType.KEYWORD: _keyword_adapter,
}


def evaluate_answer(answer: str, record: BenchmarkRecord) -> EvaluationResult:
    return EVALUATORS[record.evaluation_type](answer, record)
