from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from budgetroute.evaluation.dataset import load_dataset, load_dataset_spec


def fixture_data(tmp_path: Path, project_root: Path) -> tuple[Path, Path, list[dict[str, object]]]:
    from budgetroute.experiments.pilot_data import normalized_question_hash

    spec = load_dataset_spec(project_root / "configs/datasets/mmlu.yaml")
    spec = spec.model_copy(update={"prompt_version": "direct-mcq-v2", "name": "renamed"})
    spec_path = tmp_path / "spec.yaml"
    import yaml

    spec_path.write_text(yaml.safe_dump(spec.model_dump()), encoding="utf-8")
    entries = [{"source_index": 0, "group_sha256": normalized_question_hash("Question 0?")}]
    exclusions = tmp_path / "exclusions.json"
    exclusions.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source": spec.source,
                "revision": spec.revision,
                "split": spec.split,
                "parent_sha256": "5b25d20f38e91120bd3303ad32cc35d54ae9eb0326468532f41dcfaa2b31ede7",
                "entries": entries,
                "entries_sha256": hashlib.sha256(
                    json.dumps(entries, sort_keys=True).encode()
                ).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    rows = [
        {
            "question": f"Question {i}?",
            "choices": ["a", "b", "c", "d"],
            "answer": 0,
            "subject": str(i % 5),
        }
        for i in range(230)
    ]
    rows.append({**rows[0], "question": "  QUESTION   0? "})
    return spec_path, exclusions, rows


def test_pilot_is_disjoint_deterministic_and_excludes_renamed_duplicate(
    tmp_path: Path,
    project_root: Path,
) -> None:
    from budgetroute.experiments.pilot_data import prepare_pilot_data

    spec, exclusions, rows = fixture_data(tmp_path, project_root)
    with patch("budgetroute.experiments.pilot_data._load_upstream", return_value=rows):
        first = prepare_pilot_data(spec, exclusions, tmp_path / "one")
        second = prepare_pilot_data(spec, exclusions, tmp_path / "two")
    assert first == second
    assert (len(first.train_ids), len(first.calibration_ids), len(first.test_ids)) == (120, 40, 40)
    assert not set(first.train_ids) & set(first.test_ids)
    records = load_dataset(tmp_path / "one/pilot.jsonl")
    assert len(records) == len({r.group_id for r in records}) == 200
    assert not {0, 230} & {r.metadata["source_index"] for r in records}


def test_bad_or_missing_exclusions_fail_before_download(tmp_path: Path, project_root: Path) -> None:
    from budgetroute.exceptions import BudgetRouteError
    from budgetroute.experiments.pilot_data import prepare_pilot_data

    spec, exclusions, _ = fixture_data(tmp_path, project_root)
    raw = json.loads(exclusions.read_text())
    raw["entries"][0]["source_index"] = 5
    exclusions.write_text(json.dumps(raw))
    with pytest.raises(BudgetRouteError, match="exclusion"):
        prepare_pilot_data(spec, exclusions, tmp_path / "out")
    with pytest.raises(BudgetRouteError, match="exclusion"):
        prepare_pilot_data(spec, tmp_path / "missing", tmp_path / "out")


def test_historical_recovery_refuses_wrong_parent(tmp_path: Path, project_root: Path) -> None:
    from budgetroute.exceptions import BudgetRouteError
    from budgetroute.experiments.pilot_data import build_historical_exclusions

    source = tmp_path / "wrong.jsonl"
    source.write_text("{}\n")
    with pytest.raises(BudgetRouteError, match="verified original"):
        build_historical_exclusions(
            source, project_root / "configs/datasets/mmlu.yaml", tmp_path / "out.json"
        )


@pytest.mark.parametrize(
    "answer,score",
    [
        ("A", 1),
        ("(a)", 1),
        (" A ", 1),
        ("A or B", 0),
        ("The answer might be A", 0),
        ("", 0),
        ("E", 0),
        ("(A", 0),
    ],
)
def test_strict_mcq(answer: str, score: int) -> None:
    from budgetroute.evaluation.evaluators import strict_mcq

    assert strict_mcq(answer, "A", 4).score == score


def test_truncated_valid_letter_scores_zero_in_harness(fake_config: object) -> None:
    from budgetroute.config import AppConfig
    from budgetroute.evaluation.harness import evaluate_records
    from budgetroute.inference.service import build_service
    from budgetroute.schemas import BenchmarkRecord

    assert isinstance(fake_config, AppConfig)
    config = fake_config.model_copy(
        update={"routing": fake_config.routing.model_copy(update={"policy": "always_large"})}
    )
    service = build_service(config)
    row = BenchmarkRecord(
        id="q",
        category="test",
        prompt="Pick A",
        reference_answer="A",
        evaluation_type="classification",
        metadata={"prompt_version": "direct-mcq-v2", "choice_count": 4},
    )
    try:
        service.initialize()
        response = service.generate(
            __import__("budgetroute.schemas", fromlist=["GenerationRequest"]).GenerationRequest(
                prompt="Pick A"
            )
        )
        response.answer = "A"
        response.execution.answer_truncated = True
        with patch.object(service, "generate", return_value=response):
            predictions, _, _ = evaluate_records(service, [row], fake=True)
        assert predictions[0].quality_score == 0
        assert predictions[0].answer_truncated is True
    finally:
        service.close()
