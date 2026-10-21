from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "report",
    [
        {"status": "completed_with_errors", "fake": False, "checks": {}},
        {"status": "passed", "fake": True, "checks": {}},
        {
            "status": "passed",
            "fake": False,
            "checks": {"retrieval": {"status": "passed", "fake": True}},
        },
        {
            "status": "passed",
            "fake": False,
            "checks": {"generation": {"status": "failed", "fake": False}},
        },
    ],
)
def test_component_gate_rejects_hidden_fake_or_failed_checks(report: dict[str, object]) -> None:
    from budgetroute.exceptions import ArtifactError
    from budgetroute.validation.colab import validate_component_report

    with pytest.raises(ArtifactError):
        validate_component_report(report)


def test_missing_evidence_is_not_accepted(tmp_path: Path) -> None:
    from budgetroute.exceptions import ArtifactError
    from budgetroute.validation.colab import validate_run_artifacts

    with pytest.raises(ArtifactError):
        validate_run_artifacts(tmp_path)


def test_passed_labels_without_actual_component_evidence_are_rejected() -> None:
    from budgetroute.exceptions import ArtifactError
    from budgetroute.validation.colab import REQUIRED_CHECKS, validate_component_report

    report = {
        "status": "passed",
        "fake": False,
        "checks": {name: {"status": "passed", "fake": False} for name in REQUIRED_CHECKS},
    }
    with pytest.raises(ArtifactError, match="evidence"):
        validate_component_report(report)


def test_source_notebook_compiles_and_calls_versioned_stages(project_root: Path) -> None:
    notebook = json.loads((project_root / "notebooks/routing_pilot_colab.ipynb").read_text())
    cells = notebook["cells"]
    stages = []
    for cell in cells:
        if cell["cell_type"] != "code":
            continue
        assert cell["outputs"] == [] and cell["execution_count"] is None
        tree = ast.parse("".join(cell["source"]))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "run_pilot_stage"
            ):
                stages.extend(
                    k.value.value
                    for k in node.keywords
                    if k.arg == "stage" and isinstance(k.value, ast.Constant)
                )
    assert stages == ["collect", "fit", "diagnostic", "timing", "summarize", "export"]
    assert notebook["metadata"]["budgetroute"]["role"] == "pilot_not_confirmation"


@pytest.mark.parametrize(
    "source",
    [
        {"commit": "a" * 40, "dirty": True},
        {"commit": "b" * 40, "dirty": False},
        {"commit": None, "dirty": None},
    ],
)
def test_source_gate_rejects_dirty_wrong_or_unknown_commit(
    monkeypatch: pytest.MonkeyPatch, source: dict[str, object]
) -> None:
    from budgetroute.exceptions import ArtifactError
    from budgetroute.validation.colab import source_identity

    monkeypatch.setenv("BUDGETROUTE_REVIEWED_SHA", "a" * 40)
    monkeypatch.setattr("budgetroute.validation.colab.git_metadata", lambda path: source)
    with pytest.raises(ArtifactError, match="source"):
        source_identity()


def test_empty_source_sha_stops_notebook_before_install(project_root: Path) -> None:
    notebook = json.loads((project_root / "notebooks/routing_pilot_colab.ipynb").read_text())
    setup = next(cell for cell in notebook["cells"] if cell["id"] == "setup")
    with pytest.raises(AssertionError, match="explicit"):
        exec(
            compile("".join(setup["source"]), "notebook-setup", "exec"),
            {"input": lambda prompt: ""},
        )
