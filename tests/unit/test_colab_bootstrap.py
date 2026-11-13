from __future__ import annotations

import ast
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("name", ["colab-pilot.yaml", "colab-pilot-fallback.yaml"])
def test_colab_profiles_load_with_cache_disabled(project_root: Path, name: str) -> None:
    from budgetroute.config import load_config

    config = load_config(project_root / "configs/benchmarks" / name, apply_environment=False)
    assert config.benchmark.cache.mode == "off"
    assert config.mode == "gpu"


def test_setup_imports_newly_installed_source_in_current_kernel(
    tmp_path: Path,
    project_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    notebook = json.loads((project_root / "notebooks/routing_pilot_colab.ipynb").read_text())
    source = "".join(next(c for c in notebook["cells"] if c["id"] == "setup")["source"])
    clone = tmp_path / "clone"
    (clone / "src").mkdir(parents=True)
    marker = clone / "src/_budgetroute_notebook_marker.py"
    marker.write_text("PINNED_SOURCE = True\n")

    class Paths(ast.NodeTransformer):
        def visit_Constant(self, node: ast.Constant) -> ast.Constant:
            if node.value == "/content/budgetroute-pilot-source":
                return ast.copy_location(ast.Constant(str(clone)), node)
            return node

    tree = ast.fix_missing_locations(Paths().visit(ast.parse(source)))
    answers = iter(["a" * 40, "primary"])
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=0))
    monkeypatch.setattr(
        subprocess, "check_output", lambda args, **kwargs: "a" * 40 if "rev-parse" in args else ""
    )
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.delitem(sys.modules, "budgetroute", raising=False)
    monkeypatch.delitem(sys.modules, "transformers", raising=False)
    exec(compile(tree, "notebook-setup", "exec"), {"input": lambda prompt: next(answers)})
    found = importlib.util.find_spec("_budgetroute_notebook_marker")
    assert found is not None and Path(found.origin or "") == marker


def test_setup_rejects_kernel_that_already_imported_project(
    project_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    notebook = json.loads((project_root / "notebooks/routing_pilot_colab.ipynb").read_text())
    source = "".join(next(c for c in notebook["cells"] if c["id"] == "setup")["source"])
    monkeypatch.setitem(sys.modules, "budgetroute", SimpleNamespace())

    def must_not_run(*args, **kwargs):
        pytest.fail("setup mutated the checkout or dependencies before requiring a restart")

    monkeypatch.setattr(subprocess, "run", must_not_run)
    monkeypatch.setattr(subprocess, "check_output", must_not_run)
    answers = iter(["b" * 40, "primary"])
    with pytest.raises(RuntimeError, match="Restart"):
        exec(compile(source, "notebook-setup", "exec"), {"input": lambda prompt: next(answers)})


def test_component_failure_keeps_environment_and_config(
    project_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from budgetroute.exceptions import BackendError
    from budgetroute.validation.colab import run_component_checks

    environment = {
        "hardware": {"gpu": "TEST FIXTURE NOT REAL"},
        "packages": {"fixture": "0"},
        "gpu_available": True,
    }
    monkeypatch.setattr(
        "budgetroute.validation.colab.validate_colab_environment", lambda: environment
    )
    monkeypatch.setattr(
        "budgetroute.validation.colab.source_identity", lambda: {"commit": "a" * 40, "dirty": False}
    )

    class FailedService:
        def initialize(self) -> None:
            raise BackendError("injected component failure")

        def close(self) -> None:
            pass

    monkeypatch.setattr(
        "budgetroute.validation.colab.build_service", lambda config: FailedService()
    )
    with pytest.raises(BackendError, match="injected"):
        run_component_checks(project_root / "configs/benchmarks/colab-pilot.yaml", tmp_path)
    assert json.loads((tmp_path / "components.json").read_text())["status"] == "failed"
    assert json.loads((tmp_path / "environment.json").read_text()) == environment
    assert (tmp_path / "resolved-config.yaml").is_file()


def test_retrieval_probe_persists_and_searches_reloaded_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from budgetroute.config import RetrievalConfig
    from budgetroute.retrieval.embeddings import FakeEmbedder
    from budgetroute.retrieval.index import ExactCosineIndex
    from budgetroute.validation import colab

    # Exercise real file persistence offline. Only the Colab probe can prove FAISS/MiniLM.
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.txt").write_text("Paris is the capital of France.\n")
    (corpus / "b.txt").write_text("Bananas are a yellow fruit.\n")
    config = RetrievalConfig(corpus_path=corpus, index_path=tmp_path / "index.npz")
    monkeypatch.setattr(colab, "FaissHNSWIndex", ExactCosineIndex)
    probe = getattr(colab, "_probe_retrieval", None)
    assert callable(probe), "the component gate needs a persistence round-trip probe"
    report = probe(config, FakeEmbedder(128))
    assert config.index_path.is_file()
    assert report["hits"][0]["chunk_id"] == report["reloaded_hits"][0]["chunk_id"]
    assert "Paris" in report["reloaded_hits"][0]["text"]
