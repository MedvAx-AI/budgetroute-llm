import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from budgetroute.exceptions import ArtifactError
from budgetroute.validation.colab import validate_colab_environment


def test_missing_colab_package_is_an_artifact_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Path, "is_dir", lambda self: True)

    def missing(name: str):
        raise ModuleNotFoundError(name)

    monkeypatch.setattr("budgetroute.validation.colab.importlib.util.find_spec", missing)
    with pytest.raises(ArtifactError, match="Colab"):
        validate_colab_environment()


def test_no_gpu_cannot_pass_remote_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Path, "is_dir", lambda self: True)
    monkeypatch.setattr(
        "budgetroute.validation.colab.importlib.util.find_spec", lambda name: object()
    )
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    )
    with pytest.raises(ArtifactError, match="GPU unavailable"):
        validate_colab_environment()
