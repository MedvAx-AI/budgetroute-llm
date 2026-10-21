"""Validated JSONL benchmark datasets and future adapter protocol."""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal, Protocol

import yaml
from pydantic import BaseModel, Field, ValidationError

from budgetroute.exceptions import BudgetRouteError
from budgetroute.experiments.artifacts import ArtifactWriter
from budgetroute.schemas import BenchmarkRecord, EvaluationType


class DatasetAdapter(Protocol):
    def load(self) -> list[BenchmarkRecord]: ...


class JsonlDatasetAdapter:
    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> list[BenchmarkRecord]:
        return load_dataset(self.path)


class DatasetSpec(BaseModel):
    """Pinned source and deterministic conversion contract for a public dataset."""

    schema_version: int = 1
    name: str = Field(min_length=1)
    adapter: Literal["gsm8k", "mmlu", "hotpot_qa"]
    source: str = Field(min_length=1)
    config_name: str | None = None
    revision: str = Field(min_length=7)
    split: str = Field(min_length=1)
    license: str = Field(min_length=1)
    homepage: str
    prompt_version: str = "v1"


class DatasetManifest(BaseModel):
    schema_version: int = 1
    name: str
    adapter: str
    source: str
    config_name: str | None
    source_revision: str
    source_split: str
    license: str
    homepage: str
    prompt_version: str
    record_count: int = Field(ge=1)
    records_sha256: str
    corpus_sha256: str | None = None
    selection: dict[str, int | str | None]
    datasets_version: str


def load_dataset_spec(path: str | Path) -> DatasetSpec:
    source = Path(path)
    if not source.is_file():
        raise BudgetRouteError(f"dataset specification does not exist: {source}")
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
        return DatasetSpec.model_validate(raw)
    except (OSError, yaml.YAMLError, ValidationError) as exc:
        raise BudgetRouteError(f"invalid dataset specification {source}: {exc}") from exc


def _stable_group(value: str) -> str:
    return hashlib.sha256(value.strip().encode("utf-8")).hexdigest()[:20]


def normalized_question_hash(question: str) -> str:
    normalized = " ".join(unicodedata.normalize("NFKC", question).casefold().split())
    return hashlib.sha256(normalized.encode()).hexdigest()


def _gsm8k_answer(value: object) -> str:
    text = str(value)
    match = re.search(r"####\s*([^\n]+)\s*$", text)
    return match.group(1).replace(",", "").strip() if match else text.strip()


def _hotpot_context(value: object) -> str:
    if isinstance(value, dict):
        titles = list(value.get("title", []))
        sentence_groups = list(value.get("sentences", []))
        return "\n\n".join(
            f"{title}\n{' '.join(str(sentence) for sentence in sentences)}"
            for title, sentences in zip(titles, sentence_groups, strict=False)
        )
    if isinstance(value, list):
        paragraphs: list[str] = []
        for item in value:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                title, sentences = item
                body = " ".join(str(sentence) for sentence in sentences)
                paragraphs.append(f"{title}\n{body}")
        return "\n\n".join(paragraphs)
    return str(value)


def convert_dataset_row(
    spec: DatasetSpec, row: dict[str, Any], source_index: int
) -> tuple[BenchmarkRecord, str | None]:
    """Convert one upstream row and optionally return its retrieval corpus document."""

    prefix = f"{spec.name}-{spec.split}-{source_index:07d}"
    source_metadata = {
        "source_index": source_index,
        "source_revision": spec.revision,
        "dataset_license": spec.license,
        "prompt_version": spec.prompt_version,
    }
    if spec.adapter == "gsm8k":
        question = str(row["question"]).strip()
        record = BenchmarkRecord(
            id=prefix,
            group_id=_stable_group(question),
            source=spec.source,
            source_split=spec.split,
            category="numeric_reasoning",
            prompt=(
                f"Solve this problem carefully. Return the final numeric answer after "
                f"the words 'Final answer:'.\n\n{question}"
            ),
            reference_answer=_gsm8k_answer(row["answer"]),
            evaluation_type=EvaluationType.NUMERIC,
            metadata=source_metadata,
        )
        return record, None
    if spec.adapter == "mmlu":
        question = str(row["question"]).strip()
        choices = [str(item) for item in row["choices"]]
        answer_index = int(row["answer"])
        if answer_index < 0 or answer_index >= len(choices):
            raise BudgetRouteError(
                f"MMLU answer index is out of range at source row {source_index}"
            )
        labels = [chr(ord("A") + index) for index in range(len(choices))]
        options = "\n".join(
            f"{label}. {choice}" for label, choice in zip(labels, choices, strict=True)
        )
        subject = str(row.get("subject", "unknown"))
        version_two = spec.prompt_version == "direct-mcq-v2"
        content_sha = hashlib.sha256(
            json.dumps([question, choices], ensure_ascii=False).encode()
        ).hexdigest()
        record = BenchmarkRecord(
            id=prefix,
            group_id=normalized_question_hash(question) if version_two else _stable_group(question),
            source=spec.source,
            source_split=spec.split,
            category=f"mmlu:{subject}",
            prompt=f"Choose the correct option. Answer with only its letter.\n\n{question}\n{options}",
            reference_answer=labels[answer_index],
            evaluation_type=EvaluationType.CLASSIFICATION,
            metadata={
                **source_metadata,
                "subject": subject,
                "choice_count": len(choices),
                **({"content_sha256": content_sha} if version_two else {}),
            },
        )
        return record, None
    question = str(row["question"]).strip()
    corpus = _hotpot_context(row["context"])
    record = BenchmarkRecord(
        id=prefix,
        group_id=_stable_group(question),
        source=spec.source,
        source_split=spec.split,
        category="multi_hop_retrieval",
        prompt=(
            "Answer the question using the configured retrieval corpus. Give a concise answer."
            f"\n\n{question}"
        ),
        reference_answer=str(row["answer"]).strip(),
        evaluation_type=EvaluationType.TOKEN_F1,
        requires_retrieval=True,
        metadata=source_metadata,
    )
    return record, corpus


def _directory_hash(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        digest.update(item.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(item.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def select_source_indices(
    upstream: Any,
    spec: DatasetSpec,
    *,
    limit: int | None,
    offset: int,
    sampling: Literal["head", "stratified"],
    seed: int,
) -> list[int]:
    """Select deterministic upstream rows without relying on process RNG state."""

    if sampling == "head":
        stop = len(upstream) if limit is None else min(len(upstream), offset + limit)
        return list(range(offset, stop))
    if spec.adapter != "mmlu":
        raise BudgetRouteError("stratified materialization currently supports only MMLU")
    if offset != 0:
        raise BudgetRouteError("stratified materialization requires offset 0")
    if limit is None:
        raise BudgetRouteError("stratified materialization requires an explicit limit")

    by_subject: dict[str, list[int]] = defaultdict(list)
    for source_index in range(len(upstream)):
        row = dict(upstream[source_index])
        by_subject[str(row.get("subject", "unknown"))].append(source_index)

    def rank(namespace: str, value: object) -> str:
        material = f"{seed}\0{namespace}\0{value}".encode()
        return hashlib.sha256(material).hexdigest()

    subjects = sorted(by_subject, key=lambda subject: rank("subject", subject))
    ranked = {
        subject: sorted(
            indices,
            key=lambda index: rank(subject, index),
        )
        for subject, indices in by_subject.items()
    }
    selected: list[int] = []
    round_index = 0
    target = min(limit, len(upstream))
    while len(selected) < target:
        added = False
        for subject in subjects:
            if round_index < len(ranked[subject]):
                selected.append(ranked[subject][round_index])
                added = True
                if len(selected) == target:
                    break
        if not added:
            break
        round_index += 1
    return selected


def materialize_dataset(
    spec_path: str | Path,
    output_path: str | Path,
    *,
    limit: int | None = None,
    offset: int = 0,
    sampling: Literal["head", "stratified"] = "head",
    seed: int = 42,
    corpus_dir: str | Path | None = None,
) -> DatasetManifest:
    """Download a pinned source revision and write deterministic benchmark artifacts."""

    if limit is not None and limit < 1:
        raise BudgetRouteError("dataset materialization limit must be at least one")
    if offset < 0:
        raise BudgetRouteError("dataset materialization offset cannot be negative")
    if importlib.util.find_spec("datasets") is None:
        raise BudgetRouteError(
            "remote dataset materialization requires `pip install -e .[datasets]`"
        )
    import datasets

    spec = load_dataset_spec(spec_path)
    try:
        upstream = datasets.load_dataset(
            spec.source,
            spec.config_name,
            split=spec.split,
            revision=spec.revision,
        )
    except Exception as exc:
        raise BudgetRouteError(
            f"could not load pinned dataset {spec.source}@{spec.revision}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    source_indices = select_source_indices(
        upstream,
        spec,
        limit=limit,
        offset=offset,
        sampling=sampling,
        seed=seed,
    )
    records: list[dict[str, Any]] = []
    corpus_target = Path(corpus_dir) if corpus_dir is not None else None
    if spec.adapter == "hotpot_qa" and corpus_target is None:
        corpus_target = (
            Path(output_path).with_suffix("").with_name(Path(output_path).stem + "-corpus")
        )
    if corpus_target is not None:
        if corpus_target.exists() and (not corpus_target.is_dir() or any(corpus_target.iterdir())):
            raise BudgetRouteError(
                f"corpus directory must be empty to avoid mixed manifests: {corpus_target}"
            )
        corpus_target.mkdir(parents=True, exist_ok=True)
    for source_index in source_indices:
        raw = dict(upstream[source_index])
        record, corpus = convert_dataset_row(spec, raw, source_index)
        records.append(record.model_dump(mode="json"))
        if corpus is not None and corpus_target is not None:
            (corpus_target / f"{record.id}.txt").write_text(corpus + "\n", encoding="utf-8")
    if not records:
        raise BudgetRouteError("dataset selection produced no records")
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    writer = ArtifactWriter(destination.parent)
    writer.write_jsonl(destination.name, records)
    manifest = DatasetManifest(
        name=spec.name,
        adapter=spec.adapter,
        source=spec.source,
        config_name=spec.config_name,
        source_revision=spec.revision,
        source_split=spec.split,
        license=spec.license,
        homepage=spec.homepage,
        prompt_version=spec.prompt_version,
        record_count=len(records),
        records_sha256=dataset_hash(destination),
        corpus_sha256=_directory_hash(corpus_target) if corpus_target is not None else None,
        selection={
            "method": sampling,
            "offset": offset,
            "limit": limit,
            "seed": seed if sampling == "stratified" else None,
            "stratify_by": "subject" if sampling == "stratified" else None,
        },
        datasets_version=importlib.metadata.version("datasets"),
    )
    writer.write_json(
        destination.with_suffix(".manifest.json").name, manifest.model_dump(mode="json")
    )
    return manifest


def materialize_router_test_split(
    router_metadata_path: str | Path,
    dataset_path: str | Path,
    source_manifest_path: str | Path,
    output_path: str | Path,
) -> DatasetManifest:
    """Write the persisted untouched router test IDs as a hashed benchmark dataset."""

    source = Path(dataset_path)
    destination = Path(output_path)
    if source.resolve() == destination.resolve():
        raise BudgetRouteError("router test split output must differ from its source dataset")
    parent_manifest = validate_dataset_manifest(source, source_manifest_path)
    try:
        metadata = json.loads(Path(router_metadata_path).read_text(encoding="utf-8"))
        test_ids = metadata["splits"]["test"]["ids"]
        seed = int(metadata["seed"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise BudgetRouteError(f"invalid router metadata {router_metadata_path}: {exc}") from exc
    if (
        not isinstance(test_ids, list)
        or not test_ids
        or not all(isinstance(item, str) and item for item in test_ids)
    ):
        raise BudgetRouteError("router metadata test IDs must be a non-empty string list")
    if len(set(test_ids)) != len(test_ids):
        raise BudgetRouteError("router metadata contains duplicate test IDs")

    records = {record.id: record for record in load_dataset(source)}
    missing = [record_id for record_id in test_ids if record_id not in records]
    if missing:
        raise BudgetRouteError(
            f"router test split references {len(missing)} IDs absent from the source dataset"
        )
    selected = [records[record_id].model_dump(mode="json") for record_id in test_ids]
    destination.parent.mkdir(parents=True, exist_ok=True)
    writer = ArtifactWriter(destination.parent)
    writer.write_jsonl(destination.name, selected)
    manifest = parent_manifest.model_copy(
        update={
            "name": f"{parent_manifest.name}-router-test",
            "record_count": len(selected),
            "records_sha256": dataset_hash(destination),
            "selection": {
                "method": "router_test_split",
                "offset": 0,
                "limit": len(selected),
                "seed": seed,
                "stratify_by": "group",
                "parent_records_sha256": parent_manifest.records_sha256,
            },
        }
    )
    writer.write_json(
        destination.with_suffix(".manifest.json").name,
        manifest.model_dump(mode="json"),
    )
    return manifest


def validate_dataset_manifest(
    dataset_path: str | Path, manifest_path: str | Path
) -> DatasetManifest:
    try:
        manifest = DatasetManifest.model_validate_json(Path(manifest_path).read_text("utf-8"))
    except (OSError, ValidationError) as exc:
        raise BudgetRouteError(f"invalid dataset manifest {manifest_path}: {exc}") from exc
    records = load_dataset(dataset_path)
    if len(records) != manifest.record_count:
        raise BudgetRouteError(
            f"dataset record count {len(records)} does not match manifest {manifest.record_count}"
        )
    actual_hash = dataset_hash(dataset_path)
    if actual_hash != manifest.records_sha256:
        raise BudgetRouteError(
            f"dataset hash {actual_hash} does not match manifest {manifest.records_sha256}"
        )
    return manifest


def load_dataset(path: str | Path) -> list[BenchmarkRecord]:
    source = Path(path)
    if not source.is_file():
        raise BudgetRouteError(f"benchmark dataset does not exist: {source}")
    records: list[BenchmarkRecord] = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = BenchmarkRecord.model_validate_json(line)
        except ValidationError as exc:
            raise BudgetRouteError(
                f"invalid benchmark record {source}:{line_number}: {exc}"
            ) from exc
        if record.id in seen_ids:
            raise BudgetRouteError(f"duplicate benchmark ID {record.id!r} at line {line_number}")
        seen_ids.add(record.id)
        records.append(record)
    if not records:
        raise BudgetRouteError(f"benchmark dataset is empty: {source}")
    return records


def dataset_hash(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dataset_summary(records: list[BenchmarkRecord]) -> dict[str, object]:
    categories: dict[str, int] = {}
    evaluation_types: dict[str, int] = {}
    for record in records:
        categories[record.category] = categories.get(record.category, 0) + 1
        value = record.evaluation_type.value
        evaluation_types[value] = evaluation_types.get(value, 0) + 1
    return {
        "records": len(records),
        "categories": categories,
        "evaluation_types": evaluation_types,
        "requires_retrieval": sum(record.requires_retrieval for record in records),
        "must_abstain": sum(record.must_abstain for record in records),
    }
