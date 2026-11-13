"""Independent, development-only MMLU pilot selection; no confirmation access."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, model_validator

from budgetroute.evaluation.dataset import (
    DatasetSpec,
    convert_dataset_row,
    dataset_hash,
    load_dataset,
    load_dataset_spec,
    normalized_question_hash,
    select_source_indices,
)
from budgetroute.exceptions import BudgetRouteError
from budgetroute.experiments.artifacts import ArtifactWriter

HISTORICAL_SHA256 = "5b25d20f38e91120bd3303ad32cc35d54ae9eb0326468532f41dcfaa2b31ede7"


class PilotSplits(BaseModel):
    schema_version: Literal[1] = 1
    role: Literal["pilot"] = "pilot"
    seed: int
    dataset_sha256: str
    train_ids: list[str]
    calibration_ids: list[str]
    test_ids: list[str]
    exclusion_sha256: str

    @model_validator(mode="after")
    def independent(self) -> PilotSplits:
        ids = self.train_ids + self.calibration_ids + self.test_ids
        if (len(self.train_ids), len(self.calibration_ids), len(self.test_ids)) != (
            120,
            40,
            40,
        ) or len(set(ids)) != 200:
            raise ValueError("pilot partitions require 120/40/40 distinct IDs")
        return self


def _load_upstream(spec: DatasetSpec) -> Any:
    import datasets

    return datasets.load_dataset(
        spec.source, spec.config_name, revision=spec.revision, split=spec.split
    )


def build_historical_exclusions(original: Path, spec_path: Path, destination: Path) -> Path:
    if dataset_hash(original) != HISTORICAL_SHA256:
        raise BudgetRouteError(
            "historical exclusions require the verified original 500-record hash"
        )
    spec = load_dataset_spec(spec_path)
    rows = _load_upstream(spec)
    originals = load_dataset(original)
    if len(originals) != 500:
        raise BudgetRouteError("historical exclusion count is not 500")
    entries = []
    for record in originals:
        index = int(record.metadata["source_index"])
        raw = dict(rows[index])
        restored, _ = convert_dataset_row(spec, raw, index)
        if restored.prompt != record.prompt or restored.reference_answer != record.reference_answer:
            raise BudgetRouteError("historical source rehydration mismatch")
        entries.append(
            {
                "id": record.id,
                "source_index": index,
                "group_sha256": normalized_question_hash(str(raw["question"])),
            }
        )
    return ArtifactWriter(destination.parent).write_json(
        destination.name,
        {
            "schema_version": 1,
            "source": spec.source,
            "revision": spec.revision,
            "split": spec.split,
            "parent_sha256": HISTORICAL_SHA256,
            "entries": entries,
            "entries_sha256": hashlib.sha256(
                json.dumps(entries, sort_keys=True).encode()
            ).hexdigest(),
        },
    )


def prepare_pilot_data(
    spec_path: Path, exclusion_path: Path, output_dir: Path, *, seed: int = 20261021
) -> PilotSplits:
    spec = load_dataset_spec(spec_path)
    if (
        spec.adapter != "mmlu"
        or spec.prompt_version != "direct-mcq-v2"
        or not re.fullmatch("[a-f0-9]{40}", spec.revision)
    ):
        raise BudgetRouteError("pilot requires pinned MMLU direct-mcq-v2")
    try:
        exclusion = json.loads(exclusion_path.read_text(encoding="utf-8"))
        entries = exclusion["entries"]
        digest = hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest()
        if (
            not entries
            or digest != exclusion["entries_sha256"]
            or exclusion["parent_sha256"] != HISTORICAL_SHA256
            or (exclusion["source"], exclusion["revision"], exclusion["split"])
            != (spec.source, spec.revision, spec.split)
            or any(
                not re.fullmatch("[a-f0-9]{64}", e["group_sha256"])
                or type(e["source_index"]) is not int
                for e in entries
            )
        ):
            raise ValueError("exclusion identity/hash mismatch")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise BudgetRouteError(f"invalid exclusion manifest: {exc}") from exc
    excluded_indices = {e["source_index"] for e in entries}
    excluded_groups = {e["group_sha256"] for e in entries}
    upstream = _load_upstream(spec)
    ranked = select_source_indices(
        upstream, spec, limit=len(upstream), offset=0, sampling="stratified", seed=seed
    )
    chosen = []
    seen = set(excluded_groups)
    for index in ranked:
        raw = dict(upstream[index])
        group = normalized_question_hash(str(raw["question"]))
        if index in excluded_indices or group in seen:
            continue
        record, _ = convert_dataset_row(spec, raw, index)
        chosen.append(record)
        seen.add(group)
        if len(chosen) == 200:
            break
    if len(chosen) != 200:
        raise BudgetRouteError("insufficient independent non-excluded pilot questions")
    # Interleave strata before partitioning; assign roles by a separate seeded hash.
    chosen.sort(key=lambda row: hashlib.sha256(f"{seed}:split:{row.id}".encode()).hexdigest())
    writer = ArtifactWriter(output_dir)
    data_path = output_dir / "pilot.jsonl"
    rows = [row.model_dump(mode="json") for row in chosen]
    candidate = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    if data_path.exists() and data_path.read_text(encoding="utf-8") != candidate:
        raise BudgetRouteError("existing pilot data differs; use a new workspace")
    writer.write_jsonl(data_path.name, rows)
    splits = PilotSplits(
        seed=seed,
        dataset_sha256=dataset_hash(data_path),
        exclusion_sha256=dataset_hash(exclusion_path),
        train_ids=[r.id for r in chosen[:120]],
        calibration_ids=[r.id for r in chosen[120:160]],
        test_ids=[r.id for r in chosen[160:]],
    )
    writer.write_json("splits.json", splits.model_dump(mode="json"))
    writer.write_json(
        "pilot.manifest.json",
        {
            "role": "pilot",
            "spec": spec.model_dump(mode="json"),
            "record_count": 200,
            "records_sha256": splits.dataset_sha256,
            "exclusion_sha256": splits.exclusion_sha256,
            "selection": "subject_stratified_then_hashed_role_assignment",
            "seed": seed,
        },
    )
    return splits
