"""Ingest files and run the canonical InvestigationController unattended."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from apps.api.src.core.database import SessionLocal, init_db
from apps.api.src.models.entities import Project
from apps.api.src.services.analysis_service import AnalysisService
from apps.api.src.services.dataset_service import DatasetService
from packages.schemas.src.analysis import AnalysisCreate
from packages.schemas.src.dataset import DatasetProfileSchema

from minded.questions import synthesize_questions
from minded.report import write_reports

DEFAULT_PROJECT_ID = "proj-minded"
SUPPORTED_SUFFIXES = {".csv", ".tsv", ".xlsx", ".xls", ".parquet", ".json"}


def ensure_project(db, project_id: str = DEFAULT_PROJECT_ID) -> Project:
    proj = db.query(Project).filter(Project.id == project_id).first()
    if proj:
        return proj
    proj = Project(
        id=project_id,
        name="Minded Autonomous Analyzer",
        description="Local workspace for unattended dataset investigations.",
        org_id="default-org",
        owner_id="default-user",
    )
    db.add(proj)
    db.commit()
    db.refresh(proj)
    return proj


def ingest_paths(
    db,
    paths: Sequence[Path],
    *,
    project_id: str,
) -> List[Any]:
    service = DatasetService(db)
    datasets = []
    for path in paths:
        data = path.read_bytes()
        datasets.append(
            service.ingest_dataset_file(
                project_id=project_id,
                filename=path.name,
                file_bytes=data,
                description=f"Autonomous ingest from {path}",
            )
        )
    return datasets


def choose_question(datasets: Sequence[Any], explicit: Optional[str]) -> str:
    if explicit and explicit.strip():
        return explicit.strip()
    questions: List[str] = []
    for ds in datasets:
        profile = ds.profile_json
        if profile:
            questions.extend(
                synthesize_questions(profile, dataset_label=ds.name, max_questions=1)
            )
    if questions:
        return questions[0]
    return "What are the most important patterns, anomalies, and drivers in the ingested data?"


def profile_summaries(datasets: Sequence[Any]) -> List[str]:
    out = []
    for ds in datasets:
        raw = ds.profile_json
        if not raw:
            out.append(f"{ds.name}: {ds.row_count} rows, {ds.column_count} columns")
            continue
        profile = DatasetProfileSchema.model_validate(raw)
        score = profile.data_quality.overall_score if profile.data_quality else None
        quality = f", quality {score:.0f}/100" if score is not None else ""
        summary = profile.summary_text or f"{profile.row_count} rows, {profile.column_count} columns"
        out.append(f"{ds.name}: {summary}{quality}")
    return out


def run_analysis(
    paths: Sequence[Path],
    *,
    question: Optional[str] = None,
    project_id: str = DEFAULT_PROJECT_ID,
    out_dir: Path = Path("reports"),
    analysis_mode: str = "DETERMINISTIC",
) -> Dict[str, Any]:
    resolved = [Path(p).expanduser().resolve() for p in paths]
    missing = [p for p in resolved if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing input files: {missing}")

    init_db()
    db = SessionLocal()
    try:
        ensure_project(db, project_id)
        datasets = ingest_paths(db, resolved, project_id=project_id)
        chosen_question = choose_question(datasets, question)
        summaries = profile_summaries(datasets)
        payload = AnalysisCreate(
            question=chosen_question,
            project_id=project_id,
            dataset_ids=[d.id for d in datasets],
            analysis_mode=analysis_mode,
        )
        result = AnalysisService(db).execute_analysis(payload)
        digest = hashlib.sha256("|".join(str(p) for p in resolved).encode("utf-8")).hexdigest()[:10]
        stem = f"{datasets[0].name}_{digest}" if datasets else f"analysis_{digest}"
        written = write_reports(
            Path(out_dir),
            stem,
            source_files=[str(p) for p in resolved],
            question=chosen_question,
            profile_summaries=summaries,
            result=result,
            extra={
                "dataset_ids": [d.id for d in datasets],
                "project_id": project_id,
            },
        )
        return {
            "question": chosen_question,
            "investigation_id": getattr(result, "id", None),
            "status": str(getattr(result, "status", "")),
            "verdict": str(getattr(result, "verdict", "")),
            "direct_answer": getattr(result, "direct_answer", None),
            "reports": {k: str(v) for k, v in written.items()},
            "datasets": [{"id": d.id, "name": d.name, "rows": d.row_count} for d in datasets],
        }
    finally:
        db.close()
