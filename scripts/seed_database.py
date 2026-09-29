"""Initialize and optionally seed a local development/test database.

No dataset files are shipped with Minded. When --seed is provided, the
synthetic generators create disposable test data at runtime under data/seed.
Production deployments should ingest user-supplied datasets instead.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from apps.api.src.core.database import SessionLocal, init_db
from apps.api.src.models.entities import Dataset, Project
from apps.api.src.services.dataset_service import DatasetService


def seed_db(*, generate: bool = False) -> None:
    init_db()
    if not generate:
        print("Database initialized. No bundled or synthetic datasets were ingested.")
        return

    from scripts.generate_seed_data import generate_all_datasets

    generate_all_datasets()
    db = SessionLocal()
    try:
        project = db.query(Project).filter(Project.id == "proj-default").first()
        if project is None:
            project = Project(
                id="proj-default",
                name="Development Workspace",
                description="Empty development workspace for runtime-generated test data.",
                org_id="default-org",
                owner_id="default-user",
            )
            db.add(project)
            db.commit()
            db.refresh(project)

        seed_dir = os.path.join(os.path.dirname(__file__), "..", "data", "seed")
        dataset_service = DatasetService(db)
        for fname in ("sales.csv", "products.csv", "customers.csv", "marketing.csv", "inventory.csv"):
            fpath = os.path.join(seed_dir, fname)
            if not os.path.isfile(fpath):
                continue
            name = os.path.splitext(fname)[0]
            existing = db.query(Dataset).filter(
                Dataset.project_id == project.id,
                Dataset.name == name,
            ).first()
            if existing is not None:
                continue
            with open(fpath, "rb") as fh:
                dataset_service.ingest_dataset_file(
                    project_id=project.id,
                    filename=fname,
                    file_bytes=fh.read(),
                    description="Runtime-generated development/test dataset; not shipped.",
                )
    finally:
        db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", action="store_true", help="Generate disposable synthetic test data and seed it")
    args = parser.parse_args()
    seed_db(generate=args.seed)
