"""Online pre-computation CLI for Stage 3 JD intelligence caching."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from candidate_ranker.config import load_settings
from candidate_ranker.ingestion import read_job_description, read_schema
from candidate_ranker.services import prepare_jd_intelligence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare JD intelligence for offline candidate ranking.",
    )
    parser.add_argument("--job-description", required=True, help="Path to job description file")
    parser.add_argument("--schema", required=True, help="Path to candidate JSON schema")
    parser.add_argument("--out", required=True, help="Path for prepared JD cache JSON")
    parser.add_argument(
        "--model",
        default=None,
        help="Optional OpenRouter model override. Defaults to OPENROUTER_PRIMARY_MODEL.",
    )
    return parser.parse_args()


def _require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    if not path.is_file():
        raise ValueError(f"{label} must be a file: {path}")


async def _run() -> None:
    args = parse_args()
    jd_path = Path(args.job_description)
    schema_path = Path(args.schema)
    out_path = Path(args.out)

    _require_file(jd_path, "Job description")
    _require_file(schema_path, "Candidate schema")

    print("Loading job description...")
    jd_text = read_job_description(jd_path.name, jd_path.read_bytes())
    if not jd_text.strip():
        raise ValueError(f"Job description is empty or could not be read: {jd_path}")

    print("Loading schema...")
    schema = read_schema(schema_path.read_bytes())

    print("Preparing JD intelligence...")
    settings = load_settings()
    jd_intelligence = await prepare_jd_intelligence(
        jd_text=jd_text,
        schema=schema,
        model=args.model or settings.openrouter_primary_model,
    )

    print(f"Writing {out_path}...")
    out_path.write_text(jd_intelligence.model_dump_json(indent=2), encoding="utf-8")
    print("Done.")


def main() -> int:
    logging.basicConfig(level=logging.WARNING)
    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
