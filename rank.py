"""Stage 3 reproducible CLI for generating the submission CSV."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from candidate_ranker.config import load_settings
from candidate_ranker.export import rankings_to_csv
from candidate_ranker.ingestion import read_schema
from candidate_ranker.models import JDIntelligence
from candidate_ranker.services import run_pipeline_from_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank candidates with the same pipeline used by the Streamlit app.",
    )
    parser.add_argument(
        "--candidates",
        required=True,
        help="Path to candidates .jsonl or .jsonl.gz",
    )
    parser.add_argument("--jd-cache", required=True, help="Path to prepared JD intelligence JSON")
    parser.add_argument("--schema", required=True, help="Path to candidate JSON schema")
    parser.add_argument("--out", required=True, help="Path for submission CSV output")
    return parser.parse_args()


def _require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    if not path.is_file():
        raise ValueError(f"{label} must be a file: {path}")


def _validate_candidate_path(path: Path) -> None:
    suffix = path.suffix.lower()
    suffixes = "".join(path.suffixes[-2:]).lower()
    if suffix != ".jsonl" and suffixes != ".jsonl.gz":
        raise ValueError("Candidate dataset must be a .jsonl or .jsonl.gz file.")


async def _run() -> None:
    args = parse_args()
    candidate_path = Path(args.candidates)
    jd_cache_path = Path(args.jd_cache)
    schema_path = Path(args.schema)
    out_path = Path(args.out)

    _require_file(jd_cache_path, "JD cache")
    _require_file(schema_path, "Candidate schema")
    _require_file(candidate_path, "Candidate dataset")
    _validate_candidate_path(candidate_path)

    print("Loading JD intelligence...")
    jd_intelligence = JDIntelligence.model_validate_json(jd_cache_path.read_text(encoding="utf-8"))

    print("Loading schema...")
    schema = read_schema(schema_path.read_bytes())

    print("Loading candidates...")
    settings = load_settings()

    print("Running ranking...")
    result = await run_pipeline_from_jsonl(
        candidate_path=str(candidate_path),
        schema=schema,
        settings=settings,
        jd_intelligence=jd_intelligence,
    )

    print(f"Writing {out_path}...")
    csv_data = rankings_to_csv(result.ranked, result.explanations.explanations)
    out_path.write_text(csv_data, encoding="utf-8", newline="")
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
