# AI Candidate Ranking Platform

> Scan 100,000+ candidate records against a job description in seconds — powered by OpenRouter LLMs, semantic embeddings, and local feature scoring — and surface the top 100 best-fit candidates for any role.

🔗 **Live App:** [ai-candidate-ranker-hackathon.streamlit.app](https://ai-candidate-ranker-hackathon.streamlit.app/)

---

## What It Does

Recruiters deal with hundreds of thousands of resumes. This platform automates the hardest part of the screening process: turning an unstructured job description and a massive candidate dataset into a ranked shortlist of the most qualified people, complete with scores, breakdowns, and exportable results.

The platform handles the entire pipeline — from parsing a job description to returning a scored, ranked CSV of the top 100 candidates — with no manual filtering required.

---

## How It Works

### Step 1 — Upload Your Inputs

The recruiter uploads three things:

- **Job Description** — accepted as PDF, DOCX, TXT, or Markdown
- **Candidate Dataset** — a `.jsonl` file containing up to 100,000+ candidate records
- **Candidate Schema** — a `.json` file describing the structure/fields of the candidate data

### Step 2 — AI-Powered Job Description Analysis

The job description is sent to an OpenRouter LLM (temperature `0` for deterministic output), which extracts structured intelligence about the role — required skills, experience levels, responsibilities, seniority signals, and more — and returns it as strict JSON. This structured JD profile drives all downstream ranking logic.

### Step 3 — Dynamic Schema Mapping

The candidate schema is analyzed and mapped through a `SchemaMap`, so the ranker can resolve candidate fields dynamically regardless of how the JSON records are structured. There is no hardcoded expectation of fields like `candidate.skills` — the platform adapts to whatever schema is provided.

### Step 4 — Bulk Candidate Scoring

Every candidate record is streamed from disk and scored locally using a multi-signal feature engineering pipeline. Scoring considers:

- **Skill overlap** between the candidate and the extracted JD requirements
- **Relevant experience** — years, seniority alignment, domain match
- **Integrity checks** — honesty/fraud detection signals (e.g., inflated titles, inconsistent timelines)
- **Semantic retrieval** — FAISS-backed embedding similarity with a NumPy deterministic fallback for constrained environments
- **Composite scoring** — all signals are combined into a single score, then normalized to a 0–1 range using min-max scaling

The pipeline uses a top-K heap to keep only the best 500 candidates in memory while processing all records, ensuring it stays fast and memory-efficient even at scale.

### Step 5 — Results, Details & Export

The top 100 candidates are surfaced to the recruiter across several views:

- **Results tab** — ranked table of top 100 candidates with scores
- **Candidate Detail tab** — per-candidate breakdown of every scoring dimension, plus explanation
- **Analytics tab** — score distribution histogram and feature importance bar chart for the top 20
- **Export tab** — one-click download of a `.csv` containing candidate ID, rank, score, and all relevant fields

---

## Key Features

**Handles 100K+ Records**
The JSONL streaming pipeline never loads the full dataset into memory. Candidates are scored on the fly and only the top-K survive to the next stage.

**Dynamic Schema Adapts to Any Dataset**
The `SchemaMap` layer means this platform works with any candidate JSON structure — no field name assumptions, no re-coding per dataset.

**LLM Used Surgically**
The AI layer is intentionally narrow: it extracts structured JD intelligence once. All ranking, retrieval, and scoring runs locally and deterministically.

**Integrity & Fraud Detection**
Built-in checks flag candidates with signals of inflated credentials, inconsistent experience claims, or other integrity concerns, which factor into their composite score.

**Normalized, Comparable Scores**
Raw scores are scaled with a min-max scaler to a 0–1 range, making it easy to compare candidates across very different datasets.

**Semantic Retrieval with Graceful Fallback**
Candidate retrieval uses FAISS-based dense embeddings for semantic similarity. If the embedding model is unavailable (cold start or constrained environment), the system falls back to deterministic hash-based embeddings so the app always runs.

**Analytics Dashboard**
Score histograms and feature-breakdown charts give recruiters instant signal on the distribution of the candidate pool and which factors drove rankings.

---

## App Tabs

| Tab | What It Shows |
|---|---|
| Upload | Job description, candidate dataset, and schema upload |
| JD Analysis | Structured JSON extracted from the job description by the LLM |
| Schema Analysis | Resolved `SchemaMap` from the candidate schema file |
| Ranking Progress | Live pipeline status and funnel summary |
| Results | Ranked top-100 table with composite scores |
| Candidate Detail | Per-candidate scoring breakdown and AI explanation |
| Analytics | Score distribution and feature importance charts |
| Export | Download CSV of top-100 candidates |

---

## Architecture

```
app.py
  └── candidate_ranker/services.py       # Orchestrates the full pipeline
        ├── ingestion.py                 # Reads JD, schema, and JSONL candidates
        ├── schema_mapping.py            # Builds SchemaMap from candidate schema
        ├── ai_service.py                # LLM calls: JD extraction + explanations
        │     └── ai/openrouter_client.py  # Only module allowed to call OpenRouter
        ├── retrieval.py                 # FAISS embeddings + NumPy fallback
        ├── ranking.py                   # Feature engineering, scoring, integrity checks
        ├── export.py                    # CSV generation, scoring reasoning
        └── models.py                    # Pydantic models (JDIntelligence, CandidateScore, SchemaMap)
```

**Design principles:**

- `ai/openrouter_client.py` is the single point of contact with the LLM API — no other module calls OpenRouter
- All ranking, retrieval, scoring, fraud detection, and exports run **locally** — no LLM in the hot path
- Pydantic models enforce strict data contracts throughout the pipeline
- FAISS retrieval degrades gracefully to a deterministic NumPy fallback with zero user impact

---

## Running Locally

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/candidate-scanner-hackathon.git
cd candidate-scanner-hackathon

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate       # macOS/Linux
.venv\Scripts\activate          # Windows
source .venv/Scripts/activate   #GitBash Terminal

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
cp .env.example .env
# Open .env and set your OPENROUTER_API_KEY

# 5. Launch the app
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## Stage 3 Code Reproduction

Run the same ranking pipeline used by the Streamlit app without opening the UI:

```bash
python rank.py \
  --candidates candidates.jsonl.gz \
  --job-description job_description.md \
  --schema candidate_schema.json \
  --out submission.csv
```

The candidate file may be `.jsonl` or `.jsonl.gz`. The generated `submission.csv` contains exactly:

```text
candidate_id,rank,score,reasoning
```

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `OPENROUTER_API_KEY` | Yes | Your OpenRouter API key for LLM calls |
| `DEPLOYMENT_TARGET` | No | Set to `streamlit` when running on Streamlit Cloud |
| `CANDIDATE_UPLOAD_PORT` | No | Port for the local chunked upload server (default: `8765`) |
| `CANDIDATE_UPLOAD_HOST` | No | Host for the upload server (default: `127.0.0.1`) |
| `CANDIDATE_UPLOAD_PUBLIC_URL` | No | Public URL for the upload server |

---

## Tests

```bash
pytest
```

Test coverage spans ingestion, schema mapping, ranking, retrieval, export, and the OpenRouter client.

---

## Known Limitations

- PDF and DOCX parsing uses pragmatic text extraction optimized for ranking workflows, not full visual/layout reconstruction.
- The embedding model downloads on first use. If unavailable, retrieval automatically falls back to deterministic hash embeddings — the app remains fully functional.
- Free-tier OpenRouter models may encounter rate limits; automatic retry with model failover is implemented.
- Streamlit Cloud enforces a file upload size limit; for very large JSONL files (500 MB+), running locally is recommended.
- The version deployed on Streamlit Cloud may take more than 8–10 minutes to complete the full ranking process. This is due to the resource limitations of the free-tier deployment, which provides restricted CPU and memory capacity.

---

## Tech Stack

- **Frontend & App Framework** — Streamlit
- **LLM API** — OpenRouter (model-agnostic; configurable)
- **Semantic Search** — FAISS with sentence-transformers
- **Data Validation** — Pydantic v2
- **Data & Visualization** — Pandas, Plotly, Altair
- **Testing** — pytest

---

## License

MIT
