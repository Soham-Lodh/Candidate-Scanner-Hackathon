# AI Candidate Ranking Platform

> **Team Code Paradox** — *Data & AI Challenge: Intelligent Candidate Discovery*  
> Built by Soham Lodh and Dhrupad Paitandy to answer one question most systems ignore:  
> _Not "does this candidate match the skills?" — but "what is the probability this candidate gets hired?"_

[![Live App](https://img.shields.io/badge/Live%20App-Streamlit-FF4B4B?logo=streamlit)](https://ai-candidate-ranker-hackathon.streamlit.app/)
[![GitHub](https://img.shields.io/badge/Source-GitHub-181717?logo=github)](https://github.com/Soham-Lodh/Candidate-Scanner-Hackathon)
[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker)](https://docker.com)

---

## Table of Contents

1. [What We Built](#what-we-built)
2. [Performance & Constraints](#performance--constraints)
3. [How It Works — End-to-End Pipeline](#how-it-works--end-to-end-pipeline)
4. [Ranking Methodology — The Scoring Engine](#ranking-methodology--the-scoring-engine)
5. [Explainability & Integrity Validation](#explainability--integrity-validation)
6. [Architecture](#architecture)
7. [App Tabs](#app-tabs)
8. [Tech Stack](#tech-stack)
9. [Local Setup — Step-by-Step](#local-setup--step-by-step)
10. [Docker Setup](#docker-setup)
11. [Stage 3 CLI Reproduction (Offline Ranking)](#stage-3-cli-reproduction-offline-ranking)
12. [Running Tests](#running-tests)
13. [Configuration Reference](#configuration-reference)
14. [Known Limitations](#known-limitations)
15. [Challenges We Faced](#challenges-we-faced)
16. [Submission Assets](#submission-assets)

---

## What We Built

Recruiters face an impossible task: hundreds of thousands of resumes, dozens of active roles, and no scalable way to find who actually fits. This platform automates the hardest part of the screening process.

**Given:**
- A job description (PDF, DOCX, TXT, or Markdown)
- A candidate dataset in JSONL format (up to 100,000+ records)
- A JSON schema describing the candidate data structure

**The platform outputs:** A ranked shortlist of the top 100 best-fit candidates, complete with composite scores, per-dimension breakdowns, skill gap analysis, integrity flags, AI-generated recruiter rationale, and a one-click exportable CSV.

The entire pipeline — from unstructured JD to ranked shortlist — requires no manual configuration, no field name assumptions, and no per-candidate LLM calls.

---

## Performance & Constraints

This solution was built and validated against the challenge's hardware and runtime constraints.

| Metric | Value |
|---|---|
| **Runtime (100K candidates)** | **5 minutes 26.162 seconds** |
| **Hardware** | CPU only — no GPU required |
| **Memory** | 16 GB RAM |
| **Dataset size** | 100,000+ candidate records |
| **Max file size accepted** | 500 MB (both locally and on deployed Streamlit) |
| **Deployed platform runtime** | 5–10 minutes (Streamlit Cloud free tier) |

A demo video of the app running the full 100K dataset inside a Docker container with 16 GB RAM and 1 CPU is available here:  
📹 **[Video Demo (Google Drive)](https://drive.google.com/drive/folders/1CSKugCC6FlCjdih-3ZUJ_vJ3DyQI-QRy)**

### How We Meet the Compute Constraints

- **No LLM in the hot path.** The AI layer runs exactly once — to extract structured JD intelligence. All candidate scoring is local and deterministic.
- **Streaming JSONL pipeline.** Candidates are processed one-by-one from disk. The full dataset is never loaded into memory.
- **Top-K heap.** Only the top 500 candidates are kept in memory at any time during scoring.
- **Semantic retrieval narrows the field.** A FAISS-backed embedding funnel reduces 100K candidates to 5,000 before the expensive feature scoring runs.
- **Graceful fallback.** If FAISS or the embedding model is unavailable, the system degrades to a deterministic hash-based embedding — without user impact.

---

## How It Works — End-to-End Pipeline

### Step 1 — Upload Your Inputs

The recruiter uploads three things through the Streamlit UI:

| Input | Accepted Formats | Purpose |
|---|---|---|
| **Job Description** | PDF, DOCX, TXT, Markdown | Describes the open role |
| **Candidate Dataset** | `.jsonl` or `.jsonl.gz` (up to 500 MB) | The full candidate pool |
| **Candidate Schema** | JSON Schema Draft 7 (`.json`) | Defines the structure of candidate records |

### Step 2 — AI-Powered JD Intelligence Extraction

The job description is sent once to an OpenRouter LLM (temperature `0.0` for deterministic output). The model extracts a structured `JDIntelligence` object containing:

- Required and preferred skills
- Seniority level and experience minimums
- Responsibilities and production signals
- Behavioral expectations
- Education keywords and location preferences

This structured profile drives all downstream ranking logic. It is cached to disk (`jd_cache.json`) and reused for offline reproduction without any further LLM calls.

### Step 3 — Dynamic Schema Mapping

The uploaded JSON schema is analyzed and mapped into a `SchemaMap` through semantic keyword matching. The ranker resolves candidate fields (skills, experience, location, seniority, etc.) dynamically, regardless of how the schema is structured.

**There are zero hardcoded field name assumptions.** A schema using `capabilities` or `toolkit` is handled identically to one using `skills`.

### Step 4 — Semantic Retrieval Funnel

A `SemanticRetriever` (FAISS + `sentence-transformers/all-MiniLM-L6-v2`) encodes the job description and all candidate profiles as dense vectors and retrieves the top 5,000 most semantically similar candidates for scoring.

The funnel: **100K → 5,000 (retrieval) → 500 (feature scoring) → 100 (final output)**

### Step 5 — Multi-Signal Local Scoring

Every retrieved candidate is scored against 21 independent signals across four categories:

| Category | Signals |
|---|---|
| **Technical Fit** | Required skill coverage, preferred skill coverage, skill depth, production signals, lexical relevance |
| **Role Relevance** | Role fit, semantic match, relevant experience, seniority alignment, career consistency |
| **Behavioral Signals** | Recruiter response rate, interview completion, GitHub activity, availability, notice period |
| **Trust & Integrity** | Profile completeness, consistency analysis, fraud detection heuristics, evidence confidence |

The final composite score formula:

```
Final Score = f(Technical Fit, Role Relevance, Behavioral Signals, Trust Signals)
              × Must-Have Gate × Integrity Multiplier
```

Scores are normalized to a 0–1 range using min-max scaling across the exported top 100, giving recruiters a clean relative comparison.

### Step 6 — Results, Detail & Export

| Tab | Content |
|---|---|
| Results | Ranked top-100 table with composite scores |
| Candidate Detail | Per-candidate breakdown of all 21 scoring dimensions |
| Analytics | Score distribution histogram, feature importance bar chart |
| Export | One-click CSV download: `candidate_id`, `rank`, `score`, `reasoning` |

---

## Ranking Methodology — The Scoring Engine

### Signal Categories

**Technical Fit (weighted 55% of base score)**
- `required_skill_coverage` — token + substring match of JD-required skills against candidate profile
- `preferred_skill_coverage` — same for preferred skills
- `skill_depth` — proficiency level, endorsement count, months of evidence per skill record
- `production_experience_signals` — keywords and quantitative metrics indicating production delivery
- `lexical_relevance` — density of JD terms across full candidate text

**Role Relevance (weighted 32% of base score)**
- `role_fit` — overlap of JD title and responsibilities against candidate's current role
- `semantic_match` — FAISS cosine similarity score (rescaled)
- `relevant_experience` — years of experience vs. JD minimum, with surplus bonus
- `seniority_match` — title-level signal, seniority term detection, year-adjusted score
- `career_consistency` — average tenure, job progression, title alignment

**Behavioral Signals and Hireability (weighted 13% of base score)**
- `behavioral_score` — recruiter response rate, offer acceptance rate, profile views, interview completion
- `availability` — notice period, open-to-work flag, relocation willingness
- `location_match` — candidate location vs. JD location preferences
- `education_relevance` — degree keywords, institution tier signal, education field match
- `evidence_confidence` — richness of profile data (history entries, skill records, signal fields)

**Must-Have Gate** — a gate score computed from required skill coverage, role fit, experience, and integrity. Candidates who fail required skills are down-weighted even with high behavioral scores.

**Integrity Multiplier** — a penalty applied for profiles with sparse data, promotional language patterns, or inconsistent experience claims.

---

## Explainability & Integrity Validation

Every ranking decision is auditable.

### Per-Candidate Breakdown
Each candidate surfaces:
- A composite score and rank
- An individual score for every one of the 21 dimensions
- Matched skills (with evidence: proficiency, duration, endorsement count)
- Missing required skills
- Integrity flags (sparse data, inflated language, date inconsistencies)
- Recruiter-ready reasoning built from scoring evidence — not generated from raw profile text

### Preventing Hallucinations
- LLM usage is limited to exactly two calls: JD intelligence extraction and (optionally) explanation generation.
- All AI responses are validated through strict Pydantic schemas before use.
- Candidate explanations are generated from structured scoring outputs. The model is instructed never to invent facts.
- All ranking decisions are produced by deterministic local scoring — the LLM has zero influence on rank order.

### Integrity Checks Built Into Scoring
- Sparse profile detection (thin text, low token diversity)
- Promotional language pattern detection ("rockstar", "guru", repeated superlatives)
- Open-ended experience date validation
- Profile completeness signal from platform data
- Inconsistency signals reduce composite score through the Integrity Multiplier

---

## Architecture

```
app.py
  └── candidate_ranker/services.py         # Orchestrates the full pipeline
        ├── ingestion.py                   # Reads JD (PDF/DOCX/TXT/MD), schema, JSONL candidates
        ├── schema_mapping.py              # Builds SchemaMap via semantic keyword ranking
        ├── ai_service.py                  # LLM calls: JD extraction + candidate explanations
        │     └── ai/openrouter_client.py  # Single point of LLM contact; retries + failover
        ├── retrieval.py                   # FAISS semantic retrieval + NumPy fallback
        ├── ranking.py                     # 21-signal feature engine, top-K heap, integrity checks
        ├── export.py                      # CSV generation, min-max normalization, scoring reasoning
        ├── upload_server.py               # Sidecar chunked upload server for large JSONL files
        └── models.py                      # Pydantic models: JDIntelligence, CandidateScore, SchemaMap
```

### Design Principles

- `ai/openrouter_client.py` is the **single point of contact** with any LLM API. No other module calls OpenRouter.
- **All ranking, retrieval, scoring, and fraud detection runs locally** — zero LLM calls in the hot path.
- **Pydantic v2 enforces strict data contracts** at every boundary: JD intelligence, schema maps, candidate scores, and AI explanations all have typed models.
- **FAISS degrades gracefully** to a deterministic SHA-256 hash embedding if the embedding model is unavailable — the app always runs.
- **LLM failover chain:** DeepSeek V3 → Qwen 3 235B → Llama 3.3 70B. If the primary model rate-limits, the client retries with exponential backoff and then promotes to the next model automatically.

---

## App Tabs

| Tab | What It Shows |
|---|---|
| 📤 Upload | Job description, candidate dataset, schema upload; pipeline trigger |
| 📄 JD Analysis | Structured JSON extracted from the job description by the LLM |
| 🗂️ Schema | Resolved `SchemaMap` — dynamic field paths derived from the candidate schema |
| ⚙️ Progress | Live pipeline status, funnel summary, stage-by-stage progress |
| 📊 Results | Ranked top-100 table with composite scores and progress bars |
| 👤 Candidate Detail | Per-candidate 21-dimension breakdown, matched/missing skills, AI explanation |
| 📈 Analytics | Score distribution histogram, feature importance bar chart for top 20 |
| 💾 Export | One-click CSV download of top-100 candidates |

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Frontend & App** | Streamlit 1.35 |
| **LLM API** | OpenRouter (DeepSeek V3, Qwen 3 235B, Llama 3.3 70B) |
| **Semantic Search** | FAISS (`faiss-cpu`) + `sentence-transformers` |
| **Data Validation** | Pydantic v2 |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Plotly, Altair |
| **Document Parsing** | pypdf (PDF), python-docx (DOCX) |
| **Containerization** | Docker |
| **Testing** | pytest, pytest-asyncio |

---

## Local Setup — Step-by-Step

### Prerequisites

- Python 3.11 or higher
- `pip`
- An [OpenRouter API key](https://openrouter.ai/) (free tier works; rate limits trigger automatic model failover)

### 1. Clone the Repository

```bash
git clone https://github.com/Soham-Lodh/Candidate-Scanner-Hackathon.git
cd Candidate-Scanner-Hackathon
```

### 2. Create a Virtual Environment

**macOS / Linux:**
```bash
python -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\activate
```

**Windows (Git Bash):**
```bash
python -m venv .venv
source .venv/Scripts/activate
```

> ⚠️ Make sure the virtual environment is active before running any further commands. Your terminal prompt should show `(.venv)`.

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

This installs all packages including PyTorch (CPU-only build), FAISS, sentence-transformers, Streamlit, Pydantic, Pandas, and document parsing libraries. Expect the first install to take a few minutes — PyTorch and sentence-transformers are large.

### 4. Configure Environment Variables

```bash
cp .env.example .env
```

Open `.env` in a text editor and fill in your OpenRouter API key:

```env
OPENROUTER_API_KEY=sk-or-xxxxxxxxxxxxxxxxxxxxxxxx

# Optional overrides (defaults shown):
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1/chat/completions
OPENROUTER_PRIMARY_MODEL=deepseek/deepseek-chat-v3-0324:free
OPENROUTER_TIMEOUT_SECONDS=45

APP_TOP_K_RETRIEVAL=5000
APP_TOP_K_FEATURES=500
APP_TOP_K_EXPLAIN=100
APP_ENABLE_AI_EXPLANATIONS=0
```

> **Getting an API key:** Sign up at [openrouter.ai](https://openrouter.ai), go to Keys, and create a free key. The free tier supports the default DeepSeek V3 model used by this app.

### 5. Launch the App

```bash
streamlit run app.py
```

The app opens at **http://localhost:8501**.

From there, use the **Upload** tab to load your job description, candidate JSONL, and schema — then click **Run Ranking Analysis**.

---

## Docker Setup

Docker is the recommended path for reproducing the exact hackathon environment (16 GB RAM, CPU-only).

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running

### 1. Set Your API Key

Edit `docker-compose.yml` and fill in `OPENROUTER_API_KEY`:

```yaml
environment:
  OPENROUTER_API_KEY: "sk-or-xxxxxxxxxxxxxxxxxxxxxxxx"
```

### 2. Build and Start

```bash
docker-compose up --build
```

The first build takes several minutes (downloads PyTorch CPU, FAISS, sentence-transformers). Subsequent starts are fast.

### 3. Access the App

Open **http://localhost:8501** in your browser.

### Docker Configuration

The `docker-compose.yml` is pre-configured to match the hackathon constraints:

```yaml
mem_limit: 16g          # 16 GB RAM limit
memswap_limit: 16g      # No swap
cpus: "1"               # 1 CPU core
ports:
  - "8501:8501"         # Streamlit UI
  - "8765:8765"         # Chunked upload sidecar
```

---

## Stage 3 CLI Reproduction (Offline Ranking)

The CLI uses a two-phase approach so the final ranking step is **fully offline** — no network access, no LLM calls, fully reproducible.

### Prerequisites

Complete [Local Setup](#local-setup--step-by-step) steps 1–4 before running these commands.

### Phase 1 — Pre-Computation (Online, Run Once Per JD)

This step calls OpenRouter once to extract structured JD intelligence and writes a reusable cache file to disk.

**Git Bash / macOS / Linux:**
```bash
python prepare.py \
  --job-description ./job_description.docx \
  --schema ./candidate_schema.json \
  --out jd_cache.json
```

**Windows PowerShell:**
```powershell
python prepare.py `
  --job-description ./job_description.docx `
  --schema ./candidate_schema.json `
  --out jd_cache.json
```

**Arguments:**

| Argument | Description |
|---|---|
| `--job-description` | Path to job description file (`.pdf`, `.docx`, `.txt`, `.md`) |
| `--schema` | Path to JSON Schema file describing candidate structure |
| `--out` | Output path for the JD intelligence cache (`.json`) |
| `--model` | Optional: override the default OpenRouter model |

**What this does:**
1. Parses the job description (supports PDF, DOCX, TXT, Markdown)
2. Calls OpenRouter once with the JD text and schema context
3. Extracts structured `JDIntelligence` (skills, seniority, responsibilities, etc.)
4. Writes the result as `jd_cache.json` — used by Phase 2

### Phase 2 — Offline Ranking (No Network, No LLM)

This is the command used to generate the submission CSV. It performs **no LLM calls** and **no network requests**.

**Git Bash / macOS / Linux:**
```bash
python rank.py \
  --candidates ./candidates.jsonl \
  --jd-cache jd_cache.json \
  --schema ./candidate_schema.json \
  --out submission.csv
```

**Windows PowerShell:**
```powershell
python rank.py `
  --candidates ./candidates.jsonl `
  --jd-cache jd_cache.json `
  --schema ./candidate_schema.json `
  --out submission.csv
```

**Arguments:**

| Argument | Description |
|---|---|
| `--candidates` | Path to candidate dataset (`.jsonl` or `.jsonl.gz`) |
| `--jd-cache` | Path to JD intelligence cache from Phase 1 |
| `--schema` | Path to JSON Schema file |
| `--out` | Output path for the submission CSV |

**What this does:**
1. Loads `JDIntelligence` from cache (no LLM calls)
2. Streams JSONL candidates from disk (no full-file memory load)
3. Runs semantic retrieval funnel (FAISS → top 5,000)
4. Runs the 21-signal feature scoring engine (keeps top 500 in a heap)
5. Exports the top 100 as a CSV

### Output CSV Format

```csv
candidate_id,rank,score,reasoning
CAND_0000042,1,1.00000000,"Backend Engineer with 6.9 yrs Toronto, Canada; 3 matched JD skills: Python (advanced, 26 mo), FastAPI (intermediate), PostgreSQL; ..."
CAND_0001337,2,0.83200000,...
```

| Column | Description |
|---|---|
| `candidate_id` | Unique identifier from the candidate record |
| `rank` | Integer rank (1 = best fit) |
| `score` | Min-max normalized float in `[0.1, 1.0]` |
| `reasoning` | Recruiter-facing explanation grounded in scoring evidence |

---

## Running Tests

```bash
pytest
```

The test suite covers:

| Test Module | Coverage |
|---|---|
| `test_export.py` | CSV generation, score normalization, reasoning output |
| `test_ingestion.py` | JSONL and gzipped JSONL parsing |
| `test_openrouter_client.py` | JSON validation, failover model ordering |
| `test_ranking.py` | Skill matching, stream scoring, top-K heap |
| `test_retrieval.py` | Streaming semantic retrieval, batch scoring |
| `test_schema_mapping.py` | Dynamic field resolution, path ranking |
| `test_services.py` | End-to-end pipeline with cached JD intelligence |

Run with verbose output:
```bash
pytest -v
```

Run a specific test file:
```bash
pytest tests/test_ranking.py -v
```

---

## Configuration Reference

All settings are loaded from environment variables (or `.env`):

| Variable | Default | Description |
|---|---|---|
| `OPENROUTER_API_KEY` | _(required)_ | Your OpenRouter API key |
| `OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1/chat/completions` | OpenRouter endpoint |
| `OPENROUTER_PRIMARY_MODEL` | `deepseek/deepseek-chat-v3-0324` | Primary LLM model |
| `OPENROUTER_TIMEOUT_SECONDS` | `45` | HTTP timeout per LLM request |
| `APP_TOP_K_RETRIEVAL` | `5000` | Candidates kept after semantic retrieval |
| `APP_TOP_K_FEATURES` | `500` | Candidates kept after feature scoring |
| `APP_TOP_K_EXPLAIN` | `100` | Candidates passed to AI explanation (if enabled) |
| `APP_ENABLE_AI_EXPLANATIONS` | `0` | Set to `1` to generate AI explanations per candidate |

**LLM Failover Chain** (automatic, no configuration needed):

1. `deepseek/deepseek-chat-v3-0324` (primary)
2. `qwen/qwen3-235b-a22b` (first fallback)
3. `meta-llama/llama-3.3-70b-instruct` (second fallback)

Each model retries with exponential backoff (2s → 4s → 8s) before failing over.

---

## Known Limitations

- **PDF/DOCX parsing** uses text extraction optimized for ranking workflows. Complex layouts with tables, multi-column text, or scanned pages may not extract perfectly.
- **Embedding model cold start:** `sentence-transformers/all-MiniLM-L6-v2` downloads (~90 MB) on first use. If unavailable, retrieval automatically switches to a deterministic hash-based fallback — ranking still works.
- **OpenRouter rate limits:** Free-tier accounts have per-minute limits. The automatic retry + failover chain handles this in most cases, but very high request volume may require a paid key.
- **Streamlit file size limit:** The deployed Streamlit Cloud app accepts files up to 500 MB. For datasets significantly larger than this, run locally or via Docker.
- **Deployed platform speed:** Streamlit Cloud's free tier provides limited CPU and memory, so the 100K ranking run takes 5–10 minutes. Locally or in Docker, the same run completes in under 6 minutes.

---

## Challenges We Faced

### 1. Making FAISS Work Under Constrained Memory
FAISS is fast but memory-hungry at 100K scale. We had to tune batch sizes for the streaming retrieval path and ensure vectors were kept as `float32` (not `float64`) to cut memory in half. When FAISS failed in cold-start environments (Streamlit Cloud), we built a NumPy dot-product fallback so the app never broke — it just got slightly slower.

### 2. Schema-Agnostic Field Resolution
The candidate dataset uses a schema we don't control. Instead of hardcoding field names like `candidate.skills` or `profile.years_of_experience`, we built a semantic `SchemaMap` layer that scans the uploaded JSON schema and ranks field paths by keyword relevance. This made the ranker truly dataset-agnostic — and it surfaced edge cases (arrays of arrays, nested objects with no `name` field) that required careful path resolution logic.

### 3. Preventing LLM-Driven Ranking from Failing at Scale
Early designs considered sending each candidate to an LLM for scoring. At 100K candidates this would cost hundreds of dollars and take hours. We redesigned around the principle that the LLM touches only the JD (once), not the candidates. All scoring is local, deterministic, and reproducible — which also means the final rank order is identical every run.

### 4. Stopping Keyword Stuffing from Gaming the Rankings
A candidate who writes "Python, Python, Python" in their summary would score high on a naive keyword counter. Our `_integrity` checks and the `must_have_gate` formula discount profiles with inflated promotional language, low evidence diversity, and missing corroborating signals (skill duration, endorsements, career history). This pushes authentic profiles up and gaming attempts down.

### 5. Chunked Uploads for Large Files on Streamlit
Streamlit's native uploader assembles the entire file in memory before the app receives it. For 400–500 MB JSONL files, this was initially impractical, so we built a sidecar chunked upload server (upload_server.py) — a lightweight Python HTTP server that accepted browser-sliced chunks and wrote them directly to disk, bypassing Streamlit's in-memory path entirely. We later discovered that Streamlit's uploader can be configured to accept files up to 500 MB by setting maxUploadSize = 500 and maxMessageSize = 500 in .streamlit/config.toml. With that in place, the native uploader handles the full 100K candidate dataset without the chunked server, so we removed the port-based sidecar from the active upload path entirely. The upload_server.py remains in the codebase but the default flow now uses Streamlit's built-in uploader — simpler, no extra port, and fully compatible with both local and deployed environments.

### 6. Scoring Reasoning That Doesn't Feel Generic
Early export CSV reasoning outputs were AI-generated and generic: "Strong candidate with relevant experience." We replaced this with evidence-grounded reasoning built directly from scored fields: skill proficiency levels, endorsement counts, GitHub activity scores, notice periods — real signal, not boilerplate. The model, when used for explanations, is explicitly instructed not to invent facts not present in the scoring output.

---

## Submission Assets

| Asset | Link |
|---|---|
| 🌐 **Live App** | [ai-candidate-ranker-hackathon.streamlit.app](https://ai-candidate-ranker-hackathon.streamlit.app/) |
| 💻 **GitHub Repository** | [github.com/Soham-Lodh/Candidate-Scanner-Hackathon](https://github.com/Soham-Lodh/Candidate-Scanner-Hackathon) |
| 📹 **Demo Video + Submission CSV** | [Google Drive](https://drive.google.com/drive/folders/1CSKugCC6FlCjdih-3ZUJ_vJ3DyQI-QRy) |

The demo video shows the full pipeline running inside Docker with 16 GB RAM and a single CPU — including the ranked candidate CSV output generated from the 100K dataset.

---

## License

MIT