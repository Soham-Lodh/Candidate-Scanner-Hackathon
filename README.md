# AI Candidate Ranking Platform

Production-oriented Streamlit app for ranking candidates against a job description with dynamic schema mapping, semantic retrieval, local feature scoring, integrity checks, and OpenRouter-only explanation generation.

## Architecture Summary

- `ai/openrouter_client.py` is the only module allowed to call OpenRouter.
- `candidate_ranker/ai_service.py` owns the two model interactions:
  - JD + schema intelligence extraction, strict JSON, temperature `0`.
  - Top-candidate explanation generation, strict JSON, temperature `0.2`.
- Ranking, retrieval, fraud detection, schema resolution, and exports run locally.
- Candidate fields are resolved through `SchemaMap`; downstream code does not assume fixed paths like `candidate.skills`.

## Dependency Graph

`app.py` -> services, UI session state  
`candidate_ranker/services.py` -> ingestion, schema mapper, AI service, retrieval, ranking, export  
`candidate_ranker/ai_service.py` -> `ai.openrouter_client` + Pydantic models  
`candidate_ranker/ranking.py` -> schema map, local text utilities, integrity checks  
`candidate_ranker/retrieval.py` -> embeddings + FAISS, with deterministic NumPy fallback for constrained environments

## Folder Structure

```text
ai/
  openrouter_client.py
candidate_ranker/
  ai_service.py
  config.py
  export.py
  ingestion.py
  models.py
  ranking.py
  retrieval.py
  schema_mapping.py
  services.py
  text.py
app.py
tests/
```

## Run

```bash
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
copy .env.example .env
streamlit run app.py
```

Set `OPENROUTER_API_KEY` in `.env`. All AI calls use OpenRouter-compatible model IDs.

## Docker

```bash
docker compose up --build
```

## Tests

```bash
pytest
```

## Known Limitations

- PDF and DOCX parsing use pragmatic text extraction suitable for ranking workflows, not full visual layout reconstruction.
- Embedding model download happens on first use. If unavailable, retrieval falls back to deterministic hash embeddings so the app remains runnable.
- The free OpenRouter models may be rate limited; automatic retry and failover are implemented.
