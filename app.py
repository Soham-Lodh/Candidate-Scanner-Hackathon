"""Streamlit UI for the AI Candidate Ranking Platform."""

from __future__ import annotations
import tempfile
import asyncio
import logging
import os
import time
import uuid
from typing import Any

import altair as alt
import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components

from candidate_ranker.config import load_settings
from candidate_ranker.export import (
    TOP_CANDIDATE_EXPORT_LIMIT,
    ranking_export_rows,
    rankings_to_csv,
    scoring_reasoning,
)
from candidate_ranker.ingestion import read_job_description, read_schema
from candidate_ranker.models import MODEL_OPTIONS
from candidate_ranker.schema_mapping import build_schema_map
from candidate_ranker.services import RankingResult, run_pipeline_from_jsonl
from candidate_ranker.upload_server import ensure_upload_server, latest_session_upload, reset_session_upload

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
LOGGER = logging.getLogger(__name__)
CANDIDATE_UPLOAD_LIMIT_MB = 500
CANDIDATE_UPLOAD_LIMIT_BYTES = CANDIDATE_UPLOAD_LIMIT_MB * 1024 * 1024


def main() -> None:
    """Render the complete Streamlit application."""

    st.set_page_config(page_title="AI Candidate Ranking", layout="wide")
    st.title("AI Candidate Ranking Platform")
    _show_upload_limit_warning()
    settings = load_settings()
    _init_state(settings.openrouter_primary_model)
    # with st.sidebar:
    #     st.header("Model")
    #     label_by_id = {value: key for key, value in MODEL_OPTIONS.items()}
    #     current_label = label_by_id.get(st.session_state.selected_model, "DeepSeek V3")
    #     selected_label = st.selectbox("OpenRouter model", list(MODEL_OPTIONS), index=list(MODEL_OPTIONS).index(current_label))
    #     st.session_state.selected_model = MODEL_OPTIONS[selected_label]
    #     st.caption(st.session_state.selected_model)
    tabs = st.tabs(
        [
            "Upload",
            "JD Analysis",
            "Schema Analysis",
            "Ranking Progress",
            "Results",
            "Candidate Detail",
            "Analytics",
            "Export",
        ]
    )
    with tabs[0]:
        _upload_tab(settings)
    with tabs[1]:
        _jd_tab()
    with tabs[2]:
        _schema_tab()
    with tabs[3]:
        _progress_tab()
    with tabs[4]:
        _results_tab()
    with tabs[5]:
        _detail_tab()
    with tabs[6]:
        _analytics_tab()
    with tabs[7]:
        _export_tab()


def _init_state(default_model: str) -> None:
    defaults: dict[str, Any] = {
        "selected_model": default_model,
        "jd_text": "",
        "schema": None,
        "candidate_upload": None,
        "upload_session_id": uuid.uuid4().hex,
        "result": None,
        "pipeline_status": "Waiting for uploads",
        "upload_polling_active": False,
        "last_upload_check": 0.0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def is_streamlit_cloud() -> bool:
    return os.getenv("DEPLOYMENT_TARGET", "").lower() == "streamlit"


def _streamlit_candidate_uploader() -> None:
    candidate_file = st.file_uploader(
        "Candidate Dataset",
        type=["jsonl"],
        key="candidate_dataset",
    )

    if candidate_file is None:
        return

    tmp = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".jsonl",
    )

    tmp.write(candidate_file.getbuffer())
    tmp.close()

    st.session_state.candidate_upload = {
        "upload_id": "streamlit-upload",
        "filename": candidate_file.name,
        "path": tmp.name,
        "size": candidate_file.size,
        "received": candidate_file.size,
        "complete": True,
        "error": None,
    }

    st.success(
        f"Uploaded {candidate_file.name} "
        f"({candidate_file.size / 1024 / 1024:.1f} MB)"
    )

def _upload_tab(settings: Any) -> None:
    jd_file = st.file_uploader(
        "Job Description",
        type=["pdf", "docx", "txt", "md"]
    )

    upload_port = int(
        os.getenv("CANDIDATE_UPLOAD_PORT", "8765")
    )

    upload_host = os.getenv(
        "CANDIDATE_UPLOAD_HOST",
        "127.0.0.1"
    )

    upload_base_url = os.getenv(
        "CANDIDATE_UPLOAD_PUBLIC_URL",
        f"http://127.0.0.1:{upload_port}"
    )

    st.markdown("#### Candidate Dataset")

    if is_streamlit_cloud():

        st.info(
            "Running on Streamlit Cloud. "
            "Using native Streamlit uploader."
        )

        _streamlit_candidate_uploader()

    else:

        ensure_upload_server(
            upload_host,
            upload_port
        )

        with st.expander(
            "📋 How to upload your candidate file",
            expanded=True
        ):
            st.markdown(
                """
1. **Choose file** — click *Browse files* inside the upload card and select your `.jsonl` file from your computer.

2. **Upload** — click **Upload JSONL** to start the chunked upload *(supports files up to 500 MB)*.

3. **Confirm** — click **Refresh Upload Status** once the progress bar reaches 100% to verify completion.

4. **Replace** — to swap the file at any point, click **Replace Candidate File** and repeat the steps above.
                """
            )

        _chunked_candidate_uploader(
            upload_base_url,
            st.session_state.upload_session_id
        )

        _candidate_upload_controls()
        _render_candidate_upload_status()

    schema_file = st.file_uploader(
        "Candidate Schema",
        type=["json"]
    )

    if jd_file:
        st.session_state.jd_text = read_job_description(
            jd_file.name,
            jd_file.getvalue()
        )

        st.success(
            f"Loaded job description "
            f"({len(st.session_state.jd_text):,} characters)"
        )

    if schema_file:
        st.session_state.schema = read_schema(
            schema_file.getvalue()
        )

        st.success("Loaded candidate schema")

    candidate_upload = st.session_state.candidate_upload

    ready = bool(
        st.session_state.jd_text
        and candidate_upload
        and candidate_upload.get("complete")
        and st.session_state.schema
    )

    if st.button(
        "Run Ranking",
        disabled=not ready,
        type="primary"
    ):
        with st.spinner(
            "Running retrieval, scoring, and OpenRouter AI calls..."
        ):
            try:
                st.session_state.pipeline_status = "Running"

                st.session_state.result = asyncio.run(
                    run_pipeline_from_jsonl(
                        jd_text=st.session_state.jd_text,
                        candidate_path=candidate_upload["path"],
                        schema=st.session_state.schema,
                        settings=settings,
                        model=st.session_state.selected_model,
                    )
                )

                st.session_state.pipeline_status = "Complete"

                if not is_streamlit_cloud():
                    reset_session_upload(
                        st.session_state.upload_session_id,
                        delete_file=True
                    )

                st.session_state.upload_session_id = uuid.uuid4().hex
                st.session_state.candidate_upload = None
                st.session_state.upload_polling_active = False

                st.success("Ranking complete")

                if not is_streamlit_cloud():
                    st.info(
                        "Uploaded candidate file deleted from temporary storage."
                    )

            except Exception as exc:
                st.session_state.pipeline_status = f"Failed: {exc}"

                LOGGER.exception("Pipeline failed")

                st.error(str(exc))


def _jd_tab() -> None:
    result: RankingResult | None = st.session_state.result
    if result:
        st.json(result.jd_intelligence.model_dump())
    elif st.session_state.jd_text:
        st.text_area("Parsed job description", st.session_state.jd_text, height=320)
    else:
        st.info("Upload a job description to begin.")


def _schema_tab() -> None:
    if st.session_state.schema:
        schema_map = st.session_state.result.schema_map if st.session_state.result else build_schema_map(st.session_state.schema)
        st.json(schema_map.model_dump())
    else:
        st.info("Upload a JSON Schema Draft 7 file.")


def _progress_tab() -> None:
    st.metric("Status", st.session_state.pipeline_status)
    st.write(
        "Fast ranking funnel: stream JSONL from disk -> score each candidate locally -> "
        "keep top 500 -> show/export top 100."
    )


def _results_tab() -> None:
    result: RankingResult | None = st.session_state.result
    if not result:
        st.info("Run a ranking job to see results.")
        return
    df = _scores_df(result)
    st.caption(f"Showing top {len(df):,} candidates. CSV export uses the same rows.")
    st.dataframe(df, use_container_width=True, hide_index=True)


def _detail_tab() -> None:
    result: RankingResult | None = st.session_state.result
    if not result:
        st.info("Run a ranking job first.")
        return
    options = {f"{score.display_name} ({score.candidate_id})": score for score in result.ranked}
    selected = st.selectbox("Candidate", list(options))
    score = options[selected]
    explanation = {
        item.candidate_id: item for item in result.explanations.explanations
    }.get(score.candidate_id)
    left, right = st.columns([1, 1])
    with left:
        st.metric("Score", score.composite_score)
        st.subheader("Scoring Reasoning")
        st.write(scoring_reasoning(score))
        st.subheader("Score Breakdown")
        st.json(_score_detail(score))
    with right:
        st.subheader("Raw Candidate")
        st.json(score.raw_candidate)
        if explanation:
            st.subheader("AI Explanation")
            st.json(explanation.model_dump())


def _analytics_tab() -> None:
    result: RankingResult | None = st.session_state.result
    if not result:
        st.info("Analytics appear after ranking.")
        return
    df = _scores_df(result)
    st.plotly_chart(px.histogram(df, x="score", nbins=20), use_container_width=True)
    top_breakdowns = pd.DataFrame([score.breakdown for score in result.ranked[:20]])
    if not top_breakdowns.empty:
        chart_data = top_breakdowns.mean().reset_index()
        chart_data.columns = ["feature", "average_score"]
        st.altair_chart(
            alt.Chart(chart_data).mark_bar().encode(x="average_score:Q", y=alt.Y("feature:N", sort="-x")),
            use_container_width=True,
        )


def _export_tab() -> None:
    result: RankingResult | None = st.session_state.result
    if not result:
        st.info("Run ranking before export.")
        return
    csv_data = rankings_to_csv(result.ranked, result.explanations.explanations)
    st.download_button(
        f"Download Top {TOP_CANDIDATE_EXPORT_LIMIT} CSV",
        csv_data,
        file_name="top_100_candidate_rankings.csv",
        mime="text/csv",
    )


def _scores_df(result: RankingResult) -> pd.DataFrame:
    return pd.DataFrame(ranking_export_rows(result.ranked, result.explanations.explanations))


def _score_detail(score: Any) -> dict[str, Any]:
    data = score.model_dump(exclude={"raw_candidate", "composite_score"})
    data["score"] = score.composite_score
    return data


def _candidate_upload_controls() -> None:
    left, right = st.columns([1, 1])
    with left:
        if st.button("Refresh Upload Status", use_container_width=True):
            _refresh_candidate_upload()
            st.rerun()
    with right:
        if st.button("Replace Candidate File", use_container_width=True):
            reset_session_upload(st.session_state.upload_session_id, delete_file=True)
            st.session_state.upload_session_id = uuid.uuid4().hex
            st.session_state.candidate_upload = None
            st.session_state.upload_polling_active = False
            st.session_state.result = None
            st.session_state.pipeline_status = "Waiting for uploads"
            st.session_state.last_upload_check = 0.0
            st.rerun()


def _chunked_candidate_uploader(upload_base_url: str, session_id: str) -> None:
    components.html(
        f"""
        <style>
            :root {{
                color-scheme: light;
                font-family: "Source Sans Pro", Arial, sans-serif;
            }}

            .upload-card {{
                border: 1px solid #d6d9df;
                border-radius: 8px;
                padding: 16px;
                background: #000000;
                color: #ffffff;
            }}

            .upload-row {{
                display: flex;
                gap: 10px;
                align-items: center;
                flex-wrap: wrap;
            }}

            .upload-title {{
                font-size: 15px;
                font-weight: 600;
                margin: 0 0 10px;
            }}

            .upload-btn {{
                background: #ff4b4b;
                color: white;
                border: none;
                padding: 0.55rem 0.85rem;
                border-radius: 6px;
                cursor: pointer;
                font-weight: 600;
            }}

            .upload-btn:hover {{
                background: #ff3333;
            }}

            .upload-btn:disabled {{
                background: #c7ccd4;
                cursor: not-allowed;
            }}

            .file-meta {{
                margin-top: 10px;
                color: #ffffff;
                font-size: 13px;
            }}

            progress {{
                width: 100%;
                height: 14px;
                margin-top: 10px;
            }}

            .status {{
                margin-top: 10px;
                font-weight: 500;
                color: #262730;
            }}

            .stats {{
                display: flex;
                justify-content: space-between;
                margin-top: 8px;
                font-size: 12px;
                color: #555867;
                gap: 8px;
            }}

            input[type="file"] {{
                max-width: 100%;
            }}
        </style>

        <div class="upload-card">
            <div class="upload-title">Browse & Upload</div>
            <div class="upload-row">
                <input id="candidate-file" type="file" accept=".jsonl" />
                <button id="upload-button" class="upload-btn">Upload JSONL</button>
            </div>

            <div id="file-info" class="file-meta">
                No candidate file selected. Limit: {CANDIDATE_UPLOAD_LIMIT_MB} MB.
            </div>

            <progress id="upload-progress" max="100" value="0"></progress>

            <div class="stats">
                <span id="uploaded-size">0 MB</span>
                <span id="speed">0 MB/s</span>
                <span id="eta">ETA --</span>
            </div>

            <div id="upload-status" class="status"></div>

        </div>

        <script>
            const baseUrl = {upload_base_url!r};
            const sessionId = {session_id!r};

            const maxBytes = {CANDIDATE_UPLOAD_LIMIT_BYTES};
            const chunkSize = 4 * 1024 * 1024;
            const maxAttempts = 3;

            const fileInput = document.getElementById("candidate-file");
            const uploadBtn = document.getElementById("upload-button");

            const progress = document.getElementById("upload-progress");
            const status = document.getElementById("upload-status");
            const fileInfo = document.getElementById("file-info");

            const uploadedSize = document.getElementById("uploaded-size");
            const speedText = document.getElementById("speed");
            const etaText = document.getElementById("eta");

            function formatMB(bytes) {{
                return (bytes / 1024 / 1024).toFixed(1);
            }}

            function setStatus(text) {{
                status.innerText = text;
            }}

            async function fetchWithRetry(url, options) {{
                let lastError;
                for (let attempt = 1; attempt <= maxAttempts; attempt++) {{
                    try {{
                        const response = await fetch(url, options);
                        if (response.ok || response.status === 409 || response.status === 413) {{
                            return response;
                        }}
                        lastError = new Error(`HTTP ${{response.status}}`);
                    }} catch (error) {{
                        lastError = error;
                    }}
                    await new Promise(resolve => setTimeout(resolve, 350 * attempt));
                }}
                throw lastError;
            }}

            fileInput.addEventListener("change", () => {{
                const file = fileInput.files[0];

                if (!file) {{
                    fileInfo.innerText = "No candidate file selected. Limit: {CANDIDATE_UPLOAD_LIMIT_MB} MB.";
                    return;
                }}

                fileInfo.innerHTML =
                    `<b>${{file.name}}</b> · ${{formatMB(file.size)}} MB`;
                progress.value = 0;
                uploadedSize.innerText = "0 MB";
                speedText.innerText = "0 MB/s";
                etaText.innerText = "ETA --";
                setStatus("");
            }});

            uploadBtn.addEventListener("click", async () => {{

                const file = fileInput.files[0];

                if (!file) {{
                    setStatus("Select a JSONL file first.");
                    return;
                }}

                if (!file.name.toLowerCase().endsWith(".jsonl")) {{
                    setStatus("Only JSONL files are supported.");
                    return;
                }}

                if (file.size > maxBytes) {{
                    setStatus("File exceeds 500MB limit.");
                    return;
                }}

                uploadBtn.disabled = true;

                const startTime = Date.now();

                try {{

                    setStatus("Initializing upload...");

                    const startResponse = await fetchWithRetry(
                        `${{baseUrl}}/uploads/start`,
                        {{
                            method: "POST",
                            headers: {{
                                "Content-Type": "application/json"
                            }},
                            body: JSON.stringify({{
                                filename: file.name,
                                size: file.size,
                                session_id: sessionId
                            }})
                        }}
                    );

                    const startPayload = await startResponse.json();

                    if (!startResponse.ok)
                        throw new Error(startPayload.error);

                    const uploadId = startPayload.upload.upload_id;

                    let offset = 0;

                    while (offset < file.size) {{

                        const chunk = file.slice(
                            offset,
                            offset + chunkSize
                        );

                        const response = await fetchWithRetry(
                            `${{baseUrl}}/uploads/${{uploadId}}/chunk`,
                            {{
                                method: "POST",
                                headers: {{
                                    "X-Chunk-Offset": String(offset)
                                }},
                                body: chunk
                            }}
                        );

                        const payload = await response.json();

                        if (!response.ok)
                            throw new Error(payload.error);

                        offset += chunk.size;

                        const percent =
                            Math.round(
                                offset / file.size * 100
                            );

                        progress.value = percent;

                        const elapsed =
                            (Date.now() - startTime) / 1000;

                        const speed =
                            offset / elapsed;

                        const remaining =
                            (file.size - offset) /
                            Math.max(speed,1);

                        uploadedSize.innerText =
                            `${{formatMB(offset)}} / ${{formatMB(file.size)}} MB`;

                        speedText.innerText =
                            `${{formatMB(speed)}} MB/s`;

                        etaText.innerText =
                            `ETA ${{
                                Math.round(remaining)
                            }}s`;

                        setStatus(
                            `Uploading... ${{percent}}%`
                        );
                    }}

                    const completeResponse =
                        await fetchWithRetry(
                            `${{baseUrl}}/uploads/${{uploadId}}/complete`,
                            {{
                                method: "POST"
                            }}
                        );

                    const completePayload =
                        await completeResponse.json();

                    if (!completeResponse.ok)
                        throw new Error(
                            completePayload.error
                        );

                    progress.value = 100;
                    setStatus("Upload completed. Syncing...");

                }}
                catch(err) {{
                    setStatus(
                        "Upload failed: " + err.message
                    );
                }}
                finally {{
                    uploadBtn.disabled = false;
                }}
            }});
        </script>
        """,
        height=220,
    )


def _refresh_candidate_upload() -> None:
    """Fetch latest upload state from server."""
    state = latest_session_upload(st.session_state.upload_session_id)
    st.session_state.candidate_upload = None if state is None else {
        "upload_id": state.upload_id,
        "filename": state.filename,
        "path": state.path,
        "size": state.size,
        "received": state.received,
        "complete": state.complete,
        "error": state.error,
    }
    st.session_state.last_upload_check = time.time()


def _render_candidate_upload_status() -> None:
    """Display upload status with smart auto-polling."""
    _refresh_candidate_upload()

    upload = st.session_state.candidate_upload

    st.subheader("Candidate Upload Status")

    if not upload:
        st.info("No candidate dataset uploaded yet.")
        return

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("File", upload["filename"])

    with col2:
        st.metric("Size", f"{upload['size']/1024/1024:.1f} MB")

    with col3:
        st.metric("Received", f"{upload['received']/1024/1024:.1f} MB")

    with col4:
        st.metric("Status", "Ready" if upload.get("complete") else "Uploading")

    percent = (upload["received"] / upload["size"] * 100) if upload["size"] else 0
    st.progress(min(100, int(percent)))

    if upload.get("complete"):
        st.success(
            f"""
            Candidate dataset ready

            File: {upload['filename']}

            Size: {upload['size']/1024/1024:.1f} MB

            Ready for ranking.
            """
        )
        st.session_state.upload_polling_active = False
    elif upload.get("error"):
        st.error(upload["error"])
        st.session_state.upload_polling_active = False
    else:
        st.info(f"Uploading... {percent:.1f}%")
        # Auto-poll: check every 1.5 seconds if not complete
        current_time = time.time()
        if not st.session_state.upload_polling_active or (current_time - st.session_state.last_upload_check) >= 1.5:
            st.session_state.upload_polling_active = True
            time.sleep(1.5)
            st.rerun()



def _show_upload_limit_warning() -> None:
    upload_limit = int(st.get_option("server.maxUploadSize") or 0)
    message_limit = int(st.get_option("server.maxMessageSize") or 0)
    if upload_limit < CANDIDATE_UPLOAD_LIMIT_MB or message_limit < CANDIDATE_UPLOAD_LIMIT_MB:
        st.error(
            "This Streamlit server is still running with an upload/message limit below "
            f"{CANDIDATE_UPLOAD_LIMIT_MB}MB. Restart with "
            "`streamlit run app.py --server.maxUploadSize=500 --server.maxMessageSize=500`."
        )


if __name__ == "__main__":
    main()