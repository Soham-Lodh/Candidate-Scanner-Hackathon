"""Professional UI redesign - AI Candidate Ranking Platform."""

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

# Custom color palette
COLORS = {
    "bg_primary": "#030027",
    "bg_secondary": "#151e3f",
    "text_primary": "#f2f3d9",
    "accent_warm": "#dc9e82",
    "accent_cool": "#c16e70",
    "text_secondary": "#b0b5c1",
    "border": "#2a3555",
    "success": "#66d9a6",
    "error": "#ff6b6b",
}


def _inject_custom_css() -> None:
    """Inject custom CSS for professional dark theme."""
    st.markdown(
        f"""
        <style>
            /* Root variables */
            :root {{
                --bg-primary: {COLORS['bg_primary']};
                --bg-secondary: {COLORS['bg_secondary']};
                --text-primary: {COLORS['text_primary']};
                --accent-warm: {COLORS['accent_warm']};
                --accent-cool: {COLORS['accent_cool']};
                --text-secondary: {COLORS['text_secondary']};
                --border: {COLORS['border']};
                --success: {COLORS['success']};
            }}

            /* Global styles */
            [data-testid="stApp"] {{
                background: linear-gradient(135deg, {COLORS['bg_primary']} 0%, {COLORS['bg_secondary']} 100%);
            }}

            .main {{
                background: transparent;
            }}

            /* Typography */
            h1, h2, h3 {{
                color: {COLORS['text_primary']};
                font-weight: 700;
                letter-spacing: -0.5px;
            }}

            h1 {{
                font-size: 2.5rem;
                margin-bottom: 1rem;
                background: linear-gradient(135deg, {COLORS['text_primary']} 0%, {COLORS['accent_warm']} 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
            }}

            h2 {{
                font-size: 1.75rem;
                margin-top: 2rem;
                margin-bottom: 1.5rem;
            }}

            h3 {{
                font-size: 1.25rem;
                margin-top: 1.5rem;
                margin-bottom: 1rem;
            }}

            p, span, label {{
                color: {COLORS['text_primary']};
            }}

            /* Tabs styling */
            [data-baseweb="tab-list"] {{
                border-bottom-color: {COLORS['border']};
                gap: 0.5rem;
            }}

            [role="tab"] {{
                color: {COLORS['text_secondary']};
                border-bottom-color: transparent;
                padding: 0.75rem 1.5rem;
                font-weight: 600;
                transition: all 0.3s ease;
            }}

            [role="tab"]:hover {{
                color: {COLORS['text_primary']};
            }}

            [role="tab"][aria-selected="true"] {{
                color: {COLORS['accent_warm']};
                border-bottom-color: {COLORS['accent_warm']};
            }}

            /* Buttons */
            .stButton > button {{
                background: linear-gradient(135deg, {COLORS['accent_warm']} 0%, {COLORS['accent_cool']} 100%);
                color: {COLORS['bg_primary']};
                border: none;
                border-radius: 8px;
                padding: 0.75rem 1.5rem;
                font-weight: 700;
                font-size: 1rem;
                cursor: pointer;
                transition: all 0.3s ease;
                box-shadow: 0 4px 15px rgba({int(COLORS['accent_warm'].lstrip('#')[0:2], 16)}, {int(COLORS['accent_warm'].lstrip('#')[2:4], 16)}, {int(COLORS['accent_warm'].lstrip('#')[4:6], 16)}, 0.2);
            }}

            .stButton > button:hover {{
                transform: translateY(-2px);
                box-shadow: 0 6px 20px rgba({int(COLORS['accent_warm'].lstrip('#')[0:2], 16)}, {int(COLORS['accent_warm'].lstrip('#')[2:4], 16)}, {int(COLORS['accent_warm'].lstrip('#')[4:6], 16)}, 0.35);
            }}

            .stButton > button:disabled {{
                opacity: 0.5;
                cursor: not-allowed;
                transform: none;
            }}

            /* File uploader */
            [data-testid="stFileUploadDropzone"] {{
                background: {COLORS['bg_secondary']};
                border: 2px dashed {COLORS['border']};
                border-radius: 12px;
                padding: 2rem;
            }}

            [data-testid="stFileUploadDropzone"]:hover {{
                border-color: {COLORS['accent_warm']};
                background: rgba({int(COLORS['accent_warm'].lstrip('#')[0:2], 16)}, {int(COLORS['accent_warm'].lstrip('#')[2:4], 16)}, {int(COLORS['accent_warm'].lstrip('#')[4:6], 16)}, 0.05);
            }}

            /* Metric cards */
            [data-testid="stMetricContainer"] {{
                background: {COLORS['bg_secondary']};
                border: 1px solid {COLORS['border']};
                border-radius: 12px;
                padding: 1.5rem;
                transition: all 0.3s ease;
            }}

            [data-testid="stMetricContainer"]:hover {{
                border-color: {COLORS['accent_warm']};
                box-shadow: 0 4px 20px rgba({int(COLORS['accent_warm'].lstrip('#')[0:2], 16)}, {int(COLORS['accent_warm'].lstrip('#')[2:4], 16)}, {int(COLORS['accent_warm'].lstrip('#')[4:6], 16)}, 0.15);
            }}

            /* DataFrames */
            [data-testid="dataframe"] {{
                background: {COLORS['bg_secondary']};
                border-radius: 12px;
                border: 1px solid {COLORS['border']};
            }}

            /* Spinner */
            .stSpinner > div > div {{
                border-color: {COLORS['accent_warm']};
                border-right-color: transparent;
            }}

            /* Progress bar */
            .stProgress > div > div > div {{
                background: linear-gradient(90deg, {COLORS['accent_warm']}, {COLORS['accent_cool']});
            }}

            /* Info, Success, Error messages */
            [data-testid="stInfo"], [data-testid="stSuccess"], [data-testid="stError"], [data-testid="stWarning"] {{
                background: {COLORS['bg_secondary']};
                border-radius: 12px;
                border-left: 4px solid {COLORS['accent_warm']};
                padding: 1rem;
                color: {COLORS['text_primary']};
            }}

            [data-testid="stSuccess"] {{
                border-left-color: {COLORS['success']};
            }}

            /* Expander */
            .streamlit-expanderHeader {{
                background: {COLORS['bg_secondary']};
                color: {COLORS['text_primary']};
                border-radius: 8px;
                border: 1px solid {COLORS['border']};
            }}

            .streamlit-expanderHeader:hover {{
                border-color: {COLORS['accent_warm']};
            }}

            /* Selectbox and inputs */
            [data-baseweb="select"] {{
                background: {COLORS['bg_secondary']};
                color: {COLORS['text_primary']};
                border-radius: 8px;
            }}

            input, select, textarea {{
                background: {COLORS['bg_secondary']};
                color: {COLORS['text_primary']};
                border: 1px solid {COLORS['border']};
                border-radius: 8px;
                padding: 0.75rem;
            }}

            input:focus, select:focus, textarea:focus {{
                border-color: {COLORS['accent_warm']};
                box-shadow: 0 0 0 3px rgba({int(COLORS['accent_warm'].lstrip('#')[0:2], 16)}, {int(COLORS['accent_warm'].lstrip('#')[2:4], 16)}, {int(COLORS['accent_warm'].lstrip('#')[4:6], 16)}, 0.1);
            }}

            /* Skeleton loaders */
            .skeleton {{
                background: linear-gradient(
                    90deg,
                    {COLORS['bg_secondary']} 0%,
                    {COLORS['border']} 50%,
                    {COLORS['bg_secondary']} 100%
                );
                background-size: 200% 100%;
                animation: skeleton-loading 1.5s infinite;
                border-radius: 8px;
            }}

            @keyframes skeleton-loading {{
                0% {{ background-position: 200% 0; }}
                100% {{ background-position: -200% 0; }}
            }}

            .skeleton-text {{
                height: 1rem;
                margin-bottom: 0.5rem;
                border-radius: 4px;
            }}

            .skeleton-card {{
                height: 120px;
                border-radius: 12px;
                margin-bottom: 1rem;
            }}

            /* JSON viewer */
            [data-testid="stJson"] {{
                background: {COLORS['bg_secondary']};
                border-radius: 12px;
                padding: 1.5rem;
                border: 1px solid {COLORS['border']};
                color: {COLORS['text_primary']};
            }}

            /* Text area */
            .stTextArea {{
                background: {COLORS['bg_secondary']};
            }}

            /* Sidebar */
            [data-testid="stSidebar"] {{
                background: {COLORS['bg_secondary']};
                border-right: 1px solid {COLORS['border']};
            }}

            /* Custom gradient separators */
            .gradient-divider {{
                height: 2px;
                background: linear-gradient(90deg, transparent, {COLORS['accent_warm']}, transparent);
                margin: 2rem 0;
            }}

            /* Card styling helper */
            .card {{
                background: {COLORS['bg_secondary']};
                border: 1px solid {COLORS['border']};
                border-radius: 12px;
                padding: 1.5rem;
                transition: all 0.3s ease;
            }}

            .card:hover {{
                border-color: {COLORS['accent_warm']};
                box-shadow: 0 4px 20px rgba({int(COLORS['accent_warm'].lstrip('#')[0:2], 16)}, {int(COLORS['accent_warm'].lstrip('#')[2:4], 16)}, {int(COLORS['accent_warm'].lstrip('#')[4:6], 16)}, 0.15);
            }}

            /* Badge */
            .badge {{
                display: inline-block;
                background: linear-gradient(135deg, {COLORS['accent_warm']}, {COLORS['accent_cool']});
                color: {COLORS['bg_primary']};
                padding: 0.35rem 0.85rem;
                border-radius: 20px;
                font-size: 0.85rem;
                font-weight: 600;
            }}

            /* Progress indicator */
            .progress-step {{
                display: inline-block;
                width: 40px;
                height: 40px;
                background: {COLORS['bg_secondary']};
                border: 2px solid {COLORS['border']};
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                color: {COLORS['text_secondary']};
                font-weight: 700;
                margin: 0 0.5rem;
                transition: all 0.3s ease;
            }}

            .progress-step.active {{
                background: linear-gradient(135deg, {COLORS['accent_warm']}, {COLORS['accent_cool']});
                border-color: {COLORS['accent_warm']};
                color: {COLORS['bg_primary']};
                box-shadow: 0 0 20px rgba({int(COLORS['accent_warm'].lstrip('#')[0:2], 16)}, {int(COLORS['accent_warm'].lstrip('#')[2:4], 16)}, {int(COLORS['accent_warm'].lstrip('#')[4:6], 16)}, 0.4);
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    """Render the complete redesigned Streamlit application."""
    st.set_page_config(
        page_title="AI Candidate Ranking",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    _inject_custom_css()

    # Header section
    col1, col2 = st.columns([0.85, 0.15])
    with col1:
        st.markdown(
            "# Candidate Ranking AI",
        )
        st.markdown(
            "_Intelligent candidate analysis powered by advanced AI models_",
            help="This platform uses AI to analyze job descriptions and rank candidates based on fit.",
        )
    with col2:
        status_color = "🟢" if st.session_state.get("pipeline_status") == "Complete" else "🟡" if "Running" in st.session_state.get("pipeline_status", "") else "⚫"
        st.metric(
            "Status",
            st.session_state.get("pipeline_status", "Ready"),
        )

    st.markdown("---")

    settings = load_settings()
    _init_state(settings.openrouter_primary_model)

    # Tab navigation
    tabs = st.tabs(
        [
            "📤 Upload",
            "📄 JD Analysis",
            "🗂️ Schema",
            "⚙️ Progress",
            "📊 Results",
            "👤 Candidate Detail",
            "📈 Analytics",
            "💾 Export",
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
    """Initialize session state with defaults."""
    defaults: dict[str, Any] = {
        "selected_model": default_model,
        "jd_text": "",
        "schema": None,
        "candidate_upload": None,
        "upload_session_id": uuid.uuid4().hex,
        "result": None,
        "pipeline_status": "Ready",
        "upload_polling_active": False,
        "last_upload_check": 0.0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def is_streamlit_cloud() -> bool:
    """Check if running on Streamlit Cloud."""
    return os.getenv("DEPLOYMENT_TARGET", "").lower() == "streamlit"


def _skeleton_loader(height: int = 120) -> None:
    """Display animated skeleton loader card."""
    st.markdown(
        f'<div class="skeleton skeleton-card" style="height: {height}px;"></div>',
        unsafe_allow_html=True,
    )


def _skeleton_text(width: str = "100%") -> None:
    """Display skeleton text loader."""
    st.markdown(
        f'<div class="skeleton skeleton-text" style="width: {width};"></div>',
        unsafe_allow_html=True,
    )


def _streamlit_candidate_uploader() -> None:
    """File uploader for Streamlit Cloud."""
    st.markdown("##### 📁 Upload Candidate Dataset")
    candidate_file = st.file_uploader(
        "Select JSONL file (Max 500 MB)",
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

    st.markdown(
        f"""
        <div class="card" style="background: linear-gradient(135deg, {COLORS['bg_secondary']}, {COLORS['border']});">
            <div style="color: {COLORS['success']}; font-weight: 700; margin-bottom: 0.5rem;">✓ Upload Successful</div>
            <div style="font-size: 0.9rem;">
                <div><strong>File:</strong> {candidate_file.name}</div>
                <div><strong>Size:</strong> {candidate_file.size / 1024 / 1024:.1f} MB</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _upload_tab(settings: Any) -> None:
    """Upload tab with file inputs and pipeline trigger."""
    st.markdown("### 📤 Data Upload & Configuration")

    # Three-column layout for uploads
    col1, col2= st.columns(2)

    with col1:
        st.markdown("##### 📋 Job Description")
        st.markdown(
            "_Upload the position description in any format_",
            help="Supports PDF, DOCX, TXT, and Markdown files",
        )
        jd_file = st.file_uploader(
            "Job Description",
            type=["pdf", "docx", "txt", "md"],
            key="jd_uploader",
            label_visibility="collapsed",
        )

    with col2:
        st.markdown("##### 🗂️ Candidate Schema")
        st.markdown(
            "_Define candidate field mappings_",
            help="JSON Schema Draft 7 format",
        )
        schema_file = st.file_uploader(
            "Candidate Schema",
            type=["json"],
            key="schema_uploader",
            label_visibility="collapsed",
        )


    st.markdown("---")

    # Process uploads
    if jd_file:
        st.session_state.jd_text = read_job_description(
            jd_file.name,
            jd_file.getvalue()
        )
        st.success(
            f"✓ Job description loaded ({len(st.session_state.jd_text):,} characters)"
        )

    if schema_file:
        st.session_state.schema = read_schema(
            schema_file.getvalue()
        )
        st.success("✓ Candidate schema loaded")

    # Candidate uploader section
    st.markdown("### 👥 Candidate Dataset")

    upload_port = int(os.getenv("CANDIDATE_UPLOAD_PORT", "8765"))
    upload_host = os.getenv("CANDIDATE_UPLOAD_HOST", "127.0.0.1")
    upload_base_url = os.getenv(
        "CANDIDATE_UPLOAD_PUBLIC_URL",
        f"http://127.0.0.1:{upload_port}"
    )

    if is_streamlit_cloud():
        _streamlit_candidate_uploader()
    else:
        ensure_upload_server(upload_host, upload_port)

        with st.expander(
            "📋 How to upload your candidate file",
            expanded=True
        ):
            st.markdown(
                """
                1. **Choose file** — click *Browse files* and select your `.jsonl` file
                2. **Upload** — click **Upload JSONL** to start (supports up to 500 MB)
                3. **Confirm** — click **Refresh Upload Status** once complete
                4. **Replace** — use **Replace Candidate File** to swap files
                """
            )

        _chunked_candidate_uploader(
            upload_base_url,
            st.session_state.upload_session_id
        )

        _candidate_upload_controls()
        _render_candidate_upload_status()

    # Readiness check and action button
    st.markdown("---")

    candidate_upload = st.session_state.candidate_upload
    ready = bool(
        st.session_state.jd_text
        and candidate_upload
        and candidate_upload.get("complete")
        and st.session_state.schema
    )

    # Status indicators
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        status = "✓" if st.session_state.jd_text else "○"
        st.metric(status + " Job Description", "Loaded" if st.session_state.jd_text else "Pending")
    with col2:
        status = "✓" if candidate_upload and candidate_upload.get("complete") else "○"
        st.metric(status + " Candidates", "Ready" if (candidate_upload and candidate_upload.get("complete")) else "Pending")
    with col3:
        status = "✓" if st.session_state.schema else "○"
        st.metric(status + " Schema", "Loaded" if st.session_state.schema else "Pending")
    with col4:
        status = "✓" if ready else "○"
        st.metric(status + " Ready to Rank", "Yes" if ready else "No")

    st.markdown("")

    # Main action button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button(
            "🚀 Run Ranking Analysis",
            disabled=not ready,
            type="primary",
            use_container_width=True,
        ):
            _run_ranking_pipeline(settings)


def _run_ranking_pipeline(settings: Any) -> None:
    """Execute the ranking pipeline with progress tracking."""
    st.session_state.pipeline_status = "Running"

    progress_placeholder = st.empty()
    status_placeholder = st.empty()

    with progress_placeholder.container():
        progress_bar = st.progress(0)

    stages = [
        ("📄 Parsing Job Description", 0.15),
        ("🔍 Analyzing Requirements", 0.30),
        ("👥 Processing Candidates", 0.50),
        ("🤖 Scoring with AI", 0.85),
        ("📊 Finalizing Rankings", 1.0),
    ]

    try:
        for i, (stage_name, progress_value) in enumerate(stages):
            with status_placeholder.container():
                st.markdown(
                    f"""
                    <div class="card">
                        <div style="color: {COLORS['accent_warm']}; font-weight: 700; margin-bottom: 0.5rem;">
                            {stage_name}
                        </div>
                        <div style="font-size: 0.85rem; color: {COLORS['text_secondary']};">
                            Please wait while we analyze your data... This may take 5-10 minutes.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            progress_bar.progress(min(progress_value, 0.99))
            time.sleep(0.5)

        with st.spinner("Running final analysis..."):
            candidate_upload = st.session_state.candidate_upload
            st.session_state.result = asyncio.run(
                run_pipeline_from_jsonl(
                    jd_text=st.session_state.jd_text,
                    candidate_path=candidate_upload["path"],
                    schema=st.session_state.schema,
                    settings=settings,
                    model=st.session_state.selected_model,
                )
            )

        progress_bar.progress(1.0)
        status_placeholder.empty()

        st.session_state.pipeline_status = "Complete"

        if not is_streamlit_cloud():
            reset_session_upload(
                st.session_state.upload_session_id,
                delete_file=True
            )

        st.session_state.upload_session_id = uuid.uuid4().hex
        st.session_state.candidate_upload = None
        st.session_state.upload_polling_active = False

        # Success message
        st.markdown(
            f"""
            <div class="card" style="background: linear-gradient(135deg, {COLORS['bg_secondary']}, {COLORS['border']});">
                <div style="color: {COLORS['success']}; font-weight: 700; font-size: 1.2rem; margin-bottom: 1rem;">
                    ✓ Ranking Complete!
                </div>
                <div style="font-size: 0.95rem;">
                    View detailed results in the <strong>Results</strong> and <strong>Analytics</strong> tabs.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if not is_streamlit_cloud():
            st.info("🔒 Uploaded candidate file securely deleted from temporary storage.")

    except Exception as exc:
        st.session_state.pipeline_status = f"Failed: {str(exc)}"
        LOGGER.exception("Pipeline failed")
        st.error(f"❌ Ranking failed: {str(exc)}")


def _jd_tab() -> None:
    """Display job description analysis."""
    result: RankingResult | None = st.session_state.result

    st.markdown("### 📄 Job Description Analysis")

    if result:
        st.markdown("#### Extracted Intelligence")
        st.json(result.jd_intelligence.model_dump())
    elif st.session_state.jd_text:
        st.markdown("#### Parsed Job Description")
        st.text_area(
            "Job Description Content",
            st.session_state.jd_text,
            height=400,
            disabled=True,
        )
    else:
        st.markdown(
            f"""
            <div class="card">
                <div style="text-align: center; padding: 2rem;">
                    <div style="font-size: 2rem; margin-bottom: 1rem;">📋</div>
                    <div style="font-weight: 600; margin-bottom: 0.5rem;">No Job Description Yet</div>
                    <div style="font-size: 0.9rem; color: {COLORS['text_secondary']};">
                        Upload a job description in the <strong>Upload</strong> tab to get started.
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _schema_tab() -> None:
    """Display candidate schema."""
    st.markdown("### 🗂️ Candidate Schema")

    if st.session_state.schema:
        schema_map = st.session_state.result.schema_map if st.session_state.result else build_schema_map(st.session_state.schema)
        st.json(schema_map.model_dump())
    else:
        st.markdown(
            f"""
            <div class="card">
                <div style="text-align: center; padding: 2rem;">
                    <div style="font-size: 2rem; margin-bottom: 1rem;">🗂️</div>
                    <div style="font-weight: 600; margin-bottom: 0.5rem;">No Schema Loaded</div>
                    <div style="font-size: 0.9rem; color: {COLORS['text_secondary']};">
                        Upload a JSON Schema Draft 7 file in the <strong>Upload</strong> tab.
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _progress_tab() -> None:
    """Display ranking progress information."""
    st.markdown("### ⚙️ Ranking Progress")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.metric("Current Status", st.session_state.get("pipeline_status", "Ready"))

    with col2:
        st.markdown(
            """
            **Ranking Funnel Process:**
            1. Stream JSONL candidates from disk
            2. Score each candidate locally
            3. Keep top 500 candidates
            4. Show & export top 100 results
            """
        )

    if "Running" in st.session_state.get("pipeline_status", ""):
        st.info(
            "⏳ Analysis in progress... This typically takes 5-10 minutes depending on candidate volume."
        )

        # Skeleton loaders while processing
        st.markdown("#### Processing Preview")
        col1, col2, col3 = st.columns(3)
        with col1:
            _skeleton_loader()
        with col2:
            _skeleton_loader()
        with col3:
            _skeleton_loader()


def _results_tab() -> None:
    """Display ranking results."""
    result: RankingResult | None = st.session_state.result

    st.markdown("### 📊 Ranking Results")

    if not result:
        st.markdown(
            f"""
            <div class="card">
                <div style="text-align: center; padding: 2rem;">
                    <div style="font-size: 2rem; margin-bottom: 1rem;">🎯</div>
                    <div style="font-weight: 600; margin-bottom: 0.5rem;">No Results Yet</div>
                    <div style="font-size: 0.9rem; color: {COLORS['text_secondary']};">
                        Run a ranking job in the <strong>Upload</strong> tab to see results.
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    df = _scores_df(result)

    st.markdown(f"#### Top {len(df):,} Candidates")
    st.caption(
        f"Showing the highest-scored candidates. This data is also available for export."
    )

    # Custom dataframe styling
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "composite_score": st.column_config.ProgressColumn(
                "Score",
                min_value=0,
                max_value=100,
            ),
        },
    )


def _detail_tab() -> None:
    """Display detailed candidate information."""
    result: RankingResult | None = st.session_state.result

    st.markdown("### 👤 Candidate Details")

    if not result:
        st.markdown(
            f"""
            <div class="card">
                <div style="text-align: center; padding: 2rem;">
                    <div style="font-size: 2rem; margin-bottom: 1rem;">🔍</div>
                    <div style="font-weight: 600; margin-bottom: 0.5rem;">No Candidates Yet</div>
                    <div style="font-size: 0.9rem; color: {COLORS['text_secondary']};">
                        Complete a ranking job first to view candidate details.
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    options = {f"{score.display_name} ({score.candidate_id})": score for score in result.ranked}
    selected = st.selectbox("Select Candidate", list(options), label_visibility="collapsed")
    score = options[selected]

    explanation = {
        item.candidate_id: item for item in result.explanations.explanations
    }.get(score.candidate_id)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("#### 📈 Score Metrics")
        st.metric("Composite Score", f"{score.composite_score:.1f}/100")

        st.markdown("#### 💡 Scoring Reasoning")
        st.markdown(scoring_reasoning(score))

        st.markdown("#### 🔢 Score Breakdown")
        st.json(_score_detail(score))

    with col2:
        st.markdown("#### 📋 Raw Candidate Data")
        st.json(score.raw_candidate)

        if explanation:
            st.markdown("#### 🤖 AI Explanation")
            st.json(explanation.model_dump())


def _analytics_tab() -> None:
    """Display analytics and insights."""
    result: RankingResult | None = st.session_state.result

    st.markdown("### 📈 Analytics & Insights")

    if not result:
        st.markdown(
            f"""
            <div class="card">
                <div style="text-align: center; padding: 2rem;">
                    <div style="font-size: 2rem; margin-bottom: 1rem;">📊</div>
                    <div style="font-weight: 600; margin-bottom: 0.5rem;">No Data Yet</div>
                    <div style="font-size: 0.9rem; color: {COLORS['text_secondary']};">
                        Analytics will appear after you complete a ranking job.
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    df = _scores_df(result)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Score Distribution")
        st.plotly_chart(
            px.histogram(df, x="score", nbins=20, title="Candidate Score Distribution"),
            use_container_width=True,
        )

    with col2:
        st.markdown("#### Top Features (Top 20 Candidates)")
        top_breakdowns = pd.DataFrame([score.breakdown for score in result.ranked[:20]])
        if not top_breakdowns.empty:
            chart_data = top_breakdowns.mean().reset_index()
            chart_data.columns = ["feature", "average_score"]
            st.altair_chart(
                alt.Chart(chart_data).mark_bar().encode(
                    x="average_score:Q",
                    y=alt.Y("feature:N", sort="-x")
                ),
                use_container_width=True,
            )


def _export_tab() -> None:
    """Display export options."""
    result: RankingResult | None = st.session_state.result

    st.markdown("### 💾 Export Results")

    if not result:
        st.markdown(
            f"""
            <div class="card">
                <div style="text-align: center; padding: 3rem;">
                    <div style="font-size: 2rem; margin-bottom: 1rem;">📥</div>
                    <div style="font-weight: 600; margin-bottom: 0.5rem;">No Results to Export</div>
                    <div style="font-size: 0.9rem; color: {COLORS['text_secondary']};">
                        Complete a ranking job to export results.
                    </div>
                </div>
            </div>
            """,

            unsafe_allow_html=True,
        )
        return

    st.markdown(
        f"""
        <div class="card">
            <div style="margin-bottom: 1.5rem;">
                <div style="font-weight: 600; font-size: 1.1rem; margin-bottom: 0.5rem;">
                    📊 Top {TOP_CANDIDATE_EXPORT_LIMIT} Candidates
                </div>
                <div style="font-size: 0.9rem; color: {COLORS['text_secondary']};">
                    Download your ranking results as a CSV file with detailed scoring information.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    csv_data = rankings_to_csv(result.ranked, result.explanations.explanations)

    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        st.download_button(
            f"📥 Download CSV (Top {TOP_CANDIDATE_EXPORT_LIMIT})",
            csv_data,
            file_name="candidate_rankings.csv",
            mime="text/csv",
            use_container_width=True,
            type="primary",
        )


def _scores_df(result: RankingResult) -> pd.DataFrame:
    """Convert ranking results to DataFrame."""
    return pd.DataFrame(ranking_export_rows(result.ranked, result.explanations.explanations))


def _score_detail(score: Any) -> dict[str, Any]:
    """Extract score details."""
    data = score.model_dump(exclude={"raw_candidate", "composite_score"})
    data["score"] = score.composite_score
    return data


def _candidate_upload_controls() -> None:
    """Display upload control buttons."""
    st.markdown("")
    left, right = st.columns([1, 1])
    with left:
        if st.button("🔄 Refresh Upload Status", use_container_width=True):
            _refresh_candidate_upload()
            st.rerun()
    with right:
        if st.button("🔁 Replace Candidate File", use_container_width=True):
            reset_session_upload(st.session_state.upload_session_id, delete_file=True)
            st.session_state.upload_session_id = uuid.uuid4().hex
            st.session_state.candidate_upload = None
            st.session_state.upload_polling_active = False
            st.session_state.result = None
            st.session_state.pipeline_status = "Ready"
            st.session_state.last_upload_check = 0.0
            st.rerun()


def _chunked_candidate_uploader(upload_base_url: str, session_id: str) -> None:
    """Render chunked file upload component."""
    components.html(
        f"""
        <style>
            :root {{
                --bg-primary: {COLORS['bg_primary']};
                --bg-secondary: {COLORS['bg_secondary']};
                --accent-warm: {COLORS['accent_warm']};
                --text-primary: {COLORS['text_primary']};
                --text-secondary: {COLORS['text_secondary']};
                --border: {COLORS['border']};
            }}

            .upload-card {{
                border: 2px dashed var(--border);
                border-radius: 12px;
                padding: 2rem;
                background: linear-gradient(135deg, var(--bg-secondary), {COLORS['bg_primary']});
                transition: all 0.3s ease;
            }}

            .upload-card:hover {{
                border-color: var(--accent-warm);
                background: linear-gradient(135deg, var(--bg-secondary), {COLORS['border']});
            }}

            .upload-row {{
                display: flex;
                gap: 12px;
                align-items: center;
                flex-wrap: wrap;
                margin-bottom: 1rem;
            }}

            .upload-title {{
                font-size: 15px;
                font-weight: 700;
                color: var(--text-primary);
                margin: 0 0 1rem;
            }}

            .upload-btn {{
                background: linear-gradient(135deg, var(--accent-warm), {COLORS['accent_cool']});
                color: {COLORS['bg_primary']};
                border: none;
                padding: 0.7rem 1.5rem;
                border-radius: 8px;
                cursor: pointer;
                font-weight: 700;
                font-size: 0.95rem;
                transition: all 0.3s ease;
                box-shadow: 0 4px 15px rgba(220, 158, 130, 0.2);
            }}

            .upload-btn:hover {{
                transform: translateY(-2px);
                box-shadow: 0 6px 20px rgba(220, 158, 130, 0.35);
            }}

            .upload-btn:disabled {{
                opacity: 0.5;
                cursor: not-allowed;
                transform: none;
            }}

            .file-meta {{
                margin-top: 1rem;
                color: var(--text-primary);
                font-size: 13px;
                font-weight: 500;
            }}

            progress {{
                width: 100%;
                height: 8px;
                margin: 1rem 0;
                border-radius: 4px;
                accent-color: var(--accent-warm);
            }}

            .stats {{
                display: flex;
                justify-content: space-between;
                margin: 0.8rem 0;
                font-size: 12px;
                color: var(--text-secondary);
                gap: 8px;
            }}

            .upload-status {{
                margin-top: 1rem;
                font-weight: 600;
                color: var(--accent-warm);
                font-size: 0.9rem;
            }}

            input[type="file"] {{
                color: var(--text-primary);
                accent-color: var(--accent-warm);
            }}

            input[type="file"]::file-selector-button {{
                background: var(--bg-secondary);
                color: var(--text-primary);
                border: 1px solid var(--border);
                padding: 0.5rem 1rem;
                border-radius: 6px;
                cursor: pointer;
                font-weight: 600;
                transition: all 0.3s ease;
            }}

            input[type="file"]::file-selector-button:hover {{
                border-color: var(--accent-warm);
                background: {COLORS['border']};
            }}
        </style>

        <div class="upload-card">
            <div class="upload-title">📁 Browse & Upload JSONL File</div>
            <div class="upload-row">
                <input id="candidate-file" type="file" accept=".jsonl" />
                <button id="upload-button" class="upload-btn">Upload JSONL</button>
            </div>

            <div id="file-info" class="file-meta">
                No file selected. Maximum size: {CANDIDATE_UPLOAD_LIMIT_MB} MB
            </div>

            <progress id="upload-progress" max="100" value="0"></progress>

            <div class="stats">
                <span id="uploaded-size">0 MB / 0 MB</span>
                <span id="speed">0 MB/s</span>
                <span id="eta">ETA --</span>
            </div>

            <div id="upload-status" class="upload-status"></div>
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

            function setStatus(text, color = "var(--accent-warm)") {{
                status.innerText = text;
                status.style.color = color;
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
                    fileInfo.innerText = "No file selected. Maximum size: {CANDIDATE_UPLOAD_LIMIT_MB} MB";
                    return;
                }}
                fileInfo.innerHTML = `<strong>${{file.name}}</strong> · ${{formatMB(file.size)}} MB`;
                progress.value = 0;
                setStatus("");
            }});

            uploadBtn.addEventListener("click", async () => {{
                const file = fileInput.files[0];

                if (!file) {{
                    setStatus("⚠️ Select a JSONL file first.", "var(--accent-warm)");
                    return;
                }}

                if (!file.name.toLowerCase().endsWith(".jsonl")) {{
                    setStatus("⚠️ Only JSONL files are supported.", "var(--accent-warm)");
                    return;
                }}

                if (file.size > maxBytes) {{
                    setStatus("❌ File exceeds 500MB limit.", "#ff6b6b");
                    return;
                }}

                uploadBtn.disabled = true;
                const startTime = Date.now();

                try {{
                    setStatus("⏳ Initializing upload...");

                    const startResponse = await fetchWithRetry(
                        `${{baseUrl}}/uploads/start`,
                        {{
                            method: "POST",
                            headers: {{"Content-Type": "application/json"}},
                            body: JSON.stringify({{
                                filename: file.name,
                                size: file.size,
                                session_id: sessionId
                            }})
                        }}
                    );

                    const startPayload = await startResponse.json();
                    if (!startResponse.ok) throw new Error(startPayload.error);

                    const uploadId = startPayload.upload.upload_id;
                    let offset = 0;

                    while (offset < file.size) {{
                        const chunk = file.slice(offset, offset + chunkSize);

                        const response = await fetchWithRetry(
                            `${{baseUrl}}/uploads/${{uploadId}}/chunk`,
                            {{
                                method: "POST",
                                headers: {{"X-Chunk-Offset": String(offset)}},
                                body: chunk
                            }}
                        );

                        const payload = await response.json();
                        if (!response.ok) throw new Error(payload.error);

                        offset += chunk.size;
                        const percent = Math.round(offset / file.size * 100);
                        progress.value = percent;

                        const elapsed = (Date.now() - startTime) / 1000;
                        const speed = offset / elapsed;
                        const remaining = (file.size - offset) / Math.max(speed, 1);

                        uploadedSize.innerText = `${{formatMB(offset)}} / ${{formatMB(file.size)}} MB`;
                        speedText.innerText = `${{formatMB(speed)}} MB/s`;
                        etaText.innerText = `ETA ${{Math.round(remaining)}}s`;
                        setStatus(`⏳ Uploading... ${{percent}}%`);
                    }}

                    const completeResponse = await fetchWithRetry(
                        `${{baseUrl}}/uploads/${{uploadId}}/complete`,
                        {{method: "POST"}}
                    );

                    const completePayload = await completeResponse.json();
                    if (!completeResponse.ok) throw new Error(completePayload.error);

                    progress.value = 100;
                    setStatus("✓ Upload completed successfully!", "#66d9a6");
                }} catch(err) {{
                    setStatus(`❌ Upload failed: ${{err.message}}`, "#ff6b6b");
                }} finally {{
                    uploadBtn.disabled = false;
                }}
            }});
        </script>
        """,
        height=260,
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

    if not upload:
        st.info("📤 No candidate dataset uploaded yet.")
        return

    st.markdown("#### 📊 Upload Status")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("📄 Filename", upload["filename"][-20:] if len(upload["filename"]) > 20 else upload["filename"])

    with col2:
        st.metric("💾 Total Size", f"{upload['size']/1024/1024:.1f} MB")

    with col3:
        st.metric("⬆️ Received", f"{upload['received']/1024/1024:.1f} MB")

    with col4:
        status_text = "✓ Ready" if upload.get("complete") else "⏳ Uploading"
        st.metric("🔄 Status", status_text)

    percent = (upload["received"] / upload["size"] * 100) if upload["size"] else 0
    st.progress(min(100, int(percent)))

    if upload.get("complete"):
        st.markdown(
            f"""
            <div class="card" style="background: linear-gradient(135deg, {COLORS['bg_secondary']}, {COLORS['border']});">
                <div style="color: {COLORS['success']}; font-weight: 700; margin-bottom: 1rem; font-size: 1.1rem;">
                    ✓ Dataset Ready for Ranking
                </div>
                <div style="font-size: 0.9rem; line-height: 1.6;">
                    <div><strong>File:</strong> {upload['filename']}</div>
                    <div><strong>Size:</strong> {upload['size']/1024/1024:.1f} MB</div>
                    <div style="margin-top: 0.5rem; color: {COLORS['text_secondary']};">Ready to proceed with ranking analysis.</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.session_state.upload_polling_active = False
    elif upload.get("error"):
        st.error(f"❌ Upload error: {upload['error']}")
        st.session_state.upload_polling_active = False
    else:
        st.info(f"⏳ Uploading... {percent:.1f}% complete")
        current_time = time.time()
        if not st.session_state.upload_polling_active or (current_time - st.session_state.last_upload_check) >= 1.5:
            st.session_state.upload_polling_active = True
            time.sleep(1.5)
            st.rerun()


if __name__ == "__main__":
    main()