import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(
    page_title="FraudGuard Intelligence",
    page_icon="🛡️",
    layout="wide"
)

# Load model assets using a project-relative path for stability in different environments.
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "dataset" / "model.joblib"
FEATURE_NAMES_PATH = BASE_DIR / "dataset" / "feature_names.joblib"

model = joblib.load(MODEL_PATH)
feature_names = joblib.load(FEATURE_NAMES_PATH)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    :root {
        --bg-primary: #f4f7fb;
        --bg-secondary: #edf3ff;
        --panel: rgba(255, 255, 255, 0.88);
        --panel-strong: #ffffff;
        --line: rgba(148, 163, 184, 0.28);
        --sidebar: #0f172a;
        --sidebar-alt: #111827;
        --text: #0f172a;
        --muted: #475569;
        --primary: #3b82f6;
        --primary-soft: rgba(59, 130, 246, 0.14);
        --success: #10b981;
        --success-soft: rgba(16, 185, 129, 0.12);
        --danger: #ef4444;
        --danger-soft: rgba(239, 68, 68, 0.12);
        --warning: #f59e0b;
        --warning-soft: rgba(245, 158, 11, 0.14);
        --shadow: 0 22px 50px rgba(15, 23, 42, 0.10);
    }

    html, body, [data-testid="stAppViewContainer"], .main {
        background: linear-gradient(135deg, #f8fbff 0%, #edf3ff 35%, #f7fafc 100%);
        font-family: 'Inter', sans-serif;
    }

    .main > div {
        padding-top: 1rem;
        padding-bottom: 2rem;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b1220 0%, #111827 100%);
        border-right: 1px solid rgba(148, 163, 184, 0.18);
    }

    [data-testid="stSidebar"] .block-container {
        padding-top: 1.5rem;
        padding-bottom: 1.5rem;
    }

    div[data-testid="stNumberInput"] > div {
        background: rgba(15, 23, 42, 0.04);
        border: 1px solid var(--line);
        border-radius: 14px;
        box-shadow: inset 0 1px 2px rgba(15, 23, 42, 0.04);
    }

    div[data-testid="stNumberInput"] label {
        color: #dfe8f6 !important;
        font-weight: 600;
        font-size: 0.92rem;
    }

    div[data-testid="stNumberInput"] input {
        background: transparent !important;
        color: #e2e8f0 !important;
        font-weight: 600;
        border: none !important;
    }

    .stButton > button {
        width: 100%;
        border: none;
        border-radius: 14px;
        padding: 0.78rem 1rem;
        font-weight: 700;
        letter-spacing: 0.01em;
        background: linear-gradient(135deg, #2563eb, #3b82f6);
        color: white;
        box-shadow: 0 12px 25px rgba(37, 99, 235, 0.28);
        transition: transform 0.2s ease, box-shadow 0.2s ease, opacity 0.2s ease;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 16px 30px rgba(37, 99, 235, 0.30);
    }

    .stButton > button:focus {
        outline: none;
        box-shadow: 0 0 0 3px rgba(96, 165, 250, 0.35);
    }

    .metric-card, .panel, .status-card, .soft-card, .hero {
        background: rgba(255, 255, 255, 0.85);
        border: 1px solid rgba(148, 163, 184, 0.18);
        box-shadow: var(--shadow);
        border-radius: 22px;
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
    }

    .hero {
        padding: 1.5rem 1.5rem 1.1rem;
        margin-bottom: 1rem;
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.12), rgba(148, 163, 184, 0.10));
        border: 1px solid rgba(59, 130, 246, 0.12);
    }

    .metric-card {
        padding: 1.2rem 1.15rem;
        min-height: 124px;
    }

    .panel {
        padding: 1.35rem;
        margin-top: 0.5rem;
    }

    .soft-card {
        padding: 1rem 1.2rem;
    }

    .status-card {
        padding: 1.3rem 1.5rem;
        text-align: center;
        font-weight: 800;
        font-size: 1.28rem;
        border-radius: 20px;
    }

    .status-safe {
        border-color: rgba(16, 185, 129, 0.26);
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.08), rgba(255, 255, 255, 0.86));
        color: #065f46;
    }

    .status-danger {
        border-color: rgba(239, 68, 68, 0.28);
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.10), rgba(255, 255, 255, 0.86));
        color: #991b1b;
    }

    .status-chip {
        display: inline-block;
        padding: 0.45rem 0.8rem;
        border-radius: 999px;
        font-size: 0.74rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }

    .status-chip.safe {
        background: rgba(16, 185, 129, 0.12);
        color: #065f46;
    }

    .status-chip.danger {
        background: rgba(239, 68, 68, 0.12);
        color: #991b1b;
    }

    .kpi-label {
        color: var(--muted);
        font-size: 0.76rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }

    .kpi-value {
        margin-top: 0.42rem;
        font-size: 1.5rem;
        font-weight: 800;
        color: var(--text);
    }

    .section-title {
        font-size: 1.06rem;
        font-weight: 800;
        color: var(--text);
        margin: 0.25rem 0 0.85rem;
    }

    .subtle {
        color: var(--muted);
        line-height: 1.65;
    }

    .footer-note {
        margin-top: 1.6rem;
        color: var(--muted);
        text-align: center;
        font-size: 0.82rem;
    }

    @media (max-width: 768px) {
        .main > div {
            padding-left: 0.8rem;
            padding-right: 0.8rem;
        }

        .hero {
            padding: 1.25rem 1.1rem;
        }

        .status-card {
            font-size: 1.08rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown(
    """
    <div style="padding: 0.3rem 0.4rem 0.8rem;">
        <div style="display:flex; align-items:center; gap:0.8rem; margin-bottom: 1.25rem;">
            <div style="width:42px; height:42px; border-radius:14px; background:linear-gradient(135deg,#3b82f6,#60a5fa); display:flex; align-items:center; justify-content:center; font-size:1.2rem;">🛡️</div>
            <div>
                <div style="font-size:0.72rem; text-transform:uppercase; letter-spacing:0.08em; color:#93c5fd;">Risk Console</div>
                <div style="font-weight:700; color:white; font-size:1.05rem;">FraudGuard</div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.header("Transaction Details")

amount = st.sidebar.number_input(
    "Amount",
    min_value=0.0,
    max_value=100000.0,
    value=1500.0,
    step=100.0,
)

time = st.sidebar.number_input(
    "Time",
    min_value=0.0,
    max_value=172792.0,
    value=300.0,
)

st.sidebar.markdown(
    """
    <div style="margin-top:15px; padding:0.9rem 1rem; background: rgba(148,163,184,0.08); border:1px solid rgba(148,163,184,0.12); border-radius:16px; color:#e2e8f0;">
        <div style="font-size:0.72rem; letter-spacing:0.08em; text-transform:uppercase; color:#93c5fd; margin-bottom:0.35rem;">Monitoring</div>
        <div style="font-weight:600;">Account protection is active</div>
    </div>
    """,
    unsafe_allow_html=True,
)

header_col, chip_col = st.columns([4, 1])

with header_col:
    st.markdown(
        """
        <div class="hero">
            <h1 style="margin:0; font-size:2.25rem; letter-spacing:-0.04em; color:#0f172a;">FraudGuard Intelligence</h1>
            <p class="subtle" style="margin:0.5rem 0 0; font-size:1rem;">Intelligent transaction monitoring for secure, risk-aware decisions.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with chip_col:
    st.markdown(
        """
        <div class="soft-card" style="padding:0.9rem 1rem; margin-top:0.2rem; text-align:center;">
            <div class="status-chip safe">Live</div>
            <div class="kpi-value" style="font-size:1rem; margin-top:0.5rem;">Monitoring</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

kpi_1, kpi_2, kpi_3 = st.columns(3)
with kpi_1:
    st.markdown(
        """
        <div class="metric-card">
            <div class="kpi-label">Model</div>
            <div class="kpi-value">Random Forest</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with kpi_2:
    st.markdown(
        """
        <div class="metric-card">
            <div class="kpi-label">Risk Review</div>
            <div class="kpi-value">Instant</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with kpi_3:
    st.markdown(
        """
        <div class="metric-card">
            <div class="kpi-label">Status</div>
            <div class="kpi-value">Protected</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

left_panel, right_panel = st.columns([1.7, 1.0])

with left_panel:
    st.markdown(
        """
        <div class="panel">
            <div class="section-title">Transaction Profile</div>
            <div class="subtle">Review the incoming transaction and assess risk based on the trained fraud detection model.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    transaction = {
        "Time": time,
        "Amount": amount,
        **{f"V{i}": 0.0 for i in range(1, 29)},
    }

    X = pd.DataFrame(
        [[transaction[col] for col in feature_names]],
        columns=feature_names,
    )

    prediction = int(model.predict(X)[0])

    st.markdown(
        """
        <div class="panel">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.7rem;">
                <div class="section-title" style="margin:0;">Current Snapshot</div>
                <span class="status-chip safe">Ready</span>
            </div>
            <div class="subtle">Amount: <strong style="color:#0f172a;">$%.2f</strong></div>
            <div class="subtle" style="margin-top:0.35rem;">Elapsed time: <strong style="color:#0f172a;">%.2f</strong></div>
        </div>
        """
        % (amount, time),
        unsafe_allow_html=True,
    )

with right_panel:
    if prediction == 1:
        st.markdown(
            """
            <div class="status-card status-danger">
                <span class="status-chip danger">Potential Fraud</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.warning("Potential Fraud")
        st.write(
            "The transaction is flagged as potentially fraudulent. Please review the details and take necessary actions."
        )
    else:
        st.markdown(
            """
            <div class="status-card status-safe">
                <span class="status-chip safe">Legitimate</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.success("Legitimate")
        st.write(
            "This transaction appears to be legitimate based on the model's analysis. It shows no significant signs of fraudulent activity."
        )

st.markdown(
    """
    <div class="footer-note">FraudGuard Intelligence Platform • Transaction monitoring suite</div>
    """,
    unsafe_allow_html=True,
)
    