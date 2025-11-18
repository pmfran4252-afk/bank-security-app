import os
import sys
import base64

import numpy as np
import pandas as pd
import streamlit as st

# Ensure project root is on sys.path (helps Streamlit find qid_core/qid_security)
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from qid_security import (
    generate_synthetic_transactions,
    SecurityInterferenceEncoder,
    SecurityPruner,
    CHANNELS,  # harmless if unused
)
from qid_security.amplitude_masks_security import get_security_mask

# -----------------------------------------------------------------------------
# Streamlit page config
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="QID Bank Security Demo",
    layout="wide",
)

# -----------------------------------------------------------------------------
# U.S. Bank–style theme (CSS injection)
# -----------------------------------------------------------------------------
US_BANK_CSS = """
<style>

    /* ============================
       U.S. BANK SIDEBAR STYLING
       ============================ */
    [data-testid="stSidebar"] {
        background-color: #0A2640 !important;  /* Dark navy */
        color: #FFFFFF !important;
    }

    [data-testid="stSidebar"] * {
        color: #FFFFFF !important;
    }

    [data-testid="stSidebar"] button {
        background-color: #0056B3 !important;  /* Primary blue */
        color: #FFFFFF !important;
        border-radius: 6px !important;
        border: 1px solid #0072CE !important;
        padding: 0.4rem 0.75rem !important;
        font-weight: 500 !important;
    }

    [data-testid="stSidebar"] button:hover {
        background-color: #0072CE !important;
        border-color: #FFFFFF !important;
    }

    /* Sliders */
    .stSlider > div > div > div > div {
        background: #0072CE !important;
    }
    .stSlider > div > div > div:nth-child(3) > div {
        background: #0056B3 !important;
        border: 2px solid #FFFFFF !important;
    }

    /* Sidebar radio highlight */
    .stRadio > label > div[role='radiogroup'] > div[aria-checked='true'] {
        color: #FFFFFF !important;
        background-color: #0072CE20 !important;
        border-left: 3px solid #0072CE !important;
        padding-left: 8px !important;
        border-radius: 4px !important;
    }
    /* Bright blue for labels in light mode */
    .block-container label {
    color: #0056B3 !important;
    font-weight: 600 !important;
    }

    /* Slider min/max number color fix */
    .stSlider label, 
    .stSlider div, 
    .stSlider input {
    color: #FFFFFF !important;
    }

    /* ============================
       MAIN PAGE BACKGROUND + TEXT
       ============================ */

    html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
        background-color: #EDF2F7 !important;  /* Soft bluish grey */
        background-image: none !important;
    }

    /* Top header bar (Share / star / GitHub) */
    [data-testid="stHeader"] {
        background-color: #0A2640 !important;  /* match sidebar */
    }
    [data-testid="stHeader"] * {
        color: #FFFFFF !important;
    }
    /* Remove mysterious empty white box under banners */
    .element-container:has(> div:empty) {
    display: none !important;
    height: 0 !important;
    padding: 0 !important;
    margin: 0 !important;
    }

    /* Default text color in main area */
    body,
    .block-container,
    .block-container p,
    .block-container li,
    .block-container span,
    .markdown-text-container,
    .markdown-text-container p {
        color: #0A2640 !important;  /* Navy text */
    }

    .block-container {
        background-color: #EDF2F7 !important;
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
    }

    /* Headings */
    h1, h2, h3, h4 {
        color: #0A2640 !important;  /* U.S. Bank Navy */
        font-weight: 700 !important;
    }

    .fraud-pill {
    display: inline-block;
    padding: 0.15rem 0.6rem;
    border-radius: 999px;
    background: #0056B3;
    color: #FFFFFF !important;   /* FORCE white text */
    font-size: 0.8rem;
    font-weight: 700;
    }

    /* Metric numbers bright blue */
    [data-testid="stMetricValue"] {
        color: #0056B3 !important;  /* bright blue */
        font-weight: 700 !important;
    }
    [data-testid="stMetricLabel"] {
        color: #0A2640 !important;
    }

    /* ============================
       CARD SECTION STYLES
       ============================ */
    .us-section {
        border-radius: 18px;
        padding: 1.75rem 1.75rem 1.5rem 1.75rem;
        margin-bottom: 2rem;
        box-shadow: 0 20px 40px rgba(10, 38, 64, 0.06);
    }

    .us-section-white {
        background-color: #FFFFFF !important;
    }

    .us-section-grey {
        background-color: #F5F7FA !important; /* Light grey */
    }

    /* Accent text (for hints / explanations) */
    .us-accent {
        color: #0056B3 !important;  /* Bright blue */
        font-weight: 600;
    }

    .fraud-flag {
        color: #D00000 !important;  /* Brand red */
        font-weight: 700;
    }

    /* Blue pill for "Label_fraud = 1" */
    .fraud-pill {
        display: inline-block;
        padding: 0.15rem 0.6rem;
        border-radius: 999px;
        background: #0056B3;
        color: #FFFFFF;
        font-size: 0.8rem;
        font-weight: 600;
    }

    /* ============================
       TABLES / DATAFRAMES
       ============================ */

    /* st.dataframe */
    div[data-testid="stDataFrame"] {
        background-color: #FFFFFF !important;
        color: #0A2640 !important;
    }
    div[data-testid="stDataFrame"] table {
        background-color: #FFFFFF !important;
        color: #0A2640 !important;
    }
    div[data-testid="stDataFrame"] tbody tr td {
        background-color: #FFFFFF !important;
        color: #0A2640 !important;
    }

    /* st.table */
    div[data-testid="stTable"] {
        background-color: #FFFFFF !important;
        color: #0A2640 !important;
    }
    div[data-testid="stTable"] table,
    div[data-testid="stTable"] tbody tr td {
        background-color: #FFFFFF !important;
        color: #0A2640 !important;
    }

    /* ============================
       SELECT BOXES & INPUTS
       ============================ */

    /* Main area selects: white background, navy text */
    .block-container div[data-baseweb="select"] > div {
        background-color: #FFFFFF !important;
        color: #0A2640 !important;
        border-radius: 8px !important;
    }

    .block-container div[data-baseweb="select"] input {
        color: #0A2640 !important;
    }

    .block-container input,
    .block-container textarea {
        background-color: #FFFFFF !important;
        color: #0A2640 !important;
    }

    /* ============================
       CHART BACKGROUNDS
       ============================ */

    div[data-testid="stChart"] {
        background-color: #FFFFFF !important;
        border-radius: 12px !important;
        padding: 0.75rem !important;
    }

</style>
"""
st.markdown(US_BANK_CSS, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Banner helper (bright blue pill)
# -----------------------------------------------------------------------------
def blue_banner(message: str):
    st.markdown(
        f"""
        <div style="
            margin-top: 0.75rem;
            margin-bottom: 0.75rem;
            padding: 0.9rem 1.2rem;
            border-radius: 999px;
            background: #0056B3;
            color: #FFFFFF;
            font-weight: 500;
            text-align: center;
            box-shadow: 0 12px 30px rgba(0, 86, 179, 0.35);
        ">
            {message}
        </div>
        """,
        unsafe_allow_html=True,
    )

# -----------------------------------------------------------------------------
# Title & intro
# -----------------------------------------------------------------------------
st.title("QID Bank Security Interference Demo")

st.markdown(
    """
This app simulates a **large fictional bank transaction dataset** and runs it through a
**Quantum-Inspired Interference (QID) Core** with **bank-specific amplitude masks**.

**Conceptual story for U.S. Bank:**

- Each account's transaction history is encoded as a **complex-valued waveform**.
- A **typology-specific amplitude mask** (ALL_FRAUD, CARD_FRAUD, ATO, STRUCTURING, MULE)
  tells QID what to "listen for."
"""
)

# -----------------------------------------------------------------------------
# Baseline rules score: simple, explainable rule-based risk per account
# -----------------------------------------------------------------------------
def baseline_rules_score_account(events: pd.DataFrame, typology: str) -> float:
    """
    Deliberately simple account-level rules score.
    """
    g = events
    n = len(g)
    if n == 0:
        return 0.0

    frac_cross_border = (g["dst_country"] != "US").mean()
    frac_new_device = (g["device_age_days"] < 7).mean()
    frac_high_amount_200 = (g["amount"] > 200).mean()
    frac_high_amount_3000 = (g["amount"] > 3000).mean()
    frac_structuring_band = ((g["amount"] >= 8000) & (g["amount"] < 10000)).mean()
    frac_zelle_ach = g["channel"].isin(["ZELLE", "ACH"]).mean()
    frac_card_ecom = (g["channel"] == "CARD_ECOM").mean()
    frac_wire_or_zelle = g["channel"].isin(["WIRE_INTL", "ZELLE"]).mean()

    # ALL_FRAUD uses a blended baseline
    if typology == "ALL_FRAUD":
        return (
            1.2 * frac_card_ecom
            + 1.2 * frac_zelle_ach
            + 1.2 * frac_wire_or_zelle
            + 1.3 * frac_cross_border
            + 1.0 * frac_new_device
            + 1.0 * frac_high_amount_200
            + 1.2 * frac_structuring_band
        )

    if typology == "CARD_FRAUD":
        return (
            1.5 * frac_card_ecom
            + 1.5 * frac_cross_border
            + 1.0 * frac_new_device
            + 1.0 * frac_high_amount_200
        )

    if typology == "ATO":
        return (
            2.0 * (frac_wire_or_zelle * frac_high_amount_3000)
            + 1.5 * frac_new_device
            + 1.0 * frac_cross_border
        )

    if typology == "STRUCTURING":
        return (
            2.0 * frac_structuring_band
            + 1.0 * g["channel"].isin(["ACH", "ATM"]).mean()
        )

    if typology == "MULE":
        return (
            1.5 * frac_zelle_ach
            + 1.5 * frac_cross_border
            + 1.0 * g["amount"].between(200, 2500).mean()
        )

    return 0.0


# -----------------------------------------------------------------------------
# Sidebar controls + logo
# -----------------------------------------------------------------------------
st.sidebar.header("")


def load_svg_as_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


# Sidebar SVG logo (if present)
svg_path = os.path.join(ROOT_DIR, "assets", "logo.svg")  # adjust name if needed
if os.path.exists(svg_path):
    svg_b64 = load_svg_as_base64(svg_path)
    st.sidebar.markdown(
        f"""
        <div style="text-align: center; margin-bottom: 50px;">
            <img src="data:image/svg+xml;base64,{svg_b64}" style="width: 180px;"/>
        </div>
        """,
        unsafe_allow_html=True,
    )

if "df" not in st.session_state:
    st.session_state["df"] = None
    st.session_state["scores"] = None
    st.session_state["encoder"] = SecurityInterferenceEncoder(signature_dim=64)
    st.session_state["pruner"] = SecurityPruner(st.session_state["encoder"])

n_accounts = st.sidebar.slider("Number of accounts", 2000, 20000, 10000, step=1000)
days = st.sidebar.slider("Simulation window (days)", 30, 120, 60, step=10)
avg_txn_per_day = st.sidebar.slider(
    "Avg. transactions per account per day", 1.0, 5.0, 3.0, step=0.5
)
fraud_ratio = st.sidebar.slider(
    "Fraction of accounts with injected suspicious behavior", 0.01, 0.10, 0.03, step=0.01
)
max_accounts_to_score = st.sidebar.slider(
    "Max accounts to score (for speed)", 1000, 10000, 5000, step=500
)

typology = st.sidebar.selectbox(
    "QID Typology Mask",
    [
        "ALL_FRAUD",   # catch-all, broad radar
        "CARD_FRAUD",
        "ATO",
        "STRUCTURING",
        "MULE",
    ],
    index=0,
)

st.sidebar.caption(
    """
**ALL_FRAUD** = broad, catch-all radar across patterns.  
Other options zoom in on a specific typology.
"""
)

high_risk_countries = st.sidebar.multiselect(
    "High-risk countries (boosted by mask)",
    options=["MX", "BR", "CN", "RU", "NG", "UA"],
    default=["MX", "BR", "CN", "RU", "NG"],
)

# Put dataset generation in an expander at bottom of sidebar
with st.sidebar.expander("⚙️ Advanced: Dataset Controls", expanded=False):
    generate_btn = st.button(
        "Generate / Regenerate Synthetic Dataset",
        key="btn_generate_dataset",
    )

run_qid_btn = st.sidebar.button(
    "Run QID Scoring with Current Mask",
    key="btn_run_qid",
)

# Bottom-of-sidebar view toggle
view_mode = st.sidebar.radio(
    "Main view",
    ["QID Rankings & Drilldown", "QID vs Baseline Rules"],
    index=0,
)

# -----------------------------------------------------------------------------
# Generate dataset
# -----------------------------------------------------------------------------
if generate_btn or (st.session_state["df"] is None):
    with st.spinner("Generating synthetic dataset..."):
        df = generate_synthetic_transactions(
            n_accounts=n_accounts,
            days=days,
            avg_txn_per_day=avg_txn_per_day,
            fraud_ratio=fraud_ratio,
            seed=42,
        )
        st.session_state["df"] = df
        st.session_state["scores"] = None
    blue_banner(
        f"Generated dataset with {len(df):,} transactions for {n_accounts:,} accounts."
    )

df = st.session_state["df"]

if df is None:
    st.stop()

# -----------------------------------------------------------------------------
# Dataset overview section (white card)
# -----------------------------------------------------------------------------
st.markdown('<div class="us-section us-section-white">', unsafe_allow_html=True)

st.subheader("Dataset Overview")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Transactions", f"{len(df):,}")
with col2:
    st.metric("Accounts", f"{df['account_id'].nunique():,}")
with col3:
    st.metric("Fraudulent Txns (Injected)", f"{(df['label_fraud'] == 1).sum():,}")
with col4:
    accounts_with_fraud = df.groupby("account_id")["label_fraud"].max().sum()
    st.metric("Accounts with Any Fraud", f"{accounts_with_fraud:,}")

st.markdown(
    "Use the sidebar to select a **QID typology mask** and then click "
    "**Run QID Scoring** to rank accounts by masked interference score."
)

st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Run QID scoring
# -----------------------------------------------------------------------------
if run_qid_btn:
    with st.spinner("Running QID scoring & pruning..."):
        amp_mask = get_security_mask(
            typology,
            params={"high_risk_countries": high_risk_countries},
        )
        pruner: SecurityPruner = st.session_state["pruner"]
        score_df = pruner.score_all_accounts(
            df,
            amp_mask=amp_mask,
            max_accounts=max_accounts_to_score,
        )
        st.session_state["scores"] = score_df
    blue_banner("QID scoring complete.")

score_df = st.session_state.get("scores")

# If we have scores, ensure baseline rules score is computed; else just keep None.
if score_df is not None and not score_df.empty and "rules_score" not in score_df.columns:
    grouped = df.groupby("account_id", sort=False)
    rules_scores = []
    for acc_id in score_df["account_id"]:
        g = grouped.get_group(acc_id)
        rs = baseline_rules_score_account(g, typology)
        rules_scores.append(rs)
    score_df = score_df.copy()
    score_df["rules_score"] = rules_scores
    st.session_state["scores"] = score_df

# -----------------------------------------------------------------------------
# View: QID Rankings & Drilldown
# -----------------------------------------------------------------------------
if view_mode == "QID Rankings & Drilldown":
    if score_df is None or score_df.empty:
        blue_banner("Run QID Scoring from the sidebar to populate results.")
    else:
        # Section: QID-ranked accounts (grey card)
        st.markdown('<div class="us-section us-section-grey">', unsafe_allow_html=True)

        st.subheader("QID-Ranked Accounts (Top Suspicious Entities Under Current Mask)")

        st.caption(
            "Each account is assigned a QID score based on its masked interference signature. "
            "Higher scores indicate more suspicious behavior under the chosen typology mask."
        )

        top_k = st.slider("Show top K accounts", 10, 200, 50, step=10, key="top_k_qid")
        top_accounts = score_df.head(top_k)

        labeled = int(top_accounts["label_fraud"].sum())
        st.markdown(
            f"Among the top **{top_k}** accounts by **QID score**, **{labeled}** have injected suspicious activity "
            f'(<span class="fraud-pill">Label_fraud = 1</span>) in this fictional dataset.',
            unsafe_allow_html=True,
        )

        st.dataframe(top_accounts)

        st.markdown('</div>', unsafe_allow_html=True)

        # Section: Account drilldown + explanation + signature (white card)
        st.markdown('<div class="us-section us-section-white">', unsafe_allow_html=True)

        st.markdown("### Account-level Drilldown")

        selected_account_id = st.selectbox(
            "Select an account to inspect",
            options=top_accounts["account_id"].tolist(),
            key="drill_account_qid",
        )

        acc_events = df[df["account_id"] == selected_account_id].copy()
        acc_events = acc_events.sort_values("timestamp")
        st.write(
            f"Showing **{len(acc_events)}** transactions for account `{selected_account_id}`."
        )

        # Timeline chart
        acc_events["time_hours"] = acc_events["timestamp"] / 3600.0
        st.line_chart(
            acc_events.set_index("time_hours")[["amount"]],
            height=200,
        )

        st.markdown("**Transaction details (first 200 rows)**")
        acc_head = acc_events[
            [
                "txn_id",
                "timestamp",
                "channel",
                "merchant_type",
                "src_country",
                "dst_country",
                "amount",
                "device_age_days",
                "account_age_days",
                "risk_segment",
                "label_fraud",
                "label_typology",
            ]
        ].head(200)
        st.dataframe(acc_head)

        # Transaction-level QID reason explorer
        st.markdown("### Transaction-level QID Explanation")

        txn_choice = st.selectbox(
            "Select a transaction to explain (by txn_id)",
            options=acc_head["txn_id"].tolist(),
            key="txn_explain_select",
        )

        txn_row = acc_head[acc_head["txn_id"] == txn_choice].iloc[0]

        amp_mask = get_security_mask(
            typology,
            params={"high_risk_countries": high_risk_countries},
        )
        explanation = amp_mask.explain_event(txn_row)

        st.write(f"**QID event weight for this transaction:** {explanation['weight']:.3f}")

        st.markdown("**Why QID boosted this transaction under the current mask:**")
        for reason in explanation["reasons"]:
            st.markdown(f"- {reason}")

        # QID Signature for this account
        st.markdown("### QID Signature (Magnitude per Component)")

        encoder: SecurityInterferenceEncoder = st.session_state["encoder"]
        sig = encoder.encode_sequence(acc_events, amp_mask)

        # Only show the FFT portion (0..signature_dim-1)
        fft_len = encoder.signature_dim  # 64
        sig_fft = sig[:fft_len]
        sig_df = pd.DataFrame(
            {
                "component": np.arange(fft_len),
                "value": sig_fft,
            }
        )
        st.bar_chart(sig_df.set_index("component"))

        st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# View: QID vs Baseline Rules
# -----------------------------------------------------------------------------
elif view_mode == "QID vs Baseline Rules":
    if score_df is None or score_df.empty:
        blue_banner("Run QID Scoring from the sidebar to compare QID vs Baseline Rules.")
    else:
        # Section: Scatter + Precision@K (grey card)
        st.markdown('<div class="us-section us-section-grey">', unsafe_allow_html=True)

        st.subheader("QID vs Baseline Rules Scoring")

        st.markdown(
            """
Here we compare:
- **QID score**: energy of the masked interference signature.
- **Baseline rules score**: a simple rules-based risk score (amount, corridor, new device, etc.).

You can see how often QID and baseline rules **agree** and where they **disagree**.
"""
        )

        plot_df = score_df.copy()
        plot_df["label_fraud_str"] = plot_df["label_fraud"].astype(str)

        # Scatter: QID vs Baseline Rules
        st.markdown("#### Scatter: QID Score vs Baseline Rules Score")
        st.caption(
            "Top-right: both methods think it's risky. "
            "Top-left: QID finds complex pattern risk that rules miss. "
            "Bottom-right: rules scream but QID sees a benign pattern."
        )

        st.scatter_chart(
            plot_df,
            x="rules_score",
            y="qid_score",
            color="label_fraud_str",
        )

        # Precision@K
        st.markdown("#### Precision@K: QID vs Baseline Rules")

        k_compare = st.slider(
            "K (number of accounts in top list for comparison)",
            10,
            min(len(score_df), 200),
            50,
            step=10,
            key="k_compare",
        )

        top_qid = score_df.nlargest(k_compare, "qid_score")
        top_rules = score_df.nlargest(k_compare, "rules_score")

        qid_hits = int(top_qid["label_fraud"].sum())
        rules_hits = int(top_rules["label_fraud"].sum())

        qid_prec = qid_hits / k_compare
        rules_prec = rules_hits / k_compare

        summary_df = pd.DataFrame(
            {
                "Method": ["QID", "Baseline Rules"],
                "Top K": [k_compare, k_compare],
                "Fraud Accounts in Top K": [qid_hits, rules_hits],
                "Precision@K": [qid_prec, rules_prec],
            }
        )

        st.dataframe(summary_df)

        st.markdown('</div>', unsafe_allow_html=True)

        # Section: Agreement + Disagreement (white card)
        st.markdown('<div class="us-section us-section-white">', unsafe_allow_html=True)

        # Agreement matrix (2x2)
        st.markdown("#### Agreement Matrix (QID High/Low × Baseline High/Low)")

        # Thresholds: upper 80th percentile = "High"
        qid_high_thresh = score_df["qid_score"].quantile(0.8)
        rules_high_thresh = score_df["rules_score"].quantile(0.8)

        qid_high = score_df["qid_score"] >= qid_high_thresh
        qid_low = ~qid_high
        rules_high = score_df["rules_score"] >= rules_high_thresh
        rules_low = ~rules_high

        def cell(mask):
            n = int(mask.sum())
            if n == 0:
                return "0 (0.00% fraud)"
            fraud = int(score_df.loc[mask, "label_fraud"].sum())
            rate = fraud / n
            return f"{n} ({rate:.2%} fraud)"

        mat_df = pd.DataFrame(
            {
                "QID Low": [
                    cell(rules_low & qid_low),
                    cell(rules_high & qid_low),
                ],
                "QID High": [
                    cell(rules_low & qid_high),
                    cell(rules_high & qid_high),
                ],
            },
            index=["Baseline Low", "Baseline High"],
        )

        st.table(mat_df)

        st.caption(
            """
- **Bottom-right (Baseline High, QID High)**: obvious bad actors — both methods agree.
- **Top-right (Baseline Low, QID High)**: **QID-only finds** — pattern anomalies that rules miss.
- **Bottom-left (Baseline High, QID Low)**: likely **rules false positives** — QID down-ranks these.
- **Top-left (Baseline Low, QID Low)**: benign population — both methods comfortable.
"""
        )

        # Disagreement Examples
        st.markdown("#### Disagreement Examples")

        # Use median as "low" for disagreement filters to get a richer sample
        rules_low_thresh = score_df["rules_score"].quantile(0.5)
        qid_low_thresh = score_df["qid_score"].quantile(0.5)

        qid_high_rules_low = score_df[
            (score_df["qid_score"] >= qid_high_thresh)
            & (score_df["rules_score"] <= rules_low_thresh)
        ].head(20)

        rules_high_qid_low = score_df[
            (score_df["rules_score"] >= rules_high_thresh)
            & (score_df["qid_score"] <= qid_low_thresh)
        ].head(20)

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("**QID-high / Baseline-low (QID-only finds)**")
            st.caption(
                """
Accounts where:
- QID score is high (top 20%),
- Baseline rules score is low (bottom 50%).

Interpretation:
- These are **pattern-anomaly accounts** that do not trigger naive thresholds.
- On real data, this is where you expect early mules, subtle structuring,
  synthetic IDs, or low-and-slow fraud precursors to show up.
"""
            )
            st.dataframe(qid_high_rules_low)

        with col_b:
            st.markdown("**Baseline-high / QID-low (likely false positives)**")
            st.caption(
                """
Accounts where:
- Baseline rules score is high (top 20%),
- QID score is low (bottom 50%).

Interpretation:
- These are classic **rule-noise cases**: large amounts, cross-border, or new device,
  but the overall behaviour pattern is benign.
- QID can be used to **down-rank** these to reduce analyst workload without
  sacrificing true fraud coverage.
"""
            )
            st.dataframe(rules_high_qid_low)

        st.markdown('</div>', unsafe_allow_html=True)
