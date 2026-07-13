"""
ABSA Review Tool
----------------
Streamlit app for manually reviewing ABSA model predictions.

Run:
    pip install streamlit pandas
    streamlit run absa_review_app.py

Logic:
- Upload a CSV in the app.
- Each row = one review (aspect is already one per review).
- See the review text, all predicted features (aspect, polarity,
  confidence, proba_*) as a big probability-distribution chart, the
  model's final decision, and everything the LLM added (llm_reason,
  final_aspect, final_polarity, corrected_aspect, corrected_polarity,
  comment).
- Click "Correct" -> human_checked=True, keep=1.
- Click "False"   -> opens a form to enter the correct aspect/polarity,
  then sets human_checked=False, keep=0.
- Download the updated CSV at any time.
"""

import streamlit as st
import pandas as pd

st.set_page_config(page_title="ABSA Review Tool", layout="wide")

# ----------------------------------------------------------------------
# Columns with known semantic roles. Everything else in the file is still
# shown (in the "Other fields" expander) so nothing is ever hidden.
# ----------------------------------------------------------------------
TEXT_COL = "text"
PRED_COLS_KNOWN = ["aspect", "polarity", "proba_predicted_label", "confidence"]
LLM_COLS_KNOWN = [
    "llm_reason", "final_aspect", "final_polarity",
    "corrected_aspect", "corrected_polarity", "comment",
]
STATUS_COLS = ["human_checked", "keep"]

POLARITY_OPTIONS = ["positive", "negative", "neutral", "conflict"]

POLARITY_COLORS = {
    "positive": "#2ecc71",
    "negative": "#e74c3c",
    "neutral": "#95a5a6",
    "conflict": "#f39c12",
}


def get_proba_cols(df):
    return [c for c in df.columns if c.startswith("proba_") and c != "proba_predicted_label"]


def polarity_color(label):
    label = str(label).lower().strip()
    return POLARITY_COLORS.get(label, "#3498db")


def init_state():
    if "df" not in st.session_state:
        st.session_state.df = None
    if "pos" not in st.session_state:
        st.session_state.pos = 0
    if "filename" not in st.session_state:
        st.session_state.filename = None
    if "correcting" not in st.session_state:
        st.session_state.correcting = False


init_state()

st.title("ABSA Review Tool")

# ----------------------------------------------------------------------
# File upload
# ----------------------------------------------------------------------
uploaded = st.file_uploader("Upload your labeled CSV", type=["csv"])

if uploaded is not None and st.session_state.filename != uploaded.name:
    df = pd.read_csv(uploaded)

    if "human_checked" not in df.columns:
        df["human_checked"] = False
    if "keep" not in df.columns:
        df["keep"] = pd.NA

    st.session_state.df = df
    st.session_state.filename = uploaded.name
    st.session_state.pos = 0
    st.session_state.correcting = False

if st.session_state.df is None:
    st.info("Upload a CSV to start reviewing.")
    st.stop()

df = st.session_state.df
n_total = len(df)

# ----------------------------------------------------------------------
# Sidebar: filter + progress + download
# ----------------------------------------------------------------------
with st.sidebar:
    st.header("Navigation")

    mode = st.radio(
        "Show",
        ["All reviews", "Unreviewed only", "Marked False only"],
    )

    if mode == "Unreviewed only":
        mask = ~df["human_checked"].isin([True, False])
    elif mode == "Marked False only":
        mask = df["human_checked"] == False
    else:
        mask = pd.Series([True] * n_total, index=df.index)

    indices = list(df.index[mask])

    n_checked = int((df["human_checked"] == True).sum())
    n_false = int((df["human_checked"] == False).sum())
    st.progress(n_checked / n_total if n_total else 0)
    st.caption(f"Reviewed: {n_checked}/{n_total} | Marked False: {n_false}")

    st.divider()
    csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "Download updated CSV",
        data=csv_bytes,
        file_name=f"reviewed_{st.session_state.filename}",
        mime="text/csv",
        use_container_width=True,
    )

if not indices:
    st.success("No reviews left in this category 🎉")
    st.stop()

if st.session_state.pos >= len(indices):
    st.session_state.pos = 0
if st.session_state.pos < 0:
    st.session_state.pos = len(indices) - 1

current_idx = indices[st.session_state.pos]
row = df.loc[current_idx]

# ----------------------------------------------------------------------
# Top nav bar
# ----------------------------------------------------------------------
nav_l, nav_c, nav_r = st.columns([1, 2, 1])
with nav_l:
    if st.button("⬅ Back", use_container_width=True, disabled=st.session_state.pos == 0):
        st.session_state.pos -= 1
        st.session_state.correcting = False
        st.rerun()
with nav_c:
    st.markdown(
        f"<div style='text-align:center'>Review {st.session_state.pos + 1} of {len(indices)} "
        f"(row #{current_idx})</div>",
        unsafe_allow_html=True,
    )
with nav_r:
    if st.button("Next ➡", use_container_width=True, disabled=st.session_state.pos == len(indices) - 1):
        st.session_state.pos += 1
        st.session_state.correcting = False
        st.rerun()

st.divider()

# ----------------------------------------------------------------------
# Review text
# ----------------------------------------------------------------------
st.subheader("Review")
review_text = row[TEXT_COL] if TEXT_COL in df.columns else "('text' column not found)"
st.markdown(
    f"<div style='background:#1e1e1e;color:#eee;padding:20px;border-radius:8px;"
    f"font-size:16px;line-height:1.5'>{review_text}</div>",
    unsafe_allow_html=True,
)

st.write("")

# ----------------------------------------------------------------------
# Big probability distribution + final decision box
# ----------------------------------------------------------------------
proba_cols = get_proba_cols(df)

# Final decision: prefer a human correction, then the LLM's final call,
# then the raw model prediction.
final_decision = (
    row.get("corrected_polarity") or row.get("final_polarity") or row.get("polarity")
)
final_decision_label = str(final_decision) if final_decision is not None else "N/A"
decision_color = polarity_color(final_decision_label)

dist_col, decision_col = st.columns([2, 1])

with dist_col:
    st.subheader("Probability Distribution")
    if proba_cols:
        max_val = max([float(row[c]) for c in proba_cols if pd.notna(row[c])], default=0)
        for c in proba_cols:
            val = row[c]
            if pd.isna(val):
                continue
            val = float(val)
            label = c.replace("proba_", "").capitalize()
            is_max = val == max_val
            bar_color = polarity_color(label) if is_max else "#444"
            st.markdown(
                f"""
                <div style='margin-bottom:10px'>
                  <div style='display:flex;justify-content:space-between;font-size:14px;
                       font-weight:{"700" if is_max else "400"};color:{"#fff" if is_max else "#aaa"}'>
                    <span>{label}{" ⭐" if is_max else ""}</span>
                    <span>{val*100:.1f}%</span>
                  </div>
                  <div style='background:#2a2a2a;border-radius:6px;height:16px;width:100%;overflow:hidden'>
                    <div style='background:{bar_color};height:100%;width:{val*100:.1f}%'></div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.info("No proba_* columns found in this file.")

with decision_col:
    st.subheader("Final Decision")
    st.markdown(
        f"""
        <div style='background:{decision_color}22;border:2px solid {decision_color};
             border-radius:12px;padding:24px;text-align:center;height:100%'>
          <div style='font-size:14px;color:#aaa;margin-bottom:6px'>ASPECT</div>
          <div style='font-size:20px;font-weight:600;color:#fff;margin-bottom:16px'>
            {row.get('final_aspect') or row.get('corrected_aspect') or row.get('aspect') or 'N/A'}
          </div>
          <div style='font-size:14px;color:#aaa;margin-bottom:6px'>POLARITY</div>
          <div style='font-size:32px;font-weight:800;color:{decision_color}'>
            {final_decision_label.upper()}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")

# ----------------------------------------------------------------------
# Model prediction / LLM correction detail tables
# ----------------------------------------------------------------------
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("Model Prediction")
    pred_cols = [c for c in PRED_COLS_KNOWN if c in df.columns]
    pred_df = pd.DataFrame({"field": pred_cols, "value": [row[c] for c in pred_cols]})
    st.dataframe(pred_df, hide_index=True, use_container_width=True)

with col_b:
    st.subheader("LLM Correction")
    llm_cols = [c for c in LLM_COLS_KNOWN if c in df.columns]
    llm_df = pd.DataFrame({"field": llm_cols, "value": [row[c] for c in llm_cols]})
    st.dataframe(llm_df, hide_index=True, use_container_width=True)

# ----------------------------------------------------------------------
# Everything else, so no column is ever hidden
# ----------------------------------------------------------------------
other_cols = [
    c for c in df.columns
    if c not in [TEXT_COL] + PRED_COLS_KNOWN + LLM_COLS_KNOWN + STATUS_COLS + proba_cols
]
if other_cols:
    with st.expander("Other fields"):
        other_df = pd.DataFrame({"field": other_cols, "value": [row[c] for c in other_cols]})
        st.dataframe(other_df, hide_index=True, use_container_width=True)

# Current status indicator
hc = row.get("human_checked")
if hc is True:
    st.success("Current status: confirmed (Correct)")
elif hc is False:
    st.error("Current status: marked as incorrect (False)")
else:
    st.warning("Current status: not yet reviewed")

st.write("")

# ----------------------------------------------------------------------
# Correct / False buttons
# ----------------------------------------------------------------------
if not st.session_state.correcting:
    btn_false, btn_correct = st.columns(2)

    with btn_false:
        if st.button("❌ False (incorrect)", use_container_width=True, type="secondary"):
            st.session_state.correcting = True
            st.rerun()

    with btn_correct:
        if st.button("✅ Correct", use_container_width=True, type="primary"):
            df.at[current_idx, "human_checked"] = True
            df.at[current_idx, "keep"] = 1
            st.session_state.df = df
            if st.session_state.pos < len(indices) - 1:
                st.session_state.pos += 1
            st.rerun()

else:
    st.subheader("Enter the correct label")

    default_aspect = (
        row.get("corrected_aspect") or row.get("final_aspect") or row.get("aspect") or ""
    )
    default_polarity = (
        row.get("corrected_polarity") or row.get("final_polarity") or row.get("polarity") or "negative"
    )
    default_polarity = str(default_polarity).lower().strip()
    if default_polarity not in POLARITY_OPTIONS:
        default_polarity = "negative"

    with st.form("correction_form"):
        new_aspect = st.text_input("Correct aspect", value=str(default_aspect))
        new_polarity = st.selectbox(
            "Correct polarity",
            POLARITY_OPTIONS,
            index=POLARITY_OPTIONS.index(default_polarity),
        )
        note = st.text_input("Comment (optional)", value=str(row.get("comment") or ""))

        save_col, cancel_col = st.columns(2)
        submitted = save_col.form_submit_button("💾 Save correction", use_container_width=True, type="primary")
        cancelled = cancel_col.form_submit_button("Cancel", use_container_width=True)

    if submitted:
        df.at[current_idx, "human_checked"] = False
        df.at[current_idx, "keep"] = 0
        if "corrected_aspect" in df.columns:
            df.at[current_idx, "corrected_aspect"] = new_aspect
        if "corrected_polarity" in df.columns:
            df.at[current_idx, "corrected_polarity"] = new_polarity
        if "comment" in df.columns:
            df.at[current_idx, "comment"] = note
        st.session_state.df = df
        st.session_state.correcting = False
        if st.session_state.pos < len(indices) - 1:
            st.session_state.pos += 1
        st.rerun()

    if cancelled:
        st.session_state.correcting = False
        st.rerun()