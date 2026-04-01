import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from utils import (
    TARGET_META,
    TARGETS,
    FEATURE_COLS,
    generate_data,
    encode_df
)

# ---------------- SAFE COLOR FUNCTION ----------------

def safe_fill_color(color, alpha=0.04):

    if isinstance(color, str) and color.startswith("rgb("):
        return color.replace("rgb(", "rgba(").replace(")", f",{alpha})")

    if isinstance(color, str) and color.startswith("rgba("):
        return color

    if isinstance(color, str) and color.startswith("#") and len(color) == 7:

        r = int(color[1:3], 16)
        g = int(color[3:5], 16)
        b = int(color[5:7], 16)

        return f"rgba({r},{g},{b},{alpha})"

    return f"rgba(0,0,0,{alpha})"


# ---------------- BASE THEME ----------------

PT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#7c8fad', family='Plus Jakarta Sans'),
    xaxis=dict(
        gridcolor='rgba(255,255,255,0.04)',
        zeroline=False,
        tickfont=dict(size=10)
    ),
    yaxis=dict(
        gridcolor='rgba(255,255,255,0.04)',
        zeroline=False,
        tickfont=dict(size=10)
    ),
    margin=dict(l=0, r=0, t=10, b=0),
    legend=dict(
        bgcolor='rgba(0,0,0,0)',
        font=dict(size=10)
    ),
)

TARGET_COLORS = {
    'Attrition': '#ff6b6b',
    'PerformanceRating': '#f59e0b',
    'AbsentDays': '#00a8ff',
    'PromotionLikelihood': '#00d4a1',
}


# ---------------- SAFE LAYOUT HELPER ----------------

def apply_layout(fig, top_margin=10, legend_override=None):

    layout = PT.copy()

    if legend_override is not None:
        layout.pop("legend", None)

    fig.update_layout(
        **layout,
        margin=dict(
            l=0,
            r=0,
            t=top_margin,
            b=0
        )
    )

    if legend_override:

        fig.update_layout(
            legend=legend_override
        )


# ==========================================================
# MAIN PAGE
# ==========================================================

def show():

    if not st.session_state.get("models_trained", False):

        st.markdown("""
        <div style="padding:1.2rem;background:rgba(245,158,11,0.08);
                    border:1px solid rgba(245,158,11,0.2);
                    border-radius:10px;color:#f59e0b;">
          ⚠ Train models first on the Model Training page
        </div>
        """, unsafe_allow_html=True)

        return

    results = st.session_state.results
    scaler = st.session_state.scaler

    # ---------------- Heatmap ----------------

    st.markdown(
        '<div class="chart-card"><div class="chart-title">'
        'Feature Importance Heatmap — Random Forest'
        '</div>',
        unsafe_allow_html=True
    )

    imp_data = {}

    for target, tres in results.items():

        rf = tres.get("Random Forest")

        if rf and hasattr(
            rf["model"],
            "feature_importances_"
        ):

            imp_data[target] = \
                rf["model"].feature_importances_

    if imp_data:

        imp_df = pd.DataFrame(
            imp_data,
            index=FEATURE_COLS
        )

        fig_heat = px.imshow(
            imp_df.T,
            text_auto=".3f",
            aspect="auto"
        )

        apply_layout(fig_heat)

        fig_heat.update_layout(
            height=280
        )

        st.plotly_chart(
            fig_heat,
            width="stretch"
        )

    st.markdown("</div>", unsafe_allow_html=True)

    # ---------------- Target Selection ----------------

    col1, col2 = st.columns(2)

    with col1:

        target_sel = st.selectbox(
            "Prediction target",
            list(TARGETS.keys())
        )

    with col2:

        model_sel = st.selectbox(
            "Model",
            list(results[target_sel].keys())
        )

    mres = results[target_sel][model_sel]

    color = TARGET_COLORS[target_sel]

    # ---------------- Sensitivity Sweep ----------------

    st.markdown("<hr>", unsafe_allow_html=True)

    sweep_feat = st.selectbox(
        "Sweep feature",
        FEATURE_COLS,
        index=6
    )

    df_base = generate_data(100)

    df_enc, _ = encode_df(df_base)

    X_base = pd.DataFrame(
        scaler.transform(
            df_enc[FEATURE_COLS]
        ),
        columns=FEATURE_COLS
    )

    f_min = int(df_base[sweep_feat].min())
    f_max = int(df_base[sweep_feat].max())

    sweep_vals = np.linspace(
        f_min,
        f_max,
        35
    )

    sweep_res = {
        t: []
        for t in TARGETS
    }

    for val in sweep_vals:

        X_orig = pd.DataFrame(
            scaler.inverse_transform(X_base),
            columns=FEATURE_COLS
        )

        X_orig[sweep_feat] = val

        X_rsc = pd.DataFrame(
            scaler.transform(X_orig),
            columns=FEATURE_COLS
        )

        for target, task_t in TARGETS.items():

            model = results[target][
                "Random Forest"
            ]["model"]

            if task_t == "clf" and \
               hasattr(model, "predict_proba"):

                p = model.predict_proba(X_rsc)

                sweep_res[target].append(
                    p[:, 1].mean()
                )

            else:

                sweep_res[target].append(
                    model.predict(X_rsc).mean()
                )

    fig_sw = go.Figure()

    for target, vals in sweep_res.items():

        c = TARGET_COLORS[target]

        fig_sw.add_trace(

            go.Scatter(
                x=sweep_vals,
                y=vals,
                mode="lines",
                name=TARGET_META[target]["label"],
                line=dict(
                    color=c,
                    width=2.5
                ),
                fill="tozeroy",
                fillcolor=safe_fill_color(c)
            )
        )

    apply_layout(
        fig_sw,
        legend_override=dict(
            orientation="h",
            y=-0.2
        )
    )

    fig_sw.update_layout(
        height=360,
        xaxis_title=sweep_feat,
        yaxis_title="Predicted Value / Probability"
    )

    st.plotly_chart(
        fig_sw,
        width="stretch"
    )
