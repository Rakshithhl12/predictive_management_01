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

# =====================================
# SAFE COLOR
# =====================================

def safe_fill_color(color, alpha=0.05):

    if color.startswith("rgb("):
        return color.replace("rgb(", "rgba(").replace(")", f",{alpha})")

    if color.startswith("rgba("):
        return color

    if color.startswith("#"):
        r = int(color[1:3], 16)
        g = int(color[3:5], 16)
        b = int(color[5:7], 16)
        return f"rgba({r},{g},{b},{alpha})"

    return f"rgba(0,0,0,{alpha})"


# =====================================
# THEME
# =====================================

PT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#7c8fad"),
    xaxis=dict(gridcolor="rgba(255,255,255,0.04)", zeroline=False),
    yaxis=dict(gridcolor="rgba(255,255,255,0.04)", zeroline=False),
    margin=dict(l=0, r=0, t=10, b=0),
    legend=dict(bgcolor="rgba(0,0,0,0)")
)

TARGET_COLORS = {
    "Attrition": "#ff6b6b",
    "PerformanceRating": "#f59e0b",
    "AbsentDays": "#00a8ff",
    "PromotionLikelihood": "#00d4a1",
}


# =====================================
# SAFE LAYOUT
# =====================================

def apply_layout(fig, top_margin=10, legend_override=None):

    layout = PT.copy()

    layout.pop("margin", None)

    if legend_override:
        layout.pop("legend", None)

    fig.update_layout(**layout)

    fig.update_layout(
        margin=dict(l=0, r=0, t=top_margin, b=0)
    )

    if legend_override:
        fig.update_layout(legend=legend_override)


# =====================================
# MAIN PAGE
# =====================================

def show():

    if not st.session_state.get("models_trained", False):
        st.warning("Train models first on Model Training page")
        return

    results = st.session_state.results
    scaler = st.session_state.scaler

    # =====================================
    # HEATMAP
    # =====================================

    st.subheader("Feature Importance Heatmap")

    imp_data = {}

    for target, tres in results.items():

        rf = tres.get("Random Forest")

        if rf and hasattr(rf["model"], "feature_importances_"):

            imp_data[target] = rf["model"].feature_importances_

    if imp_data:

        imp_df = pd.DataFrame(
            imp_data,
            index=FEATURE_COLS
        )

        fig_heat = px.imshow(
            imp_df.T,
            text_auto=".2f",
            aspect="auto"
        )

        apply_layout(fig_heat)

        fig_heat.update_layout(height=280)

        st.plotly_chart(fig_heat, width="stretch")

    # =====================================
    # SELECTORS
    # =====================================

    col1, col2 = st.columns(2)

    with col1:

        target_sel = st.selectbox(
            "Prediction Target",
            list(TARGETS.keys())
        )

    with col2:

        model_sel = st.selectbox(
            "Model",
            list(results[target_sel].keys())
        )

    # =====================================
    # FEATURE SELECTION
    # =====================================

    st.subheader("Feature Sensitivity Analysis")

    sweep_feat = st.selectbox(
        "Select Feature",
        FEATURE_COLS
    )

    # Always start from ORIGINAL data
    df_base = generate_data(100)

    # Detect categorical
    is_categorical = (
        df_base[sweep_feat].dtype == "object"
    )

    if is_categorical:

        sweep_vals = (
            df_base[sweep_feat]
            .dropna()
            .unique()
            .tolist()
        )

    else:

        sweep_vals = np.linspace(
            df_base[sweep_feat].min(),
            df_base[sweep_feat].max(),
            35
        )

    sweep_res = {t: [] for t in TARGETS}

    # =====================================
    # RUN SWEEP
    # =====================================

    for val in sweep_vals:

        # COPY ORIGINAL DATA
        df_temp = df_base.copy()

        # APPLY CHANGE
        df_temp[sweep_feat] = val

        # ENCODE
        df_encoded, _ = encode_df(df_temp)

        # SCALE
        X_scaled = pd.DataFrame(
            scaler.transform(
                df_encoded[FEATURE_COLS]
            ),
            columns=FEATURE_COLS
        )

        # PREDICT
        for target, task in TARGETS.items():

            model = results[target]["Random Forest"]["model"]

            if task == "clf" and hasattr(model, "predict_proba"):

                prob = model.predict_proba(X_scaled)

                sweep_res[target].append(
                    prob[:, 1].mean()
                )

            else:

                pred = model.predict(X_scaled)

                sweep_res[target].append(
                    pred.mean()
                )

    # =====================================
    # PLOT
    # =====================================

    fig_sw = go.Figure()

    for target, vals in sweep_res.items():

        c = TARGET_COLORS[target]

        fig_sw.add_trace(
            go.Scatter(
                x=sweep_vals,
                y=vals,
                mode="lines",
                name=TARGET_META[target]["label"],
                line=dict(color=c, width=2.5),
                fill="tozeroy",
                fillcolor=safe_fill_color(c)
            )
        )

    apply_layout(
        fig_sw,
        legend_override=dict(
            orientation="h",
            y=-0.25
        )
    )

    fig_sw.update_layout(
        height=360,
        xaxis_title=sweep_feat,
        yaxis_title="Prediction / Probability"
    )

    st.plotly_chart(fig_sw, width="stretch")
