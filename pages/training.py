import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import roc_curve

from utils import train_all_models, TARGETS, TARGET_META


# ---------------- BASE PLOT THEME ----------------

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


# ==========================================================
# MAIN PAGE
# ==========================================================

def show():

    # ---------------- SAFE SESSION STATE ----------------

    if "models_trained" not in st.session_state:
        st.session_state.models_trained = False

    if "results" not in st.session_state:
        st.session_state.results = None

    st.markdown("""
    <div class="page-title">
      <div class="section-label">Machine Learning</div>
      <h1>Model Training</h1>
      <p>Train 8 models across 4 targets · results cached after first run</p>
    </div>
    """, unsafe_allow_html=True)

    # ---------------- CONFIG ----------------

    with st.container():

        st.markdown(
            '<div class="chart-card" style="padding-bottom:1.2rem;">',
            unsafe_allow_html=True
        )

        col_a, col_b = st.columns([3, 1], gap="medium")

        with col_a:

            n_samples = st.slider(
                "Training samples",
                min_value=500,
                max_value=3000,
                value=1500,
                step=100
            )

            seed = st.number_input(
                "Random seed",
                min_value=0,
                max_value=999,
                value=42
            )

        with col_b:

            st.markdown("<br><br>", unsafe_allow_html=True)

            train_btn = st.button(
                "▶ Train All Models",
                use_container_width=True
            )

        st.markdown('</div>', unsafe_allow_html=True)

    # ---------------- TRAIN TRIGGER ----------------

    if train_btn or st.session_state.get("models_trained", False):

        with st.spinner("Training models… (cached after first run)"):

            try:

                results, scaler, encoders, df = train_all_models(
                    n_samples,
                    seed
                )

                if results is None:
                    st.error("Training failed — no results returned.")
                    return

                st.session_state.results = results
                st.session_state.scaler = scaler
                st.session_state.encoders = encoders
                st.session_state.df = df
                st.session_state.models_trained = True

            except Exception as e:

                st.error("Training failed.")
                st.exception(e)
                return

        results = st.session_state.results

        # ---------------- TABS ----------------

        tab1, tab2, tab3 = st.tabs([
            "📊 Model Comparison",
            "📈 ROC Curves",
            "🧩 Confusion Matrices"
        ])

        # ==========================================================
        # ROC CURVES
        # ==========================================================

        with tab2:

            binary_targets = {
                k: v for k, v in results.items()
                if TARGETS[k] == 'clf'
                and len(
                    np.unique(
                        list(v.values())[0]['y_te']
                    )
                ) == 2
            }

            if binary_targets:

                fig_roc = go.Figure()

                palette = [
                    '#00d4a1',
                    '#00a8ff',
                    '#ff6b6b',
                    '#f59e0b'
                ]

                for (target, tres), color in zip(
                        binary_targets.items(),
                        palette
                ):

                    for model_name, model_res in tres.items():

                        model = model_res.get("model")

                        if hasattr(model, "predict_proba"):

                            fpr, tpr, _ = roc_curve(
                                model_res["y_te"],
                                model.predict_proba(
                                    model_res["X_te"]
                                )[:, 1]
                            )

                            fig_roc.add_trace(
                                go.Scatter(
                                    x=fpr,
                                    y=tpr,
                                    mode='lines',
                                    name=f"{target} · {model_name}",
                                    line=dict(
                                        width=2,
                                        color=color
                                    ),
                                    opacity=0.85
                                )
                            )

                fig_roc.add_trace(

                    go.Scatter(
                        x=[0, 1],
                        y=[0, 1],
                        mode='lines',
                        line=dict(
                            dash='dot',
                            color='rgba(255,255,255,0.1)'
                        ),
                        showlegend=False
                    )
                )

                fig_roc.update_layout(
                    **PT,
                    height=430,
                    xaxis_title='False Positive Rate',
                    yaxis_title='True Positive Rate'
                )

                fig_roc.update_layout(

                    legend=dict(
                        orientation='h',
                        y=-0.2
                    )
                )

                st.plotly_chart(
                    fig_roc,
                    use_container_width=True
                )

            else:

                st.info(
                    "No binary classification targets available for ROC."
                )

        # ==========================================================
        # CONFUSION MATRICES
        # ==========================================================

        with tab3:

            clf_targets = {
                k: v for k, v in results.items()
                if TARGETS[k] == 'clf'
            }

            if not clf_targets:

                st.warning("No classification targets found.")
                return

            items = list(clf_targets.items())

            for i in range(0, len(items), 2):

                row_cols = st.columns(2, gap="medium")

                for col, (target, tres) in zip(
                        row_cols,
                        items[i:i+2]
                ):

                    try:

                        best_model = max(
                            tres,
                            key=lambda m:
                            tres[m]['cv_accuracy']
                        )

                        cm = tres[best_model]['cm']

                        fig_cm = px.imshow(
                            cm,
                            text_auto=True,
                            color_continuous_scale=[
                                [0, '#0c1220'],
                                [1, TARGET_COLORS[target]]
                            ],
                            labels=dict(
                                x='Predicted',
                                y='Actual'
                            ),
                            title=f"{target} — {best_model}"
                        )

                        layout_cm = PT.copy()

                        layout_cm["margin"] = dict(
                            l=0,
                            r=0,
                            t=45,
                            b=0
                        )

                        fig_cm.update_layout(
                            **layout_cm,
                            height=270,
                            coloraxis_showscale=False,
                            title_font=dict(
                                size=11,
                                color='#7c8fad'
                            )
                        )

                        col.plotly_chart(
                            fig_cm,
                            use_container_width=True
                        )

                    except Exception as e:

                        col.error(
                            f"Failed to render {target}"
                        )
                        col.exception(e)
