import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import roc_curve
from utils import train_all_models, TARGETS, TARGET_META

# ---------------- BASE PLOT THEME ----------------

PT = dict(
    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#7c8fad', family='Plus Jakarta Sans'),
    xaxis=dict(gridcolor='rgba(255,255,255,0.04)', zeroline=False, tickfont=dict(size=10)),
    yaxis=dict(gridcolor='rgba(255,255,255,0.04)', zeroline=False, tickfont=dict(size=10)),
    margin=dict(l=0, r=0, t=10, b=0),
    legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(size=10)),
)

TARGET_COLORS = {
    'Attrition':           '#ff6b6b',
    'PerformanceRating':   '#f59e0b',
    'AbsentDays':          '#00a8ff',
    'PromotionLikelihood': '#00d4a1',
}


def show():
    st.markdown("""
    <div class="page-title">
      <div class="section-label">Machine Learning</div>
      <h1>Model Training</h1>
      <p>Train 8 models across 4 targets · results cached after first run</p>
    </div>
    """, unsafe_allow_html=True)

    # ---------------- CONFIG ----------------

    with st.container():
        st.markdown('<div class="chart-card" style="padding-bottom:1.2rem;">', unsafe_allow_html=True)

        col_a, col_b = st.columns([3, 1], gap="medium")

        with col_a:
            n_samples = st.slider("Training samples", 500, 3000, 1500, 100)
            seed      = st.number_input("Random seed", 0, 999, 42)

        with col_b:
            st.markdown("<br><br>", unsafe_allow_html=True)
            train_btn = st.button("▶  Train All Models", use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # ---------------- TRAIN ----------------

    if train_btn or st.session_state.models_trained:

        with st.spinner("Training models… (cached after first run)"):

            results, scaler, encoders, df = train_all_models(n_samples, seed)

            st.session_state.results        = results
            st.session_state.scaler         = scaler
            st.session_state.encoders       = encoders
            st.session_state.df             = df
            st.session_state.models_trained = True

        # ---------------- CONFUSION MATRICES ----------------

        tab1, tab2, tab3 = st.tabs([
            "📊  Model Comparison",
            "📈  ROC Curves",
            "🧩  Confusion Matrices"
        ])

        # ---------------- ROC CURVES ----------------

        with tab2:

            binary_t = {
                k:v for k,v in results.items()
                if TARGETS[k]=='clf'
                and len(np.unique(list(v.values())[0]['y_te']))==2
            }

            if binary_t:

                fig_roc = go.Figure()

                pal = ['#00d4a1','#00a8ff','#ff6b6b','#f59e0b']

                for (target, tres), color in zip(binary_t.items(), pal):

                    for mname, mres in tres.items():

                        if hasattr(mres['model'], 'predict_proba'):

                            fpr, tpr, _ = roc_curve(
                                mres['y_te'],
                                mres['model'].predict_proba(
                                    mres['X_te']
                                )[:,1]
                            )

                            fig_roc.add_trace(
                                go.Scatter(
                                    x=fpr,
                                    y=tpr,
                                    mode='lines',
                                    name=f"{target} · {mname} ({mres.get('auc',0):.2f})",
                                    line=dict(width=2, color=color),
                                    opacity=0.85
                                )
                            )

                fig_roc.add_trace(
                    go.Scatter(
                        x=[0,1],
                        y=[0,1],
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

        # ---------------- CONFUSION MATRICES (FIXED) ----------------

        with tab3:

            clf_t = {
                k:v for k,v in results.items()
                if TARGETS[k]=='clf'
            }

            items = list(clf_t.items())

            for i in range(0, len(items), 2):

                row_cols = st.columns(2, gap="medium")

                for col, (target, tres) in zip(row_cols, items[i:i+2]):

                    best_m = max(
                        tres,
                        key=lambda m: tres[m]['cv_accuracy']
                    )

                    cm = tres[best_m]['cm']

                    fig_cm = px.imshow(
                        cm,
                        text_auto=True,
                        color_continuous_scale=[
                            [0,'#0c1220'],
                            [1,TARGET_COLORS[target]]
                        ],
                        labels=dict(
                            x='Predicted',
                            y='Actual'
                        ),
                        title=f"{target} — {best_m}"
                    )

                    # -------- FIX: avoid duplicate margin argument --------

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
