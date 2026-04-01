import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils import TARGET_META, TARGETS, FEATURE_COLS, generate_data, encode_df

# ---------------- SAFE COLOR FUNCTION (FIX) ----------------

def safe_fill_color(color, alpha=0.04):
    """
    Convert hex or rgb color to valid Plotly rgba format.
    Prevents deployment errors with fillcolor.
    """

    # rgb -> rgba
    if isinstance(color, str) and color.startswith("rgb("):
        return color.replace("rgb(", "rgba(").replace(")", f",{alpha})")

    # rgba already
    if isinstance(color, str) and color.startswith("rgba("):
        return color

    # hex -> rgba
    if isinstance(color, str) and color.startswith("#") and len(color) == 7:
        try:
            r = int(color[1:3], 16)
            g = int(color[3:5], 16)
            b = int(color[5:7], 16)
            return f"rgba({r},{g},{b},{alpha})"
        except Exception:
            pass

    # fallback
    return f"rgba(0,0,0,{alpha})"


PT = dict(
    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#7c8fad', family='Plus Jakarta Sans'),
    xaxis=dict(gridcolor='rgba(255,255,255,0.04)', zeroline=False, tickfont=dict(size=10)),
    yaxis=dict(gridcolor='rgba(255,255,255,0.04)', zeroline=False, tickfont=dict(size=10)),
    margin=dict(l=0, r=0, t=10, b=0),
    legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(size=10)),
)

TARGET_COLORS = {
    'Attrition':'#ff6b6b',
    'PerformanceRating':'#f59e0b',
    'AbsentDays':'#00a8ff',
    'PromotionLikelihood':'#00d4a1',
}


def show():
    st.markdown("""
    <div class="page-title">
      <div class="section-label">Explainability</div>
      <h1>Model Insights</h1>
      <p>Feature importances · error analysis · sensitivity sweeps</p>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.models_trained:
        st.markdown("""
        <div style="padding:1.2rem 1.5rem;background:rgba(245,158,11,0.08);
                    border:1px solid rgba(245,158,11,0.2);border-radius:10px;
                    font-size:0.85rem;color:#f59e0b;font-family:'IBM Plex Mono',monospace;">
          ⚠  Train models first on the Model Training page
        </div>""", unsafe_allow_html=True)
        return

    results = st.session_state.results
    scaler  = st.session_state.scaler

    # ---------------- Heatmap ----------------

    st.markdown('<div class="chart-card"><div class="chart-title">Feature Importance Heatmap — Random Forest (all targets)</div>', unsafe_allow_html=True)

    imp_data = {}

    for target, tres in results.items():
        rf = tres.get('Random Forest')

        if rf and hasattr(rf['model'], 'feature_importances_'):
            imp_data[target] = rf['model'].feature_importances_

    if imp_data:
        imp_df = pd.DataFrame(imp_data, index=FEATURE_COLS)

        fig_heat = px.imshow(
            imp_df.T,
            text_auto='.3f',
            aspect='auto',
            color_continuous_scale=[
                [0,'#0c1220'],
                [0.5,'#1a3a4a'],
                [1,'#00d4a1']
            ],
            labels=dict(
                x='Feature',
                y='Target',
                color='Importance'
            )
        )

        fig_heat.update_layout(
            **PT,
            height=280,
            coloraxis_colorbar=dict(
                tickfont=dict(size=9, color='#4a5a72'),
                title=dict(font=dict(size=9, color='#4a5a72'))
            )
        )

        fig_heat.update_xaxes(tickangle=-35)

        st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ---------------- Target selection ----------------

    sel_col1, sel_col2 = st.columns(2)

    with sel_col1:
        target_sel = st.selectbox(
            "Prediction target",
            list(TARGETS.keys())
        )

    with sel_col2:
        model_sel = st.selectbox(
            "Model",
            list(results[target_sel].keys())
        )

    mres  = results[target_sel][model_sel]
    task  = mres['task']
    color = TARGET_COLORS[target_sel]

    col1, col2 = st.columns([1,1], gap="medium")

    # ---------------- Feature importance ----------------

    with col1:
        st.markdown(f'<div class="chart-card"><div class="chart-title">Top Feature Importances — {target_sel}</div>', unsafe_allow_html=True)

        model = mres['model']

        if hasattr(model, 'feature_importances_'):

            imp = pd.Series(
                model.feature_importances_,
                index=FEATURE_COLS
            ).sort_values(ascending=True)

            fig = px.bar(
                imp.tail(12),
                orientation='h',
                color=imp.tail(12).values,
                color_continuous_scale=[
                    [0,'#1a2540'],
                    [1,color]
                ]
            )

            fig.update_traces(marker_line_width=0)

            fig.update_layout(
                **PT,
                height=370,
                coloraxis_showscale=False
            )

            st.plotly_chart(fig, use_container_width=True)

        elif hasattr(model, 'coef_'):

            coefs = model.coef_

            coef_vals = (
                np.abs(coefs).mean(axis=0)
                if coefs.ndim > 1
                else np.abs(coefs)
            )

            imp = pd.Series(
                coef_vals,
                index=FEATURE_COLS
            ).sort_values(ascending=True)

            fig = px.bar(
                imp.tail(12),
                orientation='h',
                color=imp.tail(12).values,
                color_continuous_scale=[
                    [0,'#1a2540'],
                    [1,'#a855f7']
                ]
            )

            fig.update_traces(marker_line_width=0)

            fig.update_layout(
                **PT,
                height=370,
                coloraxis_showscale=False
            )

            st.plotly_chart(fig, use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # ---------------- Error analysis ----------------

    with col2:
        st.markdown('<div class="chart-card"><div class="chart-title">Prediction Error Analysis</div>', unsafe_allow_html=True)

        y_te   = mres['y_te'].values
        y_pred = mres['y_pred']

        if task == 'reg':

            residuals = y_te - y_pred

            fig_res = px.scatter(
                x=y_pred,
                y=residuals,
                opacity=0.55,
                color=np.abs(residuals),
                color_continuous_scale=[
                    [0,'#10b981'],
                    [0.5,'#f59e0b'],
                    [1,'#ff6b6b']
                ],
                labels={
                    'x':'Predicted',
                    'y':'Residuals'
                }
            )

            fig_res.add_hline(
                y=0,
                line_dash='dot',
                line_color='rgba(255,255,255,0.15)'
            )

            fig_res.update_traces(
                marker=dict(
                    size=4,
                    line=dict(width=0)
                )
            )

            fig_res.update_layout(
                **PT,
                height=320,
                coloraxis_showscale=False
            )

            st.plotly_chart(fig_res, use_container_width=True)

            mc1, mc2 = st.columns(2)

            mc1.metric(
                "MAE",
                f"{np.abs(residuals).mean():.2f} days"
            )

            mc2.metric(
                "R²",
                f"{mres['r2']:.3f}"
            )

        else:

            classes = sorted(np.unique(y_te))

            class_acc = [
                {
                    'Class': str(c),
                    'Accuracy': (y_pred[y_te==c]==c).mean(),
                    'Count': int((y_te==c).sum())
                }
                for c in classes
                if (y_te==c).sum() > 0
            ]

            if class_acc:

                df_ca = pd.DataFrame(class_acc)

                fig_ca = px.bar(
                    df_ca,
                    x='Class',
                    y='Accuracy',
                    color='Accuracy',
                    text=df_ca['Accuracy'].round(2),
                    color_continuous_scale=[
                        [0,'#1a2540'],
                        [1,color]
                    ]
                )

                fig_ca.update_traces(
                    textposition='outside',
                    textfont_size=10,
                    marker_line_width=0
                )

                fig_ca.update_layout(
                    **PT,
                    height=300,
                    yaxis_range=[0,1.15],
                    coloraxis_showscale=False
                )

                st.plotly_chart(fig_ca, use_container_width=True)

            mc1, mc2 = st.columns(2)

            mc1.metric(
                "Accuracy",
                f"{mres['accuracy']:.3f}"
            )

            if mres.get('auc'):

                mc2.metric(
                    "AUC-ROC",
                    f"{mres['auc']:.3f}"
                )

        st.markdown('</div>', unsafe_allow_html=True)

    # ---------------- Sensitivity sweep ----------------

    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-bottom:1rem;">
      <div class="section-label">Sensitivity Analysis</div>
      <p style="font-size:0.82rem;color:#4a5a72;margin:0;">
        Sweep one feature across its full range and observe how all model predictions respond
      </p>
    </div>
    """, unsafe_allow_html=True)

    sweep_feat = st.selectbox(
        "Sweep feature",
        FEATURE_COLS,
        index=6
    )

    if sweep_feat in ['Gender','Department','Role','Education']:

        st.markdown(
            '<div class="info-box">Select a numeric feature for the sensitivity sweep.</div>',
            unsafe_allow_html=True
        )

    else:

        df_base = generate_data(100)

        df_enc, _ = encode_df(df_base)

        X_base = pd.DataFrame(
            scaler.transform(df_enc[FEATURE_COLS]),
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

                m = results[target]['Random Forest']['model']

                if task_t == 'clf' and hasattr(m,'predict_proba'):

                    p = m.predict_proba(X_rsc)

                    sweep_res[target].append(
                        p[:,1].mean()
                        if p.shape[1] == 2
                        else m.predict(X_rsc).mean()
                    )

                else:

                    sweep_res[target].append(
                        m.predict(X_rsc).mean()
                    )

        fig_sw = go.Figure()

        for target, vals in sweep_res.items():

            c = TARGET_COLORS[target]

            fig_sw.add_trace(
                go.Scatter(
                    x=sweep_vals,
                    y=vals,
                    mode='lines',
                    name=TARGET_META[target]['label'],
                    line=dict(
                        color=c,
                        width=2.5
                    ),
                    fill='tozeroy',
                    fillcolor=safe_fill_color(c)  # FIXED LINE
                )
            )

        fig_sw.update_layout(
            **PT,
            height=360,
            xaxis_title=sweep_feat,
            yaxis_title='Predicted Value / Probability',
            legend=dict(
                orientation='h',
                y=-0.2
            )
        )

        st.plotly_chart(
            fig_sw,
            use_container_width=True
        )
