import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import roc_curve
from utils import train_all_models, TARGETS, TARGET_META

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

    # ── Config ────────────────────────────────────────────────────────────────
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

    if train_btn or st.session_state.models_trained:
        with st.spinner("Training models… (cached after first run)"):
            results, scaler, encoders, df = train_all_models(n_samples, seed)
            st.session_state.results        = results
            st.session_state.scaler         = scaler
            st.session_state.encoders       = encoders
            st.session_state.df             = df
            st.session_state.models_trained = True

        st.markdown("""
        <div style="display:flex;align-items:center;gap:0.5rem;margin:1rem 0;padding:0.75rem 1rem;
                    background:rgba(16,185,129,0.08);border:1px solid rgba(16,185,129,0.2);
                    border-radius:8px;font-size:0.82rem;color:#10b981;
                    font-family:'IBM Plex Mono',monospace;">
          ✓ &nbsp; All models trained and cached
        </div>""", unsafe_allow_html=True)

        # ── Summary metrics ────────────────────────────────────────────────
        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
        cols = st.columns(4)
        for col, (target, task) in zip(cols, TARGETS.items()):
            meta  = TARGET_META[target]
            tres  = results[target]
            color = TARGET_COLORS[target]
            if task == 'clf':
                best_m = max(tres, key=lambda m: tres[m]['cv_accuracy'])
                bres   = tres[best_m]
                val    = f"{bres['cv_accuracy']*100:.1f}%"
                sub    = f"CV Acc  ·  AUC {bres['auc']:.2f}" if bres.get('auc') else f"CV Acc · {best_m}"
            else:
                best_m = max(tres, key=lambda m: tres[m]['cv_r2'])
                bres   = tres[best_m]
                val    = f"{bres['cv_r2']:.3f}"
                sub    = f"CV R²  ·  MAE {bres['mae']:.1f}"
            col.markdown(f"""
            <div class="stat-card" style="border-bottom:2px solid {color}40;">
              <div class="stat-label">{meta['icon']}  {target}</div>
              <div class="stat-value" style="font-size:1.6rem;color:{color};">{val}</div>
              <div class="stat-delta neu" style="color:#4a5a72;font-size:0.68rem;">{sub}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)

        tab1, tab2, tab3 = st.tabs(["📊  Model Comparison", "📈  ROC Curves", "🧩  Confusion Matrices"])

        with tab1:
            rows = []
            for target, task in TARGETS.items():
                for mname, mres in results[target].items():
                    rows.append({
                        'Target': target, 'Model': mname,
                        'Score': mres['cv_accuracy'] if task=='clf' else max(mres['cv_r2'],0),
                        'Metric': 'CV Accuracy' if task=='clf' else 'CV R²'
                    })
            dfp = pd.DataFrame(rows)
            fig = px.bar(dfp, x='Target', y='Score', color='Model',
                         barmode='group', text=dfp['Score'].round(3),
                         color_discrete_sequence=['#00d4a1','#00a8ff','#ff6b6b','#f59e0b'])
            fig.update_traces(textposition='outside', textfont_size=9, marker_line_width=0)
            fig.update_layout(**PT, height=380, yaxis_range=[0,1.12],
                              bargap=0.25, bargroupgap=0.08)
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("📋  Full Metrics Table"):
                rows2 = []
                for target, task in TARGETS.items():
                    for mname, mres in results[target].items():
                        row = {'Target':target,'Model':mname,'Task':task.upper()}
                        if task == 'clf':
                            row.update({'Accuracy':round(mres['accuracy'],3),
                                        'CV Acc':round(mres['cv_accuracy'],3),
                                        'AUC':round(mres['auc'],3) if mres.get('auc') else '-'})
                        else:
                            row.update({'MAE':round(mres['mae'],2),
                                        'R²':round(mres['r2'],3),
                                        'CV R²':round(mres['cv_r2'],3)})
                        rows2.append(row)
                st.dataframe(pd.DataFrame(rows2), use_container_width=True)

        with tab2:
            binary_t = {k:v for k,v in results.items()
                        if TARGETS[k]=='clf' and len(np.unique(list(v.values())[0]['y_te']))==2}
            if binary_t:
                fig_roc = go.Figure()
                pal = ['#00d4a1','#00a8ff','#ff6b6b','#f59e0b']
                for (target, tres), color in zip(binary_t.items(), pal):
                    for mname, mres in tres.items():
                        if hasattr(mres['model'], 'predict_proba'):
                            fpr, tpr, _ = roc_curve(
                                mres['y_te'],
                                mres['model'].predict_proba(mres['X_te'])[:,1])
                            fig_roc.add_trace(go.Scatter(
                                x=fpr, y=tpr, mode='lines',
                                name=f"{target} · {mname} ({mres.get('auc',0):.2f})",
                                line=dict(width=2, color=color),
                                opacity=0.85
                            ))
                fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines',
                    line=dict(dash='dot', color='rgba(255,255,255,0.1)'), showlegend=False))
                # FIX: unpack PT first, then set legend separately to avoid
                # "multiple values for keyword argument 'legend'" error
                fig_roc.update_layout(**PT, height=430,
                                      xaxis_title='False Positive Rate',
                                      yaxis_title='True Positive Rate')
                fig_roc.update_layout(legend=dict(orientation='h', y=-0.2))
                st.plotly_chart(fig_roc, use_container_width=True)

        with tab3:
            clf_t = {k:v for k,v in results.items() if TARGETS[k]=='clf'}
            items  = list(clf_t.items())
            for i in range(0, len(items), 2):
                row_cols = st.columns(2, gap="medium")
                for col, (target, tres) in zip(row_cols, items[i:i+2]):
                    best_m = max(tres, key=lambda m: tres[m]['cv_accuracy'])
                    cm     = tres[best_m]['cm']
                    fig_cm = px.imshow(cm, text_auto=True,
                                       color_continuous_scale=[[0,'#0c1220'],[1,TARGET_COLORS[target]]],
                                       labels=dict(x='Predicted', y='Actual'),
                                       title=f"{target} — {best_m}")
                    fig_cm.update_layout(**PT, height=270,
                                          margin=dict(l=0,r=0,t=45,b=0),
                                          coloraxis_showscale=False,
                                          title_font=dict(size=11, color='#7c8fad'))
                    col.plotly_chart(fig_cm, use_container_width=True)

    else:
        st.markdown("""
        <div style="background:linear-gradient(135deg,#111827,#0c1220);
                    border:1px solid rgba(255,255,255,0.06);border-radius:12px;
                    padding:2.5rem;margin-top:1rem;">
          <div style="font-family:'IBM Plex Mono',monospace;font-size:0.65rem;
                      color:#00d4a1;text-transform:uppercase;letter-spacing:0.15em;
                      margin-bottom:1.2rem;">What gets trained</div>
          <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:1rem;">
            <div style="padding:1rem;background:rgba(255,107,107,0.06);border:1px solid rgba(255,107,107,0.15);border-radius:8px;">
              <div style="color:#ff6b6b;font-weight:700;margin-bottom:0.3rem;">🔴 Attrition</div>
              <div style="font-size:0.75rem;color:#4a5a72;">Random Forest · Gradient Boosting · Logistic Reg</div>
            </div>
            <div style="padding:1rem;background:rgba(245,158,11,0.06);border:1px solid rgba(245,158,11,0.15);border-radius:8px;">
              <div style="color:#f59e0b;font-weight:700;margin-bottom:0.3rem;">⭐ Performance</div>
              <div style="font-size:0.75rem;color:#4a5a72;">Random Forest · Gradient Boosting · Logistic Reg</div>
            </div>
            <div style="padding:1rem;background:rgba(0,168,255,0.06);border:1px solid rgba(0,168,255,0.15);border-radius:8px;">
              <div style="color:#00a8ff;font-weight:700;margin-bottom:0.3rem;">📅 Absent Days</div>
              <div style="font-size:0.75rem;color:#4a5a72;">Random Forest Regressor · Ridge Regression</div>
            </div>
            <div style="padding:1rem;background:rgba(0,212,161,0.06);border:1px solid rgba(0,212,161,0.15);border-radius:8px;">
              <div style="color:#00d4a1;font-weight:700;margin-bottom:0.3rem;">🚀 Promotion</div>
              <div style="font-size:0.75rem;color:#4a5a72;">Random Forest · Gradient Boosting · Logistic Reg</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)