import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils import predict_employee, TARGET_META, generate_data, FEATURE_COLS

PT = dict(
    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#7c8fad', family='Plus Jakarta Sans'),
    xaxis=dict(gridcolor='rgba(255,255,255,0.04)', zeroline=False, tickfont=dict(size=10)),
    yaxis=dict(gridcolor='rgba(255,255,255,0.04)', zeroline=False, tickfont=dict(size=10)),
    margin=dict(l=0, r=0, t=10, b=0),
    legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(size=10)),
)


def show():
    st.markdown("""
    <div class="page-title">
      <div class="section-label">Batch Processing</div>
      <h1>Batch Inference</h1>
      <p>Upload a CSV to score your entire workforce in seconds</p>
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

    results  = st.session_state.results
    scaler   = st.session_state.scaler
    encoders = st.session_state.encoders

    # ── Upload area ────────────────────────────────────────────────────────────
    template = generate_data(5)[FEATURE_COLS]
    col_dl, _ = st.columns([1, 3])
    with col_dl:
        st.download_button("⬇  Download Template CSV",
                           template.to_csv(index=False).encode(),
                           "hr_template.csv", "text/csv",
                           use_container_width=True)

    uploaded = st.file_uploader("", type=["csv"], label_visibility="collapsed")

    if not uploaded:
        st.markdown("""
        <div style="background:linear-gradient(135deg,#0c1220,#070b12);
                    border:1px dashed rgba(255,255,255,0.08);border-radius:12px;
                    padding:3rem 2rem;text-align:center;margin-top:0.5rem;">
          <div style="font-size:2.5rem;margin-bottom:0.8rem;">📂</div>
          <div style="font-size:0.9rem;color:#4a5a72;margin-bottom:0.4rem;">
            Drop your CSV here or use the uploader above
          </div>
          <div style="font-family:'IBM Plex Mono',monospace;font-size:0.68rem;color:#1a2e26;">
            Required: Age · Gender · Department · Role · Education · Tenure · Salary<br>
            JobSatisfaction · Workload · ManagerScore · NumPromotions · TrainingHours<br>
            Overtime · DistanceFromOffice · NumProjects
          </div>
        </div>""", unsafe_allow_html=True)
        return

    try:
        df_upload = pd.read_csv(uploaded)
        st.markdown(f'<div class="badge badge-green" style="margin-bottom:1rem;">✓  Loaded {len(df_upload):,} employees</div>',
                    unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return

    missing = [c for c in FEATURE_COLS if c not in df_upload.columns]
    if missing:
        st.error(f"Missing columns: {missing}")
        return

    with st.spinner(f"Scoring {len(df_upload):,} employees…"):
        all_preds = []
        progress  = st.progress(0)
        for i, (_, row) in enumerate(df_upload.iterrows()):
            p = predict_employee(row.to_dict(), results, scaler, encoders)
            entry = row.to_dict()
            for target, pred in p.items():
                entry[f'Pred_{target}'] = pred['value']
                if pred.get('proba') is not None and len(pred['proba'])==2:
                    entry[f'Prob_{target}'] = round(pred['proba'][1], 3)
            all_preds.append(entry)
            progress.progress((i+1)/len(df_upload))

    result_df = pd.DataFrame(all_preds)
    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Summary KPIs ───────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🔴 Attrition Risk",
              f"{result_df['Pred_Attrition'].mean()*100:.1f}%",
              f"{result_df['Pred_Attrition'].sum():.0f} flagged")
    c2.metric("⭐ Avg Performance",
              f"{result_df['Pred_PerformanceRating'].mean():.2f}/4")
    c3.metric("📅 Avg Absent Days",
              f"{result_df['Pred_AbsentDays'].mean():.1f}")
    c4.metric("🚀 Promotion Ready",
              f"{result_df['Pred_PromotionLikelihood'].mean()*100:.1f}%",
              f"{result_df['Pred_PromotionLikelihood'].sum():.0f} employees")

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    # ── Charts ─────────────────────────────────────────────────────────────────
    col1, col2 = st.columns(2, gap="medium")
    with col1:
        st.markdown('<div class="chart-card"><div class="chart-title">Attrition Probability Distribution</div>', unsafe_allow_html=True)
        if 'Prob_Attrition' in result_df.columns:
            fig = px.histogram(result_df, x='Prob_Attrition', nbins=25,
                               color_discrete_sequence=['#ff6b6b'], opacity=0.8)
            fig.add_vline(x=0.5, line_dash='dot', line_color='#f59e0b',
                          annotation_text='Threshold',
                          annotation_font=dict(size=10, color='#f59e0b'))
            fig.update_traces(marker_line_width=0)
            fig.update_layout(**PT, height=280)
            st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="chart-card"><div class="chart-title">Absent Days vs Attrition Risk</div>', unsafe_allow_html=True)
        y_col = 'Prob_Attrition' if 'Prob_Attrition' in result_df.columns else 'Pred_Attrition'
        c_col = 'Department' if 'Department' in result_df.columns else None
        fig2  = px.scatter(result_df, x='Pred_AbsentDays', y=y_col,
                           color=c_col, opacity=0.6,
                           color_discrete_sequence=px.colors.qualitative.Pastel)
        fig2.update_traces(marker=dict(size=5, line=dict(width=0)))
        fig2.update_layout(**PT, height=280)
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── High-risk table ────────────────────────────────────────────────────────
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    if 'Prob_Attrition' in result_df.columns:
        high_risk = result_df[result_df['Prob_Attrition'] > 0.5].sort_values(
            'Prob_Attrition', ascending=False)
    else:
        high_risk = result_df[result_df['Pred_Attrition']==1]

    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:0.6rem;margin-bottom:0.8rem;">
      <span class="badge badge-red">⚠  High Risk — {len(high_risk)} employees</span>
      <span style="font-size:0.78rem;color:#4a5a72;">above 50% attrition threshold</span>
    </div>""", unsafe_allow_html=True)

    display_cols = [c for c in ['Department','Role','Age','Tenure','Salary',
                                 'JobSatisfaction','Prob_Attrition',
                                 'Pred_PerformanceRating'] if c in high_risk.columns]
    st.dataframe(high_risk[display_cols].head(20), use_container_width=True, height=290)

    st.download_button("⬇  Download Scored Results",
                       result_df.to_csv(index=False).encode(),
                       "hr_scored.csv", "text/csv",
                       use_container_width=True)
