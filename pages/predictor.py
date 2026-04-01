import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from utils import predict_employee, TARGET_META

PT = dict(
    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#7c8fad', family='Plus Jakarta Sans'),
    margin=dict(l=30, r=30, t=20, b=20),
)


def show():
    st.markdown("""
    <div class="page-title">
      <div class="section-label">Inference</div>
      <h1>Live Predictor</h1>
      <p>Configure an employee profile and get instant predictions across all 4 targets</p>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.models_trained:
        st.markdown("""
        <div style="padding:1.5rem;background:rgba(245,158,11,0.08);border:1px solid rgba(245,158,11,0.2);
                    border-radius:10px;font-size:0.85rem;color:#f59e0b;font-family:'IBM Plex Mono',monospace;">
          ⚠  Train models first on the Model Training page
        </div>""", unsafe_allow_html=True)
        return

    results  = st.session_state.results
    scaler   = st.session_state.scaler
    encoders = st.session_state.encoders

    with st.form("employee_form"):
        st.markdown("""
        <div style="font-family:'IBM Plex Mono',monospace;font-size:0.68rem;
                    color:#00d4a1;text-transform:uppercase;letter-spacing:0.15em;
                    margin-bottom:1rem;">Employee Profile</div>
        """, unsafe_allow_html=True)

        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.markdown('<div style="font-size:0.72rem;color:#4a5a72;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.5rem;">Personal & Role</div>', unsafe_allow_html=True)
            age      = st.slider("Age", 22, 60, 34)
            tenure   = st.slider("Tenure (years)", 0, 35, 4)
            salary   = st.number_input("Salary ($)", 30000, 200000, 70000, 5000)
            overtime = st.selectbox("Overtime", [0,1], format_func=lambda x: "Yes" if x else "No")
            dept     = st.selectbox("Department", ['Engineering','Sales','HR','Finance','Marketing','Operations'])
            role     = st.selectbox("Role", ['Analyst','Manager','Director','IC','Lead'])
            edu      = st.selectbox("Education", ['High School','Bachelor','Master','PhD'])
            gender   = st.selectbox("Gender", ['Male','Female','Other'])

        with c2:
            st.markdown('<div style="font-size:0.72rem;color:#4a5a72;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.5rem;">Work Metrics</div>', unsafe_allow_html=True)
            satisfaction  = st.slider("Job Satisfaction  (1–5)", 1, 5, 3)
            workload      = st.slider("Workload  (1–5)", 1, 5, 3)
            manager_score = st.slider("Manager Score  (1–5)", 1, 5, 3)
            training_hrs  = st.slider("Training Hours / yr", 0, 80, 30)
            num_promotions= st.slider("Past Promotions", 0, 6, 1)
            distance      = st.slider("Distance from Office  (km)", 1, 60, 15)
            num_projects  = st.slider("Active Projects", 1, 8, 3)

        submitted = st.form_submit_button("◈  Run Prediction", use_container_width=True)

    if submitted:
        employee = {
            'Age':age,'Gender':gender,'Department':dept,'Role':role,'Education':edu,
            'Tenure':tenure,'Salary':float(salary),'JobSatisfaction':satisfaction,
            'Workload':workload,'ManagerScore':manager_score,'NumPromotions':num_promotions,
            'TrainingHours':training_hrs,'Overtime':overtime,'DistanceFromOffice':distance,
            'NumProjects':num_projects,
        }
        preds = predict_employee(employee, results, scaler, encoders)
        st.session_state.predictions_log.append(
            {**employee, **{k:v['value'] for k,v in preds.items()}}
        )

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("""
        <div class="section-label" style="margin-bottom:0.8rem;">Prediction Results</div>
        """, unsafe_allow_html=True)

        # ── Result cards ──────────────────────────────────────────────────
        CARD_CFG = {
            'Attrition': {
                'fmt': lambda v,p: ("HIGH RISK" if v==1 else "LOW RISK",
                                    f"{p[1]*100:.0f}% probability" if p is not None else "",
                                    '#ff6b6b' if v==1 else '#10b981'),
            },
            'PerformanceRating': {
                'fmt': lambda v,p: (f"Rating {v}/4",
                                    ['','Needs Work','Meets Exp.','Exceeds','Outstanding'][v],
                                    ['','#ff6b6b','#f59e0b','#10b981','#00d4a1'][v]),
            },
            'AbsentDays': {
                'fmt': lambda v,p: (f"{v:.0f}  days/yr",
                                    "High risk" if v>10 else ("Moderate" if v>6 else "Healthy"),
                                    '#ff6b6b' if v>10 else ('#f59e0b' if v>6 else '#10b981')),
            },
            'PromotionLikelihood': {
                'fmt': lambda v,p: ("LIKELY" if v==1 else "UNLIKELY",
                                    f"{p[1]*100:.0f}% probability" if p is not None else "",
                                    '#00d4a1' if v==1 else '#4a5a72'),
            },
        }
        card_html = '<div class="resp-card-grid">'
        for target, cfg in CARD_CFG.items():
            pred  = preds[target]
            val   = pred['value']
            proba = pred.get('proba')
            meta  = TARGET_META[target]
            title, sub, color = cfg['fmt'](val, proba)
            card_html += f"""
            <div class="pred-card" style="border:1px solid {color}22;
                 background:linear-gradient(135deg,rgba(20,29,46,0.9),rgba(12,18,32,0.9));">
              <div class="pred-card::after" style="background:{color};"></div>
              <div style="font-size:1.6rem;margin-bottom:0.5rem;">{meta['icon']}</div>
              <div style="font-family:'IBM Plex Mono',monospace;font-size:0.6rem;
                          color:#4a5a72;text-transform:uppercase;letter-spacing:0.12em;
                          margin-bottom:0.5rem;">{meta['label']}</div>
              <div style="font-size:clamp(0.9rem,2vw,1.1rem);font-weight:800;
                          color:{color};letter-spacing:-0.02em;line-height:1.1;
                          font-family:'Plus Jakarta Sans',sans-serif;">{title}</div>
              <div style="font-family:'IBM Plex Mono',monospace;font-size:0.68rem;
                          color:#4a5a72;margin-top:0.4rem;">{sub}</div>
              <div style="position:absolute;bottom:0;left:0;right:0;height:2px;
                          background:{color};opacity:0.4;border-radius:0 0 12px 12px;"></div>
            </div>"""
        card_html += '</div>'
        st.markdown(card_html, unsafe_allow_html=True)

        # ── Radar ─────────────────────────────────────────────────────────
        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
        binary_vals = {
            TARGET_META[t]['label']: p['proba'][1]*100
            for t, p in preds.items()
            if p.get('proba') is not None and len(p['proba'])==2
        }
        if binary_vals:
            cats   = list(binary_vals.keys())
            vals   = list(binary_vals.values())
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=vals+[vals[0]], theta=cats+[cats[0]],
                fill='toself',
                fillcolor='rgba(0,212,161,0.08)',
                line=dict(color='#00d4a1', width=2),
                marker=dict(color='#00d4a1', size=7,
                            line=dict(color='#000', width=1.5))
            ))
            fig.update_layout(
                **PT,
                polar=dict(
                    bgcolor='rgba(12,18,32,0.8)',
                    radialaxis=dict(visible=True, range=[0,100],
                                   gridcolor='rgba(255,255,255,0.05)',
                                   tickfont=dict(size=9, color='#4a5a72'),
                                   color='#4a5a72'),
                    angularaxis=dict(gridcolor='rgba(255,255,255,0.05)',
                                     tickfont=dict(size=10, color='#7c8fad')),
                ),
                height=340,
            )
            c_left, c_mid, c_right = st.columns([1,2,1])
            with c_mid:
                st.plotly_chart(fig, use_container_width=True)

    # ── Log ───────────────────────────────────────────────────────────────────
    if st.session_state.predictions_log:
        st.markdown("<hr>", unsafe_allow_html=True)
        with st.expander(f"📋  Predictions Log  ({len(st.session_state.predictions_log)} entries)"):
            log_df = pd.DataFrame(st.session_state.predictions_log)
            st.dataframe(log_df, use_container_width=True)
            c1, c2 = st.columns(2)
            with c1:
                st.download_button("⬇  Export Log",
                                   log_df.to_csv(index=False).encode(),
                                   "predictions_log.csv", "text/csv",
                                   use_container_width=True)
            with c2:
                if st.button("🗑  Clear Log", use_container_width=True):
                    st.session_state.predictions_log = []
                    st.rerun()
