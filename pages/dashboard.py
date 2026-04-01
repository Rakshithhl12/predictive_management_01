import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
"""Dashboard — KPI overview"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils import generate_data

PLOTLY_THEME = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(13,17,32,0.8)',
    font=dict(color='#a0aec0', family='DM Sans'),
    xaxis=dict(gridcolor='#1e2540', zeroline=False),
    yaxis=dict(gridcolor='#1e2540', zeroline=False),
)


def show():
    st.markdown("## 🏠 HR Intelligence Dashboard")
    st.markdown("*Real-time workforce analytics powered by predictive ML*")

    df = generate_data()
    if st.session_state.df is None:
        st.session_state.df = df

    # ── KPI row ───────────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    c1.metric("👥 Total Employees", f"{len(df):,}")
    c2.metric("🔴 Attrition Rate",
              f"{df['Attrition'].mean()*100:.1f}%",
              delta=f"{(df['Attrition'].mean()-0.16)*100:+.1f}% vs benchmark",
              delta_color="inverse")
    c3.metric("⭐ Avg Performance", f"{df['PerformanceRating'].mean():.2f}/4")

    c4, c5, _ = st.columns(3)
    c4.metric("📅 Avg Absent Days", f"{df['AbsentDays'].mean():.1f}")
    c5.metric("🚀 Promotion Rate",  f"{df['PromotionLikelihood'].mean()*100:.1f}%")

    st.markdown("---")

    # ── Charts row 1 ─────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Attrition by Department")
        dept_attr = (df.groupby('Department')['Attrition']
                     .agg(['mean', 'sum', 'count']).reset_index())
        dept_attr.columns = ['Department', 'Rate', 'Count', 'Total']
        dept_attr['Rate%'] = (dept_attr['Rate'] * 100).round(1)
        fig = px.bar(dept_attr.sort_values('Rate'), x='Rate%', y='Department',
                     orientation='h', color='Rate%',
                     color_continuous_scale=['#34d399', '#fbbf24', '#f87171'],
                     text='Rate%')
        fig.update_traces(texttemplate='%{text}%', textposition='outside')
        fig.update_layout(**PLOTLY_THEME, coloraxis_showscale=False,
                          height=300, margin=dict(l=0, r=20, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Satisfaction vs Attrition")
        sat = df.groupby('JobSatisfaction').agg(
            AttrRate=('Attrition', 'mean'),
            AbsentAvg=('AbsentDays', 'mean')
        ).reset_index()
        sat['AttrRate%'] = (sat['AttrRate'] * 100).round(1)
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=sat['JobSatisfaction'], y=sat['AttrRate%'],
                              name='Attrition %', marker_color='#f87171'))
        fig2.add_trace(go.Scatter(x=sat['JobSatisfaction'], y=sat['AbsentAvg'],
                                  name='Avg Absent Days', yaxis='y2',
                                  line=dict(color='#60a5fa', width=2),
                                  mode='lines+markers'))
        # FIX: 'transparent' is not valid in Plotly — use rgba(0,0,0,0) instead
        fig2.update_layout(**PLOTLY_THEME,
                           yaxis2=dict(overlaying='y', side='right',
                                       gridcolor='rgba(0,0,0,0)', color='#60a5fa'),
                           height=300, margin=dict(l=0, r=30, t=10, b=0))
        fig2.update_layout(legend=dict(orientation='h', y=-0.2))
        st.plotly_chart(fig2, use_container_width=True)

    # ── Charts row 2 ─────────────────────────────────────────────────────────
    col3, col4 = st.columns(2)

    with col3:
        st.markdown("#### Performance Distribution")
        perf = df['PerformanceRating'].value_counts().sort_index().reset_index()
        perf.columns = ['Rating', 'Count']
        perf['Label'] = perf['Rating'].map({
            1: 'Needs Improvement', 2: 'Meets Expectations',
            3: 'Exceeds', 4: 'Outstanding'
        })
        fig3 = px.pie(perf, values='Count', names='Label', hole=0.55,
                      color_discrete_sequence=['#f87171', '#fbbf24', '#34d399', '#4f6ef7'])
        fig3.update_layout(**PLOTLY_THEME, height=300,
                           margin=dict(l=0, r=0, t=10, b=0))
        fig3.update_layout(legend=dict(orientation='h', y=-0.15))
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.markdown("#### Salary Distribution by Department")
        fig4 = px.violin(df, x='Department', y='Salary', color='Department',
                         box=True, points=False,
                         color_discrete_sequence=px.colors.qualitative.Pastel)
        fig4.update_layout(**PLOTLY_THEME, height=300,
                           margin=dict(l=0, r=0, t=10, b=0),
                           showlegend=False)
        st.plotly_chart(fig4, use_container_width=True)

    # ── High-risk table ───────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### ⚠️ High-Risk Employee Segments")
    risk = df[
        (df['JobSatisfaction'] <= 2) |
        (df['Attrition'] == 1) |
        (df['AbsentDays'] >= 15)
    ][['Department', 'Role', 'Age', 'Tenure', 'JobSatisfaction',
       'AbsentDays', 'Attrition', 'PerformanceRating']].head(10)
    st.dataframe(
        risk.style
            .background_gradient(subset=['JobSatisfaction'], cmap='RdYlGn')
            .background_gradient(subset=['AbsentDays'], cmap='OrRd')
            .format({'Attrition': lambda x: '🔴 Yes' if x == 1 else '🟢 No'}),
        use_container_width=True, height=320
    )