import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils import generate_data

PT = dict(
    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#7c8fad', family='Plus Jakarta Sans'),
    xaxis=dict(gridcolor='rgba(255,255,255,0.04)', zeroline=False, tickfont=dict(size=10)),
    yaxis=dict(gridcolor='rgba(255,255,255,0.04)', zeroline=False, tickfont=dict(size=10)),
    margin=dict(l=0, r=0, t=10, b=0),
    legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(size=10)),
)

NUM_COLS = ['Age','Tenure','Salary','JobSatisfaction','Workload','ManagerScore',
            'NumPromotions','TrainingHours','Overtime','DistanceFromOffice',
            'NumProjects','Attrition','PerformanceRating','AbsentDays','PromotionLikelihood']
CAT_COLS = ['Gender','Department','Role','Education']
PASTEL   = px.colors.qualitative.Pastel


def show():
    st.markdown("""
    <div class="page-title">
      <div class="section-label">Analysis</div>
      <h1>EDA Explorer</h1>
      <p>Interactively explore distributions, correlations, and custom segments</p>
    </div>
    """, unsafe_allow_html=True)

    df = generate_data()

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊  Distributions", "🔗  Correlations", "🔍  Segments", "🗂  Raw Data"]
    )

    with tab1:
        col_a, col_b = st.columns([1, 3], gap="medium")
        with col_a:
            st.markdown('<div class="info-box">Choose a feature and split variable to explore its distribution.</div>', unsafe_allow_html=True)
            feature    = st.selectbox("Feature", NUM_COLS, index=0)
            group_by   = st.selectbox("Split by", ['None'] + CAT_COLS + ['Attrition','PerformanceRating'])
            chart_type = st.radio("Chart type", ["Histogram","Box","Violin","ECDF"])
        with col_b:
            color_col = None if group_by == 'None' else group_by
            if chart_type == "Histogram":
                fig = px.histogram(df, x=feature, color=color_col, barmode='overlay',
                                   opacity=0.75, nbins=30, color_discrete_sequence=PASTEL)
            elif chart_type == "Box":
                fig = px.box(df, x=color_col or feature,
                             y=feature if color_col else None,
                             color=color_col, color_discrete_sequence=PASTEL)
            elif chart_type == "Violin":
                fig = px.violin(df, x=color_col, y=feature, color=color_col,
                                box=True, color_discrete_sequence=PASTEL)
            else:
                fig = px.ecdf(df, x=feature, color=color_col, color_discrete_sequence=PASTEL)
            fig.update_layout(**PT, height=400)
            st.plotly_chart(fig, use_container_width=True)

        with st.expander("📋  Descriptive Statistics"):
            st.dataframe(
                df[NUM_COLS].describe().T
                  .style.background_gradient(subset=['mean'], cmap='Blues')
                        .format(precision=2),
                use_container_width=True
            )

    with tab2:
        corr_cols = st.multiselect("Select features for correlation matrix",
                                    NUM_COLS, default=NUM_COLS[:10])
        if len(corr_cols) >= 2:
            corr     = df[corr_cols].corr()
            fig_corr = px.imshow(corr, color_continuous_scale='RdBu_r',
                                  zmin=-1, zmax=1, text_auto='.2f', aspect='auto')
            fig_corr.update_layout(**PT, height=480,
                                    coloraxis_colorbar=dict(
                                        tickfont=dict(size=9, color='#7c8fad'),
                                        title=dict(font=dict(size=9, color='#7c8fad'))
                                    ))
            st.plotly_chart(fig_corr, use_container_width=True)

        st.markdown("<hr>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            x_feat = st.selectbox("X axis", NUM_COLS, index=0, key='sx')
        with col2:
            y_feat = st.selectbox("Y axis", NUM_COLS, index=6, key='sy')
        with col3:
            c_feat = st.selectbox("Color by", ['Attrition','PerformanceRating','Department'], key='sc')
        fig_sc = px.scatter(df.sample(600, random_state=1), x=x_feat, y=y_feat,
                            color=c_feat, opacity=0.65, trendline='ols',
                            color_discrete_sequence=PASTEL)
        fig_sc.update_layout(**PT, height=380)
        st.plotly_chart(fig_sc, use_container_width=True)

    with tab3:
        st.markdown("""
        <div class="info-box">Filter employees using the controls below to analyse custom segments.</div>
        """, unsafe_allow_html=True)
        col1, col2 = st.columns(2, gap="medium")
        with col1:
            dept_f = st.multiselect("Department", df['Department'].unique(),
                                     default=list(df['Department'].unique()))
            age_r  = st.slider("Age range", 22, 60, (25, 50))
        with col2:
            sat_f  = st.multiselect("Satisfaction", [1,2,3,4,5], default=[1,2,3,4,5])

        filtered = df[
            df['Department'].isin(dept_f) &
            df['Age'].between(*age_r) &
            df['JobSatisfaction'].isin(sat_f)
        ]

        m1, m2, m3 = st.columns(3)
        m1.metric("Employees",       f"{len(filtered):,}")
        m2.metric("Attrition Rate",  f"{filtered['Attrition'].mean()*100:.1f}%" if len(filtered) else "—")
        m3.metric("Avg Satisfaction",f"{filtered['JobSatisfaction'].mean():.2f}" if len(filtered) else "—")

        if len(filtered) > 10:
            fig_seg = px.scatter(filtered, x='Tenure', y='Salary',
                                  color='Attrition', size='AbsentDays',
                                  hover_data=['Department','Role','PerformanceRating'],
                                  color_discrete_map={0:'#10b981', 1:'#ff6b6b'},
                                  opacity=0.7)
            fig_seg.update_layout(**PT, height=420)
            st.plotly_chart(fig_seg, use_container_width=True)
        else:
            st.info("Adjust filters to show more employees.")

    with tab4:
        st.markdown(f'<div class="badge badge-teal">{len(df):,} employees · {len(df.columns)} features</div>',
                    unsafe_allow_html=True)
        st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)
        search  = st.text_input("🔍  Search rows", "", placeholder="e.g. Engineering")
        show_df = df
        if search:
            mask    = df.astype(str).apply(lambda r: r.str.contains(search, case=False)).any(axis=1)
            show_df = df[mask]
        st.dataframe(show_df, use_container_width=True, height=420)
        st.download_button("⬇  Export CSV",
                           show_df.to_csv(index=False).encode(),
                           "hr_filtered.csv", "text/csv")
