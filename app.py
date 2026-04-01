"""
HR Intelligence Platform — Redesigned UI
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st

st.set_page_config(
    page_title="HR Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="auto",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=IBM+Plex+Mono:wght@300;400;500&display=swap');

/* ════════════════════════════════════════════
   CSS VARIABLES
════════════════════════════════════════════ */
:root {
  --bg:        #070b12;
  --bg2:       #0c1220;
  --bg3:       #111827;
  --surface:   #141d2e;
  --surface2:  #1a2540;
  --border:    rgba(255,255,255,0.07);
  --border2:   rgba(0,212,161,0.25);
  --accent:    #00d4a1;
  --accent2:   #00a8ff;
  --accent3:   #ff6b6b;
  --amber:     #f59e0b;
  --text:      #e2e8f4;
  --text2:     #7c8fad;
  --text3:     #4a5a72;
  --glow:      0 0 20px rgba(0,212,161,0.15);
  --glow2:     0 0 40px rgba(0,212,161,0.08);
  --radius:    12px;
  --radius2:   8px;
}

/* ════════════════════════════════════════════
   GLOBAL RESET
════════════════════════════════════════════ */
*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"] {
  font-family: 'Plus Jakarta Sans', sans-serif !important;
  color: var(--text);
}

.stApp {
  background: var(--bg) !important;
  background-image:
    linear-gradient(rgba(0,212,161,0.015) 1px, transparent 1px),
    linear-gradient(90deg, rgba(0,212,161,0.015) 1px, transparent 1px) !important;
  background-size: 40px 40px !important;
}

/* ════════════════════════════════════════════
   MAIN CONTENT
════════════════════════════════════════════ */
.block-container {
  padding: clamp(1rem, 3vw, 2rem) clamp(1rem, 3vw, 2.5rem) 3rem !important;
  max-width: 100% !important;
}

/* ════════════════════════════════════════════
   SIDEBAR
════════════════════════════════════════════ */
[data-testid="stSidebar"] {
  background: var(--bg2) !important;
  border-right: 1px solid var(--border) !important;
  min-width: 240px !important;
}
[data-testid="stSidebar"]::before {
  content: '';
  position: absolute;
  left: 0; top: 0; bottom: 0;
  width: 3px;
  background: linear-gradient(180deg, transparent, var(--accent), transparent);
}
[data-testid="stSidebar"] * { color: var(--text2) !important; }

/* ── FIX: Hide Streamlit's native sidebar chrome to prevent double sidebar ── */
[data-testid="stSidebarHeader"],
[data-testid="stSidebarCollapseButton"],
[data-testid="collapsedControl"],
[data-testid="stSidebarNav"],
button[kind="header"] {
  display: none !important;
}
/* Remove extra top padding Streamlit injects for its own header */
[data-testid="stSidebar"] > div:first-child {
  padding-top: 0 !important;
}

/* Nav buttons */
[data-testid="stSidebar"] .stButton > button {
  width: 100% !important;
  background: transparent !important;
  border: 1px solid transparent !important;
  color: var(--text2) !important;
  border-radius: var(--radius2) !important;
  padding: 0.55rem 1rem !important;
  font-family: 'Plus Jakarta Sans', sans-serif !important;
  font-size: 0.82rem !important;
  font-weight: 500 !important;
  text-align: left !important;
  transition: all 0.2s ease !important;
  letter-spacing: 0.01em !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
  background: rgba(0,212,161,0.08) !important;
  border-color: var(--border2) !important;
  color: var(--accent) !important;
  transform: translateX(3px) !important;
}

/* ════════════════════════════════════════════
   HEADINGS
════════════════════════════════════════════ */
h1, h2 {
  font-family: 'Plus Jakarta Sans', sans-serif !important;
  font-weight: 800 !important;
  color: #ffffff !important;
  letter-spacing: -0.03em !important;
  font-size: clamp(1.4rem, 3.5vw, 2rem) !important;
  line-height: 1.1 !important;
}
h3 {
  font-family: 'Plus Jakarta Sans', sans-serif !important;
  font-weight: 700 !important;
  color: var(--text) !important;
  font-size: clamp(1rem, 2vw, 1.15rem) !important;
  letter-spacing: -0.02em !important;
}
h4 {
  font-family: 'Plus Jakarta Sans', sans-serif !important;
  font-weight: 600 !important;
  color: var(--text2) !important;
  font-size: 0.85rem !important;
  text-transform: uppercase !important;
  letter-spacing: 0.1em !important;
}
p, li, span { font-size: clamp(0.82rem, 1.3vw, 0.9rem) !important; color: var(--text2) !important; }

/* ════════════════════════════════════════════
   METRIC CARDS
════════════════════════════════════════════ */
[data-testid="metric-container"] {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  padding: clamp(0.9rem, 2vw, 1.3rem) !important;
  min-width: 0 !important;
  overflow: hidden !important;
  position: relative !important;
  transition: border-color 0.3s, box-shadow 0.3s !important;
}
[data-testid="metric-container"]:hover {
  border-color: var(--border2) !important;
  box-shadow: var(--glow) !important;
}
[data-testid="metric-container"]::after {
  content: '';
  position: absolute;
  bottom: 0; left: 0; right: 0;
  height: 2px;
  background: linear-gradient(90deg, var(--accent), transparent);
  opacity: 0.6;
}
[data-testid="metric-container"] label {
  color: var(--text3) !important;
  font-size: 0.68rem !important;
  font-weight: 600 !important;
  text-transform: uppercase !important;
  letter-spacing: 0.12em !important;
  font-family: 'IBM Plex Mono', monospace !important;
}
[data-testid="stMetricValue"] {
  color: #ffffff !important;
  font-family: 'Plus Jakarta Sans', sans-serif !important;
  font-weight: 800 !important;
  font-size: clamp(1.4rem, 3vw, 2rem) !important;
  letter-spacing: -0.03em !important;
  line-height: 1 !important;
}
[data-testid="stMetricDelta"] {
  font-size: 0.72rem !important;
  font-family: 'IBM Plex Mono', monospace !important;
}

/* ════════════════════════════════════════════
   TABS
════════════════════════════════════════════ */
.stTabs [data-baseweb="tab-list"] {
  background: var(--surface) !important;
  border-radius: var(--radius2) !important;
  padding: 4px !important;
  border: 1px solid var(--border) !important;
  overflow-x: auto !important;
  flex-wrap: nowrap !important;
  -webkit-overflow-scrolling: touch !important;
  scrollbar-width: none !important;
  gap: 2px !important;
}
.stTabs [data-baseweb="tab-list"]::-webkit-scrollbar { display: none; }
.stTabs [data-baseweb="tab"] {
  background: transparent !important;
  color: var(--text3) !important;
  border-radius: 6px !important;
  font-size: clamp(0.72rem, 1.3vw, 0.82rem) !important;
  font-weight: 600 !important;
  white-space: nowrap !important;
  padding: 0.45rem 0.9rem !important;
  letter-spacing: 0.02em !important;
  transition: all 0.2s !important;
}
.stTabs [data-baseweb="tab"]:hover { color: var(--text) !important; }
.stTabs [aria-selected="true"] {
  background: linear-gradient(135deg, rgba(0,212,161,0.15), rgba(0,168,255,0.1)) !important;
  color: var(--accent) !important;
  border: 1px solid var(--border2) !important;
}

/* ════════════════════════════════════════════
   BUTTONS
════════════════════════════════════════════ */
.stButton > button {
  background: linear-gradient(135deg, var(--accent), #00b890) !important;
  color: #000 !important;
  border: none !important;
  border-radius: var(--radius2) !important;
  padding: clamp(0.45rem,1.2vw,0.6rem) clamp(1rem,2.5vw,1.8rem) !important;
  font-family: 'Plus Jakarta Sans', sans-serif !important;
  font-weight: 700 !important;
  font-size: clamp(0.78rem, 1.4vw, 0.88rem) !important;
  letter-spacing: 0.02em !important;
  transition: all 0.25s ease !important;
  width: 100% !important;
  box-shadow: 0 4px 15px rgba(0,212,161,0.25) !important;
}
.stButton > button:hover {
  transform: translateY(-2px) !important;
  box-shadow: 0 8px 25px rgba(0,212,161,0.4) !important;
  filter: brightness(1.08) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ════════════════════════════════════════════
   FORM INPUTS
════════════════════════════════════════════ */
[data-testid="stSlider"] > div > div {
  background: var(--accent) !important;
}
[data-testid="stSlider"] .st-emotion-cache-1xhj18k {
  background: var(--surface2) !important;
}
.stSelectbox [data-baseweb="select"] > div,
.stNumberInput input,
.stTextInput input {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius2) !important;
  color: var(--text) !important;
  font-family: 'Plus Jakarta Sans', sans-serif !important;
  font-size: 0.85rem !important;
  transition: border-color 0.2s !important;
}
.stSelectbox [data-baseweb="select"] > div:focus-within,
.stNumberInput input:focus,
.stTextInput input:focus {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 2px rgba(0,212,161,0.15) !important;
}
[data-baseweb="select"] [data-baseweb="popover"] {
  background: var(--surface) !important;
  border: 1px solid var(--border2) !important;
}

/* ════════════════════════════════════════════
   DATAFRAMES
════════════════════════════════════════════ */
[data-testid="stDataFrame"] {
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  overflow: hidden !important;
  overflow-x: auto !important;
  -webkit-overflow-scrolling: touch !important;
}
[data-testid="stDataFrame"] th {
  background: var(--surface2) !important;
  color: var(--text2) !important;
  font-family: 'IBM Plex Mono', monospace !important;
  font-size: 0.72rem !important;
  text-transform: uppercase !important;
  letter-spacing: 0.08em !important;
}

/* ════════════════════════════════════════════
   ALERTS / INFO / WARNING
════════════════════════════════════════════ */
.stAlert {
  border-radius: var(--radius) !important;
  border: 1px solid var(--border) !important;
  background: var(--surface) !important;
  font-size: 0.85rem !important;
}
[data-testid="stNotification"] { border-radius: var(--radius) !important; }

/* ════════════════════════════════════════════
   EXPANDER
════════════════════════════════════════════ */
.streamlit-expanderHeader {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius2) !important;
  color: var(--text2) !important;
  font-size: 0.85rem !important;
  font-weight: 600 !important;
}
.streamlit-expanderContent {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-top: none !important;
  border-radius: 0 0 var(--radius2) var(--radius2) !important;
}

/* ════════════════════════════════════════════
   PROGRESS BAR
════════════════════════════════════════════ */
.stProgress > div > div {
  background: linear-gradient(90deg, var(--accent), var(--accent2)) !important;
  border-radius: 99px !important;
}
.stProgress > div {
  background: var(--surface2) !important;
  border-radius: 99px !important;
}

/* ════════════════════════════════════════════
   DIVIDER
════════════════════════════════════════════ */
hr {
  border: none !important;
  border-top: 1px solid var(--border) !important;
  margin: 1.5rem 0 !important;
}

/* ════════════════════════════════════════════
   CUSTOM COMPONENTS
════════════════════════════════════════════ */

/* Section label */
.section-label {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.68rem;
  font-weight: 500;
  color: var(--accent);
  text-transform: uppercase;
  letter-spacing: 0.15em;
  margin-bottom: 0.4rem;
}

/* Page title block */
.page-title {
  margin-bottom: 1.8rem;
}
.page-title h1 {
  margin: 0 0 0.3rem !important;
}
.page-title p {
  color: var(--text2) !important;
  font-size: 0.88rem !important;
  margin: 0 !important;
}

/* Stat card (custom HTML) */
.stat-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: clamp(1rem,2.5vw,1.4rem);
  position: relative;
  overflow: hidden;
  transition: all 0.3s ease;
}
.stat-card:hover {
  border-color: var(--border2);
  box-shadow: var(--glow);
  transform: translateY(-2px);
}
.stat-card::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 1px;
  background: linear-gradient(90deg, transparent, var(--accent), transparent);
  opacity: 0.5;
}
.stat-icon {
  font-size: 1.4rem;
  margin-bottom: 0.6rem;
  display: block;
}
.stat-label {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.65rem;
  color: var(--text3);
  text-transform: uppercase;
  letter-spacing: 0.12em;
  margin-bottom: 0.35rem;
}
.stat-value {
  font-size: clamp(1.5rem, 3vw, 2rem);
  font-weight: 800;
  color: #fff;
  letter-spacing: -0.04em;
  line-height: 1;
  font-family: 'Plus Jakarta Sans', sans-serif;
}
.stat-delta {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.7rem;
  margin-top: 0.4rem;
}
.stat-delta.up   { color: #10b981; }
.stat-delta.down { color: var(--accent3); }
.stat-delta.neu  { color: var(--text3); }

/* Prediction result cards */
.resp-card-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
  gap: 1rem;
  margin-top: 1.2rem;
}
.pred-card {
  background: var(--surface);
  border-radius: var(--radius);
  padding: clamp(1rem, 2vw, 1.4rem);
  text-align: center;
  min-width: 0;
  transition: all 0.3s ease;
  position: relative;
  overflow: hidden;
}
.pred-card:hover {
  transform: translateY(-3px);
}
.pred-card::after {
  content: '';
  position: absolute;
  bottom: 0; left: 0; right: 0;
  height: 2px;
  opacity: 0.8;
}

/* Info box */
.info-box {
  background: var(--surface);
  border: 1px solid var(--border);
  border-left: 3px solid var(--accent);
  border-radius: 0 var(--radius2) var(--radius2) 0;
  padding: 1rem 1.2rem;
  margin: 1rem 0;
  font-size: 0.85rem;
  color: var(--text2);
}

/* Chart container */
.chart-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1.2rem 1rem 0.5rem;
  margin-bottom: 0.5rem;
}
.chart-title {
  font-size: 0.8rem;
  font-weight: 600;
  color: var(--text2);
  text-transform: uppercase;
  letter-spacing: 0.08em;
  font-family: 'IBM Plex Mono', monospace;
  margin-bottom: 0.5rem;
  padding-bottom: 0.5rem;
  border-bottom: 1px solid var(--border);
}

/* Badge */
.badge {
  display: inline-block;
  padding: 0.2rem 0.65rem;
  border-radius: 99px;
  font-size: 0.7rem;
  font-weight: 600;
  font-family: 'IBM Plex Mono', monospace;
  letter-spacing: 0.05em;
}
.badge-green  { color: #10b981; background: rgba(16,185,129,0.12); border: 1px solid rgba(16,185,129,0.25); }
.badge-red    { color: var(--accent3); background: rgba(255,107,107,0.12); border: 1px solid rgba(255,107,107,0.25); }
.badge-amber  { color: var(--amber); background: rgba(245,158,11,0.12); border: 1px solid rgba(245,158,11,0.25); }
.badge-teal   { color: var(--accent); background: rgba(0,212,161,0.1); border: 1px solid var(--border2); }

/* ════════════════════════════════════════════
   MOBILE  ≤ 640px
════════════════════════════════════════════ */
@media (max-width: 640px) {
  .block-container { padding: 0.75rem 0.75rem 2rem !important; }
  [data-testid="column"] {
    width: 100% !important;
    flex: 1 1 100% !important;
    min-width: 100% !important;
  }
  [data-testid="stSidebar"] { min-width: 0 !important; }
  [data-testid="stMetricValue"] { font-size: 1.4rem !important; }
  .stTabs [data-baseweb="tab"] { font-size: 0.7rem !important; padding: 0.35rem 0.6rem !important; }
  .resp-card-grid { grid-template-columns: repeat(2, 1fr) !important; }
  h1, h2 { font-size: 1.3rem !important; }
  .stat-value { font-size: 1.5rem !important; }
}

/* ════════════════════════════════════════════
   TABLET  641px – 1024px
════════════════════════════════════════════ */
@media (min-width: 641px) and (max-width: 1024px) {
  .block-container { padding: 1.2rem 1.2rem 2rem !important; }
  .resp-card-grid { grid-template-columns: repeat(2, 1fr) !important; }
  [data-testid="stMetricValue"] { font-size: 1.6rem !important; }
}

/* ════════════════════════════════════════════
   DESKTOP  > 1024px
════════════════════════════════════════════ */
@media (min-width: 1025px) {
  .resp-card-grid { grid-template-columns: repeat(4, 1fr) !important; }
}
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────
defaults = {
    "page": "🏠 Dashboard",
    "models_trained": False,
    "df": None,
    "results": {},
    "predictions_log": [],
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:1.5rem 1rem 1.8rem; border-bottom:1px solid rgba(255,255,255,0.06);">
      <div style="display:flex;align-items:center;gap:0.6rem;margin-bottom:0.4rem;">
        <span style="font-size:1.3rem;">◈</span>
        <span style="font-family:'Plus Jakarta Sans',sans-serif;font-size:1.05rem;
                     font-weight:800;color:#fff;letter-spacing:-0.02em;">HR Intelligence</span>
      </div>
      <div style="font-family:'IBM Plex Mono',monospace;font-size:0.62rem;
                  color:#2d4a3e;text-transform:uppercase;letter-spacing:0.2em;
                  padding-left:1.9rem;">Predictive Platform</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='padding:0.8rem 0 0.4rem;'>", unsafe_allow_html=True)

    pages = {
        "🏠 Dashboard":       "Overview & KPIs",
        "🔬 EDA Explorer":    "Interactive data analysis",
        "🤖 Model Training":  "Train & benchmark models",
        "🎯 Live Predictor":  "Predict individual employees",
        "📁 Batch Inference": "Upload CSV & score all",
        "📊 Model Insights":  "Feature importance & sensitivity",
    }

    for page, desc in pages.items():
        active = st.session_state.page == page
        if active:
            st.markdown(f"""
            <div style="background:rgba(0,212,161,0.1);border:1px solid rgba(0,212,161,0.25);
                        border-radius:8px;padding:0.55rem 1rem;margin:0.2rem 0;
                        font-size:0.82rem;font-weight:600;color:#00d4a1;
                        font-family:'Plus Jakarta Sans',sans-serif;cursor:default;">
                {page}
            </div>""", unsafe_allow_html=True)
        else:
            if st.button(page, key=f"nav_{page}", help=desc, use_container_width=True):
                st.session_state.page = page
                st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<hr style='border-color:rgba(255,255,255,0.06);margin:1rem 0;'>", unsafe_allow_html=True)

    if st.session_state.models_trained:
        st.markdown("""
        <div style="display:flex;align-items:center;gap:0.5rem;padding:0.6rem 0.8rem;
                    background:rgba(16,185,129,0.08);border:1px solid rgba(16,185,129,0.2);
                    border-radius:8px;font-size:0.78rem;color:#10b981;
                    font-family:'IBM Plex Mono',monospace;">
          <span style="width:6px;height:6px;background:#10b981;border-radius:50%;
                       box-shadow:0 0 6px #10b981;flex-shrink:0;"></span>
          Models ready
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="display:flex;align-items:center;gap:0.5rem;padding:0.6rem 0.8rem;
                    background:rgba(245,158,11,0.08);border:1px solid rgba(245,158,11,0.2);
                    border-radius:8px;font-size:0.78rem;color:#f59e0b;
                    font-family:'IBM Plex Mono',monospace;">
          <span style="width:6px;height:6px;background:#f59e0b;border-radius:50%;
                       flex-shrink:0;"></span>
          Train models first
        </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div style="position:absolute;bottom:1.5rem;left:0;right:0;text-align:center;
                font-family:'IBM Plex Mono',monospace;font-size:0.6rem;color:#1a2e26;">
      v2.0 · Scikit-learn · Streamlit
    </div>""", unsafe_allow_html=True)

# ── Route ──────────────────────────────────────────────────────────────────
page = st.session_state.page
if page == "🏠 Dashboard":
    from pages import dashboard; dashboard.show()
elif page == "🔬 EDA Explorer":
    from pages import eda; eda.show()
elif page == "🤖 Model Training":
    from pages import training; training.show()
elif page == "🎯 Live Predictor":
    from pages import predictor; predictor.show()
elif page == "📁 Batch Inference":
    from pages import batch; batch.show()
elif page == "📊 Model Insights":
    from pages import insights; insights.show()