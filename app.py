"""
FraudShield — End-to-End Fraud Detection Portfolio Project
Dark Neon Edition  ·  v2

Tabs:
  1. Exploratory Analysis
  2. Model Performance  (SHAP · Calibration · Threshold optimiser · Cross-val)
  3. Live Fraud Scorer  (SHAP waterfall · Session history)
  4. Business Impact    (Cost-matrix calculator · Savings projection)

Run:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle, os, sys
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import roc_curve, precision_recall_curve

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from pipeline import preprocess, train, NUMERIC_FEATURES, CATEGORICAL_FEATURES

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FraudShield",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Palette ────────────────────────────────────────────────────────────────────
BG    = "#070B1A"
BG2   = "#0D1328"
CARD  = "#101828"
GRID  = "#1A2545"
TICK  = "#4A5680"
TEXT  = "#E2E8F0"
MUTED = "#64748B"

NEON = {
    "cyan":   "#00F5FF",
    "purple": "#8B5CF6",
    "pink":   "#EC4899",
    "green":  "#10B981",
    "yellow": "#F59E0B",
    "blue":   "#3B82F6",
    "red":    "#F43F5E",
    "teal":   "#14B8A6",
    "orange": "#F97316",
}

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Syne:wght@600;700;800&family=JetBrains+Mono:wght@400;500&family=DM+Sans:wght@300;400;500&display=swap');
  html,body{{background:{BG}!important}}
  [class*="css"],.stApp{{background:{BG}!important;font-family:'DM Sans',sans-serif;color:{TEXT}}}
  h1,h2,h3{{font-family:'Syne',sans-serif!important;color:{TEXT}!important}}
  .stTabs [data-baseweb="tab-list"]{{background:{BG2};border-radius:10px;padding:4px;gap:4px}}
  .stTabs [data-baseweb="tab"]{{background:transparent;color:{MUTED};font-family:'DM Sans',sans-serif;font-size:14px;font-weight:500;border-radius:8px;padding:8px 18px}}
  .stTabs [aria-selected="true"]{{background:{CARD};color:{TEXT}!important}}
  .stTabs [data-baseweb="tab-highlight"]{{display:none}}
  .stDataFrame{{background:{CARD}!important}}
  section[data-testid="stSidebar"]{{background:{BG2}}}
  .kpi-card{{background:linear-gradient(135deg,{CARD} 0%,#0F1E3A 100%);border:1px solid {GRID};border-radius:14px;padding:22px 20px;text-align:center;position:relative;overflow:hidden}}
  .kpi-card::before{{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,{NEON["cyan"]},{NEON["purple"]})}}
  .kpi-val{{font-family:'Syne',sans-serif;font-size:28px;font-weight:800;color:{TEXT};line-height:1.1}}
  .kpi-lbl{{font-size:11px;color:{MUTED};margin-top:5px;text-transform:uppercase;letter-spacing:.06em}}
  .kpi-sub{{font-size:11px;margin-top:6px;font-weight:500;color:{NEON["cyan"]}}}
  .risk-high{{color:{NEON["red"]};background:#1F0A0F;border:1px solid {NEON["red"]};padding:5px 16px;border-radius:100px;font-size:13px;font-weight:600}}
  .risk-medium{{color:{NEON["yellow"]};background:#1F180A;border:1px solid {NEON["yellow"]};padding:5px 16px;border-radius:100px;font-size:13px;font-weight:600}}
  .risk-low{{color:{NEON["green"]};background:#0A1F14;border:1px solid {NEON["green"]};padding:5px 16px;border-radius:100px;font-size:13px;font-weight:600}}
  .stForm{{background:{CARD};border:1px solid {GRID};border-radius:14px;padding:4px}}
  .stSelectbox>div,.stNumberInput>div>div{{background:{BG2}!important;border-color:{GRID}!important;color:{TEXT}!important}}
  label{{color:{MUTED}!important;font-size:12px!important}}
  .stButton>button{{background:linear-gradient(135deg,{NEON["blue"]},{NEON["purple"]});color:white;border:none;font-weight:600;border-radius:8px}}
  hr{{border-color:{GRID}!important}}
  .insight-card{{background:{CARD};border:1px solid {GRID};border-radius:10px;padding:20px 18px}}
  .model-card{{background:{CARD};border:1px solid {GRID};border-radius:12px;padding:18px 20px;margin-bottom:8px}}
  .stSlider [data-testid="stSlider"]{{color:{NEON["cyan"]}}}
  div[data-testid="stExpander"]{{background:{CARD};border:1px solid {GRID};border-radius:10px}}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PLOT HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def dark_fig(w=6, h=3.6, ncols=1, nrows=1):
    fig, axes = plt.subplots(nrows, ncols, figsize=(w, h))
    fig.patch.set_facecolor(BG2)
    axlist = np.array(axes).flatten() if ncols * nrows > 1 else [axes]
    for ax in axlist:
        ax.set_facecolor(BG2)
        ax.spines[:].set_visible(False)
        ax.tick_params(colors=TICK, labelsize=8.5, length=0, labelcolor=TICK)
        ax.xaxis.label.set_color(MUTED)
        ax.yaxis.label.set_color(MUTED)
        ax.grid(color=GRID, linewidth=0.7, linestyle="-", zorder=0)
        ax.set_axisbelow(True)
    return fig, axes


def neon_line(ax, x, y, color, lw=2.2, glow=True, label=None, marker=None):
    if glow:
        ax.plot(x, y, color=color, linewidth=lw + 4,   alpha=0.12, solid_capstyle="round")
        ax.plot(x, y, color=color, linewidth=lw + 1.5, alpha=0.28, solid_capstyle="round")
    ax.plot(x, y, color=color, linewidth=lw, solid_capstyle="round",
            label=label, marker=marker, markersize=5 if marker else 0,
            markerfacecolor=color, markeredgecolor=BG2, markeredgewidth=1.2)


def neon_area(ax, x, y, color, alpha_fill=0.25, lw=2.0, label=None, baseline=0):
    neon_line(ax, x, y, color, lw=lw, label=label)
    ax.fill_between(x, baseline, y, alpha=alpha_fill, color=color, zorder=2)
    ax.fill_between(x, baseline, y, alpha=0.08,        color=color, zorder=1)


def neon_bar_h(ax, categories, values, color_start, color_end, annotate=True):
    cmap  = LinearSegmentedColormap.from_list("nb", [color_start, color_end])
    n     = len(values)
    cols  = [cmap(i / max(n - 1, 1)) for i in range(n)]
    bars  = ax.barh(categories, values, color=cols, height=0.55, zorder=3, edgecolor="none")
    ax.barh(categories, values, color=cols, height=0.65, alpha=0.15, zorder=2, edgecolor="none")
    if annotate:
        for bar, v in zip(bars, values):
            ax.text(bar.get_width() + max(values) * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{v:.2%}", va="center", fontsize=8.5, color=TEXT, fontweight="500")
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=1))
    ax.grid(axis="x", color=GRID, linewidth=0.7)
    ax.grid(axis="y", visible=False)
    return bars


def annotation_box(ax, x, y, text, color):
    ax.annotate(text, xy=(x, y), xytext=(8, 8), textcoords="offset points",
                fontsize=8.5, fontweight="600", color=BG2,
                bbox=dict(boxstyle="round,pad=0.35", facecolor=color,
                          edgecolor="none", alpha=0.95),
                arrowprops=dict(arrowstyle="-", color=color, lw=1.2))


def kpi_html(val, lbl, sub="", accent=NEON["cyan"]):
    sh = f'<div class="kpi-sub">{sub}</div>' if sub else ""
    return (f'<div class="kpi-card">'
            f'<div class="kpi-val">{val}</div>'
            f'<div class="kpi-lbl">{lbl}</div>{sh}</div>')


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════
DATA_PATH     = "data/FraudShield_Banking_Data.csv"
ARTIFACT_PATH = "models/artifacts.pkl"


@st.cache_data(show_spinner=False)
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_resource(show_spinner=False)
def load_or_train():
    if os.path.exists(ARTIFACT_PATH):
        with open(ARTIFACT_PATH, "rb") as f:
            return pickle.load(f)
    with st.spinner("Training model — ~60 seconds on first run…"):
        return train(DATA_PATH, output_dir="models")


# ── Session state for scorer history ──────────────────────────────────────────
if "score_history" not in st.session_state:
    st.session_state.score_history = []


# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div style="padding:36px 0 16px">
  <div style="font-size:11px;font-weight:600;color:{NEON["cyan"]};letter-spacing:.1em;
              text-transform:uppercase;margin-bottom:12px">
    ◆ &nbsp;End-to-End Portfolio Project
  </div>
  <h1 style="font-size:50px;letter-spacing:-2px;margin:0;line-height:1.02;color:{TEXT}">
    FraudShield <span style="color:{NEON["cyan"]}">Detection</span>
  </h1>
  <p style="color:{MUTED};font-size:15px;font-weight:300;margin-top:12px;
            max-width:600px;line-height:1.7">
    A full-stack fraud detection system — EDA insights, ML model comparison,
    SHAP explainability, threshold optimisation, and a live transaction scorer
    with real-time business impact analysis.
  </p>
</div>
<hr>
""", unsafe_allow_html=True)

df   = load_data()
arts = load_or_train()
eda  = arts["eda"]

# ── KPI strip ──────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5, c6 = st.columns(6)
best_res = arts["model_results"][arts["best_name"]]
cards = [
    (f"{eda['total_transactions']:,}",      "Total Transactions",  ""),
    (f"{eda['total_fraud']:,}",             "Fraud Cases",         ""),
    (f"{eda['fraud_rate']:.2%}",            "Fraud Rate",          "Class imbalance: 20:1"),
    (f"{eda['total_amount']:,.0f}M",        "Total Volume",        ""),
    (arts["best_name"].split()[0],          "Best Model",
     f"ROC-AUC {best_res['roc_auc']:.4f}"),
    (f"{best_res['cv_mean']:.4f} ± {best_res['cv_std']:.4f}",
     "5-Fold CV AUC", "Stratified"),
]
for col, (val, lbl, sub) in zip([c1, c2, c3, c4, c5, c6], cards):
    col.markdown(kpi_html(val, lbl, sub), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "📊  Exploratory Analysis",
    "🤖  Model Performance",
    "🎯  Live Fraud Scorer",
    "💼  Business Impact",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — EDA
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    cyan_color = NEON["cyan"]
    st.markdown(
        f"### <span style='color:{TEXT}'>Fraud Patterns</span> "
        f"<span style='color:{cyan_color}'>at a Glance</span>",
        unsafe_allow_html=True,
    )
    st.caption("Where, when, and how fraud occurs across 50,000 transactions.")
    st.markdown("")

    # ── Row 1: Transaction Type + Merchant Category ───────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        data = pd.DataFrame(eda["fraud_by_type"]).sort_values("fraud_rate")
        fig, ax = dark_fig(5.5, 2.8)
        neon_bar_h(ax, data["Transaction_Type"], data["fraud_rate"],
                   NEON["blue"], NEON["cyan"])
        ax.set_title("Fraud Rate by Transaction Type", fontsize=11,
                     fontweight="bold", color=TEXT, pad=12, loc="left")
        fig.tight_layout(pad=1.5)
        st.pyplot(fig, use_container_width=True); plt.close()

    with col2:
        data = pd.DataFrame(eda["fraud_by_merchant"]).sort_values("fraud_rate")
        fig, ax = dark_fig(5.5, 2.8)
        neon_bar_h(ax, data["Merchant_Category"], data["fraud_rate"],
                   NEON["purple"], NEON["pink"])
        ax.set_title("Fraud Rate by Merchant Category", fontsize=11,
                     fontweight="bold", color=TEXT, pad=12, loc="left")
        fig.tight_layout(pad=1.5)
        st.pyplot(fig, use_container_width=True); plt.close()

    # ── Row 2: City + Hour ────────────────────────────────────────────────────
    col3, col4 = st.columns(2)
    with col3:
        data = pd.DataFrame(eda["fraud_by_location"]).sort_values("fraud_rate")
        fig, ax = dark_fig(5.5, 3.8)
        norm_c = plt.Normalize(data["fraud_rate"].min(), data["fraud_rate"].max())
        cmap_c = LinearSegmentedColormap.from_list("city", [NEON["blue"], NEON["cyan"]])
        cols_c = [cmap_c(norm_c(v)) for v in data["fraud_rate"]]
        bars   = ax.barh(data["Transaction_Location"], data["fraud_rate"],
                         color=cols_c, height=0.58, zorder=3, edgecolor="none")
        ax.barh(data["Transaction_Location"], data["fraud_rate"],
                color=cols_c, height=0.68, alpha=0.12, zorder=2, edgecolor="none")
        ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=1))
        ax.set_title("Fraud Rate by City", fontsize=11,
                     fontweight="bold", color=TEXT, pad=12, loc="left")
        ax.grid(axis="x", color=GRID, linewidth=0.7)
        ax.grid(axis="y", visible=False)
        for bar, rate in zip(bars, data["fraud_rate"]):
            ax.text(bar.get_width() + 0.0006, bar.get_y() + bar.get_height() / 2,
                    f"{rate:.2%}", va="center", fontsize=8.5, color=TEXT, fontweight="500")
        fig.tight_layout(pad=1.5)
        st.pyplot(fig, use_container_width=True); plt.close()

    with col4:
        data = pd.DataFrame(eda["fraud_by_hour"]).dropna(subset=["Hour"])
        data = data.sort_values("Hour")
        data["Hour"] = data["Hour"].astype(int)
        fig, ax = dark_fig(5.5, 3.8)
        x, y = data["Hour"].values, data["fraud_rate"].values
        neon_area(ax, x, y, NEON["cyan"], alpha_fill=0.3, lw=2.2)
        peak_idx = np.argmax(y)
        ax.scatter(x[peak_idx], y[peak_idx], color=NEON["cyan"],
                   s=80, zorder=10, edgecolors=BG2, linewidths=1.5)
        annotation_box(ax, x[peak_idx], y[peak_idx], f"{y[peak_idx]:.1%}", NEON["cyan"])
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=1))
        ax.set_xlabel("Hour of Day", fontsize=9)
        ax.set_xlim(0, 23)
        ax.set_title("Fraud Rate by Hour of Day", fontsize=11,
                     fontweight="bold", color=TEXT, pad=12, loc="left")
        fig.tight_layout(pad=1.5)
        st.pyplot(fig, use_container_width=True); plt.close()

    # ── Row 3: Amount distribution + Prior fraud ──────────────────────────────
    col5, col6 = st.columns(2)
    with col5:
        fig, ax = dark_fig(5.5, 3.2)
        bins = np.linspace(0, 10, 22)
        n_vals, _ = np.histogram(eda["amount_normal"], bins=bins, density=True)
        f_vals, _ = np.histogram(eda["amount_fraud"],  bins=bins, density=True)
        bcs = (bins[:-1] + bins[1:]) / 2
        neon_area(ax, bcs, n_vals, NEON["green"], alpha_fill=0.25, lw=2.0, label="Normal")
        neon_area(ax, bcs, f_vals, NEON["red"],   alpha_fill=0.3,  lw=2.0, label="Fraud")
        ax.legend(fontsize=9, frameon=False, labelcolor=TEXT, loc="upper right")
        ax.set_xlabel("Transaction Amount (M)", fontsize=9)
        ax.set_title("Amount Distribution: Fraud vs Normal", fontsize=11,
                     fontweight="bold", color=TEXT, pad=12, loc="left")
        fig.tight_layout(pad=1.5)
        st.pyplot(fig, use_container_width=True); plt.close()

    with col6:
        prev  = pd.DataFrame(eda["fraud_by_prev_fraud"])
        fig, ax = dark_fig(5.5, 3.2)
        xlbls  = prev["Previous_Fraud_Count"].astype(str).tolist()
        rates  = prev["fraud_rate"].values
        cmap2  = LinearSegmentedColormap.from_list("rb", [NEON["yellow"], NEON["red"]])
        norm2  = plt.Normalize(rates.min(), rates.max())
        cols2  = [cmap2(norm2(v)) for v in rates]
        bars   = ax.bar(xlbls, rates, color=cols2, width=0.5, zorder=3, edgecolor="none")
        ax.bar(xlbls, rates, color=cols2, width=0.6, alpha=0.15, zorder=2, edgecolor="none")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=1))
        ax.set_xlabel("Previous Fraud Count", fontsize=9)
        ax.set_title("Fraud Rate by Prior Fraud History", fontsize=11,
                     fontweight="bold", color=TEXT, pad=12, loc="left")
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0012,
                    f"{bar.get_height():.1%}", ha="center", fontsize=9.5,
                    color=TEXT, fontweight="600")
        fig.tight_layout(pad=1.5)
        st.pyplot(fig, use_container_width=True); plt.close()

    # ── Row 4: Risk flag combo + Distance violin ──────────────────────────────
    col7, col8 = st.columns(2)
    with col7:
        combo  = pd.DataFrame(eda["fraud_by_combo"]).sort_values("fraud_rate")
        fig, ax = dark_fig(5.5, 3.0)
        cmap_c2 = LinearSegmentedColormap.from_list("combo", [NEON["blue"], NEON["red"]])
        n_c     = len(combo)
        cols_c2 = [cmap_c2(i / max(n_c - 1, 1)) for i in range(n_c)]
        bars_c  = ax.barh(combo["Combo"], combo["fraud_rate"],
                          color=cols_c2, height=0.5, zorder=3, edgecolor="none")
        ax.barh(combo["Combo"], combo["fraud_rate"],
                color=cols_c2, height=0.62, alpha=0.14, zorder=2, edgecolor="none")
        ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=1))
        ax.set_title("Fraud Rate: Intl × New Merchant Combinations",
                     fontsize=11, fontweight="bold", color=TEXT, pad=12, loc="left")
        ax.grid(axis="x", color=GRID, linewidth=0.7)
        ax.grid(axis="y", visible=False)
        for bar in bars_c:
            ax.text(bar.get_width() + 0.0006, bar.get_y() + bar.get_height() / 2,
                    f"{bar.get_width():.2%}", va="center", fontsize=9,
                    color=TEXT, fontweight="600")
        fig.tight_layout(pad=1.5)
        st.pyplot(fig, use_container_width=True); plt.close()

    with col8:
        fig, ax = dark_fig(5.5, 3.0)
        vp = ax.violinplot(
            [np.array(eda["distance_normal"]), np.array(eda["distance_fraud"])],
            positions=[0, 1], showmedians=True, showextrema=False,
        )
        for body, color in zip(vp["bodies"], [NEON["green"], NEON["red"]]):
            body.set_facecolor(color); body.set_alpha(0.35)
            body.set_edgecolor(color); body.set_linewidth(1.5)
        vp["cmedians"].set_color(TEXT); vp["cmedians"].set_linewidth(2)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Normal", "Fraud"], fontsize=10, color=TEXT)
        ax.set_ylabel("Distance (km)", fontsize=9)
        ax.set_title("Distance from Home: Fraud vs Normal", fontsize=11,
                     fontweight="bold", color=TEXT, pad=12, loc="left")
        fig.tight_layout(pad=1.5)
        st.pyplot(fig, use_container_width=True); plt.close()

    # ── Correlation heatmap ───────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(f"#### <span style='color:{TEXT}'>Feature Correlation Matrix</span>",
                unsafe_allow_html=True)
    corr_data = eda["correlation_matrix"]
    corr_arr  = np.array(corr_data["values"])
    cols_corr = [c.replace(" (in Million)", "").replace("Transaction_", "Tx_")
                 for c in corr_data["columns"]]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    fig.patch.set_facecolor(BG2); ax.set_facecolor(BG2)
    cmap_h = LinearSegmentedColormap.from_list(
        "hm", [NEON["purple"], BG2, NEON["cyan"]]
    )
    im = ax.imshow(corr_arr, cmap=cmap_h, vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cols_corr)));  ax.set_xticklabels(cols_corr, rotation=35,
                                                               ha="right", fontsize=8, color=TICK)
    ax.set_yticks(range(len(cols_corr)));  ax.set_yticklabels(cols_corr, fontsize=8, color=TICK)
    ax.spines[:].set_visible(False);       ax.tick_params(length=0)
    for i in range(len(cols_corr)):
        for j in range(len(cols_corr)):
            v    = corr_arr[i, j]
            clr  = TEXT if abs(v) < 0.5 else BG2
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=7, color=clr, fontweight="500")
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.ax.tick_params(labelsize=7, colors=TICK)
    cbar.outline.set_visible(False)
    fig.suptitle("Pearson Correlations with Fraud Label (bottom-right = Is_Fraud)",
                 fontsize=9, color=MUTED, x=0.02, ha="left")
    fig.tight_layout(pad=1.8)
    st.pyplot(fig, use_container_width=True); plt.close()

    # ── Insight cards ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(f"#### <span style='color:{TEXT}'>💡 Key Insights</span>",
                unsafe_allow_html=True)
    ins_cols = st.columns(4)
    insights = [
        (NEON["cyan"],   "🌍", "International + New Merchant",
         "The 'Both' combo carries the highest fraud rate — the single strongest combined signal in the dataset."),
        (NEON["purple"], "🕐", "Late-Night Spike",
         "Fraud peaks after 22:00. `Unusual_Time_Transaction` ranks in the top 3 predictors across all models."),
        (NEON["pink"],   "📍", "Distance from Home",
         "Fraudulent transactions occur on average 2× farther from home — a strong standalone risk indicator."),
        (NEON["green"],  "📈", "Prior Fraud = #1 Predictor",
         "Even a single prior fraud incident dramatically increases probability. Recidivism is the strongest single feature."),
    ]
    for col, (accent, icon, title, body) in zip(ins_cols, insights):
        col.markdown(
            f"""<div style="background:{CARD};border:1px solid {GRID};
                            border-left:3px solid {accent};border-radius:10px;
                            padding:20px 18px;">
              <div style="font-size:22px;margin-bottom:10px">{icon}</div>
              <div style="font-family:Syne,sans-serif;font-weight:700;font-size:14px;
                          color:{TEXT};margin-bottom:8px">{title}</div>
              <div style="font-size:13px;color:{MUTED};line-height:1.65">{body}</div>
            </div>""",
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown(
        f"### <span style='color:{TEXT}'>Model Comparison</span> "
        f"<span style='color:{NEON['purple']}'>& Evaluation</span>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Three classifiers with class balancing · "
        "5-fold stratified CV · SHAP explainability · Threshold optimiser"
    )
    st.markdown("")

    results = arts["model_results"]

    # ── Model comparison table ────────────────────────────────────────────────
    comp_data = []
    for name, r in results.items():
        rep = r["report"]
        comp_data.append({
            "Model":                name,
            "ROC-AUC (test)":       f"{r['roc_auc']:.4f}",
            "PR-AUC (test)":        f"{r['pr_auc']:.4f}",
            "CV AUC (5-fold)":      f"{r['cv_mean']:.4f} ± {r['cv_std']:.4f}",
            "Brier Score ↓":        f"{r['brier']:.4f}",
            "Precision (Fraud)":    f"{rep['1']['precision']:.4f}",
            "Recall (Fraud)":       f"{rep['1']['recall']:.4f}",
            "F1 (Fraud)":           f"{rep['1']['f1-score']:.4f}",
            "🏆":                   "✅" if name == arts["best_name"] else "",
        })
    st.dataframe(pd.DataFrame(comp_data), use_container_width=True, hide_index=True)
    st.markdown("")

    # ── ROC + PR curves ───────────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    curve_colors = [NEON["cyan"], NEON["purple"], NEON["pink"]]

    with col1:
        fig, ax = dark_fig(5.5, 4.2)
        for (name, r), color in zip(results.items(), curve_colors):
            fpr, tpr, _ = roc_curve(r["y_test"], r["y_prob"])
            neon_line(ax, fpr, tpr, color, lw=2.2,
                      label=f"{name.split()[0]}  {r['roc_auc']:.3f}")
            ax.fill_between(fpr, tpr, alpha=0.05, color=color)
        ax.plot([0, 1], [0, 1], "--", color=GRID, linewidth=1.2, label="Random")
        ax.set_xlabel("False Positive Rate", fontsize=9)
        ax.set_ylabel("True Positive Rate", fontsize=9)
        ax.set_title("ROC Curve", fontsize=11, fontweight="bold",
                     color=TEXT, pad=12, loc="left")
        ax.legend(fontsize=8.5, frameon=False, labelcolor=TEXT)
        fig.tight_layout(pad=1.5)
        st.pyplot(fig, use_container_width=True); plt.close()

    with col2:
        fig, ax = dark_fig(5.5, 4.2)
        for (name, r), color in zip(results.items(), curve_colors):
            prec, rec, _ = precision_recall_curve(r["y_test"], r["y_prob"])
            neon_line(ax, rec, prec, color, lw=2.2,
                      label=f"{name.split()[0]}  {r['pr_auc']:.3f}")
            ax.fill_between(rec, prec, alpha=0.05, color=color)
        ax.axhline(eda["fraud_rate"], linestyle="--", color=GRID, linewidth=1.2,
                   label=f"Baseline {eda['fraud_rate']:.2%}")
        ax.set_xlabel("Recall", fontsize=9)
        ax.set_ylabel("Precision", fontsize=9)
        ax.set_title("Precision-Recall Curve", fontsize=11, fontweight="bold",
                     color=TEXT, pad=12, loc="left")
        ax.legend(fontsize=8.5, frameon=False, labelcolor=TEXT)
        fig.tight_layout(pad=1.5)
        st.pyplot(fig, use_container_width=True); plt.close()

    # ── Confusion matrix + Feature importance ─────────────────────────────────
    col3, col4 = st.columns(2)
    with col3:
        st.markdown(f"**Confusion Matrix — {arts['best_name']}**")
        cm_arr = np.array(results[arts["best_name"]]["cm"])
        fig, ax = plt.subplots(figsize=(4.8, 3.8))
        fig.patch.set_facecolor(BG2); ax.set_facecolor(BG2)
        cmap_cm = LinearSegmentedColormap.from_list("neon_cm", [BG2, NEON["cyan"]])
        ax.imshow(cm_arr, cmap=cmap_cm, aspect="auto")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Normal", "Fraud"], fontsize=10, color=TEXT)
        ax.set_yticklabels(["Normal", "Fraud"], fontsize=10, color=TEXT)
        ax.set_xlabel("Predicted", fontsize=9, color=MUTED)
        ax.set_ylabel("Actual",    fontsize=9, color=MUTED)
        ax.tick_params(length=0); ax.spines[:].set_visible(False)
        for i in range(2):
            for j in range(2):
                clr = BG2 if cm_arr[i, j] > cm_arr.max() * 0.5 else TEXT
                ax.text(j, i, f"{cm_arr[i,j]:,}", ha="center", va="center",
                        fontsize=17, fontweight="800", color=clr)
        labels = ["TN", "FP", "FN", "TP"]
        for idx, (i, j) in enumerate([(0,0),(0,1),(1,0),(1,1)]):
            ax.text(j + 0.38, i + 0.38, labels[idx], ha="right", va="bottom",
                    fontsize=7.5, color=MUTED, fontweight="500")
        fig.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

    with col4:
        st.markdown("**Top 15 Feature Importances**")
        fi = arts["feature_importance"].head(15).copy().sort_values("importance")
        fig, ax = dark_fig(4.8, 3.8)
        cmap_fi = LinearSegmentedColormap.from_list("fi", [NEON["purple"], NEON["cyan"]])
        n_fi = len(fi)
        cols_fi = [cmap_fi(i / max(n_fi - 1, 1)) for i in range(n_fi)]
        ax.barh(fi["feature"], fi["importance"], color=cols_fi, height=0.58, zorder=3)
        ax.barh(fi["feature"], fi["importance"], color=cols_fi, height=0.70, alpha=0.12, zorder=2)
        ax.set_title("Tree Feature Importance", fontsize=11, fontweight="bold",
                     color=TEXT, pad=12, loc="left")
        ax.grid(axis="x", color=GRID, linewidth=0.7)
        ax.grid(axis="y", visible=False)
        ax.tick_params(labelsize=8)
        fig.tight_layout(pad=1.5)
        st.pyplot(fig, use_container_width=True); plt.close()

    # ── SHAP summary plot ─────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(f"#### <span style='color:{TEXT}'>🔬 SHAP Feature Explainability</span>",
                unsafe_allow_html=True)
    st.caption(
        "SHAP (SHapley Additive exPlanations) shows not just *which* features matter "
        "but *how* they push predictions higher or lower — across 500 test transactions."
    )

    shap_data = arts.get("shap_data")
    if shap_data is not None:
        col_shap1, col_shap2 = st.columns(2)

        with col_shap1:
            # Mean |SHAP| bar chart (global importance)
            mean_abs = shap_data["mean_abs"].head(15).sort_values()
            fig, ax  = dark_fig(5.5, 4.2)
            cmap_s   = LinearSegmentedColormap.from_list("sv", [NEON["teal"], NEON["purple"]])
            n_s      = len(mean_abs)
            cols_s   = [cmap_s(i / max(n_s - 1, 1)) for i in range(n_s)]
            ax.barh(mean_abs.index, mean_abs.values, color=cols_s,
                    height=0.55, zorder=3, edgecolor="none")
            ax.barh(mean_abs.index, mean_abs.values, color=cols_s,
                    height=0.68, alpha=0.13, zorder=2, edgecolor="none")
            for i, (feat, val) in enumerate(zip(mean_abs.index, mean_abs.values)):
                ax.text(val + mean_abs.values.max() * 0.01, i,
                        f"{val:.4f}", va="center", fontsize=8, color=TEXT)
            ax.set_xlabel("Mean |SHAP value|", fontsize=9)
            ax.set_title("Global Feature Impact (SHAP)",
                         fontsize=11, fontweight="bold", color=TEXT, pad=12, loc="left")
            ax.grid(axis="x", color=GRID, linewidth=0.7)
            ax.grid(axis="y", visible=False)
            fig.tight_layout(pad=1.5)
            st.pyplot(fig, use_container_width=True); plt.close()

        with col_shap2:
            # SHAP beeswarm-style dot plot (top 10 features)
            sv   = shap_data["shap_values"]
            X_sh = shap_data["X_shap"]
            top_features = shap_data["mean_abs"].head(10).index.tolist()
            top_idx      = [list(X_sh.columns).index(f) for f in top_features]
            sv_top       = sv[:, top_idx]
            X_top        = X_sh.iloc[:, top_idx].values

            fig, ax = dark_fig(5.5, 4.2)
            for i, feat in enumerate(top_features[::-1]):
                feat_i = len(top_features) - 1 - i
                shap_col  = sv_top[:, feat_i]
                feat_vals = X_top[:, feat_i]

                # Normalise feature values to [0,1] for colour mapping
                fv_norm = (feat_vals - feat_vals.min()) / (feat_vals.ptp() + 1e-9)
                fv_norm = np.clip(fv_norm, 0, 1)
                point_colors = [
                    (NEON["blue"] if v < 0.5 else NEON["red"])
                    for v in fv_norm
                ]

                jitter = np.random.RandomState(i).uniform(-0.25, 0.25, len(shap_col))
                ax.scatter(shap_col, np.full_like(shap_col, i) + jitter,
                           c=point_colors, s=5, alpha=0.5, zorder=3, linewidths=0)

            ax.set_yticks(range(len(top_features)))
            ax.set_yticklabels(top_features[::-1], fontsize=8, color=TICK)
            ax.axvline(0, color=GRID, linewidth=1.2, zorder=1)
            ax.set_xlabel("SHAP value  (pushes prediction → Fraud)", fontsize=8.5)
            ax.set_title("SHAP Beeswarm  (blue=low feature val · red=high)",
                         fontsize=10, fontweight="bold", color=TEXT, pad=12, loc="left")
            ax.grid(axis="x", color=GRID, linewidth=0.7)
            ax.grid(axis="y", visible=False)
            fig.tight_layout(pad=1.5)
            st.pyplot(fig, use_container_width=True); plt.close()
    else:
        st.info("SHAP not available. Run `pip install shap` and retrain to enable explainability charts.")

    # ── Threshold optimiser ───────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(f"#### <span style='color:{TEXT}'>⚙️ Decision Threshold Optimiser</span>",
                unsafe_allow_html=True)
    st.caption(
        "Fraud models are never deployed at threshold = 0.5. "
        "Slide to see how precision, recall, and F1 trade off in real time."
    )

    thresh_df = pd.DataFrame(arts["threshold_analysis"]["data"])
    opt_f1    = arts["threshold_analysis"]["optimal_f1_threshold"]
    opt_cost  = arts["threshold_analysis"]["optimal_cost_threshold"]

    t_slider = st.slider(
        "Decision threshold", min_value=0.05, max_value=0.95,
        value=float(opt_f1), step=0.05,
        format="%.2f",
        help="Probability cut-off above which a transaction is flagged as fraud",
    )

    row = thresh_df[thresh_df["threshold"].between(t_slider - 0.01, t_slider + 0.01)].iloc[0]

    m1, m2, m3, m4, m5 = st.columns(5)
    def _metric(col, val, lbl, col_accent):
        col.markdown(
            f'<div style="background:{CARD};border:1px solid {GRID};border-radius:10px;'
            f'padding:14px 12px;text-align:center">'
            f'<div style="font-family:Syne,sans-serif;font-size:22px;font-weight:800;'
            f'color:{col_accent}">{val}</div>'
            f'<div style="font-size:11px;color:{MUTED};margin-top:4px;text-transform:uppercase;'
            f'letter-spacing:.05em">{lbl}</div></div>',
            unsafe_allow_html=True,
        )

    _metric(m1, f"{row['precision']:.3f}", "Precision",  NEON["cyan"])
    _metric(m2, f"{row['recall']:.3f}",    "Recall",     NEON["purple"])
    _metric(m3, f"{row['f1']:.3f}",        "F1 Score",   NEON["green"])
    _metric(m4, f"{int(row['fp']):,}",     "False Pos",  NEON["yellow"])
    _metric(m5, f"{int(row['fn']):,}",     "False Neg",  NEON["red"])

    st.markdown("<br>", unsafe_allow_html=True)
    fig, ax = dark_fig(10, 3.5)
    x_t = thresh_df["threshold"].values
    neon_line(ax, x_t, thresh_df["precision"].values, NEON["cyan"],   lw=1.8, label="Precision")
    neon_line(ax, x_t, thresh_df["recall"].values,    NEON["purple"], lw=1.8, label="Recall")
    neon_line(ax, x_t, thresh_df["f1"].values,        NEON["green"],  lw=2.2, label="F1")
    ax.axvline(t_slider, color=TEXT,       linewidth=1.5, linestyle="--", alpha=0.8)
    ax.axvline(opt_f1,   color=NEON["green"], linewidth=1, linestyle=":",  alpha=0.6,
               label=f"Opt F1 = {opt_f1:.2f}")
    ax.set_xlabel("Decision Threshold", fontsize=9)
    ax.set_ylabel("Score",              fontsize=9)
    ax.set_title("Precision / Recall / F1 vs Threshold",
                 fontsize=11, fontweight="bold", color=TEXT, pad=12, loc="left")
    ax.legend(fontsize=8.5, frameon=False, labelcolor=TEXT, ncol=4)
    ax.set_xlim(0.05, 0.95); ax.set_ylim(0, 1.05)
    fig.tight_layout(pad=1.5)
    st.pyplot(fig, use_container_width=True); plt.close()

    # ── Calibration curve ─────────────────────────────────────────────────────
    st.markdown("---")
    col_cal1, col_cal2 = st.columns([1, 1])
    with col_cal1:
        st.markdown(f"#### <span style='color:{TEXT}'>📐 Probability Calibration</span>",
                    unsafe_allow_html=True)
        st.caption(
            "A well-calibrated model's predicted probabilities match actual fraud rates. "
            f"Brier Score = **{best_res['brier']:.4f}** (lower is better; 0 = perfect)."
        )
        cal = arts["calibration"]
        fig, ax = dark_fig(5.5, 4.0)
        neon_line(ax, cal["prob_pred"], cal["prob_true"], NEON["cyan"], lw=2.2,
                  label=f"{arts['best_name'].split()[0]}", marker="o")
        ax.fill_between(cal["prob_pred"], cal["prob_pred"], cal["prob_true"],
                        alpha=0.15, color=NEON["yellow"], label="Calibration gap")
        ax.plot([0, 1], [0, 1], "--", color=GRID, linewidth=1.5, label="Perfectly calibrated")
        ax.set_xlabel("Mean Predicted Probability", fontsize=9)
        ax.set_ylabel("Fraction of Positives",      fontsize=9)
        ax.set_title("Reliability Diagram",
                     fontsize=11, fontweight="bold", color=TEXT, pad=12, loc="left")
        ax.legend(fontsize=8.5, frameon=False, labelcolor=TEXT)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        fig.tight_layout(pad=1.5)
        st.pyplot(fig, use_container_width=True); plt.close()

    with col_cal2:
        st.markdown(f"#### <span style='color:{TEXT}'>📋 Model Card</span>",
                    unsafe_allow_html=True)
        st.markdown(f"""
<div style="background:{CARD};border:1px solid {GRID};border-radius:12px;padding:20px 22px;
            font-size:13px;line-height:1.9;color:{MUTED}">
  <div style="color:{TEXT};font-family:Syne,sans-serif;font-weight:700;font-size:15px;
              margin-bottom:12px">🏆 {arts["best_name"]}</div>
  <div><span style="color:{NEON["cyan"]};font-weight:600">Dataset</span>
    &nbsp;50,000 banking transactions · 4.8% fraud rate</div>
  <div><span style="color:{NEON["cyan"]};font-weight:600">Imbalance</span>
    &nbsp;Handled via class weighting / sample weighting</div>
  <div><span style="color:{NEON["cyan"]};font-weight:600">Evaluation</span>
    &nbsp;Stratified 80/20 split + 5-fold CV</div>
  <div><span style="color:{NEON["cyan"]};font-weight:600">Primary metric</span>
    &nbsp;PR-AUC (appropriate for imbalanced classification)</div>
  <div><span style="color:{NEON["cyan"]};font-weight:600">Optimal threshold</span>
    &nbsp;{opt_f1:.2f} (F1) · {opt_cost:.2f} (cost-optimal)</div>
  <div><span style="color:{NEON["yellow"]};font-weight:600">Limitations</span>
    &nbsp;Trained on synthetic data; calibration may drift on real distributions</div>
  <div><span style="color:{NEON["yellow"]};font-weight:600">Bias check</span>
    &nbsp;No demographic features used — no protected-class risk</div>
</div>""", unsafe_allow_html=True)

    with st.expander("📋 Full Classification Report"):
        rep     = results[arts["best_name"]]["report"]
        rep_df  = pd.DataFrame(rep).T
        rep_df  = rep_df[rep_df.index.isin(["0","1","macro avg","weighted avg"])]
        rep_df.index = ["Normal","Fraud","Macro Avg","Weighted Avg"]
        st.dataframe(rep_df.style.format("{:.4f}"), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — LIVE SCORER
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown(
        f"### <span style='color:{TEXT}'>Live</span> "
        f"<span style='color:{NEON['green']}'>Fraud Scorer</span>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Enter a transaction for instant fraud probability, risk tier, "
        "SHAP-powered explanation, and session history."
    )
    st.markdown("")

    col_form, col_result = st.columns([1.2, 1])

    with col_form:
        with st.form("scorer_form"):
            st.markdown("**Transaction Details**")
            fc1, fc2 = st.columns(2)
            amount       = fc1.number_input("Amount (M)",              min_value=0.0, max_value=10.0,  value=5.0, step=0.5)
            balance      = fc2.number_input("Account Balance (M)",     min_value=0.0, max_value=40.0, value=20.0, step=1.0)
            distance     = fc1.number_input("Distance from Home (km)", min_value=0,   max_value=600,   value=50)
            tx_time      = fc2.text_input("Transaction Time (HH:MM)",  value="14:30")
            tx_type      = fc1.selectbox("Transaction Type",  ["Online","ATM","POS"])
            merchant_cat = fc2.selectbox("Merchant Category", ["Restaurant","ATM","Fuel","Clothing","Grocery","Electronics"])
            card_type    = fc1.selectbox("Card Type",         ["Debit","Credit"])
            tx_location  = fc2.selectbox("Transaction Location",
                ["Karachi","Lahore","Islamabad","Faisalabad","Multan",
                 "Dubai","London","Singapore","Bangkok","Kuala Lumpur"])
            home_loc     = fc1.selectbox("Home Location",
                ["Karachi","Lahore","Islamabad","Faisalabad","Multan",
                 "Dubai","London","Singapore","Bangkok","Kuala Lumpur"])

            st.markdown("**Behavioural Signals**")
            bc1, bc2, bc3 = st.columns(3)
            daily_tx   = bc1.number_input("Daily Tx",    min_value=1, max_value=7,   value=3)
            weekly_tx  = bc2.number_input("Weekly Tx",   min_value=1, max_value=24,  value=10)
            avg_amount = bc3.number_input("Avg Amt (M)", min_value=1.0, max_value=5.0, value=3.0, step=0.5)
            max_24h    = bc1.number_input("Max 24h (M)", min_value=1.0, max_value=9.0, value=4.0, step=0.5)
            failed     = bc2.number_input("Failed Tx",   min_value=0, max_value=2,    value=0)
            prev_fraud = bc3.number_input("Prev Fraud",  min_value=0, max_value=1,    value=0)

            st.markdown("**Risk Flags**")
            fl1, fl2, fl3 = st.columns(3)
            is_intl = fl1.selectbox("International?", ["No","Yes"])
            is_new  = fl2.selectbox("New Merchant?",  ["No","Yes"])
            unusual = fl3.selectbox("Unusual Time?",  ["No","Yes"])
            submitted = st.form_submit_button("⚡  Analyse Transaction", use_container_width=True)

    with col_result:
        if submitted:
            row_input = pd.DataFrame([{
                "Transaction_Amount (in Million)":     amount,
                "Transaction_Time":                    tx_time,
                "Transaction_Date":                    "2025-01-01",
                "Transaction_Type":                    tx_type,
                "Merchant_Category":                   merchant_cat,
                "Transaction_Location":                tx_location,
                "Customer_Home_Location":              home_loc,
                "Distance_From_Home":                  distance,
                "Card_Type":                           card_type,
                "Account_Balance (in Million)":        balance,
                "Daily_Transaction_Count":             daily_tx,
                "Weekly_Transaction_Count":            weekly_tx,
                "Avg_Transaction_Amount (in Million)": avg_amount,
                "Max_Transaction_Last_24h (in Million)": max_24h,
                "Is_International_Transaction":        is_intl,
                "Is_New_Merchant":                     is_new,
                "Failed_Transaction_Count":            failed,
                "Unusual_Time_Transaction":            unusual,
                "Previous_Fraud_Count":                prev_fraud,
                "Transaction_ID": 0, "Customer_ID": 0, "Merchant_ID": 0,
                "Device_ID": 0, "IP_Address": "0.0.0.0", "Fraud_Label": "Normal",
            }])

            X_input = preprocess(row_input, encoders=arts["encoders"], fit=False)
            X_input = X_input.reindex(columns=arts["feature_names"], fill_value=0)
            prob    = arts["best_model"].predict_proba(X_input)[0][1]

            if prob >= 0.60:
                tier, tier_class, color = "HIGH",   "risk-high",   NEON["red"]
            elif prob >= 0.30:
                tier, tier_class, color = "MEDIUM", "risk-medium", NEON["yellow"]
            else:
                tier, tier_class, color = "LOW",    "risk-low",    NEON["green"]

            # ── Probability card ───────────────────────────────────────────
            st.markdown(f"""
            <div style="background:{CARD};border:1px solid {GRID};border-radius:14px;
                        padding:28px 24px;position:relative;overflow:hidden;">
              <div style="position:absolute;top:0;left:0;right:0;height:2px;
                          background:linear-gradient(90deg,{color},{NEON['purple']})"></div>
              <div style="font-size:11px;color:{MUTED};font-weight:500;letter-spacing:.08em;
                          text-transform:uppercase;margin-bottom:8px">Fraud Probability</div>
              <div style="font-family:'Syne',sans-serif;font-size:60px;font-weight:800;
                          color:{color};line-height:1;text-shadow:0 0 30px {color}55">{prob:.1%}</div>
              <div style="margin-top:16px"><span class="{tier_class}">{tier} RISK</span></div>
            </div>""", unsafe_allow_html=True)

            # Gauge bar
            st.markdown("<br>", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(4.5, 0.5))
            fig.patch.set_facecolor(BG2); ax.set_facecolor(BG2)
            ax.barh([0], [1],    color=GRID,  height=0.38, zorder=1)
            ax.barh([0], [prob], color=color, height=0.38, zorder=2)
            ax.barh([0], [prob], color=color, height=0.55, alpha=0.18, zorder=1)
            ax.set_xlim(0, 1); ax.axis("off")
            plt.tight_layout(pad=0)
            st.pyplot(fig, use_container_width=True); plt.close()

            st.markdown(f"<div style='color:{MUTED};font-size:12px;margin-bottom:14px'>"
                        f"{prob:.1%} probability · {tier} risk · "
                        f"Threshold opt-F1 = {arts['threshold_analysis']['optimal_f1_threshold']:.2f}"
                        f"</div>", unsafe_allow_html=True)

            # ── SHAP waterfall for this transaction ────────────────────────
            shap_expl = arts.get("shap_explainer")
            if shap_expl is not None:
                try:
                    sv_single = shap_expl.shap_values(X_input)
                    if isinstance(sv_single, list):
                        sv_single = sv_single[1]
                    sv_flat = sv_single[0]

                    feat_names  = arts["feature_names"]
                    shap_series = pd.Series(sv_flat, index=feat_names)
                    top_pos     = shap_series.nlargest(5)
                    top_neg     = shap_series.nsmallest(5)
                    waterfall   = pd.concat([top_pos, top_neg]).sort_values()

                    fig, ax = dark_fig(4.8, 3.4)
                    bar_colors = [NEON["red"] if v > 0 else NEON["green"]
                                  for v in waterfall.values]
                    bars_w = ax.barh(waterfall.index, waterfall.values,
                                     color=bar_colors, height=0.55, zorder=3, edgecolor="none")
                    ax.barh(waterfall.index, waterfall.values,
                            color=bar_colors, height=0.68, alpha=0.14, zorder=2, edgecolor="none")
                    ax.axvline(0, color=GRID, linewidth=1.2)
                    for bar, v in zip(bars_w, waterfall.values):
                        ax.text(v + (0.001 if v >= 0 else -0.001),
                                bar.get_y() + bar.get_height() / 2,
                                f"{v:+.4f}", va="center",
                                ha="left" if v >= 0 else "right",
                                fontsize=8, color=TEXT)
                    ax.set_xlabel("SHAP contribution to fraud probability", fontsize=8.5)
                    ax.set_title("Why this score? (SHAP waterfall)",
                                 fontsize=10, fontweight="bold", color=TEXT, pad=10, loc="left")
                    ax.tick_params(labelsize=7.5)
                    fig.tight_layout(pad=1.5)
                    st.pyplot(fig, use_container_width=True); plt.close()
                except Exception:
                    pass   # SHAP waterfall gracefully skipped on error

            # ── Rule-based risk signals ────────────────────────────────────
            st.markdown(f"**<span style='color:{TEXT}'>Risk signals detected</span>**",
                        unsafe_allow_html=True)
            flags = []
            if is_intl == "Yes":        flags.append(f"🌍 **International transaction** — higher-risk geography")
            if is_new  == "Yes":        flags.append(f"🏪 **New merchant** — no prior transaction history")
            if unusual == "Yes":        flags.append(f"🕐 **Unusual hour** — outside normal activity window")
            if tx_location != home_loc: flags.append(f"📍 **Location mismatch** — {tx_location} vs home {home_loc}")
            if distance > 400:          flags.append(f"📏 **High distance** — {distance} km from home")
            if failed > 0:              flags.append(f"❌ **{int(failed)} failed transaction(s)** in session")
            if prev_fraud > 0:          flags.append(f"⚠️ **Prior fraud history** — {int(prev_fraud)} incident(s)")
            if amount > avg_amount * 2: flags.append(f"💰 **Amount spike** — {amount}M vs avg {avg_amount}M ({amount/avg_amount:.1f}×)")
            if flags:
                for f in flags:
                    st.markdown(f"- {f}")
            else:
                st.markdown("- ✅ No strong individual risk signals detected")

            # ── Session log ────────────────────────────────────────────────
            st.session_state.score_history.append({
                "Amount":   f"{amount}M",
                "Type":     tx_type,
                "Location": tx_location,
                "Prob":     f"{prob:.1%}",
                "Tier":     tier,
            })
            st.markdown(f"<div style='margin-top:14px;font-size:11px;color:{MUTED}'>"
                        f"Model: {arts['best_name']} · "
                        f"ROC-AUC {arts['model_results'][arts['best_name']]['roc_auc']:.4f}"
                        f"</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background:{CARD};border:1px dashed {GRID};border-radius:14px;
                        padding:56px 24px;text-align:center;margin-top:4px;">
              <div style="font-size:42px;margin-bottom:14px">🛡️</div>
              <div style="font-size:15px;font-weight:600;color:{TEXT};margin-bottom:8px">Ready to Score</div>
              <div style="font-size:13px;color:{MUTED};line-height:1.7">
                Fill in the transaction details on the left<br>and click
                <strong>Analyse Transaction</strong>
              </div>
            </div>""", unsafe_allow_html=True)

    # ── Session history table ─────────────────────────────────────────────────
    if st.session_state.score_history:
        st.markdown("---")
        st.markdown(f"**<span style='color:{TEXT}'>Session Transaction History</span>**",
                    unsafe_allow_html=True)
        hist_df = pd.DataFrame(st.session_state.score_history[::-1])
        hist_df.insert(0, "#", range(len(hist_df), 0, -1))
        st.dataframe(hist_df, use_container_width=True, hide_index=True)
        if st.button("Clear history"):
            st.session_state.score_history = []
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — BUSINESS IMPACT
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown(
        f"### <span style='color:{TEXT}'>Business</span> "
        f"<span style='color:{NEON['yellow']}'>Impact Analysis</span>",
        unsafe_allow_html=True,
    )
    st.caption(
        "ML metrics alone don't tell the full story. "
        "This calculator translates model performance into financial impact "
        "using a configurable cost matrix."
    )
    st.markdown("")

    # ── Cost model inputs ─────────────────────────────────────────────────────
    st.markdown(f"#### <span style='color:{TEXT}'>💰 Cost Matrix Configuration</span>",
                unsafe_allow_html=True)
    st.markdown(
        f"<div style='font-size:13px;color:{MUTED};margin-bottom:16px;line-height:1.7'>"
        "Set the cost of each error type. "
        "<b>False Negatives</b> (missed fraud) cost the bank the transaction amount. "
        "<b>False Positives</b> (blocked legit transactions) cost customer friction, "
        "investigation overhead, and potential churn."
        "</div>",
        unsafe_allow_html=True,
    )

    ci1, ci2, ci3 = st.columns(3)
    avg_fraud_amt  = float(eda["avg_fraud_amount"])
    cost_fn        = ci1.number_input(
        "Cost per False Negative (Missed Fraud, M)",
        min_value=0.1, max_value=10.0,
        value=round(avg_fraud_amt, 2), step=0.1,
        help="Default = dataset average fraud amount",
    )
    cost_fp        = ci2.number_input(
        "Cost per False Positive (Blocked Legit, M)",
        min_value=0.0001, max_value=0.01,
        value=0.00005, step=0.00001, format="%.5f",
        help="Investigation + churn cost per incorrectly blocked transaction",
    )
    monthly_txns   = ci3.number_input(
        "Expected Monthly Transactions",
        min_value=1000, max_value=500000,
        value=50000, step=5000,
    )

    # ── Recompute cost curve with user inputs ─────────────────────────────────
    thresh_df2 = pd.DataFrame(arts["threshold_analysis"]["data"]).copy()
    thresh_df2["cost_fn_user"]  = thresh_df2["fn"] * cost_fn
    thresh_df2["cost_fp_user"]  = thresh_df2["fp"] * cost_fp
    thresh_df2["total_cost"]    = thresh_df2["cost_fn_user"] + thresh_df2["cost_fp_user"]
    thresh_df2["fraud_caught_val"] = thresh_df2["tp"] * cost_fn
    thresh_df2["net_benefit"]   = thresh_df2["fraud_caught_val"] - thresh_df2["cost_fp_user"]

    scale = monthly_txns / 10000   # dataset test set ≈ 10k rows
    thresh_df2["total_cost_scaled"]    = thresh_df2["total_cost"]    * scale
    thresh_df2["fraud_caught_scaled"]  = thresh_df2["fraud_caught_val"] * scale
    thresh_df2["net_benefit_scaled"]   = thresh_df2["net_benefit"]   * scale

    opt_row    = thresh_df2.loc[thresh_df2["total_cost_scaled"].idxmin()]
    opt_thresh = float(opt_row["threshold"])

    # ── KPIs at optimal threshold ─────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        f"#### <span style='color:{TEXT}'>📊 Monthly Impact at "
        f"<span style='color:{NEON['yellow']}'>"
        f"Optimal Threshold = {opt_thresh:.2f}</span></span>",
        unsafe_allow_html=True,
    )

    bk1, bk2, bk3, bk4 = st.columns(4)
    biz_cards = [
        (f"M {opt_row['fraud_caught_scaled']:,.1f}",
         "Fraud Value Caught / Mo", NEON["green"]),
        (f"M {opt_row['cost_fn_user'] * scale:,.1f}",
         "Fraud Missed / Mo",       NEON["red"]),
        (f"M {opt_row['cost_fp_user'] * scale:,.4f}",
         "FP Investigation Cost",   NEON["yellow"]),
        (f"M {opt_row['net_benefit_scaled']:,.1f}",
         "Net Monthly Benefit",     NEON["cyan"]),
    ]
    for col, (val, lbl, accent) in zip([bk1, bk2, bk3, bk4], biz_cards):
        col.markdown(
            f'<div style="background:{CARD};border:1px solid {GRID};border-radius:12px;'
            f'padding:20px 16px;text-align:center;border-top:2px solid {accent}">'
            f'<div style="font-family:Syne,sans-serif;font-size:22px;font-weight:800;'
            f'color:{accent}">{val}</div>'
            f'<div style="font-size:11px;color:{MUTED};margin-top:6px;'
            f'text-transform:uppercase;letter-spacing:.05em">{lbl}</div></div>',
            unsafe_allow_html=True,
        )

    # ── Cost curve chart ──────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    fig, ax = dark_fig(10, 3.8)
    x_t = thresh_df2["threshold"].values
    neon_area(ax, x_t, thresh_df2["total_cost_scaled"].values,
              NEON["red"],    alpha_fill=0.15, lw=2.0, label="Total Cost (FP + FN)")
    neon_line(ax, x_t, thresh_df2["cost_fn_user"] * scale,
              NEON["orange"], lw=1.6, label="FN Cost (missed fraud)")
    neon_line(ax, x_t, thresh_df2["cost_fp_user"] * scale,
              NEON["yellow"], lw=1.6, label="FP Cost (investigations)")
    ax.axvline(opt_thresh, color=NEON["green"], linewidth=1.8, linestyle="--",
               label=f"Optimal = {opt_thresh:.2f}")
    ax.scatter(opt_thresh, float(opt_row["total_cost_scaled"]),
               color=NEON["green"], s=100, zorder=10, edgecolors=BG2, linewidths=1.5)
    annotation_box(ax, opt_thresh, float(opt_row["total_cost_scaled"]),
                   f"Min cost @ {opt_thresh:.2f}", NEON["green"])
    ax.set_xlabel("Decision Threshold", fontsize=9)
    ax.set_ylabel("Monthly Cost (M)",   fontsize=9)
    ax.set_title("Business Cost Curve vs Threshold  (scaled to monthly volume)",
                 fontsize=11, fontweight="bold", color=TEXT, pad=12, loc="left")
    ax.legend(fontsize=8.5, frameon=False, labelcolor=TEXT, ncol=4)
    ax.set_xlim(0.05, 0.95)
    fig.tight_layout(pad=1.5)
    st.pyplot(fig, use_container_width=True); plt.close()

    # ── Savings vs no-model baseline ──────────────────────────────────────────
    st.markdown("---")
    st.markdown(f"#### <span style='color:{TEXT}'>📈 Savings vs No-Model Baseline</span>",
                unsafe_allow_html=True)
    st.caption(
        "Baseline = all fraud missed (threshold = 1.0). "
        "Shows how much fraud value the model recovers at each threshold setting."
    )

    baseline_cost = float(thresh_df2["cost_fn_user"].iloc[-1] * scale)   # all FN
    thresh_df2["savings_vs_baseline"] = baseline_cost - thresh_df2["total_cost_scaled"]

    fig, ax = dark_fig(10, 3.5)
    x_t = thresh_df2["threshold"].values
    sv  = thresh_df2["savings_vs_baseline"].values
    neon_area(ax, x_t, sv, NEON["cyan"], alpha_fill=0.22, lw=2.2,
              label="Savings vs no-model")
    ax.axhline(0, color=GRID, linewidth=1.2)
    ax.axvline(opt_thresh, color=NEON["green"], linewidth=1.8, linestyle="--",
               label=f"Optimal threshold = {opt_thresh:.2f}")
    best_saving = float(thresh_df2.loc[thresh_df2["savings_vs_baseline"].idxmax(), "savings_vs_baseline"])
    best_thresh = float(thresh_df2.loc[thresh_df2["savings_vs_baseline"].idxmax(), "threshold"])
    annotation_box(ax, best_thresh, best_saving,
                   f"M {best_saving:,.1f} / mo", NEON["cyan"])
    ax.set_xlabel("Decision Threshold", fontsize=9)
    ax.set_ylabel("Monthly Savings (M)", fontsize=9)
    ax.set_title("Monthly Savings vs No-Model Baseline",
                 fontsize=11, fontweight="bold", color=TEXT, pad=12, loc="left")
    ax.legend(fontsize=8.5, frameon=False, labelcolor=TEXT)
    ax.set_xlim(0.05, 0.95)
    fig.tight_layout(pad=1.5)
    st.pyplot(fig, use_container_width=True); plt.close()

    # ── Annual projection ─────────────────────────────────────────────────────
    st.markdown("---")
    annual_saving  = best_saving * 12
    annual_cost    = float(opt_row["total_cost_scaled"]) * 12
    fraud_stopped  = float(opt_row["tp"]) / 10000 * monthly_txns

    proj_cols = st.columns(3)
    proj_data = [
        (f"M {annual_saving:,.1f}",   "Est. Annual Savings",          NEON["cyan"]),
        (f"M {annual_cost:,.1f}",     "Est. Annual Remaining Cost",   NEON["yellow"]),
        (f"{fraud_stopped:,.0f}",     "Fraud Txns Caught / Month",    NEON["green"]),
    ]
    for col, (val, lbl, accent) in zip(proj_cols, proj_data):
        col.markdown(
            f'<div style="background:{CARD};border:1px solid {GRID};'
            f'border-left:3px solid {accent};border-radius:10px;'
            f'padding:22px 18px;text-align:center">'
            f'<div style="font-family:Syne,sans-serif;font-size:26px;font-weight:800;'
            f'color:{accent}">{val}</div>'
            f'<div style="font-size:12px;color:{MUTED};margin-top:6px;'
            f'text-transform:uppercase;letter-spacing:.05em">{lbl}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        f"<div style='margin-top:20px;padding:16px 20px;background:{CARD};"
        f"border:1px solid {GRID};border-radius:10px;font-size:12px;color:{MUTED};line-height:1.8'>"
        f"<b style='color:{TEXT}'>Assumptions & Disclaimers:</b> "
        f"Cost figures are illustrative and denominated in the same units as the dataset (Millions). "
        f"FP cost ({cost_fp}M ≈ small investigation overhead) and FN cost ({cost_fn}M ≈ avg fraud amount) "
        f"should be calibrated to your institution's actual cost structure. "
        f"Projections scale linearly from the 10,000-row test set to monthly volume."
        f"</div>",
        unsafe_allow_html=True,
    )


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    f"<div style='text-align:center;font-size:12px;color:{MUTED};padding:8px 0 28px'>"
    "FraudShield · Python · scikit-learn · SHAP · Streamlit · 50,000 transactions"
    "</div>",
    unsafe_allow_html=True,
)