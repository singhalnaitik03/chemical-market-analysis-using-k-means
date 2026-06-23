# analysis.py
# Indian Chemical Sector Financial Analysis - FY2019 to FY2024
# Naitik Singhal, IIT Kanpur

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'data'))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from indian_chemicals_data import build_master_df

PLOTS_DIR = "/home/claude/indian-chemical-sector-analysis/plots"
DATA_DIR  = "/home/claude/indian-chemical-sector-analysis/data"
os.makedirs(PLOTS_DIR, exist_ok=True)

# Simple, readable colour palette - nothing flashy
# Using muted tones that look like something you'd pick yourself
SCOL = {
    "Specialty":     "#3a6b9e",
    "Agrochemical":  "#4a8c6f",
    "Commodity":     "#b85450",
    "Dye & Pigment": "#8b7355",
}

# Plain matplotlib style - no custom backgrounds, just clean
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "axes.facecolor":    "white",
    "figure.facecolor":  "white",
    "axes.grid":         True,
    "grid.color":        "#dddddd",
    "grid.linewidth":    0.5,
    "grid.linestyle":    "--",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.spines.left":  True,
    "axes.spines.bottom":True,
    "axes.edgecolor":    "#888888",
    "axes.labelsize":    10,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   9,
    "axes.titlesize":    11,
    "axes.titleweight":  "bold",
    "figure.dpi":        100,
})

df = build_master_df()

YEARS      = [2019, 2020, 2021, 2022, 2023, 2024]
FYEARS     = ["FY19","FY20","FY21","FY22","FY23","FY24"]
EBITDA_COLS = [f"EBITDA_FY{str(y)[2:]}" for y in YEARS]
REV_COLS    = [f"Rev_FY{str(y)[2:]}"    for y in YEARS]
PAT_COLS    = [f"PAT_FY{str(y)[2:]}"    for y in YEARS]
ROE_COLS    = [f"ROE_FY{str(y)[2:]}"    for y in YEARS]
SUBSECTORS  = ["Specialty","Agrochemical","Commodity","Dye & Pigment"]


# PLOT 1 - EBITDA margin trends by sub-sector
def plot_ebitda_trend():
    fig, ax = plt.subplots(figsize=(10, 5.5))
    
    markers = {"Specialty":"o","Agrochemical":"s","Commodity":"^","Dye & Pigment":"D"}
    
    for ss in SUBSECTORS:
        sub = df[df["Sub_Sector"] == ss]
        means = sub[EBITDA_COLS].mean().values
        ax.plot(FYEARS, means,
                marker=markers[ss], linewidth=1.8, markersize=5.5,
                color=SCOL[ss], label=ss, zorder=3)
        # end label
        ax.text(5.12, means[-1], f"  {ss}", fontsize=8,
                color=SCOL[ss], va="center")

    # light shading for covid period
    ax.axvspan(0.8, 1.8, alpha=0.07, color="grey", lw=0)
    ax.text(1.3, 8.5, "COVID", fontsize=7.5, color="grey", ha="center")
    
    ax.set_xlim(-0.2, 6.0)
    ax.set_ylim(5, 42)
    ax.set_xlabel("Financial Year")
    ax.set_ylabel("Average EBITDA Margin (%)")
    ax.set_title("Average EBITDA Margins by Sub-sector (FY2019-FY2024)")
    ax.legend(loc="upper left", frameon=True, edgecolor="#cccccc")
    
    fig.tight_layout()
    path = f"{PLOTS_DIR}/01_ebitda_margin_trend.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved: {path}")


# PLOT 2 - Revenue CAGR vs average EBITDA margin
def plot_cagr_vs_ebitda():
    fig, ax = plt.subplots(figsize=(11, 6.5))
    
    for ss in SUBSECTORS:
        sub = df[df["Sub_Sector"] == ss]
        # bubble size based on FY24 revenue, scaled down
        sizes = sub["Rev_FY24"] / 40
        ax.scatter(sub["Rev_CAGR_5Y"], sub["Avg_EBITDA"],
                   s=sizes, color=SCOL[ss], alpha=0.65,
                   edgecolors="#ffffff", linewidth=0.7,
                   label=ss, zorder=3)
        for _, row in sub.iterrows():
            ax.annotate(row["Ticker"],
                        (row["Rev_CAGR_5Y"], row["Avg_EBITDA"]),
                        textcoords="offset points", xytext=(5,3),
                        fontsize=7.5, color=SCOL[ss])

    # quadrant reference lines
    mx = df["Rev_CAGR_5Y"].mean()
    my = df["Avg_EBITDA"].mean()
    ax.axvline(mx, color="#aaaaaa", lw=0.8, ls="--")
    ax.axhline(my, color="#aaaaaa", lw=0.8, ls="--")
    
    # quiet quadrant labels
    ylim = ax.get_ylim(); xlim = ax.get_xlim()
    ax.text(mx+0.3, ylim[1]-1.2, "High growth, high margin",
            fontsize=7.5, color="#777777")
    ax.text(xlim[0]+0.2, ylim[1]-1.2, "Low growth, high margin",
            fontsize=7.5, color="#777777")
    ax.text(mx+0.3, ylim[0]+0.5, "High growth, low margin",
            fontsize=7.5, color="#777777")
    ax.text(xlim[0]+0.2, ylim[0]+0.5, "Low growth, low margin",
            fontsize=7.5, color="#777777")

    ax.set_xlabel("5-Year Revenue CAGR, FY2019 to FY2024 (%)")
    ax.set_ylabel("Average EBITDA Margin, FY2019-FY2024 (%)")
    ax.set_title("Growth vs Profitability - 22 NSE-listed Chemical Companies\n(bubble size proportional to FY2024 revenue)")
    ax.legend(loc="lower right", frameon=True, edgecolor="#cccccc")
    
    fig.tight_layout()
    path = f"{PLOTS_DIR}/02_growth_profitability_matrix.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved: {path}")


# PLOT 3 - K-Means clustering
def plot_kmeans_clusters():
    features = ["Rev_CAGR_5Y","Avg_EBITDA","Avg_PAT","Avg_ROE",
                "Export_Pct","RnD_Pct","Capex_Pct"]
    X = df[features].values
    X_sc = StandardScaler().fit_transform(X)

    # elbow
    inertias = []
    for k in range(2,8):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_sc)
        inertias.append(km.inertia_)

    km = KMeans(n_clusters=4, random_state=42, n_init=10)
    km.fit(X_sc)
    df["Cluster"] = km.labels_

    pca = PCA(n_components=2, random_state=42)
    Xp  = pca.fit_transform(X_sc)

    # remap cluster IDs by descending avg EBITDA so cluster 0 = best
    order = df.groupby("Cluster")["Avg_EBITDA"].mean().sort_values(ascending=False).index
    remap = {old:new for new,old in enumerate(order)}
    df["Cluster"] = df["Cluster"].map(remap)

    CCOL = ["#3a6b9e","#4a8c6f","#b85450","#8b7355"]
    CLABELS = ["Cluster 1","Cluster 2","Cluster 3","Cluster 4"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # elbow plot
    ax = axes[0]
    ax.plot(range(2,8), inertias, marker="o", color="#555555",
            linewidth=1.6, markersize=5)
    ax.axvline(4, color="#b85450", ls="--", lw=1.0, alpha=0.7)
    ax.text(4.15, max(inertias)*0.9, "k = 4 selected",
            fontsize=8.5, color="#b85450")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Within-cluster sum of squares")
    ax.set_title("Elbow Method - Choosing Optimal k")

    # PCA scatter
    ax = axes[1]
    for c in range(4):
        mask = df["Cluster"] == c
        ax.scatter(Xp[mask,0], Xp[mask,1],
                   color=CCOL[c], s=70, alpha=0.8,
                   edgecolors="white", linewidth=0.6,
                   label=CLABELS[c], zorder=3)
        for i, row in df[mask].iterrows():
            ax.annotate(row["Ticker"],
                        (Xp[i,0], Xp[i,1]),
                        textcoords="offset points", xytext=(4,2),
                        fontsize=7.2, color=CCOL[c])

    ve = pca.explained_variance_ratio_ * 100
    ax.set_xlabel(f"PC1 ({ve[0]:.1f}% variance explained)")
    ax.set_ylabel(f"PC2 ({ve[1]:.1f}% variance explained)")
    ax.set_title("K-Means Clusters - PCA 2D Projection (k=4)")
    ax.legend(frameon=True, edgecolor="#cccccc")

    fig.suptitle("Unsupervised Segmentation of Chemical Companies",
                 fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
    path = f"{PLOTS_DIR}/03_kmeans_clustering.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved: {path}")

    df[["Company","Ticker","Sub_Sector","Cluster",
        "Rev_CAGR_5Y","Avg_EBITDA","Avg_PAT","Avg_ROE"]].to_csv(
        f"{DATA_DIR}/cluster_assignments.csv", index=False)
    return df


# PLOT 4 - Boxplot distributions
def plot_margin_boxplot():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    for ax, cols, ylabel, title in [
        (axes[0], EBITDA_COLS, "EBITDA Margin (%)", "EBITDA Margin Distribution by Sub-sector"),
        (axes[1], ROE_COLS,    "Return on Equity (%)","ROE Distribution by Sub-sector"),
    ]:
        data   = [df[df["Sub_Sector"]==ss][cols].values.flatten() for ss in SUBSECTORS]
        colors = [SCOL[ss] for ss in SUBSECTORS]
        
        bp = ax.boxplot(data, patch_artist=True,
                        medianprops=dict(color="white", linewidth=2),
                        whiskerprops=dict(linewidth=1.2, color="#888888"),
                        capprops=dict(linewidth=1.2, color="#888888"),
                        flierprops=dict(marker="o", markersize=3.5,
                                        markerfacecolor="#aaaaaa", alpha=0.5))
        for patch, col in zip(bp["boxes"], colors):
            patch.set_facecolor(col)
            patch.set_alpha(0.7)
        
        ax.set_xticklabels(SUBSECTORS, fontsize=8.5)
        ax.set_ylabel(ylabel)
        ax.set_title(title)

        # median text
        for i, d in enumerate(data):
            med = np.median(d)
            ax.text(i+1, med+0.5, f"{med:.1f}%",
                    ha="center", fontsize=8, fontweight="bold",
                    color=colors[i])

    fig.suptitle("Profitability Distributions - FY2019 to FY2024 (all company-year observations)",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    path = f"{PLOTS_DIR}/04_margin_boxplot.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved: {path}")


# PLOT 5 - Revenue by sub-sector FY19 vs FY24
def plot_revenue_waterfall():
    agg = df.groupby("Sub_Sector").agg(
        R19=("Rev_FY19","sum"), R24=("Rev_FY24","sum")).reset_index()
    agg["cagr"] = ((agg["R24"]/agg["R19"])**(1/5)-1)*100
    agg = agg.sort_values("R24")

    fig, ax = plt.subplots(figsize=(10, 5))
    y  = np.arange(len(agg))
    h  = 0.32
    
    ax.barh(y - h/2, agg["R19"], h,
            color=[SCOL[s] for s in agg["Sub_Sector"]],
            alpha=0.4, label="FY2019")
    ax.barh(y + h/2, agg["R24"], h,
            color=[SCOL[s] for s in agg["Sub_Sector"]],
            alpha=0.85, label="FY2024")

    ax.set_yticks(y)
    ax.set_yticklabels(agg["Sub_Sector"], fontsize=9)
    ax.set_xlabel("Aggregate Revenue (INR Crores)")
    ax.set_title("Sub-sector Aggregate Revenue: FY2019 vs FY2024\n(22 NSE-listed companies)")

    for i, (_, row) in enumerate(agg.iterrows()):
        ax.text(row["R24"]+200, i+h/2,
                f"  {row['cagr']:.1f}% CAGR",
                va="center", fontsize=8.5, color=SCOL[row["Sub_Sector"]],
                fontweight="bold")

    ax.legend(frameon=True, edgecolor="#cccccc")
    fig.tight_layout()
    path = f"{PLOTS_DIR}/05_revenue_waterfall.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved: {path}")


# PLOT 6 - Correlation heatmap
def plot_correlation_heatmap():
    cols = ["Rev_CAGR_5Y","Avg_EBITDA","Avg_PAT","Avg_ROE",
            "Export_Pct","RnD_Pct","Capex_Pct"]
    labels = ["Rev CAGR (5Y)","Avg EBITDA","Avg PAT","Avg ROE",
              "Export %","R&D %","Capex %"]
    corr = df[cols].corr()

    # simple diverging colormap - red to white to blue
    cmap = LinearSegmentedColormap.from_list("div",
           ["#b85450","#f5f5f5","#3a6b9e"])

    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1, aspect="auto")

    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(labels, rotation=38, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)

    for i in range(len(cols)):
        for j in range(len(cols)):
            v = corr.values[i,j]
            tc = "white" if abs(v) > 0.55 else "#333333"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=8.5, color=tc)

    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03,
                 label="Pearson r")
    ax.set_title("Correlation Matrix - Financial Metrics (22 companies)")
    fig.tight_layout()
    path = f"{PLOTS_DIR}/06_correlation_heatmap.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved: {path}")


# PLOT 7 - Export intensity vs EBITDA
def plot_export_vs_margin():
    fig, ax = plt.subplots(figsize=(10, 6))

    for ss in SUBSECTORS:
        sub = df[df["Sub_Sector"]==ss]
        ax.scatter(sub["Export_Pct"], sub["Avg_EBITDA"],
                   color=SCOL[ss], s=70, alpha=0.75,
                   edgecolors="white", linewidth=0.7,
                   label=ss, zorder=3)
        for _, row in sub.iterrows():
            ax.annotate(row["Ticker"],
                        (row["Export_Pct"], row["Avg_EBITDA"]),
                        textcoords="offset points", xytext=(5,2),
                        fontsize=7.5, color=SCOL[ss])

    # trendline
    z = np.polyfit(df["Export_Pct"], df["Avg_EBITDA"], 1)
    xl = np.linspace(df["Export_Pct"].min(), df["Export_Pct"].max(), 100)
    ax.plot(xl, np.poly1d(z)(xl), "--", color="#aaaaaa", lw=1.2,
            label=f"Linear trend")

    r = df["Export_Pct"].corr(df["Avg_EBITDA"])
    ax.text(0.05, 0.93, f"r = {r:.2f}", transform=ax.transAxes,
            fontsize=10, color="#333333",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#cccccc"))

    ax.set_xlabel("Export Revenue as % of Total Revenue (FY2024)")
    ax.set_ylabel("Average EBITDA Margin, FY2019-FY2024 (%)")
    ax.set_title("Export Orientation vs Profitability")
    ax.legend(frameon=True, edgecolor="#cccccc")
    fig.tight_layout()
    path = f"{PLOTS_DIR}/07_export_vs_margin.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved: {path}")


# PLOT 8 - Composite score ranking
def plot_composite_ranking():
    df2 = df.copy()
    for col in ["Avg_EBITDA","Avg_ROE","Rev_CAGR_5Y","RnD_Pct"]:
        mn, mx = df2[col].min(), df2[col].max()
        df2[col+"_n"] = (df2[col]-mn)/(mx-mn)*100
    df2["Score"] = (0.40*df2["Avg_EBITDA_n"] + 0.30*df2["Avg_ROE_n"] +
                    0.20*df2["Rev_CAGR_5Y_n"] + 0.10*df2["RnD_Pct_n"])
    top = df2.nlargest(10,"Score").sort_values("Score")

    fig, ax = plt.subplots(figsize=(10,5.5))
    bcolors = [SCOL[ss] for ss in top["Sub_Sector"]]
    bars = ax.barh(top["Ticker"], top["Score"],
                   color=bcolors, alpha=0.78, edgecolor="white")

    for bar, (_, row) in zip(bars, top.iterrows()):
        w = bar.get_width()
        ax.text(w+0.4, bar.get_y()+bar.get_height()/2,
                f"  {w:.1f}  ({row['Sub_Sector']})",
                va="center", fontsize=8.2)

    ax.set_xlabel("Composite Score (0 to 100)\n[40% EBITDA + 30% ROE + 20% Revenue CAGR + 10% R&D spend]")
    ax.set_title("Top 10 Companies by Composite Financial Score")
    legend_patches = [mpatches.Patch(color=SCOL[ss], label=ss, alpha=0.78)
                      for ss in SUBSECTORS]
    ax.legend(handles=legend_patches, frameon=True, edgecolor="#cccccc")
    fig.tight_layout()
    path = f"{PLOTS_DIR}/08_composite_ranking.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved: {path}")
    return df2


# PLOT 9 - R&D vs Capex
def plot_rnd_capex():
    fig, ax = plt.subplots(figsize=(10, 6))

    for ss in SUBSECTORS:
        sub = df[df["Sub_Sector"]==ss]
        # size = ebitda margin
        sizes = sub["Avg_EBITDA"] * 7
        ax.scatter(sub["RnD_Pct"], sub["Capex_Pct"],
                   s=sizes, color=SCOL[ss], alpha=0.65,
                   edgecolors="white", linewidth=0.7,
                   label=ss, zorder=3)
        for _, row in sub.iterrows():
            ax.annotate(row["Ticker"],
                        (row["RnD_Pct"], row["Capex_Pct"]),
                        textcoords="offset points", xytext=(5,2),
                        fontsize=7.5, color=SCOL[ss])

    ax.axvline(df["RnD_Pct"].mean(), color="#aaaaaa", ls="--", lw=0.8)
    ax.axhline(df["Capex_Pct"].mean(), color="#aaaaaa", ls="--", lw=0.8)

    ax.text(0.97, 0.04, "Bubble size = avg EBITDA margin",
            transform=ax.transAxes, fontsize=7.5, ha="right",
            color="#888888", style="italic")

    ax.set_xlabel("R&D Spend as % of Revenue (FY2024)")
    ax.set_ylabel("Average Capex as % of Revenue (FY2022-FY2024)")
    ax.set_title("R&D and Capex Investment Intensity\n(bubble size proportional to profitability)")
    ax.legend(frameon=True, edgecolor="#cccccc")
    fig.tight_layout()
    path = f"{PLOTS_DIR}/09_rnd_vs_capex.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved: {path}")


# PLOT 10 - EBITDA heatmap company x year
def plot_ebitda_heatmap():
    dfs = df.sort_values(["Sub_Sector","Avg_EBITDA"],
                          ascending=[True,False])
    hdata   = dfs[EBITDA_COLS].values
    ylabels = dfs["Ticker"].tolist()

    cmap = LinearSegmentedColormap.from_list("g",
           ["#f7fbf7","#74c476","#006d2c"])

    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(hdata, cmap=cmap, aspect="auto", vmin=5, vmax=44)

    ax.set_xticks(range(6))
    ax.set_yticks(range(len(ylabels)))
    ax.set_xticklabels(FYEARS, fontsize=10)
    ax.set_yticklabels(ylabels, fontsize=8.5)

    for i in range(len(ylabels)):
        for j in range(6):
            v = hdata[i,j]
            tc = "white" if v > 28 else "#1a1a1a"
            ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                    fontsize=7.8, color=tc)

    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02,
                 label="EBITDA Margin (%)")

    # sub-sector separators
    ss_list = dfs["Sub_Sector"].tolist()
    for i in range(1, len(ss_list)):
        if ss_list[i] != ss_list[i-1]:
            ax.axhline(i-0.5, color="white", lw=2.2)

    ax.set_title("EBITDA Margins Heatmap - All Companies, FY2019 to FY2024 (%)")
    fig.tight_layout()
    path = f"{PLOTS_DIR}/10_ebitda_heatmap.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved: {path}")


if __name__ == "__main__":
    print("Running analysis...\n")
    plot_ebitda_trend()
    plot_cagr_vs_ebitda()
    plot_kmeans_clusters()
    plot_margin_boxplot()
    plot_revenue_waterfall()
    plot_correlation_heatmap()
    plot_export_vs_margin()
    plot_composite_ranking()
    plot_rnd_capex()
    plot_ebitda_heatmap()
    print("\nDone. All plots saved.")
