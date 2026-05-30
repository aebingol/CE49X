from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import Markdown, display
from matplotlib.ticker import FuncFormatter


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
EARTHQUAKE_QUERY_URL = (
    "https://earthquake.usgs.gov/fdsnws/event/1/query"
    "?format=csv&starttime=1900-01-01&endtime=2026-03-24"
    "&minlatitude=40&maxlatitude=41.5&minlongitude=27&maxlongitude=31"
    "&minmagnitude=2&orderby=time-asc"
)
DISTRICT_BOOKLETS_URL = (
    "https://depremzemin.ibb.istanbul/tr/"
    "olasi-deprem-kayip-tahminleri-ilce-kitapciklari"
)
VULNERABILITY_URL = (
    "https://depremzemin.ibb.istanbul/uploads/"
    "prefix-afetler-karsisinda-sosyal-hasar-gorebilirlik-yonetici-ozeti-66866b8952173.pdf"
)


def _apply_style():
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update(
        {
            "axes.titlesize": 16,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.dpi": 120,
        }
    )


def _comma_formatter(decimals=0):
    def _fmt(value, _pos):
        return f"{value:,.{decimals}f}"

    return FuncFormatter(_fmt)


def _top3_sentence(values):
    names = list(values)
    if len(names) < 3:
        return ", ".join(names)
    return f"{names[0]}, {names[1]}, and {names[2]}"


def _build_dataset_docs(earthquakes_df, building_df, vulnerability_df):
    return {
        "earthquakes": {
            "title": "Dataset 1: Earthquake Catalog",
            "source_url": EARTHQUAKE_QUERY_URL,
            "date_accessed": "2026-03-24",
            "method": (
                "Downloaded CSV from the official USGS earthquake API using a Marmara "
                "bounding box (40 to 41.5 N, 27 to 31 E), 1900-01-01 to 2026-03-24, "
                "minimum magnitude 2.0."
            ),
            "records": len(earthquakes_df),
            "cleaning": (
                "Parsed UTC timestamps, converted latitude/longitude/depth/magnitude to "
                "numeric types, dropped rows missing any core seismic variable, and added "
                "a year column for time-based analysis."
            ),
        },
        "buildings": {
            "title": "Dataset 2: Building / Population Data",
            "source_url": DISTRICT_BOOKLETS_URL,
            "date_accessed": "2026-03-24",
            "method": (
                "Downloaded 11 official IBB district earthquake-loss booklets and extracted "
                "the district population, total analyzed building stock, and Mw 7.5 damage "
                "estimates into a single district table."
            ),
            "records": len(building_df),
            "cleaning": (
                "Standardized district names, converted all count fields to integers, and "
                "added severe- and moderate-plus-damage totals derived from the booklet "
                "damage columns."
            ),
        },
        "vulnerability": {
            "title": "Dataset 3: Seismic Vulnerability Data",
            "source_url": VULNERABILITY_URL,
            "date_accessed": "2026-03-24",
            "method": (
                "Extracted district-level social vulnerability scores for the same 11 "
                "districts from the official IBB executive-summary PDF; added approximate "
                "district centroids to support a district-level risk bubble map."
            ),
            "records": len(vulnerability_df),
            "cleaning": (
                "Converted vulnerability scores and centroid coordinates to numeric values "
                "and grouped the scores into high, upper-mid, and lower-mid vulnerability "
                "classes for easier comparison."
            ),
        },
    }


def load_lab04_data():
    _apply_style()

    earthquakes_df = pd.read_csv(DATA_DIR / "usgs_marmara_earthquakes.csv")
    earthquakes_df["time"] = pd.to_datetime(earthquakes_df["time"], utc=True)
    earthquakes_df["year"] = earthquakes_df["time"].dt.year
    for col in ["latitude", "longitude", "depth", "mag"]:
        earthquakes_df[col] = pd.to_numeric(earthquakes_df[col], errors="coerce")
    earthquakes_df = (
        earthquakes_df.dropna(subset=["latitude", "longitude", "depth", "mag"])
        .sort_values("time")
        .reset_index(drop=True)
    )

    building_df = pd.read_csv(DATA_DIR / "district_building_population.csv")
    vulnerability_df = pd.read_csv(DATA_DIR / "district_vulnerability.csv")

    count_columns = [
        "population",
        "total_buildings",
        "very_heavy_damage",
        "heavy_damage",
        "medium_damage",
        "light_damage",
    ]
    building_df[count_columns] = building_df[count_columns].apply(pd.to_numeric)
    vulnerability_df[["vulnerability_score", "latitude", "longitude"]] = vulnerability_df[
        ["vulnerability_score", "latitude", "longitude"]
    ].apply(pd.to_numeric)

    building_df["severe_damage"] = (
        building_df["very_heavy_damage"] + building_df["heavy_damage"]
    )
    building_df["moderate_plus_damage"] = (
        building_df["severe_damage"] + building_df["medium_damage"]
    )
    building_df["severe_damage_rate"] = (
        building_df["severe_damage"] / building_df["total_buildings"]
    )
    building_df["moderate_plus_rate"] = (
        building_df["moderate_plus_damage"] / building_df["total_buildings"]
    )

    district_df = building_df.merge(vulnerability_df, on="district", validate="one_to_one")
    district_df["risk_index"] = (
        district_df["population"]
        * district_df["moderate_plus_rate"]
        * district_df["vulnerability_score"]
        / 100_000
    )
    district_df["damage_per_1000_buildings"] = district_df["severe_damage_rate"] * 1000
    district_df = district_df.sort_values("risk_index", ascending=False).reset_index(drop=True)

    dataset_docs = _build_dataset_docs(earthquakes_df, building_df, vulnerability_df)
    return earthquakes_df, building_df, vulnerability_df, district_df, dataset_docs


def show_dataset_summary(dataset_key, df, dataset_docs):
    doc = dataset_docs[dataset_key]
    display(
        Markdown(
            f"**{doc['title']}**  \n"
            f"**Source URL:** [{doc['source_url']}]({doc['source_url']})  \n"
            f"**Date accessed:** {doc['date_accessed']}  \n"
            f"**How obtained:** {doc['method']}  \n"
            f"**Number of records:** {doc['records']}  \n"
            f"**Cleaning steps:** {doc['cleaning']}"
        )
    )
    display(df.head())
    numeric_df = df.select_dtypes(include="number")
    display(numeric_df.describe().round(2))


def _draw_timeline(ax, earthquakes_df, add_colorbar=True):
    m6_count = int((earthquakes_df["mag"] >= 6.0).sum())
    sizes = np.clip((earthquakes_df["mag"] - 1.5) ** 3 * 14, 18, 520)
    scatter = ax.scatter(
        earthquakes_df["time"],
        earthquakes_df["mag"],
        s=sizes,
        c=earthquakes_df["depth"],
        cmap="viridis",
        alpha=0.78,
        edgecolor="white",
        linewidth=0.35,
    )
    if add_colorbar:
        cbar = ax.figure.colorbar(scatter, ax=ax, pad=0.01)
        cbar.set_label("Depth (km)")

    annotations = [
        ("1912-08-09", "1912 Mürefte\nM7.23", 2_200, 0.35),
        ("1953-03-18", "1953 Gönen\nM7.30", 1_600, -0.55),
        ("1999-08-17", "1999 İzmit\nM7.60", -3_600, -0.65),
    ]
    for event_date, label, day_offset, mag_offset in annotations:
        event_ts = pd.Timestamp(event_date, tz="UTC")
        row = earthquakes_df.iloc[(earthquakes_df["time"] - event_ts).abs().argmin()]
        ax.annotate(
            label,
            xy=(row["time"], row["mag"]),
            xytext=(row["time"] + pd.Timedelta(days=day_offset), row["mag"] + mag_offset),
            arrowprops={"arrowstyle": "->", "color": "#334155", "lw": 1},
            fontsize=10,
            bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#cbd5e1", "alpha": 0.92},
        )

    ax.set_title(
        f"The Marmara catalog contains {m6_count} earthquakes above M6.0 since 1900, "
        "with the largest shocks concentrated on the region's eastern and western margins"
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Magnitude")
    ax.grid(alpha=0.25)
    return scatter


def plot_seismicity_timeline(earthquakes_df):
    fig, ax = plt.subplots(figsize=(14, 6.5))
    _draw_timeline(ax, earthquakes_df, add_colorbar=True)
    fig.tight_layout()
    plt.show()
    return fig


def plot_magnitude_frequency(earthquakes_df):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8))

    # 0.2-magnitude bins match the catalog precision while keeping the distribution readable.
    bins = np.arange(
        np.floor(earthquakes_df["mag"].min() * 5) / 5,
        np.ceil(earthquakes_df["mag"].max() * 5) / 5 + 0.2,
        0.2,
    )
    sns.histplot(
        earthquakes_df["mag"],
        bins=bins,
        kde=True,
        color="#2563eb",
        edgecolor="white",
        ax=axes[0],
    )
    mean_mag = earthquakes_df["mag"].mean()
    median_mag = earthquakes_df["mag"].median()
    axes[0].axvline(mean_mag, color="#dc2626", linestyle="--", linewidth=2, label=f"Mean = {mean_mag:.2f}")
    axes[0].axvline(
        median_mag,
        color="#0f766e",
        linestyle="-.",
        linewidth=2,
        label=f"Median = {median_mag:.2f}",
    )
    axes[0].set_title("Most Marmara earthquakes cluster below M4, with a long upper-magnitude tail")
    axes[0].set_xlabel("Magnitude")
    axes[0].set_ylabel("Event count")
    axes[0].legend(frameon=True)

    magnitude_bins = np.arange(2.0, earthquakes_df["mag"].max() + 0.1, 0.1)
    cumulative_counts = np.array([(earthquakes_df["mag"] >= mag).sum() for mag in magnitude_bins])
    valid_mask = cumulative_counts > 0
    mags = magnitude_bins[valid_mask]
    counts = cumulative_counts[valid_mask]
    fit_mask = (mags >= 3.0) & (mags <= 5.5)
    slope, intercept = np.polyfit(mags[fit_mask], np.log10(counts[fit_mask]), 1)
    b_value = -slope

    axes[1].plot(mags, np.log10(counts), marker="o", ms=4, lw=1.6, color="#1d4ed8", label="Observed")
    axes[1].plot(
        mags[fit_mask],
        slope * mags[fit_mask] + intercept,
        color="#dc2626",
        linewidth=2.2,
        label=f"Linear fit (M 3.0-5.5), b = {b_value:.2f}",
    )
    axes[1].set_title("The Gutenberg-Richter plot becomes close to linear once the catalog is log-transformed")
    axes[1].set_xlabel("Magnitude threshold")
    axes[1].set_ylabel("log10(cumulative count)")
    axes[1].legend(frameon=True)
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    plt.show()
    return fig


def plot_district_comparison(district_df):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))

    ranked = district_df.sort_values("vulnerability_score", ascending=False)
    highlight = set(ranked.head(3)["district"])
    palette = [
        "#b91c1c" if district in highlight else "#94a3b8"
        for district in ranked["district"]
    ]
    axes[0].bar(range(len(ranked)), ranked["vulnerability_score"], color=palette)
    axes[0].set_xticks(range(len(ranked)))
    axes[0].set_xticklabels(ranked["district"], rotation=45, ha="right")
    axes[0].set_title("Social vulnerability peaks in the fast-growing peripheral districts")
    axes[0].set_xlabel("District")
    axes[0].set_ylabel("IBB vulnerability score")

    scatter = axes[1].scatter(
        district_df["population"],
        district_df["severe_damage"],
        s=district_df["total_buildings"] / 48,
        c=district_df["damage_per_1000_buildings"],
        cmap="flare",
        alpha=0.9,
        edgecolor="white",
        linewidth=0.7,
    )
    for row in district_df.itertuples():
        axes[1].annotate(
            row.district,
            (row.population, row.severe_damage),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=9,
        )
    cbar = fig.colorbar(scatter, ax=axes[1], pad=0.01)
    cbar.set_label("Severe damage per 1,000 buildings")
    axes[1].set_title("Population alone understates districts with sharply higher expected severe damage")
    axes[1].set_xlabel("Population (2019 residents)")
    axes[1].set_ylabel("Expected severe damage (very heavy + heavy buildings)")
    axes[1].xaxis.set_major_formatter(_comma_formatter())
    axes[1].yaxis.set_major_formatter(_comma_formatter())
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    plt.show()
    return fig


def _draw_spatial_hazard(ax, district_df, add_colorbar=True):
    scatter = ax.scatter(
        district_df["longitude"],
        district_df["latitude"],
        s=district_df["moderate_plus_damage"] / 8,
        c=district_df["risk_index"],
        cmap="YlOrRd",
        alpha=0.92,
        edgecolor="#1f2937",
        linewidth=0.6,
        zorder=3,
    )
    ax.axhspan(40.94, 40.985, color="#dbeafe", alpha=0.65, zorder=0)
    ax.axvline(29.03, color="#0f766e", linestyle="--", linewidth=1.3, alpha=0.7, zorder=1)
    ax.text(28.68, 40.947, "Sea of Marmara", color="#1d4ed8", fontsize=11, weight="bold")
    ax.text(29.035, 41.065, "Bosphorus", color="#0f766e", fontsize=10, rotation=90, va="top")

    top_risk = set(district_df.nlargest(4, "risk_index")["district"])
    for row in district_df.itertuples():
        ax.annotate(
            row.district,
            (row.longitude, row.latitude),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=10 if row.district in top_risk else 8.5,
            weight="bold" if row.district in top_risk else "normal",
            color="#111827" if row.district in top_risk else "#334155",
        )

    if add_colorbar:
        cbar = ax.figure.colorbar(scatter, ax=ax, pad=0.01)
        cbar.set_label("Population-weighted damage pressure index")

    ax.set_title(
        "The western Marmara corridor and the dense historical core form the clearest "
        "district-level concentration of earthquake risk"
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xlim(28.66, 29.31)
    ax.set_ylim(40.94, 41.10)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.18)
    return scatter


def plot_spatial_hazard(district_df):
    fig, ax = plt.subplots(figsize=(11, 8))
    _draw_spatial_hazard(ax, district_df, add_colorbar=True)
    fig.tight_layout()
    plt.show()
    return fig


def plot_dashboard(earthquakes_df, district_df):
    top3 = _top3_sentence(district_df.nlargest(3, "risk_index")["district"])
    m6_count = int((earthquakes_df["mag"] >= 6.0).sum())

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 4, height_ratios=[1.05, 1.0, 1.15], hspace=0.45, wspace=0.4)

    ax_timeline = fig.add_subplot(gs[0, :])
    ax_hist = fig.add_subplot(gs[1, 0:2])
    ax_bar = fig.add_subplot(gs[1, 2:4])
    ax_map = fig.add_subplot(gs[2, 0:3])
    ax_text = fig.add_subplot(gs[2, 3])

    _draw_timeline(ax_timeline, earthquakes_df, add_colorbar=False)

    bins = np.arange(
        np.floor(earthquakes_df["mag"].min() * 5) / 5,
        np.ceil(earthquakes_df["mag"].max() * 5) / 5 + 0.2,
        0.2,
    )
    sns.histplot(
        earthquakes_df["mag"],
        bins=bins,
        kde=True,
        color="#2563eb",
        edgecolor="white",
        ax=ax_hist,
    )
    ax_hist.axvline(earthquakes_df["mag"].mean(), color="#dc2626", linestyle="--", linewidth=2)
    ax_hist.axvline(earthquakes_df["mag"].median(), color="#0f766e", linestyle="-.", linewidth=2)
    ax_hist.set_title("Magnitude distribution")
    ax_hist.set_xlabel("Magnitude")
    ax_hist.set_ylabel("Event count")

    ranked = district_df.nlargest(6, "risk_index").sort_values("risk_index")
    bar_colors = plt.cm.Reds_r(np.linspace(0.2, 0.85, len(ranked)))
    ax_bar.barh(ranked["district"], ranked["risk_index"], color=bar_colors)
    ax_bar.set_title("Top six districts by composite risk index")
    ax_bar.set_xlabel("Population-weighted damage pressure index")
    ax_bar.set_ylabel("")

    _draw_spatial_hazard(ax_map, district_df, add_colorbar=False)

    ax_text.axis("off")
    text = (
        "Executive takeaway\n"
        f"Top composite-risk districts: {top3}.\n\n"
        f"M6+ earthquakes since 1900: {m6_count}\n"
        f"Catalog size: {len(earthquakes_df):,} events\n"
        f"Highest vulnerability score: {district_df['vulnerability_score'].max():.2f}\n"
        f"Largest district population in sample: {district_df['population'].max():,}\n\n"
        "Derived metric\n"
        "Risk index = population x moderate-plus damage rate x vulnerability score / 100,000\n\n"
        "Interpretation\n"
        "This prioritizes districts where shaking consequences, exposed residents, and social "
        "fragility overlap."
    )
    ax_text.text(
        0.02,
        0.98,
        text,
        va="top",
        ha="left",
        fontsize=11,
        bbox={"boxstyle": "round,pad=0.45", "fc": "#f8fafc", "ec": "#cbd5e1"},
    )

    fig.suptitle(
        f"{top3} form the clearest overlap of hazard, exposure, and social vulnerability in this Istanbul sample",
        fontsize=18,
        y=0.98,
    )
    output_path = BASE_DIR / "dashboard.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    fig.subplots_adjust(top=0.90)
    plt.show()
    print(f"Dashboard saved to: {output_path}")
    return fig
