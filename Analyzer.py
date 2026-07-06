import os
import re
import csv
import html
import time
import zipfile
import warnings
import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest, RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, PageBreak, Spacer, Table, TableStyle
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor, white
from reportlab.lib.pagesizes import A4
from reportlab.lib.enums import TA_CENTER

warnings.filterwarnings("ignore")
plt.switch_backend("Agg")

try:
    plt.style.use("seaborn-v0_8-whitegrid")
except:
    plt.style.use("ggplot")

start = time.time()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_BASE = BASE_DIR
OUT = os.path.join(BASE_DIR, "Output")
CHART = os.path.join(OUT, "charts")
os.makedirs(CHART, exist_ok=True)

def clean_name(x):
    x = re.sub(r"[^0-9a-zA-Z]+", "_", str(x).strip().lower())
    return re.sub(r"_+", "_", x).strip("_")

def E(x):
    if pd.isna(x):
        return "N/A"
    return html.escape(str(x))

def fmt_int(x):
    try:
        return f"{int(round(float(x))):,}"
    except:
        return "N/A"

def fmt_num(x, d=3):
    try:
        return f"{float(x):,.{d}f}"
    except:
        return "N/A"

def fmt_pct(x):
    try:
        x = float(x)
        if abs(x) <= 1:
            x *= 100
        return f"{x:.1f}%"
    except:
        return "N/A"

def discover_files():
    files = []
    extract_root = os.path.join(BASE_DIR, "unzipped_input")
    os.makedirs(extract_root, exist_ok=True)

    for root, dirs, names in os.walk(INPUT_BASE):
        if os.path.abspath(root).startswith(os.path.abspath(OUT)):
            continue
        if os.path.abspath(root).startswith(os.path.abspath(extract_root)):
            continue
        for name in names:
            path = os.path.join(root, name)
            low = name.lower()
            if low.endswith((".csv", ".xlsx", ".xls", ".json", ".sqlite", ".db")):
                files.append(path)
            elif low.endswith(".zip"):
                target = os.path.join(extract_root, clean_name(os.path.splitext(name)[0]))
                os.makedirs(target, exist_ok=True)
                try:
                    with zipfile.ZipFile(path, "r") as z:
                        z.extractall(target)
                except:
                    pass

    for root, dirs, names in os.walk(extract_root):
        for name in names:
            path = os.path.join(root, name)
            if name.lower().endswith((".csv", ".xlsx", ".xls", ".json", ".sqlite", ".db")):
                files.append(path)

    return sorted(set(files))

def read_csv_flexible(path):
    for enc in ["utf-8-sig", "utf-8", "latin1"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except pd.errors.ParserError:
            with open(path, "r", encoding=enc, errors="replace", newline="") as f:
                rows = list(csv.reader(f))
            if not rows:
                return pd.DataFrame()
            header = rows[0]
            width = len(header)
            fixed = []
            for row in rows[1:]:
                if len(row) < width:
                    row = row + [""] * (width - len(row))
                elif len(row) > width:
                    row = row[:width - 1] + [",".join(row[width - 1:])]
                fixed.append(row)
            return pd.DataFrame(fixed, columns=header)
        except UnicodeDecodeError:
            pass
    return pd.read_csv(path, engine="python", on_bad_lines="skip")

def load_file(path):
    low = path.lower()
    if low.endswith(".csv"):
        return read_csv_flexible(path)
    if low.endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    if low.endswith(".json"):
        return pd.read_json(path)
    if low.endswith((".sqlite", ".db")):
        conn = sqlite3.connect(path)
        tables = pd.read_sql("select name from sqlite_master where type='table'", conn)["name"].tolist()
        best = pd.DataFrame()
        for t in tables:
            temp = pd.read_sql(f'select * from "{t}"', conn)
            if len(temp) > len(best):
                best = temp
        conn.close()
        return best
    return pd.DataFrame()

def find_dataset():
    files = discover_files()
    if not files:
        raise ValueError(f"No input files found under {INPUT_BASE}.")
    loaded = []
    for path in files:
        try:
            df = load_file(path)
            cols = [clean_name(c) for c in df.columns]
            if len(df) > 0 and "points" in cols and "price" in cols and "variety" in cols:
                loaded.append((path, df))
        except:
            pass
    if not loaded:
        raise ValueError("No readable wine review dataset found.")
    return max(loaded, key=lambda x: len(x[1]))

data_path, df = find_dataset()
df.columns = [clean_name(c) for c in df.columns]

if "unnamed_0" in df.columns:
    df = df.drop(columns=["unnamed_0"])

for c in ["points", "price"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

df = df.dropna(subset=["points"]).copy()
df["price"] = df["price"].fillna(df["price"].median())
df["value_score"] = df["points"] / np.log1p(df["price"])
df["premium_flag"] = (df["price"] >= df["price"].quantile(0.75)).astype(int)
df["high_rating_flag"] = (df["points"] >= 90).astype(int)

for c in df.select_dtypes(include=["object"]).columns:
    df[c] = df[c].fillna("Unknown").astype(str).str.strip()
    df.loc[df[c] == "", c] = "Unknown"

title_style = ParagraphStyle("title", fontSize=22, leading=28, alignment=TA_CENTER, textColor=HexColor("#7C2D12"), spaceAfter=18)
subtitle_style = ParagraphStyle("subtitle", fontSize=12, leading=16, alignment=TA_CENTER, textColor=HexColor("#334155"), spaceAfter=22)
heading_style = ParagraphStyle("heading", fontSize=16, leading=21, textColor=HexColor("#7C2D12"), spaceAfter=10)
subheading_style = ParagraphStyle("subheading", fontSize=12.5, leading=17, textColor=HexColor("#111827"), spaceAfter=7)
body_style = ParagraphStyle("body", fontSize=9.4, leading=13.5, textColor=HexColor("#1F2937"), spaceAfter=8)
small_style = ParagraphStyle("small", fontSize=7.4, leading=9.1, textColor=HexColor("#1F2937"))
header_style = ParagraphStyle("header", fontSize=7.4, leading=9.1, textColor=white)

report_path = os.path.join(OUT, "Wine_Reviews_Analytics_Report.pdf")
doc = SimpleDocTemplate(report_path, pagesize=A4, leftMargin=36, rightMargin=36, topMargin=38, bottomMargin=38)
elements = []
chart_no = 0

def add_heading(x):
    elements.append(Paragraph(E(x), heading_style))

def add_body(x):
    elements.append(Paragraph(x, body_style))

def add_table(title, rows, headers, widths=None, max_rows=20):
    if rows is None or len(rows) == 0:
        return
    elements.append(Paragraph(E(title), subheading_style))
    rows = rows[:max_rows]
    data = [[Paragraph(f"<b>{E(h)}</b>", header_style) for h in headers]]
    for row in rows:
        data.append([Paragraph(E(v), small_style) for v in row])
    if widths is None:
        widths = [6.9 * inch / len(headers)] * len(headers)
    table = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#7C2D12")),
        ("GRID", (0, 0), (-1, -1), 0.25, HexColor("#CBD5E1")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor("#FFF7ED"), white])
    ]))
    elements.append(table)
    elements.append(Spacer(1, 0.1 * inch))

def add_chart(title, fig, insight):
    global chart_no
    chart_no += 1
    path = os.path.join(CHART, f"{chart_no:02d}_{clean_name(title)[:70]}.png")
    try:
        fig.tight_layout()
    except:
        pass
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    elements.append(Paragraph(E(title), subheading_style))
    elements.append(Image(path, width=6.55 * inch, height=3.7 * inch))
    elements.append(Paragraph(insight, body_style))
    elements.append(Spacer(1, 0.09 * inch))

elements.append(Paragraph("Wine Reviews Dataset Analytics Report", title_style))
elements.append(Paragraph("Automated analysis for wine ratings, price patterns, countries, varieties, wineries, value scoring, predictive modeling, clustering and anomaly detection", subtitle_style))
add_body(f"This report analyzes the uploaded dataset <b>{E(os.path.basename(data_path))}</b>. It contains <b>{fmt_int(len(df))}</b> wine reviews and <b>{fmt_int(len(df.columns))}</b> columns after preprocessing.")

summary_rows = [
    ["Rows", fmt_int(len(df)), "Total usable wine reviews"],
    ["Columns", fmt_int(len(df.columns)), "Total available fields after preprocessing"],
    ["Average points", fmt_num(df["points"].mean(), 2), "Mean wine review score"],
    ["Median price", fmt_num(df["price"].median(), 2), "Median bottle price"],
    ["Average price", fmt_num(df["price"].mean(), 2), "Mean bottle price"],
    ["High-rated wines", fmt_int(df["high_rating_flag"].sum()), "Reviews with points greater than or equal to 90"],
    ["High-rated share", fmt_pct(df["high_rating_flag"].mean()), "Share of wines rated at least 90 points"],
    ["Premium price threshold", fmt_num(df["price"].quantile(0.75), 2), "75th percentile price cutoff"]
]

for c in ["country", "province", "variety", "winery", "taster_name"]:
    if c in df.columns:
        summary_rows.append([f"Unique {c}", fmt_int(df[c].nunique()), f"Distinct values in {c}"])

add_table("Executive Summary Metrics", summary_rows, ["Metric", "Value", "Interpretation"], [2.0 * inch, 1.5 * inch, 3.4 * inch], 15)

missing_rows = [[c, fmt_int(df[c].isna().sum()), fmt_pct(df[c].isna().mean())] for c in df.columns if df[c].isna().sum() > 0]
add_table("Missing Value Summary", missing_rows, ["Column", "Missing Count", "Missing Percent"], [2.8 * inch, 1.5 * inch, 1.5 * inch], 20)

elements.append(PageBreak())
add_heading("Rating and Price Distribution")

fig, ax = plt.subplots(figsize=(8, 4.8))
ax.hist(df["points"].dropna(), bins=25, color="#7C2D12", alpha=0.85)
ax.set_title("Wine Points Distribution")
ax.set_xlabel("Points")
ax.set_ylabel("Reviews")
add_chart("Wine Points Distribution", fig, f"The average rating is <b>{fmt_num(df['points'].mean(), 2)}</b> points and the median rating is <b>{fmt_num(df['points'].median(), 2)}</b> points.")

fig, ax = plt.subplots(figsize=(8, 4.8))
price_clip = df["price"].clip(upper=df["price"].quantile(0.99))
ax.hist(price_clip.dropna(), bins=40, color="#B45309", alpha=0.85)
ax.set_title("Wine Price Distribution")
ax.set_xlabel("Price")
ax.set_ylabel("Reviews")
add_chart("Wine Price Distribution", fig, f"The median price is <b>{fmt_num(df['price'].median(), 2)}</b>, while the 99th percentile price is <b>{fmt_num(df['price'].quantile(0.99), 2)}</b>.")

fig, ax = plt.subplots(figsize=(8.4, 5))
sample_df = df.sample(min(len(df), 12000), random_state=42)
ax.scatter(sample_df["price"].clip(upper=df["price"].quantile(0.99)), sample_df["points"], alpha=0.28)
ax.set_title("Price vs Points")
ax.set_xlabel("Price")
ax.set_ylabel("Points")
add_chart("Price vs Points", fig, "This scatter plot compares wine price and review points after clipping extreme prices at the 99th percentile.")

if "country" in df.columns:
    top_country = df["country"].value_counts().head(15)
    fig, ax = plt.subplots(figsize=(8.8, 5))
    ax.barh(top_country.index[::-1], top_country.values[::-1], color="#7C2D12")
    ax.set_title("Top Countries by Review Count")
    ax.set_xlabel("Reviews")
    ax.tick_params(axis="y", labelsize=8)
    add_chart("Top Countries by Review Count", fig, f"The most represented country is <b>{E(top_country.index[0])}</b> with <b>{fmt_int(top_country.iloc[0])}</b> reviews.")

if "variety" in df.columns:
    top_variety = df["variety"].value_counts().head(15)
    fig, ax = plt.subplots(figsize=(8.8, 5))
    ax.barh(top_variety.index[::-1], top_variety.values[::-1], color="#B45309")
    ax.set_title("Top Wine Varieties by Review Count")
    ax.set_xlabel("Reviews")
    ax.tick_params(axis="y", labelsize=8)
    add_chart("Top Wine Varieties by Review Count", fig, f"The most reviewed variety is <b>{E(top_variety.index[0])}</b> with <b>{fmt_int(top_variety.iloc[0])}</b> reviews.")

elements.append(PageBreak())
add_heading("Country, Variety and Winery Performance")

if "country" in df.columns:
    country_stats = df.groupby("country").agg(reviews=("points", "size"), avg_points=("points", "mean"), avg_price=("price", "mean"), median_price=("price", "median"), high_rating_rate=("high_rating_flag", "mean"), avg_value_score=("value_score", "mean")).reset_index()
    country_stats = country_stats[country_stats["reviews"] >= max(10, len(df) * 0.001)].sort_values(["avg_points", "reviews"], ascending=False)
    rows = [[r["country"], fmt_int(r["reviews"]), fmt_num(r["avg_points"], 2), fmt_num(r["avg_price"], 2), fmt_num(r["median_price"], 2), fmt_pct(r["high_rating_rate"]), fmt_num(r["avg_value_score"], 3)] for _, r in country_stats.head(20).iterrows()]
    add_table("Top Countries by Average Rating", rows, ["Country", "Reviews", "Avg Points", "Avg Price", "Median Price", "High Rating Rate", "Value Score"], [1.2 * inch, 0.8 * inch, 0.9 * inch, 0.9 * inch, 0.9 * inch, 1.0 * inch, 1.0 * inch], 20)

if "variety" in df.columns:
    variety_stats = df.groupby("variety").agg(reviews=("points", "size"), avg_points=("points", "mean"), avg_price=("price", "mean"), high_rating_rate=("high_rating_flag", "mean"), avg_value_score=("value_score", "mean")).reset_index()
    variety_stats = variety_stats[variety_stats["reviews"] >= max(20, len(df) * 0.001)].sort_values(["avg_points", "reviews"], ascending=False)
    rows = [[r["variety"], fmt_int(r["reviews"]), fmt_num(r["avg_points"], 2), fmt_num(r["avg_price"], 2), fmt_pct(r["high_rating_rate"]), fmt_num(r["avg_value_score"], 3)] for _, r in variety_stats.head(20).iterrows()]
    add_table("Top Varieties by Average Rating", rows, ["Variety", "Reviews", "Avg Points", "Avg Price", "High Rating Rate", "Value Score"], [1.8 * inch, 0.8 * inch, 0.9 * inch, 0.9 * inch, 1.0 * inch, 1.0 * inch], 20)

if "winery" in df.columns:
    winery_stats = df.groupby("winery").agg(reviews=("points", "size"), avg_points=("points", "mean"), avg_price=("price", "mean"), high_rating_rate=("high_rating_flag", "mean"), avg_value_score=("value_score", "mean")).reset_index()
    winery_stats = winery_stats[winery_stats["reviews"] >= max(5, len(df) * 0.0005)].sort_values(["avg_points", "reviews"], ascending=False)
    rows = [[r["winery"], fmt_int(r["reviews"]), fmt_num(r["avg_points"], 2), fmt_num(r["avg_price"], 2), fmt_pct(r["high_rating_rate"]), fmt_num(r["avg_value_score"], 3)] for _, r in winery_stats.head(20).iterrows()]
    add_table("Top Wineries by Average Rating", rows, ["Winery", "Reviews", "Avg Points", "Avg Price", "High Rating Rate", "Value Score"], [1.8 * inch, 0.8 * inch, 0.9 * inch, 0.9 * inch, 1.0 * inch, 1.0 * inch], 20)

fig, ax = plt.subplots(figsize=(8.8, 5))
top_value = df.groupby("variety")["value_score"].mean().sort_values(ascending=False).head(15) if "variety" in df.columns else df.groupby("country")["value_score"].mean().sort_values(ascending=False).head(15)
ax.barh(top_value.index[::-1], top_value.values[::-1], color="#7C2D12")
ax.set_title("Top Average Value Score Segments")
ax.set_xlabel("Value Score")
ax.tick_params(axis="y", labelsize=8)
add_chart("Top Average Value Score Segments", fig, "Value score is calculated as points divided by the logarithm of price plus one.")

elements.append(PageBreak())
add_heading("Predictive Modeling")

model_df = df.copy()
categorical_candidates = [c for c in ["country", "province", "region_1", "region_2", "variety", "winery", "taster_name"] if c in model_df.columns]
feature_parts = [model_df[["price", "value_score", "premium_flag", "high_rating_flag"]].drop(columns=["high_rating_flag"], errors="ignore")]

for c in categorical_candidates:
    top = model_df[c].value_counts().head(30).index
    temp = pd.get_dummies(model_df[c].where(model_df[c].isin(top), "Other"), prefix=c, dummy_na=False)
    feature_parts.append(temp)

X = pd.concat(feature_parts, axis=1).replace([np.inf, -np.inf], np.nan).fillna(0)
y = model_df["points"]

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.22, random_state=42)

regressors = [
    ("Random Forest Regressor", RandomForestRegressor(n_estimators=220, random_state=42, min_samples_leaf=2, n_jobs=-1)),
    ("Gradient Boosting Regressor", GradientBoostingRegressor(random_state=42))
]

reg_rows = []
best_reg = None
best_rmse = None
best_r2 = None
best_reg_name = ""

for name, model in regressors:
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    rmse = np.sqrt(mean_squared_error(yte, pred))
    r2 = r2_score(yte, pred)
    reg_rows.append([name, fmt_num(rmse, 4), fmt_num(r2, 4)])
    if best_rmse is None or rmse < best_rmse:
        best_rmse = rmse
        best_r2 = r2
        best_reg = model
        best_reg_name = name

add_table("Wine Points Regression Performance", reg_rows, ["Model", "RMSE", "R2"], [2.8 * inch, 1.4 * inch, 1.4 * inch], 5)

if hasattr(best_reg, "feature_importances_"):
    imp = pd.Series(best_reg.feature_importances_, index=X.columns).sort_values(ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.barh(imp.index[::-1], imp.values[::-1], color="#7C2D12")
    ax.set_title("Wine Points Prediction Feature Importance")
    ax.set_xlabel("Importance")
    ax.tick_params(axis="y", labelsize=7.5)
    add_chart("Wine Points Prediction Feature Importance", fig, f"The best model is <b>{E(best_reg_name)}</b> with RMSE <b>{fmt_num(best_rmse, 4)}</b> and R2 <b>{fmt_num(best_r2, 4)}</b>.")

elements.append(PageBreak())
add_heading("Clustering and Anomaly Detection")

cluster_cols = ["points", "price", "value_score", "premium_flag", "high_rating_flag"]
Xc = df[cluster_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
Xcs = StandardScaler().fit_transform(Xc)

k = 4
km = KMeans(n_clusters=k, random_state=42, n_init=20)
df["cluster"] = km.fit_predict(Xcs)

cluster_summary = df.groupby("cluster").agg(records=("points", "size"), avg_points=("points", "mean"), avg_price=("price", "mean"), high_rating_rate=("high_rating_flag", "mean"), avg_value_score=("value_score", "mean")).reset_index()
rows = [[r["cluster"], fmt_int(r["records"]), fmt_num(r["avg_points"], 2), fmt_num(r["avg_price"], 2), fmt_pct(r["high_rating_rate"]), fmt_num(r["avg_value_score"], 3)] for _, r in cluster_summary.iterrows()]
add_table("Wine Review Cluster Summary", rows, ["Cluster", "Records", "Avg Points", "Avg Price", "High Rating Rate", "Value Score"], [0.8 * inch, 0.9 * inch, 0.9 * inch, 0.9 * inch, 1.1 * inch, 1.0 * inch], 10)

fig, ax = plt.subplots(figsize=(7.5, 4.8))
cluster_counts = df["cluster"].value_counts().sort_index()
ax.bar(cluster_counts.index.astype(str), cluster_counts.values, color="#B45309")
ax.set_title("Cluster Size Distribution")
ax.set_xlabel("Cluster")
ax.set_ylabel("Reviews")
add_chart("Wine Cluster Size Distribution", fig, f"KMeans grouped wine reviews into <b>{k}</b> clusters based on points, price and value-score patterns.")

fig, ax = plt.subplots(figsize=(8.4, 5))
plot_df = df.sample(min(len(df), 12000), random_state=42)
ax.scatter(plot_df["price"].clip(upper=df["price"].quantile(0.99)), plot_df["points"], c=plot_df["cluster"], alpha=0.35)
ax.set_title("Cluster Separation Using Price and Points")
ax.set_xlabel("Price")
ax.set_ylabel("Points")
add_chart("Cluster Separation Using Price and Points", fig, "This chart visualizes wine-review clusters using price and points.")

contamination = min(0.05, max(0.01, 120 / len(df)))
iso = IsolationForest(contamination=contamination, random_state=42)
df["anomaly_flag"] = iso.fit_predict(Xcs)
anomaly_count = int((df["anomaly_flag"] == -1).sum())

fig, ax = plt.subplots(figsize=(7.5, 4.8))
ax.bar(["Regular", "Anomalous"], [len(df) - anomaly_count, anomaly_count], color=["#15803D", "#B91C1C"])
ax.set_title("Isolation Forest Anomaly Detection")
ax.set_ylabel("Reviews")
add_chart("Anomaly Detection Summary", fig, f"Isolation Forest flagged <b>{fmt_int(anomaly_count)}</b> unusual wine reviews based on rating, price and value-score behavior.")

sample_cols = [c for c in ["country", "province", "variety", "winery", "points", "price", "value_score"] if c in df.columns]
anomaly_rows = df.loc[df["anomaly_flag"] == -1, sample_cols].head(15).values.tolist()
add_table("Sample Potentially Anomalous Wine Reviews", anomaly_rows, sample_cols, None, 15)

elements.append(PageBreak())
add_heading("Final Analytical Summary")

top_country_text = df["country"].value_counts().index[0] if "country" in df.columns else "N/A"
top_variety_text = df["variety"].value_counts().index[0] if "variety" in df.columns else "N/A"

final_lines = [
    f"The dataset contains <b>{fmt_int(len(df))}</b> valid wine reviews.",
    f"The average wine rating is <b>{fmt_num(df['points'].mean(), 2)}</b> points.",
    f"The median wine price is <b>{fmt_num(df['price'].median(), 2)}</b>.",
    f"The most represented country is <b>{E(top_country_text)}</b>.",
    f"The most represented variety is <b>{E(top_variety_text)}</b>.",
    f"The best points-prediction model is <b>{E(best_reg_name)}</b> with RMSE <b>{fmt_num(best_rmse, 4)}</b> and R2 <b>{fmt_num(best_r2, 4)}</b>.",
    f"The clustering section separates wines into <b>{k}</b> price-rating-value groups.",
    f"The anomaly-detection section flags <b>{fmt_int(anomaly_count)}</b> unusual records for closer review.",
    f"The report generated <b>{fmt_int(chart_no)}</b> charts and saved all outputs in <b>{E(OUT)}</b>."
]

add_body("<br/><br/>".join(final_lines))

doc.build(elements)

processed_csv_path = os.path.join(OUT, "processed_wine_reviews_dataset.csv")
df.to_csv(processed_csv_path, index=False)

print("Input File:", data_path)
print("Rows Processed:", len(df))
print("Charts Generated:", chart_no)
print("PDF Report Generated:", report_path)
print("Processed CSV Generated:", processed_csv_path)
print("Execution Time:", round(time.time() - start, 2), "seconds")