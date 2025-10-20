import os
import glob
import re
import math
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns

data_dir = r"D:\DataScienceLab\lab4\data"
out_dir = os.path.join(data_dir, "extended_analysis")
os.makedirs(out_dir, exist_ok=True)

def try_read_csv(path):
    for enc in ["utf-8", "gbk", "latin1", "cp1252"]:
        try:
            df = pd.read_csv(path, encoding=enc, low_memory=False)
            return df
        except Exception:
            continue
    return pd.read_csv(path, encoding="latin1", low_memory=False)

def parse_export_csv(path, subject):
    df = try_read_csv(path)
    df = df.dropna(how="all")
    cols = [str(c).strip() for c in df.columns]
    df.columns = cols
    if len(cols) >= 7:
        col0 = cols[0]
        col1 = cols[1]
        col2 = cols[2] if len(cols) > 2 else None
        col3 = cols[3] if len(cols) > 3 else None
        col4 = cols[4] if len(cols) > 4 else None
        col5 = cols[5] if len(cols) > 5 else None
        coln = cols[-1]
        df2 = pd.DataFrame()
        df2["rank"] = df[col0].astype(str).str.extract(r"(\d+)").astype(float)
        df2["institution"] = df[col1].astype(str).str.strip()
        df2["country"] = df[col2].astype(str).str.strip() if col2 else ""
        df2["documents"] = pd.to_numeric(df[col3].astype(str).str.replace(r"[^\d\-\.]", "", regex=True), errors="coerce")
        df2["cites"] = pd.to_numeric(df[col4].astype(str).str.replace(r"[^\d\-\.]", "", regex=True), errors="coerce")
        df2["cites_per_paper"] = pd.to_numeric(df[col5].astype(str).str.replace(r"[^\d\-\.]", "", regex=True), errors="coerce")
        df2["top_papers"] = pd.to_numeric(df[coln].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
        df2["subject"] = subject
        return df2
    else:
        return pd.DataFrame()

csv_paths = glob.glob(os.path.join(data_dir, "*.csv"))
frames = []
for p in csv_paths:
    subj = Path(p).stem
    parsed = parse_export_csv(p, subj)
    if not parsed.empty:
        frames.append(parsed)
if not frames:
    raise SystemExit("No parsed data")
df_all = pd.concat(frames, ignore_index=True)
df_all["institution"] = df_all["institution"].astype(str)
df_all["country"] = df_all["country"].astype(str)
df_all["subject"] = df_all["subject"].astype(str)
df_all["rank"] = pd.to_numeric(df_all["rank"], errors="coerce")
df_all["documents"] = pd.to_numeric(df_all["documents"], errors="coerce")
df_all["cites"] = pd.to_numeric(df_all["cites"], errors="coerce")
df_all["cites_per_paper"] = pd.to_numeric(df_all["cites_per_paper"], errors="coerce")
df_all["top_papers"] = pd.to_numeric(df_all["top_papers"], errors="coerce")
df_all = df_all.dropna(subset=["institution"]).reset_index(drop=True)
df_all["institution_norm"] = df_all["institution"].str.upper().str.replace(r"\s+", " ", regex=True).str.strip()

agg = df_all.groupby("institution_norm").agg(
    subjects_count=("subject", "nunique"),
    mean_rank=("rank", "mean"),
    median_rank=("rank", "median"),
    best_rank=("rank", "min"),
    worst_rank=("rank", "max"),
    mean_cites_per_paper=("cites_per_paper", "mean"),
    total_documents=("documents", "sum"),
    total_cites=("cites", "sum"),
    total_top_papers=("top_papers", "sum")
).reset_index()
agg = agg.fillna(0)
features = agg[["subjects_count", "mean_rank", "median_rank", "best_rank", "mean_cites_per_paper", "total_documents", "total_cites", "total_top_papers"]].copy()
features = features.replace([np.inf, -np.inf], np.nan).fillna(0)
scaler = StandardScaler()
X = scaler.fit_transform(features)
sil_scores = {}
for k in range(2,9):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    sc = silhouette_score(X, labels)
    sil_scores[k] = sc
best_k = max(sil_scores, key=sil_scores.get)
km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
labels = km.fit_predict(X)
agg["cluster"] = labels
cluster_centers = scaler.inverse_transform(km.cluster_centers_)
centers_df = pd.DataFrame(cluster_centers, columns=features.columns)
centers_df["cluster"] = range(best_k)
centers_df.to_csv(os.path.join(out_dir, "cluster_centers.csv"), index=False)
agg.to_csv(os.path.join(out_dir, "institution_aggregate.csv"), index=False)
sil_df = pd.DataFrame(list(sil_scores.items()), columns=["k", "silhouette"])
sil_df.to_csv(os.path.join(out_dir, "silhouette_scores.csv"), index=False)

def interpret_clusters(centers_df):
    rows = []
    for _, r in centers_df.iterrows():
        cid = int(r["cluster"])
        sc = r.to_dict()
        rows.append({"cluster": cid, "subjects_count": sc["subjects_count"], "mean_rank": sc["mean_rank"], "mean_cites_per_paper": sc["mean_cites_per_paper"], "total_documents": sc["total_documents"], "total_top_papers": sc["total_top_papers"]})
    return pd.DataFrame(rows)

try:
    cluster_summary = interpret_clusters(centers_df)
    cluster_summary.to_csv(os.path.join(out_dir, "cluster_summary.csv"), index=False)
except Exception:
    pass

ecnu_name_upper = "EAST CHINA NORMAL UNIVERSITY"
ecnu_rows = agg[agg["institution_norm"].str.contains(ecnu_name_upper)]
if ecnu_rows.empty:
    matched = agg[agg["institution_norm"].str.contains("EAST CHINA NORMAL") | agg["institution_norm"].str.contains("E.C.N.U") | agg["institution_norm"].str.contains("华东")]
    if not matched.empty:
        ecnu_rows = matched
if ecnu_rows.empty:
    raise SystemExit("ECNU not found in aggregated institutions")
ecnu_vector = ecnu_rows[["subjects_count", "mean_rank", "median_rank", "best_rank", "mean_cites_per_paper", "total_documents", "total_cites", "total_top_papers"]].values
X2 = scaler.transform(features)
sim = cosine_similarity(ecnu_vector, X2)[0]
agg["similarity_to_ecnu"] = sim
similar_sorted = agg.sort_values("similarity_to_ecnu", ascending=False)
top_similar = similar_sorted[similar_sorted["institution_norm"] != ecnu_rows.iloc[0]["institution_norm"]].head(20)
top_similar.to_csv(os.path.join(out_dir, "similar_to_ecnu.csv"), index=False)

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X)
agg["pc1"] = X_pca[:,0]
agg["pc2"] = X_pca[:,1]
plt.figure(figsize=(10,7))
palette = sns.color_palette("tab10", best_k)
sns.scatterplot(data=agg.sample(min(len(agg),1000)), x="pc1", y="pc2", hue="cluster", palette=palette, s=40, legend="full")
plt.title("University clusters (PCA projected)")
plt.savefig(os.path.join(out_dir, "clusters_pca.png"), dpi=150)
plt.close()

ecnu_profile = df_all[df_all["institution_norm"].str.contains(ecnu_name_upper)]
if ecnu_profile.empty:
    ecnu_profile = df_all[df_all["institution_norm"].str.contains("EAST CHINA NORMAL")]
subject_profile = ecnu_profile.groupby("subject").agg(rank=("rank","min"), documents=("documents","sum"), cites=("cites","sum"), cites_per_paper=("cites_per_paper","mean"), top_papers=("top_papers","sum")).reset_index().sort_values("rank")
subject_profile.to_csv(os.path.join(out_dir, "ecnu_subject_profile.csv"), index=False)

plt.figure(figsize=(10,6))
sns.barplot(data=subject_profile, x="rank", y="subject", orient="h")
plt.xlabel("Rank (lower is better)")
plt.ylabel("Subject")
plt.title("ECNU subject ranks")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "ecnu_subject_ranks.png"), dpi=150)
plt.close()

subject_profile["rank_percentile"] = subject_profile["rank"].rank(pct=True)
subject_profile["z_rank"] = (subject_profile["rank"] - subject_profile["rank"].mean()) / subject_profile["rank"].std(ddof=0)
subject_profile.to_csv(os.path.join(out_dir, "ecnu_subject_profile_with_metrics.csv"), index=False)

def train_and_evaluate_per_subject(df_all, out_dir):
    results = []
    models_info = {}
    subjects = sorted(df_all["subject"].unique())
    for subj in subjects:
        subdf = df_all[df_all["subject"] == subj].dropna(subset=["rank"])
        if len(subdf) < 50:
            continue
        subdf = subdf.copy()
        subdf["documents"] = subdf["documents"].fillna(0)
        subdf["cites"] = subdf["cites"].fillna(0)
        subdf["cites_per_paper"] = subdf["cites_per_paper"].fillna(0)
        subdf["top_papers"] = subdf["top_papers"].fillna(0)
        subdf = subdf.sort_values("rank").reset_index(drop=True)
        n = len(subdf)
        i60 = int(math.floor(n*0.6))
        i80 = int(math.floor(n*0.8))
        train = subdf.iloc[:i60]
        val = subdf.iloc[i60:i80]
        test = subdf.iloc[i80:]
        features_cols = ["documents", "cites", "cites_per_paper", "top_papers"]
        X_train = train[features_cols].values
        y_train = train["rank"].values
        X_val = val[features_cols].values if len(val)>0 else None
        y_val = val["rank"].values if len(val)>0 else None
        X_test = test[features_cols].values if len(test)>0 else None
        y_test = test["rank"].values if len(test)>0 else None
        if len(X_train) < 10 or (X_test is None) or len(X_test) < 5:
            continue
        scaler_m = StandardScaler()
        X_train_s = scaler_m.fit_transform(X_train)
        X_val_s = scaler_m.transform(X_val) if X_val is not None else None
        X_test_s = scaler_m.transform(X_test)
        models = {
            "ridge": Ridge(alpha=1.0),
            "rf": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
            "gbr": GradientBoostingRegressor(n_estimators=200, random_state=42)
        }
        best_model_name = None
        best_rmse = float("inf")
        best_model = None
        for name, m in models.items():
            m.fit(X_train_s, y_train)
            preds = m.predict(X_test_s)
            rmse = math.sqrt(mean_squared_error(y_test, preds))
            spearman = spearmanr(y_test, preds).correlation
            r2 = r2_score(y_test, preds)
            results.append({"subject": subj, "model": name, "rmse": rmse, "spearman": spearman if not math.isnan(spearman) else None, "r2": r2, "n_train": len(X_train), "n_test": len(X_test)})
            if rmse < best_rmse:
                best_rmse = rmse
                best_model_name = name
                best_model = m
        models_info[subj] = {"best_model": best_model_name, "best_rmse": best_rmse}
        try:
            import joblib
            joblib.dump(best_model, os.path.join(out_dir, f"model_{subj.replace(' ','_')}.joblib"))
        except Exception:
            pass
    resdf = pd.DataFrame(results)
    resdf.to_csv(os.path.join(out_dir, "model_performance_by_subject.csv"), index=False)
    with open(os.path.join(out_dir, "models_summary.json"), "w", encoding="utf-8") as f:
        json.dump(models_info, f, indent=2)
    return resdf, models_info

res_df, models_info = train_and_evaluate_per_subject(df_all, out_dir)

report_lines = []
report_lines.append("# Extended Analysis Report")
report_lines.append("")
report_lines.append("## 8. Global university categories and similar universities to ECNU")
report_lines.append("")
report_lines.append(f"Cluster count selected by silhouette: {best_k}")
report_lines.append("")
for i, row in centers_df.iterrows():
    report_lines.append(f"- Cluster {int(row['cluster'])} center: " + ", ".join([f"{c}:{float(row[c]):.2f}" for c in features.columns]))
report_lines.append("")
report_lines.append("Top 20 universities similar to ECNU (by cosine similarity):")
report_lines.append("")
report_lines.append(top_similar[["institution_norm","subjects_count","mean_rank","similarity_to_ecnu"]].head(20).to_csv(index=False))
report_lines.append("")
report_lines.append("## 9. ECNU subject profile (top subjects by rank)")
report_lines.append("")
report_lines.append(subject_profile.to_csv(index=False))
report_lines.append("")
report_lines.append("## 10. Ranking prediction models per subject: summary")
report_lines.append("")
report_lines.append(res_df.groupby("model").agg(avg_rmse=("rmse","mean"), avg_spearman=("spearman","mean")).to_csv())
with open(os.path.join(out_dir, "extended_analysis_report.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))

print("Completed. Outputs saved to", out_dir)
print("Files generated:")
for fn in os.listdir(out_dir):
    print("-", fn)
