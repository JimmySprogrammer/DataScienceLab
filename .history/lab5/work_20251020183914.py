import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

DATA_DIR = r"D:\DataScienceLab\lab5\data"
RESULT_DIR = r"D:\DataScienceLab\lab5\results"
os.makedirs(RESULT_DIR, exist_ok=True)


def load_all_data():
    data_frames = {}
    for file in os.listdir(DATA_DIR):
        if file.endswith(".csv"):
            path = os.path.join(DATA_DIR, file)
            try:
                df = pd.read_csv(path, encoding="latin1")
                data_frames[file.replace(".csv", "")] = df
                print(f"✅ 成功读取: {file} ({len(df)} 行)")
            except Exception as e:
                print(f"❌ 读取失败: {file} -> {e}")
    return data_frames


def analyze_global_clusters(data):
    print("\n=== 第8题：全球高校分类分析 ===")
    all_df = []
    for subject, df in data.items():
        df = df.copy()
        df["subject"] = subject
        all_df.append(df)
    all_df = pd.concat(all_df, ignore_index=True)

    # 只保留主要数值特征
    cols = ["rank", "cites_per_paper", "documents", "top_papers"]
    all_df = all_df.dropna(subset=cols)
    X = all_df[cols]
    X_scaled = StandardScaler().fit_transform(X)

    # 聚类
    kmeans = KMeans(n_clusters=4, random_state=42)
    all_df["cluster"] = kmeans.fit_predict(X_scaled)

    cluster_summary = all_df.groupby("cluster")[cols].mean()
    print("\n全球高校大致可分为以下几类（按科研实力区分）:")
    print(cluster_summary)

    cluster_summary.to_excel(os.path.join(RESULT_DIR, "global_clusters.xlsx"))

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=all_df,
        x="cites_per_paper",
        y="documents",
        hue="cluster",
        palette="Set2"
    )
    plt.title("Global University Clusters")
    plt.savefig(os.path.join(RESULT_DIR, "global_clusters.png"))
    plt.close()

    return all_df


def find_similar_universities(all_df):
    print("\n=== 与华东师大类似高校分析 ===")
    ecnu_df = all_df[all_df["institution"].str.contains("EAST CHINA NORMAL", case=False, na=False)]
    if ecnu_df.empty:
        print("未找到华东师范大学数据，检查CSV文件。")
        return
    ecnu_cluster = ecnu_df["cluster"].mode()[0]
    similar = all_df[all_df["cluster"] == ecnu_cluster]

    print(f"\n华东师范大学属于 cluster {ecnu_cluster} 类，与以下高校类似（示例5所）:")
    print(similar["institution"].value_counts().head(5))

    similar.to_excel(os.path.join(RESULT_DIR, "similar_universities.xlsx"), index=False)


def profile_ecnu(data):
    print("\n=== 第9题：华东师大学科画像 ===")
    ecnu_list = []
    for subject, df in data.items():
        matched = df[df["institution"].str.contains("EAST CHINA NORMAL", case=False, na=False)]
        if not matched.empty:
            matched["subject"] = subject
            ecnu_list.append(matched)

    if not ecnu_list:
        print("未找到华东师范大学数据。")
        return
    ecnu = pd.concat(ecnu_list)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=ecnu, x="subject", y="rank", palette="viridis")
    plt.xticks(rotation=80)
    plt.title("ECNU Subject Rankings")
    plt.ylabel("Rank (lower is better)")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "ecnu_ranking_profile.png"))
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=ecnu, x="documents", y="cites_per_paper", size="top_papers", legend=False)
    plt.title("ECNU Research Output and Impact")
    plt.savefig(os.path.join(RESULT_DIR, "ecnu_research_impact.png"))
    plt.close()

    ecnu.to_excel(os.path.join(RESULT_DIR, "ecnu_profile.xlsx"), index=False)
    print("✅ 已生成华东师大学科画像。")


def model_ranking_prediction(data):
    print("\n=== 第10题：学科排名预测模型 ===")
    all_df = []
    for subject, df in data.items():
        df = df.copy()
        df["subject"] = subject
        all_df.append(df)
    all_df = pd.concat(all_df, ignore_index=True)
    all_df = all_df.dropna(subset=["rank", "cites_per_paper", "documents", "top_papers"])

    X = all_df[["cites_per_paper", "documents", "top_papers"]]
    y = all_df["rank"]

    n = len(all_df)
    split = int(0.6 * n)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"模型性能: MAE={mae:.2f}, R²={r2:.3f}")
    coef_df = pd.DataFrame({
        "feature": X.columns,
        "coefficient": model.coef_
    })
    print("\n特征重要性：")
    print(coef_df)

    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.xlabel("真实排名")
    plt.ylabel("预测排名")
    plt.title("预测 vs 实际排名")
    plt.savefig(os.path.join(RESULT_DIR, "ranking_prediction.png"))
    plt.close()

    coef_df.to_excel(os.path.join(RESULT_DIR, "ranking_model_coefficients.xlsx"), index=False)


def main():
    data = load_all_data()
    if not data:
        print("No parsed data")
        return

    all_df = analyze_global_clusters(data)
    find_similar_universities(all_df)
    profile_ecnu(data)
    model_ranking_prediction(data)

    print("\n✅ 所有分析完成！结果已保存到:", RESULT_DIR)


if __name__ == "__main__":
    main()
