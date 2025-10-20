# analysis.py
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def main():
    data_path = r"D:\DataScienceLab\lab5\data\merged_all.csv"
    print(f"📊 正在读取合并数据：{data_path}\n")

    df = pd.read_csv(data_path, encoding="latin1")

    df.columns = [c.strip() for c in df.columns]
    if 'Discipline' not in df.columns:
        print("❌ 缺少 Discipline 列，请确认 merged_all.csv 结构是否正确")
        return

    df = df.dropna(subset=['Institutions', 'Countries/Regions', 'Web of Science Documents', 'Cites'])
    df = df[df['Web of Science Documents'].apply(lambda x: str(x).isdigit())]
    df['Web of Science Documents'] = df['Web of Science Documents'].astype(int)
    df['Cites'] = df['Cites'].astype(int)

    print(f"✅ 数据加载成功，总行数: {len(df)}")
    print(f"🌍 包含国家数量: {df['Countries/Regions'].nunique()}，学科数量: {df['Discipline'].nunique()}\n")

    # ==== 1️⃣ 国家层面统计 ====
    print("=== 各国论文总量 Top10 ===")
    country_papers = df.groupby("Countries/Regions")["Web of Science Documents"].sum().sort_values(ascending=False).head(10)
    print(country_papers, "\n")

    plt.figure(figsize=(10, 5))
    country_papers.plot(kind="bar", color="black", edgecolor="gray")
    plt.title("Top10 Countries by Publications")
    plt.xlabel("Country")
    plt.ylabel("Documents")
    plt.tight_layout()
    plt.show()

    # ==== 2️⃣ 学科层面统计 ====
    print("=== 各学科平均引文数 Top10 ===")
    discipline_cites = df.groupby("Discipline")["Cites"].mean().sort_values(ascending=False).head(10)
    print(discipline_cites, "\n")

    plt.figure(figsize=(10, 5))
    discipline_cites.plot(kind="barh", color="dimgray", edgecolor="black")
    plt.title("Top10 Disciplines by Average Citations")
    plt.xlabel("Average Cites")
    plt.ylabel("Discipline")
    plt.tight_layout()
    plt.show()

    # ==== 3️⃣ 全球高校科研影响力聚类 ====
    print("=== KMeans 聚类分析（科研产出与引用） ===")
    X = df[["Web of Science Documents", "Cites"]]
    X_scaled = StandardScaler().fit_transform(X)
    kmeans = KMeans(n_clusters=4, random_state=42)
    df["Cluster"] = kmeans.fit_predict(X_scaled)

    cluster_summary = df.groupby("Cluster")[["Web of Science Documents", "Cites"]].mean().round(1)
    print(cluster_summary, "\n")

    plt.figure(figsize=(7, 6))
    colors = ["black", "gray", "silver", "lightgray"]
    for i in range(4):
        cluster = df[df["Cluster"] == i]
        plt.scatter(cluster["Web of Science Documents"], cluster["Cites"], s=10, color=colors[i], label=f"Cluster {i}")
    plt.xlabel("Web of Science Documents")
    plt.ylabel("Cites")
    plt.title("Global University Clusters by Research Influence")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ==== 4️⃣ 结果导出 ====
    output_path = r"D:\DataScienceLab\lab5\data\global_clusters.csv"
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"✅ 聚类结果已保存: {output_path}")

if __name__ == "__main__":
    main()
