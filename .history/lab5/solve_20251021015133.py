# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def main():
    path = r"D:\DataScienceLab\lab5\data\merged_all.csv"
    print(f"📊 正在读取合并数据：{path}\n")

    # ---------- 读取与清洗 ----------
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin1")

    # 修正列名乱码
    df.columns = [c.replace("ï»¿", "").strip() for c in df.columns]
    print(f"✅ 数据加载成功，总行数: {len(df)}")
    print(f"📑 列: {list(df.columns)}")

    # ---------- 第8题：全球高校聚类分析 ----------
    print("\n=== 第8题：全球高校分类分析 ===")
    data = df.dropna(subset=["Cites", "Web of Science Documents", "Cites/Paper", "Top Papers"])
    features = ["Web of Science Documents", "Cites", "Cites/Paper", "Top Papers"]
    X = data[features]
    X_scaled = StandardScaler().fit_transform(X)

    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    data["Cluster"] = kmeans.fit_predict(X_scaled)
    print("✅ 成功完成聚类分析（共分5类）")
    print(data["Cluster"].value_counts().to_frame("Institutions"))

    # 可视化聚类结果
    plt.figure(figsize=(8, 6))
    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=data["Cluster"], cmap="rainbow", s=10)
    plt.title("Global University Clusters", fontsize=14)
    plt.xlabel("Standardized Web of Science Documents")
    plt.ylabel("Standardized Cites")
    plt.savefig("cluster_plot.png", dpi=300, bbox_inches="tight")
    print("📈 已生成聚类分布图 cluster_plot.png")

    # 查找与华东师范大学相似高校
    target_name = "EAST CHINA NORMAL UNIVERSITY"
    if target_name in data["Institutions"].values:
        target_cluster = data.loc[data["Institutions"] == target_name, "Cluster"].values[0]
        similar = data[data["Cluster"] == target_cluster][["Institutions", "Cites", "Web of Science Documents"]].head(10)
        print(f"\n与【{target_name}】相似的高校包括：")
        print(similar)
    else:
        print(f"\n未找到高校：{target_name}")

    # ---------- 第9题：华东师范大学学科画像 ----------
    print("\n=== 第9题：华东师范大学学科画像 ===")
    ecnu = df[df["Institutions"] == "EAST CHINA NORMAL UNIVERSITY"]
    if ecnu.empty:
        print("⚠️ 未找到华东师范大学相关数据，跳过此部分。")
    else:
        discipline_stats = ecnu.groupby("Discipline")[["Web of Science Documents", "Cites", "Top Papers"]].sum()
        discipline_stats.plot(kind="barh", figsize=(8, 6), color=["#4C72B0", "#55A868", "#C44E52"])
        plt.title("Discipline Profile of EAST CHINA NORMAL UNIVERSITY", fontsize=14)
        plt.xlabel("Value")
        plt.tight_layout()
        plt.savefig("ecnu_profile.png", dpi=300)
        print("📊 已生成华东师大学科画像 ecnu_profile.png")

    # ---------- 第10题：排名预测模型（优化版） ----------
    print("\n=== 第10题：排名预测模型（优化版） ===")
    model_df = df.copy()
    model_df = model_df.dropna(subset=["Cites", "Web of Science Documents", "Cites/Paper", "Top Papers"])

    features = ["Web of Science Documents", "Cites/Paper", "Top Papers"]
    target = "Cites"
    train_list, test_list = [], []

    # 按学科划分训练集与测试集
    for disc, group in model_df.groupby("Discipline"):
        group = group.sort_values(by="Cites", ascending=False)
        n = len(group)
        if n < 10:
            continue
        train_end = int(n * 0.6)
        test_start = int(n * 0.8)
        train_list.append(group.iloc[:train_end])
        test_list.append(group.iloc[test_start:])

    train_df = pd.concat(train_list)
    test_df = pd.concat(test_list)

    X_train, y_train = train_df[features], train_df[target]
    X_test, y_test = test_df[features], test_df[target]

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)

    print(f"📈 使用各学科前60%训练、后20%测试")
    print(f"训练样本数: {len(train_df)}, 测试样本数: {len(test_df)}")
    print(f"模型预测 R² 得分: {r2:.4f}")

    result_df = test_df[["Institutions", "Discipline", "Cites"]].copy()
    result_df["Predicted Cites"] = y_pred
    print("\n示例预测结果前10行：")
    print(result_df.head(10))

if __name__ == "__main__":
    main()
