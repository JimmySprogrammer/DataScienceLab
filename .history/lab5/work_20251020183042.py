# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

DATA_DIR = r"D:\DataScienceLab\lab5\data"

def load_all_data():
    data_frames = {}
    print("📁 当前数据路径:", DATA_DIR)
    print("📄 目录下文件:", os.listdir(DATA_DIR))
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

# ============ 第8题：全球高校分类 & 与华师大相似高校 ============
def analyze_global_patterns(data):
    combined = []
    for subject, df in data.items():
        if "University" in df.columns and "Rank" in df.columns:
            df = df[["University", "Country", "Rank"]].copy()
            df["Subject"] = subject
            combined.append(df)
    all_data = pd.concat(combined, ignore_index=True)
    pivot = all_data.pivot_table(index="University", values="Rank", aggfunc="mean").fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(pivot)
    kmeans = KMeans(n_clusters=5, random_state=42)
    pivot["Cluster"] = kmeans.fit_predict(X_scaled)
    ecnu_cluster = pivot.loc["EAST CHINA NORMAL UNIVERSITY", "Cluster"] if "EAST CHINA NORMAL UNIVERSITY" in pivot.index else None
    similar_universities = pivot[pivot["Cluster"] == ecnu_cluster].index.tolist() if ecnu_cluster is not None else []
    print("\n🎯 全球高校可分为5类:")
    print(pivot["Cluster"].value_counts().sort_index())
    print(f"\n🏫 与华东师范大学相似的高校（Cluster {ecnu_cluster}）:")
    print(similar_universities[:15])
    plt.figure(figsize=(6,4))
    plt.hist(pivot["Cluster"], bins=5, color='skyblue', edgecolor='black')
    plt.title("全球高校聚类分布")
    plt.xlabel("Cluster")
    plt.ylabel("高校数量")
    plt.tight_layout()
    plt.show()
    return pivot, similar
