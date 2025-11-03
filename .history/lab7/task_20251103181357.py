import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt

# 路径设置
data_path = r"D:\DataScienceLab\lab5\data\merged_all.csv"
output_dir = r"D:\DataScienceLab\lab7"
os.makedirs(output_dir, exist_ok=True)

print("📊 读取数据:", data_path)
df = pd.read_csv(data_path)

# 清洗数据
df = df.dropna(subset=["Institutions", "Cites", "Web of Science Documents"])
df = df[df["Cites"] > 0]
print("数据行数:", len(df))

# 将学科转换为标签
df["Discipline"] = df["Discipline"].astype(str)
disciplines = df["Discipline"].unique()

# ==========================
# 🔹 1. 聚类分析（为模型提供辅助特征）
# ==========================
print("\n🔍 执行KMeans聚类 (k=8)...")
num_features = ["Cites", "Web of Science Documents", "Cites/Paper", "Top Papers"]
X_cluster = df[num_features].fillna(0)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

kmeans = KMeans(n_clusters=8, random_state=42)
df["ClusterLabel"] = kmeans.fit_predict(X_scaled)

cluster_csv = os.path.join(output_dir, "clustering_results_v2.csv")
df.to_csv(cluster_csv, index=False)
print(f"✅ 聚类结果已保存到: {cluster_csv}")

# 可视化聚类
plt.figure(figsize=(8,6))
plt.scatter(X_scaled[:,0], X_scaled[:,1], c=df["ClusterLabel"], cmap="tab10", s=10)
plt.title("University Clusters based on ESI Data")
plt.xlabel("Cites (scaled)")
plt.ylabel("Documents (scaled)")
plt.savefig(os.path.join(output_dir, "cluster_visualization_v2.png"))
plt.close()
print("✅ 聚类可视化图已保存")

# ==========================
# 🔹 2. 深度学习学科排名预测模型
# ==========================
print("\n🏫 开始训练改进版深度学习模型...")

results = []
for disc in disciplines:
    sub = df[df["Discipline"] == disc]
    if len(sub) < 300:
        continue

    X = sub[["Web of Science Documents", "Cites", "Cites/Paper", "Top Papers", "ClusterLabel"]].fillna(0)
    y = np.arange(len(sub))  # 模拟排名（按出现顺序）

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler2 = StandardScaler()
    X_train_scaled = scaler2.fit_transform(X_train)
    X_test_scaled = scaler2.transform(X_test)

    # 构建改进模型
    model = Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1)
    ])

    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mse'])
    model.fit(X_train_scaled, y_train, epochs=25, batch_size=16, verbose=0)

    preds = model.predict(X_test_scaled).flatten()
    mse = mean_squared_error(y_test, preds)
    mape = mean_absolute_percentage_error(y_test, preds)

    results.append((disc, len(sub), round(mse, 3), round(mape, 4)))
    print(f"[{disc}] n={len(sub)} MSE={mse:.2f} MAPE={mape:.3f}")

# 保存预测结果
results_df = pd.DataFrame(results, columns=["Discipline", "Samples", "MSE", "MAPE"])
results_path = os.path.join(output_dir, "ranking_predictions_v2.csv")
results_df.to_csv(results_path, index=False)
print(f"\n✅ 改进版预测结果已保存到: {results_path}")

# ==========================
# 🔹 3. 输出总结结果
# ==========================
summary = results_df.sort_values("MSE").reset_index(drop=True)
print("\n🏁 模型表现最佳的前5个学科:")
print(summary.head())

print(f"\n🎯 全部完成！结果文件位于: {output_dir}")
