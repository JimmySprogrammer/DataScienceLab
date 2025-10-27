# -*- coding: utf-8 -*-
"""
Lab6 深度学习排名预测与聚类分析
数据来源: D:\DataScienceLab\lab5\data\merged_all.csv
输出文件:
  - D:\DataScienceLab\lab6\ranking_predictions.csv
  - D:\DataScienceLab\lab6\clustering_results.csv
  - D:\DataScienceLab\lab6\cluster_visualization.png
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

# ========== 1. 路径与文件检查 ==========
data_path = r"D:\DataScienceLab\lab5\data\merged_all.csv"
save_dir = r"D:\DataScienceLab\lab6"
os.makedirs(save_dir, exist_ok=True)

print(f"📊 读取数据: {data_path}")
data = pd.read_csv(data_path)
print(f"原始行数: {len(data)}")
print("列:", list(data.columns))

# ========== 2. 数据清洗 ==========
data = data.dropna(subset=['Institutions', 'Discipline', 'Cites', 'Web of Science Documents'])
data = data[data['Cites'] > 0]
data = data[data['Web of Science Documents'] > 0]

# 计算每个机构每个学科的“影响得分”
data['Impact_Score'] = data['Cites'] / data['Web of Science Documents']

# ========== 3. 深度学习模型预测排名 ==========
print("\n🏫 开始训练学科排名预测模型（Deep Learning）...")
results = []

for discipline, group in data.groupby('Discipline'):
    if len(group) < 300:
        continue
    X = group[['Web of Science Documents', 'Cites', 'Cites/Paper', 'Top Papers']].values
    y = group['Impact_Score'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42)

    model = Sequential([
        Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(0.01), loss='mse')
    model.fit(X_train, y_train, epochs=30, batch_size=16, verbose=0)

    y_pred = model.predict(X_test).flatten()
    mse = mean_squared_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    results.append((discipline, len(group), mse, mape))

    print(f"[{discipline}] n={len(group)} MSE={mse:.2f} MAPE={mape:.3f}")

# 保存预测结果
ranking_df = pd.DataFrame(results, columns=['Discipline', 'Samples', 'MSE', 'MAPE'])
ranking_csv = os.path.join(save_dir, "ranking_predictions.csv")
ranking_df.to_csv(ranking_csv, index=False, encoding='utf-8-sig')
print(f"\n✅ 深度学习排名预测结果已保存到: {ranking_csv}")

# ========== 4. 聚类分析 ==========
print("\n🔍 开始聚类分析 (KMeans)...")

# 聚类特征
features = ['Web of Science Documents', 'Cites', 'Cites/Paper', 'Top Papers']
X = data[features].values
X_scaled = StandardScaler().fit_transform(X)

# 聚类
kmeans = KMeans(n_clusters=8, random_state=42)
data['Cluster'] = kmeans.fit_predict(X_scaled)

# 保存聚类结果
cluster_csv = os.path.join(save_dir, "clustering_results.csv")
data[['Institutions', 'Discipline', 'Cluster']].to_csv(cluster_csv, index=False, encoding='utf-8-sig')
print(f"✅ 聚类结果已保存到: {cluster_csv}")

# 寻找与“East China Normal University”类似的学校
target_name = "East China Normal University"
if target_name in data['Institutions'].values:
    target_cluster = data[data['Institutions'] == target_name]['Cluster'].iloc[0]
    similar = data[data['Cluster'] == target_cluster]['Institutions'].unique()
    print(f"\n🏫 与 {target_name} 类似的学校:")
    print(similar[:10])
else:
    print(f"\n⚠️ 未找到 {target_name}，请检查数据中机构名称是否一致")

# 可视化
pca = PCA(n_components=2)
reduced = pca.fit_transform(X_scaled)
plt.figure(figsize=(8,6))
plt.scatter(reduced[:,0], reduced[:,1], c=data['Cluster'], cmap='tab10', alpha=0.7)
plt.title("ESI Institutions Clustering Visualization")
plt.xlabel("PCA-1")
plt.ylabel("PCA-2")

img_path = os.path.join(save_dir, "cluster_visualization.png")
plt.savefig(img_path, dpi=300)
plt.close()
print(f"✅ 聚类可视化图已保存到: {img_path}")

print("\n🎯 全部完成！结果文件已生成在 D:\\DataScienceLab\\lab6\\")
