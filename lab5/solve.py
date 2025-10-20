import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import warnings
import os

warnings.filterwarnings("ignore")
os.environ["LOKY_MAX_CPU_COUNT"] = "8"

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("📊 正在读取合并数据：D:\\DataScienceLab\\lab5\\data\\merged_all.csv")

df = pd.read_csv("D:\\DataScienceLab\\lab5\\data\\merged_all.csv")
print(f"✅ 数据加载成功，总行数: {len(df)}")
print("📑 列:", list(df.columns))

# ========== 第8题 ==========
print("=== 第8题：全球高校分类分析 ===")

features = ["Web of Science Documents", "Cites", "Cites/Paper", "Top Papers"]
df_clean = df.dropna(subset=features).copy()
X_scaled = StandardScaler().fit_transform(df_clean[features])

kmeans = KMeans(n_clusters=5, random_state=42)
df_clean.loc[:, "Cluster"] = kmeans.fit_predict(X_scaled)

cluster_summary = df_clean.groupby("Cluster")["Institutions"].count().to_frame("count")
print("✅ 成功完成聚类分析（共分5类）\n", cluster_summary)

sns.countplot(x="Cluster", data=df_clean, palette="viridis")
plt.title("全球高校聚类分布")
plt.xlabel("Cluster")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("cluster_plot.png", dpi=300)
print("📈 已生成聚类分布图 cluster_plot.png")

target_uni = "EAST CHINA NORMAL UNIVERSITY"
target_cluster = df_clean[df_clean["Institutions"].str.contains(target_uni, case=False, na=False)]["Cluster"].iloc[0]
similar_universities = df_clean[df_clean["Cluster"] == target_cluster].nlargest(10, "Cites")[["Institutions", "Cites", "Web of Science Documents"]]
print(f"\n与【{target_uni}】相似的高校包括：\n", similar_universities)

# ========== 第9题 ==========
print("\n=== 第9题：华东师范大学学科画像 ===")

ecnu_data = df[df["Institutions"].str.contains(target_uni, case=False, na=False)]
discipline_stats = ecnu_data.groupby("Discipline")["Cites"].sum().sort_values(ascending=False).head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x=discipline_stats.values, y=discipline_stats.index, palette="coolwarm")
plt.title("华东师大学科画像（按总被引频次）")
plt.xlabel("总被引频次")
plt.ylabel("学科")
plt.tight_layout()
plt.savefig("ecnu_profile.png", dpi=300)
print("📊 已生成华东师大学科画像 ecnu_profile.png")

# ========== 第10题 ==========
print("\n=== 第10题：排名预测模型（优化版） ===")

model_data = df.dropna(subset=["Cites", "Web of Science Documents", "Cites/Paper", "Top Papers"]).copy()
X = model_data[["Web of Science Documents", "Cites/Paper", "Top Papers"]]
y = model_data["Cites"]
idx = model_data.index  # 保存原始索引

# 按要求：前60%训练，后20%测试（中间20%丢弃）
train_end = int(0.6 * len(model_data))
test_start = int(0.8 * len(model_data))

X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
X_test, y_test = X.iloc[test_start:], y.iloc[test_start:]
test_idx = idx[test_start:]  # 对应索引

print(f"📈 使用各学科前60%训练、后20%测试")
print(f"训练样本数: {len(X_train)}, 测试样本数: {len(X_test)}")

model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

scores = cross_val_score(model, X, y, cv=5, scoring="r2")
print(f"模型预测 R² 平均得分: {scores.mean():.4f}")

y_pred = model.predict(X_test)

# 使用 test_idx 映射回原数据
test_result = model_data.loc[test_idx, ["Institutions", "Discipline", "Cites"]].copy()
test_result.loc[:, "Predicted Cites"] = y_pred

print("\n示例预测结果前10行：")
print(test_result.head(10))
