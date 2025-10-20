import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score

# === 路径配置 ===
path = r"D:\DataScienceLab\lab5\data\merged_all.csv"

print(f"📊 正在读取合并数据：{path}")
data = pd.read_csv(path, encoding='utf-8')
print(f"✅ 数据加载成功，总行数: {len(data)}")
print(f"📑 列: {list(data.columns)}")

# 确认列名映射（防止有的版本叫 Discipline）
if "Discipline" in data.columns:
    data.rename(columns={"Discipline": "学科"}, inplace=True)

# ===== 第8题：全球高校分类分析 =====
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

print("\n=== 第8题：全球高校分类分析 ===")
df = data.copy()
features = ["Web of Science Documents", "Cites", "Cites/Paper", "Top Papers"]
df = df.dropna(subset=features)

X = df[features]
X_scaled = MinMaxScaler().fit_transform(X)
kmeans = KMeans(n_clusters=5, random_state=42)
df["Cluster"] = kmeans.fit_predict(X_scaled)

print("✅ 成功完成聚类分析（共分5类）")
print(df["Cluster"].value_counts().to_frame())

plt.figure(figsize=(8, 6))
plt.scatter(df["Web of Science Documents"], df["Cites"], c=df["Cluster"], cmap="tab10", s=10)
plt.title("Global University Clusters")
plt.xlabel("Web of Science Documents")
plt.ylabel("Cites")
plt.savefig("cluster_plot.png", dpi=300)
plt.close()
print("📈 已生成聚类分布图 cluster_plot.png")

# 与ECNU相似的院校
target = "EAST CHINA NORMAL UNIVERSITY"
if target in df["Institutions"].values:
    target_cluster = df.loc[df["Institutions"] == target, "Cluster"].iloc[0]
    similar = df[df["Cluster"] == target_cluster].sort_values("Cites", ascending=False).head(10)
    print(f"\n与【{target}】相似的高校包括：")
    print(similar[["Institutions", "Cites", "Web of Science Documents"]])
else:
    print(f"\n⚠️ 未找到目标院校 {target}")

# ===== 第9题：华东师范大学学科画像 =====
print("\n=== 第9题：华东师范大学学科画像 ===")
ecnu = data[data["Institutions"].str.contains("EAST CHINA NORMAL UNIVERSITY", case=False, na=False)]
profile = ecnu.groupby("学科")[["Cites", "Web of Science Documents", "Top Papers"]].sum().sort_values("Cites", ascending=False)
profile.plot(kind="bar", figsize=(10, 6))
plt.title("华东师范大学学科画像")
plt.ylabel("数量")
plt.tight_layout()
plt.savefig("ecnu_profile.png", dpi=300)
plt.close()
print("📊 已生成华东师大学科画像 ecnu_profile.png")

# ===== 第10题：排名预测模型（优化版） =====
print("\n=== 第10题：排名预测模型（优化版） ===")
print("📈 使用各学科前60%训练、后20%测试")

all_train, all_test = [], []
r2_scores = []

for subject, group in data.groupby("学科"):
    group = group.dropna(subset=["Cites", "Web of Science Documents"])
    if len(group) < 10:
        continue

    group = group.sort_values("Cites", ascending=False).reset_index(drop=True)
    n = len(group)
    train_end = int(n * 0.6)
    test_start = int(n * 0.8)

    train = group.iloc[:train_end]
    test = group.iloc[test_start:]

    features = ["Web of Science Documents", "Cites/Paper", "Top Papers"]
    X_train = train[features].fillna(0)
    X_test = test[features].fillna(0)
    y_train = train["Cites"]
    y_test = test["Cites"]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    test["Predicted Cites"] = y_pred
    all_test.append(test)
    score = r2_score(y_test, y_pred)
    r2_scores.append(score)

final_results = pd.concat(all_test)
mean_r2 = np.mean(r2_scores)

print(f"训练样本数: {sum(len(g) for g in data.groupby('学科')) * 0.6:.0f}, 测试样本数: {len(final_results)}")
print(f"模型预测 R² 平均得分: {mean_r2:.4f}")

print("\n示例预测结果前10行：")
print(final_results[["Institutions", "学科", "Cites", "Predicted Cites"]].head(10))

# 保存预测结果
final_results.to_csv("prediction_results.csv", index=False, encoding="utf-8-sig")
print("\n💾 已保存预测结果至 prediction_results.csv")
