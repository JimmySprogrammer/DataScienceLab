import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# ===========================
# 基础设置
# ===========================
data_path = r"D:\DataScienceLab\lab5\data\merged_all.csv"
print(f"📊 正在读取合并数据：{data_path}")

# 读取数据（自动尝试多种编码）
try:
    data = pd.read_csv(data_path, encoding="utf-8")
except:
    data = pd.read_csv(data_path, encoding="latin1")

# 去除乱码列名
data.columns = [col.replace("ï»¿", "").strip() for col in data.columns]
print(f"✅ 数据加载成功，总行数: {len(data)}")
print(f"📑 列: {list(data.columns)}")

# ===========================
# 第8题：全球高校分类分析
# ===========================
print("\n=== 第8题：全球高校分类分析 ===")

features = ["Web of Science Documents", "Cites", "Cites/Paper", "Top Papers"]
df = data.dropna(subset=features)
X = df[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_scaled)

cluster_summary = df.groupby("Cluster")["Institutions"].count().to_frame()
print("✅ 成功完成聚类分析（共分5类）")
print(cluster_summary)

plt.figure(figsize=(7, 5))
cluster_summary["Institutions"].plot(kind="bar", color="black", edgecolor="gray")
plt.title("Global University Clusters")
plt.ylabel("Institution Count")
plt.xlabel("Cluster")
plt.tight_layout()
plt.savefig("cluster_plot.png")
print("📈 已生成聚类分布图 cluster_plot.png")

# 查找与华东师范大学相似的高校
target = df[df["Institutions"].str.contains("EAST CHINA NORMAL UNIVERSITY", case=False, na=False)]
if not target.empty:
    cluster_id = target["Cluster"].iloc[0]
    similar = df[df["Cluster"] == cluster_id].sort_values("Cites", ascending=False).head(10)
    print(f"\n与【EAST CHINA NORMAL UNIVERSITY】相似的高校包括：")
    print(similar[["Institutions", "Cites", "Web of Science Documents"]])
else:
    print("⚠️ 数据中未找到 EAST CHINA NORMAL UNIVERSITY")

# ===========================
# 第9题：华东师范大学学科画像
# ===========================
print("\n=== 第9题：华东师范大学学科画像 ===")

subject_col = "学科" if "学科" in data.columns else "Discipline"
ecnu = data[data["Institutions"].str.contains("EAST CHINA NORMAL UNIVERSITY", case=False, na=False)]

if not ecnu.empty:
    plt.figure(figsize=(10, 6))
    plt.bar(ecnu[subject_col], ecnu["Cites"], color="black", edgecolor="gray")
    plt.xticks(rotation=75, ha="right")
    plt.title("ECNU Subject Profile (Citations)")
    plt.ylabel("Citations")
    plt.tight_layout()
    plt.savefig("ecnu_profile.png")
    print("📊 已生成华东师大学科画像 ecnu_profile.png")
else:
    print("⚠️ 未找到华东师范大学相关数据，无法绘制画像。")

# ===========================
# 第10题：排名预测模型（优化版）
# ===========================
print("\n=== 第10题：排名预测模型（优化版） ===")

df_model = data.dropna(subset=features + [subject_col]).copy()
X = df_model[["Web of Science Documents", "Cites/Paper", "Top Papers"]]
y = df_model["Cites"]

results = []

for subject, group in df_model.groupby(subject_col):
    if len(group) < 30:
        continue
    group_sorted = group.sort_values("Cites", ascending=False).reset_index(drop=True)
    n = len(group_sorted)
    train_end = int(n * 0.6)
    test_end = int(n * 0.8)

    train = group_sorted.iloc[:train_end]
    test = group_sorted.iloc[train_end:test_end]

    X_train = train[["Web of Science Documents", "Cites/Paper", "Top Papers"]]
    y_train = train["Cites"]
    X_test = test[["Web of Science Documents", "Cites/Paper", "Top Papers"]]
    y_test = test["Cites"]

    if len(X_train) < 5 or len(X_test) < 5:
        continue

    model = RandomForestRegressor(n_estimators=150, random_state=42)
    model.fit(X_train, y_train)
    r2 = model.score(X_test, y_test)

    preds = model.predict(X_test)
    group_result = test.copy()
    group_result["Predicted Cites"] = preds
    group_result["R2"] = r2
    results.append(group_result)

if results:
    final_results = pd.concat(results)
    print(f"📈 使用各学科前60%训练、后20%测试")
    print(f"训练样本数: {int(len(df_model) * 0.6)}, 测试样本数: {int(len(df_model) * 0.2)}")
    print(f"模型预测 R² 平均得分: {np.mean([r for r in final_results['R2'] if not pd.isna(r)]):.4f}")

    print("\n示例预测结果前10行：")
    print(final_results[["Institutions", subject_col, "Cites", "Predicted Cites"]].head(10))
else:
    print("⚠️ 数据不足，无法完成建模。")
