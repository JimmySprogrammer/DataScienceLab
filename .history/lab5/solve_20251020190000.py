# eight.py
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

path = r"D:\DataScienceLab\lab5\data\merged_all.csv"

print(f"📊 正在读取合并数据：{path}")
df = pd.read_csv(path, encoding="latin1")

# 去除BOM头并规范列名
df.columns = [c.strip().replace("ï»¿", "") for c in df.columns]

print(f"\n✅ 数据加载成功，总行数: {len(df)}")
print(f"📑 列: {df.columns.tolist()[:8]}")

# 去除空值
df = df.dropna(subset=["Institutions", "Countries/Regions", "Cites", "Web of Science Documents"])

# ========== 第8题：全球高校分类分析 ==========
print("\n=== 第8题：全球高校分类分析 ===")

# 聚合各高校的论文与引用表现
inst_summary = df.groupby("Institutions").agg({
    "Web of Science Documents": "sum",
    "Cites": "sum",
    "Cites/Paper": "mean",
    "Top Papers": "sum"
}).reset_index()

# 标准化处理
features = inst_summary[["Web of Science Documents", "Cites", "Cites/Paper", "Top Papers"]].fillna(0)
X = StandardScaler().fit_transform(features)

# KMeans聚类
kmeans = KMeans(n_clusters=5, random_state=42)
inst_summary["Cluster"] = kmeans.fit_predict(X)

print("✅ 成功完成聚类分析（共分5类）")
print(inst_summary.groupby("Cluster").agg({"Institutions": "count"}))

# 绘制聚类散点图
plt.figure(figsize=(8,6))
plt.scatter(inst_summary["Cites"], inst_summary["Web of Science Documents"],
            c=inst_summary["Cluster"], cmap="tab10", s=20, alpha=0.7)
plt.xlabel("Cites")
plt.ylabel("Web of Science Documents")
plt.title("Global University Clustering")
plt.savefig(r"D:\DataScienceLab\lab5\cluster_plot.png", dpi=300)
print("📈 已生成聚类分布图 cluster_plot.png")

# 找出与华东师范大学相似的高校
target = "EAST CHINA NORMAL UNIVERSITY"
if target in inst_summary["Institutions"].values:
    cluster_id = inst_summary.loc[inst_summary["Institutions"] == target, "Cluster"].iloc[0]
    similar = inst_summary[inst_summary["Cluster"] == cluster_id].head(10)
    print(f"\n与【{target}】相似的高校包括：")
    print(similar[["Institutions", "Cites", "Web of Science Documents"]])
else:
    print(f"⚠️ 数据中未找到 {target}")

# ========== 第9题：学科画像 ==========
print("\n=== 第9题：华东师范大学学科画像 ===")
ecnu = df[df["Institutions"].str.contains("EAST CHINA NORMAL UNIVERSITY", case=False, na=False)]

if len(ecnu) > 0:
    plt.figure(figsize=(8,6))
    plt.barh(ecnu["Discipline"], ecnu["Cites"], color="black")
    plt.xlabel("Citations")
    plt.ylabel("Discipline")
    plt.title("Discipline Profile of East China Normal University")
    plt.tight_layout()
    plt.savefig(r"D:\DataScienceLab\lab5\ecnu_profile.png", dpi=300)
    print("📊 已生成华东师大学科画像 ecnu_profile.png")
else:
    print("⚠️ 数据中未找到华东师范大学的学科记录")

# ========== 第10题：预测模型 ==========
print("\n=== 第10题：排名预测模型 ===")
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

df["Rank"] = df.groupby("Discipline")["Cites"].rank(ascending=False, method="dense")

X = df[["Cites", "Web of Science Documents", "Cites/Paper", "Top Papers"]].fillna(0)
y = df["Rank"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f"模型预测R²得分: {r2_score(y_test, y_pred):.3f}")
