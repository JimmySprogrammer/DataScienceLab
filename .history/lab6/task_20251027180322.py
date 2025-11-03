# tasks11_12.py
# -*- coding: utf-8 -*-
"""
任务11 & 12 脚本
- 数据位置: D:\DataScienceLab\lab5\data\merged_all.csv
- 输出：
    - models/ (保存训练的模型权重)
    - predictions_{discipline}.csv (每个学科的测试集预测与评估)
    - dl_summary.csv (汇总每个学科 MSE, MAPE)
    - cluster_plot.png, ecnu_cluster_similars.csv (聚类结果与与ECNU相似高校)
    - cluster_analysis.csv (簇内特征均值，用于原因分析)
"""
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ---------------------------
# 配置
# ---------------------------
DATA_PATH = r"D:\DataScienceLab\lab5\data\merged_all.csv"
OUT_DIR = r"D:\DataScienceLab\lab5\outputs_tasks11_12"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "models"), exist_ok=True)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# ---------------------------
# 读取与清洗数据
# ---------------------------
print("📊 读取数据:", DATA_PATH)
df = pd.read_csv(DATA_PATH, encoding="utf-8")
print("原始行数:", len(df))
print("列:", list(df.columns))

# 保留必要列并转为数值类型
cols_needed = ["Institutions", "Discipline", "Web of Science Documents", "Cites", "Cites/Paper", "Top Papers"]
for c in cols_needed:
    if c not in df.columns:
        raise SystemExit(f"缺少列: {c}，请检查 merged_all.csv 列名。")

df = df[cols_needed].copy()
df["Web of Science Documents"] = pd.to_numeric(df["Web of Science Documents"], errors="coerce").fillna(0)
df["Cites"] = pd.to_numeric(df["Cites"], errors="coerce").fillna(0)
df["Cites/Paper"] = pd.to_numeric(df["Cites/Paper"], errors="coerce").fillna(0)
df["Top Papers"] = pd.to_numeric(df["Top Papers"], errors="coerce").fillna(0)

# 为每个学科计算“Rank”（基于 Cites 降序）
df["Rank"] = df.groupby("Discipline")["Cites"].rank(method="first", ascending=False)
# 将 Rank 转为整数（排名从1开始）
df["Rank"] = df["Rank"].astype(int)

# 过滤掉样本数非常少的学科（例如 < 30），但仍会记录
discipline_counts = df["Discipline"].value_counts()
print("学科样本数（前10）：\n", discipline_counts.head(10))

# ---------------------------
# 第11题：按学科训练深度学习模型预测排名
# 策略：对每个学科单独建模
#   - 按 Cites 排序（与之前一致），取前60%训练、后20%测试（中间20%忽略）
#   - 特征： ["Web of Science Documents", "Cites/Paper", "Top Papers"]（可扩展）
#   - 模型：小型多层感知器（MLP），输出预测 Rank（回归）
#   - 评估指标：MSE, MAPE
# ---------------------------

features = ["Web of Science Documents", "Cites/Paper", "Top Papers"]
target = "Rank"

summary_rows = []
all_test_results = []

# Keras 默认日志太多，降低 verbosity
tf.get_logger().setLevel('ERROR')

for disc, group in df.groupby("Discipline"):
    n = len(group)
    if n < 30:
        print(f"跳过学科（样本太少）: {disc} (n={n})")
        continue

    # 按 Cites 排序，确保划分含义一致（越高的 Cites => 更好排名）
    group_sorted = group.sort_values(by="Cites", ascending=False).reset_index(drop=True)

    train_end = int(0.6 * n)
    test_start = int(0.8 * n)

    train_df = group_sorted.iloc[:train_end].reset_index(drop=True)
    test_df = group_sorted.iloc[test_start:].reset_index(drop=True)

    X_train = train_df[features].values
    y_train = train_df[target].values.astype(float)
    X_test = test_df[features].values
    y_test = test_df[target].values.astype(float)

    # 标准化（基于训练集）
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 构建小型 MLP
    # 输入层 -> Dense(64) -> Dense(32) -> 输出（线性）
    model = keras.Sequential([
        layers.Input(shape=(len(features),)),
        layers.Dense(64, activation="relu"),
        layers.Dense(32, activation="relu"),
        layers.Dense(1, activation="linear")
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                  loss="mse",
                  metrics=[keras.metrics.MeanSquaredError()])

    # 训练（使用早停）
    early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
    history = model.fit(
        X_train_scaled, y_train,
        validation_split=0.1,
        epochs=200,
        batch_size=32,
        callbacks=[early_stop],
        verbose=0
    )

    # 预测并评估
    y_pred = model.predict(X_test_scaled).reshape(-1)
    mse = mean_squared_error(y_test, y_pred)
    # sklearn 的 MAPE 在新版本使用 mean_absolute_percentage_error
    try:
        mape = mean_absolute_percentage_error(y_test, y_pred)
    except Exception:
        # fallback: compute manually
        mape = np.mean(np.abs((y_test - y_pred) / np.where(y_test == 0, 1e-8, y_test)))

    # 保存模型与 scaler
    safe_name = "".join(ch if ch.isalnum() else "_" for ch in disc)[:120]
    model.save(os.path.join(OUT_DIR, "models", f"dl_model_{safe_name}.keras"))
    # 保存 scaler parameters
    scaler_df = pd.DataFrame({"mean": scaler.mean_, "scale": scaler.scale_}, index=features)
    scaler_df.to_csv(os.path.join(OUT_DIR, f"scaler_{safe_name}.csv"), encoding="utf-8-sig")

    # 记录汇总
    summary_rows.append({
        "Discipline": disc,
        "n_samples": n,
        "train_n": len(X_train),
        "test_n": len(X_test),
        "MSE": float(mse),
        "MAPE": float(mape)
    })

    # 保存测试预测对比
    test_out = test_df[["Institutions", "Discipline", "Cites", "Rank"]].copy()
    test_out["Pred_Rank_DL"] = y_pred
    test_out["Error"] = test_out["Pred_Rank_DL"] - test_out["Rank"]
    fname = os.path.join(OUT_DIR, f"predictions_{safe_name}.csv")
    test_out.to_csv(fname, index=False, encoding="utf-8-sig")
    all_test_results.append(test_out)

    print(f"[{disc}] n={n} train={len(X_train)} test={len(X_test)} MSE={mse:.3f} MAPE={mape:.3f}")

# 汇总所有学科评估
summary_df = pd.DataFrame(summary_rows).sort_values("MSE")
summary_df.to_csv(os.path.join(OUT_DIR, "dl_summary.csv"), index=False, encoding="utf-8-sig")
print("\n✅ 深度学习模型训练与评估完成，汇总保存为 dl_summary.csv")

# 可选：把所有测试集合并保存
if all_test_results:
    pd.concat(all_test_results, ignore_index=True).to_csv(os.path.join(OUT_DIR, "all_test_predictions.csv"),
                                                          index=False, encoding="utf-8-sig")

# ---------------------------
# 第12题：对 ESI 数据进行聚类，找出与华师大相似的学校，并分析原因
# ---------------------------
print("\n=== 第12题：ESI 聚类与 ECNU 相似学校分析 ===")

# 使用同一份原始 df（未按学科划分）
cluster_df = df.copy()
cluster_features = ["Web of Science Documents", "Cites", "Cites/Paper", "Top Papers"]
cluster_df = cluster_df.dropna(subset=cluster_features).reset_index(drop=True)

# 标准化并 KMeans
scaler_cl = StandardScaler()
CF = scaler_cl.fit_transform(cluster_df[cluster_features])

k = 6  # 聚类数可以调整
kmeans = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=20)
cluster_df["Cluster"] = kmeans.fit_predict(CF)

# 保存聚类分配
cluster_df.to_csv(os.path.join(OUT_DIR, "clustered_all.csv"), index=False, encoding="utf-8-sig")

# 找到 ECNU 在哪一簇
target_name = "EAST CHINA NORMAL UNIVERSITY"
ecnu_rows = cluster_df[cluster_df["Institutions"].str.contains(target_name, case=False, na=False)]
if ecnu_rows.empty:
    print("⚠️ 数据中未找到 EAST CHINA NORMAL UNIVERSITY，无法进行相似学校查找。")
else:
    ecnu_cluster = int(ecnu_rows["Cluster"].iloc[0])
    similars = cluster_df[cluster_df["Cluster"] == ecnu_cluster].copy()
    # 取按 Cites 排序的前 50 个作为相似学校列表（若簇内数量较少则全部）
    similars_top = similars.sort_values("Cites", ascending=False).head(50)
    out_similar_path = os.path.join(OUT_DIR, "ecnu_cluster_similars.csv")
    similars_top.to_csv(out_similar_path, index=False, encoding="utf-8-sig")
    print(f"ECNU 在簇 {ecnu_cluster}，已保存簇内相似高校（前50）到 {out_similar_path}")

    # 簇内特征均值，用于分析“为什么这些学校与 ECNU 类似”
    cluster_stats = cluster_df.groupby("Cluster")[cluster_features].mean().round(3)
    cluster_stats.to_csv(os.path.join(OUT_DIR, "cluster_analysis.csv"), encoding="utf-8-sig")
    print("已保存每个簇的特征均值到 cluster_analysis.csv")

    # 生成一张簇的雷达/条形比较图：ECNU vs 簇内均值
    ecnu_profile = cluster_df[cluster_df["Institutions"].str.contains(target_name, case=False, na=False)][cluster_features].mean()
    cluster_mean = cluster_stats.loc[ecnu_cluster]

    # 绘图：条形比较
    plt.figure(figsize=(8, 5))
    ind = np.arange(len(cluster_features))
    width = 0.35
    plt.bar(ind - width/2, ecnu_profile.values, width, label="ECNU")
    plt.bar(ind + width/2, cluster_mean.values, width, label=f"Cluster {ecnu_cluster} mean")
    plt.xticks(ind, cluster_features, rotation=20)
    plt.ylabel("Standardized / raw scale")
    plt.title("ECNU vs Cluster mean (features)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "ecnu_vs_cluster_mean.png"), dpi=300)
    print("已生成图 ecnu_vs_cluster_mean.png 供直观比较")

# ---------------------------
# 小结输出
# ---------------------------
print("\n--- 运行完成 ---")
print("输出目录:", OUT_DIR)
print("包含文件样例：", os.listdir(OUT_DIR)[:20])
print("请查看 dl_summary.csv（按学科 MSE / MAPE），以及 ecnu_cluster_similars.csv（与 ECNU 同簇高校）")
