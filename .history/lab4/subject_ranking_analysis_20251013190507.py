# =============================================================
# 课业作业：大学排名数据分析 (使用 SQLite + SQL 查询)
# =============================================================

import os
import pandas as pd
import sqlite3

# ----------------------------
# 1. 设置数据路径与数据库文件
# ----------------------------
data_dir = r"D:\DataScienceLab\lab4\data"
db_path = os.path.join(data_dir, "university.db")

# ----------------------------
# 2. 读取所有 CSV 文件
# ----------------------------
csv_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
if not csv_files:
    raise FileNotFoundError("没有找到任何 CSV 文件，请检查路径！")

all_data = []
for file in csv_files:
    subject = os.path.splitext(file)[0]
    file_path = os.path.join(data_dir, file)
    df = pd.read_csv(file_path)
    df["Subject"] = subject
    all_data.append(df)

data = pd.concat(all_data, ignore_index=True)
print(f"共加载 {len(csv_files)} 个学科文件，总计 {len(data)} 条记录。")

# ----------------------------
# 3. 统一列名并清洗数据
# ----------------------------
data.columns = [c.strip().replace(" ", "_").lower() for c in data.columns]
if "university" not in data.columns:
    # 自动寻找可能的大学名列
    for c in data.columns:
        if "univer" in c:
            data.rename(columns={c: "university"}, inplace=True)
            break

if "country" not in data.columns:
    for c in data.columns:
        if "country" in c:
            data.rename(columns={c: "country"}, inplace=True)
            break

if "region" not in data.columns:
    data["region"] = data["country"]

if "rank" not in data.columns:
    for c in data.columns:
        if "rank" in c:
            data.rename(columns={c: "rank"}, inplace=True)
            break

data = data[["university", "country", "region", "subject", "rank"]].dropna()

# ----------------------------
# 4. 导入 SQLite 数据库
# ----------------------------
conn = sqlite3.connect(db_path)
data.to_sql("university_rankings", conn, if_exists="replace", index=False)
print("✅ 数据已导入 SQLite 数据库：university_rankings")

# ----------------------------
# 5. 执行 SQL 查询分析
# ----------------------------

# 华东师范大学在各学科的排名
query_ecnu = """
SELECT subject, rank
FROM university_rankings
WHERE university LIKE '%East China Normal%' OR university LIKE '%华东师范%'
ORDER BY rank;
"""
ecnu_rank = pd.read_sql_query(query_ecnu, conn)
print("\n🎓 华东师范大学在各学科的排名：")
display(ecnu_rank)

# 中国（大陆）大学在各学科的表现
query_china = """
SELECT subject,
       COUNT(*) AS num_universities,
       AVG(rank) AS avg_rank
FROM university_rankings
WHERE country LIKE '%China%'
GROUP BY subject
ORDER BY avg_rank;
"""
china_perf = pd.read_sql_query(query_china, conn)
print("\n🇨🇳 中国（大陆）大学在各学科的表现：")
display(china_perf)

# 全球不同区域在各学科的表现
query_region = """
SELECT region,
       subject,
       AVG(rank) AS avg_rank,
       COUNT(*) AS num_universities
FROM university_rankings
GROUP BY region, subject
ORDER BY subject, avg_rank;
"""
region_perf = pd.read_sql_query(query_region, conn)
print("\n🌍 全球不同区域在各学科的表现：")
display(region_perf)

# ----------------------------
# 6. 保存结果
# ----------------------------
excel_path = os.path.join(data_dir, "analysis_results.xlsx")
with pd.ExcelWriter(excel_path) as writer:
    ecnu_rank.to_excel(writer, sheet_name="ECNU_Ranking", index=False)
    china_perf.to_excel(writer, sheet_name="China_Performance", index=False)
    region_perf.to_excel(writer, sheet_name="Global_Regions", index=False)

print(f"\n📁 分析结果已导出到: {excel_path}")
conn.close()
