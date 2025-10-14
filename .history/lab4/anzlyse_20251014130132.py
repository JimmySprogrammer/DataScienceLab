import os
import pandas as pd
import sqlite3

data_dir = r"D:\DataScienceLab\lab4\data"
db_path = os.path.join(data_dir, "university_ranking.db")

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

cursor.execute("DROP TABLE IF EXISTS rankings")
cursor.execute("""
CREATE TABLE rankings (
    subject TEXT,
    rank INTEGER,
    institution TEXT,
    country TEXT,
    documents INTEGER,
    cites INTEGER,
    cites_per_paper REAL,
    top_papers INTEGER
)
""")

def clean_and_load_csv(file_path, subject):
    for enc in ["utf-8", "gbk", "latin1"]:
        try:
            df = pd.read_csv(file_path, encoding=enc, skiprows=1)
            break
        except Exception:
            df = None
    if df is None or df.shape[1] < 2:
        print(f"⚠️ 无法读取: {file_path}")
        return

    df.columns = [str(c).strip() for c in df.columns]
    df = df.dropna(how="all")
    df = df.rename(columns={
        df.columns[0]: "rank",
        df.columns[1]: "institution",
        df.columns[2]: "country",
        df.columns[3]: "documents",
        df.columns[4]: "cites",
        df.columns[5]: "cites_per_paper",
        df.columns[-1]: "top_papers"
    })
    df["subject"] = subject
    df = df[["subject", "rank", "institution", "country", "documents", "cites", "cites_per_paper", "top_papers"]]
    df.to_sql("rankings", conn, if_exists="append", index=False)
    print(f"✅ 导入成功: {subject} ({len(df)} rows)")

# 批量导入所有 CSV
for file in os.listdir(data_dir):
    if file.endswith(".csv"):
        subject = file.replace(".csv", "")
        clean_and_load_csv(os.path.join(data_dir, file), subject)

# === 查询 1: 华东师范大学在各学科的排名 ===
sql_ecnu = """
SELECT subject, rank, cites_per_paper, documents, top_papers
FROM rankings
WHERE institution LIKE '%EAST CHINA NORMAL UNIVERSITY%'
ORDER BY rank;
"""
ecnu_df = pd.read_sql_query(sql_ecnu, conn)
print("\n华东师范大学(ECNU) 各学科表现:")
print(ecnu_df if not ecnu_df.empty else "未找到记录")

# === 查询 2: 中国大陆高校在各学科的表现 ===
sql_china = """
SELECT subject,
       COUNT(*) AS num_institutions,
       AVG(rank) AS avg_rank,
       SUM(CASE WHEN rank <= 10 THEN 1 ELSE 0 END) AS top10,
       SUM(CASE WHEN rank <= 100 THEN 1 ELSE 0 END) AS top100
FROM rankings
WHERE country LIKE '%CHINA MAINLAND%'
GROUP BY subject;
"""
china_df = pd.read_sql_query(sql_china, conn)
print("\n中国大陆高校总体表现:")
print(china_df.head(10))

# === 查询 3: 全球各地区表现 ===
sql_region = """
SELECT subject,
       country,
       COUNT(*) AS num_institutions,
       AVG(rank) AS avg_rank
FROM rankings
GROUP BY subject, country
ORDER BY subject, avg_rank;
"""
region_df = pd.read_sql_query(sql_region, conn)
print("\n全球地区表现（前5行示例）:")
print(region_df.head(5))

# 导出结果
output_path = os.path.join(data_dir, "analysis_results.xlsx")
with pd.ExcelWriter(output_path) as writer:
    ecnu_df.to_excel(writer, sheet_name="ECNU", index=False)
    china_df.to_excel(writer, sheet_name="China", index=False)
    region_df.to_excel(writer, sheet_name="Regions", index=False)

print(f"\n✅ 已保存分析结果到: {output_path}")
conn.close()
