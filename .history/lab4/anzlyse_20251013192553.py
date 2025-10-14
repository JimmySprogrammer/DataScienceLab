import os
import sqlite3
import pandas as pd
import re

def main():
    # 1. 设置文件路径与数据库
    data_dir = r"D:\DataScienceLab\lab4\data"
    db_path = os.path.join(data_dir, "ranking.db")

    # 如果数据库存在则删除，避免重复导入
    if os.path.exists(db_path):
        os.remove(db_path)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 2. 定义数据库 schema（优化关系结构）
    cursor.executescript("""
    DROP TABLE IF EXISTS subjects;
    DROP TABLE IF EXISTS institutions;
    DROP TABLE IF EXISTS rankings;

    CREATE TABLE subjects (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE
    );

    CREATE TABLE institutions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        name_norm TEXT,
        country TEXT,
        region TEXT
    );

    CREATE TABLE rankings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        subject_id INTEGER,
        institution_id INTEGER,
        rank INTEGER,
        score REAL,
        FOREIGN KEY(subject_id) REFERENCES subjects(id),
        FOREIGN KEY(institution_id) REFERENCES institutions(id)
    );
    """)

    # 3. 读取所有CSV并导入数据库
    all_data = []
    csv_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
    print(f"Detected CSV files: {csv_files}")

    for file in csv_files:
        subject = os.path.splitext(file)[0].strip()
        file_path = os.path.join(data_dir, file)
        try:
            # 尝试不同编码打开
            try:
                df = pd.read_csv(file_path, encoding="utf-8")
            except UnicodeDecodeError:
                df = pd.read_csv(file_path, encoding="latin1")
        except Exception as e:
            print(f"❌ 读取失败: {file} ({e})")
            continue

        # 标准化列名
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

        # 尝试自动识别列
        possible_rank_cols = [c for c in df.columns if "rank" in c]
        possible_score_cols = [c for c in df.columns if "score" in c]
        possible_inst_cols = [c for c in df.columns if "institution" in c or "university" in c]
        possible_country_cols = [c for c in df.columns if "country" in c]

        if not possible_inst_cols or not possible_rank_cols:
            print(f"⚠️ 跳过 {file}: 缺少关键列")
            continue

        inst_col = possible_inst_cols[0]
        rank_col = possible_rank_cols[0]
        score_col = possible_score_cols[0] if possible_score_cols else None
        country_col = possible_country_cols[0] if possible_country_cols else None

        df = df[[inst_col, rank_col] + ([score_col] if score_col else []) + ([country_col] if country_col else [])]
        df.columns = ["institution", "rank", "score", "country"][:len(df.columns)]

        # 提取数字排名
        df["rank"] = df["rank"].astype(str).apply(lambda x: int(re.sub(r"\D", "", x)) if re.search(r"\d", x) else None)

        df["subject"] = subject
        all_data.append(df)

    if not all_data:
        print("❌ 没有成功加载的文件")
        return

    data = pd.concat(all_data, ignore_index=True)
    data.dropna(subset=["institution"], inplace=True)

    # 统一命名（大写转小写）
    data["institution_norm"] = data["institution"].str.lower().str.strip()
    data["country"] = data["country"].fillna("Unknown")

    # 插入 subject 表
    for subject in data["subject"].unique():
        cursor.execute("INSERT INTO subjects (name) VALUES (?)", (subject,))
    conn.commit()

    # 插入 institution 表
    for name, country in data[["institution", "institution_norm", "country"]].drop_duplicates(subset=["institution_norm"]).itertuples(index=False):
        region = (
            "Asia" if "china" in country.lower()
            else "North America" if country in ["USA", "United States"]
            else "Europe" if country in ["UK", "United Kingdom", "Germany", "France"]
            else "Other"
        )
        cursor.execute(
            "INSERT INTO institutions (name, name_norm, country, region) VALUES (?, ?, ?, ?)",
            (name, name.lower(), country, region)
        )
    conn.commit()

    # 插入 ranking 表
    for row in data.itertuples(index=False):
        cursor.execute("SELECT id FROM subjects WHERE name=?", (row.subject,))
        subject_id = cursor.fetchone()[0]

        cursor.execute("SELECT id FROM institutions WHERE name_norm=?", (row.institution_norm,))
        institution_id = cursor.fetchone()
        if not institution_id:
            continue
        institution_id = institution_id[0]

        cursor.execute(
            "INSERT INTO rankings (subject_id, institution_id, rank, score) VALUES (?, ?, ?, ?)",
            (subject_id, institution_id, row.rank, row.score if hasattr(row, "score") else None)
        )
    conn.commit()

    # ========================= 查询任务 =========================
    print("\n=== 华东师范大学在各学科的排名 ===")
    q_ecnu = """
    SELECT s.name AS subject, r.rank, r.score, i.name AS institution, i.country
    FROM rankings r
    JOIN subjects s ON r.subject_id = s.id
    JOIN institutions i ON r.institution_id = i.id
    WHERE i.name_norm LIKE '%east china normal university%'
       OR (i.name_norm LIKE '%华东%' AND i.name_norm LIKE '%师范%')
    ORDER BY s.name, r.rank;
    """
    ecnu_df = pd.read_sql_query(q_ecnu, conn)
    print(ecnu_df if not ecnu_df.empty else "⚠️ 没有找到ECNU数据")

    print("\n=== 中国（大陆）大学在各学科表现 ===")
    q_china = """
    SELECT s.name AS subject,
           COUNT(*) AS num_institutions_on_list,
           AVG(r.rank) AS avg_rank_of_listed,
           SUM(CASE WHEN r.rank <= 10 THEN 1 ELSE 0 END) AS num_top10,
           SUM(CASE WHEN r.rank <= 100 THEN 1 ELSE 0 END) AS num_top100
    FROM rankings r
    JOIN institutions i ON r.institution_id = i.id
    JOIN subjects s ON r.subject_id = s.id
    WHERE i.country LIKE '%China%'
    GROUP BY s.name
    ORDER BY avg_rank_of_listed ASC;
    """
    china_df = pd.read_sql_query(q_china, conn)
    print(china_df if not china_df.empty else "⚠️ 没有中国数据")

    print("\n=== 全球不同区域在各学科的表现 ===")
    q_region = """
    SELECT s.name AS subject, i.region, COUNT(*) AS num_records, ROUND(AVG(r.rank),2) AS avg_rank
    FROM rankings r
    JOIN institutions i ON r.institution_id = i.id
    JOIN subjects s ON r.subject_id = s.id
    GROUP BY s.name, i.region
    ORDER BY s.name, avg_rank ASC;
    """
    region_df = pd.read_sql_query(q_region, conn)
    print(region_df.head(20))

    # 导出结果
    out_path = os.path.join(data_dir, "analysis_results.xlsx")
    with pd.ExcelWriter(out_path) as writer:
        ecnu_df.to_excel(writer, sheet_name="ECNU", index=False)
        china_df.to_excel(writer, sheet_name="China", index=False)
        region_df.to_excel(writer, sheet_name="Region", index=False)

    print(f"\n✅ 所有结果已保存到 {out_path}")

    conn.close()

if __name__ == "__main__":
    main()
