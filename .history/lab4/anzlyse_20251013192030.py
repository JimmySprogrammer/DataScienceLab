import os
import glob
import sqlite3
import pandas as pd

data_dir = r"D:\DataScienceLab\lab4\data"
db_path = os.path.join(data_dir, "university.db")
csv_paths = glob.glob(os.path.join(data_dir, "*.csv"))

def try_read_csv(path):
    encodings = ["utf-8", "gbk", "latin1", "cp1252"]
    for e in encodings:
        try:
            return pd.read_csv(path, encoding=e, low_memory=False)
        except Exception:
            continue
    try:
        import chardet
        with open(path, "rb") as f:
            sample = f.read(100000)
            enc = chardet.detect(sample).get("encoding") or "latin1"
        return pd.read_csv(path, encoding=enc, low_memory=False)
    except Exception:
        return pd.read_csv(path, encoding="latin1", low_memory=False)

def find_col(cols, keywords):
    cols_lower = [c.lower() for c in cols]
    for kw in keywords:
        for i,c in enumerate(cols_lower):
            if kw in c:
                return cols[i]
    return None

dfs = []
for p in csv_paths:
    df = try_read_csv(p)
    df["__subject_file"] = os.path.splitext(os.path.basename(p))[0]
    dfs.append(df)
if not dfs:
    raise SystemExit("No CSV files found in " + data_dir)
raw = pd.concat(dfs, ignore_index=True)

cols = list(raw.columns)
inst_col = find_col(cols, ["institution","university","school","name"])
country_col = find_col(cols, ["country","nation","location"])
region_col = find_col(cols, ["region","area","continent"])
rank_col = find_col(cols, ["rank","position","pos","world rank"])
score_col = find_col(cols, ["score","points","total score","overall"])

if inst_col is None:
    inst_col = cols[0]
if country_col is None:
    raw["country"] = ""
    country_col = "country"
if region_col is None:
    raw["region"] = raw[country_col]
    region_col = "region"

raw = raw.rename(columns={inst_col: "institution", country_col: "country", region_col: "region", rank_col or "": "rank"})
if score_col:
    raw = raw.rename(columns={score_col: "score"})
raw["institution"] = raw["institution"].astype(str).str.strip()
raw["country"] = raw["country"].astype(str).str.strip()
raw["region"] = raw["region"].astype(str).str.strip()
raw["subject"] = raw["__subject_file"].astype(str).str.strip()
if "rank" in raw.columns:
    raw["rank"] = pd.to_numeric(raw["rank"].astype(str).str.extract(r"(\d+)", expand=False), errors="coerce")
else:
    raw["rank"] = pd.NA
if "score" in raw.columns:
    raw["score"] = pd.to_numeric(raw["score"], errors="coerce")
else:
    raw["score"] = pd.NA

def normalize_name(s):
    s = str(s).strip()
    s = s.replace(".", "").replace(",", "").lower()
    s = " ".join(s.split())
    return s

raw["inst_norm"] = raw["institution"].apply(normalize_name)
raw["country_norm"] = raw["country"].astype(str).str.lower().str.strip()

conn = sqlite3.connect(db_path)
cur = conn.cursor()
cur.executescript("""
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS subjects (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT UNIQUE NOT NULL);
CREATE TABLE IF NOT EXISTS institutions (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, name_norm TEXT NOT NULL, country TEXT, region TEXT, UNIQUE(name_norm, country));
CREATE TABLE IF NOT EXISTS rankings (id INTEGER PRIMARY KEY AUTOINCREMENT, subject_id INTEGER NOT NULL, institution_id INTEGER NOT NULL, rank INTEGER, score REAL, extra TEXT, FOREIGN KEY(subject_id) REFERENCES subjects(id), FOREIGN KEY(institution_id) REFERENCES institutions(id));
""")
conn.commit()

def get_or_create_subject(name):
    cur.execute("SELECT id FROM subjects WHERE name = ?", (name,))
    r = cur.fetchone()
    if r:
        return r[0]
    cur.execute("INSERT INTO subjects (name) VALUES (?)", (name,))
    conn.commit()
    return cur.lastrowid

def get_or_create_institution(name, name_norm, country, region):
    cur.execute("SELECT id FROM institutions WHERE name_norm = ? AND country = ?", (name_norm, country))
    r = cur.fetchone()
    if r:
        return r[0]
    cur.execute("INSERT INTO institutions (name, name_norm, country, region) VALUES (?,?,?,?)", (name, name_norm, country, region))
    conn.commit()
    return cur.lastrowid

subjects = raw["subject"].unique().tolist()
for subj in subjects:
    get_or_create_subject(subj)

for _, row in raw.iterrows():
    subj = row["subject"]
    sid = get_or_create_subject(subj)
    name = row["institution"]
    name_norm = row["inst_norm"]
    country = row["country"] if pd.notna(row["country"]) else ""
    region = row["region"] if pd.notna(row["region"]) else ""
    iid = get_or_create_institution(name, name_norm, country, region)
    rnk = int(row["rank"]) if pd.notna(row["rank"]) else None
    scr = float(row["score"]) if ("score" in row and pd.notna(row["score"])) else None
    cur.execute("INSERT INTO rankings (subject_id, institution_id, rank, score, extra) VALUES (?,?,?,?,?)", (sid, iid, rnk, scr, None))
conn.commit()

q_ecnu = """
SELECT s.name AS subject, r.rank, r.score, i.name AS institution, i.country
FROM rankings r
JOIN subjects s ON r.subject_id = s.id
JOIN institutions i ON r.institution_id = i.id
WHERE i.name_norm LIKE '%east china normal%' OR i.name_norm LIKE '%华东师范%'
ORDER BY s.name, r.rank;
"""
ecnu_df = pd.read_sql_query(q_ecnu, conn)
print("ECNU results:")
print(ecnu_df.to_string(index=False))

q_china = """
SELECT s.name AS subject,
       COUNT(DISTINCT i.id) AS num_institutions_on_list,
       AVG(r.rank) AS avg_rank_of_listed,
       SUM(CASE WHEN r.rank<=10 THEN 1 ELSE 0 END) AS num_top10,
       SUM(CASE WHEN r.rank<=100 THEN 1 ELSE 0 END) AS num_top100
FROM rankings r
JOIN subjects s ON r.subject_id = s.id
JOIN institutions i ON r.institution_id = i.id
WHERE i.country LIKE '%China%' OR i.country LIKE '%中国%' OR i.country LIKE '%mainland%'
GROUP BY s.name
ORDER BY s.name;
"""
china_df = pd.read_sql_query(q_china, conn)
print("\nChina (Mainland) performance by subject:")
print(china_df.to_string(index=False))

region_map = {
    "china":"Asia","chinese":"Asia","united states":"North America","usa":"North America","us":"North America",
    "england":"Europe","uk":"Europe","united kingdom":"Europe","france":"Europe","germany":"Europe",
    "japan":"Asia","india":"Asia","australia":"Oceania","canada":"North America"
}
cur.execute("DROP TABLE IF EXISTS tmp_inst_region;")
cur.execute("CREATE TABLE tmp_inst_region AS SELECT id, name, country, region FROM institutions;")
for k,v in region_map.items():
    cur.execute("UPDATE tmp_inst_region SET region = ? WHERE lower(country) LIKE ?", (v, f"%{k}%"))
cur.execute("UPDATE tmp_inst_region SET region = COALESCE(region, 'Other') WHERE region IS NULL OR region = ''")
conn.commit()

q_region = """
SELECT s.name AS subject, t.region,
       COUNT(r.id) AS num_records,
       AVG(r.rank) AS avg_rank
FROM rankings r
JOIN subjects s ON r.subject_id = s.id
JOIN tmp_inst_region t ON r.institution_id = t.id
GROUP BY s.name, t.region
ORDER BY s.name, num_records DESC;
"""
region_df = pd.read_sql_query(q_region, conn)
print("\nRegion performance (subject x region):")
print(region_df.to_string(index=False))

out_xlsx = os.path.join(data_dir, "analysis_results.xlsx")
with pd.ExcelWriter(out_xlsx) as writer:
    ecnu_df.to_excel(writer, sheet_name="ECNU", index=False)
    china_df.to_excel(writer, sheet_name="China_Mainland", index=False)
    region_df.to_excel(writer, sheet_name="Region_By_Subject", index=False)

conn.close()
print("\nSaved results to", out_xlsx)
