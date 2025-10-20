import os
import pandas as pd
from glob import glob

def load_data(data_dir):
    csv_files = glob(os.path.join(data_dir, "*.csv"))
    data = {}

    for file in csv_files:
        df = None
        encodings = ["utf-8", "latin1", "gb18030"]
        for enc in encodings:
            try:
                df = pd.read_csv(file, encoding=enc, header=None)
                break
            except Exception:
                continue

        if df is None:
            print(f"❌ 无法读取 {os.path.basename(file)}，跳过。")
            continue

        # 找出第一行中包含关键字 "Institutions" 的行，作为 header
        header_row = None
        for i in range(min(10, len(df))):
            if df.iloc[i].astype(str).str.contains("Institution", case=False, na=False).any():
                header_row = i
                break

        if header_row is None:
            print(f"⚠️ 未找到表头: {os.path.basename(file)}，跳过。")
            continue

        df.columns = df.iloc[header_row]
        df = df.drop(index=range(header_row + 1))
        df = df.reset_index(drop=True)

        df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

        rename_map = {}
        for c in df.columns:
            if "institution" in c:
                rename_map[c] = "institution"
            elif "country" in c:
                rename_map[c] = "country"
            elif "document" in c:
                rename_map[c] = "documents"
            elif "cite/paper" in c or "cites_per_paper" in c:
                rename_map[c] = "cites_per_paper"
            elif "top" in c and "paper" in c:
                rename_map[c] = "top_papers"
            elif "rank" in c:
                rename_map[c] = "rank"

        df = df.rename(columns=rename_map)

        # 保留关键列
        keep_cols = ["institution", "country", "documents", "cites_per_paper", "top_papers"]
        df = df[[c for c in keep_cols if c in df.columns]]
        df["field"] = os.path.splitext(os.path.basename(file))[0]
        data[os.path.basename(file)] = df

        print(f"✅ 成功解析: {os.path.basename(file)} ({len(df)} 行)")

    return data

def analyze_global_clusters(data):
    all_df = []
    for name, df in data.items():
        if len(df) < 10:
            continue
        all_df.append(df)
    if not all_df:
        raise ValueError("❌ 没有可用数据，请检查CSV结构")

    merged = pd.concat(all_df, ignore_index=True)
    print(f"✅ 成功合并 {len(merged)} 行数据")
    return merged

def analyze_universities(merged):
    ecnus = merged[merged["institution"].str.contains("EAST CHINA NORMAL", case=False, na=False)]
    if not ecnus.empty:
        print("\n=== 华东师范大学各学科表现 ===")
        print(ecnus[["field", "documents", "cites_per_paper", "top_papers"]])
    else:
        print("\n⚠️ 未找到华东师范大学记录")

    cn_universities = merged[merged["country"].str.contains("CHINA", case=False, na=False)]
    print("\n=== 中国高校总体表现（按学科平均引文数） ===")
    summary_cn = cn_universities.groupby("field")["cites_per_paper"].mean().reset_index().sort_values("cites_per_paper", ascending=False)
    print(summary_cn)

    print("\n=== 全球地区表现（平均引文数Top10） ===")
    region_summary = merged.groupby("country")["cites_per_paper"].mean().reset_index().sort_values("cites_per_paper", ascending=False).head(10)
    print(region_summary)

def main():
    data_dir = r"D:\DataScienceLab\lab5\data"
    print(f"📂 从 {data_dir} 加载数据...\n")
    data = load_data(data_dir)
    print("\n=== 第8题：全球高校分类分析 ===")
    merged = analyze_global_clusters(data)
    analyze_universities(merged)

if __name__ == "__main__":
    main()
