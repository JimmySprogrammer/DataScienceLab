import os
import pandas as pd
from glob import glob

def load_data(data_dir):
    csv_files = glob(os.path.join(data_dir, "*.csv"))
    data = {}

    for file in csv_files:
        df = None
        encodings_to_try = ["utf-8", "latin1", "gb18030"]

        for enc in encodings_to_try:
            try:
                df = pd.read_csv(file, encoding=enc)
                print(f"✅ 成功读取: {os.path.basename(file)} ({len(df)} 行, 编码={enc})")
                break
            except Exception:
                continue

        if df is None:
            print(f"❌ 读取失败: {os.path.basename(file)}，编码不兼容")
            continue

        # 清洗列名
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

        # 自动字段匹配
        rename_map = {}
        for col in df.columns:
            if 'rank' in col:
                rename_map[col] = 'rank'
            elif 'cite' in col:
                rename_map[col] = 'cites_per_paper'
            elif 'doc' in col:
                rename_map[col] = 'documents'
            elif 'top' in col and 'paper' in col:
                rename_map[col] = 'top_papers'
            elif 'institution' in col:
                rename_map[col] = 'institution'
            elif 'country' in col:
                rename_map[col] = 'country'

        df = df.rename(columns=rename_map)

        data[os.path.splitext(os.path.basename(file))[0]] = df

    return data

def analyze_global_clusters(data):
    all_df = []
    for field, df in data.items():
        df['field'] = field
        required_cols = ['rank', 'cites_per_paper', 'documents', 'top_papers']

        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            print(f"⚠️ {field} 缺少列 {missing}，跳过该文件")
            continue

        df = df.dropna(subset=required_cols)
        all_df.append(df)

    if not all_df:
        raise ValueError("❌ 没有可用数据，请检查CSV结构")

    merged = pd.concat(all_df, ignore_index=True)
    print(f"✅ 成功合并 {len(merged)} 行数据")
    return merged

def analyze_universities(merged):
    # 华东师范大学的表现
    ecnus = merged[merged['institution'].str.contains("EAST CHINA NORMAL", case=False, na=False)]
    if not ecnus.empty:
        print("\n=== 华东师范大学各学科排名 ===")
        print(ecnus[['field', 'rank', 'cites_per_paper', 'documents', 'top_papers']])
    else:
        print("\n⚠️ 未找到华东师范大学相关记录")

    # 中国（大陆）大学表现
    cn_universities = merged[merged['country'].str.contains("CHINA", case=False, na=False)]
    print("\n=== 中国大陆高校总体表现（按学科平均排名） ===")
    summary_cn = cn_universities.groupby("field")["rank"].mean().reset_index().sort_values("rank")
    print(summary_cn)

    # 全球不同区域表现
    print("\n=== 全球不同区域表现（平均排名） ===")
    region_summary = merged.groupby("country")["rank"].mean().reset_index().sort_values("rank").head(10)
    print(region_summary)

def main():
    data_dir = r"D:\DataScienceLab\lab5\data"
    print(f"📂 正在从 {data_dir} 加载数据...\n")

    data = load_data(data_dir)
    print("\n=== 第8题：全球高校分类分析 ===")
    merged = analyze_global_clusters(data)
    analyze_universities(merged)

if __name__ == "__main__":
    main()
