import os
import pandas as pd
from glob import glob

def load_data(data_dir):
    csv_files = glob(os.path.join(data_dir, "*.csv"))
    data = {}
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            # 自动清理列名
            df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
            print(f"✅ 成功读取: {os.path.basename(file)} ({len(df)} 行)")
            data[os.path.splitext(os.path.basename(file))[0]] = df
        except Exception as e:
            print(f"❌ 读取失败: {file}, 错误: {e}")
    return data

def analyze_global_clusters(data):
    all_df = []
    for field, df in data.items():
        df['field'] = field

        # 尝试自动匹配列名
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

        df = df.rename(columns=rename_map)

        # 检查是否成功匹配到关键列
        missing_cols = [c for c in ['rank', 'cites_per_paper', 'documents', 'top_papers'] if c not in df.columns]
        if missing_cols:
            print(f"⚠️ {field} 缺少列: {missing_cols}, 将跳过")
            continue

        df = df.dropna(subset=['rank', 'cites_per_paper', 'documents', 'top_papers'])
        all_df.append(df)

    if not all_df:
        raise ValueError("❌ 没有找到包含有效数据的文件，请检查CSV结构")

    all_df = pd.concat(all_df, ignore_index=True)
    print(f"✅ 成功合并 {len(all_df)} 行数据用于聚类分析")
    return all_df

def main():
    data_dir = r"D:\DataScienceLab\lab5\data"
    data = load_data(data_dir)
    print("\n=== 第8题：全球高校分类分析 ===")
    all_df = analyze_global_clusters(data)

    summary = (
        all_df.groupby("field")
        .agg({
            "rank": "mean",
            "cites_per_paper": "mean",
            "documents": "sum",
            "top_papers": "sum"
        })
        .reset_index()
    )
    print("\n📊 各领域高校平均与总量指标：")
    print(summary.head())

if __name__ == "__main__":
    main()
