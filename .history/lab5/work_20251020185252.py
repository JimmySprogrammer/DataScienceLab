import pandas as pd
import os

def read_csv_safely(filepath):
    encodings = ['utf-8', 'utf-8-sig', 'latin1', 'iso-8859-1', 'cp1252']
    for enc in encodings:
        for skip in range(0, 10):  # 尝试跳过前0~9行
            try:
                df = pd.read_csv(filepath, encoding=enc, skiprows=skip)
                if len(df.columns) > 3 and not df.columns[0].startswith("Indicators"):
                    print(f"✅ 成功读取: {os.path.basename(filepath)}，编码={enc}，跳过前{skip}行，列={list(df.columns)[:5]}")
                    return df
            except Exception:
                continue
    print(f"❌ 无法解析 {os.path.basename(filepath)} 的数据结构")
    return None

def load_all_data(data_dir):
    data = []
    print(f"\n📂 从 {data_dir} 加载数据...\n")
    for fname in os.listdir(data_dir):
        if fname.endswith(".csv"):
            fpath = os.path.join(data_dir, fname)
            df = read_csv_safely(fpath)
            if df is not None and len(df.columns) > 1:
                df['Discipline'] = fname.replace('.csv', '')
                data.append(df)
    return data

def analyze_global_clusters(dataframes):
    if not dataframes:
        raise ValueError("❌ 没有可用数据，请检查CSV结构")
    merged = pd.concat(dataframes, ignore_index=True)
    print(f"✅ 成功合并 {len(dataframes)} 个学科，总行数: {len(merged)}")
    return merged

def main():
    data_dir = r"D:\DataScienceLab\lab5\data"
    data = load_all_data(data_dir)

    print("\n=== 第8题：全球高校分类分析 ===")
    merged = analyze_global_clusters(data)

    output_path = os.path.join(data_dir, "merged_all.csv")
    merged.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"✅ 已保存合并后的文件: {output_path}")

if __name__ == "__main__":
    main()
