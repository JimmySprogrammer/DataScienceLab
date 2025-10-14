import pandas as pd

paths = [
    r"D:\DataScienceLab\lab4\data\GEOSCIENCES.csv",
    r"D:\DataScienceLab\lab4\data\IMMUNOLOGY.csv"
]

for path in paths:
    print(f"== {path} ==")
    for enc in ["utf-8", "gbk", "latin1"]:
        try:
            df = pd.read_csv(path, encoding=enc)
            print(f"✅ 成功使用编码: {enc}")
            print("Columns:", df.columns.tolist())
            print(df.head(5))
            break
        except Exception as e:
            print(f"❌ 编码 {enc} 失败: {e}")
    print("\n")
