import pandas as pd

paths = [
    r"D:\DataScienceLab\lab4\data\GEOSCIENCES.csv",
    r"D:\DataScienceLab\lab4\data\IMMUNOLOGY.csv"
]

for path in paths:
    print(f"== {path} ==")
    try:
        df = pd.read_csv(path, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding='gbk')
    print("Columns:", df.columns.tolist())
    print(df.head(5))
    print("\n")
