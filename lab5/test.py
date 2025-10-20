import pandas as pd
import os

data_path = r"D:\DataScienceLab\lab5\data"
files = [f for f in os.listdir(data_path) if f.endswith('.csv')]

for f in files:
    file_path = os.path.join(data_path, f)
    print(f"\n=== 检查文件: {f} ===")
    try:
        with open(file_path, 'rb') as fh:
            first_200 = fh.read(200)
            print("前200字节：", first_200[:200])
        # 尝试不同编码和分隔符
        for enc in ['utf-8', 'utf-16', 'latin1']:
            for sep in [',', ';', '\t']:
                try:
                    df = pd.read_csv(file_path, sep=sep, encoding=enc, nrows=5)
                    print(f"✅ 成功读取: 编码={enc}, 分隔符='{sep}'，列={list(df.columns)[:5]}")
                    raise SystemExit
                except Exception:
                    continue
    except Exception as e:
        print("❌ 读取失败:", e)
