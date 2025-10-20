import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

data_path = r"D:\DataScienceLab\lab5\data"

def load_all_data(path):
    datasets = {}
    print(f"📂 从 {path} 加载数据...\n")
    for file in os.listdir(path):
        if not file.endswith(".csv"):
            continue
        file_path = os.path.join(path, file)
        try:
            df = pd.read_csv(file_path, encoding='utf-8', sep=',', skiprows=1)
            df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
            df['discipline'] = os.path.splitext(file)[0]
            datasets[file] = df
            print(f"✅ 成功读取: {file} ({len(df)} 行)")
        except Exception as e:
            print(f"❌ 无法读取 {file}，错误：{e}")
    return datasets

def analyze_global_clusters(data_dict):
    merged = []
    for name, df in data_dict.items():
        if all(col in df.columns for col in ['institutions', 'countries/regions', 'cites/paper', 'web_of_science_documents']):
            temp = df[['institutions', 'countries/regions', 'cites/paper', 'web_of_science_documents']].copy()
            temp['discipline'] = name
            merged.append(temp)
    if not merged:
        raise ValueError("❌ 没有可用数据，请检查CSV结构")
    all_data = pd.concat(merged)
    all_data = all_data.dropna()
    X = all_data[['cites/paper', 'web_of_science_documents']].values
    X_scaled = StandardScaler().fit_transform(X)
    kmeans = KMeans(n_clusters=4, random_state=42)
    all_data['cluster'] = kmeans.fit_predict(X_scaled)
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(X_scaled)
    plt.scatter(reduced[:,0], reduced[:,1], c=all_data['cluster'])
    plt.title("全球高校聚类分布")
    plt.xlabel("PCA1"); plt.ylabel("PCA2")
    plt.savefig("global_clusters.png")
    print("\n✅ 已保存图像：global_clusters.png")
    print("📊 每类高校的代表国家：")
    print(all_data.groupby('cluster')['countries/regions'].agg(lambda x: x.value_counts().index[0]))
    return all_data

def analyze_ecnu_profile(data_dict):
    ecnu = []
    for name, df in data_dict.items():
        for col in df.columns:
            if 'institutions' in col:
                col_inst = col
                break
        sub = df[df[col_inst].str.contains("East China Normal University", case=False, na=False)]
        if not sub.empty:
            sub['discipline'] = name
            ecnu.append(sub)
    if not ecnu:
        print("⚠️ 数据中未找到华东师范大学记录")
        return
    ecnu_data = pd.concat(ecnu)
    print("\n🎓 华东师范大学学科画像：")
    print(ecnu_data[['discipline', 'cites/paper', 'web_of_science_documents']].head(10))
    plt.barh(ecnu_data['discipline'], ecnu_data['cites/paper'])
    plt.title("华东师范大学各学科影响力")
    plt.xlabel("Cites per Paper")
    plt.tight_layout()
    plt.savefig("ecnu_profile.png")
    print("✅ 已保存图像：ecnu_profile.png")

def build_ranking_model(data_dict):
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score, mean_absolute_error

    all_df = []
    for name, df in data_dict.items():
        df = df.rename(columns=lambda x: x.strip().lower())
        if all(c in df.columns for c in ['cites/paper', 'web_of_science_documents', 'rank']):
            temp = df[['cites/paper', 'web_of_science_documents', 'rank']].copy()
            temp['discipline'] = name
            all_df.append(temp)
    if not all_df:
        print("⚠️ 没有包含排名的学科数据")
        return
    df = pd.concat(all_df)
    df = df.dropna()
    X = df[['cites/paper', 'web_of_science_documents']]
    y = df['rank']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression().fit(X_train, y_train)
    preds = model.predict(X_test)
    print("\n📈 排名预测模型结果：")
    print("R² =", round(r2_score(y_test, preds), 3))
    print("MAE =", round(mean_absolute_error(y_test, preds), 3))

def main():
    data = load_all_data(data_path)
    print("\n=== 第8题：全球高校分类分析 ===")
    merged = analyze_global_clusters(data)
    print("\n=== 第9题：华东师范大学学科画像 ===")
    analyze_ecnu_profile(data)
    print("\n=== 第10题：学科排名预测模型 ===")
    build_ranking_model(data)

if __name__ == "__main__":
    main()
