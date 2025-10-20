import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# ========== 1️⃣ 读取数据 ==========
data = pd.read_csv('data.csv', encoding='utf-8')
print(f"✅ 数据读取成功，共 {len(data)} 条记录，包含列：{list(data.columns)}")

# ========== 2️⃣ 题8：按学科建立多元线性回归模型 ==========
coef_list = []

for subject, group in data.groupby('学科'):
    X = group.drop(columns=['成绩', '学科'])
    y = group['成绩']
    
    model = LinearRegression()
    model.fit(X, y)
    
    coef_df = pd.DataFrame({
        '学科': subject,
        '特征': X.columns,
        '系数': model.coef_
    })
    coef_list.append(coef_df)

coef_all = pd.concat(coef_list, ignore_index=True)
coef_all.to_csv('result_q8_coefficients.csv', index=False, encoding='utf-8-sig')
print("📊 第8题：各学科回归系数已保存为 result_q8_coefficients.csv")

# ========== 3️⃣ 题9：模型评估（R² 与 MSE） ==========
eval_list = []

for subject, group in data.groupby('学科'):
    X = group.drop(columns=['成绩', '学科'])
    y = group['成绩']
    
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    
    eval_list.append({
        '学科': subject,
        'R²': r2,
        'MSE': mse,
        '样本数': len(group)
    })

eval_df = pd.DataFrame(eval_list)
eval_df.to_csv('result_q9_evaluation.csv', index=False, encoding='utf-8-sig')
print("📈 第9题：各学科模型评估结果已保存为 result_q9_evaluation.csv")

# ========== 4️⃣ 题10：优化版（前60%训练集、后20%测试集） ==========
split_eval = []

for subject, group in data.groupby('学科'):
    group = group.sort_values(by=group.columns[0])  # 用第1列排序（如学号或时间）
    n = len(group)
    train_end = int(0.6 * n)
    test_start = int(0.8 * n)
    
    train = group.iloc[:train_end]
    test = group.iloc[test_start:]
    
    if len(test) < 3:  # 样本过少跳过
        continue
    
    X_train = train.drop(columns=['成绩', '学科'])
    y_train = train['成绩']
    X_test = test.drop(columns=['成绩', '学科'])
    y_test = test['成绩']
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    
    split_eval.append({
        '学科': subject,
        'R²(测试集)': r2,
        'MSE(测试集)': mse,
        '训练样本数': len(train),
        '测试样本数': len(test)
    })

split_df = pd.DataFrame(split_eval)
split_df.to_csv('result_q10_split_evaluation.csv', index=False, encoding='utf-8-sig')
print("✅ 第10题：按比例划分训练/测试集的评估结果已保存为 result_q10_split_evaluation.csv")

# ========== 5️⃣ 可视化输出 ==========
plt.figure(figsize=(10, 6))
plt.bar(eval_df['学科'], eval_df['R²'], color='skyblue')
plt.title('各学科模型R²分布（题9）', fontsize=14)
plt.ylabel('R²')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('R2_by_subject.png', dpi=300)
plt.close()

plt.figure(figsize=(10, 6))
plt.bar(split_df['学科'], split_df['R²(测试集)'], color='lightgreen')
plt.title('各学科模型测试集R²（题10）', fontsize=14)
plt.ylabel('R²(测试集)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('R2_testset_by_subject.png', dpi=300)
plt.close()

print("🎨 图表已生成：R2_by_subject.png 与 R2_testset_by_subject.png")
print("✅ 实验8、9、10全部完成！结果文件已输出。")
