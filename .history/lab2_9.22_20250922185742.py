# -*- coding: utf-8 -*-
"""
房屋价格数据预处理完整代码
实验二：数据预处理的基本方法
"""

# =============================================================================
# 1. 导入必要的库
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

print("库导入完成")

# =============================================================================
# 2. 加载数据并进行初步探索
# =============================================================================
try:
    # 加载数据
    df = pd.read_csv('train.csv')
    print(f"数据加载成功！数据集形状: {df.shape}")
    
    # 检查目标变量是否存在（可能是SalePrice或price）
    if 'SalePrice' in df.columns:
        target_col = 'SalePrice'
    elif 'price' in df.columns:
        target_col = 'price'
    else:
        # 如果都没有，假设第一列是目标变量或者使用其他逻辑
        target_col = df.columns[-1]  # 假设最后一列是目标变量
        print(f"未找到标准的目标变量名，使用 '{target_col}' 作为目标变量")
    
    # 初步数据探索
    print("\n数据基本信息:")
    print(f"数据集形状: {df.shape}")
    print(f"行数: {df.shape[0]}, 列数: {df.shape[1]}")
    
    print("\n前3行数据:")
    display(df.head(3))
    
    print("\n数据列信息:")
    df.info()
    
    print("\n数值型特征的描述性统计:")
    display(df.describe())
    
except FileNotFoundError:
    print("错误: 未找到 'train.csv' 文件")
    print("请确保文件在当前工作目录中")
    # 创建示例数据框架供演示（实际使用时请注释掉）
    print("创建示例数据供演示...")
    np.random.seed(42)
    n_samples = 1000
    df = pd.DataFrame({
        'SalePrice': np.random.normal(180000, 50000, n_samples),
        'OverallQual': np.random.randint(1, 11, n_samples),
        'GrLivArea': np.random.normal(1500, 500, n_samples),
        'GarageCars': np.random.randint(0, 4, n_samples),
        'TotalBsmtSF': np.random.normal(1000, 300, n_samples),
        'LotArea': np.random.normal(10000, 3000, n_samples),
        'YearBuilt': np.random.randint(1950, 2010, n_samples),
        'MSSubClass': np.random.choice([20, 30, 40, 50, 60, 70, 80, 90, 120, 150, 160, 180, 190], n_samples),
        'MSZoning': np.random.choice(['RL', 'RM', 'FV', 'RH'], n_samples),
        'LotFrontage': np.random.normal(70, 20, n_samples)
    })
    # 人为添加一些缺失值
    for col in ['LotFrontage', 'GarageCars']:
        df.loc[df.sample(frac=0.1).index, col] = np.nan
    target_col = 'SalePrice'
    print("✅ 示例数据创建完成")

# =============================================================================
# 3. 缺失值检测与处理
# =============================================================================
print("\n" + "="*60)
print("3. 缺失值检测与处理")
print("="*60)

# 3.1 缺失值检测
missing_ratio = (df.isnull().sum() / len(df)) * 100
missing_data = pd.DataFrame({
    '缺失数量': df.isnull().sum(),
    '缺失比例%': missing_ratio
})
missing_data = missing_data[missing_data['缺失数量'] > 0].sort_values('缺失比例%', ascending=False)

print("缺失值统计:")
if len(missing_data) > 0:
    display(missing_data)
else:
    print("数据集中没有缺失值")

# 可视化缺失值
if len(missing_data) > 0:
    plt.figure(figsize=(12, 6))
    missing_ratio_plot = missing_ratio[missing_ratio > 0].sort_values(ascending=False)
    sns.barplot(x=missing_ratio_plot.index, y=missing_ratio_plot.values)
    plt.xticks(rotation=90)
    plt.title('特征缺失值比例')
    plt.ylabel('缺失百分比 (%)')
    plt.tight_layout()
    plt.show()

# 3.2 缺失值处理
df_cleaned = df.copy()

print("\n开始处理缺失值...")

if len(missing_data) > 0:
    # 策略1：删除缺失率过高的特征（阈值设为20%）
    high_missing_threshold = 20
    high_missing_columns = missing_ratio[missing_ratio > high_missing_threshold].index
    if len(high_missing_columns) > 0:
        df_cleaned = df_cleaned.drop(columns=high_missing_columns)
        print(f"已删除缺失率 > {high_missing_threshold}% 的特征: {list(high_missing_columns)}")
    else:
        print(f"没有缺失率 > {high_missing_threshold}% 的特征")

    # 更新类别和数值特征列表
    cat_columns = df_cleaned.select_dtypes(include=['object']).columns
    num_columns = df_cleaned.select_dtypes(include=[np.number]).columns

    # 策略2：填充类别特征
    categorical_fill_columns = [col for col in cat_columns if col in df_cleaned.columns and df_cleaned[col].isnull().sum() > 0]
    
    if len(categorical_fill_columns) > 0:
        print("🔧 处理类别特征缺失值...")
        for col in categorical_fill_columns:
            # 对于表示"没有"的特征，用'None'填充
            none_fill_columns = ['Alley', 'Fence', 'MiscFeature', 'FireplaceQu', 'GarageType', 
                               'GarageFinish', 'GarageQual', 'GarageCond', 'BsmtQual', 
                               'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'PoolQC']
            
            if col in none_fill_columns:
                df_cleaned[col].fillna('None', inplace=True)
                print(f"   {col}: 用 'None' 填充")
            else:
                # 其他类别特征用众数填充
                mode_val = df_cleaned[col].mode()[0] if not df_cleaned[col].mode().empty else 'Unknown'
                df_cleaned[col].fillna(mode_val, inplace=True)
                print(f"   {col}: 用众数 '{mode_val}' 填充")

    # 策略3：填充数值特征
    numerical_fill_columns = [col for col in num_columns if col in df_cleaned.columns and df_cleaned[col].isnull().sum() > 0]
    
    if len(numerical_fill_columns) > 0:
        print("🔧 处理数值特征缺失值...")
        # 使用中位数填充（更稳健）
        for col in numerical_fill_columns:
            if col != target_col:  # 不填充目标变量
                median_val = df_cleaned[col].median()
                df_cleaned[col].fillna(median_val, inplace=True)
                print(f"   {col}: 用中位数 {median_val:.2f} 填充")

    # 验证是否还有缺失值
    remaining_missing = df_cleaned.isnull().sum().sum()
    print(f"缺失值处理完成！剩余缺失值数量: {remaining_missing}")

else:
    print("数据集中没有缺失值，跳过缺失值处理步骤")

print(f"清理后数据集形状: {df_cleaned.shape}")

# =============================================================================
# 4. 异常值检测
# =============================================================================
print("\n" + "="*60)
print("4. 异常值检测")
print("="*60)

# 更新数值列（处理缺失值后）
num_columns_cleaned = df_cleaned.select_dtypes(include=[np.number]).columns

# 4.1 Z-score 异常值检测 (|Z| > 3)
print("使用Z-score方法检测异常值...")
z_scores = stats.zscore(df_cleaned[num_columns_cleaned], nan_policy='omit')
# 处理可能的NaN值（由于标准差为0导致的）
z_scores = np.where(np.isnan(z_scores), 0, z_scores)

outliers_z = (np.abs(z_scores) > 3).sum(axis=0)
outliers_z_data = pd.DataFrame({
    '特征': num_columns_cleaned,
    '异常值数量': outliers_z
}).sort_values('异常值数量', ascending=False)

print("Z-score检测结果（异常值数量>0的特征）:")
display(outliers_z_data[outliers_z_data['异常值数量'] > 0].head(10))

# 4.2 IQR 异常值检测
print("\n使用IQR方法检测异常值...")
def detect_outliers_iqr(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return series[(series < lower_bound) | (series > upper_bound)]

# 对关键数值特征进行IQR检测
key_features = ['LotArea', 'GrLivArea', 'TotalBsmtSF', target_col]
key_features = [f for f in key_features if f in df_cleaned.columns]

outliers_iqr_summary = {}
for feature in key_features:
    outliers = detect_outliers_iqr(df_cleaned[feature])
    outliers_iqr_summary[feature] = len(outliers)

outliers_iqr_data = pd.DataFrame.from_dict(outliers_iqr_summary, 
                                         orient='index', 
                                         columns=['异常值数量']).sort_values('异常值数量', ascending=False)

print("IQR检测结果（关键特征）:")
display(outliers_iqr_data)

# 4.3 可视化关键特征的异常值
print("\n异常值可视化...")
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.ravel()

for i, feature in enumerate(key_features[:4]):  # 只显示前4个特征
    if i < len(axes):
        # 箱线图
        df_cleaned.boxplot(column=feature, ax=axes[i])
        axes[i].set_title(f'{feature} - 箱线图')
        
plt.tight_layout()
plt.show()

# 4.4 散点图可视化（GrLivArea vs SalePrice）
if 'GrLivArea' in df_cleaned.columns and target_col in df_cleaned.columns:
    plt.figure(figsize=(10, 6))
    plt.scatter(df_cleaned['GrLivArea'], df_cleaned[target_col], alpha=0.6)
    plt.title(f'GrLivArea vs {target_col} (异常值检测)')
    plt.xlabel('Above grade living area (平方英尺)')
    plt.ylabel(f'{target_col} (美元)')
    
    # 标记可能的异常值区域
    Q1_gr = df_cleaned['GrLivArea'].quantile(0.25)
    Q3_gr = df_cleaned['GrLivArea'].quantile(0.75)
    IQR_gr = Q3_gr - Q1_gr
    upper_bound_gr = Q3_gr + 1.5 * IQR_gr
    
    Q1_price = df_cleaned[target_col].quantile(0.25)
    Q3_price = df_cleaned[target_col].quantile(0.75)
    IQR_price = Q3_price - Q1_price
    lower_bound_price = Q1_price - 1.5 * IQR_price
    
    # 标记异常区域
    plt.axvline(x=upper_bound_gr, color='red', linestyle='--', alpha=0.7, label='GrLivArea异常阈值')
    plt.axhline(y=lower_bound_price, color='orange', linestyle='--', alpha=0.7, label=f'{target_col}异常阈值')
    plt.legend()
    plt.show()

# 4.5 处理异常值（选择性删除）
print("\n处理异常值...")
original_shape = df_cleaned.shape

# 示例：删除GrLivArea过大但价格异常低的点
if 'GrLivArea' in df_cleaned.columns and target_col in df_cleaned.columns:
    # 计算IQR边界
    Q1_gr = df_cleaned['GrLivArea'].quantile(0.25)
    Q3_gr = df_cleaned['GrLivArea'].quantile(0.75)
    IQR_gr = Q3_gr - Q1_gr
    upper_bound_gr = Q3_gr + 1.5 * IQR_gr
    
    Q1_price = df_cleaned[target_col].quantile(0.25)
    Q3_price = df_cleaned[target_col].quantile(0.75)
    IQR_price = Q3_price - Q1_price
    lower_bound_price = Q1_price - 1.5 * IQR_price
    
    # 删除明显的异常点（右下角的点：面积大但价格低）
    outlier_mask = (df_cleaned['GrLivArea'] > upper_bound_gr) & (df_cleaned[target_col] < lower_bound_price)
    df_cleaned = df_cleaned[~outlier_mask]
    
    removed_count = outlier_mask.sum()
    print(f"删除了 {removed_count} 个明显异常点")

print(f"异常值处理完成！数据集从 {original_shape} 变为 {df_cleaned.shape}")

# =============================================================================
# 5. 特征间的相关性分析
# =============================================================================
print("\n" + "="*60)
print("5. 特征间的相关性分析")
print("="*60)

# 更新数值列
num_columns_final = df_cleaned.select_dtypes(include=[np.number]).columns

if target_col in num_columns_final:
    # 5.1 计算与目标变量的相关性
    correlation_with_target = df_cleaned[num_columns_final].corr()[target_col].sort_values(ascending=False)
    
    print(f"特征与 {target_col} 的相关性排名:")
    print("正相关最高的10个特征:")
    display(correlation_with_target.head(11))  # 显示11个（包含目标变量自身）
    
    print("\n负相关最高的5个特征:")
    display(correlation_with_target.tail(5))
    
    # 5.2 绘制相关性热图（前15个相关特征）
    top_corr_features = correlation_with_target.head(16).index  # 取前16个（包含目标变量）
    top_corr_features = [f for f in top_corr_features if f != target_col][:15]  # 排除目标变量，取15个
    top_corr_features.append(target_col)  # 最后加入目标变量
    
    plt.figure(figsize=(12, 10))
    correlation_matrix = df_cleaned[top_corr_features].corr()
    
    # 创建mask来隐藏上三角
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    
    sns.heatmap(correlation_matrix, 
                mask=mask,
                annot=True, 
                cmap='RdBu_r', 
                center=0,
                fmt='.2f',
                square=True,
                cbar_kws={"shrink": .8})
    plt.title('Top 15 数值特征相关性热图')
    plt.tight_layout()
    plt.show()
    
    # 5.3 绘制相关性条形图
    plt.figure(figsize=(12, 6))
    top_15_corr = correlation_with_target.head(16)[1:16]  # 排除目标变量自身，取2-16
    colors = ['red' if x > 0 else 'blue' for x in top_15_corr.values]
    
    sns.barplot(x=top_15_corr.values, y=top_15_corr.index, palette=colors)
    plt.title(f'与 {target_col} 相关性最高的15个特征')
    plt.xlabel('相关系数')
    plt.tight_layout()
    plt.show()
    
else:
    print(f"目标变量 {target_col} 不是数值类型，无法进行相关性分析")

# =============================================================================
# 6. 对 Price 属性进行标准化
# =============================================================================
print("\n" + "="*60)
print("6. 对 Price 属性进行标准化")
print("="*60)

if target_col in df_cleaned.columns:
    # 创建标准化后的新列
    scaler = StandardScaler()
    standardized_col_name = f'{target_col}_Standardized'
    
    df_cleaned[standardized_col_name] = scaler.fit_transform(df_cleaned[[target_col]])
    
    # 查看标准化前后的对比
    print("标准化前后对比:")
    print(f"原始 {target_col} - 均值: {df_cleaned[target_col].mean():.2f}, 标准差: {df_cleaned[target_col].std():.2f}")
    print(f"标准化后 - 均值: {df_cleaned[standardized_col_name].mean():.2f}, 标准差: {df_cleaned[standardized_col_name].std():.2f}")
    
    # 可视化标准化前后的分布
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # 原始分布
    ax1.hist(df_cleaned[target_col], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    ax1.axvline(df_cleaned[target_col].mean(), color='red', linestyle='--', label=f'均值: {df_cleaned[target_col].mean():.2f}')
    ax1.set_title(f'原始 {target_col} 分布')
    ax1.set_xlabel(target_col)
    ax1.set_ylabel('频率')
    ax1.legend()
    
    # 标准化后分布
    ax2.hist(df_cleaned[standardized_col_name], bins=30, color='lightcoral', edgecolor='black', alpha=0.7)
    ax2.axvline(df_cleaned[standardized_col_name].mean(), color='red', linestyle='--', label=f'均值: {df_cleaned[standardized_col_name].mean():.2f}')
    ax2.set_title(f'标准化后的 {target_col} 分布')
    ax2.set_xlabel('Standardized ' + target_col)
    ax2.set_ylabel('频率')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
    
else:
    print(f"目标变量 {target_col} 不存在于数据集中")

# =============================================================================
# 7. 根据 Price 属性进行离散化
# =============================================================================
print("\n" + "="*60)
print("7. 根据 Price 属性进行离散化")
print("="*60)

if target_col in df_cleaned.columns:
    # 方法1：等宽分箱（基于值范围）
    df_cleaned['Price_Category_EqualWidth'] = pd.cut(df_cleaned[target_col], 
                                                   bins=3, 
                                                   labels=['低价', '中价', '高价'])
    
    # 方法2：等频分箱（基于数据量）
    df_cleaned['Price_Category_EqualFreq'] = pd.qcut(df_cleaned[target_col], 
                                                    q=3, 
                                                    labels=['低价', '中价', '高价'])
    
    print("离散化结果统计:")
    
    print("\n等宽分箱结果:")
    equal_width_counts = df_cleaned['Price_Category_EqualWidth'].value_counts().sort_index()
    display(equal_width_counts)
    
    print("\n等频分箱结果:")
    equal_freq_counts = df_cleaned['Price_Category_EqualFreq'].value_counts().sort_index()
    display(equal_freq_counts)
    
    # 可视化离散化结果
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # 等宽分箱
    equal_width_counts.plot(kind='bar', ax=ax1, color=['lightblue', 'lightgreen', 'salmon'])
    ax1.set_title('等宽分箱 - 价格等级分布')
    ax1.set_xlabel('价格等级')
    ax1.set_ylabel('数量')
    
    # 等频分箱
    equal_freq_counts.plot(kind='bar', ax=ax2, color=['lightblue', 'lightgreen', 'salmon'])
    ax2.set_title('等频分箱 - 价格等级分布')
    ax2.set_xlabel('价格等级')
    ax2.set_ylabel('数量')
    
    plt.tight_layout()
    plt.show()
    
    # 显示每个价格区间的实际范围
    print("\n等宽分箱的价格范围:")
    price_ranges = pd.cut(df_cleaned[target_col], bins=3)
    print(price_ranges.value_counts().sort_index())
    
else:
    print(f"目标变量 {target_col} 不存在于数据集中")

# =============================================================================
# 8. 找出与 Price 相关性最高的三个特征并解释
# =============================================================================
print("\n" + "="*60)
print("8. 找出与 Price 相关性最高的三个特征并解释")
print("="*60)

if target_col in num_columns_final:
    # 获取与目标变量相关性最高的三个特征（排除目标变量自身）
    top_3_features = correlation_with_target.index[1:4]  # 索引0是目标变量自身
    top_3_corr_values = correlation_with_target.values[1:4]
    
    print("与房价相关性最高的三个特征是:")
    for i, (feature, corr_val) in enumerate(zip(top_3_features, top_3_corr_values), 1):
        print(f"{i}. {feature}: {corr_val:.4f}")
    
    # 绘制这三个特征与房价的关系图
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for i, (feature, corr_val) in enumerate(zip(top_3_features, top_3_corr_values)):
        if feature in df_cleaned.columns:
            axes[i].scatter(df_cleaned[feature], df_cleaned[target_col], alpha=0.6, color=f'C{i}')
            axes[i].set_title(f'{feature} vs {target_col}\n(相关系数: {corr_val:.3f})')
            axes[i].set_xlabel(feature)
            axes[i].set_ylabel(target_col)
            
            # 添加趋势线
            z = np.polyfit(df_cleaned[feature], df_cleaned[target_col], 1)
            p = np.poly1d(z)
            axes[i].plot(df_cleaned[feature], p(df_cleaned[feature]), "r--", alpha=0.8)
    
    plt.tight_layout()
    plt.show()
    
    # 对这三个特征进行详细解释
    print("\n特征解释:")
    
    feature_explanations = {
        'OverallQual': {
            '解释': '整体材料和工艺质量',
            '说明': '这是评估房屋建造质量的最重要指标。高质量意味着更好的建筑材料、精美的装修和卓越的工艺，直接影响房屋的价值。',
            '影响': '质量等级每提高一级，房价通常会有显著提升'
        },
        'GrLivArea': {
            '解释': '地面以上居住面积',
            '说明': '代表房屋的实际可用居住空间大小。更大的居住面积意味着更多的功能空间和舒适度，是房价的基本决定因素。',
            '影响': '面积与房价呈强正相关，是购房者最关注的指标之一'
        },
        'GarageCars': {
            '解释': '车库容量（可停放车辆数）',
            '说明': '在现代生活中，车库不仅是停车场所，也是重要的存储空间。能容纳更多车辆的车库大大增加了房屋的实用性。',
            '影响': '车库容量越大，房屋价值越高，特别是在汽车普及的地区'
        },
        'TotalBsmtSF': {
            '解释': '地下室总面积',
            '说明': '地下室提供了额外的可用空间，可用于存储、娱乐或扩展居住面积，增加了房屋的功能性。',
            '影响': '地下室面积与房价正相关，但相关性通常低于地上居住面积'
        },
        '1stFlrSF': {
            '解释': '一层面积',
            '说明': '房屋主要生活区域的大小，直接影响居住体验和功能性。',
            '影响': '一层面积是居住面积的重要组成部分，与房价强相关'
        }
    }
    
    for i, feature in enumerate(top_3_features, 1):
        if feature in feature_explanations:
            explanation = feature_explanations[feature]
            print(f"\n{i}. {feature}:")
            print(f"解释: {explanation['解释']}")
            print(f"说明: {explanation['说明']}")
            print(f"影响: {explanation['影响']}")
        else:
            print(f"\n{i}. {feature}:")
            print("这是与房价高度相关的数值特征，具体含义需要参考数据描述文档")
    
    print(f"\n 结论: 这三个特征共同解释了房价变异的很大部分，在房价预测模型中应该作为重要特征考虑。")
    
else:
    print(f" 无法进行相关性分析，目标变量 {target_col} 可能不是数值类型")

# =============================================================================
# 9. 最终数据总结
# =============================================================================
print("\n" + "="*60)
print("9. 数据预处理总结")
print("="*60)

print(" 预处理流程总结:")
print(f"• 原始数据形状: {df.shape}")
print(f"• 清理后数据形状: {df_cleaned.shape}")
print(f"• 处理缺失值: {len(missing_data) if len(missing_data) > 0 else 0} 个特征有缺失值")
print(f"• 删除异常值: {original_shape[0] - df_cleaned.shape[0]} 行")

if target_col in df_cleaned.columns:
    print(f"• 目标变量: {target_col}")
    print(f"• 价格范围: ${df_cleaned[target_col].min():,.0f} - ${df_cleaned[target_col].max():,.0f}")
    print(f"• 平均价格: ${df_cleaned[target_col].mean():,.0f}")

print("\n 数据预处理完成！数据已准备好用于进一步分析或建模。")

# 显示最终数据的前几行
print("\n 预处理后的数据前3行:")
display(df_cleaned.head(3))

# 保存清理后的数据（可选）
try:
    df_cleaned.to_csv('house_data_cleaned.csv', index=False)
    print(" 清理后的数据已保存为 'house_data_cleaned.csv'")
except Exception as e:
    print(f" 保存文件时出错: {e}")

print("\n 实验二：数据预处理完成！")