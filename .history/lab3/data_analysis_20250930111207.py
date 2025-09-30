import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

class ESIAnalyzer:
    def __init__(self, file_path=None):
        # 设置文件路径
        if file_path is None:
            # 使用你提供的路径
            self.file_path = r"D:\DataScienceLab\lab3\IndicatorsExport.xlsx"
        else:
            self.file_path = file_path
        self.data = None
        self.ecnu_analysis = None
        
    def check_file_exists(self):
        """检查文件是否存在"""
        if os.path.exists(self.file_path):
            print(f"找到文件: {self.file_path}")
            return True
        else:
            print(f"文件不存在: {self.file_path}")
            # 尝试在目录中查找其他Excel文件
            directory = os.path.dirname(self.file_path)
            if os.path.exists(directory):
                excel_files = glob.glob(os.path.join(directory, "*.xlsx")) + glob.glob(os.path.join(directory, "*.xls"))
                if excel_files:
                    print(f"在目录中找到以下Excel文件:")
                    for file in excel_files:
                        print(f"  - {file}")
                    self.file_path = excel_files[0]
                    print(f"使用文件: {self.file_path}")
                    return True
            return False
    
    def load_data(self):
        """加载并清洗数据"""
        if not self.check_file_exists():
            print("无法找到数据文件，请检查文件路径")
            return None
        
        try:
            print(f"正在读取文件: {self.file_path}")
            
            # 尝试不同的读取方式
            for skip_rows in [5, 4, 3, 2, 1, 0]:
                try:
                    self.data = pd.read_excel(self.file_path, skiprows=skip_rows)
                    print(f"跳过{skip_rows}行读取成功")
                    
                    # 检查数据是否包含必要的列
                    if len(self.data.columns) >= 2 and not self.data.empty:
                        break
                except Exception as e:
                    print(f"跳过{skip_rows}行读取失败: {e}")
                    continue
            
            if self.data is None or self.data.empty:
                print("无法读取有效数据")
                return None
            
            # 显示数据基本信息
            print(f"\n原始数据形状: {self.data.shape}")
            print("前5行数据:")
            print(self.data.head())
            print("\n列名:")
            print(self.data.columns.tolist())
            
            # 数据清洗
            self.clean_data()
            
            print(f"\n数据清洗完成！")
            print(f"有效数据行数: {len(self.data)}")
            return self.data
            
        except Exception as e:
            print(f"读取文件时出错: {e}")
            return None
    
    def clean_data(self):
        """清洗数据"""
        # 删除完全空白的行
        self.data = self.data.dropna(how='all')
        
        # 查找包含学科数据的行（第一列是数字排名，第二列是学科名称）
        valid_rows = []
        
        for idx, row in self.data.iterrows():
            if len(row) >= 2:
                col0 = str(row.iloc[0]).strip()
                col1 = str(row.iloc[1]).strip()
                
                # 检查是否包含有效的学科数据
                if (col0.replace('.', '').isdigit() and 
                    len(col1) > 3 and 
                    not any(keyword in col1.upper() for keyword in ['COPYRIGHT', 'INDICATORS', 'FILTER', 'RESULTS'])):
                    valid_rows.append(idx)
        
        if valid_rows:
            self.data = self.data.iloc[valid_rows].copy()
            
            # 重命名列
            if len(self.data.columns) >= 6:
                self.data.columns = ['Rank', 'Research_Fields', 'Web_of_Science_Documents', 
                                   'Cites', 'Cites_per_Paper', 'Top_Papers']
            elif len(self.data.columns) >= 2:
                # 如果列数不够，只保留排名和学科名称
                self.data = self.data.iloc[:, :2].copy()
                self.data.columns = ['Rank', 'Research_Fields']
            
            # 转换数据类型
            self.data['Rank'] = pd.to_numeric(self.data['Rank'], errors='coerce')
            self.data = self.data.dropna(subset=['Rank'])
            self.data['Rank'] = self.data['Rank'].astype(int)
            
            # 清理学科名称
            self.data['Research_Fields'] = self.data['Research_Fields'].astype(str).str.strip()
    
    def analyze_ecnu_potential(self):
        """分析华东师范大学的潜在优势学科"""
        if self.data is None:
            print("请先加载数据")
            return None
        
        print("\n正在分析华东师范大学潜在优势学科...")
        
        # 华东师范大学的传统优势学科关键词
        ecnu_strength_keywords = [
            'COMPUTER SCIENCE', 'MATHEMATICS', 'PHYSICS', 'CHEMISTRY',
            'ENGINEERING', 'MATERIALS SCIENCE', 'BIOLOGY', 'ENVIRONMENT',
            'ECOLOGY', 'SOCIAL SCIENCE', 'PSYCHOLOGY', 'EDUCATION',
            'GEOSCIENCE', 'ECONOMICS', 'BUSINESS'
        ]
        
        # 在当前数据中匹配相关学科
        potential_fields = []
        for _, row in self.data.iterrows():
            field_name = str(row['Research_Fields']).upper()
            for keyword in ecnu_strength_keywords:
                if keyword in field_name:
                    potential_fields.append(row)
                    break
        
        if potential_fields:
            self.ecnu_analysis = pd.DataFrame(potential_fields)
            
            # 计算学科影响力评分
            self.calculate_impact_scores()
            
            # 按排名排序
            self.ecnu_analysis = self.ecnu_analysis.sort_values('Rank')
            
            print(f"找到 {len(self.ecnu_analysis)} 个华东师范大学潜在优势学科")
            
        return self.ecnu_analysis
    
    def calculate_impact_scores(self):
        """计算学科影响力评分"""
        # 基于排名计算影响力评分（排名越靠前，分数越高）
        max_rank = self.ecnu_analysis['Rank'].max()
        min_rank = self.ecnu_analysis['Rank'].min()
        
        # 排名归一化（排名越靠前，值越大）
        if max_rank > min_rank:
            rank_norm = 1 - (self.ecnu_analysis['Rank'] - min_rank) / (max_rank - min_rank)
        else:
            rank_norm = 1  # 如果所有排名相同
        
        # 如果有引用数据，加入引用指标
        if 'Cites_per_Paper' in self.ecnu_analysis.columns:
            max_cites = self.ecnu_analysis['Cites_per_Paper'].max()
            min_cites = self.ecnu_analysis['Cites_per_Paper'].min()
            if max_cites > min_cites:
                cites_norm = (self.ecnu_analysis['Cites_per_Paper'] - min_cites) / (max_cites - min_cites)
            else:
                cites_norm = 0.5
            # 综合评分
            self.ecnu_analysis['Impact_Score'] = (rank_norm * 0.7 + cites_norm * 0.3) * 100
        else:
            # 仅基于排名
            self.ecnu_analysis['Impact_Score'] = rank_norm * 100
    
    def create_visualizations(self):
        """创建可视化图表"""
        if self.ecnu_analysis is None or len(self.ecnu_analysis) == 0:
            print("没有足够的数据创建图表")
            return
        
        # 创建图表目录
        chart_dir = r"D:\DataScienceLab\lab3\charts"
        os.makedirs(chart_dir, exist_ok=True)
        
        try:
            # 设置图表风格
            plt.style.use('default')
            
            # 1. 学科排名图
            plt.figure(figsize=(14, 10))
            fields = self.ecnu_analysis['Research_Fields'].tolist()
            ranks = self.ecnu_analysis['Rank'].tolist()
            
            # 截断过长的学科名称
            short_fields = [f[:20] + '...' if len(f) > 20 else f for f in fields]
            
            colors = plt.cm.plasma(np.linspace(0, 0.8, len(fields)))
            bars = plt.barh(range(len(fields)), ranks, color=colors, alpha=0.7)
            
            plt.yticks(range(len(fields)), short_fields, fontsize=10)
            plt.xlabel('全球排名（数值越小越好）', fontsize=12)
            plt.title('华东师范大学潜在优势学科全球排名', fontsize=14, fontweight='bold')
            plt.gca().invert_yaxis()
            
            # 在条形上添加排名数字
            for i, v in enumerate(ranks):
                plt.text(v + max(ranks)*0.01, i, f'#{v}', va='center', fontsize=9, 
                        fontweight='bold', color='darkblue')
            
            plt.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(chart_dir, 'ecnu_ranks.png'), dpi=300, bbox_inches='tight')
            plt.show()
            
            # 2. 影响力评分图
            if 'Impact_Score' in self.ecnu_analysis.columns:
                plt.figure(figsize=(14, 10))
                impact_scores = self.ecnu_analysis['Impact_Score'].tolist()
                
                colors = plt.cm.viridis(np.linspace(0, 1, len(fields)))
                bars = plt.barh(range(len(fields)), impact_scores, color=colors)
                
                plt.yticks(range(len(fields)), short_fields, fontsize=10)
                plt.xlabel('影响力评分（0-100）', fontsize=12)
                plt.title('华东师范大学潜在优势学科影响力评分', fontsize=14, fontweight='bold')
                plt.gca().invert_yaxis()
                
                for i, v in enumerate(impact_scores):
                    plt.text(v + 1, i, f'{v:.1f}', va='center', fontsize=9, 
                            fontweight='bold', color='darkred')
                
                plt.grid(axis='x', alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(chart_dir, 'ecnu_impact_scores.png'), dpi=300, bbox_inches='tight')
                plt.show()
            
            # 3. 排名分布饼图
            plt.figure(figsize=(10, 8))
            rank_ranges = {
                '前50': len(self.ecnu_analysis[self.ecnu_analysis['Rank'] <= 50]),
                '51-100': len(self.ecnu_analysis[(self.ecnu_analysis['Rank'] > 50) & (self.ecnu_analysis['Rank'] <= 100)]),
                '101-200': len(self.ecnu_analysis[(self.ecnu_analysis['Rank'] > 100) & (self.ecnu_analysis['Rank'] <= 200)]),
                '200+': len(self.ecnu_analysis[self.ecnu_analysis['Rank'] > 200])
            }
            
            labels = [k for k, v in rank_ranges.items() if v > 0]
            sizes = [v for k, v in rank_ranges.items() if v > 0]
            colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
            
            plt.pie(sizes, labels=labels, colors=colors[:len(labels)], autopct='%1.1f%%', startangle=90)
            plt.axis('equal')
            plt.title('华东师范大学学科排名分布', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(chart_dir, 'ecnu_rank_distribution.png'), dpi=300, bbox_inches='tight')
            plt.show()
            
            print(f"可视化图表已生成在: {chart_dir}")
            
        except Exception as e:
            print(f"创建图表时出错: {e}")
    
    def generate_analysis_report(self):
        """生成详细分析报告"""
        if self.ecnu_analysis is None or len(self.ecnu_analysis) == 0:
            print("没有足够的数据生成报告")
            return None
        
        report = f"""
# 华东师范大学ESI学科数据分析报告

## 报告概述
- **分析时间**：{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
- **数据来源**：Clarivate ESI 学科数据
- **分析学科数量**：{len(self.ecnu_analysis)} 个潜在优势学科
- **数据文件**：{self.file_path}

## 一、总体情况分析

### 1.1 学科排名概况
在分析的 {len(self.ecnu_analysis)} 个潜在优势学科中：

- **最佳排名学科**：{self.ecnu_analysis.iloc[0]['Research_Fields']}（全球第{self.ecnu_analysis.iloc[0]['Rank']}名）
- **平均排名**：{self.ecnu_analysis['Rank'].mean():.1f}
- **中位数排名**：{self.ecnu_analysis['Rank'].median():.1f}
- **排名标准差**：{self.ecnu_analysis['Rank'].std():.1f}

### 1.2 排名区间统计
"""
        
        # 统计排名区间
        rank_ranges = [
            (1, 50, "前50名"),
            (51, 100, "51-100名"),
            (101, 200, "101-200名"),
            (201, 500, "201-500名"),
            (501, float('inf'), "500名以上")
        ]
        
        for start, end, label in rank_ranges:
            count = len(self.ecnu_analysis[
                (self.ecnu_analysis['Rank'] >= start) & 
                (self.ecnu_analysis['Rank'] <= end)
            ])
            if count > 0:
                percentage = (count / len(self.ecnu_analysis)) * 100
                report += f"- **{label}**：{count} 个学科 ({percentage:.1f}%)\\n"
        
        report += f"""
## 二、优势学科详细分析

### 2.1 顶尖学科（排名前50）
"""
        
        top_50 = self.ecnu_analysis[self.ecnu_analysis['Rank'] <= 50]
        if not top_50.empty:
            for _, row in top_50.iterrows():
                report += f"- **{row['Research_Fields']}** - 全球第{row['Rank']}名"
                if 'Impact_Score' in row and pd.notna(row['Impact_Score']):
                    report += f"（影响力评分：{row['Impact_Score']:.1f}）"
                report += "\\n"
        else:
            report += "暂无排名前50的学科\\n"
        
        report += """
### 2.2 所有潜在优势学科列表
（按全球排名升序排列）

| 排名 | 学科领域 | 影响力评分 |
|------|----------|------------|
"""
        
        for _, row in self.ecnu_analysis.iterrows():
            score = row['Impact_Score'] if 'Impact_Score' in row and pd.notna(row['Impact_Score']) else 'N/A'
            report += f"| {row['Rank']} | {row['Research_Fields']} | {score} |\\n"
        
        report += f"""
## 三、关键发现与建议

### 3.1 主要发现

1. **优势领域突出**：华东师范大学在 **{self.ecnu_analysis.iloc[0]['Research_Fields']}** 领域表现最为突出，排名全球第{self.ecnu_analysis.iloc[0]['Rank']}名

2. **学科覆盖面广**：在{len(self.ecnu_analysis)}个ESI学科领域均有布局，体现了学校的综合性特点

3. **发展不均衡**：学科间排名差异较大，从第{self.ecnu_analysis['Rank'].min()}名到第{self.ecnu_analysis['Rank'].max()}名

### 3.2 战略建议

#### 短期策略（1-2年）
1. **重点突破**：集中资源支持排名前50的学科冲击更高排名
2. **特色强化**：巩固在 **{self.ecnu_analysis.iloc[0]['Research_Fields']}** 等优势领域的领先地位
3. **短板提升**：对排名靠后的学科进行诊断和改进

#### 中长期策略（3-5年）
1. **学科交叉**：促进优势学科与其他学科的交叉融合
2. **人才引进**：在关键领域引进高水平学术带头人
3. **国际合作**：加强与世界一流大学的科研合作

## 四、结论

基于ESI数据分析，华东师范大学在多个学科领域具备良好的发展基础和竞争潜力。通过实施差异化的学科发展战略，有望在未来的学科评估中取得更好成绩，为建设世界一流大学奠定坚实基础。

---

**报告生成说明**：
- 本分析基于华东师范大学的传统优势学科与ESI学科数据进行匹配
- 影响力评分综合考虑了全球排名和科研影响力指标
- 具体数据请以官方发布的学科评估结果为准

**附件**：
1. 华东师范大学ESI详细分析数据.xlsx
2. 学科排名可视化图表
3. 排名分布分析图
"""
        
        # 保存报告
        try:
            report_path = r"D:\DataScienceLab\lab3\华东师范大学ESI学科分析报告.md"
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report)
            
            # 保存详细数据
            data_path = r"D:\DataScienceLab\lab3\华东师范大学ESI详细分析数据.xlsx"
            self.ecnu_analysis.to_excel(data_path, index=False)
            
            print(f"分析报告已生成: {report_path}")
            print(f"详细数据已保存: {data_path}")
            
            return report
            
        except Exception as e:
            print(f"保存报告时出错: {e}")
            return None

def main():
    print("=" * 60)
    print("       华东师范大学ESI学科数据分析系统")
    print("=" * 60)
    
    # 初始化分析器
    analyzer = ESIAnalyzer(r"D:\DataScienceLab\lab3\IndicatorsExport.xlsx")
    
    # 加载数据
    print("\n步骤1: 加载数据...")
    data = analyzer.load_data()
    if data is None:
        print("数据加载失败，请检查文件路径和格式")
        return
    
    print("\n步骤2: 分析华东师范大学潜在优势学科...")
    ecnu_analysis = analyzer.analyze_ecnu_potential()
    if ecnu_analysis is not None and len(ecnu_analysis) > 0:
        print(f"\n找到 {len(ecnu_analysis)} 个潜在优势学科:")
        print(ecnu_analysis[['Research_Fields', 'Rank']].to_string(index=False))
        
        print("\n步骤3: 生成可视化图表...")
        analyzer.create_visualizations()
        
        print("\n步骤4: 生成分析报告...")
        analyzer.generate_analysis_report()
        
        print("\n" + "=" * 60)
        print("分析完成！生成的文件：")
        print("1. 华东师范大学ESI学科分析报告.md")
        print("2. 华东师范大学ESI详细分析数据.xlsx")
        print("3. charts/ 目录下的可视化图表")
        print("=" * 60)
    else:
        print("未能识别出华东师范大学的潜在优势学科")

if __name__ == "__main__":
    main()