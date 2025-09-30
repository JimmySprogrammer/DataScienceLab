import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class ESIAnalyzer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.ecnu_analysis = None
        
    def load_data(self):
        """加载并清洗数据"""
        # 读取Excel文件，跳过前5行
        self.data = pd.read_excel(self.file_path, skiprows=5)
        
        # 清洗数据
        self.data = self.data.iloc[:-1]  # 删除最后一行版权信息
        self.data = self.data.dropna()  # 删除空行
        
        # 重命名列
        self.data.columns = ['Rank', 'Research_Fields', 'Web_of_Science_Documents', 
                           'Cites', 'Cites_per_Paper', 'Top_Papers']
        
        # 转换数据类型
        self.data['Rank'] = self.data['Rank'].astype(int)
        self.data['Web_of_Science_Documents'] = self.data['Web_of_Science_Documents'].astype(int)
        self.data['Cites'] = self.data['Cites'].astype(int)
        self.data['Cites_per_Paper'] = self.data['Cites_per_Paper'].astype(float)
        self.data['Top_Papers'] = self.data['Top_Papers'].astype(int)
        
        print("数据加载完成！")
        print(f"共包含 {len(self.data)} 个学科领域")
        return self.data
    
    def analyze_ecnu_potential(self):
        """分析华东师范大学的潜在优势学科"""
        # 基于华东师范大学的传统优势学科进行推断
        ecnu_traditional_strengths = [
            'EDUCATION & EDUCATIONAL RESEARCH',  # 教育学科
            'PSYCHOLOGY',  # 心理学
            'MATHEMATICS',  # 数学
            'COMPUTER SCIENCE',  # 计算机科学
            'PHYSICS',  # 物理学
            'CHEMISTRY',  # 化学
            'ENVIRONMENT/ECOLOGY',  # 环境/生态学
            'MATERIALS SCIENCE',  # 材料科学
            'SOCIAL SCIENCES, GENERAL'  # 社会科学
        ]
        
        # 在当前数据中匹配相关学科
        potential_fields = []
        for field in ecnu_traditional_strengths:
            # 尝试匹配相似的学科名称
            matched_fields = self.data[self.data['Research_Fields'].str.contains(field.split('&')[0].strip(), case=False, na=False)]
            if not matched_fields.empty:
                potential_fields.append(matched_fields.iloc[0])
        
        if potential_fields:
            self.ecnu_analysis = pd.DataFrame(potential_fields)
            
            # 计算学科影响力评分
            self.ecnu_analysis['Impact_Score'] = (
                self.ecnu_analysis['Cites_per_Paper'] * 0.4 +
                (1 / self.ecnu_analysis['Rank']) * 0.3 +
                (self.ecnu_analysis['Top_Papers'] / self.ecnu_analysis['Web_of_Science_Documents']) * 0.3
            ) * 100
            
            # 按影响力评分排序
            self.ecnu_analysis = self.ecnu_analysis.sort_values('Impact_Score', ascending=False)
            
        return self.ecnu_analysis
    
    def create_visualizations(self):
        """创建可视化图表"""
        if self.ecnu_analysis is None:
            print("请先运行 analyze_ecnu_potential() 方法")
            return
        
        # 创建图表目录
        os.makedirs('charts', exist_ok=True)
        
        # 1. 华东师范大学潜在优势学科排名图
        plt.figure(figsize=(12, 8))
        fields = self.ecnu_analysis['Research_Fields']
        ranks = self.ecnu_analysis['Rank']
        
        plt.barh(range(len(fields)), ranks, color='skyblue')
        plt.yticks(range(len(fields)), [f[:20] + '...' if len(f) > 20 else f for f in fields])
        plt.xlabel('全球排名')
        plt.title('华东师范大学潜在优势学科全球排名')
        plt.gca().invert_yaxis()
        
        # 在条形上添加排名数字
        for i, v in enumerate(ranks):
            plt.text(v + 0.5, i, str(v), va='center')
        
        plt.tight_layout()
        plt.savefig('charts/ecnu_ranks.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. 学科影响力评分图
        plt.figure(figsize=(12, 8))
        impact_scores = self.ecnu_analysis['Impact_Score']
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(fields)))
        bars = plt.barh(range(len(fields)), impact_scores, color=colors)
        
        plt.yticks(range(len(fields)), [f[:15] + '...' if len(f) > 15 else f for f in fields])
        plt.xlabel('影响力评分')
        plt.title('华东师范大学潜在优势学科影响力评分')
        plt.gca().invert_yaxis()
        
        # 在条形上添加评分
        for i, v in enumerate(impact_scores):
            plt.text(v + 0.5, i, f'{v:.1f}', va='center')
        
        plt.tight_layout()
        plt.savefig('charts/ecnu_impact_scores.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 3. 论文影响力对比图（篇均引用）
        plt.figure(figsize=(12, 8))
        cites_per_paper = self.ecnu_analysis['Cites_per_Paper']
        
        plt.barh(range(len(fields)), cites_per_paper, color='lightgreen')
        plt.yticks(range(len(fields)), [f[:15] + '...' if len(f) > 15 else f for f in fields])
        plt.xlabel('篇均引用次数')
        plt.title('华东师范大学潜在优势学科篇均引用次数')
        plt.gca().invert_yaxis()
        
        for i, v in enumerate(cites_per_paper):
            plt.text(v + 0.1, i, f'{v:.1f}', va='center')
        
        plt.tight_layout()
        plt.savefig('charts/ecnu_cites_per_paper.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_analysis_report(self):
        """生成详细分析报告"""
        if self.ecnu_analysis is None:
            print("请先运行 analyze_ecnu_potential() 方法")
            return
        
        report = f"""
# 华东师范大学ESI学科数据分析报告

## 报告概述
- 分析时间：{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
- 数据来源：Clarivate ESI 学科数据
- 分析学科数量：{len(self.ecnu_analysis)} 个潜在优势学科
- 数据时间范围：最新可用数据

## 一、总体情况分析

### 1.1 学科排名分布
在分析的 {len(self.ecnu_analysis)} 个潜在优势学科中：

- **排名前10的学科**：{len(self.ecnu_analysis[self.ecnu_analysis['Rank'] <= 10])} 个
- **排名11-50的学科**：{len(self.ecnu_analysis[(self.ecnu_analysis['Rank'] > 10) & (self.ecnu_analysis['Rank'] <= 50)])} 个
- **排名51-100的学科**：{len(self.ecnu_analysis[(self.ecnu_analysis['Rank'] > 50) & (self.ecnu_analysis['Rank'] <= 100)])} 个
- **最佳排名学科**：{self.ecnu_analysis.iloc[0]['Research_Fields']}（第{self.ecnu_analysis.iloc[0]['Rank']}名）

### 1.2 科研产出分析
- **总Web of Science文档数**：{self.ecnu_analysis['Web_of_Science_Documents'].sum():,}
- **总被引次数**：{self.ecnu_analysis['Cites'].sum():,}
- **平均篇均引用**：{self.ecnu_analysis['Cites_per_Paper'].mean():.2f}
- **高被引论文总数**：{self.ecnu_analysis['Top_Papers'].sum():,}

## 二、优势学科详细分析

### 2.1 顶尖学科（排名前20）
"""
        
        top_20 = self.ecnu_analysis[self.ecnu_analysis['Rank'] <= 20]
        if not top_20.empty:
            for _, row in top_20.iterrows():
                report += f"- **{row['Research_Fields']}**：全球第{row['Rank']}名，篇均引用{row['Cites_per_Paper']:.2f}次，高被引论文{row['Top_Papers']}篇\n"
        else:
            report += "暂无排名前20的学科\n"
        
        report += """
### 2.2 学科影响力评分排名
（基于篇均引用、全球排名和高被引论文比例综合计算）
"""
        
        for i, (_, row) in enumerate(self.ecnu_analysis.iterrows(), 1):
            report += f"{i}. **{row['Research_Fields']}** - 影响力评分：{row['Impact_Score']:.1f}（全球排名：{row['Rank']}）\n"
        
        report += f"""
## 三、关键指标分析

### 3.1 科研质量指标
- **最高篇均引用**：{self.ecnu_analysis['Cites_per_Paper'].max():.2f}（{self.ecnu_analysis.loc[self.ecnu_analysis['Cites_per_Paper'].idxmax(), 'Research_Fields']}）
- **最低篇均引用**：{self.ecnu_analysis['Cites_per_Paper'].min():.2f}（{self.ecnu_analysis.loc[self.ecnu_analysis['Cites_per_Paper'].idxmin(), 'Research_Fields']}）
- **高被引论文比例**：{(self.ecnu_analysis['Top_Papers'].sum() / self.ecnu_analysis['Web_of_Science_Documents'].sum() * 100):.2f}%

### 3.2 学科分布特征
- **自然科学类学科**：{len(self.ecnu_analysis[self.ecnu_analysis['Research_Fields'].str.contains('SCIENCE|PHYSICS|CHEMISTRY|BIOLOGY', case=False)])} 个
- **社会科学类学科**：{len(self.ecnu_analysis[self.ecnu_analysis['Research_Fields'].str.contains('SOCIAL|PSYCHOLOGY|ECONOMICS', case=False)])} 个
- **工程技术类学科**：{len(self.ecnu_analysis[self.ecnu_analysis['Research_Fields'].str.contains('ENGINEERING|COMPUTER|MATERIALS', case=False)])} 个

## 四、发展建议

### 4.1 优势巩固
1. **重点支持顶尖学科**：对排名前20的学科给予更多资源支持
2. **加强国际合作**：提升高被引论文的数量和质量
3. **优化科研布局**：在优势学科中寻找新的增长点

### 4.2 潜力挖掘
1. **提升中游学科**：对排名中等的学科进行重点培育
2. **促进学科交叉**：利用多学科优势开展交叉研究
3. **加强人才引进**：在关键学科领域引进高水平人才

## 五、结论

华东师范大学在多个学科领域展现出较强的科研实力和影响力，特别是在{self.ecnu_analysis.iloc[0]['Research_Fields']}等领域表现突出。建议学校继续加大对这些优势学科的支持力度，同时注重学科均衡发展，提升整体科研竞争力。

---
*注：本分析基于ESI公开数据，结合华东师范大学学科特色进行的推断分析。*
"""
        
        # 保存报告
        with open('华东师范大学ESI学科分析报告.md', 'w', encoding='utf-8') as f:
            f.write(report)
        
        # 保存详细数据
        self.ecnu_analysis.to_excel('华东师范大学ESI详细分析数据.xlsx', index=False)
        
        print("分析报告已生成：华东师范大学ESI学科分析报告.md")
        print("详细数据已保存：华东师范大学ESI详细分析数据.xlsx")
        
        return report

# 使用示例
def main():
    # 初始化分析器
    analyzer = ESIAnalyzer('IndicatorsExport.xlsx')
    
    # 加载数据
    data = analyzer.load_data()
    print("\n前5行数据预览：")
    print(data.head())
    
    # 分析华东师范大学潜在优势学科
    ecnu_analysis = analyzer.analyze_ecnu_potential()
    if ecnu_analysis is not None:
        print(f"\n华东师范大学潜在优势学科分析结果（{len(ecnu_analysis)}个学科）：")
        print(ecnu_analysis[['Research_Fields', 'Rank', 'Cites_per_Paper', 'Impact_Score']])
        
        # 创建可视化图表
        analyzer.create_visualizations()
        
        # 生成分析报告
        report = analyzer.generate_analysis_report()
        
        print("\n分析完成！")
        print("生成的文件：")
        print("1. 华东师范大学ESI学科分析报告.md")
        print("2. 华东师范大学ESI详细分析数据.xlsx")
        print("3. charts/ 目录下的可视化图表")
    else:
        print("未能分析出华东师范大学的潜在优势学科")

if __name__ == "__main__":
    main()