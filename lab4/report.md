# 华东师范大学ESI学科数据分析报告

## 报告概述
- **分析时间**：2025-09-30 11:22:18  
- **数据来源**：Clarivate ESI 学科数据  
- **分析学科数量**：14 个潜在优势学科  
- **数据文件**：D:\DataScienceLab\lab3\IndicatorsExport.xlsx  

---

## 一、总体情况分析

### 1.1 学科排名概况
在分析的 14 个潜在优势学科中：

- **最佳排名学科**：CHEMISTRY（全球第2名）  
- **平均排名**：10.4  
- **中位数排名**：8.5  
- **排名标准差**：6.4  

### 1.2 排名区间统计
- **前50名**：14 个学科 (100.0%)

---

## 二、数据处理与SQL分析

### 2.1 数据导入与数据库选择
为实现对ESI学科数据的高效存储与查询，本次分析选用 **MySQL 8.0** 作为关系型数据库系统。  
将 `IndicatorsExport.xlsx` 文件中的学科指标数据导入至数据库表中，主要字段包括：

| 字段名 | 含义 | 示例 |
|--------|------|------|
| id | 主键 | 1 |
| university | 学校名称 | East China Normal University |
| country | 国家/地区 | China |
| region | 所在大区 | Asia |
| field | 学科领域 | Chemistry |
| global_rank | 全球排名 | 2 |
| influence_score | 影响力评分 | 88.21 |

---

### 2.2 数据优化与Schema设计

设计优化后的数据库 Schema 如下：

```sql
CREATE TABLE universities (
    university_id INT PRIMARY KEY AUTO_INCREMENT,
    university_name VARCHAR(255),
    country VARCHAR(100),
    region VARCHAR(100)
);

CREATE TABLE subjects (
    subject_id INT PRIMARY KEY AUTO_INCREMENT,
    subject_name VARCHAR(255)
);

CREATE TABLE esi_rankings (
    ranking_id INT PRIMARY KEY AUTO_INCREMENT,
    university_id INT,
    subject_id INT,
    global_rank INT,
    influence_score DECIMAL(10,2),
    data_year YEAR,
    FOREIGN KEY (university_id) REFERENCES universities(university_id),
    FOREIGN KEY (subject_id) REFERENCES subjects(subject_id)
);
```

该结构通过 **主外键关联** 实现：
- 学校与地区信息独立存储，便于跨学科分析；
- 学科表统一管理所有ESI学科；
- 排名表 `esi_rankings` 存储年度数据、影响力及排名等核心指标；
- 可扩展性强，支持未来年份或其他大学数据导入。

---

### 2.3 查询一：华东师范大学在各个学科中的排名

```sql
SELECT s.subject_name AS 学科, e.global_rank AS 全球排名, e.influence_score AS 影响力评分
FROM esi_rankings e
JOIN universities u ON e.university_id = u.university_id
JOIN subjects s ON e.subject_id = s.subject_id
WHERE u.university_name = 'East China Normal University'
ORDER BY e.global_rank ASC;
```

**说明**：  
此查询返回华东师范大学在所有ESI学科的全球排名及影响力评分，结果按排名升序排列。

---

### 2.4 查询二：中国（大陆地区）大学在各个学科中的表现

```sql
SELECT s.subject_name AS 学科, 
       COUNT(DISTINCT u.university_name) AS 入榜高校数, 
       AVG(e.global_rank) AS 平均排名,
       AVG(e.influence_score) AS 平均影响力
FROM esi_rankings e
JOIN universities u ON e.university_id = u.university_id
JOIN subjects s ON e.subject_id = s.subject_id
WHERE u.country = 'China'
GROUP BY s.subject_name
ORDER BY AVG(e.global_rank);
```

**说明**：  
该查询统计了中国（大陆地区）在各ESI学科领域的总体表现，包括入榜高校数量、平均全球排名和平均影响力。

---

### 2.5 查询三：全球不同区域在各个学科中的表现

```sql
SELECT s.subject_name AS 学科, 
       u.region AS 区域, 
       COUNT(DISTINCT u.university_name) AS 入榜高校数,
       AVG(e.global_rank) AS 平均排名,
       AVG(e.influence_score) AS 平均影响力
FROM esi_rankings e
JOIN universities u ON e.university_id = u.university_id
JOIN subjects s ON e.subject_id = s.subject_id
GROUP BY s.subject_name, u.region
ORDER BY s.subject_name, AVG(e.global_rank);
```

**说明**：  
该语句分析全球不同区域（如亚洲、欧洲、北美等）在各学科的总体实力分布，可揭示国际科研版图的区域差异性。

---

## 三、优势学科详细分析

### 3.1 顶尖学科（排名前50）
- **CHEMISTRY** - 全球第2名（影响力评分：88.2）  
- **ENGINEERING** - 全球第3名（影响力评分：78.7）  
- **MATERIALS SCIENCE** - 全球第4名（影响力评分：85.5）  
- **BIOLOGY & BIOCHEMISTRY** - 全球第5名（影响力评分：79.3）  
- **ENVIRONMENT/ECOLOGY** - 全球第6名（影响力评分：73.0）  
- **PHYSICS** - 全球第7名（影响力评分：63.1）  
- **MOLECULAR BIOLOGY & GENETICS** - 全球第8名（影响力评分：77.9）  
- **SOCIAL SCIENCES, GENERAL** - 全球第9名（影响力评分：50.4）  
- **GEOSCIENCES** - 全球第12名（影响力评分：47.5）  
- **PSYCHIATRY/PSYCHOLOGY** - 全球第15名（影响力评分：34.2）  
- **COMPUTER SCIENCE** - 全球第16名（影响力评分：28.9）  
- **ECONOMICS & BUSINESS** - 全球第18名（影响力评分：22.8）  
- **MICROBIOLOGY** - 全球第19名（影响力评分：25.7）  
- **MATHEMATICS** - 全球第21名（影响力评分：0.0）  

---

### 3.2 所有潜在优势学科列表
（按全球排名升序排列）

| 排名 | 学科领域 | 影响力评分 |
|------|----------|------------|
| 2 | CHEMISTRY | 88.20855614973262 |
| 3 | ENGINEERING | 78.72220658598367 |
| 4 | MATERIALS SCIENCE | 85.50591049817055 |
| 5 | BIOLOGY & BIOCHEMISTRY | 79.32169997185477 |
| 6 | ENVIRONMENT/ECOLOGY | 73.017168589924 |
| 7 | PHYSICS | 63.0629045876724 |
| 8 | MOLECULAR BIOLOGY & GENETICS | 77.89473684210526 |
| 9 | SOCIAL SCIENCES, GENERAL | 50.37362792006754 |
| 12 | GEOSCIENCES | 47.50281452293836 |
| 15 | PSYCHIATRY/PSYCHOLOGY | 34.15071770334927 |
| 16 | COMPUTER SCIENCE | 28.875598086124405 |
| 18 | ECONOMICS & BUSINESS | 22.817337461300312 |
| 19 | MICROBIOLOGY | 25.65719110610752 |
| 21 | MATHEMATICS | 0.0 |

---

## 四、关键发现与建议

### 4.1 主要发现
1. **优势领域突出**：华东师范大学在 **CHEMISTRY** 领域表现最为突出，排名全球第2名。  
2. **学科覆盖面广**：在14个ESI学科领域均有布局，体现了学校的综合性特点。  
3. **发展不均衡**：学科间排名差异较大，从第2名到第21名。  

---

### 4.2 战略建议

#### 短期策略（1-2年）
1. **重点突破**：集中资源支持排名前50的学科冲击更高排名。  
2. **特色强化**：巩固在 **CHEMISTRY** 等优势领域的领先地位。  
3. **短板提升**：对排名靠后的学科进行诊断和改进。  

#### 中长期策略（3-5年）
1. **学科交叉**：促进优势学科与其他学科的交叉融合。  
2. **人才引进**：在关键领域引进高水平学术带头人。  
3. **国际合作**：加强与世界一流大学的科研合作。  

---

## 五、结论

基于ESI数据分析，华东师范大学在多个学科领域具备良好的发展基础和竞争潜力。  
通过结合数据库结构化管理与SQL分析，可以更系统地挖掘学校的科研优势与全球定位。  
未来，通过实施差异化的学科发展战略，有望在世界大学排名与学科评估中实现新的突破。

---

**报告生成说明**：  
- 本报告结合 ESI 数据与数据库分析完成。  
- 所有 SQL 查询可直接运行于 MySQL、PostgreSQL 等主流数据库系统。  
- 数据结果仅供研究与教学参考，以官方发布为准。  
 