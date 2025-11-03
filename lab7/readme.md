# 实验七：ESI 学科排名的深度学习升级与相似高校聚类分析

---

## 一、实验目的

1. 在上一实验（Lab6）深度学习模型的基础上，进一步优化学科排名预测模型。  
2. 通过更合理的数据划分、特征标准化与神经网络结构调整，提升模型精度。  
3. 利用聚类算法（如 KMeans）对全球高校进行相似性分析，识别与华东师范大学在学科表现上相近的学校。  
4. 通过 MSE、MAPE 等指标量化模型性能，并可视化分析聚类结果。

---

## 二、实验环境

- **操作系统**：Windows 10 / 11  
- **开发环境**：VS Code + Python 3.10  
- **主要依赖库**：
  - pandas
  - numpy
  - scikit-learn
  - tensorflow / keras
  - matplotlib / seaborn

---

## 三、数据来源与预处理

- **数据文件**：  
  `D:\DataScienceLab\lab5\data\merged_all.csv`

- **数据说明**：
  - 来源：Web of Science（ESI 数据）  
  - 含字段：  
    - Institutions（高校名称）  
    - Countries/Regions（地区）  
    - Web of Science Documents（文献数量）  
    - Cites（被引次数）  
    - Cites/Paper（平均被引次数）  
    - Top Papers（高被引论文数）  
    - Discipline（学科类别）

- **预处理步骤**：
  1. 去除缺失值与异常值；
  2. 按学科分组，分别计算指标；
  3. 各学科样本按比例拆分：
     - 前 60%：训练集  
     - 中 20%：验证集  
     - 后 20%：测试集；
  4. 对数值特征进行标准化（StandardScaler）。

---

## 四、模型设计与优化

### 1. 模型结构

采用改进版全连接神经网络（MLP）：

```python
model = Sequential([
    Dense(128, activation='relu', input_dim=X_train.shape[1]),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1)  # 输出预测学科排名
])
```

### 2. 训练配置

- 优化器：`Adam(lr=0.001)`
- 损失函数：`MSE`
- 监控指标：`MAPE`
- 训练轮次：100
- 批大小：32
- 提前停止：若验证集损失 10 轮未改进则停止

---

## 五、实验结果

### 1. 模型评估指标（示例）

| 学科 | MSE | MAPE | 说明 |
|------|------|------|------|
| AGRICULTURAL SCIENCES | 165,881 | 0.318 | 表现良好，预测偏差约 31% |
| ENGINEERING | 201,542 | 0.276 | 模型收敛良好 |
| CLINICAL MEDICINE | 244,910 | 0.294 | 稳定性强 |

### 2. 聚类分析结果

- 使用 KMeans(n_clusters=8) 对高校进行聚类；
- 指标：Cites、Cites/Paper、Top Papers；
- 发现与 **华东师范大学** 相似的高校包括：
  - 北京师范大学  
  - 南京大学  
  - 浙江大学  
  - 复旦大学  

这些学校在自然科学与社会科学领域的论文产出及被引模式与华师大接近。

---

## 六、结果可视化

1. 各学科预测结果对比图（真实 vs 预测）；
2. 各聚类簇的高校分布散点图；
3. MAPE 分布直方图；
4. 各指标的热力图。

---

## 七、结论与展望

1. 深度学习模型较传统线性模型有更强的拟合能力；  
2. 对不同学科单独建模显著提升预测精度；  
3. 聚类分析能揭示高校间的学科结构相似性，为学科建设提供参考；  
4. 后续可引入时间序列特征（如历年排名变化）进行动态预测。

---

## 八、文件结构

```
lab7/
│
├── task.py               # 主程序（深度学习 + 聚类）
├── README.md             # 实验报告说明文件
├── results/
│   ├── model_evaluation.csv
│   ├── similar_universities.csv
│   └── visualization/
│       ├── mse_distribution.png
│       ├── clusters.png
│       └── predictions.png
```