# 实验六：ESI 数据深度学习预测与聚类分析

---

## 一、实验目的

1. 利用 **深度学习方法** 构建学科排名预测模型，自动学习 ESI 数据中的潜在规律。  
2. 使用 **均方误差 (MSE)**、**平均绝对百分比误差 (MAPE)** 等指标评估预测模型效果。  
3. 通过 **聚类分析 (K-Means)** 识别出与华东师范大学 (East China Normal University) 类似的高校，并探讨相似性原因。  
4. 综合使用机器学习与可视化技术，提升对科研数据的建模与解释能力。

---

## 二、数据说明

实验数据来源于各学科的 ESI（Essential Science Indicators）指标文件，  
存放于目录：

```
D:\DataScienceLab\lab5\data
```

主要字段示例：

| 列名 | 含义 |
|------|------|
| Institutions | 机构名称 |
| Countries/Regions | 所在国家/地区 |
| Web of Science Documents | 发表论文数 |
| Cites | 总被引次数 |
| Cites/Paper | 平均每篇论文被引次数 |
| Rank | 学科排名 |

---

## 三、模型与方法

### 1. 学科排名预测（任务 11）

- **方法**：使用 `TensorFlow/Keras` 构建全连接神经网络（3 层隐藏层），输入为标准化后的特征。
- **训练与测试集划分**：
  - 每个学科的前 60% 数据用于训练
  - 后 20% 数据用于测试
- **指标评价**：
  - 均方误差 (MSE)
  - 平均绝对百分比误差 (MAPE)
- **输出文件**：`ranking_predictions.csv`

| 学科 | 样本数 | MSE | MAPE |
|------|--------|------|------|
| AGRICULTURAL SCIENCES | ... | ... | ... |
| COMPUTER SCIENCE | ... | ... | ... |

---

### 2. 学校聚类分析（任务 12）

- **方法**：对所有机构的 ESI 指标进行标准化后，用 `KMeans` 聚类。  
- **聚类数**：k = 5（可调整）
- **降维与可视化**：采用 `PCA` 将高维特征映射到二维平面。
- **输出文件**：
  - `clustering_results.csv` — 各学校对应的聚类编号
  - `cluster_visualization.png` — 聚类结果可视化图

---

## 四、文件输出说明

| 文件名 | 说明 |
|---------|------|
| `ranking_predictions.csv` | 各学科预测模型的误差结果 |
| `clustering_results.csv` | 每所学校的聚类结果与学科对应关系 |
| `cluster_visualization.png` | 聚类二维分布图（PCA 降维后展示） |

所有结果文件自动生成在：

```
D:\DataScienceLab\lab6\
```

---

## 五、运行方法

在命令行中进入项目目录后执行：

```bash
cd D:\DataScienceLab\lab6
python -u task.py
```

程序将自动完成数据读取、模型训练、结果评估与文件输出。

---

## 六、实验总结

本实验通过深度学习与聚类分析，探索了 ESI 数据中的规律与模式。  
深度学习模型能较好地预测排名位置，说明学科指标间存在可学习的非线性关系；  
聚类结果揭示了机构间的科研结构相似性，为高校科研定位与发展策略提供了参考。

---

> ✨ *扩展建议*：  
> - 可尝试使用 `RandomForestRegressor` 或 `XGBoost` 进行模型对比；  
> - 可通过 `t-SNE` 可视化探索更细致的聚类结构。
