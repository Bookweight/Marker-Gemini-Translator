#Data_Mining #Anomaly_Detection #Review_Paper
# 簡介:

- ### 作者:

- ### 年分:

- ### 研討會/期刊:

# 摘要:

一篇關於機器學習異常偵測的review paper

# 名詞解釋:
#### Anomalies are classified into three main categories:
1. **Point Anomalies**: A single data is anomaly. (單獨的異常)
2. **Contextual Anomalies**: If in a particular context a data instance is anomalous, but not in another context. There are two attributes of contextual anomalies: contextual attributes(情境屬性) and behavioral attributes(行為屬性).(在某些情境下異常)
3. **Collective anomalies**:If a set of associated data instances is anomalous for the entire dataset, even though each individual point may not seem abnormal on its own.(拆開來正常，組在一起異常)

#### Statical anomaly detection(非ML):
 1. **proximity based**: 以距離或相似度判斷，如KNN，LOF
 2. **parametric**: 假設模型屬於統計分布，如高斯分布
 3. **non-parametric**: 不做假設分布，直接以資料推估，如KDE
 4. **semi-parametric**: 結合參數式與非參數式

#### Machine learning (ML) techniques:
1. Supervised:以標記後的異常及普通資料進行訓練。難題:訓練資料過少、難以精確標籤
2. Semi-supervised:只對普通資料進行標記並訓練，較supervised容易實現。
3. Unsupervised:不須訓練資料集，以資料中的比例來判別異常，但在異常資料偏多時會失效。

Fractal(分形模型)/multi-fractal models(多重分形模型):在地球科學仍然是重要的異常偵測工具，尤其在處理空間分佈不均或具多尺度特性的資料時。


![[nassi4-3083060-small.gif]]
#### Feature Selection/Extraction:
1. Selection Filters 特徵選擇過濾器:
	- Correlation-Based Feature Selection (CFS)：==核心概念==:好的特徵子集應該包含與類別（目標變數）高度相關的特徵，同時這些特徵彼此之間的相關性要盡可能低。==優點==:速度快、減少冗餘、提高模型性能、具可擴展性
	- Consistency-based filter (CONS)：
	- Self-Organizing Feature Map (SOFM) 自組織特徵圖：
2. Selection Wrappers 特徵選擇包裝器:
	- Sequential Selection 序列選擇:
	- Time 時間：
	- Heuristic Search 啟發式搜索：
3. Extraction 特徵提取:
	- Principle Component Analysis (PCA) 主成分分析：
	- Independent Component Analysis (ICA) 獨立成分分析：
	- Singular Value Decomposition (SVD) 奇異值分解：
	- Practical Least Square (PLS)：
	- Non-Gaussian Score (NGS) 非高斯分數：
![[nassi3-3083060-small.gif]]
#### Classification:
1. Support Vector Machine(SVM) 支持向量機:
	- One-class SVM: 
	- Two-class SVM: 
	- Core Vector Machine (CVM):
	- Kernel Method:
2. Decision Tree 決策樹:
	- Random Tree (RT):
	- Random Forest (RF) 隨機森林：
3. Bayesian Network (BN) 貝葉斯網路:
4. Neural Network 神經網路:
	- Artificial Neural Network (ANN) / 人工神經網絡：
	- Kernel Neural Network (KNN) / 核神經網絡：
	- Convolutional Neural Network (CNN) / 卷積神經網絡：
	- Recurrent Neural Network (RNN) / 循環神經網絡：
	- Restricted Boltzmann Machine (RBM) / 限制玻爾茲曼機：
	- Self Organizing Map (SOM) / 自組織映射：
5. Cluster 聚合:
6. Linear Kernel 線性核:
7. Radial Basis Function (RBF) 徑向基函數:

#### Optimization:
1. Genetic Algorithm (GA)遺傳算法:
2. Linear Embedding 線性嵌入:

#### Ensemble:
1. AdaBoost：
#### Rule system:
1. Fuzzy / 模糊邏輯：
#### Clustering:
1. K-nearest Neighbors (K-NN) / K-最近鄰：
2. K-Means / K-均值：
3. Hierarchical Clustering (HC) / 層次聚類：
4. Fuzzy Clustering / 模糊聚類：
5. Nearest Clustering / 最近鄰聚類：

#### Regression:
1. Logistic / 邏輯迴歸：
2. Linear / 線性迴歸：

# 問題:

# 解決方法:

# 實驗結果:
![[nassi5-3083060-small.gif]]
![[nassi6-3083060-small.gif]]
# 心得/問題:

# 連結: