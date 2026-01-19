---
title: PointNet++ Deep Hierarchical Feature Learning on Point Sets in a Metric Space
field: Deep_Learning
status: Imported
created_date: 2026-01-19
pdf_link: "[[PointNet++ Deep Hierarchical Feature Learning on Point Sets in a Metric Space.pdf]]"
tags:
  - paper
  - Deep_learning
---

# PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space
# PointNet++：度量空間中點集上的深度分層特徵學習

**Charles R. Qi Li Yi Hao Su Leonidas J. Guibas**
**Stanford University**
**史丹佛大學**

## Abstract
## 摘要

Few prior works study deep learning on point sets. PointNet [20] is a pioneer in this direction. However, by design PointNet does not capture local structures induced by the metric space points live in, limiting its ability to recognize fine-grained patterns and generalizability to complex scenes. In this work, we introduce a hierarchical neural network that applies PointNet recursively on a nested partitioning of the input point set. By exploiting metric space distances, our network is able to learn local features with increasing contextual scales. With further observation that point sets are usually sampled with varying densities, which results in greatly decreased performance for networks trained on uniform densities, we propose novel set learning layers to adaptively combine features from multiple scales. Experiments show that our network called PointNet++ is able to learn deep point set features efficiently and robustly. In particular, results significantly better than state-of-the-art have been obtained on challenging benchmarks of 3D point clouds.

很少有先前的工作研究點集上的深度學習。PointNet [20] 是這方向的先驅。然而，就設計而言，PointNet 無法捕捉由點所在的度量空間所引發的局部結構，這限制了其識別細粒度模式的能力以及對複雜場景的泛化能力。在這項工作中，我們引入了一個分層神經網絡，該網絡在輸入點集的嵌套分割上遞歸地應用 PointNet。通過利用度量空間距離，我們的網絡能夠學習具有增加上下文尺度的局部特徵。進一步觀察到點集通常以不同的密度進行採樣，這導致在均勻密度上訓練的網絡性能大幅下降，因此我們提出了新穎的集合學習層，以自適應地組合來自多個尺度的特徵。實驗表明，我們稱為 PointNet++ 的網絡能夠高效且穩健地學習深度點集特徵。特別是，在充滿挑戰的 3D 點雲基準測試中，獲得了明顯優於最先進技術的結果。

# 1 Introduction
# 1 介紹

We are interested in analyzing geometric point sets which are collections of points in a Euclidean space. A particularly important type of geometric point set is point cloud captured by 3D scanners, e.g., from appropriately equipped autonomous vehicles. As a set, such data has to be invariant to permutations of its members. In addition, the distance metric defines local neighborhoods that may exhibit different properties. For example, the density and other attributes of points may not be uniform across different locations — in 3D scanning the density variability can come from perspective effects, radial density variations, motion, etc.

我們有興趣分析幾何點集，即歐幾里得空間中的點集合。一種特別重要的幾何點集是由 3D 掃描儀捕獲的點雲，例如來自配備適當裝備的自動駕駛車輛。作為一個集合，此類數據必須對其成員的排列具有不變性。此外，距離度量定義了可能表現出不同屬性的局部鄰域。例如，點的密度和其他屬性在不同位置可能不均勻——在 3D 掃描中，密度變化可能來自透視效應、徑向密度變化、運動等。

Few prior works study deep learning on point sets. PointNet [20] is a pioneering effort that directly processes point sets. The basic idea of PointNet is to learn a spatial encoding of each point and then aggregate all individual point features to a global point cloud signature. By its design, PointNet does not capture local structure induced by the metric. However, exploiting local structure has proven to be important for the success of convolutional architectures. A CNN takes data defined on regular grids as the input and is able to progressively capture features at increasingly larger scales along a multi-resolution hierarchy. At lower levels neurons have smaller receptive fields whereas at higher levels they have larger receptive fields. The ability to abstract local patterns along the hierarchy allows better generalizability to unseen cases.

很少有先前的工作研究點集上的深度學習。PointNet [20] 是直接處理點集的開創性工作。PointNet 的基本思想是學習每個點的空間編碼，然後將所有單獨的點特徵聚合為一個全域點雲簽名。從設計上看，PointNet 沒有捕捉由度量引發的局部結構。然而，利用局部結構已被證明對卷積架構的成功至關重要。CNN 將定義在規則網格上的數據作為輸入，並且能夠沿著多解析度層次結構在越來越大的尺度上逐步捕捉特徵。在較低層次，神經元具有較小的感受野，而在較高層次，它們具有較大的感受野。沿層次結構抽象局部模式的能力允許對未見過的情況有更好的泛化能力。

We introduce a hierarchical neural network, named as PointNet++, to process a set of points sampled in a metric space in a hierarchical fashion. The general idea of PointNet++ is simple. We first partition the set of points into overlapping local regions by the distance metric of the underlying space. Similar to CNNs, we extract local features capturing fine geometric structures from small neighborhoods; such local features are further grouped into larger units and processed to produce higher level features. This process is repeated until we obtain the features of the whole point set.

我們引入了一個名為 PointNet++ 的分層神經網絡，以分層方式處理在度量空間中採樣的點集。PointNet++ 的總體思路很簡單。我們首先根據底層空間的距離度量將點集分割成重疊的局部區域。類似於 CNN，我們從小的鄰域中提取捕捉精細幾何結構的局部特徵；這些局部特徵被進一步分組為更大的單元並進行處理以產生更高層次的特徵。重複此過程，直到我們獲得整個點集的特徵。

The design of PointNet++ has to address two issues: how to generate the partitioning of the point set, and how to abstract sets of points or local features through a local feature learner. The two issues are correlated because the partitioning of the point set has to produce common structures across partitions, so that weights of local feature learners can be shared, as in the convolutional setting. We choose our local feature learner to be PointNet. As demonstrated in that work, PointNet is an effective architecture to process an unordered set of points for semantic feature extraction. In addition, this architecture is robust to input data corruption. As a basic building block, PointNet abstracts sets of local points or features into higher level representations. In this view, PointNet++ applies PointNet recursively on a nested partitioning of the input set.

PointNet++ 的設計必須解決兩個問題：如何生成點集的分割，以及如何通過局部特徵學習器抽象點集或局部特徵。這兩個問題是相關的，因為點集的分割必須在分區之間產生共同的結構，以便可以共享局部特徵學習器的權重，就像在卷積設置中一樣。我們選擇我們的局部特徵學習器為 PointNet。正如該工作所證明的，PointNet 是處理無序點集以進行語義特徵提取的有效架構。此外，該架構對輸入數據損壞具有魯棒性。作為基本構建塊，PointNet 將局部點集或特徵抽象為更高級別的表示。從這個角度來看，PointNet++ 在輸入集的嵌套分割上遞歸地應用 PointNet。

One issue that still remains is how to generate overlapping partitioning of a point set. Each partition is defined as a neighborhood ball in the underlying Euclidean space, whose parameters include centroid location and scale. To evenly cover the whole set, the centroids are selected among input point set by a farthest point sampling (FPS) algorithm. Compared with volumetric CNNs that scan the space with fixed strides, our local receptive fields are dependent on both the input data and the metric, and thus more efficient and effective.

仍然存在的一個問題是如何生成點集的重疊分割。每個分區被定義為底層歐幾里得空間中的一個鄰域球，其參數包括質心位置和尺度。為了均勻覆蓋整個集合，質心是通過最遠點採樣 (FPS) 演算法在輸入點集中選擇的。與以固定步長掃描空間的體積 CNN 相比，我們的局部感受野依賴於輸入數據和度量，因此更加高效和有效。

Deciding the appropriate scale of local neighborhood balls, however, is a more challenging yet intriguing problem, due to the entanglement of feature scale and non-uniformity of input point set. We assume that the input point set may have variable density at different areas, which is quite common in real data such as Structure Sensor scanning [18] (see Fig. 1). Our input point set is thus very different from CNN inputs which can be viewed as data defined on regular grids with uniform constant density. In CNNs, the counterpart to local partition scale is the size of kernels. [25] shows that using smaller kernels helps to improve the ability of CNNs. Our experiments on point set data, however, give counter evidence to this rule. Small neighborhood may consist of too few points due to sampling deficiency, which might be insufficient to allow PointNets to capture patterns robustly.

然而，決定局部鄰域球的適當尺度是一個更具挑戰性但也更有趣的問題，這是由於特徵尺度與輸入點集的不均勻性相互糾纏。我們假設輸入點集在不同區域可能具有可變密度，這在真實數據中非常常見，例如 Structure Sensor 掃描 [18]（見圖 1）。因此，我們的輸入點集與 CNN 輸入非常不同，後者可以被視為定義在具有均勻恆定密度的規則網格上的數據。在 CNN 中，與局部劃分尺度相對應的是卷積核的大小。[25] 表明使用較小的卷積核有助於提高 CNN 的能力。然而，我們在點集數據上的實驗給出了與此規則相反的證據。由於採樣不足，小鄰域可能包含太少的點，這可能不足以讓 PointNet 穩健地捕捉模式。

A significant contribution of our paper is that PointNet++ leverages neighborhoods at multiple scales to achieve both robustness and detail capture. Assisted with random input dropout during training, the network learns to adaptively weight patterns detected at different scales and combine multi-scale features according to the input data. Experiments show that our PointNet++ is able to process point sets efficiently and robustly. In particular, results that are significantly better than state-of-the-art have been obtained on challenging benchmarks of 3D point clouds.

我們論文的一個重要貢獻是 PointNet++ 利用多個尺度的鄰域來實現魯棒性和細節捕捉。在訓練期間隨機輸入 dropout 的輔助下，網絡學習自適應地加權在不同尺度檢測到的模式，並根據輸入數據組合多尺度特徵。實驗表明，我們的 PointNet++ 能夠高效且穩健地處理點集。特別是，在充滿挑戰的 3D 點雲基準測試中，獲得了明顯優於最先進技術的結果。

# 2 Problem Statement
# 2 問題陳述

Suppose that $\mathcal{X} = (M, d)$ is a discrete metric space whose metric is inherited from a Euclidean space $\mathbb{R}^n$, where $M \subseteq \mathbb{R}^n$ is the set of points and $d$ is the distance metric. In addition, the density of $M$ in the ambient Euclidean space may not be uniform everywhere. We are interested in learning set functions $f$ that take such $\mathcal{X}$ as the input (along with additional features for each point) and produce information of semantic interest regrading $\mathcal{X}$. In practice, such $f$ can be classification function that assigns a label to $\mathcal{X}$ or a segmentation function that assigns a per point label to each member of $M$.

假設 $\mathcal{X} = (M, d)$ 是一個離散度量空間，其度量繼承自歐幾里得空間 $\mathbb{R}^n$，其中 $M \subseteq \mathbb{R}^n$ 是點的集合，$d$ 是距離度量。此外，$M$ 在周圍歐幾里得空間中的密度可能並非處處均勻。我們感興趣的是學習集函數 $f$，該函數以 $\mathcal{X}$ 作為輸入（連同每個點的附加特徵），並產生關於 $\mathcal{X}$ 的語義興趣信息。實際上，這樣的 $f$ 可以是分配標籤給 $\mathcal{X}$ 的分類函數，或者是分配逐點標籤給 $M$ 的每個成員的分割函數。

# 3 Method
# 3 方法

Our work can be viewed as an extension of PointNet [20] with added hierarchical structure. We first review PointNet (Sec. 3.1) and then introduce a basic extension of PointNet with hierarchical structure (Sec. 3.2). Finally, we propose our PointNet++ that is able to robustly learn features even in non-uniformly sampled point sets (Sec. 3.3).

我們的工作可以被視為具有附加分層結構的 PointNet [20] 的擴展。我們首先回顧 PointNet（第 3.1 節），然後介紹具有分層結構的 PointNet 的基本擴展（第 3.2 節）。最後，我們提出了我們的 PointNet++，即使在非均勻採樣的點集中也能穩健地學習特徵（第 3.3 節）。

## 3.1 Review of PointNet [20]: A Universal Continuous Set Function Approximator
## 3.1 PointNet [20] 回顧：通用連續集函數逼近器

Given an unordered point set $\{x_1, x_2, ..., x_n\}$ with $x_i \in \mathbb{R}^d$, one can define a set function $f : \mathcal{X} \to \mathbb{R}$ that maps a set of points to a vector:

給定一個無序點集 $\{x_1, x_2, ..., x_n\}$，其中 $x_i \in \mathbb{R}^d$，我們可以定義一個集函數 $f : \mathcal{X} \to \mathbb{R}$ 將一組點映射到一個向量：

$$f(x_1, x_2, ..., x_n) = \gamma \left( \max_{i=1,...,n} \{h(x_i)\} \right) \quad (1)$$

where $\gamma$ and $h$ are usually multi-layer perceptron (MLP) networks.

其中 $\gamma$ 和 $h$ 通常是多層感知機 (MLP) 網絡。

The set function $f$ in Eq. 1 is invariant to input point permutations and can arbitrarily approximate any continuous set function [20]. Note that the response of $h$ can be interpreted as the spatial encoding of a point (see [20] for details).

公式 1 中的集函數 $f$ 對輸入點的排列具有不變性，並且可以任意逼近任何連續集函數 [20]。注意，$h$ 的響應可以解釋為點的空間編碼（詳見 [20]）。

PointNet achieved impressive performance on a few benchmarks. However, it lacks the ability to capture local context at different scales. We will introduce a hierarchical feature learning framework in the next section to resolve the limitation.

PointNet 在一些基準測試中取得了令人印象深刻的性能。然而，它缺乏在不同尺度捕捉局部上下文的能力。我們將在下一節介紹一個分層特徵學習框架來解決這個限制。

## 3.2 Hierarchical Point Set Feature Learning
## 3.2 分層點集特徵學習

While PointNet uses a single max pooling operation to aggregate the whole point set, our new architecture builds a hierarchical grouping of points and progressively abstract larger and larger local regions along the hierarchy.

雖然 PointNet 使用單個最大池化操作來聚合整個點集，但我們的新架構建立了點的分層分組，並沿著層次結構逐步抽象越來越大的局部區域。

Our hierarchical structure is composed by a number of set abstraction levels (Fig. 2). At each level, a set of points is processed and abstracted to produce a new set with fewer elements. The set abstraction level is made of three key layers: Sampling layer, Grouping layer and PointNet layer. The Sampling layer selects a set of points from input points, which defines the centroids of local regions. Grouping layer then constructs local region sets by finding “neighboring” points around the centroids. PointNet layer uses a mini-PointNet to encode local region patterns into feature vectors.

我們的分層結構由若干個集合抽象層組成（圖 2）。在每一層，一組點被處理並抽象以產生一個元素較少的新集合。集合抽象層由三個關鍵層組成：採樣層 (Sampling layer)、分組層 (Grouping layer) 和 PointNet 層 (PointNet layer)。採樣層從輸入點中選擇一組點，這些點定義了局部區域的質心。分組層隨後通過尋找質心周圍的「鄰近」點來構建局部區域集。PointNet 層使用一個微型 PointNet 將局部區域模式編碼為特徵向量。

A set abstraction level takes an $N \times (d + C)$ matrix as input that is from $N$ points with $d$-dim coordinates and $C$-dim point feature. It outputs an $N' \times (d + C')$ matrix of $N'$ subsampled points with $d$-dim coordinates and new $C'$-dim feature vectors summarizing local context. We introduce the layers of a set abstraction level in the following paragraphs.

一個集合抽象層將一個 $N \times (d + C)$ 矩陣作為輸入，該矩陣來自具有 $d$ 維坐標和 $C$ 維點特徵的 $N$ 個點。它輸出一個 $N' \times (d + C')$ 矩陣，包含 $N'$ 個下採樣點，具有 $d$ 維坐標和總結局部上下文的新的 $C'$ 維特徵向量。我們將在以下段落中介紹集合抽象層的各個層。

**Sampling layer.** Given input points $\{x_1, x_2, ..., x_n\}$, we use iterative farthest point sampling (FPS) to choose a subset of points $\{x_{i_1}, x_{i_2}, ..., x_{i_m}\}$, such that $x_{i_j}$ is the most distant point (in metric distance) from the set $\{x_{i_1}, x_{i_2}, ..., x_{i_{j-1}}\}$ with regard to the rest points. Compared with random sampling, it has better coverage of the entire point set given the same number of centroids. In contrast to CNNs that scan the vector space agnostic of data distribution, our sampling strategy generates receptive fields in a data dependent manner.

**採樣層 (Sampling layer)。** 給定輸入點 $\{x_1, x_2, ..., x_n\}$，我們使用迭代最遠點採樣 (FPS) 來選擇點的一個子集 $\{x_{i_1}, x_{i_2}, ..., x_{i_m}\}$，使得 $x_{i_j}$ 是相對於其餘點而言，距離集合 $\{x_{i_1}, x_{i_2}, ..., x_{i_{j-1}}\}$ 最遠的點（按度量距離）。與隨機採樣相比
