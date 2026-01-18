---
title: PointNet Deep Learning on Point Sets for 3D Classification and Segmentation
field: Deep_Learning
status: Imported
created_date: 2026-01-18
pdf_link: "[[PointNet Deep Learning on Point Sets for 3D Classification and Segmentation.pdf]]"
tags:
  - paper
  - Deep_learning
---

# PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation
# PointNet：用於 3D 分類和分割的點集深度學習

**Charles R. Qi* Hao Su* Kaichun Mo Leonidas J. Guibas**
**Stanford University**
**史丹佛大學**

## Abstract
## 摘要

Point cloud is an important type of geometric data structure. Due to its irregular format, most researchers transform such data to regular 3D voxel grids or collections of images.
點雲是一種重要的幾何資料結構類型。由於其格式不規則，大多數研究人員將此類數據轉換為規則的 3D 體素網格或圖像集合。

This, however, renders data unnecessarily voluminous and causes issues. In this paper, we design a novel type of neural network that directly consumes point clouds, which well respects the permutation invariance of points in the input.
然而，這使得數據變得不必要地龐大並引發問題。在本文中，我們設計了一種新型的神經網路，直接處理點雲，並很好地尊重輸入中點的排列不變性。

Our network, named PointNet, provides a unified architecture for applications ranging from object classification, part segmentation, to scene semantic parsing. Though simple, PointNet is highly efficient and effective.
我們的網路名為 PointNet，為從物體分類、部件分割到場景語義解析的應用提供了一個統一的架構。雖然簡單，但 PointNet 非常高效且有效。

Empirically, it shows strong performance on par or even better than state of the art. Theoretically, we provide analysis towards understanding of what the network has learnt and why the network is robust with respect to input perturbation and corruption.
在實證上，它表現出強大的性能，與最先進的技術相當甚至更好。在理論上，我們提供了分析，以理解網路學到了什麼，以及為什麼網路對輸入的擾動和損壞具有魯棒性。

## 1. Introduction
## 1. 介紹

In this paper we explore deep learning architectures capable of reasoning about 3D geometric data such as point clouds or meshes. Typical convolutional architectures require highly regular input data formats, like those of image grids or 3D voxels, in order to perform weight sharing and other kernel optimizations.
在本文中，我們探討了能夠對 3D 幾何數據（如點雲或網格）進行推理的深度學習架構。典型的卷積架構需要高度規則的輸入數據格式，例如圖像網格或 3D 體素，以便執行權重共享和其他核心優化。

Since point clouds or meshes are not in a regular format, most researchers typically transform such data to regular 3D voxel grids or collections of images (e.g, views) before feeding them to a deep net architecture.
由於點雲或網格不是規則格式，大多數研究人員通常在將此類數據輸入深度網路架構之前，將其轉換為規則的 3D 體素網格或圖像集合（例如，視圖）。

This data representation transformation, however, renders the resulting data unnecessarily voluminous — while also introducing quantization artifacts that can obscure natural invariances of the data.
然而，這種數據表示轉換使得結果數據變得不必要地龐大——同時也引入了可能模糊數據自然不變性的量化偽影。

For this reason we focus on a different input representation for 3D geometry using simply point clouds – and name our resulting deep nets PointNets. Point clouds are simple and unified structures that avoid the combinatorial irregularities and complexities of meshes, and thus are easier to learn from.
因此，我們專注於使用簡單的點雲作為 3D 幾何的不同輸入表示——並將我們得到的深度網路命名為 PointNet。點雲是簡單且統一的結構，避免了網格的組合不規則性和複雜性，因此更容易從中學習。

The PointNet, however, still has to respect the fact that a point cloud is just a set of points and therefore invariant to permutations of its members, necessitating certain symmetrizations in the net computation. Further invariances to rigid motions also need to be considered.
然而，PointNet 仍然必須尊重這樣一個事實：點雲只是一組點，因此對其成員的排列具有不變性，這需要在網路計算中進行某些對稱化。還需要考慮對剛性運動的不變性。

Our PointNet is a unified architecture that directly takes point clouds as input and outputs either class labels for the entire input or per point segment/part labels for each point of the input. The basic architecture of our network is surprisingly simple as in the initial stages each point is processed identically and independently.
我們的 PointNet 是一個統一的架構，直接將點雲作為輸入，並輸出整個輸入的類別標籤或輸入中每個點的區段/部件標籤。我們網路的基本架構出奇地簡單，因為在初始階段，每個點都是被相同且獨立地處理的。

In the basic setting each point is represented by just its three coordinates $(x, y, z)$. Additional dimensions may be added by computing normals and other local or global features.
在基本設置中，每個點僅由其三個座標 $(x, y, z)$ 表示。可以通過計算法線和其他局部或全局特徵來添加額外的維度。

Key to our approach is the use of a single symmetric function, max pooling. Effectively the network learns a set of optimization functions/criteria that select interesting or informative points of the point cloud and encode the reason for their selection.
我們方法的關鍵是使用單一對稱函數：最大池化（max pooling）。實際上，網路學習了一組優化函數/準則，這些函數/準則選擇點雲中有趣或具資訊量的點，並對其選擇原因進行編碼。

The final fully connected layers of the network aggregate these learnt optimal values into the global descriptor for the entire shape as mentioned above (shape classification) or are used to predict per point labels (shape segmentation).
網路最後的全連接層將這些學習到的最優值聚合成上述整個形狀的全局描述符（形狀分類），或者用於預測每個點的標籤（形狀分割）。

Our input format is easy to apply rigid or affine transformations to, as each point transforms independently. Thus we can add a data-dependent spatial transformer network that attempts to canonicalize the data before the PointNet processes them, so as to further improve the results.
我們的輸入格式很容易應用剛性或仿射變換，因為每個點都是獨立變換的。因此，我們可以添加一個數據依賴的空間變換網路，試圖在 PointNet 處理數據之前將其標準化，從而進一步改善結果。

*Figure 1 Caption:*
**Figure 1. Applications of PointNet.** We propose a novel deep net architecture that consumes raw point cloud (set of points) without voxelization or rendering. It is a unified architecture that learns both global and local point features, providing a simple, efficient and effective approach for a number of 3D recognition tasks.
**圖 1. PointNet 的應用。** 我們提出了一種新穎的深度網路架構，它處理原始點雲（點集），無需體素化或渲染。這是一個統一的架構，可以學習全局和局部點特徵，為許多 3D 識別任務提供了一種簡單、高效且有效的方法。

We provide both a theoretical analysis and an experimental evaluation of our approach. We show that our network can approximate any set function that is continuous.
我們提供了我們方法的理論分析和實驗評估。我們證明了我們的網路可以近似任何連續的集合函數。

More interestingly, it turns out that our network learns to summarize an input point cloud by a sparse set of key points, which roughly corresponds to the skeleton of objects according to visualization.
更有趣的是，事實證明我們的網路學會了通過一組稀疏的關鍵點來總結輸入點雲，根據可視化結果，這些關鍵點大致對應於物體的骨架。

The theoretical analysis provides an understanding why our PointNet is highly robust to small perturbation of input points as well as to corruption through point insertion (outliers) or deletion (missing data).
理論分析提供了一種理解，解釋了為什麼我們的 PointNet 對輸入點的小擾動以及通過點插入（異常值）或刪除（缺失數據）造成的損壞具有高度的魯棒性。

On a number of benchmark datasets ranging from shape classification, part segmentation to scene segmentation, we experimentally compare our PointNet with state-of-the-art approaches based upon multi-view and volumetric representations.
在從形狀分類、部件分割到場景分割的許多基準數據集上，我們通過實驗將我們的 PointNet 與基於多視圖和體積表示的最先進方法進行了比較。

Under a unified architecture, not only is our PointNet much faster in speed, but it exhibits strong performance on par or even better than state of the art.
在統一的架構下，我們的 PointNet 不僅速度快得多，而且表現出與最先進技術相當甚至更好的強大性能。

The key contributions of our work are as follows:
我們工作的主要貢獻如下：

*   We design a novel deep net architecture suitable for consuming unordered point sets in 3D;
    我們設計了一種新穎的深度網路架構，適用於處理 3D 中的無序點集；
*   We show how such a net can be trained to perform 3D shape classification, shape part segmentation and scene semantic parsing tasks;
    我們展示了如何訓練這樣的網路來執行 3D 形狀分類、形狀部件分割和場景語義解析任務；
*   We provide thorough empirical and theoretical analysis on the stability and efficiency of our method;
    我們對我們方法的穩定性和效率提供了詳盡的實證和理論分析；
*   We illustrate the 3D features computed by the selected neurons in the net and develop intuitive explanations for its performance.
    我們展示了網路中選定神經元計算的 3D 特徵，並為其性能開發了直觀的解釋。

The problem of processing unordered sets by neural nets is a very general and fundamental problem – we expect that our ideas can be transferred to other domains as well.
神經網路處理無序集合的問題是一個非常通用且基礎的問題——我們期望我們的想法也可以轉移到其他領域。

## 2. Related Work
## 2. 相關工作

**Point Cloud Features** Most existing features for point cloud are handcrafted towards specific tasks. Point features often encode certain statistical properties of points and are designed to be invariant to certain transformations, which are typically classified as intrinsic [2, 24, 3] or extrinsic [20, 19, 14, 10, 5].
**點雲特徵** 大多數現有的點雲特徵都是針對特定任務手工製作的。點特徵通常編碼點的某些統計屬性，並被設計為對某些變換具有不變性，這些特徵通常被分類為內在特徵 [2, 24, 3] 或外在特徵 [20, 19, 14, 10, 5]。

They can also be categorized as local features and global features. For a specific task, it is not trivial to find the optimal feature combination.
它們也可以被分類為局部特徵和全局特徵。對於特定任務，找到最佳特徵組合併非易事。

**Deep Learning on 3D Data** 3D data has multiple popular representations, leading to various approaches for learning.
**3D 數據上的深度學習** 3D 數據有多種流行的表示形式，導致了各種學習方法。

*Volumetric CNNs:* [28, 17, 18] are the pioneers applying 3D convolutional neural networks on voxelized shapes. However, volumetric representation is constrained by its resolution due to data sparsity and computation cost of 3D convolution.
*體積 CNN：* [28, 17, 18] 是將 3D 卷積神經網路應用於體素化形狀的先驅。然而，由於數據稀疏性和 3D 卷積的計算成本，體積表示受到其解析度的限制。

FPNN [13] and Vote3D [26] proposed special methods to deal with the sparsity problem; however, their operations are still on sparse volumes, it’s challenging for them to process very large point clouds.
FPNN [13] 和 Vote3D [26] 提出了特殊方法來處理稀疏性問題；然而，它們的操作仍然是在稀疏體積上進行的，處理非常大的點雲對它們來說具有挑戰性。

*Multiview CNNs:* [23, 18] have tried to render 3D point cloud or shapes into 2D images and then apply 2D conv nets to classify them. With well engineered image CNNs, this line of methods have achieved dominating performance on shape classification and retrieval tasks [21].
*多視圖 CNN：* [23, 18] 嘗試將 3D 點雲或形狀渲染成 2D 圖像，然後應用 2D 卷積網路對其進行分類。憑藉精心設計的圖像 CNN，這一系列方法在形狀分類和檢索任務上取得了主導性能 [21]。

However, it’s nontrivial to extend them to scene understanding or other 3D tasks such as point classification and shape completion.
然而，將它們擴展到場景理解或其他 3D 任務（如點分類和形狀補全）並非易事。

*Spectral CNNs:* Some latest works [4, 16] use spectral CNNs on meshes. However, these methods are currently constrained on manifold meshes such as organic objects and it’s not obvious how to extend them to non-isometric shapes such as furniture.
*頻譜 CNN：* 一些最新的工作 [4, 16] 在網格上使用頻譜 CNN。然而，這些方法目前受限於流形網格（如有機物體），且不明顯如何將其擴展到非等距形狀（如家具）。

*Feature-based DNNs:* [6, 8] firstly convert the 3D data into a vector, by extracting traditional shape features and then use a fully connected net to classify the shape. We think they are constrained by the representation power of the features extracted.
*基於特徵的 DNN：* [6, 8] 首先通過提取傳統形狀特徵將 3D 數據轉換為向量，然後使用全連接網路對形狀進行分類。我們認為它們受到提取特徵的表示能力的限制。

**Deep Learning on Unordered Sets** From a data structure point of view, a point cloud is an unordered set of vectors. While most works in deep learning focus on regular input representations like sequences (in speech and language processing), images and volumes (video or 3D data), not much work has been done in deep learning on point sets.
**無序集合上的深度學習** 從資料結構的角度來看，點雲是一組無序的向量集合。雖然深度學習的大多數工作都集中在規則的輸入表示上，如序列（在語音和語言處理中）、圖像和體積（視頻或 3D 數據），但在點集上的深度學習工作並不多。

One recent work from Oriol Vinyals et al [25] looks into this problem. They use a read-process-write network with attention mechanism to consume unordered input sets and show that their network has the ability to sort numbers.
Oriol Vinyals 等人 [25] 的一項近期工作研究了這個問題。他們使用帶有注意力機制的讀-寫-處理（read-process-write）網路來處理無序輸入集，並證明他們的網路具有排序數字的能力。

However, since their work focuses on generic sets and NLP applications, there lacks the role of geometry in the sets.
然而，由於他們的工作側重於通用集合和 NLP 應用，因此缺乏幾何在集合中的作用。

## 3. Problem Statement
## 3. 問題陳述

We design a deep learning framework that directly consumes unordered point sets as inputs. A point cloud is represented as a set of 3D points $\{P_i | i = 1, ..., n\}$, where each point $P_i$ is a vector of its $(x, y, z)$ coordinate plus extra feature channels such as color, normal etc. For simplicity and clarity, unless otherwise noted, we only use the $(x, y, z)$ coordinate as our point’s channels.
我們設計了一個直接處理無序點集作為輸入的深度學習框架。點雲表示為一組 3D 點 $\{P_i | i = 1, ..., n\}$，其中每個點 $P_i$ 是其 $(x, y, z)$ 座標加上額外特徵通道（如顏色、法線等）的向量。為簡單和清晰起見，除非另有說明，我們僅使用 $(x, y, z)$ 座標作為我們點的通道。

For the object classification task, the input point cloud is either directly sampled from a shape or pre-segmented from a scene point cloud. Our proposed deep network outputs $k$ scores for all the $k$ candidate classes.
對於物體分類任務，輸入點雲是直接從形狀採樣或從場景點雲預分割而來的。我們提出的深度網路輸出 $k$ 個候選類別的 $k$ 個分數。

For semantic segmentation, the input can be a single object for part region segmentation, or a sub-volume from a 3D scene for object region segmentation. Our model will output $n \times m$ scores for each of the $n$ points and each of the $m$ semantic sub-categories.
對於語義分割，輸入可以是單個物體進行部件區域分割，或是 3D 場景中的子體積進行物體區域分割。我們的模型將輸出 $n \times m$ 個分數，對應於 $n$ 個點中的每一個和 $m$ 個語義子類別中的每一個。

*Figure 2 Caption:*
**Figure 2. PointNet Architecture.** The classification network takes $n$ points as input, applies input and feature transformations, and then aggregates point features by max pooling. The output is classification scores for $k$ classes. The segmentation network is an extension to the classification net. It concatenates global and local features and outputs per point scores. “mlp” stands for multi-layer perceptron, numbers in bracket are layer sizes. Batchnorm is used for all layers with ReLU. Dropout layers are used for the last mlp in classification net.
**圖 2. PointNet 架構。** 分類網路以 $n$ 個點作為輸入，應用輸入和特徵變換，然後通過最大池化聚合點特徵。輸出是 $k$ 個類別的分類分數。分割網路是分類網路的擴展。它串聯全局和局部特徵並輸出每個點的分數。「mlp」代表多層感知器，括號中的數字是層的大小。BatchNorm 用於所有帶有 ReLU 的層。Dropout 層用於分類網路中的最後一個 mlp。

## 4. Deep Learning on Point Sets
## 4. 點集上的深度學習

The architecture of our network (Sec 4.2) is inspired by the properties of point sets in $\mathbb{R}^n$ (Sec 4.1).
我們網路的架構（第 4.2 節）受到 $\mathbb{R}^n$ 中點集屬性的啟發（第 4.1 節）。

### 4.1. Properties of Point Sets in $\mathbb{R}^n$
### 4.1. $\mathbb{R}^n$ 中點集的屬性

Our input is a subset of points from an Euclidean space. It has three main properties:
我們的輸入是歐幾里得空間中點的子集。它具有三個主要屬性：

*   **Unordered.** Unlike pixel arrays in images or voxel arrays in volumetric grids, point cloud is a set of points without specific order. In other words, a network that consumes $N$ 3D point sets needs to be invariant to $N!$ permutations of the input set in data feeding order.
    **無序性。** 與圖像中的像素陣列或體積網格中的體素陣列不同，點雲是一組沒有特定順序的點。換句話說，處理 $N$ 個 3D 點集的網路需要對數據輸入順序中輸入集合的 $N!$ 種排列保持不變。
*   **Interaction among points.** The points are from a space with a distance metric. It means that points are not isolated, and neighboring points form a meaningful subset. Therefore, the model needs to be able to capture local structures from nearby points, and the combinatorial interactions among local structures.
    **點之間的交互。** 這些點來自具有距離度量的空間。這意味著點不是孤立的，相鄰點形成有意義的子集。因此，模型需要能夠捕捉來自附近點的局部結構，以及局部結構之間的組合交互。
*   **Invariance under transformations.** As a geometric object, the learned representation of the point set should be invariant to certain transformations. For example, rotating and translating points all together should not modify the global point cloud category nor the segmentation of the points.
    **變換下的不變性。** 作為幾何物體，學習到的點集表示應該對某些變換保持不變。例如，同時旋轉和平移所有點不應修改全局點雲類別或點的分割。

### 4.2. PointNet Architecture
### 4.2. PointNet 架構

Our full network architecture is visualized in Fig 2, where the classification network and the segmentation network share a great portion of structures. Please read the caption of Fig 2 for the pipeline.
我們完整的網路架構如圖 2 所示，其中分類網路和分割網路共享很大一部分結構。請閱讀圖 2 的標題以了解流程。

Our network has three key modules: the max pooling layer as a symmetric function to aggregate information from all the points, a local and global information combination structure, and two joint alignment networks that align both input points and point features.
我們的網路有三個關鍵模塊：作為對稱函數的最大池化層，用於聚合來自所有點的資訊；局部和全局資訊組合結構；以及兩個聯合對齊網路，用於對齊輸入點和點特徵。

We will discuss our reason behind these design choices in separate paragraphs below.
我們將在下面的單獨段落中討論這些設計選擇背後的原因。

**Symmetry Function for Unordered Input** In order to make a model invariant to input permutation, three strategies exist: 1) sort input into a canonical order; 2) treat the input as a sequence to train an RNN, but augment the training data by all kinds of permutations; 3) use a simple symmetric function to aggregate the information from each point.
**無序輸入的對稱函數** 為了使模型對輸入排列保持不變，存在三種策略：1）將輸入排序為規範順序；2）將輸入視為序列來訓練 RNN，但通過各種排列增強訓練數據；3）使用簡單的對稱函數來聚合來自每個點的資訊。

Here, a symmetric function takes $n$ vectors as input and outputs a new vector that is invariant to the input order. For example, $+$ and $*$ operators are symmetric binary functions.
在這裡，對稱函數將 $n$ 個向量作為輸入，並輸出一個對輸入順序不變的新向量。例如，$+$ 和 $*$ 運算符是對稱二元函數。

While sorting sounds like a simple solution, in high dimensional space there in fact does not exist an ordering that is stable w.r.t. point perturbations in the general sense. This can be easily shown by contradiction.
雖然排序聽起來是一個簡單的解決方案，但在高維空間中，實際上不存在在一般意義上對點擾動穩定的排序。這可以通過反證法輕鬆證明。

If such an ordering strategy exists, it defines a bijection map between a high-dimensional space and a $1d$ real line. It is not hard to see, to require an ordering to be stable w.r.t point perturbations is equivalent to requiring that this map preserves spatial proximity as the dimension reduces, a task that cannot be achieved in the general case.
如果存在這樣的排序策略，它定義了高維空間和 $1d$ 實線之間的雙射映射。不難看出，要求排序對點擾動穩定等同於要求該映射在維度降低時保持空間鄰近性，這是在一般情況下無法完成的任務。

Therefore, sorting does not fully resolve the ordering issue, and it’s hard for a network to learn a consistent mapping from input to output as the ordering issue persists. As shown in experiments (Fig 5), we find that applying a MLP directly on the sorted point set performs poorly, though slightly better than directly processing an unsorted input.
因此，排序並不能完全解決順序問題，並且由於順序問題持續存在，網路很難學習從輸入到輸出的一致映射。如實驗所示（圖 5），我們發現直接在排序的點集上應用 MLP 效果不佳，儘管比直接處理未排序的輸入稍好。

The idea to use RNN considers the point set as a sequential signal and hopes that by training the RNN with randomly permuted sequences, the RNN will become invariant to input order. However in “OrderMatters” [25] the authors have shown that order does matter and cannot be totally omitted.
使用 RNN 的想法將點集視為序列訊號，並希望通過使用隨機排列的序列訓練 RNN，使 RNN 對輸入順序變得不變。然而在「OrderMatters」[25] 中，作者表明順序確實很重要，不能完全忽略。

While RNN has relatively good robustness to input ordering for sequences with small length (dozens), it’s hard to scale to thousands of input elements, which is the common size for point sets. Empirically, we have also shown that model based on RNN does not perform as well as our proposed method (Fig 5).
雖然 RNN 對於長度較小（幾十個）的序列的輸入順序具有相對較好的魯棒性，但很難擴展到數千個輸入元素，這是點集的常見大小。在實證上，我們也表明基於 RNN 的模型表現不如我們提出的方法（圖 5）。

Our idea is to approximate a general function defined on a point set by applying a symmetric function on transformed elements in the set:
我們的想法是通過在集合中的變換元素上應用對稱函數來近似定義在點集上的通用函數：

$$f(\{x_1, \dots, x_n\}) \approx g(h(x_1), \dots, h(x_n)), \quad (1)$$

where $f : 2^{\mathbb{R}^N} \to \mathbb{R}$, $h : \mathbb{R}^N \to \mathbb{R}^K$ and $g : \underbrace{\mathbb{R}^K \times \cdots \times \mathbb{R}^K}_{n} \to \mathbb{R}$ is a symmetric function.
其中 $f : 2^{\mathbb{R}^N} \to \mathbb{R}$, $h : \mathbb{R}^N \to \mathbb{R}^K$ 且 $g : \underbrace{\mathbb{R}^K \times \cdots \times \mathbb{R}^K}_{n} \to \mathbb{R}$ 是一個對稱函數。

Empirically, our basic module is very simple: we approximate $h$ by a multi-layer perceptron network and $g$ by a composition of a single variable function and a max pooling function. This is found to work well by experiments.
在實證上，我們的基本模塊非常簡單：我們用多層感知器網路近似 $h$，用單變量函數和最大池化函數的組合近似 $g$。實驗發現這效果很好。

Through a collection of $h$, we can learn a number of $f$’s to capture different properties of the set.
通過 $h$ 的集合，我們可以學習多個 $f$ 來捕捉集合的不同屬性。

While our key module seems simple, it has interesting properties (see Sec 5.3) and can achieve strong performace (see Sec 5.1) in a few different applications. Due to the simplicity of our module, we are also able to provide theoretical analysis as in Sec 4.3.
雖然我們的關鍵模塊看起來很簡單，但它具有有趣的屬性（見第 5.3 節），並且可以在幾個不同的應用中實現強大的性能（見第 5.1 節）。由於我們模塊的簡單性，我們也能夠提供如第 4.3 節所述的理論分析。

**Local and Global Information Aggregation** The output from the above section forms a vector $[f_1, \dots, f_K]$, which is a global signature of the input set. We can easily train a SVM or multi-layer perceptron classifier on the shape global features for classification.
**局部和全局資訊聚合** 上一節的輸出形成一個向量 $[f_1, \dots, f_K]$，它是輸入集的全局簽名。我們可以很容易地在形狀全局特徵上訓練 SVM 或多層感知器分類器進行分類。

However, point segmentation requires a combination of local and global knowledge. We can achieve this by a simple yet highly effective manner.
然而，點分割需要局部和全局知識的結合。我們可以通過一種簡單但非常有效的方式來實現這一點。

Our solution can be seen in Fig 2 (*Segmentation Network*). After computing the global point cloud feature vector, we feed it back to per point features by concatenating the global feature with each of the point features. Then we extract new per point features based on the combined point features - this time the per point feature is aware of both the local and global information.
我們的解決方案見圖 2（*分割網路*）。在計算出全局點雲特徵向量後，我們通過將全局特徵與每個點特徵串聯，將其反饋給逐點特徵。然後我們基於組合的點特徵提取新的逐點特徵——這一次，逐點特徵同時感知到局部和全局資訊。

With this modification our network is able to predict per point quantities that rely on both local geometry and global semantics. For example we can accurately predict per-point normals (fig in supplementary), validating that the network is able to summarize information from the point’s local neighborhood. In experiment session, we also show that our model can achieve state-of-the-art performance on shape part segmentation and scene segmentation.
通過這種修改，我們的網路能夠預測依賴於局部幾何和全局語義的逐點量。例如，我們可以準確地預測逐點法線（圖在補充材料中），驗證了網路能夠總結來自點的局部鄰域的資訊。在實驗部分，我們還展示了我們的模型可以在形狀部件分割和場景分割上達到最先進的性能。

**Joint Alignment Network** The semantic labeling of a point cloud has to be invariant if the point cloud undergoes certain geometric transformations, such as rigid transformation. We therefore expect that the learnt representation by our point set is invariant to these transformations.
**聯合對齊網路** 如果點雲經歷某些幾何變換（如剛性變換），點雲的語義標註必須保持不變。因此，我們期望我們的點集學習到的表示對這些變換是不變的。

A natural solution is to align all input set to a canonical space before feature extraction. Jaderberg et al. [9] introduces the idea of spatial transformer to align 2D images through sampling and interpolation, achieved by a specifically tailored layer implemented on GPU.
一個自然的解決方案是在特徵提取之前將所有輸入集對齊到規範空間。Jaderberg 等人 [9] 引入了空間變換器的概念，通過採樣和插值來對齊 2D 圖像，這是通過在 GPU 上實現的專門定製層來實現的。

Our input form of point clouds allows us to achieve this goal in a much simpler way compared with [9]. We do not need to invent any new layers and no alias is introduced as in the image case. We predict an affine transformation matrix by a mini-network (T-net in Fig 2) and directly apply this transformation to the coordinates of input points.
與 [9] 相比，我們的點雲輸入形式允許我們以更簡單的方式實現這一目標。我們不需要發明任何新層，也不會像圖像情況那樣引入混疊。我們通過一個微型網路（圖 2 中的 T-net）預測仿射變換矩陣，並直接將此變換應用於輸入點的座標。

The mini-network itself resembles the big network and is composed by basic modules of point independent feature extraction, max pooling and fully connected layers. More details about the T-net are in the supplementary.
微型網路本身類似於大網路，由點獨立特徵提取、最大池化和全連接層等基本模塊組成。關於 T-net 的更多細節在補充材料中。

This idea can be further extended to the alignment of feature space, as well. We can insert another alignment network on point features and predict a feature transformation matrix to align features from different input point clouds.
這個想法還可以進一步擴展到特徵空間的對齊。我們可以在點特徵上插入另一個對齊網路，並預測特徵變換矩陣來對齊來自不同輸入點雲的特徵。

However, transformation matrix in the feature space has much higher dimension than the spatial transform matrix, which greatly increases the difficulty of optimization. We therefore add a regularization term to our softmax training loss. We constrain the feature transformation matrix to be close to orthogonal matrix:
然而，特徵空間中的變換矩陣具有比空間變換矩陣高得多的維度，這大大增加了優化的難度。因此，我們在 softmax 訓練損失中添加了一個正則化項。我們約束特徵變換矩陣接近正交矩陣：

$$L_{reg} = ||I - AA^T||_F^2, \quad (2)$$

where $A$ is the feature alignment matrix predicted by a mini-network. An orthogonal transformation will not lose information in the input, thus is desired. We find that by adding the regularization term, the optimization becomes more stable and our model achieves better performance.
其中 $A$ 是微型網路預測的特徵對齊矩陣。正交變換不會丟失輸入中的資訊，因此是理想的。我們發現通過添加正則化項，優化變得更加穩定，我們的模型實現了更好的性能。

### 4.3. Theoretical Analysis
### 4.3. 理論分析

**Universal approximation** We first show the universal approximation ability of our neural network to continuous set functions. By the continuity of set functions, intuitively, a small perturbation to the input point set should not greatly change the function values, such as classification or segmentation scores.
**通用近似** 我們首先展示我們的神經網路對連續集合函數的通用近似能力。根據集合函數的連續性，直觀上，對輸入點集的小擾動不應極大地改變函數值，例如分類或分割分數。

Formally, let $\mathcal{X} = \{S : S \subseteq [0, 1]^m \text{ and } |S| = n\}$, $f : \mathcal{X} \to \mathbb{R}$ is a continuous set function on $\mathcal{X}$ w.r.t to Hausdorff distance $d_H(\cdot, \cdot)$, i.e., $\forall \epsilon > 0, \exists \delta > 0$, for any $S, S' \in \mathcal{X}$, if $d_H(S, S') < \delta$, then $|f(S) - f(S')| < \epsilon$. Our theorem says that $f$ can be arbitrarily approximated by our network given enough neurons at the max pooling layer, i.e., $K$ in (1) is sufficiently large.
形式上，令 $\mathcal{X} = \{S : S \subseteq [0, 1]^m \text{ and } |S| = n\}$，$f : \mathcal{X} \to \mathbb{R}$ 是 $\mathcal{X}$ 上關於豪斯多夫距離 $d_H(\cdot, \cdot)$ 的連續集合函數，即 $\forall \epsilon > 0, \exists \delta > 0$，對於任何 $S, S' \in \mathcal{X}$，如果 $d_H(S, S') < \delta$，則 $|f(S) - f(S')| < \epsilon$。我們的定理表明，只要最大池化層有足夠的神經元，即 (1) 中的 $K$ 足夠大，我們的網路就可以任意近似 $f$。

*Figure 3 Caption:*
**Figure 3. Qualitative results for part segmentation.** We visualize the CAD part segmentation results across all 16 object categories. We show both results for partial simulated Kinect scans (left block) and complete ShapeNet CAD models (right block).
**圖 3. 部件分割的定性結果。** 我們可視化了所有 16 個物體類別的 CAD 部件分割結果。我們展示了部分模擬 Kinect 掃描（左區塊）和完整 ShapeNet CAD 模型（右區塊）的結果。

**Theorem 1.** Suppose $f : \mathcal{X} \to \mathbb{R}$ is a continuous set function w.r.t Hausdorff distance $d_H(\cdot, \cdot)$. $\forall \epsilon > 0, \exists$ a continuous function $h$ and a symmetric function $g(x_1, \dots, x_n) = \gamma \circ \text{MAX}$, such that for any $S \in \mathcal{X}$,
**定理 1.** 假設 $f : \mathcal{X} \to \mathbb{R}$ 是關於豪斯多夫距離 $d_H(\cdot, \cdot)$ 的連續集合函數。$\forall \epsilon > 0, \exists$ 一個連續函數 $h$ 和一個對稱函數 $g(x_1, \dots, x_n) = \gamma \circ \text{MAX}$，使得對於任何 $S \in \mathcal{X}$，

$$ \left| f(S) - \gamma \left( \underset{x_i \in S}{\text{MAX}} \{h(x_i)\} \right) \right| < \epsilon $$

where $x_1, \dots, x_n$ is the full list of elements in $S$ ordered arbitrarily, $\gamma$ is a continuous function, and $\text{MAX}$ is a vector max operator that takes $n$ vectors as input and returns a new vector of the element-wise maximum.
其中 $x_1, \dots, x_n$ 是 $S$ 中任意排序的完整元素列表，$\gamma$ 是一個連續函數，$\text{MAX}$ 是一個向量最大值運算符，它將 $n$ 個向量作為輸入並返回一個元素級最大值的新向量。

The proof to this theorem can be found in our supplementary material. The key idea is that in the worst case the network can learn to convert a point cloud into a volumetric representation, by partitioning the space into equal-sized voxels. In practice, however, the network learns a much smarter strategy to probe the space, as we shall see in point function visualizations.
該定理的證明可以在我們的補充材料中找到。關鍵思想是，在最壞的情況下，網路可以學習通過將空間劃分為等大小的體素，將點雲轉換為體積表示。然而，在實踐中，網路學習了一種更聰明的策略來探索空間，正如我們將在點函數可視化中看到的那樣。

**Bottleneck dimension and stability** Theoretically and experimentally we find that the expressiveness of our network is strongly affected by the dimension of the max pooling layer, i.e., $K$ in (1). Here we provide an analysis, which also reveals properties related to the stability of our model.
**瓶頸維度和穩定性** 在理論和實驗上，我們發現我們網路的表達能力受到最大池化層維度的強烈影響，即 (1) 中的 $K$。在這裡，我們提供一個分析，這也揭示了與我們模型穩定性相關的屬性。

We define $\mathbf{u} = \underset{x_i \in S}{\text{MAX}} \{h(x_i)\}$ to be the sub-network of $f$ which maps a point set in $[0, 1]^m$ to a $K$-dimensional vector. The following theorem tells us that small corruptions or extra noise points in the input set are not likely to change the output of our network:
我們定義 $\mathbf{u} = \underset{x_i \in S}{\text{MAX}} \{h(x_i)\}$ 為 $f$ 的子網路，它將 $[0, 1]^m$ 中的點集映射為 $K$ 維向量。以下定理告訴我們，輸入集中的小損壞或額外噪聲點不太可能改變我們網路的輸出：

**Theorem 2.** Suppose $\mathbf{u} : \mathcal{X} \to \mathbb{R}^K$ such that $\mathbf{u} = \underset{x_i \in S}{\text{MAX}} \{h(x_i)\}$ and $f = \gamma \circ \mathbf{u}$. Then,
**定理 2.** 假設 $\mathbf{u} : \mathcal{X} \to \mathbb{R}^K$ 使得 $\mathbf{u} = \underset{x_i \in S}{\text{MAX}} \{h(x_i)\}$ 且 $f = \gamma \circ \mathbf{u}$。那麼，

(a) $\forall S, \exists \mathcal{C}_S, \mathcal{N}_S \subseteq \mathcal{X}, f(T) = f(S)$ if $\mathcal{C}_S \subseteq T \subseteq \mathcal{N}_S$;
(b) $|\mathcal{C}_S| \le K$

We explain the implications of the theorem. (a) says that $f(S)$ is unchanged up to the input corruption if all points in $\mathcal{C}_S$ are preserved; it is also unchanged with extra noise points up to $\mathcal{N}_S$. (b) says that $\mathcal{C}_S$ only contains a bounded number of points, determined by $K$ in (1).
我們解釋該定理的含義。(a) 表示如果 $\mathcal{C}_S$ 中的所有點都保留，則 $f(S)$ 在輸入損壞下保持不變；在達到 $\mathcal{N}_S$ 之前的額外噪聲點下也保持不變。(b) 表示 $\mathcal{C}_S$ 僅包含有限數量的點，由 (1) 中的 $K$ 決定。

In other words, $f(S)$ is in fact totally determined by a finite subset $\mathcal{C}_S \subseteq S$ of less or equal to $K$ elements. We therefore call $\mathcal{C}_S$ the *critical point set* of $S$ and $K$ the *bottleneck dimension* of $f$.
換句話說，$f(S)$ 實際上完全由小於或等於 $K$ 個元素的有限子集 $\mathcal{C}_S \subseteq S$ 決定。因此，我們稱 $\mathcal{C}_S$ 為 $S$ 的**關鍵點集**，稱 $K$ 為 $f$ 的**瓶頸維度**。

Combined with the continuity of $h$, this explains the robustness of our model w.r.t point perturbation, corruption and extra noise points. The robustness is gained in analogy to the sparsity principle in machine learning models.
結合 $h$ 的連續性，這解釋了我們的模型對點擾動、損壞和額外噪聲點的魯棒性。魯棒性的獲得類似於機器學習模型中的稀疏性原則。

**Intuitively, our network learns to summarize a shape by a sparse set of key points.** In experiment section we see that the key points form the skeleton of an object.
**直觀地說，我們的網路學會了通過一組稀疏的關鍵點來總結形狀。** 在實驗部分，我們看到關鍵點形成了物體的骨架。

## 5. Experiment
## 5. 實驗

Experiments are divided into four parts. First, we show PointNets can be applied to multiple 3D recognition tasks (Sec 5.1). Second, we provide detailed experiments to validate our network design (Sec 5.2). At last we visualize what the network learns (Sec 5.3) and analyze time and space complexity (Sec 5.4).
實驗分為四個部分。首先，我們展示 PointNet 可以應用於多種 3D 識別任務（第 5.1 節）。其次，我們提供詳細的實驗來驗證我們的網路設計（第 5.2 節）。最後，我們可視化網路學到的內容（第 5.3 節）並分析時間和空間複雜度（第 5.4 節）。

### 5.1. Applications
### 5.1. 應用

In this section we show how our network can be trained to perform 3D object classification, object part segmentation and semantic scene segmentation [^1]. Even though we are working on a brand new data representation (point sets), we are able to achieve comparable or even better performance on benchmarks for several tasks.
在本節中，我們展示如何訓練我們的網路來執行 3D 物體分類、物體部件分割和語義場景分割 [^1]。即使我們使用的是全新的數據表示（點集），我們也能夠在幾個任務的基準測試上取得相當甚至更好的性能。

[^1]: More application examples such as correspondence and point cloud based CAD model retrieval are included in supplementary material.
[^1]: 更多應用範例（如對應關係和基於點雲的 CAD 模型檢索）包含在補充材料中。

**3D Object Classification** Our network learns global point cloud feature that can be used for object classification. We evaluate our model on the ModelNet40 [28] shape classification benchmark. There are 12,311 CAD models from 40 man-made object categories, split into 9,843 for training and 2,468 for testing.
**3D 物體分類** 我們的網路學習可用於物體分類的全局點雲特徵。我們在 ModelNet40 [28] 形狀分類基準上評估我們的模型。來自 40 個人造物體類別的 12,311 個 CAD 模型，分為 9,843 個用於訓練和 2,468 個用於測試。

*Table 1 Caption:*
**Table 1. Classification results on ModelNet40.** Our net achieves state-of-the-art among deep nets on 3D input.
**表 1. ModelNet40 上的分類結果。** 我們的網路在基於 3D 輸入的深度網路中達到了最先進的水平。

| | 輸入 | #視圖 | 平均類別準確率 | 整體準確率 |
|---|---|---|---|---|
| SPH [11] | 網格 | - | 68.2 | - |
| 3DShapeNets [28] | 體積 | 1 | 77.3 | 84.7 |
| VoxNet [17] | 體積 | 12 | 83.0 | 85.9 |
| Subvolume [18] | 體積 | 20 | 86.0 | **89.2** |
| LFD [28] | 圖像 | 10 | 75.5 | - |
| MVCNN [23] | 圖像 | 80 | **90.1** | - |
| Ours baseline | 點 | - | 72.6 | 77.4 |
| Ours PointNet | 點 | 1 | 86.2 | **89.2** |

While previous methods focus on volumetric and mult-view image representations, we are the first to directly work on raw point cloud.
雖然以前的方法側重於體積和多視圖圖像表示，但我們是第一個直接處理原始點雲的。

We uniformly sample 1024 points on mesh faces according to face area and normalize them into a unit sphere. During training we augment the point cloud on-the-fly by randomly rotating the object along the up-axis and jitter the position of each points by a Gaussian noise with zero mean and 0.02 standard deviation.
我們根據面面積在網格面上均勻採樣 1024 個點，並將它們標準化為單位球體。在訓練期間，我們通過沿上軸隨機旋轉物體並通過零均值和 0.02 標準差的高斯噪聲抖動每個點的位置來即時增強點雲。

In Table 1, we compare our model with previous works as well as our baseline using MLP on traditional features extracted from point cloud (point density, D2, shape contour etc.).
在表 1 中，我們將我們的模型與以前的工作以及我們在使用從點雲提取的傳統特徵（點密度、D2、形狀輪廓等）上的 MLP 基線進行了比較。

Our model achieved state-of-the-art performance among methods based on 3D input (volumetric and point cloud). With only fully connected layers and max pooling, our net gains a strong lead in inference speed and can be easily parallelized in CPU as well.
我們的模型在基於 3D 輸入（體積和點雲）的方法中取得了最先進的性能。僅使用全連接層和最大池化，我們的網路在推理速度上獲得了強大的領先優勢，並且也可以在 CPU 中輕鬆並行化。

There is still a small gap between our method and multi-view based method (MVCNN [23]), which we think is due to the loss of fine geometry details that can be captured by rendered images.
我們的方法與基於多視圖的方法（MVCNN [23]）之間仍然存在很小的差距，我們認為這是由於丟失了渲染圖像可以捕捉到的精細幾何細節。

**3D Object Part Segmentation** Part segmentation is a challenging fine-grained 3D recognition task. Given a 3D scan or a mesh model, the task is to assign part category label (e.g. chair leg, cup handle) to each point or face.
**3D 物體部件分割** 部件分割是一項具有挑戰性的細粒度 3D 識別任務。給定 3D 掃描或網格模型，任務是為每個點或面分配部件類別標籤（例如椅腳、杯柄）。

We evaluate on ShapeNet part data set from [29], which contains 16,881 shapes from 16 categories, annotated with 50 parts in total. Most object categories are labeled with two to five parts. Ground truth annotations are labeled on sampled points on the shapes.
我們在 [29] 的 ShapeNet 部件數據集上進行評估，該數據集包含來自 16 個類別的 16,881 個形狀，總共註釋了 50 個部件。大多數物體類別標有兩到五個部件。地面實況註釋標記在形狀的採樣點上。

We formulate part segmentation as a per-point classification problem. Evaluation metric is mIoU on points. For each shape $S$ of category $C$, to calculate the shape’s mIoU: For each part type in category $C$, compute IoU between groundtruth and prediction. If the union of groundtruth and prediction points is empty, then count part IoU as 1.
我們將部件分割制定為逐點分類問題。評估指標是點上的 mIoU。對於類別 $C$ 的每個形狀 $S$，計算形狀的 mIoU：對於類別 $C$ 中的每個部件類型，計算地面實況和預測之間的 IoU。如果地面實況和預測點的並集為空，則將部件 IoU 計為 1。

Then we average IoUs for all part types in category $C$ to get mIoU for that shape. To calculate mIoU for the category, we take average of mIoUs for all shapes in that category.
然後我們平均類別 $C$ 中所有部件類型的 IoU 以獲得該形狀的 mIoU。為了計算類別的 mIoU，我們取該類別中所有形狀的 mIoU 平均值。

In this section, we compare our segmentation version PointNet (a modified version of Fig 2, *Segmentation Network*) with two traditional methods [27] and [29] that both take advantage of point-wise geometry features and correspondences between shapes, as well as our own 3D CNN baseline. See supplementary for the detailed modifications and network architecture for the 3D CNN.
在本節中，我們將我們的分割版 PointNet（圖 2 的修改版，*分割網路*）與兩種傳統方法 [27] 和 [29] 進行比較，這兩種方法都利用了逐點幾何特徵和形狀之間的對應關係，以及我們自己的 3D CNN 基線。有關 3D CNN 的詳細修改和網路架構，請參閱補充材料。

*Table 2 Caption:*
**Table 2. Segmentation results on ShapeNet part dataset.** Metric is mIoU(%) on points. We compare with two traditional methods [27] and [29] and a 3D fully convolutional network baseline proposed by us. Our PointNet method achieved the state-of-the-art in mIoU.
**表 2. ShapeNet 部件數據集上的分割結果。** 指標是點上的 mIoU(%)。我們與兩種傳統方法 [27] 和 [29] 以及我們提出的 3D 全卷積網路基線進行了比較。我們的 PointNet 方法在 mIoU 方面達到了最先進的水平。

| | 平均 | 飛機 | 包 | 帽 | 車 | 椅 | 耳機 | 吉他 | 刀 | 燈 | 筆電 | 摩托 | 杯 | 手槍 | 火箭 | 滑板 | 桌 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| # 形狀 | | 2690 | 76 | 55 | 898 | 3758 | 69 | 787 | 392 | 1547 | 451 | 202 | 184 | 283 | 66 | 152 | 5271 |
| Wu [27] | - | 63.2 | - | - | - | 73.5 | - | - | - | 74.4 | - | - | - | - | - | - | 74.8 |
| Yi [29] | 81.4 | 81.0 | 78.4 | 77.7 | **75.7** | 87.6 | 61.9 | **92.0** | 85.4 | **82.5** | **95.7** | **70.6** | 91.9 | **85.9** | 53.1 | 69.8 | 75.3 |
| 3DCNN | 79.4 | 75.1 | 72.8 | 73.3 | 70.0 | 87.2 | 63.5 | 88.4 | 79.6 | 74.4 | 93.9 | 58.7 | 91.8 | 76.4 | 51.2 | 65.3 | 77.1 |
| Ours | **83.7** | **83.4** | **78.7** | **82.5** | 74.9 | **89.6** | **73.0** | 91.5 | **85.9** | 80.8 | 95.3 | 65.2 | **93.0** | 81.2 | **57.9** | **72.8** | **80.6** |

In Table 2, we report per-category and mean IoU(%) scores. We observe a 2.3% mean IoU improvement and our net beats the baseline methods in most categories.
在表 2 中，我們報告了每個類別和平均 IoU(%) 分數。我們觀察到 2.3% 的平均 IoU 提升，並且我們的網路在大多數類別中擊敗了基線方法。

We also perform experiments on simulated Kinect scans to test the robustness of these methods. For every CAD model in the ShapeNet part data set, we use Blensor Kinect Simulator [7] to generate incomplete point clouds from six random viewpoints.
我們還在模擬 Kinect 掃描上進行實驗，以測試這些方法的魯棒性。對於 ShapeNet 部件數據集中的每個 CAD 模型，我們使用 Blensor Kinect 模擬器 [7] 從六個隨機視點生成不完整的點雲。

We train our PointNet on the complete shapes and partial scans with the same network architecture and training setting. Results show that we lose only 5.3% mean IoU. In Fig 3, we present qualitative results on both complete and partial data. One can see that though partial data is fairly challenging, our predictions are reasonable.
我們使用相同的網路架構和訓練設置在完整形狀和部分掃描上訓練我們的 PointNet。結果顯示我們僅損失了 5.3% 的平均 IoU。在圖 3 中，我們展示了完整和部分數據的定性結果。可以看出，雖然部分數據相當具挑戰性，但我們的預測是合理的。

**Semantic Segmentation in Scenes** Our network on part segmentation can be easily extended to semantic scene segmentation, where point labels become semantic object classes instead of object part labels.
**場景中的語義分割** 我們的部件分割網路可以很容易地擴展到語義場景分割，其中點標籤變為語義物體類別而不是物體部件標籤。

We experiment on the Stanford 3D semantic parsing data set [1]. The dataset contains 3D scans from Matterport scanners in 6 areas including 271 rooms. Each point in the scan is annotated with one of the semantic labels from 13 categories (chair, table, floor, wall etc. plus clutter).
我們在 Stanford 3D 語義解析數據集 [1] 上進行實驗。該數據集包含來自 Matterport 掃描儀的 6 個區域（包括 271 個房間）的 3D 掃描。掃描中的每個點都標有來自 13 個類別（椅、桌、地板、牆壁等以及雜物）的語義標籤之一。

To prepare training data, we firstly split points by room, and then sample rooms into blocks with area 1m by 1m. We train our segmentation version of PointNet to predict per point class in each block.
為了準備訓練數據，我們首先按房間分割點，然後將房間採樣為 1m x 1m 的區塊。我們訓練分割版 PointNet 來預測每個區塊中的逐點類別。

*Table 3 Caption:*
**Table 3. Results on semantic segmentation in scenes.** Metric is average IoU over 13 classes (structural and furniture elements plus clutter) and classification accuracy calculated on points.
**表 3. 場景語義分割結果。** 指標是 13 個類別（結構和家具元素加上雜物）的平均 IoU 和在點上計算的分類準確率。

| | 平均 IoU | 整體準確率 |
|---|---|---|
| Ours baseline | 20.12 | 53.19 |
| Ours PointNet | **47.71** | **78.62** |

*Table 4 Caption:*
**Table 4. Results on 3D object detection in scenes.** Metric is average precision with threshold IoU 0.5 computed in 3D volumes.
**表 4. 場景中 3D 物體檢測結果。** 指標是在 3D 體積中計算的閾值 IoU 為 0.5 的平均精度。

| | 桌 | 椅 | 沙發 | 櫃 | 平均 |
|---|---|---|---|---|---|
| # 實例 | 455 | 1363 | 55 | 137 | |
| Armeni et al. [1] | 46.02 | 16.15 | **6.78** | 3.91 | 18.22 |
| Ours | **46.67** | **33.80** | 4.76 | **11.72** | **24.24** |

Each point is represented by a 9-dim vector of XYZ, RGB and normalized location as to the room (from 0 to 1). At training time, we randomly sample 4096 points in each block on-the-fly. At test time, we test on all the points. We follow the same protocol as [1] to use k-fold strategy for train and test.
每個點由 XYZ、RGB 和相對於房間的標準化位置（從 0 到 1）的 9 維向量表示。在訓練時，我們即時在每個區塊中隨機採樣 4096 個點。在測試時，我們對所有點進行測試。我們遵循與 [1] 相同的協議，使用 k-折策略進行訓練和測試。

We compare our method with a baseline using handcrafted point features. The baseline extracts the same 9-dim local features and three additional ones: local point density, local curvature and normal. We use standard MLP as the classifier.
我們將我們的方法與使用手工製作點特徵的基線進行比較。基線提取相同的 9 維局部特徵和三個額外的特徵：局部點密度、局部曲率和法線。我們使用標準 MLP 作為分類器。

Results are shown in Table 3, where our PointNet method significantly outperforms the baseline method. In Fig 4, we show qualitative segmentation results. Our network is able to output smooth predictions and is robust to missing points and occlusions.
結果如表 3 所示，我們的 PointNet 方法顯著優於基線方法。在圖 4 中，我們展示了定性分割結果。我們的網路能夠輸出平滑的預測，並且對缺失點和遮擋具有魯棒性。

Based on the semantic segmentation output from our network, we further build a 3D object detection system using connected component for object proposal (see supplementary for details). We compare with previous state-of-the-art method in Table 4.
基於我們網路的語義分割輸出，我們進一步構建了一個使用連通分量進行物體提議的 3D 物體檢測系統（詳見補充材料）。我們在表 4 中與以前的最先進方法進行了比較。

The previous method is based on a sliding shape method (with CRF post processing) with SVMs trained on local geometric features and global room context feature in voxel grids. Our method outperforms it by a large margin on the furniture categories reported.
以前的方法基於滑動形狀方法（帶 CRF 後處理），使用在體素網格中的局部幾何特徵和全局房間上下文特徵上訓練的 SVM。我們的方法在報告的家具類別上大幅超越了它。

### 5.2. Architecture Design Analysis
### 5.2. 架構設計分析

In this section we validate our design choices by control experiments. We also show the effects of our network’s hyperparameters.
在本節中，我們通過對照實驗驗證我們的設計選擇。我們還展示了網路超參數的影響。

**Comparison with Alternative Order-invariant Methods** As mentioned in Sec 4.2, there are at least three options for consuming unordered set inputs. We use the ModelNet40 shape classification problem as a test bed for comparisons of those options, the following two control experiment will also use this task.
**與替代順序不變方法的比較** 如第 4.2 節所述，至少有三種處理無序集合輸入的選項。我們使用 ModelNet40 形狀分類問題作為比較這些選項的測試平台，接下來的兩個對照實驗也將使用此任務。

The baselines (illustrated in Fig 5) we compared with include multi-layer perceptron on unsorted and sorted points as $n \times 3$ arrays, RNN model that considers input point as a sequence, and a model based on symmetry functions.
我們比較的基線（如圖 5 所示）包括在作為 $n \times 3$ 陣列的未排序和排序點上的多層感知器，將輸入點視為序列的 RNN 模型，以及基於對稱函數的模型。

*Figure 4 Caption:*
**Figure 4. Qualitative results for semantic segmentation.** Top row is input point cloud with color. Bottom row is output semantic segmentation result (on points) displayed in the same camera viewpoint as input.
**圖 4. 語義分割的定性結果。** 上排是帶顏色的輸入點雲。下排是顯示在與輸入相同相機視點的輸出語義分割結果（在點上）。

*Figure 5 Caption:*
**Figure 5. Three approaches to achieve order invariance.** Multi-layer perceptron (MLP) applied on points consists of 5 hidden layers with neuron sizes 64,64,64,128,1024, all points share a single copy of MLP. The MLP close to the output consists of two layers with sizes 512,256.
**圖 5. 實現順序不變性的三種方法。** 應用於點的多層感知器 (MLP) 由 5 個隱藏層組成，神經元大小為 64,64,64,128,1024，所有點共享 MLP 的單個副本。接近輸出的 MLP 由兩層組成，大小為 512,256。

| | 準確率 |
|---|---|
| MLP (未排序輸入) | 24.2 |
| MLP (排序輸入) | 45.0 |
| LSTM | 78.5 |
| Attention sum | 83.0 |
| Average pooling | 83.8 |
| Max pooling | **87.1** |

The symmetry operation we experimented include max pooling, average pooling and an attention based weighted sum. The attention method is similar to that in [25], where a scalar score is predicted from each point feature, then the score is normalized across points by computing a softmax.
我們實驗的對稱操作包括最大池化、平均池化和基於注意力的加權和。注意力方法與 [25] 中的類似，其中從每個點特徵預測標量分數，然後通過計算 softmax 對分數進行跨點歸一化。

The weighted sum is then computed on the normalized scores and the point features. As shown in Fig 5, max-pooling operation achieves the best performance by a large winning margin, which validates our choice.
然後在歸一化分數和點特徵上計算加權和。如圖 5 所示，最大池化操作以巨大的優勢獲得了最佳性能，這驗證了我們的選擇。

**Effectiveness of Input and Feature Transformations** In Table 5 we demonstrate the positive effects of our input and feature transformations (for alignment). It’s interesting to see that the most basic architecture already achieves quite reasonable results.
**輸入和特徵變換的有效性** 在表 5 中，我們展示了輸入和特徵變換（用於對齊）的積極效果。有趣的是，最基本的架構已經取得了相當合理的結果。

Using input transformation gives a 0.8% performance boost. The regularization loss is necessary for the higher dimension transform to work. By combining both transformations and the regularization term, we achieve the best performance.
使用輸入變換帶來了 0.8% 的性能提升。正則化損失對於高維變換的運作是必要的。通過結合兩種變換和正則化項，我們實現了最佳性能。

**Robustness Test** We show our PointNet, while simple and effective, is robust to various kinds of input corruptions. We use the same architecture as in Fig 5’s max pooling network. Input points are normalized into a unit sphere. Results are in Fig 6.
**魯棒性測試** 我們展示了我們的 PointNet 雖然簡單有效，但對各種輸入損壞具有魯棒性。我們使用與圖 5 的最大池化網路相同的架構。輸入點被歸一化為單位球體。結果如圖 6 所示。

As to missing points, when there are 50% points missing, the accuracy only drops by 2.4% and 3.8% w.r.t. furthest and random input sampling. Our net is also robust to outlier points, if it has seen those during training.
關於缺失點，當缺失 50% 的點時，相對於最遠和隨機輸入採樣，準確率僅下降 2.4% 和 3.8%。如果網路在訓練期間見過異常值點，它對異常值點也具有魯棒性。

*Table 5 Caption:*
**Table 5. Effects of input feature transforms.** Metric is overall classification accuracy on ModelNet40 test set.
**表 5. 輸入特徵變換的效果。** 指標是 ModelNet40 測試集上的整體分類準確率。

| 變換 | 準確率 |
|---|---|
| 無 | 87.1 |
| 輸入 (3x3) | 87.9 |
| 特徵 (64x64) | 86.9 |
| 特徵 (64x64) + reg. | 87.4 |
| 兩者 | **89.2** |

We evaluate two models: one trained on points with $(x, y, z)$ coordinates; the other on $(x, y, z)$ plus point density. The net has more than 80% accuracy even when 20% of the points are outliers. Fig 6 right shows the net is robust to point perturbations.
我們評估了兩個模型：一個在具有 $(x, y, z)$ 座標的點上訓練；另一個在 $(x, y, z)$ 加上點密度上訓練。即使有 20% 的點是異常值，該網路的準確率也超過 80%。圖 6 右側顯示該網路對點擾動具有魯棒性。

### 5.3. Visualizing PointNet
### 5.3. 可視化 PointNet

In Fig 7, we visualize *critical point sets* $\mathcal{C}_S$ and *upper-bound shapes* $\mathcal{N}_S$ (as discussed in Thm 2) for some sample shapes $S$. The point sets between the two shapes will give exactly the same global shape feature $f(S)$.
在圖 7 中，我們可視化了一些樣本形狀 $S$ 的**關鍵點集** $\mathcal{C}_S$ 和**上界形狀** $\mathcal{N}_S$（如定理 2 所討論）。這兩個形狀之間的點集將給出完全相同的全局形狀特徵 $f(S)$。

We can see clearly from Fig 7 that the *critical point sets* $\mathcal{C}_S$, those contributed to the max pooled feature, summarizes the skeleton of the shape. The *upper-bound shapes* $\mathcal{N}_S$ illustrates the largest possible point cloud that give the same global shape feature $f(S)$ as the input point cloud $S$.
從圖 7 可以清楚地看出，**關鍵點集** $\mathcal{C}_S$（那些對最大池化特徵有貢獻的點）總結了形狀的骨架。**上界形狀** $\mathcal{N}_S$ 說明了與輸入點雲 $S$ 給出相同全局形狀特徵 $f(S)$ 的最大可能點雲。

$\mathcal{C}_S$ and $\mathcal{N}_S$ reflect the robustness of PointNet, meaning that losing some non-critical points does not change the global shape signature $f(S)$ at all.
$\mathcal{C}_S$ 和 $\mathcal{N}_S$ 反映了 PointNet 的魯棒性，這意味著丟失一些非關鍵點根本不會改變全局形狀簽名 $f(S)$。

The $\mathcal{N}_S$ is constructed by forwarding all the points in a edge-length-2 cube through the network and select points $p$ whose point function values $(h_1(p), h_2(p), \dots, h_K(p))$ are no larger than the global shape descriptor.
$\mathcal{N}_S$ 是通過將邊長為 2 的立方體中的所有點通過網路轉發並選擇點函數值 $(h_1(p), h_2(p), \dots, h_K(p))$ 不大於全局形狀描述符的點 $p$ 來構建的。

*Figure 6 Caption:*
**Figure 6. PointNet robustness test.** The metric is overall classification accuracy on ModelNet40 test set. Left: Delete points. Furthest means the original 1024 points are sampled with furthest sampling. Middle: Insertion. Outliers uniformly scattered in the unit sphere. Right: Perturbation. Add Gaussian noise to each point independently.
**圖 6. PointNet 魯棒性測試。** 指標是 ModelNet40 測試集上的整體分類準確率。左：刪除點。「最遠」表示原始 1024 個點是通過最遠採樣進行採樣的。中：插入。異常值均勻散布在單位球體中。右：擾動。獨立地向每個點添加高斯噪聲。

*Figure 7 Caption:*
**Figure 7. Critical points and upper bound shape.** While critical points jointly determine the global shape feature for a given shape, any point cloud that falls between the critical points set and the upper bound shape gives exactly the same feature. We color-code all figures to show the depth information.
**圖 7. 關鍵點和上界形狀。** 雖然關鍵點共同決定了給定形狀的全局形狀特徵，但落在關鍵點集和上界形狀之間的任何點雲都會給出完全相同的特徵。我們對所有圖進行顏色編碼以顯示深度資訊。

### 5.4. Time and Space Complexity Analysis
### 5.4. 時間和空間複雜度分析

Table 6 summarizes space (number of parameters in the network) and time (floating-point operations/sample) complexity of our classification PointNet. We also compare PointNet to a representative set of volumetric and multi-view based architectures in previous works.
表 6 總結了我們分類 PointNet 的空間（網路中的參數數量）和時間（浮點運算/樣本）複雜度。我們還將 PointNet 與以前工作中一組具有代表性的基於體積和多視圖的架構進行了比較。

While MVCNN [23] and Subvolume (3D CNN) [18] achieve high performance, PointNet is orders more efficient in computational cost (measured in FLOPs/sample: $141x$ and $8x$ more efficient, respectively).
雖然 MVCNN [23] 和 Subvolume (3D CNN) [18] 實現了高性能，但 PointNet 在計算成本方面效率高出幾個數量級（以 FLOPs/樣本衡量：分別高出 $141x$ 和 $8x$）。

Besides, PointNet is much more space efficient than MVCNN in terms of #param in the network ($17x$ less parameters). Moreover, PointNet is much more scalable – it’s space and time complexity is $O(N)$ – linear in the number of input points.
此外，PointNet 在網路參數數量方面比 MVCNN 更節省空間（參數少 $17x$）。此外，PointNet 的可擴展性要好得多——其空間和時間複雜度為 $O(N)$——與輸入點數呈線性關係。

However, since convolution dominates computing time, multi-view method’s time complexity grows *squarely* on image resolution and volumetric convolution based method grows *cubically* with the volume size.
然而，由於卷積主導計算時間，多視圖方法的時間複雜度隨圖像解析度呈**平方**增長，而基於體積卷積的方法隨體積大小呈**立方**增長。

Empirically, PointNet is able to process more than one million points per second for point cloud classification (around 1K objects/second) or semantic segmentation (around 2 rooms/second) with a 1080X GPU on TensorFlow, showing great potential for real-time applications.
在實證上，PointNet 能夠在 TensorFlow 上使用 1080X GPU 每秒處理超過一百萬個點進行點雲分類（約 1K 個物體/秒）或語義分割（約 2 個房間/秒），顯示出即時應用的巨大潛力。

*Table 6 Caption:*
**Table 6. Time and space complexity of deep architectures for 3D data classification.** PointNet (vanilla) is the classification PointNet without input and feature transformations. FLOP stands for floating-point operation. The “M” stands for million. Subvolume and MVCNN used pooling on input data from multiple rotations or views, without which they have much inferior performance.
**表 6. 用於 3D 數據分類的深度架構的時間和空間複雜度。** PointNet (vanilla) 是沒有輸入和特徵變換的分類 PointNet。FLOP 代表浮點運算。「M」代表百萬。Subvolume 和 MVCNN 對來自多個旋轉或視圖的輸入數據使用了池化，如果沒有這些，它們的性能會差很多。

| | #參數 | FLOPs/樣本 |
|---|---|---|
| PointNet (vanilla) | 0.8M | 148M |
| PointNet | 3.5M | 440M |
| Subvolume [18] | 16.6M | 3633M |
| MVCNN [23] | 60.0M | 62057M |

## 6. Conclusion
## 6. 結論

In this work, we propose a novel deep neural network *PointNet* that directly consumes point cloud. Our network provides a unified approach to a number of 3D recognition tasks including object classification, part segmentation and semantic segmentation, while obtaining on par or better results than state of the arts on standard benchmarks.
在這項工作中，我們提出了一種新穎的深度神經網路 *PointNet*，它直接處理點雲。我們的網路為包括物體分類、部件分割和語義分割在內的許多 3D 識別任務提供了一種統一的方法，同時在標準基準上獲得了與最先進技術相當或更好的結果。

We also provide theoretical analysis and visualizations towards understanding of our network.
我們還提供了理論分析和可視化，以幫助理解我們的網路。

**Acknowledgement.** The authors gratefully acknowledge the support of a Samsung GRO grant, ONR MURI N00014-13-1-0341 grant, NSF grant IIS-1528025, a Google Focused Research Award, a gift from the Adobe corporation and hardware donations by NVIDIA.
**致謝。** 作者衷心感謝三星 GRO 撥款、ONR MURI N00014-13-1-0341 撥款、NSF 撥款 IIS-1528025、Google 重點研究獎、Adobe 公司的捐贈以及 NVIDIA 的硬體捐贈的支援。

## References
## 參考文獻

[1] I. Armeni, O. Sener, A. R. Zamir, H. Jiang, I. Brilakis, M. Fischer, and S. Savarese. 3d semantic parsing of large-scale indoor spaces. In *Proceedings of the IEEE International Conference on Computer Vision and Pattern Recognition*, 2016.
[2] M. Aubry, U. Schlickewei, and D. Cremers. The wave kernel signature: A quantum mechanical approach to shape analysis. In *Computer Vision Workshops (ICCV Workshops), 2011 IEEE International Conference on*, pages 1626–1633. IEEE, 2011.
[3] M. M. Bronstein and I. Kokkinos. Scale-invariant heat kernel signatures for non-rigid shape recognition. In *Computer Vision and Pattern Recognition (CVPR), 2010 IEEE Conference on*, pages 1704–1711. IEEE, 2010.
[4] J. Bruna, W. Zaremba, A. Szlam, and Y. LeCun. Spectral networks and locally connected networks on graphs. *arXiv preprint arXiv:1312.6203*, 2013.
[5] D.-Y. Chen, X.-P. Tian, Y.-T. Shen, and M. Ouhyoung. On visual similarity based 3d model retrieval. In *Computer graphics forum*, volume 22, pages 223–232. Wiley Online Library, 2003.
[6] Y. Fang, J. Xie, G. Dai, M. Wang, F. Zhu, T. Xu, and E. Wong. 3d deep shape descriptor. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, pages 2319–2328, 2015.
[7] M. Gschwandtner, R. Kwitt, A. Uhl, and W. Pree. BlenSor: Blender Sensor Simulation Toolbox Advances in Visual Computing. volume 6939 of *Lecture Notes in Computer Science*, chapter 20, pages 199–208. Springer Berlin / Heidelberg, Berlin, Heidelberg, 2011.
[8] K. Guo, D. Zou, and X. Chen. 3d mesh labeling via deep convolutional neural networks. *ACM Transactions on Graphics (TOG)*, 35(1):3, 2015.
[9] M. Jaderberg, K. Simonyan, A. Zisserman, et al. Spatial transformer networks. In *NIPS 2015*.
[10] A. E. Johnson and M. Hebert. Using spin images for efficient object recognition in cluttered 3d scenes. *IEEE Transactions on pattern analysis and machine intelligence*, 21(5):433–449, 1999.
[11] M. Kazhdan, T. Funkhouser, and S. Rusinkiewicz. Rotation invariant spherical harmonic representation of 3 d shape descriptors. In *Symposium on geometry processing*, volume 6, pages 156–164, 2003.
[12] Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. Gradientbased learning applied to document recognition. *Proceedings of the IEEE*, 86(11):2278–2324, 1998.
[13] Y. Li, S. Pirk, H. Su, C. R. Qi, and L. J. Guibas. Fpnn: Field probing neural networks for 3d data. *arXiv preprint arXiv:1605.06240*, 2016.
[14] H. Ling and D. W. Jacobs. Shape classification using the inner-distance. *IEEE transactions on pattern analysis and machine intelligence*, 29(2):286–299, 2007.
[15] L. v. d. Maaten and G. Hinton. Visualizing data using t-sne. *Journal of Machine Learning Research*, 9(Nov):2579–2605, 2008.
[16] J. Masci, D. Boscaini, M. Bronstein, and P. Vandergheynst. Geodesic convolutional neural networks on riemannian manifolds. In *Proceedings of the IEEE International Conference on Computer Vision Workshops*, pages 37–45, 2015.
[17] D. Maturana and S. Scherer. Voxnet: A 3d convolutional neural network for real-time object recognition. In *IEEE/RSJ International Conference on Intelligent Robots and Systems*, September 2015.
[18] C. R. Qi, H. Su, M. Nießner, A. Dai, M. Yan, and L. Guibas. Volumetric and multi-view cnns for object classification on 3d data. In *Proc. Computer Vision and Pattern Recognition (CVPR), IEEE*, 2016.
[19] R. B. Rusu, N. Blodow, and M. Beetz. Fast point feature histograms (fpfh) for 3d registration. In *Robotics and Automation, 2009. ICRA’09. IEEE International Conference on*, pages 3212–3217. IEEE, 2009.
[20] R. B. Rusu, N. Blodow, Z. C. Marton, and M. Beetz. Aligning point cloud views using persistent feature histograms. In *2008 IEEE/RSJ International Conference on Intelligent Robots and Systems*, pages 3384–3391. IEEE, 2008.
[21] M. Savva, F. Yu, H. Su, M. Aono, B. Chen, D. Cohen-Or, W. Deng, H. Su, S. Bai, X. Bai, et al. Shrec16 track largescale 3d shape retrieval from shapenet core55.
[22] P. Y. Simard, D. Steinkraus, and J. C. Platt. Best practices for convolutional neural networks applied to visual document analysis. In *ICDAR*, volume 3, pages 958–962, 2003.
[23] H. Su, S. Maji, E. Kalogerakis, and E. G. Learned-Miller. Multi-view convolutional neural networks for 3d shape recognition. In *Proc. ICCV, to appear*, 2015.
[24] J. Sun, M. Ovsjanikov, and L. Guibas. A concise and provably informative multi-scale signature based on heat diffusion. In *Computer graphics forum*, volume 28, pages 1383–1392. Wiley Online Library, 2009.
[25] O. Vinyals, S. Bengio, and M. Kudlur. Order matters: Sequence to sequence for sets. *arXiv preprint arXiv:1511.06391*, 2015.
[26] D. Z. Wang and I. Posner. Voting for voting in online point cloud object detection. *Proceedings of the Robotics: Science and Systems, Rome, Italy*, 1317, 2015.
[27] Z. Wu, R. Shou, Y. Wang, and X. Liu. Interactive shape cosegmentation via label propagation. *Computers & Graphics*, 38:248–254, 2014.
[28] Z. Wu, S. Song, A. Khosla, F. Yu, L. Zhang, X. Tang, and J. Xiao. 3d shapenets: A deep representation for volumetric shapes. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, pages 1912–1920, 2015.
[29] L. Yi, V. G. Kim, D. Ceylan, I.-C. Shen, M. Yan, H. Su, C. Lu, Q. Huang, A. Sheffer, and L. Guibas. A scalable active framework for region annotation in 3d shape collections. *SIGGRAPH Asia*, 2016.

## Supplementary
## 補充材料

### A. Overview
### A. 概述

This document provides additional quantitative results, technical details and more qualitative test examples to the main paper.
本文件提供了主論文的額外定量結果、技術細節和更多定性測試範例。

In Sec B we extend the robustness test to compare PointNet with VoxNet on incomplete input. In Sec C we provide more details on neural network architectures, training parameters and in Sec D we describe our detection pipeline in scenes.
在第 B 節中，我們擴展了魯棒性測試，在不完整輸入上比較 PointNet 和 VoxNet。在第 C 節中，我們提供了關於神經網路架構、訓練參數的更多細節，在第 D 節中，我們描述了場景中的檢測流程。

Then Sec E illustrates more applications of PointNet, while Sec F shows more analysis experiments. Sec G provides a proof for our theory on PointNet. At last, we show more visualization results in Sec H.
然後第 E 節說明了 PointNet 的更多應用，而第 F 節展示了更多分析實驗。第 G 節提供了我們 PointNet 理論的證明。最後，我們在第 H 節展示更多可視化結果。

### B. Comparison between PointNet and VoxNet (Sec 5.2)
### B. PointNet 與 VoxNet 的比較（第 5.2 節）

We extend the experiments in Sec 5.2 Robustness Test to compare PointNet and VoxNet [17] (a representative architecture for volumetric representation) on robustness to missing data in the input point cloud.
我們擴展了第 5.2 節魯棒性測試中的實驗，以比較 PointNet 和 VoxNet [17]（體積表示的代表性架構）對輸入點雲中缺失數據的魯棒性。

Both networks are trained on the same train test split with 1024 number of points as input. For VoxNet we voxelize the point cloud to $32 \times 32 \times 32$ occupancy grids and augment the training data by random rotation around up-axis and jittering.
兩個網路都在相同的訓練測試分割上進行訓練，輸入點數為 1024。對於 VoxNet，我們將點雲體素化為 $32 \times 32 \times 32$ 的佔用網格，並通過繞上軸隨機旋轉和抖動來增強訓練數據。

At test time, input points are randomly dropped out by a certain ratio. As VoxNet is sensitive to rotations, its prediction uses average scores from 12 viewpoints of a point cloud.
在測試時，輸入點按一定比例隨機丟棄。由於 VoxNet 對旋轉敏感，其預測使用來自點雲 12 個視點的平均分數。

As shown in Fig 8, we see that our PointNet is much more robust to missing points. VoxNet’s accuracy dramatically drops when half of the input points are missing, from 86.3% to 46.0% with a 40.3% difference, while our PointNet only has a 3.7% performance drop.
如圖 8 所示，我們看到我們的 PointNet 對缺失點的魯棒性要強得多。當一半的輸入點缺失時，VoxNet 的準確率急劇下降，從 86.3% 降至 46.0%，差異為 40.3%，而我們的 PointNet 僅有 3.7% 的性能下降。

This can be explained by the theoretical analysis and explanation of our PointNet – it is learning to use a collection of *critical points* to summarize the shape, thus it is very robust to missing data.
這可以通過我們 PointNet 的理論分析和解釋來解釋——它正在學習使用一組**關鍵點**來總結形狀，因此它對缺失數據非常魯棒。

*Figure 8 Caption:*
**Figure 8. PointNet v.s. VoxNet [17] on incomplete input data.** Metric is overall classification accurcacy on ModelNet40 test set. Note that VoxNet is using 12 viewpoints averaging while PointNet is using only one view of the point cloud. Evidently PointNet presents much stronger robustness to missing points.
**圖 8. 不完整輸入數據上的 PointNet 與 VoxNet [17]。** 指標是 ModelNet40 測試集上的整體分類準確率。注意 VoxNet 使用 12 個視點平均，而 PointNet 僅使用點雲的一個視圖。顯然 PointNet 對缺失點表現出更強的魯棒性。

### C. Network Architecture and Training Details (Sec 5.1)
### C. 網路架構和訓練細節（第 5.1 節）

**PointNet Classification Network** As the basic architecture is already illustrated in the main paper, here we provides more details on the joint alignment/transformation network and training parameters.
**PointNet 分類網路** 由於基本架構已在主論文中說明，這裡我們提供關於聯合對齊/變換網路和訓練參數的更多細節。

The first transformation network is a mini-PointNet that takes raw point cloud as input and regresses to a $3 \times 3$ matrix. It’s composed of a shared $MLP(64, 128, 1024)$ network (with layer output sizes 64, 128, 1024) on each point, a max pooling across points and two fully connected layers with output sizes 512, 256.
第一個變換網路是一個微型 PointNet，它將原始點雲作為輸入並回歸到一個 $3 \times 3$ 矩陣。它由每個點上的共享 $MLP(64, 128, 1024)$ 網路（層輸出大小為 64, 128, 1024）、跨點的最大池化和兩個輸出大小為 512, 256 的全連接層組成。

The output matrix is initialized as an identity matrix. All layers, except the last one, include ReLU and batch normalization. The second transformation network has the same architecture as the first one except that the output is a $64 \times 64$ matrix.
輸出矩陣初始化為單位矩陣。除最後一層外，所有層都包含 ReLU 和批次歸一化。第二個變換網路具有與第一個相同的架構，只是輸出為 $64 \times 64$ 矩陣。

The matrix is also initialized as an identity. A regularization loss (with weight 0.001) is added to the softmax classification loss to make the matrix close to orthogonal.
矩陣也初始化為單位矩陣。正則化損失（權重為 0.001）被添加到 softmax 分類損失中，以使矩陣接近正交。

We use dropout with keep ratio 0.7 on the last fully connected layer, whose output dimension 256, before class score prediction. The decay rate for batch normalization starts with 0.5 and is gradually increased to 0.99.
我們在最後一個全連接層（輸出維度 256，在類別分數預測之前）使用保持率為 0.7 的 dropout。批次歸一化的衰減率從 0.5 開始，並逐漸增加到 0.99。

We use adam optimizer with initial learning rate 0.001, momentum 0.9 and batch size 32. The learning rate is divided by 2 every 20 epochs. Training on ModelNet takes 3-6 hours to converge with TensorFlow and a GTX1080 GPU.
我們使用 adam 優化器，初始學習率為 0.001，動量為 0.9，批次大小為 32。學習率每 20 個 epoch 除以 2。在 TensorFlow 和 GTX1080 GPU 上訓練 ModelNet 需要 3-6 小時才能收斂。

**PointNet Segmentation Network** The segmentation network is an extension to the classification PointNet. Local point features (the output after the second transformation network) and global feature (output of the max pooling) are concatenated for each point. No dropout is used for segmentation network. Training parameters are the same as the classification network.
**PointNet 分割網路** 分割網路是分類 PointNet 的擴展。局部點特徵（第二個變換網路後的輸出）和全局特徵（最大池化的輸出）對每個點進行串聯。分割網路不使用 dropout。訓練參數與分類網路相同。

As to the task of shape part segmentation, we made a few modifications to the basic segmentation network architecture (Fig 2 in main paper) in order to achieve best performance, as illustrated in Fig 9.
關於形狀部件分割任務，我們對基本分割網路架構（主論文中的圖 2）進行了一些修改，以獲得最佳性能，如圖 9 所示。

We add a one-hot vector indicating the class of the input and concatenate it with the max pooling layer’s output. We also increase neurons in some layers and add skip links to collect local point features in different layers and concatenate them to form point feature input to the segmentation network.
我們添加了一個指示輸入類別的 one-hot 向量，並將其與最大池化層的輸出串聯。我們還增加了一些層中的神經元，並添加跳躍連接以收集不同層中的局部點特徵，並將它們串聯起來形成分割網路的點特徵輸入。

While [27] and [29] deal with each object category independently, due to the lack of training data for some categories (the total number of shapes for all the categories in the data set are shown in the first line), we train our PointNet across categories (but with one-hot vector input to indicate category).
雖然 [27] 和 [29] 獨立處理每個物體類別，但由於某些類別缺乏訓練數據（數據集中所有類別的形狀總數顯示在第一行），我們跨類別訓練我們的 PointNet（但使用 one-hot 向量輸入來指示類別）。

To allow fair comparison, when testing these two models, we only predict part labels for the given specific object category.
為了進行公平比較，在測試這兩個模型時，我們僅預測給定特定物體類別的部件標籤。

As to semantic segmentation task, we used the architecture as in Fig 2 in the main paper.
至於語義分割任務，我們使用了主論文中圖 2 所示的架構。

It takes around six to twelve hours to train the model on ShapeNet part dataset and around half a day to train on the Stanford semantic parsing dataset.
在 ShapeNet 部件數據集上訓練模型大約需要 6 到 12 小時，在 Stanford 語義解析數據集上訓練大約需要半天。

*Figure 9 Caption:*
**Figure 9. Network architecture for part segmentation.** T1 and T2 are alignment/transformation networks for input points and features. FC is fully connected layer operating on each point. MLP is multi-layer perceptron on each point. One-hot is a vector of size 16 indicating category of the input shape.
**圖 9. 部件分割的網路架構。** T1 和 T2 是輸入點和特徵的對齊/變換網路。FC 是在每個點上運行的全連接層。MLP 是每個點上的多層感知器。One-hot 是一個大小為 16 的向量，指示輸入形狀的類別。

**Baseline 3D CNN Segmentation Network** In ShapeNet part segmentation experiment, we compare our proposed segmentation version PointNet to two traditional methods as well as a 3D volumetric CNN network baseline. In Fig 10, we show the baseline 3D volumetric CNN network we use.
**基線 3D CNN 分割網路** 在 ShapeNet 部件分割實驗中，我們將我們提出的分割版 PointNet 與兩種傳統方法以及 3D 體積 CNN 網路基線進行了比較。在圖 10 中，我們展示了我們使用的基線 3D 體積 CNN 網路。

We generalize the well-known 3D CNN architectures, such as VoxNet [17] and 3DShapeNets [28] to a fully convolutional 3D CNN segmentation network.
我們將著名的 3D CNN 架構（如 VoxNet [17] 和 3DShapeNets [28]）推廣為全卷積 3D CNN 分割網路。

For a given point cloud, we first convert it to the volumetric representation as a occupancy grid with resolution $32 \times 32 \times 32$. Then, five 3D convolution operations each with 32 output channels and stride of 1 are sequentially applied to extract features.
對於給定的點雲，我們首先將其轉換為解析度為 $32 \times 32 \times 32$ 的佔用網格的體積表示。然後，依次應用五個 3D 卷積操作（每個具有 32 個輸出通道和 1 的步長）來提取特徵。

The receptive field is 19 for each voxel. Finally, a sequence of 3D convolutional layers with kernel size $1 \times 1 \times 1$ is appended to the computed feature map to predict segmentation label for each voxel. ReLU and batch normalization are used for all layers except the last one.
每個體素的感受野為 19。最後，將內核大小為 $1 \times 1 \times 1$ 的 3D 卷積層序列附加到計算出的特徵圖，以預測每個體素的分割標籤。除最後一層外，所有層都使用 ReLU 和批次歸一化。

The network is trained across categories, however, in order to compare with other baseline methods where object category is given, we only consider output scores in the given object category.
網路是跨類別訓練的，但是，為了與給定物體類別的其他基線方法進行比較，我們僅考慮給定物體類別中的輸出分數。

*Figure 10 Caption:*
**Figure 10. Baseline 3D CNN segmentation network.** The network is fully convolutional and predicts part scores for each voxel.
**圖 10. 基線 3D CNN 分割網路。** 該網路是全卷積的，並預測每個體素的部件分數。

### D. Details on Detection Pipeline (Sec 5.1)
### D. 檢測流程詳情（第 5.1 節）

We build a simple 3D object detection system based on the semantic segmentation results and our object classification PointNet.
我們基於語義分割結果和我們的物體分類 PointNet 構建了一個簡單的 3D 物體檢測系統。

We use connected component with segmentation scores to get object proposals in scenes. Starting from a random point in the scene, we find its predicted label and use BFS to search nearby points with the same label, with a search radius of 0.2 meter.
我們使用帶有分割分數的連通分量來獲取場景中的物體提議。從場景中的一個隨機點開始，我們找到它的預測標籤，並使用 BFS 搜索具有相同標籤的附近點，搜索半徑為 0.2 米。

If the resulted cluster has more than 200 points (assuming a 4096 point sample in a 1m by 1m area), the cluster’s bounding box is marked as one object proposal. For each proposed object, it’s detection score is computed as the average point score for that category.
如果結果聚類超過 200 個點（假設在 1m x 1m 區域中有 4096 個點樣本），則該聚類的邊界框被標記為一個物體提議。對於每個提議的物體，其檢測分數計算為該類別的平均點分數。

Before evaluation, proposals with extremely small areas/volumes are pruned. For tables, chairs and sofas, the bounding boxes are extended to the floor in case the legs are separated with the seat/surface.
在評估之前，具有極小面積/體積的提議將被修剪。對於桌子、椅子和沙發，邊界框延伸到地板，以防腿部與座椅/表面分離。

We observe that in some rooms such as auditoriums lots of objects (e.g. chairs) are close to each other, where connected component would fail to correctly segment out individual ones. Therefore we leverage our classification network and uses sliding shape method to alleviate the problem for the chair class.
我們觀察到，在禮堂等一些房間中，許多物體（例如椅子）彼此靠近，連通分量將無法正確分割出單個物體。因此，我們利用我們的分類網路並使用滑動形狀方法來緩解椅子類別的問題。

We train a binary classification network for each category and use the classifier for sliding window detection. The resulted boxes are pruned by non-maximum suppression. The proposed boxes from connected component and sliding shapes are combined for final evaluation.
我們為每個類別訓練一個二元分類網路，並使用該分類器進行滑動窗口檢測。結果框通過非極大值抑制進行修剪。來自連通分量和滑動形狀的提議框被組合用於最終評估。

In Fig 11, we show the precision-recall curves for object detection. We trained six models, where each one of them is trained on five areas and tested on the left area. At test phase, each model is tested on the area it has never seen. The test results for all six areas are aggregated for the PR curve generation.
在圖 11 中，我們展示了物體檢測的精確率-召回率曲線。我們訓練了六個模型，其中每個模型都在五個區域上進行訓練，並在剩餘的一個區域上進行測試。在測試階段，每個模型都在其未見過的區域上進行測試。所有六個區域的測試結果被匯總以生成 PR 曲線。

*Figure 11 Caption:*
**Figure 11. Precision-recall curves for object detection in 3D point cloud.** We evaluated on all six areas for four categories: table, chair, sofa and board. IoU threshold is 0.5 in volume.
**圖 11. 3D 點雲中物體檢測的精確率-召回率曲線。** 我們評估了四個類別的所有六個區域：桌、椅、沙發和板。體積 IoU 閾值為 0.5。

### E. More Applications (Sec 5.1)
### E. 更多應用（第 5.1 節）

**Model Retrieval from Point Cloud** Our PointNet learns a global shape signature for every given input point cloud. We expect geometrically similar shapes have similar global signature. In this section, we test our conjecture on the shape retrieval application.
**從點雲檢索模型** 我們的 PointNet 為每個給定的輸入點雲學習一個全局形狀簽名。我們期望幾何相似的形狀具有相似的全局簽名。在本節中，我們在形狀檢索應用上測試我們的猜想。

To be more specific, for every given query shape from ModelNet test split, we compute its global signature (output of the layer before the score prediction layer) given by our classification PointNet and retrieve similar shapes in the train split by nearest neighbor search. Results are shown in Fig 12.
具體來說，對於來自 ModelNet 測試分割的每個給定查詢形狀，我們計算由我們的分類 PointNet 給出的全局簽名（分數預測層之前的層的輸出），並通過最近鄰搜索在訓練分割中檢索相似形狀。結果如圖 12 所示。

*Figure 12 Caption:*
**Figure 12. Model retrieval from point cloud.** For every given point cloud, we retrieve the top-5 similar shapes from the ModelNet test split. From top to bottom rows, we show examples of chair, plant, nightstand and bathtub queries. Retrieved results that are in wrong category are marked by red boxes.
**圖 12. 從點雲檢索模型。** 對於每個給定的點雲，我們從 ModelNet 測試分割中檢索前 5 個相似形狀。從上到下，我們展示了椅子、植物、床頭櫃和浴缸查詢的範例。錯誤類別的檢索結果用紅框標記。

**Shape Correspondence** In this section, we show that point features learnt by PointNet can be potentially used to compute shape correspondences. Given two shapes, we compute the correspondence between their *critical point sets* $\mathcal{C}_S$’s by matching the pairs of points that activate the same dimensions in the global features.
**形狀對應** 在本節中，我們展示 PointNet 學習的點特徵可以潛在地位於計算形狀對應。給定兩個形狀，我們通過匹配在全局特徵中激活相同維度的點對來計算其**關鍵點集** $\mathcal{C}_S$ 之間的對應關係。

Fig 13 and Fig 14 show the detected shape correspondence between two similar chairs and tables.
圖 13 和圖 14 顯示了檢測到的兩個相似椅子和桌子之間的形狀對應關係。

*Figure 13 Caption:*
**Figure 13. Shape correspondence between two chairs.** For the clarity of the visualization, we only show 20 randomly picked correspondence pairs.
**圖 13. 兩把椅子之間的形狀對應。** 為了可視化的清晰度，我們僅顯示 20 個隨機挑選的對應對。

*Figure 14 Caption:*
**Figure 14. Shape correspondence between two tables.** For the clarity of the visualization, we only show 20 randomly picked correspondence pairs.
**圖 14. 兩張桌子之間的形狀對應。** 為了可視化的清晰度，我們僅顯示 20 個隨機挑選的對應對。

### F. More Architecture Analysis (Sec 5.2)
### F. 更多架構分析（第 5.2 節）

**Effects of Bottleneck Dimension and Number of Input Points** Here we show our model’s performance change with regard to the size of the first max layer output as well as the number of input points. In Fig 15 we see that performance grows as we increase the number of points however it saturates at around 1K points.
**瓶頸維度和輸入點數的影響** 在這裡，我們展示了模型性能隨第一個最大層輸出大小以及輸入點數的變化。在圖 15 中，我們看到性能隨著點數的增加而增長，但在約 1K 點時飽和。

The max layer size plays an important role, increasing the layer size from 64 to 1024 results in a 2−4% performance gain. It indicates that we need enough point feature functions to cover the 3D space in order to discriminate different shapes.
最大層大小起著重要作用，將層大小從 64 增加到 1024 會帶來 2-4% 的性能提升。這表明我們需要足夠的點特徵函數來覆蓋 3D 空間，以便區分不同的形狀。

It’s worth notice that even with 64 points as input (obtained from furthest point sampling on meshes), our network can achieve decent performance.
值得注意的是，即使只有 64 個點作為輸入（從網格上的最遠點採樣獲得），我們的網路也能達到不錯的性能。

*Figure 15 Caption:*
**Figure 15. Effects of bottleneck size and number of input points.** The metric is overall classification accuracy on ModelNet40 test set.
**圖 15. 瓶頸大小和輸入點數的影響。** 指標是 ModelNet40 測試集上的整體分類準確率。

**MNIST Digit Classification** While we focus on 3D point cloud learning, a sanity check experiment is to apply our network on a 2D point clouds - pixel sets.
**MNIST 數字分類** 雖然我們專注於 3D 點雲學習，但一個完整性檢查實驗是將我們的網路應用於 2D 點雲 - 像素集。

To convert an MNIST image into a 2D point set we threshold pixel values and add the pixel (represented as a point with $(x, y)$ coordinate in the image) with values larger than 128 to the set. We use a set size of 256. If there are more than 256 pixels int he set, we randomly sub-sample it; if there are less, we pad the set with the one of the pixels in the set (due to our max operation, which point to use for the padding will not affect outcome).
為了將 MNIST 圖像轉換為 2D 點集，我們對像素值進行閾值處理，並將值大於 128 的像素（表示為圖像中具有 $(x, y)$ 座標的點）添加到集合中。我們使用 256 的集合大小。如果集合中有超過 256 個像素，我們隨機對其進行子採樣；如果較少，我們用集合中的一個像素填充集合（由於我們的最大操作，使用哪個點進行填充不會影響結果）。

As seen in Table 7, we compare with a few baselines including multi-layer perceptron that considers input image as an ordered vector, a RNN that consider input as sequence from pixel (0,0) to pixel (27,27), and a vanilla version CNN.
如表 7 所示，我們與幾個基線進行了比較，包括將輸入圖像視為有序向量的多層感知器，將輸入視為從像素 (0,0) 到像素 (27,27) 的序列的 RNN，以及普通版 CNN。

While the best performing model on MNIST is still well engineered CNNs (achieving less than 0.3% error rate), it’s interesting to see that our PointNet model can achieve reasonable performance by considering image as a 2D point set.
雖然 MNIST 上表現最好的模型仍然是精心設計的 CNN（達到低於 0.3% 的錯誤率），但有趣的是，我們的 PointNet 模型可以通過將圖像視為 2D 點集來實現合理的性能。

*Table 7 Caption:*
**Table 7. MNIST classification results.** We compare with vanilla versions of other deep architectures to show that our network based on point sets input is achieving reasonable performance on this traditional task.
**表 7. MNIST 分類結果。** 我們與其他深度架構的普通版本進行比較，以顯示我們基於點集輸入的網路在這項傳統任務上取得了合理的性能。

| | 輸入 | 錯誤率 (%) |
|---|---|---|
| Multi-layer perceptron [22] | 向量 | 1.60 |
| LeNet5 [12] | 圖像 | 0.80 |
| Ours PointNet | 點集 | **0.78** |

**Normal Estimation** In segmentation version of PointNet, local point features and global feature are concatenated in order to provide context to local points. However, it’s unclear whether the context is learnt through this concatenation. In this experiment, we validate our design by showing that our segmentation network can be trained to predict point normals, a local geometric property that is determined by a point’s neighborhood.
**法線估計** 在 PointNet 的分割版本中，局部點特徵和全局特徵被串聯起來，以便為局部點提供上下文。然而，尚不清楚上下文是否是通過這種串聯學習的。在這個實驗中，我們通過展示我們的分割網路可以被訓練來預測點法線（由點的鄰域決定的局部幾何屬性）來驗證我們的設計。

We train a modified version of our segmentation PointNet in a supervised manner to regress to the groundtruth point normals. We just change the last layer of our segmentation PointNet to predict normal vector for each point. We use absolute value of cosine distance as loss.
我們以監督方式訓練分割 PointNet 的修改版本，以回歸到地面實況點法線。我們只是更改分割 PointNet 的最後一層來預測每個點的法線向量。我們使用餘弦距離的絕對值作為損失。

Fig. 16 compares our PointNet normal prediction results (the left columns) to the ground-truth normals computed from the mesh (the right columns). We observe a reasonable normal reconstruction. Our predictions are more smooth and continuous than the ground-truth which includes flipped normal directions in some region.
圖 16 將我們的 PointNet 法線預測結果（左欄）與從網格計算的地面實況法線（右欄）進行了比較。我們觀察到合理的法線重建。我們的預測比包含某些區域翻轉法線方向的地面實況更加平滑和連續。

*Figure 16 Caption:*
**Figure 16. PointNet normal reconstrution results.** In this figure, we show the reconstructed normals for all the points in some sample point clouds and the ground-truth normals computed on the mesh.
**圖 16. PointNet 法線重建結果。** 在此圖中，我們展示了一些樣本點雲中所有點的重建法線以及在網格上計算的地面實況法線。

**Segmentation Robustness** As discussed in Sec 5.2 and Sec B, our PointNet is less sensitive to data corruption and missing points for classification tasks since the global shape feature is extracted from a collection of *critical points* from the given input point cloud.
**分割魯棒性** 如第 5.2 節和第 B 節所討論，我們的 PointNet 對分類任務的數據損壞和缺失點不太敏感，因為全局形狀特徵是從給定輸入點雲的**關鍵點**集合中提取的。

In this section, we show that the robustness holds for segmentation tasks too. The per-point part labels are predicted based on the combination of per-point features and the learnt global shape feature.
在本節中，我們展示魯棒性也適用於分割任務。逐點部件標籤是基於逐點特徵和學習到的全局形狀特徵的組合預測的。

In Fig 17, we illustrate the segmentation results for the given input point clouds $S$ (the left-most column), the *critical point sets* $\mathcal{C}_S$ (the middle column) and the *upper-bound shapes* $\mathcal{N}_S$.
在圖 17 中，我們說明了給定輸入點雲 $S$（最左列）、**關鍵點集** $\mathcal{C}_S$（中間列）和**上界形狀** $\mathcal{N}_S$ 的分割結果。

*Figure 17 Caption:*
**Figure 17. The consistency of segmentation results.** We illustrate the segmentation results for some sample given point clouds $S$, their *critical point sets* $\mathcal{C}_S$ and *upper-bound shapes* $\mathcal{N}_S$. We observe that the shape family between the $\mathcal{C}_S$ and $\mathcal{N}_S$ share a consistent segmentation results.
**圖 17. 分割結果的一致性。** 我們展示了一些給定點雲 $S$、其**關鍵點集** $\mathcal{C}_S$ 和**上界形狀** $\mathcal{N}_S$ 的分割結果。我們觀察到 $\mathcal{C}_S$ 和 $\mathcal{N}_S$ 之間的形狀族共享一致的分割結果。

**Network Generalizability to Unseen Shape Categories** In Fig 18, we visualize the *critical point sets* and the *upper-bound shapes* for new shapes from unseen categories (face, house, rabbit, teapot) that are not present in ModelNet or ShapeNet.
**網路對未見形狀類別的泛化能力** 在圖 18 中，我們可視化了來自 ModelNet 或 ShapeNet 中不存在的未見類別（人臉、房子、兔子、茶壺）的新形狀的**關鍵點集**和**上界形狀**。

It shows that the learnt per-point functions are generalizable. However, since we train mostly on man-made objects with lots of planar structures, the reconstructed upper-bound shape in novel categories also contain more planar surfaces.
這表明學習到的逐點函數具有泛化能力。然而，由於我們主要在具有大量平面結構的人造物體上進行訓練，因此新類別中重建的上界形狀也包含更多平面表面。

*Figure 18 Caption:*
**Figure 18. The critical point sets and the upper-bound shapes for unseen objects.** We visualize the critical point sets and the upper-bound shapes for teapot, bunny, hand and human body, which are not in the ModelNet or ShapeNet shape repository to test the generalizability of the learnt per-point functions of our PointNet on other unseen objects. The images are color-coded to reflect the depth information.
**圖 18. 未見物體的關鍵點集和上界形狀。** 我們可視化了茶壺、兔子、手和人體的關鍵點集和上界形狀，這些物體不在 ModelNet 或 ShapeNet 形狀庫中，以測試我們 PointNet 學習到的逐點函數在其他未見物體上的泛化能力。圖像經過顏色編碼以反映深度資訊。

### G. Proof of Theorem (Sec 4.3)
### G. 定理證明（第 4.3 節）

Let $\mathcal{X} = \{S : S \subseteq [0, 1] \text{ and } |S| = n\}$.
令 $\mathcal{X} = \{S : S \subseteq [0, 1] \text{ and } |S| = n\}$。

$f : \mathcal{X} \to \mathbb{R}$ is a continuous function on $\mathcal{X}$ w.r.t to Hausdorff distance $d_H(\cdot, \cdot)$ if the following condition is satisfied:
$f : \mathcal{X} \to \mathbb{R}$ 是 $\mathcal{X}$ 上關於豪斯多夫距離 $d_H(\cdot, \cdot)$ 的連續函數，如果滿足以下條件：

$\forall \epsilon > 0, \exists \delta > 0, \text{ for any } S, S' \in \mathcal{X}, \text{ if } d_H(S, S') < \delta, \text{ then } |f(S) - f(S')| < \epsilon.$
$\forall \epsilon > 0, \exists \delta > 0, \text{ 對於任何 } S, S' \in \mathcal{X}, \text{ 如果 } d_H(S, S') < \delta, \text{ 則 } |f(S) - f(S')| < \epsilon.$

We show that $f$ can be approximated arbitrarily by composing a symmetric function and a continuous function.
我們證明 $f$ 可以通過組合對稱函數和連續函數來任意近似。

**Theorem 1.** Suppose $f : \mathcal{X} \to \mathbb{R}$ is a continuous set function w.r.t Hausdorff distance $d_H(\cdot, \cdot)$. $\forall \epsilon > 0, \exists$ a continuous function $h$ and a symmetric function $g(x_1, \dots, x_n) = \gamma \circ \text{MAX}$, where $\gamma$ is a continuous function, $\text{MAX}$ is a vector max operator that takes $n$ vectors as input and returns a new vector of the element-wise maximum, such that for any $S \in \mathcal{X}$,
**定理 1.** 假設 $f : \mathcal{X} \to \mathbb{R}$ 是關於豪斯多夫距離 $d_H(\cdot, \cdot)$ 的連續集合函數。$\forall \epsilon > 0, \exists$ 一個連續函數 $h$ 和一個對稱函數 $g(x_1, \dots, x_n) = \gamma \circ \text{MAX}$，其中 $\gamma$ 是一個連續函數，$\text{MAX}$ 是一個向量最大值運算符，它將 $n$ 個向量作為輸入並返回一個元素級最大值的新向量，使得對於任何 $S \in \mathcal{X}$，

$|f(S) - \gamma(\text{MAX}(h(x_1), \dots, h(x_n)))| < \epsilon$

where $x_1, \dots, x_n$ are the elements of $S$ extracted in certain order,
其中 $x_1, \dots, x_n$ 是按特定順序提取的 $S$ 的元素，

*Proof.* By the continuity of $f$, we take $\delta_\epsilon$ so that $|f(S) - f(S')| < \epsilon$ for any $S, S' \in \mathcal{X}$ if $d_H(S, S') < \delta_\epsilon$.
*證明。* 根據 $f$ 的連續性，我們取 $\delta_\epsilon$ 使得對於任何 $S, S' \in \mathcal{X}$，如果 $d_H(S, S') < \delta_\epsilon$，則 $|f(S) - f(S')| < \epsilon$。

Define $K = \lceil 1/\delta_\epsilon \rceil$, which split $[0, 1]$ into $K$ intervals evenly and define an auxiliary function that maps a point to the left end of the interval it lies in:
定義 $K = \lceil 1/\delta_\epsilon \rceil$，將 $[0, 1]$ 均勻分成 $K$ 個區間，並定義一個輔助函數，將點映射到其所在區間的左端：

$\sigma(x) = \frac{\lfloor Kx \rfloor}{K}$

Let $\tilde{S} = \{\sigma(x) : x \in S\}$, then
令 $\tilde{S} = \{\sigma(x) : x \in S\}$，則

$|f(S) - f(\tilde{S})| < \epsilon$

because $d_H(S, \tilde{S}) < 1/K \le \delta_\epsilon$.
因為 $d_H(S, \tilde{S}) < 1/K \le \delta_\epsilon$。

Let $h_k(x) = e^{-d(x, [\frac{k-1}{K}, \frac{k}{K}])}$ be a soft indicator function where $d(x, I)$ is the point to set (interval) distance. Let $\mathbf{h}(x) = [h_1(x); \dots; h_K(x)]$, then $\mathbf{h} : \mathbb{R} \to \mathbb{R}^K$.
令 $h_k(x) = e^{-d(x, [\frac{k-1}{K}, \frac{k}{K}])}$ 為軟指示函數，其中 $d(x, I)$ 是點到集合（區間）的距離。令 $\mathbf{h}(x) = [h_1(x); \dots; h_K(x)]$，則 $\mathbf{h} : \mathbb{R} \to \mathbb{R}^K$。

Let $v_j(x_1, \dots, x_n) = \max\{\tilde{h}_j(x_1), \dots, \tilde{h}_j(x_n)\}$, indicating the occupancy of the $j$-th interval by points in $S$. Let $\mathbf{v} = [v_1; \dots; v_K]$, then $\mathbf{v} : \underbrace{\mathbb{R} \times \dots \times \mathbb{R}}_{n} \to \{0, 1\}^K$ is a symmetric function, indicating the occupancy of each interval by points in $S$.
令 $v_j(x_1, \dots, x_n) = \max\{\tilde{h}_j(x_1), \dots, \tilde{h}_j(x_n)\}$，表示 $S$ 中點對第 $j$ 個區間的佔用情況。令 $\mathbf{v} = [v_1; \dots; v_K]$，則 $\mathbf{v} : \underbrace{\mathbb{R} \times \dots \times \mathbb{R}}_{n} \to \{0, 1\}^K$ 是一個對稱函數，表示 $S$ 中點對每個區間的佔用情況。

Define $\tau : \{0, 1\}^K \to \mathcal{X}$ as $\tau(\mathbf{v}) = \{\frac{k-1}{K} : v_k \ge 1\}$, which maps the occupancy vector to a set which contains the left end of each occupied interval. It is easy to show:
定義 $\tau : \{0, 1\}^K \to \mathcal{X}$ 為 $\tau(\mathbf{v}) = \{\frac{k-1}{K} : v_k \ge 1\}$，它將佔用向量映射到一個包含每個被佔用區間左端的集合。很容易證明：

$\tau(\mathbf{v}(x_1, \dots, x_n)) \equiv \tilde{S}$

where $x_1, \dots, x_n$ are the elements of $S$ extracted in certain order.
其中 $x_1, \dots, x_n$ 是按特定順序提取的 $S$ 的元素。

Let $\gamma : \mathbb{R}^K \to \mathbb{R}$ be a continuous function such that $\gamma(\mathbf{v}) = f(\tau(\mathbf{v}))$ for $v \in \{0, 1\}^K$. Then,
令 $\gamma : \mathbb{R}^K \to \mathbb{R}$ 是一個連續函數，使得 $\gamma(\mathbf{v}) = f(\tau(\mathbf{v}))$ 對於 $v \in \{0, 1\}^K$。那麼，

$|\gamma(\mathbf{v}(x_1, \dots, x_n)) - f(S)| = |f(\tau(\mathbf{v}(x_1, \dots, x_n))) - f(S)| < \epsilon$

Note that $\gamma(\mathbf{v}(x_1, \dots, x_n))$ can be rewritten as follows:
注意 $\gamma(\mathbf{v}(x_1, \dots, x_n))$ 可以重寫如下：

$\gamma(\mathbf{v}(x_1, \dots, x_n)) = \gamma(\text{MAX}(\mathbf{h}(x_1), \dots, \mathbf{h}(x_n)))$
$= (\gamma \circ \text{MAX})(\mathbf{h}(x_1), \dots, \mathbf{h}(x_n))$

Obviously $\gamma \circ \text{MAX}$ is a symmetric function. $\square$
顯然 $\gamma \circ \text{MAX}$ 是一個對稱函數。 $\square$

Next we give the proof of Theorem 2. We define $\mathbf{u} = \underset{x_i \in S}{\text{MAX}} \{h(x_i)\}$ to be the sub-network of $f$ which maps a point set in $[0, 1]^m$ to a $K$-dimensional vector. The following theorem tells us that small corruptions or extra noise points in the input set is not likely to change the output of our network:
接下來我們給出定理 2 的證明。我們定義 $\mathbf{u} = \underset{x_i \in S}{\text{MAX}} \{h(x_i)\}$ 為 $f$ 的子網路，它將 $[0, 1]^m$ 中的點集映射為 $K$ 維向量。以下定理告訴我們，輸入集中的小損壞或額外噪聲點不太可能改變我們網路的輸出：

**Theorem 2.** Suppose $\mathbf{u} : \mathcal{X} \to \mathbb{R}^K$ such that $\mathbf{u} = \underset{x_i \in S}{\text{MAX}} \{h(x_i)\}$ and $f = \gamma \circ \mathbf{u}$. Then,
**定理 2.** 假設 $\mathbf{u} : \mathcal{X} \to \mathbb{R}^K$ 使得 $\mathbf{u} = \underset{x_i \in S}{\text{MAX}} \{h(x_i)\}$ 且 $f = \gamma \circ \mathbf{u}$。那麼，

(a) $\forall S, \exists \mathcal{C}_S, \mathcal{N}_S \subseteq \mathcal{X}, f(T) = f(S)$ if $\mathcal{C}_S \subseteq T \subseteq \mathcal{N}_S$;
(b) $|\mathcal{C}_S| \le K$

*Proof.* Obviously, $\forall S \in \mathcal{X}, f(S)$ is determined by $\mathbf{u}(S)$. So we only need to prove that $\forall S, \exists \mathcal{C}_S, \mathcal{N}_S \subseteq \mathcal{X}, f(T) = f(S)$ if $\mathcal{C}_S \subseteq T \subseteq \mathcal{N}_S$.
*證明。* 顯然，$\forall S \in \mathcal{X}, f(S)$ 由 $\mathbf{u}(S)$ 決定。所以我們只需要證明 $\forall S, \exists \mathcal{C}_S, \mathcal{N}_S \subseteq \mathcal{X}, f(T) = f(S)$ 如果 $\mathcal{C}_S \subseteq T \subseteq \mathcal{N}_S$。

For the $j$-th dimension as the output of $\mathbf{u}$, there exists at least one $x_j \in \mathcal{X}$ such that $h_j(x_j) = u_j$, where $h_j$ is the $j$-th dimension of the output vector from $h$. Take $\mathcal{C}_S$ as the union of all $x_j$ for $j = 1, \dots, K$. Then, $\mathcal{C}_S$ satisfies the above condition.
對於 $\mathbf{u}$ 輸出的第 $j$ 個維度，至少存在一個 $x_j \in \mathcal{X}$ 使得 $h_j(x_j) = u_j$，其中 $h_j$ 是 $h$ 輸出向量的第 $j$ 個維度。取 $\mathcal{C}_S$ 為所有 $x_j$ ($j = 1, \dots, K$) 的並集。那麼，$\mathcal{C}_S$ 滿足上述條件。

Adding any additional points $x$ such that $h(x) \le \mathbf{u}(S)$ at all dimensions to $\mathcal{C}_S$ does not change $\mathbf{u}$, hence $f$. Therefore, $\mathcal{N}_S$ can be obtained adding the union of all such points to $\mathcal{N}_S$. $\square$
向 $\mathcal{C}_S$ 添加任何滿足 $h(x) \le \mathbf{u}(S)$ 在所有維度上的額外點 $x$ 都不會改變 $\mathbf{u}$，因此也不會改變 $f$。因此，$\mathcal{N}_S$ 可以通過將所有這些點的並集添加到 $\mathcal{N}_S$ 來獲得。 $\square$

### H. More Visualizations
### H. 更多可視化

**Classification Visualization** We use t-SNE[15] to embed point cloud global signature (1024-dim) from our classification PointNet into a 2D space. Fig 20 shows the embedding space of ModelNet 40 test split shapes. Similar shapes are clustered together according to their semantic categories.
**分類可視化** 我們使用 t-SNE[15] 將我們分類 PointNet 的點雲全局簽名（1024 維）嵌入到 2D 空間中。圖 20 顯示了 ModelNet 40 測試分割形狀的嵌入空間。相似的形狀根據其語義類別聚集在一起。

**Segmentation Visualization** We present more segmentation results on both complete CAD models and simulated Kinect partial scans. We also visualize failure cases with error analysis. Fig 21 and Fig 22 show more segmentation results generated on complete CAD models and their simulated Kinect scans. Fig 23 illustrates some failure cases. Please read the caption for the error analysis.
**分割可視化** 我們展示了完整 CAD 模型和模擬 Kinect 部分掃描的更多分割結果。我們還通過錯誤分析可視化了失敗案例。圖 21 和圖 22 顯示了在完整 CAD 模型及其模擬 Kinect 掃描上生成的更多分割結果。圖 23 說明了一些失敗案例。請閱讀標題以了解錯誤分析。

**Scene Semantic Parsing Visualization** We give a visualization of semantic parsing in Fig 24 where we show input point cloud, prediction and ground truth for both semantic segmentation and object detection for two office rooms and one conference room. The area and the rooms are unseen in the training set.
**場景語義解析可視化** 我們在圖 24 中給出了語義解析的可視化，其中我們展示了兩個辦公室和一個會議室的語義分割和物體檢測的輸入點雲、預測和地面實況。該區域和房間在訓練集中未出現過。

**Point Function Visualization** Our classification PointNet computes $K$ (we take $K = 1024$ in this visualization) dimension point features for each point and aggregates all the per-point local features via a max pooling layer into a single $K$-dim vector, which forms the global shape descriptor.
**點函數可視化** 我們的分類 PointNet 為每個點計算 $K$（本可視化中取 $K = 1024$）維點特徵，並通過最大池化層將所有逐點局部特徵聚合為單個 $K$ 維向量，形成全局形狀描述符。

To gain more insights on what the learnt per-point functions $h$’s detect, we visualize the points $p_i$’s that give high per-point function value $f(p_i)$ in Fig 19. This visualization clearly shows that different point functions learn to detect for points in different regions with various shapes scattered in the whole space.
為了更深入地了解學習到的逐點函數 $h$ 檢測到了什麼，我們在圖 19 中可視化了給出高逐點函數值 $f(p_i)$ 的點 $p_i$。這種可視化清楚地表明，不同的點函數學習檢測散布在整個空間中具有各種形狀的不同區域中的點。

*Figure 19 Caption:*
**Figure 19. Point function visualization.** For each per-point function $h$, we calculate the values $h(p)$ for all the points $p$ in a cube of diameter two located at the origin, which spatially covers the unit sphere to which our input shapes are normalized when training our PointNet. In this figure, we visualize all the points $p$ that give $h(p) > 0.5$ with function values color-coded by the brightness of the voxel. We randomly pick 15 point functions and visualize the activation regions for them.
**圖 19. 點函數可視化。** 對於每個逐點函數 $h$，我們計算位於原點直徑為 2 的立方體中所有點 $p$ 的 $h(p)$ 值，該立方體在空間上覆蓋了我們訓練 PointNet 時輸入形狀標準化的單位球體。在此圖中，我們可視化了所有給出 $h(p) > 0.5$ 的點 $p$，函數值由體素的亮度進行顏色編碼。我們隨機選取 15 個點函數並可視化它們的激活區域。

*Figure 20 Caption:*
**Figure 20. 2D embedding of learnt shape global features.** We use t-SNE technique to visualize the learnt global shape features for the shapes in ModelNet40 test split.
**圖 20. 學習到的形狀全局特徵的 2D 嵌入。** 我們使用 t-SNE 技術來可視化 ModelNet40 測試分割中形狀的學習到的全局形狀特徵。

*Figure 21 Caption:*
**Figure 21. PointNet segmentation results on complete CAD models.**
**圖 21. 完整 CAD 模型上的 PointNet 分割結果。**

*Figure 22 Caption:*
**Figure 22. PointNet segmentation results on simulated Kinect scans.**
**圖 22. 模擬 Kinect 掃描上的 PointNet 分割結果。**

*Figure 23 Caption:*
**Figure 23. PointNet segmentation failure cases.** In this figure, we summarize six types of common errors in our segmentation application. The prediction and the ground-truth segmentations are given in the first and second columns, while the difference maps are computed and shown in the third columns. The red dots correspond to the wrongly labeled points in the given point clouds.
**圖 23. PointNet 分割失敗案例。** 在此圖中，我們總結了我們分割應用中的六種常見錯誤類型。預測和地面實況分割分別在第一和第二列中給出，而差異圖在第三列中計算並顯示。紅點對應於給定點雲中被錯誤標記的點。

(a) illustrates the most common failure cases: the points on the boundary are wrongly labeled. In the examples, the label predictions for the points near the intersections between the table/chair legs and the tops are not accurate. However, most segmentation algorithms suffer from this error.
(a) 說明了最常見的失敗案例：邊界上的點被錯誤標記。在範例中，桌/椅腳與頂部之間交叉點附近的點的標籤預測不準確。然而，大多數分割演算法都存在這種錯誤。

(b) shows the errors on exotic shapes. For examples, the chandelier and the airplane shown in the figure are very rare in the data set.
(b) 顯示了奇異形狀上的錯誤。例如，圖中顯示的吊燈和飛機在數據集中非常罕見。

(c) shows that small parts can be overwritten by nearby large parts. For example, the jet engines for airplanes (yellow in the figure) are mistakenly classified as body (green) or the plane wing (purple).
(c) 顯示小部件可能被附近的大部件覆蓋。例如，飛機的噴氣發動機（圖中黃色）被錯誤地分類為機身（綠色）或機翼（紫色）。

(d) shows the error caused by the inherent ambiguity of shape parts. For example, the two bottoms of the two tables in the figure are classified as table legs and table bases (category *other* in [29]), while ground-truth segmentation is the opposite.
(d) 顯示了由形狀部件固有的模糊性引起的錯誤。例如，圖中兩張桌子的兩個底部被分類為桌腳和桌座（[29] 中的類別*其他*），而地面實況分割則相反。

(e) illustrates the error introduced by the incompleteness of the partial scans. For the two caps in the figure, almost half of the point clouds are missing.
(e) 說明了由部分掃描的不完整性引入的錯誤。對於圖中的兩個帽子，幾乎一半的點雲缺失。

(f) shows the failure cases when some object categories have too less training data to cover enough variety. There are only 54 bags and 39 caps in the whole dataset for the two categories shown here.
(f) 顯示了當某些物體類別的訓練數據太少而無法覆蓋足夠的多樣性時的失敗案例。這裡顯示的兩個類別在整個數據集中只有 54 個包和 39 個帽子。

*Figure 24 Caption:*
**Figure 24. Examples of semantic segmentation and object detection.** First row is input point cloud, where walls and ceiling are hided for clarity. Second and third rows are prediction and ground-truth of semantic segmentation on points, where points belonging to different semantic regions are colored differently (chairs in red, tables in purple, sofa in orange, board in gray, bookcase in green, floors in blue, windows in violet, beam in yellow, column in magenta, doors in khaki and clutters in black). The last two rows are object detection with bounding boxes, where predicted boxes are from connected components based on semantic segmentation prediction.
**圖 24. 語義分割和物體檢測範例。** 第一排是輸入點雲，為了清晰起見，隱藏了牆壁和天花板。第二排和第三排是點上語義分割的預測和地面實況，其中屬於不同語義區域的點顏色不同（椅子紅色，桌子紫色，沙發橙色，板灰色，書櫃綠色，地板藍色，窗戶紫色，梁黃色，柱子洋紅色，門卡其色，雜物黑色）。最後兩排是帶有邊界框的物體檢測，其中預測框來自基於語義分割預測的連通分量。
