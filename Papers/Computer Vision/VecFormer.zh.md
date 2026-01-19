---
title: "VecFormer"
field: "Papers"
status: "Imported"
created_date: 2026-01-19
pdf_link: "[[VecFormer.pdf]]"
tags: [paper, Papers]
---

# Point or Line? Using Line-based Representation for Panoptic Symbol Spotting in CAD Drawings
# 點還是線？使用基於線的表示法進行 CAD 圖紙中的全景符號識別

**Xingguang Wei, Haomin Wang, Shenglong Ye, Ruifeng Luo, Yanting Zhang, Lixin Gu, Jifeng Dai, Yu Qiao, Wenhai Wang, Hongjie Zhang**
**魏星光, 王昊敏, 葉盛龍, 羅瑞豐, 張艷婷, 顧立信, 戴季峰, 喬宇, 王文海, 張紅傑**

1University of Science and Technology of China 2Shanghai AI Laboratory
3Shanghai Jiao Tong University 4Tongji University 5Donghua University
6Tsinghua University 7The Chinese University of Hong Kong
8Arcplus East China Architectural Design & Research Institute Co., Ltd.
1中國科學技術大學 2上海人工智能實驗室
3上海交通大學 4同濟大學 5東華大學
6清華大學 7香港中文大學
8華東建築設計研究院有限公司

The code is available at https://github.com/WesKwong/VecFormer
代碼可在 https://github.com/WesKwong/VecFormer 獲取

## Abstract
## 摘要

We study the task of panoptic symbol spotting, which involves identifying both individual instances of countable *things* and the semantic regions of uncountable *stuff* in computer-aided design (CAD) drawings composed of vector graphical primitives.
我們研究了全景符號識別（panoptic symbol spotting）任務，該任務涉及識別由矢量圖形圖元組成的計算機輔助設計（CAD）圖紙中的可數*物體（things）*的個體實例以及不可數*材質（stuff）*的語義區域。

Existing methods typically rely on image rasterization, graph construction, or point-based representation, but these approaches often suffer from high computational costs, limited generality, and loss of geometric structural information.
現有方法通常依賴於圖像光柵化、圖結構構建或基於點的表示，但這些方法往往面臨計算成本高、通用性有限以及幾何結構信息丟失的問題。

In this paper, we propose *VecFormer*, a novel method that addresses these challenges through *line-based representation* of primitives.
在本文中，我們提出了 *VecFormer*，這是一種通過圖元的*基於線的表示（line-based representation）*來解決這些挑戰的新穎方法。

This design preserves the geometric continuity of the original primitive, enabling more accurate shape representation while maintaining a computation-friendly structure, making it well-suited for vector graphic understanding tasks.
這種設計保留了原始圖元的幾何連續性，在保持對計算友好的結構的同時，實現了更精確的形狀表示，使其非常適合矢量圖形理解任務。

To further enhance prediction reliability, we introduce a *Branch Fusion Refinement* module that effectively integrates instance and semantic predictions, resolving their inconsistencies for more coherent panoptic outputs.
為了進一步提高預測的可靠性，我們引入了一個*分支融合優化（Branch Fusion Refinement）*模塊，該模塊有效地整合了實例預測和語義預測，解決了它們之間的不一致性，從而產生更連貫的全景輸出。

Extensive experiments demonstrate that our method establishes a new state-of-the-art, achieving 91.1 PQ, with Stuff-PQ improved by 9.6 and 21.2 points over the second-best results under settings with and without prior information, respectively—highlighting the strong potential of line-based representation as a foundation for vector graphic understanding.
大量的實驗表明，我們的方法建立了一個新的最先進水平（SOTA），實現了 91.1 的 PQ，在有和沒有先驗信息的設置下，Stuff-PQ 分別比第二好的結果提高了 9.6 和 21.2 個百分點——突顯了基於線的表示作為矢量圖形理解基礎的巨大潛力。

## 1 Introduction
## 1 引言

Panoptic symbol spotting refers to the task of detecting and classifying all symbols within a CAD drawing, including both countable object instances (e.g., windows, doors, furniture) and uncountable *stuff* regions (e.g., walls, railings) [1, 2, 3].
全景符號識別是指檢測和分類 CAD 圖紙內所有符號的任務，包括可數的對象實例（例如窗戶、門、家具）和不可數的*材質*區域（例如牆壁、欄杆）[1, 2, 3]。

This capability is crucial in CAD-based applications, serving as a foundation for automated design review and for generating 3D Building Information Models (BIM).
這種能力在基於 CAD 的應用中至關重要，是自動化設計審查和生成 3D 建築信息模型（BIM）的基礎。

However, spotting each symbol, which typically comprises a group of graphical primitives, remains highly challenging due to factors such as occlusion, clutter, appearance variations, and severe class imbalance across different symbol categories.
然而，識別通常由一組圖形圖元組成的每個符號仍然極具挑戰性，這是由於遮擋、雜亂、外觀變化以及不同符號類別之間嚴重的類別不平衡等因素造成的。

Earlier approaches to this problem either rasterize CAD drawings and apply image-based detection or segmentation methods [1, 4], or directly construct graph representations of CAD drawings and leverage GNN-based techniques [5, 6, 7].
解決此問題的早期方法要麼將 CAD 圖紙光柵化並應用基於圖像的檢測或分割方法 [1, 4]，要麼直接構建 CAD 圖紙的圖表示並利用基於 GNN 的技術 [5, 6, 7]。

However, both paradigms incur substantial computational costs, particularly when applied to large-scale CAD drawings.
然而，這兩種範式都會產生巨大的計算成本，特別是在應用於大規模 CAD 圖紙時。

To better handle primitive-level data, recent methods treat CAD drawings as sets of points corresponding to graphical primitives and leverage point cloud analysis for symbol spotting.
為了更好地處理圖元級數據，最近的方法將 CAD 圖紙視為對應於圖形圖元的點集，並利用點雲分析進行符號識別。

For example, SymPoint [8] represents each primitive as a point with handcrafted features, encoding attributes such as primitive type and length.
例如，SymPoint [8] 將每個圖元表示為具有手工特徵的點，編碼了諸如圖元類型和長度等屬性。

However, this manually defined representation is restricted to four predefined primitive types (line, arc, circle, and ellipse) and struggles to accommodate the more complex and diverse shapes frequently encountered in real-world CAD drawings.
然而，這種人工定義的表示僅限於四種預定義的圖元類型（線、圓弧、圓和橢圓），難以適應現實世界 CAD 圖紙中經常遇到的更複雜和多樣的形狀。

In contrast, the recent CADSpotting [9] forgoes explicit primitive types by densely sampling points along each primitive and representing each point using only its coordinate and color.
相比之下，最近的 CADSpotting [9] 放棄了顯式的圖元類型，通過沿每個圖元密集採樣點，並僅使用坐標和顏色來表示每個點。

Although this design eliminates reliance on primitive types, it lacks geometric structure and primitive-level awareness, which may hinder the model’s ability to delineate symbol boundaries, resolve overlapping symbols, and capture structural configurations essential for accurate symbol spotting.
雖然這種設計消除了對圖元類型的依賴，但它缺乏幾何結構和圖元級感知，這可能會阻礙模型描繪符號邊界、解決重疊符號以及捕獲對準確符號識別至關重要的結構配置的能力。

In this work, we propose *VecFormer*, a Transformer-based [10] model built on a *line-based representation* that serves as an expressive and type-agnostic formulation for vector graphical primitives.
在這項工作中，我們提出了 *VecFormer*，這是一個基於 Transformer [10] 的模型，建立在*基於線的表示*之上，該表示作為矢量圖形圖元的一種富有表現力且與類型無關的公式。

It employs line sampling to generate a sequence of line segments along each primitive, with each line represented by its intrinsic geometric attributes and associated primitive-level statistics, forming a compact and informative feature set.
它採用線採樣沿每個圖元生成一系列線段，每條線由其內在的幾何屬性和相關的圖元級統計數據表示，形成一個緊湊且信息豐富的特徵集。

Figure 1 illustrates a visual comparison of different primitive representations.
圖 1 展示了不同圖元表示的視覺比較。

SymPoint [8] encodes each primitive as a single point, which is too coarse to capture the fine-grained structures, especially for long primitives commonly found in stuff regions, leading to degraded performance.
SymPoint [8] 將每個圖元編碼為單個點，這對於捕獲細粒度結構來說過於粗糙，特別是對於在材質區域中常見的長圖元，導致性能下降。

To ensure a fair comparison, we adopt the same sampling density across sampling-based methods.
為了確保公平比較，我們在基於採樣的方法中採用相同的採樣密度。

As shown in Figure 1, unlike CADSpotting [9] which suffers from blurred symbol boundaries, our line-based VecFormer yields results with clearer structure and better alignment to ground-truth, demonstrating higher geometric and structural fidelity.
如圖 1 所示，與 CADSpotting [9] 遭受模糊的符號邊界不同，我們基於線的 VecFormer 產生的結果具有更清晰的結構和與真值更好的對齊，展示了更高的幾何和結構保真度。

This more compact yet expressive representation is also better suited for Transformer-based architecture, which is sensitive to input sequence length.
這種更緊湊但更具表現力的表示也更適合對輸入序列長度敏感的 Transformer 架構。

Further discussion on sequence length across different representations is detailed in Appendix C.
關於不同表示的序列長度的進一步討論詳見附錄 C。

Additionally, inspired by OneFormer3D [11], we adopt a dual-branch Transformer decoder to guide the representation learning of vector graphical primitives, leveraging its strong multi-tasking capability to jointly model instance- and semantic-level information.
此外，受 OneFormer3D [11] 的啟發，我們採用雙分支 Transformer 解碼器來指導矢量圖形圖元的表示學習，利用其強大的多任務處理能力來聯合建模實例級和語義級信息。

To produce a more coherent panoptic output, we further propose a lightweight, training-free post-processing module, termed *Branch Fusion Refinement* (BFR), which combines predictions from the instance and semantic branches through confidence-based fusion.
為了產生更連貫的全景輸出，我們進一步提出了一個輕量級、無需訓練的後處理模塊，稱為*分支融合優化*（BFR），它通過基於置信度的融合結合了來自實例分支和語義分支的預測。

This refinement enhances label consistency, mitigates mask fragmentation, and improves the overall coherence of panoptic symbol predictions.
這種優化增強了標籤的一致性，減輕了掩碼碎片化，並提高了全景符號預測的整體連貫性。

To summarize, our main contributions are:
總而言之，我們的主要貢獻是：

(1) We introduce *VecFormer*, a novel approach that leverages a type-agnostic and expressive *line-based representation* of vector graphical primitives, instead of traditional point-based methods, leading to more accurate and efficient panoptic symbol spotting.
(1) 我們介紹了 *VecFormer*，這是一種利用矢量圖形圖元的類型無關且富有表現力的*基於線的表示*的新方法，取代了傳統的基於點的方法，從而實現了更準確和高效的全景符號識別。

(2) We propose a *Branch Fusion Refinement* (BFR) module that effectively integrates instance and semantic predictions via confidence-based fusion, resolving their inconsistencies for more coherent panoptic outputs, yielding a performance gain of approximately 2 points in panoptic quality (PQ) on the FloorPlanCAD [1] dataset.
(2) 我們提出了一個*分支融合優化*（BFR）模塊，通過基於置信度的融合有效地整合實例和語義預測，解決它們之間的不一致性以獲得更連貫的全景輸出，在 FloorPlanCAD [1] 數據集上獲得了約 2 個百分點的全景質量（PQ）性能提升。

(3) We conduct extensive experiments on the FloorPlanCAD [1] dataset, where our *VecFormer* achieves a PQ of 91.1, setting a new state-of-the-art in the panoptic symbol spotting task.
(3) 我們在 FloorPlanCAD [1] 數據集上進行了廣泛的實驗，其中我們的 *VecFormer* 達到了 91.1 的 PQ，在全景符號識別任務中創立了新的最先進水平。

Notably, it improves Stuff-PQ by 9.6 and 21.2 points over the second-best results under settings with and without prior information, respectively, underscoring its superior performance and robustness in real-world CAD applications.
值得注意的是，在有和沒有先驗信息的設置下，它的 Stuff-PQ 分別比第二好的結果提高了 9.6 和 21.2 個百分點，強調了其在現實世界 CAD 應用中的卓越性能和魯棒性。

## 2 Related Work
## 2 相關工作

### 2.1 Panoptic Image Segmentation
### 2.1 全景圖像分割

Panoptic segmentation [12] aims to unify semantic [13, 14, 15, 16, 17] and instance segmentation [18, 19, 20, 21] by assigning each pixel both a class label and an instance ID, effectively covering both *things* (countable objects) and *stuff* (amorphous regions).
全景分割 [12] 旨在通過為每個像素分配類別標籤和實例 ID 來統一語義分割 [13, 14, 15, 16, 17] 和實例分割 [18, 19, 20, 21]，有效地覆蓋*物體*（可數對象）和*材質*（無定形區域）。

Early approaches predominantly relied on CNN-based architectures [22, 23, 24, 25], which, while effective, often required separate branches for different segmentation tasks.
早期方法主要依賴於基於 CNN 的架構 [22, 23, 24, 25]，雖然有效，但通常需要為不同的分割任務設置單獨的分支。

Recent advancements have seen a shift towards Transformer-based models, which offer unified architectures for various segmentation tasks.
最近的進展見證了向基於 Transformer 的模型的轉變，這些模型為各種分割任務提供了統一的架構。

Notably, Mask2Former [26] unifies panoptic, instance, and semantic segmentation using masked attention.
值得注意的是，Mask2Former [26] 使用掩碼注意力統一了全景、實例和語義分割。

SegFormer [17] improves efficiency with hierarchical encoders and lightweight decoders.
SegFormer [17] 通過層次編碼器和輕量級解碼器提高了效率。

OneFormer [20] further introduces task-conditioned training to jointly handle multiple segmentation tasks.
OneFormer [20] 進一步引入了任務條件訓練來聯合處理多個分割任務。

Despite these successes in raster image domains, pixel-centric segmentation models face challenges when applied to vector graphics tasks, such as Panoptic Symbol Spotting in CAD drawings.
儘管在光柵圖像領域取得了這些成功，但以像素為中心的分割模型在應用於矢量圖形任務（如 CAD 圖紙中的全景符號識別）時仍面臨挑戰。

Their reliance on dense pixel grids overlooks the inherent structure of vector primitives, making it difficult to capture precise geometric relationships, maintain topological consistency, and resolve overlapping symbols.
它們對密集像素網格的依賴忽視了矢量圖元的固有結構，使得難以捕獲精確的幾何關係、維持拓撲一致性以及解決重疊符號問題。

These limitations hinder performance in structured, symbol-rich vector environments.
這些限制阻礙了在結構化、符號豐富的矢量環境中的性能。

### 2.2 Panoptic Symbol Spotting
### 2.2 全景符號識別

The panoptic symbol spotting task, first introduced in [1], aims to simultaneously detect and classify architectural symbols (e.g., doors, windows, walls) in floor plan computer-aided design (CAD) drawings.
全景符號識別任務最早在 [1] 中引入，旨在同時檢測和分類平面圖計算機輔助設計（CAD）圖紙中的建築符號（例如門、窗、牆）。

While earlier approaches [2] primarily addressed instances of countable *things* (e.g., windows, doors, tables), Fan *et al.* [1], inspired by [12], extended the task to include semantic regions of uncountable *stuff* (e.g., wall, railing).
雖然早期方法 [2] 主要解決可數*物體*（例如窗、門、桌子）的實例，但 Fan *et al.* [1] 受 [12] 啟發，將任務擴展到包括不可數*材質*（例如牆、欄杆）的語義區域。

To support this task, they introduced the FloorPlanCAD benchmark and proposed PanCADNet as a baseline, which combines Faster R-CNN [27] for detecting countable *things* with Graph Convolutional Networks [28] for segmenting uncountable *stuff*.
為了支持這一任務，他們引入了 FloorPlanCAD 基準並提出了 PanCADNet 作為基線，該基線結合了用於檢測可數*物體*的 Faster R-CNN [27] 和用於分割不可數*材質*的圖卷積網絡 [28]。

Subsequently, Fan *et al.* [4] proposed CADTransformer, utilizing HRNetV2-W48 [29] and Vision Transformers [30] for primitive tokenization and embedding aggregation.
隨後，Fan *et al.* [4] 提出了 CADTransformer，利用 HRNetV2-W48 [29] 和 Vision Transformers [30] 進行圖元標記化和嵌入聚合。

Zheng *et al.* [6] adopted graph-based representations with Graph Attention Networks [31] for instance- and semantic-level predictions.
Zheng *et al.* [6] 採用了基於圖的表示和圖注意力網絡 [31] 進行實例級和語義級預測。

Liu *et al.* [8] introduced SymPoint, exploring point-based representations with handcrafted features, later enhanced by SymPoint-V2 [32] through layer feature encoding and position-guided training.
Liu *et al.* [8] 引入了 SymPoint，探索了具有手工特徵的基於點的表示，後來通過層特徵編碼和位置引導訓練由 SymPoint-V2 [32] 進行了增強。

Recently, CADSpotting [9] densely samples points along primitives to generate dense point data for feature extraction and employs Sliding Window Aggregation for efficient panoptic segmentation of large-scale CAD drawings.
最近，CADSpotting [9] 沿圖元密集採樣點以生成用於特徵提取的密集點數據，並採用滑動窗口聚合對大規模 CAD 圖紙進行高效的全景分割。

Although point-based representations are widely adopted in existing state-of-the-art methods [8, 32, 9], they exhibit notable limitations in complex and densely annotated CAD drawings, including redundant sampling, loss of geometric continuity, and reduced ability to distinguish adjacent or overlapping symbols, as shown in Figure 1.
儘管基於點的表示在現有的最先進方法 [8, 32, 9] 中被廣泛採用，但它們在複雜且密集標註的 CAD 圖紙中表現出明顯的局限性，包括冗餘採樣、幾何連續性喪失以及區分相鄰或重疊符號的能力降低，如圖 1 所示。

## 3 Method
## 3 方法

In this section, we first describe how heterogeneous vector graphic primitives are converted into a unified line-based representation.
在本節中，我們先描述異構矢量圖形圖元如何轉換為統一的基於線的表示。

We then present the panoptic symbol spotting framework built upon this representation.
然後，我們展示建立在這種表示之上的全景符號識別框架。

Finally, we introduce our post-processing optimization strategy, Branch Fusion Refinement. An overview of the entire pipeline is shown in Figure 2.
最後，我們介紹我們的後處理優化策略，分支融合優化。整個流程的概述如圖 2 所示。

### 3.1 Line Sampling
### 3.1 線採樣

Existing point-based representations [8, 32, 9] suffer from limited geometric continuity, structural expressiveness, and generality across diverse primitive types.
現有的基於點的表示 [8, 32, 9] 在幾何連續性、結構表現力和跨多種圖元類型的通用性方面受到限制。

To address these issues, we propose *Line Sampling*, a line-based approximation that encodes primitives as sequences of line segments, enabling unified and geometry-preserving modeling of heterogeneous vector graphics.
為了解決這些問題，我們提出了*線採樣*，這是一種基於線的近似方法，將圖元編碼為線段序列，從而實現對異構矢量圖形的統一且保留幾何特徵的建模。

Specifically, given a vector primitive with a unique identifier $j$, we first convert it into a vector path $\gamma_j(t) : [0, 1] \rightarrow \mathbb{R}^2$.
具體而言，給定一個具有唯一標識符 $j$ 的矢量圖元，我們首先將其轉換為矢量路徑 $\gamma_j(t) : [0, 1] \rightarrow \mathbb{R}^2$。

Then, we perform uniform or dynamic path sampling over its parameter interval to generate a sequence of sampled points $\mathcal{P}_j = \{ \mathbf{p}_i = \gamma_j(t_i) \mid i = 1, \dots, K \}$.
然後，我們在其參數區間上執行均勻或動態路徑採樣，以生成採樣點序列 $\mathcal{P}_j = \{ \mathbf{p}_i = \gamma_j(t_i) \mid i = 1, \dots, K \}$。

Here, $0 = t_1 < t_2 < \dots < t_K = 1$, where the number of samples $K$ and the sampling parameters $t_i$ can be dynamically adjusted based on geometric features such as the length and curvature of the primitive.
這裡，$0 = t_1 < t_2 < \dots < t_K = 1$，其中樣本數量 $K$ 和採樣參數 $t_i$ 可以根據幾何特徵（如圖元的長度和曲率）動態調整。

For simplicity, we adopt a uniform sampling strategy defined as: $t_i = \frac{i-1}{K-1}$, and use a hyperparameter called the *sampling ratio* to control the number of samples $K$.
為簡單起見，我們採用定義為 $t_i = \frac{i-1}{K-1}$ 的均勻採樣策略，並使用稱為*採樣率*的超參數來控制樣本數量 $K$。

Specifically, for line primitives, we initially set $K = 2$; for all other types of primitives, we initially set $K = 9$.
具體來說，對於直線圖元，我們初始設置 $K = 2$；對於所有其他類型的圖元，我們初始設置 $K = 9$。

Given a sampling ratio $\alpha_{\text{sample}}$, we constrain the maximum allowable distance between adjacent sample points to be no greater than $\alpha_{\text{sample}} \cdot \min(\text{width}, \text{height})$, where *width* and *height* denote the dimensions of the CAD drawing.
給定採樣率 $\alpha_{\text{sample}}$，我們將相鄰採樣點之間的最大允許距離限制為不大於 $\alpha_{\text{sample}} \cdot \min(\text{width}, \text{height})$，其中 *width* 和 *height* 表示 CAD 圖紙的尺寸。

If this condition is violated, we iteratively increase the number of samples by setting $K \leftarrow K + 1$ until the constraint is satisfied.
如果違反此條件，我們會迭代增加樣本數量，設置 $K \leftarrow K + 1$，直到滿足約束。

Next, adjacent sampling points are pairwise connected to construct a sequence of line segments:
接下來，相鄰的採樣點成對連接以構建線段序列：
$$ \mathcal{L}_j = \{ \mathbf{l}_i = (\mathbf{p}^s, \mathbf{p}^e) \mid \mathbf{p}^s = \mathbf{p}_i, \mathbf{p}^e = \mathbf{p}_{i+1}, i = 1, \dots, K - 1 \} \quad (1) $$
which approximates the geometric features of the original primitive.
這近似了原始圖元的幾何特徵。

### 3.2 Panoptic Symbol Spotting via Line-based Representation
### 3.2 基於線表示的全景符號識別

The process of panoptic symbol spotting via the line-based representation consists of three main stages: first, using a backbone to extract line-level features; second, pooling the line-level features into primitive-level features; and third, utilizing a 6-layer Transformer decoder to generate instance proposals and semantic predictions.
通過基於線的表示進行全景符號識別的過程包括三個主要階段：第一，使用主幹網絡提取線級特徵；第二，將線級特徵池化為圖元級特徵；第三，利用 6 層 Transformer 解碼器生成實例提議和語義預測。

#### 3.2.1 Backbone
#### 3.2.1 主幹網絡

We choose Point Transformer V3 (PTv3) [33] as our backbone for feature extraction due to its excellent performance in handling unordered data with irregular spatial distributions.
我們選擇 Point Transformer V3 (PTv3) [33] 作為特徵提取的主幹網絡，因為它在處理具有不規則空間分布的無序數據方面表現出色。

Given a sampled line $\mathbf{l}_i$, with its starting point $\mathbf{p}^s = (x_1, y_1)$ and endpoint $\mathbf{p}^e = (x_2, y_2)$, the primitive ID $j$ indicates the primitive to which the line segment belongs, and the layer ID $k$ indicates the layer on the CAD drawing where the primitive is located. we will now describe how to convert it into the position vector $\mathbf{coord}_i \in \mathbb{R}^3$ and the corresponding feature vector $\mathbf{feat}_i \in \mathbb{R}^C$ ($C$ is the dimensionality of the feature vector) suitable for input to the PTv3 backbone.
給定採樣線 $\mathbf{l}_i$，其起點 $\mathbf{p}^s = (x_1, y_1)$，終點 $\mathbf{p}^e = (x_2, y_2)$，圖元 ID $j$ 指示線段所屬的圖元，圖層 ID $k$ 指示圖元所在的 CAD 圖紙圖層。我們現在將描述如何將其轉換為位置向量 $\mathbf{coord}_i \in \mathbb{R}^3$ 和相應的特徵向量 $\mathbf{feat}_i \in \mathbb{R}^C$（$C$ 是特徵向量的維度），以適合作為 PTv3 主幹網絡的輸入。

**Normalization.** The initial step involves the normalization of the raw line features to a standardized range of $[-0.5, 0.5]$.
**歸一化。** 第一步涉及將原始線特徵歸一化到 $[-0.5, 0.5]$ 的標準化範圍。

For the starting point $\mathbf{p}^s = (x_1, y_1)$, the normalization is performed as follows:
對於起點 $\mathbf{p}^s = (x_1, y_1)$，歸一化執行如下：
$$ \mathbf{p}^s = (x_1, y_1) = \left( \frac{x_1 - x_{\text{origin}}}{\text{width}} - 0.5, \frac{y_1 - y_{\text{origin}}}{\text{height}} - 0.5 \right). \quad (2) $$

In this formulation, the coordinates $(x_{\text{origin}}, y_{\text{origin}})$ denote the origin of the coordinate system employed in the CAD drawing.
在此公式中，坐標 $(x_{\text{origin}}, y_{\text{origin}})$ 表示 CAD 圖紙中使用的坐標系原點。

The terms *width* and *height* denote the dimensions of the CAD drawing.
術語 *width* 和 *height* 表示 CAD 圖紙的尺寸。

The normalization for the endpoint $\mathbf{p}^e$ is achieved through an analogous transformation.
終點 $\mathbf{p}^e$ 的歸一化通過類似的變換實現。

For the normalization of layer ID, let $k_{\min}$ and $k_{\max}$ represent the minimum and maximum layer ID values observed within the CAD drawing, respectively. The normalized layer ID $k$ is then calculated as:
對於圖層 ID 的歸一化，令 $k_{\min}$ 和 $k_{\max}$ 分別表示 CAD 圖紙中觀察到的最小和最大圖層 ID 值。歸一化的圖層 ID $k$ 計算如下：
$$ k = \frac{k - k_{\min}}{k_{\max} - k_{\min}} - 0.5. \quad (3) $$

**Line Position.** To simultaneously capture both the position information and the layer information, we use the midpoint of the line $(c_x, c_y)$ for the first two dimensions and the layer ID $k$ for the third dimension:
**線位置。** 為了同時捕獲位置信息和圖層信息，我們使用線的中點 $(c_x, c_y)$ 作為前兩個維度，使用圖層 ID $k$ 作為第三個維度：
$$ \mathbf{coord}_i = (x_i, y_i, z_i) = (c_x, c_y, \text{id}) = \left( \frac{x_1 + x_2}{2}, \frac{y_1 + y_2}{2}, k \right). \quad (4) $$

**Line Feature.** We set the dimensionality $C = 7$, and define the line feature $\mathbf{feat}_i \in \mathbb{R}^7$ as:
**線特徵。** 我們設置維度 $C = 7$，並將線特徵 $\mathbf{feat}_i \in \mathbb{R}^7$ 定義為：
$$ \mathbf{feat}_i = (l, d_x, d_y, c_x, c_y, c_x^p, c_y^p). \quad (5) $$

Here, $l = \sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2}$ represents the length of the line. The terms $d_x = (x_1 - x_2)/l$ and $d_y = (y_1 - y_2)/l$ denote the unit vectors for displacement in the $x$ and $y$ directions, respectively.
這裡，$l = \sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2}$ 表示線的長度。項 $d_x = (x_1 - x_2)/l$ 和 $d_y = (y_1 - y_2)/l$ 分別表示 $x$ 和 $y$ 方向位移的單位向量。

The coordinates $(c_x, c_y)$ specify the midpoint of the line. These features are chosen because any point on the line can be expressed as: $(x, y) = (c_x + t d_x, c_y + t d_y), t \in [-\frac{l}{2}, \frac{l}{2}]$, which provides a parametric representation of the line segment based on its center and unit vector.
坐標 $(c_x, c_y)$ 指定線的中點。選擇這些特徵是因為線上的任何點都可以表示為：$(x, y) = (c_x + t d_x, c_y + t d_y), t \in [-\frac{l}{2}, \frac{l}{2}]$，這提供了基於中心和單位向量的線段參數化表示。

Furthermore, $(c_x^p, c_y^p)$ indicates the geometric centroid of the primitive. This centroid is determined by calculating the average of the midpoint coordinates from all lines sampled within the primitive $j$ to which the specific line belongs:
此外，$(c_x^p, c_y^p)$ 表示圖元的幾何質心。該質心是通過計算特定線所屬的圖元 $j$ 內採樣的所有線的中點坐標的平均值來確定的：
$$ (c_x^p, c_y^p) = \left( \frac{\sum_{\mathbf{l}_i \in \mathcal{L}_j} c_x}{|\mathcal{L}_j|}, \frac{\sum_{\mathbf{l}_i \in \mathcal{L}_j} c_y}{|\mathcal{L}_j|} \right). \quad (6) $$

#### 3.2.2 Line Pooling
#### 3.2.2 線池化

To obtain primitive-level features, we apply *Line Pooling*, which combines max and average pooling over line-level features within each primitive, effectively preserving geometric information and enhancing feature richness.
為了獲得圖元級特徵，我們應用*線池化*，結合每個圖元內線級特徵的最大池化和平均池化，有效地保留幾何信息並增強特徵豐富度。

For each primitive $j$, line features $\mathbf{f}_i \in \mathbb{R}^C$ from $\mathbf{l}_i \in \mathcal{L}_j$ are aggregated via both max and average pooling, whose results are summed to produce the final primitive feature $\mathbf{F}_j \in \mathbb{R}^C$:
對於每個圖元 $j$，來自 $\mathbf{l}_i \in \mathcal{L}_j$ 的線特徵 $\mathbf{f}_i \in \mathbb{R}^C$ 通過最大池化和平均池化進行聚合，其結果相加以產生最終的圖元特徵 $\mathbf{F}_j \in \mathbb{R}^C$：
$$ \mathbf{F}_j = \mathbf{F}_j^{\max} + \mathbf{F}_j^{\text{avg}} = \max_{\mathbf{l}_i \in \mathcal{L}_j} \mathbf{f}_i + \frac{1}{|\mathcal{L}_j|} \sum_{\mathbf{l}_i \in \mathcal{L}_j} \mathbf{f}_i. \quad (7) $$

#### 3.2.3 Layer Feature Enhancement
#### 3.2.3 圖層特徵增強

Inspired by SymPoint-V2 [32], we adopt a *Layer Feature Enhancement* (LFE) module in our method.
受 SymPoint-V2 [32] 的啟發，我們在方法中採用了*圖層特徵增強*（LFE）模塊。

Specifically, we aggregate the features of primitives within the same layer using average pooling, max pooling, and attention pooling, and fuse the resulting layer-level context back into each primitive feature.
具體來說，我們使用平均池化、最大池化和注意力池化聚合同一圖層內圖元的特徵，並將所得的圖層級上下文融合回每個圖元特徵中。

This fusion enhances the model’s ability to capture intra-layer contextual dependencies and improves the semantic discrimination of similar primitives.
這種融合增強了模型捕獲層內上下文依賴關係的能力，並提高了相似圖元的語義辨別力。

#### 3.2.4 Query Decoder
#### 3.2.4 查詢解碼器

Motivated by OneFormer3D [11], we initialize the queries using a *Query Selection* strategy, which is widely adopted in state-of-the-art 2D object detection and instance segmentation methods [34, 35, 36].
受 OneFormer3D [11] 的啟發，我們使用*查詢選擇*策略初始化查詢，該策略在最先進的 2D 目標檢測和實例分割方法 [34, 35, 36] 中被廣泛採用。

Subsequently, a six-layer Transformer decoder performs self-attention on the queries and cross-attention with key-value pairs derived from primitive features.
隨後，一個六層 Transformer 解碼器對查詢執行自注意力，並與源自圖元特徵的鍵值對執行交叉注意力。

The decoder outputs are then passed to an *Instance Branch* for generating instance proposals and a *Semantic Branch* for producing semantic predictions.
然後，解碼器輸出被傳遞給一個*實例分支*用於生成實例提議，以及一個*語義分支*用於生成語義預測。

**Query Selection.** With the primitive features $\mathcal{L} \in \mathbb{R}^{N \times C}$ derived from the previous stage, where $N$ denotes the number of primitives and $C$ is the dimensionality of each feature vector, the Query Selection strategy randomly selects a proportion $\alpha_{\text{select}} \in [0, 1]$ of the primitive features to initialize the queries $Q \in \mathbb{R}^{M \times C}$, with $M = \alpha_{\text{select}} \cdot N$ representing the number of queries.
**查詢選擇。** 對於前一階段導出的圖元特徵 $\mathcal{L} \in \mathbb{R}^{N \times C}$，其中 $N$ 表示圖元數量，$C$ 是每個特徵向量的維度，查詢選擇策略隨機選擇比例為 $\alpha_{\text{select}} \in [0, 1]$ 的圖元特徵來初始化查詢 $Q \in \mathbb{R}^{M \times C}$，其中 $M = \alpha_{\text{select}} \cdot N$ 表示查詢的數量。

Following the configuration in OneFormer3D [11], we set $\alpha_{\text{select}} = 0.5$ during training to reduce computational cost, which also serves as a form of data augmentation.
遵循 OneFormer3D [11] 中的配置，我們在訓練期間設置 $\alpha_{\text{select}} = 0.5$ 以降低計算成本，這也作為一種數據增強形式。

During inference, we set $\alpha_{\text{select}} = 1.0$ in order to preserve the complete information of the CAD drawings.
在推理期間，我們設置 $\alpha_{\text{select}} = 1.0$ 以保留 CAD 圖紙的完整信息。

**Instance Branch.** In this branch, each query embedding is mapped to a $K + 1$ dimensional space as class label logits, where $K$ denotes the number of classes and an extra $+1$ for the background predictions.
**實例分支。** 在此分支中，每個查詢嵌入被映射到 $K + 1$ 維空間作為類別標籤 logits，其中 $K$ 表示類別數量，額外的 $+1$ 用於背景預測。

Simultaneously, we use an einsum operation between the query embedding and the primitive features to generate the instance mask.
同時，我們在查詢嵌入和圖元特徵之間使用 einsum 操作來生成實例掩碼。

**Semantic Branch.** This branch aims to produce dense, per-primitive semantic predictions. We project the output queries from the decoder into a $K + 1$ dimensional space as semantic logits.
**語義分支。** 此分支旨在產生密集的、逐圖元的語義預測。我們將解碼器的輸出查詢投影到 $K + 1$ 維空間作為語義 logits。

The prediction for each query is assigned to the primitive that was selected to initialize the query during the Query Selection process, thereby providing semantic label of each primitive.
每個查詢的預測被分配給在查詢選擇過程中被選中用於初始化該查詢的圖元，從而提供每個圖元的語義標籤。

#### 3.2.5 Loss Function
#### 3.2.5 損失函數

To jointly optimize instance and semantic predictions, we adopt a composite loss function:
為了聯合優化實例和語義預測，我們採用復合損失函數：
$$ L_{\text{total}} = \lambda_{\text{cls}} L_{\text{cls}} + \lambda_{\text{bce}} L_{\text{bce}} + \lambda_{\text{dice}} L_{\text{dice}} + \lambda_{\text{sem}} L_{\text{sem}}. \quad (8) $$

Here, $L_{\text{cls}}$ is a cross-entropy loss for instance classification, $L_{\text{bce}}$ and $L_{\text{dice}}$ [37, 38] are used for instance mask prediction to balance foreground-background accuracy and mask overlap, respectively.
這裡，$L_{\text{cls}}$ 是用於實例分類的交叉熵損失，$L_{\text{bce}}$ 和 $L_{\text{dice}}$ [37, 38] 用於實例掩碼預測，以分別平衡前景-背景準確率和掩碼重疊。

$L_{\text{sem}}$ denotes the cross-entropy loss for semantic segmentation. The weights $\lambda_{\text{cls}}, \lambda_{\text{bce}}, \lambda_{\text{dice}}, \lambda_{\text{sem}}$ control the influence of each term.
$L_{\text{sem}}$ 表示語義分割的交叉熵損失。權重 $\lambda_{\text{cls}}, \lambda_{\text{bce}}, \lambda_{\text{dice}}, \lambda_{\text{sem}}$ 控制每項的影響。

### 3.3 Branch Fusion Refinement
### 3.3 分支融合優化

To effectively integrate information from both the Semantic Branch and the Instance Branch, we propose a post-processing strategy named *Branch Fusion Refinement* (BFR). This method consists of three steps: *Overriding*, *Voting*, and *Remasking*.
為了有效地整合來自語義分支和實例分支的信息，我們提出了一種名為*分支融合優化*（BFR）的後處理策略。該方法包括三個步驟：*覆蓋（Overriding）*、*投票（Voting）*和*重掩碼（Remasking）*。

**Overriding.** This step is primarily designed to resolve conflicts between instance predictions and semantic predictions at the per-primitive level.
**覆蓋。** 此步驟主要旨在解決逐圖元級別的實例預測和語義預測之間的衝突。

Given a primitive $p_i$, the semantic branch outputs a semantic label $l_{\text{sem}}(p_i) \in \{1, \dots, K + 1\}$ and a corresponding confidence score $s_{\text{sem}}(p_i) \in [0, 1]$.
給定一個圖元 $p_i$，語義分支輸出一個語義標籤 $l_{\text{sem}}(p_i) \in \{1, \dots, K + 1\}$ 和一個相應的置信度分數 $s_{\text{sem}}(p_i) \in [0, 1]$。

Meanwhile, if $p_i$ is assigned to $N$ instance proposals, each such proposal provides an instance label $l_{\text{inst}}^j \in \{1, \dots, K + 1\}$ and an associated confidence score $s_{\text{inst}}^j \in [0, 1]$, where $j \in \{1, \dots, N\}$ indexes the proposals that include $p_i$.
同時，如果 $p_i$ 被分配給 $N$ 個實例提議，每個這樣的提議提供一個實例標籤 $l_{\text{inst}}^j \in \{1, \dots, K + 1\}$ 和一個相關聯的置信度分數 $s_{\text{inst}}^j \in [0, 1]$，其中 $j \in \{1, \dots, N\}$ 索引包含 $p_i$ 的提議。

To resolve the conflict, we compare the semantic and instance confidence scores. If the highest instance score for $p_i$ is greater than the semantic score, i.e., $\max_j s_{\text{inst}}^j(p_i) > s_{\text{sem}}(p_i)$, then the semantic prediction for $p_i$ is overridden by the instance label and score of the highest-confidence proposal:
為了解決衝突，我們比較語義和實例置信度分數。如果 $p_i$ 的最高實例分數大於語義分數，即 $\max_j s_{\text{inst}}^j(p_i) > s_{\text{sem}}(p_i)$，那麼 $p_i$ 的語義預測將被最高置信度提議的實例標籤和分數覆蓋：
$$ l_{\text{sem}}^{\text{refined}}(p_i) = l_{\text{inst}}^{j^*}(p_i), \quad s_{\text{sem}}^{\text{refined}}(p_i) = s_{\text{inst}}^{j^*}(p_i), \quad \text{where} \quad j^* = \arg \max_{j \in \{1, \dots, N\}} s_{\text{inst}}^j(p_i). \quad (9) $$

If no instance score exceeds the semantic score, the original semantic prediction is retained.
如果沒有實例分數超過語義分數，則保留原始語義預測。

**Voting.** Given an instance proposal that contains $M$ primitives, its instance label is refined based on the most frequently occurring semantic class among those primitives. Formally, the instance label $l_{\text{inst}}$ for this proposal is refined as:
**投票。** 給定一個包含 $M$ 個圖元的實例提議，其實例標籤根據這些圖元中出現頻率最高的語義類別進行優化。形式上，此提議的實例標籤 $l_{\text{inst}}$ 優化為：
$$ l_{\text{inst}} = \arg \max_{k \in \{1, \dots, K\}} \sum_{i=1}^M \mathbb{I}(l_{\text{sem}}(p_i) = k), \quad (10) $$
where $\mathbb{I}(\cdot)$ is the indicator function that returns 1 if the condition is true and 0 otherwise. This majority voting strategy ensures that the instance label aligns with the dominant semantic context of its constituent primitives.
其中 $\mathbb{I}(\cdot)$ 是指示函數，如果條件為真則返回 1，否則返回 0。這種多數投票策略確保實例標籤與其組成圖元的主要語義上下文保持一致。

**Remasking.** For each primitive $p_i$, if it belongs to an instance mask $\mathcal{M}_{\text{inst}}$, but its semantic label $l_{\text{sem}}(p_i)$ disagrees with the instance’s majority-voted label $l_{\text{inst}}$, it is removed from the mask:
**重掩碼。** 對於每個圖元 $p_i$，如果它屬於一個實例掩碼 $\mathcal{M}_{\text{inst}}$，但其語義標籤 $l_{\text{sem}}(p_i)$ 與實例的多數投票標籤 $l_{\text{inst}}$ 不一致，則將其從掩碼中移除：
$$ p_i \in \mathcal{M}_{\text{inst}} \quad \text{and} \quad l_{\text{sem}}(p_i) \neq l_{\text{inst}} \quad \Rightarrow \quad p_i \notin \mathcal{M}_{\text{inst}}. \quad (11) $$

This operation effectively eliminates label contamination in the mask caused by prediction inconsistencies, thereby improving the purity and semantic consistency of the instance segmentation results.
此操作有效地消除了掩碼中由預測不一致引起的標籤污染，從而提高了實例分割結果的純度和語義一致性。

## 4 Experiments
## 4 實驗

### 4.1 Dataset and Metrics
### 4.1 數據集和指標

FloorPlanCAD [1] dataset consists of 11,602 diverse CAD drawings of various floor plans, each annotated with fine-grained semantic and instance labels.
FloorPlanCAD [1] 數據集包含 11,602 張各種平面圖的多樣化 CAD 圖紙，每張圖紙都標註了細粒度的語義和實例標籤。

We follow the official data split, which includes 6,965 samples for training, 810 for validation, and 3,827 for testing. The annotations cover 30 *thing* classes and 5 *stuff* classes.
我們遵循官方數據劃分，包括 6,965 個訓練樣本、810 個驗證樣本和 3,827 個測試樣本。標註涵蓋 30 個*物體（thing）*類別和 5 個*材質（stuff）*類別。

Following [1, 4], we use the Panoptic Quality (PQ) defined on vector graphics as our main metric to evaluate the performance of panoptic symbol spotting.
遵循 [1, 4]，我們使用定義在矢量圖形上的全景質量（PQ）作為我們評估全景符號識別性能的主要指標。

The Panoptic Quality (PQ) serves as a comprehensive metric that simultaneously evaluates the recognition correctness and segmentation accuracy of symbol-level predictions in vector graphics.
全景質量（PQ）作為一個綜合指標，同時評估矢量圖形中符號級預測的識別正確性和分割準確性。

A graphical primitive is denoted as $e = (l, z)$, where $l$ is the semantic label, $z$ is the instance index. A symbol is represented by a collection of primitives and is defined as $s = \{e_i \in J \mid l = l_i, z = z_i\}$, where $J$ is a set of primitives. The metric is defined as:
圖形圖元表示為 $e = (l, z)$，其中 $l$ 是語義標籤，$z$ 是實例索引。符號由一組圖元表示，定義為 $s = \{e_i \in J \mid l = l_i, z = z_i\}$，其中 $J$ 是一組圖元。該指標定義為：
$$ PQ = \frac{|TP|}{|TP| + \frac{1}{2}|FP| + \frac{1}{2}|FN|} \times \frac{\sum_{(s_p, s_g) \in TP} \text{IoU}(s_p, s_g)}{|TP|} = \frac{\sum_{(s_p, s_g) \in TP} \text{IoU}(s_p, s_g)}{|TP| + \frac{1}{2}|FP| + \frac{1}{2}|FN|}. \quad (12) $$

Here, $s_p = (l_p, z_p)$ is the predicted symbol, and $s_g = (l_g, z_g)$ is the ground truth symbol. $|TP|$, $|FP|$, and $|FN|$ represent the number of true positives, false positives, and false negatives, respectively.
這裡，$s_p = (l_p, z_p)$ 是預測符號，$s_g = (l_g, z_g)$ 是真值符號。$|TP|$、$|FP|$ 和 $|FN|$ 分別表示真陽性、假陽性和假陰性的數量。

A predicted symbol is matched to a ground truth symbol if and only if $l_p = l_g$ and $\text{IoU}(s_p, s_g) > 0.5$.
當且僅當 $l_p = l_g$ 且 $\text{IoU}(s_p, s_g) > 0.5$ 時，預測符號與真值符號匹配。

The IoU between two symbols is defined as:
兩個符號之間的 IoU 定義為：
$$ \text{IoU}(s_p, s_g) = \frac{\sum_{e_i \in s_p \cap s_g} \log(1 + L(e_i))}{\sum_{e_j \in s_p \cup s_g} \log(1 + L(e_j))}, \quad (13) $$
where $L(e)$ denotes the length of a geometric primitive $e$.
其中 $L(e)$ 表示幾何圖元 $e$ 的長度。

### 4.2 Implementation Details
### 4.2 實施細節

During training, we adopt the AdamW [39] optimizer with a weight decay of 0.05.
在訓練期間，我們採用 AdamW [39] 優化器，權重衰減為 0.05。

The initial learning rate is set to 0.0001, with a warm-up ratio of 0.05, followed by cosine decay applied over 20% of the total training epochs.
初始學習率設置為 0.0001，預熱比例為 0.05，隨後在總訓練周期的 20% 內應用餘弦衰減。

The model is trained for 500 epochs with a batch size of 2 per GPU on 8 NVIDIA A100 GPUs.
模型在 8 個 NVIDIA A100 GPU 上訓練 500 個 epoch，每個 GPU 的批大小為 2。

To improve model generalization, we apply several data augmentation strategies during training, including random horizontal and vertical flips with a probability of 0.5, random rotations, random scaling within the range $[0.8, 1.2]$, and random translations up to 10% of the CAD drawing size along both axes.
為了提高模型的泛化能力，我們在訓練期間應用了多種數據增強策略，包括概率為 0.5 的隨機水平和垂直翻轉、隨機旋轉、$[0.8, 1.2]$ 範圍內的隨機縮放，以及沿兩個軸最大為 CAD 圖紙尺寸 10% 的隨機平移。

Furthermore, we empirically set the loss weight as $\lambda_{\text{cls}} : \lambda_{\text{bce}} : \lambda_{\text{dice}} : \lambda_{\text{sem}} = 2.5 : 5.0 : 5.0 : 5.0$.
此外，我們根據經驗將損失權重設置為 $\lambda_{\text{cls}} : \lambda_{\text{bce}} : \lambda_{\text{dice}} : \lambda_{\text{sem}} = 2.5 : 5.0 : 5.0 : 5.0$。

### 4.3 Quantitative Evaluation
### 4.3 定量評估

**Panoptic Symbol Spotting.** We compare our method with existing approaches on FloorPlanCAD [1] for panoptic symbol spotting, as shown in Table 1a.
**全景符號識別。** 我們在 FloorPlanCAD [1] 上將我們的方法與現有方法進行全景符號識別的比較，如表 1a 所示。

Our method achieves the highest Panoptic Quality (PQ) across *Total*, *Thing*, and *Stuff* categories, demonstrating superior and more balanced performance.
我們的方法在*總體*、*物體*和*材質*類別中都實現了最高的全景質量（PQ），展示了卓越且更均衡的性能。

Existing methods tend to perform better on *Thing* than *Stuff* categories, revealing an imbalance in recognition. For example, SymPoint [8] scores 84.1 in Thing-PQ but only 48.2 in Stuff-PQ.
現有方法往往在*物體*類別上的表現優於*材質*類別，揭示了識別上的不平衡。例如，SymPoint [8] 在 Thing-PQ 中得分 84.1，但在 Stuff-PQ 中僅得分 48.2。

**Table 1: Quantitative evaluation results**
**(a) Panoptic symbol spotting results on FloorPlanCAD [1] dataset.**
**(a) FloorPlanCAD [1] 數據集上的全景符號識別結果。**

| Method | w/o Prior (無先驗) | | | w/ Prior (有先驗) | | |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| | PQ | PQth | PQst | PQ | PQth | PQst |
| PanCADNet [1] | 59.5 | 65.6 | 58.7 | - | - | - |
| CADTransformer [4] | 68.9 | 78.5 | 58.6 | - | - | - |
| GAT-CADNet [6] | 73.7 | - | - | - | - | - |
| SymPoint [8] | 83.3 | 84.1 | 48.2 | - | - | - |
| SymPoint-V2 [32] | 83.2 | 85.8 | 49.3 | 90.1 | 90.8 | 80.8 |
| CADSpotting [9] | - | - | - | 88.9 | 89.7 | 80.6 |
| DPSS [40] | 86.2 | 88.0 | 64.7 | 89.5 | 90.4 | 79.7 |
| **VecFormer (Ours)** | **88.4** (+2.2) | **90.9** (+2.9) | **85.9** (+21.2) | **91.1** (+1.0) | **91.8** (+1.0) | **90.4** (+9.6) |

**(b) Primitive-level semantic quality. wF1: length-weighted F1.**
**(b) 圖元級語義質量。 wF1：長度加權 F1。**

| Method | GAT-CADNet [6] | SymPoint [8] | SymPoint-V2 [32] | VecFormer (Ours) |
| :--- | :---: | :---: | :---: | :---: |
| F1 | 85.0 | 86.8 | 89.5 | **93.8** (+4.3) |
| wF1 | 82.3 | 85.5 | 88.3 | **92.2** (+3.9) |

In contrast, our method achieves more balanced results and shows a marked advantage in the *Stuff* classes, in particular, surpassing the current state-of-the-art method, SymPoint-V2 [32], by 9.6 in Stuff-PQ.
相比之下，我們的方法獲得了更均衡的結果，並在*材質*類別中顯示出明顯的優勢，特別是在 Stuff-PQ 方面超過當前最先進的方法 SymPoint-V2 [32] 9.6 分。

To reflect real-world conditions where detailed annotations (such as layers) are often unavailable, we evaluate current mainstream methods without using prior information.
為了反映詳細標註（如圖層）通常不可用的現實世界條件，我們評估了不使用先驗信息的當前主流方法。

As shown in Table 1a, existing state-of-the-art methods exhibit strong reliance on prior, particularly for *Stuff* categories.
如表 1a 所示，現有的最先進方法表現出對先驗的強烈依賴，特別是對於*材質*類別。

Specifically, SymPoint-V2 [32] and DPSS [40] suffer significant performance drops in Stuff-PQ when evaluated without prior, decreasing by 31.5 and 15 points, respectively.
具體來說，SymPoint-V2 [32] 和 DPSS [40] 在沒有先驗的情況下評估時，Stuff-PQ 性能顯著下降，分別下降了 31.5 和 15 個百分點。

In contrast, our method VecFormer consistent performance across both settings by using primitive IDs instead of layer IDs as z-coordinate of position vector, i.e., use $\mathbf{coord}_i = (c_x, c_y, j)$, but not $\mathbf{coord}_i = (c_x, c_y, k)$ described in subsection 3.1.
相比之下，我們的方法 VecFormer 通過使用圖元 ID 代替圖層 ID 作為位置向量的 z 坐標，即使用 $\mathbf{coord}_i = (c_x, c_y, j)$，而不是 3.1 節中描述的 $\mathbf{coord}_i = (c_x, c_y, k)$，在兩種設置下均保持一致的性能。

As shown in Table 1a, VecFormer achieves a PQ of 88.4 and 90.9 in the *Total* and *Thing* categories, outperforming the second-best methods by 2.2 and 2.9, respectively.
如表 1a 所示，VecFormer 在*總體*和*物體*類別中分別達到了 88.4 和 90.9 的 PQ，分別優於第二好的方法 2.2 和 2.9 分。

For the more challenging *Stuff* category, VecFormer demonstrates particularly strong performance, achieving a PQ of 85.9 with a notable gain of 21.2 over the second-best result.
對於更具挑戰性的*材質*類別，VecFormer 表現出特別強勁的性能，達到了 85.9 的 PQ，比第二好的結果顯著提高了 21.2 分。

These results demonstrate that VecFormer maintains excellent generalization and robustness even without relying on prior information, making it more suitable for practical deployment in real-world CAD scenarios.
這些結果表明，即使不依賴先驗信息，VecFormer 也能保持出色的泛化能力和魯棒性，使其更適合在現實世界 CAD 場景中實際部署。

**Primitive-Level Semantic Quality.** We assess the model’s semantic prediction performance for each graphical primitive by computing the F1 and wF1 score.
**圖元級語義質量。** 我們通過計算 F1 和 wF1 分數來評估模型對每個圖形圖元的語義預測性能。

As summarised in Table 1b, our VecFormer consistently surpasses all prior methods, achieving an improvement of 4.3 in F1 and 3.9 in wF1, compared to SymPoint-V2 [32].
如表 1b 所總結，我們的 VecFormer 始終超越所有先前的方法，與 SymPoint-V2 [32] 相比，F1 提高了 4.3，wF1 提高了 3.9。

The qualitative results are shown in Figure 3. For more qualitative studies, please refer to Appendix D.
定性結果如圖 3 所示。更多定性研究請參考附錄 D。

### 4.4 Ablation Studies
### 4.4 消融研究

**Impact of Sampling Strategy.** As shown in Table 2a, line sampling outperforms point sampling in both settings—with and without prior information.
**採樣策略的影響。** 如表 2a 所示，無論是在有還是沒有先驗信息的設置下，線採樣都優於點採樣。

The point sampling variant omits line-specific features $(l, d_x, d_y)$, leading to inferior results, confirming the superiority of line-based representations for vector graphic understanding.
點採樣變體省略了線的特定特徵 $(l, d_x, d_y)$，導致結果較差，證實了基於線的表示在矢量圖形理解方面的優越性。

**Choice of Sampling Ratio.** As shown in Table 2b, reducing the sampling ratio $\alpha_{\text{sample}}$ from 0.1 to 0.01 steadily improves performance, with the best PQ (91.1) at $\alpha_{\text{sample}} = 0.01$—also yielding peak $\text{PQ}_{\text{th}}$ and $\text{PQ}_{\text{st}}$ scores.
**採樣率的選擇。** 如表 2b 所示，將採樣率 $\alpha_{\text{sample}}$ 從 0.1 降低到 0.01 會穩步提高性能，在 $\alpha_{\text{sample}} = 0.01$ 時達到最佳 PQ (91.1)——同時產生峰值 $\text{PQ}_{\text{th}}$ 和 $\text{PQ}_{\text{st}}$ 分數。

Further reduction to $\alpha_{\text{sample}} = 0.005$ slightly degrades performance while increasing computational cost, making $\alpha_{\text{sample}} = 0.01$ the optimal trade-off between accuracy and efficiency.
進一步降低到 $\alpha_{\text{sample}} = 0.005$ 會略微降低性能，同時增加計算成本，使得 $\alpha_{\text{sample}} = 0.01$ 成為準確性和效率之間的最佳權衡。

**Table 2: Ablation studies on sampling strategy, sampling ratio, BFR, and prior information.**
**表 2：關於採樣策略、採樣率、BFR 和先驗信息的消融研究。**

**(a) Ablation studies on sampling strategy**
**(a) 關於採樣策略的消融研究**

| Prior | Strategy | PQ | PQth | PQst |
| :--- | :---: | :---: | :---: | :---: |
| w/o | Point | 87.8 | 89.9 | 85.4 |
| | **Line** | **88.4** | **90.9** | **85.9** |
| w/ | Point | 89.8 | 90.0 | 89.5 |
| | **Line** | **91.1** | **91.8** | **90.4** |

**(b) Ablation studies on sampling ratio**
**(b) 關於採樣率的消融研究**

| Ratio | PQ | PQth | PQst |
| :---: | :---: | :---: | :---: |
| 0.1 | 90.4 | 91.1 | 89.7 |
| 0.05 | 90.6 | 91.3 | 89.7 |
| 0.01 | **91.1** | **91.8** | **90.4** |
| 0.005 | 90.4 | 91.0 | 89.8 |

**(c) Ablation studies on BFR**
**(c) 關於 BFR 的消融研究**

| Method | PQ | PQth | PQst |
| :--- | :---: | :---: | :---: |
| w/o BFR | 89.2 | 90.4 | 88.1 |
| **w/ BFR** | **91.1** | **91.8** | **90.4** |
| Gain | (+1.9) | (+1.4) | (+2.3) |

**(d) Ablation studies of prior information**
**(d) 關於先驗信息的消融研究**

| Base | Layer | LFE | PQ | PQth | PQst |
| :---: | :---: | :---: | :---: | :---: | :---: |
| ✓ | | | 88.4 | 90.9 | 85.9 |
| ✓ | ✓ | | 90.2 | 91.7 | 88.4 |
| ✓ | ✓ | ✓ | **91.1** | **91.8** | **90.4** |

**Effects of the Branch Fusion Refinement Strategy.** We conduct controlled experiments to evaluate the effectiveness of the proposed Branch Fusion Refinement (BFR) strategy.
**分支融合優化策略的效果。** 我們進行了對照實驗，以評估所提出的分支融合優化（BFR）策略的有效性。

As shown in Table 2c, incorporating BFR significantly boosts performance across all metrics, demonstrating its essential role in improving prediction accuracy and robustness.
如表 2c 所示，納入 BFR 顯著提升了所有指標的性能，證明了其在提高預測準確性和魯棒性方面的關鍵作用。

**Effects of Prior Information.** As shown in Table 2d, replacing the primitive ID $j$ with the layer ID $k$ in the position vector boosts PQ from 88.4 to 90.2, highlighting the value of layer priors.
**先驗信息的效果。** 如表 2d 所示，在位置向量中用圖層 ID $k$ 替換圖元 ID $j$ 將 PQ 從 88.4 提升至 90.2，突顯了圖層先驗的價值。

Adding the Layer Feature Enhancement (LFE) module further improves PQ to 91.1, demonstrating that structural priors and LFE together enhance geometric understanding.
添加圖層特徵增強（LFE）模塊進一步將 PQ 提高到 91.1，證明結構先驗和 LFE 共同增強了幾何理解。

## 5 Conclusions
## 5 結論

We present *VecFormer*, a novel method that employs an expressive and type-agnostic line-based representation to enhance feature learning for vector graphical primitives by preserving geometric continuity and structural relationships, which are critical for symbol-rich vector graphics.
我們提出了 *VecFormer*，這是一種新穎的方法，它採用富有表現力且與類型無關的基於線的表示，通過保留幾何連續性和結構關係來增強矢量圖形圖元的特徵學習，這對於符號豐富的矢量圖形至關重要。

To unify instance- and semantic-level predictions from a dual-branch Transformer decoder, we propose the *Branch Fusion Refinement* (BFR) module, which resolves inconsistencies and improves panoptic quality.
為了統一來自雙分支 Transformer 解碼器的實例級和語義級預測，我們提出了*分支融合優化*（BFR）模塊，該模塊解決了不一致性並提高了全景質量。

A current limitation lies in the use of uniform line sampling for simplicity, which may underperform in regions of high geometric complexity.
目前的局限性在於為了簡單起見使用了均勻線採樣，這在幾何複雜度高的區域可能表現不佳。

Future work will explore a geometry-aware dynamic sampling strategy to better adapt to diverse structural patterns in vector graphics.
未來的工作將探索幾何感知的動態採樣策略，以更好地適應矢量圖形中多樣化的結構模式。

To the best of our knowledge, the proposed method does not pose any identifiable negative societal risks.
據我們所知，所提出的方法不會構成任何可識別的負面社會風險。

## Acknowledgements
## 致謝

This work was supported by National Key R&D Program of China (Grant No. 2022ZD0161301).
本工作得到了中國國家重點研發計劃（批准號 2022ZD0161301）的支持。

## References
## 參考文獻

[1] Zhiwen Fan, Lingjie Zhu, Honghua Li, Xiaohao Chen, Siyu Zhu, and Ping Tan. Floorplancad: A large-scale cad drawing dataset for panoptic symbol spotting. In *Proceedings of the IEEE/CVF international conference on computer vision*, pages 10128–10137, 2021.

[2] Alireza Rezvanifar, Melissa Cote, and Alexandra Branzan Albu. Symbol spotting for architectural drawings: state-of-the-art and new industry-driven developments. *IPSJ Transactions on Computer Vision and Applications*, 11:1–22, 2019.

[3] Alireza Rezvanifar, Melissa Cote, and Alexandra Branzan Albu. Symbol spotting on digital architectural floor plans using a deep learning-based framework. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops*, pages 568–569, 2020.

[4] Zhiwen Fan, Tianlong Chen, Peihao Wang, and Zhangyang Wang. Cadtransformer: Panoptic symbol spotting transformer for cad drawings. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 10986–10996, 2022.

[5] Xinyang Jiang, Lu Liu, Caihua Shan, Yifei Shen, Xuanyi Dong, and Dongsheng Li. Recognizing vector graphics without rasterization. *Advances in Neural Information Processing Systems*, 34:24569–24580, 2021.

[6] Zhaohua Zheng, Jianfang Li, Lingjie Zhu, Honghua Li, Frank Petzold, and Ping Tan. Gat-cadnet: Graph attention network for panoptic symbol spotting in cad drawings. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pages 11747–11756, 2022.

[7] Bingchen Yang, Haiyong Jiang, Hao Pan, and Jun Xiao. Vectorfloorseg: Two-stream graph attention network for vectorized roughcast floorplan segmentation. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 1358–1367, 2023.

[8] Wenlong Liu, Tianyu Yang, Yuhan Wang, Qizhi Yu, and Lei Zhang. Symbol as points: Panoptic symbol spotting via point-based representation. *arXiv preprint arXiv:2401.10556*, 2024.

[9] Jiazuo Mu, Fuyi Yang, Yanshun Zhang, Junxiong Zhang, Yongjian Luo, Lan Xu, Yujiao Shi, Jingyi Yu, and Yingliang Zhang. Cadspotting: Robust panoptic symbol spotting on large-scale cad drawings. *arXiv preprint arXiv:2412.07377*, 2024.

[10] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. *Advances in neural information processing systems*, 30, 2017.

[11] Maxim Kolodiazhnyi, Anna Vorontsova, Anton Konushin, and Danila Rukhovich. Oneformer3d: One transformer for unified point cloud segmentation. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 20943–20953, 2024.

[12] Alexander Kirillov, Kaiming He, Ross Girshick, Carsten Rother, and Piotr Dollár. Panoptic segmentation. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pages 9404–9413, 2019.

[13] Jonathan Long, Evan Shelhamer, and Trevor Darrell. Fully convolutional networks for semantic segmentation. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, pages 3431–3440, 2015.

[14] Liang-Chieh Chen, George Papandreou, Iasonas Kokkinos, Kevin Murphy, and Alan L Yuille. Deeplab: Semantic image segmentation with deep convolutional nets, atrous convolution, and fully connected crfs. *IEEE transactions on pattern analysis and machine intelligence*, 40(4):834–848, 2017.

[15] Huikai Wu, Junge Zhang, Kaiqi Huang, Kongming Liang, and Yizhou Yu. Fastfcn: Rethinking dilated convolution in the backbone for semantic segmentation. *arXiv preprint arXiv:1903.11816*, 2019.

[16] Liang-Chieh Chen. Rethinking atrous convolution for semantic image segmentation. *arXiv preprint arXiv:1706.05587*, 2017.

[17] Enze Xie, Wenhai Wang, Zhiding Yu, Anima Anandkumar, Jose M Alvarez, and Ping Luo. Segformer: Simple and efficient design for semantic segmentation with transformers. *Advances in neural information processing systems*, 34:12077–12090, 2021.

[18] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical image segmentation. In *Medical image computing and computer-assisted intervention–MICCAI 2015: 18th international conference, Munich, Germany, October 5-9, 2015, proceedings, part III 18*, pages 234–241. Springer, 2015.

[19] Kaiming He, Georgia Gkioxari, Piotr Dollár, and Ross Girshick. Mask r-cnn. In *Proceedings of the IEEE international conference on computer vision*, pages 2961–2969, 2017.

[20] Jitesh Jain, Jiachen Li, Mang Tik Chiu, Ali Hassani, Nikita Orlov, and Humphrey Shi. Oneformer: One transformer to rule universal image segmentation. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 2989–2998, 2023.

[21] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, pages 4015–4026, 2023.

[22] Alexander Kirillov, Ross Girshick, Kaiming He, and Piotr Dollár. Panoptic feature pyramid networks. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pages 6399–6408, 2019.

[23] Ke Sun, Yang Zhao, Borui Jiang, Tianheng Cheng, Bin Xiao, Dong Liu, Yadong Mu, Xinggang Wang, Wenyu Liu, and Jingdong Wang. High-resolution representations for labeling pixels and regions. *arXiv preprint arXiv:1904.04514*, 2019.

[24] Yanwei Li, Xinze Chen, Zheng Zhu, Lingxi Xie, Guan Huang, Dalong Du, and Xingang Wang. Attention-guided unified network for panoptic segmentation. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pages 7026–7035, 2019.

[25] Liang-Chieh Chen, Huiyu Wang, and Siyuan Qiao. Scaling wide residual networks for panoptic segmentation. *arXiv preprint arXiv:2011.11675*, 2020.

[26] Bowen Cheng, Ishan Misra, Alexander G Schwing, Alexander Kirillov, and Rohit Girdhar. Masked-attention mask transformer for universal image segmentation. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pages 1290–1299, 2022.

[27] Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun. Faster r-cnn: Towards real-time object detection with region proposal networks. *IEEE transactions on pattern analysis and machine intelligence*, 39(6):1137–1149, 2016.

[28] Thomas N Kipf and Max Welling. Semi-supervised classification with graph convolutional networks. *arXiv preprint arXiv:1609.02907*, 2016.

[29] Ke Sun, Bin Xiao, Dong Liu, and Jingdong Wang. Deep high-resolution representation learning for human pose estimation. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pages 5693–5703, 2019.

[30] Alexey Dosovitskiy. An image is worth 16x16 words: Transformers for image recognition at scale. *arXiv preprint arXiv:2010.11929*, 2020.

[31] Petar Veličković, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Lio, and Yoshua Bengio. Graph attention networks. *arXiv preprint arXiv:1710.10903*, 2017.

[32] Wenlong Liu, Tianyu Yang, Qizhi Yu, and Lei Zhang. Sympoint revolutionized: Boosting panoptic symbol spotting with layer feature enhancement. *arXiv preprint arXiv:2407.01928*, 2024.

[33] Xiaoyang Wu, Li Jiang, Peng-Shuai Wang, Zhijian Liu, Xihui Liu, Yu Qiao, Wanli Ouyang, Tong He, and Hengshuang Zhao. Point transformer v3: Simpler, faster, stronger. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 4840–4851, 2024.

[34] Xizhou Zhu, Weijie Su, Lewei Lu, Bin Li, Xiaogang Wang, and Jifeng Dai. Deformable detr: Deformable transformers for end-to-end object detection. *arXiv preprint arXiv:2010.04159*, 2020.

[35] Hao Zhang, Feng Li, Shilong Liu, Lei Zhang, Hang Su, Jun Zhu, Lionel M Ni, and Heung-Yeung Shum. Dino: Detr with improved denoising anchor boxes for end-to-end object detection. *arXiv preprint arXiv:2203.03605*, 2022.

[36] Feng Li, Hao Zhang, Huaizhe Xu, Shilong Liu, Lei Zhang, Lionel M Ni, and Heung-Yeung Shum. Mask dino: Towards a unified transformer-based framework for object detection and segmentation. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pages 3041–3050, 2023.

[37] Fausto Milletari, Nassir Navab, and Seyed-Ahmad Ahmadi. V-net: Fully convolutional neural networks for volumetric medical image segmentation. In *2016 fourth international conference on 3D vision (3DV)*, pages 565–571. Ieee, 2016.

[38] Ruoxi Deng, Chunhua Shen, Shengjun Liu, Huibing Wang, and Xinru Liu. Learning to predict crisp boundaries. In *Proceedings of the European conference on computer vision (ECCV)*, pages 562–578, 2018.

[39] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. *arXiv preprint arXiv:1711.05101*, 2017.

[40] Ruifeng Luo, Zhengjie Liu, Tianxiao Cheng, Jie Wang, Tongjie Wang, Xingguang Wei, Haomin Wang, YanPeng Li, Fu Chai, Fei Cheng, et al. Archcad-400k: An open large-scale architectural cad dataset and new baseline for panoptic symbol spotting. *arXiv preprint arXiv:2503.22346*, 2025.

## Appendix
## 附錄

### A Detailed Visual Comparisons across Different Representations.
### A 不同表示之間的詳細視覺比較。

We present additional fine-grained visualizations to facilitate a more detailed comparison of different representations.
我們提供了額外的細粒度可視化，以便對不同表示進行更詳細的比較。

As illustrated in Figure 4, our proposed line-based representation demonstrates closer visual alignment with the ground truth than point-based methods (e.g., SymPoint [8], CADSpotting [9]), effectively preserving geometric continuity and structural integrity across a variety of primitive types.
如圖 4 所示，我們提出的基於線的表示比基於點的方法（例如 SymPoint [8]、CADSpotting [9]）顯示出與真值更緊密的視覺對齊，有效地保留了各種圖元類型的幾何連續性和結構完整性。

### B Visual Comparison under Varying Sampling Ratios
### B 不同採樣率下的視覺比較

To further investigate the impact of sampling density on representation quality, we conduct a comparative analysis across different representations under varying sampling ratios $\alpha_{\text{sample}}$.
為了進一步研究採樣密度對表示質量的影響，我們在不同的採樣率 $\alpha_{\text{sample}}$ 下對不同表示進行了比較分析。

As shown in Figure 5, our line-based representation consistently maintains higher geometric fidelity and structural coherence, even under lower sampling densities.
如圖 5 所示，即使在較低的採樣密度下，我們基於線的表示也始終保持較高的幾何保真度和結構連貫性。

In contrast, point-based representations tend to suffer from fragmentation and loss of continuity as the sampling ratio decreases.
相比之下，隨著採樣率的降低，基於點的表示往往會遭受碎片化和連續性喪失的困擾。

These visual results highlight the robustness of our approach in preserving essential geometric and topological features, suggesting its suitability for vector graphics tasks where structural integrity is critical under constrained sampling conditions.
這些視覺結果突顯了我們的方法在保留基本幾何和拓撲特徵方面的魯棒性，表明其適用於結構完整性在受限採樣條件下至關重要的矢量圖形任務。

### C Sequence Length Analysis of Point- and Line-based Representations
### C 基於點和基於線的表示的序列長度分析

This section analyzes the differences in sequence length between point-based and line-based representations on FloorPlanCAD [1] dataset.
本節分析 FloorPlanCAD [1] 數據集上基於點和基於線的表示在序列長度上的差異。

We begin by configuring the line-based representation with a sampling ratio of $\alpha_{\text{sample}} = 0.01$, consistent with our experimental setup.
我們首先將基於線的表示配置為採樣率 $\alpha_{\text{sample}} = 0.01$，與我們的實驗設置一致。

For the point-based representation, we set $\alpha_{\text{sample}} = 0.001$, which yields a similar sampling density to that used in CADSpotting [9], although the sampling strategies differ.
對於基於點的表示，我們設置 $\alpha_{\text{sample}} = 0.001$，雖然採樣策略不同，但產生的採樣密度與 CADSpotting [9] 中使用的相似。

As illustrated in Figure 6, this setting results in CADSpotting, the point-based method, producing sequences that are approximately 8 times longer than our line-based counterpart.
如圖 6 所示，這種設置導致基於點的方法 CADSpotting 產生的序列比我們基於線的方法長約 8 倍。

Despite the significantly shorter sequence length, our method achieves higher Panoptic Quality (PQ), as demonstrated in the main results (Table 1a).
儘管序列長度明顯較短，但我們的方法實現了更高的全景質量（PQ），如主要結果（表 1a）所示。

To ensure a fair comparison, we further evaluate both representations under the same sampling ratio of $\alpha_{\text{sample}} = 0.01$.
為了確保公平比較，我們進一步在相同的採樣率 $\alpha_{\text{sample}} = 0.01$ 下評估這兩種表示。

Even in this setting, the line-based representation yields approximately 15% fewer tokens than the point-based representation.
即使在這種設置下，基於線的表示產生的標記也比基於點的表示少約 15%。

Moreover, ablation results in Table 2a confirm that our approach not only reduces sequence length but also achieves superior performance.
此外，表 2a 中的消融結果證實，我們的方法不僅減少了序列長度，而且實現了卓越的性能。

These findings underscore the efficiency and representational strength of the line-based approach: by encoding primitives through fewer yet structurally meaningful elements, it preserves geometric fidelity while enhancing learning effectiveness.
這些發現強調了基於線的方法的效率和表現力：通過使用更少但結構上有意義的元素對圖元進行編碼，它在保留幾何保真度的同時增強了學習效果。

This compact, structure-aware design leads to more accurate segmentation and improved overall performance, making line-based representation a more effective and scalable solution for vector graphic understanding.
這種緊湊、結構感知的設計導致更準確的分割和整體性能的提高，使基於線的表示成為矢量圖形理解的更有效和可擴展的解決方案。

### D Additional Qualitative Evaluation
### D 額外的定性評估

This section provides additional qualitative results through visualizations.
本節通過可視化提供額外的定性結果。

The color scheme for each category is defined in Figure 7, and further examples are illustrated in Figure 8, Figure 9, and Figure 10.
每個類別的配色方案在圖 7 中定義，更多示例在圖 8、圖 9 和圖 10 中展示。
