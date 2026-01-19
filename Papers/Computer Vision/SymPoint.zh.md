---
title: "SymPoint"
field: "Papers"
status: "Imported"
created_date: 2026-01-19
pdf_link: "[[SymPoint.pdf]]"
tags: [paper, Papers]
---

# SymPoint Revolutionized: Boosting Panoptic Symbol Spotting with Layer Feature Enhancement
# SymPoint 的革命性進化：透過層特徵增強提升全景符號識別

Wenlong Liu, Tianyu Yang, Qizhi Yu, and Lei Zhang
International Digital Economy Academy, Vanyi Tech

**Abstract.** SymPoint [12] is an initial attempt that utilizes point set representation to solve the panoptic symbol spotting task on CAD drawing. Despite its considerable success, it overlooks graphical layer information and suffers from prohibitively slow training convergence. To tackle this issue, we introduce SymPoint-V2, a robust and efficient solution featuring novel, streamlined designs that overcome these limitations. In particular, we first propose a Layer Feature-Enhanced module (LFE) to encode the graphical layer information into the primitive feature, which significantly boosts the performance. We also design a Position-Guided Training (PGT) method to make it easier to learn, which accelerates the convergence of the model in the early stages and further promotes performance. Extensive experiments show that our model achieves better performance and faster convergence than its predecessor SymPoint on the public benchmark. Our code and trained models are available at https://github.com/nicehuster/SymPointV2.
**摘要**。SymPoint [12] 是一個初步的嘗試，利用點集表示法來解決 CAD 圖紙上的全景符號識別任務。儘管它取得了相當大的成功，但它忽略了圖形圖層資訊，並且面臨訓練收斂速度過慢的問題。為了解決這個問題，我們引入了 SymPoint-V2，這是一個穩健且高效的解決方案，具有新穎、精簡的設計，克服了上述限制。具體而言，我們首先提出了一個層特徵增強模組（LFE），將圖形圖層資訊編碼到基元特徵中，這顯著提升了性能。我們還設計了一種位置引導訓練（PGT）方法，使其更容易學習，從而加速模型在早期的收斂並進一步提升性能。廣泛的實驗顯示，我們的模型在公共基準測試中比其前身 SymPoint 實現了更好的性能和更快的收斂速度。我們的程式碼和訓練好的模型可在 https://github.com/nicehuster/SymPointV2 獲取。

**Keywords:** panoptic symbol spotting, CAD
**關鍵詞：** 全景符號識別，CAD

## 1 Introduction
## 1 介紹

Symbol spotting is a fundamental task in computer graphics and vision and has a broad range of applications, including document image analysis community[17] and architecture, engineering and construction (AEC) industries[4]. In architecture, CAD drawings are instrumental in presenting the exact geometry, detailed semantics, and specialized knowledge relevant to product design, with basic geometric primitives, such as line segments, circles, ellipses, arcs and etc. Spotting and recognizing symbols in CAD drawings is a critical initial step in comprehending their contents, essential for a wide range of practical industrial applications. For example, Building Information Modeling (BIM) is increasingly sought after across various architectural and engineering domains, including pipe arrangement, construction inspection, and equipment maintenance. A CAD drawing typically provides a comprehensive depiction of a storey, presented in an orthogonal top-down view. Therefore, a BIM model can be precisely reconstructed from a group of 2D floor plans with accurate semantic and instance annotations, as shown in Fig 1.
符號識別是電腦圖學和視覺中的一項基本任務，具有廣泛的應用，包括文檔圖像分析社群 [17] 以及建築、工程和施工（AEC）產業 [4]。在建築領域，CAD 圖紙對於呈現與產品設計相關的精確幾何形狀、詳細語義和專業知識至關重要，其由基本的幾何基元組成，如線段、圓、橢圓、弧線等。在 CAD 圖紙中識別和定位符號是理解其內容的關鍵第一步，對於廣泛的實際工業應用至關重要。例如，建築資訊模型（BIM）在各種建築和工程領域中越來越受歡迎，包括管道佈置、施工檢查和設備維護。CAD 圖紙通常以正交俯視圖的形式提供樓層的全面描述。因此，可以從一組具有準確語義和實例標註的 2D 平面圖中精確重建 BIM 模型，如圖 1 所示。

Fig. 1: A 2D floorplan (a) and its panoptic symbol spotting results (b), in which the semantics of segments are indicated through different color and instances are highlighted by semi-transparent rectangles. The BIM model (c) with complete semantic and precise geometry can be reconstructed from such an annotated floor plan. We only present the 3D model of windows, doors, and walls for clarity.
圖 1：一個 2D 平面圖 (a) 及其全景符號識別結果 (b)，其中線段的語義通過不同的顏色表示，實例通過半透明矩形突出顯示。具有完整語義和精確幾何形狀的 BIM 模型 (c) 可以從這樣標註的平面圖中重建。為了清晰起見，我們僅展示窗戶、門和牆壁的 3D 模型。

Unlike images that are structured on regular pixel grids, CAD drawings are made up of graphical primitives such as segments, arcs, circles, ellipses, polylines, and others. Spotting each symbol (a set of graphical primitives) within a CAD drawing is challenging due to occlusions, clustering, variations in appearance, and a significant imbalance in the distribution of categories. Typical approaches[4,3,5,16,28] for tackling the task of panoptic symbol spotting in CAD drawings involve initially converting the CAD drawings into images and then processing it with powerful image-based detection or segmentation methods[15,18]. Another type of methods [6,26,22] uses graph convolutional networks to directly recognize primitives, avoiding the procedure of rastering vector graphics into images. Recently, SymPoint [12] provides a novel insight, which treats CAD drawing as a set of 2D points and applies point cloud segmentation methods to tackle it, leading to impressive results. Its superior performance surpasses all other methods, motivating us to further pursue this avenue of exploration.
與構建在規則像素網格上的圖像不同，CAD 圖紙由圖形基元組成，如線段、弧線、圓、橢圓、聚合線等。在 CAD 圖紙中識別每個符號（一組圖形基元）具有挑戰性，因為存在遮擋、聚類、外觀變化以及類別分佈的顯著不平衡。解決 CAD 圖紙中全景符號識別任務的典型方法 [4,3,5,16,28] 涉及最初將 CAD 圖紙轉換為圖像，然後使用強大的基於圖像的檢測或分割方法 [15,18] 進行處理。另一類方法 [6,26,22] 使用圖卷積網路直接識別基元，避免了將向量圖形光柵化為圖像的過程。最近，SymPoint [12] 提供了一個新穎的見解，它將 CAD 圖紙視為一組 2D 點，並應用點雲分割方法來解決它，從而獲得了令人印象深刻的結果。其優越的性能超越了所有其他方法，激勵我們進一步探索這一途徑。

Despite its great success, SymPoint is still an initial attempt which adopts a point-based backbone to extract primitive features and utilizes a transformer decoder to spot and recognize symbols. On the one hand, the former *ignores the graphical layer information* of CAD drawings, which can assign objects of the same or similar types to the same layer and associate them. For example, layers can be created separately for walls, windows, curtains, mobile furniture, fixed furniture, sanitary ware, and etc., to facilitate later drawing management. These layer information is crucial for identifying relationships between primitives. In other words, any CAD drawing can be split into multiple sub-drawings based on graphical layer information, which is crucial for recognizing complex CAD drawings, as shown in 2a. On the other hand, current transformer decoder suffers from the issue of *slow convergence in the early stages*. As shown in 2b, the model (without center queries) manifests slow convergence and lags behind our method by a large margin, particularly in the early stage of training.
儘管 SymPoint 取得了巨大的成功，但它仍然是一個初步的嘗試，它採用基於點的主幹網路來提取基元特徵，並利用 Transformer 解碼器來識別和定位符號。一方面，前者*忽略了 CAD 圖紙的圖形圖層資訊*，這些資訊可以將相同或相似類型的物件分配到同一圖層並將它們關聯起來。例如，可以分別為牆壁、窗戶、窗簾、移動家具、固定家具、衛生潔具等建立圖層，以便於後續的圖紙管理。這些圖層資訊對於識別基元之間的關係至關重要。換句話說，任何 CAD 圖紙都可以根據圖形圖層資訊拆分為多個子圖，這對於識別複雜的 CAD 圖紙至關重要，如圖 2a 所示。另一方面，目前的 Transformer 解碼器面臨*早期階段收斂緩慢*的問題。如圖 2b 所示，該模型（無中心查詢）表現出收斂緩慢，並且大幅落後於我們的方法，特別是在訓練的早期階段。

Based on the above observations and analysis, we propose our SymPoint-V2 upon SymPoint [12]. we propose two core designs: Layer Feature-Enhanced (LFE) module and Position-Guided Training (PGT) method. LFE aggregates layer information into primitive features, enhancing interaction between primitives in the same layer while PGT adopts a group of additional center queries to guide the training of the transformer decoder, which bypasses bipartite graph matching and directly learns the target, which is crucial in reducing training difficulty and accelerating convergence.
基於上述觀察和分析，我們在 SymPoint [12] 的基礎上提出了 SymPoint-V2。我們提出了兩個核心設計：層特徵增強（LFE）模組和位置引導訓練（PGT）方法。LFE 將圖層資訊聚合到基元特徵中，增強了同一圖層中基元之間的交互，而 PGT 採用一組額外的中心查詢來引導 Transformer 解碼器的訓練，這繞過了二分圖匹配並直接學習目標，這對於降低訓練難度和加速收斂至關重要。

Fig. 2: (a) A CAD drawing is composed of multiple graphical layers. (b) Comparision curves of with and without center queries.
圖 2：(a) 一張 CAD 圖紙由多個圖形圖層組成。(b) 有無中心查詢的比較曲線。

In conclusion, we propose SymPoint-V2, which improves SymPoint from several perspectives:
總結來說，我們提出了 SymPoint-V2，它從幾個角度改進了 SymPoint：

*   We proposes a Layer Feature-Enhanced module by fully utilizing graphical layer information in CAD drawings, which effectively and significantly improves the performance.
    我們提出了一個層特徵增強模組，通過充分利用 CAD 圖紙中的圖形圖層資訊，有效且顯著地提升了性能。
*   We desgin a Position-Guided Training (PGT) method by constructing a group of center queries for the transformer decoder, which manifests faster convergence and demonstrates higher performance.
    我們設計了一種位置引導訓練（PGT）方法，通過為 Transformer 解碼器構建一組中心查詢，表現出更快的收斂速度並展示了更高的性能。
*   Experiments on public benchmarks show that our approach achieves a new state-of-the-art result of 90.1 PQ on FloorplanCAD, surpassing its predecessor SymPoint (83.3 PQ) by a large margin.
    在公共基準測試上的實驗顯示，我們的方法在 FloorplanCAD 上達到了 90.1 PQ 的新技術水平（SOTA），大幅超越了其前身 SymPoint（83.3 PQ）。

## 2 Related Work
## 2 相關工作

### 2.1 Panoptic Symbol Spotting
### 2.1 全景符號識別

Traditional symbol spotting[17] usually deals with instance symbols representing *countable things*– countable symbols such as windows, tables, sofas, and beds. Following the idea in [8], [4] extended the definition by recognizing semantic of *uncountable stuff* such as wall, railing and parking spot, named it *panoptic symbol spotting*. Therefore, all components in a CAD drawing are covered in one task altogether. Fan et al. [4] propose PanCADNet, which adopts Faster-RCNN [15] to recognize countable things instances and introduces Graph Convolutional Networks (GCNs) [7] to reason the stuff semantics. Fan et al.[3] propose CADTransformer, instead utilize HRNetV2-W48 [18] to tokenize graphical primitives and modify existing ViTs [2] to aggregate graphical primitives’ embeddings for the panoptic symbol spotting task. Zheng et al.[26] convert CAD drawing as a graph and utilize Graph Attention Network(GAT) to predict the semantic and instance attributes of every graphical primitive. Besides, Liu et al.[12] pursue a different direction, and propose SymPoint to explore the feasibility of point set representation to tackle panoptic symbol spotting task.
傳統的符號識別 [17] 通常處理代表*可數物件 (countable things)* 的實例符號——如窗戶、桌子、沙發和床等可數符號。遵循 [8] 中的想法，[4] 通過識別*不可數材質 (uncountable stuff)*（如牆壁、欄杆和停車位）的語義擴展了定義，將其命名為*全景符號識別*。因此，CAD 圖紙中的所有組件都包含在一個任務中。Fan 等人 [4] 提出了 PanCADNet，它採用 Faster-RCNN [15] 來識別可數物件實例，並引入圖卷積網路 (GCNs) [7] 來推理材質語義。Fan 等人 [3] 提出了 CADTransformer，改用 HRNetV2-W48 [18] 對圖形基元進行標記化 (tokenize)，並修改現有的 ViTs [2] 以聚合圖形基元的嵌入，用於全景符號識別任務。Zheng 等人 [26] 將 CAD 圖紙轉換為圖形，並利用圖注意力網路 (GAT) 預測每個圖形基元的語義和實例屬性。此外，Liu 等人 [12] 追求不同的方向，提出了 SymPoint 來探索利用點集表示法解決全景符號識別任務的可行性。

### 2.2 Ease Training for DETRs
### 2.2 簡化 DETR 的訓練

Vision transformer is hard to train because globally searching for an object is non-trivial. This phenomenon exists in both detection and segmentation. In detection, DETR[1] suffers from slow convergence requiring 500 training epochs for convergence. Recently, researchers have dived into the meaning of the learnable queries[11,13,21,27]. They either express the queries as reference points or anchor boxes. [10,23] proposed to add noised ground truth boxes as positional queries for denoising training and they speed up detection greatly. In segmentation, Mask2Former proposed mask attention which makes training easier and speeds up convergence when compared with MaskFormer. Furthermore, Mask-Piloted (MP) training approach proposed in MP-Former[24] which additionally feeds noised groundtruth masks in masked-attention and trains the model to reconstruct the original ones. Conversely, MAFT[9] abandons the mask attention design and resort to an auxiliary center regression task instead.
視覺 Transformer 很難訓練，因為全局搜尋物件並非易事。這種現象存在於檢測和分割中。在檢測中，DETR [1] 面臨收斂緩慢的問題，需要 500 個訓練週期才能收斂。最近，研究人員深入探討了可學習查詢的意義 [11,13,21,27]。他們將查詢表示為參考點或錨框。[10,23] 提出添加帶噪聲的真實標註框（Ground Truth boxes）作為位置查詢以進行去噪訓練，這大大加快了檢測速度。在分割方面，Mask2Former 提出了遮罩注意力機制，與 MaskFormer 相比，這使得訓練更容易並加速了收斂。此外，MP-Former [24] 中提出的遮罩引導（MP）訓練方法，額外將帶噪聲的真實標註遮罩輸入到遮罩注意力中，並訓練模型重建原始遮罩。相反，MAFT [9] 放棄了遮罩注意力設計，轉而採用輔助中心回歸任務。

## 3 Approach
## 3 方法

Fig. 3: The overview of our framework.
圖 3：我們框架的概述。

We analyze the limitations of SymPoint [12] (SPv1) and propose our SymPoint-V2 (SPv2), including two improved modules upon SPv1, As shown in Fig 3. Similar to SPv1, SPv2 receives a CAD drawing and treats it as point sets to represent the graphical primitives, and then the backbone is used to extract primitive features. Subsequently, Layer Feature Enchanced (LFE) module using primitive features and layer information as inputs, integrates layer information to enhance interaction among primitives that are laid out on the same layer. Finally, the enhanced primitive features together with two kinds of query: learnable queries and center queries, are fed into the transformer decoder for query refinement. The first type of query can obtain recognized results through an MLP head, while the second type of query is used to guide the training of the transformer decoder, which bypasses bipartite graph matching and directly assigns ground truth labels to learn the target.
我們分析了 SymPoint [12] (SPv1) 的局限性並提出了我們的 SymPoint-V2 (SPv2)，包括 SPv1 之上的兩個改進模組，如圖 3 所示。與 SPv1 類似，SPv2 接收 CAD 圖紙並將其視為點集來表示圖形基元，然後使用主幹網路提取基元特徵。隨後，層特徵增強 (LFE) 模組使用基元特徵和圖層資訊作為輸入，整合圖層資訊以增強佈置在同一圖層上的基元之間的交互。最後，增強的基元特徵連同兩種類型的查詢：可學習查詢和中心查詢，被送入 Transformer 解碼器進行查詢細化。第一種類型的查詢可以通過 MLP 頭部獲得識別結果，而第二種類型的查詢用於引導 Transformer 解碼器的訓練，這繞過了二分圖匹配並直接分配真實標籤以學習目標。

### 3.1 Preliminaries
### 3.1 預備知識

**Task Formulation.** Given a CAD drawing represented by a set of graphical primitives $\{p_k\}$, the *panoptic symbol spotting* task requires a map $F_p : p_k \to (l_k, z_k) \in \mathcal L \times \mathbf N$, where $\mathcal L := \{0, \dots , L - 1\}$ is a set of predetermined set of object classes, and $\mathbf N$ is the number of possible instances. The semantic label set $\mathcal L$ can be partitioned into stuff and things subsets, namely $\mathcal L = \mathcal L^{st} \cup \mathcal L^{th}$ and $\mathcal L^{st} \cap \mathcal L^{th} = \emptyset$. We can degrade panoptic symbol spotting to *semantic symbol spotting* task or *instance symbol spotting* task, if we ignore the instance indices or only focus on the thing classes.
**任務制定。** 給定由一組圖形基元 $\{p_k\}$ 表示的 CAD 圖紙，*全景符號識別*任務需要一個映射 $F_p : p_k \to (l_k, z_k) \in \mathcal L \times \mathbf N$，其中 $\mathcal L := \{0, \dots , L - 1\}$ 是一組預定義的物件類別集合，$\mathbf N$ 是可能的實例數量。語義標籤集 $\mathcal L$ 可以劃分為材質 (stuff) 和物件 (things) 子集，即 $\mathcal L = \mathcal L^{st} \cup \mathcal L^{th}$ 且 $\mathcal L^{st} \cap \mathcal L^{th} = \emptyset$。如果我們忽略實例索引或僅關注物件類別，我們可以將全景符號識別降級為*語義符號識別*任務或*實例符號識別*任務。

**SPv1.** The SPv1[12] architecture consists of a backbone, a symbol spotting head, and an MLP head. Firstly, the graphical primitives of CAD drawings are formed as point sets representation $\mathcal P = \{p_k \mid (x_k, f_k)\}$, where $x_k \in \mathbb R^2$ represents the point position, and $f_k \in \mathbb R^6$ represents the point features. Secondly, the point sets $\mathcal P$ are fed into the backbone to get the primitive features $\mathcal F \in R^{N \times D}$, where $N$ is the number of feature tokens and $D$ is the feature dimension. The learnable object queries $\mathcal X$ and the primitive features $\mathcal F$ are fed into the transformer decoder, which refers to symbol spotting head in SPv1[12], resulting in the final object queries, The object queries are parsed to the symbol mask and the classification scores through an MLP head which is mask predicting module in SPv1[12]. For each decoder layer $l$, the process of query updating and mask predicting can be formulated as,
**SPv1。** SPv1 [12] 架構由主幹網路、符號識別頭部和 MLP 頭部組成。首先，CAD 圖紙的圖形基元形成點集表示 $\mathcal P = \{p_k \mid (x_k, f_k)\}$，其中 $x_k \in \mathbb R^2$ 代表點位置，$f_k \in \mathbb R^6$ 代表點特徵。其次，點集 $\mathcal P$ 被送入主幹網路以獲得基元特徵 $\mathcal F \in R^{N \times D}$，其中 $N$ 是特徵標記的數量，$D$ 是特徵維度。可學習的物件查詢 $\mathcal X$ 和基元特徵 $\mathcal F$ 被送入 Transformer 解碼器（在 SPv1 [12] 中稱為符號識別頭部），產生最終的物件查詢。這些物件查詢通過 MLP 頭部（SPv1 [12] 中的遮罩預測模組）被解析為符號遮罩和分類分數。對於每個解碼層 $l$，查詢更新和遮罩預測的過程可以公式化為：

$$
X_{l} = \mathrm{softmax}(A_{l-1} + Q_lK_l^T)V_l + X_{l-1}, \quad (1)
$$
$$
Y_l = f_{Y}(X_l),\quad M_l = f_{M}(X_l)F_0^T, \quad (2)
$$

where $X_l \in R^{O \times D}$ is the query features. $O$ is the number of query features. $Q_l = f_Q(X_{l-1})$, $K_l = f_K(F_r)$ and $V_l = f_V(F_r)$ are query, key and value features projected by MLP layers. $A_{l-1}$ is the attention mask. The object mask $M_l \in R^{O \times N}$ and its corresponding category $Y_l \in R^{O \times C}$ are obtained by projecting the query features using two MLP layers $f_Y$ and $f_M$, where $C$ is the category number and $N$ is the number of primitives. Meanwhile, the Attention with Connection Module (ACM) and Contrastive Connection Learning scheme (CCL) are also proposed by SPv1 to effectively utilize connections between primitives.
其中 $X_l \in R^{O \times D}$ 是查詢特徵。$O$ 是查詢特徵的數量。$Q_l = f_Q(X_{l-1})$，$K_l = f_K(F_r)$ 和 $V_l = f_V(F_r)$ 是由 MLP 層投影的查詢 (query)、鍵 (key) 和值 (value) 特徵。$A_{l-1}$ 是注意力遮罩。物件遮罩 $M_l \in R^{O \times N}$ 及其對應的類別 $Y_l \in R^{O \times C}$ 是通過使用兩個 MLP 層 $f_Y$ 和 $f_M$ 投影查詢特徵獲得的，其中 $C$ 是類別數量，$N$ 是基元數量。同時，SPv1 還提出了帶有連接模組的注意力 (ACM) 和對比連接學習方案 (CCL)，以有效利用基元之間的連接。

**Baseline.** We build our baseline upon SPv1, Although connection relationships between primitives are widespread in CAD drawings, their impact on model performance is limited in complex CAD drawings. Therefore, for simplicity, we abandoned ACM and CCL which are proposed by SPv1.
**基準 (Baseline)。** 我們在 SPv1 的基礎上構建我們的基準，雖然基元之間的連接關係在 CAD 圖紙中很普遍，但它們對複雜 CAD 圖紙中模型性能的影響有限。因此，為了簡單起見，我們放棄了 SPv1 提出的 ACM 和 CCL。

### 3.2 Layer Feature Enchanced Module
### 3.2 層特徵增強模組

Fig. 4: The framework of our LFE.
圖 4：我們 LFE 的框架。

In CAD drawing, graphical layers are tools for effectively organizing and managing design elements. They allow designers to categorize different types of symbols ( walls, windows, curtains, mobile furniture, fixed furniture, sanitary ware, etc.), facilitating control over visibility, editing, and attribute assignment of these elements. One straightforward idea is to integrate layer information into the process of extracting primitive features in the backbone. But, to be compatible with different point-based backbones, We thus propose the Layer Feature-Enhanced (LFE) module and insert it after the backbone. The input of this module is the primitive features $\mathcal F$ and the corresponding layer IDs for each primitive as is shown in Fig. 4. This module has two important parts: *pool function* $\varphi(\cdot)$ and *fusion function* $f(\cdot)$. The former calculates global layer features, while the latter integrates these global layer features into each primitive feature.
在 CAD 圖紙中，圖形圖層是有效組織和管理設計元素的工具。它們允許設計師將不同類型的符號分類（牆壁、窗戶、窗簾、移動家具、固定家具、衛生潔具等），便於控制這些元素的可見性、編輯和屬性分配。一個直觀的想法是將圖層資訊整合到主幹網路提取基元特徵的過程中。但是，為了兼容不同的基於點的主幹網路，我們因此提出了層特徵增強 (LFE) 模組並將其插入到主幹網路之後。如圖 4 所示，該模組的輸入是基元特徵 $\mathcal F$ 和每個基元對應的圖層 ID。該模組有兩個重要部分：*池化函數 (pool function)* $\varphi(\cdot)$ 和 *融合函數 (fusion function)* $f(\cdot)$。前者計算全局圖層特徵，而後者將這些全局圖層特徵整合到每個基元特徵中。

**Pool Function.** Since the layer number can be directly obtained from the CAD drawing, as shown in Fig. 3, after obtaining the primitive features $\mathcal F$ from backbone, it can be divided into $L$ groups $\mathcal G = \{g_1, g_2, g_3, \dots, g_L\}$ based on the graphical layer IDs, where $L$ is total number of graphical layers.
**池化函數。** 由於圖層編號可以直接從 CAD 圖紙中獲得，如圖 3 所示，在從主幹網路獲得基元特徵 $\mathcal F$ 後，可以根據圖形圖層 ID 將其分為 $L$ 組 $\mathcal G = \{g_1, g_2, g_3, \dots, g_L\}$，其中 $L$ 是圖形圖層的總數。

We utilize the pool function for each group of primitive features since the number of primitives laid out on different layers varies greatly. We use a combination of mean pooling $p_1$, max pooling $p_2$, and attention pooling $p_3$ to extract multi-scale global layer features $\mathcal U$.
由於佈置在不同圖層上的基元數量差異很大，我們對每組基元特徵使用池化函數。我們結合使用平均池化 $p_1$、最大池化 $p_2$ 和注意力池化 $p_3$ 來提取多尺度全局圖層特徵 $\mathcal U$。

$$
\mathcal U(g_i)= \varphi (p_1(g_i) \odot p_2(g_i) \odot p_3(g_i)), g_i\in \mathcal G, i:=\{0,\ldots ,L\} \quad (3)
$$

where, $\odot$ is concat operation, $\varphi(\cdot)$ is a three-layer MLP.
其中，$\odot$ 是串聯操作，$\varphi(\cdot)$ 是一個三層 MLP。

**Fusion Function.** After extracting global layer features $\mathcal U$, we fuses it and primitive features $\mathcal F$ with broadcast sum or concat. This fusion strategy has the following advantages. (1) *Global-to-Local*. The global layer features with strong layer information can enhance the original primitive features and make global layer information transfer to each primitive feature. (2) *Simple*. This fusion strategy is simple, without introducing extra computational cost. In our experiments, we use the concat operation by default.
**融合函數。** 在提取全局圖層特徵 $\mathcal U$ 後，我們使用廣播求和 (broadcast sum) 或串聯 (concat) 將其與基元特徵 $\mathcal F$ 融合。這種融合策略具有以下優點。(1) *全局到局部 (Global-to-Local)*。具有強大圖層資訊的全局圖層特徵可以增強原始基元特徵，並使全局圖層資訊轉移到每個基元特徵。(2) *簡單*。這種融合策略很簡單，不會引入額外的計算成本。在我們的實驗中，我們默認使用串聯操作。

To integrate layer information to primitive features, we apply LFE module in the mask predicting process. Therefore, Eq. 2 can be reformulated as,
為了將圖層資訊整合到基元特徵中，我們在遮罩預測過程中應用 LFE 模組。因此，公式 2 可以重新表述為：

$$
Y_l = f_{Y}(X_l),\quad M_l = f_{M}(X_l)f_{LFE}(F_0)^T, \quad (4)
$$

where, $f_{LFE}$ is LFE module, and we only applied it on the highest resolution primitives for efficiency.
其中，$f_{LFE}$ 是 LFE 模組，為了效率，我們僅將其應用於最高解析度的基元。

### 3.3 Position-Guided Training
### 3.3 位置引導訓練

To address the slow convergence problem, inspired by DN-DETR[10] and MP-Former[24], we proposed the Position-Guided Training (PGT) method. We construct center queries and along with the learnable queries to feed into the transformer decoder for query refinement. The learnable queries match to GT one by one using bipartite graph matching, while the center queries are assigned to GT to directly learn the target. This training method has the following advantages. (1) *Make learning easier*. The center queries bypass the bipartite graph matching and serve as a shortcut to directly learn mask refinement. By doing so, the transformer decoder learning becomes easier, making bipartite graph matching more stable. (2)*Make learning more stable*. Due to tremendous differences in the distribution of primitives between each graphical layer, the LFE module could easily cause fluctuations in mask cross-entropy loss. The introduction of center queries makes the model converge more stably.
為了解決收斂緩慢的問題，受 DN-DETR [10] 和 MP-Former [24] 的啟發，我們提出了位置引導訓練（PGT）方法。我們構建中心查詢，並與可學習查詢一起輸入到 Transformer 解碼器進行查詢細化。可學習查詢使用二分圖匹配與真實值 (GT) 一一匹配，而中心查詢被分配給 GT 以直接學習目標。這種訓練方法具有以下優點。(1) *使學習更容易*。中心查詢繞過了二分圖匹配，作為直接學習遮罩細化的捷徑。這樣一來，Transformer 解碼器的學習變得更容易，使得二分圖匹配更加穩定。(2) *使學習更穩定*。由於每個圖形圖層之間基元分佈的巨大差異，LFE 模組很容易引起遮罩交叉熵損失的波動。引入中心查詢使得模型收斂更加穩定。

Our center query consists of two parts: class embedding $Q_c$ and positional encoding $Q_p$. The former represents feature information, which can be parsed to the mask/box and the classification scores through an MLP head, while the latter represents positional information, which is the corresponding positional encoding.
我們的中心查詢由兩部分組成：類別嵌入 $Q_c$ 和位置編碼 $Q_p$。前者代表特徵資訊，可以通過 MLP 頭部解析為遮罩/框和分類分數，而後者代表位置資訊，即相應的位置編碼。

**Class Embedding.** We use the class embeddings of ground-truth categories as queries because queries will dot-product with primitive features to get mask prediction as in Eq. 2 and an intuitive way to distinguish instances/stuff is to use their categories. The class embedding is defined as follows:
**類別嵌入。** 我們使用真實類別的類別嵌入作為查詢，因為查詢將與基元特徵進行點積以獲得遮罩預測（如公式 2 所示），區分實例/材質的一種直觀方法是使用它們的類別。類別嵌入定義如下：

$$
Q_c = f_{embed}(l) \quad (5)
$$

where $l$ is ground truth class label and $f_{embed}$ is learnable embedding function.
其中 $l$ 是真實類別標籤，$f_{embed}$ 是可學習的嵌入函數。

**Positional Encoding.** We take the center of the instance from ground truth and use Fourier positional encodings[19] to calculate $Q_p$, Since we do not require accurate center coordinates, we perturb the center point to increase diversity, as follows:
**位置編碼。** 我們從真實值中獲取實例的中心，並使用傅立葉位置編碼 [19] 來計算 $Q_p$。由於我們不需要精確的中心座標，我們對中心點進行擾動以增加多樣性，如下所示：

$$
Q_p = f_{fourier}(Q_{gt}), Q_{gt} \sim \mathcal N(p_{ct}, {\sigma }^2) \quad (6)
$$

where $f_{fourier}$ is fourier positional encodings and $p_{ct}$ is instance center. $\mathcal N$ means the Gaussian distribution and $\sigma$ represents deviation. $\sigma = (\epsilon \cdot w, \epsilon \cdot h)$ and $w, h$ is the width and height of instance. $\epsilon$ is the scale factor.
其中 $f_{fourier}$ 是傅立葉位置編碼，$p_{ct}$ 是實例中心。$\mathcal N$ 表示高斯分佈，$\sigma$ 代表偏差。$\sigma = (\epsilon \cdot w, \epsilon \cdot h)$，$w, h$ 是實例的寬度和高度。$\epsilon$ 是縮放因子。

The main difference between our PTG and DN-DETR and MP-Former is that the intrinsicality of DN-DETR and MP-Former are denoising training methods, which feed GT bounding boxes or masks with noises into the transformer decoder and train the model to reconstruct the original boxes or masks. However, our method does not construct any regression task to obtain the accurate object center position, we only construct ground truth center queries to guide the transformer decoder to focus on the position of symbols.
我們的 PTG 與 DN-DETR 和 MP-Former 之間的主要區別在於，DN-DETR 和 MP-Former 本質上是去噪訓練方法，它們將帶有噪聲的 GT 邊界框或遮罩輸入到 Transformer 解碼器中，並訓練模型重建原始的框或遮罩。然而，我們的方法並不構建任何回歸任務來獲得準確的物件中心位置，我們僅構建真實中心查詢來引導 Transformer 解碼器關注符號的位置。

### 3.4 Training and Inference
### 3.4 訓練與推論

**Training.** During the training phase, we adopt bipartite matching and set prediction loss to assign ground truth to predictions with the smallest matching cost. The overall loss is defined as:
**訓練。** 在訓練階段，我們採用二分圖匹配並設定預測損失，將真實值分配給具有最小匹配成本的預測。總損失定義為：

$$
\mathcal L= \mathcal L_Q + \mathcal L_{aux}, \mathcal L_Q=\lambda _{bce}L_{bce}+ \lambda _{dice}L_{dice}+\lambda _{cls}L_{cls} \quad (7)
$$

where $\mathcal L_Q$ is the loss for learnable queries and $\mathcal L_{aux}$ is for center queries. We use the same losses to supervise the center queries. $L_{bce}$ is the binary cross-entropy loss (over the foreground and background of that mask). $L_{dice}$ is the mask Dice loss and $L_{cls}$ is the default cross-entropy loss. The value of $\{\lambda_{bce}, \lambda_{dice}, \lambda_{cls}\}$ is same as SPv1.
其中 $\mathcal L_Q$ 是可學習查詢的損失，$\mathcal L_{aux}$ 是中心查詢的損失。我們使用相同的損失來監督中心查詢。$L_{bce}$ 是二元交叉熵損失（針對該遮罩的前景和背景）。$L_{dice}$ 是遮罩 Dice 損失，$L_{cls}$ 是預設的交叉熵損失。$\{\lambda_{bce}, \lambda_{dice}, \lambda_{cls}\}$ 的值與 SPv1 相同。

**Inference.** During the test phase, center queries will not be generated. That is, we only parse learnable queries for predicting mask and classification scores by an MLP head.
**推論。** 在測試階段，不會生成中心查詢。也就是說，我們僅解析可學習查詢，通過 MLP 頭部預測遮罩和分類分數。

## 4 Experiments
## 4 實驗

### 4.1 Experimental Setup
### 4.1 實驗設置

**Dataset and Metrics.** We conduct our experiments on FloorPlanCAD dataset[4], which has 11,602 CAD drawings of various floor plans with segment-grained panoptic annotation and covering 30 things and 5 stuff classes. We use the panoptic quality (PQ) defined on vector graphics as our main metric to evaluate the performance of panoptic symbol spotting. PQ is defined as the product of segmentation quality (SQ) and recognition quality (RQ), expressed by the formula,
**資料集和指標。** 我們在 FloorPlanCAD 資料集 [4] 上進行實驗，該資料集擁有 11,602 張各種平面圖的 CAD 圖紙，具有線段粒度的全景標註，涵蓋 30 個物件 (things) 和 5 個材質 (stuff) 類別。我們使用定義在向量圖形上的全景品質 (PQ) 作為主要指標來評估全景符號識別的性能。PQ 定義為分割品質 (SQ) 和識別品質 (RQ) 的乘積，公式表示為：

$$
PQ = \underbrace {\frac {\sum _{(s_p,s_g)\in TP} \text {IoU}(s_p,s_g)} {\vert TP \vert }}_{\text {SQ}} \times \underbrace {\frac {\vert TP \vert }{\vert TP \vert + \frac {1}{2} \vert FP \vert + \frac {1}{2} \vert FN \vert }}_{\text {RQ}} \quad (8)
$$

where a graphical primitive $p = (l, z)$ with a semantic label $l$ and an instance index $z$, $s_p = (l_p, z_p)$ is the predicted symbol. $s_g = (l_g, z_g)$ is the ground truth symbol. A certain predicted symbol is considered as matched if it finds a ground truth symbol, with $l_p = l_g$ and $\text{IoU}(s_p, s_g) > 0.5$. The IoU between two primitives is calculated based on arc length $L(\cdot)$,
其中圖形基元 $p = (l, z)$ 具有語義標籤 $l$ 和實例索引 $z$，$s_p = (l_p, z_p)$ 是預測符號。$s_g = (l_g, z_g)$ 是真實符號。如果預測符號找到一個真實符號，且 $l_p = l_g$ 且 $\text{IoU}(s_p, s_g) > 0.5$，則認為該預測符號是匹配的。兩個基元之間的 IoU 是基於弧長 $L(\cdot)$ 計算的，

$$
\text {IoU}(s_p, s_g) = \frac {\Sigma _{p_i \in s_p \cap s_g } log(1 + L(p_i)) } {\Sigma _{p_j \in s_p \cup s_g } log(1 + L(p_j))} \quad (9)
$$

The aforementioned three metrics can be adapted for both *thing* and *stuff* categories, represented as $PQ^{Th}, PQ^{St}, RQ^{Th}, RQ^{St}, SQ^{Th}, SQ^{St}$, respectively.
上述三個指標可以分別適用於*物件 (thing)* 和*材質 (stuff)* 類別，表示為 $PQ^{Th}, PQ^{St}, RQ^{Th}, RQ^{St}, SQ^{Th}, SQ^{St}$。

**Implementation Details.** Our model is trained on 8 NVIDIA Tesla A100 GPUs with a global batch size of 16 for 250 epochs. Our other basic setup mostly follows the SPv1 framework, except for the following adaptations: 1) The initial learning rate is $2e^{-4}$ and optimizer weight decay is 0.1, while SPv1 is $1e^{-4}$ and 0.001 respectively; 2) We use cosine annealing schedule; 3) We use gradient clipping trick for stable training. As shown in Table 3, our baseline method trained for only 250 epochs achieves 82.1 PQ on floorplanCAD while SPv1 trained for 1000 epochs achieves 83.3 PQ.
**實作細節。** 我們的模型在 8 個 NVIDIA Tesla A100 GPU 上訓練，全局批次大小為 16，訓練 250 個週期。我們的其他基本設置大多遵循 SPv1 框架，除了以下調整：1) 初始學習率為 $2e^{-4}$，優化器權重衰減為 0.1，而 SPv1 分別為 $1e^{-4}$ 和 0.001；2) 我們使用餘弦退火排程；3) 我們使用梯度裁剪技巧以進行穩定訓練。如表 3 所示，我們的基準方法僅訓練 250 個週期就在 floorplanCAD 上達到了 82.1 PQ，而 SPv1 訓練 1000 個週期達到 83.3 PQ。

Fig. 5: Performance comparison with SPv1[12], the currently best performing panoptic symbol spotting approach. Per-class PQ results for 35 classes of FloorplanCAD are presented. Note that, we skip the classes that contain less than 1k graphical primitives.
圖 5：與目前全景符號識別方法中表現最好的 SPv1 [12] 的性能比較。展示了 FloorplanCAD 中 35 個類別的每類 PQ 結果。注意，我們跳過了包含少於 1k 個圖形基元的類別。

### 4.2 Benchmark Results
### 4.2 基準測試結果

As reported in [4,26,3,12], in this section, we also compare our methods with previous works in three tasks: semantic symbol spotting, instance symbol spotting and panoptic symbol spotting. In each benchmark, the **red bold** font and the blue font indicate the best two results.
正如 [4,26,3,12] 中所報告的，在本節中，我們還在三個任務中比較了我們的方法與以前的工作：語義符號識別、實例符號識別和全景符號識別。在每個基準測試中，**紅色粗體**字體和藍色字體表示最佳的兩個結果。

Table 1: **Semantic Symbol Spotting** comparison results with previous works. wF1: length-weighted F1.
**表 1：語義符號識別**與先前工作的比較結果。wF1：長度加權 F1。

| Methods | PanCAD.[4] | CADTrans.[3] | GAT-CAD.[26] | SPv1[12] | SPv2(Ours) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| F1 | 80.6 | 82.2 | 85.0 | <span style="color:blue">86.8</span> | <span style="color:red">**89.5**</span> |
| wF1 | 79.8 | 80.1 | 82.3 | <span style="color:blue">85.5</span> | <span style="color:red">**88.3**</span> |

Table 2: **Instance Symbol Spotting** comparison results with image detection methods.
**表 2：實例符號識別**與圖像檢測方法的比較結果。

| Method | Backbone | AP50 | AP75 | mAP | #Params | Speed |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| FasterRCNN [15] | R101 | 60.2 | 51.0 | 45.2 | 61M | 59ms |
| YOLOv3 [14] | DarkNet53 | 63.9 | 45.2 | 41.3 | 62M | 11ms |
| FCOS [20] | R101 | 62.4 | 49.1 | 45.3 | 51M | 57ms |
| DINO [23] | R50 | 64.0 | 54.9 | 47.5 | 47M | 42ms |
| SPv1[12] | PointT[25] | <span style="color:blue">66.3</span> | <span style="color:blue">55.7</span> | <span style="color:blue">52.8</span> | 35M | 66ms |
| **SPv2(ours)** | PointT[25] | <span style="color:red">**71.3**</span> | <span style="color:red">**60.7**</span> | <span style="color:red">**60.1**</span> | 35M | 95ms |

**Semantic symbol spotting.** We compare our methods with symbol spotting methods[4,26,3]. The main test results are summarized in Table 1. Our SPv2 outperforms all existing approaches in the task of semantic symbol spotting. More importantly, compared to SPv1[12], we achieve an absolute improvement of **2.7% F1** and **2.8% wF1** respectively.
**語義符號識別。** 我們將我們的方法與符號識別方法 [4,26,3] 進行比較。主要的測試結果總結在表 1 中。我們的 SPv2 在語義符號識別任務中優於所有現有方法。更重要的是，與 SPv1 [12] 相比，我們分別實現了 **2.7% F1** 和 **2.8% wF1** 的絕對提升。

**Instance symbol spotting.** We additionally conduct comparisons between our method and a range of image detection methods, including FasterRCNN [15], YOLOv3 [14], FCOS [20], and recent DINO [23]. similar to SPv1[12], We calculate the maximum bounding box of the predicted mask for box AP metric. The main comparison results are listed in Table 2. Compared to SPv1, we outperform SPv1 by an absolute improvement of **7.3% mAP** and **5.0% AP50**, respectively. It is worth noting that the additional parameters introduced amount to less than 0.5M, and the inference time has increased by only 29ms.
**實例符號識別。** 我們另外比較了我們的方法與一系列圖像檢測方法，包括 FasterRCNN [15]、YOLOv3 [14]、FCOS [20] 和最近的 DINO [23]。與 SPv1 [12] 類似，我們計算預測遮罩的最大邊界框以用於框 AP 指標。主要的比較結果列在表 2 中。與 SPv1 相比，我們的 mAP 和 AP50 分別有 **7.3%** 和 **5.0%** 的絕對提升。值得注意的是，引入的額外參數少於 0.5M，而推論時間僅增加了 29ms。

**Panoptic symbol spotting.** We mainly compare our method with its predecessor SPv1[12] , which is the first framework using point sets representation to perform panoptic symbol spotting task. Table 3 shows comparison results of panoptic symbol spotting performance. Our method SPv2 surpasses SPv1 by an absolute improvement of **6.8% PQ**, **4.9% SQ** and **2.5% RQ** respectively. Notably, SPv2 greatly outperforms the baseline by an absolute improvement of **30.5%** on $PQ^{St}$, demonstrating its significant superiority in recognizing *stuff* category. Additionally, Fig. 5 presents per-class PQ in the dataset compared to SPv1. Our SPv2 surpasses SPv1 in most classes.
**全景符號識別。** 我們主要將我們的方法與其前身 SPv1 [12] 進行比較，後者是第一個使用點集表示法執行全景符號識別任務的框架。表 3 顯示了全景符號識別性能的比較結果。我們的方法 SPv2 分別在 **PQ**、**SQ** 和 **RQ** 上比 SPv1 絕對提升了 **6.8%**、**4.9%** 和 **2.5%**。值得注意的是，SPv2 在 $PQ^{St}$ 上比基準絕對提升了 **30.5%**，展示了其在識別*材質 (stuff)* 類別方面的顯著優勢。此外，圖 5 展示了與 SPv1 相比資料集中各類別的 PQ。我們的 SPv2 在大多數類別中都超越了 SPv1。

### 4.3 Qualitative Results
### 4.3 定性結果

In Fig. 6, we present qualitative panoptic symbol spotting results on FloorplanCAD as compared to the ground truth masks and those of SPv1[12]. The showcased scenes are from the test splits of this dataset, and they are diverse in terms of the type of scenes they exhibit, e.g. residential buildings and core of towers, shopping malls, and schools. It can be observed that, with our proposed method, more precise instance/stuff masks are obtained as compared to the current state-of-the-art. The highlighted red arrows clearly outline examples where SPv1 predicts wrong instances and merged instances that contain many background primitives, while our method, which effectively utilizes graphical layer information and position guided training method, is able to distinguish between instances and background primitives, and perceive the object position.
在圖 6 中，我們展示了 FloorplanCAD 上的全景符號識別定性結果，並與真實遮罩和 SPv1 [12] 的結果進行了比較。展示的場景來自該資料集的測試分割，它們在場景類型方面具有多樣性，例如住宅建築、塔樓核心、購物中心和學校。可以觀察到，與當前的技術水平相比，使用我們提出的方法可以獲得更精確的實例/材質遮罩。紅色箭頭突出的例子清楚地勾勒出 SPv1 預測錯誤實例和合併了包含許多背景基元的實例的情況，而我們的方法有效利用了圖形圖層資訊和位置引導訓練方法，能夠區分實例和背景基元，並感知物件位置。

Table 3: Panoptic symbol spotting results on FloorplanCAD dataset[4]. $^{\ddagger}$: trained on 1000 epochs.
**表 3：FloorplanCAD 資料集 [4] 上的全景符號識別結果**。$^{\ddagger}$：訓練 1000 個週期。

| Method | Total PQ | Total SQ | Total RQ | Total mIoU | Thing PQ | Thing SQ | Thing RQ | Thing mAP | Stuff PQ | Stuff SQ | Stuff RQ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| PanCADNet[4] | 59.5 | 82.6 | 66.9 | - | 65.6 | 86.1 | 76.1 | - | 58.7 | 81.3 | 72.2 |
| CADTransormer[3] | 68.9 | 88.3 | 73.3 | - | 78.5 | 94.0 | 83.5 | - | 58.6 | 81.9 | 71.5 |
| GAT-CADNet[26] | 73.7 | 91.4 | 80.7 | - | - | - | - | - | - | - | - |
| SPv1$^{\ddagger}$[12] | <span style="color:blue">83.3</span> | <span style="color:blue">91.4</span> | <span style="color:blue">91.1</span> | <span style="color:blue">69.7</span> | 84.1 | <span style="color:blue">94.7</span> | 88.8 | 52.8 | 48.2 | 69.5 | 69.4 |
| baseline | 82.1 | 90.8 | 90.4 | 68.7 | <span style="color:blue">84.6</span> | 92.0 | <span style="color:blue">91.9</span> | <span style="color:blue">52.9</span> | <span style="color:blue">50.3</span> | <span style="color:blue">70.6</span> | <span style="color:blue">71.3</span> |
| **SPv2(ours)** | <span style="color:red">**90.1**</span> | <span style="color:red">**96.3**</span> | <span style="color:red">**93.6**</span> | <span style="color:red">**74.0**</span> | <span style="color:red">**90.8**</span> | <span style="color:red">**96.6**</span> | <span style="color:red">**94.0**</span> | <span style="color:red">**60.1**</span> | <span style="color:red">**80.8**</span> | <span style="color:red">**90.9**</span> | <span style="color:red">**88.9**</span> |

### 4.4 Ablation Studies
### 4.4 消融研究

In this section, we conduct a component-wise analysis to demonstrate the effectiveness of SPv2.
在本節中，我們進行逐個組件的分析以證明 SPv2 的有效性。

**Effects of Components.** We ablate each component that improves the performance of SPv2 in Table 4a. Our proposed LFE and PGT promote the baseline method by absolute **6.5% PQ (4.8% $PQ^{Th}$, 29.4% $PQ^{St}$)** and **2.5% PQ (3.1% $PQ^{Th}$)**, respectively.
**組件的效果。** 我們在表 4a 中對提高 SPv2 性能的每個組件進行了消融。我們提出的 LFE 和 PGT 分別使基準方法絕對提升了 **6.5% PQ (4.8% $PQ^{Th}$, 29.4% $PQ^{St}$)** 和 **2.5% PQ (3.1% $PQ^{Th}$)**。

**Layer Feature-Enhanced Module.** In section 3.2, we design the LFE module to integrate graphical layer information. we make additional analysis on pool types, feature dim of $\varphi(\cdot)$ and multi-level LFE. **1) Pool Types.** We compare different types of pool function to explore the impact of performance. As shown in Table. 4c, our proposed multi-scale global fusion effectively promote of performance. **2) Feature Dim of $\varphi(\cdot)$.** We ablate the hidden feature dims of MLP $\varphi(\cdot)$ used in Eq. 3.2 to explore its impact on performance. As shown in Table. 4d, the performance can be improved slightly as the number of parameters increases, we select 256 by default for parameter efficiency. **3) Multi-scale LFE.** In section 3.2, SPv2 refines learnable queries by iteratively attending to primitive features at different scaled outputs from the backbone. For simplicity, we only apply LFE module to the highest resolution primitive features $\mathcal F_0$ by default. We also provide the result in Table. 4e when applying it to multi-scale primitive features ($\mathcal F_0, \mathcal F_1, \mathcal F_2, \mathcal F_3, \mathcal F_4$). It can even lead to improved performance. But it also increases the inference time greatly, reaching 212ms.
**層特徵增強模組。** 在 3.2 節中，我們設計了 LFE 模組以整合圖形圖層資訊。我們對池化類型、$\varphi(\cdot)$ 的特徵維度和多級 LFE 進行了額外分析。**1) 池化類型。** 我們比較了不同類型的池化函數以探索其對性能的影響。如表 4c 所示，我們提出的多尺度全局融合有效地提升了性能。**2) $\varphi(\cdot)$ 的特徵維度。** 我們消融了公式 3.2 中使用的 MLP $\varphi(\cdot)$ 的隱藏特徵維度，以探索其對性能的影響。如表 4d 所示，隨著參數數量的增加，性能可以略微提高，為了參數效率，我們默認選擇 256。**3) 多尺度 LFE。** 在 3.2 節中，SPv2 通過迭代關注來自與主幹網路不同尺度輸出的基元特徵來細化可學習查詢。為了簡單起見，我們默認僅將 LFE 模組應用於最高解析度的基元特徵 $\mathcal F_0$。我們還在表 4e 中提供了將其應用於多尺度基元特徵 ($\mathcal F_0, \mathcal F_1, \mathcal F_2, \mathcal F_3, \mathcal F_4$) 時的結果。這甚至可以導致性能提升。但它也大大增加了推論時間，達到 212ms。

Fig. 6: Visual comparison between SPv1[12] and ours. The red arrows highlight the key regions.
圖 6：SPv1 [12] 與我們的方法之間的視覺比較。紅色箭頭突出了關鍵區域。

Table 4: Ablation studies on different techniques, pool type, feat dim of $\varphi(\cdot)$, Multi-scale LFE, ablation on ceter query, positional encoding type, and training method.
**表 4：關於不同技術、池化類型、$\varphi(\cdot)$ 特徵維度、多尺度 LFE、中心查詢消融、位置編碼類型和訓練方法的消融研究。**

(a) Ablation studies on different components.
(a) 不同組件的消融研究。

| Base | LFE | PGT | PQ | PQ Th | PQ St | Param | Time |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| ✓ | | | 82.1 | 84.6 | 50.3 | 35.06M | 66ms |
| ✓ | ✓ | | 88.6 | 89.4 | 79.7 | 35.14M | 95ms |
| ✓ | | ✓ | 84.6 | 87.7 | 49.2 | 35.06M | 66ms |
| ✓ | ✓ | ✓ | 90.1 | 90.8 | 80.8 | 35.14M | 95ms |

(b) Ablation studies on center query.
(b) 中心查詢的消融研究。

| ClsE. | PosE. | PQ | RQ | SQ |
| :---: | :---: | :---: | :---: | :---: |
| | | 88.6 | 92.9 | 95.4 |
| ✓ | | 89.2 | 93.1 | 95.8 |
| | ✓ | 89.8 | 93.7 | 95.8 |
| ✓ | ✓ | 90.1 | 93.6 | 96.3 |

(c) Pool type
(c) 池化類型

| Pool Type | PQ | RQ | SQ |
| :--- | :---: | :---: | :---: |
| baseline | 82.1 | 90.4 | 90.8 |
| avepool($p_1$) | 86.3 | 91.9 | 93.9 |
| maxpool($p_2$) | 87.3 | 92.6 | 94.2 |
| attnpool($p_3$) | 87.9 | 92.8 | 94.7 |
| concat($p_1 \odot p_2 \odot p_3$) | 88.6 | 92.9 | 95.4 |

(d) Feature dim of $\varphi(\cdot)$.
(d) $\varphi(\cdot)$ 的特徵維度。

| Feat Dim | PQ | RQ | SQ |
| :---: | :---: | :---: | :---: |
| 128 | 87.1 | 91.3 | 95.4 |
| 256 | 88.6 | 92.9 | 95.4 |
| 512 | 89.1 | 92.9 | 95.9 |
| 1024 | 88.6 | 92.7 | 95.6 |
| 2048 | 88.6 | 92.5 | 95.8 |

(e) Multi-scale LFE
(e) 多尺度 LFE

| Multi-scale | PQ | RQ | SQ |
| :---: | :---: | :---: | :---: |
| | 90.1 | 93.6 | 96.3 |
| ✓ | 90.7 | 94.3 | 96.1 |

(f) Positional encoding type
(f) 位置編碼類型

| Enc Type | PQ | RQ | SQ |
| :---: | :---: | :---: | :---: |
| Sine | 89.7 | 93.4 | 96.1 |
| Fourier | 90.1 | 94.6 | 95.2 |

(g) Training method
(g) 訓練方法

| Method | PQ | RQ | SQ |
| :---: | :---: | :---: | :---: |
| MPT[24] | 88.9 | 93.0 | 95.5 |
| PGT | 90.1 | 94.6 | 95.2 |

Fig. 7: (a)Comparisions of the recall of instance masks at each trainning epoch. (b) Sensitivity of hyper-parameters on the scale factor. (c) The curve of mask ce loss.
圖 7：(a) 每個訓練週期實例遮罩召回率的比較。(b) 超參數對縮放因子的敏感性。(c) 遮罩交叉熵損失曲線。

**Position Guiding Training.** In section 3.3, we introduce center queries to implicitly guide model training. As Fig. 7a shows, compared to no PGT, PGT can easily capture the objects in a scene with a higher recall in the early stages(before 100 epochs), which is crucial in reducing training difficulty and accelerating convergence. Additionally, we also conduct analysis on ablation of center query, sensitivity of scale factor, type of positional encoding, and comparison with MP-Former[24]. **1) Ablation of the center query.** We conducted extra ablation analysis on the two parts of the center query: class embedding(ClsE.) and position encoding(PosE.). The results are shown in Table. 4b. In line with[24], using only ClsE. to guide training can also bring an absolute 0.6% PQ, while using PosE. can get 1.2% PQ absolute improvement, which confirms the importance of the center position. **2) Sensitivity of scale factor.** We perturb the center query to increase diversity by introducing a scale factor to control the degree of distance between the sampling point and the instance center point. The larger the scale factor, the farther away the sampling point is from the instance center point, while the smaller the scale factor, the closer the sampling point is to the instance center point. From Fig. 7b we can see that as the sampling point gets closer to the instance center, the performance first increases and then decreases, which means that too far away is not conducive to perceive the position of the object, while too close away reduces diversity and is not conducive to learning. **3) Type of positional encodings.** We also experiment with different positional encodings used in the center query. Results can be found in Table. 4f. Sine positional encoding can achieve comparable results as Fourier positional encoding, but perhaps the latter is more suitable for our tasks. **4) Stability of mask ce loss.** Due to the huge variety in the distribution of primitives on different graphical layers, directly applying LFE module results in mask ce loss fluctuation during later stages, as the learning rate decreases, as shown in Fig. 7c. After introducing position-guided training, convergence can be accelerated in the early stages and stable training can be achieved in the later stages. **5) Comparison with MP-Former.** MP-Former[24] feeds noised GT masks and reconstructs the original ones to alleviate inconsistent optimization of mask predictions for image segmentation which bears some similarity with our PGT. We adapt it to our task. As shown in Table. 4g, our position-guided training method surpasses the mask-piloted training[24] by 2.2% and 1.6% in terms of PQ and $PQ^{Th}$, It shows that our method has a strong modeling ability for instance position.
**位置引導訓練。** 在 3.3 節中，我們引入中心查詢來隱式地引導模型訓練。如圖 7a 所示，與沒有 PGT 相比，PGT 可以在早期階段（100 個週期之前）以更高的召回率輕鬆捕捉場景中的物件，這對於降低訓練難度和加速收斂至關重要。此外，我們還對中心查詢的消融、縮放因子的敏感性、位置編碼的類型以及與 MP-Former [24] 的比較進行了分析。**1) 中心查詢的消融。** 我們對中心查詢的兩個部分進行了額外的消融分析：類別嵌入 (ClsE.) 和位置編碼 (PosE.)。結果如表 4b 所示。與 [24] 一致，僅使用 ClsE. 來引導訓練也能帶來 0.6% PQ 的絕對提升，而使用 PosE. 可以獲得 1.2% PQ 的絕對提升，這證實了中心位置的重要性。**2) 縮放因子的敏感性。** 我們通過引入縮放因子來擾動中心查詢以增加多樣性，控制採樣點與實例中心點之間的距離程度。縮放因子越大，採樣點離實例中心點越遠，而縮放因子越小，採樣點離實例中心點越近。從圖 7b 可以看出，隨著採樣點越來越接近實例中心，性能先上升後下降，這意味著太遠不利於感知物件的位置，而太近則減少了多樣性，不利於學習。**3) 位置編碼的類型。** 我們還實驗了中心查詢中使用的不同位置編碼。結果見表 4f。正弦位置編碼可以達到與傅立葉位置編碼相當的結果，但也許後者更適合我們的任務。**4) 遮罩交叉熵損失的穩定性。** 由於不同圖形圖層上基元分佈的巨大差異，直接應用 LFE 模組會導致後期遮罩交叉熵損失波動，隨著學習率的降低，如圖 7c 所示。引入位置引導訓練後，可以在早期加速收斂，並在後期實現穩定訓練。**5) 與 MP-Former 的比較。** MP-Former [24] 輸入帶噪聲的 GT 遮罩並重建原始遮罩，以緩解圖像分割中遮罩預測優化不一致的問題，這與我們的 PGT 有一些相似之處。我們將其調整以適應我們的任務。如表 4g 所示，我們的位置引導訓練方法在 PQ 和 $PQ^{Th}$ 方面分別超過遮罩引導訓練 [24] 2.2% 和 1.6%，這表明我們的方法對實例位置具有很強的建模能力。

Fig. 8: Two typical failed cases of SPv2. The red arrows highlight the key regions.
圖 8：SPv2 的兩個典型失敗案例。紅色箭頭突出了關鍵區域。

## 5 Conclusions
## 5 結論

We have presented SymPoint-V2 (SPv2), a simple yet effective approach for panoptic symbol spotting in CAD drawings. Our work makes two non-trivial improvements upon SymPoint-V1[12], including a graphical layer feature-enhanced module to integrate layer information which is laid out in CAD drawing and a position-guided training method. Our SPv2 model achieves new state-of-the-art performance on panoptic symbol spotting benchmarks.
我們提出了 SymPoint-V2 (SPv2)，這是一種簡單而有效的 CAD 圖紙全景符號識別方法。我們在 SymPoint-V1 [12] 的基礎上做了兩個不平凡的改進，包括一個用於整合 CAD 圖紙中佈局圖層資訊的圖形層特徵增強模組和一個位置引導訓練方法。我們的 SPv2 模型在全景符號識別基準測試中達到了新的技術水平 (SOTA)。

**Limitations** Our SymPoint-V2 surpasses existing state-of-the-art methods by a large margin. There are still limitations. Two failed cases are shown in Fig. 8. In some cases, simple symbols may go unrecognized or be incorrectly identified, leading to mislabeling or significant variations in the graphical representation. For example, our model spots most of the quadrilateral tables, but we still missed two tables. Future work would focus on failed cases and improve the robustness of our model.
**局限性** 我們的 SymPoint-V2 大幅超越了現有的技術水平。但仍存在局限性。圖 8 顯示了兩個失敗案例。在某些情況下，簡單的符號可能無法被識別或被錯誤識別，導致標籤錯誤或圖形表示的顯著變化。例如，我們的模型識別出了大多數四邊形桌子，但我們仍然遺漏了兩張桌子。未來的工作將集中在失敗案例上，並提高我們模型的穩健性。

## References
## 參考文獻
*(Translator's Note: References are kept in their original English format as per standard academic citation practices.)*

1. Carion, N., Massa, F., Synnaeve, G., Usunier, N., Kirillov, A., Zagoruyko, S.: End-to-end object detection with transformers. In: European conference on computer vision. pp. 213–229. Springer (2020)
2. Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., et al.: An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929 (2020)
3. Fan, Z., Chen, T., Wang, P., Wang, Z.: Cadtransformer: Panoptic symbol spotting transformer for cad drawings. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 10986–10996 (2022)
4. Fan, Z., Zhu, L., Li, H., Chen, X., Zhu, S., Tan, P.: Floorplancad: A large-scale cad drawing dataset for panoptic symbol spotting. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 10128–10137 (2021)
5. Goyal, S., Mistry, V., Chattopadhyay, C., Bhatnagar, G.: Bridge: building plan repository for image description generation, and evaluation. In: 2019 International Conference on Document Analysis and Recognition (ICDAR). pp. 1071–1076. IEEE (2019)
6. Jiang, X., Liu, L., Shan, C., Shen, Y., Dong, X., Li, D.: Recognizing vector graphics without rasterization. Advances in Neural Information Processing Systems 34, 24569–24580 (2021)
7. Kipf, T.N., Welling, M.: Semi-supervised classification with graph convolutional networks. arXiv preprint arXiv:1609.02907 (2016)
8. Kirillov, A., He, K., Girshick, R., Rother, C., Doll´ar, P.: Panoptic segmentation. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 9404–9413 (2019)
9. Lai, X., Yuan, Y., Chu, R., Chen, Y., Hu, H., Jia, J.: Mask-attention-free transformer for 3d instance segmentation. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 3693–3703 (2023)
10. Li, F., Zhang, H., Liu, S., Guo, J., Ni, L.M., Zhang, L.: Dn-detr: Accelerate detr training by introducing query denoising. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 13619–13627 (2022)
11. Liu, S., Li, F., Zhang, H., Yang, X., Qi, X., Su, H., Zhu, J., Zhang, L.: Dab-detr: Dynamic anchor boxes are better queries for detr. arXiv preprint arXiv:2201.12329 (2022)
12. Liu, W., Yang, T., Wang, Y., Yu, Q., Zhang, L.: Symbol as points: Panoptic symbol spotting via point-based representation. In: International Conference on Learning Representations (2024)
13. Meng, D., Chen, X., Fan, Z., Zeng, G., Li, H., Yuan, Y., Sun, L., Wang, J.: Conditional detr for fast training convergence. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 3651–3660 (2021)
14. Redmon, J., Farhadi, A.: Yolov3: An incremental improvement. arXiv preprint arXiv:1804.02767 (2018)
15. Ren, S., He, K., Girshick, R., Sun, J.: Faster r-cnn: Towards real-time object detection with region proposal networks. Advances in neural information processing systems 28 (2015)
16. Rezvanifar, A., Cote, M., Albu, A.B.: Symbol spotting on digital architectural floor plans using a deep learning-based framework. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops. pp. 568–569 (2020)
17. Rezvanifar, A., Cote, M., Branzan Albu, A.: Symbol spotting for architectural drawings: state-of-the-art and new industry-driven developments. IPSJ Transactions on Computer Vision and Applications 11, 1–22 (2019)
18. Sun, K., Xiao, B., Liu, D., Wang, J.: Deep high-resolution representation learning for human pose estimation. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 5693–5703 (2019)
19. Tancik, M., Srinivasan, P., Mildenhall, B., Fridovich-Keil, S., Raghavan, N., Singhal, U., Ramamoorthi, R., Barron, J., Ng, R.: Fourier features let networks learn high frequency functions in low dimensional domains. Advances in Neural Information Processing Systems 33, 7537–7547 (2020)
20. Tian, Z., Shen, C., Chen, H., He, T.: Fcos: Fully convolutional one-stage object detection. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 9627–9636 (2019)
21. Wang, Y., Zhang, X., Yang, T., Sun, J.: Anchor detr: Query design for transformer-based detector. In: Proceedings of the AAAI conference on artificial intelligence. vol. 36, pp. 2567–2575 (2022)
22. Yang, B., Jiang, H., Pan, H., Xiao, J.: Vectorfloorseg: Two-stream graph attention network for vectorized roughcast floorplan segmentation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 1358–1367 (2023)
23. Zhang, H., Li, F., Liu, S., Zhang, L., Su, H., Zhu, J., Ni, L.M., Shum, H.Y.: Dino: Detr with improved denoising anchor boxes for end-to-end object detection. arXiv preprint arXiv:2203.03605 (2022)
24. Zhang, H., Li, F., Xu, H., Huang, S., Liu, S., Ni, L.M., Zhang, L.: Mp-former: Mask-piloted transformer for image segmentation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 18074–18083 (2023)
25. Zhao, H., Jiang, L., Jia, J., Torr, P.H., Koltun, V.: Point transformer. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 16259–16268 (2021)
26. Zheng, Z., Li, J., Zhu, L., Li, H., Petzold, F., Tan, P.: Gat-cadnet: Graph attention network for panoptic symbol spotting in cad drawings. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 11747–11756 (2022)
27. Zhu, X., Su, W., Lu, L., Li, B., Wang, X., Dai, J.: Deformable detr: Deformable transformers for end-to-end object detection. arXiv preprint arXiv:2010.04159 (2020)
28. Ziran, Z., Marinai, S.: Object detection in floor plan images. In: Artificial Neural Networks in Pattern Recognition: 8th IAPR TC3 Workshop, ANNPR 2018, Siena, Italy, September 19–21, 2018, Proceedings 8. pp. 383–394. Springer (2018)

## Appendix for SymPoint Revolutionized: Boosting Panoptic Symbol Spotting with Layer Feature Enhancement
## SymPoint 的革命性進化：透過層特徵增強提升全景符號識別的附錄

Wenlong Liu, Tianyu Yang, Qizhi Yu, and Lei Zhang
International Digital Economy Academy, Vanyi Tech

## 1 PyTorch code for LFE module
## 1 LFE 模組的 PyTorch 程式碼

To demonstrate the simplicity of the LFE module, we can implement our LFE module in several lines of code when the batch size is 1, as summarized in Algorithm 1.
為了展示 LFE 模組的簡單性，當批次大小為 1 時，我們可以用幾行程式碼實現我們的 LFE 模組，如演算法 1 總結所示。

**Algorithm 1** PyTorch code for LFE module.
**演算法 1** LFE 模組的 PyTorch 程式碼。

```python
# F: primitive features tensor with a shape of (N, C)
# layerids: Layer Ids tensor with a shape of (N, )

# create a blank tensor with the same dimension as F
new_F = torch.zeros_like(F)
# do loop processing each layer
for lid in torch.unique(layerids):
    ind = torch.where(layerids==lid)[0]
    layer_point_feat = element_features[ind]
    avg_pool = torch.mean(layer_point_feat, dim=0) # mean pool
    max_pool, _ = torch.max(layer_point_feat, dim=0) # max pool
    # attention pool
    attn_w = F.softmax(self.attn(layer_point_feat), dim=0)
    w_f = torch.mul(layer_point_feat, attn_w.expand_as(layer_point_feat))
    attn_pool = torch.sum(w_f, dim=0)
    # fusion
    layerf = torch.cat((avg_pool,max_pool,attn_pool), dim=0)
    layerf = self.fc1(layerf)
    layerf = F.relu(layerf)
    layerf = self.fc2(layerf)
    layerf = layerf.unsqueeze(0).expand_as(layer_point_feat)
    # concat
    fusion = torch.cat([layer_point_feat, layerf], dim=1)
    output = self.fc3(fusion)
    new_F[ind] = output
return new_F
```

## 2 Additional Quantitative Evaluations
## 2 額外的定量評估

We present a detailed evaluation of panoptic quality(PQ), segmentation quality(SQ), and recognition quality(RQ) in Tab. 1. Here, we provide the class-wise evaluations of different setting of our methods.
我們在表 1 中展示了全景品質 (PQ)、分割品質 (SQ) 和識別品質 (RQ) 的詳細評估。在這裡，我們提供了我們方法不同設置的類別評估。

Table 1: **Quantitative results for panoptic symbol spotting** of each class. In the test split, some classes have a limited number of instances, resulting in zeros and notably low values in the results.
**表 1：每個類別的全景符號識別定量結果**。在測試分割中，某些類別的實例數量有限，導致結果為零或值明顯偏低。

| Class | SPv2 PQ | SPv2 RQ | SPv2 SQ | Baseline+PGT PQ | Baseline+PGT RQ | Baseline+PGT SQ | Baseline+LFE PQ | Baseline+LFE RQ | Baseline+LFE SQ | Baseline PQ | Baseline RQ | Baseline SQ | SPv1 PQ | SPv1 RQ | SPv1 SQ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| single door | 94.4 | 97.1 | 97.3 | 91.6 | 95.9 | 95.5 | 93.4 | 96.3 | 97.1 | 90.5 | 95.1 | 95.1 | 91.7 | 96.0 | 95.5 |
| double door | 94.5 | 97.3 | 97.1 | 91.4 | 96.3 | 94.9 | 93.9 | 96.8 | 97.0 | 90.0 | 95.3 | 94.4 | 91.5 | 96.6 | 94.7 |
| sliding door | 97.2 | 97.9 | 99.3 | 94.6 | 97.5 | 97.0 | 96.8 | 97.6 | 99.2 | 93.8 | 97.5 | 96.2 | 94.8 | 97.7 | 97.0 |
| folding door | 82.7 | 90.0 | 91.9 | 64.6 | 69.8 | 92.6 | 81.7 | 87.2 | 93.7 | 70.3 | 79.1 | 88.9 | 73.8 | 87.0 | 84.8 |
| revolving door | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| rolling door | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| window | 90.1 | 93.1 | 96.8 | 79.3 | 90.7 | 87.4 | 89.4 | 92.7 | 96.4 | 77.8 | 89.6 | 86.8 | 78.9 | 90.4 | 87.3 |
| bay window | 54.1 | 55.1 | 98.3 | 25.2 | 34.2 | 73.8 | 39.8 | 41.4 | 96.2 | 19.4 | 27.1 | 71.5 | 35.4 | 42.3 | 83.6 |
| blind window | 91.0 | 92.3 | 98.5 | 79.6 | 90.8 | 87.7 | 86.1 | 89.1 | 96.6 | 77.6 | 89.6 | 86.6 | 80.6 | 92.1 | 87.5 |
| opening symbol | 40.7 | 51.6 | 78.9 | 48.2 | 61.0 | 79.0 | 32.6 | 42.6 | 76.5 | 35.1 | 45.1 | 77.9 | 33.1 | 40.9 | 80.7 |
| sofa | 85.4 | 89.2 | 95.8 | 84.6 | 90.0 | 94.0 | 83.7 | 88.3 | 94.7 | 82.4 | 87.8 | 93.8 | 83.9 | 88.8 | 94.5 |
| bed | 90.0 | 97.5 | 92.4 | 79.5 | 90.1 | 88.3 | 86.6 | 95.1 | 91.1 | 76.9 | 87.9 | 87.5 | 86.1 | 95.9 | 89.8 |
| chair | 85.5 | 89.7 | 95.3 | 84.6 | 89.6 | 94.4 | 86.3 | 91.0 | 94.9 | 85.7 | 90.7 | 94.5 | 82.7 | 88.9 | 93.1 |
| table | 72.1 | 79.7 | 90.5 | 71.5 | 80.0 | 89.4 | 70.2 | 79.3 | 88.4 | 70.9 | 81.1 | 87.3 | 70.9 | 79.1 | 89.6 |
| TV cabinet | 95.5 | 97.7 | 97.8 | 92.3 | 96.9 | 95.2 | 92.6 | 96.5 | 96.0 | 87.1 | 95.0 | 91.6 | 90.1 | 97.0 | 92.9 |
| Wardrobe | 95.6 | 97.7 | 97.9 | 86.7 | 97.0 | 89.4 | 94.2 | 96.5 | 97.6 | 85.3 | 96.3 | 88.6 | 87.7 | 96.4 | 90.9 |
| cabinet | 81.3 | 89.7 | 90.6 | 73.5 | 86.5 | 85.0 | 78.9 | 87.2 | 90.5 | 72.8 | 85.9 | 84.8 | 73.8 | 86.2 | 85.6 |
| gas stove | 96.4 | 97.0 | 99.4 | 95.4 | 96.1 | 99.3 | 97.4 | 98.9 | 98.5 | 97.0 | 98.9 | 98.1 | 97.6 | 98.9 | 98.7 |
| sink | 89.8 | 94.2 | 95.3 | 87.2 | 93.2 | 93.6 | 88.6 | 94.1 | 94.1 | 85.5 | 92.7 | 92.2 | 86.1 | 92.9 | 92.7 |
| refrigerator | 95.3 | 96.3 | 98.9 | 94.4 | 95.9 | 98.4 | 88.7 | 96.0 | 92.3 | 87.0 | 95.4 | 91.2 | 87.8 | 95.7 | 91.8 |
| airconditioner | 88.2 | 89.1 | 98.9 | 84.2 | 88.3 | 95.4 | 88.1 | 89.2 | 98.8 | 83.4 | 87.9 | 94.9 | 80.5 | 84.4 | 95.4 |
| bath | 79.2 | 86.2 | 91.9 | 73.0 | 86.2 | 84.7 | 78.4 | 86.6 | 90.5 | 71.1 | 85.5 | 83.1 | 73.2 | 85.0 | 86.1 |
| bath tub | 87.8 | 91.4 | 96.0 | 83.5 | 90.6 | 92.2 | 86.0 | 94.2 | 91.3 | 74.3 | 90.3 | 82.4 | 76.1 | 91.4 | 83.2 |
| washing machine | 91.4 | 93.6 | 97.6 | 87.3 | 93.5 | 93.3 | 89.1 | 92.3 | 96.5 | 84.3 | 92.6 | 91.0 | 86.7 | 93.8 | 92.5 |
| urinal | 95.0 | 95.6 | 99.4 | 93.2 | 95.7 | 97.4 | 94.4 | 95.6 | 98.8 | 91.6 | 95.6 | 95.8 | 93.8 | 96.7 | 96.9 |
| squat toilet | 96.1 | 97.1 | 99.0 | 94.2 | 97.1 | 97.0 | 95.8 | 97.1 | 98.6 | 91.5 | 95.7 | 95.7 | 93.6 | 97.5 | 96.1 |
| toilet | 95.6 | 97.5 | 98.1 | 93.8 | 97.0 | 96.7 | 93.6 | 97.2 | 96.3 | 92.0 | 96.9 | 95.0 | 92.9 | 97.2 | 95.6 |
| stairs | 84.8 | 89.8 | 94.4 | 76.9 | 88.6 | 86.8 | 82.8 | 89.3 | 92.7 | 72.5 | 85.5 | 84.9 | 72.5 | 85.3 | 85.0 |
| elevator | 94.2 | 96.4 | 97.7 | 91.8 | 95.9 | 95.7 | 93.4 | 96.8 | 96.6 | 90.8 | 96.0 | 94.6 | 88.8 | 94.4 | 94.1 |
| escalator | 68.7 | 80.7 | 85.2 | 60.7 | 77.6 | 78.3 | 64.8 | 77.7 | 83.4 | 51.5 | 65.4 | 78.8 | 60.6 | 75.6 | 80.2 |
| row chairs | 88.0 | 92.4 | 95.3 | 85.7 | 90.8 | 94.4 | 84.6 | 89.4 | 94.7 | 84.5 | 89.2 | 94.7 | 84.3 | 89.2 | 94.5 |
| parking spot | 93.6 | 95.6 | 97.9 | 80.1 | 92.3 | 86.8 | 87.5 | 89.7 | 97.6 | 71.2 | 85.1 | 83.7 | 73.4 | 86.7 | 84.7 |
| wall | 83.7 | 92.5 | 90.6 | 54.6 | 79.1 | 69.0 | 82.6 | 91.2 | 90.5 | 50.8 | 75.3 | 67.4 | 53.5 | 77.5 | 69.0 |
| curtain wall | 60.0 | 70.1 | 85.6 | 41.8 | 58.2 | 71.7 | 57.8 | 68.1 | 84.9 | 39.8 | 53.8 | 74.0 | 44.2 | 60.2 | 73.5 |
| railing | 70.7 | 77.0 | 91.8 | 42.3 | 53.5 | 79.0 | 64.6 | 70.8 | 91.2 | 37.7 | 48.2 | 78.1 | 53.0 | 66.3 | 80.0 |
| **total** | **90.1** | **93.6** | **96.3** | 84.6 | 91.7 | 92.2 | 88.6 | 92.9 | 95.4 | 82.1 | 90.4 | 90.8 | 83.3 | 91.1 | 91.4 |

## 3 Additional Qualitative Evaluations
## 3 額外的定性評估

The results of additional cases are visually represented in this section, you can zoom in on each picture to capture more details, primitives belonging to different classes are represented in distinct colors. More visualized results are shown in Fig. 1 2 3.
額外案例的結果在本節中以視覺方式呈現，您可以放大每張圖片以捕捉更多細節，屬於不同類別的基元以不同的顏色表示。更多可視化結果顯示在圖 1、2、3 中。

Fig. 1: Results of SPv2 on FloorPlanCAD.
圖 1：SPv2 在 FloorPlanCAD 上的結果。

Fig. 2: Results of SPv2 on FloorPlanCAD.
圖 2：SPv2 在 FloorPlanCAD 上的結果。

Fig. 3: Results of SPv2 on FloorPlanCAD.
圖 3：SPv2 在 FloorPlanCAD 上的結果。
