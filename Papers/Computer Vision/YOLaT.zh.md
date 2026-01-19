---
title: "YOLaT"
field: "Papers"
status: "Imported"
created_date: 2026-01-19
pdf_link: "[[YOLaT.pdf]]"
tags: [paper, Papers]
---

# Recognizing Vector Graphics without Rasterization
# 無需光柵化識別向量圖形

**Xinyang Jiang**, **Lu Liu**, **Caihua Shan**, **Yifei Shen**, **Xuanyi Dong**, **Dongsheng Li**
^1 Microsoft Research Asia
^2 University of Technology Sydney
^3 The Hong Kong University of Science and Technology

## Abstract
## 摘要

In this paper, we consider a different data format for images: vector graphics. In contrast to raster graphics which are widely used in image recognition, vector graphics can be scaled up or down into any resolution without aliasing or information loss, due to the analytic representation of the primitives in the document.
在本文中，我們考慮圖像的一種不同資料格式：向量圖形。與廣泛用於圖像識別的光柵圖形（點陣圖）相比，向量圖形由於文件中圖元的解析表示，可以放大或縮小到任何解析度，而不會出現鋸齒或資訊遺失。

Furthermore, vector graphics are able to give extra structural information on how low-level elements group together to form high level shapes or structures. These merits of graphic vectors have not been fully leveraged in existing methods.
此外，向量圖形能夠提供額外的結構資訊，說明低階元素如何組合成高階形狀或結構。向量圖形的這些優點在現有方法中尚未被充分利用。

To explore this data format, we target on the fundamental recognition tasks: object localization and classification. We propose an efficient CNN-free pipeline that does not render the graphic into pixels (i.e. rasterization), and takes textual document of the vector graphics as input, called YOLaT (You Only Look at Text).
為了探索這種資料格式，我們鎖定基本的識別任務：物件定位和分類。我們提出了一種高效的無 CNN 管道，它不需要將圖形渲染為像素（即光柵化），而是將向量圖形的文字文件作為輸入，稱為 YOLaT (You Only Look at Text)。

YOLaT builds multi-graphs to model the structural and spatial information in vector graphics, and a dual-stream graph neural network is proposed to detect objects from the graph.
YOLaT 構建多重圖來模擬向量圖形中的結構和空間資訊，並提出了一種雙流圖神經網路從圖中偵測物件。

Our experiments show that by directly operating on vector graphics, YOLaT outperforms raster-graphic based object detection baselines in terms of both average precision and efficiency. Code is available at https://github.com/microsoft/YOLaT-VectorGraphicsRecognition.
我們的實驗表明，通過直接對向量圖形進行操作，YOLaT 在平均精度和效率方面均優於基於光柵圖形的物件偵測基線。程式碼發佈於 https://github.com/microsoft/YOLaT-VectorGraphicsRecognition。

# 1 Introduction
# 1 前言

Raster graphics have been commonly used for image recognition due to its easy accessibility from cameras. Most existing benchmark datasets are built upon raster graphics, from ImageNet [1] for classification to COCO [2] for object detection.
由於容易從相機獲取，光柵圖形已被普遍用於圖像識別。大多數現有的基準資料集都是建立在光柵圖形之上的，從用於分類的 ImageNet [1] 到用於物件偵測的 COCO [2]。

However, due to its pixel-based fix-sized format, raster graphics may lead to aliasing when scaling up or down by interpolation. Fields like engineering design or graphic design require a more precise way to describe visual content without aliasing when scaling (e.g., graphic designs, mechanical drafts, floorplans, diagrams, etc), so another important image format emerges, namely vector graphics.
然而，由於其基於像素的固定大小格式，光柵圖形在通過插值放大或縮小時可能會導致鋸齒。工程設計或平面設計等領域需要一種更精確的方式來描述視覺內容，以便在縮放時不會出現鋸齒（例如，平面設計、機械草圖、平面圖、圖表等），因此另一種重要的圖像格式應運而生，即向量圖形。

Vector graphics achieve this powerful feature by recording how the graphics are constructed or drawn, instead of the color bitmaps represented by pixel arrays defined in the raster graphics (first row in Figure 1).
向量圖形通過記錄圖形的構建或繪製方式來實現這一強大功能，而不是像光柵圖形那樣由像素陣列定義顏色點陣圖（圖 1 第一行）。

Specifically, vector graphics contain a set of primitives like lines, curves and circles, which is defined with parametric equations in analytic geometry and some extra attributes. As shown in Figure 1, such vector graphic is usually a document where every primitive is defined precisely and written in a line of textual command.
具體來說，向量圖形包含一組圖元，如線、曲線和圓，這些圖元是用解析幾何中的參數方程和一些額外屬性定義的。如圖 1 所示，這種向量圖形通常是一個文件，其中每個圖元都被精確定義並寫成一行文字命令。

Due to the analytic representation, with few parameters, vector graphics can represent an object at any scale or even in infinite resolution, making it potentially a lot more precise and compact image format than raster graphics. Also, instead of independent pixels, vector graphics give higher level structural information on how low-level elements like points or curves group together to form high level shapes or structures. However, this powerful and widely used data format has been rarely investigated in previous computer vision literature.
由於解析表示，向量圖形僅需少量參數即可在任何尺度甚至無限解析度下表示物件，這使其可能成為比光柵圖形更精確、更緊湊的圖像格式。此外，向量圖形不是獨立的像素，而是提供更高層次的結構資訊，說明點或曲線等低階元素如何組合成高階形狀或結構。然而，這種強大且廣泛使用的資料格式在以往的電腦視覺文獻中很少被研究。

***

**Figure 1: Difference between raster graphics (row 1) and vector graphics (row 2).**
**圖 1：光柵圖形（第 1 行）和向量圖形（第 2 行）之間的差異。**

*(Text in Figure 1)*
* Magnify raster graphics: 放大光柵圖形
* Magnify vector graphics: 放大向量圖形
* Why aliasing: 為何有鋸齒
* Why no aliasing: 為何無鋸齒
* Defined by pixel arrays: 由像素陣列定義
* Magnification: select a part of the array: 放大：選擇陣列的一部分
* Defined by different types of primitives: 由不同類型的圖元定義
* Magnification: select statistics from a part of the primitives: 放大：從部分圖元中選擇統計數據

***

To explore this data format, this paper focuses on the fundamental recognition tasks: object localization and classification, with wide applications in vector graphics related field like automatic design audit, AI aided design, design graphics retrieval, etc.
為了探索這種資料格式，本文聚焦於基本的識別任務：物件定位和分類，這在自動設計審核、AI 輔助設計、設計圖形檢索等向量圖形相關領域有廣泛應用。

Existing raster graphics based methods [3, 4, 5, 6, 7] takes pixel arrays as input and cannot be directly applied on vector graphics. There have been attempt [8] dealing with this format by rendering vector graphics into raster graphics first.
現有的基於光柵圖形的方法 [3, 4, 5, 6, 7] 以像素陣列為輸入，無法直接應用於向量圖形。曾有嘗試 [8] 通過先將向量圖形渲染為光柵圖形來處理這種格式。

However, rendering the vector graphics into raster graphics could result in a pixel array with super resolutions (e.g., thousands by thousands), which brings extremely large memory cost, and would be inefficient or even intractable for the traditional models to process.
然而，將向量圖形渲染為光柵圖形可能會產生具有超高解析度（例如數千乘數千）的像素陣列，這會帶來極大的記憶體成本，並且對於傳統模型來說處理起來效率低甚至難以處理。

On the other hand, rendering a lower resolution image causes substantial information loss, and the object bounding boxes obtained from a low resolution image could be imprecise when scaled back to the original resolution.
另一方面，渲染較低解析度的圖像會導致大量的資訊遺失，並且從低解析度圖像獲得的物件邊界框在縮放回原始解析度時可能不精確。

Furthermore, the rendering process results in a set of independent pixels and discards the high-level structural information within the primitives. Some of this information could be critical for recognition, such as corners in a shape or contours, etc.
此外，渲染過程會產生一組獨立的像素，並丟棄圖元內部的高階結構資訊。其中一些資訊對於識別可能至關重要，例如形狀的角或輪廓等。

To address these issues, we resolve the tasks on vector graphics by introducing a model that does not need rasterization and takes the textual documents of vector graphics as input, called YOLaT (You Only Look at Text).
為了解決這些問題，我們通過引入一種不需要光柵化並以向量圖形的文字文件作為輸入的模型來解決向量圖形上的任務，稱為 YOLaT (You Only Look at Text)。

Instead of rendering the vector graphics into raster graphics, we propose an efficient end-to-end pipeline which predicts objects from the raw textual definitions of primitives. YOLaT first transforms different types of primitives into a unified format. Then it constructs an undirected multi-graph to model the structural and spatial information from the unified primitives.
我們提出了一種高效的端到端管道，它不是將向量圖形渲染為光柵圖形，而是從圖元的原始文字定義中預測物件。YOLaT 首先將不同類型的圖元轉換為統一格式。然後，它構建一個無向多重圖，從統一的圖元中模擬結構和空間資訊。

Compared to rendering to raster graphics, this transformation is able to preserve more complete information. YOLaT generates object proposals directly from the vector graphics, which produces precise object bounding boxes. Finally, a dual-stream graph neural network (GNN) specifically designed for vector graphics is proposed to classify the graph contained in each proposal, with no extra regression needed for bounding box refinement.
與渲染為光柵圖形相比，這種轉換能夠保留更完整的資訊。YOLaT 直接從向量圖形生成物件候選區域 (proposals)，從而產生精確的物件邊界框。最後，提出了一種專為向量圖形設計的雙流圖神經網路 (GNN)，對每個候選區域中包含的圖進行分類，無需額外的回歸來細化邊界框。

To evaluate our pipeline over vector graphics, we use two datasets, i.e., floorplans and diagrams and show the advantages of our method over the raster graphics based object detection baselines.
為了評估我們在向量圖形上的管道，我們使用兩個資料集，即平面圖 (floorplans) 和圖表 (diagrams)，並展示了我們的方法相對於基於光柵圖形的物件偵測基線的優勢。

Without pre-training, our method consistently outperforms raster graphics based object detection baselines, with significantly higher efficiency in terms of the number of parameters and FLOPs.
在沒有預訓練的情況下，我們的方法始終優於基於光柵圖形的物件偵測基線，並且在參數數量和 FLOPs 方面效率顯著提高。

Even compared with the powerful ImageNet pretrained two-stage model, YOLaT achieves comparable performance with 25 times fewer parameters and 100 times fewer FLOPs. We also show visualizations to better demonstrate why looking at the text can capture more delicate details and predicts more accurate bounding boxes.
即使與強大的 ImageNet 預訓練兩階段模型相比，YOLaT 也能以少 25 倍的參數和少 100 倍的 FLOPs 達到相當的性能。我們還展示了視覺化結果，以更好地證明為什麼查看文字可以捕捉更精細的細節並預測更準確的邊界框。

# 2 Related Work
# 2 相關工作

**Object Detection on Raster Graphics.** Currently deep learning based object detection methods dominate the research field with the superior performance. Two-stage object detection methods first generate region proposals and classify and regress the proposals to give object predictions with a deep convolutional networks.
**光柵圖形上的物件偵測。** 目前，基於深度學習的物件偵測方法以其優越的性能主導著研究領域。兩階段物件偵測方法首先生成區域候選 (region proposals)，然後使用深度卷積神經網路對候選進行分類和回歸，以給出物件預測。

R-CNN [9] and Fast-RCNN [3] use *selective search* for proposal generations. Faster-RCNN [10] speeds up the proposal generation by introducing a region proposal network. He et al. [11] proposed Mask-RCNN, adding a segmentation branch to the detection model for instance segmentation. To train a more translation-variant backbone, Dai et al. [12] proposed F-RCN – a new prediction head with position-sensitive score maps.
R-CNN [9] 和 Fast-RCNN [3] 使用 *選擇性搜索 (selective search)* 進行候選生成。Faster-RCNN [10] 通過引入區域候選網路 (region proposal network) 加速了候選生成。He 等人 [11] 提出了 Mask-RCNN，在偵測模型中添加了一個分割分支用於實例分割。為了訓練更具平移變異性的主幹網路，Dai 等人 [12] 提出了 F-RCN——一種具有位置敏感得分圖的新預測頭。

Most two-stage object detection methods have large computation overhead of the proposal generation process, and require running a classification and regression sub-network on all the region proposals. One-stage object detection methods tackle this challenge by removing the proposal generation process and directly predict the object bounding boxes in an end-to-end fashion.
大多數兩階段物件偵測方法在候選生成過程中具有巨大的計算開銷，並且需要在所有區域候選上運行分類和回歸子網路。單階段物件偵測方法通過移除候選生成過程並以端到端的方式直接預測物件邊界框來解決這一挑戰。

Anchor-based methods like SSD [4], YOLO series [5, 13, 14, 15], RetinaNet [6] densely tile anchor boxes over the image and conduct classification and bounding box coordinate refinement on each anchor box. Recently, anchor-free methods like CornerNet [16], CenterNet[7], FCOS [17] propose to directly find object without presets anchors.
基於錨點 (Anchor-based) 的方法如 SSD [4]、YOLO 系列 [5, 13, 14, 15]、RetinaNet [6] 在圖像上密集地平鋪錨點框，並在每個錨點框上進行分類和邊界框座標細化。最近，無錨點 (anchor-free) 方法如 CornerNet [16]、CenterNet [7]、FCOS [17] 提出在沒有預設錨點的情況下直接查找物件。

**Graph Neural Networks.** GNN has become a powerful tool for machine learning on graphs. It computes a state for each node in a graph, and iteratively updates the node states according to its neighbors.
**圖神經網路。** GNN 已成為圖機器學習的強大工具。它計算圖中每個節點的狀態，並根據其鄰居迭代更新節點狀態。

Spectral approaches [18, 19] define a convolution operation in the Fourier domain. Spatial approaches [20, 19, 21] define convolutions directly on the graph. EdgeConv [22] applies GNN model for classification on 3D Cloud by taking the state difference between neighboring nodes as the input of the aggregation function. [23] further applies EdgeConv to the object detection task on 3D cloud data by integrating the GNN backbone into an anchor-free detection framework.
譜方法 [18, 19] 在傅立葉域中定義卷積運算。空間方法 [20, 19, 21] 直接在圖上定義卷積。EdgeConv [22] 將 GNN 模型應用於 3D 點雲分類，將相鄰節點之間的狀態差異作為聚合函數的輸入。[23] 進一步將 EdgeConv 應用於 3D 點雲資料的物件偵測任務，將 GNN 主幹整合到無錨點偵測框架中。

The closest GNN model to our YOLaT is EdgeConv but YOLaT has extra upgrade specifically designed for vector graphics, including edge attributes, faster inference on densely connected edges, and dual-stream structure for multi-graph.
與我們的 YOLaT 最接近的 GNN 模型是 EdgeConv，但 YOLaT 具有專為向量圖形設計的額外升級，包括邊屬性、密集連接邊上的更快推論以及用於多重圖的雙流結構。

**Online Sketch and Handwriting Recognition.** Online handwriting and drawing recognition [24, 25] handles a data form that very similar to vector graphics, which contains a sequence of discrete points. Most of these methods use sequential models to handle this problem.
**線上草圖和手寫識別。** 線上手寫和繪圖識別 [24, 25] 處理的資料形式與向量圖形非常相似，包含一系列離散點。這些方法大多數使用序列模型來處理這個問題。

For example, [26] proposes to convert the point sequences to a sequence of Bézier Curves and use a LSTM for sequential modeling. Compare to online handwriting, vector graphics contain more types of unordered shapes with more attributes and properties other than polylines, and hence need more general and non-sequential method.
例如，[26] 建議將點序列轉換為貝茲曲線序列，並使用 LSTM 進行序列建模。與線上手寫相比，向量圖形包含更多類型的無序形狀，除了折線外還有更多屬性和特性，因此需要更通用和非序列的方法。

**Vector Graphics Related Application.** One of the most common application for vector graphics is design, such as architecture, graphic design, etc. Several methods in architecture drawing recognition propose to represent symbols in a floor-plan as graphs, and use rule-based graph matching method to classify and localize symbols, such as visibility graph [27] and attributed relational graph [28, 29].
**向量圖形相關應用。** 向量圖形最常見的應用之一是設計，如建築、平面設計等。建築圖紙識別中的幾種方法提出將平面圖中的符號表示為圖，並使用基於規則的圖匹配方法來分類和定位符號，例如可見性圖 [27] 和屬性關係圖 [28, 29]。

In this paper, we propose a novel scheme that directly construct graph from vector graphics based on Bézier Curve, and the object detection is conducted based on the prediction of GNN.
在本文中，我們提出了一種新穎的方案，直接基於貝茲曲線從向量圖形構建圖，並基於 GNN 的預測進行物件偵測。

Recent years, a few works develop deep learning based methods to automatically generate vector graphics for computer aided design or converting raster graphics to vector graphcis (i.e. vectorization) [30, 31, 32, 33, 34, 35], while to the best of knowledge, our paper is the first to focus on recognition task on vector graphics. Koch et al. [36] proposes a large 3D model dataset containing analytic representations, but it lacks semantic labeling to train recognition model.
近年來，一些作品開發了基於深度學習的方法來自動生成用於電腦輔助設計的向量圖形，或將光柵圖形轉換為向量圖形（即向量化）[30, 31, 32, 33, 34, 35]，然而據我們所知，我們的論文是第一篇專注於向量圖形識別任務的論文。Koch 等人 [36] 提出了一個包含解析表示的大型 3D 模型資料集，但它缺乏用於訓練識別模型的語義標註。

# 3 Detection Model
# 3 偵測模型

In this paper, we study the problem of object detection leveraging the definitions of the vector graphics without rendering them. Here, we define the task as object localization and object classification. Specifically, the model needs to predict a set of bounding box coordinates as well as the category of the object within the bounding boxes.
在本文中，我們研究了利用向量圖形的定義而不進行渲染的物件偵測問題。在這裡，我們將任務定義為物件定位和物件分類。具體來說，模型需要預測一組邊界框座標以及邊界框內物件的類別。

In this section, we describe our proposed YOLaT, which is an end-to-end efficient pipeline taking the raw definitions of the vector graphics as the input without further rendering the graphics. Figure 2 shows the overall pipeline of YOLaT.
在本節中，我們描述了我們提出的 YOLaT，這是一個端到端的高效管道，以向量圖形的原始定義作為輸入，而無需進一步渲染圖形。圖 2 顯示了 YOLaT 的整體管道。

We convert the primitives like lines and curves as a universal format of Bézier curves. Based on the Bézier curves, un-directed multi-graphs are constructed to model both spatial and structural relationships among the key-points within a primitive and among different primitives. More details on how we build the graphs can be found in Section 3.1.
我們將線和曲線等圖元轉換為貝茲曲線的通用格式。基於貝茲曲線，構建無向多重圖來模擬圖元內關鍵點之間以及不同圖元之間的空間和結構關係。關於我們如何構建圖的更多細節可以在 3.1 節中找到。

To fully explore the vector graphics based on the multi-graph, we propose a dual-stream GNN for graph feature extraction and classification. Section 3.2 shows the detailed design of the proposed dual-stream GNN.
為了充分探索基於多重圖的向量圖形，我們提出了一種用於圖特徵提取和分類的雙流 GNN。3.2 節展示了所提出的雙流 GNN 的詳細設計。

Compared to the complex prediction head commonly used in the object detection for raster graphics, YOLaT generates precise proposal bounding boxes directly from high resolution vector graphics. Hence each sub-graph in the proposals are fed into the dual-stream GNN classifier without further correction of the box coordinates. We show how to get the potential bounding boxes and predict their objectiveness and category in Section 3.3.
與光柵圖形物件偵測中常用的複雜預測頭相比，YOLaT 直接從高解析度向量圖形生成精確的候選邊界框。因此，候選中的每個子圖都被送入雙流 GNN 分類器，而無需進一步修正框座標。我們在 3.3 節中展示如何獲取潛在的邊界框並預測其客觀性和類別。

***

**Figure 2: The overall pipeline of the proposed method.**
**圖 2：所提出方法的整體管道。**

*(Text in Figure 2)*
* Vector Graphics: 向量圖形
* Format Unification: 格式統一
* Cubic Bézier Curves: 三次貝茲曲線
* Graph Construction: 圖構建
* Multi-Graph: 多重圖
* Proposal Generation: 候選生成
* Proposals: 候選區域
* GNN Layer: GNN 層
* Stroke-wise GNN Backbone: 筆劃 GNN 主幹
* Position-wise GNN Backbone: 位置 GNN 主幹
* MLP Classifier: MLP 分類器
* Dual-Stream GNN: 雙流 GNN
* Detection Results: 偵測結果

***

## 3.1 Graph Construction
## 3.1 圖構建

**Universal formats of curves.** Compared to the raster graphics represented by pixel arrays, vector graphics have more precise representations and no loss of quality and aliasing when resizing. The vector graphics consists of primitives defined by textual commands described in parametric equations, such as lines, curves, polygons and other shapes.
**曲線的通用格式。** 與由像素陣列表示的光柵圖形相比，向量圖形具有更精確的表示，並且在調整大小時沒有品質損失和鋸齒。向量圖形由參數方程中描述的文字命令定義的圖元組成，例如線、曲線、多邊形和其他形狀。

Different primitives are described with different parametric equations. Here, like pixel in raster graphics, we want to find a unified way to describe all types of primitives. We chose Bézier Curve due to its generality and capability of modeling different shapes and curves.
不同的圖元用不同的參數方程描述。在這裡，就像光柵圖形中的像素一樣，我們希望找到一種統一的方式來描述所有類型的圖元。我們選擇貝茲曲線是因為它的通用性和模擬不同形狀和曲線的能力。

Bézier Curve is defined by a set of control points, $\{ p_0, ..., p_n \}$, where $n$ is the order of the curve and $p_i \in [n+1]$ is a 2-d vector for the coordinates of point $i$. The first point and the last point are the end points of a curve while the rest of the control points usually do not sit on the curve and provide side information instead, such as directional information and curvature statistics of the curve from $p_0$ to $p_n$.
貝茲曲線由一組控制點 $\{ p_0, ..., p_n \}$ 定義，其中 $n$ 是曲線的階數，$p_i \in [n+1]$ 是點 $i$ 座標的二維向量。第一個點和最後一個點是曲線的端點，而其餘的控制點通常不在曲線上，而是提供輔助資訊，例如從 $p_0$ 到 $p_n$ 的曲線的方向資訊和曲率統計數據。

We chose cubic Bézier Curve where $n = 3$ for the balance between modeling capability and computational complexity. Formally, the cubic Bézier curve $B$ is defined as:
我們選擇 $n = 3$ 的三次貝茲曲線，以平衡模擬能力和計算複雜度。形式上，三次貝茲曲線 $B$ 定義為：

$$B(t) = (1 - t)^3p_0 + 3(1 - t)^2tp_1 + 3(1 - t)t^2p_2 + t^3p_3, 0 \leq t \leq 1 \quad (1)$$

where $B(t)$ defines the position of a specific point at the scale rate of $t$ on the curve from $p_0$ to $p_3$. Next, we introduce our graph construction based on a set of Bézier Curves.
其中 $B(t)$ 定義了從 $p_0$ 到 $p_3$ 的曲線上比例率為 $t$ 的特定點的位置。接下來，我們介紹基於一組貝茲曲線的圖構建。

**Nodes.** To improve efficiency, we only include the points from the set of start points and end points, denoted by $\mathbb{P}$, into the collections of nodes on graphs. The rest of the control points will serve as the edge attributes as defined in the following paragraph. For a point $p$, the attributes $x$ of the corresponding node include the coordinates of the point, the RGB color value $c$ and stroke width $w$:
**節點。** 為了提高效率，我們只將起點和終點集合（記為 $\mathbb{P}$）中的點包含在圖的節點集合中。其餘的控制點將作為下一段定義的邊屬性。對於點 $p$，對應節點的屬性 $x$ 包括點的座標、RGB 顏色值 $c$ 和筆劃寬度 $w$：

$$x = \text{concat}(p^x, p^y, c, w), p \in \mathbb{P} \quad (2)$$

where $p^x$ and $p^y$ denote the coordinate value of the point $p$ along the x axis and y axis respectively. These information is defined in the vector graphic documentation.
其中 $p^x$ 和 $p^y$ 分別表示點 $p$ 沿 x 軸和 y 軸的座標值。這些資訊在向量圖形文檔中定義。

**Edges.** We design the graph as a multi-graph containing two sets of edges, namely the stroke-wise edges and the position-wise edges. These two types of edges capture the node relationships from different perspectives.
**邊。** 我們將圖設計為包含兩組邊的多重圖，即筆劃邊 (stroke-wise edges) 和位置邊 (position-wise edges)。這兩類邊從不同的角度捕捉節點關係。

*Stroke-wise edges* capture the connections defined by the stroke in the vector graphics, which refers to the actual stroke drawn between the start and end point of each Bézier curve. This type of connections represents the structures and layouts of the objects in the vector graphics. Thus, an edge is built in-between two nodes if there is a Bézier Curve linking them:
*筆劃邊* 捕捉向量圖形中筆劃定義的連接，即每條貝茲曲線起點和終點之間繪製的實際筆劃。這種類型的連接代表向量圖形中物件的結構和佈局。因此，如果有貝茲曲線連接兩個節點，則在它們之間建立一條邊：

$$\mathcal{E}_s = \{(v_i, v_j) : (v_i, v_j) \in \mathbb{S}\} \quad (3)$$

where $\mathbb{S}$ denotes the set of tuples containing the start point $v_i$ and end point $v_j$ of a Bézier Curve. Other than the connections between start and end points, other attributes of a cubic Bézier curve like curvature or other appearance are described by the off-curve control points. We use the coordinates of these off-curve control points as the attributes of the stroke-wise edges:
其中 $\mathbb{S}$ 表示包含貝茲曲線起點 $v_i$ 和終點 $v_j$ 的元組集合。除了起點和終點之間的連接外，三次貝茲曲線的其他屬性（如曲率或其他外觀）由曲線外控制點描述。我們使用這些曲線外控制點的座標作為筆劃邊的屬性：

$$x^e = \text{concat}(p^x, p^y), p \notin \mathbb{P} \quad (4)$$

The stroke-wise edges only model the long-term structural connection between the vertices based on strokes, which is irrelevant to the spatial vicinity. To further capture the spatial relationship between nodes, we generate another set of edges, called *position-wise edges*. Specifically, the position-wise edges are defined as the dense connections among nodes within a regional cluster $\mathbb{C}_k$:
筆劃邊僅根據筆劃模擬頂點之間的長期結構連接，這與空間鄰近度無關。為了進一步捕捉節點之間的空間關係，我們生成另一組邊，稱為 *位置邊*。具體來說，位置邊定義為區域簇 $\mathbb{C}_k$ 內節點之間的密集連接：

$$\mathcal{E}_p = \{(v_i, v_j) : v_i, v_j \in \mathbb{C}_k\}, k \in \{1, 2, ..., m\}. \quad (5)$$

A regional cluster is a set of nodes close to each other spatially, which can by obtained in different ways. In our implementation, we obtain regional cluster in three steps. First, given our graph representation of a vector graphic, we obtain all the connected components in the graph, based on the stroke-wise edges $\mathcal{E}_s$.
區域簇是一組在空間上彼此接近的節點，可以通過不同的方式獲得。在我們的實現中，我們分三個步驟獲得區域簇。首先，給定向量圖形的圖表示，我們基於筆劃邊 $\mathcal{E}_s$ 獲得圖中的所有連通分量。

Secondly, for each pair of connected components, obtain their expanded minimum bounding rectangles and the overlapping area of the rectangles. If the expanded area of two connected components overlap, they are spatially close and are merged to be one regional cluster $\mathbb{C}_k$. The expand length is a hyper-parameter.
其次，對於每對連通分量，獲取它們擴展的最小邊界矩形和矩形的重疊區域。如果兩個連通分量的擴展區域重疊，則它們在空間上是接近的，並合併為一個區域簇 $\mathbb{C}_k$。擴展長度是一個超參數。

## 3.2 Feature Extraction with Dual-stream GNN
## 3.2 使用雙流 GNN 進行特徵提取

In the previous section, we introduced how to generate graphs, including building nodes, two sets of edges and attributes. To analyze the proposed multi-graph, YOLaT applies a GNN network specifically designed for the graph built from vector graphics.
在上一節中，我們介紹了如何生成圖，包括構建節點、兩組邊和屬性。為了分析所提出的多重圖，YOLaT 應用了一個專為從向量圖形構建的圖設計的 GNN 網路。

Since the proposed graph is defined by two sets of edges at hand, YOLaT uses a dual-stream GNN structure where a specific GNN branch is designed to update node representations based on each type of edges. The node representations extracted by the dual-stream GNN are able to leverage the spatial and structural information in vector graphics, and can better guide the following head for the downstream tasks.
由於所提出的圖是由兩組邊定義的，YOLaT 使用雙流 GNN 結構，其中特定的 GNN 分支旨在基於每種類型的邊更新節點表示。由雙流 GNN 提取的節點表示能夠利用向量圖形中的空間和結構資訊，並能更好地指導後續的下游任務頭。

In the following section, we first elaborate on the details of both streams in our GNN. Then we introduce how to get the representation of a specific region by leveraging multi-step node representations propagation and representation fusion.
在下一節中，我們首先詳細闡述我們 GNN 中兩個流的細節。然後，我們介紹如何利用多步節點表示傳播和表示融合來獲取特定區域的表示。

**Stroke-wise Stream.** For the graphs with stroke-wise edges, inspired from [37], this GNN takes the input of the concatenation of a node representation, the difference of the representation to its neighbor node, and the attributes on the edge. At time step $t + 1$, the representations $\mathbf{h}_i^{t+1}$ for a node $i$ is updated as follows:
**筆劃流。** 對於具有筆劃邊的圖，受 [37] 啟發，該 GNN 將節點表示、該表示與其鄰居節點的差異以及邊上屬性的串接作為輸入。在時間步 $t + 1$，節點 $i$ 的表示 $\mathbf{h}_i^{t+1}$ 更新如下：

$$\mathbf{h}_i^{t+1} = f^l(\mathbf{h}_i^t) + \frac{1}{|\mathcal{N}_i^s|} \sum_{j \in \mathcal{N}_i} f^s(\text{concat}(\mathbf{h}_i^t, \mathbf{h}_j^t - \mathbf{h}_i^t, \mathbf{x}_{ij}^e)), \quad (6)$$

where the initialization is calculated as Equation 2, i.e., $\mathbf{h}^0 = \boldsymbol{x}$. $\mathbf{x}_{ij}^e$ denotes the attributes on the stroke-wise edge between node $i$ and node $j$ as defined in Equation 4. $f^l$ is a linear transformation function. $\mathcal{N}_i^s$ denotes set of nodes adjacent to the $i$-th node in terms of stroke-wise edges in the graph, and $f^s$ denotes a transformation function which consists of a linear transformation, a ReLU activation function [38] and a batch normalization layer [39] in our implementation.
其中初始化按公式 2 計算，即 $\mathbf{h}^0 = \boldsymbol{x}$。$\mathbf{x}_{ij}^e$ 表示節點 $i$ 和節點 $j$ 之間筆劃邊上的屬性，如公式 4 所定義。$f^l$ 是一個線性變換函數。$\mathcal{N}_i^s$ 表示在圖中以筆劃邊而言與第 $i$ 個節點相鄰的節點集合，$f^s$ 表示一個變換函數，在我們的實現中包含線性變換、ReLU 激活函數 [38] 和批次正規化層 [39]。

**Position-wise Stream.** Since the position-wise edges are constructed densely as described in Section 3.1, the number of position-wise edges is significantly larger than that of the stroke-wise edges. To reduce the computational cost, we design a simpler GNN model for graphs with position-wise edges. At time step $t+1$, the model only takes the representation of a node and the node representation $\mathbf{z}_i^{t+1}$ is updated by considering the neighboring transformed representation:
**位置流。** 由於位置邊是如 3.1 節所述密集構建的，因此位置邊的數量明顯大於筆劃邊的數量。為了降低計算成本，我們為具有位置邊的圖設計了一個更簡單的 GNN 模型。在時間步 $t+1$，該模型僅採用節點的表示，並且通過考慮鄰近的變換表示來更新節點表示 $\mathbf{z}_i^{t+1}$：

$$\mathbf{z}_i^{t+1} = \frac{1}{|\mathcal{N}_i^p|} \sum_{j \in \mathcal{N}_i^p \cup \{i\}} f^p(\mathbf{z}_j^t), \quad (7)$$

where $\mathcal{N}_i^p$ denotes the neighbors of node $i$ defined by the positional edges (to maintain the information from $v_i$ a self-loop for each node is added). $f^p$ is a transformation function with the same structure but untied parameters as $f^s$. The initialization of the node representation is also calculated as Equation 2, i.e., $\mathbf{z}^0 = \boldsymbol{x}$.
其中 $\mathcal{N}_i^p$ 表示由位置邊定義的節點 $i$ 的鄰居（為了保留來自 $v_i$ 的資訊，為每個節點添加了一個自環）。$f^p$ 是一個變換函數，其結構與 $f^s$ 相同但參數不共享。節點表示的初始化也按公式 2 計算，即 $\mathbf{z}^0 = \boldsymbol{x}$。

Compared to stroke-wise edge, the computation complexity of $f^p$ can be reduced significantly, because the updates of the nodes in the same regional cluster $\mathbb{C}_k$ only needs to be computed once. More details of the efficient implementation for the GNN can be found in Section 4.
與筆劃邊相比，$f^p$ 的計算複雜度可以顯著降低，因為同一區域簇 $\mathbb{C}_k$ 中的節點更新只需要計算一次。關於 GNN 高效實現的更多細節可以在第 4 節中找到。

**Representation Fusion.** The representation of a specific region in vector graphics is based on the cluster of nodes representations located within this region. Specifically, given the node representations refined by the proposed dual-stream GNN, we average the node representations over the nodes inside this region $\mathcal{V}^r$ to get a region representation and concatenate the region representations for $\mathcal{T}$ steps:
**表示融合。** 向量圖形中特定區域的表示基於位於該區域內的節點表示簇。具體來說，給定由所提出的雙流 GNN 細化的節點表示，我們對該區域 $\mathcal{V}^r$ 內的節點表示進行平均以獲得區域表示，並串接 $\mathcal{T}$ 步的區域表示：

$$\mathbf{r} = \text{concat}(\mathbf{r}_s^0, \mathbf{r}_s^1, ..., \mathbf{r}_s^T, \mathbf{r}_p^0, \mathbf{r}_p^1, ..., \mathbf{r}_p^T), \quad (8)$$

$$\mathbf{r}_s^t = \frac{1}{|\mathcal{V}^r|} \sum_{i} \mathbf{h}_i^t, i \in \mathcal{V}^r, \quad \mathbf{r}_p^t = \frac{1}{|\mathcal{V}^r|} \sum_{i} \mathbf{z}_i^t, i \in \mathcal{V}^r, \quad (9)$$

where $\mathbf{r}$ denotes the fused representation of a specific region and $\mathbf{r}_s^t, \mathbf{r}_p^t$ denote the region representation at time step $t$ from the graph with stroke-wise edges and position-wise edges, respectively.
其中 $\mathbf{r}$ 表示特定區域的融合表示，$\mathbf{r}_s^t, \mathbf{r}_p^t$ 分別表示時間步 $t$ 時來自具有筆劃邊和位置邊的圖的區域表示。

## 3.3 Prediction and Loss
## 3.3 預測和損失

Here we propose a vector graphics based proposal generation method. Given a vector graphic, we first evenly slices each regional clusters $\mathbb{C}_k$ into grids. Then, we permute all vertex pairs on the grid mesh, each of which forms the top-left and bottom-right points of a rectangle region.
在這裡，我們提出一種基於向量圖形的候選生成方法。給定一個向量圖形，我們首先將每個區域簇 $\mathbb{C}_k$ 均勻切分成網格。然後，我們對網格上的所有頂點對進行排列，每一對形成一個矩形區域的左上角和右下角點。

The nodes, edges and corresponding primitives within each rectangle region forms a proposed object, whose minimum bounding rectangle is the bounding box of the proposal. Note that proposals with size larger than a threshold is filtered. Compared to generating proposals on raster graphics, YOLaT produces much fewer negative samples, and operates at highest resolution to directly produce tightest bounding boxes around the proposed object. Hence, YOLaT requires no extra regression branch for bounding box refinement.
每個矩形區域內的節點、邊和對應的圖元形成一個候選物件，其最小邊界矩形即為候選的邊界框。請注意，大於閾值的候選會被過濾掉。與在光柵圖形上生成候選相比，YOLaT 產生的負樣本少得多，並且以最高解析度運作，直接在候選物件周圍產生最緊密的邊界框。因此，YOLaT 不需要額外的回歸分支來細化邊界框。

For each proposal, a proposal $\hat{B}$’s representation $\mathbf{r}$ is obtained by the representation fusion strategy as described in previous section, which is then fed into a multi-layer perception to predict object category. During training, we only optimize the cross-entropy loss over the prediction and $\hat{B}$’s ground truth label $y$.
對於每個候選，候選 $\hat{B}$ 的表示 $\mathbf{r}$ 通過上一節描述的表示融合策略獲得，然後將其輸入到多層感知機以預測物件類別。在訓練期間，我們僅優化預測與 $\hat{B}$ 的真實標籤 $y$ 之間的交叉熵損失。

In each image, for each ground truth object box $B_i$, its label is $y_i$. We set $y$ the same as that of the ground truth $B_i$, which has the largest Intersection over Union (IoU) with the proposal $\hat{B}$. If the largest IoU is below a threshold $\alpha$, we regard this proposal as “no object” and set its ground truth label as the total number of classes $C$ (the class index is from 0). We minimize the cross entropy loss of each proposal:
在每張圖像中，對於每個真實物件框 $B_i$，其標籤為 $y_i$。我們將 $y$ 設置為與候選 $\hat{B}$ 具有最大交並比 (IoU) 的真實框 $B_i$ 相同。如果最大 IoU 低於閾值 $\alpha$，我們將此候選視為「無物件」，並將其真實標籤設置為類別總數 $C$（類別索引從 0 開始）。我們最小化每個候選的交叉熵損失：

$$\min -\log \text{Pr}(y|\hat{B}) = \min -\log \text{Pr}(y|\mathbf{r}), \quad (10)$$

$$
y = \begin{cases}
\arg \max_{y_i \in \mathcal{Y}} \text{IoU}(\hat{B}, B_i) & \text{if } \max(\text{IoU}(\hat{B}, B_i)) >= \alpha \\
C & \text{else}
\end{cases}, \quad (11)
$$

where $\mathcal{Y}$ is the set of all ground truth labels for the boxes $\{B_i\}$.
其中 $\mathcal{Y}$ 是框 $\{B_i\}$ 的所有真實標籤的集合。

During evaluation, we regard the probability of the classification as the confidence level for this proposal and select the bounding boxes with the confidence level above a set threshold as the predictions.
在評估期間，我們將分類的概率視為該候選的置信度，並選擇置信度高於設定閾值的邊界框作為預測。

# 4 Experiments
# 4 實驗

## 4.1 Implementation Details
## 4.1 實作細節

**Architecture.** In the model used in our main results comparison, we build a two-layer GNN for both position-wise stream and stroke-wise stream with dimension of all the hidden node representations set to 64. We observe no significant performance improvement with deeper GNN due to the over-smoothing effect.
**架構。** 在我們主要結果比較中使用的模型裡，我們為位置流和筆劃流建立了一個兩層 GNN，所有隱藏節點表示的維度設置為 64。我們觀察到，由於過度平滑效應，更深的 GNN 並沒有帶來顯著的性能提升。

In our graph, the number of position-wise edges is quite large due to its full connectivity within each regional cluster. To speed up the inference, we first pre-compute the transformation function $f^p$ on each node. Then for each regional cluster, we aggregate the obtained node representations with mean-pooling. The aggregated representation for each regional cluster is then assigned to each node in the cluster as their new node representation.
在我們的圖中，由於每個區域簇內的完全連接，位置邊的數量相當大。為了加速推論，我們先在每個節點上預計算變換函數 $f^p$。然後對於每個區域簇，我們使用平均池化聚合獲得的節點表示。每個區域簇的聚合表示隨後被分配給簇中的每個節點，作為它們的新節點表示。

In this way, each node only requires one transformation operation and one mean-pooling operation. Furthermore, since the fully connected graph constructed by position-wise edges could cause severe over-smoothing problem, after run $f^p$ on each node, the mean aggregate operation is only conducted on the last layer of GNN. We use a three-layer MLP as classifier, where the dimension of middle layer output is 512 and 256.
通過這種方式，每個節點只需要一次變換操作和一次平均池化操作。此外，由於由位置邊構成的完全連接圖可能會導致嚴重的過度平滑問題，在每個節點上運行 $f^p$ 後，僅在 GNN 的最後一層進行平均聚合操作。我們使用三層 MLP 作為分類器，其中中間層輸出的維度為 512 和 256。

**Graph Construction and Proposal Generation.** Our experiments use a widely used vector graphics standards called Scalable Vector Graphics (SVG). All the primitives in SVG are first converted to cubic Bézier Curves. A circle is split equally into four parts and then each part is converted into Bézier curves. We also split curves at the intersection into multiple sub-curves to model delicate differences. For proposal generation, each region cluster is slices into a grid with 10 columns and 10 rows.
**圖構建和候選生成。** 我們的實驗使用一種廣泛使用的向量圖形標準，稱為可縮放向量圖形 (SVG)。SVG 中的所有圖元首先被轉換為三次貝茲曲線。一個圓被均分為四部分，然後每一部分轉換為貝茲曲線。我們還將交點處的曲線分割成多個子曲線，以模擬細微的差異。對於候選生成，每個區域簇被切分成 10 列 10 行的網格。

**Training.** We use Adam optimizer with a learning rate of 0.0025 and a batch size of 16. For data augmentation, we randomly translate and scale the vector graphics by at most 10% of the image width and height, and the transformed vector graphics are further rotated by a random angle. The model is trained for 200 epochs from scratch which takes around 2 hours on a Nvidia V100 graphic card.
**訓練。** 我們使用 Adam 優化器，學習率為 0.0025，批次大小為 16。對於資料增強，我們隨機平移和縮放向量圖形，幅度最多為圖像寬度和高度的 10%，變換後的向量圖形再旋轉一個隨機角度。模型從頭開始訓練 200 個 epoch，在 Nvidia V100 顯卡上大約需要 2 小時。

## 4.2 Datasets
## 4.2 資料集

We use SESYD, which is a public database containing different types of vector graphic documents, with the corresponding object detection groundtruth, produced using the 3gT system. Our experiments use the floorplans and diagrams collections.
我們使用 SESYD，這是一個包含不同類型向量圖形文件的公共資料庫，具有使用 3gT 系統生成的相應物件偵測真值。我們的實驗使用平面圖和圖表集合。

**Floorplans.** This dataset includes vector graphics for floorplans. It contains 1,000 images with totally 28,065 objects in 16 categories, e.g., armchair, tables and windows. The images are evenly divided into 10 layouts. We divide half of the layouts as the training data and the other half for validation and test. The ratio of the validation and test data is 1:9.
**平面圖 (Floorplans)。** 該資料集包含平面圖的向量圖形。它包含 1,000 張圖像，共 28,065 個物件，分為 16 個類別，例如扶手椅、桌子和窗戶。圖像平均分為 10 種佈局。我們將一半的佈局作為訓練資料，另一半用於驗證和測試。驗證資料和測試資料的比例為 1:9。

**Diagrams.** This dataset includes vector graphics for diagrams. It contains 1,000 images with totally 1,4100 objects in 21 categories, e.g., diode and resistor. There are 10 layouts and 100 images for each layout. Note that scale variance of different objects is huge in this dataset. For example, a resistor is often much smaller compared to a transistor.
**圖表 (Diagrams)。** 該資料集包含圖表的向量圖形。它包含 1,000 張圖像，共 14,100 個物件，分為 21 個類別，例如二極體和電阻。有 10 種佈局，每種佈局 100 張圖像。請注意，此資料集中不同物件的尺度差異巨大。例如，電阻通常比電晶體小得多。

We divide the training, validation and test set in a way that objects from the same category are included in both training and testing set. Thus, the dataset is split as 600, 41 and 359 images for training, validation and test stage.
我們劃分訓練、驗證和測試集的方式是，同一類別的物件同時包含在訓練和測試集中。因此，資料集被分為 600、41 和 359 張圖像，分別用於訓練、驗證和測試階段。

## 4.3 Evaluation Metric
## 4.3 評估指標

We evaluate the models in terms of both accuracy and efficiency. For accuracy, we use AP50, AP75 and mAP, where AP* represents the average precision with the intersection over union (IOU) threshold for counting as detected as 50%, and 75%. mAP is the mean of the average precision for the IOU threshold between 0.50 and 0.95.
我們在準確性和效率方面評估模型。對於準確性，我們使用 AP50、AP75 和 mAP，其中 AP* 表示平均精度，交並比 (IOU) 閾值計為已偵測的分別為 50% 和 75%。mAP 是 IOU 閾值在 0.50 和 0.95 之間的平均精度的平均值。

We also evaluate the efficiency because of the real-world requirements for real-time object detection. Specifically, we use GFLOPs (Giga (one billion) Floating point operations) and inference time for model efficiency and we also report the number of parameters to meet the scenarios of limited resources. The inference time is evaluated on a Nvidia V100.
由於即時物件偵測的現實需求，我們還評估了效率。具體來說，我們使用 GFLOPs（十億次浮點運算）和推論時間來評估模型效率，我們還報告了參數數量以滿足資源有限的場景。推論時間是在 Nvidia V100 上評估的。

## 4.4 Comparison to Baselines
## 4.4 與基線的比較

We compare YOLaT with two types of object detection methods: one-stage methods, i.e., Yolov3 [14], Yolov4 [15, 40] and its variants, RetinaNet [6], and two-stage methods, i.e., faster-rcnn with Pyramid Network (FPN) [41] and its variants. For Yolov3, the -tiny variant is a smaller model and the -spp uses Spatial Pyramid Pooling. For Yolov4, we use a scaled Yolov4 [40] with slightly more parameters and potentially much better performance called Yolov4-P5. The faster-rcnn-R*-FPN model series use backbones of different scales, with ResNet18 [42], ResNet34, ResNet50 for R18, R34, R50, respectively.
我們將 YOLaT 與兩類物件偵測方法進行比較：單階段方法，即 Yolov3 [14]、Yolov4 [15, 40] 及其變體、RetinaNet [6]，以及兩階段方法，即帶有金字塔網路 (FPN) [41] 的 faster-rcnn 及其變體。對於 Yolov3，-tiny 變體是較小的模型，-spp 使用空間金字塔池化。對於 Yolov4，我們使用 scaled Yolov4 [40]，其參數稍多，性能可能更好，稱為 Yolov4-P5。faster-rcnn-R*-FPN 模型系列使用不同規模的主幹，分別為 ResNet18 [42]、ResNet34、ResNet50 對應 R18、R34、R50。

**Table 1: Performance comparison on the floorplan dataset.**
**表 1：平面圖資料集上的性能比較。**

| 方法 (Methods) | 預訓練 (Pretrained) | AP50 (%) | AP75 (%) | mAP (%) | 推論時間 (Inference time) (ms) | 參數 (Params)(M) | GFLOPs |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Yolov3-tiny | ✗ | 75.23 | 60.97 | 53.24 | **1.2** | 8.7 | 13.0 |
| Yolov3 | ✗ | 88.24 | 80.44 | 72.98 | 8.2 | 61.6 | 155.2 |
| Yolov3-spp | ✗ | 87.38 | 79.66 | 71.61 | 8.3 | 62.7 | 156.1 |
| Yolov4 | ✗ | 93.04 | 87.48 | 79.59 | 11.7 | 70.3 | 165.5 |
| faster-rcnn-R18-FPN | ✗ | 80.91 | 71.48 | 67.32 | 58.7 | 28.4 | 126.8 |
| faster-rcnn-R34-FPN | ✗ | 80.50 | 72.18 | 65.89 | 61.9 | 38.5 | 157.3 |
| faster-rcnn-R50-FPN | ✗ | 80.31 | 73.28 | 66.53 | 73.3 | 41.4 | 165.7 |
| retinanet-R50-FPN | ✗ | 87.50 | 82.91 | 79.18 | 79.2 | 38.0 | 189.2 |
| **YOLaT (Ours)** | ✗ | **98.83** | **94.65** | **90.59** | 1.3 | **1.6** | **1.5** |
| faster-rcnn-R50-FPN | ✓ | 98.04 | 95.23 | 90.25 | 71.2 | 41.4 | 165.6 |
| Yolov3 | ✓ | 74.61 | 60.33 | 53.76 | 8.2 | 61.6 | 155.2 |

**Table 2: Performance comparison on the diagram dataset.**
**表 2：圖表資料集上的性能比較。**

| 方法 (Methods) | 預訓練 (Pretrained) | AP50 (%) | AP75 (%) | mAP (%) | 推論時間 (Inference time) (ms) | 參數 (Params)(M) | GFLOPs |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Yolov3-tiny | ✗ | 88.40 | 79.53 | 71.42 | 3.6 | 8.7 | 13.0 |
| Yolov3 | ✗ | 89.69 | 81.38 | 78.20 | 10.9 | 61.6 | 155.2 |
| Yolov3-spp | ✗ | 90.29 | 84.51 | 78.68 | 10.8 | 62.7 | 156.1 |
| Yolov4 | ✗ | 88.71 | 84.65 | 76.28 | 11.1 | 70.3 | 165.5 |
| faster-rcnn-R18-FPN | ✗ | 92.79 | 89.10 | 85.89 | 34.4 | 28.4 | 121.1 |
| faster-rcnn-R34-FPN | ✗ | 90.47 | 88.74 | 85.21 | 36.0 | 38.5 | 150.0 |
| faster-rcnn-R50-FPN | ✗ | 91.88 | 90.25 | 84.65 | 44.9 | 41.7 | 158.0 |
| retinanet-R50-FPN | ✗ | 91.33 | 83.17 | 82.79 | 47.9 | 38.0 | 179.5 |
| **YOLaT (Ours)** | ✗ | **96.63** | **94.89** | **89.67** | **2.1** | **1.6** | **2.9** |
| faster-rcnn-R50-FPN | ✓ | 95.24 | 93.57 | 90.76 | 40.0 | 41.7 | 157.9 |
| Yolov3 | ✓ | 90.11 | 84.68 | 79.55 | 8.2 | 61.6 | 155.2 |

**Table 3: Ablation study and variant analysis on the floorplan dataset.**
**表 3：平面圖資料集上的消融研究和變體分析。**

(a) Ablation study on graph construction.
(a) 圖構建的消融研究。

| 方法 (Methods) | AP50(%) | AP75(%) | mAP(%) |
| :--- | :---: | :---: | :---: |
| **YOLaT** | 98.83 | 94.65 | 90.59 |
| w/o $\mathcal{E}_p$ | 95.81 | 91.03 | 87.17 |
| w/o $\mathcal{E}_s$ | 91.57 | 91.22 | 86.00 |
| w/o $\mathbf{x}_{ij}^e$ | 94.57 | 90.76 | 86.25 |

(b) Ablation study and variant analysis on GNN model.
(b) GNN 模型的消融研究和變體分析。

| | 方法 (Methods) | AP50(%) | AP75(%) | mAP(%) |
| :--- | :--- | :---: | :---: | :---: |
| **stroke-wise** | w/o $\mathbf{h}_i^t$ | 96.83 | 93.19 | 88.40 |
| **stream** | w/o $\mathbf{h}_j^t - \mathbf{h}_i^t$ | 95.87 | 92.90 | 87.83 |
| **position-wise** | early aggregate | 95.82 | 91.64 | 87.90 |
| **stream** | with $\mathbf{h}_j^t - \mathbf{h}_i^t$ | 98.67 | 94.46 | 90.39 |
| **aggregation** | GCN | 90.36 | 88.02 | 83.32 |
| **function** | GAT | 91.20 | 89.46 | 83.92 |
| | GraphSage | 92.70 | 91.17 | 85.26 |

We choose these baselines because they are the most popular methods in object detection. On both datasets, YOLaT outperforms all baselines without pretraining on ImageNet in terms of precision and efficiency as shown in Table 1 and Table 2.
我們選擇這些基線是因為它們是物件偵測中最流行的方法。在兩個資料集上，YOLaT 在精度和效率方面均優於所有未在 ImageNet 上預訓練的基線，如表 1 和表 2 所示。

We also include a baseline, i.e., faster-rcnn-R50-FPN which is pretrained on ImageNet, YOLaT shows competitive precision with around 100× less FLOPs and around 25× less model parameters. We also train Yolov3 with ImageNet pretrained backbone, but do not observe performance improvement. We conduct 3 rounds of experiments with different random seeds and the standard error in terms of AP50 is 0.0003 on floorplan and 0.0008 on diagram.
我們還包括一個基線，即在 ImageNet 上預訓練的 faster-rcnn-R50-FPN，YOLaT 顯示出具有競爭力的精度，而 FLOPs 少約 100 倍，模型參數少約 25 倍。我們也用 ImageNet 預訓練的主幹訓練 Yolov3，但未觀察到性能提升。我們進行了 3 輪不同隨機種子的實驗，AP50 的標準誤差在平面圖上為 0.0003，在圖表上為 0.0008。

For Yolov3, we use the implementation of ultralytics [43]. For Yolov4 we use the official pytorch implementation of Scaled Yolov4 [40]. Note that both the Yolov3 and Yolov4 implementation shows superior performance on COCO [2] when without ImageNet pretraining. For faster-rcnn and retina-net, we use the Detectron2 [44] library. For the non-pretrained model, we use the strategies of replacing Batch normalization to Group Normalization following [45] to improve the performance.
對於 Yolov3，我們使用 ultralytics [43] 的實作。對於 Yolov4，我們使用 Scaled Yolov4 [40] 的官方 pytorch 實作。請注意，Yolov3 和 Yolov4 的實作在沒有 ImageNet 預訓練的情況下在 COCO [2] 上顯示出優越的性能。對於 faster-rcnn 和 retina-net，我們使用 Detectron2 [44] 函式庫。對於非預訓練模型，我們使用將批次正規化替換為群組正規化的策略，遵循 [45] 以提高性能。

**Broader Impact.** Our YOLaT model may present a promising solution for applications that have the input of vector graphics. Any deployment of the proposed model however should be preceded by an analysis of the potential biases captured by the dataset sources used for training and the correction of any such undesirable biases captured by the pre-trained backbones and model.
**更廣泛的影響。** 我們的 YOLaT 模型可能為具有向量圖形輸入的應用提供有前景的解決方案。然而，在部署所提出的模型之前，應先分析用於訓練的資料集來源所捕捉的潛在偏差，並糾正預訓練主幹和模型所捕捉的任何此類不良偏差。

## 4.5 Ablation Study and Variants Analysis
## 4.5 消融研究和變體分析

**Graph Construction.** We analyze the effectiveness of the Bézier based graph construction method in YOLaT on SYSED-Floorplans dataset. Table 3a shows the results of the ablation study on position-wise edges $\mathcal{E}_p$ (as defined in Eq. 7), stroke-wise edges $\mathcal{E}_s$ (as defined in Eq. 6) and edge attributes $\mathbf{x}_{ij}^e$ (as defined in Eq. 2). The ablation of these components show a significant drop in precision.
**圖構建。** 我們在 SYSED-Floorplans 資料集上分析了 YOLaT 中基於貝茲的圖構建方法的有效性。表 3a 顯示了關於位置邊 $\mathcal{E}_p$（如公式 7 定義）、筆劃邊 $\mathcal{E}_s$（如公式 6 定義）和邊屬性 $\mathbf{x}_{ij}^e$（如公式 2 定義）的消融研究結果。這些組件的消除去除顯示精度顯著下降。

**Dual-Stream GNN.** As show in Table 3b, we conduct several experiments to analyze the effectiveness of our dual-stream GNN that is specifically designed for vector graphics recognition. We did the ablation of YOLaT without the input $\mathbf{h}_i^t$ and $\mathbf{h}_j^t - \mathbf{h}_i^t$.
**雙流 GNN。** 如表 3b 所示，我們進行了幾項實驗來分析專為向量圖形識別設計的雙流 GNN 的有效性。我們在沒有輸入 $\mathbf{h}_i^t$ 和 $\mathbf{h}_j^t - \mathbf{h}_i^t$ 的情況下對 YOLaT 進行了消融。

For position-wise stream, feature aggregation for position-wise edges is conducted on every layer in GNN, instead of only last layer. the experiment results show that early aggregation hurts the performance, due to the over-smoothing caused by fast message passing along the fully connected edges.
對於位置流，位置邊的特徵聚合在 GNN 的每一層進行，而不僅僅是最後一層。實驗結果表明，早期聚合會損害性能，這是由於沿著完全連接的邊快速傳遞訊息導致的過度平滑。

Due to the high computation complexity of aggregation function on fully connected position-wise edges, YOLaT discards neighboring feature difference in $f^p$. This experiment shows that there is no obvious performance improvement by adding neighboring feature difference on $f^p$. Meanwhile, this method significantly increases the computation complexity by and increase GFLOPs by almost 60% from 1.5 to 2.4.
由於在完全連接的位置邊上的聚合函數計算複雜度很高，YOLaT 在 $f^p$ 中捨棄了鄰居特徵差異。本實驗表明，在 $f^p$ 上增加鄰居特徵差異並沒有明顯的性能提升。同時，該方法顯著增加了計算複雜度，GFLOPs 增加了近 60%，從 1.5 增加到 2.4。

The last three rows of Figure 3b shows the performance comparison between YOLaT and some other popular GNN aggregation methods. In this experiments, we replace our proposed aggregation function with the aggregation functions in GCN [19], GAT [21] and GraphSage [20]. Since some of these methods do not directly support edge attributes, similar to our dual-stream GNN, we treat it as extra dimensions of features of a pair of adjacent nodes. The experiment shows that our GNN outperforms existing GNN methods, which further verifies the effectiveness of our vector graphics specific design.
表 3b 的最後三行顯示了 YOLaT 與其他一些流行的 GNN 聚合方法之間的性能比較。在這些實驗中，我們用 GCN [19]、GAT [21] 和 GraphSage [20] 中的聚合函數替換了我們提出的聚合函數。由於這些方法中有些不直接支持邊屬性，類似於我們的雙流 GNN，我們將其視為一對相鄰節點特徵的額外維度。實驗表明，我們的 GNN 優於現有的 GNN 方法，這進一步驗證了我們針對向量圖形設計的有效性。

## 4.6 Visualizations
## 4.6 視覺化

We visualize the detection results for Yolov3 and YOLaT as in Figure 3. The prediction results in first two columns show that the bounding box predicted by Yolo is imprecise while the bounding box predicted by YOLaT is precise and sits exactly at the border of every object.
我們將 Yolov3 和 YOLaT 的偵測結果視覺化，如圖 3 所示。前兩列的預測結果顯示，Yolo 預測的邊界框不精確，而 YOLaT 預測的邊界框精確且正好位於每個物件的邊界上。

For example, both models generate a bounding box for the table in the middle of the figure, while YOLaT outputs tighter box for the object border. This is because YOLaT directly looks at the text and leverages the information of where the positions of the curves are, while Yolo only leverages the lower resolution pixel arrays rendered from the text.
例如，兩個模型都為圖中間的桌子生成了一個邊界框，而 YOLaT 為物件邊界輸出了更緊密的框。這是因為 YOLaT 直接查看文字並利用曲線位置的資訊，而 Yolo 僅利用從文字渲染的低解析度像素陣列。

The imprecise predictions can affect the performance for higher standard detection which is reflected as the AP with higher IOU. This is why the gap between Yolo and YOLaT is bigger for mAP compared to that for AP50 as shown in Table 1. Also, Yolo gives more undetected cases under a strict threshold, such as the armchair and sofa as in Figure 3.
不精確的預測會影響更高標準偵測的性能，這反映在具有更高 IOU 的 AP 上。這就是為什麼 Yolo 和 YOLaT 在 mAP 上的差距比 AP50 更大，如表 1 所示。此外，在嚴格的閾值下，Yolo 給出了更多未偵測到的情況，例如圖 3 中的扶手椅和沙發。

The third column on Figure 3 shows that Yolov3 fails to distinguish the object details (the direction of arrows in different types of transistors) due to the limited resolution of raster graphics, while by directly operating on the vector graphics with each primitive precisely described by textual command, YOLaT is able to capture the details at very small scale.
圖 3 的第三列顯示，由於光柵圖形的解析度有限，Yolov3 無法區分物件細節（不同類型電晶體中的箭頭方向），而通過直接對向量圖形進行操作，每個圖元都由文字命令精確描述，YOLaT 能夠在非常小的尺度上捕捉細節。

***

**Figure 3: Visualizations of Yolov3 (upper line) and YOLaT (lower line)**
**圖 3：Yolov3（上行）和 YOLaT（下行）的視覺化**

*(Text in Figure 3)*
* Missing objects: 遺漏物件
* Wrong predictions: 錯誤預測
* Imprecise box: 不精確的框
* Struggled with details: 在細節上掙扎
* precise box: 精確的框
* Capture details: 捕捉細節
* show that 1) [Left] Yolov3 has more missing objects (shaded boxes in YOLaT figure) and wrong predictions (shaded boxes in Yolov3 figure).
* 顯示 1) [左] Yolov3 有更多遺漏物件（YOLaT 圖中的陰影框）和錯誤預測（Yolov3 圖中的陰影框）。
* 2) [Mid] The prediction boxes of YOLaT are tighter and more accurate.
* 2) [中] YOLaT 的預測框更緊密、更準確。
* 3) [Right] Yolov3 can not distinguish the details of transistors, e.g., the direction of the arrows, leading to wrong predictions.
* 3) [右] Yolov3 無法區分電晶體的細節，例如箭頭的方向，導致錯誤預測。

***

# 5 Conclusions
# 5 結論

We propose an efficient CNN-free pipeline does not need rasterization called YOLaT(You Only Look at Text). YOLaT builds a unified representations for all primitives in a vector graphic with un-directed multi-graph and detect objects with a dual-stream GNN specifically designed for vector graphics.
我們提出了一種不需要光柵化的高效無 CNN 管道，稱為 YOLaT (You Only Look at Text)。YOLaT 使用無向多重圖為向量圖形中的所有圖元構建統一表示，並使用專為向量圖形設計的雙流 GNN 偵測物件。

The experiments show that YOLaT outperforms both one-stage and two-stage deep learning methods with much better efficiency. Our work provides a new direction for recognition on vector graphics, and is able to draw more researchers’ attention on exploring the merits of vector graphics.
實驗表明，YOLaT 在效率方面優於單階段和兩階段深度學習方法。我們的工作為向量圖形識別提供了一個新方向，並能夠吸引更多研究人員關注探索向量圖形的優點。

In the future, there is much work to further improve YOLaT and recognition on vector graphics in general, such as leveraging both vector graphic and raster graphic based methods, building a GNN model for vector graphics that supports deeper structure, large vector graphics dataset to support backbone pre-training.
未來，還有許多工作可以進一步改進 YOLaT 和向量圖形識別，例如利用基於向量圖形和光柵圖形的方法，為向量圖形構建支持更深層結構的 GNN 模型，以及支持主幹預訓練的大型向量圖形資料集。

# References
# 參考文獻

[1] J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei, “Imagenet: A large-scale hierarchical image database,” in CVPR, 2009.
[2] T.-Y. Lin, M. Maire, S. Belongie, J. Hays, P. Perona, D. Ramanan, P. Dollár, and C. L. Zitnick, “Microsoft coco: Common objects in context,” in ECCV, 2014.
[3] R. Girshick, “Fast r-cnn,” in ICCV, 2015.
[4] W. Liu, D. Anguelov, D. Erhan, C. Szegedy, S. Reed, C.-Y. Fu, and A. C. Berg, “Ssd: Single shot multibox detector,” in ECCV, 2016.
[5] J. Redmon, S. Divvala, R. Girshick, and A. Farhadi, “You only look once: Unified, real-time object detection,” in CVPR, 2016.
[6] T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollár, “Focal loss for dense object detection,” in ICCV, 2017.
[7] X. Zhou, D. Wang, and P. Krähenbühl, “Objects as points,” in arXiv preprint arXiv:1904.07850, 2019.
[8] A. Rezvanifar, M. Cote, and A. B. Albu, “Symbol spotting on digital architectural floor plans using a deep learning-based framework,” in CVPR-W, 2020.
[9] R. Girshick, J. Donahue, T. Darrell, and J. Malik, “Rich feature hierarchies for accurate object detection and semantic segmentation,” in CVPR, 2014.
[10] S. Ren, K. He, R. Girshick, and J. Sun, “Faster r-cnn: towards real-time object detection with region proposal networks,” IEEE TPAMI, 2016.
[11] K. He, G. Gkioxari, P. Dollár, and R. Girshick, “Mask r-cnn,” in ICCV, 2017.
[12] J. Dai, Y. Li, K. He, and J. Sun, “R-fcn: object detection via region-based fully convolutional networks,” in NeurIPS, 2016.
[13] J. Redmon and A. Farhadi, “Yolo9000: better, faster, stronger,” in CVPR, 2017.
[14] J. Redmon and A.Farhadi, “Yolov3: An incremental improvement,” arXiv preprint arXiv:1804.02767, 2018.
[15] A. Bochkovskiy, C.-Y. Wang, and H.-Y. M. Liao, “Yolov4: Optimal speed and accuracy of object detection,” arXiv preprint arXiv:2004.10934, 2020.
[16] H. Law and J. Deng, “Cornernet: Detecting objects as paired keypoints,” in ECCV, 2018.
[17] Z. Tian, C. Shen, H. Chen, and T. He, “Fcos: Fully convolutional one-stage object detection,” in ICCV, 2019.
[18] J. Bruna, W. Zaremba, A. Szlam, and Y. Lecun, “Spectral networks and locally connected networks on graphs,” in ICLR, 2014.
[19] T. N. Kipf and M. Welling, “Semi-supervised classification with graph convolutional networks,” in ICLR, 2017.
[20] W. L. Hamilton, Z. Ying, and J. Leskovec, “Inductive representation learning on large graphs,” in NeurIPS, 2017.
[21] P. Veličković, G. Cucurull, A. Casanova, A. Romero, P. Lio, and Y. Bengio, “Graph attention networks,” arXiv preprint arXiv:1710.10903, 2017.
[22] Y. Wang, Y. Sun, Z. Liu, S. E. Sarma, M. M. Bronstein, and J. M. Solomon, “Dynamic graph cnn for learning on point clouds,” ACM TOG, 2019.
[23] W. Shi and R. Rajkumar, “Point-gnn: Graph neural network for 3d object detection in a point cloud,” in CVPR, 2020.
[24] D. Ha and D. Eck, “A neural representation of sketch drawings,” arXiv preprint arXiv:1704.03477, 2017.
[25] E. Aksan, T. Deselaers, A. Tagliasacchi, and O. Hilliges, “Cose: Compositional stroke embeddings,” in NeurIPS, H. Larochelle, M. Ranzato, R. Hadsell, M. F. Balcan, and H. Lin, Eds., vol. 33, 2020.
[26] V. Carbune, P. Gonnet, T. Deselaers, H. A. Rowley, A. Daryin, M. Calvo, L.-L. Wang, D. Keysers, S. Feuz, and P. Gervais, “Fast multi-language lstm-based online handwriting recognition,” IJDAR, vol. 23, no. 2, pp. 89–102, 2020.
[27] H. Locteau, S. Adam, E. Trupin, J. Labiche, and P. Héroux, “Symbol spotting using full visibility graph representation,” in Workshop on Graphics Recognition, 2007.
[28] J.-Y. Ramel, N. Vincent, and H. Emptoz, “A structural representation for understanding line drawing images,” IJDAR, 2000.
[29] K. Santosh, B. Lamiroy, and L. Wendling, “Symbol recognition using spatial relations,” Pattern Recognition Letters, 2012.
[30] A. Carlier, M. Danelljan, A. Alahi, and R. Timofte, “Deepsvg: A hierarchical generative network for vector graphics animation,” in NeurIPS, 2020.
[31] Y. Ganin, S. Bartunov, Y. Li, E. Keller, and S. Saliceti, “Computer-aided design as language,” arXiv preprint arXiv:2105.02769, 2021.
[32] P. Reddy, M. Gharbi, M. Lukac, and N. J. Mitra, “Im2vec: Synthesizing vector graphics without vector supervision,” arXiv preprint arXiv:2102.02798, 2021.
[33] R. G. Lopes, D. Ha, D. Eck, and J. Shlens, “A learned representation for scalable vector graphics,” in ICCV, 2019, pp. 7930–7939.
[34] I.-C. Shen and B.-Y. Chen, “Clipgen: A deep generative model for clipart vectorization and synthesis,” IEEE Transactions on Visualization and Computer Graphics, 2021.
[35] A. D. Parakkat, M.-P. R. Cani, and K. Singh, “Color by numbers: Interactive structuring and vectorization of sketch imagery,” in Proceedings of the 2021 CHI Conference on Human Factors in Computing Systems, 2021, pp. 1–11.
[36] S. Koch, A. Matveev, Z. Jiang, F. Williams, A. Artemov, E. Burnaev, M. Alexa, D. Zorin, and D. Panozzo, “Abc: A big cad model dataset for geometric deep learning,” in CVPR, 2019.
[37] G. Li, M. Muller, A. Thabet, and B. Ghanem, “Deepgcns: Can gcns go as deep as cnns?” in ICCV, 2019.
[38] X. Glorot, A. Bordes, and Y. Bengio, “Deep sparse rectifier neural networks,” in AISTATS, 2011.
[39] S. Ioffe and C. Szegedy, “Batch normalization: Accelerating deep network training by reducing internal covariate shift,” in ICML, 2015.
[40] C.-Y. Wang, A. Bochkovskiy, and H.-Y. M. Liao, “Scaled-yolov4: Scaling cross stage partial network,” arXiv preprint arXiv:2011.08036, 2020.
[41] T.-Y. Lin, P. Dollár, R. Girshick, K. He, B. Hariharan, and S. Belongie, “Feature pyramid networks for object detection,” in CVPR, 2017.
[42] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” in CVPR, 2016.
[43] G. Jocher, Y. Kwon, guigarfr, perry0418, J. Veitch-Michaelis, Ttayu, D. Suess, F. Baltacı, G. Bianconi, IlyaOvodov, Marc, e96031413, C. Lee, D. Kendall, Falak, F. Reveriano, FuLin, GoogleWiki, J. Nataprawira, J. Hu, LinCoce, LukeAI, NanoCode012, NirZarrabi, O. Reda, P. Skalski, SergioSanchezMontesUAM, S. Song, T. Havlik, and T. M. Shead, “ultralytics/yolov3: v9.5.0 - YOLOv5 v5.0 release compatibility update for YOLOv3,” 2021.
[44] Y. Wu, A. Kirillov, F. Massa, W.-Y. Lo, and R. Girshick, “Detectron2,” https://github.com/facebookresearch/detectron2, 2019.
[45] K. He, R. Girshick, and P. Dollár, “Rethinking imagenet pre-training,” in ICCV, 2019.

# Appendix for Recognizing Vector Graphics without Rasterization
# 無需光柵化識別向量圖形的附錄

## A Hyper-parameters
## A 超參數

**Number of GNN Layers.**
**GNN 層數。**

Table 1 shows how the number of layers in GNN influence the model performance. We observe that even a GNN with few layers can achieve satisfactory performance. Increasing layers in our GNN also does not bring very significant performance improvement or even hurts the performance due to over-smoothing.
表 1 顯示了 GNN 中的層數如何影響模型性能。我們觀察到，即使是層數很少的 GNN 也能達到令人滿意的性能。增加 GNN 中的層數並沒有帶來非常顯著的性能提升，甚至由於過度平滑而損害性能。

***

**Figure 1: AP50 comparison with different layers of GNN on floorplan dataset.**
**圖 1：平面圖資料集上不同 GNN 層數的 AP50 比較。**

*(Text in Figure 1)*
* MAP on SYSED-Floormap: SYSED-Floormap 上的 MAP
* Number of Proposals: 候選數量
* We analyze how density of generated proposals affects the performance by split the multi-graph spatially into mesh grid with different number of strides.
* 我們分析了生成的候選密度如何影響性能，方法是將多重圖在空間上分割成具有不同步幅數的網格。
* More strides in grid and more proposals bring higher detection performance. When the number of strides is over 10, the improvement is insignificant compared to the computational cost.
* 網格中的步幅越多，候選越多，偵測性能越高。當步幅數超過 10 時，與計算成本相比，改進微不足道。

***

**Table 1: The performance comparison with different number of strides in the mesh grid for proposal generation on diagram dataset**
**表 1：圖表資料集上候選生成網格中不同步幅數的性能比較**

| 步幅數 (Number of Strides) | AP50(%) | AP70(%) | mAP(%) | 平均候選數 (Average Number of Proposals) | GFLOPs |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 3 | 89.35 | 85.47 | 82.53 | 144.8 | 0.4 |
| 5 | 94.34 | 90.53 | 85.94 | 327.4 | 1.0 |
| 10 | 96.63 | 94.89 | 89.67 | 959.9 | 2.9 |
| 15 | 96.96 | 95.01 | 89.64 | 1394.4 | 4.2 |
| 20 | 97.02 | 95.46 | 89.84 | 1667.8 | 5.0 |

## B Bézier curves Conversion
## B 貝茲曲線轉換

A vector graphic file contains multiple lines of textual commands, and each line defines a specific primitive/shape in the image, including the shape type (e.g., circle, line, Bézier curve, etc) and its associated parameters (e.g., start/end/center point coordinates).
向量圖形文件包含多行文字命令，每一行定義圖像中的一個特定圖元/形狀，包括形狀類型（例如，圓、線、貝茲曲線等）及其相關參數（例如，起點/終點/中心點座標）。

After parsing the shape category and parameters from the command, each shape (or a part of the shape) can be converted into a Bézier curve with a closed-formed expression. Here we take circle as an example.
從命令中解析出形狀類別和參數後，每個形狀（或形狀的一部分）都可以轉換為具有閉合形式表達式的貝茲曲線。這裡我們以圓為例。

A circle is split into four equal sections, ie., left-up quarter, left-bottom quarter, right-up quarter and right-bottom quarter, and each is converted to a Bézier curve. For a circle centered at the origin with radius 1, and the right-up quarter start at (0, 1) and end at (1, 0), the control points of the corresponding Bézier curve can be obtained by:
一個圓被分成四個相等的部分，即左上、左下、右上和右下四分之一，每一部分都被轉換為貝茲曲線。對於以原點為中心、半徑為 1 的圓，右上四分之一從 (0, 1) 開始到 (1, 0) 結束，相應貝茲曲線的控制點可以通過以下方式獲得：

$$P_0 = (0, 1), P_1 = (c, 1), P_2 = (1, c), P_3 = (1, 0), c = 0.551915024494 \quad (1)$$

which gives a maximum radial error to the original circle less than 0.02
這使得與原始圓的最大徑向誤差小於 0.02。

Due to page limit, we only briefly introduce it in Section 5.1 Line 237-241. More details can be found in the source code in the supplemental materials and will be added into the appendix.
由於篇幅限制，我們僅在 5.1 節第 237-241 行中簡要介紹了它。更多細節可以在補充材料的原始碼中找到，並將添加到附錄中。

## C Visualizations
## C 視覺化

We provide more visualizations as in Figure 2 to shed lights on how our models do prediction.
我們提供了更多如圖 2 所示的視覺化效果，以闡明我們的模型是如何進行預測的。

***

**Figure 2: Visualization of predictions by YOLaT.**
**圖 2：YOLaT 的預測視覺化。**
