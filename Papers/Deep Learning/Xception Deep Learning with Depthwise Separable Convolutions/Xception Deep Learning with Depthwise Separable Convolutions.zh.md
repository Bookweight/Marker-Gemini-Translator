---
title: Xception Deep Learning with Depthwise Separable Convolutions
field: Deep_Learning
status: Imported
created_date: 2026-01-19
pdf_link: "[[Xception Deep Learning with Depthwise Separable Convolutions.pdf]]"
tags:
  - paper
  - Deep_learning
---

# Xception: Deep Learning with Depthwise Separable Convolutions
# Xception：基於深度可分離卷積的深度學習

**François Chollet**
**Google, Inc.**
`fchollet@google.com`

## Abstract
## 摘要

We present an interpretation of Inception modules in convolutional neural networks as being an intermediate step in-between regular convolution and the depthwise separable convolution operation (a depthwise convolution followed by a pointwise convolution).
我們提出了一種對卷積神經網路中 Inception 模組的解釋，將其視為介於常規卷積和深度可分離卷積操作（深度卷積後接逐點卷積）之間的中間步驟。

In this light, a depthwise separable convolution can be understood as an Inception module with a maximally large number of towers.
從這個角度來看，深度可分離卷積可以被理解為具有最大數量塔（towers）的 Inception 模組。

This observation leads us to propose a novel deep convolutional neural network architecture inspired by Inception, where Inception modules have been replaced with depthwise separable convolutions.
這一觀察促使我們提出了一種受 Inception 啟發的新型深度卷積神經網路架構，其中 Inception 模組已被深度可分離卷積所取代。

We show that this architecture, dubbed Xception, slightly outperforms Inception V3 on the ImageNet dataset (which Inception V3 was designed for), and significantly outperforms Inception V3 on a larger image classification dataset comprising 350 million images and 17,000 classes.
我們證明，這種被稱為 Xception 的架構在 ImageNet 資料集（Inception V3 專為此設計）上略優於 Inception V3，並且在包含 3.5 億張圖像和 17,000 個類別的更大型圖像分類資料集上顯著優於 Inception V3。

Since the Xception architecture has the same number of parameters as Inception V3, the performance gains are not due to increased capacity but rather to a more efficient use of model parameters.
由於 Xception 架構的參數數量與 Inception V3 相同，因此性能的提升並非歸因於容量的增加，而是歸因於模型參數更有效率的使用。

## 1. Introduction
## 1. 介紹

Convolutional neural networks have emerged as the master algorithm in computer vision in recent years, and developing recipes for designing them has been a subject of considerable attention.
近年來，卷積神經網路已成為電腦視覺領域的主流演算法，而開發設計它們的配方一直是備受關注的主題。

The history of convolutional neural network design started with LeNet-style models [10], which were simple stacks of convolutions for feature extraction and max-pooling operations for spatial sub-sampling.
卷積神經網路設計的歷史始於 LeNet 風格的模型 [10]，它們是用於特徵提取的卷積層和用於空間下採樣的最大池化操作的簡單堆疊。

In 2012, these ideas were refined into the AlexNet architecture [9], where convolution operations were being repeated multiple times in-between max-pooling operations, allowing the network to learn richer features at every spatial scale.
在 2012 年，這些想法被改進為 AlexNet 架構 [9]，其中卷積操作在最大池化操作之間重複多次，使網路能夠在每個空間尺度上學習更豐富的特徵。

What followed was a trend to make this style of network increasingly deeper, mostly driven by the yearly ILSVRC competition; first with Zeiler and Fergus in 2013 [25] and then with the VGG architecture in 2014 [18].
隨後出現了一種趨勢，即主要由年度 ILSVRC 競賽推動，使這種風格的網路變得越來越深；首先是 2013 年的 Zeiler 和 Fergus [25]，然後是 2014 年的 VGG 架構 [18]。

At this point a new style of network emerged, the Inception architecture, introduced by Szegedy et al. in 2014 [20] as GoogLeNet (Inception V1), later refined as Inception V2 [7], Inception V3 [21], and most recently Inception-ResNet [19].
此時出現了一種新風格的網路，即 Inception 架構，由 Szegedy 等人在 2014 年以 GoogLeNet (Inception V1) [20] 的形式引入，後來改進為 Inception V2 [7]、Inception V3 [21]，以及最近的 Inception-ResNet [19]。

Inception itself was inspired by the earlier Network-In-Network architecture [11].
Inception 本身是受到早期 Network-In-Network 架構 [11] 的啟發。

Since its first introduction, Inception has been one of the best performing family of models on the ImageNet dataset [14], as well as internal datasets in use at Google, in particular JFT [5].
自首次推出以來，Inception 一直是 ImageNet 資料集 [14] 以及 Google 內部使用的資料集（特別是 JFT [5]）上表現最好的模型家族之一。

The fundamental building block of Inception-style models is the Inception module, of which several different versions exist.
Inception 風格模型的基本構建塊是 Inception 模組，它有幾個不同的版本。

In figure 1 we show the canonical form of an Inception module, as found in the Inception V3 architecture.
在圖 1 中，我們展示了 Inception 模組的標準形式，如 Inception V3 架構中所見。

An Inception model can be understood as a stack of such modules.
Inception 模型可以被理解為此類模組的堆疊。

This is a departure from earlier VGG-style networks which were stacks of simple convolution layers.
這與早期的 VGG 風格網路不同，後者是簡單卷積層的堆疊。

While Inception modules are conceptually similar to convolutions (they are convolutional feature extractors), they empirically appear to be capable of learning richer representations with less parameters.
雖然 Inception 模組在概念上與卷積相似（它們是卷積特徵提取器），但經驗顯示它們似乎能夠以更少的參數學習更豐富的表示。

How do they work, and how do they differ from regular convolutions? What design strategies come after Inception?
它們是如何運作的？它們與常規卷積有何不同？Inception 之後有哪些設計策略？

### 1.1. The Inception hypothesis
### 1.1. Inception 假設

A convolution layer attempts to learn filters in a 3D space, with 2 spatial dimensions (width and height) and a channel dimension; thus a single convolution kernel is tasked with simultaneously mapping cross-channel correlations and spatial correlations.
卷積層試圖在 3D 空間中學習濾波器，該空間具有 2 個空間維度（寬度和高度）和一個通道維度；因此，單個卷積核的任務是同時映射跨通道相關性和空間相關性。

The idea behind the Inception module is to make this process easier and more efficient by explicitly factoring it into a series of operations that would independently look at cross-channel correlations and at spatial correlations.
Inception 模組背後的想法是通過將其顯式分解為一系列獨立查看跨通道相關性和空間相關性的操作，使這個過程更容易和更有效率。

More precisely, the typical Inception module first looks at cross-channel correlations via a set of 1x1 convolutions, mapping the input data into 3 or 4 separate spaces that are smaller than the original input space, and then maps all correlations in these smaller 3D spaces, via regular 3x3 or 5x5 convolutions.
更準確地說，典型的 Inception 模組首先通過一組 1x1 卷積查看跨通道相關性，將輸入數據映射到 3 或 4 個比原始輸入空間更小的獨立空間，然後通過常規的 3x3 或 5x5 卷積映射這些較小 3D 空間中的所有相關性。

This is illustrated in figure 1.
這在圖 1 中進行了說明。

In effect, the fundamental hypothesis behind Inception is that cross-channel correlations and spatial correlations are sufficiently decoupled that it is preferable not to map them jointly.
實際上，Inception 背後的基本假設是跨通道相關性和空間相關性充分解耦，因此最好不要聯合映射它們。

Consider a simplified version of an Inception module that only uses one size of convolution (e.g. 3x3) and does not include an average pooling tower (figure 2).
考慮一個簡化版本的 Inception 模組，它只使用一種尺寸的卷積（例如 3x3）並且不包括平均池化塔（圖 2）。

This Inception module can be reformulated as a large 1x1 convolution followed by spatial convolutions that would operate on non-overlapping segments of the output channels (figure 3).
這個 Inception 模組可以重新表述為一個大型 1x1 卷積，後面跟著在輸出通道的不重疊段上操作的空間卷積（圖 3）。

This observation naturally raises the question: what is the effect of the number of segments in the partition (and their size)?
這個觀察自然引發了一個問題：分區中的段數（及其大小）有什麼影響？

Would it be reasonable to make a much stronger hypothesis than the Inception hypothesis, and assume that cross-channel correlations and spatial correlations can be mapped completely separately?
做出比 Inception 假設更強的假設，即假設跨通道相關性和空間相關性可以完全分開映射，是否合理？

*Figure 1. A canonical Inception module (Inception V3).*
*圖 1. 一個標準的 Inception 模組 (Inception V3)。*

*Figure 2. A simplified Inception module.*
*圖 2. 一個簡化的 Inception 模組。*

### 1.2. The continuum between convolutions and separable convolutions
### 1.2. 卷積與可分離卷積之間的連續體

An “extreme” version of an Inception module, based on this stronger hypothesis, would first use a 1x1 convolution to map cross-channel correlations, and would then separately map the spatial correlations of every output channel.
基於這個更強假設的 Inception 模組的「極端」版本，將首先使用 1x1 卷積來映射跨通道相關性，然後分別映射每個輸出通道的空間相關性。

This is shown in figure 4.
這顯示在圖 4 中。

We remark that this extreme form of an Inception module is almost identical to a depthwise separable convolution, an operation that has been used in neural network design as early as 2014 [15] and has become more popular since its inclusion in the TensorFlow framework [1] in 2016.
我們注意到，这种 Inception 模組的極端形式幾乎與深度可分離卷積（depthwise separable convolution）完全相同，這是一種早在 2014 年就在神經網路設計中使用的操作 [15]，並且自 2016 年被納入 TensorFlow 框架 [1] 以來變得更加流行。

A depthwise separable convolution, commonly called “separable convolution” in deep learning frameworks such as TensorFlow and Keras, consists in a depthwise convolution, i.e. a spatial convolution performed independently over each channel of an input, followed by a pointwise convolution, i.e. a 1x1 convolution, projecting the channels output by the depthwise convolution onto a new channel space.
深度可分離卷積，在 TensorFlow 和 Keras 等深度學習框架中通常稱為「可分離卷積」，包括一個深度卷積（depthwise convolution），即在輸入的每個通道上獨立執行的空間卷積，隨後是一個逐點卷積（pointwise convolution），即一個 1x1 卷積，將深度卷積輸出的通道投影到一個新的通道空間。

This is not to be confused with a spatially separable convolution, which is also commonly called “separable convolution” in the image processing community.
這不應與空間可分離卷積混淆，後者在圖像處理界通常也稱為「可分離卷積」。

Two minor differences between and “extreme” version of an Inception module and a depthwise separable convolution would be:
Inception 模組的「極端」版本與深度可分離卷積之間有兩個微小的區別：

*   The order of the operations: depthwise separable convolutions as usually implemented (e.g. in TensorFlow) perform first channel-wise spatial convolution and then perform 1x1 convolution, whereas Inception performs the 1x1 convolution first.
    操作的順序：通常實現的深度可分離卷積（例如在 TensorFlow 中）首先執行通道式空間卷積，然後執行 1x1 卷積，而 Inception 首先執行 1x1 卷積。

*   The presence or absence of a non-linearity after the first operation. In Inception, both operations are followed by a ReLU non-linearity, however depthwise separable convolutions are usually implemented without non-linearities.
    第一個操作後是否存在非線性。在 Inception 中，兩個操作後面都跟著 ReLU 非線性，然而深度可分離卷積通常在實現時不帶非線性。

We argue that the first difference is unimportant, in particular because these operations are meant to be used in a stacked setting.
我們認為第一個區別並不重要，特別是因為這些操作是為了在堆疊設置中使用。

The second difference might matter, and we investigate it in the experimental section (in particular see figure 10).
第二個區別可能很重要，我們在實驗部分對此進行了調查（特別參見圖 10）。

We also note that other intermediate formulations of Inception modules that lie in between regular Inception modules and depthwise separable convolutions are also possible: in effect, there is a discrete spectrum between regular convolutions and depthwise separable convolutions, parametrized by the number of independent channel-space segments used for performing spatial convolutions.
我們還注意到，介於常規 Inception 模組和深度可分離卷積之間的其他 Inception 模組中間公式也是可能的：實際上，在常規卷積和深度可分離卷積之間存在一個離散頻譜，由用於執行空間卷積的獨立通道空間段的數量進行參數化。

A regular convolution (preceded by a 1x1 convolution), at one extreme of this spectrum, corresponds to the single-segment case; a depthwise separable convolution corresponds to the other extreme where there is one segment per channel; Inception modules lie in between, dividing a few hundreds of channels into 3 or 4 segments.
常規卷積（前面有一個 1x1 卷積）處於該頻譜的一個極端，對應於單段情況；深度可分離卷積對應於另一個極端，即每個通道有一個段；Inception 模組位於中間，將數百個通道分為 3 或 4 個段。

The properties of such intermediate modules appear not to have been explored yet.
這些中間模組的屬性似乎尚未被探索。

Having made these observations, we suggest that it may be possible to improve upon the Inception family of architectures by replacing Inception modules with depthwise separable convolutions, i.e. by building models that would be stacks of depthwise separable convolutions.
有了這些觀察結果，我們建議可以通過用深度可分離卷積替換 Inception 模組來改進 Inception 架構家族，即通過構建由深度可分離卷積堆疊而成的模型。

This is made practical by the efficient depthwise convolution implementation available in TensorFlow.
TensorFlow 中可用的高效深度卷積實現使這變得切實可行。

In what follows, we present a convolutional neural network architecture based on this idea, with a similar number of parameters as Inception V3, and we evaluate its performance against Inception V3 on two large-scale image classification task.
在下文中，我們提出了一個基於此想法的卷積神經網路架構，其參數數量與 Inception V3 相似，我們在兩個大規模圖像分類任務上評估了其相對於 Inception V3 的性能。

*Figure 3. A strictly equivalent reformulation of the simplified Inception module.*
*圖 3. 簡化 Inception 模組的嚴格等效重構。*

*Figure 4. An “extreme” version of our Inception module, with one spatial convolution per output channel of the 1x1 convolution.*
*圖 4. 我們 Inception 模組的「極端」版本，1x1 卷積的每個輸出通道有一個空間卷積。*

## 2. Prior work
## 2. 先前工作

The present work relies heavily on prior efforts in the following areas:
本工作在很大程度上依賴於以下領域的先前努力：

*   Convolutional neural networks [10, 9, 25], in particular the VGG-16 architecture [18], which is schematically similar to our proposed architecture in a few respects.
    卷積神經網路 [10, 9, 25]，特別是 VGG-16 架構 [18]，其在幾個方面與我們提出的架構在結構上相似。

*   The Inception architecture family of convolutional neural networks [20, 7, 21, 19], which first demonstrated the advantages of factoring convolutions into multiple branches operating successively on channels and then on space.
    卷積神經網路的 Inception 架構家族 [20, 7, 21, 19]，它首先證明了將卷積分解為多個分支，先在通道上然後在空間上操作的優勢。

*   Depthwise separable convolutions, which our proposed architecture is entirely based upon.
    深度可分離卷積，我們提出的架構完全基於此。
    While the use of spatially separable convolutions in neural networks has a long history, going back to at least 2012 [12] (but likely even earlier), the depthwise version is more recent.
    雖然在神經網路中使用空間可分離卷積由來已久，至少可以追溯到 2012 年 [12]（但可能更早），但深度版本是較新的。
    Laurent Sifre developed depthwise separable convolutions during an internship at Google Brain in 2013, and used them in AlexNet to obtain small gains in accuracy and large gains in convergence speed, as well as a significant reduction in model size.
    Laurent Sifre 於 2013 年在 Google Brain 實習期間開發了深度可分離卷積，並將其用於 AlexNet，在精度上獲得了小幅提升，在收斂速度上獲得了大幅提升，同時顯著減小了模型大小。
    An overview of his work was first made public in a presentation at ICLR 2014 [23].
    他的工作概覽首次在 ICLR 2014 的演示中公開 [23]。
    Detailed experimental results are reported in Sifre’s thesis, section 6.2 [15].
    詳細的實驗結果報告在 Sifre 的論文第 6.2 節中 [15]。
    This initial work on depthwise separable convolutions was inspired by prior research from Sifre and Mallat on transformation-invariant scattering [16, 15].
    這項關於深度可分離卷積的初步工作是受到 Sifre 和 Mallat 先前關於變換不變散射（transformation-invariant scattering）的研究 [16, 15] 的啟發。
    Later, a depthwise separable convolution was used as the first layer of Inception V1 and Inception V2 [20, 7].
    後來，深度可分離卷積被用作 Inception V1 和 Inception V2 的第一層 [20, 7]。
    Within Google, Andrew Howard [6] has introduced efficient mobile models called MobileNets using depthwise separable convolutions.
    在 Google 內部，Andrew Howard [6] 推出了使用深度可分離卷積的稱為 MobileNets 的高效移動模型。
    Jin et al. in 2014 [8] and Wang et al. in 2016 [24] also did related work aiming at reducing the size and computational cost of convolutional neural networks using separable convolutions.
    Jin 等人在 2014 年 [8] 和 Wang 等人在 2016 年 [24] 也做了相關工作，旨在利用可分離卷積減少卷積神經網路的大小和計算成本。
    Additionally, our work is only possible due to the inclusion of an efficient implementation of depthwise separable convolutions in the TensorFlow framework [1].
    此外，我們的工作之所以成為可能，完全歸功於 TensorFlow 框架 [1] 中包含的深度可分離卷積的高效實現。

*   Residual connections, introduced by He et al. in [4], which our proposed architecture uses extensively.
    殘差連接，由 He 等人在 [4] 中引入，我們提出的架構廣泛使用了它。

## 3. The Xception architecture
## 3. Xception 架構

We propose a convolutional neural network architecture based entirely on depthwise separable convolution layers.
我們提出了一種完全基於深度可分離卷積層的卷積神經網路架構。

In effect, we make the following hypothesis: that the mapping of cross-channels correlations and spatial correlations in the feature maps of convolutional neural networks can be *entirely* decoupled.
實際上，我們做出以下假設：卷積神經網路特徵圖中跨通道相關性和空間相關性的映射可以被**完全**解耦。

Because this hypothesis is a stronger version of the hypothesis underlying the Inception architecture, we name our proposed architecture *Xception*, which stands for “Extreme Inception”.
因為這個假設是 Inception 架構背後假設的更強版本，我們將我們提出的架構命名為 *Xception*，意為「極端 Inception」（Extreme Inception）。

A complete description of the specifications of the network is given in figure 5.
網路規範的完整描述見圖 5。

The Xception architecture has 36 convolutional layers forming the feature extraction base of the network.
Xception 架構有 36 個卷積層，構成了網路的特徵提取基礎。

In our experimental evaluation we will exclusively investigate image classification and therefore our convolutional base will be followed by a logistic regression layer.
在我們的實驗評估中，我們將專門研究圖像分類，因此我們的卷積基礎後面將跟著一個邏輯迴歸層。

Optionally one may insert fully-connected layers before the logistic regression layer, which is explored in the experimental evaluation section (in particular, see figures 7 and 8).
可選地，可以在邏輯迴歸層之前插入全連接層，這在實驗評估部分進行了探索（特別參見圖 7 和圖 8）。

The 36 convolutional layers are structured into 14 modules, all of which have linear residual connections around them, except for the first and last modules.
36 個卷積層被組織成 14 個模組，除第一個和最後一個模組外，所有模組周圍都有線性殘差連接。

In short, the Xception architecture is a linear stack of depthwise separable convolution layers with residual connections.
簡而言之，Xception 架構是帶有殘差連接的深度可分離卷積層的線性堆疊。

This makes the architecture very easy to define and modify; it takes only 30 to 40 lines of code using a high-level library such as Keras [2] or TensorFlow-Slim [17], not unlike an architecture such as VGG-16 [18], but rather unlike architectures such as Inception V2 or V3 which are far more complex to define.
這使得該架構非常容易定義和修改；使用像 Keras [2] 或 TensorFlow-Slim [17] 這樣的高階函式庫只需要 30 到 40 行程式碼，這與 VGG-16 [18] 等架構並沒有什麼不同，但與 Inception V2 或 V3 等定義起來要複雜得多的架構截然不同。

An open-source implementation of Xception using Keras and TensorFlow is provided as part of the Keras Applications module, under the MIT license.
Xception 的開源實現（使用 Keras 和 TensorFlow）作為 Keras Applications 模組的一部分提供，採用 MIT 許可證。

## 4. Experimental evaluation
## 4. 實驗評估

We choose to compare Xception to the Inception V3 architecture, due to their similarity of scale: Xception and Inception V3 have nearly the same number of parameters (table 3), and thus any performance gap could not be attributed to a difference in network capacity.
我們選擇將 Xception 與 Inception V3 架構進行比較，因為它們的規模相似：Xception 和 Inception V3 具有幾乎相同數量的參數（表 3），因此任何性能差距都不能歸因於網路容量的差異。

We conduct our comparison on two image classification tasks: one is the well-known 1000-class single-label classification task on the ImageNet dataset [14], and the other is a 17,000-class multi-label classification task on the large-scale JFT dataset.
我們在兩個圖像分類任務上進行比較：一個是 ImageNet 資料集 [14] 上著名的 1000 類單標籤分類任務，另一個是大型 JFT 資料集上的 17,000 類多標籤分類任務。

### 4.1. The JFT dataset
### 4.1. JFT 資料集

JFT is an internal Google dataset for large-scale image classification dataset, first introduced by Hinton et al. in [5], which comprises over 350 million high-resolution images annotated with labels from a set of 17,000 classes.
JFT 是 Google 內部的用於大規模圖像分類的資料集，由 Hinton 等人在 [5] 中首次引入，包含超過 3.5 億張高解析度圖像，標註有來自 17,000 個類別的標籤。

To evaluate the performance of a model trained on JFT, we use an auxiliary dataset, **FastEval14k**.
為了評估在 JFT 上訓練的模型的性能，我們使用一個輔助資料集，**FastEval14k**。

FastEval14k is a dataset of 14,000 images with dense annotations from about 6,000 classes (36.5 labels per image on average).
FastEval14k 是一個包含 14,000 張圖像的資料集，帶有來自約 6,000 個類別的密集註釋（平均每張圖像 36.5 個標籤）。

On this dataset we evaluate performance using Mean Average Precision for top 100 predictions (MAP@100), and we weight the contribution of each class to MAP@100 with a score estimating how common (and therefore important) the class is among social media images.
在這個資料集上，我們使用前 100 個預測的平均精度均值 (MAP@100) 來評估性能，並且我們用一個分數來加權每個類別對 MAP@100 的貢獻，該分數估計該類別在社交媒體圖像中的普遍程度（因此也包括重要性）。

This evaluation procedure is meant to capture performance on frequently occurring labels from social media, which is crucial for production models at Google.
此評估程序旨在捕捉社交媒體中頻繁出現的標籤的性能，這對於 Google 的生產模型至關重要。

### 4.2. Optimization configuration
### 4.2. 優化配置

A different optimization configuration was used for ImageNet and JFT:
ImageNet 和 JFT 使用了不同的優化配置：

*   On ImageNet:
    在 ImageNet 上：
    *   Optimizer: SGD
        優化器：SGD
    *   Momentum: 0.9
        動量：0.9
    *   Initial learning rate: 0.045
        初始學習率：0.045
    *   Learning rate decay: decay of rate 0.94 every 2 epochs
        學習率衰減：每 2 個 epoch 衰減率為 0.94

*   On JFT:
    在 JFT 上：
    *   Optimizer: RMSprop [22]
        優化器：RMSprop [22]
    *   Momentum: 0.9
        動量：0.9
    *   Initial learning rate: 0.001
        初始學習率：0.001
    *   Learning rate decay: decay of rate 0.9 every 3,000,000 samples
        學習率衰減：每 3,000,000 個樣本衰減率為 0.9

For both datasets, the same exact same optimization configuration was used for both Xception and Inception V3.
對於這兩個資料集，Xception 和 Inception V3 使用了完全相同的優化配置。

Note that this configuration was tuned for best performance with Inception V3; we did not attempt to tune optimization hyperparameters for Xception.
請注意，此配置是針對 Inception V3 的最佳性能進行調整的；我們沒有嘗試調整 Xception 的優化超參數。

Since the networks have different training profiles (figure 6), this may be suboptimal, especially on the ImageNet dataset, on which the optimization configuration used had been carefully tuned for Inception V3.
由於網路具有不同的訓練概況（圖 6），這可能是次優的，特別是在 ImageNet 資料集上，所使用的優化配置是為 Inception V3 精心調整的。

Additionally, all models were evaluated using Polyak averaging [13] at inference time.
此外，所有模型在推理時都使用 Polyak 平均 [13] 進行了評估。

### 4.3. Regularization configuration
### 4.3. 正規化配置

*   **Weight decay**: The Inception V3 model uses a weight decay (L2 regularization) rate of $4e-5$, which has been carefully tuned for performance on ImageNet.
    **權重衰減**：Inception V3 模型使用 $4e-5$ 的權重衰減（L2 正規化）率，該比率已針對 ImageNet 上的性能進行了仔細調整。
    We found this rate to be quite suboptimal for Xception and instead settled for $1e-5$.
    我們發現這個比率對 Xception 來說相當次優，因此改為 $1e-5$。
    We did not perform an extensive search for the optimal weight decay rate.
    我們沒有對最佳權重衰減率進行廣泛的搜索。
    The same weight decay rates were used both for the ImageNet experiments and the JFT experiments.
    ImageNet 實驗和 JFT 實驗都使用了相同的權重衰減率。

*   **Dropout**: For the ImageNet experiments, both models include a dropout layer of rate 0.5 before the logistic regression layer.
    **Dropout**：對於 ImageNet 實驗，兩個模型在邏輯迴歸層之前都包含一個比率為 0.5 的 dropout 層。
    For the JFT experiments, no dropout was included due to the large size of the dataset which made overfitting unlikely in any reasonable amount of time.
    對於 JFT 實驗，由於資料集規模龐大，使得在任何合理的時間內都不太可能發生過擬合，因此未包含 dropout。

*   **Auxiliary loss tower**: The Inception V3 architecture may optionally include an auxiliary tower which back-propagates the classification loss earlier in the network, serving as an additional regularization mechanism.
    **輔助損失塔**：Inception V3 架構可以選擇性地包含一個輔助塔，該塔在網路較早的位置反向傳播分類損失，作為額外的正規化機制。
    For simplicity, we choose not to include this auxiliary tower in any of our models.
    為了簡單起見，我們選擇不在我們的任何模型中包含此輔助塔。

### 4.4. Training infrastructure
### 4.4. 訓練基礎設施

All networks were implemented using the TensorFlow framework [1] and trained on 60 NVIDIA K80 GPUs each.
所有網路均使用 TensorFlow 框架 [1] 實現，並在 60 個 NVIDIA K80 GPU 上進行訓練。

For the ImageNet experiments, we used data parallelism with *synchronous* gradient descent to achieve the best classification performance, while for JFT we used *asynchronous* gradient descent so as to speed up training.
對於 ImageNet 實驗，我們使用帶有**同步**梯度下降的數據並行來實現最佳分類性能，而對於 JFT，我們使用**異步**梯度下降來加速訓練。

The ImageNet experiments took approximately 3 days each, while the JFT experiments took over one month each.
ImageNet 實驗每個大約花費 3 天，而 JFT 實驗每個花費超過一個月。

The JFT models were not trained to full convergence, which would have taken over three month per experiment.
JFT 模型沒有訓練到完全收斂，這將導致每個實驗花費超過三個月。

*Figure 5. The Xception architecture: the data first goes through the entry flow, then through the middle flow which is repeated eight times, and finally through the exit flow. Note that all Convolution and SeparableConvolution layers are followed by batch normalization [7] (not included in the diagram). All SeparableConvolution layers use a depth multiplier of 1 (no depth expansion).*
*圖 5. Xception 架構：數據首先通過入口流（entry flow），然後通過重複八次的中間流（middle flow），最後通過出口流（exit flow）。請注意，所有卷積和可分離卷積層後面都跟著批次正規化 [7]（圖中未包括）。所有可分離卷積層都使用 1 的深度乘數（無深度擴展）。*

**(Diagram Description Translated)**
**(圖表描述翻譯)**
*   **Entry flow (入口流)**
    *   299x299x3 圖像 -> 卷積 32, 3x3, 步幅=2x2 -> ReLU -> 卷積 64, 3x3 -> ReLU -> 殘差連接塊 (可分離卷積 128, 3x3 -> ReLU -> 可分離卷積 128, 3x3 -> 最大池化 3x3, 步幅=2x2) ...
*   **Middle flow (中間流)**
    *   重複 8 次: ReLU -> 可分離卷積 728, 3x3 -> ReLU -> 可分離卷積 728, 3x3 -> ReLU -> 可分離卷積 728, 3x3 ...
*   **Exit flow (出口流)**
    *   殘差連接塊 -> 可分離卷積 1536, 3x3 -> ReLU -> 可分離卷積 2048, 3x3 -> ReLU -> 全域平均池化 -> 2048 維向量 -> 可選全連接層 -> 邏輯迴歸

### 4.5. Comparison with Inception V3
### 4.5. 與 Inception V3 的比較

#### 4.5.1 Classification performance
#### 4.5.1 分類性能

All evaluations were run with a single crop of the inputs images and a single model.
所有評估均使用輸入圖像的單次裁剪和單個模型運行。

ImageNet results are reported on the validation set rather than the test set (i.e. on the non-blacklisted images from the validation set of ILSVRC 2012).
ImageNet 結果是在驗證集而不是測試集上報告的（即 ILSVRC 2012 驗證集中的非黑名單圖像）。

JFT results are reported after 30 million iterations (one month of training) rather than after full convergence.
JFT 結果是在 3000 萬次迭代（一個月的訓練）後報告的，而不是在完全收斂後。

Results are provided in table 1 and table 2, as well as figure 6, figure 7, figure 8.
結果在表 1 和表 2，以及圖 6、圖 7、圖 8 中提供。

On JFT, we tested both versions of our networks that did not include any fully-connected layers, and versions that included two fully-connected layers of 4096 units each before the logistic regression layer.
在 JFT 上，我們測試了不包含任何全連接層的網路版本，以及在邏輯迴歸層之前包含兩個各 4096 單元的全連接層的版本。

On ImageNet, Xception shows marginally better results than Inception V3.
在 ImageNet 上，Xception 顯示出比 Inception V3 略好的結果。

On JFT, Xception shows a 4.3% relative improvement on the FastEval14k MAP@100 metric.
在 JFT 上，Xception 在 FastEval14k MAP@100 指標上顯示出 4.3% 的相對改進。

We also note that Xception outperforms ImageNet results reported by He et al. for ResNet-50, ResNet-101 and ResNet-152 [4].
我們還注意到，Xception 優於 He 等人報告的 ResNet-50、ResNet-101 和 ResNet-152 的 ImageNet 結果 [4]。

**Table 1. Classification performance comparison on ImageNet (single crop, single model). VGG-16 and ResNet-152 numbers are only included as a reminder. The version of Inception V3 being benchmarked does not include the auxiliary tower.**
**表 1. ImageNet 上的分類性能比較（單次裁剪，單模型）。VGG-16 和 ResNet-152 的數字僅作為提醒包含在內。正在進行基準測試的 Inception V3 版本不包括輔助塔。**

| | Top-1 準確率 | Top-5 準確率 |
| :--- | :---: | :---: |
| **VGG-16** | 0.715 | 0.901 |
| **ResNet-152** | 0.770 | 0.933 |
| **Inception V3** | 0.782 | 0.941 |
| **Xception** | **0.790** | **0.945** |

The Xception architecture shows a much larger performance improvement on the JFT dataset compared to the ImageNet dataset.
與 ImageNet 資料集相比，Xception 架構在 JFT 資料集上顯示出更大的性能提升。

We believe this may be due to the fact that Inception V3 was developed with a focus on ImageNet and may thus be by design over-fit to this specific task.
我們認為這可能是由於 Inception V3 的開發重點是 ImageNet，因此可能在設計上過度擬合了這一特定任務。

On the other hand, neither architecture was tuned for JFT.
另一方面，兩種架構都沒有針對 JFT 進行調整。

It is likely that a search for better hyperparameters for Xception on ImageNet (in particular optimization parameters and regularization parameters) would yield significant additional improvement.
在 ImageNet 上為 Xception 尋找更好的超參數（特別是優化參數和正規化參數）很可能會產生顯著的額外改進。

**Table 2. Classification performance comparison on JFT (single crop, single model).**
**表 2. JFT 上的分類性能比較（單次裁剪，單模型）。**

| | FastEval14k MAP@100 |
| :--- | :---: |
| **Inception V3 - 無 FC 層** | 6.36 |
| **Xception - 無 FC 層** | 6.70 |
| **Inception V3 有 FC 層** | 6.50 |
| **Xception 有 FC 層** | **6.78** |

*Figure 6. Training profile on ImageNet*
*圖 6. ImageNet 上的訓練概況*

*Figure 7. Training profile on JFT, without fully-connected layers*
*圖 7. JFT 上的訓練概況，無全連接層*

#### 4.5.2 Size and speed
#### 4.5.2 大小和速度

**Table 3. Size and training speed comparison.**
**表 3. 大小和訓練速度比較。**

| | 參數數量 | 步數/秒 |
| :--- | :---: | :---: |
| **Inception V3** | 23,626,728 | 31 |
| **Xception** | 22,855,952 | 28 |

In table 3 we compare the size and speed of Inception V3 and Xception.
在表 3 中，我們比較了 Inception V3 和 Xception 的大小和速度。

Parameter count is reported on ImageNet (1000 classes, no fully-connected layers) and the number of training steps (gradient updates) per second is reported on ImageNet with 60 K80 GPUs running synchronous gradient descent.
參數計數是在 ImageNet（1000 個類別，無全連接層）上報告的，每秒訓練步數（梯度更新）是在 ImageNet 上使用 60 個 K80 GPU 運行同步梯度下降報告的。

Both architectures have approximately the same size (within 3.5%), and Xception is marginally slower.
兩種架構的大小大致相同（誤差在 3.5% 以內），Xception 略慢。

We expect that engineering optimizations at the level of the depthwise convolution operations can make Xception faster than Inception V3 in the near future.
我們預計，深度卷積操作層面的工程優化可以在不久的將來使 Xception 比 Inception V3 更快。

The fact that both architectures have almost the same number of parameters indicates that the improvement seen on ImageNet and JFT does not come from added capacity but rather from a more efficient use of the model parameters.
這兩種架構具有幾乎相同數量的參數這一事實表明，在 ImageNet 和 JFT 上看到的改進並非來自增加的容量，而是來自模型參數的更有效使用。

*Figure 8. Training profile on JFT, with fully-connected layers*
*圖 8. JFT 上的訓練概況，有全連接層*

### 4.6. Effect of the residual connections
### 4.6. 殘差連接的影響

*Figure 9. Training profile with and without residual connections.*
*圖 9. 有無殘差連接的訓練概況。*

To quantify the benefits of residual connections in the Xception architecture, we benchmarked on ImageNet a modified version of Xception that does not include any residual connections.
為了量化 Xception 架構中殘差連接的好處，我們在 ImageNet 上對不包含任何殘差連接的 Xception 修改版本進行了基準測試。

Results are shown in figure 9.
結果如圖 9 所示。

Residual connections are clearly essential in helping with convergence, both in terms of speed and final classification performance.
殘差連接在幫助收斂方面顯然至關重要，無論是在速度還是最終分類性能方面。

However we will note that benchmarking the non-residual model with the same optimization configuration as the residual model may be uncharitable and that better optimization configurations might yield more competitive results.
然而我們會注意到，使用與殘差模型相同的優化配置對非殘差模型進行基準測試可能不太公平，更好的優化配置可能會產生更具競爭力的結果。

Additionally, let us note that this result merely shows the importance of residual connections *for this specific architecture*, and that residual connections are in no way *required* in order to build models that are stacks of depthwise separable convolutions.
此外，我們要注意，這個結果僅顯示了殘差連接*對於這個特定架構*的重要性，並且殘差連接絕不是構建深度可分離卷積堆疊模型的*必要條件*。

We also obtained excellent results with non-residual VGG-style models where all convolution layers were replaced with depthwise separable convolutions (with a depth multiplier of 1), superior to Inception V3 on JFT at equal parameter count.
我們還通過非殘差 VGG 風格模型獲得了極好的結果，其中所有卷積層都替換為深度可分離卷積（深度乘數為 1），在參數數量相等的情況下，在 JFT 上的表現優於 Inception V3。

### 4.7. Effect of an intermediate activation after pointwise convolutions
### 4.7. 逐點卷積後中間激活的影響

*Figure 10. Training profile with different activations between the depthwise and pointwise operations of the separable convolution layers.*
*圖 10. 可分離卷積層的深度和逐點操作之間具有不同激活的訓練概況。*

We mentioned earlier that the analogy between depthwise separable convolutions and Inception modules suggests that depthwise separable convolutions should potentially include a non-linearity between the depthwise and pointwise operations.
我們早些時候提到，深度可分離卷積和 Inception 模組之間的類比表明，深度可分離卷積可能應該在深度和逐點操作之間包含非線性。

In the experiments reported so far, no such non-linearity was included.
在目前報告的實驗中，未包含此類非線性。

However we also experimentally tested the inclusion of either ReLU or ELU [3] as intermediate non-linearity.
然而，我們也通過實驗測試了包含 ReLU 或 ELU [3] 作為中間非線性。

Results are reported on ImageNet in figure 10, and show that the absence of any non-linearity leads to both faster convergence and better final performance.
結果在圖 10 中的 ImageNet 上報告，並顯示沒有任何非線性會導致更快的收斂和更好的最終性能。

This is a remarkable observation, since Szegedy et al. report the opposite result in [21] for Inception modules.
這是一個值得注意的觀察結果，因為 Szegedy 等人在 [21] 中報告了 Inception 模組的相反結果。

It may be that the depth of the intermediate feature spaces on which spatial convolutions are applied is critical to the usefulness of the non-linearity: for deep feature spaces (e.g. those found in Inception modules) the non-linearity is helpful, but for shallow ones (e.g. the 1-channel deep feature spaces of depthwise separable convolutions) it becomes harmful, possibly due to a loss of information.
這可能是因為應用空間卷積的中間特徵空間的深度對於非線性的有用性至關重要：對於深特徵空間（例如 Inception 模組中的特徵空間），非線性是有幫助的，但對於淺特徵空間（例如深度可分離卷積的 1 通道深特徵空間），它變得有害，可能是由於訊息丟失。

## 5. Future directions
## 5. 未來方向

We noted earlier the existence of a discrete spectrum between regular convolutions and depthwise separable convolutions, parametrized by the number of independent channel-space segments used for performing spatial convolutions.
我們早些時候注意到常規卷積和深度可分離卷積之間存在離散頻譜，由用於執行空間卷積的獨立通道空間段的數量進行參數化。

Inception modules are one point on this spectrum.
Inception 模組是此頻譜上的一點。

We showed in our empirical evaluation that the extreme formulation of an Inception module, the depthwise separable convolution, may have advantages over regular a regular Inception module.
我們在實證評估中表明，Inception 模組的極端形式，即深度可分離卷積，可能比常規 Inception 模組具有優勢。

However, there is no reason to believe that depthwise separable convolutions are optimal.
然而，沒有理由相信深度可分離卷積是最佳的。

It may be that intermediate points on the spectrum, lying between regular Inception modules and depthwise separable convolutions, hold further advantages.
位於常規 Inception 模組和深度可分離卷積之間的頻譜中間點可能具有進一步的優勢。

This question is left for future investigation.
這個問題留待將來調查。

## 6. Conclusions
## 6. 結論

We showed how convolutions and depthwise separable convolutions lie at both extremes of a discrete spectrum, with Inception modules being an intermediate point in between.
我們展示了卷積和深度可分離卷積如何位於離散頻譜的兩個極端，而 Inception 模組是介於兩者之間的中間點。

This observation has led to us to propose replacing Inception modules with depthwise separable convolutions in neural computer vision architectures.
這一觀察促使我們提議在神經電腦視覺架構中用深度可分離卷積替換 Inception 模組。

We presented a novel architecture based on this idea, named Xception, which has a similar parameter count as Inception V3.
我們提出了一種基於此想法的新穎架構，名為 Xception，其參數數量與 Inception V3 相似。

Compared to Inception V3, Xception shows small gains in classification performance on the ImageNet dataset and large gains on the JFT dataset.
與 Inception V3 相比，Xception 在 ImageNet 資料集上的分類性能顯示出小幅提升，而在 JFT 資料集上顯示出大幅提升。

We expect depthwise separable convolutions to become a cornerstone of convolutional neural network architecture design in the future, since they offer similar properties as Inception modules, yet are as easy to use as regular convolution layers.
我們預計深度可分離卷積將成為未來卷積神經網路架構設計的基石，因為它們提供了與 Inception 模組相似的屬性，但像常規卷積層一樣易於使用。

## References
[1] M. Abadi, A. Agarwal, P. Barham, E. Brevdo, Z. Chen, C. Citro, G. S. Corrado, A. Davis, J. Dean, M. Devin, S. Ghemawat, I. Goodfellow, A. Harp, G. Irving, M. Isard, Y. Jia, R. Jozefowicz, L. Kaiser, M. Kudlur, J. Levenberg, D. Mané, R. Monga, S. Moore, D. Murray, C. Olah, M. Schuster, J. Shlens, B. Steiner, I. Sutskever, K. Talwar, P. Tucker, V. Vanhoucke, V. Vasudevan, F. Viégas, O. Vinyals, P. Warden, M. Wattenberg, M. Wicke, Y. Yu, and X. Zheng. TensorFlow: Large-scale machine learning on heterogeneous systems, 2015. Software available from tensorflow.org.

[2] F. Chollet. Keras. https://github.com/fchollet/keras, 2015.

[3] D.-A. Clevert, T. Unterthiner, and S. Hochreiter. Fast and accurate deep network learning by exponential linear units (elus). arXiv preprint arXiv:1511.07289, 2015.

[4] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. arXiv preprint arXiv:1512.03385, 2015.

[5] G. Hinton, O. Vinyals, and J. Dean. Distilling the knowledge in a neural network, 2015.

[6] A. Howard. Mobilenets: Efficient convolutional neural networks for mobile vision applications. Forthcoming.

[7] S. Ioffe and C. Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift. In Proceedings of The 32nd International Conference on Machine Learning, pages 448–456, 2015.

[8] J. Jin, A. Dundar, and E. Culurciello. Flattened convolutional neural networks for feedforward acceleration. arXiv preprint arXiv:1412.5474, 2014.

[9] A. Krizhevsky, I. Sutskever, and G. E. Hinton. Imagenet classification with deep convolutional neural networks. In Advances in neural information processing systems, pages 1097–1105, 2012.

[10] Y. LeCun, L. Jackel, L. Bottou, C. Cortes, J. S. Denker, H. Drucker, I. Guyon, U. Muller, E. Sackinger, P. Simard, et al. Learning algorithms for classification: A comparison on handwritten digit recognition. Neural networks: the statistical mechanics perspective, 261:276, 1995.

[11] M. Lin, Q. Chen, and S. Yan. Network in network. arXiv preprint arXiv:1312.4400, 2013.

[12] F. Mamalet and C. Garcia. Simplifying ConvNets for Fast Learning. In International Conference on Artificial Neural Networks (ICANN 2012), pages 58–65. Springer, 2012.

[13] B. T. Polyak and A. B. Juditsky. Acceleration of stochastic approximation by averaging. SIAM J. Control Optim., 30(4):838–855, July 1992.

[14] O. Russakovsky, J. Deng, H. Su, J. Krause, S. Satheesh, S. Ma, Z. Huang, A. Karpathy, A. Khosla, M. Bernstein, et al. Imagenet large scale visual recognition challenge. 2014.

[15] L. Sifre. Rigid-motion scattering for image classification, 2014. Ph.D. thesis.

[16] L. Sifre and S. Mallat. Rotation, scaling and deformation invariant scattering for texture discrimination. In 2013 IEEE Conference on Computer Vision and Pattern Recognition, Portland, OR, USA, June 23-28, 2013, pages 1233–1240, 2013.

[17] N. Silberman and S. Guadarrama. Tf-slim, 2016.

[18] K. Simonyan and A. Zisserman. Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556, 2014.

[19] C. Szegedy, S. Ioffe, and V. Vanhoucke. Inception-v4, inception-resnet and the impact of residual connections on learning. arXiv preprint arXiv:1602.07261, 2016.

[20] C. Szegedy, W. Liu, Y. Jia, P. Sermanet, S. Reed, D. Anguelov, D. Erhan, V. Vanhoucke, and A. Rabinovich. Going deeper with convolutions. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 1–9, 2015.

[21] C. Szegedy, V. Vanhoucke, S. Ioffe, J. Shlens, and Z. Wojna. Rethinking the inception architecture for computer vision. arXiv preprint arXiv:1512.00567, 2015.

[22] T. Tieleman and G. Hinton. Divide the gradient by a running average of its recent magnitude. COURSERA: Neural Networks for Machine Learning, 4, 2012. Accessed: 2015-11-05.

[23] V. Vanhoucke. Learning visual representations at scale. ICLR, 2014.

[24] M. Wang, B. Liu, and H. Foroosh. Factorized convolutional neural networks. arXiv preprint arXiv:1608.04337, 2016.

[25] M. D. Zeiler and R. Fergus. Visualizing and understanding convolutional networks. In Computer Vision–ECCV 2014, pages 818–833. Springer, 2014.
