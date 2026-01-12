# SC2-PCR：一種用於高效穩健點雲配準的二階空間相容性
## **Published in:** 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022

Zhi Chen¹ Kun Sun² Fan Yang¹ Wenbing Tao¹*

陳治¹ 孫坤² 楊帆¹ 陶文兵¹*

¹多光譜資訊處理技術國家重點實驗室，華中科技大學人工智慧與自動化學院，中國
²智慧地理資訊處理湖北省重點實驗室，中國地質大學電腦學院，中國
{z_chen, fanyang, wenbingtao}@hust.edu.cn; sunkun@cug.edu.cn

Abstract

摘要

In this paper, we present a second order spatial compatibility (SC²) measure based method for efficient and robust point cloud registration (PCR), called SC2-PCR 1. Firstly, we propose a second order spatial compatibility (SC2) mea- sure to compute the similarity between correspondences. It considers the global compatibility instead of local consis- tency, allowing for more distinctive clustering between in- liers and outliers at early stage. Based on this measure, our registration pipeline employs a global spectral technique to find some reliable seeds from the initial correspondences. Then we design a two-stage strategy to expand each seed to a consensus set based on the SC² measure matrix. Finally, we feed each consensus set to a weighted SVD algorithm to generate a candidate rigid transformation and select the best model as the final result. Our method can guarantee to find a certain number of outlier-free consensus sets using fewer samplings, making the model estimation more efficient and robust. In addition, the proposed SC² measure is general and can be easily plugged into deep learning based frameworks. Extensive experiments are carried out to investigate the performance of our method.

在本文中，我們提出了一種基於二階空間相容性（SC²）測度的點雲高效穩健配準（PCR）方法，稱為 SC2-PCR 1。首先，我們提出一種二階空間相容性（SC²）測度來計算對應關係之間的相似性。它考慮全域相容性而非局部一致性，從而能在早期階段更清晰地區分內點和外點。基於此測度，我們的配準流程採用全域譜技術從初始對應關係中找到一些可靠的種子點。然後，我們設計了一種兩階段策略，根據 SC² 測度矩陣將每個種子點擴展為一個共識集。最後，我們將每個共識集輸入加權 SVD 演算法，以生成候選剛性變換，並選擇最佳模型作為最終結果。我們的方法可以保證使用較少的採樣找到一定數量的無外點共識集，從而使模型估計更有效率和穩健。此外，我們提出的 SC² 測度具有通用性，可以輕鬆地整合到基於深度學習的框架中。我們進行了大量實驗來驗證我們方法的性能。

* Corresponding author.
Code will be available at https://github.com/
ZhiChen902/SC2-PCR.

* 通訊作者。
程式碼將在 https://github.com/ZhiChen902/SC2-PCR 提供。

## 1. Introduction

## 1. 緒論

The alignment of two 3D scans of the same scene, known as Point Cloud Registration(PCR), plays an important role in areas such as Simultaneous Localization and Mapping (SLAM) [7, 24, 40], augmented reality [4, 11] and robotic- s applications [34]. A canonical solution first establishes feature correspondences and then estimates the 3D rotation and translation that can best align the shared parts. How- ever, due to challenges such as partial overlap or feature ambiguity, model estimation is prone to outliers in the correspondences, leading to inaccurate or wrong alignment.

將同一場景的兩個 3D 掃描對齊，即點雲配準（PCR），在諸如同時定位與地圖構建（SLAM）[7, 24, 40]、擴增實境 [4, 11] 和機器人應用 [34] 等領域扮演著重要角色。典型的解決方案首先建立特徵對應關係，然後估計能夠最佳對齊共享部分的 3D 旋轉和平移。然而，由於部分重疊或特徵模糊等挑戰，模型估計容易受到對應關係中外點的影響，導致配準不準確或錯誤。

RANSAC [25] pioneers the iterative sampling strategy for model estimation. However, it needs more time to con- verge or sometimes there is no guarantee of an accurate solution due to massive outliers. Spatial Compatibility (SC) [5, 35, 45, 57] is a widely used similarity measure for boosting the robustness and efficiency of the rigid transformation estimation. It assumes that two correspondences will have higher score if the difference of spatial distance between them, e.g. d12 - d12| or |d16 – d16| in Fig. 1(a), is minor.

RANSAC [25] 開創了用於模型估計的迭代採樣策略。然而，由於存在大量外點，它需要更多時間才能收斂，有時甚至無法保證得到準確的解。空間相容性（SC）[5, 35, 45, 57] 是一種廣泛使用的相似性度量，用於提升剛性變換估計的穩健性和效率。它假設如果兩個對應關係之間的空間距離差異（例如圖 1(a) 中的 d12 - d12| 或 |d16 – d16|）很小，那麼它們將具有較高的分數。

Figure 1. (a): A toy example in which red and green line segments represent outliers and inliers, respectively. (b): The first order compatibility matrix of (a). As highlighted by yellow, the outlier- s have very high compatibility scores with some inliers. (c): A binarized compatibility matrix of (b) after thresholding. (d): The proposed second order compatibility matrix of (a). By contrast, the values in the rows and columns of the outliers are small.

圖 1. (a)：一個玩具範例，其中紅色和綠色線段分別代表外點和內點。(b)：(a) 的一階相容性矩陣。如黃色突顯部分所示，外點與某些內點具有非常高的相容性分數。(c)：(b) 經過閾值處理後的二元化相容性矩陣。(d)：(a) 的建議二階相容性矩陣。相比之下，外點所在行和列的值都很小。

Thus, sampling from compatible correspondences increas- es the probability of getting inliers. However, such kind of first order metric still suffers from outliers due to locality and ambiguity. Fig. 1(b) is the spatial compatibility matrix of the correspondences in Fig. 1(a). As we can see from the yellow cells, C6 and c7 are outliers but they show high compatibility scores with some inliers by chance. As a re- sult, the outliers would be inevitably involved in the model estimation process, leading to performance deterioration.

因此，從相容的對應關係中採樣會增加獲得內點的機率。然而，這種一階度量仍然會因為局部性和模糊性而受到外點的影響。圖 1(b) 是圖 1(a) 中對應關係的空間相容性矩陣。從黃色儲存格中我們可以看到，C6 和 c7 是外點，但它們偶然地與一些內點顯示出很高的相容性分數。結果，這些外點將不可避免地被捲入模型估計過程中，導致性能下降。

In this paper, we propose a new global measure of the similarity between two correspondences. Specifically, we first binarize the spatial compatibility matrix into the hard form, as shown in Fig. 1 (c). Then, for two correspon- dences which are compatible, we compute the number of their commonly compatible correspondences in the global set. That is, we compute the number of correspondences that are simultaneously compatible with both of them as the new similarity between them. The globally common com- patibility is set to 0 for any two incompatible correspon- dences. Therefore, the similarity of two inliers is at least the number of inliers excluding themselves among all the correspondences. However, the outliers do not have such good properties. A toy example is shown in Fig. 1. There are five inliers {c1,c2, c3, c4, c5} and two outliers {c6, c7} in Fig. 1 (a). From Fig. 1 (b) and Fig. 1 (c), we can see that the outliers C6 and c7 are compatible with some inliers, and the inliers are compatible with each other. But from Fig. 1 (d), we can see that the similarities between any two inliers are large while the similarities between the outliers with the other correspondences are small. To be specific, in Fig. 1 (d), the similarities within the inliers {c1, c2, c3, c4, c5} are no less than 3, while the similarities related with the outliers {c6, c7} are no more than 1. Therefore, the global compati- bility matrix in Fig. 1 (d) can better distinguish inliers from outliers. Since the new measure can be expressed as the ma- trix product of the traditional first order metric (See Eq. 8), we name it as the second order spatial compatibility (SC2).

在本文中，我們提出了一種新的全域度量來衡量兩個對應關係之間的相似性。具體來說，我們首先將空間相容性矩陣二元化為硬形式，如圖 1(c) 所示。然後，對於兩個相容的對應關係，我們計算它們在全域集合中共同相容的對應關係的數量。也就是說，我們計算同時與它們兩者都相容的對應關係的數量，作為它們之間新的相似性。對於任何兩個不相容的對應關係，全域共同相容性設定為 0。因此，兩個內點的相似性至少是所有對應關係中排除它們自身的內點數量。然而，外點不具備這樣好的特性。圖 1 中展示了一個玩具範例。在圖 1(a) 中有五個內點 {c1, c2, c3, c4, c5} 和兩個外點 {c6, c7}。從圖 1(b) 和圖 1(c) 中，我們可以看到外點 C6 和 c7 與一些內點相容，且內點之間也相互相容。但從圖 1(d) 中，我們可以看到任何兩個內點之間的相似性很大，而外點與其他對應關係之間的相似性很小。具體來說，在圖 1(d) 中，內點 {c1, c2, c3, c4, c5} 之間的相似性不小於 3，而與外點 {c6, c7} 相關的相似性不大於 1。因此，圖 1(d) 中的全域相容性矩陣可以更好地將內點與外點區分開來。由於這個新的度量可以表示為傳統一階度量的矩陣乘積（見方程式 8），我們將其命名為二階空間相容性（SC2）。

The proposed second order spatial compatibility mea- sure SC2 has several advantages. 1) The inliers are much easier distinguished from the outliers. Suppose we have m inliers in n correspondences. The scores between any t- wo inliers would be no less than m-2. However, its difficult for an outlier to be simultaneously compatible with multiple correspondences and the score for it will be much smaller. 2) The traditional algorithms such as RANSAC and its vari- ants [21, 25, 42, 53] need a large number of randomly sam- plings to find an outlier-free set for robust model estima- tion. However, based on the proposed SC² matrix, for each row vector corresponding to an inlier, we can easily find an outlier-free set by selecting the top k correspondences with the highest scores. In this way, the m valid samplings can be obtained by traversing all the n rows of the SC2 matrix. Therefore, we can ensure m reliable model estimations by only n samplings, which makes the model estimation more efficient and robust. 3) We theoretically prove that the SC2 matrix significantly reduces the probability of wrong sam- pling from a probabilistic view. We define an error event, in which the score between two inliers is smaller than that between an inlier and an outlier. By computing the proba- bility distributions of this event for both the traditional first order metric and our second order metric, the SC2 matrix is much more robust to obtain reliable sampling (see Fig. 3).

我們提出的二階空間相容性度量 SC2 有幾個優點。1) 內點更容易與外點區分。假設在 n 個對應關係中有 m 個內點。任何兩個內點之間的分數將不小於 m-2。然而，一個外點很難同時與多個對應關係相容，其分數將會小得多。2) 傳統演算法如 RANSAC 及其變體 [21, 25, 42, 53] 需要大量隨機採樣才能找到一個無外點集合以進行穩健的模型估計。然而，基於我們提出的 SC² 矩陣，對於每個對應於內點的行向量，我們可以透過選擇分數最高的 k 個對應關係輕鬆找到一個無外點集合。透過這種方式，可以透過遍歷 SC² 矩陣的所有 n 行來獲得 m 個有效的採樣。因此，我們僅需 n 次採樣即可確保 m 次可靠的模型估計，這使得模型估計更有效率和穩健。3) 我們從機率的角度理論上證明了 SC² 矩陣顯著降低了錯誤採樣的機率。我們定義了一個錯誤事件，其中兩個內點之間的分數小於一個內點和一個外點之間的分數。透過計算傳統一階度量和我們二階度量下此事件的機率分佈，SC² 矩陣在獲得可靠採樣方面更為穩健（見圖 3）。

Based on the SC² measure, we design a full pipeline for point cloud registration, called SC2-PCR. Following [5, 14, 50], it first selects several seeds that are likely to be inliers. Then we select a consensus set for each seed by finding those having the highest SC2 scores with it. In order to further exclude outliers, a two-stage sampling strategy is carried out in a coarse-to-fine manner. Finally, we use the weighted SVD to estimate a tentative model for each seed and select the best one as the final output. In a nutshell, this paper distinguishes itself from existing methods in the following aspects.

基於 SC² 度量，我們設計了一個完整的點雲配準流程，稱為 SC2-PCR。遵循 [5, 14, 50] 的方法，它首先選擇幾個可能是內點的種子。然後，我們透過找到與每個種子具有最高 SC² 分數的對應關係來為每個種子選擇一個共識集。為了進一步排除外點，我們採用了從粗到精的兩階段採樣策略。最後，我們使用加權 SVD 為每個種子估計一個初步模型，並選擇最佳模型作為最終輸出。總而言之，本文在以下幾個方面與現有方法有所不同。

• A second order spatial compatibility metric called SC2 is proposed. We theoretically prove that SC2 sig- nificantly reduces the probability of an outlier being involved in the consensus set. Since the proposed method encodes richer information beyond the first or- der metric, it enhances the robustness against outliers.
• Compared with state-of-the-art deep learning methods such as [5, 19, 35, 43], our method is a light weighted solution that does not need training. It shows no bias across different datasets and generalizes well on vari- ous scenarios, which is also shown in the experiments.
• The proposed method is general. Although we im- plement it in a handcrafted fashion, it could be easily plugged into other deep learning frameworks such as PointDSC [5]. We show in the experiment that PointD- SC produces better results when combined with our method.

• 提出了一種稱為 SC² 的二階空間相容性度量。我們從理論上證明 SC² 顯著降低了外點被捲入共識集的機率。由於所提出的方法編碼了比一階度量更豐富的資訊，因此它增強了對抗外點的穩健性。
• 與最先進的深度學習方法（如 [5, 19, 35, 43]）相比，我們的方法是一種輕量級解決方案，不需要訓練。它在不同資料集之間沒有偏差，並且在各種場景中具有良好的泛化能力，這在實驗中也得到了證明。
• 所提出的方法具有通用性。雖然我們以手工製作的方式實現它，但它可以很容易地插入其他深度學習框架中，例如 PointDSC [5]。我們在實驗中表明，PointDSC 與我們的方法相結合會產生更好的結果。

## 2. Related Work

## 2. 相關工作

3D Feature Matching. The widely used Iterative Clos- est Point [10] and its variants [12, 46, 49] establish corre- spondences by searching the nearest neighbor in coordinate space. Instead of using the distance in coordinate space, lo- cal feature descriptors aim to build 3D matches in feature space. Hand-crafted descriptors represent local feature by encoding spatial distribution histogram [26, 28, 31,52], ge- ometric histogram [15, 47, 48] or other attributes [61]. Re- cently, the deep learning techniques are also introduced to learn 3D local descriptors. The pioneering 3DMatch [62] builds a Siamese Network for extracting local descriptors. A number of recent networks try to boost the performance by designing rotation invariant modules [1,22,23,58], fully

3D 特徵匹配。廣泛使用的迭代最近點（Iterative Closest Point）[10] 及其變體 [12, 46, 49] 透過在座標空間中搜索最近鄰來建立對應關係。與使用座標空間中的距離不同，局部特徵描述子旨在特徵空間中建立 3D 匹配。手工製作的描述子透過編碼空間分佈直方圖 [26, 28, 31, 52]、幾何直方圖 [15, 47, 48] 或其他屬性 [61] 來表示局部特徵。最近，深度學習技術也被引入來學習 3D 局部描述子。開創性的 3DMatch [62] 建立了一個孿生網路（Siamese Network）來提取局部描述子。許多最近的網路試圖透過設計旋轉不變模組 [1, 22, 23, 58]、全

convolution modules [20], feature detection modules [6,59], coarse-to-fine modules [44, 60] and overlapping learning modules [29, 55]. Although these methods achieve remark- able performance improvement, it can hardly establish a to- tally outlier-free correspondence set.

卷積模組 [20]、特徵檢測模組 [6, 59]、由粗到精模組 [44, 60] 和重疊學習模組 [29, 55] 來提升性能。儘管這些方法取得了顯著的性能提升，但很難建立一個完全沒有外點的對應集。

Traditional Model Fitting. The model fitting methods estimate the geometric model from a noise correspondence set. The classic RANSAC [25] provides the most common- ly adopted generation-and-verification pipeline for robust removing outliers. Many of its variants [21, 42, 53] intro- duce new sampling strategies and local optimization [21] to accelerate the estimation or boost the robustness. More recently, Graph-cut RANSAC [8] introduces the graphcut technique to better performing local optimization. Magsac [9] proposes a o-consensus to build a threshold-free method for RANSAC. For specific 3D model-fitting, FGR [66] uses the Geman-McClure cost function and estimates the mod- el through graduated non-convexity optimization. TEAS- ER [56] reformulates the registration problem using a Trun- cated Least Squares (TLS) cost and solving it by a general graph-theoretic framework.

傳統模型擬合。模型擬合方法從一個帶有雜訊的對應集合中估計幾何模型。經典的 RANSAC [25] 提供了最常用的生成與驗證流程，用於穩健地移除外點。它的許多變體 [21, 42, 53] 引入了新的採樣策略和局部優化 [21] 來加速估計或提升穩健性。最近，Graph-cut RANSAC [8] 引入了圖割技術來更好地執行局部優化。Magsac [9] 提出了一種 σ-共識來為 RANSAC 建立一個無閾值的方法。對於特定的 3D 模型擬合，FGR [66] 使用 Geman-McClure 成本函數，並透過漸進非凸優化來估計模型。TEASER [56] 使用截斷最小二乘（TLS）成本重新表述了配準問題，並透過一個通用的圖論框架來解決它。

Learning based Model Fitting. Recent works also adopt deep learning techniques, which were first studied in 2D matching area, to model fitting tasks. The 2D corre- spondence selection network CN-Net [41] and its variants [13,16,17,37,51,63-65] formulate the model fitting as com- bination of a correspondence classification module and a model estimation module. Recent attempts [5,18,19,35,43] also introduce deep learning network for 3D correspon- dence pruning. 3DRegNet [43] reformulates the CN-Net [41] into 3D form and designs a regression module to solve rigid transformaton. DGR [19] introduces the fully convo- lution to better capture global context. PointDSC [5] de- velops a spatial consistency based non-local module and a Neural Spectral matching to accelerate the model genera- tion and selection. DetarNet [18] presents decoupling solu- tions for translation and rotation. DHVR [35] exploits the deep hough voting to identify the consensus from the Hough space, so as to predict the final transformation.

基於學習的模型擬合。最近的研究也採用了深度學習技術來進行模型擬合任務，這些技術最初是在 2D 匹配領域進行研究的。2D 對應選擇網路 CN-Net [41] 及其變體 [13, 16, 17, 37, 51, 63-65] 將模型擬合問題表述為對應分類模組和模型估計模組的組合。最近的嘗試 [5, 18, 19, 35, 43] 也引入了深度學習網路用於 3D 對應修剪。3DRegNet [43] 將 CN-Net [41] 重新表述為 3D 形式，並設計了一個迴歸模組來解決剛性變換問題。DGR [19] 引入了全卷積以更好地捕捉全域上下文。PointDSC [5] 開發了一個基於空間一致性的非局部模組和一個神經譜匹配來加速模型生成和選擇。DetarNet [18] 提出了平移和旋轉的解耦方案。DHVR [35] 利用深度霍夫投票從霍夫空間中識別共識，從而預測最終的變換。

## 3. Method

## 3. 方法

### 3.1. Problem Formulation

### 3.1. 問題定義

Given two point clouds to be aligned: source point cloud X = {xi ∈ R³ | i = 1,..., N} and target point cloud Y = {yj ∈ R3 | j = 1, ..., Ny}, we first extract local fea- tures for both of them. Then, for each point in the source point cloud, we find its nearest neighbor in the feature s- pace among the target points to form N pairs of putative correspondences. The proposed method estimates the rigid transformation between the two point clouds, i.e., the rota- tion matrices (R ∈ R3×3) and translation vectors (t ∈ R³). The pipeline of proposed method is shown in Fig. 2.

給定兩個待對齊的點雲：源點雲 X = {xi ∈ R³ | i = 1,..., N} 和目標點雲 Y = {yj ∈ R³ | j = 1, ..., Ny}，我們首先為它們提取局部特徵。然後，對於源點雲中的每個點，我們在目標點的特徵空間中找到其最近鄰，以形成 N 對假定的對應關係。所提出的方法估計了兩個點雲之間的剛性變換，即旋轉矩陣（R ∈ R³×³）和平移向量（t ∈ R³）。所提出方法的流程如圖 2 所示。

### 3.2. Second Order Spatial Compatibility

### 3.2. 二階空間相容性

To analyze the effectiveness of a metric for sampling, we first define the probability of an ambiguity event as:
Pam(M) = P(Min,out > Min,in), (1)
where M is a specific metric for measuring correspondence- wise similarity. P(Z) is the probability of an event Z (For convenience we use this notation in the following part). Min,out is the similarity between an inlier and an outlier, while Min,in is that between two inliers. When Min,out > Min,in, outlier is the near neighbor of inlier, so the metric based sampling tends to fail. So the lower this probability is, the more robust the metric based sampling will be.

為了分析採樣度量的有效性，我們首先將模糊事件的機率定義為：
Pam(M) = P(Min,out > Min,in), (1)
其中 M 是衡量對應相似性的特定度量。P(Z) 是事件 Z 的機率（為方便起見，我們在下文中使用此表示法）。Min,out 是內點與外點之間的相似性，而 Min,in 是兩個內點之間的相似性。當 Min,out > Min,in 時，外點是內點的近鄰，因此基於度量的採樣容易失敗。所以這個機率越低，基於度量的採樣就越穩健。

We first introduce the commonly used first order spatial compatibility (SC) measure [5,35,36,45]. The SC measure between correspondence i and j is defined as follows:
SCij = $(dij), dij = |d(xi, xj) – d(yi, yj)| (2)
in which (xi, Yi) and (xj, yj) are the matched points of cor- respondences i and j. (·) is a monotonically decreasing kernel function. d(, ) is the Euclidean distance. As shown in Fig. 1, the distance difference between two inliers din,in should be equal to 0 due to the length consistency of rigid transformation. However, because of the noise introduced by data acquisition and point cloud downsampling, din,in is not exactly equal to 0, but less than a threshold dthr. For convenience, we assume din, in is uniformly distributed over dthr and get the probability density function (PDF) of the distance difference between two inliers as follows:
PDFin,in (l) = 1/dthr, 0 ≤ I ≤ dthr. (3)

我們首先介紹常用的一階空間相容性（SC）度量 [5, 35, 36, 45]。對應 i 和 j 之間的 SC 度量定義如下：
SCij = $(dij), dij = |d(xi, xj) – d(yi, yj)| (2)
其中 (xi, Yi) 和 (xj, yj) 是對應 i 和 j 的匹配點。(·) 是一個單調遞減的核心函數。d(·, ·) 是歐幾里德距離。如圖 1 所示，由於剛性變換的長度一致性，兩個內點之間的距離差 din,in 應等於 0。然而，由於資料獲取和點雲降採樣引入的雜訊，din,in 並不完全等於 0，但小於一個閾值 dthr。為方便起見，我們假設 din,in 在 dthr 上均勻分佈，並得到兩個內點之間距離差的機率密度函數（PDF）如下：
PDFin,in (l) = 1/dthr, 0 ≤ I ≤ dthr. (3)

Differently, there is no related constraint between two out- liers or an inlier and an outlier due to the random distri- bution of outliers. We consider the distance difference be- tween two unrelated points to be identically distributed and assume the probability density function (PDF) as F(·):
PDFin,out(l) = F(l), PDF out,out(l) = F(l); 0 < 1 < dr (4)
where dr is the range of din,out and dout,out. An empirical F function on 3DMatch dataset is presented in Fig 3 (a). Obviously, dr is much greater than dthr. So we can approx- imate that F(l) is a constant within (0, dthr) as follows:
F(l) = fo, 0 ≤ I ≤dthr. (5)

不同地，由於外點的隨機分佈，兩個外點之間或一個內點和一個外點之間沒有相關的約束。我們認為兩個不相關點之間的距離差是同分佈的，並假設機率密度函數（PDF）為 F(·)：
PDFin,out(l) = F(l), PDF out,out(l) = F(l); 0 < 1 < dr (4)
其中 dr 是 din,out 和 dout,out 的範圍。圖 3(a) 中呈現了 3DMatch 資料集上的經驗 F 函數。顯然，dr 遠大於 dthr。因此我們可以近似認為 F(l) 在 (0, dthr) 內是一個常數，如下所示：
F(l) = fo, 0 ≤ I ≤dthr. (5)

Next, we consider the ambiguity probability of SC as Eq. (1), i.e., P(SCin,out > SCin,in). According to Eq. (2), (3), (4) and (5), it can be computed as follows:
P(SCin,out > SCin,in) = P(din,out < din,in)
= ∫dthr 0 PDFin,in(l) ∫l 0 PDFin,out(x)dxdl
= ∫dthr 0 1/dthr * fodxdl = dthr*fo/2 (6)

接下來，我們考慮 SC 的模糊機率，如方程式 (1)，即 P(SCin,out > SCin,in)。根據方程式 (2)、(3)、(4) 和 (5)，可以計算如下：
P(SCin,out > SCin,in) = P(din,out < din,in)
= ∫dthr 0 PDFin,in(l) ∫l 0 PDFin,out(x)dxdl
= ∫dthr 0 1/dthr * fodxdl = dthr*fo/2 (6)

Figure 2. Pipeline of our method. 1. Computing correspondence-wise second order spatial compatibility measure. 2. Selecting some reliable correspondences as seeds. 3. Performing the two-stage sampling around each seed. 4. Performing local spectral matching to generate an estimation of R and t for each seed. 5. Selecting the best estimation as final result.

圖 2. 我們方法的流程。1. 計算對應的二階空間相容性度量。2. 選擇一些可靠的對應作為種子。3. 圍繞每個種子執行兩階段採樣。4. 執行局部譜匹配，為每個種子生成 R 和 t 的估計。5. 選擇最佳估計作為最終結果。

Taking the 3DMatch [62] dataset as an example. Follow- ing [5], we set dthr = 10cm, then the ambiguity probability of SC measure is about 0.1 as shown in Fig. 3 (a). Consid- ering the amount of outliers might be large, the number of mistakes is not negligible even at this probability.

以 3DMatch [62] 資料集為例。遵循 [5]，我們設定 dthr = 10cm，則 SC 度量的模糊機率約為 0.1，如圖 3(a) 所示。考慮到外點的數量可能很大，即使在這個機率下，錯誤的數量也是不可忽略的。

Next, we describe the proposed second order spatial compatibility measure (SC2 ∈ RN×N). Specifically, we first build a hard compatibility matrix C (C∈RN×N):
Cij = [1; dij ≤dthr, (0; dij > dthr. (7)
C considers that two correspondences satisfying length con- sistency are compatible (Ci,j is set to 0 when i = j). Then, SC counts the number of common compatibility corre- spondences of i and j when they are compatible, as follows:
SC = Cij. ∑Nk=1 Cik Ckj. (8)

接下來，我們描述所提出的二階空間相容性度量 (SC² ∈ RN×N)。具體來說，我們首先建立一個硬相容性矩陣 C (C∈RN×N)：
Cij = [1; dij ≤dthr, (0; dij > dthr. (7)
C 認為滿足長度一致性的兩個對應是相容的（當 i = j 時，Ci,j 設定為 0）。然後，SC² 計算當 i 和 j 相容時，它們的共同相容對應的數量，如下所示：
SC² = Cij. ∑Nk=1 Cik Ckj. (8)

Similarly, we analyze the ambiguity probability of this mea- sure, i.e., P(SCin,out > SCin,in). Suppose there are N pairs of correspondences and the inlier ratio is a. Then, we can prove that P(SCin,out > SCin,in) can be expressed as (The derivation is presented in the supplement materials):
P(SCin,out > SCin,in) = p · P(X > (Ν· α − 2)),
X ~ S((Να − 1)p + (N(1 − a) – 1)p², N(1 − a)p²),
p = dthr fo, (9)
where S(,) is the Skellam distribution [30, 32, 33]. Ac- cording to the properties of Skellam distribution, the val- ue of P(SCin,out > SCin, in) is going to approach 0 very quickly as a increases. In order to make a clearer compar- ison between the proposed SC2 measure and SC measure, we plot the curves of ambiguity probability for both of them according to Eq. (6) and (9). As shown in Fig. 3 (b), the ambiguity probability of the proposed SC2 measure is sig- nificantly lower than the SC measure, even when the inli- er rate is close to 0. It shows that using SC2 measure as guidance for sampling is easier to obtain an outlier-free set. When the inliers rate reaches 1%, the ambiguity probability of SC2 measure is close to 0, which ensures a robust sam- pling anti the data with low inlier rate.

同樣地，我們分析這個度量的模糊機率，即 P(SC²in,out > SC²in,in)。假設有 N 對對應關係，內點比例為 α。那麼，我們可以證明 P(SC²in,out > SC²in,in) 可以表示為（推導過程見補充材料）：
P(SC²in,out > SC²in,in) = p · P(X > (Ν· α − 2)),
X ~ S((Να − 1)p + (N(1 − a) – 1)p², N(1 − a)p²),
p = dthr fo, (9)
其中 S(·,·) 是 Skellam 分佈 [30, 32, 33]。根據 Skellam 分佈的性質，隨著 α 的增加，P(SC²in,out > SC²in,in) 的值將非常迅速地趨近於 0。為了更清楚地比較所提出的 SC² 度量和 SC 度量，我們根據方程式 (6) 和 (9) 繪製了它們的模糊機率曲線。如圖 3(b) 所示，即使內點率接近 0，所提出的 SC² 度量的模糊機率也明顯低於 SC 度量。這表明使用 SC² 度量作為採樣指導更容易獲得無外點集。當內點率達到 1% 時，SC² 度量的模糊機率接近 0，這確保了在低內點率資料下的穩健採樣。

Figure 3. (a) The empirical probability density function (F) of din,out and dout,out. (b) The ambiguity probability. SC is spatial compatibility measure. SC²-N (N = 5000, 2500, 1000) is the sec- ond order spatial compatibility measure with N correspondences.

圖 3. (a) din,out 和 dout,out 的經驗機率密度函數 (F)。(b) 模糊機率。SC 是空間相容性度量。SC²-N (N = 5000, 2500, 1000) 是具有 N 個對應的二階空間相容性度量。

### 3.3. Reliable Seed Selection

### 3.3. 可靠種子選擇

As mentioned above, there are high similarities between inlier correspondences by the proposed SC2 measure. Then, as long as we find an inlier correspondence, we can con- struct a consensus set by finding its k nearest neighbors in the metric space. Obviously, traversing all the correspon- dences must find an inlier, but it is not necessary. We on- ly need to pick some reliable points called seed points to accelerate the registration process. We perform spectral matching technique [36] to select seed points. Specifical- ly, we first build the similarity matrix for all of the corre- spondences and normalize the value in the matrix to 0-1, following [36]. Then, following [5, 36], the association of each correspondence with the leading eigen-vector is adopt- ed as the confidence for this correspondence. The leading eigen-vetctor is solved by power iteration algorithm [39]. In order to ensure an even distribution of seed points, the cor- respondences with local maximum confidence score within its neighborhood with radius R are selected. The number of seed points (N) is determined by a proportion of the num- ber of whole correspondences.

如上所述，我們提出的 SC² 度量在內點對應之間具有很高的相似性。因此，只要我們找到一個內點對應，就可以透過在度量空間中找到其 k 個最近鄰來構建一個共識集。顯然，遍歷所有對應關係必然會找到一個內點，但這並非必要。我們只需要挑選一些可靠的點，稱為種子點，來加速配準過程。我們採用譜匹配技術 [36] 來選擇種子點。具體來說，我們首先為所有對應關係建立相似性矩陣，並根據 [36] 將矩陣中的值歸一化到 0-1。然後，根據 [5, 36]，每個對應關係與領導特徵向量的關聯被用作該對應關係的置信度。領導特徵向量透過冪迭代演算法 [39] 求解。為了確保種子點的均勻分佈，我們選擇在其半徑 R 的鄰域內具有局部最大置信度分數的對應關係。種子點的數量 (Ns) 由所有對應關係數量的比例決定。

### 3.4. Two-stage Consensus Set Sampling

### 3.4. 兩階段共識集採樣

As some seed points are selected, we extend each of them into a consensus set. We adopt a two stage selec- tion strategy to perform a coarse-to-fine sampling. In the first stage, we select K₁ correspondences for each seed by finding its top-K₁ neighbors in the SC2 measure space. As mentioned before, the ambiguity probability P(SCin,out > SCin.in) is very small. Thus, when a seed is an inlier corre- spondences, the consensus set also mainly contains inliers. Meanwhile, the similarity expressed by SC2 measure fo- cuses on global information instead of local consistency. Therefore, the neighbors selected in the SC2 measure space are distributed more evenly rather than clustered together, which benefits the estimation of rigid transformation [5].

選定一些種子點後，我們將它們各自擴展成一個共識集。我們採用兩階段選擇策略來進行由粗到精的採樣。在第一階段，我們透過在 SC² 度量空間中找到每個種子的前 K₁ 個鄰居來為其選擇 K₁ 個對應。如前所述，模糊機率 P(SC²in,out > SC²in,in) 非常小。因此，當一個種子是內點對應時，其共識集也主要包含內點。同時，SC² 度量所表示的相似性著重於全域資訊而非局部一致性。因此，在 SC² 度量空間中選擇的鄰居分佈更均勻，而不是聚集在一起，這有利於剛性變換的估計 [5]。

The second stage sampling operation is adopted to fur- ther filter potential outliers in the set obtained in the first stage. The SC2 matrices are reconstructed within each set produced by the first stage instead of the whole set. We se- lect top-K2 (K2 < K₁) correspondences of the seed by the newly constructed local SC2 matrices. As shown in Fig. 3 (b), since the higher inlier rate ensures a lower ambiguity probability, so that the potential outliers can also be fur- ther pruned. Note that we only discussed the case that the seed point is inlier. In fact, when the seed point is an out- lier, it can also form a local consistency, especially when there are aggregated false matching in the correspondence set. We encourage these sets to also generate hypothesis and filter them at the final hypothesis selection step (Sec- tion 3.6) rather than at the early stage. In this way, we can avoid some correct assumptions being filtered out early.

第二階段採樣操作旨在進一步過濾第一階段獲得的集合中的潛在外點。SC² 矩陣在第一階段產生的每個集合內（而非整個集合）被重建。我們透過新建構的局部 SC² 矩陣選擇種子的前 K₂ (K₂ < K₁) 個對應。如圖 3(b) 所示，由於較高的內點率確保了較低的模糊機率，因此潛在的外點也可以被進一步修剪。請注意，我們只討論了種子點是內點的情況。事實上，當種子點是外點時，它也可以形成局部一致性，特別是當對應集合中存在聚集的錯誤匹配時。我們鼓勵這些集合也生成假設，並在最終的假設選擇步驟（第 3.6 節）而不是在早期階段過濾它們。透過這種方式，我們可以避免一些正確的假設被過早地過濾掉。

### 3.5. Local Spectral Matching

### 3.5. 局部譜匹配

In this step, we perform the weighted SVD [3] on the consensus set to generate an estimation of rigid transforma- tion for each seed. Although the previous proposed sam- pling strategy can obtain outlier-free correspondence set, we find that the weighted SVD achieves better performance than treating all corrrespondences equally. This may be be- cause that the inliers still have different degree of noises. So correspondences with bigger noise should have smaller weights when estimating rigid transformation. Traditional spectral matching [36] method analyzes the SC matrix to assign weight for each correspondence, which is effected by ambiguity problem [5]. Since the proposed SC2 mea- sure is more robust against ambiguity, we also replace the SC matrix with SC2 measure. In order to facilitate matrix analysis, we convert the SC2 measure into soft form (SC2) as follows:
SC2 = Č · (Č × Č),
Čij = ReLU(1-dj/dhr), (1 ≤ i ≤ K2,1 ≤ j ≤ K2) (10)
where is Hadamard product and × is matrix product. Then we conduct local spectral decomposition on SC2 to obtain a weight wi for correspondences i. Finally, the rotation Rk and translation tk of seed k are computed by performing weighted SVD [19] within its consensus set.

在此步驟中，我們對共識集執行加權 SVD [3] 以為每個種子生成剛性變換的估計。儘管先前提出的採樣策略可以獲得無外點的對應集，但我們發現加權 SVD 的性能優於平等對待所有對應。這可能是因為內點仍然具有不同程度的雜訊。因此，在估計剛性變換時，具有較大雜訊的對應應具有較小的權重。傳統的譜匹配 [36] 方法分析 SC 矩陣為每個對應分配權重，這會受到模糊性問題的影響 [5]。由於我們提出的 SC² 度量對模糊性更具穩健性，我們也用 SC² 度量取代了 SC 矩陣。為了便於矩陣分析，我們將 SC² 度量轉換為軟形式 (SC²) 如下：
SC² = Č · (Č × Č),
Čij = ReLU(1-dj/dhr), (1 ≤ i ≤ K2,1 ≤ j ≤ K2) (10)
其中 · 是哈達瑪積，× 是矩陣積。然後我們對 SC² 進行局域譜分解以獲得對應 i 的權重 wi。最後，透過在其共識集內執行加權 SVD [19] 來計算種子 k 的旋轉 Rk 和平移 tk。

### 3.6. Hypothesis Selection

### 3.6. 假設選擇

In final step, we select the best estimation over the rigid transformations produced by all the consensus sets. We use the same criteria as RANSAC [25], i.e. inlier count, to se- lect the final estimation. Specifically, for the estimation of k-th seed Rk and tk, we count the number of correspon- dences that satisfy the constraints of Rk and tk by a prede- fined error threshold (7) as follows:
countk = ∑Ni=1 [||Rkxi + tk|| < T], (11)
where [] is Iverson bracket. The Rk and tk with the highest inlier count are selected as the final results.

在最後一步，我們從所有共識集產生的剛性變換中選擇最佳估計。我們使用與 RANSAC [25] 相同的標準，即內點計數，來選擇最終估計。具體來說，對於第 k 個種子 Rk 和 tk 的估計，我們計算滿足 Rk 和 tk 約束的對應數量，其誤差閾值 (τ) 預先定義如下：
countk = ∑Ni=1 [||Rkxi + tk|| < T], (11)
其中 [·] 是艾佛森括號。具有最高內點計數的 Rk 和 tk 被選為最終結果。

## 4. Experiment

## 4. 實驗

### 4.1. Datasets and Experimental Setup

### 4.1. 資料集與實驗設定

Indoor scenes. We use the 3DMatch benchmark [62] for evaluating the performance on indoor scenes. For each pair of point clouds, we first use 5cm voxel grid to down- sample the point cloud. Then we extract the local feature descriptors and match them to form the putative correspon- dences. Following [5], we use FPFH [47] (handcrafted de- scriptor) and FCGF [20] (learned descriptor) as feature de- scriptors respectively. The partial overlapping registration benchmark 3DLoMatch [29] is also adopted to further veri- fy the performance of the method. Following [5,35], we use FCGF [20] and Predator [29] descriptors to generate puta- tive correspondences.

室內場景。我們使用 3DMatch基準 [62] 來評估室內場景的性能。對於每對點雲，我們首先使用 5cm 體素網格對點雲進行降採樣。然後我們提取局部特徵描述子並進行匹配，以形成假定的對應關係。遵循 [5]，我們分別使用 FPFH [47]（手工描述子）和 FCGF [20]（學習描述子）作為特徵描述子。部分重疊配準基準 3DLoMatch [29] 也被用來進一步驗證該方法的性能。遵循 [5, 35]，我們使用 FCGF [20] 和 Predator [29] 描述子來生成假定的對應關係。

Outdoor scenes. We use the KITTI dataset [27] for test- ing the effectiveness on outdoor scenes. Following [19, 20], we choose the 8 to 10 scenes, obtaining 555 pairs of point clouds for testing. Then we construct 30cm voxel grid to down-sampling the point cloud and form the correspon- dences by FPFH and FCGF descritors respectively.

戶外場景。我們使用 KITTI 資料集 [27] 來測試在戶外場景中的有效性。遵循 [19, 20]，我們選擇了 8 到 10 個場景，獲得了 555 對點雲進行測試。然後我們建構了 30cm 的體素網格來對點雲進行降採樣，並分別使用 FPFH 和 FCGF 描述子形成對應關係。

Evaluation Criteria. Following [5], we first report the registration recall (RR) under an error threshold. For the indoor scenes, the threshold is set to (15°, 30 cm), while the threshold of outdoor scenes is (5°, 60 cm). For a pair of point clouds, we calculate the errors of translation and rotation estimation separately. We compute the isotropic rotation error (RE) [38] and L2 translation error (TE) [19]. Following [5], we also report the outlier removal results us- ing following three evaluation criteria: inlier precision (IP), inlier recall (IP) and F1-measure (F1).

評估標準。遵循 [5]，我們首先報告在誤差閾值下的配準召回率（RR）。對於室內場景，閾值設定為（15°, 30 cm），而室外場景的閾值為（5°, 60 cm）。對於一對點雲，我們分別計算平移和旋轉估計的誤差。我們計算等向性旋轉誤差（RE）[38] 和 L2 平移誤差（TE）[19]。遵循 [5]，我們還使用以下三個評估標準報告了離群值移除結果：內點精確率（IP）、內點召回率（IR）和 F1-measure（F1）。

Implementation Details. When computing the SC2 ma- trix, the dthr is set to the twice as the voxel size for down- sampling (10cm for indoor scenes and 60cm for outdoor scenes). The number of seed (Ns in Section 3.3) is set to 0.2 * N, where N is the number of correspondences. When sampling consensus set, we select 30 nearest neighbors (K1 = 30) of seed point at first sampling stage, and remain 20 correspondences (K2 = 20) to form the consensus set. All the experiments are conducted on a machine with an INTEL Xeon E5-2620 CPU and a single NVIDIA GTX1080Ti.

實現細節。在計算 SC² 矩陣時，dthr 設定為降採樣體素大小的兩倍（室內場景為 10cm，室外場景為 60cm）。種子數量（第 3.3 節中的 Ns）設定為 0.2 * N，其中 N 是對應關係的數量。在採樣共識集時，我們在第一採樣階段選擇種子點的 30 個最近鄰（K₁ = 30），並保留 20 個對應關係（K₂ = 20）以形成共識集。所有實驗均在配備 INTEL Xeon E5-2620 CPU 和單一 NVIDIA GTX1080Ti 的機器上進行。

### 4.2. Evaluation on Indoor Scenes

### 4.2. 室內場景評估

We first report the results on 3DMatch dataset in Tab. 1. We compare our method with 13 baselines: DCP [54], PointNetLK [2], 3DRegNet [43], DGR [19], DHVR [35], PointDSC [5], SM [36], ICP [10], FGR [66], TEASER [56], GC-RANSAC [8], RANSAC [25], CG-SAC [45]. The first 6 methods are based on deep learning, while the last 7 meth- ods are traditional methods. For the deep learning methods, we use the provided pre-trained model of them for testing. The results of DHVR we tested have some difference with the original results, so we also report the results in their pa- per (DHVR-Origin in Tab. 1). DCP, PointNetLK and ICP are correspondence-free methods, so their results are not re- lated with the descriptor.

我們首先在表 1 中報告 3DMatch 資料集的結果。我們將我們的方法與 13 個基準進行比較：DCP [54]、PointNetLK [2]、3DRegNet [43]、DGR [19]、DHVR [35]、PointDSC [5]、SM [36]、ICP [10]、FGR [66]、TEASER [56]、GC-RANSAC [8]、RANSAC [25]、CG-SAC [45]。前 6 種方法基於深度學習，而後 7 種方法是傳統方法。對於深度學習方法，我們使用它們提供的預訓練模型進行測試。我們測試的 DHVR 結果與原始結果有些差異，因此我們也報告了他們論文中的結果（表 1 中的 DHVR-Origin）。DCP、PointNetLK 和 ICP 是無對應的方法，因此它們的結果與描述子無關。

Combined with FPFH. We first use the FPFH descrip- tor to generate the correspondences, in which the mean in- lier rate is 6.84%. As shown in Tab. 1, our method great- ly outperforms all the methods. For the registration recal- 1 (RR), which is the most important criterion, our method improves it by about 6% over the closest competitors a- mong the retested results (PointDSC and CG-SAC). Fol- lowing [5, 19], since the part of failed registration can gen- erate a large error of translation and rotation, we only com- pute the mean rotation (RE) and translation error (TE) of successfully registered point cloud pairs of each method to avoid unreliable metrics. This strategy of measuremen- t makes methods with high registration recall more likely to have large mean error, because they include more dif- ficult data when calculating mean error. Nevertheless, our method still achieves competitive results on RE and TE. Our method is slightly worse than PointDSC on TE and RE, and better than other methods. For the outlier rejection results, our method achieves the highest inlier recall (IR) and F1- measure. The F1 of our method outperforms the PointDSC by 5.35%.

結合 FPFH。我們首先使用 FPFH 描述子生成對應關係，其中平均內點率為 6.84%。如表 1 所示，我們的方法大大優於所有方法。對於最重要的標準——配準召回率（RR），我們的方法比重新測試結果中最近的競爭者（PointDSC 和 CG-SAC）提高了約 6%。遵循 [5, 19]，由於失敗的配準部分會產生較大的平移和旋轉誤差，我們只計算每種方法成功配準的點雲對的平均旋轉（RE）和平移誤差（TE），以避免不可靠的度量。這種測量策略使得具有高配準召回率的方法更有可能具有較大的平均誤差，因為它們在計算平均誤差時包含了更多困難的數據。儘管如此，我們的方法在 RE 和 TE 上仍然取得了具有競爭力的結果。我們的方法在 TE 和 RE 上略遜於 PointDSC，但優於其他方法。對於離群值剔除結果，我們的方法實現了最高的內點召回率（IR）和 F1-measure。我們方法的 F1 比 PointDSC 高出 5.35%。

Combined with FCGF. To further verify the perfor- mance, we also adopt the recent FCGF descriptor to gen- erate putative correspondences and report the registration results. The mean inlier rate of putative correspondences is 25.61%. As shown in Tab. 1, since the inlier rate is higher than the correspondences obtained by FPFH descriptor, the performace of all of the feature based methods are boosted. Our method still achieves the best performance over all the methods, achieving 1.84% improvement over RANSAC on registration recall. Besides, the mean registration time for a pair of point clouds are also reported. Since the proposed method only need to sample a few seed points with their consensus set rather than a large number of samples, it is competitive in terms of time-consuming. As shown in Tab. 1, the mean registration time of our methods is 0.11s, which is over 20x faster than RANSAC with 4M iterations.

結合 FCGF。為了進一步驗證性能，我們還採用了最近的 FCGF 描述子來生成假定的對應關係並報告配準結果。假定對應關係的平均內點率為 25.61%。如表 1 所示，由於內點率高於 FPFH 描述子獲得的對應關係，所有基於特徵的方法的性能都得到了提升。我們的方法在所有方法中仍然取得了最佳性能，在配準召回率上比 RANSAC 提高了 1.84%。此外，還報告了一對點雲的平均配準時間。由於所提出的方法只需要對少數種子點及其共識集進行採樣，而不是大量的樣本，因此在耗時方面具有競爭力。如表 1 所示，我們方法的平均配準時間為 0.11 秒，比具有 4M 次迭代的 RANSAC 快 20 倍以上。

Robustness to lower overlap. Furthermore, we report the results on the low-overlap scenarios: 3DLoMatch [29]. Following PointDSC [5] and DHVR [35], we adopt the FCGF [20] and Predator (There are two versions of Preda- tor, and we use the updated one) [29] descriptors to generate correspondences. The registration recall (RR), rotation er- ror (RE) and translation error (TE) are reported in Tab. 2. As shown by the data, whether combined with FCGF or Predator descriptor, our method achieves the highest regis- tration recall. Meanwhile, we also present some qualitative results on 3DLoMatch dataset. As shown in Fig. 4, our method can successfully align two point clouds where the low overlap ratio is clearly visible.

對較低重疊的穩健性。此外，我們報告了在低重疊場景下的結果：3DLoMatch [29]。遵循 PointDSC [5] 和 DHVR [35]，我們採用 FCGF [20] 和 Predator（有兩個版本的 Predator，我們使用更新的版本）[29] 描述子來生成對應關係。配準召回率（RR）、旋轉誤差（RE）和平移誤差（TE）報告在表 2 中。數據顯示，無論是結合 FCGF 還是 Predator 描述子，我們的方法都達到了最高的配準召回率。同時，我們也展示了一些在 3DLoMatch 資料集上的定性結果。如圖 4 所示，我們的方法可以成功對齊兩個低重疊率明顯可見的點雲。

Figure 4. Qualitative registration results on 3DLoMatch dataset.

圖 4. 3DLoMatch 資料集上的定性配準結果。

### 4.3. Evaluation on Outdoor Scenes

### 4.3. 戶外場景評估

In this experiments, we test on the outdoor KITTI [27] dataset. The results of DHVR [35], DGR [19], PointD- SC [5], RANSAC [25], FGR [66], CG-SAC [45] are report- ed as comparison. DHVR, DGR and PointDSC are deep learning based methods, while the remaining methods are non-learning. As shown in Tab. 3, our method remarkably surpasses the non-learning methods. The registration recall (RR) of our method is 25.23% higher than that of RANSAC when combined with FPFH descriptor, and 17.84% higher when combined with FCGF descriptor. The errors of trans- lation and rotation are also much lower than RANSAC. Our method with FPFH descriptor obtains the results of highest registration recall. For the learning networks, our method can achieve close performance with them.

在這些實驗中，我們在戶外 KITTI [27] 資料集上進行了測試。DHVR [35]、DGR [19]、PointDSC [5]、RANSAC [25]、FGR [66]、CG-SAC [45] 的結果作為比較。DHVR、DGR 和 PointDSC 是基於深度學習的方法，而其餘方法是非學習方法。如表 3 所示，我們的方法顯著優於非學習方法。當與 FPFH 描述子結合時，我們方法的配準召回率（RR）比 RANSAC 高 25.23%，當與 FCGF 描述子結合時，高 17.84%。平移和旋轉的誤差也遠低於 RANSAC。我們使用 FPFH 描述子的方法獲得了最高的配準召回率結果。對於學習網路，我們的方法可以達到與它們相近的性能。

### 4.4. Generalization and Robustness

### 4.4. 泛化性與穩健性

Generalization experiments. As reported above, deep learning based methods also achieve competitive perfor- mance on the 3DMatch, 3DLoMatch and KITTI datasets. Compared with these methods based on deep learning, the other advantage of our method is that it has no bias cross different datasets, while deep learning based methods have performance degradation when generalized between differ- ent datasets. To demonstrate this, we perform the gener- alization experiments on both 3DMatch, 3DLoMatch and KITTI dataset. For the recent learning based methods, in- cluding DGR and PointDSC, we report the cross-dataset re- sults. Specifically, we adopt their pre-trained model by KIT- TI to test on 3DMatch and 3DLoMatch and use 3DMatch's model to test on KITTI. As shown in Tab. 4, our method shows a significant improvement on registration recall with- out generalization problem. This further demonstrates the effectiveness of our method.

泛化實驗。如上所述，基於深度學習的方法在 3DMatch、3DLoMatch 和 KITTI 資料集上也取得了具有競爭力的性能。與這些基於深度學習的方法相比，我們方法的另一個優點是它在不同資料集之間沒有偏差，而基於深度學習的方法在不同資料集之間泛化時性能會下降。為了證明這一點，我們在 3DMatch、3DLoMatch 和 KITTI 資料集上進行了泛化實驗。對於最近的基於學習的方法，包括 DGR 和 PointDSC，我們報告了跨資料集的結果。具體來說，我們採用它們在 KITTI 上預訓練的模型在 3DMatch 和 3DLoMatch 上進行測試，並使用 3DMatch 的模型在 KITTI 上進行測試。如表 4 所示，我們的方法在配準召回率上顯示出顯著的提升，且沒有泛化問題。這進一步證明了我們方法的有效性。

Robustness Anti Noise. An important factor to measure the performance of the model fitting method is the stability under low inlier rate. In order to further verify the perfor- mance of our method, we report the results under different inlier ratio in Fig. 5. Specifically, we first use FPFH to generate initial match pairs for the 3DMatch dataset. Then, according to the inlier ratio, all the point cloud pairs are di- vided into 6 groups: <1%, 1% - 2%, 2% - 4%, 4% - 6%, 6% - 10% and > 10%. The number of point cloud pairs in each group is: 141, 208, 346, 252, 323 and 353. As shown in Fig. 5, when the inlier rate is less than 2%, our method is significantly better than other methods. It demonstrates the robustness anti noise of our method.

抗雜訊穩健性。衡量模型擬合方法性能的一個重要因素是其在低內點率下的穩定性。為了進一步驗證我們方法的性能，我們報告了在不同內點率下的結果，如圖 5 所示。具體來說，我們首先使用 FPFH 為 3DMatch 資料集生成初始匹配對。然後，根據內點率，所有點雲對被分為 6 組：<1%、1% - 2%、2% - 4%、4% - 6%、6% - 10% 和 > 10%。每組中的點雲對數量分別為：141、208、346、252、323 和 353。如圖 5 所示，當內點率低於 2% 時，我們的方法明顯優於其他方法。這證明了我們方法的抗雜訊穩健性。

Figure 5. The registration recall under the different inlier ratio of the putative correspondences.

圖 5. 在不同假定對應之內點率下的配準召回率。

### 4.5. Combined with learning network

### 4.5. 與學習網路結合

To verify the flexibility of our proposed approach, we combine our approach with a recent deep learning approach PointDSC [5]. It adopts the spatial consistency matrix to guide the non-local module. Since the proposed SC2 mea- sure is more robust to the ambiguity, we replace the spatial consistency matrix in PointDSC with SC2. Instead of re- training the network, we directly plugged our metrics into it. The registration recall of their vanilla version and com- bined version are shown in Tab. 5. It can be seen that adding our metrics can significantly boost the performance of net- work, especially for the generalization performance of net- work. It demonstrates that the proposed measure is flexible to combine with other methods.

為了驗證我們所提方法的靈活性，我們將其與最近的深度學習方法 PointDSC [5] 相結合。PointDSC 採用空間一致性矩陣來引導其非局部模組。由於我們提出的 SC² 度量對模糊性更具穩健性，我們用 SC² 取代了 PointDSC 中的空間一致性矩陣。我們沒有重新訓練網路，而是直接將我們的度量插入其中。表 5 顯示了其原始版本和組合版本的配準召回率。可以看出，加入我們的度量可以顯著提升網路的性能，特別是網路的泛化性能。這表明所提出的度量可以靈活地與其他方法相結合。

### 4.6. Ablation Study

### 4.6. 消融研究

In this section, we perform ablation study on 3DMatch dataset. We use the FPFH and FCGF descriptors to for- m correspondences respectively. The classic RANSAC is adopted as our baseline, as shown in the Row 1 and Row 7 of Tab. 6. We progressively add the proposed modules to the baseline and report the results.

在本節中，我們在 3DMatch 資料集上進行消融研究。我們分別使用 FPFH 和 FCGF 描述子來形成對應關係。經典的 RANSAC 被用作我們的基準，如表 6 的第 1 行和第 7 行所示。我們逐步將所提出的模組添加到基準中並報告結果。

Second Order Spatial Compatibility. We first add the Second Order Spatial Compatibility (SC2) measure as the guidance for the sampling of RANSAC. Each correspon- dence is extended into a consensus set by seaching the k- nearest neighbors in metric space. The Spatial Compatibil- ity (SC) adopted by previous works [5, 45, 57] is also uti- lized as the sampling guidance, and the results are reported as comparison. As shown in Row 1, 3 and 7, 9 of Tab. 6, the registration recall obtained by using SC2 measure as guidance is 14.79% higher than RANSAC when combined with FPFH, and 1.66% higher when combined with FCGF. Meanwhile, since SC2 measure can narrow the sampling s- pace, the mean registration time of SC2 measure is much s- maller than RANSAC. Besides, using SC2 measure as guid- ance can achieve better performance than using SC measure by comparing Row 2, 3 and 8, 9. This is because SC is dis- turbed by the ambiguity problem, while SC2 measure can eliminate the ambiguity.

二階空間相容性。我們首先加入二階空間相容性（SC²）度量作為 RANSAC 採樣的指導。每個對應透過在度量空間中搜索 k 個最近鄰擴展為一個共識集。先前工作 [5, 45, 57] 中採用的空間相容性（SC）也被用作採樣指導，其結果被報告以供比較。如表 6 的第 1、3 和 7、9 行所示，使用 SC² 度量作為指導獲得的配準召回率在與 FPFH 結合時比 RANSAC 高 14.79%，在與 FCGF 結合時高 1.66%。同時，由於 SC² 度量可以縮小採樣空間，SC² 度量的平均配準時間遠小於 RANSAC。此外，透過比較第 2、3 和 8、9 行，使用 SC² 度量作為指導可以獲得比使用 SC 度量更好的性能。這是因為 SC 受到模糊性問題的干擾，而 SC² 度量可以消除模糊性。

Two-stage Selection. We further adopt a two-stage se- lection strategy for generating consensus set for each seed. When one seed is an inlier correspondence, it has almost removed most of the outliers in the consensus set formed in the first stage. Since SC2 becomes more stable when the inlier rate increases, we construct a local SC2 matrix to re- move potential outliers. Comparing Row 3, 4 and 9, 10 in Tab. 6, using two-stage selection achieves a recall improve- ment of 1.96% when combined with FPFH, and 0.12% im- provement when combined with FCGF.

兩階段選擇。我們進一步採用兩階段選擇策略為每個種子生成共識集。當一個種子是內點對應時，它幾乎移除了第一階段形成的共識集中的大部分外點。由於當內點率增加時 SC² 變得更穩定，我們建構一個局部 SC² 矩陣來移除潛在的外點。比較表 6 中的第 3、4 行和第 9、10 行，使用兩階段選擇在與 FPFH 結合時實現了 1.96% 的召回率提升，在與 FCGF 結合時實現了 0.12% 的提升。

Local Spectral Matching. When a minimum set is sam- pled, RANSAC adopt the instance-equal SVD to generate an estimation of translation and rotation, which is sensi- tive to errors. We replace the instance-equal SVD [3] with the weighted SVD [19, 43], so that less reliable correspon- dences are assigned lower weights for robust registration. We construct a soft SC2 matrix in each consensus set, and then use local spectral matching to compute the association between each correspondence with the main cluster. The as- sociation value is utilized as the weight for weighted SVD. Comparing Row 4, 5 and 10, 11 in Tab. 6, using local spec- tral matching can boost the performance, especially for the mean rotation and translation error.

局部譜匹配。當採樣一個最小集時，RANSAC 採用實例相等 SVD 來生成平移和旋轉的估計，這對誤差很敏感。我們用加權 SVD [19, 43] 取代實例相等 SVD [3]，以便為不太可靠的對應分配較低的權重以實現穩健的配準。我們在每個共識集中建構一個軟 SC² 矩陣，然後使用局部譜匹配來計算每個對應與主聚類的關聯。該關聯值被用作加權 SVD 的權重。比較表 6 中的第 4、5 和 10、11 行，使用局部譜匹配可以提升性能，特別是對於平均旋轉和平移誤差。

Seed Selection. So far, each correspondence is treated as a seed. However, it do not need to generate a consensus set for all correspondences and estimate a rigid transformation. We only need to select a few reliable points, and use the aggregation among the inliers to collect the set without out- liers, so as to further improve the efficiency of registration. We use the global spectral matching combined with Non- Maximum Suppression to find several correspondences as seeds instead of all set. Row 5, 6 and 11, 12 of Tab. 6 shows that Seed Selection can reduce registration time by more than half without much performance degradation.

種子選擇。到目前為止，每個對應都被視為一個種子。然而，不需要為所有對應生成共識集並估計剛性變換。我們只需要選擇幾個可靠的點，並利用內點之間的聚合來收集無外點的集合，從而進一步提高配準的效率。我們使用全域譜匹配結合非極大值抑制來找到幾個對應作為種子，而不是所有集合。表 6 的第 5、6 和 11、12 行顯示，種子選擇可以將配準時間減少一半以上，而性能下降不多。

## 5. Conclusion

## 5. 結論

In this paper, we present a second order spatial compati- bility (SC2) measure based point cloud registration method, called SC2-PCR. The core component of our method is to cluster inliers by the proposed SC2 measure at early stage while eliminating ambiguity. Specifically, some reliable correspondences are selected by a global spectral decompo- sition with Non-Maximum Suppression firstly, called seed points. Then a two-stage sampling strategy is adopted to extend the seed points into some consensus sets. After that, each consensus set produces a rigid transformation by local spectral matching. Finally, the best estimation is selected as the final result. Extensive experiments demonstrate that our method achieves the state-of-the-art performance and high efficiency. Meanwhile, we also demonstrate the pro- posed SC2 is a flexible measure, which can be combined with learning networks to further boost their performance.

在本文中，我們提出了一種基於二階空間相容性（SC²）度量的點雲配準方法，稱為 SC²-PCR。我們方法的核心組成部分是在早期階段透過我們提出的 SC² 度量來聚類內點，同時消除模糊性。具體來說，首先透過帶有非極大值抑制的全域譜分解選擇一些可靠的對應，稱為種子點。然後採用兩階段採樣策略將種子點擴展到一些共識集中。之後，每個共識集透過局部譜匹配產生一個剛性變換。最後，選擇最佳估計作為最終結果。大量實驗證明，我們的方法達到了最先進的性能和高效率。同時，我們也證明了所提出的 SC² 是一種靈活的度量，可以與學習網路相結合以進一步提升其性能。

## 6. Acknowledgements

## 6. 致謝

This work was supported by the National Natural Science Foundation of China under Grants 62176096, 62176242 and 61991412.

本研究由國家自然科學基金（批准號：62176096、62176242、61991412）資助。

## References

## 參考文獻
