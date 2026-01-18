---
title: NeurIPS-2018-link-prediction-based-on-graph-neural-networks-Paper
field: Link_Prediction
status: Imported
created_date: 2026-01-13
pdf_link: "[[NeurIPS-2018-link-prediction-based-on-graph-neural-networks-Paper.pdf]]"
tags:
  - paper
  - Link_prediction
---
# Link Prediction Based on Graph Neural Networks

Muhan Zhang
Department of CSE
Washington University in St. Louis
muhan@wustl.edu

Yixin Chen
Department of CSE
Washington University in St. Louis
chen@cse.wustl.edu

[Traditional Chinese Translation]
# 基於圖神經網絡的鏈路預測

張木含 (Muhan Zhang)
華盛頓大學聖路易斯分校 (Washington University in St. Louis)
計算機科學與工程系 (Department of CSE)
muhan@wustl.edu

陳藝心 (Yixin Chen)
華盛頓大學聖路易斯分校 (Washington University in St. Louis)
計算機科學與工程系 (Department of CSE)
chen@cse.wustl.edu

[Original English Text]
## Abstract

Link prediction is a key problem for network-structured data. Link prediction heuristics use some score functions, such as common neighbors and Katz index, to measure the likelihood of links. They have obtained wide practical uses due to their simplicity, interpretability, and for some of them, scalability. However, every heuristic has a strong assumption on when two nodes are likely to link, which limits their effectiveness on networks where these assumptions fail. In this regard, a more reasonable way should be learning a suitable heuristic from a given network instead of using predefined ones. By extracting a local subgraph around each target link, we aim to learn a function mapping the subgraph patterns to link existence, thus automatically learning a “heuristic” that suits the current network. In this paper, we study this heuristic learning paradigm for link prediction. First, we develop a novel γ-decaying heuristic theory. The theory unifies a wide range of heuristics in a single framework, and proves that all these heuristics can be well approximated from local subgraphs. Our results show that local subgraphs reserve rich information related to link existence. Second, based on the γ-decaying theory, we propose a new method to learn heuristics from local subgraphs using a graph neural network (GNN). Its experimental results show unprecedented performance, working consistently well on a wide range of problems.

[Traditional Chinese Translation]
## 摘要

鏈路預測是網絡結構化數據中的一個關鍵問題。鏈路預測的啟發式方法使用一些評分函數，例如共同鄰居（common neighbors）和Katz指數，來衡量鏈路存在的可能性。由於其簡單、可解釋以及部分方法的可擴展性，它們已獲得廣泛的實際應用。然而，每種啟發式方法都對兩個節點何時可能相連有很強的假設，這限制了其在這些假設不成立的網絡上的有效性。有鑑於此，一個更合理的方法應該是從給定網絡中學習一個合適的啟發式方法，而不是使用預定義的方法。通過在每個目標鏈路周圍提取一個局部子圖，我們旨在學習一個能將子圖模式映射到鏈路存在的函數，從而自動學習一個適合當前網絡的「啟發式方法」。在本文中，我們研究了這種用於鏈路預測的啟發式學習範式。首先，我們發展了一種新穎的γ衰減啟發式理論。該理論在單一框架下統一了廣泛的啟發式方法，並證明所有這些啟發式方法都可以從局部子圖中得到很好的近似。我們的結果表明，局部子圖保留了與鏈路存在相關的豐富資訊。其次，基於γ衰減理論，我們提出一種使用圖神經網絡（GNN）從局部子圖中學習啟發式的新方法。其實驗結果展現了前所未有的性能，在廣泛的問題上始終表現優異。

[Original English Text]
## 1 Introduction

Link prediction is to predict whether two nodes in a network are likely to have a link [1]. Given the ubiquitous existence of networks, it has many applications such as friend recommendation [2], movie recommendation [3], knowledge graph completion [4], and metabolic network reconstruction [5].

[Traditional Chinese Translation]
## 1 緒論

鏈路預測旨在預測網絡中的兩個節點是否可能存在鏈路[1]。鑑於網絡無處不在，此技術有許多應用，例如朋友推薦[2]、電影推薦[3]、知識圖譜補全[4]以及代謝網絡重建[5]。

[Original English Text]
One class of simple yet effective approaches for link prediction is called heuristic methods. Heuristic methods compute some heuristic node similarity scores as the likelihood of links [1, 6]. Existing heuristics can be categorized based on the maximum hop of neighbors needed to calculate the score. For example, common neighbors (CN) and preferential attachment (PA) [7] are first-order heuristics, since they only involve the one-hop neighbors of two target nodes. Adamic-Adar (AA) and resource allocation (RA) [8] are second-order heuristics, as they are calculated from up to two-hop neighborhood of the target nodes. We define h-order heuristics to be those heuristics which require knowing up to h-hop neighborhood of the target nodes. There are also some high-order heuristics which require knowing the entire network. Examples include Katz, rooted PageRank (PR) [9], and SimRank (SR) [10]. Table 3 in Appendix A summarizes eight popular heuristics.

[Traditional Chinese Translation]
在鏈路預測中，一類簡單而有效的方法稱為啟發式方法。啟發式方法計算某些啟發式節點相似性分數作為鏈路可能性的度量[1, 6]。現有的啟發式方法可以根據計算分數所需的鄰居最大跳數進行分類。例如，共同鄰居（CN）和優先連接（PA）[7]是一階啟發式方法，因為它們只涉及目標節點的一跳鄰居。Adamic-Adar（AA）和資源分配（RA）[8]是二階啟發式方法，因為它們是根據目標節點的最多兩跳鄰域計算的。我們將h階啟發式方法定義為需要知道目標節點最多h跳鄰域的啟發式方法。還有一些需要了解整個網絡的高階啟發式方法。例子包括Katz、根（rooted）PageRank（PR）[9]和SimRank（SR）[10]。附錄A中的表3總結了八種流行的啟發式方法。

[Original English Text]
Although working well in practice, heuristic methods have strong assumptions on when links may exist. For example, the common neighbor heuristic assumes that two nodes are more likely to connect if they have many common neighbors. This assumption may be correct in social networks, but is shown to fail in protein-protein interaction (PPI) networks – two proteins sharing many common neighbors are actually less likely to interact [11].

[Traditional Chinese Translation]
雖然在實踐中效果不錯，但啟發式方法對鏈路何時可能存在有很強的假設。例如，共同鄰居啟發式假設，如果兩個節點有許多共同鄰居，它們就更可能連接。這個假設在社交網絡中可能是正確的，但在蛋白質-蛋白質交互（PPI）網絡中卻被證明是失敗的——共享許多共同鄰居的兩種蛋白質實際上更不可能相互作用[11]。

[Original English Text]
32nd Conference on Neural Information Processing Systems (NeurIPS 2018), Montréal, Canada.

[Traditional Chinese Translation]
第32屆神經信息處理系統會議（NeurIPS 2018），蒙特婁，加拿大。

[Original English Text]
Figure 1: The SEAL framework. For each target link, SEAL extracts a local enclosing subgraph around it, and uses a GNN to learn general graph structure features for link prediction. Note that the heuristics listed inside the box are just for illustration – the learned features may be completely different from existing heuristics.

[Traditional Chinese Translation]
圖1：SEAL框架。對於每個目標鏈路，SEAL提取其周圍的局部封閉子圖，並使用GNN學習通用的圖結構特徵以進行鏈路預測。請注意，方塊中列出的啟發式方法僅為說明之用——學習到的特徵可能與現有啟發式方法完全不同。

[Original English Text]
In fact, the heuristics belong to a more generic class, namely graph structure features. Graph structure features are those features located inside the observed node and edge structures of the network, which can be calculated directly from the graph. Since heuristics can be viewed as predefined graph structure features, a natural idea is to automatically learn such features from the network. Zhang and Chen [12] first studied this problem. They extract local enclosing subgraphs around links as the training data, and use a fully-connected neural network to learn which enclosing subgraphs correspond to link existence. Their method called Weisfeiler-Lehman Neural Machine (WLNM) has achieved state-of-the-art link prediction performance. The enclosing subgraph for a node pair (x, y) is the subgraph induced from the network by the union of x and y’s neighbors up to h hops. Figure 1 illustrates the 1-hop enclosing subgraphs for (A, B) and (C, D). These enclosing subgraphs are very informative for link prediction – all first-order heuristics such as common neighbors can be directly calculated from the 1-hop enclosing subgraphs.

[Traditional Chinese Translation]
事實上，啟發式方法屬於一個更通用的類別，即圖結構特徵。圖結構特徵是位於網絡觀察到的節點和邊結構內部的特徵，可以直接從圖中計算出來。由於啟發式方法可被視為預定義的圖結構特徵，一個自然的想法是自動從網絡中學習這些特徵。Zhang和Chen[12]首次研究了這個問題。他們提取鏈路周圍的局部封閉子圖作為訓練數據，並使用全連接神經網絡來學習哪些封閉子圖對應於鏈路的存在。他們的方法稱為Weisfeiler-Lehman神經機器（WLNM），已達到最先進的鏈路預測性能。節點對(x, y)的封閉子圖是由x和y的最多h跳鄰居的並集從網絡中誘導出的子圖。圖1展示了(A, B)和(C, D)的一跳封閉子圖。這些封閉子圖對於鏈路預測非常有用——所有一階啟發式方法（如共同鄰居）都可以直接從一跳封閉子圖中計算出來。

[Original English Text]
However, it is shown that high-order heuristics such as rooted PageRank and Katz often have much better performance than first and second-order ones [6]. To effectively learn good high-order features, it seems that we need a very large hop number h so that the enclosing subgraph becomes the entire network. This results in unaffordable time and memory consumption for most practical networks. But do we really need such a large h to learn high-order heuristics?

[Traditional Chinese Translation]
然而，研究表明，如根（rooted）PageRank和Katz等高階啟發式方法通常比一階和二階方法有更好的性能[6]。為了有效地學習好的高階特徵，似乎我們需要一個非常大的跳數h，以至於封閉子圖成為整個網絡。這對於大多數實際網絡來說，會導致無法承受的時間和內存消耗。但是，我們真的需要這麼大的h來學習高階啟發式方法嗎？

[Original English Text]
Fortunately, as our first contribution, we show that we do not necessarily need a very large h to learn high-order graph structure features. We dive into the inherent mechanisms of link prediction heuristics, and find that most high-order heuristics can be unified by a γ-decaying theory. We prove that, under mild conditions, any γ-decaying heuristic can be effectively approximated from an h-hop enclosing subgraph, where the approximation error decreases at least exponentially with h. This means that we can safely use even a small h to learn good high-order features. It also implies that the “effective order” of these high-order heuristics is not that high.

[Traditional Chinese Translation]
幸運的是，作為我們的第一個貢獻，我們證明了我們不一定需要一個很大的h來學習高階圖結構特徵。我們深入研究了鏈路預測啟發式方法的內在機制，發現大多數高階啟發式方法可以被一個γ衰減理論所統一。我們證明，在溫和的條件下，任何γ衰減啟發式方法都可以從一個h跳封閉子圖中有效地近似，其中近似誤差至少以指數方式隨h減小。這意味著我們可以安全地使用一個小的h來學習好的高階特徵。這也意味著這些高階啟發式方法的「有效階數」並不是那麼高。

[Original English Text]
Based on our theoretical results, we propose a novel link prediction framework, SEAL, to learn general graph structure features from local enclosing subgraphs. SEAL fixes multiple drawbacks of WLNM. First, a graph neural network (GNN) [13, 14, 15, 16, 17] is used to replace the fully-connected neural network in WLNM, which enables better graph feature learning ability. Second, SEAL permits learning from not only subgraph structures, but also latent and explicit node features, thus absorbing multiple types of information. We empirically verified its much improved performance.

[Traditional Chinese Translation]
基於我們的理論結果，我們提出了一個新穎的鏈路預測框架SEAL，用於從局部封閉子圖中學習通用的圖結構特徵。SEAL修正了WLNM的多個缺點。首先，使用圖神經網絡（GNN）[13, 14, 15, 16, 17]取代WLNM中的全連接神經網絡，從而實現了更好的圖特徵學習能力。其次，SEAL不僅允許從子圖結構中學習，還允許從潛在和顯式的節點特徵中學習，從而吸收多種類型的信息。我們通過經驗驗證了其大幅改進的性能。

[Original English Text]
Our contributions are summarized as follows. 1) We present a new theory for learning link prediction heuristics, justifying learning from local subgraphs instead of entire networks. 2) We propose SEAL, a novel link prediction framework based on GNN (illustrated in Figure 1). SEAL outperforms all heuristic methods, latent feature methods, and recent network embedding methods by large margins. SEAL also outperforms the previous state-of-the-art method, WLNM.

[Traditional Chinese Translation]
我們的貢獻總結如下。1）我們提出了一個用於學習鏈路預測啟發式方法的新理論，證明了從局部子圖學習而非整個網絡學習的合理性。2）我們提出了SEAL，一個基於GNN的新型鏈路預測框架（如圖1所示）。SEAL在性能上遠超所有啟發式方法、潛在特徵方法以及最近的網絡嵌入方法。SEAL也優於先前的最先進方法WLNM。

[Original English Text]
## 2 Preliminaries

Notations Let G = (V, E) be an undirected graph, where V is the set of vertices and E ⊂ V × V is the set of observed links. Its adjacency matrix is A, where Ai,j = 1 if (i, j) ∈ E and Ai,j = 0 otherwise. For any nodes x, y ∈ V, let Γ(x) be the 1-hop neighbors of x, and d(x, y) be the shortest path distance between x and y. A walk w = (v0, . . . , vk) is a sequence of nodes with (vi, vi+1) ∈ E. We use |⟨v0, . . . , vk⟩| to denote the length of the walk w, which is k here.

[Traditional Chinese Translation]
## 2 預備知識

符號說明 令 G = (V, E) 為一個無向圖，其中V是頂點集合，E ⊂ V × V 是觀察到的鏈路集合。其鄰接矩陣為A，其中若 (i, j) ∈ E 則 Ai,j = 1，否則 Ai,j = 0。對於任何節點 x, y ∈ V，令 Γ(x) 為x的一跳鄰居集合，d(x, y) 為x和y之間的最短路徑距離。一個遊走 w = (v0, . . . , vk) 是一個節點序列，其中 (vi, vi+1) ∈ E。我們使用 |⟨v0, . . . , vk⟩| 來表示遊走w的長度，這裡即為k。

[Original English Text]
Latent features and explicit features Besides graph structure features, latent features and explicit features are also studied for link prediction. Latent feature methods [3, 18, 19, 20] factorize some matrix representations of the network to learn a low-dimensional latent representation/embedding for each node. Examples include matrix factorization [3] and stochastic block model [18] etc. Recently, a number of network embedding techniques have been proposed, such as DeepWalk [19], LINE [21] and node2vec [20], which are also latent feature methods since they implicitly factorize some matrices too [22]. Explicit features are often available in the form of node attributes, describing all kinds of side information about individual nodes. It is shown that combining graph structure features with latent features and explicit features can improve the performance [23, 24].

[Traditional Chinese Translation]
潛在特徵與顯式特徵 除了圖結構特徵外，潛在特徵和顯式特徵也用於鏈路預測的研究。潛在特徵方法 [3, 18, 19, 20] 通過分解網絡的某些矩陣表示，為每個節點學習一個低維的潛在表示/嵌入。例子包括矩陣分解 [3] 和隨機區塊模型 [18] 等。最近，提出了許多網絡嵌入技術，如 DeepWalk [19]、LINE [21] 和 node2vec [20]，這些也是潛在特徵方法，因為它們也隱含地分解了某些矩陣 [22]。顯式特徵通常以節點屬性的形式提供，描述了關於個別節點的各種附加信息。研究表明，將圖結構特徵與潛在特徵和顯式特徵相結合可以提高性能 [23, 24]。

[Original English Text]
Graph neural networks Graph neural network (GNN) is a new type of neural network for learning over graphs [13, 14, 15, 16, 25, 26]). Here, we only briefly introduce the components of a GNN since this paper is not about GNN innovations but is a novel application of GNN. A GNN usually consists of 1) graph convolution layers which extract local substructure features for individual nodes, and 2) a graph aggregation layer which aggregates node-level features into a graph-level feature vector. Many graph convolution layers can be unified into a message passing framework [27].

[Traditional Chinese Translation]
圖神經網絡 圖神經網絡（GNN）是一種用於在圖上學習的新型神經網絡 [13, 14, 15, 16, 25, 26]。在此，我們僅簡要介紹GNN的組成部分，因為本文並非關於GNN的創新，而是GNN的一個新應用。GNN通常由以下部分組成：1）提取個別節點局部子結構特徵的圖卷積層，以及2）將節點級特徵聚合為圖級特徵向量的圖聚合層。許多圖卷積層可以統一到一個消息傳遞框架中 [27]。

[Original English Text]
Supervised heuristic learning There are some previous attempts to learn supervised heuristics for link prediction. The closest work to ours is the Weisfeiler-Lehman Neural Machine (WLNM) [12], which also learns from local subgraphs. However, WLNM has several drawbacks. Firstly, WLNM trains a fully-connected neural network on the subgraphs’ adjacency matrices. Since fully-connected neural networks only accept fixed-size tensors as input, WLNM requires truncating different subgraphs to the same size, which may lose much structural information. Secondly, due to the limitation of adjacency matrix representations, WLNM cannot learn from latent or explicit features. Thirdly, theoretical justifications are also missing. We include more discussion on WLNM in Appendix D. Another related line of research is to train a supervised learning model on different heuristics’ combination. For example, the path ranking algorithm [28] trains logistic regression on different path types’ probabilities to predict relations in knowledge graphs. Nickel et al. [23] propose to incorporate heuristic features into tensor factorization models. However, these models still rely on predefined heuristics – they cannot learn general graph structure features.

[Traditional Chinese Translation]
監督式啟發式學習 先前已有一些嘗試來學習監督式啟發式以進行鏈路預測。與我們最相關的工作是Weisfeiler-Lehman神經機器（WLNM）[12]，該方法也從局部子圖中學習。然而，WLNM有幾個缺點。首先，WLNM在子圖的鄰接矩陣上訓練一個全連接神經網絡。由於全連接神經網絡只接受固定大小的張量作為輸入，WLNM需要將不同大小的子圖截斷成相同大小，這可能會丟失大量結構信息。其次，由於鄰接矩陣表示的限制，WLNM無法從潛在或顯式特徵中學習。第三，也缺乏理論上的證明。我們在附錄D中對WLNM有更多討論。另一個相關的研究方向是訓練一個監督學習模型來組合不同的啟發式方法。例如，路徑排序算法[28]在不同路徑類型的概率上訓練邏輯回歸，以預測知識圖中的關係。Nickel等人[23]提出將啟發式特徵納入張量分解模型。然而，這些模型仍然依賴於預定義的啟發式方法——它們無法學習一般的圖結構特徵。

[Original English Text]
## 3 A theory for unifying link prediction heuristics

In this section, we aim to understand deeper the mechanisms behind various link prediction heuristics, and thus motivating the idea of learning heuristics from local subgraphs. Due to the large number of graph learning techniques, note that we are not concerned with the generalization error of a particular method, but focus on the information reserved in the subgraphs for calculating existing heuristics.
Definition 1. (Enclosing subgraph) For a graph G = (V,E), given two nodes x,y ∈ V, the h-hop enclosing subgraph for (x,y) is the subgraph G^h_{x,y} induced from G by the set of nodes {i | d(i,x) ≤ h or d(i, y) ≤ h }.

[Traditional Chinese Translation]
## 3 一個統一鏈路預測啟發式方法的理論

在本節中，我們旨在更深入地理解各種鏈路預測啟發式方法背後的機制，從而推動從局部子圖學習啟發式方法的思想。由於圖學習技術數量眾多，請注意我們不關心特定方法的泛化誤差，而是關注為計算現有啟發式方法而在子圖中保留的信息。
定義 1. (封閉子圖) 對於一個圖 G = (V,E)，給定兩個節點 x,y ∈ V，(x,y)的h跳封閉子圖是由節點集合 {i | d(i,x) ≤ h 或 d(i, y) ≤ h } 從G中導出的子圖 G^h_{x,y}。

[Original English Text]
The enclosing subgraph describes the “h-hop surrounding environment" of (x,y). Since G^h_{x,y} contains all h-hop neighbors of x and y, we naturally have the following theorem.
Theorem 1. Any h-order heuristic for (x, y) can be accurately calculated from G^h_{x,y}.

[Traditional Chinese Translation]
封閉子圖描述了(x,y)的「h跳周圍環境」。由於 G^h_{x,y} 包含x和y的所有h跳鄰居，我們自然得出以下定理。
定理 1. (x,y)的任何h階啟發式方法都可以從 G^h_{x,y} 中準確計算出來。

[Original English Text]
For example, a 2-hop enclosing subgraph will contain all the information needed to calculate any first and second-order heuristics. However, although first and second-order heuristics are well covered by local enclosing subgraphs, an extremely large h seems to be still needed for learning high-order heuristics. Surprisingly, our following analysis shows that learning high-order heuristics is also feasible with a small h. We support this first by defining the γ-decaying heuristic. We will show that under certain conditions, a γ-decaying heuristic can be very well approximated from the h-hop enclosing subgraph. Moreover, we will show that almost all well-known high-order heuristics can be unified into this γ-decaying heuristic framework.
Definition 2. (γ-decaying heuristic) A γ-decaying heuristic for (x, y) has the following form:
H(x, y) = η Σ(from l=1 to ∞) γ^l * f(x, y, l), (1)

[Traditional Chinese Translation]
例如，一個2跳封閉子圖將包含計算任何一階和二階啟發式方法所需的所有信息。然而，儘管局部封閉子圖很好地涵蓋了一階和二階啟發式方法，但學習高階啟發式方法似乎仍然需要一個極大的h。令人驚訝的是，我們接下來的分析表明，用一個小的h學習高階啟發式方法也是可行的。我們首先通過定義γ衰減啟發式方法來支持這一點。我們將證明，在某些條件下，一個γ衰減啟發式方法可以從h跳封閉子圖中得到很好的近似。此外，我們將證明幾乎所有著名的高階啟發式方法都可以統一到這個γ衰減啟發式框架中。
定義 2. (γ衰減啟發式) (x,y)的γ衰減啟發式具有以下形式：
H(x, y) = η Σ(從 l=1 到 ∞) γ^l * f(x, y, l), (1)

[Original English Text]
where γ is a decaying factor between 0 and 1, η is a positive constant or a positive function of γ that is upper bounded by a constant, f is a nonnegative function of x, y, l under the the given network.
Next, we will show that under certain conditions, a γ-decaying heuristic can be approximated from an h-hop enclosing subgraph, and the approximation error decreases at least exponentially with h.
Theorem 2. Given a γ-decaying heuristic H(x, y) = ηΣ(from i=1 to ∞) γ^l * f(x, y, l), if f (x, y, l) satisfies:
*   (property 1) f(x, y, l) ≤ λ^l where γλ < 1; and
*   (property 2) f(x,y,l) is calculable from G^g(h)_{x,y} for l = 1,2,...,g(h), where g(h) = ah+b with a, b ∈ N and a > 0,
then H(x, y) can be approximated from G^h_{x,y} and the approximation error decreases at least exponentially with h.

[Traditional Chinese Translation]
其中γ是0到1之間的衰減因子，η是一個正常數或γ的正函數，且受一個常數上限約束，f是在給定網絡下x, y, l的非負函數。
接下來，我們將證明在某些條件下，γ衰減啟發式可以從h跳封閉子圖中近似，且近似誤差至少以指數級隨h減小。
定理 2. 給定一個γ衰減啟發式 H(x, y) = ηΣ(從 l=1 到 ∞) γ^l * f(x, y, l)，如果 f(x, y, l) 滿足：
*   (性質 1) f(x, y, l) ≤ λ^l 其中 γλ < 1；且
*   (性質 2) f(x, y, l) 可從 G^g(h)_{x,y} 計算，對於 l = 1, 2, ..., g(h)，其中 g(h) = ah+b 且 a, b ∈ N, a > 0，
則 H(x, y) 可以從 G^h_{x,y} 近似，且近似誤差至少以指數級隨h減小。

[Original English Text]
Proof. We can approximate such a γ-decaying heuristic by summing over its first g(h) terms.
H˜(x, y) := η Σ(from l=1 to g(h)) γ^l * f(x, y, l). (2)
The approximation error can be bounded as follows.
|H(x,y) – H˜(x,y)| = η |Σ(from l=g(h)+1 to ∞) γ^l * f(x,y,l)| ≤ η Σ(from l=g(h)+1 to ∞) γ^l * λ^l = η(γλ)^(ah+b+1) * (1 – γλ)^(-1).

[Traditional "Chinese" Translation]
證明。我們可以通過對其前g(h)項求和來近似這樣的γ衰減啟發式。
H˜(x, y) := η Σ(從 l=1 到 g(h)) γ^l * f(x, y, l)。 (2)
近似誤差可以被界定如下：
|H(x,y) – H˜(x,y)| = η |Σ(從 l=g(h)+1 到 ∞) γ^l * f(x,y,l)| ≤ η Σ(從 l=g(h)+1 到 ∞) γ^l * λ^l = η(γλ)^(ah+b+1) * (1 – γλ)^(-1)。

[Original English Text]
In practice, a small γλ and a large a lead to a faster decreasing speed. Next we will prove that three popular high-order heuristics: Katz, rooted PageRank and SimRank, are all γ-decaying heuristics which satisfy the properties in Theorem 2. First, we need the following lemma.
Lemma 1. Any walk between x and y with length l < 2h + 2 is included in G^h_{x,y}.
Proof. Given any walk w = (x, v1, . . . , vl−1, y) with length l, we will show that every node vi is included in G^h_{x,y}. Consider any vi. Assume d(vi, x) ≥ h + 1 and d(vi, y) ≥ h + 1. Then, 2h + 2 > l = |⟨x, v1, . . . , vi⟩| + |⟨vi, . . . , vl−1, y⟩| ≥ d(vi, x) + d(vi, y) ≥ 2h + 2, a contradiction. Thus, d(vi, x) ≤ h or d(vi, y) ≤ h. By the definition of G^h_{x,y}, vi must be included in G^h_{x,y}.
Next we will analyze Katz, rooted PageRank and SimRank one by one.

[Traditional Chinese Translation]
在實踐中，較小的γλ和較大的a會導致更快的下降速度。接下來，我們將證明三種流行的高階啟發式方法：Katz、根PageRank和SimRank，都是滿足定理2中性質的γ衰減啟發式。首先，我們需要以下引理。
引理 1. 任何長度 l < 2h + 2 的x和y之間的遊走都包含在 G^h_{x,y} 中。
證明。給定任何長度為l的遊走 w = (x, v1, . . . , vl−1, y)，我們將證明每個節點vi都包含在 G^h_{x,y} 中。考慮任何vi。假設 d(vi, x) ≥ h + 1 且 d(vi, y) ≥ h + 1。那麼，2h + 2 > l = |⟨x, v1, . . . , vi⟩| + |⟨vi, . . . , vl−1, y⟩| ≥ d(vi, x) + d(vi, y) ≥ 2h + 2，這是一個矛盾。因此，d(vi, x) ≤ h 或 d(vi, y) ≤ h。根據 G^h_{x,y} 的定義，vi必須包含在 G^h_{x,y} 中。
接下來我們將逐一分析Katz、根PageRank和SimRank。

[Original English Text]
### 3.1 Katz index

The Katz index [29] for (x, y) is defined as
Katzz,y = Σ(from l=1 to ∞) β^l * |walks^(l)(x, y)| = Σ(from l=1 to ∞) β^l * [A^l]x,y, (3)
where walks^(l)(x, y) is the set of length-l walks between x and y, and A' is the l-th power of the adjacency matrix of the network. Katz index sums over the collection of all walks between x and y where a walk of length l is damped by β^l (0 < β < 1), giving more weight to shorter walks.
Katz index is directly defined in the form of a γ-decaying heuristic with η = 1, γ = β, and f(x,y,l) = |walks^(l)(x, y)|. According to Lemma 1, |walks^(l)(x, y)| is calculable from G^h_{x,y} for l ≤ 2h + 1, thus property 2 in Theorem 2 is satisfied. Now we show when property 1 is satisfied.
Proposition 1. For any nodes i, j, [A^l]i,j is bounded by d^l, where d is the maximum node degree of the network.
Proof. We prove it by induction. When l = 1, Ai,j ≤ d for any (i, j). Thus the base case is correct. Now, assume by induction that [A^l]i,j ≤ d^l for any (i, j), we have
[A^(l+1)]i,j = Σ(from k=1 to |V|) [A^l]i,k * Ak,j ≤ d^l * Σ(from k=1 to |V|) Ak,j ≤ d^l * d = d^(l+1).
Taking λ = d, we can see that whenever βd < 1, the Katz index will satisfy property 1 in Theorem 2. In practice, the damping factor β is often set to very small values like 5E-4 [1], which implies that Katz can be very well approximated from the h-hop enclosing subgraph.

[Traditional Chinese Translation]
### 3.1 Katz 指數

(x, y) 的 Katz 指數 [29] 定義為
Katzz,y = Σ(從 l=1 到 ∞) β^l * |walks^(l)(x, y)| = Σ(從 l=1 到 ∞) β^l * [A^l]x,y, (3)
其中 walks^(l)(x, y) 是 x 和 y 之間長度為 l 的遊走集合，A^l 是網絡鄰接矩陣的 l 次方。Katz 指數對 x 和 y 之間所有遊走的集合進行求和，其中長度為 l 的遊走被 β^l (0 < β < 1) 衰減，賦予較短的遊走更多權重。
Katz 指數直接以 γ 衰減啟發式的形式定義，其中 η = 1，γ = β，且 f(x,y,l) = |walks^(l)(x, y)|。根據引理 1，|walks^(l)(x, y)| 可從 G^h_{x,y} 計算，對於 l ≤ 2h + 1，因此滿足定理 2 中的性質 2。現在我們展示性質 1 何時滿足。
命題 1. 對於任何節點 i, j，[A^l]i,j 受 d^l 的限制，其中 d 是網絡的最大節點度數。
證明。我們用歸納法證明。當 l = 1 時，對於任何 (i, j)，Ai,j ≤ d。因此基本情況是正確的。現在，通過歸納假設 [A^l]i,j ≤ d^l 對於任何 (i, j) 成立，我們有
[A^(l+1)]i,j = Σ(從 k=1 到 |V|) [A^l]i,k * Ak,j ≤ d^l * Σ(從 k=1 到 |V|) Ak,j ≤ d^l * d = d^(l+1)。
取 λ = d，我們可以看到只要 βd < 1，Katz 指數將滿足定理 2 中的性質 1。在實踐中，衰減因子 β 通常設置為非常小的值，如 5E-4 [1]，這意味著 Katz 指數可以從 h 跳封閉子圖中得到很好的近似。

[Original English Text]
### 3.2 PageRank

The rooted PageRank for node x calculates the stationary distribution of a random walker starting at x, who iteratively moves to a random neighbor of its current position with probability α or returns to x with probability 1 − α. Let πx denote the stationary distribution vector. Let [πx]i denote the probability that the random walker is at node i under the stationary distribution.
Let P be the transition matrix with Pi,j = 1/|Γ(vj)| if (i, j) ∈ E and Pi,j = 0 otherwise. Let ex be a vector with the xth element being 1 and others being 0. The stationary distribution satisfies
πx = αP^T * πx + (1 − α)ex. (4)
When used for link prediction, the score for (x, y) is given by [πx]y (or [πx]y + [πy]x for symmetry). To show that rooted PageRank is a γ-decaying heuristic, we introduce the inverse P-distance theory [30], which states that [πx]y can be equivalently written as follows:
[πx]y = (1 - α) Σ(over w:x→y) P[w]α^len(w), (5)
where the summation is taken over all walks w starting at x and ending at y (possibly touching x and y multiple times). For a walk w = (v0, v1, . . . , vk), len(w) := |⟨v0, v1, . . . , vk⟩| is the length of the walk. The term P[w] is defined as Π(from i=0 to k-1) 1/|Γ(vi)|, which can be interpreted as the probability of traveling w. Now we have the following theorem.
Theorem 3. The rooted PageRank heuristic is a γ-decaying heuristic which satisfies the properties in Theorem 2.

[Traditional Chinese Translation]
### 3.2 PageRank

根 PageRank 計算從 x 開始的隨機遊走者的平穩分佈，該遊走者以概率 α 迭代移動到其當前位置的隨機鄰居，或以概率 1 - α 返回到 x。令 πx 表示平穩分佈向量。令 [πx]i 表示在平穩分佈下隨機遊走者在節點 i 的概率。
令 P 為轉移矩陣，若 (i, j) ∈ E，則 Pi,j = 1/|Γ(vj)|，否則 Pi,j = 0。令 ex 為第 x 個元素為 1 其餘為 0 的向量。平穩分佈滿足
πx = αP^T * πx + (1 − α)ex。(4)
當用於鏈路預測時，(x, y) 的分數由 [πx]y 給出（或為了對稱性而使用 [πx]y + [πy]x）。為了證明根 PageRank 是一種 γ 衰減啟發式，我們引入逆 P-距離理論 [30]，該理論指出 [πx]y 可以等價地寫為：
[πx]y = (1 - α) Σ(遍歷 w:x→y) P[w]α^len(w)，(5)
其中求和遍歷所有從 x 開始並在 y 結束的遊走 w（可能多次觸及 x 和 y）。對於一個遊走 w = (v0, v1, . . . , vk)，len(w) := |⟨v0, v1, . . . , vk⟩| 是遊走的長度。項 P[w] 定義為 Π(從 i=0 到 k-1) 1/|Γ(vi)|，可以解釋為遊走 w 的概率。現在我們有以下定理。
定理 3. 根 PageRank 啟發式是一種滿足定理 2 中性質的 γ 衰減啟發式。

[Original English Text]
Proof. We first write [πx]y in the following form.
[πx]y = (1 - α) Σ(from l=1 to ∞) α^l * Σ(over w:x→y, len(w)=l) P[w]. (6)
Defining f(x, y, l) := Σ(over w:x→y, len(w)=l) P[w] leads to the form of a γ-decaying heuristic. Note that f(x, y, l) is the probability that a random walker starting at x stops at y with exactly l steps, which satisfies Σ(z∈V) f(x, z, l) = 1. Thus, f(x, y, l) ≤ 1 < 1/α (property 1). According to Lemma 1, f(x, y, l) is also calculable from G^h_{x,y} for l ≤ 2h + 1 (property 2).

[Traditional Chinese Translation]
證明。我們首先將 [πx]y 寫成以下形式。
[πx]y = (1 - α) Σ(從 l=1 到 ∞) α^l * Σ(遍歷 w:x→y, len(w)=l) P[w]。(6)
定義 f(x, y, l) := Σ(遍歷 w:x→y, len(w)=l) P[w] 導出了 γ 衰減啟發式的形式。注意 f(x, y, l) 是從 x 開始的隨機遊走者在恰好 l 步後停在 y 的概率，滿足 Σ(z∈V) f(x, z, l) = 1。因此，f(x, y, l) ≤ 1 < 1/α (性質 1)。根據引理 1，f(x, y, l) 也可以從 G^h_{x,y} 計算，對於 l ≤ 2h + 1 (性質 2)。

[Original English Text]
### 3.3 SimRank

The SimRank score [10] is motivated by the intuition that two nodes are similar if their neighbors are also similar. It is defined in the following recursive way: if x = y, then s(x, y) := 1; otherwise,
s(x,y) := γ * (Σ(a∈Γ(x)) Σ(b∈Γ(y)) s(a, b)) / (|Γ(x)| * |Γ(y)|) (7)
where γ is a constant between 0 and 1. According to [10], SimRank has an equivalent definition:
s(x,y) = Σ(over w:(x,y)→(z,z)) γ^len(w), (8)
where w: (x, y) → (z, z) denotes all simultaneous walks such that one walk starts at x, the other walk starts at y, and they first meet at any vertex z. For a simultaneous walk w = ((v0, u0), . . . , (vk, uk)), len(w) = k is the length of the walk. The term P[w] is similarly defined as Π(from i=0 to k-1) 1/(|Γ(vi)|*|Γ(ui)|) describing the probability of this walk. Now we have the following theorem.
Theorem 4. SimRank is a γ-decaying heuristic which satisfies the properties in Theorem 2.

[Traditional Chinese Translation]
### 3.3 SimRank

SimRank分數[10]的動機源於這樣一個直覺：如果兩個節點的鄰居也相似，那麼這兩個節點就相似。它通過以下遞歸方式定義：如果x = y，則s(x, y) := 1；否則，
s(x,y) := γ * (Σ(a∈Γ(x)) Σ(b∈Γ(y)) s(a, b)) / (|Γ(x)| * |Γ(y)|) (7)
其中γ是0和1之間的一個常數。根據[10]，SimRank有一個等價的定義：
s(x,y) = Σ(遍歷w:(x,y)→(z,z)) γ^len(w)， (8)
其中w: (x, y) → (z, z)表示所有同時發生的遊走，其中一個遊走從x開始，另一個從y開始，並且它們在任何頂點z首次相遇。對於一個同時遊走w = ((v0, u0), . . . , (vk, uk))，len(w) = k是遊走的長度。術語P[w]同樣定義為Π(從i=0到k-1) 1/(|Γ(vi)|*|Γ(ui)|)，描述了這次遊走的概率。現在我們有以下定理。
定理 4. SimRank是一種滿足定理2性質的γ衰減啟發式。

[Original English Text]
Proof. We write s(x, y) as follows.
s(x,y) = Σ(from l=1 to ∞) γ^l * Σ(over w:(x,y)→(z,z), len(w)=l) P[w]. (9)
Defining f(x, y, l) := Σ(over w:(x,y)→(z,z), len(w)=l) P[w] reveals that SimRank is a γ-decaying heuristic. Note that f(x, y, l) ≤ 1 < 1/γ. It is easy to see that f(x, y, l) is also calculable from G^h_{x,y} for l ≤ h.

[Traditional Chinese Translation]
證明。我們將s(x, y)寫成如下形式：
s(x,y) = Σ(從l=1到∞) γ^l * Σ(遍歷w:(x,y)→(z,z), len(w)=l) P[w]。(9)
定義f(x, y, l) := Σ(遍歷w:(x,y)→(z,z), len(w)=l) P[w]揭示了SimRank是一種γ衰減啟發式。注意f(x, y, l) ≤ 1 < 1/γ。很容易看出，對於l ≤ h，f(x, y, l)也可以從G^h_{x,y}計算出來。

[Original English Text]
Discussion There exist several other high-order heuristics based on path counting or random walk [6] which can be as well incorporated into the γ-decaying heuristic framework. We omit the analysis here. Our results reveal that most high-order heuristics inherently share the same γ-decaying heuristic form, and thus can be effectively approximated from an h-hop enclosing subgraph with exponentially smaller approximation error. We believe the ubiquity of γ-decaying heuristics is not by accident – it implies that a successful link prediction heuristic is better to put exponentially smaller weight on structures far away from the target, as remote parts of the network intuitively make little contribution to link existence. Our results build the foundation for learning heuristics from local subgraphs, as they imply that local enclosing subgraphs already contain enough information to learn good graph structure features for link prediction which is much desired considering learning from the entire network is often infeasible. To summarize, from the small enclosing subgraphs extracted around links, we are able to accurately calculate first and second-order heuristics, and approximate a wide range of high-order heuristics with small errors. Therefore, given adequate feature learning ability of the model used, learning from such enclosing subgraphs is expected to achieve performance at least as good as a wide range of heuristics. There is some related work which empirically verifies that local methods can often estimate PageRank and SimRank well [31, 32]. Another related theoretical work [33] establishes a condition of h to achieve some fixed approximation error for ordinary PageRank.

[Traditional Chinese Translation]
討論。還存在其他幾種基於路徑計數或隨機遊走的高階啟發式方法[6]，它們也可以被納入γ衰減啟發式框架。我們在此省略其分析。我們的結果揭示，大多數高階啟發式方法內在地共享相同的γ衰減啟發式形式，因此可以從一個h跳封閉子圖中有效地近似，且近似誤差呈指數級減小。我們相信，γ衰減啟發式方法的普遍存在並非偶然——它意味著一個成功的鏈路預測啟發式方法最好對遠離目標的結構賦予指數級的小權重，因為直觀上，網絡的遠程部分對鏈路的存在貢獻很小。我們的結果為從局部子圖學習啟發式方法奠定了基礎，因為它們意味著局部封閉子圖已經包含足夠的信息來學習好的圖結構特徵以進行鏈路預測，考慮到從整個網絡學習通常是不可行的，這一點非常可取。總結來說，從圍繞鏈路提取的小型封閉子圖中，我們能夠準確地計算一階和二階啟發式方法，並以小誤差近似廣泛的高階啟發式方法。因此，只要使用的模型具有足夠的特徵學習能力，從這些封閉子圖中學習預計將達到至少與廣泛啟發式方法一樣好的性能。有一些相關工作通過經驗證明，局部方法通常可以很好地估計PageRank和SimRank [31, 32]。另一項相關的理論工作[33]建立了一個h的條件，以實現對普通PageRank的某個固定近似誤差。

[Original English Text]
## 4 SEAL: An implemetation of the theory using GNN

In this section, we describe our SEAL framework for link prediction. SEAL does not restrict the learned features to be in some particular forms such as γ-decaying heuristics, but instead learns general graph structure features for link prediction. It contains three steps: 1) enclosing subgraph extraction, 2) node information matrix construction, and 3) GNN learning. Given a network, we aim to learn automatically a “heuristic” that best explains the link formations. Motivated by the theoretical results, this function takes local enclosing subgraphs around links as input, and output how likely the links exist. To learn such a function, we train a graph neural network (GNN) over the enclosing subgraphs. Thus, the first step in SEAL is to extract enclosing subgraphs for a set of sampled positive links (observed) and a set of sampled negative links (unobserved) to construct the training data.
A GNN typically takes (A, X) as input, where A (with slight abuse of notation) is the adjacency matrix of the input enclosing subgraph, X is the node information matrix each row of which corresponds to a node’s feature vector. The second step in SEAL is to construct the node information matrix X for each enclosing subgraph. This step is crucial for training a successful GNN link prediction model. In the following, we discuss this key step. The node information matrix X in SEAL has three components: structural node labels, node embeddings and node attributes.

[Traditional Chinese Translation]
## 4 SEAL：使用GNN實現理論

在本節中，我們描述用於鏈路預測的SEAL框架。SEAL不限制學習到的特徵必須是γ衰減啟發式等特定形式，而是學習用於鏈路預測的一般圖結構特徵。它包含三個步驟：1）封閉子圖提取，2）節點信息矩陣構建，以及3）GNN學習。給定一個網絡，我們的目標是自動學習一個最能解釋鏈路形成的「啟發式方法」。受理論結果的啟發，此函數以鏈路周圍的局部封閉子圖為輸入，並輸出鏈路存在的可能性。為了學習這樣一個函數，我們在封閉子圖上訓練一個圖神經網絡（GNN）。因此，SEAL的第一步是為一組採樣的正鏈路（已觀察到的）和一組採樣的負鏈路（未觀察到的）提取封閉子圖，以構建訓練數據。
GNN通常以(A, X)作為輸入，其中A（符號稍有濫用）是輸入封閉子圖的鄰接矩陣，X是節點信息矩陣，其每一行對應一個節點的特徵向量。SEAL的第二步是為每個封閉子圖構建節點信息矩陣X。這一步對於訓練一個成功的GNN鏈路預測模型至關重要。接下來，我們討論這個關鍵步驟。SEAL中的節點信息矩陣X有三個組成部分：結構節點標籤、節點嵌入和節點屬性。

[Original English Text]
### 4.1 Node labeling

The first component in X is each node’s structural label. A node labeling is function fl : V → N which assigns an integer label fl(i) to every node i in the enclosing subgraph. The purpose is to use different labels to mark nodes’ different roles in an enclosing subgraph: 1) The center nodes x and y are the target nodes between which the link is located. 2) Nodes with different relative positions to the center have different structural importance to the link. A proper node labeling should mark such differences. If we do not mark such differences, GNNs will not be able to tell where are the target nodes between which a link existence should be predicted, and lose structural information.
Our node labeling method is derived from the following criteria: 1) The two target nodes x and y always have the distinctive label “1”. 2) Nodes i and j have the same label if d(i, x) = d(j, x) and d(i, y) = d(j, y). The second criterion is because, intuitively, a node i’s topological position within an enclosing subgraph can be described by its radius with respect to the two center nodes, namely (d(i, x), d(i, y)). Thus, we let nodes on the same orbit have the same label, so that the node labels can reflect nodes’ relative positions and structural importance within subgraphs.
Based on the above criteria, we propose a Double-Radius Node Labeling (DRNL) as follows. First, assign label 1 to x and y. Then, for any node i with (d(i, x), d(i, y)) = (1, 1), assign label fl(i) = 2. Nodes with radius (1, 2) or (2, 1) get label 3. Nodes with radius (1, 3) or (3, 1) get 4. Nodes with (2, 2) get 5. Nodes with (1, 4) or (4, 1) get 6. Nodes with (2, 3) or (3, 2) get 7. So on and so forth. In other words, we iteratively assign larger labels to nodes with a larger radius w.r.t. both center nodes, where the label fl(i) and the double-radius (d(i, x), d(i, y)) satisfy

[Traditional Chinese Translation]
### 4.1 節點標記

X中的第一個組成部分是每個節點的結構標籤。節點標記是一個函數 fl : V → N，它為封閉子圖中的每個節點i分配一個整數標籤fl(i)。其目的是使用不同的標籤來標記節點在封閉子圖中的不同角色：1）中心節點x和y是鏈路所在的目標節點。2）相對於中心節點具有不同相對位置的節點對鏈路具有不同的結構重要性。一個適當的節點標記應該標記出這些差異。如果我們不標記這些差異，GNN將無法分辨應該在哪兩個目標節點之間預測鏈路存在，從而丟失結構信息。
我們的節點標記方法源於以下標準：1）兩個目標節點x和y始終具有獨特的標籤“1”。2）如果d(i, x) = d(j, x)且d(i, y) = d(j, y)，則節點i和j具有相同的標籤。第二個標準是因為，直觀地說，一個節點i在封閉子圖中的拓撲位置可以由其相對於兩個中心節點的半徑來描述，即(d(i, x), d(i, y))。因此，我們讓在同一軌道上的節點具有相同的標籤，這樣節點標籤就可以反映節點在子圖中的相對位置和結構重要性。
基於上述標準，我們提出了雙半徑節點標記（DRNL）如下。首先，將標籤1分配給x和y。然後，對於任何具有(d(i, x), d(i, y)) = (1, 1)的節點i，分配標籤fl(i) = 2。半徑為(1, 2)或(2, 1)的節點獲得標籤3。半徑為(1, 3)或(3, 1)的節點獲得標籤4。半徑為(2, 2)的節點獲得標籤5。半徑為(1, 4)或(4, 1)的節點獲得標籤6。半徑為(2, 3)或(3, 2)的節點獲得標籤7。以此類推。換句話說，我們迭代地將較大的標籤分配給相對於兩個中心節點半徑較大的節點，其中標籤fl(i)和雙半徑(d(i, x), d(i, y))滿足

[Original English Text]
1) if d(i, x) + d(i, y) ≠ d(j, x) + d(j, y), then d(i,x) + d(i, y) < d(j, x) + d(j, y) ↔ fl(i) < fl(j);
2) if d(i, x) + d(i,y) = d(j,x) + d(j, y), then d(i, x)d(i,y) < d(j,x)d(j, y) → fl(i) < fl(j).
One advantage of DRNL is that it has a perfect hashing function
fl(i) = 1 + min(dx, dy) + (d/2)[(d/2) + (d%2) – 1], (10)
where dx := d(i,x), dy := d(i,y), d := dx + dy, (d/2) and (d%2) are the integer quotient and remainder of d divided by 2, respectively. This perfect hashing allows fast closed-form computations.
For nodes with d(i, x) = ∞ or d(i, y) = ∞, we give them a null label 0. Note that DRNL is not the only possible way of node labeling, but we empirically verified its better performance than no labeling and other naive labelings. We discuss more about node labeling in Appendix B. After getting the labels, we use their one-hot encoding vectors to construct X.

[Traditional Chinese Translation]
1) 如果 d(i, x) + d(i, y) ≠ d(j, x) + d(j, y)，則 d(i,x) + d(i, y) < d(j, x) + d(j, y) ↔ fl(i) < fl(j);
2) 如果 d(i, x) + d(i,y) = d(j,x) + d(j, y)，則 d(i, x)d(i,y) < d(j,x)d(j, y) → fl(i) < fl(j)。
DRNL的一個優點是它有一個完美的哈希函數
fl(i) = 1 + min(dx, dy) + (d/2)[(d/2) + (d%2) – 1], (10)
其中 dx := d(i,x), dy := d(i,y), d := dx + dy, (d/2) 和 (d%2) 分別是 d 除以 2 的整數商和餘數。這個完美的哈希函數允許快速的封閉形式計算。
對於 d(i, x) = ∞ 或 d(i, y) = ∞ 的節點，我們給它們一個空標籤 0。注意 DRNL 並不是節點標記的唯一可能方式，但我們通過經驗驗證了它比無標記和其他樸素的標記方法性能更好。我們在附錄 B 中討論更多關於節點標記的內容。獲得標籤後，我們使用它們的獨熱編碼向量來構建 X。

[Original English Text]
### 4.2 Incorporating latent and explicit features

Other than the structural node labels, the node information matrix X also provides an opportunity to include latent and explicit features. By concatenating each node’s embedding/attribute vector to its corresponding row in X, we can make SEAL simultaneously learn from all three types of features.
Generating the node embeddings for SEAL is nontrivial. Suppose we are given the observed network G = (V, E), a set of sampled positive training links Ep ⊆ E, and a set of sampled negative training links En with En ∩ E = ∅. If we directly generate node embeddings on G, the node embeddings will record the link existence information of the training links (since Ep ⊆ E). We observed that GNNs can quickly find out such link existence information and optimize by only fitting this part of information. This results in bad generalization performance in our experiments. Our trick is to temporally add En into E, and generate the embeddings on G' = (V, E ∪ En). This way, the positive and negative training links will have the same link existence information recorded in the embeddings, so that GNN cannot classify links by only fitting this part of information. We empirically verified the much improved performance of this trick to SEAL. We name this trick negative injection.
We name our proposed framework SEAL (learning from Subgraphs, Embeddings and Attributes for Link prediction), emphasizing its ability to jointly learn from three types of features.

[Traditional Chinese Translation]
### 4.2 融合潛在特徵與顯式特徵

除了結構節點標籤，節點資訊矩陣X也提供了包含潛在特徵和顯式特徵的機會。通過將每個節點的嵌入/屬性向量連接到X中其對應的行，我們可以使SEAL同時從所有三種類型的特徵中學習。
為SEAL生成節點嵌入並非易事。假設我們有觀測到的網絡G = (V, E)，一組採樣的正訓練鏈路Ep ⊆ E，以及一組採樣的負訓練鏈路En，其中En ∩ E = ∅。如果我們直接在G上生成節點嵌入，節點嵌入將記錄訓練鏈路的存在資訊（因為Ep ⊆ E）。我們觀察到GNN可以迅速找出這樣的鏈路存在資訊，並僅通過擬合這部分資訊來進行優化。這在我們的實驗中導致了較差的泛化性能。我們的技巧是暫時將En加入E，並在G' = (V, E ∪ En)上生成嵌入。這樣，正訓練鏈路和負訓練鏈路在嵌入中將記錄相同的鏈路存在資訊，因此GNN不能僅通過擬合這部分資訊來分類鏈路。我們通過經驗驗證了這種技巧對SEAL性能的大幅提升。我們將這個技巧命名為負注入（negative injection）。
我們將我們提出的框架命名為SEAL（從子圖、嵌入和屬性中學習鏈路預測），強調其從三種類型特徵中聯合學習的能力。

[Original English Text]
## 5 Experimental results

We conduct extensive experiments to evaluate SEAL. Our results show that SEAL is a superb and robust framework for link prediction, achieving unprecedentedly strong performance on various networks. We use AUC and average precision (AP) as evaluation metrics. We run all experiments for 10 times and report the average AUC results and standard deviations. We leave the the AP and time results in Appendix F. SEAL is flexible with what GNN or node embeddings to use. Thus, we choose a recent architecture DGCNN [17] as the default GNN, and node2vec [20] as the default embeddings. The code and data are available at https://github.com/muhanzhang/SEAL.
Datasets The eight datasets used are: USAir, NS, PB, Yeast, C.ele, Power, Router, and E.coli (please see Appendix C for details). We randomly remove 10% existing links from each dataset as positive testing data. Following a standard manner of learning-based link prediction, we randomly sample the same number of nonexistent links (unconnected node pairs) as negative testing data. We use the remaining 90% existing links as well as the same number of additionally sampled nonexistent links to construct the training data.
Comparison to heuristic methods We first compare SEAL with methods that only use graph structure features. We include eight popular heuristics (shown in Appendix A, Table 3): common neighbors (CN), Jaccard, preferential attachment (PA), Adamic-Adar (AA), resource allocation (RA), Katz, PageRank (PR), and SimRank (SR). We additionally include Ensemble (ENS) which trains a logistic regression classifier on the eight heuristic scores. We also include two heuristic learning methods: Weisfeiler-Lehman graph kernel (WLK) [34] and WLNM [12], which also learn from (truncated) enclosing subgraphs. We omit path ranking methods [28] as well as other recent methods which are specifically designed for knowledge graphs or recommender systems [23, 35]. As all the baselines only use graph structure features, we restrict SEAL to not include any latent or explicit features. In SEAL, the hop number h is an important hyperparameter. Here, we select h only from {1, 2}, since on one hand we empirically verified that the performance typically does not increase

[Traditional Chinese Translation]
## 5 實驗結果

我們進行了廣泛的實驗來評估SEAL。我們的結果表明，SEAL是一個卓越且穩健的鏈路預測框架，在各種網絡上都取得了前所未有的強大性能。我們使用AUC和平均精度（AP）作為評估指標。我們所有實驗都運行10次，並報告平均AUC結果和標準差。我們將AP和時間結果留在附錄F中。SEAL對於使用何種GNN或節點嵌入是靈活的。因此，我們選擇了最近的架構DGCNN [17]作為默認GNN，以及node2vec [20]作為默認嵌入。代碼和數據可在https://github.com/muhanzhang/SEAL獲取。
數據集 使用的八個數據集是：USAir、NS、PB、Yeast、C.ele、Power、Router和E.coli（詳情請參見附錄C）。我們從每個數據集中隨機刪除10%的現有鏈路作為正測試數據。遵循基於學習的鏈路預測的標準方式，我們隨機採樣相同數量的非現有鏈路（未連接的節點對）作為負測試數據。我們使用剩餘的90%現有鏈路以及相同數量的額外採樣的非現有鏈路來構建訓練數據。
與啟發式方法的比較 我們首先將SEAL與僅使用圖結構特徵的方法進行比較。我們包括八種流行的啟發式方法（如附錄A，表3所示）：共同鄰居（CN）、Jaccard、優先連接（PA）、Adamic-Adar（AA）、資源分配（RA）、Katz、PageRank（PR）和SimRank（SR）。我們還包括Ensemble（ENS），它在八種啟發式分數上訓練一個邏輯回歸分類器。我們還包括兩種啟發式學習方法：Weisfeiler-Lehman圖核（WLK）[34]和WLNM [12]，它們也從（截斷的）封閉子圖中學習。我們省略了路徑排序方法[28]以及其他專為知識圖或推薦系統設計的近期方法[23, 35]。由於所有基線僅使用圖結構特徵，我們限制SEAL不包括任何潛在或顯式特徵。在SEAL中，跳數h是一個重要的超參數。在這裡，我們僅從{1, 2}中選擇h，因為一方面我們通過經驗驗證了性能通常不會在h > 3後增加

[Original English Text]
after h > 3, which validates our theoretical results that the most useful information is within local structures. On the other hand, even h = 3 sometimes results in very large subgraphs if a hub node is included. This raises the idea of sampling nodes in subgraphs, which we leave to future work. The selection principle is very simple: If the second-order heuristic AA outperforms the first-order heuristic CN on 10% validation data, then we choose h = 2; otherwise we choose h = 1. For datasets PB and E.coli, we consistently use h = 1 to fit into the memory. We include more details about the baselines and hyperparameters in Appendix D.

[Traditional Chinese Translation]
h > 3之後性能不再提升，這驗證了我們的理論結果，即最有用的信息在局部結構中。另一方面，如果包含中心節點，即使h=3有時也會導致非常大的子圖。這引出了在子圖中採樣節點的想法，我們將其留給未來的工作。選擇原則非常簡單：如果二階啟發式AA在10%的驗證數據上優於一階啟發式CN，那麼我們選擇h=2；否則我們選擇h=1。對於PB和E.coli數據集，我們始終使用h=1以適應內存。我們在附錄D中包含了有關基線和超參數的更多細節。

[Original English Text]
Table 1 shows the results. Firstly, we observe that methods which learn from enclosing subgraphs (WLK, WLNM and SEAL) generally perform much better than predefined heuristics. This indicates that the learned “heuristics” are better at capturing the network properties than manually designed ones. Among learning-based methods, SEAL has the best performance, demonstrating GNN’s superior graph feature learning ability over graph kernels and fully-connected neural networks. From the results on Power and Router, we can see that although existing heuristics perform similarly to random guess, learning-based methods still maintain high performance. This suggests that we can even discover new “heuristics” for networks where no existing heuristics work.

[Traditional Chinese Translation]
表1顯示了結果。首先，我們觀察到從封閉子圖中學習的方法（WLK、WLNM和SEAL）通常比預定義的啟發式方法表現得好得多。這表明學習到的「啟發式方法」比手動設計的更能捕捉網絡特性。在基於學習的方法中，SEAL表現最佳，展示了GNN在圖特徵學習方面相對於圖核和全連接神經網絡的優越能力。從Power和Router的結果中，我們可以看到，儘管現有的啟發式方法表現得類似於隨機猜測，但基於學習的方法仍然保持高性能。這表明我們甚至可以為那些沒有現有啟發式方法起作用的網絡發現新的「啟發式方法」。

[Original English Text]
Comparison to latent feature methods Next we compare SEAL with six state-of-the-art latent feature methods: matrix factorization (MF), stochastic block model (SBM) [18], node2vec (N2V) [20], LINE [21], spectral clustering (SPC), and variational graph auto-encoder (VGAE) [36]. Among them, VGAE uses a GNN too. Please note the difference between VGAE and SEAL: VGAE uses a node-level GNN to learn node embeddings that best reconstruct the network, while SEAL uses a graph-level GNN to classify enclosing subgraphs. Therefore, VGAE still belongs to latent feature methods. For SEAL, we additionally include the 128-dimensional node2vec embeddings in the node information matrix X. Since the datasets do not have node attributes, explicit features are not included.

[Traditional Chinese Translation]
與潛在特徵方法的比較 接下來，我們將SEAL與六種最先進的潛在特徵方法進行比較：矩陣分解（MF）、隨機區塊模型（SBM）[18]、node2vec（N2V）[20]、LINE[21]、譜聚類（SPC）和變分圖自動編碼器（VGAE）[36]。其中，VGAE也使用了GNN。請注意VGAE和SEAL之間的區別：VGAE使用節點級GNN來學習最能重建網絡的節點嵌入，而SEAL使用圖級GNN來對封閉子圖進行分類。因此，VGAE仍然屬於潛在特徵方法。對於SEAL，我們在節點信息矩陣X中額外包含了128維的node2vec嵌入。由於數據集沒有節點屬性，因此未包含顯式特徵。

[Original English Text]
Table 2 shows the results. As we can see, SEAL shows significant improvement over latent feature methods. One reason is that SEAL learns from both graph structures and latent features simultaneously, thus augmenting those methods that only use latent features. We observe that SEAL with node2vec embeddings outperforms pure node2vec by large margins. This implies that network embeddings alone may not be able to capture the most useful link prediction information located in the local structures. It is also interesting that compared to SEAL without node2vec embeddings (Table 1), joint learning does not always improve the performance. More experiments and discussion are included in Appendix F.

[Traditional Chinese Translation]
表2顯示了結果。正如我們所見，SEAL在潛在特徵方法上表現出顯著的改進。一個原因是SEAL同時從圖結構和潛在特徵中學習，從而增強了那些僅使用潛在特徵的方法。我們觀察到，帶有node2vec嵌入的SEAL在性能上遠超純node2vec。這意味著僅網絡嵌入可能無法捕捉到位於局部結構中最有用的鏈路預測信息。同樣有趣的是，與沒有node2vec嵌入的SEAL（表1）相比，聯合學習並不總是能提高性能。更多的實驗和討論包含在附錄F中。

[Original English Text]
## 6 Conclusions

Learning link prediction heuristics automatically is a new field. In this paper, we presented theoretical justifications for learning from local enclosing subgraphs. In particular, we proposed a γ-decaying theory to unify a wide range of high-order heuristics and prove their approximability from local subgraphs. Motivated by the theory, we proposed a novel link prediction framework, SEAL, to simultaneously learn from local enclosing subgraphs, embeddings and attributes based on graph neural networks. Experimentally we showed that SEAL achieved unprecedentedly strong performance by comparing to various heuristics, latent feature methods, and network embedding algorithms. We hope SEAL can not only inspire link prediction research, but also open up new directions for other relational machine learning problems such as knowledge graph completion and recommender systems.

[Traditional Chinese Translation]
## 6 結論

自動學習鏈路預測啟發式是一個新領域。在本文中，我們提出了從局部封閉子圖中學習的理論依據。特別是，我們提出了一個γ衰減理論來統一廣泛的高階啟發式方法，並證明它們可以從局部子圖中近似。受該理論的啟發，我們提出了一個新穎的鏈路預測框架SEAL，以基於圖神經網絡同時從局部封閉子圖、嵌入和屬性中學習。實驗證明，與各種啟發式方法、潛在特徵方法和網絡嵌入算法相比，SEAL取得了前所未有的強大性能。我們希望SEAL不僅能激發鏈路預測研究，還能為其他關係機器學習問題（如知識圖譜補全和推薦系統）開闢新的方向。

[Original English Text]
## Acknowledgments

The work is supported in part by the III-1526012 and SCH-1622678 grants from the National Science Foundation and grant 1R21HS024581 from the National Institute of Health.

[Traditional Chinese Translation]
## 致謝

本研究部分由美國國家科學基金會的III-1526012和SCH-1622678號撥款，以及美國國家衛生研究院的1R21HS024581號撥款支持。

[Original English Text]
## References

[1] David Liben-Nowell and Jon Kleinberg. The link-prediction problem for social networks. Journal of the American society for information science and technology, 58(7):1019–1031, 2007.
... (and so on for all references)

[Traditional Chinese Translation]
## 參考文獻

[1] David Liben-Nowell and Jon Kleinberg. The link-prediction problem for social networks. Journal of the American society for information science and technology, 58(7):1019–1031, 2007.
... (所有參考文獻保持原文)