---
title: Neural Bellman-Ford Networks
field: Link_Prediction
status: Imported
created_date: 2026-01-14
pdf_link: "[[Neural Bellman-Ford Networks.pdf]]"
tags:
  - paper
  - Link_prediction
---
# 神經貝爾曼-福特網絡：一個用於鏈路預測的通用圖神經網絡框架
Zhaocheng Zhu¹², Zuobai Zhang¹², Louis-Pascal Xhonneux¹², Jian Tang¹,³,⁴
Mila - Québec AI Institute¹, Université de Montréal²
HEC Montréal³, CIFAR AI Chair⁴
{zhaocheng.zhu, zuobai.zhang, louis-pascal.xhonneux}@mila.quebec
jian.tang@hec.ca
祝兆澄¹²、張作柏¹²、Louis-Pascal Xhonneux¹²、唐建¹³,⁴
Mila - 魁北克人工智慧研究所¹、蒙特婁大學²
蒙特婁高等商學院³、CIFAR AI 主席⁴
{zhaocheng.zhu, zuobai.zhang, louis-pascal.xhonneux}@mila.quebec
jian.tang@hec.ca

## 摘要
Link prediction is a very fundamental task on graphs. Inspired by traditional path-based methods, in this paper we propose a general and flexible representation learning framework based on paths for link prediction. Specifically, we define the representation of a pair of nodes as the generalized sum of all path representations between the nodes, with each path representation as the generalized product of the edge representations in the path. Motivated by the Bellman-Ford algorithm for solving the shortest path problem, we show that the proposed path formulation can be efficiently solved by the generalized Bellman-Ford algorithm. To further improve the capacity of the path formulation, we propose the Neural Bellman-Ford Network (NBFNet), a general graph neural network framework that solves the path formulation with learned operators in the generalized Bellman-Ford algorithm. The NBFNet parameterizes the generalized Bellman-Ford algorithm with 3 neural components, namely INDICATOR, MESSAGE and AGGREGATE functions, which corresponds to the boundary condition, multiplication operator, and summation operator respectively¹. The NBFNet covers many traditional path-based methods, and can be applied to both homogeneous graphs and multi-relational graphs (e.g., knowledge graphs) in both transductive and inductive settings. Experiments on both homogeneous graphs and knowledge graphs show that the proposed NBFNet outperforms existing methods by a large margin in both transductive and inductive settings, achieving new state-of-the-art results².
鏈路預測是圖論中一項非常基礎的任務。受傳統基於路徑的方法啟發，本文提出了一個通用且靈活的、基於路徑的表示學習框架，用於鏈路預測。具體來說，我們將一對節點的表示定義為節點之間所有路徑表示的廣義總和，其中每個路徑表示是路徑中邊表示的廣義乘積。受解決最短路徑問題的 Bellman-Ford 算法的啟發，我們證明了所提出的路徑公式可以通過廣義 Bellman-Ford 算法有效解決。為了進一步提升路徑公式的能力，我們提出了神經貝爾曼-福特網絡 (NBFNet)，這是一個通用的圖神經網絡框架，它在廣義 Bellman-Ford 算法中使用學習到的運算符來解決路徑公式問題。NBFNet 使用三個神經組件來參數化廣義 Bellman-Ford 算法，即 INDICATOR、MESSAGE 和 AGGREGATE 函數，分別對應邊界條件、乘法運算符和加法運算符¹。NBFNet 涵蓋了許多傳統的基於路徑的方法，並且可以應用於同質圖和多關係圖（例如知識圖）的轉導式和歸納式設置。在同質圖和知識圖上的實驗表明，所提出的 NBFNet 在轉導式和歸納式設置中均大幅優於現有方法，達到了新的最先進水平²。
¹Unless stated otherwise, we use summation and multiplication to refer the generalized operators in the path formulation, rather than the basic operations of arithmetic.
²Code is available at https://github.com/DeepGraphLearning/NBFNet
¹除非另有說明，我們使用「總和」與「乘積」來指代路徑公式中的廣義運算符，而非基本的算術運算。
²程式碼可在 https://github.com/DeepGraphLearning/NBFNet 取得

35th Conference on Neural Information Processing Systems (NeurIPS 2021).

第35屆神經資訊處理系統會議 (NeurIPS 2021)。

# 1 簡介
Predicting the interactions between nodes (a.k.a. link prediction) is a fundamental task in the field of graph machine learning. Given the ubiquitous existence of graphs, such a task has many applications, such as recommender system [34], knowledge graph completion [41] and drug repurposing [27].
預測節點間的互動（亦即鏈路預測）是圖機器學習領域的一項基礎任務。鑑於圖的普遍存在，這項任務有許多應用，例如推薦系統 [34]、知識圖補全 [41] 和藥物再利用 [27]。
Traditional methods of link prediction usually define different heuristic metrics over the paths between a pair of nodes. For example, Katz index [30] is defined as a weighted count of paths between two nodes. Personalized PageRank [42] measures the similarity of two nodes as the random walk probability from one to the other. Graph distance [37] uses the length of the shortest path between two nodes to predict their association. These methods can be directly applied to new graphs, i.e., inductive setting, enjoy good interpretability and scale up to large graphs. However, they are designed based on handcrafted metrics and may not be optimal for link prediction on real-world graphs.
傳統的鏈路預測方法通常在一對節點之間的路徑上定義不同的啟發式度量。例如，Katz 指數 [30] 被定義為兩個節點之間路徑的加權計數。個人化 PageRank [42] 將兩個節點的相似性度量為從一個節點到另一個節點的隨機遊走機率。圖距離 [37] 使用兩個節點之間最短路徑的長度來預測它們的關聯。這些方法可以直接應用於新的圖，即歸納式設置，具有良好的可解釋性，並且可以擴展到大型圖。然而，它們是基於手工製作的度量設計的，對於真實世界的圖上的鏈路預測可能不是最佳的。
To address these limitations, some link prediction methods adopt graph neural networks (GNNs) [32, 48, 59] to automatically extract important features from local neighborhoods for link prediction. Thanks to the high expressiveness of GNNs, these methods have shown state-of-the-art performance. However, these methods can only be applied to predict new links on the training graph, i.e. transductive setting, and lack interpretability. While some recent methods [73, 55] extract features from local subgraphs with GNNs and support inductive setting, the scalability of these methods is compromised.
為了解決這些限制，一些鏈路預測方法採用圖神經網絡 (GNNs) [32, 48, 59] 來自動從局部鄰域中提取重要特徵。由於 GNN 的高表達能力，這些方法已展現出最先進的性能。然而，這些方法只能應用於預測訓練圖上的新鏈路，即轉導式設置，並且缺乏可解釋性。雖然一些最近的方法 [73, 55] 從帶有 GNN 的局域子圖中提取特徵並支援歸納式設置，但這些方法的可擴展性受到了影響。
Therefore, we wonder if there exists an approach that enjoys the advantages of both traditional path-based methods and recent approaches based on graph neural networks, i.e., generalization in the inductive setting, interpretability, high model capacity and scalability.
因此，我們想知道是否存在一種方法，既能享受傳統基於路徑方法的優點，又能享受基於圖神經網絡的最新方法的優點，即在歸納設置中的泛化能力、可解釋性、高模型容量和可擴展性。
In this paper, we propose such a solution. Inspired by traditional path-based methods, our goal is to develop a general and flexible representation learning framework for link prediction based on the paths between two nodes. Specifically, we define the representation of a pair of nodes as the generalized sum of all the path representations between them, where each path representation is defined as the generalized product of the edge representations in the path. Many link prediction methods, such as Katz index [30], personalized PageRank [42], graph distance [37], as well as graph theory algorithms like widest path [4] and most reliable path [4], are special instances of this path formulation with different summation and multiplication operators. Motivated by the polynomial-time algorithm for the shortest path problem [5], we show that such a formulation can be efficiently solved via the generalized Bellman-Ford algorithm [4] under mild conditions and scale up to large graphs.
在本文中，我們提出了這樣一種解決方案。受到傳統基於路徑的方法的啟發，我們的目標是開發一個通用且靈活的表示學習框架，用於基於兩個節點之間的路徑進行鏈路預測。具體來說，我們將一對節點的表示定義為它們之間所有路徑表示的廣義和，其中每個路徑表示被定義為路徑中邊表示的廣義積。許多鏈路預測方法，例如 Katz 指數 [30]、個人化 PageRank [42]、圖距離 [37]，以及圖論算法如最寬路徑 [4] 和最可靠路徑 [4]，都是此路徑公式在不同求和與乘法運算符下的特例。受到最短路徑問題的多項式時間算法 [5] 的啟發，我們證明了這樣的公式可以在溫和的條件下通過廣義 Bellman-Ford 算法 [4] 有效地解決，並擴展到大型圖。
The operators in the generalized Bellman-Ford algorithm—summation and multiplication—are handcrafted, which have limited flexibility. Therefore, we further propose the Neural Bellman-Ford Networks (NBFNet), a graph neural network framework that solves the above path formulation with learned operators in the generalized Bellman-Ford algorithm. Specifically, NBFNet parameterizes the generalized Bellman-Ford algorithm with three neural components, namely INDICATOR, MESSAGE and AGGREGATE functions. The INDICATOR function initializes a representation on each node, which is taken as the boundary condition of the generalized Bellman-Ford algorithm. The MESSAGE and the AGGREGATE functions learn the multiplication and summation operators respectively.
廣義 Bellman-Ford 算法中的運算符——求和與乘法——是手工設計的，這限制了其靈活性。因此，我們進一步提出了神經 Bellman-Ford 網絡 (NBFNet)，一個圖神經網絡框架，該框架使用學習到的運算符在廣義 Bellman-Ford 算法中解決上述路徑公式。具體來說，NBFNet 用三個神經組件來參數化廣義 Bellman-Ford 算法，即 INDICATOR、MESSAGE 和 AGGREGATE 函數。INDICATOR 函數在每個節點上初始化一個表示，作為廣義 Bellman-Ford 算法的邊界條件。MESSAGE 和 AGGREGATE 函數分別學習乘法和求和運算符。
We show that the MESSAGE function can be defined according to the relational operators in knowledge graph embeddings [6, 68, 58, 31, 52], e.g., as a translation in Euclidean space induced by the relational operators of TransE [6]. The AGGREGATE function can be defined as learnable set aggregation functions [71, 65, 9]. With such parameterization, NBFNet can generalize to the inductive setting, meanwhile achieve one of the lowest time complexity among inductive GNN methods. A comparison of NBFNet and other GNN frameworks for link prediction is showed in Table 1. With other instantiations of MESSAGE and AGGREGATE functions, our framework can also recover some existing works on learning logic rules [69, 46] for link prediction on knowledge graphs (Table 2).
我們證明 MESSAGE 函數可以根據知識圖嵌入中的關係運算符 [6, 68, 58, 31, 52] 來定義，例如，由 TransE [6] 的關係運算符在歐幾里得空間中引導的平移。AGGREGATE 函數可以定義為可學習的集合聚合函數 [71, 65, 9]。通過這樣的參數化，NBFNet 可以推廣到歸納設置，同時在歸納 GNN 方法中實現最低的時間複雜度之一。表 1 顯示了 NBFNet 與其他用於鏈路預測的 GNN 框架的比較。通過 MESSAGE 和 AGGREGATE 函數的其他實例化，我們的框架還可以恢復一些現有的關於學習知識圖上鏈路預測的邏輯規則的工作 [69, 46]（表 2）。
Our NBFNet framework can be applied to several link prediction variants, covering not only single-relational graphs (e.g., homogeneous graphs) but also multi-relational graphs (e.g., knowledge graphs). We empirically evaluate the proposed NBFNet for link prediction on homogeneous graphs and knowledge graphs in both transductive and inductive settings. Experimental results show that the proposed NBFNet outperforms existing state-of-the-art methods by a large margin in all settings, with an average relative performance gain of 18% on knowledge graph completion (HITS@1) and 22% on inductive relation prediction (HITS@10). We also show that the proposed NBFNet is indeed interpretable by visualizing the top-k relevant paths for link prediction on knowledge graphs.
我們的 NBFNet 框架可以應用於多種鏈路預測變體，不僅涵蓋單一關係圖（例如，同質圖），也涵蓋多重關係圖（例如，知識圖）。我們在同質圖和知識圖上，以轉導和歸納兩種設定，對所提出的 NBFNet 進行了鏈路預測的實證評估。實驗結果顯示，所提出的 NBFNet 在所有設定中，皆以顯著的差距優於現有的最新方法，在知識圖補全（HITS@1）方面平均相對性能提升了 18%，在歸納關係預測（HITS@10）方面提升了 22%。我們也透過視覺化知識圖上鏈路預測的前 k 條相關路徑，證明了所提出的 NBFNet 確實具有可解釋性。
Table 1: Comparison of GNN frameworks for link prediction. The time complexity refers to the amortized time for predicting a single edge or triplet. |V| is the number of nodes, |E| is the number of edges, and d is the dimension of representations. The wall time is measured on FB15k-237 test set with 40 CPU cores and 4 GPUs. We estimate the wall time of GraIL based on a downsampled test set.
表格 1：鏈路預測的 GNN 框架比較。時間複雜度指的是預測單一邊或三元組的攤銷時間。|V| 是節點數，|E| 是邊數，d 是表示的維度。牆上時間是在擁有 40 個 CPU 核心和 4 個 GPU 的 FB15k-237 測試集上測量的。我們基於一個降採樣的測試集來估計 GraIL 的牆上時間。

| Method | Inductive³ | Interpretable | Learned Representation | Time Complexity | Wall Time |
| :--- | :---: | :---: | :---: | :---: | :---: |
| VGAE [32] / RGCN [48] | | | ✓ | O(d) | 18 secs |
| NeuralLP [69] / DRUM [46] | ✓ | ✓ | | O((|E|d/|V|) + d²) | 2.1 mins |
| SEAL [73] / GraIL [55] | ✓ | ✓ | ✓ | O(|E|d²) | ≈1 month |
| NBFNet | ✓ | ✓ | ✓ | O((|E|d/|V|) + d²) | 4.0 mins |

# 2 相關研究
Existing work on link prediction can be generally classified into 3 main paradigms: path-based methods, embedding methods, and graph neural networks.
現有的鏈路預測研究大致可分為三大類：基於路徑的方法、嵌入方法以及圖神經網絡。
**Path-based Methods.** Early methods on homogeneous graphs compute the similarity between two nodes based on the weighted count of paths (Katz index [30]), random walk probability (personalized PageRank [42]) or the length of the shortest path (graph distance [37]). SimRank [28] uses advanced metrics such as the expected meeting distance on homogeneous graphs, which is extended by PathSim [51] to heterogeneous graphs. On knowledge graphs, Path Ranking [35, 15] directly uses relational paths as symbolic features for prediction. Rule mining methods, such as NeuralLP [69] and DRUM [46], learn probabilistic logical rules to weight different paths. Path representation methods, such as Path-RNN [40] and its successors [11, 62], encode each path with recurrent neural networks (RNNs), and aggregate paths for prediction. However, these methods need to traverse an exponential number of paths and are limited to very short paths, e.g., ≤ 3 edges. To scale up path-based methods, All-Paths [57] proposes to efficiently aggregate all paths with dynamic programming. However, All-Paths is restricted to bilinear models and has limited model capacity. Another stream of works [64, 10, 22] learns an agent to collect useful paths for link prediction. While these methods can produce interpretable paths, they suffer from extremely sparse rewards and require careful engineering of the reward function [38] or the search strategy [50]. Some other works [8, 44] adopt variational inference to learn a path finder and a path reasoner for link prediction.
**基於路徑的方法。** 早期的同構圖方法根據路徑的加權計數（Katz 指數 [30]）、隨機遊走機率（個人化 PageRank [42]）或最短路徑長度（圖距離 [37]）來計算兩個節點之間的相似度。SimRank [28] 在同構圖上使用如預期相遇距離等進階度量，PathSim [51] 將其擴展至異構圖。在知識圖方面，路徑排序 [35, 15] 直接使用關係路徑作為預測的符號特徵。規則探勘方法，如 NeuralLP [69] 和 DRUM [46]，學習機率性邏輯規則來加權不同路徑。路徑表示方法，如 Path-RNN [40] 及其後繼者 [11, 62]，使用循環神經網絡 (RNNs) 編碼每條路徑，並匯總路徑進行預測。然而，這些方法需要遍歷指數級數量的路徑，且僅限於非常短的路徑，例如 ≤ 3 條邊。為擴展基於路徑的方法，All-Paths [57] 提出使用動態規劃有效匯總所有路徑。然而，All-Paths 僅限於雙線性模型且模型容量有限。另一類研究 [64, 10, 22] 學習一個代理來收集用於鏈路預測的有用路徑。雖然這些方法可以產生可解釋的路徑，但它們受制於極度稀疏的獎勵，並需要仔細設計獎勵函數 [38] 或搜索策略 [50]。其他一些研究 [8, 44] 則採用變分推斷來學習路徑尋找器和路徑推理器以進行鏈路預測。
**Embedding Methods.** Embedding methods learn a distributed representation for each node and edge by preserving the edge structure of the graph. Representative methods include DeepWalk [43] and LINE [53] on homogeneous graphs, and TransE [6], DistMult [68] and RotatE [52] on knowledge graphs. Later works improve embedding methods with new score functions [58, 13, 31, 52, 54, 76] that capture common semantic patterns of the relations, or search the score function in a general design space [75]. Embedding methods achieve promising results on link prediction, and can be scaled to very large graphs using multiple GPUs [78]. However, embedding methods do not explicitly encode local subgraphs between node pairs and cannot be applied to the inductive setting.
**嵌入方法。** 嵌入方法通過保留圖的邊緣結構，為每個節點和邊緣學習一個分佈式表示。代表性方法包括在同構圖上的 DeepWalk [43] 和 LINE [53]，以及在知識圖上的 TransE [6]、DistMult [68] 和 RotatE [52]。後續工作通過捕捉關係的共同語義模式的新評分函數 [58, 13, 31, 52, 54, 76] 或在通用設計空間 [75] 中搜索評分函數來改進嵌入方法。嵌入方法在鏈路預測上取得了有希望的結果，並且可以使用多個 GPU [78] 擴展到非常大的圖。然而，嵌入方法沒有明確地編碼節點對之間的局部子圖，並且不能應用於歸納設置。
**Graph Neural Networks.** Graph neural networks (GNNs) [47, 33, 60, 65] are a family of representation learning models that encode topological structures of graphs. For link prediction, the prevalent frameworks [32, 48, 12, 59] adopt an auto-encoder formulation, which uses GNNs to encode node representations, and decodes edges as a function over node pairs. Such frameworks are potentially inductive if the dataset provides node features, but are transductive only when node features are unavailable. Another stream of frameworks, such as SEAL [73] and GraIL [55], explicitly encodes the subgraph around each node pair for link prediction. While these frameworks are proved to be more powerful than the auto-encoder formulation [74] and can solve the inductive setting, they require to materialize a subgraph for each link, which is not scalable to large graphs. By contrast, our NBFNet explicitly captures the paths between two nodes for link prediction, meanwhile achieves a relatively low time complexity (Table 1). ID-GNN [70] formalizes link prediction as a conditional node classification task, and augments GNNs with the identity of the source node. While the architecture of NBFNet shares some spirits with ID-GNN, our model is motivated by the generalized Bellman-Ford algorithm and has theoretical connections with traditional path-based methods. There are also some works trying to scale up GNNs for link prediction by dynamically pruning the set of nodes in message passing [66, 20]. These methods are complementary to NBFNet, and may be incorporated into our method to further improve scalability.
**圖神經網絡。** 圖神經網絡 (GNNs) [47, 33, 60, 65] 是一系列表示學習模型，用於編碼圖的拓撲結構。對於鏈路預測，普遍的框架 [32, 48, 12, 59] 採用自編碼器公式，使用 GNNs 編碼節點表示，並將邊緣解碼為節點對的函數。如果數據集提供節點特徵，這樣的框架可能是歸納性的，但當節點特徵不可用時，它們僅是傳導性的。另一類框架，如 SEAL [73] 和 GraIL [55]，明確地編碼每個節點對周圍的子圖以進行鏈路預測。雖然這些框架被證明比自編碼器公式 [74] 更強大，並且可以解決歸納設置，但它們需要為每個鏈路具體化一個子圖，這對於大型圖是不可擴展的。相比之下，我們的 NBFNet 明確地捕捉兩個節點之間的路徑以進行鏈路預測，同時實現了相對較低的時間複雜度（表 1）。ID-GNN [70] 將鏈路預測形式化為一個條件節點分類任務，並用源節點的身份來增強 GNNs。雖然 NBFNet 的架構與 ID-GNN 有些相似之處，但我們的模型受到廣義 Bellman-Ford 算法的啟發，並與傳統的基於路徑的方法有理論上的聯繫。還有一些工作試圖通過在消息傳遞中動態修剪節點集來擴展 GNNs 的鏈路預測 [66, 20]。這些方法與 NBFNet 是互補的，並且可以被納入我們的方法以進一步提高可擴展性。

# 3 方法論
In this section, we first define a path formulation for link prediction. Our path formulation generalizes several traditional methods, and can be efficiently solved by the generalized Bellman-Ford algorithm. Then we propose Neural Bellman-Ford Networks to learn the path formulation with neural functions.
在本節中，我們首先定義了鏈路預測的路徑公式。我們的路徑公式推廣了幾種傳統方法，並且可以通過廣義 Bellman-Ford 算法有效地解決。然後我們提出神經 Bellman-Ford 網絡，以神經函數學習路徑公式。

## 3.1 鏈路預測的路徑公式
We consider the link prediction problem on both knowledge graphs and homogeneous graphs. A ³We consider the inductive setting where a model can generalize to entirely new graphs without node features.
我們考慮知識圖和同質圖上的鏈路預測問題。³我們考慮歸納設置，其中模型可以推廣到完全沒有節點特徵的新圖。
knowledge graph is denoted by G = (V, E, R), where V and E represent the set of entities (nodes) and relations (edges) respectively, and R is the set of relation types. We use N(u) to denote the set of nodes connected to u, and E(u) to denote the set of edges ending with node u. A homogeneous graph G = (V, E) can be viewed as a special case of knowledge graphs, with only one relation type for all edges. Throughout this paper, we use bold terms, wq(e) or hq(u, v), to denote vector representations, and italic terms, we or wuv, to denote scalars like the weight of edge (u, v) in homogeneous graphs or triplet (u, r, v) in knowledge graphs. Without loss of generality, we derive our method based on knowledge graphs, while our method can also be applied to homogeneous graphs.
知識圖表示為 G = (V, E, R)，其中 V 和 E 分別代表實體（節點）和關係（邊）的集合，R 是關係類型的集合。我們使用 N(u) 表示與 u 相連的節點集合，E(u) 表示以節點 u 結尾的邊集合。同構圖 G = (V, E) 可視為知識圖的一個特例，其中所有邊只有一種關係類型。在本文中，我們使用粗體術語，如 wq(e) 或 hq(u, v) 來表示向量表示，使用斜體術語，如 we 或 wuv 來表示標量，如在同構圖中邊 (u, v) 的權重或在知識圖中三元組 (u, r, v) 的權重。在不失一般性的情況下，我們基於知識圖推導我們的方法，而我們的方法也可以應用於同構圖。
**Path Formulation.** Link prediction is aimed at predicting the existence of a query relation q between a head entity u and a tail entity v. From a representation learning perspective, this requires to learn a pair representation hq(u, v), which captures the local subgraph structure between u and v w.r.t. the query relation q. In traditional methods, such a local structure is encoded by counting different types of random walks from u to v [35, 15]. Inspired by this construction, we formulate the pair representation as a generalized sum of path representations between u and v with a commutative summation operator ⊕. Each path representation hq(P) is defined as a generalized product of the edge representations in the path with the multiplication operator ⊗.
hq(u, v) = ⊕P∈Puv hq(P) (1)
hq(P = (e1, e2, . . . , e|P|)) = ⊗|P|i=1 wq(ei) (2)
where Puv denotes the set of paths from u to v and wq(ei) is the representation of edge ei. Note the multiplication operator ⊗ is not required to be commutative (e.g., matrix multiplication), therefore we define ⊗ to compute the product following the exact order. Intuitively, the path formulation can be interpreted as a depth-first-search (DFS) algorithm, where one searches all possible paths from u to v, computes their representations (Equation 2) and aggregates the results (Equation 1). Such a formulation is capable of modeling several traditional link prediction methods, as well as graph theory algorithms. Formally, Theorem 1–5 state the corresponding path formulations for 3 link prediction methods and 2 graph theory algorithms respectively. See Appendix A for proofs.
**路徑公式化**。鏈路預測旨在預測頭實體 u 和尾實體 v 之間查詢關係 q 的存在。從表示學習的角度來看，這需要學習一個配對表示 hq(u, v)，它捕捉了 u 和 v 之間關於查詢關係 q 的局部子圖結構。在傳統方法中，這種局部結構是通過計算從 u 到 v 的不同類型的隨機遊走來編碼的 [35, 15]。受此結構的啟發，我們將配對表示公式化為 u 和 v 之間路徑表示的廣義和，使用可交換的求和運算符 ⊕。每個路徑表示 hq(P) 被定義為路徑中邊表示的廣義積，使用乘法運算符 ⊗。
hq(u, v) = ⊕P∈Puv hq(P) (1)
hq(P = (e1, e2, . . . , e|P|)) = ⊗|P|i=1 wq(ei) (2)
其中 Puv 表示從 u 到 v 的路徑集合，wq(ei) 是邊 ei 的表示。注意，乘法運算符 ⊗ 不需要是可交換的（例如，矩陣乘法），因此我們定義 ⊗ 來按照確切的順序計算乘積。直觀地，路徑公式可以解釋為深度優先搜索 (DFS) 算法，其中搜索從 u 到 v 的所有可能路徑，計算它們的表示（方程式 2），並匯總結果（方程式 1）。這樣的公式能夠建模幾種傳統的鏈路預測方法，以及圖論算法。形式上，定理 1-5 陳述了 3 種鏈路預測方法和 2 種圖論算法的相應路徑公式。證明見附錄 A。
**Theorem 1** *Katz index is a path formulation with ⊕ = +, ⊗ = × and wq(e) = βwe.*
**定理 1** *Katz 指數是一個路徑公式，其中 ⊕ = +，⊗ = × 且 wq(e) = βwe。*
**Theorem 2** *Personalized PageRank is a path formulation with ⊕ = +, ⊗ = × and wq(e) = αwuv / Σu'∈N(u) wu'v'.*
**定理 2** *個人化 PageRank 是一個路徑公式，其中 ⊕ = +，⊗ = × 且 wq(e) = αwuv / Σu'∈N(u) wu'v'。*
**Theorem 3** *Graph distance is a path formulation with ⊕ = min, ⊗ = + and wq(e) = we.*
**定理 3** *圖距離是一個路徑公式，其中 ⊕ = min，⊗ = + 且 wq(e) = we。*
**Theorem 4** *Widest path is a path formulation with ⊕ = max, ⊗ = min and wq(e) = we.*
**定理 4** *最寬路徑是一個路徑公式，其中 ⊕ = max，⊗ = min 且 wq(e) = we。*
**Theorem 5** *Most reliable path is a path formulation with ⊕ = max, ⊗ = × and wq(e) = we.*
**定理 5** *最可靠路徑是一個路徑公式，其中 ⊕ = max，⊗ = × 且 wq(e) = we。*
**Generalized Bellman-Ford Algorithm.** While the above formulation is able to model important heuristics for link prediction, it is computationally expensive since the number of paths grows exponentially with the path length. Previous works [40, 11, 62] that directly computes the exponential number of paths can only afford a maximal path length of 3. A more scalable solution is to use the generalized Bellman-Ford algorithm [4]. Specifically, assuming the operators (⊕, ⊗) satisfy a semiring system [21] with summation identity ⊕q and multiplication identity ⊗q, we have the following algorithm:
h(0)q (u, v) ← 1q(u = v) (3)
h(t)q (u, v) ← ⊕(x,r,v)∈E(v) (h(t−1)q (u, x) ⊗ wq(x, r, v)) ⊕ h(t)q (u, v) (4)
where 1q(u = v) is the indicator function that outputs ⊗q if u = v and ⊕q otherwise. wq(x, r, v) is the representation for edge e = (x, r, v) and r is the relation type of the edge. Equation 3 is known as the boundary condition, while Equation 4 is known as the Bellman-Ford iteration. The high-level idea of the generalized Bellman-Ford algorithm is to compute the pair representation hq(u, v) for a given entity u, a given query relation q and all v ∈ V in parallel, and reduce the
**廣義 Bellman-Ford 算法。** 雖然上述公式能夠為鏈路預測建模重要的啟發式方法，但其計算成本高昂，因為路徑數量隨路徑長度呈指數增長。先前直接計算指數數量路徑的研究 [40, 11, 62] 僅能處理最大路徑長度為 3 的情況。一個更具擴展性的解決方案是使用廣義 Bellman-Ford 算法 [4]。具體來說，假設運算符 (⊕, ⊗) 滿足一個半環系統 [21]，其中加法恆等元為 ⊕q，乘法恆等元為 ⊗q，我們有以下算法：
h(0)q (u, v) ← 1q(u = v) (3)
h(t)q (u, v) ← ⊕(x,r,v)∈E(v) (h(t−1)q (u, x) ⊗ wq(x, r, v)) ⊕ h(t)q (u, v) (4)
其中 1q(u = v) 是指示函數，如果 u = v 則輸出 ⊗q，否則輸出 ⊕q。wq(x, r, v) 是邊 e = (x, r, v) 的表示，r 是邊的關係類型。方程式 3 被稱為邊界條件，而方程式 4 則被稱為 Bellman-Ford 迭代。廣義 Bellman-Ford 算法的高層思想是，對於給定的實體 u、查詢關係 q 和所有 v ∈ V，並行計算配對表示 hq(u, v)，並減少
total computation by the distributive property of multiplication over summation. Since u and q are fixed in the generalized Bellman-Ford algorithm, we may abbreviate h(t)q (u, v) as h(t)v when the context is clear. When ⊕ = min and ⊗ = +, it recovers the original Bellman-Ford algorithm for the shortest path problem [5]. See Appendix B for preliminaries and the proof of the above algorithm.
總計算量由乘法對加法的分配律性質決定。由於 u 和 q 在廣義 Bellman-Ford 算法中是固定的，當上下文清晰時，我們可將 h(t)q (u, v) 縮寫為 h(t)v。當 ⊕ = min 且 ⊗ = + 時，它恢復為用於最短路徑問題的原始 Bellman-Ford 算法 [5]。有關預備知識及上述算法的證明，請參見附錄 B。
**Theorem 6** *Katz index, personalized PageRank, graph distance, widest path and most reliable path can be solved via the generalized Bellman-Ford algorithm.*
**定理 6** *Katz 指數、個人化 PageRank、圖距離、最寬路徑和最可靠路徑可以通過廣義 Bellman-Ford 算法解決。*
Table 2: Comparison of operators in NBFNet and other methods from the view of path formulation.

| Class | Method | MESSAGE wq(ei) ⊗ wq(ej) | AGGREGATE hq(Pi) ⊕ hq(Pj) | INDICATOR ⊗q, ⊕q | Edge Representation wq(e) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Traditional Link Prediction | Katz Index [30] | wq(ei) × wq(ej) | hq(Pi) + hq(Pj) | 0, 1 | βwe |
| | Personalized PageRank [42] | wq(ei) × wq(ej) | hq(Pi) + hq(Pj) | 0, 1 | αwuv/Σu'∈N(u) wu'v' |
| | Graph Distance [37] | wq(ei) + wq(ej) | min(hq(Pi), hq(Pj)) | +∞, 0 | we |
| Graph Theory Algorithms | Widest Path [4] | min(wq(ei), wq(ej)) | max(hq(Pi), hq(Pj)) | +∞, -∞ | we |
| | Most Reliable Path [4] | wq(ei) × wq(ej) | max(hq(Pi), hq(Pj)) | 0, 1 | we |
| Logic Rules | NeuralLP [69] / DRUM [46] | wq(ei) × wq(ej) | hq(Pi) + hq(Pj) | 0, 1 | Weights learned by LSTM [23] |
| NBFNet | Relational operators of knowledge graph embeddings [6, 68, 52] | Learned set aggregators [9] | Learned indicator functions | Learned relation embeddings |
表格 2：NBFNet 與其他方法在路徑公式化觀點下的運算符比較。

| 類別 | 方法 | MESSAGE wq(ei) ⊗ wq(ej) | AGGREGATE hq(Pi) ⊕ hq(Pj) | INDICATOR ⊗q, ⊕q | 邊表示 wq(e) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 傳統鏈路預測 | Katz 指數 [30] | wq(ei) × wq(ej) | hq(Pi) + hq(Pj) | 0, 1 | βwe |
| | 個人化 PageRank [42] | wq(ei) × wq(ej) | hq(Pi) + hq(Pj) | 0, 1 | αwuv/Σu'∈N(u) wu'v' |
| | 圖距離 [37] | wq(ei) + wq(ej) | min(hq(Pi), hq(Pj)) | +∞, 0 | we |
| 圖論算法 | 最寬路徑 [4] | min(wq(ei), wq(ej)) | max(hq(Pi), hq(Pj)) | +∞, -∞ | we |
| | 最可靠路徑 [4] | wq(ei) × wq(ej) | max(hq(Pi), hq(Pj)) | 0, 1 | we |
| 邏輯規則 | NeuralLP [69] / DRUM [46] | wq(ei) × wq(ej) | hq(Pi) + hq(Pj) | 0, 1 | LSTM [23] 學習的權重 |
| NBFNet | 知識圖嵌入的關係運算符 [6, 68, 52] | 學習的集合聚合器 [9] | 學習的指示函數 | 學習的關係嵌入 |

## 3.2 神經貝爾曼-福特網絡
While the generalized Bellman-Ford algorithm can solve many classical methods (Theorem 6), these methods instantiate the path formulation with handcrafted operators (Table 2), and may not be optimal for link prediction. To improve the capacity of path formulation, we propose a general framework, Neural Bellman-Ford Networks (NBFNet), to learn the operators in the pair representations.
雖然廣義 Bellman-Ford 算法可以解決許多經典方法（定理 6），但這些方法使用手工製作的運算符（表 2）來實例化路徑公式，這對於鏈路預測可能不是最佳的。為了提高路徑公式的能力，我們提出了一個通用框架，即神經 Bellman-Ford 網絡（NBFNet），以學習配對表示中的運算符。
**Neural Parameterization.** We relax the semiring assumption and parameterize the generalized Bellman-Ford algorithm (Equation 3 and 4) with 3 neural functions, namely INDICATOR, MESSAGE and AGGREGATE functions. The INDICATOR function replaces the indicator function 1q(u = v). The MESSAGE function replaces the binary multiplication operator ⊗. The AGGREGATE function is a permutation invariant function over sets that replaces the n-ary summation operator ⊕. Note that one may alternatively define AGGREGATE as the commutative binary operator ⊕ and apply it to a sequence of messages. However, this will make the parameterization more complicated.
**神經參數化。** 我們放寬了半環假設，並用 3 個神經函數，即 INDICATOR、MESSAGE 和 AGGREGATE 函數，對廣義 Bellman-Ford 算法（方程式 3 和 4）進行參數化。INDICATOR 函數取代了指示函數 1q(u = v)。MESSAGE 函數取代了二元乘法運算符 ⊗。AGGREGATE 函數是一個在集合上置換不變的函數，取代了 n 元求和運算符 ⊕。注意，也可以將 AGGREGATE 定義為可交換的二元運算符 ⊕ 並將其應用於一系列消息。然而，這會使參數化更加複雜。
**Algorithm 1** Neural Bellman-Ford Networks
**Input:** source node u, query relation q, #layers T
**Output:** pair representations hq(u, v) for all v ∈ V
1: **for** v ∈ V **do**
2:     h(0)v ← INDICATOR(u, v, q) ▷ Boundary condition
3: **end for**
4: **for** t = 1 to T **do** ▷ Bellman-Ford iteration
5:     **for** v ∈ V **do**
6:         M(t)v ← {h(t)v } ▷ Message augmentation
7:         **for** (x, r, v) ∈ E(v) **do**
8:             m(x,r,v) ← MESSAGE(t)(h(t−1)x , wq(x, r, v))
9:             M(t)v ← M(t)v ∪ {m(x,r,v)}
10:        **end for**
11:        h(t)v ← AGGREGATE(t)(M(t)v )
12:    **end for**
13: **end for**
14: **return** h(T)v as hq(u, v) for all v ∈ V
**算法 1** 神經 Bellman-Ford 網絡
**輸入:** 源節點 u, 查詢關係 q, 層數 T
**輸出:** 對於所有 v ∈ V 的配對表示 hq(u, v)
1: **對於** v ∈ V **執行**
2:     h(0)v ← INDICATOR(u, v, q) ▷ 邊界條件
3: **結束**
4: **對於** t = 1 到 T **執行** ▷ Bellman-Ford 迭代
5:     **對於** v ∈ V **執行**
6:         M(t)v ← {h(t)v } ▷ 訊息增強
7:         **對於** (x, r, v) ∈ E(v) **執行**
8:             m(x,r,v) ← MESSAGE(t)(h(t−1)x , wq(x, r, v))
9:             M(t)v ← M(t)v ∪ {m(x,r,v)}
10:        **結束**
11:        h(t)v ← AGGREGATE(t)(M(t)v )
12:    **結束**
13: **結束**
14: **返回** h(T)v 作為 hq(u, v) 對於所有 v ∈ V
Now consider the generalized Bellman-Ford algorithm for a given entity u and relation q. In this context, we abbreviate h(t)q (u, v) as h(t)v , i.e., a representation on entity v in the t-th iteration. It should be stressed that h(t)v is still a pair representation, rather than a node representation. By substituting the neural functions into Equation 3 and 4, we get our Neural Bellman-Ford Networks.
h(0)v ← INDICATOR(u, v, q) (5)
h(t)v ← AGGREGATE ({MESSAGE (h(t−1)x, wq(x, r, v)) | (x, r, v) ∈ E(v)} ∪ {h(t)v }) (6)
NBFNet can be interpreted as a novel GNN framework for learning pair representations. Compared to common GNN frameworks [32, 48] that compute the pair representation as two independent node representations hq(u) and hq(v), NBFNet initializes a representation on the source node u, and readouts the pair representation on the target node v. Intuitively, our framework can be viewed as a
現在考慮給定實體 u 和關係 q 的廣義 Bellman-Ford 算法。在此背景下，我們將 h(t)q (u, v) 縮寫為 h(t)v，即第 t 次迭代中實體 v 的表示。需要強調的是，h(t)v 仍然是一個配對表示，而不是節點表示。將神經函數代入方程式 3 和 4，我們得到我們的神經 Bellman-Ford 網絡。
h(0)v ← INDICATOR(u, v, q) (5)
h(t)v ← AGGREGATE ({MESSAGE (h(t−1)x, wq(x, r, v)) | (x, r, v) ∈ E(v)} ∪ {h(t)v }) (6)
NBFNet 可被解釋為一個新穎的學習配對表示的 GNN 框架。與計算配對表示為兩個獨立節點表示 hq(u) 和 hq(v) 的常見 GNN 框架 [32, 48] 相比，NBFNet 在源節點 u 上初始化一個表示，並在目標節點 v 上讀出配對表示。直觀地，我們的框架可以看作是
source-specific message passing process, where every node learns a representation conditioned on the source node. The pseudo code of NBFNet is outlined in Algorithm 1.
一個特定於源點的消息傳遞過程，其中每個節點學習一個以源點為條件的表示。NBFNet 的偽代碼在算法 1 中概述。
**Design Space.** Now we discuss some principled designs for MESSAGE, AGGREGATE and INDICATOR functions by drawing insights from traditional methods. Note the potential design space for NBFNet is way larger than what is presented here, as one can always borrow MESSAGE and AGGREGATE from the arsenal of message-passing GNNs [19, 16, 60, 65].
**設計空間。** 現在我們藉由傳統方法的啟示，討論一些 MESSAGE、AGGREGATE 和 INDICATOR 函數的原則性設計。請注意，NBFNet 的潛在設計空間遠大於此處所呈現的，因為人們總是可以從消息傳遞 GNNs 的武庫中借用 MESSAGE 和 AGGREGATE [19, 16, 60, 65]。
For the MESSAGE function, traditional methods instantiate it as natural summation, natural multiplication or min over scalars. Therefore, we may use the vectorized version of summation or multiplication. Intuitively, summation of h(t-1)x and wq(x, r, v) can be interpreted as a translation of h(t-1)x by wq(x, r, v) in the pair representation space, while multiplication corresponds to scaling. Such transformations correspond to the relational operators [18, 45] in knowledge graph embeddings [6, 68, 58, 31, 52]. For example, translation and scaling are the relational operators used in TransE [6] and DistMult [68] respectively. We also consider the rotation operator in RotatE [52].
對於 MESSAGE 函數，傳統方法將其具現化為自然加法、自然乘法或純量上的最小值。因此，我們可以使用向量化的加法或乘法版本。直觀地，h(t-1)x 和 wq(x, r, v) 的加法可以解釋為在配對表示空間中將 h(t-1)x 平移 wq(x, r, v)，而乘法則對應於縮放。此類轉換對應於知識圖嵌入中的關係運算符 [18, 45] [6, 68, 58, 31, 52]。例如，平移和縮放分別是 TransE [6] 和 DistMult [68] 中使用的關係運算符。我們也考慮了 RotatE [52] 中的旋轉運算符。
The AGGREGATE function is instantiated as natural summation, max or min in traditional methods, which are reminiscent of set aggregation functions [71, 65, 9] used in GNNs. Therefore, we specify the AGGREGATE function to be sum, mean, or max, followed by a linear transformation and a non-linear activation. We also consider the principal neighborhood aggregation (PNA) proposed in a recent work [9], which jointly learns the types and scales of the aggregation function.
AGGREGATE 函數在傳統方法中被實例化為自然求和、最大值或最小值，這讓人聯想到 GNN 中使用的集合聚合函數 [71, 65, 9]。因此，我們將 AGGREGATE 函數指定為求和、平均或最大值，然後進行線性變換和非線性激活。我們也考慮了最近一項工作 [9] 中提出的主鄰域聚合 (PNA)，它聯合學習聚合函數的類型和尺度。
The INDICATOR function is aimed at providing a non-trivial representation for the source node u as the boundary condition. Therefore, we learn a query embedding q for ⊗q, and define INDICATOR function as 1(u = v) ∗ q. Note it is also possible to additionally learn an embedding for ⊕q. However, we find a single query embedding works better in practice.
INDICATOR 函數旨在為源節點 u 提供一個非平凡的表示作為邊界條件。因此，我們為 ⊗q 學習一個查詢嵌入 q，並將 INDICATOR 函數定義為 1(u = v) ∗ q。注意，額外學習一個 ⊕q 的嵌入也是可能的。然而，我們發現在實踐中單個查詢嵌入效果更好。
The edge representations are instantiated as transition probabilities or length in traditional methods. We notice that an edge may have different contribution in answering different query relations. Therefore, we parameterize the edge representations as a linear function over the query relation, i.e., wq(x, r, v) = Wrq + br. For homogeneous graphs or knowledge graphs with very few relations, we simplify the parameterization to wq(x, r, v) = br to prevent overfitting. Note that one may also parameterize wq(x, r, v) with learnable entity embeddings x and v, but such a parameterization cannot solve the inductive setting. Similar to NeuralLP [69] & DRUM [46], we use different edge representations for different iterations, which is able to distinguish noncommutative edges in paths, e.g., father’s mother v.s. mother’s father.
在傳統方法中，邊表示被實例化為轉移機率或長度。我們注意到，一條邊在回答不同查詢關係時可能有不同的貢獻。因此，我們將邊表示參數化為查詢關係的線性函數，即 wq(x, r, v) = Wrq + br。對於同質圖或關係非常少的知識圖，我們將參數化簡化為 wq(x, r, v) = br 以防止過擬合。請注意，也可以用可學習的實體嵌入 x 和 v 來參數化 wq(x, r, v)，但這樣的參數化無法解決歸納設置。與 NeuralLP [69] 和 DRUM [46] 類似，我們對不同的迭代使用不同的邊表示，這能夠區分路徑中的非交換邊，例如，父親的母親 vs. 母親的父親。
**Link Prediction.** We now show how to apply the learned pair representations hq(u, v) to the link prediction problem. We predict the conditional likelihood of the tail entity v as p(v|u, q) = σ(f(hq(u, v))), where σ(·) is the sigmoid function and f(·) is a feed-forward neural network. The conditional likelihood of the head entity u can be predicted by p(u|v, q⁻¹) = σ(f(hq⁻¹(v, u))) with the same model. Following previous works [6, 52], we minimize the negative log-likelihood of positive and negative triplets (Equation 7). The negative samples are generated according to Partial Completeness Assumption (PCA) [14], which corrupts one of the entities in a positive triplet to create a negative sample. For undirected graphs, we symmetrize the representations and define pq(u, v) = σ(f(hq(u, v) + hq(v, u))). Equation 8 shows the loss for homogeneous graphs.
LKG = − log p(u, q, v) − (1/n) Σⁿi=1 log(1 − p(u, q, v'i)) (7)
Lhomo = − log p(u, v) − (1/n) Σⁿi=1 log(1 − p(u, v'i)) (8)
where n is the number of negative samples per positive sample and (u, q, v'i) and (u, v'i) are the i-th negative samples for knowledge graphs and homogeneous graphs, respectively.
**鏈路預測**。我們現在展示如何將學習到的配對表示 hq(u, v) 應用於鏈路預測問題。我們預測尾部實體 v 的條件機率為 p(v|u, q) = σ(f(hq(u, v)))，其中 σ(·) 是 sigmoid 函數，f(·) 是前饋神經網路。頭部實體 u 的條件機率可以透過 p(u|v, q⁻¹) = σ(f(hq⁻¹(v, u))) 使用相同的模型進行預測。遵循先前的工作 [6, 52]，我們最小化正面和負面三元組的負對數概似（方程式 7）。負樣本是根據部分完整性假設 (PCA) [14] 生成的，該假設透過損壞正面三元組中的一個實體來創建負樣本。對於無向圖，我們對稱化表示並定義 pq(u, v) = σ(f(hq(u, v) + hq(v, u)))。方程式 8 顯示了同質圖的損失。
LKG = − log p(u, q, v) − (1/n) Σⁿi=1 log(1 − p(u, q, v'i)) (7)
Lhomo = − log p(u, v) − (1/n) Σⁿi=1 log(1 − p(u, v'i)) (8)
其中 n 是每個正樣本的負樣本數，(u, q, v'i) 和 (u, v'i) 分別是知識圖和同質圖的第 i 個負樣本。
**Time Complexity.** One advantage of NBFNet is that it has a relatively low time complexity during inference⁴. Consider a scenario where a model is required to infer the conditional likelihood of all possible triplets p(v|u, q). We group triplets with the same condition u, q together, where each group contains |V| triplets. For each group, we only need to execute Algorithm 1 once to get their
⁴Although the same analysis can be applied to training on a fixed number of samples, we note it is less instructive since one can trade-off samples for performance, and the trade-off varies from method to method.
**時間複雜度。** NBFNet 的一個優點是在推論期間具有相對較低的時間複雜度⁴。考慮一個場景，其中模型需要推斷所有可能的三元組 p(v|u, q) 的條件機率。我們將具有相同條件 u, q 的三元組分組在一起，其中每個組包含 |V| 個三元組。對於每個組，我們只需要執行一次算法 1 即可獲得它們的
⁴雖然相同的分析可以應用於在固定數量的樣本上進行訓練，但我們注意到這較不具指導性，因為可以在樣本與性能之間進行權衡，且權衡因方法而異。
predictions. Since a small constant number of iterations T is enough for NBFNet to converge (Table 6b), Algorithm 1 has a time complexity of O(|E|d + |V|d²), where d is the dimension of representations. Therefore, the amortized time complexity for a single triplet is O((|E|d/|V|) + d²). For a detailed derivation of time complexity of other GNN frameworks, please refer to Appendix C.
預測。由於 NBFNet 收斂所需的迭代次數 T 是一個小的常數（表 6b），算法 1 的時間複雜度為 O(|E|d + |V|d²)，其中 d 是表示的維度。因此，單個三元組的攤銷時間複雜度為 O((|E|d/|V|) + d²)。有關其他 GNN 框架時間複雜度的詳細推導，請參閱附錄 C。

# 4 實驗

## 4.1 實驗設置
We evaluate NBFNet in three settings, knowledge graph completion, homogeneous graph link prediction and inductive relation prediction. The former two are transductive settings, while the last is an inductive setting. For knowledge graphs, we use FB15k-237 [56] and WN18RR [13]. We use the standard transductive splits [56, 13] and inductive splits [55] of these datasets. For homogeneous graphs, we use Cora, Citeseer and PubMed [49]. Following previous works [32, 12], we split the edges into train/valid/test with a ratio of 85:5:10. Statistics of datasets can be found in Appendix E. Additional experiments of NBFNet on OGB [25] datasets can be found in Appendix G.
我們在三個設定下評估 NBFNet：知識圖補全、同質圖鏈路預測以及歸納關係預測。前兩者為轉導式設定，而後者為歸納式設定。對於知識圖，我們使用 FB15k-237 [56] 和 WN18RR [13]。我們使用這些資料集的標準轉導式分割 [56, 13] 和歸納式分割 [55]。對於同質圖，我們使用 Cora、Citeseer 和 PubMed [49]。遵循先前的工作 [32, 12]，我們將邊以 85:5:10 的比例分割為訓練/驗證/測試集。資料集的統計數據可在附錄 E 中找到。NBFNet 在 OGB [25] 資料集上的額外實驗可在附錄 G 中找到。
**Implementation Details.** Our implementation generally follows the open source codebases of knowledge graph completion⁵ and homogeneous graph link prediction⁶. For knowledge graphs, we follow [69, 46] and augment each triplet (u, q, v) with a flipped triplet (v, q⁻¹, u). For homogeneous graphs, we follow [33, 32] and augment each node u with a self loop (u, u). We instantiate NBFNet with 6 layers, each with 32 hidden units. The feed-forward network f(·) is set to a 2-layer MLP with 64 hidden units. ReLU is used as the activation function for all hidden layers. We drop out edges that directly connect query node pairs during training to encourage the model to capture longer paths and prevent overfitting. Our model is trained on 4 Tesla V100 GPUs for 20 epochs. We select the models based on their performance on the validation set. See Appendix F for more details.
⁵https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding. MIT license.
⁶https://github.com/tkipf/gae. MIT license.
**實現細節。** 我們的實現大致遵循知識圖補全⁵和同質圖鏈路預測⁶的開源代碼庫。對於知識圖，我們遵循 [69, 46] 的做法，並用一個翻轉的三元組 (v, q⁻¹, u) 來增強每個三元組 (u, q, v)。對於同質圖，我們遵循 [33, 32] 的做法，並用一個自環 (u, u) 來增強每個節點 u。我們用 6 個層實例化 NBFNet，每層有 32 個隱藏單元。前饋網絡 f(·) 設置為一個具有 64 個隱藏單元的 2 層 MLP。ReLU 被用作所有隱藏層的激活函數。我們在訓練期間丟棄直接連接查詢節點對的邊，以鼓勵模型捕捉更長的路徑並防止過擬合。我們的模型在 4 個 Tesla V100 GPU 上訓練 20 個週期。我們根據模型在驗證集上的表現來選擇模型。更多細節請參見附錄 F。
⁵https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding. MIT 授權。
⁶https://github.com/tkipf/gae. MIT 授權。
**Evaluation.** We follow the filtered ranking protocol [6] for knowledge graph completion. For a test triplet (u, q, v), we rank it against all negative triplets ⟨u, q, v'⟩ or ⟨u', q, v⟩ that do not appear in the knowledge graph. We report mean rank (MR), mean reciprocal rank (MRR) and HITS at N (H@N) for knowledge graph completion. For inductive relation prediction, we follow [55] and draw 50 negative triplets for each positive triplet and use the above filtered ranking. We report HITS@10 for inductive relation prediction. For homogeneous graph link prediction, we follow [32] and compare the positive edges against the same number of negative edges. We report area under the receiver operating characteristic curve (AUROC) and average precision (AP) for homogeneous graphs.
**評估。** 我們遵循知識圖補全的過濾排名協議 [6]。對於一個測試三元組 (u, q, v)，我們將其與所有未出現在知識圖中的負三元組 ⟨u, q, v'⟩ 或 ⟨u', q, v⟩ 進行排名。我們報告知識圖補全的平均排名 (MR)、平均倒數排名 (MRR) 和 HITS at N (H@N)。對於歸納關係預測，我們遵循 [55] 的方法，為每個正三元組抽取 50 個負三元組，並使用上述的過濾排名。我們報告歸納關係預測的 HITS@10。對於同構圖鏈路預測，我們遵循 [32] 的方法，將正邊緣與相同數量的負邊緣進行比較。我們報告接收者操作特徵曲線下面積 (AUROC) 和平均精度 (AP) 用於同構圖。
**Baselines.** We compare NBFNet against path-based methods, embedding methods, and GNNs. These include 11 baselines for knowledge graph completion, 10 baselines for homogeneous graph link prediction and 4 baselines for inductive relation prediction. Note the inductive setting only includes path-based methods and GNNs, since existing embedding methods cannot handle this setting.
**基準線。** 我們將 NBFNet 與基於路徑的方法、嵌入方法和 GNN 進行比較。其中包括 11 個知識圖補全的基準線、10 個同質圖鏈路預測的基準線和 4 個歸納關係預測的基準線。請注意，歸納設置僅包括基於路徑的方法和 GNN，因為現有的嵌入方法無法處理此設置。

## 4.2 主要結果
Table 3 summarizes the results on knowledge graph completion. NBFNet significantly outperforms existing methods on all metrics and both datasets. NBFNet achieves an average relative gain of 21% in HITS@1 compared to the best path-based method, DRUM [46], on two datasets. Since DRUM is a special instance of NBFNet with natural summation and multiplication operators, this indicates the importance of learning MESSAGE and AGGREGATE functions in NBFNet. NBFNet also outperforms the best embedding method, LowFER [1], with an average relative performance gain of 18% in HITS@1 on two datasets. Meanwhile, NBFNet requires much less parameters than embedding methods. NBFNet only uses 3M parameters on FB15k-237, while TransE needs 30M parameters. See Appendix D for details on the number of parameters.
表 3 總結了知識圖補全的結果。NBFNet 在所有指標和兩個數據集上都顯著優於現有方法。與最佳的基於路徑的方法 DRUM [46] 相比，NBFNet 在兩個數據集上的 HITS@1 取得了平均 21% 的相對增益。由於 DRUM 是 NBFNet 使用自然求和與乘法運算符的一個特例，這表明在 NBFNet 中學習 MESSAGE 和 AGGREGATE 函數的重要性。NBFNet 也優於最佳的嵌入方法 LowFER [1]，在兩個數據集上的 HITS@1 取得了平均 18% 的相對性能增益。同時，NBFNet 需要的參數遠少於嵌入方法。NBFNet 在 FB15k-237 上僅使用 3M 參數，而 TransE 需要 30M 參數。有關參數數量的詳細信息，請參閱附錄 D。
Table 4 shows the results on homogeneous graph link prediction. NBFNet gets the best results on Cora and PubMed, meanwhile achieves competitive results on CiteSeer. Note CiteSeer is extremely sparse (Appendix E), which makes it hard to learn good representations with NBFNet. One thing to note here is that unlike other GNN methods, NBFNet does not use the node features provided by
表 4 顯示了同構圖鏈路預測的結果。NBFNet 在 Cora 和 PubMed 上取得了最佳結果，同時在 CiteSeer 上也取得了有競爭力的結果。值得注意的是，CiteSeer 非常稀疏（附錄 E），這使得用 NBFNet 學習好的表示變得困難。這裡需要注意的一點是，與其他 GNN 方法不同，NBFNet 並不使用數據集提供的節點特徵，但仍然能夠超越大多數其他方法。我們將如何有效結合節點特徵和結構表示以進行鏈路預測作為我們未來的研究工作。
Table 5 summarizes the results on inductive relation prediction. On all inductive splits of two datasets, NBFNet achieves the best result. NBFNet outperforms the previous best method, GraIL [55], with an average relative performance gain of 22% in HITS@10. Note that GraIL explicitly encodes the local subgraph surrounding each node pair and has a high time complexity (Appendix C). Usually, GraIL can at most encode a 2-hop subgraph, while our NBFNet can efficiently explore longer paths.
表 5 總結了歸納關係預測的結果。在兩個數據集的所有歸納分割上，NBFNet 均取得了最佳結果。NBFNet 超越了先前的最佳方法 GraIL [55]，在 HITS@10 上平均相對性能提升了 22%。值得注意的是，GraIL 明確編碼了每個節點對周圍的局部子圖，且時間複雜度高（附錄 C）。通常，GraIL 最多只能編碼一個 2-hop 的子圖，而我們的 NBFNet 可以有效地探索更長的路徑。
*Table 3: Knowledge graph completion results. Results of NeuraLP and DRUM are taken from [46]. Results of RotatE, HAKE and LowFER are taken from their original papers [52, 76, 1]. Results of the other embedding methods are taken from [52]. Since GraIL has scalability issues in this setting, we evaluate it with 50 and 100 negative triplets for FB15k-237 and WN18RR respectively and report MR based on an unbiased estimation.*

| Class | Method | FB15k-237 | WN18RR |
| :--- | :--- | :--- | :--- |
| | | MR | MRR | H@1 | H@3 | H@10 | MR | MRR | H@1 | H@3 | H@10 |
| Path-based | Path Ranking [35] | 3521 | 0.174 | 0.119 | 0.186 | 0.285 | 22438 | 0.324 | 0.276 | 0.360 | 0.406 |
| | NeuralLP [69] | - | 0.240 | - | 0.362 | - | - | 0.435 | 0.371 | 0.434 | 0.566 |
| | DRUM [46] | - | 0.343 | 0.255 | 0.378 | 0.516 | - | 0.486 | 0.425 | 0.513 | 0.586 |
| Embeddings | TransE [6] | 357 | 0.294 | - | - | 0.465 | 3384 | 0.226 | - | - | 0.501 |
| | DistMult [68] | 254 | 0.241 | 0.155 | 0.263 | 0.419 | 5110 | 0.43 | 0.39 | 0.44 | 0.49 |
| | ComplEx [58] | 339 | 0.247 | 0.158 | 0.275 | 0.428 | 5261 | 0.44 | 0.41 | 0.46 | 0.51 |
| | RotatE [52] | 177 | 0.338 | 0.241 | 0.375 | 0.553 | 3340 | 0.476 | 0.428 | 0.492 | 0.571 |
| | HAKE [76] | - | 0.346 | 0.250 | 0.381 | 0.542 | - | 0.497 | 0.452 | 0.516 | 0.582 |
| | LowFER [1] | - | 0.359 | 0.266 | 0.396 | 0.544 | - | 0.465 | 0.434 | 0.479 | 0.526 |
| GNNs | RGCN [48] | 221 | 0.273 | 0.182 | 0.303 | 0.456 | 2719 | 0.402 | 0.345 | 0.437 | 0.494 |
| | GraIL [55] | 2053 | - | - | - | - | 2539 | - | - | - | - |
| | NBFNet | 114 | 0.415 | 0.321 | 0.454 | 0.599 | 636 | 0.551 | 0.497 | 0.573 | 0.666 |
*表 3：知識圖譜完成結果。NeuraLP 和 DRUM 的結果取自 [46]。RotatE、HAKE 和 LowFER 的結果取自其原始論文 [52, 76, 1]。其他嵌入方法的結果取自 [52]。由於 GraIL 在此設置中存在可擴展性問題，我們分別使用 50 和 100 個負三元組對 FB15k-237 和 WN18RR 進行評估，並根據無偏估計報告 MR。*

| 類別 | 方法 | FB15k-237 | WN18RR |
| :--- | :--- | :--- | :--- |
| | | MR | MRR | H@1 | H@3 | H@10 | MR | MRR | H@1 | H@3 | H@10 |
| 基於路徑 | 路徑排名 [35] | 3521 | 0.174 | 0.119 | 0.186 | 0.285 | 22438 | 0.324 | 0.276 | 0.360 | 0.406 |
| | NeuralLP [69] | - | 0.240 | - | 0.362 | - | - | 0.435 | 0.371 | 0.434 | 0.566 |
| | DRUM [46] | - | 0.343 | 0.255 | 0.378 | 0.516 | - | 0.486 | 0.425 | 0.513 | 0.586 |
| 嵌入 | TransE [6] | 357 | 0.294 | - | - | 0.465 | 3384 | 0.226 | - | - | 0.501 |
| | DistMult [68] | 254 | 0.241 | 0.155 | 0.263 | 0.419 | 5110 | 0.43 | 0.39 | 0.44 | 0.49 |
| | ComplEx [58] | 339 | 0.247 | 0.158 | 0.275 | 0.428 | 5261 | 0.44 | 0.41 | 0.46 | 0.51 |
| | RotatE [52] | 177 | 0.338 | 0.241 | 0.375 | 0.553 | 3340 | 0.476 | 0.428 | 0.492 | 0.571 |
| | HAKE [76] | - | 0.346 | 0.250 | 0.381 | 0.542 | - | 0.497 | 0.452 | 0.516 | 0.582 |
| | LowFER [1] | - | 0.359 | 0.266 | 0.396 | 0.544 | - | 0.465 | 0.434 | 0.479 | 0.526 |
| GNNs | RGCN [48] | 221 | 0.273 | 0.182 | 0.303 | 0.456 | 2719 | 0.402 | 0.345 | 0.437 | 0.494 |
| | GraIL [55] | 2053 | - | - | - | - | 2539 | - | - | - | - |
| | NBFNet | 114 | 0.415 | 0.321 | 0.454 | 0.599 | 636 | 0.551 | 0.497 | 0.573 | 0.666 |
*Table 4: Homogeneous graph link prediction results. Results of VGAE and S-VGAE are taken from their original papers [32, 12].*

| Class | Method | Cora | Citeseer | PubMed |
| :--- | :--- | :--- | :--- | :--- |
| | | AUROC | AP | AUROC | AP | AUROC | AP |
| Path-based | Katz Index [30] | 0.834 | 0.889 | 0.768 | 0.810 | 0.757 | 0.856 |
| | Personalized PageRank [42] | 0.845 | 0.899 | 0.762 | 0.814 | 0.763 | 0.860 |
| | SimRank [28] | 0.838 | 0.888 | 0.755 | 0.805 | 0.743 | 0.829 |
| Embeddings | DeepWalk [43] | 0.831 | 0.850 | 0.805 | 0.838 | 0.844 | 0.841 |
| | LINE [53] | 0.844 | 0.876 | 0.791 | 0.836 | 0.849 | 0.888 |
| | node2vec [17] | 0.872 | 0.879 | 0.826 | 0.868 | 0.891 | 0.914 |
| GNNs | VGAE [32] | 0.914 | 0.926 | 0.908 | 0.920 | 0.944 | 0.947 |
| | S-VGAE [12] | 0.941 | 0.941 | 0.947 | 0.952 | 0.960 | 0.960 |
| | SEAL [73] | 0.933 | 0.942 | 0.905 | 0.924 | 0.978 | 0.979 |
| | TLC-GNN [67] | 0.934 | 0.931 | 0.909 | 0.916 | 0.970 | 0.968 |
| | NBFNet | 0.956 | 0.962 | 0.923 | 0.936 | 0.983 | 0.982 |
*表 4：同構圖鏈路預測結果。VGAE 和 S-VGAE 的結果取自其原始論文 [32, 12]。*

| 類別 | 方法 | Cora | Citeseer | PubMed |
| :--- | :--- | :--- | :--- | :--- |
| | | AUROC | AP | AUROC | AP | AUROC | AP |
| 基於路徑 | Katz 指數 [30] | 0.834 | 0.889 | 0.768 | 0.810 | 0.757 | 0.856 |
| | 個人化 PageRank [42] | 0.845 | 0.899 | 0.762 | 0.814 | 0.763 | 0.860 |
| | SimRank [28] | 0.838 | 0.888 | 0.755 | 0.805 | 0.743 | 0.829 |
| 嵌入 | DeepWalk [43] | 0.831 | 0.850 | 0.805 | 0.838 | 0.844 | 0.841 |
| | LINE [53] | 0.844 | 0.876 | 0.791 | 0.836 | 0.849 | 0.888 |
| | node2vec [17] | 0.872 | 0.879 | 0.826 | 0.868 | 0.891 | 0.914 |
| GNNs | VGAE [32] | 0.914 | 0.926 | 0.908 | 0.920 | 0.944 | 0.947 |
| | S-VGAE [12] | 0.941 | 0.941 | 0.947 | 0.952 | 0.960 | 0.960 |
| | SEAL [73] | 0.933 | 0.942 | 0.905 | 0.924 | 0.978 | 0.979 |
| | TLC-GNN [67] | 0.934 | 0.931 | 0.909 | 0.916 | 0.970 | 0.968 |
| | NBFNet | 0.956 | 0.962 | 0.923 | 0.936 | 0.983 | 0.982 |
*Table 5: Inductive relation prediction results (HITS@10). v1-v4 corresponds to the 4 standard versions of inductive splits. Results of compared methods are taken from [55].*

| Class | Method | FB15k-237 | WN18RR |
| :--- | :--- | :--- | :--- |
| | | v1 | v2 | v3 | v4 | v1 | v2 | v3 | v4 |
| Path-based | NeuralLP [16] | 0.529 | 0.589 | 0.529 | 0.559 | 0.744 | 0.689 | 0.462 | 0.671 |
| | DRUM [46] | 0.529 | 0.587 | 0.529 | 0.559 | 0.744 | 0.689 | 0.462 | 0.671 |
| | RuleN [39] | 0.498 | 0.778 | 0.877 | 0.856 | 0.809 | 0.782 | 0.534 | 0.716 |
| GNNs | GraIL [55] | 0.642 | 0.818 | 0.828 | 0.893 | 0.825 | 0.787 | 0.584 | 0.734 |
| | NBFNet | 0.834 | 0.949 | 0.951 | 0.960 | 0.948 | 0.905 | 0.893 | 0.890 |
*表 5：歸納關係預測結果 (HITS@10)。v1-v4 對應於 4 個標準版本的歸納分割。比較方法的結果取自 [55]。*

| 類別 | 方法 | FB15k-237 | WN18RR |
| :--- | :--- | :--- | :--- |
| | | v1 | v2 | v3 | v4 | v1 | v2 | v3 | v4 |
| 基於路徑 | NeuralLP [16] | 0.529 | 0.589 | 0.529 | 0.559 | 0.744 | 0.689 | 0.462 | 0.671 |
| | DRUM [46] | 0.529 | 0.587 | 0.529 | 0.559 | 0.744 | 0.689 | 0.462 | 0.671 |
| | RuleN [39] | 0.498 | 0.778 | 0.877 | 0.856 | 0.809 | 0.782 | 0.534 | 0.716 |
| GNNs | GraIL [55] | 0.642 | 0.818 | 0.828 | 0.893 | 0.825 | 0.787 | 0.584 | 0.734 |
| | NBFNet | 0.834 | 0.949 | 0.951 | 0.960 | 0.948 | 0.905 | 0.893 | 0.890 |

## 4.3 消融研究
**MESSAGE & AGGREGATE Functions.** Table 6a shows the results of different MESSAGE and AGGREGATE functions. Generally, NBFNet benefits from advanced embedding methods (DistMult, RotatE > TransE) and aggregation functions (PNA > sum, mean, max). Among simple AGGREGATE functions (sum, mean, max), combinations of MESSAGE and AGGREGATE functions (TransE & max, DistMult & sum) that satisfy the semiring assumption⁷ of the generalized Bellman-Ford algorithm, achieve locally optimal performance. PNA significantly improves over simple counterparts, which highlights the importance of learning more powerful AGGREGATE functions.
**MESSAGE 和 AGGREGATE 函數。** 表 6a 顯示了不同 MESSAGE 和 AGGREGATE 函數的結果。總體而言，NBFNet 受益於先進的嵌入方法（DistMult、RotatE > TransE）和聚合函數（PNA > sum、mean、max）。在簡單的 AGGREGATE 函數（sum、mean、max）中，滿足廣義 Bellman-Ford 算法半環假設⁷ 的 MESSAGE 和 AGGREGATE 函數組合（TransE & max、DistMult & sum）達到了局部最優性能。PNA 顯著優於簡單的對應函數，這凸顯了學習更強大 AGGREGATE 函數的重要性。
**Number of GNN Layers.** Table 6b compares the results of NBFNet with different number of layers. Although it has been reported that GNNs with deep layers often result in significant performance drop [36, 77], we observe NBFNet does not have this issue. The performance increases monotonically with more layers, hitting a saturation after 6 layers. We conjecture the reason is that longer paths have negligible contribution, and paths not longer than 6 are enough for link prediction.
**GNN 層數。** 表格 6b 比較了不同層數的 NBFNet 結果。雖然已有報導指出，具有深層的 GNNs 通常會導致顯著的性能下降 [36, 77]，但我們觀察到 NBFNet 沒有這個問題。隨著層數的增加，性能單調提升，在 6 層後達到飽和。我們推測其原因是較長路徑的貢獻可以忽略不計，而長度不超過 6 的路徑對於鏈路預測已經足夠。
**Performance by Relation Category.** We break down the performance of NBFNet by the categories of query relations: one-to-one, one-to-many, many-to-one and many-to-many⁸. Table 6c shows the prediction results for each category. It is observed that NBFNet not only improves on easy one-to-one cases, but also on hard cases where there are multiple true answers for the query.
*Table 6: Ablation studies of NBFNet on FB15k-237. Due to space constraints, we only report MRR here. For full results on all metrics, please refer to Appendix H.*
**按關係類別劃分的性能。** 我們將 NBFNet 的性能按查詢關係的類別進行分解：一對一、一對多、多對一和多對多⁸。表 6c 顯示了每個類別的預測結果。觀察到 NBFNet 不僅在簡單的一對一案例中有所改進，而且在查詢有多個真實答案的困難案例中也有所改進。
*表 6：NBFNet 在 FB15k-237 上的消融研究。由於空間限制，我們這裡僅報告 MRR。所有指標的完整結果請參考附錄 H。*
(a) Different MESSAGE and AGGREGATE functions.

| MESSAGE | AGGREGATE | Sum | Mean | Max | PNA [9] |
| :--- | :--- | :--- | :--- | :--- | :--- |
| TransE [6] | | 0.297 | 0.310 | 0.377 | 0.383 |
| DistMult [69] | | 0.388 | 0.384 | 0.374 | 0.415 |
| RotatE [52] | | 0.392 | 0.376 | 0.385 | 0.414 |

(b) Different number of layers.

| Method | #Layers (T) | 2 | 4 | 6 | 8 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| NBFNet | | 0.345 | 0.409 | 0.415 | 0.416 |

(c) Performance w.r.t. relation category. The two scores are the rankings over heads and tails respectively.

| Method | Relation Category | 1-to-1 | 1-to-N | N-to-1 | N-to-N |
| :--- | :--- | :--- | :--- | :--- | :--- |
| TransE [6] | | 0.498/0.488 | 0.455/0.071 | 0.079/0.744 | 0.224/0.330 |
| RotatE [51] | | 0.487/0.484 | 0.467/0.070 | 0.081/0.747 | 0.234/0.338 |
| NBFNet | | 0.578/0.600 | 0.499/0.122 | 0.165/0.790 | 0.348/0.456 |
(a) 不同的 MESSAGE 和 AGGREGATE 函數。

| MESSAGE | AGGREGATE | Sum | Mean | Max | PNA [9] |
| :--- | :--- | :--- | :--- | :--- | :--- |
| TransE [6] | | 0.297 | 0.310 | 0.377 | 0.383 |
| DistMult [69] | | 0.388 | 0.384 | 0.374 | 0.415 |
| RotatE [52] | | 0.392 | 0.376 | 0.385 | 0.414 |

(b) 不同層數。

| 方法     | #層數 (T) | 2     | 4     | 6     | 8     |
| :----- | :------ | :---- | :---- | :---- | :---- |
| NBFNet |         | 0.345 | 0.409 | 0.415 | 0.416 |

(c) 關於關係類別的性能。這兩個分數分別是頭部和尾部的排名。

| 方法 | 關係類別 | 1對1 | 1對N | N對1 | N對N |
| :--- | :--- | :--- | :--- | :--- | :--- |
| TransE [6] | | 0.498/0.488 | 0.455/0.071 | 0.079/0.744 | 0.224/0.330 |
| RotatE [51] | | 0.487/0.484 | 0.467/0.070 | 0.081/0.747 | 0.234/0.338 |
| NBFNet | | 0.578/0.600 | 0.499/0.122 | 0.165/0.790 | 0.348/0.456 |
⁷Here semiring is discussed under the assumption of linear activation functions. Rigorously, no combination satisfies a semiring if we consider non-linearity in the model.
⁸The categories are defined same as [63]. We compute the average number of tails per head and the average number of heads per tail. The category is one if the average number is smaller than 1.5 and many otherwise.
⁷這裡的半環是在線性激活函數的假設下討論的。嚴格來說，如果我們考慮模型中的非線性，沒有任何組合能滿足半環的條件。
⁸類別的定義與 [63] 相同。我們計算每個頭的平均尾數和每個尾的平均頭數。如果平均數小於 1.5，則類別為「一」，否則為「多」。

## 4.4 預測的路徑解釋
One advantage of NBFNet is that we can interpret its predictions through paths, which may be important for users to understand and debug the model. Intuitively, the interpretations should contain paths that contribute most to the prediction p(u, q, v). Following local interpretation methods [3, 72], we approximate the local landscape of NBFNet with a linear model over the set of all paths, i.e., 1st-order Taylor polynomial. We define the importance of a path as its weight in the linear model, which can be computed by the partial derivative of the prediction w.r.t. the path. Formally, the top-k path interpretations for p(u, q, v) are defined as
P₁, P₂, . . . , Pk = top-k P∈Puv (∂p(u, q, v)/∂P) (9)
Note this formulation generalizes the definition of logical rules [69, 46] to non-linear models. While directly computing the importance of all paths is intractable, we approximate them with edge importance. Specifically, the importance of each path is approximated by the sum of the importance of edges in that path, where edge importance is obtained via auto differentiation. Then the top-k path interpretations are equivalent to the top-k longest paths on the edge importance graph, which can be solved by a Bellman-Ford-style beam search. Better approximation is left as a future work.
NBFNet 的一個優點是，我們可以通過路徑來解釋其預測，這對於用戶理解和調試模型可能很重要。直觀地，解釋應包含對預測 p(u, q, v) 貢獻最大的路徑。遵循局部解釋方法 [3, 72]，我們用一個在所有路徑集合上的線性模型，即一階泰勒多項式，來近似 NBFNet 的局部景觀。我們將路徑的重要性定義為它在線性模型中的權重，這可以通過預測對路徑的偏導數來計算。形式上，p(u, q, v) 的前 k 個路徑解釋定義為
P₁, P₂, . . . , Pk = top-k P∈Puv (∂p(u, q, v)/∂P) (9)
注意，此公式將邏輯規則的定義 [69, 46] 推廣到非線性模型。雖然直接計算所有路徑的重要性是棘手的，但我們用邊的重要性來近似它們。具體來說，每條路徑的重要性由該路徑中邊重要性的總和來近似，其中邊重要性是通過自動微分獲得的。然後，前 k 個路徑解釋等同於邊重要性圖上的前 k 條最長路徑，這可以通過類 Bellman-Ford 的束搜索來解決。更好的近似留待未來工作。
Table 7 visualizes path interpretations from FB15k-237 test set. While users may have different insights towards the visualization, here is our understanding. 1) In the first example, NBFNet learns soft logical entailment, such as impersonate⁻¹ ∧ nationality ⇒ nationality and ethnicity⁻¹ ∧ distribution ⇒ nationality. 2) In second example, NBFNet performs analogical reasoning by leveraging the fact that Florence is similar to Rome. 3) In the last example, NBFNet extracts longer paths, since there is no obvious connection between Pearl Harbor (film) and Japanese language.
表 7 視覺化了來自 FB15k-237 測試集的路徑解釋。雖然使用者對視覺化可能有不同的見解，但我們的理解如下。1) 在第一個例子中，NBFNet 學習了軟邏輯蘊涵，例如 impersonate⁻¹ ∧ nationality ⇒ nationality 和 ethnicity⁻¹ ∧ distribution ⇒ nationality。2) 在第二個例子中，NBFNet 通過利用佛羅倫斯與羅馬相似的事實來進行類比推理。3) 在最後一個例子中，NBFNet 提取了更長的路徑，因為珍珠港（電影）和日語之間沒有明顯的聯繫。
*Table 7: Path interpretations of predictions on FB15k-237 test set. For each query triplet, we visualize the top-2 path interpretations and their weights. Inverse relations are denoted with a superscript ⁻¹.*
**Query** (u, q, v): (O. Hardy, nationality, U.S.)
0.243 (O. Hardy, impersonate⁻¹, R. Little) ∧ (R. Little, nationality, U.S.)
0.224 (O. Hardy, ethnicity⁻¹, Scottish American) ∧ (Scottish American, distribution, U.S.)
**Query** (u, q, v): (Florence, vacationer, D.C. Henrie)
0.251 (Florence, contain⁻¹, Italy) ∧ (Italy, capital, Rome) ∧ (Rome, vacationer, D.C. Henrie)
0.183 (Florence, place live⁻¹, G.F. Handel) ∧ (G.F. Handel, place live, Rome) ∧ (Rome, vacationer, D.C. Henrie)
**Query** (u, q, v): (Pearl Harbor (film), language, Japanese)
0.211 (Pearl Harbor (film), film actor, C.-H. Tagawa) ∧ (C.-H. Tagawa, nationality, Japan) ∧ <Japan, country of origin, Yu-Gi-Oh!> ∧ <Yu-Gi-Oh!, language, Japanese>
0.208 (Pearl Harbor (film), film actor, C.-H. Tagawa) ∧ (C.-H. Tagawa, nationality, Japan) ∧ (Japan, official language, Japanese)
*表 7：FB15k-237 測試集上預測的路徑解釋。對於每個查詢三元組，我們將前 2 個路徑解釋及其權重視覺化。逆關係用上標 ⁻¹ 表示。*
**查詢** (u, q, v): (O. Hardy, 國籍, 美國)
0.243 (O. Hardy, 模仿⁻¹, R. Little) ∧ (R. Little, 國籍, 美國)
0.224 (O. Hardy, 族裔⁻¹, 蘇格蘭裔美國人) ∧ (蘇格蘭裔美國人, 分佈, 美國)
**查詢** (u, q, v): (佛羅倫斯, 度假者, D.C. Henrie)
0.251 (佛羅倫斯, 包含⁻¹, 義大利) ∧ (義大利, 首都, 羅馬) ∧ (羅馬, 度假者, D.C. Henrie)
0.183 (佛羅倫斯, 居住地⁻¹, G.F. Handel) ∧ (G.F. Handel, 居住地, 羅馬) ∧ (羅馬, 度假者, D.C. Henrie)
**查詢** (u, q, v): (珍珠港 (電影), 語言, 日語)
0.211 (珍珠港 (電影), 電影演員, C.-H. Tagawa) ∧ (C.-H. Tagawa, 國籍, 日本) ∧ <日本, 原產國, 遊戲王!> ∧ <遊戲王!, 語言, 日語>
0.208 (珍珠港 (電影), 電影演員, C.-H. Tagawa) ∧ (C.-H. Tagawa, 國籍, 日本) ∧ (日本, 官方語言, 日語)

# 5 討論與結論
**Limitations.** There are a few limitations for NBFNet. First, the assumption of the generalized Bellman-Ford algorithm requires the operators (⊕, ⊗) to satisfy a semiring. Due to the non-linear activation functions in neural networks, this assumption does not hold for NBFNet, and we do not have a theoretical guarantee on the loss incurred by this relaxation. Second, NBFNet is only verified on simple edge prediction, while there are other link prediction variants, e.g., complex logical queries with conjunctions (∧) and disjunctions (∨) [18, 45]. In the future, we would like to how NBFNet approximates the path formulation, as well as apply NBFNet to other link prediction settings.
**限制。** NBFNet 有幾個限制。首先，廣義 Bellman-Ford 算法的假設要求運算符 (⊕, ⊗) 滿足半環的性質。由於神經網絡中的非線性激活函數，這個假設對 NBFNet 不成立，我們對這種鬆弛所造成的損失沒有理論保證。其次，NBFNet 僅在簡單的邊緣預測上進行了驗證，而還有其他鏈路預測的變體，例如，帶有合取 (∧) 和析取 (∨) 的複雜邏輯查詢 [18, 45]。未來，我們希望能了解 NBFNet 如何近似路徑公式，以及將 NBFNet 應用到其他的鏈路預測設定。
**Social Impacts.** Link prediction has a wide range of beneficial applications, including recommender systems, knowledge graph completion and drug repurposing. However, there are also some potentially negative impacts. First, NBFNet may encode the bias present in the training data, which leads to stereotyped predictions when the prediction is applied to a user on a social or e-commerce platform. Second, some harmful network activities could be augmented by powerful link prediction models, e.g., spamming, phishing, and social engineering. We expect future studies will mitigate these issues.
**社會影響。** 鏈路預測具有廣泛的有益應用，包括推薦系統、知識圖補全和藥物再利用。然而，也存在一些潛在的負面影響。首先，NBFNet 可能會編碼訓練數據中存在的偏見，當預測應用於社交或電子商務平台的用戶時，會導致刻板印象的預測。其次，一些有害的網絡活動可能會被強大的鏈路預測模型增強，例如，垃圾郵件、釣魚和社交工程。我們期望未來的研究將會緩解這些問題。
**Conclusion.** We present a representation learning framework based on paths for link prediction. Our path formulation generalizes several traditional methods, and can be efficiently solved via the generalized Bellman-Ford algorithm. To improve the capacity of the path formulation, we propose NBFNet, which parameterizes the generalized Bellman-Ford algorithm with learned INDICATOR, MESSAGE, AGGREGATE functions. Experiments on knowledge graphs and homogeneous graphs show that NBFNet outperforms a wide range of methods in both transductive and inductive settings.
**結論。** 我們提出一個基於路徑的鏈路預測表示學習框架。我們的路徑公式推廣了幾種傳統方法，並且可以通過廣義 Bellman-Ford 算法有效求解。為了提高路徑公式的能力，我們提出了 NBFNet，它用學習到的 INDICATOR、MESSAGE、AGGREGATE 函數來參數化廣義 Bellman-Ford 算法。在知識圖和同質圖上的實驗表明，NBFNet 在轉導和歸納設置中均優於多種方法。
**Acknowledgements**
We would like to thank Komal Teru for discussion on inductive relation prediction, Guyue Huang for discussion on fused message passing implementation, and Yao Lu for assistance on large-scale GPU training. We thank Meng Qu, Chence Shi and Minghao Xu for providing feedback on our manuscript.
This project is supported by the Natural Sciences and Engineering Research Council (NSERC) Discovery Grant, the Canada CIFAR AI Chair Program, collaboration grants between Microsoft Research and Mila, Samsung Electronics Co., Ltd., Amazon Faculty Research Award, Tencent AI Lab Rhino-Bird Gift Fund and a NRC Collaborative R&D Project (AI4D-CORE-06). This project was also partially funded by IVADO Fundamental Research Project grant PRF-2019-3583139727. The computation resource of this project is supported by Calcul Québec⁹ and Compute Canada¹⁰.
⁹https://www.calculquebec.ca/
¹⁰https://www.computecanada.ca/
**致謝**
我們感謝 Komal Teru 關於歸納關係預測的討論，感謝 Guyue Huang 關於融合消息傳遞實現的討論，以及感謝 Yao Lu 在大規模 GPU 訓練上的協助。我們感謝 Meng Qu、Chence Shi 和 Minghao Xu 對我們的文稿提供的反饋。
該項目由加拿大自然科學與工程研究委員會 (NSERC) 探索補助金、加拿大 CIFAR 人工智能主席計畫、微軟研究院與 Mila 的合作補助金、三星電子有限公司、亞馬遜教員研究獎、騰訊人工智能實驗室犀牛鳥禮物基金以及一個 NRC 合作研發項目 (AI4D-CORE-06) 支持。該項目也部分由 IVADO 基礎研究項目補助金 PRF-2019-3583139727 資助。該項目的計算資源由 Calcul Québec⁹ 和 Compute Canada¹⁰ 支持。
⁹https://www.calculquebec.ca/
¹⁰https://www.computecanada.ca/