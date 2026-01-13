---
title: TLogic
field: knowledge graph
status: Imported
created_date: 2026-01-13
pdf_link: "[[TLogic.pdf]]"
tags:
  - paper
  - knowledge_graph
---
#knowledge_graph
# TLogic: Temporal Logical Rules for Explainable Link Forecasting on Temporal Knowledge Graphs

# TLogic：用於可解釋性時間知識圖譜連結預測的時序邏輯規則

Yushan Liu ¹², Yunpu Ma¹², Marcel Hildebrandt¹, Mitchell Joblin¹, Volker Tresp ¹,²
¹Siemens AG, Otto-Hahn-Ring 6, 81739 Munich, Germany
{firstname.lastname}@siemens.com
²Ludwig Maximilian University of Munich, Geschwister-Scholl-Platz 1, 80539 Munich, Germany

Yushan Liu¹², Yunpu Ma¹², Marcel Hildebrandt¹, Mitchell Joblin¹, Volker Tresp ¹,²
¹西門子股份公司，Otto-Hahn-Ring 6, 81739 慕尼黑，德國
{firstname.lastname}@siemens.com
²慕尼黑大學，Geschwister-Scholl-Platz 1, 80539 慕尼黑，德國

## Abstract

Conventional static knowledge graphs model entities in relational data as nodes, connected by edges of specific relation types. However, information and knowledge evolve continuously, and temporal dynamics emerge, which are expected to influence future situations. In temporal knowledge graphs, time information is integrated into the graph by equipping each edge with a timestamp or a time range. Embedding-based methods have been introduced for link prediction on temporal knowledge graphs, but they mostly lack explainability and comprehensible reasoning chains. Particularly, they are usually not designed to deal with link forecasting – event prediction involving future timestamps. We address the task of link forecasting on temporal knowledge graphs and introduce TLogic, an explainable framework that is based on temporal logical rules extracted via temporal random walks. We compare TLogic with state-of-the-art baselines on three benchmark datasets and show better overall performance while our method also provides explanations that preserve time consistency. Furthermore, in contrast to most state-of-the-art embedding-based methods, TLogic works well in the inductive setting where already learned rules are transferred to related datasets with a common vocabulary.

## 摘要

傳統的靜態知識圖譜將關聯性資料中的實體塑造成節點，並由特定關係類型的邊連接。然而，資訊和知識不斷演變，時間動態也隨之出現，並預期會影響未來情勢。在時間知識圖譜中，時間資訊透過為每條邊配備時間戳或時間範圍來整合到圖譜中。雖然已引入基於嵌入的方法來進行時間知識圖譜的連結預測，但它們大多缺乏可解釋性和易於理解的推理鏈。特別是，這些方法通常並非設計來處理連結預測——即涉及未來時間戳的事件預測。我們針對時間知識圖譜上的連結預測任務，引入了一個名為 TLogic 的可解釋性框架，該框架基於透過時間隨機漫步提取的時序邏輯規則。我們將 TLogic 與三種基準資料集上的最新基準模型進行比較，結果顯示其整體性能更佳，同時我們的方法也提供了能保持時間一致性的解釋。此外，與大多數最新的基於嵌入的方法不同，TLogic 在歸納設定中表現良好，其中已學習的規則可被轉移到具有共通詞彙的相關資料集。

[Image]

Figure 1: A subgraph from the dataset ICEWS14 with the entities Angela Merkel, Barack Obama, France, and China. The timestamps are displayed in the format yy/mm/dd. The dotted blue line represents the correct answer to the query (Angela Merkel, consult, ?, 2014/08/09). Previous interactions between Angela Merkel and Barack Obama can be interpreted as an explanation for the prediction.

圖 1：資料集 ICEWS14 的一個子圖，包含實體 Angela Merkel、Barack Obama、France 和 China。時間戳以 yy/mm/dd 格式顯示。藍色虛線代表查詢 (Angela Merkel, consult, ?, 2014/08/09) 的正確答案。Angela Merkel 和 Barack Obama 之間的先前互動可被解釋為此預測的說明。

## Introduction

Knowledge graphs (KGs) structure factual information in the form of triples (es, r, eo), where es and eo correspond to entities in the real world and r to a binary relation, e. g., (Anna, born in, Paris). This knowledge representation leads to an interpretation as a directed multigraph, where entities are identified with nodes and relations with edge types. Each edge (es, r, eo) in the KG encodes an observed fact, where the source node es corresponds to the subject entity, the target node eo to the object entity, and the edge type r to the predicate of the factual statement.

## 引言

知識圖譜 (KGs) 將事實資訊結構化為三元組 (es, r, eo) 的形式，其中 es 和 eo 對應真實世界中的實體，r 則對應二元關係，例如 (Anna, born in, Paris)。這種知識表示法可詮釋為一個有向多重圖，其中實體被識別為節點，關係則被識別為邊的類型。KG 中的每條邊 (es, r, eo) 都編碼一個觀察到的事實，其中來源節點 es 對應主體實體，目標節點 eo 對應客體實體，邊類型 r 則對應事實陳述的謂詞。

Some real-world information also includes a temporal dimension, e. g., the event (Anna, born in, Paris) happened on a specific date. To model the large amount of available event data that induce complex interactions between entities over time, temporal knowledge graphs (tKGs) have been introduced. Temporal KGs extend the triples to quadruples (es, r, eo, t) to integrate a timestamp or time range t, where t indicates the time validity of the static event (es, r, eo), e. g., (Angela Merkel, visit, China, 2014/07/04). Figure 1 visualizes a subgraph from the dataset ICEWS14 as an example of a tKG. In this work, we focus on tKGs where each edge is equipped with a single timestamp.

一些真實世界的資訊也包含時間維度，例如事件 (Anna, born in, Paris) 發生在特定日期。為了模擬大量可用事件資料所引發的實體間複雜互動，時間知識圖譜 (tKGs) 被引入。時間知識圖譜將三元組擴展為四元組 (es, r, eo, t)，以整合時間戳或時間範圍 t，其中 t 指示靜態事件 (es, r, eo) 的時間有效性，例如 (Angela Merkel, visit, China, 2014/07/04)。圖 1 將資料集 ICEWS14 的一個子圖視覺化為 tKG 的範例。在這項工作中，我們專注於每條邊都配備單一時間戳的 tKG。

One of the common tasks on KGs is link prediction, which finds application in areas such as recommender systems (Hildebrandt et al. 2019), knowledge base completion (Nguyen et al. 2018a), and drug repurposing (Liu et al. 2021). Taking the additional temporal dimension into account, it is of special interest to forecast events for future timestamps based on past information. Notable real-world applications that rely on accurate event forecasting are, e. g., clinical decision support, supply chain management, and extreme events modeling. In this work, we address link forecasting on tKGs, where we consider queries (es, r, ?, t) for a timestamp t that has not been seen during training.

知識圖譜上的一項常見任務是連結預測，其應用領域包括推薦系統 (Hildebrandt et al. 2019)、知識庫補全 (Nguyen et al. 2018a) 以及藥物再利用 (Liu et al. 2021)。考慮到額外的時間維度，基於過去資訊預測未來時間戳的事件特別受到關注。依賴準確事件預測的著名真實世界應用包括臨床決策支援、供應鏈管理和極端事件模型等。在這項工作中，我們處理 tKG 上的連結預測，其中我們考慮時間戳 t 在訓練期間未曾出現的查詢 (es, r, ?, t)。

Several embedding-based methods have been introduced for tKGs to solve link prediction and forecasting (link prediction with future timestamps), e.g., TTransE (Leblay and Chekol 2018), TNTComplEx (Lacroix, Obozinski, and Usunier 2020), and RE-Net (Jin et al. 2019). The underlying principle is to project the entities and relations into a low-dimensional vector space while preserving the topology and temporal dynamics of the tKG. These methods can learn the complex patterns that lead to an event but often lack transparency and interpretability.

已有數種基於嵌入的方法被引入 tKG 中，以解決連結預測和預報（具未來時間戳的連結預測）的問題，例如 TTransE (Leblay and Chekol 2018)、TNTComplEx (Lacroix, Obozinski, and Usunier 2020) 和 RE-Net (Jin et al. 2019)。其基本原理是將實體和關係投射到低維向量空間中，同時保留 tKG 的拓撲結構和時間動態。這些方法可以學習導致事件發生的複雜模式，但通常缺乏透明度和可解釋性。

To increase the transparency and trustworthiness of the solutions, human-understandable explanations are necessary, which can be provided by logical rules. However, the manual creation of rules is often difficult due to the complex nature of events. Domain experts cannot articulate the conditions for the occurrence of an event sufficiently formally to express this knowledge as rules, which leads to a problem termed as the knowledge acquisition bottleneck. Generally, symbolic methods that make use of logical rules tend to suffer from scalability issues, which make them impractical for the application on large real-world datasets.

為增加解決方案的透明度和可信度，人類可理解的解釋是必要的，而這可以透過邏輯規則提供。然而，由於事件的複雜性，手動創建規則通常很困難。領域專家無法足夠形式化地闡述事件發生的條件，以將此知識表達為規則，這導致了所謂的知識獲取瓶頸問題。一般而言，使用邏輯規則的符號方法往往會遇到可擴展性問題，這使得它們在大型真實世界資料集上的應用不切實際。

We propose TLogic that automatically mines cyclic temporal logical rules by extracting temporal random walks from the graph. We achieve both high predictive performance and time-consistent explanations in the form of temporal rules, which conform to the observation that the occurrence of an event is usually triggered by previous events. The main contributions of this work are summarized as follows:

我們提出了 TLogic，它透過從圖中提取時間隨機漫步來自動探勘循環時間邏輯規則。我們在實現高預測性能和以時間規則形式提供時間一致性解釋方面都取得了成功，這與事件的發生通常由先前事件觸發的觀察結果一致。本研究的主要貢獻總結如下：

*   We introduce TLogic, a novel symbolic framework based on temporal random walks in temporal knowledge graphs. It is the first approach that directly learns temporal logical rules from tKGs and applies these rules to the link forecasting task.
*   我們介紹了 TLogic，這是一個基於時間知識圖中時間隨機漫步的新穎符號框架。這是第一個直接從 tKG 學習時間邏輯規則並將這些規則應用於連結預測任務的方法。

*   Our approach provides explicit and human-readable explanations in the form of temporal logical rules and is scalable to large datasets.
*   我們的方法以時間邏輯規則的形式提供了明確且人類可讀的解釋，並且可擴展至大型資料集。

*   We conduct experiments on three benchmark datasets (ICEWS14, ICEWS18, and ICEWS0515) and show better overall performance compared with state-of-the-art baselines.
*   我們在三個基準資料集 (ICEWS14, ICEWS18, 和 ICEWS0515) 上進行了實驗，並顯示與最先進的基準相比，整體性能更佳。

*   We demonstrate the effectiveness of our method in the inductive setting where our learned rules are transferred to a related dataset with a common vocabulary.
*   我們展示了我們的方法在歸納設定中的有效性，其中我們學習到的規則被轉移到具有共通詞彙的相關資料集。

## Related Work

Subsymbolic machine learning methods, e. g., embedding-based algorithms, have achieved success for the link prediction task on static KGs. Well-known methods include RESCAL (Nickel, Tresp, and Kriegel 2011), TransE (Bordes et al. 2013), DistMult (Yang et al. 2015), and ComplEx (Trouillon et al. 2016) as well as the graph convolutional approaches R-GCN (Schlichtkrull et al. 2018) and CompGCN (Vashishth et al. 2020). Several approaches have been recently proposed to handle tKGs, such as TTransE (Leblay and Chekol 2018), TA-DistMult (García-Durán, Dumančić, and Niepert 2018), DE-SimplE (Goel et al. 2020), TNTComplEx (Lacroix, Obozinski, and Usunier 2020), CyGNet (Zhu et al. 2021), RE-Net (Jin et al. 2019), and xERTE (Han et al. 2021). The main idea of these methods is to explicitly learn embeddings for timestamps or to integrate temporal information into the entity or relation embeddings. However, the black-box property of embeddings makes it difficult for humans to understand the predictions. Moreover, approaches with shallow embeddings are not suitable for an inductive setting with previously unseen entities, relations, or timestamps. From the above methods, only CyGNet, RE-Net, and xERTE are designed for the forecasting task. xERTE is also able to provide explanations by extracting relevant subgraphs around the query subject.

## 相關研究

次符號機器學習方法，例如基於嵌入的演算法，在靜態知識圖譜的連結預測任務上取得了成功。著名的方法包括 RESCAL (Nickel, Tresp, and Kriegel 2011)、TransE (Bordes et al. 2013)、DistMult (Yang et al. 2015) 和 ComplEx (Trouillon et al. 2016)，以及圖卷積方法 R-GCN (Schlichtkrull et al. 2018) 和 CompGCN (Vashishth et al. 2020)。最近也提出了幾種處理 tKG 的方法，例如 TTransE (Leblay and Chekol 2018)、TA-DistMult (García-Durán, Dumančić, and Niepert 2018)、DE-SimplE (Goel et al. 2020)、TNTComplEx (Lacroix, Obozinski, and Usunier 2020)、CyGNet (Zhu et al. 2021)、RE-Net (Jin et al. 2019) 和 xERTE (Han et al. 2021)。這些方法的主要思想是明確地學習時間戳的嵌入，或將時間資訊整合到實體或關係的嵌入中。然而，嵌入的黑盒子特性使得人類難以理解其預測。此外，具有淺層嵌入的方法不適用於具有先前未見實體、關係或時間戳的歸納設定。在上述方法中，只有 CyGNet、RE-Net 和 xERTE 是為預測任務而設計的。xERTE 也能夠透過在查詢主體周圍提取相關子圖來提供解釋。

Symbolic approaches for link prediction on KGs like AMIE+ (Galárraga et al. 2015) and AnyBURL (Meilicke et al. 2019) mine logical rules from the dataset, which are then applied to predict new links. StreamLearner (Omran, Wang, and Wang 2019) is one of the first methods for learning temporal rules. It employs a static rule learner to generate rules, which are then generalized to the temporal domain. However, they only consider a rather restricted set of temporal rules, where all body atoms have the same timestamp.

用於知識圖譜連結預測的符號方法，如 AMIE+ (Galárraga et al. 2015) 和 AnyBURL (Meilicke et al. 2019)，從資料集中探勘邏輯規則，然後應用這些規則來預測新的連結。StreamLearner (Omran, Wang, and Wang 2019) 是學習時間規則的首批方法之一。它採用靜態規則學習器來生成規則，然後將其推廣到時間域。然而，它們只考慮了一組相當受限的時間規則，其中所有主體原子都具有相同的時間戳。

Another class of approaches is based on random walks in the graph, where the walks can support an interpretable explanation for the predictions. For example, AnyBURL samples random walks for generating rules. The methods dynnode2vec (Mahdavi, Khoshraftar, and An 2018) and change2vec (Bian et al. 2019) alternately extract random walks on tKG snapshots and learn parameters for node embeddings, but they do not capture temporal patterns within the random walks. Nguyen et al. (2018b) extend the concept of random walks to temporal random walks on continuous-time dynamic networks for learning node embeddings, where the sequence of edges in the walk only moves forward in time.

另一類方法基於圖中的隨機漫步，其中漫步可以支持對預測的可解釋性解釋。例如，AnyBURL 對生成規則的隨機漫步進行抽樣。dynnode2vec (Mahdavi, Khoshraftar, and An 2018) 和 change2vec (Bian et al. 2019) 方法交替地在 tKG 快照上提取隨機漫步並學習節點嵌入的參數，但它們在隨機漫步中無法捕捉時間模式。Nguyen 等人 (2018b) 將隨機漫步的概念擴展到連續時間動態網絡上的時間隨機漫步，以學習節點嵌入，其中漫步中的邊序列僅隨時間向前移動。

## Preliminaries

Let [n] := {1, 2, ..., n}.

## 預備知識

令 [n] := {1, 2, ..., n}。

Temporal knowledge graph Let E denote the set of entities, R the set of relations, and T the set of timestamps. A temporal knowledge graph (tKG) is a collection of facts G ⊆ E × R × E × T, where each fact is represented by a quadruple (es, r, eo, t). The quadruple (es, r, eo, t) is also called a link or edge, and it indicates a connection between the subject entity es ∈ E and the object entity eo ∈ E via the relation r ∈ R. The timestamp t ∈ T implies the occurrence of the event (es, r, eo) at time t, where t can be measured in units such as hour, day, and year.

時間知識圖譜 令 E 表示實體集合，R 表示關係集合，T 表示時間戳集合。時間知識圖譜 (tKG) 是事實的集合 G ⊆ E × R × E × T，其中每個事實由一個四元組 (es, r, eo, t) 表示。四元組 (es, r, eo, t) 也稱為連結或邊，表示主體實體 es ∈ E 與客體實體 eo ∈ E 之間透過關係 r ∈ R 的連結。時間戳 t ∈ T 表示事件 (es, r, eo) 在時間 t 的發生，其中 t 可以用小時、天和年等單位來衡量。

For two timestamps t and t', we denote the fact that t occurs earlier than t' by t < t'. If additionally, t could represent the same time as t', we write t ≤ t'.

對於兩個時間戳 t 和 t'，我們用 t < t' 表示 t 早於 t' 發生的事實。如果此外，t 可以代表與 t' 相同的時間，我們寫作 t ≤ t'。

We define for each edge (es, r, eo, t) an inverse edge (eo, r⁻¹, es, t) that interchanges the positions of the subject and object entity to allow the random walker to move along the edge in both directions. The relation r⁻¹ ∈ R is called the inverse relation of r.

我們為每個邊 (es, r, eo, t) 定義一個反向邊 (eo, r⁻¹, es, t)，它交換了主體和客體實體的位置，以允許隨機漫步者沿著邊在兩個方向上移動。關係 r⁻¹ ∈ R 被稱為 r 的反向關係。

Link forecasting The goal of the link forecasting task is to predict new links for future timestamps. Given a query with a previously unseen timestamp (es, r, ?, t), we want to identify a ranked list of object candidates that are most likely to complete the query. For subject prediction, we formulate the query as (eo, r⁻¹, ?, t).

連結預測 連結預測任務的目標是預測未來時間戳的新連結。給定一個帶有先前未見時間戳 (es, r, ?, t) 的查詢，我們希望識別出一個最有可能完成查詢的客體候選者排名列表。對於主體預測，我們將查詢表述為 (eo, r⁻¹, ?, t)。

Temporal random walk A non-increasing temporal random walk W of length l ∈ N from entity el+1 ∈ E to entity e1 ∈ E in the tKG G is defined as a sequence of edges
((el+1, rl, el, tl), (el, rl-1, el-1, tl-1), . . . , (e2, r1, e1, t1))
with tl ≥ tl-1 ≥ · · · ≥ t1,
(1)
where (ei+1, ri, ei, ti) ∈ G for i ∈ [l].
A non-increasing temporal random walk complies with time constraints so that the edges are traversed only backward in time, where it is also possible to walk along edges with the same timestamp.

時間隨機漫步 從實體 el+1 ∈ E 到實體 e1 ∈ E 的 tKG G 中，一個長度為 l ∈ N 的非遞增時間隨機漫步 W 定義為邊的序列
((el+1, rl, el, tl), (el, rl-1, el-1, tl-1), . . . , (e2, r1, e1, t1))
其中 tl ≥ tl-1 ≥ · · · ≥ t1,
(1)
其中 (ei+1, ri, ei, ti) ∈ G 對於 i ∈ [l]。
非遞增的時間隨機漫步遵循時間限制，因此邊只能向後遍歷，也可以沿著具有相同時間戳的邊行走。

Temporal logical rule Let Ei and Ti for i ∈ [l + 1] be variables that represent entities and timestamps, respectively. Further, let r1, r2, . . . , rl, rh ∈ R be fixed.
A cyclic temporal logical rule R of length l ∈ N is defined as
((E1, rh, El+1, Tl+1) ← ∧i=1(Ei, ri, Ei+1, Ti))
with the temporal constraints
T1 ≤ T2 ≤ · · · ≤ Tl < Tl+1. (2)
The left-hand side of R is called the rule head, with rh being the head relation, while the right-hand side is called the rule body, which is represented by a conjunction of body atoms (Ei, ri, Ei+1, Ti). The rule is called cyclic because the rule head and the rule body constitute two different walks connecting the same two variables E1 and El+1. A temporal rule implies that if the rule body holds with the temporal constraints given by (2), then the rule head is true as well for a future timestamp Tl+1.
The replacement of the variables Ei and Ti by constant terms is called grounding or instantiation. For example, a grounding of the temporal rule
((E1, consult, E2, T2) ← (E1, discuss by telephone, E2, T1))
is given by the edges (Angela Merkel, discuss by telephone, Barack Obama, 2014/07/22) and (Angela Merkel, consult, Barack Obama, 2014/08/09) in Figure 1. Let rule grounding refer to the replacement of the variables in the entire rule and body grounding refer to the replacement of the variables only in the body, where all groundings must comply with the temporal constraints in (2).

時序邏輯規則 令 Ei 和 Ti (i ∈ [l + 1]) 分別為代表實體和時間戳的變數。此外，令 r1, r2, . . . , rl, rh ∈ R 為固定的。
長度為 l ∈ N 的循環時序邏輯規則 R 定義為
((E1, rh, El+1, Tl+1) ← ∧i=1(Ei, ri, Ei+1, Ti))
具有時間限制
T1 ≤ T2 ≤ · · · ≤ Tl < Tl+1。 (2)
R 的左側稱為規則頭，其中 rh 是頭部關係，而右側稱為規則體，由主體原子 (Ei, ri, Ei+1, Ti) 的合取表示。該規則被稱為循環規則，因為規則頭和規則體構成了連接相同兩個變數 E1 和 El+1 的兩個不同漫步。一個時間規則意味著，如果規則體在 (2) 給出的時間限制下成立，那麼規則頭對於未來的時間戳 Tl+1 也為真。
將變數 Ei 和 Ti 替換為常數項稱為基底化或實例化。例如，時間規則的基底化
((E1, consult, E2, T2) ← (E1, discuss by telephone, E2, T1))
由圖 1 中的邊 (Angela Merkel, discuss by telephone, Barack Obama, 2014/07/22) 和 (Angela Merkel, consult, Barack Obama, 2014/08/09) 給出。令規則基底化指整個規則中變數的替換，而主體基底化指僅在主體中替換變數，其中所有基底化都必須符合 (2) 中的時間限制。

In many domains, logical rules are frequently violated so that confidence values are determined to estimate the probability of a rule's correctness. We adapt the standard confidence to take timestamp values into account. Let (r1, r2, . . . , rl, rh) be the relations in a rule R. The body support is defined as the number of body groundings, i. e., the number of tuples (e1, . . . , el+1, t1, . . . , tl) such that (ei, ri, ei+1, ti) ∈ G for i ∈ [l] and ti < ti+1 for i ∈ [l − 1]. The rule support is defined as the number of body groundings such that there exists a timestamp tl+1 > tl with (e1, rh, el+1, tl+1) ∈ G. The confidence of the rule R, denoted by conf(R), can then be obtained by dividing the rule support by the body support.

在許多領域中，邏輯規則經常被違反，因此需要確定信賴度值來估計規則的正確性機率。我們調整標準信賴度以考慮時間戳值。令 (r1, r2, . . . , rl, rh) 為規則 R 中的關係。主體支持度定義為主體基底化的數量，即元組 (e1, . . . , el+1, t1, . . . , tl) 的數量，使得 (ei, ri, ei+1, ti) ∈ G 對於 i ∈ [l] 且 ti < ti+1 對於 i ∈ [l − 1]。規則支持度定義為主體基底化的數量，使得存在一個時間戳 tl+1 > tl 且 (e1, rh, el+1, tl+1) ∈ G。規則 R 的信賴度，表示為 conf(R)，可以透過將規則支持度除以主體支持度來獲得。

[Image]

## Our Framework

We introduce TLogic, a rule-based link forecasting framework for tKGs. TLogic first extracts temporal walks from the graph and then lifts these walks to a more abstract, semantic level to obtain temporal rules that generalize to new data. The application of these rules generates answer candidates, for which the body groundings in the graph serve as explicit and human-readable explanations. Our framework consists of the components rule learning and rule application. The pseudocode for rule learning is shown in Algorithm 1 and for rule application in Algorithm 2.

## 我們的框架

我們介紹 TLogic，一個基於規則的 tKGs 連結預測框架。TLogic 首先從圖形中提取時間漫步，然後將這些漫步提升到一個更抽象、語義的層次，以獲得可推廣到新資料的時間規則。這些規則的應用會產生答案候選，圖形中的主體基礎可作為明確且人類可讀的解釋。我們的框架由規則學習和規則應用兩個部分組成。規則學習的虛擬碼如演算法 1 所示，規則應用的虛擬碼如演算法 2 所示。

### Rule Learning

As the first step of rule learning, temporal walks are extracted from the tKG G. For a rule of length l, a walk of length l + 1 is sampled, where the additional step corresponds to the rule head.
Let rh be a fixed relation, for which we want to learn rules. For the first sampling step m = 1, we sample an edge (e1, rh, el+1, tl+1), which will serve as the rule head, uniformly from all edges with relation type rh. A temporal random walker then samples iteratively edges adjacent to the current object until a walk of length l + 1 is obtained.
For sampling step m ∈ {2, . . . , l + 1}, let (es, r, eo, t) denote the previously sampled edge and A(m, eo, t) the set of feasible edges for the next transition. To fulfill the temporal constraints in (1) and (2), we define
A(m, eo, t) :=
{
{(eo, r, e', t') | (eo, r, e', t') ∈ G, t' < t} if m = 2,
{(eo, r, e', t') | (eo, r, e', t') ∈ G', t' ≤ t} if m ∈ {3, . . . , l},
{(eo, r, e1, t') | (eo, r, e1, t') ∈ G', t' ≤ t} if m = l + 1,
where G' := G \ {(eo, r⁻¹, es, t)} excludes the inverse edge to avoid redundant rules. For obtaining cyclic walks, we sample in the last step m = l + 1 an edge that connects the walk to the first entity e1 if such edges exist. Otherwise, we sample the next walk.

### 規則學習

作為規則學習的第一步，我們從 tKG G 中提取時間漫步。對於長度為 l 的規則，我們會抽樣一個長度為 l + 1 的漫步，其中額外的步驟對應於規則頭。
令 rh 為一個固定關係，我們想為其學習規則。對於第一個抽樣步驟 m = 1，我們從所有關係類型為 rh 的邊中均勻抽樣一條邊 (e1, rh, el+1, tl+1)，它將作為規則頭。然後，一個時間隨機漫步者會迭代地抽樣與當前對象相鄰的邊，直到獲得一個長度為 l + 1 的漫步。
對於抽樣步驟 m ∈ {2, . . . , l + 1}，令 (es, r, eo, t) 表示先前抽樣的邊，A(m, eo, t) 表示下一個轉換的可行邊集合。為了滿足 (1) 和 (2) 中的時間限制，我們定義
A(m, eo, t) :=
{
{(eo, r, e', t') | (eo, r, e', t') ∈ G, t' < t} if m = 2,
{(eo, r, e', t') | (eo, r, e', t') ∈ G', t' ≤ t} if m ∈ {3, . . . , l},
{(eo, r, e1, t') | (eo, r, e1, t') ∈ G', t' ≤ t} if m = l + 1,
其中 G' := G \ {(eo, r⁻¹, es, t)} 排除了反向邊以避免冗餘規則。為了獲得循環漫步，我們在最後一步 m = l + 1 中抽樣一條將漫步連接到第一個實體 e1 的邊（如果存在這樣的邊）。否則，我們抽樣下一個漫步。

The transition distribution for sampling the next edge can be either uniform or exponentially weighted. We define an index mapping m' := (l + 1) − (m − 2) to be consistent with the indices in (1). Then, the exponentially weighted probability for choosing edge u ∈ A(m, em', tm') for m ∈ {2, . . . , l + 1} is given by
P(u; m, em', tm') = exp(tu − tm') / ∑û∈A(m,em',tm') exp(tû − tm') (3)
where tu denotes the timestamp of edge u. The exponential weighting favors edges with timestamps that are closer to the timestamp of the previous edge and probably more relevant for prediction.
The resulting temporal walk W is given by
((e1, rh, el+1, tl+1), (el+1, rl, el, tl), . . . , (e2, r1, e1, t1)). (4)
W can then be transformed to a temporal rule R by replacing the entities and timestamps with variables. While the first edge in W becomes the rule head (E1, rh, El+1, Tl+1), the other edges are mapped to body atoms, where each edge (ei+1, ri, ei, ti) is converted to the body atom (Ei, ri⁻¹, Ei+1, Ti). The final rule R is denoted by
((E1, rh, El+1, Tl+1) ← ∧i=1(Ei, ri⁻¹, Ei+1, Ti)). (5)
In addition, we impose the temporal consistency constraints T1 ≤ T2 ≤ · · · ≤ Tl < Tl+1.

抽樣下一個邊的轉移分佈可以是均勻的或指數加權的。我們定義一個索引映射 m' := (l + 1) − (m − 2)，使其與 (1) 中的索引一致。然後，對於 m ∈ {2, . . . , l + 1}，選擇邊 u ∈ A(m, em', tm') 的指數加權機率由下式給出：
P(u; m, em', tm') = exp(tu − tm') / ∑û∈A(m,em',tm') exp(tû − tm') (3)
其中 tu 表示邊 u 的時間戳。指數加權偏好於時間戳更接近前一條邊的時間戳的邊，這些邊可能與預測更相關。
由此產生的時間漫步 W 由下式給出：
((e1, rh, el+1, tl+1), (el+1, rl, el, tl), . . . , (e2, r1, e1, t1))。(4)
然後可以透過將實體和時間戳替換為變數，將 W 轉換為時間規則 R。W 中的第一條邊成為規則頭 (E1, rh, El+1, Tl+1)，而其他邊則映射到主體原子，其中每個邊 (ei+1, ri, ei, ti) 都轉換為主體原子 (Ei, ri⁻¹, Ei+1, Ti)。最終的規則 R 表示為：
((E1, rh, El+1, Tl+1) ← ∧i=1(Ei, ri⁻¹, Ei+1, Ti))。(5)
此外，我們施加了時間一致性約束 T1 ≤ T2 ≤ · · · ≤ Tl < Tl+1。

The entities (e1, . . . , el+1) in W do not need to be distinct since a pair of entities can have many interactions at different points in time. For example, Angela Merkel made several visits to China in 2014, which could constitute important information for the prediction. Repetitive occurrences of the same entity in W are replaced with the same random variable in R to maintain this knowledge.
For the confidence estimation of R, we sample from the graph a fixed number of body groundings, which have to match the body relations and the variable constraints mentioned in the last paragraph while satisfying the condition from (2). The number of unique bodies serves as the body support. The rule support is determined by counting the number of bodies for which an edge with relation type rh exists that connects e1 and el+1 from the body. Moreover, the timestamp of this edge has to be greater than all body timestamps to fulfill (2).
For every relation r ∈ R, we sample n ∈ N temporal walks for a set of prespecified lengths L ⊆ N. The set TRrl stands for all rules of length l with head relation r with their corresponding confidences. All rules for relation r are included in TRr := ∪l∈L TRrl, and the complete set of learned temporal rules is given by TR := ∪r∈R TRr.
It is possible to learn rules only for a single relation or a set of specific relations of interest. Explicitly learning rules for all relations is especially effective for rare relations that would otherwise only be sampled with a small probability. The learned rules are not specific to the graph from which they have been extracted, but they could be employed in an inductive setting where the rules are transferred to related datasets that share a common vocabulary for straightforward application.

W 中的實體 (e1, . . . , el+1) 不必是不同的，因為一對實體在不同的時間點可以有多次互動。例如，Angela Merkel 在 2014 年多次訪問中國，這可能構成預測的重要資訊。W 中相同實體的重複出現會被 R 中相同的隨機變數替換，以保持此知識。
為了估計 R 的信賴度，我們從圖中抽樣固定數量的規則體基底化，這些基底化必須與規則體關係和上一段中提到的變數限制相符，同時滿足 (2) 的條件。唯一規則體的數量作為規則體支持度。規則支持度是透過計算存在一個關係類型為 rh 的邊連接規則體中的 e1 和 el+1 的規則體數量來確定的。此外，此邊的時間戳必須大於所有規則體時間戳以滿足 (2)。
對於每個關係 r ∈ R，我們為一組預先指定的長度 L ⊆ N 抽樣 n ∈ N 個時間漫步。集合 TRrl 代表所有長度為 l 且頭部關係為 r 的規則及其對應的信賴度。關係 r 的所有規則都包含在 TRr := ∪l∈L TRrl 中，而已學習的時間規則的完整集合由 TR := ∪r∈R TRr 給出。
只為單一關係或一組特定感興趣的關係學習規則是可能的。明確地為所有關係學習規則對於罕見的關係特別有效，否則這些關係只會以很小的機率被抽樣。學習到的規則並非特定於從中提取它們的圖，但它們可以在歸納設定中被採用，其中規則被轉移到共享共同詞彙的相關資料集以進行直接應用。

### Rule Application

The learned temporal rules TR are applied to answer queries of the form q = (eq, rq, ?, tq). The answer candidates are retrieved from the target entities of body groundings in the tKG G. If there exist no rules TRrq for the query relation rq, or if there are no matching body groundings in the graph, then no answers are predicted for the given query.
To apply the rules on relevant data, a subgraph SG ⊆ G dependent on a time window w ∈ N ∪ {∞} is retrieved. For w ∈ N, the subgraph SG contains all edges from G that have timestamps t ∈ [tq − w, tq). If w = ∞, then all edges with timestamps prior to the query timestamp tq are used for rule application, i. e., SG consists of all facts with t ∈ [tmin, tq), where tmin is the minimum timestamp in the graph G.

### 規則應用

學習到的時間規則 TR 用於回答形式為 q = (eq, rq, ?, tq) 的查詢。答案候選從 tKG G 中規則體基礎的目標實體中檢索。如果查詢關係 rq 不存在規則 TRrq，或者圖中沒有匹配的規則體基礎，則不會為給定的查詢預測答案。
為了在相關資料上應用規則，會檢索一個取決於時間視窗 w ∈ N ∪ {∞} 的子圖 SG ⊆ G。對於 w ∈ N，子圖 SG 包含 G 中所有時間戳 t ∈ [tq − w, tq) 的邊。如果 w = ∞，則所有在查詢時間戳 tq 之前的時間戳的邊都用於規則應用，即 SG 由所有 t ∈ [tmin, tq) 的事實組成，其中 tmin 是圖 G 中的最小時間戳。

We apply the rules TRrq by decreasing confidence, where each rule R generates a set of answer candidates C(R). Each candidate c ∈ C(R) is then scored by a function f : TRrq × E → [0, 1] that reflects the probability of the candidate being the correct answer to the query.
Let B(R, c) be the set of body groundings of rule R that start at entity eq and end at entity c. We choose as score function f a convex combination of the rule's confidence and a function that takes the time difference tq − t1(B(R, c)) as input, where t1(B(R, c)) denotes the earliest timestamp t1 in the body. If several body groundings exist, we take from all possible t1 values the one that is closest to tq. For candidate c ∈ C(R), the score function is defined as
f(R, c) = α conf(R) + (1 − α) · exp(−λ(tq − t1(B(R, c)))) (6)
with λ > 0 and α ∈ [0, 1].

我們透過降低信賴度來應用規則 TRrq，其中每個規則 R 都會產生一組答案候選 C(R)。然後，每個候選 c ∈ C(R) 都由一個函數 f : TRrq × E → [0, 1] 進行評分，該函數反映了候選成為查詢正確答案的機率。
令 B(R, c) 為從實體 eq 開始並在實體 c 結束的規則 R 的主體基礎集合。我們選擇評分函數 f 作為規則信賴度和一個以時間差 tq − t1(B(R, c)) 為輸入的函數的凸組合，其中 t1(B(R, c)) 表示主體中最早的時間戳 t1。如果存在多個主體基礎，我們從所有可能的 t1 值中選擇最接近 tq 的值。對於候選 c ∈ C(R)，評分函數定義為
f(R, c) = α conf(R) + (1 − α) · exp(−λ(tq − t1(B(R, c)))) (6)
其中 λ > 0 且 α ∈ [0, 1]。

The intuition for this choice of f is that candidates generated by high-confidence rules should receive a higher score. Adding a dependency on the timeframe of the rule grounding is based on the observation that the existence of edges in a rule become increasingly probable with decreasing time difference between the edges. We choose the exponential distribution since it is commonly used to model interarrival times of events. The time difference tq − t1(B(R, c)) is always non-negative for a future timestamp value tq, and with the assumption that there exists a fixed mean, the exponential distribution is also the maximum entropy distribution for such a time difference variable. The exponential distribution is rescaled so that both summands are in the range [0, 1].

選擇此 f 的直覺是，由高信賴度規則產生的候選者應獲得較高的分數。加入對規則基礎時間範圍的依賴性，是基於觀察到規則中邊的存在機率隨著邊之間時間差的減少而增加。我們選擇指數分佈，因為它通常用於模擬事件的到達間隔時間。對於未來的時間戳值 tq，時間差 tq − t1(B(R, c)) 總是為非負數，並且在存在固定平均值的假設下，指數分佈也是此類時間差變數的最大熵分佈。指數分佈被重新調整，以使兩個加數都在 [0, 1] 的範圍內。

All candidates are saved with their scores as (c, f(R, c)) in C. We stop the rule application when the number of different answer candidates |{c | ∃R : (c, f(R, c)) ∈ C}| is at least k so that there is no need to go through all rules.

所有候選者及其分數 (c, f(R, c)) 都儲存在 C 中。當不同答案候選的數量 |{c | ∃R : (c, f(R, c)) ∈ C}| 至少為 k 時，我們停止規則應用，因此無需遍歷所有規則。

### Candidate Ranking

For the ranking of the answer candidates, all scores of each candidate c are aggregated through a noisy-OR calculation, which produces the final score
1 − ∏{s|(c,s)∈C}(1 − s). (7)
The idea is to aggregate the scores to produce a probability, where candidates implied by more rules should have a higher score.

### 候選排名

為了對答案候選進行排名，每個候選 c 的所有分數都透過 noisy-OR 計算進行匯總，從而產生最終分數
1 − ∏{s|(c,s)∈C}(1 − s)。(7)
這個想法是匯總分數以產生一個機率，其中由更多規則暗示的候選應該有更高的分數。

In case there are no rules for the query relation rq, or if there are no matching body groundings in the graph, it might still be interesting to retrieve possible answer candidates. In the experiments, we apply a simple baseline where the scores for the candidates are obtained from the overall object distribution in the training data if rq is a new relation. If rq already exists in the training set, we take the object distribution of the edges with relation type rq.

如果查詢關係 rq 沒有規則，或者圖中沒有匹配的主體基礎，檢索可能的答案候選仍然可能很有趣。在實驗中，我們應用了一個簡單的基準線，如果 rq 是一個新的關係，則候選的分數從訓練資料中的整體客體分佈中獲得。如果 rq 已經存在於訓練集中，我們採用關係類型為 rq 的邊的客體分佈。

## Experiments

## 實驗

### Datasets

We conduct experiments on the dataset Integrated Crisis Early Warning System¹ (ICEWS), which contains information about international events and is a commonly used benchmark dataset for link prediction on tKGs. We choose the subsets ICEWS14, ICEWS18, and ICEWS0515, which include data from the years 2014, 2018, and 2005 to 2015, respectively. Since we consider link forecasting, each dataset is split into training, validation, and test set so that the timestamps in the training set occur earlier than the timestamps in the validation set, which again occur earlier than the timestamps in the test set. To ensure a fair comparison, we use the split provided by Han et al. (2021)². The statistics of the datasets are summarized in the supplementary material.
¹https://dataverse.harvard.edu/dataverse/icews
²https://github.com/TemporalKGTeam/xERTE

### 資料集

我們在 Integrated Crisis Early Warning System¹ (ICEWS) 資料集上進行實驗，該資料集包含有關國際事件的資訊，是 tKGs 連結預測的常用基準資料集。我們選擇了子集 ICEWS14、ICEWS18 和 ICEWS0515，分別包含 2014 年、2018 年以及 2005 年至 2015 年的資料。由於我們考慮連結預測，每個資料集都被劃分為訓練集、驗證集和測試集，以便訓練集中的時間戳早於驗證集中的時間戳，而驗證集中的時間戳又早于測試集中的時間戳。為確保公平比較，我們使用 Han 等人 (2021)² 提供的劃分。資料集的統計數據摘要於補充材料中。
¹https://dataverse.harvard.edu/dataverse/icews
²https://github.com/TemporalKGTeam/xERTE

### Experimental Setup

For each test instance (eq, rq, eo, tq), we generate a list of candidates for both object prediction (eq, rq, ?, tq) and subject prediction (eo, (rq)⁻¹, ?, tq). The candidates are ranked by decreasing scores, which are calculated according to (7).
The confidence for each rule is estimated by sampling 500 body groundings and counting the number of times the rule head holds. We learn rules of the lengths 1, 2, and 3, and for application, we only consider the rules with a minimum confidence of 0.01 and minimum body support of 2.
We compute the mean reciprocal rank (MRR) and hits@k for k ∈ {1, 3, 10}, which are standard metrics for link prediction on KGs. For a rank x ∈ N, the reciprocal rank is defined as 1/x, and the MRR is the average of all reciprocal ranks of the correct query answers across all queries. The metric hits@k (h@k) indicates the proportion of queries for which the correct entity appears under the top k candidates.
Similar to Han et al. (2021), we perform time-aware filtering where all correct entities at the query timestamp except for the true query object are filtered out from the answers. In comparison to the alternative setting that filters out all other objects that appear together with the query subject and relation at any timestamp, time-aware filtering yields a more realistic performance estimate.

### 實驗設置

對於每個測試實例 (eq, rq, eo, tq)，我們為客體預測 (eq, rq, ?, tq) 和主體預測 (eo, (rq)⁻¹, ?, tq) 都產生一個候選列表。候選者按遞減的分數排名，分數根據 (7) 計算。
每個規則的信賴度是透過抽樣 500 個規則體基礎並計算規則頭成立的次數來估計的。我們學習長度為 1、2 和 3 的規則，在應用時，我們只考慮信賴度至少為 0.01 且規則體支持度至少為 2 的規則。
我們計算 k ∈ {1, 3, 10} 的平均倒數排名 (MRR) 和 hits@k，這些是知識圖譜連結預測的標準指標。對於排名 x ∈ N，倒數排名定義為 1/x，而 MRR 是所有查詢的正確答案的倒數排名的平均值。指標 hits@k (h@k) 表示正確實體出現在前 k 個候選中的查詢比例。
與 Han 等人 (2021) 類似，我們執行時間感知過濾，其中除了真實的查詢客體之外，所有在查詢時間戳處的正確實體都從答案中過濾掉。與過濾掉所有在任何時間戳與查詢主體和關係一起出現的其他客體的替代設置相比，時間感知過濾產生了更現實的性能估計。

### Baseline methods

We compare TLogic³ with the state-of-the-art baselines for static link prediction DistMult (Yang et al. 2015), ComplEx (Trouillon et al. 2016), and AnyBURL (Meilicke et al. 2019, 2020) as well as for temporal link prediction TTransE (Leblay and Chekol 2018), TA-DistMult (García-Durán, Dumančić, and Niepert 2018), DE-SimplE (Goel et al. 2020), TNTComplEx (Lacroix, Obozinski, and Usunier 2020), CyGNet (Zhu et al. 2021), RE-Net (Jin et al. 2019), and xERTE (Han et al. 2021). All baseline results except for the results on AnyBURL are from Han et al. (2021). AnyBURL samples paths based on reinforcement learning and generalizes them to rules, where the rule space also includes, e. g., acyclic rules and rules with constants. A non-temporal variant of TLogic would sample paths randomly and only learn cyclic rules, which would presumably yield worse performance than AnyBURL. Therefore, we choose AnyBURL as a baseline to assess the effectiveness of adding temporal constraints.

### 基準方法

我們將 TLogic³ 與用於靜態連結預測的最新基準方法進行比較，包括 DistMult (Yang et al. 2015)、ComplEx (Trouillon et al. 2016) 和 AnyBURL (Meilicke et al. 2019, 2020)，以及用於時間連結預測的 TTransE (Leblay and Chekol 2018)、TA-DistMult (García-Durán, Dumančić, and Niepert 2018)、DE-SimplE (Goel et al. 2020)、TNTComplEx (Lacroix, Obozinski, and Usunier 2020)、CyGNet (Zhu et al. 2021)、RE-Net (Jin et al. 2019) 和 xERTE (Han et al. 2021)。除了 AnyBURL 的結果外，所有基準結果均來自 Han 等人 (2021)。AnyBURL 根據強化學習對路徑進行採樣並將其推廣為規則，其規則空間也包括非循環規則和帶有常數的規則等。TLogic 的非時間變體會隨機採樣路徑並僅學習循環規則，這可能會導致比 AnyBURL 更差的性能。因此，我們選擇 AnyBURL 作為基準，以評估添加時間約束的有效性。

### Results

The results of the experiments are displayed in Table 1. TLogic outperforms all baseline methods with respect to the metrics MRR, hits@3, and hits@10. Only xERTE performs better than Tlogic for hits@1 on the datasets ICEWS18 and ICEWS0515.
Besides a list of possible answer candidates with corresponding scores, TLogic can also provide temporal rules and body groundings in form of walks from the graph that support the predictions. Table 2 presents three exemplary rules with high confidences that were learned from ICEWS14. For the query (Angela Merkel, consult, ?, 2014/08/09), two walks are shown in Table 2, which serve as time-consistent explanations for the correct answer Barack Obama.
³Code available at https://github.com/liu-yushan/TLogic.

### 結果

實驗結果如表 1 所示。在 MRR、hits@3 和 hits@10 指標方面，TLogic 優於所有基準方法。只有在 ICEWS18 和 ICEWS0515 資料集上，xERTE 在 hits@1 上的表現優於 Tlogic。
除了帶有相應分數的可能答案候選列表外，TLogic 還可以提供時間規則和圖中支持預測的漫步形式的規則體基礎。表 2 展示了從 ICEWS14 中學習到的三個具有高信賴度的範例規則。對於查詢 (Angela Merkel, consult, ?, 2014/08/09)，表 2 中顯示了兩條漫步，它們為正確答案 Barack Obama 提供了時間一致性的解釋。
³程式碼可在 https://github.com/liu-yushan/TLogic 取得。

[Image]

### Inductive setting

One advantage of our learned logical rules is that they are applicable to any new dataset as long as the new dataset covers common relations. This might be relevant for cases where new entities appear. For example, Donald Trump, who served as president of the United States from 2017 to 2021, is included in the dataset ICEWS18 but not in ICEWS14. The logical rules are not tied to particular entities and would still be applicable, while embedding-based methods have difficulties operating in this challenging setting. The models would need to be retrained to obtain embeddings for the new entities, where existing embeddings might also need to be adapted to the different time range.
For the two rule-based methods AnyBURL and TLogic, we apply the rules learned on the training set of ICEWS0515 (with timestamps from 2005/01/01 to 2012/08/06) to the test set of ICEWS14 as well as the rules learned on the training set of ICEWS14 to the test set of ICEWS18 (see Table 3). The performance of TLogic in the inductive setting is for all metrics close to the results in Table 1, while for AnyBURL, especially the results on ICEWS18 drop significantly. It seems that the encoded temporal information in TLogic is essential for achieving correct predictions in the inductive setting. ICEWS14 has only 7,128 entities, while ICEWS18 contains 23,033 entities. The results confirm that temporal rules from TLogic can even be transferred to a dataset with a large number of new entities and timestamps and lead to a strong performance.

### 歸納設定

我們學習到的邏輯規則的一個優點是，只要新資料集涵蓋了常見的關係，它們就適用於任何新的資料集。這對於出現新實體的情況可能很重要。例如，2017 年至 2021 年擔任美國總統的 Donald Trump 被包含在資料集 ICEWS18 中，但不在 ICEWS14 中。邏輯規則不與特定實體掛鉤，因此仍然適用，而基於嵌入的方法在這種具挑戰性的設定中操作有困難。模型需要重新訓練以獲得新實體的嵌入，現有的嵌入也可能需要適應不同的時間範圍。
對於兩種基於規則的方法 AnyBURL 和 TLogic，我們將在 ICEWS0515 訓練集上學習到的規則（時間戳從 2005/01/01 到 2012/08/06）應用於 ICEWS14 的測試集，以及將在 ICEWS14 訓練集上學習到的規則應用於 ICEWS18 的測試集（見表 3）。TLogic 在歸納設定中的所有指標的性能都接近表 1 中的結果，而對於 AnyBURL，尤其是在 ICEWS18 上的結果則顯著下降。看來 TLogic 中編碼的時間資訊對於在歸納設定中實現正確的預測至關重要。ICEWS14 只有 7,128 個實體，而 ICEWS18 包含 23,033 個實體。結果證實，TLogic 的時間規則甚至可以轉移到具有大量新實體和時間戳的資料集，並產生強大的性能。

### Analysis

The results in this section are obtained on the dataset ICEWS14, but the findings are similar for the other two datasets. More detailed results can be found in the supplementary material.

### 分析

本節中的結果是在 ICEWS14 資料集上獲得的，但研究結果與其他兩個資料集相似。更詳細的結果可在補充材料中找到。

#### Number of walks

Figure 2 shows the MRR performance on the validation set of ICEWS14 for different numbers of walks that were extracted during rule learning. We observe a performance increase with a growing number of walks. However, the performance gains saturate between 100 and 200 walks where rather small improvements are attainable.

#### 漫步次數

圖 2 顯示了在規則學習期間提取不同漫步次數時，ICEWS14 驗證集上的 MRR 性能。我們觀察到性能隨著漫步次數的增加而提升。然而，性能增益在 100 到 200 次漫步之間趨於飽和，此時只能獲得相當小的改進。

#### Transition distribution

We test two transition distributions for the extraction of temporal walks: uniform and exponentially weighted according to (3). The rationale behind using an exponentially weighted distribution is the observation that related events tend to happen within a short timeframe. The distribution of the first edge is always uniform to not restrict the variety of obtained walks. Overall, the performance of the exponential distribution consistently exceeds the uniform setting with respect to the MRR (see Figure 2).
We observe that the exponential distribution leads to more rules of length 3 than the uniform setting (11,718 compared to 8,550 rules for 200 walks), while it is the opposite for rules of length 1 (7,858 compared to 11,019 rules). The exponential setting leads to more successful longer walks because the timestamp differences between subsequent edges tend to be smaller. It is less likely that there are no feasible transitions anymore because of temporal constraints. The uniform setting, however, leads to a better exploration of the neighborhood around the start node for shorter walks.

#### 轉移分佈

我們測試了兩種用於提取時間漫步的轉移分佈：根據 (3) 的均勻分佈和指數加權分佈。使用指數加權分佈的理由是觀察到相關事件往往在短時間內發生。第一條邊的分佈總是均勻的，以不限制所獲得漫步的多樣性。總體而言，就 MRR 而言，指數分佈的性能始終優於均勻設定（見圖 2）。
我們觀察到，指數分佈導致的長度為 3 的規則比均勻設定更多（200 次漫步時，11,718 條規則對比 8,550 條），而對於長度為 1 的規則則相反（7,858 條規則對比 11,019 條）。指數設定導致更成功的長漫步，因為後續邊之間的時間戳差異往往較小。由於時間限制，不再有可行轉移的可能性較小。然而，均勻設定可以更好地探索起始節點周圍的鄰域，以進行較短的漫步。

[Image]

#### Rule length

We learn rules of lengths 1, 2, and 3. Using all rules for application results in the best performance (MRR on the validation set: 0.4373), followed by rules of only length 1 (0.4116), 3 (0.4097), and 2 (0.1563). The reason why rules of length 3 perform better than length 2 is that the temporal walks are allowed to transition back and forth between the same entities. Since we only learn cyclic rules, a rule body of length 2 must constitute a path with no recurring entities, resulting in fewer rules and rule groundings in the graph. Interestingly, simple rules of length 1 already yield very good performance.

#### 規則長度

我們學習長度為 1、2 和 3 的規則。將所有規則用於應用會產生最佳性能（驗證集上的 MRR：0.4373），其次是僅使用長度為 1 (0.4116)、3 (0.4097) 和 2 (0.1563) 的規則。長度為 3 的規則比長度為 2 的規則表現更好，原因在於時間漫步允許在相同實體之間來回轉移。由於我們只學習循環規則，長度為 2 的規則體必須構成一條沒有重複實體的路徑，這導致圖中規則和規則基礎較少。有趣的是，長度為 1 的簡單規則已經能產生非常好的性能。

#### Time window

For rule application, we define a time window for retrieving the relevant data. The performance increases with the size of the time window, even though relevant events tend to be close to the query timestamp. The second summand of the score function f in (6) takes the time difference between the query timestamp tq and the earliest body timestamp t1(B(R, c)) into account. In this case, earlier events with a large timestamp difference receive a lesser weight, while generally, as much information as possible is beneficial for prediction.

#### 時間視窗

對於規則應用，我們定義了一個用於檢索相關資料的時間視窗。性能隨著時間視窗的大小而增加，儘管相關事件往往接近查詢時間戳。評分函數 f 中的第二個加數 (6) 考慮了查詢時間戳 tq 與最早規則體時間戳 t1(B(R, c)) 之間的時間差。在這種情況下，具有較大時間戳差異的較早事件會獲得較小的權重，而一般來說，盡可能多的資訊對預測是有益的。

#### Score function

We define the score function f in (6) as a convex combination of the rule's confidence and a function that depends on the time difference tq − t1(B(R, c)). The performance of only using the confidence (MRR: 0.3869) or only using the exponential function (0.4077) is worse than the combination (0.4373), which means that both the information from the rules' confidences and the time differences are important for prediction.

#### 評分函數

我們將 (6) 中的評分函數 f 定義為規則信賴度與一個取決於時間差 tq − t1(B(R, c)) 的函數的凸組合。僅使用信賴度 (MRR: 0.3869) 或僅使用指數函數 (0.4077) 的性能都比組合 (0.4373) 差，這意味著來自規則信賴度的資訊和時間差對於預測都很重要。

#### Variance

The variance in the performance due to different rules obtained from the rule learning component is quite small. Running the same model with the best hyperparameter settings for five different seeds results in a standard deviation of 0.0012 for the MRR. The rule application component is deterministic and always leads to the same candidates with corresponding scores for the same hyperparameter setting.

#### 變異數

由於從規則學習組件獲得的不同規則，性能的變異數非常小。使用五個不同種子以最佳超參數設置運行相同模型，MRR 的標準差為 0.0012。規則應用組件是確定性的，對於相同的超參數設置，總是會產生具有相應分數的相同候選者。

#### Training and inference time

The worst-case time complexity for learning rules of length l is O(|R|nlDb), where n is the number of walks, D the maximum node degree in the training set, and b the number of body samples for estimating the confidence. The worst-case time complexity for inference is given by O(|G| + |TRrq|DL|E| log(k)), where L is the maximum rule length in TRrq and k the minimum number of candidates. For large graphs with high node degrees, it is possible to reduce the complexity to O(|G| + |TRrq|KLD|E| log(k)) by only keeping a maximum of K candidate walks during rule application.
Both training and application can be parallelized since the rule learning for each relation and the rule application for each test query are independent. Rule learning with 200 walks and exponentially weighted transition distribution for rule lengths {1, 2, 3} on a machine with 8 CPUs takes 180 sec for ICEWS14, while the application on the validation set takes 2000 sec, with w = ∞ and k = 20. For comparison, the best-performing baseline xERTE needs for training one epoch on the same machine already 5000 sec, where an MRR of 0.3953 can be obtained, while testing on the validation set takes 700 sec.

#### 訓練與推論時間

學習長度為 l 的規則的最壞情況時間複雜度為 O(|R|nlDb)，其中 n 是漫步次數，D 是訓練集中的最大節點度數，b 是估計信賴度的規則體樣本數。推論的最壞情況時間複雜度為 O(|G| + |TRrq|DL|E| log(k))，其中 L 是 TRrq 中的最大規則長度，k 是候選者的最小數量。對於具有高節點度數的大型圖，可以透過在規則應用期間僅保留最多 K 個候選漫步，將複雜度降低到 O(|G| + |TRrq|KLD|E| log(k))。
訓練和應用都可以並行化，因為每個關係的規則學習和每個測試查詢的規則應用都是獨立的。在具有 8 個 CPU 的機器上，使用 200 次漫步和指數加權轉移分佈對規則長度 {1, 2, 3} 進行規則學習，ICEWS14 需要 180 秒，而在驗證集上的應用需要 2000 秒，其中 w = ∞ 且 k = 20。相較之下，性能最佳的基準 xERTE 在同一台機器上訓練一個 epoch 就需要 5000 秒，可獲得 0.3953 的 MRR，而在驗證集上的測試需要 700 秒。

## Conclusion

We have proposed TLogic, the first symbolic framework that directly learns temporal logical rules from temporal knowledge graphs and applies these rules for link forecasting. The framework generates answers by applying rules to observed events prior to the query timestamp and scores the answer candidates depending on the rules' confidences and time differences. Experiments on three datasets indicate that TLogic achieves better overall performance compared to state-of-the-art baselines. In addition, our approach also provides time-consistent, explicit, and human-readable explanations for the predictions in the form of temporal logical rules.
As future work, it would be interesting to integrate acyclic rules, which could also contain relevant information and might boost the performance for rules of length 2. Furthermore, the simple sampling mechanism for temporal walks could be replaced by a more sophisticated approach, which is able to effectively identify the most promising walks.

## 結論

我們提出了 TLogic，這是第一個直接從時間知識圖譜中學習時序邏輯規則並將這些規則應用於連結預測的符號框架。該框架透過將規則應用於查詢時間戳之前的觀察事件來生成答案，並根據規則的信賴度和時間差異對答案候選進行評分。在三個資料集上的實驗表明，與最先進的基準模型相比，TLogic 取得了更好的整體性能。此外，我們的方法還以時序邏輯規則的形式為預測提供了時間一致、明確且人類可讀的解釋。
作為未來的工作，整合非循環規則將會很有趣，這些規則也可能包含相關資訊，並可能提升長度為 2 的規則的性能。此外，可以採用更複雜的方法來取代簡單的時間漫步抽樣機制，從而有效地識別最有希望的漫步。

## Acknowledgement

This work has been supported by the German Federal Ministry for Economic Affairs and Climate Action (BMWK) as part of the project RAKI under grant number 01MD19012C and by the German Federal Ministry of Education and Research (BMBF) under grant number 01IS18036A. The authors of this work take full responsibility for its content.

## 致謝

本研究由德國聯邦經濟事務與氣候行動部 (BMWK) 在 RAKI 計畫下 (補助號碼 01MD19012C) 以及德國聯邦教育與研究部 (BMBF) (補助號碼 01IS18036A) 資助。本研究作者對其內容負完全責任。

## References

Bian, R.; Koh, Y. S.; Dobbie, G.; and Divoli, A. 2019. Network embedding and change modeling in dynamic heterogeneous networks. In Proceedings of the Forty-Second International ACM SIGIR Conference on Research and Development in Information Retrieval.
Bordes, A.; Usunier, N.; García-Durán, A.; Weston, J.; and Yakhnenko, O. 2013. Translating embeddings for modeling multi-relational data. In Proceedings of the Twenty-Sixth International Conference on Neural Information Processing Systems.
Galárraga, L.; Teflioudi, C.; Hose, K.; and Suchanek, F. M. 2015. Fast rule mining in ontological knowledge bases with AMIE+. The VLDB Journal, 24: 707–730.
García-Durán, A.; Dumančić, S.; and Niepert, M. 2018. Learning sequence encoders for temporal knowledge graph completion. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing.
Goel, R.; Kazemi, S. M.; Brubaker, M.; and Poupart, P. 2020. Diachronic embedding for temporal knowledge graph completion. In Proceedings of the Thirty-Fourth AAAI Conference on Artificial Intelligence.
Han, Z.; Chen, P.; Ma, Y.; and Tresp, V. 2021. Explainable subgraph reasoning for forecasting on temporal knowledge graphs. In Proceedings of the Ninth International Conference on Learning Representations.
Hildebrandt, M.; Sunder, S. S.; Mogoreanu, S.; Joblin, M.; Mehta, A.; Thon, I.; and Tresp, V. 2019. A recommender system for complex real-world applications with nonlinear dependencies and knowledge graph context. In Proceedings of the Sixteenth Extended Semantic Web Conference.
Jin, W.; Zhang, C.; Szekely, P.; and Ren, X. 2019. Recurrent event network for reasoning over temporal knowledge graphs. Workshop paper at the Seventh International Conference on Learning Representations.
Lacroix, T.; Obozinski, G.; and Usunier, N. 2020. Tensor decompositions for temporal knowledge base completion. In Proceedings of the Eighth International Conference on Learning Representations.
Leblay, J.; and Chekol, M. W. 2018. Deriving validity time in knowledge graph. In Companion Proceedings of the Web Conference 2018.
Liu, Y.; Hildebrandt, M.; Joblin, M.; Ringsquandl, M.; Raissouni, R.; and Tresp, V. 2021. Neural multi-hop reasoning with logical rules on biomedical knowledge graphs. In Proceedings of the Eighteenth Extended Semantic Web Conference.
Mahdavi, S.; Khoshraftar, S.; and An, A. 2018. dynnode2vec: scalable dynamic network embedding. In Proceedings of the 2018 IEEE International Conference on Big Data.
Meilicke, C.; Chekol, M. W.; Fink, M.; and Stuckenschmidt, H. 2020. Reinforced anytime bottom-up rule learning for knowledge graph completion. arXiv:2004.04412.
Meilicke, C.; Chekol, M. W.; Ruffinelli, D.; and Stuckenschmidt, H. 2019. Anytime bottom-up rule learning for knowledge graph completion. In Proceedings of the Twenty-Eighth International Joint Conference on Artificial Intelligence.
Nguyen, D. Q.; Nguyen, T. D.; Nguyen, D. Q.; and Phung, D. 2018a. A novel embedding model for knowledge base completion based on convolutional neural network. In Proceedings of the Sixteenth Annual Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies.
Nguyen, G. H.; Lee, J. B.; Rossi, R. A.; Ahmed, N. K.; Koh, E.; and Kim, S. 2018b. Dynamic network embeddings: from random walks to temporal random Walks. In Proceedings of the 2018 IEEE International Conference on Big Data.
Nickel, M.; Tresp, V.; and Kriegel, H.-P. 2011. A three-way model for collective learning on multi-relational data. In Proceedings of the Twenty-Eighth International Conference on Machine Learning.
Omran, P. G.; Wang, K.; and Wang, Z. 2019. Learning temporal rules from knowledge graph streams. In Proceedings of the AAAI 2019 Spring Symposium on Combining Machine Learning with Knowledge Engineering.
Schlichtkrull, M.; Kipf, T. N.; Bloem, P.; Berg, R. v. d.; Titov, I.; and Welling, M. 2018. Modeling relational data with graph convolutional networks. In Proceedings of the Fifteenth Extended Semantic Web Conference.
Trouillon, T.; Welbl, J.; Riedel, S.; Gaussier, É.; and Bouchard, G. 2016. Complex embeddings for simple link prediction. In Proceedings of the Thirty-Third International Conference on Machine Learning.
Vashishth, S.; Sanyal, S.; Nitin, V.; and Talukdar, P. 2020. Composition-based multi-relational graph convolutional networks. In Proceedings of the Eighth International Conference on Learning Representations.
Yang, B.; Yih, W.-T.; He, X.; Gao, J.; and Deng, L. 2015. Embedding entities and relations for learning and inference in knowledge bases. In Proceedings of the Third International Conference on Learning Representations.
Zhu, C.; Chen, M.; Fan, C.; Cheng, G.; and Zhang, Y. 2021. Learning from history: modeling temporal knowledge graphs with sequential copy-generation networks. In Proceedings of the Thirty-Fifth AAAI Conference on Artificial Intelligence.

## 參考文獻

Bian, R.; Koh, Y. S.; Dobbie, G.; and Divoli, A. 2019. Network embedding and change modeling in dynamic heterogeneous networks. In Proceedings of the Forty-Second International ACM SIGIR Conference on Research and Development in Information Retrieval.
Bordes, A.; Usunier, N.; García-Durán, A.; Weston, J.; and Yakhnenko, O. 2013. Translating embeddings for modeling multi-relational data. In Proceedings of the Twenty-Sixth International Conference on Neural Information Processing Systems.
Galárraga, L.; Teflioudi, C.; Hose, K.; and Suchanek, F. M. 2015. Fast rule mining in ontological knowledge bases with AMIE+. The VLDB Journal, 24: 707–730.
García-Durán, A.; Dumančić, S.; and Niepert, M. 2018. Learning sequence encoders for temporal knowledge graph completion. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing.
Goel, R.; Kazemi, S. M.; Brubaker, M.; and Poupart, P. 2020. Diachronic embedding for temporal knowledge graph completion. In Proceedings of the Thirty-Fourth AAAI Conference on Artificial Intelligence.
Han, Z.; Chen, P.; Ma, Y.; and Tresp, V. 2021. Explainable subgraph reasoning for forecasting on temporal knowledge graphs. In Proceedings of the Ninth International Conference on Learning Representations.
Hildebrandt, M.; Sunder, S. S.; Mogoreanu, S.; Joblin, M.; Mehta, A.; Thon, I.; and Tresp, V. 2019. A recommender system for complex real-world applications with nonlinear dependencies and knowledge graph context. In Proceedings of the Sixteenth Extended Semantic Web Conference.
Jin, W.; Zhang, C.; Szekely, P.; and Ren, X. 2019. Recurrent event network for reasoning over temporal knowledge graphs. Workshop paper at the Seventh International Conference on Learning Representations.
Lacroix, T.; Obozinski, G.; and Usunier, N. 2020. Tensor decompositions for temporal knowledge base completion. In Proceedings of the Eighth International Conference on Learning Representations.
Leblay, J.; and Chekol, M. W. 2018. Deriving validity time in knowledge graph. In Companion Proceedings of the Web Conference 2018.
Liu, Y.; Hildebrandt, M.; Joblin, M.; Ringsquandl, M.; Raissouni, R.; and Tresp, V. 2021. Neural multi-hop reasoning with logical rules on biomedical knowledge graphs. In Proceedings of the Eighteenth Extended Semantic Web Conference.
Mahdavi, S.; Khoshraftar, S.; and An, A. 2018. dynnode2vec: scalable dynamic network embedding. In Proceedings of the 2018 IEEE International Conference on Big Data.
Meilicke, C.; Chekol, M. W.; Fink, M.; and Stuckenschmidt, H. 2020. Reinforced anytime bottom-up rule learning for knowledge graph completion. arXiv:2004.04412.
Meilicke, C.; Chekol, M. W.; Ruffinelli, D.; and Stuckenschmidt, H. 2019. Anytime bottom-up rule learning for knowledge graph completion. In Proceedings of the Twenty-Eighth International Joint Conference on Artificial Intelligence.
Nguyen, D. Q.; Nguyen, T. D.; Nguyen, D. Q.; and Phung, D. 2018a. A novel embedding model for knowledge base completion based on convolutional neural network. In Proceedings of the Sixteenth Annual Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies.
Nguyen, G. H.; Lee, J. B.; Rossi, R. A.; Ahmed, N. K.; Koh, E.; and Kim, S. 2018b. Dynamic network embeddings: from random walks to temporal random Walks. In Proceedings of the 2018 IEEE International Conference on Big Data.
Nickel, M.; Tresp, V.; and Kriegel, H.-P. 2011. A three-way model for collective learning on multi-relational data. In Proceedings of the Twenty-Eighth International Conference on Machine Learning.
Omran, P. G.; Wang, K.; and Wang, Z. 2019. Learning temporal rules from knowledge graph streams. In Proceedings of the AAAI 2019 Spring Symposium on Combining Machine Learning with Knowledge Engineering.
Schlichtkrull, M.; Kipf, T. N.; Bloem, P.; Berg, R. v. d.; Titov, I.; and Welling, M. 2018. Modeling relational data with graph convolutional networks. In Proceedings of the Fifteenth Extended Semantic Web Conference.
Trouillon, T.; Welbl, J.; Riedel, S.; Gaussier, É.; and Bouchard, G. 2016. Complex embeddings for simple link prediction. In Proceedings of the Thirty-Third International Conference on Machine Learning.
Vashishth, S.; Sanyal, S.; Nitin, V.; and Talukdar, P. 2020. Composition-based multi-relational graph convolutional networks. In Proceedings of the Eighth International Conference on Learning Representations.
Yang, B.; Yih, W.-T.; He, X.; Gao, J.; and Deng, L. 2015. Embedding entities and relations for learning and inference in knowledge bases. In Proceedings of the Third International Conference on Learning Representations.
Zhu, C.; Chen, M.; Fan, C.; Cheng, G.; and Zhang, Y. 2021. Learning from history: modeling temporal knowledge graphs with sequential copy-generation networks. In Proceedings of the Thirty-Fifth AAAI Conference on Artificial Intelligence.

[Image]

## Supplementary Material

## 補充材料

### Dataset statistics

Table 4 shows the statistics of the three datasets ICEWS14, ICEWS18, and ICEWS0515. |X| denotes the cardinality of a set X.

### 資料集統計

表 4 顯示了三個資料集 ICEWS14、ICEWS18 和 ICEWS0515 的統計數據。|X| 表示集合 X 的基數。

[Image]

### Experimental details

All experiments were conducted on a Linux machine with 16 CPU cores and 32 GB RAM. The set of tested hyperparameter ranges and best parameter values for TLogic are displayed in Table 5. Due to memory constraints, the time window w for ICEWS18 is set to 200 and for ICEWS0515 to 1000. The best hyperparameter values are chosen based on the MRR on the validation set. Due to the small variance of our approach, the shown results are based on one algorithm run. A random seed of 12 is fixed for the rule learning component to obtain reproducible results.

### 實驗細節

所有實驗皆在配備 16 個 CPU 核心和 32 GB RAM 的 Linux 機器上進行。表 5 顯示了為 TLogic 測試的超參數範圍和最佳參數值。由於記憶體限制，ICEWS18 的時間視窗 w 設為 200，ICEWS0515 則設為 1000。最佳超參數值是根據驗證集上的 MRR 選擇的。由於我們方法的變異數很小，所示結果基於一次演算法運行。規則學習組件固定了隨機種子 12 以獲得可重現的結果。

[Image]

All results in the appendix refer to the validation set of ICEWS14. However, the observations are similar for the test set and the other two datasets. All experiments use the best set of hyperparameters, where only the analyzed parameters are modified.

附錄中的所有結果均參考 ICEWS14 的驗證集。然而，觀察結果對於測試集和其他兩個資料集是相似的。所有實驗都使用最佳的超參數集，其中只修改了被分析的參數。

### Object distribution baseline

We apply a simple object distribution baseline when there are no rules for the query relation or no matching body groundings in the graph. This baseline is only added for completeness and does not improve the results in a significant way.
The proportion of cases where there are no rules for the test query relation is 15/26,444 = 0.00056 for ICEWS14, 21/99,090 = 0.00021 for ICEWS18, and 9/138,294 = 0.00007 for ICEWS0515. The proportion of cases where there are no matching body groundings is 880/26,444 = 0.0333 for ICEWS14, 2,535/99,090 = 0.0256 for ICEWS18, and 2,375/138,294 = 0.0172 for ICEWS0515.

### 物件分佈基準

當查詢關係沒有規則或圖中沒有匹配的主體基礎時，我們應用一個簡單的物件分佈基準。這個基準只是為了完整性而添加，並不會顯著改善結果。
測試查詢關係沒有規則的案例比例為 ICEWS14 的 15/26,444 = 0.00056，ICEWS18 的 21/99,090 = 0.00021，以及 ICEWS0515 的 9/138,294 = 0.00007。沒有匹配主體基礎的案例比例為 ICEWS14 的 880/26,444 = 0.0333，ICEWS18 的 2,535/99,090 = 0.0256，以及 ICEWS0515 的 2,375/138,294 = 0.0172。

### Number of walks and transition distribution

Table 6 shows the results for different choices of numbers of walks and transition distributions. The performance for all metrics increases with the number of walks. Exponentially weighted transition always outperforms uniform sampling.

### 漫步次數與轉移分佈

表 6 顯示了不同漫步次數和轉移分佈選擇的結果。所有指標的性能都隨著漫步次數的增加而提升。指數加權轉移始終優於均勻抽樣。

[Image]

### Rule length

Table 7 indicates that using rules of all lengths for application results in the best performance. Learning only cyclic rules probably makes it more difficult to find rules of length 2, where the rule body must constitute a path with no recurring entities, leading to fewer rules and body groundings in the graph.

### 規則長度

表 7 指出，使用所有長度的規則進行應用會得到最佳性能。僅學習循環規則可能使得尋找長度為 2 的規則更加困難，因為規則體必須構成一條沒有重複實體的路徑，從而導致圖中規則和規則體基礎較少。

[Image]

### Time window

Generally, the larger the time window, the better the performance (see Table 8). If taking all previous timestamps leads to a too high memory usage, the time window should be decreased.

### 時間窗

一般來說，時間窗越大，性能越好（見表 8）。如果採用所有先前的時間戳會導致過高的記憶體使用量，則應縮小時間窗。

[Image]

### Score function

Using the best hyperparameters values for α and λ, Table 9 shows in the first row the results if only the rules' confidences are used for scoring, in the second row if only the exponential component is used, and in the last row the results for the combined score function. The combination yields the best overall performance. The optimal balance between the two terms, however, depends on the application and metric prioritization.

### 評分函數

使用 α 和 λ 的最佳超參數值，表 9 在第一列顯示了僅使用規則信賴度進行評分的結果，在第二列顯示了僅使用指數分量的結果，在最後一列顯示了組合評分函數的結果。組合產生了最佳的整體性能。然而，這兩項之間的最佳平衡取決於應用和指標的優先順序。

[Image]

### Rule learning

The figures 3 and 4 show the number of rules learned under the two transition distributions. The total number of learned rules is similar for the uniform and exponential distribution, but there is a large difference for rules of lengths 1 and 3. The exponential distribution leads to more successful longer walks and thus more longer rules, while the uniform distribution leads to a better exploration of the neighborhood around the start node for shorter walks.

### 規則學習

圖 3 和圖 4 顯示了在兩種轉移分佈下學習到的規則數量。均勻分佈和指數分佈學習到的規則總數相似，但長度為 1 和 3 的規則數量有很大差異。指數分佈導致更成功的長漫步，因此也產生更多長規則，而均勻分佈則能更好地探索起始節點周圍的鄰域，以進行較短的漫步。

[Image]

[Image]

### Training and inference time

The rule learning and rule application times are shown in the figures 5 and 6, dependent on the number of extracted temporal walks during learning.

### 訓練與推論時間

規則學習和規則應用時間如圖 5 和圖 6 所示，取決於學習期間提取的時間漫步次數。

[Image]

[Image]

The worst-case time complexity for learning rules of length l is O(|R|nlDb), where n is the number of walks, D the maximum node degree in the training set, and b the number of body samples for estimating the confidence.
The worst-case time complexity for inference is given by O(|G| + |TRrq|DL|E| log(k)), where L is the maximum rule length in TRrq and k the minimum number of candidates. More detailed steps of the algorithms for understanding these complexity estimations are given by Algorithm 3 and Algorithm 4.

學習長度為 l 的規則的最壞情況時間複雜度為 O(|R|nlDb)，其中 n 是漫步次數，D 是訓練集中的最大節點度數，b 是估計信賴度的規則體樣本數。
推論的最壞情況時間複雜度為 O(|G| + |TRrq|DL|E| log(k))，其中 L 是 TRrq 中的最大規則長度，k 是候選者的最小數量。演算法 3 和演算法 4 給出了理解這些複雜度估計的更詳細步驟。

[Image]

[Image]
