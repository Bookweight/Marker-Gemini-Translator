---
title: Evolutionary Graph-Updating Training
field: Knowledge_Graph
status: Imported
created_date: 2026-01-13
pdf_link: "[[Evolutionary Graph-Updating Training.pdf]]"
tags:
  - paper
  - knowledge_graph
---
# Abstract

Updating knowledge graphs typically relies on human efforts to collect and organize data, which makes the process time-consuming and limits the ability to incorporate new information in real-time. Moreover, omissions caused by human error or data loss may occur. Therefore, many link prediction models have been developed to predict missing or unknown links based on existing graphs. These methods enable graph completion to be performed automatically and accelerate the process of information updating.

# 摘要

知識圖譜的更新通常仰賴人力來收集與組織資料，這使得過程耗時且限制了即時整合新資訊的能力。此外，人為錯誤或資料遺失也可能導致疏漏。因此，許多基於現有圖譜來預測遺失或未知連結的連結預測模型應運而生。這些方法讓圖譜補全得以自動執行，並加速了資訊更新的過程。

However, some information is time-sensitive, which means its influence gradually diminishes over time. Although such information may remain factually correct, excessive reliance on it in some downstream tasks (such as recommendation and question-answering systems) probably leads to incorrect decisions. Therefore, enabling the model to estimate the effective lifetime of information and to selectively retain or discard it can improve information utilization and reduce errors arising from outdated references.

然而，部分資訊具有時間敏感性，意即其影響力會隨著時間推移而逐漸減弱。儘管這些資訊在事實上可能仍然正確，但在某些下游任務（例如推薦系統和問答系統）中過度依賴這些資訊，可能會導致不正確的決策。因此，讓模型能夠估算資訊的有效生命週期，並選擇性地保留或捨棄它，可以改善資訊利用率，並減少因參照過時資訊所產生的錯誤。

Therefore, this study proposes a framework that jointly models influence lifetime estimation and link prediction for knowledge graph updating. In addition to graph structure and semantic features, this framework captures temporal features from the temporal information associated with entities and links, enabling the model to determine both the existence and the influence lifetime of each link. In this way, influential links are retained during each update, while weakly influential information is removed, thereby ensuring the accuracy and freshness of the knowledge graph.

因此，本研究提出了一個框架，該框架聯合建模了影響力生命週期估算與連結預測，以進行知識圖譜的更新。除了圖譜結構和語意特徵外，此框架還從與實體和連結相關的時間資訊中擷取時間特徵，使模型能夠判斷每個連結的存在性及其影響力生命週期。如此一來，在每次更新期間，有影響力的連結會被保留，而影響力較弱的資訊則被移除，從而確保知識圖譜的準確性與新鮮度。

[Image]

# Chapter 1 Introduction

Knowledge Graphs (KGs) consist of collections of fact triplets, where each fact is represented as (entity, relation, entity). This structured representation provides rich information for various downstream tasks, such as recommendation and question answering, thereby assisting models in making more informed decisions.

# 第一章 緒論

知識圖譜 (Knowledge Graphs, KGs) 由事實三元組的集合所構成，其中每個事實表示為 (實體, 關係, 實體)。這種結構化表示為各種下游任務（如推薦和問答）提供了豐富的資訊，從而幫助模型做出更明智的決策。

However, updating knowledge graphs depends on human efforts to collect and organize data, making the process slow and difficult to add new information in real-time. Omissions due to human negligence or data loss probably also occur. Therefore, several link prediction models have been developed to learn from existing graph features and predict missing or unknown links, allowing graph completion to be performed automatically and accelerating the updating of knowledge graphs.

然而，更新知識圖譜有賴於人力收集和組織資料，使得過程緩慢且難以即時加入新資訊。人為疏忽或資料遺失也可能造成疏漏。因此，學界已開發出數種連結預測模型，從現有的圖譜特徵中學習並預測缺失或未知的連結，從而實現圖譜的自動補全，並加速知識圖譜的更新。

Since the emergence of neural networks, many studies have harnessed their powerful learning capabilities to develop enhanced link prediction models [1–10]. Furthermore, as graphs are usually continuously evolving, traditional transductive methods present challenges for effective prediction, prompting research into temporal graphs, dynamic graphs, and inductive settings [4,6–8].

自神經網路出現以來，許多研究已利用其強大的學習能力來開發增強的連結預測模型 [1-10]。此外，由於圖譜通常是持續演變的，傳統的直推式方法在有效預測方面面臨挑戰，這促使了對時間圖譜、動態圖譜和歸納式設定的研究 [4,6-8]。

In the inductive setting, emerging KGs consist of unseen entities and links between them and have no intersection with the original KG, also called disconnected emerging KGs (DEKGs). Most research aims to learn the structure and semantics of the original KG to predict the enclosing links between unseen entities in emerging KGs. However, in practical applications, some bridging links between the original KG and the emerging KG often exist. If this is not considered, some critical information is possible to be lost [9].

在歸納設定中，新興知識圖譜包含未見的實體及其之間的連結，且與原始知識圖譜沒有交集，也稱為不相連的新興知識圖譜 (disconnected emerging KGs, DEKGs)。大多數研究旨在學習原始知識圖譜的結構和語意，以預測新興知識圖譜中未見實體之間的封閉連結。然而，在實際應用中，原始知識圖譜與新興知識圖譜之間通常存在一些橋接連結。如果沒有考慮到這一點，一些關鍵資訊可能會遺失 [9]。

Although bridging links can be learned and predicted in the same way as enclosing links, the nodes of bridging links belong to two non-intersecting graphs, making it difficult to establish a subgraph for this link to learn structural features. Therefore, DEKG-ILP [9] is committed to capturing global semantic features through contrastive learning to compensate for the lack of structural features. Based on this, GSELI [10] combines personalized PageRank subgraph extraction and neighboring relational paths to enhance the model's ability to learn structural features.

儘管橋接連結可以像封閉連結一樣被學習和預測，但橋接連結的節點屬於兩個不相交的圖，這使得為該連結建立子圖以學習結構特徵變得困難。因此，DEKG-ILP [9] 致力於透過對比學習來捕捉全域語意特徵，以彌補結構特徵的不足。在此基礎上，GSELI [10] 結合了個人化的 PageRank 子圖提取和相鄰關係路徑，以增強模型學習結構特徵的能力。

While DEKG-ILP [9] and GSELI [10] consider the existence of bridging links, allowing the model to add more information in graph completion applications, they ignore that some information is time-sensitive. Although this information remains true after its influence lifetime, simply adding it to the knowledge graph and referring to outdated information in downstream tasks probably leads to incorrect judgments.

雖然 DEKG-ILP [9] 和 GSELI [10] 考慮了橋接連結的存在，允許模型在圖譜補全應用中添加更多資訊，但它們忽略了某些資訊具有時間敏感性。儘管這些資訊在其影響力生命週期過後仍然是真實的，但僅僅將其添加到知識圖譜中，並在下游任務中參考過時的資訊，很可能會導致不正確的判斷。

For example, information related to class suspension or disasters caused by a typhoon is most important and accurate during the typhoon period but gradually loses its importance afterward. If all such information is indiscriminately incorporated into the knowledge graph, downstream applications possibly subsequently rely on outdated or misleading information.

例如，與颱風造成的停課或災害相關的資訊在颱風期間最為重要和準確，但之後會逐漸失去其重要性。如果所有這些資訊都被不加區別地納入知識圖譜，下游應用程式隨後可能會依賴過時或誤導性的資訊。

However, this does not imply that all low-impact information should be deleted. Certain aspects, such as the structure of the typhoon, its trajectory, and related characteristics, can remain informative and valuable for future reference.

然而，這並不意味著所有低影響力的資訊都應該被刪除。某些方面，例如颱風的結構、其路徑以及相關特徵，對於未來的參考仍然可以提供資訊和價值。

Therefore, if the model is aware of the influence lifetime of information and retains it appropriately, it can increase the amount of usable information while avoiding erroneous results caused by referencing outdated knowledge.

因此，如果模型能夠意識到資訊的影響力生命週期並適當地保留它，就可以在避免因參考過時知識而導致錯誤結果的同時，增加可用資訊的數量。

Therefore, we propose a framework that builds upon GSELI [10] as the foundation for graph updating and incorporates influence lifetime estimation. Besides the structural and semantic features, the temporal features of the graph are learned based on the temporal information of entities and links in the graph to determine whether a link exists and how long it maintains a strong influence. During each update step, influential links are retained while low-influence information is removed, thereby ensuring the accuracy and novelty of the knowledge graph.

因此，我們提出了一個以 GSELI [10] 為基礎的圖更新框架，並納入了影響力生命週期估算。除了結構和語意特徵外，圖的時間特徵是基於圖中實體和連結的時間資訊來學習的，以確定連結是否存在以及它能維持多久的強大影響力。在每個更新步驟中，有影響力的連結會被保留，而低影響力的資訊會被移除，從而確保知識圖譜的準確性和新穎性。

The primary contributions of this work are summerized as follows:

本研究的主要貢獻總結如下：

* We propose a framework designed for continuous prediction. Through evolutionary graph-updating training, the model learns to handle dynamic or unknown graph structures. It enables the model to serve as an automatic graph completion tool, reducing the need for manual annotation.

* 我們提出了一個專為連續預測設計的框架。透過演化式圖更新訓練，模型學會處理動態或未知的圖結構。它使模型能夠作為一個自動圖補全工具，減少了手動標註的需求。

* Typical temporal graph reasoning only considers fixed entity and relation sets. We extend the semi-inductive setting to temporal knowledge graphs, enabling the model to handle emerging entities and relations, making it more suitable for real-world applications.

* 典型的時間圖譜推理只考慮固定的實體和關係集合。我們將半歸納式設定擴展到時間知識圖譜，使模型能夠處理新興的實體和關係，使其更適合真實世界的應用。

* We design two graph updating strategies, which can be adopted for different scenarios. By retaining complete information or removing redundant information, it can maintain accuracy in various situations.

* 我們設計了兩種圖更新策略，可適用於不同情境。透過保留完整資訊或移除冗餘資訊，它可以在各種情況下保持準確性。

The rest of this paper is organized as follows: Chapter 2 discusses previous related works, Chapter 3 defines the problem we want to solve, Chapter 4 introduces our proposed method, Chapter 5 describes our experimental setup and results, and Chapter 6 summarizes the conclusion.

本文其餘部分的組織如下：第二章討論先前的相關研究，第三章定義我們想要解決的問題，第四章介紹我們提出的方法，第五章描述我們的實驗設定和結果，第六章總結結論。

[Image]

# Chapter 2 Related Works

This section summarizes some outstanding works from the past and introduces them according to their application scenarios.

# 第二章 相關研究

本節總結了過去一些傑出的研究，並根據其應用場景進行介紹。

## 2.1 Transductive Link Prediction

Transductive link prediction aims to predict missing links in graphs. In this situation, all entities are known, so the key challenge is in effectively predicting the existence of links based on structural, semantic, and other information in the graph. Traditional heuristic methods can accurately predict, but rely on human design, which makes it hard to automate. Therefore, SEAL [1] uses a neural network to automatically determine the appropriate heuristic function based on the graph. Combining the accuracy of heuristic methods with the power of neural networks, the model's generality is increased. RGCN [3] considers the diversity of relations in a graph, it aggregates information according to relation, and assigns different weights. In this way, the model can fully utilize relation information to learn node representations.

## 2.1 直推式連結預測

直推式連結預測旨在預測圖中的遺失連結。在這種情況下，所有實體都是已知的，因此關鍵挑戰在於根據圖中的結構、語意和其他資訊，有效地預測連結的存在。傳統的啟發式方法可以準確預測，但依賴於人工設計，難以自動化。因此，SEAL [1] 使用神經網路根據圖自動確定適當的啟發式函數。結合啟發式方法的準確性和神經網路的強大功能，模型的通用性得到了提高。RGCN [3] 考慮了圖中關係的多樣性，它根據關係聚合資訊，並分配不同的權重。透過這種方式，模型可以充分利用關係資訊來學習節點表示。

## 2.2 Inductive Link Prediction

While many studies have demonstrated excellent performance in transductive link prediction, in the real world, the entity set in a graph is usually not fixed. New entities are added over time, and these entities often contain relatively little information. Therefore, models that focus on node information are not very effective. Although the problem can be solved via re-training, it increases the model's training cost. Furthermore, if a model can be applied to different graphs and then fine-tuned, the training cost can be reduced significantly. Consequently, many studies have focused on inductive link prediction.

## 2.2 歸納式連結預測

儘管許多研究在直推式連結預測方面表現出色，但在現實世界中，圖中的實體集合通常不是固定的。隨著時間的推移，新的實體會被加入，而這些實體通常只包含相對較少的資訊。因此，專注於節點資訊的模型效果不佳。雖然這個問題可以透過重新訓練來解決，但這會增加模型的訓練成本。此外，如果一個模型可以應用於不同的圖，然後進行微調，訓練成本就可以顯著降低。因此，許多研究都集中在歸納式連結預測上。

GraIL [4] first proposed an inductive setting, aiming to learn logical rules independent of node semantics. This allows the model to focus on capturing structural information to establish entity-independent relation representations, achieving excellent results in predicting links between unseen entities.

GraIL [4] 首次提出了一種歸納式設定，旨在學習獨立於節點語意的邏輯規則。這使得模型能夠專注於捕捉結構資訊，以建立與實體無關的關係表示，在預測未見實體之間的連結方面取得了優異的成果。

## 2.3 Semi-Inductive Link Prediction

However, in addition to links between known entities or unknown entities, there are also links between known and unknown entities. If the graph is simply divided into two graphs, consisting of known entities and consisting of unknown entities, these links between known and unknown entities, which are called bridge links, will be ignored. Therefore, semi-inductive link prediction, which handles both transductive and inductive link prediction, has become a focus in recent years.

## 2.3 半歸納式連結預測

然而，除了已知實體或未知實體之間的連結外，也存在已知實體與未知實體之間的連結。如果圖被簡單地劃分為兩個圖，一個由已知實體組成，另一個由未知實體組成，那麼這些被稱為橋接連結的、介於已知和未知實體之間的連結將被忽略。因此，處理直推式和歸納式連結預測的半歸納式連結預測，近年來已成為研究焦點。

MV-HRE [5] considers the diversity of nodes and relations in heterogeneous graphs, as well as the problem of data imbalance between categories. It uses multiple aspects of information to learn, including subgraph aspect, metapath aspect, and community aspect, to assist categories with few data. Meta-iKG [11] also takes similar issues into account. This work aims to strengthen new entities and relations, or rarely appearing relations. It divides relations into large-shot relations and few-shot relations according to their frequency of appearance, that is, the amount of available data, and uses meta-learning to allow large-shot relations to assist in the learning of few-shot relations.

MV-HRE [5] 考慮了異構圖中節點和關係的多樣性，以及類別間資料不平衡的問題。它使用多方面的資訊進行學習，包括子圖、元路徑和社群等方面，以輔助資料稀少的類別。Meta-iKG [11] 也考慮了類似的問題。這項工作旨在加強新實體和關係，或罕見出現的關係。它根據關係出現的頻率，也就是可用資料的數量，將關係劃分為多樣本 (large-shot) 關係和少樣本 (few-shot) 關係，並使用元學習讓多樣本關係輔助少樣本關係的學習。

Unseen nodes often cannot effectively predict their connections due to a lack of information, so relation embeddings are used to construct unseen node representations. DEKG-ILP [9] uses the relation composition of each node for contrastive learning to make the semantics embeddings of the relation more accurate, thereby establishing a better node representation. It also learns topological features and considers the semantic and topological information for prediction. GSELI [10] adds personalized PageRank subgraph extraction to DEKG-ILP [9], allowing the model to obtain subgraphs with closer connections when extracting for each triplet. To contruct better initial embedding for nodes, GSELI [10] combines neighboring relational path modeling proposed by SNRI [12], uses a self-attention mechanism to learn structure-based relation embeddings. It also uses GRU to learn the contextual relationships of relations to obtain better structural features, thereby improving model performance.

由於資訊不足，未見節點通常無法有效地預測其連結，因此使用關係嵌入來建構未見節點的表示。DEKG-ILP [9] 使用每個節點的關係組合物進行對比學習，以使關係的語意嵌入更準確，從而建立更好的節點表示。它還學習拓撲特徵，並考慮語意和拓撲資訊進行預測。GSELI [10] 在 DEKG-ILP [9] 中加入了個人化的 PageRank 子圖提取，使模型在為每個三元組提取時能獲得連結更緊密的子圖。為了為節點建構更好的初始嵌入，GSELI [10] 結合了 SNRI [12] 提出的相鄰關係路徑模型，使用自註意力機制來學習基於結構的關係嵌入。它還使用 GRU 來學習關係的上下文關係，以獲得更好的結構特徵，從而提高模型性能。

## 2.4 Temporal Link Prediction

In the real world, the emergence of entities and relations and the evolution of graphs occur over time, which is not considered in works on static graphs. To be more fit with practice applications, some works focus on temporal link prediction, aiming to capture evolution patterns of graphs. In order to model time-varying graph structures, DySAT [7] generates dynamic node representations through joint self-attention to increase accuracy and flexibility of the model. On the other hand, CoEvoGNN [8] learns the covariance between node attributes and the overall structure of graphs to capture the mutual influence between them. TCDGE [13] proposes ToE (Timespans of Edge formation), which presents how long it takes for a triplet to form from being able to form, and combines with matrix factorization to preserve the temporal correlation between nodes.

## 2.4 時間連結預測

在現實世界中，實體和關係的出現以及圖的演化是隨時間發生的，這在關於靜態圖的研究中並未被考慮。為了更貼近實際應用，一些研究專注於時間連結預測，旨在捕捉圖的演化模式。為了對時變圖結構進行建模，DySAT [7] 透過聯合自註意力生成動態節點表示，以提高模型的準確性和靈活性。另一方面，CoEvoGNN [8] 學習節點屬性與圖整體結構之間的協方差，以捕捉它們之間的相互影響。TCDGE [13] 提出了 ToE (邊形成的時長)，它表示一個三元組從能夠形成到實際形成所需的時間，並與矩陣分解相結合，以保留節點之間的時間相關性。

Since the information imbalance that results from entities appearing at different times and frequencies, RE-GCN [14] integrates static and dynamic features to mitigate this limitation. It captures sequential patterns across timestamps and incorporates the static properties simultaneously to utilize two features effectively. CorDGT [15] proposes a new approach to extract high-order proximity to obtain comprehensive features, and uses the self-attention mechanism to enhance the expressive power.

由於實體在不同時間和頻率出現所導致的資訊不平衡，RE-GCN [14] 整合了靜態和動態特徵以減輕此限制。它捕捉跨時間戳的序列模式，並同時納入靜態屬性，以有效利用這兩種特徵。CorDGT [15] 提出了一種新方法，以提取高階鄰近性來獲得全面的特徵，並使用自註意力機制來增強表達能力。

However, most research assumes that all possible entities and relations are known, and can not really deal with unseen entities and relations. Thus, how to address emerging entities and relations is still an issue worth exploring.

然而，大多數研究假設所有可能的實體和關係都是已知的，無法真正處理未見的實體和關係。因此，如何處理新興的實體和關係仍然是一個值得探討的議題。

[Image]

# Chapter 3 Problem Statement

In this section, we present some definitions used throughout this work and formally state the problem that our model aims to solve.

# 第三章 問題陳述

在本節中，我們將介紹本研究中使用的一些定義，並正式陳述我們的模型旨在解決的問題。

Knowledge graphs containing existing facts are widely used in several applications, such as question answering and information retrieval. However, those graphs usually record what happened without when it happened, which is not enough to learn and predict the evolution of graphs. Therefore, temporal knowledge graphs that contain temporal information are needed to capture the evolving patterns for predicting future events.

包含現有事實的知識圖譜廣泛應用於多種應用，例如問答和資訊檢索。然而，這些圖譜通常只記錄了「發生了什麼」，而沒有記錄「何時發生」，這不足以學習和預測圖譜的演變。因此，需要包含時間資訊的時間知識圖譜來捕捉不斷演變的模式，以預測未來的事件。

* **Temporal Knowledge Graph**
A temporal knowledge graph *G*<sub>*t*</sub> is a collection of facts, denoted as *G*<sub>*t*</sub> = {(*u*,*r*,*v*,*t*<sub>1</sub>) | *u*,*v* ∈ *E*<sub>*t*</sub>, *r* ∈ *R*<sub>*t*</sub>, 0 ≤ *t*<sub>1</sub> ≤ *t* ≤ *T*}, where *E*<sub>*t*</sub> and *R*<sub>*t*</sub> denote the entity and relation sets at *t*, *t*<sub>1</sub> is the timestamp of each triplet formed, *t* is the last timestamp in the graph, and *T* is the last timestamp in the dataset.

* **時間知識圖譜**
一個時間知識圖譜 *G*<sub>*t*</sub> 是一系列事實的集合，表示為 *G*<sub>*t*</sub> = {(*u*,*r*,*v*,*t*<sub>1</sub>) | *u*,*v* ∈ *E*<sub>*t*</sub>, *r* ∈ *R*<sub>*t*</sub>, 0 ≤ *t*<sub>1</sub> ≤ *t* ≤ *T*}，其中 *E*<sub>*t*</sub> 和 *R*<sub>*t*</sub> 代表在時間 *t* 的實體與關係集合，*t*<sub>1* 是每個形成的三元組的時間戳，*t* 是圖中的最後一個時間戳，而 *T* 是資料集中的最後一個時間戳。

Note that each link, originally represented as a triplet, is further annotated with a timestamp. As links are constantly created over time, new entities also emerge, such as a newly-debuted athlete or a newly founded company. These form the main components of graph evolution.

請注意，每個最初表示為三元組的連結，都會進一步用時間戳進行標註。隨著時間的推移，連結不斷被創建，新的實體也會出現，例如新出道的運動員或新成立的公司。這些構成了圖演化的主要組成部分。

* **Emerging Entity Set**
In practical applications, temporal knowledge graphs are dynamic, with emerging entities *E*<sub>*t*</sub>′, where *E*<sub>*t*</sub>′ ∩ *E*<sub>*t*−1</sub> = ∅, continuously emerging over time, leading to differences between entity sets of *G*<sub>*t*−1</sub> and *G*<sub>*t*</sub>.

* **新興實體集**
在實際應用中，時間知識圖譜是動態的，會隨著時間不斷出現新興實體 *E*<sub>*t*</sub>′，其中 *E*<sub>*t*</sub>′ ∩ *E*<sub>*t*−1</sub> = ∅，這導致了 *G*<sub>*t*−1</sub> 和 *G*<sub>*t*</sub> 的實體集之間存在差異。

However, these emerging and previously unseen entities often lack connections to existing entities. Such cases may occur when links have not yet been established, are present but unobserved, or are difficult to verify their existence, as demonstrated in the following two examples.

然而，這些新興且先前未見的實體通常缺乏與現有實體的連結。這種情況可能發生在連結尚未建立、存在但未被觀察到，或難以驗證其存在時，如下面兩個例子所示。

* **Example 1**
In a knowledge graph that records sporting events, entities such as players, teams, sponsors, and events are connected through links representing relationships like player signing with a team or team participation in competitions. For sports news media, the events in which players or teams will participate are of primary interest, whereas teams are also concerned with player signings and sponsorships involving other teams. When new players emerge, both media organizations and teams actively monitor their movements in order to predict, as early as possible, which team they will join.

* **範例 1**
在一個記錄體育賽事的知識圖譜中，諸如球員、球隊、贊助商和賽事等實體，是透過代表關係的連結（例如球員與球隊簽約或球隊參加比賽）來連接的。對體育新聞媒體而言，球員或球隊將參加的賽事是主要關注點，而球隊也關心涉及其他球隊的球員簽約和贊助。當新球員出現時，媒體組織和球隊都會積極監控他們的動向，以便盡早預測他們將加入哪支球隊。

* **Example 2**
In a protein-protein interaction network, proteins are modeled as entities and their interactions are regarded as links. When biologists discover a new protein, identifying its interaction partners and interaction mechanisms as early as possible is highly desirable, but experimental validation is costly and time-consuming. Predicting potential interactions can narrow down the scope of verification.

* **範例 2**
在蛋白質交互作用網絡中，蛋白質被建模為實體，其交互作用被視為連結。當生物學家發現一種新蛋白質時，盡早確定其交互作用夥伴和交互作用機制是高度期望的，但實驗驗證既昂貴又耗時。預測潛在的交互作用可以縮小驗證的範圍。

In such cases, obtaining accurate connections is often difficult and costly. Therefore, predicting link existence through models can significantly reduce costs and enable real-time graph updates. Based on the above discussion, our main objective is defined as follows.

在這種情況下，獲得準確的連結通常是困難且昂貴的。因此，透過模型預測連結的存在可以顯著降低成本並實現即時圖更新。基於以上討論，我們的主要目標定義如下。

* **Problem Formulation**
At timestamp *t*, given the previous temporal knowledge graph *G*<sub>*t*−1</sub> and an emerging entity set *E*<sub>*t*</sub>′, the model aims to predict the content of the graph *G*<sub>*t*</sub>. This involves determing the appearance of new links in *S*<sub>*t*</sub> = *E*<sub>*t*</sub> × *R*<sub>*t*</sub> × *E*<sub>*t*</sub> – *G*<sub>*t*−1</sub> and the disappearance of existing links in *G*<sub>*t*−1</sub>, where *E*<sub>*t*</sub> = *E*<sub>*t*−1</sub> ∪ *E*<sub>*t*</sub>′ and *R*<sub>*t*</sub> are the entity and relation sets at *t*.

* **問題定義**
在時間戳 *t*，給定先前的時間知識圖譜 *G*<sub>*t*−1</sub> 和一個新興實體集 *E*<sub>*t*</sub>′，模型的目標是預測圖 *G*<sub>*t*</sub> 的內容。這包括確定在新連結集合 *S*<sub>*t*</sub> = *E*<sub>*t*</sub> × *R*<sub>*t*</sub> × *E*<sub>*t*</sub> – *G*<sub>*t*−1</sub> 中新連結的出現，以及在 *G*<sub>*t*−1</sub> 中現有連結的消失，其中 *E*<sub>*t*</sub> = *E*<sub>*t*−1</sub> ∪ *E*<sub>*t*</sub>′ 且 *R*<sub>*t*</sub> 為在時間 *t* 的實體與關係集合。

Moreover, most graph reasoning methods focus on seen entities [1,3, 14]. While some studies consider emerging entities and aim to improve model performance in semi-inductive settings [5,9,10,15], unseen relations are generally not considered. For instance, novel functions on social media, such as “virtual gifting" or "co-streaming", introduce new relations that were absent in the past. In these cases, models mentioned above either ignore triplets related to emerging relations or mispredict them as other known relations, both of which limit the scope of application scenarios. To increase model flexibility, we allow the model to deal with not only unseen entities but also unseen relations. Due to the cold-start problem of emerging relations, it is difficult to embed each emerging relation individually.

此外，大多數圖推理方法都專注於已見實體 [1,3, 14]。雖然一些研究考慮了新興實體並旨在提高半歸納式設定 [5,9,10,15] 中的模型性能，但通常不考慮未見關係。例如，社交媒體上的新功能，如「虛擬禮物」或「共同直播」，引入了過去所沒有的新關係。在這些情況下，上述模型要麼忽略與新興關係相關的三元組，要麼將它們誤判為其他已知關係，這兩種情況都限制了應用場景的範圍。為了增加模型的靈活性，我們允許模型不僅處理未見實體，還處理未見關係。由於新興關係的冷啟動問題，很難單獨嵌入每個新興關係。

If relation embeddings are suboptimal or unavailable, it will affect the entity embeddings constructed from relation embeddings, and thus affect the model performance. Therefore, we aggregate these emerging relations and model them jointly. Specifically, all emerging relations are categorized into a single representation, denoted as *r*<sup>*u*</sup>, and the model can predict links even when the relations were unseen during training.

如果關係嵌入是次優的或不可用的，它將影響從關係嵌入建構的實體嵌入，進而影響模型性能。因此，我們將這些新興關係聚合起來並聯合建模。具體來說，所有新興關係都被歸類為單一表示，記為 *r*<sup>*u*</sup>，即使在訓練期間關係是未見的，模型也可以預測連結。

In summary, we elucidate the definitions and challenges of emerging entities and relations and are dedicated to addressing these obstacles. To predict the graph *G*<sub>*t*</sub> from the previous graph *G*<sub>*t*−1</sub> under the challenges mentioned, we design a framework to model the evolution patterns of the graph, proposed in the subsequent section.

總之，我們闡明了新興實體和關係的定義與挑戰，並致力於解決這些障礙。為了在上述挑戰下從先前的圖 *G*<sub>*t*−1</sub> 預測圖 *G*<sub>*t*</sub>，我們設計了一個框架來模擬圖的演化模式，並在下一節中提出。

[Image]

# Chapter 4 Methodology

In real-world scenarios, facts continuously emerge and some of them become obsolete after a period of time. To speed up the update of graphs and reduce the effort of manual updates, we propose a framework named EvoGUT (Evolutionary Graph-Updating Training), which is designed for temporal link prediction and knowledge graph auto-updating, as illustrated in Figure 4.1.

# 第四章 方法論

在真實世界的情境中，事實不斷地出現，其中一些在一段時間後會變得過時。為了加速圖的更新並減少手動更新的工作量，我們提出了一個名為 EvoGUT (Evolutionary Graph-Updating Training) 的框架，該框架專為時間連結預測和知識圖譜自動更新而設計，如圖 4.1 所示。

[Image]
Figure 4.1: Framework of EvoGUT.

圖 4.1: EvoGUT 框架圖。

Given the base knowledge graph *G*<sub>*t*−1</sub>, emerging entity set *E*<sub>*t*</sub>′, EvoGUT aims to automatically update graphs via predicting existence and estimating the influence lifetime for each possible link, including new links in *S*<sub>*t*</sub> and existing links in *G*<sub>*t*−1</sub>. Firstly, LEP (Link Existence Predictor) integrates semantic, topological, and temporal features to estimate the likelihood of link existence and the influence lifetime of links from three complementary perspectives. EvoGUT updates the base knowledge graph with two updating strategies to get two candidate graphs, *G*<sub>*t*</sub><sup>*A*</sup> and *G*<sub>*t*</sub><sup>*P*</sup>. Finally, the next knowledge graph *G*<sub>*t*</sub> is selected from these two graphs based on the model configuration.

給定基礎知識圖譜 *G*<sub>*t*−1</sub>、新興實體集 *E*<sub>*t*</sub>′，EvoGUT 旨在透過預測每個可能連結的存在性並估計其影響壽命來自動更新圖譜，包括 *S*<sub>*t*</sub> 中的新連結和 *G*<sub>*t*−1</sub> 中的現有連結。首先，LEP (Link Existence Predictor) 整合語意、拓撲和時間特徵，從三個互補的角度估計連結存在的可能性和連結的影響壽命。EvoGUT 使用兩種更新策略更新基礎知識圖譜，以獲得兩個候選圖譜 *G*<sub>*t*</sub><sup>*A*</sup> 和 *G*<sub>*t*</sub><sup>*P*</sup>。最後，根據模型配置從這兩個圖譜中選擇下一個知識圖譜 *G*<sub>*t*</sub>。

## 4.1 Evolutionary Graph-Updating Training

Existing temporal link prediction methods, which focus on the transductive setting that assumes entities and relations are fixed, are unable to handle new content appearing over time. While semi-inductive link prediction models can handle this scenario, they seldom consider temporal information and graph evolution patterns, making it difficult to update graphs accurately. To automatically update knowledge graphs more precisely, we propose a framework that can train the model on how to update graphs and generate two version of updated graphs according to different strategies for processing redundant information.

## 4.1 演化式圖更新訓練

現有的時間連結預測方法專注於直推式設定，假設實體和關係是固定的，因此無法處理隨時間出現的新內容。雖然半歸納式連結預測模型可以處理這種情況，但它們很少考慮時間資訊和圖演化模式，因此很難準確地更新圖。為了更精確地自動更新知識圖譜，我們提出了一個框架，可以訓練模型如何更新圖，並根據處理冗餘資訊的不同策略生成兩種版本的更新圖。

Firstly, we split both the training set and validation set into two parts, pre-training and graph-updating training (GUT), as illustrated in Figure 4.2.

首先，我們將訓練集和驗證集都分成兩部分：預訓練和圖更新訓練 (GUT)，如圖 4.2 所示。

[Image]
Figure 4.2: The dataset split of EvoGUT.

圖 4.2: EvoGUT 的資料集分割。

The first part of the training set and the validation set are used as static graphs for the initial pre-training, while the second parts are used to train the model on updating graph. In this process, the model explicitly learns graph evolution across timestamps, whereas most existing temporal graph reasoning methods typically learn from independent snapshots at each timestamp, as shown in Figure 4.3.

訓練集和驗證集的第一部分用作初始預訓練的靜態圖，而第二部分則用於訓練模型更新圖。在此過程中，模型明確地學習跨時間戳的圖演化，而大多數現有的時間圖推理方法通常是從每個時間戳的獨立快照中學習，如圖 4.3 所示。

[Image]
Figure 4.3: Training flow comparison between EvoGUT and typical models.

圖 4.3: EvoGUT 與典型模型的訓練流程比較。

Initially, the model is trained on the static graph to learn static features of semantics and topology. After this pre-training step, we train the model on how to update graphs. As shown in Figure 4.1, for a base graph *G*<sub>*t*-1</sub>, the model generates a candidate link set *S*<sub>*t*</sub>′ = {*l* | *l* ∈ *S*<sub>*t*</sub> ∧ *ϕ*(*l*) > *θ*}, where *ϕ*(*l*) is the score of link *l* given by the model, *θ* is the threshold decided by the model to determine link's existence. Then the model updates *G*<sub>*t*−1</sub> based on the following strategies:

最初，模型在靜態圖上進行訓練，以學習語意和拓撲的靜態特徵。在預訓練步驟之後，我們訓練模型如何更新圖。如圖 4.1 所示，對於一個基礎圖 *G*<sub>*t*-1</sub>，模型會產生一個候選連結集 *S*<sub>*t*</sub>′ = {*l* | *l* ∈ *S*<sub>*t*</sub> ∧ *ϕ*(*l*) > *θ*}，其中 *ϕ*(*l*) 是模型給予連結 *l* 的分數，*θ* 是模型決定連結是否存在的閾值。然後，模型會根據以下策略更新 *G*<sub>*t*−1</sub>：

Accumulation: *G*<sub>*t*</sub><sup>*A*</sup> = *G*<sub>*t*−1</sub> ∪ *S*<sub>*t*</sub>′, (4.1)

累積：*G*<sub>*t*</sub><sup>*A*</sup> = *G*<sub>*t*−1</sub> ∪ *S*<sub>*t*</sub>′, (4.1)

Pruning: *G*<sub>*t*</sub><sup>*P*</sup> = {*l* | *l* ∈ *G*<sub>*t*</sub><sup>*A*</sup> ∧ *t*<sub>*l*</sub><sup>*e*</sup> > *t*}, (4.2)
where *t*<sub>*l*</sub><sup>*e*</sup> is the timestamp representing influence lifetime of link *l*.

修剪：*G*<sub>*t*</sub><sup>*P*</sup> = {*l* | *l* ∈ *G*<sub>*t*</sub><sup>*A*</sup> ∧ *t*<sub>*l*</sub><sup>*e*</sup> > *t*}，(4.2)
其中 *t*<sub>*l*</sub><sup>*e*</sup> 是表示連結 *l* 影響力生命週期的時間戳。

In the accumulation strategy, all links whose scores satisfy a predefined threshold will be added to the graph, ensuring all existing information is retained regardless of its influence. In contrast, the pruning strategy jointly considers link scores and influence lifetimes, retaining only those links whose influence persists into the subsequent timestamp. The choice between Graph *G*<sub>*t*</sub><sup>*A*</sup> and *G*<sub>*t*</sub><sup>*P*</sup> as the input graph for the next iteration is application-dependent. For instance, in some question answering systems, it only requires information that is still valid at the current time, so the pruning strategy is preffered. As for graph completion, since it aims to record facts, it places more emphasis on information's existence rather than influence, which is more suitable for using the accumulation strategy.

在累積策略中，所有分數滿足預定閾值的連結都將被添加到圖中，確保所有現有資訊無論其影響力如何都被保留。相反地，修剪策略同時考慮連結分數和影響力生命週期，僅保留那些影響力持續到下一個時間戳的連結。在圖 *G*<sub>*t*</sub><sup>*A*</sup> 和 *G*<sub>*t*</sub><sup>*P*</sup> 之間選擇何者作為下一次迭代的輸入圖取決於應用。例如，在某些問答系統中，它只需要在當前時間仍然有效的資訊，因此首選修剪策略。至於圖補全，由於其旨在記錄事實，因此更強調資訊的存在而非影響力，這更適合使用累積策略。

By repeating these steps, the model prunes and adds triplets by minimizing the total loss, enabling it to determine the best way of updating graphs.

透過重複這些步驟，模型透過最小化總損失來修剪和添加三元組，使其能夠確定更新圖的最佳方式。

## 4.2 Link Existence Predictor

Link Existence Predictor (LEP) integrates semantic, topological, and temporal features to predict links’ existence and its influence lifetime, as shown in Figure 4.4. Semantic features are learned via Contrastive Learning-based global Semantic Feature modeling (CLSF), which enhances the model’s ability to express fine-grained relational semantics. Topological features are extracted through GNN-based Enhanced Local Subgraph modeling (GELS), capturing local structural relevancies within the graph. Both proposed by DEKG-ILP [9] and GSELI [10] have been performing well in unseen-entity-contained scenarios. Besides, to fully absorb the features of both sides, the relation embeddings are shared between these two aspects. So that the semantic information can be broadcast by GNN to enrich entity embeddings. Meanwhile, the topological information that implies relative location and connection information between relations can be helpful while learning semantic features. Furthermore, since most temporal knowledge graph reasoning works are limited to transductive settings that need to know all entities and relations, this leads to poor generality. In order to extend semi-inductive and inductive settings’ model to temporal knowledge graphs, temporal features are modeled to not only predict existence but also estimate the influence lifetime of each link—a direction rarely explored in prior temporal knowledge graph studies. Firstly, LEP gives each link a score that reflects its likelihood of existence and an influence lifetime to reflect the time period during which the link remains informative and impactful for downstream tasks.

## 4.2 連結存在性預測器

連結存在性預測器 (LEP) 整合了語意、拓撲和時間特徵，以預測連結的存在性及其影響力生命週期，如圖 4.4 所示。語意特徵是透過基於對比學習的全域語意特徵模型 (CLSF) 來學習的，該模型增強了模型表達細粒度關係語意的能力。拓撲特徵是透過基於 GNN 的增強型局部子圖模型 (GELS) 來提取的，捕捉圖內的局部結構相關性。由 DEKG-ILP [9] 和 GSELI [10] 提出的這兩種方法在包含未見實體的場景中都表現良好。此外，為了充分吸收雙方的特徵，關係嵌入在這兩個方面之間共享。這樣，語意資訊可以透過 GNN 廣播以豐富實體嵌入。同時，隱含關係之間相對位置和連結資訊的拓撲資訊在學習語意特徵時會有所幫助。此外，由於大多數時間知識圖譜推理工作都僅限於需要知道所有實體和關係的直推式設定，這導致了較差的通用性。為了將半歸納式和歸納式設定的模型擴展到時間知識圖譜，時間特徵的建模不僅用於預測存在性，還用於估計每個連結的影響力生命週期——這在先前的時間知識圖譜研究中很少被探討。首先，LEP 為每個連結提供一個反映其存在可能性的分數和一個影響力生命週期，以反映該連結在下游任務中保持資訊性和影響力的時間段。

[Image]
Figure 4.4: Schematic of Link Existence Predictor.

圖 4.4: 連結存在性預測器示意圖。

### 4.2.1 Semantic and Topological Embedding Learning

Semantic features constitute information in the knowledge graph, providing complementary perspectives to topological features. However, in inductive and semi-inductive scenarios, most or all entities are unseen during training, making it difficult to construct their semantic representations directly. To overcome the obstacle, most studies derive the semantic features of an entity from the relation features associated with that entity, lead to the importance of high-quality relation embeddings.

### 4.2.1 語意與拓撲嵌入學習

語意特徵構成知識圖譜中的資訊，為拓撲特徵提供互補的視角。然而，在歸納和半歸納場景中，大多數或所有實體在訓練期間都是未見的，這使得直接建構其語意表示變得很困難。為了克服這個障礙，大多數研究從與該實體相關的關係特徵中推導出實體的語意特徵，這導致了高品質關係嵌入的重要性。

For learning the comprehensive relation features, we refer to Contrastive Learning-based Global Semantic Feature Modeling (CLSF) proposed by DEKG-ILP [9]. This module constructs entity semantic feature *e*<sub>*i*</sub> by fusing relation embeddings based on the composition of entity *i*, and calculate Euclidean distance as the measure to use contrastive learning to enhance the relation embeddings.

為了學習全面的關係特徵，我們參考了 DEKG-ILP [9] 中提出的基於對比學習的全域語意特徵模型 (CLSF)。該模塊透過融合基於實體 *i* 組成的關係嵌入來建構實體語意特徵 *e*<sub>*i*</sub>，並計算歐幾里得距離作為度量，以使用對比學習來增強關係嵌入。

Although Euclidean distance is a clear and widely used measure, it is possible to be insufficient in some situations, such as when the number of links between relations of a sample is highly imbalanced. As the example illustrated in Table 4.1, the module first generates positive and negative smaples via data augmentation and then normalizes them to eliminate bias caused by numbers of links. Next, it calculates the distances between the original sample and the augmented samples, denoted as *d*<sup>pos</sup> and *d*<sup>neg</sup>, respectively. The goal is to minimize *d*<sup>pos</sup> and maximize *d*<sup>neg</sup>, making the original sample similar to the positive one and dissimilar to the negative one. In this example, after data augmentation, the positive sample contains substantially more *r*<sub>3</sub> links than the original sample. This imbalance causes *d*<sup>pos</sup> to become close to *d*<sup>neg</sup>. In this situation, the module needs to adjust relation embeddings to maintain a margin between *d*<sup>pos</sup> and *d*<sup>neg</sup>, which probably compromises the ability of the embeddings to effectively represent relational information.

儘管歐幾里得距離是一個清晰且廣泛使用的度量，但在某些情況下可能不足，例如當樣本關係之間的連結數量高度不平衡時。如表 4.1 所示的例子，該模塊首先通過數據增強生成正樣本和負樣本，然後對其進行歸一化以消除由連結數量引起的偏差。接下來，它計算原始樣本與增強樣本之間的距離，分別表示為 *d*<sup>pos</sup> 和 *d*<sup>neg</sup>。目標是最小化 *d*<sup>pos</sup> 並最大化 *d*<sup>neg</sup>，使原始樣本與正樣本相似，而與負樣本不相似。在這個例子中，數據增強後，正樣本包含比原始樣本多得多的 *r*<sub>3</sub> 連結。這種不平衡導致 *d*<sup>pos</sup> 變得接近 *d*<sup>neg</sup>。在這種情況下，模塊需要調整關係嵌入以在 *d*<sup>pos</sup> 和 *d*<sup>neg</sup> 之間保持一個邊界，這可能會損害嵌入有效表示關係資訊的能力。

Table 4.1: An example of distance calculation.

表格 4.1: 距離計算範例。

[Image]

We choose cosine similarity to avoid the potential problems. Since cosine similarity measures the angle between vectors, it is affected by relation composition rather than the number of links of each relation. In the same example, replacing the distances *d*<sub>*i*</sub><sup>pos</sup> and *d*<sub>*i*</sub><sup>neg</sup>, we calculate the similarity between the original sample and the augmented samples, denoted as *sim*<sub>*i*</sub><sup>pos</sup> and *sim*<sub>*i*</sub><sup>neg</sup>. With this change, the goal becomes to maximize *sim*<sub>*i*</sub><sup>pos</sup> and minimize *sim*<sub>*i*</sub><sup>neg</sup>. As shown in Table 4.1, cosine similarity maintains a more obvious margin between *sim*<sub>*i*</sub><sup>pos</sup> and *sim*<sub>*i*</sub><sup>neg</sup>. Therefore, relation embeddings have more space to express their own semantic features.

我們選擇餘弦相似度以避免潛在問題。由於餘弦相似度測量的是向量之間的夾角，它受關係構成的影響，而非每個關係的連結數量。在同一個例子中，我們用 *d*<sub>*i*</sub><sup>pos</sup> 和 *d*<sub>*i*</sub><sup>neg</sup> 的距離替換，計算原始樣本和擴增樣本之間的相似度，表示為 *sim*<sub>*i*</sub><sup>pos</sup> 和 *sim*<sub>*i*</sub><sup>neg</sup>。經過這樣的改變，目標變成最大化 *sim*<sub>*i*</sub><sup>pos</sup> 並最小化 *sim*<sub>*i*</sub><sup>neg</sup>。如表 4.1 所示，餘弦相似度在 *sim*<sub>*i*</sub><sup>pos</sup> 和 *sim*<sub>*i*</sub><sup>neg</sup> 之間保持了更明顯的邊界。因此，關係嵌入有更多空間來表達其自身的語意特徵。

On the other hand, learning topological features is a fundamental task of GNN-based methods, since such information captures intrinsic structural associations between entities and enables the model to predict the existence of links. Different from semantic features, topological features express patterns in graph evolution, focusing on the formation rules of the graph's physical structure, providing the model with more intuitive information to evaluate triplets more precisely.

另一方面，學習拓撲特徵是基於 GNN 方法的一項基本任務，因為這些資訊捕捉了實體之間的內在結構關聯，並使模型能夠預測連結的存在。與語意特徵不同，拓撲特徵表達了圖演化中的模式，專注於圖物理結構的形成規則，為模型提供了更直觀的資訊，以更精確地評估三元組。

In this part, we follow the framework, GNN-based Enhanced Local Subgraph Modeling (GELS), proposed by GSELI [10]. It proposes a PageRank-based subgraph extraction to extract subgraphs with closer association, combined with the node representation initialization method and neighboring relational path representation of SNRI [12] to obtain better subgraphs. Then, it uses RGCN [3] as the graph convolutional network to utilize structural and relational information to learn entity topological embeddings and graph topological embeddings.

在這部分，我們遵循 GSELI [10] 提出的基於 GNN 的增強型局部子圖建模（GELS）框架。它提出了一種基於 PageRank 的子圖提取方法，以提取關聯更緊密的子圖，並結合 SNRI [12] 的節點表示初始化方法和相鄰關係路徑表示，以獲得更好的子圖。然後，它使用 RGCN [3] 作為圖卷積網絡，利用結構和關係資訊來學習實體拓撲嵌入和圖拓撲嵌入。

### 4.2.2 Temporal Feature Extraction and Influence Lifetime Estimation

In contrast to conventional knowledge graphs, temporal knowledge graphs contain more explicit and implicit temporal information that can help models to predict more accurately. However, it also increases the difficulty of learning, such as modeling timestamps associated with entities and triplets, as well as capturing the evolution of graph snapshots. To effectively exploit such temporal information to improve performance on temporal knowledge graphs, we design a temporal factor processor to extract temporal features and estimate the timestamp of influence lifetime of each link.

### 4.2.2 時間特徵提取與影響力生命週期估算

相較於傳統知識圖譜，時間知識圖譜包含更多明確和隱含的時間資訊，有助於模型更準確地預測。然而，這也增加了學習的難度，例如對與實體和三元組相關的時間戳進行建模，以及捕捉圖快照的演變。為了有效地利用這些時間資訊以提高在時間知識圖譜上的性能，我們設計了一個時間因子處理器，以提取時間特徵並估計每個連結的影響力生命週期的時間戳。

In addition, we use ToE (Timespans of Edge formation) [13] as one of the input information to enrich the temporal features. ToE, expressing how long it takes for a triplet to form from being able to form, is calculated as:
*t*<sub>ToE</sub> = *t*<sub>1</sub> − max(*t*<sub>*i*</sub>, *t*<sub>*j*</sub>), (4.3)
where *t*<sub>1</sub> is the timestamp that the triplet was founded, *t*<sub>*i*</sub> and *t*<sub>*j*</sub> are the timestamps that *i* and *j* appear in the graph. In this work, we use the timestamp of the earliest triplet containing *i* as *t*<sub>*i*</sub> like:
*t*<sub>*i*</sub> = {min(*t*<sub>*l*</sub>) | ∀*l*<sub>*k*</sub> ∈ *S* ∧ *i* ∈ *l*<sub>*k*</sub>}, (4.4)
and *t*<sub>*j*</sub> is defined in the same way. We also believe that learning relative temporal features is more helpful than learning absolute temporal features of entities and links. Therefore, except for *t*<sub>ToE</sub>, all input are calculated as the difference between temporal factors and the current time *t*<sub>curr</sub> by:
*t*<sub>*i*</sub>′ = |*t*<sub>*i*</sub> − *t*<sub>curr</sub>|. (4.5)

此外，我們使用 ToE (邊形成的時長) [13] 作為輸入資訊之一來豐富時間特徵。ToE 表示一個三元組從能夠形成到實際形成所需的時間，其計算方式如下：
*t*<sub>ToE</sub> = *t*<sub>1</sub> − max(*t*<sub>*i*</sub>, *t*<sub>*j*</sub>), (4.3)
其中 *t*<sub>1</sub> 是三元組建立時的時間戳，*t*<sub>*i*</sub> 和 *t*<sub>*j*</sub> 是 *i* 和 *j* 出現在圖中的時間戳。在本研究中，我們使用包含 *i* 的最早三元組的時間戳作為 *t*<sub>*i*</sub>，如下所示：
*t*<sub>*i*</sub> = {min(*t*<sub>*l*</sub>) | ∀*l*<sub>*k*</sub> ∈ *S* ∧ *i* ∈ *l*<sub>*k*</sub>}, (4.4)
而 *t*<sub>*j*</sub> 的定義方式相同。我們也認為，學習相對時間特徵比學習實體和連結的絕對時間特徵更有幫助。因此，除了 *t*<sub>ToE</sub> 之外，所有輸入都計算為時間因子與當前時間 *t*<sub>curr</sub> 之間的差值：
*t*<sub>*i*</sub>′ = |*t*<sub>*i*</sub> − *t*<sub>curr</sub>|. (4.5)

[Image]
Figure 4.5: Processing pipeline of Edge Temporal Feature Learning.

圖 4.5：邊時間特徵學習的處理流程。

For temporal feature learning, as shown in Figure 4.5, we encode each factor into embeddings for normalization, and then we adopt a linear layer as a simple encoder to encode edge temporal features *F*<sub>*l*</sub><sup>*e*</sup> of link *l* as:
*F*<sub>*l*</sub><sup>*e*</sup> = *w*<sub>1</sub>*z*<sub>*t*<sub>*i*</sub></sub> + *w*<sub>2</sub>*z*<sub>*t*<sub>*j*</sub></sub> + *w*<sub>3</sub>*z*<sub>*l*</sub> + *b*<sup>*e*</sup>, (4.6)
where *z*<sub>*t*<sub>*i*</sub></sub>, *z*<sub>*t*<sub>*j*</sub></sub>, and *z*<sub>*l*</sub> are encoded embeddings of *t*<sub>*i*</sub>′, *t*<sub>*j*</sub>′, and *t*<sub>*l*</sub> and *t*<sub>ToE</sub>, and *w*<sub>*x*</sub> are trainable weights.

對於時間特徵學習，如圖 4.5 所示，我們將每個因子編碼為嵌入以進行歸一化，然後採用一個線性層作為簡單的編碼器，將連結 *l* 的邊緣時間特徵 *F*<sub>*l*</sub><sup>*e*</sup> 編碼為：
*F*<sub>*l*</sub><sup>*e*</sup> = *w*<sub>1</sub>*z*<sub>*t*<sub>*i*</sub></sub> + *w*<sub>2</sub>*z*<sub>*t*<sub>*j*</sub></sub> + *w*<sub>3</sub>*z*<sub>*l*</sub> + *b*<sup>*e*</sup>, (4.6)
其中 *z*<sub>*t*<sub>*i*</sub></sub>, *z*<sub>*t*<sub>*j*</sub></sub>, 和 *z*<sub>*l*</sub> 是 *t*<sub>*i*</sub>′, *t*<sub>*j*</sub>′, *t*<sub>*l*</sub> 和 *t*<sub>ToE</sub> 的編碼嵌入，*w*<sub>*x*</sub> 是可訓練的權重。

As discussed earlier, the influence of a link depends not only on its start time but also on its termination time, which should be considered during model learning. However, most datasets do not provide such information. To address this limitation, we estimate the influence lifetime of each link via regression over its temporal information. This design not only enriches the information available to the model but also supplies missing temporal attributes, thereby providing a basis for updating the graph. Influence lifetime *t*<sup>end</sup> is estimated as follows:
*F*<sub>*l*</sub><sup>end</sup> = Regressor(*t*<sub>*i*</sub>, *t*<sub>*j*</sub>, *t*<sub>*l*</sub>, *t*<sub>ToE</sub>) = *w*<sub>*t*<sub>*i*</sub></sub>*t*<sub>*i*</sub> + *w*<sub>*t*<sub>*j*</sub></sub>*t*<sub>*j*</sub> + *w*<sub>*t*<sub>*l*</sub></sub>*t*<sub>*l*</sub> + *w*<sub>*troe*</sub>*t*<sub>ToE</sub> + *b*<sup>end</sup>, (4.7)
*t*<sup>end</sup> = *F*<sub>*l*</sub><sup>end</sup>*t*<sub>exp</sub>, (4.8)
where *w*<sub>*t*<sub>*i*</sub></sub>, *w*<sub>*t*<sub>*j*</sub></sub>, *w*<sub>*t*<sub>*l*</sub></sub>, and *w*<sub>*troe*</sub> are trainable weights, *F*<sub>*l*</sub><sup>end</sup> is the embedding of the influence lifetime of link *l*, *t*<sub>exp</sub> is a hyperparameter that decides the maximum of estimated influence lifetime.

如前所述，一個連結的影響不僅取決於其開始時間，也取決於其終止時間，這在模型學習過程中應予以考慮。然而，大多數資料集並未提供此類資訊。為了克服此限制，我們透過對其時間資訊進行迴歸來估計每個連結的影響力生命週期。此設計不僅豐富了模型可用的資訊，也補充了缺失的時間屬性，從而為更新圖提供了基礎。影響力生命週期 *t*<sup>end</sup> 的估算如下：
*F*<sub>*l*</sub><sup>end</sup> = Regressor(*t*<sub>*i*</sub>, *t*<sub>*j*</sub>, *t*<sub>*l*</sub>, *t*<sub>ToE</sub>) = *w*<sub>*t*<sub>*i*</sub></sub>*t*<sub>*i*</sub> + *w*<sub>*t*<sub>*j*</sub></sub>*t*<sub>*j*</sub> + *w*<sub>*t*<sub>*l*</sub></sub>*t*<sub>*l*</sub> + *w*<sub>*troe*</sub>*t*<sub>ToE</sub> + *b*<sup>end</sup>, (4.7)
*t*<sup>end</sup> = *F*<sub>*l*</sub><sup>end</sup>*t*<sub>exp</sub>, (4.8)
其中 *w*<sub>*t*<sub>*i*</sub></sub>、*w*<sub>*t*<sub>*j*</sub></sub>、*w*<sub>*t*<sub>*l*</sub></sub> 和 *w*<sub>*troe*</sub> 是可訓練的權重，*F*<sub>*l*</sub><sup>end</sup> 是連結 *l* 影響力生命週期的嵌入，*t*<sub>exp</sub> 是一個決定估計影響力生命週期最大值的超參數。

Finally, we use the sum of edge temporal features *F*<sub>*l*</sub><sup>*e*</sup> and influence lifetime embeddings *F*<sub>*l*</sub><sup>end</sup> as temporal features *F*<sub>*l*</sub><sup>temp</sup>, and fuse it into the hidden state as follows:
*F*<sub>*l*</sub><sup>temp</sup> = *F*<sub>*l*</sub><sup>*e*</sup> + *F*<sub>*l*</sub><sup>end</sup>, (4.9)
*h*<sub>*i*</sub><sup>*k*</sup> = Σ<sub>*r*<sub>*x*</sub>∈*R*</sub> Σ<sub>*j*∈*N*<sub>*r*<sub>*x*</sub></sub>(*i*)</sub> *α*<sub>*ir*<sub>*x*</sub>*j*</sub><sup>*k*</sup>*W*<sub>*r*<sub>*x*</sub></sub><sup>*k*</sup>*F*<sub>*l*</sub><sup>temp</sup>(*h*<sub>*j*</sub><sup>*k*−1</sup>, *z*<sub>*r*<sub>*x*</sub></sub><sup>*k*−1</sup>), (4.10)
*α*<sub>*ir*<sub>*x*</sub>*j*</sub><sup>*k*</sup> = *σ*<sub>*A*</sub>(*W*<sub>*A*</sub><sup>*k*</sup>*s*<sub>*ir*<sub>*x*</sub></sub> + *b*<sub>*A*</sub><sup>*k*</sup>), (4.11)
*s*<sub>*ir*<sub>*x*</sub>*j*</sub> = *σ*<sub>*B*</sub>(*W*<sub>*B*</sub><sup>*k*</sup>[*h*<sub>*i*</sub><sup>*k*−1</sup> ⊕ *h*<sub>*j*</sub><sup>*k*−1</sup> ⊕ *z*<sub>*r*<sub>*x*</sub></sub><sup>*k*−1</sup>] + *b*<sub>*B*</sub><sup>*k*</sup>), (4.12)
where *h*<sup>*k*</sup> are node hidden states of *k*-th layer, *z*<sub>*r*</sub><sup>*k*</sup> are relation embeddings of *k*-th layer, *σ*<sub>*A*</sub> and *σ*<sub>*B*</sub> are activation functions, *W*<sup>*k*</sup> is the transformation matrix, (*,*) is the fusion function. In this way, the model can extract and diffuse temporal features to find better entity and relation embeddings.

最後，我們將邊緣時間特徵 *F*<sub>*l*</sub><sup>*e*</sup> 和影響力生命週期嵌入 *F*<sub>*l*</sub><sup>end</sup> 的總和作為時間特徵 *F*<sub>*l*</sub><sup>temp</sup>，並將其融合到隱藏狀態中，如下所示：
*F*<sub>*l*</sub><sup>temp</sup> = *F*<sub>*l*</sub><sup>*e*</sup> + *F*<sub>*l*</sub><sup>end</sup>, (4.9)
*h*<sub>*i*</sub><sup>*k*</sup> = Σ<sub>*r*<sub>*x*</sub>∈*R*</sub> Σ<sub>*j*∈*N*<sub>*r*<sub>*x*</sub></sub>(*i*)</sub> *α*<sub>*ir*<sub>*x*</sub>*j*</sub><sup>*k*</sup>*W*<sub>*r*<sub>*x*</sub></sub><sup>*k*</sup>*F*<sub>*l*</sub><sup>temp</sup>(*h*<sub>*j*</sub><sup>*k*−1</sup>, *z*<sub>*r*<sub>*x*</sub></sub><sup>*k*−1</sup>), (4.10)
*α*<sub>*ir*<sub>*x*</sub>*j*</sub><sup>*k*</sup> = *σ*<sub>*A*</sub>(*W*<sub>*A*</sub><sup>*k*</sup>*s*<sub>*ir*<sub>*x*</sub></sub> + *b*<sub>*A*</sub><sup>*k*</sup>), (4.11)
*s*<sub>*ir*<sub>*x*</sub>*j*</sub> = *σ*<sub>*B*</sub>(*W*<sub>*B*</sub><sup>*k*</sup>[*h*<sub>*i*</sub><sup>*k*−1</sup> ⊕ *h*<sub>*j*</sub><sup>*k*−1</sup> ⊕ *z*<sub>*r*<sub>*x*</sub></sub><sup>*k*−1</sup>] + *b*<sub>*B*</sub><sup>*k*</sup>), (4.12)
其中 *h*<sup>*k*</sup> 是第 *k* 層的節點隱藏狀態，*z*<sub>*r*</sub><sup>*k*</sup> 是第 *k* 層的關係嵌入，*σ*<sub>*A*</sub> 和 *σ*<sub>*B*</sup> 是激活函數，*W*<sup>*k*</sup> 是轉換矩陣，(*,*) 是融合函數。透過這種方式，模型可以提取和傳播時間特徵，以找到更好的實體和關係嵌入。

## 4.3 Scoring Function, Threshold Tuning and Loss Function

Most link prediction models will calculate a score or probability for a triplet to determine whether it exists according to various features such as node features, subgraph features, relation features, and other information. In this work, we calculate scores based on semantic features and structural features, respectively, and add them as final scores.

## 4.3 評分函數、閾值調整與損失函數

大多數連結預測模型會根據節點特徵、子圖特徵、關係特徵和其他資訊，為一個三元組計算一個分數或機率，以判斷其是否存在。在本研究中，我們分別根據語意特徵和結構特徵計算分數，並將它們相加作為最終分數。

For each triplet *l* = (*u*, *r*, *v*), semantic score *ϕ*<sup>sem</sup> is calculated by DistMult [16], denoted as:
*ϕ*<sup>sem</sup>(*l*) = ⟨*e*<sub>*u*</sub>, *z*<sub>*r*</sub>, *e*<sub>*v*</sub>⟩, (4.13)
where *e*<sub>*u*</sub> and *e*<sub>*v*</sub> are semantic embeddings of entity *u* and *v*, *z*<sub>*r*</sub> is the embedding of relation *r*, and ⟨,⟩ denotes the element-wise product.

對於每個三元組 *l* = (*u*, *r*, *v*)，語意分數 *ϕ*<sup>sem</sup> 由 DistMult [16] 計算，表示為：
*ϕ*<sup>sem</sup>(*l*) = ⟨*e*<sub>*u*</sub>, *z*<sub>*r*</sub>, *e*<sub>*v*</sub>⟩, (4.13)
其中 *e*<sub>*u*</sub> 和 *e*<sub>*v*</sub> 是實體 *u* 和 *v* 的語意嵌入，*z*<sub>*r*</sub> 是關係 *r* 的嵌入，⟨,⟩ 表示逐元素乘積。

On the other hand, structural score *ϕ*<sup>str</sup> is calculated as:
*ϕ*<sup>str</sup>(*l*) = *W*[*h*<sub>*u*</sub><sup>*K*</sup> ⊕ *h*<sub>*v*</sub><sup>*K*</sup> ⊕ *z*<sub>*r*</sub><sup>*K*</sup> ⊕ *Z*<sub>*G*(*u*,*r*,*v*)</sub>], (4.14)
where *h*<sub>*u*</sub><sup>*K*</sup> and *h*<sub>*v*</sub><sup>*K*</sup> are structural embeddings of entity *u* and *v* learned by GNN, *Z*<sub>*G*(*u*,*r*,*v*)</sub> is the representation of subgraph *G*(*u*,*r*,*v*), *W* is the transformation matrix.

另一方面，結構分數 *ϕ*<sup>str</sup> 的計算方式如下：
*ϕ*<sup>str</sup>(*l*) = *W*[*h*<sub>*u*</sub><sup>*K*</sup> ⊕ *h*<sub>*v*</sub><sup>*K*</sup> ⊕ *z*<sub>*r*</sub><sup>*K*</sup> ⊕ *Z*<sub>*G*(*u*,*r*,*v*)</sub>], (4.14)
其中 *h*<sub>*u*</sub><sup>*K*</sup> 和 *h*<sub>*v*</sub><sup>*K*</sup> 是由 GNN 學習到的實體 *u* 和 *v* 的結構嵌入，*Z*<sub>*G*(*u*,*r*,*v*)</sub> 是子圖 *G*(*u*,*r*,*v*) 的表示，*W* 是轉換矩陣。

The final score *ϕ*(*l*) is denoted as:
*ϕ*(*l*) = Linear(*ϕ*<sup>sem</sup>(*l*), *ϕ*<sup>str</sup>(*l*)). (4.15)

最終分數 *ϕ*(*l*) 表示為：
*ϕ*(*l*) = Linear(*ϕ*<sup>sem</sup>(*l*), *ϕ*<sup>str</sup>(*l*)). (4.15)

The existence of a link *l* is determined by comparing its final score *ϕ*(*l*) against a threshold *θ*. To more accurately predict the presence of links, the threshold is adaptively determined during training according to the scores of positive and negative samples, as formulated below:
*θ* = (*θ*<sub>old</sub> + (*P*<sub>pos</sub><sup>*a*</sup> + *P*<sub>neg</sub><sup>*b*</sup>)/2)/2, (4.16)
where *θ*<sub>old</sub> is the previous threshold, *P*<sub>pos</sub><sup>*a*</sup> and *P*<sub>neg</sub><sup>*b*</sup> denote the a-th and b-th percentile scores of the positive and negative samples, respectively. We expect the model to find a threshold that best fits the overall score distribution through statistics, and adjust the strictness of the threshold according to *a* and *b*. Therefore, the model calculates the boundary that distinguishes positive and negative samples in each iteration and averages it with the past threshold, thus retaining the past statistical values.

連結 *l* 是否存在是透過將其最終分數 *ϕ*(*l*) 與閾值 *θ* 進行比較來確定的。為了更準確地預測連結的存在，閾值在訓練過程中會根據正樣本和負樣本的分數進行自適應性調整，其公式如下：
*θ* = (*θ*<sub>old</sub> + (*P*<sub>pos</sub><sup>*a*</sup> + *P*<sub>neg</sub><sup>*b*</sup>)/2)/2, (4.16)
其中 *θ*<sub>old</sub> 是先前的閾值，*P*<sub>pos</sub><sup>*a*</sup> 和 *P*<sub>neg</sub><sup>*b*</sup> 分別表示正樣本和負樣本的第 a 個和第 b 個百分位數分數。我們期望模型透過統計找到最適合整體分數分佈的閾值，並根據 *a* 和 *b* 調整閾值的嚴格程度。因此，模型在每次迭代中計算區分正負樣本的邊界，並將其與過去的閾值取平均，從而保留過去的統計值。

As for the loss function, in order to make the model consider both contrastive learning and structure learning, following DEKG-ILP [9] and GSELI [10], we first calculate the loss of the two separately and then add them as the total loss *L*.

至於損失函數，為了讓模型同時考慮對比學習和結構學習，我們遵循 DEKG-ILP [9] 和 GSELI [10] 的做法，首先分別計算兩者的損失，然後將它們相加作為總損失 *L*。

Contrastive learning loss *L*<sup>con</sup> is defined as:
*L*<sup>con</sup> = Σ<sub>*l*∈*S*</sub> Σ<sub>*i*∈*l*</sub> max(0, dist(*e*<sub>*i*</sub><sup>pos</sup>, *e*<sub>*i*</sub>) – dist(*e*<sub>*i*</sub><sup>neg</sup>, *e*<sub>*i*</sub>) + *γ*), (4.17)
where *S* is the sample set, dist(,) is the distance measured by the similarity function that is cosine similarity in this work, and *γ* is a hyperparameter that decides the margin.

對比學習損失 *L*<sup>con</sup> 定義為：
*L*<sup>con</sup> = Σ<sub>*l*∈*S*</sub> Σ<sub>*i*∈*l*</sub> max(0, dist(*e*<sub>*i*</sub><sup>pos</sup>, *e*<sub>*i*</sub>) – dist(*e*<sub>*i*</sub><sup>neg</sup>, *e*<sub>*i*</sub>) + *γ*), (4.17)
其中 *S* 是樣本集，dist(,) 是由本研究中使用的餘弦相似度函數測量的距離，而 *γ* 是決定邊界的超參數。

Structure learning loss *L*<sup>str</sup> is calculated as:
*L*<sup>str</sup> = Σ<sub>*l*<sub>*p*</sub>∈*S*<sup>+</sup>,*l*<sub>*n*</sub>∈*S*<sup>−</sup></sub> max(0, *ϕ*(*l*<sub>*n*</sub>) – *ϕ*(*l*<sub>*p*</sub>) + *γ*), (4.18)
where *S*<sup>+</sup> and *S*<sup>−</sup> are the positive and negative sample sets.

結構學習損失 *L*<sup>str</sup> 的計算方式如下：
*L*<sup>str</sup> = Σ<sub>*l*<sub>*p*</sub>∈*S*<sup>+</sup>,*l*<sub>*n*</sub>∈*S*<sup>−</sup></sub> max(0, *ϕ*(*l*<sub>*n*</sub>) – *ϕ*(*l*<sub>*p*</sub>) + *γ*), (4.18)
其中 *S*<sup>+</sup> 和 *S*<sup>−</sup> 分別為正樣本集和負樣本集。

Finally, the total loss *L* is denoted as:
*L* = *L*<sup>str</sup> + *β*L*<sup>con</sup>, (4.19)
where *β* is a hyperparameter controlling the proportion of *L*<sup>con</sup>.

最後，總損失 *L* 表示為：
*L* = *L*<sup>str</sup> + *β*L*<sup>con</sup>, (4.19)
其中 *β* 是控制 *L*<sup>con</sup> 比例的超參數。

[Image]

# Chapter 5 Experiments

In this section, we conduct several experiments to validate the ability of the proposed EvoGUT. The primary objective is to evaluate whether the model can effectively learn to update the graph automatically and maintain performance in multi-step scenarios.

# 第五章 實驗

在本節中，我們進行了多項實驗，以驗證所提出的 EvoGUT 的能力。主要目標是評估模型是否能有效地學會自動更新圖，並在多步驟情境下保持性能。

## 5.1 Experiment Settings

We conduct experiments under two settings. In the transductive setting, the model's performance is evaluated in the traditional static scenario. In the multi-step setting, we examine the model's ability to learn rules of graph evolution. Furthermore, we treat the link prediction problem as a binary classification and analyze the potential of the model as a graph completion tool.

## 5.1 實驗設定

我們在兩種設定下進行實驗。在直推式設定中，模型性能在傳統的靜態場景中進行評估。在多步設定中，我們檢驗模型學習圖演化規則的能力。此外，我們將連結預測問題視為二元分類，並分析模型作為圖補全工具的潛力。

In the multi-step setting, different from the typical setting, we use an independent graph as the base graph in the testing phase. This design ensures that the model's performance stems from learning evolutionary patterns sufficiently rather than relying on historical data. At the same time, the model continuously predicts triplets for multiple timestamps to examine its robustness and precision. If the model cannot accurately distinguish between positive and negative samples, the resulting errors will accumulate and affect subsequent predictions. Therefore, a model must be able to accurately predict and self-correct to maintain excellent performance in this setting.

在多步設定中，與典型設定不同，我們在測試階段使用一個獨立的圖作為基礎圖。這種設計確保了模型的性能源於充分學習演化模式，而非依賴歷史數據。同時，模型連續預測多個時間戳的三元組，以檢驗其穩健性和精確度。如果模型無法準確區分正負樣本，所產生的錯誤將會累積並影響後續預測。因此，在這種設定下，模型必須能夠準確預測並自我校正，以保持優異的性能。

In the transductive setting, following the traditional setting and previous works, the base graph contains all triplets in the training and validation sets to ensure all entities and relations are known in the testing phase. But the model needs to predict triplets of several timestamps at once to prove that it has fully learned the rules of graph evolution.

在直推式設定中，遵循傳統設定和先前研究，基礎圖包含訓練集和驗證集中的所有三元組，以確保在測試階段所有實體和關係都是已知的。但模型需要一次性預測數個時間戳的三元組，以證明它已完全學會了圖演化的規則。

For our method and GSELI [10], we set all dimensions of embeddings to 100, 64 for DEKG-ILP [9] due to the memory issue. For the hyperparameters used to tune the threshold, we set the hyperparameters a and b used to tune the threshold to 25 and 75. For GSELI [10] and DEKG-ILP [9], since margin-based ranking loss and DistMult [16] are used, the scores of positive samples usually tend to be positive, while the scores of negative samples tend to be negative or the minimum value, and there is no limit to the range of scores. Therefore, 0 is set as the threshold for both. As for the updating strategy, we choose pruning to remove the expired information from graphs. However, in the classification evaluation, link existence is determined solely by the model's final scoring function, independent of whether the corresponding triplets are retained in the updated graph afterward. For RE-GCN [14] and CorDGT [15], we follow the configuration provided by the original authors. Since CorDGT [15] does not contain relation information, we only replace the head or tail entity when negative sampling under the transductive setting.

對於我們的方法和 GSELI [10]，我們將所有嵌入的維度設為 100，對於 DEKG-ILP [9] 則因記憶體問題設為 64。用於調整閾值的超參數 a 和 b 設為 25 和 75。對於 GSELI [10] 和 DEKG-ILP [9]，由於使用了基於邊界的排序損失和 DistMult [16]，正樣本的分數通常趨向於正值，而負樣本的分數趨向於負值或最小值，且分數範圍沒有限制。因此，兩者的閾值都設為 0。至於更新策略，我們選擇修剪來移除圖中過期的資訊。然而，在分類評估中，連結是否存在僅由模型的最終評分函數決定，而與相應的三元組是否在更新後的圖中被保留無關。對於 RE-GCN [14] 和 CorDGT [15]，我們遵循原始作者提供的配置。由於 CorDGT [15] 不包含關係資訊，我們在直推式設定下的負採樣中僅替換頭實體或尾實體。

## 5.2 Datasets

We test our model and baselines on two datasets, Wikidata [17,18] and ICEWS14 [19]. Wikidata [17] is created by [18] from Wikidata, and contains the expiration timestamp of each triplet. Since the data is concentrated in certain years, we only used data from 2001 to 2020. ICEWS14 [19] consists of political events that happened in 2014, recorded in Integrated Crisis Early Warning System. For both datasets, we divided them into training, validation, and testing sets according to chronological order, but with different partitions for the multi-step setting and the transductive setting. The detailed statistics are shown in Table 5.1, 5.2, 5.3 and 5.4. In the multi-step setting, to test the model's ability to learn evolutionary patterns, it uses unseen data during training as the testing base graph. As illustrated in the base graph in Table 5.1 and 5.3, its time range T is non-overlapping with the training and validation sets. In the transductive setting, referencing RE-GCN [14] and CorDGT [15], the testing base graph uses all data used during training. As demonstated in Table 5.2 and 5.4, the time range T of the base graph is the summation of the training and validation sets.

## 5.2 資料集

我們在兩個資料集上測試我們的模型和基線：Wikidata [17,18] 和 ICEWS14 [19]。Wikidata [17] 由 [18] 從 Wikidata 創建，並包含每個三元組的到期時間戳。由於資料集中在特定年份，我們僅使用 2001 年至 2020 年的資料。ICEWS14 [19] 包含 2014 年發生的政治事件，記錄於「整合危機早期預警系統」中。對於這兩個資料集，我們都按照時間順序將其劃分為訓練集、驗證集和測試集，但多步設定和傳導式設定的分區方式不同。詳細統計數據顯示在表 5.1、5.2、5.3 和 5.4 中。在多步設定中，為了測試模型學習演化模式的能力，它使用訓練期間未見的資料作為測試基礎圖。如表 5.1 和 5.3 中的基礎圖所示，其時間範圍 T 與訓練集和驗證集不重疊。在傳導式設定中，參考 RE-GCN [14] 和 CorDGT [15]，測試基礎圖使用訓練期間使用的所有資料。如表 5.2 和 5.4 所示，基礎圖的時間範圍 T 是訓練集和驗證集的總和。

## 5.3 Evaluation

To evaluate the model's performance in head entity prediction, tail entity prediction, and relation prediction tasks, we follow GSELI [10] and randomly sample 50 negative samples to be ranked together with the corresponding positive sample. The model is then evaluated from two perspectives: ranking and binary classification. The ranking order of sample tests whether the model can assign higher scores to positive samples; the binary classification problem tests whether the model can correctly distinguish between positive and negative samples. The metrics used in these two aspects are as follows:

## 5.3 評估

為了評估模型在頭實體預測、尾實體預測和關係預測任務中的性能，我們遵循 GSELI [10] 的做法，隨機抽樣 50 個負樣本與對應的正樣本一起進行排序。然後從兩個角度評估模型：排序和二元分類。樣本的排序順序測試模型是否能為正樣本分配更高的分數；二元分類問題測試模型是否能正確區分正負樣本。這兩個方面使用的指標如下：

Table 5.1: Statistics of Wikidata in the multi-step setting. |T| is the number of timestamps, |D| implies the number of links, |ε| means the number of entities, |R| suggests the number of relations, and avg. means the average number in each timestamp.

表格 5.1：Wikidata 在多步設定下的統計資料。|T| 是時間戳的數量，|D| 代表連結的數量，|ε| 表示實體的數量，|R| 暗示關係的數量，而 avg. 則表示每個時間戳中的平均數量。

[Image]

Table 5.2: Statistics of Wikidata in the transductive setting.

表格 5.2：Wikidata 在直推設定中的統計資料。

[Image]

Table 5.3: Statistics of ICEWS14 in the multi-step setting.

表格 5.3：ICEWS14 在多步設定中的統計資料。

[Image]

Table 5.4: Statistics of ICEWS14 in the transductive setting.

表格 5.4：ICEWS14 在直推設定中的統計數據。

[Image]

* **Mean Reciprocal Rank (MRR)**
MRR is a measure of ranking accuracy that uses reciprocal rank to give lower-ranked samples lower scores; as a result, the positive samples are ranked higher, and the score is higher. It is denoted as:
MRR = 1/|D| Σ<sub>l∈D</sub> 1/rank(l) (5.1)
where D is the entire sample set.

* **平均倒數排名 (MRR)**
MRR 是一種排名準確度的衡量標準，它使用倒數排名來給予排名較低的樣本較低的分數；結果，正樣本的排名較高，分數也較高。其表示為：
MRR = 1/|D| Σ<sub>l∈D</sub> 1/rank(l) (5.1)
其中 D 是整個樣本集。

* **Hits ratio (Hits@k)**
Hits ratio calculates the proportion of samples whose ranking is less than or equal to k. In this study, we set k from 1, 5, 10 and calculate as:
Hits@k = Σ<sub>l∈D</sub>|rank(l) ≤ k| / |D| (5.2)

* **命中率 (Hits@k)**
命中率計算排名小於或等於 k 的樣本比例。在本研究中，我們設定 k 為 1、5、10 並計算如下：
Hits@k = Σ<sub>l∈D</sub>|rank(l) ≤ k| / |D| (5.2)

* **Normalized Discounted Cumulative Gain (NDCG)**
DCG considers items' relevance and rankings to make higher-relevance items have a higher score when their rankings are higher, and it compares with the DCG of ideal ranking order (IDCG) to calculate NDCG to assess the current ranking, defined as:
nDCG = DCG/IDCG = Σ<sub>i=1</sub><sup>|RELp|</sup> (2<sup>rel<sub>i</sub></sup>-1)/log<sub>2</sub>(i+1) / Σ<sub>i=1</sub><sup>|RELp|</sup> (2<sup>rel<sub>i</sub></sup>-1)/log<sub>2</sub>(i+1) (5.3)
Since in our setting, only the positive sample is the highly relevant sample, we implement NDCG by the following equation:
nDCG = DCG/IDCG = (1/log<sub>2</sub>(rank(l)+1)) / (1/log<sub>2</sub>(2)) (5.4)

* **歸一化折損累積增益 (NDCG)**
DCG 考慮項目的相關性和排名，使排名較高的較相關項目獲得較高的分數，並將其與理想排名順序的 DCG (IDCG) 進行比較，以計算 NDCG 來評估當前排名，定義如下：
nDCG = DCG/IDCG = Σ<sub>i=1</sub><sup>|RELp|</sup> (2<sup>rel<sub>i</sub></sup>-1)/log<sub>2</sub>(i+1) / Σ<sub>i=1</sub><sup>|RELp|</sup> (2<sup>rel<sub>i</sub></sup>-1)/log<sub>2</sub>(i+1) (5.3)
由於在我們的設定中，只有正樣本是高度相關的樣本，我們透過以下方程式實現 NDCG：
nDCG = DCG/IDCG = (1/log<sub>2</sub>(rank(l)+1)) / (1/log<sub>2</sub>(2)) (5.4)

* **Accuracy (ACC)**
Accuracy is the simplest metric to evaluate a classification task. It focuses on the proportion of correct prediction across all samples, denoted as:
Accuracy = (TP+TN)/|D| , (5.5)
where TP is the number of positive samples predicted correctly, and TN is the number of negative samples being predicted correctly.

* **準確率 (ACC)**
準確率是評估分類任務最簡單的指標。它關注所有樣本中正確預測的比例，表示為：
準確率 = (TP+TN)/|D| , (5.5)
其中 TP 是正確預測的正樣本數，TN 是正確預測的負樣本數。

* **Area Under Receiver Operating characteristic Curve (AUROC)**
AUROC is the area under the curve formed by FPR (False Positive Rate) and TPR (True Positive Rate). FPR represents the proportion of samples that are actually negative and are judged as positive, and TPR represents the proportion of samples that are actually positive and are judged as positive. It provides an intuitive metric for measuring a model’s ability to classify positive and negative samples.

* **接收者操作特徵曲線下面積 (AUROC)**
AUROC 是由 FPR（偽陽性率）和 TPR（真陽性率）形成的曲線下面積。FPR 代表實際為陰性但被判斷為陽性的樣本比例，而 TPR 代表實際為陽性且被判斷為陽性的樣本比例。它提供了一個直觀的指標，用於衡量模型分類陽性和陰性樣本的能力。

* **Area Under Precision-Recall Curve (AUPRC)**
AUPRC is the area under the curve formed by Precision and Recall. Precision emphasizes the proportion of samples that the model classifies as positive and are actually positive, while Recall emphasizes the proportion of samples that are actually positive and were classified as positive. It considers the impact of false positives on negative samples and evaluates the model's ability to correctly classify samples with a large number of negative samples.

* **精準率-召回率曲線下面積 (AUPRC)**
AUPRC 是由精準率和召回率所形成曲線下的面積。精準率強調模型分類為陽性且實際上為陽性的樣本比例，而召回率則強調實際上為陽性且被分類為陽性的樣本比例。它考慮了偽陽性對陰性樣本的影響，並評估模型在大量陰性樣本中正確分類樣本的能力。

* **F1-score**
F1-score is the harmonic mean of Precision and Recall, representing the consideration of both metrics. In this work, we calculate it as:
F1 = 2 × (Precision × Recall)/(Precision + Recall). (5.6)

* **F1 分數**
F1 分數是精準率和召回率的調和平均數，代表對這兩個指標的綜合考量。在本研究中，我們計算如下：
F1 = 2 × (精準率 × 召回率) / (精準率 + 召回率)。(5.6)

* **Balanced accuracy (Balanced ACC)**
Balanced accuracy considers both the correct identification of positive and negative samples, requiring the model to be as accurate as possible in its judgments of both. Sacrificing accuracy for one side at the expense of the other will decrease balanced accuracy. The equation is as follows:
Balanced accuracy = (sensitivity + specificity)/2 (5.7)
where sensitivity represents the proportion of samples that are actually positive and are judged as positive, same as Recall. Specificity represents the proportion of samples that are actually negative and are judged as negative.

* **平衡準確率 (Balanced ACC)**
平衡準確率同時考慮了對正樣本和負樣本的正確識別，要求模型在對兩者的判斷中盡可能準確。犧牲一方的準確性以換取另一方的準確性會降低平衡準確率。方程式如下：
平衡準確率 = (敏感度 + 特異度) / 2 (5.7)
其中敏感度代表實際為正且被判斷為正的樣本比例，與召回率相同。特異度代表實際為負且被判斷為負的樣本比例。

## 5.4 Baselines

We compare MODEL with the following methods, including semi-inductive link prediction and temporal knowledge graph reasoning methods.

## 5.4 基線模型

我們將 MODEL 與以下方法進行比較，包括半歸納式連結預測和時間知識圖譜推理方法。

* **DEKG-ILP** [9] extends inductive link prediction to contain bridging links to apply to more situations. It uses semantic information learning by contrastive learning and topological information to predict both enclosing and bridging links.

* **DEKG-ILP** [9] 將歸納式連結預測擴展為包含橋接連結，以適用於更多情況。它利用對比學習所學到的語意資訊和拓撲資訊來預測封閉連結和橋接連結。

* **GSELI** [10] is an improved model based on DEKG-ILP [9]. It adds a more efficient subgraph extraction module and neighboring relational paths modeling from SNRI [12], increasing prediction accuracy.

* **GSELI** [10] 是基於 DEKG-ILP [9] 的改良模型。它增加了一個更有效率的子圖提取模組，以及來自 SNRI [12] 的相鄰關係路徑模型，從而提高了預測準確度。

* **RE-GCN** [14] tries to integrate static and dynamic features into representations for temporal graph reasoning. It learns entity and relation representations by capturing structural dependencies within a single timestamp and sequential patterns across timestamps, and incorporates the static properties of the graph to contain stable features. In this way, it achieves the best performance on several datasets.

* **RE-GCN** [14] 嘗試將靜態與動態特徵整合至時間圖譜推理的表徵中。它透過捕捉單一時間戳內的結構相依性及跨時間戳的序列模式來學習實體與關係的表徵，並結合圖譜的靜態屬性以包含穩定的特徵。透過此方式，它在數個資料集上達到了最佳效能。

* **CorDGT** [15] proposes a Transformer-based model with a novel method to extract proximity more efficiently to capture comprehensive features of graphs. It employs the Poisson point process assumption to estimate temporal features and encode them with spatial features to obtain high-order proximity, and uses the multi-head self-attention mechanism to enhance expressive power.

* **CorDGT** [15] 提出了一種基於 Transformer 的模型，採用一種新穎的方法來更有效地提取鄰近性，以捕捉圖的綜合特徵。它採用泊松點過程假設來估計時間特徵，並將其與空間特徵編碼以獲得高階鄰近性，並使用多頭自註意力機制來增強表達能力。

## 5.5 Results

This section presents the experimental results across three aspects to evaluate the performance of the proposed model, EvoGUT. First, we analyze ranking results under both multi-step and transductive settings to assess prediction accuracy. Second, we examine the classification performance to verify the model's ability to predict link existence. Finally, we conduct ablation studies to quantify the contributions of each component.

## 5.5 結果

本節從三個方面介紹了實驗結果，以評估所提模型 EvoGUT 的性能。首先，我們分析了多步和直推式設定下的排名結果，以評估預測準確性。其次，我們檢驗了分類性能，以驗證模型預測連結存在的能力。最後，我們進行了消融研究，以量化每個組件的貢獻。

### 5.5.1 Ranking

We evaluate our model and baselines under multi-step and transductive settings by ranking order metrics.

### 5.5.1 排名

我們透過排名順序指標，在多步和直推式設定下評估我們的模型和基線。

#### Multi-Step Temporal Link Prediction

Tables 5.5, 5.6, and 5.7 report the experimental results of our method and baseline models under the multi-step setting. On Wikidata, although DEKG-ILP [9] and GSELI [10] are designed to handle scenarios with emerging entities, their reliance on structural features limits their performance on the sparser dataset. By enhancing GSELI [10] with temporal feature learning and a new training framework, our model achieves excellent performance. On ICEWS14, DEKG-ILP [9] maintains a similar performance on average, while GSELI [10] exhibits a noticeable increase. While our model outperforms the baselines on average, its advantage is less evident compared with its performance on Wikidata, where a clearer performance margin is observed.

#### 多步時間連結預測

表 5.5、5.6 和 5.7 報告了我們的方法和基線模型在多步設定下的實驗結果。在 Wikidata 上，儘管 DEKG-ILP [9] 和 GSELI [10] 旨在處理具有新興實體的場景，但它們對結構特徵的依賴限制了它們在稀疏數據集上的性能。通過使用時間特徵學習和新的訓練框架增強 GSELI [10]，我們的模型取得了優異的性能。在 ICEWS14 上，DEKG-ILP [9] 的平均性能保持相似，而 GSELI [10] 則表現出顯著的提升。雖然我們的模型在平均水平上優於基線，但與在 Wikidata 上的性能相比，其優勢不太明顯，在 Wikidata 上觀察到更清晰的性能差距。

Table 5.5: The average results of multi-step temporal link prediction.

表格 5.5：多步時間連結預測的平均結果。

[Image]

Table 5.6: The results of multi-step temporal link prediction on Wikidata.

表格 5.6：在 Wikidata 上進行多步時間連結預測的結果。

[Image]

Table 5.7: The results of multi-step temporal link prediction on ICEWS14.

表格 5.7：ICEWS14 上多步時間連結預測的結果。

[Image]

We speculate that this behavior arises from the intrinsic characteristics of the ICEWS14 dataset. ICEWS14 spans a relatively shorter overall time period but contains a large number of timestamps, reflecting frequent and fine-grained temporal changes. Furthermore, its content is primarily composed of news events, whose influence often decays rapidly over time. Therefore, temporal information and the duration of influence play a less critical role in this setting. Additionally, the relatively dense graph structure encourages the model to rely more heavily on structural and semantic features for prediction, rather than temporal features.

我們推測這種行為源於 ICEWS14 資料集的內在特性。ICEWS14 涵蓋的總體時間段相對較短，但包含大量的時間戳，反映了頻繁且細粒度的時間變化。此外，其內容主要由新聞事件組成，其影響力通常會隨時間迅速衰減。因此，時間資訊和影響持續時間在此設定中扮演的角色較不關鍵。此外，相對密集的圖結構促使模型更依賴結構和語意特徵進行預測，而非時間特徵。

In contrast, Wikidata exhibits coarser timestamp granularity and a sparser graph structure, resulting in suboptimal structural representations. In this case, temporal features can effectively compensate for the limitations of structural information and thereby improve overall performance. However, because our model adopts a pruning-based updating strategy, less influential triplets are discarded during graph updates. On ICEWS14, where structural information is particularly important, such pruning probably leads to structural information loss, resulting in inferior performance compared to Wikidata.

相較之下，Wikidata 表現出較粗的時間戳粒度和較稀疏的圖結構，導致次優的結構表示。在這種情況下，時間特徵可以有效地彌補結構資訊的限制，從而提高整體性能。然而，由於我們的模型採用基於修剪的更新策略，在圖更新過程中會丟棄影響力較小的三元組。在 ICEWS14 中，結構資訊尤為重要，這種修剪可能導致結構資訊的損失，從而導致與 Wikidata 相比性能較差。

For relation prediction shown in Tables 5.6 and 5.7, our model achieves suboptimal performance on Wikidata, but outperforms other methods on ICEWS14. One possible explanation is that relations are generally less sensitive to temporal information than entities. Even when the influence of certain information diminishes over time, it can still provide valuable structural cues for relation prediction. By removing such information, our model slightly decreases performance on Wikidata. In contrast, ICEWS14 is characterized by highly transient events where historical data can quickly become obsolete and accumulate as noise. In this scenario, appropriate information pruning helps reduce noise and leads to improved relation prediction performance.

對於表 5.6 和 5.7 中所示的關係預測，我們的模型在 Wikidata 上取得了次優的性能，但在 ICEWS14 上優於其他方法。一個可能的解釋是，關係通常對時間資訊的敏感度低於實體。即使某些資訊的影響力隨著時間的推移而減弱，它仍然可以為關係預測提供有價值的結構性線索。通過移除這些資訊，我們的模型在 Wikidata 上的性能略有下降。相比之下，ICEWS14 的特點是高度短暫的事件，歷史數據會迅速過時並積累為噪聲。在這種情況下，適當的資訊修剪有助於減少噪聲並提高關係預測性能。

#### Transductive Link Prediction

Tables 5.8 and 5.9 report the performance of our model and baselines on the two datasets. In the transductive setting, all entities and relations are known, which corresponds to a scenario favored by most temporal graph reasoning methods. Due to the reduced uncertainty, models typically achieve better performance in this setting than in the multi-step setting.

#### 直推式連結預測

表 5.8 和 5.9 報告了我們模型和基線在兩個數據集上的性能。在直推式設定中，所有實體和關係都是已知的，這對應於大多數時間圖推理方法所偏好的場景。由於不確定性降低，模型在此設定中通常比在多步設定中取得更好的性能。

Table 5.8: The results of transductive temporal link prediction on Wikidata.

表格 5.8：Wikidata 上直推式時間連結預測的結果。

[Image]

Table 5.9: The results of transductive temporal link prediction on ICEWS14.

表格 5.9：在 ICEWS14 上進行直推式時間連結預測的結果。

[Image]

On Wikidata, consistent with the observations in the multi-step setting, our model achieves excellent performance in entity prediction but performs relatively poorly in relation prediction. On ICEWS14, it lags behind other baselines. This is because our model focuses on the capability of handling dynamic, unknown structures and continuous prediction, strengthening its performance in the multi-step setting, thus sacrificing its ability in static graphs, which emphasize structural features. However, on Wikidata, it still maintains better performance in entity prediction by utilizing temporal features to compensate for the lack of structural features.

在 Wikidata 上，與多步設定中的觀察結果一致，我們的模型在實體預測方面取得了優異的性能，但在關係預測方面表現相對較差。在 ICEWS14 上，它落後於其他基線。這是因為我們的模型專注於處理動態、未知結構和連續預測的能力，從而增強了其在多步設定中的性能，但犧牲了其在強調結構特徵的靜態圖中的能力。然而，在 Wikidata 上，它仍然通過利用時間特徵來彌補結構特徵的不足，從而在實體預測中保持了較好的性能。

Both RE-GCN [14] and CorDGT [15], originally designed for temporal graphs, perform relatively well. However, RE-GCN [14] perform poorly in head entity prediction on Wikidata as reported in Table 5.8. This is presumably due to the inverse relations used during learning are not derived from real datasets. Since Wikidata is sparse and RE-GCN [14] does not leverage a global static graph on this dataset, it lacks aggregateble information, thus offering little help for head entity prediction. In contrast, on ICEWS14, by including a global static graph as a reference, even if the inverse relations do not actually exist, there is sufficient information to build a good representation of them. Similarly, CorDGT [15] also relies on a global static graph, thus performing slightly worse on Wikidata, the sparser dataset.

RE-GCN [14] 和 CorDGT [15] 最初都是為時間圖設計的，表現相對較好。然而，如表 5.8 所示，RE-GCN [14] 在 Wikidata 的頭實體預測上表現不佳。這大概是因為學習期間使用的逆關係並非源自真實數據集。由於 Wikidata 是稀疏的，且 RE-GCN [14] 未利用此數據集上的全域靜態圖，因此缺乏可聚合的資訊，對頭實體預測的幫助甚微。相反，在 ICEWS14 上，通過將全域靜態圖作為參考，即使逆關係實際上並不存在，也有足夠的資訊來建立它們的良好表示。同樣地，CorDGT [15] 也依賴於全域靜態圖，因此在較稀疏的數據集 Wikidata 上表現稍差。

### 5.5.2 Binary Classification

Even though the model performs well on ranking metrics, demonstrating its ability to distinguish between positive and negative samples in a relative sense, this does not necessarily demonstrate strong performance in absolute, instance-level discrimination. Therefore, we further verify the performance of our model and baselines on the link classification task under the multi-step setting, as shown in Tables 5.10 and 5.11. The most commonly used classification metrics include accuracy, AUROC, and F1-score. However, under extreme class imbalance between positive and negative samples, accuracy and AUROC can be misleading due to the dominance of negative samples, while F1-score does not consider the accuracy of negative samples. Therefore, we additionally report AUPRC and balanced accuracy, which are more suitable for evaluating performance in highly imbalanced scenarios. AUPRC focuses on the ability to avoid misclassifying positive samples among a large number of negative samples, while balanced accuracy reflects the average ability to distinguish between positive and negative samples. These two metrics are therefore particularly suitable for evaluating graph completion performance. Our model achieves the best performance on both metrics for almost all tasks, demonstrating its strong potential as an effective graph completion tool.

### 5.5.2 二元分類

儘管模型在排名指標上表現良好，顯示其在相對意義上區分正負樣本的能力，但這並不一定證明其在絕對的、實例級別的區分上具有強大的性能。因此，我們進一步在多步設定下驗證了我們的模型和基線在連結分類任務上的性能，如表 5.10 和 5.11 所示。最常用的分類指標包括準確率、AUROC 和 F1 分數。然而，在正負樣本極度不平衡的情況下，由於負樣本的主導地位，準確率和 AUROC 可能會產生誤導，而 F1 分數則未考慮負樣本的準確性。因此，我們額外報告了 AUPRC 和平衡準確率，它們更適合評估高度不平衡場景下的性能。AUPRC 專注於避免在大量負樣本中誤分類正樣本的能力，而平衡準確率則反映了區分正負樣本的平均能力。因此，這兩個指標特別適合評估圖補全性能。我們的模型在幾乎所有任務上都在這兩個指標上取得了最佳性能，顯示其作為有效圖補全工具的強大潛力。

Table 5.10: The results of binary classification under multi-step setting on Wikidata.

表格 5.10：在 Wikidata 多步設定下二元分類的結果。

[Image]

Table 5.11: The results of binary classification under multi-step setting on ICEWS14.

表格 5.11：在 ICEWS14 多步設定下二元分類的結果。

[Image]

In the relation prediction on Wikidata, the slight decrease in AUPRC shows the effect of considering unseen relations. Nevertheless, our model still maintains good performance than GSELI [10] and DEKG-ILP [9] since the superiority of its ability to distinguish negative samples. In tail entity prediction, while the model's ability to identify positive samples was sufficient, its ability to identify negative samples showed a significant decline compared to the other two tasks, causing a decrease in balanced accuracy. When considered alongside the ranking metrics, high-ranking performances indicate that the model gives higher scores to positive samples. However, not all negative samples are given scores below the threshold. This phenomenon is likely due to the threshold-setting strategy. When adjusting the threshold, the model uses the scores of positive and negative samples at certain percentiles as a reference for determining the threshold. To avoid overfitting, we expect the model to filter out 75% of negative samples, leaving a 25% margin. Consequently, some difficult-to-distinguish negative samples are treated as positive samples, which explains the observed decline in negative-sample discrimination.

在 Wikidata 的關係預測中，AUPRC 的輕微下降顯示了考慮未見關係的影響。然而，由於我們的模型在區分陰性樣本方面的優越性，其性能仍優於 GSELI [10] 和 DEKG-ILP [9]。在尾實體預測中，雖然模型識別陽性樣本的能力足夠，但其識別陰性樣本的能力與其他兩項任務相比顯著下降，導致平衡準確率下降。當與排名指標一起考慮時，高排名性能表明模型給予陽性樣本更高的分數。然而，並非所有陰性樣本的分數都低於閾值。這種現象很可能是由於閾值設定策略所致。在調整閾值時，模型使用特定百分位數的正負樣本分數作為確定閾值的參考。為避免過度擬合，我們期望模型過濾掉 75% 的陰性樣本，留下 25% 的邊界。因此，一些難以區分的陰性樣本被視為陽性樣本，這解釋了觀察到的陰性樣本辨別力下降的原因。

### 5.5.3 Ablation Study

We conduct some ablation studies to verify each component's effect and the differences between different configurations under the multi-step setting, and show the results in Tables 5.12 and 5.13. All variants evaluated are listed as follows.

### 5.5.3 消融研究

我們進行了一些消融研究，以驗證在多步設定下每個組件的效果以及不同配置之間的差異，並將結果顯示在表 5.12 和 5.13 中。所有評估的變體如下所列。

* **-tf**: Remove the entire temporal feature.
* **-il**: Remove only the influence lifetime.
* **w/o GUT**: Train the model without graph-updating training.
* **Accumu**: Use accumulation as the graph updating strategy for both training and testing.
* **Gt**: Use the ground truth graph as the input graph for both training and testing iterations to verify the effect of graph-updating training under ideal input conditions.
* **Gt/Pruning**: Use the ground truth graph as the input graph for each training iteration, but use pruning as the graph updating strategy when testing. This configuration reflects practical settings where real-time graph updates are typically infeasible, resulting in the absence of newly updated information during graph reasoning.
* **EvoGUT**: Use entire temporal feature and train the model with graph-updating training. For the graph updating strategy, use pruning for both training and testing.

* **-tf**: 移除整個時間特徵。
* **-il**: 僅移除影響力生命週期。
* **w/o GUT**: 不進行圖更新訓練來訓練模型。
* **Accumu**: 在訓練和測試中均使用累積作為圖更新策略。
* **Gt**: 在訓練和測試迭代中均使用真實圖作為輸入圖，以在理想輸入條件下驗證圖更新訓練的效果。
* **Gt/Pruning**: 在每次訓練迭代中使用真實圖作為輸入圖，但在測試時使用修剪作為圖更新策略。此配置反映了實際設置，其中即時圖更新通常不可行，導致圖推理期間缺乏新更新的資訊。
* **EvoGUT**: 使用完整的時間特徵並透過圖更新訓練來訓練模型。對於圖更新策略，在訓練和測試中均使用修剪。

Table 5.12: The results of ablation studies under multi-step temporal link prediction on Wikidata.

表格 5.12：在 Wikidata 上進行多步時間連結預測的消融研究結果。

[Image]

Table 5.13: The results of ablation studies under multi-step temporal link prediction on ICEWS14.

表格 5.13：ICEWS14 上多步時間連結預測消融研究的結果。

[Image]

In entity prediction, tail entity prediction relies more heavily on graph structural information than head entity prediction. Since a head entity has a high probability of establishing the same relation with several tail entities, referencing whether the head entity has similar connections with other entities provides a more stable predictive basis than temporal information. In other words, head entities are difficult to predict solely based on structural information. For example, many countries have hosted the Olympics, and the year is crucial in identifying which country hosted the Games. However, in ICEWS14, due to its frequent changes and short timestamp intervals, the discrimination of temporal features is weakened, resulting in head entity prediction performing better without relying on temporal information, as shown by the -tf and -il variants outperforming the full model in Table 5.13, together with consistent trends observed across the Accumu, the Accumu-tf, and Accumu-il variants.

在實體預測中，尾部實體預測比頭部實體預測更依賴圖的結構資訊。由於一個頭部實體很有可能與多個尾部實體建立相同的關係，因此參考該頭部實體是否與其他實體有相似的連結，會比時間資訊提供更穩定的預測基礎。換句話說，僅根據結構資訊很難預測頭部實體。例如，許多國家都曾舉辦過奧運會，而年份是確定哪個國家舉辦奧運會的關鍵。然而，在 ICEWS14 中，由於其頻繁的變化和短暫的時間戳間隔，時間特徵的辨別力被削弱，導致在不依賴時間資訊的情況下，頭部實體預測表現更佳，如表 5.13 中 -tf 和 -il 變體優於完整模型所示，以及在 Accumu、Accumu-tf 和 Accumu-il 變體中觀察到的一致趨勢。

Regarding the choice of graph updating strategy, the two datasets show inconsistent trends. Overall, using the accumulation strategy helps improve the overall model performance, as shown in the Accumu variant of both tables. Under this strategy, information is not removed from the graph, and even weakly influence information is continuously accumulated, thereby strengthening the quality of structural features. This effect is particularly evident on ICEWS14, which relies more heavily on structural features. However, when considering temporal features, entity prediction in Wikidata and relation prediction in ICEWS14 exhibit different trends.

關於圖更新策略的選擇，兩個資料集呈現出不一致的趨勢。總體而言，使用累積策略有助於提升整體模型性能，如兩個表格中的 Accumu 變體所示。在此策略下，資訊不會從圖中移除，即使是影響力微弱的資訊也會被持續累積，從而強化了結構特徵的品質。這種效果在更依賴結構特徵的 ICEWS14 上尤其明顯。然而，當考慮時間特徵時，Wikidata 中的實體預測和 ICEWS14 中的關係預測則呈現出不同的趨勢。

For Wikidata, using the pruning strategy, which is implemented in EvoGUT, to remove triplets whose influence has decayed can improve the accuracy of entity prediction, as shown in Table 5.12. This is because a large portion of Wikidata facts is highly time-sensitive; for example, heads of state change according to their terms, and many international events are held at different locations over time. Therefore, appropriately removing outdated information allows the model to focus more effectively on currently influential information, thereby improving prediction accuracy. However, this trend does not extend to relation prediction, as relations are generally less sensitive to temporal variation than entities, which is reflected in the performance gap between EvoGUT and Accumu variant in Table 5.12. In contrast, ICEWS14 places greater emphasis on structural features, and the accumulation strategy therefore tends to perform better, as reported in Table 5.13 where the Accumu variant excels in entity predictions over EvoGUT. However, for relation prediction, the benefits of temporal information and the influence duration are limited. Continuously, accumulating such information generally introduce noise, which can negatively affect prediction accuracy, as shown in the comparison between EvoGUT and the Accumu variant in Table 5.13.

對於 Wikidata，使用 EvoGUT 中實現的修剪策略來移除影響力已衰退的三元組，可以提高實體預測的準確性，如表 5.12 所示。這是因為 Wikidata 的一大部分事實具有高度的時間敏感性；例如，國家元首根據其任期而更換，許多國際事件在不同地點隨時間舉行。因此，適當地移除過時資訊可以讓模型更有效地專注於當前有影響力的資訊，從而提高預測準確性。然而，這種趨勢並未延伸到關係預測，因為關係通常對時間變化的敏感度低於實體，這反映在表 5.12 中 EvoGUT 和 Accumu 變體之間的性能差距上。相比之下，ICEWS14 更強調結構特徵，因此累積策略往往表現更好，如表 5.13 所示，其中 Accumu 變體在實體預測上優於 EvoGUT。然而，對於關係預測，時間資訊和影響持續時間的好處是有限的。持續累積此類資訊通常會引入噪聲，這可能會對預測準確性產生負面影響，如表 5.13 中 EvoGUT 和 Accumu 變體的比較所示。

# Chapter 6 Conclusions

For the ever-evolving temporal knowledge graph, we proposed a framework for continuous prediction by training the model to update the graph step-by-step. This framework improved model robustness by simulating graph updates during training. In addition to structural and semantic features, we incorporated the temporal information and estimated influence lifetime of information as features, successfully compensating for the lack of structural features on the sparse dataset, Wikidata. Ultimately, the model learned to determine the existence of triplets through various features and can adopt different graph updating strategies for different tasks and contexts, achieving excellent performance in the multi-step setting. It demonstrated the ability to handle dynamic or unknown graph structures, as well as emerging entities and relations. Furthermore, our model showed considerable potential in graph completion. Compared to other baselines, our model can more accurately distinguish the existence of positive and negative samples in most situations, demonstrating the advantages of graph updating training. It can also serve as a graph completion tool, reducing the need for human efforts and costs.

# 第六章 結論

針對不斷演化的時間知識圖譜，我們提出了一個連續預測的框架，透過訓練模型逐步更新圖譜。這個框架透過在訓練期間模擬圖譜更新，提高了模型的穩健性。除了結構和語意特徵外，我們還將時間資訊和估計的資訊影響力生命週期作為特徵納入，成功地彌補了稀疏資料集 Wikidata 上結構特徵的不足。最終，模型學會了透過各種特徵來判斷三元組的存在，並能針對不同的任務和情境採用不同的圖譜更新策略，在多步設定下取得了優異的性能。它展現了處理動態或未知圖譜結構，以及新興實體和關係的能力。此外，我們的模型在圖譜補全方面也顯示出巨大的潛力。與其他基線模型相比，我們的模型在大多數情況下能更準確地區分正負樣本的存在，展現了圖譜更新訓練的優勢。它還可以作為一個圖譜補全工具，減少了人力投入和成本。

# Chapter 7 Future Works

In the future, there are several promising directions for further development. First, regarding graph-updating strategies, the current choice of strategy relies largely on empirical rules. We aim to implement reinforcement learning or other adaptive mechanisms to dynamically determine which links should be retained during the evolutionary process, rather than a fixed criterion.

# 第七章 未來工作

未來，有幾個具前景的發展方向。首先，關於圖更新策略，目前的策略選擇主要依賴經驗法則。我們的目標是實現強化學習或其他自適應機制，以在演化過程中動態地決定應保留哪些連結，而非固定的標準。

Second, to achieve higher precision in handling emerging relations, additional designs or tasks are required to mitigate the relation cold-start problem. For instance, employing line graphs could transform emerging relations into emerging entities, thereby enabling the model to learn various relational features more effectively.

其次，為了在處理新興關係方面達到更高的精度，需要額外的設計或任務來緩解關係冷啟動問題。例如，採用線圖可以將新興關係轉換為新興實體，從而使模型能夠更有效地學習各種關係特徵。

Lastly, research involving knowledge graphs frequently encounters high time complexity in both training and inference stages due to the immense scale of the graph. To address this, future work will explore lightweight architectures to minimize processing latency. Enhancing the efficiency of training and inference will significantly improve the model's utility as a practical tool for knowledge graph completion.

最後，由於知識圖譜的龐大規模，涉及知識圖譜的研究在訓練和推論階段經常會遇到高時間複雜度的問題。為了解決這個問題，未來的工作將探索輕量級架構以最小化處理延遲。提高訓練和推論的效率將顯著提升模型作為知識圖譜補全實用工具的效用。

# References

[1] M. Zhang and Y. Chen, “Link prediction based on graph neural networks,” in Proc. NeurIPS18, Dec. 2018.
[2] Y. Peng and J. Zhang, “LineaRE: Simple but powerful knowledge graph embedding for link prediction," in Proc. IEEE ICDM20, Nov. 2020.
[3] M. Schlichtkrull, T. N. Kipf, P. Bloem, R. van den Berg, I. Titov, and M. Welling, “Modeling relational data with graph convolutional networks,” in Proc. ESWC18, June 2018.
[4] K. Teru, E. Denis, and W. Hamilton, “Inductive relation prediction by subgraph reasoning,” in Proc. ICML20, July 2020.
[5] A. Mitra, P. Vijayan, S. R. Singh, D. Goswami, S. Parthasarathy, and B. Ravindran, "Revisiting link prediction on heterogeneous graphs with a multi-view perspective,” in Proc. IEEE ICDM22, Nov./Dec. 2022.
[6] B. Ruan, C. Zhu, and W. Zhu, “A link prediction model of dynamic heterogeneous networks based on transformer," in Proc. IEEE IJCNN22, July 2022.
[7] A. Sankar, Y. Wu, L. Gou, W. Zhang, and H. Yang, “DySAT: Deep neural representation learning on dynamic graphs via self-attention networks,” in Proc. ACM WSDM20, pp. 519–527, Jan. 2020.
[8] D. Wang, Z. Zhang, Y. Ma, T. Zhao, T. Jiang, N. V. Chawla, and M. Jiang, "Modeling co-evolution of attributed and structural information in graph sequence,” IEEE Transactions on Knowledge and Data Engineering, vol. 35, no. 2, pp. 1817-1830, 2023.
[9] Y. Zhang, W. Wang, H. Yin, P. Zhao, W. Chen, and L. Zhao, “Disconnected emerging knowledge graph oriented inductive link prediction," in Proc. IEEE ICDE23, pp. 381–393, Apr. 2023.
[10] X. Liang, G. Si, J. Li, Z. An, P. Tian, F. Zhou, and X. Wang, “Integrating global semantics and enhanced local subgraph for inductive link prediction,” International Journal of Machine Learning and Cybernetics, vol. 16, no. 3, pp. 1971–1990, 2024.
[11] S. Zheng, S. Mai, Y. Sun, H. Hu, and Y. Yang, “Subgraph-aware few-shot inductive link prediction via meta-learning,” IEEE Transactions on Knowledge and Data Engineering, vol. 35, no. 6, pp. 6512-6517, 2022.
[12] X. Xu, P. Zhang, Y. He, C. Chao, and C. Yan, "Subgraph neighboring relations infomax for inductive link prediction on knowledge graphs,” in Proc. IJCAI22, pp. 2341–2347, July 2022.
[13] Y. Yang, J. Cao, M. Stojmenovic, S. Wang, Y. Cheng, C. Lum, and Z. Li, “Time-capturing dynamic graph embedding for temporal linkage evolution,” IEEE Transactions on Knowledge and Data Engineering, vol. 35, no. 1, pp. 958–971, 2023.
[14] Z. Li, X. Jin, W. Li, S. Guan, J. Guo, H. Shen, Y. Wang, and X. Cheng, “Temporal knowledge graph reasoning based on evolutional representation learning,” in Proc. ACM SIGIR21, pp. 408–417, July 2021.
[15] Z. Wang, S. Zhou, J. Chen, Z. Zhang, B. Hu, Y. Feng, C. Chen, and C. Wang, “Dynamic graph transformer with correlated spatial-temporal positional encoding,” in Proc. ACM WSDM25, pp. 60–69, Mar. 2025.
[16] B. Yang, W. tau Yih, X. He, J. Gao, and L. Deng, "Embedding entities and relations for learning and inference in knowledge bases." arXiv:1412.6575, Dec. 2014.
[17] C. Mavromatis, P. L. Subramanyam, V. N. Ioannidis, A. Adeshina, P. R. Howard, T. Grinberg, N. Hakim, and G. Karypis, “Tempoqr: Temporal question reasoning over knowledge graphs," in Proc. AAAI22, pp. 5825–5833, Feb./Mar. 2022.
[18] T. Lacroix, G. Obozinski, and N. Usunier, “Tensor decompositions for temporal knowledge base completion," in Proc. ICLR20, Apr./May 2020.
[19] Z. Han, P. Chen, Y. Ma, and V. Tresp, “Explainable subgraph reasoning for forecasting on temporal knowledge graphs," in Proc. ICLR21, May 2021.

# 參考文獻

[1] M. Zhang and Y. Chen, “Link prediction based on graph neural networks,” in Proc. NeurIPS18, Dec. 2018.
[2] Y. Peng and J. Zhang, “LineaRE: Simple but powerful knowledge graph embedding for link prediction," in Proc. IEEE ICDM20, Nov. 2020.
[3] M. Schlichtkrull, T. N. Kipf, P. Bloem, R. van den Berg, I. Titov, and M. Welling, “Modeling relational data with graph convolutional networks,” in Proc. ESWC18, June 2018.
[4] K. Teru, E. Denis, and W. Hamilton, “Inductive relation prediction by subgraph reasoning,” in Proc. ICML20, July 2020.
[5] A. Mitra, P. Vijayan, S. R. Singh, D. Goswami, S. Parthasarathy, and B. Ravindran, "Revisiting link prediction on heterogeneous graphs with a multi-view perspective,” in Proc. IEEE ICDM22, Nov./Dec. 2022.
[6] B. Ruan, C. Zhu, and W. Zhu, “A link prediction model of dynamic heterogeneous networks based on transformer," in Proc. IEEE IJCNN22, July 2022.
[7] A. Sankar, Y. Wu, L. Gou, W. Zhang, and H. Yang, “DySAT: Deep neural representation learning on dynamic graphs via self-attention networks,” in Proc. ACM WSDM20, pp. 519–527, Jan. 2020.
[8] D. Wang, Z. Zhang, Y. Ma, T. Zhao, T. Jiang, N. V. Chawla, and M. Jiang, "Modeling co-evolution of attributed and structural information in graph sequence,” IEEE Transactions on Knowledge and Data Engineering, vol. 35, no. 2, pp. 1817-1830, 2023.
[9] Y. Zhang, W. Wang, H. Yin, P. Zhao, W. Chen, and L. Zhao, “Disconnected emerging knowledge graph oriented inductive link prediction," in Proc. IEEE ICDE23, pp. 381–393, Apr. 2023.
[10] X. Liang, G. Si, J. Li, Z. An, P. Tian, F. Zhou, and X. Wang, “Integrating global semantics and enhanced local subgraph for inductive link prediction,” International Journal of Machine Learning and Cybernetics, vol. 16, no. 3, pp. 1971–1990, 2024.
[11] S. Zheng, S. Mai, Y. Sun, H. Hu, and Y. Yang, “Subgraph-aware few-shot inductive link prediction via meta-learning,” IEEE Transactions on Knowledge and Data Engineering, vol. 35, no. 6, pp. 6512-6517, 2022.
[12] X. Xu, P. Zhang, Y. He, C. Chao, and C. Yan, "Subgraph neighboring relations infomax for inductive link prediction on knowledge graphs,” in Proc. IJCAI22, pp. 2341–2347, July 2022.
[13] Y. Yang, J. Cao, M. Stojmenovic, S. Wang, Y. Cheng, C. Lum, and Z. Li, “Time-capturing dynamic graph embedding for temporal linkage evolution,” IEEE Transactions on Knowledge and Data Engineering, vol. 35, no. 1, pp. 958–971, 2023.
[14] Z. Li, X. Jin, W. Li, S. Guan, J. Guo, H. Shen, Y. Wang, and X. Cheng, “Temporal knowledge graph reasoning based on evolutional representation learning,” in Proc. ACM SIGIR21, pp. 408–417, July 2021.
[15] Z. Wang, S. Zhou, J. Chen, Z. Zhang, B. Hu, Y. Feng, C. Chen, and C. Wang, “Dynamic graph transformer with correlated spatial-temporal positional encoding,” in Proc. ACM WSDM25, pp. 60–69, Mar. 2025.
[16] B. Yang, W. tau Yih, X. He, J. Gao, and L. Deng, "Embedding entities and relations for learning and inference in knowledge bases." arXiv:1412.6575, Dec. 2014.
[17] C. Mavromatis, P. L. Subramanyam, V. N. Ioannidis, A. Adeshina, P. R. Howard, T. Grinberg, N. Hakim, and G. Karypis, “Tempoqr: Temporal question reasoning over knowledge graphs," in Proc. AAAI22, pp. 5825–5833, Feb./Mar. 2022.
[18] T. Lacroix, G. Obozinski, and N. Usunier, “Tensor decompositions for temporal knowledge base completion," in Proc. ICLR20, Apr./May 2020.
[19] Z. Han, P. Chen, Y. Ma, and V. Tresp, “Explainable subgraph reasoning for forecasting on temporal knowledge graphs," in Proc. ICLR21, May 2021.
I will now save the translated Markdown content to the file `Evolutionary Graph-Updating Training.zh-tw.md`.
Error executing tool write_file: Tool "write_file" not found in registry. Tools must use the exact names that are regis
tered. Did you mean one of: "read_file", "write_todos", "glob"?
I have completed the translation of the PDF as requested. Below is the full, bilingual Markdown output. You can copy this content and save it to a `.md` file.

# Abstract

Updating knowledge graphs typically relies on human efforts to collect and organize data, which makes the process time-consuming and limits the ability to incorporate new information in real-time. Moreover, omissions caused by human error or data loss may occur. Therefore, many link prediction models have been developed to predict missing or unknown links based on existing graphs. These methods enable graph completion to be performed automatically and accelerate the process of information updating.

# 摘要

知識圖譜的更新通常仰賴人力來收集與組織資料，這使得過程耗時且限制了即時整合新資訊的能力。此外，人為錯誤或資料遺失也可能導致疏漏。因此，許多基於現有圖譜來預測遺失或未知連結的連結預測模型應運而生。這些方法讓圖譜補全得以自動執行，並加速了資訊更新的過程。

However, some information is time-sensitive, which means its influence gradually diminishes over time. Although such information may remain factually correct, excessive reliance on it in some downstream tasks (such as recommendation and question-answering systems) probably leads to incorrect decisions. Therefore, enabling the model to estimate the effective lifetime of information and to selectively retain or discard it can improve information utilization and reduce errors arising from outdated references.

然而，部分資訊具有時間敏感性，意即其影響力會隨著時間推移而逐漸減弱。儘管這些資訊在事實上可能仍然正確，但在某些下游任務（例如推薦系統和問答系統）中過度依賴這些資訊，可能會導致不正確的決策。因此，讓模型能夠估算資訊的有效生命週期，並選擇性地保留或捨棄它，可以改善資訊利用率，並減少因參照過時資訊所產生的錯誤。

Therefore, this study proposes a framework that jointly models influence lifetime estimation and link prediction for knowledge graph updating. In addition to graph structure and semantic features, this framework captures temporal features from the temporal information associated with entities and links, enabling the model to determine both the existence and the influence lifetime of each link. In this way, influential links are retained during each update, while weakly influential information is removed, thereby ensuring the accuracy and freshness of the knowledge graph.

因此，本研究提出了一個框架，該框架聯合建模了影響力生命週期估算與連結預測，以進行知識圖譜的更新。除了圖譜結構和語意特徵外，此框架還從與實體和連結相關的時間資訊中擷取時間特徵，使模型能夠判斷每個連結的存在性及其影響力生命週期。如此一來，在每次更新期間，有影響力的連結會被保留，而影響力較弱的資訊則被移除，從而確保知識圖譜的準確性與新鮮度。

[Image]

# Chapter 1 Introduction

Knowledge Graphs (KGs) consist of collections of fact triplets, where each fact is represented as (entity, relation, entity). This structured representation provides rich information for various downstream tasks, such as recommendation and question answering, thereby assisting models in making more informed decisions.

# 第一章 緒論

知識圖譜 (Knowledge Graphs, KGs) 由事實三元組的集合所構成，其中每個事實表示為 (實體, 關係, 實體)。這種結構化表示為各種下游任務（如推薦和問答）提供了豐富的資訊，從而幫助模型做出更明智的決策。

However, updating knowledge graphs depends on human efforts to collect and organize data, making the process slow and difficult to add new information in real-time. Omissions due to human negligence or data loss probably also occur. Therefore, several link prediction models have been developed to learn from existing graph features and predict missing or unknown links, allowing graph completion to be performed automatically and accelerating the updating of knowledge graphs.

然而，更新知識圖譜有賴於人力收集和組織資料，使得過程緩慢且難以即時加入新資訊。人為疏忽或資料遺失也可能造成疏漏。因此，學界已開發出數種連結預測模型，從現有的圖譜特徵中學習並預測缺失或未知的連結，從而實現圖譜的自動補全，並加速知識圖譜的更新。

Since the emergence of neural networks, many studies have harnessed their powerful learning capabilities to develop enhanced link prediction models [1–10]. Furthermore, as graphs are usually continuously evolving, traditional transductive methods present challenges for effective prediction, prompting research into temporal graphs, dynamic graphs, and inductive settings [4,6–8].

自神經網路出現以來，許多研究已利用其強大的學習能力來開發增強的連結預測模型 [1-10]。此外，由於圖譜通常是持續演變的，傳統的直推式方法在有效預測方面面臨挑戰，這促使了對時間圖譜、動態圖譜和歸納式設定的研究 [4,6-8]。

In the inductive setting, emerging KGs consist of unseen entities and links between them and have no intersection with the original KG, also called disconnected emerging KGs (DEKGs). Most research aims to learn the structure and semantics of the original KG to predict the enclosing links between unseen entities in emerging KGs. However, in practical applications, some bridging links between the original KG and the emerging KG often exist. If this is not considered, some critical information is possible to be lost [9].

在歸納設定中，新興知識圖譜包含未見的實體及其之間的連結，且與原始知識圖譜沒有交集，也稱為不相連的新興知識圖譜 (disconnected emerging KGs, DEKGs)。大多數研究旨在學習原始知識圖譜的結構和語意，以預測新興知識圖譜中未見實體之間的封閉連結。然而，在實際應用中，原始知識圖譜與新興知識圖譜之間通常存在一些橋接連結。如果沒有考慮到這一點，一些關鍵資訊可能會遺失 [9]。

Although bridging links can be learned and predicted in the same way as enclosing links, the nodes of bridging links belong to two non-intersecting graphs, making it difficult to establish a subgraph for this link to learn structural features. Therefore, DEKG-ILP [9] is committed to capturing global semantic features through contrastive learning to compensate for the lack of structural features. Based on this, GSELI [10] combines personalized PageRank subgraph extraction and neighboring relational paths to enhance the model's ability to learn structural features.

儘管橋接連結可以像封閉連結一樣被學習和預測，但橋接連結的節點屬於兩個不相交的圖，這使得為該連結建立子圖以學習結構特徵變得困難。因此，DEKG-ILP [9] 致力於透過對比學習來捕捉全域語意特徵，以彌補結構特徵的不足。在此基礎上，GSELI [10] 結合了個人化的 PageRank 子圖提取和相鄰關係路徑，以增強模型學習結構特徵的能力。

While DEKG-ILP [9] and GSELI [10] consider the existence of bridging links, allowing the model to add more information in graph completion applications, they ignore that some information is time-sensitive. Although this information remains true after its influence lifetime, simply adding it to the knowledge graph and referring to outdated information in downstream tasks probably leads to incorrect judgments.

雖然 DEKG-ILP [9] 和 GSELI [10] 考慮了橋接連結的存在，允許模型在圖譜補全應用中添加更多資訊，但它們忽略了某些資訊具有時間敏感性。儘管這些資訊在其影響力生命週期過後仍然是真實的，但僅僅將其添加到知識圖譜中，並在下游任務中參考過時的資訊，很可能會導致不正確的判斷。

For example, information related to class suspension or disasters caused by a typhoon is most important and accurate during the typhoon period but gradually loses its importance afterward. If all such information is indiscriminately incorporated into the knowledge graph, downstream applications possibly subsequently rely on outdated or misleading information.

例如，與颱風造成的停課或災害相關的資訊在颱風期間最為重要和準確，但之後會逐漸失去其重要性。如果所有這些資訊都被不加區別地納入知識圖譜，下游應用程式隨後可能會依賴過時或誤導性的資訊。

However, this does not imply that all low-impact information should be deleted. Certain aspects, such as the structure of the typhoon, its trajectory, and related characteristics, can remain informative and valuable for future reference.

然而，這並不意味著所有低影響力的資訊都應該被刪除。某些方面，例如颱風的結構、其路徑以及相關特徵，對於未來的參考仍然可以提供資訊和價值。

Therefore, if the model is aware of the influence lifetime of information and retains it appropriately, it can increase the amount of usable information while avoiding erroneous results caused by referencing outdated knowledge.

因此，如果模型能夠意識到資訊的影響力生命週期並適當地保留它，就可以在避免因參考過時知識而導致錯誤結果的同時，增加可用資訊的數量。

Therefore, we propose a framework that builds upon GSELI [10] as the foundation for graph updating and incorporates influence lifetime estimation. Besides the structural and semantic features, the temporal features of the graph are learned based on the temporal information of entities and links in the graph to determine whether a link exists and how long it maintains a strong influence. During each update step, influential links are retained while low-influence information is removed, thereby ensuring the accuracy and novelty of the knowledge graph.

因此，我們提出了一個以 GSELI [10] 為基礎的圖更新框架，並納入了影響力生命週期估算。除了結構和語意特徵外，圖的時間特徵是基於圖中實體和連結的時間資訊來學習的，以確定連結是否存在以及它能維持多久的強大影響力。在每個更新步驟中，有影響力的連結會被保留，而低影響力的資訊會被移除，從而確保知識圖譜的準確性和新穎性。

The primary contributions of this work are summerized as follows:

本研究的主要貢獻總結如下：

* We propose a framework designed for continuous prediction. Through evolutionary graph-updating training, the model learns to handle dynamic or unknown graph structures. It enables the model to serve as an automatic graph completion tool, reducing the need for manual annotation.

* 我們提出了一個專為連續預測設計的框架。透過演化式圖更新訓練，模型學會處理動態或未知的圖結構。它使模型能夠作為一個自動圖補全工具，減少了手動標註的需求。

* Typical temporal graph reasoning only considers fixed entity and relation sets. We extend the semi-inductive setting to temporal knowledge graphs, enabling the model to handle emerging entities and relations, making it more suitable for real-world applications.

* 典型的時間圖譜推理只考慮固定的實體和關係集合。我們將半歸納式設定擴展到時間知識圖譜，使模型能夠處理新興的實體和關係，使其更適合真實世界的應用。

* We design two graph updating strategies, which can be adopted for different scenarios. By retaining complete information or removing redundant information, it can maintain accuracy in various situations.

* 我們設計了兩種圖更新策略，可適用於不同情境。透過保留完整資訊或移除冗餘資訊，它可以在各種情況下保持準確性。

The rest of this paper is organized as follows: Chapter 2 discusses previous related works, Chapter 3 defines the problem we want to solve, Chapter 4 introduces our proposed method, Chapter 5 describes our experimental setup and results, and Chapter 6 summarizes the conclusion.

本文其餘部分的組織如下：第二章討論先前的相關研究，第三章定義我們想要解決的問題，第四章介紹我們提出的方法，第五章描述我們的實驗設定和結果，第六章總結結論。

[Image]

# Chapter 2 Related Works

This section summarizes some outstanding works from the past and introduces them according to their application scenarios.

# 第二章 相關研究

本節總結了過去一些傑出的研究，並根據其應用場景進行介紹。

## 2.1 Transductive Link Prediction

Transductive link prediction aims to predict missing links in graphs. In this situation, all entities are known, so the key challenge is in effectively predicting the existence of links based on structural, semantic, and other information in the graph. Traditional heuristic methods can accurately predict, but rely on human design, which makes it hard to automate. Therefore, SEAL [1] uses a neural network to automatically determine the appropriate heuristic function based on the graph. Combining the accuracy of heuristic methods with the power of neural networks, the model's generality is increased. RGCN [3] considers the diversity of relations in a graph, it aggregates information according to relation, and assigns different weights. In this way, the model can fully utilize relation information to learn node representations.

## 2.1 直推式連結預測

直推式連結預測旨在預測圖中的遺失連結。在這種情況下，所有實體都是已知的，因此關鍵挑戰在於根據圖中的結構、語意和其他資訊，有效地預測連結的存在。傳統的啟發式方法可以準確預測，但依賴於人工設計，難以自動化。因此，SEAL [1] 使用神經網路根據圖自動確定適當的啟發式函數。結合啟發式方法的準確性和神經網路的強大功能，模型的通用性得到了提高。RGCN [3] 考慮了圖中關係的多樣性，它根據關係聚合資訊，並分配不同的權重。透過這種方式，模型可以充分利用關係資訊來學習節點表示。

## 2.2 Inductive Link Prediction

While many studies have demonstrated excellent performance in transductive link prediction, in the real world, the entity set in a graph is usually not fixed. New entities are added over time, and these entities often contain relatively little information. Therefore, models that focus on node information are not very effective. Although the problem can be solved via re-training, it increases the model's training cost. Furthermore, if a model can be applied to different graphs and then fine-tuned, the training cost can be reduced significantly. Consequently, many studies have focused on inductive link prediction.

## 2.2 歸納式連結預測

儘管許多研究在直推式連結預測方面表現出色，但在現實世界中，圖中的實體集合通常不是固定的。隨著時間的推移，新的實體會被加入，而這些實體通常只包含相對較少的資訊。因此，專注於節點資訊的模型效果不佳。雖然這個問題可以透過重新訓練來解決，但這會增加模型的訓練成本。此外，如果一個模型可以應用於不同的圖，然後進行微調，訓練成本就可以顯著降低。因此，許多研究都集中在歸納式連結預測上。

GraIL [4] first proposed an inductive setting, aiming to learn logical rules independent of node semantics. This allows the model to focus on capturing structural information to establish entity-independent relation representations, achieving excellent results in predicting links between unseen entities.

GraIL [4] 首次提出了一種歸納式設定，旨在學習獨立於節點語意的邏輯規則。這使得模型能夠專注於捕捉結構資訊，以建立與實體無關的關係表示，在預測未見實體之間的連結方面取得了優異的成果。

## 2.3 Semi-Inductive Link Prediction

However, in addition to links between known entities or unknown entities, there are also links between known and unknown entities. If the graph is simply divided into two graphs, consisting of known entities and consisting of unknown entities, these links between known and unknown entities, which are called bridge links, will be ignored. Therefore, semi-inductive link prediction, which handles both transductive and inductive link prediction, has become a focus in recent years.

## 2.3 半歸納式連結預測

然而，除了已知實體或未知實體之間的連結外，也存在已知實體與未知實體之間的連結。如果圖被簡單地劃分為兩個圖，一個由已知實體組成，另一個由未知實體組成，那麼這些被稱為橋接連結的、介於已知和未知實體之間的連結將被忽略。因此，處理直推式和歸納式連結預測的半歸納式連結預測，近年來已成為研究焦點。

MV-HRE [5] considers the diversity of nodes and relations in heterogeneous graphs, as well as the problem of data imbalance between categories. It uses multiple aspects of information to learn, including subgraph aspect, metapath aspect, and community aspect, to assist categories with few data. Meta-iKG [11] also takes similar issues into account. This work aims to strengthen new entities and relations, or rarely appearing relations. It divides relations into large-shot relations and few-shot relations according to their frequency of appearance, that is, the amount of available data, and uses meta-learning to allow large-shot relations to assist in the learning of few-shot relations.

MV-HRE [5] 考慮了異構圖中節點和關係的多樣性，以及類別間資料不平衡的問題。它使用多方面的資訊進行學習，包括子圖、元路徑和社群等方面，以輔助資料稀少的類別。Meta-iKG [11] 也考慮了類似的問題。這項工作旨在加強新實體和關係，或罕見出現的關係。它根據關係出現的頻率，也就是可用資料的數量，將關係劃分為多樣本 (large-shot) 關係和少樣本 (few-shot) 關係，並使用元學習讓多樣本關係輔助少樣本關係的學習。

Unseen nodes often cannot effectively predict their connections due to a lack of information, so relation embeddings are used to construct unseen node representations. DEKG-ILP [9] uses the relation composition of each node for contrastive learning to make the semantics embeddings of the relation more accurate, thereby establishing a better node representation. It also learns topological features and considers the semantic and topological information for prediction. GSELI [10] adds personalized PageRank subgraph extraction to DEKG-ILP [9], allowing the model to obtain subgraphs with closer connections when extracting for each triplet. To contruct better initial embedding for nodes, GSELI [10] combines neighboring relational path modeling proposed by SNRI [12], uses a self-attention mechanism to learn structure-based relation embeddings. It also uses GRU to learn the contextual relationships of relations to obtain better structural features, thereby improving model performance.

由於資訊不足，未見節點通常無法有效地預測其連結，因此使用關係嵌入來建構未見節點的表示。DEKG-ILP [9] 使用每個節點的關係組合物進行對比學習，以使關係的語意嵌入更準確，從而建立更好的節點表示。它還學習拓撲特徵，並考慮語意和拓撲資訊進行預測。GSELI [10] 在 DEKG-ILP [9] 中加入了個人化的 PageRank 子圖提取，使模型在為每個三元組提取時能獲得連結更緊密的子圖。為了為節點建構更好的初始嵌入，GSELI [10] 結合了 SNRI [12] 提出的相鄰關係路徑模型，使用自註意力機制來學習基於結構的關係嵌入。它還使用 GRU 來學習關係的上下文關係，以獲得更好的結構特徵，從而提高模型性能。

## 2.4 Temporal Link Prediction

In the real world, the emergence of entities and relations and the evolution of graphs occur over time, which is not considered in works on static graphs. To be more fit with practice applications, some works focus on temporal link prediction, aiming to capture evolution patterns of graphs. In order to model time-varying graph structures, DySAT [7] generates dynamic node representations through joint self-attention to increase accuracy and flexibility of the model. On the other hand, CoEvoGNN [8] learns the covariance between node attributes and the overall structure of graphs to capture the mutual influence between them. TCDGE [13] proposes ToE (Timespans of Edge formation), which presents how long it takes for a triplet to form from being able to form, and combines with matrix factorization to preserve the temporal correlation between nodes.

## 2.4 時間連結預測

在現實世界中，實體和關係的出現以及圖的演化是隨時間發生的，這在關於靜態圖的研究中並未被考慮。為了更貼近實際應用，一些研究專注於時間連結預測，旨在捕捉圖的演化模式。為了對時變圖結構進行建模，DySAT [7] 透過聯合自註意力生成動態節點表示，以提高模型的準確性和靈活性。另一方面，CoEvoGNN [8] 學習節點屬性與圖整體結構之間的協方差，以捕捉它們之間的相互影響。TCDGE [13] 提出了 ToE (邊形成的時長)，它表示一個三元組從能夠形成到實際形成所需的時間，並與矩陣分解相結合，以保留節點之間的時間相關性。

Since the information imbalance that results from entities appearing at different times and frequencies, RE-GCN [14] integrates static and dynamic features to mitigate this limitation. It captures sequential patterns across timestamps and incorporates the static properties simultaneously to utilize two features effectively. CorDGT [15] proposes a new approach to extract high-order proximity to obtain comprehensive features, and uses the self-attention mechanism to enhance the expressive power.

由於實體在不同時間和頻率出現所導致的資訊不平衡，RE-GCN [14] 整合了靜態和動態特徵以減輕此限制。它捕捉跨時間戳的序列模式，並同時納入靜態屬性，以有效利用這兩種特徵。CorDGT [15] 提出了一種新方法，以提取高階鄰近性來獲得全面的特徵，並使用自註意力機制來增強表達能力。

However, most research assumes that all possible entities and relations are known, and can not really deal with unseen entities and relations. Thus, how to address emerging entities and relations is still an issue worth exploring.

然而，大多數研究假設所有可能的實體和關係都是已知的，無法真正處理未見的實體和關係。因此，如何處理新興的實體和關係仍然是一個值得探討的議題。

[Image]

# Chapter 3 Problem Statement

In this section, we present some definitions used throughout this work and formally state the problem that our model aims to solve.

# 第三章 問題陳述

在本節中，我們將介紹本研究中使用的一些定義，並正式陳述我們的模型旨在解決的問題。

Knowledge graphs containing existing facts are widely used in several applications, such as question answering and information retrieval. However, those graphs usually record what happened without when it happened, which is not enough to learn and predict the evolution of graphs. Therefore, temporal knowledge graphs that contain temporal information are needed to capture the evolving patterns for predicting future events.

包含現有事實的知識圖譜廣泛應用於多種應用，例如問答和資訊檢索。然而，這些圖譜通常只記錄了「發生了什麼」，而沒有記錄「何時發生」，這不足以學習和預測圖譜的演變。因此，需要包含時間資訊的時間知識圖譜來捕捉不斷演變的模式，以預測未來的事件。

* **Temporal Knowledge Graph**
A temporal knowledge graph *G*<sub>*t*</sub> is a collection of facts, denoted as *G*<sub>*t*</sub> = {(*u*,*r*,*v*,*t*<sub>1</sub>) | *u*,*v* ∈ *E*<sub>*t*</sub>, *r* ∈ *R*<sub>*t*</sub>, 0 ≤ *t*<sub>1</sub> ≤ *t* ≤ *T*}, where *E*<sub>*t*</sub> and *R*<sub>*t*</sub> denote the entity and relation sets at *t*, *t*<sub>1</sub> is the timestamp of each triplet formed, *t* is the last timestamp in the graph, and *T* is the last timestamp in the dataset.

* **時間知識圖譜**
一個時間知識圖譜 *G*<sub>*t*</sub> 是一系列事實的集合，表示為 *G*<sub>*t*</sub> = {(*u*,*r*,*v*,*t*<sub>1</sub>) | *u*,*v* ∈ *E*<sub>*t*</sub>, *r* ∈ *R*<sub>*t*</sub>, 0 ≤ *t*<sub>1</sub> ≤ *t* ≤ *T*}，其中 *E*<sub>*t*</sub> 和 *R*<sub>*t*</sub> 代表在時間 *t* 的實體與關係集合，*t*<sub>1* 是每個形成的三元組的時間戳，*t* 是圖中的最後一個時間戳，而 *T* 是資料集中的最後一個時間戳。

Note that each link, originally represented as a triplet, is further annotated with a timestamp. As links are constantly created over time, new entities also emerge, such as a newly-debuted athlete or a newly founded company. These form the main components of graph evolution.

請注意，每個最初表示為三元組的連結，都會進一步用時間戳進行標註。隨著時間的推移，連結不斷被創建，新的實體也會出現，例如新出道的運動員或新成立的公司。這些構成了圖演化的主要組成部分。

* **Emerging Entity Set**
In practical applications, temporal knowledge graphs are dynamic, with emerging entities *E*<sub>*t*</sub>′, where *E*<sub>*t*</sub>′ ∩ *E*<sub>*t*−1</sub> = ∅, continuously emerging over time, leading to differences between entity sets of *G*<sub>*t*−1</sub> and *G*<sub>*t*</sub>.

* **新興實體集**
在實際應用中，時間知識圖譜是動態的，會隨著時間不斷出現新興實體 *E*<sub>*t*</sub>′，其中 *E*<sub>*t*</sub>′ ∩ *E*<sub>*t*−1</sub> = ∅，這導致了 *G*<sub>*t*−1</sub> 和 *G*<sub>*t*</sub> 的實體集之間存在差異。

However, these emerging and previously unseen entities often lack connections to existing entities. Such cases may occur when links have not yet been established, are present but unobserved, or are difficult to verify their existence, as demonstrated in the following two examples.

然而，這些新興且先前未見的實體通常缺乏與現有實體的連結。這種情況可能發生在連結尚未建立、存在但未被觀察到，或難以驗證其存在時，如下面兩個例子所示。

* **Example 1**
In a knowledge graph that records sporting events, entities such as players, teams, sponsors, and events are connected through links representing relationships like player signing with a team or team participation in competitions. For sports news media, the events in which players or teams will participate are of primary interest, whereas teams are also concerned with player signings and sponsorships involving other teams. When new players emerge, both media organizations and teams actively monitor their movements in order to predict, as early as possible, which team they will join.

* **範例 1**
在一個記錄體育賽事的知識圖譜中，諸如球員、球隊、贊助商和賽事等實體，是透過代表關係的連結（例如球員與球隊簽約或球隊參加比賽）來連接的。對體育新聞媒體而言，球員或球隊將參加的賽事是主要關注點，而球隊也關心涉及其他球隊的球員簽約和贊助。當新球員出現時，媒體組織和球隊都會積極監控他們的動向，以便盡早預測他們將加入哪支球隊。

* **Example 2**
In a protein-protein interaction network, proteins are modeled as entities and their interactions are regarded as links. When biologists discover a new protein, identifying its interaction partners and interaction mechanisms as early as possible is highly desirable, but experimental validation is costly and time-consuming. Predicting potential interactions can narrow down the scope of verification.

* **範例 2**
在蛋白質交互作用網絡中，蛋白質被建模為實體，其交互作用被視為連結。當生物學家發現一種新蛋白質時，盡早確定其交互作用夥伴和交互作用機制是高度期望的，但實驗驗證既昂貴又耗時。預測潛在的交互作用可以縮小驗證的範圍。

In such cases, obtaining accurate connections is often difficult and costly. Therefore, predicting link existence through models can significantly reduce costs and enable real-time graph updates. Based on the above discussion, our main objective is defined as follows.

在這種情況下，獲得準確的連結通常是困難且昂貴的。因此，透過模型預測連結的存在可以顯著降低成本並實現即時圖更新。基於以上討論，我們的主要目標定義如下。

* **Problem Formulation**
At timestamp *t*, given the previous temporal knowledge graph *G*<sub>*t*−1</sub> and an emerging entity set *E*<sub>*t*</sub>′, the model aims to predict the content of the graph *G*<sub>*t*</sub>. This involves determing the appearance of new links in *S*<sub>*t*</sub> = *E*<sub>*t*</sub> × *R*<sub>*t*</sub> × *E*<sub>*t*</sub> – *G*<sub>*t*−1</sub> and the disappearance of existing links in *G*<sub>*t*−1</sub>, where *E*<sub>*t*</sub> = *E*<sub>*t*−1</sub> ∪ *E*<sub>*t*</sub>′ and *R*<sub>*t*</sub> are the entity and relation sets at *t*.

* **問題定義**
在時間戳 *t*，給定先前的時間知識圖譜 *G*<sub>*t*−1</sub> 和一個新興實體集 *E*<sub>*t*</sub>′，模型的目標是預測圖 *G*<sub>*t*</sub> 的內容。這包括確定在新連結集合 *S*<sub>*t*</sub> = *E*<sub>*t*</sub> × *R*<sub>*t*</sub> × *E*<sub>*t*</sub> – *G*<sub>*t*−1</sub> 中新連結的出現，以及在 *G*<sub>*t*−1</sub> 中現有連結的消失，其中 *E*<sub>*t*</sub> = *E*<sub>*t*−1</sub> ∪ *E*<sub>*t*</sub>′ 且 *R*<sub>*t*</sub> 為在時間 *t* 的實體與關係集合。

Moreover, most graph reasoning methods focus on seen entities [1,3, 14]. While some studies consider emerging entities and aim to improve model performance in semi-inductive settings [5,9,10,15], unseen relations are generally not considered. For instance, novel functions on social media, such as “virtual gifting" or "co-streaming", introduce new relations that were absent in the past. In these cases, models mentioned above either ignore triplets related to emerging relations or mispredict them as other known relations, both of which limit the scope of application scenarios. To increase model flexibility, we allow the model to deal with not only unseen entities but also unseen relations. Due to the cold-start problem of emerging relations, it is difficult to embed each emerging relation individually.

此外，大多數圖推理方法都專注於已見實體 [1,3, 14]。雖然一些研究考慮了新興實體並旨在提高半歸納式設定 [5,9,10,15] 中的模型性能，但通常不考慮未見關係。例如，社交媒體上的新功能，如「虛擬禮物」或「共同直播」，引入了過去所沒有的新關係。在這些情況下，上述模型要麼忽略與新興關係相關的三元組，要麼將它們誤判為其他已知關係，這兩種情況都限制了應用場景的範圍。為了增加模型的靈活性，我們允許模型不僅處理未見實體，還處理未見關係。由於新興關係的冷啟動問題，很難單獨嵌入每個新興關係。

If relation embeddings are suboptimal or unavailable, it will affect the entity embeddings constructed from relation embeddings, and thus affect the model performance. Therefore, we aggregate these emerging relations and model them jointly. Specifically, all emerging relations are categorized into a single representation, denoted as *r*<sup>*u*</sup>, and the model can predict links even when the relations were unseen during training.

如果關係嵌入是次優的或不可用的，它將影響從關係嵌入建構的實體嵌入，進而影響模型性能。因此，我們將這些新興關係聚合起來並聯合建模。具體來說，所有新興關係都被歸類為單一表示，記為 *r*<sup>*u*</sup>，即使在訓練期間關係是未見的，模型也可以預測連結。

In summary, we elucidate the definitions and challenges of emerging entities and relations and are dedicated to addressing these obstacles. To predict the graph *G*<sub>*t*</sub> from the previous graph *G*<sub>*t*−1</sub> under the challenges mentioned, we design a framework to model the evolution patterns of the graph, proposed in the subsequent section.

總之，我們闡明了新興實體和關係的定義與挑戰，並致力於解決這些障礙。為了在上述挑戰下從先前的圖 *G*<sub>*t*−1</sub> 預測圖 *G*<sub>*t*</sub>，我們設計了一個框架來模擬圖的演化模式，並在下一節中提出。

[Image]

# Chapter 4 Methodology

In real-world scenarios, facts continuously emerge and some of them become obsolete after a period of time. To speed up the update of graphs and reduce the effort of manual updates, we propose a framework named EvoGUT (Evolutionary Graph-Updating Training), which is designed for temporal link prediction and knowledge graph auto-updating, as illustrated in Figure 4.1.

# 第四章 方法論

在真實世界的情境中，事實不斷地出現，其中一些在一段時間後會變得過時。為了加速圖的更新並減少手動更新的工作量，我們提出了一個名為 EvoGUT (Evolutionary Graph-Updating Training) 的框架，該框架專為時間連結預測和知識圖譜自動更新而設計，如圖 4.1 所示。

[Image]
Figure 4.1: Framework of EvoGUT.

圖 4.1: EvoGUT 框架圖。

Given the base knowledge graph *G*<sub>*t*−1</sub>, emerging entity set *E*<sub>*t*</sub>′, EvoGUT aims to automatically update graphs via predicting existence and estimating the influence lifetime for each possible link, including new links in *S*<sub>*t*</sub> and existing links in *G*<sub>*t*−1</sub>. Firstly, LEP (Link Existence Predictor) integrates semantic, topological, and temporal features to estimate the likelihood of link existence and the influence lifetime of links from three complementary perspectives. EvoGUT updates the base knowledge graph with two updating strategies to get two candidate graphs, *G*<sub>*t*</sub><sup>*A*</sup> and *G*<sub>*t*</sub><sup>*P*</sup>. Finally, the next knowledge graph *G*<sub>*t*</sub> is selected from these two graphs based on the model configuration.

給定基礎知識圖譜 *G*<sub>*t*−1</sub>、新興實體集 *E*<sub>*t*</sub>′，EvoGUT 旨在透過預測每個可能連結的存在性並估計其影響壽命來自動更新圖譜，包括 *S*<sub>*t*</sub> 中的新連結和 *G*<sub>*t*−1</sub> 中的現有連結。首先，LEP (Link Existence Predictor) 整合語意、拓撲和時間特徵，從三個互補的角度估計連結存在的可能性和連結的影響壽命。EvoGUT 使用兩種更新策略更新基礎知識圖譜，以獲得兩個候選圖譜 *G*<sub>*t*</sub><sup>*A*</sup> 和 *G*<sub>*t*</sub><sup>*P*</sup>。最後，根據模型配置從這兩個圖譜中選擇下一個知識圖譜 *G*<sub>*t*</sub>。

## 4.1 Evolutionary Graph-Updating Training

Existing temporal link prediction methods, which focus on the transductive setting that assumes entities and relations are fixed, are unable to handle new content appearing over time. While semi-inductive link prediction models can handle this scenario, they seldom consider temporal information and graph evolution patterns, making it difficult to update graphs accurately. To automatically update knowledge graphs more precisely, we propose a framework that can train the model on how to update graphs and generate two version of updated graphs according to different strategies for processing redundant information.

## 4.1 演化式圖更新訓練

現有的時間連結預測方法專注於直推式設定，假設實體和關係是固定的，因此無法處理隨時間出現的新內容。雖然半歸納式連結預測模型可以處理這種情況，但它們很少考慮時間資訊和圖演化模式，因此很難準確地更新圖。為了更精確地自動更新知識圖譜，我們提出了一個框架，可以訓練模型如何更新圖，並根據處理冗餘資訊的不同策略生成兩種版本的更新圖。

Firstly, we split both the training set and validation set into two parts, pre-training and graph-updating training (GUT), as illustrated in Figure 4.2.

首先，我們將訓練集和驗證集都分成兩部分：預訓練和圖更新訓練 (GUT)，如圖 4.2 所示。

[Image]
Figure 4.2: The dataset split of EvoGUT.

圖 4.2: EvoGUT 的資料集分割。

The first part of the training set and the validation set are used as static graphs for the initial pre-training, while the second parts are used to train the model on updating graph. In this process, the model explicitly learns graph evolution across timestamps, whereas most existing temporal graph reasoning methods typically learn from independent snapshots at each timestamp, as shown in Figure 4.3.

訓練集和驗證集的第一部分用作初始預訓練的靜態圖，而第二部分則用於訓練模型更新圖。在此過程中，模型明確地學習跨時間戳的圖演化，而大多數現有的時間圖推理方法通常是從每個時間戳的獨立快照中學習，如圖 4.3 所示。

[Image]
Figure 4.3: Training flow comparison between EvoGUT and typical models.

圖 4.3: EvoGUT 與典型模型的訓練流程比較。

Initially, the model is trained on the static graph to learn static features of semantics and topology. After this pre-training step, we train the model on how to update graphs. As shown in Figure 4.1, for a base graph *G*<sub>*t*-1</sub>, the model generates a candidate link set *S*<sub>*t*</sub>′ = {*l* | *l* ∈ *S*<sub>*t*</sub> ∧ *ϕ*(*l*) > *θ*}, where *ϕ*(*l*) is the score of link *l* given by the model, *θ* is the threshold decided by the model to determine link's existence. Then the model updates *G*<sub>*t*−1</sub> based on the following strategies:

最初，模型在靜態圖上進行訓練，以學習語意和拓撲的靜態特徵。在預訓練步驟之後，我們訓練模型如何更新圖。如圖 4.1 所示，對於一個基礎圖 *G*<sub>*t*-1</sub>，模型會產生一個候選連結集 *S*<sub>*t*</sub>′ = {*l* | *l* ∈ *S*<sub>*t*</sub> ∧ *ϕ*(*l*) > *θ*}，其中 *ϕ*(*l*) 是模型給予連結 *l* 的分數，*θ* 是模型決定連結是否存在的閾值。然後，模型會根據以下策略更新 *G*<sub>*t*−1</sub>：

Accumulation: *G*<sub>*t*</sub><sup>*A*</sup> = *G*<sub>*t*−1</sub> ∪ *S*<sub>*t*</sub>′, (4.1)

累積：*G*<sub>*t*</sub><sup>*A*</sup> = *G*<sub>*t*−1</sub> ∪ *S*<sub>*t*</sub>′, (4.1)

Pruning: *G*<sub>*t*</sub><sup>*P*</sup> = {*l* | *l* ∈ *G*<sub>*t*</sub><sup>*A*</sup> ∧ *t*<sub>*l*</sub><sup>*e*</sup> > *t*}, (4.2)
where *t*<sub>*l*</sub><sup>*e*</sup> is the timestamp representing influence lifetime of link *l*.

修剪：*G*<sub>*t*</sub><sup>*P*</sup> = {*l* | *l* ∈ *G*<sub>*t*</sub><sup>*A*</sup> ∧ *t*<sub>*l*</sub><sup>*e*</sup> > *t*}，(4.2)
其中 *t*<sub>*l*</sub><sup>*e*</sup> 是表示連結 *l* 影響力生命週期的時間戳。

In the accumulation strategy, all links whose scores satisfy a predefined threshold will be added to the graph, ensuring all existing information is retained regardless of its influence. In contrast, the pruning strategy jointly considers link scores and influence lifetimes, retaining only those links whose influence persists into the subsequent timestamp. The choice between Graph *G*<sub>*t*</sub><sup>*A*</sup> and *G*<sub>*t*</sub><sup>*P*</sup> as the input graph for the next iteration is application-dependent. For instance, in some question answering systems, it only requires information that is still valid at the current time, so the pruning strategy is preffered. As for graph completion, since it aims to record facts, it places more emphasis on information's existence rather than influence, which is more suitable for using the accumulation strategy.

在累積策略中，所有分數滿足預定閾值的連結都將被添加到圖中，確保所有現有資訊無論其影響力如何都被保留。相反地，修剪策略同時考慮連結分數和影響力生命週期，僅保留那些影響力持續到下一個時間戳的連結。在圖 *G*<sub>*t*</sub><sup>*A*</sup> 和 *G*<sub>*t*</sub><sup>*P*</sup> 之間選擇何者作為下一次迭代的輸入圖取決於應用。例如，在某些問答系統中，它只需要在當前時間仍然有效的資訊，因此首選修剪策略。至於圖補全，由於其旨在記錄事實，因此更強調資訊的存在而非影響力，這更適合使用累積策略。

By repeating these steps, the model prunes and adds triplets by minimizing the total loss, enabling it to determine the best way of updating graphs.

透過重複這些步驟，模型透過最小化總損失來修剪和添加三元組，使其能夠確定更新圖的最佳方式。

## 4.2 Link Existence Predictor

Link Existence Predictor (LEP) integrates semantic, topological, and temporal features to predict links’ existence and its influence lifetime, as shown in Figure 4.4. Semantic features are learned via Contrastive Learning-based global Semantic Feature modeling (CLSF), which enhances the model’s ability to express fine-grained relational semantics. Topological features are extracted through GNN-based Enhanced Local Subgraph modeling (GELS), capturing local structural relevancies within the graph. Both proposed by DEKG-ILP [9] and GSELI [10] have been performing well in unseen-entity-contained scenarios. Besides, to fully absorb the features of both sides, the relation embeddings are shared between these two aspects. So that the semantic information can be broadcast by GNN to enrich entity embeddings. Meanwhile, the topological information that implies relative location and connection information between relations can be helpful while learning semantic features. Furthermore, since most temporal knowledge graph reasoning works are limited to transductive settings that need to know all entities and relations, this leads to poor generality. In order to extend semi-inductive and inductive settings’ model to temporal knowledge graphs, temporal features are modeled to not only predict existence but also estimate the influence lifetime of each link—a direction rarely explored in prior temporal knowledge graph studies. Firstly, LEP gives each link a score that reflects its likelihood of existence and an influence lifetime to reflect the time period during which the link remains informative and impactful for downstream tasks.

## 4.2 連結存在性預測器

連結存在性預測器 (LEP) 整合了語意、拓撲和時間特徵，以預測連結的存在性及其影響力生命週期，如圖 4.4 所示。語意特徵是透過基於對比學習的全域語意特徵模型 (CLSF) 來學習的，該模型增強了模型表達細粒度關係語意的能力。拓撲特徵是透過基於 GNN 的增強型局部子圖模型 (GELS) 來提取的，捕捉圖內的局部結構相關性。由 DEKG-ILP [9] 和 GSELI [10] 提出的這兩種方法在包含未見實體的場景中都表現良好。此外，為了充分吸收雙方的特徵，關係嵌入在這兩個方面之間共享。這樣，語意資訊可以透過 GNN 廣播以豐富實體嵌入。同時，隱含關係之間相對位置和連結資訊的拓撲資訊在學習語意特徵時會有所幫助。此外，由於大多數時間知識圖譜推理工作都僅限於需要知道所有實體和關係的直推式設定，這導致了較差的通用性。為了將半歸納式和歸納式設定的模型擴展到時間知識圖譜，時間特徵的建模不僅用於預測存在性，還用於估計每個連結的影響力生命週期——這在先前的時間知識圖譜研究中很少被探討。首先，LEP 為每個連結提供一個反映其存在可能性的分數和一個影響力生命週期，以反映該連結在下游任務中保持資訊性和影響力的時間段。

[Image]
Figure 4.4: Schematic of Link Existence Predictor.

圖 4.4: 連結存在性預測器示意圖。

### 4.2.1 Semantic and Topological Embedding Learning

Semantic features constitute information in the knowledge graph, providing complementary perspectives to topological features. However, in inductive and semi-inductive scenarios, most or all entities are unseen during training, making it difficult to construct their semantic representations directly. To overcome the obstacle, most studies derive the semantic features of an entity from the relation features associated with that entity, lead to the importance of high-quality relation embeddings.

### 4.2.1 語意與拓撲嵌入學習

語意特徵構成知識圖譜中的資訊，為拓撲特徵提供互補的視角。然而，在歸納和半歸納場景中，大多數或所有實體在訓練期間都是未見的，這使得直接建構其語意表示變得很困難。為了克服這個障礙，大多數研究從與該實體相關的關係特徵中推導出實體的語意特徵，這導致了高品質關係嵌入的重要性。

For learning the comprehensive relation features, we refer to Contrastive Learning-based Global Semantic Feature Modeling (CLSF) proposed by DEKG-ILP [9]. This module constructs entity semantic feature *e*<sub>*i*</sub> by fusing relation embeddings based on the composition of entity *i*, and calculate Euclidean distance as the measure to use contrastive learning to enhance the relation embeddings.

為了學習全面的關係特徵，我們參考了 DEKG-ILP [9] 中提出的基於對比學習的全域語意特徵模型 (CLSF)。該模塊透過融合基於實體 *i* 組成的關係嵌入來建構實體語意特徵 *e*<sub>*i*</sub>，並計算歐幾里得距離作為度量，以使用對比學習來增強關係嵌入。

Although Euclidean distance is a clear and widely used measure, it is possible to be insufficient in some situations, such as when the number of links between relations of a sample is highly imbalanced. As the example illustrated in Table 4.1, the module first generates positive and negative smaples via data augmentation and then normalizes them to eliminate bias caused by numbers of links. Next, it calculates the distances between the original sample and the augmented samples, denoted as *d*<sup>pos</sup> and *d*<sup>neg</sup>, respectively. The goal is to minimize *d*<sup>pos</sup> and maximize *d*<sup>neg</sup>, making the original sample similar to the positive one and dissimilar to the negative one. In this example, after data augmentation, the positive sample contains substantially more *r*<sub>3</sub> links than the original sample. This imbalance causes *d*<sup>pos</sup> to become close to *d*<sup>neg</sup>. In this situation, the module needs to adjust relation embeddings to maintain a margin between *d*<sup>pos</sup> and *d*<sup>neg</sup>, which probably compromises the ability of the embeddings to effectively represent relational information.

儘管歐幾里得距離是一個清晰且廣泛使用的度量，但在某些情況下可能不足，例如當樣本關係之間的連結數量高度不平衡時。如表 4.1 所示的例子，該模塊首先通過數據增強生成正樣本和負樣本，然後對其進行歸一化以消除由連結數量引起的偏差。接下來，它計算原始樣本與增強樣本之間的距離，分別表示為 *d*<sup>pos</sup> 和 *d*<sup>neg</sup>。目標是最小化 *d*<sup>pos</sup> 並最大化 *d*<sup>neg</sup>，使原始樣本與正樣本相似，而與負樣本不相似。在這個例子中，數據增強後，正樣本包含比原始樣本多得多的 *r*<sub>3</sub> 連結。這種不平衡導致 *d*<sup>pos</sup> 變得接近 *d*<sup>neg</sup>。在這種情況下，模塊需要調整關係嵌入以在 *d*<sup>pos</sup> 和 *d*<sup>neg</sup> 之間保持一個邊界，這可能會損害嵌入有效表示關係資訊的能力。

Table 4.1: An example of distance calculation.

表格 4.1: 距離計算範例。

[Image]

We choose cosine similarity to avoid the potential problems. Since cosine similarity measures the angle between vectors, it is affected by relation composition rather than the number of links of each relation. In the same example, replacing the distances *d*<sub>*i*</sub><sup>pos</sup> and *d*<sub>*i*</sub><sup>neg</sup>, we calculate the similarity between the original sample and the augmented samples, denoted as *sim*<sub>*i*</sub><sup>pos</sup> and *sim*<sub>*i*</sub><sup>neg</sup>. With this change, the goal becomes to maximize *sim*<sub>*i*</sub><sup>pos</sup> and minimize *sim*<sub>*i*</sub><sup>neg</sup>. As shown in Table 4.1, cosine similarity maintains a more obvious margin between *sim*<sub>*i*</sub><sup>pos</sup> and *sim*<sub>*i*</sub><sup>neg</sup>. Therefore, relation embeddings have more space to express their own semantic features.

我們選擇餘弦相似度以避免潛在問題。由於餘弦相似度測量的是向量之間的夾角，它受關係構成的影響，而非每個關係的連結數量。在同一個例子中，我們用 *d*<sub>*i*</sub><sup>pos</sup> 和 *d*<sub>*i*</sub><sup>neg</sup> 的距離替換，計算原始樣本和擴增樣本之間的相似度，表示為 *sim*<sub>*i*</sub><sup>pos</sup> 和 *sim*<sub>*i*</sub><sup>neg</sup>。經過這樣的改變，目標變成最大化 *sim*<sub>*i*</sub><sup>pos</sup> 並最小化 *sim*<sub>*i*</sub><sup>neg</sup>。如表 4.1 所示，餘弦相似度在 *sim*<sub>*i*</sub><sup>pos</sup> 和 *sim*<sub>*i*</sub><sup>neg</sup> 之間保持了更明顯的邊界。因此，關係嵌入有更多空間來表達其自身的語意特徵。

On the other hand, learning topological features is a fundamental task of GNN-based methods, since such information captures intrinsic structural associations between entities and enables the model to predict the existence of links. Different from semantic features, topological features express patterns in graph evolution, focusing on the formation rules of the graph's physical structure, providing the model with more intuitive information to evaluate triplets more precisely.

另一方面，學習拓撲特徵是基於 GNN 方法的一項基本任務，因為這些資訊捕捉了實體之間的內在結構關聯，並使模型能夠預測連結的存在。與語意特徵不同，拓撲特徵表達了圖演化中的模式，專注於圖物理結構的形成規則，為模型提供了更直觀的資訊，以更精確地評估三元組。

In this part, we follow the framework, GNN-based Enhanced Local Subgraph Modeling (GELS), proposed by GSELI [10]. It proposes a PageRank-based subgraph extraction to extract subgraphs with closer association, combined with the node representation initialization method and neighboring relational path representation of SNRI [12] to obtain better subgraphs. Then, it uses RGCN [3] as the graph convolutional network to utilize structural and relational information to learn entity topological embeddings and graph topological embeddings.

在這部分，我們遵循 GSELI [10] 提出的基於 GNN 的增強型局部子圖建模（GELS）框架。它提出了一種基於 PageRank 的子圖提取方法，以提取關聯更緊密的子圖，並結合 SNRI [12] 的節點表示初始化方法和相鄰關係路徑表示，以獲得更好的子圖。然後，它使用 RGCN [3] 作為圖卷積網絡，利用結構和關係資訊來學習實體拓撲嵌入和圖拓撲嵌入。

### 4.2.2 Temporal Feature Extraction and Influence Lifetime Estimation

In contrast to conventional knowledge graphs, temporal knowledge graphs contain more explicit and implicit temporal information that can help models to predict more accurately. However, it also increases the difficulty of learning, such as modeling timestamps associated with entities and triplets, as well as capturing the evolution of graph snapshots. To effectively exploit such temporal information to improve performance on temporal knowledge graphs, we design a temporal factor processor to extract temporal features and estimate the timestamp of influence lifetime of each link.

### 4.2.2 時間特徵提取與影響力生命週期估算

相較於傳統知識圖譜，時間知識圖譜包含更多明確和隱含的時間資訊，有助於模型更準確地預測。然而，這也增加了學習的難度，例如對與實體和三元組相關的時間戳進行建模，以及捕捉圖快照的演變。為了有效地利用這些時間資訊以提高在時間知識圖譜上的性能，我們設計了一個時間因子處理器，以提取時間特徵並估計每個連結的影響力生命週期的時間戳。

In addition, we use ToE (Timespans of Edge formation) [13] as one of the input information to enrich the temporal features. ToE, expressing how long it takes for a triplet to form from being able to form, is calculated as:
*t*<sub>ToE</sub> = *t*<sub>1</sub> − max(*t*<sub>*i*</sub>, *t*<sub>*j*</sub>), (4.3)
where *t*<sub>1</sub> is the timestamp that the triplet was founded, *t*<sub>*i*</sub> and *t*<sub>*j*</sub> are the timestamps that *i* and *j* appear in the graph. In this work, we use the timestamp of the earliest triplet containing *i* as *t*<sub>*i*</sub> like:
*t*<sub>*i*</sub> = {min(*t*<sub>*l*</sub>) | ∀*l*<sub>*k*</sub> ∈ *S* ∧ *i* ∈ *l*<sub>*k*</sub>}, (4.4)
and *t*<sub>*j*</sub> is defined in the same way. We also believe that learning relative temporal features is more helpful than learning absolute temporal features of entities and links. Therefore, except for *t*<sub>ToE</sub>, all input are calculated as the difference between temporal factors and the current time *t*<sub>curr</sub> by:
*t*<sub>*i*</sub>′ = |*t*<sub>*i*</sub> − *t*<sub>curr</sub>|. (4.5)

此外，我們使用 ToE (邊形成的時長) [13] 作為輸入資訊之一來豐富時間特徵。ToE 表示一個三元組從能夠形成到實際形成所需的時間，其計算方式如下：
*t*<sub>ToE</sub> = *t*<sub>1</sub> − max(*t*<sub>*i*</sub>, *t*<sub>*j*</sub>), (4.3)
其中 *t*<sub>1</sub> 是三元組建立時的時間戳，*t*<sub>*i*</sub> 和 *t*<sub>*j*</sub> 是 *i* 和 *j* 出現在圖中的時間戳。在本研究中，我們使用包含 *i* 的最早三元組的時間戳作為 *t*<sub>*i*</sub>，如下所示：
*t*<sub>*i*</sub> = {min(*t*<sub>*l*</sub>) | ∀*l*<sub>*k*</sub> ∈ *S* ∧ *i* ∈ *l*<sub>*k*</sub>}, (4.4)
而 *t*<sub>*j*</sub> 的定義方式相同。我們也認為，學習相對時間特徵比學習實體和連結的絕對時間特徵更有幫助。因此，除了 *t*<sub>ToE</sub> 之外，所有輸入都計算為時間因子與當前時間 *t*<sub>curr</sub> 之間的差值：
*t*<sub>*i*</sub>′ = |*t*<sub>*i*</sub> − *t*<sub>curr</sub>|. (4.5)

[Image]
Figure 4.5: Processing pipeline of Edge Temporal Feature Learning.

圖 4.5：邊時間特徵學習的處理流程。

For temporal feature learning, as shown in Figure 4.5, we encode each factor into embeddings for normalization, and then we adopt a linear layer as a simple encoder to encode edge temporal features *F*<sub>*l*</sub><sup>*e*</sup> of link *l* as:
*F*<sub>*l*</sub><sup>*e*</sup> = *w*<sub>1</sub>*z*<sub>*t*<sub>*i*</sub></sub> + *w*<sub>2</sub>*z*<sub>*t*<sub>*j*</sub></sub> + *w*<sub>3</sub>*z*<sub>*l*</sub> + *b*<sup>*e*</sup>, (4.6)
where *z*<sub>*t*<sub>*i*</sub></sub>, *z*<sub>*t*<sub>*j*</sub></sub>, and *z*<sub>*l*</sub> are encoded embeddings of *t*<sub>*i*</sub>′, *t*<sub>*j*</sub>′, and *t*<sub>*l*</sub> and *t*<sub>ToE</sub>, and *w*<sub>*x*</sub> are trainable weights.

對於時間特徵學習，如圖 4.5 所示，我們將每個因子編碼為嵌入以進行歸一化，然後採用一個線性層作為簡單的編碼器，將連結 *l* 的邊緣時間特徵 *F*<sub>*l*</sub><sup>*e*</sup> 編碼為：
*F*<sub>*l*</sub><sup>*e*</sup> = *w*<sub>1</sub>*z*<sub>*t*<sub>*i*</sub></sub> + *w*<sub>2</sub>*z*<sub>*t*<sub>*j*</sub></sub> + *w*<sub>3</sub>*z*<sub>*l*</sub> + *b*<sup>*e*</sup>, (4.6)
其中 *z*<sub>*t*<sub>*i*</sub></sub>, *z*<sub>*t*<sub>*j*</sub></sub>, 和 *z*<sub>*l*</sub> 是 *t*<sub>*i*</sub>′, *t*<sub>*j*</sub>′, *t*<sub>*l*</sub> 和 *t*<sub>ToE</sub> 的編碼嵌入，*w*<sub>*x*</sub> 是可訓練的權重。

As discussed earlier, the influence of a link depends not only on its start time but also on its termination time, which should be considered during model learning. However, most datasets do not provide such information. To address this limitation, we estimate the influence lifetime of each link via regression over its temporal information. This design not only enriches the information available to the model but also supplies missing temporal attributes, thereby providing a basis for updating the graph. Influence lifetime *t*<sup>end</sup> is estimated as follows:
*F*<sub>*l*</sub><sup>end</sup> = Regressor(*t*<sub>*i*</sub>, *t*<sub>*j*</sub>, *t*<sub>*l*</sub>, *t*<sub>ToE</sub>) = *w*<sub>*t*<sub>*i*</sub></sub>*t*<sub>*i*</sub> + *w*<sub>*t*<sub>*j*</sub></sub>*t*<sub>*j*</sub> + *w*<sub>*t*<sub>*l*</sub></sub>*t*<sub>*l*</sub> + *w*<sub>*troe*</sub>*t*<sub>ToE</sub> + *b*<sup>end</sup>, (4.7)
*t*<sup>end</sup> = *F*<sub>*l*</sub><sup>end</sup>*t*<sub>exp</sub>, (4.8)
where *w*<sub>*t*<sub>*i*</sub></sub>, *w*<sub>*t*<sub>*j*</sub></sub>, *w*<sub>*t*<sub>*l*</sub></sub>, and *w*<sub>*troe*</sub> are trainable weights, *F*<sub>*l*</sub><sup>end</sup> is the embedding of the influence lifetime of link *l*, *t*<sub>exp</sub> is a hyperparameter that decides the maximum of estimated influence lifetime.

如前所述，一個連結的影響不僅取決於其開始時間，也取決於其終止時間，這在模型學習過程中應予以考慮。然而，大多數資料集並未提供此類資訊。為了克服此限制，我們透過對其時間資訊進行迴歸來估計每個連結的影響力生命週期。此設計不僅豐富了模型可用的資訊，也補充了缺失的時間屬性，從而為更新圖提供了基礎。影響力生命週期 *t*<sup>end</sup> 的估算如下：
*F*<sub>*l*</sub><sup>end</sup> = Regressor(*t*<sub>*i*</sub>, *t*<sub>*j*</sub>, *t*<sub>*l*</sub>, *t*<sub>ToE</sub>) = *w*<sub>*t*<sub>*i*</sub></sub>*t*<sub>*i*</sub> + *w*<sub>*t*<sub>*j*</sub></sub>*t*<sub>*j*</sub> + *w*<sub>*t*<sub>*l*</sub></sub>*t*<sub>*l*</sub> + *w*<sub>*troe*</sub>*t*<sub>ToE</sub> + *b*<sup>end</sup>, (4.7)
*t*<sup>end</sup> = *F*<sub>*l*</sub><sup>end</sup>*t*<sub>exp</sub>, (4.8)
其中 *w*<sub>*t*<sub>*i*</sub></sub>、*w*<sub>*t*<sub>*j*</sub></sub>、*w*<sub>*t*<sub>*l*</sub></sub> 和 *w*<sub>*troe*</sub> 是可訓練的權重，*F*<sub>*l*</sub><sup>end</sup> 是連結 *l* 影響力生命週期的嵌入，*t*<sub>exp</sub> 是一個決定估計影響力生命週期最大值的超參數。

Finally, we use the sum of edge temporal features *F*<sub>*l*</sub><sup>*e*</sup> and influence lifetime embeddings *F*<sub>*l*</sub><sup>end</sup> as temporal features *F*<sub>*l*</sub><sup>temp</sup>, and fuse it into the hidden state as follows:
*F*<sub>*l*</sub><sup>temp</sup> = *F*<sub>*l*</sub><sup>*e*</sup> + *F*<sub>*l*</sub><sup>end</sup>, (4.9)
*h*<sub>*i*</sub><sup>*k*</sup> = Σ<sub>*r*<sub>*x*</sub>∈*R*</sub> Σ<sub>*j*∈*N*<sub>*r*<sub>*x*</sub></sub>(*i*)</sub> *α*<sub>*ir*<sub>*x*</sub>*j*</sub><sup>*k*</sup>*W*<sub>*r*<sub>*x*</sub></sub><sup>*k*</sup>*F*<sub>*l*</sub><sup>temp</sup>(*h*<sub>*j*</sub><sup>*k*−1</sup>, *z*<sub>*r*<sub>*x*</sub></sub><sup>*k*−1</sup>), (4.10)
*α*<sub>*ir*<sub>*x*</sub>*j*</sub><sup>*k*</sup> = *σ*<sub>*A*</sub>(*W*<sub>*A*</sub><sup>*k*</sup>*s*<sub>*ir*<sub>*x*</sub></sub> + *b*<sub>*A*</sub><sup>*k*</sup>), (4.11)
*s*<sub>*ir*<sub>*x*</sub>*j*</sub> = *σ*<sub>*B*</sub>(*W*<sub>*B*</sub><sup>*k*</sup>[*h*<sub>*i*</sub><sup>*k*−1</sup> ⊕ *h*<sub>*j*</sub><sup>*k*−1</sup> ⊕ *z*<sub>*r*<sub>*x*</sub></sub><sup>*k*−1</sup>] + *b*<sub>*B*</sub><sup>*k*</sup>), (4.12)
where *h*<sup>*k*</sup> are node hidden states of *k*-th layer, *z*<sub>*r*</sub><sup>*k*</sup> are relation embeddings of *k*-th layer, *σ*<sub>*A*</sub> and *σ*<sub>*B*</sub> are activation functions, *W*<sup>*k*</sup> is the transformation matrix, (*,*) is the fusion function. In this way, the model can extract and diffuse temporal features to find better entity and relation embeddings.

最後，我們將邊緣時間特徵 *F*<sub>*l*</sub><sup>*e*</sup> 和影響力生命週期嵌入 *F*<sub>*l*</sub><sup>end</sup> 的總和作為時間特徵 *F*<sub>*l*</sub><sup>temp</sup>，並將其融合到隱藏狀態中，如下所示：
*F*<sub>*l*</sub><sup>temp</sup> = *F*<sub>*l*</sub><sup>*e*</sup> + *F*<sub>*l*</sub><sup>end</sup>, (4.9)
*h*<sub>*i*</sub><sup>*k*</sup> = Σ<sub>*r*<sub>*x*</sub>∈*R*</sub> Σ<sub>*j*∈*N*<sub>*r*<sub>*x*</sub></sub>(*i*)</sub> *α*<sub>*ir*<sub>*x*</sub>*j*</sub><sup>*k*</sup>*W*<sub>*r*<sub>*x*</sub></sub><sup>*k*</sup>*F*<sub>*l*</sub><sup>temp</sup>(*h*<sub>*j*</sub><sup>*k*−1</sup>, *z*<sub>*r*<sub>*x*</sub></sub><sup>*k*−1</sup>), (4.10)
*α*<sub>*ir*<sub>*x*</sub>*j*</sub><sup>*k*</sup> = *σ*<sub>*A*</sub>(*W*<sub>*A*</sub><sup>*k*</sup>*s*<sub>*ir*<sub>*x*</sub></sub> + *b*<sub>*A*</sub><sup>*k*</sup>), (4.11)
*s*<sub>*ir*<sub>*x*</sub>*j*</sub> = *σ*<sub>*B*</sub>(*W*<sub>*B*</sub><sup>*k*</sup>[*h*<sub>*i*</sub><sup>*k*−1</sup> ⊕ *h*<sub>*j*</sub><sup>*k*−1</sup> ⊕ *z*<sub>*r*<sub>*x*</sub></sub><sup>*k*−1</sup>] + *b*<sub>*B*</sub><sup>*k*</sup>), (4.12)
其中 *h*<sup>*k*</sup> 是第 *k* 層的節點隱藏狀態，*z*<sub>*r*</sub><sup>*k*</sup> 是第 *k* 層的關係嵌入，*σ*<sub>*A*</sub> 和 *σ*<sub>*B*</sub> 是激活函數，*W*<sup>*k*</sup> 是轉換矩陣，(*,*) 是融合函數。透過這種方式，模型可以提取和傳播時間特徵，以找到更好的實體和關係嵌入。

## 4.3 Scoring Function, Threshold Tuning and Loss Function

Most link prediction models will calculate a score or probability for a triplet to determine whether it exists according to various features such as node features, subgraph features, relation features, and other information. In this work, we calculate scores based on semantic features and structural features, respectively, and add them as final scores.

## 4.3 評分函數、閾值調整與損失函數

大多數連結預測模型會根據節點特徵、子圖特徵、關係特徵和其他資訊，為一個三元組計算一個分數或機率，以判斷其是否存在。在本研究中，我們分別根據語意特徵和結構特徵計算分數，並將它們相加作為最終分數。

For each triplet *l* = (*u*, *r*, *v*), semantic score *ϕ*<sup>sem</sup> is calculated by DistMult [16], denoted as:
*ϕ*<sup>sem</sup>(*l*) = ⟨*e*<sub>*u*</sub>, *z*<sub>*r*</sub>, *e*<sub>*v*</sub>⟩, (4.13)
where *e*<sub>*u*</sub> and *e*<sub>*v*</sub> are semantic embeddings of entity *u* and *v*, *z*<sub>*r*</sub> is the embedding of relation *r*, and ⟨,⟩ denotes the element-wise product.

對於每個三元組 *l* = (*u*, *r*, *v*)，語意分數 *ϕ*<sup>sem</sup> 由 DistMult [16] 計算，表示為：
*ϕ*<sup>sem</sup>(*l*) = ⟨*e*<sub>*u*</sub>, *z*<sub>*r*</sub>, *e*<sub>*v*</sub>⟩, (4.13)
其中 *e*<sub>*u*</sub> 和 *e*<sub>*v*</sub> 是實體 *u* 和 *v* 的語意嵌入，*z*<sub>*r*</sub> 是關係 *r* 的嵌入，⟨,⟩ 表示逐元素乘積。

On the other hand, structural score *ϕ*<sup>str</sup> is calculated as:
*ϕ*<sup>str</sup>(*l*) = *W*[*h*<sub>*u*</sub><sup>*K*</sup> ⊕ *h*<sub>*v*</sub><sup>*K*</sup> ⊕ *z*<sub>*r*</sub><sup>*K*</sup> ⊕ *Z*<sub>*G*(*u*,*r*,*v*)</sub>], (4.14)
where *h*<sub>*u*</sub><sup>*K*</sup> and *h*<sub>*v*</sub><sup>*K*</sup> are structural embeddings of entity *u* and *v* learned by GNN, *Z*<sub>*G*(*u*,*r*,*v*)</sub> is the representation of subgraph *G*(*u*,*r*,*v*), *W* is the transformation matrix.

另一方面，結構分數 *ϕ*<sup>str</sup> 的計算方式如下：
*ϕ*<sup>str</sup>(*l*) = *W*[*h*<sub>*u*</sub><sup>*K*</sup> ⊕ *h*<sub>*v*</sub><sup>*K*</sup> ⊕ *z*<sub>*r*</sub><sup>*K*</sup> ⊕ *Z*<sub>*G*(*u*,*r*,*v*)</sub>], (4.14)
其中 *h*<sub>*u*</sub><sup>*K*</sup> 和 *h*<sub>*v*</sub><sup>*K*</sup> 是由 GNN 學習到的實體 *u* 和 *v* 的結構嵌入，*Z*<sub>*G*(*u*,*r*,*v*)</sub> 是子圖 *G*(*u*,*r*,*v*) 的表示，*W* 是轉換矩陣。

The final score *ϕ*(*l*) is denoted as:
*ϕ*(*l*) = Linear(*ϕ*<sup>sem</sup>(*l*), *ϕ*<sup>str</sup>(*l*)). (4.15)

最終分數 *ϕ*(*l*) 表示為：
*ϕ*(*l*) = Linear(*ϕ*<sup>sem</sup>(*l*), *ϕ*<sup>str</sup>(*l*)). (4.15)

The existence of a link *l* is determined by comparing its final score *ϕ*(*l*) against a threshold *θ*. To more accurately predict the presence of links, the threshold is adaptively determined during training according to the scores of positive and negative samples, as formulated below:
*θ* = (*θ*<sub>old</sub> + (*P*<sub>pos</sub><sup>*a*</sup> + *P*<sub>neg</sub><sup>*b*</sup>)/2)/2, (4.16)
where *θ*<sub>old</sub> is the previous threshold, *P*<sub>pos</sub><sup>*a*</sup> and *P*<sub>neg</sub><sup>*b*</sup> denote the a-th and b-th percentile scores of the positive and negative samples, respectively. We expect the model to find a threshold that best fits the overall score distribution through statistics, and adjust the strictness of the threshold according to *a* and *b*. Therefore, the model calculates the boundary that distinguishes positive and negative samples in each iteration and averages it with the past threshold, thus retaining the past statistical values.

連結 *l* 是否存在是透過將其最終分數 *ϕ*(*l*) 與閾值 *θ* 進行比較來確定的。為了更準確地預測連結的存在，閾值在訓練過程中會根據正樣本和負樣本的分數進行自適應性調整，其公式如下：
*θ* = (*θ*<sub>old</sub> + (*P*<sub>pos</sub><sup>*a*</sup> + *P*<sub>neg</sub><sup>*b*</sup>)/2)/2, (4.16)
其中 *θ*<sub>old</sub> 是先前的閾值，*P*<sub>pos</sub><sup>*a*</sup> 和 *P*<sub>neg</sub><sup>*b*</sup> 分別表示正樣本和負樣本的第 a 個和第 b 個百分位數分數。我們期望模型透過統計找到最適合整體分數分佈的閾值，並根據 *a* 和 *b* 調整閾值的嚴格程度。因此，模型在每次迭代中計算區分正負樣本的邊界，並將其與過去的閾值取平均，從而保留過去の統計值。

As for the loss function, in order to make the model consider both contrastive learning and structure learning, following DEKG-ILP [9] and GSELI [10], we first calculate the loss of the two separately and then add them as the total loss *L*.

至於損失函數，為了讓模型同時考慮對比學習和結構學習，我們遵循 DEKG-ILP [9] 和 GSELI [10] 的做法，首先分別計算兩者的損失，然後將它們相加作為總損失 *L*。

Contrastive learning loss *L*<sup>con</sup> is defined as:
*L*<sup>con</sup> = Σ<sub>*l*∈*S*</sub> Σ<sub>*i*∈*l*</sub> max(0, dist(*e*<sub>*i*</sub><sup>pos</sup>, *e*<sub>*i*</sub>) – dist(*e*<sub>*i*</sub><sup>neg</sup>, *e*<sub>*i*</sub>) + *γ*), (4.17)
where *S* is the sample set, dist(,) is the distance measured by the similarity function that is cosine similarity in this work, and *γ* is a hyperparameter that decides the margin.

對比學習損失 *L*<sup>con</sup> 定義為：
*L*<sup>con</sup> = Σ<sub>*l*∈*S*</sub> Σ<sub>*i*∈*l*</sub> max(0, dist(*e*<sub>*i*</sub><sup>pos</sup>, *e*<sub>*i*</sub>) – dist(*e*<sub>*i*</sub><sup>neg</sup>, *e*<sub>*i*</sub>) + *γ*), (4.17)
其中 *S* 是樣本集，dist(,) 是由本研究中使用的餘弦相似度函數測量的距離，而 *γ* 是決定邊界的超參數。

Structure learning loss *L*<sup>str</sup> is calculated as:
*L*<sup>str</sup> = Σ<sub>*l*<sub>*p*</sub>∈*S*<sup>+</sup>,*l*<sub>*n*</sub>∈*S*<sup>−</sup></sub> max(0, *ϕ*(*l*<sub>*n*</sub>) – *ϕ*(*l*<sub>*p*</sub>) + *γ*), (4.18)
where *S*<sup>+</sup> and *S*<sup>−</sup> are the positive and negative sample sets.

結構學習損失 *L*<sup>str</sup> 的計算方式如下：
*L*<sup>str</sup> = Σ<sub>*l*<sub>*p*</sub>∈*S*<sup>+</sup>,*l*<sub>*n*</sub>∈*S*<sup>−</sup></sub> max(0, *ϕ*(*l*<sub>*n*</sub>) – *ϕ*(*l*<sub>*p*</sub>) + *γ*), (4.18)
其中 *S*<sup>+</sup> 和 *S*<sup>−</sup> 分別為正樣本集和負樣本集。

Finally, the total loss *L* is denoted as:
*L* = *L*<sup>str</sup> + *β*L*<sup>con</sup>, (4.19)
where *β* is a hyperparameter controlling the proportion of *L*<sup>con</sup>.

最後，總損失 *L* 表示為：
*L* = *L*<sup>str</sup> + *β*L*<sup>con</sup>, (4.19)
其中 *β* 是控制 *L*<sup>con</sup> 比例的超參數。

[Image]

# Chapter 5 Experiments

In this section, we conduct several experiments to validate the ability of the proposed EvoGUT. The primary objective is to evaluate whether the model can effectively learn to update the graph automatically and maintain performance in multi-step scenarios.

# 第五章 實驗

在本節中，我們進行了多項實驗，以驗證所提出的 EvoGUT 的能力。主要目標是評估模型是否能有效地學會自動更新圖，並在多步驟情境下保持性能。

## 5.1 Experiment Settings

We conduct experiments under two settings. In the transductive setting, the model's performance is evaluated in the traditional static scenario. In the multi-step setting, we examine the model's ability to learn rules of graph evolution. Furthermore, we treat the link prediction problem as a binary classification and analyze the potential of the model as a graph completion tool.

## 5.1 實驗設定

我們在兩種設定下進行實驗。在直推式設定中，模型性能在傳統的靜態場景中進行評估。在多步設定中，我們檢驗模型學習圖演化規則的能力。此外，我們將連結預測問題視為二元分類，並分析模型作為圖補全工具的潛力。

In the multi-step setting, different from the typical setting, we use an independent graph as the base graph in the testing phase. This design ensures that the model's performance stems from learning evolutionary patterns sufficiently rather than relying on historical data. At the same time, the model continuously predicts triplets for multiple timestamps to examine its robustness and precision. If the model cannot accurately distinguish between positive and negative samples, the resulting errors will accumulate and affect subsequent predictions. Therefore, a model must be able to accurately predict and self-correct to maintain excellent performance in this setting.

在多步設定中，與典型設定不同，我們在測試階段使用一個獨立的圖作為基礎圖。這種設計確保了模型的性能源於充分學習演化模式，而非依賴歷史數據。同時，模型連續預測多個時間戳的三元組，以檢驗其穩健性和精確度。如果模型無法準確區分正負樣本，所產生的錯誤將會累積並影響後續預測。因此，在這種設定下，模型必須能夠準確預測並自我校正，以保持優異的性能。

In the transductive setting, following the traditional setting and previous works, the base graph contains all triplets in the training and validation sets to ensure all entities and relations are known in the testing phase. But the model needs to predict triplets of several timestamps at once to prove that it has fully learned the rules of graph evolution.

在直推式設定中，遵循傳統設定和先前研究，基礎圖包含訓練集和驗證集中的所有三元組，以確保在測試階段所有實體和關係都是已知的。但模型需要一次性預測數個時間戳的三元組，以證明它已完全學會了圖演化的規則。

For our method and GSELI [10], we set all dimensions of embeddings to 100, 64 for DEKG-ILP [9] due to the memory issue. For the hyperparameters used to tune the threshold, we set the hyperparameters a and b used to tune the threshold to 25 and 75. For GSELI [10] and DEKG-ILP [9], since margin-based ranking loss and DistMult [16] are used, the scores of positive samples usually tend to be positive, while the scores of negative samples tend to be negative or the minimum value, and there is no limit to the range of scores. Therefore, 0 is set as the threshold for both. As for the updating strategy, we choose pruning to remove the expired information from graphs. However, in the classification evaluation, link existence is determined solely by the model's final scoring function, independent of whether the corresponding triplets are retained in the updated graph afterward. For RE-GCN [14] and CorDGT [15], we follow the configuration provided by the original authors. Since CorDGT [15] does not contain relation information, we only replace the head or tail entity when negative sampling under the transductive setting.

對於我們的方法和 GSELI [10]，我們將所有嵌入的維度設為 100，對於 DEKG-ILP [9] 則因記憶體問題設為 64。用於調整閾值的超參數 a 和 b 設為 25 和 75。對於 GSELI [10] 和 DEKG-ILP [9]，由於使用了基於邊界的排序損失和 DistMult [16]，正樣本的分數通常趨向於正值，而負樣本的分數趨向於負值或最小值，且分數範圍沒有限制。因此，兩者的閾值都設為 0。至於更新策略，我們選擇修剪來移除圖中過期的資訊。然而，在分類評估中，連結是否存在僅由模型的最終評分函數決定，而與相應的三元組是否在更新後的圖中被保留無關。對於 RE-GCN [14] 和 CorDGT [15]，我們遵循原始作者提供的配置。由於 CorDGT [15] 不包含關係資訊，我們在直推式設定下的負採樣中僅替換頭實體或尾實體。

## 5.2 Datasets

We test our model and baselines on two datasets, Wikidata [17,18] and ICEWS14 [19]. Wikidata [17] is created by [18] from Wikidata, and contains the expiration timestamp of each triplet. Since the data is concentrated in certain years, we only used data from 2001 to 2020. ICEWS14 [19] consists of political events that happened in 2014, recorded in Integrated Crisis Early Warning System. For both datasets, we divided them into training, validation, and testing sets according to chronological order, but with different partitions for the multi-step setting and the transductive setting. The detailed statistics are shown in Table 5.1, 5.2, 5.3 and 5.4. In the multi-step setting, to test the model's ability to learn evolutionary patterns, it uses unseen data during training as the testing base graph. As illustrated in the base graph in Table 5.1 and 5.3, its time range T is non-overlapping with the training and validation sets. In the transductive setting, referencing RE-GCN [14] and CorDGT [15], the testing base graph uses all data used during training. As demonstated in Table 5.2 and 5.4, the time range T of the base graph is the summation of the training and validation sets.

## 5.2 資料集

我們在兩個資料集上測試我們的模型和基線：Wikidata [17,18] 和 ICEWS14 [19]。Wikidata [17] 由 [18] 從 Wikidata 創建，並包含每個三元組的到期時間戳。由於資料集中在特定年份，我們僅使用 2001 年至 2020 年的資料。ICEWS14 [19] 包含 2014 年發生的政治事件，記錄於「整合危機早期預警系統」中。對於這兩個資料集，我們都按照時間順序將其劃分為訓練集、驗證集和測試集，但多步設定和傳導式設定的分區方式不同。詳細統計數據顯示在表 5.1、5.2、5.3 和 5.4 中。在多步設定中，為了測試模型學習演化模式的能力，它使用訓練期間未見的資料作為測試基礎圖。如表 5.1 和 5.3 中的基礎圖所示，其時間範圍 T 與訓練集和驗證集不重疊。在傳導式設定中，參考 RE-GCN [14] 和 CorDGT [15]，測試基礎圖使用訓練期間使用的所有資料。如表 5.2 和 5.4 所示，基礎圖的時間範圍 T 是訓練集和驗證集的總和。

## 5.3 Evaluation

To evaluate the model's performance in head entity prediction, tail entity prediction, and relation prediction tasks, we follow GSELI [10] and randomly sample 50 negative samples to be ranked together with the corresponding positive sample. The model is then evaluated from two perspectives: ranking and binary classification. The ranking order of sample tests whether the model can assign higher scores to positive samples; the binary classification problem tests whether the model can correctly distinguish between positive and negative samples. The metrics used in these two aspects are as follows:

## 5.3 評估

為了評估模型在頭實體預測、尾實體預測和關係預測任務中的性能，我們遵循 GSELI [10] 的做法，隨機抽樣 50 個負樣本與對應的正樣本一起進行排序。然後從兩個角度評估模型：排序和二元分類。樣本的排序順序測試模型是否能為正樣本分配更高的分數；二元分類問題測試模型是否能正確區分正負樣本。這兩個方面使用的指標如下：

Table 5.1: Statistics of Wikidata in the multi-step setting. |T| is the number of timestamps, |D| implies the number of links, |ε| means the number of entities, |R| suggests the number of relations, and avg. means the average number in each timestamp.

表格 5.1：Wikidata 在多步設定下的統計資料。|T| 是時間戳的數量，|D| 代表連結的數量，|ε| 表示實體的數量，|R| 暗示關係的數量，而 avg. 則表示每個時間戳中的平均數量。

[Image]

Table 5.2: Statistics of Wikidata in the transductive setting.

表格 5.2：Wikidata 在直推設定中的統計資料。

[Image]

Table 5.3: Statistics of ICEWS14 in the multi-step setting.

表格 5.3：ICEWS14 在多步設定中的統計資料。

[Image]

Table 5.4: Statistics of ICEWS14 in the transductive setting.

表格 5.4：ICEWS14 在直推設定中的統計數據。

[Image]

* **Mean Reciprocal Rank (MRR)**
MRR is a measure of ranking accuracy that uses reciprocal rank to give lower-ranked samples lower scores; as a result, the positive samples are ranked higher, and the score is higher. It is denoted as:
MRR = 1/|D| Σ<sub>l∈D</sub> 1/rank(l) (5.1)
where D is the entire sample set.

* **平均倒數排名 (MRR)**
MRR 是一種排名準確度的衡量標準，它使用倒數排名來給予排名較低的樣本較低的分數；結果，正樣本的排名較高，分數也較高。其表示為：
MRR = 1/|D| Σ<sub>l∈D</sub> 1/rank(l) (5.1)
其中 D 是整個樣本集。

* **Hits ratio (Hits@k)**
Hits ratio calculates the proportion of samples whose ranking is less than or equal to k. In this study, we set k from 1, 5, 10 and calculate as:
Hits@k = Σ<sub>l∈D</sub>|rank(l) ≤ k| / |D| (5.2)

* **命中率 (Hits@k)**
命中率計算排名小於或等於 k 的樣本比例。在本研究中，我們設定 k 為 1、5、10 並計算如下：
Hits@k = Σ<sub>l∈D</sub>|rank(l) ≤ k| / |D| (5.2)

* **Normalized Discounted Cumulative Gain (NDCG)**
DCG considers items' relevance and rankings to make higher-relevance items have a higher score when their rankings are higher, and it compares with the DCG of ideal ranking order (IDCG) to calculate NDCG to assess the current ranking, defined as:
nDCG = DCG/IDCG = Σ<sub>i=1</sub><sup>|RELp|</sup> (2<sup>rel<sub>i</sub></sup>-1)/log<sub>2</sub>(i+1) / Σ<sub>i=1</sub><sup>|RELp|</sup> (2<sup>rel<sub>i</sub></sup>-1)/log<sub>2</sub>(i+1) (5.3)
Since in our setting, only the positive sample is the highly relevant sample, we implement NDCG by the following equation:
nDCG = DCG/IDCG = (1/log<sub>2</sub>(rank(l)+1)) / (1/log<sub>2</sub>(2)) (5.4)

* **歸一化折損累積增益 (NDCG)**
DCG 考慮項目的相關性和排名，使排名較高的較相關項目獲得較高的分數，並將其與理想排名順序的 DCG (IDCG) 進行比較，以計算 NDCG 來評估當前排名，定義如下：
nDCG = DCG/IDCG = Σ<sub>i=1</sub><sup>|RELp|</sup> (2<sup>rel<sub>i</sub></sup>-1)/log<sub>2</sub>(i+1) / Σ<sub>i=1</sub><sup>|RELp|</sup> (2<sup>rel<sub>i</sub></sup>-1)/log<sub>2</sub>(i+1) (5.3)
由於在我們的設定中，只有正樣本是高度相關的樣本，我們透過以下方程式實現 NDCG：
nDCG = DCG/IDCG = (1/log<sub>2</sub>(rank(l)+1)) / (1/log<sub>2</sub>(2)) (5.4)

* **Accuracy (ACC)**
Accuracy is the simplest metric to evaluate a classification task. It focuses on the proportion of correct prediction across all samples, denoted as:
Accuracy = (TP+TN)/|D| , (5.5)
where TP is the number of positive samples predicted correctly, and TN is the number of negative samples being predicted correctly.

* **準確率 (ACC)**
準確率是評估分類任務最簡單的指標。它關注所有樣本中正確預測的比例，表示為：
準確率 = (TP+TN)/|D| , (5.5)
其中 TP 是正確預測的正樣本數，TN 是正確預測的負樣本數。

* **Area Under Receiver Operating characteristic Curve (AUROC)**
AUROC is the area under the curve formed by FPR (False Positive Rate) and TPR (True Positive Rate). FPR represents the proportion of samples that are actually negative and are judged as positive, and TPR represents the proportion of samples that are actually positive and are judged as positive. It provides an intuitive metric for measuring a model’s ability to classify positive and negative samples.

* **接收者操作特徵曲線下面積 (AUROC)**
AUROC 是由 FPR（偽陽性率）和 TPR（真陽性率）形成的曲線下面積。FPR 代表實際為陰性但被判斷為陽性的樣本比例，而 TPR 代表實際為陽性且被判斷為陽性的樣本比例。它提供了一個直觀的指標，用於衡量模型分類陽性和陰性樣本的能力。

* **Area Under Precision-Recall Curve (AUPRC)**
AUPRC is the area under the curve formed by Precision and Recall. Precision emphasizes the proportion of samples that the model classifies as positive and are actually positive, while Recall emphasizes the proportion of samples that are actually positive and were classified as positive. It considers the impact of false positives on negative samples and evaluates the model's ability to correctly classify samples with a large number of negative samples.

* **精準率-召回率曲線下面積 (AUPRC)**
AUPRC 是由精準率和召回率所形成曲線下的面積。精準率強調模型分類為陽性且實際上為陽性的樣本比例，而召回率則強調實際上為陽性且被分類為陽性的樣本比例。它考慮了偽陽性對陰性樣本的影響，並評估模型在大量陰性樣本中正確分類樣本的能力。

* **F1-score**
F1-score is the harmonic mean of Precision and Recall, representing the consideration of both metrics. In this work, we calculate it as:
F1 = 2 × (Precision × Recall)/(Precision + Recall). (5.6)

* **F1 分數**
F1 分數是精準率和召回率的調和平均數，代表對這兩個指標的綜合考量。在本研究中，我們計算如下：
F1 = 2 × (精準率 × 召回率) / (精準率 + 召回率)。(5.6)

* **Balanced accuracy (Balanced ACC)**
Balanced accuracy considers both the correct identification of positive and negative samples, requiring the model to be as accurate as possible in its judgments of both. Sacrificing accuracy for one side at the expense of the other will decrease balanced accuracy. The equation is as follows:
Balanced accuracy = (sensitivity + specificity)/2 (5.7)
where sensitivity represents the proportion of samples that are actually positive and are judged as positive, same as Recall. Specificity represents the proportion of samples that are actually negative and are judged as negative.

* **平衡準確率 (Balanced ACC)**
平衡準確率同時考慮了對正樣本和負樣本的正確識別，要求模型在對兩者的判斷中盡可能準確。犧牲一方的準確性以換取另一方的準確性會降低平衡準確率。方程式如下：
平衡準確率 = (敏感度 + 特異度) / 2 (5.7)
其中敏感度代表實際為正且被判斷為正的樣本比例，與召回率相同。特異度代表實際為負且被判斷為負的樣本比例。

## 5.4 Baselines

We compare MODEL with the following methods, including semi-inductive link prediction and temporal knowledge graph reasoning methods.

## 5.4 基線模型

我們將 MODEL 與以下方法進行比較，包括半歸納式連結預測和時間知識圖譜推理方法。

* **DEKG-ILP** [9] extends inductive link prediction to contain bridging links to apply to more situations. It uses semantic information learning by contrastive learning and topological information to predict both enclosing and bridging links.

* **DEKG-ILP** [9] 將歸納式連結預測擴展為包含橋接連結，以適用於更多情況。它利用對比學習所學到的語意資訊和拓撲資訊來預測封閉連結和橋接連結。

* **GSELI** [10] is an improved model based on DEKG-ILP [9]. It adds a more efficient subgraph extraction module and neighboring relational paths modeling from SNRI [12], increasing prediction accuracy.

* **GSELI** [10] 是基於 DEKG-ILP [9] 的改良模型。它增加了一個更有效率的子圖提取模組，以及來自 SNRI [12] 的相鄰關係路徑模型，從而提高了預測準確度。

* **RE-GCN** [14] tries to integrate static and dynamic features into representations for temporal graph reasoning. It learns entity and relation representations by capturing structural dependencies within a single timestamp and sequential patterns across timestamps, and incorporates the static properties of the graph to contain stable features. In this way, it achieves the best performance on several datasets.

* **RE-GCN** [14] 嘗試將靜態與動態特徵整合至時間圖譜推理的表徵中。它透過捕捉單一時間戳內的結構相依性及跨時間戳的序列模式來學習實體與關係的表徵，並結合圖譜的靜態屬性以包含穩定的特徵。透過此方式，它在數個資料集上達到了最佳效能。

* **CorDGT** [15] proposes a Transformer-based model with a novel method to extract proximity more efficiently to capture comprehensive features of graphs. It employs the Poisson point process assumption to estimate temporal features and encode them with spatial features to obtain high-order proximity, and uses the multi-head self-attention mechanism to enhance expressive power.

* **CorDGT** [15] 提出了一種基於 Transformer 的模型，採用一種新穎的方法來更有效地提取鄰近性，以捕捉圖的綜合特徵。它採用泊松點過程假設來估計時間特徵，並將其與空間特徵編碼以獲得高階鄰近性，並使用多頭自註意力機制來增強表達能力。

## 5.5 Results

This section presents the experimental results across three aspects to evaluate the performance of the proposed model, EvoGUT. First, we analyze ranking results under both multi-step and transductive settings to assess prediction accuracy. Second, we examine the classification performance to verify the model's ability to predict link existence. Finally, we conduct ablation studies to quantify the contributions of each component.

## 5.5 結果

本節從三個方面介紹了實驗結果，以評估所提模型 EvoGUT 的性能。首先，我們分析了多步和直推式設定下的排名結果，以評估預測準確性。其次，我們檢驗了分類性能，以驗證模型預測連結存在的能力。最後，我們進行了消融研究，以量化每個組件的貢獻。

### 5.5.1 Ranking

We evaluate our model and baselines under multi-step and transductive settings by ranking order metrics.

### 5.5.1 排名

我們透過排名順序指標，在多步和直推式設定下評估我們的模型和基線。

#### Multi-Step Temporal Link Prediction

Tables 5.5, 5.6, and 5.7 report the experimental results of our method and baseline models under the multi-step setting. On Wikidata, although DEKG-ILP [9] and GSELI [10] are designed to handle scenarios with emerging entities, their reliance on structural features limits their performance on the sparser dataset. By enhancing GSELI [10] with temporal feature learning and a new training framework, our model achieves excellent performance. On ICEWS14, DEKG-ILP [9] maintains a similar performance on average, while GSELI [10] exhibits a noticeable increase. While our model outperforms the baselines on average, its advantage is less evident compared with its performance on Wikidata, where a clearer performance margin is observed.

#### 多步時間連結預測

表 5.5、5.6 和 5.7 報告了我們的方法和基線模型在多步設定下的實驗結果。在 Wikidata 上，儘管 DEKG-ILP [9] 和 GSELI [10] 旨在處理具有新興實體的場景，但它們對結構特徵的依賴限制了它們在稀疏數據集上的性能。通過使用時間特徵學習和新的訓練框架增強 GSELI [10]，我們的模型取得了優異的性能。在 ICEWS14 上，DEKG-ILP [9] 的平均性能保持相似，而 GSELI [10] 則表現出顯著的提升。雖然我們的模型在平均水平上優於基線，但與在 Wikidata 上的性能相比，其優勢不太明顯，在 Wikidata 上觀察到更清晰的性能差距。

Table 5.5: The average results of multi-step temporal link prediction.

表格 5.5：多步時間連結預測的平均結果。

[Image]

Table 5.6: The results of multi-step temporal link prediction on Wikidata.

表格 5.6：在 Wikidata 上進行多步時間連結預測的結果。

[Image]

Table 5.7: The results of multi-step temporal link prediction on ICEWS14.

表格 5.7：ICEWS14 上多步時間連結預測的結果。

[Image]

We speculate that this behavior arises from the intrinsic characteristics of the ICEWS14 dataset. ICEWS14 spans a relatively shorter overall time period but contains a large number of timestamps, reflecting frequent and fine-grained temporal changes. Furthermore, its content is primarily composed of news events, whose influence often decays rapidly over time. Therefore, temporal information and the duration of influence play a less critical role in this setting. Additionally, the relatively dense graph structure encourages the model to rely more heavily on structural and semantic features for prediction, rather than temporal features.

我們推測這種行為源於 ICEWS14 資料集的內在特性。ICEWS14 涵蓋的總體時間段相對較短，但包含大量的時間戳，反映了頻繁且細粒度的時間變化。此外，其內容主要由新聞事件組成，其影響力通常會隨時間迅速衰減。因此，時間資訊和影響持續時間在此設定中扮演的角色較不關鍵。此外，相對密集的圖結構促使模型更依賴結構和語意特徵進行預測，而非時間特徵。

In contrast, Wikidata exhibits coarser timestamp granularity and a sparser graph structure, resulting in suboptimal structural representations. In this case, temporal features can effectively compensate for the limitations of structural information and thereby improve overall performance. However, because our model adopts a pruning-based updating strategy, less influential triplets are discarded during graph updates. On ICEWS14, where structural information is particularly important, such pruning probably leads to structural information loss, resulting in inferior performance compared to Wikidata.

相較之下，Wikidata 表現出較粗的時間戳粒度和較稀疏的圖結構，導致次優的結構表示。在這種情況下，時間特徵可以有效地彌補結構資訊的限制，從而提高整體性能。然而，由於我們的模型採用基於修剪的更新策略，在圖更新過程中會丟棄影響力較小的三元組。在 ICEWS14 中，結構資訊尤為重要，這種修剪可能導致結構資訊的損失，從而導致與 Wikidata 相比性能較差。

For relation prediction shown in Tables 5.6 and 5.7, our model achieves suboptimal performance on Wikidata, but outperforms other methods on ICEWS14. One possible explanation is that relations are generally less sensitive to temporal information than entities. Even when the influence of certain information diminishes over time, it can still provide valuable structural cues for relation prediction. By removing such information, our model slightly decreases performance on Wikidata. In contrast, ICEWS14 is characterized by highly transient events where historical data can quickly become obsolete and accumulate as noise. In this scenario, appropriate information pruning helps reduce noise and leads to improved relation prediction performance.

對於表 5.6 和 5.7 中所示的關係預測，我們的模型在 Wikidata 上取得了次優的性能，但在 ICEWS14 上優於其他方法。一個可能的解釋是，關係通常對時間資訊的敏感度低於實體。即使某些資訊的影響力隨著時間的推移而減弱，它仍然可以為關係預測提供有價值的結構性線索。通過移除這些資訊，我們的模型在 Wikidata 上的性能略有下降。相比之下，ICEWS14 的特點是高度短暫的事件，歷史數據會迅速過時並積累為噪聲。在這種情況下，適當的資訊修剪有助於減少噪聲並提高關係預測性能。

#### Transductive Link Prediction

Tables 5.8 and 5.9 report the performance of our model and baselines on the two datasets. In the transductive setting, all entities and relations are known, which corresponds to a scenario favored by most temporal graph reasoning methods. Due to the reduced uncertainty, models typically achieve better performance in this setting than in the multi-step setting.

#### 直推式連結預測

表 5.8 和 5.9 報告了我們模型和基線在兩個數據集上的性能。在直推式設定中，所有實體和關係都是已知的，這對應於大多數時間圖推理方法所偏好的場景。由於不確定性降低，模型在此設定中通常比在多步設定中取得更好的性能。

Table 5.8: The results of transductive temporal link prediction on Wikidata.

表格 5.8：Wikidata 上直推式時間連結預測的結果。

[Image]

Table 5.9: The results of transductive temporal link prediction on ICEWS14.

表格 5.9：在 ICEWS14 上進行直推式時間連結預測的結果。

[Image]

On Wikidata, consistent with the observations in the multi-step setting, our model achieves excellent performance in entity prediction but performs relatively poorly in relation prediction. On ICEWS14, it lags behind other baselines. This is because our model focuses on the capability of handling dynamic, unknown structures and continuous prediction, strengthening its performance in the multi-step setting, thus sacrificing its ability in static graphs, which emphasize structural features. However, on Wikidata, it still maintains better performance in entity prediction by utilizing temporal features to compensate for the lack of structural features.

在 Wikidata 上，與多步設定中的觀察結果一致，我們的模型在實體預測方面取得了優異的性能，但在關係預測方面表現相對較差。在 ICEWS14 上，它落後於其他基線。這是因為我們的模型專注於處理動態、未知結構和連續預測的能力，從而增強了其在多步設定中的性能，但犧牲了其在強調結構特徵的靜態圖中的能力。然而，在 Wikidata 上，它仍然通過利用時間特徵來彌補結構特徵的不足，從而在實體預測中保持了較好的性能。

Both RE-GCN [14] and CorDGT [15], originally designed for temporal graphs, perform relatively well. However, RE-GCN [14] perform poorly in head entity prediction on Wikidata as reported in Table 5.8. This is presumably due to the inverse relations used during learning are not derived from real datasets. Since Wikidata is sparse and RE-GCN [14] does not leverage a global static graph on this dataset, it lacks aggregateble information, thus offering little help for head entity prediction. In contrast, on ICEWS14, by including a global static graph as a reference, even if the inverse relations do not actually exist, there is sufficient information to build a good representation of them. Similarly, CorDGT [15] also relies on a global static graph, thus performing slightly worse on Wikidata, the sparser dataset.

RE-GCN [14] 和 CorDGT [15] 最初都是為時間圖設計的，表現相對較好。然而，如表 5.8 所示，RE-GCN [14] 在 Wikidata 的頭實體預測上表現不佳。這大概是因為學習期間使用的逆關係並非源自真實數據集。由於 Wikidata 是稀疏的，且 RE-GCN [14] 未利用此數據集上的全域靜態圖，因此缺乏可聚合的資訊，對頭實體預測的幫助甚微。相反，在 ICEWS14 上，通過將全域靜態圖作為參考，即使逆關係實際上並不存在，也有足夠的資訊來建立它們的良好表示。同樣地，CorDGT [15] 也依賴於全域靜態圖，因此在較稀疏的數據集 Wikidata 上表現稍差。

### 5.5.2 Binary Classification

Even though the model performs well on ranking metrics, demonstrating its ability to distinguish between positive and negative samples in a relative sense, this does not necessarily demonstrate strong performance in absolute, instance-level discrimination. Therefore, we further verify the performance of our model and baselines on the link classification task under the multi-step setting, as shown in Tables 5.10 and 5.11. The most commonly used classification metrics include accuracy, AUROC, and F1-score. However, under extreme class imbalance between positive and negative samples, accuracy and AUROC can be misleading due to the dominance of negative samples, while F1-score does not consider the accuracy of negative samples. Therefore, we additionally report AUPRC and balanced accuracy, which are more suitable for evaluating performance in highly imbalanced scenarios. AUPRC focuses on the ability to avoid misclassifying positive samples among a large number of negative samples, while balanced accuracy reflects the average ability to distinguish between positive and negative samples. These two metrics are therefore particularly suitable for evaluating graph completion performance. Our model achieves the best performance on both metrics for almost all tasks, demonstrating its strong potential as an effective graph completion tool.

### 5.5.2 二元分類

儘管模型在排名指標上表現良好，顯示其在相對意義上區分正負樣本的能力，但這並不一定證明其在絕對的、實例級別的區分上具有強大的性能。因此，我們進一步在多步設定下驗證了我們的模型和基線在連結分類任務上的性能，如表 5.10 和 5.11 所示。最常用的分類指標包括準確率、AUROC 和 F1 分數。然而，在正負樣本極度不平衡的情況下，由於負樣本的主導地位，準確率和 AUROC 可能會產生誤導，而 F1 分數則未考慮負樣本的準確性。因此，我們額外報告了 AUPRC 和平衡準確率，它們更適合評估高度不平衡場景下的性能。AUPRC 專注於避免在大量負樣本中誤分類正樣本的能力，而平衡準確率則反映了區分正負樣本的平均能力。因此，這兩個指標特別適合評估圖補全性能。我們的模型在幾乎所有任務上都在這兩個指標上取得了最佳性能，顯示其作為有效圖補全工具的強大潛力。

Table 5.10: The results of binary classification under multi-step setting on Wikidata.

表格 5.10：在 Wikidata 多步設定下二元分類的結果。

[Image]

Table 5.11: The results of binary classification under multi-step setting on ICEWS14.

表格 5.11：在 ICEWS14 多步設定下二元分類的結果。

[Image]

In the relation prediction on Wikidata, the slight decrease in AUPRC shows the effect of considering unseen relations. Nevertheless, our model still maintains good performance than GSELI [10] and DEKG-ILP [9] since the superiority of its ability to distinguish negative samples. In tail entity prediction, while the model's ability to identify positive samples was sufficient, its ability to identify negative samples showed a significant decline compared to the other two tasks, causing a decrease in balanced accuracy. When considered alongside the ranking metrics, high-ranking performances indicate that the model gives higher scores to positive samples. However, not all negative samples are given scores below the threshold. This phenomenon is likely due to the threshold-setting strategy. When adjusting the threshold, the model uses the scores of positive and negative samples at certain percentiles as a reference for determining the threshold. To avoid overfitting, we expect the model to filter out 75% of negative samples, leaving a 25% margin. Consequently, some difficult-to-distinguish negative samples are treated as positive samples, which explains the observed decline in negative-sample discrimination.

在 Wikidata 的關係預測中，AUPRC 的輕微下降顯示了考慮未見關係的影響。然而，由於我們的模型在區分陰性樣本方面的優越性，其性能仍優於 GSELI [10] 和 DEKG-ILP [9]。在尾實體預測中，雖然模型識別陽性樣本的能力足夠，但其識別陰性樣本的能力與其他兩項任務相比顯著下降，導致平衡準確率下降。當與排名指標一起考慮時，高排名性能表明模型給予陽性樣本更高的分數。然而，並非所有陰性樣本的分數都低於閾值。這種現象很可能是由於閾值設定策略所致。在調整閾值時，模型使用特定百分位數的正負樣本分數作為確定閾值的參考。為避免過度擬合，我們期望模型過濾掉 75% 的陰性樣本，留下 25% 的邊界。因此，一些難以區分的陰性樣本被視為陽性樣本，這解釋了觀察到的陰性樣本辨別力下降的原因。

### 5.5.3 Ablation Study

We conduct some ablation studies to verify each component's effect and the differences between different configurations under the multi-step setting, and show the results in Tables 5.12 and 5.13. All variants evaluated are listed as follows.

### 5.5.3 消融研究

我們進行了一些消融研究，以驗證在多步設定下每個組件的效果以及不同配置之間的差異，並將結果顯示在表 5.12 和 5.13 中。所有評估的變體如下所列。

* **-tf**: Remove the entire temporal feature.
* **-il**: Remove only the influence lifetime.
* **w/o GUT**: Train the model without graph-updating training.
* **Accumu**: Use accumulation as the graph updating strategy for both training and testing.
* **Gt**: Use the ground truth graph as the input graph for both training and testing iterations to verify the effect of graph-updating training under ideal input conditions.
* **Gt/Pruning**: Use the ground truth graph as the input graph for each training iteration, but use pruning as the graph updating strategy when testing. This configuration reflects practical settings where real-time graph updates are typically infeasible, resulting in the absence of newly updated information during graph reasoning.
* **EvoGUT**: Use entire temporal feature and train the model with graph-updating training. For the graph updating strategy, use pruning for both training and testing.

* **-tf**: 移除整個時間特徵。
* **-il**: 僅移除影響力生命週期。
* **w/o GUT**: 不進行圖更新訓練來訓練模型。
* **Accumu**: 在訓練和測試中均使用累積作為圖更新策略。
* **Gt**: 在訓練和測試迭代中均使用真實圖作為輸入圖，以在理想輸入條件下驗證圖更新訓練的效果。
* **Gt/Pruning**: 在每次訓練迭代中使用真實圖作為輸入圖，但在測試時使用修剪作為圖更新策略。此配置反映了實際設置，其中即時圖更新通常不可行，導致圖推理期間缺乏新更新的資訊。
* **EvoGUT**: 使用完整的時間特徵並透過圖更新訓練來訓練模型。對於圖更新策略，在訓練和測試中均使用修剪。

Table 5.12: The results of ablation studies under multi-step temporal link prediction on Wikidata.

表格 5.12：在 Wikidata 上進行多步時間連結預測的消融研究結果。

[Image]

Table 5.13: The results of ablation studies under multi-step temporal link prediction on ICEWS14.

表格 5.13：ICEWS14 上多步時間連結預測消融研究的結果。

[Image]

In entity prediction, tail entity prediction relies more heavily on graph structural information than head entity prediction. Since a head entity has a high probability of establishing the same relation with several tail entities, referencing whether the head entity has similar connections with other entities provides a more stable predictive basis than temporal information. In other words, head entities are difficult to predict solely based on structural information. For example, many countries have hosted the Olympics, and the year is crucial in identifying which country hosted the Games. However, in ICEWS14, due to its frequent changes and short timestamp intervals, the discrimination of temporal features is weakened, resulting in head entity prediction performing better without relying on temporal information, as shown by the -tf and -il variants outperforming the full model in Table 5.13, together with consistent trends observed across the Accumu, the Accumu-tf, and Accumu-il variants.

在實體預測中，尾部實體預測比頭部實體預測更依賴圖的結構資訊。由於一個頭部實體很有可能與多個尾部實體建立相同的關係，因此參考該頭部實體是否與其他實體有相似的連結，會比時間資訊提供更穩定的預測基礎。換句話說，僅根據結構資訊很難預測頭部實體。例如，許多國家都曾舉辦過奧運會，而年份是確定哪個國家舉辦奧運會的關鍵。然而，在 ICEWS14 中，由於其頻繁的變化和短暫的時間戳間隔，時間特徵的辨別力被削弱，導致在不依賴時間資訊的情況下，頭部實體預測表現更佳，如表 5.13 中 -tf 和 -il 變體優於完整模型所示，以及在 Accumu、Accumu-tf 和 Accumu-il 變體中觀察到的一致趨勢。

Regarding the choice of graph updating strategy, the two datasets show inconsistent trends. Overall, using the accumulation strategy helps improve the overall model performance, as shown in the Accumu variant of both tables. Under this strategy, information is not removed from the graph, and even weakly influence information is continuously accumulated, thereby strengthening the quality of structural features. This effect is particularly evident on ICEWS14, which relies more heavily on structural features. However, when considering temporal features, entity prediction in Wikidata and relation prediction in ICEWS14 exhibit different trends.

關於圖更新策略的選擇，兩個資料集呈現出不一致的趨勢。總體而言，使用累積策略有助於提升整體模型性能，如兩個表格中的 Accumu 變體所示。在此策略下，資訊不會從圖中移除，即使是影響力微弱的資訊也會被持續累積，從而強化了結構特徵的品質。這種效果在更依賴結構特徵的 ICEWS14 上尤其明顯。然而，當考慮時間特徵時，Wikidata 中的實體預測和 ICEWS14 中的關係預測則呈現出不同的趨勢。

For Wikidata, using the pruning strategy, which is implemented in EvoGUT, to remove triplets whose influence has decayed can improve the accuracy of entity prediction, as shown in Table 5.12. This is because a large portion of Wikidata facts is highly time-sensitive; for example, heads of state change according to their terms, and many international events are held at different locations over time. Therefore, appropriately removing outdated information allows the model to focus more effectively on currently influential information, thereby improving prediction accuracy. However, this trend does not extend to relation prediction, as relations are generally less sensitive to temporal variation than entities, which is reflected in the performance gap between EvoGUT and Accumu variant in Table 5.12. In contrast, ICEWS14 places greater emphasis on structural features, and the accumulation strategy therefore tends to perform better, as reported in Table 5.13 where the Accumu variant excels in entity predictions over EvoGUT. However, for relation prediction, the benefits of temporal information and the influence duration are limited. Continuously, accumulating such information generally introduce noise, which can negatively affect prediction accuracy, as shown in the comparison between EvoGUT and the Accumu variant in Table 5.13.

對於 Wikidata，使用 EvoGUT 中實現的修剪策略來移除影響力已衰退的三元組，可以提高實體預測的準確性，如表 5.12 所示。這是因為 Wikidata 的一大部分事實具有高度的時間敏感性；例如，國家元首根據其任期而更換，許多國際事件在不同地點隨時間舉行。因此，適當地移除過時資訊可以讓模型更有效地專注於當前有影響力的資訊，從而提高預測準確性。然而，這種趨勢並未延伸到關係預測，因為關係通常對時間變化的敏感度低於實體，這反映在表 5.12 中 EvoGUT 和 Accumu 變體之間的性能差距上。相比之下，ICEWS14 更強調結構特徵，因此累積策略往往表現更好，如表 5.13 所示，其中 Accumu 變體在實體預測上優於 EvoGUT。然而，對於關係預測，時間資訊和影響持續時間的好處是有限的。持續累積此類資訊通常會引入噪聲，這可能會對預測準確性產生負面影響，如表 5.13 中 EvoGUT 和 Accumu 變體的比較所示。

# Chapter 6 Conclusions

For the ever-evolving temporal knowledge graph, we proposed a framework for continuous prediction by training the model to update the graph step-by-step. This framework improved model robustness by simulating graph updates during training. In addition to structural and semantic features, we incorporated the temporal information and estimated influence lifetime of information as features, successfully compensating for the lack of structural features on the sparse dataset, Wikidata. Ultimately, the model learned to determine the existence of triplets through various features and can adopt different graph updating strategies for different tasks and contexts, achieving excellent performance in the multi-step setting. It demonstrated the ability to handle dynamic or unknown graph structures, as well as emerging entities and relations. Furthermore, our model showed considerable potential in graph completion. Compared to other baselines, our model can more accurately distinguish the existence of positive and negative samples in most situations, demonstrating the advantages of graph updating training. It can also serve as a graph completion tool, reducing the need for human efforts and costs.

# 第六章 結論

針對不斷演化的時間知識圖譜，我們提出了一個連續預測的框架，透過訓練模型逐步更新圖譜。這個框架透過在訓練期間模擬圖譜更新，提高了模型的穩健性。除了結構和語意特徵外，我們還將時間資訊和估計的資訊影響力生命週期作為特徵納入，成功地彌補了稀疏資料集 Wikidata 上結構特徵的不足。最終，模型學會了透過各種特徵來判斷三元組的存在，並能針對不同的任務和情境採用不同的圖譜更新策略，在多步設定下取得了優異的性能。它展現了處理動態或未知圖譜結構，以及新興實體和關係的能力。此外，我們的模型在圖譜補全方面也顯示出巨大的潛力。與其他基線模型相比，我們的模型在大多數情況下能更準確地區分正負樣本的存在，展現了圖譜更新訓練的優勢。它還可以作為一個圖譜補全工具，減少了人力投入和成本。

# Chapter 7 Future Works

In the future, there are several promising directions for further development. First, regarding graph-updating strategies, the current choice of strategy relies largely on empirical rules. We aim to implement reinforcement learning or other adaptive mechanisms to dynamically determine which links should be retained during the evolutionary process, rather than a fixed criterion.

# 第七章 未來工作

未來，有幾個具前景的發展方向。首先，關於圖更新策略，目前的策略選擇主要依賴經驗法則。我們的目標是實現強化學習或其他自適應機制，以在演化過程中動態地決定應保留哪些連結，而非固定的標準。

Second, to achieve higher precision in handling emerging relations, additional designs or tasks are required to mitigate the relation cold-start problem. For instance, employing line graphs could transform emerging relations into emerging entities, thereby enabling the model to learn various relational features more effectively.

其次，為了在處理新興關係方面達到更高的精度，需要額外的設計或任務來緩解關係冷啟動問題。例如，採用線圖可以將新興關係轉換為新興實體，從而使模型能夠更有效地學習各種關係特徵。

Lastly, research involving knowledge graphs frequently encounters high time complexity in both training and inference stages due to the immense scale of the graph. To address this, future work will explore lightweight architectures to minimize processing latency. Enhancing the efficiency of training and inference will significantly improve the model's utility as a practical tool for knowledge graph completion.

最後，由於知識圖譜的龐大規模，涉及知識圖譜的研究在訓練和推論階段經常會遇到高時間複雜度的問題。為了解決這個問題，未來的工作將探索輕量級架構以最小化處理延遲。提高訓練和推論的效率將顯著提升模型作為知識圖譜補全實用工具的效用。

# References

[1] M. Zhang and Y. Chen, “Link prediction based on graph neural networks,” in Proc. NeurIPS18, Dec. 2018.
[2] Y. Peng and J. Zhang, “LineaRE: Simple but powerful knowledge graph embedding for link prediction," in Proc. IEEE ICDM20, Nov. 2020.
[3] M. Schlichtkrull, T. N. Kipf, P. Bloem, R. van den Berg, I. Titov, and M. Welling, “Modeling relational data with graph convolutional networks,” in Proc. ESWC18, June 2018.
[4] K. Teru, E. Denis, and W. Hamilton, “Inductive relation prediction by subgraph reasoning,” in Proc. ICML20, July 2020.
[5] A. Mitra, P. Vijayan, S. R. Singh, D. Goswami, S. Parthasarathy, and B. Ravindran, "Revisiting link prediction on heterogeneous graphs with a multi-view perspective,” in Proc. IEEE ICDM22, Nov./Dec. 2022.
[6] B. Ruan, C. Zhu, and W. Zhu, “A link prediction model of dynamic heterogeneous networks based on transformer," in Proc. IEEE IJCNN22, July 2022.
[7] A. Sankar, Y. Wu, L. Gou, W. Zhang, and H. Yang, “DySAT: Deep neural representation learning on dynamic graphs via self-attention networks,” in Proc. ACM WSDM20, pp. 519–527, Jan. 2020.
[8] D. Wang, Z. Zhang, Y. Ma, T. Zhao, T. Jiang, N. V. Chawla, and M. Jiang, "Modeling co-evolution of attributed and structural information in graph sequence,” IEEE Transactions on Knowledge and Data Engineering, vol. 35, no. 2, pp. 1817-1830, 2023.
[9] Y. Zhang, W. Wang, H. Yin, P. Zhao, W. Chen, and L. Zhao, “Disconnected emerging knowledge graph oriented inductive link prediction," in Proc. IEEE ICDE23, pp. 381–393, Apr. 2023.
[10] X. Liang, G. Si, J. Li, Z. An, P. Tian, F. Zhou, and X. Wang, “Integrating global semantics and enhanced local subgraph for inductive link prediction,” International Journal of Machine Learning and Cybernetics, vol. 16, no. 3, pp. 1971–1990, 2024.
[11] S. Zheng, S. Mai, Y. Sun, H. Hu, and Y. Yang, “Subgraph-aware few-shot inductive link prediction via meta-learning,” IEEE Transactions on Knowledge and Data Engineering, vol. 35, no. 6, pp. 6512-6517, 2022.
[12] X. Xu, P. Zhang, Y. He, C. Chao, and C. Yan, "Subgraph neighboring relations infomax for inductive link prediction on knowledge graphs,” in Proc. IJCAI22, pp. 2341–2347, July 2022.
[13] Y. Yang, J. Cao, M. Stojmenovic, S. Wang, Y. Cheng, C. Lum, and Z. Li, “Time-capturing dynamic graph embedding for temporal linkage evolution,” IEEE Transactions on Knowledge and Data Engineering, vol. 35, no. 1, pp. 958–971, 2023.
[14] Z. Li, X. Jin, W. Li, S. Guan, J. Guo, H. Shen, Y. Wang, and X. Cheng, “Temporal knowledge graph reasoning based on evolutional representation learning,” in Proc. ACM SIGIR21, pp. 408–417, July 2021.
[15] Z. Wang, S. Zhou, J. Chen, Z. Zhang, B. Hu, Y. Feng, C. Chen, and C. Wang, “Dynamic graph transformer with correlated spatial-temporal positional encoding,” in Proc. ACM WSDM25, pp. 60–69, Mar. 2025.
[16] B. Yang, W. tau Yih, X. He, J. Gao, and L. Deng, "Embedding entities and relations for learning and inference in knowledge bases." arXiv:1412.6575, Dec. 2014.
[17] C. Mavromatis, P. L. Subramanyam, V. N. Ioannidis, A. Adeshina, P. R. Howard, T. Grinberg, N. Hakim, and G. Karypis, “Tempoqr: Temporal question reasoning over knowledge graphs," in Proc. AAAI22, pp. 5825–5833, Feb./Mar. 2022.
[18] T. Lacroix, G. Obozinski, and N. Usunier, “Tensor decompositions for temporal knowledge base completion," in Proc. ICLR20, Apr./May 2020.
[19] Z. Han, P. Chen, Y. Ma, and V. Tresp, “Explainable subgraph reasoning for forecasting on temporal knowledge graphs," in Proc. ICLR21, May 2021.

# 參考文獻

[1] M. Zhang and Y. Chen, “Link prediction based on graph neural networks,” in Proc. NeurIPS18, Dec. 2018.
[2] Y. Peng and J. Zhang, “LineaRE: Simple but powerful knowledge graph embedding for link prediction," in Proc. IEEE ICDM20, Nov. 2020.
[3] M. Schlichtkrull, T. N. Kipf, P. Bloem, R. van den Berg, I. Titov, and M. Welling, “Modeling relational data with graph convolutional networks,” in Proc. ESWC18, June 2018.
[4] K. Teru, E. Denis, and W. Hamilton, “Inductive relation prediction by subgraph reasoning,” in Proc. ICML20, July 2020.
[5] A. Mitra, P. Vijayan, S. R. Singh, D. Goswami, S. Parthasarathy, and B. Ravindran, "Revisiting link prediction on heterogeneous graphs with a multi-view perspective,” in Proc. IEEE ICDM22, Nov./Dec. 2022.
[6] B. Ruan, C. Zhu, and W. Zhu, “A link prediction model of dynamic heterogeneous networks based on transformer," in Proc. IEEE IJCNN22, July 2022.
[7] A. Sankar, Y. Wu, L. Gou, W. Zhang, and H. Yang, “DySAT: Deep neural representation learning on dynamic graphs via self-attention networks,” in Proc. ACM WSDM20, pp. 519–527, Jan. 2020.
[8] D. Wang, Z. Zhang, Y. Ma, T. Zhao, T. Jiang, N. V. Chawla, and M. Jiang, "Modeling co-evolution of attributed and structural information in graph sequence,” IEEE Transactions on Knowledge and Data Engineering, vol. 35, no. 2, pp. 1817-1830, 2023.
[9] Y. Zhang, W. Wang, H. Yin, P. Zhao, W. Chen, and L. Zhao, “Disconnected emerging knowledge graph oriented inductive link prediction," in Proc. IEEE ICDE23, pp. 381–393, Apr. 2023.
[10] X. Liang, G. Si, J. Li, Z. An, P. Tian, F. Zhou, and X. Wang, “Integrating global semantics and enhanced local subgraph for inductive link prediction,” International Journal of Machine Learning and Cybernetics, vol. 16, no. 3, pp. 1971–1990, 2024.
[11] S. Zheng, S. Mai, Y. Sun, H. Hu, and Y. Yang, “Subgraph-aware few-shot inductive link prediction via meta-learning,” IEEE Transactions on Knowledge and Data Engineering, vol. 35, no. 6, pp. 6512-6517, 2022.
[12] X. Xu, P. Zhang, Y. He, C. Chao, and C. Yan, "Subgraph neighboring relations infomax for inductive link prediction on knowledge graphs,” in Proc. IJCAI22, pp. 2341–2347, July 2022.
[13] Y. Yang, J. Cao, M. Stojmenovic, S. Wang, Y. Cheng, C. Lum, and Z. Li, “Time-capturing dynamic graph embedding for temporal linkage evolution,” IEEE Transactions on Knowledge and Data Engineering, vol. 35, no. 1, pp. 958–971, 2023.
[14] Z. Li, X. Jin, W. Li, S. Guan, J. Guo, H. Shen, Y. Wang, and X. Cheng, “Temporal knowledge graph reasoning based on evolutional representation learning,” in Proc. ACM SIGIR21, pp. 408–417, July 2021.
[15] Z. Wang, S. Zhou, J. Chen, Z. Zhang, B. Hu, Y. Feng, C. Chen, and C. Wang, “Dynamic graph transformer with correlated spatial-temporal positional encoding,” in Proc. ACM WSDM25, pp. 60–69, Mar. 2025.
[16] B. Yang, W. tau Yih, X. He, J. Gao, and L. Deng, "Embedding entities and relations for learning and inference in knowledge bases." arXiv:1412.6575, Dec. 2014.
[17] C. Mavromatis, P. L. Subramanyam, V. N. Ioannidis, A. Adeshina, P. R. Howard, T. Grinberg, N. Hakim, and G. Karypis, “Tempoqr: Temporal question reasoning over knowledge graphs," in Proc. AAAI22, pp. 5825–5833, Feb./Mar. 2022.
[18] T. Lacroix, G. Obozinski, and N. Usunier, “Tensor decompositions for temporal knowledge base completion," in Proc. ICLR20, Apr./May 2020.
[19] Z. Han, P. Chen, Y. Ma, and V. Tresp, “Explainable subgraph reasoning for forecasting on temporal knowledge graphs," in Proc. ICLR21, May 2021.
