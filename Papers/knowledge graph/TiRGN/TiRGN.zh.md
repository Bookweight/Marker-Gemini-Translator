---
title: TiRGN
field: Knowledge_Graph
status: Imported
created_date: 2026-01-14
pdf_link: "[[TiRGN.pdf]]"
tags:
  - paper
  - knowledge_graph
---

Proceedings of the Thirty-First International Joint Conference on Artificial Intelligence (IJCAI-22)

第三十一屆國際人工智慧聯合會議（IJCAI-22）議事錄

***

TiRGN: Time-Guided Recurrent Graph Network with Local-Global Historical Patterns for Temporal Knowledge Graph Reasoning

TiRGN：一個用於時序知識圖譜推理，具有局部-全域歷史模式的時間引導循環圖神經網路

***

Yujia Li, Shiliang Sun and Jing Zhao*
School of Computer Science of Technology, East China Normal University, Shanghai 200062, China
jzhao@cs.ecnu.edu.cn

李宇佳，孫仕亮，趙靜*
華東師範大學計算機科學與技術學院，上海 200062，中國
jzhao@cs.ecnu.edu.cn

***

### Abstract

### 摘要

Temporal knowledge graphs (TKGs) have been widely used in various fields that model the dynamics of facts along the timeline. In the extrapolation setting of TKG reasoning, since facts happening in the future are entirely unknowable, insight into history is the key to predicting future facts. However, it is still a great challenge for existing models as they hardly learn the characteristics of historical events adequately. From the perspective of historical development laws, comprehensively considering the sequential, repetitive, and cyclical patterns of historical facts is conducive to predicting future facts. To this end, we propose a novel representation learning model for TKG reasoning, namely TiRGN, a time-guided recurrent graph network with local-global historical patterns. Specifically, TiRGN uses a local recurrent graph encoder network to model the historical dependency of events at adjacent timestamps and uses the global history encoder network to collect repeated historical facts. After the trade-off between the two encoders, the final inference is performed by a decoder with periodicity. We use six benchmark datasets to evaluate the proposed method. The experimental results show that TiRGN outperforms the state-of-the-art TKG reasoning methods in most cases.

時序知識圖譜（TKGs）已被廣泛應用於各種領域，以模擬事實隨時間變化的動態。在 TKG 推理的外推設定中，由於未來發生的事實是完全不可知的，對歷史的洞察是預測未來事實的關鍵。然而，對於現有模型而言，這仍然是一個巨大的挑戰，因為它們很難充分學習歷史事件的特徵。從歷史發展規律的角度來看，綜合考慮歷史事實的順序性、重複性和週期性模式，有助於預測未來的事實。為此，我們提出了一種用於 TKG 推理的新穎表示學習模型，名為 TiRGN，一個具有局部-全域歷史模式的時間引導循環圖神經網路。具體來說，TiRGN 使用局部循環圖編碼器網路來模擬相鄰時間戳上事件的歷史依賴性，並使用全域歷史編碼器網路來收集重複的歷史事實。在兩個編碼器之間進行權衡之後，最終的推理由一個具有週期性的解碼器執行。我們使用六個基準數據集來評估所提出的方法。實驗結果表明，在大多數情況下，TiRGN 的性能優於最先進的 TKG 推理方法。

***

### 1 Introduction

### 1 緒論

Knowledge graphs (KGs) have been widely used in intelligent applications, such as question answering, recommendation systems, and information retrieval. However, the incompleteness of KGs limits their performance in downstream tasks. In this regard, there have been many studies focusing on the reasoning and completion of static KGs. In reality, many data often have complex dynamics, and it is a challenge to characterize the dynamics and reason over temporal knowledge graphs (TKGs).

知識圖譜（KGs）已廣泛應用於問答、推薦系統和資訊檢索等智慧應用中。然而，KGs 的不完整性限制了它們在下游任務中的性能。在這方面，已有許多研究專注於靜態 KGs 的推理與補全。在現實中，許多數據通常具有複雜的動態性，而描述動態並在時序知識圖譜（TKGs）上進行推理是一項挑戰。

***

Each fact in TKGs is represented by a quadruple containing a timestamp, which indicates that the fact occurred at a specific time. For example, (Barack Obama, impose sanctions, Sudan, 2013-11-7) indicates the fact that Obama imposed sanctions on Sudan in 2013-11-7. In this paper, we mainly solve the extrapolation problem of TKG reasoning, including entity prediction and relation prediction. This is significant for many practical applications, such as event process induction, social relation prediction, disaster relief, and financial analysis.

TKGs 中的每個事實都由一個包含時間戳的四元組表示，該時間戳表示該事實發生的特定時間。例如，(Barack Obama, impose sanctions, Sudan, 2013-11-7) 表示歐巴馬於 2013 年 11 月 7 日對蘇丹實施制裁。在本文中，我們主要解決 TKG 推理的外推問題，包括實體預測和關係預測。這對於許多實際應用具有重要意義，例如事件過程歸納、社會關係預測、災害救援和金融分析。

***

Making accurate predictions about future facts needs to learn more about historical facts based on the laws of historical development. According to historic recurrence [Trompf, 1979] and social cycle theory [Schlesinger, 1999], historical facts may have a repetitive or even cyclical pattern. Besides, according to human cognition, the historical facts at adjacent timestamps have a sequential pattern, in which the evolution law of adjacent events can be captured to predict what will happen next. For example, as shown in Figure 1, we predict the query (Barack Obama, impose sanctions, ?, 2013-11-7). We can get a series of historical adjacent facts and global facts related to the query. The countries in the relevant local facts include Iran which has recently been sanctioned, and Sudan which has recently criticized Obama. In the relevant global facts, Obama imposed sanctions on eight countries, including sanctions on Iran 46 times and Sudan eight times. If both local and global facts are considered, we can narrow the scope of the prediction results of the example query to Iran and Sudan. In addition, we find that Obama imposed sanctions on Sudan around November annually from 2009 to 2012. Considering periodicity and the other two historical patterns simultaneously, it is more likely that the predicted result will be Sudan, which is indeed the case. Therefore, capturing different historical patterns can constrain the range of the final prediction results and improve accuracy.

要準確預測未來的事實，需要根據歷史發展的規律，更多地了解歷史事實。根據歷史重現理論 [Trompf, 1979] 和社會週期理論 [Schlesinger, 1999]，歷史事實可能具有重複甚至週期性的模式。此外，根據人類認知，相鄰時間戳的歷史事實具有順序模式，可以捕捉相鄰事件的演化規律來預測接下來會發生什麼。例如，如圖 1 所示，我們預測查詢 (Barack Obama, impose sanctions, ?, 2013-11-7)。我們可以得到一系列與該查詢相關的歷史相鄰事實和全域事實。相關局部事實中的國家包括最近受到制裁的伊朗，以及最近批評歐巴馬的蘇丹。在相關的全域事實中，歐巴馬對八個國家實施了制裁，其中對伊朗的制裁有 46 次，對蘇丹的制裁有 8 次。如果同時考慮局部和全域事實，我們可以將範例查詢的預測結果範圍縮小到伊朗和蘇丹。此外，我們發現歐巴馬在 2009 年至 2012 年期間每年大約在 11 月對蘇丹實施制裁。同時考慮週期性和其他兩種歷史模式，預測結果更有可能是蘇丹，事實也確實如此。因此，捕捉不同的歷史模式可以約束最終預測結果的範圍並提高準確性。

***

Recently, some methods have tried to extract relevant historical information for different queries in a heuristic way. However, they did not comprehensively consider the different historical characteristics. RE-Net [Jin et al., 2020] and CyGNet [Zhu et al., 2021] only consider the entities or entity-relation pairs in the global history. Among them, RE-Net can only capture global facts within a limited time. Otherwise, it will cause huge time complexity. CyGNet describes global facts by occurrence frequency, which leads to the narrow results that the prediction always tends to the facts with the highest frequency. RE-GCN [Li et al., 2021b] attempts to model local historical dependency but lacks the capture of global historical information. Furthermore, although these methods extract historical information, none of them models the periodicity of historical facts.

最近，一些方法試圖以啟發式的方式為不同的查詢提取相關的歷史資訊。然而，它們沒有全面考慮不同的歷史特徵。RE-Net [Jin et al., 2020] 和 CyGNet [Zhu et al., 2021] 只考慮了全域歷史中的實體或實體-關係對。其中，RE-Net 只能在有限的時間內捕捉全域事實，否則會導致巨大的時間複雜度。CyGNet 通過出現頻率來描述全域事實，這導致預測結果總是傾向於具有最高頻率的事實，結果變得狹隘。RE-GCN [Li et al., 2021b] 試圖對局部歷史依賴性進行建模，但缺乏對全域歷史資訊的捕捉。此外，雖然這些方法提取了歷史資訊，但沒有一個模型對歷史事實的週期性進行建模。

***

To this end, we propose a model that captures multiple historical characteristics with local-global historical information, called Time-Guided Recurrent Graph Network (TiRGN). The main ideas of TiRGN are (1) to combine the local and global historical information to capture sequential, repetitive, and cyclical patterns of historical facts, (2) to regard the temporal subgraphs at adjacent timestamps as a sequence and regard the subgraphs at global timestamps as a constraint, and (3) to design a time-guided periodic decoder by using model-independent time vectors. Specifically, in order to capture the sequential pattern of historical facts, we design a graph neural network-based encoder with a double recurrent mechanism to simultaneously evolve the entity and relation representations at adjacent timestamps. In addition, we use the global one-hop or multi-hop repeated history information to capture repetitive patterns of historical facts. Furthermore, the time vectors involved in the model can capture the periodicity of facts. Finally, the scores calculated by the decoder are used to realize entity and relation predictions.

為此，我們提出了一個名為「時間引導循環圖網路」（TiRGN）的模型，該模型能夠利用局部-全域歷史資訊捕捉多種歷史特徵。TiRGN 的主要思想是：（1）結合局部和全域歷史資訊，以捕捉歷史事實的順序、重複和週期性模式；（2）將相鄰時間戳的時序子圖視為序列，並將全域時間戳的子圖視為約束；（3）通過使用與模型無關的時間向量，設計一個時間引導的週期性解碼器。具體而言，為了捕捉歷史事實的順序模式，我們設計了一個基於圖神經網路的編碼器，該編碼器具有雙重循環機制，可同時演化相鄰時間戳的實體和關係表示。此外，我們利用全域單跳或多跳的重複歷史資訊來捕捉歷史事實的重複模式。再者，模型中涉及的時間向量能夠捕捉事實的週期性。最後，由解碼器計算的分數被用於實現實體和關係的預測。

***

In general, this work presents the following contributions:

總體而言，本研究提出以下貢獻：

*   We propose a representation learning model TiRGN for TKG reasoning, which simultaneously considers the sequential, repetitive and cyclical patterns of historical facts. As far as we know, this is the first time to integrate these historical characteristics for TKG reasoning.
*   我們提出了一個用於 TKG 推理的表示學習模型 TiRGN，該模型同時考慮了歷史事實的順序、重複和週期性模式。據我們所知，這是首次將這些歷史特徵整合到 TKG 推理中。

*   We design a double recurrent mechanism with long-distance dependencies to encode the adjacent subgraph sequences and use a low-complexity global history encoder to collect repeated facts. We use periodic time vectors to guide the decoder and realize the trade-off between local and global historical information.
*   我們設計了一種具有長距離依賴性的雙重循環機制來編碼相鄰的子圖序列，並使用一個低複雜度的全域歷史編碼器來收集重複的事實。我們使用週期性時間向量來引導解碼器，並實現局部和全域歷史資訊之間的權衡。

*   Experiments on six public TKG datasets demonstrate that TiRGN is consistently effective on both entity prediction and relation prediction.
*   在六個公開的 TKG 資料集上的實驗證明，TiRGN 在實體預測和關係預測方面均持續有效。

***

### 2 Related Work

### 2 相關研究

Recently, some studies have tried to incorporate temporal information into KG reasoning, which can be classified into two settings: interpolation and extrapolation [Jin et al., 2020]. Interpolation is used to predict missing historical facts. For this setting, several embedding-based methods associate time with facts and map them to a low-dimensional space [García-Durán et al., 2018; Ma et al., 2019; Wu et al., 2020]. TTransE [Jiang et al., 2016] is a variant of TransE [Bordes et al., 2013], which treats relation and time as translation between entities. DE-SimplE [Goel et al., 2020] and ChronoR [Sadeghian et al., 2021] characterize temporal information by learning the embeddings of different timestamps. However, these models cannot predict future facts.

近期，一些研究嘗試將時序資訊納入知識圖譜推理，可分為內插（interpolation）和外插（extrapolation）兩種設定 [Jin et al., 2020]。內插用於預測缺失的歷史事實。在此設定下，數種基於嵌入（embedding-based）的方法將時間與事實關聯，並將其對應至低維度空間 [García-Durán et al., 2018; Ma et al., 2019; Wu et al., 2020]。TTransE [Jiang et al., 2016] 是 TransE [Bordes et al., 2013] 的變體，將關係與時間視為實體間的平移。DE-SimplE [Goel et al., 2020] 和 ChronoR [Sadeghian et al., 2021] 透過學習不同時間戳的嵌入來描述時序資訊。然而，這些模型無法預測未來的事實。

***

This paper focuses on the extrapolation setting of predicting future facts according to history, which has been studied recently. Know-Evolve [Trivedi et al., 2017] and DyREP [Trivedi et al., 2019] use the temporal point process to model the occurrence of facts in TKGs. Glean [Deng et al., 2020] incorporates relational and word contexts to enrich the features of facts for reasoning. RE-NET [Jin et al., 2020] employs a neighborhood aggregator and recurrent event encoder to model the historical facts as subgraph sequences. TANGO [Han et al., 2021b] explores the neural ordinary differential equation to build a continuous-time reasoning model. XERTE [Han et al., 2021a] uses a subgraph sampling technique to construct interpretable reasoning graphs. CluSTeR [Li et al., 2021a] and TITer [Sun et al., 2021] both use reinforcement learning to search a series of historical facts for reasoning. CyGNet [Zhu et al., 2021] and RE-GCN [Li et al., 2021b] are the most relevant work to us. CyGNet uses a copy-generation mechanism to capture the global repetition frequency of facts. RE-GCN learns the evolutional representations of entities and relations at each timestamp through the structures that capture local historical dependency. However, none of the above methods simultaneously consider the sequential, repetitive, and cyclical historical facts.

本文專注於根據歷史預測未來事實的外推設定，此議題近期已受到研究。Know-Evolve [Trivedi et al., 2017] 與 DyREP [Trivedi et al., 2019] 使用時序點過程來模擬 TKG 中事實的發生。Glean [Deng et al., 2020] 結合關係與詞彙脈絡來豐富事實的推理特徵。RE-NET [Jin et al., 2020] 採用鄰域聚合器與循環事件編碼器，將歷史事實建模為子圖序列。TANGO [Han et al., 2021b] 探索神經常微分方程以建立連續時間推理模型。XERTE [Han et al., 2021a] 使用子圖採樣技術建構可解釋的推理圖。CluSTeR [Li et al., 2021a] 與 TITer [Sun et al., 2021] 皆使用強化學習來搜尋一系列歷史事實以進行推理。CyGNet [Zhu et al., 2021] 與 RE-GCN [Li et al., 2021b] 是與我們最相關的研究。CyGNet 使用複製生成機制來捕捉事實的全域重複頻率。RE-GCN 透過捕捉局部歷史依賴性的結構，學習每個時間戳下實體與關係的演化表示。然而，上述方法皆未同時考慮順序性、重複性與週期性的歷史事實。

***

### 3 The Proposed Method

### 3 提議的方法

#### 3.1 Notations

#### 3.1 符號說明

In this paper, we formalize a TKG G as a sequence of subgraphs, i.e., G = {G1, G2,...,GT}. The subgraph Gt = (E, R, Ft) at t is a directed multi-relational graph, where E is the set of entities, R is the set of relations, and Ft is the set of facts at t. A fact in Ft can be formalized as a quadruple (s, r, o, t), where s, o ∈ E and r ∈ R. It represents that there is a relation r between the subject entity s and the object entity o at t. For each quadruple (s, r, o, t), an inverse relation quadruple (o, r-¹,s,t) is often added to the dataset. TKG reasoning can be classified into entity prediction (s,r,?,t) and relation prediction (s, ?, o, t) given the set of historical facts before t. For each prediction at t, we formalize the subgraph sequence of its previous m timestamps as Gt-m:t-1·

在本文中，我們將一個 TKG G 形式化為一系列子圖，即 G = {G1, G2,...,GT}。在時間 t 的子圖 Gt = (E, R, Ft) 是一個有向多重關係圖，其中 E 是實體集合，R 是關係集合，而 Ft 是在時間 t 的事實集合。在 Ft 中的一個事實可以形式化為一個四元組 (s, r, o, t)，其中 s, o ∈ E 且 r ∈ R。它表示在時間 t，主體實體 s 與客體實體 o 之間存在一個關係 r。對於每個四元組 (s, r, o, t)，一個逆關係四元組 (o, r-¹,s,t) 通常會被加入到資料集中。TKG 推理可以根據 t 之前的歷史事實集合，分類為實體預測 (s,r,?,t) 和關係預測 (s, ?, o, t)。對於在 t 的每個預測，我們將其前 m 個時間戳的子圖序列形式化為 Gt-m:t-1。

***

#### 3.2 Model Overview

#### 3.2 模型概覽

The overall framework of TiRGN is shown in Figure 2. According to the laws of historical development, TiRGN consists of three components, which are used to capture the sequential, repetitive, and cyclical patterns of historical facts, respectively. Local recurrent encoder is used to explore structural features and historical dependency. If there are facts in the subgraphs of adjacent timestamps containing the same semantic information as the query, the probability of predicting the entity in these facts will increase. According to common sense, TiRGN assumes that the closer the time of the facts, the more significant the impact on the final results. Global history encoder takes the relevant facts at all previous timestamps into account to avoid missing entities or relations that have not appeared at the adjacent timestamps. Periodicity is introduced into the decoder by the time vector generator to find possible periodic facts. Finally, TiRGN combines the periodic local and global decoders to realize the trade-off between the importance of local and global historical facts.

TiRGN 的整體架構如圖 2 所示。根據歷史發展規律，TiRGN 由三個部分組成，分別用於捕捉歷史事實的順序性、重複性和週期性模式。局部循環編碼器用於探索結構特徵和歷史依賴性。如果相鄰時間戳的子圖中存在與查詢語義資訊相同的事實，則預測這些事實中實體的機率將會增加。根據常識，TiRGN 假設事實發生的時間越近，對最終結果的影響越顯著。全域歷史編碼器會考慮所有先前時間戳的相關事實，以避免遺漏在相鄰時間戳中未出現的實體或關係。週期性是通過時間向量生成器引入解碼器中，以尋找可能的週期性事實。最後，TiRGN 結合週期性的局部和全域解碼器，以實現局部和全域歷史事實重要性之間的權衡。

***

#### 3.3 Local Recurrent Encoder

#### 3.3 局部循環編碼器

The local recurrent encoder focuses on the adjacent histories. For each query (s,r,?,t), we consider the subgraphs Gt-m:t-1 of m adjacent timestamps. We aggregate and transfer the KG information from spatial and temporal views, respectively. Specifically, the graph convolutional network (GCN) is used to make single-step aggregation and the gated recurrent unit (GRU) is used among multiple timestamps to perform multi-step evolution.

局部循環編碼器專注於相鄰的歷史記錄。對於每個查詢 (s,r,?,t)，我們考慮 m 個相鄰時間戳的子圖 Gt-m:t-1。我們分別從空間和時間的角度聚合和傳遞知識圖譜資訊。具體來說，圖卷積網路 (GCN) 用於進行單步聚合，而門控循環單元 (GRU) 則在多個時間戳之間用於執行多步演化。

***

##### Single-Step Aggregation

##### 單步聚合

At each timestamp, we want to cover the facts related to the entity in the query as many as possible. Therefore, we design a one-dimensional convolution-based GCN, a multi-relation aggregator, to merge multiple relations and multi-hop neighbor information at a single timestamp. Compared with RE-GCN which sums relation embedding to entity embedding in GCN to make single-step aggregation, our TiRGN uses the one-dimensional convolution on entity embedding and relation embedding and thus can merge them better. The aggregator is defined as:
hl+1 = σ (Σ(s,r,o)∈Ft (1/c_o) * W_r(h_o^l, r_t) + W_0 h_s^l) (1)
where h, at, h hot denote the lth layer embeddings of entities s, o at t, W, W denote learnable weights and W is relation-specific, co is a normalizing factor equal to the in-degree of o, o is the RReLu activation function, and & is the one-dimensional convolution operator. Note that when an entity does not have any relation with other entities in the subgraph, there will still be a self-loop edge to update it.

在每個時間戳，我們希望盡可能涵蓋與查詢中實體相關的事實。因此，我們設計了一個基於一維卷積的 GCN，一個多關係聚合器，用於在單一時間戳內合併多個關係和多跳鄰居資訊。與在 GCN 中將關係嵌入加總至實體嵌入以進行單步聚合的 RE-GCN 相比，我們的 TiRGN 在實體嵌入和關係嵌入上使用一維卷積，因此能更好地合併它們。聚合器定義如下：
hl+1 = σ (Σ(s,r,o)∈Ft (1/c_o) * W_r(h_o^l, r_t) + W_0 h_s^l) (1)
其中 h, at, h hot 表示在 t 時實體 s, o 的第 l 層嵌入，W, W 表示可學習的權重，W 是關係特定的，co 是等於 o 的入度的歸一化因子，o 是 RReLu 激活函數，而 & 是一維卷積算子。請注意，當一個實體在子圖中與其他實體沒有任何關係時，仍會有一個自環邊來更新它。

***

##### Multi-Step Evolution

##### 多步演化

For each query, in order to include the sequential dependencies of subgraphs at the previous timestamps, we use a double recurrent mechanism to update the representations of entities and relations progressively, i.e., entity-oriented GRU and relation-oriented GRU, so that it can obtain information at more distant timestamps. Entity-oriented GRU is used to update embeddings of entities in the sequence of subgraphs:
Ht = GRU (Ht−1, HGCN) (2)
where Ht, Ht-1 ∈ R|E|×d are the d-dimensional entity embedding matrices at t and t 1, and HGCN ∈ R|E|×d is the entity embedding matrix after singe-step aggeration at t - 1. For relations, in order to maintain the consistency with the update of the entity embedding in the subgraph sequence, relation-oriented GRU is also used for update:
r' = [pooling(Ht-1, Hr,t); r] (3)
Rt = GRU(Rt-1, R') (4)
where Hr,t is all entities connected to r at t, r' is obtained by Ht-1 and Hr,t using mean pooling operation, R' consists of r' of all relations, and Rt, Rt-1 ∈ R|R|x d are relation embedding matrices at t and t-1. Rt is finally updated by Rt-1 and R' through relation-oriented GRU.

對於每個查詢，為了包含先前時間戳子圖的順序依賴性，我們使用雙重循環機制來逐步更新實體和關係的表示，即實體導向的 GRU 和關係導向的 GRU，以便它可以在更遠的時間戳獲取資訊。實體導向的 GRU 用於更新子圖序列中實體的嵌入：
Ht = GRU (Ht−1, HGCN) (2)
其中 Ht, Ht-1 ∈ R|E|×d 是在 t 和 t-1 時的 d 維實體嵌入矩陣，而 HGCN ∈ R|E|×d 是在 t-1 進行單步聚合後的實體嵌入矩陣。對於關係，為了維持與子圖序列中實體嵌入更新的一致性，也使用關係導向的 GRU 進行更新：
r' = [pooling(Ht-1, Hr,t); r] (3)
Rt = GRU(Rt-1, R') (4)
其中 Hr,t 是在 t 時連接到 r 的所有實體，r' 是通過 Ht-1 和 Hr,t 使用平均池化操作得到的，R' 由所有關係的 r' 組成，而 Rt, Rt-1 ∈ R|R|x d 是在 t 和 t-1 時的關係嵌入矩陣。Rt 最終由 Rt-1 和 R' 通過關係導向的 GRU 更新。

***

#### 3.4 Global History Encoder

#### 3.4 全域歷史編碼器

The global history encoder is designed to get the repetitive global candidate facts, so as to provide global constraints for scoring in the decoder. For each query (s,r,?,t) or (s,?, o, t), we use this encoder to obtain candidate one-hop or multi-hop entities and relations. It is worth noting that, unlike CyGNet, we only consider whether the entity or relation has appeared before, without considering the frequency of its occurrence. We believe that directly using frequency as features may mislead prediction as the fact happened long ago does not necessarily occur in the future. The local recurrent encoder have been able to capture the impact of frequency of recent facts. This module is more to narrow the scope of prediction and avoid omissions rather than directly determining the final result. Specifically, we traverse all the subgraphs Go:t-1 before t and get the query results {c_s,r^0, c_s,r^1, ..., c_s,r^{t-1}}, {c_s,o^0, c_s,o^1, ..., c_s,o^{t-1}}. Then we take the union of the set of candidate entities at timestamp t:
C_s,r^t = c_s,r^0 ∪ c_s,r^1 ∪ ... ∪ c_s,r^{t-1} (5)
Therefore, for the query (s, r, ?, t), the candidate entity matrix M^r ∈ Z|E|×|R|×|E| assigns the values of positions existent in C_s,r^t to 1 and the non-existent to 0. The same operations and settings are also used for the candidate relation set C_s,o^t and candidate relation matrix M^o ∈ Z|E|×|E|×|R|. Although these two matrices have large dimensions, they are both sparse (0, 1)-matrices, so they have a low space complexity and time complexity during access. In addition, the global history encoder can be used for different levels of candidate fact records, including one-hop and multi-hop. At present, we only consider the one-hop candidate set.

全域歷史編碼器旨在獲取重複的全域候選事實，以便為解碼器中的評分提供全域約束。對於每個查詢 (s,r,?,t) 或 (s,?, o, t)，我們使用此編碼器來獲取候選的單跳或多跳實體和關係。值得注意的是，與 CyGNet 不同，我們只考慮實體或關係是否曾經出現過，而不考慮其出現的頻率。我們認為，直接使用頻率作為特徵可能會誤導預測，因為很久以前發生的事實不一定會在未來發生。局部循環編碼器已經能夠捕捉到近期事實頻率的影響。此模塊更傾向於縮小預測範圍並避免遺漏，而不是直接確定最終結果。具體來說，我們遍歷 t 之前的所有子圖 Go:t-1，並得到查詢結果 {c_s,r^0, c_s,r^1, ..., c_s,r^{t-1}}, {c_s,o^0, c_s,o^1, ..., c_s,o^{t-1}}。然後，我們取時間戳 t 處候選實體集的並集：
C_s,r^t = c_s,r^0 ∪ c_s,r^1 ∪ ... ∪ c_s,r^{t-1} (5)
因此，對於查詢 (s, r, ?, t)，候選實體矩陣 M^r ∈ Z|E|×|R|×|E| 將 C_s,r^t 中存在的位置值賦為 1，不存在的賦為 0。相同的操作和設置也用於候選關係集 C_s,o^t 和候選關係矩陣 M^o ∈ Z|E|×|E|×|R|。儘管這兩個矩陣維度很大，但它們都是稀疏的 (0, 1)-矩陣，因此在存取期間具有較低的空間複雜度和時間複雜度。此外，全域歷史編碼器可用於不同級別的候選事實記錄，包括單跳和多跳。目前，我們僅考慮單跳候選集。

***

#### 3.5 Time Guided Decoder

#### 3.5 時間引導解碼器

After getting the embeddings of the local entities and relations, as well as the candidate sets of global entities and relations, we use the local and global decoders to score the facts. Periodic and non-periodic time vectors guide the decoders to incorporate the periodicity of facts into the model when calculating local and global scores.

在獲得局部實體和關係的嵌入，以及全域實體和關係的候選集後，我們使用局部和全域解碼器對事實進行評分。週期性和非週期性時間向量引導解碼器在計算局部和全域分數時，將事實的週期性納入模型。

***

##### Periodic and Non-Periodic Time Vectors

##### 週期性與非週期性時間向量

Some facts happen periodically throughout the timeline, such as presidential elections, while some facts are more likely to happen within a certain period of time, e.g., the elected president will participate in more activities in a certain period of time. Therefore, it makes sense to consider both the periodicity and non-periodicity of the historical facts. We design the periodic and non-periodic time vectors, respectively:
v_t^p = f(ω_p t + φ_p) (6)
v_t^{np} = ω_{np} t + φ_{np} (7)
where v_t^p and v_t^{np} are d-dimensional periodic and non-periodic time vectors, respectively, ω_p, φ_p, ω_{np} and φ_{np} are learnable parameters, and f is a periodic activation function. We chose the sine function as the periodic function because the sine function is expected to work well when extrapolated to future and out-of-sample data [Vaswani et al., 2017]. When f = sin, ω_p and φ_p are the frequency and the phase-shift of the sine function. The period of v_t^p is 2π/ω_p, so it has the same value at t and t + 2π/ω_p. In addition, referring to Time2Vec [Kazemi et al., 2019], we can also prove that the both time vectors are invariant to the scaling of the time interval and adapt to datasets with different time intervals.

某些事實會隨著時間軸週期性地發生，例如總統選舉，而某些事實則更可能在特定時間段內發生，例如，當選總統將在特定時期內參與更多活動。因此，同時考慮歷史事實的週期性與非週期性是有意義的。我們分別設計了週期性與非週期性時間向量：
v_t^p = f(ω_p t + φ_p) (6)
v_t^{np} = ω_{np} t + φ_{np} (7)
其中 v_t^p 和 v_t^{np} 分別是 d 維的週期性與非週期性時間向量，ω_p, φ_p, ω_{np} 和 φ_{np} 是可學習的參數，f 是一個週期性激活函數。我們選擇正弦函數作為週期函數，因為預期正弦函數在推斷未來和樣本外數據時能有良好的表現 [Vaswani et al., 2017]。當 f = sin 時，ω_p 和 φ_p 分別是正弦函數的頻率和相位移。v_t^p 的週期是 2π/ω_p，因此它在 t 和 t + 2π/ω_p 時具有相同的值。此外，參考 Time2Vec [Kazemi et al., 2019]，我們也可以證明這兩個時間向量對於時間間隔的縮放是不變的，並且能適應具有不同時間間隔的數據集。

***

##### Time-ConvTransE/Time-ConvTransR

##### 時間卷積 TransE/時間卷積 TransR

After obtaining time vectors, in order to perform entity prediction and relation prediction simultaneously, we design Time-ConvTransE and Time-ConvTransR inspired by ConvTransE [Shang et al., 2019] for the two tasks, respectively. Specifically, the decoder performs one-dimensional convolution on concatation of four embeddings (entity embedding h, realtion embedding rt, two time embedding v_t^p, v_t^{np}) and scores the resulting representation. Formally, the convolution operator is computed as follows:
m_c(h_s, r_t, v_t^p, v_t^{np}, n) = Σ_{τ=0}^{K-1} [w_c(τ,0)h_s(n+τ) + w_c(τ,1)r_t(n+τ) + w_c(τ,2)v_t^p(n+τ) + w_c(τ,3)v_t^{np}(n+τ)] (8)
where c is the number of convolution kernels, K is the kernel width, n ∈ [0,d] indicates the entries in the output vector, and wc are learnable kernel parameters. h_s, r_t, v_t^p and v_t^{np} are padding versions of hs, rt, v_t^p and v_t^{np}, respectively. Similar to the improvement from TransE to TTransE, the convolution operation integrates time information while maintaining the translational property of embeddings. Thus, each convolution kernel forms an output vector m_c(h_s, r_t, v_t^p, v_t^{np}) = [m_0, m_1, ..., m_{d-1}] can be aligned to get the matrix M_conv ∈ R^{cxd}.
After nonlinear one-dimensional convolution, the final output of Time-ConvTransE is defined as follows:
ψ(h_s, r_t, v_t^p, v_t^{np}) = ReLu(vec(M_conv)W)H^T (9)
where vec is a feature map operator, and W ∈ R^{cd×d} is a matrix for linear transformation. Time-ConvTransR calculates the scores in the same way, only replacing rt with hs.

在獲得時間向量後，為了同時執行實體預測和關係預測，我們分別為這兩項任務設計了受 ConvTransE [Shang et al., 2019] 啟發的 Time-ConvTransE 和 Time-ConvTransR。具體來說，解碼器對四個嵌入（實體嵌入 h、關係嵌入 rt、兩個時間嵌入 v_t^p、v_t^{np}）的串聯執行一維卷積，並對得到的表示進行評分。形式上，卷積算子計算如下：
m_c(h_s, r_t, v_t^p, v_t^{np}, n) = Σ_{τ=0}^{K-1} [w_c(τ,0)h_s(n+τ) + w_c(τ,1)r_t(n+τ) + w_c(τ,2)v_t^p(n+τ) + w_c(τ,3)v_t^{np}(n+τ)] (8)
其中 c 是卷積核的數量，K 是核寬度，n ∈ [0,d] 表示輸出向量中的條目，wc 是可學習的核參數。h_s, r_t, v_t^p 和 v_t^{np} 分別是 hs、rt、v_t^p 和 v_t^{np} 的填充版本。與從 TransE 到 TTransE 的改進類似，卷積操作在保持嵌入的平移性的同時，整合了時間資訊。因此，每個卷積核形成一個輸出向量 m_c(h_s, r_t, v_t^p, v_t^{np}) = [m_0, m_1, ..., m_{d-1}]，可以對齊以獲得矩陣 M_conv ∈ R^{cxd}。
經過非線性一維卷積後，Time-ConvTransE 的最終輸出定義如下：
ψ(h_s, r_t, v_t^p, v_t^{np}) = ReLu(vec(M_conv)W)H^T (9)
其中 vec 是一個特徵圖算子，W ∈ R^{cd×d} 是一個用於線性變換的矩陣。Time-ConvTransR 以相同的方式計算分數，僅將 rt 替換為 hs。

***

#### 3.6 Scoring Function and Training Objective

#### 3.6 評分函數與訓練目標

Since the impact of local and global historical facts may be different, we weight their importance through a variable factor. If ht and rt in Eq. (9) are the entity and relation embeddings obtained by the local recurrent encoder, then we get the output of local time-guided decoder. The output of the global decoder needs to mask the position where the value of the candidate matrix is equal to 0. After calculating the local probability decoding score plocal and the global probability decoding score pglobal by softmax activation function, the final score is obtained by summing in proportion:
pscore = softmax(ψscore) (10)
pfinal = α × pglobal + (1 − α) × plocal (11)
where variable factor α ∈ [0,1]. Entity prediction and relation prediction calculate the score in the same way, but can take different values of α.

由於局部和全域歷史事實的影響可能不同，我們通過一個可變因子來權衡它們的重要性。如果公式 (9) 中的 ht 和 rt 是由局部循環編碼器獲得的實體和關係嵌入，那麼我們就得到了局部時間引導解碼器的輸出。全域解碼器的輸出需要遮蔽候選矩陣值為 0 的位置。通過 softmax 激活函數計算出局部機率解碼分數 plocal 和全域機率解碼分數 pglobal 後，最終分數由按比例求和得出：
pscore = softmax(ψscore) (10)
pfinal = α × pglobal + (1 − α) × plocal (11)
其中可變因子 α ∈ [0,1]。實體預測和關係預測以相同的方式計算分數，但可以取不同的 α 值。

***

We regard both entity and relation predictions as multi-label learning problems and train them together. Therefore, the total loss that contains the loss of entity prediction Le and the loss of relation prediction Lr is formalized as:
L = Le + Lr = -Σ(s,r,o,t)∈G y_o log p(o|s,r,t) - Σ(s,r,o,t)∈G y_r log p(r|s,o,t) (12)
where p(o|s,r,t) and p(r|s,o,t) are the final probabilistic scores of entity and relation predictions. y_o ∈ R|E| and y_r ∈ R|R| are the label vectors for the two tasks, of which the element is 1 if the fact occurs, otherwise 0.

我們將實體和關係預測都視為多標籤學習問題，並將它們一起訓練。因此，包含實體預測損失 Le 和關係預測損失 Lr 的總損失形式化為：
L = Le + Lr = -Σ(s,r,o,t)∈G y_o log p(o|s,r,t) - Σ(s,r,o,t)∈G y_r log p(r|s,o,t) (12)
其中 p(o|s,r,t) 和 p(r|s,o,t) 是實體和關係預測的最終機率分數。y_o ∈ R|E| 和 y_r ∈ R|R| 是這兩項任務的標籤向量，如果事實發生，其元素為 1，否則為 0。

***

### 4 Experiments

### 4 實驗

#### 4.1 Setup

#### 4.1 設置

##### Datasets

##### 資料集

We use six TKG datasets to evaluate TiRGN on entity prediction and relation prediction tasks, including ICEWS14 [García-Durán et al., 2018], ICEWS18 [Jin et al., 2020], ICEWS05-15 [García-Durán et al., 2018], GDELT [Jin et al., 2020], WIKI [Leblay and Chekol, 2018], and YAGO [Mahdisoltani et al., 2015]. We follow the preprocessing strategy for datasets in RE-NET.

我們使用六個 TKG 資料集來評估 TiRGN 在實體預測和關係預測任務上的表現，包括 ICEWS14 [García-Durán et al., 2018]、ICEWS18 [Jin et al., 2020]、ICEWS05-15 [García-Durán et al., 2018]、GDELT [Jin et al., 2020]、WIKI [Leblay and Chekol, 2018] 和 YAGO [Mahdisoltani et al., 2015]。我們遵循 RE-NET 中資料集的預處理策略。

##### Evaluation Metrics

##### 評估指標

We adopt two widely used metrics to evaluate the model performance on TKG reasoning, mean reciprocal rank (MRR) and Hits@k. MRR is the average reciprocal values of the ranks of the true entity candidates for all queries, and Hits@k represents the proportion of times that the true entity candidates appear in the top k of the ranked candidates. Some recent works mention that the filtered setting [Bordes et al., 2013] is not suitable for extrapolation on TKG reasoning [Han et al., 2021a; Han et al., 2021b]. Therefore, we use the time-aware filtered setting to report the experimental results.

我們採用兩種廣泛使用的指標來評估模型在 TKG 推理上的性能：平均倒數排名（MRR）和 Hits@k。MRR 是所有查詢的真實實體候選排名倒數的平均值，而 Hits@k 則表示真實實體候選出現在排名前 k 位的次數比例。一些近期的研究提到，過濾設置 [Bordes et al., 2013] 不適用於 TKG 推理的外推 [Han et al., 2021a; Han et al., 2021b]。因此，我們使用時間感知過濾設置來報告實驗結果。

***

##### Implementation Details

##### 實作細節

For all the datasets, the embedding dimension d is set to 200. The number of one-dimensional convolution-based GCN layers is set to 2 and the dropout rate for each layer is set to 0.2. The optimal local history lengths m are set to 10, 9, 15, 2, 1, and 7 for ICEWS18, ICEWS14, ICEWS05-15, WIKI, YAGO, and GDELT, respectively. Similar to RE-GCN, static graph constraints are added for ICEWS14, ICEWS18, and ICEWS05-15. For time-guided decoders, the number of channels is set to 50 and the kernel size is set to 4×3. We tried multiple α values from 0.1 to 0.9 and finally selected 0.3 as the global weight for all the datasets. Adam is used for parameter learning, and the learning rate is set to 0.001. The code is available in https://github.com/Liyyy2122/TiRGN.

對於所有資料集，嵌入維度 d 設定為 200。一維卷積 GCN 層數設定為 2，每層的 dropout 率設定為 0.2。最佳的局部歷史長度 m 分別為 ICEWS18、ICEWS14、ICEWS05-15、WIKI、YAGO 和 GDELT 設定為 10、9、15、2、1 和 7。與 RE-GCN 類似，為 ICEWS14、ICEWS18 和 ICEWS05-15 添加了靜態圖約束。對於時間引導解碼器，通道數設定為 50，核心大小設定為 4×3。我們嘗試了從 0.1 到 0.9 的多個 α 值，最終為所有資料集選擇 0.3 作為全域權重。使用 Adam 進行參數學習，學習率設定為 0.001。程式碼可在 https://github.com/Liyyy2122/TiRGN 取得。

***

#### 4.2 Results

#### 4.2 結果

##### Results on Entity Prediction

##### 實體預測結果

The results of the entity prediction task are shown in Tables 1 and 2. On the six benchmark datasets, TiRGN continuously outperforms all baselines. Specifically, the performance of TiRGN is better than CyGNet because it not only considers the global historical repetitive facts but also pays attention to the facts at the adjacent timestamps and the periodicity of facts. RGCRN, RE-NET, TANGO, xERTE and RE-GCN consider the facts of adjacent timestamps and show strong performance in the experiment. Nevertheless, TiRGN uses a one-dimensional convolution-based GCN and double recurrence mechanism to capture more comprehensive structural features and historical dependency. Therefore, TiRGN is superior to these models that capture a single historical characteristics. Due to the historical facts search strategy based on reinforcement learning, TITer performs well on YAGO. The historical facts search strategy of TITer is suitable for the datasets with fewer timestamps and facts. However, once the dataset has too many timestamps, TITer will cause performance degradation due to its inability to model the ample search space, such as on the GDELT. In contrast, TiRGN only obtains the set of candidate entities and relations that record the occurrence of global facts without worrying about excessive complexity. Therefore, TiRGN has more substantial applicability for different datasets.
As shown in Tables 1 and 2, TiRGN has significantly improved the performance on ICEWS05-15 and GDELT with a large number of timestamps. It proves the effectiveness of capturing longer local historical dependencies through a double recurrent network. Besides, TiRGN has the most obvious effect on the datasets with more facts such as ICEWS05-15, WIKI, and GDELT, which also demonstrates that it is necessary to consider different historical characteristics when historical information is sufficient.

實體預測任務的結果如表 1 和表 2 所示。在六個基準數據集上，TiRGN 持續優於所有基線模型。具體來說，TiRGN 的性能優於 CyGNet，因為它不僅考慮了全域歷史重複事實，還關注了相鄰時間戳的事實和事實的週期性。RGCRN、RE-NET、TANGO、xERTE 和 RE-GCN 考慮了相鄰時間戳的事實，並在實驗中表現出強大的性能。儘管如此，TiRGN 使用一維卷積 GCN 和雙重循環機制來捕捉更全面的結構特徵和歷史依賴性。因此，TiRGN 優於這些僅捕捉單一歷史特徵的模型。由於基於強化學習的歷史事實搜索策略，TITer 在 YAGO 上表現良好。TITer 的歷史事實搜索策略適用於時間戳和事實較少的數據集。然而，一旦數據集有太多時間戳，TITer 會因為無法對充足的搜索空間進行建模（例如在 GDELT 上）而導致性能下降。相比之下，TiRGN 僅獲取記錄全域事實發生的候選實體和關係集，而無需擔心過度的複雜性。因此，TiRGN 對於不同的數據集具有更實質的適用性。
如表 1 和表 2 所示，TiRGN 在具有大量時間戳的 ICEWS05-15 和 GDELT 上的性能顯著提高。這證明了通過雙重循環網絡捕捉更長的局部歷史依賴性的有效性。此外，TiRGN 在具有更多事實的數據集（如 ICEWS05-15、WIKI 和 GDELT）上效果最為明顯，這也表明在歷史信息充足時，有必要考慮不同的歷史特徵。

##### Results on Relation Prediction

##### 關係預測結果

Since some models are not designed for relation prediction, we select temporal models that can be used for relation prediction. As shown in Table 3, TiRGN performs better than all baselines. TiRGN achieves limited improvement on datasets that is easy to achieve high performance due to few relations. For the datasets with a larger number of relations, the performance of TiRGN improves significantly, which verifies the observation results mentioned in the entity prediction again.

由於某些模型並非為關係預測而設計，我們選擇了可用於關係預測的時序模型。如表 3 所示，TiRGN 的表現優於所有基線模型。在關係數量少、容易達到高性能的資料集上，TiRGN 的改進有限。對於關係數量較多的資料集，TiRGN 的性能顯著提升，這再次驗證了在實體預測中提到的觀察結果。

***

#### 4.3 Ablation Studies

#### 4.3 消融研究

To better understand the effectiveness of different model components that capture the corresponding historical characteristics, we conduct ablation studies. As shown in Table 4, the local recurrent encoder (le) has the greatest impact on performance, which indicates that adjacent historical facts are crucial for the prediction. The global history encoder (ge) has a consistent impact on all the datasets, which shows the necessity of avoiding omissions for prediction. Besides, a single global history encoder that directly uses frequency (ge+fre) as features performs worse than that without frequency, which is in line with our assumptions. Since periodic facts are special cases in the datasets, the periodic time decoder (td) does not greatly improve the performance, but it will not reduce the performance even for non-periodic facts. Therefore, these results further show that different historical characteristics are all helpful to the prediction.

為了更深入地理解捕捉相應歷史特徵的不同模型組件的有效性，我們進行了消融研究。如表 4 所示，局部循環編碼器 (le) 對性能影響最大，這表明相鄰的歷史事實對於預測至關重要。全域歷史編碼器 (ge) 對所有資料集都有一致的影響，這顯示了避免預測遺漏的必要性。此外，直接使用頻率 (ge+fre) 作為特徵的單一全域歷史編碼器表現比不使用頻率的差，這與我們的假設一致。由於週期性事實在資料集中是特殊情況，週期性時間解碼器 (td) 並未大幅提升性能，但即使對於非週期性事實，它也不會降低性能。因此，這些結果進一步表明，不同的歷史特徵對預測都有幫助。

***

#### 4.4 Sensitivity Analysis

#### 4.4 敏感度分析

To explore the importance of global and local historical facts to the prediction results, we conduct a sensitivity analysis. α is a variable trade-off factor between global and local historical facts. As shown in Figure 3, neither ignoring the facts at adjacent timestamps nor ignoring the global repetitive facts can make effective prediction, which further demonstrates the necessity of combining local and global historical patterns in TiRGN. Besides, the results show that the performance is better when the value of α is from 0.2 to 0.5. This phenomenon shows that the facts of adjacent timestamps are more important than global repeated facts. By adjusting the value of α on the validation set, TiRGN can obtain the best trade-off between local and global historical facts.

為探討全域與局部歷史事實對預測結果的重要性，我們進行了敏感度分析。α 是全域與局部歷史事實之間的一個可變權衡因子。如圖 3 所示，無論是忽略相鄰時間戳的事實，還是忽略全域重複事實，都無法做出有效的預測，這進一步證明了在 TiRGN 中結合局部與全域歷史模式的必要性。此外，結果顯示，當 α 的值在 0.2 到 0.5 之間時，性能更佳。此現象表明，相鄰時間戳的事實比全域重複事實更為重要。通過在驗證集上調整 α 的值，TiRGN 能夠在局部與全域歷史事實之間取得最佳的權衡。

***

### 5 Conclusion

### 5 結論

In this paper, we propose a model named TiRGN for TKG reasoning, which learns the representations of entities and relations by capturing multiple characteristics of historical facts. We combine a local encoder that captures the structural dependency of local histories with a global encoder that captures the pattern of global repetitive histories, so as to conduct reasoning through a time-guided decoder with periodicity. The experimental results on six benchmark datasets demonstrate the significant advantages and effectiveness of TiRGN on temporal entity and relation predictions. In addition, ablation experiments show that these characteristics of historical facts play positive roles in TKG reasoning.

在本文中，我們提出了一個名為 TiRGN 的 TKG 推理模型，該模型透過捕捉歷史事實的多重特徵來學習實體和關係的表示。我們結合了一個捕捉局部歷史結構依賴性的局部編碼器與一個捕捉全域重複歷史模式的全域編碼器，從而透過一個具有週期性的時間引導解碼器進行推理。在六個基準資料集上的實驗結果證明了 TiRGN 在時序實體和關係預測上的顯著優勢與有效性。此外，消融實驗表明，這些歷史事實的特徵在 TKG 推理中扮演著正面的角色。

***

### Acknowledgements

### 致謝

This work was supported by the Shanghai Municipal Project 20511100900, the NSFC Projects 62076096 and 62006078, Shanghai Knowledge Service Platform Project ZF1213, STCSM Project 22ZR1421700 and the Fundamental Research Funds for the Central Universities.

本研究由上海市級項目 20511100900、國家自然科學基金項目 62076096 和 62006078、上海知識服務平台項目 ZF1213、上海市科委項目 22ZR1421700 以及中央高校基本科研業務費專項資金資助。

***

### References

### 參考文獻

[Bordes et al., 2013] Antoine Bordes, Nicolas Usunier, Alberto García-Durán, Jason Weston, and Oksana Yakhnenko. Translating embeddings for modeling multi-relational data. In NeurIPS, pages 2787–2795, 2013.

[Bordes et al., 2013] Antoine Bordes, Nicolas Usunier, Alberto García-Durán, Jason Weston, and Oksana Yakhnenko. 用於模擬多重關係數據的翻譯嵌入。於 NeurIPS，第 2787–2795 頁，2013。

[Deng et al., 2020] Songgaojun Deng, Huzefa Rangwala, and Yue Ning. Dynamic knowledge graph based multi-event forecasting. In KDD, pages 1585-1595, 2020.

[Deng et al., 2020] Songgaojun Deng, Huzefa Rangwala, and Yue Ning. 基於動態知識圖的多事件預測。於 KDD，第 1585-1595 頁，2020。

[García-Durán et al., 2018] Alberto García-Durán, Sebastijan Dumancic, and Mathias Niepert. Learning sequence encoders for temporal knowledge graph completion. In EMNLP, pages 4816–4821, 2018.

[García-Durán et al., 2018] Alberto García-Durán, Sebastijan Dumancic, and Mathias Niepert. 用於時序知識圖補全的學習序列編碼器。於 EMNLP，第 4816–4821 頁，2018。

[Goel et al., 2020] Rishab Goel, Seyed Mehran Kazemi, Marcus Brubaker, and Pascal Poupart. Diachronic embedding for temporal knowledge graph completion. In AAAI, pages 3988-3995, 2020.

[Goel et al., 2020] Rishab Goel, Seyed Mehran Kazemi, Marcus Brubaker, and Pascal Poupart. 用於時序知識圖補全的歷時性嵌入。於 AAAI，第 3988-3995 頁，2020。

[Han et al., 2021a] Zhen Han, Peng Chen, Yunpu Ma, and Volker Tresp. Explainable subgraph reasoning for forecasting on temporal knowledge graphs. In ICLR, pages 1-24, 2021.

[Han et al., 2021a] Zhen Han, Peng Chen, Yunpu Ma, and Volker Tresp. 用於時序知識圖預測的可解釋子圖推理。於 ICLR，第 1-24 頁，2021。

[Han et al., 2021b] Zhen Han, Zifeng Ding, Yunpu Ma, Yujia Gu, and Volker Tresp. Learning neural ordinary equations for forecasting future links on temporal knowledge graphs. In EMNLP, pages 8352-8364, 2021.

[Han et al., 2021b] Zhen Han, Zifeng Ding, Yunpu Ma, Yujia Gu, and Volker Tresp. 學習神經常微分方程以預測時序知識圖上的未來連結。於 EMNLP，第 8352-8364 頁，2021。

[Jiang et al., 2016] Tingsong Jiang, Tianyu Liu, Tao Ge, Lei Sha, Baobao Chang, Sujian Li, and Zhifang Sui. Towards time-aware knowledge graph completion. In COLING, pages 1715-1724, 2016.

[Jiang et al., 2016] Tingsong Jiang, Tianyu Liu, Tao Ge, Lei Sha, Baobao Chang, Sujian Li, and Zhifang Sui. 邁向時間感知的知識圖補全。於 COLING，第 1715-1724 頁，2016。

[Jin et al., 2020] Woojeong Jin, Meng Qu, Xisen Jin, and Xiang Ren. Recurrent event network: Autoregressive structure inference over temporal knowledge graphs. In EMNLP, pages 6669–6683, 2020.

[Jin et al., 2020] Woojeong Jin, Meng Qu, Xisen Jin, and Xiang Ren. 循環事件網絡：時序知識圖上的自回歸結構推斷。於 EMNLP，第 6669–6683 頁，2020。

[Kazemi et al., 2019] Seyed Mehran Kazemi, Rishab Goel, Sepehr Eghbali, Janahan Ramanan, Jaspreet Sahota, Sanjay Thakur, Stella Wu, Cathal Smyth, Pascal Poupart, and Marcus Brubaker. Time2vec: Learning a vector representation of time. arXiv preprint arXiv:1907.05321, 2019.

[Kazemi et al., 2019] Seyed Mehran Kazemi, Rishab Goel, Sepehr Eghbali, Janahan Ramanan, Jaspreet Sahota, Sanjay Thakur, Stella Wu, Cathal Smyth, Pascal Poupart, and Marcus Brubaker. Time2vec：學習時間的向量表示。arXiv 預印本 arXiv:1907.05321，2019。

[Leblay and Chekol, 2018] Julien Leblay and Melisachew Wudage Chekol. Deriving validity time in knowledge graph. In WWW, pages 1771–1776, 2018.

[Leblay and Chekol, 2018] Julien Leblay and Melisachew Wudage Chekol. 在知識圖中推導有效時間。於 WWW，第 1771–1776 頁，2018。

[Li et al., 2021a] Zixuan Li, Xiaolong Jin, Saiping Guan, Wei Li, Jiafeng Guo, Yuanzhuo Wang, and Xueqi Cheng. Search from history and reason for future: Two-stage reasoning on temporal knowledge graphs. In ACL, pages 4732-4743, 2021.

[Li et al., 2021a] Zixuan Li, Xiaolong Jin, Saiping Guan, Wei Li, Jiafeng Guo, Yuanzhuo Wang, and Xueqi Cheng. 從歷史中搜索並為未來推理：時序知識圖上的兩階段推理。於 ACL，第 4732-4743 頁，2021。

[Li et al., 2021b] Zixuan Li, Xiaolong Jin, Wei Li, Saiping Guan, Jiafeng Guo, Huawei Shen, Yuanzhuo Wang, and Xueqi Cheng. Temporal knowledge graph reasoning based on evolutional representation learning. In SIGIR, pages 408-417, 2021.

[Li et al., 2021b] Zixuan Li, Xiaolong Jin, Wei Li, Saiping Guan, Jiafeng Guo, Huawei Shen, Yuanzhuo Wang, and Xueqi Cheng. 基於演化表示學習的時序知識圖推理。於 SIGIR，第 408-417 頁，2021。

[Ma et al., 2019] Yunpu Ma, Volker Tresp, and Erik A. Daxberger. Embedding models for episodic knowledge graphs. J. Web Semant., 59:100490, 2019.

[Ma et al., 2019] Yunpu Ma, Volker Tresp, and Erik A. Daxberger. 情節式知識圖的嵌入模型。網路語義學期刊，59:100490，2019。

[Mahdisoltani et al., 2015] Farzaneh Mahdisoltani, Joanna Biega, and Fabian M. Suchanek. Yago3: A knowledge base from multilingual wikipedias. In CIDR, pages 1–11, 2015.

[Mahdisoltani et al., 2015] Farzaneh Mahdisoltani, Joanna Biega, and Fabian M. Suchanek. Yago3：一個來自多語言維基百科的知識庫。於 CIDR，第 1–11 頁，2015。

[Sadeghian et al., 2021] Ali Sadeghian, Mohammadreza Armanpour, Anthony Colas, and Daisy Zhe Wang. Chronor: Rotation based temporal knowledge graph embedding. In AAAI, pages 6471-6479, 2021.

[Sadeghian et al., 2021] Ali Sadeghian, Mohammadreza Armanpour, Anthony Colas, and Daisy Zhe Wang. Chronor：基於旋轉的時序知識圖嵌入。於 AAAI，第 6471-6479 頁，2021。

[Schlesinger, 1999] Arthur M Schlesinger. The cycles of American history. Houghton Mifflin Harcourt, 1999.

[Schlesinger, 1999] Arthur M Schlesinger. 美國歷史的循環。Houghton Mifflin Harcourt，1999。

[Seo et al., 2018] Youngjoo Seo, Michaël Defferrard, Pierre Vandergheynst, and Xavier Bresson. Structured sequence modeling with graph convolutional recurrent networks. In ICONIP, pages 362-373, 2018.

[Seo et al., 2018] Youngjoo Seo, Michaël Defferrard, Pierre Vandergheynst, and Xavier Bresson. 具有圖卷積循環網絡的結構化序列建模。於 ICONIP，第 362-373 頁，2018。

[Shang et al., 2019] Chao Shang, Yun Tang, Jing Huang, Jinbo Bi, Xiaodong He, and Bowen Zhou. End-to-end structure-aware convolutional networks for knowledge base completion. In AAAI, pages 3060–3067, 2019.

[Shang et al., 2019] Chao Shang, Yun Tang, Jing Huang, Jinbo Bi, Xiaodong He, and Bowen Zhou. 用於知識庫補全的端到端結構感知卷積網絡。於 AAAI，第 3060–3067 頁，2019。

[Sun et al., 2021] Haohai Sun, Jialun Zhong, Yunpu Ma, Zhen Han, and Kun He. Timetraveler: Reinforcement learning for temporal knowledge graph forecasting. In EMNLP, pages 8306–8319, 2021.

[Sun et al., 2021] Haohai Sun, Jialun Zhong, Yunpu Ma, Zhen Han, and Kun He. Timetraveler：用於時序知識圖預測的強化學習。於 EMNLP，第 8306–8319 頁，2021。

[Trivedi et al., 2017] Rakshit Trivedi, Hanjun Dai, Yichen Wang, and Le Song. Know-evolve: Deep temporal reasoning for dynamic knowledge graphs. In ICML, pages 3462-3471, 2017.

[Trivedi et al., 2017] Rakshit Trivedi, Hanjun Dai, Yichen Wang, and Le Song. Know-evolve：用於動態知識圖的深度時序推理。於 ICML，第 3462-3471 頁，2017。

[Trivedi et al., 2019] Rakshit Trivedi, Mehrdad Farajtabar, Prasenjeet Biswal, and Hongyuan Zha. Dyrep: Learning representations over dynamic graphs. In ICLR, pages 1-25, 2019.

[Trivedi et al., 2019] Rakshit Trivedi, Mehrdad Farajtabar, Prasenjeet Biswal, and Hongyuan Zha. Dyrep：學習動態圖上的表示。於 ICLR，第 1-25 頁，2019。

[Trompf, 1979] Garry Winston Trompf. The idea of historical recurrence in Western thought: From antiquity to the Reformation. Univ of California Press, 1979.

[Trompf, 1979] Garry Winston Trompf. 西方思想中的歷史重現觀念：從古代到宗教改革。加州大學出版社，1979。

[Vaswani et al., 2017] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In NeurIPS, pages 5998–6008, 2017.

[Vaswani et al., 2017] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. 注意力就是你所需要的一切。於 NeurIPS，第 5998–6008 頁，2017。

[Wu et al., 2020] Jiapeng Wu, Meng Cao, Jackie Chi Kit Cheung, and William L. Hamilton. Temp: Temporal message passing for temporal knowledge graph completion. In EMNLP, pages 5730–5746, 2020.

[Wu et al., 2020] Jiapeng Wu, Meng Cao, Jackie Chi Kit Cheung, and William L. Hamilton. Temp：用於時序知識圖補全的時序訊息傳遞。於 EMNLP，第 5730–5746 頁，2020。

[Zhu et al., 2021] Cunchao Zhu, Muhao Chen, Changjun Fan, Guangquan Cheng, and Yan Zhang. Learning from history: modeling temporal knowledge graphs with sequential copy-generation networks. In AAAI, pages 4732-4740, 2021.

[Zhu et al., 2021] Cunchao Zhu, Muhao Chen, Changjun Fan, Guangquan Cheng, and Yan Zhang. 從歷史中學習：用序列複製生成網絡建模時序知識圖。於 AAAI，第 4732-4740 頁，2021。