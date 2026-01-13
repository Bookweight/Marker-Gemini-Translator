---
title: Towards Foundation Model on Temporal Knowledge Graph Reasoning
field: Papers
status: Imported
created_date: 2026-01-13
pdf_link: "[[Towards Foundation Model on Temporal Knowledge Graph Reasoning.pdf]]"
tags:
  - paper
  - knowledge_graph
---
# 邁向時間知識圖譜推理的基礎模型

# Towards Foundation Model on Temporal Knowledge Graph Reasoning

Jiaxin Pan¹, Mojtaba Nayyeri¹, Osama Mohammed¹, Daniel Hernández¹, Rongchuan Zhang¹, Cheng Cheng¹, Steffen Staab¹,²
¹University of Stuttgart ²University of Southampton
{jiaxin.pan, mojtaba.nayyeri, osama.mohammed, daniel.hernandez}@ki.uni-stuttgart.de
{st191486, st180913}@stud.uni-stuttgart.de
{steffen.staab}@ki.uni-stuttgart.de

Jiaxin Pan¹、Mojtaba Nayyeri¹、Osama Mohammed¹、Daniel Hernández¹、Rongchuan Zhang¹、Cheng Cheng¹、Steffen Staab¹’²
¹斯圖加特大學 ²南安普敦大學
{jiaxin.pan, mojtaba.nayyeri, osama.mohammed, daniel.hernandez}@ki.uni-stuttgart.de
{st191486, st180913}@stud.uni-stuttgart.de
{steffen.staab}@ki.uni-stuttgart.de

## 摘要

## Abstract

Temporal Knowledge Graphs (TKGs) store temporal facts with quadruple formats (s,p,ο, τ). Existing Temporal Knowledge Graph Embedding (TKGE) models perform link prediction tasks in transductive or semi-inductive settings, which means the entities, relations, and temporal information in the test graph are fully or partially observed during training. Such reliance on seen elements during inference limits the models' ability to transfer to new domains and generalize to real-world scenarios. A central limitation is the difficulty in learning representations for entities, relations, and timestamps that are transferable and not tied to dataset-specific vocabularies. To overcome these limitations, we introduce the first fully-inductive approach to temporal knowledge graph link prediction. Our model employs sinusoidal positional encodings to capture fine-grained temporal patterns and generates adaptive entity and relation representations using message passing conditioned on both local and global temporal contexts. Our model design is agnostic to temporal granularity and time span, effectively addressing temporal discrepancies across TKGs and facilitating time-aware structural information transfer. As a pretrained, scalable, and transferable model, POSTRA demonstrates strong zero-shot performance on unseen temporal knowledge graphs, effectively generalizing to novel entities, relations, and timestamps. Extensive theoretical analysis and empirical results show that a single pretrained model can improve zero-shot performance on various inductive temporal reasoning scenarios, marking a significant step toward a foundation model for temporal KGs.

時間知識圖譜（TKGs）以四元組格式（s, p, o, τ）儲存時間事實。現有的時間知識圖譜嵌入（TKGE）模型在傳導式（transductive）或半傳導式（semi-inductive）設定下執行連結預測任務，這意味著測試圖譜中的實體、關係和時間資訊在訓練期間已完全或部分被觀察到。這種在推論過程中對已見元素的依賴，限制了模型轉移到新領域和泛化到真實世界場景的能力。一個核心的限制是難以學習實體、關係和時間戳的可轉移且不依賴於特定資料集詞彙的表示。為了克服這些限制，我們引入了第一個完全歸納式（fully-inductive）的時間知識圖譜連結預測方法。我們的模型採用正弦位置編碼來捕捉細微的時間模式，並使用基於局部和全域時間脈絡的訊息傳遞來生成自適應的實體和關係表示。我們的模型設計與時間粒度和時間跨度無關，有效地解決了不同TKG之間的時間差異，並促進了時間感知結構資訊的轉移。作為一個預訓練、可擴展且可轉移的模型，POSTRA在未見過的時間知識圖譜上展現出強大的零樣本（zero-shot）性能，有效地泛化到新的實體、關係和時間戳。廣泛的理論分析和實證結果表明，單一的預訓練模型可以在各種歸納式時間推理場景中提高零樣本性能，這標誌著朝向時間知識圖譜基礎模型邁出了重要一步。

## 1 簡介

## 1 Introduction

Temporal Knowledge Graphs (TKGs) extend static knowledge graphs with temporal information. In TKGs, temporal facts are represented by the quadruple format (s, p, o, t), where s, p, o and τ denote head entity, relation name, tail entity and timestamp, respectively. Temporal Knowledge Graphs (TKGs) have been researched in various fields, including question answering [Saxena et al., 2021], data integration [Ao et al., 2022], and entity alignment [Cai et al., 2024].

時間知識圖譜（TKGs）透過時間資訊擴展靜態知識圖譜。在TKG中，時間事實由四元組格式（s, p, o, τ）表示，其中s、p、o和τ分別代表頭實體、關係名稱、尾實體和時間戳。時間知識圖譜（TKGs）已在多個領域進行研究，包括問答系統 [Saxena et al., 2021]、資料整合 [Ao et al., 2022] 和實體對齊 [Cai et al., 2024]。

Link prediction is a crucial task in temporal knowledge graphs (TKGs), which involves predicting missing entities in temporal facts. Given a query of the form (?, p, o, t) or (s, p, ?, τ), the objective is to infer head or tail entity at a specific given timestamp τ. This task is essential for learning effective temporal embeddings that support various downstream applications. However, existing temporal knowledge graph embedding methods that follow transductive or semi-inductive settings suffer from transferability issues. As shown in Fig. 1 (a), the transductive setting (temporal knowledge graph interpolation) shares all the entities, relations, and timestamps information in both training and testing.

連結預測是時間知識圖譜（TKGs）中的一項關鍵任務，其涉及預測時間事實中缺失的實體。給定形式為 (?, p, o, t) 或 (s, p, ?, τ) 的查詢，目標是在特定的給定時間戳 τ 推斷頭實體或尾實體。此任務對於學習有效的時間嵌入至關重要，這些嵌入支援各種下游應用。然而，現有的時間知識圖譜嵌入方法遵循傳導式或半歸納式設定，存在可轉移性問題。如圖 1 (a) 所示，傳導式設定（時間知識圖譜內插）在訓練和測試中共享所有實體、關係和時間戳資訊。

Preprint. Under review.

預印本。審核中。

[Image]

Fig. 1 (b) depicts the semi-inductive setting (temporal knowledge graph extrapolation or forecasting), where the timestamp information in the test graph is not available during training. However, the entities and/or relations remain consistent between the training and test graphs.

圖 1 (b) 描述了半歸納設定（時間知識圖譜外推或預測），其中測試圖譜中的時間戳資訊在訓練期間不可用。然而，實體和/或關係在訓練和測試圖譜之間保持一致。

[Image]

Figure 1: Subfigure (a) shows the transductive setting, where the test graph contains only entities, relation names, and timestamps that have already been seen during training. Subfigure (b) demonstrates the semi-inductive setting, where all entities and relation names in the test graph are present in the training graph, but the timestamps (e.g., 2014-12-30) are strictly later than those observed during training. Subfigure (c) illustrates the proposed fully-inductive setting, where the test graph includes unseen entities (e.g., Mileva Marić), relation names (e.g., lives in), and timestamps (e.g., 1905).

圖 1：子圖 (a) 顯示了傳導式設定，其中測試圖譜僅包含在訓練期間已見過的實體、關係名稱和時間戳。子圖 (b) 展示了半傳導式設定，其中測試圖譜中的所有實體和關係名稱都存在於訓練圖譜中，但時間戳（例如，2014-12-30）嚴格晚於訓練期間觀察到的時間戳。子圖 (c) 說明了所提出的完全歸納式設定，其中測試圖譜包含未見過的實體（例如，Mileva Marić）、關係名稱（例如，lives in）和時間戳（例如，1905）。

As existing TKG embedding models rely on the entity, relation vocabularies and used timestamps to train specific embeddings for prediction, they need to be retrained whenever a new TKG is introduced. This limitation hinders their generalization to real-world scenarios when new entities, relations, and timestamps emerge (Fig. 1 c). To overcome the generalization limitation in TKG models, we present a fully inductive TKG learning framework, POSTRA, a model for positional time-aware, and transferable inductive TKG representations. Specifically, we address two key challenges: 1) Temporal discrepancies across TKGs and 2) Temporal-aware structural information transfer across TKGs.

由於現有的 TKG 嵌入模型依賴實體、關係詞彙表和使用的時間戳來訓練特定的預測嵌入，因此每當引入新的 TKG 時，它們都需要重新訓練。這個限制阻礙了它們在出現新實體、關係和時間戳的真實世界場景中的泛化能力（圖 1 c）。為了克服 TKG 模型中的泛化限制，我們提出了一個完全歸納的 TKG 學習框架 POSTRA，這是一個用於位置時間感知和可轉移歸納 TKG 表示的模型。具體來說，我們解決了兩個關鍵挑戰：1) 不同 TKG 之間的時間差異，以及 2) 不同 TKG 之間的時間感知結構資訊傳遞。

Starting with the first challenge, different TKGs utilize varying time units (e.g., minute/day/month) and cover diverse temporal spans (e.g., 1 month/100 years) depending on the frequency of temporal facts. Existing TKG embedding models [Lacroix et al., 2020, Zhang et al., 2022] often rely on dataset-specific trained time embeddings, which limits their ability to transfer learned temporal information across TKGs with different granularities and time spans. To address this, we leverage the transferability of relative temporal ordering between connected facts along temporal paths. We organize a TKG as a sequence of temporal snapshots, where each snapshot contains all facts sharing the same timestamp. We focus on the relative ordering of quadruples in different snapshots, rather than the specific timestamp values, and encode this ordering using sinusoidal positional encodings as time series embeddings. For example, in Figure 2, although the temporal facts in two TKGs below have different timestamps, the relative temporal ordering difference between connected temporal facts is the same as shown in the relation graphs above. Therefore, this representation captures the temporal dependencies between facts in a way that is invariant to the underlying time units or time spans and transferable to new domains, effectively bypassing the temporal discrepancies challenge. By integrating these relative temporal signals into the message passing process of relation encoder and quadruple encoder, the model can learn temporal patterns without relying on dataset-specific granularity or time span.

從第一個挑戰開始，不同的TKG使用不同的時間單位（例如，分鐘/天/月），並涵蓋不同的時間跨度（例如，1個月/100年），這取決於時間事實的頻率。現有的TKG嵌入模型 [Lacroix et al., 2020, Zhang et al., 2022] 通常依賴於特定於資料集的訓練時間嵌入，這限制了它們在具有不同粒度和時間跨度的TKG之間轉移學習到的時間資訊的能力。為了解決這個問題，我們利用了沿著時間路徑連接的事實之間的相對時間順序的可轉移性。我們將TKG組織為一個時間快照序列，其中每個快照包含共享相同時間戳的所有事實。我們專注於不同快照中四元組的相對順序，而不是特定的時間戳值，並使用正弦位置編碼將此順序編碼為時間序列嵌入。例如，在圖2中，儘管下面兩個TKG中的時間事實具有不同的時間戳，但連接的時間事實之間的相對時間順序差異與上面關係圖中所示的相同。因此，這種表示以一種對底層時間單位或時間跨度不變且可轉移到新域的方式捕捉了事實之間的時間依賴性，有效地繞過了時間差異的挑戰。通過將這些相對時間信號整合到關係編碼器和四元組編碼器的消息傳遞過程中，模型可以在不依賴於特定於資料集的粒度或時間跨度的情況下學習時間模式。

To address the second challenge of temporal-aware structural information transfer, we adopt four core relation interaction types-head-head, head-tail, tail-head, and tail-tail—that are independent of dataset-specific relation names[Galkin et al., 2024, Lee et al., 2023b] and introduce a temporal-aware message passing mechanism. By constructing entity and relation embeddings as functions conditioned on these intrinsic interactions, a pre-trained model can inductively generalize to any knowledge graph through the message-passing process, even when encountering unseen entities and relations. To incorporate temporal dynamics, POSTRA injects temporal positional embeddings into the message-passing process, enabling the resulting entity and relation representations to capture

為了應對第二個挑戰，即時間感知結構資訊的轉移，我們採用了四種核心關係互動類型——頭對頭、頭對尾、尾對頭和尾對尾——這些類型獨立於特定於資料集的關係名稱[Galkin et al., 2024, Lee et al., 2023b]，並引入了一種時間感知的訊息傳遞機制。通過將實體和關係嵌入建構成以這些內在互動為條件的函數，預訓練模型可以通過訊息傳遞過程歸納地推廣到任何知識圖譜，即使遇到未見的實體和關係。為了納入時間動態，POSTRA將時間位置嵌入注入到訊息傳遞過程中，使產生的實體和關係表示能夠捕捉

time-dependent patterns, as illustrated in Figure 2. Moreover, to differentiate between queries that share the same entities and relation names but occur at different times (e.g.,(s, p, o, τ) v.s. (s, p, ο, τ')), POSTRA adopts a dual training strategy that separately learns local and global temporal structures. Global training aggregates information across the entire training graph, capturing long-term temporal dependencies, while local training restricts updates to a fixed temporal window around each target query, focusing on short-term temporal patterns. By combining these two complementary signals, POSTRA effectively adapts to TKGs with diverse temporal dynamics and varying time spans.

如圖 2 所示，時間相關的模式。此外，為了區分共享相同實體和關係名稱但在不同時間發生的查詢（例如，(s, p, o, τ) v.s. (s, p, o, τ')），POSTRA 採用雙重訓練策略，分別學習局部和全局時間結構。全局訓練聚合整個訓練圖的資訊，捕捉長期時間依賴性，而局部訓練則將更新限制在每個目標查詢周圍的固定時間窗口內，專注於短期時間模式。通過結合這兩種互補信號，POSTRA 能有效適應具有不同時間動態和不同時間跨度的 TKG。

Different from existing TKGE models, our model does not rely on any dataset-specific entity, relation, or time embeddings (dataset-specific vocabulary embedding) and can generalize to any new TKG. While this allows the pretrained model to perform zero-shot inference across diverse TKGs, it also remains compatible with transductive and semi-inductive TKG link prediction tasks, providing flexibility across different task settings. Our model's strong performance in transferability and cross-domain generalization across diverse datasets underscores its effectiveness and marks a key advance toward a foundation model for temporal knowledge graphs.

與現有的 TKGE 模型不同，我們的模型不依賴任何特定於資料集的實體、關係或時間嵌入（特定於資料集的詞彙嵌入），並且可以推廣到任何新的 TKG。雖然這使得預訓練模型能夠在不同的 TKG 上執行零樣本推論，但它也與傳導式和半歸納式 TKG 連結預測任務相容，從而在不同的任務設定中提供了靈活性。我們的模型在不同資料集上的可轉移性和跨域泛化方面的強勁表現，突顯了其有效性，並標誌著朝向時間知識圖譜基礎模型的關鍵進展。

## 2 任務制定

## 2 Task Formulation

Let V be a finite set of entities, R be a finite set of relation names, and T = (Ti)|T|i=1 be a finite ordered set of timestamps. A temporal knowledge graph is a quad G = (V,R,T, Q), where QCV×R×V× T. The elements of Q are called temporal facts in quadruple form. Given a positive natural number i, with i < |T|, the i-snapshot of G, denoted Gi, is the subgraph of G that contains all temporal facts whose timestamp is the i-th element of T. We write ti for the timestamp at snapshot Gį. Intuitively, a temporal fact describes a relationship between two entities at a determined timestamp, and a snapshot describes all relationships that occur simultaneously at a determined timestamp.

設 V 為實體的有限集合，R 為關係名稱的有限集合，T = (Ti)|T|i=1 為時間戳的有限有序集合。時間知識圖譜是一個四元組 G = (V,R,T, Q)，其中 Q ⊆ V×R×V×T。Q 的元素稱為四元組形式的時間事實。給定一個正自然數 i，且 i < |T|，G 的第 i 個快照，記為 Gi，是 G 的子圖，包含所有時間戳為 T 的第 i 個元素的時間事實。我們用 ti 表示快照 Gį 的時間戳。直觀地說，一個時間事實描述了兩個實體在確定時間戳下的關係，而一個快照則描述了在確定時間戳下同時發生的所有關係。

[Image]

Figure 2: The bottom TKGs depict temporal knowledge graphs from different domains. The upper relation graphs show relative relation representations learned via fundamental interaction patterns (See Section 4.1) and the relative temporal ordering between corresponding facts (At) (See Section 4.2) which are transferrable across TKGs. More details are shown in Section E in Appendix.

圖 2：底部的 TKG 描述了來自不同領域的時間知識圖譜。上方的關係圖顯示了通過基本互動模式（見第 4.1 節）學習到的相對關係表示，以及對應事實（At）（見第 4.2 節）之間的相對時間排序，這些都可以在 TKG 之間轉移。更多細節請見附錄 E。

Link Prediction. We aim at link prediction, the fundamental task for TKG reasoning which predicts missing entities of queries (s,p,?,t) or (?,p,o,t). For this task, we assume: (i) one training graph Gtrain = (Vtrain, Rtrain, Ttrain, Qtrain). (ii) one inference graph Ginf = (Vinf, Rinf,Tinf, Qinf). We divide Qinf into three pairwise disjoint sets, such that Qinf = QoUQvalid ∪ Qtest. Go = (Vo, Ro, To, Qo), Gvalid = (Vvalid,Rvalid, Tvalid,Qvalid), Gtest = (Vtest, Rtest, Ttest, Qtest) refers to the observed graph, validation graph and test graph, respectively.

連結預測。我們的目標是連結預測，這是 TKG 推理的基本任務，它預測查詢 (s,p,?,t) 或 (?,p,o,t) 中缺失的實體。對於這個任務，我們假設：（i）一個訓練圖 Gtrain = (Vtrain, Rtrain, Ttrain, Qtrain)。（ii）一個推論圖 Ginf = (Vinf, Rinf,Tinf, Qinf)。我們將 Qinf 分為三個兩兩不相交的集合，使得 Qinf = Qo ∪ Qvalid ∪ Qtest。Go = (Vo, Ro, To, Qo)，Gvalid = (Vvalid,Rvalid, Tvalid,Qvalid)，Gtest = (Vtest, Rtest, Ttest, Qtest) 分別指觀察圖、驗證圖和測試圖。

At training time, we use Gtrain to train the model to predict Qtrain. At inference, we use Qo to compute the embeddings and tune hyperparameters based on the model's performance on Qvalid. Finally, we test the model's performance on Qtest.

在訓練時，我們使用 Gtrain 來訓練模型以預測 Qtrain。在推論時，我們使用 Qo 來計算嵌入並根據模型在 Qvalid 上的性能來調整超參數。最後，我們在 Qtest 上測試模型的性能。

Transductive Inference This setting requires that all entities, relation names, and timestamps are known during the training process. The inference graph contains only known entities, relation names, and timestamps (see Fig. 1-a). In this setting, we have Vtrain = Vinf, Rtrain = Rinf and Ttrain = Tinf· This implies that the entity set, relation name set, and timestamp set remain identical in both the training and test processes.

傳導式推論此設定要求所有實體、關係名稱和時間戳在訓練過程中都是已知的。推論圖只包含已知的實體、關係名稱和時間戳（參見圖 1-a）。在此設定中，我們有 Vtrain = Vinf, Rtrain = Rinf 和 Ttrain = Tinf。這意味著實體集、關係名稱集和時間戳集在訓練和測試過程中保持相同。

Semi-Inductive Inference In this setting, the entity and relation name sets are shared between the training and test graphs: Vtrain = Vinf,Rtrain = Rinf. Similarly, times are shared between the training and the observed graphs: To = Ttrain. However, this task enforces the constraint: ∀ti ∈ Ttrain, ∀tj ∈ Tvalid,∀Tk ∈ Ttest, Ti < Tj < Tk (see Fig. 1-b). This ensures that all timestamps in the validation set are strictly later than those in the training set, and all timestamps in the test set are strictly later than

在半歸納式推論中，實體與關係名稱集在訓練與測試圖之間共享：Vtrain = Vinf, Rtrain = Rinf。同樣地，時間在訓練與觀察圖之間共享：To = Ttrain。然而，此任務強制執行以下約束：∀ti ∈ Ttrain, ∀tj ∈ Tvalid, ∀Tk ∈ Ttest, Ti < Tj < Tk（見圖 1-b）。這確保了驗證集中的所有時間戳都嚴格晚於訓練集中的時間戳，且測試集中的所有時間戳都嚴格晚於

[Image]

Figure 3: The overall architecture of POSTRA. The model first constructs a Global Relation Graph by treating relations as nodes and the four types of interactions of relations as edges. The Relation Encoder learns global relation representations on the global relation graph through meassage-passing, which are then used in downstream quadruple encoding(See Section 4.1). Temporal Embedding encodes timestamps' relative ordering in the TKG by sine and cosine functions(See Section 4.2). The Quadruple Encoder processes entity representations from both Global Entity Graphs and Local Entity Graphs (see Section 4.3), generating Global and Local Quadruple Representations. Finally, these representations, along with temporal embeddings, are fused through an MLP to produce the final Prediction.

圖 3：POSTRA 的整體架構。該模型首先將關係視為節點，將四種類型的關係互動視為邊，從而建構一個全域關係圖。關係編碼器通過訊息傳遞在全域關係圖上學習全域關係表示，然後用於下游的四元組編碼（參見第 4.1 節）。時間嵌入通過正弦和餘弦函數對 TKG 中時間戳的相對順序進行編碼（參見第 4.2 節）。四元組編碼器處理來自全域實體圖和局部實體圖的實體表示（參見第 4.3 節），生成全域和局部四元組表示。最後，這些表示與時間嵌入一起通過 MLP 融合以產生最終預測。

those in the validation set, thereby meeting the requirement for predicting future temporal facts. It enables semi-inductive inference on future timestamps.

那些在驗證集中的資料，從而滿足了預測未來時間事實的要求。它能夠對未來的時間戳進行半歸納推斷。

However, the two existing inference types do not support cross-dataset inductive inference, as the entity and relation sets remain shared across datasets.

然而，這兩種現有的推論類型不支持跨資料集的歸納推論，因為實體和關係集在不同資料集之間是共享的。

Fully-Inductive Inference We propose the more challenging task of fully inductive temporal knowledge graph inference, in which the inference graph contains entities, relation names, and timestamps that never appear in the training data (see Fig. 1-c). This formally means Vtrain ∩ Vinf = 0,Rtrain ∩ Rinf = 0, Ttrain ∩ Tinf = 0. Unlike temporal extrapolation inference, this setting does not impose constraints on the sequence of timestamps between the training and inference graphs. In this setting, the pretrained model utilizes weights learned from the training graph to generate adaptive embeddings for the inference graph (with a totally new graph vocabulary). Crucially, a genuine foundation model for temporal knowledge graphs must work in this fully inductive setting and achieve maximal transferability of temporal structural patterns. To our knowledge, no existing structural temporal knowledge graph learning model works entirely in a fully inductive setting- a gap this paper aims to close as an important step toward foundation models for temporal knowledge graphs.

全歸納推論 我們提出更具挑戰性的全歸納時間知識圖譜推論任務，其中推論圖包含從未出現在訓練資料中的實體、關係名稱和時間戳（參見圖 1-c）。這在形式上意味著 Vtrain ∩ Vinf = ∅，Rtrain ∩ Rinf = ∅，Ttrain ∩ Tinf = ∅。與時間外推推論不同，此設定不對訓練圖和推論圖之間的時間戳序列施加約束。在此設定中，預訓練模型利用從訓練圖中學到的權重，為推論圖（具有全新的圖詞彙）生成自適應嵌入。至關重要的是，一個真正的時間知識圖譜基礎模型必須在這種全歸納設定下工作，並實現時間結構模式的最大可轉移性。據我們所知，目前沒有任何結構化的時間知識圖譜學習模型能夠完全在全歸納設定下工作——本文旨在填補這一空白，作為邁向時間知識圖譜基礎模型的重要一步。

## 3 相關工作

## 3 Related Work

### 3.1 時間知識圖譜嵌入

### 3.1 Temporal Knowledge Graph Embedding

Transductive Models TTransE [Leblay and Chekol, 2018] and TA-DistMult [García-Durán et al., 2018] are among the earliest models to incorporate temporal information into score functions by treating time as an additional element. T(NT)ComplEx [Lacroix et al., 2020] formulates temporal knowledge graph completion as a fourth-order tensor completion problem using semantic matching. Later work TLT-KGE [Zhang et al., 2022] extends it to quaternion space and exchanges information through quaternion operations. HGE [Pan et al., 2024] embeds temporal knowledge graphs into a

傳導模型 TTransE [Leblay and Chekol, 2018] 和 TA-DistMult [García-Durán et al., 2018] 是最早將時間資訊納入評分函數的模型之一，它們將時間視為一個附加元素。T(NT)ComplEx [Lacroix et al., 2020] 將時間知識圖譜補全問題表述為使用語義匹配的四階張量補全問題。後續工作 TLT-KGE [Zhang et al., 2022] 將其擴展到四元數空間，並通過四元數運算交換資訊。HGE [Pan et al., 2024] 將時間知識圖譜嵌入到一個

product space of suitable manifolds and adopts an attention mechanism that captures the similarities of structural patterns and manifolds. This line of work lacks generalization to unseen TKGs.

合適流形的乘積空間，並採用注意力機制來捕捉結構模式和流形的相似性。這類工作缺乏對未見過的 TKG 的泛化能力。

Temporal Extrapolation Models Most existing models utilize graph neural network architectures with various updating strategies to predict future facts [Li et al., 2022b,a, Sun et al., 2021, Liang et al., 2023]. For improved explainability, TLogic [Liu et al., 2022] extracts temporal rules using temporal random walks, while xERTE [Han et al., 2020] performs predictions based on explainable subgraphs. Additionally, TPAR [Chen et al., 2024] is a model capable of handling both transductive and temporal extrapolation tasks. It updates entity representations using Bellman-Ford-based recursive encoding on temporal paths. However, TPAR is unable to perform inductive inference where new entities, relations, and timestamps emerge.

時間外推模型 大多數現有模型利用圖神經網路架構和各種更新策略來預測未來事實 [Li et al., 2022b,a, Sun et al., 2021, Liang et al., 2023]。為了提高可解釋性，TLogic [Liu et al., 2022] 使用時間隨機遊走提取時間規則，而 xERTE [Han et al., 2020] 則基於可解釋子圖進行預測。此外，TPAR [Chen et al., 2024] 是一個能夠處理傳導式和時間外推任務的模型。它使用基於 Bellman-Ford 的遞歸編碼在時間路徑上更新實體表示。然而，TPAR 無法執行歸納式推論，其中會出現新的實體、關係和時間戳。

### 3.2 知識圖譜上的歸納學習

### 3.2 Inductive Learning on KGs

Structural Inductive Learning Early inductive models, such as NBFNet [Zhu et al., 2021], Grail [Teru et al., 2020], INDIGO [Liu et al., 2021], and Morse [Chen et al., 2022], learn relational patterns in knowledge graphs and can handle unseen entities. However, they require that the relations in the training and inference graphs remain the same. Later works, such as INGRAM [Lee et al., 2023b], TARGI [Ding et al., 2025] and ULTRA [Galkin et al., 2024], construct relation graphs where relations are treated as nodes and interactions between relations as edges. The relations are learned through similar interaction structures utilized for entity representation learning. This enables structural information sharing across training and inference graphs, allowing them to predict unseen entities and relations. However, none of these models can infer temporal facts with unseen timestamps.

結構歸納學習 早期的歸納模型，例如 NBFNet [Zhu et al., 2021]、Grail [Teru et al., 2020]、INDIGO [Liu et al., 2021] 和 Morse [Chen et al., 2022]，學習知識圖譜中的關係模式，並能處理未見實體。然而，它們要求訓練和推論圖譜中的關係保持不變。後來的研究，例如 INGRAM [Lee et al., 2023b]、TARGI [Ding et al., 2025] 和 ULTRA [Galkin et al., 2024]，建構了關係圖，其中關係被視為節點，而關係之間的互動則被視為邊。關係是透過用於實體表示學習的相似互動結構來學習的。這使得結構資訊能夠在訓練和推論圖譜之間共享，從而能夠預測未見的實體和關係。然而，這些模型都無法推斷具有未見時間戳的時間事實。

Textual Descriptions and LLMs SST-BERT [Chen et al., 2023], ECOLA [Han et al., 2023], and PPT [Xu et al., 2023] utilize BERT-based architectures by pretraining on a TKG corpus, encoding additional textual information of facts, and converting link prediction into masked token prediction, respectively. ICL [Lee et al., 2023a] explores in-context learning of temporal facts within LLMs, while GenTKG [Liao et al., 2024] encodes temporal paths retrieved through logical rule-based methods in LLMs. zrLLM [Ding et al., 2024] first generates relation representations by inputting their textual descriptions into LLMs. All these methods demand substantial computational resources or rely on text descriptions of entities and relations. In contrast, our method is purely structure-driven: it learns directly from the topology of temporal knowledge graphs and does not require any textual annotations for entities or relations.

文字描述與大型語言模型 SST-BERT [Chen et al., 2023]、ECOLA [Han et al., 2023] 和 PPT [Xu et al., 2023] 利用基於 BERT 的架構，透過在 TKG 語料庫上進行預訓練、編碼事實的附加文字資訊，並將連結預測轉換為遮蔽詞元預測。ICL [Lee et al., 2023a] 探索在大型語言模型中對時間事實進行情境學習，而 GenTKG [Liao et al., 2024] 則在大型語言模型中編碼透過基於邏輯規則的方法檢索到的時間路徑。zrLLM [Ding et al., 2024] 首先透過將其文字描述輸入大型語言模型來生成關係表示。所有這些方法都需要大量的計算資源或依賴實體和關係的文字描述。相比之下，我們的方法是純粹由結構驅動的：它直接從時間知識圖譜的拓撲結構中學習，不需要任何實體或關係的文字註釋。

## 4 方法論

## 4 Methodology

To enable fully inductive inference on TKGs, we propose the POSTRA model. The overall architecture of the model is illustrated in Figure 3.

為了在 TKG 上實現完全歸納推論，我們提出了 POSTRA 模型。該模型的整體架構如圖 3 所示。

### 4.1 關係結構學習

### 4.1 Relational Structure Learning

To enable the relational transferability across datasets, we adopt the relation encoding method proposed by ULTRA [Galkin et al., 2024]. ULTRA constructs a relation-relation interaction graph for every knowledge graph with four common types of interactions between relations in the knowledge graph: H = {h2h,h2t,t2h,t2t}, which means head-to-head, head-to-tail, tail-to-head, and tail-to-tail interactions. Interactions of Relations Construction part in Fig. 3 illustrates these interactions.

為了實現跨資料集的關係可轉移性，我們採用了 ULTRA [Galkin et al., 2024] 提出的關係編碼方法。ULTRA 為每個知識圖譜建構了一個關係-關係互動圖，其中包含知識圖譜中關係之間的四種常見互動類型：H = {h2h, h2t, t2h, t2t}，分別代表頭對頭、頭對尾、尾對頭和尾對尾互動。圖 3 中的關係建構互動部分說明了這些互動。

Given a TKG G, we construct a relation graph G₁ = (R,H,O). In G, the nodes are the relation names of G, and the edges are the interactions of these relations. For example, (p1,h2h, p2) belongs to O if and only if there are two temporal facts of the form (v1, P1, V2, T1) and (v1, P2, V3, T2) in graph G.

給定一個 TKG G，我們建構一個關係圖 G₁ = (R,H,O)。在 G 中，節點是 G 的關係名稱，邊是這些關係的互動。例如，(p1, h2h, p2) 屬於 O 若且唯若在圖 G 中存在兩個形式為 (v1, P1, V2, T1) 和 (v1, P2, V3, T2) 的時間事實。

Given a temporal query (s, p, ?, τ) and a relation graph Gr, we then obtain d-dimensional relation representations rq, q ∈ R via message-passing with neighboring relation nodes:
rqP = 1v=s *rp, v∈G, (1)
e(l+1) = AGG (T-MSG(e(l) , r(l) , g(l+1)(TE )) | w ∈ N (v), q ∈ R,∀t : (e , q,v, t ) ∈ G), (2)
where r(l+1) stands for the relation q representation at (l + 1)-th layer. Nh(p) stands for the neighboring relation nodes of p which are connected by interations in H. AGG is the aggregation function

給定一個時間查詢 (s, p, ?, τ) 和一個關係圖 Gr，我們接著透過與相鄰關係節點的訊息傳遞，得到 d 維的關係表示 rq, q ∈ R：
rqP = 1v=s *rp, v∈G, (1)
e(l+1) = AGG (T-MSG(e(l) , r(l) , g(l+1)(TE )) | w ∈ N (v), q ∈ R,∀t : (e , q,v, t ) ∈ G), (2)
其中 r(l+1) 代表在第 (l+1) 層的關係 q 表示。Nh(p) 代表與 p 透過 H 中的互動相連的相鄰關係節點。AGG 是聚合函數。

(e.g., sum), and MSG is the message function (e.g., DistMult's MSG [Yang et al., 2015] is rw|p *h, TransE's MSG [Bordes et al., 2013] is r rw\p+h), which defines how information is passed from a node's neighbors to that node. rop is initialized by detecting if the relation node is equal to the query relation p. The GNN architecture follows Neural Bellman-Ford Network[Zhu et al., 2021]. As these four interactions are universal and independent of datasets, the relational transferability can be achieved by transferring the embeddings of the four interactions.

（例如，sum），而 MSG 是訊息函數（例如，DistMult 的 MSG [Yang et al., 2015] 是 rw|p * h，TransE 的 MSG [Bordes et al., 2013] 是 r rw\p + h），它定義了資訊如何從一個節點的鄰居傳遞到該節點。rop 是透過檢測關係節點是否等於查詢關係 p 來初始化的。GNN 架構遵循 Neural Bellman-Ford Network[Zhu et al., 2021]。由於這四種互動是通用的且獨立於資料集，因此可以通過轉移這四種互動的嵌入來實現關係的可轉移性。

### 4.2 時間嵌入

### 4.2 Temporal Embedding

While previous works have used universal structures in graphs as vocabularies for pretraining Graph Foundation Models (GFM) [Sun et al., 2025, Wang et al., 2024], temporal information differs fundamentally from nodes and edges in graphs, as time is inherently sequential. To address this, we represent a temporal knowledge graph (TKG) as a sequence of TKG snapshots Gi, where timestamp index i denotes the position of G¡ in the sequence. Given that sinusoidal positional encodings have been shown to effectively capture sequential dependencies, we adopt a similar structure to encode temporal information in TKG sequences. Specifically, we utilize sine and cosine functions with different frequencies to encode the temporal information for Gi with the same dimension size as the proposed model.
[TE(i)]2n = sin(@ni), [TE(i)]2n+1 = cos(@ni),
@n∈ R, 0<n< dpe/2,
where n is the dimension index of the embedding. On can be trainable or set to Wn = β^(-2n/dpe) where dpe is the dimension size of the temporal embedding and quadruple representation. We also denote representation of ti by TEti = TE(i) where i is the timestamp index. The original Transformer architecture sets β as 10000. In this way, each dimension of the positional encoding corresponds to a sinusoidal function, with wavelengths forming a geometric progression from 2π to 10000 · 2π. We chose this function for its computational simplicity and dataset-agnostic design, which preserves temporal transferability. Moreover, prior work has demonstrated that such positional encodings enable the model to effectively learn relative position dependencies [Su et al., 2024, Vaswani et al., 2017], which we argue is crucial for capturing temporal dependencies between temporal facts.

雖然先前的研究已使用圖中的通用結構作為預訓練圖形基礎模型（GFM）的詞彙 [Sun et al., 2025, Wang et al., 2024]，但時間資訊與圖中的節點和邊緣有根本的不同，因為時間本質上是序列性的。為了解決這個問題，我們將時間知識圖（TKG）表示為 TKG 快照 Gi 的序列，其中時間戳索引 i 表示 Gi 在序列中的位置。鑑於正弦位置編碼已被證明能有效捕捉序列依賴性，我們採用類似的結構來編碼 TKG 序列中的時間資訊。具體來說，我們利用具有不同頻率的正弦和餘弦函數來編碼 Gi 的時間資訊，其維度大小與所提出的模型相同。
[TE(i)]2n = sin(ωni), [TE(i)]2n+1 = cos(ωni),
ωn ∈ R, 0 < n < dpe/2,
其中 n 是嵌入的維度索引。ωn 可以是可訓練的，也可以設置為 Wn = β^(-2n/dpe)，其中 dpe 是時間嵌入和四元組表示的維度大小。我們還用 TEti = TE(i) 表示 ti 的表示，其中 i 是時間戳索引。原始的 Transformer 架構將 β 設置為 10000。這樣，位置編碼的每個維度都對應一個正弦函數，其波長從 2π 到 10000 · 2π 形成一個幾何級數。我們選擇這個函數是因為其計算簡單性和與數據集無關的設計，這保留了時間的可轉移性。此外，先前的研究已經證明，這樣的位置編碼使模型能夠有效地學習相對位置依賴性 [Su et al., 2024, Vaswani et al., 2017]，我們認為這對於捕捉時間事實之間的時間依賴性至關重要。

### 4.3 時間感知四元學習

### 4.3 Temporal-Aware Quadruple Learning

Global Quadruple Representation We incorporate temporal information by integrating the temporal embedding into quadruple-level message passing. For a given temporal query (s, p, ?, ti), we first derive the d-dimensional entity representation ev|s, v ∈ G on the entity graph conditioned on the temporal query with head entity s, following a similar approach as described in Section 4.1:
e(0) = 1v=s *rp, v∈G, (3)
e(l+1) = AGG (T-MSG(e(l), r(l), g(l+1)(TEtj)) | w ∈ Ng(v), q ∈ R,∀tj : (ew, q,v, tj) ∈ G), (4)
vs
where et¹ is the quadruple representation at the (l+1)th layer, Np(s) denotes the neighboring entity nodes of s, and TEt; embedding of all timestamp indices where (s, p, v, tj) occurs in the training graph during training and in the inference graph during testing. g'+1(.) is a linear function defined per layer. T-MSG is a temporal aggregation mechanism that incorporates the temporal embedding described in Section 4.2. In our experiments, we employ TTransE [Leblay and Chekol, 2018], TComplEx [Lacroix et al., 2020], and TNTComplEx[Lacroix et al., 2020] as temporal modeling approaches. The initial entity representation e(0) is determined by checking whether the entity node v matches the query entity s. If so, it is initialized using the relation representation rp learned in Section 4.1. Since both the relation representations from Section 4.1 and the temporal representations from Section 4.2 are transferable, the learned entity embeddings also satisfy the transferability requirement.

全域四元表示 我們透過將時間嵌入整合到四元級訊息傳遞中來納入時間資訊。對於給定的時間查詢 (s, p, ?, ti)，我們首先按照第 4.1 節中描述的類似方法，推導出以頭實體 s 為條件的實體圖上 d 維實體表示 ev|s, v ∈ G：
e(0) = 1v=s *rp, v∈G, (3)
e(l+1) = AGG (T-MSG(e(l), r(l), g(l+1)(TEtj)) | w ∈ Ng(v), q ∈ R,∀tj : (ew, q,v, tj) ∈ G), (4)
vs
其中 e(l+1) 是第 (l+1) 層的四元表示，Np(s) 表示 s 的相鄰實體節點，TEtj 是訓練期間和測試期間在訓練圖和推論圖中出現 (s, p, v, tj) 的所有時間戳索引的嵌入。g(l+1)(.) 是每層定義的線性函數。T-MSG 是一種時間聚合機制，它納入了第 4.2 節中描述的時間嵌入。在我們的實驗中，我們採用 TTransE [Leblay and Chekol, 2018]、TComplEx [Lacroix et al., 2020] 和 TNTComplEx[Lacroix et al., 2020] 作為時間建模方法。初始實體表示 e(0) 是通過檢查實體節點 v 是否與查詢實體 s 匹配來確定的。如果是，則使用第 4.1 節中學習的關係表示 rp 進行初始化。由於第 4.1 節的關係表示和第 4.2 節的時間表示都是可轉移的，因此學習到的實體嵌入也滿足可轉移性要求。

Local Quadruple Representation Relations in TKGs may exhibit different frequencies of change, ranging from fully static to rapidly evolving behaviors [Lacroix et al., 2020]. For example, the relation CapitalOf tends to remain stable over time, whereas the relation Consult is typically short-term. To effectively model these dynamic relations, which are more influenced by adjacent quadruples,

局部四元表示 TKG 中的關係可能表現出不同的變化頻率，從完全靜態到快速演變的行為 [Lacroix et al., 2020]。例如，關係 CapitalOf 傾向於隨時間保持穩定，而關係 Consult 通常是短期的。為了有效地模擬這些受相鄰四元組影響更大的動態關係，

we construct a local graph for a given temporal query (s, p, ?, τ₁) ∈ G₁. Specifically, we define the local graph as Glocal = {Gi−k,...,G₁, ..., Gi+k}, where k is time window size. The local quadruple representation ev|s,ti,local is calculated as:
e(0) = 1v=s * rp, v ∈ Glocal,
e(l+1) = AGG (T-MSG(e(l), r(l), g(l+1)(TEtj)) | w ∈ Nq(v), q ∈ R, ∀tj ∈ T (ew, q, v, tj) ∈ Glocal) (5)
vis, ti,local
Here, we reuse the message-passing structure from Equation 3 to minimize the number of parameters while maintaining consistency in the model architecture.

我們為給定的時間查詢 (s, p, ?, τ₁) ∈ G₁ 建構一個局部圖。具體來說，我們將局部圖定義為 Glocal = {Gi−k,...,G₁, ..., Gi+k}，其中 k 是時間窗口大小。局部四元表示 ev|s,ti,local 的計算方式如下：
e(0) = 1v=s * rp, v ∈ Glocal,
e(l+1) = AGG (T-MSG(e(l), r(l), g(l+1)(TEtj)) | w ∈ Nq(v), q ∈ R, ∀tj ∈ T (ew, q, v, tj) ∈ Glocal) (5)
vis, ti,local
在這裡，我們重用方程式 3 中的訊息傳遞結構，以最小化參數數量，同時保持模型架構的一致性。

Prediction Given a temporal query (s, p,?, ti) and a tail candidate ocand, the final score is:
Vs,p,ocand = αeocand|s,ti,local +(1-α) eocands, ti,
S(s, p,ocand, Ti) = fθ (Vs,p,ocand, TEt;), (6)
where Vs,p,ocand is the time-aware triple representation (from message passing), fθ(.) is a multilayer perceptron with parameters θ, and α is a hyperparameter, providing a tradeoff between local and global information. The method offers theoretical benefits, including the ability to model relations occurring at different temporal frequencies; formal theorems and their proofs are in Appendix I.

預測 給定一個時間查詢 (s, p, ?, ti) 和一個尾部候選者 ocand，最終分數為：
Vs,p,ocand = αeocand|s,ti,local + (1-α)eocands, ti,
S(s, p, ocand, Ti) = fθ(Vs,p,ocand, TEt;), (6)
其中 Vs,p,ocand 是時間感知的三元組表示（來自訊息傳遞），fθ(.) 是帶有參數 θ 的多層感知器，α 是一個超參數，用於在局部和全局資訊之間進行權衡。該方法提供了理論上的好處，包括能夠模擬在不同時間頻率上發生的關係；形式化的定理及其證明見附錄 I。

## 5 實驗設置

## 5 Experimental Setup

Datasets To evaluate the effectiveness of the proposed model on the fully-inductive inference task, we conduct link prediction experiments on five widely-used temporal knowledge graph benchmark datasets: ICEWS14 [García-Durán et al., 2018], ICEWS05-15 [García-Durán et al., 2018], GDELT [Trivedi et al., 2017], ICEWS18 [Jin et al., 2020], and YAGO [Jin et al., 2020]. These datasets cover a diverse range of domains and temporal characteristics:

資料集 為了評估所提模型在全歸納推論任務上的有效性，我們在五個廣泛使用的時間知識圖譜基準資料集上進行了連結預測實驗：ICEWS14 [García-Durán et al., 2018]、ICEWS05-15 [García-Durán et al., 2018]、GDELT [Trivedi et al., 2017]、ICEWS18 [Jin et al., 2020] 和 YAGO [Jin et al., 2020]。這些資料集涵蓋了不同的領域和時間特性：

1. Inference Type: ICEWS14, ICEWS05-15, and GDELT were designed for transductive inference, while ICEWS18 and YAGO were developed for temporal extrapolation tasks.
2. Domain Coverage: ICEWS14, ICEWS05-15, and ICEWS18 are subsets of the Integrated Crisis Early Warning System (ICEWS) [Lautenschlager et al., 2015], consisting of news event data. GDELT is a large-scale graph focused on capturing human behavior events. YAGO is derived from the YAGO knowledge base for commonsense facts.
3. Temporal Granularity: ICEWS14, ICEWS05-15, and ICEWS18 use daily timestamps. GDELT provides fine-grained 15-minute intervals, while YAGO uses yearly granularity.
4. Time Span: ICEWS14 covers the year 2014, ICEWS05-15 spans from 2005 to 2015, and ICEWS18 includes events from January 1 to October 31, 2018. GDELT spans one year from April 2015 to March 2016, while YAGO spans a long range of 189 years.

1. 推論類型：ICEWS14、ICEWS05-15 和 GDELT 是為傳導式推論設計的，而 ICEWS18 和 YAGO 是為時間外推任務開發的。
2. 領域覆蓋：ICEWS14、ICEWS05-15 和 ICEWS18 是綜合危機預警系統 (ICEWS) [Lautenschlager et al., 2015] 的子集，由新聞事件資料組成。GDELT 是一個專注於捕捉人類行為事件的大規模圖。YAGO 衍生自 YAGO 知識庫，用於常識性事實。
3. 時間粒度：ICEWS14、ICEWS05-15 和 ICEWS18 使用每日時間戳。GDELT 提供細粒度的 15 分鐘間隔，而 YAGO 使用年度粒度。
4. 時間跨度：ICEWS14 涵蓋 2014 年，ICEWS05-15 跨越 2005 年至 2015 年，ICEWS18 包括 2018 年 1 月 1 日至 10 月 31 日的事件。GDELT 跨越 2015 年 4 月至 2016 年 3 月的一年，而 YAGO 則跨越 189 年的漫長範圍。

To assess the model's fully-inductive inference capabilities, we perform cross-dataset zero-shot evaluations, where the model is trained on one dataset and evaluated on another, ensuring no overlap in entities, relations, or timestamps. Table 6 in the Appendix shows the detailed splits of the datasets.

為了評估模型的完全歸納推論能力，我們執行跨資料集的零樣本評估，其中模型在一個資料集上進行訓練，並在另一個資料集上進行評估，確保實體、關係或時間戳沒有重疊。附錄中的表 6 顯示了資料集的詳細劃分。

Baselines We compare POSTRA with two SOTA fully-inductive inference models, INGRAM[Lee et al., 2023b] and ULTRA[Galkin et al., 2024], which can handle unknown entities and relations for fully-inductive inference. We also compare with two LLM-based temporal knowledge graph reasoning models, ICL[Lee et al., 2023a] and GenTKG[Liao et al., 2024], which focus on temporal extrapolation inference task.

基準線 我們將 POSTRA 與兩種最先進的完全歸納推論模型進行比較，INGRAM [Lee et al., 2023b] 和 ULTRA [Galkin et al., 2024]，它們可以處理未知實體和關係以進行完全歸納推論。我們還與兩種基於 LLM 的時間知識圖譜推理模型進行比較，ICL [Lee et al., 2023a] 和 GenTKG [Liao et al., 2024]，它們專注於時間外推推論任務。

Evaluation Metrics We adopt the link prediction task to evaluate our proposed model. Link prediction infers the missing entities for incomplete facts. During the test step, we follow the procedure of [Xu et al., 2020] to generate candidate quadruples. From a test quadruple (s,p, o, t), we replace s with all ŝ∈ V and o with all ō ∈ V to get candidate answer quadruples (s,p,ō, t) and (ŝ, p,ο, τ) to queries (s, p, ?, t) and (?, p,o, τ). For each query, all candidate answer quadruples will be ranked by their scores using a time-aware filtering strategy [Goel et al., 2020]. We evaluate our models with four metrics: Mean Reciprocal Rank (MRR), the mean of the reciprocals of predicted ranks of correct

評估指標 我們採用連結預測任務來評估我們提出的模型。連結預測推斷不完整事實中缺失的實體。在測試階段，我們遵循 [Xu et al., 2020] 的程序來生成候選四元組。從一個測試四元組 (s,p, o, t)，我們將 s 替換為所有 ŝ∈ V，將 o 替換為所有 ō ∈ V，以得到候選答案四元組 (s,p,ō, t) 和 (ŝ, p,ο, τ)，對應查詢 (s, p, ?, t) 和 (?, p,o, τ)。對於每個查詢，所有候選答案四元組將根據其分數使用時間感知過濾策略 [Goel et al., 2020] 進行排名。我們使用四個指標評估我們的模型：平均倒數排名 (MRR)，即正確預測排名的倒數的平均值

[Image]

quadruples, and Hits@(1/10), the percentage of ranks not higher than 1/10. For all experiments, the higher the better. For ICEWS18 and YAGO, we perform the single-step prediction as mentioned in [Gastinger et al., 2023]. We provide the hyperparameters and training details in Section F in the appendix.

四元組，以及 Hits@(1/10)，即排名不大於 1/10 的百分比。對於所有實驗，越高越好。對於 ICEWS18 和 YAGO，我們執行 [Gastinger et al., 2023] 中提到的單步預測。我們在附錄 F 中提供了超參數和訓練細節。

## 6 實驗結果

## 6 Experimental Result

### 6.1 完全歸納推理結果

### 6.1 Fully-Inductive Inference Results

We evaluate POSTRA's fully-inductive inference performance across multiple zero-shot scenarios, where no fine-tuning is applied during testing, as presented in Table 1. POSTRA consistently outperforms baseline models across multiple evaluation metrics, demonstrating its strong fully-inductive learning capability. From the result, we have following observations:

我們評估了 POSTRA 在多個零樣本場景下的完全歸納推論性能，在測試期間不進行微調，如表 1 所示。POSTRA 在多個評估指標上始終優於基線模型，展現了其強大的完全歸納學習能力。從結果中，我們有以下觀察：

1) POSTRA generalizes well to datasets with varying time spans and temporal granularities. The time span of the experimental datasets ranges from just 1 month (GDELT) to 189 years (YAGO), while their temporal granularities vary from 15 minutes (GDELT) to 1 year (YAGO). Despite being trained on one dataset and evaluated on another with significantly different temporal characteristics, POSTRA consistently achieves strong performance across all scenarios. This demonstrates the robustness and generalization ability of our sequential temporal embedding in handling diverse and unseen temporal information. Furthermore, the performance gap between POSTRA and ULTRA is more pronounced when trained on ICEWS14 compared to ICEWS05-15, suggesting that the proposed sequential temporal embedding is particularly beneficial for smaller datasets.

1) POSTRA 對具有不同時間跨度和時間粒度的資料集具有良好的泛化能力。實驗資料集的時間跨度從僅 1 個月 (GDELT) 到 189 年 (YAGO)，而其時間粒度則從 15 分鐘 (GDELT) 到 1 年 (YAGO) 不等。儘管在一個資料集上進行訓練，並在另一個具有顯著不同時間特性的資料集上進行評估，POSTRA 在所有情境下始終取得優異的性能。這證明了我們的序列時間嵌入在處理多樣化和未見過的時間資訊方面的穩健性和泛化能力。此外，與在 ICEWS05-15 上訓練相比，在 ICEWS14 上訓練時 POSTRA 和 ULTRA 之間的性能差距更為顯著，這表明所提出的序列時間嵌入對於較小的資料集特別有益。

2) POSTRA generalizes well across datasets from different domains and varying densities. The experimental datasets cover a wide range of domains, from encyclopedic knowledge in YAGO to diplomatic event data in ICEWS. POSTRA achieves impressive results even when trained on one domain and evaluated on another (see "Trained on ICEWS14 to YAGO"), demonstrating the model's ability to transfer across domains. This highlights that the learned quadruple representations from the relation encoder and quadruple encoder are capable of capturing domain-specific structural knowledge. Moreover, the datasets also vary in density: GDELT has more frequent events per timestamp, while ICEWS14 is relatively sparse. POSTRA consistently performs well in cross-dataset evaluation, demonstrating its robustness in event densities.

2) POSTRA在不同領域和不同密度的數據集上都表現出良好的泛化能力。實驗數據集涵蓋了廣泛的領域，從YAGO中的百科知識到ICEWS中的外交事件數據。即使在一個領域進行訓練並在另一個領域進行評估（參見“Trained on ICEWS14 to YAGO”），POSTRA也取得了令人矚目的成果，證明了模型跨領域轉移的能力。這凸顯了從關係編碼器和四元組編碼器中學習到的四元組表示能夠捕獲特定領域的結構知識。此外，數據集的密度也各不相同：GDELT每個時間戳有更頻繁的事件，而ICEWS14相對稀疏。POSTRA在跨數據集評估中始終表現良好，證明了其在事件密度方面的穩健性。

3) POSTRA generalizes well to both temporal knowledge graph interpolation and extrapolation tasks. As shown in Table 2, POSTRA achieves strong results when trained to predict historical unseen events and tested on future unseen events, even outperforming LLM-based models, which are designed for extrapolation tasks. Notably, the cost of LLM-based models to compute the standard MRR metric

3) POSTRA在時間知識圖譜內插和外推任務中均表現出良好的泛化能力。如表2所示，當訓練用於預測歷史未見事件並在未來未見事件上進行測試時，POSTRA取得了優異的結果，甚至優於專為外推任務設計的基於LLM的模型。值得注意的是，計算標準MRR指標時，基於LLM的模型的成本

over all candidate entities is prohibitive, whereas POSTRA efficiently supports full evaluation. This highlights both the flexibility and the computational efficiency of POSTRA.

對所有候選實體而言，其成本過高，而 POSTRA 能有效支援全面評估。這凸顯了 POSTRA 的靈活性與計算效率。

[Image]

### 6.2 消融研究

### 6.2 Ablation Study

Module Ablation To validate the effectiveness of each module in POSTRA, we conducted ablation studies on three key components: Temporal Embedding (TE), Global Quadruple Representation (GQR), and Local Quadruple Representation (LQR). As shown in Table 3, the results highlight the importance of each component. Removing the Temporal Embedding module results in a substantial performance drop, emphasizing Temporal Embedding's transferability between different temporal knowledge graphs. Excluding the Global Quadruple Representation significantly reduces the model's Hits@10 score,

模組消融 為了驗證 POSTRA 中每個模組的有效性，我們對三個關鍵組件進行了消融研究：時間嵌入（TE）、全域四元表示（GQR）和局部四元表示（LQR）。如表 3 所示，結果突顯了每個組件的重要性。移除時間嵌入模組會導致性能大幅下降，強調了時間嵌入在不同時間知識圖譜之間的可轉移性。排除全域四元表示會顯著降低模型的 Hits@10 分數，

[Image]

supporting our hypothesis that it captures long-term global temporal information. Similarly, removing the Local Quadruple Representation notably decreases the Hits@1 score, aligning with our assumption that it models the concurrent interactions of temporal events. We provide a more detailed visualization of GQR and LQR in Section H in the Appendix.

支持我們的假設，即它捕捉了長期的全局時間資訊。同樣地，移除局部四元表示會顯著降低 Hits@1 分數，這與我們的假設一致，即它模擬了時間事件的並發互動。我們在附錄 H 中提供了 GQR 和 LQR 的更詳細視覺化。

Performance on Temporal Structural Patterns To further assess the model's ability to capture temporal structural patterns, we evaluate on two temporal structural patterns: symmetric and inverse. We construct subsets from the ICEWS05-15 test set that contain such structural patterns¹ and report results in Table 4. A pair of quadruples is symmetric if for (s, p,ο, τ₁), (o, p,s, t₂) exists. A relation pair (p,p') is inverse if (s,p,o, t₁) implies (o, p',s, t2). For both patterns, POSTRA consistently outperforms the baseline ULTRA. These results demonstrate that incorporating relative temporal signals significantly enhances the model's ability to learn and generalize structural patterns in TKGs.

時間結構模式的表現 為了進一步評估模型捕捉時間結構模式的能力，我們在對稱和反向這兩種時間結構模式上進行評估。我們從 ICEWS05-15 測試集中建構包含此類結構模式的子集¹，並在表 4 中報告結果。如果對於 (s, p, o, τ₁)，存在 (o, p, s, t₂)，則一對四元組是對稱的。如果 (s, p, o, t₁) 意味著 (o, p', s, t₂)，則關係對 (p, p') 是反向的。對於這兩種模式，POSTRA 一致優於基線 ULTRA。這些結果表明，納入相對時間信號顯著增強了模型學習和推廣 TKG 中結構模式的能力。

[Image]

## 7 結論

## 7 Conclusion

We introduce POSTRA, which advances beyond existing transductive or semi-inductive approaches to support fully inductive inference. POSTRA exhibits strong universal transferability across diverse time spans, temporal granularities, domains, and prediction tasks, overcoming limitations of prior models that depended on dataset-specific entity, relation, and timestamp embeddings. Extensive evaluations across multiple scenarios show that POSTRA consistently outperforms existing state-of-the-art models in zero-shot settings. In future work, we aim to extend POSTRA to support time prediction tasks and explore richer temporal representations.

我們介紹了 POSTRA，它超越了現有的傳導式或半歸納式方法，以支援完全歸納式推論。POSTRA 在不同的時間跨度、時間粒度、領域和預測任務中表現出強大的通用可轉移性，克服了先前模型依賴於特定資料集的實體、關係和時間戳嵌入的限制。在多種情境下的廣泛評估表明，POSTRA 在零樣本設定中始終優於現有的最先進模型。在未來的工作中，我們旨在擴展 POSTRA 以支援時間預測任務，並探索更豐富的時間表示。

Acknowledgment

致謝

This work has received funding from the CHIPS Joint Undertaking (JU) under grant agreement No. 101140087 (SMARTY). The JU receives support from the European Union's Horizon Europe research and innovation programme. Furthermore, on national level this work is supported by the German Federal Ministry of Education and Research (BMBF) under the sub-project with the funding number 16MEE0444. The authors gratefully acknowledge the computing time provided on the high-performance computer HoreKa by the National High-Performance Computing Center at KIT (NHR@KIT). This center is jointly supported by the Federal Ministry of Education and Research and the Ministry of Science, Research and the Arts of Baden-Württemberg, as part of the National High-Performance Computing (NHR) joint funding program (https://www.nhr-verein.de/en/our-partners). HoreKa is partly funded by the German Research Foundation (DFG).

本研究獲 CHIPS 聯合承辦單位 (JU) 依據補助協議編號 101140087 (SMARTY) 補助。JU 獲歐盟展望歐洲研究與創新計畫支持。此外，本研究在國家層級獲德國聯邦教育與研究部 (BMBF) 依據補助編號 16MEE0444 之子計畫支持。作者由衷感謝卡爾斯魯厄理工學院國家高效能運算中心 (NHR@KIT) 於高效能電腦 HoreKa 上提供之運算時間。該中心由德國聯邦教育與研究部及巴登-符騰堡邦科學、研究與藝術部共同支持，為國家高效能運算 (NHR) 聯合補助計畫 (https://www.nhr-verein.de/en/our-partners) 之一環。HoreKa 部分由德國研究基金會 (DFG) 補助。

## 參考文獻

## References

Jing Ao, Jon Doyle, Christopher Healey, and Ranga Raju Vatsavai. Temporal Knowledge Graphs: Integration, Querying, and Analytics. PhD thesis, North Carolina State University, 2022. ΑΑΙ30283568.

Jing Ao、Jon Doyle、Christopher Healey 和 Ranga Raju Vatsavai。《時間知識圖譜：整合、查詢與分析》。博士論文，北卡羅來納州立大學，2022 年。ΑΑΙ30283568。

Antoine Bordes, Nicolas Usunier, Alberto Garcia-Duran, Jason Weston, and Oksana Yakhnenko. Translating embeddings for modeling multi-relational data. Advances in neural information processing systems, 26, 2013.

Antoine Bordes、Nicolas Usunier、Alberto Garcia-Duran、Jason Weston 和 Oksana Yakhnenko。用於模擬多重關係資料的翻譯嵌入。神經資訊處理系統進展，26，2013。

Li Cai, Xin Mao, Yuhao Zhou, Zhaoguang Long, Changxu Wu, and Man Lan. A survey on temporal knowledge graph: Representation learning and applications, 2024. URL https://arxiv.org/abs/2403.04782.

Li Cai、Xin Mao、Yuhao Zhou、Zhaoguang Long、Changxu Wu 和 Man Lan。時間知識圖譜綜述：表示學習與應用，2024。網址 https://arxiv.org/abs/2403.04782。

Kai Chen, Ye Wang, Yitong Li, Aiping Li, Han Yu, and Xin Song. A unified temporal knowledge graph reasoning model towards interpolation and extrapolation. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 117–132, 2024.

Kai Chen、Ye Wang、Yitong Li、Aiping Li、Han Yu 和 Xin Song。一個統一的時間知識圖譜推理模型，用於內插和外推。在第 62 屆計算語言學協會年會論文集（第 1 卷：長論文）中，第 117–132 頁，2024 年。

Mingyang Chen, Wen Zhang, Yushan Zhu, Hongting Zhou, Zonggang Yuan, Changliang Xu, and Huajun Chen. Meta-knowledge transfer for inductive knowledge graph embedding. In Proceedings of the 45th international ACM SIGIR conference on research and development in information retrieval, pages 927–937, 2022.

陳明陽、張文、朱玉山、周鴻婷、袁宗剛、徐昌亮、陳華鈞。元知識轉移用於歸納知識圖譜嵌入。在第45屆ACM SIGIR信息檢索研究與發展國際會議論文集，927-937頁，2022年。

Zhongwu Chen, Chengjin Xu, Fenglong Su, Zhen Huang, and Yong Dou. Incorporating structured sentences with time-enhanced bert for fully-inductive temporal relation prediction. In Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval, pages 889–899, 2023.

陳忠武、徐承錦、蘇豐隆、黃震、竇勇。結合結構化句子與時間增強型BERT進行全歸納式時間關係預測。在第46屆ACM SIGIR資訊檢索研究與發展國際會議論文集，889-899頁，2023年。

Ling Ding, Lei Huang, Zhizhi Yu, Di Jin, and Dongxiao He. Towards global-topology relation graph for inductive knowledge graph completion. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pages 11581–11589, 2025.

丁玲、黃磊、於志之、金迪、何東曉。邁向全域拓撲關係圖以完成歸納知識圖譜。在AAAI人工智慧會議論文集，第39卷，11581–11589頁，2025年。

Zifeng Ding, Heling Cai, Jingpei Wu, Yunpu Ma, Ruotong Liao, Bo Xiong, and Volker Tresp. zrllm: Zero-shot relational learning on temporal knowledge graphs with large language models. In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 1877-1895, 2024.

丁梓峰、蔡和嶺、吳靖培、馬雲普、廖若彤、熊波、Volker Tresp。zrllm：利用大型語言模型在時間知識圖譜上進行零樣本關係學習。在2024年北美計算語言學協會分會會議論文集：人類語言技術（第一卷：長論文），1877-1895頁，2024年。

Mikhail Galkin, Xinyu Yuan, Hesham Mostafa, Jian Tang, and Zhaocheng Zhu. Towards foundation models for knowledge graph reasoning. In The Twelfth International Conference on Learning Representations, 2024.

Mikhail Galkin、袁新宇、Hesham Mostafa、唐健和朱兆澄。邁向知識圖譜推理的基礎模型。在第十二屆國際學習表徵會議，2024年。

Alberto García-Durán, Sebastijan Dumančić, and Mathias Niepert. Learning sequence encoders for temporal knowledge graph completion. In Ellen Riloff, David Chiang, Julia Hockenmaier, and Jun'ichi Tsujii, editors, Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 4816–4821, Brussels, Belgium, October-November 2018. Association for Computational Linguistics. doi: 10.18653/v1/D18-1516. URL https://aclanthology.org/D18-1516/.

Alberto García-Durán、Sebastijan Dumančić 和 Mathias Niepert。學習用於時間知識圖譜補全的序列編碼器。在 Ellen Riloff、David Chiang、Julia Hockenmaier 和 Jun'ichi Tsujii 編輯的 2018 年自然語言處理實證方法會議論文集中，第 4816-4821 頁，比利時布魯塞爾，2018 年 10 月至 11 月。計算語言學協會。doi: 10.18653/v1/D18-1516。網址 https://aclanthology.org/D18-1516/。

Julia Gastinger, Timo Sztyler, Lokesh Sharma, Anett Schuelke, and Heiner Stuckenschmidt. Comparing apples and oranges? on the evaluation of methods for temporal knowledge graph forecasting. In Joint European conference on machine learning and knowledge discovery in databases, pages 533-549. Springer, 2023.

Julia Gastinger、Timo Sztyler、Lokesh Sharma、Anett Schuelke 和 Heiner Stuckenschmidt。比較蘋果和橘子？關於時間知識圖譜預測方法的評估。在機器學習和資料庫知識發現聯合歐洲會議上，第 533-549 頁。施普林格，2023 年。

Rishab Goel, Seyed Mehran Kazemi, Marcus Brubaker, and Pascal Poupart. Diachronic embedding for temporal knowledge graph completion. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 34, pages 3988–3995, 2020.

Rishab Goel, Seyed Mehran Kazemi, Marcus Brubaker, and Pascal Poupart. Diachronic embedding for temporal knowledge graph completion. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 34, pages 3988–3995, 2020.

Zhen Han, Peng Chen, Yunpu Ma, and Volker Tresp. Explainable subgraph reasoning for forecasting on temporal knowledge graphs. In International conference on learning representations, 2020.

韓真、陳鵬、馬雲普與Volker Tresp。可解釋的子圖推理用於時間知識圖譜預測。於國際學習表徵會議，2020年。

Zhen Han, Ruotong Liao, Jindong Gu, Yao Zhang, Zifeng Ding, Yujia Gu, Heinz Koeppl, Hinrich Schütze, and Volker Tresp. Ecola: Enhancing temporal knowledge embeddings with contextualized language representations. In Findings of the Association for Computational Linguistics: ACL 2023, pages 5433-5447, 2023.

韓真、廖若彤、顧金東、張瑤、丁梓峰、顧宇嘉、Heinz Koeppl、Hinrich Schütze、Volker Tresp。Ecola：利用情境化語言表示增強時間知識嵌入。計算語言學協會論文集：ACL 2023，第 5433-5447 頁，2023 年。

Woojeong Jin, Meng Qu, Xisen Jin, and Xiang Ren. Recurrent event network: Autoregressive structure inferenceover temporal knowledge graphs. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 6669-6683, 2020.

金宇正、曲萌、金希森、任翔。循環事件網路：時間知識圖譜上的自回歸結構推斷。在 2020 年自然語言處理實證方法會議 (EMNLP) 論文集中，第 6669-6683 頁，2020 年。

Timothée Lacroix, Guillaume Obozinski, and Nicolas Usunier. Tensor decompositions for temporal knowledge base completion. In International Conference on Learning Representations, 2020. URL https://openreview.net/forum?id=rke2P1BFwS.

Timothée Lacroix、Guillaume Obozinski 與 Nicolas Usunier。用於時間知識庫補全的張量分解。載於國際學習表徵會議，2020年。網址：https://openreview.net/forum?id=rke2P1BFwS。

Jennifer Lautenschlager, Steve Shellman, and Michael Ward. Icews event aggregations. Harvard Dataverse, 3(595):28, 2015.

Jennifer Lautenschlager、Steve Shellman 和 Michael Ward。Icews 事件聚合。哈佛數據宇宙，3(595):28，2015。

Julien Leblay and Melisachew Wudage Chekol. Deriving validity time in knowledge graph. In Companion proceedings of the the web conference 2018, pages 1771–1776, 2018.

Julien Leblay 和 Melisachew Wudage Chekol。推導知識圖譜中的有效時間。在 2018 年網路會議的配套論文集中，第 1771-1776 頁，2018 年。

Dong-Ho Lee, Kian Ahrabian, Woojeong Jin, Fred Morstatter, and Jay Pujara. Temporal knowledge graph forecasting without knowledge using in-context learning. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 544-557, 2023a.

李東浩、Kian Ahrabian、金宇正、Fred Morstatter 和 Jay Pujara。使用情境學習進行無知識的時間知識圖譜預測。在 2023 年自然語言處理實證方法會議論文集中，第 544-557 頁，2023a。

Jaejun Lee, Chanyoung Chung, and Joyce Jiyoung Whang. Ingram: Inductive knowledge graph embedding via relation graphs. In International Conference on Machine Learning, pages 18796–18809. PMLR, 2023b.

李在俊、鄭燦英、黃智英。Ingram：透過關係圖進行歸納知識圖譜嵌入。於國際機器學習會議，頁18796–18809。PMLR，2023b。

Yujia Li, Shiliang Sun, and Jing Zhao. Tirgn: Time-guided recurrent graph network with local-global historical patterns for temporal knowledge graph reasoning. In IJCAI, pages 2152–2158, 2022a.

李宇嘉、孫世亮、趙晶。Tirgn：基於時間引導的遞歸圖網路，結合局部-全域歷史模式進行時間知識圖譜推理。IJCAI，頁 2152-2158，2022a。

Zixuan Li, Saiping Guan, Xiaolong Jin, Weihua Peng, Yajuan Lyu, Yong Zhu, Long Bai, Wei Li, Jiafeng Guo, and Xueqi Cheng. Complex evolutional pattern learning for temporal knowledge graph reasoning. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), pages 290–296, 2022b.

李姿萱、關賽平、金小龍、彭偉華、呂雅娟、朱勇、白龍、李威、郭嘉豐、程學旗。複雜演化模式學習於時間知識圖譜推理。第60屆計算語言學協會年會論文集（第二卷：短篇論文），290-296頁，2022b。

Ke Liang, Lingyuan Meng, Meng Liu, Yue Liu, Wenxuan Tu, Siwei Wang, Sihang Zhou, and Xinwang Liu. Learn from relational correlations and periodic events for temporal knowledge graph reasoning. In Proceedings of the 46th international ACM SIGIR conference on research and development in information retrieval, pages 1559–1568, 2023.

梁科、孟令元、劉猛、劉悅、屠文軒、王思危、周思航、劉新旺。從關係相關性和週期性事件中學習時間知識圖譜推理。在第46屆ACM SIGIR信息檢索研究與發展國際會議論文集，1559-1568頁，2023年。

Ruotong Liao, Xu Jia, Yangzhe Li, Yunpu Ma, and Volker Tresp. Gentkg: Generative forecasting on temporal knowledge graph with large language models. In NAACL-HLT (Findings), 2024.

廖若彤、賈旭、李揚哲、馬雲普、Volker Tresp。Gentkg：基於大型語言模型的時間知識圖譜生成式預測。NAACL-HLT (Findings)，2024。

Shuwen Liu, Bernardo Grau, Ian Horrocks, and Egor Kostylev. Indigo: Gnn-based inductive knowledge graph completion using pair-wise encoding. Advances in Neural Information Processing Systems, 34:2034-2045, 2021.

劉書文、Bernardo Grau、Ian Horrocks 和 Egor Kostylev。Indigo：基於 GNN 的歸納知識圖譜補全，使用成對編碼。神經資訊處理系統進展，34:2034-2045，2021。

Yushan Liu, Yunpu Ma, Marcel Hildebrandt, Mitchell Joblin, and Volker Tresp. Tlogic: Temporal logical rules for explainable link forecasting on temporal knowledge graphs. In Proceedings of the AAAI conference on artificial intelligence, volume 36, pages 4120–4127, 2022.

劉玉山、馬雲普、Marcel Hildebrandt、Mitchell Joblin 和 Volker Tresp。Tlogic：用於時間知識圖譜上可解釋連結預測的時間邏輯規則。在 AAAI 人工智慧會議論文集，第 36 卷，第 4120–4127 頁，2022 年。

Jiaxin Pan, Mojtaba Nayyeri, Yinan Li, and Steffen Staab. Hge: embedding temporal knowledge graphs in a product space of heterogeneous geometric subspaces. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pages 8913–8920, 2024.

潘嘉欣、Mojtaba Nayyeri、李一男和 Steffen Staab。Hge：將時間知識圖譜嵌入到異構幾何子空間的乘積空間中。在 AAAI 人工智慧會議論文集，第 38 卷，第 8913-8920 頁，2024 年。

Apoorv Saxena, Soumen Chakrabarti, and Partha Talukdar. Question answering over temporal knowledge graphs. In Chengqing Zong, Fei Xia, Wenjie Li, and Roberto Navigli, editors, Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 6663–6676, Online, August 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021.acl-long.520. URL https://aclanthology.org/2021.acl-long.520/.

Apoorv Saxena、Soumen Chakrabarti 和 Partha Talukdar。時間知識圖譜上的問答。在宗成慶、夏鯡、李文捷和 Roberto Navigli 編輯的第 59 屆計算語言學協會年會暨第 11 屆自然語言處理國際聯合會議論文集（第一卷：長論文）中，第 6663-6676 頁，線上，2021 年 8 月。計算語言學協會。doi: 10.18653/v1/2021.acl-long.520。網址 https://aclanthology.org/2021.acl-long.520/。

Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063, 2024.

蘇建林、Murtadha Ahmed、陸羽、潘勝豐、薄文、劉雲峰。Roformer：具有旋轉位置嵌入的增強型 transformer。Neurocomputing，568:127063，2024。

Haohai Sun, Jialun Zhong, Yunpu Ma, Zhen Han, and Kun He. Timetraveler: Reinforcement learning for temporal knowledge graph forecasting. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 8306-8319, 2021.

孫浩海、鍾家倫、馬雲普、韓真、何昆。《時間旅行者：用於時間知識圖譜預測的強化學習》。在《2021年自然語言處理實證方法會議論文集》，8306-8319頁，2021年。

Li Sun, Zhenhao Huang, Suyang Zhou, Qiqi Wan, Hao Peng, and Philip S Yu. Riemanngfm: Learning a graph foundation model from structural geometry. In THE WEB CONFERENCE 2025, 2025.

孫力、黃振浩、周蘇陽、萬琪琪、彭浩、余世維。Riemanngfm：從結構幾何中學習圖基礎模型。2025年網路會議，2025年。

Komal Teru, Etienne Denis, and Will Hamilton. Inductive relation prediction by subgraph reasoning. In International conference on machine learning, pages 9448–9457. PMLR, 2020.

Komal Teru、Etienne Denis 和 Will Hamilton。基於子圖推理的歸納關係預測。國際機器學習會議，9448-9457頁。PMLR，2020。

Rakshit Trivedi, Hanjun Dai, Yichen Wang, and Le Song. Know-evolve: Deep temporal reasoning for dynamic knowledge graphs. In international conference on machine learning, pages 3462–3471. PMLR, 2017.

Rakshit Trivedi、戴漢君、王一辰、宋樂。知演化：動態知識圖譜的深度時間推理。國際機器學習會議，3462-3471頁。PMLR，2017。

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.

Ashish Vaswani、Noam Shazeer、Niki Parmar、Jakob Uszkoreit、Llion Jones、Aidan N Gomez、Łukasz Kaiser 和 Illia Polosukhin。注意力就是你所需要的一切。神經信息處理系統的進展，30，2017。

Zehong Wang, Zheyuan Zhang, Nitesh Chawla, Chuxu Zhang, and Yanfang Ye. Gft: Graph foundation model with transferable tree vocabulary. Advances in Neural Information Processing Systems, 37: 107403-107443, 2024.

王澤宏、張哲源、Nitesh Chawla、張楚楚、葉燕芳。Gft：具有可轉移樹詞彙的圖形基礎模型。神經信息處理系統進展，37：107403-107443，2024。

Chengjin Xu, Mojtaba Nayyeri, Fouad Alkhoury, Hamed Shariat Yazdi, and Jens Lehmann. Tero: A time-aware knowledge graph embedding via temporal rotation. In Proceedings of the 28th International Conference on Computational Linguistics, pages 1583–1593, 2020.

許承錦、Mojtaba Nayyeri、Fouad Alkhoury、Hamed Shariat Yazdi 和 Jens Lehmann。Tero：一種透過時間旋轉實現的時間感知知識圖譜嵌入。在第 28 屆國際計算語言學會議論文集中，第 1583-1593 頁，2020 年。

Wenjie Xu, Ben Liu, Miao Peng, Xu Jia, and Min Peng. Pre-trained language model with prompts for temporal knowledge graph completion. In Findings of the Association for Computational Linguistics: ACL 2023, pages 7790–7803, 2023.

徐文傑、劉犇、彭淼、賈旭、彭敏。帶有提示的預訓練語言模型用於時間知識圖譜補全。計算語言學協會發現：ACL 2023，7790-7803頁，2023年。

Bishan Yang, Scott Wen-tau Yih, Xiaodong He, Jianfeng Gao, and Li Deng. Embedding entities and relations for learning and inference in knowledge bases. In Proceedings of the International Conference on Learning Representations (ICLR) 2015, 2015.

楊必珊、易文韜、何曉冬、高建峰、鄧力。知識庫中實體與關係的嵌入以利學習與推論。於國際學習表徵會議（ICLR）論文集，2015年。

Fuwei Zhang, Zhao Zhang, Xiang Ao, Fuzhen Zhuang, Yongjun Xu, and Qing He. Along the time: Timeline-traced embedding for temporal knowledge graph completion. In Proceedings of the 31st ACM International Conference on Information & Knowledge Management, pages 2529–2538, 2022.

張福偉、張釗、敖翔、莊福臻、徐勇軍、何清。沿著時間：時間線追溯嵌入用於時間知識圖譜補全。在第31屆ACM信息與知識管理國際會議論文集，2529-2538頁，2022年。

Zhaocheng Zhu, Zuobai Zhang, Louis-Pascal Xhonneux, and Jian Tang. Neural bellman-ford networks: A general graph neural network framework for link prediction. Advances in neural information processing systems, 34:29476–29490, 2021.

朱兆澄、張作柏、Louis-Pascal Xhonneux 和唐健。神經貝爾曼-福特網路：一個用於連結預測的通用圖神經網路框架。神經資訊處理系統進展，34:29476-29490，2021。

A Parameter Count and Complexity

參數數量與複雜度

From Table 5, we observe that the number of parameters in POSTRA is agnostic to the dataset size. Since POSTRA does not initialize embeddings based on |V|, |R|, or |T|, but instead relies on a fixed number of relation interactions |H|, its parameter count is only related to the embedding dimension d. This property makes POSTRA especially suitable for transfer learning scenarios.

從表 5 中，我們觀察到 POSTRA 的參數數量與資料集大小無關。由於 POSTRA 不根據 |V|、|R| 或 |T| 初始化嵌入，而是依賴固定數量的關係互動 |H|，其參數數量僅與嵌入維度 d 相關。此特性使 POSTRA 特別適用於遷移學習場景。

The time complexity of POSTRA is primarily determined by the quadruple encoder, as the number of relations |R| is significantly smaller than the number of entities |V|, allowing us to omit the complexity contribution of the relation encoder. Furthermore, since the local entity graph is much smaller than the global entity graph, the computational upper bound is dominated by the global quadruple representation. Utilizing NBFNet [Zhu et al., 2021] as the encoder, the time complexity for a single layer is O(|Q|d+ |V|d²). For A layers, the total time complexity becomes O(A(|Q|d+|V|d²)).

POSTRA 的時間複雜度主要由四元編碼器決定，因為關係數量 |R| 遠小於實體數量 |V|，使我們能夠忽略關係編碼器的複雜度貢獻。此外，由於局部實體圖遠小於全域實體圖，計算上界由全域四元表示主導。使用 NBFNet [Zhu et al., 2021] 作為編碼器，單層的時間複雜度為 O(|Q|d + |V|d²)。對於 A 層，總時間複雜度變為 O(A(|Q|d + |V|d²))。

The memory complexity of POSTRA is also linear in the number of edges, expressed as O(A|Q|d), as the quadruple encoder maintains and updates representations for each edge in the temporal knowledge graph.

POSTRA 的記憶體複雜度也與邊的數量呈線性關係，表示為 O(A|Q|d)，因為四元編碼器會維護並更新時間知識圖譜中每條邊的表示。

[Image]

B Pre-training Results

預訓練結果

Table 7 presents the pretrained performance of POSTRA. It achieves comparable performance with TComplEx while using significantly fewer parameters especially on datasets which involve a large number of entities and relations. POSTRA generates adaptive representations dynamically through a conditional message passing mechanism. This design enables effective generalization with a compact model size, making it highly scalable and suitable for large temporal knowledge graphs.

表 7 呈現了 POSTRA 的預訓練性能。它在 TComplEx 上取得了相當的性能，同時使用了明顯更少的參數，尤其是在涉及大量實體和關係的數據集上。POSTRA 通過條件訊息傳遞機制動態生成自適應表示。這種設計使得模型能夠以緊湊的模型大小實現有效的泛化，使其具有高度的可擴展性，並適用於大型時間知識圖譜。

[Image]

C Limitations

雖然 POSTRA 在零樣本情境中展現出強大的全歸納推理性能，但仍存在一些限制。首先，在較大的圖上訓練不一定能帶來更好的性能。我們假設不同大小的資料集之間時間模式的變化可能會影響結果。其次，對於密集的時間知識圖譜，POSTRA 的計算量可能很大，因為其空間複雜度與事件數量成正比。第三，POSTRA 無法執行時間預測。我們旨在探索更高效的架構和自適應的訓練策略，以應對未來工作中的這些挑戰。

Although POSTRA demonstrates strong fully-inductive inference performance in zero-shot scenarios, several limitations remain. First, training on larger graphs does not always lead to improved performance. We hypothesize that variations in temporal patterns across datasets of different sizes may influence results. Second, POSTRA can be computationally intensive for dense temporal knowledge graphs, as its space complexity scales with the number of events. Third, POSTRA is unable to perform time prediction. We aim to explore more efficient architectures and adaptive training strategies to address these challenges in future work.

儘管 POSTRA 在零樣本情境中展現出強大的完全歸納推理性能，但仍存在一些限制。首先，在較大的圖上訓練不一定能帶來更好的性能。我們假設不同大小的數據集之間的時間模式變化可能會影響結果。其次，對於密集的時間知識圖譜，POSTRA 的計算量可能很大，因為其空間複雜度與事件數量成正比。第三，POSTRA 無法執行時間預測。我們旨在探索更高效的架構和自適應的訓練策略，以應對未來工作中的這些挑戰。

D Ethics Statement

D 倫理聲明

POSTRA can be applied to a wide range of tasks and datasets beyond their originally intended scope. Given the sensitive and time-dependent nature of temporal knowledge graphs (TKGs), there is a risk that such models could be used to infer patterns from historical or anonymized temporal data in ways that may compromise privacy or be exploited for surveillance, misinformation, or manipulation. On the positive side, fully-inductive models for TKGs offer significant benefits by enabling zero-shot transfer across domains, datasets, and temporal granularities. This reduces the need to train separate models for each new scenario, lowering the computational cost for training repeated models.

POSTRA 可應用於超出其原始設計範圍的各種任務和資料集。鑑於時間知識圖譜（TKGs）的敏感性和時間依賴性，存在這樣的風險：此類模型可能被用於從歷史或匿名時間資料中推斷模式，從而可能損害隱私，或被用於監視、散佈假訊息或操縱。從積極的方面來看，用於 TKGs 的完全歸納模型通過實現跨領域、資料集和時間粒度的零樣本遷移，提供了顯著的好處。這減少了為每個新場景訓練單獨模型的需要，從而降低了重複訓練模型的計算成本。

E Relative Representation Transfer

E 相對表示轉移

[Image]

F Hyperparameter and Training Details

F 超參數與訓練細節

Our experiments were conducted on 4 NVIDIA A100 GPUs with 48GB of RAM. We set the maximal training epoch as 10 and negative samples as 512. We use a batch size of 16, 2, and 1 for training ICEWS14, ICEWS05-15 and GDELT respectively. We set the dimension size d as 64. We employed the AdamW optimizer and set the learning rate as 0.0005. More details can be seen in Table 8. For the baseline methods, we utilized the settings in their original papers.

我們的實驗在 4 個配備 48GB RAM 的 NVIDIA A100 GPU 上進行。我們將最大訓練週期設定為 10，負樣本數為 512。我們分別使用 16、2 和 1 的批次大小來訓練 ICEWS14、ICEWS05-15 和 GDELT。我們將維度大小 d 設定為 64。我們採用 AdamW 優化器，並將學習率設定為 0.0005。更多細節可見表 8。對於基準方法，我們使用了其原始論文中的設定。

[Image]

G Hyperparameter Analysis

G 超參數分析

We conduct a sensitivity analysis of three key hyperparameters, k, a, and ẞ on model performance by pre-training on the ICEWS14 dataset. Figure 5 presents the MRR and H@10 results under different values of each hyperparameter while keeping all others fixed at their optimal settings.

我們透過在 ICEWS14 資料集上進行預訓練，對三個關鍵超參數 k、α 和 β 進行了模型性能的敏感度分析。圖 5 呈現了在每個超參數的不同值下，同時將所有其他超參數固定在其最佳設定時的 MRR 和 H@10 結果。

Effect of k. As shown in Figure 5(a), increasing k, which controls the length of the local entity graph, leads to a gradual decline in both MRR and H@10. As ICEWS14 is a news dataset and contains mainly short-term temporal relations, deeper propagation may cause oversmoothing and decrease the model's performance.

k 的影響。如圖 5(a) 所示，增加 k（控制局部實體圖的長度）會導致 MRR 和 H@10 逐漸下降。由於 ICEWS14 是一個新聞資料集，主要包含短期時間關係，更深的傳播可能會導致過度平滑並降低模型性能。

Effect of a. Figure 5(b) shows that performance is highly sensitive to the choice of a, which balances the contribution of local and global temporal contexts. Both MRR and H@10 significantly improve as a increases from 0 to 0.5, indicating the benefit of incorporating both types of temporal information. However, performance degrades when a approaches 1, implying that overemphasis on either context harms generalization. The results highlight the importance of maintaining a moderate balance between local and global temporal patterns.

α 的影響。圖 5(b) 顯示，性能對 α 的選擇高度敏感，α 平衡了局部和全局時間上下文的貢獻。當 α 從 0 增加到 0.5 時，MRR 和 H@10 均顯著改善，表明納入兩種時間資訊的好處。然而，當 α 接近 1 時，性能會下降，這意味著過分強調任一上下文都會損害泛化能力。結果突顯了在局部和全局時間模式之間保持適度平衡的重要性。

Effect of ẞ. As illustrated in Figure 5(c), the model is highly robust to the choice of ẞ, which controls the sinusoidal frequency of temporal embeddings. Across a wide range of values from 10² to 10⁶-MRR and H@10 remain largely stable, with negligible performance fluctuations.

ß 的影響。如圖 5(c) 所示，模型對 ß 的選擇非常穩健，ß 控制著時間嵌入的正弦頻率。在 10² 到 10⁶ 的廣泛數值範圍內，MRR 和 H@10 保持高度穩定，性能波動可忽略不計。

The hyperparameter analysis reveals that a is the most critical parameter and requires careful tuning to balance local and global temporal information. The choice of k should be dataset-dependent; for datasets characterized by short-term temporal patterns, smaller values of k are preferable. In contrast, the model is relative insensitive to ẞ, so we set it as 10,000 across all datasets.

超參數分析顯示，α 是最關鍵的參數，需要仔細調整以平衡局部和全局時間資訊。k 的選擇應視資料集而定；對於以短期時間模式為特徵的資料集，較小的 k 值更可取。相反，模型對 β 相對不敏感，因此我們在所有資料集中將其設定為 10,000。

[Image]

H Case Study

為了直觀地展示 POSTRA 在生成緊湊且可區分嵌入方面的能力，我們採用 PCA 來視覺化五個代表性四元組的嵌入分佈：

To intuitively demonstrate POSTRA's capability in generating compact and distinguishable embeddings, we employ PCA to visualize the distribution of embeddings for five representative quadruples:

為了直觀地展示 POSTRA 在生成緊湊且可區分嵌入方面的能力，我們採用 PCA 來視覺化五個代表性四元組的嵌入分佈：

[Image]

I Theoretical Analysis

理論分析

In this section, we provide a theoretical analysis demonstrating the model's ability to transfer across time and effectively capture a variety of periodic patterns.

在本節中，我們提供了一個理論分析，展示了模型跨時間轉移並有效捕捉各種週期性模式的能力。

Temporal Transferability As illustrated in Figure 2 and 4, the temporal difference is the transferable information in TKGs. Therefore, we expect that such transferability must be followed in the time embedding space. Here, we aim to show that if four time points' indices-T1, T2 in the training graph and τί, τㄥ in the test graph-hold AT = τ2 - τ₁ = ΔT' = τή – τ', then we expect that such transferability is preserved via sinusoidal positional encoding in the form of Euclidean distance ||TE(τ2) – TE(τι)|| = ||TE(τή) – TE(τ₁)|| that is governed solely by their time difference AT, rather than the absolute times.

時間可轉移性 如圖 2 和圖 4 所示，時間差是 TKG 中可轉移的資訊。因此，我們期望這種可轉移性必須在時間嵌入空間中得到遵循。在這裡，我們的目標是證明，如果訓練圖中的四個時間點索引 T1、T2 和測試圖中的 τί、τߓ 滿足 AT = τ2 - τ₁ = ΔT' = τή – τ'，那麼我們期望這種可轉移性通過正弦位置編碼以歐幾里得距離 ||TE(τ2) – TE(τι)|| = ||TE(τή) – TE(τ₁)|| 的形式得以保留，該距離僅由它們的時間差 AT 而非絕對時間決定。

The following theorem states and proves this:

以下定理陳述並證明了這一點：

Theorem 1 (Time-Shift Invariance in Sinusoidal Positional Embeddings). Let d ∈ N be even and fix positive frequencies @, @1,………, @d/2-1 > 0. Define the sinusoidal temporal embedding TE : N → Rd component-wise by
[TE(t)]2n = sin(@nt), [TE(t)]2n+1 = cos(@nt), 0<n< d/2.
For any four time-points indices τι, τη, τί, τη ∈ N set
Δτ := τε - τι, Δ' := τή- τή.
If Ar = Δ' then
||TE(τ2) – TE(τι)|| = ||TE(τή) – TE(τί)||.

定理 1 (正弦位置嵌入中的時移不變性)。設 d ∈ N 為偶數，並固定正頻率 ω, ω1, ..., ωd/2-1 > 0。定義正弦時間嵌入 TE : N → Rd，其分量為
[TE(t)]2n = sin(ωnt), [TE(t)]2n+1 = cos(ωnt), 0 < n < d/2。
對於任意四個時間點索引 τι, τη, τί, τη ∈ N，設
Δτ := τε - τι, Δ' := τή- τή。
如果 Ar = Δ'，則
||TE(τ2) – TE(τι)|| = ||TE(τή) – TE(τί)||。

That is, the Euclidean distance between two sinusoidal embeddings depends only on the time difference, not on the absolute timestamps.

也就是說，兩個正弦嵌入之間的歐幾里得距離僅取決於時間差，而非絕對時間戳。

Step-by-step proof.

逐步證明。

Fix an index n ∈ {0,..., d/2 - 1} and two times t1, t2 ∈ N. Denote
A = t2-t1, Σ = t1 + t2.

固定索引 n ∈ {0,..., d/2 - 1} 和兩個時間 t1, t2 ∈ N。表示
A = t2-t1, Σ = t1 + t2。

Using the standard sum-and-difference identities,
sin(@nt2) – sin(@nt₁) = 2 sin(@nΔ/2) cos(@nΣ/2),
cos(@nt2) - cos(@nt1) = -2 sin(@nΔ/2) sin(@nΣ/2).

使用標準的和差恆等式，
sin(ωnt₂) – sin(ωnt₁) = 2 sin(ωnΔ/2) cos(ωnΣ/2),
cos(ωnt₂) - cos(ωnt₁) = -2 sin(ωnΔ/2) sin(ωnΣ/2).

Thus, the squared contribution of this frequency to the Euclidean distance is
[sin(@nt2) - sin(@nt1)]² + [cos(@nt2) - cos(@nt1)]² = 4 sin²(@nΔ/2),
because cos² + sin² = 1 removes any dependence on Σ.

因此，此頻率對歐幾里德距離的平方貢獻為
[sin(ωnt₂) - sin(ωnt₁)]² + [cos(ωnt₂) - cos(ωnt₁)]² = 4 sin²(ωnΔ/2),
因為 cos² + sin² = 1 消除了對 Σ 的任何依賴。

Summing over all n gives
||TE(t2) – TE(t1)||² = ∑(n=0 to d/2-1) 4 sin²(ωnΔ/2) = 4 ∑(n=0 to d/2-1) sin²(ωnΔ/2),
which is a function of ∆ = t2 - t₁ alone. Consequently, if τ2 – τ₁ = τή – τ' then Δτ = Δτ', and
||TE(τ2) - TE(τ1) ||² = 4∑(n=0 to d/2-1) sin²(ωnΔτ/2) = 4∑(n=0 to d/2-1) sin(ωnΔτ'/2) = ||TE(τή) – TE(τί) ||².

對所有 n 求和可得
||TE(t₂) – TE(t₁)||² = Σ(n=0 到 d/2-1) 4 sin²(ωnΔ/2) = 4 Σ(n=0 到 d/2-1) sin²(ωnΔ/2)，
此為僅與 Δ = t₂ - t₁ 相關的函數。因此，若 τ₂ – τ₁ = τ₂' – τ₁'，則 Δτ = Δτ'，且
||TE(τ₂) - TE(τ₁)||² = 4Σ(n=0 到 d/2-1) sin²(ωnΔτ/2) = 4Σ(n=0 到 d/2-1) sin(ωnΔτ'/2) = ||TE(τ₂') – TE(τ₁')||²。

Taking square roots results in the claimed equality of norms.

取平方根即可得所宣稱的範數等式。

Capturing Diverse Periodicities Temporal facts expressed as (s, p,o, t), where τ∈ T represents the time annotation, can capture periodic phenomena, such as the Olympic Games occurring every four years or annual events like the Nobel Prize ceremonies. Modeling diverse periodic behaviors is crucial in the zero-shot transfer learning, as it enables systems to accurately represent and analyze the dynamic evolution of diverse relationships and events. In this section, we prove that POSTRA is capable of capturing periodic (with different frequencies) temporal facts. In the following theorem, we prove that the scorer in Equation 6 can capture and express various frequencies even if we assume there fe is simply a linear function (with m linear nodes for representing m different frequencies). For simplicity, we loosely use as the timestamp index i in ti in the following.

捕捉多樣化的週期性 以 (s, p, o, t) 表示的時間事實，其中 τ ∈ T 代表時間註釋，可以捕捉週期性現象，例如每四年舉辦一次的奧運會或像諾貝爾獎頒獎典禮這樣的年度事件。在零樣本遷移學習中，模擬多樣化的週期性行為至關重要，因為它使系統能夠準確地表示和分析不同關係和事件的動態演變。在本節中，我們證明 POSTRA 能夠捕捉週期性（具有不同頻率）的時間事實。在下面的定理中，我們證明即使我們假設 fe 是一個簡單的線性函數（具有 m 個線性節點來表示 m 個不同的頻率），方程式 6 中的評分器也可以捕捉和表達各種頻率。為簡單起見，我們在下面將時間戳索引 i 寬鬆地用作 ti 中的 i。

Theorem 2 (Multi-Frequency Affine Scorer). Let
• T > 1 be an integer time horizon ;
• dpE > T be an even positional-encoding dimension;
• P1,..., Pm, such that Pi | T for every i = 1, ..., m, is a family of positive integers.
• [TE(T)]2n = sin(ωπτ), [TE(τ)]2n+1 = cos(@nt), 0 < n < dPE/2, with arbitrary real frequencies ω,..., ωdpe/2−1, be (standard) sinusoidal positional encoding defined for any τ.
• G = {V1,...,Vm} CRdv be m fixed context vectors and,
• gi: {0,1,..., T – 1} → R be a non-constant sequence for each i = 1,...,m, that we extend to all Z by Pi-periodicity:
gi(t) := gi(t mod P₁), τΕΖ.

定理 2 (多頻率仿射評分器)。令
• T > 1 為一整數時間範圍；
• dpE > T 為一偶數位置編碼維度；
• P1,..., Pm，使得對於每個 i = 1, ..., m，Pi | T，為一正整數族。
• [TE(τ)]2n = sin(ωnτ), [TE(τ)]2n+1 = cos(ωnτ), 0 < n < dPE/2，其中任意實頻率 ω,..., ωdpe/2−1，為對任意 τ 定義的（標準）正弦位置編碼。
• G = {V1,...,Vm} ⊂ R^dv 為 m 個固定上下文向量，且
• 對於每個 i = 1,...,m，gi: {0,1,..., T – 1} → R 為一非恆定序列，我們透過 Pi-週期性將其擴展至所有 Z：
gi(t) := gi(t mod Pi), τ ∈ Z。

Define

定義

• and
M = [TE(0)^T; TE(1)^T; ...; TE(T-1)^T] ∈ R^(T x dPE)
B(i) = [TE(Pi)^T - TE(0)^T; TE(Pi+1)^T - TE(1)^T; ...; TE(T-1+Pi)^T - TE(T-1)^T] ∈ R^(T x dPE)
be encoding matrices, we stack B(i) block-diagonally:
B := diag(B(1),...,B(m)) ∈ R^(mT x mdPE).

• 並且
M = [TE(0)ᵀ; TE(1)ᵀ; ...; TE(T-1)ᵀ] ∈ R^(T x dPE)
B(i) = [TE(Pᵢ)ᵀ - TE(0)ᵀ; TE(Pᵢ+1)ᵀ - TE(1)ᵀ; ...; TE(T-1+Pᵢ)ᵀ - TE(T-1)ᵀ] ∈ R^(T x dPE)
為編碼矩陣，我們將 B(i) 堆疊成塊對角形式：
B := diag(B(¹), ..., B(ᵐ)) ∈ R^(mT x mdPE)。

g := [g1; ...; gm] ∈ R^(mT)
w_PE := [w_PE^(1); ...; w_PE^(m)] ∈ R^(mdPE)

g := [g₁; ...; gₘ] ∈ R^(mT)
w_PE := [w_PE⁽¹⁾; ...; w_PE⁽ᵐ⁾] ∈ R^(mdPE)

(A) Compatibility (C). There exists wpe ∈ Rmdpe solving the linear system
[Im ⊗ M; B] wpe = [g; 0] (C)
where Im ⊗ M is the block-diagonal Kronecker product whose i-th diagonal block equals M.

(A) 相容性 (C)。存在 wpe ∈ Rmdpe 使得線性系統
[Im ⊗ M; B] wpe = [g; 0] (C)
成立，其中 Im ⊗ M 是第 i 個對角塊等於 M 的塊對角克羅內克積。

(B) Existence of a scorer (S). There exist parameters
θ = (Wpe, wv, b) ∈ Rdpe×m × Rdv × R
such that the affine map
fθ (V, TE(τ)) := V^T wv + Wpe^T TE(τ) + b ∈ Rm
obeys, for each i = 1, ..., m,
(i) Pi-periodicity in τ ∈ Z;
(ii) non-constancy as a function of τ;
(iii) interpolation on the basic window:
[fθ(Vi, TE(τ))]i = wv^T Vi+gi(τ), 0 <τ< Τ.
Given C and S, the following holds
C <=> S
When these equivalent conditions hold one may choose any solution wpe of (C), set
WPE:= [wpe^(1) | ... | wpe^(m)], b = 0,
and pick an arbitrary wv; the resulting θ satisfies S.

(B) 評分器 (S) 的存在性。存在參數
θ = (Wpe, wv, b) ∈ Rdpe×m × Rdv × R
使得仿射對應
fθ (V, TE(τ)) := Vᵀ wv + Wpeᵀ TE(τ) + b ∈ Rm
對於每個 i = 1, ..., m，滿足
(i) τ ∈ Z 的 Pi-週期性；
(ii) 作為 τ 的函數的非恆定性；
(iii) 在基礎窗口上的插值：
[fθ(Vi, TE(τ))]i = wvᵀ Vi + gi(τ), 0 < τ < T。
給定 C 和 S，以下成立
C <=> S
當這些等價條件成立時，可以選擇 (C) 的任何解 wpe，設定
WPE := [wpe⁽¹⁾ | ... | wpe⁽ᵐ⁾], b = 0，
並選擇任意 wv；所得的 θ 滿足 S。

Step-by-step proof of C ⇔ S. We write WPE = [wpe(1)| ... |wpe(m)] with each block wpe(i) ∈ RdPE.

逐步證明 C ⇔ S。我們將 WPE 寫成 [wpe(1)| ... |wpe(m)]，其中每個區塊 wpe(i) ∈ R^(dPE)。

Part I: C ⇒ S. Assume a vector wpe satisfies system (C). Here, we show that if the compatibility condition holds, then a scorer exists. We divide the proof into four parts as follows:

第一部分：C ⇒ S。假設一個向量 wpe 滿足系統 (C)。這裡，我們證明如果相容性條件成立，則存在一個評分器。我們將證明分為以下四個部分：

Step 1: The top block (Im⊗M)w_PE = g is equivalent to
M w_PE^(i) = g_i|_(0<τ<T), i=1,...,m. (1)

步驟 1：頂部區塊 (Im ⊗ M)w_PE = g 等價於
M w_PE^(i) = g_i|_(0<τ<T)，i=1,...,m。(1)

Step 2: The block Bw_PE = 0 gives for every i
B^(i) w_PE^(i) = 0 ⇒ w_PE^(i)T (TE(τ+Pi) - TE(τ)) = 0, 0 ≤ τ < T.
Thus
w_PE^(i)T TE(τ+Pi) = w_PE^(i)T TE(τ), 0 < τ < T.
Replacing τ by τ + Pi and iterating k times shows
w_PE^(i)T TE(τ+kPi) = w_PE^(i)T TE(τ), ∀τ ∈ {0,..., T − 1}, k ∈ Z,
i.e., Pi-periodicity on all integers.

步驟 2：區塊 Bw_PE = 0 對每個 i 給出
B^(i) w_PE^(i) = 0 ⇒ w_PE^(i)T (TE(τ+Pi) - TE(τ)) = 0，0 ≤ τ < T。
因此
w_PE^(i)T TE(τ+Pi) = w_PE^(i)T TE(τ)，0 ≤ τ < T。
將 τ 替換為 τ + Pi 並迭代 k 次可得
w_PE^(i)T TE(τ+kPi) = w_PE^(i)T TE(τ)，∀τ ∈ {0,..., T − 1}，k ∈ Z，
即所有整數上的 Pi 週期性。

Step 3: Combine (1) with periodicity: for every τ∈ Z
w_PE^(i)T TE(τ) = gi(τ mod Pi). (3)

步驟 3：結合 (1) 與週期性：對於每個 τ ∈ Z
w_PE^(i)T TE(τ) = gi(τ mod Pi)。(3)

Step 4: Define
W_PE := [w_PE^(1)| ... |w_PE^(m)], b := 0, choose any w_v ∈ R^dv.
Then
f_θ(V, TE(τ)) = w_v^T V + [w_PE^(1)T TE(τ); ...; w_PE^(m)T TE(τ)],
whose i-th component equals the right-hand side of (3). Therefore:
a) it is Pi-periodic by construction; b) it is non-constant because each gi is non-constant; c) on 0 < τ < T, (3) reduces to wvT Vi + gi(τ).
Hence S holds.

步驟 4：定義
W_PE := [w_PE^(1)| ... |w_PE^(m)]，b := 0，選擇任意 w_v ∈ R^dv。
然後
f_θ(V, TE(τ)) = w_v^T V + [w_PE^(1)T TE(τ); ...; w_PE^(m)T TE(τ)]，
其第 i 個分量等於 (3) 的右側。因此：
a) 根據建構，它是 Pi-週期的；b) 因為每個 gi 都不是常數，所以它也不是常數；c) 在 0 < τ < T 時，(3) 化簡為 wv^T Vi + gi(τ)。
因此 S 成立。

Part II: S ⇒ C.
Conversely, suppose parameters θ = (WPE, wv,b) satisfy S. Write the columns as WPE = [wpe(1)| ... |wpe(m)].

第二部分：S ⇒ C。
反之，假設參數 θ = (WPE, wv, b) 滿足 S。將列寫為 WPE = [wpe(1)| ... |wpe(m)]。

Step 1: Evaluating property (iii) at 0 < τ < T gives
M w_PE^(i) = g_i|_(0<τ<T), i=1,...,m. (4)

步驟 1：在 0 < τ < T 時評估性質 (iii) 可得
M w_PE^(i) = g_i|_(0<τ<T)，i=1,...,m。(4)

Step 2: Pi-periodicity implies the B(i) equations. Property (i) yields, for every integer τ, w_PE^(i)T TE(τ+Pi) = w_PE^(i)T TE(τ). Specialising to τ = 0, ..., T – 1 one obtains B(i) w_PE^(i) = 0.

步驟 2：Pi 週期性意味著 B(i) 方程式。性質 (i) 產生，對於每個整數 τ，w_PE^(i)T TE(τ+Pi) = w_PE^(i)T TE(τ)。將 τ 特化為 0, ..., T – 1，可得 B(i) w_PE^(i) = 0。

Step 3: Collecting (4) for all i and the B(i) equations in a single vector w_PE yields exactly system (C). Thus C holds.

步驟 3：將所有 i 的 (4) 和 B(i) 方程式收集到一個單一向量 w_PE 中，即可得到系統 (C)。因此 C 成立。

Both directions are proven; therefore C ⇔ S. The final comment about choosing (WPE,wv,b) is precisely the construction in Part I, Step 4.

兩個方向都已證明；因此 C ⇔ S。關於選擇 (WPE, wv, b) 的最終評論正是第一部分第四步的建構。

Following Theorem 2, we are interested in knowing under which conditions the compatibility condition holds itself, i.e., when the system has a solution.

根據定理 2，我們有興趣了解在什麼條件下相容性條件本身成立，即系統有解。

Theorem 3 (Universal Compatibility under harmonically-aligned frequencies). Fix
T > 1, dPE ≥ T, P1,...,Pm∈N, Pi|T.
Let L := lcm(P₁,...,Pm)² and choose any collection of integers ko,k1,...,k(dPE/2)-1. Define the frequencies
ωn := 2πkn/L, 0<n<dPE/2,
and build the sinusoidal positional encoding
[TE(τ)]2n = sin(ωnτ), [TE(τ)]2n+1 = cos(ωnτ), τ∈N.

定理 3（諧波對齊頻率下的通用相容性）。固定
T > 1，dPE ≥ T，P1,...,Pm ∈ N，Pi|T。
令 L := lcm(P₁,...,Pm)² 並選擇任意整數集合 ko,k1,...,k(dPE/2)-1。定義頻率
ωn := 2πkn/L，0 < n < dPE/2，
並建構正弦位置編碼
[TE(τ)]2n = sin(ωnτ)，[TE(τ)]2n+1 = cos(ωnτ)，τ ∈ N。

Form the matrices as follows
M = [TE(0)^T; ...; TE(T-1)^T] ∈ R^(TxdPE),
B(i) = [TE(Pi)^T - TE(0)^T; ...; TE(T-1+Pi)^T - TE(T-1)^T] ∈ R^(TxdPE).
For these frequencies, every B(i) vanishes identically; hence, the stacked compatibility system
[Im ⊗ M; diag(B(1),...,B(m))] WPE = [g; 0]
is always solvable, regardless of the choice of any target sequences gi : {0,...,T −1} → R. (C)

構成以下矩陣
M = [TE(0)ᵀ; ...; TE(T-1)ᵀ] ∈ R^(TxdPE),
B(i) = [TE(Pᵢ)ᵀ - TE(0)ᵀ; ...; TE(T-1+Pᵢ)ᵀ - TE(T-1)ᵀ] ∈ R^(TxdPE)。
對於這些頻率，每個 B(i) 都恆等於零；因此，堆疊的相容性系統
[Im ⊗ M; diag(B(¹),...,B(m))] WPE = [g; 0]
總是有解，無論任何目標序列 gi : {0,...,T −1} → R 的選擇如何。(C)

We present the Step-by-step proof as follows. Step 1. Encoding is L-periodic. Because ωnL = 2πkn,
sin(ωn(τ+L)) = sin(ωnτ), cos(ωn(τ+L)) = cos(ωnτ), ∀τ∈ Z,
so TE(τ + L) = TE(τ).

我們提出如下的逐步證明。步驟 1. 編碼是 L-週期的。因為 ωnL = 2πkn，
sin(ωn(τ+L)) = sin(ωnτ)，cos(ωn(τ+L)) = cos(ωnτ)，∀τ∈ Z，
所以 TE(τ + L) = TE(τ)。

Step 2. Encoding is also Pi-periodic. Each Pi divides L, hence TE(τ + Pi) = TE(τ). Therefore
B(i) = 0 ∈ R^(T x dPE), i=1,...,m.

步驟 2. 編碼也是 Pi-週期的。每個 Pi 整除 L，因此 TE(τ + Pi) = TE(τ)。因此
B(i) = 0 ∈ R^(T x dPE)，i=1,...,m。

Step 3. Compatibility degenerates to a single block. With every B(i) equal to zero, system (C) becomes
(Im⊗M) WPE = g. (C')

步驟 3. 相容性退化為單一區塊。由於每個 B(i) 皆為零，系統 (C) 變為
(Im ⊗ M) WPE = g。(C')

Step 4. Solve the reduced system. Write WPE = (wpe(1),..., wpe(m)) and g = (g1,..., gm) with blocks in RT. The Kronecker structure of Im ⊗ M splits (C') into m independent systems
MWPE^(i) = gi, i=1,...,m. (Ci)
Because dpe ≥ T, the rows of M are linearly dependent at worst; pick any right inverse M+ (for instance the Moore-Penrose pseudoinverse). Then
WPE^(i) := M+gi (i = 1,...,m)
solves each (Ci), and hence WPE solves the full system (C').

步驟 4. 求解簡化系統。將 WPE 寫成 (wpe(1), ..., wpe(m))，g 寫成 (g1, ..., gm)，區塊在 R^T 中。Im ⊗ M 的克羅內克結構將 (C') 分解為 m 個獨立系統
M WPE^(i) = gi，i=1,...,m。(Ci)
因為 dpe ≥ T，M 的行在最壞情況下是線性相關的；選擇任意右逆 M⁺（例如 Moore-Penrose 偽逆）。然後
WPE^(i) := M⁺ gi (i = 1, ..., m)
求解每個 (Ci)，因此 WPE 求解整個系統 (C')。

Step 5. Conclusion. Compatibility holds for every choice of data {gi} once the frequencies obey ωn = 2πkn/L.
²least common multiple

步驟 5. 結論。一旦頻率滿足 ωn = 2πkn/L，對於任何資料 {gi} 的選擇，相容性都成立。
²最小公倍數
