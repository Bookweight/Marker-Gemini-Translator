---
title: A Survey on Temporal Knowledge GraphR
field: Knowledge_Graph
status: Imported
created_date: 2026-01-13
pdf_link: "[[A Survey on Temporal Knowledge GraphR.pdf]]"
tags:
  - paper
  - knowledge_graph
---
# A Survey on Temporal Knowledge Graph: Representation Learning and Applications
# 時態知識圖譜綜述：表示學習與應用

Li Caia,b, Xin Maoª, Yuhao Zhou, Zhaoguang Longa, Changxu Wu, Man Lana, *
蔡力a,b, 毛鑫ª, 周宇浩, 龍兆光a, 吳長旭, 蘭曼a, *

"School of Computer Science and Technology, East China Nomal University, 3663 North Zhongshan Road, Shanghai, 200062, China
a華東師範大學電腦科學與技術學院，中國上海市中山北路3663號，郵編200062

College of Computer Science and Technology, Guizhou University, 2708 South Huaxi Avenue, Guiyang, 550025, Guizhou, China
b貴州大學電腦科學與技術學院，中國貴州省貴陽市花溪大道南段2708號，郵編550025

Department of Industrial Engineering, Tsinghua University, 30 Shuangqing Road, Beijing, 100084, China
c清華大學工業工程系，中國北京市雙清路30號，郵編100084

## Abstract
## 摘要

Knowledge graphs have garnered significant research attention and are widely used to enhance downstream applications. However, most current studies mainly focus on static knowledge graphs, whose facts do not change with time, and dis- regard their dynamic evolution over time. As a result, temporal knowledge graphs have attracted more attention because a large amount of structured knowledge exists only within a specific period. Knowledge graph representa- tion learning aims to learn low-dimensional vector embeddings for entities and relations in a knowledge graph. The representation learning of temporal knowl- edge graphs incorporates time information into the standard knowledge graph framework and can model the dynamics of entities and relations over time. In this paper, we conduct a comprehensive survey of temporal knowledge graph representation learning and its applications. We begin with an introduction to the definitions, datasets, and evaluation metrics for temporal knowledge graph representation learning. Next, we propose a taxonomy based on the core tech- nologies of temporal knowledge graph representation learning methods, and provide an in-depth analysis of different methods in each category. Finally, we present various downstream applications related to the temporal knowledge graphs. In the end, we conclude the paper and have an outlook on the future research directions in this area.
知識圖譜已引起廣泛的研究關注，並被廣泛用於增強下游應用。然而，目前大多數研究主要集中在靜態知識圖譜，其事實不隨時間變化，並忽略了其隨時間的動態演變。因此，時態知識圖譜吸引了更多關注，因為大量結構化知識僅在特定時期內存在。知識圖譜表示學習旨在為知識圖譜中的實體和關係學習低維向量嵌入。時態知識圖譜的表示學習將時間資訊納入標準知識圖譜框架中，並可以對實體和關係隨時間的動態進行建模。在本文中，我們對時態知識圖譜表示學習及其應用進行了全面綜述。我們首先介紹時態知識圖譜表示學習的定義、資料集和評估指標。接下來，我們根據時態知識圖譜表示學習方法的核心技術提出了一個分類法，並對每個類別中的不同方法進行了深入分析。最後，我們介紹了與時態知識圖譜相關的各種下游應用。最後，我們總結了本文，並對該領域未來的研究方向進行了展望。

Keywords: Temporal knowledge graph, Representation learning, Knowledge reasoning, Entity alignment, Question answering
關鍵詞：時態知識圖譜、表示學習、知識推理、實體對齊、問答

* Corresponding author.
* 通訊作者。

Email addresses: lcai2020@stu.ecnu.edu.com (Li Cai), xmao@stu.ecnu.edu.cn (Xin Mao), 51265900018@stu.ecnu.edu.cn (Yuhao Zhou), 51265901014@stu.ecnu.edu.cn (Zhaoguang Long), wuchangxu@tsinghua.edu.cn (Changxu Wu), mlan@cs.ecnu.edu.cn (Man Lan)
電子郵件地址：lcai2020@stu.ecnu.edu.com (蔡力), xmao@stu.ecnu.edu.cn (毛鑫), 51265900018@stu.ecnu.edu.cn (周宇浩), 51265901014@stu.ecnu.edu.cn (龍兆光), wuchangxu@tsinghua.edu.cn (吳長旭), mlan@cs.ecnu.edu.cn (蘭曼)

Preprint submitted to Arxiv
預印本提交至 Arxiv

March 11, 2024
2024年3月11日

## 1. Introduction
## 1. 簡介

Knowledge graphs (KGs) describe the real world with structured facts. A fact consists of two entities and a relation connecting them, which can be for- mally represented as a triple (head, relation, tail), and an instance of a fact is (Barack Obama, make statement, Iran). Knowledge graph representation learning (KGRL) [33] seeks to learn the low-dimentional vector embeddings of entities and relations and use these embeddings for downstream tasks such as in- formation retrieval [16], question answering [29], and recommender systems [2].
知識圖譜（KGs）以結構化事實描述真實世界。一個事實由兩個實體和一個連接它們的關係組成，可以形式化地表示為一個三元組（頭實體，關係，尾實體），例如一個事實的實例是（Barack Obama, make statement, Iran）。知識圖譜表示學習（KGRL）[33] 旨在學習實體和關係的低維向量嵌入，並將這些嵌入用於下游任務，例如資訊檢索 [16]、問答 [29] 和推薦系統 [2]。

Existing KGs ignore the timestamp indicating when a fact occurred and cannot reflect their dynamic evolution over time. In order to represent KGs more accurately, Wikidata [73] and YOGO2 [32] add temporal information to the facts, and some event knowledge graphs [55, 42] also contain the timestamps indicating when the events occurred. The knowledge graphs with temporal in- formation are called temporal knowledge graphs (TKGs). Figure 1 is a subgraph of the temporal knowledge graph. The fact in TKGs are expanded into quadru- ple (head, relation, tail, timestamp), a specific instance is (Barack Obama, make statement, Iran, 2014-6-19).
現有的知識圖譜忽略了事實發生的時間戳，無法反映其隨時間的動態演變。為了更準確地表示知識圖譜，Wikidata [73] 和 YOGO2 [32] 為事實添加了時間資訊，一些事件知識圖譜 [55, 42] 也包含指示事件發生時間的時間戳。帶有時間資訊的知識圖譜稱為時態知識圖譜（TKGs）。圖 1 是時態知識圖譜的一個子圖。在 TKGs 中，事實被擴展為四元組（頭實體，關係，尾實體，時間戳），一個具體實例是（Barack Obama, make statement, Iran, 2014-6-19）。

[Image]

Figure 1: An example of temporal knowledge graph (a subgraph of ICEWS14).
圖 1：時態知識圖譜示例（ICEWS14 的子圖）。

The emergence of TKGs has led to increased researcher interest in temporal knowledge graph representation learning (TKGRL) [53]. The acquired low- dimensional vector representations are capable of modelling the dynamics of entities and relations over time, thereby improving downstream applications such as time-aware knowledge reasoning [34], entity alignment [81], and question answering [59].
TKGs 的出現引起了研究人員對時態知識圖譜表示學習 (TKGRL) [53] 日益增長的興趣。所獲得的低維向量表示能夠對實體和關係隨時間的動態進行建模，從而改進下游應用，例如時間感知知識推理 [34]、實體對齊 [81] 和問答 [59]。

Temporal knowledge graph representation learning and applications are at the forefront of current research. Nevertheless, as of now, a comprehensive survey on the topic is not yet available. Ji et al. [33] provides a survey on KGs, which includes a section on TKGs. However, this section only covers a limited number of early methods related to TKGRL. The paper [53] is a survey on TKGs, with one section dedicated to introducing representation learning methods of static knowledge graphs, and only five models related to TKGRL are elaborated in detail. The survey [6] is about temporal knowledge graph completion (TKGC), and it focuses solely on the interpolation-based temporal knowledge graph reasoning application.
時態知識圖譜表示學習與應用是當前研究的前沿。然而，截至目前，尚無關於此主題的全面綜述。Ji 等人 [33] 的論文對知識圖譜進行了綜述，其中包含一個關於時態知識圖譜的章節。然而，該章節僅涵蓋了與時態知識圖譜表示學習相關的有限數量的早期方法。論文 [53] 是一篇關於時態知識圖譜的綜述，其中一節專門介紹靜態知識圖譜的表示學習方法，僅詳細闡述了五個與時態知識圖譜表示學習相關的模型。綜述 [6] 關於時態知識圖譜補全 (TKGC)，且僅專注於基於內插的時態知識圖譜推理應用。

This paper comprehensively summarizes the current research of TKGRL and its related applications. Our main contributions are summarized as follows: (1) We conduct an extensive investigation on various TKGRL methods up to the present, analyze their core technologies, and propose a new classification taxonomy. (2) We divide the TKGRL methods into ten distinct categories. Within each category, we provide detailed information on the key components of different methods and analyze the strengths and weaknesses of these methods. (3) We introduce the latest development of different applications related to TKGs, including temporal knowledge graph reasoning, entity alignment between temporal knowledge graphs, and question answering over temporal knowledge graphs. (4) We summarize the existing research of TKGRL and point out the future directions which can guide further work.
本文全面總結了 TKGRL 及其相關應用的當前研究。我們的觽獻總結如下：(1) 我們對截至目前的各種 TKGRL 方法進行了廣泛的調查，分析了其核心技術，並提出了一個新的分類法。(2) 我們將 TKGRL 方法分為十個不同的類別。在每個類別中，我們提供有關不同方法關鍵組件的詳細資訊，並分析這些方法的優缺點。(3) 我們介紹了與 TKG 相關的不同應用的最新發展，包括時態知識圖譜推理、時態知識圖譜之間的實體對齊以及時態知識圖譜上的問答。(4) 我們總結了 TKGRL 的現有研究，並指出了可以指導未來工作的未來方向。

The remainder of this paper is organized as follows: Chapter 2 introduces the background of temporal knowledge graphs, including definitions, datasets, and evaluation metrics. Chapter 3 summarizes various temporal knowledge graph representation learning methods, including transformation-based meth- ods, decomposition-based methods, graph neural networks-based methods, cap- sule network-based methods, autoregression-based methods, temporal point process- based methods, interpretability-based methods, language model methods, few- shot learning methods and others. Chapter 4 introduces the related applications of the temporal knowledge graph, such as temporal knowledge graph reasoning, entity alignment between temporal knowledge graphs, and question answering over temporal knowledge graphs. Chapter 5 highlights the future directions of Temporal Knowledge Graph Representation Learning (TKGRL), encompassing Scalability, Interpretability, Information Fusion, and the Integration of Large Language Models. Chapter 6 gives a conclusion of this paper.
本文的其餘部分組織如下：第 2 章介紹了時態知識圖譜的背景，包括定義、資料集和評估指標。第 3 章總結了各種時態知識圖譜表示學習方法，包括基於變換的方法、基於分解的方法、基於圖神經網路的方法、基於膠囊網路的方法、基於自回歸的方法、基於時態點過程的方法、基於可解釋性的方法、語言模型方法、少樣本學習方法等。第 4 章介紹了時態知識圖譜的相關應用，例如時態知識圖譜推理、時態知識圖譜之間的實體對齊以及時態知識圖譜上的問答。第 5 章重點介紹了時態知識圖譜表示學習 (TKGRL) 的未來方向，包括可擴展性、可解釋性、資訊融合和大型語言模型的整合。第 6 章對本文進行了總結。

## 2. Background
## 2. 背景

### 2.1. Problem Formulation
### 2.1. 問題陳述

A temporal knowledge graph is a directed multi-relational graph containing structured facts. It is usually expressed as G = (E, R,T, F), where E, R, and T are the sets of entities, relations, and timestamps, respectively, and FC E×R×EXT is the set of all possible facts. A fact f is denoted as (h, r, t, ㅜ), where h,r,t, and are the head entity, relation, tail entity, and timestamp, respectively.
時態知識圖譜是一個包含結構化事實的有向多重關係圖。通常表示為 G = (E, R, T, F)，其中 E、R 和 T 分別是實體、關係和時間戳的集合，而 F ⊆ E×R×E×T 是所有可能事實的集合。一個事實 f 表示為 (h, r, t, τ)，其中 h、r、t 和 τ 分別是頭實體、關係、尾實體和時間戳。

[Image]

Figure 2: Temporal knowledge graph representation learning and applications.
圖 2：時態知識圖譜表示學習與應用。

Take Figure 1 for example, where the entity set E contains (Barack Obama, Iran, Iraq), the relation set contains (make statement, express intent to provide military aid, provide military aid, provide military protection or peacekeeping, receive deployment peacekeeper), the time set contains (2014-6-18, 2014-6-19, 2014-6-20, 2014-6-22), and the fact set contains ((Iran, Provide military protec- tion or peacekeeping, Iraq, 2014-6-18)), (Iraq, receive deployment peacekeeper, Iran, 2014-6-18), (Barack Obama, Make statement, Iran, 2014-6-19), (Iran, Make statement, Barack Obaта, 2014-6-20),(Barack Obama, express intent to provide military aid, Iraq, 2014-6-19), (Barack Obama, provide military aid, Iraq, 2014-6-22)).
以圖 1 為例，其中實體集 E 包含 {Barack Obama, Iran, Iraq}，關係集包含 {make statement, express intent to provide military aid, provide military aid, provide military protection or peacekeeping, receive deployment peacekeeper}，時間集包含 {2014-6-18, 2014-6-19, 2014-6-20, 2014-6-22}，事實集包含 {(Iran, Provide military protection or peacekeeping, Iraq, 2014-6-18)}, {(Iraq, receive deployment peacekeeper, Iran, 2014-6-18)}, {(Barack Obama, Make statement, Iran, 2014-6-19)}, {(Iran, Make statement, Barack Obama, 2014-6-20)}, {(Barack Obama, express intent to provide military aid, Iraq, 2014-6-19)}, {(Barack Obama, provide military aid, Iraq, 2014-6-22)}。

TKGRL aims to effectively learn low-dimensional vector representations of entities h, t, relations r, and timestamps T for downstream tasks such as knowl- edge reasoning, entity alignment and question answering (as shown in Figure 2).
TKGRL旨在有效地學習實體h、t、關係r和時間戳τ的低維向量表示，用於知識推理、實體對齊和問答等下游任務（如圖2所示）。

### 2.2. Datasets
### 2.2. 資料集

There are four commonly used datasets for temporal knowledge graph rep- resentation learning.
時態知識圖譜表示學習有四個常用的資料集。

ICEWS The integrated crisis early warning system (ICEWS) [55] captures and processes millions of data points from digitized news, social media, and other sources to predict, track and respond to events around the world, primarily for early warning. Three subsets are typically used: ICEWS14, ICEWS05-15, and ICEWS18, which contain events in 2014, 2005-2015, and 2018, respectively.
ICEWS 綜合危機預警系統 (ICEWS) [55] 捕獲並處理來自數位化新聞、社交媒體和其他來源的數百萬個數據點，以預測、跟蹤和應對世界各地的事件，主要用於早期預警。通常使用三個子集：ICEWS14、ICEWS05-15 和 ICEWS18，分別包含 2014 年、2005-2015 年和 2018 年的事件。

GDELT The global database of events, language, and tone (GDELT) [42] is a global database of society. It includes the world's broadcast, print, and web news from across every country in over 100 languages and continually updates every 15 minutes.
GDELT 全球事件、語言和語氣資料庫 (GDELT) [42] 是一個全球性的社會資料庫。它包括來自全球各國、超過 100 種語言的廣播、印刷和網路新聞，並每 15 分鐘持續更新。

Wikidata The Wikidata [73] is a collaborative, multilingual auxiliary knowl- edge base hosted by the Wikimedia Foundation to support resource sharing and other Wikimedia projects. It is a free and open knowledge base that can be read and edited by both humans and machines. Many items in Wikidata have temporal information.
Wikidata [73] 是一個由維基媒體基金會託管的協作式、多語言輔助知識庫，以支援資源共享和其他維基媒體專案。它是一個可供人類和機器讀取和編輯的自由開放的知識庫。Wikidata 中的許多項目都具有時間資訊。

YAGO The YAGO [32] is a linked database developed by the Max Planck Institute in Germany. YAGO integrates data from Wikipedia, WordNet, and GeoNames. YAGO integrates WordNet's word definitions with Wikipedia's clas- sification system, adding temporal and spatial information to many knowledge items.
YAGO [32] 是由德國馬克斯·普朗克研究所開發的一個連結資料庫。YAGO 整合了來自維基百科、WordNet 和 GeoNames 的資料。YAGO 將 WordNet 的詞義定義與維基百科的分類系統相結合，為許多知識項添加了時間和空間資訊。

[Image]

The datasets of TKGs often require unique data processing methods for dif- ferent downstream applications. Table 1 presents the statistics of datasets for various tasks of TKGs. In knowledge reasoning tasks, datasets are typically divided into different training sets, validation sets, and test sets based on task type (interpolation and extrapolation). In entity alignment tasks, as the same entites in the real world need to be aligned between different KGs, a dataset always includes two temporal knowledge graphs that must be learned simulta- neously. For question answering tasks, the datasets not only include temporal knowledge graphs used to search for answers but also include temporal-related questions (which have not been showed here).
TKG 的資料集通常需要針對不同的下游應用採用獨特的資料處理方法。表 1 顯示了 TKG 在各種任務中的資料集統計資料。在知識推理任務中，資料集通常根據任務類型（內插和外插）劃分為不同的訓練集、驗證集和測試集。在實體對齊任務中，由於需要在不同 KG 之間對齊現實世界中的相同實體，因此資料集始終包含兩個必須同時學習的時態知識圖譜。對於問答任務，資料集不僅包含用於搜索答案的時態知識圖譜，還包含與時間相關的問題（此處未顯示）。

### 2.3. Evaluation Metrics
### 2.3. 評估指標

The evaluation metrics for verifying the performance of TKGRL are MRR (mean reciprocal rank) and Hit@k.
用於驗證 TKGRL 性能的評估指標是 MRR（平均倒數排名）和 Hit@k。

MRR The MRR represents the average of the reciprocal ranks of the cor- rect answers. It can be calculated as follows:
MRR MRR 代表正確答案倒數排名的平均值。其計算公式如下：

[Image]

where S is the set of all correct answers, S is the number of the sets. The predicted result is a set sorted by the probability of the answer from high to low, and ranki is the rank of the i-th correct answer in the prediction result. The higher the MRR, the better the performance.
其中 S 是所有正確答案的集合，|S| 是集合的數量。預測結果是根據答案的機率從高到低排序的集合，ranki 是第 i 個正確答案在預測結果中的排名。MRR 越高，性能越好。

Hit@k The Hits@k reports the proportion of correct answers in the top k predict results. It can be calculated by the following equation:
Hit@k Hit@k 報告了預測結果中前 k 個正確答案的比例。它可以通过以下等式计算：

[Image]

[Image]

where the S and ranki are the same as above, I(·) is the indicator function (If the condition (ranki ≤ k) is true, the function value is 1, otherwise 0). Typically, k is 1,3,5 or 10. Hit@1 represents the percentage that the first-rank predicted result is the correct answer, equivalent to the Accuracy. Hit@10 represents the percentage of the top ten predictions containing correct answers. The higher the Hit@k, the better the performance.
其中 S 和 ranki 與上述相同，I(·) 是指示函數（如果條件 (ranki ≤ k) 為真，則函數值為 1，否則為 0）。通常，k 為 1、3、5 或 10。Hit@1 表示排名第一的預測結果是正確答案的百分比，相當於準確率。Hit@10 表示前十個預測中包含正確答案的百分比。Hit@k 越高，性能越好。

## 3. Temporal Knowledge Graph Representation Learning Methods
## 3. 時態知識圖譜表示學習方法

Compared to KGs, TKGs contain additional timestamps, which are taken into account in the construction of TKGRL methods. These methods can be broadly categorized into transformation-based, decomposition-based, graph neural networks-based, and capsule network-based approaches. Additionally, temporal knowledge graphs can be viewed as sequences of snapshots captured at different timestamps or events that occur over continuous time, and can be learned using autoregressive and temporal point process techniques. More- over, some methods prioritize interpretability, language model, and few-shot learning. Thus, based on the core technologies employed by TKGRL methods, we group them into the following categories: transformation-based methods,
與知識圖譜相比，時態知識圖譜包含額外的時間戳，在建構 TKGRL 方法時會加以考量。這些方法可大致分為基於轉換、基於分解、基於圖神經網路和基於膠囊網路的方法。此外，時態知識圖譜可被視為在不同時間戳捕獲的快照序列或隨時間連續發生的事件，並可使用自回歸和時態點過程技術進行學習。此外，一些方法優先考慮可解釋性、語言模型和少樣本學習。因此，根據 TKGRL 方法採用的核心技術，我們將它們分為以下幾類：基於轉換的方法、

[Image]

decomposition-based methods, graph neural networks-based methods, capsule network-based methods, autoregression-based methods, temporal point process- based methods, interpretability-based methods, language model methods, few- shot learning methods, and others. A visualization of the categorization of TKGRL methods is presented in Figure 3.
基於分解的方法、基於圖神經網路的方法、基於膠囊網路的方法、基於自回歸的方法、基於時態點過程的方法、基於可解釋性的方法、語言模型方法、少樣本學習方法等。TKGRL 方法的分類可視化見圖 3。

The notations in these methods are varied, and we define our notations to describe them uniformly. We use lower-case letters to denote scalars, bold lower- case letters to denote vectors, bold upper-case letters to denote matrices, bold calligraphy upper-case letters to denote order 3 tensors, and bold script upper- case letters to denote order 4 tensors. The main notations and their descriptions are listed in table 2.
這些方法中的符號各不相同，我們定義我們的符號來統一描述它們。我們使用小寫字母表示純量，粗體小寫字母表示向量，粗體大寫字母表示矩陣，粗體書法體大寫字母表示三階張量，粗體手寫體大寫字母表示四階張量。主要符號及其描述列於表 2。

### 3.1. Transformation-based Methods
### 3.1. 基於變換的方法

In the transformation-based method, timestamps or relations are regarded as the transformation between entities. The representation learning of TKGs is carried out by integrating temporal information or mapping entities and rela- tions to temporal hyperplanes based on the existing KGRL methods. There are translation-based transformations and rotation-based transformations.
在基於變換的方法中，時間戳或關係被視為實體之間的變換。時態知識圖譜的表示學習是透過整合時間資訊或基於現有的 KGRL 方法將實體和關係映射到時態超平面來進行的。存在基於平移的變換和基於旋轉的變換。

Translation-based The translation-based representation learning methods of TKGs are developed based on TransE [11]. TransE is a classic translation- based KGRL method. It regards the relation r as a translation from the head entity h to the tail entity t, that is, h + r ≈ t, so the score function is ||h +r- t||1/2.
基於平移的 TKG 表示學習方法是基於 TransE [11] 開發的。TransE 是一種經典的基於平移的 KGRL 方法。它將關係 r 視為從頭實體 h 到尾實體 t 的平移，即 h + r ≈ t，因此分數函數為 ||h + r - t||1/2。

TTransE [40] concatenate the temporal information to the relations, and the quadruple (h, r, t, r) is converted into a triple (h, [r|r], t). The new relation embeddings are expressed as r + ㅜ, and the score function is ||h+r+r-t||1/2.
TTransE [40] 將時間資訊與關係連接起來，並將四元組 (h, r, t, τ) 轉換為三元組 (h, [r|τ], t)。新的關係嵌入表示為 r + τ，分數函數為 ||h + r + τ - t||1/2。

Compared with the simple connection of temporal information and relations by TTransE, TA-TransE [19] creates more expressive synthetic relations, which treats relations as sequences containing temporal information, such as (make statement, 2y, 0y,1y,4y,06m,1d,9d). Then TA-TransE use LSTM [31] to learn the relation embedding rseq, and the score function is ||h + rseq - t||2.
與 TTransE 簡單連接時間資訊和關係相比，TA-TransE [19] 創建了更具表達力的合成關係，它將關係視為包含時間資訊的序列，例如 (make statement, 2y, 0y,1y,4y,06m,1d,9d)。然後 TA-TransE 使用 LSTM [31] 來學習關係嵌入 rseq，分數函數為 ||h + rseq - t||2。

Different from TTransE and TA-TransE, HyTE [13] learns the represen- tations of entities, relations, and temporal information jointly. It splits the temporal knowledge graph into multiple static subgraphs, each of which corre- sponds to a timestamp. Then it incorporates time in the entity-relation space by associating each timestamp with a corresponding hyperplane. For a timestamp τ, the corresponding hyperplane is w₁, if a triple (h, r, t) is valid for the times- tamp, then the projected representations of the triple on the hyperplane are h₁ = h − (wh)w₁,r₁ = r (w/r)w₁, t₁ = t − (wt)w, the score function is ||h+r+ - tr||1/2.
與 TTransE 和 TA-TransE 不同，HyTE [13] 聯合學習實體、關係和時間資訊的表示。它將時態知識圖譜分割成多個靜態子圖，每個子圖對應一個時間戳。然後，它通過將每個時間戳與相應的超平面相關聯，將時間納入實體-關係空間。對於時間戳 τ，對應的超平面為 wτ，如果三元組 (h, r, t) 在該時間戳有效，則三元組在超平面上的投影表示為 hτ = h − (wτh)wτ，rτ = r − (wτr)wτ，tτ = t − (wτt)wτ，分數函數為 ||hτ + rτ - tτ||1/2。

Rotation-based The translation-based methods can infer the inverse and composition relation patterns, except the symmetry pattern, because all the symmetry relations will be presented by a 0 vector, so the rotation-based trans- formation methods appear. RotatE [66] is the first method based on rotation. It regards relation as a rotation from head entity to tail entity, expands the representation space from real-valued point-wise space (Rd) to complex vector space (Cd, c = a + bi, a, b ∈ Rd), and expects t = hor, where o is Hadamard (element-wise) product. The score function is ||hor-t||1.
基於旋轉的平移方法可以推斷逆關係和組合關係模式，除了對稱模式，因為所有對稱關係都將由 0 向量表示，因此出現了基於旋轉的變換方法。RotatE [66] 是第一個基於旋轉的方法。它將關係視為從頭實體到尾實體的旋轉，將表示空間從實值點空間 (Rd) 擴展到複向量空間 (Cd, c = a + bi, a, b ∈ Rd)，並期望 t = h ○ r，其中 ○ 是哈達瑪（逐元素）乘積。分數函數為 ||h ○ r - t||1。

Tero [79] regards timestamps as the rotation of entities in complex space. The mapping function is h₁ = h °r, t₁ = tot, then the entity representations contain the temporal information. It regards the relation embedding ras the translation from the time-specific head entity embedding h, to the conjugate of the time-specific tail entity embedding t₁. The score function is ||h, + r -t7||1.
Tero [79] 將時間戳視為複數空間中實體的旋轉。映射函數為 hτ = h ○ τ，tτ = t ○ τ，則實體表示包含時間資訊。它將關係嵌入 r 視為從特定時間的頭實體嵌入 hτ 到特定時間的尾實體嵌入 tτ 共軛的平移。分數函數為 ||hτ + r - tτ||1。

ChronoR [58] proposes a model with a k-dimensional rotation transforma- tion. It regards the relations and timestamps as the rotation of the head entity in k-dimensional space and expects the head entity falls near the tail entity after the rotation. The rotation is defined as hr, = h॰ [r|T]○r2, where | presents concatenation operator, r2 is the static relation representation without considering temporal information. The score function is (hr,,t).
ChronoR [58] 提出了一個具有 k 維旋轉變換的模型。它將關係和時間戳視為頭實體在 k 維空間中的旋轉，並期望頭實體在旋轉後落在尾實體附近。旋轉定義為 hr,τ = h ○ [r|τ] ○ r2，其中 | 表示串聯運算符，r2 是不考慮時間資訊的靜態關係表示。分數函數為 (hr,τ, t)。

RotateQVS [9] utilizes a hypercomplex (quaternion) vector space ( Hd, q = a + bi + cj + dk, a,b,c,d ∈ Rd) to represent entities, relations, and timestamps. The time-specific entity representations are learned through tem- poral rotation transformations in 3D space, where the mapping function is denoted as h₁ = Thr®1,t₁ = TtT¯¹, and the score function is defined as ||h + r -t7||2.
RotateQVS [9] 利用超複數（四元數）向量空間（Hd, q = a + bi + cj + dk, a,b,c,d ∈ Rd）來表示實體、關係和時間戳。特定時間的實體表示是通過在 3D 空間中進行時間旋轉變換來學習的，其中映射函數表示為 hτ = τhτ⁻¹，tτ = τtτ⁻¹，分數函數定義為 ||hτ + r - tτ||2。

### 3.2. Decomposition-based Methods
### 3.2. 基於分解的方法

The main task of representation learning is to learn the low-dimensional vector representation of the knowledge graph. Tensor decomposition has three applications: dimension reduction, missing data completion, and implicit rela- tion mining, which meet the needs of knowledge graph representation learning.
表示學習的主要任務是學習知識圖譜的低維向量表示。張量分解有三個應用：降維、缺失數據補全和隱含關係挖掘，這些都滿足了知識圖譜表示學習的需求。

The knowledge graph consists of triples and can be represented by an order 3 tensor. For the temporal knowledge graph, the additional temporal information can be represented by an order 4 tensor, and each tensor dimension is the head entity, relation, tail entity, and timestamp, respectively. Tensor decomposition includes Canonical Polyadic (CP) decomposition [30] and Tucker decomposi- tion [69].
知識圖譜由三元組組成，可用三階張量表示。對於時態知識圖譜，額外的時間資訊可用四階張量表示，每個張量維度分別為頭實體、關係、尾實體和時間戳。張量分解包括典型多線性 (CP) 分解 [30] 和塔克 (Tucker) 分解 [69]。

CP decomposition For an order 3 tensor X ∈ RN1Xn2Xn3, CP decom- position factorize X as X ≈ (A,B,C) = Σα=1 Α,α & Βα& Ca, where presents the Kronecker product, A ∈ Rn1×d, B ∈ Rn2×d,C ∈ Rn3xd are decomposed matrices.
CP 分解 對於一個三階張量 X ∈ RN1Xn2Xn3，CP 分解將 X 分解為 X ≈ (A,B,C) = Σdα=1 A·,α ⊗ B·,α ⊗ C·,α，其中 ⊗ 表示克羅內克積，A ∈ Rn1×d, B ∈ Rn2×d, C ∈ Rn3×d 為分解矩陣。

DE-SimplE [21] learns the diachronic embeddings of entities and uses the score function in SimplE [37] for temporal knowledge graph completion. Sim- plE is an enhancement of CP decomposition, which can learn the entity em- beddings independently. The diachronic entity embedding is defined as e-[n] = [e[n]σ(w[n]+b[n]), 1 ≤ n ≤ yd , and the score function is ((h, r, t₁) + (t,r-1,h)).
DE-SimplE [21] 學習實體的歷時嵌入，並使用 SimplE [37] 中的評分函數進行時間知識圖譜補全。SimplE 是 CP 分解的增強版，可以獨立學習實體嵌入。歷時實體嵌入定義為 et[n] = { e[n]σ(wt[n]+bt[n]), 1 ≤ n ≤ yd e[n], yd ≤ n ≤ d }，評分函數為 (⟨h, r, tt⟩ + ⟨t, r⁻¹, ht⟩)。

T-SimplE [48] regards the temporal knowledge graph as an order 4 tensor and uses tensor decomposition to learn the embeddings of entities, relations, and timestamps. The score function is (<h, r, t, ㅜ) + (t,r-1,h,r)).
T-SimplE [48] 將時態知識圖譜視為一個四階張量，並使用張量分解來學習實體、關係和時間戳的嵌入。評分函數為 (⟨h, r, t, τ⟩ + ⟨t, r⁻¹, h, τ⟩)。

TComplEx [39] extends the representation space to complex vector space and uses the tensor decomposition based on ComplEx [68]. ComplEx is a simple link prediction method with complex embeddings. TComplex adds the temporal embeddings and takes re((h,r,ł,ㅜ)) as the score function.
TComplEx [39] 將表示空間擴展到複向量空間，並使用基於 ComplEx [68] 的張量分解。ComplEx 是一種使用複數嵌入的簡單連結預測方法。TComplEx 添加了時間嵌入，並將 re(⟨h, r, t, τ⟩) 作為分數函數。

TLT-KGE [83] integrates time representation into the representation of en- tities and relations, and verifies its validity in both complex and hypercomplex (quaternion) spaces. Specifically, in the case of complex space, the represen- tation of entities and relations is a combination of their semantic representa- tion (real part) and temporal representation (imaginary part): h = h + Tei, t = t + Tei, r = r + Tri. Then calculate C(h,r,t) = hr = (ho r - Te © Tr) + (h • Tr + Te © r)i = Co + C₁i, let C'(h,r,r) = C1 + coi, the score function is ¢º(h,r,t,r) = (C(h,r,r), t) + (C'(h,r,r),t) = (co, t) + (C1, Te) + (Co, Te) + (C1, t). In the case of hypercomplex space, half of the quaternion is dedicated to semantic representation while the other half is ded- icated to temporal representation: h = ha + hbi + Te,cj + Te,dk, t = ta + tbi + Te,cj + Te,dk, rq = ra + rbi + Tr,cj + Tr,dk, the score function is ¢º(h,r,t,r) = (Q(h,r,r),tq) + (Q′(h,r,r), tq). In addition, the author in- troduced a shared time window module for capturing the periodicity of entities and relationships, as well as a relation-timestamp composition module to model relation representations at specific time. The model was further regularized with respect to entities, relationships, and time to improve its generalization perfor- mance and mitigate overfitting. Experimental results indicate that the model in hypercomplex space achieves state-of-the-art results, but it also requires a large amount of storage space.
TLT-KGE [83] 將時間表示整合到實體和關係的表示中，並在複數和超複數（四元數）空間中驗證其有效性。具體而言，在複數空間中，實體和關係的表示是其語義表示（實部）和時間表示（虛部）的組合：h = h + τei, t = t + τei, r = r + τri。然後計算 C(h,r,t) = h ⊙ r = (h ⊙ r - τe ⊙ τr) + (h ⊙ τr + τe ⊙ r)i = c0 + c1i，令 C'(h,r,τ) = c1 + c0i，分數函數為 φ(h,r,t,τ) = ⟨C(h,r,τ), t⟩ + ⟨C'(h,r,τ), t⟩ = ⟨c0, t⟩ + ⟨c1, τe⟩ + ⟨c0, τe⟩ + ⟨c1, t⟩。在超複數空間中，四元數的一半用於語義表示，另一半用於時間表示：hq = ha + hbi + τe,cj + τe,dk, tq = ta + tbi + τe,cj + τe,dk, rq = ra + rbi + τr,cj + τr,dk，分數函數為 φq(h,r,t,τ) = ⟨Q(h,r,τ), tq⟩ + ⟨Q'(h,r,τ), tq⟩。此外，作者引入了一個共享時間窗口模組來捕捉實體和關係的周期性，以及一個關係-時間戳組合模組來建模特定時間的關係表示。該模型進一步針對實體、關係和時間進行正則化，以提高其泛化性能並減輕過擬合。實驗結果表明，該模型在超複數空間中取得了最先進的結果，但它也需要大量的儲存空間。

Tucker decomposition Tucker decomposition is a more general tensor de- composition technique, and CP decomposition is its particular case. Tucker decomposition factorizes a tensor into a core tensor multiplied by a matrix in each dimension, such as X ≈ 〈W; A,B,C) = (W×1A×2B×3C), where W∈ Rd1×d2×ds is the core tensor, A ∈ Rn1×d1, B ∈ Rn2×d2, C∈ Rn3×d3 are decomposed matrices. When W is a hyper-diagonal tensor and d₁ = d2 = d3, Tucker decomposition is equivalent to CP decomposition. TuckER [3] intro- duces the Tucker decomposition for link prediction on knowledge graphs, which regards the knowledge graph as an order 3 tensor and decomposes it into a core tensor, head entity embedding, relation embedding, and tail entity embedding, the score function is 〈W; h, r,t).
塔克分解 塔克分解是一種更通用的張量分解技術，CP 分解是其特例。塔克分解將張量分解為一個核心張量乘以每個維度的一個矩陣，例如 X ≈ ⟨W; A,B,C⟩ = (W×₁A×₂B×₃C)，其中 W ∈ Rd₁×d₂×d₃ 是核心張量，A ∈ Rn₁×d₁, B ∈ Rn₂×d₂, C ∈ Rn₃×d₃ 是分解矩陣。當 W 是超對角張量且 d₁ = d₂ = d₃ 時，塔克分解等同於 CP 分解。TuckER [3] 介紹了用於知識圖譜連結預測的塔克分解，它將知識圖譜視為一個三階張量，並將其分解為一個核心張量、頭實體嵌入、關係嵌入和尾實體嵌入，分數函數為 ⟨W; h, r, t⟩。

TuckERT [64] proposes an order 4 tensor decomposition model based on Tucker decomposition. The model is fully expressive and effective for temporal knowledge graph completion. The score function is (M; h, r, t, ㅜ), where M is an order 4 tensor.
TuckERT [64] 提出了一種基於塔克分解的四階張量分解模型。該模型對於時態知識圖譜補全任務具有充分的表達能力和有效性。其評分函數為 <M; h, r, t, τ>，其中 M 為一個四階張量。

### 3.3. Graph Neural Networks-based Methods
### 3.3. 基於圖神經網路的方法

Graph Neural Networks (GNN) [60] have powerful structure modeling abil- ity. The entity can enrich its representation with the attribute feature and the global structure feature by GNN. Typical graph neural networks include Graph Convolutional Networks (GCN) [38] and Graph Attention Networks (GAT) [72]. GCN gets the representation of nodes by aggregating neighbor embeddings, and GAT uses a multi-head attention mechanism to get the representation of nodes by aggregating weighted neighbor embedding. The knowledge graph is a kind of graph that has different relations. The relation-aware graph neural networks are developed to learn the representations of entities in the knowledge graph. Rela- tional Graph Convolutional Networks (R-GCN) [61] is a graph neural network model for relational data. It learns the representation for each relation and obtains entity representation by aggregating neighborhood information under different relation representations. Temporal knowledge graphs have additional temporal information, and some methods enhance the representation of entities by a time-aware mechanism.
圖神經網路 (GNN) [60] 具有強大的結構建模能力。實體可以通過 GNN 用屬性特徵和全局結構特徵來豐富其表示。典型的圖神經網路包括圖卷積網路 (GCN) [38] 和圖注意力網路 (GAT) [72]。GCN 通過聚合鄰居嵌入來獲得節點的表示，而 GAT 使用多頭注意力機制通過聚合加權鄰居嵌入來獲得節點的表示。知識圖譜是一種具有不同關係的圖。關係感知圖神經網路被開發用於學習知識圖譜中實體的表示。關係圖卷積網路 (R-GCN) [61] 是一種用於關係數據的圖神經網路模型。它學習每個關係的表示，並通過聚合不同關係表示下的鄰域資訊來獲得實體表示。時態知識圖譜具有額外的時間資訊，一些方法通過時間感知機制來增強實體的表示。

TEA-GNN [81] learns entity representations through a time-aware graph at- tention network, which incorporates relational and temporal information into the GNN structure. Specifically, it assigns different weights to different entities with orthogonal transformation matrices computed from the neighborhood's relational embeddings and temporal embeddings and obtains the entity repre- sentations by aggregating the neighborhood.
TEA-GNN [81] 透過一個時間感知的圖形注意力網路來學習實體表示，該網路將關係和時間資訊納入 GNN 結構中。具體來說，它利用從鄰域的關係嵌入和時間嵌入計算出的正交變換矩陣，為不同的實體分配不同的權重，並透過聚合鄰域來獲得實體表示。

TREA [82] learns more expressive entity representation through a temporal relational graph attention mechanism. It first maps entities, relations, and timestamps into an embedding space, then integrates entities' relational and temporal features through a temporal relational graph attention mechanism from their neighborhood, and finally, uses a margin-based log-loss to train the model and obtains the optimized representations.
TREA [82] 透過時態關係圖注意力機制學習更具表達力的實體表示。它首先將實體、關係和時間戳映射到嵌入空間，然後透過其鄰域的時態關係圖注意力機制整合實體的關係和時態特徵，最後使用基於邊界的對數損失來訓練模型並獲得優化的表示。

DEGAT [74] proposes a dynamic embedding graph attention network. It first uses the GAT to learn the static representations of entities by aggregating the features of neighbor nodes and relations, then adopts a diachronic embedding function to learn the dynamic representations of entities, and finally concate- nates the two representations and uses the ConvKB as the decoder to obtain the score.
DEGAT [74] 提出了一種動態嵌入圖注意力網路。它首先使用 GAT 透過聚合鄰居節點和關係的特徵來學習實體的靜態表示，然後採用歷時嵌入函數來學習實體的動態表示，最後將兩種表示串聯起來，並使用 ConvKB 作為解碼器來獲得分數。

T2TKG[84], or Latent relations Learning method for Temporal Knowledge Graph reasoning, is a novel approach that addresses the limitations of existing methods in explicitly capturing intra-time and inter-time latent relations for accurate prediction of future facts in Temporal Knowledge Graphs. It first employs a Structural Encoder (SE) to capture representations of entities at each timestamp, encoding their structural information. Then, it introduces a Latent Relations Learning (LRL) module to mine and exploit latent relations both within the same timestamp (intra-time) and across different timestamps (inter-time). Finally, the method fuses the temporal representations obtained from SE and LRL to enhance entity prediction tasks.
T2TKG[84]，或稱時態知識圖譜推理的潛在關係學習方法，是一種新穎的方法，旨在解決現有方法在明確捕捉時態知識圖譜中時間內和時間間潛在關係以準確預測未來事實方面的局限性。它首先採用結構編碼器 (SE) 來捕捉每個時間戳下實體的表示，對其結構資訊進行編碼。然後，它引入了一個潛在關係學習 (LRL) 模組，以挖掘和利用同一時間戳內 (時間內) 和不同時間戳之間 (時間間) 的潛在關係。最後，該方法融合從 SE 和 LRL 獲得的時間表示，以增強實體預測任務。

### 3.4. Capsule Network-based Methods
### 3.4. 基於膠囊網路的方法

CapsNet is first proposed for computer vision tasks to solve the problem that CNN needs lots of training data and cannot recognize the spatial transformation of targets. The capsule network is composed of multiple capsules, and one capsule is a group of neurons. The capsules in the lowest layer are called primary capsules, usually implemented by convolution layers to detect the presence and pose of a particular pattern (such as eyes, nose, or mouth). The capsules in the higher level are called routing capsules, which are used to detect more complex patterns (such as faces). The output of a capsule is a vector whose length represents the probability that the pattern is present and whose orientation represents the pose of the pattern.
CapsNet 最初是為了解決電腦視覺任務中 CNN 需要大量訓練資料且無法辨識目標空間變換的問題而提出的。膠囊網路由多個膠囊組成，一個膠囊是一組神經元。最底層的膠囊稱為主膠囊，通常由卷積層實現，用於偵測特定模式（例如眼睛、鼻子或嘴巴）的存在和姿態。較高層的膠囊稱為路由膠囊，用於偵測更複雜的模式（例如臉部）。膠囊的輸出是一個向量，其長度表示模式存在的機率，其方向表示模式的姿態。

CapsE [54] explores the application of capsule network in the knowledge graph. It represents a triplet as a three-column matrix in which each column represents the embedding of the head entity, relation, and tail entity, respec- tively. The matrix was fed to the capsule network, which first maps the different features of the triplet by a CNN layer, then captures the various patterns by the primary capsule layer, and finally routes the patterns to the next capsule layer to obtain the continuous output vector whose length indicates whether the triplet is valid.
CapsE [54] 探討了膠囊網路在知識圖譜中的應用。它將一個三元組表示為一個三列表，其中每一列分別表示頭實體、關係和尾實體的嵌入。該矩陣被輸入膠囊網路，該網路首先透過 CNN 層映射三元組的不同特徵，然後透過主膠囊層捕捉各種模式，最後將模式路由到下一個膠囊層以獲得連續輸出向量，其長度表示三元組是否有效。

TempCaps [18] incorporates temporal information and proposes a capsule network for temporal knowledge graph completion. The model first selects the neighbors of the head entity in a time window and obtains the embeddings of these neighbors with the capsules, then adopts a dynamic routing process to connect the lower capsules and higher capsules and gets the head entity embedding, and finally uses a multi-layer perceptron (MLP) [20] as the decoder to produce the scores of all candidate entities.
TempCaps [18] 整合了時間資訊，並提出了一種用於時態知識圖譜補全的膠囊網路。該模型首先在時間窗口內選擇頭實體的鄰居，並利用膠囊獲得這些鄰居的嵌入，然後採用動態路由過程連接較低的膠囊和較高的膠囊，得到頭實體嵌入，最後使用多層感知器 (MLP) [20] 作為解碼器，產生所有候選實體的分數。

BiQCap [85] utilizes biquaternions in hypercomplex space (Hd, q = a + bi +cj + dk, a, b, c, d ∈ Cd) and capsule networks to learn the representations of entities, relations, and timestamps. The model first represents the head entities, relations, tail entities, and timestamps using biquaternions. It then treats the timestamp as the translation of the head entity and obtains the time-specific representation (h₁ = h + r) of the head entity. Next, it rotates hī and consid- ers the rotated representation (hヶ,r = h┳⊙r) as being close to the tail entity to train the model. The score function is || h₁,rb-t||, where b is a learn- able parameter. Finally, the trained representations of entities, relations, and timestamps are fed into the capsule network to obtain the final representation.
BiQCap [85] 利用超複數空間中的雙四元數 (Hq, q = a + bi + cj + dk, a, b, c, d ∈ Cq) 和膠囊網路來學習實體、關係和時間戳的表示。該模型首先使用雙四元數表示頭實體、關係、尾實體和時間戳。然後將時間戳視為頭實體的平移，並獲得頭實體的特定時間表示 (hτ = h + τ)。接下來，它旋轉 hτ 並將旋轉後的表示 (hτ,r = hτ ⊙ r) 視為接近尾實體來訓練模型。分數函數為 || hτ,r - b - t||，其中 b 是可學習的參數。最後，將訓練好的實體、關係和時間戳的表示輸入膠囊網路以獲得最終表示。

DuCape [86] represents entities, relations, and time using dual quaternions in hypercomplex space (Hd, q = qo + qıξ, qo = ao+ boi + coj + dok, q1 = a1 + b₁i + c₁j + d₁k, ao, bo, co, do, a1, b1, C1, d₁ ∈ Rd). Dual quaternions enable modeling of both rotation and translation operations simultaneously. The model first transforms the head entity through relations and timestamps in the dual quaternion space, where the representation is close to that of the tail entity. The scoring function is ||horor-t||. The learned representations are then inputted into the capsule network to obtain the final representation.
DuCape [86] 在超複數空間中使用對偶四元數（Hq，q = q0 + q1ε，q0 = a0 + b0i + c0j + d0k，q1 = a1 + b1i + c1j + d1k，a0, b0, c0, d0, a1, b1, c1, d1 ∈ Rq）表示實體、關係和時間。對偶四元數能夠同時對旋轉和平移操作進行建模。該模型首先在對偶四元數空間中通過關係和時間戳轉換頭部實體，其中表示接近於尾部實體的表示。評分函數為 ||h ◦ r ◦ τ - t||。然後將學習到的表示輸入到膠囊網絡中以獲得最終表示。

### 3.5. Autoregression-based Methods
### 3.5. 基於自迴歸的方法

The representation learning methods based on autoregression consider that the above methods cannot model the evolution of the temporal knowledge graph over time, so they cannot predict the knowledge graph in the future. It assumes that the knowledge graph of time 7 can be inferred from the knowledge graph of last time and samples the temporal knowledge graph G according to the timestamp to obtain a series of subgraphs (or snapshots) {G1, G2, ..., GT}. Each subgraph contains the facts of the TKG at a timestamp. By modeling the subgraphs recurrently, the autoregression-based methods learn the evolutional representations of entities and relations to infer the facts GT+1 in the future.
基於自回歸的表示學習方法認為，上述方法無法對時態知識圖譜隨時間的演變進行建模，因此無法預測未來的知識圖譜。它假設時間 τ 的知識圖譜可以從上一時間的知識圖譜中推斷出來，並根據時間戳對時態知識圖譜 G 進行採樣，以獲得一系列子圖（或快照）{G1, G2, ..., GT}。每個子圖包含 TKG 在一個時間戳的事實。通過對子圖進行遞迴建模，基於自回歸的方法學習實體和關係的演化表示，以推斷未來的事實 GT+1。

RE-NET [34] proposes a recurrent event network to model the TKG for predicting future facts. It believes that the facts in Gr at timestamp T depend on the facts in the past m subgraphs GT-1:T-m before T. It first uses the R-GCN to learn the global structural representations and local neighborhood representations of the head entity at each timestamp. Then it utilizes the gated recurrent units (GRU) [11] to update the above representations and pass these representations to an MLP as a decoder to infer the facts at timestamp T. This method only models the representations of the specific entity and relation in the query triple, ignoring the structural dependency between all triples in each subgraph, which may lose some important information from the entities not in the query triple.
RE-NET [34] 提出了一種遞歸事件網路來對 TKG 進行建模以預測未來事實。它認為時間戳 T 的 Gτ 中的事實取決於 T 之前的過去 m 個子圖 GT-m:T-1 中的事實。它首先使用 R-GCN 來學習每個時間戳上頭實體的全局結構表示和局部鄰域表示。然後它利用門控遞歸單元 (GRU) [11] 來更新上述表示，並將這些表示傳遞給 MLP 作為解碼器以推斷時間戳 T 的事實。該方法僅對查詢三元組中特定實體和關係的表示進行建模，忽略了每個子圖中所有三元組之間的結構依賴性，這可能會丟失查詢三元組中未包含的實體的一些重要資訊。

Glean [14] thinks that most of the existing representation learning meth- ods use the structural information of TKG, ignoring the unstructured infor-
mation such as semantic information of words. It proposes a temporal graph neural network with heterogeneous data fusion. Specifically, it first constructs temporal event graphs based on historical facts at each timestamp and tempo- ral word graphs from event summaries at each timestamp. Then it uses the CompGCN [70] to learn the structural representations of entities and relations in the temporal event graphs and the GCN to learn the textual semantic rep- resentations of entities and relations in the temporal word graphs. Finally, it fuses the two representations and utilizes a recurrent encoder to model temporal features for final prediction.
Glean [14] 認為大多數現有的表示學習方法使用 TKG 的結構化資訊，而忽略了非結構化資訊，例如單詞的語義資訊。它提出了一種具有異構數據融合的時態圖神經網路。具體來說，它首先根據每個時間戳的歷史事實構建時態事件圖，並根據每個時間戳的事件摘要構建時態詞圖。然後它使用 CompGCN [70] 來學習時態事件圖中實體和關係的結構化表示，並使用 GCN 來學習時態詞圖中實體和關係的文本語義表示。最後，它融合了兩種表示，並利用循環編碼器對時間特徵進行建模以進行最終預測。

RE-GCN [46] splits the TKG into a sequence of KG according to the times- tamps and encodes the facts in the past m steps recurrently to predict the entities and relations in the future. This model proposes a recurrent evolution network based on a graph convolution network to model the evolutional repre- sentations by incorporating the structural dependencies among concurrent facts, the sequential patterns across temporally adjacent facts, and the static prop- erties. It uses the relation-aware GCN to capture the structural dependency and utilizes GRU to obtain the sequential pattern. Then it combines the static properties learned by R-GCN as a constraint to learn the evolutional represen- tations of entities and relations and adopts ConvTransE [62] as the decoder to predict the probability of entities and relations at next timestamp.
RE-GCN [46] 根據時間戳將 TKG 分割成一個 KG 序列，並遞迴地編碼過去 m 個步驟中的事實，以預測未來的實體和關係。該模型提出了一種基於圖卷積網路的遞迴演化網路，通過納入並行事實之間的結構依賴性、時間相鄰事實的序列模式以及靜態屬性來對演化表示進行建模。它使用關係感知 GCN 來捕捉結構依賴性，並利用 GRU 來獲得序列模式。然後，它將 R-GCN 學習到的靜態屬性作為約束來學習實體和關係的演化表示，並採用 ConvTransE [62] 作為解碼器來預測下一個時間戳的實體和關係的機率。

TiRGN [43] argues that the above methods can only capture the local his- torical dependence of the adjacent timestamps and cannot fully learn the his- torical characteristics of the facts. It proposes a time-guided recurrent graph neural network with local-global historical patterns which can model the histor- ical dependency of events at adjacent snapshots with a local recurrent encoder, the same as RE-GCN, and collect repeated historical facts by a global history encoder. The final representations are fed into a time-guided decoder named Time-ConvTransE/Time-ConvTransR to predict the entities and relations in the future.
TiRGN [43] 認為上述方法只能捕捉相鄰時間戳的局部歷史依賴性，無法充分學習事實的歷史特徵。它提出了一種帶有局部-全局歷史模式的時間引導循環圖神經網路，該網路可以使用局部循環編碼器（與 RE-GCN 相同）對相鄰快照處事件的歷史依賴性進行建模，並通過全局歷史編碼器收集重複的歷史事實。最終的表示被輸入一個名為 Time-ConvTransE/Time-ConvTransR 的時間引導解碼器，以預測未來的實體和關係。

Cen [44] believes that modeling historical facts with fixed time steps could not discover the complex evolutional patterns that vary in length. It proposes a complex evolutional network that use the evolution unit in RE-GCN as a sequence encoder to learn the representations of entities in each subgraph and utilizes the CNN as the decoder to obtain the feature maps of historical snap- shots with different length. The curriculum learning strategy is used to learn the complex evolution pattern with different lengths of historical facts from short to long and automatically select the optimal maximum length to promote the prediction.
岑 [44] 認為，用固定時間步長建模歷史事實無法發現長度可變的複雜演化模式。他提出了一個複雜的演化網路，該網路使用 RE-GCN 中的演化單元作為序列編碼器來學習每個子圖中實體的表示，並利用 CNN 作為解碼器來獲得不同長度的歷史快照的特徵圖。課程學習策略用於從短到長學習具有不同歷史事實長度的複雜演化模式，並自動選擇最佳最大長度以促進預測。

### 3.6. Temporal Point Process-based Methods
### 3.6. 基於時態點過程的方法

The autoregression-based methods sample the TKG into discrete snapshots according to a fixed time interval, which cannot effectively model the facts with irregular time intervals. Temporal point process (TPP) [12] is a stochastic process composed of a series of events in a continuous time domain. The rep- resentation learning methods based on TPP regard the TKG as a list of events changing continuously with time and formalize it as (G, O), where G is the ini- tialized TKG at time to, O is a series of observed events (h, r, t, r). At any time T > To, the TKG can be updated by the events before time T. The TPP can be characterized by conditional intensity function λ(τ). Given the historical events before a timestamp, if we can find a conditional intensity function to character- ize them, then we can Predict whether the events will occur in the future with a conditional density and when the events will occur with an expectation.
自迴歸方法根據固定時間間隔將 TKG 採樣為離散快照，無法有效模擬時間間隔不規則的事實。時態點過程 (TPP) [12] 是由連續時間域中的一系列事件組成的隨機過程。基於 TPP 的表示學習方法將 TKG 視為隨時間連續變化的事件列表，並將其形式化為 (G, O)，其中 G 是時間 t0 初始化的 TKG，O 是一系列觀察到的事件 (h, r, t, τ)。在任何時間 T > T0，TKG 都可以由時間 T 之前的事件更新。TPP 可以由條件強度函數 λ(τ) 來表徵。給定時間戳之前的歷史事件，如果我們能找到一個條件強度函數來表徵它們，那麼我們就可以用條件密度預測未來事件是否會發生，以及用期望值預測事件何時會發生。

Know-Evolve [67] combines the TPP and the deep neural network framework to model the occurrence of facts as a multi-dimensional TPP. It characterizes the TPP with the Rayleigh process and uses neural networks to simulate the intensity function. RNN is used to learn the dynamic representation of the entities, and bilinear relationship score is used to capture multiple relational interactions between entities to modulate the intensity function. Thus, it can predict whether and when the event will occur.
Know-Evolve [67] 結合 TPP 和深度神經網路框架，將事實的發生建模為多維 TPP。它使用瑞利過程來表徵 TPP，並使用神經網路來模擬強度函數。RNN 用於學習實體的動態表示，雙線性關係分數用於捕捉實體之間的多重關係交互以調節強度函數。因此，它可以預測事件是否會發生以及何時發生。

GHNN [27] believes that the Hawkes process [28] based on the neural network can effectively capture the influence of past facts on future facts, and proposes a graph Hawkes neural network (GHNN). Firstly, it solves the problem that Know-Evolve could not deal with co-occurrence facts and uses a neighborhood aggregation module to process multiple facts of entities co-occurring. Then, it utilizes the continuous-time LSTM (cLSTM) model [51] to simulate the Hawkes process to capture the evolving dynamics of the facts to implement link predic- tion and time prediction.
GHNN [27] 認為基於神經網路的霍克斯過程 [28] 能有效捕捉過去事實對未來事實的影響，並提出了一種圖霍克斯神經網路 (GHNN)。首先，它解決了 Know-Evolve 無法處理共現事實的問題，並使用鄰域聚合模組處理實體共現的多個事實。然後，它利用連續時間長短期記憶 (cLSTM) 模型 [51] 模擬霍克斯過程，以捕捉事實的不斷演變的動態，從而實現連結預測和時間預測。

EvoKG [57] argues that the above methods based on TPP lack to model the evolving network structure, and the methods based on autoregression lack to model the event time. It proposes a model jointly modeling the evolving network structure and event time. First, it uses an extended R-GCN and RNN to learn the time-evolving structural representations of entities and relations and utilizes an MLP with softmax to model the conditional probability of event triple. Then, it uses the same framework to learn the time-evolving temporal representations and adopts the TPP based on conditional density estimation with a mixture of log-normal distributions to model the event time. Finally, it jointly trains the two tasks and predicts the event and time in the future.
EvoKG [57] 認為上述基於 TPP 的方法缺乏對演化網路結構的建模，而基於自回歸的方法缺乏對事件時間的建模。它提出了一個聯合建模演化網路結構和事件時間的模型。首先，它使用擴展的 R-GCN 和 RNN 來學習實體和關係的時間演化結構表示，並利用帶有 softmax 的 MLP 來建模事件三元組的條件機率。然後，它使用相同的框架來學習時間演化時間表示，並採用基於條件密度估計的 TPP，並混合對數正態分佈來建模事件時間。最後，它聯合訓練這兩個任務，並預測未來的事件和時間。

### 3.7. Interpretability-based Methods
### 3.7. 基於可解釋性的方法

The aforementioned methods has resulted in a lack of interpretability and transparency in the generated results. As a result, we categorize interpretability- based methods as a separate category to underscore the crucial role of inter- pretability in developing reliable and transparent models. These methods aim to provide explanations for the predictions made by the models. Two pop- ular types of such methods are subgraph reasoning-based and reinforcement learning-based approaches.
前述方法在產生的結果中缺乏可解釋性和透明度。因此，我們將基於可解釋性的方法歸為一個單獨的類別，以強調可解釋性在開發可靠和透明模型中的關鍵作用。這些方法旨在為模型所做的預測提供解釋。此類方法的兩種流行類型是基於子圖推理和基於強化學習的方法。

Subgraph Reasoning xERTE [23] is a subgraph reasoning-based method that proposes an explainable reasoning framework for predicting facts in the future. It starts from the head entity in the query and utilizes a temporal rela- tional graph attention mechanism to learn the entity representation and relation representation. Then it samples the edges and temporal neighbors iteratively to construct the subgraph after several rounds of expansion and pruning. Finally, it predicts the tail entity in the subgraph.
子圖推理 xERTE [23] 是一種基於子圖推理的方法，它提出了一個用於預測未來事實的可解釋推理框架。它從查詢中的頭實體開始，並利用時態關係圖注意力機制來學習實體表示和關係表示。然後，它反覆迭代地對邊和時態鄰居進行採樣，以在幾輪擴展和修剪後構建子圖。最後，它預測子圖中的尾實體。

Reinforcement Learning Reinforcement learning (RL) [35] is usually mod- eled as a Markov Decision Process (MDP) [4], which includes a specific environ- ment and an agent. The agent has an initial state. After performing an action, it receives a reward from the environment and transitions to a new state. The goal of reinforcement learning is to find the policy network to obtain the maximum reward from all action strategies.
強化學習 強化學習 (RL) [35] 通常被建模為一個馬可夫決策過程 (MDP) [4]，其中包括一個特定的環境和一個代理。代理有一個初始狀態。在執行一個動作後，它從環境中獲得一個獎勵並轉換到一個新的狀態。強化學習的目標是找到策略網路以從所有動作策略中獲得最大獎勵。

CluSTeR [45] proposes a two-stage reasoning strategy to predict the facts in the future. First, the clue related to a given query is searched and deduced from the history based on reinforcement learning. Then, the clue at different timestamps is regarded as a subgraph related to the query, and the R-GCN and GRU are used to learn the evolving representations of entities in the subgraph. Finally, the two stages are jointly trained, and the prediction is inferred.
CluSTeR [45] 提出了一種預測未來事實的兩階段推理策略。首先，根據強化學習從歷史中搜索並推導出與給定查詢相關的線索。然後，將不同時間戳的線索視為與查詢相關的子圖，並使用 R-GCN 和 GRU 來學習子圖中實體的演化表示。最後，對這兩個階段進行聯合訓練，並推斷出預測結果。

TITer [65] directly uses the temporal path-based reinforcement learning model to learn the representations of the TKG and reasons for future facts. It adds temporal edges to connect each historical snapshot of the TKG. The agent starts from the head entity of the query, transitions to the new node ac- cording to the policy network, and searches for the answer node. The method designs a time-shaped reward based on Dirichlet distribution to guide the model learning.
TITer [65] 直接使用基於時間路徑的強化學習模型來學習 TKG 的表示，並為未來的事實進行推理。它添加時間邊來連接 TKG 的每個歷史快照。代理從查詢的頭實體開始，根據策略網路轉換到新節點，並搜索答案節點。該方法設計了一種基於狄利克雷分佈的時間形狀獎勵來引導模型學習。

### 3.8. Language Model
### 3.8. 語言模型

In the domain of TKG, the rapid development of language models has prompted researchers to explore their application for predictive tasks. The current methodologies employing language models in the TKG domain predom- inantly encompass two distinct approaches: In-Context Learning and Supervised Fine-Tuning.
在 TKG 領域，語言模型的快速發展促使研究人員探索其在預測任務中的應用。目前在 TKG 領域採用語言模型的方法主要包括兩種不同的方法：情境學習和監督式微調。

In-Context Learning ICLTKG[41] introduces a novel TKG forecasting ap- proach that leverages large language models (LLMs) through in-context learning (ICL) [5] to efficiently capture and utilize irregular patterns of historical facts for accurate predictions. The implementation algorithm of this paper involves a three-stage pipeline designed to harness the capabilities of large language mod- els (LLMs) for temporal knowledge graph (TKG) forecasting. The first stage focuses on selecting relevant historical facts from the TKG based on the pre- diction query. These facts are then used as context for the LLM, enabling it to capture temporal patterns and relationships between entities. Then the contex- tual facts are transformed into a lexical prompt that represents the prediction task. Finally, the output of the LLM is decoded into a probability distribution over the entities within the TKG. Throughout this pipeline, the algorithm con- trols the selection of background knowledge, the prompting strategy, and the decoding process to ensure accurate and efficient TKG forecasting. By lever- aging the capabilities of LLMs and harnessing the irregular patterns embedded within historical data, this approach achieves competitive performance across a diverse range of TKG benchmarks without the need for extensive supervised training or specialized architectures.
情境學習 ICLTKG[41] 提出了一種新穎的 TKG 預測方法，該方法利用大型語言模型 (LLMs) 透過情境學習 (ICL) [5] 來有效捕捉和利用歷史事實的不規則模式以進行準確預測。本文的實現演算法涉及一個三階段管道，旨在利用大型語言模型 (LLMs) 進行時態知識圖譜 (TKG) 預測。第一階段側重於根據預測查詢從 TKG 中選擇相關的歷史事實。然後將這些事實用作 LLM 的上下文，使其能夠捕捉實體之間的時間模式和關係。然後將上下文事實轉換為表示預測任務的詞彙提示。最後，將 LLM 的輸出解碼為 TKG 內實體的機率分佈。在整個管道中，該演算法控制背景知識的選擇、提示策略和解碼過程，以確保準確高效的 TKG 預測。透過利用 LLMs 的能力並利用歷史資料中嵌入的不規則模式，該方法在各種 TKG 基準測試中實現了具有競爭力的性能，而無需進行廣泛的監督訓練或專門的架構。

zrLLM[17] introduces a novel approach that leverages large language models (LLMs) to enhance zero-shot relational learning on temporal knowledge graphs (TKGs). It proposes a method to first use an LLM to generate enriched relation descriptions based on textual descriptions of KG relations, and then a second LLM is employed to generate relation representations, which capture semantic information. Additionally, a relation history learner is developed to capture tem- poral patterns in relations, further enabling better reasoning over TKGs. The zrLLM approach is shown to significantly improve the performance of TKGF models in recognizing and forecasting facts with previously unseen zero-shot relations. Importantly, zrLLM achieves this without further fine-tuning of the LLMs, demonstrating the potential of alignment between the natural language space of LLMs and the embedding space of TKGF models. Experimental results exhibit substantial gains in zero-shot relational learning on TKGs, confirming the effectiveness and adaptability of the proposed zrLLM approach.
zrLLM[17] 提出了一種新穎的方法，利用大型語言模型 (LLM) 來增強時態知識圖譜 (TKG) 上的零樣本關係學習。它提出了一種方法，首先使用 LLM 根據 KG 關係的文本描述生成豐富的關係描述，然後使用第二個 LLM 生成捕獲語義資訊的關係表示。此外，還開發了一個關係歷史學習器來捕捉關係中的時間模式，從而進一步實現對 TKG 的更好推理。zrLLM 方法被證明可以顯著提高 TKGF 模型在識別和預測具有以前未見過的零樣本關係的事實方面的性能。重要的是，zrLLM 實現了這一點，而無需對 LLM 進行進一步的微調，這表明 LLM 的自然語言空間與 TKGF 模型的嵌入空間之間存在對齊的潛力。實驗結果在 TKG 上的零樣本關係學習中表現出顯著的增益，證實了所提出的 zrLLM 方法的有效性和適應性。

Supervised Fine-Tuning ECOLA [25] proposes a joint learning model by leveraging text knowledge to enhance temporal knowledge graph represen- tations. As existing TKGs often lack fact description information, the authors construct three new datasets that contain such information. During model train- ing, they jointly optimize the knowledge-text prediction (KTP) objective and the temporal knowledge embedding (tKE) objective to improve the representa- tion of TKGs. KTP employs pre-trained language models such as transform- ers [71], while tKE can utilize an existing TKGRL model such as DyERNIE [26]. By augmenting the temporal knowledge graph representation with text descrip- tions, the model achieves significant performance gains.
監督式微調 ECOLA [25] 提出了一種聯合學習模型，利用文本知識來增強時態知識圖譜表示。由於現有的時態知識圖譜通常缺乏事實描述資訊，作者構建了三個包含此類資訊的新資料集。在模型訓練期間，他們聯合優化了知識-文本預測 (KTP) 目標和時態知識嵌入 (tKE) 目標，以改善時態知識圖譜的表示。KTP 採用了預訓練的語言模型，例如 transformers [71]，而 tKE 可以利用現有的 TKGRL 模型，例如 DyERNIE [26]。通過用文本描述來擴充時態知識圖譜表示，該模型取得了顯著的性能提升。

Frameworks such as GenTKG [47] and Chain of History [50] adopt retrieval augmented generation method for prediction. They utilize specific strategies to retrieve historical facts with high temporal relevance and logical coherence. Subsequently, these frameworks apply supervised fine-tuning language models to predict the future based on the retrieved historical facts. The input of the language model comprises historical facts and prediction queries, with the model outputting forecasted results. The authors have constructed a bespoke dataset of instructional data, which is utilized to train the language model, resulting in exemplary performance.
諸如 GenTKG [47] 和 Chain of History [50] 等框架採用檢索增強生成方法進行預測。它們利用特定的策略來檢索具有高時間相關性和邏輯連貫性的歷史事實。隨後，這些框架應用監督式微調語言模型，根據檢索到的歷史事實來預測未來。語言模型的輸入包括歷史事實和預測查詢，模型輸出預測結果。作者們構建了一個定製的指令資料集，用於訓練語言模型，從而取得了典範的性能。

### 3.9. Few-shot Learning Methods
### 3.9. 少樣本學習方法

Few-shot learning (FSL) [76] is a type of machine learning problems that deals with the problem of learning new concepts or tasks from only a few ex- amples. FSL has applications in a variety of domains, including computer vi- sion [36], natural language processing [77], and robotics [49], where data may be scarce or expensive to acquire. In TKGs, some entities and relations are only exist in a limited number of facts, and new entities and relations emerge over time. The latest TKGRL models now have the ability to perform FSL [8], which is essential for better representing these limited data. Due to the differences in handling data and learning methods for few entities and few relations, we will introduce them separately.
少樣本學習 (FSL) [76] 是一種機器學習問題，旨在從少量樣本中學習新概念或任務。FSL 在電腦視覺 [36]、自然語言處理 [77] 和機器人學 [49] 等多個領域都有應用，在這些領域，資料可能稀少或獲取成本高昂。在時態知識圖譜中，一些實體和關係僅存在於有限數量的事實中，並且隨著時間的推移會出現新的實體和關係。最新的時態知識圖譜表示學習模型現在具有執行少樣本學習的能力 [8]，這對於更好地表示這些有限的資料至關重要。由於處理少量實體和少量關係的資料和學習方法存在差異，我們將分別介紹它們。

Few Entities MetaTKG [78] reveals that new entities emerge over time in TKGs, and appear in only a few facts. Consequently, learning their representa- tion from limited historical information leads to poor performance. Therefore, the authors propose a temporal meta-learning framework to address this issue. Specifically, they first divide the TKG into multiple temporal meta-tasks, then employ a temporal meta-learner to learn evolving meta-knowledge across these tasks, finally, the learned meta-knowledge guides the backbone (which can be an existing TKGRL model, such as RE-GCN [46]) to adapt to new data.
少實體 MetaTKG [78] 揭示了隨著時間的推移，TKG 中會出現新實體，並且只出現在少數事實中。因此，從有限的歷史資訊中學習它們的表示會導致性能不佳。因此，作者提出了一個時態元學習框架來解決這個問題。具體來說，他們首先將 TKG 分為多個時態元任務，然後採用時態元學習器來學習跨這些任務不斷發展的元知識，最後，學習到的元知識指導主幹（可以是現有的 TKGRL 模型，例如 RE-GCN [46]）適應新數據。

MetaTKGR [75] confirms that emerging entities, which exist in only a small number of facts, are insufficient to learn their representations using existing models. To address this issue, the authors propose a meta temporal knowledge graph reasonging framework. The model leverages the temporal supervision signal of future facts as feedback to dynamically adjust the sampling and aggre- gation neighborhood strategy, and encoder the new entity representations. The optimized parameters can be learned via a bi-level optimization, where inner optimization initializes entity-specific parameters using the global parameters and fine-tunes them on the support set, while outer optimization operates on the query set using a temporal adaptation regularizer to stabilize meta tempo- ral reasoning over time. The learned parameters can be easily adapted to new entities.
MetaTKGR [75] 證實，新興實體僅存在於少量事實中，使用現有模型不足以學習其表示。為了解決這個問題，作者提出了一個元時態知識圖譜推理框架。該模型利用未來事實的時間監督信號作為回饋，動態調整採樣和聚合鄰域策略，並對新的實體表示進行編碼。優化的參數可以通過雙層優化來學習，其中內部優化使用全局參數初始化特定於實體的參數，並在支持集上對其進行微調，而外部優化則在查詢集上使用時間適應正則化器來穩定元時態推理。學習到的參數可以很容易地適應新的實體。

As existing datasets themselves often contain new entities that are associated with only a few facts, it is possible to directly divide the existing dataset into tasks to construct the support set and query set without requiring the creation of a new dataset.
由於現有資料集本身通常包含僅與少量事實相關聯的新實體，因此可以直接將現有資料集劃分為任務，以構建支持集和查詢集，而無需創建新資料集。

Few Relations TR-Match [22] identifies that most relations in TKGs have only a few quadruples, and new relations are added over time. Existing models are inadequate in addressing the few-shot scenario and may not fully repre- sent the evolving temporal and relational features of entities in TKGs. To ad- dress these issues, the authors propose a temporal-relational matching network. Specifically, the proposed approach incorporates a multi-scale time-relation at- tention encoder to adaptively capture local and global information based on time and relation to tackle the dynamic properties problem. A new matching processor is designed to address the few-shot problem by mapping the query to a few support quadruples in a relation-agnostic manner. To address the chal- lenge posed by few relations in temporal knowledge graphs, three new datasets, namely ICEWS14-few, ICEWS05-15-few, and ICEWS18-few, are constructed based on existing TKG datasets. The proposed TR-Match framework is evalu- ated on these datasets, and the experimental results demonstrate its capability to achieve excellent performance in few-shot relation scenarios.
少關係 TR-Match [22] 指出，時態知識圖譜中的大多數關係只有少量四元組，並且隨著時間的推移會添加新關係。現有模型不足以解決少樣本場景，並且可能無法完全表示時態知識圖譜中實體不斷發展的時間和關係特徵。為了解決這些問題，作者提出了一種時態關係匹配網路。具體來說，所提出的方法結合了一個多尺度時間關係注意力編碼器，以根據時間和關係自適應地捕獲局部和全局資訊，以解決動態屬性問題。設計了一種新的匹配處理器，通過以關係無關的方式將查詢映射到少量支持四元組來解決少樣本問題。為了解決時態知識圖譜中少關係帶來的挑戰，基於現有的時態知識圖譜資料集構建了三個新資料集，即 ICEWS14-few、ICEWS05-15-few 和 ICEWS18-few。所提出的 TR-Match 框架在這些資料集上進行了評估，實驗結果證明了其在少樣本關係場景中取得優異性能的能力。

MTKGE [10] recognizes that TKGs are subject to the emergence of un- seen entities and relations over time. To address this challenge, the authors propose a meta-learning-based temporal knowledge graph extrapolation model. The proposed approach includes a Relative Position Pattern Graph (RPPG) to construct several position patterns, a Temporal Sequence Pattern Graph (TSPG) to learn different temporal sequence patterns, and a Graph Convolu- tional Network (GCN) module for extrapolation. This model leverages meta-
learning techniques to adapt to new data and extract useful information from the existing TKG. The proposed MTKGE framework represents an important advancement in TKGRL by introducing a novel approach to knowledge graph extrapolation.
MTKGE [10] 認識到，隨著時間的推移，TKGs 會出現未見過的實體和關係。為應對這一挑戰，作者提出了一種基於元學習的時態知識圖譜外推模型。該方法包括用於構建多個位置模式的相對位置模式圖 (RPPG)、用於學習不同時間序列模式的時間序列模式圖 (TSPG) 以及用於外推的圖卷積網路 (GCN) 模組。該模型利用元學習技術來適應新數據並從現有的 TKG 中提取有用的資訊。所提出的 MTKGE 框架通過引入一種新穎的知識圖譜外推方法，代表了 TKGRL 的重要進展。

### 3.10. Other Methods
### 3.10. 其他方法

Other methods leverage the unique characteristics of TKGs to learn entity, relation, and timestamp representations. For instance, One approach explores the repetitive patterns of TKG and learns a more expressive representation us- ing the copy-generation patterns. Alternatively, other methods employ various geometric and algorithmic techniques to capture the structural properties of TKGs and learn effective representations.
其他方法利用 TKGs 的獨特特性來學習實體、關係和時間戳表示。例如，一種方法探索 TKG 的重複模式，並使用複製生成模式學習更具表現力的表示。或者，其他方法採用各種幾何和演算法技術來捕捉 TKGs 的結構特性並學習有效的表示。

Copy Generation CygNet [88] finds that many facts show repeated pat- terns along the timeline, which indicates that potential knowledge can be learned from historical facts. Therefore, it proposes a time-aware copy-generation model which can predict future facts concerning the known facts. It constructs a his- torical vocabulary with multi-hot vectors of head entity and relation in each snapshot. In copy mode, it generates an index vector with an MLP to obtain the probability of the tail entities in the historical entity vocabulary. In gener- ation mode, it uses an MLP to get the probability of tail entities in the entire entity vocabulary. After combining the two probabilities, it predicts the tail entity.
複製生成 CygNet [88] 發現，許多事實沿著時間線顯示出重複的模式，這表明可以從歷史事實中學習潛在的知識。因此，它提出了一個時間感知的複製生成模型，可以預測關於已知事實的未來事實。它在每個快照中用頭實體和關係的多熱向量構建一個歷史詞彙表。在複製模式下，它用一個 MLP 生成一個索引向量，以獲得歷史實體詞彙表中尾實體的機率。在生成模式下，它使用一個 MLP 來獲得整個實體詞彙表中尾實體的機率。在結合了兩種機率之後，它預測尾實體。

Neural Ordinary Equation TANGO [24] contends that existing approaches for modeling TKGs are inadequate in capturing the continuous evolution of TKGs over time due to their reliance on discrete state spaces. To address this limitation, TANGO proposes a novel model based on Neural Ordinary Differential Equations (NODEs). More specifically, the proposed model first employs a multi-relational graph convolutional network module to capture the graph structure information at each point in time. A graph transformation module is also utilized to model changes in edge connectivity between entities and their neighbors. The output of these modules is integrated to obtain dy- namic representations of entities and relations. Subsequently, an ODE solver is adopted to solve these dynamic representations, thereby enabling TANGO to learn continuous-time representations of TKGs. TANGO's novel approach based on ODEs offers a more effective and accurate method for modeling the dy- namic evolution of TKGs compared to existing techniques that rely on discrete representation space.
神經常微分方程 TANGO [24] 認為，由於依賴離散狀態空間，現有的 TKG 建模方法不足以捕捉 TKG 隨時間的連續演化。為了解決這一局限性，TANGO 提出了一種基於神經常微分方程 (NODE) 的新模型。更具體地說，所提出的模型首先採用多關係圖卷積網路模組來捕捉每個時間點的圖結構資訊。還利用圖變換模組來建模實體及其鄰居之間邊緣連接性的變化。這些模組的輸出被整合以獲得實體和關係的動態表示。隨後，採用 ODE 求解器來求解這些動態表示，從而使 TANGO 能夠學習 TKG 的連續時間表示。與依賴離散表示空間的現有技術相比，TANGO 基於 ODE 的新穎方法為建模 TKG 的動態演化提供了一種更有效、更準確的方法。

Geometric model The transformation models represent TKGs in a Eu- clidean space, while Dyernie [26] maps the TKGs to a non-Euclidean space and employs Riemannian manifold product to learn evolving entity representations. This approach can model multiple simultaneous non-Euclidean structures, such as hierarchical and cyclic structures, to more accurately capture the complex structural properties of TKGs. By leveraging these additional structural fea- tures, the Dyernie method can more effectively capture the relationships between entities in TKGs, resulting in improved performance on TKGRL tasks.
幾何模型 變換模型在歐幾里德空間中表示 TKG，而 Dyernie [26] 將 TKG 映射到非歐幾里德空間，並採用黎曼流形積來學習演化的實體表示。這種方法可以同時建模多個非歐幾里德結構，例如層級結構和循環結構，以更準確地捕捉 TKG 的複雜結構特性。通過利用這些額外的結構特徵，Dyernie 方法可以更有效地捕捉 TKG 中實體之間的關係，從而在 TKGRL 任務上獲得更好的性能。

BoxTE [52] is an innovative model that builds on the work of BoxE [1] and proposes a flexible and powerful framework that is fully expressive and inductive. It represents entities using a formulation that combines a base po- sition vector h, t ∈ Rd, a translational bump vector bh, bt ∈ Rd, and a time bump vector ㅜ" ∈ Rd. Specifically, the entity representations are defined as hr(h,t|r) = h + bt + Tr, tr(h,t|r) = t+ bh + τ". Furthermore, BoxTE repre- sents boxes using a time-induced relation head box rh|r = rh - Tr and a time- induced relation tail box rt| = rt -T". To determine whether a fact r(h, t|r) holds, BoxTE checks if the representations for hand t lie within their corre- sponding boxes. This is expressed mathematically as follows: hr(h,t|r) ∈ rh, and tr(h,t|r) ∈rt. By substracting ㅜ" from both sides, it obtains the equivalent expressions: hr(h,t) ∈ ph|r, tr(h,t) ∈ rt|T. BoxTE enables the model to capture rigid inference patterns and cross-time inference patterns, thereby making it a powerful tool for TKGRL.
BoxTE [52] 是一個建立在 BoxE [1] 工作基礎上的創新模型，它提出了一個靈活而強大的框架，該框架具有完全的表達性和歸納性。它使用結合了基底位置向量 h, t ∈ Rᵈ、平移碰撞向量 bh, bt ∈ Rᵈ 和時間碰撞向量 τ'' ∈ Rᵈ 的公式來表示實體。具體來說，實體表示定義為 hr(h,t|τ) = h + bt + τr, tr(h,t|τ) = t + bh + τr。此外，BoxTE 使用時間誘導的關係頭部框 rh|τ = rh - τr 和時間誘導的關係尾部框 rt|τ = rt - τr 來表示框。為了確定事實 r(h, t|τ) 是否成立，BoxTE 檢查 h 和 t 的表示是否位於其相應的框內。這在數學上表示為：hr(h,t|τ) ∈ rh，以及 tr(h,t|τ) ∈ rt。通過從兩邊減去 τ''，它得到了等價的表達式：hr(h,t) ∈ rh|τ, tr(h,t) ∈ rt|τ。BoxTE 使模型能夠捕捉剛性推理模式和跨時間推理模式，從而使其成為 TKGRL 的強大工具。

TGeomE [80] moves beyond the use of complex or hypercomplex spaces for TKGRL and instead proposes a novel geometric algebra-based embedding ap- proach. This method utilizes multivector representations and performs fourth-
order tensor factorization of TKGs while also introducing a new linear temporal regularization for time representation learning. The proposed TGeomE ap- proach can naturally model temporal relations and enhance the performance of TKGRL models.
TGeomE [80] 超越了 TKGRL 中複數或超複數空間的使用，而是提出了一種新穎的基於幾何代數的嵌入方法。該方法利用多向量表示，並對 TKG 進行四階張量分解，同時還引入了一種新的線性時間正則化來進行時間表示學習。所提出的 TGeomE 方法可以自然地對時間關係進行建模，並提高 TKGRL 模型的性能。

### 3.11. Summary
### 3.11. 摘要

In this section, we divide the TKGRL methods into ten categories and in- troduce the core technologies of these methods in detail. Table 3 shows the summary of the methods, including the representation space, the encoder for mapping the entities and relations to the vector space, and the decoder for predicting the answer.
在本節中，我們將 TKGRL 方法分為十類，並詳細介紹了這些方法的核心技術。表 3 顯示了這些方法的摘要，包括表示空間、用於將實體和關係映射到向量空間的編碼器，以及用於預測答案的解碼器。

## 4. Applications of Temporal Knowledge Graph
## 4. 時態知識圖譜的應用

By introducing temporal information, TKG can express the facts in the real world more accurately, improve the quality of knowledge graph representation learning, and answer temporal questions more reliably. It is helpful for the applications such as reasoning, entity alignment, and question answering.
通過引入時間資訊，時態知識圖譜可以更準確地表達真實世界中的事實，提高知識圖譜表示學習的品質，並更可靠地回答時間性問題。它有助於推理、實體對齊和問答等應用。

### 4.1. Temporal Knowledge Graph Reasoning
### 4.1. 時態知識圖譜推理

TKGRL methods are widely used in temporal knowledge graph reasoning (TKGR) tasks which automatically infers new facts by learning the existing facts in the KG. TKGR usually has three subtasks: entity prediction, relation prediction, and time prediction. Entity prediction is the basic task of link pre- diction, which can be expressed as two queries (?, r, t, r) and (h, r, ?, 7). Relation prediction and time prediction can be expressed as (h,?, t, r) and (h, r,t,?), re- spectively.
TKGRL 方法廣泛用於時態知識圖譜推理 (TKGR) 任務，該任務透過學習 KG 中現有事實自動推斷新事實。TKGR 通常有三個子任務：實體預測、關係預測和時間預測。實體預測是連結預測的基本任務，可表示為兩個查詢 (?, r, t, τ) 和 (h, r, ?, τ)。關係預測和時間預測可分別表示為 (h, ?, t, τ) 和 (h, r, t, ?)。

TKGR can be divided into two categories based on when the predictions of facts occur, namely interpolation and extrapolation. Suppose that a TKG is available from time to to TT. The primary objective of interpolation is to retrieve the missing facts at a specific point in time τ (το ≤ t ≤ ττ). This process is also known as temporal knowledge graph completion (TKGC). On the other hand, extrapolation aims to predict the facts that will occur in the future (T> TT) and is referred to as temporal knowledge graph forecasting.
TKGR 可根據事實預測發生的時間分為兩類，即內插和外插。假設 TKG 在時間 t0 到 tT 可用。內插的主要目標是檢索在特定時間點 τ (t0 ≤ τ ≤ tT) 缺失的事實。此過程也稱為時態知識圖譜補全 (TKGC)。另一方面，外插旨在預測未來將發生的事實 (τ > tT)，並被稱為時態知識圖譜預測。

Several methods have been proposed for Temporal Knowledge Graph Com- pletion (TKGC) including transformation-based, decomposition-based, graph neural networks-based, capsule Network-based, and other geometric methods. These techniques aim to address the problem of missing facts in TKGs by leveraging various mathematical models and neural networks. In contrast, predicting future facts in TKGs requires a different approach that can model the temporal evolution of the graph. Autoregression-based, temporal point process-based, and few-shot learning methods are commonly used for this task. Interpretability-based methods are used to increase the reliability of predic- tion results. These techniques provide evidence to support predictions, helping to establish trust and improving the overall quality of predictions made by the model. To further enhance the performance of TKGRL, semantic augmentation technology can be employed to improve the quality and quantity of semantic information of TKGs. Utilizing entity and relation names, as well as textual de- scriptions of fact associations, can enrich their representation and promote the development of downstream tasks of TKGs. In addition, large language models
已經提出了幾種用於時態知識圖譜補全 (TKGC) 的方法，包括基於變換、基於分解、基於圖神經網路、基於膠囊網路以及其他幾何方法。這些技術旨在通過利用各種數學模型和神經網路來解決 TKG 中缺失事實的問題。相比之下，預測 TKG 中的未來事實需要一種能夠對圖的時間演化進行建模的不同方法。自回歸、時態點過程和少樣本學習方法通常用於此任務。基於可解釋性的方法用於提高預測結果的可靠性。這些技術為預測提供證據，有助於建立信任並提高模型所做預測的整體品質。為了進一步提高 TKGRL 的性能，可以採用語義增強技術來提高 TKG 語義資訊的品質和數量。利用實體和關係名稱以及事實關聯的文本描述可以豐富其表示並促進 TKG 下游任務的發展。此外，大型語言模型

(LLMs) for natural language processing (NLP) can facilitate the acquisition of rich semantic information about entities and relations, further augmenting the performance of TKGRL models.
用於自然語言處理 (NLP) 的大型語言模型 (LLM) 可以促進獲取有關實體和關係的豐富語義資訊，從而進一步增強 TKGRL 模型的性能。

### 4.2. Entity Alignment Between Temporal Knowledge Graphs
### 4.2. 時態知識圖譜間的實體對齊

Entity alignment (EA) aims to find equivalent entities between different KGs, which is important to promote the knowledge fusion between multi-source and multi-lingual KGs. Defining G₁ = (E1, R1, T1, F1) and G2 = (E2, R2, T2, F2) to be two TKGs, S = {(e₁₁, e2;)|e1; ∈ E1,2; ∈ E2} is the set of alignment seeds between G1 and G2. EA seeks to find new alignment entities according to the alignment seeds S. The methods of EA between TKGS mainly adopt the GNN-based model.
實體對齊 (EA) 旨在尋找不同知識圖譜之間的等價實體，這對於促進多源和多語言知識圖譜之間的知識融合非常重要。定義 G₁ = (E₁, R₁, T₁, F₁) 和 G₂ = (E₂, R₂, T₂, F₂) 為兩個時態知識圖譜，S = {(e₁ᵢ, e₂ⱼ)|e₁ᵢ ∈ E₁, e₂ⱼ ∈ E₂} 是 G₁ 和 G₂ 之間的對齊種子集。EA 旨在根據對齊種子 S 尋找新的對齊實體。時態知識圖譜之間的 EA 方法主要採用基於 GNN 的模型。

Currently, exploring the entity alignment (EA) between Temporal Knowl- edge Graphs (TKGs) is an active area of research. TEA-GNN [81] was the first method to incorporate temporal information via a time-aware attention Graph Neural Network (GNN) to enhance EA. TREA [82] utilizes a temporal relational attention GNN to integrate relational and temporal features of enti- ties for improved EA performance. STEA [7] identifies that the timestamps in many TKGs are uniform and proposes a simple GNN-based model with a tem- poral information matching mechanism to enhance EA. Initially, the structure and relation features of an entity are fused together to generate the entity em- bedding. Then, the entity embedding is updated using GNN aggregation from neighborhood. Finally, the entity embedding is obtained by concatenating the embedding of each layer of the GNN. STEA not only updates the representation of entities but also calculates time similarity by considering associated times- tamps. The method combines both the similarities of entity embeddings and the similarities of entity timestamps to obtain aligned entities. Overall, STEA offers an effective way of improving entity representation in TKGs and provides a reliable solution for aligning entities over time.
目前，探索時態知識圖譜 (TKG) 之間的實體對齊 (EA) 是一個活躍的研究領域。TEA-GNN [81] 是第一個透過時間感知注意力圖神經網路 (GNN) 納入時間資訊以增強 EA 的方法。TREA [82] 利用時態關係注意力 GNN 來整合實體的關係和時間特徵，以提高 EA 性能。STEA [7] 發現許多 TKG 中的時間戳是統一的，並提出了一種帶有時間資訊匹配機制的簡單基於 GNN 的模型來增強 EA。最初，實體的結構和關係特徵被融合在一起以生成實體嵌入。然後，使用來自鄰域的 GNN 聚合來更新實體嵌入。最後，通過連接 GNN 各層的嵌入來獲得實體嵌入。STEA 不僅更新實體的表示，還通過考慮相關的時間戳來計算時間相似性。該方法結合實體嵌入的相似性和實體時間戳的相似性來獲得對齊的實體。總體而言，STEA 提供了一種有效的方法來改進 TKG 中的實體表示，並為隨時間推移對齊實體提供了可靠的解決方案。

### 4.3. Question Answering Over Temporal Knowledge Graphs
### 4.3. 時態知識圖譜上的問答

Question answering over KG (KGQA) aims to answer natural language ques- tions based on KG. The answer to the question is usually an entity in the KG. In order to answer the question, one-hop or multi-hop reasoning is required on the KG. Question answering over TKG (TKGQA) aims to answer temporal natural language questions based on TKG, the answer to the question is entity or timestamp in the TKG, and the reasoning on TKG is more complex than it on KG.
基於知識圖譜的問答 (KGQA) 旨在根據知識圖譜回答自然語言問題。問題的答案通常是知識圖譜中的一個實體。為了回答問題，需要在知識圖譜上進行單跳或多跳推理。基於時態知識圖譜的問答 (TKGQA) 旨在根據時態知識圖譜回答時間性自然語言問題，問題的答案是時態知識圖譜中的實體或時間戳，而時態知識圖譜上的推理比知識圖譜上的推理更複雜。

Research on TKGQA is in progress. CRONKGQA [59] release a new dataset named CRONQUESTIONS and propose a model combining representation of TKG and question for TKGQA. It first uses TComplEx to obtain the represen- tation of entities and timestamps in the TKG, and utilizes BERT [15] to obtain their representations in the question, then calculates the scores of all entities and times, and finally concatenated the score vectors to obtain the answer.
TKGQA 的研究正在進行中。CRONKGQA [59] 發布了一個名為 CRONQUESTIONS 的新資料集，並提出了一個結合 TKG 和問題表示的模型用於 TKGQA。它首先使用 TComplEx 來獲取 TKG 中實體和時間戳的表示，並利用 BERT [15] 來獲取它們在問題中的表示，然後計算所有實體和時間的分數，最後將分數向量連接起來以獲得答案。

TSQA [63] argues existing TKGQA methods haven't explore the implicit temporal feature in TKGs and temporal questions. It proposes a time sen- sitive question answering model which consists of a time-aware TKG encoder and a time-sensitive question answering module. The time-aware TKG encoder uses TComplex with time-order constraints to obtain the representations of entities and timestamps. The time-sensitive question answering module first decomposes the question into entities and a temporal expression. It uses the entities to extract the neighbor graph to reduce the search space of timestamps and answer entities. The temporal expression is fed into the BERT to learn the temporal question representations. Finally, entity and temporal question representations are combined to estimate the time and predict the entity with contrastive learning.
TSQA [63] 認為現有的 TKGQA 方法沒有探索 TKG 和時間問題中的隱含時間特徵。它提出了一個時間敏感的問答模型，該模型由一個時間感知的 TKG 編碼器和一個時間敏感的問答模組組成。時間感知的 TKG 編碼器使用帶有時間順序約束的 TComplEx 來獲得實體和時間戳的表示。時間敏感的問答模組首先將問題分解為實體和時間表達式。它使用實體來提取鄰居圖以減少時間戳和答案實體的搜索空間。時間表達式被饋送到 BERT 中以學習時間問題表示。最後，將實體和時間問題表示結合起來以估計時間並使用對比學習來預測實體。

## 5. Future Directions
## 5. 未來方向

Despite the significant progress made in TKGRL research, there remain sev- eral important future directions. These include addressing scalability challenges, improving interpretability, incorporating information from multiple modalities, and leveraging large language models to enhance the ability of representing dynamic and evolving TKGs.
儘管 TKGRL 研究取得了重大進展，但仍有幾個重要的未來方向。其中包括應對可擴展性挑戰、提高可解釋性、整合來自多種模態的資訊，以及利用大型語言模型來增強表示動態和不斷發展的 TKG 的能力。

### 5.1. Scalability
### 5.1. 可擴展性

The current datasets available for TKG are insufficient in size compared to real-world knowledge graphs. Moreover, TKGRL methods tend to prioritize improving task-specific performance and often overlook the issue of scalability. Therefore, there is a pressing need for research on effective methods of learning TKG representations that can accommodate the growing demand for data. A possible avenue for future research in this field is to investigate various strategies for enhancing the scalability of TKGRL models.
與真實世界的知識圖譜相比，目前可用於 TKG 的資料集在規模上不足。此外，TKGRL 方法傾向於優先提高特定任務的性能，而常常忽略可擴展性問題。因此，迫切需要研究學習 TKG 表示的有效方法，以適應日益增長的資料需求。該領域未來研究的一個可能途徑是研究各種增強 TKGRL 模型可擴展性的策略。

One approach for improving the scalability of TKGRL models is to employ distributed computing techniques, such as parallel processing or distributed training, to enable more efficient processing of large-scale knowledge graphs. Parallel processing involves partitioning the dataset into smaller subsets and processing each subset simultaneously. In contrast, distributed training trains the model on various machines concurrently, with the outcomes combined to en- hance the overall accuracy of the model. This approach could prove especially beneficial for real-time processing of extensive knowledge graphs in applications that require quick response times.
提升 TKGRL 模型可擴展性的一種方法是採用分佈式計算技術，例如並行處理或分佈式訓練，以實現對大規模知識圖譜更高效的處理。並行處理涉及將資料集劃分為更小的子集並同時處理每個子集。相反，分佈式訓練在多台機器上同時訓練模型，並將結果結合起來以增強模型的整體準確性。這種方法對於需要快速響應時間的應用中對廣泛知識圖譜的實時處理尤其有益。

Another approach is to use sampling techniques to reduce the size of the knowledge graph without sacrificing accuracy. For example, researchers could use clustering algorithms to identify groups of entities and events that are highly interconnected and then sample a representative subset of these groups for train- ing the model. This approach could help to reduce the computational complex- ity of the model without sacrificing accuracy. Sampling techniques can also be used for negative sampling in TKGRL. Negative sampling involves selecting negative samples that are not present in the knowledge graph to balance out the positive samples during training. By employing efficient negative sampling techniques, researchers can significantly reduce the computational complexity of the TKGRL model while maintaining high accuracy levels.
另一種方法是使用抽樣技術來縮減知識圖譜的大小，而不會犧牲準確性。例如，研究人員可以使用聚類演算法來識別高度相互關聯的實體和事件群組，然後對這些群組的代表性子集進行抽樣以訓練模型。這種方法有助於降低模型的計算複雜性，而不會犧牲準確性。抽樣技術也可用於 TKGRL 中的負抽樣。負抽樣涉及選擇知識圖譜中不存在的負樣本，以平衡訓練期間的正樣本。通過採用高效的負抽樣技術，研究人員可以顯著降低 TKGRL 模型的計算複雜性，同時保持高準確性水平。

Overall, addressing issues related to scalability will be critical for advancing the state-of-the-art in temporal knowledge graph research and enabling practical applications in real-world scenarios.
總體而言，解決與可擴展性相關的問題對於推進時態知識圖譜研究的最新技術並在實際場景中實現實際應用至關重要。

### 5.2. Interpretability
### 5.2. 可解釋性

The enhancement of interpretability is a crucial research direction, as it allows for better understanding of how model outputs are derived and ensures the reliability and applicability of the model's results. Despite the availability of interpretable methods, developing more interpretable models and techniques for temporal knowledge graphs remains a vital research direction.
可解釋性的增強是一個至關重要的研究方向，因為它有助於更好地理解模型輸出的推導過程，並確保模型結果的可靠性和適用性。儘管已有可解釋的方法，但為時態知識圖譜開發更具可解釋性的模型和技術仍然是一個重要的研究方向。

One promising approach involves incorporating attention mechanisms to identify the most relevant entities and events in the knowledge graph at dif- ferent points in time. This approach would allow users to understand which parts of the graph are most important for a given prediction, which could im- prove the interpretability of the model.
一種有前景的方法是納入注意力機制，以識別知識圖譜中不同時間點最相關的實體和事件。這種方法將使用戶能夠了解圖形的哪些部分對於給定的預測最重要，從而提高模型的可解釋性。

In addition, researchers could explore the use of visualization techniques to help users understand the structure and evolution of the knowledge graph over time. For example, interactive visualizations could enable users to explore the graph and understand how different entities and events are connected.
此外，研究人員可以探索使用視覺化技術來幫助使用者理解知識圖譜隨時間的結構和演變。例如，互動式視覺化可以讓使用者探索圖譜並理解不同實體和事件之間的關聯。

By making TKGRL more interpretable, we can gain greater insights into complex real-world phenomena, support decision-making processes, and ensure that these models are useful for practical applications.
通過使 TKGRL 更具可解釋性，我們可以更深入地了解複雜的現實世界現象，支持決策過程，並確保這些模型可用於實際應用。

### 5.3. Information Fusion
### 5.3. 資訊融合

Most TKGRL methods only utilize the structural information of TKGs, with few models incorporating textual information of entities and relations. However, text data contains rich features that can be leveraged to enhance TKGs' representation. Therefore, effectively fusing various features of TKGs, including entity feature, relation feature, time feature, structure feature and textual feature, represents a promising future research direction.
大多數 TKGRL 方法僅利用 TKG 的結構資訊，很少有模型納入實體和關係的文本資訊。然而，文本資料包含豐富的特徵，可用於增強 TKG 的表示。因此，有效融合 TKG 的各種特徵，包括實體特徵、關係特徵、時間特徵、結構特徵和文本特徵，代表了一個有前途的未來研究方向。

One approach to information fusion in TKGRL is to use multi-modal data sources. For example, researchers can combine textual data, such as news ar- ticles or social media posts, with structured data from knowledge graphs to improve the accuracy of the model. This approach can help the TKGRL model to capture more relationships between entities and events that may not be ap- parent from structured data alone.
TKGRL 中資訊融合的一種方法是使用多模態資料來源。例如，研究人員可以將文本資料（如新聞文章或社交媒體貼文）與知識圖譜中的結構化資料相結合，以提高模型的準確性。這種方法可以幫助 TKGRL 模型捕捉實體和事件之間更多僅從結構化資料中可能不明顯的關係。

Another approach is to use attention mechanisms to dynamically weight the importance of different sources of information at different points in time. This approach would allow the model to focus on the most relevant information for a given prediction, which could improve the accuracy of the model while reducing computational complexity.
另一種方法是使用注意力機制，在不同時間點動態加權不同資訊來源的重要性。這種方法將允許模型專注於給定預測最相關的資訊，從而提高模型的準確性，同時降低計算複雜度。

In general, information fusion is a powerful tool in TKGRL that can help researchers improve the accuracy and reliability of the model by combining information from multiple sources. However, it is essential to carefully weigh the benefits and costs of using different fusion techniques, depending on the specific dataset and research goals.
總體而言，資訊融合是 TKGRL 中一個強大的工具，可以幫助研究人員通過結合來自多個來源的資訊來提高模型的準確性和可靠性。然而，根據具體的資料集和研究目標，仔細權衡使用不同融合技術的利弊至關重要。

### 5.4. Incorporating Large Language Models
### 5.4. 整合大型語言模型

Recent advances in natural language processing, such as the development of large language models (LLMs) [87] has been largely advanced by both academia and industry. A notable achievement in the field of LLMs is the introduction of ChatGPT 1, a highly advanced AI chatbot. Developed using LLMs, ChatGPT has generated significant interest and attention from both the research commu- nity and society at large. ChatGPT uses the generative pre-trained transformer (GPT) such as GPT-4 [56], have led to significant improvements in various nat- ural language tasks. LLMs have been shown to be highly effective at capturing complex semantic relationships between words and phrases, and they may be able to provide valuable insights into the meaning and context of entities and relations in a knowledge graph. Efficiently combining LLMs with TKGRL is an novel research direction for the future.
自然語言處理的最新進展，例如大型語言模型 (LLM) [87] 的發展，已在學術界和工業界取得了長足的進步。LLM 領域的一個顯著成就是推出了 ChatGPT 1，這是一款高度先進的人工智慧聊天機器人。使用 LLM 開發的 ChatGPT 引起了研究界和整個社會的極大興趣和關注。ChatGPT 使用生成式預訓練變壓器 (GPT)，例如 GPT-4 [56]，已導致各種自然語言任務的顯著改進。LLM 已被證明在捕捉單詞和短語之間複雜的語義關係方面非常有效，它們可能能夠為知識圖譜中實體和關係的含義和上下文提供有價值的見解。有效地將 LLM 與 TKGRL 相結合是未來一個新穎的研究方向。

One approach to incorporating LLMs into TKGRL is to use LLMs to gener- ate embeddings for entities and relations. These embeddings could be used as input to a TKGRL model, enabling it to capture more rich feature of entities and relations over time.
將大型語言模型 (LLM) 納入時態知識圖譜表示學習 (TKGRL) 的一種方法是使用 LLM 為實體和關係生成嵌入。這些嵌入可用作 TKGRL 模型的輸入，使其能夠隨著時間的推移捕捉實體和關係更豐富的特徵。

Another potential approach is to use LLMs to generate textual descriptions of entities and facts in the TKGs. These descriptions could be used to enrich the TKGs with additional semantic information, which could then be used to improve the accuracy of predictions.
另一種潛在的方法是使用大型語言模型（LLM）為 TKG 中的實體和事實生成文本描述。這些描述可用於豐富 TKG 的附加語義資訊，從而提高預測的準確性。

Aboveall, incorporating LLMs into TKGRL has the potential to significantly improve the accuracy and effectiveness of these models, and it is an exciting area for future research. However, it is essential to carefully consider the challenges and limitations of using LLMs, such as computational complexity and potential bias in the pre-trained data.
總而言之，將大型語言模型（LLM）納入時態知識圖譜表示學習（TKGRL）有潛力顯著提高這些模型的準確性和有效性，這是一個令人興奮的未來研究領域。然而，仔細考慮使用 LLM 的挑戰和局限性至關重要，例如計算複雜性和預訓練數據中的潛在偏差。

## 6. Conclusion
## 6. 結論

Temporal knowledge graphs (TKGs) provide a powerful framework for rep- resenting and analyzing complex real-world phenomena that evolve over time. Temporal knowledge graph representation learning (TKGRL) is an active area of research that investigates methods for automatically extracting meaningful representations from TKGs.
時態知識圖譜（TKG）提供了一個強大的框架，用於表示和分析隨時間演變的複雜現實世界現象。時態知識圖譜表示學習（TKGRL）是一個活躍的研究領域，旨在研究從 TKG 中自動提取有意義表示的方法。

In this paper, we provide a detailed overview of the development, methods, and applications of TKGRL. We begin by the definition of TKGRL and dis- cussing the datasets and evaluation metrics commonly used in this field. Next, we categorize TKGRL methods based on their core technologies and analyze the fundamental ideas and techniques employed by each category. Furthermore, we offer a comprehensive review of various applications of TKGRL, followed by a discussion of future directions for research in this area. By focusing on these areas, we can continue to drive advancements in TKGRL and enable practical applications in real-world scenarios.
在本文中，我們詳細概述了 TKGRL 的發展、方法和應用。我們首先定義了 TKGRL，並討論了該領域常用的資料集和評估指標。接下來，我們根據 TKGRL 方法的核心技術對其進行分類，並分析每個類別所採用的基本思想和技術。此外，我們對 TKGRL 的各種應用進行了全面回顧，隨後討論了該領域的未來研究方向。通過專注於這些領域，我們可以繼續推動 TKGRL 的進步，並在現實世界場景中實現實際應用。

## Acknowledgements
## 誌謝

We appreciate the support from National Natural Science Foundation of China with the Main Research Project on Machine Behavior and Human-Machine Collaborated Decision Making Methodology (72192820 & 72192824), Pudong New Area Science & Technology Development Fund (PKX2021-R05), Science and Technology Commission of Shanghai Municipality (22DZ-2229004), and Shanghai Trusted Industry Internet Software Collaborative Innovation Center.
我們感謝國家自然科學基金委「機器行為與人機協同決策方法論」重大研究計畫（72192820 & 72192824）、浦東新區科技發展基金（PKX2021-R05）、上海市科學技術委員會（22DZ-2229004）以及上海可信工業互聯網軟體協同創新中心的資助。

## References
## 參考文獻
