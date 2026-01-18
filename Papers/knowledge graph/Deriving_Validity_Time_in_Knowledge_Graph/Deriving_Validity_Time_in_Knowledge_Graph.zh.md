---
title: Deriving_Validity_Time_in_Knowledge_Graph
field: Knowledge_Graph
status: Imported
created_date: 2026-01-14
pdf_link: "[[Deriving_Validity_Time_in_Knowledge_Graph.pdf]]"
tags:
  - knowledge_graph
---
# 推導知識圖譜中的有效時間

# Deriving Validity Time in Knowledge Graph

**Julien Leblay**
Artificial Intelligence Research Center
AIST Tokyo Waterfront
Tokyo, Japan
firstname.lastname@aist.go.jp

**Julien Leblay**
人工智慧研究中心
AIST 東京臨海
日本東京
firstname.lastname@aist.go.jp

**Melisachew Wudage Chekol**
Data and Web Science Group
University of Mannheim
Mannheim, Germany
mel@informatik.uni-mannheim.de

**Melisachew Wudage Chekol**
數據與網絡科學小組
曼海姆大學
德國曼海姆
mel@informatik.uni-mannheim.de

### ABSTRACT

Knowledge Graphs (KGs) are a popular means to represent knowledge on the Web, typically in the form of node/edge labelled directed graphs. We consider temporal KGs, in which edges are further annotated with time intervals, reflecting when the relationship between entities held in time. In this paper, we focus on the task of predicting time validity for unannotated edges. We introduce the problem as a variation of relational embedding. We adapt existing approaches, and explore the importance of example selection and the incorporation of side information in the learning process. We present our experimental evaluation in details.

### 摘要

知識圖譜 (KGs) 是一種在網絡上表示知識的流行方式，通常以節點/邊標記的有向圖形式呈現。我們考慮的是時態知識圖譜，其中邊被進一步標註時間間隔，以反映實體之間的關係在何時成立。在本文中，我們專注於預測未標註邊的時間有效性。我們將這個問題作為關聯嵌入的一種變體來介紹。我們調整了現有方法，並探討了樣本選擇和在學習過程中納入輔助資訊的重要性。我們詳細介紹了我們的實驗評估。

### CCS CONCEPTS

• **Computing methodologies** → Temporal reasoning; Supervised learning;

### CCS 概念

• **計算方法學** → 時態推理；監督式學習；

### KEYWORDS

Temporal Knowledge Graph, Factorization Machines

### 關鍵詞

時態知識圖譜，分解機

### ACM Reference Format:

Julien Leblay and Melisachew Wudage Chekol. 2018. Deriving Validity Time in Knowledge Graph. In WWW ’18 Companion: The 2018 Web Conference Companion, April 23–27, 2018, Lyon, France. ACM, New York, NY, USA, 6 pages. https://doi.org/10.1145/3184558.3191639

### ACM 參考格式：

Julien Leblay and Melisachew Wudage Chekol. 2018. Deriving Validity Time in Knowledge Graph. In WWW ’18 Companion: The 2018 Web Conference Companion, April 23–27, 2018, Lyon, France. ACM, New York, NY, USA, 6 pages. https://doi.org/10.1145/3184558.3191639

## 1 INTRODUCTION

Knowledge Graphs (KGs) encompass a class of knowledge representation models, in which nodes correspond to entities, and directed labelled edges the relationships between them. Some well-known examples of KGs include Google’s Knowledge Vault [5], NELL [4], YAGO [6], and DBpedia [1]. Whether the data is generated and maintained by users or computer programs, mistakes and omissions can easily proliferate, and the data can quickly become outdated. To make matters worse, some of the most popular formats used for data publishing, including RDF, JSON or CSV, do not provide built-in mechanisms to easily capture and retain information as the data changes over time. As an example, consider the following facts extracted from the DBpedia (http://dbpedia.org/page/Grover_Cleveland) dataset about Grover Cleveland, the 22th and 24th president of the USA.

## 1 緒論

知識圖譜 (KGs) 包含一類知識表示模型，其中節點對應於實體，有向標記邊則對應於它們之間的關係。一些著名的 KG 例子包括谷歌的知識庫 (Knowledge Vault) [5]、NELL [4]、YAGO [6] 和 DBpedia [1]。無論數據是由用戶還是電腦程式生成和維護，錯誤和遺漏都很容易增生，數據也可能很快過時。更糟的是，一些最流行的數據發布格式，包括 RDF、JSON 或 CSV，沒有內建機制來輕易地捕捉和保留隨時間變化的資訊。舉例來說，請參考以下從 DBpedia (http://dbpedia.org/page/Grover_Cleveland) 數據集中提取的關於美國第 22 屆和第 24 屆總統格羅弗·克利夫蘭的事實。

∗Dr. Leblay’s work is supported by the KAKENHI grant number 17K12786.

∗Leblay 博士的研究由 KAKENHI 撥款號 17K12786 支持。

This paper is published under the Creative Commons Attribution 4.0 International (CC BY 4.0) license. Authors reserve their rights to disseminate the work on their personal and corporate Web sites with the appropriate attribution.
WWW ’18 Companion, April 23–27, 2018, Lyon, France
© 2018 IW3C2 (International World Wide Web Conference Committee), published under Creative Commons CC BY 4.0 License.
ACM ISBN 978-1-4503-5640-4/18/04.
https://doi.org/10.1145/3184558.3191639

本文根據知識共享署名 4.0 國際 (CC BY 4.0) 許可協議發布。作者保留在個人和公司網站上傳播該作品的權利，並附帶適當的署名。
WWW ’18 Companion, April 23–27, 2018, Lyon, France
© 2018 IW3C2 (International World Wide Web Conference Committee), published under Creative Commons CC BY 4.0 License.
ACM ISBN 978-1-4503-5640-4/18/04.
https://doi.org/10.1145/3184558.3191639

(GCleveland, birthPlace, Caldwell),
(GCleveland, office, POTUS),
(GCleveland, office, NewYork_Governor)

(GCleveland, birthPlace, Caldwell),
(GCleveland, office, POTUS),
(GCleveland, office, NewYork_Governor)

The lack of temporal information is problematic in this example for several reasons. None of these facts is independently false, yet Grover Cleveland could not have been president and governor at the same time. Moreover, this information is missing since Grover Cleveland has been president twice, during two non-consecutive periods. So, clearly temporal metadata would lift some ambiguity, yet not all facts typically need such metadata. For instance, his birth place is not expected to change over time.

在這個例子中，缺乏時間資訊會引發幾個問題。這些事實本身沒有一個是錯誤的，但格羅弗·克利夫蘭不可能同時擔任總統和州長。此外，這方面的資訊是缺失的，因為格羅弗·克利夫蘭曾兩次在兩個不連續的任期內擔任總統。因此，時間元數據顯然可以消除一些模糊性，但並非所有事實通常都需要這樣的元數據。例如，他的出生地預計不會隨時間改變。

Many KGs do not contain the validity period of facts, i.e., the period during which the fact is considered to hold. Notable exceptions include Wikidata [20] and YAGO, in which some facts that are endowed with temporal information. Our goal is to learn temporal meta-data on a knowledge graph where such information is incomplete. For the above example, we want to derive annotations of the following form:
(GCleveland, office, POTUS):[1885-1889;1893-1897]
(GCleveland, office, NewYork_Governor):[1883-1885]
Note that Grover Cleveland was president during two distinct, non consecutive terms.

許多知識圖譜未包含事實的有效期間，也就是事實被認為成立的期間。值得注意的例外包括 Wikidata [20] 和 YAGO，其中部分事實具備時間資訊。我們的目標是在資訊不完整的知識圖譜上學習時間元數據。對於上述範例，我們希望推導出以下形式的註解：
(GCleveland, office, POTUS):[1885-1889;1893-1897]
(GCleveland, office, NewYork_Governor):[1883-1885]
請注意，格羅弗·克利夫蘭在兩個不同且不連續的任期內擔任總統。

In the following section, we provide some formal background and review the related work. In Section 3, we first attempt to carry over techniques from relational embedding models, and study the limitations of these approaches. Then, we proceed to show that factorization machines are particularly well-suited for our temporal scope prediction task, allowing to take valuable side-information into account. In Section 4, we report early experimental results.

在下一節中，我們提供了一些正式的背景知識並回顧了相關工作。在第 3 節中，我們首先嘗試從關係嵌入模型中轉移技術，並研究這些方法的局限性。然後，我們接著說明分解機特別適用於我們的時間範圍預測任務，允許將有價值的輔助資訊納入考量。在第 4 節中，我們報告了早期的實驗結果。

## 2 PRELIMINARIES

In the following, we introduce temporal knowledge graphs formally, as well as the problem addressed in this paper. We present possible extensions of relational embedding approaches and factorization machines.

## 2 預備知識

接下來，我們將正式介紹時態知識圖譜以及本文所要解決的問題。我們將介紹關係嵌入方法和分解機的可能擴展。

### 2.1 Temporal Knowledge Graphs

We are considering KGs of the form G = (E, R), where E is a set of labeled nodes known as entities, and R is a set of labeled edges known as relations. Alternatively, we can refer to G, as a set of triples of the form (subject, predicate, object), where subject and object are node-labels, and predicate is an edge label. Labels act as unique identifiers for subjects and predicates, and either as identifier or literal value for objects. Hence, the presence of an edge p between two nodes s and o indicates that the fact (s, p, o) holds. In practice, knowledge is not static in time, thus we would like to capture when a given fact held over time. Thus, we assume a set of discrete time points T, and an additional labeling scheme on edges, which takes a set of time intervals over T, denoting the periods within which a fact was considered true. This yields a temporal KG.

### 2.1 時態知識圖譜

我們考慮形式為 G = (E, R) 的知識圖譜，其中 E 是稱為實體的標記節點集合，R 是稱為關係的標記邊集合。或者，我們可以將 G 視為一組 (subject, predicate, object) 形式的三元組，其中 subject 和 object 是節點標籤，而 predicate 是邊標籤。標籤作為 subject 和 predicate 的唯一標識符，並作為 object 的標識符或文字值。因此，在兩個節點 s 和 o 之間存在一條邊 p 表示事實 (s, p, o) 成立。在實踐中，知識並非靜態不變，因此我們希望能捕捉特定事實隨時間變化的情況。因此，我們假設存在一組離散的時間點 T，以及一個附加在邊上的標籤方案，該方案取 T 上的一組時間區間，表示一個事實被認為是真實的時期。這就產生了一個時態知識圖譜。

### 2.2 Problem statement

Our goal is to learn associations between facts of a KG and one or more time points in T. This gives us the ability to tackle the following tasks:

**Time prediction:** given a query of the form (s, p, o, ?), predict the time point(s) for which the fact is consider valid/true.

**Time-dependent query answering:** given a point in time and a fact with missing subject, predicate or object, predict the most likely label.

### 2.2 問題陳述

我們的目標是學習知識圖譜的事實與 T 中一個或多個時間點之間的關聯。這使我們能夠處理以下任務：

**時間預測：** 給定一個 (s, p, o, ?) 形式的查詢，預測該事實被視為有效的一個或多個時間點。

**時間相關查詢回答：** 給定一個時間點和一個缺少主詞、謂詞或受詞的事實，預測最可能的標籤。

### 2.3 Related Work

We present the related work from three different angles: (i) temporal scoping of knowledge graph facts, (ii) relational embedding for link prediction, and (iii) factorization machines for triple classification.

### 2.3 相關工作

我們從三個不同角度介紹相關工作：（i）知識圖譜事實的時間範圍界定，（ii）用於連結預測的關係嵌入，以及（iii）用於三元組分類的分解機。

#### 2.3.1 Temporal scoping of KG facts.

The study of deriving the temporal scopes of KG facts has recently gained momentum. The most recent of which is Know-Evolve [19]. A temporal KG in KnowEvolve is a set of facts where each fact has a timestamped relation. For embedding entities and timestamped relations, they use a bilinear model (RESCAL) and employ a deep recurrent neural network in order to learn non-linearly evolving entities. The learning phase espouses a point-process, by which the estimation of whether a fact hold at time t is based on the state at time t − 1. That said, they do not exploit side information as we do in this work. Another closely related work is the time-aware KG embedding model of Jiang et al. [7]. They focus on the prediction of an entity or relation given a time point in which the fact is supposed to be valid. Both Know-Evolve and time-aware KG completion methods use relational embedding models which are discussed below. Furthermore, in [18], the authors use tensor decomposition to assign validity scopes for KG facts. However, as reported in the paper, their models do not perform sufficiently well. Nonetheless, this can be improved by including side information as we did here.
In contrast, Rula et al. [14] extract time information contained in Web pages using syntactic rules. This process has three phases whereby candidate intervals for facts are matched, selected and then merged according to temporal consistency rules. YAGO [6] is another earlier example, in which both time and space scopes were extracted using linguistic extraction rules, followed by conflict resolving post-processing.
In [21], the authors formulate the temporal scoping problem as a state change detection problem. In doing so, they enrich temporal profiles of entities with relevant contextual information (these are unigrams and bigrams surrounding mentions of an entity, for instance, for the entity Barack Obama relevant unigrams include ‘elect’, ‘senator’ and so on). From there, they learn vectors that reflect change patterns in the contexts. For example, after ‘becoming president’, US presidents often see a drop in mentions of their previous job title state such as ‘senator’ or ‘governor’ in favor of ‘president’.
Another temporal scoping system developed by [15] relies on a language model consisting of patterns automatically derived from Wikipedia sentences that contain the main entity of a page and temporal slot-fillers extracted from the corresponding infoboxes.
Talukdar et al. [17] use frequency counts of fact mentions to define temporal profiles (basically a time-series of the occurrences of facts over time in a corpus of historical documents) of facts and analyze how the mentions of those facts rise and fall over time. They identify temporal scope over input facts, using a 3-phase procedure. Yet, the approach is rather brittle in that it does not automatically adapt to new relations, and requires human experts at several steps in the process.
Bader et al. [2] used matrix decomposition on the Enron email dataset, to estimate relationship among the scandal’s stakeholders over time. Unlike in our settings, the relationships were not labeled.

#### 2.3.1 知識圖譜事實的時間範圍界定。

推導知識圖譜事實的時間範圍的研究最近獲得了發展動力。其中最新的是 Know-Evolve [19]。在 Know-Evolve 中，一個時態知識圖譜是一組事實，其中每個事實都有一個帶時間戳的關係。為了嵌入實體和帶時間戳的關係，他們使用雙線性模型 (RESCAL) 並採用深度循環神經網絡，以學習非線性演化的實體。學習階段採用點過程，藉此根據 t-1 時刻的狀態來估計一個事實是否在 t 時刻成立。話雖如此，他們不像我們在這項工作中那樣利用輔助資訊。另一個密切相關的工作是 Jiang 等人 [7] 的時間感知知識圖譜嵌入模型。他們專注於在事實應該有效的給定時間點預測實體或關係。Know-Evolve 和時間感知知識圖譜完成方法都使用關係嵌入模型，這將在下面討論。此外，在 [18] 中，作者使用張量分解為知識圖譜事實分配有效範圍。然而，正如論文中所報導的，他們的模型表現不夠好。儘管如此，這可以通過像我們在這裡所做的那樣包含輔助資訊來改進。
相比之下，Rula 等人 [14] 使用句法規則從網頁中提取時間資訊。此過程分為三個階段，即匹配、選擇事實的候選區間，然後根據時間一致性規則進行合併。YAGO [6] 是另一個早期的例子，其中時間和空間範圍都是使用語言提取規則提取的，然後進行衝突解決後處理。
在 [21] 中，作者將時間範圍問題描述為狀態變化檢測問題。為此，他們通過相關的上下文資訊來豐富實體的時間剖析（這些是圍繞實體提及的 unigrams 和 bigrams，例如，對於實體巴拉克·歐巴馬，相關的 unigrams 包括「選舉」、「參議員」等等）。然後，他們從中學習反映上下文中變化模式的向量。例如，在「成為總統」之後，美國總統們通常會看到對他們以前的工作頭銜狀態（如「參議員」或「州長」）的提及減少，而 لصالح 「總統」。
另一個由 [15] 開發的時間範圍系統依賴於一個語言模型，該模型由從包含頁面主要實體的維基百科句子和從相應資訊框中提取的時間槽填充詞自動導出的模式組成。
Talukdar 等人 [17] 使用事實提及的頻率計數來定義事實的時間剖析（基本上是事實發生在歷史文獻語料庫中隨時間變化的時間序列），並分析這些事實的提及如何隨時間起伏。他們使用三階段程序來識別輸入事實的時間範圍。然而，這種方法相當脆弱，因為它不能自動適應新的關係，並且在過程的幾個步驟中需要人類專家的參與。
Bader 等人 [2] 在安隆電子郵件數據集上使用矩陣分解，以估計醜聞利益相關者之間的關係隨時間的變化。與我們的設置不同，這些關係沒有被標記。

#### 2.3.2 Relational Embedding approaches.

Our problem is more generally related to relational embedding models, a paradigm of relational learning in low dimensional vector space, which has been widely used for tasks such as link prediction and fact classification. Such embeddings can be viewed as a special case of graph embedding, a very active research topic, which we omitted here for conciseness. We can broadly divide the models into three categories based on: (i) translational distance, (ii) tensor factorization (bilinear models), and more recently, (iii) neural networks. Vectors are used to learn entity and relation embeddings in translational models, whereas additional matrices are used in the case of bilinear models and neural networks. While the translational models use a distance metric to measure the plausability of facts, bilinear models rely on the dot product of entity and relational embeddings. One of the most well known translational models is TransE [3]. Its simplicity allows for straightforward extensions [9]. The translation embedding of a triple (s, p, o) corresponds to s + p ≈ o. A scoring function score(s, p, o), either the ℓ1 or ℓ2 norm, is used to measure the distance (i.e., similarity) as:

score(s,p,o) = −||s + p − o||ℓ1/2 (1)

The training set contains positive examples (G), and negative examples (G′) generated as follows:
G′(s,p,o)∈G = {(s′, p, o) | s′ ∈ E, (s′, p, o) ∉ G} ∪
{(s, p, o′) | o′ ∈ E, (s, p, o′) ∉ G}.
Hence, G′ contains triples with either s or o replaced by a random entity from the set E.
RESCAL [11], also referred to as bilinear model, uses a tensor factorization model by representing triples in a tensor. That is, for each triple xijk = (si, pk, oj), yijk = {0, 1} denotes its existence or nonexistence in a tensor Y ∈ {0, 1}|E|×|E|×|R|. RESCAL learns vector embeddings of entities and a matrix Wp ∈ Rd×d for each relation r ∈ R where each slice Y is factorized as: Y ≈ s⊤Wpo. Hence, the scoring function for the bilinear model is:

score(s,p,o) = s⊤Wpo. (2)

Other notable relational embedding models are HolE [10] and Neural Tensor Networks (NTN) [16]. HolE improves the efficiency of RESCAL by using a circular correlation operation (it compresses the interaction between two entities) for scoring triples.
Almost all relational embedding approaches minimize a marginbased ranking loss function L over some training dataset. L is given by the following equation:

L = Σ(s,p,o)∈G Σ(s,p,o)'∈G'(s,p,o) [γ + score((s,p,o)) - score((s,p,o)')]+ (3)

where [x]+ denotes the positive part of x, γ > 0 is a margin hyperparameter. Different optimization functions such as stochastic gradient descent are used to minimize L.

#### 2.3.2 關係嵌入方法。

我們的問題更廣泛地與關係嵌入模型有關，這是一種在低維向量空間中進行關係學習的範式，已被廣泛用於連結預測和事實分類等任務。此類嵌入可視為圖嵌入的一個特例，這是一個非常活躍的研究課題，為求簡潔，我們在此予以省略。我們可以將這些模型大致分為三類：（i）平移距離，（ii）張量分解（雙線性模型），以及最近的（iii）神經網絡。向量用於在平移模型中學習實體和關係嵌入，而在雙線性模型和神經網絡的情況下則使用附加矩陣。平移模型使用距離度量來衡量事實的合理性，而雙線性模型則依賴於實體和關係嵌入的點積。最著名的平移模型之一是 TransE [3]。其簡單性使其易於直接擴展 [9]。三元組 (s, p, o) 的平移嵌入對應於 s + p ≈ o。評分函數 score(s, p, o)（ℓ1 或 ℓ2 範數）用於測量距離（即相似性）：

score(s,p,o) = −||s + p − o||ℓ1/2 (1)

訓練集包含正例 (G) 和負例 (G′)，生成方式如下：
G′(s,p,o)∈G = {(s′, p, o) | s′ ∈ E, (s′, p, o) ∉ G} ∪
{(s, p, o′) | o′ ∈ E, (s, p, o′) ∉ G}。
因此，G′ 包含的 s 或 o 被集合 E 中的隨機實體所取代的三元組。
RESCAL [11]，也稱為雙線性模型，通過在張量中表示三元組來使用張量分解模型。也就是說，對於每個三元組 xijk = (si, pk, oj)，yijk = {0, 1} 表示其在張量 Y ∈ {0, 1}|E|×|E|×|R| 中的存在或不存在。RESCAL 為每個關係 r ∈ R 學習實體的向量嵌入和矩陣 Wp ∈ Rd×d，其中每個切片 Y 被分解為：Y ≈ s⊤Wpo。因此，雙線性模型的評分函數為：

score(s,p,o) = s⊤Wpo. (2)

其他著名的關係嵌入模型還有 HolE [10] 和神經張量網絡 (NTN) [16]。HolE 通過使用循環相關運算（它壓縮兩個實體之間的相互作用）來提高 RESCAL 對三元組進行評分的效率。
幾乎所有的關係嵌入方法都在某個訓練數據集上最小化一個基於邊界的排名損失函數 L。L 由以下方程式給出：

L = Σ(s,p,o)∈G Σ(s,p,o)'∈G'(s,p,o) [γ + score((s,p,o)) - score((s,p,o)')]+ (3)

其中 [x]+ 表示 x 的正部分，γ > 0 是一個邊界超參數。使用不同的優化函數（如隨機梯度下降）來最小化 L。

#### 2.3.3 Factorization Machines.

Unlike vector space embedding models, Factorization Machines (FMs) allow us to incorporate contextual information which improves prediction performance. Rendle [12] introduced FMs to model the interaction between features using factorized parameters. One big advantage of FMs is that they allow to estimate all interactions between features even with very sparse data. In addition, FMs can mimic many different matrix factorization models such as biased matrix factorization, Singular Value Decomposition (SVD++) [8], and Pairwise Interaction Tensor Factorization (PITF) [13]. FMs provide flexibility in feature engineering as well as high prediction accuracy. Moreover, FMs can be applied to the following tasks: regression, binary classification, and ranking. The model of a factorization machine is given by the following equation:

score(x) := w0 + Σni=1wixi + Σni=1Σnj=i+1⟨vi,vj⟩xixj,
⟨vi,vj⟩ := Σkf=1vi,fvj,f

where score: Rn → T is a prediction function from a real valued feature vector x ∈ Rn to a target domain, T = R for regression, T = {+,−} for classification and so on. The model parameters: w0 denotes the global bias; wi within w ∈ Rn indicates the strength of the i-th variable with n being the size of the feature vector; ⟨vi,vj⟩ models the interaction between the i-th and j-th variables. ⟨., .⟩ is the dot product of two vectors of size k. Furthermore, the model parameter vi in V ∈ Rn×k describes the i-th variable with k factors. k is a hyperparameter that defines the dimension of the factorization. In this work, since we need to predict the validity of facts of (possible many) time points, we use factorization machine for classification rather than regression or ranking.

#### 2.3.3 分解機。

與向量空間嵌入模型不同，分解機（FMs）允許我們納入上下文資訊，從而提高預測性能。Rendle [12] 引入 FMs 來使用分解參數模擬特徵之間的交互作用。FMs 的一個巨大優勢是，即使在數據非常稀疏的情況下，它們也允許估計所有特徵之間的交互作用。此外，FMs 可以模擬許多不同的矩陣分解模型，例如偏置矩陣分解、奇異值分解（SVD++）[8] 和成對交互張量分解（PITF）[13]。FMs 在特徵工程方面提供了靈活性，並具有很高的預測準確性。此外，FMs 可應用於以下任務：回歸、二元分類和排名。分解機的模型由以下方程式給出：

score(x) := w0 + Σni=1wixi + Σni=1Σnj=i+1⟨vi,vj⟩xixj,
⟨vi,vj⟩ := Σkf=1vi,fvj,f

其中 score: Rn → T 是從實值特徵向量 x ∈ Rn 到目標域的預測函數，T = R 用於回歸，T = {+,−} 用於分類等。模型參數：w0 表示全局偏差；w ∈ Rn 中的 wi 表示第 i 個變量的強度，n 為特徵向量的大小；⟨vi,vj⟩ 模擬第 i 個和第 j 個變量之間的交互。⟨., .⟩是兩個大小為 k 的向量的點積。此外，V ∈ Rn×k 中的模型參數 vi 描述了具有 k 個因子的第 i 個變量。k 是一個超參數，它定義了分解的維度。在這項工作中，由於我們需要預測（可能很多）時間點的事實有效性，我們使用分解機進行分類而不是回歸或排名。

## 3 TEMPORAL SCOPE PREDICTION

In the following we consider relational embedding models and factorization machines for temporal scope prediction.

## 3 時間範圍預測

接下來，我們將考慮關係嵌入模型和分解機用於時間範圍預測。

### 3.1 Relational Embedding Models for Temporal KGs

We propose various approaches for representing temporal knowledge graphs in vector space. In particular, we investigate several extensions of existing relational embedding approaches.

### 3.1 時態知識圖譜的關係嵌入模型

我們提出了多種在向量空間中表示時態知識圖譜的方法。特別是，我們研究了現有關係嵌入方法的幾種擴展。

#### 3.1.1 TTransE.

Short for Temporal TransE, this is an extension of the well known embedding model TransE [3], by substituting its scoring function.

(a) Naive-TTransE: time is encoded by way of synthetic relations. For each relation r in the vocabulary and each time point t ∈ T, we assume a synthetic relation r:t. For instance, the temporal fact (GCleveland, office, POTUS):1888, is encoding as (GCleveland, office:1888, POTUS). The scoring function is unchanged (as in equation (1)):

score(s,p:t,o) = −||s + p:t − o||ℓ1/2 (4)

While this model is simple, it is not scalable. Besides the link prediction does not distinguish between two consecutive timepoints, for instance, for the task (GCleveland, ?, POTUS), office:1988 and office:1989 are equally possible links.

(b) Vector-based TTransE: in this approach, time is represented in the same vector space as entities and relations. The scoring function becomes:

score(s,p,o,t) = −||s + p + t − o||ℓ1/2 (5)

In this approach, time points have embedding representations, just like entities and relations. The rationale behind this scoring function is to drive a (subject, predicate)-pair close to the correct object, relative to any valid point in time.

(c) Coefficient-based TTransE: time points (or rather a normalization thereof) are used as a coefficient affecting the subject and relation embeddings of a triple.

score(s,p,o,t) = −||t ∗ (s + p) − o||ℓ1/2 (6)

As a variant of this, only the relation is affected by time:

score(s,p,o,t) = −||s + t ∗ p − o||ℓ1/2 (7)

Unlike Vector-based TTransE, time points are represented as real values in (0, 1], and thus are not directly affected by the optimization.

#### 3.1.1 TTransE。

TTransE 是 Temporal TransE 的縮寫，它是著名嵌入模型 TransE [3] 的擴展，通過替換其評分函數而來。

(a) Naive-TTransE：時間通過合成關係進行編碼。對於詞彙表中的每個關係 r 和每個時間點 t ∈ T，我們假設存在一個合成關係 r:t。例如，時態事實 (GCleveland, office, POTUS):1888 編碼為 (GCleveland, office:1888, POTUS)。評分函數保持不變（如方程式（1）所示）：

score(s,p:t,o) = −||s + p:t − o||ℓ1/2 (4)

雖然這個模型很簡單，但它不具備可擴展性。此外，連結預測無法區分兩個連續的時間點，例如，對於任務 (GCleveland, ?, POTUS)，office:1988 和 office:1989 的連結可能性是相同的。

(b) Vector-based TTransE：在此方法中，時間與實體和關係在同一個向量空間中表示。評分函數變為：

score(s,p,o,t) = −||s + p + t − o||ℓ1/2 (5)

在這種方法中，時間點與實體和關係一樣，都有嵌入表示。此評分函數背後的原理是將（主詞，謂詞）對驅動到靠近正確的客體，相對於任何有效的時間點。

(c) Coefficient-based TTransE：時間點（或其歸一化值）被用作影響三元組的主詞和關係嵌入的係數。

score(s,p,o,t) = −||t ∗ (s + p) − o||ℓ1/2 (6)

作為此變體，只有關係受時間影響：

score(s,p,o,t) = −||s + t ∗ p − o||ℓ1/2 (7)

與 Vector-based TTransE 不同，時間點表示為 (0, 1] 中的實數值，因此不受優化直接影響。

#### 3.1.2 TRESCAL.

TRESCAL is a temporal extension of RESCAL. We extend its bilinear temporal scoring function as follows. As in Naive-TTransE, time is encoded by means of synthetic relations just like Naive-TTransE.

score(s,p,o,t) = s⊤Wp:to (8)

This model is straight forward extension of the bilinear model. Despite its simplicity, it does not scale well, besides, the prediction results are quite poor.

#### 3.1.2 TRESCAL。

TRESCAL 是 RESCAL 的時態擴展。我們如下擴展其雙線性時態評分函數。與 Naive-TTransE 中一樣，時間是通過合成關係編碼的，就像 Naive-TTransE 一樣。

score(s,p,o,t) = s⊤Wp:to (8)

這個模型是雙線性模型的直接擴展。儘管它很簡單，但擴展性不好，而且預測結果也很差。

### 3.2 Factorization Machines for Temporal KGs

Among the approaches described so far, the naive ones do not scale well with time domains of increasing size or resolution. Although the vector-based TTransE approach performs overall better than the other techniques, it did not show good enough performance to solve our problem in practice. In the following, we show how we used factorization machines to solve both or scability and performance issues.

Data/Feature Representation. We consider a knowledge graph G = Gt ∪ Gc where Gt is a set of quadruples or timestamped triples, and Gc is a set of atemporal triples that we refer to as a context graph. For instance, the following is a temporal graph Gt:

(GCleveland, office, POTUS): 1888,
(GCleveland, office, POTUS): 1895,

### 3.2 時態知識圖譜的分解機

到目前為止所描述的方法中，那些較為簡單的方法在時間域大小或分辨率增加時，其擴展性不佳。雖然基於向量的 TTransE 方法整體性能優於其他技術，但在實踐中，它並未顯示出足夠好的性能來解決我們的問題。在下文中，我們將展示如何使用分解機來解決我們的擴展性和性能問題。

數據/特徵表示。我們考慮一個知識圖譜 G = Gt ∪ Gc，其中 Gt 是一組四元組或帶時間戳的三元組，而 Gc 是我們稱之為上下文圖的一組非時態三元組。例如，以下是一個時態圖 Gt：

(GCleveland, office, POTUS): 1888,
(GCleveland, office, POTUS): 1895,

and its context graph Gc is given below:

(GCleveland, birthPlace, Caldwell).

An input to an FM is a feature vector representation of the pair (Gt, Gc). The feature vector encoding can be constructed in several ways such as one-hot encoding, bag-of-words (representing KG entities and relations in a bag or multiset) and so on [12]. The features associated with a fact of the form (s, p, o) are {bow(s), p, bow(o)}, where bow(x) returns the bag of words of all the literals in relations with subject x.

Example Generation. To generate positive examples, we used temporal sampling, guiding by input parameter TS, which consists in sampling uniformly st time points within the fact’s validity intervals. A second parameter, NS, guides negative sampling, producing sn for each positive time-point-based fact/example, using the same random corruption techniques as in [3].

其上下文圖 Gc 如下所示：

(GCleveland, birthPlace, Caldwell)。

FM 的輸入是 (Gt, Gc) 對的特徵向量表示。特徵向量編碼可以通過多種方式構建，例如 one-hot 編碼、詞袋（將知識圖譜實體和關係表示為詞袋或多重集）等 [12]。與 (s, p, o) 形式的事實相關的特徵是 {bow(s), p, bow(o)}，其中 bow(x) 返回與主詞 x 相關的所有文字的詞袋。

樣本生成。為了生成正樣本，我們使用了時間採樣，由輸入參數 TS 指導，該參數包括在事實有效期間內均勻採樣 st 個時間點。第二個參數 NS 指導負採樣，為每個基於時間點的正樣本/事實生成 sn 個負樣本，使用與 [3] 中相同的隨機損壞技術。

## 4 EXPERIMENTS

We implemented our approach based on the scikit-kge library of RESCAL and TransE¹, and libFM/pywFM².

## 4 實驗

我們的實作是基於 RESCAL 和 TransE¹ 的 scikit-kge 函式庫，以及 libFM/pywFM²。

### 4.1 Datasets

We originally experimented with the Freebase database often used in the related work (our first set of experiments). However, the facts on those dataset not having temporal information, we randomly generate such metadata for a subset of them, by picking two random years and using them as start and end validity dates. For this reason, it is hard to compare our results with corresponding work in the non-temporal relational embedding scenarios. Freebase has approximately 14K entities, and 1000 relations, with 60k examples. We later decided to switch to Wikidata, a knowledge base with reasonably high quality time information. Moreover Wikidata is much larger and up-to-date. We only briefly present the result obtained in the former dataset, and results were largely negative. Besides, using the Freebase and WordNet data set with the factorization machines approach was not possible because of the lack of side-information to exploit; the data sets contain very little plain text. Our process in preparing the Wikidata data set was the following. We extracted triples from a recent dump, and partitioned them into two sets: (i) temporal facts: facts having some temporal annotations, such as point-in-time, start time, end time or any sub-property thereof, (ii) atemporal facts: atemporal facts, having no such annotations. Temporal properties annotating temporal facts include "start time", "inception", "demolition time", etc. In this work, we only consider years, and thus normalize all years to the Gregorian calendar and discard information of finer granularity. Facts annotated with a single point-in-time are associated with that time-point as start and end time.
During the learning phase, temporal facts are used to generate positive and negative examples and atemporal facts are used to collect side information. The complete data has 4.2M temporal facts. Out of approximately 3600 distinct properties, 2770 are strictly atemporal, i.e., none of their corresponding triples are temporal annotation. Out of the remaining properties, 17 are strictly temporal, i.e., all their corresponding triples have temporal annotations, while for the remaining 813 properties, only some triples are annotated. We partition the triples into two sets, respectively with and without temporal annotations, the former being our original example set. From this example set (temporal facts), we exclude the strictly temporal ones (since they are not candidate for prediction), the fact featuring the most frequent single frequent property — covering nearly 1.2M examples —, and those with properties covering less than 10 examples (approximately 397 properties). Ultimately, our example set contains 2.5M examples, much more than most datasets used in related approaches (see for example [3, 10]). We also report our results on a reduced version of this data set, containing 180K temporally annotated facts (i.e., approx. 5% of the overall data). Our dataset can be found online for reproducibility³.
The second set of triples (atemporal facts) is used for generating features. We also remove the set of triples with low semantic content such as those mapping a Wikidata entity ID to that of other datasets.

### 4.1 資料集

我們最初使用相關研究中常用的 Freebase 資料庫進行實驗（我們的第一組實驗）。然而，由於這些資料集上的事實缺乏時間資訊，我們隨機為其中一部分生成此類元數據，方法是挑選兩個隨機年份作為開始和結束的有效日期。因此，很難將我們的結果與非時態關係嵌入場景中的相應工作進行比較。Freebase 大約有 14K 個實體、1000 個關係和 60k 個範例。我們後來決定改用 Wikidata，這是一個擁有相當高品質時間資訊的知識庫。此外，Wikidata 的規模更大且資訊更新。我們僅簡要介紹前一個資料集獲得的結果，且結果大多為負面。此外，由於缺乏可利用的輔助資訊，使用 Freebase 和 WordNet 資料集搭配分解機方法是不可行的；這些資料集包含的純文字非常少。我們準備 Wikidata 資料集的過程如下。我們從最近的轉儲中提取三元組，並將它們劃分為兩組：（i）時態事實：具有某些時間註釋的事實，例如時間點、開始時間、結束時間或其任何子屬性；（ii）非時態事實：沒有此類註釋的非時態事實。註釋時態事實的時間屬性包括「開始時間」、「成立時間」、「拆除時間」等。在這項工作中，我們只考慮年份，因此將所有年份標準化為公曆並捨棄更精細的資訊。註釋有單一時間點的事實與該時間點的開始和結束時間相關聯。
在學習階段，時態事實被用來產生正、負樣本，而非時態事實則被用來收集輔助資訊。完整的資料集擁有 420 萬個時態事實。在大約 3600 個不同的屬性中，2770 個是嚴格非時態的，也就是說，它們對應的三元組都沒有時間註釋。在剩下的屬性中，17 個是嚴格時態的，也就是說，它們所有對應的三元組都有時間註釋，而剩下的 813 個屬性，只有部分三元組被註釋。我們將三元組分為兩組，分別有和沒有時間註釋，前者是我們最初的樣本集。從這個樣本集（時態事實）中，我們排除了嚴格時態的（因為它們不是預測的候選者）、具有最頻繁單一屬性的事實——涵蓋近 120 萬個樣本——以及那些屬性涵蓋少於 10 個樣本的（大約 397 個屬性）。最終，我們的樣本集包含 250 萬個樣本，遠多於相關方法中使用的大多數資料集（例如，參見 [3, 10]）。我們還報告了我們在這個資料集的縮減版本上的結果，該版本包含 18 萬個帶有時間註釋的事實（即，大約佔總資料的 5%）。我們的資料集可在線上取得以供重現³。
第二組三元組（非時態事實）用於生成特徵。我們還移除了語意內容低的三元組集，例如那些將 Wikidata 實體 ID 映射到其他資料集 ID 的三元組。

¹https://github.com/mnick/scikit-kge
²https://github.com/srendle/libfm

¹https://github.com/mnick/scikit-kge
²https://github.com/srendle/libfm

³http://staff.aist.go.jp/julien.leblay/datasets/

³http://staff.aist.go.jp/julien.leblay/datasets/

### 4.2 Temporal relational embeddings

For this experiment, we use the modified Freebase dataset, and evaluated the approaches with a slightly modified version of that from the related work, which evaluate using query triples, i.e., facts in which one item is omitted and need to be predicted by the models. For query answering, s or o is omitted, while p is omitted in link prediction. The evaluation metrics are the Mean Rank of the correct answers among all answers order by their predicating probability. The lower, the better. Metrics also include the “Hits@K”, i.e., the percentage of case in which the correct answer is in the top K results. Hits@10 is a popular metric, yet for small domain (such as in link prediction), Hits@1 is usually preferred. In our setting, we deal with quadruple, therefore we extend the process to time prediction in which, time is omitted, and will evaluate how often a predicted validity time point is within the actual validity interval of the fact.
In Table 1, we only report the best results obtained with each approaches. We ran the approaches with learning rates (LR) among {.01, .1}, margins (M) among {2, 10}, dimensionalities of the vector space (D) among {20, 50, 100, 200}, and learning over 500 or 1000 epochs (E). It is clear from the table that the performs are not satisfying. However, we can distinguish two general problems. For the naive methods (Eq. 4-8), the space explodes from the multiplication of “virtual relations” entailed by the methods. This is why performance are poor despite significant cost reductions achieved through the learning process. The other methods however do not achieve much cost reduction all together. Our best explanation for this is that learning time validity simple from the structure of the graph (i.e., using no other external information) is simply to hard. This conclusion led us to turn to the Factorization Machine approach, more akin to the incorporation of side information.

### 4.2 時態關係嵌入

在這個實驗中，我們使用修改過的 Freebase 資料集，並使用相關工作中略微修改的版本來評估這些方法，這些版本使用查詢三元組進行評估，也就是說，事實中省略了一項，需要模型來預測。對於查詢回答，s 或 o 被省略，而在連結預測中，p 被省略。評估指標是正確答案在所有按其預測機率排序的答案中的平均排名。排名越低越好。指標還包括「Hits@K」，也就是說，正確答案在前 K 個結果中的案例百分比。Hits@10 是一個流行的指標，但對於小領域（例如連結預測），通常首選 Hits@1。在我們的設置中，我們處理四元組，因此我們將過程擴展到時間預測，其中時間被省略，並將評估預測的有效時間點在事實的實際有效區間內的頻率。
在表 1 中，我們僅報告了每種方法獲得的最佳結果。我們在學習率 (LR) 為 {.01, .1}，邊界 (M) 為 {2, 10}，向量空間維度 (D) 為 {20, 50, 100, 200}，以及學習超過 500 或 1000 個時期 (E) 的情況下運行了這些方法。從表中可以清楚地看出，性能並不令人滿意。然而，我們可以區分兩個普遍問題。對於樸素方法（方程式 4-8），由於方法所產生的「虛擬關係」的乘積，空間會爆炸。這就是為什麼儘管通過學習過程實現了顯著的成本降低，但性能仍然很差。然而，其他方法並未共同實現太多的成本降低。我們對此的最佳解釋是，僅從圖的結構（即不使用任何其他外部資訊）學習時間有效性實在太難了。這個結論促使我們轉向分解機方法，更類似於納入輔助資訊。

| Approach | LR | M | D | E | MR (p) | Hits@1 (p) | MR (o) | Hits@10 (o) | MR (t) | Hits@10 (t) | Cost Red. |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Eq. 4 | 0.1 | 2 | 100 | 1000 | 537.51 | 0.6 | 2578.4 | 11.0 | 59.2 | 10.3 | 99.75% |
| Eq. 5 | 0.01 | 1 | 200 | 1000 | 141.67 | 22.69 | 1295.54 | 13.59 | 58.44 | 7.76 | 45.32% |
| Eq. 6 | 0.1 | 10 | 100 | 500 | 835.22 | 0.55 | 9884.69 | 0.91 | 58.50 | 8.62 | 0.13% |
| Eq. 7 | 0.01 | 2 | 50 | 500 | 796.65 | 0.18 | 9374.92 | 0.19 | 58.50 | 8.62 | 0.45% |
| Eq. 8 | 0.01 | 2 | 100 | 1000 | 483.32 | 3.1 | 6588.6 | 1.9 | 58.5 | 12.1 | 99.99% |

**Table 1: Mean Rank (MR), Hits@{1,10} and cost reduction for our temporal embeddings methods on the Freebase dataset.**

**表 1：我們在 Freebase 資料集上的時態嵌入方法的平均排名 (MR)、Hits@{1,10} 和成本降低。**

### 4.3 Classification task on FM

For the classification task, the learning is done on quadruples of the form (s, p, o, t) = ±1, modeling whether the triple (s, p, o) held at time t or not. After the sampling, the effective number of examples increase. For instance with TS = 3, (GCleveland, office, POTUS):[1885, 1889], will generate positive examples for the time points 1885, 1887, 1889. The evaluation, in turn, is performed on time points rather than time intervals. We use the standard definition of precision, recall, F-measure and accuracy. The definitions of these measures are given below:

precision = #true positives / #positive predictions
recall = #true positives / #ground truth positives
F-measure = 2 × (precision × recall) / (precision + recall)
accuracy = #correct predictions / #all predictions

We used the optimization functions Alternating Least Square (ALS), and Markov Chain Monte Carlo (MCMC). We report the precision, recall, F-measure and accuracy in Table 2, which shows the results for experiments run on a Wikidata data set of 180K and 2.5M examples, using bag-of-words as side information, with increasing temporal sampling size. The results for high NS are omitted since the greater number of negative examples tends to biases the model towards negative predictions, resulting in high accuracy, despite poor precision. With a balanced set of positive and negative examples, precision is positively correlated with TS. Using a temporal sampling of 100, with our smaller dataset, precision and recall peak at 74.5% and 92% respectively after 100 iterations, with an F1-measure and accuracy around 82%. Using a temporal sampling size of 10, with our bigger dataset, the F1-measure and accuracy reach 90%. Increasing the sample size, also improves performance, yet producing positive examples for all time points within a time interval degrades the performance, probably due to over-fitting. Our result also shows that a precision of around 70% can be achieved with only 10 iterations.
Our most demanding experiment took slightly over 6 hours to complete on a regular laptop, with 16GB of RAM, and a 2.8 GHz Intel Core i5 processor.
We have excluded experimental results for TTransE and TRESCAL as our result showed the methods were not competitive.

### 4.3 FM 上的分類任務

對於分類任務，學習是在 (s, p, o, t) = ±1 形式的四元組上完成的，模擬三元組 (s, p, o) 在時間 t 是否成立。採樣後，有效樣本數增加。例如，當 TS = 3 時，(GCleveland, office, POTUS):[1885, 1889] 將為時間點 1885、1887、1889 生成正樣本。反過來，評估是在時間點而不是時間間隔上執行的。我們使用精確率、召回率、F-量測和準確率的標準定義。這些指標的定義如下：

精確率 = 真陽性數 / 預測陽性數
召回率 = 真陽性數 / 真實陽性數
F-量測 = 2 × (精確率 × 召回率) / (精確率 + 召回率)
準確率 = 正確預測數 / 所有預測數

我們使用了交替最小二乘法 (ALS) 和馬可夫鏈蒙地卡羅 (MCMC) 優化函數。我們在表 2 中報告了精確率、召回率、F-量測和準確率，該表顯示了在 180K 和 2.5M 範例的 Wikidata 資料集上運行的實驗結果，使用詞袋作為輔助資訊，並增加時間採樣大小。高 NS 的結果被省略，因為大量的負樣本傾向於使模型偏向負預測，導致高準確率但精確率差。在正負樣本平衡的情況下，精確率與 TS 呈正相關。使用 100 的時間採樣，在我們的較小資料集上，精確率和召回率在 100 次迭代後分別達到 74.5% 和 92% 的峰值，F1-量測和準確率約為 82%。使用 10 的時間採樣大小，在我們的較大資料集上，F1-量測和準確率達到 90%。增加樣本大小也會提高性能，但在時間間隔內為所有時間點生成正樣本會降低性能，可能是由於過度擬合。我們的結果還表明，僅需 10 次迭代即可達到約 70% 的精確率。
我們最耗時的實驗在一台配備 16GB 記憶體和 2.8 GHz Intel Core i5 處理器的普通筆記型電腦上花費了 6 個多小時才完成。
我們排除了 TTransE 和 TRESCAL 的實驗結果，因為我們的結果顯示這些方法不具競爭力。

| OM | TS | Precision | Recall | F1 | Accuracy |
| :--- | :--- | :--- | :--- | :--- | :--- |
| WD_180K ALS | 1 | 58.44% | 71.27% | 64.22% | 60.23% |
| WD_180K ALS | 10 | 67.94% | 88.95% | 77.04% | 73.48% |
| WD_180K ALS | 100 | 74.56% | 92.47% | 82.56% | 80.45% |
| WD_2.5M ALS | 10 | 78.15% | 97.64% | 86.81% | 85.16% |
| WD_180K MCMC | 1 | 64.98% | 81.07% | 72.14% | 68.64% |
| WD_180K MCMC | 10 | 69.55% | 89.69% | 78.35% | 75.21% |
| WD_180K MCMC | 100 | 79.28% | 92.28% | 85.28% | 84.07% |
| WD_2.5M MCMC | 10 | 85.41% | 97.64% | 91.12% | 90.48% |

**Table 2: Precision, recall, F1-measure and accuracy on the WD_180K dataset with varying temporal sampling at 100 iterations (OM: Optimization Method, TS: temporal sample size).**

**表 2：在 100 次迭代中，WD_180K 數據集上不同時間採樣的精確率、召回率、F1 度量和準確率（OM：優化方法，TS：時間採樣大小）。**

## 5 CONCLUSION

In this work, we studied the problem of temporal scope prediction. We adapted several existing relational embedding approaches in which our experimental results have shown that they suffer from either scalability or accuracy. Factorization machines overcome these shortcomings as they provide a way to incorporate side information which improves prediction performance. We designed a new dataset by carefully analyzing Wikidata and carried out several experiments. We believed our experimental results are quite promising. Next, we plan to turn our attention to neural network-based approaches, extend our current framework to support time-aware link prediction and query answering, and applies our finding to other types of context prediction, such as space or provenance. We all plan to apply the approach in an open information extraction setting.

## 5 結論

在這項工作中，我們研究了時間範圍預測的問題。我們調整了幾種現有的關係嵌入方法，我們的實驗結果顯示，這些方法存在可擴展性或準確性的問題。分解機克服了這些缺點，因為它們提供了一種納入輔助資訊的方法，從而提高了預測性能。我們通過仔細分析 Wikidata 設計了一個新的資料集，並進行了幾項實驗。我們相信我們的實驗結果非常有前景。接下來，我們計劃將注意力轉向基於神經網絡的方法，擴展我們目前的框架以支援時間感知的連結預測和查詢回答，並將我們的發現應用於其他類型的上下文預測，例如空間或來源。我們都計劃在開放資訊提取的環境中應用該方法。

### REFERENCES

[1] Sören Auer, Christian Bizer, Georgi Kobilarov, Jens Lehmann, Richard Cyganiak, and Zachary Ives. 2007. Dbpedia: A nucleus for a web of open data. In The semantic web. Springer, 722–735.
[2] Brett W Bader, Richard A Harshman, and Tamara G Kolda. 2007. Temporal analysis of semantic graphs using ASALSAN. In Data Mining, 2007. ICDM 2007. Seventh IEEE International Conference on. IEEE, 33–42.
[3] Antoine Bordes, Nicolas Usunier, Alberto Garcia-Duran, Jason Weston, and Oksana Yakhnenko. 2013. Translating embeddings for modeling multi-relational data. In Advances in neural information processing systems. 2787–2795.
[4] Andrew Carlson, Justin Betteridge, Bryan Kisiel, Burr Settles, Estevam R Hruschka Jr, and Tom M Mitchell. 2010. Toward an Architecture for Never-Ending Language Learning.. In AAAI, Vol. 5. 3.
[5] Xin Dong, Evgeniy Gabrilovich, Geremy Heitz, Wilko Horn, Ni Lao, Kevin Murphy, Thomas Strohmann, Shaohua Sun, and Wei Zhang. 2014. Knowledge Vault: A Web-Scale Approach to Probabilistic Knowledge Fusion. In SIGKDD. 601–610.
[6] Johannes Hoffart, Fabian M Suchanek, Klaus Berberich, Edwin Lewis-Kelham, Gerard De Melo, and Gerhard Weikum. 2011. YAGO2: exploring and querying world knowledge in time, space, context, and many languages. In Proceedings of the 20th international conference companion on World wide web. ACM, 229–232.
[7] Tingsong Jiang, Tianyu Liu, Tao Ge, Lei Sha, Baobao Chang, Sujian Li, and Zhifang Sui. 2016. Towards Time-Aware Knowledge Graph Completion.. In COLING. 1715–1724.
[8] Yehuda Koren. 2008. Factorization meets the neighborhood: a multifaceted collaborative filtering model. In Proceedings of the 14th ACM SIGKDD international conference on Knowledge discovery and data mining. ACM, 426–434.
[9] Dat Quoc Nguyen. 2017. An overview of embedding models of entities and relationships for knowledge base completion. arXiv preprint arXiv:1703.08098 (2017).
[10] Maximilian Nickel, Lorenzo Rosasco, Tomaso A Poggio, and others. 2016. Holographic Embeddings of Knowledge Graphs.. In AAAI. 1955–1961.
[11] Maximilian Nickel, Volker Tresp, and Hans-Peter Kriegel. 2011. A three-way model for collective learning on multi-relational data. In Proceedings of the 28th international conference on machine learning (ICML-11). 809–816.
[12] Steffen Rendle. 2012. Factorization machines with libfm. ACM Transactions on Intelligent Systems and Technology (TIST) 3, 3 (2012), 57.
[13] Steffen Rendle and Lars Schmidt-Thieme. 2010. Pairwise interaction tensor factorization for personalized tag recommendation. In Proceedings of the third ACM international conference on Web search and data mining. ACM, 81–90.
[14] Anisa Rula, Matteo Palmonari, Axel-Cyrille Ngonga Ngomo, Daniel Gerber, Jens Lehmann, and Lorenz Bühmann. 2014. Hybrid acquisition of temporal scopes for rdf data. In European Semantic Web Conference. Springer, 488–503.
[15] Avirup Sil and Silviu Cucerzan. 2014. Temporal scoping of relational facts based on Wikipedia data. CoNLL-2014 (2014), 109.
[16] Richard Socher, Danqi Chen, Christopher D Manning, and Andrew Ng. 2013. Reasoning with neural tensor networks for knowledge base completion. In Advances in neural information processing systems. 926–934.
[17] Partha Pratim Talukdar, Derry Wijaya, and Tom Mitchell. 2012. Coupled temporal scoping of relational facts. In Proceedings of the fifth ACM international conference on Web search and data mining. ACM, 73–82.
[18] Volker Tresp, Yunpu Ma, Stephan Baier, and Yinchong Yang. 2017. Embedding Learning for Declarative Memories. Springer International Publishing, Cham, 202–216. DOI: https://doi.org/10.1007/978-3-319-58068-5_13
[19] Rakshit Trivedi, Mehrdad Farajtabar, Yichen Wang, Hanjun Dai, Hongyuan Zha, and Le Song. 2017. Know-Evolve: Deep Reasoning in Temporal Knowledge Graphs. arXiv preprint arXiv:1705.05742 (2017).
[20] Denny Vrandečić and Markus Krötzsch. 2014. Wikidata: a free collaborative knowledgebase. Commun. ACM 57, 10 (2014), 78–85.
[21] Derry Tanti Wijaya, Ndapandula Nakashole, and Tom M Mitchell. 2014. CTPs: Contextual Temporal Profiles for Time Scoping Facts using State Change Detection.. In EMNLP. 1930–1936.

### 參考文獻

[1] Sören Auer, Christian Bizer, Georgi Kobilarov, Jens Lehmann, Richard Cyganiak, and Zachary Ives. 2007. Dbpedia: A nucleus for a web of open data. In The semantic web. Springer, 722–735.
[2] Brett W Bader, Richard A Harshman, and Tamara G Kolda. 2007. Temporal analysis of semantic graphs using ASALSAN. In Data Mining, 2007. ICDM 2007. Seventh IEEE International Conference on. IEEE, 33–42.
[3] Antoine Bordes, Nicolas Usunier, Alberto Garcia-Duran, Jason Weston, and Oksana Yakhnenko. 2013. Translating embeddings for modeling multi-relational data. In Advances in neural information processing systems. 2787–2795.
[4] Andrew Carlson, Justin Betteridge, Bryan Kisiel, Burr Settles, Estevam R Hruschka Jr, and Tom M Mitchell. 2010. Toward an Architecture for Never-Ending Language Learning.. In AAAI, Vol. 5. 3.
[5] Xin Dong, Evgeniy Gabrilovich, Geremy Heitz, Wilko Horn, Ni Lao, Kevin Murphy, Thomas Strohmann, Shaohua Sun, and Wei Zhang. 2014. Knowledge Vault: A Web-Scale Approach to Probabilistic Knowledge Fusion. In SIGKDD. 601–610.
[6] Johannes Hoffart, Fabian M Suchanek, Klaus Berberich, Edwin Lewis-Kelham, Gerard De Melo, and Gerhard Weikum. 2011. YAGO2: exploring and querying world knowledge in time, space, context, and many languages. In Proceedings of the 20th international conference companion on World wide web. ACM, 229–232.
[7] Tingsong Jiang, Tianyu Liu, Tao Ge, Lei Sha, Baobao Chang, Sujian Li, and Zhifang Sui. 2016. Towards Time-Aware Knowledge Graph Completion.. In COLING. 1715–1724.
[8] Yehuda Koren. 2008. Factorization meets the neighborhood: a multifaceted collaborative filtering model. In Proceedings of the 14th ACM SIGKDD international conference on Knowledge discovery and data mining. ACM, 426–434.
[9] Dat Quoc Nguyen. 2017. An overview of embedding models of entities and relationships for knowledge base completion. arXiv preprint arXiv:1703.08098 (2017).
[10] Maximilian Nickel, Lorenzo Rosasco, Tomaso A Poggio, and others. 2016. Holographic Embeddings of Knowledge Graphs.. In AAAI. 1955–1961.
[11] Maximilian Nickel, Volker Tresp, and Hans-Peter Kriegel. 2011. A three-way model for collective learning on multi-relational data. In Proceedings of the 28th international conference on machine learning (ICML-11). 809–816.
[12] Steffen Rendle. 2012. Factorization machines with libfm. ACM Transactions on Intelligent Systems and Technology (TIST) 3, 3 (2012), 57.
[13] Steffen Rendle and Lars Schmidt-Thieme. 2010. Pairwise interaction tensor factorization for personalized tag recommendation. In Proceedings of the third ACM international conference on Web search and data mining. ACM, 81–90.
[14] Anisa Rula, Matteo Palmonari, Axel-Cyrille Ngonga Ngomo, Daniel Gerber, Jens Lehmann, and Lorenz Bühmann. 2014. Hybrid acquisition of temporal scopes for rdf data. In European Semantic Web Conference. Springer, 488–503.
[15] Avirup Sil and Silviu Cucerzan. 2014. Temporal scoping of relational facts based on Wikipedia data. CoNLL-2014 (2014), 109.
[16] Richard Socher, Danqi Chen, Christopher D Manning, and Andrew Ng. 2013. Reasoning with neural tensor networks for knowledge base completion. In Advances in neural information processing systems. 926–934.
[17] Partha Pratim Talukdar, Derry Wijaya, and Tom Mitchell. 2012. Coupled temporal scoping of relational facts. In Proceedings of the fifth ACM international conference on Web search and data mining. ACM, 73–82.
[18] Volker Tresp, Yunpu Ma, Stephan Baier, and Yinchong Yang. 2017. Embedding Learning for Declarative Memories. Springer International Publishing, Cham, 202–216. DOI: https://doi.org/10.1007/978-3-319-58068-5_13
[19] Rakshit Trivedi, Mehrdad Farajtabar, Yichen Wang, Hanjun Dai, Hongyuan Zha, and Le Song. 2017. Know-Evolve: Deep Reasoning in Temporal Knowledge Graphs. arXiv preprint arXiv:1705.05742 (2017).
[20] Denny Vrandečić and Markus Krötzsch. 2014. Wikidata: a free collaborative knowledgebase. Commun. ACM 57, 10 (2014), 78–85.
[21] Derry Tanti Wijaya, Ndapandula Nakashole, and Tom M Mitchell. 2014. CTPs: Contextual Temporal Profiles for Time Scoping Facts using State Change Detection.. In EMNLP. 1930–1936.

文件翻譯完成。