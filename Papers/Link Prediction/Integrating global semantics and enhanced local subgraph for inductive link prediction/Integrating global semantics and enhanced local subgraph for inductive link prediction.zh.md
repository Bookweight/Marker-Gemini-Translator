---
title: Integrating global semantics and enhanced local subgraph for inductive link prediction
field: Link_Prediction
status: Imported
created_date: 2026-01-14
pdf_link: "[[Integrating global semantics and enhanced local subgraph for inductive link prediction.pdf]]"
tags:
  - paper
  - Link_prediction
---
# Integrating global semantics and enhanced local subgraph for inductive link prediction
# 整合全域語意與增強型區域子圖以進行歸納式連結預測
Xinyu Liang¹ · Guannan Si¹ · Jianxin Li¹ · Zhaoliang An¹ · Pengxin Tian¹ · Fengyu Zhou² · Xiaoliang Wang³

Received: 22 August 2023 / Accepted: 26 August 2024 / Published online: 6 September 2024
© The Author(s), under exclusive licence to Springer-Verlag GmbH Germany, part of Springer Nature 2024
梁馨予¹ · 司冠男¹ · 李建新¹ · 安兆亮¹ · 田鹏鑫¹ · 周凤宇² · 王晓亮³

收稿日期：2023年8月22日 / 接受日期：2024年8月26日 / 線上發表日期：2024年9月6日
© 作者獨家授權予德國斯普林格出版社（Springer-Verlag GmbH Germany），該公司為施普林格自然（Springer Nature）之一部分 2024
## Abstract
Inductive link prediction (ILP) predicts missing triplets involving unseen entities in knowledge graphs (KGs). Existing ILP research mainly addresses seen-unseen entities in the original KG (semi-inductive link prediction) and unseen-unseen entities in emerging KGs (fully-inductive link prediction). Bridging-inductive link prediction, which focuses on unseen entities that carry evolutionary information from the original KG to the emerging KG, has not been extensively studied so far. This study introduces a novel model called GSELI (integrating global semantics and enhanced local subgraph for inductive link prediction), which comprises three components. (1) The contrastive learning-based global semantic features (CLSF) module extracts relation-specific semantic features between the original and emerging KGs and employs semantic-aware contrastive learning to optimize these features. (2) The GNN-based enhanced local subgraph (GELS) module employs personalized PageRank (PPR)-based local clustering to sample tightly-related subgraphs and incorporates complete neighboring relations to enhance the topological information of subgraphs. (3) Joint contrastive learning and supervised learning training. Experimental results on various benchmark datasets demonstrate that GSELI outperforms the baseline models in both fully-inductive and bridging-inductive link predictions.
## 摘要
歸納式連結預測（Inductive link prediction, ILP）旨在預測知識圖譜（Knowledge graphs, KGs）中涉及未見實體的缺失三元組。現有的ILP研究主要處理原始KG中已見與未見實體之間的連結（半歸納式連結預測），以及新興KG中未見實體之間的連結（全歸納式連結預測）。然而，一種介於兩者之間的橋接歸納式連結預測，即關注那些將原始KG的演化資訊帶到新興KG中的未見實體，至今尚未得到廣泛研究。本研究提出了一個名為GSELI（整合全域語意與增強型區域子圖以進行歸納式連結預測）的新模型，該模型包含三個部分：（1）基於對比學習的全域語意特徵（Contrastive learning-based global semantic features, CLSF）模組，用於提取原始與新興KG之間特定於關聯的語意特徵，並採用語意感知對比學習來優化這些特徵。（2）基於GNN的增強型區域子圖（GNN-based enhanced local subgraph, GELS）模組，採用個人化佩吉排名（Personalized PageRank, PPR）的區域聚類方法來取樣高度相關的子圖，並整合完整的鄰近關聯以增強子圖的拓撲資訊。（3）聯合對比學習與監督式學習的訓練方式。在多個基準資料集上的實驗結果顯示，GSELI在全歸納式和橋接歸納式連結預測方面均優於基線模型。
**Keywords** Inductive link prediction · Fully-inductive · Bridging-inductive · Unseen entities
**關鍵詞** 歸納式連結預測 · 全歸納式 · 橋接歸納式 · 未見實體
* Guannan Si
sign@sdjtu.edu.cn
Xinyu Liang
liangxinyu0514@163.com
Jianxin Li
lijx7617@163.com
Zhaoliang An
anzhaoliang1@163.com
Pengxin Tian
tianpengxin1123@163.com
Fengyu Zhou
zhoufengyu@sdu.edu.cn
Xiaoliang Wang
xliang0505@163.com

1 School of information Science and Electrical Engineering,
Shandong Jiaotong University, Jinan 250357, State, China
2 School of Control Science and Engineering, Shandong
University, Jinan 250000, State, China
3 Shandong Longxihanzhang Technology Development Co.,
Ltd, Jinan 250013, State, China
* 司冠男
sign@sdjtu.edu.cn
梁馨予
liangxinyu0514@163.com
李建新
lijx7617@163.com
安兆亮
anzhaoliang1@163.com
田鹏鑫
tianpengxin1123@163.com
周凤宇
zhoufengyu@sdu.edu.cn
王晓亮
xliang0505@163.com

1 山東交通大學資訊科學與電氣工程學院，濟南 250357，中國
2 山東大學控制科學與工程學院，濟南 250000，中國
3 山東龍熙漢章科技發展有限公司，濟南 250013，中國
## 1 Introduction
Knowledge graphs (KGs) are structured collections of factual triplets (entity, relation, entity), where each entity is depicted as a node, and each relation is denoted as an edge. KGs play a crucial role in knowledge-intensive tasks such as intelligent question answering [1], recommendation systems [2], and semantic search [3]. However, due to the exponential growth of real-world data, most existing KGs suffer from noise and incompleteness. For instance, in the open KG Freebase, approximately 71% of people lack birthplace information, and 99% do not have ethnic information [4]. Consequently, predicting missing facts within KGs, a task commonly referred to as link prediction (LP), has garnered substantial research interest.
Despite this success, LP remains challenging in various real-world scenarios. Traditional transductive LP assumes that all entities and relations are present during training at test time. As real-world KGs are dynamic, the emergence of unseen entities necessitates the retraining of the KGs. For instance, DBpedia observed a daily average addition of 200 entities between late 2015 and early 2016 [5].
## 1 緒論
知識圖譜（KGs）是事實三元組（實體、關聯、實體）的結構化集合，其中每個實體被描繪成一個節點，每個關聯則被表示為一條邊。知識圖譜在知識密集型任務中扮演著關鍵角色，例如智慧問答[1]、推薦系統[2]和語意搜索[3]。然而，由於現實世界數據的指數級增長，大多數現有的知識圖譜都存在噪音和不完整性的問題。例如，在開放知識圖譜 Freebase 中，約有 71% 的人物缺乏出生地資訊，99% 的人物沒有種族資訊[4]。因此，預測知識圖譜中缺失的事實，即通常所說的連結預測（LP），已引起了廣泛的研究興趣。
儘管取得了這些成功，但在各種現實場景中，LP 仍然具有挑戰性。傳統的直推式 LP 假設所有實體和關聯在訓練時都已存在於測試時。由於現實世界的知識圖譜是動態的，未見實體的出現需要重新訓練知識圖譜。例如，DBpedia 在 2015 年末至 2016 年初之間，每日平均新增 200 個實體[5]。
However, the frequent increase of entities can result in a notable rise in operational overhead. In response to this challenge, certain scholars focus on inductive link prediction (ILP), aiming to predict missing triplets involving unseen entities without training KGs from scratch.
Recent years have seen the emergence of successful graph neural networks (GNNs), such as GraphSAGE [6] and GraphSAINT [7], for inductive embedding within graphs. GNN-based models [8-10] facilitate the embedding of unseen entities into the original KG by aggregating features of neighboring nodes. Logic rule-based methods inherently exhibit inductive capabilities for triplets involving unseen-unseen entities, owing to the entity-independent nature of logic rules. Subgraphs represent effective path combinations connecting two target entities, providing a more comprehensive and informative representation than a single rule. Subgraph-based approaches, which leverage subgraph topological information and exploit GNNs' embedding capacity, have gained substantial attention [11-16].
Ali et al. categorize ILP into two settings: semi-inductive and fully-inductive [17]. In a semi-inductive setting, out-of-knowledge KG (OOKG) entities, as shown in Fig. 1a, are associated with seen entities in the KG. As shown by the green dashed lines in Fig. 1, the auxiliary triplet (Lily, friend_of, Justin) and existing triplets are used to predict the triplet (Justin, aunt_of, Grace). In the fully-inductive setting, predicting triplets in the emerging KG involves unseen entities. As shown by the orange dashed lines in Fig. 1c, all entities in the triplet (Zeke, mother_of, Alisa) are unseen. The overlooked bridging inductive links connect seen entities in the original KG with unseen entities in the emerging KG, such as the predicted triplet (Chioe, friend_of, Zeke) shown by the red dashed lines in Fig. 1. However, "Chioe" and"Zeke"exist within disconnected KGs, lacking the connected subgraph surrounding the target entity. As a result, the aforementioned models [11-15] heavily depend on target entity connectivity to reason within the connected subgraph, rendering them incapable of handling bridging-inductive links. Furthermore, in 2015, law enforcement achieved a breakthrough by connecting a criminal case with another apparently unrelated case, showcasing the significance and feasibility of bridging-inductive links.
The latest research on DEKG-ILP [18] has extended subgraph-based approaches to achieve bridging-inductive links. Firstly, contrastive learning-based relation-specific feature modeling (CLRM) leverages global relational semantic information shared between the original KG and the emerging KG. Secondly, GNN-based subgraph modeling (GSM) utilizes local topological information within the KGs. However, GSM's focus on simple local subgraphs entails certain limitations: (1) During subgraph extraction, as the k-hop value increases, the number of sampled neighbors exponentially escalates, possibly weakening the expressive capacity of GNNs; (2) Solely considering local subgraphs may lead to incomplete neighboring relations. For instance, an enclosing subgraph consists of a subset of nodes and all their relations in a KG. As shown in Fig. 1b, the enclosing subgraph includes the entities“Grace”, “Chioe”, and “David'along with the orange lines connecting them. Although the neighboring relations "aunt_of"and"friend_of”between “Grace”and“Chioe"contain valuable information, they are not included in the enclosing subgraph. Additionally, when enclosing subgraphs are empty or sparse, GSELI's effectiveness may be compromised. For instance, in Fig. 1c, no viable paths connect the entities"Zeke" and "Alisa."
Based on the aforementioned observations, we introduce GSELI, a model that effectively integrates global semantics and enhanced local subgraphs to facilitate bridging-inductive links. Inspired by LCILP [15] and SNRI [14], GSELI enhances the GSM module within DEKG-ILP [18] and introduces the pioneering GNN-based enhanced local subgraph (GELS) module. Specifically, the GELS module employs local clustering based on personalized PageRank (PPR) to sample tightly related subgraphs and utilizes neighboring relational features and relational paths to enhance the topological information of the local subgraph. In summary, our main contributions can be summarized as follows:
然而，實體頻繁增加會導致運營開銷顯著上升。為應對此挑戰，部分學者專注於歸納式連結預測（ILP），旨在無需從頭訓練知識圖譜的情況下，預測涉及未見實體的缺失三元組。
近年來，圖神經網絡（GNNs）如 GraphSAGE [6] 和 GraphSAINT [7] 在圖內的歸納式嵌入方面取得了成功。基於 GNN 的模型 [8-10] 透過聚合鄰近節點的特徵，促進了將未見實體嵌入原始知識圖譜。邏輯規則方法由於邏輯規則的實體獨立性，天生具備處理涉及未見-未見實體三元組的歸納能力。子圖代表了連接兩個目標實體的有效路徑組合，提供了比單一規則更全面、更豐富的表示。基於子圖的方法利用子圖拓撲資訊並發揮 GNNs 的嵌入能力，已獲得廣泛關注 [11-16]。
Ali 等人將 ILP 分為半歸納式和全歸納式兩種設定 [17]。在半歸納式設定中，如圖 1a 所示，知識圖譜外（OOKG）實體與知識圖譜中的已見實體相關聯。如圖 1 中的綠色虛線所示，輔助三元組（Lily, friend_of, Justin）和現有三元組被用來預測三元組（Justin, aunt_of, Grace）。在全歸納式設定中，預測新興知識圖譜中的三元組涉及未見實體。如圖 1c 中的橙色虛線所示，三元組（Zeke, mother_of, Alisa）中的所有實體都是未見的。被忽略的橋接歸納連結將原始知識圖譜中的已見實體與新興知識圖譜中的未見實體連接起來，例如圖 1 中紅色虛線所示的預測三元組（Chioe, friend_of, Zeke）。然而，「Chioe」和「Zeke」存在於不相連的知識圖譜中，缺乏圍繞目標實體的連接子圖。因此，前述模型 [11-15] 嚴重依賴目標實體的連通性在連接子圖內进行推理，使其無法處理橋接歸納連結。此外，2015 年，執法部門通過將一個刑事案件與另一個表面上無關的案件聯繫起來取得了突破，展示了橋接歸納連結的重要性與可行性。
關於 DEKG-ILP [18] 的最新研究擴展了基於子圖的方法以實現橋接歸納連結。首先，基於對比學習的關係特定特徵建模 (CLRM) 利用原始 KG 和新興 KG 之間共享的全局關係語義信息。其次，基於 GNN 的子圖建模 (GSM) 利用局部拓撲信息。然而，GSM 對簡單局部子圖的關註帶來了一些限制：(1) 在子圖提取過程中，隨著 k-hop 值的增加，採樣鄰居的數量呈指數級增長，可能削弱 GNN 的表達能力；(2) 僅考慮局部子圖可能導致鄰近關係不完整。例如，一個封閉子圖由節點子集及其在 KG 中的所有關係組成。如圖 1b 所示，封閉子圖包括實體「Grace」、「Chioe」和「David」以及連接它們的橙色線。儘管「Grace」和「Chioe」之間的鄰近關係「aunt_of」和「friend_of」包含有價值的信息，但它們未包含在封閉子圖中。此外，當封閉子圖為空或稀疏時，GSELI 的有效性可能會受到影響。例如，在圖 1c 中，沒有可行的路徑連接實體「Zeke」和「Alisa」。
基於上述觀察，我們介紹了 GSELI，這是一個有效整合全域語意和增強型區域子圖以促進橋接歸納連結的模型。受 LCILP [15] 和 SNRI [14] 的啟發，GSELI 增強了 DEKG-ILP [18] 中的 GSM 模組，并引入了開創性的基於 GNN 的增強型區域子圖 (GELS) 模組。具體而言，GELS 模組採用基於個人化 PageRank (PPR) 的區域聚類來取樣緊密相關的子圖，並利用鄰近的關聯特徵和關聯路徑來增強區域子圖的拓撲資訊。總而言之，我們的主要貢獻可歸納如下：
Fig. 1 Semi-inductive LP links
OOKG entities to the origi-
nal KG. Fully-inductive link
prediction (LP) predicts missing
links involving unseen-unseen
entities in the emerging KG.
The more challenging bridging-
inductive LP predicts missing
links involving unseen entities
from the original KG to the
emerging KG
圖 1 半歸納式 LP 將 OOKG 實體連結至原始 KG。全歸納式連結預測 (LP) 預測新興 KG 中涉及未見實體的缺失連結。更具挑戰性的橋接歸納式 LP 預測原始 KG 到新興 KG 之間涉及未見實體的缺失連結。
seen entities unseen entities enclosing subgraph neighboring relational path Semi-inductive Fully-inductive -> Bridging-inductive
(a). OOKG entity (b). Original KG (c). Emerging KG
已見實體 未見實體 封閉子圖 鄰近關係路徑 半歸納式 全歸納式 -> 橋接歸納式
(a). OOKG 實體 (b). 原始 KG (c). 新興 KG
• We are among the first scholars to study bridging-inductive link prediction and also conduct research on fully-inductive link prediction.
• A unified ILP framework, GSELI, integrates global semantics with enhanced local subgraphs, effectively addressing inductive link prediction tasks.
• We innovatively propose a GELS module, which uses PPR-based local clustering to sample tightly related subgraphs and integrates complete neighboring relations into the subgraphs.
• Experimental results across multiple benchmark datasets showcase GSELI's superiority over existing models in both fully-inductive and bridging-inductive link predictions.
• 我們是首批研究橋接歸納式連結預測的學者之一，同時也對全歸納式連結預測進行了研究。
• 一個統一的 ILP 框架，GSELI，整合了全域語意與增強的區域子圖，有效地解決了歸納式連結預測任務。
• 我們創新地提出了一個 GELS 模組，它使用基於 PPR 的局部聚類來採樣緊密相關的子圖，並將完整的鄰近關聯整合到子圖中。
• 在多個基準數據集上的實驗結果表明，GSELI 在全歸納式和橋接歸納式連結預測方面均優於現有模型。
The structure of the remaining sections of this paper is as follows: Sect. 2 reviews related work on link prediction and contrastive learning. Section 3 defines the terms used in this paper and classifies current inductive scenarios. Section 4 describes our proposed GSELI model in detail. Section 5 presents the experimental setup and results analysis. Section 6 discusses further experiments and analysis. Finally, Sect. 7 summarizes the findings.
本文其餘部分的結構如下：第二節回顧了連結預測和對比學習的相關工作。第三節定義了本文使用的術語並對當前的歸納場景進行了分類。第四節詳細描述了我們提出的 GSELI 模型。第五節介紹了實驗設置和結果分析。第六節討論了進一步的实验和分析。最后，第七节总结了研究结果。
## 2 Related work
This section describes related work, encompassing methods for both link prediction (see Sect. 2.1) and contrastive learning (see Sect. 2.2).
## 2 相關工作
本節描述了相關工作，涵蓋連結預測（見 2.1 節）和對比學習（見 2.2 節）的方法。
### 2.1 Link prediction methods
### 2.1 連結預測方法
#### 2.1.1 Transductive methods
Transductive LP requires that entities appearing during testing must have been seen during training. Translation-based methods [19, 20] treat relations as translational transformations projecting entities into lower-dimensional spaces. Tensor decomposition-based methods [21, 22] decompose KGs into three-way tensors for efficient computation of embedding representations. However, these methods overlook the hidden structural information present in KGs. In contrast, GNN-based approaches [23, 24] utilize GNNs to aggregate neighbor information for capturing global structural patterns. Unfortunately, the aforementioned methods necessitate retraining KGs from scratch when encountering unseen entities, leading to significant time and computational overhead.
#### 2.1.1 直推式方法
傳導式LP要求在測試期間出現的實體必須在訓練期間已經出現過。基於翻譯的方法[19, 20]將關係視為將實體投影到低維空間的平移轉換。基於張量分解的方法[21, 22]將知識圖譜分解為三向張量，以有效計算嵌入表示。然而，這些方法忽略了知識图譜中存在的隱藏結構信息。相比之下，基於GNN的方法[23, 24]利用GNN聚合鄰居信息以捕獲全局結構模式。不幸的是，上述方法在遇到未見實體時需要从頭重新訓練知識圖譜，導致顯著的時間和計算開銷。
#### 2.1.2 Rule-based inductive methods
Rule-based methods offer inherent inductive and interpretable capabilities when predicting triplets involving unseen-unseen entities in emerging KGs, as logical rules are independent of entities. Traditional rule-based methods primarily focus on identifying good rules within KGs [25-27]. However, these methods face challenges in scalability and computational complexity. Although neural networks are known for their robustness and efficiency in addressing such issues [28-30], they struggle to scale to large KGs and fully leverage complex structural information, limiting their reasoning performance.
#### 2.1.2 基於規則的歸納法
基於規則的方法在預測涉及新興知識圖譜中未見實體的三元組時，提供了固有的歸納和可解釋能力，因為邏輯規則與實體無關。傳統的基於規則的方法主要專注於在知識圖譜中識別優良規則[25-27]。然而，這些方法在可擴展性和計算複雜性方面面臨挑戰。雖然神經網絡以其在解決此類問題上的穩健性和效率而聞名[28-30]，但它們難以擴展到大型知識圖譜，也無法充分利用複雜的結構信息，從而限制了其推理性能。
#### 2.1.3 GNN-based inductive methods
Early researchers employed GNNs to aggregate seen neighboring entities around unseen entities to generate embeddings for the unseen entities [8-10]. However, these methods require unseen entities to be connected to seen entities, which does not address predicting missing triplets involving unseen-unseen entities in emerging KG. Recently, subgraph-based methods have rapidly advanced, leveraging subgraph topology information and the expressive power of GNNs [11-15]. However, the aforementioned models only predict links involving unseen entities in the original KG and the emerging KG separately. DEKG-ILP [18] further introduces bridging-inductive links that account for unseen entities carrying evolutionary information between the source KG and the emerging KG. However, this research is in its early stages and requires further exploration.
Table 1 summarizes the scenarios addressed by various LP models. Semi-inductive LP links a seen entity from the original KG to an unseen entity related to the original KG. Fully-inductive LP handles links between unseen-unseen entities, which are unrelated to the original KG. Bridging-inductive LP connects a seen entity from the original KG to an unseen entity unrelated to the original KG. In summary, semi-inductive LP leverages partial known information for higher accuracy and reliability, fully-inductive LP offers strong generalization and broad applicability, and bridging-inductive LP integrates known and unknown information, enhancing performance and adaptability in complex scenarios. The challenge levels are: bridging-inductive > fully-inductive > semi-inductive. Notably, DEKG-ILP and our proposed GSELI model handle both traditional LP tasks and more challenging bridging-inductive scenarios.
#### 2.1.3 基於 GNN 的歸納法
早期研究人員利用GNNs聚合圍繞未見實體的已見鄰近實體，為未見實體生成嵌入[8-10]。然而，這些方法要求未見實體與已見實體相連，這無法解決預測新興KG中涉及未見-未見實體的缺失三元組的問題。最近，基於子圖的方法發展迅速，利用子圖拓撲信息和GNNs的表達能力[11-15]。然而，前述模型僅分別預測原始KG和新興KG中涉及未見實體的鏈接。DEKG-ILP [18]進一步引入了橋接歸納鏈接，該鏈接考慮了在源KG和新興KG之間攜帶演化信息的未見實體。然而，這項研究仍處於早期階段，需要進一步探索。
表1總結了各種LP模型所處理的場景。半歸納式LP將原始KG中的一個已見實體鏈接到與原始KG相關的一個未見實體。全歸納式LP處理未見實體之間的鏈接，這些實體與原始KG無關。橋接歸納式LP將原始KG中的一個已見實體連接到與原始KG無關的一個未見實體。總之，半归纳式LP利用部分已知信息以获得更高的准确性和可靠性，全归纳式LP提供强大的泛化能力和广泛的适用性，而桥接归纳式LP整合已知和未知信息，从而在复杂场景中增强性能和适应性。挑战级别为：桥接归纳式 > 全归纳式 > 半归纳式。值得注意的是，DEKG-ILP和我们提出的GSELI模型都能处理传统的LP任务以及更具挑战性的桥接归纳式场景。
Table 1 Summary of applicable scenarios for link prediction models. The ✓ means you can handle the scenarios, and the x means you can't

| Models | Transductive link prediction | Inductive link prediction |
| :--- | :--- | :--- |
| | | Semi-inductive | Fully-inductive | Bridging-inductive |
| Transductive models |
| TransE [19] | ✓ | x | x | x |
| RESCAL [21] | ✓ | x | x | x |
| RGCN [23] | ✓ | x | x | x |
| Inductive models |
| MEAN [8] | ✓ | ✓ | x | x |
| RuleN [26] | ✓ | ✓ | ✓ | x |
| GraIL [11] | ✓ | ✓ | ✓ | x |
| DEKG-ILP [18] | ✓ | ✓ | ✓ | ✓ |
| GSELI (ours) | ✓ | ✓ | ✓ | ✓ |
表1 連結預測模型適用場景總結。✓ 表示您可以處理該場景，x 表示您無法處理。

| 模型 | 轉導式連結預測 | 歸納式連結預測 |
| :--- | :--- | :--- |
| | | 半歸納式 | 全歸納式 | 橋接歸納式 |
| 轉導式模型 |
| TransE [19] | ✓ | x | x | x |
| RESCAL [21] | ✓ | x | x | x |
| RGCN [23] | ✓ | x | x | x |
| 歸納式模型 |
| MEAN [8] | ✓ | ✓ | x | x |
| RuleN [26] | ✓ | ✓ | ✓ | x |
| GraIL [11] | ✓ | ✓ | ✓ | x |
| DEKG-ILP [18] | ✓ | ✓ | ✓ | ✓ |
| GSELI (我們的) | ✓ | ✓ | ✓ | ✓ |
### 2.2 Contrastive learning
Contrastive learning (CL) is a self-supervised learning framework widely employed in natural language processing and computer vision. It aims to obtain representations by encoding similar or dissimilar samples. Recently, researchers have explored the application of CL in the field of ILP. SimKGC [31] introduces negative sampling (in-batch negatives, pre-batch negatives, and self-negatives) to increase the size of negative samples to thousands. RPC-IR [32] addresses the issue of deficient rule supervision in subgraphs by constructing positive and negative relational paths. SGI [33] maximizes the mutual information (MI) between the target relation and its enclosing subgraph, and it selects negative samples using a pre-trained MI estimator. SNRI [14] models subgraphs in a global manner by maximizing the subgraph-graph MI.
### 2.2 對比學習
對比學習（CL）是一種自我監督學習框架，廣泛應用於自然語言處理和電腦視覺。它旨在通過編碼相似或不相似的樣本來獲得表示。最近，研究人員已經探索了 CL 在 ILP 領域的應用。SimKGC [31] 引入了負採樣（批內負樣本、批前負樣本和自負樣本）來將負樣本的大小增加到數千個。RPC-IR [32] 通過構建正負關係路徑來解決子圖中規則監督不足的問題。SGI [33] 最大化了目標關係與其封閉子圖之間的互信息（MI），並使用預訓練的 MI 估計器來選擇負樣本。SNRI [14] 通過最大化子圖-圖的 MI 來全局建模子圖。
## 3 Definition
This section defines the original KG and the emerging KG (see Sect. 3.1) and categorizes LP tasks into transductive and inductive LP (see Sect. 3.2). Inductive link prediction is further subdivided into semi-inductive, fully-inductive, and bridging-inductive LP (see Sect. 3.3).
## 3 定義
本節定義了原始 KG 和新興 KG（參見第 3.1 節），並將 LP 任務分為直推式和歸納式 LP（參見第 3.2 節）。歸納式鏈接預測進一步細分為半歸納式、全歸納式和橋接歸納式 LP（參見第 3.3 節）。
### 3.1 Original and emerging KG
**Definition 1** (Original knowledge graph) The original KG is defined as: G(E,R) = {(h, r,t) | h, t ∈ E, r∈ R} ⊆ E × R × E, where E and R represent the sets of entities and relations, respectively. The fact is in a triplet format (h, r, t), where h, t, and r represent the head entity, the tail entity, and the relation between them, respectively.
### 3.1 原始知識圖譜與新興知識圖譜
**定義 1** (原始知識圖譜) 原始知識圖譜定義為：G(E,R) = {(h, r,t) | h, t ∈ E, r∈ R} ⊆ E × R × E，其中 E 和 R 分別表示實體和關係的集合。事實以三元組 (h, r, t) 的格式表示，其中 h、t 和 r 分別表示頭實體、尾實體以及它們之間的關係。
**Definition 2** (Emerging knowledge graph) The emerging KG is defined as: G'(E', R) = {(h, r, t) | h, t ∈ E', r∈ R} ⊆ E'×R×E', where E'∩E= Ø. The emerging KG G'(E', R) consists of the unseen entity set E' and the seen relation set R shared with the original KG G(E, R).
**定義 2** (新興知識圖譜) 新興知識圖譜定義為：G'(E', R) = {(h, r, t) | h, t ∈ E', r∈ R} ⊆ E'×R×E'，其中 E'∩E= Ø。新興知識圖譜 G'(E', R) 由未見實體集 E' 和與原始知識圖譜 G(E, R) 共享的已見關係集 R 組成。
### 3.2 Transductive and inductive LP
LP can be categorized into two scenarios: transductive and inductive LP.

**Definition 3** (Transductive link prediction) Transductive LP is defined as predicting missing triplets (h, r, t) ∈ E×R×E, where both the entity set E and the relation set R are seen.

**Definition 4** (Inductive link prediction) Inductive LP is defined as predicting missing triplets (h, r, t) ∈ (EUE*) × (R U R*) × (EU E*) where (h, r, t) contains at least one unseen element (entity or relation). Here, En E* = Ø and R ∩ R* = Ø, with E* and R* representing the sets of unseen entities and unseen relations, respectively.
### 3.2 直推式與歸納式 LP
LP 可分為兩種情境：直推式和歸納式 LP。

**定義 3** (直推式連結預測) 直推式 LP 定義為預測缺失的三元組 (h, r, t) ∈ E×R×E，其中實體集 E 和關係集 R 皆為已知。

**定義 4** (歸納式連結預測) 歸納式 LP 定義為預測缺失的三元組 (h, r, t) ∈ (EUE*) × (R U R*) × (EU E*)，其中 (h, r, t) 包含至少一個未見元素（實體或關係）。此處，En E* = Ø 且 R ∩ R* = Ø，E* 和 R* 分別代表未見實體和未見關係的集合。
### 3.3 Semi-inductive, fully-inductive, and bridging-inductive LP
Current research classifies inductive LP into three categories: semi-inductive (SI), fully-inductive (FI), and bridging-inductive (BI). This study focuses on scenarios where entities are unseen but relations are seen. Below are their definitions.

**Definition 5** (Semi-inductive link prediction) Semi-inductive LP is defined as predicting the missing triplets (h, r, t) ∈ (E× R x Ë) U (Ë × R × E). Here, Ë represents the unseen entities, where E n Ë = Ø. In this scenario, either h or t is from the original KG, and the other is an unseen entity associated with the original KG (i.e., OOKG entity). As shown in Fig. 1, the triplet (unseen head entity"Justin", seen relation"aunt_of", unseen tail entity"Grace") is predicted.
### 3.3 半歸納式、全歸納式及橋接歸納式 LP
目前的研究將歸納式 LP 分為三類：半歸納式 (SI)、全歸納式 (FI) 和橋接歸納式 (BI)。本研究著重於實體未見但關係已見的場景。以下是它們的定義。

**定義 5** (半歸納式連結預測) 半歸納式 LP 定義為預測遺失的三元組 (h, r, t) ∈ (E× R x Ë) U (Ë × R × E)。此處，Ë 代表未見實體，其中 E n Ë = Ø。在此情境中，h 或 t 來自原始知識圖譜，另一個則是與原始知識圖譜相關的未見實體 (即 OOKG 實體)。如圖 1 所示，預測了三元組 (未見頭實體「Justin」、已見關係「aunt_of」、未見尾實體「Grace」)。
**Definition 6** (Fully-inductive link prediction) Fully-inductive LP is defined as predicting the missing triplets of the emerging KG (h, r,t) ∈ E' × R × E'. In this scenario, both h and t are unseen entities that are not associated with the original KG, i.e., from the emerging KG. As shown in Fig. 1, the triplet (unseen head entity"Zeke", seen relation"mother_of", unseen tail entity “Alisa”) is predicted.
**定義 6** (全歸納式連結預測) 全歸納式 LP 定義為預測新興知識圖譜中的缺失三元組 (h, r, t) ∈ E' × R × E'。在此場景中，h 和 t 均為與原始知識圖譜無關的未見實體，即來自新興知識圖譜。如圖 1 所示，預測了三元組 (未見頭實體「Zeke」，已見關係「mother_of」，未見尾實體「Alisa」)。
**Definition 7** (Bridging-inductive link prediction) Bridging-inductive LP is defined as predicting the missing triplets from the original KG to the emerging KG (h, r, t) ∈ (E× R x E') u (E' ×R×E). In this scenario, either h or t is from the original KG, and the other is an unseen entity from the emerging KG that is not associated with the original KG. As shown in Fig. 1, the triplet (seen head entity "Chioe”, seen relation“friend_of", unseen tail entity"Zeke") is predicted.
**定義 7** (橋接歸納式連結預測) 橋接歸納式 LP 定義為預測從原始 KG 到新興 KG 的缺失三元組 (h, r, t) ∈ (E× R x E') u (E' ×R×E)。在此場景中，h 或 t 來自原始 KG，另一個是來自與原始 KG 無關的新興 KG 的未見實體。如圖 1 所示，預測了三元組 (已見頭實體「Chioe」，已見關係「friend_of」，未見尾實體「Zeke」)。
Notably, although both semi-inductive and bridging-inductive LP address the issue of seen-unseen entities, they differ significantly. In semi-inductive LP, the unseen entities are related to the original KG (i.e., OOKG entities). In contrast, in bridging-inductive LP, the unseen entities are not associated with the original KG (from an emerging KG), making bridging-inductive LP more challenging.
值得注意的是，雖然半歸納式和橋接歸納式 LP 都處理已見-未見實體的問題，但它們有顯著差異。在半归纳式 LP 中，未见实体与原始 KG（即 OOKG 实体）相关。相比之下，在桥接归纳式 LP 中，未见实体与原始 KG（来自新兴 KG）无关，这使得桥接归纳式 LP 更具挑战性。
## 4 Methods
This section offers an overview of our proposed GSELI model(see Sect. 4.1), and provides a detailed introduction to its three constituent components: the CL-based global semantic features module (CLSF) (see Sect. 4.2), the GNN-based enhanced local subgraph module (GELS) (see Sect. 4.3), and the joint training strategy (see Sect. 4.4).
## 4 方法
本節概述了我們提出的 GSELI 模型（參見 4.1 節），並詳細介紹了其三個組成部分：基於 CL 的全局語義特徵模塊（CLSF）（參見 4.2 節）、基於 GNN 的增強局部子圖模塊（GELS）（參見 4.3 節）以及聯合訓練策略（參見 4.4 節）。
### 4.1 Overview of GSELI
The overview of our proposed model GSELI is shown in Fig. 2. The model is composed of three parts: CLSF, GELS, and a joint training strategy. Compared to existing studies [11-14, 16] that focus solely on fully-inductive LP, this model supports both fully-inductive and bridging-inductive LP.
### 4.1 GSELI 概觀
我們提出的 GSELI 模型概覽如圖 2 所示。該模型由三部分組成：CLSF、GELS 和聯合訓練策略。與現有的僅專注於全歸納式 LP 的研究 [11-14, 16] 相比，本模型同時支援全歸納式和橋接歸納式 LP。
Fig. 2 Overview of GSELI. The CLSF module merges relation-component tables A and A, to represent relation-specific semantic features su and s₁. It further generates positive samples Apos, Apos and negative samples Anes, Anes, forming shos, spos and sneg, sneg, optimized through contrastive learning. The GELS module employs local clustering to identify the most relevant subgraphs, initializing node representations h with topological position hos and neighbor relation hel features. It combines local subgraph he and neighboring relation path pe information to generate the subgraph representation sg, and calculates the loss C through contrastive Lcon and supervised Lsup learning
圖2 GSELI 概覽。CLSF 模組融合關係成分表 A 和 A，以表示特定關係的語意特徵 su 和 s₁。它進一步生成正樣本 Apos、Apos 和負樣本 Anes、Anes，形成 shos、spos 和 sneg、sneg，並透過對比學習进行優化。GELS 模組採用局部聚類來識別最相關的子圖，使用拓撲位置 hos 和鄰居關係 hel 特徵來初始化節點表示 h。它結合了局部子圖 he 和鄰居關係路徑 pe 資訊來生成子圖表示 sg，並透過對比學習 Lcon 和監督學習 Lsup 來計算損失 C。
Specifically, CLSF extracts globally shared relation-based semantic features across knowledge graphs and optimizes these features using a semantic-aware contrastive learning strategy. The CLSF module represents the relation-specific semantic features s₁₁ and s, by merging the relation component tables A and A, of the target entities u and v. By generating positive samples Apos and Apos, and negative samples Anes and Ares from these tables, it creates positive sos and spos and negative sne and she relation-specific semantic features, which are further optimized through contrastive learning. Notably, by sharing the relational space, entities in both the original KG and emerging KGs are connected. This resolves the issue of lacking connected subgraphs around target entities, thereby improving bridging-inductive LP.
GELS extracts topological features from local subgraphs around target entities through three sub-modules: subgraph extraction, node feature initialization, and subgraph embedding. It first identifies the most relevant subgraphs using local clustering techniques, then initializes node representations h for node i with topological position features hos and neighbor relation features hel using a target relation-aware neighbor attention mechanism arr₁. Finally, it combines local subgraph information he and neighboring relation path information pg to generate the final topological representation of the subgraph sg. Notably, this module leverages local topological information around target entities in two disconnected KGs, facilitating both fully-inductive and bridging-inductive LP.
Finally, the model is trained using joint contrastive Lcon and supervised learning Lsup losses, where the supervised learning computes the target link score by combining the semantic perspective learned from CLSF and the topological perspective learned from GELS.
具體來說，CLSF 提取跨知識圖譜的全局共享基於關聯的語義特徵，並使用語義感知對比學習策略優化這些特徵。CLSF 模組透過合併目標實體 u 和 v 的關聯成分表 A 和 A 來表示特定關聯的語義特徵 s₁₁ 和 s。透過從這些表中生成正樣本 Apos 和 Apos，以及負樣本 Anes 和 Ares，它創建了正向 sos 和 spos 以及負向 sne 和 she 的特定關聯語義特徵，這些特徵透過對比學習進一步優化。值得注意的是，透過共享關聯空間，原始 KG 和新興 KG 中的實體得以连接。這解決了目標實體周圍缺乏連接子圖的問題，從而改善了橋接歸納 LP。
GELS 透過三個子模組從目標實體周圍的局部子圖中提取拓撲特徵：子圖提取、節點特徵初始化和子圖嵌入。它首先使用局部聚類技術識別最相關的子圖，然後使用目標關聯感知鄰居注意力機制 arr₁ 初始化節點 i 的節點表示 h，其中包含拓撲位置特徵 hos 和鄰居關聯特徵 hel。最後，它結合局部子圖資訊 he 和鄰居關聯路徑資訊 pg，生成子圖 sg 的最終拓撲表示。值得注意的是，此模組利用兩個不相連的 KG 中目標實體周圍的局部拓撲資訊，促進了全歸納和橋接歸納 LP。
最後，該模型使用聯合對比 Lcon 和監督學習 Lsup 損失進行訓練，其中監督學習通過結合從 CLSF 學到的語義視角和從 GELS 學到的拓撲視角來計算目標鏈接分數。
### 4.2 CLSF module
### 4.2 CLSF 模組
#### 4.2.1 Relation-specific semantic features
The semantics of entities in KGs are influenced by their associated relations. Following [18], we utilize relational features to represent the seen entities in the original KG. Given that both the original and the emerging KGs share relations R, we can also employ these relational features to represent the unseen entities in the emerging KG. This approach allows us to embed both seen and unseen entities into a unified feature space, effectively resolving the issue of disconnected entities between the original and emerging KGs.
Specifically, we achieve the semantic representation s; of entity i by integrating the relation-component table A; of entity i with the relation-specific features F of the relation R:
s₁ =y (A₁, F) = Σα / Σ=1 A₁ ={a | i ∈ E, r₁ ∈ R}, F ={f, | r, ∈ R}, (1) (2) (3)
where y() is the fusion function, |R| = n, f, is the semantic embedding of relation r₁. A; is a table that describes the semantic information of entity i by recording the number of triplets a associated with different relations r₁. If there are no triplets associated with relation r₁, a is set to 0. The relation-component tables A₁₁ and A, for entities u and v are shown in Fig. 2.
Notably, this method models data in an entity-independent manner, allowing for natural generalization to the representation of unseen entities. Finally, the semantic likelihood score psem(u, r₁, v) of the target triplet is calculated as follows:
psem (u, rt, v) = (su,rsem, s), (4)
where rsem is the learned embedding of relation r, from the semantic perspective, and (,,) denotes the element-wise product of embedding vectors inspired by DistMult [22].
#### 4.2.1 特定關係的語意特徵
知識圖譜中實體的語義受其相關聯的關係影響。遵循[18]的方法，我們利用關係特徵來表示原始知識圖譜中的已見實體。鑑於原始知識圖譜和新興知識圖譜共享關係 R，我們也可以利用這些關係特徵來表示新興知識圖譜中的未見實體。這種方法使我們能够將已見和未見實體嵌入到統一的特徵空間中，有效解決原始知識圖譜和新興知識圖譜之間實體不連通的問題。
具體來說，我們透過整合實體 i 的關係成分表 A; 與關係 R 的特定關係特徵 F，來得到實體 i 的語意表示 s;：
s₁ =y (A₁, F) = Σα / Σ=1 A₁ ={a | i ∈ E, r₁ ∈ R}, F ={f, | r, ∈ R}, (1) (2) (3)
其中 y() 是融合函數，|R| = n，f, 是關係 r₁ 的語義嵌入。A; 是一個表格，通过記錄与不同關係 r₁ 相關的三元組数量 a 来描述實體 i 的語義資訊。如果沒有与關係 r₁ 相關的三元組，則 a 设為 0。实体 u 和 v 的关系-成分表 A₁₁ 和 A, 如圖 2 所示。
值得注意的是，此方法以實體無關的方式為數據建模，從而可以自然地推廣到未見實體的表示。最後，目標三元組的語意可能性分數 psem(u, r₁, v) 的計算方式如下：
psem (u, rt, v) = (su,rsem, s), (4)
其中，rsem 是從語意角度學習到的關係 r 的嵌入，而（,,）表示受 DistMult [22] 啟發的嵌入向量的元素積。
#### 4.2.2 Semantic-aware contrastive learning
To optimize relation-specific features F during training, we utilize a contrastive learning-based approach along with a semantic-aware sampling strategy.
Following [18], we first model the semantic changes of entities by defining three random operations for the relation-component table A; of each entity i: relation variation o₁(), relation addition 02(·), and relation deletion 03(.). Specifically, operations 0₁(·), 02(·), and 03(·) involve selecting a number from sets {a | a ∈ A₁ ^ a ≠ 0}, {a | a ∈ A₁ ^ a = 0}, and {a | a ∈ A₁ ^ a ≠ 0} respectively, and are then randomly assigned integers within the ranges [1, m; 0], [1, m; 0], and 0 respectively, where e is the hyperparameter of the scaling factor, controlling the degree of random variation or addition of these relations, and m; is the average number of triplets associated with each relation, expressed as:
m₁ = Σα / |{a¦ ¦ ¦ ∈ A₁ ^ a = 0}| (5)
#### 4.2.2 語意感知對比學習
為在訓練過程中優化特定關係的特徵 F，我們採用基於對比學習的方法以及語義感知取樣策略。
依循 [18]，我們首先藉由定義每個實體 i 的關係成分表 A; 的三個隨機操作來模擬實體的語義變化：關係變異 o₁()、關係增加 02(·) 和關係刪除 03(.)。具體而言，操作 0₁(·)、02(·) 和 03(·) 分別涉及從集合 {a | a ∈ A₁ ^ a ≠ 0}、{a | a ∈ A₁ ^ a = 0} 和 {a | a ∈ A₁ ^ a ≠ 0} 中選取一个數字，然後分別隨機分配範圍在 [1, m; 0]、[1, m; 0] 和 0 的整數，其中 e 是縮放因子的超參數，用以控制這些關係的隨機變異或增加的程度，而 m; 是與每個關係相關聯的三元組的平均數，表示為：
m₁ = Σα / |{a¦ ¦ ¦ ∈ A₁ ^ a = 0}| (5)
Secondly, by employing these three random operations, we utilize operation 0₁(·) to generate Aºs, and apply both operations 02(.) and 03(.) to derive Ares. We subsequently produce the corresponding sPos and sneg, denoted as:
$pos = (Apos, F), $res = (Anes, F). (6)
Overall, the functions of the relation-component table are: (1) describing the semantic information of entities by recording their associations with various relations; (2) generating positive and negative samples during model training and optimizing the model through contrastive learning; and (3) constructing the semantic representation of entities by combining the relation-component table with relation-specific features. This approach enables the model to share relation features across different KGs, thereby enhancing its generalization ability.
Then the contrastive learning loss Loon is calculated to optimize the relation-specific features by maximizing the similarity between positive samples (spos, s₁) and minimizing the similarity between negative samples (se, s₁), denoted as:
Lcon = [sim(spos, si) – sim(se, s₁) + 7]+, (7)
where sim() represents a similarity function that measures the similarity between two embedded vectors by computing their Euclidean distance, y is the hyperparameter controlling the margin, and [x]+ = max {0, x}.
其次，透過採用這三個隨機操作，我们利用操作 0₁(·) 來生成 Aºs，並同時應用操作 02(.) 和 03(.) 來推導出 Ares。我們接著產生對應的 sPos 和 sneg，表示為：
$pos = (Apos, F), $res = (Anes, F). (6)
總體而言，關係成分表的功能如下：（1）透過記錄實體與各種關係的關聯性來描述實體的語意資訊；（2）在模型訓練期間生成正負樣本，並透過對比學習來優化模型；以及（3）結合關係成分表與特定關係的特徵來建構實體的語意表示。這種方法使模型能夠跨不同知識圖譜共享關係特徵，從而增強其泛化能力。
然後計算對比學習損失 Loon，以優化關係特定特徵，方法是最大化正樣本（spos, s₁）之間的相似性，並最小化負樣本（se, s₁）之間的相似性，表示為：
Lcon = [sim(spos, si) – sim(se, s₁) + 7]+, (7)
其中 sim() 代表一個相似度函數，它透過計算兩個嵌入向量之間的歐幾里德距離來衡量它們的相似度；y 是控制邊界的超參數；且 [x]+ = max {0, x}。
### 4.3 GELS module
Algorithm 1 PPR-based subgraph extraction
Require: Adjacency matrix A, Seed nodes Ω = {u, v}, Teleportation probability a, Residual error e
Ensure: Local cluster subgraph G(u, rt, v)
1: Initialize:
2: PageRank vector p ← 0
3: Residual vector r[i] ← 1/|Ω| for i ∈ Ω
4: Queue q ← deque(Ω)
5: Degree vector d ← Σ(j=0 to n-1) Aij
6: Node volume v ← 0; V ← Σ(i=0 to n-1) d[i]
7: Cut value c ← 0
8: Best conductance Φ_best ← 1
9: Best community set S_best ← {Ω[0]}
10: while queue is not empty do
11: i ← queue.pop()
12: δ_pro ← r[i] – 0.5ed[i]
13: r[i] ← 0.5ed[i]
14: p[i] ← p[i] + (1 – α)δ_pro
15: δ_dis ← αδ_pro
16: for each neighbor k of i do
17: r_old[k] ← r[k]
18: r[k] ← r[k] + δ_dis * A_ik / d_k
19: if r[k] > ed[k] ^ r_old[k] < ed[k] then
20: queue.append(k)
21: end if
22: end for
23: end while
24: Sort vertices by score: p[i1] ≥ p[i2] ≥ ... ≥ p[iz]
25: for node ij in sorted nodes do
26: v ← v + d[ij]
27: for neighbor x in neighbors of ij do
28: c ← c - 1 if x ∈ S else c + 1
29: end for
30: S ← S U {ij}
31: if v = V then
32: break
33: end if
34: Φ ← c / min(v, V-v)
35: if Φ < Φ_best then
36: Φ_best ← Φ
37: S_best ← S
38: end if
39: end for
40: Prune nodes not on the path between u and v, resulting in the final local cluster subgraph G(u,rt, v).
### 4.3 GELS 模組
演算法 1 基於 PPR 的子圖提取
要求：鄰接矩陣 A，種子節點 Ω = {u, v}，瞬移機率 a，殘差 e
確保：局部聚類子圖 G(u, rt, v)
1：初始化：
2: PageRank 向量 p ← 0
3：對於 i ∈ Ω，殘差向量 r[i] ← 1/|Ω|
4: 佇列 q ← deque(Ω)
5：度向量 d ← Σ(j=0 to n-1) Aij
6：節點體積 v ← 0；V ← Σ(i=0 to n-1) d[i]
7: 切割值 c ← 0
8: 最佳電導 Φ_best ← 1
9：最佳社群集合 S_best ← {Ω[0]}
10：當佇列非空時
11： i ← 佇列彈出
12：δ_pro ← r[i] – 0.5ed[i]
13：r[i] ← 0.5ed[i]
14: p[i] ← p[i] + (1 – α)δ_pro
15：δ_dis ← αδ_pro
16：對於 i 的每個鄰居 k
17：r_old[k] ← r[k]
18：r[k] ← r[k] + δ_dis * A_ik / d_k
19：如果 r[k] > ed[k] 且 r_old[k] < ed[k] 那麼
20：佇列附加 k
21：結束如果
22：結束 for
23：結束 while
24：按分數對頂點排序：p[i1] ≥ p[i2] ≥ ... ≥ p[iz]
25：對於已排序節點中的節點 ij
26：v ← v + d[ij]
27：對於 ij 的鄰居 x
28：如果 x ∈ S，則 c ← c - 1，否則 c ← c + 1
29：結束 for
30：S ← S U {ij}
31：如果 v = V 那麼
32：中斷
33：結束如果
34：Φ ← c / min(v, V-v)
35：如果 Φ < Φ_best 那麼
36：Φ_best ← Φ
37：S_best ← S
38：結束如果
39：結束 for
40：修剪不在 u 和 v 之間路徑上的節點，得到最終的局部聚類子圖 G(u,rt, v)。
#### 4.3.1 Subgraph extraction
The PPR-based subgraph extraction algorithm involves two main steps. First, it employs an approximate PPR method to score nodes based on their proximity to a seed set of target entities. Second, it creates nested local clusters in descending order of these scores and evaluates them using conductance. The detailed procedure is provided in Algorithm 1.
The PageRank vector p is initialized to zero, and for each node i in the seed set Ω, the residual vector r[i] is set to 1/|Ω|. An empty double-ended queue (deque) is created to store nodes for processing. The degree vector d represents each node's degree and is obtained by summing the elements of the i-th column of the adjacency matrix A. Each element Aij indicates the connection between node i and node j, d[i] denotes the degree of node i. The node volume v to 0 and V to the sum of the degrees of all nodes. The cut value c are set to zero. The initial best conductance Φ_best is set to 1, and the best community set S_best is initialized with the first node in the seed set Ω. These steps lay the groundwork for the subsequent PPR-based subgraph extraction.
First, initialize a deque 'queueʻ to store nodes to be processed. In the main loop, when 'queue' is not empty, pop a node i from the deque and calculate δ_pro, the value that the current node will propagate to its neighboring nodes. Update the residual value r[i] of node i and its PageRank vector p[i]. Next, compute δ_dis to be distributed to the neighboring nodes by multiplying δ_pro by the teleportation probability α. For each neighboring node k of node i, save its old residual value r_old[k] and update its residual value r[k]. Calculate the threshold (εd[k]) for neighboring node k. If the updated residual value exceeds the threshold and the old residual value is less than the threshold, add the neighboring node to the deque 'queue'. The main loop ends when the deque is empty. This process iteratively traverses and updates the PageRank values and residuals of each node and its neighbors until all nodes are processed, resulting in a vector p containing the approximate PPR values for each node.
Then the best sweep set is determined using PageRank scores and conductance calculations, resulting in a local cluster subgraph. First, vertices are sorted in descending order by PageRank scores: p[i₁] ≥ p[iz] ≥ ≥ p[i]. Next, the sorted vertices i; are iterated. For each vertex, its volume v is updated by adding the degree d[i;] of the current vertex i;. The neighbors x of each vertex i; are then iterated. Based on whether a neighbor is in the sweep set S, the cut value c is adjusted: if x ∈ S, c is decreased by 1; otherwise, c is increased by 1. The current vertex i; is then added to the sweep set S. If the volume v reaches the specified value V, the loop terminates. The conductance $ of the current subset is then calculated and compared with the best conductance Φ_best. If the current conductance is smaller, the best conductance Φ_best and the best sweep set S_best are updated. Finally, by pruning nodes not on the path between seed nodes u and v, a tightly connected and highly relevant local cluster subgraph G(u, r₁, v) is obtained.
#### 4.3.1 子圖擷取
基於PPR的子圖提取算法包括兩個主要步驟。首先，它採用一種近似的PPR方法，根據節點與目標實體種子集的接近程度對其進行評分。其次，它按照这些分數的降序創建嵌套的局部簇，並使用電導率對其進行評估。詳細過程見演算法1。
PageRank 向量 p 初始化為零，對於種子集 Ω 中的每個節點 i，殘差向量 r[i] 設為 1/|Ω|。創建一個空的雙端佇列 (deque) 來儲存待處理的節點。度向量 d 表示每個節點的度，是透過對鄰接矩陣 A 的第 i 行元素求和得到的。每個元素 Aij 表示節點 i 和節點 j 之間的連接，d[i] 表示節點 i 的度。節點體積 v 設為 0，V 設為所有節點度的總和。切割值 c 設為零。初始最佳電導 Φ_best 設為 1，最佳社群集 S_best 初始化為種子集 Ω 中的第一個節點。這些步驟為後續基於 PPR 的子圖提取奠定了基礎。
首先，初始化一个雙端佇列 'queue' 來儲存待處理的節點。在主迴圈中，當 'queue' 不為空時，從佇列中彈出一个節點 i，並計算 δ_pro，即當前節點將傳播給其鄰近節點的值。更新節點 i 的殘差值 r[i] 及其 PageRank 向量 p[i]。接下來，透過將 δ_pro 乘以瞬移機率 α 來計算將分配給鄰近節點的 δ_dis。對於節點 i 的每個鄰近節點 k，保存其舊的殘差值 r_old[k] 並更新其殘差值 r[k]。計算鄰近節點 k 的閾值 (εd[k])。如果更新後的殘差值超過閾值且舊的殘差值小於閾值，則將該鄰近節點加入佇列 'queue'。當佇列為空時，主迴圈结束。此過程迭代地遍歷並更新每個節點及其鄰居的 PageRank 值和殘差，直到所有節點都被處理完畢，最終得到一個包含每個節點近似 PPR 值的向量 p。
然後使用 PageRank 分數和電導計算來確定最佳掃描集，從而產生局部聚類子圖。首先，根據 PageRank 分數按降序對頂點進行排序：p[i₁] ≥ p[iz] ≥ ≥ p[i]。接下來，對排序後的頂點 i; 進行迭代。對於每個頂點，通过加上当前顶点 i; 的度 d[i;] 來更新其體積 v。然後對每個頂點 i; 的鄰居 x 進行迭代。根據鄰居是否在掃描集 S 中，調整切割值 c：如果 x ∈ S，則 c 減 1；否則 c 加 1。然後将当前顶点 i; 添加到掃描集 S 中。如果體積 v 達到指定值 V，則循環終止。然後計算當前子集的電導 $，並與最佳電導 Φ_best 進行比較。如果當前電導較小，則更新最佳電導 Φ_best 和最佳掃描集 S_best。最後，通过修剪不在种子节点 u 和 v 之间路径上的节点，得到一个紧密连接且高度相关的局部聚类子图 G(u, r₁, v)。
#### 4.3.2 Node initialization
Since GNN requires a node feature matrix as input [34], ILP cannot utilize node attributes. Following [14], we initialize the node features h of the node i by combining topological locational features hºs and neighboring relational features hel.
Specifically, we initially acquire topological locational features hos through a double radius vertex labeling scheme, denoted as:
hos = [one-hot(d(i, u)) ⊕ one-hot(d(i, v))], (8)
where the two target nodes u and v are uniquely labeled as (0,1) and (1,0), respectively, d(i, u) is the shortest distance from the node i to the target node u without any path through v (similarly, d(i, v) is the shortest distance to the target node v without any path through u), and denotes the concatenation of two vectors. To address the topological limitations between the original KGs and the emerging KGs, and facilitate the implementation of bridging-inductive links, following [18], we retain the nodes {i | d(i, u) > t v d(i, v) > t} in t-top while setting d(i, •) = −1 and one-hot(-1) = 0 if d(i,) > t.
Secondly, we employ a target relation-aware neighbor attention mechanism for message passing on the node i to acquire neighboring relational features hel, expressed as:
hel = Σα_Arr reN, (i) = softmax(r, r₁) = exp(rTr₁) / Σ'∈N,(i) exp(r'Tr₁) (9) (10)
where N, (i) is the set of direct neighbors of the node i under relation r, ar, is the importance of relation r to the node i under target relation r₁, and r and r, represent the embeddings of the neighbor relation r and the target relation r₁ respectively.
#### 4.3.2 節點初始化
由於 GNN 需要一個節點特徵矩陣作為輸入 [34]，ILP 無法利用節點屬性。遵循 [14] 的方法，我们透過結合拓撲位置特徵 hºs 和鄰近關係特徵 hel 來初始化節點 i 的節點特徵 h。
具體來說，我们最初透過雙半徑頂點標記方案獲取拓撲位置特徵 hos，表示為：
hos = [one-hot(d(i, u)) ⊕ one-hot(d(i, v))], (8)
其中兩個目標節點 u 和 v 分別被唯一標記為 (0,1) 和 (1,0)，d(i, u) 是從節點 i 到目標節點 u 的最短距離，且路徑不經過 v（類似地，d(i, v) 是到目標節點 v 的最短距離，且路徑不經過 u），並表示兩個向量的串接。為了處理原始 KG 和新興 KG 之間的拓撲限制，并促進橋接歸納連結的實作，我們遵循 [18] 的作法，在 t-top 中保留節點 {i | d(i, u) > t v d(i, v) > t}，同时若 d(i,) > t，则設定 d(i, •) = −1 且 one-hot(-1) = 0。
其次，我們對節點 i 採用目標關係感知鄰居注意力機制进行訊息傳遞，以獲取鄰近關係特徵 hel，表示为：
hel = Σα_Arr reN, (i) = softmax(r, r₁) = exp(rTr₁) / Σ'∈N,(i) exp(r'Tr₁) (9) (10)
其中 N, (i) 是在關係 r 下節點 i 的直接鄰居集合，ar, 是在目標關係 r₁ 下關係 r 對節點 i 的重要性，r 和 r, 分別代表鄰居關係 r 和目標關係 r₁ 的嵌入。
Finally, the initial embedding of the node i is defined as:
h = Wo[hoshel], (11)
where Wo is the corresponding transformation matrix.
最後，節點 i 的初始嵌入定義為：
h = Wo[hoshel], (11)
其中 Wo 是對應的轉換矩陣。
#### 4.3.3 Subgraph embedding
To address the issue of sparse subgraphs, we merge the extracted local subgraph hg(u,r,v) with the neighboring relational paths PG(u,r,v) to form a comprehensive representation of the subgraph's topology information sG(u,r,,v).
Firstly, we input the target triplet (u, r₁, v) of the subgraph G(u, r₁, v) into the GNN to encode the extracted local subgraph. Specifically, we account for the interaction between nodes and relations, and express the node aggregation function at GNN's kth layer as:
= ΣΣαφ(rk-1,hk-1), (12)
rER JEN, (i)
ak_arj=61(W*sir + b), (13)
r,
Sir =62(W2[h-1h-1rk-1rk-1] + b), (14)
where R is the set of relations in the KG, a denotes the attention weights of edges connecting nodes i and j via relation r at layer k, W is the transformation matrix of relation r for propagating messages, pr-1, h−¹) is the fusion operation to share the hidden features of nodes and relations, and σ₁ and 62 are the activation functions. Simultaneously, upon updating the node embeddings as defined in Eq. (12), the relation embedding also undergoes a transformation to project both nodes and relations into a shared embedding space, denoted as:
rk = Wkerk-1. (15)
Given K as the number of GNN layers, the representation of the entire extracted local subgraph hg(u,r,,v) is obtained by averaging the node representations as follows:
hG(u,r,v) = 1/|VG(u,v)| Σ(ieguv) h_K^i (16)
where VG(u,r,,v) represents the set of nodes in the subgraph.
Secondly, we extract relational sequences among target nodes using an attention mechanism and a Gated Recurrent Unit (GRU) [35] to model the representations of neighboring relational paths PG(u,r,v), as expressed below:
PG(u,r,v) = Σap p (17)
PEPry (u,v)
a =softmax(p, r₁) = exp(pTr₁) / Σp'∈P(u,v) exp(p'Tr₁) (18)
p =GRU(p) = GRU(r,r,r), (19)
where P, (u, v) denotes the set of all neighboring relational paths between target nodes u and v under the target relation r₁. The final representation of the subgraph's topological information SG(u,r,v) is as follows:
SG(u,r,v) = hG(u,r,v) PG(u,r,v)· (20)
Importantly, we integrate complete neighboring relations into the subgraph from two aspects: the neighboring relational features of node features and the neighboring relational paths of the sparse subgraph. Finally, the topological likelihood score q¹po(u, r₁, v) of the target triplet (u, r₁, v) is given by:
ppo (u, r₁, v) = W[hhrp_tpo SG(u,r,v)], (21)
where rp is the learned embedding of relation r, from the topological perspective.
#### 4.3.3 子圖嵌入
為了處理稀疏子圖問題，我們將提取的局部子圖 hg(u,r,v) 與鄰近的關係路徑 PG(u,r,v) 合併，以形成子圖拓撲資訊 sG(u,r,,v) 的全面表示。
首先，我們將子圖 G(u, r₁, v) 的目標三元組 (u, r₁, v) 輸入 GNN，以編碼提取的局部子圖。具体來說，我们考慮了節點和關係之間的交互作用，并将 GNN 第 k 層的節點聚合函數表示为：
= ΣΣαφ(rk-1,hk-1), (12)
rER JEN, (i)
ak_arj=61(W*sir + b), (13)
r,
Sir =62(W2[h-1h-1rk-1rk-1] + b), (14)
其中 R 是知识图谱中的關係集合，a 表示在第 k 層透過關係 r 連接節點 i 和 j 的邊的注意力權重，W 是關係 r 用於傳播訊息的轉換矩陣，pr-1, h−¹) 是共享節點和關係隱藏特徵的融合操作，σ₁ 和 62 是激活函數。同時，在根據方程式 (12) 更新節點嵌入時，關係嵌入也進行轉換，將節點和關係都投影到共享的嵌入空間中，表示為：
rk = Wkerk-1. (15)
給定 K 為 GNN 層數，整個提取的局部子圖 hg(u,r,,v) 的表示可透過對節點表示取平均值獲得，如下所示：
hG(u,r,v) = 1/|VG(u,v)| Σ(ieguv) h_K^i (16)
其中 VG(u,r,,v) 表示子圖中的節點集合。
其次，我們使用注意力機制和門控循環單元 (GRU) [35] 提取目標節點之間的關係序列，以建模鄰近關係路徑 PG(u,r,v) 的表示，如下所示：
PG(u,r,v) = Σap p (17)
PEPry (u,v)
a =softmax(p, r₁) = exp(pTr₁) / Σp'∈P(u,v) exp(p'Tr₁) (18)
p =GRU(p) = GRU(r,r,r), (19)
其中 P, (u, v) 表示在目標關係 r₁ 下，目標節點 u 和 v 之間所有鄰近關係路徑的集合。子圖拓撲資訊 SG(u,r,v) 的最終表示如下：
SG(u,r,v) = hG(u,r,v) PG(u,r,v)· (20)
重要的是，我們從兩個方面將完整的鄰近關係整合到子圖中：節點特徵的鄰近關係特徵和稀疏子圖的鄰近關係路徑。最後，目標三元組 (u, r₁, v) 的拓撲可能性分數 q¹po(u, r₁, v) 由以下公式給出：
ppo (u, r₁, v) = W[hhrp_tpo SG(u,r,v)], (21)
其中 rp 是從拓撲角度学习到的关系 r 的嵌入。
Algorithm 2 Training process of GSELI
Require: Knowledge graph G and target triple (u, rt, v).
Ensure: Prediction score for (u,rt, v).
1: Extract relation-specific semantic features su and su of entities u and v (Eqs. (1), (2), (3));
2: Compute semantic likelihood score osem (u, rt, v) (Eq. (4));
3: Optimize su and su via semantic-aware contrastive loss Lcon (Eqs. (5), (6), (7));
4: Extract PPR-based local clustering subgraph G(u, rt, v) (Algorithm. (1));
5: Initialize node features hi with local topological locational features hos and neighboring relational features hel features (Eqs. (8), (9), (10), (11));
6: Get embedding of local subgraph hg(u,rt,v) using GNN (Eqs. (12), (13), (14), (15), (16));
7: Get embedding of neighboring relational paths PG(u,rt,v) (Eqs. (17), (18), (19));
8: Combine hg(u,rt,v) and PG(u,rt,v) to complete subgraph topology embedding SG(u,rt,v) (Eq. (20));
9: Compute topological likelihood score otro (u, rt, v) (Eq. (21));
10: Compute final scores (u, rt, v) combining osem (u, rt, v) and otpo(u, rt,v) (Eq. (23));
11: Calculate supervised loss Lsup (Eqs. (22), (24));
12: Minimize final loss C by combining Lsup and Leon (Eq. (25)).
演算法 2 GSELI 的訓練過程
需要：知識圖譜 G 和目標三元組 (u, rt, v)。
確保：(u,rt, v) 的預測分數。
1：提取實體 u 和 v 的特定關係語義特徵 su 和 su（方程式（1）、（2）、（3））；
2：計算語義可能性分數 osem (u, rt, v)（方程式（4））；
3：透過語義感知對比損失 Lcon 優化 su 和 su（方程式（5）、（6）、（7））；
4：提取基於 PPR 的局部聚類子圖 G(u, rt, v)（演算法（1））；
5：使用局部拓撲位置特徵 hos 和鄰近關係特徵 hel 初始化節點特徵 hi（方程式（8）、（9）、（10）、（11））；
6：使用 GNN 獲取局部子圖 hg(u,rt,v) 的嵌入（方程式（12）、（13）、（14）、（15）、（16））；
7：獲取鄰近關係路徑 PG(u,rt,v) 的嵌入（方程式（17）、（18）、（19））；
8：結合 hg(u,rt,v) 和 PG(u,rt,v) 以完成子圖拓撲嵌入 SG(u,rt,v)（方程式（20））；
9：計算拓撲可能性分數 otro (u, rt, v)（方程式（21））；
10：結合 osem (u, rt, v) 和 otpo(u, rt,v) 計算最終分數 (u, rt, v)（方程式（23））；
11：計算監督損失 Lsup（方程式（22）、（24））；
12：透過結合 Lsup 和 Leon 最小化最終損失 C（方程式（25））。
#### 4.4 Joint training strategy
Our work aims to achieve the ultimate goal through a combination of supervised loss Lsup and contrastive loss Lcon.
Firstly, for supervised learning, we treat all triplets in the original KG as positive triplets Tpos and construct negative triplets Tneg by randomly replacing head or tail entities with another entity in E. Formally, the set of negative triplets Tneg is defined as:
Tneg = {(u', rk, v) | u' ∈ E} ∪ {(u, rk, v') | ν' ∈ E}. (22)
Secondly, the total score for the supervised learning p(u, r₁, v) of the target triplet (u, r₁, v) is calculated by combining the global semantic information from Eq. (4) and the local topological information from Eq. (21). Formally, it is defined as:
φ(u, r₁, v) = psem (u, r₁, v) + φ¹po (u, r₁, v). (23)
Its margin-based loss function is defined as follows:
Lsup(u, r₁, v) = [φ(u', r₁, v') – φ(u, r₁, v) + γ]++ (24)
The final loss is calculated by combining the supervised learning loss from Eq. (24) and the contrastive learning loss from Eq. (7), and it is minimized through joint training, expressed as:
L = ΣΣ £sup + λ Σ Lcon (25)
Tpos Tneg Tpos
where λ represents a hyper-parameter adjusting the proportion between the supervised learning and contrastive learning losses.
#### 4.4 聯合訓練策略
我們的工作旨在透過結合監督式損失 Lsup 和對比式損失 Lcon 來實現最終目標。
首先，对于监督学习，我们将原始知识图谱中的所有三元组视为正三元组 Tpos，并通过随机替换头实体或尾实体为 E 中的另一个实体来构建负三元组 Tneg。形式上，负三元组集合 Tneg 定义为：
Tneg = {(u', rk, v) | u' ∈ E} ∪ {(u, rk, v') | ν' ∈ E}. (22)
其次，監督式學習目標三元組 (u, r₁, v) 的總分 p(u, r₁, v) 是透過結合方程式 (4) 的全域語意資訊和方程式 (21) 的區域拓撲資訊計算而得。形式上，其定義如下：
φ(u, r₁, v) = psem (u, r₁, v) + φ¹po (u, r₁, v). (23)
其基於邊界的損失函數定義如下：
Lsup(u, r₁, v) = [φ(u', r₁, v') – φ(u, r₁, v) + γ]++ (24)
最終損失是結合式（24）的監督學習損失和式（7）的對比學習損失計算得出，並透過聯合訓練使其最小化，表示為：
L = ΣΣ £sup + λ Σ Lcon (25)
Tpos Tneg Tpos
其中 λ 代表一个超参数，用于调整监督学习和对比学习损失之间的比例。
Table 2 Statistics of the inductive datasets proposed by DEKG: |R|, |E|, and|T| represent the number of relations, entities, and triplets, respectively

| | Train | test mix | test_bridging | test_fully |
| :--- | :--- | :--- | :--- | :--- |
| | |R| |E| |T| |R| |E| |T| |R| |E| |T| |R| |E| |T| |
| FB15K-237 |
| EQ | 180 | 1594 | 4734 | 180 | 2687 | 6648 | 180 | 2687 | 6443 | 142 | 1093 | 2198 |
| MB | 200 | 2608 | 10,905 | 200 | 4268 | 15,318 | 200 | 4268 | 14,840 | 172 | 1660 | 4623 |
| MF | 215 | 3668 | 20,180 | 215 | 6169 | 26,689 | 215 | 6169 | 25,824 | 183 | 2501 | 8271 |
| NELL-995 |
| EQ | 14 | 3088 | 5101 | 14 | 3312 | 5718 | 14 | 3312 | 5618 | 14 | 225 | 933 |
| MB | 88 | 2544 | 9141 | 88 | 4551 | 14,201 | 88 | 4551 | 13,727 | 79 | 2074 | 5062 |
| MF | 142 | 4539 | 18,240 | 142 | 7914 | 25,625 | 142 | 7914 | 24,820 | 122 | 3514 | 8857 |
| WN18RR |
| EQ | 9 | 2746 | 6040 | 9 | 3668 | 7404 | 9 | 3668 | 7216 | 8 | 922 | 1806 |
| MB | 10 | 6954 | 17,100 | 10 | 9711 | 20,596 | 10 | 9711 | 20,155 | 10 | 2757 | 4452 |
| MF | 11 | 12,078 | 28,998 | 11 | 17,154 | 33,130 | 11 | 17,154 | 32,525 | 11 | 5084 | 6932 |
表 2 DEKG 提出的歸納數據集統計：|R|、|E| 和 |T| 分別代表關係、實體和三元組的數量

| | 訓練 | 測試混合 | 測試橋接 | 測試全量 |
| :--- | :--- | :--- | :--- | :--- |
| | |R| |E| |T| |R| |E| |T| |R| |E| |T| |R| |E| |T| |
| FB15K-237 |
| 等式 | 180 | 1594 | 4734 | 180 | 2687 | 6648 | 180 | 2687 | 6443 | 142 | 1093 | 2198 |
| MB | 200 | 2608 | 10,905 | 200 | 4268 | 15,318 | 200 | 4268 | 14,840 | 172 | 1660 | 4623 |
| MF | 215 | 3668 | 20,180 | 215 | 6169 | 26,689 | 215 | 6169 | 25,824 | 183 | 2501 | 8271 |
| NELL-995 |
| 等式 | 14 | 3088 | 5101 | 14 | 3312 | 5718 | 14 | 3312 | 5618 | 14 | 225 | 933 |
| MB | 88 | 2544 | 9141 | 88 | 4551 | 14,201 | 88 | 4551 | 13,727 | 79 | 2074 | 5062 |
| MF | 142 | 4539 | 18,240 | 142 | 7914 | 25,625 | 142 | 7914 | 24,820 | 122 | 3514 | 8857 |
| WN18RR |
| 等式 | 9 | 2746 | 6040 | 9 | 3668 | 7404 | 9 | 3668 | 7216 | 8 | 922 | 1806 |
| MB | 10 | 6954 | 17,100 | 10 | 9711 | 20,596 | 10 | 9711 | 20,155 | 10 | 2757 | 4452 |
| MF | 11 | 12,078 | 28,998 | 11 | 17,154 | 33,130 | 11 | 17,154 | 32,525 | 11 | 5084 | 6932 |
## 5 Experiments
This section, "Experiments,”includes the following parts: dataset (see Sect. 5.1), experimental details (see Sect. 5.2), results on DEKG datasets (see Sect. 5.3), results on GraIL datasets (see Sect. 5.4), and ablation study (see Sect. 5.5).

### 5.1 Dataset
We adhere to the benchmark datasets proposed in DEKG-ILP for fully inductive and bridging inductive link prediction. DEKG-ILP extends the FB15K237 [36], NELL-995 [37], and WN18RR [38] datasets (v1, v2, and v3) introduced in Grail by creating EQ (equal inductive links), MB (more bridging-inductive links), and MF (more fully-inductive links) versions. For example, we use the FB15K237 v1, v2, and v3 training sets to construct the EQ, MB, and MF training sets and the FB15K237 v1, v2, and v3 test sets to construct the EQ, MB, and MF fully-inductive test sets. The bridging-inductive evaluation datasets are created using the training sets and fully-inductive test sets, where one entity is in the training set, and the other is in the fully-inductive test set. We also construct the EQ, MB, and MF mixed evaluation datasets using the fully-inductive and bridging-inductive evaluation datasets in ratios of 1:1, 1:2, and 2:1. The relevant statistical data are listed in Table 2. Additionally, we use the v1, v2, v3, and v4 datasets of FB15K237 and WN18RR from GraIL to comprehensively evaluate the fully-inductive LP performance. The statistical data are presented in Table 3.
## 5 實驗
本節「實驗」包括以下部分：資料集（參見 5.1 節）、實驗細節（參見 5.2 節）、DEKG 資料集上的結果（參見 5.3 節）、GraIL 資料集上的結果（參見 5.4 節）和消融研究（參見 5.5 節）。

### 5.1 資料集
我們遵循 DEKG-ILP 中提出的基準數據集，用於全歸納式和橋接歸納式鏈接預測。DEKG-ILP 擴展了 Grail 中引入的 FB15K237 [36]、NELL-995 [37] 和 WN18RR [38] 數據集（v1、v2 和 v3），創建了 EQ（等歸納式鏈接）、MB（更多橋接歸納式鏈接）和 MF（更多全歸納式鏈接）版本。例如，我们使用 FB15K237 v1、v2 和 v3 訓練集來構建 EQ、MB 和 MF 訓練集，並使用 FB15K237 v1、v2 和 v3 測試集來構建 EQ、MB 和 MF 全歸納式測試集。橋接歸納式評估數據集是使用訓練集和全歸納式測試集創建的，其中一個實體在訓練集中，另一個在全归纳式測試集中。我們還使用全归纳式和桥接归纳式評估數據集，以 1:1、1:2 和 2:1 的比例構建 EQ、MB 和 MF 混合評估數據集。相關統計數據列於表 2。此外，我们使用 GraIL 中的 FB15K237 和 WN18RR 的 v1、v2、v3 和 v4 數據集，以全面評估全归纳式 LP 的性能。統計數據呈現在表 3 中。
Table 3 Statistics of the fully-inductive datasets proposed by GraIL: |R|, |E|, and [7] represent the number of relations, entities, and triplets, respectively

| | FB15K-237 | WN18RR |
| :--- | :--- | :--- |
| | |R| |E| |[7]| |R| |E| |[7]|
| v1 | Train | 180 | 1594 | 5226 | 9 | 2764 | 6678 |
| | Test | 142 | 1093 | 2404 | 8 | 922 | 1991 |
| v2 | Train | 200 | 2608 | 12,085 | 10 | 6954 | 18,968 |
| | Test | 172 | 1660 | 5092 | 10 | 2757 | 4863 |
| v3 | Train | 215 | 3668 | 22,394 | 11 | 12,078 | 32,150 |
| | Test | 183 | 2501 | 9137 | 11 | 5084 | 7470 |
| v4 | Train | 219 | 4707 | 33,916 | 9 | 3861 | 9842 |
| | Test | 200 | 3501 | 14,554 | 9 | 7084 | 15,157 |
表 3 GraIL 提出的全歸納數據集統計：|R|、|E| 和 [7] 分別代表關係、實體和三元組的數量

| | FB15K-237 | WN18RR |
| :--- | :--- | :--- |
| | |R| |E| |[7]| |R| |E| |[7]|
| v1 | 火車 | 180 | 1594 | 5226 | 9 | 2764 | 6678 |
| | 測試 | 142 | 1093 | 2404 | 8 | 922 | 1991 |
| v2 | 火車 | 200 | 2608 | 12,085 | 10 | 6954 | 18,968 |
| | 測試 | 172 | 1660 | 5092 | 10 | 2757 | 4863 |
| v3 | 火車 | 215 | 3668 | 22,394 | 11 | 12,078 | 32,150 |
| | 測試 | 183 | 2501 | 9137 | 11 | 5084 | 7470 |
| v4 | 火車 | 219 | 4707 | 33,916 | 9 | 3861 | 9842 |
| | 測試 | 200 | 3501 | 14,554 | 9 | 7084 | 15,157 |
### 5.2 Experimental details

#### 5.2.1 Evaluation protocol
We follow the evaluation protocol used in DEKG-ILP [18], which includes mean reciprocal rank (MRR) and Hits@N metrics (where N=1, 5, 10). We conducted evaluations for both relation prediction (h, ?, t) and entity prediction (?, r, t) and (h, r, ?) tasks. MRR represents the average reciprocal rank of all test triplets, while Hits@N measures the proportion of correctly ranked entities and relations within the top N. Negative triplets are obtained by replacing elements in the triplets with candidate sets of entities and relations that include all entities and relations in the KG. It is worth noting that higher MRR and Hits@N values indicate better link prediction performance. We conducted five experiments using different random seeds and recorded the average results.

#### 5.2.2 Hyper-parameter setting
Our experiment is implemented using PyTorch with the Adam optimizer [39]. For subgraph extraction, we use a 3-hop subgraph. The hyperparameters are manually specified during the training process: a learning rate of 0.01, an edge dropout rate of 0.5, a relation-specific embedding dimension of 32, and a margin y of 10 in the loss function for Eqs. (7) and (24). Additionally, we select hyperparameters θ and λ, which correspond to adjusting the scaling factor of the number of relations in the relational component table in Sect. 4.2.2 and adjusting the loss coefficient of the weight for contrastive learning in Eq. (25), respectively. We choose values for θ and λ from the range [0.1, 0.9], and the rationale for this selection will be explained in Sect. 6.2.
### 5.2 實驗細節

#### 5.2.1 評估協議
我們遵循 DEKG-ILP [18] 中使用的評估協議，其中包括平均倒數排名 (MRR) 和 Hits@N 指標（其中 N=1, 5, 10）。我們對關係預測 (h, ?, t) 和實體預測 (?, r, t) 和 (h, r, ?) 任務都進行了評估。MRR 表示所有測試三元組的平均倒數排名，而 Hits@N 衡量在前 N 個中正確排序的實體和關係的比例。負三元組是通过用包含知識圖譜中所有实体和关系的候选实体和关系集替换三元组中的元素来获得的。值得注意的是，较高的 MRR 和 Hits@N 值表示更好的链接预测性能。我们使用不同的随机种子进行了五次实验，并记录了平均结果。

#### 5.2.2 超參數設定
我們的實驗使用 PyTorch 和 Adam 優化器 [39] 實現。對於子圖提取，我們使用 3-hop 子圖。超參數在訓練過程中手動指定：學習率為 0.01，邊緣丟失率為 0.5，關係特定嵌入維度為 32，以及損失函數中 Eqs. (7) 和 (24) 的邊界 y 為 10。此外，我们选择超参数 θ 和 λ，它们分别对应于调整 4.2.2 节中关系组件表中关系数量的缩放因子和调整 Eq. (25) 中对比学习权重的损失系数。我们从范围 [0.1, 0.9] 中选择 θ 和 λ 的值，选择的理由将在 6.2 节中解释。
Table 4 The results of MRR and Hits@N (N = 1, 5, 10) on the EQ, MB, and ME versions of the FB15K-237, NELL-995, and WN18RR datasets

| Model | EQ | MB | MF | Avg |
| :--- | :--- | :--- | :--- | :--- |
| | MRR | H@1 | H@5 | H@10 | MRR | H@1 | H@5 | H@10 | MRR | H@1 | H@5 | H@10 | |
| **MRR and Hits@N (N = 1, 5, 10) results on the FB15K-237 datasets** |
| Grail | 0.282 | 0.223 | 0.319 | 0.339 | 0.270 | 0.213 | 0.291 | 0.301 | 0.509 | 0.432 | 0.586 | 0.616 | 0.365 |
| TATC | 0.318 | 0.263 | 0.345 | 0.354 | 0.276 | 0.227 | 0.296 | 0.305 | 0.484 | 0.411 | 0.595 | 0.363 |
| COMPILE | 0.316 | 0.256 | 0.353 | 0.375 | 0.282 | 0.230 | 0.301 | 0.314 | 0.522 | 0.452 | 0.589 | 0.608 | 0.383 |
| SNRI | 0.362 | 0.281 | 0.419 | 0.496 | 0.313 | 0.224 | 0.375 | 0.478 | 0.415 | 0.303 | 0.528 | 0.620 | 0.401 |
| RMPI | 0.327 | 0.253 | 0.378 | 0.441 | 0.288 | 0.226 | 0.320 | 0.359 | 0.517 | 0.441 | 0.579 | 0.640 | 0.397 |
| DEKG-ILP | 0.421 | 0.285 | 0.572 | 0.699 | 0.542 | 0.409 | 0.693 | 0.825 | 0.632 | 0.511 | 0.777 | 0.889 | 0.605 |
| GSELI(Ours) | **0.502** | **0.350** | **0.682** | **0.834** | **0.616** | **0.473** | **0.790** | **0.903** | **0.666** | **0.542** | **0.814** | **0.934** | **0.676** |
| **MRR and Hits@N (N = 1, 5, 10) results on the NELL-995 datasets** |
| Grail | 0.463 | 0.400 | 0.520 | 0.520 | 0.306 | 0.220 | 0.381 | 0.429 | 0.595 | 0.516 | 0.677 | 0.700 | 0.477 |
| TATC | 0.503 | 0.435 | 0.552 | 0.560 | 0.348 | 0.227 | 0.401 | 0.440 | 0.593 | 0.504 | 0.690 | 0.718 | 0.498 |
| COMPILE | 0.475 | 0.422 | 0.517 | 0.517 | 0.341 | 0.283 | 0.364 | 0.382 | 0.629 | 0.558 | 0.698 | 0.718 | 0.492 |
| SNRI | 0.505 | 0.402 | 0.582 | 0.702 | 0.309 | 0.218 | 0.371 | 0.458 | 0.551 | 0.456 | 0.655 | 0.718 | 0.435 |
| RMPI | 0.486 | 0.411 | 0.547 | 0.607 | 0.347 | 0.278 | 0.389 | 0.468 | 0.656 | 0.579 | 0.734 | 0.785 | 0.524 |
| DEKG-ILP | 0.449 | 0.320 | 0.576 | 0.740 | 0.387 | 0.226 | 0.593 | 0.727 | 0.519 | 0.397 | 0.662 | 0.737 | 0.528 |
| GSELI(Ours) | 0.468 | 0.333 | 0.615 | 0.763 | 0.465 | 0.327 | 0.637 | 0.746 | 0.454 | 0.295 | 0.651 | 0.784 | 0.545 |
| **MRR and Hits@N (N = 1, 5, 10) results on the WN18RR datasets** |
| Grail | 0.423 | 0.392 | 0.420 | 0.420 | 0.293 | 0.267 | 0.272 | 0.272 | 0.412 | 0.373 | 0.430 | 0.431 | 0.367 |
| TATC | 0.423 | 0.401 | 0.420 | 0.420 | 0.285 | 0.269 | 0.272 | 0.272 | 0.418 | 0.388 | 0.429 | 0.430 | 0.369 |
| COMPILE | 0.421 | 0.400 | 0.417 | 0.417 | 0.274 | 0.255 | 0.260 | 0.260 | 0.399 | 0.367 | 0.400 | 0.400 | 0.356 |
| SNRI | 0.462 | 0.405 | 0.489 | 0.545 | 0.356 | 0.294 | 0.379 | 0.455 | 0.444 | 0.383 | 0.474 | 0.505 | 0.441 |
| RMPI | 0.474 | 0.422 | 0.494 | 0.539 | 0.354 | 0.312 | 0.379 | 0.437 | 0.468 | 0.429 | 0.477 | 0.505 | 0.457 |
| DEKG-ILP | 0.405 | 0.289 | 0.544 | 0.690 | 0.351 | 0.224 | 0.493 | 0.627 | 0.381 | 0.258 | 0.503 | 0.715 | 0.457 |
| GSELI(Ours) | 0.418 | 0.293 | 0.563 | 0.718 | 0.356 | 0.228 | 0.493 | 0.642 | 0.368 | 0.243 | 0.503 | 0.703 | 0.461 |
表 4 FB15K-237、NELL-995 和 WN18RR 資料集之 EQ、MB 和 ME 版本的 MRR 和 Hits@N (N = 1, 5, 10) 結果

| 模型 | EQ | MB | MF | 平均 |
| :--- | :--- | :--- | :--- | :--- |
| | MRR | H@1 | H@5 | H@10 | MRR | H@1 | H@5 | H@10 | MRR | H@1 | H@5 | H@10 | |
| **FB15K-237 資料集的 MRR 和 Hits@N (N = 1, 5, 10) 結果** |
| 聖杯 | 0.282 | 0.223 | 0.319 | 0.339 | 0.270 | 0.213 | 0.291 | 0.301 | 0.509 | 0.432 | 0.586 | 0.616 | 0.365 |
| TATC | 0.318 | 0.263 | 0.345 | 0.354 | 0.276 | 0.227 | 0.296 | 0.305 | 0.484 | 0.411 | 0.595 | 0.363 |
| 編譯 | 0.316 | 0.256 | 0.353 | 0.375 | 0.282 | 0.230 | 0.301 | 0.314 | 0.522 | 0.452 | 0.589 | 0.608 | 0.383 |
| SNRI | 0.362 | 0.281 | 0.419 | 0.496 | 0.313 | 0.224 | 0.375 | 0.478 | 0.415 | 0.303 | 0.528 | 0.620 | 0.401 |
| RMPI | 0.327 | 0.253 | 0.378 | 0.441 | 0.288 | 0.226 | 0.320 | 0.359 | 0.517 | 0.441 | 0.579 | 0.640 | 0.397 |
| DEKG-ILP | 0.421 | 0.285 | 0.572 | 0.699 | 0.542 | 0.409 | 0.693 | 0.825 | 0.632 | 0.511 | 0.777 | 0.889 | 0.605 |
| GSELI（我們的） | **0.502** | **0.350** | **0.682** | **0.834** | **0.616** | **0.473** | **0.790** | **0.903** | **0.666** | **0.542** | **0.814** | **0.934** | **0.676** |
| **NELL-995 資料集的 MRR 和 Hits@N (N = 1, 5, 10) 結果** |
| 聖杯 | 0.463 | 0.400 | 0.520 | 0.520 | 0.306 | 0.220 | 0.381 | 0.429 | 0.595 | 0.516 | 0.677 | 0.700 | 0.477 |
| TATC | 0.503 | 0.435 | 0.552 | 0.560 | 0.348 | 0.227 | 0.401 | 0.440 | 0.593 | 0.504 | 0.690 | 0.718 | 0.498 |
| 編譯 | 0.475 | 0.422 | 0.517 | 0.517 | 0.341 | 0.283 | 0.364 | 0.382 | 0.629 | 0.558 | 0.698 | 0.718 | 0.492 |
| SNRI | 0.505 | 0.402 | 0.582 | 0.702 | 0.309 | 0.218 | 0.371 | 0.458 | 0.551 | 0.456 | 0.655 | 0.718 | 0.435 |
| RMPI | 0.486 | 0.411 | 0.547 | 0.607 | 0.347 | 0.278 | 0.389 | 0.468 | 0.656 | 0.579 | 0.734 | 0.785 | 0.524 |
| DEKG-ILP | 0.449 | 0.320 | 0.576 | 0.740 | 0.387 | 0.226 | 0.593 | 0.727 | 0.519 | 0.397 | 0.662 | 0.737 | 0.528 |
| GSELI（我們的） | 0.468 | 0.333 | 0.615 | 0.763 | 0.465 | 0.327 | 0.637 | 0.746 | 0.454 | 0.295 | 0.651 | 0.784 | 0.545 |
| **WN18RR 資料集的 MRR 和 Hits@N (N = 1, 5, 10) 結果** |
| 聖杯 | 0.423 | 0.392 | 0.420 | 0.420 | 0.293 | 0.267 | 0.272 | 0.272 | 0.412 | 0.373 | 0.430 | 0.431 | 0.367 |
| TATC | 0.423 | 0.401 | 0.420 | 0.420 | 0.285 | 0.269 | 0.272 | 0.272 | 0.418 | 0.388 | 0.429 | 0.430 | 0.369 |
| 編譯 | 0.421 | 0.400 | 0.417 | 0.417 | 0.274 | 0.255 | 0.260 | 0.260 | 0.399 | 0.367 | 0.400 | 0.400 | 0.356 |
| SNRI | 0.462 | 0.405 | 0.489 | 0.545 | 0.356 | 0.294 | 0.379 | 0.455 | 0.444 | 0.383 | 0.474 | 0.505 | 0.441 |
| RMPI | 0.474 | 0.422 | 0.494 | 0.539 | 0.354 | 0.312 | 0.379 | 0.437 | 0.468 | 0.429 | 0.477 | 0.505 | 0.457 |
| DEKG-ILP | 0.405 | 0.289 | 0.544 | 0.690 | 0.351 | 0.224 | 0.493 | 0.627 | 0.381 | 0.258 | 0.503 | 0.715 | 0.457 |
| GSELI（我們的） | 0.418 | 0.293 | 0.563 | 0.718 | 0.356 | 0.228 | 0.493 | 0.642 | 0.368 | 0.243 | 0.503 | 0.703 | 0.461 |
The EQ, MB, and ME test sets include both fully inductive and bridging inductive links, distributed in ratios of 1:1, 1:2, and 2:1, respectively.
The best results are in bold, and the second-best are underlined
EQ、MB 和 ME 測試集包含全歸納式和橋接歸納式連結，分佈比例分別為 1:1、1:2 和 2:1。
最佳結果以粗體顯示，次佳結果則加底線。
#### 5.2.3 Baselines
We evaluated our model using the baseline models: GraIL, TACT, COMPILE, SNRI, RMPI, and DEKG-ILP. These models use subgraph methods for ILP and support fully-inductive LP. Additionally, DEKG-ILP is designed specifically for bridging-inductive LP. We used publicly available source code and implemented these models with optimal parameters.
Tailored for fully-inductive LP are as follows:
- GraIL [11] has become the first model used for fully-inductive links by employing local subgraph sampling, subgraph node labeling, and GNN scoring.
- TACT [12] extends GraIL by categorizing all pairs of relations into seven topological patterns, aiming to leverage the topology-aware correlations between relations.
- COMPILE [13] extends GraIL by introducing a new node-edge communication mechanism to enhance relational information.
- SNRI [14] extends GraIL by integrating complete neighboring relations into enclosing subgraphs, maximizing local-global MI between subgraph-graph to globally model neighboring relations.
- RMPI [16] extends GraIL through effective relational message passing and relational patterns to achieve subgraph reasoning.
Tailored for bridging-inductive LP are as follows:
- DEKG-ILP [18] is the first model to leverage subgraphs and global relational semantic features shared between the original KG and emerging KG for bridging-inductive LP.
#### 5.2.3 基線
我們使用基準模型 GraIL、TACT、COMPILE、SNRI、RMPI 和 DEKG-ILP 來評估我們的模型。這些模型使用子圖方法進行 ILP 並支援全歸納式 LP。此外，DEKG-ILP 專為橋接歸納式 LP 設計。我们使用了公开可用的源代码，并以最佳参数实现了这些模型。
专为全归纳式 LP 设计的模型如下：
- GraIL [11] 已成為第一個用於全歸納式連結的模型，它採用局部子圖取樣、子圖節點標記和 GNN 評分。
- TACT [12] 擴展了 GraIL，將所有關係對分為七種拓撲模式，旨在利用關係之間的拓撲感知相關性。
- COMPILE [13] 擴充了 GraIL，引入了一種新的節點-邊緣通信機制以增強關係信息。
- SNRI [14] 擴展了 GraIL，将完整的邻近关系整合到封闭子图中，最大化子图与图之间的局部-全局 MI，以全局建模邻近关系。
- RMPI [16] 透過有效的關係訊息傳遞和關係模式擴展了 GraIL，以實現子圖推理。
专为桥接归纳式 LP 定制的模型如下：
- DEKG-ILP [18] 是第一個利用原始知識圖譜與新興知識圖譜之間共享的子圖及全域關係語意特徵進行橋接歸纳 LP 的模型。
Fig. 3 Statistical significance analysis of MRR and Hits@N (N=1, 5, 10) on EQ, MB, and ME versions of FB15K-237, NELL-995, and WN18RR datasets
圖 3 FB15K-237、NELL-995 和 WN18RR 資料集之 EQ、MB 和 ME 版本的 MRR 和 Hits@N (N=1, 5, 10) 統計顯著性分析
### 5.3 Results on DEKG datasets

#### 5.3.1 Main results
Table 4 presents a comparison of the MRR and Hits@N (1, 5, 10) results of our proposed GSELI model with the baseline models on the FB15k-237, NELL-995, and WN18RR datasets using the EQ, MB, and MF versions. The test datasets include fully-inductive and bridging-inductive links in ratios of 1:1, 1:2, and 2:1. The results indicate that the GSELI model significantly outperforms the baseline models on most datasets. From Table 4, it can be observed that:
(1) GSELI shows more significant improvements on the FB15k-237 and NELL-995 datasets compared to WN18RR, particularly in the FB15k-237 dataset, where both MRR and Hits@N have increased. This improvement can be attributed to the larger number of relations in FB15k-237 and NELL-995, which allow GSELI to extract richer relation-specific semantic features and more comprehensive neighborhood relation information.
(2) Compared to the EQ and MF versions, GSELI and DEKG-ILP demonstrate more significant improvements in the MB version due to its inclusion of more bridging-inductive links. This improvement is attributed to the optimization of GSELI and DEKG-ILP for bridging-inductive links, while also maintaining strong performance in fully-inductive links. In contrast, other baseline models are specifically designed for fully-inductive links.
(3) GSELI shows limited ranking capabilities in the EQ and MF versions of the NELL-995 and WN18RR datasets. Its high-precision metrics (Hits@1 and Hits@5) are lower than those of fully-inductive LP models, although it surpasses other baseline models in the Hits@10 metric. This limitation may stem from the complexity and diversity of the EQ and MF datasets. These versions typically include more varied and intricate relational patterns, which GSELI fails to fully capture, leading to subpar high-precision performance. Additionally, GSELI is better suited for capturing broad relational patterns rather than the fine distinctions needed for high-precision ranking.
### 5.3 DEKG 資料集上的結果

#### 5.3.1 主要成果
表 4 比較了我們提出的 GSELI 模型與基線模型在 FB15k-237、NELL-995 和 WN18RR 數據集上使用 EQ、MB 和 MF 版本的 MRR 和 Hits@N (1, 5, 10) 結果。測試數據集包含比例為 1:1、1:2 和 2:1 的全歸納和橋接歸納鏈路。結果表明，GSELI 模型在大多數數據集上顯著優於基線模型。從表 4 可以觀察到：
(1) 與 WN18RR 相比，GSELI 在 FB15k-237 和 NELL-995 數據集上表現出更顯著的改進，尤其是在 FB15k-237 數據集中，MRR 和 Hits@N 均有增加。這種改進可歸因於 FB15k-237 和 NELL-995 中關係數量較多，這使得 GSELI 能夠提取更豐富的特定於關係的語義特徵和更全面的鄰域關係信息。
(2) 相較於 EQ 和 MF 版本，GSELI 和 DEKG-ILP 在 MB 版本中表現出更顯著的提升，因其包含更多橋接歸納連結。此提升歸因於 GSELI 和 DEKG-ILP 針對橋接歸納連結的優化，同時在全歸納連結中也維持強勁的效能。相較之下，其他基線模型則專為全歸納連結而設計。
(3) GSELI 在 NELL-995 和 WN18RR 資料集的 EQ 和 MF 版本中顯示出有限的排名能力。其高精度指標（Hits@1 和 Hits@5）低於全歸納式 LP 模型，儘管它在 Hits@10 指標上超越了其他基線模型。此限制可能源於 EQ 和 MF 資料集的複雜性和多樣性。這些版本通常包含更多樣化和複雜的關係模式，而 GSELI 未能完全捕捉，導致高精度表現不佳。此外，GSELI 更適合捕捉廣泛的關係模式，而非高精度排名所需的細微差別。
Fig. 4 Hits@10 results on the EQ, MB, and MF of the FB15k-237, NELL-995, and WN18RR datasets, including tests with solely fully-inductive links and solely bridging-inductive links
圖 4 FB15k-237、NELL-995 和 WN18RR 資料集的 EQ、MB 和 MF 上的 Hits@10 結果，包括僅含全歸納連結和僅含橋接歸納連結的測試
Figure 3 presents the statistical significance analysis of the EQ, MB, and MF versions of the FB15K-237, NELL-995, and WN18RR datasets on MRR and Hits@N (N = 1, 5, 10), comparing GSELI's performance against baseline models. A larger t-statistic (whether positive or negative) indicates a greater mean difference, while a smaller p-value suggests this difference is less likely due to random error. For the FB15K-237 dataset, all t-statistics are positive and significant, with p-values less than 0.005, indicating GSELI's superiority on all metrics. For the NELL-995 dataset, most EQ and MB metrics show significant differences, especially in MRR and Hits@1, with p values less than 0.001. However, the MF metrics perform worse on Hits@1 and Hits@5, with negative t-statistics, indicating GSELI's inferiority on these metrics. For the WN18RR dataset, the EQ and MB metrics are significant on Hits@1, Hits@5, and Hits@10, with p-values less than 0.01, but show no significant difference on MRR. Additionally, the MF metrics display significant negative differences, indicating that GSELI performs worse than the baseline models on some metrics in this dataset. Overall, GSELI generally outperforms the baseline models, though inconsistently.
圖 3 呈現了在 FB15K-237、NELL-995 和 WN18RR 資料集的 EQ、MB 和 MF 版本上，關於 MRR 和 Hits@N (N = 1, 5, 10) 的統計顯著性分析，比較了 GSELI 與基線模型的表現。較大的 t 統計量（無論正負）表示平均差異較大，而較小的 p 值則表示此差異較不可能由隨機誤差造成。對於 FB15K-237 資料集，所有 t 統計量皆為正值且顯著，p 值小於 0.005，顯示 GSELI 在所有指標上均具優越性。對於 NELL-995 資料集，多數 EQ 和 MB 指標呈現顯著差異，尤其在 MRR 和 Hits@1 上，p 值小於 0.001。然而，MF 指標在 Hits@1 和 Hits@5 上表現較差，t 統計量為負值，表示 GSELI 在這些指標上表現較差。對於 WN18RR 資料集，EQ 和 MB 指標在 Hits@1、Hits@5 和 Hits@10 上均顯著，p 值小於 0.01，但在 MRR 上無顯著差異。此外，MF 指標呈現顯著的負差異，表示 GSELI 在此資料集的某些指標上表現不如基線模型。總體而言，GSELI 整體表現優於基線模型，但並非始終如一。
#### 5.3.2 Respective results
This section examines the Hits@10 results for the EQ, MB, and MF versions of the FB15k-237, NELL-995, and WN18RR datasets, focusing on both fully-inductive links and bridging-inductive links. Due to space limitations, we have not included the MRR, Hits@1, and Hits@5 metrics. According to Fig. 4, we observe the following:
(1) Overall, GSELI, leveraging contrastive learning-based global semantic features and GNN-based enhanced local subgraphs, outperforms most baseline models in Hits@10 across various benchmark datasets for both fully-inductive and bridging-inductive links, especially excelling in bridging-inductive link prediction.
(2) In fully-inductive links, GSELI demonstrates excellent performance on the FB15k-237 and WN18RR datasets, thanks to its meticulous design. However, since GSELI is not specifically optimized for fully-inductive links, its performance on the NELL-995 dataset falls short compared to models explicitly designed for fully-inductive links, such as RMPI and GraIL.
(3) In bridging-inductive links, GraIL, TATC, and COMPILE perform poorly due to topological limitations. SNRI demonstrates the ability to handle bridging-inductive links through global modeling of neighboring relations, while RMPI uses relational message-passing networks. GSELI and DEKG-ILP, by leveraging global semantic features, outperform SNRI and RMPI. However, GSELI surpasses DEKG-ILP by utilizing enhanced local subgraphs.
#### 5.3.2 各自的結果
本節檢視了 FB15k-237、NELL-995 和 WN18RR 資料集之 EQ、MB 和 MF 版本的 Hits@10 結果，重點關注全歸納式連結和橋接歸納式連結。由於篇幅限制，我們未包含 MRR、Hits@1 和 Hits@5 指標。根據圖 4，我們觀察到以下幾點：
(1) 整體而言，GSELI 憑藉基於對比學習的全域語意特徵和基於 GNN 的增強型區域子圖，在全歸納式和橋接歸納式連結的各種基準資料集上，其 Hits@10 表現優於大多數基線模型，尤其在橋接歸納式連結預測方面表現出色。
(2) 在全歸納式連結中，GSELI 在 FB15k-237 和 WN18RR 資料集上表現出色，這得益於其精心的設計。然而，由於 GSELI 並未針對全歸納式連結進行特別優化，因此其在 NELL-995 資料集上的表現不及 RMPI 和 GraIL 等專為全歸納式連結設計的模型。
(3) 在橋接歸納連結中，GraIL、TATC 和 COMPILE 由於拓樸限制而表現不佳。SNRI 透過對鄰近關係进行全域建模，展现了处理桥接归纳连接的能力，而 RMPI 使用关系訊息傳遞網路。GSELI 和 DEKG-ILP 透過利用全域語意特徵，表現優於 SNRI 和 RMPI。然而，GSELI 透過利用增強的局部子圖，超越了 DEKG-ILP。
Fig. 5 Statistical significance analysis of Hits@10 on v1, v2, v3, and v4 versions of the FB15K-237, NELL-995, and WN18RR datasets
圖 5 FB15K-237、NELL-995 和 WN18RR 資料集 v1、v2、v3 和 v4 版本 Hits@10 的統計顯著性分析
### 5.4 Results on Grall datasets
To further evaluate fully-inductive link prediction, we used the FB15k-237 and WN18RR datasets in versions v1, v2, v3, and v4 proposed by GraIL and reported the Hits@10 results, as shown in Table 5.
Table 5 shows that our GSELI model performs exceptionally well across multiple versions of both datasets. On the FB15k-237 dataset, GSELI achieves the highest performance in versions v2 and v4 and the second-highest performance in versions v1 and v3. On the WN18RR dataset, GSELI achieves the highest performance across all versions, with its average performance surpassing other models. Notably, DEKG-ILP also performs well on these datasets, though it is slightly outperformed by GSELI. This indicates that although GSELI is specifically designed for bridging-inductive links, it also demonstrates excellent performance in fully-inductive link prediction tasks.
Figure 5 presents the statistical significance analysis of the v1, v2, v3, and v4 versions of the FB15K-237 and WN18RR datasets on Hits@10, comparing GSELI's performance against baseline models. For the FB15K-237 dataset, all versions have positive t-statistics and p values less than 0.015, indicating significant differences compared to baseline models, with version v4 performing best (t-statistic = 6.00, p value = 0.0018). For the WN18RR dataset, all versions also have positive t-statistics and p values less than 0.016, showing significant differences, with version v1 performing best (t-statistic = 5.46, p value = 0.0028). Overall, GSELI significantly outperforms the baseline models in all versions across both datasets.
### 5.4 Grall 資料集上的結果
為了進一步評估全歸納式連結預測，我們使用了 GraIL 提出的 FB15k-237 和 WN18RR 資料集的 v1、v2、v3 和 v4 版本，並報告了 Hits@10 的結果，如表 5 所示。
表 5 顯示，我們的 GSELI 模型在兩個數據集的多個版本中都表現出色。在 FB15k-237 數據集上，GSELI 在 v2 和 v4 版本中取得了最高性能，在 v1 和 v3 版本中取得了次高性能。在 WN18RR 數據集上，GSELI 在所有版本中都取得了最高性能，其平均性能超过了其他模型。值得注意的是，DEKG-ILP 在這些數據集上表現也不錯，儘管略遜于 GSELI。這表明，儘管 GSELI 是專為橋接歸納鏈接設計的，但它在全归纳鏈接預測任務中也表現出色。
圖 5 展示了在 FB15K-237 和 WN18RR 資料集的 v1、v2、v3 和 v4 版本上，GSELI 與基線模型在 Hits@10 上的統計顯著性分析。對於 FB15K-237 資料集，所有版本皆具正 t 統計值和 p 值小於 0.015，顯示與基線模型有顯著差異，其中 v4 版本表現最佳（t 統計值 = 6.00，p 值 = 0.0018）。對於 WN18RR 資料集，所有版本也皆具正 t 統計值和 p 值小於 0.016，顯示顯著差異，其中 v1 版本表現最佳（t 統計值 = 5.46，p 值 = 0.0028）。總體而言，GSELI 在兩個資料集的所有版本中皆顯著優於基線模型。
Table 5 Hits@10 results on FB15k-237 and WN18RR for versions v1, v2, v3, and v4

| Model | FB15k-237 | WN18RR |
| :--- | :--- | :--- |
| | v1 | v2 | v3 | v4 | Avg | v1 | v2 | v3 | v4 | Avg |
| Grail | 64.15 | 81.80 | 82.83 | 89.29 | 79.52 | 84.25 | 78.68 | 58.43 | 73.41 | 73.69 |
| TATC | 62.20 | 80.02 | 84.16 | 88.41 | 78.70 | 82.45 | 78.68 | 58.60 | 73.41 | 73.29 |
| COMPILE | 67.66 | 82.98 | 84.67 | 87.44 | 80.69 | 83.60 | 79.82 | 60.69 | 75.49 | 74.90 |
| SNRI | 71.79 | 86.50 | 89.59 | 89.39 | 84.32 | 87.23 | 83.10 | 67.31 | 83.32 | 80.24 |
| RMPI | 71.71 | 83.37 | 86.10 | 88.69 | 82.47 | 87.77 | 82.43 | 73.14 | 81.24 | 81.15 |
| DEKG-ILP | **83.01** | 91.33 | **94.68** | 94.78 | 90.95 | 91.31 | 88.81 | 76.36 | 87.12 | 85.90 |
| GSELI(Ours) | 81.46 | **93.09** | 93.60 | **96.04** | **91.05** | **93.44** | **89.94** | **77.16** | **88.26** | **87.20** |
表 5 FB15k-237 和 WN18RR 在 v1、v2、v3 和 v4 版本上的 Hits@10 結果

| 模型 | FB15k-237 | WN18RR |
| :--- | :--- | :--- |
| | v1 | v2 | v3 | v4 | 平均 | v1 | v2 | v3 | v4 | 平均 |
| 聖杯 | 64.15 | 81.80 | 82.83 | 89.29 | 79.52 | 84.25 | 78.68 | 58.43 | 73.41 | 73.69 |
| TATC | 62.20 | 80.02 | 84.16 | 88.41 | 78.70 | 82.45 | 78.68 | 58.60 | 73.41 | 73.29 |
| 編譯 | 67.66 | 82.98 | 84.67 | 87.44 | 80.69 | 83.60 | 79.82 | 60.69 | 75.49 | 74.90 |
| SNRI | 71.79 | 86.50 | 89.59 | 89.39 | 84.32 | 87.23 | 83.10 | 67.31 | 83.32 | 80.24 |
| RMPI | 71.71 | 83.37 | 86.10 | 88.69 | 82.47 | 87.77 | 82.43 | 73.14 | 81.24 | 81.15 |
| DEKG-ILP | **83.01** | 91.33 | **94.68** | 94.78 | 90.95 | 91.31 | 88.81 | 76.36 | 87.12 | 85.90 |
| GSELI（我們的） | 81.46 | **93.09** | 93.60 | **96.04** | **91.05** | **93.44** | **89.94** | **77.16** | **88.26** | **87.20** |
### 5.5 Ablation study
This section presents the ablation study results for MRR and Hits@10 on the EQ, MB, and MF versions of the FB15k-237 dataset, focusing on fully-inductive and bridging-inductive links.
Our goal is to analyze the individual impact of each GSELI component by removing their respective contributions. These components include: (1) relation-specific semantic features (GSELI-S), (2) semantic-aware contrastive learning (GSELI-C), (3) PPR-based local clustering subgraph extraction (GSELI-L), and (4) complete neighboring relations within subgraphs (GSELI-R).
Figure 6 clearly demonstrates the superior MRR and Hits@10 performance of the original GSELI compared to its variants. Specifically:
(1) Overall, GSELI's Hits@10 outperforms all other variants across datasets. For MRR, GSELI is lower than the GSELI-R variant in FB15k-237 EQ's fully-inductive links and FB15k-237 MB's bridging-inductive links, but higher in all other cases. This indicates that while our components enhance overall performance, complete neighboring relations contribute more significantly to MRR.
(2) GSEIL-S: The gap between GSELI-S and other variants highlights the importance of extracting relation-specific semantic features from the original KG. This demonstrates that rich semantic features positively impact both fully-inductive and bridging-inductive links, especially the latter.
(3) GSEIL-C: Semantic-aware contrastive learning optimizes relation-specific semantic features by generating positive and negative samples. The performance drop of GSELI-C compared to GSELI underscores this component's effectiveness.
(4) GSEIL-L: PPR-based local clustering samples tightly related subgraphs, effectively mitigating the exponential growth of neighbors within fixed hops. The improvement from GSELI-L to GSELI highlights this method's advantage.
(5) GSEIL-R: Incorporating complete neighboring relations (including features and paths) addresses the issue of sparse subgraphs. The enhancement from GSELI-R to GSELI demonstrates this approach's effectiveness.
(6) The reasons why certain modules have a smaller impact on the experiments or even outperform GSELI are as follows: these modules exhibit strong independence, resulting in minimal effect on overall performance; dataset characteristics make them particularly effective on specific datasets; parameter tuning makes them more suitable for the current task; they perform well individually but may cause conflicts or redundancy when combined with others; complex models may lead to overfitting, whereas simpler modules can better capture the essential data characteristics.
### 5.5 消融研究
本節呈現了在 FB15k-237 資料集的 EQ、MB 和 MF 版本上，針對全歸納式和橋接歸納式連結的 MRR 和 Hits@10 的消融研究結果。
我們的目標是透過移除各個 GSELI 組件的貢獻，來分析其個別影響。這些組件包括：(1) 特定關係的語意特徵 (GSELI-S)，(2) 語意感知的對比學習 (GSELI-C)，(3) 基於 PPR 的局部聚類子圖提取 (GSELI-L)，以及 (4) 子圖內的完整鄰近關係 (GSELI-R)。
圖 6 清楚地展示了原始 GSELI 相較於其變體在 MRR 和 Hits@10 效能上的優越性。具體而言：
(1) 整體而言，GSELI 的 Hits@10 在所有資料集上均優於其他變體。在 MRR 方面，GSELI 在 FB15k-237 EQ 的全歸納連結和 FB15k-237 MB 的橋接歸納連結中低於 GSELI-R 變體，但在所有其他情況下均較高。這表示雖然我們的组件能提升整體效能，但完整的鄰近關係對 MRR 的貢獻更為顯著。
(2) GSEIL-S：GSELI-S 與其他變體之間的差距凸顯了從原始知識圖譜中提取特定關係語義特徵的重要性。這表明豐富的語義特徵對全歸納式和橋接歸納式連結，尤其是後者，都有正向影響。
(3) GSEIL-C：語意感知的對比學習透過生成正負樣本來優化特定關係的語意特徵。GSELI-C 相較於 GSELI 的效能下降，凸顯了此組件的有效性。
(4) GSEIL-L：基於 PPR 的局部聚類方法能夠取樣緊密相關的子圖，有效緩解了固定跳數內鄰居節點數量指數增長的問題。從 GSELI-L 到 GSELI 的改進，凸顯了該方法的優勢。
(5) GSEIL-R：整合完整的鄰近關係（包括特徵和路徑）解決了稀疏子圖的問題。从 GSELI-R 到 GSELI 的增強證明了此方法的有效性。
(6) 某些模塊對實驗影響較小甚至超越 GSELI 的原因如下：這些模塊表現出很強的獨立性，對整體性能影響甚微；數據集特徵使其在特定數據集上特別有效；參數調整使其更適合當前任務；它們單獨表現良好，但與其他模塊結合時可能引起衝突或冗餘；複雜模型可能導致過擬合，而較簡單的模塊更能捕捉基本數據特徵。
Table 6 Percentage reduction of subgraph size for the v1, v2, v3, and v4 versions of the FB15k-237, NELL-995, and WN18RR datasets

| | FB15k-237 | NELL-995 | WN18RR |
| :--- | :--- | :--- | :--- |
| | v1 | v2 | v3 | v4 | v1 | v2 | v3 | v4 | v1 | v2 | v3 | v4 |
| GraIL | 433.7 | 804.66 | 1470.68 | 1965.24 | 76.45 | 666.37 | 1427.08 | 1282.89 | 7.34 | 8.23 | 9.47 | 7.29 |
| GraIL+PPR | 6.85 | 5.16 | 4.05 | 3.58 | 11.45 | 7.45 | 6.14 | 4.12 | 4.46 | 4.44 | 4.73 | 4.46 |
| Reduction (%) | 98.42 | 99.36 | 99.72 | 99.82 | 85.02 | 98.88 | 99.57 | 99.68 | 39.24 | 46.05 | 50.05 | 38.82 |
表 6 FB15k-237、NELL-995 和 WN18RR 資料集的 v1、v2、v3 和 v4 版本子圖大小的百分比縮減

| | FB15k-237 | NELL-995 | WN18RR |
| :--- | :--- | :--- | :--- |
| | v1 | v2 | v3 | v4 | v1 | v2 | v3 | v4 | v1 | v2 | v3 | v4 |
| GraIL | 433.7 | 804.66 | 1470.68 | 1965.24 | 76.45 | 666.37 | 1427.08 | 1282.89 | 7.34 | 8.23 | 9.47 | 7.29 |
| GraIL+PPR | 6.85 | 5.16 | 4.05 | 3.58 | 11.45 | 7.45 | 6.14 | 4.12 | 4.46 | 4.44 | 4.73 | 4.46 |
| 減少（%） | 98.42 | 99.36 | 99.72 | 99.82 | 85.02 | 98.88 | 99.57 | 99.68 | 39.24 | 46.05 | 50.05 | 38.82 |
## 6 Further experiments
This section, “Further Experiments, includes the following parts: subgraph extraction efficiency (see Sect. 6.1), hyper-parameter sensitivity study (see Sect. 6.2), efficiency analysis (see Sect. 6.3), and Case Study (see Sect. 6.4).

### 6.1 Subgraph extraction efficiency
This section validates the performance of PPR-based subgraph extraction by examining subgraph size. We compare GraIL and GraIL+PPR, as all baseline models use conventional extraction methods. Table 6 shows the percentage reduction in subgraph size for the v1, v2, v3, and v4 versions of the FB15k-237, NELL-995, and WN18RR datasets.
The table 6 reveals that GraIL+PPR significantly reduces subgraph size in all datasets and versions. Notably, the FB15k-237 and NELL-995 datasets exhibit reductions up to 99.82%. In the WN18RR dataset, reductions are smaller but still substantial, with a minimum of 38.82%. These results indicate that PPR-based subgraph extraction substantially reduces subgraph size, decreasing the nodes and edges processed during graph neural network computations. This reduction lowers computational complexity and memory usage without compromising model accuracy and effectiveness. Thus, PPR-based subgraph extraction effectively removes redundant information while retaining essential data, enhancing overall model performance.

### 6.2 Hyper-parameter sensitivity study
This section explores the parameter sensitivity of GSELI on different datasets. The hyperparameter θ (controlling the degree of random operations of the relations) and the parameter λ (adjusting the ratio of supervised learning to contrastive loss) are critical factors. We used the FB15K-237 and WN18RR datasets, focusing on fully-inductive and bridging-inductive links, to evaluate GSELI's Hits@10 results with parameters θ and λ ranging from 0.1 to 0.9 in increments of 0.2.
As shown in Fig. 7, our observations indicate that the Hits@10 results in the EQ of the FB15K-237 and NELL-995 datasets are influenced by parameters θ and λ. For the FB15K-237 EQ dataset, performance is generally better when θ∈ [0.3, 0.5] and λ ∈ [0.1, 0.5]. Specifically, for fully-inductive links, the highest performance is achieved with θ = 0.5 and λ = 0.3. For bridging-inductive links, the highest performance is reached with θ = 0.3 and λ = 0.3, as well as θ = 0.5 and λ = 0.1. Additionally, in the EQ of the NELL-995 dataset, the best performance for both fully-inductive and bridging-inductive links is achieved with θ = 0.5 and λ = 0.3. Therefore, we select θ = 0.5 and λ = 0.3 as the optimal parameter settings.
## 6 進一步實驗
本節「進一步實驗」包含以下部分：子圖提取效率（參見 6.1 節）、超參數敏感性研究（參見 6.2 節）、效率分析（參見 6.3 節）和案例研究（參見 6.4 節）。

### 6.1 子圖提取效率
本節透過檢視子圖大小來驗證基於 PPR 的子圖提取效能。我們比較 GraIL 和 GraIL+PPR，因為所有基線模型都使用傳統的提取方法。表 6 顯示了 FB15k-237、NELL-995 和 WN18RR 資料集的 v1、v2、v3 和 v4 版本的子圖大小百分比縮減。
表 6 顯示，GraIL+PPR 在所有數據集和版本中都顯著減小了子圖大小。值得注意的是，FB15k-237 和 NELL-995 數據集的縮減率高達 99.82%。在 WN18RR 數據集中，縮減率較小但仍然可觀，最低為 38.82%。這些結果表明，基於 PPR 的子圖提取顯著減小了子圖大小，減少了圖神經網路計算過程中處理的節點和邊。這種縮減降低了計算複雜度和記憶體使用量，同時不影響模型的準確性和有效性。因此，基於 PPR 的子圖提取有效地移除了冗餘資訊，同時保留了必要數據，增強了模型的整體性能。

### 6.2 超參數敏感性研究
本節探討 GSELI 在不同資料集上的參數敏感性。超參數 θ（控制關係隨機操作的程度）和參數 λ（調整監督式學習與對比式損失的比例）是關鍵因素。我们使用 FB15K-237 和 WN18RR 資料集，重點關注全歸納式和橋接歸納式連結，以評估 GSELI 在參數 θ 和 λ 從 0.1 到 0.9（增量為 0.2）範圍內的 Hits@10 結果。
如圖 7 所示，我們的觀察表明，FB15K-237 和 NELL-995 資料集的 EQ 中的 Hits@10 結果受參數 θ 和 λ 的影響。對於 FB15K-237 EQ 資料集，當 θ∈ [0.3, 0.5] 且 λ ∈ [0.1, 0.5] 時，性能通常更好。具體來說，對於全歸納式連結，在 θ = 0.5 和 λ = 0.3 時達到最高性能。對於橋接歸納式連結，在 θ = 0.3 和 λ = 0.3，以及 θ = 0.5 和 λ = 0.1 時達到最高性能。此外，在 NELL-995 資料集的 EQ 中，全歸納式和橋接歸納式連結的最佳性能均在 θ = 0.5 和 λ = 0.3 時達到。因此，我們選擇 θ = 0.5 和 λ = 0.3 作為最佳參數設置。
Fig. 7 The heat map of the Hit@10 results on EQ of FB15k-237 and NELL-995, for both solely fully-inductive links and solely bridging-inductive links
圖 7 FB15k-237 和 NELL-995 之 EQ 上 Hit@10 結果的熱圖，適用於僅全歸納連結和僅橋接歸納連結
Fig. 8 GPU and CPU memory usage during the training phase for EQ, MB, and MF on FB15K-237, WN18RR, and NELL-995 datasets
圖 8 在 FB15K-237、WN18RR 和 NELL-995 資料集上，EQ、MB 和 MF 訓練階段的 GPU 和 CPU 記憶體使用情況
Fig. 9 Total runtime on EQ, MB, and MF for the FB15K-237, WN18RR, and NELL-995 datasets, including training and testing on mixed, bridging-inductive, and fully-inductive datasets
圖 9 FB15K-237、WN18RR 和 NELL-995 資料集在 EQ、MB 和 MF 上的總運行時間，包括混合、橋接歸納和全歸納資料集的訓練和測試
### 6.3 Efficiency analysis
This section examines the memory usage and total runtime of GSELI and baseline models on the EQ, MB, and MF versions of the FB15K-237, WN18RR, and NELL-995 datasets. All baseline models utilized both GPU and CPU during the training phase. However, only DEKG and GSELI continued using GPU and CPU during the ranking test phase, while the other models used only the CPU. To ensure a fair comparison, this paper reports memory usage only during the training phase. The total runtime includes both the training time and the testing time on mix, bridging, and fully-inductive datasets.
Figure 8 illustrates the GPU and CPU usage of GSELI and baseline models across the EQ, MB, and MF versions of the FB15K-237, NELL-995, and WN18RR datasets. The results indicate that GSELI significantly reduces GPU memory usage in most cases compared to other models. Notably, on the FB15K-237 and NELL-995 datasets, GSELI has the lowest GPU memory usage, demonstrating superior memory efficiency. Although GSELI does not show significant advantages in CPU memory usage, its consumption remains within acceptable limits. Overall, GSELI excels in GPU memory usage and maintains stability in CPU memory usage.
Figure 9 depicts the total runtime of GSELI and baseline models across the EQ, MB, and MF versions of the FB15K-237, NELL-995, and WN18RR datasets. The results show that GSELI has a significantly lower runtime for the MB and MF versions on the FB15K-237 dataset. For the NELL-995 dataset, GSELI has the shortest total runtime for the EQ and MF versions. On the WN18RR dataset, GSELI's runtime for the MB and MF versions is also significantly lower than other models. Overall, GSELI demonstrates significantly lower runtime across various datasets, showcasing higher operational efficiency.
### 6.3 效率分析
本節檢視 GSELI 和基線模型在 FB15K-237、WN18RR 和 NELL-995 資料集的 EQ、MB 和 MF 版本上的記憶體使用量和總執行時間。所有基線模型在訓練階段都同時使用了 GPU 和 CPU。然而，只有 DEKG 和 GSELI 在排名測試階段繼續使用 GPU 和 CPU，而其他模型僅使用 CPU。為確保公平比較，本文僅報告訓練階段的記憶體使用量。總執行時間包括在混合、橋接和全歸納資料集上的訓練時間和測試時間。
圖 8 說明了 GSELI 和基線模型在 FB15K-237、NELL-995 和 WN18RR 資料集的 EQ、MB 和 MF 版本上的 GPU 和 CPU 使用情況。結果表明，在大多數情況下，GSELI 與其他模型相比，顯著降低了 GPU 記憶體使用量。值得注意的是，在 FB15K-237 和 NELL-995 資料集上，GSELI 的 GPU 記憶體使用量最低，展現出卓越的記憶體效率。儘管 GSELI 在 CPU 記憶體使用量上沒有顯著優勢，但其消耗量仍在可接受範圍內。總體而言，GSELI 在 GPU 記憶體使用方面表現出色，並在 CPU 記憶體使用方面保持穩定。
圖 9 描繪了 GSELI 和基線模型在 FB15K-237、NELL-995 和 WN18RR 資料集的 EQ、MB 和 MF 版本上的總執行時間。結果顯示，GSELI 在 FB15K-237 資料集的 MB 和 MF 版本上執行時間顯著較短。對於 NELL-995 資料集，GSELI 在 EQ 和 MF 版本上的總執行時間最短。在 WN18RR 資料集上，GSELI 在 MB 和 MF 版本上的執行時間也顯著低於其他模型。總體而言，GSELI 在各種資料集上展現出顯著較低的執行時間，顯示出更高的運作效率。
Table 7 Rule generation from NELL-995 EQ and WN18RR EQ datasets

| Rule | Score |
| :--- | :--- |
| NELL-995 EQ | |
| company_economic_sector(X, Y) ^ agent_belongs_to_organization(Y, Z) ^ acquired(Z, W) ∧ organization_hired_person(W, V) ⇒ works_for(X, V) | 0.349 |
| company_economic_sector(X, Y) ∧ works_for(Y, Z) ∧ acquired(Z, W) ∧ organization_hired_person(W, V) ⇒ works_for(X, V) | 0.347 |
| organization_hired_person(X, Y) ^ agent_belongs_to_organization(Y, Z) ^ acquired(Z, W) ^ organization_hired_person(W, V) ⇒ works_for(X, V) | 0.330 |
| subpart_of(X, Y) ∧ company_economic_sector(Y, Z) ^ works_for(Z, W) ∧ subpart_of(W, V) ⇒ subpart_of(X, V) | 0.700 |
| subpart_of(X, Y) ∧ subpart_of_organization(Y, Z) ∧ works_for(Z, W) ∧ subpart_of(W, V) ⇒ subpart_of(X, V) | 0.698 |
| subpart_of(X, Y) ∧ organization_headquartered_in_city(Y, Z) ^ works_for(Z, W) ∧ subpart_of(W, V) ⇒ subpart_of(X, V) | 0.695 |
| WN18RR EQ | |
| hypernym (X, Y) ∧ similar_to(Y, Z) ^ similar_to(Z, W) ∧ similar_to(W, V) ⇒ has_part(X, V) | 0.973 |
| similar_to(X, Y) ∧ similar_to(Y, Z) ^ instance_hypernym(Z, W) ^ member_meronym(W, V) ⇒ has_part(X, V) | 0.970 |
| similar_to(X, Y) ∧ similar_to(Y, Z) ∧ has_part(Z, W) ∧ similar_to(W, V) ⇒ has_part(X, V) | 0.969 |
| verb_group(X, Y) ∧ similar_to(Y, Z) ∧ similar_to(Z, W) ∧ synset_domain_topic_of(W, V) ⇒ also_see(X, V) | 0.993 |
| also_see(X, Y) ∧^ also_see(Y, Z) ∧ instance_hypernym(Z, W) ^ has_part(W, V) ⇒ also_see(X, V) | 0.992 |
| verb_group(X, Y) ∧ synset_domain_topic_of(Y, Z) ^ similar_to(Z, W) ∧ synset_domain_topic_of(W, V) ⇒ also_see(X, V) | 0.992 |
表 7 從 NELL-995 EQ 和 WN18RR EQ 資料集生成規則

| 規則 | 分數 |
| :--- | :--- |
| NELL-995 EQ | |
| 公司經濟部門(X, Y) ^ 代理人屬於組織(Y, Z) ^ 收購(Z, W) ∧ 組織僱用人員(W, V) ⇒ 為...工作(X, V) | 0.349 |
| 公司經濟部門(X, Y) ∧ 為...工作(Y, Z) ∧ 收購(Z, W) ∧ 組織僱用人員(W, V) ⇒ 為...工作(X, V) | 0.347 |
| 組織僱用人員(X, Y) ^ 代理人屬於組織(Y, Z) ^ 收購(Z, W) ^ 組織僱用人員(W, V) ⇒ 為...工作(X, V) | 0.330 |
| 子部分(X, Y) ∧ 公司經濟部門(Y, Z) ^ 為...工作(Z, W) ∧ 子部分(W, V) ⇒ 子部分(X, V) | 0.700 |
| subpart_of(X, Y) ∧ subpart_of_organization(Y, Z) ∧ works_for(Z, W) ∧ subpart_of(W, V) ⇒ subpart_of(X, V) | 0.698 |
| subpart_of(X, Y) ∧ organization_headquartered_in_city(Y, Z) ^ works_for(Z, W) ∧ subpart_of(W, V) ⇒ subpart_of(X, V) | 0.695 |
| WN18RR EQ | |
| 上位詞 (X, Y) ∧ 相似於(Y, Z) ^ 相似於(Z, W) ∧ 相似於(W, V) ⇒ 具有部分(X, V) | 0.973 |
| 相似於(X, Y) ∧ 相似於(Y, Z) ^ 實例上位詞(Z, W) ^ 成員部分詞(W, V) ⇒ 具有部分(X, V) | 0.970 |
| similar_to(X, Y) ∧ similar_to(Y, Z) ∧ has_part(Z, W) ∧ similar_to(W, V) ⇒ has_part(X, V) | 0.969 |
| 動詞組(X, Y) ∧ 相似於(Y, Z) ∧ 相似於(Z, W) ∧ 同義詞集領域主題(W, V) ⇒ 也參見(X, V) | 0.993 |
| also_see(X, Y) ∧^ also_see(Y, Z) ∧ instance_hypernym(Z, W) ^ has_part(W, V) ⇒ also_see(X, V) | 0.992 |
| 動詞組(X, Y) ∧ 同義詞集領域主題(Y, Z) ^ 相似於(Z, W) ∧ 同義詞集領域主題(W, V) ⇒ 也參見(X, V) | 0.992 |
Additionally, Table 2 reveals that the number of the EQ, MB, and MF versions of the FB15K-237, NELL-995, and WN18RR datasets gradually improves. However, the results from Figs. 8 and 9 show that GSELI maintains high efficiency in GPU memory usage and total runtime. This further demonstrates GSELI's ability to remain efficient and scalable when handling large-scale datasets, giving it an advantage in large-scale data processing scenarios.

### 6.4 Case study
Subgraphs, as effective path combinations between target entities, offer more comprehensive insights than single rules, revealing the rationale of each target link (u, r₁, v) within the cycle. We generated all relation rule cycles of lengths up to four and input them into the GSELI model for scoring. These rules include the target relation (rule head) and the relation path (rule body). Using the sigmoid function, we normalized the scores and selected the top three cycles with the highest scores.
Table 7 presents examples of EQ rule generation by GSELI on the NELL-995 and WN18RR datasets, highlighting its explanatory power and applicability to complex datasets. GSELI identifies and interprets complex relations through rule cycles, generating high-scoring paths for more meaningful link explanations. This enhances the model's prediction accuracy, transparency, and interpretability, demonstrating GSELI's robust capability and potential in handling diverse and complex datasets.
此外，表 2 揭示了 FB15K-237、NELL-995 和 WN18RR 資料集的 EQ、MB 和 MF 版本的數量逐漸提升。然而，圖 8 和圖 9 的結果顯示，GSELI 在 GPU 記憶體使用和總執行時間方面保持高效率。這進一步證明了 GSELI 在處理大規模資料集時能夠保持高效和可擴展性，使其在大型資料處理場景中具有優勢。

### 6.4 案例研究
子圖作為目標實體之間的有效路徑組合，相較於單一規則，能提供更全面的洞見，揭示循環中每個目標連結 (u, r₁, v) 的理據。我們生成了所有長度最多為四的關係規則循環，並將其輸入 GSELI 模型进行評分。這些規則包括目標關係（規則頭）和關係路徑（規則體）。利用 sigmoid 函數，我们將分數正規化，並選出得分最高的前三個循環。
表 7 展示了 GSELI 在 NELL-995 和 WN18RR 資料集上生成 EQ 規則的範例，突顯其在複雜資料集上的解釋能力和適用性。GSELI 透過規則循環識別並解釋複雜的關係，為更有意義的連結解釋生成高分路徑。這增強了模型的預測準確性、透明度和可解釋性，展示了 GSELI 在處理多樣化和複雜資料集方面的強大能力和潛力。
## 7 Conclusion
This paper presents GSELI, an ILP model that integrates global semantic features and enhanced local subgraphs. It enables predictions for both unseen-unseen entities in emerging KGs (fully-inductive links) and the more challenging unseen entities from the original KG to the emerging KG (bridging-inductive links). GSELI extracts CL-based global relation semantic features, effectively addressing the topological limitations of both the original KG and the emerging KG. To further enhance the topological information of subgraphs, GSELI employs a PPR-based local clustering approach to sample tightly-related subgraphs, mitigating the exponential growth of neighbors within a fixed number of hops. Additionally, it integrates complete neighboring relations to address sparse subgraphs. Experiments show GSELI's superior performance compared to state-of-the-art models for both fully-inductive and bridging-inductive links.
Our future research plans include: (1) conducting a comprehensive evaluation of fully-inductive and bridging-inductive links; (2) advancing more challenging ILP involving unseen entities and relations; (3) leveraging large language models to enhance ILP.

**Acknowledgements** This research is supported by National Natural Science Foundation of China (Grant Number: 61375084), the Shandong Natural Science Foundation of China (Grant Number: ZR2019MF064), and the Technological Small and Medium-sized Enterprise Innovation Ability Enhancement Project (Grant Number: 2022TSGC2189). Thanks are due to all the anonymous reviewers.

**Data availability** The datasets supporting the results of this article are included within the references.
## 7 結論
本文提出了 GSELI，一個整合了全域語意特徵和增強型區域子圖的 ILP 模型。它能够對新興知識圖譜中的未見實體（全歸納式連結）以及更具挑戰性的從原始知識圖譜到新興知識圖譜的未見實體（橋接歸納式連結）进行預測。GSELI 提取了基於 CL 的全域關聯語意特徵，有效解决了原始知識圖譜和新興知識圖譜的拓撲限制。为了进一步增强子图的拓扑信息，GSELI 採用基於 PPR 的局部聚類方法對緊密相關的子圖进行採樣，緩解了固定跳數內鄰居節點數量呈指数級增長的問題。此外，它整合了完整的鄰近關係以處理稀疏子圖。實驗表明，GSELI 在全歸納式和橋接歸納式連結方面均優於最先進的模型。
我們未來的研究計畫包括：(1) 對全歸納式和橋接歸納式連結進行全面評估；(2) 推進涉及未見實體和關係的更具挑戰性的 ILP；(3) 利用大型語言模型來增強 ILP。

**致謝** 本研究受國家自然科學基金（批准號：61375084）、山東省自然科學基金（批准號：ZR2019MF064）及科技型中小企業創新能力提升計畫（批准號：2022TSGC2189）資助。謹此感謝所有匿名審稿人。

**數據可用性** 支持本文結果的數據集包含在參考文獻中。
**Declarations**

**Conflict of interest** The authors declare that they have no conflict of interest.
**聲明**

**利益衝突** 作者聲明無利益衝突。
### References
1. Huang X, Zhang J, Li D et al (2019) Knowledge graph embedding based question answering. In: Proceedings of the twelfth ACM international conference on web search and data mining, pp 105-113
2. Wang X, Wang D, Xu C et al (2019) Explainable reasoning over knowledge graphs for recommendation. In: Proceedings of the AAAI conference on artificial intelligence, pp 5329-5336
3. Xiong C, Power R, Callan J (2017) Explicit semantic ranking for academic search via knowledge graph embedding. In: Proceedings of the 26th international conference on world wide web, pp 1271-1279
4. Dong X, Gabrilovich E, Heitz G et al (2014) Knowledge vault: a web-scale approach to probabilistic knowledge fusion. In: Proceedings of the 20th ACM SIGKDD international conference on Knowledge discovery and data mining, pp 601-610
5. Xie R, Liu Z, Jia J et al (2016) Representation learning of knowledge graphs with entity descriptions. In: Proceedings of the AAAI conference on artificial intelligence, pp 2659-2665
6. Hamilton W, Ying Z, Leskovec J (2017) Inductive representation learning on large graphs. In: Advances in neural information processing systems, vol 30
7. Zeng H, Zhou H, Srivastava A et al (2019) Graphsaint: graph sampling based inductive learning method. arXiv:1907.04931
8. Hamaguchi T, Oiwa H, Shimbo M et al (2017) Knowledge transfer for out-of-knowledge-base entities: a graph neural network approach. In: Proceedings of the 26th International Joint Conference on Artificial Intelligence, pp 1802-1808
9. Bi Z, Zhang T, Zhou P et al (2020) Knowledge transfer for out-of-knowledge-base entities: improving graph-neural-network-based embedding using convolutional layers. IEEE Access 8:159039-159049
10. Wang C, Zhou X, Pan S et al (2022) Exploring relational semantics for inductive knowledge graph completion. In: Proceedings of the AAAI conference on artificial intelligence, pp 4184-4192
11. Teru K, Denis E, Hamilton W (2020) Inductive relation prediction by subgraph reasoning. In: International conference on machine learning, pp 9448-9457
12. Chen J, He H, Wu F et al (2021) Topology-aware correlations between relations for inductive link prediction in knowledge graphs. In: Proceedings of the AAAI conference on artificial intelligence, pp 6271-6278
13. Mai S, Zheng S, Yang Y et al (2021) Communicative message passing for inductive relation reasoning. In: Proceedings of the AAAI conference on artificial intelligence, pp 4294-4302
14. Xu X, Zhang P, He Y et al (2022) Subgraph neighboring relations infomax for inductive link prediction on knowledge graphs. arXiv:2208.00850
15. Mohamed HA, Pilutti D, James S et al (2023) Locality-aware subgraphs for inductive link prediction in knowledge graphs. Pattern Recogn Lett 167:90-97
16. Geng Y, Chen J, Pan JZ et al (2023) Relational message passing for fully inductive knowledge graph completion. In: Proceedings of the 39th international conference on data engineering (ICDE), pp 1221-1233
17. Ali M, Berrendorf M, Galkin M et al (2021) Improving inductive link prediction using hyper-relational facts. In: Proceedings of the 20th international semantic web conference, pp 74-92
18. Zhang Y, Wang W, Yin H et al (2023) Disconnected emerging knowledge graph oriented inductive link prediction. In: Proceedings of the 39th international conference on data engineering (ICDE), pp 381-393
19. Bordes A, Usunier N, Garcia-Duran A et al (2013) Translating embeddings for modeling multi-relational data. In: Advances in neural information processing systems, vol 26
20. Wang Z, Zhang J, Feng J et al (2014) Knowledge graph embedding by translating on hyperplanes. In: Proceedings of the AAAI conference on artificial intelligence, pp 1112-1119
21. Nickel M, Tresp V, Kriegel HP et al (2011) A three-way model for collective learning on multi-relational data. In: Proceedings of the 28th international conference on machine learning, pp 3104482-3104584
22. Yang B, Yih W, He X et al (2014) Embedding entities and relations for learning and inference in knowledge bases. arXiv:1412.6575
23. Schlichtkrull M, Kipf TN, Bloem P et al (2018) Modeling relational data with graph convolutional networks. In: Proceedings of the 15th international semantic web conference, pp 593-607
24. Vashishth S, Sanyal S, Nitin V et al (2019) Composition-based multi-relational graph convolutional networks. arXiv:1911.03082
25. Galárraga LA, Teflioudi C, Hose K et al (2013) Amie: association rule mining under incomplete evidence in ontological knowledge bases. In: Proceedings of the 22nd international conference on world wide web, pp 413-422
26. Meilicke C, Fink M, Wang Y et al (2018) Fine-grained evaluation of rule-and embedding-based systems for knowledge graph completion. In: Proceedings of the 17th international semantic web conference, pp 3-20
27. Meilicke C, Chekol MW, Ruffinelli D et al (2019) Anytime bottom-up rule learning for knowledge graph completion. In: Proceedings of the 28th international joint conference on artificial intelligence, pp 3137-3143
28. Yang F, Yang Z, Cohen WW (2017) Differentiable learning of logical rules for knowledge base reasoning. In: Advances in neural information processing systems, vol 30
29. Sadeghian A, Armandpour M, Ding P et al (2019) Drum: end-to-end differentiable rule mining on knowledge graphs. In: Advances in Neural Information Processing Systems, vol 32
30. Qu M, Chen J, Xhonneux LP et al (2020) Rnnlogic: learning logic rules for reasoning on knowledge graphs. arXiv:2010.04029
31. Wang L, Zhao W, Wei Z, et al (2022) Simkgc: Simple contrastive knowledge graph completion with pre-trained language models. In: Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics, 4281-4294
32. Pan Y, Liu J, Zhang L et al (2021) Learning first-order rules with relational path contrast for inductive relation reasoning. arXiv:2110.08810
33. Kwak H, Jung HBK (2022) Subgraph representation learning with hard negative samples for inductive link prediction. In: Proceedings of the 2022 international conference on acoustics, speech and signal processing (ICASSP), pp 4768-4772
34. Gilmer J, Schoenholz SS, Riley PF et al (2017) Neural message passing for quantum chemistry. In: International conference on machine learning, pp 1263-1272
35. Cho K, Bart, Bahdanau D et al (2014) Learning phrase representations using rnn encoder-decoder for statistical machine translation. In: Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP), pp 1724-1734
36. Toutanova K, Chen D, Pantel P et al (2015) Representing text for joint embedding of text and knowledge bases. In: Proceedings of the 2015 conference on empirical methods in natural language processing, pp 1499-1509
37. Xiong W, Hoang T, Wang WY (2017) Deeppath: a reinforcement learning method for knowledge graph reasoning. In: Proceedings of the 2017 conference on empirical methods in natural language processing, pp 564-573
38. Dettmers T, Minervini P, Stenetorp P et al (2018) Convolutional 2d knowledge graph embeddings. In: Proceedings of the AAAI conference on artificial intelligence, pp 1811-1818
39. Kingma DP, Ba J (2014) Adam: a method for stochastic optimization. arXiv:1412.6980

**Publisher's Note** Springer Nature remains neutral with regard to jurisdictional claims in published maps and institutional affiliations.

Springer Nature or its licensor (e.g. a society or other partner) holds exclusive rights to this article under a publishing agreement with the author(s) or other rightsholder(s); author self-archiving of the accepted manuscript version of this article is solely governed by the terms of such publishing agreement and applicable law.
### 參考文獻
1. Huang X, Zhang J, Li D et al (2019) Knowledge graph embedding based question answering. In: Proceedings of the twelfth ACM international conference on web search and data mining, pp 105-113
2. Wang X, Wang D, Xu C et al (2019) Explainable reasoning over knowledge graphs for recommendation. In: Proceedings of the AAAI conference on artificial intelligence, pp 5329-5336
3. Xiong C, Power R, Callan J (2017) Explicit semantic ranking for academic search via knowledge graph embedding. In: Proceedings of the 26th international conference on world wide web, pp 1271-1279
4. Dong X, Gabrilovich E, Heitz G et al (2014) Knowledge vault: a web-scale approach to probabilistic knowledge fusion. In: Proceedings of the 20th ACM SIGKDD international conference on Knowledge discovery and data mining, pp 601-610
5. Xie R, Liu Z, Jia J et al (2016) Representation learning of knowledge graphs with entity descriptions. In: Proceedings of the AAAI conference on artificial intelligence, pp 2659-2665
6. Hamilton W, Ying Z, Leskovec J (2017) Inductive representation learning on large graphs. In: Advances in neural information processing systems, vol 30
7. Zeng H, Zhou H, Srivastava A et al (2019) Graphsaint: graph sampling based inductive learning method. arXiv:1907.04931
8. Hamaguchi T, Oiwa H, Shimbo M et al (2017) Knowledge transfer for out-of-knowledge-base entities: a graph neural network approach. In: Proceedings of the 26th International Joint Conference on Artificial Intelligence, pp 1802-1808
9. Bi Z, Zhang T, Zhou P et al (2020) Knowledge transfer for out-of-knowledge-base entities: improving graph-neural-network-based embedding using convolutional layers. IEEE Access 8:159039-159049
10. Wang C, Zhou X, Pan S et al (2022) Exploring relational semantics for inductive knowledge graph completion. In: Proceedings of the AAAI conference on artificial intelligence, pp 4184-4192
11. Teru K, Denis E, Hamilton W (2020) Inductive relation prediction by subgraph reasoning. In: International conference on machine learning, pp 9448-9457
12. Chen J, He H, Wu F et al (2021) Topology-aware correlations between relations for inductive link prediction in knowledge graphs. In: Proceedings of the AAAI conference on artificial intelligence, pp 6271-6278
13. Mai S, Zheng S, Yang Y et al (2021) Communicative message passing for inductive relation reasoning. In: Proceedings of the AAAI conference on artificial intelligence, pp 4294-4302
14. Xu X, Zhang P, He Y et al (2022) Subgraph neighboring relations infomax for inductive link prediction on knowledge graphs. arXiv:2208.00850
15. Mohamed HA, Pilutti D, James S et al (2023) Locality-aware subgraphs for inductive link prediction in knowledge graphs. Pattern Recogn Lett 167:90-97
16. Geng Y, Chen J, Pan JZ et al (2023) Relational message passing for fully inductive knowledge graph completion. In: Proceedings of the 39th international conference on data engineering (ICDE), pp 1221-1233
17. Ali M, Berrendorf M, Galkin M et al (2021) Improving inductive link prediction using hyper-relational facts. In: Proceedings of the 20th international semantic web conference, pp 74-92
18. Zhang Y, Wang W, Yin H et al (2023) Disconnected emerging knowledge graph oriented inductive link prediction. In: Proceedings of the 39th international conference on data engineering (ICDE), pp 381-393
19. Bordes A, Usunier N, Garcia-Duran A et al (2013) Translating embeddings for modeling multi-relational data. In: Advances in neural information processing systems, vol 26
20. Wang Z, Zhang J, Feng J et al (2014) Knowledge graph embedding by translating on hyperplanes. In: Proceedings of the AAAI conference on artificial intelligence, pp 1112-1119
21. Nickel M, Tresp V, Kriegel HP et al (2011) A three-way model for collective learning on multi-relational data. In: Proceedings of the 28th international conference on machine learning, pp 3104482-3104584
22. Yang B, Yih W, He X et al (2014) Embedding entities and relations for learning and inference in knowledge bases. arXiv:1412.6575
23. Schlichtkrull M, Kipf TN, Bloem P et al (2018) Modeling relational data with graph convolutional networks. In: Proceedings of the 15th international semantic web conference, pp 593-607
24. Vashishth S, Sanyal S, Nitin V et al (2019) Composition-based multi-relational graph convolutional networks. arXiv:1911.03082
25. Galárraga LA, Teflioudi C, Hose K et al (2013) Amie: association rule mining under incomplete evidence in ontological knowledge bases. In: Proceedings of the 22nd international conference on world wide web, pp 413-422
26. Meilicke C, Fink M, Wang Y et al (2018) Fine-grained evaluation of rule-and embedding-based systems for knowledge graph completion. In: Proceedings of the 17th international semantic web conference, pp 3-20
27. Meilicke C, Chekol MW, Ruffinelli D et al (2019) Anytime bottom-up rule learning for knowledge graph completion. In: Proceedings of the 28th international joint conference on artificial intelligence, pp 3137-3143
28. Yang F, Yang Z, Cohen WW (2017) Differentiable learning of logical rules for knowledge base reasoning. In: Advances in neural information processing systems, vol 30
29. Sadeghian A, Armandpour M, Ding P et al (2019) Drum: end-to-end differentiable rule mining on knowledge graphs. In: Advances in Neural Information Processing Systems, vol 32
30. Qu M, Chen J, Xhonneux LP et al (2020) Rnnlogic: learning logic rules for reasoning on knowledge graphs. arXiv:2010.04029
31. Wang L, Zhao W, Wei Z, et al (2022) Simkgc: Simple contrastive knowledge graph completion with pre-trained language models. In: Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics, 4281-4294
32. Pan Y, Liu J, Zhang L et al (2021) Learning first-order rules with relational path contrast for inductive relation reasoning. arXiv:2110.08810
33. Kwak H, Jung HBK (2022) Subgraph representation learning with hard negative samples for inductive link prediction. In: Proceedings of the 2022 international conference on acoustics, speech and signal processing (ICASSP), pp 4768-4772
34. Gilmer J, Schoenholz SS, Riley PF et al (2017) Neural message passing for quantum chemistry. In: International conference on machine learning, pp 1263-1272
35. Cho K, Bart, Bahdanau D et al (2014) Learning phrase representations using rnn encoder-decoder for statistical machine translation. In: Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP), pp 1724-1734
36. Toutanova K, Chen D, Pantel P et al (2015) Representing text for joint embedding of text and knowledge bases. In: Proceedings of the 2015 conference on empirical methods in natural language processing, pp 1499-1509
37. Xiong W, Hoang T, Wang WY (2017) Deeppath: a reinforcement learning method for knowledge graph reasoning. In: Proceedings of the 2017 conference on empirical methods in natural language processing, pp 564-573
38. Dettmers T, Minervini P, Stenetorp P et al (2018) Convolutional 2d knowledge graph embeddings. In: Proceedings of the AAAI conference on artificial intelligence, pp 1811-1818
39. Kingma DP, Ba J (2014) Adam: a method for stochastic optimization. arXiv:1412.6980

**出版商註記** Springer Nature 對於已出版地圖和機構附屬關係中的管轄權主張保持中立。

Springer Nature 或其授權方（例如，學會或其他合作夥伴）根據與作者或其他權利持有人的出版協議，對本文擁有專有權；作者對已接受手稿版本的自我存檔，完全受此類出版協議的條款和適用法律的管轄。