
## **Published in:** Proceedings of the 30th VLDB Conference, 2004

# 資料庫查詢結果的機率排序

Surajit Chaudhuri, Gautam Das
Microsoft Research
One Microsoft Way
Redmond, WA 98053
USA
{surajitc, gautamd}@microsoft.com

Vagelis Hristidis
School of Comp. Sci.
Florida Intl. University
Miami, FL 33199
USA
vagelis@cs.fiu.edu

Gerhard Weikum
MPI Informatik
Stuhlsatzenhausweg 85
D-66123 Saarbruecken
Germany
weikum@mpi-sb.mpg.de

## Abstract

## 摘要

We investigate the problem of ranking answers to a database query when many tuples are returned. We adapt and apply principles of probabilistic models from Information Retrieval for structured data. Our proposed solution is domain independent. It leverages data and workload statistics and correlations. Our ranking functions can be further customized for different applications. We present results of preliminary experiments which demonstrate the efficiency as well as the quality of our ranking system.

我們研究了當返回許多元組時對資料庫查詢答案進行排序的問題。我們為結構化資料調整和應用了資訊檢索中機率模型的原則。我們提出的解決方案是領域無關的。它利用了資料和工作負載的統計資料和相關性。我們的排序函數可以為不同的應用進一步客製化。我們展示了初步實驗的結果，這些結果證明了我們排序系統的效率和品質。

## 1. Introduction

## 1. 緒論

Database systems support a simple Boolean query retrieval model, where a selection query on a SQL database returns all tuples that satisfy the conditions in the query. This often leads to the Many-Answers Problem: when the query is not very selective, too many tuples may be in the answer. We use the following running example throughout the paper:

資料庫系統支援一種簡單的布林查詢檢索模型，其中對 SQL 資料庫的選擇查詢會返回滿足查詢條件的所有元組。這通常會導致「多答案問題」：當查詢的選擇性不是很強時，答案中可能會包含太多的元組。在本文中，我們將使用以下運行範例：

Example: Consider a realtor database consisting of a single table with attributes such as (TID, Price, City, Bedrooms, Bathrooms, LivingArea, SchoolDistrict, View, Pool, Garage, BoatDock ...). Each tuple represents a home for sale in the US.

範例：考慮一個由單一表格組成的房地產經紀人資料庫，其屬性包括（TID、價格、城市、臥室數、浴室數、居住面積、學區、景觀、游泳池、車庫、船塢…）。每個元組代表一棟在美國待售的房屋。

Consider a potential home buyer searching for homes in this database. A query with a not very selective condition such as "City=Seattle and View=Waterfront" may result in too many tuples in the answer, since there are many homes with waterfront views in Seattle.

考慮一位潛在購房者在此資料庫中搜索房屋。一個選擇性不高的查詢，例如「城市=西雅圖 且 景觀=水岸」，可能會導致答案中有太多的元組，因為在西雅圖有很多擁有水岸景觀的房屋。

The Many-Answers Problem has been investigated outside the database area, especially in Information Retrieval (IR), where many documents often satisfy a given keyword-based query. Approaches to overcome this problem range from query reformulation techniques (e.g., the user is prompted to refine the query to make it more selective), to automatic ranking of the query results by their degree of “relevance" to the query (though the user may not have explicitly specified how) and returning only the top-K subset.

「多答案問題」已在資料庫領域之外被廣泛研究，尤其是在資訊檢索（IR）領域，其中許多文件通常滿足給定的基於關鍵字的查詢。克服這個問題的方法包括查詢重構技術（例如，提示使用者優化查詢以使其更具選擇性），以及根據查詢結果與查詢的「相關性」程度（儘管使用者可能沒有明確說明如何確定）對其進行自動排序，並僅返回前 K 個子集。

It is evident that automated ranking can have compelling applications in the database context. For instance, in the earlier example of a homebuyer searching for homes in Seattle with waterfront views, it may be preferable to first return homes that have other desirable attributes, such as good school districts, boat docks, etc. In general, customers browsing product catalogs will find such functionality attractive.

顯然，自動排序在資料庫領域具有引人注目的應用。例如，在前面搜索西雅圖水岸住宅的購房者例子中，可能更可取的是首先返回那些具有其他理想屬性（例如，優良的學區、船塢等）的住宅。一般來說，瀏覽產品目錄的顧客會發現此類功能很有吸引力。

In this paper we propose an automated ranking approach for the Many-Answers Problem for database queries. Our solution is principled, comprehensive, and efficient. We summarize our contributions below.

在本文中，我們為資料庫查詢的「多答案問題」提出了一種自動排序方法。我們的解決方案是原則性的、全面的且高效的。我們在下面總結我們的貢獻。

Any ranking function for the Many-Answers Problem has to look beyond the attributes specified in the query, because all answer tuples satisfy the specified conditions¹. However, investigating unspecified attributes is particularly tricky since we need to determine what the user's preferences for these unspecified attributes are. In this paper we propose that the ranking function of a tuple depends on two factors: (a) a global score which captures the global importance of unspecified attribute values, and (b) a conditional score which captures the strengths of dependencies (or correlations) between specified and unspecified attribute values. For example, for the query "City = Seattle and View = Waterfront", a home that is also located in a "SchoolDistrict = Excellent" gets high rank because good school districts are globally desirable. A home with also "BoatDock = Yes" gets high rank because people desiring a waterfront are likely to want a boat dock. While these scores may be estimated by the help of domain expertise or through user feedback, we propose an automatic estimation of these scores via workload as well as data analysis. For example, past workload may reveal that a large fraction of users seeking homes with a waterfront view have also requested for boat docks.

任何用於「多答案問題」的排序函數都必須超越查詢中指定的屬性，因為所有答案元組都滿足指定的條件¹。然而，研究未指定的屬性特別棘手，因為我們需要確定使用者對這些未指定屬性的偏好。在本文中，我們提出元組的排序函數取決於兩個因素：(a) 一個捕捉未指定屬性值全域重要性的全域分數，以及 (b) 一個條件分數，該分數捕獲指定和未指定屬性值之間的依賴性（或相關性）強度。例如，對於查詢「城市 = 西雅圖和景觀 = 水岸」，同時位於「學區 = 優秀」的房屋排名較高，因為好的學區在全球範圍內都是可取的。一棟同時擁有「船塢 = 是」的房屋排名也較高，因為想要水岸的人很可能也想要一個船塢。雖然這些分數可以借助領域專業知識或使用者反饋來估計，但我們建議透過工作負載和資料分析來自動估計這些分數。例如，過去的工作負載可能會顯示，尋求具有水岸景觀房屋的使用者中有很大一部分也要求有船塢。

The next challenge is how do we translate these basic intuitions into principled and quantitatively describable ranking functions? To achieve this, we develop ranking functions that are based on Probabilistic Information Retrieval (PIR) ranking models. We chose PIR models because we could extend them to model data dependencies and correlations (the critical ingredients of our approach) in a more principled manner than if we had worked with alternate IR ranking models such as the Vector-Space model. We note that correlations are often ignored in IR because they are very difficult to capture in the very high-dimensional and sparsely populated feature spaces of text data, whereas there are often strong correlations between attribute values in relational data (with functional dependencies being extreme cases), which is a much lower-dimensional, more explicitly structured and densely populated space that our ranking functions can effectively work on.

下一個挑戰是我們如何將這些基本直覺轉化為有原則且可量化描述的排序函數？為此，我們開發了基於機率資訊檢索（PIR）排序模型的排序函數。我們選擇 PIR 模型是因為，與我們使用向量空間模型等其他 IR 排序模型相比，我們可以以更有原則的方式擴展它們來模型化資料依賴性和相關性（我們方法的關鍵要素）。我們注意到，在 IR 中，相關性通常被忽略，因為在文字資料的非常高維和稀疏的特徵空間中很難捕捉到它們，而在關聯式資料中，屬性值之間通常存在很強的相關性（函數依賴性是極端情況），這是一個維度低得多、結構更明確、資料更密集的空間，我們的排序函數可以有效地在其中工作。

The architecture of our ranking has a pre-processing component that collects database as well as workload statistics to determine the appropriate ranking function. The extracted ranking function is materialized in an intermediate knowledge representation layer, to be used later by a query processing component for ranking the results of queries. The ranking functions are encoded in the intermediate layer via intuitive, easy-to-understand "atomic" numerical quantities that describe (a) the global importance of a data value in the ranking process, and (b) the strengths of correlations between pairs of values (e.g., "if a user requests tuples containing value y of attribute Y, how likely is she to be also interested in value x of attribute X?"). Although our ranking approach derives these quantities automatically, our architecture allows users and/or domain experts to tune these quantities further, thereby customizing the ranking functions for different applications.

我們的排序架構有一個預處理元件，它收集資料庫和工作負載統計資料以確定適當的排序函數。提取的排序函數被具體化在一個中間知識表示層中，供以後的查詢處理元件用於對查詢結果進行排序。排序函數在中間層中透過直觀、易於理解的「原子」數值量進行編碼，這些數值量描述了 (a) 資料值在排序過程中的全域重要性，以及 (b) 值對之間相關性的強度（例如，「如果使用者請求包含屬性 Y 的值 y 的元組，她對屬性 X 的值 x 也感興趣的可能性有多大？」）。儘管我們的排序方法會自動推導出這些數量，但我們的架構允許使用者和/或領域專家進一步調整這些數量，從而為不同的應用程式客製化排序函數。

We report on a comprehensive set of experimental results. We first demonstrate through user studies on real datasets that our rankings are superior in quality to previous efforts on this problem. We also demonstrate the efficiency of our ranking system. Our implementation is especially tricky because our ranking functions are relatively complex, involving dependencies/correlations between data values. We use novel pre-computation techniques which reduce this complex problem to a problem efficiently solvable using Top-K algorithms.

我們報告了一系列全面的實驗結果。我們首先透過對真實資料集的使用者研究證明，我們的排序在品質上優於先前對此問題的努力。我們還展示了我們排序系統的效率。我們的實作特別棘手，因為我們的排序函數相對複雜，涉及資料值之間的依賴/相關性。我們使用新穎的預計算技術將這個複雜問題簡化為一個可以使用 Top-K 演算法有效解決的問題。

The rest of this paper is organized as follows. In Section 2 we discuss related work. In Section 3 we define the problem and outline the architecture of our solution. In Section 4 we discuss our approach to ranking based on probabilistic models from information retrieval. In Section 5 we describe an efficient implementation of our ranking system. In Section 6 we discuss the results of our experiments, and we conclude in Section 7.

本文的其餘部分安排如下。在第 2 節中，我們討論了相關工作。在第 3 節中，我們定義了問題並概述了我們解決方案的架構。在第 4 節中，我們討論了我們基於資訊檢索機率模型的排序方法。在第 5 節中，我們描述了我們排序系統的高效實作。在第 6 節中，我們討論了我們的實驗結果，並在第 7 節中得出結論。

¹In the case of document retrieval, ranking functions are often based on the frequency of occurrence of query values in documents (term frequency, or TF). However, in the database context, especially in the case of categorical data, TF is irrelevant as tuples either contain or do not contain a query value. Hence ranking functions need to also consider values of unspecified attributes.

¹在文件檢索的情況下，排序函數通常基於查詢詞在文件中出現的頻率（詞頻，或 TF）。然而，在資料庫情境下，特別是對於分類資料，TF 是無關緊要的，因為元組要麼包含查詢值，要麼不包含。因此，排序函數也需要考慮未指定屬性的值。

Permission to copy without fee all or part of this material is granted provided that the copies are not made or distributed for direct commercial advantage, the VLDB copyright notice and the title of the publication and its date appear, and notice is given that copying is by permission of the Very Large Data Base Endowment. To copy otherwise, or to republish, requires a fee and/or special permission from the Endowment
Proceedings of the 30th VLDB Conference,
Toronto, Canada, 2004

允許免費複製本資料的全部或部分內容，前提是複製不用於直接商業利益，且須註明 VLDB 版權聲明、出版物標題及其日期，並聲明複製是經 Very Large Data Base Endowment 許可。若要以其他方式複製或重新發布，則需要付費和/或特殊許可。
第 30 屆 VLDB 會議論文集，多倫多，加拿大，2004

## 2. Related Work

## 2. 相關工作

Extracting ranking functions has been extensively investigated in areas outside database research such as Information Retrieval. The vector space model as well as probabilistic information retrieval (PIR) models [4, 28, 29] and statistical language models [14] are very successful in practice. While our approach has been inspired by PIR models, we have adapted and extended them in ways unique to our situation, e.g., by leveraging the structure as well as correlations present in the structured data and the database workload.

提取排序函數已在資料庫研究之外的領域進行了廣泛的研究，例如資訊檢索。向量空間模型以及機率資訊檢索 (PIR) 模型 [4, 28, 29] 和統計語言模型 [14] 在實踐中非常成功。雖然我們的方法受到 PIR 模型的啟發，但我們以獨特於我們情況的方式對其進行了調整和擴展，例如，透過利用結構化資料和資料庫工作負載中存在的結構和相關性。

In database research, there has been some work on ranked retrieval from a database. The early work of [23] considered vague/imprecise similarity-based querying of databases. The problem of integrating databases and information retrieval systems has been attempted in several works [12, 13, 17, 18]. Information retrieval based approaches have been extended to XML retrieval (e.g., see [8]). The papers [11, 26, 27, 32] employ relevance-feedback techniques for learning similarity in multimedia and relational databases. Keyword-query based retrieval systems over databases have been proposed in [1, 5, 20]. In [21, 24] the authors propose SQL extensions in which users can specify ranking functions via soft constraints in the form of preferences. The distinguishing aspect of our work from the above is that we espouse automatic extraction of PIR-based ranking functions through data and workload statistics.

在資料庫研究中，已經有一些關於排序檢索的工作。 [23] 的早期工作考慮了對資料庫的模糊/不精確的基於相似性的查詢。在幾項工作 [12, 13, 17, 18] 中已經嘗試了整合資料庫和資訊檢索系統的問題。基於資訊檢索的方法已擴展到 XML 檢索（例如，參見 [8]）。論文 [11, 26, 27, 32] 採用相關性反饋技術來學習多媒體和關聯式資料庫中的相似性。已經提出了基於關鍵字查詢的資料庫檢索系統 [1, 5, 20]。在 [21, 24] 中，作者提出了 SQL 擴充，其中使用者可以透過偏好形式的軟約束來指定排序函數。我們工作與上述工作的區別在於，我們主張透過資料和工作負載統計自動提取基於 PIR 的排序函數。

The work most closely related to our paper is [2] which briefly considered the Many-Answers Problem (although its main focus was on the Empty-Answers Problem, which occurs when a query is too selective, resulting in an empty answer set). It too proposed automatic ranking methods that rely on workload as well as data analysis. In contrast, however, the current paper has the following novel strengths: (a) we use more principled probabilistic PIR techniques rather than ad-hoc techniques "loosely based” on the vector-space model, and (b) we take into account dependencies and correlations between data values, whereas [2] only proposed a form of global score for ranking.

與我們論文最相關的工作是[2]，它簡要地考慮了「多答案問題」（雖然其主要焦點是「空答案問題」，當查詢過於選擇性，導致答案集為空時發生）。它也提出了依賴於工作負載和資料分析的自動排序方法。然而，相比之下，本論文具有以下新穎的優勢：(a) 我們使用更有原則的機率性 PIR 技術，而不是「鬆散地基於」向量空間模型的臨時技術，以及 (b) 我們考慮了資料值之間的依賴性和相關性，而[2]只提出了一種全域分數的排序形式。

Ranking is also an important component in collaborative filtering research [7]. These methods require training data using queries as well as their ranked results. In contrast, we require workloads containing queries only.

在協同過濾研究中，排序也是一個重要的元件[7]。這些方法需要使用查詢及其排序結果的訓練資料。相比之下，我們只需要包含查詢的工作負載。

A major concern of this paper is the query processing techniques for supporting ranking. Several techniques have been previously developed in database research for the Top-K problem [8, 9, 15, 16, 31]. We adopt the Threshold Algorithm of [16, 19, 25] for our purposes, and show how novel pre-computation techniques can be used to produce a very efficient implementation of the Many-Answers Problem. In contrast, an efficient implementation for the Many-Answers Problem was left open in [2].

本文的一個主要關注點是支援排序的查詢處理技術。資料庫研究中已經為 Top-K 問題開發了幾種技術 [8, 9, 15, 16, 31]。我們為我們的目的採用了 [16, 19, 25] 中的閾值演算法，並展示了如何使用新穎的預計算技術來產生一個非常高效的「多答案問題」實作。相比之下，「多答案問題」的高效實作在 [2] 中仍是懸而未決的。

## 3. Problem Definition and Architecture

## 3. 問題定義與架構

In this section, we formally define the Many-Answers Problem in ranking database query results, and also outline a general architecture of our solution.

在本節中，我們正式定義了資料庫查詢結果排序中的「多答案問題」，並概述了我們解決方案的總體架構。

### 3.1 Problem Definition

### 3.1 問題定義

We start by defining the simplest problem instance. Consider a database table D with n tuples {t1, ..., tn} over a set of m categorical attributes A = {A1, ..., Am}. Consider a "SELECT * FROM D" query Q with a conjunctive selection condition of the form “WHERE X1=x1 AND ... AND Xs=xs", where each Xi is an attribute from A and xi is a value in its domain. The set of attributes X = {X1, ..., Xs} ⊆ A is known as the set of attributes specified by the query, while the set Y = A – X is known as the set of unspecified attributes. Let S ⊆ {t1, ..., tn} be the answer set of Q. The Many-Answers Problem occurs when the query is not too selective, resulting in a large S.

我們從定義最簡單的問題實例開始。考慮一個包含 n 個元組 {t1, ..., tn} 的資料庫表 D，它建立在 m 個分類屬性 A = {A1, ..., Am} 的集合之上。考慮一個 "SELECT * FROM D" 查詢 Q，其連接選擇條件為「WHERE X1=x1 AND ... AND Xs=xs」，其中每個 Xi 是來自 A 的屬性，而 xi 是其定義域中的一個值。屬性集 X = {X1, ..., Xs} ⊆ A 被稱為查詢指定的屬性集，而集合 Y = A – X 則被稱為未指定的屬性集。令 S ⊆ {t1, ..., tn} 為 Q 的答案集。「多答案問題」發生在查詢選擇性不強，導致 S 很大的情況下。

The above scenario only represents the simplest problem instance. For example, the type of queries described above are fairly restrictive; we refer to them as point queries because they specify single-valued equality conditions on each of the specified attributes. In a more general setting, queries may contain range/IN conditions, and/or Boolean operators other than conjunctions. Likewise, databases may be multi-tabled, may contain a mix of categorical and numeric data, as well as missing or NULL values. While our techniques extend to all these generalizations, in the interest of clarity (and due to lack of space), the main focus of this paper is on ranking the results of conjunctive point queries on a single categorical table (without NULL values).

上述場景僅代表最簡單的問題實例。例如，上述查詢類型相當受限；我們稱之為點查詢，因為它們在每個指定屬性上都指定了單值等式條件。在更一般的情況下，查詢可能包含範圍/IN 條件，和/或除了連接之外的布林運算子。同樣，資料庫可能是多表的，可能包含分類和數字資料的混合，以及缺失值或 NULL 值。雖然我們的技術可以擴展到所有這些概括，但為了清晰起見（並且由於篇幅所限），本文的主要重點是對單個分類表（沒有 NULL 值）上的連接點查詢結果進行排序。

### 3.2 General Architecture of our Approach

### 3.2 我們方法的總體架構

[Image]
Figure 1: Architecture of Ranking System

[圖片]
圖 1：排序系統架構

Figure 1 shows the architecture of our proposed system for enabling ranking of database query results. As mentioned in the introduction, the main components are the preprocessing component, an intermediate knowledge representation layer in which the ranking functions are encoded and materialized, and a query processing component. The modular and generic nature of our system allows for easy customization of the ranking functions for different applications.

圖 1 顯示了我們提出的用於實現資料庫查詢結果排序的系統架構。如緒論中所述，主要元件是預處理元件、一個中間知識表示層（其中編碼並實現了排序函數）以及一個查詢處理元件。我們系統的模組化和通用性使得可以輕鬆地為不同應用客製化排序函數。

In the next section we discuss PIR-based ranking functions for structured data.

在下一節中，我們將討論基於 PIR 的結構化資料排序函數。

## 4. Ranking Functions: Adaptation of PIR Models for Structured Data

## 4. 排序函數：PIR 模型在結構化資料上的應用

In this section we discuss PIR-based ranking functions, and then show how they can be adapted for structured data. We discuss the semantics of the atomic building blocks that are used to encode these ranking functions in the intermediate layer. We also show how these atomic numerical quantities can be estimated from a variety of knowledge sources, such as data and workload statistics, as well as domain knowledge.

在本節中，我們將討論基於 PIR 的排序函數，然後說明如何將其應用於結構化資料。我們將討論用於在中間層中編碼這些排序函數的原子構建塊的語義。我們還將展示如何從各種知識來源（例如資料和工作負載統計，以及領域知識）估計這些原子數值量。

### 4.1 Review of Probabilistic Information Retrieval

### 4.1 機率資訊檢索回顧

Much of the material of this subsection can be found in textbooks on Information Retrieval, such as [4] (see also [28, 29]). We will need the following basic formulas from probability theory:
Bayes' Rule: p(a|b) = p(b|a)p(a)/p(b)
Product Rule: p(a,b|c) = p(a|c)p(b|a,c)

本小節的大部分內容可以在資訊檢索教科書中找到，例如 [4]（另見 [28, 29]）。我們需要以下機率論的基本公式：
貝氏定理：p(a|b) = p(b|a)p(a)/p(b)
乘法法則：p(a,b|c) = p(a|c)p(b|a,c)

Consider a document collection D. For a (fixed) query Q, let R represent the set of relevant documents, and R_bar = D - R be the set of irrelevant documents. In order to rank any document t in D, we need to find the probability of the relevance of t for the query given the text features of t (e.g., the word/term frequencies in t), i.e., p(R|t). More formally, in probabilistic information retrieval, documents are ranked by decreasing order of their odds of relevance, defined as the following score:

考慮一個文件集合 D。對於一個（固定的）查詢 Q，令 R 表示相關文件的集合，而 R_bar = D - R 為不相關文件的集合。為了對 D 中的任何文件 t 進行排序，我們需要找出在給定 t 的文字特徵（例如，t 中的詞/術語頻率）的情況下，t 對於該查詢的相關性機率，即 p(R|t)。更正式地說，在機率資訊檢索中，文件按照其相關性賠率的遞減順序進行排序，其分數定義如下：

Score(t) = p(R|t)/p(R_bar|t) = (p(t|R)p(R)/p(t))/(p(t|R_bar)p(R_bar)/p(t)) = p(t|R)/p(t|R_bar)

The main issue is, how are these probabilities computed, given that R and R_bar are unknown at query time? The usual techniques in IR are to make some simplifying assumptions, such as estimating R through user feedback, approximating R_bar as D (since R is usually small compared to D), and assuming some form of independence between query terms (e.g., the Binary Independence Model).

主要問題在於，在查詢時 R 和 R_bar 是未知的，這些機率是如何計算的？ IR 中的常用技術是進行一些簡化假設，例如透過使用者反饋估計 R，將 R_bar 近似為 D（因為 R 通常遠小於 D），並假設查詢詞之間存在某種形式的獨立性（例如，二元獨立模型）。

In the next subsection we show how we adapt PIR models for structured databases, in particular for conjunctive queries over a single categorical table. Our approach is more powerful than the Binary Independence Model as we also leverage data dependencies.

在下一小節中，我們將展示如何為結構化資料庫，特別是針對單一分類表上的連接查詢，調整 PIR 模型。由於我們也利用了資料依賴性，我們的方法比二元獨立模型更強大。

### 4.2 Adaptation of PIR Models for Structured Data

### 4.2 將 PIR 模型應用於結構化資料

In our adaptation of PIR models for structured databases, each tuple in a single database table D is effectively treated as a "document". For a (fixed) query Q, our objective is to derive Score(t) for any tuple t, and use this score to rank the tuples. Since we focus on the Many-Answers problem, we only need to concern ourselves with tuples that satisfy the query conditions. Recall the notation from Section 3.1, where X is the set of attributes specified in the query, and Y is the remaining set of unspecified attributes. We denote any tuple t as partitioned into two parts, t(X) and t(Y), where t(X) is the subset of values corresponding to the attributes in X, and t(Y) is the remaining subset of values corresponding to the attributes in Y. Often, when the tuple t is clear from the context, we overload notation and simply write t as consisting of two parts, X and Y (in this context, X and Y are thus sets of values rather than sets of attributes).

在我們將 PIR 模型應用於結構化資料庫的過程中，單個資料庫表 D 中的每個元組實際上都被視為一個「文件」。對於一個（固定的）查詢 Q，我們的目標是推導出任何元組 t 的分數 Score(t)，並用此分數對元組進行排序。由於我們專注於多答案問題，我們只需要關心滿足查詢條件的元組。回顧第 3.1 節的符號，其中 X 是查詢中指定的屬性集，Y 是剩餘的未指定屬性集。我們將任何元組 t 表示為分為兩部分，t(X) 和 t(Y)，其中 t(X) 是對應於 X 中屬性的值子集，而 t(Y) 是對應於 Y 中屬性的剩餘值子集。通常，當元組 t 在上下文中清晰時，我們會重載符號，簡單地將 t 寫成由 X 和 Y 兩部分組成（在此上下文中，X 和 Y 因此是值的集合而不是屬性的集合）。

Replacing t with X and Y (and R_bar as D as mentioned in Section 4.1 is commonly done in IR), we get

用 X 和 Y 替換 t（並且如第 4.1 節所述，在 IR 中通常將 R_bar 視為 D），我們得到

Score(t) ∝ p(t|R)/p(t|D) = p(X,Y|R)/p(X,Y|D) = (p(X|R)p(Y|X,R))/(p(X|D)p(Y|X,D))

Since for the Many-Answers problem we are only interested in ranking tuples that satisfy the query conditions, and all such tuples have the same X values, we can treat any quantity not involving Y as a constant. We thus get

由於對於「多答案」問題，我們只對滿足查詢條件的元組進行排序感興趣，而所有這些元組都具有相同的 X 值，因此我們可以將任何不涉及 Y 的量視為常數。因此我們得到

Score(t) ∝ p(Y|X,R)/p(Y|X,D)

Furthermore, the relevant set R for the Many-Answers problem is a subset of all tuples that satisfy the query conditions. One way to understand this is to imagine that R is the "ideal" set of tuples the user had in mind, but who only managed to partially specify it when preparing the query. Consequently the numerator p(Y|X,R) may be replaced by p(Y|R). We thus get

此外，對於「多答案」問題，相關集合 R 是所有滿足查詢條件的元組的子集。理解這一點的一種方法是，想像 R 是使用者心目中「理想」的元組集合，但在準備查詢時只設法部分地指定了它。因此，分子 p(Y|X,R) 可以被 p(Y|R) 取代。我們因此得到

Score(t) ∝ p(Y|R)/p(Y|X,D)   (1)

We are not quite finished with our derivation of Score(t) yet, but let us illustrate Equation 1 with an example. Consider a query with condition “City=Kirkland and Price=High" (Kirkland is an upper class suburb of Seattle close to a lake). Such buyers may also ideally desire homes with waterfront or greenbelt views, but homes with views looking out into streets may be somewhat less desirable. Thus, p(View=Greenbelt|R) and p(View=Waterfront|R) may both be high, but p(View=Street|R) may be relatively low. Furthermore, if in general there is an abundance of selected homes with greenbelt views as compared to waterfront views, (i.e., the denominator p(View=Greenbelt | City=Kirkland, Price=High, D) is larger than p(View=Waterfront | City=Kirkland, Price=High, D)), our final rankings would be homes with waterfront views, followed by homes with greenbelt views, followed by homes with street views. Note that for simplicity, we have ignored the remaining unspecified attributes in this example.

我們對 Score(t) 的推導尚未完全完成，但讓我們用一個例子來說明方程式 1。考慮一個條件為「城市=柯克蘭且價格=高」的查詢（柯克蘭是西雅圖附近一個靠近湖泊的高檔郊區）。這樣的買家可能也理想地希望擁有水岸或綠地景觀的房屋，但面向街道景觀的房屋可能吸引力稍差。因此，p(景觀=綠地|R) 和 p(景觀=水岸|R) 可能都很高，但 p(景觀=街道|R) 可能相對較低。此外，如果總體上，與水岸景觀相比，所選房屋中有大量的綠地景觀，（即，分母 p(景觀=綠地 | 城市=柯克蘭，價格=高, D) 大於 p(景觀=水岸 | 城市=柯克蘭，價格=高, D)），我們的最終排名將是水岸景觀的房屋，其次是綠地景觀的房屋，再其次是街道景觀的房屋。請注意，為簡單起見，我們在此範例中忽略了其餘未指定的屬性。

### 4.2.1 Limited Independence Assumptions

### 4.2.1 有限獨立性假設

One possible way of continuing the derivation of Score(t) would be to make independence assumptions between values of different attributes, like in the Binary Independence Model in IR. However, while this is reasonable with text data (because estimating model parameters like the conditional probabilities p(Y|X) poses major accuracy and efficiency problems with sparse and high-dimensional data such as text), we have earlier argued that with structured data, dependencies between data values can be better captured and would more significantly impact the result ranking. An extreme alternative to making sweeping independence assumptions would be to construct comprehensive dependency models of the data (e.g. probabilistic graphical models such as Markov Random Fields or Bayesian Networks [30]), and derive ranking functions based on these models. However, our preliminary investigations suggested that such approaches, particularly for large datasets, have unacceptable pre-processing and query processing costs.

繼續推導 Score(t) 的一種可能方法是在不同屬性的值之間做出獨立性假設，就像在 IR 的二元獨立模型中一樣。然而，雖然這對於文字資料是合理的（因為估計像條件機率 p(Y|X) 這樣的模型參數對於稀疏和高維資料（如文字）會帶來重大的準確性和效率問題），但我們之前已經論證，對於結構化資料，資料值之間的依賴關係可以被更好地捕獲，並且會更顯著地影響結果排名。與做出全面獨立性假設的極端替代方案是構建資料的綜合依賴性模型（例如，機率圖形模型，如馬可夫隨機場或貝葉斯網路[30]），並基於這些模型推導排序函數。然而，我們的初步調查表明，這種方法，特別是對於大型資料集，具有不可接受的預處理和查詢處理成本。

Consequently, in this paper we espouse an approach that strikes a middle ground. We only make limited forms of independence assumptions – given a query Q and a tuple t, the X (and Y) values within themselves are assumed to be independent, though dependencies between the X and Y values are allowed. More precisely, we assume limited conditional independence, i.e., p(X|C) (resp. p(Y|C)) may be written as Π_{x∈X}p(x|C) (resp. Π_{y∈Y}p(y|C)) where C is any condition that only involves Y values (resp. X values), R, or D.

因此，在本文中，我們主張一種折衷的方法。我們只做有限形式的獨立性假設——給定一個查詢 Q 和一個元組 t，X（和 Y）值本身被假定為獨立的，儘管允許 X 和 Y 值之間的依賴性。更準確地說，我們假設有限的條件獨立性，即 p(X|C)（或 p(Y|C)）可以寫成 Π_{x∈X}p(x|C)（或 Π_{y∈Y}p(y|C)），其中 C 是僅涉及 Y 值（或 X 值）、R 或 D 的任何條件。

While this assumption is patently false in many cases (for instance, in the example in Section 4.2 this assumes that there is no dependency between homes in Kirkland and high-priced homes), nevertheless the remaining dependencies that we do leverage, i.e., between the specified and unspecified values, prove to be significant for ranking. Moreover, as we shall show in Section 5, the resulting simplified functional form of the ranking function enables the efficient adaptation of known Top-K algorithms through novel data structuring techniques.

雖然這個假設在許多情況下顯然是錯誤的（例如，在 4.2 節的例子中，這假設柯克蘭的房屋和高價房屋之間沒有依賴關係），但我們確實利用的剩餘依賴關係，即指定值和未指定值之間的依賴關係，對排序來說是顯著的。此外，正如我們將在第 5 節中展示的，排序函數由此產生的簡化函數形式，能夠透過新穎的資料結構技術有效地適應已知的 Top-K 演算法。

We continue the derivation of the score of a tuple under the above assumptions:

在上述假設下，我們繼續推導元組的分數：
```
Score(t) ∝ Π_{y∈Y} (p(y|R) / p(y|X,D))
         = Π_{y∈Y} (p(y|R) / (p(X,D|y)p(y)/p(X,D)))
         = Π_{y∈Y} (p(y|R) / (p(D|y)p(X|y,D)p(y)/p(D)))
         = Π_{y∈Y} p(y|R) * Π_{y∈Y} (1 / (p(D|y)p(y) * Π_{x∈X}p(x|y,D))) * p(D)
This simplifies to
Score(t) ∝ Π_{y∈Y} p(y|R) * Π_{x∈X} (1 / p(x|y,D))
```
Although Equation 2 represents a simplification over Equation 1, it is still not directly computable, as R is unknown. We discuss how to estimate the quantities p(y|R) next.

儘管方程式（2）相對於方程式（1）是一個簡化，但它仍然無法直接計算，因為 R 是未知的。接下來我們討論如何估計數量 p(y|R)。

### 4.2.2 Workload-Based Estimation of p(y|R)

### 4.2.2 基於工作負載的 p(y|R) 估計

Estimating the quantities p(y|R) requires knowledge of R, which is unknown at query time. The usual technique for estimating R in IR is through user feedback (relevance feedback) at query time, or through other forms of training. In our case, we provide an automated approach that leverages available workload information for estimating p(y|R).

估計 p(y|R) 這個量需要知道 R，但 R 在查詢時是未知的。在 IR 中估計 R 的常用技術是通過查詢時的使用者反饋（相關性反饋），或通過其他形式的訓練。在我們的案例中，我們提供了一種自動化方法，利用可用的工作負載信息來估計 p(y|R)。

We assume that we have at our disposal a workload W, i.e., a collection of ranking queries that have been executed on our system in the past. We first provide some intuition of how we intend to use the workload in ranking. Consider the example in Section 4.2 where a user has requested for high-priced homes in Kirkland. The workload may perhaps reveal that, in the past a large fraction of users that had requested for high-priced homes in Kirkland had also requested for waterfront views. Thus for such users, it is desirable to rank homes with waterfront views over homes without such views.

我們假設我們有一個工作負載 W，即過去在我們系統上執行過的一系列排序查詢。我們首先提供一些關於我們打算如何在排序中使用工作負載的直覺。考慮 4.2 節中的例子，使用者請求了柯克蘭的高價住宅。工作負載或許會揭示，過去請求柯克蘭高價住宅的使用者中，有很大一部分也請求了水岸景觀。因此，對於這些使用者，將帶有水岸景觀的住宅排在沒有此類景觀的住宅之上是可取的。

We note that this dependency information may not be derivable from the data alone, as a majority of such homes may not have waterfront views (i.e., data dependencies do not indicate user preferences as workload dependencies do). Of course, the other option is for a domain expert (or even the user) to provide this information (and in fact, as we shall discuss later, our ranking architecture is generic enough to allow further customization by human experts).

我們注意到，這種依賴信息可能無法僅從資料中推導出來，因為大多數此類房屋可能沒有水岸景觀（即，資料依賴性並不像工作負載依賴性那樣指示使用者偏好）。當然，另一種選擇是讓領域專家（甚至使用者）提供此信息（事實上，正如我們稍後將討論的，我們的排序架構足夠通用，允許人類專家進一步客製化）。

More generally, the workload W is represented as a set of "tuples", where each tuple represents a query and is a vector containing the corresponding values of the specified attributes. Consider an incoming query Q which specifies a set X of attribute values. We approximate R as all query "tuples" in W that also request for X. This approximation is novel to this paper, i.e., that all properties of the set of relevant tuples R can be obtained by only examining the subset of the workload that contains queries that also request for X. So for a query such as "City=Kirkland and Price=High", we look at the workload in determining what such users have also requested for often in the past.

更廣泛地說，工作負載 W 被表示為一組「元組」，其中每個元組代表一個查詢，並且是一個包含指定屬性相應值的向量。考慮一個傳入的查詢 Q，它指定了一組屬性值 X。我們將 R 近似為 W 中所有也請求 X 的查詢「元組」。這種近似是本文的新穎之處，即相關元組集 R 的所有屬性都可以通過僅檢查包含也請求 X 的查詢的工作負載子集來獲得。因此，對於像「城市=柯克蘭和價格=高」這樣的查詢，我們會查看工作負載以確定這些使用者過去還經常請求什麼。

We can thus write, for query Q, with specified attribute set X, p(y|R) as p(y|X,W). Making this substitution in Equation 2, we get

因此，對於帶有指定屬性集 X 的查詢 Q，我們可以將 p(y|R) 寫成 p(y|X,W)。將此代入方程 2，我們得到
```
Score(t) ∝ Π_{y∈Y} (p(y|X,W) / p(y|D)) * Π_{x∈X} (1 / p(x|y,D))
         = Π_{y∈Y} ( (p(X,W|y)p(y)/p(X,W)) / p(y|D) ) * Π_{x∈X} (1 / p(x|y,D))
         = Π_{y∈Y} p(W|y)p(X|W,y)p(y) * Π_{x∈X} (1 / p(x|y,D))
         ∝ Π_{y∈Y} p(y|W) * Π_{x∈X} p(x|y,W) / (p(y|D) * Π_{x∈X} p(x|y,D))
This can be finally rewritten as:
Score(t) ∝ Π_{y∈Y} (p(y|W) / p(y|D)) * Π_{y∈Y} Π_{x∈X} (p(x|y,W) / p(x|y,D)) (3)
```

Equation 3 is the final ranking formula that we use in the rest of this paper. Note that unlike Equation 2, we have effectively eliminated R from the formula, and are only left with having to compute quantities such as p(y|W), p(y|D), p(x|y,W), and p(x|y,D). In fact, these are the "atomic" numerical quantities referred to at various places earlier in the paper.

方程式 3 是我們在本文其餘部分使用的最終排序公式。請注意，與方程式 2 不同，我們已有效地從公式中消除了 R，只剩下計算諸如 p(y|W)、p(y|D)、p(x|y,W) 和 p(x|y,D) 之類的量。事實上，這些是本文前面多處提到的「原子」數值量。

Also note that the score in Equation 3 is composed of two large factors. The first factor may be considered as the global part of the score, while the second factor may be considered as the conditional part of the score. Thus, in the example in Section 4.2, the first part measures the global importance of unspecified values such as waterfront, greenbelt and street views, while the second part measures the dependencies between these values and specified values “City=Kirkland” and “Price=High".

還要注意，方程式 3 中的分數由兩個主要因素組成。第一個因素可以被視為分數的全域部分，而第二個因素可以被視為分數的條件部分。因此，在 4.2 節的例子中，第一部分衡量了諸如水岸、綠地和街景等未指定值的全域重要性，而第二部分則衡量了這些值與指定值「城市=柯克蘭」和「價格=高」之間的依賴關係。

### 4.3 Computing the Atomic Probabilities

### 4.3 計算原子機率

Our strategy is to pre-compute each of the atomic quantities for all distinct values in the database. The quantities p(y|W) and p(y|D) are simply the relative frequencies of each distinct value y in the workload and database, respectively (the latter is similar to IDF, or the inverse document frequency concept in IR), while the quantities p(x|y,W) and p(x|y,D) may be estimated by computing the confidences of pair-wise association rules [3] in the workload and database, respectively. Once this pre-computation has been completed, we store these quantities as auxiliary tables in the intermediate knowledge representation layer. At query time, the necessary quantities may be retrieved and appropriately composed for performing the rankings. Further details of the implementation are discussed in Section 5.

我們的策略是預先計算資料庫中所有不同值的每個原子量。數量 p(y|W) 和 p(y|D) 分別是工作負載和資料庫中每個不同值 y 的相對頻率（後者類似於 IR 中的 IDF，或逆向文件頻率概念），而數量 p(x|y,W) 和 p(x|y,D) 則可以透過分別計算工作負載和資料庫中成對關聯規則 [3] 的置信度來估計。一旦這個預計算完成，我們將這些數量作為輔助表儲存在中間知識表示層中。在查詢時，可以檢索必要的數量並適當地組合成績。實作的更多細節將在第 5 節中討論。

While the above is an automated approach based on workload analysis, it is possible that sometimes the workload may be insufficient and/or unreliable. In such instances, it may be necessary for domain experts to be able to tune the ranking function to make it more suitable for the application at hand.

雖然以上是基於工作負載分析的自動化方法，但有時工作負載可能不足和/或不可靠。在這種情況下，可能需要領域專家來調整排序函數，使其更適合手頭的應用程式。

[Image]
Figure 2: Detailed Architecture of Ranking System

[圖片]
圖 2：排序系統的詳細架構

## 5. Implementation

## 5. 實作

In this section we discuss the implementation of our database ranking system. Figure 2 shows the detailed architecture, including the pre-processing and query processing components as well as their sub-modules. We discuss several novel data structures and algorithms that were necessary for good performance of our system.

在本節中，我們將討論我們的資料庫排序系統的實作。圖 2 展示了詳細的架構，包括預處理和查詢處理元件及其子模組。我們討論了幾個為我們系統的良好性能所必需的新穎的資料結構和演算法。

### 5.1 Pre-Processing

### 5.1 預處理

This component is divided into several modules. First, the Atomic Probabilities Module computes the quantities p(y|W), p(y|D), p(x|y,W), and p(x|y,D) for all distinct values x and y. These quantities are computed by scanning the workload and data, respectively (while the latter two quantities can be computed by running a general association rule mining algorithm such as [3] on the workload and data, we instead chose to directly compute all pair-wise co-occurrence frequencies by a single scan of the workload and data respectively). The observed probabilities are then smoothened using the Bayesian m-estimate method [10].

該元件分為幾個模組。首先，原子機率模組計算所有不同值 x 和 y 的數量 p(y|W)、p(y|D)、p(x|y,W) 和 p(x|y,D)。這些數量分別透過掃描工作負載和資料來計算（而後兩個數量可以透過在工作負載和資料上運行通用關聯規則挖掘演算法（例如[3]）來計算，我們選擇直接透過單次掃描工作負載和資料來計算所有成對的共現頻率）。然後使用貝葉斯 m 估計方法 [10] 對觀察到的機率進行平滑處理。

These atomic probabilities are stored as database tables in the intermediate knowledge representation layer, with appropriate indexes to enable easy retrieval. In particular, p(y|W) and p(y|D) are respectively stored in two tables, each with columns {AttName, AttVal, Prob} and with a composite B+ tree index on (AttName, AttVal), while p(x|y,W) and p(x|y,D) are respectively stored in two tables, each with columns {AttNameLeft, AttValLeft, AttNameRight, AttValRight, Prob} and with a composite B+ tree index on (AttNameLeft, AttValLeft, AttNameRight, AttValRight). These atomic quantities can be further customized by human experts if necessary.

這些原子機率作為資料庫表儲存在中間知識表示層中，並帶有適當的索引以便於檢索。具體來說，p(y|W) 和 p(y|D) 分別儲存在兩個表中，每個表都有 {AttName, AttVal, Prob} 欄位，並在 (AttName, AttVal) 上有一個複合 B+ 樹索引，而 p(x|y,W) 和 p(x|y,D) 分別儲存在兩個表中，每個表都有 {AttNameLeft, AttValLeft, AttNameRight, AttValRight, Prob} 欄位，並在 (AttNameLeft, AttValLeft, AttNameRight, AttValRight) 上有一個複合 B+ 樹索引。如有必要，這些原子量可以由人類專家進一步客製化。

This intermediate layer now contains enough information for computing the ranking function, and a naïve query processing algorithm (henceforth referred to as the Scan algorithm) can indeed be designed, which, for any query, first selects the tuples that satisfy the query condition, then scans and computes the score for each such tuple using the information in this intermediate layer, and finally returns the Top-K tuples. However, such an approach can be inefficient for the Many-Answers problem, since the number of tuples satisfying the query an be very large. At the other extreme, we could pre-compute the Top-K tuples for all possible queries (i.e., for all possible sets of values X), and at query time, simply return the appropriate result set. Of course, due to the combinatorial explosion, this is infeasible in practice. We thus pose the question: how can we appropriately trade off between pre-processing and query processing, i.e., what additional yet reasonable pre-computations are possible that can enable faster query-processing algorithms than Scan?

這個中間層現在包含足夠的資訊來計算排序函數，並且確實可以設計一個簡單的查詢處理演算法（以下稱為掃描演算法），該演算法對於任何查詢，首先選擇滿足查詢條件的元組，然後使用該中間層中的資訊掃描並計算每個此類元組的分數，最後返回 Top-K 元組。然而，對於「多答案」問題，這種方法可能效率低下，因為滿足查詢條件的元組數量可能非常大。在另一個極端，我們可以為所有可能的查詢（即，對於所有可能的值集 X）預先計算 Top-K 元組，並在查詢時簡單地返回相應的結果集。當然，由於組合爆炸，這在實踐中是不可行的。因此，我們提出以下問題：我們如何在預處理和查詢處理之間進行適當的權衡，即，還有哪些額外但合理的預計算可以實現比掃描更快的查詢處理演算法？

The high-level intuition behind our approach to the above problem is as follows. Instead of pre-computing the Top-K tuples for all possible queries, we pre-compute ranked lists of the tuples for all possible “atomic” queries – each distinct value x in the table defines an atomic query Qx that specifies the single value {x}. Then at query time, given an actual query that specifies a set of values X, we "merge" the ranked lists corresponding to each x in X to compute the final Top-K tuples.

我們解決上述問題的方法的高層直覺如下。我們不是為所有可能的查詢預先計算 Top-K 元組，而是為所有可能的「原子」查詢預先計算元組的排序列表——表中的每個不同值 x 都定義了一個原子查詢 Qx，它指定了單個值 {x}。然後在查詢時，給定一個實際指定一組值 X 的查詢，我們「合併」與 X 中每個 x 對應的排序列表以計算最終的 Top-K 元組。

Of course, for this high-level idea to work, the main challenge is to be able to perform the merging without having to scan any of the ranked lists in its entirety. One idea would be to try and adapt well-known Top-K algorithms such as the Threshold Algorithm (TA) and its derivatives [9, 15, 16, 19, 25] for this problem. However, it is not immediately obvious how a feasible adaptation can be easily accomplished. For example, it is especially critical to keep the number of sorted streams (an access mechanism required by TA) small, as it is well-known that TA's performance rapidly deteriorates as this number increases. Upon examination of our ranking function in Equation 3 (which involves all attribute values of the tuple, and not just the specified values), the number of sorted streams in any naïve adaptation of TA would depend on the total number of attributes in the database, which would cause major performance problems.

當然，要使這個高層想法奏效，主要的挑戰是能夠在不必完整掃描任何排序列表的情況下執行合併。一個想法是嘗試改編著名的 Top-K 演算法，例如閾值演算法 (TA) 及其衍生演算法 [9, 15, 16, 19, 25] 來解決這個問題。然而，如何輕鬆實現可行的改編並非一目了然。例如，保持排序流的數量（TA 所需的存取機制）較小尤為關鍵，因為眾所周知，隨著這個數量的增加，TA 的性能會迅速惡化。在檢查我們在方程 3 中的排序函數（它涉及元組的所有屬性值，而不僅僅是指定的值）後，任何對 TA 的天真改編中的排序流數量都將取決於資料庫中的屬性總數，這將導致嚴重的性能問題。

In what follows, we show how to pre-compute data structures that indeed enable us to efficiently adapt TA for our problem. At query time we do a TA-like merging of several ranked lists (i.e. sorted streams). However, the required number of sorted streams depends only on s and not on m (s is the number of specified attribute values in the query while m is the total number of attributes in the database, see Section 3.1). We emphasize that such a merge operation is only made possible due to the specific functional form of our ranking function resulting from our limited independence assumptions as discussed in Section 4.2.1. It is unlikely that TA can be adapted, at least in a feasible manner, for ranking functions that rely on more comprehensive dependency models of the data.

接下來，我們將展示如何預先計算資料結構，這些結構確實使我們能夠有效地為我們的問題調整 TA。在查詢時，我們對幾個已排序的列表（即排序流）進行類似 TA 的合併。然而，所需的排序流數量僅取決於 s 而非 m（s 是查詢中指定的屬性值數量，而 m 是資料庫中的屬性總數，請參見第 3.1 節）。我們強調，這樣的合併操作之所以成為可能，僅僅是因為我們在第 4.2.1 節中討論的有限獨立性假設所產生的排序函數的特定函數形式。對於依賴更全面依賴性模型的排序函數，TA 不太可能以可行的方式進行調整。

We next give the details of these data structures. They are pre-computed by the Index Module of the pre-processing component. This module (see Figure 3 for the algorithm) takes as inputs the association rules and the database, and for every distinct value x, creates two lists Cx and Gx, each containing the tuple-ids of all data tuples that contain x, ordered in specific ways. These two lists are defined as follows:

我們接下來給出這些資料結構的細節。它們由預處理元件的索引模組預先計算。該模組（見圖 3 的演算法）以關聯規則和資料庫為輸入，並為每個不同的值 x 創建兩個列表 Cx 和 Gx，每個列表包含所有包含 x 的資料元組的元組 ID，並以特定的方式排序。這兩個列表定義如下：

1. Conditional List Cx: This list consists of pairs of the form <TID, CondScore>, ordered by descending CondScore, where TID is the tuple-id of a tuple t that contains x and CondScore = Π_{z∈t} (p(x|z,W)/p(x|z,D)) where z ranges over all attribute values of t.
2. Global List Gx: This list consists of pairs of the form <TID, GlobScore>, ordered by descending GlobScore, where TID is the tuple-id of a tuple t that contains x and GlobScore = Π_{z∈t} (p(z|W)/p(z|D))

1. 條件列表 Cx：此列表由 <TID, CondScore> 形式的對組成，按 CondScore 降序排列，其中 TID 是包含 x 的元組 t 的元組 ID，且 CondScore = Π_{z∈t} (p(x|z,W)/p(x|z,D))，其中 z 遍歷 t 的所有屬性值。
2. 全域列表 Gx：此列表由 <TID, GlobScore> 形式的對組成，按 GlobScore 降序排列，其中 TID 是包含 x 的元組 t 的元組 ID，且 GlobScore = Π_{z∈t} (p(z|W)/p(z|D))

These lists enable efficient computation of the score of a tuple t for any query as follows: given a query Q specifying conditions for a set of attribute values, say X = {x1, ..., xs}, at query time we retrieve and multiply the scores of t in the lists Cx1, ..., Cxs and in one of Gx1, ..., Gxs. This requires only s+1 multiplications and results in a score² that is proportional to the actual score. Clearly this is more efficient than computing the score "from scratch" by retrieving the relevant atomic probabilities from the intermediate layer and composing them appropriately.

這些列表能夠有效地計算任何查詢中元組 t 的分數，如下所示：給定一個指定屬性值集合 X = {x1, ..., xs} 的查詢 Q，在查詢時，我們檢索並乘以 t 在列表 Cx1, ..., Cxs 和 Gx1, ..., Gxs 中其中一個列表中的分數。這只需要 s+1 次乘法，並得到一個與實際分數成正比的分數²。顯然，這比透過從中間層檢索相關的原子機率並進行組合來「從頭開始」計算分數更有效率。

We need to enable two kinds of access operations efficiently on these lists. First, given a value x, it should be possible to perform a GetNextTID operation on lists Cx and Gx in constant time, i.e., the tuple-ids in the lists should be efficiently retrievable one-by-one in order of decreasing score. This corresponds to the sorted stream access of TA. Second, it should be possible to perform random access on the lists, i.e., given a TID, the corresponding score (CondScore or GlobScore) should be retrievable in constant time. To enable these operations efficiently, we materialize these lists as database tables – all the conditional lists are maintained in one table called CondList (with columns {AttName, AttVal, TID, CondScore}) while all the global lists are maintained in another table called GlobList (with columns {AttName, AttVal, TID, GlobScore}). The tables have composite B+ tree indices on (AttName, AttVal, CondScore) and (AttName, AttVal, GlobScore) respectively. This enables efficient performance of both access operations. Further details of how these data structures and their access methods are used in query processing are discussed in Section 5.2.

我們需要有效地啟用對這些列表的兩種訪問操作。首先，給定一個值 x，應該可以在常數時間內對列表 Cx 和 Gx 執行 GetNextTID 操作，即列表中的元組 ID 應該可以按分數遞減的順序逐個有效地檢索。這對應於 TA 的排序流訪問。其次，應該可以對列表進行隨機訪問，即給定一個 TID，相應的分數（CondScore 或 GlobScore）應該可以在常數時間內檢索。為了有效地啟用這些操作，我們將這些列表實現為資料庫表——所有條件列表都保存在一個名為 CondList 的表中（列為 {AttName, AttVal, TID, CondScore}），而所有全局列表都保存在另一個名為 GlobList 的表中（列為 {AttName, AttVal, TID, GlobScore}）。這些表在 (AttName, AttVal, CondScore) 和 (AttName, AttVal, GlobScore) 上分別具有複合 B+ 樹索引。這使得兩種訪問操作都能高效執行。關於這些資料結構及其訪問方法如何在查詢處理中使用的更多細節將在 5.2 節中討論。

```
Index Module
Input: Data table, atomic probabilities tables
Output: Conditional and global lists
FOR EACH distinct value x of database DO
  Cx = Gx = {}
  FOR EACH tuple t containing x with tuple-id = TID DO
    CondScore = Π_{z∈t} (p(x|z,W)/p(x|z,D))
    Add <TID, CondScore> to Cx
    GlobScore = Π_{z∈t} (p(z|W)/p(z|D))
    Add <TID, GlobScore> to Gx
  END FOR
  Sort Cx and Gx by decreasing CondScore and GlobScore resp.
END FOR
```

[Image]
Figure 3: The Index Module

[圖片]
圖 3：索引模組

²This score is proportional, but not equal, to the actual score because it contains extra factors of the form p(x|z,W)/p(x|z,D) where z∈X. However, these extra factors are common to all selected tuples, hence the rank order is unchanged.

²此分數與實際分數成正比，但不相等，因為它包含形式為 p(x|z,W)/p(x|z,D) 的額外因子，其中 z∈X。然而，這些額外因子對於所有選定的元組都是共同的，因此排序順序不變。

### 5.2 Query Processing Component

### 5.2 查詢處理元件

In this subsection we describe the query processing component. The naïve Scan algorithm has already been described in Section 5.1, so our focus here is on the alternate List Merge algorithm (see Figure 4). This is an adaptation of TA, whose efficiency crucially depends on the data structures pre-computed by the Index Module.

在本小節中，我們將描述查詢處理元件。簡單的掃描演算法已在 5.1 節中描述，因此我們這裡的重點是備選的列表合併演算法（見圖 4）。這是 TA 的一種改編，其效率關鍵取決於索引模組預先計算的資料結構。

The List Merge algorithm operates as follows. Given a query Q specifying conditions for a set X = {x1,..,xs} of attributes, we execute TA on the following s+1 lists: Cx1,...,Cxs, and Gxb, where Gxb is the shortest list among Gx1,...,Gxs (in principle, any list from Gx1,...,Gxs would do, but the shortest list is likely to be more efficient). During each iteration, the TID with the next largest score is retrieved from each list using sorted access. Its score in every other list is retrieved via random access, and all these retrieved scores are multiplied together, resulting in the final score of the tuple (which, as mentioned in Section 5.1, is proportional to the actual score derived in Equation 3). The termination criterion guarantees that no more GetNextTID operations will be needed on any of the lists. This is accomplished by maintaining an array T which contains the last scores read from all the lists at any point in time by GetNextTID operations. The product of the scores in T represents the score of the very best tuple we can hope to find in the data that is yet to be seen. If this value is no more than the tuple in the Top-K buffer with the smallest score, the algorithm successfully terminates.

列表合併演算法的操作如下。給定一個查詢 Q，它為一組屬性 X = {x1,...,xs} 指定條件，我們在以下 s+1 個列表上執行 TA：Cx1,...,Cxs 和 Gxb，其中 Gxb 是 Gx1,...,Gxs 中最短的列表（原則上，Gx1,...,Gxs 中的任何列表都可以，但最短的列表可能更有效率）。在每次迭代期間，使用排序存取從每個列表中檢索具有下一個最大分數的 TID。它在其他每個列表中的分數是透過隨機存取檢索的，所有這些檢索到的分數相乘，得到元組的最終分數（如 5.1 節所述，它與方程式 3 中導出的實際分數成正比）。終止標準保證在任何列表上都不再需要 GetNextTID 操作。這是透過維護一個陣列 T 來實現的，該陣列包含在任何時間點透過 GetNextTID 操作從所有列表中讀取的最後一個分數。T 中分數的乘積代表了我們希望在尚未看到的資料中找到的最好元組的分數。如果此值不大於 Top-K 緩衝區中分數最小的元組，則演算法成功終止。

```
List Merge Algorithm
Input: Query, data table, global and conditional lists
Output: Top-K tuples
Let Gxb be the shortest list among Gx1,...,Gxs
Let B = {} be a buffer that can hold K tuples ordered by score
Let T be an array of size s+1 storing the last score from each list
Initialize B to empty
REPEAT
  FOR EACH list L in Cx1,..., Cxs, and Gxb DO
    TID = GetNextTID(L)
    Update T with score of TID in L
    Get score of TID from other lists via random access
    IF all lists contain TID THEN
      Compute Score(TID) by multiplying retrieved scores
      Insert <TID, Score(TID)> in the correct position in B
    END IF
  END FOR
UNTIL B[K].Score ≥ Π_{i=1}^{s+1} T[i]
RETURN B
```

[Image]
Figure 4: The List Merge Algorithm

[圖片]
圖 4：列表合併演算法

### 5.2.1 Limited Available Space

### 5.2.1 有限的可用空間

So far we have assumed that there is enough space available to build the conditional and global lists. A simple analysis indicates that the space consumed by these lists is O(mn) bytes (m is the number of attributes and n the number of tuples of the database table). However, there may be applications where space is an expensive resource (e.g., when lists should preferably be held in memory and compete for that space or even for space in the processor cache hierarchy). We show that in such cases, we can store only a subset of the lists at pre-processing time, at the expense of an increase in the query processing time.

到目前為止，我們一直假設有足夠的空間來構建條件列表和全域列表。一個簡單的分析表明，這些列表消耗的空間為 O(mn) 位元組（m 是屬性的數量，n 是資料庫表的元組數量）。然而，在某些應用中，空間可能是一種昂貴的資源（例如，當列表最好保存在記憶體中並競爭該空間，甚至競爭處理器快取層次結構中的空間時）。我們表明，在這種情況下，我們可以在預處理時僅儲存列表的子集，代價是查詢處理時間的增加。

Determining which lists to retain/omit at pre-processing time may be accomplished by analyzing the workload. A simple solution is to store the conditional lists Cx and the corresponding global lists Gx only for those attribute values x that occur most frequently in the workload. At query time, since the lists of some of the specified attributes may be missing, the intuitive idea is to probe the intermediate knowledge representation layer (where the "relatively raw" data is maintained, i.e., the atomic probabilities) and directly compute the missing information. More specifically, we use a modification of TA described in [9], where not all sources have sorted stream access.

在預處理時確定要保留/省略哪些列表可以透過分析工作負載來完成。一個簡單的解決方案是僅為工作負載中最常出現的那些屬性值 x 儲存條件列表 Cx 和相應的全域列表 Gx。在查詢時，由於某些指定屬性的列表可能會丟失，直觀的想法是探測中間知識表示層（維護「相對原始」資料的地方，即原子機率）並直接計算缺失的資訊。更具體地說，我們使用 [9] 中描述的 TA 的一個修改版本，其中並非所有來源都具有排序流存取權限。

[Table]
Figure 5: Sizes of Datasets

[表格]
圖 5：資料集大小

## 6. Experiments

## 6. 實驗

In this section we report on the results of an experimental evaluation of our ranking method as well as some of the competitors. We evaluated both the quality of the rankings obtained, as well as the performance of the various approaches. We mention at the outset that preparing an experimental setup for testing ranking quality was extremely challenging, as unlike IR, there are no standard benchmarks available, and we had to conduct user studies to evaluate the rankings produced by the various algorithms.

在本節中，我們報告了我們的排序方法以及一些競爭方法的實驗評估結果。我們評估了獲得的排序品質以及各種方法的性能。我們一開始就提到，準備一個測試排序品質的實驗裝置極具挑戰性，因為與資訊檢索不同，沒有標準的基準可用，我們必須進行使用者研究來評估各種演算法產生的排序。

For our evaluation, we use real datasets from two different domains. The first domain was the MSN HomeAdvisor database (http://houseandhome.msn.com/), from which we prepared a table of homes for sale in the US, with attributes such as Price, Year, City, Bedrooms, Bathrooms, Sqft, Garage, etc. (we converted numerical attributes into categorical ones by discretizing them into meaningful ranges). The original database table also had a text column called Remarks, which contained descriptive information about the home. From this column, we extracted additional Boolean attributes such as Fireplace, View, Pool, etc. To evaluate the role of the size of the database, we also performed experiments on a subset of the HomeAdvisor database, consisting only of homes sold in the Seattle area.

為了進行評估，我們使用了來自兩個不同領域的真實資料集。第一個領域是 MSN HomeAdvisor 資料庫 (http://houseandhome.msn.com/)，我們從中準備了一個美國待售房屋的表格，屬性包括價格、年份、城市、臥室、浴室、面積、車庫等（我們將數值屬性透過離散化轉換為有意義的範圍，從而變為類別屬性）。原始資料庫表還有一個名為「備註」的文字欄位，其中包含有關房屋的描述性資訊。我們從此欄位中提取了額外的布林屬性，例如壁爐、景觀、游泳池等。為了評估資料庫大小的作用，我們還對 HomeAdvisor 資料庫的一個子集進行了實驗，該子集僅包含在西雅圖地區出售的房屋。

[Table]

Figure 9: Time and Space Consumed by Index Module

圖 9：索引模組消耗的時間和空間

If space is a critical issue, we can adopt the space saving variation of the List Merge algorithm as discussed in Section 5.2.1. We report on this next.

如果空間是一個關鍵問題，我們可以採用 5.2.1 節中討論的 List Merge 演算法的節省空間變體。我們接下來將報告這一點。

Space Saving Variations: In this experiment we show how the performance of the algorithms changes when only a subset of the set of global and conditional lists are stored. Recall from Section 5.2.1 that we only retain lists for the values of the frequently occurring attributes in the workload. For this experiment we consider Top-10 queries with selection conditions that specify two attributes (queries generated by randomly picking a pair of attributes and a domain value for each attribute), and measure their execution times. The compared algorithms are:

節省空間的變體：在這個實驗中，我們展示了當只儲存全域和條件列表集合的一個子集時，演算法性能的變化。回想一下 5.2.1 節，我們只保留工作負載中頻繁出現屬性值的列表。對於這個實驗，我們考慮具有指定兩個屬性的選擇條件的 Top-10 查詢（透過隨機選擇一對屬性和每個屬性的域值生成的查詢），並測量它們的執行時間。比較的演算法是：

* LM: List Merge with all lists available
* LMM: List Merge where lists for one of the two specified attributes are missing, halving space
* Scan

* LM：列表合併，所有列表均可用
* LMM：列表合併，其中兩個指定屬性之一的列表遺失，空間減半
* 掃描

Figure 10 shows the execution times of the queries over the Seattle Homes database as a function of the total number of tuples that satisfy the selection condition. The times are averaged over 10 queries.

圖 10 顯示了在 Seattle Homes 資料庫上查詢的執行時間，作為滿足選擇條件的元組總數的函式。時間是 10 個查詢的平均值。

We first note that LM is extremely fast when compared to the other algorithms (its times are less than one second for each run, consequently its graph is almost along the x-axis). This is to be expected as most of the computations have been accomplished at pre-processing time. The performance of Scan degrades when the total number of selected tuples increases, because the scores of more tuples need to be calculated at runtime. In contrast, the performance of LM and LMM actually improves slightly. This interesting phenomenon occurs because if more tuples satisfy the selection condition, smaller prefixes of the lists need to be read and merged before the stopping condition is reached.

我們首先注意到，與其他演算法相比，LM 非常快（它的每次運行時間都不到一秒，因此它的圖形幾乎沿著 x 軸）。這是預料之中的，因為大部分計算已在預處理時完成。當選定元組的總數增加時，Scan 的性能會下降，因為需要在執行時計算更多元組的分數。相比之下，LM 和 LMM 的性能實際上略有提高。這個有趣的現象發生是因為如果更多的元組滿足選擇條件，那麼在達到停止條件之前需要讀取和合併的列表前綴就更小。

[Image]

Figure 10: Execution Times of Different Variations of List Merge and Scan for Seattle Homes Dataset

圖 10：西雅圖房屋資料集上列表合併和掃描不同變體的執行時間

Thus, List Merge and its variations are preferable if the number of tuples satisfying the query condition is large (which is exactly the situation we are interested in, i.e., the Many-Answers problem). This conclusion was reconfirmed when we repeated the experiment with LM and Scan on the much larger US Homes dataset with queries satisfying many more tuples (see Figure 11).

因此，如果滿足查詢條件的元組數量很大（這正是我們感興趣的情況，即「多答案問題」），那麼列表合併及其變體是更可取的。當我們在更大的 US Homes 資料集上重複使用 LM 和 Scan 進行實驗，查詢滿足更多的元組時，這個結論得到了再次證實（見圖 11）。

[Table]

Figure 11: Execution Times of List Merge and Scan for US Homes Dataset

圖 11：US Homes 資料集上列表合併和掃描的執行時間

Varying Number of Specified Attributes: Figure 12 shows how the query processing performance of the algorithms varies with the number of attributes specified in the selection conditions of the queries over the US Homes database (the results for the other databases are similar). The times are averaged over 10 Top-10 queries. Note that the times increase sharply for both algorithms with the number of specified attributes. The LM algorithm becomes slower because more lists need to be merged, which delays the termination condition. The Scan algorithm becomes slower because the selection time increases with the number of specified attributes. This experiment demonstrates the criticality of keeping the number of sorted streams small in our adaptation of TA.

變更指定屬性的數量：圖 12 顯示了在 US Homes 資料庫上，演算法的查詢處理性能如何隨著查詢選擇條件中指定的屬性數量而變化（其他資料庫的結果類似）。時間是 10 個 Top-10 查詢的平均值。請注意，隨著指定屬性數量的增加，兩種演算法的時間都急劇增加。LM 演算法變慢是因為需要合併更多的列表，這會延遲終止條件。Scan 演算法變慢是因為選擇時間隨著指定屬性數量的增加而增加。這個實驗證明了在我們對 TA 的改編中保持排序流數量較少的重要性。

[Image]

Figure 12: Varying Number of Specified Attributes for US Homes Dataset

圖 12：US Homes 資料集上變更指定屬性數量的影響

## 7. Conclusions

## 7. 結論

We proposed a completely automated approach for the Many-Answers Problem which leverages data and workload statistics and correlations. Our ranking functions are based upon the probabilistic IR models, judiciously adapted for structured data. We presented results of preliminary experiments which demonstrate the efficiency as well as the quality of our ranking system.

我們針對「多答案問題」提出了一個完全自動化的方法，該方法利用了資料、工作負載統計數據以及其間的關聯性。我們的排序函式基於機率性資訊檢索（IR）模型，並審慎地為結構化資料進行了調整。我們展示的初步實驗結果證明了我們排序系統的效率與品質。

Our work brings forth several intriguing open problems. For example, many relational databases contain text columns in addition to numeric and categorical columns. It would be interesting to see whether correlations between text and non-text data can be leveraged in a meaningful way for ranking. Finally, comprehensive quality benchmarks for database ranking need to be established. This would provide future researchers with a more unified and systematic basis for evaluating their retrieval algorithms.

我們的研究引出了幾個有趣的開放性問題。例如，許多關聯式資料庫除了數值和類別欄位外，還包含文字欄位。探討文字與非文字資料之間的關聯性是否能以有意義的方式應用於排序，將會是個有趣的研究方向。最後，需要為資料庫排序建立全面的品質基準。這將為未來的研究人員提供一個更統一和系統化的基礎，以評估他們的檢索演算法。

## References

## 參考文獻

[1] S. Agrawal, S. Chaudhuri G. Das. DBXplorer: A System for Keyword Based Search over Relational Databases. ICDE 2002.
[2] S. Agrawal, S. Chaudhuri, G. Das and A. Gionis. Automated Ranking of Database Query Results. CIDR, 2003.
[3] R. Agrawal, H. Mannila, R. Srikant, H. Toivonen and A. I. Verkamo. Fast Discovery of Association Rules. Advances in Knowledge Discovery and Data Mining, 1995.
[4] R. Baeza-Yates and B. Ribeiro-Neto. Modern Information Retrieval. ACM Press, 1999.
[5] G. Bhalotia, C. Nakhe, A. Hulgeri, S. Chakrabarti and S. Sudarshan. Keyword Searching and Browsing in Databases using BANKS. ICDE 2002.
[6] H. M. Blanken, T. Grabs, H.-J. Schek, R. Schenkel, G. Weikum (Eds.): Intelligent Search on XML Data: Applications, Languages, Models, Implementations, and Benchmarks. LNCS 2818 Springer 2003.
[7] J. Breese, D. Heckerman and C. Kadie. Empirical Analysis of Predictive Algorithms for Collaborative Filtering. 14th Conference on Uncertainty in Artificial Intelligence, 1998.
[8] N. Bruno, L. Gravano, and S. Chaudhuri. Top-K Selection Queries over Relational Databases: Mapping Strategies and Performance Evaluation. ACM TODS, 2002.
[9] N. Bruno, L. Gravano, A. Marian. Evaluating Top-K Queries over Web-Accessible Databases. ICDE 2002.
[10] B. Cestnik. Estimating Probabilities: A Crucial Task in Machine Learning, European Conf. in AI, 1990.
[11] K. Chakrabarti, K. Porkaew and S. Mehrotra. Efficient Query Ref. in Multimedia Databases. ICDE 2000.
[12] W. Cohen. Integration of Heterogeneous Databases Without Common Domains Using Queries Based on Textual Similarity. SIGMOD, 1998.
[13] W. Cohen. Providing Database-like Access to the Web Using Queries Based on Textual Similarity. SIGMOD 1998.
[14] W.B. Croft, J. Lafferty. Language Modeling for Information Retrieval. Kluwer 2003.
[15] R. Fagin. Fuzzy Queries in Multimedia Database Systems. PODS 1998.
[16] R. Fagin, A. Lotem and M. Naor. Optimal Aggregation Algorithms for Middleware. PODS 2001.
[17] N. Fuhr. A Probabilistic Framework for Vague Queries and Imprecise Information in Databases. VLDB 1990.
[18] N. Fuhr. A Probabilistic Relational Model for the Integration of IR and Databases. ACM SIGIR Conference on Research and Development in Information Retrieval, 1993.
[19] U. Güntzer, W.-T. Balke, W. Kießling: Optimizing Multi-Feature Queries for Image Databases. VLDB 2000.
[20] V. Hristidis, Y. Papakonstantinou. DISCOVER: Keyword Search in Relational Databases. VLDB 2002.
[21] W. Kießling. Foundations of Preferences in Database Systems. VLDB 2002.
[22] D. Kossmann, F. Ramsak, S. Rost: Shooting Stars in the Sky: An Online Algorithm for Skyline Queries. VLDB 2002.
[23] A. Motro. VAGUE: A User Interface to Relational Databases that Permits Vague Queries. TOIS 1988, 187-214.
[24] Z. Nazeri, E. Bloedorn and P. Ostwald. Experiences in Mining Aviation Safety Data. SIGMOD 2001.
[25] S. Nepal, M. V. Ramakrishna: Query Processing Issues in Image (Multimedia) Databases. ICDE 1999.
[26] M. Ortega-Binderberger, K. Chakrabarti and S. Mehrotra. An Approach to Integrating Query Refinement in SQL, EDBT 2002, 15-33.
[27] Y. Rui, T. S. Huang and S. Merhotra. Content-Based Image Retrieval with Relevance Feedback in MARS. IEEE Conf. on Image Processing, 1997.
[28] K. Sparck Jones, S. Walker, S. E. Robertson: A Probabilistic Model of Information Retrieval: Development and Comparative Experiments - Part 1. Inf. Process. Manage. 36(6): 779-808, 2000.
[29] K. Sparck Jones, S. Walker, S. E. Robertson: A Probabilistic Model of Information Retrieval: Development and Comparative Experiments - Part 2. Inf. Process. Manage. 36(6): 809-840, 2000.
[30] J. Whittaker. Graphical Models in Applied Multivariate Statistics. Wiley, 1990.
[31] L. Wimmers, L. M. Haas, MT. Roth and C. Braendli. Using Fagin's Algorithm for Merging Ranked Results in Multimedia Middleware. CoopIS 1999.
[32] L. Wu, C. Faloutsos, K. Sycara and T. Payne. FALCON: Feedback Adaptive Loop for Content-Based Retrieval. VLDB 2000.