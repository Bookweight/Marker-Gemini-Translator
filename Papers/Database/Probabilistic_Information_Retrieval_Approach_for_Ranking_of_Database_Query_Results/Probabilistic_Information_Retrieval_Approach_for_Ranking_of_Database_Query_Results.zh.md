
# Probabilistic Information Retrieval Approach for Ranking of Database Query Results
# 用於資料庫查詢結果排序的機率性資訊檢索方法

SURAJIT CHAUDHURI
Microsoft Research
GAUTAM DAS
University of Texas at Arlington
VAGELIS HRISTIDIS
Florida International University
and
GERHARD WEIKUM
Max Planck Institut fur Informatik

SURAJIT CHAUDHURI
微軟研究院
GAUTAM DAS
德州大學阿靈頓分校
VAGELIS HRISTIDIS
佛羅里達國際大學
以及
GERHARD WEIKUM
馬克斯·普朗克資訊科學研究所

We investigate the problem of ranking the answers to a database query when many tuples are returned. In particular, we present methodologies to tackle the problem for conjunctive and range queries, by adapting and applying principles of probabilistic models from information retrieval for structured data. Our solution is domain independent and leverages data and workload statistics and correlations. We evaluate the quality of our approach with a user survey on a real database. Furthermore, we present and experimentally evaluate algorithms to efficiently retrieve the top ranked results, which demonstrate the feasibility of our ranking system.
我們研究當資料庫查詢返回許多元組時對答案進行排序的問題。特別是，我們提出了處理連接查詢和範圍查詢問題的方法，透過調整和應用資訊檢索中的機率模型原理來處理結構化資料。我們的解決方案是領域獨立的，並利用資料和工作負載的統計數據和相關性。我們透過在真實資料庫上進行使用者調查來評估我們方法的品質。此外，我們提出並透過實驗評估了能有效檢索排名靠前結果的演算法，這證明了我們排序系統的可行性。

Categories and Subject Descriptors: H.3.3 [Information Storage and Retrieval]: Information Search and Retrieval; H.2.4 [Database Management]: Systems
類別與主題描述符：H.3.3 [資訊儲存與檢索]：資訊搜尋與檢索；H.2.4 [資料庫管理]：系統

General Terms: Experimentation, Performance, Theory
一般術語：實驗、效能、理論

Additional Key Words and Phrases: Probabilistic information retrieval, user survey, experimentation, indexing, automatic ranking, relational queries, workload
附加關鍵詞與短語：機率性資訊檢索、使用者調查、實驗、索引、自動排序、關聯式查詢、工作負載

V. Hristidis has been partially supported by NSF grant IIS-0534530.
V. Hristidis 的部分研究由美國國家科學基金會（NSF）補助金 IIS-0534530 支持。

Part of this work was performed while G. Das was a researcher, V. Hristidis was an intern, and G. Weikum was a visitor at Microsoft Research.
此部分工作是在 G. Das 擔任研究員、V. Hristidis 擔任實習生以及 G. Weikum 擔任微軟研究院訪問學者期間完成的。

A conference version of this article titled "Probabilistic Ranking of Database Query Results." appeared in Proceedings of VLDB 2004.
本文的會議版本，標題為「資料庫查詢結果的機率性排序」，曾發表於 2004 年 VLDB 會議論文集。

## 1. INTRODUCTION
## 1. 緒論

Database systems support a simple Boolean query retrieval model, where a selection query on a SQL database returns all tuples that satisfy the conditions in the query. This often leads to the Many-Answers Problem: when the query is not very selective, too many tuples may be in the answer. We use the following running example throughout the article:
資料庫系統支援一種簡單的布林查詢檢索模型，其中對 SQL 資料庫的選擇查詢會返回所有滿足查詢條件的元組。這通常會導致「多答案問題」（Many-Answers Problem）：當查詢的選擇性不是很強時，答案中可能會包含太多的元組。我們在整篇文章中使用以下運行範例：

Example: Consider a realtor database consisting of a single table with attributes such as (TID, Price, City, Bedrooms, Bathrooms, LivingArea, SchoolDistrict, View, Pool, Garage, BoatDock...). Each tuple represents a home for sale in the US.
範例：考慮一個房地產經紀人資料庫，其中包含一個單一表格，屬性包括（TID, Price, City, Bedrooms, Bathrooms, LivingArea, SchoolDistrict, View, Pool, Garage, BoatDock...）。每個元組代表一棟在美國待售的房屋。

Consider a potential home buyer searching for homes in this database. A query with a not very selective condition such as "City=Seattle and View= Waterfront" may result in too many tuples in the answer, since there are many homes with waterfront views in Seattle.
考慮一位潛在的購房者正在此資料庫中搜尋房屋。一個選擇性不是很強的查詢條件，例如「City=Seattle and View=Waterfront」，可能會導致答案中有太多的元組，因為西雅圖有許多擁有水景的房屋。

The Many-Answers Problem has also been investigated in information retrieval (IR), where many documents often satisfy a given keyword-based query. Approaches to overcome this problem range from query reformulation techniques (e.g., the user is prompted to refine the query to make it more selective), to automatic ranking of the query results by their degree of "relevance" to the query (though the user may not have explicitly specified how) and returning only the top-k subset.
「多答案問題」在資訊檢索（IR）領域也曾被研究，其中許多文件通常會滿足一個給定的關鍵字查詢。解決這個問題的方法包括查詢重構技術（例如，提示使用者細化查詢以使其更具選擇性），以及根據查詢結果與查詢的「相關性」程度（儘管使用者可能沒有明確說明如何衡量）自動對其進行排序，並僅返回前 k 個子集。

It is evident that automated ranking can have compelling applications in the database context. For instance, in the earlier example of a homebuyer searching for homes in Seattle with waterfront views, it may be preferable to first return homes that have other desirable attributes, such as good school districts, boat docks, etc. In general, customers browsing product catalogs will find such functionality attractive.
顯然，自動排序在資料庫領域具有引人注目的應用。例如，在前面提到的購房者在西雅圖尋找水景房的例子中，優先返回那些具有其他理想屬性（如優良學區、船塢等）的房屋可能更為可取。總的來說，瀏覽產品目錄的顧客會發現此類功能很有吸引力。

In this article we propose an automated ranking approach for the Many-Answers Problem for database queries. Our solution is principled, comprehensive, and efficient. We summarize our contributions below.
在本文中，我們針對資料庫查詢的「多答案問題」提出了一種自動化的排序方法。我們的解決方案具有原則性、全面性且高效率。我們在下面總結我們的貢獻。

Any ranking function for the Many-Answers Problem has to look beyond the attributes specified in the query, because all answer tuples satisfy the specified conditions.¹ However, investigating unspecified attributes is particularly tricky since we need to determine what the user's preferences for these unspecified attributes are. In this article we propose that the ranking function of a tuple depends on two factors: (a) a global score which captures the global importance of unspecified attribute values, and (b) a conditional score which captures the strengths of dependencies (or correlations) between specified and unspecified attribute values. For example, for the query “City = Seattle and View = Waterfront" (we also consider IN queries, e.g., City IN (Seattle, Redmond)), a home that is also located in a "SchoolDistrict = Excellent" gets high rank because good school districts are globally desirable. A home with also “BoatDock = Yes"
任何針對「多答案問題」的排序函數都必須考慮查詢中未指定的屬性，因為所有答案元組都滿足指定的條件。¹然而，研究未指定的屬性特別棘手，因為我們需要確定使用者對這些未指定屬性的偏好。在本文中，我們提出元組的排序函數取決於兩個因素：(a) 一個捕捉未指定屬性值全域重要性的全域分數，以及 (b) 一個捕捉指定與未指定屬性值之間依賴（或相關）強度的條件分數。例如，對於查詢「City = Seattle and View = Waterfront」（我們也考慮 IN 查詢，例如 City IN (Seattle, Redmond)），一棟同時位於「SchoolDistrict = Excellent」的房屋會獲得高排名，因為好的學區是普遍受歡迎的。一棟同時擁有「BoatDock = Yes」的房屋也會獲得高排名。

¹In the case of document retrieval, ranking functions are often based on the frequency of occurrence of query values in documents (term frequency, or TF). However, in the database context, especially in the case of categorical data, TF is irrelevant as tuples either contain or do not contain a query value. Hence ranking functions need to also consider values of unspecified attributes.
¹在文件檢索中，排序函數通常基於查詢詞在文件中的出現頻率（詞頻，或 TF）。然而，在資料庫情境中，特別是對於分類數據，TF 是無關緊net的，因為元組要麼包含查詢值，要麼不包含。因此，排序函數也需要考慮未指定屬性的值。

ACM Transactions on Database Systems, Vol. 31, No. 3, September 2006.
ACM Transactions on Database Systems, Vol. 31, No. 3, September 2006.

gets high rank because people desiring a waterfront are likely to want a boat dock. While these scores may be estimated by the help of domain expertise or through user feedback, we propose an automatic estimation of these scores via workload as well as data analysis. For example, past workload may reveal that a large fraction of users seeking homes with a waterfront view have also requested boat docks. We extend our framework to also support numeric attributes (e.g., age), in addition to categorical, by exploiting state-of-the-art bucketing methods based on histograms.
之所以排名高，是因為想要水景的人很可能也想要一個船塢。雖然這些分數可以借助領域專家的幫助或透過使用者回饋來估計，但我們提出了一種透過工作負載和資料分析自動估計這些分數的方法。例如，過去的工作負載可能會顯示，尋找水景房屋的使用者中有很大一部分也要求有船塢。我們將我們的框架擴展到除了分類屬性外，還支援數值屬性（例如年齡），方法是利用基於直方圖的先進分桶方法。

The next challenge is: how do we translate these basic intuitions into principled and quantitatively describable ranking functions? To achieve this, we develop ranking functions that are based on probabilistic information retrieval (PIR) ranking models. We chose PIR models because we could extend them to model data dependencies and correlations (the critical ingredients of our approach) in a more principled manner than if we had worked with alternative IR ranking models such as the Vector-Space model. We note that correlations are sometimes ignored in IR data-important exceptions are relevance feedback-based IR systems-because they are very difficult to capture in the very high-dimensional and sparsely populated feature spaces of text whereas there are often strong correlations between attribute values in relational data (with functional dependencies being extreme cases), which is a much lower-dimensional, more explicitly structured, and densely populated space that our ranking functions can effectively work on. Furthermore, we exploit possible functional dependencies in the database to improve the quality of the ranking.
下一個挑戰是：我們如何將這些基本的直覺轉化為有原則且可量化描述的排序函數？為了實現這一點，我們開發了基於機率資訊檢索（PIR）排序模型的排序函數。我們選擇 PIR 模型，是因為我們可以比使用其他 IR 排序模型（如向量空間模型）更有原則地擴展它們，以模型化資料的依賴性和相關性（我們方法的關鍵要素）。我們注意到，在 IR 資料中，相關性有時會被忽略——基於相關性回饋的 IR 系統是重要的例外——因為在文本的極高維度和稀疏填充的特徵空間中很難捕捉到它們，而在關聯式資料中，屬性值之間通常存在很強的相關性（功能性依賴是極端情況），這是一個維度低得多、結構更明確、填充更密集的空間，我們的排序函數可以在其上有效運作。此外，我們利用資料庫中可能的功能性依賴來提高排序的品質。

The architecture of our ranking has a preprocessing component that collects database as well as workload statistics to determine the appropriate ranking function. The extracted ranking function is materialized in an intermediate knowledge representation layer, to be used later by a query processing component for ranking the results of queries. The ranking functions are encoded in the intermediate layer via intuitive, easy-to-understand “atomic” numerical quantities that describe (a) the global importance of a data value in the ranking process, and (b) the strengths of correlations between pairs of values (e.g., "if a user requests tuples containing value y of attribute Y, how likely is she to be also interested in value x of attribute X?”). Although our ranking approach derives these quantities automatically, our architecture allows users and/or domain experts to tune these quantities further, thereby customizing the ranking functions for different applications.
我們的排序架構包含一個預處理元件，該元件收集資料庫及工作負載統計資料以決定適當的排序函數。提取的排序函數會被具體化到一個中介知識表示層，供稍後的查詢處理元件用於對查詢結果進行排序。排序函數透過直觀、易於理解的「原子」數值量在中介層進行編碼，這些數值描述了 (a) 資料值在排序過程中的全域重要性，以及 (b) 值對之間的相關強度（例如，「如果使用者請求包含屬性 Y 的值 y 的元組，她對屬性 X 的值 x 也感興趣的可能性有多大？」）。雖然我們的排序方法會自動推導出這些量，但我們的架構允許使用者和/或領域專家進一步調整這些量，從而為不同的應用程式客製化排序函數。

We report on a comprehensive set of experimental results. We first demonstrate through user studies on real datasets that our rankings are superior in quality to previous efforts on this problem. We also demonstrate the efficiency of our ranking system. Our implementation is especially tricky because our ranking functions are relatively complex, involving dependencies/correlations between data values. We use interesting precomputation techniques which reduce this complex problem to a problem efficiently solvable using top-k algorithms.
我們報告了一套全面的實驗結果。我們首先透過在真實資料集上的使用者研究證明，我們的排序在品質上優於先前在此問題上的努力。我們也展示了我們排序系統的效率。我們的實作特別棘手，因為我們的排序函數相對複雜，涉及資料值之間的依賴/相關性。我們使用有趣的預計算技術，將這個複雜的問題簡化為一個可以使用 top-k 演算法有效解決的問題。

The rest of this article is organized as follows. In Section 2 we discuss related work. In Section 3 we define the problem. In Section 4 we discuss our approach
本文其餘部分的組織如下。在第 2 節中，我們討論相關工作。在第 3 節中，我們定義了問題。在第 4 節中，我們討論了我們的方法。

ACM Transactions on Database Systems, Vol. 31, No. 3, September 2006.
ACM Transactions on Database Systems, Vol. 31, No. 3, September 2006.

to ranking based on probabilistic models from information retrieval, along with various extensions and special cases. In Section 5 we describe an efficient implementation of our ranking system. In Section 6 we discuss the results of our experiments, and we conclude in Section 7.
基於資訊檢索的機率模型進行排序，以及各種擴展和特殊情況。在第 5 節中，我們描述了我們排序系統的高效實作。在第 6 節中，我們討論了我們的實驗結果，並在第 7 節中得出結論。

## 2. RELATED WORK
## 2. 相關工作

A preliminary version of this article appeared in Chaudhuri et al. [2004], where we presented the basic principles of using probabilistic information retrieval models to answer database queries. However, our earlier article only handled point queries (see Section 3). In this work, we show how IN and range queries can be handled and how this makes the algorithms to produce efficiently the top results more challenging (Sections 4.4.1 and 5.4). Furthermore Chaudhuri et al. [2004] focused on only categorical attributes, whereas we have a complete study of numerical attributes as well (Section 4.4.2). Chaudhuri et al. [2004] also ignored functional dependencies, which as we show can improve the quality of the results (Section 4.2.2). In this work, we also present specialized solutions for cases where no workload is available (Section 4.3.1), and no dependencies exist between attributes (Section 4.3.2). We also generalize to the case where the data resides on multiple tables (Section 4.4.3). Finally, we extend Chaudhuri et al. [2004] with a richer set of quality and performance experiments. On the quality level, we show results for IN queries and also compare them to the results of a "random" algorithm. On the performance level, we include experiments on how the number k of requested results affects the performance of the algorithms.
本文的初版見於 Chaudhuri 等人 [2004]，其中我們介紹了使用機率性資訊檢索模型來回答資料庫查詢的基本原則。然而，我們早期的文章只處理了點查詢（見第 3 節）。在這項工作中，我們展示了如何處理 IN 和範圍查詢，以及這如何使得高效產生頂級結果的演算法更具挑戰性（第 4.4.1 和 5.4 節）。此外，Chaudhuri 等人 [2004] 只關注分類屬性，而我們則對數值屬性進行了完整的研究（第 4.4.2 節）。Chaudhuri 等人 [2004] 也忽略了功能性依賴，而我們證明這可以提高結果的品質（第 4.2.2 節）。在這項工作中，我們還為沒有可用工作負載（第 4.3.1 節）和屬性之間不存在依賴關係（第 4.3.2 節）的情況提供了專門的解決方案。我們還將其推廣到資料位於多個表格的情況（第 4.4.3 節）。最後，我們用更豐富的品質和效能實驗擴展了 Chaudhuri 等人 [2004] 的工作。在品質層面上，我們展示了 IN 查詢的結果，並將其與「隨機」演算法的結果進行比較。在效能層面上，我們包含了關於請求結果的數量 k 如何影響演算法效能的實驗。

Ranking functions have been extensively investigated in information retrieval. The vector space model as well as probabilistic information retrieval (PIR) models [Baeza-Yates and Ribeiro-Neto 1999; Grossman and Frieder 2004; Sparck Jones et al. 2000a, 2000b] and statistical language models [Croft and Lafferty 2003; Grossman and Frieder 2004] are very successful in practice. Feedback-based IR systems (e.g., relevance feedback [Harper and Van Rijsbergen 1978], pseudorelevance feedback [Xu and Croft 1996]) are based on inferring term correlations and modeling term dependencies, which are related to our approach of inferring correlations within workloads and data. While our approach has been inspired by PIR models, we have adapted and extended them in ways unique to our situation, for example, by leveraging the structure as well as correlations present in the structured data and the database workload.
排序函數在資訊檢索領域已被廣泛研究。向量空間模型以及機率性資訊檢索（PIR）模型 [Baeza-Yates and Ribeiro-Neto 1999; Grossman and Frieder 2004; Sparck Jones et al. 2000a, 2000b] 和統計語言模型 [Croft and Lafferty 2003; Grossman and Frieder 2004] 在實務中非常成功。基於回饋的 IR 系統（例如，相關性回饋 [Harper and Van Rijsbergen 1978]、偽相關性回饋 [Xu and Croft 1996]）基於推斷詞彙相關性並模型化詞彙依賴性，這與我們在工作負載和資料中推斷相關性的方法有關。雖然我們的方法受到 PIR 模型的啟發，但我們以獨特的方式對其進行了調整和擴展，例如，利用結構化資料和資料庫工作負載中存在的結構和相關性。

In database research, there has been significant work on ranked retrieval from a database. The early work of Motro [1988] considered vague/imprecise similarity-based querying of databases. Probabilistic databases have been addressed in Barbara et al. [1992], Cavallo and Pittarelli [1987], Dalvi and Suciu [2005], and Lakshmanan et al. [1997]. Recently, a broader view of the needs for managing uncertain data has been evolving (see, e.g., Widom [2005]).
在資料庫研究中，關於從資料庫進行排序檢索已有大量工作。Motro [1988] 的早期工作考慮了基於模糊/不精確相似性的資料庫查詢。機率性資料庫已在 Barbara 等人 [1992]、Cavallo 和 Pittarelli [1987]、Dalvi 和 Suciu [2005] 以及 Lakshmanan 等人 [1997] 的研究中得到處理。最近，管理不確定性資料需求的更廣泛視角一直在發展（例如，參見 Widom [2005]）。

The challenging problem of integrating databases and information retrieval systems has been addressed in a number of seminal papers [Cohen 1998a, 1998b; Fuhr 1990, 1993; Fuhr and Roelleke 1997, 1998] and has gained much attention lately Amer-Yahia et al. [2005a]. More recently, information retrieval-based approaches have been extended to XML retrieval [Amer-Yahia et al. 2005b; Chinenyanga and Kushmerick 2002; Carmel et al. 2003; Fuhr
整合資料庫和資訊檢索系統這個具挑戰性的問題，已在多篇開創性論文中被探討 [Cohen 1998a, 1998b; Fuhr 1990, 1993; Fuhr and Roelleke 1997, 1998]，並在最近引起了 Amer-Yahia 等人 [2005a] 的廣泛關注。更近地，基於資訊檢索的方法已被擴展到 XML 檢索 [Amer-Yahia et al. 2005b; Chinenyanga and Kushmerick 2002; Carmel et al. 2003; Fuhr]。

and Grossjohann 2004; Guo et al. 2003; Hristidis et al. 2003b; Lalmas and Roelleke 2004; Theobald and Weikum 2002; Theobald et al. 2005]. The articles Chakrabarti et al.[2002], Ortega-Binderberger et al. [2002], Rui et al. [1997], and Wu et al. [2000] employed relevance-feedback techniques for learning similarity in multimedia and relational databases. Our approach of leveraging workloads is motivated by and related to IR models that aim to leverage query-log information (e.g., see Radlinski and Joachims [2005] and Shen et al. [2005]). Keyword-query-based retrieval systems over databases have been proposed in Agrawal et al. [2002], Bhalotia et al. [2002], Hristidis and Papakonstantinou [2002], and Hristidis et al. [2003a]. In Kiessling [2002] and Nazeri et al. [2001], the authors proposed SQL extensions in which users can specify ranking functions via soft constraints in the form of preferences. The distinguishing aspect of our work from the above is that we espouse automatic extraction of PIR-based ranking functions through data and workload statistics.
與 Grossjohann 2004; 郭等人 2003; Hristidis 等人 2003b; Lalmas 與 Roelleke 2004; Theobald 與 Weikum 2002; Theobald 等人 2005]。Chakrabarti 等人 [2002]、Ortega-Binderberger 等人 [2002]、Rui 等人 [1997] 以及 Wu 等人 [2000] 的文章採用了相關性回饋技術來學習多媒體和關聯式資料庫中的相似性。我們利用工作負載的方法受到旨在利用查詢日誌資訊的 IR 模型的啟發並與之相關（例如，參見 Radlinski 與 Joachims [2005] 以及 Shen 等人 [2005]）。Agrawal 等人 [2002]、Bhalotia 等人 [2002]、Hristidis 與 Papakonstantinou [2002] 以及 Hristidis 等人 [2003a] 提出了基於關鍵字查詢的資料庫檢索系統。在 Kiessling [2002] 和 Nazeri 等人 [2001] 中，作者提出了 SQL 擴展，使用者可以透過偏好形式的軟約束來指定排序函數。我們工作與上述工作的區別在於，我們主張透過資料和工作負載統計自動提取基於 PIR 的排序函數。

The work most closely related to our article is Agrawal et al. [2003], which briefly considered the Many-Answers Problem (although its main focus was on the Empty-Answers Problem, which occurs when a query is too selective, resulting in an empty answer set). It too proposed automatic ranking methods that rely on workload as well as data analysis. In contrast, however, our article has the following novel strengths: (a) we use more principled probabilistic PIR techniques rather than ad hoc techniques “loosely based" on the vector-space model, and (b) we take into account dependencies and correlations between data values, whereas Agrawal et al. [2003] only proposed a form of global score for ranking.
與我們文章最相關的工作是 Agrawal 等人 [2003] 的研究，該研究簡要考慮了「多答案問題」（儘管其主要焦點是「空答案問題」，即當查詢過於選擇性導致答案集為空時發生的問題）。它也提出了依賴於工作負載和資料分析的自動排序方法。然而，相比之下，我們的文章具有以下新穎的優點：(a) 我們使用更有原則的機率性 PIR 技術，而不是「鬆散基於」向量空間模型的臨時技術；(b) 我們考慮了資料值之間的依賴性和相關性，而 Agrawal 等人 [2003] 只提出了一種用於排序的全域分數形式。

Ranking is also an important component in collaborative filtering research [Breese et al. 1998]. These methods require training data using queries as well as their ranked results. In contrast, we require workloads containing queries only.
排序在協同過濾研究中也是一個重要組成部分 [Breese et al. 1998]。這些方法需要使用查詢及其排序結果作為訓練資料。相比之下，我們只需要包含查詢的工作負載。

A major concern of this article is the query processing techniques for supporting ranking. Several techniques have been previously developed in database research for the top-k problem [Bruno et al. 2002a, 2002b; Fagin 1998; Fagin et al. 2001; Wimmers et al. 1999]. We adopt the Threshold Algorithm of Fagin et al. [2001] Güntzer et al. [2000], and Nepal and Ramakrishna [1999] for our purposes, and develop interesting precomputation techniques to produce a very efficient implementation of the Many-Answers Problem. In contrast, an efficient implementation for the Many-Answers Problem was left open in Agrawal et al. [2003].
本文的一個主要關注點是支援排序的查詢處理技術。先前在資料庫研究中，已經為 top-k 問題開發了幾種技術 [Bruno et al. 2002a, 2002b; Fagin 1998; Fagin et al. 2001; Wimmers et al. 1999]。我們採用了 Fagin 等人 [2001]、Güntzer 等人 [2000] 以及 Nepal 和 Ramakrishna [1999] 的閾值演算法，並為我們的目的開發了有趣的預計算技術，以產生一個非常高效的「多答案問題」實作。相比之下，Agrawal 等人 [2003] 中並未提供「多答案問題」的高效實作。

## 3. PROBLEM DEFINITION
## 3. 問題定義

In this section, we formally define the Many-Answers Problem in ranking database query results and its different variants. We start by defining the simplest problem instance, which we later extend to more complex scenarios.
在本節中，我們正式定義了在資料庫查詢結果排序中的「多答案問題」及其不同變體。我們從定義最簡單的問題實例開始，稍後再將其擴展到更複雜的場景。

### 3.1 The Many-Answers Problem
### 3.1 多答案問題

Consider a database table D with n tuples {t₁, . . ., tn} over a set of m categorical attributes A = {A1, . . ., Am}. Consider a “SELECT * FROM D” query Q with a conjunctive selection condition of the form “WHERE X1=x1 AND ... AND
考慮一個資料庫表格 D，其中有 n 個元組 {t₁, . . ., tn}，這些元組建立在一組 m 個分類屬性 A = {A1, . . ., Am} 上。考慮一個「SELECT * FROM D」查詢 Q，其連接選擇條件的形式為「WHERE X1=x1 AND ... AND」。


Xs=xs," where each Xi is an attribute from A and x₁ is a value in its domain. The set of attributes X = {X1, ..., X$} ⊆ A is known as the set of attributes specified by the query, while the set Y = A − X is known as the set of unspecified attributes. Let S ⊆ {t1, ..., tn} be the answer set of Q. The Many-Answers Problem occurs when the query is not too selective, resulting in a large S. The focus in this article is on automatically deriving an appropriate ranking function such that only a few (say top-k) tuples can be efficiently retrieved.
Xs=xs」，其中每個 Xi 是來自 A 的屬性，而 xᵢ 是其定義域中的一個值。屬性集合 X = {X1, ..., X$} ⊆ A 被稱為查詢指定的屬性集，而集合 Y = A − X 被稱為未指定的屬性集。令 S ⊆ {t1, ..., tn} 為 Q 的答案集。「多答案問題」發生在查詢不夠選擇性，導致 S 很大時。本文的重點是自動推導出一個適當的排序函數，以便能有效率地檢索少量（例如前 k 個）元組。

### 3.2 The Empty-Answers Problem
### 3.2 空答案問題

If the selection condition of a query is very restrictive, it may happen that very few tuples, or even no tuples, will satisfy the condition—that is, S is empty or very small. This is known as the Empty-Answers Problem. In such cases, it is of interest to derive an appropriate ranking function that can also retrieve tuples that closely (though not completely) match the query condition. We do not consider the Empty-Answers Problem any further in this article.
如果查詢的選擇條件非常嚴格，可能會發生很少甚至沒有元組滿足該條件的情況——也就是說，S 是空的或非常小。這被稱為「空答案問題」。在這種情況下，有興趣推導出一個適當的排序函數，該函數也可以檢索與查詢條件緊密（雖然不完全）匹配的元組。我們在本文中不再考慮「空答案問題」。

### 3.3 Point Queries Versus Range/IN Queries and other Generalizations
### 3.3 點查詢與範圍/IN 查詢及其他推廣

The scenario in Section 3.1 only represents the simplest problem instance. For example, the type of queries described above are fairly restrictive; we refer to them as point queries because they specify single-valued equality conditions on each of the specified attributes. In a more general setting, queries may contain range/IN conditions. IN queries contain selection conditions of the form “ X 1 IN (X1,1 ... X1,r1) AND ... AND X IN (Xs,1 ... Xs,rs)." Such queries are a very convenient way of expressing alternatives in desired attribute values which are not possible to express using point queries.
第 3.1 節中的場景僅代表最簡單的問題實例。例如，上述查詢類型相當嚴格；我們稱之為點查詢，因為它們在每個指定屬性上都指定了單值相等條件。在更一般的情況下，查詢可能包含範圍/IN 條件。IN 查詢包含形式為「X1 IN (X1,1 ... X1,r1) AND ... AND Xs IN (Xs,1 ... Xs,rs)」的選擇條件。此類查詢是表達所需屬性值中替代方案的一種非常方便的方式，而這無法使用點查詢來表達。

Also, databases may be multitabled, and may contain a mix of categorical and numeric data. In this article, we develop techniques to handle the ranking problem for all these generalizations, though for the sake of simplicity of exposition, our focus in the earlier part of the article is on point queries over a single categorical table.
此外，資料庫可能是多表格的，並且可能包含分類和數值資料的混合。在本文中，我們開發了處理所有這些推廣的排序問題的技術，儘管為了闡述的簡潔性，本文前半部分的重點是單一分類表格上的點查詢。

### 3.4 Evaluation Measures
### 3.4 評估指標

We evaluate our ranking functions both in terms of quality as well as performance. Quality of the results produced is measured using the standard IR measures of precision and recall. We also evaluate the performance of our ranking functions, especially what time and space is necessary for preprocessing as well as for query processing.
我們從品質和效能兩方面評估我們的排序函數。產出結果的品質使用標準的 IR 精確率和召回率指標來衡量。我們也評估我們排序函數的效能，特別是預處理和查詢處理所需的時間和空間。

## 4. RANKING FUNCTIONS: ADAPTATION OF PIR MODELS FOR STRUCTURED DATA
## 4. 排序函數：PIR 模型在結構化資料上的應用

In this section we first review probabilistic information retrieval (PIR) techniques in IR (Section 4.1). We then show in Section 4.2 how they can be adapted for structured data for the special case of ranking the results of point queries over a single categorical table. In Section 4.3 we present two interesting special cases of these ranking functions, while in Section 4.4 we extend our techniques to handle IN queries, numeric attributes, and other generalizations.
在本節中，我們首先回顧資訊檢索（IR）中的機率性資訊檢索（PIR）技術（第 4.1 節）。然後，我們在第 4.2 節中展示如何將它們應用於結構化資料，特別是在對單一分類表格上的點查詢結果進行排序的特殊情況下。在第 4.3 節中，我們介紹了這些排序函數的兩個有趣的特例，而在第 4.4 節中，我們將我們的技術擴展到處理 IN 查詢、數值屬性和其他一般化情況。

ACM Transactions on Database Systems, Vol. 31, No. 3, September 2006.
ACM Transactions on Database Systems, Vol. 31, No. 3, September 2006.

### 4.1 Review of Probabilistic Information Retrieval
### 4.1 機率性資訊檢索回顧

Much of the material of this subsection can be found in textbooks on information retrieval, such as those by Baeza-Yates and Ribeiro-Neto [1999] (see also Sparck Jones et al. [2000a; 2000b]). Probabilistic Information Retrieval (PIR) makes use of the following basic formulae from probability theory:
本小節的大部分內容可以在資訊檢索的教科書中找到，例如 Baeza-Yates 和 Ribeiro-Neto [1999] 的著作（另見 Sparck Jones 等人 [2000a; 2000b]）。機率性資訊檢索（PIR）利用了機率論中的以下基本公式：

Bayes' rule: p(a | b) = p(b|a)p(a) / p(b)
貝氏定理：p(a | b) = p(b|a)p(a) / p(b)

Product rule: p(a, b | c) = p(a|c)p(b|a, c).
乘法法則：p(a, b | c) = p(a|c)p(b|a, c)。

Consider a document collection D. For a (fixed) query Q, let R represent the set of relevant documents, and R=D-R be the set of irrelevant documents. In order to rank any document t in D, we need to find the probability of the relevance of t for the query given the text features of t (e.g., the word/term frequencies in t), that is, p (R|t). More formally, in probabilistic information retrieval, documents are ranked by decreasing order of their odds of relevance, defined as the following score:
考慮一個文件集合 D。對於一個（固定的）查詢 Q，令 R 代表相關文件的集合，而 R=D-R 代表不相關文件的集合。為了對 D 中的任何文件 t 進行排序，我們需要找到在給定 t 的文本特徵（例如 t 中的詞/術語頻率）下，t 對於該查詢的相關性機率，即 p(R|t)。更正式地說，在機率性資訊檢索中，文件是按照其相關性機率的遞減順序進行排序的，定義為以下分數：

Score(t) ∝ p(R|t) / p(R|t) = p(t|R)p(R) / p(t) / (p(t|R)p(R) / p(t)) ∝ p(t|R) / p(t|R)
分數(t) ∝ p(R|t) / p(R|t) = p(t|R)p(R) / p(t) / (p(t|R)p(R) / p(t)) ∝ p(t|R) / p(t|R)

The final simplification in the above equation follows from the fact that p(R)and p(R)are the same for every document t and thus mere constants that do not influence the ranking of documents. The main issue now is: how are these probabilities computed, given that R and R are unknown at query time? The usual techniques in IR are to make some simplifying assumptions, such as estimating R through user feedback, approximating R as D (since R is usually small compared to D), and assuming some form of independence between query terms (e.g., the Binary Independence Model, theLinked Dependence Model, or theTree Dependence Model [Yu and Meng 1998; Baeza-Yates and Ribeiro-Neto 1999; Grossman and Frieder 2004]).
上述方程式的最終簡化源於 p(R) 和 p(R) 對於每個文件 t 都是相同的，因此是不影響文件排序的常數。現在的主要問題是：在查詢時 R 和 R 未知的情況下，這些機率是如何計算的？資訊檢索中常用的技術是做一些簡化假設，例如透過使用者回饋來估計 R，將 R 近似為 D（因為 R 通常遠小於 D），以及假設查詢詞之間存在某種形式的獨立性（例如，二元獨立模型、連結依賴模型或樹狀依賴模型 [Yu and Meng 1998; Baeza-Yates and Ribeiro-Neto 1999; Grossman and Frieder 2004]）。

In the next subsection we show how we adapt PIR models for structured databases, in particular for conjunctive queries over a single categorical table. Whereas the Binary Independence Model makes an independence assumption over all terms, we apply in the following a limited independence assumption, that is, we consider two dependent conjuncts, and view the atomic events of each conjunction to be independent.
在下一小節中，我們將展示如何將 PIR 模型應用於結構化資料庫，特別是針對單一分類表格上的連接查詢。二元獨立模型對所有術語做出獨立性假設，而我們在下文中應用有限的獨立性假設，即我們考慮兩個相依的連接詞，並將每個連接詞的原子事件視為獨立。

### 4.2 Adaptation of PIR Models for Structured Data
### 4.2 PIR 模型在結構化資料上的應用

In our adaptation of PIR models for structured databases, each tuple in a single database table D is effectively treated as a "document." For a (fixed) query Q, our objective is to derive Score(t) for any tuple t, and use this score to rank the tuples. Since we focus on the Many-Answers problem, we only need to concern ourselves with tuples that satisfy the query conditions. Recall the notation from Section 3, where X is the set of attributes specified in the query, and Y is the remaining set of unspecified attributes. We denote any tuple t as partitioned
在我們將 PIR 模型應用於結構化資料庫的過程中，單一資料庫表格 D 中的每個元組都被有效地視為一份「文件」。對於一個（固定的）查詢 Q，我們的目標是為任何元組 t 推導出 Score(t)，並使用此分數對元組進行排序。由於我們專注於「多答案問題」，我們只需要關心滿足查詢條件的元組。回顧第 3 節的符號，其中 X 是查詢中指定的屬性集，Y 是其餘未指定的屬性集。我們將任何元組 t 表示為分區的。


into two parts, t(X) and t(Y), where t(X) is the subset of values corresponding to the attributes in X, and t(Y) is the remaining subset of values corresponding to the attributes in Y. Often, when the tuple t is clear from the context, we overload notation and simply write t as consisting of two parts, X and Y (in this context, X and Y are thus sets of values rather than sets of attributes). Replacing t with X and Y (and R as D, as mentioned in Section 4.1, is commonly done in IR), we get
分為兩部分，t(X) 和 t(Y)，其中 t(X) 是對應於 X 中屬性的值子集，t(Y) 是對應於 Y 中屬性的其餘值子集。通常，當元組 t 在上下文中很清楚時，我們會重載符號，簡單地將 t 寫成由 X 和 Y 兩部分組成（在這種情況下，X 和 Y 因此是值的集合而不是屬性的集合）。用 X 和 Y 替換 t（並且如第 4.1 節所述，將 R 視為 D，這在 IR 中是常見的做法），我們得到

Score(t) ∝ p(t|R) / p(t|D) = p(X,Y|R) / p(X,Y|D) = p(Y|R)p(X|Y, R) / (p(Y|D)p(X|Y, D))
分數(t) ∝ p(t|R) / p(t|D) = p(X,Y|R) / p(X,Y|D) = p(Y|R)p(X|Y, R) / (p(Y|D)p(X|Y, D))

where the last equality is obtained by applying Bayes' Theorem. Then, because R ⊆ X (i.e., all relevant tuples have the same X values specified in the query), we obtain P(X|Y, R) = 1 which leads to
其中最後一個等式是透過應用貝氏定理得到的。然後，因為 R ⊆ X（即所有相關元組都具有查詢中指定的相同 X 值），我們得到 P(X|Y, R) = 1，這導致

Score(t) ∝ p(Y|R) / (p(Y|D) p(X|Y, D)) (1)
分數(t) ∝ p(Y|R) / (p(Y|D) p(X|Y, D)) (1)

Let us illustrate Equation (1) with an example. Consider a query with condition "City=Kirkland and Price=High" (Kirkland is an upper-class suburb of Seattle close to a lake). Such buyers may also ideally desire homes with waterfront or greenbelt views, but homes with views looking out into streets may be somewhat less desirable. Thus, p(View=Greenbelt |R) and p(View=Waterfront |R) may both be high, but p(View=Street|R) may be relatively low. Furthermore, if in general there is an abundance of selected homes with greenbelt views as compared to waterfront views, (i.e., the denominator p(View=Greenbelt | City=Kirkland, Price=High, D) is larger than p(View=Waterfront | City=Kirkland, Price=High, D), our final rankings would be homes with waterfront views, followed by homes with greenbelt views, followed by homes with street views. For simplicity, we have ignored the remaining unspecified attributes in this example.
讓我們用一個例子來說明方程式（1）。考慮一個條件為「City=Kirkland and Price=High」的查詢（Kirkland 是西雅圖附近一個靠近湖泊的高級郊區）。這樣的買家可能也理想地希望房屋有水濱或綠化帶景觀，但面向街道的房屋可能就不那麼受歡迎了。因此，p(View=Greenbelt |R) 和 p(View=Waterfront |R) 可能都很高，但 p(View=Street|R) 可能相對較低。此外，如果總體上，與水濱景觀相比，有綠化帶景觀的待選房屋數量更多（即，分母 p(View=Greenbelt | City=Kirkland, Price=High, D) 大於 p(View=Waterfront | City=Kirkland, Price=High, D)），我們的最終排名將是水濱景觀的房屋，其次是綠化帶景觀的房屋，再其次是街道景觀的房屋。為簡單起見，我們在此範例中忽略了其餘未指定的屬性。

#### 4.2.1 Limited Independence Assumptions.
#### 4.2.1 有限獨立性假設

One possible way of continuing the derivation of Score(t) would be to make independence assumptions between values of different attributes, like in the Binary Independence Model in IR. However, while this is reasonable with text data (because estimating model parameters like the conditional probabilities p(Y|X) poses major accuracy and efficiency problems with sparse and high-dimensional data such as text), we have earlier argued that, with structured data, dependencies between data values can be better captured and would more significantly impact the result ranking. An extreme alternative to making sweeping independence assumptions would be to construct comprehensive dependency models of the data (e.g., probabilistic graphical models such as Markov Random Fields or Bayesian Networks [Whittaker 1990]), and derive ranking functions based on these models. However, our preliminary investigations suggested that such approaches have unacceptable preprocessing and query processing costs.
繼續推導 Score(t) 的一種可能方法是在不同屬性的值之間做出獨立性假設，就像在 IR 中的二元獨立模型一樣。然而，雖然這對於文本資料是合理的（因為估計像條件機率 p(Y|X) 這樣的模型參數對於像文本這樣的稀疏和高維資料會帶來重大的準確性和效率問題），但我們早先已經論證過，對於結構化資料，資料值之間的依賴關係可以被更好地捕捉，並且會更顯著地影響結果排名。與做出全面的獨立性假設相反，一個極端的替代方案是建構資料的綜合依賴模型（例如，機率圖形模型，如馬可夫隨機場或貝氏網路 [Whittaker 1990]），並基於這些模型推導排序函數。然而，我們的初步調查表明，這種方法的預處理和查詢處理成本是不可接受的。

Consequently, in this article we espouse an approach that strikes a middle ground. We only make limited forms of independence assumptions-given a query Q and a tuple t, the X (and Y) values within themselves are assumed to be
因此，在本文中，我們採用一種折衷的方法。我們==只做有限形式的獨立性假設==——給定一個查詢 Q 和一個元組 t，其 X（和 Y）值本身被假定為


independent, though dependencies between the X and Y values are allowed. More precisely, we assume limited conditional independence, that is, p(X|C) (respectively p(Y |C)) may be written as (Πx∈X p(x|C)respectively Πy∈Y p(y|C)), where C is any condition that only involves Y values (respectively X values), R, or D.
獨立的，儘管 X 和 Y 值之間允許存在依賴關係。更準確地說，我們假設有限的條件獨立性，即 p(X|C)（分別為 p(Y|C)）可以寫成（Πx∈X p(x|C) 分別為 Πy∈Y p(y|C)），其中 C 是任何只涉及 Y 值（分別為 X 值）、R 或 D 的條件。

While this assumption is patently false in many cases (for instance, in the example early in Section 4.2 this assumes that there is no dependency between homes in Kirkland and high-priced homes), nevertheless the remaining dependencies that we do leverage, that is, between the specified and unspecified values, prove to be significant for ranking. Moreover, as we shall show in Section 5, the resulting simplified functional form of the ranking function enables the efficient adaptation of known top-k algorithms through novel data structuring techniques.
雖然==這個假設在許多情況下顯然是錯誤的==（例如，在第 4.2 節早期的例子中，這假設在 Kirkland 的房屋和高價房屋之間沒有依賴關係），但我們利用的其餘依賴關係，即指定值和未指定值之間的依賴關係，證明對排名具有重要意義。此外，正如我們將在第 5 節中展示的，排序函數的簡化函數形式使得可以透過新穎的資料結構技術有效地調整已知的 top-k 演算法。

We continue the derivation of a tuple's score under the above assumptions and obtain
我們在上述假設下繼續推導元組的分數，並得到

Score(t) ∝ p(Y|R) / (p(Y|D) p(X|Y, D)) = (Πy∈Y p(y|R)) / (Πy∈Y p(y|D)) * 1 / (Πx∈X Πy∈Y p(x|y, D)) (2)
分數(t) ∝ p(Y|R) / (p(Y|D) p(X|Y, D)) = (Πy∈Y p(y|R)) / (Πy∈Y p(y|D)) * 1 / (Πx∈X Πy∈Y p(x|y, D)) (2)

#### 4.2.2 Presence of Functional Dependencies.
#### 4.2.2 功能相依性的存在

To reach Equation (2), we assumed limited conditional independence. In certain special cases such as for attributes related through functional dependencies, we can derive the equation without having to make this assumption. In the realtor database, an example of a functional dependency may be "Zipcode → City." Note that functional dependencies only apply to the data, since the workload does not have to satisfy them. For example, a query Q of the workload that specifies a requested zipcode may not have specified the city, and vice versa. Thus functional dependencies affect the denominator but not the numerator of Equation (2). The key property used to remove the independence assumption between attributes connected through functional dependencies is the following.
為了得到方程式（2），我們假設了有限的條件獨立性。在某些特殊情況下，例如對於透過功能性依賴相關的屬性，我們可以在不做此假設的情況下推導出該方程式。在房地產經紀人資料庫中，一個功能性依賴的例子可能是「郵遞區號 → 城市」。請注意，功能性依賴僅適用於資料，因為工作負載不必滿足它們。例如，工作負載中的一個查詢 Q 指定了請求的郵遞區號，但可能沒有指定城市，反之亦然。因此，功能性依賴影響方程式（2）的分母，但不影響分子。用於消除透過功能性依賴連接的屬性之間獨立性假設的關鍵屬性如下。

We first consider functional dependencies between attributes in Y. Assume that yi → yj is a functional dependency between a pair of attributes yi, yj in Y. This means that {t | t.yi = ai ∧ t.yj = aj} = {t|t.yi = ai} for all attribute values ai, aj. In this case an expression such as p(yi, yj| D) can be simplified as p(yi|D)p(yj|yi, D) = p(yi|D). More generally, the expression in Equation (1) may be simplified to Πy∈Y' p(y|D), where Y′ = {y ∈ Y | ¬∃y′ ∈ Y, FD : y' → y}.
我們首先考慮 Y 中屬性之間的功能性依賴。假設 yi → yj 是 Y 中一對屬性 yi, yj 之間的功能性依賴。這意味著對於所有屬性值 ai, aj，{t | t.yi = ai ∧ t.yj = aj} = {t|t.yi = ai}。在這種情況下，像 p(yi, yj| D) 這樣的表達式可以簡化為 p(yi|D)p(yj|yi, D) = p(yi|D)。更一般地，方程式 (1) 中的表達式可以簡化為 Πy∈Y' p(y|D)，其中 Y′ = {y ∈ Y | ¬∃y′ ∈ Y, FD : y' → y}。

Functional dependencies may also exist between attributes in X. Thus, the expression p(X|Y,D) in Equation (1) may be simplified to Πy∈Y' Πx∈X' p(x|y,D), where X' = {x ∈ X | ¬∃x′ ∈ X, FD : x' → x}.
X 中的屬性之間也可能存在功能性依賴。因此，方程式 (1) 中的表達式 p(X|Y,D) 可以簡化為 Πy∈Y' Πx∈X' p(x|y,D)，其中 X' = {x ∈ X | ¬∃x′ ∈ X, FD : x' → x}。

Applying these derivations to Equation (1), we get the following modification to Equation (2) (where X' and Y' are defined as above):
將這些推導應用於方程式（1），我們得到對方程式（2）的以下修改（其中 X' 和 Y' 的定義如上）：

Score(t) ∝ (Πy∈Y' p(y|R)) * 1 / (Πy∈Y' p(y|D)) * 1 / (Πx∈X' Πy∈Y' p(x|y, D)) (3)
分數(t) ∝ (Πy∈Y' p(y|R)) * 1 / (Πy∈Y' p(y|D)) * 1 / (Πx∈X' Πy∈Y' p(x|y, D)) (3)

Notice that before applying the above formula, we need to first compute the transitive closure of functional dependencies, for the following reason.
請注意，在應用上述公式之前，我們需要先計算功能性依賴的遞移閉包，原因如下。

Assume there are functional dependencies x' → y and y → x where x, x' ∈ X and y ∈ Y. Then, if we do not calculate the closure of functional dependencies, there would be no x' ∈ X with functional dependency x' → x, and hence Equation (3) would be the same as Equation (2). Notice that Equations (2) and (3) are equivalent if there are no functional dependencies or the only functional dependencies (in the closure) are of the form x → y or y → x, where x ∈ X and y ∈ Y.
假設存在功能性依賴 x' → y 和 y → x，其中 x, x' ∈ X 且 y ∈ Y。那麼，如果我們不計算功能性依賴的閉包，就不會有 x' ∈ X 具有功能性依賴 x' → x，因此方程式 (3) 將與方程式 (2) 相同。請注意，如果沒有功能性依賴，或者（在閉包中）唯一的功能性依賴形式為 x → y 或 y → x，其中 x ∈ X 且 y ∈ Y，則方程式 (2) 和 (3) 是等價的。

Although Equations (2) and (3) represent simplifications over Equation (1), they are still not directly computable, as R is unknown. We discuss how to estimate the quantities p(y|R) next.
雖然方程式（2）和（3）是對方程式（1）的簡化，但由於 R 未知，它們仍然無法直接計算。接下來我們討論如何估計 p(y|R) 的量。

#### 4.2.3 Workload-Based Estimation of p(y|R).
#### 4.2.3 基於工作負載的 p(y|R) 估計

Estimating the quantities p(y|R) requires knowledge of R, which is unknown at query time. The usual technique for estimating R in IR is through user feedback (relevance feedback) at query time, or through other forms of training. In our case, we provide an automated approach that leverages available workload information for estimating p(y|R). Our approach is motivated by and related to IR models that aim to leverage query-log information (e.g., see Radlinski and Joachims [2005] and Shen et al. [2005]). For example, if the multikeyword queries “a b c d," "a b,” and "a b c" constitute a (short) query log, then we could estimate p(a |c, queries) = 2/3.
估計 p(y|R) 的量需要 R 的知識，而 R 在查詢時是未知的。在資訊檢索中，估計 R 的常用技術是透過查詢時的使用者回饋（相關性回饋），或透過其他形式的訓練。在我們的案例中，我們提供了一種自動化方法，利用可用的工作負載資訊來估計 p(y|R)。我們的方法受到旨在利用查詢日誌資訊的 IR 模型的啟發並與之相關（例如，參見 Radlinski 和 Joachims [2005] 以及 Shen 等人 [2005]）。例如，如果多關鍵字查詢「a b c d」、「a b」和「a b c」構成一個（簡短的）查詢日誌，那麼我們可以估計 p(a |c, queries) = 2/3。

We assume that we have at our disposal a workload W, that is, a collection of ranking queries that have been executed on our system in the past. We first provide some intuition of how we intend to use the workload in ranking. Consider the example in Section 4.2 where a user has requested for high-priced homes in Kirkland. The workload may perhaps reveal that in the past a large fraction of users that had requested for high-priced homes in Kirkland had also requested for waterfront views. Thus for such users, it is desirable to rank homes with waterfront views over homes without such views. The IR equivalent would be to have many past queries including all of the terms "Kirkland," "high-priced," and "waterfront view," and a new query "Kirkland high-priced" arrives.
我們假設我們有一個工作負載 W，也就是過去在我們系統上執行過的一系列排序查詢。我們首先提供一些關於我們打算如何利用工作負載進行排序的直覺。考慮第 4.2 節中的例子，一位使用者請求了 Kirkland 的高價房屋。工作負載或許會揭示，過去請求 Kirkland 高價房屋的使用者中，有很大一部分也請求了水景。因此，對於這些使用者來說，將有水景的房屋排在沒有水景的房屋之前是可取的。在資訊檢索中，這相當於過去有許多查詢包含了「Kirkland」、「高價」和「水景」等所有詞彙，而現在來了一個新的查詢「Kirkland 高價」。

We note that this dependency information may not be derivable from the data alone, as a majority of such homes may not have waterfront views (i.e., data dependencies do not indicate user preferences as workload dependencies do). Of course, the other option is for a domain expert (or even the user) to provide this information (and in fact, as we shall discuss later, our ranking architecture is generic enough to allow further customization by human experts).
我們注意到，這種依賴資訊可能無法僅從資料中推導出來，因為大多數此類房屋可能沒有水景（即，資料依賴性不像工作負載依賴性那樣能反映使用者偏好）。當然，另一種選擇是由領域專家（甚至使用者）提供此資訊（事實上，正如我們稍後將討論的，我們的排序架構足夠通用，允許人類專家進一步客製化）。

More generally, the workload W is represented as a set of “tuples," where each tuple represents a query and is a vector containing the corresponding values of the specified attributes. Consider an incoming query Q which specifies a set X of attribute values. We approximate R as all query “tuples” in W that also request for X. This approximation is novel to this article, that is, that all properties of the set of relevant tuples R can be obtained by only examining the subset of the workload that contains queries that also request for X. So for a query such as "City=Kirkland and Price=High," we look at the workload in determining what such users have also requested for often in the past.
更一般地，工作負載 W 表示為一組「元組」，其中每個元組代表一個查詢，並且是一個包含指定屬性相應值的向量。考慮一個傳入的查詢 Q，它指定了一組屬性值 X。我們將 R 近似為 W 中所有也請求 X 的查詢「元組」。這種近似是本文的新穎之處，即相關元組集 R 的所有屬性都可以僅透過檢查工作負載中也請求 X 的查詢子集來獲得。因此，對於像「City=Kirkland and Price=High」這樣的查詢，我們會查看工作負載以確定這類使用者過去還經常請求什麼。

ACM Transactions on Database Systems, Vol. 31, No. 3, September 2006.
ACM Transactions on Database Systems, Vol. 31, No. 3, September 2006.

We can thus write, for query Q, with specified attribute set X, p(y|R) as p(y|X, W). Making this substitution in Equation (2), we get
因此，對於給定屬性集 X 的查詢 Q，我們可以將 p(y|R) 寫成 p(y|X, W)。將此代換入方程式 (2)，我們得到

Score(X, Y) ∝ P(Y|X, W) / (P(Y|D) P(X|Y, D))
分數(X, Y) ∝ P(Y|X, W) / (P(Y|D) P(X|Y, D))

Applying Bayes' rule for P(Y|X, W) we get
對 P(Y|X, W) 應用貝氏定理，我們得到

P(Y|X, W) = P(X, W, Y) / P(X, W) = P(W) · P(Y|W) · P(X|Y, W) / (P(W) · P(X|W))
P(Y|X, W) = P(X, W, Y) / P(X, W) = P(W) · P(Y|W) · P(X|Y, W) / (P(W) · P(X|W))

Then by dropping the constant P(W)/P(X,W) we get
然後，透過捨去常數 P(W)/P(X,W)，我們得到

Score(X, Y) ∝ P(Y|W) P(X|Y, W) / (P(Y|D) P(X|Y, D)) = (Πy∈Y p(y|W)) / (Πy∈Y p(y|D)) * (Πx∈X Πy∈Y p(x|y, W)) / (Πx∈X Πy∈Y p(x|y, D)) (4)
分數(X, Y) ∝ P(Y|W) P(X|Y, W) / (P(Y|D) P(X|Y, D)) = (Πy∈Y p(y|W)) / (Πy∈Y p(y|D)) * (Πx∈X Πy∈Y p(x|y, W)) / (Πx∈X Πy∈Y p(x|y, D)) (4)

Equation (4) is the final ranking formula, assuming no functional dependencies. If we also consider functional dependencies then we have
方程式 (4) 是最終的排序公式，假設沒有功能性依賴。如果我們也考慮功能性依賴，那麼我們有

Score(X, Y) ∝ (Πy∈Y' p(y|W)) * 1 / (Πy∈Y' p(y|D)) * (Πx∈X' Πy∈Y' p(x|y, W)) * 1 / (Πx∈X' Πy∈Y' p(x|y, D)) (5)
分數(X, Y) ∝ (Πy∈Y' p(y|W)) * 1 / (Πy∈Y' p(y|D)) * (Πx∈X' Πy∈Y' p(x|y, W)) * 1 / (Πx∈X' Πy∈Y' p(x|y, D)) (5)

where X', Y' are defined as in Equation (3).
其中 X'、Y' 的定義如方程式 (3) 所示。

Note that unlike Equations (2) and (3), we have effectively eliminated R from the formulas in Equations (4) and (5), and are only left with having to compute quantities such as p(y|W), p(x|y, W), p(y|D), and p(x|y, D). In fact, these are the "atomic" numerical quantities referred to at various places earlier in this article. Also, note that Equations (4) and (5) have been derived for point queries; the formulas get more involved when we allow IN/range conditions, as discussed in Section 4.4.1.
請注意，與方程式（2）和（3）不同，我們已有效地從方程式（4）和（5）的公式中消除了 R，只剩下計算諸如 p(y|W)、p(x|y, W)、p(y|D) 和 p(x|y, D) 等量。事實上，這些正是本文前面各處提到的「原子」數值量。另外，請注意，方程式（4）和（5）是為點查詢推導的；當我們允許 IN/範圍條件時，公式會變得更加複雜，如第 4.4.1 節所述。

Also note that the score in Equations (4) and (5) is composed of two large factors. The first factor (first product in Equations (4) and two first products in Equation (5)) may be considered as the global part of the score, while the second factor may be considered as the conditional part of the score. Thus, in the example in Section 4.2, the first part measures the global importance of unspecified values such as waterfront, greenbelt, and street views, while the second part measures the dependencies between these values and the specified values "City=Kirkland” and “Price=High."
另請注意，方程式 (4) 和 (5) 中的分數由兩個主要因素組成。第一個因素（方程式 (4) 中的第一個乘積和方程式 (5) 中的前兩個乘積）可被視為分數的全域部分，而第二個因素可被視為分數的條件部分。因此，在第 4.2 節的範例中，第一部分衡量了未指定值（如水景、綠化帶和街景）的全域重要性，而第二部分則衡量了這些值與指定值「城市=柯克蘭」和「價格=高」之間的依賴關係。

#### 4.2.4 Computing the Atomic Probabilities.
#### 4.2.4 計算原子機率

This section explains how to calculate the atomic probabilities for categorical attributes. Section 4.4.2 explains how numerical attributes can be split into ranges which are then effectively treated as categorical attributes. Our strategy is to precompute each of the atomic quantities for all distinct values in the database. The quantities p(y|W)and p(y|D) are simply the relative frequencies of each distinct value y in the workload and database, respectively (the latter is similar to IDF, or the inverse document frequency concept in IR), while the quantities p(x|y, W) and p(x|y, D) may be estimated by computing the confidences of pairwise association rules [Agrawal et al. 1995] in the workload and database, respectively.
本節說明如何計算分類屬性的原子機率。第 4.4.2 節說明如何將數值屬性分割成範圍，然後有效地將其視為分類屬性。我們的策略是預先計算資料庫中所有不同值的每個原子量。數量 p(y|W) 和 p(y|D) 分別是工作負載和資料庫中每個不同值 y 的相對頻率（後者類似於 IR 中的 IDF，或逆向文件頻率概念），而數量 p(x|y, W) 和 p(x|y, D) 可以透過分別計算工作負載和資料庫中成對關聯規則 [Agrawal et al. 1995] 的信賴度來估計。

ACM Transactions on Database Systems, Vol. 31, No. 3, September 2006.
ACM Transactions on Database Systems, Vol. 31, No. 3, September 2006.

Once this precomputation has been completed, we store these quantities as auxiliary tables in the intermediate knowledge representation layer. At query time, the necessary quantities may be retrieved and appropriately composed for performing the rankings. Further details of the implementation are discussed in Section 5.
一旦這個預計算完成，我們就將這些量作為輔助表格儲存在中介知識表示層。在查詢時，可以檢索必要的量並適當地組合以執行排名。實作的進一步細節在第 5 節中討論。

While the above is an automated approach based on workload analysis, it is possible that sometimes the workload may be insufficient and/or unreliable. In such instances, it may be necessary for domain experts to be able to tune the ranking function to make it more suitable for the application at hand. That is, our framework allows both informative (e.g., set by domain expert) as well as noninformative (e.g., inferred by query workload) prior probability distributions to be used in the preference function. In this article, we focus on noninformative priors, which are inferred by the query workload and the data.
雖然上述是基於工作負載分析的自動化方法，但有時工作負載可能不足和/或不可靠。在這種情況下，可能需要領域專家能夠調整排序函數，使其更適合手邊的應用。也就是說，我們的框架允許在偏好函數中使用資訊性（例如，由領域專家設定）和非資訊性（例如，由查詢工作負載推斷）的先驗機率分佈。在本文中，我們專注於由查詢工作負載和資料推斷的非資訊性先驗。

### 4.3 Special Cases
### 4.3 特殊情況

In this subsection we present two important special cases for which our ranking function can be further simplified: (a) ranking in the absence of workloads, and (b) ranking assuming no dependencies between attributes.
在本小節中，我們介紹了兩個重要的特殊情況，在這些情況下，我們的排序函數可以進一步簡化：(a) 在沒有工作負載的情況下進行排序，以及 (b) 假設屬性之間沒有依賴關係的情況下進行排序。

#### 4.3.1 Ranking Function in the Absence of a Workload.
#### 4.3.1 無工作負載時的排序函數

We first consider Equation (4), which describes our ranking function assuming no functional dependencies—we shall consider Equation (5) later. So far we have assumed that there exists a workload, which is used to approximate the set R of relevant tuples. If no workload is available, then we can assume that p(x|W) is the same for all distinct values x, and correspondingly p(x | y, W) is the same for all pairs of distinct values x and y. Hence, as constants, they do not affect the ranking. Thus, Equation (4) reduces to
我們首先考慮方程式（4），它描述了我們假設沒有功能性依賴的排序函數——我們稍後會考慮方程式（5）。到目前為止，我們一直假設存在一個工作負載，用於近似相關元組的集合 R。如果沒有可用的工作負載，那麼我們可以假設對於所有不同的值 x，p(x|W) 是相同的，相應地，對於所有不同的值 x 和 y 對，p(x | y, W) 也是相同的。因此，作為常數，它們不影響排名。因此，方程式（4）簡化為

Score(t) ∝ 1 / (p(Y|D) p(X|Y, D)) = 1 / (Πy∈Y p(y|D)) * 1 / (Πx∈X Πy∈Y p(x|y, D)) (6)
分數(t) ∝ 1 / (p(Y|D) p(X|Y, D)) = 1 / (Πy∈Y p(y|D)) * 1 / (Πx∈X Πy∈Y p(x|y, D)) (6)

The intuitive explanation of Equation (6) is similar to the idea of inverse document frequency (IDF) in information retrieval. In particular, the first product assigns a higher score to tuples whose unspecified attribute values y are infrequent in the database. The second product is similar to a "conditional” version of the IDF concept. That is, tuples with low correlations between the specified and the unspecified attribute values are ranked higher. This means, that tuples with infrequent combinations of values are ranked higher. For example, if the user searches for low-priced houses, then a house with high square footage is ranked high since this combination of values (low price and high square footage) is infrequent. Of course this ranking can potentially also lead to unintuitive results, for example, looking for high-priced houses may return low-square-footage ones.
方程式（6）的直觀解釋類似於資訊檢索中的逆向文件頻率（IDF）概念。特別是，第一個乘積會給那些未指定屬性值 y 在資料庫中不常見的元組分配更高的分數。第二個乘積類似於 IDF 概念的「條件」版本。也就是說，指定屬性值和未指定屬性值之間相關性低的元組排名更高。這意味著，具有不常見值組合的元組排名更高。例如，如果使用者搜尋低價房屋，那麼具有高平方英尺的房屋排名會很高，因為這種值組合（低價和高平方英尺）不常見。當然，這種排名也可能導致不直觀的結果，例如，尋找高價房屋可能會返回低平方英尺的房屋。

Equation (6) can be extended in a straightforward manner to account for the presence of functional dependencies (similarley to the way Equation (4) was extended to Equation (5)).
方程式（6）可以以直接的方式擴展，以考慮功能性依賴的存在（類似於方程式（4）擴展到方程式（5）的方式）。

#### 4.3.2 Ranking Function Assuming No Dependencies Between Attributes.
#### 4.3.2 假設屬性間無相依性的排序函數

As mentioned in Section 4.2.1, a simpler approach to the ranking problem would be to make independence assumptions between all attributes (e.g., as is done
如第 4.2.1 節所述，一個更簡單的排序問題方法是在所有屬性之間做出獨立性假設（例如，如同在

ACM Transactions on Database Systems, Vol. 31, No. 3, September 2006.
ACM Transactions on Database Systems, Vol. 31, No. 3, September 2006.

in the binary independence model in IR). Whereas, in Section 4.2, we viewed X and Y as dependent events, we show here the special case of viewing X and Y as independent events. Then the linked independence assumption holds for both, the workload W and the database D. We obtain
在 IR 的二元獨立模型中）。然而，在第 4.2 節中，我們將 X 和 Y 視為相依事件，這裡我們展示將 X 和 Y 視為獨立事件的特殊情況。那麼，連結獨立性假設對工作負載 W 和資料庫 D 都成立。我們得到

Score(t) = p(Y|W) p(X|Y, W) / (p(Y|D) p(X|Y, D)) ≈ p(Y|W) p(X|W) / (p(Y|D) p(X|D))
分數(t) = p(Y|W) p(X|Y, W) / (p(Y|D) p(X|Y, D)) ≈ p(Y|W) p(X|W) / (p(Y|D) p(X|D))

Here, the fraction p(X|W)/p(X|D) is constant for all query result tuples; hence:
此處，對於所有查詢結果元組，分數 p(X|W)/p(X|D) 是常數；因此：

Score(t) ∝ p(Y|W) / p(Y|D) = Πy∈Y p(y|W) / p(y|D) (7)
分數(t) ∝ p(Y|W) / p(Y|D) = Πy∈Y p(y|W) / p(y|D) (7)

Intuitively, the numerator describes the absolute importance of the unspecified attribute values in the workload, while the denominator resembles the IDF concept in IR. This formula is similar to the ranking formula for the Many-Answers problem developed in Agrawal et al. [2003] based on the vector-space model. The main difference between this formula and the corresponding formula in Agrawal et al. [2003] is that the latter did not have the denominator quantities, and also expressed the score in terms of logarithms. This provides formal credibility to the intuition behind the development of the algorithm in Agrawal et al. [2003].
直觀地說，分子描述了工作負載中未指定屬性值的絕對重要性，而分母則類似於 IR 中的 IDF 概念。這個公式與 Agrawal 等人 [2003] 基於向量空間模型為「多答案問題」開發的排序公式相似。這個公式與 Agrawal 等人 [2003] 中相應公式的主要區別在於，後者沒有分母量，並且還用對數來表示分數。這為 Agrawal 等人 [2003] 演算法開發背後的直覺提供了形式上的可信度。

### 4.4 Generalizations
### 4.4 推廣

In this subsection we present several important generalizations of our ranking techniques. In particular, we show how our techniques can be extended to handle IN queries, numeric attributes, and multitable databases.
在本小節中，我們將介紹我們排序技術的幾個重要推廣。特別是，我們將展示如何擴展我們的技術以處理 IN 查詢、數值屬性和多表資料庫。

#### 4.4.1 IN Queries.
#### 4.4.1 IN 查詢

IN queries are a generalization of point queries, in which selection conditions have the form “X1 IN (x1,1 ... x1,r1) AND ... AND Xs IN (xs,1 ... xs,rs)". As an example, consider a query with a selection condition such as “City IN (Kirkland, Redmond) AND Price IN (High, Moderate)." This might represent the desire of a homebuyer who is interested in either moderate or high-priced homes in either Kirkland or Redmond. Such queries are a very convenient way of expressing alternatives in desired attribute values which are not possible to express using point queries.
IN 查詢是點查詢的一種推廣，其選擇條件的形式為「X1 IN (x1,1 ... x1,r1) AND ... AND Xs IN (xs,1 ... xs,rs)」。舉例來說，考慮一個選擇條件為「City IN (Kirkland, Redmond) AND Price IN (High, Moderate)」的查詢。這可能代表一位購房者的意願，他對 Kirkland 或 Redmond 的中價位或高價位房屋感興趣。這類查詢提供了一種非常方便的方式來表達所需屬性值的替代方案，而這是點查詢無法做到的。

Accommodating IN queries in our ranking infrastructure presents the challenge of automatically determining which of the alternatives are more relevant to the user—this knowledge can then be incorporated into a suitable ranking function. (This concept is related to work on vague/fuzzy predicates [Fuhr 1990, 1993; Fuhr and Roelleke 1997, 1998]. In our case, the objective is essentially to determine the probability function that can assign different weights to the different alternative values.)
在我們的排序基礎設施中容納 IN 查詢帶來了一個挑戰，即自動確定哪些替代方案與使用者更相關——這些知識隨後可以被納入一個合適的排序函數中。（這個概念與模糊/模糊謂詞的研究有關 [Fuhr 1990, 1993; Fuhr and Roelleke 1997, 1998]。在我們的案例中，目標基本上是確定可以為不同替代值分配不同權重的機率函數。）

First the ranking function derived in Equation (4) (and Equation (5)) have to be modified to allow IN conditions in the specified attributes. The complication stems from the fact that two tuples that satisfy the query condition may differ in their specific X values. In the above example, a moderate-priced home in Redmond will satisfy the query, as will an expensive home in Kirkland. However, since the specific X values of the two homes are different, this prevents
首先，方程式 (4)（和方程式 (5)）中推導出的排序函數必須進行修改，以允許在指定的屬性中使用 IN 條件。複雜之處在於，滿足查詢條件的兩個元組可能在其特定的 X 值上有所不同。在上面的例子中，雷德蒙德的一棟中等價位的房屋將滿足查詢，柯克蘭的一棟昂貴的房屋也將滿足查詢。然而，由於這兩棟房屋的具體 X 值不同，這就阻止了

ACM Transactions on Database Systems, Vol. 31, No. 3, September 2006.
ACM Transactions on Database Systems, Vol. 31, No. 3, September 2006.

us from factoring out the X as we so successfully did in the derivation of Equation (4). This requires nontrivial extensions to the execution algorithms, as shown in Section 5. Second, the existence of IN queries complicates the generation of the association rules in the workload, as we discuss later in this subsection.
我們像在推導方程式（4）時那樣成功地將 X 分解出來。這需要對執行演算法進行非平凡的擴展，如第 5 節所示。其次，IN 查詢的存在使工作負載中關聯規則的生成變得複雜，我們將在本小節稍後討論。

##### 4.4.1.1 IN Conditions in the Query.
##### 4.4.1.1 查詢中的 IN 條件

For simplicity, let us assume the case where there are no functional dependencies and the workload has point queries, but the query may have IN conditions. Later we will extend the discussion to the case where the workload also has IN conditions.
為簡單起見，讓我們假設沒有功能性依賴，且工作負載只有點查詢，但查詢本身可以有 IN 條件。稍後我們將把討論擴展到工作負載也包含 IN 條件的情況。

Consider a query that specifies conditions C, where C is a conjunction of IN conditions such as “City IN (Bellevue, Carnation) AND SchoolDistrict IN(Good, Excellent)." Note that we distinguish C from X; the latter are atomic values of specified attributes in a specific tuple, whereas the former refers to the query and contains a set of values for each specified attribute. Recall from Section 4.2 that
考慮一個指定條件 C 的查詢，其中 C 是 IN 條件的合取，例如「City IN (Bellevue, Carnation) AND SchoolDistrict IN(Good, Excellent)」。請注意，我們將 C 與 X 區分開來；後者是特定元組中指定屬性的原子值，而前者指的是查詢，並包含每個指定屬性的一組值。回顧第 4.2 節，

Score(t) ∝ p(t|R) / p(t|D) = p(X, Y|R) / p(X, Y|D) ∝ p(X|R) p(Y|X, R) / (p(X|D) p(Y|X, D))
分數(t) ∝ p(t|R) / p(t|D) = p(X, Y|R) / p(X, Y|D) ∝ p(X|R) p(Y|X, R) / (p(X|D) p(Y|X, D))

In what follows, we shall assume that R = C, W, that is, R is the set of tuples in W that specify C. This is in tune with the corresponding assumption in Section 4.2.3 for the case of point queries, and intuitively means that R is represented by all queries in the workload that also request for C. Of course, since here we are assuming that the workload only has point queries, we need to figure out how to evaluate this in a reasonable manner.
接下來，我們將假設 R = C, W，也就是說，R 是 W 中指定 C 的元組集合。這與第 4.2.3 節中針對點查詢情況的相應假設一致，直觀地意味著 R 由工作負載中所有也請求 C 的查詢表示。當然，由於這裡我們假設工作負載只有點查詢，我們需要找出如何以合理的方式評估這一點。

Consider the second part of the above formula for Score(t), that is, p(Y|X, R)/p(Y|X, D). This can be rewritten as p(Y|X, C,W)/p(Y|X,C, D). Since we are considering the Many-Answers problem, if X is true, C is also true (recall that X is the set of attribute values of a result-tuple for the query-specified attributes). Thus this part of the formula can be simplified as p(Y|X, W)/p(Y|X, D). Consequently, it can be further simplified in exactly the same way as the derivations described earlier for point queries, that is, in Equations (1) through (4).
考慮上述 Score(t) 公式中的第二部分，即 p(Y|X, R)/p(Y|X, D)。這可以重寫為 p(Y|X, C,W)/p(Y|X,C, D)。由於我們正在考慮「多答案問題」，如果 X 為真，則 C 也為真（回想一下，X 是查詢指定屬性的結果元組的屬性值集合）。因此，公式的這一部分可以簡化為 p(Y|X, W)/p(Y|X, D)。因此，它可以以與前面為點查詢描述的推導完全相同的方式進一步簡化，即在方程式 (1) 到 (4) 中。

Now consider the first part of the formula, p(X|R)/p(X|D). Unlike the point query case, however, we cannot assume p(X|R)/p(X|D) is a constant for all tuples. In what follows, we shall assume that x is a variable that varies over the set X, and c is a variable that varies over the set C. When x and c refer to the same attribute, it is clear that, if x is true, then c is also true. We have the following sequence of derivations:
現在考慮公式的第一部分，p(X|R)/p(X|D)。然而，與點查詢的情況不同，我們不能假設 p(X|R)/p(X|D) 對所有元組都是常數。接下來，我們將假設 x 是一個在集合 X 上變化的變數，而 c 是一個在集合 C 上變化的變數。當 x 和 c 指的是同一個屬性時，很明顯，如果 x 為真，那麼 c 也為真。我們有以下推導序列：

p(X|R) / p(X|D) = p(X|C, W) / p(X|D) = (Πx∈X p(x|C, W)) / (Πx∈X p(x|D)) ∝ (Πx∈X p(C, W|x)p(x)) / p(x|D) = (Πx∈X p(W|x)p(x)p(C|x, W)) / p(x|D) ∝ Πx∈X (p(x|W) / p(x|D)) * Πc∈C p(c|x, W)
p(X|R) / p(X|D) = p(X|C, W) / p(X|D) = (Πx∈X p(x|C, W)) / (Πx∈X p(x|D)) ∝ (Πx∈X p(C, W|x)p(x)) / p(x|D) = (Πx∈X p(W|x)p(x)p(C|x, W)) / p(x|D) ∝ Πx∈X (p(x|W) / p(x|D)) * Πc∈C p(c|x, W)

ACM Transactions on Database Systems, Vol. 31, No. 3, September 2006.
ACM Transactions on Database Systems, Vol. 31, No. 3, September 2006.

Recall that we assume limited conditional independence, that is, that dependency exists only between the X and Y attributes, and not within the X attributes (recall that X and C specify the same set of attributes). Let A(x) (respectively A(c)) refer to the attribute of x (respectively c). Then p(c|x, W) is equal to p(c|W) when A(x) <> A(c), and is equal to 1 otherwise. Let c(x) represent the IN condition in C corresponding the attribute of x, that is, A(c(x)) = A(x). Consequently, we have
回想一下，我們假設有限的條件獨立性，也就是說，依賴關係只存在於 X 和 Y 屬性之間，而不在 X 屬性內部（回想一下 X 和 C 指定了相同的屬性集）。令 A(x)（分別為 A(c)）表示 x（分別為 c）的屬性。那麼當 A(x) <> A(c) 時，p(c|x, W) 等於 p(c|W)，否則等於 1。令 c(x) 表示 C 中對應於 x 屬性的 IN 條件，即 A(c(x)) = A(x)。因此，我們有

Πc∈C p(c|x, W) = (Πc∈C p(c|W)) / p(c(x)|W)
Πc∈C p(c|x, W) = (Πc∈C p(c|W)) / p(c(x)|W)

Hence, continuing with the above derivation, we have p(X|R)/p(X|D) proportional to
因此，繼續上述推導，我們有 p(X|R)/p(X|D) 正比於

(Πx∈X p(x|W) / p(x|D)) * (Πc∈C p(c|W)) / (Πx∈X p(c(x)|W)) ∝ Πx∈X p(x|W) / p(x|D)
(Πx∈X p(x|W) / p(x|D)) * (Πc∈C p(c|W)) / (Πx∈X p(c(x)|W)) ∝ Πx∈X p(x|W) / p(x|D)

This is the extra factor that needs to be multiplied to the score derived in Equation (4). Hence, the equivalent of Equation (4) for IN queries is
這是需要乘到方程式 (4) 中推導出的分數的額外因子。因此，對於 IN 查詢，方程式 (4) 的等價形式是

Score(t) ∝ Πz∈t p(z|W) / p(z|D) * Πz∈t Πy∈Y p(x|y, W) / p(x|y, D) (8)
分數(t) ∝ Πz∈t p(z|W) / p(z|D) * Πz∈t Πy∈Y p(x|y, W) / p(x|y, D) (8)

Equation (8) differs from Equation (4) in the global part. In particular, we now need to consider all attribute values of each result-tuple t, because they may be different, whereas, in Equation (4), only the unspecified values of t were used for the global part. Notice that Equation (8) can be used for point queries as well since in this case the specified values of t are common for all result-tuples and hence would only multiply the score by a common factor. However, as we explain in Section 5.4, it is more complicated to efficiently evaluate Equation (8) for IN queries than for point queries because of the fact that all result-tuples share the same specified (X) values in point queries.
方程式 (8) 與方程式 (4) 在全域部分有所不同。特別是，我們現在需要考慮每個結果元組 t 的所有屬性值，因為它們可能不同，而在方程式 (4) 中，只有 t 的未指定值用於全域部分。請注意，方程式 (8) 也可用於點查詢，因為在這種情況下，t 的指定值對於所有結果元組都是相同的，因此只會將分數乘以一個公因子。然而，正如我們在第 5.4 節中解釋的，對於 IN 查詢，有效評估方程式 (8) 比點查詢更複雜，因為在點查詢中，所有結果元組共享相同的指定 (X) 值。

We note that Equation (8) can be generalized in a straightforward manner to allow for the presence of functional dependencies.
我們注意到，方程式（8）可以以直接的方式推廣，以允許功能性依賴的存在。

##### 4.1.1.2 IN Conditions in the Workload.
##### 4.1.1.2 工作負載中的 IN 條件

We had assumed above that the query at runtime was allowed to have IN conditions, but that the workload only had point queries. We now tackle the problem of exploiting IN queries in the workload as well. This is reduced to the problem of precomputing atomic probabilities such as p(z |W) and p(x |y, W) from such a workload. These atomic probabilities are necessary for computing the ranking function derived in Equation (8).
我們上面假設執行時的查詢允許有 IN 條件，但工作負載只有點查詢。我們現在也來處理在工作負載中利用 IN 查詢的問題。這簡化為從這樣的工作負載中預先計算原子機率，例如 p(z |W) 和 p(x |y, W) 的問題。這些原子機率是計算方程式 (8) 中推導出的排序函數所必需的。

Our approach is to "conceptually expand" the workload by splitting each IN query into sets of appropriately weighted point queries. For example, a query with IN conditions such as “City IN (Bellevue, Redmond, Carnation) AND Price IN (High, Moderate)" may be split into 3 × 2 = 6 point queries, each representing specific combinations of values from the IN conditions. In this example, each such point query is given a weight of 1/6; this weighting is necessary to make
我們的方法是透過將每個 IN 查詢分割成一組適當加權的點查詢來「概念性地擴展」工作負載。例如，一個帶有 IN 條件的查詢，如「City IN (Bellevue, Redmond, Carnation) AND Price IN (High, Moderate)」，可以被分割成 3 × 2 = 6 個點查詢，每個點查詢代表 IN 條件中值的特定組合。在這個例子中，每個這樣的點查詢被賦予 1/6 的權重；這個權重是必要的，以使得

ACM Transactions on Database Systems, Vol. 31, No. 3, September 2006.
ACM Transactions on Database Systems, Vol. 31, No. 3, September 2006.

sure that queries with large IN conditions do not dominate the calculations of the atomic probabilities.
確保具有大 IN 條件的查詢不會主導原子機率的計算。

Atomic probabilities may now be computed as follows: p(z|W) is the (weighted) fraction of the queries in the expanded workload that refer to z, while p(x|y, W) is the (weighted) fraction of all queries that refer to x from all queries that refer to y in the expanded workload. Of course, the workload is not literally expanded; these probabilities can be easily computed from the original workload that contain the IN queries.
現在可以如下計算原子機率：p(z|W) 是擴展工作負載中引用 z 的查詢的（加權）分數，而 p(x|y, W) 是擴展工作負載中所有引用 y 的查詢中引用 x 的查詢的（加權）分數。當然，工作負載並非字面上擴展；這些機率可以很容易地從包含 IN 查詢的原始工作負載中計算出來。

#### 4.4.2 Numeric Attributes.
#### 4.4.2 數值屬性

Thus far in the article we have only been considering categorical data. We now extend our results to the case when the data also has numeric attributes. For example, in the homes database, we may have numeric attributes such as square footage, age, etc. Queries may now have range conditions, such as "Age BETWEEN (5, 10) AND Sqft BETWEEN (2500, 3000)."
到目前為止，在本文中我們只考慮了分類資料。我們現在將我們的結果擴展到資料也具有數值屬性的情況。例如，在房屋資料庫中，我們可能有數值屬性，如平方英尺、屋齡等。查詢現在可以有範圍條件，例如「屋齡介於 (5, 10) 之間且平方英尺介於 (2500, 3000) 之間」。

One obvious way of handling numeric data and queries is to simply treat them as categorical data to consider every distinct numerical value in the database as a categorical value. Queries with range conditions can be then converted to queries with corresponding IN conditions, and we can then apply the methods outlined in Section 4.4.1. However, the main problem arising with such an approach is that the sheer size of the numeric domain ensures that many, in fact most, distinct values are not adequately represented in the workload. For example, perhaps numerous workload queries have requested for homes between 3000 and 4000 sqft. However, there may be one or two 2995-sqft homes in the database, but unfortunately these homes would be considered far less popular by the ranking algorithm.
處理數值資料和查詢的一個顯而易見的方法是簡單地將它們視為分類資料，將資料庫中的每個不同數值都視為一個分類值。帶有範圍條件的查詢可以轉換為帶有相應 IN 條件的查詢，然後我們可以應用第 4.4.1 節中概述的方法。然而，這種方法產生的主要問題是，數值域的龐大規模確保了許多，事實上是大多數，不同的值在工作負載中沒有得到充分的體現。例如，也許大量的工作負載查詢請求了 3000 到 4000 平方英尺之間的房屋。然而，資料庫中可能有一兩棟 2995 平方英尺的房屋，但不幸的是，這些房屋會被排序演算法認為遠不那麼受歡迎。

A simple strategy for overcoming this problem is to discretize the numerical domain into buckets, which can then be treated as categorical data. However, most simple bucketing techniques are errorprone because inappropriate choices of bucket boundaries may separate two values that are otherwise close to each other. In fact, complex bucketing techniques for numeric data have been extensively studied in other domains, such as in the construction of histograms for approximating data distributions (see Poosala et al. [1996; Jagadish et al. 1998]) and in earlier database ranking algorithms (see Agrawal et al. [2003]), as well as in discretization methods in classification studies (see Martinez et al. [2004]). In this article too, we investigate the bucketing problem that arises in our context in a systematic manner, and present principled solutions that are adaptations of well-known methods for histogram construction.
克服這個問題的一個簡單策略是將數值域離散化為桶，然後可以將其視為分類資料。然而，大多數簡單的分桶技術都容易出錯，因為不當的桶邊界選擇可能會將兩個原本相近的值分開。事實上，數值資料的複雜分桶技術已在其他領域被廣泛研究，例如在建構直方圖以近似資料分佈（見 Poosala 等人 [1996；Jagadish 等人 1998]）和早期的資料庫排序演算法（見 Agrawal 等人 [2003]），以及分類研究中的離散化方法（見 Martinez 等人 [2004]）。在本文中，我們也系統地研究了在我們的背景下出現的分桶問題，並提出了基於著名直方圖建構方法的原則性解決方案。

Let us consider where exactly the problem of numeric attributes arises in our case. Given a query Q, the problem arises when we attempt to compute the score of a tuple t based on the ranking formula in Equation (8). We need accurate estimations of the atomic probabilities p(z | W), p(z | D), p(x | y, W), and p(x | y, D) when some of these values are numeric. What is really needed is a way of "smoothening" the computations of these atomic probabilities, so that, for example, if p(z | W) is high for a numeric value (i.e., z has been referenced many times in the workload), p(z+ɛ| W) should also be high for nearby values z+ɛ. Similar smoothening techniques should be applied to the other types of atomic
讓我們考慮一下，在我們的案例中，數值屬性的問題究竟出現在哪裡。給定一個查詢 Q，當我們試圖根據方程式 (8) 中的排序公式計算元組 t 的分數時，問題就出現了。當其中一些值是數值時，我們需要準確估計原子機率 p(z | W)、p(z | D)、p(x | y, W) 和 p(x | y, D)。真正需要的是一種「平滑化」這些原子機率計算的方法，以便，例如，如果對於一個數值 z，p(z | W) 很高（即 z 在工作負載中被多次引用），那麼對於附近的值 z+ɛ，p(z+ɛ| W) 也應該很高。類似的平滑化技術也應該應用於其他類型的原子。

ACM Transactions on Database Systems, Vol. 31, No. 3, September 2006.
ACM Transactions on Database Systems, Vol. 31, No. 3, September 2006.

probabilities, p(z|D), p(x|y, W) and p(x | y, D). Furthermore, these probabilities have to be precomputed earlier, and should only be "looked up" at query time. In the following we discuss our solutions in more detail.
機率，p(z|D)、p(x|y, W) 和 p(x | y, D)。此外，這些機率必須預先計算，並且在查詢時只能「查閱」。下面我們將更詳細地討論我們的解決方案。

##### 4.4.2.1 Estimating p(z |D) and p(x|y, D).
##### 4.4.2.1 估計 p(z |D) 和 p(x|y, D)

We first discuss how to estimate p(z | D). Let z be a value of some numeric attribute, say A. As mentioned earlier, the naïve but inaccurate way of estimating p(z | D) would be to simply treat A as a categorical attribute—thus p(z | D) would be the relative frequency of the occurrence of z in the database. Instead, our approach is to assume that p(z | D) is the density, at point z, of a continuous probability density function (pdf) p(z | D) over the domain of A. We therefore use standard density estimation techniques in our case, histograms-to approximate this pdf using the values of A occurring in the database. There are a wide variety of histogram techniques for density estimation, such as equiwidth histograms, equidepth histograms, and even "optimal” histograms where bucket boundaries are set such that the squared error between the actual data distribution and the distribution represented by the histogram is minimized (see Poosala et al. [1996]; Jagadish et al. [1998] for relevant results on histogram construction). In our case, we use the popular and efficient technique of equidepth histograms, where the range is divided into a set of nonoverlapping buckets such that each bucket contains the same number of values.² Once this histogram has been precomputed, the density p(z | D) at any point z is looked up at runtime by determining the bucket to which z belongs.
我們首先討論如何估計 p(z | D)。令 z 為某個數值屬性 A 的值。如前所述，估計 p(z | D) 的天真但不準確的方法是簡單地將 A 視為分類屬性——因此 p(z | D) 將是 z 在資料庫中出現的相對頻率。相反，我們的方法是假設 p(z | D) 是在 A 的定義域上，連續機率密度函數 (pdf) p(z | D) 在點 z 的密度。因此，我們在我們的案例中使用標準的密度估計技術，即直方圖，來利用資料庫中出現的 A 值來近似這個 pdf。有各種各樣的直方圖技術用於密度估計，例如等寬直方圖、等深直方圖，甚至「最佳」直方圖，其中桶的邊界被設定為使得實際資料分佈與直方圖表示的分佈之間的平方誤差最小化（關於直方圖建構的相關結果，請參見 Poosala 等人 [1996]；Jagadish 等人 [1998]）。在我們的案例中，我們使用流行且高效的等深直方圖技術，其中範圍被劃分為一組不重疊的桶，使得每個桶包含相同數量的數值。² 一旦這個直方圖被預先計算，任何點 z 的密度 p(z | D) 就可以在執行時透過確定 z 所屬的桶來查閱。

We next discuss how to estimate p(x|y, D). Intuitively, our approach is to compute a two-dimensional histogram that represents the distribution of all (x, y) pairs that occur in the database. At runtime, we look up this histogram to determine the density, at point x, of the marginal distribution p(x|y, D).
接下來我們討論如何估計 p(x|y, D)。直觀地說，我們的方法是計算一個二維直方圖，它表示資料庫中所有 (x, y) 對的分佈。在執行時，我們查閱這個直方圖來確定在點 x 處，邊際分佈 p(x|y, D) 的密度。

Consider first the case where the attribute A of x is numeric, but the attribute B of y is categorical. Our approach for this problem is to compute, for each distinct value y of B, the histogram over all values of A that cooccur with y in the database. Each such histogram represents the marginal probability density function p(x | y, D). One issue that arises is if there are numerous distinct values for B, which may result in too many histograms. We circumvent this problem by only building histograms for those y values for which the corresponding number of A values occurring in the database is larger than a given threshold.
首先考慮 x 的屬性 A 是數值型，而 y 的屬性 B 是類別型的情況。我們對這個問題的方法是，對於 B 的每個不同值 y，計算資料庫中所有與 y 共同出現的 A 值的直方圖。每個這樣的直方圖代表邊際機率密度函數 p(x | y, D)。一個出現的問題是，如果 B 有許多不同的值，這可能會導致太多的直方圖。我們透過只為那些在資料庫中出現的相應 A 值數量大於給定閾值的 y 值建立直方圖來規避這個問題。

We next consider the case where A is categorical whereas B is numeric. We first compute the histogram of the distribution p(y|D) as explained above. We then compute pairwise association rules of the form b → x where b is any bucket of p(y |D) and x is any value of A. Then the density p(x|y, D) is approximated as the confidence of the association rule b→ x where b is the bucket to which y belongs.
接下來我們考慮 A 是分類而 B 是數值的情況。我們首先如上所述計算分佈 p(y|D) 的直方圖。然後我們計算形式為 b → x 的成對關聯規則，其中 b 是 p(y |D) 的任何桶，x 是 A 的任何值。然後密度 p(x|y, D) 近似為關聯規則 b→ x 的信賴度，其中 b 是 y 所屬的桶。

Finally, consider the case where A and B are both numeric. As above, we first compute the histogram for p(y|D). Then, for each bucket b of the histogram corresponding to p(y|D), we compute the histogram over all values of A that cooccur with b in the database. Each such histogram represents the marginal
最後，考慮 A 和 B 都是數值的情況。如上所述，我們首先計算 p(y|D) 的直方圖。然後，對於對應於 p(y|D) 的直方圖的每個桶 b，我們計算資料庫中所有與 b 共同出現的 A 值的直方圖。每個這樣的直方圖代表邊際

²In our approach, we set the number of buckets to 50.
²在我們的方法中，我們將桶的數量設定為 50。

ACM Transactions on Database Systems, Vol. 31, No. 3, September 2006.
ACM Transactions on Database Systems, Vol. 31, No. 3, September 2006.

probability density function p(x|y, D). As before, if there are numerous buckets of p(y | D), this may result in too many histograms, so we only build histograms for those buckets for which the corresponding number of A values occurring in the database is larger than a given threshold.
機率密度函數 p(x|y, D)。和以前一樣，如果 p(y | D) 的桶過多，可能會導致直方圖過多，所以我們只為那些在資料庫中出現的相應 A 值數量大於給定閾值的桶建立直方圖。

##### 4.4.2.2 Estimating p(z | W) and p(x|y, W).
##### 4.4.2.2 估計 p(z | W) 和 p(x|y, W)

The estimation of these quantities is similar to the corresponding methods outlined above, except that the various histograms have to be built using the workload rather than the database. The further complication is that, unlike the database where histograms are built over sets of point data, the workload contains range queries, and thus the histograms have to be built over sets of ranges. We outline the extensions necessary for the estimation of p(z | W); the extensions for estimating p(x|y, W) are straightforward and omitted.
這些量的估計與上面概述的相應方法類似，只是各種直方圖必須使用工作負載而不是資料庫來建構。更複雜的是，與資料庫中直方圖是建立在點資料集上不同，工作負載包含範圍查詢，因此直方圖必須建立在範圍集上。我們概述了估計 p(z | W) 所需的擴展；估計 p(x|y, W) 的擴展是直接的，因此省略。

Let z be a value of a numeric attribute A. As before, our approach is to assume that p(z | W) is the density, at point z, of a continuous probability density function p(z | W) over the domain of A. However, we cannot directly use standard density estimation techniques such as histograms because, unlike the database, the workload specifies a set of ranges over the domain of A, rather than a set of points over the domain of A.
令 z 為數值屬性 A 的一個值。和以前一樣，我們的方法是假設 p(z | W) 是在 A 的定義域上，連續機率密度函數 p(z | W) 在點 z 的密度。然而，我們不能直接使用標準的密度估計技術，例如直方圖，因為與資料庫不同，工作負載指定了 A 定義域上的一組範圍，而不是 A 定義域上的一組點。

We extend the concept of equidepth histograms to sets of ranges as follows. Let query Qi in the workload specify the range (zLi, zRi). If this is the only query in the workload, we can view this as a probability density function over the domain of A, where the density is 1/(zRi – zLi) for all points zLi≤ z < zRi, and 0 for all other points. The pdf for the entire workload is computed by averaging these individual distributions at all points over all queries—thus the pdf for the workload will resemble a histogram with a potentially large number of buckets (proportional to the number of queries in the workload).
我們將等深直方圖的概念擴展到範圍集合，如下所示。令工作負載中的查詢 Qi 指定範圍 (zLi, zRi)。如果這是工作負載中唯一的查詢，我們可以將其視為 A 定義域上的機率密度函數，其中對於所有點 zLi≤ z < zRi，密度為 1/(zRi – zLi)，對於所有其他點，密度為 0。整個工作負載的機率密度函數是透過在所有查詢的所有點上對這些個別分佈進行平均來計算的——因此，工作負載的機率密度函數將類似於一個具有潛在大量桶的直方圖（與工作負載中的查詢數量成正比）。

We now have to approximate this "raw" histogram using an equidepth histogram with far fewer buckets. The bucket boundaries of the equidepth histogram should be selected such that the probability mass within each bucket is the same. Construction of this equidepth histogram is straightforward and is omitted. At runtime, given a value z, the density can be easily looked up by determining the bucket to which z belongs.
我們現在必須使用一個桶數少得多的等深直方圖來近似這個「原始」直方圖。等深直方圖的桶邊界應該被選擇，使得每個桶內的機率質量相同。這個等深直方圖的建構是直接的，因此在此省略。在執行時，給定一個值 z，可以透過確定 z 所屬的桶來輕鬆查閱密度。

#### 4.4.3 Multitable Databases.
#### 4.4.3 多表資料庫

Another aspect to consider is when the database spans across more than one table. Important multitable scenarios are star/snowflake schemas where fact tables are logically connected to dimension tables via foreign key joins. For example, while the actual homes for sale may be recorded in a fact table, various properties of each home, such as demographics of neighborhood, builder characteristics, etc., may be found in corresponding dimension tables. In this case, we create a logical view representing the join of all these tables—thus this view contains all the attributes of interest—and apply our ranking methodology on this view. As shall be evident later, if we follow the precomputation method of Section 5.2, then there is no need to materialize the logical view, since the execution is then based on the precomputed lists and the logical view would only be accessed at the final stage to output the top results.
另一個需要考慮的方面是當資料庫跨越多個表格時。重要的多表格場景是星型/雪花型結構，其中事實表格透過外鍵連接邏輯上連接到維度表格。例如，雖然待售房屋的實際資訊可能記錄在事實表格中，但每棟房屋的各種屬性，如社區人口統計、建商特徵等，可能可以在相應的維度表格中找到。在這種情況下，我們建立一個代表所有這些表格連接的邏輯視圖——因此該視圖包含所有感興趣的屬性——並對該視圖應用我們的排序方法。稍後將會清楚，如果我們遵循第 5.2 節的預計算方法，那麼就無需具體化邏輯視圖，因為執行是基於預計算的列表，而邏輯視圖只會在最後階段被存取以輸出頂級結果。

ACM Transactions on Database Systems, Vol. 31, No. 3, September 2006.
ACM Transactions on Database Systems, Vol. 31, No. 3, September 2006.

[Image]
[圖片]

Fig. 1. Architecture of ranking system.
圖 1. 排序系統架構。

## 5. IMPLEMENTATION
## 5. 實作

In this section we discuss the architecture and the implementation of our database ranking system.
在本節中，我們將討論我們資料庫排序系統的架構和實作。

### 5.1 General Architecture of our Approach
### 5.1 我們方法的總體架構

Figure 1 shows the architecture of our proposed system for enabling ranking of database query results. As mentioned in the introduction, the main components are the preprocessing component, an intermediate knowledge representation layer in which the ranking functions are encoded and materialized, and a query processing component. The modular and generic nature of our system allows for easy customization of the ranking functions for different applications.
圖 1 展示了我們提出的用於實現資料庫查詢結果排序的系統架構。如緒論所述，主要元件包括預處理元件、一個用於編碼和具體化排序函數的中介知識表示層，以及一個查詢處理元件。我們系統的模組化和通用性使得可以輕鬆地為不同應用客製化排序函數。

### 5.2 Preprocessing
### 5.2 預處理

This component is divided into several modules. First, the Atomic Probabilities Module computes the quantities p(y|W), p(y|D), p(x|y, W), and p(x|y, D) for all distinct values x and y. These quantities are computed by scanning the workload and data, respectively. While the latter two quantities for categorical data can be computed by running a general association rule mining algorithm such as that given in Agrawal et al. [1995] on the workload and data, we instead chose to directly compute all pairwise cooccurrence frequencies by a single scan of the workload and data, respectively. The observed probabilities are then smoothened using the Bayesian m-estimate method [Cestnik 1990]. (We note that more sophisticated Bayesian methods that use an informative prior may be employed instead.) For numeric attributes, we compute
此元件分為數個模組。首先，原子機率模組計算所有不同值 x 和 y 的 p(y|W)、p(y|D)、p(x|y, W) 和 p(x|y, D) 等量。這些量分別透過掃描工作負載和資料來計算。雖然後兩個量對於分類資料可以透過在工作負載和資料上執行通用的關聯規則挖掘演算法（例如 Agrawal 等人 [1995] 中給出的演算法）來計算，但我們選擇透過分別對工作負載和資料進行單次掃描來直接計算所有成對共現頻率。然後使用貝氏 m-估計方法 [Cestnik 1990] 對觀察到的機率進行平滑處理。（我們注意到，可以改用使用資訊性先驗的更複雜的貝氏方法。）對於數值屬性，我們計算

ACM Transactions on Database Systems, Vol. 31, No. 3, September 2006.
ACM Transactions on Database Systems, Vol. 31, No. 3, September 2006.

p(y|W), p(y|D), p(x|y, W), and p(x|y, D) as histograms, as described in Section 4.4.2.
p(y|W)、p(y|D)、p(x|y, W) 和 p(x|y, D) 作為直方圖，如第 4.4.2 節所述。

These atomic probabilities are stored as database tables in the intermediate knowledge representation layer, with appropriate indexes to enable easy retrieval. In particular, p(y|W) and p(y|D) are, respectively, stored in two tables, each with columns {AttName, AttVal, Prob} and with a composite B+ tree index on (AttName, AttVal), while p(x|y, W)and p(x|y, D), respectively, are stored in two tables, each with columns {AttNameLeft, AttValLeft, AttNameRight, AttValRight, Prob} and with a composite B+ tree index on (AttNameLeft, AttValLeft, AttNameRight, AttValRight). For numeric quantities, attribute values are essentially the ranges of the corresponding buckets. These atomic quantities can be further customized by human experts if necessary.
這些原子機率以資料庫表格的形式儲存在中介知識表示層，並帶有適當的索引以便於檢索。特別是，p(y|W) 和 p(y|D) 分別儲存在兩個表格中，每個表格都有 {AttName, AttVal, Prob} 欄位，並在 (AttName, AttVal) 上建有複合 B+ 樹索引；而 p(x|y, W) 和 p(x|y, D) 則分別儲存在兩個表格中，每個表格都有 {AttNameLeft, AttValLeft, AttNameRight, AttValRight, Prob} 欄位，並在 (AttNameLeft, AttValLeft, AttNameRight, AttValRight) 上建有複合 B+ 樹索引。對於數值量，屬性值基本上是相應桶的範圍。如有必要，這些原子量可以由人類專家進一步客製化。

This intermediate layer now contains enough information for computing the ranking function, and a naïve query processing algorithm (henceforth referred to as the Scan algorithm) can indeed be designed, which, for any query, first selects the tuples that satisfy the query condition, then scans and computes the score for each such tuple using the information in this intermediate layer, and finally returns the top-k tuples. However, such an approach can be inefficient for the Many-Answers problem, since the number of tuples satisfying the query condition can be very large. At the other extreme, we could precompute the top-k tuples for all possible queries (i.e., for all possible sets of values X), and, at query time, simply return the appropriate result set. Of course, due to the combinatorial explosion, this is infeasible in practice.
這個中介層現在包含了足夠的資訊來計算排序函數，並且確實可以設計一個簡單的查詢處理演算法（以下稱為掃描演算法），該演算法對於任何查詢，首先選擇滿足查詢條件的元組，然後使用此中介層中的資訊掃描並計算每個此類元組的分數，最後返回前 k 個元組。然而，對於「多答案問題」，這種方法可能效率低下，因為滿足查詢條件的元組數量可能非常大。==在另一個極端，我們可以為所有可能的查詢（即，為所有可能的值集 X）預先計算前 k 個元組，然後在查詢時，簡單地返回適當的結果集。當然，由於組合爆炸，這在實務中是不可行的==。

We thus pose the question: how can we appropriately trade off between preprocessing and query processing, that is, what additional yet reasonable precomputations are possible that can enable faster query-processing algorithms than Scan? (We note that tradeoffs between preprocessing and query processing techniques are common in IR systems [Grossman and Frieder 2004].)
因此，我們提出一個問題：我們如何在預處理和查詢處理之間進行適當的權衡，也就是說，有哪些額外但合理的預計算是可能的，可以實現比掃描更快的查詢處理演算法？（我們注意到，預處理和查詢處理技術之間的權衡在資訊檢索系統中很常見 [Grossman and Frieder 2004]。）

The high-level intuition behind our approach to the above problem is as follows. Instead of precomputing the top-k tuples for all possible queries, we precompute ranked lists of the tuples for all possible atomic queries each distinct value x in the table defines an atomic query Qx that specifies the single value {x}. For example, "SELECT * FROM HOMES WHERE CITY=Kirkland" is an atomic query. Then at query time, given an actual query that specifies a set of values X, we “merge" the ranked lists corresponding to each x in X to compute the final top-k tuples.
我們解決上述問題的方法，其高層次的直覺如下。我們不為所有可能的查詢預先計算前 k 個元組，而是為所有可能的原子查詢預先計算元組的排序列表，其中表格中的每個不同值 x 都定義了一個指定單一值 {x} 的原子查詢 Qx。例如，「SELECT * FROM HOMES WHERE CITY=Kirkland」就是一個原子查詢。然後在查詢時，給定一個指定值集合 X 的實際查詢，我們「合併」對應於 X 中每個 x 的排序列表，以計算最終的前 k 個元組。

This high-level idea is conceptually related to the merging of inverted lists in IR. However, our main challenge is to be able to perform the merging without having to scan any of the ranked lists in its entirety. One idea would be to try and adapt well-known top-k algorithms such as the Threshold Algorithm (TA) and its derivatives [Bruno et al. 2002b; Fagin 1998; Fagin et al. 2001; Güntzer et al. 2000; Nepal and Ramakrishna 1999] for this problem. However, it is not immediately obvious how a feasible adaptation can be easily accomplished. For example, it is especially critical to keep the number of sorted streams (an access mechanism required by TA) small, as it is well known that TA's performance rapidly deteriorates as this number increases. Upon examination of our ranking function in Equation (4) (which involves all attribute values of the tuple, and not
這個高層次的想法在概念上與資訊檢索中倒排列表的合併有關。然而，我們的主要挑戰是能夠在不完全掃描任何排序列表的情況下執行合併。一個想法是嘗試並調整著名的 top-k 演算法，例如閾值演算法（TA）及其衍生演算法 [Bruno et al. 2002b; Fagin 1998; Fagin et al. 2001; Güntzer et al. 2000; Nepal and Ramakrishna 1999] 來解決這個問題。然而，如何輕鬆地實現可行的調整並非顯而易見。例如，保持排序流的數量（TA 所需的存取機制）較少尤為關鍵，因為眾所周知，隨著這個數量的增加，TA 的效能會迅速惡化。在檢查我們在方程式（4）中的排序函數（它涉及元組的所有屬性值，而不僅僅是

ACM Transactions on Database Systems, Vol. 31, No. 3, September 2006.
ACM Transactions on Database Systems, Vol. 31, No. 3, September 2006.

Fig. 2. The Index Module.
圖 2. 索引模組。

just the specified values), the number of sorted streams in any naïve adaptation of TA would depend on the total number of attributes in the database, which would cause major performance problems.
僅指定的值），任何對 TA 的天真改編中排序流的數量都將取決於資料庫中屬性的總數，這將導致嚴重的效能問題。

In what follows, we show how to precompute data structures that indeed enable us to efficiently adapt TA for our problem. At query time, we do a TA-like merging of several ranked lists (i.e., of sorted streams). However, the required number of sorted streams depends only on s and not on m (s is the number of specified attribute values in the query, while m is the total number of attributes in the database). We emphasize that such a merge operation is only made possible due to the specific functional form of our ranking function resulting from our limited independence assumptions, as discussed in Section 4.2.1. It is unlikely that TA can be adapted, at least in a feasible manner, for ranking functions that rely on more comprehensive dependency models of the data.
接下來，我們將展示如何預先計算資料結構，使我們能夠有效地將 TA 應用於我們的問題。在查詢時，我們對幾個排序列表（即排序流）進行類似 TA 的合併。然而，所需的排序流數量僅取決於 s 而非 m（s 是查詢中指定屬性值的數量，而 m 是資料庫中屬性的總數）。我們強調，這種合併操作之所以成為可能，僅僅是因為我們在第 4.2.1 節中討論的有限獨立性假設所產生的排序函數的特定函數形式。對於依賴更全面的資料依賴模型的排序函數，TA 至少在可行的方式下不太可能被調整。

We next give the details of these data structures. They are precomputed by the Index Module of the preprocessing component. This module (see Figure 2 for the algorithm) takes as inputs the association rules and the database, and, for every distinct value x, creates two lists Cx and Gx, each containing the tuple-ids of all data tuples that contain x, ordered in specific ways. These two lists are defined as follows:
接下來我們給出這些資料結構的細節。它們由預處理元件的索引模組預先計算。該模組（演算法見圖 2）以關聯規則和資料庫為輸入，並為每個不同的值 x 創建兩個列表 Cx 和 Gx，每個列表包含所有包含 x 的資料元組的元組 ID，並以特定方式排序。這兩個列表定義如下：

(1) Conditional list Cx: This list consists of pairs of the form <TID,CondScore>, ordered by descending CondScore, where TID is the tuple-id of a tuplet that contains x and
(1) 條件列表 Cx：此列表由 <TID,CondScore> 形式的配對組成，按 CondScore 降序排列，其中 TID 是包含 x 的元組的元組 ID，且

CondScore = Πz∈t p(x|z, W) / p(x|z, D)
CondScore = Πz∈t p(x|z, W) / p(x|z, D)

where z ranges over all attribute values of t.
其中 z 遍歷 t 的所有屬性值。

(2) Global list Gx: This list consists of pairs of the form <TID, GlobScore>, ordered by descending GlobScore, where TID is the tuple-id of a tuple t
(2) 全域列表 Gx：此列表由 <TID, GlobScore> 形式的配對組成，按 GlobScore 降序排列，其中 TID 是元組 t 的元組 ID

ACM Transactions on Database Systems, Vol. 31, No. 3, September 2006.
ACM Transactions on Database Systems, Vol. 31, No. 3, September 2006.

that contains x and
包含 x 且

GlobScore = Πz∈t p(z|W) / p(z|D)
GlobScore = Πz∈t p(z|W) / p(z|D)

These lists enable efficient computation of the score of a tuple t for any query as follows: given query Q specifying conditions for a set of attribute values, say X = {x1,.., xs}, at query time we retrieve and multiply the scores of t in the lists Cx1,..., Cxs and in one of Gx1,...,Gxs. This requires only s + 1 multiplications and results in a score³ that is proportional to the actual score. Clearly this is more efficient than computing the score "from scratch" by retrieving the relevant atomic probabilities from the intermediate layer and composing them appropriately.
這些列表可以有效地計算任何查詢下元組 t 的分數，方法如下：給定一個查詢 Q，它指定了一組屬性值的條件，例如 X = {x1,.., xs}，在查詢時，我們檢索並乘以 t 在列表 Cx1,..., Cxs 和 Gx1,...,Gxs 其中之一中的分數。這只需要 s + 1 次乘法，就能得到一個與實際分數成正比的分數³。顯然，這比從中介層檢索相關的原子機率並適當地組合它們來「從頭開始」計算分數更有效率。

We need to enable two kinds of access operations efficiently on these lists. First, given a value x, it should be possible to perform a GetNextTID operation on lists Cx and Gx in constant time, that is, the tuple-ids in the lists should be efficiently retrievable one by one in order of decreasing score. This corresponds to the sorted stream access of TA. Second, it should be possible to perform random access on the lists, that is, given a TID, the corresponding score (CondScore or GlobScore) should be retrievable in constant time. To enable these operations efficiently, we materialize these lists as database tables-all the conditional lists are maintained in one table called CondList (with columns {AttName, AttVal, TID, CondScore}), while all the global lists are maintained in another table called GlobList (with columns {AttName, AttVal, TID, GlobScore}). The tables have composite B+ tree indices on (AttName, AttVal, CondScore) and (AttName, AttVal, GlobScore), respectively. This enables efficient performance of both access operations. Further details of how these data structures and their access methods are used in query processing are discussed in Section 5.3.
我們需要能夠有效地對這些列表執行兩種存取操作。首先，給定一個值 x，應該能夠在常數時間內對列表 Cx 和 Gx 執行 GetNextTID 操作，也就是說，列表中的元組 ID 應該能夠按照分數遞減的順序有效地逐一檢索。這對應於 TA 的排序流存取。其次，應該能夠對列表執行隨機存取，也就是說，給定一個 TID，應該能夠在常數時間內檢索到相應的分數（CondScore 或 GlobScore）。為了有效地實現這些操作，我們將這些列表具體化為資料庫表格——所有條件列表都維護在一個名為 CondList 的表格中（欄位為 {AttName, AttVal, TID, CondScore}），而所有全域列表都維護在另一個名為 GlobList 的表格中（欄位為 {AttName, AttVal, TID, GlobScore}）。這些表格分別在 (AttName, AttVal, CondScore) 和 (AttName, AttVal, GlobScore) 上建有複合 B+ 樹索引。這使得兩種存取操作都能高效執行。關於這些資料結構及其存取方法如何在查詢處理中使用的更多細節，將在第 5.3 節中討論。

#### 5.2.1 Presence of Functional Dependencies.
#### 5.2.1 功能相依性的存在

If we consider functional dependencies, then the content of the conditional and global lists is changed as follows.
如果我們考慮功能性依賴，那麼條件列表和全域列表的內容將如下變更。

CondScore = Πz∈t' p(x|z, W) * Πz∈t' 1/p(x|z, D), if x ∈ A', otherwise Πz∈t' p(x|z, W)
CondScore = Πz∈t' p(x|z, W) * Πz∈t' 1/p(x|z, D), 如果 x ∈ A', 否則 Πz∈t' p(x|z, W)

and
與

GlobScore = Πz∈t' p(z|W) * Πz∈t' 1/p(z|D)
GlobScore = Πz∈t' p(z|W) * Πz∈t' 1/p(z|D)

where A' = {Ai ∈ A | ¬∃Aj ∈ A, FD : Aj → Ai} and t' is the subset of the attribute values of t that belong to A'.
其中 A' = {Ai ∈ A | ¬∃Aj ∈ A, FD : Aj → Ai} 且 t' 是屬於 A' 的 t 的屬性值子集。

³ This score is proportional, but not equal, to the actual score because it contains extra factors of the form p(x|z, W)/p(x|z, D), where z ∈ X. However, these extra factors are common to all selected tuples; hence the rank order is unchanged.
³ 這個分數與實際分數成正比，但不相等，因為它包含了 p(x|z, W)/p(x|z, D) 形式的額外因子，其中 z ∈ X。然而，這些額外因子對於所有被選中的元組都是相同的；因此排名順序不變。

ACM Transactions on Database Systems, Vol. 31, No. 3, September 2006.
ACM Transactions on Database Systems, Vol. 31, No. 3, September 2006.

Fig. 3. The List Merge Algorithm.
圖 3. 列表合併演算法。

### 5.3 Query Processing
### 5.3 查詢處理

In this subsection we describe the query processing component. The naïve Scan algorithm has already been described in Section 5.2, so our focus here is on the alternate List Merge algorithm (see Figure 3). This is an adaptation of TA, whose efficiency crucially depends on the data structures pre-computed by the Index Module.
在本小節中，我們將描述查詢處理元件。簡單的掃描演算法已在第 5.2 節中描述，因此我們這裡的重點是替代的列表合併演算法（見圖 3）。這是 TA 的一種改編，其效率關鍵取決於索引模組預先計算的資料結構。

The List Merge algorithm operates as follows. Given a query Q specifying conditions for a set X = {x1, .., xs} of attributes, we execute TA on the following s+1 lists: Cx1,...,Cxs, and Gxb, where Gxb is the shortest list among Gx1,...,Gxs (in principle, any list from Gx1,...,Gxs would do, but the shortest list is likely to be more efficient). During each iteration, the TID with the next largest score is retrieved from each list using sorted access. Its score in every other list is retrieved via random access, and all these retrieved scores are multiplied together, resulting in the final score of the tuple (which, as mentioned in Section 5.2, is proportional to the actual score derived in Equation 4). The termination criterion guarantees that no more GetNextTID operations will be needed on any of the lists. This is accomplished by maintaining an array T which contains the last scores read from all the lists at any point in time by GetNextTID operations. The product of the scores in T represents the score of the very best tuple we can hope to find in the data that is yet to be seen. If this value is no more than the tuple in the top-k buffer with the smallest score, the algorithm successfully terminates.
列表合併演算法的運作方式如下。給定一個查詢 Q，它指定了一組屬性 X = {x1, .., xs} 的條件，我們在以下 s+1 個列表上執行 TA：Cx1,...,Cxs，以及 Gxb，其中 Gxb 是 Gx1,...,Gxs 中最短的列表（原則上，Gx1,...,Gxs 中的任何列表都可以，但最短的列表可能更有效率）。在每次迭代中，使用排序存取從每個列表中檢索具有下一個最大分數的 TID。它在其他每個列表中的分數透過隨機存取檢索，然後將所有這些檢索到的分數相乘，得到元組的最終分數（如第 5.2 節所述，該分數與方程式 4 中推導的實際分數成正比）。終止準則保證在任何列表上都不再需要 GetNextTID 操作。這是透過維護一個陣列 T 來實現的，該陣列包含在任何時間點透過 GetNextTID 操作從所有列表中讀取的最後分數。T 中分數的乘積代表了我們希望在尚未看到的資料中找到的最好元組的分數。如果此值不大於 top-k 緩衝區中分數最小的元組，則演算法成功終止。

#### 5.3.1 Limited Available Space.
#### 5.3.1 可用空間有限

So far we have assumed that there is enough space available to build the conditional and global lists. A simple
到目前為止，我們一直假設有足夠的可用空間來建立條件列表和全域列表。一個簡單的

ACM Transactions on Database Systems, Vol. 31, No. 3, September 2006.
ACM Transactions on Database Systems, Vol. 31, No. 3, September 2006.

analysis indicates that the space consumed by these lists is O(mn) bytes (m is the number of attributes and n the number of tuples of the database table). However, there may be applications where space is an expensive resource (e.g., when lists should preferably be held in memory and compete for that space or even for space in the processor cache hierarchy). We show that, in such cases, we can store only a subset of the lists at preprocessing time, at the expense of an increase in the query processing time.
分析表明，這些列表消耗的空間為 O(mn) 位元組（m 是屬性的數量，n 是資料庫表格的元組數量）。然而，在某些應用中，空間可能是一種昂貴的資源（例如，當列表最好保存在記憶體中並爭奪該空間，甚至爭奪處理器快取層次結構中的空間時）。我們證明，在這種情況下，我們可以在預處理時只儲存列表的一個子集，但代價是查詢處理時間的增加。

Determining which lists to retain/omit at preprocessing time may be accomplished by analyzing the workload. A simple solution is to store the conditional lists Cx and the corresponding global lists Gx only for those attribute values x that occur most frequently in the workload. At query time, since the lists of some of the specified attributes may be missing, the intuitive idea is to probe the intermediate knowledge representation layer (where the “relatively raw" data is maintained, i.e., the atomic probabilities) and directly compute the missing information. More specifically, we use a modification of TA described in Bruno et al. [2002b], where not all sources have sorted stream access.
在預處理時決定保留或省略哪些列表，可以透過分析工作負載來完成。一個簡單的解決方案是，只為那些在工作負載中出現最頻繁的屬性值 x 儲存條件列表 Cx 和相應的全域列表 Gx。在查詢時，由於某些指定屬性的列表可能缺失，直觀的想法是探測中介知識表示層（其中維護著「相對原始」的資料，即原子機率）並直接計算缺失的資訊。更具體地說，我們使用了 Bruno 等人 [2002b] 中描述的 TA 的一個修改版本，其中並非所有來源都具有排序流存取。

### 5.4 Evaluating IN and Range Queries
### 5.4 評估 IN 和範圍查詢

As mentioned in Section 4.4.1, executing IN queries is more involved because each result tuple has possibly different specified values. This makes the application of the List Merge algorithm more challenging, since the Scan algorithm computes the score of each result tuple from the information in this intermediate layer. In particular, List Merge is complicated in two ways:
如第 4.4.1 節所述，執行 IN 查詢更為複雜，因為每個結果元組可能具有不同的指定值。這使得列表合併演算法的應用更具挑戰性，因為掃描演算法是根據此中介層的資訊計算每個結果元組的分數。特別是，列表合併在兩個方面變得複雜：

(a) We cannot use a single conditional list for a specified attribute with an IN condition, since a single conditional list only contains tuples containing a single attribute values. For example, for the query "City IN (Redmond, Bellevue)" we must merge the conditional lists CRedmond and CBellevue.
(a) 我們不能對具有 IN 條件的指定屬性使用單一條件列表，因為單一條件列表只包含具有單一屬性值的元組。例如，對於查詢「City IN (Redmond, Bellevue)」，我們必須合併條件列表 CRedmond 和 CBellevue。

(b) More seriously, we can no longer use a single conditional Cx list for a specified attribute Xi (with or without an IN condition), if there is another specified attribute Xj with an IN condition. The reason is that the product Πz∈t p(x|z,W) / p(x|z,D) stored in Cx (x is an attribute value for attribute Xi) spans across all attribute values of t and not only across the unspecified attribute values Y as required by Equation (8). This was not a problem for the case of point queries (Equations (4) and (5)) because the factors p(x|z,W)/p(x|z,D), where z ∈ X of the above product, are common for all result-tuples, and hence the scores are multiplied by a common constant. On the other hand, if there is an attribute Xj with IN condition, then the factor p(xi|z,W)/p(xi|z,D), where z is an attribute value for Xj, is not common and hence cannot be ignored.
(b) 更嚴重的是，如果存在另一個具有 IN 條件的指定屬性 Xj，我們就不能再對指定的屬性 Xi（無論是否帶有 IN 條件）使用單一的條件 Cx 列表。原因在於，儲存在 Cx 中的乘積 Πz∈t p(x|z,W) / p(x|z,D)（x 是屬性 Xi 的一個屬性值）涵蓋了 t 的所有屬性值，而不僅僅是方程式 (8) 所要求的未指定屬性值 Y。這對於點查詢的情況（方程式 (4) 和 (5)）不是問題，因為上述乘積的因子 p(x|z,W)/p(x|z,D)（其中 z ∈ X）對於所有結果元組都是相同的，因此分數乘以一個公共常數。另一方面，如果存在一個具有 IN 條件的屬性 Xj，那麼因子 p(xi|z,W)/p(xi|z,D)（其中 z 是 Xj 的一個屬性值）就不是公共的，因此不能被忽略。

To overcome these challenges, we split each IN query to a set of point queries, which are evaluated as usual and then their results are merged. In particular, suppose we have the IN query Q: “X1 IN (x1,1 ... x1,r1) and . . . and Xs IN (xs,1 ... xs,rs).” First we split Q into r1r2...rs point queries, one for each combination of selecting a single value from each specified attribute. Then these point queries are evaluated separately and their results (along with their scores) are merged. To see that such a splitting approach yields the correct results, note
為了克服這些挑戰，我們將每個 IN 查詢拆分成一組點查詢，這些點查詢照常評估，然後將其結果合併。特別是，假設我們有 IN 查詢 Q：「X1 IN (x1,1 ... x1,r1) and . . . and Xs IN (xs,1 ... xs,rs)」。首先，我們將 Q 拆分成 r1r2...rs 個點查詢，每個點查詢對應於從每個指定屬性中選擇單一值的組合。然後，這些點查詢被單獨評估，其結果（連同其分數）被合併。要了解這種拆分方法能產生正確的結果，請注意

ACM Transactions on Database Systems, Vol. 31, No. 3, September 2006.
ACM Transactions on Database Systems, Vol. 31, No. 3, September 2006.

that the first (global) part of the ranking function in Equation (8) is the same for both the point and the IN query and is equal to the scores in the Global Lists. The conditional part of Equation (8) only depends on the values of the tuple t and the set of specified attributes but not on the particular conditions of the query. Hence, the point queries will assign the same scores as the IN query. Finally, it should be clear that the same set of tuples is returned as results in both cases.
方程式 (8) 中排序函數的第一個（全域）部分對於點查詢和 IN 查詢是相同的，並且等於全域列表中的分數。方程式 (8) 的條件部分僅取決於元組 t 的值和指定的屬性集，而不取決於查詢的特定條件。因此，點查詢將分配與 IN 查詢相同的分數。最後，應該清楚的是，在這兩種情況下，返回的元組集是相同的。

The splitting method is efficient only if a relatively small number of point queries results from the split, that is, if r1r2 . . . rs is small. The key advantage of this approach is that no additional conditional lists need to be created to support IN queries. An alternate approach described next is preferable when the IN conditions frequently involve the same small set of attributes. We illustrate this idea through an example. Suppose queries specifying IN condition only on the City attribute are popular. Then, we create a new conditional list CCity for every attribute value x not in the City attribute, using the formula CondScore = Πz∈(t−t.City) p(x|z, W) / p(x|z, D), and use these conditional lists whenever a query with an IN condition only on City is submitted.
分割方法只有在分割產生的點查詢數量相對較少時才有效，也就是說，如果 r1r2 . . . rs 很小。這種方法的主要優點是不需要創建額外的條件列表來支援 IN 查詢。當 IN 條件頻繁地涉及相同的小屬性集時，接下來描述的替代方法更可取。我們透過一個例子來說明這個想法。假設僅在 City 屬性上指定 IN 條件的查詢很受歡迎。然後，我們為 City 屬性中沒有的每個屬性值 x 創建一個新的條件列表 CCity，使用公式 CondScore = Πz∈(t−t.City) p(x|z, W) / p(x|z, D)，並在提交僅在 City 上有 IN 條件的查詢時使用這些條件列表。

Finally, note that range queries—that is, queries with ranges on numeric attributes-may be evaluated using techniques similar to queries with IN conditions. For example, if a condition such as “A BETWEEN (x1, x2)" is specified, then this condition is discretized into an IN condition by replacing the range with buckets from the precomputed histogram p(x|W) that overlap with the range. In case the range only partially overlaps with the leading/trailing buckets, the retrieved tuples that do not satisfy the query condition are discarded in a final filtering phase.
最後，請注意，範圍查詢——即在數值屬性上有範圍的查詢——可以使用類似於帶有 IN 條件的查詢的技術進行評估。例如，如果指定了諸如「A BETWEEN (x1, x2)」之類的條件，則透過將範圍替換為與該範圍重疊的預計算直方圖 p(x|W) 中的桶，將此條件離散化為 IN 條件。如果範圍僅部分與前導/後隨桶重疊，則在最終的過濾階段會丟棄不滿足查詢條件的檢索元組。

## 6. EXPERIMENTS
## 6. 實驗

In this section we report on the results of an experimental evaluation of our ranking method as well as some of the competitors. We evaluated both the quality of the rankings obtained, as well as the performance of the various approaches. We mention at the outset that preparing an experimental setup for testing ranking quality was extremely challenging, as unlike IR, there are no standard benchmarks available, and we had to conduct user studies to evaluate the rankings produced by the various algorithms.
在本節中，我們報告了我們的排序方法以及一些競爭方法的實驗評估結果。我們評估了所獲得排序的品質以及各種方法的效能。我們在一開始就提到，準備一個用於測試排序品質的實驗設置極具挑戰性，因為與資訊檢索不同，沒有可用的標準基準，我們必須進行使用者研究來評估各種演算法產生的排序。

For our evaluation, we used real datasets from two different domains. The first domain was the MSN HomeAdvisor database (http://houseandhome.msn.com/), from which we prepared a table of homes for sale in the U.S., with a mix of categorical as well as numeric attributes such as Price, Year, City, Bedrooms, Bathrooms, Sqft, Garage, etc. The original database table also had a text column called Remarks, which contained descriptive information about the home. From this column, we extracted additional Boolean attributes such as Fireplace, View, Pool, etc. To evaluate the role of the size of the database, we also performed experiments on a subset of the HomeAdvisor database, consisting only of homes sold in the Seattle area.
為了我們的評估，我們使用了來自兩個不同領域的真實資料集。第一個領域是 ==MSN HomeAdvisor 資料庫== (http://houseandhome.msn.com/)，我們從中準備了一份美國待售房屋的表格，其中混合了分類和數值屬性，例如價格、年份、城市、臥室、浴室、面積、車庫等。原始資料庫表格還有一個名為「備註」的文本欄位，其中包含有關房屋的描述性資訊。我們從該欄位中提取了額外的布林屬性，例如壁爐、景觀、游泳池等。為了評估資料庫大小的作用，我們還在 HomeAdvisor 資料庫的一個子集上進行了實驗，該子集僅包含在西雅圖地區出售的房屋。

The second domain was the Internet Movie Database (http://www.imdb.com), from which we prepared a table of movies, with attributes such
第二個領域是==網路電影資料庫==（http://www.imdb.com），我們從中準備了一份電影表格，屬性包括

ACM Transactions on Database Systems, Vol. 31, No. 3, September 2006.
ACM Transactions on Database Systems, Vol. 31, No. 3, September 2006.

| Table | NumTuples | Database Size (MB) |
| :--- | :--- | :--- |
| Seattle Homes | 17463 | 1.936 |
| U.S. Homes | 1380762 | 140.432 |
| Movies | 1446 | Less than 1 |
Table I. Sizes of Datasets
表格 I. 資料集大小

as Title, Year, Genre, Director, FirstActor, SecondActor, Certificate, Sound, Color, etc. We first selected a set of movies by the 30 most prolific actors for our experiments. From this we removed the 250 most well-known movies, as we did not wish our users to be biased with information they already might know about these movies, especially information that is not captured by the attributes that we had selected for our experiments.
例如標題、年份、類型、導演、第一主角、第二主角、證書、音效、色彩等。我們首先為我們的實驗挑選了由 30 位最多產的演員主演的一組電影。我們從中移除了 250 部最知名的電影，因為我們不希望我們的使用者被他們可能已經知道的關於這些電影的資訊所偏見，特別是那些未被我們為實驗選擇的屬性所捕捉的資訊。

The sizes of the various (single-table) datasets used in our experiments are shown in Table I. The quality experiments were conducted on the Seattle Homes and Movies tables, while the performance experiments were conducted on the Seattle Homes and the U.S. Homes tables—we omitted performance experiments on the Movies table on account of its small size. We used Microsoft SQL Server 2000 RDBMS on a P4 2.8-GHz PC with 1 GB of RAM for our experiments. We implemented all algorithms in C#, and connected to the RDBMS through DAO. We created single-attribute indices on all table attributes, to be used during the selection phase of the Scan algorithm. Note that these indices are not used by the List Merge algorithm.
我們實驗中使用的各種（單表）資料集的大小如表 I 所示。品質實驗在 Seattle Homes 和 Movies 表上進行，而效能實驗在 Seattle Homes 和 U.S. Homes 表上進行——由於 Movies 表的大小很小，我們省略了對其的效能實驗。我們的實驗使用 Microsoft SQL Server 2000 RDBMS，在配備 1 GB RAM 的 P4 2.8-GHz PC 上進行。我們用 C# 實作了所有演算法，並透過 DAO 連接到 RDBMS。我們在所有表屬性上創建了單屬性索引，以供掃描演算法的選擇階段使用。請注意，列表合併演算法不使用這些索引。

### 6.1 Quality Experiments
### 6.1 品質實驗

We evaluated the quality of three different ranking methods: (a) our ranking method, henceforth referred to as Conditional; (b) the ranking method described in Agrawal et al. [2003], henceforth known as Global; and (c) a baseline Random algorithm, which simply ranks and returns the top-k tuples in arbitrary order. This evaluation was accomplished using surveys involving 14 employees of Microsoft Research.
我們評估了三種不同排序方法的品質：(a) 我們的排序方法，以下稱為 Conditional；(b) Agrawal 等人 [2003] 中描述的排序方法，以下稱為 Global；以及 (c) 一個基準的隨機演算法，它只是以任意順序排序並返回前 k 個元組。此評估是透過涉及 14 名微軟研究院員工的調查完成的。

For the Seattle Homes table, we first created several different profiles of home buyers, for example, young dual-income couples, singles, middle-class family who like to live in the suburbs, rich retirees, etc. Then, we collected a workload from our users by requesting them to behave like these home buyers and post queries against the database-for example, a middle-class home-buyer with children looking for a suburban home would post a typical query such as "Bedrooms=4 and Price=Moderate and SchoolDistrict=Excellent." We collected several hundred queries by this process, each typically specifying two to four attributes. We then trained our ranking algorithm on this workload.
對於 Seattle Homes 表格，我們首先創建了幾種不同的購房者畫像，例如，年輕的雙收入夫婦、單身人士、喜歡住在郊區的中產階級家庭、富有的退休人員等。然後，我們透過要求我們的使用者扮演這些購房者並對資料庫發出查詢來收集工作負載——例如，一個有孩子、尋找郊區房屋的中產階級購房者會發出一個典型的查詢，如「臥室=4 且價格=中等且學區=優秀」。我們透過這個過程收集了數百個查詢，每個查詢通常指定二到四個屬性。然後，我們用這個工作負載來訓練我們的排序演算法。

We prepared a similar experimental setup for the Movies table. We first created several different profiles of moviegoers, for example, teenage males wishing to see action thrillers, people interested in comedies from the 1980s, etc. We disallowed users from specifying the movie title in the queries, as the title is a key of the table. As with homes, here too we collected several hundred workload queries, and trained our ranking algorithm on this workload.
我們為電影資料表準備了類似的實驗設置。我們首先創建了幾種不同的電影觀眾畫像，例如，想看動作驚悚片的青少年男性、對 1980 年代喜劇感興趣的人等等。我們不允許使用者在查詢中指定電影標題，因為標題是資料表的關鍵。與房屋一樣，我們在這裡也收集了數百個工作負載查詢，並在此工作負載上訓練了我們的排序演算法。

ACM Transactions on Database Systems, Vol. 31, No. 3, September 2006.
ACM Transactions on Database Systems, Vol. 31, No. 3, September 2006.

We first describe a few sample results informally, and then present a more formal evaluation of our rankings.
我們首先非正式地描述一些範例結果，然後對我們的排名進行更正式的評估。

#### 6.1.1 Examples of Ranking Results.
#### 6.1.1 排序結果範例

For the Seattle Homes dataset, both Conditional as well as Global produced rankings that were intuitive and reasonable. There were interesting examples where Conditional produced rankings that were superior to Global. For example, for a query with condition "City=Seattle and Bedroom=1," Conditional ranked condos with garages the highest. Intuitively, this is because private parking in downtown is usually very scarce, and condos with garages are highly sought after. However, Global was unable to recognize the importance of garages for this class of homebuyers, because most users (i.e., over the entire workload) do not explicitly request for garages since most homes have garages. As another example, for a query such as "Bedrooms=4 and City=Kirkland and Price=Expensive," Conditional ranked homes with waterfront views the highest, whereas Global ranked homes in good school districts the highest. This is as expected, because for very rich homebuyers a waterfront view is perhaps a more desirable feature than a good school district, even though the latter may be globally more popular across all homebuyers.
對於西雅圖房屋資料集，Conditional 和 Global 產生的排名都直觀且合理。有一些有趣的例子顯示 Conditional 產生的排名優於 Global。例如，對於條件為「城市=西雅圖且臥室=1」的查詢，Conditional 將帶車庫的公寓排在最高。直觀地說，這是因為市中心的私人停車位通常非常稀缺，帶車庫的公寓非常搶手。然而，Global 無法識別車庫對這類購房者的重要性，因為大多數使用者（即在整個工作負載中）不會明確要求車庫，因為大多數房屋都有車庫。再舉一個例子，對於諸如「臥室=4 且城市=柯克蘭且價格=昂貴」的查詢，Conditional 將有水景的房屋排在最高，而 Global 將位於好學區的房屋排在最高。這符合預期，因為對於非常富有的購房者來說，水景可能比好學區更具吸引力，即使後者可能在所有購房者中更受歡迎。

Likewise, for the Movies dataset, Conditional often produced rankings that were superior to Global. For example, for a query such as "Year=1980s and Genre=Thriller," Conditional ranked movies such as Indiana Jones and the Temple of Doom higher than Commando, because the workload indicated that Harrison Ford was a better-known actor than Arnold Schwarzenegger during that era, although the latter actor was globally more popular over the entire workload.
同樣地，對於電影資料集，Conditional 產生的排名通常優於 Global。例如，對於像「年份=1980年代且類型=驚悚片」這樣的查詢，Conditional 將《印第安納瓊斯與魔宮傳奇》等電影的排名高於《魔鬼司令》，因為工作負載顯示，在那個時代，哈里遜·福特是比阿諾·史瓦辛格更知名的演員，儘管後者在整個工作負載中全球更受歡迎。

As for Random, it produced quite irrelevant results in most cases.
至於 Random，它在大多數情況下產生了相當不相關的結果。

#### 6.1.2 Ranking Evaluation.
#### 6.1.2 排名評估

We now present a more formal evaluation of the ranking quality produced by the ranking algorithms. We conducted two surveys; the first compared the rankings against user rankings using standard precision/recall metrics, while the second was a simpler survey that asked users to rate which algorithm's rankings they preferred.
我們現在對排序演算法產生的排序品質進行更正式的評估。我們進行了兩項調查；第一項使用標準的精確率/召回率指標將排序與使用者排序進行比較，而第二項是一個更簡單的調查，要求使用者評價他們更喜歡哪個演算法的排序。

##### 6.1.2.1 First Survey.
##### 6.1.2.1 第一次調查

Since requiring users to rank the entire database for each query for the first survey would have been extremely tedious, we used the following strategy. For each dataset, for each test query Qi we generated a set Hi of 30 tuples likely to contain a good mix of relevant and irrelevant tuples to the query. We did this by mixing the top-10 results of both the Conditional and Global ranking algorithms, removing ties, and adding a few randomly selected tuples. Finally, we presented the queries along with their corresponding Hi's (with tuples randomly permuted) to each user in our study. Each user's responsibility was to mark 10 tuples in Hi as most relevant to the query Qi. We then measured how closely the 10 tuples marked as relevant by the user (i.e., the "ground truth") matched the 10 tuples returned by each algorithm.
由於要求使用者在第一次調查中為每個查詢對整個資料庫進行排名會非常繁瑣，我們採用了以下策略。對於每個資料集，對於每個測試查詢 Qi，我們產生了一個包含 30 個元組的集合 Hi，這些元組可能包含與查詢相關和不相關的元組的良好混合。我們透過混合 Conditional 和 Global 排序演算法的前 10 個結果，移除重複項，並添加一些隨機選擇的元組來做到這一點。最後，我們將查詢及其對應的 Hi（元組隨機排列）呈現給我們研究中的每個使用者。每個使用者的責任是在 Hi 中標記 10 個與查詢 Qi 最相關的元組。然後，我們測量使用者標記為相關的 10 個元組（即「地面實況」）與每個演算法返回的 10 個元組的匹配程度。

We used the formal precision/recall metrics to measure this overlap. Precision is the ratio of the number of retrieved tuples that are relevant to the total
我們使用正式的精確率/召回率指標來衡量這種重疊。精確率是檢索到的相關元組數與總數的比率。

ACM Transactions on Database Systems, Vol. 31, No. 3, September 2006.
ACM Transactions on Database Systems, Vol. 31, No. 3, September 2006.

[Image]
[圖片]

Fig. 4. Average precision.
圖 4. 平均精確率。

number of retrieved tuples, while Recall is the ratio of the fraction of the number of retrieved tuples that are relevant to the total number of relevant tuples (see Baeza-Yates and Ribeiro-Neto [1999]). In our case, the total number of relevant tuples was 10, so Precision and Recall were equal. (We reiterate that this is only an artefact of our experimental setu—the “true” Recall can be measured only if the user is able to mark the entire dataset, which was unfeasible in our case).
檢索到的元組數量，而召回率是檢索到的相關元組數量與相關元組總數的比例（參見 Baeza-Yates and Ribeiro-Neto [1999]）。在我們的案例中，相關元組的總數為 10，因此精確率和召回率相等。（我們重申，這只是我們實驗設置的人為結果——「真實」的召回率只有在使用者能夠標記整個資料集時才能測量，這在我們的案例中是不可行的）。

We experimented with several sets of queries in this survey. We first present the results for the following four IN/Range queries for the Seattle Homes dataset:
我們在這次調查中實驗了幾組查詢。我們首先展示了西雅圖房屋資料集的以下四個 IN/範圍查詢的結果：

Q1: Bedrooms=4 AND City IN{Redmond, Kirkland, Bellevue};
Q1: Bedrooms=4 AND City IN{Redmond, Kirkland, Bellevue};

Q2: City IN {Redmond, Kirkland, Bellevue} AND Price BETWEEN ($700K, $1000K);
Q2: City IN {Redmond, Kirkland, Bellevue} AND Price BETWEEN ($700K, $1000K);

Q3: Price BETWEEN ($700K, $1000K);
Q3: Price BETWEEN ($700K, $1000K);

Q4: School=1 AND Price BETWEEN ($100K, $200K).
Q4: School=1 AND Price BETWEEN ($100K, $200K).

The precision (averaged over these queries) of the different ranking methods is shown in Figure 4 (a). As can be seen, the quality of Conditional ranking was superior to Global, while Random was significantly worse than either.
不同排序方法的精確率（在這些查詢上取平均值）如圖 4 (a) 所示。可以看出，Conditional 排序的品質優於 Global，而 Random 則明顯差於兩者。

We next present our survey results for the following five point queries for the Movies dataset (where precision was measured as described above for the Seattle Homes dataset):
接下來，我們展示我們對電影資料集的以下五個點查詢的調查結果（其中精確率的測量方法如上所述，與西雅圖房屋資料集相同）：

Q1: Genre=thriller AND Certificate=PG-13;
Q1: Genre=thriller AND Certificate=PG-13;

Q2: YearMade=1980 AND Certificate=PG-13;
Q2: YearMade=1980 AND Certificate=PG-13;

Q3: Certificate=G AND Sound=Mono;
Q3: Certificate=G AND Sound=Mono;

Q4: Actor1=Dreyfuss, Richard;
Q4: Actor1=Dreyfuss, Richard;

Q5: Genre=Sci-Fi.
Q5: Genre=Sci-Fi.

The results are shown in Figure 4 (b). The quality of Conditional ranking was superior to Global, while Random was worse than either.
結果如圖 4 (b) 所示。Conditional 排序的品質優於 Global，而 Random 則比兩者都差。

##### 6.1.2.2 Second Survey.
##### 6.1.2.2 第二次調查

In addition to the above precision/recall experiments, we also conducted a simpler survey in which users were
除了上述的精確率/召回率實驗外，我們還進行了一項更簡單的調查，其中使用者被

ACM Transactions on Database Systems, Vol. 31, No. 3, September 2006.
ACM Transactions on Database Systems, Vol. 31, No. 3, September 2006.

[Image]
[圖片]

Fig. 5. Percent of users preferring each algorithm.
圖 5. 偏好各演算法的使用者百分比。

given the top-5 results of the three ranking methods for five queries (different from the previous survey), and were asked to choose which rankings they preferred.
給予三種排序方法對五個查詢（與前次調查不同）的前五名結果，並被要求選擇他們偏好的排序。

We used the following IN/Range queries for the Seattle Homes dataset:
我們對西雅圖房屋資料集使用了以下 IN/範圍查詢：

Q1: Bedrooms=4 AND City IN (Redmond, Kirkland, Bellevue);
Q1: Bedrooms=4 AND City IN (Redmond, Kirkland, Bellevue);

Q2: City IN (Bellevue, Kirkland) AND Price BETWEEN ($700K, $1000K);
Q2: City IN (Bellevue, Kirkland) AND Price BETWEEN ($700K, $1000K);

Q3: Price BETWEEN ($500K, $700K) AND Bedrooms=4 AND Year > 1990;
Q3: Price BETWEEN ($500K, $700K) AND Bedrooms=4 AND Year > 1990;

Q4: City=Seattle AND Year > 1990;
Q4: City=Seattle AND Year > 1990;

Q5: City=Seattle AND Bedrooms=2 AND Price=500K.
Q5: City=Seattle AND Bedrooms=2 AND Price=500K.

We also used the following point queries for the Movies dataset:
我們也對電影資料集使用了以下點查詢：

Q1: YearMade=1980 AND Genre=Thriller;
Q1: YearMade=1980 AND Genre=Thriller;

Q2: Actor1=De Niro, Robert;
Q2: Actor1=De Niro, Robert;

Q3: YearMade=1990 AND Genre=Thriller;
Q3: YearMade=1990 AND Genre=Thriller;

Q4: YearMade=1995 AND Genre=Comedy;
Q4: YearMade=1995 AND Genre=Comedy;

Q5: YearMade=1970 AND Genre=Western.
Q5: YearMade=1970 AND Genre=Western.

Figure 5 shows the percent of users that prefer the results of each algorithm: The results of the above experiments show that Conditional generally produces rankings of higher quality compared to Global, especially for the Seattle Homes dataset. While these experiments indicate that our ranking approach has promise, we caution that much larger-scale user studies are necessary to conclusively establish findings of this nature.
圖 5 顯示了偏好每種演算法結果的使用者百分比：上述實驗的結果表明，與 Global 相比，Conditional 通常能產生更高品質的排名，尤其是在 Seattle Homes 資料集上。雖然這些實驗表明我們的排名方法很有前景，但我們提醒，需要更大規模的使用者研究才能最終確定此類發現。

### 6.2 Performance Experiments
### 6.2 效能實驗

In this subsection we report on experiments that compared the performance of the various implementations of the Conditional algorithm: List Merge, its space-saving variants, and Scan. We do not report on the corresponding implementations of Global as they had similar performance. We used the Seattle Homes and U.S. Homes datasets for these experiments. We report performance results of our algorithms on point queries—we do not report results for IN/range queries, as each such query is split into a collection of point
在本小節中，我們報告了比較 Conditional 演算法各種實作效能的實驗：List Merge、其節省空間的變體以及 Scan。我們不報告 Global 的相應實作，因為它們的效能相似。我們使用 Seattle Homes 和 U.S. Homes 資料集進行這些實驗。我們報告我們的演算法在點查詢上的效能結果——我們不報告 IN/範圍查詢的結果，因為每個此類查詢都被拆分成一組點

ACM Transactions on Database Systems, Vol. 31, No. 3, September 2006.
ACM Transactions on Database Systems, Vol. 31, No. 3, September 2006.

| Datasets | List Building Time | List Size |
| :--- | :--- | :--- |
| Seattle Homes | 1500 ms | 7.8 MB |
| U.S. Homes | 80,000 ms | 457.6 MB |
Table II. Time and Space Consumed by Index Module
表格 II. 索引模組消耗的時間與空間

queries whose results are then merged in a straightforward manner as described in Section 5.4.
查詢，其結果隨後以第 5.4 節所述的直接方式合併。

#### 6.2.1 Preprocessing Time and Space.
#### 6.2.1 預處理時間與空間

Since the preprocessing performance of the List Merge algorithm is dominated by the Index Module, we omit reporting results for the Atomic Probabilities Module. Table II shows the space and time required to build all the conditional and global lists. The time and space scale linearly with table size, which is expected. Notice that the space consumed by the lists is three times the size of the data table. While this may seemingly appear excessive, note that a fair comparison would be against a Scan algorithm that has B+ tree indices built on all attributes (so that all kinds of selections can be performed efficiently). In such a case, the total space consumed by these B+ tree indices would rival the space consumed by these lists.
由於列表合併演算法的預處理效能主要由索引模組決定，我們省略了原子機率模組的結果報告。表 II 顯示了建立所有條件列表和全域列表所需的空間和時間。時間和空間與表格大小成線性關係，這符合預期。請注意，列表消耗的空間是資料表格大小的三倍。雖然這看起來可能過多，但請注意，一個公平的比較應該是與在所有屬性上都建有 B+ 樹索引的掃描演算法進行比較（以便可以有效地執行所有類型的選擇）。在這種情況下，這些 B+ 樹索引消耗的總空間將與這些列表消耗的空間相當。

If space is a critical issue, we can adopt the space-saving variation of the List Merge algorithm as discussed in Section 5.3. We report on this next.
如果空間是個關鍵問題，我們可以採用第 5.3 節中討論的列表合併演算法的節省空間變體。我們接下來將報告這一點。

#### 6.2.2 Space-Saving Variations.
#### 6.2.2 節省空間的變體

In this experiment, we showed how the performance of the algorithms changes when only a subset of the set of global and conditional lists are stored. Recall from Section 5.3 that we only retain lists for the values of the frequently occurring attributes in the workload. For this experiment, we considered top-10 queries with selection conditions that specify two attributes (queries generated by randomly picking a pair of attributes and a domain value for each attribute), and measured their execution times. The compared algorithms were
在這個實驗中，我們展示了當只儲存全域和條件列表集合的一個子集時，演算法效能的變化。回顧第 5.3 節，我們只保留工作負載中頻繁出現屬性值的列表。對於這個實驗，我們考慮了選擇條件指定兩個屬性的 top-10 查詢（透過隨機挑選一對屬性和每個屬性的一個定義域值來產生查詢），並測量了它們的執行時間。比較的演算法是

-LM: List Merge with all lists available;
-LM：所有列表皆可用的列表合併；

-LMM: List Merge where lists for one of the two specified attributes are missing, halving space;
-LMM：列表合併，其中兩個指定屬性之一的列表缺失，空間減半；

-Scan.
-掃描。

Figure 6 shows the execution times of the queries over the Seattle Homes database as a function of the total number of tuples that satisfy the selection condition. The times are averaged over 10 queries.
圖 6 顯示了在西雅圖房屋資料庫上，查詢的執行時間與滿足選擇條件的元組總數的函數關係。時間是 10 次查詢的平均值。

We first note that LM is extremely fast when compared to the other algorithms (its times are less than 1 s for each run, and consequently its graph is almost along the x-axis). This is to be expected as most of the computations were accomplished at preprocessing time. The performance of Scan degraded when the total number of selected tuples increased, because the scores of more tuples need to be calculated at runtime. In contrast, the performance of LM and LMM actually improved slightly. This interesting phenomenon occurred because, if more tuples satisfy the selection condition, smaller prefixes of the lists need to be read and merged before the stopping condition is reached.
我們首先注意到，與其他演算法相比，LM 非常快（每次執行的時間都小於 1 秒，因此其圖形幾乎沿著 x 軸）。這是可以預料的，因為大部分計算都在預處理時完成了。當選定元組的總數增加時，Scan 的效能會下降，因為需要在執行時計算更多元組的分數。相比之下，LM 和 LMM 的效能實際上略有改善。這個有趣的現象發生是因為，如果更多的元組滿足選擇條件，那麼在達到停止條件之前，需要讀取和合併的列表前綴就更短。

ACM Transactions on Database Systems, Vol. 31, No. 3, September 2006.
ACM Transactions on Database Systems, Vol. 31, No. 3, September 2006.

[Image]
[圖片]

Fig. 6. Execution times of different variations of list merge and scan for seattle homes dataset.
圖 6. 西雅圖房屋資料集列表合併與掃描不同變體的執行時間。

| NumSelectedTuples | LM Time (ms) | Scan Time (ms) |
| :--- | :--- | :--- |
| 350 | 800 | 6515 |
| 2000 | 700 | 39,234 |
| 5000 | 600 | 11,5282 |
| 30000 | 550 | 56,6516 |
| 80000 | 500 | 3,806,531 |
Table III. Execution Times of List Merge for U.S. Homes Dataset
表格 III. 美國房屋資料集列表合併執行時間

Thus, List Merge and its variations are preferable if the number of tuples satisfying the query condition is large (which is exactly the situation we are interested in, i.e., the Many-Answers problem). This conclusion was reconfirmed when we repeated the experiment with LM and Scan on the much larger U.S. Homes dataset with queries satisfying many more tuples (see Table III).
因此，如果滿足查詢條件的元組數量很大（這正是我們感興趣的情況，即「多答案問題」），那麼列表合併及其變體是更可取的。當我們在更大的美國房屋資料集上重複使用 LM 和 Scan 的實驗，且查詢滿足更多的元組時，這個結論得到了再次證實（見表 III）。

#### 6.2.3 Varying Number of Specified Attributes.
#### 6.2.3 改變指定屬性的數量

Figure 7 shows how the query processing performance of the algorithms varies with the number of attributes specified in the selection conditions of the queries over the U.S. Homes database (the results for the other databases are similar). The times are averaged over 10 top-10 queries. Note that the times increase sharply for both algorithms with the number of specified attributes. The LM algorithm becomes slower because more lists need to be merged, which delays the termination condition. The Scan algorithm becomes slower because the selection time increased with the number of specified attributes. This experiment demonstrates the criticality of keeping the number of sorted streams small in our adaptation of TA.
圖 7 顯示了在美國房屋資料庫上，演算法的查詢處理效能如何隨著查詢選擇條件中指定的屬性數量而變化（其他資料庫的結果類似）。時間是 10 個 top-10 查詢的平均值。請注意，隨著指定屬性數量的增加，兩種演算法的時間都急劇增加。LM 演算法變慢是因為需要合併更多的列表，這會延遲終止條件。Scan 演算法變慢是因為選擇時間隨著指定屬性數量的增加而增加。這個實驗證明了在我們對 TA 的改編中，保持排序流數量較少的重要性。

#### 6.2.4 Varying K in Top-k.
#### 6.2.4 在 Top-k 中改變 K

This experiment showed how the performance of the algorithms decreases with the number K of requested results. The graphs are shown in Figures 8(a) and 8(b) for the Seattle and the U.S. databases respectively. For both datasets, we selected queries with two attributes, which returned about 500 results. Notice that the performance of Scan was not affected by K, since it is not a top-k algorithm. In contrast, LM degraded with K because a longer prefix of the lists needed to be processed. Also notice that Scan
這個實驗顯示了演算法的效能如何隨著請求結果的數量 K 而下降。圖 8(a) 和 8(b) 分別顯示了西雅圖和美國資料庫的圖表。對於這兩個資料集，我們選擇了具有兩個屬性的查詢，這些查詢返回了大約 500 個結果。請注意，Scan 的效能不受 K 的影響，因為它不是一個 top-k 演算法。相比之下，LM 隨著 K 的增加而效能下降，因為需要處理更長的列表前綴。另請注意，Scan

ACM Transactions on Database Systems, Vol. 31, No. 3, September 2006.
ACM Transactions on Database Systems, Vol. 31, No. 3, September 2006.

[Image]
[圖片]

Fig. 7. Varying number of specified atributes for U.S. Homes dataset.
圖 7. 美國房屋資料集指定屬性數量的變化。

[Image]
[圖片]

Fig. 8. Varying number K of requested results.
圖 8. 請求結果數量 K 的變化。

took about the same time for both datasets because the number of the results returned by the selection was the same (500).
對於兩個資料集，所花費的時間大致相同，因為選擇返回的結果數量是相同的（500）。

## 7. CONCLUSIONS AND FUTURE WORK
## 7. 結論與未來工作

We propose a completely automated approach for the Many-Answers Problem which leverages data and workload statistics and correlations. Our ranking functions are based upon the probabilistic IR models, judiciously adapted for structured data. We presented results of preliminary experiments which demonstrate the efficiency as well as the quality of our ranking system.
我們針對「多答案問題」提出了一種完全自動化的方法，該方法利用了資料和工作負載的統計數據及相關性。我們的排序函數基於機率性 IR 模型，並針對結構化資料進行了審慎的調整。我們展示了初步實驗的結果，證明了我們排序系統的效率和品質。

Our work brings forth several intriguing open problems. For example, many relational databases contain text columns in addition to numeric and categorical columns. It would be interesting to see whether correlations between text and nontext data can be leveraged in a meaningful way for ranking. Second, rather than just query strings present in the workload, can more comprehensive user interactions be leveraged in ranking algorithms-for example, tracking the actual tuples that the users select in response to query results? Finally, comprehensive quality benchmarks for database ranking need to be established. This would provide future researchers with a more unified and systematic basis for evaluating their retrieval algorithms.
我們的研究引出了幾個有趣的開放性問題。例如，許多關聯式資料庫除了包含數值和分類欄位外，還包含文本欄位。探討文本和非文本資料之間的相關性是否能以有意義的方式用於排序，將會很有趣。其次，除了工作負載中存在的查詢字串外，是否可以在排序演算法中利用更全面的使用者互動——例如，追蹤使用者在回應查詢結果時選擇的實際元組？最後，需要為資料庫排序建立全面的品質基準。這將為未來的研究人員提供一個更統一和系統的基礎來評估他們的檢索演算法。

ACM Transactions on Database Systems, Vol. 31, No. 3, September 2006.
ACM Transactions on Database Systems, Vol. 31, No. 3, September 2006.

## ACKNOWLEDGMENTS
## 致謝

We thank the anonymous referees for their extremely useful comments on an earlier draft of this article.
我們感謝匿名審稿人對本文初稿提出的極其有用的意見。

## REFERENCES
## 參考文獻

AGRAWAL, S., CHAUDHURI, S., AND DAS, G. 2002. DBXplorer: A system for keyword based search over relational databases. In proceedings of ICDE.
AGRAWAL, S., CHAUDHURI, S., AND DAS, G. 2002. DBXplorer: 一個基於關鍵字在關聯式資料庫上進行搜尋的系統。發表於 ICDE 會議論文集。

AGRAWAL, S., CHAUDHURI, S., DAS, G., AND GIONIS, A. 2003. Automated ranking of database query results. In proceedings of CIDR.
AGRAWAL, S., CHAUDHURI, S., DAS, G., AND GIONIS, A. 2003. 資料庫查詢結果的自動排序。發表於 CIDR 會議論文集。

AMER-YAHIA, S., CASE, P., ROELLEKE, T., SHANMUGASUNDARAM, J., AND WEIKUM. G. 2005a. Report on the DB/IR panel at SIGMOD 2005. ACM SIGMOD Rec. 34, 4, 71-74.
AMER-YAHIA, S., CASE, P., ROELLEKE, T., SHANMUGASUNDARAM, J., AND WEIKUM. G. 2005a. 2005 年 SIGMOD DB/IR 小組報告。ACM SIGMOD Rec. 34, 4, 71-74。

AMER-YAHIA, S., KOUDAS, N., MARIAN, A., SRIVASTAVA, D., AND TOMAN, D. 2005b. Structure and content scoring for XML. In proceedings of VLDB.
AMER-YAHIA, S., KOUDAS, N., MARIAN, A., SRIVASTAVA, D., AND TOMAN, D. 2005b. XML 的結構與內容評分。發表於 VLDB 會議論文集。

AGRAWAL, R., MANNILA, H., SRIKANT, R., TOIVONEN, H., AND VERKAMO, A. I. 1995. Fast discovery of association rules. In proceedings of KDD.
AGRAWAL, R., MANNILA, H., SRIKANT, R., TOIVONEN, H., AND VERKAMO, A. I. 1995. 關聯規則的快速發現。發表於 KDD 會議論文集。

BARBARA, D., GARCIA-MOLINA, H., AND PORTER, D. 1992. The management of probabilistic data. IEEE Trans. Knoual. Data Eng. 4, 5, 487-502.
BARBARA, D., GARCIA-MOLINA, H., AND PORTER, D. 1992. 機率性資料的管理。IEEE Trans. Knoual. Data Eng. 4, 5, 487-502。

BRUNO, N., GRAVANO, L., AND CHAUDHURI, S. 2002a. Top-k selection queries over relational databases: Mapping strategies and performance evaluation. ACM Trans. Database Syst.
BRUNO, N., GRAVANO, L., AND CHAUDHURI, S. 2002a. 關聯式資料庫上的 Top-k 選擇查詢：對映策略與效能評估。ACM Trans. Database Syst.。

BRUNO, N., GRAVANO, L., AND MARIAN, A. 2002b. Evaluating top-k queries over Web-accessible databases. In proceedings of ICDE.
BRUNO, N., GRAVANO, L., AND MARIAN, A. 2002b. 評估網頁可存取資料庫上的 top-k 查詢。發表於 ICDE 會議論文集。

BREESE, J., HECKERMAN, D., AND KADIE, C. 1998. Empirical analysis of predictive algorithms for collaborative filtering. In proceedings of the 14th Conference on Uncertainty in Artificial Intelligence.
BREESE, J., HECKERMAN, D., AND KADIE, C. 1998. 協同過濾預測演算法的實證分析。發表於第 14 屆人工智慧不確定性會議論文集。

BHALOTIA, G., NAKHE, C., HULGERI, A., CHAKRABARTI, S., AND SUDARSHAN, S. 2002. Keyword searching and browsing in databases using BANKS. In Proceedings of ICDE.
BHALOTIA, G., NAKHE, C., HULGERI, A., CHAKRABARTI, S., AND SUDARSHAN, S. 2002. 使用 BANKS 在資料庫中進行關鍵字搜尋與瀏覽。發表於 ICDE 會議論文集。

BAEZA-YATES, R. AND RIBEIRO-NETO, B. 1999. Modern Information Retrieval, 1st ed. Addison-Wesley, Reading, MA.
BAEZA-YATES, R. AND RIBEIRO-NETO, B. 1999. 現代資訊檢索，第一版。Addison-Wesley, Reading, MA。

CESTNIK, B. 1990. Estimating probabilities: A crucial task in machine learning. In Proceedings of the European Conference on artificial Intelligence.
CESTNIK, B. 1990. 估計機率：機器學習中的一項關鍵任務。發表於歐洲人工智慧會議論文集。

CAVALLO, R. AND PITTARELLI, M. 1987. The theory of probabilistic databases. In Proceedings of VLDB.
CAVALLO, R. AND PITTARELLI, M. 1987. 機率性資料庫理論。發表於 VLDB 會議論文集。

CHAUDHURI, S., DAS, G., HRISTIDIS, V., AND WEIKUM, G. 2004. Probabilistic ranking of database query results. In Proceedings of VLDB.
CHAUDHURI, S., DAS, G., HRISTIDIS, V., AND WEIKUM, G. 2004. 資料庫查詢結果的機率性排序。發表於 VLDB 會議論文集。

CHINENYANGA, T. T. AND KUSHMERICK, N. 2002. An expressive and efficient language for XML information retrieval. J. Amer. Soc. Inform. Sci. Tech. 53, 6, 438-453.
CHINENYANGA, T. T. AND KUSHMERICK, N. 2002. 一種用於 XML 資訊檢索的表達性強且高效的語言。J. Amer. Soc. Inform. Sci. Tech. 53, 6, 438-453。

CROFT, W. B. AND LAFFERTY, J. 2003. Language Modeling for Information Retrieval. Kluwer, Norwell, MA.
CROFT, W. B. AND LAFFERTY, J. 2003. 用於資訊檢索的語言模型。Kluwer, Norwell, MA。

CARMEL, D, MAAREK, Y. S., MANDELBROD, M., MASS, Y., AND SOFFER, A. 2003. Searching XML documents via XML fragments. In Proceedings of SIGIR.
CARMEL, D, MAAREK, Y. S., MANDELBROD, M., MASS, Y., AND SOFFER, A. 2003. 透過 XML 片段搜尋 XML 文件。發表於 SIGIR 會議論文集。

COHEN, W. 1998a. Integration of heterogeneous databases without common domains using queries based on textual similarity. In Proceedings of SIGMOD.
COHEN, W. 1998a. 使用基於文本相似性的查詢整合無共同領域的異質資料庫。發表於 SIGMOD 會議論文集。

COHEN, W. 1998b. Providing database-like access to the Web using queries based on textual similarity. In Proceedings of SIGMOD.
COHEN, W. 1998b. 使用基於文本相似性的查詢提供類似資料庫的網頁存取。發表於 SIGMOD 會議論文集。

CHAKRABARTI, K., PORKAEW, K., AND MEHROTRA, S. 2000. Efficient query references in multimedia databases. In Proceedings of ICDE.
CHAKRABARTI, K., PORKAEW, K., AND MEHROTRA, S. 2000. 多媒體資料庫中的高效查詢參考。發表於 ICDE 會議論文集。

DALVI, N. N. AND SUCIU, D. 2005. Answering queries from statistics and probabilistic Views. In Proceedings of VLDB.
DALVI, N. N. AND SUCIU, D. 2005. 從統計和機率性視圖回答查詢。發表於 VLDB 會議論文集。

FAGIN, R. 1998. Fuzzy queries in multimedia database systems. In Proceedings of PODS.
FAGIN, R. 1998. 多媒體資料庫系統中的模糊查詢。發表於 PODS 會議論文集。

FAGIN, R., LOTEM, A., AND NAOR, M. 2001. Optimal aggregation algorithms for middleware. In Proceedings of PODS.
FAGIN, R., LOTEM, A., AND NAOR, M. 2001. 中介軟體的最佳聚合演算法。發表於 PODS 會議論文集。

FUHR, N. 1990. A probabilistic framework for vague queries and imprecise information in databases. In Proceedings of VLDB.
FUHR, N. 1990. 資料庫中模糊查詢與不精確資訊的機率性框架。發表於 VLDB 會議論文集。

FUHR, N. 1993. A probabilistic relational model for the integration of IR and databases. In Proceedings of ACM SIGIR Conference on Research and Development in Information Retrieval.
FUHR, N. 1993. 用於整合 IR 和資料庫的機率性關聯模型。發表於 ACM SIGIR 資訊檢索研究與發展會議論文集。

ACM Transactions on Database Systems, Vol. 31, No. 3, September 2006.
ACM Transactions on Database Systems, Vol. 31, No. 3, September 2006.

FUHR, N. AND GROSSJOHANN, K. 2004. XIRQL: An XML query language based on information retrieval concepts. ACM Trans. Inform. Syst. 22, 2, 313-356.
FUHR, N. AND GROSSJOHANN, K. 2004. XIRQL: 一種基於資訊檢索概念的 XML 查詢語言。ACM Trans. Inform. Syst. 22, 2, 313-356。

FUHR, N. AND ROELLEKE, T. 1997. A probabilistic relational algebra for the integration of information retrieval and database systems. ACM Trans. Inform. Syst. 15, 1, 32-66.
FUHR, N. AND ROELLEKE, T. 1997. 用於整合資訊檢索與資料庫系統的機率性關聯代數。ACM Trans. Inform. Syst. 15, 1, 32-66。

FUHR, N. AND ROELLEKE, T. 1998. HySpirit-a probabilistic inference engine for hypermedia retrieval in large databases. In Proceedings of EDBT.
FUHR, N. AND ROELLEKE, T. 1998. HySpirit－一個用於大型資料庫中超媒體檢索的機率性推論引擎。發表於 EDBT 會議論文集。

GROSSMAN, D. AND FRIEDER, O. 2004. Information Retrieval-Algorithms and Heuristics. Springer, Berlin, Germany.
GROSSMAN, D. AND FRIEDER, O. 2004. 資訊檢索－演算法與啟發式方法。Springer, Berlin, Germany。

GÜNTZER, U., BALKE, W.-T., AND KIESSLING, W. 2000. Optimizing multi-feature queries for image databases. In Proceedings of VLDB.
GÜNTZER, U., BALKE, W.-T., AND KIESSLING, W. 2000. 最佳化影像資料庫的多特徵查詢。發表於 VLDB 會議論文集。

GUO, L., SHAO, F., BOTEV, C., AND SHANMUGASUNDARAM. J. 2003. XRANK: Ranked keyword search over XML documents. In Proceedings of SIGMOD.
GUO, L., SHAO, F., BOTEV, C., AND SHANMUGASUNDARAM. J. 2003. XRANK: XML 文件上的排序關鍵字搜尋。發表於 SIGMOD 會議論文集。

HARPER, D. AND VAN RIJSBERGEN, C. J. 1978. An evaluation of feedback in document retrieval using co-occurrence data. J. Document. 34, 3, 189-216.
HARPER, D. AND VAN RIJSBERGEN, C. J. 1978. 使用共現資料評估文件檢索中的回饋。J. Document. 34, 3, 189-216。

HRISTIDIS, V. AND PAPAKONSTANTINOU, Y. 2002. DISCOVER: Keyword search in relational databases. In Proceedings of VLDB.
HRISTIDIS, V. AND PAPAKONSTANTINOU, Y. 2002. DISCOVER: 關聯式資料庫中的關鍵字搜尋。發表於 VLDB 會議論文集。

HRISTIDIS, V., GRAVANO, L., AND PAPAKONSTANTINOU, Y. 2003a. Efficient IR-style keyword search over relational databases. In Proceedings of VLDB.
HRISTIDIS, V., GRAVANO, L., AND PAPAKONSTANTINOU, Y. 2003a. 關聯式資料庫上高效的 IR 風格關鍵字搜尋。發表於 VLDB 會議論文集。

HRISTIDIS, V., PAPAKONSTANTINOU, Y., AND BALMIN, A. 2003b. Keyword proximity search on XML graphs. In Proceedings of ICDE.
HRISTIDIS, V., PAPAKONSTANTINOU, Y., AND BALMIN, A. 2003b. XML 圖上的關鍵字鄰近搜尋。發表於 ICDE 會議論文集。

JAGADISH, H. V., POOSALA, V., KOUDAS, N., SEVCIK, K., MUTHUKRISHNAN, S., AND SUEL, T. 1998. Optimal histograms with quality guarantees. In Proceedings of VLDB.
JAGADISH, H. V., POOSALA, V., KOUDAS, N., SEVCIK, K., MUTHUKRISHNAN, S., AND SUEL, T. 1998. 具有品質保證的最佳直方圖。發表於 VLDB 會議論文集。

KIESSLING, W. 2002. Foundations of preferences in database systems. In Proceedings of VLDB.
KIESSLING, W. 2002. 資料庫系統中偏好的基礎。發表於 VLDB 會議論文集。

LAKSHMANAN, L. V. S., LEONE, N., Ross, R., AND SUBRAHMANIAN, V. S. 1997. ProbView: A flexible probabilistic database system. ACM Trans. Database Syst. 22, 3, 419-469.
LAKSHMANAN, L. V. S., LEONE, N., Ross, R., AND SUBRAHMANIAN, V. S. 1997. ProbView: 一個彈性的機率性資料庫系統。ACM Trans. Database Syst. 22, 3, 419-469。

LALMAS, M. AND ROELLEKE, T. 2004. Modeling vague content and structure querying in XML retrieval with a probabilistic object-relational framework. In Proceedings of FQAS.
LALMAS, M. AND ROELLEKE, T. 2004. 使用機率性物件關聯框架模型化 XML 檢索中的模糊內容與結構查詢。發表於 FQAS 會議論文集。

MARTINEZ, W., MARTINEZ, A., AND WEGMAN, E. 2004. Document classification and clustering using weighted text proximity matrices. In Proceedings of Interface.
MARTINEZ, W., MARTINEZ, A., AND WEGMAN, E. 2004. 使用加權文本鄰近矩陣進行文件分類與分群。發表於 Interface 會議論文集。

MOTRO, A. 1988. VAGUE: A user interface to relational databases that permits vague queries. ACM Trans. Informat. Syst. 6, 3 (July), 187-214.
MOTRO, A. 1988. VAGUE: 一個允許模糊查詢的關聯式資料庫使用者介面。ACM Trans. Informat. Syst. 6, 3 (July), 187-214。

NAZERI, Z., BLOEDORN, E., AND OSTWALD, P. 2001. Experiences in mining aviation safety data. In Proceedings of SIGMOD.
NAZERI, Z., BLOEDORN, E., AND OSTWALD, P. 2001. 挖掘航空安全資料的經驗。發表於 SIGMOD 會議論文集。

NEPAL, S. AND RAMAKRISHNA, M. V. 1999. Query processing issues in image (multimedia) databases. In Proceedings of ICDE.
NEPAL, S. AND RAMAKRISHNA, M. V. 1999. 影像（多媒體）資料庫中的查詢處理問題。發表於 ICDE 會議論文集。

ORTEGA-BINDERBERGER, M., CHAKRABARTI, K., AND MEHROTRA, S. 2002. An approach to integrating query refinement in SQL. In Proceedings of EDBT. 15-33.
ORTEGA-BINDERBERGER, M., CHAKRABARTI, K., AND MEHROTRA, S. 2002. 一種在 SQL 中整合查詢精煉的方法。發表於 EDBT 會議論文集。15-33。

POOSALA, V., IOANNIDIS, Y. E., HAAS, P. J., AND SHEKITA, E. J. 1996. Improved histograms for selectivity estimation of range predicates. In Proceedings of SIGMOD. 294-305.
POOSALA, V., IOANNIDIS, Y. E., HAAS, P. J., AND SHEKITA, E. J. 1996. 用於範圍謂詞選擇性估計的改進直方圖。發表於 SIGMOD 會議論文集。294-305。

RADLINSKI, F. AND JOACHIMS, T. 2005. Query chains: Learning to rank from implicit feedback. In Proceedings of KDD.
RADLINSKI, F. AND JOACHIMS, T. 2005. 查詢鏈：從隱性回饋中學習排序。發表於 KDD 會議論文集。

RUI, Y., HUANG, T. S., AND MEHROTRA, S. 1997. Content-based image retrieval with relevance feedback in MARS. In Proceedings of the IEEE Conference on Image Processing.
RUI, Y., HUANG, T. S., AND MEHROTRA, S. 1997. MARS 中基於內容的影像檢索與相關性回饋。發表於 IEEE 影像處理會議論文集。

SHEN, X., TAN, B. AND ZHAI, C. 2005. Context-sensitive information retrieval using implicit feedback. In Proceedings of SIGIR.
SHEN, X., TAN, B. AND ZHAI, C. 2005. 使用隱性回饋的上下文感知資訊檢索。發表於 SIGIR 會議論文集。

SPARCK JONES, K., WALKER, S., AND ROBERTSON, S. E. 2000a. A probabilistic model of information retrieval: Development and comparative experiments-Part 1. Inf. Process. Man. 36, 6, 779-808.
SPARCK JONES, K., WALKER, S., AND ROBERTSON, S. E. 2000a. 資訊檢索的機率模型：發展與比較實驗－第一部分。Inf. Process. Man. 36, 6, 779-808。

SPARCK JONES, K., WALKER, S., AND ROBERTSON, S. E. 2000a. A probabilistic model of information retrieval: Development and comparative experiments-Part 2. Inf. Process. Man. 36, 6, 809-840.
SPARCK JONES, K., WALKER, S., AND ROBERTSON, S. E. 2000a. 資訊檢索的機率模型：發展與比較實驗－第二部分。Inf. Process. Man. 36, 6, 809-840。

THEOBALD, A. AND WEIKUM, G. 2002. The index-based XXL search engine for querying XML data with relevance ranking. In Proceedings of EDBT.
THEOBALD, A. AND WEIKUM, G. 2002. 用於查詢帶有關聯性排序的 XML 資料的基於索引的 XXL 搜尋引擎。發表於 EDBT 會議論文集。

THEOBALD, M., SCHENKEL, R., AND WEIKUM, G. 2005. An efficient and versatile query engine for topX search. In Proceedings of VLDB.
THEOBALD, M., SCHENKEL, R., AND WEIKUM, G. 2005. 一個高效且多功能的 topX 搜尋查詢引擎。發表於 VLDB 會議論文集。

WU, L., FALOUTSOS, C., SYCARA, K., AND PAYNE, T. 2000. FALCON: Feedback adaptive loop for content-based retrieval. In Proceedings of VLDB.
WU, L., FALOUTSOS, C., SYCARA, K., AND PAYNE, T. 2000. FALCON: 用於基於內容檢索的回饋自適應迴路。發表於 VLDB 會議論文集。

WHITTAKER, J. 1990. Graphical Models in Applied Multivariate Statistics. Wiley, New York, NY.
WHITTAKER, J. 1990. 應用多變量統計中的圖形模型。Wiley, New York, NY。

WIDOM, J. 2005. Trio: A system for integrated management of data, accuracy, and lineage. CIDR.
WIDOM, J. 2005. Trio: 一個用於資料、準確性和血緣整合管理的系統。CIDR。

WIMMERS, L., HAAS, L. M., ROTH, M. T., AND BRAENDLI, C. 1999. Using Fagin's algorithm for merging ranked results in multimedia middleware. In Proceedings of CoopIS.
WIMMERS, L., HAAS, L. M., ROTH, M. T., AND BRAENDLI, C. 1999. 在多媒體中介軟體中使用 Fagin 演算法合併排序結果。發表於 CoopIS 會議論文集。

XU, J. AND CROFT, W. B. 1996. Query expansion using local and global document analysis, In Proceedings of SIGIR. 4-11.
XU, J. AND CROFT, W. B. 1996. 使用局部和全域文件分析進行查詢擴展，發表於 SIGIR 會議論文集。4-11。

YU, C.T. AND MENG, W. 1998. Principles of Database Query Processing for Advanced Applications. Morgan Kaufmann, San Francisco, CA.
YU, C.T. AND MENG, W. 1998. 高級應用的資料庫查詢處理原則。Morgan Kaufmann, San Francisco, CA。

Received November 2005; revised June 2006; accepted June 2006
收件於 2005 年 11 月；修訂於 2006 年 6 月；接受於 2006 年 6 月

ACM Transactions on Database Systems, Vol. 31, No. 3, September 2006.
ACM Transactions on Database Systems, Vol. 31, No. 3, September 2006.
