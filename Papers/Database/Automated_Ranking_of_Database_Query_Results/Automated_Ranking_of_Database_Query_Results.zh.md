---
title: "Automated_Ranking_of_Database_Query_Results"
field: "Database"
status: "Imported"
created_date: 2026-01-12
pdf_link: "[[Automated_Ranking_of_Database_Query_Results.pdf]]"
tags: [paper, Database]
---


# Automated Ranking of Database Query Results

# 自動化資料庫查詢結果排名

**Agam Shah, Surajit Chaudhuri, Gautam Das, Venkatesh Ganti, Dong Xin**
Microsoft Research, University of Texas at Arlington
{agamshah, surajitc, vganti, dongxin}@microsoft.com, gdas@uta.edu

**ABSTRACT**
Relational database systems are very successful in storing and querying data. However, a key missing functionality in existing RDBMS is their inability to rank the results of a SQL query. Such a ranking functionality is important for interactive data exploration scenarios, where the user is interested in finding the “most interesting” answers first. In this paper, we address the problem of ranking query results by assigning a score to each tuple in the result of a SQL query. We propose a principled approach to ranking that leverages the knowledge of the application’s data model and query workload. Our approach is based on the intuition that tuples that are “related” to a larger number of other interesting tuples are more important. We formalize this intuition through a graph-based model. We present a scalable algorithm for ranking the results of a given SQL query. We show the effectiveness of our ranking by presenting the results of our experiments on two real-life datasets: (a) the TPC-E dataset, which models a stock-trading application, and (b) the Microsoft Academic Search dataset.

**摘要**
關聯式資料庫系統在儲存和查詢資料方面非常成功。然而，現有 RDBMS 的一個關鍵缺失功能是它們無法對 SQL 查詢的結果進行排名。這種排名功能對於互動式資料探索場景非常重要，在這些場景中，使用者有興趣首先找到「最有趣」的答案。在本文中，我們透過為 SQL 查詢結果中的每個元組分配一個分數來解決查詢結果的排名問題。我們提出了一種有原則的排名方法，該方法利用了應用程式資料模型和查詢工作負載的知識。我們的方法基於這樣一種直覺：與更多其他有趣元組「相關」的元組更重要。我們透過基於圖形的模型將這種直覺形式化。我們提出了一種可擴展的演算法，用於對給定 SQL 查詢的結果進行排名。我們透過展示我們在兩個真實數據集上的實驗結果來證明我們排名的有效性：(a) TPC-E 數據集，它模擬了一個股票交易應用程式，以及 (b) Microsoft 學術搜尋數據集。

## 1. INTRODUCTION
Relational database systems are a popular choice for storing and querying data. However, a key missing functionality in existing RDBMS is their inability to rank the results of a SQL query. Such a ranking functionality is important for interactive data exploration scenarios, where the user is interested in finding the “most interesting” answers first. For example, consider a stock-trading application that stores its data in a relational database. An analyst using this application may be interested in finding the most “active” customers. The analyst may express her interest by issuing a SQL query that returns all customers. However, if the number of customers is large, she may be overwhelmed by the large number of answers. In such a scenario, it is desirable to rank the customers by their “activity” level.

## 1. 緒論
關聯式資料庫系統是儲存和查詢資料的熱門選擇。然而，現有 RDBMS 的一個關鍵缺失功能是它們無法對 SQL 查詢的結果進行排名。這種排名功能對於互動式資料探索場景非常重要，在這些場景中，使用者有興趣首先找到「最有趣」的答案。例如，考慮一個將其資料儲存在關聯式資料庫中的股票交易應用程式。使用此應用程式的分析師可能有興趣找到最「活躍」的客戶。分析師可以透過發出返回所有客戶的 SQL query 來表達她的興趣。但是，如果客戶數量龐大，她可能會被大量的答案所淹沒。在這種情況下，最好按客戶的「活動」等級對其進行排名。

The problem of ranking is well-studied in the context of Information Retrieval (IR) and Web search. However, the techniques used in IR and Web search are not directly applicable to the problem of ranking results of a SQL query. This is because the IR and Web search techniques are designed for ranking a set of “documents” in response to a “keyword query”. In our case, the objects to be ranked are tuples in the result of a SQL query, and the query is a SQL query, not a keyword query.

排名問題在資訊檢索 (IR) 和網路搜尋的背景下得到了充分的研究。然而，IR 和網路搜尋中使用的技術不能直接應用於對 SQL 查詢結果進行排名的問題。這是因為 IR 和網路搜尋技術旨在對一組「文件」進行排名以回應「關鍵字查詢」。在我們的例子中，要排名的對像是 SQL 查詢結果中的元組，而查詢是 SQL 查詢，而不是關鍵字查詢。

In this paper, we address the problem of ranking the results of a SQL query. We propose a principled approach to ranking that leverages the knowledge of the application’s data model and query workload. Our approach is based on the intuition that tuples that are “related” to a larger number of other interesting tuples are more important. We formalize this intuition through a graph-based model. We present a scalable algorithm for ranking the results of a given SQL query. We show the effectiveness of our ranking by presenting the results of our experiments on two real-life datasets: (a) the TPC-E dataset, which models a stock-trading application, and (b) the Microsoft Academic Search dataset.

在本文中，我們解決了對 SQL 查詢結果進行排名的問題。我們提出了一種有原則的排名方法，該方法利用了應用程式資料模型和查詢工作負載的知識。我們的方法基於這樣一種直覺：與更多其他有趣元組「相關」的元組更重要。我們透過基於圖形的模型將這種直覺形式化。我們提出了一種可擴展的演算法，用於對給定 SQL 查詢的結果進行排名。我們透過展示我們在兩個真實數據集上的實驗結果來證明我們排名的有效性：(a) TPC-E 數據集，它模擬了一個股票交易應用程式，以及 (b) Microsoft 學術搜尋數據集。

Our contributions are as follows:
*   We propose a principled approach to ranking the results of a SQL query. Our approach is based on a graph-based model that captures the relationships between tuples in the database.
*   We present a scalable algorithm for ranking the results of a given SQL query.
*   We show the effectiveness of our ranking by presenting the results of our experiments on two real-life datasets.

我們的貢獻如下：
*   我們提出了一種對 SQL 查詢結果進行排名的原則性方法。我們的方法基於一個圖形模型，該模型捕捉資料庫中元組之間的關係。
*   我們提出了一種可擴展的演算法，用於對給定 SQL 查詢的結果進行排名。
*   我們透過展示我們在兩個真實數據集上的實驗結果來證明我們排名的有效性。

The rest of the paper is organized as follows. In Section 2, we present our graph-based model for ranking. In Section 3, we present our algorithm for ranking the results of a SQL query. In Section 4, we present the results of our experiments. In Section 5, we discuss related work. Finally, in Section 6, we conclude the paper.

本文的其餘部分組織如下。在第 2 節中，我們介紹了我們用於排名的基於圖形的模型。在第 3 節中，我們介紹了我們用於對 SQL 查詢結果進行排名的演算法。在第 4 節中，我們介紹了我們的實驗結果。在第 5 節中，我們討論了相關工作。最後，在第 6 節中，我們總結了本文。

## 2. GRAPH-BASED MODEL FOR RANKING
In this section, we present our graph-based model for ranking. We first define the notion of a “relationship graph”, which captures the relationships between tuples in the database. We then define the notion of “importance” of a tuple, which is used to rank the tuples.

## 2. 用於排名的基於圖形的模型
在本節中，我們介紹了我們用於排名的基於圖形的模型。我們首先定義「關係圖」的概念，它捕捉資料庫中元組之間的關係。然後我們定義元組「重要性」的概念，它用於對元組進行排名。

### 2.1 Relationship Graph
A relationship graph is a directed graph G = (V, E), where V is the set of all tuples in the database, and E is a set of directed edges between tuples. An edge from tuple t1 to tuple t2 exists if t1 is “related” to t2. The notion of “relatedness” is application-dependent. In this paper, we consider two tuples to be related if they are joined by a foreign key constraint. Specifically, if there is a foreign key constraint from a table T1 to a table T2, then for every tuple t1 in T1, there is an edge from t1 to the corresponding tuple t2 in T2. We also add an edge from t2 to t1. Thus, the relationship graph is an undirected graph.

### 2.1 關係圖
關係圖是一個有向圖 G = (V, E)，其中 V 是資料庫中所有元組的集合，E 是元組之間的有向邊的集合。如果元組 t1 與 t2 「相關」，則存在從 t1 到 t2 的邊。「相關性」的概念取決於應用程式。在本文中，如果兩個元組透過外鍵約束連接，我們就認為它們是相關的。具體來說，如果存在從表 T1 到表 T2 的外鍵約束，那麼對於 T1 中的每個元組 t1，都存在一條從 t1 到 T2 中相應元組 t2 的邊。我們還添加了一條從 t2 到 t1 的邊。因此，關係圖是一個無向圖。

In addition to the edges derived from foreign key constraints, we also add edges based on the query workload. Specifically, if a query joins two tables T1 and T2 on a predicate p, then for every pair of tuples (t1, t2) such that t1 is in T1, t2 is in T2, and p(t1, t2) is true, we add an edge between t1 and t2. The weight of the edge is proportional to the number of times the query is executed.

除了源自外鍵約束的邊之外，我們還根據查詢工作負載添加邊。具體來說，如果查詢在謂詞 p 上連接兩個表 T1 和 T2，那麼對於每一對元組 (t1, t2)，其中 t1 在 T1 中，t2 在 T2 中，並且 p(t1, t2) 為真，我們在 t1 和 t2 之間添加一條邊。邊的權重與查詢執行的次數成正比。

### 2.2 Importance of a Tuple
The importance of a tuple t is defined as the sum of the importances of its neighboring tuples. This is similar to the definition of PageRank. Specifically, the importance of a tuple t, denoted by I(t), is defined as:
I(t) = sum_{t' in N(t)} w(t', t) * I(t')
where N(t) is the set of neighbors of t, and w(t', t) is the weight of the edge from t' to t. The weights are normalized such that the sum of the weights of the outgoing edges from any tuple is 1.

### 2.2 元組的重要性
元組 t 的重要性定義為其相鄰元組重要性的總和。這與 PageRank 的定義相似。具體來說，元組 t 的重要性，表示為 I(t)，定義為：
I(t) = sum_{t' in N(t)} w(t', t) * I(t')
其中 N(t) 是 t 的鄰居集合，w(t', t) 是從 t' 到 t 的邊的權重。權重被標準化，使得從任何元組出發的傳出邊的權重總和為 1。

The importance of all tuples can be computed by iteratively applying the above equation. The initial importance of all tuples is set to 1. The iteration stops when the importances of all tuples converge.

所有元組的重要性可以透過迭代應用上述方程式來計算。所有元組的初始重要性設定為 1。當所有元組的重要性收斂時，迭代停止。

## 3. RANKING ALGORITHM
In this section, we present our algorithm for ranking the results of a SQL query. Our algorithm consists of two phases: (a) an offline phase, where we compute the importance of all tuples in the database, and (b) an online phase, where we rank the results of a given SQL query.

## 3. 排名演算法
在本節中，我們介紹了我們用於對 SQL 查詢結果進行排名的演算法。我們的演算法包括兩個階段：(a) 離線階段，我們計算資料庫中所有元組的重要性，以及 (b) 線上階段，我們對給定 SQL 查詢的結果進行排名。

### 3.1 Offline Phase
In the offline phase, we compute the importance of all tuples in the database. This is done by first constructing the relationship graph, and then iteratively computing the importances of all tuples until they converge. The relationship graph can be constructed by analyzing the database schema (for foreign key constraints) and the query workload (for join predicates). The number of iterations required for convergence depends on the structure of the graph. In our experiments, we found that the importances converge within a few tens of iterations.

### 3.1 離線階段
在離線階段，我們計算資料庫中所有元組的重要性。這是透過首先建構關係圖，然後迭代計算所有元組的重要性直到它們收斂來完成的。關係圖可以透過分析資料庫模式（對於外鍵約束）和查詢工作負載（對於連接謂詞）來建構。收斂所需的迭代次數取決於圖的結構。在我們的實驗中，我們發現重要性在幾十次迭代內收斂。

### 3.2 Online Phase
In the online phase, we are given a SQL query Q. We first execute the query to get the set of result tuples R. For each tuple t in R, we look up its pre-computed importance I(t). We then rank the tuples in R in descending order of their importances.

### 3.2 線上階段
在線上階段，我們得到一個 SQL 查詢 Q。我們首先執行該查詢以獲得結果元組 R 的集合。對於 R 中的每個元組 t，我們查找其預先計算的重要性 I(t)。然後我們按重要性的降序對 R 中的元組進行排名。

The online phase is very fast, as it only involves a lookup of the pre-computed importances. The main cost is the execution of the query Q.

線上階段非常快，因為它只涉及預先計算的重要性的查找。主要成本是查詢 Q 的執行。

## 4. EXPERIMENTS
In this section, we present the results of our experiments. We used two real-life datasets in our experiments: (a) the TPC-E dataset, which models a stock-trading application, and (b) the Microsoft Academic Search dataset.

## 4. 實驗
在本節中，我們介紹了我們的實驗結果。我們在實驗中使用了兩個真實的數據集：(a) TPC-E 數據集，它模擬了一個股票交易應用程式，以及 (b) Microsoft 學術搜尋數據集。

### 4.1 TPC-E Dataset
The TPC-E dataset is a benchmark for online transaction processing (OLTP) systems. It models a stock-trading application. The database consists of 10 tables. We used a scale factor of 1, which corresponds to a database of size 1 GB. We used the query workload that comes with the benchmark. The workload consists of 10 types of transactions.

### 4.1 TPC-E 數據集
TPC-E 數據集是在線交易處理 (OLTP) 系統的基準。它模擬了一個股票交易應用程式。資料庫由 10 個表組成。我們使用了 1 的比例因子，對應於 1 GB 大小的資料庫。我們使用了基準附帶的查詢工作負載。工作負載由 10 種類型的交易組成。

We considered the task of ranking the customers by their “activity” level. We issued a query that returns all customers. We then ranked the customers using our proposed approach. We also compared our ranking with a baseline ranking, which ranks the customers by the number of trades they have made. We found that our ranking is more effective than the baseline ranking in identifying the most “active” customers. For example, the top-ranked customer in our ranking had made a large number of trades, and was also related to a large number of other important tuples, such as brokers and companies.

我們考慮了按「活動」等級對客戶進行排名的任務。我們發出了一個返回所有客戶的查詢。然後我們使用我們提出的方法對客戶進行排名。我們還將我們的排名與基線排名進行了比較，基線排名按客戶進行的交易數量對其進行排名。我們發現，在識別最「活躍」的客戶方面，我們的排名比基線排名更有效。例如，我們排名中排名最高的客戶進行了大量交易，並且還與大量其他重要元組相關，例如經紀人和公司。

### 4.2 Microsoft Academic Search Dataset
The Microsoft Academic Search dataset contains information about academic publications. The database consists of 5 tables: Paper, Author, Journal, Conference, and Keyword. We used a subset of the dataset, which contains about 1 million papers. We used a query workload that consists of queries that search for papers by keywords.

### 4.2 Microsoft 學術搜尋數據集
Microsoft 學術搜尋數據集包含有關學術出版物的資訊。資料庫由 5 個表組成：Paper、Author、Journal、Conference 和 Keyword。我們使用了數據集的一個子集，其中包含約 100 萬篇論文。我們使用了一個查詢工作負載，其中包含按關鍵字搜尋論文的查詢。

We considered the task of ranking the papers returned by a keyword query. We used our proposed approach to rank the papers. We also compared our ranking with a baseline ranking, which ranks the papers by their citation count. We found that our ranking is more effective than the baseline ranking in identifying the most “important” papers. For example, the top-ranked paper in our ranking was a seminal paper in its field, and was co-authored by several influential authors.

我們考慮了對關鍵字查詢返回的論文進行排名的任務。我們使用我們提出的方法對論文進行排名。我們還將我們的排名與基線排名進行了比較，基線排名按論文的引用次數對其進行排名。我們發現，在識別最「重要」的論文方面，我們的排名比基線排名更有效。例如，我們排名中排名最高的論文是其領域的開創性論文，並由幾位有影響力的作者合著。

## 5. RELATED WORK
The problem of ranking has been extensively studied in the context of Information Retrieval (IR) and Web search. The most famous ranking algorithm is PageRank, which is used by Google to rank web pages. Our approach is inspired by PageRank. However, there are some key differences. First, PageRank is designed for ranking web pages, whereas our approach is designed for ranking tuples in a relational database. Second, Page-Rank uses the link structure of the web to determine the importance of a page, whereas our approach uses the relationships between tuples in the database, which are derived from foreign key constraints and query workload.

## 5. 相關工作
排名問題已在資訊檢索 (IR) 和網路搜尋的背景下得到廣泛研究。最著名的排名演算法是 PageRank，Google 使用它來對網頁進行排名。我們的方法受到 PageRank 的啟發。但是，存在一些關鍵差異。首先，PageRank 旨在對網頁進行排名，而我們的方法旨在對關聯式資料庫中的元組進行排名。其次，PageRank 使用網路的連結結構來確定頁面的重要性，而我們的方法使用資料庫中元組之間的關係，這些關係源自外鍵約束和查詢工作負載。

There has been some work on ranking in databases. For example, the RankSQL system allows users to specify a ranking function as part of a SQL query. However, the user has to manually specify the ranking function. In our approach, the ranking function is automatically derived from the database schema and the query workload.

在資料庫中已經有一些關於排名的工作。例如，RankSQL 系統允許使用者將排名函數指定為 SQL 查詢的一部分。但是，使用者必須手動指定排名函數。在我們的方法中，排名函數是從資料庫模式和查詢工作負載中自動導出的。

## 6. CONCLUSION
In this paper, we addressed the problem of ranking the results of a SQL query. We proposed a principled approach to ranking that leverages the knowledge of the application’s data model and query workload. Our approach is based on a graph-based model that captures the relationships between tuples in the database. We presented a scalable algorithm for ranking the results of a given SQL query. We showed the effectiveness of our ranking by presenting the results of our experiments on two real-life datasets.

## 6. 結論
在本文中，我們解決了對 SQL 查詢結果進行排名的問題。我們提出了一種有原則的排名方法，該方法利用了應用程式資料模型和查詢工作負載的知識。我們的方法基於一個圖形模型，該模型捕捉資料庫中元組之間的關係。我們提出了一種可擴展的演算法，用於對給定 SQL 查詢的結果進行排名。我們透過展示我們在兩個真實數據集上的實驗結果來證明我們排名的有效性。
