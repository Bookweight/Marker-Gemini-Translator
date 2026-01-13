---
title: "Designing Fair Ranking Schemes"
field: "Database"
status: "Imported"
created_date: 2026-01-12
pdf_link: "[[Designing Fair Ranking Schemes.pdf]]"
tags: [paper, Database]
---


## SIGMOD '19, June 30-July 5, 2019, Amsterdam, Netherlands

# Designing Fair Ranking Schemes

# 設計公平的排名方案

Abolfazl Asudeh, H. V. Jagadish, Julia Stoyanovich, Gautam Das


**University of Michigan; *New York University; $University of Texas at Arlington
{asudeh,jag}@umich.edu; stoyanovich@nyu.edu; gdas@uta.edu

**密西根大學；*紐約大學；德州大學阿靈頓分校
{asudeh,jag}@umich.edu；stoyanovich@nyu.edu；gdas@uta.edu

## ABSTRACT

## 摘要

Items from a database are often ranked based on a combi-nation of criteria. The weight given to each criterion in the combination can greatly affect the fairness of the produced ranking, for example, preferring men over women. A user may have the flexibility to choose combinations that weigh these criteria differently, within limits. In this paper, we de-velop a system that helps users choose criterion weights that lead to greater fairness. We consider ranking functions that compute the score of each item as a weighted sum of (numeric) attribute values, and then sort items on their score. Each ranking function can be expressed as a point in a multi-dimensional space. For a broad range of fairness criteria, including proportionality, we show how to efficiently iden-tify regions in this space that satisfy these criteria. Using this identification method, our system is able to tell users whether their proposed ranking function satisfies the desired fairness criteria and, if it does not, to suggest the smallest modifica-tion that does. Our extensive experiments on real datasets demonstrate that our methods are able to find solutions that satisfy fairness criteria effectively (usually with only small changes to proposed weight vectors) and efficiently (in in-teractive time, after some initial pre-processing).

資料庫中的項目通常根據多個標準的組合進行排名。組合中給予每個標準的權重會極大地影響所產生排名的公平性，例如，偏好男性而非女性。使用者可能在一定限制內，有彈性地選擇不同權重組合。在本文中，我們開發了一個系統，幫助使用者選擇能帶來更大公平性的標準權重。我們考慮的排名函數是將每個項目的分數計算為（數值）屬性值的加權總和，然後根據分數對項目進行排序。每個排名函數都可以表示為多維空間中的一個點。對於包括比例性在內的廣泛公平性標準，我們展示了如何有效地識別滿足這些標準的空間區域。利用這種識別方法，我們的系統能夠告知使用者其提議的排名函數是否滿足所需的公平性標準，如果不滿足，則建議進行最小的修改。我們在真實數據集上進行的廣泛實驗表明，我們的方法能夠有效地（通常只需對提議的權重向量進行微小更改）和高效地（在互動時間內，經過一些初始預處理後）找到滿足公平性標準的解決方案。

### KEYWORDS

### 關鍵詞

Data Ethics; Responsible Data Management; Fairness

數據倫理；負責任的數據管理；公平性

### ACM Reference Format:

### ACM 參考格式：

Abolfazl Asudeh, H. V. Jagadish, Julia Stoyanovich, Gautam Das. 2019. Designing Fair Ranking Schemes. In 2019 International Con-ference on Management of Data (SIGMOD '19), June 30-July 5, 2019, Amsterdam, Netherlands. ACM, New York, NY, USA, 18 pages. https: //doi.org/10.1145/3299869.3300079

Abolfazl Asudeh, H. V. Jagadish, Julia Stoyanovich, Gautam Das. 2019. 設計公平排名方案。在 2019 年國際數據管理會議 (SIGMOD '19)，2019 年 6 月 30 日至 7 月 5 日，阿姆斯特丹，荷蘭。ACM，紐約，紐約州，美國，18 頁。https://doi.org/10.1145/3299869.3300079

## 1 INTRODUCTION

## 1 緒論

Ranking of individuals is commonplace today, and is used, for example, to establish credit worthiness, desirability for college admissions and employment, and attractiveness as dating partners. Properly, topics such as ranking, top-k query processing, and building indexes to efficiently answer such queries, have recently been increasingly relevant to database research. A prominent family of ranking schemes are score-based rankers, which compute the score of each individual from some database D, sort the individuals in decreasing or-der of score, and finally return either the full ranked list, or its highest-scoring sub-set, the top-k. Many score-based rankers compute the score as a linear combination of attribute values, with non-negative weights.

對個人進行排名在今日已是司空見慣，例如用於建立信用價值、大學錄取和就業的合意性，以及作為約會對象的吸引力。排名、top-k 查詢處理以及建立索引以有效回答此類查詢等主題，近年來與資料庫研究的關聯性日益增加。一個重要的排名方案家族是基於分數的排名器，它從某個資料庫 D 中計算每個個體的分數，按分數遞減的順序對個體進行排序，最後返回完整的排名列表，或其得分最高的子集，即 top-k。許多基於分數的排名器將分數計算為屬性值的線性組合，並帶有非負權重。

This sort of linear-weighted scoring and ranking is ubiq-uitous. Many sports use such schemes. For example, tennis players have an ATP rank based on a score that weights each level of success (winner, finalist, semi-finalist, and so on) at each type of tournament, and adds these up. A score with more serious implications is the credit score that each person has in many countries, meant to indicate creditworthiness. Even in the context of academic research, we see such scor-ing: many funding agencies compute a score for a research proposal as a weighted sum of scores for its attributes.

這種線性加權評分和排名無處不在。許多體育運動都使用此類方案。例如，網球運動員的 ATP 排名基於一個分數，該分數對每種類型錦標賽中每個級別的成功（冠軍、決賽選手、半決賽選手等）進行加權，並將它們相加。一個具有更嚴重影響的分數是許多國家中每個人的信用評分，旨在表明其信譽。即使在學術研究的背景下，我們也看到這種評分方式：許多資助機構將研究提案的分數計算為其屬性分數的加權總和。

Because of the potential impact of such rankings on indi-viduals and on population groups, issues of algorithmic bias and discrimination are coming to the forefront of societal and technological discourse [1]. In their seminal work Fried-man and Nissenbaum [2] define a biased computer system as one that (1) systematically and unfairly discriminates against some individuals or groups in favor of others, and (2) joins this discrimination with an unfair outcome.

由於此類排名對個人和群體可能產生影響，演算法偏見和歧視問題正成為社會和技術論述的前沿議題 [1]。在其開創性著作中，Friedman 和 Nissenbaum [2] 將有偏見的電腦系統定義為 (1) 系統性且不公平地歧視某些個人或群體，以偏袒他人，以及 (2) 將此歧視與不公平的結果結合在一起。

We desire a ranking scheme that is fair, in the sense that it mitigates preexisting bias with respect to a protected fea-ture embodied in the data. In line with prior work [3–7], a protected feature denotes membership of an individual in a legally-protected category, such as persons with disabilities, or under-represented minorities by gender or ethnicity. We refer to such categories (e.g., minority ethnicity) as protected groups, and to the attributes that define them (e.g., ethnicity) as sensitive attributes. Interpreting the definition of Fried-man and Nissenbaum [2] for rankings, discrimination occurs when the outcome is systematic and unfavorable, for exam-ple, when minority ethnicity or female gender systematically lead to placing individuals at lower ranks.

我們期望一個公平的排名方案，能夠減輕資料中存在的關於受保護特徵的既有偏見。根據先前的工作 [3–7]，受保護特徵指的是個人在法律上受保護類別中的成員身份，例如殘疾人士，或按性別或族裔劃分的代表性不足的少數群體。我們將此類類別（例如，少數族裔）稱為受保護群體，並將定義它們的屬性（例如，族裔）稱為敏感屬性。將 Friedman 和 Nissenbaum [2] 的定義詮釋於排名中，當結果是系統性且不利時，就會發生歧視，例如，當少數族裔或女性身份系統性地導致個人排名較低時。

Numerous fairness definitions have been considered in the recent literature [7, 8]. A useful dichotomy is between in-dividual fairness and group fairness, also known as statistical parity. The former requires that similar individuals be treated similarly, while the latter requires that demographics of those receiving a particular outcome are identical or similar to the demographics of the population as a whole [8]. These two requirements represent intrinsically different world views, and accommodating both requires trade-offs [9]. We focus on group fairness in this paper. While our techniques apply to a broad range of group fairness criteria, to make our discus-sion concrete we will define fairness in terms of minimum bounds on the number of selected members of a protected group at the top-k, for some reasonable value of k [10].

最近的文獻中已經考慮了許多公平性的定義 [7, 8]。一個有用的二分法是個體公平性和群體公平性，也稱為統計均等。前者要求相似的個體受到相似的對待，而後者要求接受特定結果的人口統計特徵與整個人口的統計特徵相同或相似 [8]。這兩個要求代表了本質上不同的世界觀，並且要同時滿足兩者需要權衡 [9]。本文我們專注於群體公平性。雖然我們的技術適用於廣泛的群體公平性標準，但為了使我們的討論更具體，我們將根據在 top-k 中選定的受保護群體成員數量的最小界限來定義公平性，對於某些合理的 k 值 [10]。

Designing a ranking scheme amounts to selecting a set of weights, one for each attribute. In some situations, we may have access to good labeled training data, and be able to use machine learning techniques. But in many situations, such as to rank tennis players, research proposals, or academic departments, we do not have access to any ground truth. Therefore, we resort to subjectively selected weights in a simple additive scoring function. Such a function is typically defined over a handful of attributes, due to the cognitive burden of selecting the scoring criteria and coming up with an appropriate weight vector. The question we address in this paper is how to introduce fairness into this subjective weight selection process. Consider an example.

設計排名方案相當於為每個屬性選擇一組權重。在某些情況下，我們可能可以取得良好的標記訓練資料，並能夠使用機器學習技術。但在許多情況下，例如對網球運動員、研究計畫或學術部門進行排名，我們無法取得任何真實情況。因此，我們訴諸於在簡單的加法評分函數中主觀選擇的權重。由於選擇評分標準和提出適當權重向量的認知負擔，此類函數通常定義在少數幾個屬性上。我們在本文中要解決的問題是，如何將公平性引入這個主觀的權重選擇過程中。請看一個例子。

EXAMPLE 1. A college admissions officer is designing a rank-ing scheme to evaluate a pool of applicants, each with several potentially relevant attributes. For simplicity, let us focus on two of these attributes – high school GPA and SAT score. Sup-pose that our fairness criterion is that the admitted class com-prise at least 40% women. As the first step, to make the score components comparable, GPA and SAT scores may be normal-ized and standardized. We will denote the resulting values g for GPA and s for SAT. The admissions officer may believe a priori that g and s should have an approximately equal weight, com-puting the score of an applicant t ∈ Das f(t) = 0.5×s+0.5×g, ranking the applicants, and returning the top 500 individuals.

範例 1. 一位大學招生官正在設計一個排名方案，以評估一批申請者，每位申請者都有幾個潛在的相關屬性。為簡單起見，我們只關注其中兩個屬性——高中 GPA 和 SAT 分數。假設我們的公平標準是錄取班級中至少有 40% 的女性。第一步，為了使分數組成部分具有可比性，GPA 和 SAT 分數可以進行標準化和標準化。我們將 GPA 的結果值表示為 g，SAT 的結果值表示為 s。招生官可能先驗地認為 g 和 s 應該具有大致相等的權重，計算申請者 t ∈ D 的分數為 f(t) = 0.5×s+0.5×g，對申請者進行排名，並返回前 500 名。

Upon inspection, it may be determined that an insufficient number of women is returned among the top-k: at least 200 women were expected to be among the top-500, and only 150 were returned, violating our fairness constraint. This violation may be due to a gender disparity in the data: in 2014, women scored about 25 points lower on average than men in the SAT test [11]. Note that the admissions officer was not looking at the sensitive attribute (gender, in our example), and proposed a scoring function that is not obviously biased against women: the lack of fairness is only observed in the outcome.

經檢查後，可能會發現在 top-k 中返回的女性人數不足：預計前 500 名中至少有 200 名女性，但只返回了 150 名，違反了我們的公平性限制。這種違反可能是由於數據中的性別差異：2014 年，女性在 SAT 考試中的平均得分比男性低約 25 分 [11]。請注意，招生官並未查看敏感屬性（在我們的例子中是性別），並提出了一個對女性沒有明顯偏見的評分函數：公平性的缺乏僅在結果中觀察到。

Our goal in this paper is to build a system that will assist the admissions officer in identifying alternative scoring functions that meet the fairness constraint and are close to the original function f in terms of attribute weights, thereby reflecting the admission officer's a priori notion of quality. After a few cycles of such interaction with the system, the admissions officer may choose f'(t) = 0.45 × s + 0.55 × g as the final scoring function.

我們在本文中的目標是建立一個系統，協助招生官員識別符合公平性限制且在屬性權重方面接近原始函數 f 的替代評分函數，從而反映招生官員對品質的先驗概念。在與系統進行幾輪互動後，招生官員可能會選擇 f'(t) = 0.45 × s + 0.55 × g 作為最終評分函數。

As underscored by Example 1, we wish to produce results that are both fair – as stated by the fairness constraints, and of high quality – as stated by the initial scoring function weights. These initial scoring function weights will only approximate quality, for two reasons. First, observational data usually contains imperfect proxies of "true" aspects of quality (e.g., SAT score vs. intelligence, and GPA vs. grit) [9]. Second, future outcomes cannot be perfectly predicted based on present observations, irrespective of whether a simple score-based ranker or a complex learned model is used.

如範例 1 所強調，我們希望產生的結果既公平——如公平性限制所述，又高品質——如初始評分函數權重所述。這些初始評分函數權重只能近似品質，原因有二。首先，觀察數據通常包含品質「真實」面向的不完美代理（例如，SAT 分數對比智力，GPA 對比毅力）[9]。其次，無論是使用簡單的基於分數的排名器還是複雜的學習模型，都無法根據目前的觀察完美預測未來的結果。

Rather than adjusting the scoring function, one could meet fairness constraints by having different cutoff scores for dif-ferent demographic groups. For example, the admissions officer in Example 1 could have stuck with the original func-tion, and used a lower score threshold for admitting women compared to the one for men. While such a fix is technically easy, it is illegal in many jurisdictions, because it amounts to disparate treatment – to the explicit use of a protected characteristic such as gender or race to make decisions. Our proposal in this paper navigates the trade-off between dis-parate treatment and disparate impact – providing outputs that hurt members of a protected group more frequently than members of other groups. If a small adjustment to the weights of the scoring function can achieve fairness, that may be both preferable for legal reasons, and acceptable from the point of view of utility, particularly since the original weights likely were approximate values chosen subjectively.

與其調整評分函數，不如為不同的人口群體設定不同的截止分數來滿足公平性限制。例如，範例 1 中的招生官員可以堅持使用原始函數，並對女性採用比男性更低的錄取分數門檻。雖然這種修正在技術上很容易，但在許多司法管轄區是非法的，因為它構成了差別待遇——明確使用受保護的特徵，如性別或種族來做決定。我們在本文中的提議旨在權衡差別待遇和差別影響——提供更頻繁地傷害受保護群體成員的輸出。如果對評分函數的權重進行微小調整可以實現公平，這在法律上和效用上都可能是更可取的，特別是考慮到原始權重可能是主觀選擇的近似值。

One may also argue that if a scoring function is specified by a human expert, algorithmic bias is not an issue. Yet, there is a long history of people using justifiable models to be able to discriminate. For example, legacy was added to the variables considered at admission, and given a high weight, to keep down the number of Jewish students, since "too many" of them would have been admitted considering academic achievement alone [12, 13].

有人可能還會爭辯說，如果評分函數是由人類專家指定的，那麼演算法偏見就不是問題。然而，歷史上一直有人利用看似合理的模型進行歧視。例如，在招生時考慮的變數中加入了「legacy」，並給予很高的權重，以減少猶太學生的數量，因為如果僅考慮學業成就，「太多」的猶太學生會被錄取 [12, 13]。

Of course, our proposed methods will not prevent institu-tional racism and other kinds of intentional discrimination. That said, it is increasingly recognized that this challenge cannot be addressed by technology alone, and that respon-sibility to determine the context of use of a tool should fall squarely on legal and policy frameworks. Rather than dictat-ing a particular choice of policy, we enable decision makers to transparently enact a policy of their choosing, by supporting an explicit specification of fairness constraints, and incor-porating them into ranking scheme design. Transparency by design is now required by legal frameworks like the New York City algorithmic transparency law [14].

當然，我們提出的方法無法防止制度性種族主義和其他形式的蓄意歧視。話雖如此，人們越來越認識到，僅靠技術無法解決這一挑戰，確定工具使用情境的責任應完全落在法律和政策框架上。我們並非強制規定特定的政策選擇，而是讓決策者能夠透明地制定他們選擇的政策，方法是支持公平性限制的明確規範，並將其納入排名方案設計中。設計透明化現在已成為紐約市演算法透明度法 [14] 等法律框架的要求。

Whether a fairness criterion is met can only be assessed with respect to a specific dataset. However, we can assess the fairness of a scoring function if we have a characterization of the distribution of data points in any data set to which the function will be applied. In other words, we can work with a representative "training" data set to design a fair scoring function. We can then expect this scoring function to remain fair until the data distribution changes substantially.

公平標準是否得到滿足，只能針對特定資料集進行評估。然而，如果我們對函數將應用的任何資料集中資料點的分佈特徵有所瞭解，我們就可以評估評分函數的公平性。換句話說，我們可以使用具代表性的「訓練」資料集來設計一個公平的評分函數。然後，我們可以預期這個評分函數在資料分佈發生重大變化之前會一直保持公平。

Our technical goal is to build a system to assist a human designer of a scoring function in tuning attribute weights to achieve fairness. Since this tuning process does not occur too often, it may be acceptable for it to take some time. However, we know that humans are able to produce superior results when they get quick feedback in a design or analysis loop. Indeed, it is precisely this need that is a central motivation for OLAP, rather than having only long-running analytics queries. Ideally, a designer of a ranking scheme would want the system to support her work through interactive response times. Our goal is to meet this need, to the extent possible.

我們的技術目標是建立一個系統，以協助人類設計師調整屬性權重以實現評分函數的公平性。由於這個調整過程不常發生，因此花費一些時間是可以接受的。然而，我們知道，當人類在設計或分析循環中獲得快速回饋時，能夠產生更優異的結果。事實上，正是這種需求成為 OLAP 的核心動機，而不是只有長時間運行的分析查詢。理想情況下，排名方案的設計師會希望系統能以互動式的回應時間來支援她的工作。我們的目標是盡可能滿足這種需求。

As we will later show, it is computationally challenging to find a scoring function that is both fair and close to the user-specified scoring function, particularly when more than two scoring attributes must be considered. In order to overcome this challenge, we introduce techniques from combinato-rial geometry and, since the direct application of existing algorithm does not scale in practice, we propose the arrange-ment tree data structure. We then propose a grid-partitioning preprocessing method that enables approximate query an-swering. Preprocessed data can be reused if the dataset does not change significantly over time, thus amortizing the cost of pre-computation. In addition to the offline preprocessing method, we also study sampling for on-the-fly query answer-ing based on it. We provide a negative result that states that methods based on function sampling cannot provide any guarantees for the discovery of an approximate solution.

我們稍後將會證明，要找到一個既公平又接近使用者指定評分函數的評分函數，在計算上是具有挑戰性的，特別是當必須考慮兩個以上評分屬性時。為了克服這個挑戰，我們引入了組合幾何的技術，並且由於現有演算法的直接應用在實務上無法擴展，我們提出了排列樹資料結構。然後，我們提出了一種網格分區預處理方法，可以實現近似查詢回答。如果資料集不隨時間發生顯著變化，預處理的資料可以重複使用，從而分攤預計算的成本。除了離線預處理方法，我們還研究了基於它的即時查詢回答的抽樣方法。我們提供了一個負面結果，指出基於函數抽樣的方法無法為發現近似解提供任何保證。

In the remainder of this paper, we will present a query answering system that assists the user in designing fair score-based rankers. The system first preprocesses a dataset of candidates off-line and then handles user requests in real time. The user specifies a query in the form of a scoring function f that associates non-negative weights with item attributes and computes items scores. Items are then sorted on their scores. We assume the existence of a fairness oracle that, given an ordered list of items, returns true if the list meets fairness criteria and so is satisfactory, and returns false otherwise. The fairness oracle relies on its knowledge about the supported fairness definitions to achieve scalability.

在本文的其餘部分，我們將介紹一個查詢回答系統，該系統可協助使用者設計基於公平分數的排名器。該系統首先離線預處理候選人資料集，然後即時處理使用者請求。使用者以評分函數 f 的形式指定查詢，該函數將非負權重與項目屬性關聯並計算項目分數。然後根據分數對項目進行排序。我們假設存在一個公平性神諭，給定一個有序的項目列表，如果該列表符合公平性標準，則返回 true，表示滿意，否則返回 false。公平性神諭依賴其對所支援的公平性定義的知識來實現可擴展性。

If the list of items is found to be unsatisfactory, we suggest to the user an alternative scoring function f' that is both satisfactory and close to the query f. The user may accept f', or she may decide to manually adjust the query and invoke our system once again.

如果發現項目清單不令人滿意，我們會向使用者建議一個替代的評分函數 f'，它既令人滿意又接近查詢 f。使用者可以接受 f'，或者她可以決定手動調整查詢並再次調用我們的系統。

Summary of contributions: We propose a system that as-sists the user in designing fair score-based ranking schemes.

貢獻摘要：我們提出一個系統，協助使用者設計基於公平分數的排名方案。

• We characterize the space of linear scoring functions (with their corresponding weight vectors), and characterize por-tions of this space based on the ordering of items induced by these functions. We develop algorithms to determine boundaries that partition the space into regions where the desired fairness constraint is satisfied, called satisfactory regions, and regions where the constraint is not satisfied. Given a user's query, in the form of a scoring function f, we develop efficient exact algorithms to find the nearest scoring function f' that satisfies the constraint, or to state that the constraint is not satisfiable.

• 我們描述了線性評分函數的空間（及其對應的權重向量），並根據這些函數引導的項目排序來描述該空間的部分。我們開發了演算法來確定將空間劃分為滿足所需公平性約束的區域（稱為滿意區域）和不滿足約束的區域的邊界。給定使用者以評分函數 f 形式的查詢，我們開發了高效的精確演算法來找到滿足約束的最近評分函數 f'，或說明該約束不可滿足。

• We develop approximation algorithms for efficiently iden-tifying and indexing satisfactory regions. We also intro-duce sampling heuristics for on-the-fly query processing in large-scale settings.

• 我們開發了近似演算法，以有效地識別和索引令人滿意的區域。我們還介紹了在大規模設置中用於即時查詢處理的抽樣啟發式方法。

• We conduct an extensive experimental evaluation on real datasets that validates our proposal.

• 我們在真實數據集上進行了廣泛的實驗評估，驗證了我們的提議。

While fairness in algorithmic systems is an active area of research [7], our work is among a small handful of studies that focus on fairness in ranking [5, 6, 10], and is the first to support the user in designing fair ranking schemes.

雖然演算法系統中的公平性是一個活躍的研究領域 [7]，但我們的研究是少數幾個關注排名公平性的研究之一 [5, 6, 10]，並且是第一個支持使用者設計公平排名方案的研究。

Finally, we note that the methods of this paper have appli-cations beyond fairness, and can be used more generally to engineer ranking functions that satisfy input constraints. We choose to focus on fairness to make our discussion concrete, and to contribute to the important emerging area of research in fairness, accountability and transparency (FAT).

最後，我們注意到本文的方法不僅適用於公平性，更可以廣泛地用於設計滿足輸入限制的排名函數。我們選擇專注於公平性，以使我們的討論更具體，並為公平性、問責制和透明度 (FAT) 這個重要的新興研究領域做出貢獻。

## 2 PRELIMINARIES

## 2 預備知識

Data model: We consider a dataset D of n items, each with d scalar scoring attributes. (Additional non-scalar attributes are considered in the fairness model.) We represent an item t as a d-long vector of scoring attributes, (t[1], t[2], ..., t[d]). Without loss of generality, we assume that each scoring attribute is a non-negative number and that larger values are preferred. This assumption is straightforward to relax with some additional notation and bookkeeping.

資料模型：我們考慮一個包含 n 個項目的資料集 D，每個項目都有 d 個純量評分屬性。（在公平性模型中會考慮額外的非純量屬性。）我們將一個項目 t 表示為一個 d 長的評分屬性向量，(t[1], t[2], ..., t[d])。不失一般性，我們假設每個評分屬性都是一個非負數，且較大的值更受青睞。這個假設可以透過一些額外的符號和簿記來輕鬆放寬。

Ranking model: We focus on the class of linear scoring functions that use a weight vector w = (W1, W2,..., wa) to compute a goodness score f(t) of item t as 2=1wjt[j]. To simplify notation, we use f(t) to refer to f(t). Without loss of generality, we assume each weight wj ≥ 0. The scores of items are used for ranking them. We assume that an item with a higher score outranks an item with a lower score. We denote the ranking of items in D based on f with ∇f(D).

排名模型：我們專注於使用權重向量 w = (W1, W2,..., wa) 來計算項目 t 的優良分數 f(t) 為 2=1wjt[j] 的線性評分函數類別。為簡化符號，我們使用 f(t) 來表示 f(t)。不失一般性，我們假設每個權重 wj ≥ 0。項目的分數用於對它們進行排名。我們假設分數較高的項目排名高於分數較低的項目。我們用 ∇f(D) 表示基於 f 的 D 中項目的排名。

[Image]

Figure 1: Effect of weight choice on output fairness

圖 1：權重選擇對輸出公平性的影響

Our ranking model has an intuitive geometric interpre-tation: items are represented by points in Rd, and a linear scoring function f is represented by a ray starting from the origin and passing through the point w = (W1, W2, ..., wd). The score-based ordering of the points induced by f cor-responds to the ordering of their projections onto the ray for w. Figure 1 shows the items of an example dataset with d = 2 as points in R². The function f = x + y is represented in Figure la as a ray stating from the origin and passing through the point (1, 1). Projections of the points onto the ray specify their ordering based on f.

我們的排名模型有一個直觀的幾何解釋：項目被表示為 Rd 中的點，而線性評分函數 f 則被表示為一條從原點出發並穿過點 w = (W1, W2, ..., wd) 的射線。由 f 引起的分數排序對應於這些點在 w 射線上的投影排序。圖 1 顯示了一個 d = 2 的範例資料集的項目，作為 R² 中的點。函數 f = x + y 在圖 1a 中表示為一條從原點出發並穿過點 (1, 1) 的射線。這些點在射線上的投影決定了它們基於 f 的排序。

Note that the rays corresponding to functions f and f' are the same if the weight vector of f' is a linear scaling of the weight vector of f. This is because a weight vector w = (W1, W2,..., wa) induces the same ordering on the items as does its linear scaling w' = (c × W1, c × W2, . . ., c × wd), for any c > 0. Hence, the distance between two functions f and f' is considered as the angular distance between their corresponding rays in Rd. For example, the distance between f = x + y and f' = 100x + 100y is 0, while the distance between f = x + y and f″ = x is 4, the angular distance between the ray corresponding to f in Figure 1a and the x-axis. For every item t ∈ D, contour of t on f is the value combinations in Rd with the same score as f(t) [15, 16]. For linear functions, the contour of an item t is the hyperplane h that is perpendicular to the ray of f and passes through t.

請注意，如果函數 f' 的權重向量是 f 的權重向量的線性縮放，則對應於 f 和 f' 的射線是相同的。這是因為權重向量 w = (W1, W2,..., wa) 在項目上產生的排序與其線性縮放 w' = (c × W1, c × W2, . . ., c × wd) 相同，對於任何 c > 0。因此，兩個函數 f 和 f' 之間的距離被視為它們在 Rd 中對應射線之間的角距離。例如，f = x + y 和 f' = 100x + 100y 之間的距離為 0，而 f = x + y 和 f'' = x 之間的距離為 4，即圖 1a 中對應於 f 的射線與 x 軸之間的角距離。對於每個項目 t ∈ D，t 在 f 上的輪廓是 Rd 中與 f(t) [15, 16] 具有相同分數的值組合。對於線性函數，項目 t 的輪廓是垂直於 f 的射線並穿過 t 的超平面 h。

Fairness model: We adopt a general ranked fairness model, in which a fairness oracle O takes as input an ordered list of items from D, and determines whether the list meets fairness constraints: O : Vf(D) → {T, 1}. A scoring function f that gives rise to a fair ordering over D is said to be satisfactory.

公平性模型：我們採用一個通用的排名公平性模型，其中公平性神諭 O 接收來自 D 的有序項目列表作為輸入，並確定該列表是否滿足公平性約束：O : Vf(D) → {T, 1}。一個在 D 上產生公平排序的評分函數 f 被稱為是令人滿意的。

In addition to scoring attributes, discussed in the data model, items are associated with one or several type at-tributes. A type corresponds to a protected feature such as gender or race. We discussed bias with respect to a protected feature in the introduction. In the example in Figure 1, there is a single binary type attribute, denoted by blue and orange colors. Suppose that the fairness oracle returns true if the top-4 items contain an equal number of items of each type. Function f = x + y in Figure la is not satisfactory as it has 3 orange points and one blue point in its top-4, while f' = 0.97x + 1.3y in Figure 1b contains two points of each type in its top-4 and is satisfactory.

除了在資料模型中討論的評分屬性外，項目還與一個或多個類型屬性相關聯。類型對應於受保護的特徵，例如性別或種族。我們在引言中討論了關於受保護特徵的偏見。在圖 1 的範例中，有一個單一的二元類型屬性，用藍色和橙色表示。假設如果前 4 個項目包含每種類型相同數量的項目，則公平性神諭返回 true。圖 1a 中的函數 f = x + y 不令人滿意，因為其前 4 名中有 3 個橙色點和 1 個藍色點，而圖 1b 中的 f' = 0.97x + 1.3y 在其前 4 名中包含每種類型的兩個點，因此是令人滿意的。

While our fairness model is general, in our experimen-tal evaluation we focus on fairness constraints that were considered in recent literature [4, 6, 10]. We work with pro-portionality constraints that bound the number of items belonging to a particular demographic group (as represented by an assignment of a value to a categorical type attribute) at the top-k, for some given value of k.

雖然我們的公平性模型是通用的，但在我們的實驗評估中，我們專注於最近文獻中考慮的公平性限制 [4, 6, 10]。我們使用比例性限制，該限制在 top-k 中限制了屬於特定人口群體（由分類類型屬性的值分配表示）的項目數量，對於給定的 k 值。

### 2.1 Problem statement

### 2.1 問題陳述

A given query f, with a weight vector w, may not satisfy the required fairness constraints. Our problem is to propose a scoring function f', with a weight vector similar to that of f, that does satisfy the constraints, if one exists.

給定的查詢 f，其權重向量為 w，可能不滿足所需的公平性限制。我們的問題是提出一個評分函數 f'，其權重向量與 f 的權重向量相似，並且確實滿足限制，如果存在的話。

Of course, the user may not accept our proposal. Instead, she may try a different weight vector of her liking, which we can again examine and either approve or propose an alternative. The final choice of an acceptable scoring function is up to the user. We now formally state our problem.

當然，使用者可能不接受我們的建議。相反，她可能會嘗試一個她喜歡的不同權重向量，我們可以再次檢查，並批准或提出替代方案。可接受的評分函數的最終選擇權在於使用者。我們現在正式陳述我們的問題。

Closest Satisfactory Function: Given a dataset D with n items over d scalar scoring attributes, a fairness oracle O : Vf(D) → {T, 1}, and a linear scoring function f with the weight vector w = (W1, W2, …, wa), find the function f' with the weight vector w' such that O(∇f'(D)) = T and the angular distance between w and w' is minimized.

最近的滿意函數：給定一個包含 n 個項目、d 個純量評分屬性的資料集 D，一個公平性神諭 O : Vf(D) → {T, 1}，以及一個帶有權重向量 w = (W1, W2, …, wa) 的線性評分函數 f，找到一個帶有權重向量 w' 的函數 f'，使得 O(∇f'(D)) = T 且 w 和 w' 之間的角距離最小化。

High-level idea: From the system's viewpoint, the chal-lenge is to propose similar weight vectors that satisfy the fairness constraints, in interactive time. To accomplish this, our solution will operate with an offline phase and then an online phase. In the offline phase, we will process the (representative sample) dataset, and develop data structures that will be useful in the online phase. In the online phase, we will exploit these data structures to interactively pro-pose similar satisfactory weight vectors to assist the human designer. Once the human designer has selected the satisfac-tory weights, it may continue to be satisfactory, as long as the datasets have roughly the same distribution. (Later, we also develop a function sampling technique that removes the need for pre-processing and can sometimes be effective).

高層次思想：從系統的角度來看，挑戰在於在互動時間內提出滿足公平性限制的相似權重向量。為此，我們的解決方案將分兩階段運作：離線階段和線上階段。在離線階段，我們將處理（具代表性的樣本）資料集，並開發在線上階段有用的資料結構。在線上階段，我們將利用這些資料結構，以互動方式向人類設計師提出相似的滿意權重向量。一旦人類設計師選擇了滿意權重，只要資料集的分佈大致相同，它就可能繼續令人滿意。（稍後，我們還將開發一種函數抽樣技術，無需預處理，有時可能有效）。

In the next section, we consider the easier to visualize 2D case, in which the dataset contains 2 scalar scoring attributes. The terms and techniques discussed in § 3 will help us in § 4 for developing algorithms for the general multi-dimensional case, where the number of scalar scoring attributes is d > 2.

在下一節中，我們將考慮更容易視覺化的 2D 情況，其中資料集包含 2 個純量評分屬性。第 3 節中討論的術語和技術將有助於我們在第 4 節中為一般多維情況開發演算法，其中純量評分屬性的數量為 d > 2。

[Image]

Figure 2: ordering exchange between a pair of points

圖 2：一對點之間的順序交換

## 3 THE TWO-DIMENSIONAL CASE

## 3 二維案例

In this section we consider a simplified version of the prob-lem in which only two scalar attributes (x and y) participate in the ranking. We begin by introducing the central notion of ordering exchange that partitions the space of linear func-tions into disjoint regions. Then, we use this concept to develop two algorithms: an offline algorithm to identify and index the satisfactory regions, and an online algorithms that can be used repeatedly, as the domain expert interactively tunes weights, to obtain a desired ranking function.

在本節中，我們考慮問題的簡化版本，其中只有兩個純量屬性（x 和 y）參與排名。我們首先介紹排序交換的核心概念，它將線性函數空間劃分為不相交的區域。然後，我們利用這個概念開發了兩種演算法：一種用於識別和索引滿意區域的離線演算法，以及一種可以重複使用的線上演算法，領域專家可以透過互動方式調整權重，以獲得所需的排名函數。

### 3.1 Ordering exchange

### 3.1 訂購交換

Each item in a 2-dimensional dataset can be represented as a point in R2, and each ranking function f can be represented as a ray starting from the origin. The ordering of the items is the ordering of their projections on the ray of f. For instance, Figure 1 specifies the projection of the points on the ray of f = x + y. One can see that the set of rays between the x and y axes represents the set of possible ranking functions in 2D. Even though an infinite number of rays exists between x and y, the number of possible orderings of n items is limited to n!, the number of their permutations. Our central insight is that we do not need to consider every possible ranking function: we only need to consider at most as many as there are orderings of the items, as we discuss next.

二維資料集中的每個項目都可以表示為 R2 中的一個點，每個排名函數 f 都可以表示為從原點出發的一條射線。項目的排序是它們在 f 射線上的投影排序。例如，圖 1 指定了點在 f = x + y 射線上的投影。可以看出，x 軸和 y 軸之間的射線集代表了二維中可能的排名函數集。儘管 x 和 y 之間存在無限多條射線，但 n 個項目的可能排序數量限制為 n!，即它們的排列數。我們的核心見解是，我們不需要考慮每個可能的排名函數：我們只需要考慮最多與項目排序數量一樣多的排序，我們將在下面討論。

Consider two points t₁ (1, 2) and t2 (2, 1), shown in Figure 2. The projections of t₁ and t2 on the x-axis are the points x = 1 and x = 2, respectively. Hence, the ordering based on f = x is t2 > t1, which denotes that t2 is preferred to t₁ by f. Moving away from the x-axis towards the y-axis, the distance between the projections of t₁ and t2 on the ray decreases, and becomes zero at f = x + y. Then, moving from f = x + y to the y-axis, the ordering between these two points changes to t₁ > t2. As we continue moving towards the y-axis, the distance between the projections of t₁ and t2 increases, and their order remains t₁ > t2. Using this observation, we can partition the set of scoring functions based on their angle with the x-axis into F₁ = [0, π/4] and F2 = [π/4, π/2], such that for every f ∈ F₁ the ordering is t2 ≥ t₁ and for every f' ∈ F2 the ordering is t₁ ≥ t2.

考慮圖 2 中所示的兩個點 t₁ (1, 2) 和 t2 (2, 1)。t₁ 和 t2 在 x 軸上的投影分別是點 x = 1 和 x = 2。因此，基於 f = x 的排序是 t2 > t1，這表示 f 偏好 t2 勝過 t₁。從 x 軸向 y 軸移動，t₁ 和 t2 在射線上的投影之間的距離減小，並在 f = x + y 處變為零。然後，從 f = x + y 移動到 y 軸，這兩個點之間的排序變為 t₁ > t2。當我們繼續向 y 軸移動時，t₁ 和 t2 的投影之間的距離增加，它們的順序保持為 t₁ > t2。利用這個觀察，我們可以根據評分函數與 x 軸的角度將其集合劃分為 F₁ = [0, π/4] 和 F2 = [π/4, π/2]，使得對於每個 f ∈ F₁，排序為 t2 ≥ t₁，對於每個 f' ∈ F2，排序為 t₁ ≥ t2。

We define the ordering exchange as the ranking functions that score t₁ and t2 equally. In 2D, the ordering exchange of a pair of points is at most a single function.

我們將排序交換定義為對 t₁ 和 t2 評分相等的排名函數。在二維中，一對點的排序交換最多是一個單一函數。

For any specified ordering of items, the fairness constraint either is satisfied or it is not. If this ordering is changed, the satisfaction of the fairness constraint may change as well. Therefore, in the space of possible ranking functions, every boundary between a satisfactory region and an unsatisfac-tory region must comprise ordering exchange functions.

對於任何指定的項目排序，公平性限制要麼被滿足，要麼不被滿足。如果這個排序改變了，公平性限制的滿足情況也可能改變。因此，在可能的排名函數空間中，滿意區域和不滿意區域之間的每個邊界都必須包含排序交換函數。

### 3.2 Offline processing

### 3.2 離線處理

Offline processing is for identifying and indexing the satis-factory functions, for efficient answering of online queries. Following the example in Figure 2, we propose a ray sweeping algorithm for identifying satisfactory functions in 2D.

離線處理旨在識別和索引令人滿意的函數，以便有效回答線上查詢。繼圖 2 中的範例之後，我們提出了一種射線掃描演算法，用於在二維中識別令人滿意的函數。

To identify the ordering exchanges of pairs of items, we transform items into a dual space [17], where every item t is transformed into the line d(t), as follows:
d(t): t[1]x + t[2]y = 1 (1)

為了識別項目對的排序交換，我們將項目轉換到對偶空間 [17]，其中每個項目 t 都轉換為線 d(t)，如下所示：
d(t): t[1]x + t[2]y = 1 (1)

The ordering of the items based on a function f with the weight vector (w1, w2) is the ordering of the intersections of the lines d(t) with the ray starting from the origin and passing through the point (W1, W2), where the closer intersec-tions to the origin are ranked higher. For example, Figure 4 shows the dual transformation (using Equation 1) of the 2D dataset provided in Figure 3. Therefore, the ordering ex-change of a pair t₁ and tj is the intersection of d(ti) and d(tj). For example, in Figure 4, the ordering exchange of t₁ and t2 is the top-left intersection (of lines d(t₁) and d(t2)).

基於權重向量 (w1, w2) 的函數 f 的項目排序，是線 d(t) 與從原點出發並穿過點 (W1, W2) 的射線的交點的排序，其中離原點較近的交點排名較高。例如，圖 4 顯示了圖 3 中提供的二維資料集的對偶變換（使用方程式 1）。因此，一對 t₁ 和 tj 的排序交換是 d(ti) 和 d(tj) 的交點。例如，在圖 4 中，t₁ 和 t2 的排序交換是左上角的交點（線 d(t₁) 和 d(t2) 的交點）。

Using Equation 1, the intersection of the lines d(ti) and d(tj) can be computed by the following system of equations:
Xd(ti),d(tj): { ti[1]x + ti[2]y = 1 tj[1]x + tj[2]y = 1

使用方程式 1，線 d(ti) 和 d(tj) 的交點可以透過以下方程組計算：
Xd(ti),d(tj): { ti[1]x + ti[2]y = 1 tj[1]x + tj[2]y = 1

The ordering exchange is identified by the angle:
Oti,t; = arctan (tj[1] – ti[1]) / (ti[2] - tj[2]) (2)

排序交換由角度確定：
Oti,t; = arctan (tj[1] – ti[1]) / (ti[2] - tj[2]) (2)

Now, we use ordering exchanges to design the ray sweep-ing algorithm 2DRAYSWEEP (Algorithm 1). The algorithm uses a min-heap to maintain the ordering exchanges. It uses the fact that, sweeping from the x to y-axis, at any moment a pair of items that are adjacent in the ordered list may ex-change ordering. Therefore, it first orders the items based on the x-axis and gradually updates the order as it sweeps the ray toward the y-axis (angle π/2), by swapping the order of pairs of items in their ordering exchanges.

現在，我們使用排序交換來設計射線掃描演算法 2DRAYSWEEP（演算法 1）。該演算法使用最小堆來維護排序交換。它利用了這樣一個事實：從 x 軸掃描到 y 軸時，在任何時刻，有序列表中相鄰的一對項目都可能交換排序。因此，它首先根據 x 軸對項目進行排序，並在將射線掃描到 y 軸（角度 π/2）時，透過交換其排序交換中項目對的順序來逐漸更新順序。

The algorithm initially computes the ordering exchanges between all pairs of adjacent items that do not dominate each other¹ using Equation 2, and adds them to the heap. Next, the algorithm starts sweeping toward the y-axis by removing the ordering exchange with the smallest angle from the heap. Upon visiting an ordering exchange, the algorithm swaps the items that exchange order in the ordered list Ω. The two swapped items can exchange order with their new neighbors in Ω. The algorithm updates the heap by adding these new ordering exchanges. Upon finding a satisfactory sector, the algorithm continues attaching neighboring sectors as long as they are satisfactory, to generate a satisfactory region. Algorithm 1 stores the borders of satisfactory regions in S as pairs (0,0/1), where (0, 0) represents that e is the start of a satisfactory region, while (0, 1) represents that is the end of the region. Consider Figure 5 and suppose that the green sectors are labeled as satisfactory by the fairness oracle. Figure 6 shows the satisfactory regions produced by Algorithm 1. Note that the third region from the left is the union of two neighboring satisfactory sectors in Figure 5.

演算法首先使用方程式 2 計算所有不互相支配¹的相鄰項目對之間的排序交換，並將它們加入堆中。接著，演算法開始朝 y 軸掃描，方法是從堆中移除具有最小角度的排序交換。在訪問排序交換時，演算法會交換有序列表 Ω 中交換順序的項目。這兩個交換的項目可以與它們在 Ω 中的新鄰居交換順序。演算法透過加入這些新的排序交換來更新堆。在找到一個滿意扇區後，演算法會繼續附加相鄰的扇區，只要它們是滿意的，以產生一個滿意區域。演算法 1 將滿意區域的邊界儲存在 S 中，作為成對的 (0,0/1)，其中 (0, 0) 表示 e 是一個滿意區域的開始，而 (0, 1) 表示是該區域的結束。考慮圖 5，並假設綠色扇區被公平性神諭標記為滿意。圖 6 顯示了演算法 1 產生的滿意區域。請注意，左邊的第三個區域是圖 5 中兩個相鄰滿意扇區的聯集。

[Image]

Figure 3: A 2D dataset Figure 4: Fig. 3 in dual space Figure 5: Satisfactory sectors Figure 6: Satisfactory regions

圖 3：一個二維資料集 圖 4：圖 3 的對偶空間 圖 5：滿意扇區 圖 6：滿意區域

Algorithm 1 2DRAYSWEEP
Input: dataset D and fairness oracle O
Output: sorted satisfactory regions S

演算法 1 2DRAYSWEEP
輸入：資料集 D 和公平性神諭 O
輸出：已排序的滿意區域 S

THEOREM 1. Algorithm 1 has time complexity O(n²(log n + Yn)), where Yn is the time complexity of the fairness oracle for input of size of n.

定理 1. 演算法 1 的時間複雜度為 O(n²(log n + Yn))，其中 Yn 是公平性神諭對於大小為 n 的輸入的時間複雜度。

The proofs of all theorems are provided in Appendix F.

所有定理的證明均在附錄 F 中提供。

In our experiments we focus on fairness constraints that bound the number of items belonging to a particular type at the top-k. In general, we expect the fairness oracle to decide in a single pass over the ranking, and so Yn is typically O(n).

在我們的實驗中，我們專注於限制 top-k 中屬於特定類型項目數量的公平性限制。一般來說，我們期望公平性神諭在一次排名遍歷中做出決定，因此 Yn 通常為 O(n)。

### 3.3 Online processing

### 3.3 線上處理

Having the sorted list of 2D satisfactory regions constructed in the offline phase allows us to design an efficient algorithm for online answering of queries. Recall that a query is a proposed set of weights for a linear ranking function. Our task is to determine whether these weights result in a fair ranking, and to suggest weight modifications if they do not. Online processing is implemented by Algorithm 2 that, given f, applies binary search on the sorted list of satisfactory re-gions. If f falls in a satisfactory region, the algorithm returns f, otherwise it returns the satisfactory border closest to f.

在離線階段建構了排序的二維滿意區域列表，這使我們能夠設計一個高效的演算法來線上回答查詢。回想一下，查詢是為線性排名函數提出的一組權重。我們的任務是確定這些權重是否會產生公平的排名，如果不會，則建議修改權重。線上處理由演算法 2 實現，給定 f，它會對排序的滿意區域列表進行二進位搜尋。如果 f 落在滿意區域內，演算法返回 f，否則返回最接近 f 的滿意邊界。

Algorithm 2 2DONLINE
Input: sorted satisfactory regions S, function f: (W1, W2)
Output: weight vector (w₁, w₂)

演算法 2 2DONLINE
輸入：已排序的滿意區域 S，函數 f: (W1, W2)
輸出：權重向量 (w₁, w₂)

THEOREM 2. Algorithm 2 has time complexity O(log n).

定理 2. 演算法 2 的時間複雜度為 O(log n)。

## 4 THE MULTI-DIMENSIONAL CASE

## 4 多維案例

If two attributes are used for ranking, we only have one degree of freedom – the relative weights of the two attributes so the problem is simple. When there are three or more attributes, there are many ways in which we can perturb a given weight vector. We now extend the basic framework introduced in § 3 to handle multi-dimensional cases.

如果使用兩個屬性進行排名，我們只有一個自由度——這兩個屬性的相對權重，所以問題很簡單。當有三個或更多屬性時，我們可以透過多種方式擾動給定的權重向量。我們現在將第 3 節中介紹的基本框架擴展到處理多維情況。

Regions of interest are no longer simple planar wedges, as in the 2D case. Rather, they are high-dimensional objects, with multiple bounding facets. To manage the geometry bet-ter, we first introduce an angle coordinate system, and show that ordering exchanges form hyperplanes in this system. Identifying and indexing satisfactory regions during offline processing is similar to constructing the arrangement of these hyperplanes [17]. We then propose an exact online algorithm that works based on the indexed satisfactory regions.

感興趣的區域不再是像二維情況那樣的簡單平面楔形。相反，它們是具有多個邊界面的高維物件。為了更好地管理幾何形狀，我們首先引入一個角度座標系統，並證明排序交換在該系統中形成超平面。在離線處理期間識別和索引滿意區域類似於建構這些超平面的排列 [17]。然後，我們提出一個基於索引滿意區域的精確線上演算法。

### 4.1 The geometry of ordering exchanges

### 4.1 排序交換的幾何

Consider function f with weight vector w = (w1, W2,…, wd). The score of each tuple ti based on f is a =1Wkti[k]. For ev-ery pair of items t₁ and tj, the ordering exchange is the set of functions that give the same score to both items. As in the previous section, we consider the dual space, transforming item t into a (d – 1)-dimensional hyperplane in Rd:
d(t): Σ k=1 t[k].xk = 1 (3)

考慮權重向量 w = (w1, W2,…, wd) 的函數 f。每個元組 ti 基於 f 的分數為 a =1Wkti[k]。對於每對項目 t₁ 和 tj，排序交換是給予這兩個項目相同分數的函數集合。如同前一節，我們考慮對偶空間，將項目 t 轉換為 Rd 中的一個 (d – 1) 維超平面：
d(t): Σ k=1 t[k].xk = 1 (3)

For a pair of items t₁ and tj, the intersection of d(ti) and d(tj) is a (d-2)-dimensional structure. For instance, in R³ the dual transformation of an item is a plane, and the intersection of two planes is a line. The intersection between d(t₁) and d(tj) can be computed using the system of equations:
Xd(ti), d(tj): Σ=1 k=1 ti[k].xk = 1 Σ=1 tj[k].xk = 1 (4)

對於一對項目 t₁ 和 tj，d(ti) 和 d(tj) 的交集是一個 (d-2) 維結構。例如，在 R³ 中，一個項目的對偶變換是一個平面，兩個平面的交集是一條線。d(t₁) 和 d(tj) 之間的交集可以透過以下方程組計算：
Xd(ti), d(tj): Σ=1 k=1 ti[k].xk = 1 Σ=1 tj[k].xk = 1 (4)

The set of origin-starting rays passing through the points p∈ Xd(ti), d(tj) represents the ordering exchange of tį and tj. Hence, the (d – 1)-dimensional hyperplane defined by Xd(ti), d(tj) and the origin (Equation 5) contains these rays.
∑(ti[k] - tj[k])wk = 0 (5)

從原點出發穿過點 p∈ Xd(ti), d(tj) 的射線集合代表了 tį 和 tj 的排序交換。因此，由 Xd(ti), d(tj) 和原點定義的 (d – 1) 維超平面（方程式 5）包含這些射線。
∑(ti[k] - tj[k])wk = 0 (5)

Consider items t₁ = {1, 2, 3} and t2 = {2, 4, 1} in Figure 7. Us-ing Equation 5, the ordering exchange of t₁ and t2 is defined by the magenta plane w₁ + 2w2 - 2w3 = 0 in Figure 8.

考慮圖 7 中的項目 t₁ = {1, 2, 3} 和 t2 = {2, 4, 1}。使用方程式 5，t₁ 和 t2 的排序交換由圖 8 中的洋紅色平面 w₁ + 2w2 - 2w3 = 0 定義。

As explained in § 2, linear functions over d attributes (rays in Rd) are identified by d – 1 angles, each between 0 and π/2. For instance, in § 3, we identify every function in 2D by an angle θ ∈ [0, π/2]. Similarly, in multiple dimensions, we identify the functions by their angles. We now introduce the angle coordinate system for this purpose.

如第 2 節所述，d 個屬性上的線性函數（Rd 中的射線）由 d – 1 個角度確定，每個角度介於 0 和 π/2 之間。例如，在第 3 節中，我們透過一個角度 θ ∈ [0, π/2] 來識別 2D 中的每個函數。同樣，在多維中，我們透過它們的角度來識別函數。我們現在為此目的引入角度座標系統。

Angle coordinate system: Consider the Rd-1 coordinate system, where every axis θ; ∈ [0, π/2] stands for the angle θ₁ in the polar representation of points in Rd. Every function (ray in Rd) is represented by the point (01, 02, ···, θα−1) in the angle coordinate system. For example, as depicted in Figure 9, a function f in R³ is the combination of two angles 01 and 02, each over the range [0, π/2].

角度座標系統：考慮 Rd-1 座標系統，其中每個軸 θ; ∈ [0, π/2] 代表 Rd 中點的極座標表示中的角度 θ₁。每個函數（Rd 中的射線）由角度座標系統中的點 (01, 02, ···, θα−1) 表示。例如，如圖 9 所示，R³ 中的函數 f 是兩個角度 01 和 02 的組合，每個角度的範圍都在 [0, π/2] 之內。

Following Equation 5, the ordering exchange of a pair of items forms a (d – 2)-dimensional hyperplane in the angle coordinate system. For example, in 3D, the ordering exchange of ti and tj forms a line. We use hi,j to refer to the ordering exchange of ti and tj in the angle coordinate system. In Appendix A, we discuss how to compute ordering exchanges in the angle coordinate system.

根據方程式 5，一對項目的排序交換在角度座標系統中形成一個 (d – 2) 維超平面。例如，在 3D 中，ti 和 tj 的排序交換形成一條線。我們使用 hi,j 來表示 ti 和 tj 在角度座標系統中的排序交換。在附錄 A 中，我們討論如何計算角度座標系統中的排序交換。

### 4.2 Construction of satisfactory regions

### 4.2 滿意區域的建構

The construction of satisfactory regions relates to the ar-rangement [17] of ordering exchange hyperplanes in the angle coordinate system. Consider the arrangement of hi,j, Vti, tj ∈ D. Items t₁ and tj switch order on the two sides of hi, j, while inside each convex region in the arrangement their relative ordering does not change. In the following, we construct all convex regions in the arrangement and check if the ordering inside each is satisfactory.

滿意區域的建構與角度座標系統中排序交換超平面的排列 [17] 有關。考慮 hi,j, Vti, tj ∈ D 的排列。項目 t₁ 和 tj 在 hi, j 的兩側交換順序，而在排列中的每個凸區域內，它們的相對順序不變。接下來，我們將建構排列中的所有凸區域，並檢查每個區域內的排序是否令人滿意。

A convex region is defined as the intersection of a set of half-spaces [17]. Every hyperplane h divides the space into two half-spaces h⁺ and h¯. The ordering between tį and tj switches for each hyperplane hi,j, moving from h; to hi,j. Inspired by the algorithm proposed in [17], we develop the function SATREGIONS (Algorithm 5 in Appendix A), an incremental algorithm for discovering the convex regions in the arrangement. Intuitively, the algorithm adds the hy-perplanes one after the other to the arrangement. At every iteration, it finds the set of regions in the arrangement with which the new hyperplane intersects. Recall that hij is in the form of k=1 1 hi,j[k]0k = 1. Hence, the half-space hj can be considered as the constraint = hi,j[k]0k ≥ 1 and hijas hi,j[k]θk ≤ 1. The set of points inside a convex region R = {(hR1, +/-), (hR2, +/-),... } satisfy constraints OR as defined in Equation 6.
OR: { ✓ half-space(h', +) ∈ R, Σ¢- h'[k]θk ≥ 1 V half-space(h', −) ∈ R, Σ=¦ h'[k]θk ≤ 1 (6)

凸區域被定義為一組半空間的交集 [17]。每個超平面 h 將空間劃分為兩個半空間 h⁺ 和 h¯。tį 和 tj 之間的排序對於每個超平面 hi,j 都會切換，從 h; 移動到 hi,j。受 [17] 中提出的演算法啟發，我們開發了函數 SATREGIONS（附錄 A 中的演算法 5），這是一種用於發現排列中凸區域的增量演算法。直觀地說，該演算法將超平面一個接一個地添加到排列中。在每次迭代中，它都會找到與新超平面相交的排列中的區域集。回想一下，hij 的形式為 k=1 1 hi,j[k]0k = 1。因此，半空間 hj 可以被視為約束 = hi,j[k]0k ≥ 1 和 hijas hi,j[k]θk ≤ 1。凸區域 R = {(hR1, +/-), (hR2, +/-),... } 內的點集滿足方程式 6 中定義的約束 OR。
OR: { ✓ half-space(h', +) ∈ R, Σ¢- h'[k]θk ≥ 1 V half-space(h', −) ∈ R, Σ=¦ h'[k]θk ≤ 1 (6)

Using Equation 6, a hyperplane h intersects with a convex region R if there exists a point p∈h such that the con-straints in or are satisfied. The existence of such a point can be determined using linear programming (LP). If the new hyperplane intersects with R, Algorithm 5 breaks it down into two convex regions that represent the intersections of R with half-spaces h⁺ and h¯.

使用方程式 6，如果存在一個點 p∈h 使得 or 中的限制條件得到滿足，則超平面 h 與凸區域 R 相交。這樣一個點的存在可以透過線性規劃 (LP) 來確定。如果新的超平面與 R 相交，演算法 5 將其分解為兩個凸區域，分別代表 R 與半空間 h⁺ 和 h¯ 的交集。

[Image]

Figure 7: A dataset Figure 8: Ordering exchanges of Fig. 7 Figure 9: Angles in R³ Figure 10: Arrangement tree

圖 7：一個資料集 圖 8：圖 7 的排序交換 圖 9：R³ 中的角度 圖 10：排列樹

Having constructed the arrangement, SATREGIONS uses linear programming to find a point 0 that satisfies Or, and uses 0 to check if R is satisfactory. If R is not satisfactory, it is removed from the set of satisfactory regions R.

建構完排列後，SATREGIONS 使用線性規劃找到一個滿足 Or 的點 0，並使用 0 來檢查 R 是否令人滿意。如果 R 不令人滿意，則將其從滿意區域 R 的集合中移除。

THEOREM 3. For a fixed number of dimensions, the time complexity of the function SATREGIONS (Algorithm 5) is O(n2d-1(n Lp(n²) + Yn log n)), where Lp(n²) is the time of solv-ing a linear programming problem of n² constrains and a fixed number of variables, and Yn is the time complexity of the fair-ness oracle for an input of size n.

定理 3. 對於固定數量的維度，函數 SATREGIONS（演算法 5）的時間複雜度為 O(n2d-1(n Lp(n²) + Yn log n))，其中 Lp(n²) 是解決具有 n² 個限制和固定數量變數的線性規劃問題的時間，而 Yn 是公平性神諭對於大小為 n 的輸入的時間複雜度。

To add a new hyperplane, the algorithm SATREGIONS checks the intersection of every region with the hyperplane. But in practice most regions do not intersect with it. We define the arrangement tree (Appendix B), which keeps tracks of the space partitioning in a hierarchical manner, and can quickly rule out many regions. While this does not change the as-ymptotic worst case complexity, we find that it greatly helps in practice, as we will demonstrate experimentally in § 7.4.

為了加入一個新的超平面，演算法 SATREGIONS 會檢查每個區域與該超平面的交集。但在實務上，大多數區域並不會與它相交。我們定義了排列樹（附錄 B），它以分層的方式追蹤空間分割，並能快速排除許多區域。雖然這不會改變漸近最壞情況的複雜性，但我們發現在實務上它有很大的幫助，我們將在 § 7.4 中透過實驗證明。

### 4.3 Online processing

### 4.3 線上處理

Thus far in this section, we studied how to preprocess the data and construct satisfactory regions R in multiple dimen-sions. Next, given a query (a function f) and R, we aim to find the closest satisfactory function f' to f. To do so, MD-BASELINE (Algorithm 6 in Appendix C) solves a non-linear programming problem for each satisfactory region to find the closest point of the region to f, and returns the function with the minimum angle distance with f.

到目前為止，在本節中，我們研究了如何預處理數據並在多維中建構滿意區域 R。接下來，給定一個查詢（函數 f）和 R，我們的目標是找到最接近 f 的滿意函數 f'。為此，MD-BASELINE（附錄 C 中的演算法 6）為每個滿意區域解決一個非線性規劃問題，以找到該區域中最接近 f 的點，並返回與 f 具有最小角距離的函數。

THEOREM 4. For a constant number of dimensions, the time complexity of Algorithm 6 is O(n2(d−1)NLp(n²)), where NLp(n²) is the time for solving a non-linear programming problem of n² constraints and a fixed number of variables.

定理 4. 對於常數維數，演算法 6 的時間複雜度為 O(n2(d−1)NLp(n²))，其中 NLp(n²) 是解決具有 n² 個限制和固定數量變數的非線性規劃問題的時間。

## 5 APPROXIMATION

## 5 近似

A user developing a scoring function requires interactive response time from the system. MDBASELINE is not practi-cal for query answering as it needs to solve a non-linear programming problem for each satisfactory region, before answering each query. In this section, we propose an efficient algorithm for obtaining approximate answers quickly. Our approach relies on first partitioning the angle space, based on a user-controlled parameter N, into N cells, where each cell c is a hypercube of (d – 1)-dimensions. To do so, we use the equi-volume partitioning proposed in [19]. During preprocessing, we assign a satisfactory function f' to every cell c such that, for every function f, the angle between f and f' is within a bounded threshold (based on the value of N) from f and its optimal answer. To do so, we first identify the cells that intersect with a satisfactory region, and assign the corresponding satisfactory function to each such cell. Then, we assign the cells that are outside of the satisfactory regions to the nearest discovered satisfactory function.

開發評分函數的使用者需要系統的互動式回應時間。MDBASELINE 不適用於查詢回答，因為它需要在回答每個查詢之前，為每個滿意區域解決一個非線性規劃問題。在本節中，我們提出一個有效率的演算法，以快速取得近似答案。我們的方法首先依賴於將角度空間根據使用者控制的參數 N 分割成 N 個單元，其中每個單元 c 都是一個 (d – 1) 維的超立方體。為此，我們使用 [19] 中提出的等體積分割。在預處理期間，我們為每個單元 c 指派一個滿意函數 f'，使得對於每個函數 f，f 和 f' 之間的角度在 f 及其最佳答案的（基於 N 值的）有界閾值內。為此，我們首先識別與滿意區域相交的單元，並將對應的滿意函數指派給每個這樣的單元。然後，我們將位於滿意區域之外的單元指派給最近發現的滿意函數。

### 5.1 Identifying cells in satisfactory regions

### 5.1 識別滿意區域中的儲存格

After partitioning the angle space, our objective here is to find cells in Cells, the set of all cells, that intersect with at least one satisfactory region R ∈ R. Formally,
C = {c ∈ Cells | ∃R ∈ R s.t. R∩ c ≠ 0} (7)

在對角度空間進行分割後，我們的目標是找到 Cells（所有儲存格的集合）中與至少一個滿意區域 R ∈ R 相交的儲存格。形式上，
C = {c ∈ Cells | ∃R ∈ R s.t. R∩ c ≠ 0} (7)

A brute force algorithm follows Equation 7 literally. This algorithm needs to first construct a complete arrangement and then check the intersection of all N × |R| pairs of cells and satisfactory regions. This is inefficient when N and the size of R are large. As discussed in § 4, and experimentally shown in § 7, the complexity of the arrangement and the run-ning time of the algorithm SATREGIONS highly depends on the number of hyperplanes in the arrangement. Even though the first few hyperplanes are quickly added to the arrange-ment, adding the later hyperplanes is more time consuming. This observation motivates us to limit the construction of the arrangement to subsets of hyperplanes, as opposed to constructing the complete arrangement all at once. Besides, the changes in the ordering in every cell is limited to the hy-perplanes passing through it. As a result, for finding out if a cell intersects with a satisfactory region, it is enough to only consider the arrangement of these hyperplanes. In Appen-dix D, we explain how to efficiently identify the hyperplanes passing through each cell.

暴力演算法直接遵循方程式 7。此演算法需要先建構一個完整的排列，然後檢查所有 N × |R| 對儲存格和滿意區域的交集。當 N 和 R 的大小很大時，這是低效的。如 § 4 所述，並在 § 7 中實驗證明，排列的複雜度和演算法 SATREGIONS 的執行時間高度依賴於排列中超平面的數量。儘管前幾個超平面很快被加入到排列中，但加入後面的超平面更耗時。這個觀察促使我們將排列的建構限制在超平面的子集上，而不是一次建構完整的排列。此外，每個儲存格中排序的變化僅限於穿過它的超平面。因此，為了找出一個儲存格是否與一個滿意區域相交，只考慮這些超平面的排列就足夠了。在附錄 D 中，我們解釋瞭如何有效地識別穿過每個儲存格的超平面。

After identifying HC (the sets of hyperplanes passing through the cells), for each cell c ∈ Cells, we limit the ar-rangement to HC[c]. Moreover, note that in this step our goal is to find a satisfactory function inside c. This is differ-ent from our objective in SATREGIONS, where we wanted to find all satisfactory regions. This gives us the opportunity to apply a stop early strategy, as follows: at every iteration, while using the arrangement tree for construction, check a function inside the newly added regions, and stop as soon as a satisfactory function is discovered.

在識別出 HC（穿過儲存格的超平面集合）後，對於每個儲存格 c ∈ Cells，我們將排列限制為 HC[c]。此外，請注意，在此步驟中，我們的目標是在 c 內部找到一個令人滿意的函數。這與我們在 SATREGIONS 中的目標不同，後者旨在找到所有令人滿意的區域。這為我們提供了應用提早停止策略的機會，如下所示：在每次迭代中，在使用排列樹進行建構時，檢查新加入區域內的函數，並在發現令人滿意的函數後立即停止。

[Image]

Figure 11: Cells that intersect a plane Figure 12: Early stopping in arrangement construction

圖 11：與平面相交的儲存格 圖 12：排列建構中的提早停止

We develop the algorithm MARKCELL (Algorithm 7 in Ap-pendix D) that assigns a satisfactory function to the cells that intersect with a satisfactory region R. Using an arrange-ment tree, the algorithm iteratively adds the hyperplanes passing through each cell and checks if a function inside the new regions is satisfactory. Upon finding a satisfactory function, the algorithm stops and assigns the function to the cell. Figure 12 illustrates how MARKCELL finds a satisfac-tory function for a cell c. After adding hyperplanes hc₁ and hc2, since functions f₁ to fo are unsatisfactory (denoted by red color), MARKCELL adds hc3 to the construction. In this example, hc3 does not pass through {hcī, hc₂ }, but it passes through R = {hcī, hc+ }, dividing it into R₁ = R ∪ hc and R₁ = R Uhc. Although f₁ ∈ R₁ is unsatisfactory, f8 ∈ R, is satisfactory. The algorithm assigns fs to c and stops.

我們開發了演算法 MARKCELL（附錄 D 中的演算法 7），它將一個滿意函數分配給與滿意區域 R 相交的儲存格。該演算法使用排列樹，迭代地加入穿過每個儲存格的超平面，並檢查新區域內是否有一個函數是滿意的。一旦找到一個滿意函數，演算法就會停止並將該函數分配給該儲存格。圖 12 說明了 MARKCELL 如何為儲存格 c 找到一個滿意函數。在加入超平面 hc₁ 和 hc2 之後，由於函數 f₁ 到 fo 都不滿意（以紅色表示），MARKCELL 將 hc3 加入建構中。在此範例中，hc3 不穿過 {hcī, hc₂ }，但穿過 R = {hcī, hc+ }，將其劃分為 R₁ = R ∪ hc 和 R₁ = R Uhc。雖然 f₁ ∈ R₁ 不滿意，但 f8 ∈ R, 是滿意的。演算法將 fs 分配給 c 並停止。

Let |HC[c]| be the number of hyperplanes passing through cell c; the complexity of the arrangement of c is O(|HC[c]|d−1). Then the complexity of MARKCELL iS O(|HC[c]|dLp(|HC[c]|)+ |HC[c]|d−¹n log non) when d is fixed, following Theorem 3.

令 |HC[c]| 為穿過儲存格 c 的超平面數量；c 的排列複雜度為 O(|HC[c]|d−1)。那麼，當 d 固定時，根據定理 3，MARKCELL 的複雜度為 O(|HC[c]|dLp(|HC[c]|)+ |HC[c]|d−¹n log non)。

So far, we identified cells C that intersect with some satis-factory region, and assigned a satisfactory function to each of them. Next, we consider the cells C that do not contain a satisfactory function and assign them to the closest discov-ered satisfactory function. To do so, we use monotonicity of the angular distance and adopt Dijkstra's algorithm [20], see Appendix E for details. After this step, and assuming the existence of at least one satisfactory region, every cell in the partitioned angle space is assigned a satisfactory function. We store the cell coordinates, together with the assigned satisfactory functions, as an index that enables online an-swering of user queries, discussed next.

到目前為止，我們識別了與某些滿意區域相交的儲存格 C，並為每個儲存格分配了一個滿意函數。接下來，我們考慮不包含滿意函數的儲存格 C，並將它們分配給最接近的已發現滿意函數。為此，我們利用角距離的單調性並採用 Dijkstra 演算法 [20]，詳情請參閱附錄 E。在此步驟之後，並假設至少存在一個滿意區域，分割角度空間中的每個儲存格都被分配了一個滿意函數。我們將儲存格座標與分配的滿意函數一起儲存為索引，以便能夠線上回答使用者查詢，詳情將在下文討論。

### 5.2 Online processing

### 5.2 線上處理

Given an unsatisfactory function f, we need to find the cell to which f belongs, and to return its satisfactory function. This is implemented in MDONLINE (Algorithm 3) that trans-forms the weight vector of f to polar coordinates, performs binary search on each dimension to identify cell c to which f belongs, and return its satisfactory function F[c].

給定一個不令人滿意的函數 f，我們需要找到 f 所屬的儲存格，並返回其令人滿意的函數。這在 MDONLINE（演算法 3）中實現，它將 f 的權重向量轉換為極座標，在每個維度上執行二進位搜尋以識別 f 所屬的儲存格 c，並返回其令人滿意的函數 F[c]。

Algorithm 3 MDONLINE
Input: partitioned space T, assigned functions F, dataset D, fairness oracle O, and weight vector w
Output: satisfactory weight vector w'

演算法 3 MDONLINE
輸入：分割空間 T、指派函數 F、資料集 D、公平性神諭 O 及權重向量 w
輸出：滿意的權重向量 w'

THEOREM 5. Algorithm MDONLINE runs in O(log N) time.

定理 5. 演算法 MDONLINE 的執行時間為 O(log N)。

THEOREM 6. Let fopt and Oopt be the closest function and its angle distance to a queried function f. Also, let fapp and Oapp be the function and its angle distance that the algorithm MDONLINE returns for f, and 0, the diameter the cells. Then, Oapp ≤ 0opt + 20,.

定理 6. 令 fopt 和 Oopt 為最接近查詢函數 f 的函數及其角距離。此外，令 fapp 和 Oapp 為演算法 MDONLINE 對 f 返回的函數及其角距離，而 0 為儲存格的直徑。則 Oapp ≤ 0opt + 20,。

## 6 SAMPLING

## 6 抽樣

In this section, we describe two sampling-based approaches that improve the performance of our preprocessing and on-line processing: item sampling, and function sampling, re-spectively. A critical requirement of our system is to be effi-cient during online query processing, and it is fine for it to spend more time during offline preprocessing. As discussed in § 4 and § 5, the running time of proposed offline algo-rithms is polynomial for a fixed value of d. In addition, the arrangement tree (c.f. § 4) and the techniques of § 5 speed up preprocessing in practice. However, preprocessing can still be slow, particularly for a large number of items. We reduce preprocessing time using item sampling. We also present a negative result about function sampling, and show how it provides a practical method for on-the-fly query processing.

在本節中，我們描述了兩種基於抽樣的方法，分別是項目抽樣和函數抽樣，它們可以改善我們的預處理和線上處理的效能。我們系統的一個關鍵要求是在線上查詢處理期間要有效率，而在離線預處理期間花費更多時間是可以接受的。如 § 4 和 § 5 所述，對於固定的 d 值，所提出的離線演算法的執行時間是多項式的。此外，排列樹（參見 § 4）和 § 5 的技術在實務上加快了預處理速度。然而，預處理仍然可能很慢，特別是對於大量的項目。我們使用項目抽樣來減少預處理時間。我們還提出了關於函數抽樣的負面結果，並展示了它如何為即時查詢處理提供一種實用的方法。

### 6.1 Item Sampling

### 6.1 項目抽樣

The arrangement construction cost increases significantly for a large number of items in the dataset. On the other hand, in practice, fairness criteria often allow some degree of freedom in the ranking between the items. For example, a popular class of fairness models treat the top-k items in the ranking as a set, and if the proportion of protected group members in the set is within an acceptable range, they consider the over-all ranking to be fair. Having at least 20% minorities and at least 30% females in the top-20% of the ranking is an example of such a fairness model. We propose to use item sampling for these situations.

對於資料集中大量的項目，排列建構的成本會顯著增加。另一方面，在實務上，公平性標準通常允許項目之間的排名有一定的自由度。例如，一類流行的公平性模型將排名中的前 k 個項目視為一個集合，如果集合中受保護群體成員的比例在可接受的範圍內，他們就認為整體排名是公平的。例如，在排名前 20% 中至少有 20% 的少數族裔和至少 30% 的女性，就是這種公平性模型的一個例子。我們建議在這些情況下使用項目抽樣。

The main idea is that a uniform sample of the data main-tains the properties of the underlying data distribution. There-fore, if a function is satisfactory for a dataset D, it is expected to be satisfactory for a uniformly sampled subset of D. Hence, for a datasets with a large number of items, one can do pre-processing on a uniformly sampled subset to find functions that are expected to be satisfactory for each cell. We confirm the efficiency and effectiveness of this method experimen-tally on a dataset with over one million items in § 7.

主要思想是，資料的均勻樣本能維持基礎資料分佈的特性。因此，如果一個函數對於資料集 D 是令人滿意的，那麼它對於 D 的一個均勻抽樣的子集也應該是令人滿意的。因此，對於具有大量項目的資料集，可以在一個均勻抽樣的子集上進行預處理，以找到預期對每個儲存格都令人滿意的函數。我們在 § 7 中對一個超過一百萬個項目的資料集進行了實驗，證實了這種方法的效率和有效性。

### 6.2 Function sampling

### 6.2 函數抽樣

Every origin-starting ray passes through one and only one point on the surface of the unit d-dimensional hyper-sphere (known as the d-sphere). This maps the universe of origin-starting rays (linear functions) to the surface of the first quadrant of the unit d-sphere. As a result, uniform sampling from this surface provides uniform samples from the func-tion space. Using this observation, [21, 22] propose uniform sampling of the function space, by adopting the methods of [23, 24]. The idea is that, since the Normal distribution has a constant probability on the surfaces of d-spheres with common centers, taking samples based on this distribution from the weight space provides uniform samples from the function space. This idea is extended in [22] for taking unbi-ased samples in the e-vicinity of a specific ray. We can use this for on-the-fly query processing.

每條從原點出發的射線都會穿過單位 d 維超球面（稱為 d 球面）表面上的一個且僅一個點。這將從原點出發的射線（線性函數）的宇宙對應到單位 d 球面第一象限的表面。因此，從該表面進行均勻抽樣可提供函數空間的均勻樣本。利用此觀察，[21, 22] 採用 [23, 24] 的方法，提出對函數空間進行均勻抽樣。其思想是，由於常態分佈在具有共同中心的 d 球面表面上具有恆定的機率，因此從權重空間中基於此分佈進行抽樣可提供函數空間的均勻樣本。此思想在 [22] 中被擴展，用於在特定射線的 ε-鄰域內進行無偏抽樣。我們可以用它來進行即時查詢處理。

In this paper, we conduct preprocessing that enables an efficient way of answering user queries. The alternative sce-nario is on-the-fly processing of the queries, without any preprocessing. Consider the case where the objective is to find a satisfactory function that has at most the angle dis-tance of @ with the user-provided function, called the region of interest in [22]. Constructing the arrangement at query time is intractable. Instead, one can use uniform random samples for exploring the neighborhood. To do so, we take a uniform random function sample in the e-vicinity of the input function and return it if satisfactory. Otherwise, we take another sample, and continue until a budget of s sam-ples is exhausted. Unfortunately, the negative result is that, based on Theorem 7, function sampling cannot provide any guarantees for the discovery of an approximation solution.

在本文中，我們進行預處理，以便能有效回答使用者查詢。另一種情況是即時處理查詢，無需任何預處理。考慮目標是找到一個與使用者提供的函數角度距離最多為 @ 的滿意函數，這在 [22] 中稱為感興趣區域。在查詢時建構排列是難以處理的。相反，可以使用均勻隨機樣本來探索鄰域。為此，我們在輸入函數的 ε-鄰域內取一個均勻隨機函數樣本，如果滿意則返回。否則，我們再取一個樣本，直到用盡 s 個樣本的預算。不幸的是，負面結果是，根據定理 7，函數抽樣無法為發現近似解提供任何保證。

THEOREM 7. For any arbitrarily small probability p > 0 and any arbitrarily large number s, one cannot guarantee the discovery of a satisfactory function with probability at least p, using s uniform random function samples.

定理 7. 對於任意小的機率 p > 0 和任意大的數字 s，使用 s 個均勻隨機函數樣本，無法保證以至少 p 的機率發現一個令人滿意的函數。

Although function sampling is efficient in drawing a func-tion from large satisfactory regions, one cannot guarantee that all possible rankings will be discovered and, therefore, cannot ensure the non-existence of a satisfactory function.

雖然函數抽樣在從大型滿意區域中抽取函數方面效率很高，但它無法保證所有可能的排名都會被發現，因此也無法確保不存在滿意函數。

On the other hand, as studied in [22], the rankings sup-ported by small regions are unstable, and thus question-able. Following the function sampling strategy for on-the-fly query processing, we expect to find a satisfactory function, if the volume ratio of the satisfactory regions to the volume of the region of interest is more than 1/s. Taking more samples increases the chance of hitting smaller satisfactory regions, but reduces efficiency. Given that we stop exploring as soon as a satisfactory function is discovered, as we shall show in § 7, this method is efficient in practice, for the cases in which it finds a satisfactory function. However, in other cases, it may fail to find a satisfactory function although one exists.

另一方面，如 [22] 所研究，小區域支援的排名不穩定，因此值得懷疑。遵循即時查詢處理的函數抽樣策略，如果滿意區域的體積與感興趣區域的體積之比大於 1/s，我們期望找到一個滿意函數。取更多樣本會增加命中較小滿意區域的機會，但會降低效率。鑑於我們一發現滿意函數就停止探索，如我們將在 § 7 中展示的，這種方法在實務中對於找到滿意函數的情況是有效率的。然而，在其他情況下，即使存在滿意函數，它也可能找不到。

## 7 EXPERIMENTAL EVALUATION

## 7 實驗評估

### 7.1 Experimental Setup

### 7.1 實驗設定

Datasets. COMPAS: a dataset collected and published by ProPublica as part of their investigation into racial bias in criminal risk assessment software [25]. The dataset con-tains demographics, recidivism scores produced by the COM-PAS software, and criminal offense information for 6,889 individuals. We used c_days_from_compas, juv_other_count, days_b_screening_arrest, start, end, age, and priors_count as scoring attributes. We normalized attribute values as (val-min)/(max – min). For all attributes except age, a higher value corresponded to a higher score. In addition to the scor-ing attributes, we consider attributes sex (0:male, 1: female), age_binary (0: less than 35 yo, 1: more than 36 yo), race (0: African American, 1: Caucasian, 2: Other), and age_bucketized (0: less than 30 yo, 1: 31 to 40 yo, 2: more than 40 yo), as the type attributes. COMPAS is our default experimental dataset.

資料集。COMPAS：由 ProPublica 收集並發布的資料集，作為其對刑事風險評估軟體中種族偏見調查的一部分 [25]。該資料集包含 6,889 名個人的個人資料、COMPAS 軟體產生的再犯分數以及刑事犯罪資訊。我們使用 c_days_from_compas、juv_other_count、days_b_screening_arrest、start、end、age 和 priors_count 作為評分屬性。我們將屬性值標準化為 (val-min)/(max – min)。除了年齡之外的所有屬性，較高的值對應較高的分數。除了評分屬性，我們還考慮屬性 sex (0:男性, 1:女性)、age_binary (0: 小於 35 歲, 1: 大於 36 歲)、race (0: 非裔美國人, 1: 白種人, 2: 其他) 和 age_bucketized (0: 小於 30 歲, 1: 31 至 40 歲, 2: 大於 40 歲) 作為類型屬性。COMPAS 是我們的預設實驗資料集。

US Department of Transportation (DOT): the flight on-time database published by DOT is widely used by third-party websites [26]. We collected 1.3M records, for the flights con-ducted by 14 US carriers in the first three months of 2016. We use this to study sampling for large-scale settings, and to showcase the application of our techniques for diversity.

美國運輸部 (DOT)：由 DOT 發布的航班準點資料庫被第三方網站廣泛使用 [26]。我們收集了 2016 年前三個月由 14 家美國航空公司執飛的 130 萬筆航班記錄。我們用它來研究大規模設置的抽樣，並展示我們的技術在多樣性方面的應用。

Hardware and platform. The experiments were performed on a Linux machine with a 2.6 GHz Core I7 CPU and 8GB memory. The algorithms were implemented using Python2.7, with scipy.optimize package for LP optimizations.

硬體與平台。實驗在配備 2.6 GHz Core I7 CPU 和 8GB 記憶體的 Linux 機器上進行。演算法使用 Python 2.7 實現，並使用 scipy.optimize 套件進行 LP 優化。

Fairness models. We evaluate the performance of our meth-ods over two general fairness models, see § 2.

公平性模型。我們在兩種通用的公平性模型上評估我們方法的效能，請參閱 § 2。

FM1, proportional representation on a single type attribute, is the default fairness model in our experiments. This model can express common proportionality constraints from the literature [3, 7, 8], including also for ranked outputs [6] and for set selection [4]. The distinguishing features of FM1 are (1) that the type attribute partitions the input dataset D into groups and (2) that the proportion of members of a particular group is bounded from below, from above, or both. For the COMPAS dataset, unless noted otherwise, we state FM1 over the type attribute race as follows: African Americans consti-tute about 50% of the dataset; a fairness oracle will consider a ranking to be satisfactory if at most 60% (or about 10% more than in D) of the top-ranked 30% are African American.

FM1，單一類型屬性上的比例代表，是我們實驗中的預設公平性模型。此模型可以表達文獻中常見的比例性限制 [3, 7, 8]，也包括排名輸出 [6] 和集合選擇 [4]。FM1 的顯著特徵是 (1) 類型屬性將輸入資料集 D 分割成群組，以及 (2) 特定群組成員的比例有下限、上限或兩者皆有。對於 COMPAS 資料集，除非另有說明，我們將 FM1 陳述在類型屬性 race 上，如下：非裔美國人約佔資料集的 50%；如果排名前 30% 的非裔美國人最多佔 60%（或比 D 中多約 10%），則公平性神諭會認為排名是令人滿意的。

[Image]

Figure 13: Angle between input/output functions Figure 14: 2D; preprocess- ing time, varying n Figure 15: The advantage of arrangement tree Figure 16: Incremental ar- rangement cost (d = 3)

圖 13：輸入/輸出函數之間的角度 圖 14：二維；預處理時間，隨 n 變化 圖 15：排列樹的優點 圖 16：增量排列成本 (d = 3)

FM2, proportional representation on multiple, possibly overlapping, type attributes, is a generalization of FM1 that can express proportionality constraints of [10]. As in [10], we bound the number of members of a group from above. For example, for COMPAS, we specify the maximum number of items among the top-ranked 30% based on sex (80% of D are male), race (50% are African American), and age_bucketized (42% are 30 years old or younger, 34% are between 31 and 50, and 24% are over 50). A ranking is considered satisfactory if the proportion of members of a particular demographic group is no more than 10% higher than its proportion in D.

FM2，在多個可能重疊的類型屬性上的比例代表，是 FM1 的一個推廣，可以表達 [10] 的比例性限制。如同 [10]，我們從上方限制一個群體的成員數量。例如，對於 COMPAS，我們根據性別（D 中 80% 是男性）、種族（50% 是非裔美國人）和 age_bucketized（42% 是 30 歲或更年輕，34% 在 31 到 50 歲之間，24% 超過 50 歲）來指定排名前 30% 的項目中的最大數量。如果特定人口群體的成員比例不高於其在 D 中比例的 10%，則認為排名是令人滿意的。

### 7.2 Validation experiments

### 7.2 驗證實驗

In our first experiment, we show that our methods are ef-fective - that they can identify scoring functions that are both satisfactory and similar to the user's query. We use COMPAS with d = 3 (scoring attributes c_days_from_compas, juv_other_count, start), and with fairness model FM1 on race (at most 60% African Americans among the top 30%).

在我們的第一個實驗中，我們證明了我們的方法是有效的——它們可以識別出既令人滿意又與使用者查詢相似的評分函數。我們使用 COMPAS，d = 3（評分屬性 c_days_from_compas, juv_other_count, start），並在種族上使用公平性模型 FM1（前 30% 中最多 60% 的非裔美國人）。

We issued 100 random queries, and observed that 52 of them were satisfactory, and so no further intervention was needed. For the remaining 48 functions, we used our meth-ods to suggest the nearest satisfactory function. Figure 13 presents a cumulative plot of the results for these 48 cases, showing the angle distance (f, f') between the input f and the output f' on the x-axis, and the number of queries with at most that distance on the y-axis.

我們發出了 100 個隨機查詢，觀察到其中 52 個是令人滿意的，因此不需要進一步干預。對於其餘 48 個函數，我們使用我們的方法來建議最接近的令人滿意的函數。圖 13 呈現了這 48 個案例結果的累積圖，x 軸顯示輸入 f 和輸出 f' 之間的角度距離 (f, f')，y 軸顯示最多具有該距離的查詢數量。

We observe that a satisfactory function f' was found close to the input function f in all cases. Specifically, note that 0(f, f') < 0.6 in all cases, and recall that θ ∈ [0, π/2], with lower values corresponding to higher similarity. (For a more intuitive measure: the value of 0 = 0.6 corresponds to cosine similarity of 0.82, where 1 is best, and 0 is worst). Among the 48 cases, 38 had (f, f') < 0.4 (cosine similarity 0.92).

我們觀察到，在所有情況下，都找到了接近輸入函數 f 的滿意函數 f'。具體來說，請注意在所有情況下 0(f, f') < 0.6，並回想 θ ∈ [0, π/2]，較低的值對應較高的相似度。（為更直觀地衡量：0 = 0.6 的值對應於 0.82 的餘弦相似度，其中 1 是最好的，0 是最差的）。在這 48 個案例中，有 38 個的 (f, f') < 0.4（餘弦相似度 0.92）。

Next, we give an intuitive understanding of the layout of satisfactory regions in the space of ranking functions. We use COMPAS with age (lower is better) and juv_other_count (higher is better) for scoring. The intuition behind this scor-ing function is that individuals who are younger, and who have a higher number of juvenile offenses, are considered to be more likely to re-offend, and so may be given higher priority for particular supportive services or interventions.

接下來，我們將直觀地了解排名函數空間中滿意區域的佈局。我們使用 COMPAS 的年齡（越低越好）和 juv_other_count（越高越好）進行評分。這個評分函數背後的直覺是，年紀較輕且有較多青少年犯罪記錄的個人，被認為更有可能再犯，因此可能會在特定的支持服務或干預措施中獲得更高的優先順序。

Naturally, a scoring function that associates a high weight with age will include mostly members of the younger age group at top ranks. About 60% of COMPAS are 35 years old or younger. Consider a fairness oracle that uses FM1 over age_binary (with groups 91: 35 year old or younger, and 92: over 35 years old), and that considers a ranking satisfactory if at most 70% of the top-100 results are in g₁. Because of the correlation (by design) between one of the scoring attributes and the type attribute, there is only one satisfactory region for this problem set-up it corresponds to the set of func-tions in which the weight on age is close to 0, and whose angle with the x-axis (juv_other_count) is at most 0.31.

很自然地，一個與年齡高度相關的評分函數會將年輕年齡組的成員主要排在前面。COMPAS 中約 60% 的人年齡在 35 歲或以下。考慮一個在 age_binary 上使用 FM1 的公平性神諭（群組 91：35 歲或以下，92：35 歲以上），如果 top-100 結果中最多 70% 在 g₁ 中，則認為排名是令人滿意的。由於評分屬性之一與類型屬性之間的相關性（設計使然），這個問題設定只有一個令人滿意的區域——它對應於年齡權重接近 0，且與 x 軸（juv_other_count）的角度最多為 0.31 的函數集。

Next, suppose that we use the same scoring attributes, but a different fairness oracle one that applies FM1 on the attribute race, requiring that at most 60 of the top-100 are African American. This time, there exist several satisfactory regions. For any assignment of weights to the two scoring attributes, there exists a satisfactory function f' such that 0(f, f') < 0.11 (cosine similarity is more than 0.99).

接下來，假設我們使用相同的評分屬性，但使用不同的公平性神諭——一個在屬性 race 上應用 FM1 的神諭，要求 top-100 中最多有 60 名非裔美國人。這次，存在幾個令人滿意的區域。對於這兩個評分屬性的任何權重分配，都存在一個令人滿意的函數 f'，使得 0(f, f') < 0.11（餘弦相似度大於 0.99）。

Finally, we use juv_other_count and c_days_from_compas for scoring, with fairness model FM2 that considers a ranking satisfactory if there are at most 90 males, at most 60 African Americans, and at most 52 persons who are 30 years old or younger at the top-100. This fairness model is stricter than in the preceding experiment (with FM1 on race), making the gaps between the satisfactory regions wider. Still, the maximum angle between f and f' was less than 0.28, which corresponds to the minimum cosine similarity of 0.96.

最後，我們使用 juv_other_count 和 c_days_from_compas 進行評分，並採用公平性模型 FM2，如果排名前 100 名中最多有 90 名男性、最多 60 名非裔美國人、以及最多 52 名 30 歲或以下的人，則認為排名是令人滿意的。這個公平性模型比前一個實驗（在種族上使用 FM1）更嚴格，使得滿意區域之間的差距更寬。儘管如此，f 和 f' 之間的最大角度小於 0.28，這對應於 0.96 的最小餘弦相似度。

### 7.3 Performance of query answering

### 7.3 查詢回答效能

While preprocessing can take more time, a critical require-ment of our system is to be fast when answering users' queries. In this section, we use the COMPAS dataset and evaluate the performance of 2DONLINE and MDONLINE, the 2D and MD algorithms for online query answering. We show that queries can be answered in interactive time. We use the default fairness model (i.e., at most 60% African Americans in the top-30%) and the scoring attributes in the same ordering provided in the description of COMPAS dataset.

雖然預處理可能需要更多時間，但我們系統的一個關鍵要求是在回答使用者查詢時要快速。在本節中，我們使用 COMPAS 資料集並評估 2DONLINE 和 MDONLINE（用於線上查詢回答的 2D 和 MD 演算法）的效能。我們證明查詢可以在互動時間內得到回答。我們使用預設的公平性模型（即，top-30% 中最多 60% 的非裔美國人）和 COMPAS 資料集描述中提供的相同順序的評分屬性。

2D. 2DONLINE does not need to access the raw data at query time. It only needs to apply binary search on the sorted list of satisfactory ranges to locate the position of the input function f. In this experiment, we compare the required time for ordering the results based on the input function, averaged over 30 runs of 2DONLINE on random inputs. Con-firming the theoretical O(log n) complexity of 2DONLINE V.S. the O(n log n) for the ordering, 2DONLINE only required 30 µsec on average, while even ordering the results based on f (to check if f is satisfactory) required 25 msec to complete.

2D. 2DONLINE 在查詢時不需要存取原始資料。它只需要在已排序的滿意範圍清單上應用二進位搜尋，以定位輸入函數 f 的位置。在此實驗中，我們比較了根據輸入函數排序結果所需的時間，在 30 次隨機輸入的 2DONLINE 執行中取平均值。確認 2DONLINE 的理論 O(log n) 複雜度與排序的 O(n log n) 複雜度相比，2DONLINE 平均只需要 30 微秒，而即使根據 f 排序結果（以檢查 f 是否滿意）也需要 25 毫秒才能完成。

MD. In this experiment, just as for 2D, we took the average running time of 30 random queries. Upon arrival of a query function f, MDONLINE finds the cell to which f belongs in O(log N), where N is the number of cells, and returns the proper satisfactory function f'. In all experiments the run-ning time was less than 200 µsec whereas the time required to order the items was 25 msec.

MD. 在這個實驗中，如同 2D，我們取了 30 個隨機查詢的平均執行時間。當查詢函數 f 到達時，MDONLINE 在 O(log N) 時間內找到 f 所屬的儲存格，其中 N 是儲存格的數量，並返回適當的滿意函數 f'。在所有實驗中，執行時間都小於 200 微秒，而排序項目所需的時間為 25 毫秒。

[Image]

Figure 17: MD; effect of n on |H| (d = 3) Figure 18: MD; # of hyper- planes intersecting a cell Figure 19: MD; effect of n on preprocessing time Figure 20: MD; effect of d on preprocessing time

圖 17：MD；n 對 |H| 的影響 (d = 3) 圖 18：MD；與儲存格相交的超平面數量 圖 19：MD；n 對預處理時間的影響 圖 20：MD；d 對預處理時間的影響

[Image]

Figure 21: On-the-fly pro- cessing in π/20-vicinity Figure 22: On-the-fly pro- cessing; varying vicinity

圖 21：在 π/20 鄰域內的即時處理 圖 22：在不同鄰域內的即時處理

### 7.4 Performance of preprocessing

### 7.4 預處理效能

To study preprocessing performance, we again use the COM-PAS dataset, with the default fairness model (at most 60% African Americans at the top 30%).

為了研究預處理效能，我們再次使用 COMPAS 資料集，並採用預設的公平性模型（前 30% 中最多 60% 的非裔美國人）。

2D. We start by evaluating the efficiency of 2DRAYSWEEP, the 2D preprocessing algorithm proposed in § 3. We study the effect of n (the number of items in the dataset) on the performance of the algorithm 2DRAYSWEEP and evaluate the number of ordering exchanges and the running time of it. Figure 14 shows the experiment results for varying the num-ber of items from 200 to 6,800. The x-axis shows the values of n (in log-scale), and the left and right y-axes show the running time of 2DRAYSWEEP and the number of ordering exchanges, respectively. Looking at the right y-axis, one can observe that the number of ordering exchanges is much smaller than the theoretical O(n²) upper-bound. For example, while the upper-bound on the number of ordering exchanges for n = 4k is 16M, the observed number in this experiment was around 400k. This is because the pairs of items in which one dominates the other do not have an ordering exchange. Also, looking at the left y-axis, one can confirm that the algorithm managed to finish the preprocessing within the reasonable time of 80 sec, even for the largest setting.

2D. 我們首先評估 § 3 中提出的 2D 預處理演算法 2DRAYSWEEP 的效率。我們研究 n（資料集中的項目數）對演算法 2DRAYSWEEP 效能的影響，並評估其排序交換次數和執行時間。圖 14 顯示了將項目數從 200 變為 6,800 的實驗結果。x 軸顯示 n 的值（以對數尺度），左右 y 軸分別顯示 2DRAYSWEEP 的執行時間和排序交換次數。觀察右 y 軸，可以發現排序交換次數遠小於理論上的 O(n²) 上限。例如，當 n = 4k 時，排序交換次數的上限為 16M，而本實驗中觀察到的數字約為 400k。這是因為其中一個項目支配另一個項目的項目對沒有排序交換。此外，觀察左 y 軸，可以確認即使在最大設定下，演算法也能在 80 秒的合理時間內完成預處理。

MD, the effect of using arrangement tree. In § 4, we pro-posed the arrangement tree data structure for constructing the arrangement of hyperplanes, in order to skip compar-ing a new hyperplane with all current regions. Here, as the first MD experiment, we run the algorithm SATREGIONS as the baseline and also use AT+ for adding the hyperplanes using the arrangement tree. Figure 15 shows the incremental cost of adding hyperplanes to the arrangement when d = 3. While the baseline (SATREGIONS) needed 8,000 sec for adding the first 250 hyperplanes, using the arrangement tree helped save around 7,740 secs. Fixing the budget to 8,000 sec, the baseline could construct the arrangement for the first 250 hyperplanes, while using the arrangement tree allowed us to extend the construction to 1,200 hyperplanes.

MD，使用排列樹的效果。在 § 4 中，我們提出了用於建構超平面排列的排列樹資料結構，以跳過將新超平面與所有當前區域進行比較。在這裡，作為第一個 MD 實驗，我們將演算法 SATREGIONS 作為基準，並使用 AT+ 來使用排列樹加入超平面。圖 15 顯示了當 d = 3 時，向排列中加入超平面的增量成本。雖然基準（SATREGIONS）需要 8,000 秒來加入前 250 個超平面，但使用排列樹幫助節省了約 7,740 秒。將預算固定為 8,000 秒，基準可以為前 250 個超平面建構排列，而使用排列樹則允許我們將建構擴展到 1,200 個超平面。

Recall from §4 that the number of regions at step i is O(i2(d−1)), and hence, adding the subsequent hyperplanes is more expensive. This is presented in Figure 16, where the y-axis shows the number of regions in the arrangement (|R|) for different number of hyperplanes. Observe that the num-ber of regions for the first 50 hyperplanes is less than 200; it increases to more than 5,000 regions for the hyperplanes that are added after 250th iteration. As a result, while adding a hy-perplane (without using the arrangement tree) at the first 50 iterations requires checking fewer than 200 regions, adding a hyperplane after iteration 250 requires checking more than 5,000 regions, and so is significantly more expensive.

回想 §4，第 i 步的區域數為 O(i2(d−1))，因此，加入後續的超平面會更昂貴。這在圖 16 中呈現，其中 y 軸顯示了不同超平面數量的排列中的區域數 (|R|)。觀察到，前 50 個超平面的區域數少於 200；在第 250 次迭代後加入的超平面，區域數增加到 5,000 多個。因此，雖然在前 50 次迭代中加入一個超平面（不使用排列樹）需要檢查少於 200 個區域，但在第 250 次迭代後加入一個超平面需要檢查 5,000 多個區域，因此成本顯著更高。

MD, preprocessing. We now evaluate the algorithms pro-posed in § 5 for preprocessing. Recall that in § 5, we partition the angle space into N cells and assign a satisfactory function to each cell. This enables interactive query processing, since, for each query f, we simply return the satisfactory func-tion of the cell to which f belongs. Theorem 6 provides an upper-bound on the quality of the approximation introduced by the algorithm. Following the upper-bound for d = 3 and N = 40k, for example, the maximum distance of the approx-imate output to the input function is about 0.004 degrees more than the optimal distance. We could not evaluate an average approximation, as the exact baseline solution did not finish for any of the settings.

MD，預處理。我們現在評估 § 5 中提出的預處理演算法。回想一下，在 § 5 中，我們將角度空間分割成 N 個儲存格，並為每個儲存格分配一個滿意函數。這使得互動式查詢處理成為可能，因為對於每個查詢 f，我們只需返回 f 所屬儲存格的滿意函數。定理 6 提供了演算法引入的近似品質的上限。例如，遵循 d = 3 和 N = 40k 的上限，近似輸出與輸入函數的最大距離比最佳距離多約 0.004 度。我們無法評估平均近似值，因為精確的基準解決方案在任何設定下都無法完成。

First, similar to the 2D experiments, varying n from 200 to 6,000, in Figure 17 we observe |H| (the number of hyper-planes) as well as the time for constructing the hyperplanes in the angle coordinate system. Comparing this figure with Figure 14 (remember that intersections in 2D and hyper-planes in MD refer to the ordering exchanges), we observe that |H| gets closer to n² as the number of dimensions in-crease. This is because, as the number of dimensions in-creases, the probability that one in a pair of items dominate the other decreases, and therefore |H| gets closer to n². Also, looking at the right y-axis and the dashed orange line, and comparing it with |H| (the left y-axis) confirms that the total running time is linear to the number of hyperplanes.

首先，與二維實驗類似，將 n 從 200 變為 6,000，在圖 17 中我們觀察到 |H|（超平面的數量）以及在角度座標系統中建構超平面的時間。將此圖與圖 14（記住二維中的交點和多維中的超平面指的是排序交換）進行比較，我們觀察到隨著維數的增加，|H| 越來越接近 n²。這是因為，隨著維數的增加，一對項目中一個支配另一個的機率降低，因此 |H| 越來越接近 n²。此外，觀察右 y 軸和虛線橙色線，並將其與 |H|（左 y 軸）進行比較，證實了總執行時間與超平面的數量成線性關係。

In the previous experiment, we observed the major effect of the number of hyperplanes on the construction time. Thus, in § 5, we limit the arrangement construction for each cell to the hyperplanes passing through it. In Figure 18, setting n = 100 and d = 4, we evaluated the number of hyperplanes passing through the cells. The x-axis in Figure 18 is the cells sorted by |HC[c]| (the number of hyperplanes passing through a cell c), and the y-axis shows |HC[c]|. More than 80% of the cells have fewer than 100 hyperplanes passing through them, and even the complete arrangement construc-tion inside them is reasonable. Also, recall from § 5 that MARKCELL stops arrangement construction as soon as a satis-factory function is identified.

在先前的實驗中，我們觀察到超平面數量對建構時間的主要影響。因此，在 § 5 中，我們將每個儲存格的排列建構限制在穿過它的超平面上。在圖 18 中，設定 n = 100 和 d = 4，我們評估了穿過儲存格的超平面數量。圖 18 中的 x 軸是按 |HC[c]|（穿過儲存格 c 的超平面數量）排序的儲存格，y 軸顯示 |HC[c]|。超過 80% 的儲存格有少於 100 個超平面穿過它們，即使在它們內部進行完整的排列建構也是合理的。此外，回想 § 5，MARKCELL 一旦識別出滿意函數，就會停止排列建構。

Figures 19 and 20 show the required time for different steps of preprocessing, as well as the total preprocessing time. Figure 19 shows the cost for varying n, with d = 3 and N = 40,000. We note that in practice, humans tend to define rankings over a limited number of attributes on account of the cognitive burden involved. Even then, coming up with a weight vector for a limited set of attributes is challenging. Still, there are situations with more attributes, and it makes sense to study the performance of our proposals for these cases. Hence, in Figure 20 we vary d from 3 to 11.

圖 19 和 20 顯示了預處理不同步驟所需的時間，以及總預處理時間。圖 19 顯示了在 d = 3 和 N = 40,000 的情況下，改變 n 的成本。我們注意到，在實務上，由於涉及的認知負擔，人類傾向於在有限數量的屬性上定義排名。即便如此，為有限的屬性集提出一個權重向量也是具有挑戰性的。儘管如此，仍存在具有更多屬性的情況，因此研究我們針對這些情況的提案的效能是有意義的。因此，在圖 20 中，我們將 d 從 3 變為 11。

Since the number of attributes in COMPAS and DOT is limited, we executed the experiment in Figure 20 on synthetic data, generated using the Zipfian distribution. We normalized attribute values in the range [0, 1] and added a binary type attribute with values values assigned uniformly at random. Similarly to our default fairness oracle FM1, we consider a ranking to be fair if at most 60% of its top-30% are of type 1.

由於 COMPAS 和 DOT 中的屬性數量有限，我們在圖 20 的實驗中使用了使用 Zipfian 分佈生成的合成數據。我們將屬性值標準化在 [0, 1] 範圍內，並添加了一個二元類型屬性，其值是隨機均勻分配的。與我們的預設公平性神諭 FM1 類似，如果其 top-30% 中最多 60% 屬於類型 1，我們就認為排名是公平的。

The yellow line in both figures shows the required time for identifying the hyperplanes passing through each cell. Applying CELLPLANEX for finding the cells for each hyper-plane helps skip a large portion of the cells. Still its running time increases significantly as n increases. This is because the number of ordering exchanges |H| is in O(n²). Similarly, as the number of attributes increases, the chance that items dominate each other decreases and hence |H| increases. Also, for a fixed N, the hyperplanes intersect with more cells. As a result, the time taken by CELLPLANEx increases by the num-ber of attributes. The red dashed line shows the arrangement construction cost. As expected, in most settings the majority of the time is taken by this step. Still, the optimizations pro-posed in § 4 and 5 result in reasonable performance of this step. First, limiting the construction of the arrangement for each cell c, to the hyperplanes passing through it, reduces the complexity of the arrangement to |HC[c]|d−1. Second, as shown in Figure 15, the arrangement tree data structure helps rule out checking the intersection of the hyperplanes with all regions. Finally, the early stop condition is effective at reducing the running time. The final step is to assign the function of the closest satisfactory cell to each unsatisfactory cell. This step uses a priority queue, and is observed in all the settings in Figures 19 and 20 to be be fast, as expected.

兩圖中的黃線顯示了識別穿過每個儲存格的超平面所需的時間。應用 CELLPLANEX 尋找每個超平面的儲存格有助於跳過大部分儲存格。但隨著 n 的增加，其執行時間仍顯著增加。這是因為排序交換的數量 |H| 為 O(n²)。同樣，隨著屬性數量的增加，項目互相支配的機會減少，因此 |H| 增加。此外，對於固定的 N，超平面與更多儲存格相交。因此，CELLPLANEX 所需的時間隨著屬性數量的增加而增加。紅色虛線顯示了排列建構的成本。正如預期的那樣，在大多數設定中，大部分時間都花在這個步驟上。儘管如此，§ 4 和 5 中提出的優化使得這個步驟的效能合理。首先，將每個儲存格 c 的排列建構限制在穿過它的超平面上，將排列的複雜度降低到 |HC[c]|d−1。其次，如圖 15 所示，排列樹資料結構有助於排除檢查超平面與所有區域的交集。最後，提早停止條件能有效減少執行時間。最後一步是將最接近的滿意儲存格的函數分配給每個不滿意的儲存格。此步驟使用優先佇列，在圖 19 和 20 的所有設定中都觀察到其速度很快，正如預期的那樣。

### 7.5 Performance of sampling

### 7.5 抽樣效能

Item sampling for a large-scale setting. As explained in § 6.1, item sampling can be used for reducing preprocess-ing time for large datasets. In this experiment, we use the DOT dataset, with three scoring attributes, departure delay, arrival delay, and taxi in. The fairness oracle uses FM1 with airline name as the type attribute. A ranking is satis-factory if the percentage of outcomes from each of four ma-jor companies Delta Airlines (DL), American Airlines (AA), Southwest (WN), and United Airlines (UA) in the top 10% is at most 5% higher than their proportion in the dataset. We sample 1K records uniformly at random from the dataset of 1.3M records and use it for preprocessing with N = 40K. Pre-processing took 20 min. Next, we used the complete dataset and checked if the function assigned to the cells using the sample are in fact satisfactory. For all assigned functions the percentage of results from each of four major airlines in the top 10% was at most 5% higher than their proportion in the whole dataset - all of them were satisfactory.

大規模設定的項目抽樣。如 § 6.1 所述，項目抽樣可用於減少大型資料集的預處理時間。在本實驗中，我們使用 DOT 資料集，其中包含三個評分屬性：出發延誤、到達延誤和滑行時間。公平性神諭使用 FM1，並以航空公司名稱作為類型屬性。如果達美航空 (DL)、美國航空 (AA)、西南航空 (WN) 和聯合航空 (UA) 這四家主要公司中，每家公司在前 10% 的結果中所佔的百分比最多比其在資料集中的比例高 5%，則排名是令人滿意的。我們從 130 萬筆記錄的資料集中隨機均勻抽樣 1000 筆記錄，並使用 N = 40K 進行預處理。預處理耗時 20 分鐘。接下來，我們使用完整的資料集，並檢查使用樣本分配給儲存格的函數是否確實令人滿意。對於所有分配的函數，四家主要航空公司中每家在前 10% 的結果中所佔的百分比最多比其在整個資料集中的比例高 5%——所有這些函數都是令人滿意的。

Function sampling for on-the-fly query processing. In § 6.2, we discussed function sampling for our problem. Unfor-tunately, as stated in Theorem 7, function sampling cannot provide a guarantee on finding an approximation solution for the problem. Still, we suggest it for on-the-fly query process-ing in order to explore the e-vicinity of the input function.

即時查詢處理的函數抽樣。在 § 6.2 中，我們討論了我們問題的函數抽樣。不幸的是，如定理 7 所述，函數抽樣無法保證找到問題的近似解。儘管如此，我們建議將其用於即時查詢處理，以探索輸入函數的 ε-鄰域。

We choose the COMPAS dataset and set n = 6,000 and d = 3. First, we generate 100 random queries and use a budget of 500 random functions to explore the (π/20)-vicinity of the input functions. For 27 of those, the algorithm exhausted its budget but could not find a satisfactory function, taking about 11 sec. We plotted the running time of the algorithm for the other cases in Figure 21. Around 60% of the successful queries finished in less than 0.1 sec. Those are the ones that are either satisfactory, or are in mostly satisfactory regions. There are a few cases in which the exploration was successful but it took a few seconds to find a satisfactory function. In summary, in most cases the algorithm either quickly finds a satisfactory function or it never finds one.

我們選擇 COMPAS 資料集，並設定 n = 6,000 和 d = 3。首先，我們產生 100 個隨機查詢，並使用 500 個隨機函數的預算來探索輸入函數的 (π/20)-鄰域。其中 27 個，演算法耗盡了預算但找不到滿意的函數，耗時約 11 秒。我們在圖 21 中繪製了其他情況下演算法的執行時間。大約 60% 的成功查詢在不到 0.1 秒內完成。這些查詢要麼是滿意的，要麼處於大部分滿意的區域。有少數情況下，探索成功了，但花了幾秒鐘才找到滿意的函數。總之，在大多數情況下，演算法要麼很快找到滿意的函數，要麼永遠找不到。

Next, we evaluate the effect of the width of e-vicinity. To do so, we draw 30 random functions, vary θ from π/50 to π/10, and use a budget of 500 function samples for explo-ration for each input function. The results are provided in Figure 22. The left y-axis shows the percentage of queries in which the algorithm could find a satisfactory result. In gen-eral, the wider the exploration space, the higher the chance of finding a satisfactory result. This is reflected in the figure, as the success rate increased from around 50% at 0 = π/50 to around 90% at 0 = π/10. The right y-axis of the plot shows the average running time of the algorithm for each setting. The running time drops from around 5.5 sec at 0 = π/50 to less than 2 sec at θ = π/10. The reason is that for wider vicinities, the algorithm has a higher chance of finding a satisfactory function and stopping early.

接下來，我們評估 e-鄰域寬度的影響。為此，我們抽取 30 個隨機函數，將 θ 從 π/50 變為 π/10，並為每個輸入函數使用 500 個函數樣本的預算進行探索。結果如圖 22 所示。左 y 軸顯示演算法能夠找到滿意結果的查詢百分比。一般來說，探索空間越寬，找到滿意結果的機會就越高。這反映在圖中，成功率從 0 = π/50 時的約 50% 增加到 0 = π/10 時的約 90%。圖的右 y 軸顯示了每種設定下演算法的平均執行時間。執行時間從 0 = π/50 時的約 5.5 秒下降到 θ = π/10 時的不到 2 秒。原因是對於更寬的鄰域，演算法有更高的機會找到滿意函數並提早停止。

## 8 RELATED WORK

## 8 相關工作

There is a robust body of prior work on computing prefer-ences over datasets, divided into two major categories: (1) ranking and top-k [27] query processing for cases where the user has a scoring function in mind, and (2) finding represen-tatives such as skyline [18, 28, 29], and its subsets such as regret minimizing sets [16, 21, 30], in the absence of a scoring function. The primary focus of this work is on efficient query processing, for which methods include threshold-based [31], view-based [32], and indexing-based [33]. Subsequent work considered efficient query processing for computing pref-erences over spatial [34] and noisy [35] databases. None of these consider adjusting the scoring function.

關於計算資料集偏好的先前研究成果豐碩，主要分為兩大類：(1) 當使用者心中有評分函數時的排名和 top-k [27] 查詢處理，以及 (2) 在沒有評分函數的情況下尋找代表，例如 skyline [18, 28, 29] 及其子集，例如後悔最小化集 [16, 21, 30]。這項工作的主要重點是高效的查詢處理，其方法包括基於閾值 [31]、基於視圖 [32] 和基於索引 [33] 的方法。後續工作考慮了在空間 [34] 和雜訊 [35] 資料庫上高效計算偏好的查詢處理。這些工作都沒有考慮調整評分函數。

Our work is among a small handful of studies that focus on fairness in ranking [5, 6, 10]. Several recent papers focus on measuring fairness in ranked lists [5, 6], on constructing ranked lists that meet fairness criteria [10], and on fair and diverse set selection [4]. Fairness in top-k over a single binary type attribute (such as gender, ethnicity, or disability status) is studied in Zehlike et al. [6], where the goal is to ensure that the proportion of members of a protected group in every prefix of the ranking remains statistically above a given minimum. Celis et al. [10] provide a theoretical investigation of ranking with fairness constraints. In their work, fairness in a ranked list is quantified as an upper bound or a lower bound on the number of members of a protected group in every prefix of the ranking. They show that, for a single binary type attribute, a fair ranking can be constructed in polynomial time. For multiple, possibly overlapping, type attributes, they show that the problem is NP-hard, and propose an approximation algorithm. In contrast to these papers, our goal is to assist the user in designing a fair ranking scheme, rather than to produce a fair ranking.

我們的研究是少數幾個關注排名公平性的研究之一 [5, 6, 10]。最近有幾篇論文專注於衡量排名列表中的公平性 [5, 6]，建構符合公平性標準的排名列表 [10]，以及公平和多樣化的集合選擇 [4]。Zehlike 等人 [6] 研究了在單一二元類型屬性（例如性別、種族或殘疾狀況）上的 top-k 公平性，其目標是確保在排名的每個前綴中，受保護群體成員的比例在統計上保持在給定的最小值之上。Celis 等人 [10] 對具有公平性限制的排名進行了理論研究。在他們的工作中，排名列表中的公平性被量化為在排名的每個前綴中，受保護群體成員數量的上限或下限。他們證明，對於單一二元類型屬性，可以在多項式時間內建構一個公平的排名。對於多個可能重疊的類型屬性，他們證明該問題是 NP-hard，並提出了一個近似演算法。與這些論文相反，我們的目標是協助使用者設計一個公平的排名方案，而不是產生一個公平的排名。

Our work is also related to the literature on fairness in machine learning [7, 8]. A common approach is to add fairness constraints to the objective function of a classifier [3, 36]. For example, Zafar et al. [3] propose to add fairness constraints to the logistic regression and support vector ma-chine (SVM) classifiers. They show that, for a single binary type attribute, the problem is convex and can be solved efficiently. For multiple type attributes, they show that the problem is not convex, and propose a heuristic. In contrast to our work, these papers focus on classification, rather than ranking, and assume the existence of labeled training data.

我們的研究也與機器學習中的公平性文獻相關 [7, 8]。一種常見的方法是將公平性限制加入分類器的目標函數中 [3, 36]。例如，Zafar 等人 [3] 建議將公平性限制加入邏輯迴歸和支持向量機 (SVM) 分類器中。他們證明，對於單一二元類型屬性，該問題是凸的，可以有效解決。對於多個類型屬性，他們證明該問題不是凸的，並提出了一種啟發式方法。與我們的研究不同，這些論文專注於分類，而非排名，並假設存在標記的訓練資料。

## 9 CONCLUSION

## 9 結論

We have presented a system to assist a user in designing a fair ranking scheme. Our system is based on a geometric interpretation of the space of linear ranking functions. We have shown how to partition this space into regions that are satisfactory, in that they meet a specified fairness criterion, and those that are not. We have developed efficient algo-rithms to find the closest satisfactory function to a given unsatisfactory one. We have also developed approximation techniques that make our system interactive. Our extensive experiments on real datasets demonstrate that our methods are able to find solutions that satisfy fairness criteria effec-tively (usually with only small changes to proposed weight vectors) and efficiently (in interactive time, after some initial pre-processing).

我們提出了一個系統，以協助使用者設計公平的排名方案。我們的系統基於對線性排名函數空間的幾何解釋。我們展示了如何將這個空間劃分為令人滿意的區域（滿足指定的公平性標準）和不令人滿意的區域。我們開發了高效的演算法，以找到最接近給定不滿意函數的滿意函數。我們還開發了近似技術，使我們的系統具有互動性。我們在真實數據集上進行的廣泛實驗表明，我們的方法能夠有效地（通常只需對提議的權重向量進行微小更改）和高效地（在互動時間內，經過一些初始預處理後）找到滿足公平性標準的解決方案。

## 10 REFERENCES

## 10 參考文獻

[1] C. O'Neil. 2016. Weapons of Math Destruction: How Big Data Increases Inequality and Threatens Democracy. Crown.

[1] C. O'Neil. 2016. 數學毀滅武器：大數據如何加劇不平等並威脅民主。Crown。

[2] B. Friedman and H. Nissenbaum. 1996. Bias in computer systems. ACM Transactions on Information Systems (TOIS) 14, 3 (1996), 330-347.

[2] B. Friedman and H. Nissenbaum. 1996. 電腦系統中的偏見。ACM 資訊系統彙刊 (TOIS) 14, 3 (1996), 330-347。

[3] M. B. Zafar, I. Valera, M. G. Rodriguez, and K. P. Gummadi. 2017. Fairness constraints: Mechanisms for fair classification. In AISTATS.

[3] M. B. Zafar, I. Valera, M. G. Rodriguez, and K. P. Gummadi. 2017. 公平性限制：公平分類的機制。在 AISTATS。

[4] J. Stoyanovich, K. Lum, C. Dwork, and S. Barocas. 2017. On the use of machine learning in public policy: A case study in criminal justice. Tutorial at the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining.

[4] J. Stoyanovich, K. Lum, C. Dwork, and S. Barocas. 2017. 關於機器學習在公共政策中的應用：刑事司法案例研究。第 23 屆 ACM SIGKDD 國際知識發現與數據挖掘會議教程。

[5] S. P. F. Singh and J. Joachims. 2018. Fairness of exposure in rankings. In KDD.

[5] S. P. F. Singh and J. Joachims. 2018. 排名中曝光的公平性。在 KDD。

[6] M. Zehlike, F. Bonchi, C. Castillo, S. Hajian, M. Megahed, and R. Baeza-Yates. 2017. Fa* ir: A fair top-k ranking algorithm. In CIKM.

[6] M. Zehlike, F. Bonchi, C. Castillo, S. Hajian, M. Megahed, and R. Baeza-Yates. 2017. Fa* ir：一種公平的 top-k 排名演算法。在 CIKM。

[7] S. Barocas and A. D. Selbst. 2016. Big data's disparate impact. California Law Review 104 (2016), 671.

[7] S. Barocas and A. D. Selbst. 2016. 大數據的差別影響。加州法律評論 104 (2016), 671。

[8] S. Verma and J. Rubin. 2018. Fairness definitions explained. In Proceedings of the 1st International Workshop on Software Fairness. 1-7.

[8] S. Verma and J. Rubin. 2018. 公平性定義解釋。在第一屆國際軟體公平性研討會論文集。1-7。

[9] J. Kleinberg, S. Mullainathan, and M. Raghavan. 2016. Inherent trade-offs in the fair determination of risk scores. arXiv preprint arXiv:1609.05807 (2016).

[9] J. Kleinberg, S. Mullainathan, and M. Raghavan. 2016. 風險評分公平確定中的內在權衡。arXiv 預印本 arXiv:1609.05807 (2016)。

[10] L. E. Celis, M. P. Devanur, N. R. Devanur, and N. Vishnoi. 2018. Ranking with fairness constraints. In International Conference on Machine Learning. 794-803.

[10] L. E. Celis, M. P. Devanur, N. R. Devanur, and N. Vishnoi. 2018. 具有公平性限制的排名。在國際機器學習會議。794-803。

[11] College Board. 2014. 2014 College-Bound Seniors Total Group Profile Report. https://secure-media.collegeboard.org/digitalServices/pdf/sat/ 2014-total-group-sat-report.pdf.

[11] 大學理事會。2014 年。2014 年大學預科生總體概況報告。https://secure-media.collegeboard.org/digitalServices/pdf/sat/ 2014-total-group-sat-report.pdf。

[12] J. Karabel. 2005. The chosen: The hidden history of admission and exclusion at Harvard, Yale, and Princeton. Houghton Mifflin Harcourt.

[12] J. Karabel. 2005. 被選中的人：哈佛、耶魯和普林斯頓招生與排斥的隱藏歷史。Houghton Mifflin Harcourt。

[13] M. Gladwell. 2005. Getting in. The New Yorker (2005).

[13] M. Gladwell. 2005. 進入。紐約客 (2005)。

[14] New York City Council. 2018. A local law to amend the administrative code of the city of New York, in relation to automated decision systems used by agencies. https://legistar.council.nyc.gov/LegislationDetail.aspx? ID=3137815&GUID=591414A3-C07E-467A-95B8-B0534E35B63A.

[14] 紐約市議會。2018 年。一項修改紐約市行政法典的地方法律，涉及機構使用的自動化決策系統。https://legistar.council.nyc.gov/LegislationDetail.aspx? ID=3137815&GUID=591414A3-C07E-467A-95B8-B0534E35B63A。

[15] A. Asudeh, N. Koudas, G. Das, and A. C. An. 2017. On the complexity of query result diversification. In ICDT.

[15] A. Asudeh, N. Koudas, G. Das, and A. C. An. 2017. 關於查詢結果多樣化的複雜性。在 ICDT。

[16] A. Asudeh, A. C. An, N. Koudas, and G. Das. 2018. Efficient computation of regret-ratio minimizing sets. In SIGMOD.

[16] A. Asudeh, A. C. An, N. Koudas, and G. Das. 2018. 後悔比率最小化集的有效計算。在 SIGMOD。

[17] M. de Berg, O. Cheong, M. van Kreveld, and M. Overmars. 2008. Computational geometry. Springer.

[17] M. de Berg, O. Cheong, M. van Kreveld, and M. Overmars. 2008. 計算幾何。Springer。

[18] S. Börzsönyi, D. Kossmann, and K. Stocker. 2001. The skyline operator. In ICDE.

[18] S. Börzsönyi, D. Kossmann, and K. Stocker. 2001. 天際線運算子。在 ICDE。

[19] A. Asudeh, G. Das, and H. V. Jagadish. 2018. Assessing the quality of a single user's preferences. In SIGMOD.

[19] A. Asudeh, G. Das, and H. V. Jagadish. 2018. 評估單一使用者偏好的品質。在 SIGMOD。

[20] E. W. Dijkstra. 1959. A note on two problems in connexion with graphs. Numerische mathematik 1, 1 (1959), 269-271.

[20] E. W. Dijkstra. 1959. 關於圖論中兩個問題的註記。Numerische mathematik 1, 1 (1959), 269-271。

[21] A. Asudeh, G. Das, and H. V. Jagadish. 2018. On the complexity of finding the region of interest for a user. In PODS.

[21] A. Asudeh, G. Das, and H. V. Jagadish. 2018. 關於尋找使用者感興趣區域的複雜性。在 PODS。

[22] A. Asudeh, G. Das, and H. V. Jagadish. 2018. Querying the preferences of a user. In SIGMOD.

[22] A. Asudeh, G. Das, and H. V. Jagadish. 2018. 查詢使用者的偏好。在 SIGMOD。

[23] G. E. P. Box. 1958. A note on the generation of random normal deviates. The Annals of Mathematical Statistics (1958), 610-611.

[23] G. E. P. Box. 1958. 關於產生隨機常態離差的註記。數學統計年鑑 (1958), 610-611。

[24] M. E. Muller. 1959. A note on a method for generating points uniformly on n-dimensional spheres. Communications of the ACM 2, 4 (1959), 19-20.

[24] M. E. Muller. 1959. 關於在 n 維球面上均勻產生點的方法的註記。ACM 通訊 2, 4 (1959), 19-20。

[25] J. Angwin, J. Larson, S. Mattu, and L. Kirchner. 2016. Machine bias. ProPublica (2016).

[25] J. Angwin, J. Larson, S. Mattu, and L. Kirchner. 2016. 機器偏見。ProPublica (2016)。

[26] US Department of Transportation. 2016. Airline on-time statistics. https://www.transtats.bts.gov/OT_Delay/OT_DelayCause1.asp.

[26] 美國運輸部。2016 年。航空公司準點統計。https://www.transtats.bts.gov/OT_Delay/OT_DelayCause1.asp。

[27] I. F. Ilyas, G. Beskales, and M. A. Soliman. 2008. A survey of top-k query processing techniques in relational database systems. ACM Computing Surveys (CSUR) 40, 4 (2008), 11.

[27] I. F. Ilyas, G. Beskales, and M. A. Soliman. 2008. 關係資料庫系統中 top-k 查詢處理技術綜述。ACM 計算調查 (CSUR) 40, 4 (2008), 11。

[28] J. Chomicki, P. Godfrey, J. Gryz, and D. Liang. 2003. Skyline queries, constrained, and top-k. In CIKM.

[28] J. Chomicki, P. Godfrey, J. Gryz, and D. Liang. 2003. 天際線查詢、約束和 top-k。在 CIKM。

[29] D. Papadias, Y. Tao, G. Fu, and B. Seeger. 2005. Progressive skyline computation in database systems. ACM Transactions on Database Systems (TODS) 30, 1 (2005), 41-82.

[29] D. Papadias, Y. Tao, G. Fu, and B. Seeger. 2005. 資料庫系統中的漸進式天際線計算。ACM 資料庫系統彙刊 (TODS) 30, 1 (2005), 41-82。

[30] T. Chester, A. C. An, G. Das, and N. Koudas. 2014. On the quality and efficiency of query result diversification. In EDBT.

[30] T. Chester, A. C. An, G. Das, and N. Koudas. 2014. 關於查詢結果多樣化的品質和效率。在 EDBT。

[31] R. Fagin, A. Lotem, and M. Naor. 2003. Optimal aggregation algorithms for middleware. Journal of computer and system sciences 66, 4 (2003), 614-656.

[31] R. Fagin, A. Lotem, and M. Naor. 2003. 中介軟體的最佳聚合演算法。電腦與系統科學期刊 66, 4 (2003), 614-656。

[32] H. H. Park, S. J. Lee, and K. S. Kim. 2005. On the performance of view-based top-k query processing. In CIKM.

[32] H. H. Park, S. J. Lee, and K. S. Kim. 2005. 關於基於視圖的 top-k 查詢處理的效能。在 CIKM。

[33] Y. C. Chang, E. L. F. Chang, and W. P. Yang. 2000. A new indexing scheme for multidimensional data. In International Conference on Database Theory. Springer, 339-353.

[33] Y. C. Chang, E. L. F. Chang, and W. P. Yang. 2000. 一種新的多維資料索引方案。在國際資料庫理論會議。Springer, 339-353。

[34] Y. Tao, D. Papadias, and X. Lian. 2006. Reverse k-skyband queries. In ICDE.

[34] Y. Tao, D. Papadias, and X. Lian. 2006. 反向 k-skyband 查詢。在 ICDE。

[35] M. A. Soliman, I. F. Ilyas, and G. Beskales. 2009. Probabilistic top-k and ranking-aggregate queries. ACM Transactions on Database Systems (TODS) 34, 3 (2009), 17.

[35] M. A. Soliman, I. F. Ilyas, and G. Beskales. 2009. 機率性 top-k 和排名聚合查詢。ACM 資料庫系統彙刊 (TODS) 34, 3 (2009), 17。

[36] S. Corbett-Davies, E. Pierson, A. Feller, S. Goel, and A. Huq. 2017. Algorithmic decision making and the cost of fairness. In KDD.

[36] S. Corbett-Davies, E. Pierson, A. Feller, S. Goel, and A. Huq. 2017. 演算法決策與公平性的代價。在 KDD。
