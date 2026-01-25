---
title: AnyBURL
field: Knowledge_Graph
status: Imported
created_date: 2026-01-14
pdf_link: "[[AnyBURL.pdf]]"
tags:
  - paper
  - knowledge_graph
---

Loaded cached credentials.


I will now begin translating the document `AnyBURL.pdf` into Traditional Chinese as requested. I will process the document page by page, following the specified paragraph-by-paragraph bilingual format.

# Reinforced Anytime Bottom Up Rule Learning for Knowledge Graph Completion

# 用於知識圖譜補全的增強隨時下至上規則學習

Christian Meilicke¹, Melisachew Wudage Chekol², Manuel Fink¹, Heiner Stuckenschmidt¹
¹Research Group Data and Web Science, University Mannheim
²Utrecht University
{christian,manuel,heiner}@informatik-uni-mannheim.de, m.w.chekol@uu.nl

Christian Meilicke¹, Melisachew Wudage Chekol², Manuel Fink¹, Heiner Stuckenschmidt¹
¹曼海姆大學數據與網絡科學研究組
²烏特勒支大學
{christian,manuel,heiner}@informatik-uni-mannheim.de, m.w.chekol@uu.nl

**Abstract**

**摘要**

Most of today’s work on knowledge graph completion is concerned with sub-symbolic approaches that focus on the concept of embedding a given graph in a low dimensional vector space. Against this trend, we propose an approach called AnyBURL that is rooted in the symbolic space. Its core algorithm is based on sampling paths, which are generalized into Horn rules. Previously published results show that the prediction quality of AnyBURL is on the same level as current state of the art with the additional benefit of offering an explanation for the predicted fact. In this paper, we are concerned with two extensions of AnyBURL. Firstly, we change AnyBURL’s interpretation of rules from O-subsumption into O-subsumption under Object Identity. Secondly, we introduce reinforcement learning to better guide the sampling process. We found out that reinforcement learning helps finding more valuable rules earlier in the search process. We measure the impact of both extensions and compare the resulting approach with current state of the art approaches. Our results show that AnyBURL outperforms most sub-symbolic methods.

當前大多數關於知識圖譜補全的工作都關注於將給定圖譜嵌入低維向量空間的次符號方法。與此趨勢相反，我們提出了一種名為 AnyBURL 的方法，該方法植根於符號空間。其核心演算法基於路徑採樣，並將其泛化為 Horn 規則。先前發表的結果表明，AnyBURL 的預測質量與當前最先進的技術水平相當，且具有為預測事實提供解釋的額外優點。在本文中，我們關注 AnyBURL 的兩個擴展。首先，我們將 AnyBURL 的規則解釋從 O-subsumption 更改為 O-subsumption under Object Identity。其次，我們引入強化學習以更好地引導採樣過程。我們發現強化學習有助於在搜索過程中更早地發現更有價值的規則。我們衡量了這兩種擴展的影響，並將由此產生的方法與當前最先進的方法進行了比較。我們的結果顯示，AnyBURL 的表現優於大多數次符號方法。

of entities and billions of facts. As pointed out in [8], knowledge graphs are often incomplete. The task to construct missing triples using the vocabulary already used in the graph is known as knowledge graph completion or link prediction. This task can be solved with the additional help of external resources (e.g., text in web-pages) or by inferring new triples solely from the triples in a given knowledge graph. We are concerned with the latter problem.

實體和數十億個事實。如 [8] 中所指出的，知識圖譜通常是不完整的。使用圖譜中已有的詞彙來構建缺失三元組的任務被稱為知識圖譜補全或鏈接預測。此任務可以借助外部資源（例如，網頁中的文本）來解決，也可以僅從給定知識圖譜中的三元組推斷出新的三元組。我們關注的是後一個問題。

An approach that does not use external information must rely on the statistics, patterns, distributions or any other kind of regularity that can be found in the given knowledge graph. An intuitive choice for solving such a task is to learn and apply an explicit, symbolic representation of these patterns. While there is long history of approaches that are concerned with learning symbolic representations, such as inductive logic programming [19] and relational association rule mining [6], today’s research is following a different paradigm. The vast majority of methods that are developed nowadays learn a low dimensional, sub-symbolic representation of a given knowledge graph. Inspired by early models such as RESCAL [20] and TransE [3], a large number of new models have been developed within the last decade. As a result, symbolic approaches are underrepresented in knowledge graph completion research.

不使用外部資訊的方法必須依賴於給定知識圖譜中可以找到的統計、模式、分佈或任何其他類型的規律性。解決此類任務的一個直觀選擇是學習和應用這些模式的明確、符號表示。雖然有著悠久的學習符號表示方法的歷史，例如歸納邏輯編程 [19] 和關聯規則挖掘 [6]，但今日的研究正遵循著不同的範式。現今開發的絕大多數方法學習給定知識圖譜的低維、次符號表示。受到 RESCAL [20] 和 TransE [3] 等早期模型的啟發，過去十年中開發了大量新模型。因此，符號方法在知識圖譜補全研究中代表性不足。

We have developed a symbolic approach [16] with a language bias that mines especially those rules that might be relevant for the task at hand. We called our approach AnyBURL (Anytime Bottom-up Rule Learning) due to its anytime behaviour and fact that it is based on a sampling

我們開發了一種符號方法 [16]，其語言偏見特別挖掘那些可能與手頭任務相關的規則。我們稱我們的方法為 AnyBURL（隨時由下而上規則學習），因為它的隨時行為以及它基於採樣的事實。

component that generalizes paths into rules. Our results as well as the results reported in an independent evaluation of the current state of the art [23] revealed that AnyBURL is not just a symbolic baseline, but performs on the same level as the best models proposed in the last five years. In this paper, we further improve AnyBURL and report about the impact of two extensions.

將路徑泛化為規則的組件。我們的結果以及在對當前最先進技術 [23] 的獨立評估中報告的結果顯示，AnyBURL 不僅僅是一個符號基線，其性能與過去五年提出的最佳模型處於同一水平。在本文中，我們進一步改進了 AnyBURL，並報告了兩個擴展的影響。

With this paper we give for the first time an elaborate description of AnyBURL. Aside from the original algorithm, we describe our extensions and improvements and report about comprehensive experiments. In particular, the paper contains the following contributions.

通過本文，我們首次對 AnyBURL 進行了詳盡的描述。除了原始演算法之外，我們還描述了我們的擴展和改進，並報告了全面的實驗。特別是，本文包含以下貢獻。

*   We take up the concept of Object Identity [25] and report about experiments that illustrate its benefits w.r.t knowledge graph completion. Our results show that it prevents learning a large number of quasi-redundant rules with misleading confidence scores.

*   我們採用了物件恆等性 [25] 的概念，並報告了說明其在知識圖譜補全方面優勢的實驗。我們的結果表明，它能防止學習大量帶有誤導性置信度分數的準冗餘規則。

*   We introduce reinforcement learning to guide the search during sampling paths. We argue that reinforcement learning is more robust, allows to leverage the specifics of a given knowledge graph, and is less affected by choosing a wrong parameter setting.

*   我們引入強化學習來引導搜索採樣路徑。我們認為強化學習更穩健，允許利用給定知識圖譜的特性，並且較少受到選擇錯誤參數設置的影響。

The results of our experiments show that the improved version of AnyBURL is one of the best approaches available for the knowledge graph completion task.

我們的實驗結果表明，AnyBURL 的改進版本是目前知識圖譜補全任務中最好的方法之一。

## 2 Bottom Up Rule Learning

## 2 由下而上的規則學習

We first introduce the type of rules that can be learned by AnyBURL before we describe how we create these rules from sampling paths. Parts of this were already presented in a different form in [16]. Then we explain the concept of Object Identity that was introduced in [25], and argue why it is important for our use case. Object Identity was partially implemented in the previous version of AnyBURL without understanding its importance.

我們首先介紹 AnyBURL 可以學習的規則類型，然後再描述我們如何從採樣路徑中創建這些規則。其中部分內容已在 [16] 中以不同形式呈現。然後我們解釋在 [25] 中引入的物件恆等性概念，並論證其在我們用例中的重要性。物件恆等性在先前版本的 AnyBURL 中有部分實現，但未完全理解其重要性。

### 2.1 Language Bias

### 2.1 語言偏誤

We distinguish in the following between three types of rules that we call binary rules (B), unary rules ending with a dangling atom (Ud) and unary rules ending with an atom that includes a constant (Uc)¹.

我們在下面區分我們稱之為二元規則 (B)、以懸空原子結尾的一元規則 (Ud) 和以包含常數的原子結尾的一元規則 (Uc)¹ 的三種類型。

B: h(A₀, Aₙ) ← ⋀ᵢ₌₁ⁿ bᵢ(Aᵢ₋₁, Aᵢ)
Ud: h(A₀, c) ← ⋀ᵢ₌₁ⁿ⁻¹ bᵢ(Aᵢ₋₁, Aᵢ)
Uc: h(A₀, c) ← ⋀ᵢ₌₁ⁿ⁻¹ bᵢ(Aᵢ₋₁, Aᵢ) ∧ bₙ(Aₙ₋₁, c')

B: h(A₀, Aₙ) ← ⋀ᵢ₌₁ⁿ bᵢ(Aᵢ₋₁, Aᵢ)
Ud: h(A₀, c) ← ⋀ᵢ₌₁ⁿ⁻¹ bᵢ(Aᵢ₋₁, Aᵢ)
Uc: h(A₀, c) ← ⋀ᵢ₌₁ⁿ⁻¹ bᵢ(Aᵢ₋₁, Aᵢ) ∧ bₙ(Aₙ₋₁, c')

In contrast to binary rules, the head atom in unary rules contains a constant and only one instead of two variables. Such an expression can also be understood as a complex way to write down a unary predicate, which is the reason for naming these rules unary rules. Typical examples are head atoms such as gender(X, female) or citizen(X, spain).

與二元規則相反，一元規則中的頭部原子包含一個常數，並且只有一個而非兩個變數。這樣的表達式也可以被理解為寫下一元謂詞的一種複雜方式，這也是將這些規則命名為一元規則的原因。典型的例子是頭部原子，例如 `gender(X, female)` 或 `citizen(X, spain)`。

We refer to rules of these types as path rules, because the body atoms form a path. Note that our language bias also includes rule variations with flipped variables in the atoms: given a knowledge graph G, a path of length n is a sequence of n triples pi (ci, Ci+1) with pi (Ci, Ci+1) ∈ Gor Pi(Ci+1, Ci) ∈ G for 0 < i < n. The abstract rule patterns shown above are said to have a length of n as their body can be instantiated to a path of length n. Instead of Ai we will sometimes use A, B, C, and so on as names for the variables. Moreover, we will usually replace the variables that appear in the head by X for the subject and Y for the object.

我們將這些類型的規則稱為路徑規則，因為主體原子形成了一條路徑。請注意，我們的語言偏見還包括原子中變數翻轉的規則變體：給定一個知識圖譜 G，長度為 n 的路徑是 n 個三元組 pᵢ(cᵢ, cᵢ₊₁) 的序列，其中 pᵢ(cᵢ, cᵢ₊₁) ∈ G 或 pᵢ(cᵢ₊₁, cᵢ) ∈ G，對於 0 < i < n。上面顯示的抽象規則模式被稱為長度為 n，因為它們的主體可以被實例化為長度為 n 的路徑。我們有時會使用 A、B、C 等作為變數的名稱，而不是 Aᵢ。此外，我們通常會將頭部出現的變數替換為主體的 X 和客體的 Y。

B rules and Uc rules are also called closed connected rules. They can be learned by the mining system AMIE described in [11, 10]. Uą rules are not closed because An is a variable that appears only once.

B 規則和 Uc 規則也稱為封閉連接規則。它們可以通過 [11, 10] 中描述的挖掘系統 AMIE 來學習。Ua 規則不是封閉的，因為 Aₙ 是一個只出現一次的變數。

Examples for binary rules are Rules (1) and (2) shown below. They describe the relation between X and Y via an alternative path between X and Y. This path can contain a single relation or a chain of several relations. We allow recursive rules, i.e., the relation in the head can appear one or several times in the body as shown in Rule (2). Rule (3) is a Uc rule which states that a person is female, if she is married to a person that is male. A typical example for a Ua rule is Rule (4), which says that an actor is someone

二元規則的例子如規則 (1) 和 (2) 所示。它們透過 X 和 Y 之間的替代路徑來描述 X 和 Y 之間的關係。此路徑可以包含單一關係或多個關係的鏈。我們允許遞迴規則，即頭部中的關係可以在主體中出現一次或多次，如規則 (2) 所示。規則 (3) 是一個 Uc 規則，它說明如果一個人與一個男性結婚，那麼這個人就是女性。Ua 規則的一個典型例子是規則 (4)，它說明演員是某個

---
¹In [16] we called binary rules cyclic rules and unary rules acyclic rules. This convention was slightly confusing, because a unary rule can also be sampled from a cyclic path.

¹在[16]中，我們稱二元規則為循環規則，一元規則為非循環規則。這個慣例有點令人困惑，因為一元規則也可以從循環路徑中取樣。
---

who acts (in a film).
hypernym(X, Y) ← hyponym(Y, X) (1)
prod(X,Y) ← prod(X, A), sequel(A, Y) (2)
gen(X, female) ← married(X, A), gen(A, male) (3)
prof(X, actor) ← actedin(X, A) (4)

在電影中表演的人。
上位詞(X, Y) ← 下位詞(Y, X) (1)
prod(X,Y) ← prod(X, A), sequel(A, Y) (2)
gen(X, female) ← married(X, A), gen(A, male) (3)
prof(X, actor) ← actedin(X, A) (4)

All considered rules are probabilistic which means they are annotated with confidence scores that represent the probability of predicting a correct fact with this rule. The fraction of body groundings that result in a correct head grounding (as measured on the training data) is called the confidence of a rule. It is important to understand the relation between the three rule types. It is particularly interesting in the context of probabilistic rules. For that purpose, consider the following set of rules (fictitious confidence scores added in square brackets).

所有考慮的規則都是概率性的，這意味著它們都帶有置信度分數的註釋，代表使用此規則預測正確事實的概率。在訓練數據上測量，產生正確頭部基實例的主體基實例的比例被稱為規則的置信度。理解這三種規則類型之間的關係很重要。在概率規則的背景下尤其有趣。為此，請考慮以下規則集（方括號中添加了虛構的置信度分數）。

speaks(X, Y) ← lives(X, A), lang(Y, A) [0.8] (5)
speaks(X, english) ← lives(X,A) [0.62] (6)
speaks(X, french) ← lives(X,france) [0.88] (7)
speaks(X, german) ← lives(X, germany) [0.95] (8)

speaks(X, Y) ← lives(X, A), lang(Y, A) [0.8] (5)
speaks(X, english) ← lives(X,A) [0.62] (6)
speaks(X, french) ← lives(X,france) [0.88] (7)
speaks(X, german) ← lives(X, germany) [0.95] (8)

Let the relation lives(A, B) be used to say that a person A lives in country B, and let lang(A, B) be used to say that a A is (one of) the official languages of B. Thus, B rule (5) states that X speaks a certain language Y, if X lives in a country A where Y is the official language. Ud Rule (6) is a specialization for predicting english speakers and the remaining Uc rules relate a specific language to a specific country. The interesting aspect of this rule set is the fact that Rule (6) can be generated from Rule (5) by removing the second atom in the body and by grounding Y in the head. Likewise, Rules (7) and (8) can be constructed by additionally grounding A. It seems that we do not need these specialized rule variants, if we already have a more general rule. However, this is wrong for two reasons: (i) it might be the case that the given knowledge graph does not contain information about the official languages of France or Germany; and (ii) the confidences of the specific rules (6)-(8) differ from the confidences of the more general rules. The confidence of a general rule is closely related to the (weighted) average over the specific confidences (e.g. by aggregating over all countries and languages). For that reason, it is necessary to generate both types of rules, even though they might carry partially redundant information.

讓關係 lives(A, B) 用於表示一個人 A 住在國家 B，讓 lang(A, B) 用於表示 A 是 B 的官方語言（之一）。因此，B 規則 (5) 陳述如果 X 住在哪個國家 A，而 Y 是該國的官方語言，那麼 X 會說某種語言 Y。Ud 規則 (6) 是預測英語使用者的特化，其餘的 Uc 規則將特定語言與特定國家關聯起來。這個規則集的有趣之處在於，規則 (6) 可以透過從規則 (5) 的主體中移除第二個原子並在頭部中具體化 Y 來生成。同樣地，規則 (7) 和 (8) 可以透過額外具體化 A 來建構。如果我們已經有了一個更通用的規則，似乎就不需要這些特化的規則變體。然而，這在兩個原因下是錯誤的：(i) 給定的知識圖譜可能不包含關於法國或德國官方語言的資訊；以及 (ii) 特定規則 (6)-(8) 的置信度與更通用規則的置信度不同。通用規則的置信度與特定置信度的（加權）平均值密切相關（例如，透過聚合所有國家和語言）。因此，即使它們可能帶有部分冗餘的資訊，也有必要生成這兩種類型的規則。

### 2.2 Sampling Rules

### 2.2 採樣規則

We propose a bottom-up approach for learning rules from bottom rules, i.e. grounded rules from sampled paths in the knowledge graph. It is divided into the following steps:
1. Sample a path from a given knowledge graph.
2. Construct a bottom rule from the sampled path.
3. Build a generalization lattice rooted in the bottom rule.
4. Store all useful rules that appear in the lattice.

我們提出了一種從底部規則學習規則的自下而上方法，即從知識圖譜中抽樣路徑的實體化規則。它分為以下幾個步驟：
1. 從給定的知識圖譜中抽樣一條路徑。
2. 從抽樣的路徑建構一個底部規則。
3. 建立一個以底部規則為根的泛化格。
4. 儲存出現在格中的所有有用規則。

The above sketch of our approach reminds of the algorithm implemented in Aleph [26]. However, Aleph uses the bottom rule to define the boundaries of a top-down search. It begins with the most general rule and uses the atoms that appear in the bottom rule to create a specialization lattice. Similarly, AMIE also does a top-down search, which in contrast to Aleph is complete because it does not limit which atoms to use to specialize a rule. Our approach differs fundamentally from both algorithms because we create a generalization lattice beginning from the bottom rule. We argue in the following that all relevant rules within the generalization lattice instantiate one of the rule types defined in the previous section. Based on this insight, we can directly instantiate these rule types without the need to create the complete lattice.

上述我們方法的簡要描述讓人想起在 Aleph [26] 中實現的演算法。然而，Aleph 使用底部規則來定義自頂向下搜索的邊界。它從最一般的規則開始，並使用出現在底部規則中的原子來創建一個特化格。同樣，AMIE 也進行自頂向下搜索，與 Aleph 相比，它是完整的，因為它不限制使用哪些原子來特化規則。我們的方法與這兩種演算法有根本的不同，因為我們從底部規則開始創建一個泛化格。我們在接下來的內容中論證，泛化格中的所有相關規則都實例化了上一節中定義的規則類型之一。基於這一見解，我們可以繞過創建完整的格，直接實例化這些規則類型。


Figure 1: A knowledge graph G used for sampling paths. We marked the path that corresponds to Rule 9 blue, Rule 10 green, and Rule 11 red.

圖 1：用於採樣路徑的知識圖譜 G。我們將對應規則 9 的路徑標記為藍色，規則 10 標記為綠色，規則 11 標記為紅色。

To find rules for a fixed relation, AnyBURL samples multiple triples of that relation from the training set, and

為了找到一個固定關係的規則，AnyBURL 從訓練集中抽樣該關係的多個三元組，然後

each time creates rules from it. Figure 1 shows a small subset of a knowledge graph G. We use it to demonstrate how rules for the relation speaks would be learned from it. We construct bottom rules of length n, beginning from speaks(ed, d) (Ed speaks Dutch), which will be the head of the rules. To do this, we randomly walk n steps in the graph, starting either from ed or d. Together with the head triple, the result is a path of length n + 1. We have marked three paths in Figure 1 that could be found for n = 2 or n = 1, respectively. The green and blue paths are acyclic, while the red path, including speaks(ed, d), is cyclic. We convert these paths into the bottom rules (9), (10), and (11).

每次都從中創建規則。圖 1 顯示了一個知識圖譜 G 的小子集。我們用它來演示如何從中學習關係 `speaks` 的規則。我們從 `speaks(ed, d)` (Ed 說荷蘭語) 開始，構造長度為 n 的底層規則，這將是規則的頭部。為此，我們從 `ed` 或 `d` 開始，在圖中隨機走 n 步。加上頭部三元組，結果是一條長度為 n + 1 的路徑。我們在圖 1 中標記了三條路徑，分別對應 n = 2 或 n = 1 的情況。綠色和藍色路徑是非循環的，而紅色路徑，包括 `speaks(ed, d)`，是循環的。我們將這些路徑轉換為底層規則 (9)、(10) 和 (11)。

speaks(ed, d) ← born(ed, a) (9)
speaks(ed, d) ← mar(ed, lisa), born(lisa, a) (10)
speaks(ed, d) ← lives(ed, nl), lang(nl, d) (11)

speaks(ed, d) ← born(ed, a) (9)
speaks(ed, d) ← mar(ed, lisa), born(lisa, a) (10)
speaks(ed, d) ← lives(ed, nl), lang(nl, d) (11)

We argue that any generalization of a path of length n + 1 will be a B, Uc or Ud rule of length n or a shorter rule, which can be constructed from a shorter path, or a rule that is not useful for making a prediction. We elaborate this point by analysing the generalization lattice rooted in Rule (10), depicted in Figure 2.

我們主張，任何長度為 n + 1 的路徑的泛化將會是：一個長度為 n 的 B、Uc 或 Ud 規則；一個可以從較短路徑建構的較短規則；或者一個對於預測無用的規則。我們通過分析植根於規則 (10) 的泛化格（如圖 2 所示）來詳細闡述這一點。

Each edge in the lattice transition stems from one of the following two generalization operations. (i) Replace all occurrences of a constant by a fresh variable. (ii) Drop one of the atoms in the body. Note that we have only depicted those rules in the lattice that have at least one variable in the head. If this would not be the case, the rule would only predict a triple that is already stated in the knowledge graph, which is useless for completion. A rule that appears in the lattice falls into one of the following categories. We have associated the symbols †, *, and ◇ to each category and used them to mark the nodes in Figure 2.

格中的每個邊轉換都源於以下兩個泛化操作之一：(i) 將常數的所有出現替換為一個新的變數。(ii) 刪除主體中的一個原子。請注意，我們只描繪了格中頭部至少有一個變數的那些規則。如果不是這種情況，該規則只會預測知識圖譜中已經陳述的三元組，這對於補全來說是無用的。出現在格中的規則屬於以下類別之一。我們將符號 †、* 和 ◇ 與每個類別相關聯，並用它們來標記圖 2 中的節點。

Ambiguous prediction† The rule has an unconnected variable in the head, which does not appear in the body of the rule. Such a rule makes a prediction that something exists, however, it does not make a concrete prediction which would be required to create a ranking of candidates.

歧義預測† 該規則在頭部有一個未連接的變數，該變數未出現在規則的主體中。這樣的規則預測某物存在，然而，它沒有做出具體的預測，而這對於創建候選排名是必需的。

Shorter bottom rule* The rule might be useful but it would also appear in the lattice of a bottom rule which originates from a shorter path. To avoid duplicate rules, we do not create it from the longer bottom rule. This point is detailed in Section 3.

較短的底部規則* 此規則可能有用，但它也會出現在源自較短路徑的底部規則的格中。為避免重複規則，我們不從較長的底部規則創建它。這一點在第 3 節中有詳細說明。

Useless atom The body contains an atom without variables or an atom with a constant and an unbound variable. Such atoms will always be true in the knowledge graph from which they were sampled and therefore do not affect the truth value of the body.

無用的原子：規則體包含一個沒有變數的原子，或者一個帶有常數和未綁定變數的原子。這樣的原子在從中採樣的知識圖譜中總是為真，因此不影響規則體的真值。

Note that a rule in the lattice marked with a † or * does not need to be generalized any further, because any resulting rule will be marked again with the same symbol.

請注意，標記為 † 或 * 的格中的規則不需要任何進一步的泛化，因為任何由此產生的規則都將再次被標記為相同的符號。

When we apply this annotation scheme to the lattice (Figure 2) that originates from the green acyclic path (in Figure 1), only two rules remain unmarked. We have highlighted these rules with a bold rectangle. These two rules are of type Ud and Uc. One can easily argue that this will always be the result when we generalize a bottom rule that originates from an acyclic path. Thus, we do not need to search over the generalization lattice but can directly create these two rules from a given acyclic path.

當我們將此註釋方案應用於源自綠色非循環路徑（圖1）的格（圖2）時，只有兩條規則未被標記。我們用粗體矩形突出了這兩條規則。這兩條規則屬於 Ud 和 Uc 類型。可以輕易地論證，當我們泛化源自非循環路徑的底部規則時，這將永遠是結果。因此，我們不需要在泛化格上搜索，而是可以直接從給定的非循環路徑創建這兩條規則。

We can observe a similar pattern when we generalize a cyclic path. It results in three rules that we can leverage for a prediction; one B rule and two Uc rules, where the head constant (subject/object) appears again in the last body atom.

當我們將循環路徑一般化時，可以觀察到類似的模式。它產生了三條我們可以利用於預測的規則；一條 B 規則和兩條 Uc 規則，其中頭部常數（主語/賓語）再次出現在最後一個身體原子中。

### 2.3 Object Identity

### 2.3 物件恆等性

Object Identity (OI) refers to an entailment framework that interprets every rule under the additional assumption that two different terms (variables or constants) that appear in a rule must refer to different entities. This means that each rule is extended by a pairwise complete set of inequality constraints. OI was first introduced in [25] and later it is used to propose refinement operators for the original framework [9]. In this work we do not focus on its theoretic properties but on its impact on correcting the confidence scores of the learned rules.

物件恆等性 (OI) 指的是一個蘊涵框架，它在附加假設下解釋每條規則，即規則中出現的兩個不同術語（變數或常數）必須指代不同的實體。這意味著每條規則都擴展了一對完整的非等式約束。OI 最初在 [25] 中引入，後來用於為原始框架 [9] 提出精化算子。在這項工作中，我們不關注其理論屬性，而是關注其對修正學習規則置信度分數的影響。

In the context of our approach, the most important property of OI is its capability to suppress redundant rules that negatively affect performance under the - subsumption [22] entailment regime. We illustrate the effect with the following two rules (h and b are two arbitrary but fixed relations).

在我們的方法的背景下，OI 最重要的特性是它能夠抑制在 -subsumption [22] 蘊涵機制下對性能產生負面影響的冗餘規則。我們用以下兩個規則（h 和 b 是兩個任意但固定的關係）來說明這個效果。


Figure 2: Generalization lattice of the acyclic path (s(ed, d), m(ed, lisa), born(lisa, a)). For legibility we use the abbreviations s = speaks, m = married and b = born.

圖 2：非循環路徑的泛化格 (s(ed, d), m(ed, lisa), born(lisa, a))。為便於閱讀，我們使用縮寫 s = speaks, m = married and b = born。

h(X, Y) ← h(X, Y) (12)
h(X, Y) ← b(X, A), b(B, A), h(B,Y) (13)

h(X, Y) ← h(X, Y) (12)
h(X, Y) ← b(X, A), b(B, A), h(B,Y) (13)

Interpreting rules under OI can be done by adding additional constraints to the rules. For instance, the body of Rule (13) would need to be extended with the inequality constraints (14).
X ≠ A, X ≠ B, X ≠ Y, A ≠ B, A ≠ Y, B ≠ Y (14)

在 OI 下解釋規則可以通過向規則添加額外的約束來完成。例如，規則 (13) 的主體需要用不等式約束 (14) 進行擴展。
X ≠ A, X ≠ B, X ≠ Y, A ≠ B, A ≠ Y, B ≠ Y (14)

Each rule constructed by AnyBURL is always interpreted under OI. Note that these inequality constraints are not shown whenever a rule is displayed or stored in a file.

AnyBURL 建構的每條規則總是在 OI 下進行解釋。請注意，每當規則被顯示或儲存在檔案中時，這些不等式約束是不會顯示的。

Rule (12) is obviously a tautology that will never generate any new facts. This is only partially true for Rule (13). The groundings of its body can be divided into the groundings with B = X, and the groundings 0' with B≠ X. In contrast to a l' grounding, a @ grounding does not predict new facts and is also more likely to result in a true body because both atoms of relation b can be ground to the same fact. This means that, without OI, the confidence score of Rule (13) overestimates its quality as it will always be used to predict unknown facts. Adding the inequality constraints will suppress the @ groundings and result in a more realistic confidence score for the task.

規則 (12) 顯然是一個永不產生任何新事實的套套邏輯。這對規則 (13) 僅部分為真。其主體的基實例可以分為 B = X 的基實例和 B ≠ X 的基實例 θ'。與 l' 基實例相反，@ 基實例不預測新事實，並且也更有可能導致一個真實的主體，因為關係 b 的兩個原子都可以基於相同的事實。這意味著，如果沒有 OI，規則 (13) 的置信度分數會高估其品質，因為它將永遠被用來預測未知的事實。添加不等式約束將抑制 @ 基實例，並產生更現實的任務置信度分數。

It is important to understand that it is not just variations of tautology rules that have this problem. For example, if there are strong rules such as m(X, Y) ← spo(Y, X) (m = married, spo = spouse) in a knowledge graph, rules like the following are also affected.
m(X, Y) ← son(X, A), son(B, A), spo(B,Y) (15)

重要的是要理解，不只是重言式規則的變體有這個問題。例如，如果知識圖譜中有像 m(X, Y) ← spo(Y, X) (m = married, spo = spouse) 這樣的強規則，那麼像下面這樣的規則也會受到影響。
m(X, Y) ← son(X, A), son(B, A), spo(B,Y) (15)

The confidence score of such a rule drastically (and rightfully) decreases under OI once we ignore groundings in which X and B are ground to the same son.

一旦我們忽略 X 和 B 被賦予相同兒子的基實例，這類規則的置信度分數在 OI 下就會急劇（且理所當然地）下降。

While OI helps us to avoid a blow-up of the rule base, a given rule is harder to evaluate under OI (see also §5.1.1 in [5]). This holds both for the confidence computation as well as for the application of the rule in the context of predicting new knowledge. If we ignore the inequality constraints, all possible (X, Y) groundings for Rule (13) can be computed with two join operations. As a result of the first join, we get the groundings for (X, B) which can be used to compute the (X, Y) groundings via a second join. However, the constraint A ≠ Y requires to know the variable bindings of A that we used for the first join when doing the second join to ensure that the constraint is not violated. Keeping track of all variable bindings makes it more complex to compute body groundings under OI.

雖然 OI 幫助我們避免規則庫的爆炸性增長，但在 OI 下評估一個給定的規則更加困難（另請參見 [5] 中的 §5.1.1）。這對於置信度計算以及在預測新知識的背景下應用規則都是如此。如果我們忽略不等式約束，所有可能的 (X, Y) 規則 (13) 的基實例都可以通過兩個連接操作來計算。第一次連接的結果，我們得到 (X, B) 的基實例，可以用來通過第二次連接計算 (X, Y) 的基實例。然而，約束 A ≠ Y 要求知道我們在進行第二次連接時用於第一次連接的 A 的變量綁定，以確保約束不被違反。跟踪所有變量綁定使得在 OI 下計算主體基實例更加複雜。

### 3 Search Strategy

### 3 搜索策略

### 3.1 Path Sampling

### 3.1 路徑採樣

In the paths that we sample for building bottom rules, each triple on a path is called a step. The steps can be made in the direction of a stated triple or in reverse direction. A step in reversed direction causes flipped terms in the corresponding atom of the resulting rule. We call a path a straight path if it does not visit the same entity twice, i.e., ci ≠ cj for each i ≠ j. An exception can be the equality of the first and last entity co = Cn. In this case, we call the path a straight cyclic path, which ends where it began.

在我們為建立底層規則而採樣的路徑中，路徑上的每個三元組被稱為一步。這些步驟可以按照陳述的三元組方向或反向進行。反向的步驟會導致結果規則中相應原子的術語翻轉。如果一條路徑不會兩次訪問同一個實體，即對每個 i ≠ j 都有 ci ≠ cj，我們稱之為直線路徑。一個例外是第一個和最後一個實體相等，即 co = cn。在這種情況下，我們稱這條路徑為直線循環路徑，它在起點處結束。

A straight cyclic path results into a binary B rule and a special form of a Uc rule where the constant in the head and body of the rule is the same. A straight acyclic path results into a Uc rule (with different constants in head and body) and a Ua rule. Our method to sample a path is to choose a random entity as a starting point of a random walk. If the walk arrives at an entity that has been visited before (prior to the last step), the procedure can be restarted until a straight path has been found. This approach can yield cyclic and acyclic paths. It can be expected that the majority of sampled paths will be acyclic. Especially for longer paths it will not often be the case that Co = Cn. This means that a pure random walk strategy will generate only few binary rules. This can be a problem for the resulting rule sets. According to the results presented in [18, 16] we know that a large fraction of correct predictions can be made with B rules.

一條直的循環路徑會產生一條二元 B 規則和一種特殊形式的 Uc 規則，其中規則的頭部和主體中的常數是相同的。一條直的非循環路徑會產生一條 Uc 規則（頭部和主體中的常數不同）和一條 Ua 規則。我們採樣路徑的方法是選擇一個隨機實體作為隨機漫步的起點。如果漫步到達一個之前訪問過的實體（在上一步之前），則可以重新啟動該過程，直到找到一條直的路徑。這種方法可以產生循環和非循環路徑。可以預期，大多數採樣的路徑將是非循環的。特別是對於較長的路徑，Co = Cn 的情況不會經常發生。這意味著純隨機漫步策略只會生成很少的二元規則。這對生成的規則集來說可能是一個問題。根據 [18, 16] 中提出的結果，我們知道很大一部分正確的預測可以用 B 規則做出。

Thus, it makes sense to design a specific strategy to search for cyclic paths. We have slightly modified the random walk strategy by explicitly looking for a fact that connects Cn-1 and Co = Cn in the last step. With an appropriate index it is possible to check the existence of a relation p with p(ci, cj) in constant time for any pair of constants. If we find such a triple, we use this as a final step in the constructed path. If we find several such triples, we pick randomly one of them. With this modification, we are able to find more cyclic paths in the same time span compared to the standard random walk. We are aware that there are more sophisticated methods for finding a path of length n, see for example [21].

因此，設計一種專門搜索循環路徑的策略是有意義的。我們稍微修改了隨機漫步策略，在上一步中明確尋找連接 cₙ₋₁ 和 c₀ = cₙ 的事實。通過適當的索引，可以在常數時間內檢查任何常數對是否存在關係 p(cᵢ, cⱼ)。如果我們找到這樣的三元組，我們就將其用作建構路徑的最後一步。如果我們找到多個這樣的三元組，我們隨機選擇其中一個。通過這種修改，我們能夠在相同的時間範圍內找到比標準隨機漫步更多的循環路徑。我們知道有更複雜的方法可以找到長度為 n 的路徑，例如參見 [21]。

### 3.2 Saturation based Search

### 3.2 基於飽和度的搜索

A detailed description of the search policy that was implemented in a previous version of AnyBURL can be found in [16]. According to that policy, called saturation-based search, the learning process is conducted in a sequence of time spans of fixed length (e.g. one second). Within a time span the algorithm learns as many rules as possible using paths sampled from a specific path profile. A path profile describes path length and whether the path is cyclic or acylic. When a time span is over, the rules found within this span are evaluated. Let R refer to the rules that have been learned in the previous time spans, let Rs refer to the rules found in the current time span, and let R' = R ∪ Rs refer to the rules found in the current time span that have also been found in one of the previous iterations. If |R'|/|Rs| is above a saturation boundary, which needs to be defined as a parameter, the path length of the profile is increased by one. Initially, the algorithm starts with paths of length 2 resulting in rules of length 1. The higher the path length, the more time spans are usually required to reach the saturation boundary. The difference between cyclic and acyclic paths is taken into account by flipping every time span between the cyclic and acyclic profiles. The rule counts generated by cyclic and acyclic rules are independent.

先前版本的 AnyBURL 中實現的搜索策略的詳細描述可以在 [16] 中找到。根據該策略，稱為基於飽和度的搜索，學習過程在一系列固定長度的時間跨度（例如一秒）中進行。在一個時間跨度內，演算法使用從特定路徑輪廓中採樣的路徑盡可能多地學習規則。路徑輪廓描述路徑長度以及路徑是循環還是非循環。當一個時間跨度結束時，將評估在此跨度內找到的規則。令 R 表示在先前時間跨度中學到的規則，令 Rs 表示在當前時間跨度中找到的規則，令 R' = R ∪ Rs 表示在當前時間跨度中找到的也曾在先前迭代中找到過的規則。如果 |R'|/|Rs| 高於需要定義為參數的飽和邊界，則輪廓的路徑長度增加一。最初，演算法從長度為 2 的路徑開始，產生長度為 1 的規則。路徑長度越高，通常需要更多的時間跨度來達到飽和邊界。循環和非循環路徑之間的差異通過在循環和非循環輪廓之間每個時間跨度翻轉來考慮。由循環和非循環規則生成的規則計數是獨立的。

It is an advantage of the saturation-based approach that it does not require to mine all cyclic (or acyclic) rules of length n before the algorithm looks at cyclic (or acyclic) rules of length n + 1. Instead of that, the rule length is increased if a sufficient saturation has been reached. However, it is unclear how to set the required saturation degree (the default value is 0.99). If this value is set too high, the algorithm spends a lot of time sampling paths resulting into rules that have already been generated previously. If the value is set too low, important rules might be missed and cannot be found any more. Another disadvantage of the algorithm is that the ad hoc setting to spend exactly half of the time to search for cyclic paths and half for acyclic paths. To overcome these shortcomings we propose a reinforced approach presented in the following section.

基於飽和度的方法的一個優點是，它不需要在演算法查看長度為 n+1 的循環（或非循環）規則之前，挖掘所有長度為 n 的循環（或非循環）規則。取而代之的是，如果達到了足夠的飽和度，規則長度就會增加。然而，如何設定所需的飽和度（預設值為 0.99）尚不清楚。如果此值設定得太高，演算法會花費大量時間對導致先前已生成規則的路徑進行採樣。如果此值設定得太低，可能會錯過重要的規則，再也找不到了。該演算法的另一個缺點是，它臨時設定花費一半的時間搜索循環路徑，一半的時間搜索非循環路徑。為了克服這些缺點，我們在下一節中提出了一種增強方法。

### 3.3 Reinforced Search

### 3.3 強化搜索

#### 3.3.1 Reward

#### 3.3.1 獎勵

In the following we consider the path sampling problem as a special kind of multi-armed bandit problem [13]. In each time span we have to decide how much effort to spend on which path profile. A path profile in our scenario corresponds to an arm of a bandit in the classical reinforcement learning setting. Each arm (or slot machine) in the bandit problem gives a reward when pulling that

在下文中，我們將路徑採樣問題視為一種特殊的多臂老虎機問題 [13]。在每個時間跨度中，我們必須決定在哪個路徑輪廓上花費多少精力。在我們的情景中，一個路徑輪廓對應於經典強化學習設置中老虎機的一個臂。在老虎機問題中，每次拉動該臂都會獲得獎勵。

arm. What corresponds in our scenario to the reward of pulling an arm, i.e., the reward of creating rules from the paths that belong to a certain profile?

臂。在我們的情景中，拉動一個臂的獎勵，即從屬於某個輪廓的路徑創建規則的獎勵，對應著什麼？

In the following we develop three different reward strategies. They are based on the notion of measuring the reward paid out by a profile in terms of the explanatory quality of the rules that were created by that profile. The explanatory quality of a rule set can be measured in terms of the number of triples of a given knowledge graph that can be reconstructed with the help of the rules from the set. Thus, summing up the support of the rules seems to be a well suited metric. We refer to this as reward strategy Rs.

接下來，我們發展了三種不同的獎勵策略。它們基於衡量一個輪廓所支付獎勵的概念，以該輪廓所創建規則的解釋能力來衡量。一個規則集的解釋能力可以用藉助該集合中的規則可以重構的給定知識圖譜的三元組數量來衡量。因此，將規則的支持度加總似乎是一個很合適的指標。我們將此稱為獎勵策略 Rs。

R(S) = ∑ support(r) (16)
reS
where S is a set of rules and support(r) is the support of a rule r. Given a ruler = r ← ř, we denote by rex the (partially) grounded rule where all occurrences of X are replaced by some constant. Consequently, support and confidence of r can be defined as follows:

其中 S 是一組規則，support(r) 是規則 r 的支持度。給定一個規則 r = r ← ř，我們用 r_ex 表示部分實例化的規則，其中所有 X 的出現都被某個常數替換。因此，r 的支持度和置信度可以定義如下：

support(r) = |{θxy | ∃θz řθxyz ∧ řθxy}|

support(r) = |{θxy | ∃θz řθxyz ∧ řθxy}|

conf(r) = |{θxy | ∃θzřθχγζ ∧ řθχγ}| / |{θxy | ∃θzřθxyz}|
where θxy refers to a grounding for variables X and Y, which appear in the head of r. θz is a grounding for the variables that appear in the body of r that are different from X and Y. θxyz refers to the union of θxy and θz.

其中 θxy 指的是規則 r 頭部出現的變數 X 和 Y 的一個基實例。θz 是規則 r 主體中不同於 X 和 Y 的變數的一個基實例。θxyz 指的是 θxy 和 θz 的聯集。

We are especially interested in rules that make many correct predictions with high confidence. Since, predictions with high confidence are more likely to appear as a top ranked candidate. For that reason we propose a second reward strategy Rsxc that multiplies the number of correct predictions by their confidence:
Rsxc(S) = ∑ support(r) × conf(r) (17)
reS

我們對那些能夠做出許多具有高置信度的正確預測的規則特別感興趣。因為，具有高置信度的預測更有可能作為排名靠前的候選者出現。因此，我們提出了第二個獎勵策略 Rsxc，它將正確預測的數量乘以其置信度：
Rsxc(S) = ∑ support(r) × conf(r) (17)
reS

where S is a set of rules, support(r) is the support and conf(r) is the approximate confidence of a rule r.
We define a third reward strategy that takes rule length into account as follows
Rsxc/2l(S) = ∑ (support(r) × conf(r)) / 2^l(r) (18)
reS
where l(r) denotes the length of a rule r. This reward strategy is a variant of Rsxc that favours shorter over longer rules. It enforces a constraint that assigns at the beginning of the search more computational effort to short rules. Thus, the search constraint has some similarities with a softened saturation-based search as long as we are only concerned with rule length.

其中 S 是一組規則，support(r) 是規則 r 的支持度，conf(r) 是規則 r 的近似置信度。
我們定義了第三種獎勵策略，該策略將規則長度納入考慮，如下所示：
Rsxc/2l(S) = ∑ (support(r) × conf(r)) / 2^l(r) (18)
reS
其中 l(r) 表示規則 r 的長度。這個獎勵策略是 Rsxc 的一個變體，它偏愛較短的規則而非較長的規則。它強制執行一個約束，在搜索開始時將更多的計算精力分配給短規則。因此，只要我們只關心規則長度，搜索約束就與基於飽和度的軟化搜索有一些相似之處。

All metrics are based on the capability of a rule set to reconstruct parts of a given knowledge graph in terms of the training set. An alternative approach would have been to compute the same or similar scores with respect to the prediction of the validation set. If we focus on the training set, we can directly reuse the scores that we already computed. Additional computational effort is not required.

所有指標都基於一個規則集在訓練集方面重構給定知識圖譜部分的能力。另一種方法是計算關於驗證集預測的相同或相似分數。如果我們專注於訓練集，我們可以重複使用我們已經計算的分數。不需要額外的計算工作。

#### 3.3.2 Policy

#### 3.3.2 策略

All three reward strategies can be combined with each of the following two policies. The first policy is a well known policy referred to as e-greedy policy [28]. The parameter e is usually set to relatively small positive value, for example € = 0.1. Every time a decision needs to be made, that decision is a random decision with a probability < € and a greedy decision with a probability > €. When we talk about decisions, we mean the allocation of CPU cores to path profiles. In the e-greedy policy, a small number of decisions is randomized to reserve a small fraction of the available resources for exploration compared to an approach that would focus completely on exploitation.

所有三種獎勵策略都可以與以下兩種策略中的每一種結合。第一種策略是眾所周知的 ε-greedy 策略 [28]。參數 ε 通常設置為相對較小的正值，例如 ε = 0.1。每當需要做出決策時，該決策以 < ε 的機率是隨機決策，以 > ε 的機率是貪婪決策。當我們談論決策時，我們指的是將 CPU 核心分配給路徑配置文件。在 ε-greedy 策略中，少數決策是隨機的，以保留一小部分可用資源用於探索，而不是完全專注於利用。

In our context, a greedy decision assigns all cores, that have not been assigned randomly, to the path profile that generated the rule set that yielded the highest reward the last time it has been selected. Formally, for e-greedy policy, a path profile pf*, with 1 – e probability, is chosen for time span te according to the following equations:
pf* = argmax Q(pf, last(pf, tk)), pf ∈ F
Q(pf, ti) = (1 / Nt(pf)) * R(S(pf, ti) \ ∪(j=1 to i-1) S(pf, tj))
where last(pf, tk) refers to the last time span ti prior to tk (i.e., with i < k) where path profile pf has been used, Q(pf, ti) is the value of the path profile pf for time span ti, Nt(pf) quantifies the computational resources that have been allocated to pf during ti, R denotes a reward

在我們的背景下，一個貪婪的決策會將所有未被隨機分配的核心，分配給上次被選中時產生最高獎勵規則集的路徑輪廓。形式上，對於 ε-greedy 策略，一個路徑輪廓 pf*，以 1 - ε 的機率，根據以下方程式為時間跨度 te 選擇：
pf* = argmax Q(pf, last(pf, tk)), pf ∈ F
Q(pf, ti) = (1 / Nt(pf)) * R(S(pf, ti) \ ∪(j=1 to i-1) S(pf, tj))
其中 last(pf, tk) 指的是在 tk 之前使用路徑輪廓 pf 的最後一個時間跨度 ti（即 i < k），Q(pf, ti) 是路徑輪廓 pf 在時間跨度 ti 的值，Nt(pf) 量化了在 ti 期間分配給 pf 的計算資源，R 表示一個獎勵。

strategy Rs, Rsxc or Rsxc/21, and S(pf, t₁) refers to the set of rules that have been mined by the use of path profile pf during t₁. The expression S(pf, t₁) \ U(j=1 to i-1) S(pf, tj) refers to the set of new rules that have been mined in ti but not in one of the previous time spans. We quantify Nt. (pf) in terms of the number of cores that are assigned to pf during ti. This means that the reward is normalized with the number of allocated cores.

策略 Rs、Rsxc 或 Rsxc/2l，以及 S(pf, ti) 指的是在 ti 期間使用路徑輪廓 pf 所挖掘的規則集。表達式 S(pf, ti) \ U(j=1 to i-1) S(pf, tj) 指的是在 ti 中挖掘但不在先前任一時間跨度中挖掘的新規則集。我們用在 ti 期間分配給 pf 的核心數量來量化 Nt(pf)。這意味著獎勵被分配的核心數量歸一化了。

Note that our scenario differs from the classical multi-armed bandit setting in the sense that the expected reward of a certain profile will decrease any time we use this profile for generating rules. The more often we use that profile, the more probably it is to draw a path that results into a previously learned rule, which was created from the same or from a different path. For that reason we do not base our decision on the average over all previous time spans, but look at the last time span that this profile has been used. The reward of a profile is shrinking continuously, with random ups and downs that are caused by drawing only a limited number of samples. This results into flips between different profiles that are not caused by knowing more (exploration) but by the impact of exhausting profiles over time.

請注意，我們的情境不同於經典的多臂老虎機設定，因為每當我們使用某個輪廓生成規則時，該輪廓的預期獎勵都會減少。我們越常使用該輪廓，就越有可能抽到一條導致先前學習規則的路徑，該規則是從相同或不同路徑創建的。因此，我們的決策不是基於所有先前時間跨度的平均值，而是看該輪廓上次使用的時間跨度。輪廓的獎勵會持續縮小，伴隨著因僅抽取有限樣本而引起的隨機起伏。這導致不同輪廓之間因時間推移而耗盡輪廓的影響，而非因了解更多（探索）而翻轉。

The e-greedy policy might not be a good choice if one profile pf creates higher rewards than another profile pf', however, pf' would also generate relatively good rules. Suppose further that both profiles are relatively stable, i.e., their reward decreases only slightly when they are used for generating rules. In such a setting, we might prefer to draw rules not only from pf but also from pf'. For that reason we propose a second policy where we distribute the available computational resources to all profiles proportional to the reward that has been observed the last time they have been used. We refer to this policy as weighted policy. For each CPU core with a probability < €, we take a random decision; and with a probability > € we proceed as follows. For each profile pf ∈ F we compute the probability of resource allocation Pk (pf) at time span tk, given by the following formula:

如果一個輪廓 pf 創建的獎勵高於另一個輪廓 pf'，但 pf' 也會生成相對不錯的規則，那麼 ε-greedy 策略可能不是一個好的選擇。進一步假設兩個輪廓都相對穩定，即，它們的獎勵在使用它們生成規則時僅略有下降。在這種情況下，我們可能不僅希望從 pf 中提取規則，也希望從 pf' 中提取規則。因此，我們提出了第二個策略，即將可用的計算資源按比例分配給所有輪廓，該比例與上次使用它們時觀察到的獎勵成正比。我們將此策略稱為加權策略。對於每個 CPU 核心，我們以 < ε 的機率做出隨機決策；並以 > ε 的機率按如下方式進行。對於每個輪廓 pf ∈ F，我們計算在時間跨度 tk 時資源分配的機率 Pk(pf)，由以下公式給出：

Pk(pf) = Q(pf, last(pf, tk)) / Σ(pf'∈F) Q(pf', last(pf', tk))
where Q and last are introduced above. For each core that is not yet assigned to a profile due to the random assignment in the <e case, we throw a dice and assign one of the path profile pf ∈ F with probability Pk (pf).

其中 Q 和 last 如上所述。對於每個尚未因 <e 情況下的隨機分配而分配給輪廓的核心，我們擲骰子並以機率 Pk(pf) 分配路徑輪廓 pf ∈ F 中的一個。

To better understand the impact of combining different reward strategies and policies, AnyBURL can be run with a completely random policy where each profile has always the same probability. This can be achieved by setting € = 1. This setting is not necessarily bad. If there are K different profiles, in the worst scenario one of these profiles would generate many useful rules and none of the other profiles would generate such rules. An algorithm that makes perfect decisions would arrive K times faster at the same result as the random policy. However, at the same time we can assume – and the results published in [16] support this assumption – that the most beneficial rules are often mined first. This means that running the random policy and the weighted policy for the same time span will not yield results that are K times worse. The random policy might even outperform the previous, saturation-based implementation of AnyBURL. This will be the case if the saturation threshold is chosen too low or too high.

為了更好地理解結合不同獎勵策略和政策的影響，AnyBURL 可以採用完全隨機的政策運行，其中每個配置檔案始終具有相同的機率。這可以通過設置 € = 1 來實現。這個設置不一定不好。如果有 K 個不同的配置檔案，在最壞的情況下，其中一個配置檔案會生成許多有用的規則，而其他配置檔案則不會生成這樣的規則。一個做出完美決策的演算法會比隨機政策快 K 倍達到相同的結果。然而，我們同時可以假設——並且 [16] 中發表的結果支持這個假設——最有益的規則通常是首先被挖掘出來的。這意味著在相同的時間跨度內運行隨機政策和加權政策不會產生 K 倍差的結果。隨機政策甚至可能勝過之前基於飽和度的 AnyBURL 實現。如果飽和閾值選擇得太低或太高，就會出現這種情況。

## 4 Experiments

## 4 實驗

### 4.1 Datasets and Settings

### 4.1 數據集與設定

We use in our experiments the datasets FB15(k), its modified variant FB15-237, WN18, and its modified variant WN18RR. The FB (WN) datasets are based on a subset of FreeBase (WordNet). FB15 and WN18 have been first used in [3]. They have been criticised in several papers [29, 7], where the authors argued that due to redundancies a large fraction of testcases can be solved by exploiting rather simple rules. FB15-237 [29] and WN18RR [7] have been proposed as modified variants with suppressed redundancies. The dataset YAGO03-10 (in short YAGO) is described in [15] and has first been used in the context of knowledge completion in [7]. It is two times larger in number of triples compared to FB15. An overview is given in Table 1.

我們在實驗中使用了 FB15(k)、其修改版 FB15-237、WN18 及其修改版 WN18RR 等數據集。FB (WN) 數據集基於 FreeBase (WordNet) 的一個子集。FB15 和 WN18 最早於 [3] 中使用。它們在多篇論文 [29, 7] 中受到批評，作者認為由於冗餘，很大一部分測試案例可以通過利用相當簡單的規則來解決。FB15-237 [29] 和 WN18RR [7] 被提出作為抑制了冗餘的修改版本。YAGO03-10 數據集（簡稱 YAGO）在 [15] 中有描述，並最早在 [7] 中用於知識補全的背景。它的三元組數量是 FB15 的兩倍大。表 1 給出了一個概述。

The most commonly used evaluation metrics are the filtered hits@1 and hits@10 scores introduced in [3]. The hits@k scores measure how often (fraction of test cases) the correct answer, which is defined by the triple that the test case originates from, is among the top-k ranked entities. In the following we always refer to the filtered scores without explicitly stating it. Another important value is the filtered MRR (mean rank reciprocal). As our approach is not designed to compute complete rankings but top-k rankings only, we compute a lower bound by assuming that any candidate which would be ranked at a position >k is not a correct prediction.

最常用的評估指標是在 [3] 中引入的過濾後的 hits@1 和 hits@10 分數。hits@k 分數衡量正確答案（由測試案例源自的三元組定義）出現在排名前 k 的實體中的頻率（測試案例的比例）。在下文中，我們總是引用過濾後的分數，而不做明確說明。另一個重要的值是過濾後的 MRR（平均倒數排名）。由於我們的方法不是為了計算完整排名，而只是計算前 k 名，我們通過假設任何排名在 >k 位置的候選者都不是正確的預測來計算一個下界。

As described in [16], we use max aggregation to generate predictions from the rule set. We learn rules up to length 3 from cyclic paths (length 5 for WN and WN18RR) and restrict the length of rules learned from acyclic paths to 1. Confidences of rules are approximated by sampling and evaluating groundings on the training set followed by a laplace smoothing with parameter pc = 5. We keep all rules with a confidence higher than 0.0001 that reconstructed at least two triples in the training set. If not stated otherwise, we use the weighted reinforced policy together with reward strategy Rsxc. We are running AnyBURL on a CPU sever with 24 Intel(R) Xeon(R) CPU E5-2630 v2 @ 2.60GHz cores. We use 22 threads and reserve 50 GB RAM for our experiments.

如 [16] 所述，我們使用最大聚合從規則集中生成預測。我們從循環路徑學習長度最多為 3 的規則（對於 WN 和 WN18RR 為 5），並將從非循環路徑學習的規則長度限制為 1。規則的置信度是通過在訓練集上抽樣和評估基實例，然後使用參數 pc = 5 進行拉普拉斯平滑來近似的。我們保留所有置信度高於 0.0001 且在訓練集中至少重建了兩個三元組的規則。除非另有說明，我們使用加權強化策略和獎勵策略 Rsxc。我們在配備 24 個 Intel(R) Xeon(R) CPU E5-2630 v2 @ 2.60GHz 核心的 CPU 伺服器上運行 AnyBURL。我們使用 22 個線程並為實驗保留 50 GB RAM。

### 4.2 Object Identity

### 4.2 物件恆等性

We first run AnyBURL with deactivated OI constraints for a fixed amount of time (1000 seconds) on the WN18 dataset. For that purpose we deactivate rules with constants and learn only binary rules of length 1 to 5. We run our experiments in two settings. In a strict setting, we set the minimum support to 100 and the minimum confidence to 0.5. In a relaxed setting, we use lower thresholds, i.e., we set the minimum support to 10 and the confidence threshold to 0.1. In a post processing step, we activate the OI constraints and recompute confidences for the previously computed rule sets. Then we count the fraction of rules that remain above these thresholds. We evaluate both rules sets in both settings on the test set to measure the quality of the resulting predictions.

我們首先在 WN18 數據集上運行 AnyBURL，禁用 OI 約束，固定時間為 1000 秒。為此，我們禁用帶有常數的規則，僅學習長度為 1 到 5 的二元規則。我們在兩種設置下運行實驗。在嚴格設置中，我們將最小支持度設置為 100，最小置信度設置為 0.5。在寬鬆設置中，我們使用較低的閾值，即最小支持度設置為 10，置信度閾值設置為 0.1。在後處理步驟中，我們激活 OI 約束，並為先前計算的規則集重新計算置信度。然後我們計算保持在這些閾值之上的規則的比例。我們在測試集上評估兩種設置下的兩種規則集，以衡量最終預測的質量。

To avoid inaccuracies caused by approximated confidences, we filter only those rules for which the scores are significantly lower than the threshold. Otherwise, we would not know if a rule that was only slightly above the threshold falls below the threshold due to a sampling inaccuracy or due to an OI constraint. We assume that a confidence or support score is significantly lower if the value is lower than half of the originally chosen threshold.

為避免近似置信度造成的不準確，我們只過濾那些分數遠低於閾值的規則。否則，我們將無法判斷一個僅略高於閾值的規則是因為抽樣不準確還是因為 OI 約束而跌破閾值。我們假設，如果一個置信度或支持度分數的值低於最初選擇閾值的一半，則該分數顯著較低。

The results of our experiments are shown in Table 2. We can see that activating OI constraints has a strong impact on rules that have a high confidence without these constraints. This is highlighted by the results for the strict setting, shown in the first two rows. The number of rules drops from 2004 to 215, which means that only 1/10 of the rules remain above the thresholds. Furthermore, the predictive results support the theoretic considerations from Section 2.3. The fact that hits@1 increases from 0.739 to 0.938, is a strong indicator that many of the rules, which have been filtered, had a confidence score that was too high without OI constraints. It is also important to note that OI constraints are not only useful to obtain a high precision, which is reflected in the hits@1 score, but also we observe a significant improvement for hits@10.

我們的實驗結果如表 2 所示。我們可以看到，啟用 OI 約束對那些沒有這些約束但具有高置信度的規則有很強的影響。這在嚴格設定的結果中得到了突顯，如前兩行所示。規則數量從 2004 條下降到 215 條，這意味著只有 1/10 的規則保持在閾值之上。此外，預測結果支持第 2.3 節的理論考慮。hits@1 從 0.739 增加到 0.938 的事實，是一個強有力的指標，表明許多被過濾掉的規則在沒有 OI 約束的情況下具有過高的置信度分數。同樣重要的是要注意，OI 約束不僅有助於獲得高精度（這反映在 hits@1 分數中），而且我們還觀察到 hits@10 的顯著改進。

The impact on filtering is less strict for the second setting. Around half of the rules are filtered out. However, the chosen thresholds are relatively low. Nevertheless, the impact on the predictive quality is similar to the first setting. This is caused by the modified confidence scores. The results are not surprising, because the rule set generated in the first setting is the subset of the second rule set that includes the most influential rules. However, if we compare both settings under OI, the second setting achieves better hits@1
and hits@10 scores. This means, that rules with a confidence lower than 0.5 are now able to contribute to the generated ranking.

在第二個設定中，過濾的影響較不嚴格。大約一半的規則被過濾掉。然而，選擇的閾值相對較低。儘管如此，對預測質量的影響與第一個設定相似。這是由修改後的置信度分數引起的。結果並不令人驚訝，因為在第一個設定中生成的規則集是第二個規則集的子集，它包含了最具影響力的規則。然而，如果我們比較 OI 下的兩種設定，第二個設定在 hits@1 和 hits@10 分數上表現更好。這意味著，置信度低於 0.5 的規則現在能夠對生成的排名做出貢獻。

As argued in several studies [29, 7, 17], WN18 allows to learn many simple rules that have a high predictive power. These rules can then become redundant building blocks in longer rules. As soon as we interpret these rules under OI, their scores are corrected, resulting in better predictions and, if we apply a threshold, into smaller rule sets. While the impact might be less strong on other datasets, the underlying pattern will always have an impact as long as there are rules of different length and some of the shorter rules have a relatively high score.

正如幾項研究 [29, 7, 17] 所論證的，WN18 允許學習許多具有高預測能力的簡單規則。這些規則隨後可以在較長的規則中成為多餘的構建塊。一旦我們在 OI 下解釋這些規則，它們的分數就會被修正，從而得到更好的預測，如果我們應用閾值，則會得到更小的規則集。雖然對其他數據集的影響可能不那麼強烈，但只要存在不同長度的規則並且一些較短的規則具有相對較高的分數，潛在的模式就總會產生影響。

### 4.3 Reinforcement Learning

### 4.3 強化學習

We used the largest dataset YAGO03-10 to compare the (i) saturation based approach with different saturation boundaries (0.9, 0.99, and 0.999) against the (ii) random policy and the (iii) weighted reinforcement policy together with Rsxc in a first experiment. We learned rules in each of these settings for 1000 seconds, taking frequent snapshots of the learned rules. For each of these snapshots, we computed the predictions against the test set. The resulting hits@10 scores are depicted in Figure 3. We observe a very good anytime behaviour for most settings. The weighted policy causes the fastest increase: after 200 seconds we have learned a rule set with a hits@10 score of 68.6%. This score is only slightly improved by 0.4% when increasing the available time to 1000 seconds. The second best approach is the random policy, if we consider a quick improvement at the beginning as important. After a short time (50 to 200 seconds), it achieves better results than the saturation based approach with a boundary set to 0.99. There is also a time period in which a saturation-based approach performs slightly better.

我們使用最大的數據集 YAGO03-10 來比較 (i) 具有不同飽和邊界（0.9、0.99 和 0.999）的基於飽和度的方法，與 (ii) 隨機策略和 (iii) 加權強化學習策略以及 Rsxc 在第一個實驗中。我們在這些設置中的每一個都學習了 1000 秒的規則，並頻繁地對學習到的規則進行快照。對於每個快照，我們都針對測試集計算了預測。由此產生的 hits@10 分數如圖 3 所示。我們觀察到大多數設置都具有非常好的即時性。加權策略導致最快的增長：200 秒後，我們學到了一個規則集，其 hits@10 分數為 68.6%。當可用時間增加到 1000 秒時，該分數僅略微提高了 0.4%。如果我們認為開始時的快速改進很重要，那麼第二好的方法是隨機策略。在短時間（50 到 200 秒）後，它比邊界設置為 0.99 的基於飽和度的方法取得了更好的結果。還有一段時間，基於飽和度的方法表現稍好。

We observed for all three settings of the saturation-based approach, that the saturation for Ua and Uc rules of length one created from acyclic paths does not reach the boundary within 1000 seconds. This is different for the rules generated from cyclic paths. The saturation boundary of 0.9 has been passed after 4 seconds for rules of length one and again after 9 seconds for rules of length two. The corresponding times were 16 and 198 seconds for 0.99, and 307 and 397 seconds for 0.999 respectively.

我們觀察到，在所有三種基於飽和度的方法設置中，從非循環路徑創建的長度為一的 Ua 和 Uc 規則的飽和度在 1000 秒內未達到邊界。這與從循環路徑生成的規則不同。對於長度為一的規則，0.9 的飽和度邊界在 4 秒後被超過，對於長度為二的規則，則在 9 秒後再次被超過。對於 0.99，相應的時間分別為 16 和 198 秒，對於 0.999，則為 307 和 397 秒。

A saturation boundary of 0.9 is too low. The early jump to the longer paths causes that some beneficial rules cannot be found. This seems not to be the case for the boundary 0.99. However, the results for 0.99 are slightly worse than the results for 0.999 after mining rules for 1000 seconds. While a high boundary of 0.999 is beneficial on the long run, it prevents that good scores are achieved early. This is caused by the importance of some B rules of length three, which are not generated before 397 seconds have passed.

飽和邊界 0.9 太低。過早跳到較長的路徑會導致一些有益的規則無法被找到。對於邊界 0.99 來說，情況似乎並非如此。然而，在挖掘規則 1000 秒後，0.99 的結果比 0.999 的結果稍差。雖然 0.999 的高邊界從長遠來看是有益的，但它阻礙了早期取得好成績。這是由於一些長度為三的 B 規則的重要性，這些規則在 397 秒之前不會被生成。

In the following, we compare the random baseline against all possible combinations of policies and reward strategies. Results are shown in Table 3. We evaluated each setting three times (six times for 50s and 100s) and report the resulting averages. In particular, we compare each reinforcement setting against the random policy on the three largest datasets showing the difference in terms of reinforced versus random approach. We observe improvements compared to the random policy for FB15-237 and YAGO for each combination of policy and reward strategy. However, there is not a single combination that performs clearly better than the other ones. This is a bit surprising, as we would have expected a positive impact of taking confidence into account.

接下來，我們將隨機基線與所有可能的策略和獎勵策略組合進行比較。結果如表 3 所示。我們對每個設置評估了三次（對於 50 秒和 100 秒則為六次），並報告了由此產生的平均值。特別是，我們在三個最大的數據集上比較了每個強化設置與隨機策略，顯示了強化方法與隨機方法的差異。我們觀察到，對於 FB15-237 和 YAGO，每種策略和獎勵策略的組合都比隨機策略有所改進。然而，沒有一種組合明顯優於其他組合。這有點令人驚訝，因為我們預期將置信度納入考慮會產生積極影響。

To better understand the meaning of the numbers in Table 3, we take a look at the hits@10 gain of +0.6% in the Weighted/Rsxc column of the YAGO03-10 500 seconds row. A plus of 0.6% looks like a minor improvement at first sight. However, changing from random to the reinforcement policy achieves 67.8%+0.6% = 68.4% hits@10 which is higher than the 1000 seconds score of the random policy (68.3%). Thus, the same (or slightly better) results are achieved in half of the time.

為了更好地理解表 3 中數字的含義，我們看一下 YAGO03-10 500 秒行中加權/Rsxc 列的 hits@10 增益 +0.6%。乍一看，+0.6% 似乎是一個微小的改進。然而，從隨機策略改為強化學習策略，達到了 67.8%+0.6% = 68.4% 的 hits@10，這比隨機策略 1000 秒的分數 (68.3%) 還要高。因此，在一半的時間內就取得了相同（或稍好）的結果。

The results for FB15 after 50 and 100 seconds are an exception from this trend. All policies guide the search into the wrong direction at the beginning and the reward strategies, that do not take into account rule length, perform also worse after 100 seconds. After 500 seconds the random policy and the other policies achieve similar results. A possible explanation for this behaviour can be the fact that FB15 has many redundancies and a high number of relations. This implies that regularities that require longer rules can be expressed in many different ways by replacing one atom by an (nearly) equivalent atom. For that reason, cyclic path profiles of length two and three receive a reward by Rs and Rsxc that is too high compared to short rules. This is also the reason why the greedy policy together with reward strategy Rsxc/21 performs best on FB15. It favours short rules over longer rules and mitigates the described effect without negative impact on the results measured for the other datasets.

FB15 在 50 秒和 100 秒後的結果是此趨勢的一個例外。所有策略在開始時都將搜索引導到錯誤的方向，而不考慮規則長度的獎勵策略在 100 秒後也表現得更差。500 秒後，隨機策略和其他策略取得了相似的結果。這種行為的一個可能解釋是 FB15 有許多冗餘和大量的關係。這意味著需要更長規則的規律性可以通過用一個（幾乎）等效的原子替換一個原子來以許多不同的方式表達。因此，長度為二和三的循環路徑輪廓通過 Rs 和 Rsxc 獲得了與短規則相比過高的獎勵。這也是為什麼貪婪策略與獎勵策略 Rsxc/21 在 FB15 上表現最好的原因。它偏愛短規則而非長規則，並減輕了所描述的效應，而對其他數據集測量的結果沒有負面影響。

### 4.4 State of the Art

### 4.4 當前技術水平

In the first block of Table 4 we compare the results of AnyBURL against 16 different models presented in [23]. The second block (marked with *) lists the results from [24]. Here the authors report about the performance of the classic models RESCAL, TransE, DistMult, ComplEx and ConvE, arguing that these models perform better than usually reported and quite comparable to each other if the training strategies and other relevant hyperparameters are correctly tuned. While [23] reports numbers for all datasets that we used, [24] report only results related to WN18RR and FB15-237. In the AnyBURL block, the †10000s row refers to the 10000 seconds run of the previous AnyBURL version reported in [16], while the rows below refer to the new version.

在表 4 的第一區塊中，我們將 AnyBURL 的結果與 [23] 中提出的 16 種不同模型進行了比較。第二區塊（標有 *）列出了 [24] 的結果。在此，作者報告了經典模型 RESCAL、TransE、DistMult、ComplEx 和 ConvE 的性能，認為如果訓練策略和其他相關超參數調整得當，這些模型的表現會比通常報導的要好，並且彼此之間相當具有可比性。雖然 [23] 報告了我們使用的所有數據集的數據，但 [24] 僅報告了與 WN18RR 和 FB15-237 相關的結果。在 AnyBURL 區塊中，†10000s 行指的是 [16] 中報告的先前 AnyBURL 版本的 10000 秒運行結果，而下面的行則指的是新版本。

The test and validation set of FB15-237 have been filtered by removing triples that connect two entities which are already connected in the training set. According to the training set of FB15-237 we are sometimes right to say that h(c, c') holds, if we already know that b(c, c') or b(c', c) holds. Contrary to this, the specific setup will punish such conclusions. We check prior to the prediction, whether the validation set connects entities not connected in the training set. If this is not the case, we block any prediction of a triple with two entities that are already connected. Note that this setting, which is always activated, has no impact on any other dataset, while it improves results for FB15-237 by ≈2%.

FB15-237 的測試集和驗證集已通過移除連接訓練集中已連接的兩個實體的三元組進行了過濾。根據 FB15-237 的訓練集，如果我們已經知道 b(c, c') 或 b(c', c) 成立，我們有時會說 h(c, c') 成立是正確的。與此相反，特定的設置會懲罰這樣的結論。我們在預測之前檢查驗證集是否連接了訓練集中未連接的實體。如果不是這種情況，我們會阻止任何對已連接的兩個實體的三元組的預測。請注意，此設置始終處於活動狀態，對任何其他數據集沒有影響，但將 FB15-237 的結果提高了約 2%。

AnyBURL is not capable of learning rules with a head such as h(X, X). This means that it cannot predict that an entity is related to itself via h. However, in some of the datasets (FB15 and FB15-237), a small subset in the training and test sets consist such triples. To allow AnyBURL to learn meaningful rules in such a situation, we rewrite triples like h(c, c) to h(c, self) by introducing a new constant self. Thus, AnyBURL can, for example, learn a Ua or Uc rule such as h(X, self) ← b(X, A) or h(X, self) ← b(X, c). After applying the rules, we convert a prediction of the form h(c, self) into h(c, c).

AnyBURL 無法學習頭部為 h(X, X) 這類型的規則。這意味著它無法預測一個實體透過 h 關係與自身相關。然而，在某些資料集（FB15 和 FB15-237）中，訓練集和測試集的一小部分包含這類三元組。為了讓 AnyBURL 在這種情況下學習有意義的規則，我們將像 h(c, c) 這樣的三元組改寫為 h(c, self)，並引入一個新的常數 self。因此，AnyBURL 可以，例如，學習一個 Ua 或 Uc 規則，像是 h(X, self) ← b(X, A) 或 h(X, self) ← b(X, c)。在應用規則後，我們將形式為 h(c, self) 的預測轉換回 h(c, c)。

We have ranked all approaches for each combination of metric and dataset and present in the last row the rank that was achieved by applying the rules that AnyBURL learned after 10000 seconds. AnyBURL is in six cases on the first position, in six cases on the second position, and in the remaining three cases on position 3, 5 and 9. ComplEx performs also quite well on some datasets, however, it is not among the best models for WN18 and WN18RR. Moreover, AnyBURL performs in particular very good if we look at the hits@1 score. For each dataset, AnyBURL is the best or the second best system if we look only at the top-ranked candidate. Another remarkable result, is the capability of AnyBURL to learn in short time a rule set, that is already competitive compared to the other approaches. This can be concluded from the results achieved after 100 seconds. It is worth noting that unlike embedding models, AnyBURL does not require time-consuming hyperparameter tuning to achieve these numbers.

我們對每個指標和數據集的組合的所有方法進行了排名，並在最後一行呈現了應用 AnyBURL 在 10000 秒後學習的規則所達到的排名。AnyBURL 在六種情況下排名第一，六種情況下排名第二，其餘三種情況下分別排名第 3、第 5 和第 9。ComplEx 在某些數據集上也表現得相當不錯，然而，它並非 WN18 和 WN18RR 的最佳模型之一。此外，如果我們看 hits@1 分數，AnyBURL 的表現尤其出色。對於每個數據集，如果我們只看排名最高的候選者，AnyBURL 是最好或第二好的系統。另一個顯著的結果是，AnyBURL 能夠在短時間內學習到一個與其他方法相比已經具有競爭力的規則集。這可以從 100 秒後達到的結果中得出結論。值得注意的是，與嵌入模型不同，AnyBURL 不需要耗時的超參數調整來達到這些數字。

If we compare our results against the previous version of AnyBURL there seems to be only a small improvement. However, this improvement takes place in a range where it is hard to make better predictions. When manually analysing some of the given predictions, we realized that the predictions made by AnyBURL are sometimes right, even though they are not specified as facts in test, training or validation. These new correct facts, that are counted as wrong predictions, might share a common characteristic. An approach that looks into the validation set might be able to tune its hyperparameters to avoid these predictions. AnyBURL cannot capture such a regularity.

如果我們將我們的結果與先前版本的 AnyBURL 進行比較，似乎只有微小的改進。然而，這種改進發生在很難做出更好預測的範圍內。當手動分析一些給定的預測時，我們意識到 AnyBURL 做出的預測有時是正確的，儘管它們在測試、訓練或驗證中沒有被指定為事實。這些被算作錯誤預測的新的正確事實，可能具有共同的特徵。一種查看驗證集的方法可能能夠調整其超參數以避免這些預測。AnyBURL 無法捕捉到這種規律性。

## 5 Related Work

## 5 相關工作

Most knowledge graph completion techniques are based on the concept of embeddings. There are also some approaches that try to combine embeddings and rules. An example is the system Ruge [12], which learns rules, materializes these rules, and injects the materialized triples as new training examples with soft labels into the process of learning the embedding. The authors report results on FB15 which are worse than the results achieved by AnyBURL after 100 seconds. The benefits of combining rules and embeddings can only be understood, if we know first how far one can get with each method on its own. With our work, we show that rules on their own perform surprisingly well, which should not be neglected in further work on combining embeddings and rules.

大多數知識圖譜補全技術都基於嵌入的概念。也有一些方法試圖結合嵌入和規則。一個例子是 Ruge 系統 [12]，它學習規則，將這些規則實體化，並將實體化的三元組作為帶有軟標籤的新訓練樣本注入到嵌入學習過程中。作者報告了在 FB15 上的結果，這些結果比 AnyBURL 在 100 秒後取得的結果要差。只有當我們先知道每種方法單獨能走多遠時，才能理解結合規則和嵌入的好處。通過我們的工作，我們表明規則本身表現得出奇地好，這在未來結合嵌入和規則的工作中不應被忽視。

Recently, reinforcement learning has been used for the task of query answering in [4, 14, 30]. These approaches have been applied to knowledge graph completion. Similar to AnyBURL, they provide explanations, however, they rely on vector representations and not on symbols. While these approaches use a reward strategy for paths that lead to answer nodes, AnyBURL uses reward strategies for path profiles that provide paths which result into rules. Even though in [4, 14] FB15-237 and WN18RR are used, the results are based on a different evaluation procedure and/or a different test data split. Under this evaluation set up, the reinforced approaches perform as good or worse as ConvE, which is included in Table 4 for comparison.

最近，強化學習已被用於查詢回答任務 [4, 14, 30]。這些方法已被應用於知識圖譜補全。與 AnyBURL 類似，它們提供了解釋，然而，它們依賴於向量表示而不是符號。雖然這些方法使用導致答案節點的路徑的獎勵策略，但 AnyBURL 使用為提供導致規則的路徑的路徑配置文件提供獎勵的策略。儘管在 [4, 14] 中使用了 FB15-237 和 WN18RR，但結果基於不同的評估程序和/或不同的測試數據分割。在此評估設置下，強化方法的表現與 ConvE 一樣好或更差，ConvE 已包含在表 4 中以供比較。

## 6 Conclusion

## 6 結論

We introduced two extensions of our rule mining system AnyBURL. We explained and argued, based on our experimental results, that a rule-based solution to the knowledge graph completion problem should be based on Object Identity. As second contribution we introduced a reinforcement learning technique to guide the sampling process in order to use available computational resources in a reasonable way. Both extensions are implemented in the new version of AnyBURL available at http://web.informatik.uni-mannheim.de/AnyBURL/. We have evaluated this new version and compared the results against current state of the art embedding based techniques. The results show that most of these approaches cannot achieve the predictive quality of our approach, nor can they explain their predictions, which is a significant disadvantage compared to a symbolic approach.

我們介紹了我們的規則挖掘系統 AnyBURL 的兩個擴展。我們根據我們的實驗結果解釋並論證了，基於規則的知識圖譜補全問題的解決方案應該基於物件恆等性。作為第二個貢獻，我們引入了一種強化學習技術來指導採樣過程，以便以合理的方式使用可用的計算資源。這兩個擴展都在新版本的 AnyBURL 中實現，可在 http://web.informatik.uni-mannheim.de/AnyBURL/ 上獲取。我們評估了這個新版本，並將結果與當前最先進的基於嵌入的技術進行了比較。結果表明，這些方法中的大多數都無法達到我們方法的預測質量，也無法解釋它們的預測，這與符號方法相比是一個顯著的缺點。

**References**

**參考文獻**

[1] Sören Auer, Christian Bizer, Georgi Kobilarov, Jens Lehmann, Richard Cyganiak, and Zachary Ives. Dbpedia: A nucleus for a web of open data. In The semantic web, pages 722–735. Springer, 2007.

[1] Sören Auer, Christian Bizer, Georgi Kobilarov, Jens Lehmann, Richard Cyganiak, and Zachary Ives. Dbpedia: A nucleus for a web of open data. In The semantic web, pages 722–735. Springer, 2007.

[2] Kurt Bollacker, Colin Evans, Praveen Paritosh, Tim Sturge, and Jamie Taylor. Freebase: a collaboratively created graph database for structuring human knowledge. In Proceedings of the 2008 ACM SIGMOD international conference on Management of data, pages 1247–1250, 2008.

[2] Kurt Bollacker, Colin Evans, Praveen Paritosh, Tim Sturge, and Jamie Taylor. Freebase: a collaboratively created graph database for structuring human knowledge. In Proceedings of the 2008 ACM SIGMOD international conference on Management of data, pages 1247–1250, 2008.

[3] Antoine Bordes, Nicolas Usunier, Alberto Garcia-Duran, Jason Weston, and Oksana Yakhnenko. Translating embeddings for modeling multi-relational data. In Advances in neural information processing systems, pages 2787–2795, 2013.

[3] Antoine Bordes, Nicolas Usunier, Alberto Garcia-Duran, Jason Weston, and Oksana Yakhnenko. Translating embeddings for modeling multi-relational data. In Advances in neural information processing systems, pages 2787–2795, 2013.

[4] Rajarshi Das, Shehzaad Dhuliawala, Manzil Zaheer, Luke Vilnis, Ishan Durugkar, Akshay Krishnamurthy, Alex Smola, and Andrew McCallum. Go for a walk and arrive at the answer: Reasoning over paths in knowledge bases using reinforcement learning. In Sixth International Conference on Learning Representations (ICLR 2018), 2018.

[4] Rajarshi Das, Shehzaad Dhuliawala, Manzil Zaheer, Luke Vilnis, Ishan Durugkar, Akshay Krishnamurthy, Alex Smola, and Andrew McCallum. Go for a walk and arrive at the answer: Reasoning over paths in knowledge bases using reinforcement learning. In Sixth International Conference on Learning Representations (ICLR 2018), 2018.

[5] Luc De Raedt. Logical and relational learning. Springer Science & Business Media, 2008.

[5] Luc De Raedt. Logical and relational learning. Springer Science & Business Media, 2008.

[6] Luc Dehaspe and Hannu Toivonen. Discovery of relational association rules. In Relational data mining, pages 189-212. Springer, 2001.

[6] Luc Dehaspe and Hannu Toivonen. Discovery of relational association rules. In Relational data mining, pages 189-212. Springer, 2001.

[7] Tim Dettmers, Pasquale Minervini, Pontus Stenetorp, and Sebastian Riedel. Convolutional 2d knowledge graph embeddings. In Thirty-Second AAAI Conference on Artificial Intelligence, 2018.

[7] Tim Dettmers, Pasquale Minervini, Pontus Stenetorp, and Sebastian Riedel. Convolutional 2d knowledge graph embeddings. In Thirty-Second AAAI Conference on Artificial Intelligence, 2018.

[8] Xin Dong, Evgeniy Gabrilovich, Geremy Heitz, Wilko Horn, Ni Lao, Kevin Murphy, Thomas Strohmann, Shaohua Sun, and Wei Zhang. Knowledge vault: A web-scale approach to probabilistic knowledge fusion. In Proceedings of the 20th ACM SIGKDD international conference on Knowledge discovery and data mining, pages 601-610, 2014.

[8] Xin Dong, Evgeniy Gabrilovich, Geremy Heitz, Wilko Horn, Ni Lao, Kevin Murphy, Thomas Strohmann, Shaohua Sun, and Wei Zhang. Knowledge vault: A web-scale approach to probabilistic knowledge fusion. In Proceedings of the 20th ACM SIGKDD international conference on Knowledge discovery and data mining, pages 601-610, 2014.

[9] Floriana Esposito, Angela Laterza, Donato Malerba, and Giovanni Semeraro. Refinement of datalog programs. In Proceedings of the MLnet familiarization workshop on data mining with inductive logic programming, pages 73-94, 1996.

[9] Floriana Esposito, Angela Laterza, Donato Malerba, and Giovanni Semeraro. Refinement of datalog programs. In Proceedings of the MLnet familiarization workshop on data mining with inductive logic programming, pages 73-94, 1996.

[10] Luis Galárraga, Christina Teflioudi, Katja Hose, and Fabian M Suchanek. Fast rule mining in ontological knowledge bases with AMIE+. The VLDB Journal-The International Journal on Very Large Data Bases, 24(6):707–730, 2015.

[10] Luis Galárraga, Christina Teflioudi, Katja Hose, and Fabian M Suchanek. Fast rule mining in ontological knowledge bases with AMIE+. The VLDB Journal-The International Journal on Very Large Data Bases, 24(6):707–730, 2015.

[11] Luis Antonio Galárraga, Christina Teflioudi, Katja Hose, and Fabian Suchanek. Amie: association rule mining under incomplete evidence in ontological knowledge bases. In Proceedings of the 22nd international conference on World Wide Web, pages 413-422. ACM, 2013.

[11] Luis Antonio Galárraga, Christina Teflioudi, Katja Hose, and Fabian Suchanek. Amie: association rule mining under incomplete evidence in ontological knowledge bases. In Proceedings of the 22nd international conference on World Wide Web, pages 413-422. ACM, 2013.

[12] Shu Guo, Quan Wang, Lihong Wang, Bin Wang, and Li Guo. Knowledge graph embedding with iterative guidance from soft rules. In Thirty-Second AAAI Conference on Artificial Intelligence, 2018.

[12] Shu Guo, Quan Wang, Lihong Wang, Bin Wang, and Li Guo. Knowledge graph embedding with iterative guidance from soft rules. In Thirty-Second AAAI Conference on Artificial Intelligence, 2018.

[13] Michael N Katehakis and Arthur F Veinott Jr. The multi-armed bandit problem: decomposition and computation. Mathematics of Operations Research, 12(2):262-268, 1987.

[13] Michael N Katehakis and Arthur F Veinott Jr. The multi-armed bandit problem: decomposition and computation. Mathematics of Operations Research, 12(2):262-268, 1987.

[14] Xi Victoria Lin, Richard Socher, and Caiming Xiong. Multi-hop knowledge graph reasoning with reward shaping. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing (EMNLP 2018), 2018.

[14] Xi Victoria Lin, Richard Socher, and Caiming Xiong. Multi-hop knowledge graph reasoning with reward shaping. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing (EMNLP 2018), 2018.

[15] Farzaneh Mahdisoltani, Joanna Biega, and Fabian M Suchanek. Yago3: A knowledge base from multilingual wikipedias. In Proceedings of CIDR 2015, 2015.

[15] Farzaneh Mahdisoltani, Joanna Biega, and Fabian M Suchanek. Yago3: A knowledge base from multilingual wikipedias. In Proceedings of CIDR 2015, 2015.

[16] Christian Meilicke, Melisachew Wudage Chekol, Daniel Ruffinelli, and Heiner Stuckenschmidt. Anytime bottom-up rule learning for knowledge graph completion. In Proceedings of the 28th International Joint Conference on Artificial Intelligence (IJCAI). IJCAI/AAAI Press, 2019.

[16] Christian Meilicke, Melisachew Wudage Chekol, Daniel Ruffinelli, and Heiner Stuckenschmidt. Anytime bottom-up rule learning for knowledge graph completion. In Proceedings of the 28th International Joint Conference on Artificial Intelligence (IJCAI). IJCAI/AAAI Press, 2019.

[17] Christian Meilicke, Manuel Fink, Yanjie Wang, Daniel Ruffinelli, Rainer Gemulla, and Heiner Stuckenschmidt. Fine-grained evaluation of rule- and embedding-based systems for knowledge graph completion. In Proceedings of the International Semantic Web Conference, pages 3–20. Springer International Publishing, 2018.

[17] Christian Meilicke, Manuel Fink, Yanjie Wang, Daniel Ruffinelli, Rainer Gemulla, and Heiner Stuckenschmidt. Fine-grained evaluation of rule- and embedding-based systems for knowledge graph completion. In Proceedings of the International Semantic Web Conference, pages 3–20. Springer International Publishing, 2018.

[18] Christian Meilicke, Manuel Fink, Yanjie Wang, Daniel Ruffinelli, Rainer Gemulla, and Heiner Stuckenschmidt. Fine-grained evaluation of rule- and embedding-based systems for knowledge graph completion. In International Semantic Web Conference, pages 3-20. Springer, 2018.

[18] Christian Meilicke, Manuel Fink, Yanjie Wang, Daniel Ruffinelli, Rainer Gemulla, and Heiner Stuckenschmidt. Fine-grained evaluation of rule- and embedding-based systems for knowledge graph completion. In International Semantic Web Conference, pages 3-20. Springer, 2018.

[19] Stephen Muggleton and Luc De Raedt. Inductive logic programming: Theory and methods. The Journal of Logic Programming, 19:629-679, 1994.

[19] Stephen Muggleton and Luc De Raedt. Inductive logic programming: Theory and methods. The Journal of Logic Programming, 19:629-679, 1994.

[20] Maximilian Nickel, Volker Tresp, and Hans-Peter Kriegel. A three-way model for collective learning on multi-relational data. In ICML, volume 11, pages 809-816, 2011.

[20] Maximilian Nickel, Volker Tresp, and Hans-Peter Kriegel. A three-way model for collective learning on multi-relational data. In ICML, volume 11, pages 809-816, 2011.

[21] Stefano Pallottino. Shortest-path methods: Complexity, interrelations and new propositions. Networks, 14(2):257-267, 1984.

[21] Stefano Pallottino. Shortest-path methods: Complexity, interrelations and new propositions. Networks, 14(2):257-267, 1984.

[22] John Alan Robinson. A machine-oriented logic based on the resolution principle. Journal of the ACM (JACM), 12(1):23–41, 1965.

[22] John Alan Robinson. A machine-oriented logic based on the resolution principle. Journal of the ACM (JACM), 12(1):23–41, 1965.

[23] Andrea Rossi, Donatella Firmani, Antonio Matinata, Paolo Merialdo, and Denilson Barbosa. Knowledge graph embedding for link prediction: A comparative analysis, 2020.

[23] Andrea Rossi, Donatella Firmani, Antonio Matinata, Paolo Merialdo, and Denilson Barbosa. Knowledge graph embedding for link prediction: A comparative analysis, 2020.

[24] Daniel Ruffinelli, Samuel Broscheit, and Rainer Gemulla. You can teach an old dog new tricks! on training knowledge graph embeddings. In Proceedings of ICLR 2020, 2020.

[24] Daniel Ruffinelli, Samuel Broscheit, and Rainer Gemulla. You can teach an old dog new tricks! on training knowledge graph embeddings. In Proceedings of ICLR 2020, 2020.

[25] Giovanni Semeraro, Floriana Esposito, Donato Malerba, Clifford Brunk, and Michael Pazzani. Avoiding non-termination when learning logic programs: A case study with foil and focl. In Logic Program Synthesis and Transformation—Meta-Programming in Logic, pages 183-198. Springer, 1994.

[25] Giovanni Semeraro, Floriana Esposito, Donato Malerba, Clifford Brunk, and Michael Pazzani. Avoiding non-termination when learning logic programs: A case study with foil and focl. In Logic Program Synthesis and Transformation—Meta-Programming in Logic, pages 183-198. Springer, 1994.

[26] Ashwin Srinivasan. The aleph manual(techical report). Technical report, Computing Laboratory, Oxford University, 2000.

[26] Ashwin Srinivasan. The aleph manual(techical report). Technical report, Computing Laboratory, Oxford University, 2000.

[27] Fabian M Suchanek, Gjergji Kasneci, and Gerhard Weikum. Yago: a core of semantic knowledge. In Proceedings of the 16th international conference on World Wide Web, pages 697-706, 2007.

[27] Fabian M Suchanek, Gjergji Kasneci, and Gerhard Weikum. Yago: a core of semantic knowledge. In Proceedings of the 16th international conference on World Wide Web, pages 697-706, 2007.

[28] Richard S Sutton and Andrew G Barto. Reinforcement learning: An introduction. MIT press, 2 edition, 2018.

[28] Richard S Sutton and Andrew G Barto. Reinforcement learning: An introduction. MIT press, 2 edition, 2018.

[29] Kristina Toutanova and Danqi Chen. Observed versus latent features for knowledge base and text inference. In Proceedings of the 3rd Workshop on Continuous Vector Space Models and their Compositionality, pages 57-66, 2015.

[29] Kristina Toutanova and Danqi Chen. Observed versus latent features for knowledge base and text inference. In Proceedings of the 3rd Workshop on Continuous Vector Space Models and their Compositionality, pages 57-66, 2015.

[30] Wenhan Xiong, Thien Hoang, and William Yang Wang. Deeppath: A reinforcement learning method for knowledge graph reasoning. In Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing (EMNLP 2017), 2017.

[30] Wenhan Xiong, Thien Hoang, and William Yang Wang. Deeppath: A reinforcement learning method for knowledge graph reasoning. In Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing (EMNLP 2017), 2017.

The translation of the document is complete. I have followed all the formatting and content instructions.