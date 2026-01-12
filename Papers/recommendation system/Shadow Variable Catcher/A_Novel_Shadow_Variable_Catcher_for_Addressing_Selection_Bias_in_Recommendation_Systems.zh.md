#Translated_paper
## **Published in:** 2024 IEEE International Conference on Data Mining (ICDM), 2024

## 2024 IEEE 資料探勘國際會議 (ICDM)

# 一種新穎的影子變數捕捉器，用於解決推薦系統中的選擇偏誤

Qingfeng Chen, Boquan Weiº, Debo Cheng, Jiuyong Li, Lin Liu and Shichao Zhang

Qingfeng Chen, Boquan Weiº, Debo Cheng, Jiuyong Li, Lin Liu and Shichao Zhang

"School of Computer, Electronics and Information, Guangxi University, Nanning, China
bUniSA STEM, University of South Australia, Adelaide, Australia
Guangxi Key Lab of Multi-source Information Mining and Security, Guangxi Normal University, Guilin, China

"廣西大學計算機、電子與資訊學院，南寧，中國
b南澳大利亞大學 STEM 學院，阿德萊德，澳大利亞
廣西多源資訊探勘與安全重點實驗室，廣西師範大學，桂林，中國

**Abstract**-Recommender systems rely on observational data to predict user ratings for unseen items. Since the observational data is typically missing not at random (MNAR), they contain biases, predominantly selection bias, thus models trained on such data are inherently biased. If a shadow variable, which is a variable used instead of user's latent variables that influence both the treatment and the outcome, can be identified, it is possible to build unbiased models for recommender systems. To overcome the challenge of manually identifying valid shadow variables in the data, we propose a Shadow Variable Catcher (SVC), a model designed to learn the latent representation of shadow variables from observational data. By using the learned representation of shadow variables by SVC, we develop the Shadow Debiased Recommender (SDR) method to build an unbiased collaborative filtering model for addressing selection bias in recommender systems. Comprehensive experiments on both synthetic and real-world datasets, have verified the performance of SDR and demonstrated its effectiveness and robustness, and offer new insights into the mitigation of bias in recommender systems.

**摘要**-推薦系統依賴觀測數據來預測使用者對未見過項目的評分。由於觀測數據通常是「非隨機缺失」(MNAR)，因此其中包含偏誤，主要是選擇偏誤，從而導致基於此類數據訓練的模型本質上存在偏誤。如果能夠識別出一個影子變數（一個用來替代影響處理和結果的使用者潛在變數的變數），就有可能為推薦系統建立無偏誤的模型。為了克服手動識別有效影子變數的挑戰，我們提出了一種名為「影子變數捕捉器」(SVC) 的模型，旨在從觀測數據中學習影子變數的潛在表示。通過利用 SVC 學習到的影子變數表示，我們開發了「影子去偏推薦器」(SDR) 方法，以建立一個無偏誤的協同過濾模型，解決推薦系統中的選擇偏誤。在合成和真實世界數據集上的綜合實驗，驗證了 SDR 的性能，並證明了其有效性和穩健性，為減輕推薦系統中的偏誤提供了新的見解。

**Index Terms**-Causal Inference, Recommender Systems, Selection Bias, Shadow Variables

**索引詞**-因果推論、推薦系統、選擇偏誤、影子變數

## I. INTRODUCTION

## I. 緒論

Recommender systems have become indispensable in today's Internet landscape, which significantly enhance user experience by accurately identifying the required information amid a complex network of data [1]-[6]. They are widely used in various domains, including e-commerce [7], social media [3], music [8] and video streaming [9]. In the recommendation domain [10], collaborative filtering models suggest items to users that align with their preferences by analyzing user characteristics, historical interactions, and other relevant data. From the perspective of causal inference, collaborative filtering seeks to estimate the causal effect of system exposure on user feedback. This involves answering the counterfactual question: What kind of feedback would be given if the user were recommended that unseen item? By answering this counterfactual question, we can better understand and improve the effectiveness of recommender systems.

推薦系統已成為當今網路世界中不可或缺的一部分，它們在複雜的數據網絡中準確識別所需信息，從而顯著提升使用者體驗 [1]-[6]。它們被廣泛應用於各種領域，包括電子商務 [7]、社交媒體 [3]、音樂 [8] 和影片串流 [9]。在推薦領域 [10] 中，協同過濾模型通過分析使用者特徵、歷史互動和其他相關數據，向使用者推薦符合其偏好的項目。從因果推論的角度來看，協同過濾旨在估計系統曝光對使用者反饋的因果效應。這涉及到回答一個反事實問題：如果向使用者推薦那個未見過的項目，會得到什麼樣的反饋？通过回答這個反事實問題，我們可以更好地理解和提高推薦系統的有效性。

Understanding and addressing the issue of data being missing not at random (MNAR) is crucial for accurately modeling and mitigating selection bias in recommender systems. Ideally, the data collected for recommender systems would be missing at random (MAR), allowing the model to learn real causal effects directly from historical interactions. However, in the real-world, the recommendation data used for training is usually MNAR. In a recommendation data, users tend to rate items they like and are more likely to rate items that are good or bad [5]. These specific missing mechanisms create the collider bias between exposure and ratings, also known as selection bias [10], [11]. This selection bias causes the model to fail in learning the correct rating distribution, especially if the discrepancy between the training rating distribution and the true rating distribution is significant [12], [13]. For example, in Figure 1, we illustrate the difference in the distribution of ratings under random collection and user selection in the Yahoo!R3 dataset. If the missing mechanism can be correctly captured, selection bias can be effectively addressed [14]. Existing research has frequently overlooked that the training data are MNAR and therefore ignores the effects of selection bias. As a result, these algorithms have been adversely affected by selection bias.

理解和解決「非隨機缺失」(MNAR) 數據問題，對於準確建模和減輕推薦系統中的選擇偏誤至關重要。理想情況下，為推薦系統收集的數據應為「隨機缺失」(MAR)，從而讓模型能直接從歷史互動中學習到真實的因果效應。然而，在現實世界中，用於訓練的推薦數據通常是 MNAR。在推薦數據中，使用者傾向於評價他們喜歡的項目，並且更可能評價好的或壞的項目 [5]。這些特定的缺失機制在曝光和評分之間產生了對撞偏誤，也稱為選擇偏誤 [10], [11]。這種选择偏誤導致模型無法學習到正確的評分分佈，特別是當訓練評分分佈與真實評分分佈之間的差異顯著時 [12], [13]。例如，在圖 1 中，我們展示了在 Yahoo!R3 數據集中，隨機收集和使用者選擇下的評分分佈差異。如果缺失機制能被準確捕捉，選擇偏誤就可以得到有效解決 [14]。現有研究經常忽略訓練數據是 MNAR 的事實，因此忽略了選擇偏誤的影響。結果，這些演算法受到了選擇偏誤的不利影響。

Shadow variables [15] are a special type of covariate characterized by their independence from the missing mechanisms when conditioned on the remaining covariates and outcomes [16]-[18]. Recently, research on leveraging shadow variables to ensure identifiability of models built from MNAR data has gained popularity [19], [20]. Figure 2 shows a simple example for illustrating shadow variables in recommender systems. It is known that user rating preference may cause selection bias, e.g., users tend to prefer rating items they particularly like or particularly dislike [21]. A user's rating preference does not affect the system exposure or the user's rating value, but it constitutes a collider structure, as shown in the causal graph in Figure 2. In this case, user interests is a valid shadow variable. It affects the probability of item exposure and also impacts ratings, but not directly related to the user's rating preferences.

影子變數 [15] 是一種特殊的共變數，其特點是在以其餘共變數和結果為條件時，與缺失機制無關 [16]-[18]。最近，利用影子變數來確保從 MNAR 數據建立的模型的可識別性的研究越來越受歡迎 [19], [20]。圖 2 展示了一個簡單的例子，用以說明推薦系統中的影子變數。眾所周知，使用者評分偏好可能導致選擇偏誤，例如，使用者傾向於評價他們特別喜歡或特別不喜歡的項目 [21]。使用者的評分偏好不影響系統曝光或使用者的評分值，但它構成了一個對撞結構，如圖 2 的因果圖所示。在這種情況下，使用者興趣是一個有效的影子變數。它影響項目曝光的概率，也影響評分，但與使用者的評分偏好沒有直接關係。

Fig. 1. Distributional differences in ratings between (a) user-selected and (b) randomly collected data in the Yahoo!R3 dataset.

圖 1. Yahoo!R3 資料集中 (a) 使用者選擇和 (b) 隨機收集的評分分佈差異。

Fig. 2. An example of a shadow variable represented in a causal graph, where Rating Preferences is a collider.

圖 2. 一個以因果圖表示的影子變數範例，其中「評分偏好」是一個對撞因子。

To address selection bias in recommender systems, we propose a novel multi-loss model, i.e., Shadow Variable Catcher (SVC), to learn the latent representation of shadow variables from the observational data by using deep generative paradigm [6], [22], [23]. Furthermore, we propose a novel framework, i.e., Shadow Debiased Recommendation (SDR for short), to use the learned representation of shadow variables for addressing the selection bias in recommender systems. Subsequently, we implement unbiased training of the recommendation model based on the shadow variables to obtain unbiased estimator under MNAR data. Our main contributions are listed below:

為了解決推薦系統中的選擇偏誤，我們提出了一種新穎的多重損失模型，即影子變數捕捉器 (SVC)，利用深度生成範式 [6], [22], [23] 從觀測數據中學習影子變數的潛在表示。此外，我們提出了一個新穎的框架，即影子去偏推薦 (簡稱 SDR)，利用學習到的影子變數表示來解決推薦系統中的選擇偏誤。隨後，我們基於影子變數實現了推薦模型的無偏誤訓練，以在 MNAR 數據下獲得無偏誤的估計器。我們的主要貢獻如下：

* We propose the SVC model that captures the latent representation of shadow variables from observational data. The captured shadow variables are highly valuable as a special kind of covariate, as well as proxy variables of unmeasured confounders, for both deconfounding and de-selection bias in recommender systems.
* We propose a novel recommendation framework, SDR, which uses the learned representation of shadow variables to mitigate selection bias in collaborative filtering models. To the best of our knowledge, our SDR model is the first attempt at addressing selection bias using shadow variables in recommender system.
* We conduct extensive experiments on one synthetic and three real-world datasets to validate the performance of the proposed method for debiasing.

* 我們提出了 SVC 模型，它能從觀測數據中捕捉影子變數的潛在表示。捕捉到的影子變數作為一種特殊的共變數以及未測量混淆因子的代理變數，對於推薦系統中的去混淆和去選擇偏誤都非常有價值。
* 我們提出了一個新穎的推薦框架 SDR，它利用學習到的影子變數表示來減輕協同過濾模型中的選擇偏誤。據我們所知，我們的 SDR 模型是首次嘗試在推薦系統中使用影子變數來解決選擇偏誤。
* 我們在一個合成數據集和三個真實世界數據集上進行了廣泛的實驗，以驗證所提出的去偏誤方法的性能。

## II. RELATED WORK

## II. 相關研究

In this section, we briefly overview the development of causal inference in recommender systems and review related work that addresses selection bias.

在本節中，我們簡要概述了因果推論在推薦系統中的發展，並回顧了處理選擇偏誤的相關研究。

### A. Debiasing methods in recommendations

### A. 推薦系統中的去偏誤方法

For recommender systems, performance is often affected by various biases due to the difficulty of capturing true relations in biased data. The best way to deal with this problem is to perform randomized experiments such as online A/B testing to collect unbiased data. However, recommendation models are time-sensitive. Running randomized experiments at regular intervals would severely degrade the user experience and impact the system's operation, which is unacceptable. Causal inference [11], [24], a discipline that infers causality from data, has achieved remarkable advances in recent years when used for dealing with bias in recommendations.

對於推薦系統而言，由於在有偏誤的數據中難以捕捉真實關係，其性能常受到各種偏誤的影響。處理此問題的最佳方法是進行隨機實驗，例如線上 A/B 測試，以收集無偏誤的數據。然而，推薦模型具有時間敏感性。定期進行隨機實驗會嚴重降低使用者體驗並影響系統運作，這是不可接受的。因果推論 [11], [24] 是一門從數據中推斷因果關係的學科，近年來在處理推薦中的偏誤方面取得了顯著進展。

There are two fundamental biases [5]: confounding bias caused by unmeasured confounders and selection bias caused by controlled colliders. Leveraging the wealth of research on confounding bias in causal inference, relevant works have proposed numerous methods for addressing confounding bias in recommender systems. The backdoor path-based approaches [25], [26] implement backdoor adjustment to block indirect causal paths. Some methods assume the availability of certain variables within the data, such as instrumental variables (e.g., search logs) or mediators (e.g., click feedback), subsequently performing classic IV-estimation [27], [28] or front door adjustment [29]. Methods learn the representation of unmeasured confounders [30], [31] from proxy variables through generative models. These generative debiasing methods, while capturing the variables implicit in the data to mitigate confounding bias, did not extend the theory and structure further. The study by [32] adopted invariant learning methods to learn users' invariant interests in the presence of unmeasured confounders. The multi-task residual model designed in study [33] also provides a viable solution to unmeasured confounders. In contrast, methods for dealing with selection bias are relatively few and usually require stricter assumptions.

存在兩種基本偏誤 [5]：由未測量混淆因子引起的混淆偏誤，以及由受控對撞因子引起的選擇偏誤。利用因果推論中關於混淆偏誤的豐富研究，相關工作提出了多種方法來解決推薦系統中的混淆偏誤。基於後門路徑的方法 [25], [26] 實作後門調整以阻斷間接因果路徑。一些方法假設數據中存在某些變數，例如工具變數（如搜尋日誌）或中介變數（如點擊反饋），隨後執行經典的 IV 估計 [27], [28] 或前門調整 [29]。有些方法通過生成模型從代理變數中學習未測量混淆因子的表示 [30], [31]。這些生成式去偏誤方法雖然捕捉了數據中隱含的變數以減輕混淆偏誤，但並未進一步擴展其理論和結構。[32] 的研究採用不變學習方法，在存在未測量混淆因子的情況下學習使用者的不變興趣。[33] 研究中設計的多任務殘差模型也為未測量混淆因子提供了一個可行的解決方案。相比之下，處理選擇偏誤的方法相對較少，且通常需要更嚴格的假設。

### B. Methods addressing selection bias in recommendations

### B. 處理推薦中選擇偏誤的方法

Selection bias is prevalent in recommender systems where ratings collected in non-random user interactions inevitably introduce selection bias [21]. The propensity-based method IPS [34] uses inverse probability weighting to adjust the model's training loss, providing an unbiased estimation of the true risk. But, propensity-based methods only achieve unbiased estimation if they have the true propensity. As propensity estimation suffering from a high variance problem [35], model performance is not stable. Subsequent work [36] has further improved propensity estimation based on sensitivity analyses, and the work in [37] also proposed a new policy learning to enhance the propensity model. A study [7] decomposed the direct effect of users on ratings as a way to mitigate the impact of user selection issues on the model.

選擇偏誤在推薦系統中普遍存在，其中在非隨機使用者互動中收集的評分不可避免地會引入選擇偏誤 [21]。基於傾向性的方法 IPS [34] 使用逆機率加權來調整模型的訓練損失，從而提供對真實風險的無偏估計。但是，基於傾向性的方法只有在擁有真實傾向性時才能實現無偏估計。由於傾向性估計存在高變異數問題 [35]，模型性能不穩定。後續工作 [36] 基於敏感度分析進一步改進了傾向性估計，而 [37] 的工作也提出了一種新的策略學習來增強傾向性模型。一項研究 [7] 將使用者對評分的直接影響進行分解，以減輕使用者選擇問題對模型的影響。

One aspect of the study focuses on the non-random characteristics of missing data, generating pseudo-labels for missing data by an imputation model and then training a recommendation model on a complete matrix, e.g., EIB [13], making the matrix close to the ideal uniform matrix by imputation pseudo-labels. The research in [35] proposed the asymmetric training framework to improve the imputation method. However, the imputation model suffers from empirical inaccuracy and cannot guarantee the availability of an unbiased prediction model. Doubly Robust Joint Learning (DR) [38] combines the benefits of IPS and EIB, allowing unbiased models to be trained as long as the imputed errors or propensities are accurate. Stable-DR [39] provides a more stable DR based on cyclical learning. Different from prior paradigms dealing with selection bias, this paper presents a viable approach to mitigate selection bias from a novel perspective.

研究的一個方面集中於缺失數據的非隨機特性，通過一個插補模型為缺失數據生成偽標籤，然後在一個完整的矩陣上訓練推薦模型，例如 EIB [13]，通過插補偽標籤使矩陣接近理想的均勻矩陣。[35] 的研究提出了非對稱訓練框架以改進插補方法。然而，插補模型存在經驗上的不準確性，無法保證無偏預測模型的可用性。雙重穩健聯合學習 (DR) [38] 結合了 IPS 和 EIB 的優點，只要插補誤差或傾向性是準確的，就可以訓練無偏模型。Stable-DR [39] 提供了一種基於循環學習的更穩定的 DR。與以往處理選擇偏誤的範式不同，本文從一個新穎的視角提出了一種可行的減輕選擇偏誤的方法。

## III. PROBLEM FORMULATION

## III. 問題建構

In this section, we introduce the basic notations used in our work and formulate recommendation tasks based on causal graphs. We also use causal graphs [11] to describe shadow variables and address the issues caused by selection bias.

在本節中，我們介紹了我們工作中使用的基本符號，並基於因果圖來建構推薦任務。我們也使用因果圖 [11] 來描述影子變數並解決由選擇偏誤引起的問題。

### A. Notations

### A. 符號

Let U = {u} and I = {i} denote the sets of users and items, respectively. As illustrated in Figure 3, recommendation focuses on the following elements:

令 U = {u} 和 I = {i} 分別表示使用者和項目的集合。如圖 3 所示，推薦主要關注以下元素：

* Unit: The basic unit is user-item pair (u, i).
* Covariate: The explicit features of users and items, denoted as X = {Xu, Xi}, where Xu represents features of the users and X₁ represents features of the items.
* Treatment: O = {ou,i} is the exposure status. Ou,i = 1 when the rating of this unit (u,i) is observed, otherwise Ou,i = 0.
* Outcome: R = {ru,i} is the feedback status. ru,i represents the feedback rating of user u for item i. The ratings ru,i sorted from highest to lowest, form the list of recommendations.
* Selection Indicator & Collider: S = {Sk,u,i | k = 1,2,..., K}, where Sk,u,i ∈ {0,1}, ∀k ∈ K, is selection indicator or collider as shown in Figure 3. When the data is MNAR caused by Sk,u,i, it means that only or most of the data has sk,u,i = 1, while for Sk,u,i = 0, the data is completely missing or minimal. We use S = 1 to indicate the presence of a majority of colliders as 1 in the data, i.e., the data is MNAR. Conversely, S = 0 denotes the unobserved case where colliders as 0.

* 單位：基本單位是使用者-項目對 (u, i)。
* 共變數：使用者和項目的明確特徵，表示為 X = {Xu, Xi}，其中 Xu 代表使用者的特徵，X₁ 代表項目的特徵。
* 處理：O = {ou,i} 是曝光狀態。當此單位 (u,i) 的評分被觀測到時，Ou,i = 1，否則 Ou,i = 0。
* 結果：R = {ru,i} 是回饋狀態。ru,i 代表使用者 u 對項目 i 的回饋評分。評分 ru,i 從高到低排序，形成推薦列表。
* 選擇指標與對撞因子：S = {Sk,u,i | k = 1,2,..., K}，其中 Sk,u,i ∈ {0,1}，∀k ∈ K，是選擇指標或如圖 3 所示的對撞因子。當數據因 Sk,u,i 而成為 MNAR 時，表示只有或大部分數據具有 sk,u,i = 1，而對於 Sk,u,i = 0，數據完全缺失或極少。我們使用 S = 1 來表示數據中存在大多數對撞因子為 1，即數據是 MNAR。反之，S = 0 表示對撞因子為 0 的未觀測情況。

Let P denote the probability mass function for discrete variables and f denote the probability density function for continuous variables. Under the recommendation task, our goal is to train a model to predict the user's rating of an item, represented by the probability mass function P(R = 1 | O, X, S). A higher rating means that the user is more likely to click or buy the item, thereby creating a Top-K recommendation list. Due to the problem of MNAR data, after training we will only get P(R = 1 | O, X, S = 1), implying that some propensity interferes with the causal effect of exposure O on rating R. To eliminate the bias caused by MNAR data, we need to impute the missing data S = 0 to adjust the model training.

令 P 表示離散變數的機率質量函數，f 表示連續變數的機率密度函數。在推薦任務下，我們的目標是訓練一個模型來預測使用者對某個項目的評分，以機率質量函數 P(R = 1 | O, X, S) 表示。較高的評分意味著使用者更有可能點擊或購買該項目，從而產生一個 Top-K 推薦列表。由於 MNAR 數據的問題，訓練後我們只能得到 P(R = 1 | O, X, S = 1)，這意味著某些傾向性干擾了曝光 O 對評分 R 的因果效應。為了消除 MNAR 數據造成的偏誤，我們需要插補缺失的數據 S = 0 來調整模型訓練。

Specifically, we calculate the S average effect based on the conditional probability formula as follows:
P(R = 1 |O, X, S) = Σ P(R = 1 |O, X, S = d)P(S = d|O, X) (1)
The main missing component of Eq. (1) is P(R = 1 | O, X, S = 0). Solving the MNAR problem is equivalent to recovering P(R = 1 | O, X, S = 0) from data.

具體來說，我們根據條件機率公式計算 S 的平均效應如下：
P(R = 1 |O, X, S) = Σ P(R = 1 |O, X, S = d)P(S = d|O, X) (1)
方程式 (1) 中主要缺失的成分是 P(R = 1 | O, X, S = 0)。解決 MNAR 問題等同於從數據中恢復 P(R = 1 | O, X, S = 0)。

### B. Shadow variables

### B. 影子變數

The shadow variable is a special type of covariate, e.g., the variable Z represented in the causal graph in Figure 3, which must satisfy three conditional independence assumptions outlined in Assumption 1.
Assumption 1: A variable Z is a shadow variable if it satisfies (1) Z ⊥ O | X, R, S = 1; (2) Z ⊥ R | X,O, S = 1; and (3) Z ⊥ S | X,O, R [18].
where, ⊥ denotes independent and ̸⊥ denotes dependent. The first two conditional independence assumptions represent the relationships of the shadow variable Z with treament O and outcome R, respectively. From a causal graph perspective, shadow variable Z maintains paths to O and R after the other indirect causal paths have been blocked. The last assumption requires that the shadow variable Z has no effect on the collider S when conditioning on the other observed variables. There is no direct causal path from Z to S in the causal graph, they are conditionally independent.

影子變數是一種特殊的共變數，例如圖 3 因果圖中表示的變數 Z，它必須滿足假設 1 中概述的三個條件獨立性假設。
假設 1：如果變數 Z 滿足 (1) Z ⊥ O | X, R, S = 1；(2) Z ⊥ R | X,O, S = 1；以及 (3) Z ⊥ S | X,O, R [18]，則 Z 是一個影子變數。
其中，⊥ 表示獨立，̸⊥ 表示相依。前兩個條件獨立性假設分別表示影子變數 Z 與處理 O 和結果 R 的關係。從因果圖的角度來看，在其他間接因果路徑被阻斷後，影子變數 Z 仍然維持著到 O 和 R 的路徑。最後一個假設要求在以其他觀測變數為條件時，影子變數 Z 對對撞因子 S 沒有影響。在因果圖中，從 Z 到 S 沒有直接的因果路徑，它們是條件獨立的。

Previous studies [40] have indicated that unbiased models under MNAR data cannot be directly identified without further hypotheses and prior knowledge. However, it is possible to identify true effects if there are shadow variables in the data that correspond to the selection indicator S [19]. Finding convincing shadow variables in datasets collected under the usual operation of recommender systems is challenging. This does not imply that there are no implicit shadow variables lurking behind the observational data. In this paper, we will recover the latent representation of the shadow variables and apply them to calculate the average causal effect, thereby achieving unbiased training with MNAR data.

先前的研究 [40] 指出，在沒有進一步的假設和先驗知識的情況下，無法直接識別 MNAR 數據下的無偏模型。然而，如果數據中存在與選擇指標 S [19] 相對應的影子變數，則有可能識別出真實的效應。在推薦系統常規操作下收集的數據集中尋找有說服力的影子變數是具有挑戰性的。這並不意味著觀測數據背後沒有潛在的影子變數。在本文中，我們將恢復影子變數的潛在表示，並應用它們來計算平均因果效應，從而實現 MNAR 數據的無偏訓練。

## IV. THE PROPOSED SDR METHOD

## IV. 提出的 SDR 方法

In this section, we first introduce our proposed shadow variable catcher for learning the latent representation of the shadow variables. Then, we describe the method for addressing selection bias with shadow variables. Finally, we introduce our proposed shadow-debiased recommendation by training recommendation model using shadow variables.

在本節中，我們首先介紹我們提出的用於學習影子變數潛在表示的影子變數捕捉器。然後，我們描述了使用影子變數解決選擇偏誤的方法。最後，我們介紹了我們提出的通过使用影子變數訓練推薦模型的影子去偏誤推薦方法。

### A. Shadow variable catcher (SVC)

### A. 影子變數捕捉器 (SVC)

In this work, we aim to recover the latent representation of shadow variables via the generative models [22], [41]. We propose a novel generative model (SVC), to learn the latent representation of shadow variables from data directly.
In our SVC model, we employ the Variational Autoencoder (VAE) [22] as our generative model to learn and generate the latent representation of the shadow variables. Note that we impose the three conditions outlined in Assumption 1 on the measured data to learn shadow variable representation for each user that satisfy these conditions as robustly as possible.

在這項工作中，我們的目標是通過生成模型 [22], [41] 來恢復影子變數的潛在表示。我們提出了一種新穎的生成模型 (SVC)，直接從數據中學習影子變數的潛在表示。
在我們的 SVC 模型中，我們採用變分自動編碼器 (VAE) [22] 作為我們的生成模型 ，以學習和生成影子變數的潛在表示。請注意，我們將假設 1 中概述的三個條件強加於測量數據，以便為每個使用者學習滿足這些條件的影子變數表示，並使其盡可能穩健。

SVC takes user exposure Ou and user features Xu as inputs. The Encoder module in our SVC outputs the mean and variance, followed by a reparameterization trick to sample the generative variable. To ensure that there is sufficient information to learn the latent representation Z of shadow variables, we set the prior distribution of the latent representation, which prevents overfitting features in the unordered latent space. Referring to the encoder of the iVAE [42], user features Xu serves as an auxiliary variable in the Encoder, providing a source for the prior distribution of the latent variables, thereby ensuring the identifiability of the latent representation Z. That is, the prior distribution of Z is modeled as Po(Ζ | Xu).

SVC 以使用者曝光 Ou 和使用者特徵 Xu 作為輸入。我們 SVC 中的編碼器模塊輸出平均值和變異數，然後通過重參數化技巧對生成變數進行採樣。為確保有足夠的資訊來學習影子變數的潛在表示 Z，我們設定了潛在表示的先驗分佈，以防止在無序的潛在空間中過度擬合特徵。參考 iVAE [42] 的編碼器，使用者特徵 Xu 在編碼器中作為輔助變數，為潛在變數的先驗分佈提供來源，從而確保潛在表示 Z 的可識別性。也就是說，Z 的先驗分佈被建模為 Po(Ζ | Xu)。

We assume that the prior distribution Po(Z | Xu) belongs to the Gaussian location-scale family. The distribution q(Z | Xu, Ou) is sampled from the approximate posterior. We constrain this posterior distribution q¢(Z | Xu, Ou) with Kullback-Leibler divergence of two Gaussian distributions:
LKL = -KL(q$(Z|Xu, Ou)||Po(Z | Xu)) = – KL(N(μq(Xu, Ou), σ₄(Xu, Ou))||N(μp(Xu),σ₂(Xu))) (2)
where μq, ση, μp, and of are parameters modeled by four different Multilayer Perceptron (MLP) models. Next, we design three constraint terms corresponding to the three conditions in Assumption 1 to generate the latent representation Z.

我們假設先驗分佈 Po(Z | Xu) 屬於高斯位置-尺度族。分佈 q(Z | Xu, Ou) 從近似後驗中採樣。我們用兩個高斯分佈的 Kullback-Leibler 散度來約束這個後驗分佈 q¢(Z | Xu, Ou)：
LKL = -KL(q$(Z|Xu, Ou)||Po(Z | Xu)) = – KL(N(μq(Xu, Ou), σ₄(Xu, Ou))||N(μp(Xu),σ₂(Xu))) (2)
其中 μq、ση、μp 和 of 是由四個不同的多層感知器 (MLP) 模型建模的參數。接下來，我們設計了三個對應於假設 1 中三個條件的約束項，以生成潛在表示 Z。

Loss function LRec for condition 1: Z ⊥ O | X, R, S = 1. The goal of condition 1 is to ensure that Z has enough influence over O. Since the latent variables are extracted from the exposure O, condition 1 indicates whether Z has sufficient capacity to reconstruct O. SVC models the Decoder using an MLP and implements this constraint based on binary cross entropy (BCE) loss. The Decoder outputs an approximation of Ou, denoted as Õu. The reconstruction loss can be expressed as LRec = BCE(Ou, Õu).

條件 1 的損失函數 LRec：Z ⊥ O | X, R, S = 1。條件 1 的目標是確保 Z 對 O 有足夠的影響力。由於潛在變數是從曝光 O 中提取的，條件 1 表示 Z 是否有足夠的能力來重建 O。SVC 使用 MLP 來建模解碼器，並基於二元交叉熵 (BCE) 損失來實現此約束。解碼器輸出 Ou 的近似值，表示為 Õu。重建損失可以表示為 LRec = BCE(Ou, Õu)。

Loss function Lpred for condition 2: Z ⊥ R | X,O, S = 1. Condition 2 aims to verify the effect of Z on the outcome R. This is equivalent to the predictive ability of Z for R. It is consistent with our final purpose for recommendation task, and therefore it is the primary loss. We adopt an MLP as the Predictor of the outcome R. The output of the predictor is Ru, then, the prediction loss based on BCE loss is LPred = BCE(Ru, Ru).

條件 2 的損失函數 Lpred：Z ⊥ R | X,O, S = 1。條件 2 旨在驗證 Z 對結果 R 的影響。這相當於 Z 對 R 的預測能力。這與我們推薦任務的最終目的一致，因此是主要的損失。我們採用 MLP 作為結果 R 的預測器。預測器的輸出是 Ru，那麼，基於 BCE 損失的預測損失是 LPred = BCE(Ru, Ru)。

Loss function LTest for condition 3: Z ⊥ S | X,O,R. Condition 3 requires that the representation Z and Selection Indicator S be conditionally independent, which can not be tested directly because we miss the data for the S = 0 part. According to a previous study [15], we can guarantee condition 3 by the following reasonable test, which is shown as follows:
E [S/Q(R) | Z] = 1 (3)
where Q is an arbitrary function, which can be modelled by parametric or non-parametric estimation. In case of S = 0, S/Q(R) is constant at 0.

條件 3 的損失函數 LTest：Z ⊥ S | X,O,R。條件 3 要求表示 Z 和選擇指標 S 是條件獨立的，這無法直接測試，因為我們缺少 S = 0 部分的數據。根據先前的一項研究 [15]，我們可以通過以下合理的測試來保證條件 3，如下所示：
E [S/Q(R) | Z] = 1 (3)
其中 Q 是一個任意函數，可以通過參數或非參數估計來建模。在 S = 0 的情況下，S/Q(R) 在 0 處為常數。

In accordance with Theorem 1, the feasibility of Condition 3 can be evaluated by determining whether there exists a solution Q∈ (0,1] to Eq. (3) for R under the condition Z. Satisfying such a solution facilitates the construction of a data distribution in which the shadow variable Z and the collider S adhere to Condition 3.

根據定理 1，條件 3 的可行性可以通過確定在條件 Z 下是否存在 R 的方程式 (3) 的解 Q∈ (0,1] 來評估。滿足這樣的解有助於構建一個數據分佈，其中影子變數 Z 和對撞因子 S 遵守條件 3。

In practice, we use non-parametric estimation to implement the test in Theorem 1. The Estimator module employs an MLP to fit the function 1/Q(R), and is equipped with L2 regularization to mitigate the ill-posed problem of non-parametric estimation. According to the problem formulation in Section III, the observed data are virtually missing for S = 0, i.e. O can be taken as an approximate substitution for S, which is a typical treatment in previous research on MNAR [34], [36], [43]. If the output of Estimator is Q, denote D = Q×Ou. Then, based on the BCE loss, Eq. (3) having a solution equivalent to minimising LTest = BCE(D,Ou).

在實踐中，我們使用非參數估計來實現定理 1 中的檢定。估計器模塊採用一個 MLP 來擬合函數 1/Q(R)，並配備 L2 正規化以減輕非參數估計的不適定問題。根據第三節中的問題建構，觀測數據在 S = 0 時幾乎是缺失的，即 O 可以作為 S 的近似替代，這是先前關於 MNAR 的研究中的典型處理方法 [34], [36], [43]。如果估計器的輸出是 Q，記 D = Q×Ou。然後，基於 BCE 損失，方程式 (3) 有一個等價於最小化 LTest = BCE(D,Ou) 的解。

Theorem 1: Suppose Z ⊥ O | X, R, S = 1, Z ⊥ R | X,O, S = 1, and the overlap assumption¹ hold. Then Z ⊥ S | X, O, R can be rejected if and only if there exists no solution Q to Eq. (3) that belongs to (0,1] [15].

定理 1：假設 Z ⊥ O | X, R, S = 1，Z ⊥ R | X,O, S = 1，且重疊假設¹ 成立。那麼 Z ⊥ S | X, O, R 可以被拒絕，若且唯若方程式 (3) 不存在屬於 (0,1] 的解 Q [15]。

The overall structure of the SVC model is shown in the left part of Figure 4. Valid shadow variables (e.g., user interests) can be trained out as close as possible with the above three losses. As all three losses are calculated with BCE loss, the final loss of our SVC is defined as:
LSVC = LRec + LPred + LTest + LKL (4)

SVC 模型的整體結構如圖 4 的左側部分所示。有效的影子變數（例如，使用者興趣）可以通過上述三個損失函數盡可能地被訓練出來。由於這三個損失都是用 BCE 損失計算的，我們 SVC 的最終損失定義為：
LSVC = LRec + LPred + LTest + LKL (4)

The constraint losses in Lsve are inherently collaborative, not conflicting. Therefore, they can be harmoniously co-optimized. Finally, we select the optimal weights that achieve the best performance on LPred and generate the latent representation Z of shadow variables for each user. Once we obtain Z, we can use it as a shadow variable to remove the selection bias caused by data with MNAR.

Lsve 中的約束損失本質上是協同的，而非衝突的。因此，它們可以和諧地共同優化。最後，我們選擇在 LPred 上達到最佳性能的最佳權重，並為每個使用者生成影子變數的潛在表示 Z。一旦我們獲得 Z，我們就可以將其用作影子變數，以消除由 MNAR 數據引起的選擇偏誤。

### B. Shadow variable debiased recommendation

### B. 影子變數去偏誤推薦

In this section, we introduce our proposed Shadow Debiased Recommendation (SDR). The framework of SDR is depicted in the right part of Figure 4.
The Odds Ratio (OR) is commonly employed to quantify the influence of a factor on an outcome [44]. Consequently, we utilize OR to encode the disparity between the distributions (O, X, Z, R, S = 1) and (O, X, Z, R, S = 0). Since the shadow variable Z is independent of the selection indicator S, the OR of the selection indicator S to R is calculated as follows:
OR(R|O, X, Z) = OR(R|O, X) = P(S = 0 | R,O,X)P(S = 1 | R = 0, O, X) / P(S = 0 | R = 0, O, X) P(S = 1 | R,O,X) (5)

在本節中，我們介紹我們提出的影子去偏推薦 (SDR)。SDR 的框架如圖 4 的右側部分所示。
勝算比 (OR) 通常用於量化一個因素對結果的影響 [44]。因此，我們利用 OR 來編碼分佈 (O, X, Z, R, S = 1) 和 (O, X, Z, R, S = 0) 之間的差異。由於影子變數 Z 與選擇指標 S 無關，選擇指標 S 對 R 的 OR 計算如下：
OR(R|O, X, Z) = OR(R|O, X) = P(S = 0 | R,O,X)P(S = 1 | R = 0, O, X) / P(S = 0 | R = 0, O, X) P(S = 1 | R,O,X) (5)

where R = 0 is a reference value that can be replaced by any value within the range of R values. When R = 0 is used as a reference value, it is evident that OR(R = 0 | O, X) = 1. It must be ensured that OR(R | O,X) > 0 and E[OR(R | O,X) | X, Z, S = 1] < +∞. While OR(R = 0 | O,X) = OR(R = 1 | O,X) = 1, it means that there is no disparity between distributions (O, X, Z, R, S = 1) and (O, X, Z, R, S = 0). With the aid of shadow variables, we have the proposition of identification for recovering the missing distribution P(R = 1 | O, X, Z, S = 0) according to previous works [18], [19]. We do not present this proposition here due to page limitations.

其中 R = 0 是一個參考值，可以被 R 值範圍內的任何值替換。當 R = 0 作為參考值時，很明顯 OR(R = 0 | O, X) = 1。必須確保 OR(R | O,X) > 0 且 E[OR(R | O,X) | X, Z, S = 1] < +∞。當 OR(R = 0 | O,X) = OR(R = 1 | O,X) = 1 時，表示分佈 (O, X, Z, R, S = 1) 和 (O, X, Z, R, S = 0) 之間沒有差異。借助影子變數，根據先前的研究 [18], [19]，我們有恢復缺失分佈 P(R = 1 | O, X, Z, S = 0) 的識別命題。由於頁面限制，我們在此不呈現此命題。

We are able to recover the missing distributions P(R = 1 | O, X, Z, S = 0) by using shadow variables with the following steps. Firstly we calculate OR by:
OR(R|O,X) = OR(RO,X) / E[OR(R|O,X) | O, X, S = 1] (6)
The distributional disparity OR can be captured by the shadow variables Z as:
E[OR(R|O,X)|O, X, Z, S = 1] = f(Z|O, X, S = 0) / f(Z|O, X, S = 1) (7)

我們能夠通過使用影子變數，通過以下步驟恢復缺失的分佈 P(R = 1 | O, X, Z, S = 0)。首先，我們計算 OR：
OR(R|O,X) = OR(RO,X) / E[OR(R|O,X) | O, X, S = 1] (6)
分佈差異 OR 可以被影子變數 Z 捕捉為：
E[OR(R|O,X)|O, X, Z, S = 1] = f(Z|O, X, S = 0) / f(Z|O, X, S = 1) (7)

This equation is a Fredholm integral equation of the first kind, with OR to be solved for. Using OR, we have the recovery equation for the missing distribution P(R = 1 | O, X, S = 0) as follows:
P(R = 1 | O, X, Z, S = 0) = OR(R = 1 | O,X)P(R = 1 | O, X, Z, S = 1) / E[OR(R|O,X) | O, X, Z, S = 1] (8)

這個方程式是第一類 Fredholm 積分方程式，其中 OR 是待解的。利用 OR，我們得到缺失分佈 P(R = 1 | O, X, S = 0) 的恢復方程式如下：
P(R = 1 | O, X, Z, S = 0) = OR(R = 1 | O,X)P(R = 1 | O, X, Z, S = 1) / E[OR(R|O,X) | O, X, Z, S = 1] (8)

Thus, we have the following theorem for identification of P(R = 1 | O, X, Z, S) using shadow variables Z.
Theorem 2 (Identification of P(R = 1 | O, X, Z, S) [18]): Under Assumption 1 and the completeness condition of P(R = 1 | O,X, Z,S = 1)², Eq. (6) has a unique solution. Thus, OR(R|O,X) and P(R = 1 |O, X, Z, S) can be identified.

因此，我們有以下使用影子變數 Z 識別 P(R = 1 | O, X, Z, S) 的定理。
定理 2 (P(R = 1 | O, X, Z, S) 的識別 [18])：在假設 1 和 P(R = 1 | O,X, Z,S = 1)² 的完備性條件下，方程式 (6) 有唯一解。因此，OR(R|O,X) 和 P(R = 1 |O, X, Z, S) 可以被識別。

Theorem 2 ensures non-parametric identification under MNAR data through shadow variable. The theory involves only the proposed completeness conditions and observational data, and it can be justified without extra model assumptions on the missing data distribution.

定理 2 確保了在 MNAR 數據下通過影子變數進行非參數識別。該理論僅涉及所提出的完備性條件和觀測數據，並且可以在沒有關於缺失數據分佈的額外模型假設的情況下得到證明。

Identification Under Collaborative Filtering. The purpose of collaborative filtering (CF) is to provide a pre-ranking with a lightweight model, with an emphasis on ranking. Estimating precise P(R = 1 | O,X,Z,S = 1) will undermine the performance of the model. To recover the missing distributions under CF, it is necessary to adjust the identification process. Following the above process, we approximate the probability distribution of S by using the probability distribution of O as a substitute. Based on the approximate substitution P(S | R,O,X) ≈ P(O | R, X), we adjust Eq. (5) as follows:
OR(R|O,X) = P(O = 0 | R, X)P(O = 1 | R = 0, X) / P(O = 0 | R = 0, X)P(O = 1 | R, X) (9)

協同過濾下的識別。協同過濾 (CF) 的目的是提供一個輕量級模型的預排序，重點在於排序。估計精確的 P(R = 1 | O,X,Z,S = 1) 會損害模型的性能。為了在 CF 下恢復缺失的分佈，有必要調整識別過程。遵循上述過程，我們使用 O 的機率分佈作為 S 的機率分佈的近似替代。基於近似替代 P(S | R,O,X) ≈ P(O | R, X)，我們將方程式 (5) 調整如下：
OR(R|O,X) = P(O = 0 | R, X)P(O = 1 | R = 0, X) / P(O = 0 | R = 0, X)P(O = 1 | R, X) (9)

where P(O | R, X) can be accessed by either Naive Bayes or Logistic regression [34]. Eq. (9) allows us to estimate the distributional disparity OR from data. Hereafter, based on Eqs. (6), (7), (8) and approximate substitutions, we have:
P(R = 1 |O, X, Z, S = 0) = ... = f(Z | X,O = 1) OR(R = 1 | O,X)P(R = 1 | O, X, Z, S = 1) / f(Z | X, O = 0) (10)

其中 P(O | R, X) 可以通過樸素貝葉斯或邏輯回歸 [34] 來獲取。方程式 (9) 允許我們從數據中估計分佈差異 OR。此後，基於方程式 (6), (7), (8) 和近似替換，我們有：
P(R = 1 |O, X, Z, S = 0) = ... = f(Z | X,O = 1) OR(R = 1 | O,X)P(R = 1 | O, X, Z, S = 1) / f(Z | X, O = 0) (10)

Therefore, the last step is the approximate substitution of O for S. Eq. (10) is designed to retain as much information as possible, thereby facilitating us to achieve unbiased training of CF model. In our SDR method, we use the learned Z by SVC to train recommendation models without selection bias based on the recovery Eq. (10) and the average causal effect Eq. (1). Distribution P(R = 1 | O, X, Z, S = 1) can be modelled with any collaborative filtering model.

因此，最後一步是 O 對 S 的近似替換。方程式 (10) 旨在盡可能多地保留信息，從而有助於我們實現 CF 模型的無偏訓練。在我們的 SDR 方法中，我們使用 SVC 學習到的 Z，基於恢復方程式 (10) 和平均因果效應方程式 (1) 來訓練沒有選擇偏誤的推薦模型。分佈 P(R = 1 | O, X, Z, S = 1) 可以用任何協同過濾模型來建模。

We use Matrix Factorization (MF) as the backbone for our SDR method, modelling based on a simple additive model fp(R = 1 | O, X, Z, S = 1) = f₁(u,i) + f₂(zu, i). According to the proposition, for the unbiased model fy(R = 1 | O, X, Z, S), we have:
f(R = 1 |O, X, Z, S) = ... = P(O = 1 | X) f₄(R = 1 | O, X, Z, S = 1) + α· P(O = 0 | X) f₄(R = 1 | O, X, Z, S = 1) (11)
where a = OR(R=1|O,X)f(Z|X,O=0)/f(Z|X,O=1) and P(O = 1 | X), P(O = 0 | X), OR(R = 1 | O, X) can be estimated from the dataset. The ratio between f(Z | X, O = 1) and f(Z | X, O = 0) can be estimated by any available density estimation model. We applied Real NVP [45] to estimate the discrepancy between the distributions under O = 1 and O = 0. Specifically, Real NVP models h₁(Z) and ho(Z) are respectively trained by optimizing the following negative log-likelihood functions:
Lz₁ = -log(h1(Zi)), Lzo = Σ -log(ho(Zi)) (12)

我們使用矩陣分解 (MF) 作為我們 SDR 方法的主幹，基於一個簡單的加法模型 fp(R = 1 | O, X, Z, S = 1) = f₁(u,i) + f₂(zu, i) 進行建模。根據命題，對於無偏模型 fy(R = 1 | O, X, Z, S)，我們有：
f(R = 1 |O, X, Z, S) = ... = P(O = 1 | X) f₄(R = 1 | O, X, Z, S = 1) + α· P(O = 0 | X) f₄(R = 1 | O, X, Z, S = 1) (11)
其中 a = OR(R=1|O,X)f(Z|X,O=0)/f(Z|X,O=1) 且 P(O = 1 | X)、P(O = 0 | X)、OR(R = 1 | O, X) 可以從數據集中估計。f(Z | X, O = 1) 和 f(Z | X, O = 0) 之間的比例可以通過任何可用的密度估計模型來估計。我們應用 Real NVP [45] 來估計 O = 1 和 O = 0 下分佈之間的差異。具體來說，Real NVP 模型 h₁(Z) 和 ho(Z) 分別通過優化以下負對數概似函數進行訓練：
Lz₁ = -log(h1(Zi)), Lzo = Σ -log(ho(Zi)) (12)

To ensure the efficiency of the recommender system, we aim to train the unbiased model f(R = 1 | O, X, Z, S) directly to reduce inference time. For this purpose, we adjusted the training loss to be:
LSDR = Σφ(f(R = 1 | O, X, Z, S = 1)/ā, ru,i) (13)
where ã = P(O = 1 | X) + α· P(O = 0 | X), and $() is a pointwise loss for recommendation. We use mean squared error (MSE) loss for training our SDR. The process of our proposed SDR is illustrated in the Algorithm 2. The debiasing technique in this work is done at training. Compared to Backbone MF, SDR only requires additional calculation of low-dimensional shadow variables during model inference, which is friendly to real-time recommendations.

為了確保推薦系統的效率，我們的目標是直接訓練無偏模型 f(R = 1 | O, X, Z, S) 以減少推論時間。為此，我們將訓練損失調整為：
LSDR = Σφ(f(R = 1 | O, X, Z, S = 1)/ā, ru,i) (13)
其中 ã = P(O = 1 | X) + α· P(O = 0 | X)，且 $() 是推薦的逐點損失。我們使用均方誤差 (MSE) 損失來訓練我們的 SDR。我們提出的 SDR 過程如演算法 2 所示。這項工作中的去偏誤技術是在訓練時完成的。與主幹 MF 相比，SDR 在模型推論期間僅需要額外計算低維影子變數，這對即時推薦很友好。

Limitations. The validity of our SDR rests on the assumption that shadow variables about collision variables can be captured from user interactions U and user features X. If certain collision variables do not meet this assumption, the selection bias they introduce may persist, potentially undermining the model's accuracy.

限制。我們的 SDR 的有效性取決於一個假設，即關於碰撞變數的影子變數可以從使用者互動 U 和使用者特徵 X 中捕捉。如果某些碰撞變數不符合此假設，它們引入的選擇偏誤可能會持續存在，從而可能損害模型的準確性。

## V. EXPERIMENTS

## V. 實驗

In this section, we evaluate the effectiveness and robustness of our SDR³ method from three perspectives.
* RQ1: Does our proposed SDR model outperform existing debiased models on real-world datasets?
* RQ2: How does the proposed model perform under different strengths of selection bias?
* RQ3: What concrete effect does each component contribute?

在本節中，我們從三個角度評估我們的 SDR³ 方法的有效性和穩健性。
* RQ1：我們提出的 SDR 模型在真實世界數據集上的表現是否優於現有的去偏誤模型？
* RQ2：在不同強度的選擇偏誤下，所提出的模型表現如何？
* RQ3：每個組件具體貢獻了什麼效果？

### A. Experimental Settings

### A. 實驗設定

a) Three Real-world Datasets: To compare the debiasing ability of the models in recommender systems, we conduct experiments on three real-world datasets: Coat, Yahoo!R3, and KuaiRand. The details of the datasets are summarised in Table I. Each dataset contains biased training data collected from operating recommender systems and unbiased test data collected from randomized experiments. The unbiased data minimize the introduction of biases such as confounding bias and selection bias, allowing us to test the effectiveness of model debiasing. Coat and Yahoo!R3 ratings range from 1 to 5 stars, while KuaiRand consists of positive or negative samples defined by the signal “IsClick”. Based on the design of previous work [31], [32], [36], [46], Coat and Yahoo!R3 regard ratings ≥ 4 as positive feedback, otherwise negative. We use all biased data as the training set, 30% unbiased data as the validation set, and 70% unbiased data as the test set. We adopt Naive Bayes, drawing 5% of the total unbiased dataset from the validation set for parameter estimation.

a) 三個真實世界數據集：為了比較推薦系統中模型的去偏誤能力，我們在三個真實世界數據集上進行了實驗：Coat、Yahoo!R3 和 KuaiRand。數據集的詳細資訊總結在表 I 中。每個數據集都包含從運營中的推薦系統收集的有偏誤的訓練數據，以及從隨機實驗中收集的無偏誤的測試數據。無偏誤數據最大限度地減少了諸如混淆偏誤和選擇偏誤等偏誤的引入，使我們能夠測試模型去偏誤的有效性。Coat 和 Yahoo!R3 的評分範圍為 1 到 5 星，而 KuaiRand 由信號“IsClick”定義的正樣本或負樣本組成。根據先前工作 [31], [32], [36], [46] 的設計，Coat 和 Yahoo!R3 將評分 ≥ 4 視為正回饋，否則為負回饋。我們使用所有有偏誤的數據作為訓練集，30% 的無偏誤數據作為驗證集，70% 的無偏誤數據作為測試集。我們採用樸素貝葉斯，從驗證集中抽取 5% 的總無偏誤數據集用於參數估計。

b) Synthetic Data Generation: To test whether the model captures users' real interests under MNAR data, we generative a synthetic dataset in line with causal Figure 3. The synthetic dataset consists of 1,200 users and 400 items. For each user, there is a one-dimensional explicit feature X ∈ {1,2,3,4,5}. Each value of the feature represents a unique influence. To test the ability of hidden variable learning, we add shadow variables to the synthetic data, which exist as implied confounding variables. Shadow variables are influenced by the explicit feature, and the conditional distribution of Z follows:
Zk | Χ ~ Ν(μκ(Χ),σ(X)), k ∈ {1,2} (14)
where Zk represents the k-th dimension of Z. For the exposure matrix O, its generation follows:
O | Z, X ~ Bernoulli(c(Z, X)) (15)
where Xe is the effect of X, sampled from a Gaussian distribution, and β is a hyper-parameter that controls the sparsity of the exposure matrix. e2x1 and elxl are randomly generated item-wise embedding vector. The matrices M2×2 and M2x1 are samples from a uniform distribution representing association patterns. For each user-item pair, the true rating rui = I(exei + Zuezi + XuCxi + Eui). The function I is a normalization function that maps ratings to integer ratings from 1 to 5. Eui is an i.i.d. random noise, making the rating close to the real world scenarios.

b) 合成數據生成：為了測試模型在 MNAR 數據下是否能捕捉到使用者的真實興趣，我們根據因果圖 3 生成了一個合成數據集。該合成數據集包含 1,200 名使用者和 400 個項目。對於每個使用者，都有一個一維的顯式特徵 X ∈ {1,2,3,4,5}。特徵的每個值代表一種獨特的影響。為了測試隱變數學習的能力，我們在合成數據中加入了影子變數，這些變數作為隱含的混淆變數存在。影子變數受顯式特徵的影響，Z 的條件分佈如下：
Zk | Χ ~ Ν(μκ(Χ),σ(X)), k ∈ {1,2} (14)
其中 Zk 代表 Z 的第 k 維。對於曝光矩陣 O，其生成遵循：
O | Z, X ~ Bernoulli(c(Z, X)) (15)
其中 Xe 是 X 的效應，從高斯分佈中採樣，β 是一個控制曝光矩陣稀疏性的超參數。e2x1 和 elxl 是隨機生成的逐項嵌入向量。矩陣 M2×2 和 M2x1 是代表關聯模式的均勻分佈樣本。對於每個使用者-項目對，真實評分 rui = I(exei + Zuezi + XuCxi + Eui)。函數 I 是一個將評分映射到 1 到 5 整數評分的標準化函數。Eui 是一個獨立同分佈的隨機噪聲，使評分接近真實世界場景。

Next, we construct the MNAR data caused by the user's selection. We impose a collision variable S, which is affected by the covariate X but not by the shadow variable Z. After interference by selection bias, the exposure matrix Os follows:
O = SO
S|X ~ Bernoulli(sigmoid(Se + X)) (16)
The smaller the collider S, the lower the probability of being selected by user, i.e., O has minimal S = 0 data. For the true distribution of tests, P(rui) = 0.5 + γ · (3 – rui)/4 where γε [0, 1]. γ is a hyper-parameter that represents the strength of selection bias. When y = 0, the training and test sets have the same distribution. The larger the y, the greater the discrepancy between the two distributions.

接下來，我們建構由使用者選擇引起的 MNAR 數據。我們強加一個碰撞變數 S，它受到共變數 X 的影響，但不受影子變數 Z 的影響。經過選擇偏誤的干擾後，曝光矩陣 Os 遵循：
O = SO
S|X ~ Bernoulli(sigmoid(Se + X)) (16)
對撞因子 S 越小，被使用者選擇的機率就越低，即 O 的 S = 0 數據最少。對於測試的真實分佈，P(rui) = 0.5 + γ · (3 – rui)/4，其中 γε [0, 1]。γ 是一個代表選擇偏誤強度的超參數。當 y = 0 時，訓練集和測試集具有相同的分佈。y 越大，兩個分佈之間的差異就越大。

c) Evaluation Metrics: We use two classic metrics in recommendation: NDCG@K and Recall@K. Recall@K measures whether items are correctly included in the Top-K recommended list, while NDCG@K focuses on whether the Top-K list is sorted correctly. In the experiment, the mean and variance with 10 different random seeds at K = 5 are reported. All experiments were conducted in the following hardware environments: 12th Gen Intel(R) Core(TM) i5-12400F, with one GeForce RTX 4060 GPU.

c) 評估指標：我們在推薦中使用兩個經典指標：NDCG@K 和 Recall@K。Recall@K 衡量項目是否被正確地包含在 Top-K 推薦列表中，而 NDCG@K 則關注 Top-K 列表是否被正確排序。在實驗中，我們報告了在 K = 5 時，使用 10 個不同隨機種子的平均值和變異數。所有實驗都在以下硬體環境中進行：第 12 代 Intel(R) Core(TM) i5-12400F，配備一張 GeForce RTX 4060 GPU。

d) Baselines: We compare our method with existing advanced methods for both de-selection bias and hidden confounding learning, incouding (1) MF [47]. MF is a classic recommendation model that is widely used in collaborative filtering. We use MF as a backbone to test the improvement of all debiasing methods. (2) IPS [34]. Inverse Propensity Score (IPS) is a fundamental propensity-based approach to dealing with selection bias. This approach adjusts training loss using a propensity score to alleviate selection bias. (3) DR [38]. DR is a joint learning method for de-selection bias that skilfully combines propensity-based and imputation-based methods to improve the robustness for addressing selection biases. (4) RD-IPS [36]. Robust Deconfounder (RD) is a modification of the propensity-based approach. Based on sensitivity analysis, it introduce an uncertainty set to estimate the propensity. We take IPS version to test the effect of the propensity modifications. (5) InvPref [32]. InvPref applies invariant learning methods [48] to discover the user's invariant preferences from data containing unmeasured confounding biases. We consider this as a method of hidden confounding learning. (6) DeepDCF-MF [30]. Deep Deconfounder (DeepDCF) is a method of hidden confounding learning. The method learns confounding applying VAE and utilises user features to reduce the variance of the model. We select the MF version to ensure backbone consistency. (7) IDCF [31]. IDCF is an improvement of hidden confounding learning. This approach learned hidden confounding with IVAE, which theoretically guarantees the identifiability of the hidden confounding, enhanced ability to learn unmeasured confounding.

d) 基準方法：我們將我們的方法與現有的用於去選擇偏誤和隱藏混淆學習的先進方法進行比較，包括 (1) MF [47]。MF 是一種廣泛用於協同過濾的經典推薦模型。我們使用 MF 作為主幹來測試所有去偏誤方法的改進效果。(2) IPS [34]。逆傾向分數 (IPS) 是一種處理選擇偏誤的基本的基於傾向性的方法。該方法使用傾向性分數調整訓練損失以減輕選擇偏誤。(3) DR [38]。DR 是一種用於去選擇偏誤的聯合學習方法，它巧妙地結合了基於傾向性和基於插補的方法，以提高處理選擇偏誤的穩健性。(4) RD-IPS [36]。穩健去混淆器 (RD) 是對基於傾向性方法的修改。基於敏感性分析，它引入了一個不確定性集來估計傾向性。我們採用 IPS 版本來測試傾向性修改的效果。(5) InvPref [32]。InvPref 應用不變學習方法 [48] 從包含未測量混淆偏誤的數據中發現使用者的不變偏好。我們將此視為一種隱藏混淆學習的方法。(6) DeepDCF-MF [30]。深度去混淆器 (DeepDCF) 是一種隱藏混淆學習的方法。該方法應用 VAE 學習混淆，並利用使用者特徵來減少模型的變異數。我們選擇 MF 版本以確保主幹的一致性。(7) IDCF [31]。IDCF 是隱藏混淆學習的一種改進。該方法利用 IVAE 學習隱藏混淆，從理論上保證了隱藏混淆的可識別性，增強了學習未測量混淆的能力。

e) Hyper-parameter Search: We use grid search to select hyper-parameters according to their performance on the validation set. For the larger dataset KuaiRand, we use a smaller learning rate and a larger embedding space. The range of the grid search is given in Table II. For Hyper-parameters that are unique to the model which are available in the code.

e) 超參數搜索：我們使用網格搜索根據驗證集上的性能來選擇超參數。對於較大的數據集 KuaiRand，我們使用較小的學習率和較大的嵌入空間。網格搜索的範圍如表 II 所示。對於模型獨有的、在代碼中可用的超參數。

### B. Performance Comparison (RQ1)

### B. 性能比較 (RQ1)

The experimental results on the three real-world datasets are reported in Table III and we have the following consequences:
* Our proposed SDR model achieves the best prediction accuracy on all datasets with excellent mean and variance. All reported p-values indicate statistical significance, except for Coat's Recall@5 metrics, as it is a small dataset and cannot delicately test the model. This result demonstrates the effectiveness of learning shadow variables and unbiased training for debiasing, which we will further validate in subsequent experiments.
* IPS, DB, and RD-IPS were designed to address selection bias but did not consider the impact of confounding bias, resulting in suboptimal performance. In contrast, SDR captured shadow variables are used as proxies for hidden confounding variables to mitigate confounding bias, in addition to addressing selection bias.
* In the smaller dataset, Coat, SDR performs similarly to IDCF and DCF, but as the size of the dataset increases, our model shows superior performance, indicating that SDR fully exploits the information in the dataset to capture the real interests of users.

在三個真實世界數據集上的實驗結果報告在表 III 中，我們得出以下結論：
* 我們提出的 SDR 模型在所有數據集上都取得了最佳的預測準確性，並且具有出色的平均值和變異數。所有報告的 p 值都顯示出統計顯著性，除了 Coat 的 Recall@5 指標，因為它是一個小型數據集，無法對模型進行精細的測試。這個結果證明了學習影子變數和無偏訓練對去偏誤的有效性，我們將在後續實驗中進一步驗證。
* IPS、DB 和 RD-IPS 被設計用來解決選擇偏誤，但沒有考慮混淆偏誤的影響，導致性能欠佳。相比之下，SDR 捕捉到的影子變數被用作隱藏混淆變數的代理，以減輕混淆偏誤，同時也解決了選擇偏誤。
* 在較小的數據集 Coat 上，SDR 的表現與 IDCF 和 DCF 相似，但隨著數據集規模的增加，我們的模型表現出更優越的性能，這表明 SDR 充分利用了數據集中的信息來捕捉使用者的真實興趣。

### C. Experiments on Synthetic Data (RQ2)

### C. 合成數據實驗 (RQ2)

We verify whether the proposed method captures the real interests of users under MNAR data. We generate a synthetic dataset which is adjusted by three main hyper-parameters. The parameter β∈ (0,1] controls the sparsity of the exposure matrix; a smaller value of β results in a sparser matrix. γε [0,1]represents the strength of the selection bias. When y = 0, the data has no selection bias. As y increases, the strength of the selection bias increases. e is the random noise in the exposure matrix used to simulate error exposure.

我們驗證了所提出的方法是否能在 MNAR 數據下捕捉到使用者的真實興趣。我們生成了一個由三個主要超參數調整的合成數據集。參數 β∈ (0,1] 控制曝光矩陣的稀疏性；較小的 β 值會導致更稀疏的矩陣。γε [0,1] 代表選擇偏誤的強度。當 y = 0 時，數據沒有選擇偏誤。隨著 y 的增加，選擇偏誤的強度也隨之增加。e 是用於模擬誤差曝光的曝光矩陣中的隨機噪聲。

a) De-selection Bias Experiments: Experiments were conducted with varying strengths of selection bias on β = 0.1 and € = 10. Experimental results are reported in IV, and we have the following observations:
* At y = 0, the dataset exhibits no discernible selection bias, with only a minimal degree of confounding bias. IDCF achieves the best baseline due to its effective handling of confounding bias. SDR achieves similar results to IDCF as it is also capable of addressing confounding bias.
* At y = 0.8, selection bias dominates the dataset. In this case, methods for dealing with selection bias such as IPS-RD, RD, and IPS achieve results similar to the advanced IDCF. Our SDR performs optimally because it mitigates both selection bias and confounding bias.
* DeepDCF performs poorly, which is attributed to the direct introduction of features within the context of complex feature patterns. SDR and IDCF utilize explicit features only indirectly and are not affected by complex features.

a) 去選擇偏誤實驗：在 β = 0.1 和 € = 10 的條件下，進行了不同選擇偏誤強度的實驗。實驗結果報告在 IV 中，我們有以下觀察：
* 在 y = 0 時，數據集沒有表現出可辨識的選擇偏誤，只有極小程度的混淆偏誤。IDCF 因其有效處理混淆偏誤而達到了最佳基線。SDR 取得了與 IDCF 相似的結果，因為它同樣能夠處理混淆偏誤。
* 在 y = 0.8 時，選擇偏誤主導了數據集。在這種情況下，處理選擇偏誤的方法，如 IPS-RD、RD 和 IPS，取得了與先進的 IDCF 相似的結果。我們的 SDR 表現最佳，因為它同時減輕了選擇偏誤和混淆偏誤。
* DeepDCF 表現不佳，這歸因於在複雜特徵模式的背景下直接引入特徵。SDR 和 IDCF 僅間接利用顯式特徵，不受複雜特徵的影響。

b) Implicit Variable Learning: Furthermore, we test the model's ability to learn implicit variables on synthetic data. To show clear results, we conduct experiments under β = 0.8 and € = 0. As shown in Figure 5: (a) shows the true distribution of shadow variables, (b) presents our SVC, and (c) and (d) indicates the two models IDCF and DeepDCF.
In this experiment, we observe that: (1) The SVC, based on elaborate constraints, achieves the clearest distinction. (2) IDCF utilizes explicit features, thus achieving sub-optimal results with insufficient distinction between distributions. (3) DeepDCF is superior to IDCF in terms of distinction but overly clustered in distribution. This justifies the ability of SVC to mine implicit shadow variables from observational recommendation data.

b) 隱含變數學習：此外，我們在合成數據上測試了模型學習隱含變數的能力。為了清楚地展示結果，我們在 β = 0.8 和 € = 0 的條件下進行了實驗。如圖 5 所示：(a) 顯示了影子變數的真實分佈，(b) 展示了我們的 SVC，(c) 和 (d) 分別表示 IDCF 和 DeepDCF 這兩個模型。
在這個實驗中，我們觀察到：(1) 基於精心設計的約束，SVC 實現了最清晰的區分。(2) IDCF 利用了顯式特徵，因此在分佈之間區分不足的情況下取得了次優的結果。(3) DeepDCF 在區分方面優於 IDCF，但在分佈上過於聚集。這證明了 SVC 從觀測推薦數據中挖掘隱含影子變數的能力。

### D. In-depth Analysis (RQ3)

### D. 深度分析 (RQ3)

Next we will analyse precisely the contribution of each component in our SDR.

接下來，我們將精確分析我們 SDR 中每個組件的貢獻。

a) SDR Ablation Study: In SDR, a sequential process involves learning shadow variables and then proceeding to unbiased training. Table V presents the results of this ablation experiment on the Yahoo! R3 dataset. We report the outcomes on both the validation and test sets, along with the relative improvement (RI) of the model over the baseline Matrix Factorization (MF). The Matrix Factorization With Variables (MF-WV) model incorporates only the learned shadow variables Z and does not include the unbiased training module.

a) SDR 消融研究：在 SDR 中，一個順序過程涉及學習影子變數，然後進行無偏訓練。表 V 展示了在 Yahoo! R3 數據集上此消融實驗的結果。我們報告了在驗證集和測試集上的結果，以及模型相對於基線矩陣分解 (MF) 的相對改進 (RI)。帶變數的矩陣分解 (MF-WV) 模型僅納入了學習到的影子變數 Z，不包含無偏訓練模塊。

It can be observed that the addition of the shadow variables is the most significant improvement to the model. This suggests that learned shadow variables are of enormous value in debasing. And with the addition of unbiased training which does not incur the cost of model inference, the model is clearly enhanced in both validation and testing.

可以觀察到，影子變數的加入是模型最顯著的改進。這表明學習到的影子變數在去偏誤方面具有巨大的價值。並且，在加入了不增加模型推論成本的無偏訓練後，模型在驗證和測試上都得到了明顯的增強。

b) SVC Ablation Study: We conducted an in-depth study of the impact of the loss components of SVC. Table VI presents the test set performance of different loss-learned variables on model MF-WV in the Yahoo!R3 dataset. Upon analysis of the experiments, the following findings emerged:
* SVC learned variables perform worst when w/o LRec. Since under the experimental setup (Section V-A), the rating matrix contained less information, it is not possible to guarantee the source of the latent variables.
* The loss LRec ensures that sufficient information is available to generate the shadow variables. While LPred and LTest are two auxiliary losses that constrain the generated variables into required shadow variables for recommendation debiasing.
* The loss LTest has the least contribution to the SVC. Despite not enhancing the model accuracy markedly, LTest empirically adjusts the distribution of Z to align with the Condition 3 for shadow variables.

b) SVC 消融研究：我們對 SVC 損失組件的影響進行了深入研究。表 VI 展示了在 Yahoo!R3 數據集上，不同損失學習變數在模型 MF-WV 上的測試集性能。經實驗分析，得出以下結論：
* 當沒有 LRec 時，SVC 學習到的變數表現最差。由於在實驗設置（第五-A節）下，評分矩陣包含的資訊較少，因此無法保證潛在變數的來源。
* LRec 損失確保有足夠的資訊來生成影子變數。而 LPred 和 LTest 是兩個輔助損失，將生成的變數約束為推薦去偏誤所需的影子變數。
* LTest 損失對 SVC 的貢獻最小。儘管沒有顯著提高模型準確性，但 LTest 經驗性地調整了 Z 的分佈，以使其與影子變數的條件 3 對齊。

## VI. CONCLUSION

## VI. 結論

In this work, we have developed a method to combat selection bias in collaborative filtering. The proposed SVC model effectively learns shadow variables (e.g., user interests) from the data. The shadow variable, as a special covariate, opens up a new causal path between exposure and rating. Based on causal inference techniques, we have also proposed a SDR method for addressing selection bias. The SDR framework recovers the missing distribution due to selection bias by utilizing shadow variables, thereby facilitating the training of unbiased recommendation models with average effects. Extensive experiments have validated the effectiveness and robustness of our proposed SDR method in mitigating selection bias in recommender systems. In our future work, we will incorporate real-time user feedback and preferences to dynamically adjust shadow variables and improve recommendation accuracy.

在這項工作中，我們開發了一種對抗協同過濾中選擇偏誤的方法。所提出的 SVC 模型能有效地從數據中學習影子變數（例如，使用者興趣）。影子變數作為一種特殊的共變數，開啟了一條介於曝光和評分之間的新因果路徑。基於因果推論技術，我們也提出了一種 SDR 方法來解決選擇偏誤。SDR 框架利用影子變數恢復由選擇偏誤引起的缺失分佈，從而促進了具有平均效應的無偏推薦模型的訓練。廣泛的實驗驗證了我們提出的 SDR 方法在減輕推薦系統中選擇偏誤方面的有效性和穩健性。在我們未來的工作中，我們將納入即時使用者回饋和偏好，以動態調整影子變數並提高推薦準確性。

## VII. ACKNOLEDGEMENT

## VII. 致謝

This work was partially supported by the Specific Research Project of Guangxi for Research Bases and Talents (GuiKe AD24010011) and the Australian Research Council (grant number: DP230101122).

本研究部分由廣西科技基地和人才專項研究計畫 (桂科 AD24010011) 和澳大利亞研究委員會 (撥款號：DP230101122) 資助。

## REFERENCES

## 參考文獻

[1] S. Zhang, X. Li et al., "Learning k for knn classification," ACM Transactions on Intelligent Systems and Technology (TIST), vol. 8, no. 3, pp. 1-19, 2017.
[2] X. Zhu, S. Zhang et al., "Low-rank sparse subspace for spectral clustering," IEEE Transactions on knowledge and data engineering, vol. 31, no. 8, pp. 1532-1543, 2018.
[3] W. Fan, Y. Ma et al., "Graph neural networks for social recommendation," in The world wide web conference, 2019, pp. 417-426.
[4] W. Lan, T. Yang et al., "Multiview subspace clustering via low-rank symmetric affinity graph," IEEE Transactions on Neural Networks and Learning Systems, 2023.
[5] J. Chen, H. Dong et al., "Bias and debias in recommender system: A survey and future directions," ACM Transactions on Information Systems, vol. 41, no. 3, pp. 1-39, 2023.
[6] S. Li, Q. Chen, Z. Liu, S. Pan, and S. Zhang, "Bi-sgtar: A simple yet efficient model for circrna-disease association prediction based on known association pair only," Knowledge-Based Systems, vol. 291, p. 111622, 2024.
[7] T. Wei, F. Feng et al., "Model-agnostic counterfactual reasoning for eliminating popularity bias in recommender system," in Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery & Data Mining, 2021, pp. 1791-1800.
[8] C. Hansen, C. Hansen et al., "Contextual and sequential user embeddings for large-scale music recommendation," in Proceedings of the 14th ACM Conference on Recommender Systems, 2020, pp. 53-62.
[9] C. Gao, S. Li et al., "Kuairand: an unbiased sequential recommendation dataset with randomly exposed videos," in Proceedings of the 31st ACM International Conference on Information & Knowledge Management, 2022, pp. 3953-3957.
[10] C. Gao, Y. Zheng et al., "Causal inference in recommender systems: A survey and future directions," ACM Transactions on Information Systems, vol. 42, no. 4, pp. 1-32, 2024.
[11] J. Pearl, Causality. Cambridge university press, 2009.
[12] J. M. Hernández-Lobato et al., "Probabilistic matrix factorization with non-random missing data," in International conference on machine learning. PMLR, 2014, pp. 1512-1520.
[13] H. Steck, "Evaluation of recommendations: rating-prediction and ranking," in Proceedings of the 7th ACM conference on Recommender systems, 2013, pp. 213-220.
[14] M. Wang, M. Gong et al., "Modeling dynamic missingness of implicit feedback for recommendation," Advances in neural information processing systems, vol. 31, 2018.
[15] X. d'Haultfoeuille, "A new instrumental method for dealing with endogenous selection," Journal of Econometrics, vol. 154, no. 1, pp. 1-15, 2010.
[16] F. Elwert and C. Winship, "Endogenous selection bias: The problem of conditioning on a collider variable," Annual review of sociology, vol. 40, pp. 31-53, 2014.
[17] J. Zhao and J. Shao, "Semiparametric pseudo-likelihoods in generalized linear models with nonignorable missing data," Journal of the American Statistical Association, vol. 110, no. 512, pp. 1577-1590, 2015.
[18] W. Miao, L. Liu et al., "Identification, doubly robust estimation, and semiparametric efficiency theory of nonignorable missing data with a shadow variable," arXiv preprint arXiv:1509.02556, 2015.
[19] W. Miao and E. J. Tchetgen Tchetgen, "On varieties of doubly robust estimators under missingness not at random with a shadow variable," Biometrika, vol. 103, no. 2, pp. 475-482, 2016.
[20] W. Li, W. Miao, and E. Tchetgen Tchetgen, "Non-parametric inference about mean functionals of non-ignorable non-response data without identifying the joint distribution," Journal of the Royal Statistical Society Series B: Statistical Methodology, vol. 85, no. 3, pp. 913-935, 2023.
[21] B. Marlin, R. S. Zemel et al., "Collaborative filtering and the missing at random assumption," arXiv preprint arXiv:1206.5267, 2012.
[22] D. P. Kingma and M. Welling, "Auto-encoding variational bayes," arXiv preprint arXiv:1312.6114, 2013.
[23] Z. Liu, Q. Chen, W. Lan, H. Lu, and S. Zhang, "Ssldti: A novel method for drug-target interaction prediction based on self-supervised learning," Artificial Intelligence in Medicine, vol. 149, p. 102778, 2024.
[24] D. Cheng, J. Li et al., "Data-driven causal effect estimation based on graphical causal modelling: A survey," ACM Computing Surveys, vol. 56, no. 5, pp. 1-37, 2024.
[25] W. Wang, F. Feng et al., "Deconfounded recommendation for alleviating bias amplification," in Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery & Data Mining, 2021, pp. 1717-1725.
[26] Y. Zhang, F. Feng et al., "Causal intervention for leveraging popularity bias in recommendation," in Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval, 2021, pp. 11-20.
[27] X. Zhu, Y. Zhang et al., "Mitigating hidden confounding effects for causal recommendation," IEEE Transactions on Knowledge and Data Engineering, 2024.
[28] S. Xu, J. Tan et al., "Deconfounded causal collaborative filtering," ACM Transactions on Recommender Systems, vol. 1, no. 4, pp. 1-25, 2023.
[29] H. Liu, D. Tang et al., "Rating distribution calibration for selection bias mitigation in recommendations," in Proceedings of the ACM Web Conference 2022, 2022, pp. 2048-2057.
[30] Y. Zhu, J. Yi et al., "Deep causal reasoning for recommendations," ACM Transactions on Intelligent Systems and Technology, 2022.
[31] Q. Zhang, X. Zhang et al., "Debiasing recommendation by learning identifiable latent confounders," in Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, 2023, pp. 3353-3363.
[32] Z. Wang, Y. He et al., "Invariant preference learning for general debiasing in recommendation," in Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, 2022, pp. 1969-1978.
[33] H. Li, K. Wu et al., "Removing hidden confounding in recommendation: a unified multi-task learning approach," in Proceedings of the 37th International Conference on Neural Information Processing Systems, ser. NIPS '23. Red Hook, NY, USA: Curran Associates Inc., 2024.
[34] T. Schnabel, A. Swaminathan et al., "Recommendations as treatments: Debiasing learning and evaluation," in international conference on machine learning. PMLR, 2016, pp. 1670-1679.
[35] Y. Saito, "Asymmetric tri-training for debiasing missing-not-at-random explicit feedback," in Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval, 2020, pp. 309-318.
[36] S. Ding, P. Wu et al., "Addressing unmeasured confounder for recommendation with sensitivity analysis," in Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, 2022, pp. 305-315.
[37] Y. Liu, J.-N. Yen et al., “Practical counterfactual policy learning for top-k recommendations," in Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, 2022, pp. 1141-1151.
[38] X. Wang, R. Zhang et al., "Doubly robust joint learning for recommendation on data missing not at random," in International Conference on Machine Learning. PMLR, 2019, pp. 6638-6647.
[39] H. Li, C. Zheng, and P. Wu, "Stabledr: Stabilized doubly robust learning for recommendation on data missing not at random," 2023. [Online]. Available: https://arxiv.org/abs/2205.04701
[40] M. B. Mathur and I. Shpitser, "Simple graphical rules to assess selection bias in general-population and selected-sample treatment effects."
[41] D. Cheng, Z. Xu et al., "Causal inference with conditional instruments using deep generative models," in Proceedings of the AAAI conference on artificial intelligence, vol. 37, no. 6, 2023, pp. 7122-7130.
[42] I. Khemakhem, D. Kingma et al., "Variational autoencoders and nonlinear ica: A unifying framework," in International Conference on Artificial Intelligence and Statistics. PMLR, 2020, pp. 2207-2217.
[43] X. Wang, R. Zhang et al., "Combating selection biases in recommender systems with a few unbiased ratings," in Proceedings of the 14th ACM International Conference on Web Search and Data Mining, 2021, pp. 427-435.
[44] M. Szumilas, "Explaining odds ratios," Journal of the Canadian academy of child and adolescent psychiatry, vol. 19, no. 3, p. 227, 2010.
[45] L. Dinh, J. Sohl-Dickstein, and S. Bengio, "Density estimation using real nvp," arXiv preprint arXiv:1605.08803, 2016.
[46] Y. Wang, D. Liang et al., "Causal inference for recommender systems," in Proceedings of the 14th ACM Conference on Recommender Systems, 2020, pp. 426-431.
[47] Y. Koren, R. Bell, and C. Volinsky, "Matrix factorization techniques for recommender systems," Computer, vol. 42, no. 8, pp. 30-37, 2009.
[48] M. Arjovsky, L. Bottou et al., "Invariant risk minimization," arXiv preprint arXiv: 1907.02893, 2019.
