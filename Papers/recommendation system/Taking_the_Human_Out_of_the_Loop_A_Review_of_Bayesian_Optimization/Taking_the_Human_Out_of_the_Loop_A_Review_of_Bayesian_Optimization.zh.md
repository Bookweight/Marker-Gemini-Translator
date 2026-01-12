
# 擺脫人類介入：貝氏最佳化回顧

The paper introduces the reader to Bayesian optimization, highlighting its methodical aspects and showcasing its applications.
本文旨在向讀者介紹貝氏最佳化，重點介紹其方法論層面並展示其應用。

By BOBAK SHAHRIARI, KEVIN SWERSKY, ZIYU WANG, RYAN P. ADAMS, AND NANDO DE FREITAS
作者：BOBAK SHAHRIARI, KEVIN SWERSKY, ZIYU WANG, RYAN P. ADAMS, AND NANDO DE FREITAS

### ABSTRACT
ABSTRACT | Big Data applications are typically associated with systems involving large numbers of users, massive complex software systems, and large-scale heterogeneous computing and storage architectures. The construction of such systems involves many distributed design choices. The end products (e.g., recommendation systems, medical analysis tools, real-time game engines, speech recognizers) thus involve many tunable configuration parameters. These parameters are often specified and hard-coded into the software by various developers or teams. If optimized jointly, these parameters can result in significant improvements. Bayesian optimization is a powerful tool for the joint optimization of design decisions. It is a principled approach that has been successfully applied to a wide range of problems, from sensor management to robotics, and from combinatorial optimization to advertising and recommendation. This review paper introduces the reader to Bayesian optimization, highlights its methodical aspects, and showcases its applications.

### 摘要
摘要 | 大數據應用通常與涉及大量使用者、龐大複雜的軟體系統以及大規模異構計算和儲存架構的系統相關聯。建構此類系統涉及許多分散式的設計選擇。因此，最終產品（例如，推薦系統、醫療分析工具、即時遊戲引擎、語音辨識器）涉及許多可調整的組態參數。這些參數通常由不同的開發人員或團隊指定並硬式編碼到軟體中。如果對這些參數進行聯合最佳化，可以帶來顯著的改善。貝氏最佳化是針對設計決策進行聯合最佳化的強大工具。它是一種有原則的方法，已成功應用於從感測器管理到機器人技術，從組合最佳化到廣告和推薦等廣泛問題。本回顧論文旨在向讀者介紹貝氏最佳化，重點介紹其方法論層面並展示其應用。

### KEYWORDS
KEYWORDS | A/B testing; active learning; automatic algorithm configuration; Bayesian optimization; Gaussian processes; hyperparameter tuning; sequential experimental design; surrogate models

### 關鍵詞
關鍵詞 | A/B testing; active learning; automatic algorithm configuration; Bayesian optimization; Gaussian processes; hyperparameter tuning; sequential experimental design; surrogate models

## I. INTRODUCTION
## I. 前言

Consider the problem of finding the optimal configuration of the 167 parameters of the IBM ILOG CPLEX software package, a state-of-the-art commercial solver for mixed integer programming [1]. Or consider the problem of tuning the more than 20 parameters of a deep neural network for image recognition [2]. Or the problem of finding the best chemical composition for a new drug [3]. Or the problem of deciding which advertisement to show to a user of a web site [4]. These problems, and many others like them, can be cast as the optimization of a black-box function:
試想一下，要找到 IBM ILOG CPLEX 軟體套件的最佳組態，這是一個用於混合整數規劃的頂尖商業求解器，它有 167 個參數 [1]。或者，試想一下，要為一個用於影像辨識的深度神經網路調整超過 20 個參數 [2]。或者，要為一種新藥找到最佳的化學成分 [3]。或者，要決定向網站使用者展示哪個廣告 [4]。這些問題以及許多類似的問題，都可以被視為對一個黑箱函數進行最佳化：

f(x) : X → R, (1)
f(x) : X → R, (1)

where the domain X is some complicated, high-dimensional set and the function f is non-convex and expensive to evaluate. By expensive, we mean that it may take hours or days to obtain the result of a single function evaluation. For example, in the case of CPLEX, a single evaluation of f(x) on a particular problem instance might involve running the solver with the parameter configuration x for several hours to assess its performance.
其中定義域 X 是一些複雜的高維度集合，而函數 f 是非凸的且評估成本高昂。所謂成本高昂，是指單次函數評估可能需要數小時或數天的時間才能獲得結果。例如，在 CPLEX 的情況下，在特定問題實例上對 f(x) 進行單次評估，可能需要使用參數組態 x 執行求解器數小時以評估其效能。

Bayesian optimization is a sequential strategy for the global optimization of black-box functions like the one in (1). The basic idea is to use a surrogate model for the expensive function f and to choose where to sample next by making a trade-off between exploration (sampling in areas of high uncertainty) and exploitation (sampling in areas that are likely to be good). The surrogate model is a probabilistic model, typically a Gaussian process (GP), that provides a posterior distribution over the function f. This posterior distribution represents our belief about the function given the data we have observed so far. The trade-off between exploration and exploitation is managed by an acquisition function that uses the posterior distribution to decide where to sample next.
貝氏最佳化是一種用於對如 (1) 式中的黑箱函數進行全域最佳化的循序策略。其基本思想是為昂貴的函數 f 使用一個代理模型，並透過在探索（在高不確定性區域取樣）和利用（在可能表現良好的区域取樣）之間進行權衡來選擇下一個取樣點。代理模型是一個機率模型，通常是高斯過程 (GP)，它提供了函數 f 的後驗分佈。這個後驗分佈代表了我們在給定迄今為止觀察到的數據下對函數的信念。探索和利用之間的權衡由一個採集函數管理，該函數利用後驗分佈來決定下一個取樣點。

The general Bayesian optimization procedure is shown in Algorithm 1. We start with some initial data D1 = {(x1, y1), . . . , (xn, yn)}, where yi = f(xi) + εi and εi is some observation noise. Then, for a number of iterations t = n, n + 1, . . . , we do the following:
1) Update the posterior distribution on f using all available data Dt.
2) Choose the next query point xt+1 by maximizing an acquisition function α(x; Dt) over X.
3) Evaluate yt+1 = f(xt+1) + εt+1.
4. Augment the data Dt+1 = Dt ∪ {(xt+1, yt+1)} and repeat.
一般的貝氏最佳化程序如演算法 1 所示。我們從一些初始數據 D1 = {(x1, y1), . . . , (xn, yn)} 開始，其中 yi = f(xi) + εi 且 εi 是一些觀測噪聲。然後，在 t = n, n + 1, . . . 的迭代次數中，我們執行以下操作：
1) 使用所有可用數據 Dt 更新 f 的後驗分佈。
2) 透過在 X 上最大化採集函數 α(x; Dt) 來選擇下一個查詢點 xt+1。
3) 評估 yt+1 = f(xt+1) + εt+1。
4) 擴增數據 Dt+1 = Dt ∪ {(xt+1, yt+1)} 並重複。

This sequential process is illustrated in Fig. 1. The top panel shows the true function f (which is unknown to the algorithm), the noisy observations, and the GP surrogate model. The shaded region represents the uncertainty of the GP. The bottom panel shows the acquisition function, which is high in areas of high uncertainty (exploration) and in areas where the GP mean is high (exploitation). The next point to be evaluated is the one that maximizes the acquisition function.
這個循序過程如圖 1 所示。頂部面板顯示了真實函數 f（演算法未知）、帶有噪聲的觀測值以及 GP 代理模型。陰影區域代表 GP 的不確定性。底部面板顯示了採集函數，它在高不確定性區域（探索）和 GP 均值高的區域（利用）值較高。下一個要評估的點是使採集函數最大化的點。

[Image]
Fig. 1. Bayesian optimization. The top panel shows the true function (dashed line), the noisy observations (crosses), and the GP posterior. The shaded region represents the 95% confidence interval. The bottom panel shows the acquisition function. The next point to be evaluated is the one that maximizes the acquisition function.
圖 1. 貝氏最佳化。頂部面板顯示真實函數（虛線）、帶有噪聲的觀測值（十字）以及 GP 後驗。陰影區域代表 95% 的信賴區間。底部面板顯示採集函數。下一個要評估的點是使採集函數最大化的點。

The two main components of a Bayesian optimization algorithm are the surrogate model and the acquisition function. We will discuss these in detail in the following sections.
貝氏最佳化演算法的兩個主要組成部分是代理模型和採集函數。我們將在以下章節中詳細討論這些內容。

## II. THE SURROGATE MODEL
## II. 代理模型

The surrogate model is a probabilistic model for the objective function f. The most common choice is a Gaussian process (GP), which is a distribution over functions. A GP is specified by a mean function m(x) and a covariance function k(x, x'). The mean function is usually assumed to be zero, and the covariance function (or kernel) encodes our prior beliefs about the function, such as its smoothness. Given a set of observations Dt = {(x1, y1), . . . , (xt, yt)}, the posterior distribution over f is also a GP with updated mean and covariance functions. Specifically, the posterior predictive distribution at a new point x is a Gaussian with mean µt(x) and variance σt^2(x) given by
代理模型是目標函數 f 的機率模型。最常見的選擇是高斯過程 (GP)，它是一種函數上的分佈。GP 由一個均值函數 m(x) 和一個協方差函數 k(x, x') 指定。均值函数通常假設為零，而協方差函數（或核心）則編碼了我們對函數的先驗信念，例如其平滑度。給定一組觀測值 Dt = {(x1, y1), . . . , (xt, yt)}，f 的後驗分佈也是一個具有更新的均值和協方差函數的 GP。具體來說，在新點 x 的後驗預測分佈是一個高斯分佈，其均值 µt(x) 和變異數 σt^2(x) 由下式給出：

µt(x) = k(x, Xt)(K + σ^2I)^-1 yt
σt^2(x) = k(x, x) - k(x, Xt)(K + σ^2I)^-1 k(Xt, x) (2)
µt(x) = k(x, Xt)(K + σ^2I)^-1 yt
σt^2(x) = k(x, x) - k(x, Xt)(K + σ^2I)^-1 k(Xt, x) (2)

where Xt = [x1, . . . , xt]^T, yt = [y1, . . . , yt]^T, K is the t × t matrix of kernel evaluations Kij = k(xi, xj), and k(x, Xt) is the 1 × t vector [k(x, x1), . . . , k(x, xt)]. The hyper-parameters of the kernel (e.g., length-scales, signal variance) and the noise variance σ^2 are typically learned by maximizing the marginal likelihood of the data.
其中 Xt = [x1, . . . , xt]^T, yt = [y1, . . . , yt]^T, K 是核心評估的 t × t 矩陣 Kij = k(xi, xj)，而 k(x, Xt) 是 1 × t 的向量 [k(x, x1), . . . , k(x, xt)]。核心的超參數（例如，長度尺度、信號變異數）和噪聲變異數 σ^2 通常是透過最大化數據的邊際概似來學習的。

The choice of kernel is crucial and depends on the properties of the function being modeled. A common choice is the squared exponential kernel,
核心的選擇至關重要，取決於被建模函數的屬性。一個常見的選擇是平方指數核心，

kSE(x, x') = σf^2 exp(-1/2l^2 ||x - x'||^2) (3)
kSE(x, x') = σf^2 exp(-1/2l^2 ||x - x'||^2) (3)

which assumes that the function is infinitely differentiable. The Matérn family of kernels is a more general class that allows for controlling the smoothness of the function. For example, the Matérn-5/2 kernel,
它假設函數是無限可微的。Matérn 核心家族是一個更通用的類別，允許控制函數的平滑度。例如，Matérn-5/2 核心，

kM52(x, x') = σf^2 (1 + sqrt(5)r/l + 5r^2/3l^2) exp(-sqrt(5)r/l) (4)
kM52(x, x') = σf^2 (1 + sqrt(5)r/l + 5r^2/3l^2) exp(-sqrt(5)r/l) (4)

where r = ||x - x'||, assumes that the function is twice differentiable. This is often a more realistic assumption than infinite differentiability.
其中 r = ||x - x'||，假設函數是二次可微的。這通常是一個比無限可微性更現實的假設。

Other surrogate models have also been used for Bayesian optimization, including random forests [5] and deep neural networks [6]. However, GPs remain the most popular choice due to their flexibility, analytical tractability, and ability to provide well-calibrated uncertainty estimates.
其他代理模型也已用於貝氏最佳化，包括隨機森林 [5] 和深度神經網路 [6]。然而，由於其靈活性、分析易處理性以及提供良好校準的不確定性估計的能力，GP 仍然是最受歡迎的選擇。

## III. ACQUISITION FUNCTIONS
## III. 採集函數

The acquisition function is a heuristic for deciding where to sample next. It is a function of x that is cheap to evaluate and is maximized to select the next query point. The goal of the acquisition function is to trade off exploration and exploitation. Exploration means sampling in areas of high uncertainty, where we might find a better solution. Exploitation means sampling in areas that are likely to be good, i.e., where the current model predicts a high objective value.
採集函數是一種決定下一個取樣點的啟發式方法。它是一個關於 x 的函數，評估成本低廉，並透過最大化它來選擇下一個查詢點。採集函數的目標是在探索和利用之間進行權衡。探索意味著在高不確定性區域取樣，在那裡我們可能會找到更好的解決方案。利用意味著在可能表現良好的區域取樣，即當前模型預測目標值高的區域。

Let f(x+) be the best value observed so far, where x+ = argmax xi∈{x1,...,xt} f(xi). Several popular acquisition functions have been proposed.
令 f(x+) 為迄今為止觀察到的最佳值，其中 x+ = argmax xi∈{x1,...,xt} f(xi)。已經提出了幾種流行的採集函數。

### A. Probability of Improvement (PI)
### A. 改善機率 (PI)

The Probability of Improvement (PI) acquisition function [7] chooses the point that has the highest probability of improving upon the current best value f(x+).
改善機率 (PI) 採集函數 [7] 選擇最有可能改善當前最佳值 f(x+) 的點。

αPI(x) = P(f(x) ≥ f(x+) + ξ) = Φ(µt(x) - f(x+) - ξ / σt(x)) (5)
αPI(x) = P(f(x) ≥ f(x+) + ξ) = Φ(µt(x) - f(x+) - ξ / σt(x)) (5)

where Φ(·) is the standard normal cumulative distribution function and ξ ≥ 0 is a trade-off parameter that encourages exploration. A larger ξ will favor more exploration.
其中 Φ(·) 是標準常態累積分布函數，ξ ≥ 0 是一個權衡參數，用於鼓勵探索。較大的 ξ 將有利於更多的探索。

### B. Expected Improvement (EI)
### B. 預期改善 (EI)

The Expected Improvement (EI) acquisition function [8] is one of the most popular choices. It computes the expected improvement over the current best value f(x+).
預期改善 (EI) 採集函數 [8] 是最受歡迎的選擇之一。它計算相對於當前最佳值 f(x+) 的預期改善量。

αEI(x) = E[max(f(x) - f(x+), 0)]
= (µt(x) - f(x+) - ξ)Φ(Z) + σt(x)φ(Z) if σt(x) > 0
= 0 if σt(x) = 0 (6)
αEI(x) = E[max(f(x) - f(x+), 0)]
= (µt(x) - f(x+) - ξ)Φ(Z) + σt(x)φ(Z) if σt(x) > 0
= 0 if σt(x) = 0 (6)

where Z = (µt(x) - f(x+) - ξ) / σt(x) and φ(·) is the standard normal probability density function. Like PI, EI has a parameter ξ that can be used to control the trade-off between exploration and exploitation.
其中 Z = (µt(x) - f(x+) - ξ) / σt(x) 且 φ(·) 是標準常態機率密度函數。與 PI 一樣，EI 有一個參數 ξ，可用於控制探索和利用之間的權衡。

### C. Upper Confidence Bound (UCB)
### C. 上限信賴區間 (UCB)

The Upper Confidence Bound (UCB) acquisition function [9] is derived from the principle of optimism in the face of uncertainty. It chooses the point with the highest upper confidence bound on the function value.
上限信賴區間 (UCB) 採集函數 [9] 源於面對不確定性時的樂觀主義原則。它選擇函數值上信賴區間最高的點。

αUCB(x) = µt(x) + κt σt(x) (7)
αUCB(x) = µt(x) + κt σt(x) (7)

The parameter κt ≥ 0 controls the trade-off between exploration and exploitation. A larger κt encourages more exploration. Theoretical guarantees on the performance of UCB-based algorithms exist, which state that they can find a near-optimal solution with high probability.
參數 κt ≥ 0 控制探索和利用之間的權衡。較大的 κt 鼓勵更多的探索。基於 UCB 的演算法的效能存在理論保證，指出它們可以高機率地找到接近最佳的解決方案。

### D. Entropy Search (ES) and Predictive Entropy Search (PES)
### D. 熵搜尋 (ES) 和預測熵搜尋 (PES)

More recent acquisition functions are based on information theory. Entropy Search (ES) [10] and Predictive Entropy Search (PES) [11] choose the point that is expected to provide the most information about the location of the optimum x*. They do this by measuring the expected reduction in entropy of the posterior distribution over x*.
更新的採集函數基於資訊理論。熵搜尋 (ES) [10] 和預測熵搜尋 (PES) [11] 選擇預期能提供關於最佳解 x* 位置最多資訊的點。它們透過測量 x* 的後驗分佈熵的預期減少量來實現這一點。

These methods are more computationally expensive than PI, EI, or UCB, but they can lead to better performance in terms of the number of function evaluations required.
這些方法在計算上比 PI、EI 或 UCB 更昂貴，但就所需的函數評估次數而言，它們可以帶來更好的效能。

## IV. PRACTICAL CONSIDERATIONS
## IV. 實務考量

### A. High-Dimensional Bayesian Optimization
### A. 高維度貝氏最佳化

Bayesian optimization can be challenging in high dimensions due to the curse of dimensionality. The number of samples required to cover the space grows exponentially with the dimension. This makes it difficult to build an accurate surrogate model.
由於維度詛咒，貝氏最佳化在高維度中可能具有挑戰性。覆蓋空間所需的樣本數量隨維度呈指數增長。這使得建立準確的代理模型變得困難。

Several approaches have been proposed to address this issue. One is to use a low-dimensional embedding of the high-dimensional space [12]. Another is to use an additive model, where the high-dimensional function is assumed to be a sum of low-dimensional functions [13]. A third approach, called Random EMbedding Bayesian Optimization (REMBO) [14], performs optimization in a low-dimensional random embedding of the original space.
已經提出了幾種方法來解決這個問題。一種是使用高維度空間的低維度嵌入 [12]。另一種是使用加法模型，其中高維度函數被假設為低維度函數的總和 [13]。第三種方法，稱為隨機嵌入貝氏最佳化 (REMBO) [14]，在原始空間的低維度隨機嵌入中執行最佳化。

### B. Constrained Bayesian Optimization
### B. 受約束的貝氏最佳化

In many real-world problems, the objective function is subject to constraints. For example, in drug design, we might want to maximize the efficacy of a drug subject to the constraint that its toxicity is below a certain threshold.
在許多現實世界的問題中，目標函數受到約束。例如，在藥物設計中，我們可能希望在藥物毒性低於某個閾值的約束下，最大化藥物的功效。

Bayesian optimization can be extended to handle constraints by modeling both the objective function and the constraint functions with GPs. The acquisition function is then modified to take the constraints into account. For example, one can multiply the original acquisition function by the probability of satisfying the constraints [15].
貝氏最佳化可以透過使用 GP 對目標函數和約束函數進行建模來擴展以處理約束。然後修改採集函數以考慮約束。例如，可以將原始採集函數乘以滿足約束的機率 [15]。

### C. Parallel Bayesian Optimization
### C. 平行貝氏最佳化

The sequential nature of Bayesian optimization can be a bottleneck when function evaluations can be performed in parallel. Several approaches have been proposed to parallelize Bayesian optimization. One is to select a batch of points to evaluate at each iteration. This can be done by greedily maximizing the acquisition function multiple times, or by using a batch acquisition function that explicitly accounts for the parallel evaluations [16].
當函數評估可以並行執行時，貝氏最佳化的循序特性可能成為瓶頸。已經提出了幾種方法來並行化貝氏最佳化。一種是在每次迭代中選擇一批點進行評估。這可以透過貪婪地多次最大化採集函數來完成，或者使用明確考慮並行評估的批次採集函數 [16]。

## V. APPLICATIONS
## V. 應用

Bayesian optimization has been successfully applied to a wide range of problems. Here we highlight a few examples.
貝氏最佳化已成功應用於廣泛的問題。這裡我們重點介紹幾個例子。

### A. Automatic Algorithm Configuration
### A. 自動演算法組態

As mentioned in the introduction, Bayesian optimization can be used to automatically configure the parameters of complex algorithms. This has been applied to solvers for mixed integer programming [1], machine learning algorithms [2], and evolutionary algorithms [17].
如前言所述，貝氏最佳化可用於自動組態複雜演算法的參數。這已應用於混合整數規劃的求解器 [1]、機器學習演算法 [2] 和演化演算法 [17]。

### B. Robotics
### B. 機器人學

In robotics, Bayesian optimization has been used for policy search in reinforcement learning [18], for learning locomotion gaits [19], and for sensor placement [20].
在機器人學中，貝氏最佳化已用於強化學習中的策略搜尋 [18]、學習運動步態 [19] 以及感測器放置 [20]。

### C. Environmental Monitoring
### C. 環境監測

Bayesian optimization can be used to guide the collection of data in environmental monitoring applications. For example, it has been used to find the location of the highest concentration of a contaminant in a lake [21] and to map the temperature of a sensor network [22].
貝氏最佳化可用於指導環境監測應用中的數據收集。例如，它已被用於尋找湖泊中污染物最高濃度的位置 [21] 以及繪製感測器網路的溫度圖 [22]。

### D. A/B Testing and Recommendation
### D. A/B 測試與推薦

In online advertising and recommendation, Bayesian optimization can be used to select which ad or item to show to a user. This is a form of A/B testing, where the goal is to find the best option among a set of alternatives. Bayesian optimization provides a principled way to trade off exploration (trying out new options) and exploitation (showing the best option so far) [4].
在線上廣告和推薦中，貝氏最佳化可用於選擇向使用者展示哪個廣告或項目。這是 A/B 測試的一種形式，其目標是在一組備選方案中找到最佳選項。貝氏最佳化提供了一種有原則的方法來權衡探索（嘗試新選項）和利用（展示迄今為止的最佳選項）[4]。

## VI. CONCLUSION
## VI. 結論

Bayesian optimization is a powerful and flexible tool for the global optimization of expensive black-box functions. It has a solid theoretical foundation and has been successfully applied to a wide range of problems. The two main components of a Bayesian optimization algorithm are the surrogate model and the acquisition function. The choice of these components depends on the specific problem at hand.
貝氏最佳化是針對昂貴的黑箱函數進行全域最佳化的強大而靈活的工具。它具有堅實的理論基礎，並已成功應用於廣泛的問題。貝氏最佳化演算法的兩個主要組成部分是代理模型和採集函數。這些組件的選擇取決於手頭的具體問題。

Future research in Bayesian optimization is likely to focus on developing more scalable methods for high-dimensional problems, better methods for handling complex constraints, and more sophisticated acquisition functions that can capture complex trade-offs.
貝氏最佳化的未來研究可能會集中在開發更具可擴展性的高維度問題方法、更好的處理複雜約束的方法，以及能夠捕捉複雜權衡的更複雜的採集函數。

## REFERENCES
## 參考文獻

[1] F. Hutter, H. H. Hoos, and K. Leyton-Brown, “Sequential model-based optimization for general algorithm configuration,” in Proc. 5th Int. Conf. Learn. Intell. Optim. (LION), 2011, pp. 507–523.
[1] F. Hutter, H. H. Hoos, and K. Leyton-Brown, “Sequential model-based optimization for general algorithm configuration,” in Proc. 5th Int. Conf. Learn. Intell. Optim. (LION), 2011, pp. 507–523.

[2] J. Snoek, H. Larochelle, and R. P. Adams, “Practical Bayesian optimization of machine learning algorithms,” in Proc. Adv. Neural Inf. Process. Syst. (NIPS), 2012, pp. 2951–2959.
[2] J. Snoek, H. Larochelle, and R. P. Adams, “Practical Bayesian optimization of machine learning algorithms,” in Proc. Adv. Neural Inf. Process. Syst. (NIPS), 2012, pp. 2951–2959.

[3] D. Balcells, J. M. Bofill, J. M. Anglada, and R. Crehuet, “Bayesian optimization for the discovery of new molecules,” J. Chem. Inf. Model., vol. 55, no. 8, pp. 1647–1658, 2015.
[3] D. Balcells, J. M. Bofill, J. M. Anglada, and R. Crehuet, “Bayesian optimization for the discovery of new molecules,” J. Chem. Inf. Model., vol. 55, no. 8, pp. 1647–1658, 2015.

[4] D. Scott, “Multi-armed bandit problems,” in Bayesian Statistics 9. Oxford, U.K.: Oxford Univ. Press, 2011, pp. 1–27.
[4] D. Scott, “Multi-armed bandit problems,” in Bayesian Statistics 9. Oxford, U.K.: Oxford Univ. Press, 2011, pp. 1–27.

[5] F. Hutter, H. H. Hoos, and K. Leyton-Brown, “An evaluation of sequential model-based optimization for expensive blackbox functions,” in Proc. 14th Int. Conf. Genet. Evol. Comput. Conf. (GECCO), 2012, pp. 1209–1216.
[5] F. Hutter, H. H. Hoos, and K. Leyton-Brown, “An evaluation of sequential model-based optimization for expensive blackbox functions,” in Proc. 14th Int. Conf. Genet. Evol. Comput. Conf. (GECCO), 2012, pp. 1209–1216.

[6] J. Snoek, O. Rippel, K. Swersky, R. Kiros, N. Satish, N. Sundaram, M. Patwary, M. Prabhat, and R. P. Adams, “Scalable Bayesian optimization using deep neural networks,” in Proc. 32nd Int. Conf. Mach. Learn. (ICML), 2015, pp. 2171–2180.
[6] J. Snoek, O. Rippel, K. Swersky, R. Kiros, N. Satish, N. Sundaram, M. Patwary, M. Prabhat, and R. P. Adams, “Scalable Bayesian optimization using deep neural networks,” in Proc. 32nd Int. Conf. Mach. Learn. (ICML), 2015, pp. 2171–2180.

[7] H. J. Kushner, “A new method of locating the maximum point of an arbitrary multipeak curve in the presence of noise,” J. Basic Eng., vol. 86, no. 1, pp. 97–106, 1964.
[7] H. J. Kushner, “A new method of locating the maximum point of an arbitrary multipeak curve in the presence of noise,” J. Basic Eng., vol. 86, no. 1, pp. 97–106, 1964.

[8] J. Mockus, “On Bayesian methods for seeking the extremum,” in Proc. Int. Symp. Inf. Theory, 1975, pp. 1–5.
[8] J. Mockus, “On Bayesian methods for seeking the extremum,” in Proc. Int. Symp. Inf. Theory, 1975, pp. 1–5.

[9] N. Srinivas, A. Krause, S. M. Kakade, and M. Seeger, “Gaussian process optimization in the bandit setting: No regret and experimental design,” in Proc. 27th Int. Conf. Mach. Learn. (ICML), 2010, pp. 1015–1022.
[9] N. Srinivas, A. Krause, S. M. Kakade, and M. Seeger, “Gaussian process optimization in the bandit setting: No regret and experimental design,” in Proc. 27th Int. Conf. Mach. Learn. (ICML), 2010, pp. 1015–1022.

[10] P. Hennig and C. J. Schuler, “Entropy search for information-efficient global optimization,” J. Mach. Learn. Res., vol. 13, pp. 1809–1837, Jun. 2012.
[10] P. Hennig and C. J. Schuler, “Entropy search for information-efficient global optimization,” J. Mach. Learn. Res., vol. 13, pp. 1809–1837, Jun. 2012.

[11] J. M. Hernández-Lobato, M. W. Hoffman, and Z. Ghahramani, “Predictive entropy search for efficient global optimization of black-box functions,” in Proc. Adv. Neural Inf. Process. Syst. (NIPS), 2014, pp. 918–926.
[11] J. M. Hernández-Lobato, M. W. Hoffman, and Z. Ghahramani, “Predictive entropy search for efficient global optimization of black-box functions,” in Proc. Adv. Neural Inf. Process. Syst. (NIPS), 2014, pp. 918–926.

[12] Z. Wang, C. Li, S. Jegelka, and P. Kohli, “Batched high-dimensional Bayesian optimization via structural kernel learning,” 2016, arXiv:1602.06445. [Online]. Available: http://arxiv.org/abs/1602.06445
[12] Z. Wang, C. Li, S. Jegelka, and P. Kohli, “Batched high-dimensional Bayesian optimization via structural kernel learning,” 2016, arXiv:1602.06445. [Online]. Available: http://arxiv.org/abs/1602.06445

[13] K. Kandasamy, J. Schneider, and B. Póczos, “High dimensional Bayesian optimisation and bandits via additive models,” in Proc. 32nd Int. Conf. Mach. Learn. (ICML), 2015, pp. 295–304.
[13] K. Kandasamy, J. Schneider, and B. Póczos, “High dimensional Bayesian optimisation and bandits via additive models,” in Proc. 32nd Int. Conf. Mach. Learn. (ICML), 2015, pp. 295–304.

[14] Z. Wang, M. Zoghi, F. Hutter, D. Matheson, and N. de Freitas, “Bayesian optimization in high dimensions via random embeddings,” in Proc. 23rd Int. Joint Conf. Artif. Intell. (IJCAI), 2013, pp. 1778–1784.
[14] Z. Wang, M. Zoghi, F. Hutter, D. Matheson, and N. de Freitas, “Bayesian optimization in high dimensions via random embeddings,” in Proc. 23rd Int. Joint Conf. Artif. Intell. (IJCAI), 2013, pp. 1778–1784.

[15] M. J. Gardner, M. J. Kusner, Z. E. Xu, K. Q. Weinberger, and J. R. Cunningham, “Bayesian optimization with inequality constraints,” in Proc. 31st Int. Conf. Mach. Learn. (ICML), 2014, pp. 937–945.
[15] M. J. Gardner, M. J. Kusner, Z. E. Xu, K. Q. Weinberger, and J. R. Cunningham, “Bayesian optimization with inequality constraints,” in Proc. 31st Int. Conf. Mach. Learn. (ICML), 2014, pp. 937–945.

[16] J. Snoek, K. Swersky, R. Zemel, and R. P. Adams, “Input warping for Bayesian optimization of non-stationary functions,” 2014, arXiv:1402.0929. [Online]. Available: http://arxiv.org/abs/1402.0929
[16] J. Snoek, K. Swersky, R. Zemel, and R. P. Adams, “Input warping for Bayesian optimization of non-stationary functions,” 2014, arXiv:1402.0929. [Online]. Available: http://arxiv.org/abs/1402.0929

[17] M. A. B. de Athayde, G. A. L. P. de Oliveira, and F. J. V. de Sousa, “A Bayesian optimization approach for tuning the parameters of an evolutionary algorithm,” in Proc. IEEE Congr. Evol. Comput. (CEC), 2014, pp. 199–206.
[17] M. A. B. de Athayde, G. A. L. P. de Oliveira, and F. J. V. de Sousa, “A Bayesian optimization approach for tuning the parameters of an evolutionary algorithm,” in Proc. IEEE Congr. Evol. Comput. (CEC), 2014, pp. 199–206.

[18] P. Brochu, V. M. Cora, and N. de Freitas, “A tutorial on Bayesian optimization of expensive cost functions, with application to active user modeling and hierarchical reinforcement learning,” 2010, arXiv:1012.2599. [Online]. Available: http://arxiv.org/abs/1012.2599
[18] P. Brochu, V. M. Cora, and N. de Freitas, “A tutorial on Bayesian optimization of expensive cost functions, with application to active user modeling and hierarchical reinforcement learning,” 2010, arXiv:1012.2599. [Online]. Available: http://arxiv.org/abs/1012.2599

[19] A. Lizotte, T. Wang, M. Bowling, and D. Schuurmans, “Automatic gait optimization with Gaussian process regression,” in Proc. 20th Int. Joint Conf. Artif. Intell. (IJCAI), 2007, pp. 944–949.
[19] A. Lizotte, T. Wang, M. Bowling, and D. Schuurmans, “Automatic gait optimization with Gaussian process regression,” in Proc. 20th Int. Joint Conf. Artif. Intell. (IJCAI), 2007, pp. 944–949.

[20] A. Krause, A. Singh, and C. Guestrin, “Near-optimal sensor placements in Gaussian processes: Theory, efficient algorithms and empirical studies,” J. Mach. Learn. Res., vol. 9, pp. 235–284, Feb. 2008.
[20] A. Krause, A. Singh, and C. Guestrin, “Near-optimal sensor placements in Gaussian processes: Theory, efficient algorithms and empirical studies,” J. Mach. Learn. Res., vol. 9, pp. 235–284, Feb. 2008.

[21] N. Srinivas, A. Krause, S. M. Kakade, and M. Seeger, “Information-theoretic regret bounds for Gaussian process optimization in the bandit setting,” IEEE Trans. Inf. Theory, vol. 58, no. 5, pp. 3250–3265, May 2012.
[21] N. Srinivas, A. Krause, S. M. Kakade, and M. Seeger, “Information-theoretic regret bounds for Gaussian process optimization in the bandit setting,” IEEE Trans. Inf. Theory, vol. 58, no. 5, pp. 3250–3265, May 2012.

[22] J. M. Hernández-Lobato, A. G. de G. Matthews, and Z. Ghahramani, “Bayesian optimization for adaptive spatial sampling,” in Proc. 18th Int. Conf. Artif. Intell. Statist. (AISTATS), 2015, pp. 364–372.
[22] J. M. Hernández-Lobato, A. G. de G. Matthews, and Z. Ghahramani, “Bayesian optimization for adaptive spatial sampling,” in Proc. 18th Int. Conf. Artif. Intell. Statist. (AISTATS), 2015, pp. 364–372.
