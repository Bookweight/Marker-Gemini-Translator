---
title: Soft Actor-Critic Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor
field: Deep_Learning
status: Imported
created_date: 2026-01-19
pdf_link: "[[Soft Actor-Critic Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor.pdf]]"
tags:
  - paper
  - Deep_learning
---

# Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor
# Soft Actor-Critic：具有隨機演員的異策略最大熵深度強化學習

**Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, Sergey Levine**
**Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, Sergey Levine**

## Abstract
## 摘要

Model-free deep reinforcement learning (RL) algorithms have been demonstrated on a range of challenging decision making and control tasks.
無模型深度強化學習（RL）演算法已在許多具挑戰性的決策和控制任務中獲得驗證。

However, these methods typically suffer from two major challenges: very high sample complexity and brittle convergence properties, which necessitate meticulous hyperparameter tuning.
然而，這些方法通常面臨兩大挑戰：極高的樣本複雜度和脆弱的收斂特性，這使得必須進行精細的超參數調整。

Both of these challenges severely limit the applicability of such methods to complex, real-world domains.
這兩項挑戰嚴重限制了此類方法在複雜現實世界領域中的應用。

In this paper, we propose soft actor-critic, an off-policy actor-critic deep RL algorithm based on the maximum entropy reinforcement learning framework.
在本文中，我們提出了 Soft Actor-Critic（SAC），這是一種基於最大熵強化學習框架的異策略（off-policy）演員-評論家（actor-critic）深度 RL 演算法。

In this framework, the actor aims to maximize expected reward while also maximizing entropy.
在這個框架中，演員的目標是最大化預期獎勵，同時最大化熵。

That is, to succeed at the task while acting as randomly as possible.
換句話說，即在盡可能隨機行動的同時成功完成任務。

Prior deep RL methods based on this framework have been formulated as Q-learning methods.
先前基於此框架的深度 RL 方法已被公式化為 Q-learning 方法。

By combining off-policy updates with a stable stochastic actor-critic formulation, our method achieves state-of-the-art performance on a range of continuous control benchmark tasks, outperforming prior on-policy and off-policy methods.
透過結合異策略更新與穩定的隨機演員-評論家公式，我們的方法在一系列連續控制基準任務上實現了最先進的性能，優於先前的同策略（on-policy）和異策略方法。

Furthermore, we demonstrate that, in contrast to other off-policy algorithms, our approach is very stable, achieving very similar performance across different random seeds.
此外，我們證明，與其他異策略演算法相比，我們的方法非常穩定，在不同的隨機種子下都能達到非常相似的性能。

## 1. Introduction
## 1. 介紹

Model-free deep reinforcement learning (RL) algorithms have been applied in a range of challenging domains, from games (Mnih et al., 2013; Silver et al., 2016) to robotic control (Schulman et al., 2015).
無模型深度強化學習（RL）演算法已應用於一系列具挑戰性的領域，從遊戲（Mnih 等人，2013；Silver 等人，2016）到機器人控制（Schulman 等人，2015）。

The combination of RL and high-capacity function approximators such as neural networks holds the promise of automating a wide range of decision making and control tasks, but widespread adoption of these methods in real-world domains has been hampered by two major challenges.
RL 與神經網路等高容量函數逼近器的結合，有望使廣泛的決策和控制任務自動化，但這兩大挑戰阻礙了這些方法在現實世界領域的廣泛採用。

First, model-free deep RL methods are notoriously expensive in terms of their sample complexity.
首先，無模型深度 RL 方法以其高昂的樣本複雜度而聞名。

Even relatively simple tasks can require millions of steps of data collection, and complex behaviors with high-dimensional observations might need substantially more.
即使是相對簡單的任務也可能需要數百萬步的數據收集，而具有高維觀察的複雜行為可能需要更多。

Second, these methods are often brittle with respect to their hyperparameters: learning rates, exploration constants, and other settings must be set carefully for different problem settings to achieve good results.
其次，這些方法通常對其超參數非常敏感且脆弱：學習率、探索常數和其他設定必須針對不同的問題設定進行仔細設置才能獲得良好的結果。

Both of these challenges severely limit the applicability of model-free deep RL to real-world tasks.
這兩項挑戰嚴重限制了無模型深度 RL 在現實世界任務中的適用性。

One cause for the poor sample efficiency of deep RL methods is on-policy learning: some of the most commonly used deep RL algorithms, such as TRPO (Schulman et al., 2015), PPO (Schulman et al., 2017b) or A3C (Mnih et al., 2016), require new samples to be collected for each gradient step.
深度 RL 方法樣本效率低落的一個原因是同策略（on-policy）學習：一些最常用的深度 RL 演算法，如 TRPO（Schulman 等人，2015）、PPO（Schulman 等人，2017b）或 A3C（Mnih 等人，2016），要求為每個梯度步驟收集新的樣本。

This quickly becomes extravagantly expensive, as the number of gradient steps and samples per step needed to learn an effective policy increases with task complexity.
這很快就會變得非常昂貴，因為學習有效策略所需的梯度步驟數量和每步樣本數隨著任務複雜度的增加而增加。

Off-policy algorithms aim to reuse past experience.
異策略（Off-policy）演算法旨在重用過去的經驗。

This is not directly feasible with conventional policy gradient formulations, but is relatively straightforward for Q-learning based methods (Mnih et al., 2015).
這在傳統的策略梯度公式中並不直接可行，但對於基於 Q-learning 的方法（Mnih 等人，2015）來說相對簡單。

Unfortunately, the combination of off-policy learning and high-dimensional, nonlinear function approximation with neural networks presents a major challenge for stability and convergence (Bhatnagar et al., 2009).
不幸的是，異策略學習與神經網路的高維非線性函數逼近相結合，對穩定性和收斂性構成了重大挑戰（Bhatnagar 等人，2009）。

This challenge is further exacerbated in continuous state and action spaces, where a separate actor network is often used to perform the maximization in Q-learning.
這一挑戰在連續狀態和動作空間中進一步加劇，在這些空間中，通常使用單獨的演員網路來執行 Q-learning 中的最大化。

A commonly used algorithm in such settings, deep deterministic policy gradient (DDPG) (Lillicrap et al., 2015), provides for sample-efficient learning but is notoriously challenging to use due to its extreme brittleness and hyperparameter sensitivity (Duan et al., 2016; Henderson et al., 2017).
在這種環境下常用的一種演算法是深度確定性策略梯度（DDPG）（Lillicrap 等人，2015），它提供了樣本效率高的學習，但由於其極端的脆弱性和超參數敏感性，使用起來極具挑戰性（Duan 等人，2016；Henderson 等人，2017）。

We explore how to design an efficient and stable model-free deep RL algorithm for continuous state and action spaces.
我們探討如何為連續狀態和動作空間設計一種高效且穩定的無模型深度 RL 演算法。

To that end, we draw on the maximum entropy framework, which augments the standard maximum reward reinforcement learning objective with an entropy maximization term (Ziebart et al., 2008; Toussaint, 2009; Rawlik et al., 2012; Fox et al., 2016; Haarnoja et al., 2017).
為此，我們利用最大熵框架，該框架在標準的最大獎勵強化學習目標中增加了一個熵最大化項（Ziebart 等人，2008；Toussaint，2009；Rawlik 等人，2012；Fox 等人，2016；Haarnoja 等人，2017）。

Maximum entropy reinforcement learning alters the RL objective, though the original objective can be recovered using a temperature parameter (Haarnoja et al., 2017).
最大熵強化學習改變了 RL 目標，儘管原始目標可以使用溫度參數恢復（Haarnoja 等人，2017）。

More importantly, the maximum entropy formulation provides a substantial improvement in exploration and robustness: as discussed by Ziebart (2010), maximum entropy policies are robust in the face of model and estimation errors, and as demonstrated by (Haarnoja et al., 2017), they improve exploration by acquiring diverse behaviors.
更重要的是，最大熵公式在探索和穩健性方面提供了顯著的改進：正如 Ziebart (2010) 所討論的，最大熵策略在面對模型和估計誤差時具有穩健性，並且正如 (Haarnoja 等人，2017) 所證明的，它們通過獲取多樣化的行為來改善探索。

Prior work has proposed model-free deep RL algorithms that perform on-policy learning with entropy maximization (O’Donoghue et al., 2016), as well as off-policy methods based on soft Q-learning and its variants (Schulman et al., 2017a; Nachum et al., 2017a; Haarnoja et al., 2017).
先前的工作已經提出了執行帶有熵最大化的同策略學習的無模型深度 RL 演算法（O’Donoghue 等人，2016），以及基於 soft Q-learning 及其變體的異策略方法（Schulman 等人，2017a；Nachum 等人，2017a；Haarnoja 等人，2017）。

However, the on-policy variants suffer from poor sample complexity for the reasons discussed above, while the off-policy variants require complex approximate inference procedures in continuous action spaces.
然而，由於上述原因，同策略變體面臨樣本複雜度差的問題，而異策略變體在連續動作空間中需要複雜的近似推斷過程。

In this paper, we demonstrate that we can devise an off-policy maximum entropy actor-critic algorithm, which we call soft actor-critic (SAC), which provides for both sample-efficient learning and stability.
在本文中，我們證明我們可以設計一種異策略最大熵演員-評論家演算法，我們稱之為 Soft Actor-Critic（SAC），它同時提供了樣本效率高的學習和穩定性。

This algorithm extends readily to very complex, high-dimensional tasks, such as the Humanoid benchmark (Duan et al., 2016) with 21 action dimensions, where off-policy methods such as DDPG typically struggle to obtain good results (Gu et al., 2016).
該演算法很容易擴展到非常複雜的高維任務，例如具有 21 個動作維度的人形機器人（Humanoid）基準測試（Duan 等人，2016），而在這些任務中，像 DDPG 這樣的異策略方法通常難以獲得良好的結果（Gu 等人，2016）。

SAC also avoids the complexity and potential instability associated with approximate inference in prior off-policy maximum entropy algorithms based on soft Q-learning (Haarnoja et al., 2017).
SAC 還避免了先前基於 soft Q-learning 的異策略最大熵演算法中與近似推斷相關的複雜性和潛在的不穩定性（Haarnoja 等人，2017）。

We present a convergence proof for policy iteration in the maximum entropy framework, and then introduce a new algorithm based on an approximation to this procedure that can be practically implemented with deep neural networks, which we call soft actor-critic.
我們提出了最大熵框架中策略迭代的收斂證明，然後介紹了一種基於此過程近似的新演算法，該演算法可以實際通過深度神經網路實現，我們稱之為 Soft Actor-Critic。

We present empirical results that show that soft actor-critic attains a substantial improvement in both performance and sample efficiency over both off-policy and on-policy prior methods.
我們提出的實證結果表明，Soft Actor-Critic 在性能和樣本效率方面都比先前的異策略和同策略方法有顯著提高。

We also compare to twin delayed deep deterministic (TD3) policy gradient algorithm (Fujimoto et al., 2018), which is a concurrent work that proposes a deterministic algorithm that substantially improves on DDPG.
我們還與雙延遲深度確定性（TD3）策略梯度演算法（Fujimoto 等人，2018）進行了比較，這是一項與我們同時期的工作，提出了一種大幅改進 DDPG 的確定性演算法。

## 2. Related Work
## 2. 相關工作

Our soft actor-critic algorithm incorporates three key ingredients: an actor-critic architecture with separate policy and value function networks, an off-policy formulation that enables reuse of previously collected data for efficiency, and entropy maximization to enable stability and exploration.
我們的 Soft Actor-Critic 演算法結合了三個關鍵要素：具有獨立策略和價值函數網路的演員-評論家架構、能夠重用先前收集的數據以提高效率的異策略公式，以及能夠實現穩定性和探索的熵最大化。

We review prior works that draw on some of these ideas in this section.
我們在本節中回顧了借鑒這些想法的先前工作。

Actor-critic algorithms are typically derived starting from policy iteration, which alternates between policy evaluation—computing the value function for a policy—and policy improvement—using the value function to obtain a better policy (Barto et al., 1983; Sutton & Barto, 1998).
演員-評論家演算法通常源於策略迭代，該迭代在策略評估（計算策略的價值函數）和策略改進（使用價值函數獲得更好的策略）之間交替進行（Barto 等人，1983；Sutton & Barto，1998）。

In large-scale reinforcement learning problems, it is typically impractical to run either of these steps to convergence, and instead the value function and policy are optimized jointly.
在大規模強化學習問題中，運行這些步驟中的任何一個直至收斂通常是不切實際的，取而代之的是聯合優化價值函數和策略。

In this case, the policy is referred to as the actor, and the value function as the critic.
在這種情況下，策略被稱為演員（actor），價值函數被稱為評論家（critic）。

Many actor-critic algorithms build on the standard, on-policy policy gradient formulation to update the actor (Peters & Schaal, 2008), and many of them also consider the entropy of the policy, but instead of maximizing the entropy, they use it as an regularizer (Schulman et al., 2017b; 2015; Mnih et al., 2016; Gruslys et al., 2017).
許多演員-評論家演算法建立在標準的同策略策略梯度公式之上來更新演員（Peters & Schaal，2008），其中許多也考慮了策略的熵，但它們不是最大化熵，而是將其用作正則化項（Schulman 等人，2017b；2015；Mnih 等人，2016；Gruslys 等人，2017）。

On-policy training tends to improve stability but results in poor sample complexity.
同策略訓練傾向於提高穩定性，但導致樣本複雜度較差。

There have been efforts to increase the sample efficiency while retaining robustness by incorporating off-policy samples and by using higher order variance reduction techniques (O’Donoghue et al., 2016; Gu et al., 2016).
已經有人努力通過結合異策略樣本和使用高階方差縮減技術來提高樣本效率，同時保持穩健性（O’Donoghue 等人，2016；Gu 等人，2016）。

However, fully off-policy algorithms still attain better efficiency.
然而，完全異策略演算法仍然能獲得更好的效率。

A particularly popular off-policy actor-critic method, DDPG (Lillicrap et al., 2015), which is a deep variant of the deterministic policy gradient (Silver et al., 2014) algorithm, uses a Q-function estimator to enable off-policy learning, and a deterministic actor that maximizes this Q-function.
一種特別流行的異策略演員-評論家方法 DDPG（Lillicrap 等人，2015），是確定性策略梯度（Silver 等人，2014）演算法的深度變體，它使用 Q 函數估計器來實現異策略學習，並使用確定性演員來最大化此 Q 函數。

As such, this method can be viewed both as a deterministic actor-critic algorithm and an approximate Q-learning algorithm.
因此，該方法既可以視為確定性演員-評論家演算法，也可以視為近似 Q-learning 演算法。

Unfortunately, the interplay between the deterministic actor network and the Q-function typically makes DDPG extremely difficult to stabilize and brittle to hyperparameter settings (Duan et al., 2016; Henderson et al., 2017).
不幸的是，確定性演員網路和 Q 函數之間的相互作用通常使得 DDPG 極難穩定，並且對超參數設置非常脆弱（Duan 等人，2016；Henderson 等人，2017）。

As a consequence, it is difficult to extend DDPG to complex, high-dimensional tasks, and on-policy policy gradient methods still tend to produce the best results in such settings (Gu et al., 2016).
因此，很難將 DDPG 擴展到複雜的高維任務，而在這種環境下，同策略策略梯度方法仍然傾向於產生最佳結果（Gu 等人，2016）。

Our method instead combines off-policy actor-critic training with a stochastic actor, and further aims to maximize the entropy of this actor with an entropy maximization objective.
我們的方法反而是將異策略演員-評論家訓練與隨機演員相結合，並進一步旨在使用熵最大化目標最大化此演員的熵。

We find that this actually results in a considerably more stable and scalable algorithm that, in practice, exceeds both the efficiency and final performance of DDPG.
我們發現這實際上導致了一個更加穩定和可擴展的演算法，實際上，它在效率和最終性能方面都超過了 DDPG。

A similar method can be derived as a zero-step special case of stochastic value gradients (SVG(0)) (Heess et al., 2015).
類似的方法可以推導為隨機價值梯度（SVG(0)）的零步特例（Heess 等人，2015）。

However, SVG(0) differs from our method in that it optimizes the standard maximum expected return objective, and it does not make use of a separate value network, which we found to make training more stable.
然而，SVG(0) 與我們的方法不同之處在於它優化的是標準最大預期回報目標，並且不使用單獨的價值網路，而我們發現單獨的價值網路能使訓練更加穩定。

Maximum entropy reinforcement learning optimizes policies to maximize both the expected return and the expected entropy of the policy.
最大熵強化學習優化策略以最大化預期回報和策略的預期熵。

This framework has been used in many contexts, from inverse reinforcement learning (Ziebart et al., 2008) to optimal control (Todorov, 2008; Toussaint, 2009; Rawlik et al., 2012).
這個框架已被用於許多情境中，從逆向強化學習（Ziebart 等人，2008）到最優控制（Todorov，2008；Toussaint，2009；Rawlik 等人，2012）。

In guided policy search (Levine & Koltun, 2013; Levine et al., 2016), the maximum entropy distribution is used to guide policy learning towards high-reward regions.
在引導式策略搜索（Levine & Koltun，2013；Levine 等人，2016）中，最大熵分佈用於引導策略學習朝向高獎勵區域。

More recently, several papers have noted the connection between Q-learning and policy gradient methods in the framework of maximum entropy learning (O’Donoghue et al., 2016; Haarnoja et al., 2017; Nachum et al., 2017a; Schulman et al., 2017a).
最近，有幾篇論文指出了在最大熵學習框架下 Q-learning 和策略梯度方法之間的聯繫（O’Donoghue 等人，2016；Haarnoja 等人，2017；Nachum 等人，2017a；Schulman 等人，2017a）。

While most of the prior model-free works assume a discrete action space, Nachum et al. (2017b) approximate the maximum entropy distribution with a Gaussian and Haarnoja et al. (2017) with a sampling network trained to draw samples from the optimal policy.
雖然大多數先前的無模型工作假設離散動作空間，但 Nachum 等人 (2017b) 用高斯分佈近似最大熵分佈，而 Haarnoja 等人 (2017) 用訓練來從最優策略中提取樣本的採樣網路進行近似。

Although the soft Q-learning algorithm proposed by Haarnoja et al. (2017) has a value function and actor network, it is not a true actor-critic algorithm: the Q-function is estimating the optimal Q-function, and the actor does not directly affect the Q-function except through the data distribution.
儘管 Haarnoja 等人 (2017) 提出的 soft Q-learning 演算法具有價值函數和演員網路，但它並不是真正的演員-評論家演算法：Q 函數估計的是最優 Q 函數，演員除了透過數據分佈外不直接影響 Q 函數。

Hence, Haarnoja et al. (2017) motivates the actor network as an approximate sampler, rather than the actor in an actor-critic algorithm.
因此，Haarnoja 等人 (2017) 將演員網路定位為近似採樣器，而不是演員-評論家演算法中的演員。

Crucially, the convergence of this method hinges on how well this sampler approximates the true posterior.
至關重要的是，這種方法的收斂性取決於該採樣器逼近真實後驗的程度。

In contrast, we prove that our method converges to the optimal policy from a given policy class, regardless of the policy parameterization.
相比之下，我們證明我們的方法從給定的策略類別收斂到最優策略，而不管策略參數化如何。

Furthermore, these prior maximum entropy methods generally do not exceed the performance of state-of-the-art off-policy algorithms, such as DDPG, when learning from scratch, though they may have other benefits, such as improved exploration and ease of fine-tuning.
此外，這些先前的最大熵方法在從頭開始學習時，通常不會超過最先進的異策略演算法（如 DDPG）的性能，儘管它們可能具有其他好處，例如改進的探索和易於微調。

In our experiments, we demonstrate that our soft actor-critic algorithm does in fact exceed the performance of prior state-of-the-art off-policy deep RL methods by a wide margin.
在我們的實驗中，我們證明我們的 Soft Actor-Critic 演算法實際上大大超過了先前最先進的異策略深度 RL 方法的性能。

## 3. Preliminaries
## 3. 預備知識

We first introduce notation and summarize the standard and maximum entropy reinforcement learning frameworks.
我們先介紹符號，並總結標準和最大熵強化學習框架。

### 3.1. Notation
### 3.1. 符號

We address policy learning in continuous action spaces.
我們解決連續動作空間中的策略學習問題。

We consider an infinite-horizon Markov decision process (MDP), defined by the tuple $(S, A, p, r)$, where the state space $S$ and the action space $A$ are continuous, and the unknown state transition probability $p : S \times S \times A \to [0, \infty)$ represents the probability density of the next state $s_{t+1} \in S$ given the current state $s_t \in S$ and action $a_t \in A$.
我們考慮一個無限視界的馬可夫決策過程（MDP），由元組 $(S, A, p, r)$ 定義，其中狀態空間 $S$ 和動作空間 $A$ 是連續的，未知的狀態轉移概率 $p : S \times S \times A \to [0, \infty)$ 表示在給定當前狀態 $s_t \in S$ 和動作 $a_t \in A$ 的情況下，下一個狀態 $s_{t+1} \in S$ 的概率密度。

The environment emits a bounded reward $r : S \times A \to [r_{\min}, r_{\max}]$ on each transition.
環境在每次轉移時發出一個有界獎勵 $r : S \times A \to [r_{\min}, r_{\max}]$。

We will use $\rho_\pi(s_t)$ and $\rho_\pi(s_t, a_t)$ to denote the state and state-action marginals of the trajectory distribution induced by a policy $\pi(a_t|s_t)$.
我們將使用 $\rho_\pi(s_t)$ 和 $\rho_\pi(s_t, a_t)$ 來表示由策略 $\pi(a_t|s_t)$ 誘導的軌跡分佈的狀態和狀態-動作邊緣分佈。

### 3.2. Maximum Entropy Reinforcement Learning
### 3.2. 最大熵強化學習

Standard RL maximizes the expected sum of rewards $\sum_t \mathbb{E}_{(s_t,a_t)\sim\rho_\pi}[r(s_t, a_t)]$.
標準 RL 最大化獎勵的預期總和 $\sum_t \mathbb{E}_{(s_t,a_t)\sim\rho_\pi}[r(s_t, a_t)]$。

We will consider a more general maximum entropy objective (see e.g. Ziebart (2010)), which favors stochastic policies by augmenting the objective with the expected entropy of the policy over $\rho_\pi(s_t)$:
我們將考慮一個更通用的最大熵目標（見例如 Ziebart (2010)），它透過增加策略在 $\rho_\pi(s_t)$ 上的預期熵來偏好隨機策略：

$$J(\pi) = \sum_{t=0}^T \mathbb{E}_{(s_t,a_t)\sim\rho_\pi} [r(s_t, a_t) + \alpha \mathcal{H}(\pi(\cdot |s_t))] . \quad (1)$$

The temperature parameter $\alpha$ determines the relative importance of the entropy term against the reward, and thus controls the stochasticity of the optimal policy.
溫度參數 $\alpha$ 決定了熵項相對於獎勵的相對重要性，從而控制了最優策略的隨機性。

The maximum entropy objective differs from the standard maximum expected reward objective used in conventional reinforcement learning, though the conventional objective can be recovered in the limit as $\alpha \to 0$.
最大熵目標不同於傳統強化學習中使用的標準最大預期獎勵目標，儘管傳統目標可以在 $\alpha \to 0$ 的極限下恢復。

For the rest of this paper, we will omit writing the temperature explicitly, as it can always be subsumed into the reward by scaling it by $\alpha^{-1}$.
在本文的其餘部分，我們將省略明確寫出溫度，因為它總是可以通過將獎勵縮放 $\alpha^{-1}$ 來包含在獎勵中。

This objective has a number of conceptual and practical advantages.
這個目標具有許多概念和實踐上的優勢。

First, the policy is incentivized to explore more widely, while giving up on clearly unpromising avenues.
首先，策略被激勵去更廣泛地探索，同時放棄明顯沒有希望的途徑。

Second, the policy can capture multiple modes of near-optimal behavior.
其次，策略可以捕捉近乎最優行為的多種模式。

In problem settings where multiple actions seem equally attractive, the policy will commit equal probability mass to those actions.
在多個動作看起來同樣具有吸引力的問題設定中，策略將對這些動作分配相等的概率質量。

Lastly, prior work has observed improved exploration with this objective (Haarnoja et al., 2017; Schulman et al., 2017a), and in our experiments, we observe that it considerably improves learning speed over state-of-art methods that optimize the conventional RL objective function.
最後，先前的工作已經觀察到使用此目標可以改善探索（Haarnoja 等人，2017；Schulman 等人，2017a），並且在我們的實驗中，我們觀察到它比優化傳統 RL 目標函數的最先進方法顯著提高了學習速度。

We can extend the objective to infinite horizon problems by introducing a discount factor $\gamma$ to ensure that the sum of expected rewards and entropies is finite.
我們可以通過引入折扣因子 $\gamma$ 將目標擴展到無限視界問題，以確保預期獎勵和熵的總和是有限的。

Writing down the maximum entropy objective for the infinite horizon discounted case is more involved (Thomas, 2014) and is deferred to Appendix A.
寫下無限視界折扣情況的最大熵目標更為複雜（Thomas，2014），並將推遲到附錄 A。

Prior methods have proposed directly solving for the optimal Q-function, from which the optimal policy can be recovered (Ziebart et al., 2008; Fox et al., 2016; Haarnoja et al., 2017).
先前的方法已經提出直接求解最優 Q 函數，從中可以恢復最優策略（Ziebart 等人，2008；Fox 等人，2016；Haarnoja 等人，2017）。

We will discuss how we can devise a soft actor-critic algorithm through a policy iteration formulation, where we instead evaluate the Q-function of the current policy and update the policy through an off-policy gradient update.
我們將討論如何通過策略迭代公式設計 Soft Actor-Critic 演算法，在該公式中，我們改為評估當前策略的 Q 函數，並通過異策略梯度更新來更新策略。

Though such algorithms have previously been proposed for conventional reinforcement learning, our method is, to our knowledge, the first off-policy actor-critic method in the maximum entropy reinforcement learning framework.
雖然此類演算法先前已針對傳統強化學習提出，但據我們所知，我們的方法是最大熵強化學習框架中的第一個異策略演員-評論家方法。

## 4. From Soft Policy Iteration to Soft Actor-Critic
## 4. 從軟策略迭代到 Soft Actor-Critic

Our off-policy soft actor-critic algorithm can be derived starting from a maximum entropy variant of the policy iteration method.
我們的異策略 Soft Actor-Critic 演算法可以從策略迭代方法的最大熵變體開始推導。

We will first present this derivation, verify that the corresponding algorithm converges to the optimal policy from its density class, and then present a practical deep reinforcement learning algorithm based on this theory.
我們將首先展示這一推導，驗證相應的演算法從其密度類別收斂到最優策略，然後提出基於此理論的實用深度強化學習演算法。

### 4.1. Derivation of Soft Policy Iteration
### 4.1. 軟策略迭代的推導

We will begin by deriving soft policy iteration, a general algorithm for learning optimal maximum entropy policies that alternates between policy evaluation and policy improvement in the maximum entropy framework.
我們將首先推導軟策略迭代，這是一種在最大熵框架中交替進行策略評估和策略改進的學習最優最大熵策略的通用演算法。

Our derivation is based on a tabular setting, to enable theoretical analysis and convergence guarantees, and we extend this method into the general continuous setting in the next section.
我們的推導基於表格設定，以實現理論分析和收斂保證，我們在下一節中將此方法擴展到一般連續設定。

We will show that soft policy iteration converges to the optimal policy within a set of policies which might correspond, for instance, to a set of parameterized densities.
我們將證明軟策略迭代收斂到一組策略內的最優策略，這些策略可能對應於例如一組參數化密度。

In the policy evaluation step of soft policy iteration, we wish to compute the value of a policy $\pi$ according to the maximum entropy objective in Equation 1.
在軟策略迭代的策略評估步驟中，我們希望根據方程式 1 中的最大熵目標計算策略 $\pi$ 的價值。

For a fixed policy, the soft Q-value can be computed iteratively, starting from any function $Q : S \times A \to \mathbb{R}$ and repeatedly applying a modified Bellman backup operator $\mathcal{T}^\pi$ given by
對於固定策略，軟 Q 值可以迭代計算，從任何函數 $Q : S \times A \to \mathbb{R}$ 開始，重複應用修正的貝爾曼備份算子 $\mathcal{T}^\pi$，如下所示

$$\mathcal{T}^\pi Q(s_t, a_t) \triangleq r(s_t, a_t) + \gamma \mathbb{E}_{s_{t+1}\sim p} [V(s_{t+1})], \quad (2)$$

where
其中

$$V(s_t) = \mathbb{E}_{a_t\sim\pi} [Q(s_t, a_t) - \log \pi(a_t|s_t)] \quad (3)$$

is the soft state value function.
是軟狀態價值函數。

We can obtain the soft value function for any policy $\pi$ by repeatedly applying $\mathcal{T}^\pi$ as formalized below.
我們可以通過重複應用 $\mathcal{T}^\pi$ 來獲得任何策略 $\pi$ 的軟價值函數，如下形式化所述。

**Lemma 1 (Soft Policy Evaluation).** Consider the soft Bellman backup operator $\mathcal{T}^\pi$ in Equation 2 and a mapping $Q^0 : S \times A \to \mathbb{R}$ with $|A| < \infty$, and define $Q^{k+1} = \mathcal{T}^\pi Q^k$. Then the sequence $Q^k$ will converge to the soft Q-value of $\pi$ as $k \to \infty$.
**引理 1（軟策略評估）。** 考慮方程式 2 中的軟貝爾曼備份算子 $\mathcal{T}^\pi$ 和映射 $Q^0 : S \times A \to \mathbb{R}$，其中 $|A| < \infty$，並定義 $Q^{k+1} = \mathcal{T}^\pi Q^k$。那麼序列 $Q^k$ 將隨著 $k \to \infty$ 收斂到 $\pi$ 的軟 Q 值。

*Proof.* See Appendix B.1.
*證明。* 見附錄 B.1。

In the policy improvement step, we update the policy towards the exponential of the new Q-function.
在策略改進步驟中，我們將策略更新為朝向新 Q 函數的指數形式。

This particular choice of update can be guaranteed to result in an improved policy in terms of its soft value.
這種特定的更新選擇可以保證在軟價值方面產生改進的策略。

Since in practice we prefer policies that are tractable, we will additionally restrict the policy to some set of policies $\Pi$, which can correspond, for example, to a parameterized family of distributions such as Gaussians.
由於在實踐中我們偏好易於處理的策略，我們將額外限制策略為某組策略 $\Pi$，這可以對應於例如參數化分佈族（如高斯分佈）。

To account for the constraint that $\pi \in \Pi$, we project the improved policy into the desired set of policies.
為了考慮 $\pi \in \Pi$ 的約束，我們將改進的策略投影到所需的策略集中。

While in principle we could choose any projection, it will turn out to be convenient to use the information projection defined in terms of the Kullback-Leibler divergence.
雖然原則上我們可以選擇任何投影，但事實證明使用根據 Kullback-Leibler 散度定義的信息投影會很方便。

In the other words, in the policy improvement step, for each state, we update the policy according to
換句話說，在策略改進步驟中，對於每個狀態，我們根據下式更新策略

$$\pi_{\text{new}} = \arg \min_{\pi' \in \Pi} D_{\text{KL}} \left( \pi'(\cdot|s_t) \bigg\| \frac{\exp(Q^{\pi_{\text{old}}}(s_t, \cdot))}{Z^{\pi_{\text{old}}}(s_t)} \right). \quad (4)$$

The partition function $Z^{\pi_{\text{old}}}(s_t)$ normalizes the distribution, and while it is intractable in general, it does not contribute to the gradient with respect to the new policy and can thus be ignored, as noted in the next section.
配分函數 $Z^{\pi_{\text{old}}}(s_t)$ 對分佈進行歸一化，雖然它通常難以處理，但它對新策略的梯度沒有貢獻，因此可以忽略，如下一節所述。

For this projection, we can show that the new, projected policy has a higher value than the old policy with respect to the objective in Equation 1. We formalize this result in Lemma 2.
對於這個投影，我們可以證明新的、投影後的策略在方程式 1 的目標方面比舊策略具有更高的價值。我們在引理 2 中形式化了這個結果。

**Lemma 2 (Soft Policy Improvement).** Let $\pi_{\text{old}} \in \Pi$ and let $\pi_{\text{new}}$ be the optimizer of the minimization problem defined in Equation 4. Then $Q^{\pi_{\text{new}}}(s_t, a_t) \ge Q^{\pi_{\text{old}}}(s_t, a_t)$ for all $(s_t, a_t) \in S \times A$ with $|A| < \infty$.
**引理 2（軟策略改進）。** 設 $\pi_{\text{old}} \in \Pi$ 且 $\pi_{\text{new}}$ 為方程式 4 中定義的最小化問題的最優解。那麼對於所有 $(s_t, a_t) \in S \times A$ 且 $|A| < \infty$，有 $Q^{\pi_{\text{new}}}(s_t, a_t) \ge Q^{\pi_{\text{old}}}(s_t, a_t)$。

*Proof.* See Appendix B.2.
*證明。* 見附錄 B.2。

The full soft policy iteration algorithm alternates between the soft policy evaluation and the soft policy improvement steps, and it will provably converge to the optimal maximum entropy policy among the policies in $\Pi$ (Theorem 1).
完整的軟策略迭代演算法在軟策略評估和軟策略改進步驟之間交替進行，並且它將可證明地收斂到 $\Pi$ 中策略裡的最優最大熵策略（定理 1）。

Although this algorithm will provably find the optimal solution, we can perform it in its exact form only in the tabular case.
雖然此演算法可證明能找到最優解，但我們只能在表格情況下以其精確形式執行它。

Therefore, we will next approximate the algorithm for continuous domains, where we need to rely on a function approximator to represent the Q-values, and running the two steps until convergence would be computationally too expensive.
因此，我們接下來將針對連續域近似該演算法，在連續域中我們需要依賴函數逼近器來表示 Q 值，並且運行這兩個步驟直到收斂在計算上會過於昂貴。

The approximation gives rise to a new practical algorithm, called soft actor-critic.
這種近似產生了一種新的實用演算法，稱為 Soft Actor-Critic。

**Theorem 1 (Soft Policy Iteration).** Repeated application of soft policy evaluation and soft policy improvement from any $\pi \in \Pi$ converges to a policy $\pi^*$ such that $Q^{\pi^*}(s_t, a_t) \ge Q^\pi(s_t, a_t)$ for all $\pi \in \Pi$ and $(s_t, a_t) \in S \times A$, assuming $|A| < \infty$.
**定理 1（軟策略迭代）。** 從任何 $\pi \in \Pi$ 重複應用軟策略評估和軟策略改進會收斂到策略 $\pi^*$，使得對於所有 $\pi \in \Pi$ 和 $(s_t, a_t) \in S \times A$，有 $Q^{\pi^*}(s_t, a_t) \ge Q^\pi(s_t, a_t)$，假設 $|A| < \infty$。

*Proof.* See Appendix B.3.
*證明。* 見附錄 B.3。

### 4.2. Soft Actor-Critic
### 4.2. Soft Actor-Critic

As discussed above, large continuous domains require us to derive a practical approximation to soft policy iteration.
如上所述，大型連續域要求我們推導軟策略迭代的實用近似。

To that end, we will use function approximators for both the Q-function and the policy, and instead of running evaluation and improvement to convergence, alternate between optimizing both networks with stochastic gradient descent.
為此，我們將對 Q 函數和策略使用函數逼近器，而不是運行評估和改進直到收斂，而是在使用隨機梯度下降優化兩個網路之間交替。

We will consider a parameterized state value function $V_\psi(s_t)$, soft Q-function $Q_\theta(s_t, a_t)$, and a tractable policy $\pi_\phi(a_t|s_t)$.
我們將考慮參數化的狀態價值函數 $V_\psi(s_t)$、軟 Q 函數 $Q_\theta(s_t, a_t)$ 和易於處理的策略 $\pi_\phi(a_t|s_t)$。

The parameters of these networks are $\psi$, $\theta$, and $\phi$.
這些網路的參數是 $\psi$、$\theta$ 和 $\phi$。

For example, the value functions can be modeled as expressive neural networks, and the policy as a Gaussian with mean and covariance given by neural networks.
例如，價值函數可以建模為表現力強的神經網路，策略可以建模為均值和協方差由神經網路給出的高斯分佈。

We will next derive update rules for these parameter vectors.
我們接下來將推導這些參數向量的更新規則。

The state value function approximates the soft value.
狀態價值函數近似軟價值。

There is no need in principle to include a separate function approximator for the state value, since it is related to the Q-function and policy according to Equation 3.
原則上不需要為狀態價值包含單獨的函數逼近器，因為根據方程式 3，它與 Q 函數和策略相關。

This quantity can be estimated from a single action sample from the current policy without introducing a bias, but in practice, including a separate function approximator for the soft value can stabilize training and is convenient to train simultaneously with the other networks.
這個量可以從當前策略的單個動作樣本中估計出來而不引入偏差，但在實踐中，包含一個單獨的軟價值函數逼近器可以穩定訓練，並且便於與其他網路同時訓練。

The soft value function is trained to minimize the squared residual error
軟價值函數被訓練為最小化殘差平方誤差

$$J_V(\psi) = \mathbb{E}_{\mathbf{s}_t \sim \mathcal{D}} \left[ \frac{1}{2} (V_\psi(\mathbf{s}_t) - \mathbb{E}_{\mathbf{a}_t \sim \pi_\phi} [Q_\theta(\mathbf{s}_t, \mathbf{a}_t) - \log \pi_\phi(\mathbf{a}_t|\mathbf{s}_t)])^2 \right] \quad (5)$$

where $\mathcal{D}$ is the distribution of previously sampled states and actions, or a replay buffer.
其中 $\mathcal{D}$ 是先前採樣的狀態和動作的分佈，或重播緩衝區。

The gradient of Equation 5 can be estimated with an unbiased estimator
方程式 5 的梯度可以用無偏估計器估計

$$\hat{\nabla}_\psi J_V(\psi) = \nabla_\psi V_\psi(\mathbf{s}_t) (V_\psi(\mathbf{s}_t) - Q_\theta(\mathbf{s}_t, \mathbf{a}_t) + \log \pi_\phi(\mathbf{a}_t|\mathbf{s}_t)), \quad (6)$$

where the actions are sampled according to the current policy, instead of the replay buffer.
其中動作是根據當前策略採樣的，而不是重播緩衝區。

The soft Q-function parameters can be trained to minimize the soft Bellman residual
軟 Q 函數參數可以被訓練為最小化軟貝爾曼殘差

$$J_Q(\theta) = \mathbb{E}_{(\mathbf{s}_t, \mathbf{a}_t) \sim \mathcal{D}} \left[ \frac{1}{2} (Q_\theta(\mathbf{s}_t, \mathbf{a}_t) - \hat{Q}(\mathbf{s}_t, \mathbf{a}_t))^2 \right], \quad (7)$$

with
其中

$$\hat{Q}(\mathbf{s}_t, \mathbf{a}_t) = r(\mathbf{s}_t, \mathbf{a}_t) + \gamma \mathbb{E}_{\mathbf{s}_{t+1} \sim p} [V_{\bar{\psi}}(\mathbf{s}_{t+1})], \quad (8)$$

which again can be optimized with stochastic gradients
這同樣可以用隨機梯度優化

$$\hat{\nabla}_\theta J_Q(\theta) = \nabla_\theta Q_\theta(\mathbf{a}_t, \mathbf{s}_t) (Q_\theta(\mathbf{s}_t, \mathbf{a}_t) - r(\mathbf{s}_t, \mathbf{a}_t) - \gamma V_{\bar{\psi}}(\mathbf{s}_{t+1})). \quad (9)$$

The update makes use of a target value network $V_{\bar{\psi}}$, where $\bar{\psi}$ can be an exponentially moving average of the value network weights, which has been shown to stabilize training (Mnih et al., 2015).
更新利用了目標價值網路 $V_{\bar{\psi}}$，其中 $\bar{\psi}$ 可以是價值網路權重的指數移動平均值，這已被證明可以穩定訓練（Mnih 等人，2015）。

Alternatively, we can update the target weights to match the current value function weights periodically (see Appendix E).
或者，我們可以定期更新目標權重以匹配當前價值函數權重（見附錄 E）。

Finally, the policy parameters can be learned by directly minimizing the expected KL-divergence in Equation 4:
最後，策略參數可以通過直接最小化方程式 4 中的預期 KL 散度來學習：

$$J_\pi(\phi) = \mathbb{E}_{\mathbf{s}_t \sim \mathcal{D}} \left[ D_{\text{KL}} \left( \pi_\phi(\cdot|\mathbf{s}_t) \bigg\| \frac{\exp(Q_\theta(\mathbf{s}_t, \cdot))}{Z_\theta(\mathbf{s}_t)} \right) \right]. \quad (10)$$

There are several options for minimizing $J_\pi$.
最小化 $J_\pi$ 有幾個選項。

A typical solution for policy gradient methods is to use the likelihood ratio gradient estimator (Williams, 1992), which does not require backpropagating the gradient through the policy and the target density networks.
策略梯度方法的典型解決方案是使用似然比梯度估計器（Williams，1992），它不需要透過策略和目標密度網路反向傳播梯度。

However, in our case, the target density is the Q-function, which is represented by a neural network an can be differentiated, and it is thus convenient to apply the reparameterization trick instead, resulting in a lower variance estimator.
然而，在我們的情況下，目標密度是 Q 函數，它由神經網路表示並且可以微分，因此應用重參數化技巧很方便，從而產生較低方差的估計器。

To that end, we reparameterize the policy using a neural network transformation
為此，我們使用神經網路變換對策略進行重參數化

$$\mathbf{a}_t = f_\phi(\epsilon_t; \mathbf{s}_t), \quad (11)$$

where $\epsilon_t$ is an input noise vector, sampled from some fixed distribution, such as a spherical Gaussian.
其中 $\epsilon_t$ 是輸入噪聲向量，從某個固定分佈（如球形高斯分佈）中採樣。

We can now rewrite the objective in Equation 10 as
我們現在可以將方程式 10 中的目標重寫為

$$J_\pi(\phi) = \mathbb{E}_{\mathbf{s}_t \sim \mathcal{D}, \epsilon_t \sim \mathcal{N}} [\log \pi_\phi(f_\phi(\epsilon_t; \mathbf{s}_t)|\mathbf{s}_t) - Q_\theta(\mathbf{s}_t, f_\phi(\epsilon_t; \mathbf{s}_t))], \quad (12)$$

where $\pi_\phi$ is defined implicitly in terms of $f_\phi$, and we have noted that the partition function is independent of $\phi$ and can thus be omitted.
其中 $\pi_\phi$ 是根據 $f_\phi$ 隱式定義的，我們注意到配分函數與 $\phi$ 無關，因此可以省略。

We can approximate the gradient of Equation 12 with
我們可以用下式近似方程式 12 的梯度

$$\hat{\nabla}_\phi J_\pi(\phi) = \nabla_\phi \log \pi_\phi(\mathbf{a}_t|\mathbf{s}_t) + (\nabla_{\mathbf{a}_t} \log \pi_\phi(\mathbf{a}_t|\mathbf{s}_t) - \nabla_{\mathbf{a}_t} Q(\mathbf{s}_t, \mathbf{a}_t)) \nabla_\phi f_\phi(\epsilon_t; \mathbf{s}_t), \quad (13)$$

where $\mathbf{a}_t$ is evaluated at $f_\phi(\epsilon_t; \mathbf{s}_t)$.
其中 $\mathbf{a}_t$ 在 $f_\phi(\epsilon_t; \mathbf{s}_t)$ 處評估。

This unbiased gradient estimator extends the DDPG style policy gradients (Lillicrap et al., 2015) to any tractable stochastic policy.
這個無偏梯度估計器將 DDPG 風格的策略梯度（Lillicrap 等人，2015）擴展到任何易於處理的隨機策略。

Our algorithm also makes use of two Q-functions to mitigate positive bias in the policy improvement step that is known to degrade performance of value based methods (Hasselt, 2010; Fujimoto et al., 2018).
我們的演算法還利用兩個 Q 函數來減輕策略改進步驟中的正偏差，這種偏差已知會降低基於價值的方法的性能（Hasselt，2010；Fujimoto 等人，2018）。

In particular, we parameterize two Q-functions, with parameters $\theta_i$, and train them independently to optimize $J_Q(\theta_i)$.
具體來說，我們參數化兩個 Q 函數，參數為 $\theta_i$，並獨立訓練它們以優化 $J_Q(\theta_i)$。

We then use the minimum of the Q-functions for the value gradient in Equation 6 and policy gradient in Equation 13, as proposed by Fujimoto et al. (2018).
然後，我們使用 Q 函數的最小值來進行方程式 6 中的價值梯度和方程式 13 中的策略梯度，正如 Fujimoto 等人 (2018) 所提出的。

Although our algorithm can learn challenging tasks, including a 21-dimensional Humanoid, using just a single Q-function, we found two Q-functions significantly speed up training, especially on harder tasks.
雖然我們的演算法僅使用單個 Q 函數就能學習具挑戰性的任務，包括 21 維的人形機器人，但我們發現兩個 Q 函數顯著加快了訓練速度，特別是在更難的任務上。

The complete algorithm is described in Algorithm 1.
完整的演算法描述在演算法 1 中。

The method alternates between collecting experience from the environment with the current policy and updating the function approximators using the stochastic gradients from batches sampled from a replay buffer.
該方法在用當前策略從環境中收集經驗和使用從重播緩衝區採樣的批次隨機梯度更新函數逼近器之間交替進行。

In practice, we take a single environment step followed by one or several gradient steps (see Appendix D).
在實踐中，我們採取單個環境步驟，然後進行一個或多個梯度步驟（見附錄 D）。

**Algorithm 1 Soft Actor-Critic**
**演算法 1 Soft Actor-Critic**

Initialize parameter vectors $\psi, \bar{\psi}, \theta, \phi$.
初始化參數向量 $\psi, \bar{\psi}, \theta, \phi$。

**for** each iteration **do**
**對於** 每個迭代 **做**

&nbsp;&nbsp;&nbsp;&nbsp;**for** each environment step **do**
&nbsp;&nbsp;&nbsp;&nbsp;**對於** 每個環境步驟 **做**

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$\mathbf{a}_t \sim \pi_\phi(\mathbf{a}_t|\mathbf{s}_t)$
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$\mathbf{a}_t \sim \pi_\phi(\mathbf{a}_t|\mathbf{s}_t)$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$\mathbf{s}_{t+1} \sim p(\mathbf{s}_{t+1}|\mathbf{s}_t, \mathbf{a}_t)$
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$\mathbf{s}_{t+1} \sim p(\mathbf{s}_{t+1}|\mathbf{s}_t, \mathbf{a}_t)$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$\mathcal{D} \leftarrow \mathcal{D} \cup \{(\mathbf{s}_t, \mathbf{a}_t, r(\mathbf{s}_t, \mathbf{a}_t), \mathbf{s}_{t+1})\}$
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$\mathcal{D} \leftarrow \mathcal{D} \cup \{(\mathbf{s}_t, \mathbf{a}_t, r(\mathbf{s}_t, \mathbf{a}_t), \mathbf{s}_{t+1})\}$

&nbsp;&nbsp;&nbsp;&nbsp;**end for**
&nbsp;&nbsp;&nbsp;&nbsp;**結束**

&nbsp;&nbsp;&nbsp;&nbsp;**for** each gradient step **do**
&nbsp;&nbsp;&nbsp;&nbsp;**對於** 每個梯度步驟 **做**

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$\psi \leftarrow \psi - \lambda_V \hat{\nabla}_\psi J_V(\psi)$
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$\psi \leftarrow \psi - \lambda_V \hat{\nabla}_\psi J_V(\psi)$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$\theta_i \leftarrow \theta_i - \lambda_Q \hat{\nabla}_{\theta_i} J_Q(\theta_i)$ for $i \in \{1, 2\}$
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;對於 $i \in \{1, 2\}$，$\theta_i \leftarrow \theta_i - \lambda_Q \hat{\nabla
