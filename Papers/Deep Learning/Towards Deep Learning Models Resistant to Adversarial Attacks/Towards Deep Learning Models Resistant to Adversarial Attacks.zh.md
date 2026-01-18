---
title: Towards Deep Learning Models Resistant to Adversarial Attacks
field: Deep_Learning
status: Imported
created_date: 2026-01-18
pdf_link: "[[Towards Deep Learning Models Resistant to Adversarial Attacks.pdf]]"
tags:
  - paper
  - Deep_learning
---

# Abstract
# 摘要

Recent work has demonstrated that deep neural networks are vulnerable to adversarial examples—inputs that are almost indistinguishable from natural data and yet classified incorrectly by the network.
最近的研究表明，深度神經網路容易受到對抗性範例 (adversarial examples) 的攻擊——這些輸入與自然數據幾乎無法區分，但卻會被網路錯誤分類。

In fact, some of the latest findings suggest that the existence of adversarial attacks may be an inherent weakness of deep learning models.
事實上，一些最新的發現表明，對抗性攻擊的存在可能是深度學習模型固有的弱點。

To address this problem, we study the adversarial robustness of neural networks through the lens of robust optimization.
為了解決這個問題，我們透過穩健最佳化 (robust optimization) 的角度來研究神經網路的對抗性穩健性 (adversarial robustness)。

This approach provides us with a broad and unifying view on much of the prior work on this topic.
這種方法為我們提供了一個廣闊且統一的視角，以此審視關於該主題的大量先前工作。

Its principled nature also enables us to identify methods for both training and attacking neural networks that are reliable and, in a certain sense, universal.
其原則性的本質也使我們能夠識別出可靠的且在某種意義上通用的神經網路訓練和攻擊方法。

In particular, they specify a concrete security guarantee that would protect against *any* adversary.
特別是，它們指定了一個具體的安全性保證，可以防禦 *任何* 對手。

These methods let us train networks with significantly improved resistance to a wide range of adversarial attacks.
這些方法讓我們能夠訓練出對廣泛的對抗性攻擊具有顯著提升抵抗力的網路。

They also suggest the notion of security against a *first-order adversary* as a natural and broad security guarantee.
它們還提出將針對 *一階對手 (first-order adversary)* 的安全性概念作為一種自然且廣泛的安全性保證。

We believe that robustness against such well-defined classes of adversaries is an important stepping stone towards fully resistant deep learning models.
我們相信，針對此類定義明確的對手類別的穩健性，是邁向完全抵抗性深度學習模型的重要基石。

# 1 Introduction
# 1 緒論

Recent breakthroughs in computer vision [17, 12] and natural language processing [7] are bringing trained classifiers into the center of security-critical systems.
電腦視覺 [17, 12] 和自然語言處理 [7] 的近期突破，正將訓練好的分類器帶入安全關鍵系統的核心。

Important examples include vision for autonomous cars, face recognition, and malware detection.
重要的例子包括自動駕駛汽車的視覺系統、臉部辨識和惡意軟體檢測。

These developments make security aspects of machine learning increasingly important.
這些發展使得機器學習的安全性方面變得越來越重要。

In particular, resistance to *adversarially chosen inputs* is becoming a crucial design goal.
特別是，對 *對抗性選擇輸入 (adversarially chosen inputs)* 的抵抗力正成為一個關鍵的設計目標。

While trained models tend to be very effective in classifying benign inputs, recent work [2, 28, 22] shows that an adversary is often able to manipulate the input so that the model produces an incorrect output.
雖然訓練好的模型在分類良性輸入方面往往非常有效，但最近的研究 [2, 28, 22] 顯示，對手通常能夠操縱輸入，使模型產生錯誤的輸出。

This phenomenon has received particular attention in the context of deep neural networks, and there is now a quickly growing body of work on this topic [11, 9, 27, 18, 23, 29].
這種現象在深度神經網路的背景下受到了特別關注，現在關於這個主題的研究正迅速增加 [11, 9, 27, 18, 23, 29]。

Computer vision presents a particularly striking challenge: very small changes to the input image can fool state-of-the-art neural networks with high confidence [28, 21].
電腦視覺提出了一個特別驚人的挑戰：對輸入圖像進行非常微小的更改就能以高置信度欺騙最先進的神經網路 [28, 21]。

This holds even when the benign example was classified correctly, and the change is imperceptible to a human.
即使良性範例被正確分類，且這種更改對人類來說是難以察覺的，情況也是如此。

Apart from the security implications, this phenomenon also demonstrates that our current models are not learning the underlying concepts in a robust manner.
除了安全隱含意義外，這種現象還表明我們目前的模型並沒有以穩健的方式學習潛在的概念。

All these findings raise a fundamental question:
所有這些發現都提出了一個基本問題：

*How can we train deep neural networks that are robust to adversarial inputs?*
*我們如何訓練對對抗性輸入具有穩健性的深度神經網路？*

There is now a sizable body of work proposing various attack and defense mechanisms for the adversarial setting.
現在已有大量工作針對對抗性環境提出了各種攻擊和防禦機制。

Examples include defensive distillation [24, 6], feature squeezing [31, 14], and several other adversarial example detection approaches [5].
例子包括防禦性蒸餾 (defensive distillation) [24, 6]、特徵擠壓 (feature squeezing) [31, 14] 以及其他幾種對抗性範例檢測方法 [5]。

These works constitute important first steps in exploring the realm of possibilities here.
這些工作構成了探索此領域可能性的重要第一步。

They, however, do not offer a good understanding of the *guarantees* they provide.
然而，它們並未對其提供的 *保證 (guarantees)* 提供良好的理解。

We can never be certain that a given attack finds the “most adversarial” example in the context, or that a particular defense mechanism prevents the existence of some well-defined *class* of adversarial attacks.
我們永遠無法確定給定的攻擊是否在該情境下找到了「最具對抗性」的範例，或者特定的防禦機制是否阻止了某種定義明確的對抗性攻擊 *類別* 的存在。

This makes it difficult to navigate the landscape of adversarial robustness or to fully evaluate the possible security implications.
這使得我們難以在對抗性穩健性的領域中導航，或充分評估可能的安全隱含意義。

In this paper, we study the adversarial robustness of neural networks through the lens of robust optimization.
在本文中，我們透過穩健最佳化的角度研究神經網路的對抗性穩健性。

We use a natural saddle point (min-max) formulation to capture the notion of security against adversarial attacks in a principled manner.
我們使用自然的鞍點 (saddle point) (min-max) 公式，以原則性的方式捕捉針對對抗性攻擊的安全性概念。

This formulation allows us to be precise about the type of security *guarantee* we would like to achieve, i.e., the broad *class* of attacks we want to be resistant to (in contrast to defending only against specific known attacks).
這個公式使我們能夠精確地說明我們希望實現的安全性 *保證* 類型，即我們希望抵抗的廣泛攻擊 *類別*（與僅防禦特定的已知攻擊相反）。

The formulation also enables us to cast both *attacks* and *defenses* into a common theoretical framework, naturally encapsulating most prior work on adversarial examples.
該公式還使我們能夠將 *攻擊* 和 *防禦* 納入一個共同的理論框架中，自然地涵蓋了大多數關於對抗性範例的先前工作。

In particular, adversarial training directly corresponds to optimizing this saddle point problem.
特別是，對抗性訓練 (adversarial training) 直接對應於最佳化這個鞍點問題。

Similarly, prior methods for attacking neural networks correspond to specific algorithms for solving the underlying constrained optimization problem.
同樣地，先前攻擊神經網路的方法對應於解決潛在約束最佳化問題的特定演算法。

Equipped with this perspective, we make the following contributions.
具備了這個視角，我們做出了以下貢獻。

1. We conduct a careful experimental study of the optimization landscape corresponding to this saddle point formulation.
1. 我們對應於這個鞍點公式的最佳化地景 (optimization landscape) 進行了仔細的實驗研究。

Despite the non-convexity and non-concavity of its constituent parts, we find that the underlying optimization problem *is* tractable after all.
儘管其組成部分具有非凸性和非凹性，但我們發現潛在的最佳化問題終究 *是* 可處理的。

In particular, we provide strong evidence that first-order methods can reliably solve this problem.
特別是，我們提供了強有力的證據，證明一階方法可以可靠地解決這個問題。

We supplement these insights with ideas from real analysis to further motivate projected gradient descent (PGD) as a universal “first-order adversary”, i.e., the strongest attack utilizing the local first order information about the network.
我們用實分析 (real analysis) 的思想補充這些見解，以進一步激發投影梯度下降 (Projected Gradient Descent, PGD) 作為通用的「一階對手」，即利用關於網路的局部一階資訊的最強攻擊。

2. We explore the impact of network architecture on adversarial robustness and find that model capacity plays an important role here.
2. 我們探討了網路架構對對抗性穩健性的影響，並發現模型容量 (model capacity) 在此扮演重要角色。

To reliably withstand strong adversarial attacks, networks require a larger capacity than for correctly classifying benign examples only.
為了可靠地抵禦強大的對抗性攻擊，網路需要比僅正確分類良性範例更大的容量。

This shows that a robust decision boundary of the saddle point problem can be significantly more complicated than a decision boundary that simply separates the benign data points.
這顯示鞍點問題的穩健決策邊界可能比僅分隔良性資料點的決策邊界複雜得多。

3. Building on the above insights, we train networks on MNIST [19] and CIFAR10 [16] that are robust to a wide range of adversarial attacks.
3. 基於上述見解，我們在 MNIST [19] 和 CIFAR10 [16] 上訓練了對廣泛的對抗性攻擊具有穩健性的網路。

Our approach is based on optimizing the aforementioned saddle point formulation and uses PGD as a reliable first-order adversary.
我們的方法基於最佳化上述鞍點公式，並使用 PGD 作為可靠的一階對手。

Our best MNIST model achieves an accuracy of more than 89% against the strongest adversaries in our test suite.
我們最好的 MNIST 模型在我們的測試套件中針對最強對手達到了超過 89% 的準確率。

In particular, our MNIST network is even robust against *white box* attacks of an *iterative* adversary.
特別是，我們的 MNIST 網路甚至對 *迭代* 對手的 *白箱* 攻擊具有穩健性。

Our CIFAR10 model achieves an accuracy of 46% against the same adversary.
我們的 CIFAR10 模型針對同一對手達到了 46% 的準確率。

Furthermore, in case of the weaker *black box/transfer* attacks, our MNIST and CIFAR10 networks achieve the accuracy of more than 95% and 64%, respectively. (More detailed overview can be found in Tables 1 and 2.)
此外，在較弱的 *黑箱/遷移* 攻擊的情況下，我們的 MNIST 和 CIFAR10 網路分別達到了超過 95% 和 64% 的準確率。（更詳細的概述可在表 1 和表 2 中找到。）

To the best of our knowledge, we are the first to achieve these levels of robustness on MNIST and CIFAR10 against such a broad set of attacks.
據我們所知，我們是第一個在 MNIST 和 CIFAR10 上針對如此廣泛的攻擊達到這些穩健性水準的團隊。

Overall, these findings suggest that secure neural networks are within reach.
總體而言，這些發現表明安全的神經網路指日可待。

In order to further support this claim, we invite the community to attempt attacks against our MNIST and CIFAR10 networks in the form of a challenge.
為了進一步支持這一說法，我們邀請社群以挑戰的形式嘗試攻擊我們的 MNIST 和 CIFAR10 網路。

This will let us evaluate its robustness more accurately, and potentially lead to novel attack methods in the process.
這將讓我們更準確地評估其穩健性，並可能在此過程中引發新穎的攻擊方法。

The complete code, along with the description of the challenge, is available at `https://github.com/MadryLab/mnist_challenge` and `https://github.com/MadryLab/cifar10_challenge`.
完整的程式碼以及挑戰的描述可在 `https://github.com/MadryLab/mnist_challenge` 和 `https://github.com/MadryLab/cifar10_challenge` 取得。

# 2 An Optimization View on Adversarial Robustness
# 2 對抗性穩健性的最佳化觀點

Much of our discussion will revolve around an optimization view of adversarial robustness.
我們的大部分討論將圍繞對抗性穩健性的最佳化觀點展開。

This perspective not only captures the phenomena we want to study in a precise manner, but will also inform our investigations.
這個視角不僅以精確的方式捕捉我們想要研究的現象，還將為我們的調查提供資訊。

To this end, let us consider a standard classification task with an underlying data distribution $\mathcal{D}$ over pairs of examples $x \in \mathbb{R}^d$ and corresponding labels $y \in [k]$.
為此，讓我們考慮一個標準分類任務，其具有關於範例對 $x \in \mathbb{R}^d$ 和對應標籤 $y \in [k]$ 的潛在資料分佈 $\mathcal{D}$。

We also assume that we are given a suitable loss function $L(\theta, x, y)$, for instance the cross-entropy loss for a neural network.
我們也假設給定了一個合適的損失函數 $L(\theta, x, y)$，例如神經網路的交叉熵損失 (cross-entropy loss)。

As usual, $\theta \in \mathbb{R}^p$ is the set of model parameters.
像往常一樣，$\theta \in \mathbb{R}^p$ 是模型參數的集合。

Our goal then is to find model parameters $\theta$ that minimize the risk $\mathbb{E}_{(x,y)\sim \mathcal{D}}[L(x, y, \theta)]$.
我們的目標是找到最小化風險 $\mathbb{E}_{(x,y)\sim \mathcal{D}}[L(x, y, \theta)]$ 的模型參數 $\theta$。

Empirical risk minimization (ERM) has been tremendously successful as a recipe for finding classifiers with small population risk.
經驗風險最小化 (Empirical Risk Minimization, ERM) 作為尋找具有小群體風險 (population risk) 的分類器的方法已取得了巨大的成功。

Unfortunately, ERM often does not yield models that are robust to adversarially crafted examples [2, 28].
不幸的是，ERM 通常不會產生對對抗性精心製作的範例具有穩健性的模型 [2, 28]。

Formally, there are efficient algorithms (“adversaries”) that take an example $x$ belonging to class $c_1$ as input and find examples $x^{\text{adv}}$ such that $x^{\text{adv}}$ is very close to $x$ but the model incorrectly classifies $x^{\text{adv}}$ as belonging to class $c_2 \neq c_1$.
形式上，存在高效的演算法（「對手」），它們將屬於類別 $c_1$ 的範例 $x$ 作為輸入，並找到範例 $x^{\text{adv}}$，使得 $x^{\text{adv}}$ 非常接近 $x$，但模型錯誤地將 $x^{\text{adv}}$ 分類為屬於類別 $c_2 \neq c_1$。

In order to *reliably* train models that are robust to adversarial attacks, it is necessary to augment the ERM paradigm appropriately.
為了 *可靠地* 訓練對對抗性攻擊具有穩健性的模型，有必要適當地擴充 ERM 範式。

Instead of resorting to methods that directly focus on improving the robustness to specific attacks, our approach is to first propose a concrete *guarantee* that an adversarially robust model should satisfy.
我們的方法不是訴諸於直接專注於提高對特定攻擊穩健性的方法，而是首先提出一個對抗性穩健模型應該滿足的具體 *保證*。

We then adapt our training methods towards achieving this guarantee.
然後我們調整我們的訓練方法以實現此保證。

The first step towards such a guarantee is to specify an *attack model*, i.e., a precise definition of the attacks our models should be resistant to.
邁向此類保證的第一步是指定一個 *攻擊模型*，即我們模型應該抵抗的攻擊的精確定義。

For each data point $x$, we introduce a set of allowed perturbations $\mathcal{S} \subseteq \mathbb{R}^d$ that formalizes the manipulative power of the adversary.
對於每個資料點 $x$，我們引入一組允許的擾動 $\mathcal{S} \subseteq \mathbb{R}^d$，將對手的操縱能力形式化。

In image classification, we choose $\mathcal{S}$ so that it captures perceptual similarity between images.
在圖像分類中，我們選擇 $\mathcal{S}$ 以捕捉圖像之間的感知相似性。

For instance, the $\ell_\infty$-ball around $x$ has recently been studied as a natural notion for adversarial perturbations [11].
例如，圍繞 $x$ 的 $\ell_\infty$-ball 最近被研究作為對抗性擾動的自然概念 [11]。

While we focus on robustness against $\ell_\infty$-bounded attacks in this paper, we remark that more comprehensive notions of perceptual similarity are an important direction for future research.
雖然本文重點關注針對 $\ell_\infty$ 邊界攻擊的穩健性，但我們注意到更全面的感知相似性概念是未來研究的重要方向。

Next, we modify the definition of population risk $\mathbb{E}_{\mathcal{D}}[L]$ by incorporating the above adversary.
接下來，我們透過結合上述對手來修改群體風險 $\mathbb{E}_{\mathcal{D}}[L]$ 的定義。

Instead of feeding samples from the distribution $\mathcal{D}$ directly into the loss $L$, we allow the adversary to perturb the input first.
我們不將來自分佈 $\mathcal{D}$ 的樣本直接饋送到損失 $L$ 中，而是允許對手先擾動輸入。

This gives rise to the following saddle point problem, which is our central object of study:
這產生了以下鞍點問題，這是我們研究的核心對象：

$$
\min_{\theta} \rho(\theta), \quad \text{where} \quad \rho(\theta) = \mathbb{E}_{(x,y)\sim \mathcal{D}} \left[ \max_{\delta \in \mathcal{S}} L(\theta, x + \delta, y) \right]. \quad (2.1)
$$

Formulations of this type (and their finite-sample counterparts) have a long history in robust optimization, going back to Wald [30].
這種類型的公式（及其有限樣本對應物）在穩健最佳化方面有著悠久的歷史，可以追溯到 Wald [30]。

It turns out that this formulation is also particularly useful in our context.
事實證明，這個公式在我們的背景下也特別有用。

First, this formulation gives us a unifying perspective that encompasses much prior work on adversarial robustness.
首先，這個公式為我們提供了一個統一的視角，涵蓋了許多關於對抗性穩健性的先前工作。

Our perspective stems from viewing the saddle point problem as the composition of an *inner maximization* problem and an *outer minimization* problem.
我們的觀點源於將鞍點問題視為 *內部最大化* 問題和 *外部最小化* 問題的組合。

Both of these problems have a natural interpretation in our context.
這兩個問題在我們的背景下都有自然的解釋。

The inner maximization problem aims to find an adversarial version of a given data point $x$ that achieves a high loss.
內部最大化問題旨在找到給定資料點 $x$ 的對抗性版本，以達到高損失。

This is precisely the problem of attacking a given neural network.
這正是攻擊給定神經網路的問題。

On the other hand, the goal of the outer minimization problem is to find model parameters so that the “adversarial loss” given by the inner attack problem is minimized.
另一方面，外部最小化問題的目標是找到模型參數，使內部攻擊問題給出的「對抗性損失」最小化。

This is precisely the problem of training a robust classifier using adversarial training techniques.
這正是使用對抗性訓練技術訓練穩健分類器的問題。

Second, the saddle point problem specifies a clear goal that an ideal robust classifier should achieve, as well as a quantitative measure of its robustness.
其次，鞍點問題指定了一個理想的穩健分類器應該達到的明確目標，以及其穩健性的定量度量。

In particular, when the parameters $\theta$ yield a (nearly) vanishing risk, the corresponding model is perfectly robust to attacks specified by our attack model.
特別是，當參數 $\theta$ 產生（幾乎）消失的風險時，相應的模型對我們的攻擊模型指定的攻擊具有完全的穩健性。

Our paper investigates the structure of this saddle point problem in the context of deep neural networks.
我們的論文在深度神經網路的背景下調查了這個鞍點問題的結構。

These investigations then lead us to training techniques that produce models with high resistance to a wide range of adversarial attacks.
這些調查隨後引導我們找到了訓練技術，這些技術產生的模型對廣泛的對抗性攻擊具有高度抵抗力。

Before turning to our contributions, we briefly review prior work on adversarial examples and describe in more detail how it fits into the above formulation.
在轉向我們的貢獻之前，我們先簡要回顧一下關於對抗性範例的先前工作，並更詳細地描述它如何適應上述公式。

## 2.1 A Unified View on Attacks and Defenses
## 2.1 攻擊與防禦的統一觀點

Prior work on adversarial examples has focused on two main questions:
先前關於對抗性範例的研究主要集中在兩個問題上：

1. How can we produce strong adversarial examples, i.e., adversarial examples that fool a model with high confidence while requiring only a small perturbation?
1. 我們如何產生強大的對抗性範例，即需要很小的擾動就能以高置信度欺騙模型的對抗性範例？

2. How can we train a model so that there are no adversarial examples, or at least so that an adversary cannot find them easily?
2. 我們如何訓練一個模型，使其沒有對抗性範例，或者至少讓對手無法輕易找到它們？

Our perspective on the saddle point problem (2.1) gives answers to both these questions.
我們對鞍點問題 (2.1) 的看法給出了這兩個問題的答案。

On the attack side, prior work has proposed methods such as the Fast Gradient Sign Method (FGSM) [11] and multiple variations of it [18].
在攻擊方面，先前的工作提出了諸如快速梯度符號法 (Fast Gradient Sign Method, FGSM) [11] 及其多種變體 [18] 等方法。

FGSM is an attack for an $\ell_\infty$-bounded adversary and computes an adversarial example as
FGSM 是一種針對 $\ell_\infty$ 邊界對手的攻擊，並計算對抗性範例如下

$$
x + \varepsilon \text{sgn}(\nabla_x L(\theta, x, y)).
$$

One can interpret this attack as a simple one-step scheme for maximizing the inner part of the saddle point formulation.
我們可以將這種攻擊解釋為最大化鞍點公式內部部分的簡單一步方案。

A more powerful adversary is the multi-step variant, which is essentially projected gradient descent (PGD) on the negative loss function
更強大的對手是多步變體，它本質上是負損失函數上的投影梯度下降 (PGD)

$$
x^{t+1} = \Pi_{x+\mathcal{S}} \left( x^t + \alpha \text{sgn}(\nabla_x L(\theta, x, y)) \right).
$$

Other methods like FGSM with random perturbation have also been proposed [29].
其他方法如帶有隨機擾動的 FGSM 也被提出 [29]。

Clearly, all of these approaches can be viewed as specific attempts to solve the inner maximization problem in (2.1).
顯然，所有這些方法都可以被視為解決 (2.1) 中內部最大化問題的具體嘗試。

On the defense side, the training dataset is often augmented with adversarial examples produced by FGSM.
在防禦方面，訓練資料集通常會增加由 FGSM 產生的對抗性範例。

This approach also directly follows from (2.1) when linearizing the inner maximization problem.
當將內部最大化問題線性化時，這種方法也直接遵循 (2.1)。

To solve the simplified robust optimization problem, we replace every training example with its FGSM-perturbed counterpart.
為了解決簡化的穩健最佳化問題，我們將每個訓練範例替換為其 FGSM 擾動的對應物。

More sophisticated defense mechanisms such as training against multiple adversaries can be seen as better, more exhaustive approximations of the inner maximization problem.
更複雜的防禦機制，如針對多個對手進行訓練，可以被視為內部最大化問題的更好、更詳盡的近似。
