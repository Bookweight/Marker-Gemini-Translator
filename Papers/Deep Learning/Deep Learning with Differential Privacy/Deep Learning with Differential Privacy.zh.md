---
title: Deep Learning with Differential Privacy
field: Deep_Learning
status: Imported
created_date: 2026-01-19
pdf_link: "[[Deep Learning with Differential Privacy.pdf]]"
tags:
  - paper
  - Deep_learning
---

# Deep Learning with Differential Privacy
# 具有差分隱私的深度學習

**October 25, 2016**
**2016年10月25日**

Martín Abadi, Andy Chu, Ian Goodfellow, H. Brendan McMahan, Ilya Mironov, Kunal Talwar, Li Zhang

## ABSTRACT
## 摘要

Machine learning techniques based on neural networks are achieving remarkable results in a wide variety of domains.
基於神經網絡的機器學習技術在各個領域都取得了顯著的成果。

Often, the training of models requires large, representative datasets, which may be crowdsourced and contain sensitive information.
通常，模型的訓練需要大型、具有代表性的數據集，這些數據集可能是眾包的，並且包含敏感信息。

The models should not expose private information in these datasets.
模型不應暴露這些數據集中的隱私信息。

Addressing this goal, we develop new algorithmic techniques for learning and a refined analysis of privacy costs within the framework of differential privacy.
為了實現這一目標，我們開發了新的學習算法技術，並在差分隱私的框架內對隱私成本進行了精細的分析。

Our implementation and experiments demonstrate that we can train deep neural networks with non-convex objectives, under a modest privacy budget, and at a manageable cost in software complexity, training efficiency, and model quality.
我們的實作和實驗表明，我們可以在適度的隱私預算下，以可控的軟件複雜度、訓練效率和模型質量成本，訓練具有非凸目標函數的深度神經網絡。

## 1. INTRODUCTION
## 1. 介紹

Recent progress in neural networks has led to impressive successes in a wide range of applications, including image classification, language representation, move selection for Go, and many more.
神經網絡的最新進展在廣泛的應用中取得了令人印象深刻的成功，包括圖像分類、語言表示、圍棋的著法選擇等等。

These advances are enabled, in part, by the availability of large and representative datasets for training neural networks.
這些進步部分歸功於可用於訓練神經網絡的大型且具代表性的數據集。

These datasets are often crowdsourced, and may contain sensitive information.
這些數據集通常是眾包的，並且可能包含敏感信息。

Their use requires techniques that meet the demands of the applications while offering principled and rigorous privacy guarantees.
使用這些數據需要既能滿足應用需求，又能提供原則性且嚴格的隱私保證的技術。

In this paper, we combine state-of-the-art machine learning methods with advanced privacy-preserving mechanisms, training neural networks within a modest (“single-digit”) privacy budget.
在本文中，我們將最先進的機器學習方法與先進的隱私保護機制相結合，在適度（「個位數」）的隱私預算內訓練神經網絡。

We treat models with non-convex objectives, several layers, and tens of thousands to millions of parameters.
我們處理具有非凸目標函數、多層結構以及數萬到數百萬個參數的模型。

(In contrast, previous work obtains strong results on convex models with smaller numbers of parameters, or treats complex neural networks but with a large privacy loss.)
（相比之下，以前的工作要在參數較少的凸模型上獲得強大的結果，或者雖然處理複雜的神經網絡但伴隨著巨大的隱私損失。）

For this purpose, we develop new algorithmic techniques, a refined analysis of privacy costs within the framework of differential privacy, and careful implementation strategies:
為此，我們開發了新的算法技術，在差分隱私框架內對隱私成本進行了精細分析，並制定了謹慎的實作策略：

1. We demonstrate that, by tracking detailed information (higher moments) of the privacy loss, we can obtain much tighter estimates on the overall privacy loss, both asymptotically and empirically.
1. 我們證明，通過追蹤隱私損失的詳細信息（高階矩），我們可以在漸近和經驗上獲得對整體隱私損失更緊密的估計。

2. We improve the computational efficiency of differentially private training by introducing new techniques. These techniques include efficient algorithms for computing gradients for individual training examples, subdividing tasks into smaller batches to reduce memory footprint, and applying differentially private principal projection at the input layer.
2. 我們通過引入新技術提高了差分隱私訓練的計算效率。這些技術包括計算單個訓練樣本梯度的有效算法、將任務細分為更小的批次以減少內存佔用，以及在輸入層應用差分隱私主成分投影。

3. We build on the machine learning framework TensorFlow [3] for training models with differential privacy. We evaluate our approach on two standard image classification tasks, MNIST and CIFAR-10.
3. 我們基於機器學習框架 TensorFlow [3] 構建了用於訓練差分隱私模型的系統。我們在兩個標準圖像分類任務 MNIST 和 CIFAR-10 上評估了我們的方法。

We chose these two tasks because they are based on public datasets and have a long record of serving as benchmarks in machine learning.
我們選擇這兩個任務是因為它們基於公共數據集，並且長期以來一直作為機器學習的基準。

Our experience indicates that privacy protection for deep neural networks can be achieved at a modest cost in software complexity, training efficiency, and model quality.
我們的經驗表明，深度神經網絡的隱私保護可以在軟件複雜度、訓練效率和模型質量方面以適度的成本實現。

Machine learning systems often comprise elements that contribute to protecting their training data.
機器學習系統通常包含有助於保護其訓練數據的元素。

In particular, regularization techniques, which aim to avoid overfitting to the examples used for training, may hide details of those examples.
特別是正則化技術，其目的是避免對用於訓練的樣本過度擬合，這可能會隱藏這些樣本的細節。

On the other hand, explaining the internal representations in deep neural networks is notoriously difficult, and their large capacity entails that these representations may potentially encode fine details of at least some of the training data.
另一方面，解釋深度神經網絡中的內部表示是出了名的困難，而且其巨大的容量意味著這些表示可能潛在地編碼至少部分訓練數據的精細細節。

In some cases, a determined adversary may be able to extract parts of the training data.
在某些情況下，堅定的攻擊者可能有能力提取部分訓練數據。

For example, Fredrikson et al. demonstrated a model-inversion attack that recovers images from a facial recognition system [24].
例如，Fredrikson 等人展示了一種模型反轉攻擊，可以從面部識別系統中恢復圖像 [24]。

While the model-inversion attack requires only “blackbox” access to a trained model (that is, interaction with the model via inputs and outputs), we consider adversaries with additional capabilities, much like Shokri and Shmatikov [50].
雖然模型反轉攻擊僅需要對訓練模型進行「黑盒」訪問（即通過輸入和輸出與模型交互），但我們考慮具有額外能力的對手，這與 Shokri 和 Shmatikov [50] 非常相似。

Our approach offers protection against a strong adversary with full knowledge of the training mechanism and access to the model’s parameters.
我們的方法針對完全了解訓練機制並能訪問模型參數的強大對手提供保護。

This protection is attractive, in particular, for applications of machine learning on mobile phones, tablets, and other devices.
這種保護特別適用於手機、平板電腦和其他設備上的機器學習應用。

Storing models on-device enables power-efficient, low-latency inference, and may contribute to privacy since inference does not require communicating user data to a central server; on the other hand, we must assume that the model parameters themselves may be exposed to hostile inspection.
在設備上存儲模型可以實現高能效、低延遲的推論，並且可能有助於隱私保護，因為推論不需要將用戶數據傳輸到中央服務器；另一方面，我們必須假設模型參數本身可能會暴露給惡意的檢查。

Furthermore, when we are concerned with preserving the privacy of one record in the training data, we allow for the possibility that the adversary controls some or even all of the rest of the training data.
此外，當我們關注保護訓練數據中一條記錄的隱私時，我們允許對手控制其餘部分甚至全部訓練數據的可能性。

In practice, this possibility cannot always be excluded, for example when the data is crowdsourced.
在實踐中，這種可能性不能總是被排除，例如當數據是眾包的時候。

The next section reviews background on deep learning and on differential privacy. Sections 3 and 4 explain our approach and implementation. Section 5 describes our experimental results. Section 6 discusses related work, and Section 7 concludes. Deferred proofs appear in the Appendix.
下一節回顧深度學習和差分隱私的背景。第 3 節和第 4 節解釋我們的方法和實作。第 5 節描述我們的實驗結果。第 6 節討論相關工作，第 7 節總結。推遲的證明出現在附錄中。

## 2. BACKGROUND
## 2. 背景

In this section we briefly recall the definition of differential privacy, introduce the Gaussian mechanism and composition theorems, and overview basic principles of deep learning.
在本節中，我們簡要回顧差分隱私的定義，介紹高斯機制和組合定理，並概述深度學習的基本原理。

### 2.1 Differential Privacy
### 2.1 差分隱私

Differential privacy [19, 16, 20] constitutes a strong standard for privacy guarantees for algorithms on aggregate databases.
差分隱私 [19, 16, 20] 構成了聚合數據庫算法隱私保證的強大標準。

It is defined in terms of the application-specific concept of adjacent databases.
它是根據特定於應用的「相鄰數據庫」概念來定義的。

In our experiments, for instance, each training dataset is a set of image-label pairs; we say that two of these sets are adjacent if they differ in a single entry, that is, if one image-label pair is present in one set and absent in the other.
例如，在我們的實驗中，每個訓練數據集都是一組圖像-標籤對；我們說如果這兩組數據集僅在一個條目上不同，即一個圖像-標籤對存在於一組中而不存在於另一組中，則它們是相鄰的。

**Definition 1.** A randomized mechanism $\mathcal{M}: \mathcal{D} \rightarrow \mathcal{R}$ with domain $\mathcal{D}$ and range $\mathcal{R}$ satisfies $(\varepsilon, \delta)$-differential privacy if for any two adjacent inputs $d, d' \in \mathcal{D}$ and for any subset of outputs $S \subseteq \mathcal{R}$ it holds that
**定義 1.** 一個隨機機制 $\mathcal{M}: \mathcal{D} \rightarrow \mathcal{R}$，其定義域為 $\mathcal{D}$，值域為 $\mathcal{R}$，如果對於任意兩個相鄰輸入 $d, d' \in \mathcal{D}$ 以及輸出的任意子集 $S \subseteq \mathcal{R}$，滿足以下條件，則稱其滿足 $(\varepsilon, \delta)$-差分隱私：

$$Pr[\mathcal{M}(d) \in S] \le e^{\varepsilon} Pr[\mathcal{M}(d') \in S] + \delta.$$

The original definition of $\varepsilon$-differential privacy does not include the additive term $\delta$.
$\varepsilon$-差分隱私的原始定義不包括加法項 $\delta$。

We use the variant introduced by Dwork et al. [17], which allows for the possibility that plain $\varepsilon$-differential privacy is broken with probability $\delta$ (which is preferably smaller than $1/|d|$).
我們使用 Dwork 等人 [17] 引入的變體，它允許普通的 $\varepsilon$-差分隱私以 $\delta$ 的概率被打破（$\delta$ 最好小於 $1/|d|$）。

Differential privacy has several properties that make it particularly useful in applications such as ours: composability, group privacy, and robustness to auxiliary information.
差分隱私具有幾個特性，使其特別適用於我們的應用：可組合性、群體隱私以及對輔助信息的魯棒性。

Composability enables modular design of mechanisms: if all the components of a mechanism are differentially private, then so is their composition.
可組合性使得機制的模塊化設計成為可能：如果一個機制的所有組件都是差分隱私的，那麼它們的組合也是如此。

Group privacy implies graceful degradation of privacy guarantees if datasets contain correlated inputs, such as the ones contributed by the same individual.
群體隱私意味著如果數據集包含相關的輸入（例如由同一個人貢獻的輸入），隱私保證會優雅地降級。

Robustness to auxiliary information means that privacy guarantees are not affected by any side information available to the adversary.
對輔助信息的魯棒性意味著隱私保證不受對手可獲得的任何邊信信息的影響。

A common paradigm for approximating a deterministic real-valued function $f : D \rightarrow R$ with a differentially private mechanism is via additive noise calibrated to $f$'s sensitivity $S_f$, which is defined as the maximum of the absolute distance $|f(d) - f(d')|$ where $d$ and $d'$ are adjacent inputs.
用差分隱私機制逼近確定性實值函數 $f : D \rightarrow R$ 的一個常見範式是通過校準到 $f$ 的敏感度 $S_f$ 的加性噪聲，敏感度定義為最大絕對距離 $|f(d) - f(d')|$，其中 $d$ 和 $d'$ 是相鄰輸入。

(The restriction to a real-valued function is intended to simplify this review, but is not essential.)
（限制為實值函數是為了簡化本回顧，但並非必要。）

For instance, the Gaussian noise mechanism is defined by
例如，高斯噪聲機制定義為：

$$\mathcal{M}(d) \triangleq f(d) + \mathcal{N}(0, S_f^2 \cdot \sigma^2),$$

where $\mathcal{N}(0, S_f^2 \cdot \sigma^2)$ is the normal (Gaussian) distribution with mean 0 and standard deviation $S_f\sigma$.
其中 $\mathcal{N}(0, S_f^2 \cdot \sigma^2)$ 是均值為 0 標準差為 $S_f\sigma$ 的正態（高斯）分佈。

A single application of the Gaussian mechanism to function $f$ of sensitivity $S_f$ satisfies $(\varepsilon, \delta)$-differential privacy if $\delta \ge \frac{4}{5} \exp(-(\sigma\varepsilon)^2/2)$ and $\varepsilon < 1$ [20, Theorem 3.22].
如果 $\delta \ge \frac{4}{5} \exp(-(\sigma\varepsilon)^2/2)$ 且 $\varepsilon < 1$，則對敏感度為 $S_f$ 的函數 $f$ 單次應用高斯機制滿足 $(\varepsilon, \delta)$-差分隱私 [20, 定理 3.22]。

Note that this analysis of the mechanism can be applied *post hoc*, and, in particular, that there are infinitely many $(\varepsilon, \delta)$ pairs that satisfy this condition.
請注意，對該機制的這種分析可以 *事後* 應用，特別是，有無限多個 $(\varepsilon, \delta)$ 對滿足此條件。

Differential privacy for repeated applications of additive noise mechanisms follows from the basic composition theorem [17, 18], or from advanced composition theorems and their refinements [22, 32, 21, 10].
重複應用加性噪聲機制的差分隱私遵循基本組合定理 [17, 18]，或高級組合定理及其改進 [22, 32, 21, 10]。

The task of keeping track of the accumulated privacy loss in the course of execution of a composite mechanism, and enforcing the applicable privacy policy, can be performed by the *privacy accountant*, introduced by McSherry [40].
在複合機制的執行過程中追蹤累積隱私損失，並執行適用的隱私策略的任務，可以由 McSherry [40] 引入的 *隱私會計師 (privacy accountant)* 來執行。

The basic blueprint for designing a differentially private additive-noise mechanism that implements a given functionality consists of the following steps: approximating the functionality by a sequential composition of bounded-sensitivity functions; choosing parameters of additive noise; and performing privacy analysis of the resulting mechanism. We follow this approach in Section 3.
設計實現給定功能的差分隱私加性噪聲機制的基本藍圖包括以下步驟：通過有界敏感度函數的順序組合來逼近該功能；選擇加性噪聲的參數；並對生成的機制進行隱私分析。我們在第 3 節中遵循此方法。

### 2.2 Deep Learning
### 2.2 深度學習

Deep neural networks, which are remarkably effective for many machine learning tasks, define parameterized functions from inputs to outputs as compositions of many layers of basic building blocks, such as affine transformations and simple nonlinear functions.
深度神經網絡在許多機器學習任務中非常有效，它將從輸入到輸出的參數化函數定義為許多層基本構建模塊（如仿射變換和簡單非線性函數）的組合。

Commonly used examples of the latter are sigmoids and rectified linear units (ReLUs).
後者的常用例子是 Sigmoid 和整流線性單元 (ReLU)。

By varying parameters of these blocks, we can “train” such a parameterized function with the goal of fitting any given finite set of input/output examples.
通過改變這些模塊的參數，我們可以「訓練」這樣一個參數化函數，目標是擬合任何給定的有限輸入/輸出樣本集。

More precisely, we define a loss function $\mathcal{L}$ that represents the penalty for mismatching the training data.
更準確地說，我們定義了一個損失函數 $\mathcal{L}$，代表對訓練數據不匹配的懲罰。

The loss $\mathcal{L}(\theta)$ on parameters $\theta$ is the average of the loss over the training examples $\{x_1, \dots, x_N\}$, so $\mathcal{L}(\theta) = \frac{1}{N} \sum_i \mathcal{L}(\theta, x_i)$.
參數 $\theta$ 上的損失 $\mathcal{L}(\theta)$ 是訓練樣本 $\{x_1, \dots, x_N\}$ 上的損失平均值，因此 $\mathcal{L}(\theta) = \frac{1}{N} \sum_i \mathcal{L}(\theta, x_i)$。

Training consists in finding $\theta$ that yields an acceptably small loss, hopefully the smallest loss (though in practice we seldom expect to reach an exact global minimum).
訓練在於尋找能產生可接受的小損失的 $\theta$，希望是最小損失（儘管在實踐中我們很少期望達到精確的全局最小值）。

For complex networks, the loss function $\mathcal{L}$ is usually non-convex and difficult to minimize.
對於複雜的網絡，損失函數 $\mathcal{L}$ 通常是非凸的，難以最小化。

In practice, the minimization is often done by the mini-batch stochastic gradient descent (SGD) algorithm.
在實踐中，最小化通常通過小批量隨機梯度下降 (SGD) 算法來完成。

In this algorithm, at each step, one forms a batch $B$ of random examples and computes $\mathbf{g}_B = 1/|B| \sum_{x \in B} \nabla_\theta \mathcal{L}(\theta, x)$ as an estimation to the gradient $\nabla_\theta \mathcal{L}(\theta)$.
在該算法中，每一步都會形成一個隨機樣本的批次 $B$，並計算 $\mathbf{g}_B = 1/|B| \sum_{x \in B} \nabla_\theta \mathcal{L}(\theta, x)$ 作為梯度 $\nabla_\theta \mathcal{L}(\theta)$ 的估計。

Then $\theta$ is updated following the gradient direction $-\mathbf{g}_B$ towards a local minimum.
然後沿著梯度方向 $-\mathbf{g}_B$ 更新 $\theta$，朝向局部最小值。

Several systems have been built to support the definition of neural networks, to enable efficient training, and then to perform efficient inference (execution for fixed parameters) [29, 12, 3].
已經建立了多個系統來支持神經網絡的定義，以實現高效訓練，然後執行高效推論（固定參數的執行）[29, 12, 3]。

We base our work on TensorFlow, an open-source dataflow engine released by Google [3].
我們的工作基於 TensorFlow，這是 Google 發布的一個開源數據流引擎 [3]。

TensorFlow allows the programmer to define large computation graphs from basic operators, and to distribute their execution across a heterogeneous distributed system.
TensorFlow 允許程序員從基本運算符定義大型計算圖，並在異構分佈式系統中分發執行。

TensorFlow automates the creation of the computation graphs for gradients; it also makes it easy to batch computation.
TensorFlow 自動創建梯度的計算圖；它還使得批量計算變得容易。

## 3. OUR APPROACH
## 3. 我們的方法

This section describes the main components of our approach toward differentially private training of neural networks: a differentially private stochastic gradient descent (SGD) algorithm, the moments accountant, and hyperparameter tuning.
本節描述了我們實現神經網絡差分隱私訓練方法的主要組成部分：差分隱私隨機梯度下降 (SGD) 算法、矩會計師 (moments accountant) 和超參數調整。

### 3.1 Differentially Private SGD Algorithm
### 3.1 差分隱私 SGD 算法

One might attempt to protect the privacy of training data by working only on the final parameters that result from the training process, treating this process as a black box.
人們可能會嘗試僅通過處理訓練過程產生的最終參數來保護訓練數據的隱私，將此過程視為黑盒。

Unfortunately, in general, one may not have a useful, tight characterization of the dependence of these parameters on the training data; adding overly conservative noise to the parameters, where the noise is selected according to the worst-case analysis, would destroy the utility of the learned model.
不幸的是，一般來說，人們可能無法對這些參數與訓練數據的依賴關係進行有用、緊密的描述；向參數添加過於保守的噪聲（根據最壞情況分析選擇噪聲）將破壞學習模型的效用。

Therefore, we prefer a more sophisticated approach in which we aim to control the influence of the training data during the training process, specifically in the SGD computation.
因此，我們傾向於一種更複雜的方法，即在訓練過程中，特別是在 SGD 計算中，控制訓練數據的影響。

This approach has been followed in previous works (e.g., [52, 7]); we make several modifications and extensions, in particular in our privacy accounting.
這種方法已被以前的工作採用（例如，[52, 7]）；我們進行了一些修改和擴展，特別是在我們的隱私核算方面。

Algorithm 1 outlines our basic method for training a model with parameters $\theta$ by minimizing the empirical loss function $\mathcal{L}(\theta)$.
算法 1 概述了我們通過最小化經驗損失函數 $\mathcal{L}(\theta)$ 來訓練參數為 $\theta$ 的模型的基本方法。

At each step of the SGD, we compute the gradient $\nabla_\theta \mathcal{L}(\theta, x_i)$ for a random subset of examples, clip the $\ell_2$ norm of each gradient, compute the average, add noise in order to protect privacy, and take a step in the opposite direction of this average noisy gradient.
在 SGD 的每一步中，我們計算隨機樣本子集的梯度 $\nabla_\theta \mathcal{L}(\theta, x_i)$，裁剪每個梯度的 $\ell_2$ 範數，計算平均值，添加噪聲以保護隱私，並沿著該平均噪聲梯度的反方向邁出一步。

At the end, in addition to outputting the model, we will also need to compute the privacy loss of the mechanism based on the information maintained by the privacy accountant. Next we describe in more detail each component of this algorithm and our refinements.
最後，除了輸出模型外，我們還需要根據隱私會計師維護的信息計算機制的隱私損失。接下來我們將更詳細地描述該算法的每個組成部分以及我們的改進。

**Algorithm 1 Differentially private SGD (Outline)**
**算法 1 差分隱私 SGD（大綱）**

**Input:** Examples $\{x_1, \dots, x_N\}$, loss function $\mathcal{L}(\theta) = \frac{1}{N} \sum_i \mathcal{L}(\theta, x_i)$. Parameters: learning rate $\eta_t$, noise scale $\sigma$, group size $L$, gradient norm bound $C$.
**輸入：** 樣本 $\{x_1, \dots, x_N\}$，損失函數 $\mathcal{L}(\theta) = \frac{1}{N} \sum_i \mathcal{L}(\theta, x_i)$。參數：學習率 $\eta_t$，噪聲尺度 $\sigma$，組大小 $L$，梯度範數界限 $C$。

**Initialize** $\theta_0$ randomly
**初始化** $\theta_0$ 隨機

**for** $t \in [T]$ **do**
**對於** $t \in [T]$ **執行**
  Take a random sample $L_t$ with sampling probability $L/N$
  以採樣概率 $L/N$ 抽取隨機樣本 $L_t$

  **Compute gradient**
  **計算梯度**
  For each $i \in L_t$, compute $\mathbf{g}_t(x_i) \leftarrow \nabla_{\theta_t}\mathcal{L}(\theta_t, x_i)$
  對於每個 $i \in L_t$，計算 $\mathbf{g}_t(x_i) \leftarrow \nabla_{\theta_t}\mathcal{L}(\theta_t, x_i)$

  **Clip gradient**
  **裁剪梯度**
  $\bar{\mathbf{g}}_t(x_i) \leftarrow \mathbf{g}_t(x_i) / \max(1, \frac{||\mathbf{g}_t(x_i)||_2}{C})$

  **Add noise**
  **添加噪聲**
  $\tilde{\mathbf{g}}_t \leftarrow \frac{1}{L} (\sum_i \bar{\mathbf{g}}_t(x_i) + \mathcal{N}(0, \sigma^2 C^2 \mathbf{I}))$

  **Descent**
  **下降**
  $\theta_{t+1} \leftarrow \theta_t - \eta_t \tilde{\mathbf{g}}_t$

**Output** $\theta_T$ and compute the overall privacy cost $(\varepsilon, \delta)$ using a privacy accounting method.
**輸出** $\theta_T$ 並使用隱私核算方法計算整體隱私成本 $(\varepsilon, \delta)$。

**Norm clipping:** Proving the differential privacy guarantee of Algorithm 1 requires bounding the influence of each individual example on $\tilde{\mathbf{g}}_t$.
**範數裁剪：** 證明算法 1 的差分隱私保證需要限制每個單獨樣本對 $\tilde{\mathbf{g}}_t$ 的影響。

Since there is no *a priori* bound on the size of the gradients, we clip each gradient in $\ell_2$ norm; i.e., the gradient vector $\mathbf{g}$ is replaced by $\mathbf{g} / \max(1, \frac{||\mathbf{g}||_2}{C})$, for a clipping threshold $C$.
由於梯度的大小沒有 *先驗* 界限，我們對每個梯度的 $\ell_2$ 範數進行裁剪；即，對於裁剪閾值 $C$，梯度向量 $\mathbf{g}$ 被替換為 $\mathbf{g} / \max(1, \frac{||\mathbf{g}||_2}{C})$。

This clipping ensures that if $||\mathbf{g}||_2 \le C$, then $\mathbf{g}$ is preserved, whereas if $||\mathbf{g}||_2 > C$, it gets scaled down to be of norm $C$.
這種裁剪確保了如果 $||\mathbf{g}||_2 \le C$，則 $\mathbf{g}$ 被保留，而如果 $||\mathbf{g}||_2 > C$，它將被縮小為範數 $C$。

We remark that gradient clipping of this form is a popular ingredient of SGD for deep networks for non-privacy reasons, though in that setting it usually suffices to clip after averaging.
我們注意到，這種形式的梯度裁剪是出於非隱私原因的深度網絡 SGD 的流行成分，儘管在這種情況下通常在平均後裁剪就足夠了。

**Per-layer and time-dependent parameters:** The pseudocode for Algorithm 1 groups all the parameters into a single input $\theta$ of the loss function $\mathcal{L}(\cdot)$.
**逐層和時間依賴參數：** 算法 1 的偽代碼將所有參數分組為損失函數 $\mathcal{L}(\cdot)$ 的單個輸入 $\theta$。

For multi-layer neural networks, we consider each layer separately, which allows setting different clipping thresholds $C$ and noise scales $\sigma$ for different layers.
對於多層神經網絡，我們分別考慮每一層，這允許為不同層設置不同的裁剪閾值 $C$ 和噪聲尺度 $\sigma$。

Additionally, the clipping and noise parameters may vary with the number of training steps $t$. In results presented in Section 5 we use constant settings for $C$ and $\sigma$.
此外，裁剪和噪聲參數可能會隨著訓練步驟 $t$ 的變化而變化。在第 5 節呈現的結果中，我們對 $C$ 和 $\sigma$ 使用恆定設置。

**Lots:** Like the ordinary SGD algorithm, Algorithm 1 estimates the gradient of $\mathcal{L}$ by computing the gradient of the loss on a group of examples and taking the average.
**Lots (批組):** 像普通的 SGD 算法一樣，算法 1 通過計算一組樣本的損失梯度並取平均值來估計 $\mathcal{L}$ 的梯度。

This average provides an unbiased estimator, the variance of which decreases quickly with the size of the group.
該平均值提供了一個無偏估計量，其方差隨組的大小迅速減小。

We call such a group a *lot*, to distinguish it from the computational grouping that is commonly called a *batch*.
我們稱這樣的一組為 *lot (批組)*，以將其與通常稱為 *batch (批次)* 的計算分組區分開來。

In order to limit memory consumption, we may set the batch size much smaller than the lot size $L$, which is a parameter of the algorithm.
為了限制內存消耗，我們可以將批次大小設置得比批組大小 $L$ 小得多，$L$ 是算法的一個參數。

We perform the computation in batches, then group several batches into a lot for adding noise.
我們分批次進行計算，然後將幾個批次組合成一個批組以添加噪聲。

In practice, for efficiency, the construction of batches and lots is done by randomly permuting the examples and then partitioning them into groups of the appropriate sizes.
在實踐中，為了效率，批次和批組的構建是通過隨機排列樣本，然後將其劃分為適當大小的組來完成的。

For ease of analysis, however, we assume that each lot is formed by independently picking each example with probability $q = L/N$, where $N$ is the size of the input dataset.
然而，為了便於分析，我們假設每個批組是通過以概率 $q = L/N$ 獨立選取每個樣本形成的，其中 $N$ 是輸入數據集的大小。

As is common in the literature, we normalize the running time of a training algorithm by expressing it as the number of *epochs*, where each epoch is the (expected) number of batches required to process $N$ examples. In our notation, an epoch consists of $N/L$ lots.
正如文獻中常見的那樣，我們通過將訓練算法的運行時間表示為 *epochs (週期)* 數來對其進行歸一化，其中每個週期是處理 $N$ 個樣本所需的（預期）批次數。在我們的符號中，一個週期由 $N/L$ 個批組組成。

**Privacy accounting:** For differentially private SGD, an important issue is computing the overall privacy cost of the training.
**隱私核算：** 對於差分隱私 SGD，一個重要的問題是計算訓練的整體隱私成本。

The composability of differential privacy allows us to implement an “accountant” procedure that computes the privacy cost at each access to the training data, and accumulates this cost as the training progresses.
差分隱私的可組合性允許我們實現一個「會計師」程序，該程序在每次訪問訓練數據時計算隱私成本，並隨著訓練的進行累積該成本。

Each step of training typically requires gradients at multiple layers, and the accountant accumulates the cost that corresponds to all of them.
每個訓練步驟通常需要多層的梯度，會計師會累積對應於所有層的成本。

**Moments accountant:** Much research has been devoted to studying the privacy loss for a particular noise distribution as well as the composition of privacy losses.
**矩會計師：** 許多研究致力於研究特定噪聲分佈的隱私損失以及隱私損失的組合。

For the Gaussian noise that we use, if we choose $\sigma$ in Algorithm 1 to be $\sqrt{2 \log \frac{1.25}{\delta}} / \varepsilon$, then by standard arguments [20] each step is $(\varepsilon, \delta)$-differentially private with respect to the lot.
對於我們使用的高斯噪聲，如果我們在算法 1 中選擇 $\sigma$ 為 $\sqrt{2 \log \frac{1.25}{\delta}} / \varepsilon$，則根據標準論證 [20]，每一步相對於批組都是 $(\varepsilon, \delta)$-差分隱私的。

Since the lot itself is a random sample from the database, the privacy amplification theorem [33, 8] implies that each step is $(O(q\varepsilon), q\delta)$-differentially private with respect to the full database where $q = L/N$ is the sampling ratio per lot and $\varepsilon \le 1$.
由於批組本身是數據庫的隨機樣本，隱私放大定理 [33, 8] 意味著每一步相對於完整數據庫是 $(O(q\varepsilon), q\delta)$-差分隱私的，其中 $q = L/N$ 是每個批組的採樣率，且 $\varepsilon \le 1$。

The result in the literature that yields the best overall bound is the strong composition theorem [22].
文獻中產生最佳整體界限的結果是強組合定理 [22]。

However, the strong composition theorem can be loose, and does not take into account the particular noise distribution under consideration.
然而，強組合定理可能比較鬆散，並且沒有考慮所考慮的特定噪聲分佈。

In our work, we invent a stronger accounting method, which we call the moments accountant.
在我們的工作中，我們發明了一種更強的核算方法，我們稱之為矩會計師 (moments accountant)。

It allows us to prove that Algorithm 1 is $(O(q\varepsilon\sqrt{T}), \delta)$-differentially private for appropriately chosen settings of the noise scale and the clipping threshold.
它允許我們證明，對於適當選擇的噪聲尺度和裁剪閾值設置，算法 1 是 $(O(q\varepsilon\sqrt{T}), \delta)$-差分隱私的。

Compared to what one would obtain by the strong composition theorem, our bound is tighter in two ways: it saves a $\sqrt{\log(1/\delta)}$ factor in the $\varepsilon$ part and a $Tq$ factor in the $\delta$ part.
與通過強組合定理獲得的結果相比，我們的界限在兩個方面更緊密：它在 $\varepsilon$ 部分節省了 $\sqrt{\log(1/\delta)}$ 因子，在 $\delta$ 部分節省了 $Tq$ 因子。

Since we expect $\delta$ to be small and $T \gg 1/q$ (i.e., each example is examined multiple times), the saving provided by our bound is quite significant. This result is one of our main contributions.
由於我們預期 $\delta$ 很小且 $T \gg 1/q$（即每個樣本被檢查多次），我們的界限提供的節省是非常顯著的。這個結果是我們的主要貢獻之一。

**Theorem 1.** *There exist constants $c_1$ and $c_2$ so that given the sampling probability $q = L/N$ and the number of steps $T$, for any $\varepsilon < c_1q^2T$, Algorithm 1 is $(\varepsilon, \delta)$-differentially private for any $\delta > 0$ if we choose*
**定理 1.** *存在常數 $c_1$ 和 $c_2$，使得給定採樣概率 $q = L/N$ 和步驟數 $T$，對於任何 $\varepsilon < c_1q^2T$，如果我們選擇以下 $\sigma$，則算法 1 對於任何 $\delta > 0$ 都是 $(\varepsilon, \delta)$-差分隱私的：*

$$\sigma \ge c_2 \frac{q \sqrt{T \log(1/\delta)}}{\varepsilon}.$$

If we use the strong composition theorem, we will then need to choose $\sigma = \Omega(q\sqrt{T \log(1/\delta) \log(T/\delta)}/\varepsilon)$.
如果我們使用強組合定理，我們將需要選擇 $\sigma = \Omega(q\sqrt{T \log(1/\delta) \log(T/\delta)}/\varepsilon)$。

Note that we save a factor of $\sqrt{\log(T/\delta)}$ in our asymptotic bound.
請注意，我們在漸近界限中節省了一個 $\sqrt{\log(T/\delta)}$ 因子。

The moments accountant is beneficial in theory, as this result indicates, and also in practice, as can be seen from Figure 2 in Section 4.
正如該結果所示，矩會計師在理論上是有益的，在實踐中也是如此，這可以從第 4 節的圖 2 中看出。

For example, with $L = 0.01N, \sigma = 4, \delta = 10^{-5}$, and $T = 10000$, we have $\varepsilon \approx 1.26$ using the moments accountant.
例如，使用 $L = 0.01N, \sigma = 4, \delta = 10^{-5}$ 和 $T = 10000$，使用矩會計師我們得到 $\varepsilon \approx 1.26$。

As a comparison, we would get a much larger $\varepsilon \approx 9.34$ using the strong composition theorem.
作為比較，使用強組合定理我們會得到一個大得多的 $\varepsilon \approx 9.34$。

### 3.2 The Moments Accountant: Details
### 3.2 矩會計師：細節

The moments accountant keeps track of a bound on the moments of the privacy loss random variable (defined below in Eq. (1)).
矩會計師追蹤隱私損失隨機變量（定義見下文公式 (1)）的矩的界限。

It generalizes the standard approach of tracking $(\varepsilon, \delta)$ and using the strong composition theorem.
它推廣了追蹤 $(\varepsilon, \delta)$ 並使用強組合定理的標準方法。

While such an improvement was known previously for composing Gaussian mechanisms, we show that it applies also for composing Gaussian mechanisms with random sampling and can provide much tighter estimate of the privacy loss of Algorithm 1.
雖然這種改進以前在組合高斯機制時就已為人所知，但我們證明它也適用於具有隨機採樣的高斯機制組合，並且可以提供對算法 1 隱私損失的更緊密估計。

Privacy loss is a random variable dependent on the random noise added to the algorithm.
隱私損失是一個依賴於添加到算法中的隨機噪聲的隨機變量。

That a mechanism $\mathcal{M}$ is $(\varepsilon, \delta)$-differentially private is equivalent to a certain tail bound on $\mathcal{M}$'s privacy loss random variable.
機制 $\mathcal{M}$ 是 $(\varepsilon, \delta)$-差分隱私的，等價於 $\mathcal{M}$ 的隱私損失隨機變量的某個尾部界限。

While the tail bound is very useful information on a distribution, composing directly from it can result in quite loose bounds.
雖然尾部界限是關於分佈的非常有用的信息，但直接從它進行組合可能會導致相當鬆散的界限。

We instead compute the log moments of the privacy loss random variable, which compose linearly.
相反，我們計算隱私損失隨機變量的對數矩，它們線性組合。

We then use the moments bound, together with the standard Markov inequality, to obtain the tail bound, that is the privacy loss in the sense of differential privacy.
然後我們使用矩界限，連同標準馬爾可夫不等式，來獲得尾部界限，即差分隱私意義上的隱私損失。

More specifically, for neighboring databases $d, d' \in \mathcal{D}^n$, a mechanism $\mathcal{M}$, auxiliary input aux, and an outcome $o \in \mathcal{R}$, define the privacy loss at $o$ as
更具體地說，對於相鄰數據庫 $d, d' \in \mathcal{D}^n$，機制 $\mathcal{M}$，輔助輸入 aux，和結果 $o \in \mathcal{R}$，定義 $o$ 處的隱私損失為

$$c(o; \mathcal{M}, \text{aux}, d, d') \triangleq \log \frac{\text{Pr}[\mathcal{M}(\text{aux}, d) = o]}{\text{Pr}[\mathcal{M}(\text{aux}, d') = o]}. (1)$$

A common design pattern, which we use extensively in the paper, is to update the state by sequentially applying differentially private mechanisms.
我們在論文中廣泛使用的一種常見設計模式是通過順序應用差分隱私機制來更新狀態。

This is an instance of *adaptive composition*, which we model by letting the auxiliary input of the $k^{th}$ mechanism $\mathcal{M}_k$ be the output of all the previous mechanisms.
這是 *自適應組合* 的一個實例，我們通過讓第 $k$ 個機制 $\mathcal{M}_k$ 的輔助輸入成為所有先前機制的輸出來建模。

For a given mechanism $\mathcal{M}$, we define the $\lambda^{th}$ moment $\alpha_{\mathcal{M}}(\lambda; \text{aux}, d, d')$ as the log of the moment generating function evaluated at the value $\lambda$:
對於給定的機制 $\mathcal{M}$，我們定義第 $\lambda$ 階矩 $\alpha_{\mathcal{M}}(\lambda; \text{aux}, d, d')$ 為在值 $\lambda$ 處評估的矩生成函數的對數：

$$\alpha_{\mathcal{M}}(\lambda; \text{aux}, d, d') \triangleq \log \mathbb{E}_{o \sim \mathcal{M}(\text{aux}, d)} [\exp(\lambda c(o; \mathcal{M}, \text{aux}, d, d'))]. (2)$$

In order to prove privacy guarantees of a mechanism, it is useful to bound all possible $\alpha_{\mathcal{M}}(\lambda; \text{aux}, d, d')$. We define
為了證明機制的隱私保證，限制所有可能的 $\alpha_{\mathcal{M}}(\lambda; \text{aux}, d, d')$ 是有用的。我們定義

$$\alpha_{\mathcal{M}}(\lambda) \triangleq \max_{\text{aux}, d, d'} \alpha_{\mathcal{M}}(\lambda; \text{aux}, d, d'),$$

where the maximum is taken over all possible aux and all the neighboring databases $d, d'$.
其中最大值取自所有可能的 aux 和所有相鄰數據庫 $d, d'$。

We state the properties of $\alpha$ that we use for the moments accountant.
我們陳述用於矩會計師的 $\alpha$ 的屬性。

**Theorem 2.** *Let $\alpha_{\mathcal{M}}(\lambda)$ defined as above. Then*
**定理 2.** *設 $\alpha_{\mathcal{M}}(\lambda)$ 定義如上。那麼*

1. *[Composability] Suppose that a mechanism $\mathcal{M}$ consists of a sequence of adaptive mechanisms $\mathcal{M}_1, \dots, \mathcal{M}_k$ where $\mathcal{M}_i : \prod_{j=1}^{i-1} \mathcal{R}_j \times \mathcal{D} \rightarrow \mathcal{R}_i$. Then, for any $\lambda$*
1. *[可組合性] 假設機制 $\mathcal{M}$ 由一系列自適應機制 $\mathcal{M}_1, \dots, \mathcal{M}_k$ 組成，其中 $\mathcal{M}_i : \prod_{j=1}^{i-1} \mathcal{R}_j \times \mathcal{D} \rightarrow \mathcal{R}_i$。那麼，對於任何 $\lambda$*

$$\alpha_{\mathcal{M}}(\lambda) \le \sum_{i=1}^k \alpha_{\mathcal{M}_i}(\lambda).$$

2. *[Tail bound] For any $\varepsilon > 0$, the mechanism $\mathcal{M}$ is $(\varepsilon, \delta)$-differentially private for*
2. *[尾部界限] 對於任何 $\varepsilon > 0$，機制 $\mathcal{M}$ 是 $(\varepsilon, \delta)$-差分隱私的，其中*

$$\delta = \min_\lambda \exp(\alpha_{\mathcal{M}}(\lambda) - \lambda\varepsilon).$$

In particular, Theorem 2.1 holds when the mechanisms themselves are chosen based on the (public) output of the previous mechanisms.
特別是，當機制本身是根據先前機制的（公開）輸出選擇時，定理 2.1 成立。

By Theorem 2, it suffices to compute, or bound, $\alpha_{\mathcal{M}_i}(\lambda)$ at each step and sum them to bound the moments of the mechanism overall.
根據定理 2，只需計算或限制每一步的 $\alpha_{\mathcal{M}_i}(\lambda)$ 並將其求和，即可限制整個機制的矩。

We can then use the tail bound to convert the moments bound to the $(\varepsilon, \delta)$-differential privacy guarantee.
然後我們可以使用尾部界限將矩界限轉換為 $(\varepsilon, \delta)$-差分隱私保證。

The main challenge that remains is to bound the value $\alpha_{\mathcal{M}_t}(\lambda)$ for each step.
剩下的主要挑戰是限制每一步的值 $\alpha_{\mathcal{M}_t}(\lambda)$。

In the case of a Gaussian mechanism with random sampling, it suffices to estimate the following moments.
對於具有隨機採樣的高斯機制，只需估計以下矩。

Let $\mu_0$ denote the probability density function (pdf) of $\mathcal{N}(0, \sigma^2)$, and $\mu_1$ denote the pdf of $\mathcal{N}(1, \sigma^2)$.
設 $\mu_0$ 表示 $\mathcal{N}(0, \sigma^2)$ 的概率密度函數 (pdf)，$\mu_1$ 表示 $\mathcal{N}(1, \sigma^2)$ 的 pdf。

Let $\mu$ be the mixture of two Gaussians $\mu = (1-q)\mu_0 + q\mu_1$. Then we need to compute $\alpha(\lambda) = \log \max(E_1, E_2)$ where
設 $\mu$ 為兩個高斯分佈的混合 $\mu = (1-q)\mu_0 + q\mu_1$。那麼我們需要計算 $\alpha(\lambda) = \log \max(E_1, E_2)$ 其中

$$E_1 = \mathbb{E}_{z \sim \mu_0} [(\mu_0(z)/\mu(z))^\lambda], (3)$$
$$E_2 = \mathbb{E}_{z \sim \mu} [(\mu(z)/\mu_0(z))^\lambda]. (4)$$

In the implementation of the moments accountant, we carry out numerical integration to compute $\alpha(\lambda)$.
在矩會計師的實作中，我們進行數值積分來計算 $\alpha(\lambda)$。

In addition, we can show the asymptotic bound
此外，我們可以展示漸近界限

$$\alpha(\lambda) \le q^2\lambda(\lambda+1)/(1-q)\sigma^2 + O(q^3\lambda^3/\sigma^3).$$

Together with Theorem 2, the above bound implies our main Theorem 1. The details can be found in the Appendix.
結合定理 2，上述界限暗示了我們的主要定理 1。詳情可見附錄。

### 3.3 Hyperparameter Tuning
### 3.3 超參數調整

We identify characteristics of models relevant for privacy and, specifically, hyperparameters that we can tune in order to balance privacy, accuracy, and performance.
我們識別與隱私相關的模型特徵，特別是我們可以調整的超參數，以平衡隱私、準確性和性能。

In particular, through experiments, we observe that model accuracy is more sensitive to training parameters such as batch size and noise level than to the structure of a neural network.
特別是，通過實驗，我們觀察到模型準確性對訓練參數（如批次大小和噪聲水平）比對神經網絡的結構更敏感。

If we try several settings for the hyperparameters, we can trivially add up the privacy costs of all the settings, possibly via the moments accountant.
如果我們嘗試幾種超參數設置，我們可以簡單地將所有設置的隱私成本相加，可能通過矩會計師進行。

However, since we care only about the setting that gives us the most accurate model, we can do better, such as applying a version of a result from Gupta et al. [27] restated as Theorem D.1 in the Appendix.
然而，由於我們只關心給出最準確模型的設置，我們可以做得更好，例如應用 Gupta 等人 [27] 的結果的一個版本，重述為附錄中的定理 D.1。

We can use insights from theory to reduce the number of hyperparameter settings that need to be tried.
我們可以使用理論見解來減少需要嘗試的超參數設置的數量。

While differentially private optimization of convex objective functions is best achieved using batch sizes as small as 1, non-convex learning, which is inherently less stable, benefits from aggregation into larger batches.
雖然凸目標函數的差分隱私優化最好使用小至 1 的批次大小來實現，但本質上不太穩定的非凸學習受益於聚合成更大的批次。

At the same time, Theorem 1 suggests that making batches too large increases the privacy cost, and a reasonable tradeoff is to take the number of batches per epoch to be of the same order as the desired number of epochs.
同時，定理 1 表明，使批次過大會增加隱私成本，一個合理的權衡是讓每個週期的批次數量與期望的週期數量處於同一數量級。

The learning rate in non-private training is commonly adjusted downwards carefully as the model converges to a local optimum.
在非隱私訓練中，隨著模型收斂到局部最優，學習率通常會小心地向下調整。

In contrast, we never need to decrease the learning rate to a very small value, because differentially private training never reaches a regime where it would be justified.
相比之下，我們永遠不需要將學習率降低到非常小的值，因為差分隱私訓練永遠不會達到需要這樣做的狀態。

On the other hand, in our experiments, we do find that there is a small benefit to starting with a relatively large learning rate, then linearly decaying it to a smaller value in a few epochs, and keeping it constant afterwards.
另一方面，在我們的實驗中，我們確實發現從相對較大的學習率開始，然後在幾個週期內線性衰減到較小的值，並在此後保持不變，會有微小的好處。

## 4. IMPLEMENTATION
## 4. 實作

We have implemented the differentially private SGD algorithms in TensorFlow. The source code is available under an Apache 2.0 license from github.com/tensorflow/models.
我們已經在 TensorFlow 中實作了差分隱私 SGD 算法。源代碼在 Apache 2.0 許可證下可從 github.com/tensorflow/models 獲得。

For privacy protection, we need to “sanitize” the gradient before using it to update the parameters.
為了隱私保護，我們需要在用梯度更新參數之前對其進行「消毒」。

In addition, we need to keep track of the “privacy spending” based on how the sanitization is done.
此外，我們需要根據消毒的方式追蹤「隱私支出」。

Hence our implementation mainly consists of two components: `sanitizer`, which preprocesses the gradient to protect privacy, and `privacy_accountant`, which keeps track of the privacy spending over the course of training.
因此，我們的實作主要由兩個部分組成：`sanitizer`（消毒器），用於預處理梯度以保護隱私；以及 `privacy_accountant`（隱私會計師），用於追蹤訓練過程中的隱私支出。

Figure 1 contains the TensorFlow code snippet (in Python) of `DPSGD_Optimizer`, which minimizes a loss function using a differentially private SGD, and `DPTrain`, which iteratively invokes `DPSGD_Optimizer` using a privacy accountant to bound the total privacy loss.
圖 1 包含 `DPSGD_Optimizer` 的 TensorFlow 代碼片段（Python），它使用差分隱私 SGD 最小化損失函數，以及 `DPTrain`，它使用隱私會計師迭代調用 `DPSGD_Optimizer` 以限制總隱私損失。

In many cases, the neural network model may benefit from the processing of the input by projecting it on the principal directions (PCA) or by feeding it through a convolutional layer.
在許多情況下，神經網絡模型可能會受益於輸入的處理，即將其投影在主方向 (PCA) 上或將其饋送到卷積層。

We implement differentially private PCA and apply pre-trained convolutional layers (learned on public data).
我們實作了差分隱私 PCA 並應用了預訓練的卷積層（在公共數據上學習）。

**Sanitizer.** In order to achieve privacy protection, the sanitizer needs to perform two operations: (1) limit the sensitivity of each individual example by clipping the norm of the gradient for each example; and (2) add noise to the gradient of a batch before updating the network parameters.
**消毒器 (Sanitizer)。** 為了實現隱私保護，消毒器需要執行兩個操作：(1) 通過裁剪每個樣本的梯度範數來限制每個單獨樣本的敏感度；(2) 在更新網絡參數之前向批次梯度添加噪聲。

In TensorFlow, the gradient computation is batched for performance reasons, yielding $\mathbf{g}_B = 1/|B| \sum_{x \in B} \nabla_\theta \mathcal{L}(\theta, x)$ for a batch $B$ of training examples.
在 TensorFlow 中，出於性能原因，梯度計算是分批進行的，對於訓練樣本批次 $B$，產生 $\mathbf{g}_B = 1/|B| \sum_{x \in B} \nabla_\theta \mathcal{L}(\theta, x)$。

To limit the sensitivity of updates, we need to access each individual $\nabla_\theta \mathcal{L}(\theta, x)$.
為了限制更新的敏感度，我們需要訪問每個單獨的 $\nabla_\theta \mathcal{L}(\theta, x)$。

To this end, we implemented `per_example_gradient` operator in TensorFlow, as described by Goodfellow [25].
為此，我們在 TensorFlow 中實作了 `per_example_gradient` 算子，如 Goodfellow [25] 所述。

This operator can compute a batch of individual $\nabla_\theta \mathcal{L}(\theta, x)$.
該算子可以計算一批單獨的 $\nabla_\theta \mathcal{L}(\theta, x)$。

With this implementation there is only a modest slowdown in training, even for larger batch size.
有了這個實作，即使對於較大的批次大小，訓練速度也只有適度的減慢。

Our current implementation supports batched computation for the loss function $\mathcal{L}$, where each $x_i$ is singly connected to $\mathcal{L}$, allowing us to handle most hidden layers but not, for example, convolutional layers.
我們目前的實作支持損失函數 $\mathcal{L}$ 的批量計算，其中每個 $x_i$ 單獨連接到 $\mathcal{L}$，允許我們處理大多數隱藏層，但不能處理例如卷積層。

Once we have the access to the per-example gradient, it is easy to use TensorFlow operators to clip its norm and to add noise.
一旦我們可以訪問逐個樣本的梯度，就可以很容易地使用 TensorFlow 運算符來裁剪其範數並添加噪聲。

**Privacy accountant.** The main component in our implementation is `PrivacyAccountant` which keeps track of privacy spending over the course of training.
**隱私會計師 (Privacy accountant)。** 我們實作中的主要組件是 `PrivacyAccountant`，它追蹤訓練過程中的隱私支出。

As discussed in Section 3, we implemented the moments accountant that additively accumulates the log of the moments of the privacy loss at each step.
如第 3 節所述，我們實作了矩會計師，它在每一步累加隱私損失矩的對數。

Dependent on the noise distribution, one can compute $\alpha(\lambda)$ by either applying an asymptotic bound, evaluating a closed-form expression, or applying numerical integration.
根據噪聲分佈，可以通過應用漸近界限、評估閉式表達式或應用數值積分來計算 $\alpha(\lambda)$。

```python
class DPSGD_Optimizer():
    def __init__(self, accountant, sanitizer):
        self._accountant = accountant
        self._sanitizer = sanitizer

    def Minimize(self, loss, params,
                 batch_size, noise_options):
        # Accumulate privacy spending before computing
        # 累積隱私支出，在計算之前
        # and using the gradients.
        # 和使用梯度之前。
        priv_accum_op =
            self._accountant.AccumulatePrivacySpending(
            batch_size, noise_options)
        with tf.control_dependencies(priv_accum_op):
            # Compute per example gradients
            # 計算逐個樣本的梯度
            px_grads = per_example_gradients(loss, params)
            # Sanitize gradients
            # 消毒梯度
            sanitized_grads = self._sanitizer.Sanitize(
            px_grads, noise_options)
            # Take a gradient descent step
            # 採取梯度下降步驟
            return apply_gradients(params, sanitized_grads)

def DPTrain(loss, params, batch_size, noise_options):
    accountant = PrivacyAccountant()
    sanitizer = Sanitizer()
    dp_opt = DPSGD_Optimizer(accountant, sanitizer)
    sgd_op = dp_opt.Minimize(
        loss, params, batch_size, noise_options)
    eps, delta = (0, 0)
    # Carry out the training as long as the privacy
    # 只要隱私在預設限制內
    # is within the pre-set limit.
    # 就進行訓練。
    while within_limit(eps, delta):
        sgd_op.run()
        eps, delta = accountant.GetSpentPrivacy()
```

**Figure 1: Code snippet of `DPSGD_Optimizer` and `DPTrain`.**
**圖 1：`DPSGD_Optimizer` 和 `DPTrain` 的代碼片段。**

The first option would recover the generic advanced composition theorem, and the latter two give a more accurate accounting of the privacy loss.
第一個選項將恢復通用的高級組合定理，而後兩個選項則對隱私損失進行更準確的核算。

For the Gaussian mechanism we use, $\alpha(\lambda)$ is defined according to Eqs. (3) and (4).
對於我們使用的高斯機制，$\alpha(\lambda)$ 根據公式 (3) 和 (4) 定義。

In our implementation, we carry out numerical integration to compute both $E_1$ and $E_2$ in those equations.
在我們的實作中，我們進行數值積分來計算這些方程中的 $E_1$ 和 $E_2$。

Also we compute $\alpha(\lambda)$ for a range of $\lambda$'s so we can compute the best possible $(\varepsilon, \delta)$ values using Theorem 2.2.
我們還計算了一系列 $\lambda$ 的 $\alpha(\lambda)$，以便我們可以使用定理 2.2 計算盡可能最佳的 $(\varepsilon, \delta)$ 值。

We find that for the parameters of interest to us, it suffices to compute $\alpha(\lambda)$ for $\lambda \le 32$.
我們發現，對於我們感興趣的參數，計算 $\lambda \le 32$ 的 $\alpha(\lambda)$ 就足夠了。

At any point during training, one can query the privacy loss in the more interpretable notion of $(\varepsilon, \delta)$ privacy using Theorem 2.2.
在訓練期間的任何時候，都可以使用定理 2.2 以更易解釋的 $(\varepsilon, \delta)$ 隱私概念查詢隱私損失。

Rogers et al. [47] point out risks associated with adaptive choice of privacy parameters.
Rogers 等人 [47] 指出了與自適應選擇隱私參數相關的風險。

We avoid their attacks and negative results by fixing the number of iterations and privacy parameters ahead of time.
我們通過提前固定迭代次數和隱私參數來避免他們的攻擊和負面結果。

More general implementations of a privacy accountant must correctly distinguish between two modes of operation—as a privacy odometer or a privacy filter (see [47] for more details).
隱私會計師的更通用實作必須正確區分兩種操作模式——作為隱私里程表或隱私過濾器（詳見 [47]）。

**Differentially private PCA.** Principal component analysis (PCA) is a useful method for capturing the main features of the input data.
**差分隱私 PCA。** 主成分分析 (PCA) 是捕獲輸入數據主要特徵的有用方法。

We implement the differentially private PCA algorithm as described in [23].
我們實作了如 [23] 所述的差分隱私 PCA 算法。

More specifically, we take a random sample of the training examples, treat them as vectors, and normalize each vector to unit $\ell_2$ norm to form the matrix $A$, where each vector is a row in the matrix.
更具體地說，我們對訓練樣本進行隨機採樣，將它們視為向量，並將每個向量歸一化為單位 $\ell_2$ 範數以形成矩陣 $A$，其中每個向量是矩陣中的一行。

We then add Gaussian noise to the covariance matrix $A^T A$ and compute the principal directions of the noisy covariance matrix.
然後，我們向協方差矩陣 $A^T A$ 添加高斯噪聲，並計算噪聲協方差矩陣的主方向。

Then for each input example we apply the projection to these principal directions before feeding it into the neural network.
然後對於每個輸入樣本，我們在將其饋送到神經網絡之前將投影應用於這些主方向。

We incur a privacy cost due to running a PCA. However, we find it useful for both improving the model quality and for reducing the training time, as suggested by our experiments on the MNIST data. See Section 4 for details.
由於運行 PCA，我們會產生隱私成本。然而，我們發現它對於提高模型質量和減少訓練時間都很有用，正如我們在 MNIST 數據上的實驗所表明的那樣。詳見第 4 節。

**Convolutional layers.** Convolutional layers are useful for deep neural networks.
**卷積層。** 卷積層對深度神經網絡很有用。

However, an efficient per-example gradient computation for convolutional layers remains a challenge within the TensorFlow framework, which motivates creating a separate workflow.
然而，在 TensorFlow 框架內，卷積層的高效逐樣本梯度計算仍然是一個挑戰，這促使我們創建一個單獨的工作流程。

For example, some recent work argues that even random convolutions often suffice [46, 13, 49, 55, 14].
例如，最近的一些工作認為，即使是隨機卷積通常也足夠了 [46, 13, 49, 55, 14]。

Alternatively, we explore the idea of learning convolutional layers on public data, following Jarrett et al. [30].
或者，我們探索在公共數據上學習卷積層的想法，遵循 Jarrett 等人 [30] 的方法。

Such convolutional layers can be based on GoogLeNet or AlexNet features [54, 35] for image models or on pretrained word2vec or GloVe embeddings in language models [41, 44].
這種卷積層可以基於圖像模型的 GoogLeNet 或 AlexNet 特徵 [54, 35]，或語言模型中預訓練的 word2vec 或 GloVe 嵌入 [41, 44]。

## 5. EXPERIMENTAL RESULTS
## 5. 實驗結果

This section reports on our evaluation of the moments accountant, and results on two popular image datasets: MNIST and CIFAR-10.
本節報告我們對矩會計師的評估，以及在兩個流行的圖像數據集上的結果：MNIST 和 CIFAR-10。

### 5.1 Applying the Moments Accountant
### 5.1 應用矩會計師

As shown by Theorem 1, the moments accountant provides a tighter bound on the privacy loss compared to the generic strong composition theorem.
如定理 1 所示，與通用的強組合定理相比，矩會計師提供了對隱私損失更緊密的界限。

Here we compare them using some concrete values.
在這裡，我們使用一些具體的值比較它們。

The overall privacy loss $(\varepsilon, \delta)$ can be computed from the noise level $\sigma$, the sampling ratio of each lot $q = L/N$ (so each epoch consists of $1/q$ batches), and the number of epochs $E$ (so the number of steps is $T = E/q$).
整體隱私損失 $(\varepsilon, \delta)$ 可以從噪聲水平 $\sigma$、每個批組的採樣率 $q = L/N$（因此每個週期包含 $1/q$ 個批次）和週期數 $E$（因此步數為 $T = E/q$）計算得出。

We fix the target $\delta = 10^{-5}$, the value used for our MNIST and CIFAR experiments.
我們固定目標 $\delta = 10^{-5}$，這是我們 MNIST 和 CIFAR 實驗中使用的值。

In our experiment, we set $q = 0.01, \sigma = 4$, and $\delta = 10^{-5}$, and compute the value of $\varepsilon$ as a function of the training epoch $E$.
在我們的實驗中，我們設置 $q = 0.01, \sigma = 4, \delta = 10^{-5}$，並計算 $\varepsilon$ 作為訓練週期 $E$ 的函數的值。

Figure 2 shows two curves corresponding to, respectively, using the strong composition theorem and the moments accountant.
圖 2 顯示了兩條曲線，分別對應於使用強組合定理和矩會計師。

We can see that we get a much tighter estimation of the privacy loss by using the moments accountant.
我們可以看到，通過使用矩會計師，我們對隱私損失的估計要緊密得多。

For examples, when $E = 100$, the values are 9.34 and 1.26 respectively, and for $E = 400$, the values are 24.22 and 2.55 respectively.
例如，當 $E = 100$ 時，值分別為 9.34 和 1.26，當 $E = 400$ 時，值分別為 24.22 和 2.55。

That is, using the moments bound, we achieve $(2.55, 10^{-5})$-differential privacy, whereas previous techniques only obtain the significantly worse guarantee of $(24.22, 10^{-5})$.
也就是說，使用矩界限，我們實現了 $(2.55, 10^{-5})$-差分隱私，而以前的技術只能獲得明顯更差的保證 $(24.22, 10^{-5})$。

### 5.2 MNIST
### 5.2 MNIST

We conduct experiments on the standard MNIST dataset for handwritten digit recognition consisting of 60,000 training examples and 10,000 testing examples [36].
我們在標準 MNIST 數據集上進行實驗，該數據集用於手寫數字識別，包含 60,000 個訓練樣本和 10,000 個測試樣本 [36]。

Each example is a $28 \times 28$ size gray-level image.
每個樣本都是一個 $28 \times 28$ 大小的灰度圖像。

We use a simple feedforward neural network with ReLU units and softmax of 10 classes (corresponding to the 10 digits) with cross-entropy loss and an optional PCA input layer.
我們使用一個簡單的前饋神經網絡，帶有 ReLU 單元和 10 個類別的 softmax（對應於 10 個數字），採用交叉熵損失和可選的 PCA 輸入層。

**Baseline model.**
**基準模型。**

Our baseline model uses a 60-dimensional PCA projection layer and a single hidden layer with 1,000 hidden units.
我們的基準模型使用一個 60 維 PCA 投影層和一個包含 1,000 個隱藏單元的單隱藏層。

Using the lot size of 600, we can reach accuracy of 98.30% in about 100 epochs.
使用 600 的批組大小，我們可以在大約 100 個週期內達到 98.30% 的準確率。

This result is consistent with what can be achieved with a vanilla neural network [36].
這個結果與普通神經網絡所能達到的結果一致 [36]。

**Figure 2: The $\varepsilon$ value as a function of epoch $E$ for $q = 0.01, \sigma = 4, \delta = 10^{-5}$, using the strong composition theorem and the moments accountant respectively.**
**圖 2：對於 $q = 0.01, \sigma = 4, \delta = 10^{-5}$，分別使用強組合定理和矩會計師時，$\varepsilon$ 值隨週期 $E$ 變化的函數。**

**Differentially private model.**
**差分隱私模型。**

For the differentially private version, we experiment with the same architecture with a 60-dimensional PCA projection layer, a single 1,000-unit ReLU hidden layer, and a lot size of 600.
對於差分隱私版本，我們實驗使用相同的架構，包括 60 維 PCA 投影層、單個 1,000 單元 ReLU 隱藏層和 600 的批組大小。

To limit sensitivity, we clip the gradient norm of each layer at 4.
為了限制敏感度，我們將每一層的梯度範數裁剪為 4。

We report results for three choices of the noise scale, which we call small ($\sigma = 2, \sigma_p = 4$), medium ($\sigma = 4, \sigma_p = 7$), and large ($\sigma = 8, \sigma_p = 16$).
我們報告了三種噪聲尺度選擇的結果，我們稱之為小 ($\sigma = 2, \sigma_p = 4$)、中 ($\sigma = 4, \sigma_p = 7$) 和大 ($\sigma = 8, \sigma_p = 16$)。

Here $\sigma$ represents the noise level for training the neural network, and $\sigma_p$ the noise level for PCA projection.
這裡 $\sigma$ 代表訓練神經網絡的噪聲水平，$\sigma_p$ 代表 PCA 投影的噪聲水平。

The learning rate is set at 0.1 initially and linearly decreased to 0.052 over 10 epochs and then fixed to 0.052 thereafter.
學習率最初設置為 0.1，在 10 個週期內線性降低到 0.052，此後固定為 0.052。

We have also experimented with multi-hidden-layer networks. For MNIST, we found that one hidden layer combined with PCA works better than a two-layer network.
我們還嘗試了多隱藏層網絡。對於 MNIST，我們發現一個結合 PCA 的隱藏層比兩層網絡效果更好。

Figure 3 shows the results for different noise levels.
圖 3 顯示了不同噪聲水平的結果。

In each plot, we show the evolution of the training and testing accuracy as a function of the number of epochs as well as the corresponding $\delta$ value, keeping $\varepsilon$ fixed.
在每個圖中，我們展示了訓練和測試準確率隨週期數的演變，以及相應的 $\delta$ 值，保持 $\varepsilon$ 固定。

We achieve 90%, 95%, and 97% test set accuracy for $(0.5, 10^{-5})$, $(2, 10^{-5})$, and $(8, 10^{-5})$-differential privacy respectively.
我們分別在 $(0.5, 10^{-5})$、$(2, 10^{-5})$ 和 $(8, 10^{-5})$-差分隱私下實現了 90%、95% 和 97% 的測試集準確率。

One attractive consequence of applying differentially private SGD is the small difference between the model’s accuracy on the training and the test sets, which is consistent with the theoretical argument that differentially private training generalizes well [6].
應用差分隱私 SGD 的一個吸引人的結果是模型在訓練集和測試集上的準確率差異很小，這與差分隱私訓練泛化能力強的理論論點一致 [6]。

In contrast, the gap between training and testing accuracy in non-private training, i.e., evidence of overfitting, increases with the number of epochs.
相比之下，非隱私訓練中訓練和測試準確率之間的差距（即過擬合的證據）隨著週期數的增加而增加。

By using the moments accountant, we can obtain a $\delta$ value for any given $\varepsilon$.
通過使用矩會計師，我們可以獲得任何給定 $\varepsilon$ 的 $\delta$ 值。

We record the accuracy for different $(\varepsilon, \delta)$ pairs in Figure 4.
我們在圖 4 中記錄了不同 $(\varepsilon, \delta)$ 對的準確率。

In the figure, each curve corresponds to the best accuracy achieved for a fixed $\delta$, as it varies between $10^{-5}$ and $10^{-2}$.
在圖中，每條曲線對應於固定 $\delta$（在 $10^{-5}$ 和 $10^{-2}$ 之間變化）時實現的最佳準確率。

For example, we can achieve 90% accuracy for $\varepsilon = 0.25$ and $\delta = 0.01$.
例如，對於 $\varepsilon = 0.25$ 和 $\delta = 0.01$，我們可以達到 90% 的準確率。

As can be observed from the figure, for a fixed $\delta$, varying the value of $\varepsilon$ can have large impact on accuracy, but for any fixed $\varepsilon$, there is less difference with different $\delta$ values.
從圖中可以看出，對於固定的 $\delta$，改變 $\varepsilon$ 的值會對準確率產生很大影響，但對於任何固定的 $\varepsilon$，不同的 $\delta$ 值之間的差異較小。

**Effect of the parameters.**
**參數的影響。**

Classification accuracy is determined by multiple factors that must be carefully tuned for optimal performance.
分類準確率由多個因素決定，必須仔細調整這些因素以獲得最佳性能。

These factors include the topology of the network, the number of PCA dimensions and the number of hidden units, as well as parameters of the training procedure such as the lot size and the learning rate.
這些因素包括網絡的拓撲結構、PCA 維數和隱藏單元的數量，以及訓練過程的參數，如批組大小和學習率。

Some parameters are specific to privacy, such as the gradient norm clipping bound and the noise level.
有些參數是隱私特有的，例如梯度範數裁剪界限和噪聲水平。

To demonstrate the effects of these parameters, we manipulate them individually, keeping the rest constant.
為了展示這些參數的影響，我們單獨操作它們，保持其餘不變。

We set the reference values as follows: 60 PCA dimensions, 1,000 hidden units, 600 lot size, gradient norm bound of 4, initial learning rate of 0.1 decreasing to a final learning rate of 0.052 in 10 epochs, and noise $\sigma$ equal to 4 and 7 respectively for training the neural network parameters and for the PCA projection.
我們將參考值設置如下：60 個 PCA 維度，1,000 個隱藏單元，600 的批組大小，梯度範數界限為 4，初始學習率 0.1 在 10 個週期內降至最終學習率 0.052，訓練神經網絡參數和 PCA 投影的噪聲 $\sigma$ 分別為 4 和 7。

For each combination of values, we train until the point at which $(2, 10^{-5})$-differential privacy would be violated (so, for example, a larger $\sigma$ allows more epochs of training). The results are presented in Figure 5.
對於每種值組合，我們訓練直到違反 $(2, 10^{-5})$-差分隱私的點（因此，例如，較大的 $\sigma$ 允許更多週期的訓練）。結果如圖 5 所示。

**Figure 3: Results on the accuracy for different noise levels on the MNIST dataset.**
**圖 3：MNIST 數據集上不同噪聲水平的準確率結果。**

In all the experiments, the network uses 60 dimension PCA projection, 1,000 hidden units, and is trained using lot size 600 and clipping threshold 4. The noise levels $(\sigma, \sigma_p)$ for training the neural network and for PCA projection are set at (8, 16), (4, 7), and (2, 4), respectively, for the three experiments.
在所有實驗中，網絡使用 60 維 PCA 投影、1,000 個隱藏單元，並使用 600 的批組大小和 4 的裁剪閾值進行訓練。三個實驗中，用於訓練神經網絡和 PCA 投影的噪聲水平 $(\sigma, \sigma_p)$ 分別設置為 (8, 16)、(4, 7) 和 (2, 4)。

**Figure 4: Accuracy of various $(\varepsilon, \delta)$ privacy values on the MNIST dataset.**
**圖 4：MNIST 數據集上各種 $(\varepsilon, \delta)$ 隱私值的準確率。**

Each curve corresponds to a different $\delta$ value.
每條曲線對應一個不同的 $\delta$ 值。

**PCA projection.** In our experiments, the accuracy is fairly stable as a function of the PCA dimension, with the best results achieved for 60. (Not doing PCA reduces accuracy by about 2%.)
**PCA 投影。** 在我們的實驗中，準確率隨 PCA 維度的變化相當穩定，在 60 維時達到最佳結果。（不做 PCA 會使準確率降低約 2%。）

Although in principle the PCA projection layer can be replaced by an additional hidden layer, we achieve better accuracy by training the PCA layer separately.
雖然原則上 PCA 投影層可以被額外的隱藏層取代，但我們通過單獨訓練 PCA 層獲得了更好的準確率。

By reducing the input size from 784 to 60, PCA leads to an almost 10× reduction in training time.
通過將輸入大小從 784 減少到 60，PCA 導致訓練時間減少了近 10 倍。

The result is fairly stable over a large range of the noise levels for the PCA projection and consistently better than the accuracy using random projection, which is at about 92.5% and shown as a horizontal line in the plot.
結果在 PCA 投影的較大噪聲水平範圍內相當穩定，並且始終優於使用隨機投影的準確率（約為 92.5%，在圖中顯示為水平線）。

**Number of hidden units.** Including more hidden units makes it easier to fit the training set.
**隱藏單元數量。** 包含更多隱藏單元使得擬合訓練集更容易。

For non-private training, it is often preferable to use more units, as long as we employ techniques to avoid overfitting.
對於非隱私訓練，只要我們採用避免過擬合的技術，通常最好使用更多單元。

However, for differentially private training, it is not a priori clear if more hidden units improve accuracy, as more hidden units increase the sensitivity of the gradient, which leads to more noise added at each update.
然而，對於差分隱私訓練，增加隱藏單元是否會提高準確率並不是先驗清楚的，因為更多隱藏單元會增加梯度的敏感度，這導致每次更新時添加更多噪聲。

Somewhat counterintuitively, increasing the number of hidden units does not decrease accuracy of the trained model.
有些反直覺的是，增加隱藏單元的數量並不會降低訓練模型的準確率。

One possible explanation that calls for further analysis is that larger networks are more tolerant to noise. This property is quite encouraging as it is common in practice to use very large networks.
一個需要進一步分析的可能解釋是，較大的網絡對噪聲的容忍度更高。這一特性非常令人鼓舞，因為在實踐中使用非常大的網絡是很常見的。

**Lot size.** According to Theorem 1, we can run $N/L$ epochs while staying within a constant privacy budget.
**批組大小 (Lot size)。** 根據定理 1，我們可以在保持恆定隱私預算的同時運行 $N/L$ 個週期。

Choosing the lot size must balance two conflicting objectives.
選擇批組大小必須平衡兩個相互衝突的目標。

On the one hand, smaller lots allow running more epochs, i.e., passes over data, improving accuracy.
一方面，較小的批組允許運行更多週期，即遍歷數據，從而提高準確率。

On the other hand, for a larger lot, the added noise has a smaller relative effect.
另一方面，對於較大的批組，添加的噪聲具有較小的相對影響。

Our experiments show that the lot size has a relatively large impact on accuracy.
我們的實驗表明，批組大小對準確率有相對較大的影響。

Empirically, the best lot size is roughly $\sqrt{N}$ where $N$ is the number of training examples.
根據經驗，最佳批組大小大約是 $\sqrt{N}$，其中 $N$ 是訓練樣本的數量。

**Learning rate.** Accuracy is stable for a learning rate in the range of [0.01, 0.07] and peaks at 0.05, as shown in Figure 5(4).
**學習率。** 學習率在 [0.01, 0.07] 範圍內時準確率穩定，並在 0.05 處達到峰值，如圖 5(4) 所示。

However, accuracy decreases significantly if the learning rate is too large.
然而，如果學習率過大，準確率會顯著下降。

Some additional experiments suggest that, even for large learning rates, we can reach similar levels of accuracy by reducing the noise level and, accordingly, by training less in order to avoid exhausting the privacy budget.
一些額外的實驗表明，即使對於較大的學習率，我們也可以通過降低噪聲水平並相應地減少訓練（以避免耗盡隱私預算）來達到類似的準確率水平。

**Clipping bound.** Limiting the gradient norm has two opposing effects: clipping destroys the unbiasedness of the gradient estimate, and if the clipping parameter is too small, the average clipped gradient may point in a very different direction from the true gradient.
**裁剪界限。** 限制梯度範數有兩個相反的效果：裁剪破壞了梯度估計的無偏性，如果裁剪參數太小，平均裁剪梯度可能指向與真實梯度非常不同的方向。

On the other hand, increasing the norm bound $C$ forces us to add more noise to the gradients (and hence the parameters), since we add noise based on $\sigma C$.
另一方面，增加範數界限 $C$ 迫使我們向梯度（以及參數）添加更多噪聲，因為我們基於 $\sigma C$ 添加噪聲。

In practice, a good way to choose a value for $C$ is by taking the median of the norms of the unclipped gradients over the course of training.
在實踐中，選擇 $C$ 值的一個好方法是取訓練過程中未裁剪梯度範數的中位數。

**Noise level.** By adding more noise, the per-step privacy loss is proportionally smaller, so we can run more epochs within a given cumulative privacy budget.
**噪聲水平。** 通過添加更多噪聲，每步隱私損失成比例減小，因此我們可以在給定的累積隱私預算內運行更多週期。

In Figure 5(5), the x-axis is the noise level $\sigma$. The choice of this value has a large impact on accuracy.
在圖 5(5) 中，x 軸是噪聲水平 $\sigma$。這個值的選擇對準確率有很大影響。

From the experiments, we observe the following.
從實驗中，我們觀察到以下幾點。

1. The PCA projection improves both model accuracy and training performance. Accuracy is quite stable over a large range of choices for the projection dimensions and the noise level used in the PCA stage.
1. PCA 投影提高了模型準確率和訓練性能。在 PCA 階段使用的投影維度和噪聲水平的較大選擇範圍內，準確率相當穩定。

2. The accuracy is fairly stable over the network size. When we can only run smaller number of epochs, it is more beneficial to use a larger network.
2. 準確率在網絡規模上相當穩定。當我們只能運行較少數量的週期時，使用較大的網絡更有益。

3. The training parameters, especially the lot size and the noise scale $\sigma$, have a large impact on the model accuracy. They both determine the “noise-to-signal” ratio of the sanitized gradients as well as the number of epochs we are able to go through the data before reaching the privacy limit.
3. 訓練參數，特別是批組大小和噪聲尺度 $\sigma$，對模型準確率有很大影響。它們都決定了消毒梯度的「信噪比」，以及我們在達到隱私限制之前能夠遍歷數據的週期數。

Our framework allows for adaptive control of the training parameters, such as the lot size, the gradient norm bound $C$, and noise level $\sigma$.
我們的框架允許自適應控制訓練參數，例如批組大小、梯度範數界限 $C$ 和噪聲水平 $\sigma$。

Our initial experiments with decreasing noise as training progresses did not show a significant improvement, but it is interesting to consider more sophisticated schemes for adaptively choosing these parameters.
我們最初關於隨著訓練進行降低噪聲的實驗沒有顯示出顯著的改進，但考慮更複雜的方案來自適應選擇這些參數是有趣的。

**Figure 5: MNIST accuracy when one parameter varies, and the others are fixed at reference values.**
**圖 5：當一個參數變化，而其他參數固定在參考值時的 MNIST 準確率。**

### 5.3 CIFAR
### 5.3 CIFAR

We also conduct experiments on the CIFAR-10 dataset, which consists of color images classified into 10 classes such as ships, cats, and dogs, and partitioned into 50,000 training examples and 10,000 test examples [1].
我們還在 CIFAR-10 數據集上進行了實驗，該數據集由分為 10 類（如船、貓和狗）的彩色圖像組成，並分為 50,000 個訓練樣本和 10,000 個測試樣本 [1]。

Each example is a $32 \times 32$ image with three channels (RGB).
每個樣本都是一個具有三個通道 (RGB) 的 $32 \times 32$ 圖像。

For this learning task, nearly all successful networks use convolutional layers.
對於這個學習任務，幾乎所有成功的網絡都使用卷積層。

The CIFAR-100 dataset has similar parameters, except that images are classified into 100 classes; the examples and the image classes are different from those of CIFAR-10.
CIFAR-100 數據集具有類似的參數，只是圖像被分為 100 類；樣本和圖像類別與 CIFAR-10 不同。

We use the network architecture from the TensorFlow convolutional neural networks tutorial [2].
我們使用 TensorFlow 卷積神經網絡教程中的網絡架構 [2]。

Each $32 \times 32$ image is first cropped to a $24 \times 24$ one by taking the center patch.
每個 $32 \times 32$ 圖像首先通過取中心補丁裁剪為 $24 \times 24$。

The network architecture consists of two convolutional layers followed by two fully connected layers.
網絡架構由兩個卷積層和隨後的兩個全連接層組成。

The convolutional layers use $5 \times 5$ convolutions with stride 1, followed by a ReLU and $2 \times 2$ max pools, with 64 channels each.
卷積層使用 $5 \times 5$ 卷積，步長為 1，後跟 ReLU 和 $2 \times 2$ 最大池化，每個層有 64 個通道。

Thus the first convolution outputs a $12 \times 12 \times 64$ tensor for each image, and the second outputs a $6 \times 6 \times 64$ tensor.
因此，第一個卷積為每個圖像輸出 $12 \times 12 \times 64$ 張量，第二個輸出 $6 \times 6 \times 64$ 張量。

The latter is flattened to a vector that gets fed into a fully connected layer with 384 units, and another one of the same size.
後者被展平為一個向量，饋送到一個具有 384 個單元的全連接層，以及另一個相同大小的層。

This architecture, non-privately, can get to about 86% accuracy in 500 epochs. Its simplicity makes it an appealing choice for our work.
這種架構在非隱私情況下，可以在 500 個週期內達到約 86% 的準確率。它的簡單性使其成為我們工作的一個有吸引力的選擇。

We should note however that by using deeper networks with different non-linearities and other advanced techniques, one can obtain significantly better accuracy, with the state-of-the-art being about 96.5% [26].
然而我們應該注意，通過使用具有不同非線性和其他先進技術的更深層網絡，可以獲得明顯更好的準確率，最先進的水平約為 96.5% [26]。

As is standard for such image datasets, we use *data augmentation* during training.
正如這類圖像數據集的標準做法，我們在訓練期間使用 *數據增強*。

For each training image, we generate a new distorted image by randomly picking a $24 \times 24$ patch from the image, randomly flipping the image along the left-right direction, and randomly distorting the brightness and the contrast of the image.
對於每個訓練圖像，我們通過隨機選取圖像中的 $24 \times 24$ 補丁，沿左右方向隨機翻轉圖像，並隨機扭曲圖像的亮度和對比度來生成新的失真圖像。

In each epoch, these distortions are done independently. We refer the reader to the TensorFlow tutorial [2] for additional details.
在每個週期中，這些失真是獨立完成的。我們建議讀者參考 TensorFlow 教程 [2] 以獲取更多詳細信息。

As the convolutional layers have shared parameters, computing per-example gradients has a larger computational overhead.
由於卷積層具有共享參數，計算逐樣本梯度具有較大的計算開銷。

Previous work has shown that convolutional layers are often transferable: parameters learned from one dataset can be used on another one without retraining [30].
以前的工作表明，卷積層通常是可遷移的：從一個數據集學習的參數可以用於另一個數據集而無需重新訓練 [30]。

We treat the CIFAR-100 dataset as a public dataset and use it to train a network with the same architecture.
我們將 CIFAR-100 數據集視為公共數據集，並用它來訓練具有相同架構的網絡。

We use the convolutions learned from training this dataset.
我們使用從訓練該數據集中學到的卷積。

Retraining only the fully connected layers with this architecture for about 250 epochs with a batch size of 120 gives us approximately 80% accuracy, which is our non-private baseline.
使用此架構僅重新訓練全連接層約 250 個週期，批次大小為 120，使我們獲得大約 80% 的準確率，這是我們的非隱私基準。

**Differentially private version.**
**差分隱私版本。**

For the differentially private version, we use the same architecture.
對於差分隱私版本，我們使用相同的架構。

As discussed above, we use pre-trained convolutional layers.
如上所述，我們使用預訓練的卷積層。

The fully connected layers are initialized from the pre-trained network as well.
全連接層也從預訓練網絡初始化。

We train the softmax layer, and either the top or both fully connected layers.
我們訓練 softmax 層，以及頂部或兩個全連接層。

Based on looking at gradient norms, the softmax layer gradients are roughly twice as large as the other two layers, and we keep this ratio when we try clipping at a few different values between 3 and 10.
基於觀察梯度範數，softmax 層梯度大約是其他兩層的兩倍，當我們嘗試在 3 到 10 之間的一些不同值進行裁剪時，我們保持這個比例。

The lot size is an additional knob that we tune: we tried 600, 2,000, and 4,000.
批組大小是我們調整的另一個旋鈕：我們嘗試了 600、2,000 和 4,000。

With these settings, the per-epoch training time increases from approximately 40 seconds to 180 seconds.
使用這些設置，每個週期的訓練時間從大約 40 秒增加到 180 秒。

In Figure 6, we show the evolution of the accuracy and the privacy cost, as a function of the number of epochs, for a few different parameter settings.
在圖 6 中，我們展示了對於幾種不同參數設置，準確率和隱私成本隨週期數變化的演變。

The various parameters influence the accuracy one gets, in ways not too different from that in the MNIST experiments.
各種參數影響獲得的準確率，其方式與 MNIST 實驗中的方式沒有太大區別。

A lot size of 600 leads to poor results on this dataset and we need to increase it to 2,000 or more for results reported in Figure 6.
600 的批組大小在這個數據集上導致結果不佳，我們需要將其增加到 2,000 或更多以獲得圖 6 中報告的結果。

Compared to the MNIST dataset, where the difference in accuracy between a non-private baseline and a private model is about 1.3%, the corresponding drop in accuracy in our CIFAR-10 experiment is much larger (about 7%).
與 MNIST 數據集（非隱私基準和隱私模型之間的準確率差異約為 1.3%）相比，我們的 CIFAR-10 實驗中相應的準確率下降要大得多（約 7%）。

We leave closing this gap as an interesting test for future research in differentially private machine learning.
我們將縮小這一差距作為差分隱私機器學習未來研究的一個有趣測試。

**Figure 6: Results on accuracy for different noise levels on CIFAR-10.**
**圖 6：CIFAR-10 上不同噪聲水平的準確率結果。**

With $\delta$ set to $10^{-5}$, we achieve accuracy 67%, 70%, and 73%, with $\varepsilon$ being 2, 4, and 8, respectively.
將 $\delta$ 設置為 $10^{-5}$，我們分別在 $\varepsilon$ 為 2、4 和 8 時達到 67%、70% 和 73% 的準確率。

The first graph uses a lot size of 2,000, (2) and (3) use a lot size of 4,000. In all cases, $\sigma$ is set to 6, and clipping is set to 3.
第一張圖使用 2,000 的批組大小，(2) 和 (3) 使用 4,000 的批組大小。在所有情況下，$\sigma$ 設置為 6，裁剪設置為 3。

## 6. RELATED WORK
## 6. 相關工作

The problem of privacy-preserving data mining, or machine learning, has been a focus of active work in several research communities since the late 90s [5, 37].
自 90 年代末以來，隱私保護數據挖掘或機器學習問題一直是多個研究社區積極工作的焦點 [5, 37]。

The existing literature can be broadly classified along several axes: the class of models, the learning algorithm, and the privacy guarantees.
現有文獻大致可以沿著幾個軸進行分類：模型類別、學習算法和隱私保證。

**Privacy guarantees.** Early works on privacy-preserving learning were done in the framework of secure function evaluation (SFE) and secure multi-party computations (MPC), where the input is split between two or more parties, and the focus is on minimizing information leaked during the joint computation of some agreed-to functionality.
**隱私保證。** 早期的隱私保護學習工作是在安全函數評估 (SFE) 和安全多方計算 (MPC) 的框架下完成的，其中輸入在兩方或多方之間分割，重點是最小化在聯合計算某些商定功能期間洩露的信息。

In contrast, we assume that data is held centrally, and we are concerned with leakage from the functionality’s output (i.e., the model).
相比之下，我們假設數據是集中保存的，我們關注的是功能輸出（即模型）的洩露。

Another approach, $k$-anonymity and closely related notions [53], seeks to offer a degree of protection to underlying data by generalizing and suppressing certain identifying attributes.
另一種方法，$k$-匿名及其密切相關的概念 [53]，試圖通過泛化和抑制某些識別屬性來為底層數據提供一定程度的保護。

The approach has strong theoretical and empirical limitations [4, 9] that make it all but inapplicable to de-anonymization of high-dimensional, diverse input datasets.
該方法具有強烈的理論和經驗局限性 [4, 9]，使其幾乎不適用於高維、多樣化輸入數據集的去匿名化。

Rather than pursue input sanitization, we keep the underlying raw records intact and perturb derived data instead.
我們不追求輸入消毒，而是保持底層原始記錄完整，轉而擾動衍生數據。

The theory of differential privacy, which provides the analytical framework for our work, has been applied to a large collection of machine learning tasks that differed from ours either in the training mechanism or in the target model.
差分隱私理論為我們的工作提供了分析框架，它已被應用於大量的機器學習任務，這些任務在訓練機制或目標模型方面與我們的不同。

The moments accountant is closely related to the notion of Rényi differential privacy [42], which proposes (scaled) $\alpha(\lambda)$ as a means of quantifying privacy guarantees.
矩會計師與 Rényi 差分隱私 [42] 的概念密切相關，後者提出（縮放的）$\alpha(\lambda)$ 作為量化隱私保證的一種手段。

In a concurrent and independent work Bun and Steinke [10] introduce a relaxation of differential privacy (generalizing the work of Dwork and Rothblum [20]) defined via a linear upper bound on $\alpha(\lambda)$.
在同時期的一項獨立工作中，Bun 和 Steinke [10] 引入了差分隱私的一種鬆弛（推廣了 Dwork 和 Rothblum [20] 的工作），通過 $\alpha(\lambda)$ 的線性上界定義。

Taken together, these works demonstrate that the moments accountant is a useful technique for theoretical and empirical analyses of complex privacy-preserving algorithms.
綜合起來，這些工作證明了矩會計師是複雜隱私保護算法的理論和經驗分析的有用技術。

**Learning algorithm.** A common target for learning with privacy is a class of convex optimization problems amenable to a wide variety of techniques [18, 11, 34].
**學習算法。** 隱私學習的一個常見目標是一類適用於多種技術的凸優化問題 [18, 11, 34]。

In concurrent work, Wu et al. achieve 83% accuracy on MNIST via convex empirical risk minimization [57].
在同時期的工作中，Wu 等人通過凸經驗風險最小化在 MNIST 上達到了 83% 的準確率 [57]。

Training multi-layer neural networks is non-convex, and typically solved by an application of SGD, whose theoretical guarantees are poorly understood.
訓練多層神經網絡是非凸的，通常通過應用 SGD 來解決，其理論保證知之甚少。

For the CIFAR neural network we incorporate differentially private training of the PCA projection matrix [23], which is used to reduce dimensionality of inputs.
對於 CIFAR 神經網絡，我們結合了 PCA 投影矩陣的差分隱私訓練 [23]，用於降低輸入的維數。

**Model class.** The first end-to-end differentially private system was evaluated on the Netflix Prize dataset [39], a version of a collaborative filtering problem.
**模型類別。** 第一個端到端差分隱私系統在 Netflix Prize 數據集 [39] 上進行了評估，這是一個協同過濾問題的版本。

Although the problem shared many similarities with ours—high-dimensional inputs, non-convex objective function—the approach taken by McSherry and Mironov differed significantly.
儘管該問題與我們的問題有許多相似之處——高維輸入、非凸目標函數——但 McSherry 和 Mironov 採取的方法明顯不同。

They identified the core of the learning task, effectively sufficient statistics, that can be computed in a differentially private manner via a Gaussian mechanism. In our approach no such sufficient statistics exist.
他們確定了學習任務的核心，即有效的充分統計量，可以通過高斯機制以差分隱私的方式計算。在我們的方法中不存在這樣的充分統計量。

In a recent work Shokri and Shmatikov [50] designed and evaluated a system for *distributed training* of a deep neural network.
在最近的一項工作中，Shokri 和 Shmatikov [50] 設計並評估了一個用於深度神經網絡 *分佈式訓練* 的系統。

Participants, who hold their data closely, communicate sanitized updates to a central authority.
持有數據的參與者將消毒後的更新傳達給中央機構。

The sanitization relies on an additive-noise mechanism, based on a sensitivity estimate, which could be improved to a hard sensitivity guarantee.
消毒依賴於基於敏感度估計的加性噪聲機制，該機制可以改進為硬敏感度保證。

They compute privacy loss per parameter (not for an entire model). By our preferred measure, the total privacy loss per participant on the MNIST dataset exceeds several thousand.
他們計算每個參數的隱私損失（而不是整個模型的）。按照我們首選的衡量標準，MNIST 數據集上每個參與者的總隱私損失超過數千。

A different, recent approach towards differentially private deep learning is explored by Phan et al. [45].
Phan 等人 [45] 探索了一種不同的、最近的差分隱私深度學習方法。

This work focuses on learning autoencoders. Privacy is based on perturbing the objective functions of these autoencoders.
這項工作側重於學習自編碼器。隱私基於擾動這些自編碼器的目標函數。

## 7. CONCLUSIONS
## 7. 結論

We demonstrate the training of deep neural networks with differential privacy, incurring a modest total privacy loss, computed over entire models with many parameters.
我們展示了具有差分隱私的深度神經網絡訓練，產生了適度的總隱私損失，該損失是在具有許多參數的整個模型上計算的。

In our experiments for MNIST, we achieve 97% training accuracy and for CIFAR-10 we achieve 73% accuracy, both with $(8, 10^{-5})$-differential privacy.
在我們的 MNIST 實驗中，我們達到了 97% 的訓練準確率，對於 CIFAR-10，我們達到了 73% 的準確率，均在 $(8, 10^{-5})$-差分隱私下。

Our algorithms are based on a differentially private version of stochastic gradient descent; they run on the TensorFlow software library for machine learning.
我們的算法基於隨機梯度下降的差分隱私版本；它們運行在用於機器學習的 TensorFlow 軟件庫上。

Since our approach applies directly to gradient computations, it can be adapted to many other classical and more recent first-order optimization methods, such as NAG [43], Momentum [48], AdaGrad [15], or SVRG [31].
由於我們的方法直接應用於梯度計算，它可以適應許多其他經典的和最近的一階優化方法，如 NAG [43]、Momentum [48]、AdaGrad [15] 或 SVRG [31]。

A new tool, which may be of independent interest, is a mechanism for tracking privacy loss, the moments accountant.
一個可能具有獨立興趣的新工具是用於追蹤隱私損失的機制——矩會計師。

It permits tight automated analysis of the privacy loss of complex composite mechanisms that are currently beyond the reach of advanced composition theorems.
它允許對複雜複合機制的隱私損失進行緊密的自動分析，這些機制目前超出了高級組合定理的範圍。

A number of avenues for further work are attractive. In particular, we would like to consider other classes of deep networks.
許多進一步工作的途徑是有吸引力的。特別是，我們想考慮其他類別的深度網絡。

Our experience with MNIST and CIFAR-10 should be helpful, but we see many opportunities for new research, for example in applying our techniques to LSTMs used for language modeling tasks.
我們在 MNIST 和 CIFAR-10 方面的經驗應該是有幫助的，但我們看到了許多新研究的機會，例如將我們的技術應用於語言建模任務中使用的 LSTM。

In addition, we would like to obtain additional improvements in accuracy.
此外，我們希望獲得準確率的額外提高。

Many training datasets are much larger than those of MNIST and CIFAR-10; accuracy should benefit from their size.
許多訓練數據集比 MNIST 和 CIFAR-10 大得多；準確率應該受益於它們的大小。

## 8. ACKNOWLEDGMENTS
## 8. 致謝

We are grateful to Úlfar Erlingsson and Dan Ramage for many useful discussions, and to Mark Bun and Thomas Steinke for sharing a draft of [10].
我們感謝 Úlfar Erlingsson 和 Dan Ramage 的許多有益討論，以及 Mark Bun 和 Thomas Steinke 分享 [10] 的草稿。

## 9. REFERENCES
## 9. 參考文獻

[1] CIFAR-10 and CIFAR-100 datasets. www.cs.toronto.edu/~kriz/cifar.html.
[2] TensorFlow convolutional neural networks tutorial. www.tensorflow.org/tutorials/deep_cnn.
[3] TensorFlow: Large-scale machine learning on heterogeneous systems, 2015. Software available from tensorflow.org.
[4] C. C. Aggarwal. On k-anonymity and the curse of dimensionality. In VLDB, pages 901–909, 2005.
[5] R. Agrawal and R. Srikant. Privacy-preserving data mining. In SIGMOD, pages 439–450. ACM, 2000.
[6] R. Bassily, K. Nissim, A. Smith, T. Steinke, U. Stemmer, and J. Ullman. Algorithmic stability for adaptive data analysis. In STOC, pages 1046–1059. ACM, 2016.
[7] R. Bassily, A. D. Smith, and A. Thakurta. Private empirical risk minimization: Efficient algorithms and tight error bounds. In FOCS, pages 464–473. IEEE, 2014.
[8] A. Beimel, H. Brenner, S. P. Kasiviswanathan, and K. Nissim. Bounds on the sample complexity for private learning and private data release. Machine Learning, 94(3):401–437, 2014.
[9] J. Brickell and V. Shmatikov. The cost of privacy: Destruction of data-mining utility in anonymized data publishing. In KDD, pages 70–78. ACM, 2008.
[10] M. Bun and T. Steinke. Concentrated differential privacy: Simplifications, extensions, and lower bounds. In TCC-B.
[11] K. Chaudhuri, C. Monteleoni, and A. D. Sarwate. Differentially private empirical risk minimization. J. Machine Learning Research, 12:1069–1109, 2011.
[12] R. Collobert, K. Kavukcuoglu, and C. Farabet. Torch7: A Matlab-like environment for machine learning. In BigLearn, NIPS Workshop, number EPFL-CONF-192376, 2011.
[13] D. D. Cox and N. Pinto. Beyond simple features: A large-scale feature search approach to unconstrained face recognition. In FG 2011, pages 8–15. IEEE, 2011.
[14] A. Daniely, R. Frostig, and Y. Singer. Toward deeper understanding of neural networks: The power of initialization and a dual view on expressivity. CoRR, abs/1602.05897, 2016.
[15] J. Duchi, E. Hazan, and Y. Singer. Adaptive subgradient methods for online learning and stochastic optimization. J. Machine Learning Research, 12:2121–2159, July 2011.
[16] C. Dwork. A firm foundation for private data analysis. Commun. ACM, 54(1):86–95, Jan. 2011.
[17] C. Dwork, K. Kenthapadi, F. McSherry, I. Mironov, and M. Naor. Our data, ourselves: Privacy via distributed noise generation. In EUROCRYPT, pages 486–503. Springer, 2006.
[18] C. Dwork and J. Lei. Differential privacy and robust statistics. In STOC, pages 371–380. ACM, 2009.
[19] C. Dwork, F. McSherry, K. Nissim, and A. Smith. Calibrating noise to sensitivity in private data analysis. In TCC, pages 265–284. Springer, 2006.
[20] C. Dwork and A. Roth. The algorithmic foundations of differential privacy. Foundations and Trends in Theoretical Computer Science, 9(3–4):211–407, 2014.
[21] C. Dwork and G. N. Rothblum. Concentrated differential privacy. CoRR, abs/1603.01887, 2016.
[22] C. Dwork, G. N. Rothblum, and S. Vadhan. Boosting and differential privacy. In FOCS, pages 51–60. IEEE, 2010.
[23] C. Dwork, K. Talwar, A. Thakurta, and L. Zhang. Analyze Gauss: Optimal bounds for privacy-preserving principal component analysis. In STOC, pages 11–20. ACM, 2014.
[24] M. Fredrikson, S. Jha, and T. Ristenpart. Model inversion attacks that exploit confidence information and basic countermeasures. In CCS, pages 1322–1333. ACM, 2015.
[25] I. Goodfellow. Efficient per-example gradient computations. CoRR, abs/1510.01799v2, 2015.
[26] B. Graham. Fractional max-pooling. CoRR, abs/1412.6071, 2014.
[27] A. Gupta, K. Ligett, F. McSherry, A. Roth, and K. Talwar. Differentially private combinatorial optimization. In SODA, pages 1106–1125, 2010.
[28] K. He, X. Zhang, S. Ren, and J. Sun. Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification. In ICCV, pages 1026–1034. IEEE, 2015.
[29] R. Ierusalimschy, L. H. de Figueiredo, and W. Filho. Lua—an extensible extension language. Software: Practice and Experience, 26(6):635–652, 1996.
[30] K. Jarrett, K. Kavukcuoglu, M. Ranzato, and Y. LeCun. What is the best multi-stage architecture for object recognition? In ICCV, pages 2146–2153. IEEE, 2009.
[31] R. Johnson and T. Zhang. Accelerating stochastic gradient descent using predictive variance reduction. In NIPS, pages 315–323, 2013.
[32] P. Kairouz, S. Oh, and P. Viswanath. The composition theorem for differential privacy. In ICML, pages 1376–1385. ACM, 2015.
[33] S. P. Kasiviswanathan, H. K. Lee, K. Nissim, S. Raskhodnikova, and A. Smith. What can we learn privately? SIAM J. Comput., 40(3):793–826, 2011.
[34] D. Kifer, A. D. Smith, and A. Thakurta. Private convex optimization for empirical risk minimization with applications to high-dimensional regression. In COLT, pages 25.1–25.40, 2012.
[35] A. Krizhevsky, I. Sutskever, and G. E. Hinton. ImageNet classification with deep convolutional neural networks. In NIPS, pages 1097–1105, 2012.
[36] Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11), 1998.
[37] Y. Lindell and B. Pinkas. Privacy preserving data mining. In CRYPTO, pages 36–54. Springer, 2000.
[38] C. J. Maddison, A. Huang, I. Sutskever, and D. Silver. Move evaluation in Go using deep convolutional neural networks. In ICLR, 2015.
[39] F. McSherry and I. Mironov. Differentially private recommender systems: Building privacy into the Netflix Prize contenders. In KDD, pages 627–636. ACM, 2009.
[40] F. D. McSherry. Privacy integrated queries: An extensible platform for privacy-preserving data analysis. In SIGMOD, pages 19–30. ACM, 2009.
[41] T. Mikolov, K. Chen, G. Corrado, and J. Dean. Efficient estimation of word representations in vector space. CoRR, abs/1301.3781, 2013.
[42] I. Mironov. Rényi differential privacy. Private communication, 2016.
[43] Y. Nesterov. Introductory Lectures on Convex Optimization. A Basic Course. Springer, 2004.
[44] J. Pennington, R. Socher, and C. D. Manning. GloVe: Global vectors for word representation. In EMNLP, pages 1532–1543, 2014.
[45] N. Phan, Y. Wang, X. Wu, and D. Dou. Differential privacy preservation for deep auto-encoders: an application of human behavior prediction. In AAAI, pages 1309–1316, 2016.
[46] N. Pinto, Z. Stone, T. E. Zickler, and D. Cox. Scaling up biologically-inspired computer vision: A case study in unconstrained face recognition on Facebook. In CVPR, pages 35–42. IEEE, 2011.
[47] R. M. Rogers, A. Roth, J. Ullman, and S. P. Vadhan. Privacy odometers and filters: Pay-as-you-go composition. In NIPS.
[48] D. E. Rumelhart, G. E. Hinton, and R. J. Williams. Learning representations by back-propagating errors. Nature, 323:533–536, Oct. 1986.
[49] A. Saxe, P. W. Koh, Z. Chen, M. Bhand, B. Suresh, and A. Ng. On random weights and unsupervised feature learning. In ICML, pages 1089–1096. ACM, 2011.
[50] R. Shokri and V. Shmatikov. Privacy-preserving deep learning. In CCS, pages 1310–1321. ACM, 2015.
[51] D. Silver, A. Huang, C. J. Maddison, A. Guez, L. Sifre, G. van den Driessche, J. Schrittwieser, I. Antonoglou, V. Panneershelvam, M. Lanctot, S. Dieleman, D. Grewe, J. Nham, N. Kalchbrenner, I. Sutskever, T. Lillicrap, M. Leach, K. Kavukcuoglu, T. Graepel, and D. Hassabis. Mastering the game of Go with deep neural networks and tree search. Nature, 529(7587):484–489, 2016.
[52] S. Song, K. Chaudhuri, and A. Sarwate. Stochastic gradient descent with differentially private updates. In GlobalSIP Conference, 2013.
[53] L. Sweeney. k-anonymity: A model for protecting privacy. International J. of Uncertainty, Fuzziness and Knowledge-Based Systems, 10(05):557–570, 2002.
[54] C. Szegedy, W. Liu, Y. Jia, P. Sermanet, S. Reed, D. Anguelov, D. Erhan, V. Vanhoucke, and A. Rabinovich. Going deeper with convolutions. In CVPR, pages 1–9. IEEE, 2015.
[55] S. Tu, R. Roelofs, S. Venkataraman, and B. Recht. Large scale kernel learning using block coordinate descent. CoRR, abs/1602.05310, 2016.
[56] O. Vinyals, L. Kaiser, T. Koo, S. Petrov, I. Sutskever, and G. E. Hinton. Grammar as a foreign language. In NIPS, pages 2773–2781, 2015.
[57] X. Wu, A. Kumar, K. Chaudhuri, S. Jha, and J. F. Naughton. Differentially private stochastic gradient descent for in-RDBMS analytics. CoRR, abs/1606.04722, 2016.

## APPENDIX
## 附錄

### A. PROOF OF THEOREM 2
### A. 定理 2 的證明

Here we restate and prove Theorem 2.
我們在此重述並證明定理 2。

**Theorem 2.** *Let $\alpha_{\mathcal{M}}(\lambda)$ defined as*
**定理 2.** *設 $\alpha_{\mathcal{M}}(\lambda)$ 定義為*

$$\alpha_{\mathcal{M}}(\lambda) \triangleq \max_{\text{aux}, d, d'} \alpha_{\mathcal{M}}(\lambda; \text{aux}, d, d'),$$

*where the maximum is taken over all auxiliary inputs and neighboring databases $d, d'$. Then*
*其中最大值取自所有輔助輸入和相鄰數據庫 $d, d'$。那麼*

1. *[Composability] Suppose that a mechanism $\mathcal{M}$ consists of a sequence of adaptive mechanisms $\mathcal{M}_1, \dots, \mathcal{M}_k$ where $\mathcal{M}_i : \prod_{j=1}^{i-1} \mathcal{R}_j \times \mathcal{D} \rightarrow \mathcal{R}_i$. Then, for any $\lambda$*
1. *[可組合性] 假設機制 $\mathcal{M}$ 由一系列自適應機制 $\mathcal{M}_1, \dots, \mathcal{M}_k$ 組成，其中 $\mathcal{M}_i : \prod_{j=1}^{i-1} \mathcal{R}_j \times \mathcal{D} \rightarrow \mathcal{R}_i$。那麼，對於任何 $\lambda$*

$$\alpha_{\mathcal{M}}(\lambda) \le \sum_{i=1}^k \alpha_{\mathcal{M}_i}(\lambda).$$

2. *[Tail bound] For any $\varepsilon > 0$, the mechanism $\mathcal{M}$ is $(\varepsilon, \delta)$-differentially private for*
2. *[尾部界限] 對於任何 $\varepsilon > 0$，機制 $\mathcal{M}$ 是 $(\varepsilon, \delta)$-差分隱私的，其中*

$$\delta = \min_\lambda \exp(\alpha_{\mathcal{M}}(\lambda) - \lambda\varepsilon).$$

**Proof. Composition of moments.** For brevity, let $\mathcal{M}_{1:i}$ denote $(\mathcal{M}_1, \dots, \mathcal{M}_i)$, and similarly let $o_{1:i}$ denote $(o_1, \dots, o_i)$. For neighboring databases $d, d' \in D^n$, and a sequence of outcomes $o_1, \dots, o_k$ we write
**證明. 矩的組合。** 為簡潔起見，設 $\mathcal{M}_{1:i}$ 表示 $(\mathcal{M}_1, \dots, \mathcal{M}_i)$，同樣設 $o_{1:i}$ 表示 $(o_1, \dots, o_i)$。對於相鄰數據庫 $d, d' \in D^n$，以及結果序列 $o_1, \dots, o_k$，我們寫作

$$c(o_{1:k}; \mathcal{M}_{1:k}, o_{1:(k-1)}, d, d') = \sum_{i=1}^k c(o_i; \mathcal{M}_i, o_{1:(i-1)}, d, d').$$

Thus
因此

$$\mathbb{E}_{o'_{1:k} \sim \mathcal{M}_{1:k}(d)} [\exp(\lambda c(o'_{1:k}; \mathcal{M}_{1:k}, d, d')) \mid \forall i < k: o'_i = o_i]$$
$$= \exp \left( \sum_{i=1}^k \alpha_i(\lambda; o_{1:(i-1)}, d, d') \right).$$

The claim follows.
命題得證。

**Tail bound by moments.** The proof is based on the standard Markov’s inequality argument used in proofs of measure concentration. We have
**矩的尾部界限。** 證明基於測度集中證明中使用的標準馬爾可夫不等式論證。我們有

$$\text{Pr}_{o \sim \mathcal{M}(d)} [c(o) \ge \varepsilon] \le \exp(\alpha - \lambda\varepsilon).$$

Let $B = \{o: c(o) \ge \varepsilon\}$. Then for any $S$,
設 $B = \{o: c(o) \ge \varepsilon\}$。那麼對於任何 $S$，

$$\text{Pr}[\mathcal{M}(d) \in S] \le \exp(\varepsilon) \text{Pr}[\mathcal{M}(d') \in S] + \exp(\alpha - \lambda\varepsilon).$$

The second part follows by an easy calculation. $\square$
第二部分通過簡單的計算得出。 $\square$

The proof demonstrates a tail bound on the privacy loss, making it stronger than differential privacy for a fixed value of $\varepsilon, \delta$.
證明展示了隱私損失的尾部界限，使其比固定 $\varepsilon, \delta$ 值的差分隱私更強。

### B. PROOF OF LEMMA 3
### B. 引理 3 的證明

The proof of the main theorem relies on the following moments bound on Gaussian mechanism with random sampling.
主定理的證明依賴於以下具有隨機採樣的高斯機制的矩界限。

**Lemma 3.** *Suppose that $f : D \rightarrow \mathbb{R}^p$ with $||f(\cdot)||_2 \le 1$. Let $\sigma \ge 1$ and let $J$ be a sample from $[n]$ where each $i \in [n]$ is chosen independently with probability $q < \frac{1}{16\sigma}$. Then for any positive integer $\lambda \le \sigma^2 \ln \frac{1}{q\sigma}$, the mechanism $\mathcal{M}(d) = \sum_{i \in J} f(d_i) + \mathcal{N}(0, \sigma^2\mathbf{I})$ satisfies*
**引理 3.** *假設 $f : D \rightarrow \mathbb{R}^p$ 且 $||f(\cdot)||_2 \le 1$。設 $\sigma \ge 1$，令 $J$ 為從 $[n]$ 中抽取的樣本，其中每個 $i \in [n]$ 以概率 $q < \frac{1}{16\sigma}$ 獨立選擇。那麼對於任何正整數 $\lambda \le \sigma^2 \ln \frac{1}{q\sigma}$，機制 $\mathcal{M}(d) = \sum_{i \in J} f(d_i) + \mathcal{N}(0, \sigma^2\mathbf{I})$ 滿足*

$$\alpha_{\mathcal{M}}(\lambda) \le \frac{q^2\lambda(\lambda+1)}{(1-q)\sigma^2} + O(q^3\lambda^3/\sigma^3).$$

**Proof.** Fix $d'$ and let $d = d' \cup \{d_n\}$. Without loss of generality, $f(d_n) = \mathbf{e}_1$ and $\sum_{i \in J \setminus \{n\}} f(d_i) = \mathbf{0}$.
**證明。** 固定 $d'$ 並設 $d = d' \cup \{d_n\}$。不失一般性，設 $f(d_n) = \mathbf{e}_1$ 且 $\sum_{i \in J \setminus \{n\}} f(d_i) = \mathbf{0}$。

Thus $\mathcal{M}(d)$ and $\mathcal{M}(d')$ are distributed identically except for the first coordinate and hence we have a one-dimensional problem.
因此 $\mathcal{M}(d)$ 和 $\mathcal{M}(d')$ 除了第一個坐標外分佈相同，因此我們有一個一維問題。

Let $\mu_0$ denote the pdf of $\mathcal{N}(0, \sigma^2)$ and let $\mu_1$ denote the pdf of $\mathcal{N}(1, \sigma^2)$. Thus:
設 $\mu_0$ 表示 $\mathcal{N}(0, \sigma^2)$ 的 pdf，$\mu_1$ 表示 $\mathcal{N}(1, \sigma^2)$ 的 pdf。因此：

$$\mathcal{M}(d') \sim \mu_0,$$
$$\mathcal{M}(d) \sim \mu \triangleq (1-q)\mu_0 + q\mu_1.$$

We want to show that
我們想要證明

$$\mathbb{E}_{z \sim \mu} [(\mu(z)/\mu_0(z))^\lambda] \le \alpha,$$
$$\text{and } \mathbb{E}_{z \sim \mu_0} [(\mu_0(z)/\mu(z))^\lambda] \le \alpha,$$

for some explicit $\alpha$ to be determined later.
對於稍後確定的某個顯式 $\alpha$。

We will use the same method to prove both bounds. Assume we have two distributions $\nu_0$ and $\nu_1$, and we wish to bound
我們將使用相同的方法證明這兩個界限。假設我們有兩個分佈 $\nu_0$ 和 $\nu_1$，我們希望限制

$$\mathbb{E}_{z \sim \nu_0} [(\nu_0(z)/\nu_1(z))^\lambda] = \mathbb{E}_{z \sim \nu_1} [(\nu_0(z)/\nu_1(z))^{\lambda+1}].$$

Using binomial expansion, we have
使用二項式展開，我們有

$$\mathbb{E}_{z \sim \nu_1} [(\nu_0(z)/\nu_1(z))^{\lambda+1}] = \sum_{t=0}^{\lambda+1} \binom{\lambda+1}{t} \mathbb{E}_{z \sim \nu_1} [((\nu_0(z) - \nu_1(z))/\nu_1(z))^t]. (5)$$

The first term in (5) is 1, and the second term is 0.
(5) 中的第一項是 1，第二項是 0。

To prove the lemma it suffices to show show that for both $\nu_0 = \mu, \nu_1 = \mu_0$ and $\nu_0 = \mu_0, \nu_1 = \mu$, the third term is bounded by $q^2\lambda(\lambda+1)/(1-q)\sigma^2$ and that this bound dominates the sum of the remaining terms.
為了證明引理，只需證明對於 $\nu_0 = \mu, \nu_1 = \mu_0$ 和 $\nu_0 = \mu_0, \nu_1 = \mu$，第三項都受 $q^2\lambda(\lambda+1)/(1-q)\sigma^2$ 限制，並且該界限主導剩餘項的總和。

Under the assumptions on $q, \sigma$, and $\lambda$, it is easy to check that the three terms, and their sum, drop off geometrically fast in $t$ for $t > 3$.
在關於 $q, \sigma$ 和 $\lambda$ 的假設下，很容易檢查這三項及其總和在 $t > 3$ 時隨 $t$ 幾何級數下降。

Hence the binomial expansion (5) is dominated by the $t = 3$ term, which is $O(q^3\lambda^3/\sigma^3)$. The claim follows. $\square$
因此，二項式展開 (5) 由 $t = 3$ 項主導，即 $O(q^3\lambda^3/\sigma^3)$。命題得證。 $\square$

To derive Theorem 1, we use the above moments bound along with the tail bound from Theorem 2, optimizing over the choice of $\lambda$.
為了推導定理 1，我們使用上述矩界限以及定理 2 的尾部界限，並優化 $\lambda$ 的選擇。

**Theorem 1.** *There exist constants $c_1$ and $c_2$ so that given the sampling probability $q = L/N$ and the number of steps $T$, for any $\varepsilon < c_1q^2T$, Algorithm 1 is $(\varepsilon, \delta)$-differentially private for any $\delta > 0$ if we choose*
**定理 1.** *存在常數 $c_1$ 和 $c_2$，使得給定採樣概率 $q = L/N$ 和步驟數 $T$，對於任何 $\varepsilon < c_1q^2T$，如果我們選擇以下 $\sigma$，則算法 1 對於任何 $\delta > 0$ 都是 $(\varepsilon, \delta)$-差分隱私的：*

$$\sigma \ge c_2 \frac{q \sqrt{T \log(1/\delta)}}{\varepsilon}.$$

**Proof.** Assume for now that $\sigma, \lambda$ satisfy the conditions in Lemma 3.
**證明。** 暫且假設 $\sigma, \lambda$ 滿足引理 3 中的條件。

By Theorem 2.1 and Lemma 3, the log moment of Algorithm 1 can be bounded as follows $\alpha(\lambda) \le T q^2\lambda^2/\sigma^2$.
根據定理 2.1 和引理 3，算法 1 的對數矩可以限制如下 $\alpha(\lambda) \le T q^2\lambda^2/\sigma^2$。

By Theorem 2, to guarantee Algorithm 1 to be $(\varepsilon, \delta)$-differentially private, it suffices that
根據定理 2，為了保證算法 1 是 $(\varepsilon, \delta)$-差分隱私的，只需

$$Tq^2\lambda^2/\sigma^2 \le \lambda\varepsilon/2,$$
$$\exp(-\lambda\varepsilon/2) \le \delta.$$

In addition, we need $\lambda \le \sigma^2 \log(1/q\sigma)$.
此外，我們需要 $\lambda \le \sigma^2 \log(1/q\sigma)$。

It is now easy to verify that when $\varepsilon = c_1q^2T$, we can satisfy all these conditions by setting
現在很容易驗證，當 $\varepsilon = c_1q^2T$ 時，我們可以通過設置以下各項來滿足所有這些條件

$$\sigma = c_2 \frac{q\sqrt{T \log(1/\delta)}}{\varepsilon}$$

for some explicit constants $c_1$ and $c_2$. $\square$
對於某些顯式常數 $c_1$ 和 $c_2$。 $\square$

### C. FROM DIFFERENTIAL PRIVACY TO MOMENTS BOUNDS
### C. 從差分隱私到矩界限

One can also translate a differential privacy guarantee into a moment bound.
人們也可以將差分隱私保證轉化為矩界限。

**Lemma C.1.** *Let $\mathcal{M}$ be $\varepsilon$-differentially private. Then for any $\lambda > 0$, $\mathcal{M}$ satisfies*
**引理 C.1.** *設 $\mathcal{M}$ 為 $\varepsilon$-差分隱私。那麼對於任何 $\lambda > 0$，$\mathcal{M}$ 滿足*

$$\alpha_\lambda \le \lambda\varepsilon(e^\varepsilon - 1) + \lambda^2\varepsilon^2e^{2\varepsilon}/2.$$

**Proof.** Let $Z$ denote the random variable $c(\mathcal{M}(d))$. Then differential privacy implies that
**證明。** 設 $Z$ 表示隨機變量 $c(\mathcal{M}(d))$。那麼差分隱私意味著

*   $\mu \triangleq \mathbb{E}[Z] \le \varepsilon(e^\varepsilon - 1)$.
*   $|Z| \le \varepsilon$, so that $|Z - \mu| \le \varepsilon e^\varepsilon$.

Then $\mathbb{E}[\exp(\lambda Z)] = \exp(\lambda\mu) \cdot \mathbb{E}[\exp(\lambda(Z - \mu))]$.
那麼 $\mathbb{E}[\exp(\lambda Z)] = \exp(\lambda\mu) \cdot \mathbb{E}[\exp(\lambda(Z - \mu))]$。

Since $Z$ is in a bounded range $[-\varepsilon e^\varepsilon, \varepsilon e^\varepsilon]$ and $f(x) = \exp(\lambda x)$ is convex, we can bound $f(x)$ by a linear interpolation between the values at the two endpoints of the range.
由於 $Z$ 在有界範圍 $[-\varepsilon e^\varepsilon, \varepsilon e^\varepsilon]$ 內，且 $f(x) = \exp(\lambda x)$ 是凸函數，我們可以通過範圍兩個端點值的線性插值來限制 $f(x)$。

Basic calculus then implies that
基本微積分表明

$$\mathbb{E}[f(Z)] \le f(\mathbb{E}[Z]) \cdot \exp(\lambda^2\varepsilon^2 \exp(2\varepsilon)/2),$$

which concludes the proof. $\square$
證明結束。 $\square$

Lemma C.1 and Theorem 2 give a way of getting a composition theorem for differentially private mechanisms, which is roughly equivalent to unrolling the proof of the strong composition theorem of [22].
引理 C.1 和定理 2 提供了一種獲得差分隱私機制組合定理的方法，這大致相當於展開 [22] 的強組合定理的證明。

The power of the moments accountant comes from the fact that, for many mechanisms of choice, directly bounding in the moments gives a stronger guarantee than one would get by establishing differential privacy and applying Lemma C.1.
矩會計師的強大之處在於，對於許多選擇的機制，直接限制矩比建立差分隱私並應用引理 C.1 能提供更強的保證。

### D. HYPERPARAMETER SEARCH
### D. 超參數搜索

Here we state Theorem 10.2 from [27] that we use to account for the cost of hyperparameter search.
在這裡，我們陳述 [27] 中的定理 10.2，我們用它來核算超參數搜索的成本。

**Theorem D.1** (Gupta et al. [27]). *Let $\mathcal{M}$ be an $\varepsilon$-differentially private mechanism such that for a query function $q$ with sensitivity 1, and a parameter $Q$, it holds that $\text{Pr}_{r \sim \mathcal{M}(d)}[q(d, r) \ge Q] \ge p$ for some $p \in (0, 1)$. Then for any $\delta > 0$ and any $\varepsilon' \in (0, \frac{1}{2})$, there is a mechanism $\mathcal{M}'$ which satisfies the following properties:*
**定理 D.1** (Gupta 等人 [27])。*設 $\mathcal{M}$ 為 $\varepsilon$-差分隱私機制，使得對於敏感度為 1 的查詢函數 $q$ 和參數 $Q$，對於某個 $p \in (0, 1)$，滿足 $\text{Pr}_{r \sim \mathcal{M}(d)}[q(d, r) \ge Q] \ge p$。那麼對於任何 $\delta > 0$ 和任何 $\varepsilon' \in (0, \frac{1}{2})$，存在一個機制 $\mathcal{M}'$ 滿足以下屬性：*

*   $\text{Pr}_{r \sim \mathcal{M}'(d)} [q(d, r) \ge Q - \frac{4}{\varepsilon'} \log(\frac{1}{\varepsilon'\delta p})] \ge 1 - \delta$.
*   $\mathcal{M}'$ makes $(\frac{1}{\varepsilon'\delta p})^2 \log(\frac{1}{\varepsilon'\delta p})$ calls to $\mathcal{M}$.
*   $\mathcal{M}'$ is $(\varepsilon + 8\varepsilon')$-differentially private.

Suppose that we have a differentially private mechanism $\mathcal{M}_i$ for each of $K$ choices of hyperparameters. Let $\tilde{\mathcal{M}}$ be the mechanism that picks a random choice of hyperparameters, and runs the corresponding $\mathcal{M}_i$.
假設我們對於 $K$ 個超參數選擇中的每一個都有一個差分隱私機制 $\mathcal{M}_i$。設 $\tilde{\mathcal{M}}$ 為隨機選擇超參數並運行相應 $\mathcal{M}_i$ 的機制。

Let $q(d, r)$ denote the number of examples from the validation set the $r$ labels correctly, and let $Q$ be a target accuracy.
設 $q(d, r)$ 表示驗證集中 $r$ 標籤正確的樣本數，設 $Q$ 為目標準確率。

Assuming that one of the hyperparameter settings gets accuracy at least $Q$, $\tilde{\mathcal{M}}$ satisfies the pre-conditions of the theorem for $p = \frac{1}{K}$.
假設其中一個超參數設置至少獲得 $Q$ 的準確率，則 $\tilde{\mathcal{M}}$ 滿足定理的前提條件，其中 $p = \frac{1}{K}$。

Then with high probability, the mechanism implied by the theorem gets accuracy close to $Q$.
那麼該定理隱含的機制以高概率獲得接近 $Q$ 的準確率。

We remark that the proof of Theorem D.1 actually implies a stronger $\max(\varepsilon, 8\varepsilon')$-differential privacy for the setting of interest here.
我們注意到，定理 D.1 的證明實際上意味著對於這裡感興趣的設置，有更強的 $\max(\varepsilon, 8\varepsilon')$-差分隱私。

Putting in some numbers, for a target accuracy of 95% on a validation set of size 10,000, we get $Q = 9,500$.
代入一些數字，對於大小為 10,000 的驗證集上的 95% 目標準確率，我們得到 $Q = 9,500$。

Thus, if, for instance, we allow $\varepsilon' = 0.5$, and $\delta = 0.05$, we lose at most 1% in accuracy as long as $100 > 8 \ln \frac{40}{p}$.
因此，例如，如果我們允許 $\varepsilon' = 0.5$ 和 $\delta = 0.05$，只要 $100 > 8 \ln \frac{40}{p}$，我們最多損失 1% 的準確率。

This is satisfied as long as $p \ge \frac{1}{6700}$. In other words, one can try 6,700 different parameter settings at privacy cost $\varepsilon = 4$ for the validation set.
只要 $p \ge \frac{1}{6700}$，這就滿足了。換句話說，人們可以在隱私成本 $\varepsilon = 4$ 下為驗證集嘗試 6,700 種不同的參數設置。

In our experiments, we tried no more than a hundred settings, so that this bound is easily satisfied.
在我們的實驗中，我們嘗試的設置不超過一百個，因此這個界限很容易滿足。

In practice, as our graphs show, $p$ for our hyperparameter search is significantly larger than $\frac{1}{K}$, so that a slightly smaller $\varepsilon'$ should suffice.
在實踐中，正如我們的圖表所示，我們的超參數搜索的 $p$ 顯著大於 $\frac{1}{K}$，因此稍微小一點的 $\varepsilon'$ 應該就足夠了。
