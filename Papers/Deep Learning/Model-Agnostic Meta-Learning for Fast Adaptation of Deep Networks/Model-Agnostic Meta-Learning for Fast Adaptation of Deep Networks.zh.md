---
title: Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks
field: Deep_Learning
status: Imported
created_date: 2026-01-18
pdf_link: "[[Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks.pdf]]"
tags:
  - paper
  - Deep_learning
---

# Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks
# 用於深度網路快速適應的模型無關元學習

**Chelsea Finn** 1 **Pieter Abbeel** 1 2 **Sergey Levine** 1
**Chelsea Finn** 1 **Pieter Abbeel** 1 2 **Sergey Levine** 1

**Abstract**
**摘要**

We propose an algorithm for meta-learning that is model-agnostic, in the sense that it is compatible with any model trained with gradient descent and applicable to a variety of different learning problems, including classification, regression, and reinforcement learning. The goal of meta-learning is to train a model on a variety of learning tasks, such that it can solve new learning tasks using only a small number of training samples. In our approach, the parameters of the model are explicitly trained such that a small number of gradient steps with a small amount of training data from a new task will produce good generalization performance on that task. In effect, our method trains the model to be easy to fine-tune. We demonstrate that this approach leads to state-of-the-art performance on two few-shot image classification benchmarks, produces good results on few-shot regression, and accelerates fine-tuning for policy gradient reinforcement learning with neural network policies.

我們提出了一種模型無關的元學習（Meta-Learning）演算法，所謂模型無關，是指它與任何使用梯度下降訓練的模型相容，並適用於各種不同的學習問題，包括分類、回歸和強化學習。元學習的目標是在各種學習任務上訓練模型，使其僅需少量的訓練樣本即可解決新的學習任務。在我們的方法中，我們明確地訓練模型的參數，使得利用來自新任務的少量訓練數據進行幾次梯度步驟後，就能在該任務上產生良好的泛化性能。實際上，我們的方法訓練出的模型很容易進行微調（fine-tune）。我們證明，這種方法在兩個少樣本（few-shot）圖像分類基準測試中達到了最先進的性能，在少樣本回歸上產生了良好的結果，並加速了神經網路策略在策略梯度強化學習中的微調。

## 1. Introduction
## 1. 介紹

Learning quickly is a hallmark of human intelligence, whether it involves recognizing objects from a few examples or quickly learning new skills after just minutes of experience. Our artificial agents should be able to do the same, learning and adapting quickly from only a few examples, and continuing to adapt as more data becomes available. This kind of fast and flexible learning is challenging, since the agent must integrate its prior experience with a small amount of new information, while avoiding overfitting to the new data. Furthermore, the form of prior experience and new data will depend on the task. As such, for the greatest applicability, the mechanism for learning to learn (or meta-learning) should be general to the task and the form of computation required to complete the task.

快速學習是人類智能的一個標誌，無論是從幾個例子中識別物體，還是在短短幾分鐘的體驗後快速學習新技能。我們的人工智慧代理（agents）應該也能夠做到這一點，僅從幾個例子中快速學習和適應，並隨著更多數據的可用而繼續適應。這種快速而靈活的學習具有挑戰性，因為代理必須將其先前的經驗與少量的新資訊整合起來，同時避免對新數據產生過擬合（overfitting）。此外，先前經驗和新數據的形式將取決於任務。因此，為了獲得最大的適用性，學習如何學習（或稱元學習）的機制應該對任務以及完成任務所需的計算形式具有通用性。

In this work, we propose a meta-learning algorithm that is general and model-agnostic, in the sense that it can be directly applied to any learning problem and model that is trained with a gradient descent procedure. Our focus is on deep neural network models, but we illustrate how our approach can easily handle different architectures and different problem settings, including classification, regression, and policy gradient reinforcement learning, with minimal modification. In meta-learning, the goal of the trained model is to quickly learn a new task from a small amount of new data, and the model is trained by the meta-learner to be able to learn on a large number of different tasks.

在這項工作中，我們提出了一種通用且模型無關的元學習演算法，這意味著它可以直接應用於任何使用梯度下降程序訓練的學習問題和模型。我們的重點是深度神經網路模型，但我們說明了我們的方法如何只需極小的修改就能輕鬆處理不同的架構和不同的問題設置，包括分類、回歸和策略梯度強化學習。在元學習中，受訓模型的目標是從少量新數據中快速學習新任務，而元學習器（meta-learner）訓練該模型是為了使其能夠在大量不同的任務上進行學習。

The key idea underlying our method is to train the model’s initial parameters such that the model has maximal performance on a new task after the parameters have been updated through one or more gradient steps computed with a small amount of data from that new task. Unlike prior meta-learning methods that learn an update function or learning rule (Schmidhuber, 1987; Bengio et al., 1992; Andrychowicz et al., 2016; Ravi & Larochelle, 2017), our algorithm does not expand the number of learned parameters nor place constraints on the model architecture (e.g. by requiring a recurrent model (Santoro et al., 2016) or a Siamese network (Koch, 2015)), and it can be readily combined with fully connected, convolutional, or recurrent neural networks. It can also be used with a variety of loss functions, including differentiable supervised losses and non-differentiable reinforcement learning objectives.

我們方法的核心理念是訓練模型的初始參數，使得模型在利用來自新任務的少量數據計算出的一步或多步梯度更新參數後，在該新任務上具有最佳性能。與先前學習更新函數或學習規則的元學習方法（Schmidhuber, 1987; Bengio et al., 1992; Andrychowicz et al., 2016; Ravi & Larochelle, 2017）不同，我們的演算法不增加學習參數的數量，也不對模型架構施加限制（例如要求循環模型 (Santoro et al., 2016) 或孿生網路 (Koch, 2015)），並且可以隨時與全連接、卷積或循環神經網路結合使用。它還可以用於各種損失函數，包括可微分的監督損失和不可微分的強化學習目標。

The process of training a model’s parameters such that a few gradient steps, or even a single gradient step, can produce good results on a new task can be viewed from a feature learning standpoint as building an internal representation that is broadly suitable for many tasks. If the internal representation is suitable to many tasks, simply fine-tuning the parameters slightly (e.g. by primarily modifying the top layer weights in a feedforward model) can produce good results. In effect, our procedure optimizes for models that are easy and fast to fine-tune, allowing the adaptation to happen in the right space for fast learning. From a dynamical systems standpoint, our learning process can be viewed as maximizing the sensitivity of the loss functions of new tasks with respect to the parameters: when the sensitivity is high, small local changes to the parameters can lead to large improvements in the task loss.

訓練模型參數以使幾個梯度步驟，甚至單個梯度步驟，就能在新任務上產生良好結果的過程，從特徵學習的角度來看，可以視為構建一個廣泛適用於許多任務的內部表示。如果內部表示適用於許多任務，那麼只需稍微微調參數（例如主要修改前饋模型中的頂層權重）即可產生良好的結果。實際上，我們的程序優化了那些易於且能快速微調的模型，允許在正確的空間中進行適應以實現快速學習。從動力系統的角度來看，我們的學習過程可以視為最大化新任務損失函數對參數的敏感度：當敏感度高時，對參數進行小的局部更改可以在任務損失上帶來巨大的改進。

The primary contribution of this work is a simple model- and task-agnostic algorithm for meta-learning that trains a model’s parameters such that a small number of gradient updates will lead to fast learning on a new task. We demonstrate the algorithm on different model types, including fully connected and convolutional networks, and in several distinct domains, including few-shot regression, image classification, and reinforcement learning. Our evaluation shows that our meta-learning algorithm compares favorably to state-of-the-art one-shot learning methods designed specifically for supervised classification, while using fewer parameters, but that it can also be readily applied to regression and can accelerate reinforcement learning in the presence of task variability, substantially outperforming direct pretraining as initialization.

這項工作的主要貢獻是一個簡單的、模型和任務無關的元學習演算法，它訓練模型的參數，使得少量的梯度更新就能導致對新任務的快速學習。我們在不同的模型類型（包括全連接和卷積網路）以及幾個不同的領域（包括少樣本回歸、圖像分類和強化學習）中展示了該演算法。我們的評估表明，我們的元學習演算法與專為監督分類設計的最先進的一次學習（one-shot learning）方法相比毫不遜色，同時使用的參數更少，而且它還可以輕鬆應用於回歸，並在存在任務變異性的情況下加速強化學習，其表現大幅優於直接預訓練作為初始化的方法。

## 2. Model-Agnostic Meta-Learning
## 2. 模型無關元學習

We aim to train models that can achieve rapid adaptation, a problem setting that is often formalized as few-shot learning. In this section, we will define the problem setup and present the general form of our algorithm.

我們的目標是訓練能夠實現快速適應的模型，這個問題設置通常被形式化為少樣本學習。在本節中，我們將定義問題設置並提出我們演算法的一般形式。

### 2.1. Meta-Learning Problem Set-Up
### 2.1. 元學習問題設置

The goal of few-shot meta-learning is to train a model that can quickly adapt to a new task using only a few datapoints and training iterations. To accomplish this, the model or learner is trained during a meta-learning phase on a set of tasks, such that the trained model can quickly adapt to new tasks using only a small number of examples or trials. In effect, the meta-learning problem treats entire tasks as training examples. In this section, we formalize this meta-learning problem setting in a general manner, including brief examples of different learning domains. We will discuss two different learning domains in detail in Section 3.

少樣本元學習的目標是訓練一個模型，使其能夠僅使用極少的數據點和訓練迭代就能快速適應新任務。為了實現這一點，模型或學習器在元學習階段在一組任務上進行訓練，以便受訓後的模型可以僅使用少量的範例或試驗就能快速適應新任務。實際上，元學習問題將整個任務視為訓練範例。在本節中，我們以通用的方式形式化這個元學習問題設置，包括不同學習領域的簡短範例。我們將在第 3 節詳細討論兩個不同的學習領域。

We consider a model, denoted $f$, that maps observations $\mathbf{x}$ to outputs $\mathbf{a}$. During meta-learning, the model is trained to be able to adapt to a large or infinite number of tasks. Since we would like to apply our framework to a variety of learning problems, from classification to reinforcement learning, we introduce a generic notion of a learning task below. Formally, each task $\mathcal{T} = \{\mathcal{L}(\mathbf{x}_1, \mathbf{a}_1, \ldots, \mathbf{x}_H, \mathbf{a}_H), q(\mathbf{x}_1), q(\mathbf{x}_{t+1}|\mathbf{x}_t, \mathbf{a}_t), H\}$ consists of a loss function $\mathcal{L}$, a distribution over initial observations $q(\mathbf{x}_1)$, a transition distribution $q(\mathbf{x}_{t+1}|\mathbf{x}_t, \mathbf{a}_t)$, and an episode length $H$. In i.i.d. supervised learning problems, the length $H = 1$. The model may generate samples of length $H$ by choosing an output $\mathbf{a}_t$ at each time $t$. The loss $\mathcal{L}(\mathbf{x}_1, \mathbf{a}_1, \ldots, \mathbf{x}_H, \mathbf{a}_H) \rightarrow \mathbb{R}$, provides task-specific feedback, which might be in the form of a misclassification loss or a cost function in a Markov decision process.

我們考慮一個模型，記為 $f$，它將觀察值 $\mathbf{x}$ 映射到輸出 $\mathbf{a}$。在元學習過程中，模型被訓練以適應大量或無限數量的任務。由於我們希望將我們的框架應用於從分類到強化學習的各種學習問題，我們在下面引入學習任務的通用概念。形式上，每個任務 $\mathcal{T} = \{\mathcal{L}(\mathbf{x}_1, \mathbf{a}_1, \ldots, \mathbf{x}_H, \mathbf{a}_H), q(\mathbf{x}_1), q(\mathbf{x}_{t+1}|\mathbf{x}_t, \mathbf{a}_t), H\}$ 由損失函數 $\mathcal{L}$、初始觀察分佈 $q(\mathbf{x}_1)$、轉移分佈 $q(\mathbf{x}_{t+1}|\mathbf{x}_t, \mathbf{a}_t)$ 和片段長度 $H$ 組成。在獨立同分佈（i.i.d.）監督學習問題中，長度 $H = 1$。模型可以通過在每個時間 $t$ 選擇輸出 $\mathbf{a}_t$ 來生成長度為 $H$ 的樣本。損失 $\mathcal{L}(\mathbf{x}_1, \mathbf{a}_1, \ldots, \mathbf{x}_H, \mathbf{a}_H) \rightarrow \mathbb{R}$ 提供任務特定的反饋，這可能是誤分類損失的形式，或者是馬可夫決策過程中的成本函數。

Figure 1. Diagram of our model-agnostic meta-learning algorithm (MAML), which optimizes for a representation $\theta$ that can quickly adapt to new tasks.

圖 1. 我們的模型無關元學習演算法（MAML）的圖解，該演算法針對可以快速適應新任務的表示 $\theta$ 進行了最佳化。

In our meta-learning scenario, we consider a distribution over tasks $p(\mathcal{T})$ that we want our model to be able to adapt to. In the $K$-shot learning setting, the model is trained to learn a new task $\mathcal{T}_i$ drawn from $p(\mathcal{T})$ from only $K$ samples drawn from $q_i$ and feedback $\mathcal{L}_{\mathcal{T}_i}$ generated by $\mathcal{T}_i$. During meta-training, a task $\mathcal{T}_i$ is sampled from $p(\mathcal{T})$, the model is trained with $K$ samples and feedback from the corresponding loss $\mathcal{L}_{\mathcal{T}_i}$ from $\mathcal{T}_i$, and then tested on new samples from $\mathcal{T}_i$. The model $f$ is then improved by considering how the *test* error on new data from $q_i$ changes with respect to the parameters. In effect, the test error on sampled tasks $\mathcal{T}_i$ serves as the training error of the meta-learning process. At the end of meta-training, new tasks are sampled from $p(\mathcal{T})$, and meta-performance is measured by the model’s performance after learning from $K$ samples. Generally, tasks used for meta-testing are held out during meta-training.

在我們的元學習場景中，我們考慮一個任務分佈 $p(\mathcal{T})$，我們希望我們的模型能夠適應它。在 $K$-shot 學習設置中，模型被訓練以僅從 $q_i$ 中抽取的 $K$ 個樣本和由 $\mathcal{T}_i$ 生成的反饋 $\mathcal{L}_{\mathcal{T}_i}$ 來學習從 $p(\mathcal{T})$ 中抽取的新任務 $\mathcal{T}_i$。在元訓練期間，從 $p(\mathcal{T})$ 中採樣一個任務 $\mathcal{T}_i$，模型使用 $K$ 個樣本和來自 $\mathcal{T}_i$ 的相應損失 $\mathcal{L}_{\mathcal{T}_i}$ 的反饋進行訓練，然後在來自 $\mathcal{T}_i$ 的新樣本上進行測試。然後透過考慮來自 $q_i$ 的新數據的 *測試* 誤差如何隨參數變化來改進模型 $f$。實際上，採樣任務 $\mathcal{T}_i$ 的測試誤差充當了元學習過程的訓練誤差。在元訓練結束時，從 $p(\mathcal{T})$ 中採樣新任務，並通過從 $K$ 個樣本學習後的模型性能來衡量元性能。通常，用於元測試的任務在元訓練期間是保留不用的。

### 2.2. A Model-Agnostic Meta-Learning Algorithm
### 2.2. 一種模型無關的元學習演算法

In contrast to prior work, which has sought to train recurrent neural networks that ingest entire datasets (Santoro et al., 2016; Duan et al., 2016b) or feature embeddings that can be combined with nonparametric methods at test time (Vinyals et al., 2016; Koch, 2015), we propose a method that can learn the parameters of any standard model via meta-learning in such a way as to prepare that model for fast adaptation. The intuition behind this approach is that some internal representations are more transferrable than others. For example, a neural network might learn internal features that are broadly applicable to all tasks in $p(\mathcal{T})$, rather than a single individual task. How can we encourage the emergence of such general-purpose representations? We take an explicit approach to this problem: since the model will be fine-tuned using a gradient-based learning rule on a new task, we will aim to learn a model in such a way that this gradient-based learning rule can make rapid progress on new tasks drawn from $p(\mathcal{T})$, without overfitting. In effect, we will aim to find model parameters that are *sensitive* to changes in the task, such that small changes in the parameters will produce large improvements on the loss function of any task drawn from $p(\mathcal{T})$, when altered in the direction of the gradient of that loss (see Figure 1).

與先前試圖訓練攝取整個數據集的循環神經網路（Santoro et al., 2016; Duan et al., 2016b）或在測試時可與非參數方法結合的特徵嵌入（Vinyals et al., 2016; Koch, 2015）的工作相比，我們提出了一種方法，可以通過元學習學習任何標準模型的參數，從而為該模型的快速適應做好準備。這種方法背後的直覺是，某些內部表示比其他表示更具可遷移性。例如，神經網路可能會學習到廣泛適用於 $p(\mathcal{T})$ 中所有任務的內部特徵，而不是單個特定任務。我們如何鼓勵這種通用表示的出現？我們對這個問題採取了明確的方法：由於模型將在新任務上使用基於梯度的學習規則進行微調，我們的目標是學習一個模型，使得這種基於梯度的學習規則可以在從 $p(\mathcal{T})$ 中抽取的新任務上取得快速進展，而不會過擬合。實際上，我們的目標是找到對任務變化 *敏感* 的模型參數，這樣當參數沿著損失梯度的方向改變時，微小的參數變化將在從 $p(\mathcal{T})$ 中抽取的任何任務的損失函數上產生巨大的改進（見圖 1）。

We make no assumption on the form of the model, other than to assume that it is parametrized by some parameter vector $\theta$, and that the loss function is smooth enough in $\theta$ that we can use gradient-based learning techniques.

我們對模型的形式不做任何假設，只假設它由某個参数向量 $\theta$ 參數化，並且損失函數在 $\theta$ 上足夠平滑，以便我們可以使用基於梯度的學習技術。

Formally, we consider a model represented by a parametrized function $f_\theta$ with parameters $\theta$. When adapting to a new task $\mathcal{T}_i$, the model’s parameters $\theta$ become $\theta'_i$. In our method, the updated parameter vector $\theta'_i$ is computed using one or more gradient descent updates on task $\mathcal{T}_i$. For example, when using one gradient update,
$$ \theta'_i = \theta - \alpha \nabla_\theta \mathcal{L}_{\mathcal{T}_i}(f_\theta) $$
The step size $\alpha$ may be fixed as a hyperparameter or meta-learned. For simplicity of notation, we will consider one gradient update for the rest of this section, but using multiple gradient updates is a straightforward extension.

形式上，我們考慮由參數 $\theta$ 的參數化函數 $f_\theta$ 表示的模型。當適應新任務 $\mathcal{T}_i$ 時，模型的參數 $\theta$ 變為 $\theta'_i$。在我們的方法中，更新後的参数向量 $\theta'_i$ 是使用任務 $\mathcal{T}_i$ 上的一個或多個梯度下降更新計算的。例如，當使用一個梯度更新時，
$$ \theta'_i = \theta - \alpha \nabla_\theta \mathcal{L}_{\mathcal{T}_i}(f_\theta) $$
步長 $\alpha$ 可以固定為超參數或進行元學習。為了符號簡潔，我們將在本節的其餘部分考慮一個梯度更新，但使用多個梯度更新是一個直接的擴展。

The model parameters are trained by optimizing for the performance of $f_{\theta'_i}$ with respect to $\theta$ across tasks sampled from $p(\mathcal{T})$. More concretely, the meta-objective is as follows:
$$ \min_\theta \sum_{\mathcal{T}_i \sim p(\mathcal{T})} \mathcal{L}_{\mathcal{T}_i}(f_{\theta'_i}) = \sum_{\mathcal{T}_i \sim p(\mathcal{T})} \mathcal{L}_{\mathcal{T}_i}(f_{\theta - \alpha \nabla_\theta \mathcal{L}_{\mathcal{T}_i}(f_\theta)}) $$

模型參數的訓練是通過優化 $f_{\theta'_i}$ 對於從 $p(\mathcal{T})$ 中採樣的任務的 $\theta$ 的性能來進行的。更具體地說，元目標如下：
$$ \min_\theta \sum_{\mathcal{T}_i \sim p(\mathcal{T})} \mathcal{L}_{\mathcal{T}_i}(f_{\theta'_i}) = \sum_{\mathcal{T}_i \sim p(\mathcal{T})} \mathcal{L}_{\mathcal{T}_i}(f_{\theta - \alpha \nabla_\theta \mathcal{L}_{\mathcal{T}_i}(f_\theta)}) $$

Note that the meta-optimization is performed over the model parameters $\theta$, whereas the objective is computed using the updated model parameters $\theta'$. In effect, our proposed method aims to optimize the model parameters such that one or a small number of gradient steps on a new task will produce maximally effective behavior on that task.

請注意，元優化是在模型參數 $\theta$ 上執行的，而目標是使用更新後的模型參數 $\theta'$ 計算的。實際上，我們提出的方法旨在優化模型參數，使得在新任務上進行一步或少量梯度步驟將在該任務上產生最有效的行為。

The meta-optimization across tasks is performed via stochastic gradient descent (SGD), such that the model parameters $\theta$ are updated as follows:
$$ \theta \leftarrow \theta - \beta \nabla_\theta \sum_{\mathcal{T}_i \sim p(\mathcal{T})} \mathcal{L}_{\mathcal{T}_i}(f_{\theta'_i}) \quad (1) $$
where $\beta$ is the meta step size. The full algorithm, in the general case, is outlined in Algorithm 1.

跨任務的元優化是通過隨機梯度下降 (SGD) 執行的，使得模型參數 $\theta$ 更新如下：
$$ \theta \leftarrow \theta - \beta \nabla_\theta \sum_{\mathcal{T}_i \sim p(\mathcal{T})} \mathcal{L}_{\mathcal{T}_i}(f_{\theta'_i}) \quad (1) $$
其中 $\beta$ 是元步長。一般情況下的完整演算法在演算法 1 中概述。

**Algorithm 1 Model-Agnostic Meta-Learning**
**演算法 1 模型無關元學習**

*   **Require:** $p(\mathcal{T})$: distribution over tasks
    **要求：** $p(\mathcal{T})$：任務分佈
*   **Require:** $\alpha, \beta$: step size hyperparameters
    **要求：** $\alpha, \beta$：步長超參數
*   1: randomly initialize $\theta$
    1: 隨機初始化 $\theta$
*   2: **while** not done **do**
    2: **當** 未完成 **時執行**
*   3: Sample batch of tasks $\mathcal{T}_i \sim p(\mathcal{T})$
    3: 採樣一批任務 $\mathcal{T}_i \sim p(\mathcal{T})$
*   4: **for all** $\mathcal{T}_i$ **do**
    4: **對所有** $\mathcal{T}_i$ **執行**
*   5: Evaluate $\nabla_\theta \mathcal{L}_{\mathcal{T}_i}(f_\theta)$ with respect to $K$ examples
    5: 針對 $K$ 個範例評估 $\nabla_\theta \mathcal{L}_{\mathcal{T}_i}(f_\theta)$
*   6: Compute adapted parameters with gradient descent: $\theta'_i = \theta - \alpha \nabla_\theta \mathcal{L}_{\mathcal{T}_i}(f_\theta)$
    6: 使用梯度下降計算適應後的參數：$\theta'_i = \theta - \alpha \nabla_\theta \mathcal{L}_{\mathcal{T}_i}(f_\theta)$
*   7: **end for**
    7: **結束迴圈**
*   8: Update $\theta \leftarrow \theta - \beta \nabla_\theta \sum_{\mathcal{T}_i \sim p(\mathcal{T})} \mathcal{L}_{\mathcal{T}_i}(f_{\theta'_i})$
    8: 更新 $\theta \leftarrow \theta - \beta \nabla_\theta \sum_{\mathcal{T}_i \sim p(\mathcal{T})} \mathcal{L}_{\mathcal{T}_i}(f_{\theta'_i})$
*   9: **end while**
    9: **結束迴圈**

The MAML meta-gradient update involves a gradient through a gradient. Computationally, this requires an additional backward pass through $f$ to compute Hessian-vector products, which is supported by standard deep learning libraries such as TensorFlow (Abadi et al., 2016). In our experiments, we also include a comparison to dropping this backward pass and using a first-order approximation, which we discuss in Section 5.2.

MAML 元梯度更新涉及一個梯度的梯度。在計算上，這需要額外的一次通過 $f$ 的反向傳播來計算 Hessian 向量積，這在 TensorFlow (Abadi et al., 2016) 等標準深度學習庫中得到支持。在我們的實驗中，我們還包含了一個與捨棄此反向傳播並使用一階近似的比較，我們將在第 5.2 節中討論這一點。

## 3. Species of MAML
## 3. MAML 的種類

In this section, we discuss specific instantiations of our meta-learning algorithm for supervised learning and reinforcement learning. The domains differ in the form of loss function and in how data is generated by the task and presented to the model, but the same basic adaptation mechanism can be applied in both cases.

在本節中，我們討論我們的元學習演算法在監督學習和強化學習中的具體實例。這些領域在損失函數的形式以及任務如何生成數據並呈現給模型方面有所不同，但相同的基本適應機制可以應用於這兩種情況。

### 3.1. Supervised Regression and Classification
### 3.1. 監督回歸和分類

Few-shot learning is well-studied in the domain of supervised tasks, where the goal is to learn a new function from only a few input/output pairs for that task, using prior data from similar tasks for meta-learning. For example, the goal might be to classify images of a Segway after seeing only one or a few examples of a Segway, with a model that has previously seen many other types of objects. Likewise, in few-shot regression, the goal is to predict the outputs of a continuous-valued function from only a few datapoints sampled from that function, after training on many functions with similar statistical properties.

少樣本學習在監督任務領域得到了充分研究，其目標是僅使用該任務的少數輸入/輸出對來學習新函數，並使用來自類似任務的先前數據進行元學習。例如，目標可能是在僅看過一個或幾個 Segway 的例子後對 Segway 的圖像進行分類，而模型之前已經看過許多其他類型的物體。同樣，在少樣本回歸中，目標是僅從該函數中採樣的少數數據點預測連續值函數的輸出，而在這之前已經在許多具有相似統計屬性的函數上進行過訓練。

To formalize the supervised regression and classification problems in the context of the meta-learning definitions in Section 2.1, we can define the horizon $H = 1$ and drop the timestep subscript on $\mathbf{x}_t$, since the model accepts a single input and produces a single output, rather than a sequence of inputs and outputs. The task $\mathcal{T}_i$ generates $K$ i.i.d. observations $\mathbf{x}$ from $q_i$, and the task loss is represented by the error between the model’s output for $\mathbf{x}$ and the corresponding target values $\mathbf{y}$ for that observation and task.

為了在第 2.1 節的元學習定義的背景下形式化監督回歸和分類問題，我們可以定義時間範圍 $H = 1$ 並去掉 $\mathbf{x}_t$ 上的時間步下標，因為模型接受單個輸入並產生單個輸出，而不是輸入和輸出的序列。任務 $\mathcal{T}_i$ 從 $q_i$ 生成 $K$ 個獨立同分佈（i.i.d.）的觀察值 $\mathbf{x}$，任務損失由模型對 $\mathbf{x}$ 的輸出與該觀察和任務的相應目標值 $\mathbf{y}$ 之間的誤差表示。

Two common loss functions used for supervised classification and regression are cross-entropy and mean-squared error (MSE), which we will describe below; though, other supervised loss functions may be used as well. For regression tasks using mean-squared error, the loss takes the form:
$$ \mathcal{L}_{\mathcal{T}_i}(f_\phi) = \sum_{\mathbf{x}^{(j)}, \mathbf{y}^{(j)} \sim \mathcal{T}_i} \| f_\phi(\mathbf{x}^{(j)}) - \mathbf{y}^{(j)} \|_2^2, \quad (2) $$
where $\mathbf{x}^{(j)}, \mathbf{y}^{(j)}$ are an input/output pair sampled from task $\mathcal{T}_i$. In $K$-shot regression tasks, $K$ input/output pairs are provided for learning for each task.

用於監督分類和回歸的兩個常見損失函數是交叉熵和均方誤差 (MSE)，我們將在下面進行描述；儘管也可以使用其他監督損失函數。對於使用均方誤差的回歸任務，損失形式如下：
$$ \mathcal{L}_{\mathcal{T}_i}(f_\phi) = \sum_{\mathbf{x}^{(j)}, \mathbf{y}^{(j)} \sim \mathcal{T}_i} \| f_\phi(\mathbf{x}^{(j)}) - \mathbf{y}^{(j)} \|_2^2, \quad (2) $$
其中 $\mathbf{x}^{(j)}, \mathbf{y}^{(j)}$ 是從任務 $\mathcal{T}_i$ 中採樣的輸入/輸出對。在 $K$-shot 回歸任務中，為每個任務的學習提供 $K$ 個輸入/輸出對。

Similarly, for discrete classification tasks with a cross-entropy loss, the loss takes the form:
$$ \mathcal{L}_{\mathcal{T}_i}(f_\phi) = \sum_{\mathbf{x}^{(j)}, \mathbf{y}^{(j)} \sim \mathcal{T}_i} \mathbf{y}^{(j)} \log f_\phi(\mathbf{x}^{(j)}) + (1 - \mathbf{y}^{(j)}) \log(1 - f_\phi(\mathbf{x}^{(j)})) \quad (3) $$

同樣，對於具有交叉熵損失的離散分類任務，損失形式如下：
$$ \mathcal{L}_{\mathcal{T}_i}(f_\phi) = \sum_{\mathbf{x}^{(j)}, \mathbf{y}^{(j)} \sim \mathcal{T}_i} \mathbf{y}^{(j)} \log f_\phi(\mathbf{x}^{(j)}) + (1 - \mathbf{y}^{(j)}) \log(1 - f_\phi(\mathbf{x}^{(j)})) \quad (3) $$

According to the conventional terminology, $K$-shot classification tasks use $K$ input/output pairs from each class, for a total of $NK$ data points for $N$-way classification. Given a distribution over tasks $p(\mathcal{T}_i)$, these loss functions can be directly inserted into the equations in Section 2.2 to perform meta-learning, as detailed in Algorithm 2.

根據常規術語，$K$-shot 分類任務使用每個類別的 $K$ 個輸入/輸出對，對於 $N$-way 分類，總共有 $NK$ 個數據點。給定任務分佈 $p(\mathcal{T}_i)$，這些損失函數可以直接插入第 2.2 節的方程中以執行元學習，如演算法 2 詳述。

**Algorithm 2 MAML for Few-Shot Supervised Learning**
**演算法 2 用於少樣本監督學習的 MAML**

*   **Require:** $p(\mathcal{T})$: distribution over tasks
    **要求：** $p(\mathcal{T})$：任務分佈
*   **Require:** $\alpha, \beta$: step size hyperparameters
    **要求：** $\alpha, \beta$：步長超參數
*   1: randomly initialize $\theta$
    1: 隨機初始化 $\theta$
*   2: **while** not done **do**
    2: **當** 未完成 **時執行**
*   3: Sample batch of tasks $\mathcal{T}_i \sim p(\mathcal{T})$
    3: 採樣一批任務 $\mathcal{T}_i \sim p(\mathcal{T})$
*   4: **for all** $\mathcal{T}_i$ **do**
    4: **對所有** $\mathcal{T}_i$ **執行**
*   5: Sample $K$ datapoints $\mathcal{D} = \{\mathbf{x}^{(j)}, \mathbf{y}^{(j)}\}$ from $\mathcal{T}_i$
    5: 從 $\mathcal{T}_i$ 中採樣 $K$ 個數據點 $\mathcal{D} = \{\mathbf{x}^{(j)}, \mathbf{y}^{(j)}\}$
*   6: Evaluate $\nabla_\theta \mathcal{L}_{\mathcal{T}_i}(f_\theta)$ using $\mathcal{D}$ and $\mathcal{L}_{\mathcal{T}_i}$ in Equation (2) or (3)
    6: 使用 $\mathcal{D}$ 和方程 (2) 或 (3) 中的 $\mathcal{L}_{\mathcal{T}_i}$ 評估 $\nabla_\theta \mathcal{L}_{\mathcal{T}_i}(f_\theta)$
*   7: Compute adapted parameters with gradient descent: $\theta'_i = \theta - \alpha \nabla_\theta \mathcal{L}_{\mathcal{T}_i}(f_\theta)$
    7: 使用梯度下降計算適應後的參數：$\theta'_i = \theta - \alpha \nabla_\theta \mathcal{L}_{\mathcal{T}_i}(f_\theta)$
*   8: Sample datapoints $\mathcal{D}'_i = \{\mathbf{x}^{(j)}, \mathbf{y}^{(j)}\}$ from $\mathcal{T}_i$ for the meta-update
    8: 從 $\mathcal{T}_i$ 中採樣數據點 $\mathcal{D}'_i = \{\mathbf{x}^{(j)}, \mathbf{y}^{(j)}\}$ 用於元更新
*   9: **end for**
    9: **結束迴圈**
*   10: Update $\theta \leftarrow \theta - \beta \nabla_\theta \sum_{\mathcal{T}_i \sim p(\mathcal{T})} \mathcal{L}_{\mathcal{T}_i}(f_{\theta'_i})$ using each $\mathcal{D}'_i$ and $\mathcal{L}_{\mathcal{T}_i}$ in Equation 2 or 3
    10: 使用每個 $\mathcal{D}'_i$ 和方程 2 或 3 中的 $\mathcal{L}_{\mathcal{T}_i}$ 更新 $\theta \leftarrow \theta - \beta \nabla_\theta \sum_{\mathcal{T}_i \sim p(\mathcal{T})} \mathcal{L}_{\mathcal{T}_i}(f_{\theta'_i})$
*   11: **end while**
    11: **結束迴圈**

### 3.2. Reinforcement Learning
### 3.2. 強化學習

In reinforcement learning (RL), the goal of few-shot meta-learning is to enable an agent to quickly acquire a policy for a new test task using only a small amount of experience in the test setting. A new task might involve achieving a new goal or succeeding on a previously trained goal in a new environment. For example, an agent might learn to quickly figure out how to navigate mazes so that, when faced with a new maze, it can determine how to reliably reach the exit with only a few samples. In this section, we will discuss how MAML can be applied to meta-learning for RL.

在強化學習 (RL) 中，少樣本元學習的目標是使代理能夠僅使用測試設置中的少量經驗，快速獲取新測試任務的策略。新任務可能涉及實現新目標或在新環境中成功完成先前訓練的目標。例如，代理可能學習快速弄清楚如何導航迷宮，以便當面對新迷宮時，它僅用幾個樣本就能確定如何可靠地到達出口。在本節中，我們將討論 MAML 如何應用於 RL 的元學習。

Each RL task $\mathcal{T}_i$ contains an initial state distribution $q_i(\mathbf{x}_1)$ and a transition distribution $q_i(\mathbf{x}_{t+1}|\mathbf{x}_t, \mathbf{a}_t)$, and the loss $\mathcal{L}_{\mathcal{T}_i}$ corresponds to the (negative) reward function $R$. The entire task is therefore a Markov decision process (MDP) with horizon $H$, where the learner is allowed to query a limited number of sample trajectories for few-shot learning. Any aspect of the MDP may change across tasks in $p(\mathcal{T})$. The model being learned, $f_\theta$, is a policy that maps from states $\mathbf{x}_t$ to a distribution over actions $\mathbf{a}_t$ at each timestep $t \in \{1, ..., H\}$. The loss for task $\mathcal{T}_i$ and model $f_\phi$ takes the form
$$ \mathcal{L}_{\mathcal{T}_i}(f_\phi) = -\mathbb{E}_{\mathbf{x}_t, \mathbf{a}_t \sim f_\phi, q_{\mathcal{T}_i}} \left[ \sum_{t=1}^H R_i(\mathbf{x}_t, \mathbf{a}_t) \right]. \quad (4) $$

每個 RL 任務 $\mathcal{T}_i$ 包含初始狀態分佈 $q_i(\mathbf{x}_1)$ 和轉移分佈 $q_i(\mathbf{x}_{t+1}|\mathbf{x}_t, \mathbf{a}_t)$，損失 $\mathcal{L}_{\mathcal{T}_i}$ 對應於（負）獎勵函數 $R$。因此，整個任務是一個長度為 $H$ 的馬可夫決策過程 (MDP)，其中允許學習者查詢有限數量的樣本軌跡以進行少樣本學習。MDP 的任何方面都可能在 $p(\mathcal{T})$ 中的任務之間發生變化。正在學習的模型 $f_\theta$ 是一個策略，它將每個時間步 $t \in \{1, ..., H\}$ 的狀態 $\mathbf{x}_t$ 映射到動作 $\mathbf{a}_t$ 的分佈。任務 $\mathcal{T}_i$ 和模型 $f_\phi$ 的損失形式為
$$ \mathcal{L}_{\mathcal{T}_i}(f_\phi) = -\mathbb{E}_{\mathbf{x}_t, \mathbf{a}_t \sim f_\phi, q_{\mathcal{T}_i}} \left[ \sum_{t=1}^H R_i(\mathbf{x}_t, \mathbf{a}_t) \right]. \quad (4) $$

In $K$-shot reinforcement learning, $K$ rollouts from $f_\theta$ and task $\mathcal{T}_i$, $(\mathbf{x}_1, \mathbf{a}_1, \ldots, \mathbf{x}_H)$, and the corresponding rewards $R(\mathbf{x}_t, \mathbf{a}_t)$, may be used for adaptation on a new task $\mathcal{T}_i$. Since the expected reward is generally not differentiable due to unknown dynamics, we use policy gradient methods to estimate the gradient both for the model gradient update(s) and the meta-optimization. Since policy gradients are an on-policy algorithm, each additional gradient step during the adaptation of $f_\theta$ requires new samples from the current policy $f_{\theta'_i}$. We detail the algorithm in Algorithm 3. This algorithm has the same structure as Algorithm 2, with the principal difference being that steps 5 and 8 require sampling trajectories from the environment corresponding to task $\mathcal{T}_i$. Practical implementations of this method may also use a variety of improvements recently proposed for policy gradient algorithms, including state or action-dependent baselines and trust regions (Schulman et al., 2015).

在 $K$-shot 強化學習中，來自 $f_\theta$ 和任務 $\mathcal{T}_i$ 的 $K$ 次推演（rollouts），$(\mathbf{x}_1, \mathbf{a}_1, \ldots, \mathbf{x}_H)$，以及相應的獎勵 $R(\mathbf{x}_t, \mathbf{a}_t)$，可用於新任務 $\mathcal{T}_i$ 的適應。由於未知動態，預期獎勵通常不可微分，因此我們使用策略梯度方法來估計模型梯度更新和元優化的梯度。由於策略梯度是一種在線策略（on-policy）演算法，因此在 $f_\theta$ 適應期間的每個額外梯度步驟都需要來自當前策略 $f_{\theta'_i}$ 的新樣本。我們在演算法 3 中詳細介紹了該演算法。該演算法具有與演算法 2 相同的結構，主要區別在於步驟 5 和 8 需要從對應於任務 $\mathcal{T}_i$ 的環境中採樣軌跡。該方法的實際實現還可以使用最近為策略梯度演算法提出的各種改進，包括狀態或動作依賴的基準線和信任區域 (Schulman et al., 2015)。

**Algorithm 3 MAML for Reinforcement Learning**
**演算法 3 用於強化學習的 MAML**

*   **Require:** $p(\mathcal{T})$: distribution over tasks
    **要求：** $p(\mathcal{T})$：任務分佈
*   **Require:** $\alpha, \beta$: step size hyperparameters
    **要求：** $\alpha, \beta$：步長超參數
*   1: randomly initialize $\theta$
    1: 隨機初始化 $\theta$
*   2: **while** not done **do**
    2: **當** 未完成 **時執行**
*   3: Sample batch of tasks $\mathcal{T}_i \sim p(\mathcal{T})$
    3: 採樣一批任務 $\mathcal{T}_i \sim p(\mathcal{T})$
*   4: **for all** $\mathcal{T}_i$ **do**
    4: **對所有** $\mathcal{T}_i$ **執行**
*   5: Sample $K$ trajectories $\mathcal{D} = \{(\mathbf{x}_1, \mathbf{a}_1, \ldots \mathbf{x}_H)\}$ using $f_\theta$ in $\mathcal{T}_i$
    5: 在 $\mathcal{T}_i$ 中使用 $f_\theta$ 採樣 $K$ 條軌跡 $\mathcal{D} = \{(\mathbf{x}_1, \mathbf{a}_1, \ldots \mathbf{x}_H)\}$
*   6: Evaluate $\nabla_\theta \mathcal{L}_{\mathcal{T}_i}(f_\theta)$ using $\mathcal{D}$ and $\mathcal{L}_{\mathcal{T}_i}$ in Equation 4
    6: 使用 $\mathcal{D}$ 和方程 4 中的 $\mathcal{L}_{\mathcal{T}_i}$ 評估 $\nabla_\theta \mathcal{L}_{\mathcal{T}_i}(f_\theta)$
*   7: Compute adapted parameters with gradient descent: $\theta'_i = \theta - \alpha \nabla_\theta \mathcal{L}_{\mathcal{T}_i}(f_\theta)$
    7: 使用梯度下降計算適應後的參數：$\theta'_i = \theta - \alpha \nabla_\theta \mathcal{L}_{\mathcal{T}_i}(f_\theta)$
*   8: Sample trajectories $\mathcal{D}'_i = \{(\mathbf{x}_1, \mathbf{a}_1, \ldots \mathbf{x}_H)\}$ using $f_{\theta'_i}$ in $\mathcal{T}_i$
    8: 在 $\mathcal{T}_i$ 中使用 $f_{\theta'_i}$ 採樣軌跡 $\mathcal{D}'_i = \{(\mathbf{x}_1, \mathbf{a}_1, \ldots \mathbf{x}_H)\}$
*   9: **end for**
    9: **結束迴圈**
*   10: Update $\theta \leftarrow \theta - \beta \nabla_\theta \sum_{\mathcal{T}_i \sim p(\mathcal{T})} \mathcal{L}_{\mathcal{T}_i}(f_{\theta'_i})$ using each $\mathcal{D}'_i$ and $\mathcal{L}_{\mathcal{T}_i}$ in Equation 4
    10: 使用每個 $\mathcal{D}'_i$ 和方程 4 中的 $\mathcal{L}_{\mathcal{T}_i}$ 更新 $\theta \leftarrow \theta - \beta \nabla_\theta \sum_{\mathcal{T}_i \sim p(\mathcal{T})} \mathcal{L}_{\mathcal{T}_i}(f_{\theta'_i})$
*   11: **end while**
    11: **結束迴圈**

## 4. Related Work
## 4. 相關工作

The method that we propose in this paper addresses the general problem of meta-learning (Thrun & Pratt, 1998; Schmidhuber, 1987; Naik & Mammone, 1992), which includes few-shot learning. A popular approach for meta-learning is to train a meta-learner that learns how to update the parameters of the learner’s model (Bengio et al., 1992; Schmidhuber, 1992; Bengio et al., 1990). This approach has been applied to learning to optimize deep networks (Hochreiter et al., 2001; Andrychowicz et al., 2016; Li & Malik, 2017), as well as for learning dynamically changing recurrent networks (Ha et al., 2017). One recent approach learns both the weight initialization and the optimizer, for few-shot image recognition (Ravi & Larochelle, 2017). Unlike these methods, the MAML learner’s weights are updated using the gradient, rather than a learned update; our method does not introduce additional parameters for meta-learning nor require a particular learner architecture.

我們在本文中提出的方法解決了元學習的一般問題（Thrun & Pratt, 1998; Schmidhuber, 1987; Naik & Mammone, 1992），其中包括少樣本學習。元學習的一種流行方法是訓練一個元學習器，學習如何更新學習者模型的參數（Bengio et al., 1992; Schmidhuber, 1992; Bengio et al., 1990）。這種方法已應用於學習優化深度網路（Hochreiter et al., 2001; Andrychowicz et al., 2016; Li & Malik, 2017），以及學習動態變化的循環網路（Ha et al., 2017）。最近的一種方法是針對少樣本圖像識別同時學習權重初始化和優化器（Ravi & Larochelle, 2017）。與這些方法不同，MAML 學習者的權重是使用梯度更新的，而不是學習到的更新；我們的方法不為元學習引入額外的參數，也不需要特定的學習者架構。

Few-shot learning methods have also been developed for specific tasks such as generative modeling (Edwards & Storkey, 2017; Rezende et al., 2016) and image recognition (Vinyals et al., 2016). One successful approach for few-shot classification is to learn to compare new examples in a learned metric space using e.g. Siamese networks (Koch, 2015) or recurrence with attention mechanisms (Vinyals et al., 2016; Shyam et al., 2017; Snell et al., 2017). These approaches have generated some of the most successful results, but are difficult to directly extend to other problems, such as reinforcement learning. Our method, in contrast, is agnostic to the form of the model and to the particular learning task.

針對特定任務（如生成建模（Edwards & Storkey, 2017; Rezende et al., 2016）和圖像識別（Vinyals et al., 2016））也開發了少樣本學習方法。少樣本分類的一種成功方法是學習在學習到的度量空間中比較新範例，例如使用孿生網路（Koch, 2015）或帶有注意力機制的循環網路（Vinyals et al., 2016; Shyam et al., 2017; Snell et al., 2017）。這些方法產生了一些最成功的結果，但很難直接擴展到其他問題，如強化學習。相比之下，我們的方法對模型的形式和特定的學習任務是無關的。

Another approach to meta-learning is to train memory-augmented models on many tasks, where the recurrent learner is trained to adapt to new tasks as it is rolled out. Such networks have been applied to few-shot image recognition (Santoro et al., 2016; Munkhdalai & Yu, 2017) and learning “fast” reinforcement learning agents (Duan et al., 2016b; Wang et al., 2016). Our experiments show that our method outperforms the recurrent approach on few-shot classification. Furthermore, unlike these methods, our approach simply provides a good weight initialization and uses the same gradient descent update for both the learner and meta-update. As a result, it is straightforward to fine-tune the learner for additional gradient steps.

元學習的另一種方法是在許多任務上訓練記憶增強模型，其中循環學習者被訓練以在展開時適應新任務。此類網路已應用於少樣本圖像識別（Santoro et al., 2016; Munkhdalai & Yu, 2017）和學習「快速」強化學習代理（Duan et al., 2016b; Wang et al., 2016）。我們的實驗表明，我們的方法在少樣本分類上優於循環方法。此外，與這些方法不同，我們的方法僅提供良好的權重初始化，並且對學習者和元更新使用相同的梯度下降更新。因此，對學習者進行額外梯度步驟的微調是直接了當的。

Our approach is also related to methods for initialization of deep networks. In computer vision, models pretrained on large-scale image classification have been shown to learn effective features for a range of problems (Donahue et al., 2014). In contrast, our method explicitly optimizes the model for fast adaptability, allowing it to adapt to new tasks with only a few examples. Our method can also be viewed as explicitly maximizing sensitivity of new task losses to the model parameters. A number of prior works have explored sensitivity in deep networks, often in the context of initialization (Saxe et al., 2014; Kirkpatrick et al., 2016). Most of these works have considered good random initializations, though a number of papers have addressed data-dependent initializers (Krähenbühl et al., 2016; Salimans & Kingma, 2016), including learned initializations (Husken & Goerick, 2000; Maclaurin et al., 2015). In contrast, our method explicitly trains the parameters for sensitivity on a given task distribution, allowing for extremely efficient adaptation for problems such as $K$-shot learning and rapid reinforcement learning in only one or a few gradient steps.

我們的方法也與深度網路的初始化方法有關。在電腦視覺中，在大規模圖像分類上預訓練的模型已被證明可以為一系列問題學習有效的特徵（Donahue et al., 2014）。相比之下，我們的方法明確地優化模型的快速適應性，使其能夠僅用幾個範例適應新任務。我們的方法也可以視為明確最大化新任務損失對模型參數的敏感度。許多先前的工作探索了深度網路中的敏感度，通常是在初始化的背景下（Saxe et al., 2014; Kirkpatrick et al., 2016）。大多數這些工作考慮了良好的隨機初始化，儘管有許多論文討論了數據依賴的初始化器（Krähenbühl et al., 2016; Salimans & Kingma, 2016），包括學習到的初始化（Husken & Goerick, 2000; Maclaurin et al., 2015）。相比之下，我們的方法明確地訓練參數在給定任務分佈上的敏感度，從而允許在僅一步或幾步梯度步驟中極其有效地適應 $K$-shot 學習和快速強化學習等問題。

## 5. Experimental Evaluation
## 5. 實驗評估

The goal of our experimental evaluation is to answer the following questions: (1) Can MAML enable fast learning of new tasks? (2) Can MAML be used for meta-learning in multiple different domains, including supervised regression, classification, and reinforcement learning? (3) Can a model learned with MAML continue to improve with additional gradient updates and/or examples?

我們實驗評估的目標是回答以下問題：（1）MAML 能否實現新任務的快速學習？（2）MAML 能否用於多個不同領域的元學習，包括監督回歸、分類和強化學習？（3）使用 MAML 學習的模型能否隨著額外的梯度更新和/或範例繼續改進？

All of the meta-learning problems that we consider require some amount of adaptation to new tasks at test-time. When possible, we compare our results to an oracle that receives the identity of the task (which is a problem-dependent representation) as an additional input, as an upper bound on the performance of the model. All of the experiments were performed using TensorFlow (Abadi et al., 2016), which allows for automatic differentiation through the gradient update(s) during meta-learning. The code is available online.

我們考慮的所有元學習問題都需要在測試時對新任務進行一定程度的適應。在可能的情況下，我們將結果與一個接收任務標識（這是一個問題相關的表示）作為額外輸入的 oracle 進行比較，作為模型性能的上限。所有實驗均使用 TensorFlow（Abadi et al., 2016）進行，該庫允許在元學習期間通過梯度更新進行自動微分。代碼可在線獲取。

### 5.1. Regression
### 5.1. 回歸

We start with a simple regression problem that illustrates the basic principles of MAML. Each task involves regressing from the input to the output of a sine wave, where the amplitude and phase of the sinusoid are varied between tasks. Thus, $p(\mathcal{T})$ is continuous, where the amplitude varies within $[0.1, 5.0]$ and the phase varies within $[0, \pi]$, and the input and output both have a dimensionality of 1. During training and testing, datapoints $\mathbf{x}$ are sampled uniformly from $[-5.0, 5.0]$. The loss is the mean-squared error between the prediction $f(\mathbf{x})$ and true value. The regressor is a neural network model with 2 hidden layers of size 40 with ReLU nonlinearities. When training with MAML, we use one gradient update with $K = 10$ examples with a fixed step size $\alpha = 0.01$, and use Adam as the meta-optimizer (Kingma & Ba, 2015). The baselines are likewise trained with Adam. To evaluate performance, we fine-tune a single meta-learned model on varying numbers of $K$ examples, and compare performance to two baselines: (a) pretraining on all of the tasks, which entails training a network to regress to random sinusoid functions and then, at test-time, fine-tuning with gradient descent on the $K$ provided points, using an automatically tuned step size, and (b) an oracle which receives the true amplitude and phase as input. In Appendix C, we show comparisons to additional multi-task and adaptation methods.

我們從一個簡單的回歸問題開始，說明 MAML 的基本原理。每個任務都涉及從正弦波的輸入回歸到輸出，其中正弦波的振幅和相位在任務之間變化。因此，$p(\mathcal{T})$ 是連續的，其中振幅在 $[0.1, 5.0]$ 範圍內變化，相位在 $[0, \pi]$ 範圍內變化，輸入和輸出維度均為 1。在訓練和測試期間，數據點 $\mathbf{x}$ 從 $[-5.0, 5.0]$ 均勻採樣。損失是預測值 $f(\mathbf{x})$ 與真實值之間的均方誤差。回歸器是一個具有 2 個隱藏層（大小為 40）和 ReLU 非線性的神經網路模型。在使用 MAML 訓練時，我們使用 $K = 10$ 個範例進行一次梯度更新，固定步長 $\alpha = 0.01$，並使用 Adam 作為元優化器（Kingma & Ba, 2015）。基準線同樣使用 Adam 訓練。為了評估性能，我們在不同數量的 $K$ 個範例上微調單個元學習模型，並與兩個基準線進行比較：（a）在所有任務上進行預訓練，這需要訓練一個網路回歸到隨機正弦函數，然後在測試時，使用自動調整的步長在提供的 $K$ 個點上進行梯度下降微調，以及（b）接收真實振幅和相位作為輸入的 oracle。在附錄 C 中，我們展示了與其他多任務和適應方法的比較。

We evaluate performance by fine-tuning the model learned by MAML and the pretrained model on $K = \{5, 10, 20\}$ datapoints. During fine-tuning, each gradient step is computed using the same $K$ datapoints. The qualitative results, shown in Figure 2 and further expanded on in Appendix B show that the learned model is able to quickly adapt with only 5 datapoints, shown as purple triangles, whereas the model that is pretrained using standard supervised learning on all tasks is unable to adequately adapt with so few datapoints without catastrophic overfitting. Crucially, when the $K$ datapoints are all in one half of the input range, the model trained with MAML can still infer the amplitude and phase in the other half of the range, demonstrating that the MAML trained model $f$ has learned to model the periodic nature of the sine wave. Furthermore, we observe both in the qualitative and quantitative results (Figure 3 and Appendix B) that the model learned with MAML continues to improve with additional gradient steps, despite being trained for maximal performance after one gradient step. This improvement suggests that MAML optimizes the parameters such that they lie in a region that is amenable to fast adaptation and is sensitive to loss functions from $p(\mathcal{T})$, as discussed in Section 2.2, rather than overfitting to parameters $\theta$ that only improve after one step.

我們通過在 $K = \{5, 10, 20\}$ 個數據點上微調 MAML 學習的模型和預訓練模型來評估性能。在微調期間，每個梯度步驟都使用相同的 $K$ 個數據點計算。圖 2 和附錄 B 中進一步擴展的定性結果顯示，學習到的模型能夠僅用 5 個數據點（顯示為紫色三角形）快速適應，而在所有任務上使用標準監督學習預訓練的模型無法在如此少的數據點下適當適應且不發生災難性的過擬合。至關重要的是，當 $K$ 個數據點都在輸入範圍的一半時，使用 MAML 訓練的模型仍然可以推斷出範圍另一半的振幅和相位，這表明 MAML 訓練的模型 $f$ 已經學會了對正弦波的週期性進行建模。此外，我們在定性和定量結果（圖 3 和附錄 B）中都觀察到，儘管 MAML 學習的模型是為了一次梯度步驟後的最佳性能而訓練的，但它隨著額外的梯度步驟繼續改進。這種改進表明，正如 2.2 節所討論的，MAML 優化了參數，使其位於易於快速適應且對來自 $p(\mathcal{T})$ 的損失函數敏感的區域，而不是過擬合於僅在一步後改進的參數 $\theta$。

### 5.2. Classification
### 5.2. 分類

To evaluate MAML in comparison to prior meta-learning and few-shot learning algorithms, we applied our method to few-shot image recognition on the Omniglot (Lake et al., 2011) and MiniImagenet datasets. The Omniglot dataset consists of 20 instances of 1623 characters from 50 different alphabets. Each instance was drawn by a different person. The MiniImagenet dataset was proposed by Ravi & Larochelle (2017), and involves 64 training classes, 12 validation classes, and 24 test classes. The Omniglot and MiniImagenet image recognition tasks are the most common recently used few-shot learning benchmarks (Vinyals et al., 2016; Santoro et al., 2016; Ravi & Larochelle, 2017).

為了評估 MAML 與先前的元學習和少樣本學習演算法的比較，我們將我們的方法應用於 Omniglot (Lake et al., 2011) 和 MiniImagenet 數據集上的少樣本圖像識別。Omniglot 數據集由來自 50 個不同字母表的 1623 個字符的 20 個實例組成。每個實例由不同的人繪製。MiniImagenet 數據集由 Ravi & Larochelle (2017) 提出，包含 64 個訓練類別、12 個驗證類別和 24 個測試類別。Omniglot 和 MiniImagenet 圖像識別任務是最近最常用的少樣本學習基準測試 (Vinyals et al., 2016; Santoro et al., 2016; Ravi & Larochelle, 2017)。

We follow the experimental protocol proposed by Vinyals et al. (2016), which involves fast learning of $N$-way classification with 1 or 5 shots. The problem of $N$-way classification is set up as follows: select $N$ unseen classes, provide the model with $K$ different instances of each of the $N$ classes, and evaluate the model’s ability to classify new instances within the $N$ classes. For Omniglot, we randomly select 1200 characters for training, irrespective of alphabet, and use the remaining for testing. The Omniglot dataset is augmented with rotations by multiples of 90 degrees, as proposed by Santoro et al. (2016).

我們遵循 Vinyals et al. (2016) 提出的實驗方案，涉及 1-shot 或 5-shot 的 $N$-way 分類的快速學習。$N$-way 分類問題設置如下：選擇 $N$ 個未見過的類別，為模型提供 $N$ 個類別中每個類別的 $K$ 個不同實例，並評估模型對 $N$ 個類別中的新實例進行分類的能力。對於 Omniglot，我們隨機選擇 1200 個字符進行訓練，不考慮字母表，並使用剩餘的字符進行測試。正如 Santoro et al. (2016) 所建議的，Omniglot 數據集通過旋轉 90 度的倍數進行了增強。

Our model follows the same architecture as the embedding function used by Vinyals et al. (2016), which has 4 modules with a $3 \times 3$ convolutions and 64 filters, followed by batch normalization (Ioffe & Szegedy, 2015), a ReLU nonlinearity, and $2 \times 2$ max-pooling. The Omniglot images are downsampled to $28 \times 28$, so the dimensionality of the last hidden layer is 64. As in the baseline classifier used by Vinyals et al. (2016), the last layer is fed into a softmax. For Omniglot, we used strided convolutions instead of max-pooling. For MiniImagenet, we used 32 filters per layer to reduce overfitting, as done by (Ravi & Larochelle, 2017). In order to also provide a fair comparison against memory-augmented neural networks (Santoro et al., 2016) and to test the flexibility of MAML, we also provide results for a non-convolutional network. For this, we use a network with 4 hidden layers with sizes 256, 128, 64, 64, each including batch normalization and ReLU nonlinearities, followed by a linear layer and softmax. For all models, the loss function is the cross-entropy error between the predicted and true class. Additional hyperparameter details are included in Appendix A.1.

我們的模型遵循與 Vinyals et al. (2016) 使用的嵌入函數相同的架構，該函數具有 4 個模組，每個模組包含 $3 \times 3$ 卷積和 64 個濾波器，隨後是批次歸一化（Ioffe & Szegedy，2015）、ReLU 非線性和 $2 \times 2$ 最大池化。Omniglot 圖像被下採樣至 $28 \times 28$，因此最後一個隱藏層的維度為 64。與 Vinyals et al. (2016) 使用的基線分類器一樣，最後一層被輸入到 softmax 中。對於 Omniglot，我們使用步長卷積代替最大池化。對於 MiniImagenet，我們每層使用 32 個濾波器以減少過擬合，正如 (Ravi & Larochelle, 2017) 所做的那樣。為了提供與記憶增強神經網路 (Santoro et al., 2016) 的公平比較並測試 MAML 的靈活性，我們還提供了非卷積網路的結果。為此，我們使用了一個具有 4 個隱藏層的網路，大小分別為 256、128、64、64，每個層都包括批次歸一化和 ReLU 非線性，隨後是線性層和 softmax。對於所有模型，損失函數是預測類別與真實類別之間的交叉熵誤差。其他超參數詳細資訊包含在附錄 A.1 中。

We present the results in Table 1. The convolutional model learned by MAML compares well to the state-of-the-art results on this task, narrowly outperforming the prior methods. Some of these existing methods, such as matching networks, Siamese networks, and memory models are designed with few-shot classification in mind, and are not readily applicable to domains such as reinforcement learning. Additionally, the model learned with MAML uses fewer overall parameters compared to matching networks and the meta-learner LSTM, since the algorithm does not introduce any additional parameters beyond the weights of the classifier itself. Compared to these prior methods, memory-augmented neural networks (Santoro et al., 2016) specifically, and recurrent meta-learning models in general, represent a more broadly applicable class of methods that, like MAML, can be used for other tasks such as reinforcement learning (Duan et al., 2016b; Wang et al., 2016). However, as shown in the comparison, MAML significantly outperforms memory-augmented networks and the meta-learner LSTM on 5-way Omniglot and MiniImagenet classification, both in the 1-shot and 5-shot case.

我們在表 1 中展示了結果。由 MAML 學習的卷積模型與該任務上的最先進結果相媲美，略微優於先前的方法。這些現有方法中的一些，如匹配網路、孿生網路和記憶模型，是專為少樣本分類設計的，並不適用於強化學習等領域。此外，與匹配網路和元學習器 LSTM 相比，使用 MAML 學習的模型使用的總參數更少，因為該演算法除了分類器本身的權重外不引入任何額外參數。與這些先前方法相比，記憶增強神經網路 (Santoro et al., 2016) 以及一般的循環元學習模型代表了一類更廣泛適用的方法，像 MAML 一樣，它們可用於強化學習等其他任務 (Duan et al., 2016b; Wang et al., 2016)。然而，如比較所示，MAML 在 5-way Omniglot 和 MiniImagenet 分類上的表現顯著優於記憶增強網路和元學習器 LSTM，無論是在 1-shot 還是 5-shot 的情況下。

| Omniglot (Lake et al., 2011) | 5-way Accuracy 1-shot | 5-way Accuracy 5-shot | 20-way Accuracy 1-shot | 20-way Accuracy 5-shot |
| :--- | :---: | :---: | :---: | :---: |
| MANN, no conv (Santoro et al., 2016) | 82.8% | 94.9% | – | – |
| **MAML, no conv (ours)** | **89.7 ± 1.1%** | **97.5 ± 0.6%** | – | – |
| Siamese nets (Koch, 2015) | 97.3% | 98.4% | 88.2% | 97.0% |
| matching nets (Vinyals et al., 2016) | 98.1% | 98.9% | 93.8% | 98.5% |
| neural statistician (Edwards & Storkey, 2017) | 98.1% | 99.5% | 93.2% | 98.1% |
| memory mod. (Kaiser et al., 2017) | 98.4% | 99.6% | 95.0% | 98.6% |
| **MAML (ours)** | **98.7 ± 0.4%** | **99.9 ± 0.1%** | **95.8 ± 0.3%** | **98.9 ± 0.2%** |

| MiniImagenet (Ravi & Larochelle, 2017) | 5-way Accuracy 1-shot | 5-way Accuracy 5-shot |
| :--- | :---: | :---: |
| fine-tuning baseline | 28.86 ± 0.54% | 49.79 ± 0.79% |
| nearest neighbor baseline | 41.08 ± 0.70% | 51.04 ± 0.65% |
| matching nets (Vinyals et al., 2016) | 43.56 ± 0.84% | 55.31 ± 0.73% |
| meta-learner LSTM (Ravi & Larochelle, 2017) | 43.44 ± 0.77% | 60.60 ± 0.71% |
| **MAML, first order approx. (ours)** | **48.07 ± 1.75%** | **63.15 ± 0.91%** |
| **MAML (ours)** | **48.70 ± 1.84%** | **63.11 ± 0.92%** |

A significant computational expense in MAML comes from the use of second derivatives when backpropagating the meta-gradient through the gradient operator in the meta-objective (see Equation (1)). On MiniImagenet, we show a comparison to a first-order approximation of MAML, where these second derivatives are omitted. Note that the resulting method still computes the meta-gradient at the post-update parameter values $\theta'_i$, which provides for effective meta-learning. Surprisingly however, the performance of this method is nearly the same as that obtained with full second derivatives, suggesting that most of the improvement in MAML comes from the gradients of the objective at the post-update parameter values, rather than the second order updates from differentiating through the gradient update. Past work has observed that ReLU neural networks are locally almost linear (Goodfellow et al., 2015), which suggests that second derivatives may be close to zero in most cases, partially explaining the good performance of the first-order approximation. This approximation removes the need for computing Hessian-vector products in an additional backward pass, which we found led to roughly 33% speed-up in network computation.

MAML 的一個顯著計算開銷來自於在元目標中通過梯度算子反向傳播元梯度時使用的二階導數（見方程 (1)）。在 MiniImagenet 上，我們展示了與 MAML 的一階近似的比較，其中省略了這些二階導數。請注意，結果方法仍然計算更新後參數值 $\theta'_i$ 處的元梯度，這提供了有效的元學習。然而令人驚訝的是，該方法的性能幾乎與使用全二階導數獲得的性能相同，這表明 MAML 的大部分改進來自於更新後參數值處的目標梯度，而不是來自通過梯度更新進行微分的二階更新。過去的工作觀察到 ReLU 神經網路在局部幾乎是線性的 (Goodfellow et al., 2015)，這表明在大多數情況下二階導數可能接近於零，部分解釋了一階近似的良好性能。這種近似消除了在額外的反向傳播中計算 Hessian 向量積的需要，我們發現這導致網路計算速度提高了大約 33%。

### 5.3. Reinforcement Learning
### 5.3. 強化學習

To evaluate MAML on reinforcement learning problems, we constructed several sets of tasks based off of the simulated continuous control environments in the rllab benchmark suite (Duan et al., 2016a). We discuss the individual domains below. In all of the domains, the model trained by MAML is a neural network policy with two hidden layers of size 100, with ReLU nonlinearities. The gradient updates are computed using vanilla policy gradient (REINFORCE) (Williams, 1992), and we use trust-region policy optimization (TRPO) as the meta-optimizer (Schulman et al., 2015). In order to avoid computing third derivatives, we use finite differences to compute the Hessian-vector products for TRPO. For both learning and meta-learning updates, we use the standard linear feature baseline proposed by Duan et al. (2016a), which is fitted separately at each iteration for each sampled task in the batch. We compare to three baseline models: (a) pretraining one policy on all of the tasks and then fine-tuning, (b) training a policy from randomly initialized weights, and (c) an oracle policy which receives the parameters of the task as input, which for the tasks below corresponds to a goal position, goal direction, or goal velocity for the agent. The baseline models of (a) and (b) are fine-tuned with gradient descent with a manually tuned step size.

為了評估 MAML 在強化學習問題上的表現，我們基於 rllab 基準套件中的模擬連續控制環境構建了幾組任務 (Duan et al., 2016a)。我們將在下面討論各個領域。在所有領域中，MAML 訓練的模型是一個具有兩個大小為 100 的隱藏層和 ReLU 非線性的神經網路策略。梯度更新使用普通策略梯度 (REINFORCE) (Williams, 1992) 計算，我們使用信任區域策略優化 (TRPO) 作為元優化器 (Schulman et al., 2015)。為了避免計算三階導數，我們使用有限差分來計算 TRPO 的 Hessian 向量積。對於學習和元學習更新，我們使用 Duan et al. (2016a) 提出的標準線性特徵基準線，該基準線在每次迭代中為批次中的每個採樣任務單獨擬合。我們與三個基線模型進行比較：（a）在所有任務上預訓練一個策略然後進行微調，（b）從隨機初始化的權重訓練策略，以及（c）接收任務參數作為輸入的 oracle 策略，對於下面的任務，這對應於代理的目標位置、目標方向或目標速度。（a）和（b）的基線模型使用具有手動調整步長的梯度下降進行微調。

**2D Navigation.** In our first meta-RL experiment, we study a set of tasks where a point agent must move to different goal positions in 2D, randomly chosen for each task within a unit square. The observation is the current 2D position, and actions correspond to velocity commands clipped to be in the range $[−0.1, 0.1]$. The reward is the negative squared distance to the goal, and episodes terminate when the agent is within 0.01 of the goal or at the horizon of $H = 100$. The policy was trained with MAML to maximize performance after 1 policy gradient update using 20 trajectories. Additional hyperparameter settings for this problem and the following RL problems are in Appendix A.2. In our evaluation, we compare adaptation to a new task with up to 4 gradient updates, each with 40 samples. The results in Figure 4 show the adaptation performance of models that are initialized with MAML, conventional pretraining on the same set of tasks, random initialization, and an oracle policy that receives the goal position as input. The results show that MAML can learn a model that adapts much more quickly in a single gradient update, and furthermore continues to improve with additional updates.

**2D 導航。** 在我們的第一個元 RL 實驗中，我們研究了一組任務，其中點代理必須移動到 2D 中的不同目標位置，每個任務在單位正方形內隨機選擇。觀察結果是當前的 2D 位置，動作對應於限制在 $[−0.1, 0.1]$ 範圍內的速度命令。獎勵是到目標的負平方距離，當代理在目標的 0.01 範圍內或在時間範圍 $H = 100$ 時，片段終止。策略使用 MAML 訓練，以在使用 20 條軌跡進行 1 次策略梯度更新後最大化性能。此問題及以下 RL 問題的其他超參數設置在附錄 A.2 中。在我們的評估中，我們比較了多達 4 次梯度更新（每次 40 個樣本）對新任務的適應情況。圖 4 中的結果顯示了使用 MAML 初始化、在同一組任務上進行常規預訓練、隨機初始化以及接收目標位置作為輸入的 oracle 策略的模型的適應性能。結果表明，MAML 可以學習一個在單次梯度更新中適應得更快的模型，並且隨著額外的更新繼續改進。

**Locomotion.** To study how well MAML can scale to more complex deep RL problems, we also study adaptation on high-dimensional locomotion tasks with the MuJoCo simulator (Todorov et al., 2012). The tasks require two simulated robots – a planar cheetah and a 3D quadruped (the “ant”) – to run in a particular direction or at a particular velocity. In the goal velocity experiments, the reward is the negative absolute value between the current velocity of the agent and a goal, which is chosen uniformly at random between 0.0 and 2.0 for the cheetah and between 0.0 and 3.0 for the ant. In the goal direction experiments, the reward is the magnitude of the velocity in either the forward or backward direction, chosen at random for each task in $p(\mathcal{T})$. The horizon is $H = 200$, with 20 rollouts per gradient step for all problems except the ant forward/backward task, which used 40 rollouts per step. The results in Figure 5 show that MAML learns a model that can quickly adapt its velocity and direction with even just a single gradient update, and continues to improve with more gradient steps. The results also show that, on these challenging tasks, the MAML initialization substantially outperforms random initialization and pretraining. In fact, pretraining is in some cases worse than random initialization, a fact observed in prior RL work (Parisotto et al., 2016).

**運動 (Locomotion)。** 為了研究 MAML 如何擴展到更複雜的深度 RL 問題，我們還研究了使用 MuJoCo 模擬器 (Todorov et al., 2012) 在高維運動任務上的適應性。這些任務需要兩個模擬機器人——平面獵豹 (cheetah) 和 3D 四足動物（「螞蟻 (ant)」）——以特定方向或特定速度奔跑。在目標速度實驗中，獎勵是代理當前速度與目標速度之間的負絕對值，目標速度是在獵豹的 0.0 到 2.0 和螞蟻的 0.0 到 3.0 之間均勻隨機選擇的。在目標方向實驗中，獎勵是向前或向後方向的速度幅度，在 $p(\mathcal{T})$ 中的每個任務隨機選擇。時間範圍為 $H = 200$，除螞蟻向前/向後任務每步使用 40 次推演外，所有問題每步使用 20 次推演。圖 5 中的結果表明，MAML 學習了一個模型，即使只有一次梯度更新，也能快速適應其速度和方向，並隨著更多梯度步驟繼續改進。結果還表明，在這些具有挑戰性的任務上，MAML 初始化大幅優於隨機初始化和預訓練。事實上，在某些情況下，預訓練比隨機初始化更差，這是先前 RL 工作中觀察到的一個事實 (Parisotto et al., 2016)。

## 6. Discussion and Future Work
## 6. 討論與未來工作

We introduced a meta-learning method based on learning easily adaptable model parameters through gradient descent. Our approach has a number of benefits. It is simple and does not introduce any learned parameters for meta-learning. It can be combined with any model representation that is amenable to gradient-based training, and any differentiable objective, including classification, regression, and reinforcement learning. Lastly, since our method merely produces a weight initialization, adaptation can be performed with any amount of data and any number of gradient steps, though we demonstrate state-of-the-art results on classification with only one or five examples per class. We also show that our method can adapt an RL agent using policy gradients and a very modest amount of experience.

我們介紹了一種基於通過梯度下降學習易於適應的模型參數的元學習方法。我們的方法有許多好處。它很簡單，並且不為元學習引入任何學習參數。它可以與任何適合基於梯度訓練的模型表示以及任何可微分目標相結合，包括分類、回歸和強化學習。最後，由於我們的方法僅產生權重初始化，因此可以使用任何數量的數據和任何數量的梯度步驟進行適應，儘管我們展示了僅使用每個類別一個或五個範例進行分類的最先進結果。我們還表明，我們的方法可以使用策略梯度和非常少量的經驗來適應 RL 代理。

Reusing knowledge from past tasks may be a crucial ingredient in making high-capacity scalable models, such as deep neural networks, amenable to fast training with small datasets. We believe that this work is one step toward a simple and general-purpose meta-learning technique that can be applied to any problem and any model. Further research in this area can make multitask initialization a standard ingredient in deep learning and reinforcement learning.

重用過去任務的知識可能是使高容量可擴展模型（如深度神經網路）適合小數據集快速訓練的關鍵因素。我們相信，這項工作是朝著可以應用於任何問題和任何模型的簡單且通用的元學習技術邁出的一步。該領域的進一步研究可以使多任務初始化成為深度學習和強化學習中的標準要素。

# Acknowledgements
# 致謝

The authors would like to thank Xi Chen and Trevor Darrell for helpful discussions, Yan Duan and Alex Lee for technical advice, Nikhil Mishra, Haoran Tang, and Greg Kahn for feedback on an early draft of the paper, and the anonymous reviewers for their comments. This work was supported in part by an ONR PECASE award and an NSF GRFP award.

作者感謝 Xi Chen 和 Trevor Darrell 的有益討論，Yan Duan 和 Alex Lee 的技術建議，Nikhil Mishra、Haoran Tang 和 Greg Kahn 對論文早期草稿的反饋，以及匿名審稿人的評論。這項工作部分得到了 ONR PECASE 獎項和 NSF GRFP 獎項的支持。

# References
# 參考文獻

Abadi, Martín, Agarwal, Ashish, Barham, Paul, Brevdo, Eugene, Chen, Zhifeng, Citro, Craig, Corrado, Greg S, Davis, Andy, Dean, Jeffrey, Devin, Matthieu, et al. Tensorflow: Large-scale machine learning on heterogeneous distributed systems. arXiv preprint arXiv:1603.04467, 2016.

Andrychowicz, Marcin, Denil, Misha, Gomez, Sergio, Hoffman, Matthew W, Pfau, David, Schaul, Tom, and de Freitas, Nando. Learning to learn by gradient descent by gradient descent. In Neural Information Processing Systems (NIPS), 2016.

Bengio, Samy, Bengio, Yoshua, Cloutier, Jocelyn, and Gecsei, Jan. On the optimization of a synaptic learning rule. In Optimality in Artificial and Biological Neural Networks, pp. 6–8, 1992.

Bengio, Yoshua, Bengio, Samy, and Cloutier, Jocelyn. Learning a synaptic learning rule. Université de Montréal, Département d’informatique et de recherche opérationnelle, 1990.

Donahue, Jeff, Jia, Yangqing, Vinyals, Oriol, Hoffman, Judy, Zhang, Ning, Tzeng, Eric, and Darrell, Trevor. Decaf: A deep convolutional activation feature for generic visual recognition. In International Conference on Machine Learning (ICML), 2014.

Duan, Yan, Chen, Xi, Houthooft, Rein, Schulman, John, and Abbeel, Pieter. Benchmarking deep reinforcement learning for continuous control. In International Conference on Machine Learning (ICML), 2016a.

Duan, Yan, Schulman, John, Chen, Xi, Bartlett, Peter L, Sutskever, Ilya, and Abbeel, Pieter. Rl2: Fast reinforcement learning via slow reinforcement learning. arXiv preprint arXiv:1611.02779, 2016b.

Edwards, Harrison and Storkey, Amos. Towards a neural statistician. International Conference on Learning Representations (ICLR), 2017.

Goodfellow, Ian J, Shlens, Jonathon, and Szegedy, Christian. Explaining and harnessing adversarial examples. International Conference on Learning Representations (ICLR), 2015.

Ha, David, Dai, Andrew, and Le, Quoc V. Hypernetworks. International Conference on Learning Representations (ICLR), 2017.

Hochreiter, Sepp, Younger, A Steven, and Conwell, Peter R. Learning to learn using gradient descent. In International Conference on Artificial Neural Networks. Springer, 2001.

Husken, Michael and Goerick, Christian. Fast learning for problem classes using knowledge based network initialization. In Neural Networks, 2000. IJCNN 2000, Proceedings of the IEEE-INNS-ENNS International Joint Conference on, volume 6, pp. 619–624. IEEE, 2000.

Ioffe, Sergey and Szegedy, Christian. Batch normalization: Accelerating deep network training by reducing internal covariate shift. International Conference on Machine Learning (ICML), 2015.

Kaiser, Lukasz, Nachum, Ofir, Roy, Aurko, and Bengio, Samy. Learning to remember rare events. International Conference on Learning Representations (ICLR), 2017.

Kingma, Diederik and Ba, Jimmy. Adam: A method for stochastic optimization. International Conference on Learning Representations (ICLR), 2015.

Kirkpatrick, James, Pascanu, Razvan, Rabinowitz, Neil, Veness, Joel, Desjardins, Guillaume, Rusu, Andrei A, Milan, Kieran, Quan, John, Ramalho, Tiago, Grabska-Barwinska, Agnieszka, et al. Overcoming catastrophic forgetting in neural networks. arXiv preprint arXiv:1612.00796, 2016.

Koch, Gregory. Siamese neural networks for one-shot image recognition. ICML Deep Learning Workshop, 2015.

Krähenbühl, Philipp, Doersch, Carl, Donahue, Jeff, and Darrell, Trevor. Data-dependent initializations of convolutional neural networks. International Conference on Learning Representations (ICLR), 2016.

Lake, Brenden M, Salakhutdinov, Ruslan, Gross, Jason, and Tenenbaum, Joshua B. One shot learning of simple visual concepts. In Conference of the Cognitive Science Society (CogSci), 2011.

Li, Ke and Malik, Jitendra. Learning to optimize. International Conference on Learning Representations (ICLR), 2017.

Maclaurin, Dougal, Duvenaud, David, and Adams, Ryan. Gradient-based hyperparameter optimization through reversible learning. In International Conference on Machine Learning (ICML), 2015.

Munkhdalai, Tsendsuren and Yu, Hong. Meta networks. International Conferecence on Machine Learning (ICML), 2017.

Naik, Devang K and Mammone, RJ. Meta-neural networks that learn by learning. In International Joint Conference on Neural Netowrks (IJCNN), 1992.

Parisotto, Emilio, Ba, Jimmy Lei, and Salakhutdinov, Ruslan. Actor-mimic: Deep multitask and transfer reinforcement learning. International Conference on Learning Representations (ICLR), 2016.

Ravi, Sachin and Larochelle, Hugo. Optimization as a model for few-shot learning. In International Conference on Learning Representations (ICLR), 2017.

Rei, Marek. Online representation learning in recurrent neural language models. arXiv preprint arXiv:1508.03854, 2015.

Rezende, Danilo Jimenez, Mohamed, Shakir, Danihelka, Ivo, Gregor, Karol, and Wierstra, Daan. One-shot generalization in deep generative models. International Conference on Machine Learning (ICML), 2016.

Salimans, Tim and Kingma, Diederik P. Weight normalization: A simple reparameterization to accelerate training of deep neural networks. In Neural Information Processing Systems (NIPS), 2016.

Santoro, Adam, Bartunov, Sergey, Botvinick, Matthew, Wierstra, Daan, and Lillicrap, Timothy. Meta-learning with memory-augmented neural networks. In International Conference on Machine Learning (ICML), 2016.

Saxe, Andrew, McClelland, James, and Ganguli, Surya. Exact solutions to the nonlinear dynamics of learning in deep linear neural networks. International Conference on Learning Representations (ICLR), 2014.

Schmidhuber, Jurgen. Evolutionary principles in self-referential learning. On learning how to learn: The meta-meta-... hook.) Diploma thesis, Institut f. Informatik, Tech. Univ. Munich, 1987.

Schmidhuber, Jürgen. Learning to control fast-weight memories: An alternative to dynamic recurrent networks. Neural Computation, 1992.

Schulman, John, Levine, Sergey, Abbeel, Pieter, Jordan, Michael I, and Moritz, Philipp. Trust region policy optimization. In International Conference on Machine Learning (ICML), 2015.

Shyam, Pranav, Gupta, Shubham, and Dukkipati, Ambedkar. Attentive recurrent comparators. International Conferecence on Machine Learning (ICML), 2017.

Snell, Jake, Swersky, Kevin, and Zemel, Richard S. Prototypical networks for few-shot learning. arXiv preprint arXiv:1703.05175, 2017.

Thrun, Sebastian and Pratt, Lorien. Learning to learn. Springer Science & Business Media, 1998.

Todorov, Emanuel, Erez, Tom, and Tassa, Yuval. Mujoco: A physics engine for model-based control. In International Conference on Intelligent Robots and Systems (IROS), 2012.

Vinyals, Oriol, Blundell, Charles, Lillicrap, Tim, Wierstra, Daan, et al. Matching networks for one shot learning. In Neural Information Processing Systems (NIPS), 2016.

Wang, Jane X, Kurth-Nelson, Zeb, Tirumala, Dhruva, Soyer, Hubert, Leibo, Joel Z, Munos, Remi, Blundell, Charles, Kumaran, Dharshan, and Botvinick, Matt. Learning to reinforcement learn. arXiv preprint arXiv:1611.05763, 2016.

Williams, Ronald J. Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine learning, 8(3-4):229–256, 1992.

## A. Additional Experiment Details
## A. 額外的實驗細節

In this section, we provide additional details of the experimental set-up and hyperparameters.
在本節中，我們提供實驗設置和超參數的其他詳細資訊。

### A.1. Classification
### A.1. 分類

For N-way, K-shot classification, each gradient is computed using a batch size of $NK$ examples. For Omniglot, the 5-way convolutional and non-convolutional MAML models were each trained with 1 gradient step with step size $\alpha = 0.4$ and a meta batch-size of 32 tasks. The network was evaluated using 3 gradient steps with the same step size $\alpha = 0.4$. The 20-way convolutional MAML model was trained and evaluated with 5 gradient steps with step size $\alpha = 0.1$. During training, the meta batch-size was set to 16 tasks. For MiniImagenet, both models were trained using 5 gradient steps of size $\alpha = 0.01$, and evaluated using 10 gradient steps at test time. Following Ravi & Larochelle (2017), 15 examples per class were used for evaluating the post-update meta-gradient. We used a meta batch-size of 4 and 2 tasks for 1-shot and 5-shot training respectively. All models were trained for 60000 iterations on a single NVIDIA Pascal Titan X GPU.

對於 N-way，K-shot 分類，每個梯度都是使用批量大小為 $NK$ 的範例計算的。對於 Omniglot，5-way 卷積和非卷積 MAML 模型均使用步長 $\alpha = 0.4$ 的 1 個梯度步驟和 32 個任務的元批量大小進行訓練。網路使用相同步長 $\alpha = 0.4$ 的 3 個梯度步驟進行評估。20-way 卷積 MAML 模型使用步長 $\alpha = 0.1$ 的 5 個梯度步驟進行訓練和評估。在訓練期間，元批量大小設置為 16 個任務。對於 MiniImagenet，兩個模型均使用 5 個大小為 $\alpha = 0.01$ 的梯度步驟進行訓練，並在測試時使用 10 個梯度步驟進行評估。遵循 Ravi & Larochelle (2017)，每個類別使用 15 個範例來評估更新後的元梯度。我們分別對 1-shot 和 5-shot 訓練使用了 4 和 2 個任務的元批量大小。所有模型都在單個 NVIDIA Pascal Titan X GPU 上訓練了 60000 次迭代。

### A.2. Reinforcement Learning
### A.2. 強化學習

In all reinforcement learning experiments, the MAML policy was trained using a single gradient step with $\alpha = 0.1$. During evaluation, we found that halving the learning rate after the first gradient step produced superior performance. Thus, the step size during adaptation was set to $\alpha = 0.1$ for the first step, and $\alpha = 0.05$ for all future steps. The step sizes for the baseline methods were manually tuned for each domain. In the 2D navigation, we used a meta batch size of 20; in the locomotion problems, we used a meta batch size of 40 tasks. The MAML models were trained for up to 500 meta-iterations, and the model with the best average return during training was used for evaluation. For the ant goal velocity task, we added a positive reward bonus at each timestep to prevent the ant from ending the episode.

在所有強化學習實驗中，MAML 策略使用 $\alpha = 0.1$ 的單個梯度步驟進行訓練。在評估期間，我們發現在第一個梯度步驟後將學習率減半會產生更好的性能。因此，適應期間的步長設置為第一步 $\alpha = 0.1$，所有後續步驟 $\alpha = 0.05$。基準方法的步長是針對每個領域手動調整的。在 2D 導航中，我們使用了 20 的元批量大小；在運動問題中，我們使用了 40 個任務的元批量大小。MAML 模型訓練了多達 500 次元迭代，並使用訓練期間平均回報最好的模型進行評估。對於螞蟻目標速度任務，我們在每個時間步增加了一個正獎勵獎金，以防止螞蟻結束片段。

## B. Additional Sinusoid Results
## B. 額外的正弦波結果

In Figure 6, we show the full quantitative results of the MAML model trained on 10-shot learning and evaluated on 5-shot, 10-shot, and 20-shot. In Figure 7, we show the qualitative performance of MAML and the pretrained baseline on randomly sampled sinusoids.

在圖 6 中，我們展示了在 10-shot 學習上訓練並在 5-shot、10-shot 和 20-shot 上評估的 MAML 模型的完整定量結果。在圖 7 中，我們展示了 MAML 和預訓練基準線在隨機採樣的正弦波上的定性性能。

## C. Additional Comparisons
## C. 額外的比較

In this section, we include more thorough evaluations of our approach, including additional multi-task baselines and a comparison representative of the approach of Rei (2015).

在本節中，我們包括對我們方法的更徹底評估，包括額外的多任務基準線和代表 Rei (2015) 方法的比較。

### C.1. Multi-task baselines
### C.1. 多任務基準線

The pretraining baseline in the main text trained a single network on all tasks, which we referred to as “pretraining on all tasks”. To evaluate the model, as with MAML, we fine-tuned this model on each test task using $K$ examples. In the domains that we study, different tasks involve different output values for the same input. As a result, by pre-training on all tasks, the model would learn to output the average output for a particular input value. In some instances, this model may learn very little about the actual domain, and instead learn about the range of the output space.

正文中的預訓練基準線在所有任務上訓練單個網路，我們稱之為「在所有任務上預訓練」。為了評估模型，與 MAML 一樣，我們使用 $K$ 個範例在每個測試任務上微調此模型。在我們研究的領域中，不同的任務涉及相同輸入的不同輸出值。因此，通過在所有任務上進行預訓練，模型將學習輸出特定輸入值的平均輸出。在某些情況下，該模型可能對實際領域了解甚少，而是學習有關輸出空間範圍的知識。

We experimented with a multi-task method to provide a point of comparison, where instead of averaging in the output space, we averaged in the parameter space. To achieve averaging in parameter space, we sequentially trained 500 separate models on 500 tasks drawn from $p(\mathcal{T})$. Each model was initialized randomly and trained on a large amount of data from its assigned task. We then took the average parameter vector across models and fine-tuned on 5 datapoints with a tuned step size. All of our experiments for this method were on the sinusoid task because of computational requirements. The error of the individual regressors was low: less than 0.02 on their respective sine waves.

我們嘗試了一種多任務方法來提供比較點，其中我們不是在輸出空間中取平均，而是在參數空間中取平均。為了實現參數空間的平均，我們在從 $p(\mathcal{T})$ 中抽取的 500 個任務上順序訓練了 500 個單獨的模型。每個模型隨機初始化，並使用來自其分配任務的大量數據進行訓練。然後，我們取模型的平均参数向量，並使用調整後的步長在 5 個數據點上進行微調。由於計算要求，我們對此方法的所有實驗都在正弦波任務上進行。各個回歸器的誤差很低：在各自的正弦波上小於 0.02。

We tried three variants of this set-up. During training of the individual regressors, we tried using one of the following: no regularization, standard $\ell_2$ weight decay, and $\ell_2$ weight regularization to the mean parameter vector thus far of the trained regressors. The latter two variants encourage the individual models to find parsimonious solutions. When using regularization, we set the magnitude of the regularization to be as high as possible without significantly deterring performance. In our results, we refer to this approach as “multi-task”. As seen in the results in Table 2, we find averaging in the parameter space (multi-task) performed worse than averaging in the output space (pre-training on all tasks). This suggests that it is difficult to find parsimonious solutions to multiple tasks when training on tasks separately, and that MAML is learning a solution that is more sophisticated than the mean optimal parameter vector.

我們嘗試了這種設置的三種變體。在訓練各個回歸器期間，我們嘗試使用以下其中一種：無正則化、標準 $\ell_2$ 權重衰減，以及對迄今為止受訓回歸器的平均参数向量進行 $\ell_2$ 權重正則化。後兩個變體鼓勵各個模型找到簡約的解決方案。使用正則化時，我們將正則化的幅度設置為盡可能高，且不顯著影響性能。在我們的結果中，我們將此方法稱為「多任務」。如表 2 的結果所示，我們發現參數空間中的平均（多任務）表現比輸出空間中的平均（在所有任務上預訓練）更差。這表明，在分別訓練任務時很難找到多個任務的簡約解決方案，並且 MAML 正在學習一種比平均最佳参数向量更複雜的解決方案。

| num. grad steps | 1 | 5 | 10 |
| :--- | :---: | :---: | :---: |
| multi-task, no reg | 4.19 | 3.85 | 3.69 |
| multi-task, l2 reg | 7.18 | 5.69 | 5.60 |
| multi-task, reg to mean $\theta$ | 2.91 | 2.72 | 2.71 |
| pretrain on all tasks | 2.41 | 2.23 | 2.19 |
| **MAML (ours)** | **0.67** | **0.38** | **0.35** |

### C.2. Context vector adaptation
### C.2. 上下文向量適應

Rei (2015) developed a method which learns a context vector that can be adapted online, with an application to recurrent language models. The parameters in this context vector are learned and adapted in the same way as the parameters in the MAML model. To provide a comparison to using such a context vector for meta-learning problems, we concatenated a set of free parameters $\mathbf{z}$ to the input $\mathbf{x}$, and only allowed the gradient steps to modify $\mathbf{z}$, rather than modifying the model parameters $\theta$, as in MAML. For image inputs, $\mathbf{z}$ was concatenated channel-wise with the input image. We ran this method on Omniglot and two RL domains following the same experimental protocol. We report the results in Tables 3, 4, and 5. Learning an adaptable context vector performed well on the toy pointmass problem, but sub-par on more difficult problems, likely due to a less flexible meta-optimization.

Rei (2015) 開發了一種學習上下文向量的方法，該向量可以在線適應，應用於循環語言模型。此上下文向量中的參數以與 MAML 模型中的參數相同的方式進行學習和適應。為了提供與使用這種上下文向量進行元學習問題的比較，我們將一組自由參數 $\mathbf{z}$ 連接到輸入 $\mathbf{x}$，並且只允許梯度步驟修改 $\mathbf{z}$，而不是像 MAML 中那樣修改模型參數 $\theta$。對於圖像輸入，$\mathbf{z}$ 與輸入圖像在通道方向上連接。我們遵循相同的實驗方案，在 Omniglot 和兩個 RL 領域上運行了此方法。我們在表 3、4 和 5 中報告了結果。學習可適應的上下文向量在簡單的點質量（pointmass）問題上表現良好，但在更困難的問題上表現不佳，這可能是由於元優化不夠靈活。

| Table 3. 5-way Omniglot Classification | 1-shot | 5-shot |
| :--- | :---: | :---: |
| context vector | 94.9 ± 0.9% | 97.7 ± 0.3% |
| **MAML** | **98.7 ± 0.4%** | **99.9 ± 0.1%** |

| Table 4. 2D Pointmass, average return | 0 | 1 | 2 | 3 |
| :--- | :---: | :---: | :---: | :---: |
| context vector | −42.42 | −13.90 | −5.17 | −3.18 |
| **MAML (ours)** | **−40.41** | **−11.68** | **−3.33** | −3.23 |

| Table 5. Half-cheetah forward/backward, average return | 0 | 1 | 2 | 3 |
| :--- | :---: | :---: | :---: | :---: |
| context vector | −40.49 | −44.08 | −38.27 | −42.50 |
| **MAML (ours)** | −50.69 | **293.19** | **313.48** | **315.65** |
