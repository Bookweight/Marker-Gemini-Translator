---
title: PyTorch An Imperative Style, High-Performance Deep Learning Library
field: Deep_Learning
status: Imported
created_date: 2026-01-18
pdf_link: "[[PyTorch An Imperative Style, High-Performance Deep Learning Library.pdf]]"
tags:
  - paper
  - Deep_learning
---

# PyTorch: An Imperative Style, High-Performance Deep Learning Library
# PyTorch：一種指令式風格、高效能的深度學習函式庫

**Adam Paszke**
University of Warsaw
adam.paszke@gmail.com
**Adam Paszke**
華沙大學
adam.paszke@gmail.com

**Sam Gross**
Facebook AI Research
sgross@fb.com
**Sam Gross**
Facebook 人工智慧研究部門 (FAIR)
sgross@fb.com

**Francisco Massa**
Facebook AI Research
fmassa@fb.com
**Francisco Massa**
Facebook 人工智慧研究部門 (FAIR)
fmassa@fb.com

**Adam Lerer**
Facebook AI Research
alerer@fb.com
**Adam Lerer**
Facebook 人工智慧研究部門 (FAIR)
alerer@fb.com

**James Bradbury**
Google
jekbradbury@gmail.com
**James Bradbury**
Google
jekbradbury@gmail.com

**Gregory Chanan**
Facebook AI Research
gchanan@fb.com
**Gregory Chanan**
Facebook 人工智慧研究部門 (FAIR)
gchanan@fb.com

**Trevor Killeen**
Self Employed
killeent@cs.washington.edu
**Trevor Killeen**
自由業者
killeent@cs.washington.edu

**Zeming Lin**
Facebook AI Research
zlin@fb.com
**Zeming Lin**
Facebook 人工智慧研究部門 (FAIR)
zlin@fb.com

**Natalia Gimelshein**
NVIDIA
ngimelshein@nvidia.com
**Natalia Gimelshein**
NVIDIA
ngimelshein@nvidia.com

**Luca Antiga**
Orobix
luca.antiga@orobix.com
**Luca Antiga**
Orobix
luca.antiga@orobix.com

**Alban Desmaison**
Oxford University
alban@robots.ox.ac.uk
**Alban Desmaison**
牛津大學
alban@robots.ox.ac.uk

**Andreas Köpf**
Xamla
andreas.koepf@xamla.com
**Andreas Köpf**
Xamla
andreas.koepf@xamla.com

**Edward Yang**
Facebook AI Research
ezyang@fb.com
**Edward Yang**
Facebook 人工智慧研究部門 (FAIR)
ezyang@fb.com

**Zach DeVito**
Facebook AI Research
zdevito@cs.stanford.edu
**Zach DeVito**
Facebook 人工智慧研究部門 (FAIR)
zdevito@cs.stanford.edu

**Martin Raison**
Nabla
martinraison@gmail.com
**Martin Raison**
Nabla
martinraison@gmail.com

**Alykhan Tejani**
Twitter
atejani@twitter.com
**Alykhan Tejani**
Twitter
atejani@twitter.com

**Sasank Chilamkurthy**
Qure.ai
sasankchilamkurthy@gmail.com
**Sasank Chilamkurthy**
Qure.ai
sasankchilamkurthy@gmail.com

**Benoit Steiner**
Facebook AI Research
benoitsteiner@fb.com
**Benoit Steiner**
Facebook 人工智慧研究部門 (FAIR)
benoitsteiner@fb.com

**Lu Fang**
Facebook
lufang@fb.com
**Lu Fang**
Facebook
lufang@fb.com

**Junjie Bai**
Facebook
jbai@fb.com
**Junjie Bai**
Facebook
jbai@fb.com

**Soumith Chintala**
Facebook AI Research
soumith@gmail.com
**Soumith Chintala**
Facebook 人工智慧研究部門 (FAIR)
soumith@gmail.com

### Abstract
### 摘要

Deep learning frameworks have often focused on either usability or speed, but not both.
深度學習框架通常專注於易用性或速度，但很少兩者兼顧。

PyTorch is a machine learning library that shows that these two goals are in fact compatible: it provides an imperative and Pythonic programming style that supports code as a model, makes debugging easy and is consistent with other popular scientific computing libraries, while remaining efficient and supporting hardware accelerators such as GPUs.
PyTorch 是一個機器學習函式庫，證明了這兩個目標實際上是相容的：它提供了一種指令式 (Imperative) 且符合 Python 風格 (Pythonic) 的程式設計風格，支援「程式碼即模型 (code as a model)」，使除錯變得容易，並與其他流行的科學計算函式庫保持一致，同時保持高效能並支援 GPU 等硬體加速器。

In this paper, we detail the principles that drove the implementation of PyTorch and how they are reflected in its architecture.
在本文中，我們詳細介紹了推動 PyTorch 實作的原則，以及這些原則如何反映在其架構中。

We emphasize that every aspect of PyTorch is a regular Python program under the full control of its user.
我們強調 PyTorch 的各個方面都是常規的 Python 程式，完全由使用者控制。

We also explain how the careful and pragmatic implementation of the key components of its runtime enables them to work together to achieve compelling performance.
我們也解釋了其執行時 (runtime) 關鍵組件的謹慎且務實的實作，如何使它們協同工作以實現令人矚目的效能。

We demonstrate the efficiency of individual subsystems, as well as the overall speed of PyTorch on several common benchmarks.
我們展示了個別子系統的效率，以及 PyTorch 在幾個常見基準測試中的整體速度。

33rd Conference on Neural Information Processing Systems (NeurIPS 2019), Vancouver, Canada.
第 33 屆神經資訊處理系統會議 (NeurIPS 2019)，加拿大溫哥華。

arXiv:1912.01703v1 [cs.LG] 3 Dec 2019
arXiv:1912.01703v1 [cs.LG] 2019年12月3日

---

## 1 Introduction
## 1 簡介

With the increased interest in deep learning in recent years, there has been an explosion of machine learning tools.
隨著近年來對深度學習興趣的增加，機器學習工具呈現爆炸式成長。

Many popular frameworks such as Caffe [1], CNTK [2], TensorFlow [3], and Theano [4], construct a static dataflow graph that represents the computation and which can then be applied repeatedly to batches of data.
許多流行的框架，如 Caffe [1]、CNTK [2]、TensorFlow [3] 和 Theano [4]，都建構一個靜態資料流圖 (static dataflow graph) 來表示計算，然後可以將其重複應用於批次資料。

This approach provides visibility into the whole computation ahead of time, and can theoretically be leveraged to improve performance and scalability.
這種方法提供了對整個計算的預先可見性，理論上可以利用它來提高效能和可擴展性。

However, it comes at the cost of ease of use, ease of debugging, and flexibility of the types of computation that can be represented.
然而，這是以犧牲易用性、除錯便利性以及可表示的計算類型的靈活性為代價的。

Prior work has recognized the value of dynamic eager execution for deep learning, and some recent frameworks implement this define-by-run approach, but do so either at the cost of performance (Chainer [5]) or using a less expressive, faster language (Torch [6], DyNet [7]), which limits their applicability.
先前的工作已經認識到動態即時執行 (dynamic eager execution) 對深度學習的價值，一些最近的框架實作了這種「執行即定義 (define-by-run)」的方法，但這樣做要麼犧牲了效能 (Chainer [5])，要麼使用表達能力較差但速度較快的語言 (Torch [6], DyNet [7])，這限制了它們的適用性。

However, with careful implementation and design choices, dynamic eager execution can be achieved largely without sacrificing performance.
然而，透過謹慎的實作和設計選擇，可以在很大程度上不犧牲效能的情況下實現動態即時執行。

This paper introduces PyTorch, a Python library that performs immediate execution of dynamic tensor computations with automatic differentiation and GPU acceleration, and does so while maintaining performance comparable to the fastest current libraries for deep learning.
本文介紹了 PyTorch，這是一個 Python 函式庫，它可以執行具有自動微分和 GPU 加速功能的動態張量計算的立即執行，同時保持與目前最快的深度學習函式庫相當的效能。

This combination has turned out to be very popular in the research community with, for instance, 296 ICLR 2019 submissions mentioning PyTorch.
這種組合在研究社群中非常受歡迎，例如，有 296 篇 ICLR 2019 的投稿提到了 PyTorch。

## 2 Background
## 2 背景

Four major trends in scientific computing have become increasingly important for deep learning.
科學計算中的四個主要趨勢對深度學習變得越來越重要。

First, starting in the 1960s, the development of domain specific languages such as APL [8], MATLAB [9], R [10] and Julia [11], turned multidimensional arrays (often referred to as tensors) into first-class objects supported by a comprehensive set of mathematical primitives (or operators) to manipulate them.
首先，從 1960 年代開始，領域特定語言 (DSL) 的發展，如 APL [8]、MATLAB [9]、R [10] 和 Julia [11]，將多維陣列（通常稱為張量）轉變為由一組全面的數學基元（或運算子）支援的一等物件 (first-class objects) 以進行操作。

Separately, libraries such as NumPy[12], Torch[6], Eigen[13] and Lush[14] made **array-based programming** productive in general purpose languages such as Python, Lisp, C++ and Lua.
另外，NumPy [12]、Torch [6]、Eigen [13] 和 Lush [14] 等函式庫使得在 Python、Lisp、C++ 和 Lua 等通用語言中進行**基於陣列的程式設計**變得富有成效。

Second, the development of **automatic differentiation** [15] made it possible to fully automate the daunting labor of computing derivatives.
其次，**自動微分** [15] 的發展使得完全自動化計算導數這項艱鉅的工作成為可能。

This made it significantly easier to experiment with different machine learning approaches while still allowing for efficient gradient based optimization.
這使得嘗試不同的機器學習方法變得更加容易，同時仍然允許進行高效的基於梯度的最佳化。

The autograd [16] package popularized the use of this technique for NumPy arrays, and similar approaches are used in frameworks such as Chainer [5], DyNet [7], Lush [14], Torch [6], Jax [17] and Flux.jl [18].
autograd [16] 套件普及了此技術在 NumPy 陣列中的使用，類似的方法也被用於 Chainer [5]、DyNet [7]、Lush [14]、Torch [6]、Jax [17] 和 Flux.jl [18] 等框架中。

Third, with the advent of the free software movement, the scientific community moved away from closed proprietary software such as Matlab[9], and towards the **open-source Python ecosystem** with packages like NumPy [12], SciPy [19], and Pandas [20].
第三，隨著自由軟體運動的到來，科學界逐漸遠離 Matlab [9] 等封閉的專有軟體，轉向擁有 NumPy [12]、SciPy [19] 和 Pandas [20] 等套件的**開源 Python 生態系統**。

This fulfilled most of the numerical analysis needs of researchers while allowing them to take advantage of a vast repository of libraries to handle dataset preprocessing, statistical analysis, plotting, and more.
這滿足了研究人員的大部分數值分析需求，同時允許他們利用龐大的函式庫資源庫來處理資料集預處理、統計分析、繪圖等。

Moreover, the openness, interoperability, and flexibility of free software fostered the development of vibrant communities that could quickly address new or changing needs by extending the existing functionality of a library or if needed by developing and releasing brand new ones.
此外，自由軟體的開放性、互通性和靈活性促進了充滿活力的社群發展，這些社群可以透過擴展現有函式庫的功能，或者在需要時開發和發布全新的函式庫，來快速解決新的或不斷變化的需求。

While there is a rich offering of open-source software for neural networks in languages other than Python, starting with Lush [14] in Lisp, Torch [6] in C++, Objective-C and Lua, EBLearn [21] in C++, Caffe [1] in C++, the network effects of a large ecosystem such as Python made it an essential skill to jumpstart one’s research.
雖然在 Python 以外的語言中有豐富的開源神經網路軟體，從 Lisp 中的 Lush [14]、C++、Objective-C 和 Lua 中的 Torch [6]、C++ 中的 EBLearn [21]、C++ 中的 Caffe [1] 開始，但像 Python 這樣大型生態系統的網路效應使其成為啟動研究的必備技能。

Hence, since 2014, most deep learning frameworks converged on a Python interface as an essential feature.
因此，自 2014 年以來，大多數深度學習框架都將 Python 介面作為一項基本功能。

Finally, the availability and commoditization of general-purpose massively parallel hardware such as GPUs provided the computing power required by deep learning methods.
最後，GPU 等通用大規模平行硬體的可用性和商品化提供了深度學習方法所需的計算能力。

Specialized libraries such as cuDNN [22], along with a body of academic work (such as [23] and [24]), produced a set of high-performance reusable deep learning kernels that enabled frameworks such as Caffe [1], Torch7 [25], or TensorFlow [3] to take advantage of these **hardware accelerators**.
cuDNN [22] 等專用函式庫，以及大量的學術著作（如 [23] 和 [24]），產生了一組高效能的可重複使用深度學習核心 (kernels)，使 Caffe [1]、Torch7 [25] 或 TensorFlow [3] 等框架能夠利用這些**硬體加速器**。

PyTorch builds on these trends by providing an array-based programming model accelerated by GPUs and differentiable via automatic differentiation integrated in the Python ecosystem.
PyTorch 建立在這些趨勢之上，提供了一個由 GPU 加速並可透過整合在 Python 生態系統中的自動微分進行微分的基於陣列的程式設計模型。

## 3 Design principles
## 3 設計原則

PyTorch’s success stems from weaving previous ideas into a design that balances speed and ease of use.
PyTorch 的成功源於將先前的想法編織成一個平衡速度和易用性的設計。

There are four main principles behind our choices:
我們的選擇背後有四個主要原則：

**Be Pythonic** Data scientists are familiar with the Python language, its programming model, and its tools.
**符合 Python 風格 (Be Pythonic)** 資料科學家熟悉 Python 語言、其程式設計模型及其工具。

PyTorch should be a first-class member of that ecosystem.
PyTorch 應該是該生態系統的一等公民。

It follows the commonly established design goals of keeping interfaces simple and consistent, ideally with one idiomatic way of doing things.
它遵循普遍建立的設計目標，即保持介面簡單一致，理想情況下只有一種慣用的做事方式。

It also integrates naturally with standard plotting, debugging, and data processing tools.
它還能與標準的繪圖、除錯和資料處理工具自然整合。

**Put researchers first** PyTorch strives to make writing models, data loaders, and optimizers as easy and productive as possible.
**研究人員優先 (Put researchers first)** PyTorch 致力於使編寫模型、資料載入器和最佳化器盡可能容易和高效。

The complexity inherent to machine learning should be handled internally by the PyTorch library and hidden behind intuitive APIs free of side-effects and unexpected performance cliffs.
機器學習固有的複雜性應由 PyTorch 函式庫在內部處理，並隱藏在直觀的 API 背後，沒有副作用和意外的效能驟降。

**Provide pragmatic performance** To be useful, PyTorch needs to deliver compelling performance, although not at the expense of simplicity and ease of use.
**提供務實的效能 (Provide pragmatic performance)** 為了實用，PyTorch 需要提供令人矚目的效能，但不能以犧牲簡單性和易用性為代價。

Trading 10% of speed for a significantly simpler to use model is acceptable; 100% is not.
為了讓模型明顯更易於使用而犧牲 10% 的速度是可以接受的；但犧牲 100% 則不行。

Therefore, its *implementation* accepts added complexity in order to deliver that performance.
因此，其*實作*接受增加的複雜性以提供該效能。

Additionally, providing tools that allow researchers to manually control the execution of their code will empower them to find their own performance improvements independent of those that the library provides automatically.
此外，提供允許研究人員手動控制程式碼執行的工具，將使他們能夠獨立於函式庫自動提供的改進之外，找到自己的效能改進方法。

**Worse is better** [26] Given a fixed amount of engineering resources, and all else being equal, the time saved by keeping the internal implementation of PyTorch simple can be used to implement additional features, adapt to new situations, and keep up with the fast pace of progress in the field of AI.
**更糟就是更好 (Worse is better)** [26] 在工程資源固定的情況下，在其他條件相同時，透過保持 PyTorch 內部實作簡單所節省的時間，可用於實作額外功能、適應新情況，並跟上 AI 領域快速發展的步伐。

Therefore it is better to have a simple but slightly incomplete solution than a comprehensive but complex and hard to maintain design.
因此，擁有一個簡單但稍微不完整的解決方案，比擁有一個全面但複雜且難以維護的設計要好。

## 4 Usability centric design
## 4 以可用性為中心的設計

### 4.1 Deep learning models are just Python programs
### 4.1 深度學習模型只是 Python 程式

In a surprisingly short amount of time, machine learning grew from recognizing individual digits [27] into autonomously playing StarCraft [28].
在令人驚訝的短時間內，機器學習從識別單個數字 [27] 發展到自主遊玩星海爭霸 (StarCraft) [28]。

Consequently, the neural networks themselves evolved rapidly from simple sequences of feed forward layers into incredibly varied numerical programs often composed of many loops and recursive functions.
因此，神經網路本身也迅速從簡單的前饋層序列演變為極其多樣化的數值程式，通常由許多迴圈和遞迴函式組成。

To support this growing complexity, PyTorch foregoes the potential benefits of a graph-metaprogramming based approach to preserve the imperative programming model of Python.
為了支援這種日益增長的複雜性，PyTorch 放棄了基於圖的元程式設計 (graph-metaprogramming) 方法的潛在好處，以保留 Python 的指令式程式設計模型。

This design was pioneered for model authoring by Chainer[5] and Dynet[7].
這種用於模型創作的設計由 Chainer [5] 和 Dynet [7] 首創。

PyTorch extends this to all aspects of deep learning workflows.
PyTorch 將其擴展到深度學習工作流程的所有方面。

Defining layers, composing models, loading data, running optimizers, and parallelizing the training process are all expressed using the familiar concepts developed for general purpose programming.
定義層、組合模型、載入資料、執行最佳化器和平行化訓練過程，都是使用為通用程式設計開發的熟悉概念來表達的。

This solution ensures that any new potential neural network architecture can be easily implemented with PyTorch.
這個解決方案確保了任何新的潛在神經網路架構都可以用 PyTorch 輕鬆實作。

For instance, layers (which in modern machine learning should really be understood as stateful functions with implicit parameters) are typically expressed as Python classes whose constructors create and initialize their parameters, and whose forward methods process an input activation.
例如，層（在現代機器學習中，實際上應理解為具有隱式參數的有狀態函式）通常表示為 Python 類別，其建構函式建立並初始化其參數，其前向 (forward) 方法處理輸入活化。

Similarly, models are usually represented as classes that compose individual layers, but let us state again that nothing forces the user to structure their code in that way.
同樣地，模型通常表示為組合各個層的類別，但我們要再次聲明，沒有什麼強迫使用者以這種方式建立他們的程式碼。

Listing 1 demonstrates how an entire model can be created by composing functionality provided by PyTorch such as 2d convolution, matrix multiplication, dropout, and softmax to classify gray-scale images.
Listing 1 展示了如何透過組合 PyTorch 提供的功能（如 2d 卷積、矩陣乘法、dropout 和 softmax）來建立完整的模型，以對灰階影像進行分類。

Note that linear layers are of course part of the library, but we show an example implementation to highlight how simple it is.
請注意，線性層當然是函式庫的一部分，但我們展示了一個範例實作以強調它是多麼簡單。

```python
class LinearLayer(Module):
    def __init__(self, in_sz, out_sz):
        super().__init__()
        t1 = torch.randn(in_sz, out_sz)
        self.w = nn.Parameter(t1)
        t2 = torch.randn(out_sz)
        self.b = nn.Parameter(t2)

    def forward(self, activations):
        t = torch.mm(activations, self.w)
        return t + self.b

class FullBasicModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 128, 3)
        self.fc = LinearLayer(128, 10)

    def forward(self, x):
        t1 = self.conv(x)
        t2 = nn.functional.relu(t1)
        t3 = self.fc(t1)
        return nn.functional.softmax(t3)
```
**Listing 1:** A custom layer used as a building block for a simple but complete neural network.
**Listing 1:** 一個自訂層，用作簡單但完整的神經網路的建構模組。

This “everything is a just a program” philosophy is not limited to just the models, and applies to optimizers and data loaders as well.
這種「一切都只是程式」的哲學不僅限於模型，也適用於最佳化器和資料載入器。

This facilitates the experimentation of new training techniques.
這促進了新訓練技術的實驗。

For example, to implement the very popular generative adversarial networks, one needs to specify two separate models (the generator and the discriminator), and two loss functions that depend on both models at the same time.
例如，要實作非常流行的生成對抗網路 (GAN)，需要指定兩個獨立的模型（生成器和鑑別器），以及同時依賴於這兩個模型的兩個損失函數。

Rigid APIs would struggle with this setup, but the simple design employed in PyTorch easily adapts to this setting as shown in Listing 2.
僵化的 API 會難以應對這種設置，但 PyTorch 採用的簡單設計可以輕鬆適應這種設置，如 Listing 2 所示。

```python
discriminator = create_discriminator()
generator = create_generator()
optimD = optim.Adam(discriminator.parameters())
optimG = optim.Adam(generator.parameters())

def step(real_sample):
    # (1) Update Discriminator
    errD_real = loss(discriminator(real_sample), real_label)
    errD_real.backward()
    fake = generator(get_noise())
    errD_fake = loss(discriminator(fake.detach()), fake_label)
    errD_fake.backward()
    optimD.step()
    # (2) Update Generator
    errG = loss(discriminator(fake), real_label)
    errG.backward()
    optimG.step()
```
**Listing 2:** Simplified training of a generative adversarial networks.
**Listing 2:** 生成對抗網路的簡化訓練。

Since PyTorch programs execute eagerly, all the features of Python are available throughout the whole design process.
由於 PyTorch 程式是急切執行 (eagerly execute) 的，因此 Python 的所有功能在整個設計過程中都可用。

Print statements, standard debuggers, and common visualization tools like matplotlib all work as expected.
Print 語句、標準除錯器和像 matplotlib 這樣的常見視覺化工具都能如預期般運作。

Users do not have to wait for lengthy compilation before they can start running their programs, and more importantly intermediate computations can be observed to understand how a model works and whether its results are correct.
使用者不必等待漫長的編譯即可開始執行他們的程式，更重要的是，可以觀察中間計算結果以了解模型如何運作以及其結果是否正確。

### 4.2 Interoperability and extensibility
### 4.2 互通性和可擴展性

Easy and efficient interoperability is one of the top priorities for PyTorch because it opens the possibility to leverage the rich ecosystem of Python libraries as part of user programs.
簡單高效的互通性是 PyTorch 的首要任務之一，因為它開啟了利用豐富的 Python 函式庫生態系統作為使用者程式一部分的可能性。

Hence, PyTorch allows for bidirectional exchange of data with external libraries.
因此，PyTorch 允許與外部函式庫進行雙向資料交換。

For example, it provides a mechanism to convert between NumPy arrays and PyTorch tensors using the `torch.from_numpy()` function and `.numpy()` tensor method.
例如，它提供了一種使用 `torch.from_numpy()` 函式和 `.numpy()` 張量方法在 NumPy 陣列和 PyTorch 張量之間進行轉換的機制。

Similar functionality is also available to exchange data stored using the DLPack [29] format.
類似的功能也可用於交換使用 DLPack [29] 格式儲存的資料。

Note that this exchange happens in both cases without any data copying – objects on both sides only describe how to interpret a memory region which is shared among them.
請注意，這兩種情況下的交換都不會發生任何資料複製——雙方的物件僅描述如何解釋它們之間共享的記憶體區域。

Hence, those operations are actually extremely cheap, and take constant time no matter how large the converted arrays are.
因此，這些操作實際上非常便宜，並且無論轉換的陣列有多大，都只需要常數時間。

Moreover, many of the critical systems are designed specifically to be extensible.
此外，許多關鍵系統是專門為可擴展性而設計的。

For instance, the automatic differentiation system allows users to add support for custom differentiable functions.
例如，自動微分系統允許使用者新增對自訂可微分函式及其導數的支援。

To do that users can define a new subclass of `torch.autograd.Function` that implements `forward()` and `backward()` methods, which specify the function and its derivative (or more formally the vector-Jacobian product).
為此，使用者可以定義一個 `torch.autograd.Function` 的新子類別，該子類別實作 `forward()` 和 `backward()` 方法，這些方法指定函式及其導數（或更正式地說是向量-雅可比積）。

Similarly new datasets can be added by subclassing `torch.utils.data.Dataset` and implementing two methods: `__getitem__` (the indexing operator) and `__len__` (the length operator), making datasets behave like (possibly lazy) lists.
同樣地，可以透過繼承 `torch.utils.data.Dataset` 並實作兩個方法：`__getitem__`（索引運算子）和 `__len__`（長度運算子）來新增新的資料集，使資料集表現得像（可能是延遲載入的）列表。

How these work is completely up to the implementer, and many users leverage other Python packages for data loading.
這些如何運作完全取決於實作者，許多使用者利用其他 Python 套件進行資料載入。

The `DataLoader` class consumes objects conforming to this interface and provides an iterator over the data which takes care of shuffling, batching, parallelization, and management of pinned CUDA memory to improve throughput.
`DataLoader` 類別使用符合此介面的物件，並提供資料的迭代器，該迭代器負責洗牌、批次處理、平行化和 pinned CUDA 記憶體的管理以提高吞吐量。

Most importantly, users are free to replace any component of PyTorch that does not meet the needs or performance requirements of their project.
最重要的是，使用者可以自由替換 PyTorch 中任何不符合其專案需求或效能要求的組件。

They are all designed to be completely interchangeable, and PyTorch takes great care not to impose any particular solution.
它們都被設計成完全可互換的，PyTorch 非常小心地不強加任何特定的解決方案。

### 4.3 Automatic differentiation
### 4.3 自動微分

Since gradient based optimization is vital to deep learning, PyTorch must be able to automatically compute gradients of models specified by our users, and those can be arbitrary Python programs.
由於基於梯度的最佳化對深度學習至關重要，PyTorch 必須能夠自動計算使用者指定的模型的梯度，而這些模型可以是任意的 Python 程式。

However, Python is a dynamic programming language that allows changing most behaviors at runtime, making ahead of time source-to-source differentiation cumbersome.
然而，Python 是一種動態程式語言，允許在執行時更改大多數行為，這使得提前的原始碼到原始碼微分變得繁瑣。

Instead, PyTorch uses the operator overloading approach, which builds up a representation of the computed function every time it is executed.
相反，PyTorch 使用運算子重載 (operator overloading) 方法，每次執行時都會建立計算函式的表示。

In its current implementation [30], PyTorch performs reverse-mode automatic differentiation, which computes the gradient of a scalar output with respect to a multivariate input.
在其目前的實作 [30] 中，PyTorch 執行反向模式自動微分，計算純量輸出相對於多變量輸入的梯度。

Differentiating functions with more outputs than inputs is more efficiently executed using forward-mode automatic differentiation, but this use case is less common for machine learning applications.
使用前向模式自動微分可以更有效地執行輸出多於輸入的微分函式，但這種用例在機器學習應用中較不常見。

PyTorch can be easily extended to perform forward-mode differentiation using array-level dual numbers [31, 32].
PyTorch 可以很容易地擴展，使用陣列級對偶數 (dual numbers) [31, 32] 執行前向模式微分。

Another interesting and uncommon feature of our system is that it can differentiate through code employing mutation on tensors, which is one of the basic building blocks of imperative programs.
我們系統的另一個有趣且不常見的功能是，它可以對使用張量突變 (mutation) 的程式碼進行微分，這是指令式程式的基本構建模組之一。

To ensure safety, we have implemented a versioning system for tensors, which lets us track their modifications and ensure that we always use the data we expect.
為了確保安全，我們為張量實作了一個版本控制系統，這使我們能夠追蹤它們的修改並確保我們始終使用我們期望的資料。

One interesting tradeoff is that while we could utilize techniques like copy-on-write to support arbitrary programs, we chose to not go down this path, as performance-wise it is usually beneficial for the users to rewrite their code to ensure that no copies have to be performed.
一個有趣的權衡是，雖然我們可以利用寫入時複製 (copy-on-write) 等技術來支援任意程式，但我們選擇不走這條路，因為在效能方面，使用者重寫程式碼以確保不必執行複製通常是有益的。

Hence, while most mutations are benign and can be handled automatically, the really complicated cases result in a user error, which lets them know that they likely want to restructure the program.
因此，雖然大多數突變是良性的並且可以自動處理，但真正複雜的情況會導致使用者錯誤，這讓他們知道他們可能需要重構程式。

This allows us to avoid introducing subtle and hard-to-find performance cliffs.
這使我們能夠避免引入微妙且難以發現的效能驟降。

## 5 Performance focused implementation
## 5 專注於效能的實作

Running deep learning algorithms efficiently from a Python interpreter is notoriously challenging: for instance, the global interpreter lock [33] effectively ensures that only one of any number of concurrent threads is running at any given time.
從 Python 直譯器高效地執行深度學習演算法是出了名的具有挑戰性：例如，全域直譯器鎖 (GIL) [33] 有效地確保了在任何給定時間，任意數量的並發執行緒中只有一個在執行。

Deep learning frameworks based on the construction of a static data-flow graph sidestep this problem by deferring the evaluation of the computation to a custom interpreter.
基於靜態資料流圖構建的深度學習框架透過將計算評估推遲到自訂直譯器來迴避這個問題。

PyTorch solved the problem differently, by carefully optimizing every aspect of its execution while simultaneously empowering its users to easily leverage additional optimization strategies.
PyTorch 以不同的方式解決了這個問題，透過仔細最佳化其執行的各個方面，同時讓使用者能夠輕鬆利用額外的最佳化策略。

### 5.1 An efficient C++ core
### 5.1 高效的 C++ 核心

Despite being closely integrated in the Python ecosystem, most of PyTorch is written in C++ to achieve high performance.
儘管與 Python 生態系統緊密整合，但 PyTorch 的大部分內容都是用 C++ 編寫的，以實現高效能。

This core `libtorch` library implements the tensor data structure, the GPU and CPU operators, and basic parallel primitives.
這個核心 `libtorch` 函式庫實作了張量資料結構、GPU 和 CPU 運算子以及基本的平行基元。

It also provides the automatic differentiation system, including the gradient formulas for most built-in functions.
它還提供了自動微分系統，包括大多數內建函式的梯度公式。

This ensures that the computation of the derivatives of functions composed of core PyTorch operators is executed entirely in a multithreaded evaluator which does not require holding the Python global interpreter lock [33].
這確保了由核心 PyTorch 運算子組成的函式導數的計算完全在不需要持有 Python 全域直譯器鎖 [33] 的多執行緒評估器中執行。

Python bindings are generated using YAML meta-data files.
Python 綁定 (bindings) 是使用 YAML 元資料檔案產生的。

An interesting side-effect of this approach is that it allowed our community to quickly create bindings to multiple other languages resulting in projects like NimTorch [34], hasktorch [35] and others.
這種方法的一個有趣的副作用是，它允許我們的社群快速建立對多種其他語言的綁定，從而產生了像 NimTorch [34]、hasktorch [35] 等專案。

This design also allowed us to create first-class C++ bindings and modeling libraries that can be used in places where Python is inconvenient, such as the game engine for Starcraft [36] or on mobile platforms.
這種設計還允許我們建立一流的 C++ 綁定和建模函式庫，這些函式庫可用於 Python 不方便的地方，例如星海爭霸的遊戲引擎 [36] 或行動平台。

It is even possible to take the Python code describing a PyTorch model and run it without Python using the TorchScript engine [37].
甚至可以獲取描述 PyTorch 模型的 Python 程式碼，並使用 TorchScript 引擎 [37] 在沒有 Python 的情況下執行它。

### 5.2 Separate control and data flow
### 5.2 分離的控制和資料流

PyTorch maintains a strict separation between its control (i.e. program branches, loops) and data flow (i.e. tensors and the operations performed on them).
PyTorch 在其控制（即程式分支、迴圈）和資料流（即張量和對其執行的操作）之間保持嚴格的分離。

The resolution of the control flow is handled by Python and optimized C++ code executed on the host CPU, and result in a linear sequence of operator invocations on the device.
控制流的解析由在主機 CPU 上執行的 Python 和最佳化的 C++ 程式碼處理，並導致裝置上運算子呼叫的線性序列。

Operators can be run either on CPU or on GPU.
運算子可以在 CPU 或 GPU 上執行。

PyTorch is designed to execute operators asynchronously on GPU by leveraging the CUDA stream mechanism [38] to queue CUDA kernel invocations to the GPUs hardware FIFO.
PyTorch 旨在透過利用 CUDA 串流機制 [38] 將 CUDA 核心呼叫排隊到 GPU 硬體 FIFO，在 GPU 上非同步執行運算子。

This allows the system to overlap the execution of Python code on CPU with tensor operators on GPU.
這允許系統將 CPU 上的 Python 程式碼執行與 GPU 上的張量運算子重疊。

Because the tensor operations usually take a significant amount of time, this lets us saturate the GPU and reach peak performance even in an interpreted language with fairly high overhead like Python.
因為張量運算通常需要花費大量時間，這使我們即使在像 Python 這樣具有相當高開銷的直譯語言中，也能使 GPU 飽和並達到峰值效能。

Note that this mechanism is nearly invisible to the user.
請注意，此機制對使用者來說幾乎是不可見的。

Unless they implement their own multi-stream primitives all of the CPU-GPU synchronization is handled by the library.
除非他們實作自己的多串流基元，否則所有的 CPU-GPU 同步都由函式庫處理。

PyTorch could leverage a similar mechanism to also execute operators asynchronously on the CPU.
PyTorch 可以利用類似的機制在 CPU 上非同步執行運算子。

However the costs of cross-thread communication and synchronization would negate the performance benefit of such an optimization.
然而，跨執行緒通訊和同步的成本將抵消這種最佳化的效能優勢。

### 5.3 Custom caching tensor allocator
### 5.3 自訂快取張量配置器

Almost every operator must dynamically allocate an output tensor to hold the result of its execution.
幾乎每個運算子都必須動態配置一個輸出張量來保存其執行結果。

It is therefore critical to optimize the speed of the dynamic memory allocators.
因此，最佳化動態記憶體配置器的速度至關重要。

PyTorch can rely on optimized libraries [39–41] to handle this task on CPU.
PyTorch 可以依賴最佳化的函式庫 [39–41] 在 CPU 上處理此任務。

However, on GPU the `cudaFree` routine may block its caller until all previously queued work on all GPUs completes.
然而，在 GPU 上，`cudaFree` 常式可能會阻塞其呼叫者，直到所有 GPU 上所有先前排隊的工作完成。

To avoid this bottleneck, PyTorch implements a custom allocator which incrementally builds up a cache of CUDA memory and reassigns it to later allocations without further use of CUDA APIs.
為了避免這個瓶頸，PyTorch 實作了一個自訂配置器，它逐步建立 CUDA 記憶體的快取，並將其重新分配給以後的配置，而無需進一步使用 CUDA API。

The incremental allocation is also crucial for better interoperability, because taking up all GPU memory ahead of time would prevent the user from utilizing other GPU-enabled Python packages.
增量配置對於更好的互通性也至關重要，因為提前佔用所有 GPU 記憶體將阻止使用者使用其他支援 GPU 的 Python 套件。

To further improve its effectiveness, this allocator was tuned for the specific memory usage patterns of deep learning.
為了進一步提高其有效性，此配置器針對深度學習的特定記憶體使用模式進行了調整。

For example, it rounds up allocations to multiples of 512 bytes to avoid fragmentation issues.
例如，它將配置向上取整到 512 位元組的倍數，以避免碎片問題。

Moreover, it maintains a distinct pool of memory for every CUDA stream (work queue).
此外，它為每個 CUDA 串流（工作佇列）維護一個獨特的記憶體池。

The one-pool-per-stream design assumption simplifies the implementation and improves the performance of the allocator: because the CPU runs ahead of the GPU, memory is freed on the CPU *before* its last use on the GPU finishes.
每個串流一個池 (one-pool-per-stream) 的設計假設簡化了實作並提高了配置器的效能：因為 CPU 在 GPU 之前執行，記憶體在 GPU 上最後一次使用完成*之前*就在 CPU 上釋放了。

Since streams serialize execution, if the free precedes the reallocation on the CPU, the same order will occur on the GPU.
由於串流序列化執行，如果釋放在 CPU 上的重新配置之前，則相同的順序也會發生在 GPU 上。

So the allocator can reallocate memory freed on the CPU immediately as long as the new allocation is used on the same stream as the freed region.
所以配置器可以立即重新配置在 CPU 上釋放的記憶體，只要新配置與釋放區域使用相同的串流。

However, if an allocation was last used on one stream and then allocated on another, additional synchronization is needed.
然而，如果配置最後在一個串流上使用，然後在另一個串流上配置，則需要額外的同步。

The one-pool-per-stream design seems limiting since the allocations end up fragmented per stream, but in practice PyTorch almost never uses multiple streams.
每個串流一個池的設計似乎有限制性，因為配置最終會按串流碎片化，但實際上 PyTorch 幾乎從不使用多個串流。

It is notoriously hard to write CUDA kernels in a way that would let them cooperatively share the GPU because exact scheduling is hardware controlled.
編寫能夠讓它們協作共享 GPU 的 CUDA 核心是非常困難的，因為精確的排程是由硬體控制的。

In practice, kernel writers usually resort to monolithic kernels that combine multiple tasks.
實際上，核心編寫者通常訴諸於結合多個任務的單體核心。

Data loading and distributed computing utilities are exceptions to the one stream design, and they carefully insert additional synchronization to avoid bad interactions with the allocator.
資料載入和分散式計算實用程式是單一串流設計的例外，它們小心地插入額外的同步以避免與配置器的不良互動。

While this design is susceptible to certain corner cases, it almost never exhibits unwanted behaviors in practical code.
雖然這種設計容易受到某些極端情況的影響，但在實際程式碼中幾乎從未表現出不必要的行為。

Most of our users are not aware of its existence.
我們的大多數使用者都沒有意識到它的存在。

### 5.4 Multiprocessing
### 5.4 多處理 (Multiprocessing)

Due to the global interpreter lock (GIL) Python’s default implementation does not allow concurrent threads to execute in parallel.
由於全域直譯器鎖 (GIL)，Python 的預設實作不允許並發執行緒平行執行。

To alleviate this problem, the Python community has established a standard `multiprocessing` module, containing a number of utilities that allow users to easily spawn child processes and implement basic inter-process communication primitives.
為了緩解這個問題，Python 社群建立了一個標準的 `multiprocessing` 模組，其中包含許多實用程式，允許使用者輕鬆產生子處理程序並實作基本的處理程序間通訊基元。

However, the implementation of the primitives uses the same form of serialization used for on-disk persistence, which is inefficient when dealing with large arrays.
然而，這些基元的實作使用了與磁碟持久化相同形式的序列化，這在處理大型陣列時效率低落。

Hence, PyTorch extends the Python `multiprocessing` module into `torch.multiprocessing`, which is a drop-in replacement for the built in package and automatically moves the data of tensors sent to other processes to shared memory instead of sending it over the communication channel.
因此，PyTorch 將 Python `multiprocessing` 模組擴展為 `torch.multiprocessing`，它是內建套件的直接替代品，並自動將發送到其他處理程序的張量資料移動到共享記憶體，而不是透過通訊通道發送。

This design greatly improves performance and makes the process isolation weaker, resulting in a programming model which more closely resembles regular threaded programs.
這種設計極大地提高了效能，並削弱了處理程序隔離，從而產生了一種更類似於常規執行緒程式的程式設計模型。

Users can easily implement heavily parallel programs that operate on independent GPUs but later synchronize gradients using all-reduce style primitives.
使用者可以輕鬆實作在獨立 GPU 上執行但在稍後使用 all-reduce 風格基元同步梯度的重度平行程式。

Another unique feature of this system is that it transparently handles sharing of CUDA tensors, making it easy to implement techniques like Hogwild [42].
該系統的另一個獨特功能是它透明地處理 CUDA 張量的共享，使得實作像 Hogwild [42] 這樣的技術變得容易。

### 5.5 Reference counting
### 5.5 參照計數

Users often design their models to utilize all memory available during training, and increasing batch sizes is a common technique of speeding up the process.
使用者經常設計他們的模型以利用訓練期間所有可用的記憶體，而增加批次大小 (batch sizes) 是加速該過程的常用技術。

Therefore, to deliver great performance, PyTorch has to treat memory as a scarce resource that it needs to manage carefully.
因此，為了提供出色的效能，PyTorch 必須將記憶體視為稀缺資源，需要仔細管理。

Libraries with eager semantics have to manage tensor memory without knowing how it will be used in the future.
具有急切語義 (eager semantics) 的函式庫必須在不知道將來如何使用的情況下管理張量記憶體。

Garbage collection is the typical way to handle this automatically because it has good amortized performance.
垃圾回收 (Garbage collection) 是自動處理此問題的典型方法，因為它具有良好的攤銷效能。

In this approach, the runtime periodically investigates the state of the system, enumerates used objects and frees everything else.
在這種方法中，執行時定期調查系統狀態，列舉使用的物件並釋放其他所有物件。

However, by deferring the deallocation, it causes the program to use more memory overall [43].
然而，透過延遲釋放，它會導致程式整體使用更多記憶體 [43]。

Given the scarcity of GPU memory, these overheads are unacceptable.
鑑於 GPU 記憶體的稀缺性，這些開銷是不可接受的。

In fact, Torch7 utilized the garbage collector built into Lua, and a common anti-pattern among the users was to sprinkle the program with explicit triggers to the garbage collector, hoping that the memory errors go away.
事實上，Torch7 利用了內建於 Lua 中的垃圾回收器，使用者中常見的反模式是在程式中散佈垃圾回收器的顯式觸發器，希望記憶體錯誤消失。

PyTorch takes a different approach: it relies on a reference counting scheme to track the number of uses of each tensor, and frees the underlying memory *immediately* once this count reaches zero.
PyTorch 採取了不同的方法：它依賴於參照計數方案來追蹤每個張量的使用次數，並在該計數達到零時*立即*釋放底層記憶體。

Note that PyTorch tracks both references internal to the `libtorch` library and external references made by users in their Python code by integrating with Python’s own reference counting mechanism.
請注意，PyTorch 透過與 Python 自己的參照計數機制整合，來追蹤 `libtorch` 函式庫內部的參照以及使用者在其 Python 程式碼中進行的外部參照。

This ensures that memory is released exactly when tensors become unneeded.
這確保了記憶體在張量變得不需要時被精確釋放。

One notable caveat is that we can only guarantee the desired performance characteristics in implementations of languages that either already utilize reference counting (CPython, Swift, but not PyPy or many scripting languages such as Lua), and those that allow for user-defined behavior for assignment, copies, and moves (e.g. C++, Rust).
一個值得注意的警告是，我們只能在以下語言的實作中保證所需的效能特徵：要麼已經利用參照計數（CPython, Swift，但不是 PyPy 或許多像 Lua 這樣的腳本語言），要麼允許使用者定義賦值、複製和移動的行為（例如 C++, Rust）。

Bindings to implementations that do not satisfy those criteria will have to implement their own specialized memory management on top of PyTorch.
不滿足這些標準的實作的綁定將必須在 PyTorch 之上實作其自己的專門記憶體管理。

## 6 Evaluation
## 6 評估

In this section we compare the performance of PyTorch with several other commonly-used deep learning libraries, and find that it achieves competitive performance across a range of tasks.
在本節中，我們將 PyTorch 的效能與其他幾個常用的深度學習函式庫進行比較，發現它在一系列任務中都達到了具競爭力的效能。

All experiments were performed on a workstation with two Intel Xeon E5-2698 v4 CPUs and one NVIDIA Quadro GP100 GPU.
所有實驗均在配備兩個 Intel Xeon E5-2698 v4 CPU 和一個 NVIDIA Quadro GP100 GPU 的工作站上進行。

### 6.1 Asynchronous dataflow
### 6.1 非同步資料流

We start by quantifying the ability of PyTorch to asynchronously execute dataflow on GPU.
我們首先量化 PyTorch 在 GPU 上非同步執行資料流的能力。

We use the built-in profiler [44] to instrument various benchmarks and record a timeline of the execution of a single training step.
我們使用內建的分析器 (profiler) [44] 來檢測各種基準測試，並記錄單個訓練步驟執行的時間軸。

Figure 1 shows a representative timeline of execution for the first few operations of a ResNet-50 model.
圖 1 顯示了 ResNet-50 模型前幾個操作的代表性執行時間軸。

The host CPU which queues the work quickly outpaces the execution of the operators on the GPU.
排隊工作的主機 CPU 很快就超過了 GPU 上運算子的執行速度。

This allows PyTorch to achieve almost perfect device utilization.
這使得 PyTorch 能夠實現幾乎完美的裝置利用率。

In this example, GPU execution takes around three times longer than CPU scheduling.
在此範例中，GPU 執行時間大約是 CPU 排程時間的三倍。

The exact ratio depends on the relative performance of the host CPU and the GPU, as well as the number of elements in each tensor and the average arithmetic complexity of the floating point computations to be performed on the GPU.
確切的比率取決於主機 CPU 和 GPU 的相對效能，以及每個張量中的元素數量和要在 GPU 上執行的浮點計算的平均算術複雜度。

**Figure 1:** A trace of the first few operators of Resnet-50. The top row depicts the execution of the control flow running on the host CPU. The gray areas are Python code executed by its interpreter. The colored areas correspond to the work done on the host CPU to queue various operators (convolution, batch normalization, and so on). The bottom row shows the corresponding execution of those operators on the GPU. The arrows pair the two events in time.
**圖 1:** Resnet-50 前幾個運算子的追蹤。頂行描繪了在主機 CPU 上執行的控制流的執行。灰色區域是由其直譯器執行的 Python 程式碼。彩色區域對應於在主機 CPU 上排隊各種運算子（卷積、批次正規化等）所做的工作。底行顯示了這些運算子在 GPU 上的相應執行。箭頭將這兩個事件在時間上配對。

### 6.2 Memory management
### 6.2 記憶體管理

We used the NVIDIA profiler to trace the execution of the CUDA runtime as well as the execution of the CUDA kernels launched during one training iteration of the ResNet-50 model.
我們使用 NVIDIA 分析器來追蹤 CUDA 執行時的執行，以及在 ResNet-50 模型的一次訓練迭代期間啟動的 CUDA 核心的執行。

As shown in Figure 2, the behavior of the first iteration differs significantly from that of subsequent ones.
如圖 2 所示，第一次迭代的行為與隨後的迭代有顯著差異。

At first, calls to the CUDA memory management functions (`cudaMalloc` and `cudaFree`) slow down the execution quite dramatically by blocking the CPU thread for long periods of time, hence lowering the utilization of the GPU.
起初，對 CUDA 記憶體管理函式（`cudaMalloc` 和 `cudaFree`）的呼叫透過長時間阻塞 CPU 執行緒，相當劇烈地減慢了執行速度，從而降低了 GPU 的利用率。

This effect disappears in subsequent iterations as the PyTorch caching memory allocator starts reusing previously allocated regions.
隨著 PyTorch 快取記憶體配置器開始重複使用先前配置的區域，這種效應在隨後的迭代中消失。

**Figure 2:** Annotated traces of the execution of ResNet-50 on GPU.
**圖 2:** ResNet-50 在 GPU 上執行的註釋追蹤。

### 6.3 Benchmarks
### 6.3 基準測試

Finally, we can get an overall sense of single-machine eager mode performance of PyTorch by comparing it to three popular graph-based deep learning frameworks (CNTK, MXNet and TensorFlow), a define-by-run framework (Chainer), and production oriented platform (PaddlePaddle).
最後，我們可以透過將 PyTorch 與三個流行的基於圖的深度學習框架（CNTK、MXNet 和 TensorFlow）、一個執行即定義 (define-by-run) 框架 (Chainer) 和面向生產的平台 (PaddlePaddle) 進行比較，來獲得 PyTorch 單機急切模式效能的整體概念。

The Appendix details all the steps needed to reproduce our setup.
附錄詳細介紹了重現我們的設定所需的所有步驟。

Our results are summarized in Table 1.
我們的結果總結在表 1 中。

On all the benchmarks, the performance of PyTorch is within 17% of that of of the fastest framework.
在所有基準測試中，PyTorch 的效能都在最快框架的 17% 以內。

We attribute this result to the fact that these tools offload most of the computation to the same version of the cuDNN and cuBLAS libraries.
我們將此結果歸因於這些工具將大部分計算卸載到相同版本的 cuDNN 和 cuBLAS 函式庫。

**已翻譯的繁體中文表格 Table 1**

| 框架 (Framework) | 吞吐量（越高越好）<br>Throughput (higher is better) | | | | | |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| | **AlexNet** | **VGG-19** | **ResNet-50** | **MobileNet** | **GNMTv2** | **NCF** |
| Chainer | 778 ± 15 | N/A | **219 ± 1** | N/A | N/A | N/A |
| CNTK | 845 ± 8 | 84 ± 3 | 210 ± 1 | N/A | N/A | N/A |
| MXNet | **1554 ± 22** | 113 ± 1 | 218 ± 2 | 444 ± 2 | N/A | N/A |
| PaddlePaddle | 933 ± 123 | 112 ± 2 | 192 ± 4 | **557 ± 24** | N/A | N/A |
| TensorFlow | 1422 ± 27 | 66 ± 2 | 200 ± 1 | 216 ± 15 | 9631 ± 1.3% | 4.8e6 ± 2.9% |
| PyTorch | 1547 ± 316 | **119 ± 1** | 212 ± 2 | 463 ± 17 | **15512 ± 4.8%** | **5.4e6 ± 3.4%** |

**Table 1:** Training speed for 6 models using 32bit floats. Throughput is measured in images per second for the AlexNet, VGG-19, ResNet-50, and MobileNet models, in tokens per second for the GNMTv2 model, and in samples per second for the NCF model. The fastest speed for each model is shown in bold.
**表 1:** 使用 32 位元浮點數的 6 個模型的訓練速度。AlexNet、VGG-19、ResNet-50 和 MobileNet 模型的吞吐量以每秒影像數 (images per second) 測量，GNMTv2 模型以每秒標記數 (tokens per second) 測量，NCF 模型以每秒樣本數 (samples per second) 測量。每個模型的最快速度以粗體顯示。

### 6.4 Adoption
### 6.4 採用

The validity of design decisions and their impact on ease-of-use is hard to measure.
設計決策的有效性及其對易用性的影響很難衡量。

As a proxy, we tried to quantify how well the machine learning community received PyTorch by counting how often various machine learning tools (including Caffe, Chainer, CNTK, Keras, MXNet, PyTorch, TensorFlow, and Theano) are mentioned on arXiv e-Prints since the initial release of PyTorch in January 2017.
作為代理指標，我們試圖量化機器學習社群對 PyTorch 的接受程度，方法是統計自 2017 年 1 月 PyTorch 首次發布以來，各種機器學習工具（包括 Caffe, Chainer, CNTK, Keras, MXNet, PyTorch, TensorFlow 和 Theano）在 arXiv 電子預印本中被提及的頻率。

In Figure 3 we report the monthly number of mentions of the word "PyTorch" as a percentage of all mentions among these deep learning frameworks.
在圖 3 中，我們報告了每個月提到 "PyTorch" 一詞的次數佔這些深度學習框架中所有提及次數的百分比。

We counted tools mentioned multiple times in a given paper only once, and made the search case insensitive to account for various spellings.
對於同一篇論文中多次提到的工具，我們只計算一次，並且使搜尋不區分大小寫，以考慮各種拼寫。

**Figure 3:** Among arXiv papers each month that mention common deep learning frameworks, percentage of them that mention PyTorch.
**圖 3:** 在每個月提及常見深度學習框架的 arXiv 論文中，提及 PyTorch 的百分比。

## 7 Conclusion and future work
## 7 結論與未來工作

PyTorch has become a popular tool in the deep learning research community by combining a focus on usability with careful performance considerations.
PyTorch 結合了對可用性的關注和對效能的仔細考量，已成為深度學習研究社群中的熱門工具。

In addition to continuing to support the latest trends and advances in deep learning, in the future we plan to continue to improve the speed and scalability of PyTorch.
除了繼續支援深度學習的最新趨勢和進展外，未來我們計劃繼續提高 PyTorch 的速度和可擴展性。

Most notably, we are working on the PyTorch JIT: a suite of tools that allow PyTorch programs to be executed outside of the Python interpreter where they can be further optimized.
最值得注意的是，我們正在開發 PyTorch JIT：這是一套允許 PyTorch 程式在 Python 直譯器之外執行的工具，在那裡它們可以得到進一步最佳化。

We also intend to improve support for distributed computation by providing efficient primitives for data parallelism as well as a Pythonic library for model parallelism based around remote procedure calls.
我們還打算透過提供用於資料平行的高效基元，以及基於遠端程序呼叫 (RPC) 的用於模型平行的 Python 風格函式庫，來改進對分散式計算的支援。

## 8 Acknowledgements
## 8 致謝

We are grateful to the PyTorch community for their feedback and contributions that greatly influenced the design and implementation of PyTorch.
我們感謝 PyTorch 社群的回饋和貢獻，這極大地影響了 PyTorch 的設計和實作。

We thank all the PyTorch core team members, contributors and package maintainers including Ailing Zhang, Alex Suhan, Alfredo Mendoza, Alican Bozkurt, Andrew Tulloch, Ansha Yu, Anthony Shoumikhin, Bram Wasti, Brian Vaughan, Christian Puhrsch, David Reiss, David Riazati, Davide Libenzi, Dmytro Dzhulgakov, Dwaraj Rajagopal, Edward Yang, Elias Ellison, Fritz Obermeyer, George Zhang, Hao Lu, Hong Xu, Hung Duong, Igor Fedan, Ilia Cherniavskii, Iurii Zdebskyi, Ivan Kobzarev, James Reed, Jeff Smith, Jerry Chen, Jerry Zhang, Jiakai Liu, Johannes M. Dieterich, Karl Ostmo, Lin Qiao, Martin Yuan, Michael Suo, Mike Ruberry, Mikhail Zolothukhin, Mingzhe Li, Neeraj Pradhan, Nick Korovaiko, Owen Anderson, Pavel Belevich, Peter Johnson, Pritam Damania, Raghuraman Krishnamoorthi, Richard Zou, Roy Li, Rui Zhu, Sebastian Messmer, Shen Li, Simon Wang, Supriya Rao, Tao Xu, Thomas Viehmann, Vincent Quenneville-Belair, Vishwak Srinivasan, Vitaly Fedyunin, Wanchao Liang, Wei Yang, Will Feng, Xiaomeng Yang, Xiaoqiang Zheng, Xintao Chen, Yangqing Jia, Yanli Zhao, Yinghai Lu and Zafar Takhirov.
我們感謝所有 PyTorch 核心團隊成員、貢獻者和套件維護者，包括 [名單保留原文，人名通常不翻譯]。

## References
## 參考文獻

[1] Yangqing "Jia, Evan Shelhamer, Jeff Donahue, Sergey Karayev, Jonathan Long, Ross Girshick, Sergio Guadarrama, and Trevor" Darrell. "caffe: Convolutional architecture for fast feature embedding". "arXiv preprint arXiv:1408.5093", "2014".

[2] Frank Seide and Amit Agarwal. Cntk: Microsoft’s open-source deep-learning toolkit. In Proceedings of the 22Nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, KDD ’16, pages 2135–2135, New York, NY, USA, 2016. ACM.

[3] Martín Abadi, Ashish Agarwal, Paul Barham, Eugene Brevdo, Zhifeng Chen, Craig Citro, Greg S. Corrado, Andy Davis, Jeffrey Dean, Matthieu Devin, Sanjay Ghemawat, Ian Goodfellow, Andrew Harp, Geoffrey Irving, Michael Isard, Yangqing Jia, Rafal Jozefowicz, Lukasz Kaiser, Manjunath Kudlur, Josh Levenberg, Dandelion Mané, Rajat Monga, Sherry Moore, Derek Murray, Chris Olah, Mike Schuster, Jonathon Shlens, Benoit Steiner, Ilya Sutskever, Kunal Talwar, Paul Tucker, Vincent Vanhoucke, Vijay Vasudevan, Fernanda Viégas, Oriol Vinyals, Pete Warden, Martin Wattenberg, Martin Wicke, Yuan Yu, and Xiaoqiang Zheng. TensorFlow: Large-scale machine learning on heterogeneous systems, 2015. Software available from tensorflow.org.

[4] Theano Development Team. Theano: A Python framework for fast computation of mathematical expressions. arXiv e-prints, abs/1605.02688, May 2016.

[5] Seiya Tokui, Kenta Oono, Shohei Hido, and Justin Clayton. Chainer: a next-generation open source framework for deep learning. In Proceedings of Workshop on Machine Learning Systems (LearningSys) in The Twenty-ninth Annual Conference on Neural Information Processing Systems (NIPS), 2015.

[6] Ronan Collobert, Samy Bengio, and Johnny Mariéthoz. Torch: a modular machine learning software library. Technical report, Idiap, 2002.

[7] G. Neubig, C. Dyer, Y. Goldberg, A. Matthews, W. Ammar, A. Anastasopoulos, M. Ballesteros, D. Chiang, D. Clothiaux, T. Cohn, K. Duh, M. Faruqui, C. Gan, D. Garrette, Y. Ji, L. Kong, A. Kuncoro, G. Kumar, C. Malaviya, P. Michel, Y. Oda, M. Richardson, N. Saphra, S. Swayamdipta, and P. Yin. DyNet: The Dynamic Neural Network Toolkit. ArXiv e-prints, January 2017.

[8] Philip S. Abrams. An APL Machine. PhD thesis, Stanford University, 1970.

[9] The MathWorks, Inc., Natick, Massachusetts, United States. MATLAB and Statistics Toolbox.

[10] R Core Team. R: A Language and Environment for Statistical Computing. R Foundation for Statistical Computing, Vienna, Austria.

[11] Jeff Bezanson, Alan Edelman, Stefan Karpinski, and Viral B Shah. Julia: A fresh approach to numerical computing. SIAM review, 59(1):65–98, 2017.

[12] Travis Oliphant. NumPy: A guide to NumPy. USA: Trelgol Publishing, 2006. http://www.numpy.org/.

[13] Gaël Guennebaud, Benoît Jacob, et al. Eigen v3. http://eigen.tuxfamily.org, 2010.

[14] Y LeCun and L Bottou. Lush reference manual. Technical report, code available at http://lush.sourceforge.net, 2002.

[15] Atilim Gunes Baydin, Barak A. Pearlmutter, Alexey Andreyevich Radul, and Jeffrey Mark Siskind. Automatic differentiation in machine learning: A survey. J. Mach. Learn. Res., 18(1):5595–5637, January 2017.

[16] Dougal Maclaurin. Modeling, Inference and Optimization with Composable Differentiable Procedures. PhD thesis, Harvard University, April 2016.

[17] Matthew Johnson et. al. Jax. https://github.com/google/jax, 2018.

[18] Mike Innes et. al. Flux.jl. https://github.com/FluxML/Flux.jl, 2018.

[19] Eric Jones, Travis Oliphant, Pearu Peterson, et al. SciPy: Open source scientific tools for Python, 2001–. http://www.scipy.org/.

[20] Wes McKinney. Data structures for statistical computing in python. In Proceedings of the 9th Python in Science Conference, 51-56, 2010.

[21] Pierre Sermanet, Koray Kavukcuoglu, and Yann LeCun. Eblearn: Open-source energy-based learning in c++. In 2009 21st IEEE International Conference on Tools with Artificial Intelligence, pages 693–697. IEEE, 2009.

[22] Sharan Chetlur, Cliff Woolley, Philippe Vandermersch, Jonathan D. Cohen, John Tran, Bryan Catanzaro, and Evan Shelhamer. cudnn: Efficient primitives for deep learning. CoRR, abs/1410.0759, 2014.

[23] Andrew Lavin. maxdnn: An efficient convolution kernel for deep learning with maxwell gpus, January 2015.

[24] Andrew Lavin and Scott Gray. Fast algorithms for convolutional neural networks. 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 4013–4021, 2016.

[25] Ronan Collobert, Koray Kavukcuoglu, and Clément Farabet. Torch7: A matlab-like environment for machine learning. In NIPS 2011, 2011.

[26] Richard Gabriel. The rise of worse is better. http://dreamsongs.com/RiseOfWorseIsBetter.html.

[27] Yann LeCun and Corinna Cortes. MNIST handwritten digit database. http://yann.lecun.com/exdb/mnist/.

[28] Oriol Vinyals, Timo Ewalds, Sergey Bartunov, Petko Georgiev, Alexander Sasha Vezhnevets, Michelle Yeo, Alireza Makhzani, Heinrich Küttler, John Agapiou, Julian Schrittwieser, John Quan, Stephen Gaffney, Stig Petersen, Karen Simonyan, Tom Schaul, Hado van Hasselt, David Silver, Timothy P. Lillicrap, Kevin Calderone, Paul Keet, Anthony Brunasso, David Lawrence, Anders Ekermo, Jacob Repp, and Rodney Tsing. Starcraft II: A new challenge for reinforcement learning. CoRR, abs/1708.04782, 2017.

[29] DMLC. Dlpack: Open in memory tensor structure. https://github.com/dmlc/dlpack.

[30] Adam Paszke, Sam Gross, Soumith Chintala, Gregory Chanan, Edward Yang, Zachary DeVito, Zeming Lin, Alban Desmaison, Luca Antiga, and Adam Lerer. Automatic differentiation in pytorch. In NIPS Workshop, 2017.

[31] Dan Piponi. Automatic differentiation, C++ templates, and photogrammetry. J. Graphics, GPU, & Game Tools, 9(4):41–55, 2004.

[32] Holger Leuck and Hans-Hellmut Nagel. Automatic differentiation facilitates of-integration into steering-angle-based road vehicle tracking. In 1999 Conference on Computer Vision and Pattern Recognition (CVPR ’99), 23-25 June 1999, Ft. Collins, CO, USA, pages 2360–2365, 1999.

[33] The Python team. The cpython global interpreter lock. https://wiki.python.org/moin/GlobalInterpreterLock.

[34] Giovanni Petrantoni and Jörg Wollenschläger. Nimtorch. https://github.com/fragcolor-xyz/nimtorch.

[35] Austin Huang, Junji Hashimoto, and Sam Stites. Hasktorch. https://github.com/hasktorch/hasktorch.

[36] G. Synnaeve, Z. Lin, J. Gehring, D. Gant, V. Mella, V. Khalidov, N. Carion, and N. Usunier. Forward modeling for partial observation strategy games - a starcraft defogger. In Advances in Neural Information Processing Systems, pages 10761–10771, 2018.

[37] The PyTorch team. Torch Script. https://pytorch.org/docs/stable/jit.html.

[38] Justin Luitjens. Cuda streams. GPU technology conference, 2014.

[39] Emery D. Berger, Kathryn S. McKinley, Robert D. Blumofe, and Paul R. Wilson. Hoard: A scalable memory allocator for multithreaded applications. In Proceedings of the Ninth International Conference on Architectural Support for Programming Languages and Operating Systems, ASPLOS IX, pages 117–128, New York, NY, USA, 2000. ACM.

[40] J. Evans. A scalable concurrent malloc(3) implementation for freebsd. In In BSDCan — The Technical BSD Conference, May 2006.

[41] S. Ghemawat and P. Menage. Tcmalloc: Thread-caching malloc.

[42] Benjamin Recht, Christopher Ré, Stephen J. Wright, and Feng Niu. Hogwild: A lock-free approach to parallelizing stochastic gradient descent. In Advances in Neural Information Processing Systems 24: 25th Annual Conference on Neural Information Processing Systems 2011. Proceedings of a meeting held 12-14 December 2011, Granada, Spain., pages 693–701, 2011.

[43] Matthew Hertz and Emery D. Berger. Quantifying the performance of garbage collection vs. explicit memory management. In Proceedings of the 20th Annual ACM SIGPLAN Conference on Object-oriented Programming, Systems, Languages, and Applications, OOPSLA ’05, pages 313–326, New York, NY, USA, 2005. ACM.

[44] The PyTorch team. Pytorch Autograd Profiler. https://pytorch.org/docs/1.0.1/autograd.html#profiler.
