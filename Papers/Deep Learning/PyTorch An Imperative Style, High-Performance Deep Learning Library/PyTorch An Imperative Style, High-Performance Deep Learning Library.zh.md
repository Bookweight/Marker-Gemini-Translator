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
# PyTorch：一種命令式、高性能的深度學習庫

## Abstract
Deep learning frameworks have often focused on either usability or speed, but not both. PyTorch is a machine learning library that shows that these two goals are in fact compatible: it provides an imperative and Pythonic programming style that supports code as a model, makes debugging easy and is consistent with other popular scientific computing libraries, while remaining efficient and supporting hardware accelerators such as GPUs.
深度學習框架通常專注於易用性或速度，但往往無法兼得。PyTorch 是一個機器學習庫，它證明了這兩個目標實際上是相容的：它提供了一種命令式且符合 Python 習慣的編程風格，支持將代碼作為模型，使調試變得容易，並且與其他流行的科學計算庫保持一致，同時保持高效並支持 GPU 等硬體加速器。

In this paper, we detail the principles that drove the implementation of PyTorch and how they are reflected in its architecture. We emphasize that every aspect of PyTorch is a regular Python program under the full control of its user. We also explain how the careful and pragmatic implementation of the key components of its runtime enables them to work together to achieve compelling performance. We demonstrate the efficiency of individual subsystems, as well as the overall speed of PyTorch on several common benchmarks.
在本文中，我們詳細介紹了推動 PyTorch 實現的原則以及它們如何反映在其架構中。我們強調 PyTorch 的每個方面都是一個由用戶完全控制的常規 Python 程序。我們還解釋了其運行時關鍵組件的精心且務實的實現如何使它們協同工作以實現引人注目的性能。我們展示了各個子系統的效率，以及 PyTorch 在幾個常用基準測試中的整體速度。

---

## 1 Introduction
With the increased interest in deep learning in recent years, there has been an explosion of machine learning tools. Many popular frameworks such as Caffe [1], CNTK [2], TensorFlow [3], and Theano [4], construct a static dataflow graph that represents the computation and which can then be applied repeatedly to batches of data. This approach provides visibility into the whole computation ahead of time, and can theoretically be leveraged to improve performance and scalability. However, it comes at the cost of ease of use, ease of debugging, and flexibility of the types of computation that can be represented.
隨著近年來對深度學習的興趣增加，機器學習工具出現了爆炸式增長。許多流行的框架（如 Caffe [1]、CNTK [2]、TensorFlow [3] 和 Theano [4]）構建了一個表示計算的靜態數據流圖，然後可以將其重複應用於成批數據。這種方法提供了對提前進行整個計算的可見性，並且理論上可以用於提高性能和可擴展性。然而，這是以易用性、易調試性以及可表示計算類型的靈活性為代價的。

Prior work has recognized the value of dynamic eager execution for deep learning, and some recent frameworks implement this define-by-run approach, but do so either at the cost of performance (Chainer [5]) or using a less expressive, faster language (Torch [6], DyNet [7]), which limits their applicability.
先前的工作已經認識到動態急切執行（dynamic eager execution）對深度學習的價值，一些最近的框架實現了這種「運行即定義」（define-by-run）的方法，但要麼以性能為代價（Chainer [5]），要麼使用表達能力較弱但速度較快的語言（Torch [6], DyNet [7]），這限制了它們的適用性。

However, with careful implementation and design choices, dynamic eager execution can be achieved largely without sacrificing performance. This paper introduces PyTorch, a Python library that performs immediate execution of dynamic tensor computations with automatic differentiation and GPU acceleration, and does so while maintaining performance comparable to the fastest current libraries for deep learning. This combination has turned out to be very popular in the research community with, for instance, 296 ICLR 2019 submissions mentioning PyTorch.
然而，通過精心的實現和設計選擇，可以很大程度上在不犧牲性能的情況下實現動態急切執行。本文介紹了 PyTorch，這是一個 Python 庫，它執行帶有自動微分和 GPU 加速的動態張量計算的立即執行，同時保持與當前最快的深度學習庫相當的性能。事實證明，這種組合在研究社群中非常受歡迎，例如，有 296 篇 ICLR 2019 的投稿提到了 PyTorch。

---

## 2 Background
PyTorch builds on several trends in scientific computing: multidimensional arrays (tensors) as first-class objects, automatic differentiation for efficient optimization, the shift towards the open-source Python ecosystem (NumPy, SciPy), and the commoditization of massively parallel hardware (GPUs). It provides an array-based programming model accelerated by GPUs and differentiable via automatic differentiation integrated in the Python ecosystem.
PyTorch 建立在科學計算的幾個趨勢之上：將多維數組（張量）作為一等對象、用於高效優化的自動微分、轉向開源 Python 生態系統（NumPy, SciPy）以及大規模並行硬體（GPU）的商品化。它提供了一種由 GPU 加速且可通過 Python 生態系統中集成的自動微分進行微分的基於數組的編程模型。

---

## 3 Design principles
PyTorch follows four main principles:
1. **Be Pythonic:** Integrate naturally with the Python ecosystem and follow its design goals.
2. **Put researchers first:** Handle ML complexity internally and provide intuitive APIs.
3. **Provide pragmatic performance:** Deliver compelling speed without sacrificing simplicity.
4. **Worse is better:** Prioritize simple solutions that are easy to maintain and evolve.

PyTorch 遵循四個主要原則：
1. **符合 Python 習慣 (Be Pythonic)：** 自然地與 Python 生態系統集成並遵循其設計目標。
2. **研究人員優先 (Put researchers first)：** 在內部處理機器學習的複雜性並提供直觀的 API。
3. **提供務實的性能 (Provide pragmatic performance)：** 在不犧牲簡單性的情況下提供引人注目的速度。
4. **以簡馭繁 (Worse is better)：** 優先選擇易於維護和演進的簡單解決方案。

---

## 4 Usability centric design
PyTorch preserves the imperative programming model of Python, allowing models to be expressed as regular Python programs. This makes defining layers, composing models, and debugging straightforward using familiar tools like print statements and standard debuggers.
PyTorch 保留了 Python 的命令式編程模型，允許將模型表示為常規 Python 程序。這使得定義層、組合模型以及使用 print 語句和標準調試器等熟悉工具進行調試變得非常直觀。

---

## 5 Performance focused implementation
To achieve high performance from Python, PyTorch uses:
- **An efficient C++ core (libtorch):** Implements tensor operations and autograd without being limited by the Python GIL.
- **Asynchronous GPU execution:** Overlaps CPU control flow with GPU data flow.
- **Custom caching allocator:** Efficiently manages GPU memory to avoid bottlenecks.
- **Reference counting:** Frees memory immediately when tensors are no longer needed, critical for GPU memory constraints.

為了在 Python 中實現高性能，PyTorch 使用了：
- **高效的 C++ 核心 (libtorch)：** 實現張量操作和自動微分，而不受 Python GIL 的限制。
- **異步 GPU 執行：** 將 CPU 控制流與 GPU 數據流重疊。
- **自定義快取分配器：** 高效管理 GPU 內存以避免瓶頸。
- **引用計數：** 在張量不再需要時立即釋放內存，這對於 GPU 內存限制至關重要。

---

## 6 Evaluation
Benchmarks show that PyTorch achieves performance within 17% of the fastest graph-based frameworks while offering much higher usability. Its adoption in the research community has grown significantly since its release.
基準測試顯示，PyTorch 的性能在最快基於圖的框架的 17% 以內，同時提供了更高的易用性。自發布以來，其在研究社群中的採用率顯著增長。

---

## 7 Conclusion
PyTorch has become a popular tool by balancing usability and performance. Future work includes the PyTorch JIT for optimization outside of Python and improved support for distributed computation.
PyTorch 通過平衡易用性和性能已成為一種流行的工具。未來的工作包括用於 Python 之外優化的 PyTorch JIT，以及改進對分佈式計算的支持。
