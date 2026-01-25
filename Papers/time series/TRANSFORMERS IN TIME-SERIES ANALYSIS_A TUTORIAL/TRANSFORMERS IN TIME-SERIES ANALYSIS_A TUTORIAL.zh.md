---
title: TRANSFORMERS IN TIME-SERIES ANALYSIS_A TUTORIAL
field: Time_Series
status: Imported
created_date: 2026-01-12
pdf_link: "[[TRANSFORMERS IN TIME-SERIES ANALYSIS_A TUTORIAL.pdf]]"
tags:
  - paper
  - Time_series
---

#Translated_paper

Sabeen Ahmed
Department of Machine Learning
Moffitt Cancer Center
12902 USF Magnolia Drive, Tampa, FL, 33612
sabeen.ahmed@moffitt.org

Ian E. Nielsen
Department of Electrical and Computer Engineering
Rowan University
201 Mullica Hill Rd, Glassboro, NJ, 08028
nielseni6@rowan.edu

Aakash Tripathi
Department of Machine Learning
Moffitt Cancer Center
12902 USF Magnolia Drive, Tampa, FL, 33612
aakash.tripathi@moffitt.org

Shamoon Siddiqui
Department of Electrical and Computer Engineering
Rowan University
201 Mullica Hill Rd, Glassboro, NJ, 08028
siddiq76@rowan.edu

Ravi P. Ramachandran
Department of Electrical and Computer Engineering
Rowan University
201 Mullica Hill Rd, Glassboro, NJ, 08028
ravi@rowan.edu

Ghulam Rasool
Department of Machine Learning
Moffitt Cancer Center
12902 USF Magnolia Drive, Tampa, FL, 33612
ghulam.rasool@moffitt.org

2023年7月4日

# ABSTRACT

Transformer 架構具有廣泛的應用，特別是在自然語言處理和電腦視覺領域。最近，Transformers 已被應用於時間序列分析的各個方面。本教學概述了 Transformer 架構、其應用以及時間序列分析領域近期研究的範例集合。我們深入解釋了 Transformer 的核心組件，包括 self-attention 機制、positional encoding、multi-head 以及 encoder/decoder。我們重點介紹了對初始 Transformer 架構的幾項增強功能，以應對時間序列任務。本教學也提供了最佳實踐和技術，以克服有效訓練 Transformers 進行時間序列分析的挑戰。

**Keywords** Transformer, time-series, self-attention, positional encoding

# 1 Introduction

Transformers 屬於一類機器學習模型，它們使用 self-attention 或 scaled dot-product operation 作為其主要學習機制。Transformers 最初是為神經機器翻譯而提出的——這是最具挑戰性的自然語言處理 (NLP) 任務之一 [1]。最近，Transformers 已成功應用於解決機器學習中的各種問題，並取得了最先進的性能 [2]。除了傳統的 NLP 任務外，其他領域的例子還包括影像分類 [3]、物件偵測與分割 [4]、影像與語言生成 [5]、強化學習中的循序決策 [6]、多模態（文字、語音和影像）資料處理 [7]，以及表格和時間序列資料的分析 [8]。本教學論文著重於使用 Transformers 進行時間序列分析。

時間序列資料由按時間順序記錄的有序樣本、觀察值或特徵組成。時間序列資料集在許多現實世界的應用中自然產生，其中資料是在固定的採樣間隔內記錄的。例子包括股票價格、數位化語音訊號、交通測量、天氣模式的感測器資料、生物醫學測量以及隨時間記錄的各類人口資料。時間序列分析可能包括處理多種任務的數值資料，包括預測、預測和分類。統計方法涉及使用各種類型的模型，例如 autoregressive (AR)、moving average (MA)、auto-regressive moving average (ARMA)、AR Integrated MA (ARIMA) 和頻譜分析技術。

具有專門用於處理資料序列性質的組件和架構的機器學習模型已在文獻中被廣泛提出並為社群所使用。這些機器學習模型中最著名的是 Recurrent Neural Networks (RNNs) 及其流行變體，包括 Long Short-Term Memory (LSTM) 和 Gated Recurrent Units (GRU) [9], [10], [11], [12]。這些模型循序處理成批的資料，一次一個樣本，並使用著名的梯度下降演算法優化未知的模型參數。用於更新模型參數的梯度資訊是使用 back-propagation through time (BPTT) [13] 計算的。LSTMs 和 GRUs 已成功用於許多應用 [14], [15], [16], [17], [18]。然而，由於輸入資料的循序處理以及與 BPTT 相關的挑戰，它們存在一些限制，尤其是在處理具有長依賴性的資料集時。LSTM 和 GRU 模型的訓練過程也受到 vanishing and exploding gradient problems [19], [20] 的困擾。在處理長序列時，梯度下降演算法（使用 BPTT）可能無法更新模型參數，因為梯度資訊會遺失（接近零或無窮大）。此外，這些模型通常無法從圖形處理單元 (GPUs)、張量處理單元 (TPUs) 和其他硬體加速器 [21] 提供的平行計算中受益。某些架構修改和訓練技巧可能在一定程度上幫助 LSTMs 和 GRUs 緩解與梯度相關的問題。然而，在現代硬體提供的有限平行化下，從長資料序列中學習的挑戰影響了基於 RNN 的模型的有效性和效率 [22]。

Transformer 架構允許對序列資料進行平行計算 [23]，而不會大幅增加網路的複雜性 [24]。由於其架構，Transformers 可以利用圖形處理單元 (GPUs) 和張量處理單元 (TPUs) 的平行處理能力 [25]。鑑於基於注意力的操作，Transformers 可以在序列中的所有元素之間相互關聯資訊，而不會像 RNNs 及其變體那樣遭受 vanishing gradients 的困擾 [26],[27], [28]。

Transformers 大幅改善了長期和多變量時間序列預測 [29], [30]。然而，self-attention 機制的計算複雜度和記憶體需求很高，妨礙了長序列建模。文獻中提出了各種修改來優化 Transformer 在時間序列任務上的性能 [31], [32], [33]。訓練大型 Transformer 模型具有挑戰性，特別是對於大型資料集。文獻中提出了許多技術來有效地訓練大型 Transformer 模型。這些技術包括 layer-wise adaptive large batch optimization [34]、distributed training [35]、knowledge inheritance [36]、progressive training [37] 以及將較小模型的參數對應到初始化較大模型 [38]。

本教學的貢獻包括：

1.  對 Transformer 操作背後的直覺進行說明性解釋。這些直覺有助於我們理解 Transformers 如何徹底改變 NLP 任務。
2.  討論 Transformer 架構中引入的各種方面和技術以及用於高效時間序列分析的內部操作。
3.  彙編一些 Transformers 用於時間序列分析的用例，包括比較性能。
4.  有效訓練 Transformers 的技術和技巧指南。

基於 Transformer 的模型徹底改變了 NLP、電腦視覺、時間序列分析和許多其他領域的應用。然而，我們將研究範圍限制在時間序列應用上。該研究的另一個限制與 Transformer 模型架構的變化有關。近年來，為了提高 Transformers 的性能，已經提出了數百種變體。然而，我們專注於 Vaswani 等人提出的基本架構 [1]。

我們首先在第 2 節中概述基於 self-attention、scaled dot-product、multi-head 和 positional encoding 的 Transformer 架構。第 3 節描述了 Transformers 在時間序列應用中的進展。然後，我們在第 4 節中討論一些最流行的近期時間序列 Transformer 架構。最後，在總結論文之前，我們在第 5 節中提供了訓練 Transformers 的「最佳實踐」。

# 2 Transformers: Nuts and Bolts

我們首先解釋 Vaswani 等人於 2017 年提出的 Transformer 的內部運作，以解決神經機器翻譯的挑戰性問題 [1]。然後，我們深入探討 Transformers 各個組件內部執行的操作以及這些操作背後的直覺。Transformer 的幾種變體架構已經被開發出來。然而，所使用的基本操作背後的直覺保持不變，並在本節中介紹 [39, 40, 41]。

## 2.1 The Transformer Architecture

最初的 Transformer 是一個 sequence-to-sequence 模型，設計為 encoder-decoder 類型的配置，它將源語言的單詞序列作為輸入，然後生成目標語言的翻譯 [1]。鑑於兩個序列的長度和詞彙大小不一定相同，模型必須學習將源序列編碼為固定長度的表示，然後可以將其解碼以 auto-regressive 的方式生成目標序列 [42]。這種 auto-regressive 屬性帶來了一個約束，即在生成翻譯序列期間，需要將資訊傳播回序列的開頭。同樣的約束也適用於時間序列分析。

機器學習模型一直受到在學習過程中可以考慮特定資料樣本影響多遠的限制。在某些情況下，機器學習模型訓練的 auto-regressive 性質會導致對過去觀察的記憶，而不是將訓練範例推廣到新資料 [43, 44]。Transformers 透過使用 self-attention 和 positional encoding 技術來解決這些挑戰，以在分析序列中的當前資料樣本時共同關注和編碼有序資訊。這些技術在學習時保持序列資訊的完整性，同時消除了 recurrence 的傳統概念 [45]。這些技術進一步允許 Transformers 利用 GPUs 和 TPUs 提供的平行性。最近，有一些研究嘗試將 recurrent 組件納入 Transformers [46]。

**一個簡單的翻譯範例。** 考慮一個使用傳統機器翻譯模型（LSTM 或 GRU）和 Transformer 將「I like this cell phone」翻譯成德語「Ich mag dieses Handy」的例子，如圖 1 所示。輸入的單詞必須首先使用 embedding 層進行處理，以將原始單詞轉換為大小為 d 的向量。將離散單詞嵌入到實數連續空間中的概念是 NLP 中的常見做法 [47, 48, 49, 50]。在傳統語言翻譯模型中，句子中的每個嵌入單詞都對應一個特定的 RNN/LSTM/GRU cell。後續 cell 中的操作取決於前一個 cell 的輸出。因此，輸入中的每個嵌入單詞都是循序處理的（在處理前一個單詞之後）。在基於 Transformer 架構的模型中，整個輸入序列「I like this cell phone」同時被饋送到模型中，從而消除了循序資料處理的需要。序列順序是使用 positional encoding 來追蹤的。

## 2.2 Self-Attention Operation

Transformer 架構的基礎是使用 dot product [51] 尋找各種輸入片段（在這些片段中加入位置資訊後）之間的關聯或關係。令 {xᵢ}ⁿᵢ₌₁, xᵢ ∈ Rᵈ 為單一序列中 n 個單詞（或資料點）的集合。下標 i 表示向量 xᵢ 的位置，相當於原始句子或單詞序列中單詞的位置。self-attention 操作是這些輸入向量 xᵢ 與彼此的加權 dot product。

### 2.2.1 The Intuition Behind Self-Attention

我們可以將 self-attention 操作視為一個兩步驟的過程。第一步計算給定輸入序列中所有輸入向量對之間的 normalized dot product。正規化是使用 softmax 運算子執行的，它會縮放一組給定的數字，使得輸出數字的總和為一。計算輸入片段 xᵢ 與所有其他 j = 1, ..., n 之間的正規化相關性：

wij = softmax (xᵢᵀxⱼ) = exp(xᵢᵀxⱼ) / Σₖ exp(xᵢᵀxₖ) (1)

其中 Σⁿⱼ₌₁ wᵢⱼ = 1 且 1 ≤ i, j ≤ n。在第二步中，對於給定的輸入片段 xᵢ，我們找到一個新的表示 zᵢ，它是所有輸入片段 {xⱼ}ⁿⱼ₌₁ 的加權和：

zᵢ = Σⁿⱼ₌₁ wᵢⱼxⱼ, ∀ 1 ≤ i ≤ n. (2)

我們注意到，在方程式 2 中，對於任何輸入片段 xᵢ，權重 wᵢⱼ 的總和為 1。因此，產生的表示向量 zᵢ 將類似於具有最大 attention weight wᵢⱼ 的輸入向量 xⱼ。最大的 attention weight 反過來又是由 xᵢ 和 xⱼ 之間的 normalized dot product 測量的最大相關值產生的。

### 2.2.2 Linearly Weighting Input Using Query, Key, and Value

Transformers 中的 self-attention 操作始於從輸入 {xᵢ}ⁿᵢ₌₁ 中建構三個不同的線性加權向量，稱為 query q ∈ Rᵈ¹、key k ∈ Rᵈ¹ 和 value v ∈ Rˢ。直觀地，query 是一個問題，可以是一個單詞或一組單詞。一個例子是當一個人在網路上搜尋「network」這個詞以尋找更多關於它的資訊時。搜尋引擎將 query 對應到一組 keys，例如，neural、social、circuit、deep learning、communication、computer 和 protocol。values 是擁有關於「network」這個詞資訊的候選網站。

對於輸入 xᵢ，query qᵢ、key kᵢ 和 value vᵢ 向量可以透過以下方式找到：

qᵢ = Wᵩxᵢ, kᵢ = Wₖxᵢ, and vᵢ = Wᵥxᵢ, (3)

其中 Wᵩ 和 Wₖ ∈ Rˢ¹ˣᵈ，Wᵥ ∈ Rˢˣᵈ，表示可學習的權重矩陣。輸出向量 {zᵢ}ⁿᵢ₌₁ 由下式給出：

zᵢ = Σⱼ softmax (qᵢᵀkⱼ) vⱼ. (4)

我們注意到，value 向量 vⱼ 的權重取決於位置 i 的 query 向量 qᵢ 與位置 j 的 key 向量 kⱼ 之間的對應相關性。dot product 的值往往隨著 query 和 key 向量大小的增加而增長。由於 softmax 函數對較大的值很敏感，因此 attention weights 會按 query 和 key 向量大小 dᵩ 的平方根進行縮放，如下所示：

zᵢ = Σⱼ softmax (qᵢᵀkⱼ / √dᵩ) vⱼ. (5)

以矩陣形式，我們有：

Z = softmax (QKᵀ / √dₖ) V, (6)

其中 Q 和 K ∈ Rᵈ¹ˣⁿ，V ∈ Rˢˣⁿ，Z ∈ Rˢˣⁿ 且 ᵀ 表示轉置操作。

## 2.3 Multi-Head Self-Attention

輸入資料 X 可能包含多個層次的相關資訊，學習過程可能會從以不同方式處理輸入資料中受益。引入了多個 self-attention heads，它們在相同的輸入上平行操作，並使用不同的權重矩陣 Wq、Wk 和 Wv 來提取輸入資料之間不同層次的相關性。例如，考慮句子「Do we have to turn left from here or have we left the street behind?」。句子中有兩個「left」這個詞。每個出現都有不同的含義，因此與句子中其餘單詞的關係也不同。如圖 4 所示，Transformers 可以使用多個 heads 來捕捉此類資訊。每個 head 都是使用一組獨立的 query、key 和 value 權重矩陣建構的，並與其他 heads 平行地計算輸入序列上的 self-attention。在 Transformer 中使用多個 heads 類似於在卷積神經網路的每一層中使用多個核心，其中每個核心負責學習不同的特徵或表示 [53]。

以下三個步驟可以描述 multi-head self-attention 中涉及的操作。

### 2.3.1 步驟 1 - 產生多組不同的 Query、Key 和 Value 向量

假設我們總共有 r 個 heads，總共 r 組權重矩陣 {W⁽ˡ⁾ᵩ, W⁽ˡ⁾ₖ, W⁽ˡ⁾ᵥ}ʳˡ₌₁ 將為輸入 X 產生 r 組不同的 query、key 和 value 矩陣。圖 5 說明了具有三個向量 (n = 3) 且每個向量維度為六 (d = 6) 且 s₁ = s = 4 的輸入情況。這導致輸入矩陣 X ∈ R⁶ˣ³，W⁽ˡ⁾ᵩ, W⁽ˡ⁾ₖ, W⁽ˡ⁾ᵥ ∈ R⁴ˣ⁶ 且 Q⁽ˡ⁾, K⁽ˡ⁾, 和 V⁽ˡ⁾ ∈ R⁴ˣ³。

### 2.3.2 步驟 2 - 平行進行 Scaled Dot Product 操作

此步驟包括實現如圖 5 所示的以下關係：

Z = softmax (QKᵀ / √dₖ) V. (7)

### 2.3.3 步驟 3 - 串接並線性組合輸出

最後，我們將所有 r 個 heads 的輸出 Z⁽ˡ⁾ 串接起來，並使用一個可學習的權重矩陣 Wₒ ∈ Rᵈˣʳˢ 進行線性組合。輸出是一個矩陣 Z ∈ Rᵈˣⁿ。重要的是要注意，multi-head self-attention 的輸入和輸出具有相同的維度，即 dimension(X) = dimension(Z)。

## 2.4 使用 Encoders 和 Decoders 建構 Transformers

Transformer 架構通常由稱為 encoders 和 decoders 的兩種類型組件的多個實例組成。

### 2.4.1 The Encoder Block

如圖 6 所示，一個 encoder block 由一個 multi-head self-attention 層和一個 feed-forward 層組成，它們背靠背地連接，並帶有 residual connections 和 normalization layers。Residual connections 是訓練深度神經網路的常用技術，有助於訓練穩定和學習 [54]。layer normalization 操作也常用於處理序列資料的神經網路中。它有助於模型訓練的更快收斂 [55]。feed-forward 層包含兩個線性層和一個 ReLU 激活函數 [56]。一個 encoder block 的輸出用作下一個 encoder block 的輸入。第一個 encoder block 的輸入由 word embeddings 和 positional encoding (PE) 向量的總和組成。

### 2.4.2 The Decoder Block

每個 decoder block 都包含與 encoder block 相似的層和操作。然而，decoder 接收兩個輸入，一個來自前一個 decoder，另一個來自最後一個 encoder。在 decoder 內部，三層包括 (1) multi-head self-attention，(2) an encoder-decoder attention layer，以及 (3) a feed-forward layer。如圖 6 所示，有 residual connections 和 layer normalization 操作。在 encoder-decoder attention layer 內部，一組 key 和 value 向量是從最後一個 encoder 的輸出產生的。query 向量是從 encoder-decoder 層之前的 multi-head self-attention 層的輸出產生的。

### 2.4.3 Masking in Self-Attention

在 decoder block 內部，multi-head self-attention 層在訓練階段會遮蔽目標輸入的一部分。此操作可確保 self-attention 操作不涉及未來的資料點，即 decoder 預期預測的值。在訓練階段，模型的預測輸出不會回饋到 decoder 中。取而代之的是，使用 ground truth target (word embedding) 來輔助學習。在測試階段，序列中的預測單詞在通過 word embedding 層和加上 PE 後會回饋到 decoder，如圖 7 所示。

### 2.4.4 Stacking Encoders and Decoders

Transformer 模型可能包含多個 encoder 和 decoder block 的堆疊，具體取決於要解決的問題，如圖 7 所示 [57]。堆疊的 encoder/decoder block 類似於傳統神經網路中使用的多個隱藏層。然而，重要的是要注意，通常情況下，經過 encoder 或 decoder 處理後，表示維度不會減少。第一個 encoder block 的輸入是映射到帶有 PE 的 word embeddings 的單詞序列。

### 2.4.5 The Output

最後一個 encoder 的輸出與前一個 decoder 的輸入一起被饋送到每個 decoder 中。可選地，最後一個 decoder block 的輸出通過一個線性層以將輸出維度匹配到所需的大小，例如目標語言詞彙大小。然後，映射的輸出向量通過一個 softmax 層以找到輸出序列中下一個單詞的機率。根據所需的任務，可以選擇對最後一個 decoder block 的輸出進行操作以進行分類或回歸。

## 2.5 Positional Encoding (PE)

處理序列資料最重要的方面是納入序列的順序。self-attention 操作不包含有關序列中輸入資料順序的任何資訊。Transformers (1) 使用 positional encoding 的概念將位置資訊添加到輸入中，並且 (2) 平行處理修改後的輸入，從而避免了循序處理資料的挑戰。該技術包括計算 n 個 PE 向量（表示為 p ∈ Rᵈ）並將它們添加到輸入 {xᵢ}ⁿᵢ₌₁ 中。

### 2.5.1 Sinusoidal PE

在最初的工作中，作者提出了用於預先計算輸入資料集 [1] 的 PE 向量的正弦函數。PE 向量不包含任何可學習的參數，並且直接添加到 word embedding 輸入向量中。圖 8 顯示了使用方程式 8 和 9 的 PE 向量的公式，

PE(pos,2i) = pᵢ = sin(pos / 10000²ⁱ/ᵈ) (8)
PE(pos,2i+1) = pᵢ₊₁ = cos(pos / 10000²ⁱ/ᵈ) (9)

其中 pos 是輸入單詞序列中的位置（時間步），i 是沿著 embedding 向量維度的位置，範圍從 0 到 d/2 - 1，d 表示 embedding 向量的維度。方程式 8 和 9 中使用的數字 10,000 可能會因輸入序列的長度而異。

### 2.5.2 Sinusoidal PE 與二進位編碼的關係

我們可以將提出的用於 PE 的正弦函數與一個六位元長二進位數中的交替位元進行類比，如圖 9 所示 [1, 58]。在二進位格式中，最低有效位（以橙色顯示）以最高頻率交替。向左移動（靛藍色），位元在 0 和 1 之間振盪的頻率降低。同樣，在正弦 PE 中，當我們沿著 positional encoding 向量移動時，正弦函數的頻率會發生變化。

### 2.5.3 Positional Encoding and Rotation Matrix

使用正弦函數計算的 PE 向量允許模型學習單詞的相對位置而不是其絕對位置。例如，在句子「I am enjoying this tutorial」中，「this」這個詞的絕對位置是 4。然而，相對於「am」這個詞，「this」這個詞在位置 3。因此，對於任何固定的偏移量 k，PE 向量 pᵢ₊ₖ 可以表示為 pᵢ 的線性變換。考慮 pᵢ ∈ R² 並令 T = [x₁ y₁; x₂ y₂] 為一個線性變換，其中 a 表示輸入序列中單詞的絕對位置。我們可以寫成：

T [sin(fᵢa); cos(fᵢa)] = [sin(fᵢ(a+k)); cos(fᵢ(a+k))] (10)

[x₁ y₁; x₂ y₂] [sin(fᵢa); cos(fᵢa)] = [sin(fᵢ(a+k)); cos(fᵢ(a+k))] (11)

[x₁sin(fᵢa) + y₁cos(fᵢa); x₂sin(fᵢa) + y₂cos(fᵢa)] = [sin(fᵢa)cos(fᵢk) + sin(fᵢk)cos(fᵢa); cos(fᵢa)cos(fᵢk) - sin(fᵢa)sin(fᵢk)] (12)

比較方程式 12 的兩邊，我們得到 x₁ = cos(fᵢk)，y₁ = sin(fᵢk)，x₂ = -sin(fᵢk)，以及 y₂ = cos(fᵢk)。變換 T 現在可以寫成：

T = [cos(fᵢk) sin(fᵢk); -sin(fᵢk) cos(fᵢk)] (13)

我們注意到變換 T 是一個旋轉矩陣，並且取決於單詞的相對位置，而不是絕對位置。正弦 PE 函數適用於任何長度的輸入序列，無需指定。

### 2.5.4 將 Positional Encoding 與 Word Embeddings 相結合

PE 向量會加到輸入序列中每個單詞的 word embeddings 上。我們可能會認為，加法操作可能會導致來自兩個來源（即 PE 和 word embeddings）的某些資訊遺失。然而，情況可能並非如此，因為 PE 和 word embeddings 都編碼了不同類型的資訊。PE 向量包含有關輸入序列中單詞位置的資訊，而 word embeddings 則編碼有關單詞的語義和上下文資訊。這兩種編碼方案屬於不同的子空間，它們可能彼此正交。在這種情況下，兩個此類向量的相加可能不會導致資訊遺失。我們可以考慮與數位通訊中完成的頻率調變操作的類比——一個較低頻率的訊號駕馭在一個較高頻率的訊號之上，而兩者互不干擾。

現在我們已經單獨討論了 Transformer 架構中實現的每個操作，圖 10 描述了 Transformer 架構中內部操作的端到端流程，其中包含一個單一 encoder 和三個輸入單詞，以預測接下來的三個單詞。

# 3 Transformers for Time-Series Analysis 的路線圖

自 2017 年 Transformers 問世以來 [1]，這些網路在時間序列分析方面取得了許多進展。最初的模型主要專注於 NLP 任務，但現在該架構已擴展到分類 [59]、時間序列分析 [60]、語義分割 [61] 等領域。時間序列資料並非 Transformers 最初概念的一部分。許多研究人員已經客製化並改進了該架構在時間序列分析方面的性能。為了說明這項研究的發展歷程，我們將提供時間序列任務的路線圖以及該技術的演進。

這些改進的一個共同點是修改了輸入層以適應時間序列資料。使用 Transformers 進行時間序列分析、預測和分類可完成兩項主要任務。在每項任務中，我們都提供了有用的資訊和最先進方法中使用之資料集的連結。本節的路線圖全面剖析了過去幾年取得的進展及其相互關係。每個小節的結尾都列出了該類別中包含的模型（和引文）。

## 3.1 時間序列 Transformers 的改進途徑

本小節的每個部分都概述了改進 Transformer 架構中特定機制的主要貢獻和研究文章。圖 11 顯示了研究時間序列 Transformers 的示意圖路線圖。

### 3.1.1 學習類型 - Supervised, Self-supervised, or Unsupervised

主流的 Transformer 應用大多依賴於手動標記的訓練資料。標記資料可能需要相當長的時間，導致大量未標記的資料無法使用。Self-supervised 和 unsupervised 方法旨在透過讓 Transformers 學習、分類和預測未標記的資料來改進它們。例如，有大量的未標記衛星影像資料。在 [62] 中，一個名為 SITS-BERT 的模型從未標記的資料中學習，以對衛星影像進行區域分類。其他 self-supervised 或 unsupervised 學習模型包括 anomaly Transformer [63] 和 self-supervised Transformer for Time-Series (STraTS) [64]。

### 3.1.2 資料預處理

通常會執行預處理以準備要饋送到機器學習模型的資料。已知預處理操作會影響機器學習模型的性能。一些研究人員會對輸入資料特徵添加雜訊 [62]、執行遮蔽 [65] 或變數選擇 [66]。遮蔽操作會移除輸入中的特徵，從而透過讓模型更擅長預測缺失特徵來提高性能。在訓練時向輸入添加雜訊也適用類似的概念，只是特徵變得相對更嘈雜，但並未完全被取代。這兩種操作都可以提高模型的穩健性，使其在訓練或測試資料中存在雜訊的情況下仍能獲得良好的準確性。

### 3.1.3 Positional Encoding (PEs)

最近的一些工作著重於改進 [1] 中提出的原始 PE。Transformers 使用 PE 將序列/時間資訊嵌入到模型輸入中，以便 Transformers 可以一次處理所有序列資料，而不是像 RNN、LSTM 或 GRU 那樣一次處理一個。將時間（秒、分鐘、小時、週、年等）嵌入到輸入中，可以讓模型更好地分析時間序列資料，並利用現代硬體（包括 GPU、TPU 等）提供的計算優勢。最近，已經提出了 timestamp embedding [65]、temporal encoding [67] 和其他方法來創建可以更有效訓練的 Transformers [66]。

### 3.1.4 Gating Operation

gating 操作將標準 Transformer 模型的兩個 encoder tower 的輸出合併為一組單一的輸出預測 [68]。此操作結合並選擇來自 encoder 或 decoder block 的多個輸出。Gating 也受益於在適當時應用非線性資料處理。在各種方式中應用 gating 是 Transformers 未來創新的可能途徑，不僅適用於時間序列資料，也適用於任何其他類型的資料。有幾種架構使用 gating，包括 Gated Transformer Networks (GTN) [68] 和 Temporal Fusion Transformers (TFT) [66]。GTN 的提議架構使用 gating 技術將來自兩個 Transformer tower 的資訊合併到輸出中。它透過串接每個 tower 的輸出來實現。TFT 提出了 Gated Linear Units (GLUs) [69]，以允許根據資料集強調或抑制網路的不同部分。

### 3.1.5 Attention

本節討論的模型透過修改和改進模型的 attention 機制，改進了用於時間序列資料的 Transformers。

Tightly-Coupled Convolutional Transformer (TCCT) [70] 提出了三種改進 attention 的架構。第一種稱為 Cross Stage Partial Attention (CSPAttention)。這種方法將 Cross Stage Partial Network (CSPNet) [71] 與 self-attention 機制相結合，以減少所需資源。CSPNet 透過將輸入端的 feature map 納入輸出階段來減少計算量。CSPAttention 將此概念僅應用於 attention 層，大大降低了時間複雜度和所需記憶體。第二種方法改變了 self-attention distilling 操作以及 self-attention block 的連接方式。他們使用 dilated causal convolution 而不是 canonical convolution。Dilated causal convolution 增強了局部性，並允許 Transformer 實現指數級的感受野增長。第三種架構是一種 pass-through 機制，允許堆疊多個 self-attention block。這種堆疊是透過在 self-attention 機制內串接來自多個尺度的 feature maps 來完成的。這種機制提高了 Transformer 捕捉精細尺度特徵的能力。

更多改進 attention 的方法包括 Non-Autoregressive Spatial-Temporal Transformer (NAST) [72] 和 Informer [73]。NAST 中提出的架構被稱為 Spatial-Temporal Attention Block。這種方法結合了 spatial attention block（在空間上進行預測）和 temporal attention block（在時間上進行預測）。結果改善了在空間和時間域中的學習。Informer 架構用 ProbSparse self-attention 機制取代了 canonical self-attention，這在時間複雜度和記憶體使用上更有效率。這個 Transformer 還使用了一個 self-attention distilling 函數來降低空間複雜度。這兩種機制使得 Informer 在處理極大的輸入資料序列時非常有效率。

最後，我們將討論三種改進 attention 的架構：LogSparse Transformers [74]、TFT [66] 和 YFormer [75]。LogSparse Transformers 引入了 convolutional self-attention blocks，其中包括在 attention 機制之前的一個卷積層，用於創建 queries 和 keys。TFT 中使用了一個 temporal self-attention decoder 來學習資料中的長期依賴性。YFormer 架構提出了一種稀疏 attention 機制，並結合了一個 downsampling decoder。這種架構允許模型更好地檢測資料中的長程效應。

### 3.1.6 Convolution

原始的 Transformer 架構不使用卷積層。然而，這並不意味著 Transformers 不會從卷積層的加入中受益。事實上，許多為時間序列資料設計的 Transformers 都從加入卷積層或將卷積納入現有機制中受益。大多數方法在 Transformer 內部 attention 機制之前或與之並行地納入卷積。

透過卷積改進 Transformers 的方法包括 TCCT [70]、LogSparse Transformers [74]、TabAConvBERT [65] 和 Traffic Transformers [67]。TCCT 使用了 Informer [73] 架構中稱為 dilated causal convolution 的機制，取代了 canonical convolutional layers。LogSparse Transformers 也使用了 causal convolutional layers。如前一節所述，此層為 self-attention 層生成 queries 和 keys，稱為 convolutional self-attention。TabAConvBERT 採用一維卷積，考慮到它對時間序列資料自然有效。Traffic Transformers [67] 將圖神經網路的概念納入卷積層，以產生 Graph Convolutional Filters。

### 3.1.7 Interpretability/Explainability

與 CNNs 或 LSTMs 相比，Transformers 是一類相對較新的機器學習模型。為了可靠和值得信賴地使用它們，我們必須了解這些模型的黑盒子性質並解釋它們的決策。黑盒子現象是人工智慧中一個普遍存在的問題。它指的是只能觀察到學習模型的輸入和輸出的情況。模型的參數如何相互作用以得出最終輸出並不精確地為人所知。

許多解釋和說明模型預測的方法都是事後的，也就是說，解釋是在事後做出的。事後方法幾乎適用於任何模型。這些方法中有許多提供了視覺上吸引人的結果，但可能無法準確解釋模型內部發生的情況 [76]。一種可能的方法是將解釋和可解釋性納入模型本身，而不是在事後進行近似。現在存在多個本身即可解釋的時間序列 Transformers [64, 68, 66]。這些模型可以產生解釋，從而可以更好地解釋結果並獲得更大的用戶信任。

### 3.1.8 Dense Interpolation

Transformer 模型通常由 encoder 和 decoder blocks 組成，後面跟著用於決策的 linear 和 softmax 層。一種稱為 Simply Attend and Diagnose (SAnD) 的方法用 dense interpolation 層取代了 decoder block，以將時間順序納入模型的處理中 [60]。這種方法不使用輸出嵌入，從而減少了模型中的總層數。沒有 decoder，模型需要一種方法來處理 encoder block 的輸出，以便輸入到 linear 層。簡單的串接會導致預測準確性不佳。因此，[60] 中的工作開發了一種具有超參數的 dense interpolation 演算法，可以對其進行調整以提高性能。

## 3.2 Architectural Modifications

### 3.2.1 BERT-Inspired

一個建立在原始 Transformer 論文基礎上的著名架構是 Bidirectional Encoder Representations from Transformers (BERT) [77]。該模型是透過堆疊 Transformer encoder blocks 並引入一種新的訓練方案來建構的。encoder block 是獨立於任務進行預訓練的。decoder block 可以在稍後添加，並針對手頭的任務進行微調。這種方案允許在大量未標記資料上訓練 BERT 模型。

BERT 架構啟發了許多用於時間序列資料的新 Transformer 模型 [62, 65, 78, 63]。與 NLP 任務相比，為時間序列資料創建 BERT 風格的模型存在一些挑戰。語言資料是一種標準化類型的資料，可用於各種任務，包括翻譯、文本摘要、問答、情感分析等。所有這些任務都可以使用相同的資料進行預訓練。然而，對於時間序列任務來說，情況並非如此。時間序列資料的例子包括用電量 [79]、環境溫度 [73]、交通流量 [79]、衛星影像 [62]、各種形式的醫療保健資料 [80] 等等。由於資料類型如此多樣，每個任務的預訓練過程都必須不同。這種依賴於任務的預訓練與 NLP 任務形成對比，後者可以從相同的預訓練模型開始，假設所有任務都基於相同的語言語義和結構。

### 3.2.2 GAN-Inspired

Generative adversarial networks (GANs) 由兩個深度神經網路組成：generator 和 discriminator。兩個網路相互對抗學習。GANs 通常用於影像處理以生成逼真的影像。generator 的任務是創建能夠欺騙 discriminator 的影像。discriminator 會被給予真實和生成（假）的影像，並且必須預測輸入影像是真是假。訓練有素的 GAN 可以生成對人類來說看起來非常逼真的影像。

同樣的 generator-discriminator 學習原則也已應用於時間序列預測任務 [81]。作者使用 Transformer 作為 generator 和 discriminator，並訓練模型進行準確的預測。generator 的任務是創建一個 discriminator 會將其分類為真或假的預測。隨著訓練的繼續，generator 網路將創建更逼真的資料。在訓練結束時，模型在進行預測時將具有很高的準確性。

## 3.3 Time-Series Tasks

在時間序列資料上執行的兩個主要任務是預測和分類。預測旨在從給定的時間序列資料中預測實數值，稱為回歸。最近的文獻中已經開發了許多用於時間序列資料的預測 Transformers [70, 72, 73, 74, 82, 66, 67, 81, 78, 64, 75]。分類任務涉及將給定的時間序列資料分類為一個或多個目標類別。最近在時間序列 Transformers 的分類任務方面取得了許多進展 [65, 63, 62, 60, 68]。本教學中討論的所有基於時間序列的 Transformer 模型都專注於這兩個任務之一。其中一些模型在對最後一層和損失函數進行少量修改後可以完成這兩項任務。

# 4 Time-Series Analysis - Architectures and Use Cases

## 4.1 The Informer Architecture

最近，Zhou 等人提出了 Informer，它使用 ProbSparse self-attention 機制來優化標準 Transformer 架構的計算複雜度和記憶體使用量 [73]。作者還引入了 self-attention distilling 操作，大大降低了模型的總空間複雜度。

ProbSparse self-attention 透過隨機選擇 log L 個頂級 query 向量並將其餘 query 向量值設為零來使用主要的 dot-product 對。計算複雜度和記憶體使用量隨著 L 值向量減少到 O(Llog L)。對於 J 個 encoder 的堆疊，總記憶體使用量減少到 O(JL log L)。此外，self-attention distilling 操作移除了 value 向量的冗餘組合。此操作的靈感來自於 [83] 和 [84] 中提出的 dilated convolution。multi-head self-attention 的輸出被饋送到核心大小等於 3 的 1-D 卷積濾波器中。之後，應用 exponential linear unit (ELU) 激活函數，然後是步長為 2 的 max-pooling 操作。這些操作將大小減半，從而形成如圖 12 所示的金字塔。有效地，總空間複雜度大大降低。還建構了堆疊的複本，其輸入長度為前一個堆疊的一半。圖 12 僅顯示一個複本堆疊。主堆疊和複本堆疊的輸出具有相同的維度，並被串接以形成 encoder 的最終輸出。這些複本堆疊增強了 distilling 操作的穩健性。

decoder 由兩個堆疊的 multi-head attention 層組成。decoder 的輸入是一個 start token，與預測目標序列的佔位符（初始值設為零）串接。它透過一次前向過程預測所有輸出（如圖 12 所示），從而大大減少了推論時間。

在 Informer 架構中，純量輸入使用 1-D 卷積濾波器映射到 d 維向量 uᵢ。使用基於正弦函數的固定 positional encoding (PE) 保留局部上下文。還包括一個稱為 stamp embedding (SE) 的全域時間戳，以捕獲分層時間資訊，例如週、月或年，以及偶爾發生的事件，例如假日。encoder 的輸入是純量投影向量 (uᵢ)、PE 和 SE 的總和。

Informer 架構在各種資料集上進行了測試，包括電力消耗負載和天氣資料集。該模型的性能優於最先進的 (SOTA) 方法，包括 Autoregressive Integrated Moving Average (ARIMA) [85]、Prophet [86]、LSTMa [87]、LSTnet [88] 和 DeepAR [89]。

## 4.2 LogSparse Transformer Architecture

Li 等人提出了 LogSparse Transformers 來克服記憶體挑戰，從而使 Transformers 對於具有長期依賴性的時間序列資料更具可行性 [74]。LogSparse Transformers 允許每個時間步長關注使用指數步長選擇的先前時間步長。這將每個 self-attention 層的記憶體利用率從 O(L²) 降低到 O(L log₂ L)。圖 13 顯示了 LogSparse self-attention 可用於時間序列分析的各種方式。用於特定範圍內相鄰時間步長的 canonical self-attention 機制允許收集更多資訊。超出該範圍，則應用 LogSparse self-attention 機制。另一種方法是在特定時間步長範圍後重新啟動 LogSparse 步長。

由於不同的事件，例如假日或極端天氣，時間序列資料中的模式可能會隨著時間發生顯著變化。因此，捕獲周圍時間點的資訊以確定觀察到的點是異常、變化點還是模式的一部分可能是有益的。這種行為是使用 causal convolutional self-attention 機制捕獲的，如圖 14 所示。

causal convolutional self-attention 機制確保當前位置無法存取未來資訊。核心大小大於一的卷積操作捕獲了從輸入生成的 query 和 key 向量中的局部上下文資訊。Value 向量是使用等於一的核心大小生成的。透過這種機制，可以執行更準確的預測。

## 4.3 Simply Attend and Diagnose (SAnD)

臨床資料，例如加護病房 (ICU) 測量，包括來自感測器測量、測試結果和主觀評估的多變量時間序列觀察。Song 等人引入了 Transformers 來預測 MIMIC-III 基準資料集 [80] 中的各種臨床相關變數，命名為 Simply Attend and Diagnose (SAnD) [60]。預處理的資料被傳遞到一個輸入嵌入層，該層使用 1-D 卷積將輸入映射到 d 維向量（d > 多變量時間序列資料中的變數數量）。在將硬編碼的 positional encoding 添加到嵌入資料後，它會通過 attention 模組。multi-head attention 層使用受限的 self-attention 來引入因果關係，即使用比當前時間更早的資訊執行計算。堆疊的 attention 模組的輸出傳遞到「dense interpolation layer」。該層與 positional encoding 一起用於捕獲臨床資料的時間結構。線性層和 softmax/sigmoid 用作分類的最後幾層。所提出的 Transformer 模型在所有 MIMIC-III 基準任務上進行了評估，據報導其性能優於 RNNs。

## 4.4 Traffic Transformer

交通預測旨在預測未來交通，給定一系列歷史交通觀測，如道路網路上感測器檢測到的速度、密度和流量。在這種情況下，使用序列中的 M 個先前時間步長來預測 H 個未來時間步長。編碼時間序列資料的連續性和週期性並捕獲交通預測中的時空依賴性非常重要。Traffic Transformer [67] 是基於兩個現有網路建構的，第一個是 Graph Neural Network [90, 91]，後面跟著 Transformer，如圖 15 所示。Transformer 建模時間依賴性，而 Graph Neural Network 用於建模空間依賴性。

Graph Neural Network 的輸出在納入 positional encoding 後構成 Transformer 的輸入。有兩種方法可以添加序列資訊：(1) positional encoding 向量被添加到輸入向量中，以及 (2) positional encoding 向量的 attention weights 是使用 dot product 計算的。

使用 positional attention weights 來調整 Transformer 輸入向量的 attention weights。介紹了以下四種策略來編碼交通資料的時間資訊。這些策略的不同組合可用於各種問題或資料集類型。

a) 時間序列資料的連續性：
    i. 相對位置編碼：此策略編碼源-目標序列視窗中時間步長的相對位置，而不管該時間步長在整個時間序列中的位置。因此，相同的時間步長可能會根據其在序列對中的位置而被分配不同的位置嵌入。位置編碼是使用正弦和餘弦函數完成的。
    ii. 全域位置編碼：整個序列使用正弦和餘弦函數進行編碼。這樣，時間序列資料的局部和全域位置都被捕獲。

b) 時間序列資料的週期性：
    i. 週期性位置編碼：引入了一種編碼資料每日和每週週期性的機制。每日週期性是透過編碼兩百八十八個 (24×60/5) 位置來捕獲的。每週週期性是透過編碼七個位置（一週中的天數）來捕獲的。
    ii. 時間序列片段：這是透過將每日和每週資料片段串接到「M」個最近的時間步長來完成的。

Traffic Transformer 在兩個真實世界的基準資料集上進行了測試，即 METR-LA [92] 和加州運輸局性能測量系統 ([PeMS])。

## 4.5 Self-Attention for Raw Optical Satellite Time-Series Classification

植被生命週期事件可作為識別植被類型的獨特時間訊號。這些時間訊號包含用於區分各類植被的相關特徵。使用 Transformers 從原始光學衛星影像中對植被類型 [93] 進行分類。與 LSTM-RNN [10]、MS-ResNet [94]、DuPLO [95]、TempCNN [96] 和隨機森林方法相比，Transformer 架構在原始資料上的植被分類性能更好。然而，對於預處理資料，所有方法的性能都非常相似。

## 4.6 Deep Transformer Models for Time-Series Forecasting: The Influenza Prevalence Case

Transformers 也被用於使用特定地區每週流感病例計數的資料來預測流感病例 [82]。在第一個實驗中，使用前十週的資料作為輸入來進行單一週的預測。與 ARIMA、LSTMs 和具有 attention 的 sequence-to-sequence 模型相比，Transformer 架構在「root mean square error」(RMSE) 方面表現更好。在第二個實驗中，Transformer 架構使用多變量時間序列資料進行了測試，方法是引入「週數」作為時間索引特徵，並將時間序列資料的一階和二階差分作為兩個明確的數值特徵。然而，這並未顯示出結果的顯著改善。在第三個實驗中，引入了 Time-delay embedding (TDE)，並透過將每個純量輸入 xt 嵌入到 d 維時間延遲空間中形成，如方程式 14 所示。

TDEd,τx(t) = (xt, xt-τ, ..., xt-(d-1)τ) (14)

形成不同維度 d（從 2 到 32）的時間延遲嵌入，其中 τ = 1。在所有實驗中，RMSE 值在維度為 8 時達到最小值，這與一項關於臨床資料的獨立研究的類似結果一致 [97]。

# 5 Best Practices for Training Time-Series Transformers

Transformer 架構正變得越來越流行，並引導許多研究人員尋求優化這些網路以用於各種應用的方法。這些方法包括調整架構以用於特定問題領域、調整訓練技術、超參數優化、推論方法、硬體適應等。在本節中，我們討論了訓練 Transformers 進行時間序列分析時的最佳實踐。

## 5.1 Training Transformers

對於初學者來說，從頭開始訓練 Transformers 可能並不容易。原始的 Transformer 架構 [1] 利用了許多不同的策略來在更深層的網路訓練期間穩定梯度。使用 residual connections 可以訓練更深層的網路。後來，在自適應優化器 (Adam) 旁邊添加了 layer normalization 操作，以便為不同的參數提供不同的學習率。像大多數其他深度學習模型一樣，Transformers 對學習率也很敏感。在給定最佳學習率的情況下，Transformers 的收斂速度可以比傳統序列模型更快。在最初的幾個 epochs 中，觀察到性能下降是很常見的。然而，幾個 epochs 後，模型通常會開始收斂到更好的值。在原始實現中，作者使用了一種 warm-up learning rate 策略，該策略在前 N 個訓練步驟中線性增加，然後與步數的倒數平方根成比例地減少，√N。

## 5.2 在流行框架中實現 Transformers

在這裡，我們概述了用於實現和訓練 Transformer 模型的流行框架。這些框架提供了用戶友好的介面和對自訂模型架構的支援，使研究人員和開發人員能夠試驗和調整 Transformers 以進行時間序列分析和其他任務。

Hugging Face Transformers 是一個開源函式庫，為各種自然語言處理任務提供預訓練的 Transformer 模型和用戶友好的 API。該函式庫與 PyTorch 和 TensorFlow 相容，提供了大量預訓練模型和社群貢獻的模型，使其成為研究人員和開發人員的絕佳起點。有關該函式庫的更多詳細資訊，請參閱 Hugging Face Transformers GitHub 儲存庫。

PyTorch Lightning 是 PyTorch 的一個輕量級包裝器，旨在簡化深度學習研究和開發。它提供了一種結構化的方法來訓練深度學習模型，包括 Transformer 模型，只需最少的樣板程式碼。PyTorch Lightning 與 Hugging Face Transformers 函式庫相容，使用戶能夠利用預訓練模型並針對特定任務進行微調。更多資訊可在 PyTorch Lightning GitHub 儲存庫中找到。PyTorch 中用於高效大型 Transformer 模型訓練的其他函式庫包括 Microsoft DeepSpeed 和 MosaicML Composer。

TensorFlow 是由 Google 開發的開源機器學習函式庫，支援建構和訓練深度學習模型，包括 Transformers。該函式庫提供了原始 Transformer 架構和各種預訓練模型的實現，可以針對特定任務進行微調。TensorFlow 還為實現自訂 Transformer 模型提供了廣泛的文件和社群支援。TensorFlow 網站提供了更多資訊。

## 5.3 Transformer 架構的改進以實現更好的訓練

為了解決與在更深層架構下實現穩定訓練相關的一些問題，已經提出了對 Transformer 架構的許多改進。主要是透過重新定位 layer normalization 操作和尋找更好的權重初始化技術來平衡 residual dependencies。這些改進帶來了更穩定的訓練，並且在某些情況下，消除了使用原始架構中提出的某些策略的需要。圖 16 概述了一些訓練 Transformers 的最佳實踐及其各自的原因。

原始的 Transformer 架構可以稱為 post-layer normalization (post-LN)，其中 layer normalization 位於 residual block [98] 之外。Post-LN 的收斂速度要慢得多，並且需要一個 learning rate warm-up 策略 [98]。Xiong 等人提出了一種 pre-layer normalization (pre-LN) Transformer 來解決這個問題，表明它可以幫助梯度更快地收斂，同時不需要 warm-up。Pre-LN Transformer 可以透過控制梯度大小和平衡 residual dependencies [98] 來實現這一點。雖然 pre-LN Transformer 架構不需要 learning rate warm-up，但與 post-LN Transformer 架構相比，它的經驗性能較差。

Liu 等人提出了一種 adaptive model initialization (Admin)，以從 post-LN Transformer 的易於訓練中受益，同時實現 pre-LN Transformer 的性能 [99]。Adaptive model initialization 是機器學習其他領域中使用的一種技術，用於初始化模型，以便更好地捕捉輸入和輸出變數之間的依賴關係 [99]。這種初始化技術有助於模型更有效地學習變數之間的關係並提高性能。另一種選擇是完全移除 layer normalization。ReZero 方法用一個可訓練的參數 α 取代了 layer normalization 操作，該參數在每個 residual layer [100] 中初始化為 0。因此，整個網路被初始化為計算恆等函數，並逐步和自適應地引入 self-attention 和 MLP 層的貢獻。對於訓練具有更好泛化性能的更深層 Transformer 模型，residual dependencies 似乎與 Admin 或 ReZero 平衡得很好。

## 5.4 訓練 Transformers 的實際問題

### 5.4.1 Large Model Size

選擇 Transformer 模型架構後，挑戰就變成了處理大型模型。看起來較小的模型會比較大的模型訓練得更快。然而，情況並非總是如此。例如，Li 等人 [101] 證明，在某些情況下，訓練大型模型然後壓縮結果會帶來更好的性能。lottery ticket hypothesis [102] 揭示了可以應用剪枝來減小模型大小。此外，量化到較低精度可以實現較小的模型。然而，在使用剪枝/量化後的大型模型與較小的模型之間存在不可避免的權衡。最相關的是確保訓練資料集足夠大以避免過擬合。使用小型資料集可能導致泛化能力差。一般來說，在嘗試訓練 Transformer 模型時，我們建議使用大型模型，而不是從較小的模型開始然後添加層的更傳統方法。

### 5.4.2 使用小型資料集進行訓練

我們已經看到了一些解決方案，與 vanilla Transformer 架構相比，可以訓練更深層的 Transformer 模型並提高性能。從頭開始訓練這些深層 Transformer 模型需要大型資料集，這使得使用小型資料集進行訓練具有挑戰性。小型資料集需要預訓練模型和小型 batch size 才能表現良好。然而，這兩個要求使得訓練額外的 Transformer 層更加困難。使用小型 batch size 會導致更新的變異數更大。即使 batch size 很大，模型通常也可能泛化不佳。在某些情況下，更好的初始化技術可以優化模型，使其在較小的資料集上表現良好。Xavier 初始化是 Transformers 最常用的方案 [103]。最近，T-Fixup 已被證明優於 Xavier [104]。由 Huang 等人引入的 T-Fixup 方案的動機如下。研究發現，在沒有 learning rate warm-up 的情況下訓練 Transformer 時，Adam 優化器中的變異數會被較大的初始學習率放大，從而在訓練開始時導致較大的更新 [104]。因此，learning rate warm-up 的要求來自 Adam 優化器的不穩定性以及透過 layer normalization 導致的梯度消失。為了解決這個問題，提出了 T-Fixup 方法，其中一種新的權重初始化方案提供了理論保證，可以保持模型更新有界，並移除了 warm-up 和 layer normalization [104]。在 T-Fixup 的工作之後，開發了一種依賴於資料的初始化技術（稱為 DT-Fixup）[105]。DT-Fixup 允許在給定正確的優化程序的情況下，使用小型資料集訓練更深層的 Transformer 模型。

### 5.4.3 其他需要考慮的策略

**Batch Size.** Popel 等人發現，最佳 batch size 取決於模型的複雜度 [106]。他們考慮了兩類模型，一類是「base」，另一類是「big」。對於 base 模型，較大的 batch size（高達 4,500）表現良好，而對於 big 模型，一組不同的參數產生了更好的結果。big 模型在開始收斂之前需要一個最小的 batch size（在他們的實驗中為 1,450）。雖然 batch size 幾乎完全是根據經驗選擇的，但對於 Transformer 模型，應使用一個較大的最小值。

**Learning Rate.** 在評估學習率對模型性能的影響 [107, 108, 109] 以及它們如何相互關聯方面，已經進行了大量研究。Popel 等人進行的一項研究表明，較小的學習率往往收斂較慢，而過高的學習率可能導致不收斂 [106]。

**Gradient Clipping.** 許多 Transformers 的實現在訓練期間使用 gradient clipping。如果步數過大，此過程往往可以避免梯度爆炸和潛在的發散。不建議對 Transformers 使用其他 gradient clipping 方法，例如選擇與 batch size 成比例的步長，因為這些方法可能導致收斂速度變慢 [106]。

# 6 Conclusion and Future Trends

總之，Transformer 架構已被證明是解決時間序列任務的強大工具，為 RNNs、LSTMs 和 GRUs 提供了有效的替代方案，同時也克服了它們的局限性。為了有效地處理時間序列資料，已經對原始的 Transformer 架構進行了修改。已經開發了各種訓練 Transformers 的最佳實踐，並且有許多開源框架可用於有效地訓練大型 Transformer 模型。穩健性 [110]、故障檢測 [111] 和多模態學習 [7] 是深度學習未來的一些趨勢。展望未來，使用時間序列預測中的不確定性估計來開發穩健、自我意識的 Transformer 架構是研究界目前正在追求的一個開放挑戰。具有多種模態（例如影像、影片和文本）的大型資料集的可用性，與時間序列資料相結合，可以導致開發基於 Transformer 的基礎模型，能夠學習難以察覺和微妙的特徵，這可能導致時間序列任務的前所未有的發現。

# 7 Acknowledgement

這項工作部分由美國國家科學基金會獎項 ECCS-1903466、OAC-2008690 和 OAC-2234836 支持。

# 8 Data Availability Statement

該手稿沒有相關資料。

# 9 Funding and/or Conflicts of interests/Competing interests

沒有利益衝突或競爭利益。

# References

[1] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in Neural Information Processing Systems, 30, 2017.
[2] Tianyang Lin, Yuxin Wang, Xiangyang Liu, and Xipeng Qiu. A survey of transformers. AI Open, 3:111–132, 2022.
[3] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020. Preprint at https://arxiv.org/abs/2010.11929.
[4] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. arXiv preprint arXiv:2304.02643, 2023. Preprint at https://arxiv.org/abs/2304.02643.
[5] Jiasen Lu, Christopher Clark, Rowan Zellers, Roozbeh Mottaghi, and Aniruddha Kembhavi. Unified-IO: A Unified Model for Vision, Language, and Multi-Modal Tasks. arXiv preprint arXiv:2206.08916, 2022. Preprint at https://arxiv.org/abs/2206.08916.
[6] Lili Chen, Kevin Lu, Aravind Rajeswaran, Kimin Lee, Aditya Grover, Misha Laskin, Pieter Abbeel, Aravind Srinivas, and Igor Mordatch. Decision transformer: Reinforcement learning via sequence modeling. Advances in Neural Information Processing Systems, 34:15084–15097, 2021.
[7] Asim Waqas, Aakash Tripathi, Ravi P Ramachandran, Paul Stewart, and Ghulam Rasool. Multimodal Data Integration for Oncology in the Era of Deep Neural Networks: A Review. arXiv preprint arXiv:2303.06471, 2023. Preprint at https://arxiv.org/abs/2303.06471.
[8] Inkit Padhi, Yair Schiff, Igor Melnyk, Mattia Rigotti, Youssef Mroueh, Pierre Dognin, Jerret Ross, Ravi Nair, and Erik Altman. Tabular transformers for modeling multivariate time series. In 2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 3565–3569, Toronto, 2021. IEEE. https://doi.org/10.1109/ICASSP39728.2021.9414142.
... (references continue) ...
