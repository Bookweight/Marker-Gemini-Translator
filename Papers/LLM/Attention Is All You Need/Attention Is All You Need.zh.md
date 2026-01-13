---
title: "Attention Is All You Need"
field: "LLM"
status: "Imported"
created_date: 2026-01-13
pdf_link: "[[Attention Is All You Need.pdf]]"
tags: [paper, LLM]
---

# Attention Is All You Need

Ashish Vaswani*
Google Brain
avaswani@google.com
Noam Shazeer*
Google Brain
noam@google.com
Niki Parmar*
Google Research
nikip@google.com
Llion Jones*
Google Research
llion@google.com
Aidan N. Gomez* †
University of Toronto
aidan@cs.toronto.edu
Illia Polosukhin* ‡
illia.polosukhin@gmail.com
Jakob Uszkoreit*
Google Research
usz@google.com
Łukasz Kaiser*
Google Brain
lukaszkaiser@google.com

*貢獻相等。排名隨機。Jakob 提出以自我注意力機制取代 RNN，並著手評估此想法。Ashish 與 Illia 設計並實現了第一批 Transformer 模型，並在整個研究中扮演關鍵角色。Noam 提出了縮放點積注意力、多頭注意力機制及無參數的位置表示，並成為幾乎所有細節的另一位參與者。Niki 在我們原有的程式碼庫與 tensor2tensor 中，設計、實現、調整並評估了無數個模型變體。Llion 也嘗試了新穎的模型變體，並負責我們最初的程式碼庫、高效的推論及視覺化。Lukasz 與 Aidan 花了無數個漫長的日夜設計並實現 tensor2tensor 的各個部分，取代了我們先前的程式碼庫，大幅改善了結果並極大地加速了我們的研究。
†於 Google Brain 工作期間完成。
‡於 Google Research 工作期間完成。

## Abstract
## 摘要

The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, including ensembles, by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature. We show that the Transformer generalizes well to other tasks by applying it successfully to English constituency parsing both with large and limited training data.

目前主流的序列轉換模型，是基於包含編碼器與解碼器的複雜循環神經網路或卷積神經網路。表現最好的模型，還會透過注意力機制來連接編碼器與解碼器。我們提出一種全新的簡單網路架構——Transformer，它完全基於注意力機制，完全摒棄了循環與卷積。在兩項機器翻譯任務上的實驗表明，這些模型在品質上更為優越，同時更具平行化能力，且訓練時間顯著縮短。我們的模型在 WMT 2014 英德翻譯任務上，達到了 28.4 的 BLEU 分數，超越了包括整合模型在內的現有最佳結果，提升超過 2 BLEU。在 WMT 2014 英法翻譯任務上，我們的模型在八個 GPU 上訓練 3.5 天後，創下了單一模型最新的 BLEU 分數 41.8，這只是文獻中最佳模型訓練成本的一小部分。我們證明了 Transformer 能很好地泛化到其他任務，成功地將其應用於具有大量和有限訓練數據的英語成分句法分析。

## 1 Introduction
## 1 簡介

Recurrent neural networks, long short-term memory [13] and gated recurrent [7] neural networks in particular, have been firmly established as state of the art approaches in sequence modeling and transduction problems such as language modeling and machine translation [35, 2, 5]. Numerous efforts have since continued to push the boundaries of recurrent language models and encoder-decoder architectures [38, 24, 15].

循環神經網路，特別是長短期記憶 [13] 和閘控循環 [7] 神經網路，已經穩固地成為序列建模和轉換問題（如語言建模和機器翻譯 [35, 2, 5]）的最新技術方法。此後，無數的努力持續推動著循環語言模型和編碼器-解碼器架構的界限 [38, 24, 15]。

Recurrent models typically factor computation along the symbol positions of the input and output sequences. Aligning the positions to steps in computation time, they generate a sequence of hidden states ht, as a function of the previous hidden state ht−1 and the input for position t. This inherently sequential nature precludes parallelization within training examples, which becomes critical at longer sequence lengths, as memory constraints limit batching across examples. Recent work has achieved significant improvements in computational efficiency through factorization tricks [21] and conditional computation [32], while also improving model performance in case of the latter. The fundamental constraint of sequential computation, however, remains.

遞歸模型通常沿著輸入和輸出序列的符號位置來分解計算。通過將位置與計算時間中的步驟對齊，它們生成一個隱藏狀態序列 ht，作為前一個隱藏狀態 ht−1 和位置 t 輸入的函數。這種固有的順序性排除了訓練樣本內的並行化，這在序列長度較長時變得至關重要，因為內存限制了跨樣本的批處理。最近的工作通過分解技巧 [21] 和條件計算 [32] 在計算效率上取得了顯著的改進，同時在後者的情況下也提高了模型性能。然而，順序計算的基本限制仍然存在。

Attention mechanisms have become an integral part of compelling sequence modeling and transduction models in various tasks, allowing modeling of dependencies without regard to their distance in the input or output sequences [2, 19]. In all but a few cases [27], however, such attention mechanisms are used in conjunction with a recurrent network.

注意力機制已成為各種任務中引人注目的序列建模和傳導模型不可或缺的一部分，它允許對依賴關係進行建模，而無需考慮其在輸入或輸出序列中的距離 [2, 19]。然而，除了少數情況 [27]，這種注意力機制通常與循環網路結合使用。

In this work we propose the Transformer, a model architecture eschewing recurrence and instead relying entirely on an attention mechanism to draw global dependencies between input and output. The Transformer allows for significantly more parallelization and can reach a new state of the art in translation quality after being trained for as little as twelve hours on eight P100 GPUs.

在這項工作中，我們提出了 Transformer，這是一種避開遞歸的模型架構，轉而完全依賴注意力機制來繪製輸入和輸出之間的全局依賴關係。Transformer 允許顯著更多的並行化，並且在八個 P100 GPU 上僅訓練十二小時後，就能在翻譯質量方面達到新的技術水平。

## 2 Background
## 2 背景

The goal of reducing sequential computation also forms the foundation of the Extended Neural GPU [16], ByteNet [18] and ConvS2S [9], all of which use convolutional neural networks as basic building block, computing hidden representations in parallel for all input and output positions. In these models, the number of operations required to relate signals from two arbitrary input or output positions grows in the distance between positions, linearly for ConvS2S and logarithmically for ByteNet. This makes it more difficult to learn dependencies between distant positions [12]. In the Transformer this is reduced to a constant number of operations, albeit at the cost of reduced effective resolution due to averaging attention-weighted positions, an effect we counteract with Multi-Head Attention as described in section 3.2.

減少循序計算的目標也構成了擴展神經 GPU [16]、ByteNet [18] 和 ConvS2S [9] 的基礎，所有這些都使用卷積神經網路作為基本建構模塊，為所有輸入和輸出位置並行計算隱藏表示。在這些模型中，關聯來自兩個任意輸入或輸出位置的信號所需的操作數量隨著位置之間的距離而增長，對於 ConvS2S 是線性的，對於 ByteNet 是對數的。這使得學習遠距離位置之間的依賴關係更加困難 [12]。在 Transformer 中，這被減少到常數數量的操作，儘管代價是降低了有效分辨率，因為平均了注意力加權的位置，我們用第 3.2 節中描述的多頭注意力來抵消這種影響。

Self-attention, sometimes called intra-attention is an attention mechanism relating different positions of a single sequence in order to compute a representation of the sequence. Self-attention has been used successfully in a variety of tasks including reading comprehension, abstractive summarization, textual entailment and learning task-independent sentence representations [4, 27, 28, 22].

自我注意力，有時也稱為內部注意力，是一種注意力機制，它關聯單一序列的不同位置，以計算該序列的表示。自我注意力已成功應用於各種任務，包括閱讀理解、摘要、文本蘊涵以及學習與任務無關的句子表示 [4, 27, 28, 22]。

End-to-end memory networks are based on a recurrent attention mechanism instead of sequence-aligned recurrence and have been shown to perform well on simple-language question answering and language modeling tasks [34].

端到端記憶網路是基於循環注意力機制而非序列對齊的循環，並且已證明在簡單語言問答和語言建模任務上表現良好[34]。

To the best of our knowledge, however, the Transformer is the first transduction model relying entirely on self-attention to compute representations of its input and output without using sequence-aligned RNNs or convolution. In the following sections, we will describe the Transformer, motivate self-attention and discuss its advantages over models such as [17, 18] and [9].

然而，據我們所知，Transformer 是第一個完全依賴自我注意力來計算其輸入和輸出表示的轉換模型，而沒有使用序列對齊的 RNN 或卷積。在接下來的章節中，我們將描述 Transformer，闡述自我注意力的動機，並討論其相對於 [17, 18] 和 [9] 等模型的優勢。

## 3 Model Architecture
## 3 模型架構

Most competitive neural sequence transduction models have an encoder-decoder structure [5, 2, 35]. Here, the encoder maps an input sequence of symbol representations (x1, ..., xn) to a sequence of continuous representations z = (z1, ..., zn). Given z, the decoder then generates an output sequence (y1, ..., ym) of symbols one element at a time. At each step the model is auto-regressive [10], consuming the previously generated symbols as additional input when generating the next.

大多數具競爭力的神經序列轉換模型都具有編碼器-解碼器結構 [5, 2, 35]。在此，編碼器將符號表示的輸入序列 (x1, ..., xn) 映射到連續表示的序列 z = (z1, ..., zn)。給定 z，解碼器然後一次一個元素地生成符號的輸出序列 (y1, ..., ym)。在每個步驟中，模型都是自回歸的 [10]，在生成下一個符號時，會使用先前生成的符號作為額外輸入。

[Image]

Figure 1: The Transformer - model architecture.
圖 1：Transformer - 模型架構。

The Transformer follows this overall architecture using stacked self-attention and point-wise, fully connected layers for both the encoder and decoder, shown in the left and right halves of Figure 1, respectively.
Transformer 遵循此整體架構，對編碼器和解碼器使用堆疊式自我注意和逐點全連接層，分別如圖 1 的左半部和右半部所示。

### 3.1 Encoder and Decoder Stacks
### 3.1 編碼器與解碼器堆疊

Encoder: The encoder is composed of a stack of N = 6 identical layers. Each layer has two sub-layers. The first is a multi-head self-attention mechanism, and the second is a simple, position-wise fully connected feed-forward network. We employ a residual connection [11] around each of the two sub-layers, followed by layer normalization [1]. That is, the output of each sub-layer is LayerNorm(x + Sublayer(x)), where Sublayer(x) is the function implemented by the sub-layer itself. To facilitate these residual connections, all sub-layers in the model, as well as the embedding layers, produce outputs of dimension dmodel = 512.

編碼器：編碼器由 N = 6 個相同層堆疊而成。每層有兩個子層。第一個是多頭自我注意機制，第二個是簡單的、按位置全連接的前饋網路。我們在兩個子層周圍都使用殘差連接 [11]，然後進行層歸一化 [1]。也就是說，每個子層的輸出是 LayerNorm(x + Sublayer(x))，其中 Sublayer(x) 是由子層本身實現的函數。為了方便這些殘差連接，模型中的所有子層以及嵌入層都產生維度 dmodel = 512 的輸出。

Decoder: The decoder is also composed of a stack of N = 6 identical layers. In addition to the two sub-layers in each encoder layer, the decoder inserts a third sub-layer, which performs multi-head attention over the output of the encoder stack. Similar to the encoder, we employ residual connections around each of the sub-layers, followed by layer normalization. We also modify the self-attention sub-layer in the decoder stack to prevent positions from attending to subsequent positions. This masking, combined with fact that the output embeddings are offset by one position, ensures that the predictions for position i can depend only on the known outputs at positions less than i.

解碼器：解碼器也由 N=6 個相同的層堆疊而成。除了每個編碼器層中的兩個子層外，解碼器還插入了第三個子層，它對編碼器堆疊的輸出執行多頭注意力。與編碼器類似，我們在每個子層周圍採用殘差連接，然後進行層歸一化。我們還修改解碼器堆疊中的自註意力子層，以防止位置關注後續位置。這種遮罩，結合輸出嵌入偏移一個位置的事實，確保了對位置 i 的預測只能依賴於小於 i 的位置的已知輸出。

### 3.2 Attention
### 3.2 注意力

An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is computed as a weighted sum of the values, where the weight assigned to each value is computed by a compatibility function of the query with the corresponding key.

注意力函數可以描述為將一個查詢和一組鍵值對映射到一個輸出，其中查詢、鍵、值和輸出都是向量。輸出是值的加權和，其中分配給每個值的權重由查詢與相應鍵的兼容性函數計算。

[Image]

Figure 2: (left) Scaled Dot-Product Attention. (right) Multi-Head Attention consists of several attention layers running in parallel.
圖 2：（左）縮放點積注意力。(右) 多頭注意力由數個並行運行的注意力層組成。

### 3.2.1 Scaled Dot-Product Attention
### 3.2.1 縮放點積注意力

We call our particular attention "Scaled Dot-Product Attention" (Figure 2). The input consists of queries and keys of dimension dk, and values of dimension dv. We compute the dot products of the query with all keys, divide each by √dk, and apply a softmax function to obtain the weights on the values.

我們稱我們的特定注意力機制為「縮放點積注意力」（圖 2）。輸入由維度為 dk 的查詢和鍵，以及維度為 dv 的值組成。我們計算查詢與所有鍵的點積，將每個點積除以 √dk，然後應用 softmax 函數以獲得值的權重。

In practice, we compute the attention function on a set of queries simultaneously, packed together into a matrix Q. The keys and values are also packed together into matrices K and V. We compute the matrix of outputs as:

在實踐中，我們同時在一組查詢上計算注意力函數，將它們打包成一個矩陣 Q。鍵和值也打包成矩陣 K 和 V。我們計算輸出矩陣如下：

Attention(Q, K, V) = softmax(QKT/√dk)V

The two most commonly used attention functions are additive attention [2], and dot-product (multiplicative) attention. Dot-product attention is identical to our algorithm, except for the scaling factor of 1/√dk. Additive attention computes the compatibility function using a feed-forward network with a single hidden layer. While the two are similar in theoretical complexity, dot-product attention is much faster and more space-efficient in practice, since it can be implemented using highly optimized matrix multiplication code.

兩種最常用的注意力函數是加法性注意力 [2] 和點積（乘法性）注意力。除了 1/√dk 的縮放因子外，點積注意力與我們的演算法相同。加法性注意力使用具有單一隱藏層的前饋網路計算相容性函數。雖然兩者在理論複雜度上相似，但點積注意力在實踐中速度更快、空間效率更高，因為它可以利用高度優化的矩陣乘法程式碼來實現。

While for small values of dk the two mechanisms perform similarly, additive attention outperforms dot product attention without scaling for larger values of dk [3]. We suspect that for large values of dk, the dot products grow large in magnitude, pushing the softmax function into regions where it has extremely small gradients 4. To counteract this effect, we scale the dot products by 1/√dk.

雖然對於較小的 dk 值，兩種機制的表現相似，但對於較大的 dk 值，加法注意力在沒有縮放的情況下優於點積注意力 [3]。我們懷疑對於較大的 dk 值，點積的量級會變大，將 softmax 函數推向梯度極小的區域 4。為了抵消這種影響，我們將點積乘以 1/√dk。

4To illustrate why the dot products get large, assume that the components of q and k are independent random variables with mean 0 and variance 1. Then their dot product, q · k = ∑i=1dk qiki, has mean 0 and variance dk.
4 為了說明點積為何會變大，假設 q 和 k 的分量是平均值為 0 且變異數為 1 的獨立隨機變數。那麼它們的點積 q · k = ∑i=1dk qiki 的平均值為 0，變異數為 dk。

### 3.2.2 Multi-Head Attention
### 3.2.2 多頭注意力

Instead of performing a single attention function with dmodel-dimensional keys, values and queries, we found it beneficial to linearly project the queries, keys and values h times with different, learned linear projections to dk, dk and dv dimensions, respectively. On each of these projected versions of queries, keys and values we then perform the attention function in parallel, yielding dv-dimensional output values. These are concatenated and once again projected, resulting in the final values, as depicted in Figure 2.

我們發現，與其使用 dmodel 維度的鍵、值和查詢執行單一注意力函數，不如將查詢、鍵和值分別使用不同的、學習到的線性投影 h 次，投影到 dk、dk 和 dv 維度，這樣更有利。在每個投影版本的查詢、鍵和值上，我們並行執行注意力函數，產生 dv 維度的輸出值。如圖 2 所示，這些值被串接起來，然後再次投影，得到最終值。

Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions. With a single attention head, averaging inhibits this.

多頭注意力允許模型在不同位置共同關注來自不同表示子空間的資訊。單一注意力頭則會因為平均化而抑制這種能力。

MultiHead(Q, K, V) = Concat(head1, ..., headh)WO
where headi = Attention(QWQi, KWKi, VWVi)

其中，投影是參數矩陣 WQi ∈ Rdmodel×dk, WKi ∈ Rdmodel×dk, WVi ∈ Rdmodel×dv 和 WO ∈ Rhdv×dmodel。

In this work we employ h = 8 parallel attention layers, or heads. For each of these we use dk = dv = dmodel/h = 64. Due to the reduced dimension of each head, the total computational cost is similar to that of single-head attention with full dimensionality.

在這項工作中，我們使用 h = 8 個平行的注意力層，或稱為「頭」。對於每個頭，我們使用 dk = dv = dmodel/h = 64。由於每個頭的維度降低，總計算成本與全維度的單頭注意力相似。

### 3.2.3 Applications of Attention in our Model
### 3.2.3 模型中注意力的應用

The Transformer uses multi-head attention in three different ways:

Transformer 以三種不同方式使用多頭注意力：

* In "encoder-decoder attention" layers, the queries come from the previous decoder layer, and the memory keys and values come from the output of the encoder. This allows every position in the decoder to attend over all positions in the input sequence. This mimics the typical encoder-decoder attention mechanisms in sequence-to-sequence models such as [38, 2, 9].
* 在「編碼器-解碼器注意力」層中，查詢來自前一個解碼器層，而記憶鍵和值則來自編碼器的輸出。這使得解碼器中的每個位置都能關注輸入序列中的所有位置。這模仿了序列到序列模型中典型的編碼器-解碼器注意力機制，例如 [38, 2, 9]。

* The encoder contains self-attention layers. In a self-attention layer all of the keys, values and queries come from the same place, in this case, the output of the previous layer in the encoder. Each position in the encoder can attend to all positions in the previous layer of the encoder.
* 編碼器包含自我注意層。在自我注意層中，所有的鍵、值和查詢都來自同一個地方，在本例中，即編碼器中前一層的輸出。編碼器中的每個位置都可以關注編碼器前一層的所有位置。

* Similarly, self-attention layers in the decoder allow each position in the decoder to attend to all positions in the decoder up to and including that position. We need to prevent leftward information flow in the decoder to preserve the auto-regressive property. We implement this inside of scaled dot-product attention by masking out (setting to -∞) all values in the input of the softmax which correspond to illegal connections. See Figure 2.
* 同樣地，解碼器中的自我注意層允許解碼器中的每個位置關注解碼器中直到並包括該位置的所有位置。我們需要防止解碼器中的資訊向左流動，以保持自回歸屬性。我們在縮放點積注意力內部通過遮罩掉（設置為-∞）softmax 輸入中對應於非法連接的所有值來實現這一點。請參見圖2。

### 3.3 Position-wise Feed-Forward Networks
### 3.3 逐點前饋網路

In addition to attention sub-layers, each of the layers in our encoder and decoder contains a fully connected feed-forward network, which is applied to each position separately and identically. This consists of two linear transformations with a ReLU activation in between.

除了注意力子層之外，我們編碼器和解碼器中的每一層都包含一個全連接的前饋網路，該網路分別且相同地應用於每個位置。這包括兩個線性轉換，中間有一個 ReLU 激活。

FFN(x) = max(0, xW1 + b1)W2 + b2 (2)

While the linear transformations are the same across different positions, they use different parameters from layer to layer. Another way of describing this is as two convolutions with kernel size 1. The dimensionality of input and output is dmodel = 512, and the inner-layer has dimensionality dff = 2048.

雖然線性變換在不同位置上是相同的，但它們在不同層之間使用不同的參數。另一種描述方式是使用兩個核心大小為 1 的卷積。輸入和輸出的維度為 dmodel = 512，內層的維度為 dff = 2048。

### 3.4 Embeddings and Softmax
### 3.4 嵌入與 Softmax

Similarly to other sequence transduction models, we use learned embeddings to convert the input tokens and output tokens to vectors of dimension dmodel. We also use the usual learned linear transformation and softmax function to convert the decoder output to predicted next-token probabilities. In our model, we share the same weight matrix between the two embedding layers and the pre-softmax linear transformation, similar to [30]. In the embedding layers, we multiply those weights by √dmodel.

與其他序列轉換模型類似，我們使用學習到的嵌入將輸入標記和輸出標記轉換為維度為 dmodel 的向量。我們也使用通常學習到的線性轉換和 softmax 函數將解碼器輸出轉換為預測的下一個標記的機率。在我們的模型中，我們在兩個嵌入層和 softmax 前的線性轉換之間共享相同的權重矩陣，類似於 [30]。在嵌入層中，我們將這些權重乘以 √dmodel。

### 3.5 Positional Encoding
### 3.5 位置編碼

Since our model contains no recurrence and no convolution, in order for the model to make use of the order of the sequence, we must inject some information about the relative or absolute position of the tokens in the sequence. To this end, we add "positional encodings" to the input embeddings at the bottoms of the encoder and decoder stacks. The positional encodings have the same dimension dmodel as the embeddings, so that the two can be summed. There are many choices of positional encodings, learned and fixed [9].

由於我們的模型不包含遞歸和卷積，為了讓模型利用序列的順序，我們必須注入一些關於序列中標記的相對或絕對位置的資訊。為此，我們在編碼器和解碼器堆疊的底部向輸入嵌入添加「位置編碼」。位置編碼的維度與嵌入相同，均為 dmodel，因此兩者可以相加。位置編碼有多種選擇，可以是學習的，也可以是固定的 [9]。

In this work, we use sine and cosine functions of different frequencies:

在這項工作中，我們使用不同頻率的正弦和餘弦函數：

PE(pos,2i) = sin(pos/10000^(2i/dmodel))
PE(pos,2i+1) = cos(pos/10000^(2i/dmodel))

where pos is the position and i is the dimension. That is, each dimension of the positional encoding corresponds to a sinusoid. The wavelengths form a geometric progression from 2π to 10000 · 2π. We chose this function because we hypothesized it would allow the model to easily learn to attend by relative positions, since for any fixed offset k, PEpos+k can be represented as a linear function of PEpos.

其中 pos 是位置，i 是維度。也就是說，位置編碼的每個維度都對應一個正弦曲線。波長形成一個從 2π 到 10000 · 2π 的幾何級數。我們選擇這個函數，因為我們假設它能讓模型輕鬆學會通過相對位置來進行注意力分配，因為對於任何固定的偏移量 k，PEpos+k 都可以表示為 PEpos 的線性函數。

We also experimented with using learned positional embeddings [9] instead, and found that the two versions produced nearly identical results (see Table 3 row (E)). We chose the sinusoidal version because it may allow the model to extrapolate to sequence lengths longer than the ones encountered during training.

我們也嘗試使用學習到的位置嵌入 [9]，結果發現兩種版本產生幾乎相同的結果（見表 3 行 (E)）。我們選擇了正弦版本，因為它可能允許模型推斷到比訓練期間遇到的序列更長的序列長度。

## 4 Why Self-Attention
## 4 為什麼選擇自我注意力機制

In this section we compare various aspects of self-attention layers to the recurrent and convolutional layers commonly used for mapping one variable-length sequence of symbol representations (x1, ..., xn) to another sequence of equal length (z1, ..., zn), with xi, zi ∈ Rd, such as a hidden layer in a typical sequence transduction encoder or decoder. Motivating our use of self-attention we consider three desiderata.

在本節中，我們比較了自我注意層與循環層和卷積層的各個方面，這些層通常用於將一個可變長度符號表示序列 (x1, ..., xn) 映射到另一個等長序列 (z1, ..., zn)，其中 xi, zi ∈ Rd，例如在典型序列轉換編碼器或解碼器中的隱藏層。我們考慮三個期望的特性來闡述我們使用自我注意力的動機。

One is the total computational complexity per layer. Another is the amount of computation that can be parallelized, as measured by the minimum number of sequential operations required.

其一是每層的總計算複雜度。其二是可並行化的計算量，以所需的最少順序操作數量來衡量。

The third is the path length between long-range dependencies in the network. Learning long-range dependencies is a key challenge in many sequence transduction tasks. One key factor affecting the ability to learn such dependencies is the length of the paths forward and backward signals have to traverse in the network. The shorter these paths between any combination of positions in the input and output sequences, the easier it is to learn long-range dependencies [12]. Hence we also compare the maximum path length between any two input and output positions in networks composed of the different layer types.

第三點是網絡中長距離依賴關係的路徑長度。學習長距離依賴關係是許多序列轉換任務中的一個關鍵挑戰。影響學習此類依賴關係能力的一個關鍵因素是信號在網絡中向前和向後傳播所需的路徑長度。輸入和輸出序列中任意位置組合之間的這些路徑越短，學習長距離依賴關係就越容易 [12]。因此，我們也比較了由不同層類型組成的網絡中任意兩個輸入和輸出位置之間的最大路徑長度。

[Image]

Table 1: Maximum path lengths, per-layer complexity and minimum number of sequential operations for different layer types. n is the sequence length, d is the representation dimension, k is the kernel size of convolutions and r the size of the neighborhood in restricted self-attention.
表 1：不同層類型的最大路徑長度、每層複雜度和最少順序操作次數。n 是序列長度，d 是表示維度，k 是卷積的核心大小，r 是受限自我注意力中鄰域的大小。

As noted in Table 1, a self-attention layer connects all positions with a constant number of sequentially executed operations, whereas a recurrent layer requires O(n) sequential operations. In terms of computational complexity, self-attention layers are faster than recurrent layers when the sequence length n is smaller than the representation dimensionality d, which is most often the case with sentence representations used by state-of-the-art models in machine translations, such as word-piece [38] and byte-pair [31] representations. To improve computational performance for tasks involving very long sequences, self-attention could be restricted to considering only a neighborhood of size r in the input sequence centered around the respective output position. This would increase the maximum path length to O(n/r). We plan to investigate this approach further in future work.

如表 1 所示，自註意力層用常數數量的順序執行操作連接所有位置，而循環層需要 O(n) 次順序操作。在計算複雜度方面，當序列長度 n 小於表示維度 d 時，自註意力層比循環層快，這在機器翻譯中使用的句子表示（例如詞片 [38] 和字節對 [31]）的最新模型中通常是這種情況。為了提高涉及非常長序列的任務的計算性能，可以將自註意力限制為僅考慮以相應輸出位置為中心的輸入序列中大小為 r 的鄰域。這會將最大路徑長度增加到 O(n/r)。我們計劃在未來的工作中進一步研究這種方法。

A single convolutional layer with kernel width k < n does not connect all pairs of input and output positions. Doing so requires a stack of O(n/k) convolutional layers in the case of contiguous kernels, or O(logk(n)) in the case of dilated convolutions [18], increasing the length of the longest paths between any two positions in the network. Convolutional layers are generally more expensive than recurrent layers, by a factor of k. Separable convolutions [6], however, decrease the complexity considerably, to O(k · n · d + n · d²). Even with k = n, however, the complexity of a separable convolution is equal to the combination of a self-attention layer and a point-wise feed-forward layer, the approach we take in our model.

單一卷積層的核心寬度 k < n，無法連接所有輸入和輸出位置對。若要達成此目的，在連續核心的情況下需要 O(n/k) 個卷積層堆疊，或在擴張卷積的情況下需要 O(logk(n)) 個 [18]，這會增加網路中任意兩個位置之間最長路徑的長度。卷積層通常比循環層昂貴 k 倍。然而，可分離卷積 [6] 大幅降低了複雜度，降至 O(k·n·d + n·d²)。然而，即使 k = n，可分離卷積的複雜度也等於自我注意層和逐點前饋層的組合，這正是我們模型採用的方法。

As a side benefit, self-attention could yield more interpretable models. We inspect attention distributions from our models and present and discuss examples in the appendix. Not only do individual attention heads clearly learn to perform different tasks, many appear to exhibit behavior related to the syntactic and semantic structure of the sentences.

作為附帶的好處，自我注意力機制可以產生更具可解釋性的模型。我們檢查模型中的注意力分佈，並在附錄中呈現和討論範例。不僅個別的注意力頭清楚地學會執行不同的任務，許多注意力頭似乎也表現出與句子的句法和語義結構相關的行為。

## 5 Training
## 5 訓練

This section describes the training regime for our models.
本節描述我們模型的訓練方案。

### 5.1 Training Data and Batching
### 5.1 訓練資料與批次處理

We trained on the standard WMT 2014 English-German dataset consisting of about 4.5 million sentence pairs. Sentences were encoded using byte-pair encoding [3], which has a shared source-target vocabulary of about 37000 tokens. For English-French, we used the significantly larger WMT 2014 English-French dataset consisting of 36M sentences and split tokens into a 32000 word-piece vocabulary [38]. Sentence pairs were batched together by approximate sequence length. Each training batch contained a set of sentence pairs containing approximately 25000 source tokens and 25000 target tokens.

我們在標準的 WMT 2014 英德數據集上進行了訓練，該數據集包含約 450 萬個句子對。句子使用位元組對編碼 [3] 進行編碼，該編碼具有約 37000 個標記的共享源目標詞彙表。對於英法翻譯，我們使用了更大的 WMT 2014 英法數據集，該數據集包含 3600 萬個句子，並將標記拆分為 32000 個詞片的詞彙表 [38]。句子對按近似序列長度分批處理。每個訓練批次包含一組句子對，其中包含約 25000 個源標記和 25000 個目標標記。

### 5.2 Hardware and Schedule
### 5.2 硬體與時程

We trained our models on one machine with 8 NVIDIA P100 GPUs. For our base models using the hyperparameters described throughout the paper, each training step took about 0.4 seconds. We trained the base models for a total of 100,000 steps or 12 hours. For our big models, (described on the bottom line of table 3), step time was 1.0 seconds. The big models were trained for 300,000 steps (3.5 days).

我們在一台配備 8 個 NVIDIA P100 GPU 的機器上訓練我們的模型。對於我們使用整篇論文中所述超參數的基礎模型，每個訓練步驟大約需要 0.4 秒。我們總共訓練了基礎模型 100,000 個步驟或 12 小時。對於我們的大模型（如表 3 底行所述），步驟時間為 1.0 秒。大模型訓練了 300,000 個步驟（3.5 天）。

### 5.3 Optimizer
### 5.3 優化器

We used the Adam optimizer [20] with β1 = 0.9, β2 = 0.98 and ε = 10−9. We varied the learning rate over the course of training, according to the formula:

我們使用了 Adam 優化器 [20]，其中 β1 = 0.9，β2 = 0.98 且 ε = 10−9。我們在訓練過程中根據以下公式調整學習率：

lrate = dmodel^(-0.5) * min(step_num^(-0.5), step_num * warmup_steps^(-1.5)) (3)

This corresponds to increasing the learning rate linearly for the first warmup_steps training steps, and decreasing it thereafter proportionally to the inverse square root of the step number. We used warmup_steps = 4000.
這對應於在第一個 warmup_steps 訓練步驟中線性增加學習率，然後與步驟數的平方根倒數成比例地減少學習率。我們使用了 warmup_steps = 4000。

### 5.4 Regularization
### 5.4 正規化

We employ three types of regularization during training:
我們在訓練期間採用三種正規化方法：

[Image]

Table 2: The Transformer achieves better BLEU scores than previous state-of-the-art models on the English-to-German and English-to-French newstest2014 tests at a fraction of the training cost.
表 2：在英德和英法 newstest2014 測試中，Transformer 以一小部分的訓練成本，取得了比先前最先進模型更好的 BLEU 分數。

Residual Dropout We apply dropout [33] to the output of each sub-layer, before it is added to the sub-layer input and normalized. In addition, we apply dropout to the sums of the embeddings and the positional encodings in both the encoder and decoder stacks. For the base model, we use a rate of Pdrop = 0.1.

殘差 Dropout 我們在每個子層的輸出上應用 dropout [33]，然後再將其添加到子層的輸入並進行歸一化。此外，我們在編碼器和解碼器堆疊中，對嵌入和位置編碼的總和應用 dropout。對於基礎模型，我們使用 Pdrop = 0.1 的比率。

Label Smoothing During training, we employed label smoothing of value εls = 0.1 [36]. This hurts perplexity, as the model learns to be more unsure, but improves accuracy and BLEU score.
標籤平滑 在訓練期間，我們採用了值為 εls = 0.1 的標籤平滑 [36]。這會損害困惑度，因為模型學會變得更不確定，但能提高準確度和 BLEU 分數。

## 6 Results
## 6 結果

### 6.1 Machine Translation
### 6.1 機器翻譯

On the WMT 2014 English-to-German translation task, the big transformer model (Transformer (big) in Table 2) outperforms the best previously reported models (including ensembles) by more than 2.0 BLEU, establishing a new state-of-the-art BLEU score of 28.4. The configuration of this model is listed in the bottom line of Table 3. Training took 3.5 days on 8 P100 GPUs. Even our base model surpasses all previously published models and ensembles, at a fraction of the training cost of any of the competitive models.

在 WMT 2014 英德翻譯任務中，大型 transformer 模型（表 2 中的 Transformer (big)）比先前報導的最佳模型（包括集成模型）高出 2.0 BLEU 以上，創下了 28.4 的最新 BLEU 分數。該模型的配置列在表 3 的底行。在 8 個 P100 GPU 上訓練耗時 3.5 天。即使是我們的基礎模型，也以任何競爭模型訓練成本的一小部分，超越了所有先前發布的模型和集成模型。

On the WMT 2014 English-to-French translation task, our big model achieves a BLEU score of 41.8, outperforming all of the previously published single models, at less than 1/4 the training cost of the previous state-of-the-art model. The Transformer (big) model trained for English-to-French used dropout rate Pdrop = 0.1, instead of 0.3.

在 WMT 2014 英法翻譯任務上，我們的大型模型達到了 41.8 的 BLEU 分數，超越了所有先前發表的單一模型，而訓練成本不到先前最先進模型的 1/4。用於英法翻譯的大型 Transformer 模型使用了 Pdrop = 0.1 的 dropout 率，而不是 0.3。

For the base models, we used a single model obtained by averaging the last 5 checkpoints, which were written at 10-minute intervals. For the big models, we averaged the last 20 checkpoints. We used beam search with a beam size of 4 and length penalty α = 0.6 [38]. These hyperparameters were chosen after experimentation on the development set. We set the maximum output length during inference to input length + 50, but terminate early when possible [38].

對於基礎模型，我們使用單一模型，該模型是通過平均最後 5 個檢查點獲得的，這些檢查點以 10 分鐘為間隔寫入。對於大型模型，我們平均了最後 20 個檢查點。我們使用波束大小為 4 的波束搜索和長度懲罰 α = 0.6 [38]。這些超參數是在開發集上進行實驗後選擇的。我們在推斷期間將最大輸出長度設置為輸入長度 + 50，但盡可能提前終止 [38]。

Table 2 summarizes our results and compares our translation quality and training costs to other model architectures from the literature. We estimate the number of floating point operations used to train a model by multiplying the training time, the number of GPUs used, and an estimate of the sustained single-precision floating-point capacity of each GPU 5.

表 2 總結了我們的結果，並將我們的翻譯品質和訓練成本與文獻中其他模型架構進行了比較。我們通過將訓練時間、使用的 GPU 數量以及每個 GPU 持續單精度浮點能力的估計值相乘，來估計訓練模型所用的浮點運算次數 5。

5We used values of 2.8, 3.7, 6.0 and 9.5 TFLOPS for K80, K40, M40 and P100, respectively.
5我們分別對 K80、K40、M40 和 P100 使用了 2.8、3.7、6.0 和 9.5 TFLOPS 的值。

### 6.2 Model Variations
### 6.2 模型變體

To evaluate the importance of different components of the Transformer, we varied our base model in different ways, measuring the change in performance on English-to-German translation on the development set, newstest2013. We used beam search as described in the previous section, but no checkpoint averaging. We present these results in Table 3.

為評估 Transformer 不同組件的重要性，我們以不同方式改變我們的基礎模型，並在開發集 newstest2013 上測量英德翻譯性能的變化。我們使用前一節描述的束搜索，但沒有檢查點平均。我們在表 3 中呈現這些結果。

[Image]

Table 3: Variations on the Transformer architecture. Unlisted values are identical to those of the base model. All metrics are on the English-to-German translation development set, newstest2013. Listed perplexities are per-wordpiece, according to our byte-pair encoding, and should not be compared to per-word perplexities.
表 3：Transformer 架構的變體。未列出的值與基礎模型相同。所有指標均在英德翻譯開發集 newstest2013 上得出。列出的困惑度是根據我們的位元組對編碼計算的每個詞片段的困惑度，不應與每個詞的困惑度進行比較。

In Table 3 rows (A), we vary the number of attention heads and the attention key and value dimensions, keeping the amount of computation constant, as described in Section 3.2.2. While single-head attention is 0.9 BLEU worse than the best setting, quality also drops off with too many heads.

在表 3 的 (A) 列中，我們改變了注意力頭的數量以及注意力的鍵和值的維度，同時保持計算量不變，如第 3.2.2 節所述。雖然單頭注意力比最佳設定差 0.9 BLEU，但品質也會隨著頭數過多而下降。

In Table 3 rows (B), we observe that reducing the attention key size dk hurts model quality. This suggests that determining compatibility is not easy and that a more sophisticated compatibility function than dot product may be beneficial. We further observe in rows (C) and (D) that, as expected, bigger models are better, and dropout is very helpful in avoiding over-fitting. In row (E) we replace our sinusoidal positional encoding with learned positional embeddings [9], and observe nearly identical results to the base model.

在表 3 (B) 列中，我們觀察到減小注意力鍵大小 dk 會損害模型品質。這表明確定相容性並不容易，而且比點積更複雜的相容性函數可能是有益的。我們進一步在 (C) 和 (D) 列中觀察到，正如預期的那樣，更大的模型更好，而 dropout 在避免過度擬合方面非常有用。在 (E) 列中，我們用學習到的位置嵌入 [9] 取代了我們的正弦位置編碼，並觀察到與基礎模型幾乎相同的結果。

### 6.3 English Constituency Parsing
### 6.3 英語選區剖析

To evaluate if the Transformer can generalize to other tasks we performed experiments on English constituency parsing. This task presents specific challenges: the output is subject to strong structural constraints and is significantly longer than the input. Furthermore, RNN sequence-to-sequence models have not been able to attain state-of-the-art results in small-data regimes [37].

為了評估 Transformer 是否能推廣到其他任務，我們對英語成分句法分析進行了實驗。這項任務提出了特定的挑戰：輸出受到強烈的結構約束，並且比輸入長得多。此外，在小數據體制下，RNN 序列到序列模型未能達到最先進的結果 [37]。

We trained a 4-layer transformer with dmodel = 1024 on the Wall Street Journal (WSJ) portion of the Penn Treebank [25], about 40K training sentences. We also trained it in a semi-supervised setting, using the larger high-confidence and BerkleyParser corpora from with approximately 17M sentences [37]. We used a vocabulary of 16K tokens for the WSJ only setting and a vocabulary of 32K tokens for the semi-supervised setting.

我們在賓州樹庫 [25] 的華爾街日報 (WSJ) 部分訓練了一個 4 層的 Transformer，模型維度 dmodel = 1024，訓練語句約 4 萬句。我們也在半監督的環境下訓練，使用來自 [37] 的較大高信度及 BerkleyParser 語料庫，約有 1700 萬句。我們對僅限 WSJ 的設定使用 16K 詞彙，對半監督設定使用 32K 詞彙。

We performed only a small number of experiments to select the dropout, both attention and residual (section 5.4), learning rates and beam size on the Section 22 development set, all other parameters remained unchanged from the English-to-German base translation model. During inference, we increased the maximum output length to input length + 300. We used a beam size of 21 and α = 0.3 for both WSJ only and the semi-supervised setting.

我們僅進行了少量實驗來選擇 dropout（注意力和殘差，第 5.4 節）、學習率和束大小，在第 22 節開發集上，所有其他參數均與英德基礎翻譯模型保持不變。在推論期間，我們將最大輸出長度增加到輸入長度 + 300。我們對僅 WSJ 和半監督設置都使用了 21 的束大小和 α = 0.3。

[Image]

Table 4: The Transformer generalizes well to English constituency parsing (Results are on Section 23 of WSJ)
表 4：Transformer 在英語成分句法分析中表現良好（結果在《華爾街日報》第 23 節）

Our results in Table 4 show that despite the lack of task-specific tuning our model performs surprisingly well, yielding better results than all previously reported models with the exception of the Recurrent Neural Network Grammar [8].

我們在表 4 中的結果顯示，儘管缺乏針對特定任務的調整，我們的模型表現出奇地好，除了循環神經網路文法 [8] 之外，產生的結果優於所有先前報導的模型。

In contrast to RNN sequence-to-sequence models [37], the Transformer outperforms the BerkeleyParser [29] even when training only on the WSJ training set of 40K sentences.

相較於 RNN 序列對序列模型 [37]，Transformer 即使只在 4 萬個句子的華爾街日報訓練集上訓練，其表現也優於 BerkeleyParser [29]。

## 7 Conclusion
## 7 結論

In this work, we presented the Transformer, the first sequence transduction model based entirely on attention, replacing the recurrent layers most commonly used in encoder-decoder architectures with multi-headed self-attention.

在這項工作中，我們介紹了 Transformer，這是第一個完全基於注意力機制的序列轉換模型，它用多頭自注意力機制取代了編碼器-解碼器架構中最常用的循環層。

For translation tasks, the Transformer can be trained significantly faster than architectures based on recurrent or convolutional layers. On both WMT 2014 English-to-German and WMT 2014 English-to-French translation tasks, we achieve a new state of the art. In the former task our best model outperforms even all previously reported ensembles.

對於翻譯任務，Transformer 的訓練速度明顯快於基於循環或卷積層的架構。在 WMT 2014 英德和 WMT 2014 英法翻譯任務中，我們都達到了新的技術水平。在前一項任務中，我們最好的模型甚至超越了所有先前報導的集成模型。

We are excited about the future of attention-based models and plan to apply them to other tasks. We plan to extend the Transformer to problems involving input and output modalities other than text and to investigate local, restricted attention mechanisms to efficiently handle large inputs and outputs such as images, audio and video. Making generation less sequential is another research goals of ours.

我們對基於注意力的模型的未來感到興奮，並計劃將它們應用於其他任務。我們計劃將 Transformer 擴展到涉及文本以外的輸入和輸出模態的問題，並研究局部、受限的注意力機制，以有效地處理大型輸入和輸出，如圖像、音訊和視訊。使生成過程不那麼循序是我們的另一個研究目標。

The code we used to train and evaluate our models is available at https://github.com/tensorflow/tensor2tensor.

我們用來訓練和評估模型的程式碼可在 https://github.com/tensorflow/tensor2tensor 取得。

Acknowledgements We are grateful to Nal Kalchbrenner and Stephan Gouws for their fruitful comments, corrections and inspiration.
致謝 我們感謝 Nal Kalchbrenner 和 Stephan Gouws 提出富有成效的評論、更正和啟發。

## References
## 參考文獻

[1] Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E Hinton. Layer normalization. arXiv preprint arXiv:1607.06450, 2016.
[1] Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E Hinton. Layer normalization. arXiv preprint arXiv:1607.06450, 2016.

[2] Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. Neural machine translation by jointly learning to align and translate. CoRR, abs/1409.0473, 2014.
[2] Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. Neural machine translation by jointly learning to align and translate. CoRR, abs/1409.0473, 2014.

[3] Denny Britz, Anna Goldie, Minh-Thang Luong, and Quoc V. Le. Massive exploration of neural machine translation architectures. CoRR, abs/1703.03906, 2017.
[3] Denny Britz, Anna Goldie, Minh-Thang Luong, and Quoc V. Le. Massive exploration of neural machine translation architectures. CoRR, abs/1703.03906, 2017.

[4] Jianpeng Cheng, Li Dong, and Mirella Lapata. Long short-term memory-networks for machine reading. arXiv preprint arXiv:1601.06733, 2016.
[4] Jianpeng Cheng, Li Dong, and Mirella Lapata. Long short-term memory-networks for machine reading. arXiv preprint arXiv:1601.06733, 2016.

[5] Kyunghyun Cho, Bart van Merrienboer, Caglar Gulcehre, Fethi Bougares, Holger Schwenk, and Yoshua Bengio. Learning phrase representations using rnn encoder-decoder for statistical machine translation. CoRR, abs/1406.1078, 2014.
[5] Kyunghyun Cho, Bart van Merrienboer, Caglar Gulcehre, Fethi Bougares, Holger Schwenk, and Yoshua Bengio. Learning phrase representations using rnn encoder-decoder for statistical machine translation. CoRR, abs/1406.1078, 2014.

[6] Francois Chollet. Xception: Deep learning with depthwise separable convolutions. arXiv preprint arXiv:1610.02357, 2016.
[6] Francois Chollet. Xception: Deep learning with depthwise separable convolutions. arXiv preprint arXiv:1610.02357, 2016.

[7] Junyoung Chung, Çaglar Gülçehre, Kyunghyun Cho, and Yoshua Bengio. Empirical evaluation of gated recurrent neural networks on sequence modeling. CoRR, abs/1412.3555, 2014.
[7] Junyoung Chung, Çaglar Gülçehre, Kyunghyun Cho, and Yoshua Bengio. Empirical evaluation of gated recurrent neural networks on sequence modeling. CoRR, abs/1412.3555, 2014.

[8] Chris Dyer, Adhiguna Kuncoro, Miguel Ballesteros, and Noah A. Smith. Recurrent neural network grammars. In Proc. of NAACL, 2016.
[8] Chris Dyer, Adhiguna Kuncoro, Miguel Ballesteros, and Noah A. Smith. Recurrent neural network grammars. In Proc. of NAACL, 2016.

[9] Jonas Gehring, Michael Auli, David Grangier, Denis Yarats, and Yann N. Dauphin. Convolutional sequence to sequence learning. arXiv preprint arXiv:1705.03122v2, 2017.
[9] Jonas Gehring, Michael Auli, David Grangier, Denis Yarats, and Yann N. Dauphin. Convolutional sequence to sequence learning. arXiv preprint arXiv:1705.03122v2, 2017.

[10] Alex Graves. Generating sequences with recurrent neural networks. arXiv preprint arXiv:1308.0850, 2013.
[10] Alex Graves. Generating sequences with recurrent neural networks. arXiv preprint arXiv:1308.0850, 2013.

[11] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 770–778, 2016.
[11] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 770–778, 2016.

[12] Sepp Hochreiter, Yoshua Bengio, Paolo Frasconi, and Jürgen Schmidhuber. Gradient flow in recurrent nets: the difficulty of learning long-term dependencies, 2001.
[12] Sepp Hochreiter, Yoshua Bengio, Paolo Frasconi, and Jürgen Schmidhuber. Gradient flow in recurrent nets: the difficulty of learning long-term dependencies, 2001.

[13] Sepp Hochreiter and Jürgen Schmidhuber. Long short-term memory. Neural computation, 9(8):1735-1780, 1997.
[13] Sepp Hochreiter and Jürgen Schmidhuber. Long short-term memory. Neural computation, 9(8):1735-1780, 1997.

[14] Zhongqiang Huang and Mary Harper. Self-training PCFG grammars with latent annotations across languages. In Proceedings of the 2009 Conference on Empirical Methods in Natural Language Processing, pages 832–841. ACL, August 2009.
[14] Zhongqiang Huang and Mary Harper. Self-training PCFG grammars with latent annotations across languages. In Proceedings of the 2009 Conference on Empirical Methods in Natural Language Processing, pages 832–841. ACL, August 2009.

[15] Rafal Jozefowicz, Oriol Vinyals, Mike Schuster, Noam Shazeer, and Yonghui Wu. Exploring the limits of language modeling. arXiv preprint arXiv:1602.02410, 2016.
[15] Rafal Jozefowicz, Oriol Vinyals, Mike Schuster, Noam Shazeer, and Yonghui Wu. Exploring the limits of language modeling. arXiv preprint arXiv:1602.02410, 2016.

[16] Łukasz Kaiser and Samy Bengio. Can active memory replace attention? In Advances in Neural Information Processing Systems, (NIPS), 2016.
[16] Łukasz Kaiser and Samy Bengio. Can active memory replace attention? In Advances in Neural Information Processing Systems, (NIPS), 2016.

[17] Łukasz Kaiser and Ilya Sutskever. Neural GPUs learn algorithms. In International Conference on Learning Representations (ICLR), 2016.
[17] Łukasz Kaiser and Ilya Sutskever. Neural GPUs learn algorithms. In International Conference on Learning Representations (ICLR), 2016.

[18] Nal Kalchbrenner, Lasse Espeholt, Karen Simonyan, Aaron van den Oord, Alex Graves, and Koray Kavukcuoglu. Neural machine translation in linear time. arXiv preprint arXiv:1610.10099v2, 2017.
[18] Nal Kalchbrenner, Lasse Espeholt, Karen Simonyan, Aaron van den Oord, Alex Graves, and Koray Kavukcuoglu. Neural machine translation in linear time. arXiv preprint arXiv:1610.10099v2, 2017.

[19] Yoon Kim, Carl Denton, Luong Hoang, and Alexander M. Rush. Structured attention networks. In International Conference on Learning Representations, 2017.
[19] Yoon Kim, Carl Denton, Luong Hoang, and Alexander M. Rush. Structured attention networks. In International Conference on Learning Representations, 2017.

[20] Diederik Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In ICLR, 2015.
[20] Diederik Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In ICLR, 2015.

[21] Oleksii Kuchaiev and Boris Ginsburg. Factorization tricks for LSTM networks. arXiv preprint arXiv:1703.10722, 2017.
[21] Oleksii Kuchaiev and Boris Ginsburg. Factorization tricks for LSTM networks. arXiv preprint arXiv:1703.10722, 2017.

[22] Zhouhan Lin, Minwei Feng, Cicero Nogueira dos Santos, Mo Yu, Bing Xiang, Bowen Zhou, and Yoshua Bengio. A structured self-attentive sentence embedding. arXiv preprint arXiv:1703.03130, 2017.
[22] Zhouhan Lin, Minwei Feng, Cicero Nogueira dos Santos, Mo Yu, Bing Xiang, Bowen Zhou, and Yoshua Bengio. A structured self-attentive sentence embedding. arXiv preprint arXiv:1703.03130, 2017.

[23] Minh-Thang Luong, Quoc V. Le, Ilya Sutskever, Oriol Vinyals, and Lukasz Kaiser. Multi-task sequence to sequence learning. arXiv preprint arXiv:1511.06114, 2015.
[23] Minh-Thang Luong, Quoc V. Le, Ilya Sutskever, Oriol Vinyals, and Lukasz Kaiser. Multi-task sequence to sequence learning. arXiv preprint arXiv:1511.06114, 2015.

[24] Minh-Thang Luong, Hieu Pham, and Christopher D Manning. Effective approaches to attention-based neural machine translation. arXiv preprint arXiv:1508.04025, 2015.
[24] Minh-Thang Luong, Hieu Pham, and Christopher D Manning. Effective approaches to attention-based neural machine translation. arXiv preprint arXiv:1508.04025, 2015.

[25] Mitchell P Marcus, Mary Ann Marcinkiewicz, and Beatrice Santorini. Building a large annotated corpus of english: The penn treebank. Computational linguistics, 19(2):313–330, 1993.
[25] Mitchell P Marcus, Mary Ann Marcinkiewicz, and Beatrice Santorini. Building a large annotated corpus of english: The penn treebank. Computational linguistics, 19(2):313–330, 1993.

[26] David McClosky, Eugene Charniak, and Mark Johnson. Effective self-training for parsing. In Proceedings of the Human Language Technology Conference of the NAACL, Main Conference, pages 152–159. ACL, June 2006.
[26] David McClosky, Eugene Charniak, and Mark Johnson. Effective self-training for parsing. In Proceedings of the Human Language Technology Conference of the NAACL, Main Conference, pages 152–159. ACL, June 2006.

[27] Ankur Parikh, Oscar Täckström, Dipanjan Das, and Jakob Uszkoreit. A decomposable attention model. In Empirical Methods in Natural Language Processing, 2016.
[27] Ankur Parikh, Oscar Täckström, Dipanjan Das, and Jakob Uszkoreit. A decomposable attention model. In Empirical Methods in Natural Language Processing, 2016.

[28] Romain Paulus, Caiming Xiong, and Richard Socher. A deep reinforced model for abstractive summarization. arXiv preprint arXiv:1705.04304, 2017.
[28] Romain Paulus, Caiming Xiong, and Richard Socher. A deep reinforced model for abstractive summarization. arXiv preprint arXiv:1705.04304, 2017.

[29] Slav Petrov, Leon Barrett, Romain Thibaux, and Dan Klein. Learning accurate, compact, and interpretable tree annotation. In Proceedings of the 21st International Conference on Computational Linguistics and 44th Annual Meeting of the ACL, pages 433–440. ACL, July 2006.
[29] Slav Petrov, Leon Barrett, Romain Thibaux, and Dan Klein. Learning accurate, compact, and interpretable tree annotation. In Proceedings of the 21st International Conference on Computational Linguistics and 44th Annual Meeting of the ACL, pages 433–440. ACL, July 2006.

[30] Ofir Press and Lior Wolf. Using the output embedding to improve language models. arXiv preprint arXiv:1608.05859, 2016.
[30] Ofir Press and Lior Wolf. Using the output embedding to improve language models. arXiv preprint arXiv:1608.05859, 2016.

[31] Rico Sennrich, Barry Haddow, and Alexandra Birch. Neural machine translation of rare words with subword units. arXiv preprint arXiv:1508.07909, 2015.
[31] Rico Sennrich, Barry Haddow, and Alexandra Birch. Neural machine translation of rare words with subword units. arXiv preprint arXiv:1508.07909, 2015.

[32] Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, and Jeff Dean. Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. arXiv preprint arXiv:1701.06538, 2017.
[32] Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, and Jeff Dean. Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. arXiv preprint arXiv:1701.06538, 2017.

[33] Nitish Srivastava, Geoffrey E Hinton, Alex Krizhevsky, Ilya Sutskever, and Ruslan Salakhutdinov. Dropout: a simple way to prevent neural networks from overfitting. Journal of Machine Learning Research, 15(1):1929–1958, 2014.
[33] Nitish Srivastava, Geoffrey E Hinton, Alex Krizhevsky, Ilya Sutskever, and Ruslan Salakhutdinov. Dropout: a simple way to prevent neural networks from overfitting. Journal of Machine Learning Research, 15(1):1929–1958, 2014.

[34] Sainbayar Sukhbaatar, Arthur Szlam, Jason Weston, and Rob Fergus. End-to-end memory networks. In C. Cortes, N. D. Lawrence, D. D. Lee, M. Sugiyama, and R. Garnett, editors, Advances in Neural Information Processing Systems 28, pages 2440-2448. Curran Associates, Inc., 2015.
[34] Sainbayar Sukhbaatar, Arthur Szlam, Jason Weston, and Rob Fergus. End-to-end memory networks. In C. Cortes, N. D. Lawrence, D. D. Lee, M. Sugiyama, and R. Garnett, editors, Advances in Neural Information Processing Systems 28, pages 2440-2448. Curran Associates, Inc., 2015.

[35] Ilya Sutskever, Oriol Vinyals, and Quoc VV Le. Sequence to sequence learning with neural networks. In Advances in Neural Information Processing Systems, pages 3104–3112, 2014.
[35] Ilya Sutskever, Oriol Vinyals, and Quoc VV Le. Sequence to sequence learning with neural networks. In Advances in Neural Information Processing Systems, pages 3104–3112, 2014.

[36] Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens, and Zbigniew Wojna. Rethinking the inception architecture for computer vision. CoRR, abs/1512.00567, 2015.
[36] Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens, and Zbigniew Wojna. Rethinking the inception architecture for computer vision. CoRR, abs/1512.00567, 2015.

[37] Vinyals & Kaiser, Koo, Petrov, Sutskever, and Hinton. Grammar as a foreign language. In Advances in Neural Information Processing Systems, 2015.
[37] Vinyals & Kaiser, Koo, Petrov, Sutskever, and Hinton. Grammar as a foreign language. In Advances in Neural Information Processing Systems, 2015.

[38] Yonghui Wu, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Wolfgang Macherey, Maxim Krikun, Yuan Cao, Qin Gao, Klaus Macherey, et al. Google’s neural machine translation system: Bridging the gap between human and machine translation. arXiv preprint arXiv:1609.08144, 2016.
[38] Yonghui Wu, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Wolfgang Macherey, Maxim Krikun, Yuan Cao, Qin Gao, Klaus Macherey, et al. Google’s neural machine translation system: Bridging the gap between human and machine translation. arXiv preprint arXiv:1609.08144, 2016.

[39] Jie Zhou, Ying Cao, Xuguang Wang, Peng Li, and Wei Xu. Deep recurrent models with fast-forward connections for neural machine translation. CoRR, abs/1606.04199, 2016.
[39] Jie Zhou, Ying Cao, Xuguang Wang, Peng Li, and Wei Xu. Deep recurrent models with fast-forward connections for neural machine translation. CoRR, abs/1606.04199, 2016.

[40] Muhua Zhu, Yue Zhang, Wenliang Chen, Min Zhang, and Jingbo Zhu. Fast and accurate shift-reduce constituent parsing. In Proceedings of the 51st Annual Meeting of the ACL (Volume 1: Long Papers), pages 434–443. ACL, August 2013.
[40] Muhua Zhu, Yue Zhang, Wenliang Chen, Min Zhang, and Jingbo Zhu. Fast and accurate shift-reduce constituent parsing. In Proceedings of the 51st Annual Meeting of the ACL (Volume 1: Long Papers), pages 434–443. ACL, August 2013.

[Image]

Figure 3: An example of the attention mechanism following long-distance dependencies in the encoder self-attention in layer 5 of 6. Many of the attention heads attend to a distant dependency of the verb ’making’, completing the phrase ’making...more difficult’. Attentions here shown only for the word ’making’. Different colors represent different heads. Best viewed in color.
圖 3：編碼器自我注意機制第 5 層（共 6 層）中，注意力機制追蹤長距離相依性的一個範例。許多注意力頭都關注動詞「making」的一個遠距離相依詞，完成了片語「making...more difficult」。此處僅顯示動詞「making」的注意力。不同顏色代表不同的頭。以彩色觀看效果最佳。

[Image]

Figure 4: Two attention heads, also in layer 5 of 6, apparently involved in anaphora resolution. Top: Full attentions for head 5. Bottom: Isolated attentions from just the word ’its’ for attention heads 5 and 6. Note that the attentions are very sharp for this word.
圖 4：同樣在第 5 層（共 6 層）的兩個注意力頭，顯然與回指解析有關。上圖：第 5 頭的完整注意力。下圖：僅針對單字「its」，第 5 和第 6 頭的獨立注意力。請注意，這個單字的注意力非常集中。

[Image]

Figure 5: Many of the attention heads exhibit behaviour that seems related to the structure of the sentence. We give two such examples above, from two different heads from the encoder self-attention at layer 5 of 6. The heads clearly learned to perform different tasks.
圖 5：許多注意力頭的行為似乎與句子的結構有關。我們上面給出了兩個這樣的例子，來自編碼器第 5 層（共 6 層）的兩個不同頭的自我注意力。這些頭顯然學會了執行不同的任務。
