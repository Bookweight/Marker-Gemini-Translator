---
title: "Neuro-MSBG"
field: "Papers"
status: "Imported"
created_date: 2026-01-12
pdf_link: "[[Neuro-MSBG.pdf]]"
tags: [paper, Papers]
---

# Neuro-MSBG: An End-to-End Neural Model for Hearing Loss Simulation
# Neuro-MSBG：一種用於聽力損失模擬的端到端神經模型

Hui-Guan Yuan*†, Ryandhimas E. Zezario*, Shafique Ahmed*, Hsin-Min Wang*, Kai-Lung Hua†‡, Yu Tsao*
*Academia Sinica, Taipei, Taiwan
†National Taiwan University of Science and Technology, Taipei, Taiwan
#Microsoft, Taipei, Taiwan
袁輝寰*†, Ryandhimas E. Zezario*, Shafique Ahmed*, 王新民*, 華開龍†‡, 曹昱*
*中央研究院, 台北, 台灣
†國立台灣科技大學, 台北, 台灣
#微軟, 台北, 台灣

**Abstract**-Hearing loss simulation models are essential for hearing aid deployment. However, existing models have high computational complexity and latency, which limits real-time applications, and lack direct integration with speech processing systems. To address these issues, we propose Neuro-MSBG, a lightweight end-to-end model with a personalized audiogram encoder for effective time-frequency modeling. Experiments show that Neuro-MSBG supports parallel inference and retains the intelligibility and perceptual quality of the original MSBG, with a Spearman's rank correlation coefficient (SRCC) of 0.9247 for Short-Time Objective Intelligibility (STOI) and 0.8671 for Perceptual Evaluation of Speech Quality (PESQ). Neuro-MSBG reduces simulation runtime by 46 times (from 0.970 seconds to 0.021 seconds for a 1 second input), further demonstrating its efficiency and practicality.
**摘要**-聽力損失模擬模型對於助聽器部署至關重要。然而，現有模型計算複雜度高、延遲長，限制了即時應用，且缺乏與語音處理系統的直接整合。為了解決這些問題，我們提出了 Neuro-MSBG，這是一個輕量級的端到端模型，帶有個人化的聽力圖編碼器，可進行有效的時頻建模。實驗表明，Neuro-MSBG 支援平行推論，並保留了原始 MSBG 的可懂度和感知品質，其 Spearman 等級相關係數 (SRCC) 在短時客觀可懂度 (STOI) 上為 0.9247，在語音品質感知評估 (PESQ) 上為 0.8671。Neuro-MSBG 將模擬執行時間減少了 46 倍（從 0.970 秒降至 0.021 秒，對於 1 秒的輸入），進一步證明了其效率和實用性。

**Index Terms**-hearing loss model, mamba, differentiable framework, audiogram, real-time inference
**索引詞**-聽力損失模型、mamba、可微分框架、聽力圖、即時推論

## I. INTRODUCTION
## 一、緒論

Hearing loss simulation models aim to simulate how hearing impairment affects sound processing in the auditory system and have become essential tools in both research and evaluation. For example, the Clarity Challenge [1] uses the Moore, Stone, Baer, and Glasberg (MSBG) model [2]-[5] to simulate individual perceptual conditions based on audiograms. Similarly, the Cadenza Challenge [6] and the Clarity Challenge adopt perceptually grounded metrics such as the hearing-aid speech perception index (HASPI) [7], the hearing-aid speech quality index (HASQI) [8], and the hearing-aid audio quality index (HAAQI) [9], which embed auditory processing to assess quality and intelligibility under hearing loss conditions.
聽力損失模擬模型旨在模擬聽力障礙如何影響聽覺系統中的聲音處理，並已成為研究和評估中的重要工具。例如，Clarity Challenge [1] 使用 Moore、Stone、Baer 和 Glasberg (MSBG) 模型 [2]-[5] 來根據聽力圖模擬個人感知狀況。同樣地，Cadenza Challenge [6] 和 Clarity Challenge 採用了基於感知的指標，例如助聽器語音感知指數 (HASPI) [7]、助聽器語音品質指數 (HASQI) [8] 和助聽器音訊品質指數 (HAAQI) [9]，這些指標嵌入了聽覺處理，以評估在聽力損失條件下的品質和可懂度。

Existing hearing loss models are generally divided into two categories: physiological models and engineering-oriented models. Physiological models, such as the model proposed in [10] and the transmission-line (TL) cochlear model [11], are designed to accurately model cochlear mechanics, but their complexity limits real-time integration. In contrast, engineering-oriented models, such as the Hohmann filter-bank [12], the Auditory Toolbox [13], and MSBG, balance deployment practicality with perceptual accuracy and provide computational stability, making them well suited for real-time speech processing. Among engineering-oriented models, MSBG [2] is currently the most widely used. It simulates sensorineural hearing loss based on audiograms and reproduces key perceptual effects. Despite its widespread adoption in both academic and practical applications, MSBG has two major limitations: (i) it does not support parallel processing, which reduces its applicability to real-time speech systems; and (ii) it relies on multiple filtering stages, which introduces variable delays. These limitations restrict the integration of MSBG into end-to-end learning frameworks and reduce its effectiveness in real-time or large-scale speech processing.
現有的聽力損失模型大致可分為兩類：生理模型和工程導向模型。生理模型，例如 [10] 中提出的模型和傳輸線 (TL) 耳蝸模型 [11]，旨在精確模擬耳蝸力學，但其複雜性限制了即時整合。相比之下，工程導向模型，如 Hohmann 濾波器組 [12]、Auditory Toolbox [13] 和 MSBG，在部署實用性與感知準確性之間取得了平衡，並提供計算穩定性，使其非常適合即時語音處理。在工程導向模型中，MSBG [2] 是目前應用最廣泛的模型。它基於聽力圖模擬感音神經性聽力損失，並重現了關鍵的感知效果。儘管 MSBG 在學術和實際應用中被廣泛採用，但它有兩個主要限制：(i) 它不支援平行處理，這降低了其在即時語音系統中的適用性；(ii) 它依賴多個濾波階段，這會引入可變的延遲。這些限制阻礙了 MSBG 整合到端到端學習框架中，並降低了其在即時或大規模語音處理中的有效性。

Recent studies [14], [15] have attempted to simplify both physiological and engineering-oriented models for real-time applications. For example, CoNNear [14] simplifies the TL cochlear model and supports real-time simulation of auditory nerve responses. Similarly, P Leer et al. [16] trained neural models to emulate the Verhulst auditory periphery model for varying hearing-loss profiles. However, the lack of waveform-level output generation prevents waveform-level supervision and reduces its applicability to speech processing and hearing aid systems. For engineering-oriented models, the Wakayama University Hearing Impairment Simulator (WHIS) [15] addresses the latency and computational cost issues associated with MSBG. WHIS first computes the cochlear excitation pattern of a target hearing-impaired individual using a Gammachirp filterbank, and then dynamically generates a time-varying minimum-phase filter based on this pattern to transform normal-hearing speech into its hearing-loss-simulated counterpart. This method reduces the processing time for one second of speech to approximately 10 milliseconds while maintaining near-perfect temporal alignment with the original waveform. Despite its efficiency, WHIS still relies on frame-by-frame computation of infinite impulse response (IIR) filter coefficients and dynamic gain selection, and has not been optimized for vectorized or parallel processing. Therefore, its integration into an end-to-end deep learning framework remains challenging, and joint optimization with compensation models remains an open research direction.
最近的研究 [14], [15] 試圖簡化生理模型和工程導向模型以用於即時應用。例如，CoNNear [14] 簡化了 TL 耳蝸模型，並支援聽覺神經反應的即時模擬。同樣地，P Leer 等人 [16] 訓練了神經模型來模擬 Verhulst 聽覺周邊模型，以適應不同的聽力損失情況。然而，由於缺乏波形級的輸出生成，無法進行波形級的監督，這降低了其在語音處理和助聽器系統中的適用性。對於工程導向模型，和歌山大學聽力障礙模擬器 (WHIS) [15] 解決了與 MSBG 相關的延遲和計算成本問題。WHIS 首先使用 Gammachirp 濾波器組計算目標聽障人士的耳蝸激發模式，然後基於此模式動態生成一個時變最小相位濾波器，將正常聽力的語音轉換為其聽力損失模擬的對應物。此方法將一秒語音的處理時間縮短至約 10 毫秒，同時與原始波形保持近乎完美的時間對齊。儘管效率高，WHIS 仍依賴於無限脈衝響應 (IIR) 濾波器係數和動態增益選擇的逐幀計算，並且尚未針對向量化或平行處理進行優化。因此，將其整合到端到端深度學習框架中仍然具有挑戰性，與補償模型的聯合優化仍是一個懸而未決的研究方向。

With the increasing use of differentiable hearing loss models in hearing aid compensation, optimizing their design and performance has become a research focus. In physiological modeling, the auditory nerve responses generated by CoNNear have been used as loss functions to guide the training of compensation models [17]–[19]. This approach attempts to make the neural responses of hearing-impaired people when receiving compensated speech similar to the neural responses of normal-hearing people when listening to the original signal. In engineering-oriented models, Tu et al. proposed the Differentiable Hearing Aid Speech Processing (DHASP) framework [20], which reimplements the auditory processing pipeline in HASPI [7] and uses differentiable modules for backpropagation. Tu et al. also introduced a differentiable version of the MSBG model and applied it to the training of hearing aid algorithms [21]. These engineering-oriented approaches typically consist of differentiable finite impulse response (FIR) filters and audio processing steps designed to approximate auditory mechanisms.
隨著可微分聽力損失模型在助聽器補償中的應用日益增多，優化其設計和性能已成為一個研究焦點。在生理模型中，由 CoNNear 生成的聽覺神經反應已被用作損失函數，以指導補償模型的訓練 [17]–[19]。這種方法試圖使聽障人士在接收補償後語音時的神經反應，與正常聽力人士在聆聽原始信號時的神經反應相似。在工程導向模型中，Tu 等人提出了可微分助聽器語音處理 (DHASP) 框架 [20]，該框架重新實現了 HASPI [7] 中的聽覺處理流程，並使用可微分模組進行反向傳播。Tu 等人還介紹了 MSBG 模型的可微分版本，並將其應用於助聽器演算法的訓練 [21]。這些工程導向的方法通常由可微分的有限脈衝響應 (FIR) 濾波器和旨在近似聽覺機制的音訊處理步驟組成。

Despite some progress in differentiable engineering-oriented hearing loss models, most efforts have focused on magnitude-domain simulation, with limited attention paid to the role of phase information. Meanwhile, recent advances in speech enhancement have highlighted the importance of phase modeling for perceptual quality. For instance, MP-SENet [22] adopts a joint enhancement strategy for both magnitude and phase spectra, achieving significantly better performance than traditional magnitude-only methods and highlighting the importance of incorporating phase modeling. Inspired by these findings, we investigate the role of phase information in hearing loss simulation and propose Neuro-MSBG, an end-to-end fully differentiable hearing loss model. Neuro-MSBG outputs simulated audio in the waveform domain, which can be directly integrated with modern speech enhancement systems that rely on waveform-based losses and evaluation metrics (e.g., mean squared error (MSE), short-time objective intelligibility (STOI) [23], and perceptual evaluation of speech quality (PESQ) [24]). It also supports noisy speech input, further enhancing its practical applicability. Experimental results show that the addition of phase processing significantly improves the fidelity of MSBG hearing loss simulation, highlighting the importance of phase modeling in replicating authentic auditory perception. The main contributions of our model are as follows:
儘管在可微分工程導向的聽力損失模型方面取得了一些進展，但大多數努力都集中在幅度域的模擬上，而對相位資訊的作用關注有限。與此同時，語音增強領域的最新進展突顯了相位建模對感知品質的重要性。例如，MP-SENet [22] 採用了幅度和相位譜的聯合增強策略，取得了比傳統僅幅度方法顯著更優的性能，並突顯了納入相位建模的重要性。受這些發現的啟發，我們研究了相位資訊在聽力損失模擬中的作用，並提出了 Neuro-MSBG，這是一個端到端、完全可微分的聽力損失模型。Neuro-MSBG 在波形域中輸出模擬音訊，可直接與依賴基於波形的損失和評估指標（例如，均方誤差 (MSE)、短時客觀可懂度 (STOI) [23] 和語音品質感知評估 (PESQ) [24]）的現代語音增強系統整合。它還支援帶噪語音輸入，進一步增強了其實際適用性。實驗結果表明，增加相位處理顯著提高了 MSBG 聽力損失模擬的保真度，突顯了相位建模在複製真實聽覺感知中的重要性。我們模型的主要貢獻如下：

*   **Parallelizable and lightweight simulation:** Neuro-MSBG achieves parallel inference, reducing the simulation time for one second of audio from 0.970 seconds in the original MSBG to 0.021 seconds, a 46× speedup.
*   **可平行化和輕量級模擬：** Neuro-MSBG 實現了平行推論，將一秒音訊的模擬時間從原始 MSBG 的 0.970 秒減少到 0.021 秒，速度提高了 46 倍。

*   **Seamless integration with end-to-end speech systems:** By resolving the delay issues inherent in the original MSBG, Neuro-MSBG can be integrated into modern speech compensator training pipelines.
*   **與端到端語音系統的無縫整合：** 透過解決原始 MSBG 固有的延遲問題，Neuro-MSBG 可以整合到現代語音補償器訓練流程中。

*   **Phase-aware modeling:** By incorporating phase information, Neuro-MSBG maintains the intelligibility and perceptual quality of the original MSBG, achieving a Spearman's rank correlation coefficient (SRCC) of 0.9247 for STOI and 0.8671 for PESQ.
*   **相位感知建模：** 透過納入相位資訊，Neuro-MSBG 保持了原始 MSBG 的可懂度和感知品質，在 STOI 上實現了 0.9247 的 Spearman 等級相關係數 (SRCC)，在 PESQ 上實現了 0.8671。

At the end of this paper, we also demonstrate the preliminary integration of Neuro-MSBG into a speech compensator pipeline, thus confirming its practicality as a differentiable hearing loss simulation module. The remainder of this paper is organized as follows. Section II presents the proposed method. Section III describes the experimental setup and results. Finally, Section IV presents conclusions.
在本文末尾，我們還展示了將 Neuro-MSBG 初步整合到語音補償器流程中，從而證實其作為可微分聽力損失模擬模組的實用性。本文其餘部分的組織如下。第二節介紹了所提出的方法。第三節描述了實驗設置和結果。最後，第四節提出了結論。

## II. METHODOLOGY
## 二、方法論

This section introduces the model architecture and training criteria of Neuro-MSBG. As shown in Fig. 1, the model takes normal speech signals and monaural audiograms as input. The audiogram is transformed into personalized hearing features through the Audiogram Encoder, while the speech signal is converted into the time-frequency domain through Short-Time Fourier Transform (STFT) to obtain magnitude and phase features. These three types of features, including personalized hearing features, magnitude features, and phase features, are concatenated and then input into the Neural Network Block (NN Block). The network then branches into two decoders: the Magnitude Mask Decoder and the Phase Decoder, which respectively predict the magnitude and phase shifts associated with hearing loss. Finally, the predicted magnitude and phase are combined to reconstruct the speech signal perceived by the hearing-impaired listener through inverse STFT.
本節介紹 Neuro-MSBG 的模型架構和訓練標準。如圖 1 所示，該模型以正常語音信號和單耳聽力圖作為輸入。聽力圖通過聽力圖編碼器轉換為個人化的聽力特徵，而語音信號則通過短時傅立葉變換 (STFT) 轉換到時頻域以獲得幅度和相位特徵。這三種類型的特徵，包括個人化聽力特徵、幅度特徵和相位特徵，被串聯起來，然後輸入到神經網絡區塊 (NN Block)。該網絡隨後分支為兩個解碼器：幅度掩碼解碼器和相位解碼器，它們分別預測與聽力損失相關的幅度和相位偏移。最後，預測的幅度和相位被結合起來，通過逆 STFT 重建聽障聽者感知的語音信號。

### A. Neuro-MSBG

Our model adopts an architecture based on MP-SENet [22] and the advanced SE-Mamba framework [25]. This design is inspired by our experimental findings that phase information is critical for accurate hearing loss simulation (see Section III for details). To evaluate how well different neural modules capture spectral and temporal cues, we replace the original attention-based components with alternatives such as LSTM, Transformer, CNN, and Mamba blocks. Given Mamba's efficiency in modeling long sequences and its low latency, we also control the number of parameters to ensure that the model remains lightweight and effective.
我們的模型採用了基於 MP-SENet [22] 和先進的 SE-Mamba 框架 [25] 的架構。這個設計的靈感來自於我們的實驗發現，即相位資訊對於準確的聽力損失模擬至關重要（詳見第三節）。為了評估不同神經模組捕捉光譜和時間線索的能力，我們用 LSTM、Transformer、CNN 和 Mamba 等替代方案取代了原始的基於注意力的組件。鑑於 Mamba 在建模長序列方面的效率和低延遲性，我們還控制了參數數量，以確保模型保持輕量和有效。

For audiogram integration, previous methods typically concatenate the audiogram representation along the frequency dimension. In contrast, we found that treating the audiogram as an additional input channel in addition to magnitude and phase produces more stable and effective results. This channel-based integration may enable the model to receive more consistent, spatially aligned conditioning across layers, thereby improving its ability to modulate internal feature representations.
對於聽力圖的整合，以前的方法通常是沿著頻率維度串聯聽力圖表示。相比之下，我們發現將聽力圖作為除幅度和相位之外的附加輸入通道處理，會產生更穩定和有效的結果。這種基於通道的整合可以使模型在各層之間接收到更一致、空間對齊的條件，從而提高其調節內部特徵表示的能力。

1) Audiogram Encoder: To incorporate personalized hearing profiles, we design a lightweight Audiogram Encoder that transforms the audiogram a ∈ RB×8 into a frequency-aligned representation aenc ∈ RB×F, where B denotes the batch size, and F = 201 matches the STFT resolution. The transformation process is defined as:
1) 聽力圖編碼器：為了納入個人化的聽力配置文件，我們設計了一個輕量級的聽力圖編碼器，它將聽力圖 a ∈ RB×8 轉換為頻率對齊的表示 aenc ∈ RB×F，其中 B 表示批次大小，F = 201 匹配 STFT 分辨率。轉換過程定義如下：

aenc W Flatten (AvgPool (σ (Conv(a)))), (1)
aenc W Flatten (AvgPool (σ (Conv(a)))), (1)

where Conv is a 1D convolution layer, σ denotes a ReLU activation, and W is a linear projection matrix.
其中 Conv 是一維卷積層，σ 表示 ReLU 激活函數，W 是線性投影矩陣。

The encoded vector is then broadcast along the time axis and concatenated with the magnitude and phase features to form the three-channel input RB×3×T×F of the DenseEncoder and NN Block. This channel-based integration enables the model to consistently inject hearing-profile information in both time and frequency dimensions.
編碼後的向量隨後沿時間軸廣播，並與幅度和相位特徵串聯，形成 DenseEncoder 和 NN Block 的三通道輸入 RB×3×T×F。這種基於通道的整合使模型能夠在時間和頻率維度上持續注入聽力配置文件資訊。

[Image]
Fig. 1. Overview of the proposed Neuro-MSBG framework.
圖 1. 提議的 Neuro-MSBG 框架概覽。

2) Neural Network Blocks: To capture the temporal and spectral structure of hearing-loss-affected speech, we design and compare multiple NN Blocks, each of which adopts a dual-path architecture to process the time and frequency dimensions separately. The input tensor is first rearranged and reshaped for temporal modeling, followed by a similar process for frequency modeling. Each path contains residual connections and a ConvTransposeld layer to restore the original shape to ensure compatibility with subsequent modules.
2) 神經網路區塊：為了捕捉受聽力損失影響的語音的時間和頻譜結構，我們設計並比較了多個神經網路區塊，每個區塊都採用雙路徑架構來分別處理時間和頻率維度。輸入張量首先被重新排列和重塑以進行時間建模，然後是類似的頻率建模過程。每個路徑都包含殘差連接和一個 ConvTransposeld 層，以恢復原始形狀，確保與後續模組的兼容性。

In this unified framework, we replace the core time-frequency module with one of the following four alternatives:
在這個統一的框架中，我們用以下四種選擇之一取代核心的時頻模組：

*   **Mamba Block:** Combined with bidirectional Mamba modules, time and frequency are modeled separately to provide efficient long-range dependency modeling.
*   **Mamba 區塊：** 結合雙向 Mamba 模組，分別對時間和頻率進行建模，以提供高效的長程依賴建模。
*   **Transformer Block:** Transformer encoder layers are applied to both axes, and global attention is used to capture contextual information.
*   **Transformer 區塊：** 將 Transformer 編碼器層應用於兩個軸，並使用全局注意力來捕捉上下文資訊。
*   **LSTM Block:** Bidirectional LSTM is used to model sequential patterns, and then linear projection is performed to maintain dimensional consistency.
*   **LSTM 區塊：** 使用雙向 LSTM 對序列模式進行建模，然後執行線性投影以保持維度一致性。
*   **CNN Block:** One-dimensional convolution is used to extract local features, followed by channel expansion, activation, and residual fusion.
*   **CNN 區塊：** 使用一維卷積提取局部特徵，然後進行通道擴展、激活和殘差融合。

This dual-axis design forms a flexible and stable framework for the simulation of hearing loss. It also allows for systematic comparison across architectures, demonstrating Mamba's potential for low-latency, high-fidelity speech modeling.
這種雙軸設計為聽力損失的模擬提供了一個靈活穩定的框架。它還允許跨架構進行系統性比較，展示了 Mamba 在低延遲、高保真語音建模方面的潛力。

### B. Training Criteria

We adopt a multi-objective loss to jointly supervise spectral accuracy, phase consistency, and time-domain fidelity. The magnitude loss LMag is defined as the MSE between the predicted magnitude m and the ground-truth magnitude m:
我們採用多目標損失來共同監督頻譜準確性、相位一致性和時域保真度。幅度損失 LMag 定義為預測幅度 m 和真實幅度 m 之間的均方誤差 (MSE)：

LMag = (1/N) * Σ ||mi - mi||² (2)
LMag = (1/N) * Σ ||mi - mi||² (2)

where N is the number of training samples. The phase loss Lpha is inspired by the anti-wrapping strategy proposed in [26] and consists of three components: the instantaneous phase loss LIP, the group delay loss LGD, and the integrated absolute frequency loss LIAF, defined as:
其中 N 是訓練樣本的數量。相位損失 Lpha 的靈感來自 [26] 中提出的反纏繞策略，由三個部分組成：瞬時相位損失 LIP、群延遲損失 LGD 和積分絕對頻率損失 LIAF，定義如下：

LIP = Ep,p [|| faw (p-p)||1], (3)
LIP = Ep,p [|| faw (p-p)||1], (3)

LGD = Ep,p [|| faw (ΔF(p-p))||1], (4)
LGD = Ep,p [|| faw (ΔF(p-p))||1], (4)

LIAF = Ep,p [|| faw (ΔT(p - p))||1], (5)
LIAF = Ep,p [|| faw (ΔT(p - p))||1], (5)

LPha = LIP + LGD + LIAF, (6)
LPha = LIP + LGD + LIAF, (6)

where faw() denotes the anti-wrapping function used to mitigate 2π discontinuity, ΔF(·) and ΔT(·) represent the first-order differences of the phase error along the frequency axis and time axis, respectively. The complex loss Lcom measures the MSE between the predicted complex spectrogram ĉ and the ground-truth complex spectrogram c (including both real and imaginary parts):
其中 faw() 表示用於減輕 2π 不連續性的反纏繞函數，ΔF(·) 和 ΔT(·) 分別表示相位誤差沿頻率軸和時間軸的一階差分。複數損失 Lcom 測量預測的複數譜圖 ĉ 和真實複數譜圖 c（包括實部和虛部）之間的均方誤差 (MSE)：

LCom = 2 * (1/N) * Σ ||ĉi - ci||² (7)
LCom = 2 * (1/N) * Σ ||ĉi - ci||² (7)

The time-domain loss LTime is calculated as the L1 distance between the predicted waveform x̂ and the reference waveform x to preserve temporal fidelity:
時域損失 LTime 計算為預測波形 x̂ 和參考波形 x 之間的 L1 距離，以保持時間保真度：

LTime = (1/N) * Σ ||x̂i - xi||₁ (8)
LTime = (1/N) * Σ ||x̂i - xi||₁ (8)

The total training loss is a weighted sum of all components:
總訓練損失是所有組分的加權和：

LTotal = λMag * LMag + λPha * LPha + λCom * LCom + λTime * LTime, (9)
LTotal = λMag * LMag + λPha * LPha + λCom * LCom + λTime * LTime, (9)

where each λ is a tunable scalar weight used to balance the contribution of each loss term.
其中每個 λ 都是一個可調的標量權重，用於平衡每個損失項的貢獻。

## III. EXPERIMENT
## 三、實驗

The dataset comprises 12,396 clean utterances (11,572 for training and 824 for testing) from VoiceBank [27] and their noisy counterparts from VoiceBank-DEMAND [28], created by mixing each clean utterance with one randomly selected noise type from 10 DEMAND recordings [29] and one signal-to-noise-ratio (SNR) level. To ensure generalization, 8 noise types are used for training and 2 disjoint types for testing, with SNR levels drawn from {0, 5, 10, 15} dB for training and {2.5, 7.5, 12.5, 17.5} dB for testing. Each utterance is paired with two monaural audiograms, representing different levels of hearing loss, ranging from mild to severe. Consequently, the training set is expanded to 11, 572 × 2 × 2 = 46, 288 samples, and the test set is expanded to 824 × 2 × 2 = 3,296 samples. Speech data and audiograms are disjoint across training and testing splits, ensuring that the model is evaluated on unseen utterances and unseen hearing-loss profiles.
資料集包含來自 VoiceBank [27] 的 12,396 個乾淨語音（11,572 個用於訓練，824 個用於測試）及其來自 VoiceBank-DEMAND [28] 的帶噪聲對應物，這些對應物是通過將每個乾淨語音與從 10 個 DEMAND 錄音 [29] 中隨機選擇的一種類型噪聲和一個信噪比 (SNR) 等級混合而成的。為確保泛化性，訓練使用 8 種類型噪聲，測試使用 2 種不相交的類型，訓練的 SNR 等級來自 {0, 5, 10, 15} dB，測試的 SNR 等級來自 {2.5, 7.5, 12.5, 17.5} dB。每個語音都與兩個單耳聽力圖配對，代表從輕度到重度的不同聽力損失水平。因此，訓練集擴展到 11,572 × 2 × 2 = 46,288 個樣本，測試集擴展到 824 × 2 × 2 = 3,296 個樣本。語音數據和聽力圖在訓練和測試分割中是不相交的，確保模型在未見過的語音和未見過的聽力損失配置文件上進行評估。

Specifically, this study conducted a series of experiments covering five main aspects: (i) comparison of Neuro-MSBG with different architectures and monolithic baselines (Table I); (ii) runtime evaluation (Table II); (iii) validation of the necessity of phase prediction (Table III); (iv) comparison of audiogram integration strategies (Table IV); and (v) application of Neuro-MSBG in a speech compensator (Table V). In all quantitative tables, bold highlights the best-performing model and underline marks the second best. All experiments were performed on a single NVIDIA RTX 3090. The models were trained for 200 epochs with a batch size of 3, an initial learning rate of 0.0005, and the AdamW optimizer.
具體而言，本研究進行了一系列實驗，涵蓋五個主要方面：(i) Neuro-MSBG 與不同架構和單體基線的比較（表 I）；(ii) 執行時間評估（表 II）；(iii) 相位預測必要性的驗證（表 III）；(iv) 聽力圖整合策略的比較（表 IV）；以及 (v) Neuro-MSBG 在語音補償器中的應用（表 V）。在所有量化表中，粗體突出顯示性能最佳的模型，下劃線標記次佳的模型。所有實驗均在單一 NVIDIA RTX 3090 上執行。模型使用批次大小為 3、初始學習率為 0.0005 的 AdamW 優化器進行了 200 個週期的訓練。

### A. Data Preparation

We use the MSBG model to simulate hearing loss by applying an ear-specific gain curve to each input, resulting in single-channel impaired speech. However, due to the multi-stage filtering in MSBG, the output may exhibit unpredictable delay relative to the original signal. Such misalignment is undesirable for downstream tasks, such as speech enhancement or intelligibility assessment.
我們使用 MSBG 模型通過對每個輸入應用特定耳朵的增益曲線來模擬聽力損失，從而產生單通道受損語音。然而，由於 MSBG 中的多級濾波，輸出相對於原始信號可能表現出不可預測的延遲。這種未對準對於下游任務（例如語音增強或可懂度評估）是不利的。

To estimate and correct this delay, we generate an auxiliary reference signal along with the main audio during the MSBG process [1]. This auxiliary signal is a silent waveform with the same length and sampling rate as the input, used solely for delay tracking. A unit impulse is inserted into this signal, which is defined as:
為了估計和校正這種延遲，我們在 MSBG 處理過程中與主音頻一起生成一個輔助參考信號 [1]。這個輔助信號是一個與輸入長度和採樣率相同的靜音波形，專門用於延遲跟踪。一個單位脈衝被插入到這個信號中，定義為：

δ[n - k] = {1, if n = k; 0, otherwise}, (10)
δ[n - k] = {1, 如果 n = k; 0, 否則}, (10)

where n denotes the discrete time index, and k = Fs/2 is the sample position of the impulse, where Fs represents the sampling rate in Hz.
其中 n 表示離散時間索引，k = Fs/2 是脈衝的樣本位置，其中 Fs 表示以 Hz 為單位的採樣率。

The impulse is inserted at the midpoint of the auxiliary reference signal. After MSBG simulation, we compare the pre/post impulse positions to estimate the delay introduced by MSBG. We then use the estimated delay to time-align the original normal-hearing input, the clean reference, and the impaired output to ensure accurate evaluation through metrics such as STOI and PESQ that are highly sensitive to timing errors. Each training instance consists of: 1) the single-ear impaired speech, 2) the aligned clean reference, 3) the aligned normal-hearing input, and 4) the associated 8-dimensional audiogram vector. Fig. 2 shows the waveform alignment before and after the delay-compensation shift.
脈衝被插入到輔助參考信號的中點。在 MSBG 模擬之後，我們比較脈衝前後的位置以估計 MSBG 引入的延遲。然後，我們使用估計的延遲來時間對齊原始的正常聽力輸入、乾淨的參考和受損的輸出，以確保通過對時間誤差高度敏感的指標（如 STOI 和 PESQ）進行準確評估。每個訓練實例包括：1) 單耳受損語音，2) 對齊的乾淨參考，3) 對齊的正常聽力輸入，以及 4) 相關的 8 維聽力圖向量。圖 2 顯示了延遲補償前後的波形對齊情況。

### B. Experimental Results

In the early stages of our experiment, we used monolithic models with unified architectures such as CNN, LSTM, and Transformer for hearing simulation. However, due to limited performance, we subsequently adopted the Neuro-MSBG framework, replacing only the TF-NN block with different architectures, including CNN, LSTM, Transformer, and Mamba. The results are shown in Table I. To ensure a fair comparison across different architectures, the number of parameters of all models was set to be roughly the same. From Table I, we observe that the Neuro-MSBG variants consistently outperform the monolithic baselines on unseen test data. Among them, Neuro-MSBG with Mamba achieves the best performance across all evaluation metrics.
在我們實驗的早期階段，我們使用了具有統一架構的單體模型，例如 CNN、LSTM 和 Transformer 進行聽力模擬。然而，由於性能有限，我們隨後採用了 Neuro-MSBG 框架，僅用不同的架構（包括 CNN、LSTM、Transformer 和 Mamba）替換了 TF-NN 模塊。結果如表 I 所示。為確保跨不同架構的公平比較，所有模型的參數數量都設置為大致相同。從表 I 中，我們觀察到 Neuro-MSBG 的變體在未見過的測試數據上始終優於單體基線。其中，帶有 Mamba 的 Neuro-MSBG 在所有評估指標上都取得了最佳性能。

[Image]
TABLE I
[表一]
PERFORMANCE COMPARISON BETWEEN MONOLITHIC MODELS AND OUR MODULAR NEURO-MSBG FRAMEWORK WITH DIFFERENT TF-NN BLOCKS.
單體模型與我們的模塊化 NEURO-MSBG 框架在不同 TF-NN 塊下的性能比較。

[Image]
Fig. 2. Waveform alignment before and after shifting. In A, the MSBG-processed signal (red) shows a clear delay relative to the original input (blue). In B, the waveforms are time-aligned using impulse-based method, allowing for a fair and accurate assessment of the effect of MSBG.
圖 2. 波形在移位前後的對齊。在 A 中，MSBG 處理的信號（紅色）相對於原始輸入（藍色）顯示出明顯的延遲。在 B 中，使用基於脈衝的方法對波形進行時間對齊，從而可以對 MSBG 的效果進行公平準確的評估。

Next, we compare Neuro-MSBG with an existing neural network-based model, CoNNear, in terms of model size and inference time. Although the two models have different goals—CoNNear simulates physiologically grounded auditory nerve responses, while our framework focuses on perceptual signal transformation—the comparison is appropriate given their common goal of real-time hearing loss modeling. As shown in Table II, CoNNear has approximately 11.7 million parameters, while Neuro-MSBG has only 1.45 million parameters. In addition to its lightweight architecture, Neuro-MSBG supports noisy input conditions and accommodates a wide range of audiogram configurations, providing additional advantages for practical applications involving diverse acoustic environments and personalized hearing loss profiles.
接下來，我們在模型大小和推論時間方面將 Neuro-MSBG 與現有的基於神經網絡的模型 CoNNear 進行比較。儘管這兩個模型有不同的目標——CoNNear 模擬生理上可靠的聽覺神經反應，而我們的框架專注於感知信號轉換——但考慮到它們在即時聽力損失建模方面的共同目標，這種比較是恰當的。如表 II 所示，CoNNear 約有 1170 萬個參數，而 Neuro-MSBG 只有 145 萬個參數。除了其輕量級架構外，Neuro-MSBG 還支持嘈雜的輸入條件，並能適應廣泛的聽力圖配置，為涉及多樣化聲學環境和個人化聽力損失配置的實際應用提供了額外的優勢。

Table II also shows the inference time of different models. MSBG does not support parallel processing and cannot be executed on GPU; therefore, the corresponding GPU column is marked as NA. In contrast, Neuro-MSBG (Mamba) leverages a CUDA-accelerated selective scan kernel for core operations, which currently only supports GPU execution. Therefore, only the inference time on GPU is reported. In terms of inference time, Neuro-MSBG (Mamba) achieves about 46× speedup on GPU over MSBG on CPU. For CPU-executable variants such as Neuro-MSBG (LSTM), the inference time is 0.016 seconds, which is 60× faster than MSBG's 0.970 seconds. In addition, we also implemented and evaluated the CoNNear model. Despite its larger parameter size (11.7 million), it shows fast inference in our computation environment, with a GPU runtime of 0.099 seconds and a notably fast CPU runtime of 0.025 seconds. The faster CPU runtime of CoNNear than the GPU version is likely due to the batch size of 1 used in this experiment, which limits the benefits of GPU parallelism.
表 II 還顯示了不同模型的推論時間。MSBG 不支持平行處理，也無法在 GPU 上執行；因此，相應的 GPU 欄標記為 NA。相比之下，Neuro-MSBG (Mamba) 利用 CUDA 加速的選擇性掃描核心進行核心運算，目前僅支持 GPU 執行。因此，僅報告 GPU 上的推論時間。在推論時間方面，Neuro-MSBG (Mamba) 在 GPU 上的速度比 MSBG 在 CPU 上快約 46 倍。對於可在 CPU 上執行的變體，如 Neuro-MSBG (LSTM)，推論時間為 0.016 秒，比 MSBG 的 0.970 秒快 60 倍。此外，我們還實現並評估了 CoNNear 模型。儘管其參數尺寸較大（1170 萬），但在我們的計算環境中，它顯示出快速的推論，GPU 執行時間為 0.099 秒，CPU 執行時間更是快得驚人，為 0.025 秒。CoNNear 的 CPU 執行時間比 GPU 版本快，很可能是由於本實驗中使用的批次大小為 1，這限制了 GPU 並行化的好處。

### C. Ablation Study

Previous approaches to modeling hearing loss typically predict only the magnitude spectrum while reusing the input phase for waveform reconstruction. We initially adopted this approach; however, our empirical analysis revealed that phase information plays a crucial role in MSBG-based simulation. We conducted an ablation study using Neuro-MSBG (Mamba). As shown in Table III, predicting both magnitude and phase substantially outperforms magnitude-only prediction in all metrics (STOI MSE, PESQ MSE, and waveform-level MSE). Specifically, the STOI MSE decreased from 0.0041 to 0.0006, indicating a notable improvement in intelligibility, while the PESQ MSE decreased from 2.3579 to 0.0782, reflecting improved perceptual quality. At the waveform level, the MSE decreased from 0.0986 to 0.0669, confirming that phase modeling is critical for both perceptual fidelity and accurate signal reconstruction.
以前的聽力損失建模方法通常只預測幅度譜，而重用輸入相位進行波形重建。我們最初也採用了這種方法；然而，我們的實證分析顯示，相位資訊在基於 MSBG 的模擬中扮演著至關重要的角色。我們使用 Neuro-MSBG (Mamba) 進行了一項消融研究。如表 III 所示，同時預測幅度和相位在所有指標（STOI MSE、PESQ MSE 和波形級 MSE）上都顯著優於僅預測幅度。具體來說，STOI MSE 從 0.0041 降至 0.0006，表明可懂度有顯著提高；而 PESQ MSE 從 2.3579 降至 0.0782，反映了感知品質的改善。在波形級別，MSE 從 0.0986 降至 0.0669，證實了相位建模對於感知保真度和準確信號重建都至關重要。

[Image]
TABLE II
[表二]
COMPARISON OF RUNTIME OF MSBG AND NEURO-MSBG ON DIFFERENT DEVICES.
MSBG 和 NEURO-MSBG 在不同設備上的運行時間比較。

[Image]
TABLE III
[表三]
PERFORMANCE COMPARISON OF NEURO-MSBG MODELS WITH AND WITHOUT PHASE PREDICTION.
有和沒有相位預測的 NEURO-MSBG 模型性能比較。

Furthermore, in many speech-related applications involving audiograms, a common practice is to concatenate the audiogram vector with the audio features before feeding them into the model. We initially adopted this simple approach, but found that it could not effectively capture the relationship between hearing profiles and spectral features. To address this issue, we introduced a lightweight Audiogram Encoder that transforms the 8-dimensional audiogram vector into a frequency-aligned representation. This representation is appended as a third channel along with the magnitude and phase features. As shown in Table IV, incorporating the Audiogram Encoder leads to consistent reduction in STOI MSE, PESQ MSE, and waveform-level MSE, demonstrating its effectiveness in integrating personalized hearing profiles to improve hearing loss modeling.
此外，在許多涉及聽力圖的語音相關應用中，常見的做法是將聽力圖向量與音頻特徵串聯後再輸入模型。我們最初採用了這種簡單的方法，但發現它無法有效捕捉聽力配置文件和頻譜特徵之間的關係。為了解決這個問題，我們引入了一個輕量級的聽力圖編碼器，將 8 維的聽力圖向量轉換為頻率對齊的表示。這個表示作為第三個通道，與幅度和相位特徵一起附加。如表 IV 所示，納入聽力圖編碼器可以持續降低 STOI MSE、PESQ MSE 和波形級 MSE，證明其在整合個人化聽力配置文件以改善聽力損失建模方面的有效性。

### D. Qualitative Evaluation

To evaluate the model's performance in reconstructing hearing-loss-affected speech, we compare the log-magnitude spectrograms of speech outputs of seven models and ground-truth speech (Fig. 3). Among them, Neuro-MSBG (Mamba) produces the most accurate reconstruction, preserving the harmonic structure and high-frequency energy. Neuro-MSBG with CNN, LSTM, and Transformer Blocks also preserve key spectral features but exhibit some energy imbalance or mild distortion. In contrast, the baseline models introduce more obvious artifacts: the CNN and LSTM variants lack clarity and high-frequency content, while the Transformer variant has difficulty reconstructing an accurate spectrogram.
為了評估模型在重建受聽力損失影響的語音方面的性能，我們比較了七個模型的語音輸出和真實語音的對數幅度譜圖（圖 3）。其中，Neuro-MSBG (Mamba) 產生了最準確的重建，保留了諧波結構和高頻能量。帶有 CNN、LSTM 和 Transformer 模塊的 Neuro-MSBG 也保留了關鍵的光譜特徵，但表現出一些能量不平衡或輕微失真。相比之下，基線模型引入了更明顯的偽影：CNN 和 LSTM 變體缺乏清晰度和高頻內容，而 Transformer 變體難以重建準確的譜圖。

### E. Training a Compensator with Neuro-MSBG

To advance end-to-end hearing loss compensation, recent work (e.g., NeuroAMP [30]) has integrated audiogram-aware processing directly into neural networks, replacing traditional modular pipelines with personalized, data-driven amplification. Inspired by this direction, we propose a complementary approach that connects a trainable compensator to a frozen, perceptually grounded simulator (Neuro-MSBG), enabling the compensator to shape its input to match the individual's hearing profiles.
為了推進端到端聽力損失補償，最近的工作（例如 NeuroAMP [30]）已將聽力圖感知處理直接整合到神經網絡中，用個人化、數據驅動的放大取代了傳統的模塊化流程。受此方向的啟發，我們提出了一種互補的方法，將一個可訓練的補償器連接到一個凍結的、基於感知的模擬器 (Neuro-MSBG)，使補償器能夠塑造其輸入以匹配個人的聽力特徵。

Neuro-MSBG is lightweight, differentiable, and does not require clean reference alignment, making it suitable for integration into an end-to-end hearing loss compensation system. Building on this feature, we explore a new use case: connecting a pre-trained Neuro-MSBG model to a trainable compensator to achieve personalized hearing enhancement, as illustrated in Fig. 4. The training and test sets are from VoiceBank [27].
Neuro-MSBG 輕量、可微分，且不需要乾淨的參考對齊，使其適合整合到端到端的聽力損失補償系統中。基於此特性，我們探索了一個新的應用案例：將一個預訓練的 Neuro-MSBG 模型連接到一個可訓練的補償器，以實現個人化的聽力增強，如圖 4 所示。訓練和測試集來自 VoiceBank [27]。

The goal of the compensator is to transform an input waveform into a personalized, compensated version. This compensated waveform is then passed into the frozen Neuro-MSBG model, with the training objective of closely matching the final output with the original clean speech. This design enables the compensator to function as a personalized module that adjusts the audio to each user's hearing condition. It is important to note that we did not fine-tune Neuro-MSBG, as our main objective was to initially verify the feasibility and effectiveness of integrating Neuro-MSBG into the training pipeline. The compensator adopts the same architecture as Neuro-MSBG, but with a key modification in the magnitude path: the original masking-based magnitude encoder is replaced by a mapping strategy designed to enhance or restore lost information. This adjustment better aligns the model with the goal of compensation, enabling the model to generate gain-adjusted outputs that enhance speech intelligibility for hearing-impaired users.
補償器的目標是將輸入波形轉換為個人化的補償版本。這個補償後的波形隨後被傳遞到凍結的 Neuro-MSBG 模型中，訓練目標是使最終輸出與原始的乾淨語音緊密匹配。這種設計使補償器能夠作為一個個人化模組，根據每個用戶的聽力狀況調整音頻。值得注意的是，我們沒有對 Neuro-MSBG 進行微調，因為我們的主要目標是初步驗證將 Neuro-MSBG 整合到訓練流程中的可行性和有效性。補償器採用與 Neuro-MSBG 相同的架構，但在幅度路徑上有一個關鍵修改：原始的基於掩蔽的幅度編碼器被一個旨在增強或恢復丟失資訊的映射策略所取代。這種調整使模型更符合補償的目標，使其能夠生成增益調整後的輸出，從而提高聽障用戶的語音可懂度。

[Image]
Fig. 3. Log-magnitude spectrograms of speech outputs of seven models and ground-truth speech.
圖 3. 七個模型和真實語音的語音輸出的對數幅度譜圖。

[Image]
Fig. 4. Training framework of a personalized speech compensator using Neuro-MSBG as a fixed hearing loss simulator.
圖 4. 使用 Neuro-MSBG 作為固定聽力損失模擬器的個人化語音補償器訓練框架。

[Image]
TABLE IV
[表四]
EFFECTS OF THE AUDIOGRAM ENCODER IN THE NEURO-MSBG MODEL.
聽力圖編碼器在 NEURO-MSBG 模型中的效果。

[Image]
TABLE V
[表五]
STATISTICAL COMPARISON OF HASPI SCORES BEFORE AND AFTER APPLYING THE PROPOSED COMPENSATOR.
應用提議的補償器前後 HASPI 分數的統計比較。

We evaluate the effectiveness of the proposed compensator using the HASPI metric to assess the improvement in perceptual speech intelligibility. As shown in Table V, the compensator significantly improves the average HASPI score from 0.428 to 0.616 (Δ = +0.187). This improvement is statistically significant, supported by both the paired t-test (t = -24.113, p < 0.00001) and the Wilcoxon signed-rank test (W = 292426.0, p < 0.00001). The observed effect size is moderately large (d = 0.594), indicating a substantial improvement in perceptual quality across samples.
我們使用 HASPI 指標來評估所提出的補償器在改善感知語音可懂度方面的有效性。如表 V 所示，補償器顯著地將平均 HASPI 分數從 0.428 提高到 0.616 (Δ = +0.187)。這一改善在統計上是顯著的，這得到了配對 t 檢驗 (t = -24.113, p < 0.00001) 和 Wilcoxon 符號秩檢驗 (W = 292426.0, p < 0.00001) 的支持。觀察到的效應值中等偏大 (d = 0.594)，表明在所有樣本中感知品質都有實質性的改善。

## IV. CONCLUSION
## 四、結論

This paper introduces Neuro-MSBG, a lightweight and fully differentiable hearing loss simulation model that addresses key limitations of traditional approaches, including delay issues, limited integration flexibility, and lack of parallel processing capabilities. Unlike conventional models that require clean reference signal alignment, Neuro-MSBG can be seamlessly integrated into end-to-end training pipelines and avoids timing mismatches that may affect evaluation metrics such as STOI and PESQ. Its parallelizable architecture and low-latency design make it well suited for scalable speech processing applications. In particular, the Mamba-based Neuro-MSBG achieves 46x speedup over the original MSBG, reducing the simulation time for one second of audio from 0.970 seconds to 0.021 seconds through parallel inference. Meanwhile, the LSTM-based variant achieves an inference time of 0.016 seconds, which is 60x faster than MSBG. Experimental results further demonstrate that jointly predicting magnitude and phase can significantly improve speech intelligibility and perceptual quality, with SRCC of 0.9247 for STOI and 0.8671 for PESQ. In addition, the proposed Audiogram Encoder can effectively transform audiogram vectors into frequency-aligned features, outperforming the simple concatenation method and more accurately modeling individual hearing profiles.
本文介紹了 Neuro-MSBG，這是一個輕量級且完全可微分的聽力損失模擬模型，解決了傳統方法的關鍵限制，包括延遲問題、有限的整合靈活性以及缺乏平行處理能力。與需要乾淨參考信號對齊的傳統模型不同，Neuro-MSBG 可以無縫整合到端到端的訓練流程中，並避免可能影響 STOI 和 PESQ 等評估指標的時間不匹配問題。其可平行化的架構和低延遲設計使其非常適合可擴展的語音處理應用。特別是，基於 Mamba 的 Neuro-MSBG 比原始 MSBG 提速 46 倍，通過平行推論將一秒音訊的模擬時間從 0.970 秒減少到 0.021 秒。同時，基於 LSTM 的變體實現了 0.016 秒的推論時間，比 MSBG 快 60 倍。實驗結果進一步證明，聯合預測幅度和相位可以顯著提高語音可懂度和感知品質，STOI 的 SRCC 為 0.9247，PESQ 的 SRCC 為 0.8671。此外，所提出的聽力圖編碼器可以有效地將聽力圖向量轉換為頻率對齊的特徵，優於簡單的串聯方法，並能更準確地建模個人聽力特徵。

## REFERENCES
## 參考文獻
[1] Simone Graetzer, Jon Barker, Trevor J. Cox, Michael Akeroyd, John F. Culling, Graham Naylor, Eszter Porter, and Rhoddy Viveros Muñoz, "Clarity-2021 challenges: Machine learning challenges for advancing hearing aid processing," in Proc. INTERSPEECH, 2021, pp. 686-690.
[2] Brian C. J. Moore, Brian R. Glasberg, and Thomas Baer, "A model for the prediction of thresholds, loudness, and partial loudness," Journal of the Audio Engineering Society, vol. 45, pp. 224-240, 1997.
[3] Thomas Baer and Brian C. J. Moore, "Effects of spectral smearing on the intelligibility of sentences in noise," The Journal of the Acoustical Society of America, vol. 94, pp. 1229-1241, 1993.
[4] Thomas Baer and Brian C. J. Moore, "Effects of spectral smearing on the intelligibility of sentences in the presence of interfering speech," The Journal of the Acoustical Society of America, vol. 95, no. 4, pp. 2050-2062, 1993.
[5] Brian C. J. Moore and Brian R. Glasberg, "Simulation of the effects of loudness recruitment and threshold elevation on the intelligibility of speech in quiet and in a background of speech," The Journal of the Acoustical Society of America, vol. 94, no. 4, pp. 2050-2062, 1993.
[6] Gerardo Roa Dabike, Jon Barker, John F. Culling, et al., "The ICASSP SP Cadenza challenge: Music demixing/remixing for hearing aids," arXiv preprint arXiv:2310.03480, 2023.
[7] Kathryn H. Arehart James M. Kates, "The hearing-aid speech perception index (HASPI)," Speech Communication, vol. 65, pp. 75-93, 2014.
[8] James Kates and Kathryn Arehart, "The hearing-aid speech quality index (HASQI)," AES: Journal of the Audio Engineering Society, vol. 58, pp. 363-381, 2010.
[9] J. M. Kates and K. H. Arehart, "The hearing-aid audio quality index (HAAQI)," IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 24, pp. 354-365, 2016.
[10] Muhammad S. A. Zilany and Ian C. Bruce, "Modeling auditory-nerve responses for high sound pressure levels in the normal and impaired auditory periphery," The Journal of the Acoustical Society of America, vol. 120, no. 3, pp. 1446-1466, 2006.
[11] Sarah Verhulst, Alessandro Altoè, and Viacheslav Vasilkov, “Computational modeling of the human auditory periphery: Auditory-nerve responses, evoked potentials and hearing loss," Hearing Research, vol. 360, pp. 55-75, 2018.
[12] Volker Hohmann, "Frequency analysis and synthesis using a gammatone filterbank," Acta Acustica united with Acustica, vol. 88, pp. 433-442, 2002.
[13] Malcolm Slaney, "Auditory toolbox version 2: A MATLAB toolbox for auditory modeling work," Tech. Rep. 1998-010, Interval Research Corporation, 1998.
[14] Arthur Van Den Broucke, Deepak Baby, and Sarah Verhulst, "Hearing-impaired bio-inspired cochlear models for real-time auditory applications," in Proc. INTERSPEECH, 2020, pp. 2842-2846.
[15] Toshio Irino, "Hearing impairment simulator based on auditory excitation pattern playback: WHIS," IEEE Access, vol. 11, pp. 78419-78430, 2023.
[16] Peter Leer, Jesper Jensen, Zheng-Hua Tan, Jan Østergaard, and Lars Bramsløw, "How to train your ears: Auditory-model emulation for large-dynamic-range inputs and mild-to-severe hearing losses," IEEE/ACM Trans. Audio, Speech and Lang. Proc., vol. 32, pp. 2006-2020.
[17] Fotios Drakopoulos and Sarah Verhulst, "A neural-network framework for the design of individualised hearing-loss compensation," IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 31, pp. 2395-2409, 2023.
[18] Fotios Drakopoulos and Sarah Verhulst, "A differentiable optimisation framework for the design of individualised dnn-based hearing-aid strategies," in Proc. ICASSP, 2022, pp. 351-355.
[19] Fotios Drakopoulos, Arthur Van Den Broucke, and Sarah Verhulst, "A dnn-based hearing-aid strategy for real-time processing: One size fits all," in Proc. ICASSP, 2023, pp. 1-5.
[20] Zehai Tu, Ning Ma, and Jon Barker, "Dhasp: Differentiable hearing aid speech processing," in Proc. ICASSP, 2021, pp. 296-300.
[21] Zehai Tu, Ning Ma, and Jon Barker, "Optimising hearing aid fittings for speech in noise with a differentiable hearing loss model," in Proc. INTERSPEECH, 2021, pp. 691-695.
[22] Ye-Xin Lu, Yang Ai, and Zhen-Hua Ling, "MP-SENet: A speech enhancement model with parallel denoising of magnitude and phase spectra," in Proc. INTERSPEECH, 2023, pp. 3834-3838.
[23] Cees H. Taal, Richard C. Hendriks, Richard Heusdens, and Jesper Jensen, "An algorithm for intelligibility prediction of time-frequency weighted noisy speech," IEEE Transactions on Audio, Speech, and Language Processing, vol. 19, no. 7, pp. 2125-2136, 2011.
[24] Antony W. Rix, John G. Beerends, Michael P. Hollier, and Andries P. Hekstra, "Perceptual evaluation of speech quality (PESQ)-a new method for speech quality assessment of telephone networks and codecs," in Proc. ICASSP, 2001, pp. 749-752.
[25] Rong Chao, Wen-Huang Cheng, Moreno La Quatra, Sabato Marco Siniscalchi, Chao-Han Huck Yang, Szu-Wei Fu, and Yu Tsao, "An investigation of incorporating mamba for speech enhancement," in Proc. SLT, 2024, pp. 302-308.
[26] Yuxuan Ai and Zhen-Hua Ling, "Neural speech phase prediction based on parallel estimation architecture and anti-wrapping losses," in Proc. ICASSP, 2023, pp. 1-5.
[27] Christophe Veaux, Junichi Yamagishi, and Simon King, "The voice bank corpus: Design, collection and data analysis of a large regional accent speech database," in Proc. O-COCOSDA/CASLRE, 2013, pp. 1-4.
[28] Cassia Valentini-Botinhao, "Noisy speech database for training speech enhancement algorithms and tts models," 2017, University of Edinburgh, Centre for Speech Technology Research.
[29] Joachim Thiemann, Nobutaka Ito, and Emmanuel Vincent, "The diverse environments multichannel acoustic noise database (DEMAND): A database of multichannel environmental noise recordings," in Proc. Meetings on Acoustic, 2013, pp. 1-6.
[30] Shafique Ahmed, Ryandhimas E. Zezario, Hui-Guan Yuan, Amir Hussain, Hsin-Min Wang, Wei-Ho Chung, and Yu Tsao, "Neuroamp: A novel end-to-end general purpose deep neural amplifier for personalized hearing aids," arXiv preprint arXiv:2502.10822, 2025.
