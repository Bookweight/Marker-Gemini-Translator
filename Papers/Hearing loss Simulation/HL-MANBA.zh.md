
 
# Abstract

This work proposes and validates a differentiable, end-to-end trainable hearing-loss simulation model designed to overcome the computational limitations of the traditional MSBG auditory model (Brian C. J. Moore & Brian R. Glasberg). Although the MSBG model reliably reproduces perceptual characteristics of sensorineural hearing loss, its cascaded filtering and nonlinear compression introduce high latency and poor parallelizability, making real-time integration with modern neural speech systems challenging.

本研究提出並驗證了一個可微分、可端到端訓練的聽力損失模擬模型，旨在克服傳統 MSBG 聽覺模型（Brian C. J. Moore & Brian R. Glasberg）的計算限制。儘管 MSBG 模型能夠可靠地重現感音神經性聽力損失的感知特徵，但其級聯濾波和非線性壓縮引入了高延遲和不良的並行性，使得與現代神經語音系統的即時整合充滿挑戰。

To address these issues, we introduce HL-Mamba, a neural architecture that replaces cascaded filtering with a differentiable signal-mapping pipeline. HL-Mamba incorporates Audiogram Encoding and time-frequency modeling to jointly learn amplitude attenuation and phase distortion, supported by a feed-forward structure that enables efficient parallel inference. A lightweight Audiogram Encoder embeds clinical audiograms via frequency alignment, allowing individualized degeneration patterns. Experiments further confirm the importance of phase modeling, which improves similarity to Moore-Glasberg responses and enhances perceived naturalness.

為了解決這些問題，我們引入了 HL-Mamba，這是一種神經架構，用可微分的信號映射管道取代了級聯濾波。HL-Mamba 整合了聽力圖編碼和時頻建模，以共同學習振幅衰減和相位失真，並由一個可實現高效並行推斷的前饋結構支持。一個輕量級的聽力圖編碼器通過頻率對齊嵌入臨床聽力圖，從而實現個性化的退化模式。實驗進一步證實了相位建模的重要性，它提高了與 Moore-Glasberg 響應的相似性，並增強了感知的自然度。

Overall, this work aims to (1) reduce simulation latency for real-time speech processing, (2) improve perceptual realism and clarity of degraded speech, and (3) provide a differentiable hearing-loss module for speech enhancement and compensation frameworks. Results demonstrate that the proposed Neuro-MSBG model preserves core Moore Glasberg perceptual behavior while achieving higher efficiency and perceptual similarity, showing strong integration potential in end-to-end speech pipelines.

總體而言，本研究旨在 (1) 降低即時語音處理的模擬延遲，(2) 改善退化語音的感知真實感和清晰度，以及 (3) 為語音增強和補償框架提供一個可微分的聽力損失模塊。結果表明，所提出的 Neuro-MSBG 模型在保持核心 Moore Glasberg 感知行為的同時，實現了更高的效率和感知相似性，顯示出在端到端語音管道中強大的整合潛力。

Keywords: Hearing loss simulation, Differentiable auditory modeling, End-to-end speech processing, Phase-aware degradation

關鍵詞：聽力損失模擬、可微分聽覺建模、端到端語音處理、相位感知退化

# Chapter 1 Introduction

Hearing loss simulators aim to reproduce how impaired auditory perception alters sound processing along the human auditory pathway. As their modeling fidelity and computational usability increase, these systems have become foundational tools for research, evaluation, and algorithm benchmarking. This trend is evident in initiatives such as the Clarity Challenge [10], which adopts the Moore-Stone-Baer-Glasberg (MSBG) model [19, 3, 4, 18] to generate individualized listening conditions from audiograms. Challenges including Cadenza [21] and Clarity further emphasize perceptual evaluation, incorporating HASPI [13], HASQI [15], and HAAQI [14] as objective intelligibility and quality metrics grounded in auditory physiology.

聽力損失模擬器旨在重現聽覺感知受損如何改變沿著人類聽覺通路的聲音處理。隨著其模型保真度和計算可用性的提高，這些系統已成為研究、評估和算法基準測試的基礎工具。這一趨勢在諸如 Clarity Challenge [10] 等計劃中表現得尤為明顯，該計劃採用 Moore-Stone-Baer-Glasberg (MSBG) 模型 [19, 3, 4, 18] 從聽力圖生成個性化的聆聽條件。包括 Cadenza [21] 和 Clarity 在內的挑戰進一步強調了感知評估，並將基於聽覺生理學的客觀可懂度和質量指標 HASPI [13]、HASQI [15] 和 HAAQI [14] 納入其中。

Existing simulation approaches may be broadly divided into physiological and engineering-oriented methodologies. Physiological models, such as Zilany et al. [30] and the transmission line cochlear representation of Verhulst et al. [29], provide biologically interpretable insights but often impose heavy computational costs, limiting real-time deployment. In contrast, engineering-focused solutions seek a practical balance between perceptual accuracy and compute efficiency, as seen in the Hohmann filterbank [11], the Auditory Toolbox [22], and MSBG [19]. Among these, MSBG remains dominant due to its audiogram-driven perceptual alignment. However, its sequential multi-stage filtering blocks parallelization, and its non-uniform latency complicates waveform alignment—both of which hinder integration into end-to-end learning frameworks.

現有的模擬方法大致可分為生理學導向和工程導向兩大類。生理學模型，如 Zilany 等人 [30] 的模型和 Verhulst 等人 [29] 的傳輸線耳蝸表示法，提供了具有生物學可解釋性的見解，但通常計算成本高昂，限制了即時部署。相比之下，以工程為重點的解決方案則尋求在感知準確性和計算效率之間取得實際平衡，例如 Hohmann 濾波器組 [11]、Auditory Toolbox [22] 和 MSBG [19]。其中，由於其聽力圖驅動的感知對齊，MSBG 仍占主導地位。然而，其順序多級濾波阻礙了並行化，其非均勻延遲使波形對齊複雜化——這兩者都妨礙了與端到端學習框架的整合。

To reduce computational overhead, recent work [5, 12] has moved toward neural approximations and streamlined auditory simulation. CoNNear [5] provides a neural surrogate for cochlear mechanics, facilitating real-time auditory nerve response generation, while Tan et al. [16] approximate the Verhulst auditory periphery across hearing-loss conditions. Although effective at the neural representation level, these methods do not output waveform-domain signals, which limits their suitability for time-domain supervision and downstream enhancement tasks. On the engineering side, the WHIS simulator [12] accelerates MSBG through a Gammachirp filterbank and a time-varying minimum-phase reconstruction, reducing one second of speech to roughly 10 ms of processing and maintaining near-perfect temporal alignment. Yet, WHIS still relies on frame-based IIR coefficient estimation and dynamic gain selection, lacks vectorization and parallel inference, and remains difficult to embed within differentiable training pipelines.

為了減少計算開銷，最近的研究 [5, 12] 已轉向神經近似和簡化的聽覺模擬。CoNNear [5] 提供了一個用於耳蝸力學的神經代理，促進了即時聽神經反應的生成，而 Tan 等人 [16] 則近似了 Verhulst 聽覺外圍在不同聽力損失條件下的行為。儘管在神經表示層面上有效，但這些方法不輸出波形域信號，這限制了它們在時域監督和下游增強任務中的適用性。在工程方面，WHIS 模擬器 [12] 通過 Gammachirp 濾波器組和時變最小相位重建來加速 MSBG，將一秒語音的處理時間減少到大約 10 毫秒，並保持近乎完美的時間對齊。然而，WHIS 仍然依賴於基於幀的 IIR 係數估計和動態增益選擇，缺乏向量化和並行推斷，並且難以嵌入到可微分的訓練管道中。

As differentiable auditory simulation gains traction in hearing-aid optimization, improving learnability and computational efficiency has emerged as a key research objective. CoNNear-derived auditory nerve responses have been incorporated as loss terms for training compensation networks [8, 7, 9], encouraging restored speech to approximate the neural behaviour of a normal listener. Engineering-oriented work has likewise progressed: Tu et al. [25] introduced the DHASP framework, enabling backpropagation through the HASPI processing pipeline, and later developed a differentiable MSBG implementation for hearing-aid optimization [26], demonstrating that physiologically motivated simulation can be integrated into learning-based compensation.

隨著可微分聽覺模擬在助聽器優化中受到越來越多的關注，提高可學習性和計算效率已成為一個關鍵的研究目標。源自 CoNNear 的聽神經反應已被用作訓練補償網絡 [8, 7, 9] 的損失項，鼓勵恢復的語音近似正常聽眾的神經行為。工程導向的工作同樣取得了進展：Tu 等人 [25] 介紹了 DHASP 框架，使得能夠通過 HASPI 處理管道進行反向傳播，後來又為助聽器優化開發了一個可微分的 MSBG 實現 [26]，證明了生理學動機的模擬可以整合到基於學習的補償中。

Despite these developments, most differentiable simulators remain focused on magnitude-only processing. Phase information—known to influence perceived quality in speech enhancement [17]—is rarely modeled in hearing loss simulation. Motivated by evidence that phase-aware enhancement improves perceptual realism, we propose Neuro-MSBG, a fully differentiable, end-to-end simulation model that generates impaired speech directly in the time domain. Waveform-domain operation allows seamless compatibility with loss functions such as MSE, STOI [23], and PESQ [20], and supports both clean and noisy input speech. Our experiments show that incorporating phase significantly enhances perceptual correspondence with MSBG, confirming the importance of phase-aware auditory modeling.

儘管取得了這些進展，但大多數可微分模擬器仍然只關注幅度處理。相位信息——已知會影響語音增強中的感知質量 [17]——在聽力損失模擬中很少被建模。受相位感知增強可提高感知真實感的證據啟發，我們提出了 Neuro-MSBG，一個完全可微分的端到端模擬模型，可直接在時域生成受損語音。波形域操作允許與 MSE、STOI [23] 和 PESQ [20] 等損失函數無縫兼容，並支持乾淨和帶噪聲的輸入語音。我們的實驗表明，納入相位可顯著增強與 MSBG 的感知對應性，證實了相位感知聽覺建模的重要性。

*   **Efficient and parallelizable:** One second of speech is simulated in 0.021 s, a 46× improvement over MSBG.
*   **Latency-stable and alignment-preserving:** Eliminates irregular delay and directly supports end-to-end learning pipelines.
*   **Phase-aware modeling:** Improves perceptual consistency, yielding SRCC scores of 0.9247 (STOI) and 0.8671 (PESQ).

*   **高效且可並行化：** 一秒長的語音可在 0.021 秒內完成模擬，相較於 MSBG 提升了 46 倍。
*   **延遲穩定且保持對齊：** 消除了不規則的延遲，並直接支持端到端的學習流程。
*   **相位感知建模：** 提高了感知一致性，在 STOI 和 PESQ 上的 SRCC 分數分別達到了 0.9247 和 0.8671。

We additionally demonstrate its integration with a differentiable compensation network, illustrating practical feasibility for hearing aid algorithm development. Section II details the proposed model, Section III presents experimental results, and Section IV concludes the work.

我們還展示了它與一個可微分補償網絡的整合，說明了助聽器算法開發的實際可行性。第二節詳細介紹了所提出的模型，第三節介紹了實驗結果，第四節對全文進行了總結。

# Chapter 2 Related Work

Hearing loss simulation has been widely studied from two major perspectives: (1) physiologically grounded auditory models that replicate cochlear mechanics and neural transduction, and (2) engineering-oriented models that emphasize computational efficiency and practical deployment. This chapter reviews representative works in both categories and discusses their advantages, limitations, and relevance to the development of HL-Mamba.

聽力損失模擬的研究主要有兩大視角：(1) 模擬耳蝸力學與神經轉導的生理聽覺模型，以及 (2) 強調計算效率與實際部署的工程模型。本章回顧這兩類代表性研究，並討論其優缺點，以及與 HL-Mamba 開發的關聯性。

## 2.1 Physiological Models of Hearing Loss

Physiological models aim to mimic the real auditory periphery, including basilar membrane vibration, auditory nerve firing patterns, and nonlinear cochlear dynamics. Among them, the auditory periphery model of Zilany et al. [30] and the transmission-line (TL) cochlear model proposed by Verhulst et al. [29] are two of the most influential frameworks. These models provide biologically realistic output that is valuable for analyzing neural encoding and psychoacoustic perception under sensorineural hearing loss. However, their numerical complexity and multi-stage computation make them computationally expensive, leading to slow simulation speed, limited scalability, and difficulty in real-time deployment.

生理模型旨在模擬真實的聽覺週邊，包括基底膜振動、聽神經發放模式和非線性耳蝸動力學。其中，Zilany 等人 [30] 的聽覺週邊模型和 Verhulst 等人 [29] 提出的傳輸線 (TL) 耳蝸模型是兩個最具影響力的框架。這些模型提供了生物學上真實的輸出，對於分析感音神經性聽力損失下的神經編碼和心理聲學感知非常有價值。然而，它們的數值複雜性和多階段計算使其計算成本高昂，導致模擬速度慢、可擴展性有限，並且難以進行即時部署。

To improve usability, several efforts have attempted to approximate cochlear dynamics with neural architectures. CoNNear [5] accelerates TL-model inference using convolutional networks, enabling real-time simulation of auditory nerve responses. Tan et al. [16] further demonstrated that deep learning can emulate the Verhulst auditory periphery model for diverse hearing profiles. Despite these advances, physiological models generally do not produce waveform outputs and therefore cannot be directly incorporated into speech enhancement, hearing aid learning frameworks, or perceptual objective metrics such as STOI or PESQ. As a result, their impact remains more prominent in auditory neuroscience than in deployable audio signal processing.

為了提高可用性，一些研究嘗試使用神經架構來近似耳蝸動力學。CoNNear [5] 利用卷積網絡加速 TL 模型推斷，實現了聽覺神經響應的實時模擬。Tan 等人 [16] 進一步證明，深度學習可以模擬 Verhulst 聽覺外周模型在不同聽力特徵下的表現。儘管取得了這些進展，生理模型通常不產生波形輸出，因此無法直接整合到語音增強、助聽器學習框架或諸如 STOI 或 PESQ 等感知客觀指標中。因此，它們的影響在聽覺神經科學中比在可部署的音頻信號處理中更為顯著。

## 2.2 Engineering-Oriented Models of Hearing Loss

Engineering approaches aim to provide perceptually grounded yet computationally efficient hearing loss simulation. Classic examples include the Hohmann auditory filter-bank [11], the Auditory Toolbox [22], and the Moore, Stone, Baer, and Glasberg (MSBG) model [19]. MSBG remains the most widely used model in research and evaluation due to its ability to reproduce spectral smearing, loudness recruitment, and reduced frequency selectivity based on audiograms. The Clarity [10] and Cadenza [21] challenges both adopt MSBG-generated data as perceptually aligned impaired speech, together with evaluation metrics such as HASPI, HASQI, and HAAQI [13, 15, 14].

工程方法旨在提供具有感知基礎但計算效率高的聽力損失模擬。典型範例包括霍曼聽覺濾波器組 [11]、聽覺工具箱 [22]，以及穆爾、史東、貝爾和格拉斯伯格（MSBG）模型 [19]。由於能夠根據聽力圖重現頻譜塗抹、響度重振和頻率選擇性下降等現象，MSBG 至今仍是研究和評估中使用最廣泛的模型。Clarity [10] 和 Cadenza [21] 挑戰賽均採用 MSBG 生成的數據作為感知對齊的受損語音，並結合 HASPI、HASQI 和 HAAQI [13, 15, 14] 等評估指標。

However, MSBG suffers from two primary limitations: (1) lack of parallel computing support, which hinders real-time application and GPU scaling, and (2) variable latency caused by multi-stage filtering, making end-to-end learning integration difficult. To mitigate these constraints, Irino et al. proposed WHIS [12], which constructs an excitation-based minimum-phase filter for fast waveform transformation. WHIS reduces one-second processing to 10 ms while preserving temporal alignment. Nonetheless, it still requires frame-wise IIR estimation, lacks efficient vectorization, and has not yet been fully integrated into neural frameworks.

然而，MSBG存在兩個主要限制：(1) 缺乏並行計算支持，這阻礙了實時應用和GPU擴展；(2) 多級濾波導致的可變延遲，使得端到端學習集成變得困難。為緩解這些限制，Irino等人提出了WHIS [12]，它構建了一個基於激勵的最小相位濾波器以實現快速波形轉換。WHIS將一秒的處理時間縮短至10毫秒，同時保留了時間對齊。儘管如此，它仍然需要逐幀進行IIR估計，缺乏高效的向量化，並且尚未完全整合到神經框架中。

Recent differentiable auditory models further enable end-to-end optimization. Tu et al. introduced DHASP [25] and a differentiable MSBG variant [26], demonstrating that auditory simulation modules can participate in gradient-based training. However, existing engineering-oriented models focus mainly on magnitude, while phase—critical for perceptual realism—remains under-modeled. This motivates the development of HL-Mamba, which incorporates phase modeling, supports parallel execution, and enables seamless integration into waveform-based hearing compensation pipelines.

近期的可微分聽覺模型進一步實現了端到端的優化。Tu等人介紹了DHASP [25] 和一個可微分的MSBG變體 [26]，證明了聽覺模擬模塊可以參與基於梯度的訓練。然而，現有的工程導向模型主要關注幅度，而對於感知真實性至關重要的相位仍然建模不足。這促使了HL-Mamba的發展，它結合了相位建模，支持並行執行，並能無縫整合到基於波形的聽力補償管道中。

[Image]
Figure 2.1: Overall architecture of the proposed HL-Mamba framework.
圖 2.1：所提出的 HL-Mamba 框架的整體架構。

# Chapter 3 Methodology

This chapter presents the model architecture and training objectives of the proposed HL-Mamba framework. As illustrated in Fig. 1, the model takes normal-hearing speech signals and monaural audiograms as inputs.

本章介紹了所提出的 HL-Mamba 框架的模型架構和訓練目標。如圖 1 所示，該模型以正常聽力者的語音信號和單耳聽力圖作為輸入。

The audiogram is first processed by the Audiogram Encoder to generate personalized hearing-condition features. Meanwhile, the speech signal is transformed into the time-frequency domain via Short-Time Fourier Transform (STFT), yielding its magnitude and phase components. These three types of features, namely the personalized hearing features, the magnitude features, and the phase features, are then concatenated and fed into the Neural Network Block (NN Block).

聽力圖首先由聽力圖編碼器處理，以生成個性化的聽力條件特徵。同時，語音信號通過短時傅立葉變換 (STFT) 轉換到時頻域，產生其幅度和相位分量。這三種類型的特徵，即個性化聽力特徵、幅度特徵和相位特徵，然後被串聯並輸入到神經網絡塊 (NN Block) 中。

Subsequently, the network splits into two decoding branches: the Magnitude Mask Decoder and the Phase Decoder, which respectively estimate the magnitude and phase modifications associated with hearing loss. Finally, the predicted magnitude and phase are combined and converted back to the time domain using inverse STFT, producing the speech signal as it would be perceived by a listener with hearing impairment.

隨後，網絡分成兩個解碼分支：幅度掩碼解碼器和相位解碼器，分別估計與聽力損失相關的幅度和相位修正。最後，預測的幅度和相位被結合起來，並使用逆短時傅立葉變換 (inverse STFT) 轉換回時域，產生聽力受損者感知到的語音信號。

## 3.1 HL-Mamba

The overall architecture of HL-Mamba is built upon the MP-SENet [17] and the SE-Mamba framework [6], which together offer efficient and expressive mechanisms for feature interaction. This design choice is motivated by our empirical observation that phase information plays a crucial role in accurately simulating hearing loss effects (see Section III for further analysis). To systematically assess how different neural modules capture spectral and temporal cues, we replace the original attention-based components with several alternatives, including LSTM, Transformer, CNN, and Mamba blocks. In view of Mamba’s strength in modeling long sequences with low latency, we also carefully control the model size so that the resulting architecture remains lightweight while preserving strong performance.

HL-Mamba 的整體架構建立在 MP-SENet [17] 和 SE-Mamba 框架 [6] 之上，它們共同為特徵交互提供了高效且富有表現力的機制。這一設計選擇的動機源於我們的經驗觀察，即相位信息在準確模擬聽力損失效應中起著至關重要的作用（詳見第三節的進一步分析）。為了系統地評估不同神經模塊如何捕捉光譜和時間線索，我們用幾種替代方案取代了原始的基於注意力的組件，包括 LSTM、Transformer、CNN 和 Mamba 塊。鑑於 Mamba 在以低延遲建模長序列方面的優勢，我們還仔細控制了模型大小，以使最終的架構在保持強大性能的同時保持輕量級。

Regarding audiogram integration, prior approaches often append audiogram representations along the frequency axis. In contrast, our experiments indicate that treating the audiogram as an additional input channel, alongside the magnitude and phase channels, leads to more stable and effective learning. This channel-based conditioning strategy provides the model with a spatially aligned, layer-wise consistent way of injecting hearing-profile information, thereby enhancing its capability to modulate internal feature representations.

關於聽力圖整合，先前的方法通常沿著頻率軸附加聽力圖表示。相比之下，我們的實驗表明，將聽力圖作為一個額外的輸入通道，與幅度和相位通道並列，可以帶來更穩定和有效的學習。這種基於通道的條件化策略為模型提供了一種空間對齊、層級一致的方式來注入聽力檔案信息，從而增強其調節內部特徵表示的能力。

### Audiogram Encoder

To incorporate individualized hearing profiles, we design a lightweight Audiogram Encoder that maps the audiogram a ∈ RB×8 to a frequency-aligned representation aenc ∈ RBXF, where B denotes the batch size and F = 201 matches the STFT frequency resolution. The transformation is formulated as:
aenc = W · Flatten (AvgPool (σ (Conv(a)))), (3.1)
where Conv is a 1D convolution layer, σ denotes a ReLU activation, and W is a linear projection matrix.

為了整合個人化的聽力 profile，我們設計了一個輕量級的聽力圖編碼器，它將聽力圖 a ∈ RB×8 映射到一個頻率對齊的表示 aenc ∈ RBXF，其中 B 表示批次大小，F = 201 匹配 STFT 的頻率分辨率。轉換公式如下：
aenc = W · Flatten(AvgPool(σ(Conv(a)))) (3.1)
其中 Conv 是一個一維卷積層，σ 表示一個 ReLU 激活函數，W 是一個線性投影矩陣。

The encoded vector is then broadcast along the time dimension and concatenated with the magnitude and phase features, forming a three-channel input tensor of shape RB×3×T×F to the DenseEncoder and NN Block. This channel-based integration ensures that the hearing-profile information is injected consistently across both time and frequency axes, facilitating effective conditioning throughout the network.

編碼後的向量隨後沿著時間維度廣播，並與幅度和相位特徵級聯，形成一個形狀為 RB×3×T×F 的三通道輸入張量，輸入到 DenseEncoder 和 NN Block 中。這種基於通道的整合確保了聽力輪廓信息在時間和頻率軸上的一致注入，從而促進了整個網絡的有效條件化。

### Neural Network Blocks

To effectively capture the temporal and spectral characteristics of speech affected by hearing loss, we develop and compare several NN Blocks, each following a dual-path design that separately models the time and frequency dimensions. Concretely, the input tensor is first rearranged and reshaped for temporal modeling; a similar operation is then performed for frequency modeling. Each path is equipped with residual connections and a ConvTransposeld layer to restore the original tensor shape, ensuring seamless interaction with subsequent modules.

為了有效地捕捉受聽力損失影響的語音的時間和頻譜特徵，我們開發並比較了幾種神經網絡塊（NN Blocks），每種塊都遵循雙路徑設計，分別對時間和頻率維度進行建模。具體來說，輸入張量首先被重新排列和重塑以進行時間建模；然後對頻率建模執行類似的操作。每個路徑都配備了殘差連接和一個 ConvTransposeld 層來恢復原始張量形狀，確保與後續模塊的無縫交互。

Within this unified framework, the core time-frequency modeling module is instantiated using one of the following four alternatives:

*   **Mamba Block:** Bidirectional Mamba modules are used to model time and frequency independently, providing efficient long-range dependency modeling across both dimensions.
*   **Transformer Block:** Transformer encoder layers are applied along each axis, using global attention to capture rich contextual information.
*   **LSTM Block:** Bidirectional LSTMs are employed to model sequential dynamics, followed by linear projection layers to maintain dimensional compatibility.
*   **CNN Block:** One-dimensional convolutions are used to extract local patterns, followed by channel expansion, non-linear activation, and residual fusion to enhance feature expressiveness.

在這個統一的框架內，核心的時頻建模模塊使用以下四種替代方案之一進行實例化：

*   **Mamba 模塊：** 雙向 Mamba 模塊用於獨立地建模時間和頻率，提供跨越兩個維度的高效長程依賴建模。
*   **Transformer 模塊：** Transformer 編碼器層沿著每個軸應用，使用全局注意力來捕獲豐富的上下文信息。
*   **LSTM 模塊：** 雙向 LSTM 用於建模序列動態，隨後是線性投影層以保持維度兼容性。
*   **CNN 模塊：** 一維卷積用於提取局部模式，隨後是通道擴展、非線性激活和殘差融合以增強特徵表達能力。

This dual-axis architecture yields a flexible and robust framework for hearing loss simulation. It also enables a controlled comparison of different neural architectures under the same interface, highlighting the potential of Mamba for low-latency, high-fidelity speech modeling.

這種雙軸架構為聽力損失模擬提供了一個靈活而穩健的框架。它還可以在相同的接口下對不同的神經架構進行受控比較，突顯了 Mamba 在低延遲、高保真語音建模方面的潛力。

## 3.2 Training Criteria

To guide the learning of HL-Mamba, we adopt a multi-objective loss function that jointly enforces spectral accuracy, phase consistency, and time-domain fidelity.
First, the magnitude loss LMag is defined as the mean squared error (MSE) between the predicted magnitude m and the ground-truth magnitude m:
LMag = 1/N Σ ||mi - mi||², (3.2)
where N denotes the number of training samples.

為指導 HL-Mamba 的學習，我們採用了一個多目標損失函數，該函數共同強制頻譜準確性、相位一致性和時域保真度。
首先，幅度損失 LMag 定義為預測幅度 m 與真實幅度 m 之間的均方誤差 (MSE)：
LMag = 1/N Σ ||mi - mi||², (3.2)
其中 N 表示訓練樣本的數量。

The phase loss Lpha is derived from the anti-wrapping strategy in [2] and consists of three components: the instantaneous phase loss LIP, the group delay loss LGD, and the integrated absolute frequency loss LIAF. They are defined as:
LIP = E p,p [|| faw (p - p)||1], (3.3)
LGD = E p,p [|| faw (∆F(p - p))||1], (3.4)
LIAF = E p,p [|| faw (∆T(p - p))||1], (3.5)
Lpha = LIP + LGD + LIAF, (3.6)
where faw(·) denotes the anti-wrapping function used to alleviate 2π discontinuities, and ∆F(·) and ∆T(·) represent the first-order phase differences along the frequency and time axes, respectively.

相位損失 Lpha 源自 [2] 中的反包裹策略，由三個分量組成：瞬時相位損失 LIP、群延遲損失 LGD 和積分絕對頻率損失 LIAF。它們的定義如下：
LIP = E p,p [|| faw (p - p)||1], (3.3)
LGD = E p,p [|| faw (∆F(p - p))||1], (3.4)
LIAF = E p,p [|| faw (∆T(p - p))||1], (3.5)
Lpha = LIP + LGD + LIAF, (3.6)
其中 faw(·) 表示用於減輕 2π 不連續性的反包裹函數，∆F(·) 和 ∆T(·) 分別表示沿頻率和時間軸的一階相位差。

The complex loss Lcom measures the MSE between the predicted complex spectrogram ĉ and the reference complex spectrogram c (including both real and imaginary parts):
LCom = 1/N Σ ||ĉi - ci||² (3.7)
To further preserve waveform-level fidelity, we introduce the time-domain loss LTime, computed as the L1 distance between the predicted waveform x̂ and the target waveform x:
LTime = 1/N Σ |x̂i - xi| (3.8)
The overall training objective is a weighted sum of the above components:
LTotal = λMagLMag + λPhaLPha + λComLCom + λTimeLTime, (3.9)
where each λ is a tunable scalar that controls the relative contribution of the corresponding loss term.

複雜損失 Lcom 衡量預測的複雜譜圖 ĉ 與參考複雜譜圖 c（包括實部和虛部）之間的均方誤差（MSE）：
LCom = 1/N Σ ||ĉi - ci||² (3.7)
為進一步保持波形級保真度，我們引入時域損失 LTime，計算為預測波形 x̂ 與目標波形 x 之間的 L1 距離：
LTime = 1/N Σ |x̂i - xi| (3.8)
總體訓練目標是上述分量的加權和：
LTotal = λMagLMag + λPhaLPha + λComLCom + λTimeLTime, (3.9)
其中每個 λ 是一個可調標量，用於控制相應損失項的相對貢獻。

# Chapter 4 Experiments and Results

The dataset consists of 12,396 clean utterances from VoiceBank [28], where 11,572 are used for training and 824 for testing. Their noisy counterparts originate from the VoiceBank-DEMAND corpus [27], created by mixing each clean utterance with a randomly selected noise recording from the 10 DEMAND environments [24] and a specified signal-to-noise ratio (SNR). To promote generalization, 8 noise types are allocated to the training split and 2 disjoint noise types are used exclusively for testing. SNR values are drawn from {0, 5, 10, 15} dB during training and from {2.5, 7.5, 12.5, 17.5} dB for testing. Each utterance is associated with two monaural audiograms that represent different degrees of hearing loss, spanning mild to severe impairment.

該數據集包含來自 VoiceBank [28] 的 12,396 個純淨語音，其中 11,572 個用於訓練，824 個用於測試。它們的帶噪聲對應版本源自 VoiceBank-DEMAND 語料庫 [27]，是將每個純淨語音與從 10 個 DEMAND 環境 [24] 中隨機選取的噪聲錄音以及指定的信噪比 (SNR) 混合而成。為促進泛化，8 種噪聲類型分配給訓練集，2 種不相交的噪聲類型專門用於測試。訓練期間的信噪比值從 {0, 5, 10, 15} dB 中選取，測試期間則從 {2.5, 7.5, 12.5, 17.5} dB 中選取。每個語音都與兩個單耳聽力圖相關聯，代表從輕度到重度不等的不同程度的聽力損失。

As a result, the size of the training set expands to 11,572 × 2 × 2 = 46,288 samples, while the test set expands to 824 × 2 × 2 = 3,296 samples. There is no overlap in speech content or audiograms between the training and testing splits, ensuring that evaluation is performed on previously unseen utterances and unseen hearing-loss profiles.

因此，訓練集的大小擴大到 11,572 × 2 × 2 = 46,288 個樣本，而測試集則擴大到 824 × 2 × 2 = 3,296 個樣本。訓練集和測試集在語音內容或聽力圖上沒有重疊，確保評估是在先前未見過的語音和未見過的聽力損失輪廓上進行的。

This work conducts a comprehensive set of experiments across five major dimensions: (i) comparison of HL-Mamba variants with different architectures and monolithic baselines (Table 4.1); (ii) evaluation of inference latency (Table 4.3); (iii) analysis of the importance of phase modeling (Table 4.2); (iv) investigation of audiogram integration strategies (Table 4.4); and (v) application of HL-Mamba within a speech compensation framework (Table 4.5). Across all quantitative results, bold denotes the top-performing system, while underline indicates the runner-up.

本研究在五個主要維度上進行了一系列全面的實驗：(i) HL-Mamba 變體與不同架構和單體基線的比較（表 4.1）；(ii) 推理延遲的評估（表 4.3）；(iii) 相位建模重要性的分析（表 4.2）；(iv) 聽力圖整合策略的探討（表 4.4）；以及 (v) HL-Mamba 在語音補償框架內的應用（表 4.5）。在所有量化結果中，粗體表示表現最佳的系統，而下劃線表示次佳者。

All experiments were executed on a single NVIDIA RTX 3090 GPU. Model training was conducted for 200 epochs with a batch size of 3, using an initial learning rate of 0.0005 and the AdamW optimizer.

所有實驗均在單一 NVIDIA RTX 3090 GPU 上執行。模型訓練進行了 200 個 epoch，批次大小為 3，初始學習率為 0.0005，並使用 AdamW 優化器。

## 4.1 MSBG-Based Alignment

We employ the MSBG model to emulate hearing loss by applying an ear-dependent gain profile to each input signal, producing a single-channel impaired output. However, the multi-stage filtering operations in MSBG may introduce an unknown latency relative to the original waveform. This temporal shift is undesirable for subsequent tasks such as speech enhancement or intelligibility evaluation, where accurate temporal alignment is critical.

我們採用 MSBG 模型來模擬聽力損失，方法是對每個輸入信號應用與耳朵相關的增益曲線，產生單聲道受損輸出。然而，MSBG 中的多級濾波操作可能會引入相對於原始波形的未知延遲。這種時間偏移對於後續任務（如語音增強或可懂度評估）是不利的，因為在這些任務中，準確的時間對齊至關重要。

To estimate and correct for this latency, we follow the procedure described in [10] and generate an auxiliary reference signal together with the main audio during the MSBG simulation. This reference is a silent waveform that has the same duration and sampling rate as the input, and it is used exclusively for delay monitoring. A unit impulse is embedded into this reference signal, defined as:
δ[n – k] = { 1, if n = k; 0, otherwise (4.1)
where n denotes the discrete time index, and k specifies the impulse location, with Fs being the sampling rate in Hz.

為了估計並校正此延遲，我們遵循 [10] 中描述的程序，在 MSBG 模擬期間與主音頻一起生成一個輔助參考信號。該參考信號是一個與輸入具有相同持續時間和採樣率的靜音波形，專門用於延遲監控。一個單位脈衝被嵌入到該參考信號中，定義為：
δ[n – k] = { 1, if n = k; 0, otherwise (4.1)
其中 n 表示離散時間索引，k 指定脈衝位置，Fs 為採樣率（單位：Hz）。

The impulse is placed at the center of the auxiliary reference signal. After passing through the MSBG processing, we compute the shift between the original and processed impulse positions to infer the delay introduced by MSBG. This estimated delay is then applied to synchronize the normal-hearing input, the clean reference, and the impaired output, enabling reliable computation of highly time-sensitive metrics such as STOI and PESQ.

脈衝被放置在輔助參考信號的中心。通過 MSBG 處理後，我們計算原始和處理後脈衝位置之間的偏移，以推斷 MSBG 引入的延遲。然後應用此估計延遲來同步正常聽力輸入、乾淨參考和受損輸出，從而能夠可靠地計算高度時間敏感的指標，如 STOI 和 PESQ。

Each training example therefore contains: 1) the impaired single-ear speech, 2) the aligned clean reference, 3) the aligned normal-hearing input, and 4) the corresponding 8-dimensional audiogram vector. Figure 4.1 illustrates the waveform alignment before and after applying the delay compensation.

因此，每個訓練範例包含：1) 受損的單耳語音，2) 對齊的乾淨參考語音，3) 對齊的正常聽力輸入，以及 4) 相應的 8 維聽力圖向量。圖 4.1 說明了在應用延遲補償前後的波形對齊情況。

[Image]
Figure 4.1: Waveform alignment before and after correction. (A) The MSBG output exhibits a delay relative to the input signal. (B) The proposed impulse-based correction compensates for the delay and restores temporal alignment for accurate evaluation.
圖 4.1：校正前後的波形對齊。(A) MSBG 輸出相對於輸入信號存在延遲。(B) 所提出的基於脈衝的校正補償了延遲並恢復了時間對齊，以便進行準確評估。

## 4.2 Main Performance Comparison

[Image]
Table 4.1: Comparison of the performance of monolithic models and HL-Mamba variants with different TF-NN blocks.
表格 4.1：單體模型與採用不同 TF-NN 模塊的 HL-Mamba 變體之性能比較。

In the initial phase of our study, we employed monolithic models with single, uniform architectures such as CNN, LSTM, and Transformer to simulate hearing loss. Due to their limited performance, we transitioned to the HL-Mamba framework, modifying only the TF-NN block while testing multiple architectures, including CNN, LSTM, Transformer, and Mamba. The results in Table 4.1 present the outcomes of this comparison. To ensure fairness across models, we controlled the total parameter count so that all architectures had approximately the same number of learnable parameters. As indicated in Table 4.1, the HL-Mamba variants consistently surpass the monolithic baselines on unseen test sets. Among these variants, the Mamba-based implementation of HL-Mamba demonstrates the highest performance across all evaluation metrics. We further compare HL-Mamba with an existing neural model, CoNNear, by examining model complexity and inference latency. Although the two approaches serve different purposes—CoNNear focuses on predicting physiologically accurate auditory nerve responses, whereas HL-Mamba is designed for perceptually oriented signal transformation—the comparison remains reasonable because both pursue real-time hearing loss simulation. According to Table 4.3, CoNNear contains roughly 11.7 million parameters, while HL-Mamba has only 1.45 million. Beyond being more compact, HL-Mamba can process noisy speech inputs and flexibly handle a diverse range of audiogram configurations, offering advantages for real-world applications with varying acoustic conditions and personalized hearing requirements.

在我們研究的初始階段，我們採用了具有單一、統一架構（如 CNN、LSTM 和 Transformer）的單體模型來模擬聽力損失。由於其性能有限，我們轉而採用 HL-Mamba 框架，僅修改 TF-NN 模塊，同時測試多種架構，包括 CNN、LSTM、Transformer 和 Mamba。表 4.1 展示了此比較的結果。為確保模型間的公平性，我們控制了總參數數量，使所有架構具有大致相同的可學習參數數量。如表 4.1 所示，HL-Mamba 變體在未見過的測試集上始終優於單體基線。在這些變體中，基於 Mamba 的 HL-Mamba 實現在所有評估指標上均表現出最高性能。我們進一步通過檢查模型複雜度和推理延遲，將 HL-Mamba 與現有的神經模型 CoNNear 進行比較。儘管這兩種方法服務於不同目的——CoNNear 專注於預測生理上準確的聽覺神經反應，而 HL-Mamba 則設計用於感知導向的信號轉換——但由於兩者都追求實時聽力損失模擬，因此比較仍然是合理的。根據表 4.3，CoNNear 大約包含 1170 萬個參數，而 HL-Mamba 只有 145 萬個。除了更緊湊之外，HL-Mamba 還可以處理嘈雜的語音輸入，並靈活地處理各種聽力圖配置，為具有不同聲學條件和個性化聽力需求的實際應用提供了優勢。

Table 4.3 also reports inference latency for each model. MSBG does not support parallelism and therefore cannot be executed on GPU, so its GPU runtime is listed as NA. In contrast, HL-Mamba (Mamba) relies on a CUDA-accelerated selective scan kernel for its core computations, which currently runs only on GPU; therefore, only GPU inference time is measured. In terms of speed, HL-Mamba (Mamba) achieves roughly a 46× acceleration on GPU compared to MSBG on CPU. For CPU-compatible variants such as HL-Mamba (LSTM), the inference time is 0.016 seconds, corresponding to a 60× speedup over MSBG’s 0.970 seconds. We additionally implemented CoNNear for comparison. Despite its relatively large parameter size of 11.7 million, it yields fast inference on our hardware, with GPU computation taking 0.099 seconds and notably faster CPU computation taking 0.025 seconds. The faster CPU runtime compared to GPU is likely due to the batch size of 1 adopted in this experiment, which limits the computational benefits of GPU parallelization.

表 4.3 也報告了每個模型的推斷延遲。MSBG 不支持並行處理，因此無法在 GPU 上執行，其 GPU 運行時間記為 NA。相比之下，HL-Mamba (Mamba) 的核心計算依賴於一個 CUDA 加速的選擇性掃描內核，該內核目前僅在 GPU 上運行；因此，只測量了 GPU 推斷時間。在速度方面，HL-Mamba (Mamba) 在 GPU 上的速度大約是 MSBG 在 CPU 上的 46 倍。對於與 CPU 兼容的變體，例如 HL-Mamba (LSTM)，推斷時間為 0.016 秒，相當於比 MSBG 的 0.970 秒快 60 倍。我們還額外實現了 CoNNear 以進行比較。儘管其參數大小相對較大（1170 萬），但在我們的硬件上，它的推斷速度很快，GPU 計算耗時 0.099 秒，而 CPU 計算速度明顯更快，僅需 0.025 秒。CPU 運行時間比 GPU 更快，可能是由於本實驗採用的批次大小為 1，這限制了 GPU 並行化的計算優勢。

## 4.3 Ablation Study

Previous studies on hearing loss modeling typically estimate only the magnitude spectrum and reuse the input phase when reconstructing the waveform. We initially followed this strategy as well, but our experiments showed that phase information is highly influential in MSBG-based simulation. To examine this effect, we performed an ablation study using HL-Mamba (Mamba). As reported in Table 4.2, simultaneously predicting magnitude and phase yields substantial improvements over magnitude-only prediction across all evaluation criteria (STOI MSE, PESQ MSE, and waveform MSE). Notably, STOI MSE decreases from 0.0041 to 0.0006, indicating improved intelligibility, while PESQ MSE drops from 2.3579 to 0.0782, signifying better perceptual quality. At the waveform level, the MSE decreases from 0.0986 to 0.0669, demonstrating that phase estimation is essential for perceptual fidelity as well as accurate reconstruction.

以往關於聽力損失建模的研究通常只估計幅度譜，並在重建波形時重複使用輸入相位。我們最初也遵循此策略，但我們的實驗顯示，在基於 MSBG 的模擬中，相位信息具有高度影響力。為檢驗此效應，我們使用 HL-Mamba (Mamba) 進行了一項消融研究。如表 4.2 所示，同時預測幅度和相位，在所有評估標準（STOI MSE、PESQ MSE 和波形 MSE）上，都比僅預測幅度的結果有顯著改善。值得注意的是，STOI MSE 從 0.0041 降至 0.0006，表明可懂度有所提高；而 PESQ MSE 從 2.3579 降至 0.0782，意味著感知質量更佳。在波形層面，MSE 從 0.0986 降至 0.0669，證明了相位估計對於感知保真度和準確重建至關重要。

In addition, many speech-related systems that incorporate audiometric information adopt a straightforward strategy of concatenating the audiogram vector with the spectral features before feeding them to the model. We initially applied this method, but observed that it could not sufficiently represent the relationship between hearing profiles and acoustic cues. To overcome this limitation, we designed a lightweight Audiogram Encoder that maps the 8-dimensional audiogram vector to a frequency-aligned representation, which is then added as a separate channel alongside the magnitude and phase features. According to the results in Table 4.4, the proposed encoder consistently reduces STOI MSE, PESQ MSE, and waveform-level MSE, indicating that more effective integration of individualized hearing characteristics leads to improved hearing loss simulation.

此外，許多與語音相關的系統在整合聽力測量信息時，採用了一種直接的策略，即在將頻譜特徵輸入模型之前，將聽力圖向量與其串接。我們最初也採用了這種方法，但觀察到它無法充分表達聽力特徵與聲學線索之間的關係。為克服此限制，我們設計了一個輕量級的聽力圖編碼器，將 8 維的聽力圖向量映射到一個與頻率對齊的表示，然後將其作為一個獨立的通道，與幅度和相位特徵一起添加。根據表 4.4 的結果，所提出的編碼器持續降低了 STOI MSE、PESQ MSE 和波形級 MSE，表明更有效地整合個人化聽力特徵可以改善聽力損失模擬。

[Image]
Table 4.2: Phase prediction is essential. Magnitude-only severely degrades perceptual accuracy.
表格 4.2：相位預測至關重要。僅使用幅度會嚴重降低感知準確性。

[Image]
Table 4.3: Comparison of runtime of MSBG and HL-Mamba on different devices. We measured the inference time required to process a 1-second, 44.1 kHz audio signal using an Intel Xeon Gold 6152 CPU and an NVIDIA RTX 3090 GPU.
表格 4.3：MSBG 和 HL-Mamba 在不同設備上的運行時間比較。我們測量了使用 Intel Xeon Gold 6152 CPU 和 NVIDIA RTX 3090 GPU 處理 1 秒長、44.1 kHz 音頻信號所需的推斷時間。

[Image]
Table 4.4: Audiogram Encoder improves hearing-profile conditioning.
表格 4.4：聽力圖編碼器改善了聽力特徵的條件化。

## 4.4 Qualitative Evaluation

To assess how well each model reconstructs speech affected by hearing loss, we perform a qualitative comparison of the log-magnitude spectrograms generated by seven different models against the ground-truth reference (Fig. 4.2). Among all models, HL-Mamba (Mamba) produces spectrograms that most closely match the reference, effectively retaining harmonic structures and high-frequency components. The HL-Mamba variants with CNN, LSTM, and Transformer blocks also capture important spectral characteristics, though they exhibit slight energy imbalance or mild distortion. In contrast, the baseline models introduce more noticeable artifacts: the CNN and LSTM versions show reduced clarity and insufficient high-frequency detail, while the Transformer baseline struggles to generate an accurate spectral representation.

為了評估各模型重建受聽力損失影響的語音之能力，我們對七種不同模型生成的對數幅度譜圖與真實參考（圖 4.2）進行了定性比較。在所有模型中，HL-Mamba（Mamba）產生的譜圖與參考最為接近，有效地保留了諧波結構和高頻分量。帶有 CNN、LSTM 和 Transformer 模塊的 HL-Mamba 變體也捕捉到了重要的頻譜特徵，儘管它們表現出輕微的能量不平衡或輕度失真。相比之下，基線模型引入了更明顯的假影：CNN 和 LSTM 版本顯示出清晰度降低和高頻細節不足，而 Transformer 基線則難以生成準確的頻譜表示。

[Image]
Figure 4.2: Spectrogram comparison. (A) Ground truth (B-D) Baseline CNN/LSTM/Transformer (E-H) HL-Mamba with Mamba/CNN/LSTM/Transformer Mamba reconstruction best preserves harmonic structure + high-frequency detail.
圖 4.2：語音頻譜圖比較。(A) 真實參考 (B-D) 基線 CNN/LSTM/Transformer (E-H) HL-Mamba 與 Mamba/CNN/LSTM/Transformer 的重構，Mamba 重構能最佳地保留諧波結構與高頻細節。

## 4.5 Compensator Training with HL-Mamba

To facilitate end-to-end hearing loss compensation, recent approaches such as NeuroAMP [1] have incorporated audiogram-aware processing directly into neural architectures, effectively replacing conventional modular pipelines with personalized, data-driven amplification. Motivated by this trend, we propose a complementary strategy in which a trainable compensator is connected to a frozen, perceptually informed simulator (HL-Mamba). In this setup, the compensator learns to modify its input so that the resulting output from the simulator matches each listener's hearing profile.

為促進端到端的聽力損失補償，近期的研究方法如 NeuroAMP [1] 已將聽力圖感知處理直接整合到神經架構中，有效地以個人化、數據驅動的放大方式取代了傳統的模塊化流程。受此趨勢啟發，我們提出了一種互補策略，其中一個可訓練的補償器連接到一個凍結的、基於感知的模擬器（HL-Mamba）。在此設置中，補償器學習修改其輸入，使得模擬器的最終輸出與每個聽者的聽力特徵相匹配。

HL-Mamba is lightweight, differentiable, and does not rely on time-domain alignment with clean references, characteristics that make it well-suited for building an end-to-end hearing loss compensation pipeline. Leveraging these properties, we explore a novel application: integrating a pre-trained HL-Mamba model with a trainable compensator to perform personalized hearing enhancement, as illustrated in Fig. 4.3. Both training and evaluation are conducted using the VoiceBank dataset [28].

HL-Mamba 具有輕量、可微分的特性，且不依賴與純淨參考樣本的時域對齊，這些特點使其非常適合建構端到端的聽損補償流程。利用這些特性，我們探索了一個創新的應用：將預訓練的 HL-Mamba 模型與一個可訓練的補償器整合，以實現個人化的聽力增強，如圖 4.3 所示。訓練與評估均使用 VoiceBank 資料集 [28] 進行。

The compensator is designed to convert the input waveform into a personalized, compensated signal. This processed signal is then fed into the frozen HL-Mamba model, and the system is trained such that the final output closely matches the clean speech. In this configuration, the compensator acts as an individualized module that adapts the signal according to a user’s hearing characteristics. Importantly, HL-Mamba itself remains fixed during training; our intention is to first confirm that the integration of HL-Mamba into the optimization loop is feasible and effective. The compensator follows the HL-Mamba architecture with one major modification in the magnitude path: instead of the original masking-based magnitude encoder, we employ a mapping-based strategy aimed at restoring or enhancing missing spectral information. This modification aligns the network with the objective of signal compensation by enabling the generation of gain-adjusted outputs that improve intelligibility for hearing-impaired listeners.

補償器旨在將輸入波形轉換為個人化的補償信號。此處理後的信號接著被送入凍結的 HL-Mamba 模型，系統經訓練後使其最終輸出能緊密匹配清晰語音。在此配置中，補償器作為一個個人化模組，根據使用者的聽力特性調整信號。重要的是，HL-Mamba 本身在訓練過程中保持固定；我們的目的在於首先確認將 HL-Mamba 整合至優化迴路是可行且有效的。補償器遵循 HL-Mamba 的架構，但在幅度路徑上做了一項主要修改：我們採用基於映射的策略，旨在恢復或增強缺失的頻譜資訊，而非原始的基於遮蔽的幅度編碼器。此修改使網絡與信號補償的目標一致，能生成增益調整後的輸出，以改善聽力受損者的語音清晰度。

To quantify the impact of the proposed approach, we evaluate the system using the HASPI metric, which measures perceptual speech intelligibility. As presented in Table 4.5, the compensator produces a notable increase in average HASPI score from 0.428 to 0.616 (∆ = +0.187). The improvement is statistically significant according to both a paired t-test (t = -24.113, p < 0.00001) and a Wilcoxon signed-rank test (W = 292426.0, p < 0.00001). The resulting effect size (d = 0.594) is moderately large, indicating meaningful perceptual gains across samples.

為量化所提方法的影響，我們使用 HASPI 指標評估系統，該指標衡量感知語音清晰度。如表 4.5 所示，補償器使平均 HASPI 分數從 0.428 顯著提升至 0.616 (∆ = +0.187)。此改善在配對 t 檢定 (t = -24.113, p < 0.00001) 和 Wilcoxon 符號秩檢定 (W = 292426.0, p < 0.00001) 中均達到統計顯著性。所得效應值 (d = 0.594) 為中等偏大，顯示在樣本間存在有意義的感知增益。

[Image]
Figure 4.3: Trainable compensator with HL-Mamba. Simulator remains frozen—only compensator updates to improve intelligibility.
圖 4.3：使用 HL-Mamba 的可訓練補償器。模擬器保持凍結——僅補償器更新以提高清晰度。

[Image]
Table 4.5: HASPI improvement using compensator.
表格 4.5：使用補償器後 HASPI 的改善情況。

Results confirm that HL-Mamba enables learnable, profile-aware enhancement.

結果證實 HL-Mamba 可實現可學習的、個人化輪廓感知的增強。

# Chapter 5 Conclusions

Neuro-MSBG is proposed as a differentiable simulation framework for hearing loss, aimed at resolving several limitations inherent in traditional models, such as high delay variability, limited compatibility with learning-based pipelines, and the lack of scalable parallel execution. Unlike conventional approaches that rely on explicit alignment to a clean reference signal, Neuro-MSBG can be embedded directly into end-to-end architectures without inducing temporal desynchronization, thereby preventing degradation in objective metrics such as STOI and PESQ. Its highly parallelizable structure, together with low latency, enables deployment at scale and supports real-time speech processing applications.

Neuro-MSBG 作為一個可微分的聽力損失模擬框架被提出，旨在解決傳統模型中固有的幾個限制，例如高延遲變異性、與基於學習的管道的有限兼容性以及缺乏可擴展的並行執行。與依賴於與乾淨參考信號的顯式對齊的傳統方法不同，Neuro-MSBG 可以直接嵌入到端到端架構中，而不會引起時間上的去同步，從而防止了諸如 STOI 和 PESQ 等客觀指標的性能下降。其高度可並行化的結構以及低延遲，使其能夠大規模部署並支持實時語音處理應用。

Among the developed variants, the Mamba-based model demonstrates substantial computational advantages, running approximately 46× faster than the original MSBG implementation and reducing the processing time of one second of audio from 0.970 s to 0.021 s. The LSTM-based version further improves efficiency, achieving an inference time of only 0.016 s, corresponding to roughly a 60× speedup. Experimental evidence also indicates that jointly estimating magnitude and phase leads to notable improvements in simulated speech intelligibility and perceived quality, reaching SRCC scores of 0.9247 for STOI and 0.8671 for PESQ.

在已開發的變體中，基於 Mamba 的模型展現出顯著的計算優勢，其運行速度比原始 MSBG 實現快約 46 倍，將一秒音訊的處理時間從 0.970 秒減少到 0.021 秒。基於 LSTM 的版本進一步提高了效率，實現了僅 0.016 秒的推斷時間，相當於約 60 倍的加速。實驗證據還表明，聯合估計幅度和相位可以顯著改善模擬語音的可懂度和感知質量，STOI 和 PESQ 的 SRCC 分數分別達到了 0.9247 和 0.8671。

In addition, we introduce an Audiogram Encoder that transforms audiogram representations into frequency-aligned latent features. This design surpasses simple vector concatenation strategies and offers a more faithful representation of individualized hearing characteristics.

此外，我們引入了一個聽力圖編碼器，可將聽力圖表示轉換為頻率對齊的潛在特徵。此設計優於簡單的向量串接策略，並能更忠實地呈現個人化的聽力特徵。

# References

[1] Shafique Ahmed, Ryandhimas E. Zezario, Hui-Guan Yuan, Amir Hussain, Hsin-Min Wang, Wei-Ho Chung, and Yu Tsao. Neuroamp: A novel end-to-end general purpose deep neural amplifier for personalized hearing aids. arXiv preprint arXiv:2502.10822, 2025.

[2] Yuxuan Ai and Zhen-Hua Ling. Neural speech phase prediction based on parallel estimation architecture and anti-wrapping losses. In Proc. ICASSP, pages 1–5, 2023.

[3] Thomas Baer and Brian C. J. Moore. Effects of spectral smearing on the intelligibility of sentences in noise. The Journal of the Acoustical Society of America, 94:1229–1241, 1993.

[4] Thomas Baer and Brian C. J. Moore. Effects of spectral smearing on the intelligibility of sentences in the presence of interfering speech. The Journal of the Acoustical Society of America, 95(4):2050–2062, 1993.

[5] Arthur Van Den Broucke, Deepak Baby, and Sarah Verhulst. Hearing-impaired bio-inspired cochlear models for real-time auditory applications. In Proc. INTERSPEECH, pages 2842–2846, 2020.

[6] Rong Chao, Wen-Huang Cheng, Moreno La Quatra, Sabato Marco Siniscalchi, Chao-Han Huck Yang, Szu-Wei Fu, and Yu Tsao. An investigation of incorporating mamba for speech enhancement. In Proc. SLT, pages 302–308, 2024.

[7] Fotios Drakopoulos and Sarah Verhulst. A differentiable optimisation framework for the design of individualised dnn-based hearing-aid strategies. In Proc. ICASSP, pages 351–355, 2022.

[8] Fotios Drakopoulos and Sarah Verhulst. A neural-network framework for the design of individualised hearing-loss compensation. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 31:2395–2409, 2023.

[9] Fotios Drakopoulos, Arthur Van Den Broucke, and Sarah Verhulst. A dnn-based hearing-aid strategy for real-time processing: One size fits all. In Proc. ICASSP, pages 1–5, 2023.

[10] Simone Graetzer, Jon Barker, Trevor J. Cox, Michael Akeroyd, John F. Culling, Graham Naylor, Eszter Porter, and Rhoddy Viveros Muñoz. Clarity-2021 challenges: Machine learning challenges for advancing hearing aid processing. In Proc. INTERSPEECH, pages 686–690, 2021.

[11] Volker Hohmann. Frequency analysis and synthesis using a gammatone filterbank. Acta Acustica united with Acustica, 88:433–442, 2002.

[12] Toshio Irino. Hearing impairment simulator based on auditory excitation pattern playback: WHIS. IEEE Access, 11:78419–78430, 2023.

[13] Kathryn H. Arehart James M. Kates. The hearing-aid speech perception index (HASPI). Speech Communication, 65:75–93, 2014.

[14] J. M. Kates and K. H. Arehart. The hearing-aid audio quality index (HAAQI). IEEE/ACM Transactions on Audio, Speech, and Language Processing, 24:354–365, 2016.

[15] James Kates and Kathryn Arehart. The hearing-aid speech quality index (HASQI). AES: Journal of the Audio Engineering Society, 58:363–381, 2010.

[16] Peter Leer, Jesper Jensen, Zheng-Hua Tan, Jan Østergaard, and Lars Bramsløw. How to train your ears: Auditory-model emulation for large-dynamic-range inputs and mild-to-severe hearing losses. IEEE/ACM Trans. Audio, Speech and Lang. Proc., 32:2006–2020.

[17] Ye-Xin Lu, Yang Ai, and Zhen-Hua Ling. MP-SENet: A speech enhancement model with parallel denoising of magnitude and phase spectra. In Proc. INTERSPEECH, pages 3834–3838, 2023.

[18] Brian C. J. Moore and Brian R. Glasberg. Simulation of the effects of loudness recruitment and threshold elevation on the intelligibility of speech in quiet and in a background of speech. The Journal of the Acoustical Society of America, 94(4):2050–2062, 1993.

[19] Brian C. J. Moore, Brian R. Glasberg, and Thomas Baer. A model for the prediction of thresholds, loudness, and partial loudness. Journal of the Audio Engineering Society, 45:224–240, 1997.

[20] Antony W. Rix, John G. Beerends, Michael P. Hollier, and Andries P. Hekstra. Perceptual evaluation of speech quality (PESQ)—a new method for speech quality assessment of telephone networks and codecs. In Proc. ICASSP, pages 749–752, 2001.

[21] Gerardo Roa Dabike, Jon Barker, John F. Culling, et al. The ICASSP SP Cadenza challenge: Music demixing/remixing for hearing aids. arXiv preprint arXiv:2310.03480, 2023.

[22] Malcolm Slaney. Auditory toolbox version 2: A MATLAB toolbox for auditory modeling work. Technical Report 1998-010, Interval Research Corporation, 1998.

[23] Cees H. Taal, Richard C. Hendriks, Richard Heusdens, and Jesper Jensen. An algorithm for intelligibility prediction of time-frequency weighted noisy speech. IEEE Transactions on Audio, Speech, and Language Processing, 19(7):2125–2136, 2011.

[24] Joachim Thiemann, Nobutaka Ito, and Emmanuel Vincent. The diverse environments multichannel acoustic noise database (DEMAND): A database of multichannel environmental noise recordings. In Proc. Meetings on Acoustic, pages 1–6, 2013.

[25] Zehai Tu, Ning Ma, and Jon Barker. Dhasp: Differentiable hearing aid speech processing. In Proc. ICASSP, pages 296–300, 2021.
