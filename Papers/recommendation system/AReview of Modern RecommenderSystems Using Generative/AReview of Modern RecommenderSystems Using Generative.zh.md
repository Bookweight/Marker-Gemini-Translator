---
title: AReview of Modern RecommenderSystems Using Generative
field: recommendation system
status: Imported
created_date: 2026-01-12
pdf_link: "[[AReview of Modern RecommenderSystems Using Generative.pdf]]"
tags:
  - paper
  - Recommend_System
---


# 現代推薦系統使用生成模型之綜述 (Gen-RecSys)

## **Published in:** 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD), 2024

Yashar Deldjoo
Polytechnic University of Bari
Bari, Italy
deldjooy@acm.org

Yashar Deldjoo
巴里理工大學
義大利，巴里
deldjooy@acm.org

Zhankui He
University of California
La Jolla, USA
zhh004@ucsd.edu

Zhankui He
加州大學
美國，拉霍亞
zhh004@ucsd.edu

Julian McAuley
University of California
La Jolla, USA
jmcauley@ucsd.edu

Julian McAuley
加州大學
美國，拉霍亞
jmcauley@ucsd.edu

Anton Korikov
University of Toronto
Toronto, Canada
anton.korikov@mie.utoronto.ca

Anton Korikov
多倫多大學
加拿大，多倫多
anton.korikov@mie.utoronto.ca

Scott Sanner
University of Toronto
Toronto, Canada
ssanner@mie.utoronto.ca

Scott Sanner
多倫多大學
加拿大，多倫多
ssanner@mie.utoronto.ca

Arnau Ramisa
Amazon*
Palo Alto, USA
aramisay@amazon.com

Arnau Ramisa
亞馬遜*
美國，帕羅奧圖
aramisay@amazon.com

René Vidal
Amazon*
Palo Alto, USA
vidalr@seas.upenn.edu

René Vidal
亞馬遜*
美國，帕羅奧圖
vidalr@seas.upenn.edu

Maheswaran Sathiamoorthy
Bespoke Labs
Santa Clara, USA
mahesh@bespokelabs.ai

Maheswaran Sathiamoorthy
Bespoke Labs
美國，聖塔克拉拉
mahesh@bespokelabs.ai

Atoosa Kasirzadeh
University of Edinburgh
Edinburgh, UK
atoosa.kasirzadeh@gmail.com

Atoosa Kasirzadeh
愛丁堡大學
英國，愛丁堡
atoosa.kasirzadeh@gmail.com

Silvia Milano
University of Exeter and LMU Munich
Munich, Germany
milano.silvia@gmail.com

Silvia Milano
埃克塞特大學與慕尼黑大學
德國，慕尼黑
milano.silvia@gmail.com

***

### ABSTRACT

Traditional recommender systems typically use user-item rating histories as their main data source. However, deep generative models now have the capability to model and sample from complex data distributions, including user-item interactions, text, images, and videos, enabling novel recommendation tasks. This comprehensive, multidisciplinary survey connects key advancements in RS using Generative Models (Gen-RecSys), covering: interaction-driven generative models; the use of large language models (LLM) and textual data for natural language recommendation; and the integration of multimodal models for generating and processing images/videos in RS. Our work highlights necessary paradigms for evaluating the impact and harm of Gen-RecSys and identifies open challenges. This survey accompanies a tutorial presented at ACM KDD’24, with supporting materials provided at: https://encr.pw/vDhLq.

### 摘要

傳統推薦系統通常使用使用者-項目評分歷史作為其主要資料來源。然而，深度生成模型現在有能力對複雜的資料分佈進行建模和取樣，包括使用者-項目互動、文本、圖像和影片，從而實現新穎的推薦任務。這份全面性的跨學科綜述連結了使用生成模型（Gen-RecSys）的推薦系統（RS）之關鍵進展，涵蓋：互動驅動的生成模型；使用大型語言模型（LLM）和文本資料進行自然語言推薦；以及整合多模態模型以在推薦系統中生成和處理圖像/影片。我們的研究強調了評估 Gen-RecSys 影響和危害的必要範式，並指出了開放的挑戰。本綜述附帶於 ACM KDD'24 發表的一份教學，輔助材料可在以下網址取得：https://encr.pw/vDhLq。

### CCS CONCEPTS

*   **Information systems → Recommender systems.**

### CCS 概念

*   **資訊系統 → 推薦系統。**

### KEYWORDS

Generative Models, Recommender Systems, GANs, VAEs, LLMs, Multimodal, vLLMs, Ethical and Societal Considerations

### 關鍵詞

生成模型、推薦系統、GANs、VAEs、LLMs、多模態、vLLMs、倫理與社會考量

*This work does not relate to the author's position at Amazon.

*此研究與作者在亞馬遜的職位無關。

This work is licensed under a Creative Commons Attribution International 4.0 License.

此作品根據創用 CC 姓名標示國際 4.0 授權條款進行授權。

KDD '24, August 25-29, 2024, Barcelona, Spain
© 2024 Copyright held by the owner/author(s).
ACM ISBN 979-8-4007-0490-1/24/08
https://doi.org/10.1145/3637528.3671474

KDD '24, 2024年8月25-29日, 西班牙，巴塞隆納
© 2024 版權由所有者/作者持有。
ACM ISBN 979-8-4007-0490-1/24/08
https://doi.org/10.1145/3637528.3671474

### ACM Reference Format:

Yashar Deldjoo, Zhankui He, Julian McAuley, Anton Korikov, Scott Sanner, Arnau Ramisa, René Vidal, Maheswaran Sathiamoorthy, Atoosa Kasirzadeh, and Silvia Milano. 2024. A Review of Modern Recommender Systems Using Generative Models (Gen-RecSys). In Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD '24), August 25-29, 2024, Barcelona, Spain. ACM, New York, NY, USA, 11 pages. https://doi.org/10.1145/3637528.3671474

### ACM 參考文獻格式：

Yashar Deldjoo, Zhankui He, Julian McAuley, Anton Korikov, Scott Sanner, Arnau Ramisa, René Vidal, Maheswaran Sathiamoorthy, Atoosa Kasirzadeh, and Silvia Milano. 2024. A Review of Modern Recommender Systems Using Generative Models (Gen-RecSys). In Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD '24), August 25-29, 2024, Barcelona, Spain. ACM, New York, NY, USA, 11 pages. https://doi.org/10.1145/3637528.3671474

## 1 INTRODUCTION

Advancements in generative models have significantly impacted the evolution of recommender systems (RS). Traditional RS, which relied on capturing user preferences and item features within a specific domain - often referred to as "narrow experts" are now being complemented and, in some instances, surpassed by generative models. These models have introduced innovative ways of conceptualizing and implementing recommendations. Specifically, modern generative models learn to represent and sample from complex data distributions, including not only user-item interaction histories but also text and image content, unlocking these data modalities for novel and interactive recommendation tasks.

## 1 緒論

生成模型的進步對推薦系統（RS）的發展產生了重大影響。傳統的推薦系統依賴於在特定領域內捕捉使用者偏好和項目特徵——通常被稱為「狹義專家」——現在正被生成模型所補充，在某些情況下甚至被超越。這些模型引入了創新性的概念化和實施推薦的方式。具體來說，現代生成模型學習從複雜的資料分佈中表示和取樣，不僅包括使用者與項目的互動歷史，還包括文本和圖像內容，從而為新穎和互動式的推薦任務解鎖了這些資料模態。

Moreover, advances in natural language processing (NLP) through the introduction of large language models (LLMs) such as ChatGPT [121] and Gemini [148] have showcased remarkable emergent capabilities [165], including reasoning, in-context few-shot learning, and access to extensive open-world information within their pre-trained parameters. Because of their broad generalist abilities, these pretrained generative models have opened up an exciting new research space for a wide variety of recommendation applications (see Table 1), e.g., enhanced personalization, improved conversational interfaces, and richer explanation generation, among others.

此外，自然語言處理（NLP）領域的進步，透過引入如 ChatGPT [121] 和 Gemini [148] 等大型語言模型（LLMs），展現了卓越的湧現能力 [165]，包括推理、情境中少樣本學習，以及在其預訓練參數中取用廣泛的開放世界資訊。由於其廣泛的通用能力，這些預訓練的生成模型為各種推薦應用（見表1）開闢了一個令人興奮的新研究領域，例如，增強的個人化、改進的對話介面，以及更豐富的解釋生成等。

***

Figure 1: Overview of the areas of interest in generative models in recommendation.

圖 1：生成模型在推薦領域中感興趣的領域概覽。

The core of generative models lies in their ability to model and sample from their training data distribution for various inferential purposes, which enables two primary modes of application for RS:

生成模型的核心在於其能夠對訓練資料分佈進行建模和取樣，以用於各種推論目的，這使得推薦系統（RS）有兩種主要的應用模式：

(1) Directly trained models. This approach trains generative models, such as VAE-CF (Variational AutoEncoders for Collaborative Filtering) [97] (cf. Section 2.1) directly on user-item interaction data to predict user preferences, without using large, diverse pre-training datasets. These models learn the probability distribution of items a user might like based on their previous interactions.

(1) 直接訓練模型。此方法直接在使用者-項目互動資料上訓練生成模型，例如 VAE-CF（用於協同過濾的變分自動編碼器）[97]（參見第 2.1 節），以預測使用者偏好，而無需使用大型、多樣化的預訓練資料集。這些模型根據使用者先前的互動，學習使用者可能喜歡的項目之機率分佈。

(2) Pretrained models. This strategy uses models pretrained on diverse data (text, images, videos) to understand complex patterns, relationships, and contexts that often exhibit (emergent) generalization abilities to a range of novel tasks [165]. Among a variety of applications, this survey covers the use of pretrained Gen-RecSys models in the following settings:

(2) 預訓練模型。此策略使用在多樣化資料（文本、圖像、影片）上預訓練的模型來理解複雜的模式、關係和情境，這些模型通常對一系列新任務展現出（湧現的）泛化能力[165]。在眾多應用中，本綜述涵蓋了在以下設定中使用預訓練的 Gen-RecSys 模型：

*   Zero- and Few-shot Learning (cf. Section 3.2.1), using in-context learning (ICL) for broad understanding without extra training.
*   零樣本和少樣本學習（參見第 3.2.1 節），使用情境學習（ICL）以在無需額外訓練的情況下獲得廣泛的理解。
*   Fine-Tuning (cf. Section 3.3), adjusting model parameters using specific datasets for tailored recommendations.
*   微調（參見第 3.3 節），使用特定資料集調整模型參數以進行客製化推薦。
*   Retrieval-Augmented Generation (RAG) (cf. Section 3.3), integrating information retrieval with generative modeling for contextually relevant outputs.
*   檢索增強生成（RAG）（參見第 3.3 節），將資訊檢索與生成建模相結合，以產生與情境相關的輸出。
*   Feature Extraction for Downstream Recommendation (cf. Section 3.4), e.g., generating embeddings or token sequences for complex content representation.
*   為下游推薦任務提取特徵（參見第 3.4 節），例如，為複雜內容表示生成嵌入或權杖序列。
*   Multimodal Approaches (cf. Section 4), jointly using multiple data types such as text, image, and video to enhance and improve the recommendation experience.
*   多模態方法（參見第 4 節），聯合使用多種資料類型，如文本、圖像和影片，以增強和改善推薦體驗。

### 1.1 Recent Surveys and Our Contributions

Recent Relevant Surveys. Recent surveys have marked significant advancements in the field. We highlight our contributions and distinguish our survey by its comprehensive and unique approach.

### 1.1 近期調查與我們的貢獻

近期相關調查。近期的調查標誌著該領域的重大進展。我們強調我們的貢獻，並以其全面而獨特的方法來區分我們的調查。

*   Deldjoo et al. [33] explore GAN-based RS across four different recommendation scenarios (graph-based, collaborative, hybrid, context-aware).
*   Deldjoo 等人 [33] 探討了基於 GAN 的推薦系統在四種不同推薦情境（基於圖形、協同、混合、情境感知）中的應用。
*   Li et al. [95] explore training strategies and learning objectives of LLMs for RS.
*   Li 等人 [95] 探討了用於推薦系統的 LLM 的訓練策略和學習目標。
*   Wu et al. [171] discuss both the use of LLMs to generate RS input tokens or embeddings as well as the use of LLMs as an RS;
*   Wu 等人 [171] 討論了使用 LLM 生成推薦系統輸入權杖或嵌入，以及將 LLM 作為推薦系統本身的使用。
*   Lin et al. [99] focus on adapting LLMs in RS, detailing various tasks and applications. Fan et al. [38] overview LLMs in RS, emphasizing pre-training, fine-tuning, and prompting, while Vats et al. [150] review LLM-based RS, introducing a heuristic taxonomy for categorization.
*   Lin 等人 [99] 專注於在推薦系統中調整 LLM，詳細介紹了各種任務和應用。Fan 等人 [38] 概述了推薦系統中的 LLM，強調了預訓練、微調和提示。而 Vats 等人 [150] 回顧了基於 LLM 的推薦系統，並引入了一種啟發式分類法。
*   Huang et al. [67], explore using foundation models (FMs) in RS.
*   Huang 等人 [67] 探索了在推薦系統中使用基礎模型（FMs）。
*   Wang et al. [158] introduce GeneRec, a next-gen RS that personalizes content through AI generators and interprets user instructions to gather user preferences.
*   Wang 等人 [158] 介紹了 GeneRec，這是一種下一代推薦系統，它透過人工智慧生成器個人化內容，並解釋使用者指令以收集使用者偏好。

While the mentioned surveys offer crucial insights, their scope is often limited to LLMs [38, 95, 99, 150, 171] or, more broadly, FMs [67] and/or specific models such as GANs [33], without considering the wider spectrum of generative models and data modalities. The work by [158] provides a more relevant survey on Gen-RecSys although their work is mostly on personalized content generation.

雖然上述調查提供了重要的見解，但其範圍通常僅限於 LLM [38, 95, 99, 150, 171] 或更廣泛的基礎模型（FM）[67] 和/或特定模型，如 GAN [33]，而未考慮更廣泛的生成模型和資料模態。 [158] 的研究提供了關於 Gen-RecSys 更相關的調查，儘管其研究主要集中在個人化內容生成上。

Core Contributions. Figure 1 illustrates the structure of our Gen-RecSys survey. It categorizes data sources, recommendation models, and scenarios, extending to system evaluation and challenges. We present a systematic approach to deconstructing the Gen-RecSys recommendation process into distinct components and methodologies. Our contributions are summarized as follows.

核心貢獻。圖 1 說明了我們 Gen-RecSys 調查的結構。它對資料來源、推薦模型和情境進行了分類，並擴展到系統評估和挑戰。我們提出了一種系統化的方法，將 Gen-RecSys 推薦過程分解為不同的組件和方法。我們的貢獻總結如下。

(1) Our survey is broader in scope than the surveys mentioned above, encompassing not just LLMs but a wide array of generative models in RS.

(1) 我們的調查範圍比上述調查更廣，不僅涵蓋了 LLM，還涵蓋了推薦系統中廣泛的生成模型。

(2) We have chosen to classify these models based on the type of data and modality they are used for, such as user-item data (cf. Section 2), text-driven (cf. Section 3), and multimodal (cf. Section 4) models, as shown in the Rec. Scenario layer.

(2) 我們選擇根據模型使用的資料類型和模態對這些模型進行分類，例如使用者-項目資料（參見第 2 節）、文本驅動（參見第 3 節）和多模態（參見第 4 節）模型，如推薦情境層所示。

(3) Within each modality discussion, we provide an in-depth exploration of deep generative model paradigms as shown in the Model layer, yet with a broader scope that spans multiple contexts and use cases, offering a critical analysis of their roles and effectiveness in respective sections.

(3) 在每個模態的討論中，我們對模型層中所示的深度生成模型範式進行了深入的探索，其範圍更廣，涵蓋了多個情境和用例，並對其在各個部分中的作用和有效性進行了批判性分析。

(4) We study the evaluation of Gen-RecSys with finer details, shedding light on multiple aspects such as benchmarks, evaluation for impact and harm relative to multiple stakeholders, and conversational evaluation. This evaluation framework is particularly notable as it helps to understand the complex challenges intrinsic to Gen-RecSys.

(4) 我們更詳細地研究了 Gen-RecSys 的評估，闡明了基準、相對於多個利害關係人的影響和危害評估以及對話評估等多個方面。這個評估框架特別值得注意，因為它有助於理解 Gen-RecSys 固有的複雜挑戰。

(5) We discuss several open research challenges and issues. Our survey benefits from the expertise of scholars/industry practitioners from diverse institutions and disciplines.

(5) 我們討論了幾個開放的研究挑戰和問題。我們的調查得益於來自不同機構和學科的學者/行業從業人員的專業知識。

## 2 GENERATIVE MODELS FOR INTERACTION-DRIVEN RECOMMENDATION

Interaction-driven recommendation is a setup where only the user-item interactions (e.g., "user A clicks item B") are available, which is the most general setup studied in RS. In this setup, we concentrate on the inputs of user-item interactions and outputs of item-recommended lists or grids rather than richer inputs or outputs from other modalities such as textual reviews. Even though no textual or visual information is involved, generative models [47, 64, 84, 144, 149] still show their unique usefulness. In this section, we examine the paradigms of generative models for recommendation tasks with user-item interactions, including auto-encoding models [84], auto-regressive models [64, 149], generative adversarial networks [47], diffusion models [144] and more.

## 2 互動驅動推薦的生成模型

互動驅動推薦是一種僅提供使用者-項目互動（例如，「使用者 A 點擊項目 B」）的設定，這是推薦系統中研究最普遍的設定。在此設定中，我們專注於使用者-項目互動的輸入和項目推薦列表或網格的輸出，而非來自其他模態（如文本評論）的更豐富的輸入或輸出。儘管不涉及文本或視覺資訊，生成模型 [47, 64, 84, 144, 149] 仍然展現其獨特的用處。在本節中，我們將檢視用於使用者-項目互動推薦任務的生成模型範式，包括自動編碼模型 [84]、自回歸模型 [64, 149]、生成對抗網路 [47]、擴散模型 [144] 等。

### 2.1 Auto-Encoding Models

Auto-encoding models learn to reconstruct their inputs. This capability allows them to be used for various purposes, including denoising, representation learning, and generation tasks.

### 2.1 自動編碼模型

自動編碼模型學習重建其輸入。此功能使其可用於各種目的，包括去噪、表示學習和生成任務。

#### 2.1.1 Preliminaries: Denoising Auto-Encoding Models.

Denoising Auto-Encoding models are a group of models that learn to recover the original inputs from a corrupted version of the inputs. Traditionally, denoising auto-encoding models refer to a group of Denoising Autoencoders [140, 151] with hidden layers as a "bottleneck". For example, AutoRec [140] tries to reconstruct the input vector, which is partially observed. More broadly, BERT-like models [35, 146, 172] are also treated as denoising auto-encoding models. Such models recover corrupted (i.e., masked) inputs through stacked self-attention blocks [59, 146]. For example, BERT4Rec [146] is trained to predict masked items in given user historical interaction sequences. Therefore, BERT-like [35] models can be used for next-item prediction in the inference phase [59, 146].

#### 2.1.1 初步介紹：去噪自動編碼模型。

去噪自動編碼模型是一組學習從輸入的損壞版本中恢復原始輸入的模型。傳統上，去噪自動編碼模型指的是一組以隱藏層為「瓶頸」的去噪自動編碼器 [140, 151]。例如，AutoRec [140] 試圖重建部分觀察到的輸入向量。更廣泛地說，類 BERT 模型 [35, 146, 172] 也被視為去噪自動編碼模型。此類模型透過堆疊的自註意力區塊 [59, 146] 恢復損壞（即遮蔽）的輸入。例如，BERT4Rec [146] 經過訓練，可預測給定使用者歷史互動序列中的遮蔽項目。因此，類 BERT [35] 模型可用於推論階段的下一個項目預測 [59, 146]。

#### 2.1.2 Variational Auto-Encoding Models.

Variational Autoencoders (VAEs) are models that learn stochastic mappings from an input x from a often complicated probability distribution p to a probability distribution q. This distribution, q, is typically simple (e.g., a normal distribution), enabling the use of a decoder to generate outputs x by sampling from q [84]. VAEs find wide applications in traditional RS, particularly for collaborative filtering [97], sequential recommendation [137] and slate generation [29, 74, 106]. Compared to Denoising Autoencoders, VAEs often demonstrate superior performance in collaborative filtering due to stronger modeling assumptions, such as VAE-CF [97]. Additionally, Conditional VAE (CVAE) [145] models learn distributions of preferred recommendation lists for a given user. This makes them useful for generating those lists beyond a greedy ranking schema. Examples like ListCVAE [74] and PivotCVAE [106] use VAEs to generate entire recommendation lists rather than solely ranking individual items.

#### 2.1.2 變分自動編碼模型。

變分自動編碼器（VAEs）是一種模型，它學習從一個通常複雜的機率分佈 p 的輸入 x 到一個機率分佈 q 的隨機映射。這個分佈 q 通常很簡單（例如，常態分佈），使得可以使用解碼器透過從 q 中取樣來生成輸出 x [84]。VAEs 在傳統推薦系統中有廣泛的應用，特別是用於協同過濾 [97]、序列推薦 [137] 和板塊生成 [29, 74, 106]。與去噪自動編碼器相比，由於更強的建模假設，VAEs 在協同過濾中通常表現出更優越的性能，例如 VAE-CF [97]。此外，條件 VAE（CVAE）[145] 模型學習給定使用者的偏好推薦列表的分佈。這使得它們可用於生成超越貪婪排名方案的列表。像 ListCVAE [74] 和 PivotCVAE [106] 這樣的例子使用 VAEs 來生成整個推薦列表，而不僅僅是排名單個項目。

### 2.2 Auto-Regressive Models

Given an input sequence x, at step i, auto-regressive models [12] learn the conditional probability distribution p(xi|x<i), where x<i represents the subsequence before step i. Auto-regressive models are primarily used for sequence modeling [12, 36, 149]. In RS, they find wide applications in session-based or sequential recommendations [63, 80], model attacking [181], and bundle recommendations [7, 66], with recurrent neural networks [7, 63, 66], self-attentive models [80], and more.

### 2.2 自迴歸模型

給定一個輸入序列 x，在步驟 i，自迴歸模型 [12] 學習條件機率分佈 p(xi|x<i)，其中 x<i 表示步驟 i 之前的子序列。自迴歸模型主要用於序列建模 [12, 36, 149]。在推薦系統中，它們在基於會話或序列的推薦 [63, 80]、模型攻擊 [181] 和捆綁推薦 [7, 66] 中有廣泛的應用，使用遞歸神經網路 [7, 63, 66]、自註意力模型 [80] 等。

#### 2.2.1 Recurrent Auto-Regressive Models.

Recurrent neural networks (RNNs) [25, 64] have been use to predict the next item in session-based and sequential recommendations, such as GRU4Rec [63] and its variants [62, 182] (e.g., predicting the next set of items in basket or bundle recommendations, such as set2set [66] and BGN [7]). Moreover, using the auto-regressive generative nature of recurrent networks, researchers extract model-generated user behavior sequences, which are used in the research of model attacking [181].

#### 2.2.1 遞歸自迴歸模型。

遞歸神經網路（RNNs）[25, 64] 已被用於預測基於會話和序列推薦中的下一個項目，例如 GRU4Rec [63] 及其變體 [62, 182]（例如，預測購物籃或捆綁推薦中的下一組項目，例如 set2set [66] 和 BGN [7]）。此外，利用遞歸網路的自迴歸生成特性，研究人員提取模型生成的使用者行為序列，這些序列用於模型攻擊的研究 [181]。

#### 2.2.2 Self-Attentive Auto-Regressive Models.

Self-attentive models replace the recurrent unit with self-attention and related modules, inspired by transformers [149]. This group of models can be used in session-based recommendation and sequential recommendation [80, 100, 124, 170], next-basket or bundle prediction [179], and model attacking [181]. Meanwhile, the benefits of self-attentive models are that they handle long-term dependencies better than RNNs and enable parallel training [149]. Additionally, self-attentive models are the de-facto option for pre-trained models [35] and large language models [17, 18, 165], which is gaining traction in RS. More details about using such language models for recommendations will be discussed in Section 3.

#### 2.2.2 自註意力自迴歸模型。

受 transformers [149] 的啟發，自註意力模型用自註意力及相關模塊取代了循環單元。這組模型可用於基於會話的推薦和順序推薦 [80, 100, 124, 170]、下一個購物籃或捆綁預測 [179] 以及模型攻擊 [181]。同時，自註意力模型的優點是它們比 RNNs 更好地處理長期依賴性，並能進行並行訓練 [149]。此外，自註意力模型是預訓練模型 [35] 和大型語言模型 [17, 18, 165] 的事實標準選項，這在推薦系統中越來越受歡迎。關於使用此類語言模型進行推薦的更多細節將在第 3 節中討論。

### 2.3 Generative Adversarial Networks

Generative adversarial networks (GANs) [47, 115] are composed of two primary components: a generator network and a discriminator network. These networks engage in adversarial training to enhance the performance of both the generator and the discriminator. GANs are used in RS for multiple purposes [19, 23, 153]. In the interaction-driven setup, GANs are proposed for selecting informative training samples [19, 153], for example, in IRGAN [153, 156], the generative retrieval model is leveraged to sample negative items. Meanwhile, GANs synthesize user preferences or interactions to augment training data [21, 157]. Additionally, GANs have shown effectiveness in generating recommendation lists or pages, such as [23] in whole-page recommendation settings.

### 2.3 生成對抗網路

生成對抗網路（GANs）[47, 115] 由兩個主要部分組成：一個生成器網路和一個判別器網路。這些網路進行對抗性訓練，以增強生成器和判別器的性能。GANs 在推薦系統中用於多種目的 [19, 23, 153]。在互動驅動的設定中，GANs 被提議用於選擇資訊豐富的訓練樣本 [19, 153]，例如，在 IRGAN [153, 156] 中，利用生成式檢索模型來取樣負面項目。同時，GANs 合成使用者偏好或互動以增強訓練資料 [21, 157]。此外，GANs 在生成推薦列表或頁面方面也顯示出有效性，例如 [23] 在整頁推薦設定中。

### 2.4 Diffusion Models

Diffusion models [144] generate outputs through a two-step process: (1) corrupting inputs into noise via a forward process, and (2) learning to recover the original inputs from the noise iteratively in a reverse process. Their impressive generative capabilities have attracted growing interest from the RS community.

### 2.4 擴散模型

擴散模型 [144] 透過兩步驟過程生成輸出：(1) 透過前向過程將輸入損壞為雜訊，以及 (2) 學習在反向過程中從雜訊中迭代恢復原始輸入。其令人印象深刻的生成能力已引起推薦系統社群日益增長的興趣。

First, a group of works [152, 159] learns users' future interaction probabilities through diffusion models. For example, DiffRec [159] predicts users' future interactions using corrupted noises from the users' historical interactions. Second, another group of works [104, 173] focuses on diffusion models for training sequence augmentation, showing promising results in alleviating the data sparsity and long-tail user problems in sequential recommendation.

首先，一組研究 [152, 159] 透過擴散模型學習使用者的未來互動機率。例如，DiffRec [159] 使用來自使用者歷史互動的損壞雜訊來預測使用者的未來互動。其次，另一組研究 [104, 173] 專注於用於訓練序列增強的擴散模型，在緩解序列推薦中的資料稀疏性和長尾使用者問題方面顯示出有希望的結果。

### 2.5 Other Generative Models

In addition to the previously mentioned generative models, RS also draw upon other types of generative models. For instance, VASER [191] leverages normalizing flows [132] (and VAEs [84]) for session-based recommendation. GFN4Rec [105], on the other hand, adapts generative flow networks [11, 122] for listwise recommendation. Furthermore, IDNP [37] utilizes generative neural processes [43, 44] for sequential recommendation. In summary, various generative models are explored in RS, even in settings without textual or visual modalities.

### 2.5 其他生成模型

除了前面提到的生成模型，推薦系統還利用了其他類型的生成模型。例如，VASER [191] 利用歸一化流 [132]（和 VAEs [84]）進行基於會話的推薦。另一方面，GFN4Rec [105] 則將生成流網路 [11, 122] 用於列表式推薦。此外，IDNP [37] 利用生成神經過程 [43, 44] 進行序列推薦。總之，即使在沒有文本或視覺模態的設定中，各種生成模型也在推薦系統中得到了探索。

## 3 LARGE LANGUAGE MODELS IN RECOMMENDATION

While language has been leveraged by content-based RS for over three decades [107], the advent of pretrained LLMs and their emergent abilities for generalized, multi-task natural language (NL) reasoning [17, 18, 165] has ushered in a new stage of language-based recommendation. Critically, NL constitutes a unified, expressive, and interpretable medium that can represent not only item features or user preferences, but also user-system interactions, recommendation task descriptions, and external knowledge [45]. For instance, items are often associated with rich text including titles, descriptions, semi-structured textual metadata, and reviews. Similarly, user preferences can be articulated in NL in many forms, such as reviews, search queries, liked item descriptions, and dialogue utterances.

## 3 推薦系統中的大型語言模型

雖然基於內容的推薦系統利用語言已有三十多年的歷史 [107]，但預訓練大型語言模型（LLM）的出現及其在通用、多任務自然語言（NL）推理方面的湧現能力 [17, 18, 165] 開創了基於語言的推薦新階段。至關重要的是，自然語言構成了一個統一、富有表現力且可解釋的媒介，不僅可以表示項目特徵或使用者偏好，還可以表示使用者-系統互動、推薦任務描述和外部知識 [45]。例如，項目通常與豐富的文本相關聯，包括標題、描述、半結構化文本元資料和評論。同樣，使用者偏好可以透過多種形式的自然語言來表達，例如評論、搜尋查詢、喜歡的項目描述和對話語句。

Pretrained LLMs provide new ways to exploit this textual data: recent research (e.g., [40, 45, 58, 138, 143]) has shown that in many domains, LLMs have learned useful reasoning abilities for making and explaining item recommendations based on user preferences as well as facilitating conversational recommendation dialogues. As discussed below, these pretrained abilities can be further augmented through prompting (e.g., [103, 138, 143]), fine-tuning (e.g., [45, 54, 78, 189]), retrieval (e.g., [27, 40, 65, 83, 154]), and other external tools (e.g., [40, 160, 183].

預訓練的 LLM 提供了利用這些文本資料的新方法：最近的研究（例如，[40, 45, 58, 138, 143]）表明，在許多領域，LLM 已經學會了有用的推理能力，可以根據使用者偏好提出和解釋項目推薦，並促進對話式推薦。如下文所述，這些預訓練的能力可以透過提示（例如，[103, 138, 143]）、微調（例如，[45, 54, 78, 189]）、檢索（例如，[27, 40, 65, 83, 154]）和其他外部工具（例如，[40, 160, 183]）進一步增強。

We next proceed to survey the developments in LLM-based RS's, first discussing encoder-only LLMs for dense retrieval and cross-encoding (Section 3.1) followed by generative NL recommendation and explanation with sequence-to-sequence (seq2seq) LLMs (Section 3.2). We then review the complementary use of RS and LLMs covering RAG (Section 3.3) and LLM-based feature extraction (Section 3.4), before concluding with a review of conversational recommendation methods (Section 3.5).

接下來，我們將調查基於 LLM 的推薦系統的發展，首先討論用於密集檢索和交叉編碼的僅編碼器 LLM（第 3.1 節），然後是使用序列到序列（seq2seq）LLM 的生成式自然語言推薦和解釋（第 3.2 節）。接著，我們將回顧推薦系統和 LLM 的互補使用，涵蓋 RAG（第 3.3 節）和基於 LLM 的特徵提取（第 3.4 節），最後以對話式推薦方法的回顧作結（第 3.5 節）。

### 3.1 Encoder-only LLM Recommendation

### 3.1 僅使用編碼器的 LLM 推薦

#### 3.1.1 Recommendation as Dense Retrieval.

A common task is to retrieve the most relevant items given a NL preference statement using item texts, for which dense retrieval has become a key tool. Dense retrievers [39] produce a ranked list of documents given a query by evaluating the similarity (e.g., dot product or cosine similarity) between encoder-only LLM document embeddings and the query embedding. They are highly scalable tools (especially when used with approximate search libraries like FAISS¹) because documents and queries are encoded separately, allowing for dense vector indexing of documents before querying. To use dense retrieval for recommendation [123], first, a component of each item's text content, such as its title, description, reviews, etc., is treated as a document and a dense item index is constructed. Then, a query is formed by some NL user preference description, for instance: an actual search query, the user's recently liked item titles, or a user utterance in a dialogue.

#### 3.1.1 以密集檢索方式進行推薦。

一個常見的任務是，利用項目文本，根據自然語言偏好陳述來檢索最相關的項目，而密集檢索已成為此任務的關鍵工具。密集檢索器 [39] 透過評估僅編碼器 LLM 文件嵌入與查詢嵌入之間的相似度（例如，點積或餘弦相似度），為給定查詢生成一份排序的文件列表。它們是高度可擴展的工具（特別是與 FAISS¹ 等近似搜尋函式庫一起使用時），因為文件和查詢是分開編碼的，允許在查詢前對文件進行密集的向量索引。為了將密集檢索用於推薦 [123]，首先，將每個項目文本內容的一個組成部分（例如其標題、描述、評論等）視為一份文件，並建構一個密集的項目索引。然後，透過一些自然語言使用者偏好描述來形成查詢，例如：實際的搜尋查詢、使用者最近喜歡的項目標題，或對話中的使用者話語。

Several recent works explore recommendation as standard dense retrieval with retrievers that are off-the-shelf [54, 123, 185] and fine-tuned [65, 91, 116]. More complex dense retrieval methods include review-based retrieval with contrastive BERT fine-tuning [2] and multi-aspect query decomposition [86], and the use of a second-level encoder to fuse the embedding of a user's recently liked items into a user embedding before scoring [92, 168].

最近的一些研究將推薦視為標準的密集檢索，使用了現成的 [54, 123, 185] 和微調過的 [65, 91, 116] 檢索器。更複雜的密集檢索方法包括使用對比式 BERT 微調的基於評論的檢索 [2] 和多面向查詢分解 [86]，以及在評分前使用第二層編碼器將使用者最近喜歡的項目的嵌入融合到使用者嵌入中 [92, 168]。

#### 3.1.2 Recommendation via LLM Item-Preference Fusion.

Several works approach rating prediction by jointly embedding NL item and preference descriptions in LLM cross-encoder architectures with an MLP rating prediction head [126, 169, 176, 188, 190]. Such fusion-in-encoder methods often exhibit strong performance because they allow interaction between user and item representations, but are much more computationally expensive than dense retrieval and thus may be best used for small item sets or as rerankers [116].

#### 3.1.2 透過 LLM 項目偏好融合進行推薦。

一些研究透過在帶有 MLP 評分預測頭的 LLM 交叉編碼器架構中聯合嵌入自然語言項目和偏好描述來處理評分預測 [126, 169, 176, 188, 190]。這種編碼器內融合方法通常表現出強大的性能，因為它們允許使用者和項目表示之間的互動，但它們的計算成本遠高於密集檢索，因此可能最適合用於小型項目集或作為重排器 [116]。

### 3.2 LLM-based Generative Recommendation

In LLM-based generative recommendation, tasks are expressed as token sequences – called prompts – which form an input to a seq2seq LLM. The LLM then generates another token sequence to address the task - with example outputs including: a recommended list of item titles/ids [54, 111, 138, 143], a rating [9, 78], or an explanation [45, 50, 93, 94, 118]. These methods rely on the pretraining of LLMs on large text corpora to provide knowledge about a wide range of entities, human preferences, and commonsense reasoning that can be used directly for recommendation or leveraged to improve generalization and reduce domain-specific data requirements for fine-tuning or prompting [18, 165].

### 3.2 基於 LLM 的生成式推薦

在基於 LLM 的生成式推薦中，任務被表示為權杖序列——稱為提示——它們構成 seq2seq LLM 的輸入。然後，LLM 生成另一個權杖序列來解決該任務——範例輸出包括：推薦的項目標題/ID 列表 [54, 111, 138, 143]、評分 [9, 78] 或解釋 [45, 50, 93, 94, 118]。這些方法依賴於 LLM 在大型文本語料庫上的預訓練，以提供關於廣泛實體、人類偏好和常識推理的知識，這些知識可以直接用於推薦，或用於提高泛化能力並減少微調或提示的領域特定資料需求 [18, 165]。

#### 3.2.1 Zero- and Few- Shot Generative Recommendation.

Several recent publications [78, 103, 138, 143] have evaluated with off-the-shelf LLM generative recommendation, focusing on domains that are prevalent in the LLM pre-training corpus such as movie and book recommendation. Specifically, these methods construct a prompt with a NL description of user preference (often using a sequence of recently liked item titles) and an instruction to recommend the next k item titles [103, 138, 143] or predict a rating [78, 103]. While, overall, untuned LLMs underperform supervised CF methods trained on sufficient data [78, 143], they are competitive in near cold-start settings [138, 143]. Few-shot prompting (or in-context learning), in which a prompt contains examples of input-output pairs, typically outperforms zero-shot prompting [138].

#### 3.2.1 零樣本與少樣本生成式推薦。

最近的幾篇出版物 [78, 103, 138, 143] 評估了現成的 LLM 生成式推薦，重點關注在 LLM 預訓練語料庫中普遍存在的領域，例如電影和書籍推薦。具體來說，這些方法建構一個帶有使用者偏好之自然語言描述的提示（通常使用最近喜歡的項目標題序列），以及推薦接下來 k 個項目標題 [103, 138, 143] 或預測評分 [78, 103] 的指令。雖然總體而言，未經調整的 LLM 在有足夠資料訓練的監督式協同過濾方法上表現不佳 [78, 143]，但它們在接近冷啟動的設定中具有競爭力 [138, 143]。少樣本提示（或情境學習），其中提示包含輸入-輸出對的範例，通常優於零樣本提示 [138]。

#### 3.2.2 Tuning LLMs for Generative Recommendation.

To improve an LLM's generative recommendation performance and add knowledge to its internal parameters, multiple works focus on fine-tuning [9, 45, 54, 78, 111] and prompt-tuning [26, 94, 189] strategies. Recent works fine-tune LLMs on NL input/output examples constructed from user-system interaction history and task descriptions for rating prediction [9, 78] and sequential recommendation [54, 111], or in the case of P5 [45], both preceding tasks plus top-k recommendation, explanation generation, and review summarization. Other recommendation works study prompt tuning approaches [26, 94, 189], which adjust LLM behaviour by tuning a set of continuous (or soft) prompt vectors as an alternative to tuning internal LLM weights.

#### 3.2.2 為生成式推薦微調 LLM。

為了提升 LLM 的生成式推薦性能並將知識加入其內部參數，多項研究專注於微調 [9, 45, 54, 78, 111] 和提示微調 [26, 94, 189] 策略。近期的研究針對評分預測 [9, 78] 和序列推薦 [54, 111]，在由使用者-系統互動歷史和任務描述建構的自然語言輸入/輸出範例上微調 LLM，或者在 P5 [45] 的案例中，除了前述任務外，還加上前 k 個推薦、解釋生成和評論摘要。其他推薦研究則探討提示微調方法 [26, 94, 189]，此方法透過微調一組連續（或軟性）的提示向量來調整 LLM 行為，作為調整內部 LLM 權重的替代方案。

Generative Explanation. A line of recent work focuses on explanation generation where training explanations are extracted from reviews, since reviews often express reasons why a user decided to interact with an item. Techniques include fine-tuning [45, 94, 161], prompt-tuning [93, 94], chain-of-thought prompting [129], and controllable decoding [50, 118, 119, 174] - where additional predicted parameters such as ratings steer LLM decoding.

生成式解釋。最近的一系列研究專注於解釋生成，其中訓練解釋是從評論中提取的，因為評論通常表達了使用者決定與某個項目互動的原因。技術包括微調 [45, 94, 161]、提示調整 [93, 94]、思維鏈提示 [129] 和可控解碼 [50, 118, 119, 174]——其中額外的預測參數（如評分）會引導 LLM 的解碼。

### 3.3 Retrieval Augmented Recommendation

Adding knowledge to an LLM internal memory through tuning can improve performance, but it requires many parameters and re-tuning for every system update. An alternative is retrieval-augmented generation (RAG) [15, 70, 87], which conditions output on information from an external source such as a dense retriever (Section 3.1). RAG methods facilitate online updates, reduce hallucinations, and generally require fewer LLM parameters since knowledge is externalized [15, 70, 112].

### 3.3 檢索增強推薦

透過微調將知識加入 LLM 內部記憶體可以提升效能，但每次系統更新都需要大量參數和重新微調。另一種選擇是檢索增強生成（RAG）[15, 70, 87]，它根據外部來源（如密集檢索器，見 3.1 節）的資訊來決定輸出。RAG 方法有助於線上更新、減少幻覺，並且通常需要較少的 LLM 參數，因為知識是外部化的 [15, 70, 112]。

RAG has recently begun to be explored for recommendation, with the most common approach being to first use a retriever or RS to construct a candidate item set based on a user query or interaction history, and then prompt an encoder-decoder LLM to rerank the candidate set [27, 65, 154, 166, 175]. For RAG-based explanation generation, Xie et al. [174] generate queries based on interaction history to retrieve item reviews which are used as context to generate an explanation of the recommendation. RAG is also emerging as a key paradigm in conversational recommendation (c.f. Sec 3.5): for example, RAG is used in [40] to retrieve relevant user preference descriptions from a user “memory” module to guide dialogue, and by Kemper et al. [83] to retrieve information from an item's reviews to answer user questions.

RAG 最近開始被用於推薦領域，最常見的方法是先使用檢索器或推薦系統，根據使用者查詢或互動歷史建構候選項目集，然後提示編碼器-解碼器 LLM 對候選集進行重新排序 [27, 65, 154, 166, 175]。對於基於 RAG 的解釋生成，Xie 等人 [174] 根據互動歷史生成查詢以檢索項目評論，並將其用作生成推薦解釋的上下文。RAG 也正成為對話式推薦的關鍵範式（參見第 3.5 節）：例如，在 [40] 中，RAG 用於從使用者「記憶體」模組中檢索相關的使用者偏好描述以引導對話；Kemper 等人 [83] 則用其從項目評論中檢索資訊以回答使用者問題。

### 3.4 LLM-based Feature Extraction

Conversely to how RS or retrievers are used in RAG to obtain inputs for LLMs (Section 3.3), LLMs can also be used to generate inputs for RS [54, 60, 91, 116, 130, 180]. For instance: LLM2-BERT4Rec [54] initializes BERT4Rec (Section 2.1.1) item embeddings of item texts; Query-SeqRec [60] includes LLM query embeddings as inputs to a transformer-based recommender; and TIGER [130] first uses an LLM to embed item text, then quantizes this embedding into a semantic ID, and finally trains a T5-based RS to generate new IDs given a user's item ID history. Similarly, MINT [116] and GPT4Rec [91] produce inputs for a dense retriever by prompting an LLM to generate a query given a user's interaction history.

### 3.4 基於 LLM 的特徵提取

與在 RAG 中使用推薦系統或檢索器為 LLM 獲取輸入（第 3.3 節）相反，LLM 也可用於為推薦系統生成輸入 [54, 60, 91, 116, 130, 180]。例如：LLM2-BERT4Rec [54] 初始化 BERT4Rec（第 2.1.1 節）的項目文本的項目嵌入；Query-SeqRec [60] 將 LLM 查詢嵌入作為基於 transformer 的推薦器的輸入；而 TIGER [130] 首先使用 LLM 嵌入項目文本，然後將此嵌入量化為語義 ID，最後訓練一個基於 T5 的推薦系統，以根據使用者的項目 ID 歷史生成新的 ID。同樣，MINT [116] 和 GPT4Rec [91] 透過提示 LLM 根據使用者的互動歷史生成查詢，為密集檢索器產生輸入。

### 3.5 Conversational Recommendation

The recent advances in LLMs have made fully NL system-user dialogues a feasible and novel recommendation interface, bringing in a new stage of conversational recommendation (ConvRec) research. This direction studies the application of LLMs in multi-turn, multi-task, and mixed-initiative NL recommendation conversations [40, 72], introducing dialogue history as a rich new form of interaction data. Specifically, ConvRec includes the study and integration of diverse conversational elements such as dialogue management, recommendation, explanation, QA, critiquing, and preference elicitation [72, 110]. While some research [58] approaches ConvRec with a monolithic LLM such as GPT4, other works rely on an LLM to facilitate NL dialogue and integrate calls to a recommender module which generates item recommendations based on dialogue or interaction history [5, 22, 52, 77, 96, 160, 175]. Further research advances ConvRec system architectures with multiple tool-augmented LLM modules, incorporating components for dialogue management, explanation generation, and retrieval [40, 42, 75, 83, 162, 183].

### 3.5 對話式推薦

最近 LLM 的進展使得全自然語言的系統-使用者對話成為一種可行且新穎的推薦介面，開啟了對話式推薦（ConvRec）研究的新階段。這個方向研究 LLM 在多輪、多任務和混合主動式自然語言推薦對話中的應用 [40, 72]，引入對話歷史作為一種豐富的新型互動資料。具體來說，ConvRec 包括對各種對話元素的學習與整合，例如對話管理、推薦、解釋、問答、評論和偏好引出 [72, 110]。雖然一些研究 [58] 使用單一的 LLM（如 GPT4）來處理 ConvRec，但其他研究則依賴 LLM 來促進自然語言對話，並整合對推薦模組的呼叫，該模組根據對話或互動歷史生成項目推薦 [5, 22, 52, 77, 96, 160, 175]。進一步的研究則推進了具有多個工具增強 LLM 模組的 ConvRec 系統架構，整合了用於對話管理、解釋生成和檢索的組件 [40, 42, 75, 83, 162, 183]。

## 4 GENERATIVE MULTIMODAL RECOMMENDATION SYSTEMS

In recent years, users have come to expect richer interactions than simple text or image queries. For instance, they might provide a picture of a desired product along with a natural language modification (e.g., a dress like the one in the picture but in red). Additionally, users want to visualize recommendations to see how a product fits their use case, such as how a garment might look on them or how a piece of furniture might look in their room. These interactions require new RS that can discover unique attributes in each modality. In this section, we discuss RS that utilize multiple data modalities. In Sections 4.1-4.2 we discuss motivations and challenges to the design of multimodal RS. In Sections 4.3-4.4 we review contrastive and generative approaches to multimodal RS, respectively.

## 4 生成式多模態推薦系統

近年來，使用者期望比簡單的文本或圖像查詢更豐富的互動。例如，他們可能會提供一張所需產品的圖片，並附上自然語言的修改（例如，一件像圖片中那樣但顏色是紅色的連衣裙）。此外，使用者希望將推薦視覺化，以了解產品如何符合其使用情境，例如一件衣服穿在他們身上的樣子，或一件家具在他們房間裡的樣子。這些互動需要能夠在每種模態中發現獨特屬性的新型推薦系統。在本節中，我們將討論利用多種資料模態的推薦系統。在第 4.1-4.2 節中，我們將討論設計多模態推薦系統的動機和挑戰。在第 4.3-4.4 節中，我們將回顧多模態推薦系統的對比式和生成式方法。

### 4.1 Why Multimodal Recommendation?

Retailers often have multimodal information about their customers and products, including product descriptions, images and videos, customer reviews and purchase history. However, existing RS typically process each source independently and then combine the results by fusing unimodal relevance scores.

### 4.1 為何需要多模態推薦？

零售商通常擁有關於其顧客和產品的多模態資訊，包括產品描述、圖片和影片、顧客評論和購買歷史。然而，現有的推薦系統通常獨立處理每個來源，然後透過融合單模態相關性分數來組合結果。

In practice, there are many use cases in which such a "late fusion" approach may be insufficient to satisfy the customer needs. One such use case is the cold start problem: when user behavioral data cannot be used to recommend existing products to new customers, or new products to existing customers, it is useful to gather diverse information about the items so that preference information can be transferred from existing products or customers to new ones.

在實務上，有許多使用案例中，這種「後期融合」方法可能不足以滿足顧客需求。其中一個使用案例是冷啟動問題：當無法使用使用者行為資料向新顧客推薦現有產品，或向現有顧客推薦新產品時，收集關於項目的多樣化資訊就很有用，以便將偏好資訊從現有產品或顧客轉移到新產品或顧客身上。

Another use case occurs when different modalities are needed to understand the user request. For example, to answer the request "best metal and glass black coffee table under $300 for my living room", the system would need to reason about the appearance and shape of the item in context with the appearance and shape of other objects in the customer room, which cannot be achieved by searching with either text or image independently. Other examples of multimodal requests include an image or audio of the desired item together with text modification instructions (e.g., a song like the sound clip provided but in acoustic), or a complementary related product (e.g., a kickstand for the bicycle in the picture).

另一個使用案例發生在需要不同模態來理解使用者請求時。例如，要回答「為我的客廳找一張 300 美元以下的最佳金屬和玻璃黑色咖啡桌」的請求，系統需要根據客廳中其他物體的 外觀和形狀來推斷該物品的外觀和形狀，這無法透過單獨使用文本或圖像搜尋來實現。多模態請求的其他範例包括所需物品的圖像或音訊以及文本修改說明（例如，一首像所提供音訊片段但為原聲的歌曲），或一個互補的相關產品（例如，圖中自行車的腳架）。

A third use case for multimodal understanding is in RS with complex outputs, such as virtual try-on features or intelligent multimodal conversational shopping assistants.

多模態理解的第三個應用案例是在具有複雜輸出的推薦系統中，例如虛擬試穿功能或智慧多模態對話式購物助理。

### 4.2 Challenges to Multimodal Recommendation

The development of multimodal RS faces several challenges. First, collecting data to train multimodal systems (e.g., image-text-image triplets) is significantly harder than for unimodal systems. As a result, annotations for some modalities may be incomplete [128].

### 4.2 多模態推薦的挑戰

多模態推薦系統的發展面臨幾個挑戰。首先，收集訓練多模態系統的資料（例如，圖像-文本-圖像三元組）比單模態系統要困難得多。因此，某些模態的標註可能不完整 [128]。

Second, combining different data modalities to improve recommendation results is not simple. For instance, existing contrastive learning approaches [73, 89, 90, 127] map each data modality to a common latent space in which all modalities are approximately aligned. However, such approaches often capture information that is shared across modalities (e.g., text describing visual attributes), but they overlook complementary aspects that could benefit recommendations (e.g., text describing non visual attributes) [49]. In general we would like the modalities to compensate for one another and result in a more complete joint representation. While fusion-based approaches [89, 90] do learn a joint multimodal representation, ensuring the alignment of information that is shared and leaving some flexibility to capture complementary information across modalities remains a challenge. Third, learning multimodal models requires orders of magnitude more data than learning models for individual data modalities.

其次，結合不同的資料模態以改善推薦結果並非易事。例如，現有的對比學習方法 [73, 89, 90, 127] 將每個資料模態映射到一個共同的潛在空間，其中所有模態大致對齊。然而，這種方法通常捕捉跨模態共享的資訊（例如，描述視覺屬性的文本），但忽略了可能有利於推薦的互補方面（例如，描述非視覺屬性的文本）[49]。一般來說，我們希望模態能夠相互補償，從而產生更完整的聯合表示。雖然基於融合的方法 [89, 90] 確實學習了聯合多模態表示，但確保共享資訊的對齊，並保留一些靈活性以捕捉跨模態的互補資訊仍然是一個挑戰。第三，學習多模態模型所需的資料量比學習單個資料模態的模型多出幾個數量級。

Despite these challenges, we believe multimodal generative models will become the standard approach. Indeed, recent literature shows significant advances on the necessary components to achieve effective multimodal generative models for RS, including (1) the use of LLMs and diffusion models to generate synthetic data for labeling purposes [16, 117, 135], (2) high quality unimodal encoders and decoders [56, 85], (3) better techniques for aligning the latent spaces from multiple modalities into a shared one [46, 89, 127], (4) efficient re-parametrizations and training algorithms [71], and (5) techniques to inject structure to the learned latent space to make the problem tractable [144].

儘管存在這些挑戰，我們相信多模態生成模型將成為標準方法。事實上，最近的文獻顯示，在實現有效的多模態生成模型推薦系統所需組件方面取得了重大進展，包括 (1) 使用 LLM 和擴散模型生成用於標記目的的合成資料 [16, 117, 135]，(2) 高品質的單模態編碼器和解碼器 [56, 85]，(3) 將多個模態的潛在空間對齊到一個共享空間的更好技術 [46, 89, 127]，(4) 高效的重參數化和訓練演算法 [71]，以及 (5) 將結構注入學習到的潛在空間以使問題易於處理的技術 [144]。

### 4.3 Contrastive Multimodal Recommendation

As discussed before 4.2, learning multimodal generative models is very difficult because we need to not only learn a latent representation for each modality but also ensure that they are aligned. One way to address this challenge is to first learn an alignment between multiple modalities and then learn a generative model on "well-aligned" representations. In this subsection, we discuss two representative contrastive learning approaches: CLIP and ALBEF.

### 4.3 對比式多模態推薦

如前 4.2 節所述，學習多模態生成模型非常困難，因為我們不僅需要為每個模態學習一個潛在表示，還需要確保它們是對齊的。解決這個挑戰的一種方法是，首先學習多個模態之間的對齊，然後在「良好對齊」的表示上學習一個生成模型。在本小節中，我們將討論兩種代表性的對比學習方法：CLIP 和 ALBEF。

Contrastive Language-Image Pre-training (CLIP) [127] is a popular approach, in which the task is to project images and associated text into the same point of the embedding space with parallel image and text encoders. This is achieved with a symmetric cross-entropy loss over the rows and columns of the cosine similarity matrix between all possible pairs of images and text in a training minibatch.

對比式語言-圖像預訓練（CLIP）[127] 是一種流行的方法，其任務是使用並行的圖像和文本編碼器，將圖像和相關文本投影到嵌入空間的同一個點上。這是透過在訓練小批次中所有可能的圖像和文本對之間的餘弦相似度矩陣的行和列上使用對稱交叉熵損失來實現的。

Align Before you Fuse (ALBEF) [90] augments CLIP with a multimodal encoder that fuses the text and image embeddings, and proposes three objectives to pre-train the model: Image-text contrastive learning (ITC), masked language modeling (MLM), and image-text matching (ITM). The authors also introduce momentum distillation to provide pseudo-labels in order to compensate for the potentially incomplete or wrong text descriptions in the noisy web training data. Using their proposed architecture and training objectives, ALBEF obtains better results than CLIP in several zero-shot and fine-tuned multimodal benchmarks, despite using orders of magnitude less images for pre-training.

在融合前對齊（ALBEF）[90] 以一個融合文本和圖像嵌入的多模態編碼器來增強 CLIP，並提出三個目標來預訓練模型：圖文對比學習（ITC）、遮罩語言建模（MLM）和圖文匹配（ITM）。作者還引入了動量蒸餾以提供偽標籤，以補償嘈雜的網路訓練資料中可能不完整或錯誤的文本描述。使用他們提出的架構和訓練目標，ALBEF 在幾個零樣本和微調的多模態基準測試中取得了比 CLIP 更好的結果，儘管預訓練使用的圖像數量少了幾個數量級。

Contrastive-based alignment has shown impressive zero-shot classification and retrieval results [8, 61, 120], and has been successfully fine-tuned to a multitude of tasks, such as object detection [48], segmentation [192] or action recognition [69]. The same alignment objective has also been used between other modalities [24, 51, 68], and with multiple modalities at the same time [46].

基於對比的對齊在零樣本分類和檢索方面取得了令人印象深刻的成果 [8, 61, 120]，並已成功微調至多種任務，例如物件偵測 [48]、分割 [192] 或動作辨識 [69]。相同的對齊目標也已用於其他模態之間 [24, 51, 68]，以及同時用於多個模態 [46]。

### 4.4 Generative Multimodal Recommendation

Despite their advantages, the performance of purely contrastive RS often suffers from data sparsity and uncertainty [163]. Generative models address these issues by imposing suitable structures on their latent spaces. Moreover, generative models allow for more complex recommendations, e.g., those requiring to synthesize an image. In what follows, we discuss thee representative generative approaches: VAEs, diffusion models, and multimodal LLMs.

### 4.4 生成式多模態推薦

儘管有其優點，純對比式推薦系統的性能常因資料稀疏性和不確定性而受影響 [163]。生成模型透過在其潛在空間上施加適當的結構來解決這些問題。此外，生成模型允許更複雜的推薦，例如需要合成圖像的推薦。接下來，我們將討論三種代表性的生成方法：VAEs、擴散模型和多模態 LLMs。

Multimodal VAES: While VAEs (see Section 2.1.2) could be applied directly to multimodal data, a better approach that leverages modality specific encoders and decoders trained on large corpus of data is to partition both the input and latent spaces per modality, say image and text. However, this approach reduces the multimodal VAE to two independent VAEs, one per modality. In ContrastVAE [163], both modalities are aligned by adding a contrastive loss between the unimodal latent representations to the ELBO objective. Experiments show that ContrastVAE improves upon purely contrastive models by adequately modeling data uncertainty and sparsity, and being robust to perturbations in the latent space.

多模態 VAE：雖然 VAE（見 2.1.2 節）可以直接應用於多模態資料，但一種更好的方法是利用在大型資料語料庫上訓練的特定模態編碼器和解碼器，將輸入和潛在空間按模態（例如圖像和文本）進行劃分。然而，這種方法將多模態 VAE 簡化為兩個獨立的 VAE，每個模態一個。在 ContrastVAE [163] 中，透過在 ELBO 目標中加入單模態潛在表示之間的對比損失，來對齊兩種模態。實驗表明，ContrastVAE 透過充分建模資料的不確定性和稀疏性，並對潛在空間中的擾動具有魯棒性，從而改進了純對比模型。

Diffusion models, explained in Section 2.4, are state-of-the-art models for image generation. While they can also be used for text generation, e.g., by using a discrete latent space with categorical transition probabilities [6], text encoders based on transformers or other sequence-to-sequence models are preferred in practice. As a consequence, multimodal models for both text and images, such as text-to-image generation models, combine text encoders with diffusion models for images. For instance, DALL-E [131] uses the CLIP embedding space as a starting point to generate novel images, and Stable Diffusion [134] uses a UNet autoencoder separately pretrained on a perceptual loss and a patch-based adversarial objective. Several works have built on and expanded diffusion models by increasing controllability of the generated results [187], consistency on the generated subjects identity [136], or for virtual try on [193].

擴散模型，如第 2.4 節所述，是目前最先進的圖像生成模型。雖然它們也可用於文本生成，例如，透過使用具有類別轉移機率的離散潛在空間 [6]，但在實務上，基於 transformer 或其他序列到序列模型的文本編碼器更受青睞。因此，用於文本和圖像的多模態模型，例如文本到圖像生成模型，將文本編碼器與用於圖像的擴散模型相結合。例如，DALL-E [131] 使用 CLIP 嵌入空間作為生成新圖像的起點，而穩定擴散 [134] 則使用在感知損失和基於補丁的對抗目標上單獨預訓練的 UNet 自動編碼器。一些研究透過增加生成結果的可控性 [187]、生成主體身份的一致性 [136] 或用於虛擬試穿 [193] 來建立和擴展擴散模型。

Multimodal LLMs (MLLM) provide a natural language interface for users to express their queries in multiple modalities, or even see responses in different modalities to help visualize the products. Given the complexity of training large generative models end-to-end, researchers typically assemble systems composed of discriminatively pre-trained encoders and decoders, usually connected by adaptation layers to ensure that unimodal representations are aligned. Another approach that involves little or no training is to allow a "controller" LLM to use external foundation models, or tools, to deal with the multimodal input and output [184]. Then, instruction tuning is an important step to make LLMs useful task solvers. Llava [102] is a multimodal LLM that accepts both text and image inputs, and produces useful textual responses. The authors connect a CLIP encoder with an LLM decoder using a simple linear adaptation layer. In [101] the authors change the connection layer from a linear projection to a two-layer MLP and obtain better results. Although MLLM research is still in its inception, some works already start using them in recommendation applications [81].

多模態大型語言模型（MLLM）為使用者提供了一個自然語言介面，讓他們能以多種模態表達查詢，甚至能以不同模態看到回應，以幫助視覺化產品。鑑於端到端訓練大型生成模型的複雜性，研究人員通常會組建由經過判別式預訓練的編碼器和解碼器組成的系統，這些編碼器和解碼器通常由適應層連接，以確保單模態表示的對齊。另一種幾乎不涉及訓練的方法是，允許一個「控制器」LLM 使用外部基礎模型或工具來處理多模態輸入和輸出 [184]。然後，指令微調是使 LLM 成為有用任務解決者的重要步驟。Llava [102] 是一個多模態 LLM，它接受文本和圖像輸入，並產生有用的文本回應。作者使用一個簡單的線性適應層將 CLIP 編碼器與 LLM 解碼器連接起來。在 [101] 中，作者將連接層從線性投影改為兩層 MLP，並獲得了更好的結果。儘管 MLLM 的研究仍處於起步階段，但一些研究已經開始在推薦應用中使用它們 [81]。

## 5 EVALUATING FOR IMPACT AND HARM

Evaluating RS is a complex and multifaceted task that goes beyond simply measuring a few key metrics of a single model. These systems are composed of one or more recommender models and various other ML and non-ML components, making it highly non-trivial to assess and evaluate the performance of an individual model. Moreover, these systems can have far-reaching impacts on users' experiences, opinions, and actions, which may be difficult to quantify or predict, which adds to the challenge. The introduction of Gen-RecSys further complicates the evaluation process due to the lack of well-established benchmarks and the open-ended nature of their tasks. When evaluating RS, it is crucial to distinguish between two main targets of evaluation: the system's performance and capabilities, and its potential for causing safety issues and societal harm. We review these targets, discuss evaluation metrics, and conclude with open challenges and future research directions.

## 5 評估影響與危害

評估推薦系統是一項複雜且多面向的任務，遠不止於測量單一模型的幾個關鍵指標。這些系統由一個或多個推薦模型以及各種其他機器學習和非機器學習組件組成，使得評估單一模型的性能變得非常不簡單。此外，這些系統可能對使用者的體驗、意見和行為產生深遠的影響，而這些影響可能難以量化或預測，這增加了挑戰性。Gen-RecSys 的引入進一步使評估過程複雜化，因為缺乏完善的基準和其任務的開放性。在評估推薦系統時，區分兩個主要的評估目標至關重要：系統的性能和能力，以及其造成安全問題和社會危害的潛力。我們將回顧這些目標，討論評估指標，並以開放的挑戰和未來的研究方向作結。

### 5.1 Evaluating for Offline Impact

The typical approach to evaluating a model involves understanding its accuracy in an offline setting, followed by live experiments.

### 5.1 評估離線影響

評估模型的典型方法包括在離線環境中了解其準確性，然後進行線上實驗。

#### 5.1.1 Accuracy Metrics.

The usual metrics used for discriminative tasks are recall@k, precision@k, NDCG@k, AUC, ROC, RMSE, MAE, etc. Many recent works on generative RS (e.g., [9, 58, 78, 79, 130]) incorporate such metrics for discriminative tasks.

#### 5.1.1 準確度指標。

用於判別性任務的常用指標包括 recall@k、precision@k、NDCG@k、AUC、ROC、RMSE、MAE 等。許多關於生成式推薦系統的近期研究（例如 [9, 58, 78, 79, 130]）都將此類指標納入判別性任務中。

For the generative tasks, we can borrow techniques from NLP. For example, the BLEU score is widely used for machine translation and can be useful for evaluating explanations[45], review generation, and conversational recommendations. The ROUGE score, commonly used for evaluating machine-generated summarization, could be helpful again for explanations or review summarization. Similarly, perplexity is another metric that could be broadly useful, including during the training process to ensure that the model is learning the language modeling component appropriately [108].

對於生成式任務，我們可以借鑒自然語言處理的技術。例如，BLEU 分數廣泛用於機器翻譯，對於評估解釋 [45]、評論生成和對話式推薦也很有用。ROUGE 分數通常用於評估機器生成的摘要，對於解釋或評論摘要也可能有所幫助。同樣，困惑度是另一個可能廣泛有用的指標，包括在訓練過程中確保模型正在適當地學習語言建模組件 [108]。

#### 5.1.2 Computational Efficiency.

Evaluating computational efficiency is crucial for generative recommender models, both for training and inference, owing to their computational burden. This is an upcoming area of research.

#### 5.1.2 計算效率。

評估生成式推薦模型的計算效率至關重要，無論是訓練還是推論，都因其計算負擔而備受關注。這是一個新興的研究領域。

#### 5.1.3 Benchmarks.

Many existing benchmark datasets popular in discriminative recommender models, such as Movielens [53], Amazon Reviews [57], Yelp Challenge[1], Last.fm [139], and Book-Crossing [194], are still useful in generative recommender models, but only narrowly. Some recent ones, like ReDial [96] and INSPIRED [55], are useful datasets for conversational recommendations. [30, 32, 186] propose benchmarks called cFairLLM and FaiR-LLM, to evaluate consumer fairness in LLMs based on the sensitivity of pretrained LLMs to protected attributes in tailoring recommendations. We note that some benchmarks such as BigBench[10] which are commonly used by the LLM community, have recommendations tasks. It will be specifically useful for the RS community to develop new benchmarks for tasks unlocked by Gen-RecSys models.

#### 5.1.3 基準測試。

許多在判別式推薦模型中流行的現有基準資料集，例如 Movielens [53]、Amazon Reviews [57]、Yelp Challenge[1]、Last.fm [139] 和 Book-Crossing [194]，在生成式推薦模型中仍然有用，但範圍有限。一些最近的資料集，如 ReDial [96] 和 INSPIRED [55]，則對對話式推薦很有用。[30, 32, 186] 提出了名為 cFairLLM 和 FaiR-LLM 的基準測試，以評估 LLM 中基於預訓練 LLM 對受保護屬性在客製化推薦中的敏感性來評估消費者公平性。我們注意到，一些 LLM 社群常用的基準測試，例如 BigBench[10]，也包含推薦任務。對於推薦系統社群來說，為 Gen-RecSys 模型解鎖的任務開發新的基準測試將特別有用。

### 5.2 Online and Longitudinal Evaluations

Offline experiments may not capture an accurate picture because of the interdependence of the different models used in the system and other factors. So, A/B experiments help understand the model's performance along several axes in real-world settings. Note that [155] proposes a new paradigm of using simulation using agents to evaluate recommender models. In addition to the short-term impact on engagement/satisfaction, the platform owners will be interested in understanding the long-term impact. This can be measured using business metrics such as revenue and engagement (time spent, conversions). Several metrics could be used to capture the impact on users (daily/monthly active users, user sentiment, safety, harm).

### 5.2 線上與縱向評估

離線實驗可能無法捕捉到準確的全貌，因為系統中使用的不同模型與其他因素之間存在相互依賴性。因此，A/B 實驗有助於在真實世界環境中從多個角度了解模型的性能。值得注意的是，[155] 提出了一種使用代理模擬來評估推薦模型的新範式。除了對參與度/滿意度的短期影響外，平台所有者還會對了解長期影響感興趣。這可以使用商業指標來衡量，例如收入和參與度（花費時間、轉換率）。可以使用多個指標來捕捉對使用者的影響（每日/每月活躍使用者、使用者情緒、安全性、危害）。

### 5.3 Conversational Evaluation

BLEU and perplexity are useful for conversational evaluation but should be supplemented with task-specific metrics (e.g., recall) or objective-specific metrics (e.g., response diversity [88]). Strong LLMs can act as judges, but human evaluation remains the gold standard. Toolkits like CRSLab [76] simplify building and evaluating conversational models, but lack of labeled data in industrial use cases poses a challenge. Some studies use LLM-powered user simulations to generate data.

### 5.3 對話評估

BLEU 和困惑度對於對話評估很有用，但應輔以特定任務的指標（例如，召回率）或特定目標的指標（例如，回應多樣性 [88]）。強大的 LLM 可以充當評審，但人工評估仍然是黃金標準。像 CRSLab [76] 這樣的工具包簡化了對話模型的建構和評估，但在工業應用案例中缺乏標記資料是一個挑戰。一些研究使用由 LLM 驅動的使用者模擬來生成資料。

### 5.4 Evaluating for Societal Impact

Previous work has investigated categories of interest for societal impacts of traditional RS [113] and generative models [14, 167] independently. In the context of RS literature, six categories of harms are found to be associated with RS: content, privacy violations and data misuse, threats to human autonomy and well-being, transparency and accountability, harmful social effects such as filter bubbles, polarisation, manipulability, and fairness. In addition, RS based on generative models can present new challenges [14, 167]:

### 5.4 評估社會影響

先前的研究已分別探討了傳統推薦系統 [113] 和生成模型 [14, 167] 對社會影響的關注類別。在推薦系統文獻的背景下，發現有六類危害與推薦系統相關：內容、隱私侵犯和資料濫用、對人類自主和福祉的威脅、透明度和問責制、有害的社會影響（如過濾氣泡、兩極分化、可操縱性）以及公平性。此外，基於生成模型的推薦系統可能帶來新的挑戰 [14, 167]：

*   LLMs use out-of-domain knowledge, introducing different sources of societal bias that are not easily captured by existing evaluation techniques [30, 31, 141].
*   LLM 使用領域外知識，引入了現有評估技術難以捕捉的不同社會偏見來源 [30, 31, 141]。
*   The significant computational requirements of LLMs lead to heightened environmental impacts [13, 109].
*   LLM 的巨大計算需求導致環境影響加劇 [13, 109]。
*   The automation of content creation and curation may displace human workers in industries such as journalism [28], creative writing, and content moderation, leading to social and economic disruption [4].
*   內容創作和策展的自動化可能會取代新聞 [28]、創意寫作和內容審核等行業的人力工作者，從而導致社會和經濟混亂 [4]。
*   Recommender systems powered by generative models may be susceptible to manipulation and could have unintended and unexpected consequences for users [20, 82].
*   由生成模型驅動的推薦系統可能容易受到操縱，並可能對使用者產生意想不到的後果 [20, 82]。
*   Generative recommendations can expose users to the potential pitfalls of hyper-personalization [41, 133].
*   生成式推薦可能會讓使用者面臨超個人化的潛在陷阱 [41, 133]。

### 5.5 Holistic Evaluations

As mentioned above, thoroughly evaluating RS for offline metrics, online performance, and harm is highly non-trivial. Moreover, different stakeholders (e.g. platform owners and users) [3, 114, 147] may approach evaluation differently. The complexity of Gen-RecSys evaluation presents an opportunity for further research and specialized tools. Drawing inspiration from the HELM benchmark [98], a comprehensive evaluation framework tailored for Gen-RecSys would benefit the community.

### 5.5 整體評估

如上所述，徹底評估推薦系統的離線指標、線上性能和危害是極其不平凡的。此外，不同的利害關係人（例如平台所有者和使用者）[3, 114, 147] 可能會以不同的方式進行評估。Gen-RecSys 評估的複雜性為進一步的研究和專門工具提供了機會。從 HELM 基準 [98] 中汲取靈感，一個為 Gen-RecSys 量身定制的綜合評估框架將有益於社群。

## 6 CONCLUSIONS AND FUTURE DIRECTIONS

While many directions for future work have been highlighted above, the following topics constitute especially important challenges and opportunities for Gen-RecSys:

## 6 結論與未來方向

雖然上面已經強調了許多未來工作的方向，但以下主題構成了 Gen-RecSys 特別重要的挑戰和機遇：

*   RAG (cf. Section 3.3), including: data fusion for multiple (potentially subjective) sources such as reviews [177, 178], end-to-end retriever-generator training [15, 70, 87], and systematic studies of generative reranking alternatives [125].
*   RAG（參見第 3.3 節），包括：多個（可能是主觀的）來源（如評論）的資料融合 [177, 178]，端到端的檢索器-生成器訓練 [15, 70, 87]，以及對生成式重排替代方案的系統性研究 [125]。
*   Tool-augmented LLMs for conversational recommendation, focusing on architecture design for LLM-driven control of dialogue, recommender modules, external reasoners, retrievers, and other tools [18, 40, 112, 162], especially methods for proactive conversational recommendation.
*   用於對話式推薦的工具增強 LLM，專注於由 LLM 驅動的對話控制、推薦模組、外部推理器、檢索器和其他工具的架構設計 [18, 40, 112, 162]，特別是主動式對話推薦的方法。
*   Personalized Content Generation such as virtual try-on experiences [193], which can allow users to visualize themselves wearing recommended clothing or accessories, improving customer satisfaction and reducing returns.
*   個人化內容生成，例如虛擬試穿體驗 [193]，讓使用者可以視覺化自己穿戴推薦的服裝或配飾，從而提高顧客滿意度並減少退貨。
*   Red-teaming – in addition to the standard evaluations, real-world generative RS will have to undergo red-teaming (i.e., adversarial attacks) [34, 142, 164] before deployment to stress test the system for prompt injections, robustness, alignment verification, and other factors.
*   紅隊演練——除了標準評估之外，現實世界的生成式推薦系統在部署前還必須經過紅隊演練（即對抗性攻擊）[34, 142, 164]，以對系統進行壓力測試，檢測提示注入、穩健性、對齊驗證和其他因素。

Despite being a short survey, this work has attempted to provide a foundational understanding of the rich landscape of generative models within recommendation systems. It extends the discussion beyond LLMs to a broad spectrum of generative models, exploring their applications across user-item interactions, textual data, and multimodal contexts. It highlights key evaluation challenges, addressing performance, fairness, privacy, and societal impact, thereby establishing a new benchmark for future research in the domain.

儘管這是一篇簡短的綜述，但本研究試圖為推薦系統中生成模型的豐富景觀提供基礎性的理解。它將討論從 LLM 擴展到廣泛的生成模型，探索其在使用者-項目互動、文本資料和多模態情境中的應用。它強調了關鍵的評估挑戰，涉及性能、公平性、隱私和社會影響，從而為該領域的未來研究建立新的基準。

### REFERENCES

[References have not been translated as they consist of proper nouns, titles, and technical terms.]

### 參考文獻

[參考文獻未經翻譯，因其包含專有名詞、標題和技術術語。]
