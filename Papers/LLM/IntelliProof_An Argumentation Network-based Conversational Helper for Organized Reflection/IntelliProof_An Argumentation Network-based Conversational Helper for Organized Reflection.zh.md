---
title: "IntelliProof_An Argumentation Network-based Conversational Helper for Organized Reflection"
field: "LLM"
status: "Imported"
created_date: 2026-01-12
pdf_link: "[[IntelliProof_An Argumentation Network-based Conversational Helper for Organized Reflection.pdf]]"
tags: [paper, LLM]
---

#LLM
# IntelliProof：一個基於論證網絡的對話輔助工具，用於有組織的反思


Kaveh Eskandari Miandoab¹†*, Katharine Kowalyshyn²†*, Kabir Pamnani³⁵*, Anesu Gavhera⁴†*, Vasanth Sarathy⁵†, Matthias Scheutz⁶†
†Tufts University, ⁵UST,
kaveh.eskandari_miandoab¹, katharine.kowalyshyn², kabir.pamnani³, anesu.gavhera⁴, vasanth.sarathy⁵, matthias.scheutz⁶{@tufts.edu}

## Abstract

## 摘要

We present IntelliProof, an interactive system for analyzing argumentative essays through LLMs. IntelliProof structures an essay as an argumentation graph, where claims are represented as nodes, supporting evidence is attached as node properties, and edges encode supporting or attacking relations. Unlike existing automated essay scoring systems, IntelliProof emphasizes the user experience: each relation is initially classified and scored by an LLM, then visualized for enhanced understanding. The system provides justifications for classifications and produces quantitative measures for essay coherence. It enables rapid exploration of argumentative quality while retaining human oversight. In addition, IntelliProof provides a set of tools for a better understanding of an argumentative essay and its corresponding graph in natural language, bridging the gap between the structural semantics of argumentative essays and the user's understanding of a given text. A live demo and the system are available here to try: https://intelliproof.vercel.app

我們介紹 IntelliProof，一個透過大型語言模型（LLMs）分析論說文的互動系統。IntelliProof 將一篇論文建構成一個論證圖，其中主張被表示為節點，支持性證據作為節點屬性附加，而邊則編碼支持或攻擊的關係。與現有的自動論文評分系統不同，IntelliProof 強調使用者體驗：每個關係最初由一個 LLM 進行分類和評分，然後視覺化以增強理解。該系統為分類提供理由，並產生論文連貫性的量化指標。它能夠在保留人類監督的同時，快速探索論證品質。此外，IntelliProof 提供一套工具，以便用自然語言更好地理解論說文及其對應的圖，彌合了論說文的結構語義與使用者對給定文本的理解之間的差距。此處提供現場演示和系統試用：https://intelliproof.vercel.app

## Introduction & Related Work

## 緒論與相關研究

The rise of Large Language Models (LLMs) has drastically accelerated research in computational argumentation and automated writing support. Argumentative writing is uniquely challenging, requiring a balance of claims, supporting evidence, and counterarguments within a coherent, persuasive structure. Traditional analysis methods, from rule-based systems to neural encoders, frequently struggle to capture the nuanced interrelations between claims and evidence (Elaraby and Litman 2022).

大型語言模型（LLMs）的興起，極大地加速了計算論證和自動寫作支持的研究。論證性寫作尤其具有挑戰性，需要在一個連貫、有說服力的結構中平衡主張、支持性證據和反駁論點。傳統的分析方法，從基於規則的系統到神經編碼器，往往難以捕捉主張與證據之間微妙的相互關係（Elaraby and Litman 2022）。

We introduce IntelliProof, an LLM-powered tool that analyzes arguments by modeling them as graphs (Saveleva et al. 2021). In this model, claims are represented as nodes, with their strength quantified by evidence encoded as node properties. Weighted edges denote support or attack relations between claims. An LLM is used to score, classify, and justify these relations, while allowing human overrides for transparency and control. The dynamic identification and visualization of these relationships are shown in Figure 1.

我們介紹 IntelliProof，一個由 LLM 驅動的工具，它透過將論點建模為圖形來分析論點（Saveleva et al. 2021）。在此模型中，主張被表示為節點，其強度由編碼為節點屬性的證據來量化。加權邊表示主張之間的支持或攻擊關係。LLM 用於對這些關係進行評分、分類和提供理由，同時允許人類為了透明度和控制而覆寫。這些關係的動態識別和視覺化如圖 1 所示。

By transforming essays into structured argumentation graphs, IntelliProof aims to make argumentative reasoning more interpretable, providing writers and educators with insights into essay coherence and persuasiveness. This approach contributes to the discussions on how to integrate LLMs into workflows that demand interpretability, reliability, and pedagogical value simultaneously.

透過將論文轉換為結構化的論證圖，IntelliProof 旨在使論證推理更具可詮釋性，為作者和教育者提供關於論文連貫性和說服力的見解。這種方法有助於探討如何將 LLM 整合到要求可詮釋性、可靠性和教學價值的工作流程中。

LLMs have shifted argument mining methods from encoder-based architectures to prompting and fine-tuning strategies (Cabessa, Hernault, and Mushtaq 2024; Favero et al. 2025). However, annotation bottlenecks and evaluation challenges remain (Schaefer 2025). Recent work also explores interactive systems that combine generative models with human input for constructing argument graphs (Lenz and Bergmann 2025). IntelliProof extends this work by integrating graph-based structuring directly into analysis while grounding scoring of arguments in quantifiable, mathematical metrics.

LLMs 已將論點挖掘方法從基於編碼器的架構轉向提示和微調策略（Cabessa, Hernault, and Mushtaq 2024; Favero et al. 2025）。然而，標註瓶頸和評估挑戰依然存在（Schaefer 2025）。最近的研究也探索了結合生成模型與人類輸入以建構論證圖的互動系統（Lenz and Bergmann 2025）。IntelliProof 透過將基於圖的結構化直接整合到分析中，並將論點評分建立在可量化的數學指標上，從而擴展了這項工作。

Educational applications increasingly use LLMs for essay scoring and feedback (Kim and Jo 2024; Chu et al. 2025). Although many approaches optimize predictive accuracy, few address the interpretability of argumentative quality. Surveys of persuasive applications highlight both the promise and ethical risks of LLM-driven reasoning systems (Rogiers et al. 2024). By grounding essay feedback in explicit argument graphs, IntelliProof contributes to more interpretable educational tools, which will lead to safer AI systems deployed in educational settings.

教育應用越來越多地使用 LLM 進行論文評分和回饋（Kim and Jo 2024; Chu et al. 2025）。儘管許多方法優化了預測準確性，但很少有方法解決論證品質的可詮釋性問題。關於說服性應用的調查突顯了由 LLM 驅動的推理系統的前景和倫理風險（Rogiers et al. 2024）。透過將論文回饋建立在明確的論證圖上，IntelliProof 有助於開發更具可詮釋性的教育工具，這將導致在教育環境中部署更安全的人工智慧系統。

## Intelliproof Overview

## Intelliproof 概覽

Intelliproof's functionality spans argument creation, scoring, classification, and generation techniques. Each of the features elaborated on below is integrated within our GUI front-end.

Intelliproof 的功能涵蓋論點創建、評分、分類和生成技術。以下闡述的每個功能都整合在我們的圖形使用者介面（GUI）前端。

**Graph Visualization** IntelliProof is designed to structurally visualize argumentative essays while providing an LLM-powered (GPT-4o for the instance of the demo given its performance (Shahriar et al. 2024)) toolset for the analysis of the claims. As such, users can input claims, classify them (into Fact, Policy, or Value), and establish connections between the claims via the main GUI of the tool.

**圖形視覺化** IntelliProof 旨在結構化地視覺化論說文，同時提供一個由 LLM 驅動（鑑於其性能，演示實例使用 GPT-4o (Shahriar et al. 2024)）的工具集，用於分析主張。因此，使用者可以輸入主張，將其分類（為事實、政策或價值），並透過工具的主要圖形使用者介面在主張之間建立聯繫。

**LLM Document Analysis** To establish claims, users upload supporting documents as evidence in PDF or image format. A dedicated LLM instance then processes these files, suggesting relevant text or image extracts for a specific claim. The user attaches the suggested evidence to the claim via a drag-and-drop interface, which in turn prompts the LLM to assess the claim's strength by analyzing all attached evidence. Any number of supporting or negating evidence pieces can be associated with a single claim.

**LLM 文件分析** 為了建立主張，使用者以 PDF 或圖片格式上傳支持文件作為證據。一個專用的 LLM 實例會處理這些文件，為特定主張建議相關的文本或圖像摘錄。使用者透過拖放介面將建議的證據附加到主張上，這會提示 LLM 透過分析所有附加的證據來評估主張的強度。任何數量的支持或反駁證據都可以與單一主張相關聯。

[Image]

**Figure 1: IntelliProof user interface overview using an example graph on the effect of green space on urban environments.**

**圖 1：IntelliProof 使用者介面概覽，使用一個關於綠地對都市環境影響的範例圖。**

**Claim Credibility Score** To assess overall claim strength, we combine evidence and edge scores to obtain the claim credibility score St where St = tanh(λΣfe(ei) + ΣfED(kj) * St-1). fe and fED are calculated based on the LLM's assessment of claim support based on an evidence, and based on an incoming edge, respectively, and λ is a tunable hyperparameter. Note that given the weakness of LLMs in directly generating scores (Schroeder and Wood-Doughty 2025; Cui 2025), we first generate a qualitative classification as the LLM's assessment, and then utilize the Evans coefficient interpretation (Evans 1996) to convert the qualitative assessment to numerical scores.

**主張可信度分數** 為了評估整體主張強度，我們結合證據和邊的分數來獲得主張可信度分數 St，其中 St = tanh(λΣfe(ei) + ΣfED(kj) * St-1)。fe 和 fED 分別根據 LLM 對基於證據的主張支持度的評估以及傳入邊的評估計算得出，而 λ 是一個可調整的超參數。請注意，鑑於 LLM 在直接生成分數方面的弱點（Schroeder and Wood-Doughty 2025; Cui 2025），我們首先生成一個定性分類作為 LLM 的評估，然後利用 Evans 係數解釋（Evans 1996）將定性評估轉換為數值分數。

**Report Generation** Another feature of Intelliproof is automatic report generation from the graph implementation of an argument. These reports combine evidence evaluation, edge validation, assumptions analysis, and graph critique into a singular unified report. Our system processes graph structure, evidence quality, relationship strengths, and logical patterns simultaneously and creates an eight section, comprehensive report of the argumentative essay.

**報告生成** Intelliproof 的另一個功能是從論證的圖形實現中自動生成報告。這些報告將證據評估、邊驗證、假設分析和圖形批判結合到一個單一的統一報告中。我們的系統同時處理圖形結構、證據品質、關係強度和邏輯模式，並創建一個包含八個部分的綜合性論說文報告。

**AI Copilot Chat Interface** Using our integrated chatbot, natural language queries are parsed, and one may ask questions about the graph. The responding AI is context aware, and users can get insights on arguments' strengths, weaknesses, and gaps to fill. As arguments are built, the LLM context window is also updated in real-time to contain the new information.

**AI 協作聊天介面** 使用我們整合的聊天機器人，可以解析自然語言查詢，使用者可以對圖形提出問題。回應的 AI 具有上下文感知能力，使用者可以獲得關於論點的優點、缺點和待補足之處的見解。隨著論點的建立，LLM 的上下文視窗也會即時更新以包含新資訊。

**Assumption Generation** Intelliproof analyzes claim relationships to identify three implicit assumptions that would strengthen support between claims. It also finds hidden premises and bridges assumptions needed to make arguments more robust. Each assumption includes an importance rating and a justification for why it strengthens the relationship generated based on a few-shot learning approach (Brown et al. 2020) prepared by an argumentation field expert.

**假設生成** Intelliproof 分析主張之間的關係，以識別三個能夠加強主張間支持的隱含假設。它還能找出隱藏的前提和彌合假設，使論點更加穩健。每個假設都包含一個重要性評級和一個理由，說明為何它能加強基於由論證領域專家準備的少樣本學習方法（Brown et al. 2020）所生成的關係。

**Critique Graph** To identify essay weaknesses, we deploy a state-of-the-art LLM (GPT-4o) to match the overall argument against our comprehensive Argument Patterns Bank. This Bank is a built-in YAML database, developed by an argumentation expert, containing patterns for logical fallacies, good arguments, and absurd reasoning. This process allows us to specifically identify issues like circular reasoning, straw man arguments, and false causes.

**批判圖** 為了識別論文的弱點，我們部署了一個最先進的 LLM (GPT-4o)，將整體論點與我們全面的「論證模式庫」進行比對。該庫是一個由論證專家開發的內建 YAML 資料庫，包含邏輯謬誤、良好論點和荒謬推理的模式。這個過程使我們能夠具體識別出循環論證、稻草人論證和錯誤歸因等問題。

## System Implementation

## 系統實現

IntelliProof's architecture consists of three core components. The **frontend** is built with Vite and React.js to create a dynamic user interface that handles all back-end API and database calls. The **backend** uses a Python server with FastAPI for handling requests and SupaBase (PostgreSQL) for managing user data such as profiles, evidence files, and graphs. For the large language model, we utilize GPT-4o via OpenAI's Python library, chosen for its balance of performance, cost, and availability. The design is modular, allowing GPT-4o to be easily substituted with other locally or remotely deployed LLMs.

IntelliProof 的架構由三個核心組件組成。**前端**使用 Vite 和 React.js 建立，以創建一個動態的使用者介面，處理所有後端 API 和資料庫呼叫。**後端**使用一個帶有 FastAPI 的 Python 伺服器來處理請求，並使用 SupaBase (PostgreSQL) 來管理使用者資料，如個人資料、證據文件和圖形。對於大型語言模型，我們透過 OpenAI 的 Python 函式庫使用 GPT-4o，選擇它的原因是其在性能、成本和可用性之間的平衡。該設計是模組化的，允許 GPT-4o 輕鬆地被其他本地或遠端部署的 LLM 取代。

## Conclusion

## 結論

Intelliproof is an interactive LLM platform that creates argument graphs based on provided evidence. This system is designed to be helpful in devising strong arguments, filling gaps in arguments, and utilizing an LLM to provide a detailed look at an argumentative essay. While this system may be expanded further in the future, at present, we provide a robust, functional system that demonstrates the feasibility of Intelliproof as a powerful tool for structured, LLM-driven argumentation.

Intelliproof 是一個互動式 LLM 平台，它根據提供的證據創建論證圖。該系統旨在幫助設計強有力的論點、填補論點中的空白，並利用 LLM 對論說文進行詳細的審視。雖然該系統未來可能會進一步擴展，但目前我們提供了一個穩健、功能齊全的系統，展示了 Intelliproof 作為一個用於結構化、由 LLM 驅動的論證的強大工具的可行性。

We publicly release the source code for IntelliProof at https://github.com/collective-intelligence-lab/intelliproof

我們在 https://github.com/collective-intelligence-lab/intelliproof 公開釋出 IntelliProof 的原始碼。

## Acknowledgments

## 致謝

This research was supported in part by Other Transaction award HR00112490378 from the U.S. Defense Advanced Research Projects Agency (DARPA) Friction for Accountability in Conversational Transactions (FACT) program.

本研究部分由美國國防高等研究計劃署（DARPA）的「對話交易問責摩擦」（FACT）計畫的其他交易獎項 HR00112490378 支持。

## References

## 參考文獻

Brown, T. B.; Mann, B.; Ryder, N.; Subbiah, M.; Kaplan, J.; Dhariwal, P.; Neelakantan, A.; Shyam, P.; Sastry, G.; Askell, A.; Agarwal, S.; Herbert-Voss, A.; Krueger, G.; Henighan, T.; Child, R.; Ramesh, A.; Ziegler, D. M.; Wu, J.; Winter, C.; Hesse, C.; Chen, M.; Sigler, E.; Litwin, M.; Gray, S.; Chess, B.; Clark, J.; Berner, C.; McCandlish, S.; Radford, A.; Sutskever, I.; and Amodei, D. 2020. Language Models are Few-Shot Learners. arXiv:2005.14165.

Cabessa, J.; Hernault, H.; and Mushtaq, U. 2024. Argument Mining in BioMedicine: Zero-Shot, In-Context Learning and Fine-tuning with LLMs. In Proceedings of the 10th Italian Conference on Computational Linguistics (CLiC-it 2024), 122–131.

Chu, S.; Kim, J. W.; Wong, B.; and Yi, M. Y. 2025. Rationale Behind Essay Scores: Enhancing S-LLM’s Multi-Trait Essay Scoring with Rationale Generated by LLMs. In Chiruzzo, L.; Ritter, A.; and Wang, L., eds., Findings of the Association for Computational Linguistics: NAACL 2025, 5796–5814. Albuquerque, New Mexico: Association for Computational Linguistics. ISBN 979-8-89176-195-7.

Cui, H. 2025. LLMs Are Not Scorers: Rethinking MT Evaluation with Generation-Based Methods. arXiv:2505.16129.

Elaraby, M.; and Litman, D. 2022. ArgLegalSumm: Improving Abstractive Summarization of Legal Documents with Argument Mining. In Calzolari, N.; Huang, C.-R.; Kim, H.; Pustejovsky, J.; Wanner, L.; Choi, K.-S.; Ryu, P.-M.; Chen, H.-H.; Donatelli, L.; Ji, H.; Kurohashi, S.; Paggio, P.; Xue, N.; Kim, S.; Hahm, Y.; He, Z.; Lee, T. K.; Santus, E.; Bond, F.; and Na, S.-H., eds., Proceedings of the 29th International Conference on Computational Linguistics, 6187–6194. Gyeongju, Republic of Korea: International Committee on Computational Linguistics.

Evans, J. D. 1996. Straightforward statistics for the behavioral sciences. Thomson Brooks/Cole Publishing Co.

Favero, L.; Pérez-Ortiz, J.; Käser, T.; and Oliver, N. 2025. Leveraging Small LLMs for Argument Mining in Education: Argument Component Identification, Classification, and Assessment.

Kim, S.; and Jo, M. 2024. Is GPT-4 Alone Sufficient for Automated Essay Scoring?: A Comparative Judgment Approach Based on Rater Cognition. In Proceedings of the Eleventh ACM Conference on Learning @ Scale, 315–319. ArXiv:2407.05733 [cs].

Lenz, M.; and Bergmann, R. 2025. ArgueMapper Assistant: Interactive Argument Mining Using Generative Language Models, volume 15446 of Lecture Notes in Computer Science, 189–203. Cham: Springer Nature Switzerland. ISBN 978-3-031-77914-5.

Rogiers, A.; Noels, S.; Buyl, M.; and Bie, T. D. 2024. Persuasion with Large Language Models: a Survey. (arXiv:2411.06837). ArXiv:2411.06837 [cs].

Saveleva, E.; Petukhova, V.; Mosbach, M.; and Klakow, D. 2021. Graph-based Argument Quality Assessment. In Mitkov, R.; and Angelova, G., eds., Proceedings of the International Conference on Recent Advances in Natural Language Processing (RANLP 2021), 1268–1280. Held Online: INCOMA Ltd.

Schaefer, R. 2025. On Integrating LLMs Into an Argument Annotation Workflow. In Chistova, E.; Cimiano, P.; Haddadan, S.; Lapesa, G.; and Ruiz-Dolz, R., eds., Proceedings of the 12th Argument mining Workshop, 87–99. Vienna, Austria: Association for Computational Linguistics. ISBN 979-8-89176-258-9.

Schroeder, K.; and Wood-Doughty, Z. 2025. Can You Trust LLM Judgments? Reliability of LLM-as-a-Judge. arXiv:2412.12509.

Shahriar, S.; Lund, B.; Mannuru, N. R.; Arshad, M. A.; Hayawi, K.; Bevara, R. V. K.; Mannuru, A.; and Batool, L. 2024. Putting GPT-4o to the Sword: A Comprehensive Evaluation of Language, Vision, Speech, and Multimodal Proficiency. arXiv:2407.09519.
