# Kaj Soellaart

Interdisciplinary researcher working across quantitative methods, machine learning, and philosophy of AI. Currently completing dual master's degrees in Econometrics (Data Science, Vrije Universiteit Amsterdam) and Philosophy (Philosophy of AI, University of Amsterdam).

This repository collects my thesis work, selected writing on AI and society, and code.

---

## Theses

### Warranted Trust in LLM Chain-of-Thought Explanations
**MA Philosophy thesis (in progress) | University of Amsterdam, 2025**

Large language models increasingly use chain-of-thought (CoT) prompting to show their reasoning. But when can a user actually rely on that reasoning as an explanation, rather than just a plausible-sounding trace? This thesis asks under what conditions end users can warrantedly trust an LLM's chain-of-thought as an explanation of its answer, where warranted trust means reliance supported by appropriate epistemic grounds rather than confidence alone. It develops a set of stable distinctions that connect what the user needs from an explanation to empirical questions about CoT faithfulness, shifting the evaluation from *whether CoT helps* to *whether it can justify reliance in a given setting*. The methodology combines conceptual analysis of warranted trust and user-oriented explanation with a critical assessment of the empirical literature on CoT faithfulness.

*Thesis in progress. Full text will be added upon completion.*

`theses/ma-warranted-trust-cot/`

---

### Quantifying Uncertainty in Latent Dirichlet Allocation
**MSc Econometrics thesis | Vrije Universiteit Amsterdam, 2025 | Being developed into a journal publication**

LDA is widely used to discover themes in large text collections, but its outputs carry statistical uncertainty that is rarely reported. This thesis separates three sources of that uncertainty: optimization uncertainty (different random seeds), sampling uncertainty (different draws from the population), and preprocessing choices (vocabulary decisions made before inference). It introduces a token-level bootstrap that resamples tokens within each document before fitting the model, avoiding assumptions about the data-generating process. Using controlled simulations and an empirical application to Dutch parliamentary speeches, the study shows that optimization uncertainty generally exceeds sampling uncertainty, that sampling uncertainty is nonetheless substantive enough to change results and their interpretation, and that preprocessing choices affect both the calibration of parameter estimates and the magnitude of reported uncertainty.

`theses/msc-uncertainty-lda/`

---

## Writing

### The Hidden Operations of AI: Data Annotation Labor, Extractivism, and the Politics of Invisibility
**Essay, 2026**

An examination of the structural role of low-wage human labor in AI systems, focusing on the case of OpenAI's outsourcing of content moderation to Kenyan workers. The essay draws on Kate Crawford's materialist analysis of AI, Bruno Latour's concept of the Modern Constitution, and theories of extractivism to argue that the concealment of data annotation labor is not incidental but constitutive of how AI is produced, marketed, and understood. It traces how corporate non-disclosure, the framing of AI as autonomous intelligence, and the inherited philosophical separation between technology and politics work together to sustain a mismatch between AI's public image and its material reality.

`writing/hidden-operations-of-ai/`

---

### Wanneer een AI-chatbot een 'Jij' wordt
**Blog post (Dutch), public philosophy**

A public-facing piece on AI companionship and its risks, prompted by the case of a teenager who took his own life after forming an intimate relationship with a Character.ai chatbot. The post uses Martin Buber's distinction between *Ik-Het* (I-It) and *Ik-Jij* (I-You) relationships to examine what happens when chatbots become convincing enough that users, especially young people, experience them as genuine relational partners. It argues that the absence of real reciprocity in AI interactions has consequences that current regulation does not adequately address.

`writing/wanneer-ai-chatbot-jij-wordt/`

---

## Code

### Hotel Search Ranking (LightGBM LambdaMART)
**Data Mining course, VU Amsterdam | Result: top 5 out of ~400 students (2025)**

Given a dataset of hotel search sessions from Expedia, the task was to predict the ranking of properties within each session to maximise NDCG@5. Each row represents one (search, property) impression; the targets are click and booking indicators. The model is a LightGBM LambdaMART ranker with a graded relevance target (booking = 5, click = 1, else 0). Feature engineering covers historical property-level aggregations, price normalisation, within-session ranking, star/review metrics, and competitor pricing comparisons (~140 features total). Hyperparameters were tuned via Bayesian search (Hyperopt TPE, 500 trials) and evaluated with 5-fold group cross-validation stratified by search ID.

`projects/ml-competition-model/`

---

## Earlier Thesis Work

### Unravelling Complexity: Linking Deep Neural Networks to Phenomena
**BA Philosophy thesis | 2023**

Can we use models to understand phenomena if we don't understand the models themselves? This thesis engages with Emily Sullivan's argument for "link uncertainty," which holds that the opacity of deep neural networks does not necessarily prevent them from providing scientific understanding. The thesis analyzes Sullivan's argument from both a technical and philosophical perspective, examining how concepts like opacity, explanation, and understanding apply to DNNs. It identifies limitations in Sullivan's use of simplified examples and proposes improvements to the concept of link uncertainty, including a revised definition of opacity and a more precise account of how models can be linked to the phenomena they represent.

`theses/ba-dnns-scientific-explanation/`

---

### Segmentation-Based Heatmaps for Post-Hoc Explainability in Computer Vision
**BSc thesis | 2023**

How can we explain what a convolutional neural network "sees" when it classifies an image? This thesis proposes segmentation-based heatmaps as a post-hoc explainability method. It introduces two architectures: Segmentation-based CNNs (SCNNs) and Frozen Segmentation-based CNNs (FSCNNs), which integrate a segmentation model with a CNN to produce heatmaps that highlight which input regions drive the model's decisions. The FSCNN achieves interpretability with only a small accuracy trade-off relative to the original CNN, while being roughly 400 times faster than standard pixel-level perturbation methods. The thesis also examines the epistemic limits of the approach, noting that using one black box (the segmenter) to explain another raises its own transparency questions.

`theses/bsc-computer-vision/`

---

## Contact

soellaart.kaj@gmail.com | Amsterdam, NL
