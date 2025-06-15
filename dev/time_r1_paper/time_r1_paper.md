# Time Series Forecasting as Reasoning:  A Slow-Thinking Approach with Reinforced LLMs

**Authors**: Yucong Luo, Yitong Zhou, Mingyue Cheng*, Jiahao Wang, Daoyu Wang, Tingyue Pan, Jintao Zhang

[Mermaid Diagram: Evolution of TSF Methods](figure1.mmd)

[Mermaid Diagram: Time-R1 Framework](figure2.mmd)

<!-- Source: sections/0_abstract.tex -->

**Abstract**

To advance time series forecasting (TSF), various methods have been proposed to improve prediction accuracy, evolving from statistical techniques to data-driven deep learning architectures.
Despite their effectiveness, most existing methods still adhere to a \textit{fast thinking} paradigm-relying on extracting historical patterns and mapping them to future values as their core modeling philosophy, lacking an explicit thinking process that incorporates intermediate time series reasoning. Meanwhile, emerging slow-thinking LLMs (e.g., OpenAI-o1) have shown remarkable multi-step reasoning capabilities, offering an alternative way to overcome these issues. However, prompt engineering alone presents several limitations—including high computational cost, privacy risks, and limited capacity for in-depth domain-specific time series reasoning. To address these limitations, a more promising approach is to train LLMs to develop \textit{slow thinking} capabilities and acquire strong time series reasoning skills.
For this purpose, we propose \Ours, a two-stage reinforcement fine-tuning framework designed to enhance multi-step reasoning ability of LLMs for time series forecasting. Specifically, the first stage conducts supervised fine-tuning for warmup adaptation, while the second stage employs reinforcement learning to improve the model's generalization ability. Particularly, we design a fine-grained multi-objective reward specifically for time series forecasting, and then introduce GRIP (group-based relative importance for policy optimization), which leverages non-uniform sampling to further encourage and optimize the model's exploration of effective reasoning paths.
Experiments demonstrate that \Ours significantly improves forecast performance across diverse datasets
\footnote{The corresponding author is Mingyue Cheng. The code is at  \url{https://github.com/lqzxt/Time-R1}.}.

<!-- Source: sections/1_introduction.tex -->

## Introduction

Time series forecasting (TSF)~ plays a key role in data-driven decision-making across critical domains, including financial market analysis, energy demand planning, and traffic flow management. Over the years, numerous research efforts have been proposed to advance this field. Classical statistical methods such as ARIMA , ETS , and Theta  have long been used to predict future data by leveraging the statistical properties of single samples. Machine learning methods such as XGBoost and LightGBM remain highly robust due to their interpretability and ability to model nonlinear relationships. With the arrival of computing power era, deep learning-based approaches  have since gained prominence due to their ability to capture complex temporal patterns and adapt to non-stationary real-world data. Methodological analysis covers pioneering architectures such as sequence dependency modeling of RNNs , TCNs  and transformer-based models , which improve generalization through shared representations across multiple time series.

\input{images/7_comparison}

Although the specific techniques vary, most existing TSF methods follow a similar "fast thinking" paradigm~. Specifically focusing on single-step prediction accuracy~, these methods typically employ sequential models to encode historical values and use one-step decoding to directly map past observations to future values~. Although effective in benchmarks, their underlying logic is largely based on pattern recognition~ and trend prediction, lacking an explicit reasoning process. However, in real-world scenarios, time series often reflect more complex temporal logic, which should not merely be 'fitted'—they should be understood and reasoned.

To address this issue, a growing body of research  has explored leveraging the reasoning capabilities of large language models (LLMs) to analyze temporal dynamics and generate high-quality representations, thereby enhancing lightweight TSF models . These approaches benefit from the ability of LLMs to incorporate contextual information such as textual metadata , offering stronger generalization performance across diverse domains , and produce interpretable explanations that support forecasting decisions .
However, despite their potential, current LLM-based TSF methods face three key limitations:
\textit{First}, a partial misalignment of time series domain knowledge, and limited reasoning capabilities. General linguistic knowledge in LLM often mismatches the temporal patterns and causal mechanisms required for time series tasks, leading to suboptimal performance .
\textit{Second}, a lack of generalization from experiential learning. While effective in supervised memorization, they struggle with understanding dynamics or adapting to new, unseen scenarios, which limits their out-of-distribution performance.
\textit{Third}, absence of progressive reasoning. These models map history to future directly without detecting regime changes or performing step-by-step inference, resembling fast (not deliberate) thinking for time series.
These issues lead to a central question: \textbf{Can we improve time series forecasting performance by training LLMs to acquire time series reasoning capabilities?} 

Motivated by the above question, we propose \Ours, a novel LLM-based time series forecasting framework that trains large language models to acquire slow-thinking reasoning capabilities for forecasting tasks. 
At its core, \Ours leverages LLMs as the time series reasoning backbone and introduces a two-stage reinforcement fine-tuning (RFT) optimization framework:
\textit{First}, we begin with warm-up supervised fine-tuning. The model is fine-tuned for memorization, learning both effective reasoning patterns and accurate output formatting using synthetic reasoning trajectories that demonstrate step-by-step temporal analysis. \textit{Second}, the model is refined through reinforcement learning for generalization, using fine-grained, multi-objective rewards specifically designed for forecasting tasks, improving temporal coherence and multi-horizon accuracy. Notably, we propose GRIP (Group-based Relative Importance for Policy Optimization), which optimizes LLM reasoning paths in TSF through a uniform sampling strategy and adaptive weighting. Extensive experiments are conducted on real-world datasets, showing that \Ours effectively enhances forecast performance through the slow thinking paradigm. Our main contributions are as follows:

  \setlength{- sep}{1pt} % 减少项目符号之间的垂直间距
  \setlength{\parskip}{2pt} % 减少段落之间的垂直间距
  -  We introduce time series reasoning by training LLMs to adopt a slow-thinking paradigm that generates reasoning processes supporting final forecasting.
  -  We design a two-stage RFT framework (SFT for memorization, and RL for generalization) that enhances the reasoning ability of LLMs. We introduce a fine-grained, multi-objective reward specifically for TSF, along with a novel sampling strategy for RL optimization.
  -  Extensive experiments demonstrate the effectiveness of \Ours, showing it enhances LLM reasoning and improves generalization and explainability via deliberate slow thinking.

<!-- Source: sections/2_preliminaries_and_motivation.tex -->

## Preliminaries

### Problem Definition

Let $\mathbb{D} = \{(X^i,y^i)\}_{i=1}^n$ be a temporal dataset, where each $X^i \in \mathbb{R}^{t \times m}$ is a multivariate time series with $t$ steps and $m$ channels, and $y^i \in \mathbb{R}^{h \times d}$ contains $d$-dimensional targets over $h$ future steps. The forecasting task learns a mapping $f_\theta: \mathbb{R}^{t \times m} \rightarrow \mathbb{R}^{h \times d}$ capturing temporal dependencies in $\mathbb{D}$. Under \Ours, the forecasting procedure using prompt template $P$ is: $T^i = \text{LLM}_\phi(P, X^i)$, where $T^i$ is the LLM's textual output, and $\hat{y}^i = g(T^i)$ parses it into numerical predictions $\hat{y}^i \in \mathbb{R}^{h \times d}$.

### Time Series Reasoning

Although time series forecasting methods have advanced rapidly, most existing approaches rely on pattern recognition and lack genuine \textit{time series reasoning} — the ability to dynamically infer future values by integrating observed historical data with structured logic or domain knowledge. Traditional statistical  and classical machine learning models  are inherently limited in capturing high-level temporal semantics, while deep learning models , despite their expressive power, often function as "black boxes" without offering clear reasoning paths. Recently, LLM-based methods  have attempted to address this gap by incorporating language-like interpretability and contextual understanding. However, these models are typically adapted from general-purpose LLMs not specifically designed for temporal reasoning, and they heavily depend on prompt engineering rather than training on forecasting tasks, limiting their capacity for deliberate, step-by-step temporal inference.

With the emergence of reasoning-oriented LLMs such as OpenAI-o1/o3  and DeepSeek-R1 , there is a promising path toward more structured and deliberative forecasting through mechanisms akin to "slow thinking". Nevertheless, these models are computationally intensive, not tailored for temporal tasks, and often unsuitable for privacy-sensitive domains like healthcare. Therefore, prompt engineering alone is insufficient to unlock the full potential of LLMs in TSF. To overcome these limitations, there is a pressing need for a dedicated framework that explicitly trains LLMs on structured forecasting tasks, endowing them with domain-specific slow-thinking capabilities and robust time series reasoning skills.

<!-- Source: sections/3_method.tex -->

## The Proposed \Ours

\input{images/1_framework}

### \Ours Overview

\Ours consists of a two-stage RFT framework for LLM-based time series forecasting, built upon a structured training template that standardizes input representations and encodes task-specific knowledge. In the first stage, we perform warm-up SFT using synthetic chain-of-thought trajectories to teach the model effective temporal analysis and accurate output formatting. These trajectories are generated under strict guidelines and refined iteratively to align with ground-truth forecasts. The second stage further improves the model via RL, guided by a fine-grained, multi-objective reward function tailored for time series forecasting. To optimize reasoning paths during RL, we introduce GRIP (Group-based Relative Importance for Policy Optimization), a novel strategy that leverages non-uniform sampling and adaptive weighting to balance accuracy, logical consistency, and temporal coherence. An overview of the framework is provided in Figure .

### Training Template

Our training template employs an instructional framework that standardizes inputs while encoding task-specific knowledge through five components: (1) \textit{Task Definition} establishing objectives and problem scope; (2) \textit{Dataset Description} specifying temporal characteristics and application scenarios; (3) \textit{Channel Information} delineating input signal types; (4) \textit{Testing Data} providing timestamps and historical series; and (5) \textit{Format Instruction} defining output templates. This design interleaves domain knowledge with structural constraints, reducing inference-time formatting ambiguities (see Table ).

\input{table/3_prompt_template}

### Supervised Fine-tuning for Warmup Adaptation

To mitigate the linguistic readability degradation and slow convergence caused by direct reinforcement learning on LLMs—particularly due to formatting inconsistencies—we first perform a warmup stage via SFT. This warmup SFT step is designed to stabilize training, ensure proper output formatting, and equip the model with basic reasoning capabilities, without requiring deep time series understanding.

Our SFT data construction involves three key steps. First, we leverage DeepSeek-R1 to generate time-series predictions on the training set by feeding it historical time series data paired with strict formatting guidelines. We then select the optimal prediction for each sample based on the Mean Absolute Percentage Error metric . Next, to derive a reasoning process aligned with ground-truth labels, we inject both the true prediction value and the high-quality CoT generated in the previous step into DeepSeek-R1 as prompts, guiding it to synthesize a revised CoT that logically culminates in the correct prediction. Finally, we concatenate the refined CoT and the true prediction value, demarcating the final answer using `<answer>` tags to create structured training data for SFT. The full data collection procedure is detailed in Algorithm  in the Appendix.

After constructing the training data, we perform a single-epoch fine-tuning with a small learning rate. This warm-up SFT phase effectively prepares the model for subsequent reinforcement learning, ensuring stable learning dynamics and accurate output formatting. It also enables the model to internalize reasoning patterns from synthetic trajectories, laying the foundation for more deliberate and coherent decision-making in later RL stages.

### Reinforcement Learning for Effective Reasoning Patterns

After warmup SFT, we further fine-tune the LLM using RL to generalize its reasoning and acquire slow-thinking capabilities for time series forecasting. In the following sections, we first present the reward design, and then describe the employed reinforcement learning algorithm GRIP.

#### Reward Design

To effectively apply RL for optimizing the proposed slow-thinking time series forecasting, we introduce several fine-grained and multi-objective reward functions specifically designed to enhance forecasting performance and slow thinking behavior. 

\paragraph{Format Rewards.}  
To ensure syntactic validity and completeness of the generated reasoning paths, we define two reward components to enforce both structural integrity and output completeness:
\textit{Format Reward:} A binary penalty is imposed if the output does not follow the required structured format (e.g., missing or malformed \texttt{<answer>} tags):

\begin{equation}
\gamma_{\text{Format}} = 
\begin{cases}
0 & \text{if valid } \texttt{<think>}, \texttt{<answer>} \text{ tags and proper structure} \\
-1 & \text{otherwise}
\end{cases}
\end{equation}
\textit{Length Reward:} To encourage full time point sequence generation and accelerate convergence, we provide positive feedback based on how close the generated sequence length is to the ground truth:
\begin{equation}
\gamma_{\text{Length}} = 
\begin{cases}
0.1 & \text{if } \text{len(answer)} \geq \text{len(ground\_truth)} \\
0.1 \cdot \dfrac{\text{len(answer)}}{\text{len(ground\_truth)}} & \text{otherwise}
\end{cases}
\end{equation}

\paragraph{Accuracy Rewards.}
We define two accuracy-based reward components to assess numerical precision and temporal fidelity, encouraging accurate value prediction and modeling of dynamics:

\textit{MSE Reward:} We compute the mean squared error between normalized prediction and target sequences and map it into a bounded reward signal using a sigmoid transformation:
\begin{equation}
\gamma_{\text{MSE}} = \left(1 - \frac{1}{1 + e^{-0.3 \cdot \text{MSE}}}\right) \cdot 2,
\end{equation}
\textit{Seasonal-Trend Decomposition Reward:} Both predicted and true sequences are decomposed into seasonal ($s$) and trend ($t$) components via moving average-based methods, where $s_i$ the seasonal component and $t_i$ the long-trend component. We then compute separate MSE terms for each:
\begin{equation}
\gamma_{\text{Seasonal}} = \frac{1}{n} \sum_{i=1}^{n} \left(s_i^{\text{true}} - s_i^{\text{pred}}\right)^2,
\gamma_{\text{Trend}}     = \frac{1}{n} \sum_{i=1}^{n} \left(t_i^{\text{true}} - t_i^{\text{pred}}\right)^2,
\end{equation}

\paragraph{Structural Similarity Reward.}
We evaluate structural similarity by matching predicted and ground-truth extrema within tolerance windows, ensuring change-point capture and interpretable patterns, with correct matches receiving credit:
\begin{equation}
\gamma_{\text{CP}} = \left(\frac{N_{\text{cmax}}}{N_{\text{gmax}}} \cdot 0.2 \right) + \left(\frac{N_{\text{cmin}}}{N_{\text{gmin}}} \cdot 0.2\right),
\end{equation}
where $N_{\text{cmax}}$ and $N_{\text{cmin}}$ respectively represent the counts of correctly identified local maxima and minima within a tolerance window, $N_{\text{gmax}}$ and $N_{\text{gmin}}$ are the total ground-truth extrema counts.

The overall reward for RL training is the sum of the above rewards.

#### Reinforcement Learning Algorithm: GRIP

We introduce GRIP (Group-based Relative Importance for Policy Optimization), a general RL optimization method designed to optimize entire trajectories for LLM time series forecasting reasoners. The GRIP objective function, formalized in Equation , combines a non-uniform sampling strategy with adaptive trajectory weighting within a policy gradient framework. In the following sections, we elaborate on its core components:(1) GRIP formalization; (2) non-uniform sampling strategy to balance exploration and exploitation; and (3) an adaptive weighting scheme that enhances gradient signals from high-quality reasoning paths. 
\input{images/2_grip}
\paragraph{Formalization of the GRIP Objective.}
The GRIP objective integrates the two key design components into a unified policy gradient framework, as formalized in Equation :
\begin{align}
\mathcal{J}_{\text{GRIP}}(\theta) = 
&\mathbb{E}_{\substack{q \sim P(Q), \\ \{o_j\}_{j=1}^{k \cdot G} \sim \pi_{\theta_{\text{old}}}(o|q), \\ \{o_i\}_{i=1}^G \sim \text{Sample}\left(\{o_j\}_{j=1}^{k \cdot G}; R(o_j)\right)}} 
\Bigg\{ 
\sum_{i=1}^{G} w_i^U  
\frac{1}{|o_i|} 
\bigg\{
\sum_{t=1}^{|o_i|} 
\min\bigg[ 
\frac{\pi_\theta(o_{i,t}|q, o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t}|q, o_{i,<t})} A_i,\nonumber\\
&
\text{clip}\Big( \frac{\pi_\theta(o_{i,t}|q, o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t}|q, o_{i,<t})}, 1-\epsilon, 1+\epsilon \Big) A_i 
\bigg]
- \beta \mathbb{D}_{KL} [\pi_\theta || \pi_{ref}] 
\bigg\} 
\Bigg\},

\end{align}  
where, $\epsilon$ and $\beta$ are hyperparameters. $\pi_{\text{ref}}$ is the reference model, typically initialized as the pre-trained model before reinforcement learning begins. The output $\{o_i\}$ is selected through a sampling process from policy $\pi_{\theta_{\text{old}}}$. The hyperparameter $k$ controls the size of the rollout space, while $G$ referred to as the group size. $\mathbb{D}_{KL}$ represents the KL divergence, which is incorporated into the loss function as a regularization term during training. And $A_i$ is the advantage computed using a group of rewards $\{r_1, r_2, \dots, r_G\}$ corresponding to the completion trajectories within each group:
\begin{equation}
    A_i = \frac{r_i - \text{mean}(\{r_1, r_2, \dots, r_G\})}{\text{std}(\{r_1, r_2, \dots, r_G\})},
\end{equation}
The weight $w_i^U$ denotes the adaptive weighting assigned to each trajectory. This objective balances exploration and exploitation while mitigating gradient dilution. The sampling strategy and adaptive weight will be discussed in the following section.

\paragraph{Non-uniform Sampling Strategy.}

To bridge the gap between reasoning and forecasting in time series modeling, recent RL methods like GRPO have shown promise. However, they often suffer from an exploration-exploitation imbalance.
To address this, GRIP introduces a non-uniform sampling strategy that first generates $k \cdot G$ candidate trajectories $\{o_j\}_{j=1}^{k \cdot G}$ from policy $\pi_{\theta_{\text{old}}}$ (where $k$ scales exploration and $G$ is group size), then selects $G$ elite trajectories via reward-weighted sampling $\text{Sample}(\{o_j\}; R(o_j))$. These are replicated to form the update set $\{o_i\}_{i=1}^G$, maintaining GRPO's update scale while emphasizing high-reward regions. Mathematically equivalent to importance-sampled policy gradient correction, this approach balances broad exploration with computational efficiency through its dual-phase design. To further generalize, GRIP supports two sampling strategies:  

(1) \textit{Local Random Sampling}: For each input question $q$, we first generate $k$ candidate trajectories $\{o_j\}_{j=1}^k$ by independently sampling from the old policy $\pi_{\theta_{\text{old}}}$. The trajectory with the highest reward $o^* = \arg\max_{1 \leq j \leq k} R(o_j)$ is selected as the elite sample. This process is repeated $G$ times to construct the final set $\{o_i\}_{i=1}^G$. This strategy emphasizes deterministic exploitation of the top-performing sample at each iteration while maintaining computational efficiency.  

(2) \textit{Cluster-based Random Sampling}: For each $q$, we generate $k \cdot G$ candidate trajectories $\{o_j\}_{j=1}^{k \cdot G}$. These trajectories are clustered based on their rewards (e.g., reward-binning or K-means clustering), and $G$ trajectories are randomly sampled across clusters to ensure diversity in the final update set. This method balances exploration and exploitation by preserving low-reward but potentially informative samples while still prioritizing high-reward paths.

\paragraph{Adaptive Weighting for Gradient Enhancement.}

Traditional uniform weighting like GRPO ($1/G$) across trajectories fails to account for inter-trajectory quality disparities, leading to misleading gradients from low-quality samples and diminished signals for high-quality ones. GRIP addresses this by assigning trajectory-specific weights via  softmax:  
\begin{equation}
w_i^U = \frac{\exp(\hat{x}_{q, o_i})}{\sum_{j=1}^{G} \exp(\hat{x}_{q, o_j})},
\end{equation}
where the completion score $\hat{x}_{q, o_i}$ can be flexibly configured. For example, when $\hat{x}_{q, o_i} = R(o_i)$, the weighting amplifies the influence of high-reward trajectories. This adaptive weighting suppresses noise from low-quality outputs and strengthens gradients from critical trajectories.

<!-- Source: sections/4_experimental_setup.tex -->

## Experimental Setup

\paragraph{Datasets and Evaluation Metrics.}

To ensure comprehensive evaluation across diverse scenarios, we conduct experiments in nine datasets spanning multiple domains with distinct temporal characteristics and data attributes (detailed in Table  ). These include: the ETT dataset  capturing 2016-2018 electricity load records, Exchange  tracking 1990-2016 foreign exchange rates, Wind  with 2020-2021 wind measurements captured, AQ  providing four-year air quality data, NASDAQ  provides complete stock market series including opening/closing prices, trading volumes, and daily high-low values.
All datasets were evaluated using Mean Squared Error (MSE) and  Mean Absolute Error (MAE) under a 96-step prediction setting, except NASDAQ which uses a 36-step configuration, with results reported as the MSE between predictions and ground truth. Data statistics is listed in Appendix . 

\paragraph{Baselines.}

Our baselines include competitive methods: PatchTST , DLinear , FEDformer , iTransformer , Autoformer , and TimeXer . We further incorporate LLMs-based approaches, CrossTimeNet , GPT4TS~, TimeLLM~, and DeepSeek-R1  for zeroshot. 

\paragraph{Implementation Details.}
Unlike traditional TSF methods that often require normalization, we conduct experiments in the original numerical space. For \Ours, we use Qwen2.5-7B-Instruct as the backbone. In SFT, we train on 300 synthetic samples with a learning rate of 5e-5 for one epoch. In RL, we implement GRIP using the Verl framework  with vLLM for generation. In Eq. , $\epsilon=0.2$ and $\beta=0.04$; group size $G=16$, $k=3$. The batch size is 16, learning rate is 1e-6, policy temperature is 1, and max completion length is 3000. Both stages are run on a 4-GPU A800 cluster. For DeepSeek-R1, we apply its prompt directly to time series prediction without training. Time-R1 is trained only on ETTh1 and generalized to other datasets without fine-tuning, whereas baseline methods require separate models per dataset.

<!-- Source: sections/5_experiments.tex -->

## Experimental Results

### Main Results

\input{table/2_main_results_mse}
We implemented the \Ours framework on nine datasets. The comparison with baseline models is summarized in Table . A more comprehensive list of results for metrics such as MAE can be found in Appendix Table . Key observations are as follows:

(1) \textbf{Limitations of Traditional Methods and LLM Baselines.} Traditional deep learning-based forecasting models, such as PatchTST, DLinear, and iTransformer, achieve reasonable performance but are limited by their one-step "fast thinking" paradigm, which struggles with complex temporal dependencies and high-level reasoning. LLM-based methods like TimeLLM perform better by leveraging the reasoning abilities of LLMs, especially for long-term and non-linear patterns. However, they still treat forecasting as a direct generation task without explicit step-by-step reasoning, leading to potentially inconsistent or logically flawed predictions. Moreover, their reliance on pre-trained knowledge with minimal task-specific adaptation limits their explainability in the forecasting process.

(2) \textbf{Performance Improvement and Advantage of \Ours.}
Our proposed \Ours follows a two-stage optimization framework. In the first stage, CoT-guided SFT is performed using synthetic reasoning trajectories that explicitly encode time series characteristics and task constraints. This enables the model to learn structured output formats and basic reasoning logic. In the second stage, we further enhance the model’s reasoning capabilities through GRIP, a reinforcement learning framework with fine-grained reward mechanisms. These include logical consistency, temporal coherence, and multi-horizon accuracy, which serve as feedback signals to iteratively refine the model's reasoning paths. Experimental results show that this approach not only significantly improves overall forecasting performance but also enhances generalization under zero-shot and out-of-distribution settings.

\input{table/5_ablation_study}

### Ablation Study

\paragraph{Impact of Supervised Fine-tuning and Reinforcement Learning.}

Next, we evaluate the necessity of CoT-based SFT by comparing two training strategies: (i) direct RL without SFT, and (ii) SFT followed by RL. As illustrated in Figure~(b), the model trained without SFT suffers from slower convergence and inferior performance, especially in early training stages. In contrast, initializing RL with a well-aligned SFT model significantly accelerates learning and leads to better final performance. This demonstrates that SFT provides a strong foundation for reasoning path generation, which is then further refined through rule-augmented reinforcement learning.

Furthermore, we conducted an ablation study by completely removing RL (see Table ). The results demonstrate a significant degradation in TSF performance, with absolute performance drops on the ETT and Wind dataset respectively, highlighting RL's crucial role in optimizing SFT-initialized reasoning paths. This finding indicates that while SFT establishes fundamental reasoning patterns, RL provides indispensable optimization through the following mechanisms: (1) discovering higher-reward reasoning trajectories through exploration, and (2) suppressing plausible-yet-incorrect reasoning paths via reward shaping. RL proves to be a critical factor in achieving SOTA performance.

\paragraph{Impact of Multi-objective Reward Design.}

We analyze the impact of each reward term in reinforcement learning by training models with partial reward components. As shown in Table~, removing any term degrades performance, indicating that all contribute to forecasting accuracy. The largest drops occur when MSE or Seasonal-Trend Decomposition rewards are removed, emphasizing the importance of point-wise precision and temporal structure. While Format and Length rewards have smaller effects on numerical metrics, they ensure output consistency and training stability. The Structural Similarity reward further enhances structural fidelity, especially for complex sequences. More experiments are in Appendix .

\paragraph{Impact of Training Template Component.}

Finally, we assess how different elements of our structured prompts affect model behavior. We consider two main components: (i) explicit timestamp encoding, and (ii) contextual information such as seasonal period and task constraints. Table  show that incorporating these components consistently improves both forecasting accuracy and generalization capability, especially under zero-shot and out-of-distribution scenarios. Models without timestamp information struggle to capture long-range dependencies, while those lacking contextual guidance often produce logically inconsistent outputs. 

### Performance Comparison w.r.t Different RL Optimization

We compare the performance of GRIP using two different sampling strategies — Local Random Sampling, and Cluster-based Random Sampling — against GRPO, a commonly used policy optimization method in reasoning-based reinforcement learning. As illustrated in Figure~(a), the Cluster-based Random Sampling strategy achieves the highest overall performance, slightly outperforming GRPO. This is attributed to its ability to maintain diversity in trajectory selection by clustering samples based on reward values, which helps preserve potentially informative yet low-reward reasoning paths often ignored by greedy methods. In terms of convergence speed, Cluster-based Sampling also leads, followed by Local Sampling, and finally GRPO, which converges the slowest. Although local Sampling explores a larger search space, it tends to overfit high-reward trajectories early on, leading to relatively poor generalization and suboptimal performance.

\input{images/3_compare}

### Performance Comparison w.r.t Different Model Types

We analyze the training dynamics of \Ours across model types, comparing base and instruct models. Using two Qwen2.5 variants, Qwen2.5-7B-Base and Qwen2.5-7B-Instruct. Figure (c) shows the base model converges more slowly and starts from a lower performance level. However, it demonstrates stronger learning potential and eventually achieves slightly better results. This suggests that while instruction tuning accelerates early learning in time series reasoning, iterative RL-based optimization enables the base model to reach marginally superior performance.

### Performance Comparison w.r.t Different Model Sizes
 To evaluate the scaling behavior of \Ours, we conduct experiments using models with 1.5B, 3B, and 7B parameters on TSF tasks. As shown in Figure~(d), forecasting performance consistently improves with increasing model size. The 1.5B model achieves reasonable results on simple datasets but struggles with complex temporal patterns. In contrast, the 3B model demonstrates significantly better accuracy and generalization, while the 7B model achieves the best overall performance, particularly in capturing long-term dependencies and handling out-of-distribution scenarios. These results indicate that larger models can substantially enhance temporal reasoning capabilities.

### Visualization of the Reasoning Process

\input{images/4_reasoning_case}

As shown in Figure , our case study highlights key differences among training paradigms. SFT enables imitation of reasoning patterns but often results in superficial replication, leading to flawed logic and suboptimal performance. Pure RL achieves reasonable accuracy but generates think with poor readability. In contrast, the SFT+RL paradigm not only teaches extended reasoning effectively but, through its reinforcement phase, also improves prediction accuracy while helping the model identify which reasoning components most contribute to performance gains.

\input{images/6_ts_case}

### Visualization of Prediction Results

In this visualization (Figure ), \Ours is compared against six baseline methods, including LLM-based approaches (GPT4TS and TimeLLM) and traditional models (PatchTST, iTransformer, Autoformer, and TimeXer). \Ours consistently demonstrates more accurate and smoother predictions, closely aligning with the ground truth. In contrast, the version of \Ours trained with only SFT performs poorly, yielding subpar forecasting results. On the other hand, \Ours with only RL achieves improved performance, outperforming the baselines to some extent.

<!-- Source: sections/6_conclusion.tex -->

## Conclusion

In this work, we proposed \Ours, a generative time series forecasting framework that enables LLMs to perform deliberate reasoning for improved prediction. We
introduced time series reasoning by training LLMs to adopt a slow-thinking paradigm, generating explainable intermediate reasoning steps before producing final forecasts, which achieves state-of-the-art TSF performance.
\Ours consists of a two-stage RFT framework, which first employs CoT-based SFT on synthetic reasoning data to instill multi-step temporal analysis capabilities in the model. This is followed by a novel RL phase with fine-grained, multi-objective reward signals specifically design for TSF. Notably, GRIP (group-based relative importance for policy optimization), a policy optimization method was proposed to leverage non-uniform sampling to refine reasoning paths and improve forecasting accuracy.
Additionally, by open-sourcing the training code, we provide the broader research community and society with practical access to a time series reasoning solution, enabling all to benefit from these advancements.

<!-- Source: sections/8_related_work.tex -->

## Related Work

### TimeSeries Forcasting

Time series forecasting has evolved from classical models like ARIMA, effective under ideal conditions , to modern deep learning approaches. While ARIMA offers theoretical guarantees, it struggles with real-world data complexities. Machine learning methods  remain highly robust due to their interpretability and ability to model nonlinear relationships.  The advent of deep learning introduced sequence-to-sequence models such as Recurrent Neural Networks (RNNs), which initially captured temporal dynamics well . However, RNNs face limitations like restricted receptive fields and error accumulation . Advanced architectures incorporating self-attention and convolutional networks have since been developed to capture long-range dependencies . Concurrently, integrating traditional techniques like trend-seasonal decomposition into neural networks has improved performance . Notably, even simple linear networks enhanced with decomposition strategies can achieve competitive results . Additionally, slice-based methods have shown promise in long-term forecasting by segmenting time series for better accuracy . These advancements blend classical principles with deep learning to tackle the challenges of TSF.

### LLM-based TimeSeries Forcasting

In recent years, large language models (LLMs) have attracted attention for their ability to understand and generate human-like text, now extending into time series analysis . The application of LLMs in this field primarily follows two approaches: fine-tuning and prompt-based zero-shot learning. Fine-tuning involves further training pre-trained LLMs on specific time series data to improve performance , though it requires significant labeled data and computational resources. Conversely, prompt-based zero-shot methods utilize the model's existing knowledge through carefully designed prompts, avoiding task-specific training . While more flexible and resource-efficient, these methods may not match the performance of fine-tuned models, especially in specialized tasks . Both paradigms illustrate the growing interest in using LLMs for time series, despite challenges in optimizing their effectiveness for such tasks.

### Large Language Models and Reinforcement Learning

Reinforcement Learning (RL)  allows an agent to learn decision-making through interactions with its environment, aiming to maximize cumulative rewards. RLHF introduced RL to LLMs via human feedback , initially training a reward model on human preferences and then using it for tuning the policy LLM, often with Proximal Policy Optimization (PPO). However, PPO's complexity due to multiple optimization rounds poses challenges. To simplify this, methods like Direct Preference Optimization (DPO)  and SimPO  have been proposed, offering computational efficiency but suffering from off-policy issues (Pang et al., 2024). Another approach, Group Relative Policy Optimization (GRPO) , avoids a critic model by estimating baselines from group scores, while RLOO  uses a simplified REINFORCE-style framework. Despite these advances, applying RL to enhance LLM-driven reasoning for time series forecasting tasks is still underexplored.

<!-- Source: sections/11_appendix_all.tex -->

\appendix

## Appendix

\input{sections/8_related_work}

## Preliminaries and Analysis of RL in LLMs

### Reinforcement Learning in LLMs

To leverage Reinforcement Learning (RL) for optimizing Large Language Models (LLMs) in Natural Language Processing (NLP) tasks, the initial and crucial step is to formulate the LLM's generation process as a Markov Decision Process (MDP)~. This involves clearly defining the fundamental components of an RL framework: the agent, the environment, states, actions, and rewards. In this context, the LLM itself can be viewed as the agent, interacting with an environment that encompasses the task it is trying to solve (e.g., text generation, question answering). The agent's goal is to learn a policy that maximizes a cumulative reward signal, which reflects how well it performs the given task.

\begin{figure}[h]
    
    \caption{Modeling large language models with reinforcement learning.}
    
\end{figure}

As shown in Figure~, within this MDP framework, the operationalization of RL concepts is specifically tailored to the sequential token-by-token generation characteristic of LLMs. At any discrete time step $t$, the system's current condition is captured by the state $s_t$. This state typically comprises the initial prompt concatenated with the sequence of tokens generated up to that point; thus, the initial state $s_0$ consists solely of the prompt. From a state $s_t$, the LLM executes an action $a_t$, which manifests as the selection and generation of the subsequent token from its predefined vocabulary $V$. The set of all possible tokens constitutes the action space, whose cardinality is denoted as $|V|$. The selection of a particular action $a_t$ is governed by the agent's policy $\pi(a_t|s_t)$, representing a probability distribution over the vocabulary conditional on the current state $s_t$. Following the execution of action $a_t$, the system undergoes a state transition to $s_{t+1}$. In the context of LLM generation, this transition is deterministic, defined by the concatenation $s_{t+1} = [s_t, a_t]$. Crucially for the learning process, a reward signal $r_t$ is provided, and a corresponding value function $V_t$ can be estimated. These elements are contingent upon the specific NLP objective and serve to quantify the desirability of the generated outputs or intermediate actions, thereby guiding the optimization of the LLM's policy.

### Reinforcement Learning with Verified Reward

\paragraph{Group Relative Policy Optimization.}

The GRPO method~ diverges from typical approaches by not requiring a critic model, which often has a comparable size to the policy model. Instead, GRPO calculates the baseline using scores obtained from a group of generated outputs.

Specifically, for every question $q$ sampled from the dataset distribution $P(Q)$, GRPO first employs the existing policy model $\pi_{\theta_{\text{old}}}$ to produce $G$ completions, denoted as $\{o_1, o_2, \cdots, o_G\}$. Subsequently, the policy model $\pi_{\theta}$ is optimized by maximizing a defined objective:

\begin{align}
    \mathcal{J}_{GRPO}(\theta) = 
    \mathbb{E}_{q \sim P(Q), \{o_i\}_{i=1}^G \sim \pi_{\theta_{old}}(o|q)}
    \Bigg\{  
    \frac{1}{G} \sum_{i=1}^G \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} 
    \Big\{ \min \Big[
    \frac{\pi_\theta(o_{i,t} | q, o_{i,<t})}{\pi_{\theta_{old}}(o_{i,t} | q, o_{i,<t})} A_{i},  \notag \\
    \text{clip} \big( \frac{\pi_\theta(o_{i,t} | q, o_{i,<t})}{\pi_{\theta_{old}}(o_{i,t} | q, o_{i,<t})}, 1 - \epsilon, 1 + \epsilon \big) A_{i} \Big] 
    - \beta \mathbb{D}_{KL}\left[\pi_{\theta} || \pi_{ref}\right] \Big\} \Bigg\},
    
\end{align}
\begin{equation}
    \mathbb{D}_{KL}\left[\pi_{\theta} || \pi_{ref}\right] = 
    \frac{\pi_{ref}(o_{i,t}|q,o_{i,<t})}{\pi_{\theta}(o_{i,t}|q,o_{i,<t})}
    - \log\frac{\pi_{ref}(o_{i,t}|q,o_{i,<t})}{\pi_{\theta}(o_{i,t}|q,o_{i,<t})} - 1,
\end{equation}

In this formulation, \(\epsilon\) and \(\beta\) function as hyperparameters. The reference model is denoted by $\pi_{\theta_{\text{ref}}}$, typically representing the model's initial state before reinforcement learning is applied. Furthermore, \(A_i\) signifies the advantage, calculated from a set of rewards \(\{r_1, r_2, \dots, r_G\}\) that correspond to the various completions within each group:

\begin{equation}

    A_i = \frac{r_i - \mathrm{mean}(\{r_1, r_2, \dots, r_G\})}
    {\mathrm{std}(\{r_1, r_2, \dots, r_G\})}.
\end{equation}

\paragraph{Rule-based Reward Function.}
Rather than relying on an auxiliary trained reward model, GRPO employs a rule-based system for reward computation. This system calculates the total reward \(r_i\) for a given output \(o_i\) by aggregating two distinct components, as formalized in Equation~:
\begin{equation}

2    r_i = R_{\text{format}}(o_i) + R_{\text{accuracy}}(o_i),
\end{equation}

Here, the first component, the format reward \( R_{\text{format}}(o_i) \), serves to ensure that the output adheres to the required structural specifications. The second component, the accuracy reward \( R_{\text{accuracy}}(o_i) \), is designed to assign substantially higher values to responses that are correct and precise.

### Analysis of GRPO

\paragraph{Analysing Training Cost and Performance Trade-off.}

Equation  reveals a linear relationship between GRPO's training overhead and the number of sampled completions. This is primarily due to the necessity of computing probability distributions over all completions for the policy, reference, and old policy models. Taking the DeepSeek-Math experiment as an example, sampling 16 completions per question requires 48 forward passes (16 × 3), leading to a sharp increase in computational cost. Experimental results (Figure ) show that increasing the number of completions can improve model MSE and MAE on the ETTh1 dataset, with diminishing returns in performance gains. More critically, reducing the number of completions to lower computational load significantly degrades the reasoning capability of the Qwen2.5-7B-Instruct model, highlighting the limitations of conventional approaches.

\paragraph{Analyzing Optimization Opportunities from Contribution Heterogeneity.}

Recent work  reveals substantial heterogeneity in the contribution of individual completions to training effectiveness. A small subset of high-value samples can provide optimization signals up to tens of times stronger than ordinary ones. This non-uniform distribution opens new avenues for improving training efficiency: by dynamically identifying and prioritizing high-contribution samples, it becomes possible to reduce computational overhead to as low as one-third or even one-fifth of the original level while maintaining model performance (see Section  for detailed optimization strategies). These findings not only explain the efficiency bottlenecks in existing methods but also lay a theoretical foundation for the design of adaptive sampling strategies in future work.

\paragraph{Analysising Completion Contribution.}
To evaluate the contribution of each completion to the training of the policy model, we analyze the derivative structure of the objective function in Eq.\,() with respect to the model parameters $\theta$ as:

\begin{equation}
\begin{aligned}
 \nabla_{\theta} J_{GRPO}(\theta)=&\,\mathbb{E}_{\left[q \sim P(Q), \{o_{i}\}_{i=1}^{G} \sim \pi_{\theta_{old}}(O|q)\right]} \Bigg\{ \frac{1}{G} \sum_{i=1}^{G} \frac{1}{\left|o_{i}\right|} \sum_{t=1}^{\left|o_{i}\right|} \Bigg[ \nabla_{\theta}\left(\frac{\pi_{\theta}\left(o_{i, t} | q, o_{i,<t}\right)}{\pi_{\theta_{old}}\left(o_{i, t} | q, o_{i,<t}\right)} A_i\right) \\
&\quad\quad\quad\quad - \beta\left(\nabla_{\theta} \frac{\pi_{r e f}\left(o_{i, t} | q, o_{i,<t}\right)}{\pi_{\theta}\left(o_{i, t} | q, o_{i,<t}\right)}-\nabla_{\theta} \log \frac{\pi_{r e f}\left(o_{i, t} | q, o_{i,<t}\right)}{\pi_{\theta}\left(o_{i, t} | q, o_{i,<t}\right)}\right) \Bigg]\Bigg\} \\
=&\mathbb{E}_{\left[q \sim P(Q), \{o_{i}\}_{i=1}^{G} \sim \pi_{\theta_{old}}(O|q)\right]} \Bigg\{ \frac{1}{G} \sum_{i=1}^{G} \frac{1}{\left|o_{i}\right|} \sum_{t=1}^{\left|o_{i}\right|} \Bigg[ \frac{\nabla_{\theta} \pi_{\theta}\left(o_{i, t} | q, o_{i,<t}\right)}{\pi_{\theta_{old}\left(o_{i, t} | q, o_{i,<t}\right)}}A_i \\
&+ \beta\left(\frac{\pi_{r e f}\left(o_{i, t} | q, o_{i,<t}\right)\nabla_{\theta} \pi_{\theta}\left(o_{i, t} | q, o_{i,<t}\right)}{\pi_{\theta}^{2}\left(o_{i, t} | q, o_{i,<t}\right)} - \frac{\nabla_{\theta} \pi_{\theta}\left(o_{i, t} | q, o_{i,<t}\right)}{\pi_{\theta}\left(o_{i, t} | q, o_{i,<t}\right)}\right) \Bigg] \Bigg\} \\
=&\mathbb{E}_{\left[q \sim P(Q), \{o_{i}\}_{i=1}^{G} \sim \pi_{\theta_{old}}(O|q)\right]} \Bigg\{  \frac{1}{G} \sum_{i=1}^{G} \frac{1}{\left|o_{i}\right|} \sum_{t=1}^{\left|o_{i}\right|} \Bigg[ \frac{\pi_{\theta}\left(o_{i, t} | q, o_{i,<t}\right)}{\pi_{\theta_{old}}\left(o_{i, t} | q, o_{i,<t}\right)} A_i  \\ 
&\quad\quad\quad\quad\quad\quad\quad + \beta\left(\frac{\pi_{r e f}\left(o_{i, t} | q, o_{i,<t}\right)}{\pi_{\theta}\left(o_{i, t} | q, o_{i,<t}\right)} - 1\right)\Bigg] \frac{\nabla_{\theta} \pi_{\theta}\left(o_{i, t} | q, o_{i,<t}\right)}{\pi_{\theta}\left(o_{i, t} | q, o_{i,<t}\right)}\Bigg\} \\
=&\mathbb{E}_{\left[q \sim P(Q), \{o_{i}\}_{i=1}^{G} \sim \pi_{\theta_{old}}(O|q)\right]} \Bigg\{ \frac{1}{G} \sum_{i=1}^{G} \frac{1}{\left|o_{i}\right|} \sum_{t=1}^{\left|o_{i}\right|} \Bigg[  \underbrace{\frac{\pi_{\theta}\left(o_{i, t} | q, o_{i,<t}\right)}{\pi_{\theta_{old}}\left(o_{i, t} | q, o_{i,<t}\right)}}_{
\textit{Probability ratio}} \underbrace{A_i }_{\textit{Advantage}}\\ 
&\quad\quad\quad\quad\quad + \underbrace{\beta\left(\frac{\pi_{r e f}\left(o_{i, t} | q, o_{i,<t}\right)}{\pi_{\theta}\left(o_{i, t} | q, o_{i,<t}\right)}-1\right)}_{\textit{KL divergence constraint}}   \Bigg] \underbrace{\nabla_{\theta} \log \pi_{\theta}\left(o_{i, t} | q, o_{i,<t}\right)}_{
\textit{Policy model gradient}}\Bigg\}.
\end{aligned}

\end{equation}

\begin{figure}[h]
    
    \caption{Completion number \textit{vs.} (left) MSE $\downarrow$ and (right) MAE $\downarrow$. Experiments are conducted on ETTh1 using Qwen2.5B-7B-Instruct.}
    
\end{figure}

This analysis reveals four key factors influencing policy updates:  

1. \textbf{\textit{Advantage}}, which assesses the value of a completion in improving expected returns through the advantage function. A higher advantage indicates stronger reward alignment, making the completion more influential in guiding policy updates.

2. \textbf{\textit{Probability ratio}}, wich compares the likelihood of an action under the current policy $\pi_\theta$ to that under the old policy $\pi_{\theta_{\text{old}}}$. It amplifies actions favored by the new policy and suppresses those preferred by the old one, guiding the policy toward higher rewards. A higher ratio signifies greater confidence in the action, influencing the optimization process more significantly. This term is crucial for identifying high-value completions when combined with the advantage function.

3. \textbf{\textit{KL divergence}}, which measures the deviation of the current policy from the reference model. It enforces stability during training by penalizing excessive changes, but does not directly contribute to reasoning pattern formation.

4. \textbf{\textit{Policy model gradient}}, which indicates the direction of parameter updates.

Previous research  has shown that removing the KL divergence constraint does not significantly affect the model's reasoning performance, as the core learning signal primarily comes from the advantage term aligned with rewards. Furthermore, we decompose the core expression for policy updates into two components: the \textit{probability ratio} and the \textit{advantage} term. For a completion to make a significant contribution to training, both components must have substantial values. If either of them is close to zero, the overall contribution will also be negligible. 
By removing the KL divergence term and decoupling its regularization effect, we derive the simplified form of GRPO’s objective derivative as:

\begin{equation}
\begin{aligned}
\nabla_{\theta} J_{GRPO}(\theta) &  \approx  \, \mathbb{E}_{\left[q \sim P(Q), \{o_{i}\}_{i=1}^{G} \sim \pi_{\theta_{old}}(O|q)\right]} \\
&\Bigg\{ \frac{1}{G} \sum_{i=1}^{G} \frac{1}{\left|o_{i}\right|} \sum_{t=1}^{\left|o_{i}\right|} \Big[ \underbrace{\frac{\pi_{\theta}\left(o_{i, t} | q, o_{i,<t}\right)}{\pi_{\theta_{old}}\left(o_{i, t} | q, o_{i,<t}\right)} }_{\substack{\textit{Probability ratio}}}\cdot \underbrace{A_i}_{\substack{\textit{Advantage}}}  \Big] \underbrace{\nabla_{\theta} \log \pi_{\theta}\left(o_{i, t} | q, o_{i,<t}\right)}_{\substack{\textit{Policy model gradient}}} \Bigg\},
\end{aligned}

\end{equation}

This simplified formulation focuses on the reward-driven learning signal while preserving the essential gradient dynamics required for effective policy optimization. With the help of this simplified expression, we can evaluate the advantage value of each completion before the full model computation is carried out. Specifically, if a completion exhibits a very small absolute advantage value, its contribution to the policy update is negligible. We can filter out such low-value completions at an early stage, thereby avoiding redundant forward and backward computations.

To further improve training efficiency and learning effectiveness, we could introduce a sampling strategy based on reward values or advantage estimates. Unlike uniform sampling across all completions, we prioritize those with higher advantage values for inclusion in the training process. These samples typically contain stronger learning signals and are more valuable for policy updates. By increasing the sampling probability of these high-impact completions during batch construction—or through upsampling techniques—we not only reduce computational overhead but also significantly accelerate convergence and enhance final model performance.

This approach ensures efficient training iterations while maintaining the quality of policy updates, achieving a favorable balance between computational cost and learning effectiveness. As a result, it enables a dual improvement in both training efficiency and model capability.

## Group-based Relative Importance for Policy Optimization

We introduce GRIP (Group-based Relative Importance for Policy Optimization), a general RL optimization method designed for optimizing entire trajectories generated by LLMs  as time series forecasting reasoners. Unlike GRPO, which linearly increases inference cost with the number of completions due to uniform sampling and equal weighting, GRIP adopts a non-uniform sampling strategy that selects a small subset of high-reward trajectories from a larger candidate pool, significantly reducing forward passes. Furthermore, GRIP employs adaptive weighting to amplify gradient signals from informative samples, enabling more efficient learning from high-value completions. This design not only reduces compute burden but also mitigates the diminishing returns of increased sample count, thereby enhancing both training efficiency and forecasting accuracy. The GRIP objective function, formally defined in Equation~, operates within a policy gradient framework and integrates a non-uniform sampling strategy with adaptive trajectory weighting.
\begin{align}
\mathcal{J}_{\text{GRIP}}(\theta) = 
&\mathbb{E}_{\substack{q \sim P(Q), \\ \{o_j\}_{j=1}^{k \cdot G} \sim \pi_{\theta_{\text{old}}}(o|q), \\ \{o_i\}_{i=1}^G \sim \text{Sample}\left(\{o_j\}_{j=1}^{k \cdot G}; R(o_j)\right)}} 
\Bigg\{ 
\sum_{i=1}^{G} w_i^U  
\frac{1}{|o_i|} 
\bigg\{
\sum_{t=1}^{|o_i|} 
\min\bigg[ 
\frac{\pi_\theta(o_{i,t}|q, o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t}|q, o_{i,<t})} A_i,\nonumber\\
&
\text{clip}\Big( \frac{\pi_\theta(o_{i,t}|q, o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t}|q, o_{i,<t})}, 1-\epsilon, 1+\epsilon \Big) A_i 
\bigg]
- \beta \mathbb{D}_{KL} [\pi_\theta || \pi_{ref}] 
\bigg\} 
\Bigg\}.

\end{align}  
where $\epsilon$ and $\beta$ are hyperparameters. $\pi_{\text{ref}}$ is the reference model, typically initialized as the pre-trained model before reinforcement learning begins. The output $\{o_i\}$ is selected through a sampling process from policy $\pi_{\theta_{\text{old}}}$. The hyperparameter $k$ controls the size of the rollout space, while $G$ referred to as the group size. $\mathbb{D}_{KL}$ represents the KL divergence, which is incorporated into the loss function as a regularization term during training. And $A_i$ is the advantage computed using a group of rewards $\{r_1, r_2, \dots, r_G\}$ corresponding to the completion trajectories within each group.
The weight $w_i^U$ denotes the adaptive weighting assigned to each trajectory. This objective balances exploration and exploitation while mitigating gradient dilution. The sampling strategy and adaptive weight will be discussed in the following section.

\paragraph{GRIP Pipeline.}
To elucidate its operational mechanics, the GRIP algorithm is implemented through a structured pipeline. The distinct stages of this process are outlined as follows:

(1) The old policy model samples $k$ groups of $G$ completions for each question, a total of $k\cdot G$.

(2) The reward function computes the reward for each completion (Sec. ).

(3) GRIP non-uniformly samples $G$ completions based on rewards to balance exploration and exploitation (Sec. ).

(4) The advantage of each completion is calculated, and adaptive weighting is employed to assign greater significance to high-quality reasoning paths among these completions (Sec. ).

(5) Subsequently, the policy model is updated, with its gradient signals effectively formed as a weighted average of the selected completions, reflecting this assigned path-dependent significance.

GRIP differs significantly from GRPO in both its initial sampling strategy and the mechanism for weighting completion contributions during policy updates. GRPO typically generates $G$ completions directly and employs an arithmetic mean when processing their outcomes. In contrast, GRIP first explores a broader set by sampling $k \cdot G$ completions, from which $G$ are subsequently selected for the update phase. Critically, while GRPO's use of an arithmetic mean implies equal consideration for its $G$ samples, GRIP's policy update leverages an adaptive weighted average. This ensures that high-quality reasoning paths within the selected $G$ completions exert a more substantial influence on the gradient signals, thereby fostering more robust and effective learning.

## CoT-based Supervised Fine-tuning Data collection

\begin{algorithm}
\caption{Three-stage CoT data construction for \textsc{Ours}}

\begin{algorithmic}[1]
\STATE \textbf{Input:} $\mathcal{T}$ := Set of training time series samples with historical data and ground truth labels  
\STATE \textbf{Output:} $\mathcal{D}_{\text{SFT}}$ := Structured Chain-of-Thought dataset for SFT  
\STATE $\mathcal{D}_{\text{SFT}} \gets \emptyset$ \hfill \textit{Initialize the output dataset}

\FOR{$t \in \mathcal{T}$}
    \STATE $x_{\text{hist}} \gets \text{ExtractHistorical}(t)$ 
    \STATE \hfill \textit{Extract historical time series input}
    
    \STATE $y_{\text{pred}}^{(1)}, \dots, y_{\text{pred}}^{(k)} \gets \text{DeepSeek-R1}(x_{\text{hist}}; \text{strict formatting})$ 
    \STATE \hfill \textit{Generate $k$ candidate predictions}
    
    \STATE $\text{MAPE}_i \gets \text{ComputeMAPE}(y_{\text{pred}}^{(i)}, y_{\text{true}})$ for each candidate 
    \STATE \hfill \textit{Evaluate using MAPE}
    
    \STATE $y^* \gets \text{SelectMinMAPE}(\{y_{\text{pred}}^{(i)}\})$ 
    \STATE \hfill \textit{Select best prediction}
    
    \STATE $\text{CoT}^{(1)}, \dots, \text{CoT}^{(k)} \gets \text{DeepSeek-R1}(\text{prompt}=x_{\text{hist}}, y_{\text{true}}, \text{CoT of $y^*$})$ 
    \STATE \hfill \textit{Prompt to generate reasoning paths based on ground truth label}
    
    \STATE $\text{CoT}^* \gets \text{SelectHighQuality}(\{\text{CoT}^{(i)}\}, y_{\text{true}})$ 
    \STATE \hfill \textit{Select CoT aligned with ground truth label}
    
    \STATE $\text{sample} \gets \text{Concatenate}(\text{\textless think\textgreater}, \text{CoT}^*, \text{\textless \textbackslash think\textgreater}, \text{\textless answer\textgreater},  y_{\text{true}}, \text{\textless \textbackslash answer\textgreater})$ 
    \STATE \hfill \textit{Combine reasoning and true answer}
    
    \STATE $\mathcal{D}_{\text{SFT}} \gets \mathcal{D}_{\text{SFT}} \cup \{\text{sample}\}$ 
    \STATE \hfill \textit{Add to final dataset}
\ENDFOR

\end{algorithmic}
\end{algorithm}

## Data Statistics

We presents the statistical characteristics of our datasets, shown in table .
\input{table/1_data_statistics}

## Additional Result and Analysis

### Main Results

Due to page limitations, we present the main results using the MAE evaluation metric here in Table . Compared to both traditional methods and LLM-based approaches, \Ours achieves competitive improvements.
\input{table/8_main_results_mae}

### Analysis about Reward Design

\paragraph{Distance Metric in Accuracy Reward.}
To investigate the impact of different distance metrics on the accuracy reward during the reinforcement learning phase, we conducted experiments on multiple real-world time series forecasting tasks using the Exchange, AQShunyi, and NASDAQ datasets, shown in Figure . Specifically, we evaluated Mean Squared Error (MSE), Mean Absolute Error (MAE), Dynamic Time Warping (DTW), and Mean Absolute Percentage Error (MAPE) as accuracy reward signals within our GRIP optimization framework. The experimental results demonstrate that MSE consistently outperformed other metrics, followed by MAE, while DTW and MAPE showed relatively limited improvements.
MSE penalizes the squared prediction errors, making it more sensitive to outliers and thereby guiding the model to focus on overall consistency between predictions and ground truth values. This leads to enhanced stability in multi-step forecasts. In contrast, although MAE is more robust due to its linear response to errors, it suffers from less concentrated gradient updates, affecting convergence efficiency. DTW, despite its ability to handle temporal misalignment, introduces computational complexity and asymmetry issues, making it challenging to integrate effectively into end-to-end training. Additionally, MAPE can suffer from numerical instability when target values are close to zero, limiting its practical utility in training.
In conclusion, we recommend prioritizing MSE as the primary accuracy reward metric during the reinforcement learning optimization phase to achieve superior predictive accuracy and reasoning stability. Auxiliary reward terms may be incorporated based on specific task requirements to further enhance model robustness.

\input{images/5_distance}

### Impact of Predicted Window Length

\input{table/4_predicted_window_length}

The results in Table  demonstrate that \Ours consistently outperforms TimeLLM across multiple datasets and prediction horizons, particularly in long-sequence rolling forecasting tasks. Our approach employs a rolling prediction strategy: first using the historical 96 time steps to predict the subsequent 96 steps, and then recursively feeding the predicted values back as input to forecast further into the future. In the ETTh1 dataset, \Ours achieves an MSE of 6.4146 and an MAE of 1.2693 under the 192-step forecasting window, substantially lower than TimeLLM’s MSE of 7.7990 and MAE of 1.4989, highlighting its superior capability in multi-step rolling prediction. Under the longer 336-step forecasting window, \Ours obtains an MSE of 6.6886, still significantly outperforming TimeLLM’s MSE of 9.3575, which indicates that the model maintains high accuracy and stability even after multiple recursive prediction steps.

### Case Study: an Input and Output Example of \Ours

We provide a complete input-output example of \Ours for reference, with some time steps omitted, shown in Table .

\input{table/7_full_data}

## Limitations

Despite significant advancements in enhancing the model’s ability to capture high-frequency temporal patterns and improve time series reasoning, challenges remain due to computational limitations that prevent experimentation with larger open-source models. Additionally, extending the length of the prediction window is not feasible. Consequently, our method may encounter inaccurate predictions when dealing with complex temporal structures or long-term forecasting requirements.

## Broader Impact

This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences  of our work, none which we feel must be specifically highlighted.

\clearpage