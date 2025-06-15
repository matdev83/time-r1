
# Time‑R1 Implementation Plan

*Generated on 2025-06-15T20:21:59Z*

---

## 1  Quick On‑Boarding for Coding Agents

| Key | Value |
|-----|-------|
| **Objective** | Build a complete, reproducible Python project that trains, evaluates, and serves the **Time‑R1** foundation model for time‑series forecasting |
| **Stages** | Prompt Templating → SFT Warm‑Up → GRIP RL → Slow‑Thinking Inference → Evaluation/Serving |
| **Stack** | Python ≥ 3.10 · PyTorch ≥ 2.2 · 🤗 Transformers / PEFT · trl · FastAPI · Lightning CLI · MLflow or Weights & Biases |
| **Standards** | **TDD** (pytest+coverage), **SOLID**, **DRY**, **KISS** |
| **Tooling** | `poetry` (env+build), `black` (fmt), `isort`, `ruff`, `mypy`, `pre‑commit`, GitHub Actions CI |
| **Acceptance Gate** | All tests pass · QA metrics meet thresholds · CI/CD pipeline green |

---

## 2  Repository Skeleton

```text
time-r1/
├── data/                 # Raw & processed datasets (git‑ignored)
├── src/
│   └── time_r1/              # Source package
│       ├── __init__.py
│       ├── config/           # YAML & pydantic settings
│       ├── datasets/         # DataModule implementations
│       ├── prompts/          # Jinja2 templates & builders
│       ├── models/           # Model wrappers
│       │   ├── sft/
│       │   └── grip/
│       ├── training/         # Training loops / LightningTasks
│       ├── inference/        # Slow‑thinking engine & APIs
│       ├── evaluation/       # Metrics & benchmark runners
│       └── utils/
├── scripts/              # CLI entry‑points
├── tests/                # PyTest suites (mirrors src tree)
├── docs/                 # Sphinx + MkDocs
├── pyproject.toml        # Build + dev dependencies
└── README.md
```

---

## 3  High‑Level Workflow

```mermaid
flowchart LR
    subgraph Training Pipeline
        P[Prompt Template Builder] --> SFT[SFT Warm‑Up]
        SFT --> GRIP[GRIP RL\n(Guided Reward Improvement)]
    end
    GRIP -->|checkpoint| M[Time‑R1 Weights]
    subgraph Inference & Serving
        M --> ST[Slow‑Thinking Executor]
        ST --> R[Forecast Results]
    end
    ST -->|metrics| E[Evaluation Suite]
```

---

## 4  Task & Sub‑Task Break‑Down

> **Notation**  
> **I** = Inputs • **D** = Deliverables • **T** = Tests • **A** = Acceptance Criteria

### **T‑0  Project Bootstrap**

- **Goal** Establish scalable, test‑first repo with automated CI.
- **I** 🗅 —  
- **D** `pyproject.toml`, GitHub Actions, `pre‑commit` hooks.  
- **T** `pytest -q`, flake/ruff, mypy.  
- **A** CI workflow passes on push & PR.

---

### **T‑1  Dataset Ingestion & Pre‑Processing**

- **Goal** Prepare diverse time series datasets for training and evaluation, ensuring data quality and accessibility.
- **I** Raw time series datasets (e.g., ETT, Exchange, Wind, AQ, NASDAQ as mentioned in the paper).
- **D** Processed `*.parquet` files, PyTorch LightningDataModule.
- **T** Checksum tests, schema validation unit tests, statistical drift tests.
- **A** Files exist, schema OK, $\le 1e-3$ diff in statistical drift, DataModule passes smoke train-loop.
- **Sub-Tasks**:

    | Sub‑Task | I | D | T | A |
    |----------|---|---|---|---|
    | **1.1 Downloaders** | dataset list (JSON) | `*.csv` raw | checksum test | files exist |
    | **1.2 Schema Validation** | raw csv | `pydantic` schemas | unit tests | schema OK |
    | **1.3 Feature Engineering** | raw | `*.parquet` processed | statistical drift test | ≤1e‑3 diff |
    | **1.4 DataModule** | processed | PyTorch LightningDataModule | smoke train‑loop |  ✔ |

---

### **T‑2  Prompt Template Builder**

- **Goal** Develop a standardized instructional framework for LLM inputs, encoding task-specific knowledge.
- **I** Task specification, data schema, Jinja2 templates.
- **D** `src/time_r1/prompts/PromptTemplate.py` class, YAML files for Jinja templates.
- **T** Render snapshot tests, validation against expected output formats.
- **A** 100% deterministic render for fixed seed; adheres to paper's "Training Template" components.
- **Implementation Details**:
  - The `PromptTemplate` class should encapsulate the five components described in the paper's "Training Template" section:
        1. **Task Definition**: Establish objectives and problem scope.
        2. **Dataset Description**: Specify temporal characteristics and application scenarios.
        3. **Channel Information**: Delineate input signal types.
        4. **Testing Data**: Provide timestamps and historical series.
        5. **Format Instruction**: Define output templates (e.g., `<think>`, `<answer>` tags).
  - Use Jinja2 for flexible template rendering, allowing dynamic insertion of historical data and task-specific instructions.

---

### **T‑3  Supervised Fine‑Tuning (SFT)**

- **Goal** Warm-up adaptation of the LLM for time series forecasting, ensuring stable training and proper output formatting.
- **I** Processed time series data, base LLM (e.g., Qwen2.5-7B-Instruct).
- **D** Hugging Face Dataset object, `lora_config.json`, `sft.pt` checkpoint.
- **T** Sample shape tests, parameter count tests, regression tests (loss < $\tau$).
- **A** Dataset samples have correct shape, LoRA config applies with minimal parameter diff, training loss $\le \tau=0.05$, W&B run exists.
- **Sub-Tasks**:

    | Sub‑Task | I | D | T | A |
    |----------|---|---|---|---|
    | **3.1 SFT Data Construction** | processed data | HF Dataset object | sample shape test | len>0 |
    | **3.1.1 Initial Prediction Generation** | historical data | candidate predictions | unit tests | DeepSeek-R1 generates $k$ predictions |
    | **3.1.2 Optimal Prediction Selection** | candidate predictions, ground truth | best prediction | MAPE test | selects min MAPE prediction |
    | **3.1.3 CoT Refinement** | historical data, true prediction, CoT of best pred | refined CoT | unit tests | CoT aligns with ground truth |
    | **3.1.4 Structured Data Creation** | refined CoT, true prediction | SFT training sample | format test | `<think>`, `<answer>` tags present |
    | **3.2 LoRA Config Generation** | base LLM id | `lora_config.json` | parameter count test | diff<1% |
    | **3.3 Trainer Script Implementation** | 3.1 + 3.2 | ckpt `sft.pt` | regression test (loss<τ) | loss $\le \tau=0.05$ |
    | **3.4 Logging & Experiment Tracking** | training metrics | W&B run | CI artifact test | run exists, metrics logged |

- **Implementation Details**:
  - The SFT data construction should follow the three key steps outlined in the paper's "Supervised Fine-tuning for Warmup Adaptation" section and Algorithm 1:
        1. Leverage DeepSeek-R1 (or similar LLM) to generate initial time-series predictions on the training set with strict formatting.
        2. Select the optimal prediction for each sample based on the Mean Absolute Percentage Error (MAPE) metric.
        3. Inject the true prediction value and the high-quality CoT from the previous step back into the LLM as prompts to synthesize a revised CoT that logically culminates in the correct prediction.
        4. Concatenate the refined CoT and the true prediction value, demarcating the final answer using `<answer>` tags and reasoning with `<think>` tags to create structured training data.
  - Perform a single-epoch fine-tuning with a small learning rate (e.g., 5e-5 as in the paper).

---

### **T‑4  GRIP RL Stage**

- **Goal** Refine the SFT model for generalization and slow-thinking capabilities using reinforcement learning, guided by fine-grained, multi-objective rewards.
- **I** `sft.pt` (SFT-trained model), time series data, reward function components.
- **D** `src/time_r1/training/grip_trainer.py`, `grip.pt` (final GRIP-trained model).
- **T** Parameterized reward tests, convergence tests (KL divergence, performance metrics), unit tests for checkpointing and safety filters.
- **A** Reward $\in [-1,1]$, KL < 0.1, performance improvement over SFT, top-k checkpoints saved, toxic output rate $\le 0.2\%$.
- **Sub-Tasks**:

    | Sub‑Task | I | D | T | A |
    |----------|---|---|---|---|
    | **4.1 Reward Function Design** | ground-truth, prediction | `src/time_r1/training/reward.py` | parameterized tests | reward $\in [-1,1]$ |
    | **4.1.1 Format Rewards** | model output | `gamma_Format`, `gamma_Length` | unit tests | syntactic validity, completeness |
    | **4.1.1.1 Format Reward ($\gamma_{\text{Format}}$)** | model output | binary penalty | unit tests | Ensures `<think>`, `<answer>` tags and proper structure. $\gamma_{\text{Format}} = 0$ if valid, $-1$ otherwise. |
    | **4.1.1.2 Length Reward ($\gamma_{\text{Length}}$)** | generated sequence length, ground truth length | positive feedback | unit tests | Encourages full sequence generation. $\gamma_{\text{Length}} = 0.1$ if $\text{len(answer)} \ge \text{len(ground\_truth)}$, else $0.1 \cdot \frac{\text{len(answer)}}{\text{len(ground\_truth)}}$. |
    | **4.1.2 Accuracy Rewards** | normalized pred/target | `gamma_MSE`, `gamma_Seasonal`, `gamma_Trend` | unit tests | numerical precision, temporal fidelity |
    | **4.1.2.1 MSE Reward ($\gamma_{\text{MSE}}$)** | normalized prediction, target | bounded reward signal | unit tests | Assesses numerical precision. $\gamma_{\text{MSE}} = \left(1 - \frac{1}{1 + e^{-0.3 \cdot \text{MSE}}}\right) \cdot 2$. |
    | **4.1.2.2 Seasonal-Trend Decomposition Rewards ($\gamma_{\text{Seasonal}}, \gamma_{\text{Trend}}$)** | predicted/true sequences | separate MSE terms | unit tests | Captures temporal structure. $\gamma_{\text{Seasonal}} = \frac{1}{n} \sum_{i=1}^{n} \left(s_i^{\text{true}} - s_i^{\text{pred}}\right)^2$, $\gamma_{\text{Trend}} = \frac{1}{n} \sum_{i=1}^{n} \left(t_i^{\text{true}} - t_i^{\text{pred}}\right)^2$. |
    | **4.1.3 Structural Similarity Reward ($\gamma_{\text{CP}}$)** | predicted/ground-truth extrema | credit for matches | unit tests | Ensures change-point capture. $\gamma_{\text{CP}} = \left(\frac{N_{\text{cmax}}}{N_{\text{gmax}}} \cdot 0.2 \right) + \left(\frac{N_{\text{cmin}}}{N_{\text{gmin}}} \cdot 0.2\right)$. |
    | **4.2 GRIP Trainer Implementation** | sft.ckpt, reward fn | `src/time_r1/training/grip_trainer.py`, `grip.pt` | convergence test | KL < 0.1, performance improvement |
    | **4.2.1 Non-uniform Sampling Strategy** | $k \cdot G$ candidates | `src/time_r1/training/sampling_strategy.py` | unit tests | Balances exploration/exploitation. Implement "Local Random Sampling" and "Cluster-based Random Sampling" as described in the paper. |
    | **4.2.2 Adaptive Weighting for Gradient Enhancement** | trajectory rewards | `src/time_r1/training/adaptive_weighting.py` | unit tests | Amplifies high-quality gradients. Implement $w_i^U = \frac{\exp(\hat{x}_{q, o_i})}{\sum_{j=1}^{G} \exp(\hat{x}_{q, o_j})}$, where $\hat{x}_{q, o_i} = R(o_i)$. |
    | **4.3 Checkpointing** | training state | Best ckpt tags | unit test | top-k saved, reproducible |
    | **4.4 Safety Filters** | policy outputs | `src/time_r1/inference/filters.py` | toxic rate test | $\le 0.2\%$ toxic output |

- **Implementation Details**:
  - The overall reward for RL training is the sum of the above rewards.
  - Implement GRIP objective function as formalized in Equation 6 of the paper, integrating non-uniform sampling and adaptive weighting.
  - Use hyperparameters $\epsilon=0.2$ and $\beta=0.04$ (as per paper's experimental setup).
  - Group size $G=16$, $k=3$ for candidate trajectories.
  - Batch size 16, learning rate 1e-6, policy temperature 1, max completion length 3000.
  - Training on a multi-GPU setup (e.g., 4-GPU A800 cluster as in the paper).

---

### **T‑5  Slow‑Thinking Inference Engine**

- **Goal** Implement the multi-step reasoning engine for Time-R1 inference.
- **Sub-Steps**: **draft → reflect → refine** (iterative process).
- **I** `grip.pt` (trained GRIP model), input prompt, historical time series data.
- **D** `src/time_r1/inference/slow_think.py` (modular chain API), `src/time_r1/inference/executor.py` (inference executor).
- **T** Gold forecast vs baseline, latency benchmarks.
- **A** MASE improvement $\ge 10\%$ over SFT baseline; inference latency $\le 500$ ms (A100 GPU).
- **Implementation Details**:
  - **5.1 Draft Module**: Generates initial forecast based on prompt and historical data.
  - **5.2 Reflect Module**: Evaluates the draft, identifies inconsistencies or areas for improvement (e.g., using a critic model or rule-based checks).
  - **5.3 Refine Module**: Adjusts the forecast based on reflection, potentially re-prompting the model or applying post-processing.
  - **5.4 Chain Orchestration**: Implement a flexible API (e.g., using a custom pipeline or a framework like LangChain) to orchestrate these modules iteratively until a satisfactory forecast is achieved or a maximum number of iterations is reached.

---

### **T‑6  Evaluation & Benchmark Harness**

- Metrics: **MSE**, **MAE**, **MAPE**, **R²**, **NRMSE**.  
- **I** model, test sets.  
- **D** `eval_cli.py`, pytest fixtures.  
- **T** metric reproducibility.  
- **A** CI run outputs JSON report.

---

### **T‑7  API & Service Layer**

| Sub‑Task | I | D | T | A |
|----------|---|---|---|---|
| **7.1 FastAPI App** | slow_think.py | `main.py` | `/health` e2e | 200 OK |
| **7.2 Dockerfile** | repo | image | docker run test | inf latency < 2×RT |
| **7.3 Helm Chart (opt.)** | image | chart/ | kind test | pod ready |

---

### **T‑8  Experiment Tracking & Reproducibility**

- **I** config dicts.  
- **D** `conf/` YAML + seeder.  
- **T** hash‑match test.  
- **A** run can be reproduced via `make replay HASH=…`.

---

### **T‑9  MLOps & Production Readiness**

- **Goal** Ensure robust, scalable, and secure deployment of the Time-R1 model.
- **Sub-Tasks**:

    | Sub‑Task | I | D | T | A |
    |----------|---|---|---|---|
    | **9.1 Data Versioning** | raw/processed data | DVC/MLflow artifacts | checksum test | data integrity |
    | **9.2 Model Registry** | `sft.pt`, `grip.pt` | MLflow Model Registry | API access test | model versioned |
    | **9.3 Monitoring & Alerting** | inference logs, metrics | Prometheus/Grafana setup | alert trigger test | anomaly detected |
    | **9.4 Security Hardening** | FastAPI app, Dockerfile | security best practices | penetration test | no critical vulns |
    | **9.5 Scalability Design** | traffic patterns | auto-scaling config | load test | handles peak load |

---

### **T‑10  Documentation**

- Sphinx `autodoc`, MkDocs site.  
- Smoke build test in CI.

---

## 5  Deliverables Checklist

- 🗹 `time_r1` Python package + unit tests (>90 % coverage)  
- 🗹 Pre‑trained checkpoints (`sft.pt`, `grip.pt`)  
- 🗹 Docker image pushed to registry  
- 🗹 Minimal API demonstration notebook  
- 🗹 Comprehensive docs site at `/docs`

---

## 6  Acceptance Gate Summary

| Metric | Threshold |
|--------|-----------|
| **Code‑cov** | ≥ 90 % |
| **Train loss Δ** | −tocvx until ≤ 0.05 |
| **Forecast MASE** | Better than SOTA baseline |
| **Latency (128 steps)** | ≤ 500 ms GPU A100 |
| **CI Pass Rate** | 100 % |

---

### Happy coding 🤖
