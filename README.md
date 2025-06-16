# Time-R1 Project

## 🎯 Project Goal

The primary objective is to build a complete, reproducible Python project that trains, evaluates, and serves the **Time-R1** foundation model. This model is designed to enhance multi-step reasoning ability of LLMs for time series forecasting, leveraging a two-stage reinforcement fine-tuning framework (SFT Warm-Up followed by GRIP RL).

## 🗺️ Where to Find More Details

* **Implementation Plan (`dev/PLAN.md`)**: This file contains the comprehensive, state-of-the-art implementation plan for the entire ML workflow. It breaks down tasks into sub-tasks, specifies inputs, deliverables, tests, and acceptance criteria, and includes detailed implementation notes derived directly from the academic paper. **This is your primary guide for task execution.**

* **Academic Paper (`dev/time_r1_paper/time_r1_paper.md`)**: For the foundational theoretical background, detailed algorithms, mathematical formulations, and experimental results, refer to the `time_r1_paper.md`. This paper provides the scientific basis for the Time-R1 framework.

## ⚙️ Development Setup

Install the project in editable mode with the development extras and set up
pre-commit hooks:

```bash
pip install -e .[dev]
pre-commit install
```

Running `pre-commit` will format the code with **black** and **isort**, lint with
**ruff**, type-check with **mypy**, and execute the test suite using **pytest**.
