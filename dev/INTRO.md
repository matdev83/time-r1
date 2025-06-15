# 🚀 Time-R1 Project: Agent Onboarding and Task Dispatch

Welcome, Coding Agent! You are about to contribute to the **Time-R1** project, a cutting-edge machine learning initiative focused on time series forecasting using a novel slow-thinking approach with reinforced Large Language Models (LLMs).

## 🎯 Project Goal

The primary objective is to build a complete, reproducible Python project that trains, evaluates, and serves the **Time-R1** foundation model. This model is designed to enhance multi-step reasoning ability of LLMs for time series forecasting, leveraging a two-stage reinforcement fine-tuning framework (SFT Warm-Up followed by GRIP RL).

## 📚 Onboarding Essentials: Your First Steps

To ensure a smooth start and effective contribution, please follow this onboarding sequence:

1. **Read the Project `README.md`**: Begin by thoroughly reviewing the main `README.md` file in the repository root. This provides a high-level overview of the project, its purpose, and initial setup instructions.
2. **Understand the Agent Guidelines**: Next, read `AGENTS.md`. This document outlines the project's structure, describes the roles of different modules and directories, and details key software development principles that you, as a coding agent, must strictly adhere to.

## 🗺️ Where to Find More Details

Once you have completed the initial onboarding, you can dive deeper into the project's specifics:

* **Implementation Plan (`dev/PLAN.md`)**: This file contains the comprehensive, state-of-the-art implementation plan for the entire ML workflow. It breaks down tasks into sub-tasks, specifies inputs, deliverables, tests, and acceptance criteria, and includes detailed implementation notes derived directly from the academic paper. **This is your primary guide for task execution.**
* **Academic Paper (`dev/time_r1_paper/time_r1_paper.md`)**: For the foundational theoretical background, detailed algorithms, mathematical formulations, and experimental results, refer to the `time_r1_paper.md`. This paper provides the scientific basis for the Time-R1 framework.

## 🛠️ Coding Standards & Best Practices

Adherence to high-quality coding standards and software engineering principles is paramount for this project. You are expected to follow:

* **PEP 8**: Python Enhancement Proposal 8 for code style.
* **Test-Driven Development (TDD)**: Write tests before writing the code they are meant to test. Ensure high test coverage.
* **SOLID Principles**:
  * **S**ingle Responsibility Principle
  * **O**pen/Closed Principle
  * **L**iskov Substitution Principle
  * **I**nterface Segregation Principle
  * **D**ependency Inversion Principle
* **DRY (Don't Repeat Yourself)**: Avoid duplication of code and logic.
* **KISS (Keep It Simple, Stupid)**: Favor simplicity and clarity in your designs and implementations.

**Tooling for Code Quality**: The project utilizes automated tools to enforce these standards:

* `black` (code formatter)
* `isort` (import sorter)
* `ruff` (fast Python linter)
* `mypy` (static type checker)

Your contributions will be evaluated against these standards and the acceptance criteria defined in `dev/PLAN.md`.

## Current Task

**Your current task is:** 

If the task involved creating or modifying any code, you are required to create related tests and ensure they pass.

**Ensure no regressions**: Run the test suite to ensure that all tests pass. You may not introduce any regressions to the existing codebase.
