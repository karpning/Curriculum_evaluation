# Curriculum_evaluation

This repository contains code and data for our study on **metric-driven curriculum learning (CL)** in large language models (LLMs) for mathematical reasoning.  
It provides training, evaluation, and metric estimation pipelines supporting reproducible experiments across multiple CL strategies and benchmark datasets.

---

## 📂 Project Structure

### 1. `dataset/`
This folder contains the datasets used in our experiments.

- **` dataset/MetaMathQA/`**  
  Contains the **20K original training samples** derived from the MetaMathQA dataset.  
  These samples are used as the training corpus for curriculum learning experiments.

- **`dataset/test_base/DataForTest/`**  
  Includes all **evaluation datasets** used in our experiments:
  - **ASDiv**
  - **GSM8K**
  - **MathBench**
  - **MetaMathQA (MMQA)**
  - **MATH**

  Each test set contains **512 mathematical problems** used for consistent evaluation across models and CL strategies.

---

### 2. `metrics/`
This directory contains scripts for metric estimation and curriculum construction.

- **`metrics/estimate_metric/`**  
  Includes multiple Python scripts for computing both **model-side metrics** (e.g., confidence, surprisal, uncertainty) and **problem-side metrics** (e.g., accuracy-based difficulty).

- **`metrics/SplitByMetric/`**  
  Contains scripts implementing various **curriculum sorting strategies** based on metric values.  
  These scripts generate and save reordered training datasets corresponding to different CL directions and difficulty definitions.

---

### 3. `train_base/`
Contains training scripts for the three open-source base models used in our experiments:
- **Llama3-8B**
- **Mistral-7B**
- **Gemma3-4B**

Each script supports metric-based curriculum schedules and consistent hyperparameter settings.

---

### 4. `test_base/`
Includes evaluation scripts for all **mathematical reasoning benchmarks**, ensuring consistent testing across different datasets and model configurations.

---

## ⚙️ Reproducibility

Detailed experimental settings are described in the paper (see Section `Experiment Details`).  
All code, configuration files, and example scripts are designed for reproducible execution.

---


