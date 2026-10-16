# Real-4DBench: Spatio-Temporal Physical Occlusion Benchmark for Vision-Language Models

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Benchmark: Real-4DBench](https://img.shields.io/badge/Benchmark-Real--4DBench-green.svg)]()

Official repository for **Real-4DBench**, a physical occlusion diagnostic benchmark evaluating spatio-temporal reasoning, kinematic trajectory prediction, and object permanence across Vision-Language Models (Video-LLMs).

---

## 📌 Key Highlights

- **Physical Laboratory Ground Truth:** 99 real-world dynamic occlusion video sequences captured via a calibrated Intel RealSense D435 stereo-depth sensor.
- **Diagnostic Granularity:** 297 multi-choice question-answer pairs evaluating three orthogonal cognitive axes: **Spatial Tracking**, **Occlusion State**, and **Kinematic Events**.
- **Modality Ablation (RGB vs. RealSense Turbo Depth):** Direct isolated benchmarking measuring empirical reasoning gains unlocked by dense metric depth disparity over conventional 2D optical projection.
- **Failure Mode Formalization:** Rigorous identification of the *Vanishing Fallacy* in compact models and the *Kinematic Momentum Bias* in open video architectures.

---

## 📊 Comprehensive Benchmark Results

Evaluated across all 99 physical clips (297 QA pairs) under zero-shot prompting conditions:

| Model Architecture | Parameter Scale | Input Modality | Tracking (%) | State (%) | Event (%) | Overall Accuracy | Raw Score | $\Delta$ (Depth Impact) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| *Random Guess* | — | — | 25.00 | 25.00 | 25.00 | 25.00 | 74 / 297 | — |
| **Video-LLaVA** | 7B (Open) | RGB | 18.18 | 20.20 | 79.80 | 39.39 | 117 / 297 | — |
| **Video-LLaVA** | 7B (Open) | Turbo Depth | **34.34** | 20.20 | 76.77 | **43.77** | 130 / 297 | **+4.38%** |
| **Qwen2-VL** | 2B (Open) | RGB | 0.00 | 77.78 | 15.15 | 30.98 | 92 / 297 | — |
| **Qwen2-VL** | 2B (Open) | Turbo Depth | 0.00 | 79.80 | 20.20 | 33.33 | 99 / 297 | **+2.35%** |
| **Qwen2-VL** | 7B (Open) | RGB | 1.01 | 66.67 | 64.65 | 44.11 | 131 / 297 | — |
| **Qwen2-VL** | 7B (Open) | Turbo Depth | **6.06** | **79.80** | **80.81** | **55.56** | 165 / 297 | **+11.45%** |
| **GPT-4o-mini** | Compact API | RGB | 30.30 | 52.53 | 42.42 | 41.75 | 124 / 297 | — |
| **GPT-4o-mini** | Compact API | Turbo Depth | **33.33** | 52.53 | 42.42 | **42.76** | 127 / 297 | **+1.01%** |
| **GPT-4o** | Frontier SOTA | RGB | **56.57** | **81.82** | **95.96** | **78.11** | 232 / 297 | — |
| **GPT-4o** | Frontier SOTA | Turbo Depth | 54.55 | 76.77 | **95.96** | 75.76 | 225 / 297 | -2.35% |

---

## 📂 Repository Structure

```text
real-4dbench/
├── benchmark_qa_297.json         # 297 QA pairs across 99 physical sequences
├── idea6_benchmark_log.xlsx     # Calibrated sensor metadata and trajectory logs
├── requirements.txt             # Clean environment dependencies
├── notebooks/                   # Reproducible GPU evaluation notebooks (Kaggle/Colab)
│   ├── videollava_7b_evaluation.ipynb
│   ├── qwen2vl_2b_evaluation.ipynb
│   └── qwen2vl_7b_evaluation.ipynb
├── results/                     # Metric evaluation outputs (CSV format)
│   ├── videollava_rgb_results.csv
│   ├── videollava_depth_results.csv
│   ├── qwen2vl_2b_rgb_results.csv
│   ├── qwen2vl_2b_depth_results.csv
│   ├── qwen2vl_7b_depth_and_rgb.csv
│   ├── gpt4o_mini_rgb_results.csv
│   ├── gpt4o_mini_depth_results.csv
│   ├── gpt4o_full_rgb_results.csv
│   └── gpt4o_full_depth_results.csv
└── scripts/                     # Standalone execution pipelines
    ├── evaluate_gpt4o.py        # Flagship GPT-4o evaluation
    ├── evaluate_gpt4o_mini.py   # Lightweight GPT-4o-mini evaluation
    ├── colorize_depth.py        # 16-bit metric depth to Turbo colormap converter
    └── compile_results.py       # Metrics aggregation and diagnostic table generator
