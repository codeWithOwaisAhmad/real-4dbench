# Real-4DBench: Spatio-Temporal Physical Occlusion Benchmark for Vision-Language Models

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Benchmark: Real-4DBench](https://img.shields.io/badge/Benchmark-Real--4DBench-green.svg)]()

Official repository for **Real-4DBench**, a physical occlusion diagnostic benchmark evaluating spatio-temporal reasoning, kinematic trajectory prediction, and object permanence across Vision-Language Models (Video-LLMs).

---

## 📌 Key Highlights

- **Physical Laboratory Ground Truth:** 99 real-world dynamic occlusion video sequences captured via a calibrated Intel RealSense D435 stereo-depth sensor.
- **Diagnostic Granularity:** 297 multi-choice question-answer pairs evaluating three orthogonal cognitive axes: **Spatial Tracking**, **Occlusion State**, and **Kinematic Events** across 5 physical interaction modules (Full Occlusion, Trajectory Reversal, Velocity Variation, Partial Occlusion, and Multi-Object Interaction).
- **Modality Ablation (RGB vs. Metric Depth):** Direct isolated benchmarking measuring empirical reasoning gains unlocked by dense metric depth disparity over conventional 2D optical projection.
- **Failure Mode Formalization:** Rigorous identification of the *Vanishing Fallacy* (terminal tracking collapse) across open-weight models and the *Kinematic Momentum Bias* (straight-line assumption under reversal) in open video architectures.

---

## 📊 Comprehensive Benchmark Results

Evaluated across all 99 physical clips (297 QA pairs) under zero-shot prompting conditions:

| Model Architecture | Parameter Scale | Input Modality | Tracking (%) | State (%) | Event (%) | Overall Accuracy | Raw Score | $\Delta$ (Depth Impact) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| *Random Guess* | — | — | 25.00 | 25.00 | 25.00 | 25.00 | 74 / 297 | — |
| **Video-LLaVA** | 7B (Open) | RGB | 23.23 | 0.00 | 55.56 | **26.26** | 78 / 297 | — |
| **Video-LLaVA** | 7B (Open) | Metric Depth | 25.25 | 1.01 | 55.56 | **27.27** | 81 / 297 | **+1.01%** |
| **Qwen2-VL** | 2B (Open) | RGB | 0.00 | 76.77 | 32.32 | **36.36** | 108 / 297 | — |
| **Qwen2-VL** | 2B (Open) | Metric Depth | 0.00 | 79.80 | 47.47 | **42.42** | 126 / 297 | **+6.06%** |
| **Qwen2-VL** | 7B (Open) | RGB | 1.01 | 62.63 | 51.52 | **38.38** | 114 / 297 | — |
| **Qwen2-VL** | 7B (Open) | Metric Depth | 16.16 | 41.41 | 71.72 | **43.10** | 128 / 297 | **+4.72%** |
| **GPT-4o-mini** | Compact API | RGB | 26.26 | 53.54 | 51.52 | **43.77** | 130 / 297 | — |
| **GPT-4o-mini** | Compact API | Metric Depth | 3.03 | 60.61 | 38.38 | **34.01** | 101 / 297 | -9.76% |
| **GPT-4o** | Frontier SOTA | RGB | 38.38 | 83.84 | 89.90 | **70.71** | 210 / 297 | — |
| **GPT-4o** | Frontier SOTA | Metric Depth | 19.19 | 85.86 | 69.70 | **58.25** | 173 / 297 | -12.46% |

---

## 🔬 Core Diagnostic Insights

1. **The Tracking Collapse ("The Vanishing Fallacy"):**
   Across all open-weight models, long-term spatial tracking collapses to or below the random chance baseline ($0.00\%\text{--}25.25\%$). Even frontier GPT-4o drops from **89.90%** on event recognition down to **38.38%** on tracking (and **19.19%** on depth tracking). While models readily infer what event transpires, they fail to maintain object permanence once an entity is occluded.

2. **The Modality Inversion:**
   - **Open-Weight Models (Qwen2-VL, Video-LLaVA):** Metric depth consistently **improves** overall reasoning ($+6.06\%$ on 2B, $+4.72\%$ on 7B), acting as a physical regularizer that removes complex table textures, shadow reflections, and background noise.
   - **Frontier Proprietary Models (GPT-4o, GPT-4o-mini):** Metric depth **degrades** overall reasoning ($-12.46\%$ on GPT-4o, $-9.76\%$ on Mini) due to pre-training distribution mismatch; web-scale models expect natural RGB optical cues, treating distance false-color maps as out-of-domain noise.

3. **Kinematic Momentum Bias:**
   In Module 2 (Trajectory Reversal), Video-LLaVA scores only **8.33%** (far below the 25% random guess rate). The model exhibits a persistent linear momentum prior: when an object enters behind an occluder, it predicts that it exits straight through the other side, failing to infer directional bounce boundaries.

---

## 📂 Repository Structure

```text
real-4dbench/
├── benchmark_qa_297_v2.json      # Calibrated 297 QA pairs across 99 physical sequences
├── idea6_benchmark_log.xlsx      # Calibrated sensor metadata and physical trajectory logs
├── requirements.txt              # Environment dependencies
├── notebooks/                    # GPU evaluation notebooks (Kaggle/Colab)
│   ├── fork-of-videollava-7b-evaluation.ipynb
│   ├── fork-of-qwen2vl-2b-evaluation.ipynb
│   └── fork-of-qwen2vl-7b-evaluation.ipynb
├── results/                      # Raw and compiled evaluation outputs (CSV format)
│   ├── compiled/                 # Master aggregated benchmark metrics
│   │   ├── overall_accuracy.csv
│   │   ├── depth_delta.csv
│   │   ├── accuracy_by_module.csv
│   │   └── accuracy_by_task.csv
│   ├── qwen2vl_2b_rgb_results.csv
│   ├── qwen2vl_2b_depth_results.csv
│   ├── qwen2vl_7b_rgb_results.csv
│   ├── qwen2vl_7b_depth_results.csv
│   ├── videollava_rgb_results.csv
│   ├── videollava_depth_results.csv
│   ├── gpt4o_mini_rgb_results.csv
│   ├── gpt4o_mini_depth_results.csv
│   ├── gpt4o_full_rgb_results.csv
│   └── gpt4o_full_depth_results.csv
└── scripts/                      # Evaluation and compilation pipelines
    ├── evaluate_gpt4o.py         # Frontier GPT-4o evaluation pipeline
    ├── evaluate_gpt4o_mini.py    # Compact GPT-4o-mini evaluation pipeline
    ├── colorize_depth.py         # 16-bit metric depth colormap renderer
    └── compile_results.py        # Metrics aggregation and Wilson CI calculator
