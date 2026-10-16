#!/usr/bin/env python3

import os
import glob

import pandas as pd
import numpy as np
from pathlib import Path

RESULTS_DIR = Path("results")
COMPILED_DIR = RESULTS_DIR / "compiled"


def standardize_model_name(name, filename=""):
    model_name = str(name).lower().strip()
    file_name = str(filename).lower()

    if "mini" in model_name or "mini" in file_name:
        return "GPT-4o-mini"

    if "gpt-4o" in model_name or "gpt4o" in model_name or "gpt4o_full" in file_name:
        return "GPT-4o"

    if "2b" in model_name or "2b" in file_name:
        return "Qwen2-VL-2B"

    if "7b" in model_name or "7b" in file_name or "qwen" in model_name:
        return "Qwen2-VL-7B"

    if "llava" in model_name or "videollava" in file_name:
        return "Video-LLaVA-7B"

    return name


def standardize_modality(modality, filename=""):
    modality_name = str(modality).lower().strip()
    file_name = str(filename).lower()

    if "depth" in modality_name or "depth" in file_name:
        return "Turbo Depth"

    return "RGB"


def load_all_results():
    csv_files = sorted(
        list(RESULTS_DIR.glob("*_results.csv"))
        + list(RESULTS_DIR.glob("*depth_and_rgb.csv"))
    )

    target_files = [
        file_path
        for file_path in csv_files
        if "checkpoint" not in file_path.name
        and "compiled" not in str(file_path)
    ]

    print(f"Found {len(target_files)} evaluation files in {RESULTS_DIR}:")

    for file_path in target_files:
        print(f"  - {file_path.name}")

    all_results = []

    for file_path in target_files:
        try:
            results_df = pd.read_csv(file_path)

            required_columns = [
                "question_id",
                "ground_truth",
                "prediction",
                "is_correct"
            ]

            if not all(
                column in results_df.columns
                for column in required_columns
            ):
                continue

            if "model" not in results_df.columns:
                results_df["model"] = standardize_model_name("", file_path.name)
            else:
                results_df["model"] = results_df["model"].apply(
                    lambda model: standardize_model_name(model, file_path.name)
                )

            if "modality" not in results_df.columns:
                results_df["modality"] = standardize_modality("", file_path.name)
            else:
                results_df["modality"] = results_df["modality"].apply(
                    lambda modality: standardize_modality(modality, file_path.name)
                )

            all_results.append(results_df)

        except Exception as error:
            print(f"Error reading {file_path.name}: {error}")

    if not all_results:
        raise RuntimeError("No valid result files found in results/ directory.")

    master_df = pd.concat(all_results, ignore_index=True)
    master_df = master_df.drop_duplicates(
        subset=["model", "modality", "question_id"]
    )

    return master_df


def compute_metrics(results_df):
    COMPILED_DIR.mkdir(parents=True, exist_ok=True)

    summary_records = []
    model_order = [
        "Video-LLaVA-7B",
        "Qwen2-VL-2B",
        "Qwen2-VL-7B",
        "GPT-4o-mini",
        "GPT-4o"
    ]

    for model in model_order:
        model_results = results_df[results_df["model"] == model]

        if model_results.empty:
            continue

        for modality in ["RGB", "Turbo Depth"]:
            modality_results = model_results[
                model_results["modality"] == modality
            ]

            if modality_results.empty:
                continue

            total = len(modality_results)
            correct = modality_results["is_correct"].sum()
            accuracy = (correct / total) * 100 if total > 0 else 0

            task_accuracy = (
                modality_results
                .groupby("task_type")["is_correct"]
                .mean() * 100
            )

            tracking_accuracy = task_accuracy.get("tracking", 0.0)
            state_accuracy = task_accuracy.get("state", 0.0)
            event_accuracy = task_accuracy.get("event", 0.0)

            summary_records.append({
                "Model": model,
                "Modality": modality,
                "Tracking (%)": round(tracking_accuracy, 2),
                "State (%)": round(state_accuracy, 2),
                "Event (%)": round(event_accuracy, 2),
                "Overall Acc (%)": round(accuracy, 2),
                "Score": f"{correct}/{total}"
            })

    summary_df = pd.DataFrame(summary_records)
    summary_path = COMPILED_DIR / "master_benchmark_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print("\n" + "=" * 80)
    print("MASTER BENCHMARK EVALUATION (Real-4DBench)")
    print("=" * 80)
    print(summary_df.to_string(index=False))

    print("\n" + "=" * 80)
    print("PHYSICAL MODULE DIAGNOSTIC BREAKDOWN (M1-M5)")
    print("=" * 80)

    module_df = (
        results_df
        .groupby(["model", "modality", "module"])["is_correct"]
        .mean()
        .unstack() * 100
    )
    module_df = module_df.round(2)

    print(module_df.to_string())

    module_path = COMPILED_DIR / "module_diagnostic_breakdown.csv"
    module_df.to_csv(module_path)

    print(
        f"\nSaved compiled metrics to:\n"
        f"  - {summary_path}\n"
        f"  - {module_path}"
    )


def main():
    results_df = load_all_results()
    compute_metrics(results_df)


if __name__ == "__main__":
    main()
