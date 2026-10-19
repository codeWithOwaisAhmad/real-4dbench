import os
import glob
import json
import numpy as np
import pandas as pd
from scipy import stats


# ── Wilson Score Confidence Interval ─────────────────────────────────────────
def wilson_confidence_interval(correct, total, confidence=0.95):
    """
    Wilson score interval for proportions.
    Returns (lower, upper) as percentages rounded to 1 decimal place.
    """
    if total == 0:
        return (0.0, 0.0)

    z = stats.norm.ppf((1 + confidence) / 2)
    p = correct / total
    denominator = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denominator
    margin = (z * np.sqrt(p * (1 - p) / total + z**2 / (4 * total**2))) / denominator

    lower = max(0, (center - margin) * 100)
    upper = min(100, (center + margin) * 100)
    return (round(lower, 1), round(upper, 1))


# ── Load All Real Results ────────────────────────────────────────────────────
def load_all_results(results_dir="results"):
    """
    Loads all final model result CSVs from results_dir while ignoring checkpoints.
    """
    if not os.path.exists(results_dir):
        print(f"Directory '{results_dir}' not found.")
        return None

    # Load only final result CSVs, skip intermediate checkpoints
    all_files = [
        os.path.join(results_dir, f)
        for f in os.listdir(results_dir)
        if f.endswith(".csv") and "checkpoint" not in f and "summary" not in f
    ]

    if not all_files:
        print(f"No result CSV files found in {results_dir}")
        return None

    all_dfs = []
    print(f"Found {len(all_files)} result files:")
    for filepath in sorted(all_files):
        df = pd.read_csv(filepath)
        filename = os.path.basename(filepath)
        print(f"  • Loaded: {filename:<35} | Rows: {len(df)}")
        
        # Standardize column naming if variations exist
        if "condition" in df.columns and "modality" not in df.columns:
            df["modality"] = df["condition"]
        if "question_type" in df.columns and "task_type" not in df.columns:
            df["task_type"] = df["question_type"]
        if "scenario_type" in df.columns and "scenario" not in df.columns:
            df["scenario"] = df["scenario_type"]

        all_dfs.append(df)

    combined = pd.concat(all_dfs, ignore_index=True)
    
    # Normalize model identifiers for clean reporting
    model_mapping = {
        "qwen2vl_2b": "Qwen2-VL-2B",
        "qwen2vl-2b": "Qwen2-VL-2B",
        "qwen2vl_7b": "Qwen2-VL-7B",
        "qwen2vl-7b": "Qwen2-VL-7B",
        "videollava": "Video-LLaVA-7B",
        "videollava-7b": "Video-LLaVA-7B",
        "gpt-4o-mini": "GPT-4o-mini",
        "gpt_4o_mini": "GPT-4o-mini",
        "gpt-4o": "GPT-4o",
        "gpt_4o": "GPT-4o"
    }
    combined["model"] = combined["model"].replace(model_mapping)

    print(f"\nTotal evaluated questions loaded across all conditions: {len(combined)}")
    return combined


# ── Analysis 1: Overall Accuracy ─────────────────────────────────────────────
def compute_overall_accuracy(df):
    results = []
    for (model, modality), group in df.groupby(["model", "modality"]):
        total = len(group)
        correct = int(group["is_correct"].sum())
        accuracy = (correct / total) * 100
        ci_lower, ci_upper = wilson_confidence_interval(correct, total)

        results.append({
            "model": model,
            "modality": modality,
            "total_questions": total,
            "correct": correct,
            "accuracy (%)": round(accuracy, 2),
            "ci_95%": f"[{ci_lower}%, {ci_upper}%]"
        })

    return pd.DataFrame(results).sort_values(["model", "modality"])


# ── Analysis 2: Accuracy by Physical Module ──────────────────────────────────
def compute_accuracy_by_module(df):
    results = []
    for (model, modality, module), group in df.groupby(["model", "modality", "module"]):
        total = len(group)
        correct = int(group["is_correct"].sum())
        accuracy = (correct / total) * 100
        ci_lower, ci_upper = wilson_confidence_interval(correct, total)
        scenario_name = group["scenario"].iloc[0] if "scenario" in group.columns else ""

        results.append({
            "model": model,
            "modality": modality,
            "module": module,
            "scenario": scenario_name,
            "total_questions": total,
            "correct": correct,
            "accuracy (%)": round(accuracy, 2),
            "ci_95%": f"[{ci_lower}%, {ci_upper}%]"
        })

    return pd.DataFrame(results).sort_values(["model", "modality", "module"])


# ── Analysis 3: Accuracy by Task Type ────────────────────────────────────────
def compute_accuracy_by_task(df):
    results = []
    for (model, modality, task), group in df.groupby(["model", "modality", "task_type"]):
        total = len(group)
        correct = int(group["is_correct"].sum())
        accuracy = (correct / total) * 100
        ci_lower, ci_upper = wilson_confidence_interval(correct, total)

        results.append({
            "model": model,
            "modality": modality,
            "task_type": task,
            "total_questions": total,
            "correct": correct,
            "accuracy (%)": round(accuracy, 2),
            "ci_95%": f"[{ci_lower}%, {ci_upper}%]"
        })

    return pd.DataFrame(results).sort_values(["model", "modality", "task_type"])


# ── Analysis 4: RGB vs Depth Delta ───────────────────────────────────────────
def compute_depth_delta(overall_df):
    rgb = overall_df[overall_df["modality"] == "rgb"].set_index("model")
    depth = overall_df[overall_df["modality"] == "depth"].set_index("model")

    results = []
    for model in rgb.index:
        if model in depth.index:
            rgb_acc = rgb.loc[model, "accuracy (%)"]
            depth_acc = depth.loc[model, "accuracy (%)"]
            delta = depth_acc - rgb_acc

            results.append({
                "model": model,
                "rgb_accuracy (%)": rgb_acc,
                "depth_accuracy (%)": depth_acc,
                "delta (Depth - RGB)": round(delta, 2),
                "rgb_ci": rgb.loc[model, "ci_95%"],
                "depth_ci": depth.loc[model, "ci_95%"],
                "interpretation": "Depth improves" if delta > 0 else "Depth degrades" if delta < 0 else "Equal"
            })

    return pd.DataFrame(results)


# ── Save and Display ─────────────────────────────────────────────────────────
def compile_and_save(results_dir="results", output_dir="results/compiled"):
    os.makedirs(output_dir, exist_ok=True)
    df = load_all_results(results_dir)
    if df is None or len(df) == 0:
        return

    # 1. Overall
    overall = compute_overall_accuracy(df)
    overall.to_csv(f"{output_dir}/overall_accuracy.csv", index=False)
    print("\n" + "="*85)
    print("TABLE 1: OVERALL BENCHMARK ACCURACY (Random Baseline: 25.00%)")
    print("="*85)
    print(overall.to_string(index=False))

    # 2. Modules M1 - M5
    by_module = compute_accuracy_by_module(df)
    by_module.to_csv(f"{output_dir}/accuracy_by_module.csv", index=False)
    print("\n" + "="*85)
    print("TABLE 2: ACCURACY BY PHYSICAL REASONING MODULE (M1 - M5)")
    print("="*85)
    # Pivot for clean paper view
    pivot_module = by_module.pivot_table(
        index=["model", "modality"],
        columns="module",
        values="accuracy (%)"
    ).reset_index()
    print(pivot_module.to_string(index=False))

    # 3. Task Type (Tracking, State, Event)
    by_task = compute_accuracy_by_task(df)
    by_task.to_csv(f"{output_dir}/accuracy_by_task.csv", index=False)
    print("\n" + "="*85)
    print("TABLE 3: ACCURACY BY TASK TYPE (Tracking, State, Event)")
    print("="*85)
    pivot_task = by_task.pivot_table(
        index=["model", "modality"],
        columns="task_type",
        values="accuracy (%)"
    ).reset_index()
    print(pivot_task.to_string(index=False))

    # 4. Modality Delta
    delta = compute_depth_delta(overall)
    if len(delta) > 0:
        delta.to_csv(f"{output_dir}/depth_delta.csv", index=False)
        print("\n" + "="*85)
        print("TABLE 4: DEPTH CONTRIBUTION (Depth vs. RGB Delta)")
        print("="*85)
        print(delta.to_string(index=False))

    print(f"\nAll compiled summary tables saved to: {output_dir}/")


if __name__ == "__main__":
    compile_and_save(results_dir="results", output_dir="results/compiled")