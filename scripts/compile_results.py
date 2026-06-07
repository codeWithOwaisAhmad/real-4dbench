import pandas as pd
import numpy as np
import os
import json
from scipy import stats


# ── Wilson Score Confidence Interval ─────────────────────────────────────────
def wilson_confidence_interval(correct, total, confidence=0.95):
    """
    Wilson score interval for proportions.
    More accurate than normal approximation for small samples.
    Returns (lower, upper) as percentages.
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


# ── Load All Results ──────────────────────────────────────────────────────────
def load_all_results(results_dir="results/raw_outputs"):
    """
    Load all CSV result files from raw_outputs directory.
    Each file is one model + one condition.
    """
    all_dfs = []
    csv_files = [f for f in os.listdir(results_dir) if f.endswith('.csv')]

    if not csv_files:
        print(f"No CSV files found in {results_dir}")
        return None

    for filename in csv_files:
        filepath = os.path.join(results_dir, filename)
        df = pd.read_csv(filepath)
        print(f"Loaded: {filename} — {len(df)} rows")
        all_dfs.append(df)

    combined = pd.concat(all_dfs, ignore_index=True)
    print(f"\nTotal rows loaded: {len(combined)}")
    return combined


# ── Analysis 1: Overall Accuracy Table ───────────────────────────────────────
def compute_overall_accuracy(df):
    """
    Primary Analysis 1 — Overall accuracy per model per condition.
    This is your main result table in the paper.
    """
    results = []

    for (model, condition), group in df.groupby(["model", "condition"]):
        total = len(group)
        correct = group["is_correct"].sum()
        accuracy = (correct / total) * 100
        ci_lower, ci_upper = wilson_confidence_interval(correct, total)

        results.append({
            "model": model,
            "condition": condition,
            "total_questions": total,
            "correct": correct,
            "accuracy": round(accuracy, 1),
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "ci_string": f"[{ci_lower}%, {ci_upper}%]"
        })

    return pd.DataFrame(results).sort_values(["model", "condition"])


# ── Analysis 2: Accuracy by Occlusion Type ───────────────────────────────────
def compute_accuracy_by_scenario(df):
    """
    Primary Analysis 2 — Accuracy broken down by scenario type.
    Only report this if you have at least 15 clips per scenario type.
    """
    results = []

    for (model, condition, scenario), group in df.groupby(
            ["model", "condition", "scenario_type"]):
        total = len(group)
        correct = group["is_correct"].sum()
        accuracy = (correct / total) * 100
        ci_lower, ci_upper = wilson_confidence_interval(correct, total)

        results.append({
            "model": model,
            "condition": condition,
            "scenario_type": scenario,
            "total_questions": total,
            "correct": correct,
            "accuracy": round(accuracy, 1),
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "ci_string": f"[{ci_lower}%, {ci_upper}%]"
        })

    return pd.DataFrame(results).sort_values(["model", "condition", "scenario_type"])


# ── Analysis 3: Accuracy by Question Type (Exploratory) ──────────────────────
def compute_accuracy_by_question_type(df):
    """
    Exploratory analysis — accuracy by question type.
    Label clearly as exploratory in paper. Small sample sizes.
    """
    results = []

    for (model, condition, qtype), group in df.groupby(
            ["model", "condition", "question_type"]):
        total = len(group)
        correct = group["is_correct"].sum()
        accuracy = (correct / total) * 100
        ci_lower, ci_upper = wilson_confidence_interval(correct, total)

        results.append({
            "model": model,
            "condition": condition,
            "question_type": qtype,
            "total_questions": total,
            "correct": correct,
            "accuracy": round(accuracy, 1),
            "ci_lower": ci_lower,
            "ci_upper": ci_upper
        })

    return pd.DataFrame(results).sort_values(["model", "condition", "question_type"])


# ── RGB vs RGB+Depth Comparison ───────────────────────────────────────────────
def compute_depth_delta(overall_df):
    """
    Compute the difference between RGB and RGB+Depth conditions.
    This is your depth contribution analysis.
    If confidence intervals overlap — you cannot claim depth helps.
    """
    rgb = overall_df[overall_df["condition"] == "rgb"].set_index("model")
    depth = overall_df[overall_df["condition"] == "rgb_depth"].set_index("model")

    results = []
    for model in rgb.index:
        if model in depth.index:
            rgb_acc = rgb.loc[model, "accuracy"]
            depth_acc = depth.loc[model, "accuracy"]
            delta = depth_acc - rgb_acc
            rgb_ci = rgb.loc[model, "ci_string"]
            depth_ci = depth.loc[model, "ci_string"]

            results.append({
                "model": model,
                "rgb_accuracy": rgb_acc,
                "rgb_depth_accuracy": depth_acc,
                "delta": round(delta, 1),
                "rgb_ci": rgb_ci,
                "depth_ci": depth_ci,
                "interpretation": "depth helps" if delta > 0 else "depth hurts" if delta < 0 else "no difference"
            })

    return pd.DataFrame(results)


# ── Save All Tables ───────────────────────────────────────────────────────────
def save_compiled_results(df, output_dir="results/compiled"):
    os.makedirs(output_dir, exist_ok=True)

    # Overall accuracy
    overall = compute_overall_accuracy(df)
    overall.to_csv(f"{output_dir}/overall_accuracy.csv", index=False)
    print("\n── Overall Accuracy ──────────────────────────────")
    print(overall.to_string(index=False))

    # Scenario breakdown
    scenario = compute_accuracy_by_scenario(df)
    scenario.to_csv(f"{output_dir}/accuracy_by_scenario.csv", index=False)
    print("\n── Accuracy by Scenario ──────────────────────────")
    print(scenario.to_string(index=False))

    # Question type breakdown
    qtype = compute_accuracy_by_question_type(df)
    qtype.to_csv(f"{output_dir}/accuracy_by_question_type.csv", index=False)
    print("\n── Accuracy by Question Type (Exploratory) ───────")
    print(qtype.to_string(index=False))

    # Depth delta
    depth_delta = compute_depth_delta(overall)
    if len(depth_delta) > 0:
        depth_delta.to_csv(f"{output_dir}/depth_delta.csv", index=False)
        print("\n── RGB vs RGB+Depth Delta ────────────────────────")
        print(depth_delta.to_string(index=False))

    print(f"\nAll tables saved to {output_dir}/")
    return overall, scenario, qtype


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Generate dummy results to test compiler
    print("Generating dummy results for testing...")

    models = ["gemini-2.0-flash", "gpt-4o"]
    conditions = ["rgb", "rgb_depth"]
    scenarios = ["full_occlusion", "partial_occlusion", "fast_movement",
                 "slow_movement", "multiple_objects"]
    question_types = ["tracking", "event", "state"]
    objects = ["ball", "bottle", "mug", "book", "apple"]

    dummy_rows = []
    clip_num = 0

    for model in models:
        for condition in conditions:
            for scenario in scenarios:
                for i in range(15):  # 15 questions per scenario
                    clip_num += 1
                    correct_answer = ["A", "B", "C", "D"][i % 4]

                    # Simulate ~60% accuracy for rgb, ~62% for rgb_depth
                    if condition == "rgb":
                        model_answer = correct_answer if np.random.random() < 0.60 else "B"
                    else:
                        model_answer = correct_answer if np.random.random() < 0.62 else "B"

                    dummy_rows.append({
                        "clip_id": f"clip_{clip_num:03d}",
                        "question_id": f"clip_{clip_num:03d}_q1",
                        "question_type": question_types[i % 3],
                        "question": "Dummy question",
                        "correct_answer": correct_answer,
                        "model_answer": model_answer,
                        "is_correct": model_answer == correct_answer,
                        "condition": condition,
                        "model": model,
                        "object_type": objects[i % 5],
                        "occluder_type": "cardboard_box",
                        "scenario_type": scenario
                    })

    dummy_df = pd.DataFrame(dummy_rows)
    os.makedirs("results/raw_outputs", exist_ok=True)
    dummy_df.to_csv("results/raw_outputs/dummy_combined.csv", index=False)
    print(f"Dummy data: {len(dummy_df)} rows\n")

    # Run compiler
    df = load_all_results("results/raw_outputs")
    save_compiled_results(df)
