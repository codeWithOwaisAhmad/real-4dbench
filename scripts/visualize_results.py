import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os


# ── Plot Style ────────────────────────────────────────────────────────────────
COLORS = {
    "gemini-2.0-flash": "#4285F4",
    "gpt-4o": "#10A37F",
    "video-llava": "#FF6B35",
    "internvideo2": "#9B59B6"
}
CONDITION_COLORS = {
    "rgb": "#2196F3",
    "rgb_depth": "#FF5722"
}


def setup_plot_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150
    })


# ── Plot 1: Main Result Bar Chart ─────────────────────────────────────────────
def plot_overall_accuracy(overall_df, output_path="results/compiled/fig1_overall_accuracy.png"):
    """
    Main result figure for the paper.
    Grouped bar chart — models on x-axis, RGB vs RGB+Depth as bar groups.
    """
    setup_plot_style()

    models = overall_df["model"].unique()
    conditions = ["rgb", "rgb_depth"]
    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, condition in enumerate(conditions):
        cond_df = overall_df[overall_df["condition"] == condition].set_index("model")
        accuracies = []
        errors_low = []
        errors_high = []

        for model in models:
            if model in cond_df.index:
                acc = cond_df.loc[model, "accuracy"]
                ci_low = cond_df.loc[model, "ci_lower"]
                ci_high = cond_df.loc[model, "ci_upper"]
                accuracies.append(acc)
                errors_low.append(acc - ci_low)
                errors_high.append(ci_high - acc)
            else:
                accuracies.append(0)
                errors_low.append(0)
                errors_high.append(0)

        bars = ax.bar(
            x + i * width,
            accuracies,
            width,
            label=f"RGB{'+ Depth' if condition == 'rgb_depth' else ' Only'}",
            color=CONDITION_COLORS[condition],
            alpha=0.85,
            yerr=[errors_low, errors_high],
            capsize=4,
            error_kw={"linewidth": 1.5}
        )

        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            if acc > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1,
                    f"{acc:.1f}%",
                    ha="center", va="bottom",
                    fontsize=9, fontweight="bold"
                )

    # Add random baseline
    ax.axhline(y=25, color="gray", linestyle="--",
               linewidth=1, alpha=0.7, label="Random (25%)")

    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Model Accuracy on Real-4DBench\nRGB Only vs RGB + Depth Colormap",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylim(0, 105)
    ax.legend(loc="upper right")
    ax.yaxis.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


# ── Plot 2: Accuracy by Scenario Type ────────────────────────────────────────
def plot_scenario_breakdown(scenario_df,
                             output_path="results/compiled/fig2_scenario_breakdown.png"):
    """
    Secondary result figure.
    Shows accuracy per scenario type per model — RGB only condition.
    """
    setup_plot_style()

    rgb_df = scenario_df[scenario_df["condition"] == "rgb"]
    models = rgb_df["model"].unique()
    scenarios = rgb_df["scenario_type"].unique()

    x = np.arange(len(scenarios))
    width = 0.8 / len(models)

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, model in enumerate(models):
        model_df = rgb_df[rgb_df["model"] == model].set_index("scenario_type")
        accuracies = []
        for scenario in scenarios:
            if scenario in model_df.index:
                accuracies.append(model_df.loc[scenario, "accuracy"])
            else:
                accuracies.append(0)

        color = COLORS.get(model, f"C{i}")
        bars = ax.bar(
            x + i * width,
            accuracies,
            width,
            label=model,
            color=color,
            alpha=0.85
        )

        for bar, acc in zip(bars, accuracies):
            if acc > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    f"{acc:.0f}%",
                    ha="center", va="bottom",
                    fontsize=8
                )

    ax.axhline(y=25, color="gray", linestyle="--",
               linewidth=1, alpha=0.7, label="Random (25%)")

    ax.set_xlabel("Scenario Type", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Accuracy by Occlusion Scenario Type (RGB Only)",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(
        [s.replace("_", " ").title() for s in scenarios],
        rotation=15, ha="right"
    )
    ax.set_ylim(0, 105)
    ax.legend(loc="upper right")
    ax.yaxis.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


# ── Plot 3: Depth Delta ───────────────────────────────────────────────────────
def plot_depth_delta(depth_delta_df,
                     output_path="results/compiled/fig3_depth_delta.png"):
    """
    Shows how much depth colormap helps or hurts each model.
    Positive = depth helps. Negative = depth hurts.
    """
    setup_plot_style()

    models = depth_delta_df["model"].tolist()
    deltas = depth_delta_df["delta"].tolist()
    colors = ["#4CAF50" if d > 0 else "#F44336" for d in deltas]

    fig, ax = plt.subplots(figsize=(8, 5))

    bars = ax.bar(models, deltas, color=colors, alpha=0.85, width=0.5)

    for bar, delta in zip(bars, deltas):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + (0.2 if delta >= 0 else -0.8),
            f"{delta:+.1f}%",
            ha="center", va="bottom",
            fontsize=10, fontweight="bold"
        )

    ax.axhline(y=0, color="black", linewidth=1)
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Accuracy Change (%)", fontsize=12)
    ax.set_title("Effect of Depth Colormap on Model Accuracy\n(RGB+Depth) − (RGB Only)",
                 fontsize=13, fontweight="bold")
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.yaxis.grid(True, alpha=0.3)

    green_patch = mpatches.Patch(color="#4CAF50", alpha=0.85, label="Depth helps")
    red_patch = mpatches.Patch(color="#F44336", alpha=0.85, label="Depth hurts")
    ax.legend(handles=[green_patch, red_patch])

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Load compiled results
    overall_path = "results/compiled/overall_accuracy.csv"
    scenario_path = "results/compiled/accuracy_by_scenario.csv"
    depth_path = "results/compiled/depth_delta.csv"

    if not os.path.exists(overall_path):
        print("Run compile_results.py first to generate the compiled CSVs")
        exit(1)

    overall_df = pd.read_csv(overall_path)
    scenario_df = pd.read_csv(scenario_path)

    plot_overall_accuracy(overall_df)
    plot_scenario_breakdown(scenario_df)

    if os.path.exists(depth_path):
        depth_df = pd.read_csv(depth_path)
        plot_depth_delta(depth_df)

    print("\nAll figures saved to results/compiled/")
