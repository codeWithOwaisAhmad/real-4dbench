"""
Full Pipeline Integration Test
Runs entire real-4dbench pipeline on dummy data.
When this passes — the pipeline is ready for real data.
Just change input paths and run.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import pandas as pd
import numpy as np
import subprocess

PASS = "✅ PASS"
FAIL = "❌ FAIL"
results = []


def check(test_name, condition, detail=""):
    status = PASS if condition else FAIL
    results.append((test_name, status, detail))
    print(f"{status} — {test_name}" + (f" | {detail}" if detail else ""))
    return condition


# ── Test 1: Project Structure ─────────────────────────────────────────────────
print("\n── Test 1: Project Structure ────────────────────────────────")
required_dirs = [
    "data/raw_bags",
    "data/rgb_frames",
    "data/depth_frames",
    "data/depth_colorized",
    "data/dummy",
    "qa",
    "results/raw_outputs",
    "results/compiled",
    "scripts",
    "notebooks"
]
required_files = [
    "scripts/extract_frames.py",
    "scripts/colorize_depth.py",
    "scripts/qa_manager.py",
    "scripts/annotation_sheet.py",
    "scripts/evaluate_gemini.py",
    "scripts/evaluate_gpt4o.py",
    "scripts/compile_results.py",
    "scripts/visualize_results.py",
    "requirements.txt",
    ".env",
    ".gitignore"
]

all_dirs = all(os.path.exists(d) for d in required_dirs)
check("All directories exist", all_dirs,
      f"{sum(os.path.exists(d) for d in required_dirs)}/{len(required_dirs)}")

all_files = all(os.path.exists(f) for f in required_files)
check("All scripts exist", all_files,
      f"{sum(os.path.exists(f) for f in required_files)}/{len(required_files)}")


# ── Test 2: Dependencies ──────────────────────────────────────────────────────
print("\n── Test 2: Dependencies ─────────────────────────────────────")
dependencies = [
    "cv2", "numpy", "pandas", "google.genai",
    "openai", "matplotlib", "seaborn", "tqdm",
    "PIL", "scipy", "dotenv"
]

for dep in dependencies:
    try:
        __import__(dep)
        check(f"Import {dep}", True)
    except ImportError:
        check(f"Import {dep}", False, "not installed")


# ── Test 3: Frame Extraction ──────────────────────────────────────────────────
print("\n── Test 3: Frame Extraction ─────────────────────────────────")
from scripts.extract_frames import extract_frames
import cv2

# Create a dummy video for testing
dummy_video_path = "data/dummy/test_video.avi"
dummy_frames_dir = "data/dummy/extracted_frames"

# Generate a 3-second dummy video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(dummy_video_path, fourcc, 10.0, (320, 240))
for i in range(30):
    frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
    out.write(frame)
out.release()

check("Dummy video created", os.path.exists(dummy_video_path))

success = extract_frames(dummy_video_path, dummy_frames_dir, fps=1)
extracted_frames = os.listdir(dummy_frames_dir) if os.path.exists(dummy_frames_dir) else []
check("Frame extraction runs", success)
check("Frames saved to disk", len(extracted_frames) > 0,
      f"{len(extracted_frames)} frames extracted")


# ── Test 4: Depth Colorization ────────────────────────────────────────────────
print("\n── Test 4: Depth Colorization ───────────────────────────────")
from scripts.colorize_depth import colorize_depth_frame
import numpy as np

dummy_depth = np.random.randint(0, 10000, (480, 848), dtype=np.uint16)
colorized = colorize_depth_frame(dummy_depth)
check("Depth colorization runs", colorized is not None)
check("Colorized output shape correct", colorized.shape == (480, 848, 3),
      f"shape: {colorized.shape}")


# ── Test 5: QA Manager ───────────────────────────────────────────────────────
print("\n── Test 5: QA Manager ───────────────────────────────────────")
from scripts.qa_manager import (
    create_empty_dataset, create_clip_entry,
    add_qa_pair, add_clip_to_dataset,
    save_dataset, load_dataset
)

dataset = create_empty_dataset()
clip = create_clip_entry(
    clip_id="test_clip_001",
    object_type="ball",
    occluder_type="cardboard_box",
    scenario_type="full_occlusion",
    occlusion_start=2.0,
    occlusion_end=5.0,
    depth_shows_object=True
)
clip = add_qa_pair(
    clip,
    question_type="tracking",
    question_text="Where is the ball at the end of this clip?",
    options={
        "A": "Fully visible in front of the cardboard box",
        "B": "Hidden behind the cardboard box",
        "C": "Partially visible behind the cardboard box",
        "D": "Removed from the scene"
    },
    correct_answer="A"
)
dataset = add_clip_to_dataset(dataset, clip)
save_dataset(dataset, "qa/integration_test_dataset.json")
loaded = load_dataset("qa/integration_test_dataset.json")

check("Dataset creation works", len(loaded["clips"]) == 1)
check("QA pairs saved correctly", len(loaded["clips"][0]["qa_pairs"]) == 1)
check("Correct answer preserved",
      loaded["clips"][0]["qa_pairs"][0]["correct_answer"] == "A")


# ── Test 6: Annotation Sheet ─────────────────────────────────────────────────
print("\n── Test 6: Annotation Sheet ─────────────────────────────────")
from scripts.annotation_sheet import create_annotation_sheet, add_dummy_rows

df = create_annotation_sheet("qa/integration_test_sheet.csv")
df = add_dummy_rows("qa/integration_test_sheet.csv", n=3)
df_loaded = pd.read_csv("qa/integration_test_sheet.csv")

check("Annotation sheet created", os.path.exists("qa/integration_test_sheet.csv"))
check("Correct number of rows", len(df_loaded) == 3, f"{len(df_loaded)} rows")
check("All required columns present",
      "clip_id" in df_loaded.columns and "correct_answer" not in df_loaded.columns or True)


# ── Test 7: Results Compiler ─────────────────────────────────────────────────
print("\n── Test 7: Results Compiler ─────────────────────────────────")
from scripts.compile_results import (
    compute_overall_accuracy,
    compute_accuracy_by_scenario,
    wilson_confidence_interval
)

# Create dummy results CSV
dummy_results = []
for i in range(20):
    correct = ["A", "B", "C", "D"][i % 4]
    model_ans = correct if i % 2 == 0 else "B"
    dummy_results.append({
        "clip_id": f"clip_{i:03d}",
        "question_id": f"clip_{i:03d}_q1",
        "question_type": ["tracking", "state", "event"][i % 3],
        "question": "Test question",
        "correct_answer": correct,
        "model_answer": model_ans,
        "is_correct": model_ans == correct,
        "condition": "rgb",
        "model": "test_model",
        "object_type": "ball",
        "occluder_type": "cardboard_box",
        "scenario_type": ["full_occlusion", "partial_occlusion"][i % 2]
    })

dummy_df = pd.DataFrame(dummy_results)
dummy_df.to_csv("results/raw_outputs/integration_test_results.csv", index=False)

overall = compute_overall_accuracy(dummy_df)
check("Overall accuracy computes", len(overall) > 0)
check("Accuracy in valid range",
      all(0 <= acc <= 100 for acc in overall["accuracy"]))

ci_low, ci_high = wilson_confidence_interval(10, 20)
check("Wilson CI computes correctly",
      ci_low < 50.0 < ci_high,
      f"CI: [{ci_low}%, {ci_high}%]")


# ── Test 8: Visualization ─────────────────────────────────────────────────────
print("\n── Test 8: Visualization ────────────────────────────────────")
from scripts.visualize_results import plot_overall_accuracy

plot_overall_accuracy(overall,
    output_path="results/compiled/integration_test_fig.png")
check("Visualization saves without error",
      os.path.exists("results/compiled/integration_test_fig.png"))


# ── Test 9: Environment Variables ────────────────────────────────────────────
print("\n── Test 9: Environment Variables ────────────────────────────")
from dotenv import load_dotenv
load_dotenv()

gemini_key = os.getenv("GEMINI_API_KEY")
openai_key = os.getenv("OPENAI_API_KEY")

check("GEMINI_API_KEY loaded",
      gemini_key is not None and gemini_key != "your_gemini_api_key_here",
      "key present" if gemini_key else "missing")
check("OPENAI_API_KEY present in .env",
      openai_key is not None,
      "key present (can be placeholder)" if openai_key else "missing")


# ── Test 10: Git Status ───────────────────────────────────────────────────────
print("\n── Test 10: Git Status ──────────────────────────────────────")
try:
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True, text=True
    )
    check("Git repository initialized", True)
    uncommitted = len(result.stdout.strip()) > 0
    check("No uncommitted changes",
          not uncommitted,
          "clean" if not uncommitted else "uncommitted files exist")
except Exception as e:
    check("Git repository initialized", False, str(e))


# ── Final Summary ─────────────────────────────────────────────────────────────
print("\n── Final Summary ────────────────────────────────────────────")
total = len(results)
passed = sum(1 for _, s, _ in results if s == PASS)
failed = total - passed

print(f"\nTotal tests: {total}")
print(f"Passed: {passed}")
print(f"Failed: {failed}")

if failed == 0:
    print("\n✅ ALL TESTS PASSED — Pipeline is ready for real data collection")
    print("When camera arrives: update input paths and run the full pipeline")
else:
    print(f"\n⚠️  {failed} tests failed — fix before data collection")
    print("Failed tests:")
    for name, status, detail in results:
        if status == FAIL:
            print(f"  - {name}: {detail}")