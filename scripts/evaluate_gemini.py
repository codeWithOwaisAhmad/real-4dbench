from google import genai
from google.genai import types
import os
import json
import pandas as pd
import base64
import time
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

# ── Configuration ────────────────────────────────────────────────────────────
GEMINI_MODEL = "gemini-2.0-flash"
MAX_FRAMES_PER_CLIP = 10  # Send max 10 frames per clip to API
DELAY_BETWEEN_CALLS = 10   # Seconds between API calls (rate limit safety)

# ── Setup ────────────────────────────────────────────────────────────────────
def setup_gemini():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in .env file")
    client = genai.Client(api_key=api_key)
    return client


# ── Frame Loading ─────────────────────────────────────────────────────────────
def load_frames_for_clip(frames_dir, clip_id, max_frames=MAX_FRAMES_PER_CLIP):
    """
    Load RGB frames for a clip.
    Selects evenly spaced frames up to max_frames limit.
    """
    clip_dir = os.path.join(frames_dir, clip_id)
    if not os.path.exists(clip_dir):
        print(f"WARNING: No frames directory found for {clip_id}")
        return []

    frame_files = sorted([
        f for f in os.listdir(clip_dir)
        if f.endswith('.jpg') or f.endswith('.png')
    ])

    if not frame_files:
        print(f"WARNING: No frames found in {clip_dir}")
        return []

    # Select evenly spaced frames
    if len(frame_files) > max_frames:
        step = len(frame_files) // max_frames
        frame_files = frame_files[::step][:max_frames]

    frames = []
    for filename in frame_files:
        frame_path = os.path.join(clip_dir, filename)
        with open(frame_path, "rb") as f:
            frame_data = base64.b64encode(f.read()).decode("utf-8")
        frames.append({
            "filename": filename,
            "data": frame_data
        })

    return frames


def load_frames_with_depth(rgb_dir, depth_dir, clip_id, max_frames=MAX_FRAMES_PER_CLIP):
    """
    Load RGB frames alternating with colorized depth frames.
    This is the RGB+Depth condition.
    One depth frame for every RGB frame.
    """
    rgb_frames = load_frames_for_clip(rgb_dir, clip_id, max_frames)

    depth_clip_dir = os.path.join(depth_dir, clip_id)
    if not os.path.exists(depth_clip_dir):
        print(f"WARNING: No depth frames for {clip_id} — falling back to RGB only")
        return rgb_frames

    depth_files = sorted([
        f for f in os.listdir(depth_clip_dir)
        if f.endswith('.jpg') or f.endswith('.png')
    ])

    # Alternate RGB and depth frames
    combined_frames = []
    for i, rgb_frame in enumerate(rgb_frames):
        combined_frames.append(rgb_frame)
        if i < len(depth_files):
            depth_path = os.path.join(depth_clip_dir, depth_files[i])
            with open(depth_path, "rb") as f:
                depth_data = base64.b64encode(f.read()).decode("utf-8")
            combined_frames.append({
                "filename": depth_files[i],
                "data": depth_data,
                "is_depth": True
            })

    return combined_frames


# ── Prompt Builder ────────────────────────────────────────────────────────────
def build_prompt(question, options, condition="rgb"):
    """
    Build the prompt sent to Gemini for each QA pair.
    Condition: 'rgb' or 'rgb_depth'
    """
    options_text = "\n".join([f"({k}) {v}" for k, v in options.items()])

    if condition == "rgb":
        context = "You are watching a sequence of video frames showing a physical occlusion event."
    else:
        context = (
            "You are watching a sequence of video frames showing a physical occlusion event. "
            "Every second frame is a colorized depth map where colors represent distances "
            "(blue = close, red = far). Use both RGB and depth information to answer."
        )

    prompt = f"""{context}

Question: {question}

Options:
{options_text}

Instructions:
- Watch the frames carefully in order
- Answer with ONLY the letter of the correct option: A, B, C, or D
- Do not explain your answer
- Do not write anything except the single letter

Your answer:"""

    return prompt


# ── Single QA Evaluation ──────────────────────────────────────────────────────
def evaluate_single_qa(model, frames, question, options, condition="rgb"):
    prompt = build_prompt(question, options, condition)

    content = []
    for frame in frames:
        content.append(
            types.Part.from_bytes(
                data=base64.b64decode(frame["data"]),
                mime_type="image/jpeg"
            )
        )
    content.append(prompt)

    try:
        response = model.models.generate_content(
            model=GEMINI_MODEL,
            contents=content
        )
        answer = response.text.strip().upper()

        for letter in ["A", "B", "C", "D"]:
            if letter in answer:
                return letter

        return "INVALID"

    except Exception as e:
        print(f"API Error: {e}")
        return "ERROR"


# ── Full Dataset Evaluation ───────────────────────────────────────────────────
def evaluate_dataset(dataset_path, rgb_frames_dir, depth_frames_dir,
                     output_path, condition="rgb"):
    """
    Run Gemini evaluation on full dataset.
    condition: 'rgb' or 'rgb_depth'
    Saves results incrementally — every 10 clips.
    """
    model = setup_gemini()

    with open(dataset_path, "r") as f:
        dataset = json.load(f)

    results = []
    print(f"\nEvaluating {len(dataset['clips'])} clips — Condition: {condition}")
    print(f"Model: {GEMINI_MODEL}\n")

    for i, clip in enumerate(tqdm(dataset["clips"], desc="Evaluating clips")):
        clip_id = clip["clip_id"]

        # Load frames based on condition
        if condition == "rgb":
            frames = load_frames_for_clip(rgb_frames_dir, clip_id)
        else:
            frames = load_frames_with_depth(rgb_frames_dir, depth_frames_dir, clip_id)

        if not frames:
            print(f"Skipping {clip_id} — no frames found")
            continue

        # Evaluate each QA pair
        for qa in clip["qa_pairs"]:
            time.sleep(DELAY_BETWEEN_CALLS)

            model_answer = evaluate_single_qa(
                model, frames,
                qa["question"],
                qa["options"],
                condition
            )

            result = {
                "clip_id": clip_id,
                "question_id": qa["question_id"],
                "question_type": qa["question_type"],
                "question": qa["question"],
                "correct_answer": qa["correct_answer"],
                "model_answer": model_answer,
                "is_correct": model_answer == qa["correct_answer"],
                "condition": condition,
                "model": GEMINI_MODEL,
                "object_type": clip["object_type"],
                "occluder_type": clip["occluder_type"],
                "scenario_type": clip["scenario_type"]
            }
            results.append(result)

        # Save incrementally every 10 clips
        if (i + 1) % 10 == 0:
            df = pd.DataFrame(results)
            df.to_csv(output_path, index=False)
            print(f"Checkpoint saved — {i+1} clips done")

    # Final save
    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nEvaluation complete. Results saved to {output_path}")
    print(f"Total questions evaluated: {len(results)}")

    return df


# ── Quick Test on Dummy Data ──────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate dataset with Gemini")
    parser.add_argument("--dataset", default="qa/dataset.json",
                        help="Path to dataset JSON")
    parser.add_argument("--rgb_dir", default="data/rgb_frames",
                        help="Directory containing RGB frames")
    parser.add_argument("--depth_dir", default="data/depth_colorized",
                        help="Directory containing colorized depth frames")
    parser.add_argument("--output", default="results/raw_outputs/gemini_rgb.csv",
                        help="Output CSV path")
    parser.add_argument("--condition", default="rgb",
                        choices=["rgb", "rgb_depth"],
                        help="Evaluation condition")
    args = parser.parse_args()

    df = evaluate_dataset(
        args.dataset,
        args.rgb_dir,
        args.depth_dir,
        args.output,
        args.condition
    )

    if df is not None and len(df) > 0:
        accuracy = df["is_correct"].mean() * 100
        print(f"\nOverall Accuracy: {accuracy:.1f}%")
