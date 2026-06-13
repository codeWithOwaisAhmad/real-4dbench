from openai import OpenAI
import os
import json
import pandas as pd
import base64
import time
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

# ── Configuration ────────────────────────────────────────────────────────────
GPT_MODEL = "gpt-4o"
MAX_FRAMES_PER_CLIP = 10
DELAY_BETWEEN_CALLS = 3  # Seconds between API calls

# ── Setup ────────────────────────────────────────────────────────────────────
def setup_openai():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in .env file")
    return OpenAI(api_key=api_key)


# ── Frame Loading (reuse same logic as Gemini) ────────────────────────────────
def load_frames_for_clip(frames_dir, clip_id, max_frames=MAX_FRAMES_PER_CLIP):
    clip_dir = os.path.join(frames_dir, clip_id)
    if not os.path.exists(clip_dir):
        print(f"WARNING: No frames directory found for {clip_id}")
        return []

    frame_files = sorted([
        f for f in os.listdir(clip_dir)
        if f.endswith('.jpg') or f.endswith('.png')
    ])

    if not frame_files:
        return []

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
    rgb_frames = load_frames_for_clip(rgb_dir, clip_id, max_frames)

    depth_clip_dir = os.path.join(depth_dir, clip_id)
    if not os.path.exists(depth_clip_dir):
        print(f"WARNING: No depth frames for {clip_id} — falling back to RGB only")
        return rgb_frames

    depth_files = sorted([
        f for f in os.listdir(depth_clip_dir)
        if f.endswith('.jpg') or f.endswith('.png')
    ])

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
def evaluate_single_qa(client, frames, question, options, condition="rgb"):
    """
    Send frames + question to GPT-4o and get answer.
    GPT-4o accepts images as base64 in the messages array.
    """
    prompt = build_prompt(question, options, condition)

    # Build message content — GPT-4o format
    content = []
    for frame in frames:
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{frame['data']}",
                "detail": "low"  # low detail = cheaper, sufficient for occlusion tasks
            }
        })

    # Add question as final text
    content.append({
        "type": "text",
        "text": prompt
    })

    try:
        response = client.chat.completions.create(
            model=GPT_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": content
                }
            ],
            max_tokens=10  # We only need a single letter
        )

        answer = response.choices[0].message.content.strip().upper()

        for letter in ["A", "B", "C", "D"]:
            if letter in answer:
                return letter

        return "INVALID"

    except Exception as e:
        print(f"API Error: {e}")
        return "ERROR"


# ── Cost Estimator ────────────────────────────────────────────────────────────
def estimate_cost(num_clips, num_questions_per_clip=3,
                  frames_per_clip=10, conditions=2):
    """
    Rough cost estimate before running full evaluation.
    GPT-4o pricing: ~$0.001 per image (low detail)
    """
    total_questions = num_clips * num_questions_per_clip * conditions
    total_images = total_questions * frames_per_clip
    estimated_cost = total_images * 0.001

    print(f"\nCost Estimate for GPT-4o Evaluation:")
    print(f"Clips: {num_clips}")
    print(f"Questions per clip: {num_questions_per_clip}")
    print(f"Conditions: {conditions} (RGB + RGB+Depth)")
    print(f"Total API calls: {total_questions}")
    print(f"Total images sent: {total_images}")
    print(f"Estimated cost: ${estimated_cost:.2f} USD")
    print(f"Recommended budget: ${estimated_cost * 1.3:.2f} USD (30% buffer)\n")


# ── Full Dataset Evaluation ───────────────────────────────────────────────────
def evaluate_dataset(dataset_path, rgb_frames_dir, depth_frames_dir,
                     output_path, condition="rgb"):
    client = setup_openai()

    with open(dataset_path, "r") as f:
        dataset = json.load(f)

    results = []
    print(f"\nEvaluating {len(dataset['clips'])} clips — Condition: {condition}")
    print(f"Model: {GPT_MODEL}\n")

    for i, clip in enumerate(tqdm(dataset["clips"], desc="Evaluating clips")):
        clip_id = clip["clip_id"]

        if condition == "rgb":
            frames = load_frames_for_clip(rgb_frames_dir, clip_id)
        else:
            frames = load_frames_with_depth(rgb_frames_dir, depth_frames_dir, clip_id)

        if not frames:
            print(f"Skipping {clip_id} — no frames found")
            continue

        for qa in clip["qa_pairs"]:
            time.sleep(DELAY_BETWEEN_CALLS)

            model_answer = evaluate_single_qa(
                client, frames,
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
                "model": GPT_MODEL,
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


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate dataset with GPT-4o")
    parser.add_argument("--dataset", default="qa/dataset.json")
    parser.add_argument("--rgb_dir", default="data/rgb_frames")
    parser.add_argument("--depth_dir", default="data/depth_colorized")
    parser.add_argument("--output", default="results/raw_outputs/gpt4o_rgb.csv")
    parser.add_argument("--condition", default="rgb",
                        choices=["rgb", "rgb_depth"])
    parser.add_argument("--estimate_cost", action="store_true",
                        help="Print cost estimate and exit")
    args = parser.parse_args()

    if args.estimate_cost:
        estimate_cost(num_clips=100)
    else:
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
