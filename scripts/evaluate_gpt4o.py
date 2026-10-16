import os
import re
import json
import base64
import argparse
from pathlib import Path
from io import BytesIO

import pandas as pd
from PIL import Image
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MODEL_NAME = "gpt-4o"
MAX_FRAMES = 10
DETAIL_LEVEL = "low"

DATASET_JSON = "benchmark_qa_297.json"
FRAMES_BASE_DIR = "verify_frames"
OUTPUT_DIR = "results"


def encode_image(image_path):
    with Image.open(image_path) as image:
        rgb_image = image.convert("RGB")
        buffer = BytesIO()
        rgb_image.save(buffer, format="JPEG", quality=85)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")


def sample_frames(frame_list, max_frames=MAX_FRAMES):
    if len(frame_list) <= max_frames:
        return frame_list

    step = (len(frame_list) - 1) / (max_frames - 1)
    return [
        frame_list[round(index * step)]
        for index in range(max_frames)
    ]


def parse_prediction(response_text):
    clean_text = response_text.strip().upper()
    match = re.search(r"\b([A-D])\b", clean_text)
    return match.group(1) if match else clean_text[:1]


def run_evaluation(condition="rgb"):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    checkpoint_path = os.path.join(
        OUTPUT_DIR,
        f"gpt4o_full_{condition}_checkpoint.csv"
    )
    final_output_path = os.path.join(
        OUTPUT_DIR,
        f"gpt4o_full_{condition}_results.csv"
    )

    with open(DATASET_JSON, "r") as file:
        qa_data = json.load(file)

    processed_ids = set()
    records = []

    if os.path.exists(checkpoint_path):
        checkpoint_df = pd.read_csv(checkpoint_path)
        processed_ids = set(checkpoint_df["question_id"])
        records = checkpoint_df.to_dict("records")

    print(
        f"Loaded {len(qa_data)} QA pairs | Model: {MODEL_NAME} | "
        f"Modality: {condition}"
    )
    print(f"Remaining: {len(qa_data) - len(processed_ids)}")

    for item in tqdm(qa_data, desc=f"{MODEL_NAME}/{condition}"):
        question_id = item["question_id"]

        if question_id in processed_ids:
            continue

        clip_id = item["clip_id"]
        clip_dir = Path(FRAMES_BASE_DIR) / clip_id / condition

        frame_files = sorted(
            list(clip_dir.glob("*.jpg")) + list(clip_dir.glob("*.png"))
        )
        sampled_frames = sample_frames(frame_files, max_frames=MAX_FRAMES)

        content = []

        for frame_path in sampled_frames:
            encoded_image = encode_image(frame_path)
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encoded_image}",
                    "detail": DETAIL_LEVEL
                }
            })

        user_prompt = (
            f"Chronological video sequence from clip: {clip_id}.\n\n"
            f"Question: {item['question']}\n\n"
            "Respond strictly with the single option letter (A, B, C, or D)."
        )
        content.append({"type": "text", "text": user_prompt})

        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an objective spatial reasoning evaluator. "
                            "Return only the option letter."
                        )
                    },
                    {
                        "role": "user",
                        "content": content
                    }
                ],
                max_tokens=10,
                temperature=0.0
            )

            raw_text = response.choices[0].message.content
            prediction = parse_prediction(raw_text)

        except Exception as error:
            raw_text = f"ERROR: {str(error)}"
            prediction = "FAIL"

        ground_truth = item["ground_truth"].strip().upper()

        record = {
            "clip_id": clip_id,
            "question_id": question_id,
            "task_type": item.get("task_type", ""),
            "scenario": item.get("scenario", ""),
            "module": item.get("module", ""),
            "ground_truth": ground_truth,
            "prediction": prediction,
            "is_correct": prediction == ground_truth,
            "raw_response": raw_text,
            "model": MODEL_NAME,
            "modality": condition
        }

        records.append(record)
        pd.DataFrame(records).to_csv(checkpoint_path, index=False)

    final_df = pd.DataFrame(records)
    final_df.to_csv(final_output_path, index=False)

    accuracy = final_df["is_correct"].mean() * 100
    print(
        f"\n{MODEL_NAME} | {condition} | Accuracy: {accuracy:.2f}% | "
        f"Saved to {final_output_path}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=["rgb", "depth"], required=True)
    arguments = parser.parse_args()
    run_evaluation(arguments.condition)
