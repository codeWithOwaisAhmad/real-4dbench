import json
import os
from datetime import datetime


# Standard question templates — do not modify wording
QUESTION_TEMPLATES = {
    "tracking": "Where is the {object} at the end of this clip?",
    "event": "What happens to the {object} between second {start} and second {end}?",
    "state": "At second {time} in this clip, can you see the {object}?"
}

ANSWER_OPTIONS = {
    "tracking": {
        "A": "Fully visible in front of the {occluder}",
        "B": "Hidden behind the {occluder}",
        "C": "Partially visible behind the {occluder}",
        "D": "Removed from the scene"
    },
    "event": {
        "A": "It moves behind the {occluder} and reappears",
        "B": "It stays visible throughout",
        "C": "It disappears completely and does not reappear",
        "D": "It moves to a different location"
    },
    "state": {
        "A": "Yes, fully visible",
        "B": "Yes, partially visible",
        "C": "No, it is completely hidden",
        "D": "It has already left the frame"
    }
}


def create_clip_entry(clip_id, object_type, occluder_type, scenario_type,
                       occlusion_start, occlusion_end, depth_shows_object):
    """
    Create a single clip entry in the dataset.
    Call this for every clip after recording.
    """
    return {
        "clip_id": clip_id,
        "filename": f"{clip_id}.bag",
        "object_type": object_type,
        "occluder_type": occluder_type,
        "scenario_type": scenario_type,
        "occlusion_start_sec": occlusion_start,
        "occlusion_end_sec": occlusion_end,
        "depth_shows_object_during_occlusion": depth_shows_object,
        "recorded_at": datetime.now().isoformat(),
        "qa_pairs": []
    }


def add_qa_pair(clip_entry, question_type, question_text, options, correct_answer):
    """
    Add a QA pair to a clip entry.
    correct_answer must be A, B, C, or D.
    """
    assert correct_answer in ["A", "B", "C", "D"], "correct_answer must be A, B, C, or D"
    assert question_type in QUESTION_TEMPLATES, f"question_type must be one of {list(QUESTION_TEMPLATES.keys())}"

    qa_pair = {
        "question_id": f"{clip_entry['clip_id']}_q{len(clip_entry['qa_pairs']) + 1}",
        "question_type": question_type,
        "question": question_text,
        "options": options,
        "correct_answer": correct_answer
    }
    clip_entry["qa_pairs"].append(qa_pair)
    return clip_entry


def save_dataset(dataset, output_path):
    """Save full dataset to JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=2)
    print(f"Dataset saved to {output_path} — {len(dataset['clips'])} clips")


def load_dataset(json_path):
    """Load dataset from JSON file."""
    with open(json_path, "r") as f:
        return json.load(f)


def create_empty_dataset():
    """Initialize an empty dataset."""
    return {
        "dataset_name": "real-4dbench",
        "description": "Real physical occlusion benchmark using Intel RealSense D435i",
        "created_at": datetime.now().isoformat(),
        "total_clips": 0,
        "clips": []
    }


def add_clip_to_dataset(dataset, clip_entry):
    """Add a completed clip entry to the dataset."""
    dataset["clips"].append(clip_entry)
    dataset["total_clips"] = len(dataset["clips"])
    return dataset


# ── Quick test ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Create dummy dataset to verify structure
    dataset = create_empty_dataset()

    # Dummy clip 1
    clip = create_clip_entry(
        clip_id="clip_001",
        object_type="ball",
        occluder_type="cardboard_box",
        scenario_type="full_occlusion",
        occlusion_start=2.5,
        occlusion_end=6.0,
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

    clip = add_qa_pair(
        clip,
        question_type="state",
        question_text="At second 4 in this clip, can you see the ball?",
        options={
            "A": "Yes, fully visible",
            "B": "Yes, partially visible",
            "C": "No, it is completely hidden",
            "D": "It has already left the frame"
        },
        correct_answer="C"
    )

    dataset = add_clip_to_dataset(dataset, clip)

    # Save and reload to verify
    save_dataset(dataset, "qa/dataset.json")
    loaded = load_dataset("qa/dataset.json")

    print(f"\nDataset loaded successfully")
    print(f"Total clips: {loaded['total_clips']}")
    print(f"First clip ID: {loaded['clips'][0]['clip_id']}")
    print(f"QA pairs in first clip: {len(loaded['clips'][0]['qa_pairs'])}")
    print(f"First question: {loaded['clips'][0]['qa_pairs'][0]['question']}")
    print(f"Correct answer: {loaded['clips'][0]['qa_pairs'][0]['correct_answer']}")
    print("\nQA Manager working correctly.")