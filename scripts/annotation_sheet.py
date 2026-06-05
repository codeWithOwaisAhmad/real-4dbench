import pandas as pd
import os
from datetime import datetime


def create_annotation_sheet(output_path="qa/annotation_sheet.csv"):
    """
    Create empty annotation spreadsheet.
    Fill this live during every recording session.
    """
    columns = [
        "clip_id",
        "filename",
        "object_type",
        "occluder_type",
        "scenario_type",
        "movement_speed",
        "occlusion_start_sec",
        "occlusion_end_sec",
        "depth_shows_object",
        "q1_type",
        "q1_question",
        "q1_correct_answer",
        "q2_type",
        "q2_question",
        "q2_correct_answer",
        "q3_type",
        "q3_question",
        "q3_correct_answer",
        "human_val_person1",
        "human_val_person2",
        "human_val_person3",
        "human_accuracy_rgb",
        "notes"
    ]

    df = pd.DataFrame(columns=columns)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Annotation sheet created at {output_path}")
    return df


def add_dummy_rows(output_path="qa/annotation_sheet.csv", n=5):
    """Add dummy rows to test the sheet structure."""
    df = pd.read_csv(output_path)

    dummy_objects = ["ball", "bottle", "mug", "book", "apple"]
    dummy_occluders = ["cardboard_box", "folder", "book_stack", "cloth"]
    dummy_scenarios = ["full_occlusion", "partial_occlusion", "fast_movement",
                       "slow_movement", "multiple_objects"]

    for i in range(n):
        row = {
            "clip_id": f"clip_{i+1:03d}",
            "filename": f"clip_{i+1:03d}.bag",
            "object_type": dummy_objects[i % len(dummy_objects)],
            "occluder_type": dummy_occluders[i % len(dummy_occluders)],
            "scenario_type": dummy_scenarios[i % len(dummy_scenarios)],
            "movement_speed": "slow" if i % 2 == 0 else "fast",
            "occlusion_start_sec": 2.0 + i * 0.5,
            "occlusion_end_sec": 5.0 + i * 0.5,
            "depth_shows_object": True if i % 2 == 0 else False,
            "q1_type": "tracking",
            "q1_question": f"Where is the {dummy_objects[i % len(dummy_objects)]} at the end of this clip?",
            "q1_correct_answer": "A",
            "q2_type": "state",
            "q2_question": f"At second 3 in this clip, can you see the {dummy_objects[i % len(dummy_objects)]}?",
            "q2_correct_answer": "C",
            "q3_type": "event",
            "q3_question": f"What happens to the {dummy_objects[i % len(dummy_objects)]} between second 2 and second 5?",
            "q3_correct_answer": "A",
            "human_val_person1": "",
            "human_val_person2": "",
            "human_val_person3": "",
            "human_accuracy_rgb": "",
            "notes": f"Dummy row {i+1} for testing"
        }
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

    df.to_csv(output_path, index=False)
    print(f"Added {n} dummy rows. Total rows: {len(df)}")
    return df


if __name__ == "__main__":
    df = create_annotation_sheet()
    df = add_dummy_rows()

    # Verify
    df_loaded = pd.read_csv("qa/annotation_sheet.csv")
    print(f"\nAnnotation sheet loaded successfully")
    print(f"Columns: {list(df_loaded.columns)}")
    print(f"Rows: {len(df_loaded)}")
    print("\nFirst row:")
    print(df_loaded.iloc[0].to_string())
    print("\nAnnotation sheet working correctly.")