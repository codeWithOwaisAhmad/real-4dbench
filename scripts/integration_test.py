#!/usr/bin/env python3
"""
record_clip_idea6.py
Physical Data Collection Pipeline for Idea 6 (Occlusion Reasoning Benchmark).
Captures aligned Color (RGB) and Metric Depth (Z16) from Intel RealSense D435.
Saves:
  - Aligned RGB MP4 video (640x480 @ 30 FPS)
  - Compressed metric depth array (.npz, 16-bit uint16 in millimeters)
  - 1 FPS verification stills (RGB + Jet colormap depth)
  - Appends physical metadata and balanced Q1-Q4 QA pairs to idea6_benchmark_log.xlsx
"""

import os
import sys
import time
import cv2
import numpy as np
import pyrealsense2 as rs
import pandas as pd

# ==========================================
# MASTER METADATA & SCHEMA CONFIG
# ==========================================
EXCEL_LOG_PATH = "idea6_benchmark_log.xlsx"
DATASET_ROOT = "dataset_idea6"

EXCEL_COLUMNS = [
    "clip_id", "scenario_id", "scenario_name", "target_object", "occluder_object",
    "speed", "direction", "x_start_m", "x_end_m", "z_target_m", "z_occ_m",
    "clip_duration_s", "total_frames", "fps",
    "q1_question", "q1_A", "q1_B", "q1_C", "q1_D", "q1_ground_truth",
    "q2_question", "q2_A", "q2_B", "q2_C", "q2_D", "q2_ground_truth",
    "q3_question", "q3_A", "q3_B", "q3_C", "q3_D", "q3_ground_truth",
    "q4_question", "q4_A", "q4_B", "q4_C", "q4_D", "q4_ground_truth"
]

SCENARIO_MAP = {
    "1": ("Scenario 1", "Full Occlusion"),
    "2": ("Scenario 2", "Partial Occlusion"),
    "3": ("Scenario 3", "Velocity Variation"),
    "4": ("Scenario 4", "Occluded Trajectory Reversal"),
    "5": ("Scenario 5", "Multi-Object Interaction")
}

def init_master_log():
    if not os.path.exists(EXCEL_LOG_PATH):
        df = pd.DataFrame(columns=EXCEL_COLUMNS)
        df.to_excel(EXCEL_LOG_PATH, index=False)
        print(f"[SETUP] Initialized master spreadsheet: {EXCEL_LOG_PATH}")

def get_next_clip_id():
    if not os.path.exists(EXCEL_LOG_PATH):
        return "clip_001"
    try:
        df = pd.read_excel(EXCEL_LOG_PATH)
        if df.empty or "clip_id" not in df.columns:
            return "clip_001"
        existing_ids = df["clip_id"].dropna().astype(str).tolist()
        numeric_ids = [int(i.split("_")[1]) for i in existing_ids if i.startswith("clip_") and i.split("_")[1].isdigit()]
        next_num = max(numeric_ids) + 1 if numeric_ids else 1
        return f"clip_{next_num:03d}"
    except Exception:
        return "clip_001"

# ==========================================
# DETERMINISTIC QA ENGINE
# ==========================================
def generate_qa_pairs(scenario_num, target, occluder, direction):
    start_side = "left" if "L" in direction.upper() else "right"
    other_side = "right" if start_side == "left" else "left"

    # Q1: Final Spatial Location
    q1 = {
        "question": f"Where is the {target} located at the end of the video sequence?",
        "A": f"Completely hidden behind the {occluder}.",
        "B": f"Fully visible on the {other_side} side of the {occluder}.",
        "C": f"Fully visible on the {start_side} side of the {occluder}.",
        "D": f"The {target} was completely removed from the scene."
    }
    if scenario_num in ["1", "3"]: # Full transit or velocity pass
        q1_gt = "B"
    elif scenario_num == "2":      # Partial occlusion
        q1_gt = "B"
    elif scenario_num == "4":      # Reversal back to entry side
        q1_gt = "C"
    else:                          # Scenario 5 multi-object default
        q1_gt = "B"

    # Q2: Kinematic Path & Direction
    q2 = {
        "question": f"Which statement best describes the horizontal motion trajectory of the {target}?",
        "A": f"It travels from {start_side} to {other_side} across the scene behind the barrier.",
        "B": f"It enters behind the {occluder} from the {start_side} and remains permanently inside.",
        "C": f"It enters behind the {occluder} from the {start_side}, reverses direction while hidden, and exits back on the {start_side}.",
        "D": "It remains entirely stationary on the table throughout the sequence."
    }
    if scenario_num in ["1", "2", "3"]:
        q2_gt = "A"
    elif scenario_num == "4":
        q2_gt = "C"
    else:
        q2_gt = "A"

    # Q3: Metric Depth Ordering
    q3 = {
        "question": f"What is the relative spatial depth relationship between the {target} and the {occluder}?",
        "A": f"The {target} travels along a depth plane closer to the camera than the {occluder}.",
        "B": f"The {target} travels along a depth plane further from the camera than the {occluder}.",
        "C": f"The {target} and {occluder} share the exact same depth coordinate.",
        "D": f"The {occluder} is moving while the {target} stays fixed."
    }
    q3_gt = "B" # Target is always along Z=0.83m behind occluder at Z=0.69m

    # Q4: Counterfactual Permanence
    q4 = {
        "question": f"If an observer removed the {occluder} midway through the occlusion interval, what would be observed?",
        "A": "The space behind the occluder would be completely empty.",
        "B": f"The {target} would be physically present and actively in transit across the table.",
        "C": f"The {target} would have vanished from the physical table environment.",
        "D": "The table surface would be corrupted with non-physical artifacts."
    }
    q4_gt = "B"

    return {
        "q1": q1, "q1_gt": q1_gt,
        "q2": q2, "q2_gt": q2_gt,
        "q3": q3, "q3_gt": q3_gt,
        "q4": q4, "q4_gt": q4_gt
    }

# ==========================================
# MAIN EXECUTION ROUTINE
# ==========================================
def main():
    init_master_log()
    os.makedirs(DATASET_ROOT, exist_ok=True)

    # Initialize RealSense Streams (640x480 @ 30 FPS)
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Sensor-level filtering
    align = rs.align(rs.stream.color)
    spatial = rs.spatial_filter()
    temporal = rs.temporal_filter()
    hole_filling = rs.hole_filling_filter(1)
    colorizer = rs.colorizer()
    colorizer.set_option(rs.option.visual_preset, 1) # Jet preset
    colorizer.set_option(rs.option.min_distance, 0.45)
    colorizer.set_option(rs.option.max_distance, 1.50)

    print("\n[STARTING] Initializing RealSense D435 hardware pipeline...")
    profile = pipeline.start(config)

    depth_sensor = profile.get_device().first_depth_sensor()
    if depth_sensor.supports(rs.option.emitter_enabled):
        depth_sensor.set_option(rs.option.emitter_enabled, 1)
    if depth_sensor.supports(rs.option.laser_power):
        depth_sensor.set_option(rs.option.laser_power, 250)

    print("\n=======================================================")
    print(" IDEA 6 RECORDING INTERFACE (Calibrated Table Setup)")
    print(" Controls: [SPACE] = Start/Stop Recording | [Q] = Quit")
    print("=======================================================\n")

    recording = False
    recorded_rgb = []
    recorded_depth = []
    fps = 30.0

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            filtered_depth = spatial.process(depth_frame)
            filtered_depth = temporal.process(filtered_depth)
            filtered_depth = hole_filling.process(filtered_depth)

            color_img = np.asanyarray(color_frame.get_data())
            raw_depth = np.asanyarray(filtered_depth.get_data()) # 16-bit millimeters
            depth_vis = np.asanyarray(colorizer.colorize(filtered_depth).get_data())

            # UI Display
            display_img = color_img.copy()
            next_id = get_next_clip_id()

            if recording:
                recorded_rgb.append(color_img.copy())
                recorded_depth.append(raw_depth.copy())
                status_text = f"REC [{next_id}]: {len(recorded_rgb)} frames"
                status_color = (0, 0, 255)
            else:
                status_text = f"READY: Press [SPACE] to capture {next_id}"
                status_color = (0, 255, 0)

            cv2.putText(display_img, status_text, (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, status_color, 2)
            cv2.circle(display_img, (615, 25), 10, status_color, -1)

            combined_preview = np.hstack((display_img, depth_vis))
            cv2.imshow("Idea 6 Calibrated Capture Console", combined_preview)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                if not recording:
                    # START
                    recorded_rgb = []
                    recorded_depth = []
                    recording = True
                    print(f"\n>>> [{next_id}] RECORDING STARTED.")
                    print("--> Phase 1: Hold static for 1.5s (lock auto-exposure)")
                    print("--> Phase 2: Transit smoothly behind occluder (3.0s)")
                    print("--> Phase 3: Hold static at destination (1.5s)")
                else:
                    # STOP
                    recording = False
                    total_f = len(recorded_rgb)
                    duration = total_f / fps
                    clip_id = next_id
                    print(f"\n>>> RECORDING STOPPED. Captured {total_f} frames ({duration:.2f}s).")

                    # TERMINAL METADATA INTAKE
                    print(f"\n--- METADATA ENTRY FOR {clip_id} ---")
                    print("1: Full Occlusion | 2: Partial Occlusion | 3: Velocity | 4: Reversal | 5: Multi-Object")
                    scen_input = input("Select Scenario [1-5, default=1]: ").strip() or "1"
                    scen_id, scen_name = SCENARIO_MAP.get(scen_input, ("Scenario 1", "Full Occlusion"))

                    target = input("Target Object (e.g., tennis_ball, apple, mouse, stapler): ").strip() or "target_object"
                    occluder = input("Occluder (default: manzil_box): ").strip() or "manzil_box"
                    speed = input("Speed [slow / medium / fast, default=medium]: ").strip().lower() or "medium"
                    direction = input("Direction [L2R (Left to Right) / R2L (Right to Left), default=L2R]: ").strip().upper() or "L2R"

                    # SAVE ASSETS TO DISK
                    clip_dir = os.path.join(DATASET_ROOT, clip_id)
                    os.makedirs(clip_dir, exist_ok=True)
                    verif_dir = os.path.join(clip_dir, "verification_frames")
                    os.makedirs(verif_dir, exist_ok=True)

                    # 1. Aligned Color Video (MP4)
                    video_path = os.path.join(clip_dir, f"{clip_id}_rgb.mp4")
                    writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (640, 480))
                    for f in recorded_rgb:
                        writer.write(f)
                    writer.release()

                    # 2. Metric 16-bit Depth Array (NPZ)
                    npz_path = os.path.join(clip_dir, f"{clip_id}_depth_z16.npz")
                    np.savez_compressed(npz_path, data=np.array(recorded_depth, dtype=np.uint16))

                    # 3. 1 FPS Verification Stills
                    step = int(fps)
                    for sec_idx, f_idx in enumerate(range(0, total_f, step)):
                        v_rgb = recorded_rgb[f_idx]
                        v_z16 = recorded_depth[f_idx]
                        v_depth_jet = cv2.applyColorMap(cv2.convertScaleAbs(v_z16, alpha=0.03), cv2.COLORMAP_JET)
                        cv2.imwrite(os.path.join(verif_dir, f"sec_{sec_idx:02d}_rgb.jpg"), v_rgb)
                        cv2.imwrite(os.path.join(verif_dir, f"sec_{sec_idx:02d}_depth.jpg"), v_depth_jet)

                    # 4. Deterministic QA Generation & Excel Append
                    qa = generate_qa_pairs(scen_input, target, occluder, direction)
                    x_start = -0.35 if "L" in direction else 0.35
                    x_end = 0.35 if "L" in direction else -0.35
                    if scen_input == "4": # Reversal
                        x_end = x_start

                    row = {
                        "clip_id": clip_id,
                        "scenario_id": scen_id,
                        "scenario_name": scen_name,
                        "target_object": target,
                        "occluder_object": occluder,
                        "speed": speed,
                        "direction": direction,
                        "x_start_m": x_start,
                        "x_end_m": x_end,
                        "z_target_m": 0.836, # Calibrated from your RealSense setup
                        "z_occ_m": 0.690,    # Calibrated from your RealSense setup
                        "clip_duration_s": round(duration, 2),
                        "total_frames": total_f,
                        "fps": fps,
                        "q1_question": qa["q1"]["question"], "q1_A": qa["q1"]["A"], "q1_B": qa["q1"]["B"], "q1_C": qa["q1"]["C"], "q1_D": qa["q1"]["D"], "q1_ground_truth": qa["q1_gt"],
                        "q2_question": qa["q2"]["question"], "q2_A": qa["q2"]["A"], "q2_B": qa["q2"]["B"], "q2_C": qa["q2"]["C"], "q2_D": qa["q2"]["D"], "q2_ground_truth": qa["q2_gt"],
                        "q3_question": qa["q3"]["question"], "q3_A": qa["q3"]["A"], "q3_B": qa["q3"]["B"], "q3_C": qa["q3"]["C"], "q3_D": qa["q3"]["D"], "q3_ground_truth": qa["q3_gt"],
                        "q4_question": qa["q4"]["question"], "q4_A": qa["q4"]["A"], "q4_B": qa["q4"]["B"], "q4_C": qa["q4"]["C"], "q4_D": qa["q4"]["D"], "q4_ground_truth": qa["q4_gt"]
                    }

                    df_current = pd.read_excel(EXCEL_LOG_PATH)
                    df_current = pd.concat([df_current, pd.DataFrame([row])], ignore_index=True)
                    df_current.to_excel(EXCEL_LOG_PATH, index=False)

                    print(f"[LOGGED] Registered {clip_id} to {EXCEL_LOG_PATH} successfully.")
                    print("Ready for next sequence.\n")

            elif key == ord('q'):
                print("\n[EXIT] Terminating recording script.")
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
