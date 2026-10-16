"""
idea6_occlusion — Calibrated Occlusion Benchmark Pipeline (Ubuntu / Linux)
===========================================================================
Adapted from Idea 9 motion-blindness architecture for Idea 6 Physical Occlusion.
Records RealSense D435 RGB-D clips (.db3 / .bag) to external SSD.
Logs physical metric ground truth (Z_target, Z_occ, X_start, X_end) and QA keys.
Extracts 1-fps publication-grade verification frames (RGB + Turbo Colormap).

SSD PATH: /run/media/owais/External/idea6_occlusion
"""

import pyrealsense2 as rs
import numpy as np
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment
import os
import datetime
import time
import threading
import sys

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("WARNING: opencv-python not installed. Live preview disabled.")

# ─────────────────────────────────────────────────────────────
# CONFIGURATION & STORAGE PATHS
# ─────────────────────────────────────────────────────────────
SSD_ROOT        = "/run/media/owais/External/idea6_occlusion"
RECORDINGS_DIR  = os.path.join(SSD_ROOT, "bag_files")
VERIFY_DIR      = os.path.join(SSD_ROOT, "verify_frames")
EXCEL_LOG_PATH  = os.path.join(SSD_ROOT, "idea6_benchmark_log.xlsx")

LOCAL_VERIFY    = os.path.expanduser("~/Documents/idea6_occlusion/data/verify")

# RealSense Stream settings (Calibrated to 640x480 @ 30 FPS)
COLOR_W, COLOR_H, COLOR_FPS = 640, 480, 30
DEPTH_W, DEPTH_H, DEPTH_FPS = 640, 480, 30

# Depth dynamic range for publication figures (millimeters)
MIN_DEPTH_MM = 400.0   # 0.40 m (near table edge)
MAX_DEPTH_MM = 1500.0  # 1.50 m (background wall)

# ─────────────────────────────────────────────────────────────
# METADATA CONFIGURATION (IDEA 6 BENCHMARK SCHEMA)
# ─────────────────────────────────────────────────────────────
OBJECT_TYPES = [
    "tennis_ball", "apple", "computer_mouse", "green_stapler",
    "striped_cup", "sunblock_tube", "water_bottle", "red_mug"
]

OCCLUDER_TYPES = [
    "shoebox", "cereal_box", "tissue_box", "book", "tupperware"
]

SCENARIOS = [
    "full_occlusion",             # Scenario 1: Clean pass behind barrier
    "partial_occlusion",          # Scenario 2: Top/edge visible over barrier
    "velocity_variation",         # Scenario 3: Fast vs Slow pass
    "trajectory_reversal",        # Scenario 4: Enters behind, reverses, exits same side
    "multi_object_interaction",   # Scenario 5: Dual objects crossing/containing
]

DIRECTIONS = ["L2R", "R2L", "L2L_rebound", "R2R_rebound", "stationary"]
SPEEDS     = ["medium", "slow", "fast"]

HEADERS = [
    "Clip ID", "Date", "Time",
    "Target Object", "Occluder", "Scenario",
    "Direction", "Speed",
    "Z_target (m)", "Z_occ (m)", "X_start (m)", "X_end (m)",
    "Bag File", "Duration (s)",
    "Q1 Question", "Q1 Ground Truth",
    "Q2 Question", "Q2 Ground Truth",
    "Q3 Question", "Q3 Ground Truth",
    "Q4 Question", "Q4 Ground Truth",
    "Verify Frames Path", "Clip Valid", "Notes",
]

HEADER_FILL = PatternFill("solid", fgColor="1F4E79")
HEADER_FONT = Font(bold=True, color="FFFFFF", name="Arial", size=11)
DATA_FONT   = Font(name="Arial", size=10)
PASS_FILL   = PatternFill("solid", fgColor="C6EFCE")
FAIL_FILL   = PatternFill("solid", fgColor="FFC7CE")
PEND_FILL   = PatternFill("solid", fgColor="FFEB9C")

# ─────────────────────────────────────────────────────────────
# DETERMINISTIC QA ENGINE (IDEA 6)
# ─────────────────────────────────────────────────────────────
def build_qa_rows(target, occluder, scenario, direction):
    start_side = "left" if "L" in direction.upper() else "right"
    other_side = "right" if start_side == "left" else "left"

    # Q1: Final Spatial Location
    q1_text = f"Where is the {target} located at the end of the video sequence?\n" \
              f"(A) Completely hidden behind the {occluder}\n" \
              f"(B) Fully visible on the {other_side} side of the {occluder}\n" \
              f"(C) Fully visible on the {start_side} side of the {occluder}\n" \
              f"(D) Removed from the table entirely"
    q1_gt = "C" if scenario == "trajectory_reversal" else "B"

    # Q2: Kinematic Path Trajectory
    q2_text = f"Which statement best describes the horizontal motion trajectory of the {target}?\n" \
              f"(A) Travels from {start_side} to {other_side} across the scene behind the barrier\n" \
              f"(B) Enters behind the {occluder} and remains permanently inside\n" \
              f"(C) Enters behind the {occluder}, reverses direction while hidden, and exits back on the {start_side}\n" \
              f"(D) Remains stationary throughout the entire clip"
    q2_gt = "C" if scenario == "trajectory_reversal" else "A"

    # Q3: Metric Depth Ordering
    q3_text = f"What is the relative metric spatial depth relationship between the {target} and the {occluder}?\n" \
              f"(A) The {target} is closer to the camera than the {occluder}\n" \
              f"(B) The {target} is further from the camera than the {occluder}\n" \
              f"(C) Both share the exact same depth distance from the camera\n" \
              f"(D) The {occluder} is moving while the {target} stays fixed"
    q3_gt = "B"

    # Q4: Counterfactual Permanence Reasoning
    q4_text = f"If the {occluder} were removed midway through the occlusion interval, what would be observed?\n" \
              f"(A) The space behind the occluder would be completely empty\n" \
              f"(B) The {target} would be physically present and actively in transit across the table\n" \
              f"(C) The {target} would have vanished from the physical table environment\n" \
              f"(D) The table surface would be corrupted with non-physical artifacts"
    q4_gt = "B"

    return {
        "q1_text": q1_text, "q1_gt": q1_gt,
        "q2_text": q2_text, "q2_gt": q2_gt,
        "q3_text": q3_text, "q3_gt": q3_gt,
        "q4_text": q4_text, "q4_gt": q4_gt,
    }


def init_excel():
    if os.path.exists(EXCEL_LOG_PATH):
        return
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Idea6_Clips"
    for col, header in enumerate(HEADERS, 1):
        cell = ws.cell(row=1, column=col, value=header)
        cell.font      = HEADER_FONT
        cell.fill      = HEADER_FILL
        cell.alignment = Alignment(horizontal="center", wrap_text=True)
    ws.row_dimensions[1].height = 30
    wb.save(EXCEL_LOG_PATH)
    print(f"  Excel log initialized: {EXCEL_LOG_PATH}")


def suggest_next_clip_id():
    if not os.path.exists(EXCEL_LOG_PATH):
        return "clip_001"
    try:
        wb   = openpyxl.load_workbook(EXCEL_LOG_PATH)
        ws   = wb.active
        rows = ws.max_row
        if rows <= 1:
            return "clip_001"
        last = ws.cell(row=rows, column=1).value or "clip_000"
        num  = int(str(last).split("_")[1]) + 1
        return f"clip_{num:03d}"
    except Exception:
        return "clip_001"


def log_clip(session):
    try:
        wb = openpyxl.load_workbook(EXCEL_LOG_PATH)
        ws = wb.active
    except Exception:
        init_excel()
        wb = openpyxl.load_workbook(EXCEL_LOG_PATH)
        ws = wb.active

    target_row = ws.max_row + 1
    for r in range(2, ws.max_row + 1):
        if ws.cell(row=r, column=1).value == session["clip_id"]:
            target_row = r
            break

    values = [
        session["clip_id"],
        session["date"],
        session["time"],
        session["target_object"],
        session["occluder"],
        session["scenario"],
        session["direction"],
        session["speed"],
        session["z_target"],
        session["z_occ"],
        session["x_start"],
        session["x_end"],
        session["bag_path"],
        session["duration"],
        session["q1_text"],
        session["q1_gt"],
        session["q2_text"],
        session["q2_gt"],
        session["q3_text"],
        session["q3_gt"],
        session["q4_text"],
        session["q4_gt"],
        session["verify_path"],
        session["clip_valid"],
        session["notes"],
    ]

    for col, val in enumerate(values, 1):
        cell           = ws.cell(row=target_row, column=col, value=val)
        cell.font      = DATA_FONT
        cell.alignment = Alignment(wrap_text=True, vertical="top")

    v_col  = HEADERS.index("Clip Valid") + 1
    v_cell = ws.cell(row=target_row, column=v_col)
    if session["clip_valid"] == "PASS":
        v_cell.fill = PASS_FILL
        v_cell.font = Font(name="Arial", size=10, bold=True, color="375623")
    elif session["clip_valid"] == "FAIL":
        v_cell.fill = FAIL_FILL
        v_cell.font = Font(name="Arial", size=10, bold=True, color="9C0006")
    else:
        v_cell.fill = PEND_FILL

    ws.row_dimensions[target_row].height = 80
    wb.save(EXCEL_LOG_PATH)
    print(f"  Logged to Excel: row {target_row}")


# ─────────────────────────────────────────────────────────────
# INPUT HELPERS
# ─────────────────────────────────────────────────────────────
def pick(prompt, choices):
    print(f"\n{prompt}")
    for i, c in enumerate(choices, 1):
        print(f"  {i}. {c}")
    while True:
        try:
            idx = int(input("  Enter number: ").strip())
            if 1 <= idx <= len(choices):
                return choices[idx - 1]
        except (ValueError, KeyboardInterrupt):
            pass
        print(f"  Please enter 1–{len(choices)}")


def ask(prompt, allow_empty=False, default=None):
    while True:
        try:
            default_str = f" [{default}]" if default else ""
            val = input(f"\n{prompt}{default_str}: ").strip()
            if not val and default is not None:
                return default
            if val or allow_empty:
                return val
            print("  Cannot be empty.")
        except KeyboardInterrupt:
            print("\n  Interrupted.")
            sys.exit(0)


def ask_float(prompt, default=None):
    while True:
        try:
            default_str = f" [{default}]" if default is not None else ""
            val = input(f"\n{prompt}{default_str}: ").strip()
            if not val and default is not None:
                return float(default)
            return float(val)
        except (ValueError, KeyboardInterrupt):
            print("  Please enter a valid numeric value.")


# ─────────────────────────────────────────────────────────────
# LIVE PREVIEW (Threaded, matches Idea 9 workflow)
# ─────────────────────────────────────────────────────────────
_stop_preview = threading.Event()

def colorize_turbo(depth_data):
    valid_mask = depth_data > 0
    clipped = np.clip(depth_data, MIN_DEPTH_MM, MAX_DEPTH_MM)
    norm = ((clipped - MIN_DEPTH_MM) / (MAX_DEPTH_MM - MIN_DEPTH_MM) * 255.0).astype(np.uint8)
    inverted = 255 - norm
    color_vis = cv2.applyColorMap(inverted, cv2.COLORMAP_TURBO)
    color_vis[~valid_mask] = 0
    return color_vis


def _preview_thread(pipeline):
    align = rs.align(rs.stream.color)
    while not _stop_preview.is_set():
        try:
            frames = pipeline.wait_for_frames(timeout_ms=100)
            aligned = align.process(frames)
            c_frame = aligned.get_color_frame()
            d_frame = aligned.get_depth_frame()
            if not c_frame or not d_frame:
                continue
            c_img = np.asanyarray(c_frame.get_data())
            d_raw = np.asanyarray(d_frame.get_data())
            d_vis = colorize_turbo(d_raw)

            combined = np.hstack((c_img, d_vis))
            cv2.imshow("Idea 6 Calibrated Live Preview (RGB | Turbo Depth)", combined)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except Exception:
            break
    cv2.destroyAllWindows()


# ─────────────────────────────────────────────────────────────
# RECORDING (Preview Pipeline -> Recording Pipeline on ENTER)
# ─────────────────────────────────────────────────────────────
def record(bag_path, clip_id):
    # Step 1: Preview Only (Zero disk writing)
    preview_pipeline = rs.pipeline()
    preview_config   = rs.config()
    preview_config.enable_stream(rs.stream.color, COLOR_W, COLOR_H, rs.format.bgr8, COLOR_FPS)
    preview_config.enable_stream(rs.stream.depth, DEPTH_W, DEPTH_H, rs.format.z16,  DEPTH_FPS)

    print(f"\n  Starting preview for {clip_id}...")
    preview_pipeline.start(preview_config)

    _stop_preview.clear()
    if CV2_AVAILABLE:
        t = threading.Thread(target=_preview_thread, args=(preview_pipeline,), daemon=True)
        t.start()

    print(f"\n  Live preview active.")
    print(f"  Position your target at the start mark (Z=0.836m, X=-0.35m or +0.35m).")
    input("\n  >>> Press ENTER to START recording <<<")

    # Step 2: Stop preview cleanly
    _stop_preview.set()
    time.sleep(0.2)
    preview_pipeline.stop()

    # Step 3: Record directly to ROS2 .db3 / .bag file
    record_pipeline = rs.pipeline()
    record_config   = rs.config()
    record_config.enable_stream(rs.stream.color, COLOR_W, COLOR_H, rs.format.bgr8, COLOR_FPS)
    record_config.enable_stream(rs.stream.depth, DEPTH_W, DEPTH_H, rs.format.z16,  DEPTH_FPS)
    record_config.enable_record_to_file(bag_path)

    start_time = time.time()
    record_pipeline.start(record_config)

    print(f"\n  ● RECORDING {clip_id}...")
    print("    [1] Hold stationary for 1.5s (lock exposure)")
    print("    [2] Perform smooth transit (3.0s)")
    print("    [3] Hold destination stationary for 1.5s")
    print("    >>> Press ENTER to STOP recording <<<")
    input()

    duration = round(time.time() - start_time, 1)
    record_pipeline.stop()

    print(f"\n  Capture complete. Clean Duration: {duration}s")
    print(f"  Saved : {bag_path}")
    return duration


# ─────────────────────────────────────────────────────────────
# PUBLICATION-GRADE VERIFICATION FRAME EXTRACTION
# ─────────────────────────────────────────────────────────────
def extract_verify_frames(bag_path, clip_id):
    print(f"\n  Extracting Turbo verification frames for {clip_id}...")

    ssd_clip_dir   = os.path.join(VERIFY_DIR, clip_id)
    local_clip_dir = os.path.join(LOCAL_VERIFY, clip_id)
    os.makedirs(ssd_clip_dir,   exist_ok=True)
    os.makedirs(local_clip_dir, exist_ok=True)

    pipeline  = rs.pipeline()
    config    = rs.config()
    config.enable_device_from_file(bag_path, repeat_playback=False)

    profile  = pipeline.start(config)
    playback = profile.get_device().as_playback()
    playback.set_real_time(False)

    frame_idx   = 0
    saved_count = 0

    try:
        while True:
            try:
                frames = pipeline.wait_for_frames(timeout_ms=2000)
            except RuntimeError:
                break

            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                frame_idx += 1
                continue

            # Save exactly 1 frame per second (at 30 FPS)
            if frame_idx % 30 == 0:
                sec_val = frame_idx // 30
                fname   = f"sec_{sec_val:02d}"

                color_img = np.asanyarray(color_frame.get_data())
                depth_raw = np.asanyarray(depth_frame.get_data())
                depth_turbo = colorize_turbo(depth_raw)

                cv2.putText(color_img, f"t={sec_val}s", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(depth_turbo, f"t={sec_val}s", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                for out_dir in [ssd_clip_dir, local_clip_dir]:
                    cv2.imwrite(os.path.join(out_dir, f"{fname}_rgb.jpg"),
                                color_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    cv2.imwrite(os.path.join(out_dir, f"{fname}_depth.jpg"),
                                depth_turbo, [cv2.IMWRITE_JPEG_QUALITY, 95])

                saved_count += 1

            frame_idx += 1

    finally:
        pipeline.stop()

    print(f"  Extracted {saved_count} publication-grade frame(s) cleanly.")
    print(f"  Saved to : {ssd_clip_dir}")
    return local_clip_dir, saved_count


# ─────────────────────────────────────────────────────────────
# MAIN PIPELINE LOOP
# ─────────────────────────────────────────────────────────────
def main():
    if not os.path.exists("/run/media/owais/External"):
        print("\n  ERROR: SSD not detected at /run/media/owais/External")
        print("  Plug in the external drive and rerun.")
        sys.exit(1)

    os.makedirs(RECORDINGS_DIR, exist_ok=True)
    os.makedirs(VERIFY_DIR,     exist_ok=True)
    os.makedirs(LOCAL_VERIFY,   exist_ok=True)
    init_excel()

    print("\n" + "="*60)
    print("  IDEA 6: OCCLUSION & OBJECT PERMANENCE BENCHMARK")
    print("  Owais Ahmad | Calibrated D435 Pipeline | 2026")
    print("="*60)

    # Physical calibrated defaults from your live setup
    last_z_target = 0.836
    last_z_occ    = 0.690

    while True:
        action = pick("What do you want to do?", [
            "Record a new clip",
            "Exit",
        ])

        if action == "Exit":
            print("\n  Done. All calibrated records saved to SSD.")
            break

        elif action == "Record a new clip":
            suggested_id = suggest_next_clip_id()

            print(f"\n{'='*60}")
            print("  CLIP SETUP")
            print("="*60)

            clip_id_input = ask("Enter Clip ID", allow_empty=True, default=suggested_id)
            if clip_id_input.isdigit():
                clip_id = f"clip_{int(clip_id_input):03d}"
            elif clip_id_input.startswith("clip_"):
                suffix = clip_id_input.split("_")[1]
                clip_id = f"clip_{int(suffix):03d}" if suffix.isdigit() else clip_id_input
            else:
                clip_id = clip_id_input

            print(f"\n  Active Clip ID: {clip_id}")

            now          = datetime.datetime.now()
            target_obj   = pick("Target Object:", OBJECT_TYPES)
            occluder_obj = pick("Occluder Barrier:", OCCLUDER_TYPES)
            scenario     = pick("Scenario:", SCENARIOS)
            direction    = pick("Direction:", DIRECTIONS)
            speed        = pick("Speed:", SPEEDS)
            notes        = ask("Notes (press Enter to skip)", allow_empty=True)

            qa = build_qa_rows(target_obj, occluder_obj, scenario, direction)

            bag_name = f"{clip_id}.db3"
            bag_path = os.path.join(RECORDINGS_DIR, bag_name)

            # Record
            duration = record(bag_path, clip_id)
            if duration == 0:
                print("  Recording failed — clip not saved.")
                continue

            # Extract 1-fps publication Turbo frames
            verify_path, n_frames = extract_verify_frames(bag_path, clip_id)

            # Calibrated Depth Registration
            print(f"\n{'='*60}")
            print(f"  CALIBRATED DEPTH REGISTRATION — {clip_id}")
            print("="*60)
            z_target = ask_float("Enter Z_target (m)", default=last_z_target)
            z_occ    = ask_float("Enter Z_occ (m)", default=last_z_occ)
            
            x_start  = -0.35 if "L" in direction else 0.35
            x_end    = 0.35 if direction == "L2R" else (-0.35 if direction == "R2L" else x_start)

            last_z_target = z_target
            last_z_occ    = z_occ

            # Geometric validation check
            is_valid = True
            delta_z_buffer = round(z_target - z_occ, 3)
            if delta_z_buffer <= 0.05:
                print(f"  WARNING: Occlusion buffer too small! (Delta_Z = {delta_z_buffer}m)")
                is_valid = False

            if is_valid:
                print(f"  Ground Truth Geometry: PASSED (Depth Buffer = {delta_z_buffer:+.3f}m) ✓")
                clip_valid = "PASS"
            else:
                override = input("  Inconsistency detected. Mark as PASS anyway? (y/n): ").strip().lower()
                clip_valid = "PASS" if override == "y" else "FAIL"

            session = {
                "clip_id":       clip_id,
                "date":          now.strftime("%Y-%m-%d"),
                "time":          now.strftime("%H:%M:%S"),
                "target_object": target_obj,
                "occluder":      occluder_obj,
                "scenario":      scenario,
                "direction":     direction,
                "speed":         speed,
                "z_target":      z_target,
                "z_occ":         z_occ,
                "x_start":       x_start,
                "x_end":         x_end,
                "bag_path":      bag_path,
                "duration":      duration,
                "q1_text":       qa["q1_text"],
                "q1_gt":         qa["q1_gt"],
                "q2_text":       qa["q2_text"],
                "q2_gt":         qa["q2_gt"],
                "q3_text":       qa["q3_text"],
                "q3_gt":         qa["q3_gt"],
                "q4_text":       qa["q4_text"],
                "q4_gt":         qa["q4_gt"],
                "verify_path":   verify_path,
                "clip_valid":    clip_valid,
                "notes":         notes,
            }
            log_clip(session)

            print(f"\n  Clip complete: {clip_id} | Status: {clip_valid}")
            print(f"  Extracted: {n_frames} clean frames saved to {verify_path}")
            print("="*60)


if __name__ == "__main__":
    main()