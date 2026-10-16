#!/usr/bin/env python3

import os
import glob
import argparse
from pathlib import Path

import numpy as np
import cv2
from tqdm import tqdm


def colorize_metric_depth(depth_map, min_depth_mm=200, max_depth_mm=2500):
    if depth_map.dtype in [np.float32, np.float64]:
        depth_mm = (depth_map * 1000.0).astype(np.float32)
    else:
        depth_mm = depth_map.astype(np.float32)

    valid_mask = (depth_mm > 0) & (depth_mm <= max_depth_mm)
    depth_clipped = np.clip(depth_mm, min_depth_mm, max_depth_mm)

    norm_depth = (depth_clipped - min_depth_mm) / (max_depth_mm - min_depth_mm)
    norm_depth_uint8 = ((1.0 - norm_depth) * 255.0).astype(np.uint8)

    norm_depth_uint8[~valid_mask] = 0

    turbo_bgr = cv2.applyColorMap(norm_depth_uint8, cv2.COLORMAP_TURBO)
    turbo_bgr[~valid_mask] = [0, 0, 0]

    return turbo_bgr


def process_clip_directory(input_dir, output_dir, min_depth=200, max_depth=2500):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    files = sorted(
        list(input_path.glob("*.png")) + list(input_path.glob("*.npy"))
    )

    if not files:
        print(f"No depth frames found in {input_dir}")
        return

    for file_path in files:
        if file_path.suffix == ".npy":
            raw_depth = np.load(file_path)
        else:
            raw_depth = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)

        if raw_depth is None:
            continue

        colored_depth = colorize_metric_depth(
            raw_depth,
            min_depth,
            max_depth
        )

        output_file = output_path / f"{file_path.stem}_depth.jpg"
        cv2.imwrite(
            str(output_file),
            colored_depth,
            [cv2.IMWRITE_JPEG_QUALITY, 95]
        )


def main():
    parser = argparse.ArgumentParser(
        description="Batch convert 16-bit RealSense depth maps to Turbo colormaps."
    )
    parser.add_argument(
        "--input_base",
        type=str,
        default="verify_frames",
        help="Base directory containing clip folders."
    )
    parser.add_argument(
        "--output_base",
        type=str,
        default="verify_frames",
        help="Output directory to save colored depth."
    )
    parser.add_argument(
        "--min_depth_mm",
        type=int,
        default=200,
        help="Near clipping plane in millimeters."
    )
    parser.add_argument(
        "--max_depth_mm",
        type=int,
        default=2500,
        help="Far clipping plane in millimeters."
    )

    arguments = parser.parse_args()

    clip_dirs = sorted(
        directory
        for directory in Path(arguments.input_base).iterdir()
        if directory.is_dir()
    )

    print(
        f"Found {len(clip_dirs)} clip directories in {arguments.input_base}"
    )

    for clip_dir in tqdm(clip_dirs, desc="Colorizing depth frames"):
        raw_depth_dir = clip_dir / "raw_depth"
        target_output_dir = (
            Path(arguments.output_base) / clip_dir.name / "depth"
        )

        if raw_depth_dir.exists():
            process_clip_directory(
                raw_depth_dir,
                target_output_dir,
                arguments.min_depth_mm,
                arguments.max_depth_mm
            )
        else:
            process_clip_directory(
                clip_dir,
                target_output_dir,
                arguments.min_depth_mm,
                arguments.max_depth_mm
            )


if __name__ == "__main__":
    main()
