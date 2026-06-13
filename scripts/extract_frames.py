import cv2
import os
import argparse
from tqdm import tqdm

def extract_frames(video_path, output_dir, fps=1):
    """
    Extract frames from a video file at specified FPS.
    For now works on any .mp4 file.
    When RealSense bag files arrive, we will add bag file support.
    
    Args:
        video_path: Path to input video file
        output_dir: Directory to save extracted frames
        fps: Frames per second to extract (default: 1)
    """
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"ERROR: Could not open video file: {video_path}")
        return False

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / video_fps
    frame_interval = int(video_fps / fps)

    print(f"Video FPS: {video_fps}")
    print(f"Total Frames: {total_frames}")
    print(f"Duration: {duration:.2f} seconds")
    print(f"Extracting 1 frame every {frame_interval} frames")

    saved_count = 0
    frame_count = 0

    with tqdm(total=int(duration), desc="Extracting frames") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                frame_filename = os.path.join(output_dir, f"frame_{saved_count:04d}.jpg")
                cv2.imwrite(frame_filename, frame)
                saved_count += 1
                pbar.update(1)

            frame_count += 1

    cap.release()
    print(f"\nDone. Saved {saved_count} frames to {output_dir}")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from video")
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--output", required=True, help="Output directory for frames")
    parser.add_argument("--fps", type=int, default=1, help="Frames per second to extract")
    args = parser.parse_args()

    extract_frames(args.video, args.output, args.fps)
