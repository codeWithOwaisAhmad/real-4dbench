import cv2
import numpy as np
import os
from tqdm import tqdm

def colorize_depth_frame(depth_frame, colormap=cv2.COLORMAP_JET):
    """
    Convert a raw depth frame to a colorized image.
    This is what we send to models as the 'depth input'.
    Note: Models see a rainbow-colored image, NOT raw depth values.
    """
    # Normalize depth to 0-255 range
    depth_normalized = cv2.normalize(depth_frame, None, 0, 255, cv2.NORM_MINMAX)
    depth_uint8 = np.uint8(depth_normalized)
    
    # Apply colormap
    depth_colorized = cv2.applyColorMap(depth_uint8, colormap)
    return depth_colorized


def colorize_depth_folder(depth_dir, output_dir):
    """
    Colorize all depth frames in a folder.
    Input: folder of raw depth .png files
    Output: folder of colorized depth .jpg files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    depth_files = sorted([f for f in os.listdir(depth_dir) if f.endswith('.png')])
    
    if not depth_files:
        print(f"No .png depth files found in {depth_dir}")
        return False
    
    print(f"Found {len(depth_files)} depth frames to colorize")
    
    for filename in tqdm(depth_files, desc="Colorizing depth frames"):
        depth_path = os.path.join(depth_dir, filename)
        depth_frame = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
        
        if depth_frame is None:
            print(f"WARNING: Could not read {filename}, skipping")
            continue
        
        colorized = colorize_depth_frame(depth_frame)
        
        output_filename = filename.replace('.png', '_colorized.jpg')
        output_path = os.path.join(output_dir, output_filename)
        cv2.imwrite(output_path, colorized)
    
    print(f"\nDone. Colorized frames saved to {output_dir}")
    return True


if __name__ == "__main__": 
    import argparse
    parser = argparse.ArgumentParser(description="Colorize depth frames")
    parser.add_argument("--depth_dir", required=True, help="Directory containing raw depth .png files")
    parser.add_argument("--output_dir", required=True, help="Output directory for colorized frames")
    args = parser.parse_args()

    colorize_depth_folder(args.depth_dir, args.output_dir)