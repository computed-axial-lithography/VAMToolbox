import cv2
import os
import math
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
import subprocess


# GUI Folder Picker
def pick_folder(title: str, initial_dir: str = ".") -> str:
    root = tk.Tk()
    root.withdraw()
    folder = filedialog.askdirectory(title=title, initialdir=initial_dir)
    root.destroy()
    if not folder:
        messagebox.showwarning("No folder selected", "Operation cancelled.")
        exit()
    return folder

# Configuration
print("Select the folder containing your 360 image set...")
input_dir    = pick_folder("Select INPUT folder (360 images)")

print("Select the folder where cropped images will be saved...")
output_dir   = pick_folder("Select OUTPUT folder", initial_dir=os.path.dirname(input_dir))

IMAGE_EXT    = ".png" 
IMAGE_PREFIX = "proj"           # base name prefix

IMAGE_WIDTH  = 2560
IMAGE_HEIGHT_ORIG = 4800                       
BLACK_PAD = 1600                        
IMAGE_HEIGHT = IMAGE_HEIGHT_ORIG + 2 * BLACK_PAD  
CROP_WIDTH   = 2560
CROP_HEIGHT  = 1600
TOTAL_DEGREE = 360
PITCH        = 800

STEP_SIZE   = PITCH / TOTAL_DEGREE
Y_START_MAX  = IMAGE_HEIGHT - CROP_HEIGHT
TOTAL_STEPS = math.floor(Y_START_MAX / STEP_SIZE) + 1

BOTTOM_HOLD = 10 # frames to hold at the bottom before start croping upwards
os.makedirs(output_dir, exist_ok=True)

print("=" * 60)
print(f"  Input folder   : {input_dir}")
print(f"  Output folder  : {output_dir}")
print(f"  Original img size  : {IMAGE_WIDTH}×{IMAGE_HEIGHT_ORIG}")
print(f"  Black padding      : {BLACK_PAD}px top & bottom → total height {IMAGE_HEIGHT}px") 
print(f"  Pitch          : {PITCH} px  ({TOTAL_DEGREE} degrees × {STEP_SIZE:.4f} px/degree)")
print(f"  Crop size      : {CROP_WIDTH} × {CROP_HEIGHT}")
print(f"  Vertical range : {Y_START_MAX} → 0  (top of frame)")
print(f"  Bottom hold frames : {BOTTOM_HOLD}")  
print(f"  Total crops    : {TOTAL_STEPS}")
print("=" * 60)

# Helper: load one image by 0-based index
def load_image(zero_based_index: int):
    # Builds:  proj_0000_0000.jpg  →  proj_0359_0000.jpg
    filename = f"{IMAGE_PREFIX}_{zero_based_index:04d}_0000{IMAGE_EXT}"
    path = os.path.join(input_dir, filename)
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    h, w = img.shape[:2]
    if w != IMAGE_WIDTH or h != IMAGE_HEIGHT_ORIG:
        raise ValueError(f"{filename}: expected {IMAGE_WIDTH}×{IMAGE_HEIGHT_ORIG}, got {w}×{h}")
    black = np.zeros((BLACK_PAD, IMAGE_WIDTH, 3), dtype=np.uint8)
    return np.vstack([black, img, black])   # shape: (8000, 2560, 3)

# Main loop
all_frames = []   # list of (y_top, img_index)
img_idx = 0 

# Phase 1: Hold at bottom
for _ in range(BOTTOM_HOLD):
    all_frames.append((Y_START_MAX, 0))

# Phase 2: Move crop upwards

for step in range(TOTAL_STEPS):

    # img_index = step % TOTAL_DEGREE     # 0-based: 0 → 359, wraps around

    y_top = Y_START_MAX - round(step * STEP_SIZE)
    y_top = max(0, y_top)
    all_frames.append((y_top, img_idx % TOTAL_DEGREE))
    img_idx += 1

    # if y_bot > IMAGE_HEIGHT:
    #     y_bot = IMAGE_HEIGHT
    #     y_top = y_bot - CROP_HEIGHT

# Phase 3: Hold at top
while img_idx % TOTAL_DEGREE != 0:
    all_frames.append((0, img_idx % TOTAL_DEGREE))
    img_idx += 1

# Phase 4: mirror the frames in reverse to move crop back down
for step in range(TOTAL_STEPS - 1, -1, -1):
    y_top = Y_START_MAX - round(step * STEP_SIZE)
    y_top = max(0, y_top)
    all_frames.append((y_top, img_idx % TOTAL_DEGREE))
    img_idx += 1

# Phase 5: Hold at bottom again till back to 0 degree
while img_idx % TOTAL_DEGREE != 0:
    all_frames.append((Y_START_MAX, img_idx % TOTAL_DEGREE))
    img_idx += 1

total_frames = len(all_frames)
print(f"\n  Total frames in sequence : {total_frames}")
# print(f"    Phase 1 (bottom hold)  : {BOTTOM_HOLD}")
# print(f"    Phase 2 (scroll up)    : {TOTAL_STEPS}")
# print(f"    Phase 3 (top hold→0°)  : {total_frames - BOTTOM_HOLD - TOTAL_STEPS - TOTAL_STEPS}")
# print(f"    Phase 4 (scroll down)  : {TOTAL_STEPS}")

# Main loop  (CHANGED: iterates over all_frames instead of range(TOTAL_STEPS))
frame_names = []

for frame_num, (y_top, img_index) in enumerate(all_frames):

    y_bot = y_top + CROP_HEIGHT
    # clamp (should not be needed, but safe)
    if y_bot > IMAGE_HEIGHT:
        y_bot = IMAGE_HEIGHT
        y_top = y_bot - CROP_HEIGHT

    img     = load_image(img_index)             # now returns padded 8000px image
    cropped = img[y_top:y_bot, 0:CROP_WIDTH]

    # CHANGED: filename includes frame_num so ordering is always correct
    out_name = f"frame_{frame_num:05d}_img{img_index:04d}_y{y_top}{IMAGE_EXT}"
    cv2.imwrite(os.path.join(output_dir, out_name), cropped)
    frame_names.append(out_name)

    print(f"  frame {frame_num:>5d} | img {img_index:>3d} | y_top={y_top:>4d}  →  {out_name}")

print("\nDone! All crops saved to:", output_dir)

#     # img     = load_image(img_index)
#     # cropped = img[y_top:y_bot, 0:CROP_WIDTH]

#     # Output name mirrors input convention: crop_0000_0000.jpg etc.
#     out_name = f"crop_{step:04d}_img{img_index:04d}_y{y_top}{IMAGE_EXT}"
#     cv2.imwrite(os.path.join(output_dir, out_name), cropped)

#     print(f"  step {step:>4d} | image proj_{img_index:04d}_0000 | y_top={y_top:>4d}  →  {out_name}")

# print("\nDone! All crops saved to:", output_dir)

# Assemble video
VIDEO_FPS  = 60
VIDEO_NAME = "cropped_video.mp4"
FFMPEG_PATH = r"C:\ffmpeg-8.0.1-essentials_build\bin\ffmpeg.exe"

video_path = os.path.join(output_dir, VIDEO_NAME)

# Build a text file listing all frames in order (ffmpeg concat demuxer)
list_path = os.path.join(output_dir, "frame_list.txt")
with open(list_path, "w") as f:
    for name in frame_names:                    # CHANGED: use frame_names (not recomputed)
        f.write(f"file '{os.path.join(output_dir, name)}'\n")

print(f"\nAssembling video with ffmpeg ({total_frames} frames @ {VIDEO_FPS} fps)...")

subprocess.run([
    FFMPEG_PATH,
    "-y",                    # overwrite output if exists
    "-r", str(VIDEO_FPS),              # input framerate
    "-f", "concat",                    # use concat demuxer
    "-safe", "0",                      # allow absolute paths in list
    "-i", list_path,                   # input = frame list
    "-c:v", "libx264",                 # H.264 codec
    "-pix_fmt", "yuv420p",             # max compatibility (QuickTime, browsers)
    "-crf", "18",                      # quality: 0=lossless, 51=worst, 18=near lossless
    video_path
], check=True)

print(f"\nDone! Video saved to: {video_path}")
