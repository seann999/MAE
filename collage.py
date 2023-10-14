import glob
import numpy as np
import cv2
import os
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from moviepy.editor import VideoClip



def create_video_grid(videos, grid_shape, output_file='output.mp4'):
    rows, cols = grid_shape
    frame_height, frame_width = videos[0][0].shape[:2]
    grid = np.zeros((frame_height * rows, frame_width * cols, 3), dtype=np.uint8)
    # offsets = [0] * (rows * cols)
    video_id = np.clip(np.arange(rows * cols), 0, len(videos) - 1)
    frame_id = np.zeros(rows * cols, dtype=np.int32)
    next_video_id = np.max(video_id) + 1

    fig, ax = plt.subplots(figsize=(cols, rows))
    ax.axis('off')
    img = ax.imshow(grid, interpolation='nearest')

    def update(num):
        nonlocal next_video_id

        grid.fill(255)
        for idx, (clip_id, f) in enumerate(zip(video_id, frame_id)):
            r, c = divmod(idx, cols)
            
            if f >= len(videos[clip_id]):
                if next_video_id < len(videos):
                    clip_id = next_video_id

                    video_id[idx] = clip_id
                    f = 0
                    frame_id[idx] = f

                    next_video_id += 1
                else:
                    f = len(videos[clip_id]) - 1

            grid[r*frame_height:(r+1)*frame_height, c*frame_width:(c+1)*frame_width] = videos[clip_id][f]
            frame_id[idx] += 1

        return grid

    duration = 20 #sum(len(v) for v in videos) * 0.05  # total duration of all videos
    fps = 20  # 20 frames per second

    # Convert the animation to a moviepy VideoClip
    def make_frame(t):
        frame_num = int(t * fps)
        frame = update(frame_num)
        return frame

    ani_clip = VideoClip(make_frame, duration=duration)
    ani_clip.write_videofile(output_file, fps=fps)

paths = "/data/Downloads/theta_train_np/*"
root = "datasets/grabber"
files = sorted(glob.glob(paths))
perm = np.random.permutation(len(files))

videos = []
for path in [files[i] for i in perm[:2000]]:
    x = np.load(path)
    videos.append([cv2.resize(img, (224, 224)) for img in x["img_hand"]])
        
create_video_grid(videos, (6, 6))