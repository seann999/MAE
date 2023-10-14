import glob
import numpy as np
import cv2
import os
from tqdm import tqdm

paths = "/data/Downloads/theta_train_np/*"
root = "datasets/grabber"

for path_i, path in enumerate(tqdm(sorted(glob.glob(paths)))):
    x = np.load(path)
    dpath = os.path.join(root, os.path.join(f"{path_i:06d}"))
    os.makedirs(dpath, exist_ok=True)
    
    for rgb_i, rgb in enumerate(x["img_hand"]):
        # rgb = np.rot90(rgb)
        img_path = os.path.join(dpath, f"{rgb_i:04d}.jpg")
        cv2.imwrite(img_path, rgb[:, :, ::-1])
