"""
Custom dataset to load frames to train the VAE. 
All episodes are truncated to a max length of 1000 frames.
"""

import os
import random
import torch
from torch.utils.data import Dataset


class FrameDataset(Dataset):
    def __init__(self, path, frames_per_file=999):
        self.path = path
        self.files = [
            os.path.join(path, f) for f in os.listdir(path) if f.endswith(".pth")
        ]
        self.files = sorted(self.files)
        random.shuffle(self.files)
        self.frames_per_file = frames_per_file

        # NOTE: Caching is used for efficiency. If we don't cache we'd open and close a file for every single frame, 
        # and with ~10,000 rollouts x ~999 frames, thats ~ 10M file opens per epoch, which overwhelms a shared HPC filesystem.
        # FIX: Cache one file's full contents in memory when first opened, serve all frames from that file, then move to the next.
        # This dropped training time by ~36x (12+ hours -> ~20 min per epoch). 
        # TRADEOFF: This requires us to use shuffle=False, meaning each batch is drawn from consecutive frames in one episode, 
        # which are visually very similar. Since the VAE treats every frame independently (no temporal dependency in the loss), 
        # this doesn't affect correctness, but could mean a noisier gradient steps than a fully shuffled dataset.
        self.cached_file = None
        self.cached_obs = None


    def __len__(self):
        return len(self.frames) * self.frames_per_file

    def __getitem__(self, idx):
        file_i = idx // self.frames_per_file # which file
        frame_i = idx % self.frames_per_file # frame within the file
        
        if file_i != self.cached_file:
            path = self.files[file_i]
            data = torch.load(path)
            self.cached_obs = data["observations"]
            self.cached_file = file_i
        
        obs = self.cached_obs[frame_i]
        obs = obs.float() / 255.0
        obs = torch.permute(obs, (2, 0, 1))

        return obs