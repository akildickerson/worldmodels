import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data import EpisodeDataset
from models import VariationalAutoEncoder

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, required=True)
parser.add_argument("--outdir", type=str, required=True)
parser.add_argument("--checkpoint", type=str, required=True)
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
outdir = Path(args.outdir)
outdir.mkdir(parents=True, exist_ok=True)

# build dataloader
dataset = EpisodeDataset(args.path)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# load trained Variational Auto Encoder
vae = VariationalAutoEncoder().to(device)
vae.load_state_dict(torch.load(args.checkpoint, map_location=device))
vae.eval()

# extract latents and save them to disk
idx = 0
with torch.no_grad():
    for batch in dataloader:
        obs, action = batch[0].to(device), batch[1].to(device)
        B, T, H, W, C = obs.shape
        obs = obs.reshape(B * T, H, W, C).permute(0, 3, 1, 2)  # want (B, C, H, W)
        x, z, mu, logvar = vae(obs)
        latent = mu.reshape(B, T, -1).cpu()

        # save each latent in the batch seperately
        for i in range(B):
            torch.save(
                {"latent": latent[i], "action": action[i].cpu()},
                outdir / f"{idx:06d}.pt",
            )
            idx += 1

print(f"saved {idx} episodes to {args.outdir}")
