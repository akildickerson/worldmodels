import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data import FrameDataset
from models import VAE, ELBOLoss


def vae_train(path):
    Path("checkpoints").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    data = FrameDataset(path)
    dataloader = DataLoader(data, batch_size=256, shuffle=False)

    vae = VAE().to(device)
    optimizer = torch.optim.AdamW(vae.parameters(), lr=1e-4)
    
    lossi, reconi = [], []

    for _ in range(1):
        for idx, batch in enumerate(dataloader):
            # forward pass
            obs = batch.to(device)
            optimizer.zero_grad()
            pred, _, mu, logvar = vae(obs)
            loss, recon = ELBOLoss(pred, obs, mu, logvar)

            # backward pass
            loss.backward()

            # update
            optimizer.step()

            # track stats
            lossi.append(loss.item())
            reconi.append(recon.item())
            if idx % 100 == 0:
                print(f"batch: {idx}/{len(dataloader)} | loss: {loss.item():.4f} | recon loss: {recon.item():.4f}", flush=True)

    torch.save(vae.state_dict(), "checkpoints/vae.pth") # save model weights
    # save total and reconstruction loss for plotting later
    with open("logs/vae_losses.json", "w") as f:
        json.dump({"loss":lossi, "recon":reconi}, f)

def rnn_train():
    pass

def controller_train():
    pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["vae", "mdn-rnn", "controller"])
    args = parser.parse_args()

    if args.model == "vae":
        vae_train(path="data/rollouts")
    elif args.model == "mdn-rnn":
        rnn_train()
    elif args.model == "controller":
        controller_train()
    
if __name__ == "__main__":
    main()