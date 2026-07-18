import argparse
import json
import os
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data import FrameDataset
from models import ELBOLoss, VariationalAutoEncoder


def estimate_loss(model, Xval, val_iter, device, nbatches=20):
    model.eval()
    losses, recons = [], []

    with torch.no_grad():
        for _ in range(nbatches):
            try:
                batch = next(val_iter)
            except StopIteration:
                val_iter = iter(Xval)
                batch = next(val_iter)

            obs = batch.to(device, non_blocking=True)
            pred, _, mu, logvar = model(obs)
            loss, recon = ELBOLoss(pred, obs, mu, logvar)
            losses.append(loss.item())
            recons.append(recon.item())

    model.train()
    return sum(losses) / len(losses), sum(recons) / len(recons), val_iter


def vae_train(path):
    Path("checkpoints").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    eval_interval = 200

    files = sorted(
        [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".pth")]
    )

    random.shuffle(files)
    split = int(len(files) * 0.9)
    val = files[split:]
    train = files[:split]

    train_dataset = FrameDataset(path, files=train)
    val_dataset = FrameDataset(path, files=val)

    Xtr, Xval = (
        DataLoader(train_dataset, batch_size=256, shuffle=False, pin_memory=True),
        DataLoader(val_dataset, batch_size=256, shuffle=False, pin_memory=True),
    )
    val_iter = iter(Xval)

    vae = VariationalAutoEncoder().to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-4)

    trlossi, trreconi = [], []
    vallossi, valreconi = [], []
    vsteps = []

    for _ in range(1):
        vae.train()
        for idx, batch in enumerate(Xtr):
            # forward pass
            obs = batch.to(device, non_blocking=True)
            optimizer.zero_grad()
            pred, _, mu, logvar = vae(obs)
            loss, recon = ELBOLoss(pred, obs, mu, logvar)
            trlossi.append(loss.item())
            trreconi.append(recon.item())

            # backward pass
            loss.backward()

            # update
            optimizer.step()

            if idx % eval_interval == 0:
                val_loss, val_recon, val_iter = estimate_loss(
                    vae, Xval, val_iter, device
                )
                vallossi.append(val_loss)
                valreconi.append(val_recon)
                vsteps.append(idx)
                print(
                    f"batch: {idx}/{len(Xtr)} | "
                    f"train loss: {loss.item():.4f} | val loss: {val_loss:.4f} | "
                    f"train recon: {recon.item():.4f} | val recon: {val_recon:.4f}",
                    flush=True,
                )

    torch.save(vae.state_dict(), "checkpoints/vae.pth")  # save model weights
    with open("logs/vae_losses.json", "w") as f:
        json.dump(
            {
                "train_loss": trlossi,
                "train_recon": trreconi,
                "val_loss": vallossi,
                "val_recon": valreconi,
                "val_steps": vsteps,
            },
            f,
        )


def rnn_train():
    pass


def controller_train():
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, required=True, choices=["vae", "mdn-rnn", "controller"]
    )
    parser.add_argument("--path", type=str, default="data/rollouts")
    args = parser.parse_args()

    if args.model == "vae":
        vae_train(path=args.path)
    elif args.model == "mdn-rnn":
        rnn_train()
    elif args.model == "controller":
        controller_train()


if __name__ == "__main__":
    main()
