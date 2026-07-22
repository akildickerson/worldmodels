import argparse
import json
import os
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from data import FrameDataset, LatentDataset
from models import ELBOLoss, VariationalAutoEncoder, NLL, MixtureDensityNetwork


def vae_train(path):
    # internal function to estimate loss specifically for VAE.
    def _estimate_loss(model, Xval, val_iter, device, nbatches=20):
        model.eval()
        losses, recons = [], []

        with torch.no_grad():
            for _ in range(nbatches):
                # manually reset the iter. if we use itercycle it tries to store 1.1TB of information on the CPU.
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

    # -------------
    # Train the VAE
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

            # estimate loss
            if idx % eval_interval == 0:
                val_loss, val_recon, val_iter = _estimate_loss(
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
    # store loss stats and model checkpoints
    torch.save(
        vae.state_dict(), "checkpoints/vae.pth"
    )  # (use .pt in the future because its recommended)
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


def rnn_train(path, epochs):

    Path("checkpoints").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    eval_iters = 100

    # build dataset
    data = LatentDataset(path)
    Xtr, Xval = random_split(data, [0.8, 0.2])
    Xtr = DataLoader(Xtr, batch_size=32, shuffle=True)
    Xval = DataLoader(Xval, batch_size=32, shuffle=True)

    rnn = MixtureDensityNetwork().to(device)
    optimizer = torch.optim.Adam(rnn.parameters(), lr=1e-4)

    # function to estimate train and val loss
    def _estimate_loss(model, Xtr, Xval, device, nbatches=50):
        model.eval()
        out = {}

        with torch.no_grad():
            for split, loader in [("train", Xtr), ("val", Xval)]:
                iterator = iter(loader)
                losses = []
                for _ in range(nbatches):
                    try:
                        latent, action = next(iterator)
                    except StopIteration:
                        iterator = iter(loader)
                        latent, action = next(iterator)

                    latent, action = latent.to(device), action.to(device)
                    z = latent[:, :-1, :]
                    a = action[:, :-1, :]
                    target = latent[:, 1:, :]

                    h, logits, mu, sigma = model(z, a)
                    loss = NLL(logits, mu, sigma, target)
                    losses.append(loss.item())
                out[split] = sum(losses) / len(losses)
        model.train()
        return out

    # train loop
    trloss, valloss, steps = [], [], []
    for epoch in range(epochs):
        rnn.train()
        for idx, (latent, action) in enumerate(Xtr):
            latent, action = latent.to(device), action.to(device)

            z = latent[:, :-1, :]
            a = action[:, :-1, :]
            target = latent[:, 1:, :]

            # forward pass
            optimizer.zero_grad()
            _, logits, mu, sigma = rnn(z, a)
            loss = NLL(logits, mu, sigma, target)

            # backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(rnn.parameters(), max_norm=1.0)

            # update
            optimizer.step()

            # eval loss
            if idx % eval_iters == 0:
                out = _estimate_loss(rnn, Xtr, Xval, device)
                trloss.append(out["train"])
                valloss.append(out["val"])
                steps.append(epoch * len(Xtr) + idx)
                print(
                    f"epoch: {epoch} | train loss: {out['train']:.4f} | val loss {out['val']:.4f}",
                    flush=True,
                )

        torch.save(rnn.state_dict(), f"checkpoints/rnn_{epoch}.pt")
    torch.save(rnn.state_dict(), "checkpoints/rnn.pt")
    with open("logs/rnn_losses.json", "w") as f:
        json.dump({"train_loss": trloss, "val_loss": valloss, "steps": steps}, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, required=True, choices=["vae", "rnn", "controller"]
    )
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--path", type=str, default="data/rollouts")
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    if args.model == "vae":
        vae_train(path=args.path)
    elif args.model == "rnn":
        rnn_train(path=args.path, epochs=args.epochs)
    elif args.model == "controller":
        from controller import train_controller
        train_controller(workers=args.workers)


if __name__ == "__main__":
    main()
