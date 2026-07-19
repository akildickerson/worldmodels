import json
from pathlib import Path

import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import FrameDataset
from models import VariationalAutoEncoder

CHECKPOINT_PATH = "checkpoints/vae.pth"
LOG_PATH = "logs/vae_losses.json"
SAMPLE_DATA_PATH = "data/rollouts"
OUTPUT_DIR = "figures"


def plot_loss_curves():
    # Plot loss curves with log10 scale. Plot both ELBO Loss and the Reconstruction Loss.
    # Save plots to figures.
    with open(LOG_PATH) as f:
        logs = json.load(f)

    train_loss = torch.log10(torch.tensor(logs["train_loss"]))
    val_loss = torch.log10(torch.tensor(logs["val_loss"]))
    train_recon = torch.log10(torch.tensor(logs["train_recon"]))
    val_recon = torch.log10(torch.tensor(logs["val_recon"]))
    val_steps = logs["val_steps"]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(train_loss, label="train", alpha=0.7)
    ax.plot(val_steps, val_loss, label="val")
    ax.set_title("ELBO loss (log10)")
    ax.set_xlabel("batch")
    ax.set_ylabel("log10(loss)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/vae_elbo_loss.png")
    plt.close(fig)
    print(f"saved {OUTPUT_DIR}/vae_elbo_loss.png")

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(train_recon, label="train", alpha=0.7)
    ax.plot(val_steps, val_recon, label="val")
    ax.set_title("Reconstruction loss (log10)")
    ax.set_xlabel("batch")
    ax.set_ylabel("log10(loss)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/vae_recon_loss.png")
    plt.close(fig)
    print(f"saved {OUTPUT_DIR}/vae_recon_loss.png")


def check_posterior_collapse(vae, dataset, device):
    # Check for posterior collapse. Check 1-step MSE and mu std. Low std means
    # VAE is outputting similar mu for all observations and a sign of posterior
    # collapse.
    # NOTE: In the previous implementation, a collapsed VAE showed
    # mu std ~ 0.001 and mu barely changed across frames/timesteps
    # (1-step MSE ~ 1e-8).
    idxs = list(range(0, len(dataset), max(1, len(dataset) // 20)))
    batch = torch.stack([dataset[i] for i in idxs]).to(device)

    with torch.no_grad():
        _, _, mu, _ = vae(batch)

    mu_std = mu.std().item()
    mu_mse = ((mu[1:] - mu[:-1]) ** 2).mean().item()

    print(f"mu std: {mu_std:.6f}")
    print(f"mu 1-step MSE: {mu_mse:.8f}")
    if mu_std < 0.01:
        print("WARNING: mu std is very low — possible posterior collapse.")
    else:
        print("mu shows meaningful variation across frames — no obvious collapse.")


def plot_reconstructions(vae, dataset, device):
    # Visualize reconstructions created by the VAE.
    idxs = [len(dataset) // 4, len(dataset) // 2, len(dataset) - 1]
    batch = torch.stack([dataset[i] for i in idxs]).to(device)

    with torch.no_grad():
        pred, z, mu, logvar = vae(batch)

    n = len(idxs)
    fig, axes = plt.subplots(2, n, figsize=(3 * n, 6))
    for i in range(n):
        orig = batch[i].cpu().permute(1, 2, 0).numpy()
        recon = pred[i].cpu().permute(1, 2, 0).numpy().clip(0, 1)

        axes[0, i].imshow(orig)
        axes[0, i].set_title(f"original {idxs[i]}")
        axes[0, i].axis("off")

        axes[1, i].imshow(recon)
        axes[1, i].set_title(f"reconstruction {idxs[i]}")
        axes[1, i].axis("off")

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/vae_reconstructions.png")
    plt.close(fig)
    print(f"saved {OUTPUT_DIR}/vae_reconstructions.png")


def main():
    Path(OUTPUT_DIR).mkdir(exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    vae = VariationalAutoEncoder().to(device)
    vae.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    vae.eval()

    dataset = FrameDataset(SAMPLE_DATA_PATH)

    plot_loss_curves()
    check_posterior_collapse(vae, dataset, device)
    plot_reconstructions(vae, dataset, device)


if __name__ == "__main__":
    main()
