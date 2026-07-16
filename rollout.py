"""
Script to collect rollouts used to train the vae and mdn-rnn. Designed in a way
that SLURM array jobs can be used to make rollout collection more efficient.
"""

import argparse
import os

import gymnasium as gym
import numpy as np
import torch


def make_env(render_mode=None):
    """
    wrapper method, will enventually be able to pass in different gymnasium enviornements.
    """
    return gym.make("CarRacing-v3", render_mode=render_mode)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--num", type=int, required=True)
    args = parser.parse_args()

    os.makedirs("data/rollouts", exist_ok=True)

    env = make_env()

    for i in range(args.start, args.start + args.num):
        actions, observations = [], []
        obs, _ = env.reset()
        terminated = truncated = False
        while not (terminated or truncated):
            action = env.action_space.sample()
            obs, _, terminated, truncated, _ = env.step()
            actions.append(action)
            observations.append(obs)
        torch.save(
            {
                "observations": torch.tensor(
                    np.array(observations), dtype=torch.float32
                ),
                "actions": torch.tensor(np.array(actions), dtype=torch.float32),
            },
            f"data/rollouts/rollout_{i}.pth",
        )
        # NOTE: Wrapped in a numpy array first because it is more efficient and to avoid PyTorch
        # warning that results from creating a tensor from a list of numpy.ndarrays. Much faster 
        # to convert to a single numpy.ndarray first. 
        # NOTE: Although the VAE uses individual frames to train, the MDN-RNN uses rollouts to train, 
        # therefore we must distingush between different rollouts when we save them to disk.
        env.close()


if __name__ == "__main__":
    main()
