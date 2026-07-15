import argparse
import os

import gymnasium as gym
import numpy as np
import torch


def make_env(render_mode=None):
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
        terminated, truncated = False
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
        env.close()
        pass


if __name__ == "__main__":
    main()
