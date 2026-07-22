import json
import multiprocessing as mp
from pathlib import Path

import cma
import gymnasium as gym
import numpy as np
import torch

from models import MLP, MixtureDensityNetwork, VariationalAutoEncoder

checkpoints = Path("checkpoints")

# global variables inside each worker - loaded once per worker
_vae = None
_rnn = None
_controller = None
_device = None


# --------------------------


def init_worker(device: str):
    global _vae, _rnn, _controller, _device
    _device = torch.device(device)

    _vae = VariationalAutoEncoder().to(_device)
    _vae.load_state_dict(torch.load("checkpoints/vae.pth", map_location=_device))
    _vae.eval()

    _rnn = MixtureDensityNetwork().to(_device)
    _rnn.load_state_dict(torch.load("checkpoints/rnn.pt", map_location=_device))
    _rnn.eval()

    _controller = MLP().to(_device)


def _load_params(controller: MLP, params, device=None):
    idx = 0
    for p in controller.parameters():
        size = p.numel()
        chunk = torch.FloatTensor(params[idx : idx + size]).reshape(p.shape)
        p.data = chunk.to(device) if device is not None else chunk
        idx += size


def rollout(args):
    params, seeds = args
    _load_params(_controller, params, _device)

    rewards = []
    for seed in seeds:
        env = gym.make("CarRacing-v3", continuous=True)
        obs, _ = env.reset(seed=seed)

        h = torch.zeros(1, 1, 256).to(_device)
        c = torch.zeros(1, 1, 256).to(_device)

        r = 0.0
        done = False

        while not done:
            frame = (
                torch.FloatTensor(obs).permute(2, 0, 1).unsqueeze(0).to(_device) / 255.0
            )

            with torch.no_grad():
                _, _, mu, _ = _vae(frame)
                z = mu
                z_in = z.unsqueeze(1)

                a = _controller(z.squeeze(0), h.squeeze(0).squeeze(0))
                a_in = a.unsqueeze(0).unsqueeze(0)

                _, (h, c) = _rnn.lstm(torch.cat([z_in, a_in], dim=-1), (h, c))
            obs, reward, terminated, truncated, _ = env.step(a.cpu().numpy())
            r += reward
            done = terminated or truncated

        env.close()
        rewards.append(r)
    return float(np.mean(rewards))

def evaluate_agent(params, pool, n_workers, n_rollouts, rng):
    """Evaluates a single agent (flat param vector) over
    num_workers * rollouts_per_worker total rollouts, reusing the
    existing worker pool. Mirrors the original paper's evaluation
    approach: rather than picking an arbitrary rollout count (e.g. 100),
    they reused the same per-generation compute budget already
    configured (64 workers x 16 rollouts = 1024), just running it
    1024 times against one fixed agent instead of one agent per worker.
    """
    seeds_per_worker = [
        rng.integers(0, 2**31, size=n_rollouts).tolist()
        for _ in range(n_workers)
    ]
    args = [(params, seeds) for seeds in seeds_per_worker]
    means = pool.map(rollout, args)  # one mean-of-16 per worker
    return float(np.mean(means))


def train_controller(generations=250, population=64, n_rollouts=16, eval_iter=25, workers=None):
    checkpoints.mkdir(exist_ok=True)
    device = "cpu"

    if workers is None:
        workers = mp.cpu_count()

    dummy = MLP()
    n_params = sum(p.numel() for p in dummy.parameters())
    print(
        f"workers: {workers} | population: {population} | "
        f"rollouts/agent: {n_rollouts} | controller params: {n_params}",
        flush=True,
    )

    es = cma.CMAEvolutionStrategy(n_params * [0], 0.1, {"popsize": population})
    rng = np.random.default_rng(42)
    _best = -float("inf")

    scores, gens = [], []  # for plotting the "red line"

    with mp.Pool(
        processes=workers, initializer=init_worker, initargs=(device,)
    ) as pool:
        for gen in range(generations):
            solutions = es.ask()
            seeds = [
                rng.integers(0, 2**31, size=n_rollouts).tolist() for _ in solutions
            ]
            fitnesses = pool.map(rollout, list(zip(solutions, seeds)))
            es.tell(
                solutions, [-f for f in fitnesses]
            )  # CMA-ES minimizes, we want max reward

            best, mean, worst, = max(fitnesses), np.mean(fitnesses), min(fitnesses)
            print(
                f"gen {gen:4d} | best: {best:.1f} | mean: {mean:.1f} | worst: {worst:.1f}",
                flush=True,
            )

            if best > _best:
                _best = best
                params = solutions[fitnesses.index(best)]
                controller = MLP()
                _load_params(controller, params)
                torch.save(controller.state_dict(), checkpoints / "controller.pt")
                print(f" ---> new best: {_best:.1f}", flush=True)

            # Every eval_iter generations, evaluate the current best agent
            # over a much larger rollout count for a more reliable score —
            # matches the paper's methodology for the reported 900.46 result.
            if (gen + 1) % eval_iter == 0:
                score = evaluate_agent(
                    params, pool, workers, n_rollouts, rng
                )
                scores.append(score)
                gens.append(gen + 1)
                rollouts = workers * n_rollouts
                print(
                    f"  -> eval @ gen {gen + 1}: {score:.2f} "
                    f"(avg over {rollouts} rollouts)",
                    flush=True,
                )
    with open("logs/controller_eval.json", "w") as f:
        json.dump({"gens": gens, "scores": scores}, f)

    print(f"training complete | best reward: {_best:.1f}", flush=True)


# --------------------------

if __name__ == "__main__":
    train_controller()