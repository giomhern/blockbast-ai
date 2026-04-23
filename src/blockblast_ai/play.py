from __future__ import annotations

import argparse
from pathlib import Path

from .agent import load_agent, play_episode
from .game import BlockBlastEnv


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained Block Blast agent.")
    parser.add_argument("model", type=Path)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=101)
    args = parser.parse_args()

    env = BlockBlastEnv(seed=args.seed)
    try:
        agent = load_agent(args.model)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    agent.epsilon = 0.0

    scores = []
    for episode in range(1, args.episodes + 1):
        score, moves, _ = play_episode(env, agent, train=False)
        scores.append(score)
        print(f"episode={episode:03d} score={score:7.1f} moves={moves:4d}")

    print(f"average={sum(scores) / len(scores):.1f} best={max(scores):.1f}")


if __name__ == "__main__":
    main()
