from __future__ import annotations

import argparse
from pathlib import Path

from .agent import LinearAfterStateAgent, play_episode
from .cnn_torch_agent import CnnActionValueAgent
from .game import BlockBlastEnv
from .torch_agent import TorchAfterStateAgent


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a Block Blast after-state learner.")
    parser.add_argument("--agent", choices=("linear", "torch", "cnn"), default="linear")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--save", type=Path)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--hidden-size", type=int, default=64)
    args = parser.parse_args()

    env = BlockBlastEnv(seed=args.seed)
    try:
        if args.agent == "linear":
            agent = LinearAfterStateAgent(epsilon=args.epsilon_start, seed=args.seed)
            save_path = args.save or Path("models/linear-agent.json")
        elif args.agent == "torch":
            agent = TorchAfterStateAgent(
                epsilon=args.epsilon_start,
                seed=args.seed,
                hidden_size=args.hidden_size,
            )
            save_path = args.save or Path("models/torch-agent.pt")
        else:
            agent = CnnActionValueAgent(
                epsilon=args.epsilon_start,
                seed=args.seed,
                hidden_size=args.hidden_size,
            )
            save_path = args.save or Path("models/cnn-agent.pt")
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    best_score = float("-inf")

    for episode in range(1, args.episodes + 1):
        progress = episode / max(args.episodes, 1)
        agent.epsilon = args.epsilon_start + (args.epsilon_end - args.epsilon_start) * progress
        score, moves, loss = play_episode(env, agent, train=True, batch_size=args.batch_size)
        best_score = max(best_score, score)
        if episode == 1 or episode % 25 == 0:
            print(
                f"episode={episode:04d} score={score:7.1f} "
                f"best={best_score:7.1f} moves={moves:4d} "
                f"epsilon={agent.epsilon:.3f} loss={loss:.4f}"
            )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    agent.epsilon = 0.0
    agent.save(save_path)
    print(f"saved={save_path}")


if __name__ == "__main__":
    main()
