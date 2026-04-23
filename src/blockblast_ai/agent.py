from __future__ import annotations

import json
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .features import board_features
from .game import Action, BlockBlastEnv


@dataclass(frozen=True, slots=True)
class Transition:
    state_features: tuple[float, ...]
    reward: float
    next_value: float
    done: bool


class LinearAfterStateAgent:
    """A dependency-light Q learner that values resulting board states."""

    def __init__(
        self,
        *,
        learning_rate: float = 0.03,
        discount: float = 0.95,
        epsilon: float = 1.0,
        seed: int | None = None,
        memory_size: int = 20_000,
    ) -> None:
        self.learning_rate = learning_rate
        self.discount = discount
        self.epsilon = epsilon
        self.random = random.Random(seed)
        self.weights = [0.0 for _ in board_features(BlockBlastEnv.empty_board())]
        self.memory: deque[Transition] = deque(maxlen=memory_size)

    def choose_action(self, env: BlockBlastEnv, explore: bool = True) -> tuple[Action, tuple[float, ...], float]:
        actions = env.legal_actions()
        if not actions:
            raise ValueError("Cannot choose an action when no legal actions exist.")

        if explore and self.random.random() < self.epsilon:
            action = self.random.choice(actions)
            board, _, reward, lines_cleared, _ = env.simulate(action)
            features = board_features(board, lines_cleared)
            return action, features, reward

        best_action = actions[0]
        best_features: tuple[float, ...] | None = None
        best_reward = 0.0
        best_score = float("-inf")
        for action in actions:
            board, _, reward, lines_cleared, _ = env.simulate(action)
            features = board_features(board, lines_cleared)
            score = reward + self.discount * self.value(features)
            if score > best_score:
                best_action = action
                best_features = features
                best_reward = reward
                best_score = score
        assert best_features is not None
        return best_action, best_features, best_reward

    def observe(self, transition: Transition) -> None:
        self.memory.append(transition)

    def replay(self, batch_size: int = 256) -> float:
        if not self.memory:
            return 0.0
        batch = self.random.sample(list(self.memory), min(batch_size, len(self.memory)))
        total_loss = 0.0
        for transition in batch:
            target = transition.reward
            if not transition.done:
                target += self.discount * transition.next_value
            prediction = self.value(transition.state_features)
            error = target - prediction
            total_loss += error * error
            for idx, feature in enumerate(transition.state_features):
                self.weights[idx] += self.learning_rate * error * feature
        return total_loss / len(batch)

    def best_next_value(self, env: BlockBlastEnv) -> float:
        actions = env.legal_actions()
        if not actions:
            return 0.0
        return max(
            reward + self.discount * self.value(board_features(board, lines_cleared))
            for action in actions
            for board, _, reward, lines_cleared, _ in [env.simulate(action)]
        )

    def value(self, features: tuple[float, ...]) -> float:
        return sum(weight * feature for weight, feature in zip(self.weights, features))

    def save(self, path: str | Path) -> None:
        payload = {
            "agent_type": "linear",
            "learning_rate": self.learning_rate,
            "discount": self.discount,
            "epsilon": self.epsilon,
            "weights": self.weights,
        }
        Path(path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "LinearAfterStateAgent":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        agent = cls(
            learning_rate=payload["learning_rate"],
            discount=payload["discount"],
            epsilon=payload.get("epsilon", 0.0),
        )
        agent.weights = list(payload["weights"])
        return agent


class AfterStateAgent(Protocol):
    epsilon: float

    def choose_action(self, env: BlockBlastEnv, explore: bool = True) -> tuple[Action, tuple[float, ...], float]:
        ...

    def observe(self, transition: Transition) -> None:
        ...

    def replay(self, batch_size: int = 256) -> float:
        ...

    def best_next_value(self, env: BlockBlastEnv) -> float:
        ...

    def save(self, path: str | Path) -> None:
        ...


def load_agent(path: str | Path) -> AfterStateAgent:
    model_path = Path(path)
    if model_path.suffix in {".pt", ".pth"}:
        from .torch_agent import _torch

        torch, _ = _torch()
        payload = torch.load(model_path, map_location="cpu")
        agent_type = payload.get("agent_type", "torch")
        if agent_type == "torch":
            from .torch_agent import TorchAfterStateAgent

            return TorchAfterStateAgent.load(model_path)
        if agent_type == "cnn":
            from .cnn_torch_agent import CnnActionValueAgent

            return CnnActionValueAgent.load(model_path)
        raise ValueError(f"Unknown torch agent type in {model_path}: {agent_type}")

    payload = json.loads(model_path.read_text(encoding="utf-8"))
    agent_type = payload.get("agent_type", "linear")
    if agent_type == "linear":
        return LinearAfterStateAgent.load(model_path)
    if agent_type == "torch":
        from .torch_agent import TorchAfterStateAgent

        return TorchAfterStateAgent.load(model_path)
    if agent_type == "cnn":
        from .cnn_torch_agent import CnnActionValueAgent

        return CnnActionValueAgent.load(model_path)
    raise ValueError(f"Unknown agent type in {model_path}: {agent_type}")


def play_episode(
    env: BlockBlastEnv,
    agent: AfterStateAgent,
    *,
    train: bool,
    batch_size: int = 256,
    max_moves: int = 10_000,
) -> tuple[float, int, float]:
    total_loss = 0.0
    losses = 0
    env.reset()

    while env.legal_actions() and env.moves < max_moves:
        action, features, predicted_reward = agent.choose_action(env, explore=train)
        result = env.step(action)
        next_value = agent.best_next_value(env)
        if train:
            agent.observe(
                Transition(
                    state_features=features,
                    reward=result.reward if result.reward != predicted_reward else predicted_reward,
                    next_value=next_value,
                    done=result.done,
                )
            )
            total_loss += agent.replay(batch_size)
            losses += 1

    return env.score, env.moves, total_loss / max(losses, 1)
