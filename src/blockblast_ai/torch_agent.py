from __future__ import annotations

import random
from collections import deque
from pathlib import Path

from .agent import Transition
from .features import board_features
from .game import Action, BlockBlastEnv


def _torch():
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise RuntimeError(
            "The torch agent requires PyTorch. Install it with "
            "`python -m pip install -e '.[torch]'` or `python -m pip install torch`."
        ) from exc
    return torch, nn


class TorchAfterStateAgent:
    """Neural after-state Q learner for Block Blast."""

    def __init__(
        self,
        *,
        learning_rate: float = 0.001,
        discount: float = 0.95,
        epsilon: float = 1.0,
        seed: int | None = None,
        hidden_size: int = 64,
        memory_size: int = 20_000,
    ) -> None:
        torch, nn = _torch()
        if seed is not None:
            torch.manual_seed(seed)
        self.learning_rate = learning_rate
        self.discount = discount
        self.epsilon = epsilon
        self.hidden_size = hidden_size
        self.random = random.Random(seed)
        self.memory: deque[Transition] = deque(maxlen=memory_size)
        input_size = len(board_features(BlockBlastEnv.empty_board()))
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

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
        torch, _ = _torch()
        batch = self.random.sample(list(self.memory), min(batch_size, len(self.memory)))
        features = torch.tensor([transition.state_features for transition in batch], dtype=torch.float32)
        targets = torch.tensor(
            [
                transition.reward
                if transition.done
                else transition.reward + self.discount * transition.next_value
                for transition in batch
            ],
            dtype=torch.float32,
        ).unsqueeze(1)

        predictions = self.model(features)
        loss = self.loss_fn(predictions, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.detach().item())

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
        torch, _ = _torch()
        with torch.no_grad():
            tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
            return float(self.model(tensor).squeeze().item())

    def save(self, path: str | Path) -> None:
        torch, _ = _torch()
        model_path = Path(path)
        payload = {
            "agent_type": "torch",
            "learning_rate": self.learning_rate,
            "discount": self.discount,
            "epsilon": self.epsilon,
            "hidden_size": self.hidden_size,
            "state_dict": self.model.state_dict(),
        }
        torch.save(payload, model_path)

    @classmethod
    def load(cls, path: str | Path) -> "TorchAfterStateAgent":
        torch, _ = _torch()
        payload = torch.load(Path(path), map_location="cpu")
        agent = cls(
            learning_rate=payload["learning_rate"],
            discount=payload["discount"],
            epsilon=payload.get("epsilon", 0.0),
            hidden_size=payload.get("hidden_size", 64),
        )
        agent.model.load_state_dict(payload["state_dict"])
        agent.model.eval()
        return agent
