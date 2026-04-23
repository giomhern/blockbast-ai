from __future__ import annotations

import random
from collections import deque
from pathlib import Path

from .agent import Transition
from .game import Action, Board, BlockBlastEnv
from .torch_agent import _torch


class CnnActionValueAgent:
    """CNN DQN that scores a board plus one candidate placement mask."""

    channels = 2

    def __init__(
        self,
        *,
        learning_rate: float = 0.001,
        discount: float = 0.95,
        epsilon: float = 1.0,
        seed: int | None = None,
        hidden_size: int = 128,
        memory_size: int = 20_000,
        board_size: int = 8,
    ) -> None:
        torch, nn = _torch()
        if seed is not None:
            torch.manual_seed(seed)
        self.learning_rate = learning_rate
        self.discount = discount
        self.epsilon = epsilon
        self.hidden_size = hidden_size
        self.board_size = board_size
        self.random = random.Random(seed)
        self.memory: deque[Transition] = deque(maxlen=memory_size)
        self.model = nn.Sequential(
            nn.Conv2d(self.channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * board_size * board_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

    def choose_action(self, env: BlockBlastEnv, explore: bool = True) -> tuple[Action, tuple[float, ...], float]:
        self.bind_env(env)
        actions = env.legal_actions()
        if not actions:
            raise ValueError("Cannot choose an action when no legal actions exist.")

        if explore and self.random.random() < self.epsilon:
            action = self.random.choice(actions)
            _, _, reward, _, _ = env.simulate(action)
            return action, self.action_features(env.board, action), reward

        best_action = actions[0]
        best_features: tuple[float, ...] | None = None
        best_reward = 0.0
        best_score = float("-inf")
        for action in actions:
            _, _, reward, _, _ = env.simulate(action)
            features = self.action_features(env.board, action)
            score = self.value(features)
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
        features = torch.tensor(
            [transition.state_features for transition in batch],
            dtype=torch.float32,
        ).reshape(-1, self.channels, self.board_size, self.board_size)
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
        self.bind_env(env)
        actions = env.legal_actions()
        if not actions:
            return 0.0
        return max(
            self.value(self.action_features(env.board, action))
            for action in actions
        )

    def value(self, features: tuple[float, ...]) -> float:
        torch, _ = _torch()
        with torch.no_grad():
            tensor = torch.tensor(features, dtype=torch.float32).reshape(
                1,
                self.channels,
                self.board_size,
                self.board_size,
            )
            return float(self.model(tensor).squeeze().item())

    def action_features(self, board: Board, action: Action) -> tuple[float, ...]:
        piece = self._piece_for_action(action)
        placement = [[0.0 for _ in range(self.board_size)] for _ in range(self.board_size)]
        for cell_row, cell_col in piece.cells:
            placement[action.row + cell_row][action.col + cell_col] = 1.0
        board_channel = [float(value) for row in board for value in row]
        placement_channel = [value for row in placement for value in row]
        return tuple(board_channel + placement_channel)

    def bind_env(self, env: BlockBlastEnv) -> None:
        self._current_tray = env.tray

    def _piece_for_action(self, action: Action):
        return self._current_tray[action.piece_index]

    def save(self, path: str | Path) -> None:
        torch, _ = _torch()
        payload = {
            "agent_type": "cnn",
            "learning_rate": self.learning_rate,
            "discount": self.discount,
            "epsilon": self.epsilon,
            "hidden_size": self.hidden_size,
            "board_size": self.board_size,
            "state_dict": self.model.state_dict(),
        }
        torch.save(payload, Path(path))

    @classmethod
    def load(cls, path: str | Path) -> "CnnActionValueAgent":
        torch, _ = _torch()
        payload = torch.load(Path(path), map_location="cpu")
        agent = cls(
            learning_rate=payload["learning_rate"],
            discount=payload["discount"],
            epsilon=payload.get("epsilon", 0.0),
            hidden_size=payload.get("hidden_size", 128),
            board_size=payload.get("board_size", 8),
        )
        agent.model.load_state_dict(payload["state_dict"])
        agent.model.eval()
        return agent
