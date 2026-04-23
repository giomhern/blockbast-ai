from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable

from .pieces import PIECES, Piece

Board = tuple[tuple[int, ...], ...]


@dataclass(frozen=True, slots=True)
class Action:
    piece_index: int
    row: int
    col: int


@dataclass(frozen=True, slots=True)
class StepResult:
    reward: float
    lines_cleared: int
    done: bool


class BlockBlastEnv:
    """Fast simulator for the core Block Blast placement/clear loop."""

    def __init__(self, size: int = 8, seed: int | None = None) -> None:
        self.size = size
        self.random = random.Random(seed)
        self.board: Board = self.empty_board(size)
        self.tray: tuple[Piece, ...] = ()
        self.score = 0.0
        self.moves = 0
        self.reset()

    @staticmethod
    def empty_board(size: int = 8) -> Board:
        return tuple(tuple(0 for _ in range(size)) for _ in range(size))

    def reset(self) -> Board:
        self.board = self.empty_board(self.size)
        self.tray = self.draw_tray()
        self.score = 0.0
        self.moves = 0
        return self.board

    def draw_tray(self) -> tuple[Piece, ...]:
        return tuple(self.random.choice(PIECES) for _ in range(3))

    def legal_actions(self) -> list[Action]:
        actions: list[Action] = []
        for piece_index, piece in enumerate(self.tray):
            for row in range(self.size - piece.height + 1):
                for col in range(self.size - piece.width + 1):
                    if self.can_place(piece, row, col):
                        actions.append(Action(piece_index, row, col))
        return actions

    def legal_action_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for piece_index, piece in enumerate(self.tray):
            count = 0
            for row in range(self.size - piece.height + 1):
                for col in range(self.size - piece.width + 1):
                    if self.can_place(piece, row, col):
                        count += 1
            counts[f"{piece_index}:{piece.name}"] = count
        return counts

    def can_place(self, piece: Piece, row: int, col: int, board: Board | None = None) -> bool:
        board = self.board if board is None else board
        return all(board[row + cell_row][col + cell_col] == 0 for cell_row, cell_col in piece.cells)

    def step(self, action: Action) -> StepResult:
        if action.piece_index >= len(self.tray):
            raise ValueError(f"Invalid piece index: {action.piece_index}")

        piece = self.tray[action.piece_index]
        if not self.can_place(piece, action.row, action.col):
            raise ValueError(f"Illegal placement: {action}")

        board, lines_cleared = self.place_and_clear(self.board, piece, action.row, action.col)
        remaining = tuple(piece for idx, piece in enumerate(self.tray) if idx != action.piece_index)
        self.tray = remaining if remaining else self.draw_tray()
        self.board = board
        self.moves += 1

        reward = len(piece.cells) + (lines_cleared * lines_cleared * self.size)
        done = not self.legal_actions()
        if done:
            reward -= self.size
        self.score += reward
        return StepResult(reward=reward, lines_cleared=lines_cleared, done=done)

    def simulate(self, action: Action) -> tuple[Board, tuple[Piece, ...], float, int, bool]:
        piece = self.tray[action.piece_index]
        board, lines_cleared = self.place_and_clear(self.board, piece, action.row, action.col)
        remaining = tuple(piece for idx, piece in enumerate(self.tray) if idx != action.piece_index)
        next_tray = remaining
        reward = len(piece.cells) + (lines_cleared * lines_cleared * self.size)
        done = bool(next_tray) and not any(self._legal_actions_for(board, next_tray))
        if done:
            reward -= self.size
        return board, next_tray, reward, lines_cleared, done

    def place_and_clear(self, board: Board, piece: Piece, row: int, col: int) -> tuple[Board, int]:
        mutable = [list(board_row) for board_row in board]
        for cell_row, cell_col in piece.cells:
            mutable[row + cell_row][col + cell_col] = 1

        full_rows = {idx for idx, board_row in enumerate(mutable) if all(board_row)}
        full_cols = {
            col_idx
            for col_idx in range(self.size)
            if all(mutable[row_idx][col_idx] for row_idx in range(self.size))
        }
        for row_idx in full_rows:
            for col_idx in range(self.size):
                mutable[row_idx][col_idx] = 0
        for col_idx in full_cols:
            for row_idx in range(self.size):
                mutable[row_idx][col_idx] = 0
        return tuple(tuple(row_values) for row_values in mutable), len(full_rows) + len(full_cols)

    def _legal_actions_for(self, board: Board, tray: Iterable[Piece]) -> Iterable[Action]:
        for piece_index, piece in enumerate(tray):
            for row in range(self.size - piece.height + 1):
                for col in range(self.size - piece.width + 1):
                    if self.can_place(piece, row, col, board):
                        yield Action(piece_index, row, col)
