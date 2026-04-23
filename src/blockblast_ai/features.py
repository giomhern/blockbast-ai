from __future__ import annotations

from .game import Board


def board_features(board: Board, lines_cleared: int = 0) -> tuple[float, ...]:
    """Small hand-built state vector, inspired by the tetris-ai after-state approach."""

    size = len(board)
    filled = sum(sum(row) for row in board)
    row_counts = [sum(row) for row in board]
    col_counts = [sum(board[row][col] for row in range(size)) for col in range(size)]
    empty_cells = (size * size) - filled
    nearly_full_rows = sum(1 for count in row_counts if count == size - 1)
    nearly_full_cols = sum(1 for count in col_counts if count == size - 1)
    isolated_empty = _isolated_empty_count(board)
    roughness = _roughness(row_counts) + _roughness(col_counts)

    scale = float(size * size)
    return (
        1.0,
        lines_cleared / (size * 2),
        filled / scale,
        empty_cells / scale,
        max(row_counts, default=0) / size,
        max(col_counts, default=0) / size,
        nearly_full_rows / size,
        nearly_full_cols / size,
        isolated_empty / scale,
        roughness / scale,
    )


def _roughness(counts: list[int]) -> int:
    return sum(abs(left - right) for left, right in zip(counts, counts[1:]))


def _isolated_empty_count(board: Board) -> int:
    size = len(board)
    isolated = 0
    for row in range(size):
        for col in range(size):
            if board[row][col]:
                continue
            neighbors = (
                (row - 1, col),
                (row + 1, col),
                (row, col - 1),
                (row, col + 1),
            )
            blocked = 0
            for neighbor_row, neighbor_col in neighbors:
                if not (0 <= neighbor_row < size and 0 <= neighbor_col < size):
                    blocked += 1
                elif board[neighbor_row][neighbor_col]:
                    blocked += 1
            if blocked == 4:
                isolated += 1
    return isolated
