from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Piece:
    """A placeable Block Blast piece represented by occupied (row, col) cells."""

    name: str
    cells: tuple[tuple[int, int], ...]

    @property
    def height(self) -> int:
        return max(row for row, _ in self.cells) + 1

    @property
    def width(self) -> int:
        return max(col for _, col in self.cells) + 1


def _piece(name: str, rows: tuple[str, ...]) -> Piece:
    cells = tuple(
        (row_idx, col_idx)
        for row_idx, row in enumerate(rows)
        for col_idx, value in enumerate(row)
        if value == "1"
    )
    return Piece(name=name, cells=cells)


PIECES: tuple[Piece, ...] = (
    _piece("single", ("1",)),
    _piece("domino_h", ("11",)),
    _piece("domino_v", ("1", "1")),
    _piece("triomino_h", ("111",)),
    _piece("triomino_v", ("1", "1", "1")),
    _piece("square_2", ("11", "11")),
    _piece("line_4_h", ("1111",)),
    _piece("line_4_v", ("1", "1", "1", "1")),
    _piece("line_5_h", ("11111",)),
    _piece("line_5_v", ("1", "1", "1", "1", "1")),
    _piece("l_3", ("10", "11")),
    _piece("l_3_mirror", ("01", "11")),
    _piece("corner_3", ("11", "10")),
    _piece("corner_3_mirror", ("11", "01")),
    _piece("t_4", ("111", "010")),
    _piece("t_4_down", ("010", "111")),
    _piece("z_4", ("110", "011")),
    _piece("s_4", ("011", "110")),
    _piece("rect_2x3", ("111", "111")),
    _piece("rect_3x2", ("11", "11", "11")),
    _piece("square_3", ("111", "111", "111")),
)
