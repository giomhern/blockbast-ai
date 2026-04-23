from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from textwrap import dedent

from .agent import load_agent
from .game import Action, Board, BlockBlastEnv
from .pieces import Piece


def build_replay(
    model_path: Path,
    *,
    seed: int = 101,
    max_moves: int = 250,
) -> dict[str, object]:
    env = BlockBlastEnv(seed=seed)
    try:
        agent = load_agent(model_path)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    agent.epsilon = 0.0

    frames = []
    while env.legal_actions() and env.moves < max_moves:
        board_before = env.board
        tray_before = env.tray
        score_before = env.score
        action, _, _ = agent.choose_action(env, explore=False)
        chosen_piece = tray_before[action.piece_index]
        result = env.step(action)
        legal_moves_after = len(env.legal_actions())
        frames.append(
            {
                "move": env.moves,
                "boardBefore": _board_to_lists(board_before),
                "boardAfter": _board_to_lists(env.board),
                "trayBefore": [_piece_to_dict(piece) for piece in tray_before],
                "trayAfter": [_piece_to_dict(piece) for piece in env.tray],
                "action": _action_to_dict(action),
                "piece": _piece_to_dict(chosen_piece),
                "reward": result.reward,
                "linesCleared": result.lines_cleared,
                "scoreBefore": score_before,
                "scoreAfter": env.score,
                "done": result.done,
                "legalMovesAfter": legal_moves_after,
            }
        )

    stop_reason = "max_moves" if env.moves >= max_moves and env.legal_actions() else "game_over"
    return {
        "size": env.size,
        "seed": seed,
        "model": str(model_path),
        "score": env.score,
        "moves": env.moves,
        "stopReason": stop_reason,
        "legalMoves": len(env.legal_actions()),
        "finalTray": [_piece_to_dict(piece) for piece in env.tray],
        "finalBoard": _board_to_lists(env.board),
        "frames": frames,
    }


def write_replay_html(replay: dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(replay)
    output_path.write_text(_html_document(payload), encoding="utf-8")


def _board_to_lists(board: Board) -> list[list[int]]:
    return [list(row) for row in board]


def _piece_to_dict(piece: Piece) -> dict[str, object]:
    return {
        "name": piece.name,
        "height": piece.height,
        "width": piece.width,
        "cells": [list(cell) for cell in piece.cells],
    }


def _action_to_dict(action: Action) -> dict[str, int]:
    return {
        "pieceIndex": action.piece_index,
        "row": action.row,
        "col": action.col,
    }


def _html_document(payload: str) -> str:
    escaped_payload = html.escape(payload, quote=False)
    return dedent(
        f"""\
        <!doctype html>
        <html lang="en">
        <head>
          <meta charset="utf-8">
          <meta name="viewport" content="width=device-width, initial-scale=1">
          <title>Block Blast AI Replay</title>
          <script src="https://cdn.tailwindcss.com"></script>
          <style>
            .board {{
              display: grid;
              grid-template-columns: repeat(var(--size), 1fr);
            }}
            .mini-grid {{
              display: grid;
              grid-template-columns: repeat(var(--piece-width), 1rem);
            }}
          </style>
        </head>
        <body class="min-h-screen bg-slate-50 text-slate-900 antialiased">
          <main class="mx-auto grid min-h-screen max-w-6xl gap-6 px-4 py-5 sm:px-6 lg:grid-cols-[minmax(320px,560px)_1fr] lg:items-start lg:px-8 lg:py-8">
            <section class="w-full">
              <div class="mb-4">
                <h1 class="text-2xl font-semibold tracking-normal text-slate-950">Block Blast AI Replay</h1>
                <p class="mt-1 text-sm text-slate-500" id="summary"></p>
              </div>
              <div class="board aspect-square w-full gap-1.5 rounded-lg border border-slate-300 bg-slate-200 p-2 shadow-sm" id="board"></div>
            </section>
            <section class="grid gap-4">
              <div class="grid grid-cols-3 gap-3">
                <div class="rounded-lg border border-slate-200 bg-white p-3 shadow-sm">
                  <div class="text-xs font-medium uppercase tracking-wide text-slate-500">Move</div>
                  <div class="mt-1 text-2xl font-semibold text-slate-950" id="move">0</div>
                </div>
                <div class="rounded-lg border border-slate-200 bg-white p-3 shadow-sm">
                  <div class="text-xs font-medium uppercase tracking-wide text-slate-500">Score</div>
                  <div class="mt-1 text-2xl font-semibold text-slate-950" id="score">0</div>
                </div>
                <div class="rounded-lg border border-slate-200 bg-white p-3 shadow-sm">
                  <div class="text-xs font-medium uppercase tracking-wide text-slate-500">Reward</div>
                  <div class="mt-1 text-2xl font-semibold text-slate-950" id="reward">0</div>
                </div>
              </div>

              <div class="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
                <div class="flex items-center gap-2">
                  <button class="h-10 rounded-md border border-slate-300 px-3 text-sm font-medium text-slate-700 hover:bg-slate-50" id="prev" aria-label="Previous move">Prev</button>
                  <input class="h-10 min-w-0 flex-1 accent-emerald-600" id="scrubber" type="range" min="0" value="0">
                  <button class="h-10 rounded-md border border-slate-300 px-3 text-sm font-medium text-slate-700 hover:bg-slate-50" id="next" aria-label="Next move">Next</button>
                  <button class="h-10 rounded-md bg-slate-900 px-4 text-sm font-semibold text-white hover:bg-slate-700" id="play" aria-label="Play replay">Play</button>
                </div>
              </div>

              <div>
                <div class="mb-2 text-xs font-medium uppercase tracking-wide text-slate-500">Tray after move</div>
                <div class="grid grid-cols-1 gap-3 sm:grid-cols-3" id="tray"></div>
              </div>

              <div class="rounded-lg border border-slate-200 bg-white p-4 text-sm leading-6 text-slate-700 shadow-sm" id="moveText"></div>
            </section>
          </main>
          <script id="replay-data" type="application/json">{escaped_payload}</script>
          <script>
            const replay = JSON.parse(document.getElementById("replay-data").textContent);
            const boardEl = document.getElementById("board");
            const trayEl = document.getElementById("tray");
            const scrubber = document.getElementById("scrubber");
            const frames = replay.frames;
            let index = 0;
            let timer = null;

            boardEl.style.setProperty("--size", replay.size);
            scrubber.max = Math.max(frames.length - 1, 0);
            document.getElementById("summary").textContent =
              `Seed ${{replay.seed}} | ${{replay.moves}} moves | final score ${{Math.round(replay.score)}}`;

            function placedCells(frame) {{
              const cells = new Set();
              frame.piece.cells.forEach(([row, col]) => {{
                cells.add(`${{frame.action.row + row}},${{frame.action.col + col}}`);
              }});
              return cells;
            }}

            function renderBoard(frame) {{
              boardEl.innerHTML = "";
              const placed = placedCells(frame);
              frame.boardAfter.forEach((row, rowIndex) => {{
                row.forEach((value, colIndex) => {{
                  const cell = document.createElement("div");
                  cell.className = "rounded border";
                  if (value) {{
                    cell.classList.add("border-emerald-700", "bg-emerald-600");
                  }} else {{
                    cell.classList.add("border-slate-300", "bg-white");
                  }}
                  if (placed.has(`${{rowIndex}},${{colIndex}}`)) {{
                    cell.className = "rounded border border-amber-600 bg-amber-400";
                  }}
                  boardEl.appendChild(cell);
                }});
              }});
            }}

            function renderTray(frame) {{
              trayEl.innerHTML = "";
              const tray = frame.trayAfter.length ? frame.trayAfter : replay.finalTray;
              tray.forEach((piece) => {{
                const wrapper = document.createElement("div");
                wrapper.className = "grid min-h-28 place-items-center gap-2 rounded-lg border border-slate-200 bg-white p-3 shadow-sm";
                const mini = document.createElement("div");
                mini.className = "mini-grid place-content-center gap-1";
                mini.style.setProperty("--piece-width", piece.width);
                const filled = new Set(piece.cells.map(([row, col]) => `${{row}},${{col}}`));
                for (let row = 0; row < piece.height; row += 1) {{
                  for (let col = 0; col < piece.width; col += 1) {{
                    const cell = document.createElement("div");
                    cell.className = "h-4 w-4 rounded-sm";
                    if (filled.has(`${{row}},${{col}}`)) {{
                      cell.classList.add("border", "border-emerald-700", "bg-emerald-600");
                    }}
                    mini.appendChild(cell);
                  }}
                }}
                const name = document.createElement("div");
                name.className = "text-center text-xs text-slate-500";
                name.textContent = piece.name;
                wrapper.append(mini, name);
                trayEl.appendChild(wrapper);
              }});
            }}

            function render() {{
              if (!frames.length) return;
              const frame = frames[index];
              renderBoard(frame);
              renderTray(frame);
              document.getElementById("move").textContent = frame.move;
              document.getElementById("score").textContent = Math.round(frame.scoreAfter);
              document.getElementById("reward").textContent = Math.round(frame.reward);
              const status =
                frame.legalMovesAfter === 0
                  ? "Game over: no remaining tray piece fits the board."
                  : `${{frame.legalMovesAfter}} legal moves remain.`;
              document.getElementById("moveText").textContent =
                `Placed ${{frame.piece.name}} at row ${{frame.action.row + 1}}, column ${{frame.action.col + 1}}. ` +
                `${{frame.linesCleared}} lines cleared. ${{status}}`;
              scrubber.value = index;
            }}

            function stop() {{
              if (timer) clearInterval(timer);
              timer = null;
              document.getElementById("play").textContent = "Play";
            }}

            document.getElementById("prev").addEventListener("click", () => {{
              stop();
              index = Math.max(index - 1, 0);
              render();
            }});
            document.getElementById("next").addEventListener("click", () => {{
              stop();
              index = Math.min(index + 1, frames.length - 1);
              render();
            }});
            scrubber.addEventListener("input", () => {{
              stop();
              index = Number(scrubber.value);
              render();
            }});
            document.getElementById("play").addEventListener("click", () => {{
              if (timer) {{
                stop();
                return;
              }}
              document.getElementById("play").textContent = "Pause";
              timer = setInterval(() => {{
                if (index >= frames.length - 1) {{
                  stop();
                  return;
                }}
                index += 1;
                render();
              }}, 450);
            }});

            render();
          </script>
        </body>
        </html>
        """
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a trained Block Blast agent as an HTML replay.")
    parser.add_argument("model", type=Path)
    parser.add_argument("--output", type=Path, default=Path("replays/latest.html"))
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--max-moves", type=int, default=250)
    args = parser.parse_args()

    replay = build_replay(args.model, seed=args.seed, max_moves=args.max_moves)
    write_replay_html(replay, args.output)
    print(f"wrote={args.output} moves={replay['moves']} score={replay['score']:.1f}")


if __name__ == "__main__":
    main()
