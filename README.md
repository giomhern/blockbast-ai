# Blockblast AI

Reinforcement-learning experiments for Block Blast.

The project is inspired by [nuno-faria/tetris-ai](https://github.com/nuno-faria/tetris-ai): instead of predicting button presses directly, the agents enumerate all legal placements, simulate the result of each move, and learn which move leads to the best long-term score.

## What is in the repo

- `BlockBlastEnv`: an 8x8 Block Blast-style simulator with piece trays, placement checks, row/column clears, scoring, and terminal detection
- `LinearAfterStateAgent`: a lightweight baseline that scores handcrafted board features
- `TorchAfterStateAgent`: a neural version of the after-state feature agent
- `CnnActionValueAgent`: a CNN that scores board-plus-action tensors directly
- `render.py`: generates a clean 2D HTML replay viewer for saved models

## Project Layout

```text
src/blockblast_ai/game.py          simulator
src/blockblast_ai/pieces.py        piece definitions
src/blockblast_ai/agent.py         linear agent + shared training loop
src/blockblast_ai/torch_agent.py   feature-based PyTorch agent
src/blockblast_ai/cnn_torch_agent.py CNN action-value PyTorch agent
src/blockblast_ai/train.py         training CLI
src/blockblast_ai/play.py          evaluation CLI
src/blockblast_ai/render.py        replay HTML generator
src/models/                       saved model artifacts
src/replays/                      generated replay HTML files
tests/test_game.py                 simulator tests
```

## Setup

From the repo root:

```bash
cd /Users/giomhern/04/blockbast-ai
```

Recommended: use a virtual environment or Conda environment first.

### Minimal install

For the simulator, linear agent, tests, and replay viewer:

```bash
python -m pip install -e .
```

### PyTorch install

For the Torch and CNN agents:

```bash
python -m pip install -e '.[torch]'
```

If PyTorch installation is finicky on your machine, install it with the recommended command from the official PyTorch site inside your environment, then run:

```bash
python -m pip install -e .
```

## Running

If you installed the package with `pip install -e .`, you can run the modules directly:

```bash
python -m blockblast_ai.train --episodes 500
```

If you are skipping the editable install, prefix commands with:

```bash
PYTHONPATH=src
```

Examples below use that explicit form because it works reliably in-place.

The repo currently stores generated artifacts under:

```text
src/models/
src/replays/
```

## Train Agents

### 1. Linear baseline

This is the simplest and most stable starting point.

```bash
PYTHONPATH=src python -m blockblast_ai.train \
  --agent linear \
  --episodes 500 \
  --save src/models/linear-agent.json
```

### 2. Feature-based Torch agent

This uses the same handcrafted board features as the linear agent, but learns them with a small neural network.

```bash
PYTHONPATH=src python -m blockblast_ai.train \
  --agent torch \
  --episodes 1000 \
  --hidden-size 64 \
  --save src/models/torch-agent.pt
```

### 3. CNN action-value agent

This agent sees:

- channel 1: the current board
- channel 2: the candidate placement mask

and learns a value for each legal move.

```bash
PYTHONPATH=src python -m blockblast_ai.train \
  --agent cnn \
  --episodes 1000 \
  --hidden-size 128 \
  --save src/models/cnn-agent.pt
```

Note: the CNN path is more experimental right now and may perform worse than the linear baseline without longer training or additional DQN stabilizers.

## Evaluate Saved Models

```bash
PYTHONPATH=src python -m blockblast_ai.play src/models/linear-agent.json --episodes 20 --seed 101
PYTHONPATH=src python -m blockblast_ai.play src/models/torch-agent.pt --episodes 20 --seed 101
PYTHONPATH=src python -m blockblast_ai.play src/models/cnn-agent.pt --episodes 20 --seed 101
```

The CLI prints per-episode scores plus average and best score at the end.

## Render a Replay

Generate an HTML replay from any saved model:

```bash
PYTHONPATH=src python -m blockblast_ai.render src/models/linear-agent.json --output src/replays/linear.html
PYTHONPATH=src python -m blockblast_ai.render src/models/torch-agent.pt --output src/replays/torch.html
PYTHONPATH=src python -m blockblast_ai.render src/models/cnn-agent.pt --output src/replays/cnn.html
```

Then open the generated HTML file in your browser.

The replay viewer shows:

- the board after each move
- the tray after the move
- move number, reward, and score
- play/pause, prev/next, and scrubber controls
- a legal-move status message so it is clear why a game ended

## Testing

Run the test suite:

```bash
PYTHONPATH=src pytest
```

## How the Agents Differ

### Linear

- Input: handcrafted board features
- Output: one scalar board value
- Pros: fast, simple, dependency-light
- Cons: limited by the feature design

### Torch

- Input: the same handcrafted board features
- Output: one scalar board value from a small MLP
- Pros: neural baseline without changing the state representation too much
- Cons: still depends on manual features

### CNN

- Input: raw board plus candidate placement mask
- Output: one scalar Q-value for that move
- Pros: can learn spatial structure directly
- Cons: harder to train and currently less stable

## Known Limitations

- The simulator is a simplified Block Blast-style environment, not an exact clone of every production/mobile variant
- The Torch and CNN agents do not yet use target networks or Double DQN
- The CNN agent is early-stage and may underperform the linear baseline
- There is not yet a benchmark script that automatically compares agents across fixed seeds

## Good First Checks

If you want to confirm learning is happening, compare models trained for different numbers of episodes on the same evaluation seed:

```bash
PYTHONPATH=src python -m blockblast_ai.train --agent linear --episodes 50 --save src/models/linear-50.json
PYTHONPATH=src python -m blockblast_ai.train --agent linear --episodes 500 --save src/models/linear-500.json

PYTHONPATH=src python -m blockblast_ai.play src/models/linear-50.json --episodes 100 --seed 123
PYTHONPATH=src python -m blockblast_ai.play src/models/linear-500.json --episodes 100 --seed 123
```

Do the same for the Torch or CNN agents and compare average score.

## Next Useful Improvements

- add a target network for the Torch and CNN agents
- add periodic evaluation during training
- log training curves
- tune reward shaping and piece distribution
- benchmark the agents on fixed seed sets
