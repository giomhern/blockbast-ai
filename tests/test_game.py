from blockblast_ai.agent import LinearAfterStateAgent, load_agent, play_episode
from blockblast_ai.game import Action, BlockBlastEnv
from blockblast_ai.pieces import Piece


def test_place_and_clear_full_row() -> None:
    env = BlockBlastEnv(seed=1)
    env.board = tuple(
        tuple(1 if row == 0 and col < 7 else 0 for col in range(env.size))
        for row in range(env.size)
    )
    env.tray = (Piece("single", ((0, 0),)),)

    result = env.step(Action(piece_index=0, row=0, col=7))

    assert result.lines_cleared == 1
    assert sum(env.board[0]) == 0
    assert result.reward == 9


def test_legal_actions_include_piece_index_and_position() -> None:
    env = BlockBlastEnv(seed=2)
    env.tray = (Piece("domino_h", ((0, 0), (0, 1))),)

    actions = env.legal_actions()

    assert Action(piece_index=0, row=0, col=0) in actions
    assert Action(piece_index=0, row=7, col=6) in actions
    assert Action(piece_index=0, row=7, col=7) not in actions


def test_legal_action_counts_reports_each_tray_piece() -> None:
    env = BlockBlastEnv(seed=2)
    env.tray = (
        Piece("single", ((0, 0),)),
        Piece("domino_h", ((0, 0), (0, 1))),
    )

    counts = env.legal_action_counts()

    assert counts == {"0:single": 64, "1:domino_h": 56}


def test_episode_runs_with_learning_agent() -> None:
    env = BlockBlastEnv(seed=3)
    agent = LinearAfterStateAgent(seed=3, epsilon=0.2)

    score, moves, loss = play_episode(env, agent, train=True, batch_size=8)

    assert moves > 0
    assert isinstance(score, float)
    assert loss >= 0.0


def test_linear_agent_round_trip_load(tmp_path) -> None:
    model_path = tmp_path / "linear-agent.json"
    agent = LinearAfterStateAgent(seed=4, epsilon=0.3)
    agent.save(model_path)

    loaded = load_agent(model_path)

    assert isinstance(loaded, LinearAfterStateAgent)
    assert loaded.epsilon == 0.3
