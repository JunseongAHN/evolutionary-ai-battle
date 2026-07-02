from __future__ import annotations

import json
import pathlib
import sys


EXPERIMENT_ROOT = pathlib.Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parent
for path in (EXPERIMENT_ROOT, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.run_cpc_evaluation_only_bot import main


def test_bot_only_runner_controls_player_and_ally(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_cpc_evaluation_only_bot.py",
            "--no-gui",
            "--steps",
            "2",
            "--print-debug",
            "--output-dir",
            str(tmp_path),
        ],
    )

    main()
    output_lines = capsys.readouterr().out.splitlines()

    session_dirs = [path for path in tmp_path.iterdir() if path.is_dir()]
    assert len(session_dirs) == 1
    rows = [
        json.loads(line)
        for line in (session_dirs[0] / "trajectory.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    questionnaire = json.loads(
        (session_dirs[0] / "questionnaire.json").read_text(encoding="utf-8")
    )

    assert len(rows) == 2
    assert any(row["human_action"]["move"] != 0 for row in rows)
    assert all("layers" in row for row in rows)
    assert questionnaire["answers"]["feedback"].startswith("automated bot-only")
    assert "None" not in output_lines
    assert any("layer1.cpc_intent=" in line for line in output_lines)
    assert any("player_bot.intent=" in line for line in output_lines)
