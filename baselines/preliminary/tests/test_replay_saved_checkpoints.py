from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from baselines.preliminary.replay_saved_checkpoints import filter_checkpoints, run


class ReplaySavedCheckpointsTests(unittest.TestCase):
    def test_filter_checkpoints_skips_existing_summary(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            output_dir = root / "out"
            (output_dir / "per_checkpoint" / "global_step_20").mkdir(parents=True)
            (output_dir / "per_checkpoint" / "global_step_20" / "summary.json").write_text("[]", encoding="utf-8")
            checkpoints = [
                {
                    "checkpoint_path": "/tmp/a",
                    "checkpoint_name": "global_step_20",
                    "_sort_key": (20, "global_step_20"),
                },
                {
                    "checkpoint_path": "/tmp/b",
                    "checkpoint_name": "global_step_40",
                    "_sort_key": (40, "global_step_40"),
                },
            ]
            selected = filter_checkpoints(
                checkpoints,
                output_dir=output_dir,
                min_step=-1,
                max_step=-1,
                force=False,
            )
            self.assertEqual([item["checkpoint_name"] for item in selected], ["global_step_40"])

    def test_filter_checkpoints_applies_step_range(self) -> None:
        checkpoints = [
            {"checkpoint_path": "/tmp/a", "checkpoint_name": "global_step_20", "_sort_key": (20, "global_step_20")},
            {"checkpoint_path": "/tmp/b", "checkpoint_name": "global_step_40", "_sort_key": (40, "global_step_40")},
            {"checkpoint_path": "/tmp/c", "checkpoint_name": "global_step_60", "_sort_key": (60, "global_step_60")},
        ]
        selected = filter_checkpoints(
            checkpoints,
            output_dir=None,
            min_step=30,
            max_step=50,
            force=False,
        )
        self.assertEqual([item["checkpoint_name"] for item in selected], ["global_step_40"])

    def test_run_posts_checkpoint_name_and_step(self) -> None:
        args = type(
            "Args",
            (),
            {
                "checkpoint_root": "/tmp/ckpts",
                "server_url": "http://127.0.0.1:8899",
                "output_dir": "",
                "checkpoint_limit": 0,
                "min_step": -1,
                "max_step": -1,
                "force": False,
                "wait": False,
                "wait_timeout": 0.0,
                "poll_interval": 1.0,
                "request_timeout": 3.0,
            },
        )()
        checkpoints = [
            {
                "checkpoint_path": "/tmp/ckpts/global_step_20/actor",
                "checkpoint_name": "global_step_20",
                "_sort_key": (20, "global_step_20__actor"),
            }
        ]
        posted = []

        def fake_read_json(url: str, *, timeout: float):
            return {"ok": True}

        def fake_post_json(url: str, payload, *, timeout: float):
            posted.append((url, payload))
            return {"job": {"checkpoint_name": payload["checkpoint_name"]}}

        with mock.patch("baselines.preliminary.replay_saved_checkpoints.discover_checkpoints", return_value=checkpoints), \
            mock.patch("baselines.preliminary.replay_saved_checkpoints._read_json", side_effect=fake_read_json), \
            mock.patch("baselines.preliminary.replay_saved_checkpoints._post_json", side_effect=fake_post_json):
            rc = run(args)

        self.assertEqual(rc, 0)
        self.assertEqual(len(posted), 1)
        self.assertEqual(posted[0][1]["checkpoint_name"], "global_step_20")
        self.assertEqual(posted[0][1]["global_step"], 20)


if __name__ == "__main__":
    unittest.main()
