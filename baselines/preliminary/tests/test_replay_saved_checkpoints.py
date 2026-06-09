from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from baselines.preliminary.replay_saved_checkpoints import (
    _is_fsdp_actor_dir,
    discover_eval_checkpoints,
    filter_checkpoints,
    run,
)


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
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            hf = root / "global_step_20"
            hf.mkdir(parents=True)
            (hf / "config.json").write_text("{}", encoding="utf-8")
            (hf / "model.safetensors").write_text("", encoding="utf-8")
            args = type(
                "Args",
                (),
                {
                    "checkpoint_root": str(root),
                    "server_url": "http://127.0.0.1:8899",
                    "output_dir": "",
                    "merged_root": "",
                    "checkpoint_limit": 0,
                    "min_step": -1,
                    "max_step": -1,
                    "force": False,
                    "wait": False,
                    "wait_timeout": 0.0,
                    "poll_interval": 1.0,
                    "request_timeout": 3.0,
                    "merge_timeout": 0.0,
                },
            )()
            posted = []

            def fake_read_json(url: str, *, timeout: float):
                return {"ok": True}

            def fake_post_json(url: str, payload, *, timeout: float):
                posted.append((url, payload))
                return {"job": {"checkpoint_name": payload["checkpoint_name"]}}

            with mock.patch("baselines.preliminary.replay_saved_checkpoints._read_json", side_effect=fake_read_json), \
                mock.patch("baselines.preliminary.replay_saved_checkpoints._post_json", side_effect=fake_post_json):
                rc = run(args)

        self.assertEqual(rc, 0)
        self.assertEqual(len(posted), 1)
        self.assertEqual(posted[0][1]["checkpoint_name"], "global_step_20")
        self.assertEqual(posted[0][1]["global_step"], 20)

    def test_discover_eval_checkpoints_includes_fsdp_actor_dir(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            actor = root / "global_step_20" / "actor"
            hf = actor / "huggingface"
            hf.mkdir(parents=True)
            (hf / "config.json").write_text("{}", encoding="utf-8")
            (hf / "tokenizer.json").write_text("{}", encoding="utf-8")
            (hf / "tokenizer_config.json").write_text("{}", encoding="utf-8")
            (hf / "special_tokens_map.json").write_text("{}", encoding="utf-8")
            (actor / "model_world_size_4_rank_0.pt").write_text("", encoding="utf-8")
            self.assertTrue(_is_fsdp_actor_dir(actor))
            items = discover_eval_checkpoints(root)
            self.assertEqual(len(items), 1)
            self.assertEqual(items[0]["checkpoint_name"], "global_step_20")
            self.assertEqual(Path(items[0]["checkpoint_path"]).resolve(), actor.resolve())


if __name__ == "__main__":
    unittest.main()
