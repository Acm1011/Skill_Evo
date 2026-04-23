"""``stats_rollout_prompt_tokens`` 的单元测试（不依赖大 jsonl / 本机模型）。"""
from __future__ import annotations

import unittest

from skill_src.tools.stats_rollout_prompt_tokens import _extract_content


class TestExtractContent(unittest.TestCase):
    def test_single_user_message(self) -> None:
        row = {"prompt": [{"role": "user", "content": "hello"}]}
        self.assertEqual(_extract_content(row), "hello")

    def test_missing_prompt(self) -> None:
        self.assertIsNone(_extract_content({}))
        self.assertIsNone(_extract_content({"prompt": []}))
        self.assertIsNone(_extract_content({"prompt": "bad"}))

    def test_non_string_content_coerced(self) -> None:
        row = {"prompt": [{"role": "user", "content": 123}]}
        self.assertEqual(_extract_content(row), "123")


if __name__ == "__main__":
    unittest.main()
