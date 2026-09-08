from __future__ import annotations

import unittest

from app.planning.batch_planner import _classify_shared_preparation_error


class ClassifySharedPreparationErrorTests(unittest.TestCase):
    def test_private_video_recognised(self) -> None:
        err = RuntimeError(
            "yt-dlp не смог получить metadata для https://youtu.be/xyz. "
            "stderr: ERROR: [youtube] xyz: Private video. Sign in if you've been granted access to this video"
        )
        self.assertEqual(
            _classify_shared_preparation_error(err),
            "видео приватное / нет доступа через cookies",
        )

    def test_sign_in_required_recognised(self) -> None:
        err = RuntimeError(
            "yt-dlp ... stderr: ERROR: [youtube] abc: Sign in to confirm you're not a bot"
        )
        self.assertEqual(
            _classify_shared_preparation_error(err),
            "видео приватное / нет доступа через cookies",
        )

    def test_sign_in_if_you_recognised(self) -> None:
        err = RuntimeError(
            "stderr: ERROR: [youtube] xyz: Sign in if you've been granted access to this video"
        )
        self.assertEqual(
            _classify_shared_preparation_error(err),
            "видео приватное / нет доступа через cookies",
        )

    def test_video_unavailable_recognised(self) -> None:
        err = RuntimeError(
            "yt-dlp ... stderr: ERROR: [youtube] qwe: Video unavailable. This video is no longer available"
        )
        self.assertEqual(
            _classify_shared_preparation_error(err),
            "видео недоступно (удалено или заблокировано)",
        )

    def test_removed_by_uploader_recognised(self) -> None:
        err = RuntimeError(
            "stderr: ERROR: [youtube] zzz: Video unavailable: This video has been removed by the uploader"
        )
        self.assertEqual(
            _classify_shared_preparation_error(err),
            "видео недоступно (удалено или заблокировано)",
        )

    def test_unknown_error_falls_back(self) -> None:
        err = ValueError("totally unrelated parsing error")
        self.assertEqual(
            _classify_shared_preparation_error(err),
            "не удалось подготовить строку (см. detailed log)",
        )


if __name__ == "__main__":
    unittest.main()
