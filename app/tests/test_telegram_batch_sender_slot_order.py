from __future__ import annotations

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

from app.core.models import NormalizedImage, PlannedVideo, VideoMetadata
from app.pipeline.slot_processing import SlotProcessResult
from app.publish.telegram_batch_sender import TelegramDateBatch, send_telegram_date_batch


class TelegramBatchSenderSlotOrderTests(unittest.TestCase):
    def _video(self, *, language: str, row_number: int, hour: int, minute: int) -> PlannedVideo:
        scheduled_at = datetime(2026, 3, 11, hour, minute, tzinfo=ZoneInfo("Europe/Bucharest"))
        return PlannedVideo(
            row_number=row_number,
            original_link=f"https://youtu.be/{'a' * 11}",
            normalized_link=f"https://youtu.be/{'a' * 11}",
            scheduled_at_kiev=scheduled_at,
            date_key="2026-03-11",
            date_display="2026-03-11",
            language=language,
            metadata=VideoMetadata(
                url=f"https://youtu.be/{'a' * 11}",
                title=f"Title-{language}-{row_number}",
                description="Description",
                thumbnail_url="https://example.com/thumb.jpg",
                youtube_language=language,
            ),
            thumbnail=NormalizedImage(
                bytes_data=b"img",
                extension="jpg",
                mime_type="image/jpeg",
            ),
            local_thumbnail_path=Path(f"{language}-{row_number}.jpg"),
        )

    def _slot(self, *, time_key: str, language: str, videos: list[PlannedVideo]) -> SlotProcessResult:
        return SlotProcessResult(
            slot_key=f"2026-03-11_{time_key}_{language}",
            slot_time_key=time_key,
            header_context={
                "date": "2026-03-11",
                "time_kiev": "18:00",
                "time_cet": "17:00",
                "time_gmt": "15:00",
                "form_url": "https://example.com/form",
                "contacts": "contacts",
            },
            day_videos=videos,
            language_groups={language: videos},
            merged_content_by_language={},
            merge_audit_by_language={},
            real_merge_blocks=0,
            language=language,
        )

    def _config(self) -> SimpleNamespace:
        return SimpleNamespace(
            telegram=SimpleNamespace(
                enabled=True,
                symbol_separator="=",
                separator_repeat_count=1,
                symbol_separator_start=".",
                separator_start_repeat_count=2,
                flag_repeat_count=2,
            ),
            templates=SimpleNamespace(
                telegram_sparkle_separator="SPARKLE",
            ),
        )

    def _send_and_collect_texts(self, slot_results: list[SlotProcessResult]) -> list[str]:
        telegram_client = MagicMock()
        with (
            patch("app.publish.telegram_batch_sender.build_telegram_header_text", return_value="HEADER"),
            patch("app.publish.telegram_batch_sender.build_telegram_key_form_reminder", return_value="KEY"),
            patch("app.publish.telegram_batch_sender.build_telegram_post_header_text", return_value="POST"),
            patch(
                "app.publish.telegram_batch_sender.build_telegram_language_digest_block",
                side_effect=lambda language, videos, context, config: f"DIGEST:{language}:{len(videos)}",
            ),
            patch(
                "app.publish.telegram_batch_sender.build_telegram_language_block",
                side_effect=lambda video, config, templates: (
                    f"BLOCK:{video.language}:{video.scheduled_at_kiev.strftime('%H:%M')}"
                ),
            ),
        ):
            send_telegram_date_batch(
                logger=MagicMock(),
                config=self._config(),
                telegram_client=telegram_client,
                templates=SimpleNamespace(),
                batch=TelegramDateBatch(
                    slot_results=slot_results,
                    header_context={
                        "date": "2026-03-11",
                        "time_kiev": "18:00",
                        "time_cet": "17:00",
                        "time_gmt": "15:00",
                        "form_url": "https://example.com/form",
                        "contacts": "contacts",
                    },
                    doc_url="https://example.com/doc",
                    dry_run=False,
                    date_key="2026-03-11",
                    processing_mode="nomerge",
                ),
            )
        return [str(call.args[0]) for call in telegram_client.send_text.call_args_list]

    def _assert_block_order(self, slot_results: list[SlotProcessResult], expected: list[str]) -> None:
        sent_texts: list[str] = self._send_and_collect_texts(slot_results)
        block_texts: list[str] = [text for text in sent_texts if text.startswith("BLOCK:")]
        self.assertEqual(expected, block_texts)

    def test_order_same_time_uk_then_en(self) -> None:
        slot_results = [
            self._slot(
                time_key="1800",
                language="uk",
                videos=[self._video(language="uk", row_number=1, hour=18, minute=0)],
            ),
            self._slot(
                time_key="1800",
                language="en",
                videos=[self._video(language="en", row_number=2, hour=18, minute=0)],
            ),
        ]
        self._assert_block_order(
            slot_results,
            expected=["BLOCK:uk:18:00", "BLOCK:en:18:00"],
        )

    def test_order_time_primary_then_language_priority(self) -> None:
        slot_results = [
            self._slot(
                time_key="1800",
                language="uk",
                videos=[self._video(language="uk", row_number=1, hour=18, minute=0)],
            ),
            self._slot(
                time_key="1800",
                language="ru",
                videos=[self._video(language="ru", row_number=2, hour=18, minute=0)],
            ),
            self._slot(
                time_key="2000",
                language="en",
                videos=[self._video(language="en", row_number=3, hour=20, minute=0)],
            ),
        ]
        self._assert_block_order(
            slot_results,
            expected=["BLOCK:uk:18:00", "BLOCK:ru:18:00", "BLOCK:en:20:00"],
        )

    def test_sends_language_flag_separator_for_each_slot(self) -> None:
        slot_results = [
            self._slot(
                time_key="1800",
                language="uk",
                videos=[self._video(language="uk", row_number=1, hour=18, minute=0)],
            ),
            self._slot(
                time_key="1900",
                language="en",
                videos=[self._video(language="en", row_number=2, hour=19, minute=0)],
            ),
        ]
        sent_texts: list[str] = self._send_and_collect_texts(slot_results)
        self.assertIn("🇺🇦🇺🇦", sent_texts)
        self.assertIn("🇬🇧🇬🇧", sent_texts)

    def test_sends_key_form_after_each_slot(self) -> None:
        slot_results = [
            self._slot(
                time_key="1800",
                language="uk",
                videos=[self._video(language="uk", row_number=1, hour=18, minute=0)],
            ),
            self._slot(
                time_key="1900",
                language="ru",
                videos=[self._video(language="ru", row_number=2, hour=19, minute=0)],
            ),
            self._slot(
                time_key="2000",
                language="en",
                videos=[self._video(language="en", row_number=3, hour=20, minute=0)],
            ),
        ]
        sent_texts: list[str] = self._send_and_collect_texts(slot_results)
        self.assertEqual(3, sum(1 for text in sent_texts if text == "KEY"))


if __name__ == "__main__":
    unittest.main()
