from __future__ import annotations

import unittest
from datetime import datetime

from app.paths.name_builder import NamePathBuilder


class NamePathBuilderTests(unittest.TestCase):
    def test_merge_doc_title_includes_model_names_and_docx_path_keeps_segment(self) -> None:
        builder = NamePathBuilder(
            local_image_dir_template="./image/{language}/{date}",
            local_doc_dir_template="./docs/{date}",
            preview_name_template="{index}_{language}_{title}",
            doc_title_template="{date}_{processing_mode}_Ежедневные стримы - Everyday streams{llm_models_segment}_{creation_stamp}",
            language_codes={"uk": "UA", "en": "EN", "ru": "RU"},
            max_filename_stem=120,
        )
        doc_title = builder.build_doc_title(
            date_key="090326",
            created_at=datetime(2026, 3, 8, 13, 50, 0),
            processing_mode="merge",
            llm_models_segment="_[gpt-5.1,gpt-4o]",
        )
        self.assertEqual(
            "090326_merge_Ежедневные стримы - Everyday streams_[gpt-5.1,gpt-4o]_1350_080326",
            doc_title,
        )
        docx_path = builder.build_docx_path("090326", doc_title)
        self.assertIsNotNone(docx_path)
        self.assertEqual(
            "090326_merge_Ежедневные_стримы_Everyday_streams_[gpt-5.1,gpt-4o]_1350_080326.docx",
            docx_path.name,
        )

    def test_local_docx_slug_normalization_removes_separator_artifacts(self) -> None:
        builder = NamePathBuilder(
            local_image_dir_template="./image/{language}/{date}",
            local_doc_dir_template="./docs/{date}",
            preview_name_template="{index}_{language}_{title}",
            doc_title_template="{date}_{processing_mode}_Ежедневные стримы - Everyday streams{llm_models_segment}_{creation_stamp}",
            language_codes={"uk": "UA", "en": "EN", "ru": "RU"},
            max_filename_stem=120,
        )
        docx_path = builder.build_docx_path(
            "190326",
            "190326_merge_-_Everyday_streams_[gpt-5.1]_2000_100326",
        )
        self.assertIsNotNone(docx_path)
        self.assertEqual(
            "190326_merge_Everyday_streams_[gpt-5.1]_2000_100326.docx",
            docx_path.name,
        )

    def test_local_docx_slug_normalization_keeps_mixed_language_title_readable(self) -> None:
        builder = NamePathBuilder(
            local_image_dir_template="./image/{language}/{date}",
            local_doc_dir_template="./docs/{date}",
            preview_name_template="{index}_{language}_{title}",
            doc_title_template="{date}_{processing_mode}_{title_fragment}{llm_models_segment}_{creation_stamp}",
            language_codes={"uk": "UA", "en": "EN", "ru": "RU"},
            max_filename_stem=120,
        )
        docx_path = builder.build_docx_path(
            "190326",
            "190326_merge_Новини дня - Everyday streams_日本語_[gpt-5.1]_2000_100326",
        )
        self.assertIsNotNone(docx_path)
        self.assertEqual(
            "190326_merge_Новини_дня_Everyday_streams_日本語_[gpt-5.1]_2000_100326.docx",
            docx_path.name,
        )
        self.assertNotIn("merge_-_", docx_path.name)
        self.assertNotIn("__", docx_path.name)

    def test_local_docx_slug_normalization_drops_empty_title_segment_without_garbage(self) -> None:
        builder = NamePathBuilder(
            local_image_dir_template="./image/{language}/{date}",
            local_doc_dir_template="./docs/{date}",
            preview_name_template="{index}_{language}_{title}",
            doc_title_template="{date}_{processing_mode}_{title_fragment}{llm_models_segment}_{creation_stamp}",
            language_codes={"uk": "UA", "en": "EN", "ru": "RU"},
            max_filename_stem=120,
        )
        docx_path = builder.build_docx_path(
            "190326",
            "190326_merge___-___[gpt-5.1]_2000_100326",
        )
        self.assertIsNotNone(docx_path)
        self.assertEqual(
            "190326_merge_[gpt-5.1]_2000_100326.docx",
            docx_path.name,
        )
        self.assertNotIn("-_", docx_path.name)
        self.assertNotIn("__", docx_path.name)

    def test_merge_reject_debug_json_path_uses_doc_artifact_base_dir_and_stable_slug(self) -> None:
        builder = NamePathBuilder(
            local_image_dir_template="./image/{language}/{date}",
            local_doc_dir_template="./docs/{date}",
            preview_name_template="{index}_{language}_{title}",
            doc_title_template="{date}_{processing_mode}_{title_fragment}{llm_models_segment}_{creation_stamp}",
            language_codes={"uk": "UA", "en": "EN", "ru": "RU"},
            max_filename_stem=120,
        )
        json_path = builder.build_merge_reject_debug_json_path(
            date_key="190326",
            slot_key="190326_1800",
            language="en",
            processing_mode="merge",
            source_count=4,
        )
        self.assertIsNotNone(json_path)
        assert json_path is not None
        self.assertEqual("docs", json_path.parts[0])
        self.assertEqual("190326", json_path.parts[1])
        self.assertEqual("merge_reject_debug_json", json_path.parts[2])
        self.assertEqual(
            "190326_merge_190326_1800_en_sources4_merge_rejected.json",
            json_path.name,
        )


if __name__ == "__main__":
    unittest.main()
