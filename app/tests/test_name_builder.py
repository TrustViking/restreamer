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
            doc_title_template="{date}_{processing_mode}_Everyday_streams{llm_models_segment}_{creation_stamp}",
            language_codes_json='{"uk":"UA","en":"EN","ru":"RU"}',
            max_filename_stem=120,
        )
        doc_title = builder.build_doc_title(
            date_key="090326",
            created_at=datetime(2026, 3, 8, 13, 50, 0),
            processing_mode="merge",
            llm_models_segment="_[gpt-5.1,deepseek-chat]",
        )
        self.assertEqual(
            "090326_merge_Everyday_streams_[gpt-5.1,deepseek-chat]_1350_080326",
            doc_title,
        )
        docx_path = builder.build_docx_path("090326", doc_title)
        self.assertIsNotNone(docx_path)
        self.assertEqual(
            "090326_merge_Everyday_streams_[gpt-5.1,deepseek-chat]_1350_080326.docx",
            docx_path.name,
        )


if __name__ == "__main__":
    unittest.main()
