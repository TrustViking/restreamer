from __future__ import annotations

from typing import Any, Dict, List, Tuple

from app.bootstrap.logging_config import get_logger
from app.core.models import VideoMetadata


LOGGER = get_logger(__name__)


class DocsRequestBuilder:
    DEFAULT_FONT_FAMILY: str = "Arial"
    DEFAULT_FONT_SIZE_PT: float = 13

    VIDEO_TABLE_TITLE_ROW_INDEX: int = 0
    VIDEO_TABLE_DESCRIPTION_ROW_INDEX: int = 1
    VIDEO_TABLE_URL_ROW_INDEX: int = 2
    VIDEO_TABLE_IMAGE_ROW_INDEX: int = 3

    DEFAULT_VIDEO_IMAGE_HEIGHT_PT: float = 180
    DEFAULT_VIDEO_IMAGE_WIDTH_PT: float = 320

    @staticmethod
    def build_insert_text_request(index: int, text: str) -> Dict[str, Any]:
        return {
            "insertText": {
                "location": {"index": int(index)},
                "text": text,
            }
        }

    @staticmethod
    def build_text_style_request(
        start: int,
        end: int,
        *,
        bold: bool,
        font_family: str = DEFAULT_FONT_FAMILY,
        font_size_pt: float = DEFAULT_FONT_SIZE_PT,
    ) -> Dict[str, Any]:
        return {
            "updateTextStyle": {
                "range": {
                    "startIndex": int(start),
                    "endIndex": int(end),
                },
                "textStyle": {
                    "weightedFontFamily": {"fontFamily": font_family},
                    "fontSize": {"magnitude": font_size_pt, "unit": "PT"},
                    "bold": bold,
                },
                "fields": "weightedFontFamily,fontSize,bold",
            }
        }

    @staticmethod
    def build_paragraph_style_request(
        start: int,
        end: int,
        *,
        alignment: str,
    ) -> Dict[str, Any]:
        return {
            "updateParagraphStyle": {
                "range": {
                    "startIndex": int(start),
                    "endIndex": int(end),
                },
                "paragraphStyle": {"alignment": alignment},
                "fields": "alignment",
            }
        }

    @staticmethod
    def build_cell_background_request(
        table_start_index: int,
        row: int,
        col: int,
        *,
        rgb: Tuple[float, float, float],
    ) -> Dict[str, Any]:
        red, green, blue = rgb
        return {
            "updateTableCellStyle": {
                "tableRange": {
                    "tableCellLocation": {
                        "tableStartLocation": {"index": int(table_start_index)},
                        "rowIndex": int(row),
                        "columnIndex": int(col),
                    },
                    "rowSpan": 1,
                    "columnSpan": 1,
                },
                "tableCellStyle": {
                    "backgroundColor": {
                        "color": {
                            "rgbColor": {
                                "red": red,
                                "green": green,
                                "blue": blue,
                            }
                        }
                    }
                },
                "fields": "backgroundColor",
            }
        }

    @staticmethod
    def build_inline_image_request(
        index: int,
        uri: str,
        *,
        height_pt: float,
        width_pt: float,
    ) -> Dict[str, Any]:
        return {
            "insertInlineImage": {
                "location": {"index": int(index)},
                "uri": uri,
                "objectSize": {
                    "height": {"magnitude": height_pt, "unit": "PT"},
                    "width": {"magnitude": width_pt, "unit": "PT"},
                },
            }
        }

    @staticmethod
    def build_page_break_request() -> Dict[str, Any]:
        return {"insertPageBreak": {"endOfSegmentLocation": {}}}

    @classmethod
    def build_insert_and_style_requests(
        cls,
        index: int,
        text: str,
        *,
        bold: bool,
        font_family: str = DEFAULT_FONT_FAMILY,
        font_size_pt: float = DEFAULT_FONT_SIZE_PT,
    ) -> List[Dict[str, Any]]:
        return [
            cls.build_insert_text_request(index=index, text=text),
            cls.build_text_style_request(
                start=index,
                end=index + len(text),
                bold=bold,
                font_family=font_family,
                font_size_pt=font_size_pt,
            ),
        ]

    @classmethod
    def build_format_request(
        cls,
        start_index: int,
        end_index: int,
        *,
        font_family: str = DEFAULT_FONT_FAMILY,
        font_size_pt: float = DEFAULT_FONT_SIZE_PT,
    ) -> Dict[str, Any]:
        return {
            "updateTextStyle": {
                "range": {
                    "startIndex": start_index,
                    "endIndex": end_index,
                },
                "textStyle": {
                    "weightedFontFamily": {"fontFamily": font_family},
                    "fontSize": {"magnitude": font_size_pt, "unit": "PT"},
                },
                "fields": "weightedFontFamily,fontSize",
            }
        }

    @staticmethod
    def cell_start_index(
        *,
        row_index: int,
        col_index: int,
        columns: int,
        cell_start_indices: List[int],
        use_plus_one: bool,
    ) -> int:
        index: int = row_index * columns + col_index
        if use_plus_one:
            return cell_start_indices[index] + 1
        return cell_start_indices[index]

    @classmethod
    def build_video_text_requests_payload(
        cls,
        *,
        rows: int,
        columns: int,
        cell_start_indices: List[int],
        video: VideoMetadata,
        description_text: str,
        use_plus_one: bool,
        font_family: str = DEFAULT_FONT_FAMILY,
        font_size_pt: float = DEFAULT_FONT_SIZE_PT,
    ) -> List[Dict[str, Any]]:
        _ = rows
        requests_payload: List[Dict[str, Any]] = []

        url_start_index: int = cls.cell_start_index(
            row_index=cls.VIDEO_TABLE_URL_ROW_INDEX,
            col_index=0,
            columns=columns,
            cell_start_indices=cell_start_indices,
            use_plus_one=use_plus_one,
        )
        url_text: str = f"{video.url}\n"
        requests_payload.append(
            cls.build_insert_text_request(index=url_start_index, text=url_text)
        )
        requests_payload.append(
            cls.build_format_request(
                start_index=url_start_index,
                end_index=url_start_index + len(url_text),
                font_family=font_family,
                font_size_pt=font_size_pt,
            )
        )

        description_start_index: int = cls.cell_start_index(
            row_index=cls.VIDEO_TABLE_DESCRIPTION_ROW_INDEX,
            col_index=0,
            columns=columns,
            cell_start_indices=cell_start_indices,
            use_plus_one=use_plus_one,
        )
        description_full_text: str = f"{description_text}\n"
        requests_payload.append(
            cls.build_insert_text_request(
                index=description_start_index,
                text=description_full_text,
            )
        )
        requests_payload.append(
            cls.build_format_request(
                start_index=description_start_index,
                end_index=description_start_index + len(description_full_text),
                font_family=font_family,
                font_size_pt=font_size_pt,
            )
        )

        title_start_index: int = cls.cell_start_index(
            row_index=cls.VIDEO_TABLE_TITLE_ROW_INDEX,
            col_index=0,
            columns=columns,
            cell_start_indices=cell_start_indices,
            use_plus_one=use_plus_one,
        )
        title_text: str = f"{video.title}\n"
        requests_payload.append(
            cls.build_insert_text_request(index=title_start_index, text=title_text)
        )
        requests_payload.append(
            cls.build_format_request(
                start_index=title_start_index,
                end_index=title_start_index + len(title_text),
                font_family=font_family,
                font_size_pt=font_size_pt,
            )
        )
        return requests_payload

    @classmethod
    def build_video_image_request(
        cls,
        *,
        row_index: int,
        col_index: int,
        columns: int,
        cell_start_indices: List[int],
        use_plus_one: bool,
        image_uri: str,
        height_pt: float = DEFAULT_VIDEO_IMAGE_HEIGHT_PT,
        width_pt: float = DEFAULT_VIDEO_IMAGE_WIDTH_PT,
    ) -> Dict[str, Any]:
        image_index: int = cls.cell_start_index(
            row_index=row_index,
            col_index=col_index,
            columns=columns,
            cell_start_indices=cell_start_indices,
            use_plus_one=use_plus_one,
        )
        return cls.build_inline_image_request(
            index=image_index,
            uri=image_uri,
            height_pt=height_pt,
            width_pt=width_pt,
        )

    @classmethod
    def build_video_image_link_text_request(
        cls,
        *,
        row_index: int,
        col_index: int,
        columns: int,
        cell_start_indices: List[int],
        use_plus_one: bool,
        fallback_external_thumbnail_url: str,
    ) -> Dict[str, Any]:
        image_index: int = cls.cell_start_index(
            row_index=row_index,
            col_index=col_index,
            columns=columns,
            cell_start_indices=cell_start_indices,
            use_plus_one=use_plus_one,
        )
        return cls.build_insert_text_request(
            index=image_index,
            text=f"Thumbnail: {fallback_external_thumbnail_url}\n",
        )
