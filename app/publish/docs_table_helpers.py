from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from app.bootstrap.logging_config import get_logger


LOGGER = get_logger(__name__)


@dataclass(frozen=True)
class TableSnapshot:
    table_number: int
    table_start_index: int
    cell_start_indices: List[int]


def _cell_index(row: int, col: int, columns: int) -> int:
    return row * columns + col


def _is_google_docs_bounds_error(error: Exception) -> bool:
    return "inside the bounds of an existing paragraph" in str(error).lower()


def _find_last_table_cell_paragraph_start_indices(
    doc: Dict[str, Any], rows: int, columns: int
) -> List[int]:
    """
    Возвращает startIndex абзаца для каждой ячейки (по строкам) последней таблицы.
    Предполагается, что в каждой ячейке есть хотя бы один абзац.
    """
    body: Dict[str, Any] = doc.get("body", {})
    content: List[Dict[str, Any]] = body.get("content", [])
    tables: List[Dict[str, Any]] = [
        item["table"] for item in content if "table" in item
    ]
    if not tables:
        raise RuntimeError("Не нашел таблицу в документе после insertTable.")
    table: Dict[str, Any] = tables[-1]

    table_rows: List[Dict[str, Any]] = table.get("tableRows", [])
    if len(table_rows) < rows:
        raise RuntimeError("Таблица имеет меньше строк, чем ожидается.")
    if any(len(r.get("tableCells", [])) < columns for r in table_rows[:rows]):
        raise RuntimeError("Таблица имеет меньше колонок, чем ожидается.")

    result: List[int] = []
    for row_index in range(rows):
        cells: List[Dict[str, Any]] = table_rows[row_index]["tableCells"]
        for col_index in range(columns):
            cell_content: List[Dict[str, Any]] = cells[col_index].get("content", [])
            paragraph: Optional[Dict[str, Any]] = None
            for element in cell_content:
                if "paragraph" in element:
                    paragraph = element["paragraph"]
                    break
            if paragraph is None:
                raise RuntimeError(
                    "Не нашел paragraph в ячейке таблицы (ожидался всегда)."
                )
            paragraph_container: Optional[Dict[str, Any]] = None
            for element in cell_content:
                if "paragraph" in element:
                    paragraph_container = element
                    break
            if paragraph_container is None or "startIndex" not in paragraph_container:
                raise RuntimeError(
                    "Не удалось определить startIndex paragraph контейнера в ячейке."
                )
            start_index: int = int(paragraph_container["startIndex"])
            result.append(start_index)

    if len(result) != rows * columns:
        raise RuntimeError("Внутренняя ошибка: неверное количество индексов ячеек.")
    return result


def _find_last_table_start_index(doc: Dict[str, Any]) -> int:
    body: Dict[str, Any] = doc.get("body", {})
    content: List[Dict[str, Any]] = body.get("content", [])
    table_items: List[Dict[str, Any]] = [item for item in content if "table" in item]
    if not table_items:
        raise RuntimeError("Не найдена таблица в документе.")
    start_index_raw: Optional[Any] = table_items[-1].get("startIndex")
    if start_index_raw is None:
        raise RuntimeError("Не найден startIndex последней таблицы.")
    return int(start_index_raw)


def _count_tables(doc: Dict[str, Any]) -> int:
    body: Dict[str, Any] = doc.get("body", {})
    content: List[Dict[str, Any]] = body.get("content", [])
    return sum(1 for item in content if "table" in item)


def _build_last_table_snapshot(
    doc: Dict[str, Any], rows: int, columns: int
) -> TableSnapshot:
    return TableSnapshot(
        table_number=_count_tables(doc),
        table_start_index=_find_last_table_start_index(doc=doc),
        cell_start_indices=_find_last_table_cell_paragraph_start_indices(
            doc=doc,
            rows=rows,
            columns=columns,
        ),
    )
