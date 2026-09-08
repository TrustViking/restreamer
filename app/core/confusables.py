"""Latin <-> Cyrillic homoglyph normalization.

Used to neutralize visual confusables (e.g. latin 'o' vs cyrillic 'о',
latin 'k' vs cyrillic 'к') when comparing or matching text. Pure utility,
no external dependencies.

The two translation tables below are SYMMETRIC: every entry has both a
lowercase and uppercase pair. The previous incarnation of this code was
missing the lowercase entries for b/h/k/m/t, which broke matches on real
source text containing homoglyphs like 'Жайвороноk' (latin lowercase k).
This file fixes that.
"""
from __future__ import annotations

import re

__all__ = ["normalize_confusables", "detect_target_script"]

_LATIN_TO_CYRILLIC: dict[str, str] = {
    # lowercase
    "a": "а",
    "b": "в",
    "c": "с",
    "e": "е",
    "h": "н",
    "i": "і",
    "k": "к",
    "m": "м",
    "o": "о",
    "p": "р",
    "t": "т",
    "x": "х",
    "y": "у",
    # uppercase
    "A": "А",
    "B": "В",
    "C": "С",
    "E": "Е",
    "H": "Н",
    "I": "І",
    "K": "К",
    "M": "М",
    "O": "О",
    "P": "Р",
    "T": "Т",
    "X": "Х",
    "Y": "У",
}

_CYRILLIC_TO_LATIN: dict[str, str] = {
    # lowercase
    "а": "a",
    "в": "b",
    "с": "c",
    "е": "e",
    "н": "h",
    "і": "i",
    "к": "k",
    "м": "m",
    "о": "o",
    "р": "p",
    "т": "t",
    "х": "x",
    "у": "y",
    # uppercase
    "А": "A",
    "В": "B",
    "С": "C",
    "Е": "E",
    "Н": "H",
    "І": "I",
    "К": "K",
    "М": "M",
    "О": "O",
    "Р": "P",
    "Т": "T",
    "Х": "X",
    "У": "Y",
}

_LATIN_TO_CYRILLIC_TABLE = str.maketrans(_LATIN_TO_CYRILLIC)
_CYRILLIC_TO_LATIN_TABLE = str.maketrans(_CYRILLIC_TO_LATIN)

_CYRILLIC_RE = re.compile(r"[А-Яа-яЁёІіЇїЄєҐґ]", re.UNICODE)
_LATIN_RE = re.compile(r"[A-Za-z]", re.UNICODE)


def detect_target_script(text: str) -> str:
    """Return 'cyrillic' or 'latin' based on the dominant script in text.

    Ties resolve to 'cyrillic' (matching prior behavior).
    """
    sample: str = str(text or "")
    cyrillic_count: int = len(_CYRILLIC_RE.findall(sample))
    latin_count: int = len(_LATIN_RE.findall(sample))
    return "cyrillic" if cyrillic_count >= latin_count else "latin"


def normalize_confusables(text: str, target_script: str = "auto") -> str:
    """Fold latin homoglyphs into cyrillic (or vice versa).

    target_script: 'cyrillic' | 'latin' | 'auto'.
        'auto' (default) detects the dominant script in `text`.
    Only latin/cyrillic homoglyph pairs are touched; other characters pass through.
    """
    sample: str = str(text or "")
    if target_script == "auto":
        target_script = detect_target_script(sample)
    if target_script == "cyrillic":
        return sample.translate(_LATIN_TO_CYRILLIC_TABLE)
    if target_script == "latin":
        return sample.translate(_CYRILLIC_TO_LATIN_TABLE)
    raise ValueError(
        f"normalize_confusables: target_script must be 'cyrillic', 'latin' or 'auto'; got {target_script!r}"
    )
