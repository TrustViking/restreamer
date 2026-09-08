from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ScriptMixRepairStats:
    """Diagnostics for a single repair pass over one description."""
    tokens_repaired: int
    repaired_tokens_before: tuple[str, ...]
    repaired_tokens_after: tuple[str, ...]


# Кириллические homoglyph-эквиваленты для латинских букв.
# Только символы, которые в стандартных шрифтах визуально идентичны кириллическим.
# Намеренно НЕ включаем b, d, f, g, j, l, m, n, q, r, s, t, u, v, w, z и
# их прописные аналоги (кроме перечисленных) — у них нет однозначных
# Cyrillic-двойников, и автозамена там опасна.
_SAFE_HOMOGLYPHS_LOWERCASE_BASE: dict[str, str] = {
    "a": "а",
    "c": "с",
    "e": "е",
    "k": "к",
    "o": "о",
    "p": "р",
    "x": "х",
    "y": "у",
}
_SAFE_HOMOGLYPHS_UPPERCASE_BASE: dict[str, str] = {
    "A": "А",
    "B": "В",
    "C": "С",
    "E": "Е",
    "H": "Н",
    "K": "К",
    "M": "М",
    "O": "О",
    "P": "Р",
    "T": "Т",
    "X": "Х",
    "Y": "У",
}
# Латинская "i" в украинском соответствует "і" (U+0456),
# в русском — "и" (U+0438). Маппинг строится per-language.
_I_LOWERCASE_BY_LANG: dict[str, str] = {"uk": "і", "ru": "и"}
_I_UPPERCASE_BY_LANG: dict[str, str] = {"uk": "І", "ru": "И"}

_CYRILLIC_CHAR_RE: re.Pattern[str] = re.compile(r"[А-Яа-яЁёІіЇїЄєҐґ]")
_LATIN_CHAR_RE: re.Pattern[str] = re.compile(r"[A-Za-z]")
# Токен — последовательность букв (кириллица или латиница) с возможным апострофом.
# Цифры, пунктуация, пробелы — границы токена.
_TOKEN_RE: re.Pattern[str] = re.compile(r"[A-Za-zА-Яа-яЁёІіЇїЄєҐґʼ'']+")


def _build_homoglyph_map(language: str) -> dict[str, str]:
    homoglyph_map: dict[str, str] = {}
    homoglyph_map.update(_SAFE_HOMOGLYPHS_LOWERCASE_BASE)
    homoglyph_map.update(_SAFE_HOMOGLYPHS_UPPERCASE_BASE)
    if language in _I_LOWERCASE_BY_LANG:
        homoglyph_map["i"] = _I_LOWERCASE_BY_LANG[language]
        homoglyph_map["I"] = _I_UPPERCASE_BY_LANG[language]
    return homoglyph_map


def _repair_token(token: str, homoglyph_map: dict[str, str]) -> str | None:
    """Try to repair a single token. Return repaired token, or None if no repair needed."""
    cyrillic_chars = _CYRILLIC_CHAR_RE.findall(token)
    latin_chars = _LATIN_CHAR_RE.findall(token)
    # Repair only if token is genuinely script-mixed.
    if not cyrillic_chars or not latin_chars:
        return None
    # Repair only if кириллица в большинстве (защита от обратного направления).
    if len(cyrillic_chars) <= len(latin_chars):
        return None
    # Каждая латинская буква должна иметь homoglyph-эквивалент.
    # Если хотя бы одна буква — реальная латиница без homoglyph (b, d, g, ...) —
    # это настоящий script-mix мусор (типа `twork`), не наш случай. НЕ трогаем.
    for ch in latin_chars:
        if ch not in homoglyph_map:
            return None
    # Всё ок — заменяем.
    repaired_chars: list[str] = []
    for ch in token:
        if ch in homoglyph_map:
            repaired_chars.append(homoglyph_map[ch])
        else:
            repaired_chars.append(ch)
    repaired = "".join(repaired_chars)
    if repaired == token:
        return None
    return repaired


def repair_script_mix_homoglyphs(
    *,
    description: str,
    language: str,
) -> tuple[str, ScriptMixRepairStats]:
    """
    Найти в описании токены, у которых латинский хвост — это homoglyph
    кириллических букв (например, `Жайворонok` → `Жайворонок`), и починить их.

    Применяется только для языков `uk` и `ru`, где основной алфавит —
    кириллица. Для `en` и других языков возвращает текст без изменений.

    Безопасность:
    - токены без кириллицы вообще не трогаются;
    - токены, где латиницы больше или равно кириллице, не трогаются;
    - если в токене есть хотя бы одна латинская буква без однозначного
      кириллического homoglyph-двойника — токен НЕ трогаем.
    Это значит: реальные сломанные слова типа `twork` останутся как есть и
    дальше будут легитимно ловиться `script_mix_contamination`.
    """
    if not description:
        return description, ScriptMixRepairStats(0, (), ())
    if language not in ("uk", "ru"):
        return description, ScriptMixRepairStats(0, (), ())

    homoglyph_map = _build_homoglyph_map(language)

    repaired_before: list[str] = []
    repaired_after: list[str] = []

    def _replace(match: re.Match[str]) -> str:
        token = match.group(0)
        repaired_token = _repair_token(token, homoglyph_map)
        if repaired_token is None:
            return token
        repaired_before.append(token)
        repaired_after.append(repaired_token)
        return repaired_token

    new_description: str = _TOKEN_RE.sub(_replace, description)
    stats = ScriptMixRepairStats(
        tokens_repaired=len(repaired_before),
        repaired_tokens_before=tuple(repaired_before),
        repaired_tokens_after=tuple(repaired_after),
    )
    return new_description, stats
