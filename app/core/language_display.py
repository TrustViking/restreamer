"""Dynamic language display utilities.

Converts ISO 639-1 language codes into flag emojis, short display names,
and full English names without any config dictionaries.
"""
from __future__ import annotations


# Mapping only for languages where ISO 639-1 code != ISO 3166-1 alpha-2 country code.
# For all others (fr, de, ru, pt, it, es, ...) the language code uppercased IS the country code.
_LANG_TO_COUNTRY: dict[str, str] = {
    "uk": "UA",
    "en": "GB",
    "ja": "JP",
    "ko": "KR",
    "zh": "CN",
    "ar": "SA",
    "he": "IL",
    "hi": "IN",
    "cs": "CZ",
    "da": "DK",
    "el": "GR",
    "et": "EE",
    "fa": "IR",
    "sv": "SE",
    "sl": "SI",
    "sq": "AL",
    "sr": "RS",
    "ms": "MY",
    "nb": "NO",
    "nn": "NO",
}


def language_to_flag_emoji(language: str) -> str:
    """Convert ISO 639-1 language code to a flag emoji.

    Uses Regional Indicator Symbol pairs derived from ISO 3166-1 alpha-2 country codes.
    Falls back to 🌐 for unrecognized or malformed codes.
    """
    code: str = str(language or "").strip().lower()
    country: str = _LANG_TO_COUNTRY.get(code, code.upper())
    if len(country) != 2 or not country.isalpha():
        return "\U0001f310"  # 🌐
    a: str = country[0].upper()
    b: str = country[1].upper()
    return chr(0x1F1E6 + ord(a) - ord("A")) + chr(0x1F1E6 + ord(b) - ord("A"))


def language_display_name(language: str) -> str:
    """Short display name for a language: just the ISO code uppercased.

    Examples: 'uk' -> 'UK', 'fr' -> 'FR', 'de' -> 'DE'.
    """
    code: str = str(language or "").strip()
    return code.upper() if code else "UNKNOWN"


def language_full_name(language: str) -> str:
    """Full English name of a language via pycountry.

    Examples: 'uk' -> 'Ukrainian', 'fr' -> 'French'.
    Falls back to language_display_name() if pycountry has no entry.
    """
    import pycountry

    code: str = str(language or "").strip().lower()
    if not code:
        return "Unknown"
    lang = pycountry.languages.get(alpha_2=code)
    if lang is not None:
        return str(lang.name)
    return language_display_name(language)
