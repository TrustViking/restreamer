from __future__ import annotations

import unittest

from app.pipeline.batch_runner import _date_sort_key


class TestDateSortKey(unittest.TestCase):

    def test_mixed_month_boundary(self) -> None:
        keys = ["300426", "010526", "020526", "030526"]
        result = sorted(keys, key=_date_sort_key)
        self.assertEqual(result, ["300426", "010526", "020526", "030526"])

    def test_year_boundary_december_before_january(self) -> None:
        keys = list({"010126", "311225"})
        result = sorted(keys, key=_date_sort_key)
        self.assertEqual(result, ["311225", "010126"])

    def test_invalid_key_goes_to_end(self) -> None:
        keys = ["010526", "abcdef", "020526"]
        result = sorted(keys, key=_date_sort_key)
        self.assertEqual(result, ["010526", "020526", "abcdef"])

    def test_empty_set_produces_empty_list(self) -> None:
        result = sorted(set(), key=_date_sort_key)
        self.assertEqual(result, [])
