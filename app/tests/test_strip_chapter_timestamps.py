from app.core.env_flags import strip_chapter_timestamps


class TestStripChapterTimestamps:

    def test_cyrillic_heading_with_list(self):
        inp = "Текст до.\n\nТаймкод:\n00:00 Вступ\n01:25 Разбор\n\nТекст після."
        assert strip_chapter_timestamps(inp) == "Текст до.\n\nТекст після."

    def test_cyrillic_plural_heading(self):
        inp = "Intro.\n\nТаймкоды:\n00:00 Part 1\n05:00 Part 2\n\nOutro."
        assert strip_chapter_timestamps(inp) == "Intro.\n\nOutro."

    def test_english_timestamps_heading(self):
        inp = "Description.\n\nTimestamps:\n0:00 Intro\n1:30 Main\n\nFooter."
        assert strip_chapter_timestamps(inp) == "Description.\n\nFooter."

    def test_orphan_heading_no_timecodes(self):
        inp = "Some text.\n\nТаймкод:\n\nMore text."
        assert strip_chapter_timestamps(inp) == "Some text.\n\nMore text."

    def test_no_double_blank_lines_after_removal(self):
        inp = "A.\n\nТаймкод:\n00:00 X\n\n\nB."
        assert strip_chapter_timestamps(inp) == "A.\n\nB."

    # === Реальные кейсы из боевого прогона ===
    def test_ukrainian_zmist_with_blank_line_before_list(self):
        """Реальный украинский кейс: 'Зміст:' + пустая строка + hh:mm:ss-список."""
        inp = (
            "Вступний текст.\n\n"
            "Зміст:\n\n"
            "00:01:33 Павло Бройде\n"
            "00:04:19 Єдина антикультова мережа\n"
            "00:08:09 Нещадний до ворогів рейху\n"
            "01:06:21 Світовий прецедент\n\n"
            "Хвіст тексту."
        )
        assert strip_chapter_timestamps(inp) == "Вступний текст.\n\nХвіст тексту."

    def test_russian_soderzhanie_heading(self):
        inp = "Intro.\n\nСодержание:\n00:00 Начало\n01:30 Основа\n\nEnd."
        assert strip_chapter_timestamps(inp) == "Intro.\n\nEnd."

    def test_ukrainian_rozdily_heading(self):
        inp = "Intro.\n\nРозділи:\n0:00 Початок\n2:30 Основна частина\n\nKінець."
        assert strip_chapter_timestamps(inp) == "Intro.\n\nKінець."

    def test_zmist_heading_with_video_word(self):
        """Заголовок 'Зміст відео:' с дополнительным словом."""
        inp = "Intro.\n\nЗміст відео:\n00:00 Початок\n05:00 Кінець\n\nEnd."
        assert strip_chapter_timestamps(inp) == "Intro.\n\nEnd."

    # === Защита от ложных срабатываний на широких заголовках ===
    def test_broad_heading_without_timestamps_is_preserved(self):
        """Широкий заголовок 'Зміст' без timestamp-block — НЕ удалять."""
        inp = (
            "Про фільм.\n\n"
            "Зміст\n\n"
            "Фільм розповідає про події 1943 року в Україні. "
            "Головний герой — військовий лікар."
        )
        result = strip_chapter_timestamps(inp)
        assert "Зміст" in result
        assert "Фільм розповідає" in result

    def test_broad_contents_heading_as_section_is_preserved(self):
        """'Contents' как обычный заголовок раздела без таймкодов — не удалять."""
        inp = "Overview.\n\nContents\n\nThis book covers three main areas of study."
        result = strip_chapter_timestamps(inp)
        assert "Contents" in result
        assert "This book covers" in result

    def test_broad_timeline_heading_as_section_is_preserved(self):
        """'Timeline' как обычный заголовок раздела без таймкодов — не удалять."""
        inp = "History.\n\nTimeline\n\nThe events unfolded over five years."
        result = strip_chapter_timestamps(inp)
        assert "Timeline" in result
        assert "The events unfolded" in result

    # === Strict-заголовки продолжают работать как раньше (orphan тоже удаляется) ===
    def test_strict_orphan_heading_still_removed(self):
        """Strict heading без таймкодов — удаляется (сохраняем текущее поведение)."""
        inp = "Some text.\n\nТаймкод:\n\nMore text."
        result = strip_chapter_timestamps(inp)
        assert "Таймкод" not in result
        assert "Some text." in result
        assert "More text." in result

    # === Режим B: без заголовка ===
    def test_no_heading_starts_from_zero_is_removed(self):
        """Без заголовка, блок начинается с 00:00 — удалить."""
        inp = "Видео про Python.\n\n00:00 Intro\n01:30 Basics\n05:45 Advanced\n\nПодписывайтесь."
        assert strip_chapter_timestamps(inp) == "Видео про Python.\n\nПодписывайтесь."

    def test_no_heading_not_from_zero_is_kept(self):
        """Без заголовка, не с нуля — НЕ трогать (расписание/повествование)."""
        inp = "Расписание на день.\n\n09:00 Завтрак\n12:00 Обед\n18:00 Ужин\n\nКонец дня."
        result = strip_chapter_timestamps(inp)
        assert "09:00 Завтрак" in result
        assert "12:00 Обед" in result
        assert "18:00 Ужин" in result

    def test_narrative_with_two_time_mentions_is_kept(self):
        """Повествование '11:30 встретил / 12:00 ушёл' — не трогать."""
        inp = "История дня.\n\n11:30 встретил её.\n12:00 ушёл домой.\n\nКонец."
        result = strip_chapter_timestamps(inp)
        assert "11:30 встретил" in result
        assert "12:00 ушёл" in result

    def test_single_timestamp_line_is_kept(self):
        """Одиночная таймкод-строка вне блока — не трогать."""
        inp = "Текст.\n\n00:00 Начало.\n\nПродолжение обычного текста."
        result = strip_chapter_timestamps(inp)
        assert "00:00 Начало." in result

    # === Форматы времени ===
    def test_hh_mm_ss_format_is_recognized(self):
        """Формат часы:минуты:секунды."""
        inp = (
            "Текст.\n\nTimestamps:\n00:00:00 Start\n00:30:15 Middle\n"
            "01:15:45 End\n\nПродолжение."
        )
        assert strip_chapter_timestamps(inp) == "Текст.\n\nПродолжение."

    def test_heading_with_spaces_around_colon_in_time(self):
        """Пробелы вокруг двоеточий в времени."""
        inp = "Text.\n\nChapters:\n00 : 00 Intro\n01 : 30 Part\n\nEnd."
        assert strip_chapter_timestamps(inp) == "Text.\n\nEnd."

    def test_bullet_prefixed_timestamps_with_heading_are_removed(self):
        inp = "Text.\n\nChapters:\n- 00:00 Intro\n- 01:30 Part\n\nEnd."
        result = strip_chapter_timestamps(inp)
        assert result == "Text.\n\nEnd."

    # === Режим C: headless range-блоки ===
    def test_headless_range_block_screenshot_case(self):
        """Реальный кейс со скриншота: 6 range-строк без заголовка, с hh:mm:ss."""
        inp = (
            "Text before.\n\n"
            "00:01:23 – 00:03:46 Introduction\n"
            "00:03:47 – 00:25:02 The Jakub Jahl case\n"
            "25:05 – 33:24 Where do children disappear\n"
            "33:25 – 1:00:14 General characteristics\n"
            "1:00:14 – 1:03:25 Zdeněk Vojtíšek\n"
            "1:03:26 – 1:09:15 The historical roots\n\n"
            "Text after."
        )
        assert strip_chapter_timestamps(inp) == "Text before.\n\nText after."

    def test_headless_range_no_space_after_second_time(self):
        """Range без пробела после второго времени — всё равно должен ловиться."""
        inp = (
            "Text before.\n\n"
            "00:01:23 – 00:03:46Introduction\n"
            "00:03:47 – 00:25:02The Jakub Jahl case\n"
            "25:05 – 33:24Where do children disappear\n\n"
            "Text after."
        )
        assert strip_chapter_timestamps(inp) == "Text before.\n\nText after."

    def test_headless_range_with_em_dash(self):
        """Em-dash (U+2014) как разделитель."""
        inp = (
            "Intro.\n\n"
            "00:00:00 — 00:10:00 Part 1\n"
            "00:10:00 — 00:20:00 Part 2\n"
            "00:20:00 — 00:30:00 Part 3\n\n"
            "End."
        )
        assert strip_chapter_timestamps(inp) == "Intro.\n\nEnd."

    def test_headless_range_with_plain_hyphen(self):
        """Обычный hyphen как разделитель."""
        inp = (
            "Intro.\n\n"
            "00:00:00 - 00:10:00 Part 1\n"
            "00:10:00 - 00:20:00 Part 2\n"
            "00:20:00 - 00:30:00 Part 3\n\n"
            "End."
        )
        assert strip_chapter_timestamps(inp) == "Intro.\n\nEnd."

    # === Защита от false positive: расписания и короткие блоки ===
    def test_daily_schedule_ranges_are_kept(self):
        """Расписание дня в формате диапазона — НЕ удалять (нет hh:mm:ss)."""
        inp = (
            "Schedule.\n\n"
            "09:00 – 10:00 Breakfast\n"
            "10:30 – 11:00 Meeting\n"
            "12:00 – 13:00 Lunch\n\n"
            "End."
        )
        result = strip_chapter_timestamps(inp)
        assert "09:00 – 10:00 Breakfast" in result
        assert "10:30 – 11:00 Meeting" in result
        assert "12:00 – 13:00 Lunch" in result

    def test_two_range_lines_below_threshold_are_kept(self):
        """Всего 2 range-строки — ниже порога 3, не удалять."""
        inp = "Text.\n\n00:01:00 – 00:10:00 Part 1\n00:10:01 – 00:20:00 Part 2\n\nEnd."
        result = strip_chapter_timestamps(inp)
        assert "00:01:00 – 00:10:00 Part 1" in result
        assert "00:10:01 – 00:20:00 Part 2" in result

    def test_single_range_line_is_kept(self):
        """Одиночная range-строка — не трогать."""
        inp = "Text.\n\n09:00 – 10:00 Event\n\nContinue."
        result = strip_chapter_timestamps(inp)
        assert "09:00 – 10:00 Event" in result

    # === Range-блоки с заголовком (должны удаляться, порог 3 не применяется) ===
    def test_range_block_with_chapters_heading(self):
        """Range-блок с Chapters: заголовком — удалить даже если 2 строки."""
        inp = (
            "Intro.\n\n"
            "Chapters:\n"
            "00:00:00 – 00:05:00 Part 1\n"
            "00:05:00 – 00:10:00 Part 2\n\n"
            "End."
        )
        assert strip_chapter_timestamps(inp) == "Intro.\n\nEnd."

    def test_range_block_with_broad_heading_zmist(self):
        """Range-блок с широким заголовком Зміст — удалить."""
        inp = (
            "Intro.\n\n"
            "Зміст:\n"
            "00:00:00 – 00:05:00 Part 1\n"
            "00:05:00 – 00:10:00 Part 2\n"
            "00:10:00 – 00:15:00 Part 3\n\n"
            "End."
        )
        assert strip_chapter_timestamps(inp) == "Intro.\n\nEnd."

    # === Режим D: tail orphan HH:MM:SS cleanup ===
    def test_tail_hh_mm_ss_after_cta_is_removed(self):
        """Реальный кейс со скриншота: HH:MM:SS строка в хвосте после CTA и хэштегов."""
        inp = (
            "📌 This system is not just about the wrongdoing of individuals.\n\n"
            "Join in uncovering the truth—share and spread awareness!\n"
            "#AnticultConspiracy #Truth #OrganizedCrime #Democracy\n"
            "#StopHate\n"
            "Let's not let freedom of thought and security be taken away! Share it, let everyone know!\n\n"
            "2:51:07 Final address: Time to unite and speak the truth loudly!"
        )
        result = strip_chapter_timestamps(inp)
        assert "2:51:07" not in result
        assert "Join in uncovering" in result
        assert "#AnticultConspiracy" in result
        assert "freedom of thought" in result

    def test_tail_multiple_hh_mm_ss_lines_are_removed(self):
        """Несколько HH:MM:SS строк подряд в хвосте — удалить все."""
        inp = (
            "Main text.\n\n"
            "Some conclusion paragraph.\n\n"
            "1:30:05 Part A\n"
            "2:51:07 Part B"
        )
        result = strip_chapter_timestamps(inp)
        assert "1:30:05" not in result
        assert "2:51:07" not in result
        assert "Some conclusion" in result

    def test_tail_hh_mm_ss_range_is_removed(self):
        """Range-строка с HH:MM:SS в хвосте — тоже удалить."""
        inp = (
            "Main text.\n\n"
            "Conclusion paragraph.\n\n"
            "2:51:07 – 3:00:00 Final address"
        )
        result = strip_chapter_timestamps(inp)
        assert "2:51:07" not in result
        assert "3:00:00" not in result
        assert "Conclusion" in result

    # === Защита: HH:MM:SS в середине — не трогать ===
    def test_middle_hh_mm_ss_line_is_kept(self):
        """HH:MM:SS строка в середине текста (после неё идёт обычный текст) — не трогать."""
        inp = (
            "Main text.\n\n"
            "2:51:07 Final address: important quote.\n\n"
            "Explanation after the quote."
        )
        result = strip_chapter_timestamps(inp)
        assert "2:51:07 Final address" in result

    def test_hh_mm_ss_with_text_after_in_same_paragraph_is_kept(self):
        """HH:MM:SS в строке, за которой следует текст в соседнем параграфе — не трогать."""
        inp = (
            "Intro.\n\n"
            "1:00:14 Note about timing.\n"
            "This line is a continuation of the thought."
        )
        result = strip_chapter_timestamps(inp)
        assert "1:00:14 Note about timing" in result
        assert "continuation of the thought" in result

    # === Защита: MM:SS (короткий формат) в хвосте — не трогать ===
    def test_tail_mm_ss_single_line_is_kept(self):
        """Одиночная MM:SS строка в хвосте — не трогать (защита от расписаний)."""
        inp = "Schedule text.\n\n09:00 Breakfast"
        result = strip_chapter_timestamps(inp)
        assert "09:00 Breakfast" in result

    def test_tail_mm_ss_single_with_zero_start_is_kept(self):
        """Одиночная MM:SS с 00:00 в хвосте — не трогать (существующая защита test_single_timestamp_line_is_kept)."""
        inp = "Text.\n\n00:00 Intro"
        result = strip_chapter_timestamps(inp)
        assert "00:00 Intro" in result

    # === Защита: весь текст — только HH:MM:SS строка ===
    def test_text_consisting_only_of_hh_mm_ss_line_is_kept(self):
        """Если в тексте вообще нет содержательного контента кроме HH:MM:SS — не трогать."""
        inp = "1:00:14 Only content"
        result = strip_chapter_timestamps(inp)
        assert "1:00:14 Only content" in result

    def test_text_with_only_hashtags_before_tail_hms_is_stripped(self):
        """Хэштеги — это содержательный контент, поэтому HH:MM:SS после них удаляется."""
        inp = "#tag1 #tag2\n\n2:51:07 Final address"
        result = strip_chapter_timestamps(inp)
        assert "#tag1" in result
        assert "2:51:07" not in result

    # === Защита: не должно влиять на Режимы A/B/C ===
    def test_chapters_block_with_hh_mm_ss_is_still_removed_via_c(self):
        """Range-блок с HH:MM:SS обрабатывается Режимом C, не D."""
        inp = (
            "Text before.\n\n"
            "00:01:23 – 00:03:46 Introduction\n"
            "00:03:47 – 00:25:02 The case\n"
            "25:05 – 33:24 Where\n\n"
            "Text after."
        )
        result = strip_chapter_timestamps(inp)
        assert result == "Text before.\n\nText after."

    def test_bullet_dash_timestamp_block_with_heading_is_removed(self):
        inp = (
            "Some intro text.\n\n"
            "Таймкоды:\n"
            "- 00:00 Intro\n"
            "- 03:21 Body\n"
            "- 09:45 Outro"
        )
        assert strip_chapter_timestamps(inp) == "Some intro text.\n\n"

    def test_bullet_dot_timestamp_block_is_removed(self):
        inp = (
            "• 0:00 Start\n"
            "• 1:23 Middle\n"
            "• 2:45 End"
        )
        assert strip_chapter_timestamps(inp) == ""

    def test_numbered_timestamp_block_is_removed(self):
        inp = (
            "Description.\n\n"
            "1. 00:00 Intro\n"
            "2. 02:15 Demo\n"
            "3. 05:42 Q&A"
        )
        assert strip_chapter_timestamps(inp) == "Description.\n\n"

    def test_em_dash_timestamp_block_with_heading_is_removed(self):
        inp = (
            "Description.\n\n"
            "Chapters:\n"
            "— 00:00 Intro\n"
            "— 03:15 Body\n"
            "— 09:00 Outro"
        )
        assert strip_chapter_timestamps(inp) == "Description.\n\n"

    def test_other_bullet_timestamp_blocks_are_removed(self):
        for marker in ("*", "·", "–"):
            inp = (
                "Description.\n\n"
                f"{marker} 00:00 Intro\n"
                f"{marker} 03:15 Body\n"
                f"{marker} 09:00 Outro"
            )
            assert strip_chapter_timestamps(inp) == "Description.\n\n"

    def test_tail_orphan_with_bullet_is_removed(self):
        inp = (
            "Real content here.\n\n"
            "- 00:01:23"
        )
        assert strip_chapter_timestamps(inp) == "Real content here.\n"

    def test_non_timestamp_bullet_line_is_kept(self):
        inp = "- 12 апреля, выступление"
        assert strip_chapter_timestamps(inp) == inp
