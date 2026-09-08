from app.pipeline.batch_runner import (
    _PREPARATION_PROGRESS_STEP,
    _should_emit_preparation_progress,
)


def test_step_constant_is_five() -> None:
    # The prompt fixes the throttle step at 5. If this is intentionally
    # changed in future, update both the constant and this test together.
    assert _PREPARATION_PROGRESS_STEP == 5


def test_emit_every_fifth_and_final_for_20_rows() -> None:
    expected_ticks: list[int] = [5, 10, 15, 20]
    actual_ticks: list[int] = [
        done for done in range(1, 21)
        if _should_emit_preparation_progress(done, 20)
    ]
    assert actual_ticks == expected_ticks


def test_emit_final_when_total_not_multiple_of_step() -> None:
    expected_ticks: list[int] = [5, 7]
    actual_ticks: list[int] = [
        done for done in range(1, 8)
        if _should_emit_preparation_progress(done, 7)
    ]
    assert actual_ticks == expected_ticks


def test_emit_only_final_when_total_below_step() -> None:
    expected_ticks: list[int] = [3]
    actual_ticks: list[int] = [
        done for done in range(1, 4)
        if _should_emit_preparation_progress(done, 3)
    ]
    assert actual_ticks == expected_ticks


def test_emit_nothing_when_total_is_zero() -> None:
    assert _should_emit_preparation_progress(0, 0) is False
    assert _should_emit_preparation_progress(1, 0) is False


def test_emit_nothing_when_done_is_zero_or_out_of_range() -> None:
    assert _should_emit_preparation_progress(0, 20) is False
    assert _should_emit_preparation_progress(21, 20) is False
