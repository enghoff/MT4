"""Unit tests for the bottom-pinned console UI (no terminal required).

The live path writes ANSI escapes to a real terminal, which a test runner does
not have -- so these fake ``isatty`` and capture the byte stream, checking the
sequences that matter: the pinned lines stay addressed absolutely, the
transcript scrolls inside a region that excludes them, and the prompt is
repainted last so the caret ends up where the user is typing.
"""

from __future__ import annotations

import io
import os
import re
from collections.abc import Iterator
from unittest.mock import patch

import pytest

from mt4_vision.console import BottomUI, encodable

ROWS, COLS = 24, 80
# With a 24-row terminal: prompt on 23, status on 24, scroll region rows 1-22.
PROMPT_ROW, STATUS_ROW, REGION_END = 23, 24, 22


class FakeTTY(io.StringIO):
    encoding = "utf-8"

    def isatty(self) -> bool:
        return True


@pytest.fixture
def live() -> Iterator[tuple[BottomUI, FakeTTY]]:
    """A live BottomUI writing into a StringIO, with init output discarded.

    Uses BottomUI's injectable stream rather than patching ``sys.stdout``:
    pytest's capture manager reclaims ``sys.stdout`` when the test body
    starts, so a patch installed during fixture setup would not survive.
    """
    out = FakeTTY()
    with patch(
        "shutil.get_terminal_size", lambda _default=None: os.terminal_size((COLS, ROWS))
    ), patch("mt4_vision.console.enable_vt", return_value=True):
        ui = BottomUI("ask", stream=out)
        out.truncate(0)
        out.seek(0)
        yield ui, out


def rendered(text: str, row: int) -> str:
    """The visible text last painted onto ``row``, escapes stripped."""
    chunks = re.split(rf"\033\[{row};1H", text)
    if len(chunks) < 2:
        return ""
    tail = chunks[-1]
    tail = re.split(r"\033\[\d+;\d+H", tail)[0]
    return re.sub(r"\033\[[0-9;?]*[a-zA-Z]", "", tail)


# -- non-TTY fallback ------------------------------------------------------ #


def test_falls_back_to_plain_print_without_a_tty() -> None:
    out = io.StringIO()  # not a TTY
    ui = BottomUI("ask", stream=out)
    assert not ui.live
    ui.set_status("working")
    ui.emit("a line")
    text = out.getvalue()
    assert "working" in text and "a line" in text
    assert "\033[" not in text  # no escapes when not live


def test_non_tty_read_line_uses_input() -> None:
    ui = BottomUI("ask", stream=io.StringIO())
    with patch("builtins.input", return_value="  a question  "):
        assert ui.read_line() == "a question"
    with patch("builtins.input", side_effect=EOFError):
        assert ui.read_line() is None


# -- live path ------------------------------------------------------------- #


def test_init_reserves_two_lines_and_sets_the_scroll_region() -> None:
    out = FakeTTY()
    with patch(
        "shutil.get_terminal_size", lambda _default=None: os.terminal_size((COLS, ROWS))
    ), patch("mt4_vision.console.enable_vt", return_value=True):
        BottomUI("ask", stream=out)
    text = out.getvalue()
    assert text.startswith("\n\n")                      # bottom two lines reserved
    assert f"\033[1;{REGION_END}r" in text              # region excludes 23 and 24


def test_status_and_prompt_are_addressed_absolutely(
    live: tuple[BottomUI, FakeTTY],
) -> None:
    ui, out = live
    ui.set_status("thinking")
    assert rendered(out.getvalue(), STATUS_ROW).strip() == "thinking"
    assert rendered(out.getvalue(), PROMPT_ROW).startswith("ask: ")


def test_emit_scrolls_inside_the_region_and_repaints_the_prompt(
    live: tuple[BottomUI, FakeTTY],
) -> None:
    ui, out = live
    ui.emit("first\nsecond")
    text = out.getvalue()
    # Region re-asserted every emit, so a window resize cannot strand it.
    assert f"\033[1;{REGION_END}r" in text
    # One scroll per line, driven from the region's bottom row.
    assert text.count(f"\033[{REGION_END};1H\n") == 2
    assert "first" in text and "second" in text
    # Prompt painted after the transcript, so the caret lands on the input line.
    assert text.rindex(f"\033[{PROMPT_ROW};1H") > text.rindex("second")


def test_emit_truncates_to_the_terminal_width(
    live: tuple[BottomUI, FakeTTY],
) -> None:
    ui, out = live
    ui.emit("x" * 200)
    assert "x" * (COLS - 1) in out.getvalue()
    assert "x" * COLS not in out.getvalue()


def test_long_input_never_wraps_the_prompt_line(
    live: tuple[BottomUI, FakeTTY],
) -> None:
    """A wrapped prompt line would scroll the screen and unpin the UI."""
    ui, out = live
    for ch in "y" * 200:
        ui._buf += ch  # noqa: SLF001 -- exercising the redraw path directly
    ui._redraw()  # noqa: SLF001
    assert len(rendered(out.getvalue(), PROMPT_ROW)) < COLS


def test_status_is_flattened_to_one_line(live: tuple[BottomUI, FakeTTY]) -> None:
    ui, out = live
    ui.set_status("two\nlines")
    assert rendered(out.getvalue(), STATUS_ROW).strip() == "two lines"


def test_close_releases_the_scroll_region(live: tuple[BottomUI, FakeTTY]) -> None:
    ui, out = live
    ui.close()
    assert "\033[r" in out.getvalue()


# -- encoding guard -------------------------------------------------------- #


def test_encodable_passes_through_representable_text() -> None:
    assert encodable("plain ascii") == "plain ascii"


def test_encodable_replaces_what_the_console_cannot_encode() -> None:
    class Cp1252(io.StringIO):
        encoding = "cp1252"

    out = encodable("an em—dash and \U0001f600", Cp1252())
    assert "\U0001f600" not in out
    assert "an em" in out and "dash" in out


def test_emit_survives_unencodable_model_output(
    live: tuple[BottomUI, FakeTTY],
) -> None:
    """A punctuation mark must not kill the transcript thread."""
    ui, out = live
    out.encoding = "cp1252"
    ui.emit("nothing moves — all objects \U0001f600 stationary")
    text = out.getvalue()
    assert "nothing moves" in text and "stationary" in text
    assert "\U0001f600" not in text
