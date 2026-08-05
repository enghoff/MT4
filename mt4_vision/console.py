"""Bottom-pinned terminal UI for interactive scripts.

One editable prompt line and one status line, both pinned to the bottom of
the terminal, with scrolling transcript history above them. Background
threads call :meth:`BottomUI.set_status` / :meth:`BottomUI.emit` while the
main thread blocks in :meth:`BottomUI.read_line` -- which is the whole point:
a long-running worker (arm motion, a VLM answering) can report progress
without its output racing the line the user is typing on.

Falls back to plain ``input()``/``print()`` when the output is not a TTY, so
the same script stays pipeable.
"""

from __future__ import annotations

import shutil
import sys
import threading
from typing import IO


def enable_vt(stream: IO[str] | None = None) -> bool:
    """Turn on ANSI cursor control on Windows consoles; True if usable."""
    out = stream if stream is not None else sys.stdout
    if not out.isatty():
        return False
    if sys.platform != "win32":
        return True
    import ctypes

    kernel32 = ctypes.windll.kernel32
    handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
    mode = ctypes.c_uint()
    if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
        return False
    return bool(kernel32.SetConsoleMode(handle, mode.value | 0x0004))


def encodable(text: str, stream: IO[str] | None = None) -> str:
    """Drop characters the console cannot encode, rather than raising.

    Text from a language model routinely contains em-dashes, curly quotes and
    the odd emoji; a Windows console on a legacy codepage raises
    UnicodeEncodeError on those, which would kill the thread mid-transcript
    over a punctuation mark.
    """
    out = stream if stream is not None else sys.stdout
    enc = getattr(out, "encoding", None) or "utf-8"
    try:
        text.encode(enc)
    except (UnicodeEncodeError, LookupError):
        return text.encode(enc, errors="replace").decode(enc, errors="replace")
    return text


class BottomUI:
    """Fixed prompt (2nd-to-last line) + status (last line); no scroll chatter.

    Background threads call ``set_status``; the main thread blocks in
    ``read_line`` editing the prompt line in place. ``emit`` appends to the
    transcript above without disturbing either pinned line.

    ``stream`` defaults to ``sys.stdout`` and exists so the escape sequences
    can be captured in tests without patching global state.
    """

    def __init__(self, label: str = "object", stream: IO[str] | None = None) -> None:
        self._label = f"{label}: "
        self._out = stream if stream is not None else sys.stdout
        self._lock = threading.Lock()
        # Reentrant, and separate from _lock: a worker thread's emit() must not
        # interleave its escape sequences with the main thread's prompt redraw,
        # and emit() itself ends in a redraw.
        self._out_lock = threading.RLock()
        self._status = "ready"
        self._buf = ""
        self._live = enable_vt(self._out)
        if self._live:
            # Reserve the bottom two lines once so history stays above, then
            # confine scrolling to the rest -- otherwise every transcript
            # line would push the pinned lines off the bottom.
            self._write("\n\n")
            self._set_scroll_region()
            self._redraw()

    @property
    def live(self) -> bool:
        return self._live

    def _write(self, text: str) -> None:
        with self._out_lock:
            self._out.write(text)
            self._out.flush()

    def _size(self) -> tuple[int, int]:
        sz = shutil.get_terminal_size((80, 24))
        return max(4, sz.lines), max(20, sz.columns)

    def _set_scroll_region(self) -> None:
        rows, _cols = self._size()
        # DECSTBM homes the cursor, so every caller must reposition after.
        self._write(f"\033[1;{rows - 2}r")

    def _redraw(self) -> None:
        rows, cols = self._size()
        prompt_row = rows - 1
        status_row = rows
        with self._lock:
            buf = self._buf
            status = self._status
        # Truncate so we never wrap (wrapping would scroll).
        max_buf = max(0, cols - len(self._label) - 1)
        if len(buf) > max_buf:
            buf = buf[-max_buf:]
        status = status.replace("\n", " ")
        if len(status) > cols - 1:
            status = status[: cols - 1]
        col = len(self._label) + len(buf) + 1
        self._write(
            "\033[?25l"
            f"\033[{prompt_row};1H\033[2K{self._label}{buf}"
            f"\033[{status_row};1H\033[2K{status}"
            f"\033[{prompt_row};{col}H\033[?25h"
        )

    def set_status(self, msg: str) -> None:
        msg = encodable(msg, self._out)
        with self._lock:
            self._status = msg
        if self._live:
            self._redraw()
        else:
            print(msg, file=self._out, flush=True)

    def emit(self, text: str = "") -> None:
        """Append lines to the transcript above the pinned lines.

        A line longer than the terminal width is wrapped across several rows
        -- each its own scroll step -- rather than clipped, so a long reply
        (a JSON array on one line, an unbroken URL) never silently loses its
        tail on screen. This is unlike ``_redraw``'s prompt/status truncation,
        which must stay exactly one row or it would unpin the footer.
        """
        text = encodable(text, self._out)
        if not self._live:
            print(text, file=self._out, flush=True)
            return
        rows, cols = self._size()
        last = rows - 2  # bottom row of the scroll region
        width = cols - 1
        with self._out_lock:
            self._set_scroll_region()
            out = ["\033[?25l"]
            for line in text.split("\n"):
                chunks = [line[i : i + width] for i in range(0, len(line), width)] or [""]
                for chunk in chunks:
                    # At the region's bottom row, \n scrolls the region up by one.
                    out.append(f"\033[{last};1H\n\033[2K{chunk}")
            self._write("".join(out))
            self._redraw()

    def read_line(self) -> str | None:
        """Edit the prompt line until Enter. None on EOF."""
        if not self._live:
            try:
                print(f"\n{self._label}", file=self._out, end="", flush=True)
                raw = input()
            except EOFError:
                print(file=self._out)
                return None
            return raw.strip()

        with self._lock:
            self._buf = ""
        self._redraw()

        if sys.platform == "win32":
            return self._read_win()
        return self._read_posix()

    def _take_line(self) -> str:
        with self._lock:
            line = self._buf.strip()
            self._buf = ""
        self._redraw()
        return line

    def _read_win(self) -> str | None:
        import msvcrt

        while True:
            ch = msvcrt.getwch()
            if ch in ("\r", "\n"):
                return self._take_line()
            if ch == "\x03":
                raise KeyboardInterrupt
            if ch == "\x1a":
                return None
            if ch in ("\x08", "\x7f"):
                with self._lock:
                    self._buf = self._buf[:-1]
                self._redraw()
                continue
            if ch in ("\x00", "\xe0"):
                msvcrt.getwch()
                continue
            if ch.isprintable():
                with self._lock:
                    self._buf += ch
                self._redraw()

    def _read_posix(self) -> str | None:
        import termios
        import tty

        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while True:
                ch = sys.stdin.read(1)
                if ch in ("\r", "\n"):
                    return self._take_line()
                if ch == "\x04":  # Ctrl+D
                    return None
                if ch == "\x03":
                    raise KeyboardInterrupt
                if ch in ("\x7f", "\b"):
                    with self._lock:
                        self._buf = self._buf[:-1]
                    self._redraw()
                    continue
                if ch.isprintable():
                    with self._lock:
                        self._buf += ch
                    self._redraw()
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

    def close(self) -> None:
        if not self._live:
            return
        rows, _cols = self._size()
        self._write(
            "\033[r"  # release the scroll region
            f"\033[{rows - 1};1H\033[2K\033[{rows};1H\033[2K\033[?25h\n"
        )
