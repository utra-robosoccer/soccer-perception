"""Getting Qt 6 (PySide6) to start next to OpenCV on Linux. Call `prepare()` before creating
the `QApplication`.

Two things go wrong otherwise:

1. ``import cv2`` (the pip wheel) points ``QT_QPA_PLATFORM_PLUGIN_PATH`` at the Qt 5 plugins
   bundled with OpenCV. Qt 6 then tries to load those and aborts. The variables are removed.
2. From Qt 6.5 on, the X11 ("xcb") platform plugin needs the system library
   ``libxcb-cursor.so.0``, which a default Ubuntu desktop does not install. Without it Qt
   aborts the process. `prepare()` checks first and explains what to install:

       sudo apt install libxcb-cursor0

   Without root, a copy of the library placed in ``<venv>/lib/qt-extra/`` is loaded instead
   (docs/viewer.md gives the two commands).
"""
from __future__ import annotations

import ctypes
import ctypes.util
import os
import sys
from pathlib import Path

LOCAL_LIBRARY_DIR = Path(sys.prefix) / "lib" / "qt-extra"
XCB_CURSOR = "libxcb-cursor.so.0"

INSTALL_HINT = f"""The Qt window cannot start: the system library {XCB_CURSOR} is missing.

Install it (one small package):

    sudo apt install libxcb-cursor0

or, without root, put a copy where the viewer looks for it:

    cd /tmp && apt download libxcb-cursor0 && dpkg -x libxcb-cursor0_*.deb xcb-cursor
    mkdir -p {LOCAL_LIBRARY_DIR}
    cp -a xcb-cursor/usr/lib/x86_64-linux-gnu/libxcb-cursor.so.0* {LOCAL_LIBRARY_DIR}/

The headless mode needs neither:  scripts/visualize_pipeline.py --headless --video run.mp4
"""


class QtUnavailable(RuntimeError):
    """The window cannot be opened here; the message says what to do."""


def _platform() -> str:
    explicit = os.environ.get("QT_QPA_PLATFORM", "").split(":")[0]
    if explicit:
        return explicit
    if sys.platform.startswith("linux"):
        if os.environ.get("XDG_SESSION_TYPE") == "wayland" and os.environ.get("WAYLAND_DISPLAY"):
            return "wayland"
        return "xcb" if os.environ.get("DISPLAY") else "none"
    return "native"


def prepare() -> str:
    """Make the environment safe for PySide6. Returns the platform Qt will use.

    Raises `QtUnavailable` with instructions instead of letting Qt abort the interpreter.
    """
    try:
        import PySide6  # noqa: F401
    except ImportError as error:
        raise QtUnavailable(
            'PySide6 is not installed. Install the viewer extra:  pip install -e ".[viewer]"'
        ) from error
    try:                    # import it now, so that it cannot set the variables later
        import cv2  # noqa: F401
    except ImportError:
        pass
    for name in ("QT_QPA_PLATFORM_PLUGIN_PATH", "QT_QPA_FONTDIR"):
        if "cv2" in os.environ.get(name, ""):
            del os.environ[name]
    platform = _platform()
    if platform == "none":
        raise QtUnavailable("No display found (DISPLAY is not set). Use --headless, or set "
                            "QT_QPA_PLATFORM=offscreen.")
    if platform == "xcb":
        local = LOCAL_LIBRARY_DIR / XCB_CURSOR
        if ctypes.util.find_library("xcb-cursor") is None:
            if not local.exists():
                raise QtUnavailable(INSTALL_HINT)
            # Loading it by path first makes the dynamic linker reuse it when Qt's plugin
            # asks for the same library name.
            ctypes.CDLL(str(local), mode=ctypes.RTLD_GLOBAL)
    return platform
