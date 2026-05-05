from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import Callable


SETTINGS_URLS = {
    "screen": "x-apple.systempreferences:com.apple.preference.security?Privacy_ScreenCapture",
    "accessibility": "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility",
    "microphone": "x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone",
}


@dataclass
class PermissionStatus:
    name: str
    granted: bool
    detail: str
    fix_url: str


def _check_screen() -> PermissionStatus:
    try:
        from .capture import capture_main_display

        s = capture_main_display()
        return PermissionStatus("Screen Recording", True, f"capture {s.captured_width}x{s.captured_height}", SETTINGS_URLS["screen"])
    except Exception as e:
        return PermissionStatus("Screen Recording", False, str(e), SETTINGS_URLS["screen"])


def _check_accessibility() -> PermissionStatus:
    try:
        from ApplicationServices import AXIsProcessTrustedWithOptions
        from CoreFoundation import CFDictionaryCreate, kCFAllocatorDefault

        # AXIsProcessTrustedWithOptions(None) returns current state without prompting.
        granted = bool(AXIsProcessTrustedWithOptions(None))
        return PermissionStatus(
            "Accessibility",
            granted,
            "trusted" if granted else "untrusted — required to dispatch clicks/keys",
            SETTINGS_URLS["accessibility"],
        )
    except Exception as e:
        return PermissionStatus("Accessibility", False, f"check failed: {e}", SETTINGS_URLS["accessibility"])


def _check_microphone() -> PermissionStatus:
    try:
        import sounddevice as sd

        devices = sd.query_devices()
        inputs = [d for d in devices if d.get("max_input_channels", 0) > 0]
        if not inputs:
            return PermissionStatus("Microphone", False, "no input devices visible", SETTINGS_URLS["microphone"])
        default = sd.default.device[0] if sd.default.device else None
        return PermissionStatus(
            "Microphone",
            True,
            f"{len(inputs)} input device(s)" + (f", default={default}" if default is not None else ""),
            SETTINGS_URLS["microphone"],
        )
    except Exception as e:
        return PermissionStatus("Microphone", False, str(e), SETTINGS_URLS["microphone"])


CHECKS: list[Callable[[], PermissionStatus]] = [_check_screen, _check_accessibility, _check_microphone]


def check_all() -> list[PermissionStatus]:
    return [c() for c in CHECKS]


def open_settings(url: str) -> None:
    subprocess.run(["open", url], check=False)
