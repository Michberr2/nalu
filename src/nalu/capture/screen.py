from __future__ import annotations

import io
from dataclasses import dataclass

from PIL import Image

from .. import config


@dataclass
class Screenshot:
    image: Image.Image
    display_width: int
    display_height: int
    captured_width: int
    captured_height: int

    @property
    def scale_x(self) -> float:
        return self.display_width / self.captured_width

    @property
    def scale_y(self) -> float:
        return self.display_height / self.captured_height

    def to_jpeg_bytes(self, quality: int = config.CAPTURE_JPEG_QUALITY) -> bytes:
        buf = io.BytesIO()
        self.image.convert("RGB").save(buf, format="JPEG", quality=quality)
        return buf.getvalue()


def capture_main_display(max_width: int = config.CAPTURE_MAX_WIDTH) -> Screenshot:
    """Capture the main display via Quartz. Real pixels, no mocks."""
    import Quartz
    from Quartz.CoreGraphics import CGRectInfinite, CGMainDisplayID
    from Quartz import (
        CGDisplayBounds,
        CGImageGetWidth,
        CGImageGetHeight,
        CGImageGetBytesPerRow,
        CGDataProviderCopyData,
        CGImageGetDataProvider,
    )

    display_id = CGMainDisplayID()
    bounds = CGDisplayBounds(display_id)
    display_w = int(bounds.size.width)
    display_h = int(bounds.size.height)

    img_ref = Quartz.CGWindowListCreateImage(
        CGRectInfinite,
        Quartz.kCGWindowListOptionOnScreenOnly,
        Quartz.kCGNullWindowID,
        Quartz.kCGWindowImageDefault,
    )
    if img_ref is None:
        raise RuntimeError(
            "Screen capture returned no image. Grant Screen Recording permission to your terminal/Python in "
            "System Settings -> Privacy & Security -> Screen Recording, then try again."
        )

    w = CGImageGetWidth(img_ref)
    h = CGImageGetHeight(img_ref)
    bpr = CGImageGetBytesPerRow(img_ref)
    data = CGDataProviderCopyData(CGImageGetDataProvider(img_ref))
    raw = bytes(data)

    img = Image.frombuffer("RGBA", (w, h), raw, "raw", "BGRA", bpr, 1).convert("RGB")
    if w > max_width:
        new_h = int(h * (max_width / w))
        img = img.resize((max_width, new_h), Image.LANCZOS)

    return Screenshot(
        image=img,
        display_width=display_w,
        display_height=display_h,
        captured_width=img.width,
        captured_height=img.height,
    )
