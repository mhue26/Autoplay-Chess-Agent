"""High-DPI (Retina), resizable window: draw to a full-resolution Surface and present it via SDL.

The Surface keeps the design aspect ratio; on resize it is scaled to fit and letterboxed.

pygame 2.6's ``set_mode`` and ``_sdl2.video.Window`` both drop SDL_WINDOW_ALLOW_HIGHDPI on macOS,
so the window is created through SDL directly with ctypes.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
from typing import Optional, Tuple

import pygame

_SDL_WINDOWPOS_CENTERED = 0x2FFF0000
_SDL_WINDOW_SHOWN = 0x4
_SDL_WINDOW_RESIZABLE = 0x20
_SDL_WINDOW_ALLOW_HIGHDPI = 0x2000
_SDL_RENDERER_ACCELERATED = 0x2
_SDL_RENDERER_PRESENTVSYNC = 0x4
_SDL_PIXELFORMAT_XRGB8888 = 0x16161804
_SDL_TEXTUREACCESS_STREAMING = 1
_XRGB_MASKS = (0x00FF0000, 0x0000FF00, 0x000000FF, 0)
_MIN_WINDOW_FRACTION = 0.5


class _Rect(ctypes.Structure):
    _fields_ = [("x", ctypes.c_int), ("y", ctypes.c_int), ("w", ctypes.c_int), ("h", ctypes.c_int)]


def _load_sdl() -> Optional[ctypes.CDLL]:
    bundled = os.path.join(os.path.dirname(pygame.__file__), ".dylibs", "libSDL2-2.0.0.dylib")
    path = bundled if os.path.exists(bundled) else ctypes.util.find_library("SDL2")
    if not path:
        return None
    sdl = ctypes.CDLL(path)
    vp, u32, i = ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int
    for name, res, args in [
        ("SDL_CreateWindow", vp, [ctypes.c_char_p, i, i, i, i, u32]),
        ("SDL_GetWindowSizeInPixels", None, [vp, ctypes.POINTER(i), ctypes.POINTER(i)]),
        ("SDL_GetWindowSize", None, [vp, ctypes.POINTER(i), ctypes.POINTER(i)]),
        ("SDL_SetWindowMinimumSize", None, [vp, i, i]),
        ("SDL_SetHint", ctypes.c_bool, [ctypes.c_char_p, ctypes.c_char_p]),
        ("SDL_GetRendererOutputSize", i, [vp, ctypes.POINTER(i), ctypes.POINTER(i)]),
        ("SDL_SetRenderDrawColor", i, [vp, ctypes.c_uint8, ctypes.c_uint8, ctypes.c_uint8, ctypes.c_uint8]),
        ("SDL_CreateRenderer", vp, [vp, i, u32]),
        ("SDL_CreateTexture", vp, [vp, u32, i, i, i]),
        ("SDL_UpdateTexture", i, [vp, vp, vp, i]),
        ("SDL_RenderClear", i, [vp]),
        ("SDL_RenderCopy", i, [vp, vp, vp, vp]),
        ("SDL_RenderPresent", None, [vp]),
        ("SDL_DestroyTexture", None, [vp]),
        ("SDL_DestroyRenderer", None, [vp]),
        ("SDL_DestroyWindow", None, [vp]),
    ]:
        fn = getattr(sdl, name)
        fn.restype, fn.argtypes = res, args
    return sdl


class HiDPIWindow:
    """Window sized in screen points whose backing Surface is in device pixels (``scale`` × points)."""

    def __init__(self, title: str, size: Tuple[int, int], background: Tuple[int, int, int] = (0, 0, 0)):
        sdl = _load_sdl()
        if sdl is None:
            raise RuntimeError("SDL2 library not found")
        self._sdl = sdl
        self._window = sdl.SDL_CreateWindow(
            title.encode(), _SDL_WINDOWPOS_CENTERED, _SDL_WINDOWPOS_CENTERED, size[0], size[1],
            _SDL_WINDOW_SHOWN | _SDL_WINDOW_ALLOW_HIGHDPI | _SDL_WINDOW_RESIZABLE,
        )
        if not self._window:
            raise RuntimeError("SDL_CreateWindow failed")
        sdl.SDL_SetWindowMinimumSize(
            self._window, int(size[0] * _MIN_WINDOW_FRACTION), int(size[1] * _MIN_WINDOW_FRACTION)
        )
        sdl.SDL_SetHint(b"SDL_RENDER_SCALE_QUALITY", b"linear")

        w, h = ctypes.c_int(), ctypes.c_int()
        sdl.SDL_GetWindowSizeInPixels(self._window, ctypes.byref(w), ctypes.byref(h))
        self.scale = w.value / size[0]
        self._renderer = sdl.SDL_CreateRenderer(self._window, -1, _SDL_RENDERER_ACCELERATED | _SDL_RENDERER_PRESENTVSYNC)
        self._texture = sdl.SDL_CreateTexture(
            self._renderer, _SDL_PIXELFORMAT_XRGB8888, _SDL_TEXTUREACCESS_STREAMING, w.value, h.value
        )
        if not self._renderer or not self._texture:
            self.close()
            raise RuntimeError("SDL renderer setup failed")
        sdl.SDL_SetRenderDrawColor(self._renderer, *background, 255)
        self.surface = pygame.Surface((w.value, h.value), 0, 32, _XRGB_MASKS)

    def _size(self, getter) -> Tuple[int, int]:
        w, h = ctypes.c_int(), ctypes.c_int()
        getter(ctypes.byref(w), ctypes.byref(h))
        return w.value, h.value

    def _fit(self) -> _Rect:
        """Largest aspect-preserving rect for the Surface inside the drawable, centred (output pixels)."""
        out_w, out_h = self._size(lambda w, h: self._sdl.SDL_GetRendererOutputSize(self._renderer, w, h))
        sw, sh = self.surface.get_size()
        k = min(out_w / sw, out_h / sh)
        w, h = int(sw * k), int(sh * k)
        return _Rect((out_w - w) // 2, (out_h - h) // 2, w, h)

    def _mapping(self) -> Tuple[float, _Rect, float]:
        """(output pixels per window point, Surface destination rect, Surface pixels per output pixel)."""
        win_w, _ = self._size(lambda w, h: self._sdl.SDL_GetWindowSize(self._window, w, h))
        out_w, _ = self._size(lambda w, h: self._sdl.SDL_GetRendererOutputSize(self._renderer, w, h))
        dst = self._fit()
        return out_w / max(1, win_w), dst, self.surface.get_width() / max(1, dst.w)

    def surface_per_point(self) -> float:
        px, _, k = self._mapping()
        return px * k

    def to_surface(self, pos: Tuple[int, int]) -> Tuple[int, int]:
        """Map a window-point position (mouse events) to Surface pixel coordinates."""
        px, dst, k = self._mapping()
        return int((pos[0] * px - dst.x) * k), int((pos[1] * px - dst.y) * k)

    def present(self) -> None:
        sdl = self._sdl
        sdl.SDL_UpdateTexture(self._texture, None, self.surface._pixels_address, self.surface.get_pitch())
        sdl.SDL_RenderClear(self._renderer)
        dst = self._fit()
        sdl.SDL_RenderCopy(self._renderer, self._texture, None, ctypes.byref(dst))
        sdl.SDL_RenderPresent(self._renderer)

    def close(self) -> None:
        sdl = self._sdl
        if getattr(self, "_texture", None):
            sdl.SDL_DestroyTexture(self._texture)
        if getattr(self, "_renderer", None):
            sdl.SDL_DestroyRenderer(self._renderer)
        if self._window:
            sdl.SDL_DestroyWindow(self._window)
        self._texture = self._renderer = self._window = None
