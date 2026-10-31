"""Pygame widget primitives and the background-job panel used by the Training/Benchmark tabs."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import pygame

# Common UI Theme Palette
COLOR_BG = (28, 30, 34)
COLOR_CARD = (40, 43, 48)
COLOR_PANEL = (32, 34, 37)
COLOR_PANEL_CARD = (47, 49, 54)
COLOR_CARD_HOVER = (50, 54, 60)
COLOR_PRIMARY = (76, 175, 80)        # Green accent
COLOR_PRIMARY_HOVER = (92, 195, 96)
COLOR_DANGER = (220, 53, 69)
COLOR_DANGER_HOVER = (235, 75, 90)
COLOR_TEXT = (240, 240, 240)
COLOR_TEXT_MUTED = (160, 165, 175)
COLOR_BORDER = (65, 70, 80)
COLOR_PROGRESS_BG = (30, 32, 36)


def _fit_text(
    font: pygame.font.Font,
    text: str,
    max_width: int,
    color: Tuple[int, int, int] = COLOR_TEXT,
) -> pygame.Surface:
    """Render one line without allowing it to spill outside its control."""
    rendered = font.render(text, True, color)
    if rendered.get_width() <= max_width:
        return rendered

    suffix = "..."
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        candidate = font.render(text[:mid] + suffix, True, color)
        if candidate.get_width() <= max_width:
            lo = mid
        else:
            hi = mid - 1
    return font.render(text[:lo] + suffix if lo else suffix, True, color)


class Button:
    def __init__(
        self,
        rect: Tuple[int, int, int, int] | pygame.Rect,
        text: str,
        on_click: Optional[Callable[[], None]] = None,
        color: Tuple[int, int, int] = COLOR_CARD,
        hover_color: Tuple[int, int, int] = COLOR_CARD_HOVER,
        text_color: Tuple[int, int, int] = COLOR_TEXT,
        border_radius: int = 6,
    ):
        self.rect = pygame.Rect(rect)
        self.text = text
        self.on_click = on_click
        self.color = color
        self.hover_color = hover_color
        self.text_color = text_color
        self.border_radius = border_radius
        self.is_hovered: bool = False
        self.is_disabled: bool = False

    def handle_event(self, event: pygame.event.Event) -> bool:
        if self.is_disabled:
            return False
        if event.type == pygame.MOUSEMOTION:
            self.is_hovered = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                if self.on_click:
                    self.on_click()
                return True
        return False

    def render(self, surface: pygame.Surface, font: pygame.font.Font) -> None:
        if self.is_disabled:
            bg = (self.color[0] // 2, self.color[1] // 2, self.color[2] // 2)
            txt_c = COLOR_TEXT_MUTED
        elif self.is_hovered:
            bg = self.hover_color
            txt_c = self.text_color
        else:
            bg = self.color
            txt_c = self.text_color

        pygame.draw.rect(surface, bg, self.rect, border_radius=self.border_radius)
        pygame.draw.rect(surface, COLOR_BORDER, self.rect, width=1, border_radius=self.border_radius)

        txt_surf = _fit_text(font, self.text, max(1, self.rect.width - 16), txt_c)
        txt_rect = txt_surf.get_rect(center=self.rect.center)
        surface.blit(txt_surf, txt_rect)


class TabBar:
    def __init__(
        self,
        rect: Tuple[int, int, int, int] | pygame.Rect,
        tabs: List[Tuple[str, str]],  # [(tab_id, label), ...]
        initial_tab: Optional[str] = None,
        on_tab_change: Optional[Callable[[str], None]] = None,
    ):
        self.rect = pygame.Rect(rect)
        self.tabs = tabs
        self.active_tab = initial_tab or (tabs[0][0] if tabs else "")
        self.on_tab_change = on_tab_change
        self.hovered_tab: Optional[str] = None

    def get_tab_rect(self, index: int) -> pygame.Rect:
        if not self.tabs:
            return pygame.Rect(self.rect.x, self.rect.y, 0, self.rect.height)
        tab_w = self.rect.width // len(self.tabs)
        return pygame.Rect(self.rect.x + index * tab_w, self.rect.y, tab_w, self.rect.height)

    def handle_event(self, event: pygame.event.Event) -> Optional[str]:
        if event.type == pygame.MOUSEMOTION:
            self.hovered_tab = None
            for idx, (t_id, _) in enumerate(self.tabs):
                if self.get_tab_rect(idx).collidepoint(event.pos):
                    self.hovered_tab = t_id
                    break
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            for idx, (t_id, _) in enumerate(self.tabs):
                if self.get_tab_rect(idx).collidepoint(event.pos):
                    if self.active_tab != t_id:
                        self.active_tab = t_id
                        if self.on_tab_change:
                            self.on_tab_change(t_id)
                        return t_id
        return None

    def render(self, surface: pygame.Surface, font: pygame.font.Font) -> None:
        # Background bar
        pygame.draw.rect(surface, (20, 22, 25), self.rect)
        pygame.draw.line(surface, COLOR_BORDER, (self.rect.x, self.rect.bottom - 1), (self.rect.right, self.rect.bottom - 1), 1)

        for idx, (t_id, label) in enumerate(self.tabs):
            tab_rect = self.get_tab_rect(idx)
            is_active = (t_id == self.active_tab)
            is_hover = (t_id == self.hovered_tab)

            if is_active:
                bg = COLOR_CARD
                txt_c = COLOR_PRIMARY
            elif is_hover:
                bg = COLOR_CARD_HOVER
                txt_c = COLOR_TEXT
            else:
                bg = (20, 22, 25)
                txt_c = COLOR_TEXT_MUTED

            pygame.draw.rect(surface, bg, tab_rect)

            if is_active:
                # Active bottom border indicator
                pygame.draw.rect(
                    surface,
                    COLOR_PRIMARY,
                    (tab_rect.x, tab_rect.bottom - 3, tab_rect.width, 3),
                )

            txt_surf = font.render(label, True, txt_c)
            txt_rect = txt_surf.get_rect(center=tab_rect.center)
            surface.blit(txt_surf, txt_rect)


class CycleButton:
    """Button that cycles through a list of choices on click."""
    def __init__(
        self,
        rect: Tuple[int, int, int, int] | pygame.Rect,
        label: str,
        options: List[str],
        initial_index: int = 0,
    ):
        self.rect = pygame.Rect(rect)
        self.label = label
        self.options = options
        self.selected_index = initial_index % len(options) if options else 0
        self.is_hovered: bool = False
        self.is_disabled: bool = False

    @property
    def value(self) -> str:
        return self.options[self.selected_index] if self.options else ""

    def set_options(self, options: List[str], preserve_value: bool = True) -> None:
        cur_val = self.value
        self.options = options
        if preserve_value and cur_val in options:
            self.selected_index = options.index(cur_val)
        else:
            self.selected_index = 0

    def handle_event(self, event: pygame.event.Event) -> bool:
        if self.is_disabled or not self.options:
            return False
        if event.type == pygame.MOUSEMOTION:
            self.is_hovered = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.selected_index = (self.selected_index + 1) % len(self.options)
                return True
        return False

    def render(self, surface: pygame.Surface, font: pygame.font.Font) -> None:
        bg = COLOR_CARD_HOVER if self.is_hovered else COLOR_CARD
        pygame.draw.rect(surface, bg, self.rect, border_radius=6)
        pygame.draw.rect(surface, COLOR_BORDER, self.rect, width=1, border_radius=6)

        text = f"{self.label}: {self.value}" if self.label else self.value
        txt_surf = _fit_text(font, text, max(1, self.rect.width - 16))
        txt_rect = txt_surf.get_rect(center=self.rect.center)
        surface.blit(txt_surf, txt_rect)


class ProgressBar:
    def __init__(
        self,
        rect: Tuple[int, int, int, int] | pygame.Rect,
        progress: float = 0.0,  # 0.0 to 1.0
        label: str = "",
        color: Tuple[int, int, int] = COLOR_PRIMARY,
    ):
        self.rect = pygame.Rect(rect)
        self.progress = max(0.0, min(1.0, progress))
        self.label = label
        self.color = color
        self.indeterminate = False

    def set_progress(self, progress: float, label: str = "", *, indeterminate: bool = False) -> None:
        self.indeterminate = indeterminate
        if not indeterminate:
            self.progress = max(0.0, min(1.0, progress))
        if label:
            self.label = label

    def render(self, surface: pygame.Surface, font: pygame.font.Font) -> None:
        # Background
        pygame.draw.rect(surface, COLOR_PROGRESS_BG, self.rect, border_radius=4)
        pygame.draw.rect(surface, COLOR_BORDER, self.rect, width=1, border_radius=4)

        inner = pygame.Rect(self.rect.x + 1, self.rect.y + 1, self.rect.width - 2, self.rect.height - 2)
        if self.indeterminate:
            span = max(48, inner.width // 5)
            offset = (pygame.time.get_ticks() // 18) % max(1, inner.width + span)
            x = inner.x + offset - span
            fill_rect = pygame.Rect(x, inner.y, span, inner.height)
            fill_rect.clamp_ip(inner)
            if fill_rect.width > 0:
                pygame.draw.rect(surface, self.color, fill_rect, border_radius=3)
        else:
            fill_w = int(inner.width * self.progress)
            if fill_w > 0:
                fill_rect = pygame.Rect(inner.x, inner.y, fill_w, inner.height)
                pygame.draw.rect(surface, self.color, fill_rect, border_radius=3)

        # Overlay text (omit idle "0%" when label is empty)
        if self.label:
            text = self.label
        elif self.progress > 0:
            text = f"{int(self.progress * 100)}%"
        else:
            text = ""
        if text:
            txt_surf = _fit_text(font, text, max(1, self.rect.width - 12))
            txt_rect = txt_surf.get_rect(center=self.rect.center)
            surface.blit(txt_surf, txt_rect)


class LogBox:
    def __init__(
        self,
        rect: Tuple[int, int, int, int] | pygame.Rect,
        max_lines: int = 500,
        line_height: int = 18,
    ):
        self.rect = pygame.Rect(rect)
        self.max_lines = max_lines
        self.line_height = line_height
        self.lines: List[str] = []
        self.scroll_offset = 0  # 0 means auto-scrolled to bottom
        self.auto_scroll = True

    def append(self, text: str) -> None:
        new_lines = text.split("\n")
        for line in new_lines:
            if line.strip() or self.lines:
                self.lines.append(line)
        if len(self.lines) > self.max_lines:
            self.lines = self.lines[-self.max_lines:]
        if self.auto_scroll:
            self.scroll_offset = 0

    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.MOUSEWHEEL and self.rect.collidepoint(getattr(event, "pos", pygame.mouse.get_pos())):
            # event.y > 0 is scroll up, < 0 is scroll down
            self.scroll_offset += event.y * 3
            if self.scroll_offset > 0:
                self.auto_scroll = False
            else:
                self.scroll_offset = 0
                self.auto_scroll = True

    def render(self, surface: pygame.Surface, font: pygame.font.Font) -> None:
        # Box background
        pygame.draw.rect(surface, (18, 20, 22), self.rect, border_radius=6)
        pygame.draw.rect(surface, COLOR_BORDER, self.rect, width=1, border_radius=6)

        clip_rect = pygame.Rect(self.rect.x + 8, self.rect.y + 6, self.rect.width - 16, self.rect.height - 12)
        orig_clip = surface.get_clip()
        surface.set_clip(clip_rect)

        visible_count = clip_rect.height // self.line_height
        total_lines = len(self.lines)

        # Max scroll offset: total_lines - visible_count
        max_offset = max(0, total_lines - visible_count)
        self.scroll_offset = min(max_offset, max(0, self.scroll_offset))

        # Bottom index
        end_idx = total_lines - self.scroll_offset
        start_idx = max(0, end_idx - visible_count)

        y = clip_rect.y
        for idx in range(start_idx, end_idx):
            line = self.lines[idx]
            color = COLOR_TEXT
            if "🎉" in line or "PROMOTED" in line:
                color = COLOR_PRIMARY
            elif "loss:" in line or "Step" in line:
                color = (130, 190, 240)
            elif "rejected" in line or "Error" in line:
                color = (240, 120, 120)
            elif "Game" in line:
                color = COLOR_TEXT_MUTED
            elif "Loading" in line or "Syncing" in line or "Weights" in line:
                color = (200, 180, 90)

            surf_line = font.render(line, True, color)
            surface.blit(surf_line, (clip_rect.x, y))
            y += self.line_height

        surface.set_clip(orig_clip)


class JobPanel:
    """Controls, progress bar, and log for one background job (training or benchmark).

    ``make_job()`` returns ``(start_message, kwargs)`` for ``worker.start``;
    ``describe_progress(data)`` returns ``(fraction, label)``.
    """

    def __init__(
        self,
        worker: Any,
        selectors: List[CycleButton],
        action_rect: Tuple[int, int, int, int],
        progress_rect: Tuple[int, int, int, int],
        log_rect: Tuple[int, int, int, int],
        line_height: int,
        start_text: str,
        stop_text: str,
        idle_label: str,
        make_job: Callable[[], Tuple[str, Dict[str, Any]]],
        describe_progress: Callable[[Dict[str, Any]], Tuple[float, str]],
        on_finished: Optional[Callable[[str], None]] = None,
        on_started: Optional[Callable[[], None]] = None,
        on_result: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        self.worker = worker
        self.selectors = selectors
        self.start_text = start_text
        self.stop_text = stop_text
        self.make_job = make_job
        self.describe_progress = describe_progress
        self.on_finished = on_finished
        self.on_started = on_started
        self.on_result = on_result
        self.status = "idle"

        self.action = Button(action_rect, start_text, on_click=self.toggle, color=COLOR_PRIMARY, hover_color=COLOR_PRIMARY_HOVER)
        self.progress = ProgressBar(progress_rect, label=idle_label)
        self.log = LogBox(log_rect, max_lines=600, line_height=line_height)

    def _style_action(self, running: bool) -> None:
        self.action.text = self.stop_text if running else self.start_text
        self.action.color = COLOR_DANGER if running else COLOR_PRIMARY
        self.action.hover_color = COLOR_DANGER_HOVER if running else COLOR_PRIMARY_HOVER

    def toggle(self) -> None:
        if self.worker.is_running():
            self.worker.request_stop()
            self.action.text = "Stopping..."
            self.action.is_disabled = True
            return
        message, kwargs = self.make_job()
        self.log.append(message)
        if self.worker.start(**kwargs):
            if self.on_started:
                self.on_started()
            self.status = "running"
            job = "Benchmark" if self.worker.kind == "bench" else "Training"
            self.progress.set_progress(0.0, label=f"{job} in progress...")
            self._style_action(running=True)
            for sel in self.selectors:
                sel.is_disabled = True

    def handle_worker_event(self, ev: Dict[str, Any]) -> None:
        type_, data = ev["type"], ev["data"]
        if type_ == "log":
            self.log.append(data)
        elif type_ == "progress":
            self.progress.set_progress(*self.describe_progress(data))
        elif type_ == "result":
            if self.on_result:
                self.on_result(data)
        elif type_ == "status":
            self.status = data
            if data == "running":
                job = "Benchmark" if self.worker.kind == "bench" else "Training"
                self.progress.set_progress(0.0, label=f"{job} in progress...")
            self.action.is_disabled = False
            for sel in self.selectors:
                sel.is_disabled = False
            if data != "running":
                self._style_action(running=False)
                job = "Benchmark" if self.worker.kind == "bench" else "Training"
                if data == "completed":
                    self.progress.set_progress(1.0, label=f"{job} run finished")
                elif data == "stopped":
                    self.progress.set_progress(self.progress.progress, label=f"{job} stopped")
                elif data == "error":
                    self.progress.set_progress(self.progress.progress, label=f"{job} failed")
                if self.on_finished:
                    self.on_finished(data)

    def handle_event(self, event: pygame.event.Event) -> None:
        for sel in self.selectors:
            sel.handle_event(event)
        self.action.handle_event(event)
        self.log.handle_event(event)

    def render(self, surface: pygame.Surface, font_md: pygame.font.Font, font_sm: pygame.font.Font) -> None:
        for sel in self.selectors:
            sel.render(surface, font_md)
        self.action.render(surface, font_md)
        self.progress.render(surface, font_sm)
        self.log.render(surface, font_sm)
