"""Lightweight Pygame widget primitives for tabs, buttons, log streaming, and progress bars."""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple
import pygame

# Common UI Theme Palette
COLOR_BG = (28, 30, 34)
COLOR_CARD = (40, 43, 48)
COLOR_CARD_HOVER = (50, 54, 60)
COLOR_CARD_ACTIVE = (60, 65, 74)
COLOR_PRIMARY = (76, 175, 80)        # Green accent
COLOR_PRIMARY_HOVER = (92, 195, 96)
COLOR_PRIMARY_DARK = (46, 125, 50)
COLOR_DANGER = (220, 53, 69)
COLOR_DANGER_HOVER = (235, 75, 90)
COLOR_TEXT = (240, 240, 240)
COLOR_TEXT_MUTED = (160, 165, 175)
COLOR_BORDER = (65, 70, 80)
COLOR_PROGRESS_BG = (30, 32, 36)


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
        self.is_active: bool = False

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
        elif self.is_active:
            bg = COLOR_CARD_ACTIVE
            txt_c = COLOR_PRIMARY
        elif self.is_hovered:
            bg = self.hover_color
            txt_c = self.text_color
        else:
            bg = self.color
            txt_c = self.text_color

        pygame.draw.rect(surface, bg, self.rect, border_radius=self.border_radius)
        pygame.draw.rect(surface, COLOR_BORDER, self.rect, width=1, border_radius=self.border_radius)

        txt_surf = font.render(self.text, True, txt_c)
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
        on_change: Optional[Callable[[str], None]] = None,
    ):
        self.rect = pygame.Rect(rect)
        self.label = label
        self.options = options
        self.selected_index = initial_index % len(options) if options else 0
        self.on_change = on_change
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
                if self.on_change:
                    self.on_change(self.value)
                return True
        return False

    def render(self, surface: pygame.Surface, font: pygame.font.Font) -> None:
        bg = COLOR_CARD_HOVER if self.is_hovered else COLOR_CARD
        pygame.draw.rect(surface, bg, self.rect, border_radius=6)
        pygame.draw.rect(surface, COLOR_BORDER, self.rect, width=1, border_radius=6)

        text = f"{self.label}: {self.value}" if self.label else self.value
        txt_surf = font.render(text, True, COLOR_TEXT)
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

    def set_progress(self, progress: float, label: str = "") -> None:
        self.progress = max(0.0, min(1.0, progress))
        if label:
            self.label = label

    def render(self, surface: pygame.Surface, font: pygame.font.Font) -> None:
        # Background
        pygame.draw.rect(surface, COLOR_PROGRESS_BG, self.rect, border_radius=4)
        pygame.draw.rect(surface, COLOR_BORDER, self.rect, width=1, border_radius=4)

        # Fill
        fill_w = int((self.rect.width - 2) * self.progress)
        if fill_w > 0:
            fill_rect = pygame.Rect(self.rect.x + 1, self.rect.y + 1, fill_w, self.rect.height - 2)
            pygame.draw.rect(surface, self.color, fill_rect, border_radius=3)

        # Overlay text
        text = self.label if self.label else f"{int(self.progress * 100)}%"
        txt_surf = font.render(text, True, COLOR_TEXT)
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

    def clear(self) -> None:
        self.lines.clear()
        self.scroll_offset = 0

    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.MOUSEWHEEL and self.rect.collidepoint(pygame.mouse.get_pos()):
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

            surf_line = font.render(line, True, color)
            surface.blit(surf_line, (clip_rect.x, y))
            y += self.line_height

        surface.set_clip(orig_clip)
