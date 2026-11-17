"""SignBridge CustomTkinter interface — premium dark desktop UI."""
from __future__ import annotations

import threading
import tkinter as tk
from typing import Callable, Optional

import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image, ImageTk

import config


ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")


class SignBridgeUI(ctk.CTk):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.title(config.APP_TITLE)
        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()
        width = max(980, int(screen_w * 0.68))
        height = max(620, int(screen_h * 0.72))
        width = min(width, screen_w - 40)
        height = min(height, screen_h - 40)
        self.geometry(f"{width}x{height}")
        self.minsize(980, 620)
        self.configure(fg_color=config.BG_DARK)

        self._photo = None
        self._debug_photo = None
        self._flash_until = 0.0

        self._build()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(33, self._tick)

    # ------------------------------------------------------------------ UI
    def _build(self) -> None:
        body = ctk.CTkFrame(self, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=20, pady=8)

        left = ctk.CTkFrame(body, fg_color=config.PANEL_BG, corner_radius=12)
        left.pack(side="left", fill="both", expand=True, padx=(0, 10))
        right = ctk.CTkFrame(body, fg_color="transparent", width=360)
        right.pack(side="right", fill="y", padx=(10, 0))
        right.pack_propagate(False)

        # Camera
        ctk.CTkLabel(
            left, text="Camera", font=ctk.CTkFont(size=14, weight="bold"), text_color=config.TEXT_MUTED
        ).pack(anchor="w", padx=16, pady=(12, 4))
        self.video_label = ctk.CTkLabel(left, text="", fg_color="#0A0E14", corner_radius=8)
        self.video_label.pack(fill="both", expand=True, padx=16, pady=8)

        status_row = ctk.CTkFrame(left, fg_color="transparent")
        status_row.pack(fill="x", padx=16, pady=(0, 8))
        self.hand_status = ctk.CTkLabel(status_row, text="Hand: —", text_color=config.TEXT_MUTED)
        self.hand_status.pack(side="left")
        self.stability_bar = ctk.CTkProgressBar(status_row, width=120, progress_color=config.ACCENT)
        self.stability_bar.set(0)
        self.stability_bar.pack(side="right", padx=8)
        ctk.CTkLabel(status_row, text="Stability", text_color=config.TEXT_MUTED, font=ctk.CTkFont(size=11)).pack(
            side="right"
        )

        letter_row = ctk.CTkFrame(left, fg_color="transparent")
        letter_row.pack(fill="x", padx=16, pady=(0, 16))
        self.letter_label = ctk.CTkLabel(
            letter_row,
            text="—",
            font=ctk.CTkFont(family="Segoe UI", size=56, weight="bold"),
            text_color=config.TEXT_PRIMARY,
            width=80,
        )
        self.letter_label.pack(side="left")
        self.conf_label = ctk.CTkLabel(
            letter_row, text="Confidence: —", font=ctk.CTkFont(size=14), text_color=config.TEXT_MUTED
        )
        self.conf_label.pack(side="left", padx=16)
        self.accept_flash = ctk.CTkLabel(
            letter_row, text="", font=ctk.CTkFont(size=14, weight="bold"), text_color=config.ACCENT
        )
        self.accept_flash.pack(side="left")

        # Debug crop
        debug_frame = ctk.CTkFrame(left, fg_color="transparent")
        debug_frame.pack(fill="x", padx=16, pady=(0, 12))
        ctk.CTkLabel(debug_frame, text="Model input", text_color=config.TEXT_MUTED, font=ctk.CTkFont(size=11)).pack(
            side="left"
        )
        self.debug_label = ctk.CTkLabel(debug_frame, text="", width=112, height=112, fg_color="#0A0E14")
        self.debug_label.pack(side="left", padx=8)

        # Right panels
        self._build_word_panel(right)
        self._build_sentence_panel(right)
        self._build_controls(right)
        self._build_alphabet(right)

    def _panel(self, parent, title: str) -> ctk.CTkFrame:
        f = ctk.CTkFrame(parent, fg_color=config.PANEL_BG, corner_radius=12)
        f.pack(fill="x", pady=(0, 10))
        ctk.CTkLabel(
            f, text=title, font=ctk.CTkFont(size=13, weight="bold"), text_color=config.TEXT_MUTED
        ).pack(anchor="w", padx=14, pady=(10, 4))
        return f

    def _build_word_panel(self, parent) -> None:
        f = self._panel(parent, "Word prediction")
        self.partial_label = ctk.CTkLabel(
            f, text="Partial: ", font=ctk.CTkFont(size=18, weight="bold"), text_color=config.TEXT_PRIMARY
        )
        self.partial_label.pack(anchor="w", padx=14)
        self.best_label = ctk.CTkLabel(
            f, text="Best: —", font=ctk.CTkFont(size=22, weight="bold"), text_color=config.ACCENT
        )
        self.best_label.pack(anchor="w", padx=14, pady=(4, 2))
        self.alt_frame = ctk.CTkFrame(f, fg_color="transparent")
        self.alt_frame.pack(fill="x", padx=10, pady=6)
        self.suggestion_buttons: list[ctk.CTkButton] = []
        ctk.CTkLabel(
            f,
            text="Confirm: thumbs-up  or  click a suggestion",
            font=ctk.CTkFont(size=11),
            text_color=config.TEXT_MUTED,
        ).pack(anchor="w", padx=14, pady=(0, 10))

    def _build_sentence_panel(self, parent) -> None:
        f = self._panel(parent, "Sentence")
        self.words_label = ctk.CTkLabel(
            f, text="Words: —", font=ctk.CTkFont(size=13), text_color=config.TEXT_PRIMARY, wraplength=380, justify="left"
        )
        self.words_label.pack(anchor="w", padx=14, pady=(0, 4))
        self.sentence_entry = ctk.CTkTextbox(f, height=70, font=ctk.CTkFont(size=14))
        self.sentence_entry.pack(fill="x", padx=14, pady=4)
        speak_row = ctk.CTkFrame(f, fg_color="transparent")
        speak_row.pack(fill="x", padx=14, pady=(4, 12))
        ctk.CTkButton(
            speak_row, text="Speak", fg_color=config.ACCENT, text_color="#0A0E14", command=self.controller.speak
        ).pack(side="left")
        ctk.CTkButton(
            speak_row, text="Stop", fg_color="#3A4556", command=self.controller.stop_speech
        ).pack(side="left", padx=8)
        self.speech_status = ctk.CTkLabel(speak_row, text="Audio: idle", text_color=config.TEXT_MUTED)
        self.speech_status.pack(side="left", padx=8)

    def _build_controls(self, parent) -> None:
        f = self._panel(parent, "Controls")
        grid = ctk.CTkFrame(f, fg_color="transparent")
        grid.pack(fill="x", padx=10, pady=(0, 12))
        grid.grid_columnconfigure((0, 1, 2), weight=1)
        buttons = [
            ("Accept Letter", self.controller.accept_pending_letter),
            ("Reject Letter", self.controller.reject_pending_letter),
            ("Undo Letter", self.controller.undo_letter),
            ("Undo Word", self.controller.undo_word),
            ("Clear Word", self.controller.clear_word),
            ("Clear Sentence", self.controller.clear_sentence),
            ("Confirm Best", self.controller.confirm_best),
            ("Diagnostics", self.controller.show_diagnostics),
        ]
        for i, (text, cmd) in enumerate(buttons):
            ctk.CTkButton(grid, text=text, width=105, height=32, fg_color="#243044", command=cmd).grid(
                row=i // 3, column=i % 3, padx=3, pady=3, sticky="ew"
            )

    def _build_alphabet(self, parent) -> None:
        f = self._panel(parent, "Supported letters (no J / Z)")
        letters = " ".join(config.SUPPORTED_LETTERS)
        ctk.CTkLabel(
            f, text=letters, font=ctk.CTkFont(size=12), text_color=config.TEXT_PRIMARY, wraplength=380, justify="left"
        ).pack(anchor="w", padx=14, pady=(0, 12))

    # -------------------------------------------------------------- updates
    def update_video(self, frame_bgr: Optional[np.ndarray]) -> None:
        if frame_bgr is None:
            return
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        max_w, max_h = 720, 480
        scale = min(max_w / w, max_h / h, 1.0)
        if scale < 1.0:
            rgb = cv2.resize(rgb, (int(w * scale), int(h * scale)))
        img = Image.fromarray(rgb)
        self._photo = ctk.CTkImage(light_image=img, dark_image=img, size=(img.width, img.height))
        self.video_label.configure(image=self._photo, text="")

    def update_debug(self, image_u8: Optional[np.ndarray]) -> None:
        if image_u8 is None:
            return
        if image_u8.ndim == 2:
            img = Image.fromarray(image_u8, mode="L").convert("RGB")
        else:
            img = Image.fromarray(image_u8)
        img = img.resize((112, 112))
        self._debug_photo = ctk.CTkImage(light_image=img, dark_image=img, size=(112, 112))
        self.debug_label.configure(image=self._debug_photo, text="")

    def update_recognition(self, letter: str, conf: float, stability: float, accepted: bool) -> None:
        self.letter_label.configure(text=letter if letter else "—")
        self.conf_label.configure(text=f"Confidence: {conf * 100:.0f}%")
        self.stability_bar.set(max(0.0, min(1.0, stability)))
        if accepted:
            self.accept_flash.configure(text=f"Pending {letter}")
            self.after(800, lambda: self.accept_flash.configure(text=""))

    def update_pending_letter(self, letter: str) -> None:
        self.accept_flash.configure(
            text=f"Approve: {letter}" if letter else ""
        )

    def update_hand_status(self, text: str) -> None:
        self.hand_status.configure(text=text)

    def update_word_state(self, partial: str, suggestions: list[str], words: list[str], sentence: str) -> None:
        self.partial_label.configure(text=f"Partial: {partial or '—'}")
        best = suggestions[0] if suggestions else "—"
        self.best_label.configure(text=f"Best: {best}")
        for b in self.suggestion_buttons:
            b.destroy()
        self.suggestion_buttons.clear()
        for s in suggestions[:5]:
            btn = ctk.CTkButton(
                self.alt_frame,
                text=s,
                width=70,
                height=28,
                fg_color="#243044",
                command=lambda word=s: self.controller.confirm_word(word),
            )
            btn.pack(side="left", padx=3, pady=2)
            self.suggestion_buttons.append(btn)
        self.words_label.configure(text="Words: " + (" · ".join(words) if words else "—"))
        # Only refresh sentence box if user isn't mid-edit with different text
        current = self.sentence_entry.get("1.0", "end").strip()
        if sentence and (not current or current == getattr(self, "_last_auto_sentence", "")):
            self.sentence_entry.delete("1.0", "end")
            self.sentence_entry.insert("1.0", sentence)
            self._last_auto_sentence = sentence

    def update_speech_status(self, status: str) -> None:
        self.speech_status.configure(text=f"Audio: {status}")

    def get_sentence_text(self) -> str:
        return self.sentence_entry.get("1.0", "end").strip()

    def show_message(self, title: str, message: str) -> None:
        win = ctk.CTkToplevel(self)
        win.title(title)
        win.geometry("520x360")
        box = ctk.CTkTextbox(win)
        box.pack(fill="both", expand=True, padx=12, pady=12)
        box.insert("1.0", message)
        box.configure(state="disabled")

    def _tick(self) -> None:
        try:
            self.controller.on_frame()
        except Exception as exc:  # noqa: BLE001
            self.hand_status.configure(text=f"Error: {exc}")
        self.after(33, self._tick)

    def _on_close(self) -> None:
        self.controller.shutdown()
        self.destroy()
