#!/usr/bin/env python3
"""
Pixel Shuffler — desktop application.

Run from your IDE or terminal:
    python main.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QAction, QFont, QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QSlider,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from pixel_shuffler import PixelShufflerTrainer, TrainingConfig
from pixel_shuffler.morph import build_blend_morph, build_deformation_morph, build_snapshot_morph


def numpy_to_qpixmap(arr: np.ndarray, max_side: int = 512) -> QPixmap:
    h, w, ch = arr.shape
    if ch == 3:
        fmt = QImage.Format.Format_RGB888
    else:
        fmt = QImage.Format.Format_RGBA8888
    qimg = QImage(arr.tobytes(), w, h, ch * w, fmt).copy()
    pix = QPixmap.fromImage(qimg)
    if max(h, w) > max_side:
        pix = pix.scaled(max_side, max_side, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
    return pix


class TrainWorker(QThread):
    progress = pyqtSignal(int, int, dict, object)
    finished_ok = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(
        self,
        content_path: str,
        style_path: str,
        config: TrainingConfig,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.content_path = content_path
        self.style_path = style_path
        self.config = config
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:
        try:
            trainer = PixelShufflerTrainer(self.config)

            def on_progress(step: int, total: int, losses: dict, preview_tensor) -> None:
                self.progress.emit(step, total, losses, preview_tensor)

            result = trainer.train(
                self.content_path,
                self.style_path,
                on_progress=on_progress,
                should_cancel=lambda: self._cancel,
            )
            self.finished_ok.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))


class ImageSlot(QFrame):
    """Clickable preview for content or style image."""

    clicked = pyqtSignal()

    def __init__(self, title: str, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("imageSlot")
        self.setMinimumSize(220, 220)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        layout = QVBoxLayout(self)
        self.title = QLabel(title)
        self.title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title.setObjectName("slotTitle")
        self.preview = QLabel("Click to choose image")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setObjectName("slotPreview")
        self.preview.setMinimumHeight(180)
        self.preview.setWordWrap(True)
        layout.addWidget(self.title)
        layout.addWidget(self.preview)
        self._path: str | None = None

    def mousePressEvent(self, event) -> None:
        self.clicked.emit()
        super().mousePressEvent(event)

    def set_image(self, path: str) -> None:
        self._path = path
        from PIL import Image

        img = Image.open(path).convert("RGB")
        img.thumbnail((240, 240))
        arr = np.array(img)
        self.preview.setPixmap(numpy_to_qpixmap(arr, max_side=240))
        self.preview.setText("")

    @property
    def path(self) -> str | None:
        return self._path


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Pixel Shuffler")
        self.resize(1180, 760)
        self._worker: TrainWorker | None = None
        self._morph_frames: list[np.ndarray] = []
        self._morph_index = 0
        self._result = None

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._advance_morph)

        self._build_ui()
        self._apply_stylesheet()

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # --- Left: inputs ---
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.addWidget(QLabel("Inputs"))
        self.content_slot = ImageSlot("Content (structure)")
        self.style_slot = ImageSlot("Style (appearance)")
        self.content_slot.clicked.connect(lambda: self._pick_image(self.content_slot))
        self.style_slot.clicked.connect(lambda: self._pick_image(self.style_slot))
        left_layout.addWidget(self.content_slot)
        left_layout.addWidget(self.style_slot)
        left_layout.addStretch()
        splitter.addWidget(left)

        # --- Center: preview & morph ---
        center = QWidget()
        center_layout = QVBoxLayout(center)
        center_layout.addWidget(QLabel("Preview"))
        self.main_preview = QLabel("Run optimization to see results")
        self.main_preview.setObjectName("mainPreview")
        self.main_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.main_preview.setMinimumSize(420, 420)
        center_layout.addWidget(self.main_preview, stretch=1)

        morph_row = QHBoxLayout()
        self.play_btn = QPushButton("Play morph")
        self.play_btn.clicked.connect(self._toggle_play)
        self.morph_slider = QSlider(Qt.Orientation.Horizontal)
        self.morph_slider.setMinimum(0)
        self.morph_slider.setMaximum(0)
        self.morph_slider.valueChanged.connect(self._show_frame)
        morph_row.addWidget(self.play_btn)
        morph_row.addWidget(self.morph_slider, stretch=1)
        center_layout.addLayout(morph_row)

        self.morph_type = QComboBox()
        self.morph_type.addItems(
            ["Deformation (pixel shuffle)", "Optimization timeline", "Content blend"]
        )
        self.morph_type.currentIndexChanged.connect(lambda _: self._rebuild_morph())
        center_layout.addWidget(self.morph_type)

        export_row = QHBoxLayout()
        self.export_gif_btn = QPushButton("Export GIF")
        self.export_gif_btn.clicked.connect(self._export_gif)
        self.export_png_btn = QPushButton("Save final PNG")
        self.export_png_btn.clicked.connect(self._save_final)
        export_row.addWidget(self.export_gif_btn)
        export_row.addWidget(self.export_png_btn)
        center_layout.addLayout(export_row)
        splitter.addWidget(center)

        # --- Right: controls ---
        right = QWidget()
        right.setMaximumWidth(320)
        right_layout = QVBoxLayout(right)

        preset_box = QGroupBox("Quality preset")
        preset_layout = QVBoxLayout(preset_box)
        self.preset = QComboBox()
        self.preset.addItems(["Fast preview", "Balanced", "High quality"])
        self.preset.currentIndexChanged.connect(self._apply_preset)
        preset_layout.addWidget(self.preset)
        right_layout.addWidget(preset_box)

        params_box = QGroupBox("Optimization")
        params = QGridLayout(params_box)
        self.iterations = QSpinBox()
        self.iterations.setRange(50, 5000)
        self.iterations.setValue(800)
        self.image_size = QSpinBox()
        self.image_size.setRange(128, 512)
        self.image_size.setSingleStep(64)
        self.image_size.setValue(256)
        self.lr = QDoubleSpinBox()
        self.lr.setDecimals(4)
        self.lr.setRange(1e-4, 1e-1)
        self.lr.setValue(0.003)
        self.content_w = QDoubleSpinBox()
        self.content_w.setRange(0, 100)
        self.content_w.setValue(15.0)
        self.style_w = QDoubleSpinBox()
        self.style_w.setRange(0, 5000)
        self.style_w.setValue(800.0)
        self.tv_w = QDoubleSpinBox()
        self.tv_w.setRange(0, 100)
        self.tv_w.setValue(5.0)
        self.morph_frames_spin = QSpinBox()
        self.morph_frames_spin.setRange(10, 240)
        self.morph_frames_spin.setValue(60)

        labels = [
            ("Iterations", self.iterations),
            ("Image size", self.image_size),
            ("Learning rate", self.lr),
            ("Content weight", self.content_w),
            ("Style weight", self.style_w),
            ("TV weight", self.tv_w),
            ("Morph frames", self.morph_frames_spin),
        ]
        for row, (label, widget) in enumerate(labels):
            params.addWidget(QLabel(label), row, 0)
            params.addWidget(widget, row, 1)
        right_layout.addWidget(params_box)

        self.run_btn = QPushButton("Run Pixel Shuffler")
        self.run_btn.setObjectName("primaryButton")
        self.run_btn.clicked.connect(self._start_training)
        right_layout.addWidget(self.run_btn)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._cancel_training)
        right_layout.addWidget(self.cancel_btn)

        self.progress = QProgressBar()
        right_layout.addWidget(self.progress)

        self.loss_label = QLabel("Loss: —")
        self.loss_label.setWordWrap(True)
        right_layout.addWidget(self.loss_label)
        right_layout.addStretch()
        splitter.addWidget(right)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        splitter.setStretchFactor(2, 1)
        root.addWidget(splitter)

        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("Ready — load content and style images, then run.")

        open_action = QAction("Open project folder", self)
        open_action.triggered.connect(lambda: self._open_folder(Path("output")))
        self.menuBar().addAction(open_action)

    def _apply_stylesheet(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget { background: #1a1b26; color: #c0caf5; font-size: 13px; }
            QLabel#slotTitle, QGroupBox { font-weight: 600; color: #7aa2f7; }
            QFrame#imageSlot {
                background: #24283b; border: 2px dashed #414868; border-radius: 12px;
            }
            QFrame#imageSlot:hover { border-color: #7aa2f7; }
            QLabel#mainPreview, QLabel#slotPreview {
                background: #16161e; border-radius: 12px; padding: 8px;
            }
            QPushButton {
                background: #414868; color: #c0caf5; border: none;
                padding: 10px 16px; border-radius: 8px;
            }
            QPushButton:hover { background: #565f89; }
            QPushButton#primaryButton {
                background: #7aa2f7; color: #1a1b26; font-weight: 700;
            }
            QPushButton#primaryButton:hover { background: #89b4fa; }
            QProgressBar {
                border: 1px solid #414868; border-radius: 6px; text-align: center;
                background: #16161e;
            }
            QProgressBar::chunk { background: #9ece6a; border-radius: 5px; }
            QSpinBox, QDoubleSpinBox, QComboBox {
                background: #24283b; border: 1px solid #414868;
                padding: 4px; border-radius: 6px;
            }
            QSlider::groove:horizontal { height: 6px; background: #414868; border-radius: 3px; }
            QSlider::handle:horizontal {
                width: 14px; margin: -5px 0; background: #7aa2f7; border-radius: 7px;
            }
            QSplitter::handle { background: #414868; width: 2px; }
            """
        )
        title_font = QFont()
        title_font.setPointSize(11)
        title_font.setBold(True)
        self.content_slot.title.setFont(title_font)
        self.style_slot.title.setFont(title_font)

    def _apply_preset(self, index: int) -> None:
        presets = [
            (400, 192, 0.003, 15.0, 800.0, 5.0),
            (800, 256, 0.003, 15.0, 800.0, 5.0),
            (1500, 256, 0.002, 15.0, 800.0, 8.0),
        ]
        it, size, lr, cw, sw, tv = presets[index]
        self.iterations.setValue(it)
        self.image_size.setValue(size)
        self.lr.setValue(lr)
        self.content_w.setValue(cw)
        self.style_w.setValue(sw)
        self.tv_w.setValue(tv)

    def _config(self) -> TrainingConfig:
        return TrainingConfig(
            iterations=self.iterations.value(),
            learning_rate=self.lr.value(),
            content_weight=self.content_w.value(),
            style_weight=self.style_w.value(),
            tv_weight=self.tv_w.value(),
            image_size=self.image_size.value(),
        )

    def _pick_image(self, slot: ImageSlot) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            f"Select {slot.title.text()}",
            str(Path.home()),
            "Images (*.png *.jpg *.jpeg *.webp *.bmp)",
        )
        if path:
            slot.set_image(path)

    def _start_training(self) -> None:
        if not self.content_slot.path or not self.style_slot.path:
            QMessageBox.warning(self, "Missing images", "Please select both content and style images.")
            return

        self._timer.stop()
        self.play_btn.setText("Play morph")
        self.run_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.progress.setValue(0)
        self.statusBar().showMessage("Training…")

        self._worker = TrainWorker(
            self.content_slot.path,
            self.style_slot.path,
            self._config(),
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.finished_ok.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _cancel_training(self) -> None:
        if self._worker:
            self._worker.cancel()

    def _on_progress(self, step: int, total: int, losses: dict, preview_tensor) -> None:
        from pixel_shuffler.io_utils import tensor_to_numpy_rgb

        self.progress.setMaximum(total)
        self.progress.setValue(step)
        self.loss_label.setText(
            f"Step {step}/{total}\n"
            f"content={losses['content']:.3f}  style={losses['style']:.3f}  "
            f"tv={losses['tv']:.3f}  total={losses['total']:.1f}"
        )
        arr = tensor_to_numpy_rgb(preview_tensor)
        self.main_preview.setPixmap(numpy_to_qpixmap(arr, max_side=480))

    def _on_finished(self, result) -> None:
        self._result = result
        self.run_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.statusBar().showMessage("Done — play the morph or export a GIF.")
        self._rebuild_morph()
        QMessageBox.information(
            self,
            "Complete",
            f"Optimization finished.\nFinal image saved to:\n{Path('output/final_stylized.png').resolve()}",
        )

    def _on_failed(self, message: str) -> None:
        self.run_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        QMessageBox.critical(self, "Error", message)

    def _rebuild_morph(self) -> None:
        if not self._result:
            return
        n = self.morph_frames_spin.value()
        mode = self.morph_type.currentIndex()
        if mode == 0:
            self._morph_frames = build_deformation_morph(
                self._result.style, self._result.deformation_field, num_frames=n
            )
        elif mode == 1:
            self._morph_frames = build_snapshot_morph(
                self._result.snapshots, self._result.style
            )
        else:
            self._morph_frames = build_blend_morph(
                self._result.content, self._result.final_image, num_frames=n
            )
        self.morph_slider.setMaximum(max(len(self._morph_frames) - 1, 0))
        self._morph_index = 0
        self.morph_slider.setValue(0)
        self._show_frame(0)

    def _show_frame(self, index: int) -> None:
        if not self._morph_frames:
            return
        index = max(0, min(index, len(self._morph_frames) - 1))
        self._morph_index = index
        self.main_preview.setPixmap(numpy_to_qpixmap(self._morph_frames[index], max_side=480))

    def _toggle_play(self) -> None:
        if not self._morph_frames:
            return
        if self._timer.isActive():
            self._timer.stop()
            self.play_btn.setText("Play morph")
        else:
            self._timer.start(50)
            self.play_btn.setText("Pause")

    def _advance_morph(self) -> None:
        if not self._morph_frames:
            return
        next_i = (self._morph_index + 1) % len(self._morph_frames)
        self.morph_slider.blockSignals(True)
        self.morph_slider.setValue(next_i)
        self.morph_slider.blockSignals(False)
        self._show_frame(next_i)

    def _export_gif(self) -> None:
        if not self._morph_frames:
            QMessageBox.information(self, "No frames", "Run optimization first to build a morph.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save GIF", "pixel_shuffler_morph.gif", "GIF (*.gif)"
        )
        if not path:
            return
        try:
            import imageio.v3 as iio

            iio.imwrite(path, self._morph_frames, duration=0.06, loop=0)
            self.statusBar().showMessage(f"Saved GIF: {path}")
        except Exception as exc:
            QMessageBox.critical(self, "Export failed", str(exc))

    def _save_final(self) -> None:
        if not self._result:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save PNG", "result.png", "PNG (*.png)")
        if path:
            from pixel_shuffler.io_utils import save_tensor_image

            save_tensor_image(self._result.final_image, path)
            self.statusBar().showMessage(f"Saved: {path}")

    def _open_folder(self, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        import subprocess

        subprocess.run(["open", str(folder.resolve())], check=False)


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Pixel Shuffler")
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
