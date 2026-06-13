# Copyright (c) 2025, Li-Hung Wang <therockleona@gmail.com>
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# - Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
# - Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# - Neither the name of the copyright holder nor the names of its contributors
#   may be used to endorse or promote products derived from this software
#   without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""
GUI for Vision features
"""

import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon, QImage, QPixmap
from PySide6.QtWidgets import QDockWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog, QWidget
from PySide6.QtGui import QPainter, QPen, QFont

from .._gui_common import PilotFeature
from ._vision_yolo import _yolo_detector

class VisionGui(PilotFeature):

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        # Initialize Vision GUI components here
        self.widget = QDockWidget("Vision", self._mainWindow)
        self.widget.resize(400, 300)

        # Create central widget for the dock widget
        self.central_widget = QWidget()
        self.widget.setWidget(self.central_widget)
        
        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

        self._status_layout = QHBoxLayout()
        self._status_layout.setSpacing(10)  # Set spacing between items
        self._status_layout.setContentsMargins(0, 0, 0, 0)  # Remove margins
        self._status_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)  # Align items to left
        self.layout.addLayout(self._status_layout)

        self.status_light_icon = QLabel()
        red_icon = QIcon.fromTheme("media-record")
        self.status_light_icon.setPixmap(red_icon.pixmap(16, 16))

        self._status_layout.addWidget(self.status_light_icon)
        self.status_label = QLabel("Not Activated")
        self._status_layout.addWidget(self.status_label)

        self.image_instance = QImage()
        self.image_label = QLabel()
        self.layout.addWidget(self.image_label, 1)  # Add stretch factor of 1

        self.load_image_button = QPushButton("Load Image")
        self.load_image_button.clicked.connect(self.click_load_image)
        self.layout.addWidget(self.load_image_button)
        
        self.is_vision_active = False
        self.vision_button = QPushButton("Activate Vision")
        self.vision_button.clicked.connect(self.toggle_activation)
        self.layout.addWidget(self.vision_button)

        self._mainWindow.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.widget)


    def populate_menu(self):
        # Code to populate the menu for Vision GUI
        self._add_menu_item(
            menu=self._mgr.windowMenu,
            text="Computer Vision",
            tip="Open / Close Computer Vision Window",
            func=self.toggle_visibility,
        )

    def click_load_image(self):
        # Code to handle image loading
        file_name, _ = QFileDialog.getOpenFileName(
            self.widget,
            "Open Image",
            "",
            "Image Files (*.png *.jpg *.bmp);;All Files (*)"
        )

        if file_name:
            self.image_instance.load(file_name)
            scaled_image = self.image_instance.scaled(
                self.image_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.image_label.setPixmap(QPixmap.fromImage(scaled_image))

            if not self.is_vision_active:
                return
            
            image_array = self.qimage_to_numpy(self.image_instance)
            detections = _yolo_detector.detect(image_array)

            self.draw_bboxes(image_array, detections)

    def draw_bboxes(self, image, detections):
        # Code to draw bounding boxes on the image based on detections
        
        for det in detections:
            x1, y1, w, h = det['bbox']
            label = det['label']
            score = det['score']

            # Convert numpy array to QImage for drawing
            height, width, channel = image.shape
            bytes_per_line = 3 * width
            q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
            
            # Create QPainter to draw on the image
            
            painter = QPainter(q_image)
            
            # Set pen for bounding box
            pen = QPen(Qt.GlobalColor.red)
            pen.setWidth(3)
            painter.setPen(pen)
            
            # Draw rectangle (bounding box)
            painter.drawRect(x1, y1, w, h)
            
            # Set font and draw label text
            font = QFont()
            font.setPointSize(20)
            painter.setFont(font)
            
            text = f"{label}: {score:.2f}"
            painter.drawText(x1, y1 - 5, text)
            
            painter.end()
            
        # Update the displayed image
        scaled_image = q_image.scaled(
            self.image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.image_label.setPixmap(QPixmap.fromImage(scaled_image))

    def qimage_to_numpy(self, qimage):
        # Convert QImage to numpy array
        qimage = qimage.convertToFormat(QImage.Format.Format_RGB888)
        width = qimage.width()
        height = qimage.height()

        ptr = qimage.bits()
        arr = np.array(ptr).reshape((height, width, 3))
        return arr

    def toggle_activation(self):
        # Code to toggle activation of Vision features
        if self.is_vision_active:
            self.is_vision_active = False
            _yolo_detector.deactivate()

            self.vision_button.setText("Activate Vision")
            self.status_label.setText("Vision Module Deactivated")

            red_icon = QIcon.fromTheme("media-record")
            self.status_light_icon.setPixmap(red_icon.pixmap(16, 16))
        else:
            self.is_vision_active = True
            _yolo_detector.activate()

            self.vision_button.setText("Deactivate Vision")
            self.status_label.setText("Vision Module Activated")

            green_icon = QIcon.fromTheme("media-playback-start")
            self.status_light_icon.setPixmap(green_icon.pixmap(16, 16))

    def toggle_visibility(self):
        # Code to toggle visibility of Vision GUI
        if self.widget.isVisible():
            self.widget.hide()
        else:
            self.widget.show()