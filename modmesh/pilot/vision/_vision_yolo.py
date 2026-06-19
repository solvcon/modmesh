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
Vision feature: YOLO object detection integration
"""

from ultralytics import YOLO
from ultralytics.utils import set_logging
import io
import logging

__all__ = [
    "_yolo_detector",
]


class ConsoleHandler(logging.Handler):
    """
    Custom logging handler for sending logs to python console widget
    """

    def __init__(self, console_widget):
        super().__init__()
        self.console_widget = console_widget

    def emit(self, record):
        """
        Send log record to console widget
        """
        try:
            msg = self.format(record)
            if self.console_widget:
                self.console_widget.writeToHistory(msg + "\n")
        except Exception:
            self.handleError(record)


def set_up_logger():
    global logger
    if "logger" not in globals():
        set_logging(name="ultralytics", verbose=True)
        logger = logging.getLogger("ultralytics")

        try:
            from .. import mgr

            console_widget = mgr.pycon

            if console_widget is None:
                raise RuntimeError("Python console widget is not available")

            # Create custom handler
            console_handler = ConsoleHandler(console_widget)
            console_handler.setLevel(logging.DEBUG)

            # Set formatter
            formatter = logging.Formatter("[%(name)s] %(levelname)s: %(message)s")
            console_handler.setFormatter(formatter)

            # Clear existing handlers and add the custom handler
            logger.handlers.clear()
            logger.addHandler(console_handler)
            logger.setLevel(logging.DEBUG)
            logger.propagate = False  # Avoid duplicate output

            # Test the console widget by writing a message
            if console_widget:
                console_widget.writeToHistory(
                    "[Vision] Logger initialized successfully\n"
                )

        except Exception as e:
            # If unable to get console widget, fall back to original method
            # Print the error for debugging
            print(f"[Vision] Failed to set up console widget logger: {e}")

            log_stream = io.StringIO()
            handler = logging.StreamHandler(log_stream)
            logger.addHandler(handler)
            logger.setLevel(logging.DEBUG)

            # Also add a console handler for fallback
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter("[%(name)s] %(levelname)s: %(message)s")
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)


class YoloDetector:
    """
    YOLO Detector class to manage model loading and detection
    """

    def __init__(self):
        self.model = None
        self.logger = None
        self.is_active = False

    def activate(self, model_path="./thirdparty/yolo11n.pt"):
        if self.model is None:
            self.model = YOLO(model_path)
            set_up_logger()
            self.logger = logging.getLogger("ultralytics")
        self.is_active = True

    def deactivate(self):
        self.is_active = False

    def detect(self, np_img):
        """
        Perform detection on the input image

        Args:
            np_img (numpy.ndarray): Input image as a numpy array.

        Returns:
            list of dict: Each dict contains 'bbox', 'label', and 'score'.
        """
        if self.model is None:
            raise RuntimeError(
                "YOLO model is not loaded. Please activate the detector first."
            )

        results = self.model(np_img)
        boxes = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            label_idx = int(box.cls[0])
            label = results[0].names[label_idx]
            score = float(box.conf[0])

            box_obj = {
                "bbox": [int(x1), int(y1), int(x2 - x1), int(y2 - y1)],
                "label": label,
                "score": score,
            }
            self.logger.debug(
                f"Detected {label} with confidence {score:.2f} at [{x1}, {y1}, {x2}, {y2}]"
            )
            boxes.append(box_obj)

        return boxes


_yolo_detector = YoloDetector()
