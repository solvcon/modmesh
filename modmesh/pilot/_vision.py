"""
Show YOLO Dock
"""

from ultralytics import YOLO
from ultralytics.utils import set_logging, LOGGER
import io
import logging

from ._gui_common import PilotFeature


class ConsoleHandler(logging.Handler):
    """
    Custom logging handler for RPythonConsoleDockWidget
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
                self.console_widget.writeToHistory(msg + '\n')
        except Exception:
            self.handleError(record)

class YOLOFeature(PilotFeature):
    """
    YOLO Dock for object detection.
    """

    def __init__(self, *args, **kw):
        super(YOLOFeature, self).__init__(*args, **kw)

    def _show_yolo_dock(self):
        """
        Show the YOLO dock widget.
        """
        self._mgr.showDockWidget("yolo")

def yolo_detect(np_img):
    """
    Receive a numpy array image and return a list of dicts, each containing bbox, label, and score.
    The bbox format is [x, y, w, h], label is the class name, and score is the confidence score.
    """
    # Load model (can be loaded globally only once)
    global model, logger
    if 'logger' not in globals():
        set_logging(name="ultralytics", verbose=True) 
        logger = logging.getLogger("ultralytics")
        
        # Try to get the console widget from the pilot core manager
        try:
            from . import _pilot_core
            console_widget = _pilot_core.mgr.pycon

            # Create custom handler
            console_handler = ConsoleHandler(console_widget)
            console_handler.setLevel(logging.DEBUG)

            # Set formatter
            formatter = logging.Formatter('[%(name)s] %(levelname)s: %(message)s')
            console_handler.setFormatter(formatter)
            
            # Clear existing handlers and add the custom handler
            logger.handlers.clear()
            logger.addHandler(console_handler)
            logger.setLevel(logging.DEBUG)
            logger.propagate = False  # Avoid duplicate output

        except Exception as e:
            # If unable to get console widget, fall back to original method
            log_stream = io.StringIO()
            handler = logging.StreamHandler(log_stream)
            logger.addHandler(handler)
            logger.setLevel(logging.DEBUG)

    if 'model' not in globals():
        model = YOLO('./modmesh/pilot/yolo11n.pt')  # Please check the model path

    results = model(np_img)
    boxes = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        label_idx = int(box.cls[0])
        label = results[0].names[label_idx]
        score = float(box.conf[0])
        boxes.append({
            'bbox': [int(x1), int(y1), int(x2 - x1), int(y2 - y1)],
            'label': label,
            'score': score
        })

    return boxes