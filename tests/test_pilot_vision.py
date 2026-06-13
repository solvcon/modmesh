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


import io
import os
import logging
import unittest

import numpy as np
import requests
from PIL import Image
from ultralytics import YOLO

import modmesh
from modmesh.pilot.vision._vision_yolo import _yolo_detector

try:
    from modmesh import pilot
except ImportError:
    pilot = None


@unittest.skipUnless(modmesh.HAS_PILOT, "Qt pilot is not built")
class VisionTC(unittest.TestCase):

    def test_yolo_detector(self):
        self.assertIsNotNone(_yolo_detector)

        self.assertFalse(_yolo_detector.model)
        self.assertFalse(_yolo_detector.logger)
        self.assertFalse(_yolo_detector.is_active)

        _yolo_detector.activate()

        self.assertTrue(_yolo_detector.is_active)
        self.assertIsInstance(_yolo_detector.logger, logging.Logger)
        self.assertIsInstance(_yolo_detector.model, YOLO)

        _yolo_detector.deactivate()

    def test_yolo_inference(self):

        TESTDIR = os.path.abspath(os.path.dirname(__file__))
        DATADIR = os.path.join(TESTDIR, "data/jpg")

        _yolo_detector.activate()

        # Convert to numpy array
        image = Image.open(os.path.join(DATADIR, "cat.jpg")).convert("RGB")
        image_array = np.array(image)

        results = _yolo_detector.detect(image_array)

        self.assertIsNotNone(results)
        self.assertGreater(len(results), 0)

        first_result = results[0]
        self.assertTrue("bbox" in first_result)
        self.assertTrue("label" in first_result)
        self.assertTrue("score" in first_result)

        _yolo_detector.deactivate()


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
