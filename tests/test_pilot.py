# Copyright (c) 2022, Yung-Yu Chen <yyc@solvcon.net>
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


import unittest
import numpy as np
import os

import modmesh
try:
    from modmesh import pilot
except ImportError:
    pilot = None
try:
    import PUI.PySide6
except ImportError:
    # Bypass PUI import error if modmesh is built without Qt PUI may not be
    # installed in this case.
    # If modmesh is built with Qt, the PilotTC will check if PUI is working
    # or not.
    pass

GITHUB_ACTIONS = os.getenv('GITHUB_ACTIONS', False)


@unittest.skipUnless(modmesh.HAS_PILOT, "Qt pilot is not built")
class PilotTC(unittest.TestCase):

    def test_import(self):
        self.assertTrue(hasattr(modmesh.pilot, "mgr"))
        self.assertTrue(hasattr(PUI.PySide6, "PUINode"))
        self.assertTrue(hasattr(PUI.PySide6, "PUIView"))
        self.assertEqual(PUI.PySide6.PUI_BACKEND, "PySide6",
                         "PUI backebd mismatch")

    @unittest.skip("headless testing is not ready")
    def test_pycon(self):
        self.assertTrue(pilot.mgr.pycon.python_redirect)
        pilot.mgr.pycon.python_redirect = False
        self.assertFalse(pilot.mgr.pycon.python_redirect)


class PilotCameraTB:
    camera_type = None

    @classmethod
    def setUpClass(cls):
        if modmesh.HAS_PILOT:
            widget = pilot.RManager.instance.setUp().add3DWidget()

            if cls.camera_type is not None:
                widget.setCameraType(cls.camera_type)

            cls.widget = widget
            cls.camera = widget.camera

    @classmethod
    def tearDownClass(cls):
        if modmesh.HAS_PILOT:
            cls.widget.close_and_destroy()

    def angle_axis(self, angle_deg, axis):
        a = axis
        angle = np.radians(angle_deg)

        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)

        rotation = np.identity(3) * cos_angle

        m = np.outer(a, a) * (1 - cos_angle)
        n = np.array([
            [0, -a[2], a[1]],
            [a[2], 0, -a[0]],
            [-a[1], a[0], 0]
        ]) * sin_angle

        rotation += m + n

        return rotation

    def normalize(self, vec):
        return vec / np.linalg.norm(vec)


@unittest.skipIf(GITHUB_ACTIONS or not modmesh.HAS_PILOT,
                 "GUI is not available in GitHub Actions")
class PilotCommonCameraTC(PilotCameraTB, unittest.TestCase):
    def test_value_get_set(self):
        c = self.camera

        for camera_type in ["fps", "orbit"]:
            self.widget.setCameraType(camera_type)
            c.reset()

            c.linear_speed = 123.0
            self.assertEqual(c.linear_speed, 123.0)

            c.look_speed = 456.0
            self.assertEqual(c.look_speed, 456.0)

    def test_vector_get_set(self):
        c = self.camera

        for camera_type in ["fps", "orbit"]:
            self.widget.setCameraType(camera_type)
            c.reset()

            c.position = (1, 2, 3)
            c.view_center = (4, 5, 6)
            c.up_vector = (7, 8, 9)

            self.assertEqual(c.position, (1, 2, 3))
            self.assertEqual(c.view_center, (4, 5, 6))
            self.assertEqual(c.up_vector, (7, 8, 9))

    def test_default_values(self):
        c = self.camera

        for camera_type in ["fps", "orbit"]:
            self.widget.setCameraType(camera_type)
            c.reset()

            original_position = c.default_position
            original_view_center = c.default_view_center
            original_up_vector = c.default_up_vector
            original_linear_speed = c.default_linear_speed
            original_look_speed = c.default_look_speed

            c.default_position = (1, 2, 3)
            c.default_view_center = (4, 5, 6)
            c.default_up_vector = (7, 8, 9)
            c.default_linear_speed = 123.0
            c.default_look_speed = 456.0

            c.reset()

            self.assertEqual(c.position, (1, 2, 3))
            self.assertEqual(c.view_center, (4, 5, 6))
            self.assertEqual(c.up_vector, (7, 8, 9))
            self.assertEqual(c.linear_speed, 123.0)
            self.assertEqual(c.look_speed, 456.0)

            c.default_position = original_position
            c.default_view_center = original_view_center
            c.default_up_vector = original_up_vector
            c.default_linear_speed = original_linear_speed
            c.default_look_speed = original_look_speed

    def test_translation(self):
        c = self.camera

        for camera_type in ["orbit"]:
            self.widget.setCameraType(camera_type)
            c.reset()

            speed = c.linear_speed
            delta_vec = np.array([speed, speed, -speed])
            new_position = np.array(c.position) + delta_vec
            new_view_center = np.array(c.view_center) + delta_vec

            c.move(x=1)
            self.assertEqual(c.position[0], new_position[0])

            c.move(y=1)
            self.assertEqual(c.position[1], new_position[1])

            # camera moves in negative z direction
            c.move(z=1)
            self.assertEqual(c.position[2], new_position[2])

            # camera view center should move with camera
            self.assertEqual(c.view_center[0], new_view_center[0])
            self.assertEqual(c.view_center[1], new_view_center[1])
            self.assertEqual(c.view_center[2], new_view_center[2])


@unittest.skipIf(GITHUB_ACTIONS or not modmesh.HAS_PILOT,
                 "GUI is not available in GitHub Actions")
class PilotFPSCameraTC(PilotCameraTB, unittest.TestCase):
    camera_type = "fps"

    def setUp(self):
        self.camera.reset()

    def test_reset(self):
        c = self.camera

        initial_position = c.position
        initial_view_center = c.view_center
        initial_up_vector = c.up_vector

        c.move(x=0.1, y=0.1, z=0.1)
        c.move(yaw=1, pitch=1, left_mouse_button=True)

        self.assertNotEqual(c.position, initial_position)
        self.assertNotEqual(c.view_center, initial_view_center)
        self.assertNotEqual(c.up_vector, initial_up_vector)

        c.reset()

        self.assertEqual(c.position, initial_position)
        self.assertEqual(c.view_center, initial_view_center)
        self.assertEqual(c.up_vector, initial_up_vector)

    def test_rotation(self):
        c = self.camera

        angle = c.look_speed

        initial_view_center = c.view_center
        initial_view_vector = c.view_vector

        # test camera does not rotate when left mouse button is not pressed
        c.move(yaw=1, pitch=1)
        self.assertEqual(c.view_vector, initial_view_vector)
        self.assertEqual(c.view_center, initial_view_center)

        # test camera rotates around y-axis
        c.move(yaw=1, left_mouse_button=True)

        rot_matrix = self.angle_axis(angle, (0, 1, 0))
        rot_vector = np.array(initial_view_vector) @ rot_matrix

        self.assertAlmostEqual(rot_vector[0], c.view_vector[0], delta=1e-5)
        self.assertAlmostEqual(rot_vector[1], c.view_vector[1], delta=1e-5)
        self.assertAlmostEqual(rot_vector[2], c.view_vector[2], delta=1e-5)

        # test camera rotates around x-axis
        old_view_vector = c.view_vector
        c.move(pitch=1, left_mouse_button=True)

        x_basis = -self.normalize(np.cross(old_view_vector, c.up_vector))
        rot_matrix = self.angle_axis(angle, x_basis)
        rot_vector = np.array(old_view_vector) @ rot_matrix

        self.assertAlmostEqual(rot_vector[0], c.view_vector[0], delta=1e-5)
        self.assertAlmostEqual(rot_vector[1], c.view_vector[1], delta=1e-5)
        self.assertAlmostEqual(rot_vector[2], c.view_vector[2], delta=1e-5)

        # test view center moved with the camera
        view_center = c.view_center
        new_view_center = c.view_vector + np.array(c.position)

        self.assertAlmostEqual(view_center[0], new_view_center[0], delta=1e-5)
        self.assertAlmostEqual(view_center[1], new_view_center[1], delta=1e-5)
        self.assertAlmostEqual(view_center[2], new_view_center[2], delta=1e-5)


@unittest.skipIf(GITHUB_ACTIONS or not modmesh.HAS_PILOT,
                 "GUI is not available in GitHub Actions")
class PilotOrbitCameraTC(PilotCameraTB, unittest.TestCase):
    camera_type = "orbit"

    def setUp(self):
        self.camera.reset()

    def test_reset(self):
        c = self.camera

        initial_position = c.position
        initial_view_vector = c.view_vector
        initial_view_center = c.view_center
        initial_up_vector = c.up_vector

        c.move(x=0.1, y=0.1, z=0.1)
        c.move(yaw=1, pitch=1, right_mouse_button=True)

        self.assertNotEqual(c.position, initial_position)
        self.assertNotEqual(c.view_vector, initial_view_vector)
        self.assertNotEqual(c.up_vector, initial_up_vector)

        c.reset()

        self.assertEqual(c.position, initial_position)
        self.assertEqual(c.view_vector, initial_view_vector)
        self.assertEqual(c.view_center, initial_view_center)
        self.assertEqual(c.up_vector, initial_up_vector)

    def test_rotation(self):
        c = self.camera

        angle = c.look_speed
        initial_view_center = c.view_center
        initial_view_vector = c.view_vector

        # test camera does not rotate when right mouse button is not pressed
        c.move(yaw=1, pitch=1)
        self.assertEqual(c.view_vector, initial_view_vector)
        self.assertEqual(c.view_center, initial_view_center)

        # test camera rotates around y-axis
        c.move(yaw=1, right_mouse_button=True)

        rot_matrix = self.angle_axis(angle, (0, -1, 0))
        rot_vector = np.array(initial_view_vector) @ rot_matrix

        self.assertAlmostEqual(rot_vector[0], c.view_vector[0], delta=1e-5)
        self.assertAlmostEqual(rot_vector[1], c.view_vector[1], delta=1e-5)
        self.assertAlmostEqual(rot_vector[2], c.view_vector[2], delta=1e-5)

        # test camera rotates around x-axis
        old_view_vector = c.view_vector
        c.move(pitch=1, right_mouse_button=True)

        x_basis = self.normalize(np.cross(old_view_vector, c.up_vector))
        rot_matrix = self.angle_axis(angle, x_basis)
        rot_vector = np.array(old_view_vector) @ rot_matrix

        self.assertAlmostEqual(rot_vector[0], c.view_vector[0], delta=1e-5)
        self.assertAlmostEqual(rot_vector[1], c.view_vector[1], delta=1e-5)
        self.assertAlmostEqual(rot_vector[2], c.view_vector[2], delta=1e-5)

        # camera view center should not change
        self.assertEqual(c.view_center, initial_view_center)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
