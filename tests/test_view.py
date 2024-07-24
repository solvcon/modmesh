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

import modmesh
try:
    from modmesh import view
except ImportError:
    view = None
try:
    import PUI.PySide6
except ImportError:
    # Bypass PUI import error if modmesh is built without Qt PUI may not be
    # installed in this case.
    # If modmesh is built with Qt, the ViewTC will check if PUI is working
    # or not.
    pass


@unittest.skipUnless(modmesh.HAS_VIEW, "Qt view is not built")
class ViewTC(unittest.TestCase):

    def test_import(self):
        self.assertTrue(hasattr(modmesh.view, "mgr"))
        self.assertTrue(hasattr(PUI.PySide6, "PUINode"))
        self.assertTrue(hasattr(PUI.PySide6, "PUIView"))
        self.assertEqual(PUI.PySide6.PUI_BACKEND, "PySide6",
                         "PUI backebd mismatch")

    @unittest.skip("headless testing is not ready")
    def test_pycon(self):
        self.assertTrue(view.mgr.pycon.python_redirect)
        view.mgr.pycon.python_redirect = False
        self.assertFalse(view.mgr.pycon.python_redirect)


def angle_axis(angle_deg, axis):
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


@unittest.skip("GUI is not yet available for testing")
class ViewFPSCameraTC(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        widget = view.RManager.instance.setUp().add3DWidget()
        widget.setCameraType("fps")

        cls.widget = widget
        cls.controller = widget.cameraController()

        cls.move = cls.controller.updateCameraPosition
        
        cls.pos = cls.controller.position
        cls.view_vector = cls.controller.view_vector
        cls.view_center = cls.controller.view_center
        cls.up_vector = cls.controller.up_vector

    def setUp(self):
        self.widget.resetCamera()

    def test_reset(self):
        dt = 0.01
        initial_pos = self.pos()
        initial_view_vector = self.view_vector()
        initial_view_center = self.view_center()
        initial_up_vector = self.up_vector()
        
        self.move(x=1, y=1, z=1, dt=dt)
        self.move(yaw=1, pitch=1, dt=dt, left_mouse_button=True)

        self.assertNotEqual(self.pos(), initial_pos)
        self.assertNotEqual(self.view_vector(), initial_view_vector)
        self.assertNotEqual(self.view_center(), initial_view_center)
        self.assertNotEqual(self.up_vector(), initial_up_vector)

        self.widget.resetCamera()
        
        self.assertEqual(self.pos(), initial_pos)
        self.assertEqual(self.view_vector(), initial_view_vector)
        self.assertEqual(self.view_center(), initial_view_center)
        self.assertEqual(self.up_vector(), initial_up_vector)

    def test_translation(self):
        dt = 0.01
        linear_speed = self.controller.linear_speed()

        delta = dt * linear_speed
        new_pos = np.array(self.pos()) + [delta, delta, -delta]

        self.move(x=1, dt=dt)
        self.assertAlmostEqual(self.pos()[0], new_pos[0], delta=1e-2)

        self.move(y=1, dt=dt)
        self.assertAlmostEqual(self.pos()[1], new_pos[1], delta=1e-2)

        # camera moves in negative z direction
        self.move(z=1, dt=dt)
        self.assertAlmostEqual(self.pos()[2], new_pos[2], delta=1e-2)

    def test_rotation(self):
        dt = 0.01
        rotation = 0.5
        look_speed = self.controller.look_speed()

        initial_view_center = self.controller.view_center()
        initial_view_vector = self.controller.view_vector()

        # test camera does not rotate when left mouse button is not pressed
        self.move(yaw=rotation, pitch=rotation, dt=dt)
        self.assertEqual(self.controller.view_vector(), initial_view_vector)
        self.assertEqual(self.controller.view_center(), initial_view_center)

        # test camera rotates around y-axis
        self.move(yaw=rotation, dt=dt, left_mouse_button=True)

        rotation_matrix = angle_axis(rotation * dt * look_speed, (0, 1, 0))
        rotated_vector = np.array(initial_view_vector) @ rotation_matrix
        view_vector = self.controller.view_vector()

        self.assertAlmostEqual(rotated_vector[0], view_vector[0], delta=1e-2)
        self.assertAlmostEqual(rotated_vector[1], view_vector[1], delta=1e-2)
        self.assertAlmostEqual(rotated_vector[2], view_vector[2], delta=1e-2)

        # test camera rotates around x-axis
        self.move(pitch=rotation, dt=dt, left_mouse_button=True)

        rotation_matrix = angle_axis(rotation * dt * look_speed, (-1, 0, 0))
        rotated_vector = np.array(rotated_vector) @ rotation_matrix
        view_vector = self.controller.view_vector()

        self.assertAlmostEqual(rotated_vector[0], view_vector[0], delta=1e-2)
        self.assertAlmostEqual(rotated_vector[1], view_vector[1], delta=1e-2)
        self.assertAlmostEqual(rotated_vector[2], view_vector[2], delta=1e-2)


@unittest.skip("GUI is not yet available for testing")
class ViewOrbitCameraTC(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        widget = view.RManager.instance.setUp().add3DWidget()
        widget.setCameraType("orbit")

        cls.widget = widget
        cls.controller = widget.cameraController()

        cls.move = cls.controller.updateCameraPosition

        cls.pos = cls.controller.position
        cls.view_vector = cls.controller.view_vector
        cls.view_center = cls.controller.view_center
        cls.up_vector = cls.controller.up_vector

    def setUp(self):
        self.widget.resetCamera()

    def test_reset(self):
        dt = 0.01
        initial_pos = self.pos()
        initial_view_vector = self.view_vector()
        initial_view_center = self.view_center()
        initial_up_vector = self.up_vector()

        self.move(x=1, y=1, z=1, dt=dt)
        self.move(yaw=1, pitch=1, dt=dt, right_mouse_button=True)

        self.assertNotEqual(self.pos(), initial_pos)
        self.assertNotEqual(self.view_vector(), initial_view_vector)
        self.assertNotEqual(self.view_center(), initial_view_center)
        self.assertNotEqual(self.up_vector(), initial_up_vector)

        self.widget.resetCamera()

        self.assertEqual(self.pos(), initial_pos)
        self.assertEqual(self.view_vector(), initial_view_vector)
        self.assertEqual(self.view_center(), initial_view_center)
        self.assertEqual(self.up_vector(), initial_up_vector)

    def test_translation(self):
        dt = 0.01
        linear_speed = self.controller.linear_speed()

        delta = dt * linear_speed
        new_pos = np.array(self.pos()) + [delta, delta, -delta]

        self.move(x=1, dt=dt)
        self.assertAlmostEqual(self.pos()[0], new_pos[0], delta=1e-2)

        self.move(y=1, dt=dt)
        self.assertAlmostEqual(self.pos()[1], new_pos[1], delta=1e-2)

        # camera moves in negative z direction
        self.move(z=1, dt=dt)
        self.assertAlmostEqual(self.pos()[2], new_pos[2], delta=1e-2)

    def test_rotation(self):
        dt = 0.01
        rotation = 0.5
        look_speed = self.controller.look_speed()

        initial_view_center = self.controller.view_center()
        initial_view_vector = self.controller.view_vector()

        # test camera does not rotate when right mouse button is not pressed
        self.move(yaw=rotation, pitch=rotation, dt=dt)
        self.assertEqual(self.controller.view_vector(), initial_view_vector)
        self.assertEqual(self.controller.view_center(), initial_view_center)

        # test camera rotates around y-axis
        self.move(yaw=rotation, dt=dt, right_mouse_button=True)

        rotation_matrix = angle_axis(rotation * dt * look_speed, (0, -1, 0))
        rotated_vector = np.array(initial_view_vector) @ rotation_matrix
        view_vector = self.controller.view_vector()

        self.assertAlmostEqual(rotated_vector[0], view_vector[0], delta=1e-2)
        self.assertAlmostEqual(rotated_vector[1], view_vector[1], delta=1e-2)
        self.assertAlmostEqual(rotated_vector[2], view_vector[2], delta=1e-2)

        # test camera rotates around x-axis
        self.move(pitch=rotation, dt=dt, right_mouse_button=True)

        rotation_matrix = angle_axis(rotation * dt * look_speed, (1, 0, 0))
        rotated_vector = np.array(rotated_vector) @ rotation_matrix
        view_vector = self.controller.view_vector()

        self.assertAlmostEqual(rotated_vector[0], view_vector[0], delta=1e-2)
        self.assertAlmostEqual(rotated_vector[1], view_vector[1], delta=1e-2)
        self.assertAlmostEqual(rotated_vector[2], view_vector[2], delta=1e-2)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
