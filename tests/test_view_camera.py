import unittest
import numpy as np
from modmesh import view


def angle_axis(angle_deg, axis):
    a = axis
    angle = np.radians(angle_deg)

    _cos = np.cos(angle)
    _sin = np.sin(angle)

    rotation = np.identity(3) * _cos

    m = np.outer(a, a) * (1 - _cos)
    n = np.array([
        [0, -a[2], a[1]],
        [a[2], 0, -a[0]],
        [-a[1], a[0], 0]
    ]) * _sin

    rotation += m + n

    return rotation


class ViewFPSCameraTC(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        widget = view.RManager.instance.setUp().add3DWidget()
        widget.setCameraType("fps")

        cls.widget = widget
        cls.controller = widget.cameraController()

    def test_translation(self):
        dt = 0.01
        linear_speed = self.controller.linear_speed()

        delta = dt * linear_speed
        initial_position = self.controller.position()

        self.controller.updateCameraPosition(x=1, dt=dt)
        self.assertAlmostEqual(self.controller.position()[0], initial_position[0] + delta, delta=1e-2)

        self.controller.updateCameraPosition(y=1, dt=dt)
        self.assertAlmostEqual(self.controller.position()[1], initial_position[1] + delta, delta=1e-2)

        # camera moves in negative z direction
        self.controller.updateCameraPosition(z=1, dt=dt)
        self.assertAlmostEqual(self.controller.position()[2], initial_position[2] - delta, delta=1e-2)

    def test_rotation(self):
        dt = 0.01
        rotation = 0.5
        look_speed = self.controller.look_speed()

        initial_view_center = self.controller.view_center()
        initial_view_vector = self.controller.view_vector()

        # test camera does not rotate when left mouse button is not pressed
        self.controller.updateCameraPosition(yaw=rotation, pitch=rotation, dt=dt)
        self.assertEqual(self.controller.view_vector(), initial_view_vector)
        self.assertEqual(self.controller.view_center(), initial_view_center)

        # test camera rotates around y-axis
        self.controller.updateCameraPosition(yaw=rotation, dt=dt, left_mouse_button=True)

        rotation_matrix = angle_axis(rotation * dt * look_speed, (0, 1, 0))
        rotated_vector = np.array(initial_view_vector) @ rotation_matrix
        view_vector = self.controller.view_vector()

        self.assertAlmostEqual(rotated_vector[0], view_vector[0], delta=1e-2)
        self.assertAlmostEqual(rotated_vector[1], view_vector[1], delta=1e-2)
        self.assertAlmostEqual(rotated_vector[2], view_vector[2], delta=1e-2)

        # test camera rotates around x-axis
        self.controller.updateCameraPosition(pitch=rotation, dt=dt, left_mouse_button=True)

        rotation_matrix = angle_axis(rotation * dt * look_speed, (-1, 0, 0))
        rotated_vector = np.array(rotated_vector) @ rotation_matrix
        view_vector = self.controller.view_vector()

        self.assertAlmostEqual(rotated_vector[0], view_vector[0], delta=1e-2)
        self.assertAlmostEqual(rotated_vector[1], view_vector[1], delta=1e-2)
        self.assertAlmostEqual(rotated_vector[2], view_vector[2], delta=1e-2)


class ViewCameraTC(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        widget = view.RManager.instance.setUp().add3DWidget()
        widget.setCameraType("orbit")

        cls.widget = widget
        cls.controller = widget.cameraController()

    def test_translation(self):
        dt = 0.01
        linear_speed = self.controller.linear_speed()

        delta = dt * linear_speed
        initial_position = self.controller.position()

        self.controller.updateCameraPosition(x=1, dt=dt)
        self.assertAlmostEqual(self.controller.position()[0], initial_position[0] + delta, delta=1e-2)

        self.controller.updateCameraPosition(y=1, dt=dt)
        self.assertAlmostEqual(self.controller.position()[1], initial_position[1] + delta, delta=1e-2)

        # camera moves in negative z direction
        self.controller.updateCameraPosition(z=1, dt=dt)
        self.assertAlmostEqual(self.controller.position()[2], initial_position[2] - delta, delta=1e-2)

    def test_rotation(self):
        dt = 0.01
        rotation = 0.5
        look_speed = self.controller.look_speed()

        initial_view_center = self.controller.view_center()
        initial_view_vector = self.controller.view_vector()

        # test camera does not rotate when right mouse button is not pressed
        self.controller.updateCameraPosition(yaw=rotation, pitch=rotation, dt=dt)
        self.assertEqual(self.controller.view_vector(), initial_view_vector)
        self.assertEqual(self.controller.view_center(), initial_view_center)

        # test camera rotates around y-axis
        self.controller.updateCameraPosition(yaw=rotation, dt=dt, right_mouse_button=True)

        rotation_matrix = angle_axis(rotation * dt * look_speed, (0, -1, 0))
        rotated_vector = np.array(initial_view_vector) @ rotation_matrix
        view_vector = self.controller.view_vector()

        self.assertAlmostEqual(rotated_vector[0], view_vector[0], delta=1e-2)
        self.assertAlmostEqual(rotated_vector[1], view_vector[1], delta=1e-2)
        self.assertAlmostEqual(rotated_vector[2], view_vector[2], delta=1e-2)

        # test camera rotates around x-axis
        self.controller.updateCameraPosition(pitch=rotation, dt=dt, right_mouse_button=True)

        rotation_matrix = angle_axis(rotation * dt * look_speed, (1, 0, 0))
        rotated_vector = np.array(rotated_vector) @ rotation_matrix
        view_vector = self.controller.view_vector()

        self.assertAlmostEqual(rotated_vector[0], view_vector[0], delta=1e-2)
        self.assertAlmostEqual(rotated_vector[1], view_vector[1], delta=1e-2)
        self.assertAlmostEqual(rotated_vector[2], view_vector[2], delta=1e-2)

