# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


import unittest
from solvcon.plot import plane_layer


class PlaneLayerTC(unittest.TestCase):
    def test_add_rect_with_string(self):
        layer = plane_layer.PlaneLayer()
        layer.add_figure("RECT N M1 70 800 180 40")

        ploys = layer.get_polys()
        self.assertEqual(ploys, [[
            [(70.0, 800.0), (250.0, 800.0)],
            [(250.0, 800.0), (250.0, 840.0)],
            [(250.0, 840.0), (70.0, 840.0)],
            [(70.0, 840.0), (70.0, 800.0)],
        ]])

    def test_add_poly_with_string(self):
        layer = plane_layer.PlaneLayer()
        layer.add_figure(
            "PGON N M1 70 720 410 720 410 920 70 920 "
            "70 880 370 880 370 760 70 760"
        )

        ploys = layer.get_polys()
        self.assertEqual(ploys, [[
            [(70.0, 720.0), (410.0, 720.0)],
            [(410.0, 720.0), (410.0, 920.0)],
            [(410.0, 920.0), (70.0, 920.0)],
            [(70.0, 920.0), (70.0, 880.0)],
            [(70.0, 880.0), (370.0, 880.0)],
            [(370.0, 880.0), (370.0, 760.0)],
            [(370.0, 760.0), (70.0, 760.0)],
            [(70.0, 760.0), (70.0, 720.0)],
        ]])

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
