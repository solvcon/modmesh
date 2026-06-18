# Copyright (c) 2024, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


import unittest

import modmesh as mm
from modmesh.pilot.airfoil import _naca


class Naca4TC(unittest.TestCase):
    def test_npoint(self):
        def _check_size(naca4):
            points = naca4.calc_points(5)
            self.assertEqual(points.ndim, 2)
            self.assertEqual(len(points), 11)
            self.assertEqual((11, 2), points.pack_array().shape)
            points = naca4.calc_points(11)
            self.assertEqual(points.ndim, 2)
            self.assertEqual(len(points), 23)
            self.assertEqual((23, 2), points.pack_array().shape)

        _check_size(_naca.Naca4(number='0012', open_trailing_edge=False,
                                cosine_spacing=False))
        _check_size(_naca.Naca4(number='0012', open_trailing_edge=True,
                                cosine_spacing=False))
        _check_size(_naca.Naca4(number='0012', open_trailing_edge=True,
                                cosine_spacing=False))
        _check_size(_naca.Naca4(number='0012', open_trailing_edge=True,
                                cosine_spacing=True))


class Naca4SamplerTC(unittest.TestCase):
    def test_construction(self):
        w = mm.WorldFp64()
        naca4 = _naca.Naca4(number='0012', open_trailing_edge=False,
                            cosine_spacing=False)
        _naca.Naca4Sampler(w, naca4)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
