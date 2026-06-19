# Copyright (c) 2019, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


import unittest

import solvcon


class StaticGridTC(unittest.TestCase):

    def test_ndim(self):

        self.assertEqual(1, solvcon.StaticGrid1d.NDIM)
        self.assertEqual(2, solvcon.StaticGrid2d.NDIM)
        self.assertEqual(3, solvcon.StaticGrid3d.NDIM)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
