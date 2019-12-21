# Copyright (c) 2019, Yung-Yu Chen <yyc@solvcon.net>
# BSD 3-Clause License, see COPYING


import unittest

import modmesh


class Grid1dTC(unittest.TestCase):

    def test_ndim(self):

        self.assertEqual(1, modmesh.Grid1d.NDIM);
        self.assertEqual(2, modmesh.Grid2d.NDIM);
        self.assertEqual(3, modmesh.Grid3d.NDIM);

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
