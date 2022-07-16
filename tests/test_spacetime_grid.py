# Copyright (c) 2018, Yung-Yu Chen <yyc@solvcon.net>
# BSD 3-Clause License, see COPYING

import unittest

import numpy as np

from modmesh import spacetime as libst


class GridTC(unittest.TestCase):

    def setUp(self):

        self.grid10 = libst.Grid(xmin=0.0, xmax=10.0, nelm=10)

    def test_construction(self):

        with self.assertRaisesRegex(
                ValueError,
                r"Grid::Grid\(xmin=0, xmax=10, ncelm=0\) invalid argument: "
                r"ncelm smaller than 1",
        ):
            libst.Grid(0, 10, 0)

        with self.assertRaisesRegex(
                ValueError,
                r"Grid::Grid\(xmin=10, xmax=10, ncelm=10\) invalid arguments: "
                r"xmin >= xmax",
        ):
            libst.Grid(10, 10, 10)

        with self.assertRaisesRegex(
                ValueError,
                r"Grid::Grid\(xmin=11, xmax=10, ncelm=10\) invalid arguments: "
                r"xmin >= xmax",
        ):
            libst.Grid(11, 10, 10)

        # Simply test for passing.
        libst.Grid(xloc=np.arange(2) * 0.1)

        for s in [0, 1]:
            with self.assertRaisesRegex(
                    ValueError,
                    r"Grid::init_from_array\(xloc\) invalid arguments: "
                    r"xloc.size\(\)=%d smaller than 2" % s
            ):
                libst.Grid(xloc=np.arange(s) * 0.1)

        with self.assertRaisesRegex(
                ValueError,
                r"Grid::init_from_array\(xloc\) invalid arguments: "
                r"xloc\[0\]=1 >= xloc\[1\]=0.9"
        ):
            libst.Grid(xloc=np.arange(10, -1, -1) * 0.1)

    def test_xcoord(self):

        nx = (self.grid10.ncelm + self.grid10.BOUND_COUNT)*2 + 1
        golden_x = np.arange(0.0, 10.1, 0.5)
        golden_front = golden_x[0] - golden_x[self.grid10.BOUND_COUNT:0:-1]
        golden_back = golden_x[-1] - golden_x[-2:-self.grid10.BOUND_COUNT-2:-1]
        golden_back += golden_x[-1]
        golden_x = np.hstack([golden_front, golden_x, golden_back])

        self.assertEqual(nx, len(self.grid10.xcoord))
        self.assertEqual(golden_x.tolist(),
                         self.grid10.xcoord.ndarray.tolist())
        self.grid10.xcoord.ndarray.fill(10)
        self.assertEqual([10]*nx, self.grid10.xcoord.ndarray.tolist())

    def test_number(self):

        self.assertEqual(10, self.grid10.ncelm)
        self.assertEqual(11, self.grid10.nselm)

    def test_str(self):

        self.assertEqual("Grid(xmin=0, xmax=10, ncelm=10)",
                         str(self.grid10))

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
