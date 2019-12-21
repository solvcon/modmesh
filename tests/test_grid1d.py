# Copyright (c) 2019, Yung-Yu Chen <yyc@solvcon.net>
# BSD-style license; see COPYING


import unittest

import numpy as np

import modmesh


class StaticGrid1dTC(unittest.TestCase):

    def test_getset(self):

        gd = modmesh.StaticGrid1d(11)

        self.assertEqual(11, gd.nx)
        self.assertEqual(11, len(gd))

        for it in range(len(gd)):
            gd[it] = it
        self.assertEqual([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], list(gd))

    def test_ndarray(self):

        gd = modmesh.StaticGrid1d(11)

        self.assertEqual(np.float64, gd.coord.dtype)
        gd.coord = np.arange(10, -1, -1, dtype='float64')
        self.assertEqual([10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0], list(gd))

        def check_life_cycle():

            gd2 = modmesh.StaticGrid1d(5)
            gd2.coord = np.arange(5, dtype='float64')
            return gd2.coord

        coord = check_life_cycle()
        self.assertEqual([0, 1, 2, 3, 4], coord.tolist())

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
