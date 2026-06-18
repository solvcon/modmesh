# Copyright (c) 2019, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


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

    def test_coord(self):

        gd = modmesh.StaticGrid1d(11)

        # gd.coord is a SimpleArray.
        self.assertEqual(np.float64, gd.coord.ndarray.dtype)
        gd.coord.ndarray[:] = np.arange(11, dtype='float64')
        self.assertEqual([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], list(gd))
        self.assertFalse(gd.coord.is_from_python)

        # Set gd.coord to an ndarray and uses its buffer.
        ndarray = np.arange(10, -1, -1, dtype='float64')
        gd.coord = ndarray
        self.assertEqual([10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0], list(gd))
        self.assertTrue(gd.coord.is_from_python)
        # The ndarray's buffer is used.
        ndarray[:] = 10
        self.assertEqual([10] * gd.nx, list(gd))

        # Cannot set coord with a different shape.
        with self.assertRaisesRegex(
            ValueError,
            r"80 bytes of input array differ from 88 bytes of internal array"
        ):
            gd.coord = np.arange(10, dtype='float64')

        # coord keeps living after the grid housing it deceased.
        def check_life_cycle():

            gd2 = modmesh.StaticGrid1d(5)
            gd2.coord = np.arange(5, dtype='float64')
            return gd2.coord

        coord = check_life_cycle()
        self.assertEqual([0, 1, 2, 3, 4], coord.ndarray.tolist())

    def test_fill(self):

        gd = modmesh.StaticGrid1d(11)
        gd.fill(102)
        self.assertEqual([102] * gd.nx, list(gd))

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
