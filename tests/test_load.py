# Copyright (c) 2019, Yung-Yu Chen <yyc@solvcon.net>
# BSD 3-Clause License, see COPYING


import unittest

import modmesh


class GridD1TC(unittest.TestCase):

    def test_ndim(self):

        self.assertEqual(1, modmesh.GridD1.NDIM);
        self.assertEqual(2, modmesh.GridD2.NDIM);
        self.assertEqual(3, modmesh.GridD3.NDIM);

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
