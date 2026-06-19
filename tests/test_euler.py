# Copyright (c) 2024, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

import unittest

import solvcon as sc


class EulerCoreTC(unittest.TestCase):

    def test_construct(self):
        mh = sc.StaticMesh(ndim=2, nnode=0)
        sc.EulerCore(mesh=mh, time_increment=0.0)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
