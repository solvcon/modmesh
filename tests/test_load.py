# Copyright (c) 2019, Yung-Yu Chen <yyc@solvcon.net>
# BSD 3-Clause License, see COPYING


import unittest

import modmesh


class ModMeshTC(unittest.TestCase):

    def test_dummy(self):

        self.assertEqual('dummy', modmesh._modmesh.dummy)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
