# Copyright (c) 2019, Yung-Yu Chen <yyc@solvcon.net>
# BSD-style license; see COPYING


import unittest

import numpy as np

import modmesh


class TimeRegistryTC(unittest.TestCase):

    def test_singleton(self):

        ret = modmesh.time_registry.report()
        self.assertEqual("", ret)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
