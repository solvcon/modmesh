# Copyright (c) 2020, Yung-Yu Chen <yyc@solvcon.net>
# BSD-style license; see COPYING


import unittest

import numpy as np

import modmesh


class BasicTC(unittest.TestCase):

    def test_ConcreteBuffer(self):

        buf = modmesh.ConcreteBuffer(10)
        ndarr = np.array(buf, copy=False)

        # initialization
        for it in range(len(buf)):
            buf[it] = it

        self.assertEqual(10, buf.nbytes)
        self.assertEqual(10, len(buf))

        self.assertEqual(np.int8, ndarr.dtype)

        with self.assertRaisesRegex(
            IndexError,
            "ConcreteBuffer: index 10 is out of bounds with size 10"
        ):
            buf[10]
        with self.assertRaisesRegex(
            IndexError, "index 10 is out of bounds for axis 0 with size 10"
        ):
            ndarr[10]

        buf2 = buf.clone()
        for it in range(len(buf)):
            buf[it] = 100+it
        self.assertEqual(list(buf), ndarr.tolist())
        self.assertEqual(list(range(100, 110)), ndarr.tolist())
        self.assertEqual(list(range(10)), list(buf2))

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
