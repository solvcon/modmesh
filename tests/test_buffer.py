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

        ndarr2 = buf2.ndarray
        buf2[5] = 19
        self.assertEqual(19, ndarr2[5])

    def test_SimpleArray(self):

        sarr = modmesh.SimpleArrayFloat64((2, 3, 4))
        ndarr = np.array(sarr, copy=False)

        self.assertEqual(2*3*4*8, sarr.nbytes)
        self.assertEqual((2,3,4), sarr.shape)
        self.assertEqual((12,4,1), sarr.stride) # number of skip elements

        np.ravel(ndarr)[:] = np.arange(24) # initialize contents

        # Flat indexing interface.
        self.assertEqual(24, len(sarr)) # number of elements
        self.assertEqual(list(range(24)), [sarr[i] for i in range(24)])
        self.assertEqual(list(range(24)), list(sarr))
        # Multi-dimensional interface.
        v = 0
        for i in range(2):
            for j in range(3):
                for k in range(4):
                    self.assertEqual(v, sarr[i, j, k])
                    v += 1
        v = 100
        for i in range(2):
            for j in range(3):
                for k in range(4):
                    sarr[i, j, k] = v
                    v += 1

        with self.assertRaisesRegex(
            RuntimeError,
            "SimpleArray: shape byte count 184 differs from buffer 192"
        ):
            sarr.reshape(23)

        sarr2 = sarr.reshape(24)
        self.assertEqual((24,), sarr2.shape)
        self.assertEqual(np.arange(100,124).tolist(), [sarr2[i] for i in range(24)])
        self.assertEqual(np.arange(100,124).tolist(), list(sarr2))

        ndarr2 = sarr2.ndarray
        for i in range(24):
            sarr2[i] = 200 + i
        self.assertEqual(np.arange(200,224).tolist(), ndarr2.tolist())

        self.assertEqual((1, 24), sarr.reshape((1, 24)).shape)
        self.assertEqual((12, 2), sarr.reshape((12, 2)).shape)

    def test_SimpleArray_types(self):

        self.assertEqual(6, modmesh.SimpleArrayInt8((2,3)).nbytes)
        self.assertEqual(24, modmesh.SimpleArrayInt16((3,4)).nbytes)
        self.assertEqual(28, modmesh.SimpleArrayInt32(7).nbytes)
        self.assertEqual(2*3*4*8, modmesh.SimpleArrayInt64((2,3,4)).nbytes)

        self.assertEqual(6, modmesh.SimpleArrayUint8((2,3)).nbytes)
        self.assertEqual(24, modmesh.SimpleArrayUint16((3,4)).nbytes)
        self.assertEqual(28, modmesh.SimpleArrayUint32(7).nbytes)
        self.assertEqual(2*3*4*8, modmesh.SimpleArrayUint64((2,3,4)).nbytes)

        self.assertEqual(2*3*4*5*4,
                         modmesh.SimpleArrayFloat32((2,3,4,5)).nbytes)
        self.assertEqual(13*8,
                         modmesh.SimpleArrayFloat64(13).nbytes)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
