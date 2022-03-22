# Copyright (c) 2020, Yung-Yu Chen <yyc@solvcon.net>
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# - Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
# - Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# - Neither the name of the copyright holder nor the names of its contributors
#   may be used to endorse or promote products derived from this software
#   without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.


import unittest

import numpy as np

import modmesh


class ConcreteBufferBasicTC(unittest.TestCase):

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

    def test_ConcreteBuffer_from_ndarray(self):

        buf = modmesh.ConcreteBuffer(24)
        self.assertFalse(buf.is_from_python)

        ndarr = np.arange(24, dtype='float64').reshape((2, 3, 4))

        buf = modmesh.ConcreteBuffer(array=ndarr)
        self.assertEqual(ndarr.nbytes, buf.nbytes)
        self.assertTrue(buf.is_from_python)

        # The data buffer is shared.
        self.assertFalse((ndarr == 0).all())
        buf.ndarray.fill(0)
        self.assertTrue((ndarr == 0).all())


class SimpleArrayBasicTC(unittest.TestCase):

    def test_SimpleArray(self):

        sarr = modmesh.SimpleArrayFloat64((2, 3, 4))
        ndarr = np.array(sarr, copy=False)

        self.assertEqual(2*3*4*8, sarr.nbytes)
        self.assertEqual((2, 3, 4), sarr.shape)
        self.assertEqual((12, 4, 1), sarr.stride)  # number of skip elements

        np.ravel(ndarr)[:] = np.arange(24)  # initialize contents

        # Flat indexing interface.
        self.assertEqual(24, len(sarr))  # number of elements
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
        self.assertEqual(np.arange(100, 124).tolist(),
                         [sarr2[i] for i in range(24)])
        self.assertEqual(np.arange(100, 124).tolist(), list(sarr2))

        ndarr2 = sarr2.ndarray
        for i in range(24):
            sarr2[i] = 200 + i
        self.assertEqual(np.arange(200, 224).tolist(), ndarr2.tolist())

        self.assertEqual((1, 24), sarr.reshape((1, 24)).shape)
        self.assertEqual((12, 2), sarr.reshape((12, 2)).shape)

    def test_SimpleArray_invalid_ghost(self):
        sarr = modmesh.SimpleArrayInt8(10)
        with self.assertRaisesRegex(
            IndexError, r"SimpleArray: cannot set nghost 11 > shape\(0\) 10"
        ):
            sarr.nghost = 11

        empty_sarr = modmesh.SimpleArrayInt8(())
        with self.assertRaisesRegex(
            IndexError, r"SimpleArray: cannot set nghost 1 > 0 to an empty array"
        ):
            empty_sarr.nghost = 1

    def test_SimpleArray_ghost_1d(self):

        sarr = modmesh.SimpleArrayFloat64(4 * 3 * 2)
        ndarr = np.array(sarr, copy=False)
        ndarr[:] = np.arange(24)  # initialize contents

        self.assertFalse(sarr.has_ghost)
        self.assertEqual(0, sarr.nghost)
        self.assertEqual(24, sarr.nbody)

        v = 0
        for i in range(24):
            self.assertEqual(v, sarr[i])
            v += 1

        sarr.nghost = 10

        self.assertTrue(sarr.has_ghost)
        self.assertEqual(10, sarr.nghost)
        self.assertEqual(14, sarr.nbody)

        v = 0
        for i in range(-10, 14):
            self.assertEqual(v, sarr[i])
            v += 1

        # Test out-of-bound index for getitem.
        with self.assertRaisesRegex(
            IndexError, r"SimpleArray: index -11 < -nghost: -10"
        ):
            sarr[-11]

        with self.assertRaisesRegex(
            IndexError,
            r"SimpleArray: index 14 >= 14 \(size: 24 - nghost: 10\)"
        ):
            sarr[14]

        # Test out-of-bound index for setitem.
        with self.assertRaisesRegex(
            IndexError, r"SimpleArray: index -11 < -nghost: -10"
        ):
            sarr[-11] = 1

        with self.assertRaisesRegex(
            IndexError,
            r"SimpleArray: index 14 >= 14 \(size: 24 - nghost: 10\)"
        ):
            sarr[14] = 1

    def test_SimpleArray_ghost_md(self):

        sarr = modmesh.SimpleArrayFloat64((4, 3, 2))
        ndarr = np.array(sarr, copy=False)
        np.ravel(ndarr)[:] = np.arange(24)  # initialize contents

        self.assertFalse(sarr.has_ghost)
        self.assertEqual(0, sarr.nghost)
        self.assertEqual(4, sarr.nbody)

        v = 0
        for i in range(4):
            for j in range(3):
                for k in range(2):
                    self.assertEqual(v, sarr[i, j, k])
                    v += 1

        sarr.nghost = 1

        self.assertTrue(sarr.has_ghost)
        self.assertEqual(1, sarr.nghost)
        self.assertEqual(3, sarr.nbody)

        v = 0
        for i in range(-1, 3):
            for j in range(3):
                for k in range(2):
                    self.assertEqual(v, sarr[i, j, k])
                    v += 1

        with self.assertRaisesRegex(
            IndexError,
            r"SimpleArray::validate_range\(\): cannot handle 3-dimensional "
            r"\(more than 1\) array with non-zero nghost: 1"
        ):
            sarr[-1]

        # Test out-of-bound index for getitem.
        with self.assertRaisesRegex(
            IndexError,
            r"SimpleArray::validate_shape\(\): empty index"
        ):
            invalid_empty_idx = ()
            sarr[invalid_empty_idx]

        with self.assertRaisesRegex(
            IndexError,
            r"SimpleArray: dim 0 in \[-2, 0, 0\] < -nghost: -1"
        ):
            sarr[-2, 0, 0]

        with self.assertRaisesRegex(
            IndexError,
            r"SimpleArray: dim 0 in \[3, 0, 0\] >= nbody: 3 "
            r"\(shape\[0\]: 4 - nghost: 1\)"
        ):
            sarr[3, 0, 0]

        with self.assertRaisesRegex(
            IndexError,
            r"SimpleArray: dim 1 in \[0, -1, 0\] < 0"
        ):
            sarr[0, -1, 0]

        with self.assertRaisesRegex(
            IndexError,
            r"SimpleArray: dim 2 in \[0, 2, 2\] >= shape\[2\]: 2"
        ):
            sarr[0, 2, 2]

        # Test out-of-bound index for setitem.
        with self.assertRaisesRegex(
            IndexError,
            r"SimpleArray::validate_shape\(\): empty index"
        ):
            invalid_empty_idx = ()
            sarr[invalid_empty_idx] = 1

        with self.assertRaisesRegex(
            IndexError,
            r"SimpleArray: dim 0 in \[-2, 0, 0\] < -nghost: -1"
        ):
            sarr[-2, 0, 0] = 1

        with self.assertRaisesRegex(
            IndexError,
            r"SimpleArray: dim 0 in \[3, 0, 0\] >= nbody: 3 "
            r"\(shape\[0\]: 4 - nghost: 1\)"
        ):
            sarr[3, 0, 0] = 1

        with self.assertRaisesRegex(
            IndexError,
            r"SimpleArray: dim 1 in \[0, -1, 0\] < 0"
        ):
            sarr[0, -1, 0] = 1

        with self.assertRaisesRegex(
            IndexError,
            r"SimpleArray: dim 2 in \[0, 2, 2\] >= shape\[2\]: 2"
        ):
            sarr[0, 2, 2] = 1

    def test_SimpleArray_types(self):

        def _check(sarr, nbytes, dtype, get=True):
            self.assertEqual(nbytes, sarr.nbytes)
            self.assertEqual(dtype, sarr.ndarray.dtype)
            if get:
                dtype = getattr(np, dtype)
                self.assertEqual(dtype, sarr.ndarray.dtype)

        # Boolean.
        _check(modmesh.SimpleArrayBool((2, 3, 4)), 24, 'bool', get=False)
        _check(modmesh.SimpleArrayBool((2, 3, 4)), 24, 'bool_')

        # Integer types.
        _check(modmesh.SimpleArrayInt8((2, 3)), 6, 'int8')
        _check(modmesh.SimpleArrayUint8((2, 3)), 6, 'uint8')
        _check(modmesh.SimpleArrayInt16((3, 5)), 30, 'int16')
        _check(modmesh.SimpleArrayUint16((3, 5)), 30, 'uint16')
        _check(modmesh.SimpleArrayInt32(7), 28, 'int32')
        _check(modmesh.SimpleArrayUint32(7), 28, 'uint32')
        _check(modmesh.SimpleArrayInt64((2, 3, 4)), 2*3*4*8, 'int64')
        _check(modmesh.SimpleArrayUint64((2, 3, 4)), 2*3*4*8, 'uint64')

        # Real-number types.
        _check(modmesh.SimpleArrayFloat32((2, 3, 4, 5)), 2*3*4*5*4, 'float32')
        _check(modmesh.SimpleArrayFloat64((2, 3, 4, 5)), 2*3*4*5*8, 'float64')

    def test_SimpleArray_from_ndarray(self):

        ndarr = np.arange(24, dtype='float64').reshape((2, 3, 4))

        with self.assertRaisesRegex(RuntimeError, r"dtype mismatch"):
            modmesh.SimpleArrayInt8(array=ndarr)
        with self.assertRaisesRegex(RuntimeError, r"dtype mismatch"):
            modmesh.SimpleArrayUint64(array=ndarr)
        with self.assertRaisesRegex(RuntimeError, r"dtype mismatch"):
            modmesh.SimpleArrayFloat32(array=ndarr)

        sarr_from_py = modmesh.SimpleArrayFloat64(array=ndarr)
        self.assertTrue(sarr_from_py.is_from_python)

        sarr_from_cpp = modmesh.SimpleArrayFloat64(shape=(2, 3, 4))
        self.assertFalse(sarr_from_cpp.is_from_python)

    def test_SimpleArray_from_ndarray_content(self):

        ndarr = np.arange(24, dtype='float64').reshape((2, 3, 4))
        sarr = modmesh.SimpleArrayFloat64(array=ndarr)
        # Populate using ndarray interface.
        sarr.ndarray.fill(1)
        self.assertTrue((ndarr == 1).all())
        # Set value using setitem interface.
        sarr[0, 0, 0] = 10
        self.assertFalse((ndarr == 1).all())
        # Repopulate using ndarray interface.
        sarr.ndarray.fill(100)
        self.assertTrue((ndarr == 100).all())

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
