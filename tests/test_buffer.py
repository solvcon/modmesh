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
            buf[it] = 100 + it
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


class BufferExpanderBasicTC(unittest.TestCase):

    def test_BufferExpander(self):
        ep = modmesh.BufferExpander(10)
        self.assertEqual(10, ep.capacity)
        self.assertEqual(10, len(ep))

        ep = modmesh.BufferExpander()
        self.assertEqual(0, ep.capacity)
        self.assertEqual(0, len(ep))

        with self.assertRaisesRegex(
                IndexError,
                "BufferExpander: index 0 is out of bounds with size 0"
        ):
            ep[0]

        ep.reserve(10)
        self.assertEqual(10, ep.capacity)
        self.assertEqual(0, len(ep))  # size unchanged

        with self.assertRaisesRegex(
                IndexError,
                "BufferExpander: index 0 is out of bounds with size 0"
        ):
            ep[0]

        ep.expand(10)
        self.assertEqual(10, ep.capacity)
        self.assertEqual(10, len(ep))  # size changed

        ep[9]  # should not raise an exception
        with self.assertRaisesRegex(
                IndexError,
                "BufferExpander: index 10 is out of bounds with size 10"
        ):
            ep[10]

        # initialize
        for it in range(len(ep)):
            ep[it] = it

        self.assertFalse(ep.is_concrete)
        cbuf = ep.as_concrete()
        self.assertTrue(ep.is_concrete)
        self.assertEqual(10, len(cbuf))
        self.assertEqual(list(range(10)), list(cbuf))

        # prove cbuf and gbuf share memory
        for it in range(len(cbuf)):
            cbuf[it] = it + 10
        self.assertEqual(list(i + 10 for i in range(10)), list(ep))

    def test_BufferExpanderFromConcreteBuffer(self):
        buf = modmesh.ConcreteBuffer(10)
        for it in range(len(buf)):
            buf[it] = it

        ep = modmesh.BufferExpander(buf)
        self.assertEqual(10, ep.capacity)
        self.assertEqual(10, len(ep))
        for it in range(len(ep)):
            ep[it] = it + 100

        for it in range(len(buf)):
            self.assertEqual(buf[it] + 100, ep[it])


class SimpleArrayBasicTC(unittest.TestCase):

    def test_SimpleArray(self):

        sarr = modmesh.SimpleArrayFloat64((2, 3, 4))
        ndarr = np.array(sarr, copy=False)

        self.assertEqual(2 * 3 * 4 * 8, sarr.nbytes)
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
        self.assertEqual((2, 2, 2, 3), sarr.reshape((2, 2, 2, 3)).shape)

    def test_SimpleArray_clone(self):
        sarr = modmesh.SimpleArrayFloat64((2, 3, 4))
        sarr.fill(2.0)
        sarr_ref = sarr
        sarr_clone = sarr.clone()

        self.assertTrue(sarr_ref is sarr)
        np.testing.assert_equal(sarr_ref.ndarray[...], sarr.ndarray[...])

        self.assertFalse(sarr_clone is sarr)
        np.testing.assert_equal(sarr_clone.ndarray[...], sarr.ndarray[...])

        sarr[3] = 3.0
        self.assertEqual(sarr_ref[3], 3.0)
        self.assertEqual(sarr_clone[3], 2.0)  # should be the original value

    def test_SimpleArray_transpose(self):
        def check_equal(sarr, ndarr):
            self.assertEqual(sarr.shape, ndarr.shape)
            shape = sarr.shape
            for idx1 in range(shape[0]):
                for idx2 in range(shape[1]):
                    for idx3 in range(shape[2]):
                        for idx4 in range(shape[3]):
                            ndnum = ndarr[idx1, idx2, idx3, idx4]
                            sarrnum = sarr[idx1, idx2, idx3, idx4]
                            self.assertEqual(ndnum, sarrnum)

        ndarr = np.arange(2 * 3 * 4 * 5, dtype='float64')
        ndarr = ndarr.reshape((2, 3, 4, 5))
        ndarrT = ndarr.transpose()

        sarr = modmesh.SimpleArrayFloat64(array=ndarr)
        sarr2 = sarr.transpose()
        check_equal(sarr, ndarrT)
        check_equal(sarr2, ndarrT)
        self.assertEqual(memoryview(sarr), memoryview(sarr2))

        sarr = modmesh.SimpleArrayFloat64(array=ndarr)
        sarr2 = sarr.transpose(inplace=False)
        check_equal(sarr, ndarr)
        check_equal(sarr2, ndarrT)
        self.assertNotEqual(memoryview(sarr), memoryview(sarr2))

        sarr = modmesh.SimpleArrayFloat64(array=ndarr)
        sarr2 = sarr.T
        check_equal(sarr, ndarr)
        check_equal(sarr2, ndarrT)
        self.assertNotEqual(memoryview(sarr), memoryview(sarr2))

        ndarrT = ndarr.transpose(0, 3, 2, 1)

        sarr = modmesh.SimpleArrayFloat64(array=ndarr)
        sarr2 = sarr.transpose((0, 3, 2, 1))
        check_equal(sarr, ndarrT)
        check_equal(sarr2, ndarrT)
        self.assertEqual(memoryview(sarr), memoryview(sarr2))

        sarr = modmesh.SimpleArrayFloat64(array=ndarr)
        sarr2 = sarr.transpose((0, 3, 2, 1), inplace=False)
        check_equal(sarr, ndarr)
        check_equal(sarr2, ndarrT)
        self.assertNotEqual(memoryview(sarr), memoryview(sarr2))

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

        # Test out-of-bound ghost setting.
        with self.assertRaisesRegex(
                IndexError,
                r"SimpleArray: cannot set nghost 11 > shape\(0\) 10"
        ):
            out_of_bound_ghost_sarr = modmesh.SimpleArrayInt8(10)
            out_of_bound_ghost_sarr.nghost = 11

        # Test empty array ghost setting.
        with self.assertRaisesRegex(
                IndexError,
                r"SimpleArray: cannot set nghost 1 > 0 to an empty array"
        ):
            empty_sarr = modmesh.SimpleArrayInt8(())
            empty_sarr.nghost = 1

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
                r"SimpleArray: index 14 >= 14 \(buffer size: 24 - nghost: 10\)"
        ):
            sarr[14]

        # Test out-of-bound index for setitem.
        with self.assertRaisesRegex(
                IndexError, r"SimpleArray: index -11 < -nghost: -10"
        ):
            sarr[-11] = 1

        with self.assertRaisesRegex(
                IndexError,
                r"SimpleArray: index 14 >= 14 \(buffer size: 24 - nghost: 10\)"
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
                r"SimpleArray::validate_range\(\): cannot handle "
                r"3-dimensional \(more than 1\) array with non-zero nghost: 1"
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
        _check(modmesh.SimpleArrayInt64((2, 3, 4)), 2 * 3 * 4 * 8, 'int64')
        _check(modmesh.SimpleArrayUint64((2, 3, 4)), 2 * 3 * 4 * 8, 'uint64')

        # Real-number types.
        _check(modmesh.SimpleArrayFloat32((2, 3, 4, 5)), 2 * 3 * 4 * 5 * 4,
               'float32')
        _check(modmesh.SimpleArrayFloat64((2, 3, 4, 5)), 2 * 3 * 4 * 5 * 8,
               'float64')

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

        shape = (2, 3, 5, 7)
        np_sarr = np.empty(shape, dtype='float64')
        py_sarr = modmesh.SimpleArrayFloat64(array=np_sarr)
        self.assertTupleEqual(shape, py_sarr.shape)
        self.assertEqual(np_sarr.nbytes, py_sarr.nbytes)
        self.assertEqual(np_sarr.size, py_sarr.size)

        shape = (5, 5, 5, 5)
        np_sarr = np.empty(shape, dtype='float64')
        for i in range(5):
            for j in range(5):
                for k in range(5):
                    for x in range(5):
                        np_sarr[i, j, k, x] = i * 1000 + j * 100 + k * 10 + x

        py_sarr = modmesh.SimpleArrayFloat64(array=np_sarr)
        self.assertTupleEqual(shape, py_sarr.shape)
        self.assertEqual(np_sarr.nbytes, py_sarr.nbytes)
        self.assertEqual(np_sarr.size, py_sarr.size)

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

    def test_SimpleArray_from_ndarray_slice(self):
        ndarr = np.arange(1000, dtype='float64').reshape((10, 10, 10))
        parr = ndarr[1:7:3, 6:2:-1, 3:9]
        sarr = modmesh.SimpleArrayFloat64(array=ndarr[1:7:3, 6:2:-1, 3:9])

        for i in range(2):
            for j in range(4):
                for k in range(6):
                    self.assertEqual(parr[i, j, k], sarr[i, j, k])
        self.assertEqual(parr.nbytes, sarr.nbytes)
        self.assertEqual(parr.size, sarr.size)

    def test_SimpleArray_from_ndarray_transpose(self):
        ndarr = np.arange(350, dtype='float64').reshape((5, 7, 10))
        # The following array is F contiguous.
        parr = ndarr[2:4].T
        sarr = modmesh.SimpleArrayFloat64(array=ndarr[2:4].T)

        for i in range(10):
            for j in range(7):
                for k in range(2):
                    self.assertEqual(parr[i, j, k], sarr[i, j, k])

    def test_SimpleArray_broadcast_ellipsis_shape(self):
        sarr = modmesh.SimpleArrayFloat64((2, 3, 4))
        ndarr = np.arange(2 * 3 * 4, dtype='float64').reshape((2, 3, 4))
        sarr[...] = ndarr[...]
        v = 0
        for i in range(2):
            for j in range(3):
                for k in range(4):
                    self.assertEqual(v, sarr[i, j, k])
                    v += 1

        ndarr = np.arange(2 * 3, dtype='float64').reshape((2, 3))
        with self.assertRaisesRegex(
                RuntimeError,
                r"Broadcast input array from shape\(2, 3\) "
                r"into shape\(2, 3, 4\)"
        ):
            sarr[...] = ndarr[...]

        ndarr = np.arange(2 * 4 * 3, dtype='float64').reshape((2, 4, 3))
        with self.assertRaisesRegex(
                RuntimeError,
                r"Broadcast input array from shape\(2, 4, 3\) "
                r"into shape\(2, 3, 4\)"
        ):
            sarr[...] = ndarr[...]

    def test_SimpleArray_broadcast_ellipsis_ghost_1d(self):

        N = 13
        G = 3

        sarr = modmesh.SimpleArrayFloat64(N)
        ndarr = np.arange(N, dtype='float64')
        sarr.nghost = G

        sarr[...] = ndarr[...]

        v = 0
        for i in range(-G, N - G):
            self.assertEqual(v, sarr[i])
            v += 1

    def test_SimpleArray_broadcast_ellipsis_ghost_md(self):
        N = 5
        G = 2

        sarr = modmesh.SimpleArrayFloat64((5, 3, 4))
        ndarr = np.arange(5 * 3 * 4, dtype='float64').reshape((5, 3, 4))
        sarr.nghost = G

        sarr[...] = ndarr[...]

        v = 0
        for i in range(-G, N - G):
            for j in range(3):
                for k in range(4):
                    self.assertEqual(v, sarr[i, j, k])
                    v += 1

    def test_SimpleArray_broadcast_ellipsis_stride(self):

        sarr = modmesh.SimpleArrayFloat64((2, 3, 4))
        ndarr = np.arange(
            (4 * 2) * (3 * 3) * (2 * 4), dtype='float64').reshape(
            (4 * 2, 3 * 3, 2 * 4))

        stride_arr = ndarr[::4, ::3, ::2]

        # point to the same data
        self.assertEqual(ndarr.__array_interface__['data'],
                         stride_arr.__array_interface__['data'])

        sarr[...] = stride_arr[...]

        for i in range(2):
            for j in range(3):
                for k in range(4):
                    self.assertEqual(stride_arr[i, j, k], sarr[i, j, k])

    def test_SimpleArray_broadcast_ellipsis_dtype(self):
        sarr = modmesh.SimpleArrayFloat64((2, 3, 4))
        ndarr = np.arange(2 * 3 * 4, dtype='int32').reshape((2, 3, 4))
        sarr[...] = ndarr[...]
        v = 0
        for i in range(2):
            for j in range(3):
                for k in range(4):
                    self.assertEqual(v, sarr[i, j, k])
                    v += 1

        sarr = modmesh.SimpleArrayFloat64((2, 3, 4))
        ndarr = np.arange(2 * 3 * 4, dtype='float32').reshape((2, 3, 4))
        sarr[...] = ndarr[...]
        v = 0
        for i in range(2):
            for j in range(3):
                for k in range(4):
                    self.assertEqual(v, sarr[i, j, k])
                    v += 1

    def test_SimpleArray_broadcast_slice_basic(self):
        ndarr_input = np.arange(
            1 * 2 * 3 * 4 * 5, dtype='float64').reshape((1, 2, 3, 4, 5))
        ndarr_input += 999  # just a magic number

        def init(sarr):
            v = 0
            for x1 in range(sarr.shape[0]):
                for x2 in range(sarr.shape[1]):
                    for x3 in range(sarr.shape[2]):
                        for x4 in range(sarr.shape[3]):
                            for x5 in range(sarr.shape[4]):
                                sarr[x1, x2, x3, x4, x5] = v
                                v += 1

        def check(sarr, ndarr):
            for x1 in range(sarr.shape[0]):
                for x2 in range(sarr.shape[1]):
                    for x3 in range(sarr.shape[2]):
                        for x4 in range(sarr.shape[3]):
                            for x5 in range(sarr.shape[4]):
                                self.assertEqual(
                                    ndarr[x1, x2, x3, x4, x5],
                                    sarr[x1, x2, x3, x4, x5])

        sarr = modmesh.SimpleArrayFloat64((2, 2, 3, 4, 5))
        ndarr = np.arange(2 * 2 * 3 * 4 * 5, dtype='float64').reshape(
            (2, 2, 3, 4, 5))
        init(sarr)
        sarr[::2, ...] = ndarr_input[...]
        ndarr[::2, ...] = ndarr_input[...]
        check(sarr, ndarr)

        sarr = modmesh.SimpleArrayFloat64((2, 4, 3, 4, 5))
        ndarr = np.arange(2 * 4 * 3 * 4 * 5, dtype='float64').reshape(
            (2, 4, 3, 4, 5))
        init(sarr)
        sarr[::2, ::2, ...] = ndarr_input[...]
        ndarr[::2, ::2, ...] = ndarr_input[...]
        check(sarr, ndarr)

        sarr = modmesh.SimpleArrayFloat64((2, 2, 3, 8, 10))
        ndarr = np.arange(
            2 * 2 * 3 * 8 * 10, dtype='float64').reshape((2, 2, 3, 8, 10))
        init(sarr)
        sarr[::2, ..., ::2, ::2] = ndarr_input[...]
        ndarr[::2, ..., ::2, ::2] = ndarr_input[...]
        check(sarr, ndarr)

        sarr = modmesh.SimpleArrayFloat64((1, 2, 3, 8, 10))
        ndarr = np.arange(
            1 * 2 * 3 * 8 * 10, dtype='float64').reshape((1, 2, 3, 8, 10))
        init(sarr)
        sarr[..., ::2, ::2] = ndarr_input[...]
        ndarr[..., ::2, ::2] = ndarr_input[...]
        check(sarr, ndarr)

        sarr = modmesh.SimpleArrayFloat64((2, 4, 6, 8, 10))
        ndarr = np.arange(
            2 * 4 * 6 * 8 * 10, dtype='float64').reshape((2, 4, 6, 8, 10))
        init(sarr)
        sarr[::2, ::2, ::2, ::2, ::2] = ndarr_input[...]
        ndarr[::2, ::2, ::2, ::2, ::2] = ndarr_input[...]
        check(sarr, ndarr)

        sarr = modmesh.SimpleArrayFloat64((2, 6, 3, 4, 5))
        ndarr = np.arange(
            2 * 6 * 3 * 4 * 5, dtype='float64').reshape((2, 6, 3, 4, 5))
        init(sarr)
        sarr[::2, ::3, ::, :, ::1] = ndarr_input[...]
        ndarr[::2, ::3, ::, :, ::1] = ndarr_input[...]
        check(sarr, ndarr)

    def test_SimpleArray_broadcast_slice_shape(self):
        ndarr = np.arange(2 * 3 * 4, dtype='float64').reshape((2, 3, 4))

        sarr = modmesh.SimpleArrayFloat64((4, 6, 8))
        with self.assertRaisesRegex(
                RuntimeError,
                r"Broadcast input array from shape\(2, 3, 4\) "
                r"into shape\(2, 2, 2\)"
        ):
            sarr[::2, ::3, ::4] = ndarr[...]

        sarr = modmesh.SimpleArrayFloat64((4, 6, 8))
        with self.assertRaisesRegex(
                RuntimeError,
                r"Broadcast input array from shape\(2, 3, 4\) "
                r"into shape\(2, 6, 8\)"
        ):
            sarr[::2, ::1, ...] = ndarr[...]

    def test_SimpleArray_broadcast_slice_ghost_1d(self):
        import math
        N = 13
        G = 3
        STEP = 3

        sarr = modmesh.SimpleArrayFloat64(N)
        ndarr = np.arange(math.ceil(N / STEP), dtype='float64')
        ndarr2 = np.arange(N, dtype='float64')
        sarr.nghost = G

        sarr[::STEP] = ndarr[...]
        ndarr2[::STEP] = ndarr[...]

        print(ndarr2.shape)

        for i in range(0, N, STEP):
            self.assertEqual(ndarr2[i], sarr[i - G])

    def test_SimpleArray_broadcast_slice_ghost_md(self):
        import math
        N = 5
        G = 2
        STEP = 2

        sarr = modmesh.SimpleArrayFloat64((N, 3, 4))
        ndarr = np.arange(math.ceil(N / STEP) * 3 * 4, dtype='float64') \
            .reshape((math.ceil(N / STEP), 3, 4))
        ndarr2 = np.arange(N * 3 * 4, dtype='float64').reshape((N, 3, 4))
        sarr.nghost = G

        sarr[::STEP, ...] = ndarr[...]
        ndarr2[::STEP, ...] = ndarr[...]

        for i in range(0, N, STEP):
            for j in range(3):
                for k in range(4):
                    self.assertEqual(ndarr2[i, j, k], sarr[i - G, j, k])

    def test_SimpleArray_broadcast_from_list_list(self):
        sarr = modmesh.SimpleArrayFloat64((2, 3))
        sarr[:, :] = [[1, 2, 3], [4, 5, 6]]
        for i in range(2):
            for j in range(3):
                self.assertEqual(sarr[i, j], i * 3 + j + 1)

        sarr = modmesh.SimpleArrayFloat64((2, 3))
        sarr[:1, :2] = [[1, 2]]
        self.assertEqual(sarr[0, 0], 1)
        self.assertEqual(sarr[0, 1], 2)

    def test_SimpleArray_broadcast_from_tuple_list(self):
        sarr = modmesh.SimpleArrayFloat64((2, 3))
        sarr[:, :] = [(1, 2, 3), (4, 5, 6)]
        for i in range(2):
            for j in range(3):
                self.assertEqual(sarr[i, j], i * 3 + j + 1)

        sarr = modmesh.SimpleArrayFloat64((2, 3))
        sarr[:1, :2] = [(1, 2)]
        self.assertEqual(sarr[0, 0], 1)
        self.assertEqual(sarr[0, 1], 2)

    def test_SimpleArray_broadcast_from_tuple_tuple(self):
        sarr = modmesh.SimpleArrayFloat64((2, 3))
        sarr[:, :] = ((1, 2, 3), (4, 5, 6))
        for i in range(2):
            for j in range(3):
                self.assertEqual(sarr[i, j], i * 3 + j + 1)

        sarr = modmesh.SimpleArrayFloat64((2, 3))
        sarr[:1, :2] = ((1, 2),)
        self.assertEqual(sarr[0, 0], 1)
        self.assertEqual(sarr[0, 1], 2)

    @unittest.skipUnless(modmesh.testhelper.PYTEST_HELPER_BINDING_BUILT,
                         "TestSimpleArrayHelper is not built")
    def test_SimpleArray_casting(self):
        helper = modmesh.testhelper.TestSimpleArrayHelper
        # first check the caster works if the argument is exact the same type
        array_float64 = modmesh.SimpleArrayFloat64((2, 3, 4))
        self.assertEqual(
            helper.test_load_arrayfloat64_from_arrayplex(array_float64), True)

        # init arrayplex
        arrayplex_int32 = modmesh.SimpleArray((2, 3, 4), dtype="int32")
        arrayplex_uint64 = modmesh.SimpleArray((2, 3, 4), dtype="uint64")
        arrayplex_float64 = modmesh.SimpleArray((2, 3, 4), dtype="float64")

        # check the type is the same with different data types
        self.assertTrue(type(arrayplex_int32) is type(arrayplex_uint64))
        self.assertTrue(type(arrayplex_uint64) is type(arrayplex_float64))
        self.assertEqual(
            str(type(arrayplex_int32)), "<class '_modmesh.SimpleArray'>")
        self.assertEqual(
            str(type(arrayplex_uint64)), "<class '_modmesh.SimpleArray'>")
        self.assertEqual(
            str(type(arrayplex_float64)), "<class '_modmesh.SimpleArray'>")

        # check if arrayplex can cast to simplearray
        self.assertEqual(
            helper.test_load_arrayin32_from_arrayplex(arrayplex_int32), True)

        # int32 and uint64 are different types
        with self.assertRaisesRegex(
                TypeError,
                r"incompatible function arguments"):
            helper.test_load_arrayin32_from_arrayplex(arrayplex_uint64)

        # check if arrayplex can cast to simplearray
        self.assertEqual(
            helper.test_load_arrayfloat64_from_arrayplex(arrayplex_float64),
            True)  # noqa: E501

        # float64 and int32 are differet types
        with self.assertRaisesRegex(TypeError,
                                    r"incompatible function arguments"):
            helper.test_load_arrayfloat64_from_arrayplex(arrayplex_int32)

        # explicitly check the `cast` function of the customized caster works
        # SimpleArray32 from the constructor directly
        array_int32 = modmesh.SimpleArrayInt32((2, 3, 4))
        # SimpleArray32 from casting
        array_int32_2 = helper.test_cast_to_arrayint32()
        self.assertTrue(type(array_int32) is type(array_int32_2))
        self.assertEqual(
            str(type(array_int32)), "<class '_modmesh.SimpleArrayInt32'>")
        self.assertEqual(
            str(type(array_int32_2)), "<class '_modmesh.SimpleArrayInt32'>")

    def test_SimpleArray_SimpleArrayPlex_type_switch(self):
        arrayplex_int32 = modmesh.SimpleArray((2, 3, 4), dtype="int32")

        # from plex to typed
        array_int32 = arrayplex_int32.typed
        self.assertEqual(
            str(type(array_int32)), "<class '_modmesh.SimpleArrayInt32'>")

        # from typed to plex
        arrayplex_int32_2 = array_int32.plex
        self.assertEqual(
            str(type(arrayplex_int32_2)), "<class '_modmesh.SimpleArray'>")

    def test_sort(self):
        # Note: tests on array with ghost indices should be added
        #       in the future

        test_data = [
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
            [1, 5, 10, 2, 6, 9, 7, 8, 4, 3],
            [1,  0,  1, -3, -4, -1,  1,  9,  5, -4],
            [-1.3, -4.8,  1.5,  0.3,  7.1,  2.5,  4.8, -0.1,  9.4,  7.6]
        ]

        def _check(arr, use_float=False):
            if use_float:
                narr = np.array(arr, dtype='float64')
                sarr = modmesh.SimpleArrayFloat64(array=narr)
            else:
                narr = np.array(arr, dtype='int32')
                sarr = modmesh.SimpleArrayInt32(array=narr)

            args = sarr.argsort()
            for i in range(1, len(args)):
                self.assertLessEqual(sarr[args[i - 1]], sarr[args[i]])

            sorted_arr = sarr.take_along_axis(args)
            for i in range(1, len(sorted_arr)):
                self.assertLessEqual(sorted_arr[i - 1], sorted_arr[i])

            sarr.sort()
            for i in range(1, len(sarr)):
                self.assertLessEqual(sarr[i - 1], sarr[i])

        _check(test_data[0])
        _check(test_data[1])
        _check(test_data[2])
        _check(test_data[3])
        _check(test_data[4], True)

    def test_talk_along_axis(self):
        data = [1, 5, 10, 2, 6, 9, 7, 8, 4, 3]
        narr = np.array(data, dtype='int32')
        data_arr = modmesh.SimpleArrayInt32(array=narr)

        # test 1-D indices
        idx = [0, 2, 4, 6, 1, 3, 5, 7]
        narr = np.array(idx, dtype='uint64')
        idx_arr = modmesh.SimpleArrayUint64(array=narr)

        ret_arr = data_arr.take_along_axis(idx_arr)
        for i in range(len(idx)):
            self.assertEqual(ret_arr[i], data[idx[i]])

        ret_arr = data_arr.take_along_axis_simd(idx_arr)
        for i in range(len(idx)):
            self.assertEqual(ret_arr[i], data[idx[i]])

        # test 2-D indices
        idx = [[0, 2, 4, 6], [1, 3, 5, 7]]
        narr = np.array(idx, dtype='uint64')
        idx_arr = modmesh.SimpleArrayUint64(array=narr)

        ret_arr = data_arr.take_along_axis(idx_arr)
        for i in range(len(idx)):
            for j in range(len(idx[i])):
                self.assertEqual(ret_arr[i, j], data[idx_arr[i, j]])

        ret_arr = data_arr.take_along_axis_simd(idx_arr)
        for i in range(len(idx)):
            for j in range(len(idx[i])):
                self.assertEqual(ret_arr[i, j], data[idx_arr[i, j]])

        # test out-of-range index
        idx = [[0, 1], [2, 3], [4, 20]]
        narr = np.array(idx, dtype='uint64')
        idx_arr = modmesh.SimpleArrayUint64(array=narr)

        with self.assertRaisesRegex(
            IndexError,
            r"SimpleArray::take_along_axis\(\): indices\[2, 1\] is 20, " +
            "which is out of range of the array size 10"
        ):
            ret_arr = data_arr.take_along_axis(idx_arr)

        with self.assertRaisesRegex(
            IndexError,
            r"SimpleArray::take_along_axis_simd\(\): indices\[2, 1\] is 20, " +
            "which is out of range of the array size 10"
        ):
            ret_arr = data_arr.take_along_axis_simd(idx_arr)


class SimpleArrayCalculatorsTC(unittest.TestCase):

    def test_minmaxsum(self):
        sarr = modmesh.SimpleArrayFloat64(shape=(2, 4), value=10.0)

        self.assertEqual(sarr.sum(), 10.0 * 2 * 4)
        self.assertEqual(sarr.min(), 10.0)
        self.assertEqual(sarr.max(), 10.0)
        sarr.fill(1.0)
        self.assertEqual(sarr.sum(), 1.0 * 2 * 4)
        self.assertEqual(sarr.min(), 1.0)
        self.assertEqual(sarr.max(), 1.0)
        sarr[1, 0] = 9.2
        sarr[0, 3] = -2.3
        self.assertEqual(sarr.min(), -2.3)
        self.assertEqual(sarr.max(), 9.2)

    def test_abs(self):
        sarr = modmesh.SimpleArrayInt64(shape=(3, 2), value=-2)
        self.assertEqual(sarr.sum(), -2 * 3 * 2)
        sarr = sarr.abs()
        self.assertEqual(sarr.sum(), 2 * 3 * 2)

        # Taking absolute value of unsigned type simply copies the data.
        sarr = modmesh.SimpleArrayInt8(shape=(3, 2), value=2)
        self.assertEqual(sarr.sum(), 2 * 3 * 2)
        sarr = sarr.abs()
        self.assertEqual(sarr.sum(), 2 * 3 * 2)

        # Absolute value of Boolean is special.
        sarr = modmesh.SimpleArrayBool(shape=(3, 2), value=1)
        self.assertEqual(sarr.sum(), True)
        sarr = sarr.abs()
        self.assertEqual(sarr.sum(), True)

    def test_median(self):
        nparr = np.arange(24, dtype='float64')
        np.random.shuffle(nparr)
        sarr = modmesh.SimpleArrayFloat64(array=nparr)
        npmed = np.median(nparr)
        smed = sarr.median()
        self.assertEqual(npmed, smed)

        nparr = np.arange(24, dtype='float64').reshape((2, 3, 4))
        np.random.shuffle(nparr)
        sarr = modmesh.SimpleArrayFloat64(array=nparr)
        npmed = np.median(nparr)
        smed = sarr.median()
        self.assertEqual(npmed, smed)

        nparr = np.arange(81, dtype='float64').reshape((3, 3, 3, 3))
        np.random.shuffle(nparr)
        sarr = modmesh.SimpleArrayFloat64(array=nparr)
        npmed = np.median(nparr)
        smed = sarr.median()
        self.assertEqual(npmed, smed)

        nparr = np.arange(24, dtype='complex128').reshape((2, 3, 4))
        npimg = nparr.copy()
        np.random.shuffle(nparr)
        np.random.shuffle(npimg)
        nparr = nparr + 1j * npimg
        sarr = modmesh.SimpleArrayComplex128(array=nparr)
        npmed = np.median(nparr)
        smed = sarr.median()
        self.assertEqual(npmed.real, smed.real)
        self.assertEqual(npmed.imag, smed.imag)

        # Reference: https://github.com/numpy/numpy/issues/12943
        nparr = np.array([1+10j, 2+1j, 3+0j, 0+3j], dtype='complex128')
        sarr = modmesh.SimpleArrayComplex128(array=nparr)
        npmed = np.median(nparr)
        smed = sarr.median()
        self.assertEqual(npmed.real, smed.real)
        self.assertEqual(npmed.imag, smed.imag)

        nparr = np.arange(24, dtype='float64').reshape((2, 3, 4))
        nparr = nparr[::2, ::2, ::2]
        sarr = modmesh.SimpleArrayFloat64(array=nparr)
        npavg = np.median(nparr)
        savg = sarr.median()
        self.assertEqual(npavg, savg)

    def test_median_with_axis(self):
        nparr = np.arange(120, dtype='float64').reshape((4, 5, 6))
        sarr = modmesh.SimpleArrayFloat64(array=nparr)
        npmed = np.median(nparr, axis=0)
        smed = sarr.median(axis=0)
        smed = smed.ndarray
        self.assertTrue(np.array_equal(npmed, smed))

        nparr = np.arange(120, dtype='float64').reshape((4, 5, 6))
        sarr = modmesh.SimpleArrayFloat64(array=nparr)
        npmed = np.median(nparr, axis=1)
        smed = sarr.median(axis=1)
        smed = smed.ndarray
        self.assertTrue(np.array_equal(npmed, smed))

        nparr = np.arange(120, dtype='float64').reshape((4, 5, 6))
        sarr = modmesh.SimpleArrayFloat64(array=nparr)
        npmed = np.median(nparr, axis=2)
        smed = sarr.median(axis=2)
        smed = smed.ndarray
        self.assertTrue(np.array_equal(npmed, smed))

        nparr = np.arange(120, dtype='float64').reshape((4, 5, 6))
        sarr = modmesh.SimpleArrayFloat64(array=nparr)
        npmed = np.median(nparr, axis=(0, 1))
        smed = sarr.median(axis=[0, 1])
        smed = smed.ndarray
        self.assertTrue(np.array_equal(npmed, smed))

        nparr = np.arange(120, dtype='float64').reshape((4, 5, 6))
        sarr = modmesh.SimpleArrayFloat64(array=nparr)
        npmed = np.median(nparr, axis=(1, 2))
        smed = sarr.median(axis=[1, 2])
        smed = smed.ndarray
        self.assertTrue(np.array_equal(npmed, smed))

        nparr = np.arange(120, dtype='float64').reshape((4, 5, 6))
        sarr = modmesh.SimpleArrayFloat64(array=nparr)
        npmed = np.median(nparr, axis=(0, 2))
        smed = sarr.median(axis=[0, 2])
        smed = smed.ndarray
        self.assertTrue(np.array_equal(npmed, smed))

    def test_average(self):
        nparr = np.arange(24, dtype='float64')
        np.random.shuffle(nparr)
        sarr = modmesh.SimpleArrayFloat64(array=nparr)
        npavg = np.average(nparr)
        savg = sarr.average()
        self.assertEqual(npavg, savg)

        nparr = np.arange(24, dtype='float64').reshape((2, 3, 4))
        np.random.shuffle(nparr)
        sarr = modmesh.SimpleArrayFloat64(array=nparr)
        npavg = np.average(nparr)
        savg = sarr.average()
        self.assertEqual(npavg, savg)

        nparr = np.arange(81, dtype='float64').reshape((3, 3, 3, 3))
        np.random.shuffle(nparr)
        sarr = modmesh.SimpleArrayFloat64(array=nparr)
        npavg = np.average(nparr)
        savg = sarr.average()
        self.assertEqual(npavg, savg)

        nparr = np.arange(24, dtype='float64').reshape((2, 3, 4))
        nparr = nparr[::2, ::2, ::2]
        sarr = modmesh.SimpleArrayFloat64(array=nparr)
        npavg = np.average(nparr)
        savg = sarr.average()
        self.assertEqual(npavg, savg)

        nparr = np.array([1, 2, 3, 4, 5], dtype='float64')
        weights = np.array([0.1, 0.2, 0.3, 0.2, 0.2], dtype='float64')
        sarr = modmesh.SimpleArrayFloat64(array=nparr)
        sweights = modmesh.SimpleArrayFloat64(array=weights)
        npavg = np.average(nparr, weights=weights)
        savg = sarr.average(weight=sweights)
        self.assertEqual(npavg, savg)

        nparr = np.arange(6, dtype='float64').reshape((2, 3))
        weights = np.array([0.5, 0.3, 0.2], dtype='float64')
        sarr = modmesh.SimpleArrayFloat64(array=nparr)
        sweights = modmesh.SimpleArrayFloat64(array=weights)
        npavg = np.average(nparr, weights=weights, axis=1)
        savg = sarr.average(axis=1, weight=sweights)
        savg = savg.ndarray
        self.assertTrue(np.allclose(npavg, savg))

        nparr = np.arange(24, dtype='float64').reshape((2, 3, 4))
        weights = np.array([0.25, 0.25, 0.25, 0.25], dtype='float64')
        sarr = modmesh.SimpleArrayFloat64(array=nparr)
        sweights = modmesh.SimpleArrayFloat64(array=weights)
        npavg = np.average(nparr, weights=weights, axis=2)
        savg = sarr.average(axis=2, weight=sweights)
        savg = savg.ndarray
        self.assertTrue(np.allclose(npavg, savg))

        nparr = np.arange(12, dtype='float64').reshape((3, 4))
        weights = np.array([0.5, 0.3, 0.2], dtype='float64')
        sarr = modmesh.SimpleArrayFloat64(array=nparr)
        sweights = modmesh.SimpleArrayFloat64(array=weights)
        npavg = np.average(nparr, weights=weights, axis=0)
        savg = sarr.average(axis=0, weight=sweights)
        savg = savg.ndarray
        self.assertTrue(np.allclose(npavg, savg))

    def test_average_with_axis(self):
        nparr = np.arange(120, dtype='float64').reshape((4, 5, 6))
        sarr = modmesh.SimpleArrayFloat64(array=nparr)
        npmed = np.average(nparr, axis=0)
        smed = sarr.average(axis=0)
        smed = smed.ndarray
        self.assertTrue(np.array_equal(npmed, smed))

        nparr = np.arange(120, dtype='float64').reshape((4, 5, 6))
        sarr = modmesh.SimpleArrayFloat64(array=nparr)
        npmed = np.average(nparr, axis=1)
        smed = sarr.average(axis=1)
        smed = smed.ndarray
        self.assertTrue(np.array_equal(npmed, smed))

        nparr = np.arange(120, dtype='float64').reshape((4, 5, 6))
        sarr = modmesh.SimpleArrayFloat64(array=nparr)
        npmed = np.average(nparr, axis=2)
        smed = sarr.average(axis=2)
        smed = smed.ndarray
        self.assertTrue(np.array_equal(npmed, smed))

        nparr = np.arange(120, dtype='float64').reshape((4, 5, 6))
        sarr = modmesh.SimpleArrayFloat64(array=nparr)
        npmed = np.average(nparr, axis=(0, 1))
        smed = sarr.average(axis=[0, 1])
        smed = smed.ndarray
        self.assertTrue(np.array_equal(npmed, smed))

        nparr = np.arange(120, dtype='float64').reshape((4, 5, 6))
        sarr = modmesh.SimpleArrayFloat64(array=nparr)
        npmed = np.average(nparr, axis=(1, 2))
        smed = sarr.average(axis=[1, 2])
        smed = smed.ndarray
        self.assertTrue(np.array_equal(npmed, smed))

        nparr = np.arange(120, dtype='float64').reshape((4, 5, 6))
        sarr = modmesh.SimpleArrayFloat64(array=nparr)
        npmed = np.average(nparr, axis=(0, 2))
        smed = sarr.average(axis=[0, 2])
        smed = smed.ndarray
        self.assertTrue(np.array_equal(npmed, smed))

        nparr = np.arange(120, dtype='float64').reshape((4, 5, 6))
        weights = np.array([0.1, 0.2, 0.3, 0.2, 0.2], dtype='float64')
        sarr = modmesh.SimpleArrayFloat64(array=nparr)
        sweights = modmesh.SimpleArrayFloat64(array=weights)
        npmed = np.average(nparr, weights=weights, axis=1)
        smed = sarr.average(axis=1, weight=sweights)
        smed = smed.ndarray
        self.assertTrue(np.allclose(npmed, smed))

        nparr = np.arange(120, dtype='float64').reshape((4, 5, 6))
        weights = np.array([
            [0.2, 0.25, 0.25, 0.25, 0.25, 0.25],
            [0.25, 0.2, 0.25, 0.25, 0.25, 0.25],
            [0.25, 0.25, 0.2, 0.25, 0.25, 0.25],
            [0.25, 0.25, 0.25, 0.2, 0.25, 0.25]
        ], dtype='float64')
        sarr = modmesh.SimpleArrayFloat64(array=nparr)
        sweights = modmesh.SimpleArrayFloat64(array=weights)
        npmed = np.average(nparr, weights=weights, axis=(0, 2))
        smed = sarr.average(axis=[0, 2], weight=sweights)
        smed = smed.ndarray
        self.assertTrue(np.allclose(npmed, smed))

        nparr = np.arange(60, dtype='float64').reshape((3, 4, 5))
        weights_axis0 = np.array([0.4, 0.3, 0.3], dtype='float64')
        weights_axis2 = np.array([0.2, 0.2, 0.2, 0.2, 0.2], dtype='float64')
        sarr = modmesh.SimpleArrayFloat64(array=nparr)
        sweights_axis0 = modmesh.SimpleArrayFloat64(array=weights_axis0)
        sweights_axis2 = modmesh.SimpleArrayFloat64(array=weights_axis2)

        npmed = np.average(nparr, weights=weights_axis0, axis=0)
        smed = sarr.average(axis=0, weight=sweights_axis0)
        smed = smed.ndarray
        self.assertTrue(np.allclose(npmed, smed))

        npmed = np.average(nparr, weights=weights_axis2, axis=2)
        smed = sarr.average(axis=2, weight=sweights_axis2)
        smed = smed.ndarray
        self.assertTrue(np.allclose(npmed, smed))

        nparr = np.array([[1, 2], [3, 4]], dtype='float64')
        weights = np.array([0.6, 0.4], dtype='float64')
        sarr = modmesh.SimpleArrayFloat64(array=nparr)
        sweights = modmesh.SimpleArrayFloat64(array=weights)
        npmed = np.average(nparr, weights=weights, axis=1)
        smed = sarr.average(axis=1, weight=sweights)
        smed = smed.ndarray
        self.assertTrue(np.allclose(npmed, smed))

    def test_mean(self):
        nparr = np.arange(24, dtype='float64')
        np.random.shuffle(nparr)
        sarr = modmesh.SimpleArrayFloat64(array=nparr)
        npavg = np.mean(nparr)
        savg = sarr.mean()
        self.assertEqual(npavg, savg)

        nparr = np.arange(24, dtype='float64').reshape((2, 3, 4))
        np.random.shuffle(nparr)
        sarr = modmesh.SimpleArrayFloat64(array=nparr)
        npmean = np.mean(nparr)
        smean = sarr.mean()
        self.assertEqual(npmean, smean)

        nparr = np.arange(81, dtype='float64').reshape((3, 3, 3, 3))
        np.random.shuffle(nparr)
        sarr = modmesh.SimpleArrayFloat64(array=nparr)
        npmean = np.mean(nparr)
        smean = sarr.mean()
        self.assertEqual(npmean, smean)

        nparr = np.arange(24, dtype='float64').reshape((2, 3, 4))
        nparr = nparr[::2, ::2, ::2]
        sarr = modmesh.SimpleArrayFloat64(array=nparr)
        npmean = np.mean(nparr)
        smean = sarr.mean()
        self.assertEqual(npmean, smean)

    def test_mean_with_axis(self):
        nparr = np.arange(120, dtype='float64').reshape((4, 5, 6))
        sarr = modmesh.SimpleArrayFloat64(array=nparr)
        npmed = np.mean(nparr, axis=0)
        smed = sarr.mean(axis=0)
        smed = smed.ndarray
        self.assertTrue(np.array_equal(npmed, smed))

        nparr = np.arange(120, dtype='float64').reshape((4, 5, 6))
        sarr = modmesh.SimpleArrayFloat64(array=nparr)
        npmed = np.mean(nparr, axis=1)
        smed = sarr.mean(axis=1)
        smed = smed.ndarray
        self.assertTrue(np.array_equal(npmed, smed))

        nparr = np.arange(120, dtype='float64').reshape((4, 5, 6))
        sarr = modmesh.SimpleArrayFloat64(array=nparr)
        npmed = np.mean(nparr, axis=2)
        smed = sarr.mean(axis=2)
        smed = smed.ndarray
        self.assertTrue(np.array_equal(npmed, smed))

        nparr = np.arange(120, dtype='float64').reshape((4, 5, 6))
        sarr = modmesh.SimpleArrayFloat64(array=nparr)
        npmed = np.mean(nparr, axis=(0, 1))
        smed = sarr.mean(axis=[0, 1])
        smed = smed.ndarray
        self.assertTrue(np.array_equal(npmed, smed))

        nparr = np.arange(120, dtype='float64').reshape((4, 5, 6))
        sarr = modmesh.SimpleArrayFloat64(array=nparr)
        npmed = np.mean(nparr, axis=(1, 2))
        smed = sarr.mean(axis=[1, 2])
        smed = smed.ndarray
        self.assertTrue(np.array_equal(npmed, smed))

        nparr = np.arange(120, dtype='float64').reshape((4, 5, 6))
        sarr = modmesh.SimpleArrayFloat64(array=nparr)
        npmed = np.mean(nparr, axis=(0, 2))
        smed = sarr.mean(axis=[0, 2])
        smed = smed.ndarray
        self.assertTrue(np.array_equal(npmed, smed))

    def test_var(self):
        nparr = np.arange(24, dtype='float64')
        np.random.shuffle(nparr)
        sarr = modmesh.SimpleArrayFloat64(array=nparr)
        npavg = np.var(nparr)
        savg = sarr.var()
        self.assertEqual(npavg, savg)

        nparr = np.arange(24, dtype='float64').reshape((2, 3, 4))
        np.random.shuffle(nparr)
        sarr = modmesh.SimpleArrayFloat64(array=nparr)
        npvar = np.var(nparr)
        svar = sarr.var()
        self.assertEqual(npvar, svar)

        nparr = np.arange(24, dtype='complex128').reshape((2, 3, 4))
        npimg = nparr.copy()
        np.random.shuffle(nparr)
        np.random.shuffle(npimg)
        nparr = npimg + 1j * npimg
        sarr = modmesh.SimpleArrayComplex128(array=nparr)
        npvar = np.var(nparr)
        svar = sarr.var()
        self.assertEqual(npvar, svar)

        nparr = np.arange(24, dtype='float64').reshape((2, 3, 4))
        nparr = nparr[::2, ::2, ::2]
        sarr = modmesh.SimpleArrayFloat64(array=nparr)
        npvar = np.var(nparr)
        svar = sarr.var()
        self.assertEqual(npvar, svar)

    def test_var_with_axis(self):
        nparr = np.arange(120, dtype='float64').reshape((4, 5, 6))
        sarr = modmesh.SimpleArrayFloat64(array=nparr)
        npmed = np.var(nparr, axis=0)
        smed = sarr.var(axis=0)
        smed = smed.ndarray
        self.assertTrue(np.array_equal(npmed, smed))

        nparr = np.arange(120, dtype='float64').reshape((4, 5, 6))
        sarr = modmesh.SimpleArrayFloat64(array=nparr)
        npmed = np.var(nparr, axis=1)
        smed = sarr.var(axis=1)
        smed = smed.ndarray
        self.assertTrue(np.array_equal(npmed, smed))

        nparr = np.arange(120, dtype='float64').reshape((4, 5, 6))
        sarr = modmesh.SimpleArrayFloat64(array=nparr)
        npmed = np.var(nparr, axis=2)
        smed = sarr.var(axis=2)
        smed = smed.ndarray
        self.assertTrue(np.array_equal(npmed, smed))

        nparr = np.arange(120, dtype='float64').reshape((4, 5, 6))
        sarr = modmesh.SimpleArrayFloat64(array=nparr)
        npmed = np.var(nparr, axis=(0, 1))
        smed = sarr.var(axis=[0, 1])
        smed = smed.ndarray
        self.assertTrue(np.array_equal(npmed, smed))

        nparr = np.arange(120, dtype='float64').reshape((4, 5, 6))
        sarr = modmesh.SimpleArrayFloat64(array=nparr)
        npmed = np.var(nparr, axis=(1, 2))
        smed = sarr.var(axis=[1, 2])
        smed = smed.ndarray
        self.assertTrue(np.array_equal(npmed, smed))

        nparr = np.arange(120, dtype='float64').reshape((4, 5, 6))
        sarr = modmesh.SimpleArrayFloat64(array=nparr)
        npmed = np.var(nparr, axis=(0, 2))
        smed = sarr.var(axis=[0, 2])
        smed = smed.ndarray
        self.assertTrue(np.array_equal(npmed, smed))

    def test_std(self):
        nparr = np.arange(24, dtype='float64')
        np.random.shuffle(nparr)
        sarr = modmesh.SimpleArrayFloat64(array=nparr)
        npavg = np.std(nparr)
        savg = sarr.std()
        self.assertEqual(npavg, savg)

        nparr = np.arange(24, dtype='float64').reshape((2, 3, 4))
        np.random.shuffle(nparr)
        sarr = modmesh.SimpleArrayFloat64(array=nparr)
        npstd = np.std(nparr)
        sstd = sarr.std()
        self.assertEqual(npstd, sstd)

        nparr = np.arange(24, dtype='complex128').reshape((2, 3, 4))
        npimg = nparr.copy()
        np.random.shuffle(nparr)
        np.random.shuffle(npimg)
        nparr = npimg + 1j * npimg
        sarr = modmesh.SimpleArrayComplex128(array=nparr)
        npstd = np.std(nparr)
        sstd = sarr.std()
        self.assertEqual(npstd, sstd)

        nparr = np.arange(24, dtype='float64').reshape((2, 3, 4))
        nparr = nparr[::2, ::2, ::2]
        sarr = modmesh.SimpleArrayFloat64(array=nparr)
        npstd = np.std(nparr)
        sstd = sarr.std()
        self.assertEqual(npstd, sstd)

    def test_std_with_axis(self):
        nparr = np.arange(120, dtype='float64').reshape((4, 5, 6))
        sarr = modmesh.SimpleArrayFloat64(array=nparr)
        npmed = np.std(nparr, axis=0)
        smed = sarr.std(axis=0)
        smed = smed.ndarray
        self.assertTrue(np.array_equal(npmed, smed))

        nparr = np.arange(120, dtype='float64').reshape((4, 5, 6))
        sarr = modmesh.SimpleArrayFloat64(array=nparr)
        npmed = np.std(nparr, axis=1)
        smed = sarr.std(axis=1)
        smed = smed.ndarray
        self.assertTrue(np.array_equal(npmed, smed))

        nparr = np.arange(120, dtype='float64').reshape((4, 5, 6))
        sarr = modmesh.SimpleArrayFloat64(array=nparr)
        npmed = np.std(nparr, axis=2)
        smed = sarr.std(axis=2)
        smed = smed.ndarray
        self.assertTrue(np.array_equal(npmed, smed))

        nparr = np.arange(120, dtype='float64').reshape((4, 5, 6))
        sarr = modmesh.SimpleArrayFloat64(array=nparr)
        npmed = np.std(nparr, axis=(0, 1))
        smed = sarr.std(axis=[0, 1])
        smed = smed.ndarray
        self.assertTrue(np.array_equal(npmed, smed))

        nparr = np.arange(120, dtype='float64').reshape((4, 5, 6))
        sarr = modmesh.SimpleArrayFloat64(array=nparr)
        npmed = np.std(nparr, axis=(1, 2))
        smed = sarr.std(axis=[1, 2])
        smed = smed.ndarray
        self.assertTrue(np.array_equal(npmed, smed))

        nparr = np.arange(120, dtype='float64').reshape((4, 5, 6))
        sarr = modmesh.SimpleArrayFloat64(array=nparr)
        npmed = np.std(nparr, axis=(0, 2))
        smed = sarr.std(axis=[0, 2])
        smed = smed.ndarray
        self.assertTrue(np.array_equal(npmed, smed))

    def type_convertor(self, dtype):
        return {
            'int8': modmesh.SimpleArrayInt8,
            'int16': modmesh.SimpleArrayInt16,
            'int32': modmesh.SimpleArrayInt32,
            'int64': modmesh.SimpleArrayInt64,
            'uint8': modmesh.SimpleArrayUint8,
            'uint16': modmesh.SimpleArrayUint16,
            'uint32': modmesh.SimpleArrayUint32,
            'uint64': modmesh.SimpleArrayUint64,
            'float32': modmesh.SimpleArrayFloat32,
            'float64': modmesh.SimpleArrayFloat64,
        }[dtype]

    def test_add(self):
        # test integer
        def test_add_type(type):
            arr1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            arr2 = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
            res = [3, 6, 9, 12, 15, 18, 21, 24, 27, 30]
            narr1 = np.array(arr1, dtype=type)
            narr2 = np.array(arr2, dtype=type)
            sarr1 = self.type_convertor(type)(array=narr1)
            sarr2 = self.type_convertor(type)(array=narr2)
            nres = np.add(narr1, narr2)
            sres = sarr1.add(sarr2)
            simdres = sarr1.add_simd(sarr2)
            sarr1.iadd(sarr2)
            for i in range(len(res)):
                self.assertEqual(sres[i], res[i])
                self.assertEqual(sarr1[i], res[i])
                self.assertEqual(simdres[i], res[i])
                self.assertEqual(sres[i], nres[i])

        test_add_type('int8')
        test_add_type('int16')
        test_add_type('int32')
        test_add_type('int64')
        test_add_type('uint8')
        test_add_type('uint16')
        test_add_type('uint32')
        test_add_type('uint64')
        test_add_type('float32')
        test_add_type('float64')

        # test boolean
        arr1 = [True, True, True, False, False, False]
        arr2 = [True, False, True, False, True, False]
        res = [True, True, True, False, True, False]
        narr1 = np.array(arr1, dtype='bool')
        narr2 = np.array(arr2, dtype='bool')
        sarr1 = modmesh.SimpleArrayBool(array=narr1)
        sarr2 = modmesh.SimpleArrayBool(array=narr2)
        nres = np.add(narr1, narr2)
        sres = sarr1.add(sarr2)
        sarr1.iadd(sarr2)
        for i in range(len(res)):
            self.assertEqual(sres[i], res[i])
            self.assertEqual(sarr1[i], res[i])
            self.assertEqual(sres[i], nres[i])

    def test_sub(self):
        # test integer
        def test_sub_type(type):
            arr1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            arr2 = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
            narr1 = np.array(arr1, dtype=type)
            narr2 = np.array(arr2, dtype=type)
            sarr1 = self.type_convertor(type)(array=narr1)
            sarr2 = self.type_convertor(type)(array=narr2)
            nres = np.subtract(narr2, narr1)
            sres = sarr2.sub(sarr1)
            simdres = sarr2.sub_simd(sarr1)
            sarr2.isub(sarr1)
            for i in range(len(arr1)):
                self.assertEqual(sres[i], arr1[i])
                self.assertEqual(sarr2[i], arr1[i])
                self.assertEqual(simdres[i], arr1[i])
                self.assertEqual(sres[i], nres[i])

        test_sub_type('int8')
        test_sub_type('int16')
        test_sub_type('int32')
        test_sub_type('int64')
        test_sub_type('uint8')
        test_sub_type('uint16')
        test_sub_type('uint32')
        test_sub_type('uint64')
        test_sub_type('float32')
        test_sub_type('float64')

        # test boolean
        arr1 = [True, True, True, False, False, False]
        arr2 = [True, False, True, False, True, False]
        narr1 = np.array(arr1, dtype='bool')
        narr2 = np.array(arr2, dtype='bool')
        sarr1 = modmesh.SimpleArrayBool(array=narr1)
        sarr2 = modmesh.SimpleArrayBool(array=narr2)
        with self.assertRaisesRegex(
            RuntimeError,
            r"SimpleArray<bool>::isub\(\): "
            r"boolean value doesn't support this operation"
        ):
            sarr2.sub(sarr1)

        with self.assertRaisesRegex(
            RuntimeError,
            r"SimpleArray<bool>::isub\(\): "
            r"boolean value doesn't support this operation"
        ):
            sarr2.isub(sarr1)

    def test_mul(self):
        def test_mul_type(type):
            arr1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            arr2 = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
            res = [2, 8, 18, 32, 50, 72, 98, 128, 162, 200]
            narr1 = np.array(arr1, dtype=type)
            narr2 = np.array(arr2, dtype=type)
            sarr1 = self.type_convertor(type)(array=narr1)
            sarr2 = self.type_convertor(type)(array=narr2)
            nres = np.multiply(narr2, narr1)
            sres = sarr1.mul(sarr2)
            simdres = sarr1.mul_simd(sarr2)
            sarr1.imul(sarr2)
            for i in range(len(res)):
                self.assertEqual(sres[i], res[i])
                self.assertEqual(sarr1[i], res[i])
                self.assertEqual(simdres[i], res[i])
                self.assertEqual(sres[i], nres[i])

        test_mul_type('int16')
        test_mul_type('int32')
        test_mul_type('int64')
        test_mul_type('uint8')
        test_mul_type('uint16')
        test_mul_type('uint32')
        test_mul_type('uint64')
        test_mul_type('float32')
        test_mul_type('float64')

        # test boolean
        arr1 = [True, True, True, False, False, False]
        arr2 = [True, False, True, False, True, False]
        res = [True, False, True, False, False, False]
        narr1 = np.array(arr1, dtype='bool')
        narr2 = np.array(arr2, dtype='bool')
        sarr1 = modmesh.SimpleArrayBool(array=narr1)
        sarr2 = modmesh.SimpleArrayBool(array=narr2)
        nres = np.multiply(narr2, narr1)
        sres = sarr1.mul(sarr2)
        sarr1.imul(sarr2)
        for i in range(len(res)):
            self.assertEqual(sres[i], res[i])
            self.assertEqual(sarr1[i], res[i])
            self.assertEqual(sres[i], nres[i])

    def test_div(self):
        arr1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        arr2 = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

        def test_div_type(type):
            res = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
            narr1 = np.array(arr1, dtype=type)
            narr2 = np.array(arr2, dtype=type)
            sarr1 = self.type_convertor(type)(array=narr1)
            sarr2 = self.type_convertor(type)(array=narr2)
            nres = np.divide(narr2, narr1)
            sres = sarr2.div(sarr1)
            simdres = sarr2.div_simd(sarr1)
            sarr2.idiv(sarr1)
            for i in range(len(res)):
                self.assertEqual(sres[i], res[i])
                self.assertEqual(sarr2[i], res[i])
                self.assertEqual(simdres[i], res[i])
                self.assertEqual(sres[i], nres[i])

        test_div_type('int8')
        test_div_type('int16')
        test_div_type('int32')
        test_div_type('int64')
        test_div_type('uint8')
        test_div_type('uint16')
        test_div_type('uint32')
        test_div_type('uint64')

        # test float
        res = [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]
        narr1 = np.array(arr1, dtype='float64') / 5
        narr2 = np.array(arr2, dtype='float64') / 2
        sarr1 = modmesh.SimpleArrayFloat64(array=narr1)
        sarr2 = modmesh.SimpleArrayFloat64(array=narr2)
        nres = np.divide(narr2, narr1)
        sres = sarr2.div(sarr1)
        simdres = sarr2.div_simd(sarr1)
        sarr2.idiv(sarr1)
        for i in range(len(res)):
            self.assertEqual(sres[i], res[i])
            self.assertEqual(sarr2[i], res[i])
            self.assertEqual(simdres[i], res[i])
            self.assertEqual(sres[i], nres[i])

        # test boolean
        arr1 = [True, True, True, False, False, False]
        arr2 = [True, False, True, False, True, False]
        res = [True, True, True, False, True, False]
        narr1 = np.array(arr1, dtype='bool')
        narr2 = np.array(arr2, dtype='bool')
        sarr1 = modmesh.SimpleArrayBool(array=narr1)
        sarr2 = modmesh.SimpleArrayBool(array=narr2)
        with self.assertRaisesRegex(
            RuntimeError,
            r"SimpleArray<bool>::idiv\(\): "
            r"boolean value doesn't support this operation"
        ):
            sarr2.div(sarr1)

        with self.assertRaisesRegex(
            RuntimeError,
            r"SimpleArray<bool>::idiv\(\): "
            r"boolean value doesn't support this operation"
        ):
            sarr2.idiv(sarr1)


class SimpleArraySearchTC(unittest.TestCase):

    def test_argminmax(self):
        # test 1-D data
        data = [1, 3, 5, 7, 9]
        narr = np.array(data, dtype='uint64')
        sarr = modmesh.SimpleArrayUint64(array=narr)

        self.assertEqual(sarr.argmin(), 0)
        self.assertEqual(sarr.argmax(), 4)
        self.assertEqual(narr.argmin(), sarr.argmin())
        self.assertEqual(narr.argmax(), sarr.argmax())

        # test N-D data
        data = [[1, 3, 5, 7, 9], [2, 4, 6, 8, 10], [1, 10, 1, 10, 1]]
        narr = np.array(data, dtype='float64')
        sarr = modmesh.SimpleArrayFloat64(array=narr)
        self.assertEqual(sarr.argmin(), 0)
        self.assertEqual(sarr.argmax(), 9)
        self.assertEqual(narr.argmin(), sarr.argmin())
        self.assertEqual(narr.argmax(), sarr.argmax())


class SimpleArrayPlexTC(unittest.TestCase):

    def test_SimpleArrayPlex_constructor(self):
        # 1. shape constructor
        dtype_list = [
            "bool", "int8", "uint8", "int16", "uint16", "int32", "uint32",
            "int64", "uint64", "float32", "float64"
        ]
        for dtype in dtype_list:
            modmesh.SimpleArray((2, 3, 4), dtype=dtype)

        # 2. shape and value constructor
        modmesh.SimpleArray((2, 3, 4), dtype="bool", value=True)
        with self.assertRaisesRegex(
                TypeError,
                r"Data type mismatch, expected Python bool"
        ):
            modmesh.SimpleArray((2, 3, 4), dtype="bool", value=3.3)

        dtype_list_int = [
            "int8", "uint8", "int16", "uint16", "int32", "uint32",
            "int64", "uint64"
        ]
        for dtype in dtype_list_int:
            modmesh.SimpleArray((2, 3, 4), dtype=dtype, value=3)
            with self.assertRaisesRegex(
                    TypeError,
                    r"Data type mismatch, expected Python int"
            ):
                modmesh.SimpleArray((2, 3, 4), dtype=dtype, value=3.3)

        dtype_list_float = ["float32", "float64"]
        for dtype in dtype_list_float:
            modmesh.SimpleArray((2, 3, 4), dtype=dtype, value=3.0)
            with self.assertRaisesRegex(
                    TypeError,
                    r"Data type mismatch, expected Python float"
            ):
                modmesh.SimpleArray((2, 3, 4), dtype=dtype, value=3)

        # 3. np.ndarray constructor
        # exclude bool, since it cannot use np.arange
        dtype_list_no_bool = [
            "int8", "uint8", "int16", "uint16", "int32", "uint32",
            "int64", "uint64", "float32", "float64"
        ]
        for dtype in dtype_list_no_bool:
            ndarr = np.arange(2 * 3 * 4, dtype=dtype)
            modmesh.SimpleArray(ndarr)
        boolean_array = np.array([True, False, True], dtype='bool')
        modmesh.SimpleArray(boolean_array)

    def test_SimpleArray_clone(self):
        sarr = modmesh.SimpleArray((2, 3, 4), value=2.0, dtype='float64')
        sarr_ref = sarr
        sarr_clone = sarr.clone()

        self.assertTrue(sarr_ref is sarr)
        ref_ndarr = sarr_ref.typed.ndarray[...]
        ndarr = sarr.typed.ndarray[...]
        np.testing.assert_equal(ref_ndarr, ndarr)

        self.assertFalse(sarr_clone is sarr)
        clone_ndarr = sarr_clone.typed.ndarray[...]
        ndarr = sarr.typed.ndarray[...]
        np.testing.assert_equal(clone_ndarr, ndarr)

        sarr[3] = 3.0
        self.assertEqual(sarr_ref[3], 3.0)
        self.assertEqual(sarr_clone[3], 2.0)  # should be the original value

    def test_SimpleArrayPlex_buffer(self):
        magic_number = 3.1415
        sarr = modmesh.SimpleArray(
            (2, 3, 4), value=magic_number, dtype='float64')
        ndarr = np.array(sarr, copy=False)
        self.assertEqual((2, 3, 4), ndarr.shape)
        self.assertEqual((ndarr == magic_number).all(), True)

    def test_SimpleArrayPlex_get_item(self):
        magic_number = 3.1415
        sarr = modmesh.SimpleArray(
            (2, 3, 4), value=magic_number, dtype='float64')
        self.assertEqual(sarr[1, 2, 3], magic_number)
        self.assertEqual(sarr[2], magic_number)

    def test_SimpleArrayPlex_properties(self):
        magic_number = 3.1415
        shape = (2, 3, 4)
        sarr = modmesh.SimpleArray(shape, value=magic_number, dtype='float64')

        self.assertEqual(sarr.nbytes, 2 * 3 * 4 * 8)
        self.assertEqual(sarr.size, 2 * 3 * 4)
        self.assertEqual(sarr.itemsize, 8)
        self.assertEqual(sarr.stride, (12, 4, 1))
        self.assertEqual(len(sarr), 2 * 3 * 4)  # number of elements
        self.assertEqual(sarr.nbody, 2 - 0)
        self.assertEqual(sarr.has_ghost, False)
        self.assertEqual(sarr.nghost, 0)

    def test_minmaxsum(self):
        sarr = modmesh.SimpleArray((2, 4), value=10.0, dtype='float64')

        self.assertEqual(sarr.sum(), 10.0 * 2 * 4)
        self.assertEqual(sarr.min(), 10.0)
        self.assertEqual(sarr.max(), 10.0)
        sarr.fill(1.0)
        self.assertEqual(sarr.sum(), 1.0 * 2 * 4)
        self.assertEqual(sarr.min(), 1.0)
        self.assertEqual(sarr.max(), 1.0)
        sarr[1, 0] = 9.2
        sarr[0, 3] = -2.3
        self.assertEqual(sarr.min(), -2.3)
        self.assertEqual(sarr.max(), 9.2)

    def test_abs(self):
        sarr = modmesh.SimpleArray((3, 2), value=-2, dtype='int64')
        self.assertEqual(sarr.sum(), -2 * 3 * 2)
        sarr = sarr.abs()
        self.assertEqual(sarr.sum(), 2 * 3 * 2)

    def test_SimpleArrayPlex_set_item_simple(self):
        # just test if the SimpleArrayPlex works
        # more detailed tests are in SimpleArrayBasicTC
        magic_number = 1214

        sarr = modmesh.SimpleArray(20, dtype="uint32")
        sarr[7] = magic_number
        self.assertEqual(sarr[7], magic_number)

        magic_number = 12141618
        sarr = modmesh.SimpleArray((2, 3, 4), dtype="uint64")
        sarr[1, 2, 3] = magic_number
        self.assertEqual(sarr[1, 2, 3], magic_number)

    def test_SimpleArrayPlex_set_item_ellipse(self):
        # just test if the SimpleArrayPlex works
        # more detailed tests are in SimpleArrayBasicTC
        sarr = modmesh.SimpleArray((2, 3, 4), dtype="float64")
        ndarr = np.arange(
            (4 * 2) * (3 * 3) * (2 * 4), dtype='float64').reshape(
            (4 * 2, 3 * 3, 2 * 4))

        stride_arr = ndarr[::4, ::3, ::2]

        # point to the same data
        self.assertEqual(ndarr.__array_interface__['data'],
                         stride_arr.__array_interface__['data'])

        sarr[...] = stride_arr[...]

        for i in range(2):
            for j in range(3):
                for k in range(4):
                    self.assertEqual(stride_arr[i, j, k], sarr[i, j, k])


class SimpleCollectorTC(unittest.TestCase):

    def test_construct(self):
        ct = modmesh.SimpleCollectorFloat64(10)
        self.assertEqual(10, ct.capacity)
        self.assertEqual(10, len(ct))

        ct = modmesh.SimpleCollectorFloat64()
        self.assertEqual(0, ct.capacity)
        self.assertEqual(0, len(ct))

        with self.assertRaisesRegex(
                IndexError,
                "SimpleCollector: index 0 is out of bounds with size 0"
        ):
            ct[0]

        ct.reserve(6)
        self.assertEqual(6, ct.capacity)
        self.assertEqual(0, len(ct))  # size unchanged

        with self.assertRaisesRegex(
                IndexError,
                "SimpleCollector: index 0 is out of bounds with size 0"
        ):
            ct[0]

        ct.expand(6)
        self.assertEqual(6, ct.capacity)
        self.assertEqual(6, len(ct))  # size changed

        ct[5]  # should not raise an exception
        with self.assertRaisesRegex(
                IndexError,
                "SimpleCollector: index 6 is out of bounds with size 6"
        ):
            ct[6]

        # initialize
        for it in range(6):
            ct[it] = it

        arr = ct.as_array()
        self.assertEqual(6, len(arr))
        self.assertEqual(list(range(6)), list(arr))

        # prove ct and arr share memory
        for it in range(6):
            ct[it] = it + 10
        self.assertEqual(list(it + 10 for it in range(6)), list(ct))

    def test_push_back(self):
        # Starting from 0.
        ct = modmesh.SimpleCollectorFloat64()
        self.assertEqual(0, ct.capacity)
        self.assertEqual(0, len(ct))

        ct.push_back(3.14159)
        self.assertEqual(1, ct.capacity)
        self.assertEqual(1, len(ct))
        self.assertEqual(ct[0], 3.14159)

        ct.push_back(3.14159 * 2)
        self.assertEqual(2, ct.capacity)
        self.assertEqual(2, len(ct))
        self.assertEqual(ct[1], 3.14159 * 2)

        ct.push_back(3.14159 * 3)
        self.assertEqual(4, ct.capacity)
        self.assertEqual(3, len(ct))
        self.assertEqual(ct[2], 3.14159 * 3)

        for it in range(10):
            ct.push_back(3.14159 * (3 + it + 1))
            self.assertEqual(3 + it + 1, len(ct))
            self.assertEqual(ct[3 + it], 3.14159 * (3 + it + 1))

        # Starting from non-zero and not power of 2.
        ct = modmesh.SimpleCollectorFloat64(10)
        self.assertEqual(10, ct.capacity)
        self.assertEqual(10, len(ct))

        ct.push_back(3.14159 * 4)
        self.assertEqual(20, ct.capacity)  # double capacity but not power of 2
        self.assertEqual(11, len(ct))
        self.assertEqual(ct[10], 3.14159 * 4)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
