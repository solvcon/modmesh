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
            helper.test_load_arrayfloat64_from_arrayplex(arrayplex_float64), True)  # noqa: E501

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

        self.assertEqual(sarr.nbytes, 2*3*4*8)
        self.assertEqual(sarr.size, 2*3*4)
        self.assertEqual(sarr.itemsize, 8)
        self.assertEqual(sarr.stride, (12, 4, 1))
        self.assertEqual(len(sarr), 2*3*4)  # number of elements
        self.assertEqual(sarr.nbody, 2 - 0)
        self.assertEqual(sarr.has_ghost, False)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
