# Copyright (c) 2025, Chun-Hsu Lai <as2266317@gmail.com>
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

import modmesh as mm


class ComplexTB(mm.testing.TestBase):

    def test_construct_default(self):
        cplx = self.mm_complex()
        self.assert_allclose(cplx.real, 0.0)
        self.assert_allclose(cplx.imag, 0.0)

    def test_construct_random(self):
        cplx = self.mm_complex(self.real1, self.imag1)
        self.assert_allclose(cplx.real, self.real1)
        self.assert_allclose(cplx.imag, self.imag1)

    def test_operator_add(self):
        cplx = self.mm_complex(self.real1, self.imag1)
        realv = self.np_float(2.0)

        result = cplx + realv

        self.assert_allclose(result.real, self.real1 + realv)
        self.assert_allclose(result.imag, self.imag1)

        result = realv + cplx

        self.assert_allclose(result.real, realv + cplx.real)
        self.assert_allclose(result.imag, cplx.imag)

        cplx1 = self.mm_complex(self.real1, self.imag1)
        cplx2 = self.mm_complex(self.real2, self.imag2)

        result = cplx1 + cplx2

        expected_real = self.real1 + self.real2
        expected_imag = self.imag1 + self.imag2

        self.assert_allclose(result.real, expected_real)
        self.assert_allclose(result.imag, expected_imag)

    def test_operator_sub(self):
        cplx = self.mm_complex(self.real1, self.imag1)
        realv = self.np_float(2.0)

        result = cplx - realv

        self.assert_allclose(result.real, self.real1 - realv)
        self.assert_allclose(result.imag, self.imag1)

        result = realv - cplx

        self.assert_allclose(result.real, realv - cplx.real)
        self.assert_allclose(result.imag, -cplx.imag)

        cplx1 = self.mm_complex(self.real1, self.imag1)
        cplx2 = self.mm_complex(self.real2, self.imag2)

        result = cplx1 - cplx2

        expected_real = self.real1 - self.real2
        expected_imag = self.imag1 - self.imag2

        self.assert_allclose(result.real, expected_real)
        self.assert_allclose(result.imag, expected_imag)

    def test_operator_mul(self):
        cplx = self.mm_complex(self.real1, self.imag1)
        realv = self.np_float(2.0)

        result = cplx * realv

        self.assert_allclose(result.real, self.real1 * realv)
        self.assert_allclose(result.imag, self.imag1 * realv)

        result = realv * cplx
        golden = self.mm_complex(realv, 0.0) * cplx

        self.assert_allclose(result.real, golden.real)
        self.assert_allclose(result.imag, golden.imag)

        cplx1 = self.mm_complex(self.real1, self.imag1)
        cplx2 = self.mm_complex(self.real2, self.imag2)

        result = cplx1 * cplx2

        expected_real = (self.real1 * self.real2
                         - self.imag1 * self.imag2)
        expected_imag = (self.real1 * self.imag2
                         + self.imag1 * self.real2)

        self.assert_allclose(result.real, expected_real)
        self.assert_allclose(result.imag, expected_imag)

    def test_operator_div(self):
        cplx = self.mm_complex(self.real1, self.imag1)
        realv = self.np_float(2.0)

        result = cplx / realv

        self.assert_allclose(result.real, self.real1 / realv)
        self.assert_allclose(result.imag, self.imag1 / realv)

        cplx1 = self.mm_complex(self.real1, self.imag1)
        cplx2 = self.mm_complex(self.real2, self.imag2)

        result = cplx1 / cplx2

        denominator = (self.real2 * self.real2 + self.imag2 *
                       self.imag2)
        expected_real = (self.real1 * self.real2 + self.imag1 *
                         self.imag2) / denominator
        expected_imag = (self.imag1 * self.real2 - self.real1 *
                         self.imag2) / denominator

        self.assert_allclose(result.real, expected_real)
        self.assert_allclose(result.imag, expected_imag)

    def test_operator_comparison(self):
        cplx1 = self.mm_complex(self.real1, self.imag1)
        cplx2 = self.mm_complex(self.real2, self.imag2)

        norm1 = cplx1.norm()
        norm2 = cplx2.norm()

        self.assertEqual(cplx1 < cplx2, norm1 < norm2)
        self.assertEqual(cplx1 > cplx2, norm1 > norm2)

    def test_norm(self):
        cplx = self.mm_complex(self.real1, self.imag1)

        result = cplx.norm()

        expected_val = self.real1 ** 2 + self.imag1 ** 2

        self.assert_allclose(result, expected_val)

    def test_complex_array(self):
        cplx = self.mm_complex(self.real1, self.imag1)
        sarr = self.mm_simplearraycomplex(10)
        sarr.fill(cplx)
        ndarr = np.array(sarr, copy=False, dtype=self.dtype)

        for i in range(10):
            self.assertEqual(ndarr[i].real, self.real1)
            self.assertEqual(ndarr[i].imag, self.imag1)

        self.assertEqual(ndarr.dtype, self.dtype)

        sarr = self.mm_simplearraycomplex(array=ndarr)

        for i in range(10):
            self.assertEqual(sarr[i].real, self.real1)
            self.assertEqual(sarr[i].imag, self.imag1)

        self.assertEqual(sarr.ndarray.dtype, ndarr.dtype)
        self.assertEqual(10 * self.esize, sarr.nbytes)

    def test_complex_conj(self):
        cplx = self.mm_complex(self.real1, self.imag1)
        cplx_conj = self.mm_complex(self.real1, -self.imag1)

        self.assertEqual(cplx.conj().real, cplx_conj.real)
        self.assertEqual(cplx.conj().imag, cplx_conj.imag)


class ComplexFp32TC(ComplexTB, unittest.TestCase):

    def assert_allclose(self, *args, **kw):
        if 'rtol' not in kw:
            kw['rtol'] = 1.e-7
        return super().assert_allclose(*args, **kw)

    def mm_complex(self, real=None, imag=None):
        if real is not None and imag is not None:
            return mm.complex64(real, imag)
        else:
            return mm.complex64()

    def mm_simplearraycomplex(self, size=None, array=None):
        if size is not None:
            return mm.SimpleArrayComplex64(size)
        if array is not None:
            return mm.SimpleArrayComplex64(array=array)
        return mm.SimpleArrayComplex64(0)

    def np_float(self, val):
        return np.float32(val)

    def setUp(self):
        self.real1 = np.float32(0.7)
        self.imag1 = np.float32(1.6)
        self.real2 = np.float32(2.5)
        self.imag2 = np.float32(3.4)
        self.realv = np.float32(2.0)
        self.dtype = mm.complex64.dtype()
        self.esize = 4 * 2

    def test_dtype_verification(self):
        expected_dtype = np.dtype('complex64')
        self.assertEqual(self.dtype, expected_dtype)


class ComplexFp64TC(ComplexTB, unittest.TestCase):

    def assert_allclose(self, *args, **kw):
        if 'rtol' not in kw:
            kw['rtol'] = 1.e-15
        return super().assert_allclose(*args, **kw)

    def mm_complex(self, real=None, imag=None):
        if real is not None and imag is not None:
            return mm.complex128(real, imag)
        else:
            return mm.complex128()

    def mm_simplearraycomplex(self, size=None, array=None):
        if size is not None:
            return mm.SimpleArrayComplex128(size)
        if array is not None:
            return mm.SimpleArrayComplex128(array=array)
        return mm.SimpleArrayComplex128(0)

    def np_float(self, val):
        return np.float64(val)

    def setUp(self):
        self.real1 = np.float64(4.3)
        self.imag1 = np.float64(5.2)
        self.real2 = np.float64(6.1)
        self.imag2 = np.float64(7.0)
        self.realv = np.float64(2.0)
        self.dtype = mm.complex128.dtype()
        self.esize = 8 * 2

    def test_dtype_verification(self):
        expected_dtype = np.dtype('complex128')
        self.assertEqual(self.dtype, expected_dtype)
