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


class ComplexTC(unittest.TestCase, mm.testing.TestBase):

    def assert_allclose32(self, *args, **kw):
        if 'rtol' not in kw:
            kw['rtol'] = 1.e-7
        return super().assert_allclose(*args, **kw)

    def assert_allclose64(self, *args, **kw):
        if 'rtol' not in kw:
            kw['rtol'] = 1.e-15
        return super().assert_allclose(*args, **kw)

    def setUp(self):
        self.real1_32 = np.float32(0.7)
        self.imag1_32 = np.float32(1.6)
        self.real2_32 = np.float32(2.5)
        self.imag2_32 = np.float32(3.4)

        self.real1_64 = np.float64(4.3)
        self.imag1_64 = np.float64(5.2)
        self.real2_64 = np.float64(6.1)
        self.imag2_64 = np.float64(7.0)

    def test_construct_float32_default(self):
        cplx = mm.complex64()
        self.assert_allclose32(cplx.real, 0.0)
        self.assert_allclose32(cplx.imag, 0.0)

    def test_construct_float64_default(self):
        cplx = mm.complex128()
        self.assert_allclose64(cplx.real, 0.0)
        self.assert_allclose64(cplx.imag, 0.0)

    def test_construct_float32_random(self):
        cplx = mm.complex64(self.real1_32, self.imag1_32)
        self.assert_allclose32(cplx.real, self.real1_32)
        self.assert_allclose32(cplx.imag, self.imag1_32)

    def test_construct_float64_random(self):
        cplx = mm.complex128(self.real1_64, self.imag1_64)
        self.assert_allclose64(cplx.real, self.real1_64)
        self.assert_allclose64(cplx.imag, self.imag1_64)

    def test_operator_add_float32(self):
        cplx1 = mm.complex64(self.real1_32, self.imag1_32)
        cplx2 = mm.complex64(self.real2_32, self.imag2_32)

        result = cplx1 + cplx2

        expected_real = self.real1_32 + self.real2_32
        expected_imag = self.imag1_32 + self.imag2_32

        self.assert_allclose32(result.real, expected_real)
        self.assert_allclose32(result.imag, expected_imag)

    def test_operator_add_float64(self):
        cplx1 = mm.complex128(self.real1_64, self.imag1_64)
        cplx2 = mm.complex128(self.real2_64, self.imag2_64)

        result = cplx1 + cplx2

        expected_real = self.real1_64 + self.real2_64
        expected_imag = self.imag1_64 + self.imag2_64

        self.assert_allclose64(result.real, expected_real)
        self.assert_allclose64(result.imag, expected_imag)

    def test_operator_sub_float32(self):
        cplx1 = mm.complex64(self.real1_32, self.imag1_32)
        cplx2 = mm.complex64(self.real2_32, self.imag2_32)

        result = cplx1 - cplx2

        expected_real = self.real1_32 - self.real2_32
        expected_imag = self.imag1_32 - self.imag2_32

        self.assert_allclose32(result.real, expected_real)
        self.assert_allclose32(result.imag, expected_imag)

    def test_operator_sub_float64(self):
        cplx1 = mm.complex128(self.real1_64, self.imag1_64)
        cplx2 = mm.complex128(self.real2_64, self.imag2_64)

        result = cplx1 - cplx2

        expected_real = self.real1_64 - self.real2_64
        expected_imag = self.imag1_64 - self.imag2_64

        self.assert_allclose64(result.real, expected_real)
        self.assert_allclose64(result.imag, expected_imag)

    def test_operator_mul_float32(self):
        cplx1 = mm.complex64(self.real1_32, self.imag1_32)
        cplx2 = mm.complex64(self.real2_32, self.imag2_32)

        result = cplx1 * cplx2

        expected_real = (self.real1_32 * self.real2_32
                         - self.imag1_32 * self.imag2_32)
        expected_imag = (self.real1_32 * self.imag2_32
                         + self.imag1_32 * self.real2_32)

        self.assert_allclose32(result.real, expected_real)
        self.assert_allclose32(result.imag, expected_imag)

    def test_operator_mul_float64(self):
        cplx1 = mm.complex128(self.real1_64, self.imag1_64)
        cplx2 = mm.complex128(self.real2_64, self.imag2_64)

        result = cplx1 * cplx2

        expected_real = (self.real1_64 * self.real2_64
                         - self.imag1_64 * self.imag2_64)
        expected_imag = (self.real1_64 * self.imag2_64
                         + self.imag1_64 * self.real2_64)

        self.assert_allclose64(result.real, expected_real)
        self.assert_allclose64(result.imag, expected_imag)

    def test_operator_div_float32_scalar(self):
        cplx = mm.complex64(self.real1_32, self.imag1_32)
        # Avoid division by zero by explicitly assigning the divisor
        divisor = np.float32(2.0)

        result = cplx / divisor

        expected_real = self.real1_32 / divisor
        expected_imag = self.imag1_32 / divisor

        self.assert_allclose32(result.real, expected_real)
        self.assert_allclose32(result.imag, expected_imag)

    def test_operator_div_float64_scalar(self):
        cplx = mm.complex128(self.real1_64, self.imag1_64)
        divisor = np.float64(2.0)

        result = cplx / divisor

        expected_real = self.real1_64 / divisor
        expected_imag = self.imag1_64 / divisor

        self.assert_allclose64(result.real, expected_real)
        self.assert_allclose64(result.imag, expected_imag)

    def test_operator_div_float32(self):
        cplx1 = mm.complex64(self.real1_32, self.imag1_32)
        cplx2 = mm.complex64(self.real2_32, self.imag2_32)

        result = cplx1 / cplx2

        denominator = (self.real2_32 * self.real2_32 + self.imag2_32 *
                       self.imag2_32)
        expected_real = (self.real1_32 * self.real2_32 +
                         self.imag1_32 * self.imag2_32) / denominator
        expected_imag = (self.imag1_32 * self.real2_32 -
                         self.real1_32 * self.imag2_32) / denominator

        self.assert_allclose32(result.real, expected_real)
        self.assert_allclose32(result.imag, expected_imag)

    def test_operator_div_float64(self):
        cplx1 = mm.complex128(self.real1_64, self.imag1_64)
        cplx2 = mm.complex128(self.real2_64, self.imag2_64)

        result = cplx1 / cplx2

        denominator = (self.real2_64 * self.real2_64 + self.imag2_64 *
                       self.imag2_64)
        expected_real = (self.real1_64 * self.real2_64 + self.imag1_64 *
                         self.imag2_64) / denominator
        expected_imag = (self.imag1_64 * self.real2_64 - self.real1_64 *
                         self.imag2_64) / denominator

        self.assert_allclose64(result.real, expected_real)
        self.assert_allclose64(result.imag, expected_imag)

    def test_operator_comparison_float32(self):
        cplx1 = mm.complex64(self.real1_32, self.imag1_32)
        cplx2 = mm.complex64(self.real2_32, self.imag2_32)

        norm1 = cplx1.norm()
        norm2 = cplx2.norm()

        self.assertEqual(cplx1 < cplx2, norm1 < norm2)
        self.assertEqual(cplx1 > cplx2, norm1 > norm2)

    def test_operator_comparison_float64(self):
        cplx1 = mm.complex128(self.real1_64, self.imag1_64)
        cplx2 = mm.complex128(self.real2_64, self.imag2_64)

        norm1 = cplx1.norm()
        norm2 = cplx2.norm()

        self.assertEqual(cplx1 < cplx2, norm1 < norm2)
        self.assertEqual(cplx1 > cplx2, norm1 > norm2)

    def test_norm_float32(self):
        cplx = mm.complex64(self.real1_32, self.imag1_32)

        result = cplx.norm()

        expected_val = self.real1_32 ** 2 + self.imag1_32 ** 2

        self.assert_allclose32(result, expected_val)

    def test_norm_float64(self):
        cplx = mm.complex128(self.real1_64, self.imag1_64)

        result = cplx.norm()

        expected_val = self.real1_64 ** 2 + self.imag1_64 ** 2

        self.assert_allclose64(result, expected_val)

    def test_dtype_verification_float32(self):
        dtype = mm.complex64.dtype()
        expected_dtype = np.dtype('complex64')

        self.assertEqual(dtype, expected_dtype)

    def test_dtype_verification_float64(self):
        dtype = mm.complex128.dtype()
        expected_dtype = np.dtype('complex128')

        self.assertEqual(dtype, expected_dtype)

    def test_complex_array_float32(self):
        cplx = mm.complex64(self.real1_32, self.imag1_32)
        sarr = mm.SimpleArrayComplex64(10)
        sarr.fill(cplx)
        ndarr = np.array(sarr, copy=False, dtype=mm.complex64.dtype())

        for i in range(10):
            self.assertEqual(ndarr[i].real, self.real1_32)
            self.assertEqual(ndarr[i].imag, self.imag1_32)

        self.assertEqual(ndarr.dtype, mm.complex64.dtype())

        sarr = mm.SimpleArrayComplex64(array=ndarr)

        for i in range(10):
            self.assertEqual(sarr[i].real, self.real1_32)
            self.assertEqual(sarr[i].imag, self.imag1_32)

        self.assertEqual(sarr.ndarray.dtype, ndarr.dtype)
        self.assertEqual(10 * 4 * 2, sarr.nbytes)

    def test_complex_array_float64(self):
        cplx = mm.complex128(self.real1_64, self.imag1_64)
        sarr = mm.SimpleArrayComplex128(10)
        sarr.fill(cplx)
        ndarr = np.array(sarr, copy=False, dtype=mm.complex128.dtype())

        for i in range(10):
            self.assertEqual(ndarr[i].real, self.real1_64)
            self.assertEqual(ndarr[i].imag, self.imag1_64)

        self.assertEqual(ndarr.dtype, mm.complex128.dtype())

        sarr = mm.SimpleArrayComplex128(array=ndarr)

        for i in range(10):
            self.assertEqual(sarr[i].real, self.real1_64)
            self.assertEqual(sarr[i].imag, self.imag1_64)

        self.assertEqual(sarr.ndarray.dtype, ndarr.dtype)
        self.assertEqual(10 * 8 * 2, sarr.nbytes)

    def test_complex_conj_float32(self):
        cplx = mm.complex64(self.real1_32, self.imag1_32)
        cplx_conj = mm.complex64(self.real1_32, -self.imag1_32)

        self.assertEqual(cplx.conj().real, cplx_conj.real)
        self.assertEqual(cplx.conj().imag, cplx_conj.imag)

    def test_complex_conj_float64(self):
        cplx = mm.complex128(self.real1_64, self.imag1_64)
        cplx_conj = mm.complex128(self.real1_64, -self.imag1_64)

        self.assertEqual(cplx.conj().real, cplx_conj.real)
        self.assertEqual(cplx.conj().imag, cplx_conj.imag)
