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
        cplx = mm.ComplexFloat32()
        self.assert_allclose32(cplx.real, 0.0)
        self.assert_allclose32(cplx.imag, 0.0)

    def test_construct_float64_default(self):
        cplx = mm.ComplexFloat64()
        self.assert_allclose64(cplx.real, 0.0)
        self.assert_allclose64(cplx.imag, 0.0)

    def test_construct_float32_random(self):
        cplx = mm.ComplexFloat32(self.real1_32, self.imag1_32)
        self.assert_allclose32(cplx.real, self.real1_32)
        self.assert_allclose32(cplx.imag, self.imag1_32)

    def test_construct_float64_random(self):
        cplx = mm.ComplexFloat64(self.real1_64, self.imag1_64)
        self.assert_allclose64(cplx.real, self.real1_64)
        self.assert_allclose64(cplx.imag, self.imag1_64)

    def test_operator_add_float32(self):
        cplx1 = mm.ComplexFloat32(self.real1_32, self.imag1_32)
        cplx2 = mm.ComplexFloat32(self.real2_32, self.imag2_32)

        result = cplx1 + cplx2

        expected_real = self.real1_32 + self.real2_32
        expected_imag = self.imag1_32 + self.imag2_32

        self.assert_allclose32(result.real, expected_real)
        self.assert_allclose32(result.imag, expected_imag)

    def test_operator_add_float64(self):
        cplx1 = mm.ComplexFloat64(self.real1_64, self.imag1_64)
        cplx2 = mm.ComplexFloat64(self.real2_64, self.imag2_64)

        result = cplx1 + cplx2

        expected_real = self.real1_64 + self.real2_64
        expected_imag = self.imag1_64 + self.imag2_64

        self.assert_allclose64(result.real, expected_real)
        self.assert_allclose64(result.imag, expected_imag)

    def test_operator_sub_float32(self):
        cplx1 = mm.ComplexFloat32(self.real1_32, self.imag1_32)
        cplx2 = mm.ComplexFloat32(self.real2_32, self.imag2_32)

        result = cplx1 - cplx2

        expected_real = self.real1_32 - self.real2_32
        expected_imag = self.imag1_32 - self.imag2_32

        self.assert_allclose32(result.real, expected_real)
        self.assert_allclose32(result.imag, expected_imag)

    def test_operator_sub_float64(self):
        cplx1 = mm.ComplexFloat64(self.real1_64, self.imag1_64)
        cplx2 = mm.ComplexFloat64(self.real2_64, self.imag2_64)

        result = cplx1 - cplx2

        expected_real = self.real1_64 - self.real2_64
        expected_imag = self.imag1_64 - self.imag2_64

        self.assert_allclose64(result.real, expected_real)
        self.assert_allclose64(result.imag, expected_imag)

    def test_operator_mul_float32(self):
        cplx1 = mm.ComplexFloat32(self.real1_32, self.imag1_32)
        cplx2 = mm.ComplexFloat32(self.real2_32, self.imag2_32)

        result = cplx1 * cplx2

        expected_real = (self.real1_32 * self.real2_32
                         - self.imag1_32 * self.imag2_32)
        expected_imag = (self.real1_32 * self.imag2_32
                         + self.imag1_32 * self.real2_32)

        self.assert_allclose32(result.real, expected_real)
        self.assert_allclose32(result.imag, expected_imag)

    def test_operator_mul_float64(self):
        cplx1 = mm.ComplexFloat64(self.real1_64, self.imag1_64)
        cplx2 = mm.ComplexFloat64(self.real2_64, self.imag2_64)

        result = cplx1 * cplx2

        expected_real = (self.real1_64 * self.real2_64
                         - self.imag1_64 * self.imag2_64)
        expected_imag = (self.real1_64 * self.imag2_64
                         + self.imag1_64 * self.real2_64)

        self.assert_allclose64(result.real, expected_real)
        self.assert_allclose64(result.imag, expected_imag)

    def test_operator_div_float32_scalar(self):
        cplx = mm.ComplexFloat32(self.real1_32, self.imag1_32)
        # Avoid division by zero by explicitly assigning the divisor
        divisor = np.float32(2.0)

        result = cplx / divisor

        expected_real = self.real1_32 / divisor
        expected_imag = self.imag1_32 / divisor

        self.assert_allclose32(result.real, expected_real)
        self.assert_allclose32(result.imag, expected_imag)

    def test_operator_div_float64_scalar(self):
        cplx = mm.ComplexFloat64(self.real1_64, self.imag1_64)
        divisor = np.float64(2.0)

        result = cplx / divisor

        expected_real = self.real1_64 / divisor
        expected_imag = self.imag1_64 / divisor

        self.assert_allclose64(result.real, expected_real)
        self.assert_allclose64(result.imag, expected_imag)

    def test_norm_float32(self):
        cplx = mm.ComplexFloat32(self.real1_32, self.imag1_32)

        result = cplx.norm()

        expected_val = self.real1_32 ** 2 + self.imag1_32 ** 2

        self.assert_allclose32(result, expected_val)

    def test_norm_float64(self):
        cplx = mm.ComplexFloat64(self.real1_64, self.imag1_64)

        result = cplx.norm()

        expected_val = self.real1_64 ** 2 + self.imag1_64 ** 2

        self.assert_allclose64(result, expected_val)
