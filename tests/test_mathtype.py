import unittest

import numpy as np

import modmesh as mm

from random import random


class ComplexTC(unittest.TestCase):

    def setUp(self):
        self.real1_32 = np.float32(random())
        self.imag1_32 = np.float32(random())
        self.real2_32 = np.float32(random())
        self.imag2_32 = np.float32(random())

        self.real1_64 = np.float64(random())
        self.imag1_64 = np.float64(random())
        self.real2_64 = np.float64(random())
        self.imag2_64 = np.float64(random())

    def test_construct_float32_default(self):
        cplx = mm.ComplexFloat32()
        self.assertAlmostEqual(cplx.real, 0.0)
        self.assertAlmostEqual(cplx.imag, 0.0)

    def test_construct_float64_default(self):
        cplx = mm.ComplexFloat64()
        self.assertAlmostEqual(cplx.real, 0.0)
        self.assertAlmostEqual(cplx.imag, 0.0)

    def test_construct_float32_random(self):
        cplx = mm.ComplexFloat32(self.real1_32, self.imag1_32)
        self.assertAlmostEqual(cplx.real, self.real1_32)
        self.assertAlmostEqual(cplx.imag, self.imag1_32)

    def test_construct_float64_random(self):
        cplx = mm.ComplexFloat64(self.real1_64, self.imag1_64)
        self.assertAlmostEqual(cplx.real, self.real1_64)
        self.assertAlmostEqual(cplx.imag, self.imag1_64)

    def test_operator_add_float32(self):
        cplx1 = mm.ComplexFloat32(self.real1_32, self.imag1_32)
        cplx2 = mm.ComplexFloat32(self.real2_32, self.imag2_32)

        result = cplx1 + cplx2

        expected_real = self.real1_32 + self.real2_32
        expected_imag = self.imag1_32 + self.imag2_32

        self.assertAlmostEqual(result.real, expected_real)
        self.assertAlmostEqual(result.imag, expected_imag)

    def test_operator_add_float64(self):
        cplx1 = mm.ComplexFloat64(self.real1_64, self.imag1_64)
        cplx2 = mm.ComplexFloat64(self.real2_64, self.imag2_64)

        result = cplx1 + cplx2

        expected_real = self.real1_64 + self.real2_64
        expected_imag = self.imag1_64 + self.imag2_64

        self.assertAlmostEqual(result.real, expected_real)
        self.assertAlmostEqual(result.imag, expected_imag)

    def test_operator_sub_float32(self):
        cplx1 = mm.ComplexFloat32(self.real1_32, self.imag1_32)
        cplx2 = mm.ComplexFloat32(self.real2_32, self.imag2_32)

        result = cplx1 - cplx2

        expected_real = self.real1_32 - self.real2_32
        expected_imag = self.imag1_32 - self.imag2_32

        self.assertAlmostEqual(result.real, expected_real)
        self.assertAlmostEqual(result.imag, expected_imag)

    def test_operator_sub_float64(self):
        cplx1 = mm.ComplexFloat64(self.real1_64, self.imag1_64)
        cplx2 = mm.ComplexFloat64(self.real2_64, self.imag2_64)

        result = cplx1 - cplx2

        expected_real = self.real1_64 - self.real2_64
        expected_imag = self.imag1_64 - self.imag2_64

        self.assertAlmostEqual(result.real, expected_real)
        self.assertAlmostEqual(result.imag, expected_imag)

    def test_operator_mul_float32(self):
        cplx1 = mm.ComplexFloat32(self.real1_32, self.imag1_32)
        cplx2 = mm.ComplexFloat32(self.real2_32, self.imag2_32)

        result = cplx1 * cplx2

        expected_real = (self.real1_32 * self.real2_32
                         - self.imag1_32 * self.imag2_32)
        expected_imag = (self.real1_32 * self.imag2_32
                         + self.imag1_32 * self.real2_32)

        self.assertAlmostEqual(result.real, expected_real)
        self.assertAlmostEqual(result.imag, expected_imag)

    def test_operator_mul_float64(self):
        cplx1 = mm.ComplexFloat64(self.real1_64, self.imag1_64)
        cplx2 = mm.ComplexFloat64(self.real2_64, self.imag2_64)

        result = cplx1 * cplx2

        expected_real = (self.real1_64 * self.real2_64
                         - self.imag1_64 * self.imag2_64)
        expected_imag = (self.real1_64 * self.imag2_64
                         + self.imag1_64 * self.real2_64)

        self.assertAlmostEqual(result.real, expected_real)
        self.assertAlmostEqual(result.imag, expected_imag)

    def test_operator_div_float32_scalar(self):
        cplx = mm.ComplexFloat32(self.real1_32, self.imag1_32)
        # Avoid division by zero by explicitly assigning the divisor
        divisor = np.float32(2.0)

        result = cplx / divisor

        expected_real = self.real1_32 / divisor
        expected_imag = self.imag1_32 / divisor

        self.assertAlmostEqual(result.real, expected_real)
        self.assertAlmostEqual(result.imag, expected_imag)

    def test_operator_div_float64_scalar(self):
        cplx = mm.ComplexFloat64(self.real1_64, self.imag1_64)
        divisor = np.float64(2.0)

        result = cplx / divisor

        expected_real = self.real1_64 / divisor
        expected_imag = self.imag1_64 / divisor

        self.assertAlmostEqual(result.real, expected_real)
        self.assertAlmostEqual(result.imag, expected_imag)

    def test_norm_float32(self):
        cplx = mm.ComplexFloat32(self.real1_32, self.imag1_32)

        result = cplx.norm()

        expected_val = self.real1_32 ** 2 + self.imag1_32 ** 2

        self.assertAlmostEqual(result, expected_val)

    def test_norm_float64(self):
        cplx = mm.ComplexFloat64(self.real1_64, self.imag1_64)

        result = cplx.norm()

        expected_val = self.real1_64 ** 2 + self.imag1_64 ** 2

        self.assertAlmostEqual(result, expected_val)
