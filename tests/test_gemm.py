# Copyright (c) 2025, Wayne Chou <ck10600760@gmail.com>
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
from itertools import product
import modmesh as mm


class GemmTestBase(mm.testing.TestBase):
    """Base class for matrix multiplication (GEMM) tests"""

    def test_square_matrix_multiplication(self):
        """Test basic square matrix multiplication"""
        # Create 2x2 matrices
        a_data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=self.dtype)
        b_data = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=self.dtype)

        a = self.SimpleArray(array=a_data)
        b = self.SimpleArray(array=b_data)

        # Expected result: [[19, 22], [43, 50]]
        expected = np.array([[19.0, 22.0], [43.0, 50.0]], dtype=self.dtype)

        # Test matrix multiplication
        result = a.matmul(b)

        self.assertEqual(list(result.shape), [2, 2])
        np.testing.assert_array_almost_equal(result.ndarray, expected)

    def test_rectangular_matrix_multiplication(self):
        """Test rectangular matrix multiplication"""
        # Create 2x3 and 3x2 matrices
        a_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                          dtype=self.dtype)
        b_data = np.array([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]],
                          dtype=self.dtype)

        a = self.SimpleArray(array=a_data)
        b = self.SimpleArray(array=b_data)

        # Expected result: [[58, 64], [139, 154]]
        expected = np.array([[58.0, 64.0], [139.0, 154.0]],
                            dtype=self.dtype)

        result = a.matmul(b)

        self.assertEqual(list(result.shape), [2, 2])
        np.testing.assert_array_almost_equal(result.ndarray, expected)

    def test_identity_matrix(self):
        """Test multiplication with identity matrix"""
        # 3x3 matrix
        a_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0],
                           [7.0, 8.0, 9.0]], dtype=self.dtype)

        a = self.SimpleArray(array=a_data)
        identity = self.SimpleArray.eye(3)

        result = a.matmul(identity)

        self.assertEqual(list(result.shape), [3, 3])
        np.testing.assert_array_almost_equal(result.ndarray, a_data)

    def test_eye_method(self):
        """Test eye method creates correct identity matrices"""
        # Test cases: different sizes
        test_sizes = [1, 2, 3, 4, 5, 10]

        for size in test_sizes:
            with self.subTest(size=size):
                # Create identity matrix using our eye method
                identity = self.SimpleArray.eye(size)

                # Create expected identity matrix using NumPy
                expected = np.eye(size, dtype=self.dtype)

                # Check shape
                self.assertEqual(list(identity.shape), [size, size])

                # Check array values
                np.testing.assert_array_almost_equal(identity.ndarray,
                                                     expected)

                # Verify diagonal and off-diagonal elements explicitly
                # using product
                for i, j in product(range(size), repeat=2):
                    if i == j:
                        self.assertEqual(identity[i, j], 1.0,
                                         f"Diagonal element ({i},{j}) "
                                         f"should be 1.0")
                    else:
                        self.assertEqual(identity[i, j], 0.0,
                                         f"Off-diagonal element ({i},{j}) "
                                         f"should be 0.0")

    def test_zero_matrix(self):
        """Test multiplication with zero matrix"""
        a_data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=self.dtype)
        zero_data = np.zeros((2, 2), dtype=self.dtype)

        a = self.SimpleArray(array=a_data)
        zero = self.SimpleArray(array=zero_data)

        result = a.matmul(zero)

        self.assertEqual(list(result.shape), [2, 2])
        np.testing.assert_array_almost_equal(result.ndarray, zero_data)

    def test_dimension_mismatch_error(self):
        """Test error handling for incompatible dimensions"""

        a_data = np.array([[1.0, 2.0], [3.0, 4.0]],
                          dtype=self.dtype)  # 2x2
        b_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0],
                           [7.0, 8.0, 9.0]], dtype=self.dtype)  # 3x3

        a = self.SimpleArray(array=a_data)
        b = self.SimpleArray(array=b_data)

        # Should raise error: 2x2 cannot multiply with 3x3
        # (incompatible inner dimensions: 2 != 3)
        with self.assertRaisesRegex(
            IndexError,
            r"SimpleArray::matmul\(\): shape mismatch: this=\(2,2\) other="
            r"\(3,3\)"
        ):
            a.matmul(b)

    def test_compare_with_numpy(self):
        """Compare results with NumPy using fixed test data"""

        # Test case 1: (2x3) x (3x4)
        a_data_1 = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0]
        ], dtype=self.dtype)
        b_data_1 = np.array([
            [7.0, 8.0, 9.0, 10.0],
            [11.0, 12.0, 13.0, 14.0],
            [15.0, 16.0, 17.0, 18.0]
        ], dtype=self.dtype)
        expected_1 = np.array([
            [74.0, 80.0, 86.0, 92.0],
            [173.0, 188.0, 203.0, 218.0]
        ], dtype=self.dtype)

        # Test case 2: (4x6) x (6x3)
        a_data_2 = np.array([
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0, 17.0, 18.0],
            [19.0, 20.0, 21.0, 22.0, 23.0, 24.0]
        ], dtype=self.dtype)
        b_data_2 = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0],
            [16.0, 17.0, 18.0]
        ], dtype=self.dtype)
        expected_2 = np.array([
            [231.0, 252.0, 273.0],
            [537.0, 594.0, 651.0],
            [843.0, 936.0, 1029.0],
            [1149.0, 1278.0, 1407.0]
        ], dtype=self.dtype)

        # Test case 3: (3x3) x (3x3)
        a_data_3 = np.array([
            [2.0, 1.0, 3.0],
            [1.0, 4.0, 2.0],
            [3.0, 2.0, 1.0]
        ], dtype=self.dtype)
        b_data_3 = np.array([
            [1.0, 2.0, 1.0],
            [2.0, 1.0, 3.0],
            [1.0, 3.0, 2.0]
        ], dtype=self.dtype)
        expected_3 = np.array([
            [7.0, 14.0, 11.0],
            [11.0, 12.0, 17.0],
            [8.0, 11.0, 11.0]
        ], dtype=self.dtype)

        # Test case 4: (4x6) x (6)
        a_data_4 = np.array([
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0, 17.0, 18.0],
            [19.0, 20.0, 21.0, 22.0, 23.0, 24.0]
        ], dtype=self.dtype)
        b_data_4 = np.array([1., 2., 3., 4., 5., 6], dtype=self.dtype)
        expected_4 = np.array([91., 217., 343., 469.], dtype=self.dtype)

        # Test case 5: (6) x (6 x 4)
        a_data_5 = np.array([1., 2., 3., 4., 5., 6], dtype=self.dtype)
        b_data_5 = np.array([
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0],
            [17.0, 18.0, 19.0, 20.0],
            [21.0, 22.0, 23.0, 24.0]
        ], dtype=self.dtype)
        expected_5 = np.array([301., 322., 343., 364.], dtype=self.dtype)

        # Test case 6: (3) x (3)
        a_data_6 = np.array([1., 2., 3.], dtype=self.dtype)
        b_data_6 = np.array([4., 5., 6.], dtype=self.dtype)
        expected_6 = np.array([32.], dtype=self.dtype)

        test_cases = [
            (a_data_1, b_data_1, expected_1, "2x3 x 3x4"),
            (a_data_2, b_data_2, expected_2, "4x6 x 6x3"),
            (a_data_3, b_data_3, expected_3, "3x3 x 3x3"),
            (a_data_4, b_data_4, expected_4, "4x6 x 6"),
            (a_data_5, b_data_5, expected_5, "6 x 6x3"),
            (a_data_6, b_data_6, expected_6, "3 x 3x3")
        ]

        for a_data, b_data, expected, description in test_cases:
            with self.subTest(description=description):
                a = self.SimpleArray(array=a_data)
                b = self.SimpleArray(array=b_data)

                # Compute with our implementation
                result = a.matmul(b)

                # Verify with NumPy
                np_result = np.matmul(a_data, b_data)
                np.testing.assert_array_almost_equal(expected, np_result)

                # Compare our result with expected
                self.assertEqual(list(result.shape), list(expected.shape))
                if self.dtype == np.float32:
                    np.testing.assert_array_almost_equal(
                        result.ndarray, expected, decimal=4)
                else:
                    np.testing.assert_array_almost_equal(
                        result.ndarray, expected, decimal=10)

    def test_wrong_shape_error(self):
        """Test error handling for wrong shapes"""

        a_3d_data = np.array([[[1.0, 2.0], [3.0, 4.0]],
                              [[5.0, 6.0], [7.0, 8.0]]], dtype=self.dtype)
        b_3d_data = np.array([[[1.0, 0.0], [0.0, 1.0]],
                              [[2.0, 0.0], [0.0, 2.0]]], dtype=self.dtype)
        a_3d = self.SimpleArray(array=a_3d_data)
        b_3d = self.SimpleArray(array=b_3d_data)

        with self.assertRaisesRegex(
            IndexError,
            r"SimpleArray::matmul\(\): unsupported dimensions: "
            r"this=\(2,2,2\) other=\(2,2,2\)\. SimpleArray must be 1D or 2D."
        ):
            a_3d.matmul(b_3d)

        a = np.zeros((3, 3), dtype=self.dtype)
        b = np.zeros((2, 3), dtype=self.dtype)
        a = self.SimpleArray(array=a)
        b = self.SimpleArray(array=b)
        with self.assertRaisesRegex(
            IndexError,
            r"SimpleArray::matmul\(\): shape mismatch: "
            r"this=\(3,3\) other=\(2,3\)"
        ):
            a.matmul(b)

        a = np.zeros((3, 3), dtype=self.dtype)
        b = np.zeros((2), dtype=self.dtype)
        a = self.SimpleArray(array=a)
        b = self.SimpleArray(array=b)
        with self.assertRaisesRegex(
            IndexError,
            r"SimpleArray::matmul\(\): shape mismatch: "
            r"this=\(3,3\) other=\(2\)"
        ):
            a.matmul(b)

        a = np.zeros((2), dtype=self.dtype)
        b = np.zeros((3, 3), dtype=self.dtype)
        a = self.SimpleArray(array=a)
        b = self.SimpleArray(array=b)
        with self.assertRaisesRegex(
            IndexError,
            r"SimpleArray::matmul\(\): shape mismatch: "
            r"this=\(2\) other=\(3,3\)"
        ):
            a.matmul(b)

        a = np.zeros((2), dtype=self.dtype)
        b = np.zeros((3), dtype=self.dtype)
        a = self.SimpleArray(array=a)
        b = self.SimpleArray(array=b)
        with self.assertRaisesRegex(
            IndexError,
            r"SimpleArray::matmul\(\): shape mismatch: "
            r"this=\(2\) other=\(3\)"
        ):
            a.matmul(b)

    def test_matmul_operator(self):
        """Test @ operator for matrix multiplication"""
        # Create 2x2 matrices
        a_data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=self.dtype)
        b_data = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=self.dtype)

        a = self.SimpleArray(array=a_data)
        b = self.SimpleArray(array=b_data)

        # Expected result: [[19, 22], [43, 50]]
        expected = np.array([[19.0, 22.0], [43.0, 50.0]], dtype=self.dtype)

        # Test @ operator
        result = a @ b

        self.assertEqual(list(result.shape), [2, 2])
        np.testing.assert_array_almost_equal(result.ndarray, expected)

    def test_imatmul_method(self):
        """Test imatmul() method for in-place matrix multiplication"""
        # Create 2x2 matrices
        a_data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=self.dtype)
        b_data = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=self.dtype)

        a = self.SimpleArray(array=a_data)
        b = self.SimpleArray(array=b_data)

        # Expected result: [[19, 22], [43, 50]]
        expected = np.array([[19.0, 22.0], [43.0, 50.0]], dtype=self.dtype)

        # Test imatmul() method
        a.imatmul(b)

        # Verify the result
        self.assertEqual(list(a.shape), [2, 2])
        np.testing.assert_array_almost_equal(a.ndarray, expected)

    def test_imatmul_operator(self):
        """Test @= operator for in-place matrix multiplication"""
        # Create 2x2 matrices
        a_data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=self.dtype)
        b_data = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=self.dtype)

        a = self.SimpleArray(array=a_data)
        b = self.SimpleArray(array=b_data)

        # Expected result: [[19, 22], [43, 50]]
        expected = np.array([[19.0, 22.0], [43.0, 50.0]], dtype=self.dtype)

        # Test @= operator
        a @= b

        self.assertEqual(list(a.shape), [2, 2])
        np.testing.assert_array_almost_equal(a.ndarray, expected)


class GemmFloat32TC(GemmTestBase, unittest.TestCase):
    """Test matrix multiplication with float32"""

    def setUp(self):
        self.dtype = np.float32
        self.SimpleArray = mm.SimpleArrayFloat32


class GemmFloat64TC(GemmTestBase, unittest.TestCase):
    """Test matrix multiplication with float64"""

    def setUp(self):
        self.dtype = np.float64
        self.SimpleArray = mm.SimpleArrayFloat64

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
