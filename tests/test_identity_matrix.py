# Copyright (c) 2025, Tetsuya Koyama <tetsuokoyama@example.com>
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


class IdentityMatrixTestBase(mm.testing.TestBase):
    """Base class for identity matrix tests"""

    def test_basic_identity_matrix_2x2(self):
        """Test basic 2x2 identity matrix generation"""
        identity = self.SimpleArray.eye(2)
        expected = np.eye(2, dtype=self.dtype)

        self.assertEqual(list(identity.shape), [2, 2])
        np.testing.assert_array_equal(identity.ndarray, expected)

    def test_basic_identity_matrix_3x3(self):
        """Test basic 3x3 identity matrix generation"""
        identity = self.SimpleArray.eye(3)
        expected = np.eye(3, dtype=self.dtype)

        self.assertEqual(list(identity.shape), [3, 3])
        np.testing.assert_array_equal(identity.ndarray, expected)

    def test_large_identity_matrix(self):
        """Test larger identity matrix generation"""
        n = 10
        identity = self.SimpleArray.eye(n)
        expected = np.eye(n, dtype=self.dtype)

        self.assertEqual(list(identity.shape), [n, n])
        np.testing.assert_array_equal(identity.ndarray, expected)

    def test_single_element_identity_matrix(self):
        """Test 1x1 identity matrix (edge case)"""
        identity = self.SimpleArray.eye(1)
        expected = np.eye(1, dtype=self.dtype)

        self.assertEqual(list(identity.shape), [1, 1])
        np.testing.assert_array_equal(identity.ndarray, expected)

    def test_identity_matrix_properties(self):
        """Test mathematical properties of identity matrices"""
        n = 4
        identity = self.SimpleArray.eye(n)

        # Test that diagonal elements are 1
        for i in range(n):
            self.assertEqual(identity.ndarray[i, i], 1.0)

        # Test that off-diagonal elements are 0
        for i in range(n):
            for j in range(n):
                if i != j:
                    self.assertEqual(identity.ndarray[i, j], 0.0)

    def test_identity_matrix_multiplication_property(self):
        """Test that A * I = I * A = A for any matrix A"""
        # Create a test matrix
        a_data = np.array([[1.0, 2.0, 3.0], 
                           [4.0, 5.0, 6.0], 
                           [7.0, 8.0, 9.0]], dtype=self.dtype)
        a = self.SimpleArray(array=a_data)
        identity = self.SimpleArray.eye(3)

        # Test A * I = A
        result_right = a.matmul(identity)
        np.testing.assert_array_almost_equal(result_right.ndarray, a_data)

        # Test I * A = A
        result_left = identity.matmul(a)
        np.testing.assert_array_almost_equal(result_left.ndarray, a_data)

    def test_identity_matrix_multiplication_with_rectangular_matrix(self):
        """Test identity matrix multiplication with rectangular matrices"""
        # Test with 3x4 matrix
        a_data = np.array([[1.0, 2.0, 3.0, 4.0],
                           [5.0, 6.0, 7.0, 8.0],
                           [9.0, 10.0, 11.0, 12.0]], dtype=self.dtype)
        a = self.SimpleArray(array=a_data)
        identity3 = self.SimpleArray.eye(3)
        identity4 = self.SimpleArray.eye(4)

        # Test I3 * A = A (where A is 3x4)
        result_left = identity3.matmul(a)
        np.testing.assert_array_almost_equal(result_left.ndarray, a_data)

        # Test A * I4 = A (where A is 3x4)
        result_right = a.matmul(identity4)
        np.testing.assert_array_almost_equal(result_right.ndarray, a_data)

    def test_identity_matrix_with_zeros_matrix(self):
        """Test identity matrix multiplication with zero matrix"""
        n = 3
        zero_data = np.zeros((n, n), dtype=self.dtype)
        zero_matrix = self.SimpleArray(array=zero_data)
        identity = self.SimpleArray.eye(n)

        # Test I * 0 = 0
        result = identity.matmul(zero_matrix)
        np.testing.assert_array_equal(result.ndarray, zero_data)

        # Test 0 * I = 0
        result = zero_matrix.matmul(identity)
        np.testing.assert_array_equal(result.ndarray, zero_data)

    def test_identity_matrix_comparison_with_numpy(self):
        """Compare identity matrix generation with NumPy"""
        test_sizes = [1, 2, 3, 5, 8, 10]
        
        for n in test_sizes:
            with self.subTest(size=n):
                identity = self.SimpleArray.eye(n)
                np_identity = np.eye(n, dtype=self.dtype)
                
                self.assertEqual(list(identity.shape), [n, n])
                np.testing.assert_array_equal(identity.ndarray, np_identity)

    def test_identity_matrix_data_integrity(self):
        """Test that identity matrix data is properly initialized"""
        n = 5
        identity = self.SimpleArray.eye(n)
        
        # Verify total number of elements
        self.assertEqual(identity.size, n * n)
        
        # Verify sum of diagonal elements equals n
        diagonal_sum = sum(identity.ndarray[i, i] for i in range(n))
        self.assertEqual(diagonal_sum, n)
        
        # Verify sum of all elements equals n (since only diagonal is 1)
        total_sum = np.sum(identity.ndarray)
        self.assertAlmostEqual(total_sum, n, places=10)


class IdentityMatrixFloat32TC(IdentityMatrixTestBase, unittest.TestCase):
    """Test identity matrix generation with float32"""

    def setUp(self):
        self.dtype = np.float32
        self.SimpleArray = mm.SimpleArrayFloat32


class IdentityMatrixFloat64TC(IdentityMatrixTestBase, unittest.TestCase):
    """Test identity matrix generation with float64"""

    def setUp(self):
        self.dtype = np.float64
        self.SimpleArray = mm.SimpleArrayFloat64


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4: