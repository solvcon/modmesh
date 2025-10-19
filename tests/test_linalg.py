import unittest

import numpy as np

import modmesh


class TestLinalgFactorization(unittest.TestCase):

    def test_llt_factorization_double_simple(self):
        L_desired = np.array([
            [2.0, 0.0, 0.0, 0.0],
            [1.0, 1.5, 0.0, 0.0],
            [0.5, 0.75, 1.0, 0.0],
            [0.25, 0.375, 0.5, 0.75]
        ])
        A_np = L_desired @ L_desired.T
        A = modmesh.SimpleArrayFloat64(array=A_np)
        L = modmesh.llt_factorization(A)
        assert L.shape == (4, 4)
        L_np = np.array(L)
        np.testing.assert_allclose(L_np, L_desired, rtol=1e-10)

    def test_llt_factorization_double_5x5(self):
        L_desired = np.array([
            [2.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.5, 0.0, 0.0, 0.0],
            [0.5, 0.75, 1.0, 0.0, 0.0],
            [0.25, 0.375, 0.5, 0.75, 0.0],
            [0.125, 0.1875, 0.25, 0.375, 0.5]
        ])
        A_np = L_desired @ L_desired.T
        A = modmesh.SimpleArrayFloat64(array=A_np)
        L = modmesh.llt_factorization(A)
        assert L.shape == (5, 5)
        L_np = np.array(L)
        np.testing.assert_allclose(L_np, L_desired, rtol=1e-10)

    def test_llt_factorization_complex_5x5(self):
        L_desired = np.array([
            [2.0+0.0j, 0.0+0.0j, 0.0+0.0j, 0.0+0.0j, 0.0+0.0j],
            [1.0+0.5j, 1.5+0.0j, 0.0+0.0j, 0.0+0.0j, 0.0+0.0j],
            [0.5+0.25j, 0.75+0.375j, 1.0+0.0j, 0.0+0.0j, 0.0+0.0j],
            [0.25+0.0j, 0.375+0.0j, 0.5+0.0j, 0.75+0.0j, 0.0+0.0j],
            [0.125+0.0625j, 0.1875+0.09375j, 0.25+0.0j, 0.375+0.0j, 0.5+0.0j]
        ])
        A_np = L_desired @ L_desired.conj().T
        A = modmesh.SimpleArrayComplex128(array=A_np)
        L = modmesh.llt_factorization(A)
        assert L.shape == (5, 5)
        L_np = np.array(L)
        np.testing.assert_allclose(L_np, L_desired, rtol=1e-10)

    def test_llt_factorization_invalid_input(self):
        A = modmesh.SimpleArrayFloat64([2, 3])
        with self.assertRaisesRegex(
                Exception,
                r"Llt::factorize: The first argument a must be a square "
                r"2D SimpleArray"
        ):
            modmesh.llt_factorization(A)


class TestLinalgSolver(unittest.TestCase):

    def test_llt_solve_double_simple(self):
        B_np = np.array([
            [2.0, 1.0, 0.5, 0.25],
            [0.0, 1.5, 0.75, 0.375],
            [0.0, 0.0, 1.0, 0.5],
            [0.0, 0.0, 0.0, 0.75]
        ])
        A_np = B_np.T @ B_np + np.eye(4)
        b_np = np.array([1.0, 2.0, 3.0, 4.0])
        A = modmesh.SimpleArrayFloat64(array=A_np)
        b = modmesh.SimpleArrayFloat64(array=b_np)
        x = modmesh.llt_solve(A, b)
        assert x.shape == (4,)
        x_np = np.array(x)
        A_np = np.array(A)
        b_np = np.array(b)
        Ax_np = A_np @ x_np
        np.testing.assert_allclose(Ax_np, b_np, rtol=1e-10)

    def test_llt_solve_double_5x5(self):
        B_np = np.array([
            [2.0, 1.0, 0.5, 0.25, 0.125],
            [0.0, 1.5, 0.75, 0.375, 0.1875],
            [0.0, 0.0, 1.0, 0.5, 0.25],
            [0.0, 0.0, 0.0, 0.75, 0.375],
            [0.0, 0.0, 0.0, 0.0, 0.5]
        ])
        A_np = B_np.T @ B_np + np.eye(5)
        b_np = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        A = modmesh.SimpleArrayFloat64(array=A_np)
        b = modmesh.SimpleArrayFloat64(array=b_np)
        x = modmesh.llt_solve(A, b)
        assert x.shape == (5,)
        x_np = np.array(x)
        A_np = np.array(A)
        b_np = np.array(b)
        Ax_np = A_np @ x_np
        np.testing.assert_allclose(Ax_np, b_np, rtol=1e-10)

    def test_llt_solve_complex_5x5(self):
        B_np = np.array([
            [2.0+0.0j, 1.0+0.5j, 0.5+0.25j, 0.25+0.0j, 0.125+0.0625j],
            [0.0+0.0j, 1.5+0.0j, 0.75+0.375j, 0.375+0.0j, 0.1875+0.09375j],
            [0.0+0.0j, 0.0+0.0j, 1.0+0.0j, 0.5+0.0j, 0.25+0.0j],
            [0.0+0.0j, 0.0+0.0j, 0.0+0.0j, 0.75+0.0j, 0.375+0.0j],
            [0.0+0.0j, 0.0+0.0j, 0.0+0.0j, 0.0+0.0j, 0.5+0.0j]
        ])
        A_np = B_np.conj().T @ B_np + np.eye(5, dtype=complex)
        b_np = np.array([1.0+0.0j, 2.0+0.0j, 3.0+0.0j, 4.0+0.0j, 5.0+0.0j])
        A = modmesh.SimpleArrayComplex128(array=A_np)
        b = modmesh.SimpleArrayComplex128(array=b_np)
        x = modmesh.llt_solve(A, b)
        assert x.shape == (5,)
        x_np = np.array(x)
        A_np = np.array(A)
        b_np = np.array(b)
        Ax_np = A_np @ x_np
        np.testing.assert_allclose(Ax_np, b_np, rtol=1e-10)

    def test_llt_solve_invalid_input(self):
        A = modmesh.SimpleArrayFloat64([2, 3])
        b = modmesh.SimpleArrayFloat64([2])
        with self.assertRaises(Exception):
            modmesh.llt_solve(A, b)

    def test_llt_solve_2d_multi_rhs_double(self):
        B_np = np.array([
            [2.0, 1.0, 0.5, 0.25, 0.125],
            [0.0, 1.5, 0.75, 0.375, 0.1875],
            [0.0, 0.0, 1.0, 0.5, 0.25],
            [0.0, 0.0, 0.0, 0.75, 0.375],
            [0.0, 0.0, 0.0, 0.0, 0.5]
        ])
        A_np = B_np.T @ B_np + np.eye(5)
        b_2d_np = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0]
        ])
        A = modmesh.SimpleArrayFloat64(array=A_np)
        b_2d = modmesh.SimpleArrayFloat64(array=b_2d_np)
        x_2d = modmesh.llt_solve(A, b_2d)
        assert x_2d.shape == (5, 3)
        x_2d_np = np.array(x_2d)
        A_np = np.array(A)
        b_2d_np = np.array(b_2d)
        Ax_2d_np = A_np @ x_2d_np
        np.testing.assert_allclose(Ax_2d_np, b_2d_np, rtol=1e-10)

    def test_llt_solve_2d_multi_rhs_complex(self):
        B_np = np.array([
            [2.0+0.0j, 1.0+0.5j, 0.5+0.25j, 0.25+0.0j],
            [0.0+0.0j, 1.5+0.0j, 0.75+0.375j, 0.375+0.0j],
            [0.0+0.0j, 0.0+0.0j, 1.0+0.0j, 0.5+0.0j],
            [0.0+0.0j, 0.0+0.0j, 0.0+0.0j, 0.75+0.0j]
        ])
        A_np = B_np.conj().T @ B_np + np.eye(4, dtype=complex)
        b_2d_np = np.array([
            [1.0+0.0j, 2.0+0.5j, 3.0+1.0j],
            [4.0+0.0j, 5.0+0.5j, 6.0+1.0j],
            [7.0+0.0j, 8.0+0.5j, 9.0+1.0j],
            [10.0+0.0j, 11.0+0.5j, 12.0+1.0j]
        ])
        A = modmesh.SimpleArrayComplex128(array=A_np)
        b_2d = modmesh.SimpleArrayComplex128(array=b_2d_np)
        x_2d = modmesh.llt_solve(A, b_2d)
        assert x_2d.shape == (4, 3)
        x_2d_np = np.array(x_2d)
        A_np = np.array(A)
        b_2d_np = np.array(b_2d)
        Ax_2d_np = A_np @ x_2d_np
        np.testing.assert_allclose(Ax_2d_np, b_2d_np, rtol=1e-10)

    def test_llt_solve_shape_mismatch_errors(self):
        # Test 1: A is not square
        A_rect = modmesh.SimpleArrayFloat64(array=np.array([
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0]
        ]))
        b = modmesh.SimpleArrayFloat64(array=np.array([1.0, 2.0, 3.0]))
        with self.assertRaisesRegex(
                Exception,
                r"Llt::solve: The first argument a must be a square "
                r"2D SimpleArray"
        ):
            modmesh.llt_solve(A_rect, b)

        # Test 2: A and b dimension mismatch (1D case)
        A_square = modmesh.SimpleArrayFloat64(array=np.array([
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0]
        ]))
        b_wrong_size = modmesh.SimpleArrayFloat64(array=np.array(
            [1.0, 2.0, 3.0]
        ))
        with self.assertRaisesRegex(
                Exception,
                r"Llt::solve: The first argument a and the second "
                r"argument b dimension mismatch"
        ):
            modmesh.llt_solve(A_square, b_wrong_size)

        # Test 3: A and b dimension mismatch (2D case)
        b_2d_wrong_size = modmesh.SimpleArrayFloat64(array=np.array([
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0]
        ]))
        with self.assertRaisesRegex(
                Exception,
                r"Llt::solve: The first argument a and the second "
                r"argument b dimension mismatch"
        ):
            modmesh.llt_solve(A_square, b_2d_wrong_size)

        # Test 4: b is 3D (should fail)
        b_3d = modmesh.SimpleArrayFloat64(array=np.array([
            [[1.0], [2.0]],
            [[3.0], [4.0]],
            [[5.0], [6.0]],
            [[7.0], [8.0]]
        ]))
        with self.assertRaisesRegex(
                Exception,
                r"Llt::solve: The second argument b must be 1D or 2D"
        ):
            modmesh.llt_solve(A_square, b_3d)

    def test_llt_solve_non_spd_matrix(self):
        # Create a non-SPD matrix
        A_nspd = np.array([
            [1.0, 2.0],
            [2.0, 1.0]
        ])
        b = modmesh.SimpleArrayFloat64(array=np.array([1.0, 2.0]))
        A = modmesh.SimpleArrayFloat64(array=A_nspd)
        with self.assertRaisesRegex(
                Exception,
                r"Llt::factorize: Cholesky failed: SimpleArray not "
                r"\(numerically\) SPD"
        ):
            modmesh.llt_solve(A, b)
