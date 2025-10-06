import numpy as np
import pytest

import modmesh


class TestLinalgFactorization:

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
        with pytest.raises(Exception):
            modmesh.llt_factorization(A)


class TestLinalgSolver:

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
        with pytest.raises(Exception):
            modmesh.llt_solve(A, b)
