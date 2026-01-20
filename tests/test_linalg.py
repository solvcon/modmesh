import unittest

import numpy as np

import modmesh as mm


class TestLinalgFactorization(unittest.TestCase):

    def test_llt_factorization_double_simple(self):
        L_desired = np.array([
            [2.0, 0.0, 0.0, 0.0],
            [1.0, 1.5, 0.0, 0.0],
            [0.5, 0.75, 1.0, 0.0],
            [0.25, 0.375, 0.5, 0.75]
        ])
        A_np = L_desired @ L_desired.T
        A = mm.SimpleArrayFloat64(array=A_np)
        L = mm.llt_factorization(A)
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
        A = mm.SimpleArrayFloat64(array=A_np)
        L = mm.llt_factorization(A)
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
        A = mm.SimpleArrayComplex128(array=A_np)
        L = mm.llt_factorization(A)
        assert L.shape == (5, 5)
        L_np = np.array(L)
        np.testing.assert_allclose(L_np, L_desired, rtol=1e-10)

    def test_llt_factorization_invalid_input(self):
        A = mm.SimpleArrayFloat64([2, 3])
        with self.assertRaisesRegex(
                Exception,
                r"Llt::factorize: The first argument a must be a square "
                r"2D SimpleArray"
        ):
            mm.llt_factorization(A)


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
        A = mm.SimpleArrayFloat64(array=A_np)
        b = mm.SimpleArrayFloat64(array=b_np)
        x = mm.llt_solve(A, b)
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
        A = mm.SimpleArrayFloat64(array=A_np)
        b = mm.SimpleArrayFloat64(array=b_np)
        x = mm.llt_solve(A, b)
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
        A = mm.SimpleArrayComplex128(array=A_np)
        b = mm.SimpleArrayComplex128(array=b_np)
        x = mm.llt_solve(A, b)
        assert x.shape == (5,)
        x_np = np.array(x)
        A_np = np.array(A)
        b_np = np.array(b)
        Ax_np = A_np @ x_np
        np.testing.assert_allclose(Ax_np, b_np, rtol=1e-10)

    def test_llt_solve_invalid_input(self):
        A = mm.SimpleArrayFloat64([2, 3])
        b = mm.SimpleArrayFloat64([2])
        with self.assertRaises(Exception):
            mm.llt_solve(A, b)

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
        A = mm.SimpleArrayFloat64(array=A_np)
        b_2d = mm.SimpleArrayFloat64(array=b_2d_np)
        x_2d = mm.llt_solve(A, b_2d)
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
        A = mm.SimpleArrayComplex128(array=A_np)
        b_2d = mm.SimpleArrayComplex128(array=b_2d_np)
        x_2d = mm.llt_solve(A, b_2d)
        assert x_2d.shape == (4, 3)
        x_2d_np = np.array(x_2d)
        A_np = np.array(A)
        b_2d_np = np.array(b_2d)
        Ax_2d_np = A_np @ x_2d_np
        np.testing.assert_allclose(Ax_2d_np, b_2d_np, rtol=1e-10)

    def test_llt_solve_shape_mismatch_errors(self):
        # Test 1: A is not square
        A_rect = mm.SimpleArrayFloat64(array=np.array([
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0]
        ]))
        b = mm.SimpleArrayFloat64(array=np.array([1.0, 2.0, 3.0]))
        with self.assertRaisesRegex(
                Exception,
                r"Llt::solve: The first argument a must be a square "
                r"2D SimpleArray"
        ):
            mm.llt_solve(A_rect, b)

        # Test 2: A and b dimension mismatch (1D case)
        A_square = mm.SimpleArrayFloat64(array=np.array([
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0]
        ]))
        b_wrong_size = mm.SimpleArrayFloat64(array=np.array(
            [1.0, 2.0, 3.0]
        ))
        with self.assertRaisesRegex(
                Exception,
                r"Llt::solve: The first argument a and the second "
                r"argument b dimension mismatch"
        ):
            mm.llt_solve(A_square, b_wrong_size)

        # Test 3: A and b dimension mismatch (2D case)
        b_2d_wrong_size = mm.SimpleArrayFloat64(array=np.array([
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0]
        ]))
        with self.assertRaisesRegex(
                Exception,
                r"Llt::solve: The first argument a and the second "
                r"argument b dimension mismatch"
        ):
            mm.llt_solve(A_square, b_2d_wrong_size)

        # Test 4: b is 3D (should fail)
        b_3d = mm.SimpleArrayFloat64(array=np.array([
            [[1.0], [2.0]],
            [[3.0], [4.0]],
            [[5.0], [6.0]],
            [[7.0], [8.0]]
        ]))
        with self.assertRaisesRegex(
                Exception,
                r"Llt::solve: The second argument b must be 1D or 2D"
        ):
            mm.llt_solve(A_square, b_3d)

    def test_llt_solve_non_spd_matrix(self):
        # Create a non-SPD matrix
        A_nspd = np.array([
            [1.0, 2.0],
            [2.0, 1.0]
        ])
        b = mm.SimpleArrayFloat64(array=np.array([1.0, 2.0]))
        A = mm.SimpleArrayFloat64(array=A_nspd)
        with self.assertRaisesRegex(
                Exception,
                r"Llt::factorize: Cholesky failed: SimpleArray not "
                r"\(numerically\) SPD"
        ):
            mm.llt_solve(A, b)


class KalmanFilterInitTC(unittest.TestCase):

    def test_transition_dimension_error(self):
        x0 = mm.SimpleArrayFloat64([2])
        f_wrong = mm.SimpleArrayFloat64([1, 2])
        h = mm.SimpleArrayFloat64([1, 2])

        with self.assertRaisesRegex(
                ValueError,
                "KalmanFilter::check_dimensions: The state "
                "transition SimpleArray f must be state_sizexstate_size, "
                "but got shape \\(1, 2\\)"):
            mm.KalmanFilterFp64(
                x=x0, f=f_wrong, h=h, process_noise=0.1, measurement_noise=1.0)

    def test_measurement_dimension_error(self):
        x0 = mm.SimpleArrayFloat64([2])
        f = mm.SimpleArrayFloat64([2, 2])
        h_wrong = mm.SimpleArrayFloat64([1, 1])

        with self.assertRaisesRegex(
                ValueError,
                "KalmanFilter::check_dimensions: The "
                "measurement SimpleArray h must be "
                "measurement_sizexstate_size, but got shape \\(1, 1\\)"):
            mm.KalmanFilterFp64(
                x=x0, f=f, h=h_wrong, process_noise=0.1, measurement_noise=1.0)

    def test_state_dimension_error(self):
        x0_wrong = mm.SimpleArrayFloat64([2, 2])
        f = mm.SimpleArrayFloat64([2, 2])
        h = mm.SimpleArrayFloat64([1, 2])

        with self.assertRaisesRegex(
                ValueError,
                "KalmanFilter::check_dimensions: The state SimpleArray "
                "x must be 1D of length state_size, but got shape \\(2, 2\\)"):
            mm.KalmanFilterFp64(
                x=x0_wrong, f=f, h=h, process_noise=0.1, measurement_noise=1.0)

    def test_control_1d_error(self):
        x0 = mm.SimpleArrayFloat64([2])
        f = mm.SimpleArrayFloat64([2, 2])
        b_wrong = mm.SimpleArrayFloat64([2])
        h = mm.SimpleArrayFloat64([1, 2])

        with self.assertRaisesRegex(
                ValueError,
                "KalmanFilter::check_dimensions: The control SimpleArray "
                "b must be state_sizex0 when control_size = 0, but got "
                "shape \\(2\\)"):
            mm.KalmanFilterFp64(
                x=x0, f=f, b=b_wrong, h=h, process_noise=0.1,
                measurement_noise=1.0)

    def test_control_2d_error(self):
        x0 = mm.SimpleArrayFloat64([2])
        f = mm.SimpleArrayFloat64([2, 2])
        b_wrong = mm.SimpleArrayFloat64([1, 2])
        h = mm.SimpleArrayFloat64([1, 2])

        with self.assertRaisesRegex(
                ValueError,
                "KalmanFilter::check_dimensions: The control SimpleArray "
                "b must be state_sizexcontrol_size, but got shape \\(1, 2\\)"):
            mm.KalmanFilterFp64(
                x=x0, f=f, b=b_wrong, h=h, process_noise=0.1,
                measurement_noise=1.0)


class KalmanFilterPredictTC(unittest.TestCase):

    def kf_predict_numpy(self, x, p, f, q):
        x_pred = f @ x
        p_pred = f @ p @ f.conj().T + q
        return x_pred, p_pred

    def test_predict_fp64(self):
        n = 2
        x0 = np.array([1.0, 2.0])
        f = np.array([[1.1, 0.2],
                      [0.1, 0.9]])
        p0 = np.eye(n)
        sigma_w = 0.316
        q = (sigma_w**2) * np.eye(n)
        x_pred_np, p_pred_np = self.kf_predict_numpy(x0, p0, f, q)

        x_sa = mm.SimpleArrayFloat64(array=x0)
        f_sa = mm.SimpleArrayFloat64(array=f)
        h_sa = mm.SimpleArrayFloat64([1, n])
        kf = mm.KalmanFilterFp64(
            x=x_sa, f=f_sa, h=h_sa,
            process_noise=sigma_w,
            measurement_noise=1.0,
        )
        kf.predict()

        x_pred_mm = kf.state.ndarray
        p_pred_mm = kf.covariance.ndarray
        np.testing.assert_allclose(x_pred_mm, x_pred_np, atol=1e-12, rtol=0.0)
        np.testing.assert_allclose(p_pred_mm, p_pred_np, atol=1e-12, rtol=0.0)

    def test_predict_cp128(self):
        n = 2
        x0 = np.array([1.0+0.5j, 2.0-0.3j], dtype=np.complex128)
        f = np.array([[1.1+0.1j, 0.2-0.1j],
                      [0.1+0.1j, 0.9+0.0j]], dtype=np.complex128)
        p0 = np.eye(n, dtype=np.complex128)
        sigma_w = 0.316
        q = (sigma_w**2) * np.eye(n, dtype=np.complex128)
        x_pred_np, p_pred_np = self.kf_predict_numpy(x0, p0, f, q)

        x_sa = mm.SimpleArrayComplex128(array=x0)
        f_sa = mm.SimpleArrayComplex128(array=f)
        h_sa = mm.SimpleArrayComplex128([1, n])
        kf = mm.KalmanFilterComplex128(
            x=x_sa, f=f_sa, h=h_sa,
            process_noise=sigma_w,
            measurement_noise=1.0,
        )
        kf.predict()

        x_pred_mm = kf.state.ndarray
        p_pred_mm = kf.covariance.ndarray
        np.testing.assert_allclose(x_pred_mm, x_pred_np, atol=1e-12, rtol=0.0)
        np.testing.assert_allclose(p_pred_mm, p_pred_np, atol=1e-12, rtol=0.0)


# The bug is reported in issue #603:
# https://github.com/solvcon/modmesh/issues/603
# Issue #603: F-contiguous check in SimpleArray breaks when creating
# SimpleArray
# with shape = (n, 1) or shape = (1, n) from ndarray
# This sa_from_np function provides a workaround for the F-contiguous check
# issue by manually creating SimpleArray and copying elements instead of using
# constructor
def sa_from_np(arr: np.ndarray, cls):
    arr = np.asarray(arr)
    if arr.ndim == 1:
        sa = cls([arr.shape[0]])
        for i in range(arr.shape[0]):
            sa[i] = arr[i]
        return sa
    if arr.ndim == 2:
        m, n = arr.shape
        sa = cls([m, n])
        for i in range(m):
            for j in range(n):
                sa[i, j] = arr[i, j]
        return sa
    raise ValueError("sa_from_np supports only 1D or 2D arrays")


class TestKnownIssues603(unittest.TestCase):

    @unittest.expectedFailure
    def test_issue_603_shape_n_by_1(self):
        narr = np.array([[1.0], [0.0]])
        sa = mm.SimpleArrayFloat64(array=narr)
        self.assertEqual(sa.shape(), [2, 1])

    @unittest.expectedFailure
    def test_issue_603_shape_1_by_n(self):
        narr = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
        sa = mm.SimpleArrayFloat64(array=narr)
        self.assertEqual(sa.shape(), [1, 5])


class KalmanFilterUpdateTC(unittest.TestCase):

    def kf_update_numpy(self, x_pred, p_pred, h, r_var, z):
        k = h.shape[0]
        y = z - h @ x_pred
        s = h @ p_pred @ h.conj().T + r_var * np.eye(k, dtype=h.dtype)
        k_gain = p_pred @ h.conj().T @ np.linalg.inv(s)
        x_upd = x_pred + k_gain @ y
        i_n = np.eye(p_pred.shape[0], dtype=p_pred.dtype)
        a = i_n - k_gain @ h
        p_upd = (a @ p_pred @ a.conj().T +
                 k_gain @ (r_var * np.eye(k, dtype=h.dtype)) @
                 k_gain.conj().T)
        return x_upd, p_upd

    def test_update_fp64(self):
        n = 2
        x_pred = np.array([0.3, -0.2])
        p_pred = np.eye(n)
        f = np.eye(n)
        h = np.array([[1.0, 0.2]])
        sigma_v = 0.4
        r_var = sigma_v**2
        z = np.array([0.15])

        x_upd_np, p_upd_np = self.kf_update_numpy(
            x_pred, p_pred, h, r_var, z)

        x_sa = sa_from_np(x_pred, mm.SimpleArrayFloat64)
        f_sa = sa_from_np(f, mm.SimpleArrayFloat64)
        h_sa = sa_from_np(h, mm.SimpleArrayFloat64)
        z_sa = sa_from_np(z, mm.SimpleArrayFloat64)

        kf = mm.KalmanFilterFp64(
            x=x_sa, f=f_sa, h=h_sa,
            process_noise=1.0, measurement_noise=sigma_v)
        kf.update(z_sa)

        x_upd_mm = kf.state.ndarray
        p_upd_mm = kf.covariance.ndarray

        np.testing.assert_allclose(x_upd_mm, x_upd_np, atol=1e-10, rtol=1e-10)
        np.testing.assert_allclose(p_upd_mm, p_upd_np, atol=1e-10, rtol=1e-10)
        np.testing.assert_allclose(p_upd_mm, p_upd_mm.T, atol=1e-12, rtol=0.0)
        self.assertTrue(np.all(np.diag(p_upd_mm) > 0.0))

    def test_update_cp128(self):
        n = 2
        x_pred = np.array([0.3 + 0.1j, -0.2 - 0.05j], dtype=np.complex128)
        p_pred = np.eye(n, dtype=np.complex128)
        f = np.eye(n, dtype=np.complex128)
        h = np.array([[1.0 + 0.0j, 0.2 - 0.1j]], dtype=np.complex128)
        sigma_v = 0.5
        r_var = sigma_v**2
        z = np.array([0.15 - 0.2j], dtype=np.complex128)

        x_upd_np, p_upd_np = self.kf_update_numpy(
            x_pred, p_pred, h, r_var, z)

        x_sa = sa_from_np(x_pred, mm.SimpleArrayComplex128)
        f_sa = sa_from_np(f, mm.SimpleArrayComplex128)
        h_sa = sa_from_np(h, mm.SimpleArrayComplex128)
        z_sa = sa_from_np(z, mm.SimpleArrayComplex128)

        kf = mm.KalmanFilterComplex128(
            x=x_sa, f=f_sa, h=h_sa,
            process_noise=1.0, measurement_noise=sigma_v)
        kf.update(z_sa)

        x_upd_mm = kf.state.ndarray
        p_upd_mm = kf.covariance.ndarray

        np.testing.assert_allclose(x_upd_mm, x_upd_np, atol=1e-9, rtol=1e-9)
        np.testing.assert_allclose(p_upd_mm, p_upd_np, atol=1e-9, rtol=1e-9)
        np.testing.assert_allclose(p_upd_mm, p_upd_mm.conj().T,
                                   atol=1e-12, rtol=0.0)
        self.assertTrue(np.all(np.real(np.diag(p_upd_mm)) > 0.0))
        self.assertTrue(np.all(np.abs(np.imag(np.diag(p_upd_mm))) < 1e-12))

    def test_update_measurement_dimension_error(self):
        n = 2
        x0 = mm.SimpleArrayFloat64([n])
        f = mm.SimpleArrayFloat64([n, n])
        h = mm.SimpleArrayFloat64([1, n])

        kf = mm.KalmanFilterFp64(
            x=x0, f=f, h=h,
            process_noise=0.1, measurement_noise=1.0)

        # Test wrong measurement dimension
        z_wrong = mm.SimpleArrayFloat64([2])  # Should be length 1

        with self.assertRaisesRegex(
                ValueError,
                "KalmanFilter::check_measurement: The measurement "
                "SimpleArray z must be 1D of length measurement_size \\(1\\), "
                "but got shape \\(2\\)"):
            kf.update(z_wrong)

    def test_update_measurement_2d_error(self):
        n = 2
        x0 = mm.SimpleArrayFloat64([n])
        f = mm.SimpleArrayFloat64([n, n])
        h = mm.SimpleArrayFloat64([1, n])

        kf = mm.KalmanFilterFp64(
            x=x0, f=f, h=h,
            process_noise=0.1, measurement_noise=1.0)

        # Test 2D measurement (should be 1D)
        z_2d = mm.SimpleArrayFloat64([1, 1])  # 2D array

        with self.assertRaisesRegex(
                ValueError,
                "KalmanFilter::check_measurement: The measurement "
                "SimpleArray z must be 1D of length measurement_size \\(1\\), "
                "but got shape \\(1, 1\\)"):
            kf.update(z_2d)


class KalmanFilterControlTC(unittest.TestCase):

    def kf_control_predict_numpy(self, x, p, f, q, b, u):
        x_pred = f @ x + b @ u
        p_pred = f @ p @ f.conj().T + q
        return x_pred, p_pred

    def test_control_predict_fp64(self):
        n, k = 3, 1
        x0 = np.array([0.2, -0.1, 0.3])
        p0 = np.eye(n)
        f = np.array([[1.0, 0.1, 0.0],
                      [0.0, 1.0, 0.2],
                      [0.0, 0.0, 1.0]])
        b = np.array([[0.5, 0.0],
                      [0.1, 0.3],
                      [0.0, 0.2]])
        u = np.array([1.0, -2.0])
        h = np.zeros((k, n))
        h[0, 0] = 1.0
        sigma_w = 0.05
        q = (sigma_w**2) * np.eye(n)

        x_pred_np, p_pred_np = self.kf_control_predict_numpy(
            x0, p0, f, q, b, u
        )

        x_sa = sa_from_np(x0, mm.SimpleArrayFloat64)
        f_sa = sa_from_np(f, mm.SimpleArrayFloat64)
        b_sa = sa_from_np(b, mm.SimpleArrayFloat64)
        h_sa = sa_from_np(h, mm.SimpleArrayFloat64)
        u_sa = sa_from_np(u, mm.SimpleArrayFloat64)

        kf = mm.KalmanFilterFp64(
            x=x_sa, f=f_sa, b=b_sa, h=h_sa,
            process_noise=sigma_w, measurement_noise=1.0
        )
        kf.predict(u_sa)

        x_pred_mm = kf.state.ndarray
        p_pred_mm = kf.covariance.ndarray

        np.testing.assert_allclose(x_pred_mm, x_pred_np, atol=1e-12, rtol=0.0)
        np.testing.assert_allclose(p_pred_mm, p_pred_np, atol=1e-12, rtol=0.0)

    def test_simple_predict_with_control(self):
        n, k = 2, 1
        x0 = np.array([0.0, 0.0])
        f = np.array([[1.0, 0.1],
                      [0.0, 1.0]])
        h = np.zeros((k, n))
        h[0, 0] = 1.0
        u = np.array([1.0])

        x_sa = sa_from_np(x0, mm.SimpleArrayFloat64)
        f_sa = sa_from_np(f, mm.SimpleArrayFloat64)
        h_sa = sa_from_np(h, mm.SimpleArrayFloat64)
        u_sa = sa_from_np(u, mm.SimpleArrayFloat64)

        kf = mm.KalmanFilterFp64(
            x=x_sa, f=f_sa, h=h_sa,
            process_noise=0.0, measurement_noise=1.0
        )
        with self.assertRaisesRegex(
                ValueError,
                "KalmanFilter::check_control: Control input not "
                "supported: control_size is 0"):
            kf.predict(u_sa)

    def test_wrong_shape_control_predict(self):
        n, k = 2, 1
        x0 = np.array([0.0, 0.0])
        f = np.array([[1.0, 0.1],
                      [0.0, 1.0]])
        b = np.array([[0.5],
                      [0.1]])
        h = np.zeros((k, n))
        h[0, 0] = 1.0
        u_wrong = np.array([1.0, 2.0])

        x_sa = sa_from_np(x0, mm.SimpleArrayFloat64)
        f_sa = sa_from_np(f, mm.SimpleArrayFloat64)
        b_sa = sa_from_np(b, mm.SimpleArrayFloat64)
        h_sa = sa_from_np(h, mm.SimpleArrayFloat64)
        u_wrong_sa = sa_from_np(u_wrong, mm.SimpleArrayFloat64)

        kf = mm.KalmanFilterFp64(
            x=x_sa, f=f_sa, b=b_sa, h=h_sa,
            process_noise=0.0, measurement_noise=1.0
        )
        with self.assertRaisesRegex(
                ValueError,
                "KalmanFilter::check_control: The control SimpleArray u "
                "must be 1D of length control_size \\(1\\), but got shape "
                "\\(2\\)"):
            kf.predict(u_wrong_sa)
