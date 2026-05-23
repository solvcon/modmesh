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


class KalmanFilterRadarExampleTC(unittest.TestCase):

    def test_kalmanfilter_net_radar_example(self):
        # Reference: https://kalmanfilter.net/
        dt = 5.0
        x0 = np.array([10000.0, 200.0])
        f = np.array([[1.0, dt],
                      [0.0, 1.0]])
        h = np.eye(2)
        p0 = np.array([[16.0, 0.0],
                       [0.0, 0.25]])
        q = np.array([[6.25, 2.5],
                      [2.5, 1.0]])
        r = np.array([[36.0, 0.0],
                      [0.0, 2.25]])
        z1 = np.array([11020.0, 202.0])

        kf = mm.KalmanFilterFp64(
            x=sa_from_np(x0, mm.SimpleArrayFloat64),
            f=sa_from_np(f, mm.SimpleArrayFloat64),
            h=sa_from_np(h, mm.SimpleArrayFloat64),
            q=sa_from_np(q, mm.SimpleArrayFloat64),
            r=sa_from_np(r, mm.SimpleArrayFloat64),
            p=sa_from_np(p0, mm.SimpleArrayFloat64),
            jitter=0.0,
        )

        kf.predict()

        x_pred_expected = np.array([11000.0, 200.0])
        p_pred_expected = np.array([[28.5, 3.75],
                                    [3.75, 1.25]])

        np.testing.assert_allclose(
            kf.state.ndarray, x_pred_expected, atol=1e-12, rtol=0.0)
        np.testing.assert_allclose(
            kf.covariance.ndarray, p_pred_expected, atol=1e-12, rtol=0.0)

        kf.update(sa_from_np(z1, mm.SimpleArrayFloat64))

        x_update_expected = np.array([
            11009.371124889283,
            201.42604074402126,
        ])
        p_update_expected = np.array([
            [14.572187776793623, 1.4348981399468559],
            [1.4348981399468559, 0.7074844995571303],
        ])

        np.testing.assert_allclose(
            kf.state.ndarray, x_update_expected, atol=1e-12, rtol=0.0)
        np.testing.assert_allclose(
            kf.covariance.ndarray, p_update_expected, atol=1e-12, rtol=0.0)


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


class KalmanFilterBatchFilterTC(unittest.TestCase):

    def kf_batchfilter_numpy(self, kf, zs, us=None):
        m = zs.shape[0]
        n = kf.state.shape[0]
        xs_pred_np = np.zeros((m, n))
        ps_pred_np = np.zeros((m, n, n))
        xs_upd_np = np.zeros((m, n))
        ps_upd_np = np.zeros((m, n, n))
        for i in range(m):
            if us is not None:
                u = us[i]
                u_sa = sa_from_np(u, type(kf.state))
                kf.predict(u_sa)
            else:
                kf.predict()
            x_pred = kf.state.ndarray
            p_pred = kf.covariance.ndarray
            xs_pred_np[i] = x_pred
            ps_pred_np[i] = p_pred

            z = zs[i]
            z_sa = sa_from_np(z, type(kf.state))
            kf.update(z_sa)
            x_upd = kf.state.ndarray
            p_upd = kf.covariance.ndarray
            xs_upd_np[i] = x_upd
            ps_upd_np[i] = p_upd
        return xs_pred_np, ps_pred_np, xs_upd_np, ps_upd_np

    def test_batchfilter(self):
        m = 50
        x0 = np.array([1.0, 2.0, 3.0])
        f = np.array([[1.1, 0.2, 0.3],
                      [0.1, 0.9, 0.7],
                      [4.7, 5.2, 6.7]])
        h = np.array([[1.0, 3.0, 2.0],
                      [4.0, 0.2, 0.1]])
        sigma_w = 0.316
        zs = np.zeros((m, 2))
        for i in range(m):
            zs[i] = np.array([i * i, i * 0.5 + 1.0])
        x_sa = mm.SimpleArrayFloat64(array=x0)
        f_sa = mm.SimpleArrayFloat64(array=f)
        h_sa = mm.SimpleArrayFloat64(array=h)
        zs_sa = mm.SimpleArrayFloat64(array=zs)

        kf = mm.KalmanFilterFp64(
            x=x_sa, f=f_sa, h=h_sa,
            process_noise=sigma_w,
            measurement_noise=1.0,
        )
        bps = kf.batch_filter(zs_sa)
        xs_pred = bps.prior_states
        ps_pred = bps.prior_states_covariance
        xs_upd = bps.posterior_states
        ps_upd = bps.posterior_states_covariance

        kf = mm.KalmanFilterFp64(
            x=x_sa, f=f_sa, h=h_sa,
            process_noise=sigma_w,
            measurement_noise=1.0,
        )
        bps_np = self.kf_batchfilter_numpy(kf, zs)
        xs_pred_np, ps_pred_np, xs_upd_np, ps_upd_np = bps_np

        np.testing.assert_allclose(xs_pred, xs_pred_np, atol=1e-12, rtol=0.0)
        np.testing.assert_allclose(ps_pred, ps_pred_np, atol=1e-12, rtol=0.0)
        np.testing.assert_allclose(xs_upd, xs_upd_np, atol=1e-12, rtol=0.0)
        np.testing.assert_allclose(ps_upd, ps_upd_np, atol=1e-12, rtol=0.0)

    def test_batchfilter_with_control(self):
        m = 50
        x0 = np.array([1.0, 2.0, 3.0])
        f = np.array([[1.1, 0.2, 0.3],
                      [0.1, 0.9, 0.7],
                      [4.7, 5.2, 6.7]])
        h = np.array([[1.0, 3.0, 2.0],
                      [4.0, 0.2, 0.1]])
        b = np.array([[0.7, 0.2, 5.3],
                      [3.1, 0.9, 1.7],
                      [4.7, 5.2, 6.7]])
        sigma_w = 0.316
        zs = np.zeros((m, 2))
        for i in range(m):
            zs[i] = np.array([i * i, i * 0.5 + 1.0])
        us = np.zeros((m, 3))
        for i in range(m):
            us[i] = np.array([i, pow(i, 3.5), pow(i, 0.5)])
        x_sa = mm.SimpleArrayFloat64(array=x0)
        f_sa = mm.SimpleArrayFloat64(array=f)
        b_sa = mm.SimpleArrayFloat64(array=b)
        h_sa = mm.SimpleArrayFloat64(array=h)
        zs_sa = mm.SimpleArrayFloat64(array=zs)
        us_sa = mm.SimpleArrayFloat64(array=us)

        kf = mm.KalmanFilterFp64(
            x=x_sa, f=f_sa, b=b_sa, h=h_sa,
            process_noise=sigma_w,
            measurement_noise=1.0,
        )
        bps = kf.batch_filter(zs_sa, us_sa)
        xs_pred = bps.prior_states
        ps_pred = bps.prior_states_covariance
        xs_upd = bps.posterior_states
        ps_upd = bps.posterior_states_covariance

        kf = mm.KalmanFilterFp64(
            x=x_sa, f=f_sa, b=b_sa, h=h_sa,
            process_noise=sigma_w,
            measurement_noise=1.0,
        )
        bps_np = self.kf_batchfilter_numpy(kf, zs, us)
        xs_pred_np, ps_pred_np, xs_upd_np, ps_upd_np = bps_np

        np.testing.assert_allclose(xs_pred, xs_pred_np, atol=1e-12, rtol=0.0)
        np.testing.assert_allclose(ps_pred, ps_pred_np, atol=1e-12, rtol=0.0)
        np.testing.assert_allclose(xs_upd, xs_upd_np, atol=1e-12, rtol=0.0)
        np.testing.assert_allclose(ps_upd, ps_upd_np, atol=1e-12, rtol=0.0)


def _assert_PA_equals_LU(A_np, lu_np, piv, rtol, atol):
    """Assert PA == L @ U for Lu::factorize output.

    Lu::factorize stores L in the strictly lower triangle (unit diagonal)
    and U in the upper triangle (including diagonal) of a single array.
    piv[k] is the row swapped with row k at step k; swaps are replayed
    sequentially from k = 0 to k = n - 1.
    """
    n = A_np.shape[0]
    L = np.eye(n, dtype=lu_np.dtype)
    U = np.zeros_like(lu_np)
    for i in range(n):
        for j in range(n):
            if i > j:
                L[i, j] = lu_np[i, j]
            else:
                U[i, j] = lu_np[i, j]
    PA = A_np.astype(lu_np.dtype, copy=True)
    for k in range(n):
        if piv[k] != k:
            PA[[k, piv[k]]] = PA[[piv[k], k]]
    np.testing.assert_allclose(L @ U, PA, rtol=rtol, atol=atol)


class TestLuFactorization(unittest.TestCase):
    """Verify lu_factorization() produces a decomposition with PA = LU."""

    def setUp(self):
        # 3x3 invertible matrix with no zero pivots.
        self.A_3x3 = np.array([
            [2.0, 1.0, 1.0],
            [4.0, -6.0, 0.0],
            [-2.0, 7.0, 2.0],
        ], dtype="float64")
        # 4x4 whose leading element is not the column-max, so partial
        # pivoting must trigger.
        self.A_4x4 = np.array([
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 8.0, 8.0],
            [9.0, 10.0, 11.0, 16.0],
            [13.0, 15.0, 16.0, 17.0],
        ], dtype="float64")
        # 3x3 whose A[0][0] is negligibly small; a correct implementation
        # must pivot away from it.
        self.A_tiny_pivot = np.array([
            [1e-15, 1.0, 0.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 2.0],
        ], dtype="float64")
        # 3x3 complex matrix to exercise Lu<T> instantiated for
        # Complex<double>.
        self.A_complex_3x3 = np.array([
            [2.0 + 1.0j, 1.0 + 0.0j, 3.0 - 1.0j],
            [4.0 + 0.0j, -1.0 + 2.0j, 1.0 + 1.0j],
            [1.0 - 1.0j, 5.0 + 0.0j, 2.0 + 3.0j],
        ], dtype="complex128")

    def test_factorize_3x3_reconstructs_PA(self):
        # Baseline: well-conditioned 3x3, also checks output shape/piv length.
        A = mm.SimpleArrayFloat64(array=self.A_3x3)
        lu, piv = mm.lu_factorization(A)
        lu_np = np.array(lu)
        self.assertEqual(lu_np.shape, (3, 3))
        self.assertEqual(len(piv), 3)
        _assert_PA_equals_LU(self.A_3x3, lu_np, piv, rtol=1e-12, atol=1e-14)

    def test_factorize_4x4_reconstructs_PA(self):
        # Larger 4x4 where partial pivoting must trigger on the first column.
        A = mm.SimpleArrayFloat64(array=self.A_4x4)
        lu, piv = mm.lu_factorization(A)
        lu_np = np.array(lu)
        _assert_PA_equals_LU(self.A_4x4, lu_np, piv, rtol=1e-12, atol=1e-12)

    def test_factorize_complex_3x3_reconstructs_PA(self):
        # Complex128 path: exercises the complex template instantiation.
        A = mm.SimpleArrayComplex128(array=self.A_complex_3x3)
        lu, piv = mm.lu_factorization(A)
        lu_np = np.array(lu)
        _assert_PA_equals_LU(
            self.A_complex_3x3, lu_np, piv, rtol=1e-12, atol=1e-12)

    def test_factorize_pivots_away_from_tiny_diagonal(self):
        # Numerical stability: A[0][0] ~ 1e-15 forces a row swap at step 0.
        A = mm.SimpleArrayFloat64(array=self.A_tiny_pivot)
        lu, piv = mm.lu_factorization(A)
        # With A[0][0] ~ 1e-15 and A[1][0] = 1.0, partial pivoting must
        # swap row 0 with a later row.
        self.assertNotEqual(piv[0], 0)
        lu_np = np.array(lu)
        _assert_PA_equals_LU(
            self.A_tiny_pivot, lu_np, piv, rtol=1e-10, atol=1e-10)

    def test_factorize_float32_reconstructs_PA(self):
        # Float32 path: same input as 3x3 case but in single precision.
        A_np = self.A_3x3.astype(np.float32)
        A = mm.SimpleArrayFloat32(array=A_np)
        lu, piv = mm.lu_factorization(A)
        lu_np = np.array(lu)
        _assert_PA_equals_LU(A_np, lu_np, piv, rtol=1e-5, atol=1e-5)


class TestLuSolve(unittest.TestCase):
    """Verify lu_solve() against explicit systems with known solutions."""

    def setUp(self):
        # 2x2: [[2, 3], [4, 7]] x = [5, 11]  =>  x = [1, 1]
        self.A_2x2 = np.array([[2.0, 3.0], [4.0, 7.0]], dtype="float64")
        self.b_2x2 = np.array([5.0, 11.0], dtype="float64")
        self.x_2x2_expected = np.array([1.0, 1.0], dtype="float64")

        # 3x3 system with known exact solution x = [1, 1, 2].
        self.A_3x3 = np.array([
            [2.0, 1.0, 1.0],
            [4.0, -6.0, 0.0],
            [-2.0, 7.0, 2.0],
        ], dtype="float64")
        self.b_3x3 = np.array([5.0, -2.0, 9.0], dtype="float64")
        self.x_3x3_expected = np.array([1.0, 1.0, 2.0], dtype="float64")

        # 2x2 complex: (2 + j)x + (1 - j)y = 4 + 3j, jx + 3y = 6 + j
        # Chosen so x = [1 + j, 2] is exact.
        self.A_c2x2 = np.array([
            [2.0 + 1.0j, 1.0 - 1.0j],
            [0.0 + 1.0j, 3.0 + 0.0j],
        ], dtype="complex128")
        self.x_c2x2_expected = np.array([1.0 + 1.0j, 2.0], dtype="complex128")
        self.b_c2x2 = self.A_c2x2 @ self.x_c2x2_expected

    def test_solve_2x2_matches_known_solution(self):
        # Smallest case: 2x2 with 1D rhs, also checks output shape.
        A = mm.SimpleArrayFloat64(array=self.A_2x2)
        b = mm.SimpleArrayFloat64(array=self.b_2x2)
        x = mm.lu_solve(A, b)
        self.assertEqual(x.shape, (2,))
        np.testing.assert_allclose(
            np.array(x), self.x_2x2_expected, rtol=1e-12, atol=1e-14)

    def test_solve_3x3_matches_known_solution(self):
        # 3x3 with 1D rhs against a precomputed exact solution.
        A = mm.SimpleArrayFloat64(array=self.A_3x3)
        b = mm.SimpleArrayFloat64(array=self.b_3x3)
        x = mm.lu_solve(A, b)
        self.assertEqual(x.shape, (3,))
        np.testing.assert_allclose(
            np.array(x), self.x_3x3_expected, rtol=1e-12, atol=1e-14)

    def test_solve_float32(self):
        # Float32 path: same 3x3 system in single precision.
        A = mm.SimpleArrayFloat32(array=self.A_3x3.astype(np.float32))
        b = mm.SimpleArrayFloat32(array=self.b_3x3.astype(np.float32))
        x = mm.lu_solve(A, b)
        np.testing.assert_allclose(
            np.array(x), self.x_3x3_expected.astype(np.float32),
            rtol=1e-5, atol=1e-5)

    def test_solve_complex128_matches_known_solution(self):
        # Complex128 path with a known exact complex solution.
        A = mm.SimpleArrayComplex128(array=self.A_c2x2)
        b = mm.SimpleArrayComplex128(array=self.b_c2x2)
        x = mm.lu_solve(A, b)
        np.testing.assert_allclose(
            np.array(x), self.x_c2x2_expected, rtol=1e-12, atol=1e-14)

    def test_solve_complex64(self):
        # Complex64 path: same complex system in single precision.
        A_np = self.A_c2x2.astype(np.complex64)
        b_np = self.b_c2x2.astype(np.complex64)
        A = mm.SimpleArrayComplex64(array=A_np)
        b = mm.SimpleArrayComplex64(array=b_np)
        x = mm.lu_solve(A, b)
        np.testing.assert_allclose(
            np.array(x), self.x_c2x2_expected.astype(np.complex64),
            rtol=1e-5, atol=1e-5)

    def test_solve_2d_multi_rhs(self):
        # 2D rhs: each column of B yields an independent solution column in X.
        B_np = np.array([
            [5.0, 4.0, 6.0],
            [-2.0, -10.0, -8.0],
            [9.0, 3.0, 15.0],
        ], dtype="float64")
        A = mm.SimpleArrayFloat64(array=self.A_3x3)
        B = mm.SimpleArrayFloat64(array=B_np)
        X = mm.lu_solve(A, B)
        self.assertEqual(X.shape, (3, 3))
        # Verify A @ X == B column-by-column.
        np.testing.assert_allclose(
            self.A_3x3 @ np.array(X), B_np, rtol=1e-12, atol=1e-12)

    def test_solve_complex_multi_rhs(self):
        # 2D complex rhs: exercises multi-column solve on the complex path.
        B_np = np.column_stack([
            self.b_c2x2,
            self.b_c2x2 * (1.0 + 0.5j),
            self.b_c2x2.conj(),
        ])
        A = mm.SimpleArrayComplex128(array=self.A_c2x2)
        B = mm.SimpleArrayComplex128(array=B_np)
        X = mm.lu_solve(A, B)
        self.assertEqual(X.shape, (2, 3))
        np.testing.assert_allclose(
            self.A_c2x2 @ np.array(X), B_np, rtol=1e-12, atol=1e-12)


class TestLuInv(unittest.TestCase):
    """Verify lu_inv() against matrices with known inverses."""

    def setUp(self):
        # A = [[4, 7], [2, 6]]; det = 10; A^-1 = [[0.6, -0.7], [-0.2, 0.4]].
        self.A_2x2 = np.array([[4.0, 7.0], [2.0, 6.0]], dtype="float64")
        self.A_2x2_inv_expected = np.array(
            [[0.6, -0.7], [-0.2, 0.4]], dtype="float64")
        # A 3x3 diagonal matrix whose inverse is 1/diag.
        self.A_diag = np.diag(np.array([2.0, 4.0, 5.0], dtype="float64"))
        self.A_diag_inv_expected = np.diag(
            np.array([0.5, 0.25, 0.2], dtype="float64"))

    def test_inv_2x2_matches_known_inverse(self):
        # 2x2 against a hand-computed inverse, also checks output shape.
        A = mm.SimpleArrayFloat64(array=self.A_2x2)
        A_inv = mm.lu_inv(A)
        self.assertEqual(A_inv.shape, (2, 2))
        np.testing.assert_allclose(
            np.array(A_inv), self.A_2x2_inv_expected,
            rtol=1e-12, atol=1e-14)

    def test_inv_diagonal_matches_elementwise_reciprocal(self):
        # Diagonal matrix: inverse must be elementwise 1/diag.
        A = mm.SimpleArrayFloat64(array=self.A_diag)
        A_inv = np.array(mm.lu_inv(A))
        np.testing.assert_allclose(
            A_inv, self.A_diag_inv_expected, rtol=1e-12, atol=1e-14)

    def test_inv_of_identity_is_identity(self):
        # Identity edge case: inv(I) must be I.
        identity = np.eye(4)
        A = mm.SimpleArrayFloat64(array=identity)
        A_inv = np.array(mm.lu_inv(A))
        np.testing.assert_allclose(A_inv, identity, rtol=1e-12, atol=1e-14)

    def test_inv_times_A_equals_identity(self):
        # Round-trip identity: checks A @ A_inv and A_inv @ A both equal I.
        A_np = np.array([
            [2.0, 1.0, 1.0],
            [4.0, -6.0, 0.0],
            [-2.0, 7.0, 2.0],
        ], dtype="float64")
        A = mm.SimpleArrayFloat64(array=A_np)
        A_inv = np.array(mm.lu_inv(A))
        np.testing.assert_allclose(
            A_np @ A_inv, np.eye(3), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(
            A_inv @ A_np, np.eye(3), rtol=1e-12, atol=1e-12)

    def test_inv_complex(self):
        # Complex128 path: verifies A @ A_inv == I for a complex matrix.
        A_np = np.array([
            [2.0 + 1.0j, 1.0 - 1.0j],
            [0.0 + 1.0j, 3.0 + 0.0j],
        ], dtype="complex128")
        A = mm.SimpleArrayComplex128(array=A_np)
        A_inv = np.array(mm.lu_inv(A))
        np.testing.assert_allclose(
            A_np @ A_inv, np.eye(2, dtype="complex128"),
            rtol=1e-12, atol=1e-12)


class TestLuSimpleArrayMethods(unittest.TestCase):
    """Verify .solve()/.inv() work for all floating and complex types and
    are absent on integer types."""

    # (SimpleArray class, numpy dtype, tolerance)
    _FLOAT_CASES = [
        (mm.SimpleArrayFloat32, np.float32, 1e-5),
        (mm.SimpleArrayFloat64, np.float64, 1e-12),
        (mm.SimpleArrayComplex64, np.complex64, 1e-5),
        (mm.SimpleArrayComplex128, np.complex128, 1e-12),
    ]

    @staticmethod
    def _build_system(np_dtype):
        # A = [[4, 7], [2, 6]]; det = 10.  b = [11, 8]  =>  x = [1, 1].
        A = np.array([[4.0, 7.0], [2.0, 6.0]], dtype=np_dtype)
        b = np.array([11.0, 8.0], dtype=np_dtype)
        return A, b

    def test_solve_method_matches_free_function_for_all_dtypes(self):
        # A.solve(b) must match mm.lu_solve(A, b) across float/complex dtypes.
        for sa_cls, np_dtype, tol in self._FLOAT_CASES:
            with self.subTest(cls=sa_cls.__name__):
                A_np, b_np = self._build_system(np_dtype)
                A = sa_cls(array=A_np)
                b = sa_cls(array=b_np)
                x_free = np.array(mm.lu_solve(A, b))
                x_method = np.array(A.solve(b))
                np.testing.assert_allclose(
                    x_method, x_free, rtol=tol, atol=tol)
                np.testing.assert_allclose(
                    A_np @ x_method, b_np, rtol=tol, atol=tol)

    def test_inv_method_matches_free_function_for_all_dtypes(self):
        # A.inv() must match mm.lu_inv(A) across all float/complex dtypes.
        for sa_cls, np_dtype, tol in self._FLOAT_CASES:
            with self.subTest(cls=sa_cls.__name__):
                A_np, _ = self._build_system(np_dtype)
                A = sa_cls(array=A_np)
                inv_free = np.array(mm.lu_inv(A))
                inv_method = np.array(A.inv())
                np.testing.assert_allclose(
                    inv_method, inv_free, rtol=tol, atol=tol)
                np.testing.assert_allclose(
                    A_np @ inv_method, np.eye(2, dtype=np_dtype),
                    rtol=tol, atol=tol)

    def test_solve_inv_absent_on_integer_types(self):
        # Integer/bool SimpleArrays must not expose .solve()/.inv().
        int_classes = (
            mm.SimpleArrayBool,
            mm.SimpleArrayInt8, mm.SimpleArrayInt16,
            mm.SimpleArrayInt32, mm.SimpleArrayInt64,
            mm.SimpleArrayUint8, mm.SimpleArrayUint16,
            mm.SimpleArrayUint32, mm.SimpleArrayUint64,
        )
        for cls in int_classes:
            with self.subTest(cls=cls.__name__):
                A = cls([2, 2])
                self.assertFalse(hasattr(A, 'solve'))
                self.assertFalse(hasattr(A, 'inv'))


class TestLuErrorHandling(unittest.TestCase):
    """Verify LU routines reject invalid inputs with clear errors."""

    def test_factorize_rejects_non_square(self):
        # Rectangular 2D input must raise a clear shape error.
        A_rect = mm.SimpleArrayFloat64(array=np.array(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype="float64"))
        with self.assertRaisesRegex(
                ValueError, r"must be a square 2D SimpleArray"):
            mm.lu_factorization(A_rect)

    def test_factorize_rejects_1d_input(self):
        # 1D input must raise the same shape error (not silently reshape).
        A_1d = mm.SimpleArrayFloat64(array=np.array(
            [1.0, 2.0, 3.0], dtype="float64"))
        with self.assertRaisesRegex(
                ValueError, r"must be a square 2D SimpleArray"):
            mm.lu_factorization(A_1d)

    def test_factorize_rejects_singular_duplicate_row(self):
        # Singular matrix (duplicate row) must raise the singular error.
        A_sing = np.array([
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ], dtype="float64")
        A = mm.SimpleArrayFloat64(array=A_sing)
        with self.assertRaisesRegex(
                RuntimeError, r"singular or near-singular"):
            mm.lu_factorization(A)

    def test_factorize_rejects_near_singular_tiny_eigenvalue(self):
        # A = v v^T + eps * I for v = [1, 1] has eigenvalues {2 + eps, eps}.
        # With eps = 1e-15 the smaller eigenvalue is tiny but nonzero, unlike
        # the duplicate-row test above where an eigenvalue is exactly zero.
        eps = 1e-15
        A_near_sing = np.array([
            [1.0 + eps, 1.0],
            [1.0, 1.0 + eps],
        ], dtype="float64")
        A = mm.SimpleArrayFloat64(array=A_near_sing)
        with self.assertRaisesRegex(
                RuntimeError, r"singular or near-singular"):
            mm.lu_factorization(A)

    def test_solve_rejects_non_square_A(self):
        # lu_solve must reject rectangular A with the same shape error.
        A_rect = mm.SimpleArrayFloat64(array=np.array(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype="float64"))
        b = mm.SimpleArrayFloat64(array=np.array(
            [1.0, 2.0], dtype="float64"))
        with self.assertRaisesRegex(
                ValueError, r"must be a square 2D SimpleArray"):
            mm.lu_solve(A_rect, b)

    def test_solve_rejects_1d_dimension_mismatch(self):
        # 1D rhs length must match A's dimension; otherwise raise.
        A = mm.SimpleArrayFloat64(array=np.array(
            [[1.0, 2.0], [3.0, 4.0]], dtype="float64"))
        b_wrong = mm.SimpleArrayFloat64(array=np.array(
            [1.0, 2.0, 3.0], dtype="float64"))
        with self.assertRaisesRegex(ValueError, r"dimension mismatch"):
            mm.lu_solve(A, b_wrong)

    def test_solve_rejects_2d_dimension_mismatch(self):
        # 2D rhs row-count must match A's dimension; otherwise raise.
        A = mm.SimpleArrayFloat64(array=np.array(
            [[1.0, 2.0], [3.0, 4.0]], dtype="float64"))
        B_wrong = mm.SimpleArrayFloat64(array=np.array([
            [1.0, 2.0], [3.0, 4.0], [5.0, 6.0],
        ], dtype="float64"))
        with self.assertRaisesRegex(ValueError, r"dimension mismatch"):
            mm.lu_solve(A, B_wrong)

    def test_solve_rejects_3d_rhs(self):
        # rhs must be 1D or 2D; 3D input must raise.
        A = mm.SimpleArrayFloat64(array=np.array(
            [[1.0, 2.0], [3.0, 4.0]], dtype="float64"))
        b_3d = mm.SimpleArrayFloat64(array=np.array(
            [[[1.0], [2.0]], [[3.0], [4.0]]], dtype="float64"))
        with self.assertRaisesRegex(ValueError, r"b must be 1D or 2D"):
            mm.lu_solve(A, b_3d)

    def test_solve_rejects_singular(self):
        # Singular A must cause lu_solve to raise rather than return garbage.
        A_sing = np.array([
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ], dtype="float64")
        A = mm.SimpleArrayFloat64(array=A_sing)
        b = mm.SimpleArrayFloat64(array=np.array(
            [1.0, 2.0, 3.0], dtype="float64"))
        with self.assertRaises(RuntimeError):
            mm.lu_solve(A, b)

    def test_inv_rejects_non_square(self):
        # lu_inv must reject rectangular input with the same shape error.
        A_rect = mm.SimpleArrayFloat64(array=np.array(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype="float64"))
        with self.assertRaisesRegex(
                ValueError, r"must be a square 2D SimpleArray"):
            mm.lu_inv(A_rect)

    def test_inv_rejects_singular(self):
        # Singular A must cause lu_inv to raise rather than return garbage.
        A_sing = np.array([
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ], dtype="float64")
        A = mm.SimpleArrayFloat64(array=A_sing)
        with self.assertRaises(RuntimeError):
            mm.lu_inv(A)


@unittest.skipIf(mm.EigenSystem is None,
                 "mm.EigenSystem is not built (no vendor LAPACK)")
class TestLinalgEigenSystemTC(unittest.TestCase):
    """Verify EigenSystem against DGEEV reference outputs.

    EigenSystem(A).run() populates wr/wi (real and imaginary parts of the
    eigenvalues) and vl/vr (left/right eigenvector matrices) in DGEEV's
    column-major layout (j-th column == j-th eigenvector).  For a complex
    conjugate pair (wi[j] > 0, wi[j+1] = -wi[j]) the j-th and (j+1)-th
    columns hold the real and imaginary parts of the eigenvector; the
    (j+1)-th eigenvector is the complex conjugate of the j-th.
    """

    def _solve(self, A_np):
        A = mm.SimpleArrayFloat64(array=A_np)
        solver = mm.EigenSystem(A)
        self.assertFalse(solver.done)
        solver.run()
        self.assertTrue(solver.done)
        return (np.array(solver.wr), np.array(solver.wi),
                np.array(solver.vl), np.array(solver.vr))

    def test_diagonal_eigenvalues_and_basis_eigenvectors(self):
        # diag(1, 2, 3, 4): eigenvalues are the diagonal; eigenvectors are
        # standard basis vectors up to permutation/sign.
        A_np = np.diag([1.0, 2.0, 3.0, 4.0])
        wr, wi, _vl, vr = self._solve(A_np)
        np.testing.assert_allclose(np.sort(wr), [1.0, 2.0, 3.0, 4.0],
                                   rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(wi, np.zeros(4), atol=1e-14)
        # Each column must be a (signed) standard basis vector.
        for j in range(4):
            col = vr[:, j]
            np.testing.assert_allclose(np.abs(col).sum(), 1.0,
                                       rtol=1e-12, atol=1e-14)
            self.assertEqual(np.count_nonzero(np.abs(col) > 1e-12), 1)

    def test_symmetric_known_spectrum_reconstructs_eigenpairs(self):
        # The fixture Q below is frozen (not generated at runtime) so the test
        # is independent of the NumPy RNG / QR implementation.  To regenerate,
        # run:
        #   rng = np.random.default_rng(seed=20260509)
        #   Q, _ = np.linalg.qr(rng.standard_normal((5, 5)))
        # and paste Q.tolist() back in.
        n = 5
        lam = np.array([-2.0, -0.5, 1.0, 3.5, 7.0], dtype='float64')
        Q = np.array([
            [-0.024712873175908756, 0.6640676133514902,
             0.5915632257468302, 0.4455230309449579,
             -0.09982814051493692],
            [-0.28147871622325943, -0.5692008616441233,
             0.5886397812806473, 0.1575982019764191,
             0.4748116742926241],
            [-0.9139918340187755, 0.2754225371185904,
             -0.14764410500783806, -0.2527961990356955,
             0.05528700935747367],
            [0.24639362994675912, 0.23505747176988095,
             0.4022089819816873, -0.825230080044468,
             0.2031290572268447],
            [0.15513901083013232, 0.32235848835350017,
             -0.34638895898623234, 0.17821737394071255,
             0.8486873093331871],
        ])
        # Check the fixture is orthogonal first.
        np.testing.assert_allclose(Q @ Q.T, np.eye(n), atol=1e-14)
        A_np = Q @ np.diag(lam) @ Q.T
        wr, wi, _vl, vr = self._solve(A_np)
        np.testing.assert_allclose(np.sort(wr), np.sort(lam),
                                   rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(wi, np.zeros(n), atol=1e-12)
        # Per-column eigenpair: A v_j = lambda_j v_j.
        for j in range(n):
            np.testing.assert_allclose(A_np @ vr[:, j], wr[j] * vr[:, j],
                                       rtol=1e-10, atol=1e-12)

    def test_2x2_rotation_complex_conjugate_pair(self):
        # 2x2 90-degree rotation matrix has eigenvalues +/- i.  This fixture
        # exercises DGEEV's packing of complex conjugate eigenvectors into
        # consecutive real columns.
        A_np = np.array([[0.0, -1.0], [1.0, 0.0]], dtype='float64')
        wr, wi, _vl, vr = self._solve(A_np)
        np.testing.assert_allclose(wr, np.zeros(2, dtype='float64'),
                                   atol=1e-14)
        np.testing.assert_allclose(np.sort(wi), [-1.0, 1.0], atol=1e-14)
        j = int(np.argmax(wi))  # column with positive imaginary part
        self.assertGreater(wi[j], 0.0)
        self.assertAlmostEqual(wi[j] + wi[j + 1], 0.0, places=14)
        v = vr[:, j] + 1j * vr[:, j + 1]
        lam = wr[j] + 1j * wi[j]
        np.testing.assert_allclose(A_np @ v, lam * v, rtol=1e-12, atol=1e-14)
        # The (j+1)-th eigenvector is the conjugate of the j-th and corresponds
        # to the conjugate eigenvalue.
        v_conj = vr[:, j] - 1j * vr[:, j + 1]
        lam_conj = wr[j] - 1j * wi[j]
        np.testing.assert_allclose(A_np @ v_conj, lam_conj * v_conj,
                                   rtol=1e-12, atol=1e-14)

    def test_left_eigenvectors_satisfy_left_equation(self):
        # Left eigenvectors u_j (column vector) satisfy u_j^T A = lambda_j
        # u_j^T (i.e., A^T u_j = lambda_j u_j) for real eigenvalues (general
        # case is A^H u_j = conj(lambda_j) u_j).  Use a non-symmetric matrix
        # with all-real eigenvalues: upper-triangular -> spectrum is the
        # diagonal.
        A_np = np.array([
            [2.0, 1.0, 0.5],
            [0.0, 3.0, -1.0],
            [0.0, 0.0, 5.0],
        ], dtype='float64')
        wr, wi, vl, _vr = self._solve(A_np)
        np.testing.assert_allclose(np.sort(wr), [2.0, 3.0, 5.0],
                                   rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(wi, np.zeros(3, dtype='float64'),
                                   atol=1e-12)
        for j in range(3):
            np.testing.assert_allclose(A_np.T @ vl[:, j],
                                       wr[j] * vl[:, j],
                                       rtol=1e-10, atol=1e-12)

    def test_rejects_non_square_and_non_2d_inputs(self):
        # Non-square 2D, 1D, and 3D inputs must all raise the same shape
        # error from the EigenSystem constructor.
        A_rect = mm.SimpleArrayFloat64(array=np.array(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype="float64"))
        with self.assertRaisesRegex(
                ValueError, r"must be a square 2D SimpleArray"):
            mm.EigenSystem(A_rect)

        A_1d = mm.SimpleArrayFloat64(array=np.array(
            [1.0, 2.0, 3.0], dtype="float64"))
        with self.assertRaisesRegex(
                ValueError, r"must be a square 2D SimpleArray"):
            mm.EigenSystem(A_1d)

        A_3d = mm.SimpleArrayFloat64(array=np.zeros((2, 2, 2),
                                                    dtype="float64"))
        with self.assertRaisesRegex(
                ValueError, r"must be a square 2D SimpleArray"):
            mm.EigenSystem(A_3d)

    def test_accessors_before_run_are_inert(self):
        # Construct but do not call run(); done must be False and wr must be
        # finite-valued (DGEEV hasn't written), documenting that run() is
        # required before reading results.
        A_np = np.diag([1.0, 2.0])
        A = mm.SimpleArrayFloat64(array=A_np)
        solver = mm.EigenSystem(A)
        self.assertFalse(solver.done)
        wr = solver.wr.ndarray
        self.assertTrue(np.all(np.isfinite(wr)))

    def test_matrix_property_survives_input_gc(self):
        # Construct solver, then drop the Python-side reference to A.
        # solver.matrix must still equal the original array, confirming that
        # keep_alive<1, 2>() on the constructor prevents the C++ m_matrix
        # reference from dangling.
        A_np = np.array([[3.0, 1.0], [0.0, 2.0]])
        A = mm.SimpleArrayFloat64(array=A_np)
        solver = mm.EigenSystem(A)
        del A
        np.testing.assert_array_equal(solver.matrix.ndarray, A_np)

    def test_default_constructor_flags_both_true(self):
        # Guard against an accidental default-flag flip.
        A = mm.SimpleArrayFloat64(array=np.diag([1.0, 2.0]))
        solver = mm.EigenSystem(A)
        self.assertTrue(solver.do_vl)
        self.assertTrue(solver.do_vr)

    def test_skip_vr_does_not_compute_right_eigenvectors(self):
        # do_vr=False must keep wr/wi/vl intact and reject both vr accessors.
        A_np = np.array([[2.0, 0.0], [0.0, 3.0]], dtype="float64")
        A = mm.SimpleArrayFloat64(array=A_np)
        solver = mm.EigenSystem(A, do_vr=False)
        solver.run()
        self.assertTrue(solver.done)
        self.assertTrue(solver.do_vl)
        self.assertFalse(solver.do_vr)
        np.testing.assert_allclose(np.sort(np.array(solver.wr)),
                                   [2.0, 3.0], atol=1e-14)
        self.assertEqual(np.array(solver.vl).shape, (2, 2))
        self.assertEqual(np.array(solver.get_vl()).shape, (2, 2))
        with self.assertRaisesRegex(
                RuntimeError, r"right eigenvectors were not computed"):
            solver.vr  # noqa: B018
        with self.assertRaisesRegex(
                RuntimeError, r"right eigenvectors were not computed"):
            solver.get_vr()

    def test_skip_vl_does_not_compute_left_eigenvectors(self):
        # Mirror of test_skip_vr_*; checks A v = lambda v stays exact.
        A_np = np.array([[2.0, 1.0], [0.0, 3.0]], dtype="float64")
        A = mm.SimpleArrayFloat64(array=A_np)
        solver = mm.EigenSystem(A, do_vl=False)
        solver.run()
        self.assertTrue(solver.done)
        self.assertFalse(solver.do_vl)
        self.assertTrue(solver.do_vr)
        wr = np.array(solver.wr)
        vr = np.array(solver.vr)
        for j in range(2):
            np.testing.assert_allclose(A_np @ vr[:, j], wr[j] * vr[:, j],
                                       rtol=1e-12, atol=1e-12)
        with self.assertRaisesRegex(
                RuntimeError, r"left eigenvectors were not computed"):
            solver.vl  # noqa: B018
        with self.assertRaisesRegex(
                RuntimeError, r"left eigenvectors were not computed"):
            solver.get_vl()

    def test_skip_both_computes_only_eigenvalues(self):
        # Exercises the 3*n workspace path (both jobvl=jobvr='N').
        A_np = np.diag([1.0, 5.0, 9.0])
        A = mm.SimpleArrayFloat64(array=A_np)
        solver = mm.EigenSystem(A, do_vl=False, do_vr=False)
        solver.run()
        np.testing.assert_allclose(np.sort(np.array(solver.wr)),
                                   [1.0, 5.0, 9.0], atol=1e-14)
        with self.assertRaises(RuntimeError):
            solver.vl  # noqa: B018
        with self.assertRaises(RuntimeError):
            solver.vr  # noqa: B018
        with self.assertRaises(RuntimeError):
            solver.get_vl()
        with self.assertRaises(RuntimeError):
            solver.get_vr()

    def test_get_methods_match_property_when_computed(self):
        # Property and get_v* must alias the same matrix; suppress flag
        # is a no-op when the matrix exists.
        A_np = np.diag([2.0, 4.0])
        A = mm.SimpleArrayFloat64(array=A_np)
        solver = mm.EigenSystem(A)
        solver.run()
        np.testing.assert_array_equal(np.array(solver.vl),
                                      np.array(solver.get_vl()))
        np.testing.assert_array_equal(np.array(solver.vr),
                                      np.array(solver.get_vr()))
        np.testing.assert_array_equal(
            np.array(solver.vl),
            np.array(solver.get_vl(suppress_exception=True)))
        np.testing.assert_array_equal(
            np.array(solver.vr),
            np.array(solver.get_vr(suppress_exception=True)))

    def test_get_methods_default_argument_raises(self):
        # Only suppress_exception=True silences the exception.
        A = mm.SimpleArrayFloat64(array=np.diag([1.0, 2.0]))
        solver = mm.EigenSystem(A, do_vl=False, do_vr=False)
        solver.run()
        with self.assertRaises(RuntimeError):
            solver.get_vl()
        with self.assertRaises(RuntimeError):
            solver.get_vr()
        with self.assertRaises(RuntimeError):
            solver.get_vl(suppress_exception=False)
        with self.assertRaises(RuntimeError):
            solver.get_vr(suppress_exception=False)

    def test_suppress_exception_returns_empty_when_not_computed(self):
        # suppress_exception=True returns an empty placeholder, not data.
        A = mm.SimpleArrayFloat64(array=np.diag([1.0, 2.0]))
        solver = mm.EigenSystem(A, do_vl=False, do_vr=False)
        solver.run()
        empty_vl = np.array(solver.get_vl(suppress_exception=True))
        empty_vr = np.array(solver.get_vr(suppress_exception=True))
        self.assertEqual(empty_vl.size, 0)
        self.assertEqual(empty_vr.size, 0)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
