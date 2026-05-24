import unittest

import numpy as np

import modmesh as mm


@unittest.skipIf(mm.EigenSystem is None,
                 "mm.EigenSystem is not built (no vendor LAPACK)")
class TestEigenSystemTC(unittest.TestCase):
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


@unittest.skipIf(mm.EigenSystem is None,
                 "mm.EigenSystem is not built (no vendor LAPACK)")
class TestEigenSystemPlexTC(unittest.TestCase):
    """Verify EigenSystem construction from a type-erased SimpleArray

    The type-erased SimpleArray is SimpleArrayPlex in C++.
    """

    def test_plex_float64_matches_typed_construction(self):
        # A float64 plex must construct EigenSystem and yield results identical
        # to the typed SimpleArrayFloat64 path: same data, same DGEEV call.
        A_np = np.array([
            [2.0, 1.0, 0.5],
            [0.0, 3.0, -1.0],
            [0.0, 0.0, 5.0],
        ], dtype="float64")
        plex = mm.SimpleArray(A_np)
        solver = mm.EigenSystem(plex)
        self.assertFalse(solver.done)
        solver.run()
        self.assertTrue(solver.done)
        # Identical to the typed SimpleArrayFloat64 construction.
        typed = mm.EigenSystem(mm.SimpleArrayFloat64(array=A_np))
        typed.run()
        for name in ("wr", "wi", "vl", "vr"):
            np.testing.assert_array_equal(
                np.array(getattr(solver, name)),
                np.array(getattr(typed, name)))
        # Sanity: the eigenvalues are the diagonal entries.
        np.testing.assert_allclose(np.sort(np.array(solver.wr)),
                                   [2.0, 3.0, 5.0], rtol=1e-12, atol=1e-12)

    def test_plex_rejects_non_float64_dtype(self):
        # The plex overload only accepts float64.  float32 and integer element
        # types must raise ValueError (std::invalid_argument).
        for dtype in ("float32", "int32"):
            A_np = np.array([[2.0, 0.0], [0.0, 3.0]], dtype=dtype)
            A = mm.SimpleArray(A_np)
            with self.assertRaisesRegex(
                    ValueError, r"data type must be float64"):
                mm.EigenSystem(A)

    def test_plex_non_square_rejected(self):
        # The float64 dtype check passes, so a non-square plex must still hit
        # the square-2D guard in the EigenSystem constructor.
        A_np = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype="float64")
        A = mm.SimpleArray(A_np)
        with self.assertRaisesRegex(
                ValueError, r"must be a square 2D SimpleArray"):
            mm.EigenSystem(A)

    def test_plex_forwards_do_vl_do_vr_flags(self):
        # do_vl/do_vr must pass through the plex overload unchanged: the
        # default keeps both true, and do_vr=False suppresses vr.
        A_np = np.array([[2.0, 0.0], [0.0, 3.0]], dtype="float64")
        solver = mm.EigenSystem(mm.SimpleArray(A_np))
        self.assertTrue(solver.do_vl)
        self.assertTrue(solver.do_vr)

        solver = mm.EigenSystem(mm.SimpleArray(A_np), do_vr=False)
        solver.run()
        self.assertTrue(solver.do_vl)
        self.assertFalse(solver.do_vr)
        with self.assertRaisesRegex(
                RuntimeError, r"right eigenvectors were not computed"):
            solver.vr  # noqa: B018

    def test_plex_matrix_property_survives_input_gc(self):
        # Mirror of test_matrix_property_survives_input_gc for the plex
        # overload: m_matrix references the array owned inside the plex, so
        # py::keep_alive<1, 2>() must stop it from dangling once the
        # Python-side plex reference is dropped.
        A_np = np.array([[3.0, 1.0], [0.0, 2.0]], dtype="float64")
        plex = mm.SimpleArray(A_np)
        solver = mm.EigenSystem(plex)
        del plex
        np.testing.assert_array_equal(solver.matrix.ndarray, A_np)


@unittest.skipIf(mm.EigenSystem is None,
                 "mm.EigenSystem is not built (no vendor LAPACK)")
class TestEigenCytnxCompatTC(unittest.TestCase):
    """Testing code for eigen problem solver compat layer to Cytnx

    Cytnx (https://github.com/Cytnx-dev/Cytnx) is a tensor-network library
    which also uses an eigen problem solver.
    """

    @staticmethod
    def _eig_general(mat):
        """Helper to diagonalize a real square matrix like cytnx.linalg.Eig.

        Takes a type-erased SimpleArray (SimpleArrayPlex) and returns (w,
        vecs): eigenvalues and matching column eigenvectors.
        """
        mat = np.ascontiguousarray(mat, dtype=np.float64)
        a = mm.SimpleArray(array=mat)
        solver = mm.EigenSystem(a, do_vl=False, do_vr=True)
        solver.run()
        wr = np.array(solver.wr)
        wi = np.array(solver.wi)
        vr = np.array(solver.vr)
        n = wr.shape[0]
        w = wr + 1j * wi
        vecs = np.zeros((n, n), dtype=np.complex128)
        j = 0
        while j < n:
            if abs(wi[j]) < 1e-13:
                vecs[:, j] = vr[:, j]
                j += 1
            else:
                vecs[:, j] = vr[:, j] + 1j * vr[:, j + 1]
                vecs[:, j + 1] = vr[:, j] - 1j * vr[:, j + 1]
                j += 2
        return w, vecs

    @staticmethod
    def _build_tfim_hamiltonian(length, coupling, field):
        """Dense periodic transverse-field Ising Hamiltonian.

        H = J sum_i sz_i sz_{i+1} - hx sum_i sx_i in the sz basis: bit i of the
        state index is spin i, sz sz is +1 when neighbors agree and -1
        otherwise, and sx flips a spin.  This is the operator Cytnx's
        example/ED/ed_ising.py applies matrix-free via a LinOp.
        """
        dim = 1 << length
        ham = np.zeros((dim, dim), dtype=np.float64)
        for state in range(dim):
            for i in range(length):
                j = (i + 1) % length
                spin_i = (state >> i) & 1
                spin_j = (state >> j) & 1
                ham[state, state] += coupling * (1.0 - 2.0 *
                                                 (spin_i ^ spin_j))
            for i in range(length):
                flipped = state ^ (1 << i)
                ham[flipped, state] += -field
        return ham

    def test_tfim_ground_state(self):
        """Lowest eigenpair of the transverse-field Ising chain.

        Mirrors Cytnx's exact-diagonalization ground-state example:
        https://github.com/Cytnx-dev/Cytnx/blob/master/example/ED/ed_ising.py
        """
        # L=4, J=1, transverse field 0.3: a 16x16 real symmetric matrix.
        length, coupling, field = 4, 1.0, 0.3
        ham = self._build_tfim_hamiltonian(length, coupling, field)

        w, vecs = self._eig_general(ham)
        # A Hermitian Hamiltonian has real eigenvalues.
        self.assertLess(np.max(np.abs(w.imag)), 1e-10)

        # eigvalsh is the independent reference spectrum.
        reference = np.linalg.eigvalsh(ham)
        np.testing.assert_allclose(np.sort(w.real), reference,
                                   rtol=1e-10, atol=1e-10)

        # Ground state: lowest eigenvalue and its eigenvector, what
        # Lanczos("Gnd") returns in the Cytnx example.
        gnd = int(np.argmin(w.real))
        energy = w[gnd].real
        np.testing.assert_allclose(energy, reference[0],
                                   rtol=1e-10, atol=1e-10)

        psi = vecs[:, gnd].real
        psi = psi / np.linalg.norm(psi)
        # The defining eigen-equation H |psi> = E0 |psi>.
        np.testing.assert_allclose(ham @ psi, energy * psi,
                                   rtol=1e-9, atol=1e-9)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
