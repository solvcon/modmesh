import unittest

import numpy as np
from numpy.testing import assert_almost_equal

import modmesh


def _euler_flux(u, gamma):
    """Analytic Euler conserved-variable flux f[ieq][d] for a state row u.

    Used as an independent reference for the C++ flux/Jacobian under test.
    """
    nd = u.size - 2
    rho = u[0]
    mom = u[1:1 + nd]
    energy = u[nd + 1]
    vel = mom / rho
    p = (gamma - 1.0) * (energy - 0.5 * rho * (vel @ vel))
    f = np.zeros((u.size, nd), dtype="float64")
    for d in range(nd):
        f[0, d] = mom[d]
        for k in range(nd):
            f[1 + k, d] = mom[k] * vel[d] + (p if k == d else 0.0)
        f[nd + 1, d] = (energy + p) * vel[d]
    return f


def _euler_jac_fd(u, gamma, h=1e-6):
    """Euler flux Jacobian J[ieq][jeq][d] by central-differencing the analytic
    flux.  Independent of the C++ analytic Jacobian, so it catches a wrong
    Jacobian entry or sign in the marcher."""
    neq = u.size
    nd = neq - 2
    jac = np.zeros((neq, neq, nd), dtype="float64")
    for j in range(neq):
        up = u.copy()
        um = u.copy()
        up[j] += h
        um[j] -= h
        fp = _euler_flux(up, gamma)
        fm = _euler_flux(um, gamma)
        jac[:, j, :] = (fp - fm) / (2 * h)
    return jac


def _fcmnd(mh, ec):
    clmfc = mh.clfcs.ndarray.shape[1] - 1
    return ec.sfcnd.ndarray.shape[1] // clmfc


def _calc_soln_reference(mh, ec, gamma):
    """Independent re-implementation of the CESE order-0 flux integral, using
    a finite-difference Euler Jacobian.  Reads the same so0c/so0t/so1c that
    calc_soln consumes and returns the expected so0n over the real cells."""
    nd = mh.ndim
    neq = nd + 2
    dt = ec.time_increment
    qdt, hdt = dt * 0.25, dt * 0.5
    fcmnd = _fcmnd(mh, ec)
    out = np.zeros((ec.ncell, neq), dtype="float64")
    for icl in range(ec.ncell):
        acc = np.zeros(neq, dtype="float64")
        for ifl in range(1, mh.clfcs[icl, 0] + 1):
            ifc = mh.clfcs[icl, ifl]
            jcl = mh.fccls[ifc, 0] + mh.fccls[ifc, 1] - icl
            jce = np.array([ec.cecnd[jcl, d] if jcl >= 0 else mh.clcnd[jcl, d]
                            for d in range(nd)], dtype="float64")
            bcnd = np.array([ec.cecnd[icl, ifl * nd + d]
                             for d in range(nd)], dtype="float64")
            js = np.array([ec.so0c[jcl, ieq] for ieq in range(neq)],
                          dtype="float64")
            jt = np.array([ec.so0t[jcl, ieq] for ieq in range(neq)],
                          dtype="float64")
            j1 = np.array([[ec.so1c[jcl, ieq, d] for d in range(nd)]
                           for ieq in range(neq)], dtype="float64")
            bvol = ec.cevol[icl, ifl]
            for ieq in range(neq):
                acc[ieq] += (js[ieq] + (bcnd - jce) @ j1[ieq]) * bvol
            fcn = _euler_flux(js, gamma)
            jac = _euler_jac_fd(js, gamma)
            for inf in range(mh.fcnds[ifc, 0]):
                sfi = (ifl - 1) * fcmnd + inf
                sc = np.array([ec.sfcnd[icl, sfi, d]
                               for d in range(nd)], dtype="float64")
                sn = np.array([ec.sfnml[icl, sfi, d]
                               for d in range(nd)], dtype="float64")
                usfc = qdt * jt + np.array(
                    [(sc - jce) @ j1[ieq] for ieq in range(neq)],
                    dtype="float64")
                for ieq in range(neq):
                    acc[ieq] -= hdt * ((fcn[ieq] + jac[ieq].T @ usfc) @ sn)
        out[icl] = acc / ec.cevol[icl, 0]
    return out


def _calc_dsoln_reference(mh, ec):
    """Independent re-implementation of the order-1 gradient weighting/limiter,
    reusing the verified GradientElement.solve_gradient primitive.  Returns the
    expected so1n and whether the W-3/4 limiter is actually active."""
    nd = mh.ndim
    neq = nd + 2
    hdt = ec.time_increment * 0.5
    az = 1e-200
    out = np.zeros((ec.ncell, neq, nd), dtype="float64")
    active = False
    for icl in range(ec.ncell):
        acfl = abs(ec.cflc[icl])
        sgm0 = ec.sigma0 / acfl
        tau = ec.taumin + acfl * ec.tauscale
        ge = modmesh.GradientElement(mesh=mh, cecnd=ec.cecnd, icl=icl, tau=tau)
        nfge, ofg1 = ge.nfge, ge.nfge_inverse
        grad = np.zeros((nfge, neq, nd), dtype="float64")
        widv = np.zeros((nfge, neq), dtype="float64")
        wacc = np.zeros(neq, dtype="float64")
        for ifge in range(nfge):
            faces = ge.faces(ifge)
            for ieq in range(neq):
                udf = np.zeros(nd, dtype="float64")
                for ivx in range(nd):
                    ifl = faces[ivx] - 1
                    jcl = ge.rcl(ifl)
                    val = ec.so0c[jcl, ieq] + hdt * ec.so0t[jcl, ieq] \
                        - ec.so0n[icl, ieq]
                    for d in range(nd):
                        val += ge.jdis(ifl, d) * ec.so1c[jcl, ieq, d]
                    udf[ivx] = val
                g = np.array(ge.solve_gradient(ifge, udf.tolist()),
                             dtype="float64")
                grad[ifge, ieq] = g
                wgt = 1.0 / np.sqrt(g @ g + az)
                wacc[ieq] += wgt
                widv[ifge, ieq] = wgt
        wpa = np.zeros((neq, 2), dtype="float64")
        for ifge in range(nfge):
            for ieq in range(neq):
                w = widv[ifge, ieq] / wacc[ieq] - ofg1
                widv[ifge, ieq] = w
                wpa[ieq, 0] = max(wpa[ieq, 0], w)
                wpa[ieq, 1] = min(wpa[ieq, 1], w)
        for ieq in range(neq):
            sm = min((1.0 - ofg1) / (wpa[ieq, 0] + az),
                     -ofg1 / (wpa[ieq, 1] - az), sgm0)
            if np.abs(widv[:, ieq]).max() > 1e-9:
                active = True
            for ifge in range(nfge):
                w = ofg1 + sm * widv[ifge, ieq]
                out[icl, ieq] += w * grad[ifge, ieq]
    return out, active


class _TriangleMeshBase(unittest.TestCase):
    """3 triangles around the origin."""

    @classmethod
    def setUpClass(cls):
        mh = modmesh.StaticMesh(ndim=2, nnode=4, nface=0, ncell=3)
        mh.ndcrd[:, :] = [(0, 0), (-1, -1), (1, -1), (0, 1)]
        mh.cltpn.fill(modmesh.StaticMesh.TRIANGLE)
        mh.clnds[:, :4] = [(3, 0, 1, 2), (3, 0, 2, 3), (3, 0, 3, 1)]
        mh.build_interior(do_metric=True)
        mh.build_boundary()
        mh.build_ghost()
        cls.mesh = mh
        cls.ec = modmesh.EulerCore(mesh=mh, time_increment=0.01)


class _QuadMeshBase(unittest.TestCase):
    """Single unit-square quadrilateral."""

    @classmethod
    def setUpClass(cls):
        mh = modmesh.StaticMesh(ndim=2, nnode=4, nface=0, ncell=1)
        mh.ndcrd[:, :] = [(0, 0), (1, 0), (1, 1), (0, 1)]
        mh.cltpn.fill(modmesh.StaticMesh.QUADRILATERAL)
        mh.clnds[:, :5] = [(4, 0, 1, 2, 3)]
        mh.build_interior(do_metric=True)
        mh.build_boundary()
        mh.build_ghost()
        cls.mesh = mh
        cls.ec = modmesh.EulerCore(mesh=mh, time_increment=0.01)


class _MixedMeshBase(unittest.TestCase):
    """1 quad + 2 triangles."""

    @classmethod
    def setUpClass(cls):
        mh = modmesh.StaticMesh(ndim=2, nnode=6, nface=0, ncell=3)
        mh.ndcrd[:, :] = [
            (0, 0), (1, 0), (2, 0),
            (0, 1), (1, 1), (2, 1),
        ]
        mh.cltpn[:] = [
            modmesh.StaticMesh.QUADRILATERAL,
            modmesh.StaticMesh.TRIANGLE,
            modmesh.StaticMesh.TRIANGLE,
        ]
        mh.clnds[:, :5] = [
            (4, 0, 1, 4, 3),
            (3, 1, 2, 4, 0),
            (3, 2, 5, 4, 0),
        ]
        mh.build_interior(do_metric=True)
        mh.build_boundary()
        mh.build_ghost()
        cls.mesh = mh
        cls.ec = modmesh.EulerCore(mesh=mh, time_increment=0.01)


class _TetrahedronMeshBase(unittest.TestCase):
    """Single tetrahedron (4 triangular faces)."""

    @classmethod
    def setUpClass(cls):
        mh = modmesh.StaticMesh(ndim=3, nnode=4, nface=4, ncell=1)
        mh.ndcrd[:, :] = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
        mh.cltpn.fill(modmesh.StaticMesh.TETRAHEDRON)
        mh.clnds[:, :5] = [(4, 0, 1, 2, 3)]
        mh.build_interior(do_metric=True)
        mh.build_boundary()
        mh.build_ghost()
        cls.mesh = mh
        cls.ec = modmesh.EulerCore(mesh=mh, time_increment=0.01)


class _HexahedronMeshBase(unittest.TestCase):
    """Single unit-cube hexahedron (6 quadrilateral faces)."""

    @classmethod
    def setUpClass(cls):
        mh = modmesh.StaticMesh(ndim=3, nnode=8, nface=6, ncell=1)
        mh.ndcrd[:, :] = [
            (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
            (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1),
        ]
        mh.cltpn.fill(modmesh.StaticMesh.HEXAHEDRON)
        mh.clnds[:, :9] = [(8, 0, 1, 2, 3, 4, 5, 6, 7)]
        mh.build_interior(do_metric=True)
        mh.build_boundary()
        mh.build_ghost()
        cls.mesh = mh
        cls.ec = modmesh.EulerCore(mesh=mh, time_increment=0.01)


class _PrismMeshBase(unittest.TestCase):
    """Single triangular prism (2 triangle + 3 quadrilateral faces)."""

    @classmethod
    def setUpClass(cls):
        mh = modmesh.StaticMesh(ndim=3, nnode=6, nface=5, ncell=1)
        mh.ndcrd[:, :] = [
            (0, 0, 0), (1, 0, 0), (0, 1, 0),
            (0, 0, 1), (1, 0, 1), (0, 1, 1),
        ]
        mh.cltpn.fill(modmesh.StaticMesh.PRISM)
        mh.clnds[:, :7] = [(6, 0, 1, 2, 3, 4, 5)]
        mh.build_interior(do_metric=True)
        mh.build_boundary()
        mh.build_ghost()
        cls.mesh = mh
        cls.ec = modmesh.EulerCore(mesh=mh, time_increment=0.01)


class _PyramidMeshBase(unittest.TestCase):
    """Single square pyramid (4 triangle + 1 quadrilateral faces)."""

    @classmethod
    def setUpClass(cls):
        mh = modmesh.StaticMesh(ndim=3, nnode=5, nface=5, ncell=1)
        mh.ndcrd[:, :] = [
            (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
            (0.5, 0.5, 1),
        ]
        mh.cltpn.fill(modmesh.StaticMesh.PYRAMID)
        mh.clnds[:, :6] = [(5, 0, 1, 2, 3, 4)]
        mh.build_interior(do_metric=True)
        mh.build_boundary()
        mh.build_ghost()
        cls.mesh = mh
        cls.ec = modmesh.EulerCore(mesh=mh, time_increment=0.01)


class _GradientElementBoundsBase:
    """Structural checks that do not read CE geometry values."""

    def _ge(self, icl, tau=1.0):
        mh, cecnd = self.mesh, self.ec.cecnd
        return modmesh.GradientElement(mesh=mh, cecnd=cecnd, icl=icl, tau=tau)

    def test_basic_properties(self):
        ge = self._ge(0)
        self.assertEqual(0, ge.icl)
        self.assertEqual(self.mesh.ndim, ge.ndim)
        self.assertEqual(self.mesh.clfcs[0, 0], ge.clnfc)

    def test_accessor_index_bounds(self):
        ge = self._ge(0)
        nfc, nd = ge.clnfc, ge.ndim
        # In-range access does not raise across the whole valid range.
        # The d loop reaches ndim - 1 (== 2 only in 3D), so it pins idis
        # and jdis to accept the full dimension range, not just d <= 1.
        for ifl in range(nfc):
            ge.rcl(ifl)
            for d in range(nd):
                ge.idis(ifl, d)
                ge.jdis(ifl, d)
        # Out-of-range face index raises IndexError.
        for ifl in (-1, nfc):
            with self.assertRaises(IndexError):
                ge.rcl(ifl)
            with self.assertRaises(IndexError):
                ge.idis(ifl, 0)
            with self.assertRaises(IndexError):
                ge.jdis(ifl, 0)
        # Out-of-range dimension index raises IndexError.
        for d in (-1, nd):
            with self.assertRaises(IndexError):
                ge.idis(0, d)
            with self.assertRaises(IndexError):
                ge.jdis(0, d)

    def test_fge_table(self):
        # Sub-element (FGE) table is geometry-free; it holds even when the
        # CE geometry is NaN.
        ge = self._ge(0)
        nfge = ge.nfge
        self.assertGreater(nfge, 0)
        assert_almost_equal(ge.nfge_inverse, 1.0 / nfge, decimal=12)
        for ifge in range(nfge):
            faces = ge.faces(ifge)
            self.assertEqual(self.mesh.ndim, len(faces))
            for ifl in faces:
                # 1-based face index into the per-cell face list.
                self.assertGreaterEqual(ifl, 1)
                self.assertLessEqual(ifl, ge.clnfc)

    def test_ifge_bounds(self):
        ge = self._ge(0)
        nd = self.mesh.ndim
        for ifge in (-1, ge.nfge):
            with self.assertRaises(IndexError):
                ge.faces(ifge)
            with self.assertRaises(IndexError):
                ge.displacement_matrix(ifge)
            with self.assertRaises(IndexError):
                ge.solve_gradient(ifge, [0.0] * nd)


class _GradientElementBase(_GradientElementBoundsBase):
    """Adds geometry tests that read CE data over all cells."""

    def test_displacement_matrix_nonsingular(self):
        nd = self.mesh.ndim
        for icl in range(self.mesh.ncell):
            ge = self._ge(icl)
            mat = np.array([[ge.idis(ifl, d) for d in range(nd)]
                            for ifl in range(ge.clnfc)], dtype="float64")
            # The face-displacement vectors must span R^ndim for the
            # gradient reconstruction to be well posed.  A per-simplex
            # determinant would wrongly fail for the hexahedron, whose
            # opposite faces are antiparallel.
            self.assertEqual(nd, np.linalg.matrix_rank(mat),
                             f"cell {icl}: idis does not span {nd}D")

    def test_idis_jdis_consistency(self):
        mh, cecnd, nd = self.mesh, self.ec.cecnd, self.mesh.ndim
        for icl in range(mh.ncell):
            ge = self._ge(icl)
            for ifl in range(ge.clnfc):
                jcl = ge.rcl(ifl)
                for d in range(nd):
                    jce = cecnd[jcl, d] if jcl >= 0 else mh.clcnd[jcl, d]
                    lhs = ge.idis(ifl, d) + cecnd[icl, d]
                    assert_almost_equal(lhs, ge.jdis(ifl, d) + jce, decimal=12)

    def test_tau_zero(self):
        mh, cecnd, nd = self.mesh, self.ec.cecnd, self.mesh.ndim
        for icl in range(mh.ncell):
            ge = self._ge(icl, tau=0.0)
            shift = [ge.idis(0, d) + cecnd[icl, d] - cecnd[icl, nd + d]
                     for d in range(nd)]
            for ifl in range(1, ge.clnfc):
                for d in range(nd):
                    pos = ge.idis(ifl, d) + cecnd[icl, d]
                    bce = cecnd[icl, (ifl + 1) * nd + d]
                    assert_almost_equal(pos, bce + shift[d], decimal=12)

    def test_displacement_matrix_per_fge_nonsingular(self):
        nd = self.mesh.ndim
        for icl in range(self.mesh.ncell):
            ge = self._ge(icl)
            for ifge in range(ge.nfge):
                mat = np.array(ge.displacement_matrix(ifge), dtype="float64")
                self.assertEqual((nd, nd), mat.shape)
                self.assertGreater(
                    abs(np.linalg.det(mat)), 1e-10,
                    f"cell {icl} ifge {ifge}: singular FGE matrix")

    def test_solve_gradient_linear_field(self):
        # For a linear field u(x) = c + g . x the solution delta at each
        # gradient evaluation point is exactly g . idis, so the per-FGE
        # solve must recover g exactly (up to round-off).
        nd = self.mesh.ndim
        grad = np.array([1.5, -2.7, 0.9][:nd], dtype="float64")
        for icl in range(self.mesh.ncell):
            ge = self._ge(icl)
            for ifge in range(ge.nfge):
                dst = np.array(ge.displacement_matrix(ifge), dtype="float64")
                faces = ge.faces(ifge)
                # Matrix rows are the per-face idis vectors.
                for ivx, ifl in enumerate(faces):
                    for d in range(nd):
                        assert_almost_equal(
                            dst[ivx, d], ge.idis(ifl - 1, d), decimal=12)
                udf = dst @ grad
                got = np.array(ge.solve_gradient(ifge, udf.tolist()),
                               dtype="float64")
                assert_almost_equal(got, grad, decimal=9)


class GradientElementTriangleTC(_GradientElementBase, _TriangleMeshBase):
    """Per-cell GradientElement on 3 triangles."""


class GradientElementQuadTC(_GradientElementBase, _QuadMeshBase):
    """Per-cell GradientElement on a unit-square quad."""


class GradientElementMixedTC(_GradientElementBase, _MixedMeshBase):
    """Per-cell GradientElement on a 2D mixed mesh."""


class GradientElementTetrahedronTC(_GradientElementBase, _TetrahedronMeshBase):
    """Per-cell GradientElement on a single tetrahedron (4 faces)."""


class GradientElementHexahedronTC(_GradientElementBase, _HexahedronMeshBase):
    """Per-cell GradientElement on a single hexahedron (6 faces)."""


class GradientElementPrismTC(_GradientElementBase, _PrismMeshBase):
    """Per-cell GradientElement on a single prism (5 faces)."""


class GradientElementPyramidTC(_GradientElementBase, _PyramidMeshBase):
    """Per-cell GradientElement on a single pyramid (5 faces)."""


class _EulerSolutionBase:
    """EulerCore Phase 3 solution storage and initialization."""

    GAMMA = 1.4
    RHO = 1.2
    PRES = 0.9
    VEL = (0.3, -0.15, 0.05)

    def _ec(self):
        # A fresh core per test keeps the shared class mesh read-only.
        return modmesh.EulerCore(mesh=self.mesh, time_increment=0.01)

    def _vel(self, nd):
        return list(self.VEL[:nd])

    def test_solution_array_shapes(self):
        ec = self._ec()
        nd = self.mesh.ndim
        neq = nd + 2
        total = ec.ngstcell + ec.ncell
        self.assertEqual(neq, ec.neq)
        for name in ("so0c", "so0n", "so0t", "stm"):
            self.assertEqual((total, neq), getattr(ec, name).shape)
        for name in ("so1c", "so1n"):
            self.assertEqual((total, neq, nd), getattr(ec, name).shape)
        for name in ("cflo", "cflc", "gamma"):
            self.assertEqual((total,), getattr(ec, name).shape)

    def test_init_solution_columns(self):
        ec = self._ec()
        nd = self.mesh.ndim
        v = self._vel(nd)
        ec.init_solution(gamma=self.GAMMA, rho=self.RHO, v=v, p=self.PRES)
        vsq = sum(c * c for c in v)
        energy = self.PRES / (self.GAMMA - 1.0) + 0.5 * self.RHO * vsq
        so0n = ec.so0n
        for icl in range(ec.ncell):
            assert_almost_equal(so0n[icl, 0], self.RHO)
            for d in range(nd):
                assert_almost_equal(so0n[icl, 1 + d], self.RHO * v[d])
            assert_almost_equal(so0n[icl, nd + 1], energy)
            # The pressure must be recoverable from the conserved state.
            momsq = sum(so0n[icl, 1 + d] ** 2 for d in range(nd))
            ke = momsq / (2.0 * self.RHO)
            p_rec = (self.GAMMA - 1.0) * (so0n[icl, nd + 1] - ke)
            assert_almost_equal(p_rec, self.PRES)
        # gamma is filled across every row, ghost cells included.
        total = ec.ngstcell + ec.ncell
        assert_almost_equal(ec.gamma.ndarray,
                            np.full(total, self.GAMMA, dtype="float64"))
        # init_solution leaves the conserved table's ghost rows untouched
        # (zero); ghost states are populated by boundary conditions later.
        for icl in range(-ec.ngstcell, 0):
            for ieq in range(ec.neq):
                assert_almost_equal(so0n[icl, ieq], 0.0)

    def test_init_solution_validation(self):
        ec = self._ec()
        nd = self.mesh.ndim
        good = dict(gamma=self.GAMMA, rho=self.RHO,
                    v=self._vel(nd), p=self.PRES)
        for bad in (dict(v=[0.1] * (nd - 1)),  # too-short velocity
                    dict(gamma=1.0),           # gamma must be > 1
                    dict(rho=0.0),             # rho must be > 0
                    dict(p=-1.0)):             # pressure must be >= 0
            with self.assertRaises(ValueError):
                ec.init_solution(**dict(good, **bad))

    def test_calc_cfl_uniform_field(self):
        ec = self._ec()
        nd = self.mesh.ndim
        v = self._vel(nd)
        ec.init_solution(gamma=self.GAMMA, rho=self.RHO, v=v, p=self.PRES)
        ec.calc_cfl()
        hdt = ec.time_increment / 2.0
        vsq = sum(c * c for c in v)
        wspd = np.sqrt(self.GAMMA * self.PRES / self.RHO) + np.sqrt(vsq)
        # For a positive-pressure field the energy correction is a no-op up
        # to the TINY offset, so the stored energy stays at its init value.
        energy0 = self.PRES / (self.GAMMA - 1.0) + 0.5 * self.RHO * vsq
        cecnd = ec.cecnd
        for icl in range(ec.ncell):
            clnfc = self.mesh.clfcs[icl, 0]
            dist = min(
                np.sqrt(sum((cecnd[icl, ifl * nd + d] - cecnd[icl, d]) ** 2
                            for d in range(nd)))
                for ifl in range(1, clnfc + 1))
            assert_almost_equal(ec.cflo[icl], hdt * wspd / dist)
            # Pressure is positive, so the clamped CFL equals the original.
            assert_almost_equal(ec.cflc[icl], ec.cflo[icl])
            assert_almost_equal(ec.so0n[icl, nd + 1], energy0)

    def test_calc_cfl_negative_pressure(self):
        ec = self._ec()
        nd = self.mesh.ndim
        v = self._vel(nd)
        ec.init_solution(gamma=self.GAMMA, rho=self.RHO, v=v, p=self.PRES)
        # Zero the stored energy so the recovered pressure goes negative
        # while the momentum (kinetic energy) stays finite.
        for icl in range(ec.ncell):
            ec.so0n[icl, nd + 1] = 0.0
        ec.calc_cfl()
        momsq = sum((self.RHO * v[d]) ** 2 for d in range(nd))
        ke = momsq / (2.0 * self.RHO)
        for icl in range(ec.ncell):
            # The pressure is clamped to zero, so the clamped CFL is forced
            # to 1 and the energy is rebuilt from the kinetic part alone.
            assert_almost_equal(ec.cflc[icl], 1.0)
            assert_almost_equal(ec.so0n[icl, nd + 1], ke)

    def test_update_swaps_buffers(self):
        ec = self._ec()
        nd = self.mesh.ndim
        ec.init_solution(gamma=self.GAMMA, rho=self.RHO,
                         v=self._vel(nd), p=self.PRES)
        # Seed the new-step order-1 buffer with a recognizable pattern.
        so1n_view = ec.so1n.ndarray
        so1n_view[...] = np.arange(
            so1n_view.size, dtype="float64").reshape(so1n_view.shape)
        so0n_before = ec.so0n.ndarray.copy()
        so0c_before = ec.so0c.ndarray.copy()
        so1n_before = ec.so1n.ndarray.copy()
        so1c_before = ec.so1c.ndarray.copy()
        ec.update()
        assert_almost_equal(ec.so0c.ndarray, so0n_before)
        assert_almost_equal(ec.so0n.ndarray, so0c_before)
        assert_almost_equal(ec.so1c.ndarray, so1n_before)
        assert_almost_equal(ec.so1n.ndarray, so1c_before)


class EulerSolutionTriangleTC(_EulerSolutionBase, _TriangleMeshBase):
    """EulerCore solution storage on 3 triangles."""


class EulerSolutionQuadTC(_EulerSolutionBase, _QuadMeshBase):
    """EulerCore solution storage on a unit-square quad."""


class EulerSolutionMixedTC(_EulerSolutionBase, _MixedMeshBase):
    """EulerCore solution storage on a 2D mixed mesh."""


class EulerSolutionTetrahedronTC(_EulerSolutionBase, _TetrahedronMeshBase):
    """EulerCore solution storage on a single tetrahedron."""


class EulerSolutionHexahedronTC(_EulerSolutionBase, _HexahedronMeshBase):
    """EulerCore solution storage on a single hexahedron."""


class _EulerMarchBase:
    """EulerCore phase 4 solution marching."""

    GAMMA = 1.4
    RHO = 1.2
    PRES = 0.9
    VEL = (0.3, -0.15, 0.05)

    def _ec(self):
        # A fresh core per test keeps the shared class mesh read-only.
        return modmesh.EulerCore(mesh=self.mesh, time_increment=0.01)

    def _vel(self, nd):
        return list(self.VEL[:nd])

    def _uniform_row(self, nd):
        v = self._vel(nd)
        vsq = sum(c * c for c in v)
        energy = self.PRES / (self.GAMMA - 1.0) + 0.5 * self.RHO * vsq
        return [self.RHO] + [self.RHO * v[d] for d in range(nd)] + [energy]

    def test_march_param_defaults(self):
        ec = self._ec()
        assert_almost_equal(ec.sigma0, 3.0)
        assert_almost_equal(ec.taumin, 0.0)
        assert_almost_equal(ec.tauscale, 1.0)
        # The marching parameters are writable.
        ec.sigma0, ec.taumin, ec.tauscale = 2.5, 0.1, 0.7
        assert_almost_equal(ec.sigma0, 2.5)
        assert_almost_equal(ec.taumin, 0.1)
        assert_almost_equal(ec.tauscale, 0.7)

    def test_calc_solt_euler_flux(self):
        # calc_solt sets so0t = -(Jacobian . so1c).  Seeding so1c[:, d] with a
        # scalar multiple coef[d] of the conserved state turns the Jacobian
        # product in each direction into the analytic Euler flux there (the
        # flux is homogeneous of degree one in the conserved variables), so
        # so0t == -sum_d coef[d] * flux_d.  Distinct coef[d] exercises every
        # Jacobian direction column, not just one.
        ec = self._ec()
        nd, neq = self.mesh.ndim, self.mesh.ndim + 2
        row = np.array(self._uniform_row(nd), dtype="float64")
        coef = [0.7, -1.3, 0.4][:nd]
        for icl in range(ec.ncell):
            ec.gamma[icl] = self.GAMMA
            for ieq in range(neq):
                ec.so0c[icl, ieq] = row[ieq]
                for d in range(nd):
                    ec.so1c[icl, ieq, d] = coef[d] * row[ieq]
        ec.calc_solt()
        flux = _euler_flux(row, self.GAMMA)  # flux[ieq][d]
        expect = [-sum(coef[d] * flux[ieq][d] for d in range(nd))
                  for ieq in range(neq)]
        for icl in range(ec.ncell):
            for ieq in range(neq):
                assert_almost_equal(ec.so0t[icl, ieq], expect[ieq], decimal=10)

    def test_calc_soln_freestream(self):
        # A uniform conserved state with zero gradient is preserved exactly by
        # the CESE flux integral: so0n == so0c.  Ghost rows are set to the same
        # uniform state to stand in for the (phase 5) boundary conditions.
        ec = self._ec()
        nd, neq = self.mesh.ndim, self.mesh.ndim + 2
        row = self._uniform_row(nd)
        for cell in range(-ec.ngstcell, ec.ncell):
            ec.gamma[cell] = self.GAMMA
            for ieq in range(neq):
                ec.so0c[cell, ieq] = row[ieq]
                ec.so0t[cell, ieq] = 0.0
                for d in range(nd):
                    ec.so1c[cell, ieq, d] = 0.0
        ec.calc_solt()
        ec.calc_soln()
        for icl in range(ec.ncell):
            for ieq in range(neq):
                assert_almost_equal(ec.so0n[icl, ieq], row[ieq], decimal=10)

    def test_calc_dsoln_linear_field(self):
        # For a global linear field u(x) = c + g . x sampled at every CE
        # solution point, calc_dsoln recovers the gradient g exactly: each
        # fundamental gradient element yields g, so the weighting/limiter
        # reduces to g regardless of the per-cell tau and sigma0.
        ec = self._ec()
        mh = self.mesh
        nd, neq = mh.ndim, mh.ndim + 2
        g = [[0.1 * (ieq + 1) + 0.01 * (d + 1) for d in range(nd)]
             for ieq in range(neq)]
        c = [0.5 + 0.3 * ieq for ieq in range(neq)]

        def point(cell):
            return [ec.cecnd[cell, d] if cell >= 0 else mh.clcnd[cell, d]
                    for d in range(nd)]

        for cell in range(-ec.ngstcell, ec.ncell):
            x = point(cell)
            for ieq in range(neq):
                val = c[ieq] + sum(g[ieq][d] * x[d] for d in range(nd))
                ec.so0c[cell, ieq] = val
                ec.so0n[cell, ieq] = val
                ec.so0t[cell, ieq] = 0.0
                for d in range(nd):
                    ec.so1c[cell, ieq, d] = g[ieq][d]
            ec.cflc[cell] = 0.5
        ec.calc_dsoln()
        for icl in range(ec.ncell):
            for ieq in range(neq):
                for d in range(nd):
                    assert_almost_equal(ec.so1n[icl, ieq, d], g[ieq][d],
                                        decimal=9)

    def test_calc_soln_against_reference(self):
        # Drive calc_soln with a non-uniform state (nonzero so1c and so0t and
        # distinct ghost states) so the temporal flux and the order-1 spatial
        # reconstruction are both active, then compare against an independent
        # finite-difference-Jacobian reimplementation.  Unlike the free-stream
        # case this constrains the hdt/qdt coefficients, the temporal sign, and
        # the Jacobian -- none of which a uniform state can detect.
        ec = self._ec()
        nd, neq = self.mesh.ndim, self.mesh.ndim + 2
        mh = self.mesh
        base = [1.2, 0.3, -0.15, 0.2, 2.5]
        for cell in range(-ec.ngstcell, ec.ncell):
            x = [ec.cecnd[cell, d] if cell >= 0 else mh.clcnd[cell, d]
                 for d in range(nd)]
            ec.gamma[cell] = self.GAMMA
            for ieq in range(neq):
                s = 1.0 + 0.1 * x[0] + 0.03 * ieq
                for d in range(1, nd):
                    s += 0.07 * x[d]
                ec.so0c[cell, ieq] = base[ieq] * s
                ec.so0t[cell, ieq] = 0.01 * (ieq + 1)
                for d in range(nd):
                    ec.so1c[cell, ieq, d] = 0.02 * (ieq + 1) * (d + 1)
        ec.calc_soln()
        ref = _calc_soln_reference(mh, ec, self.GAMMA)
        for icl in range(ec.ncell):
            for ieq in range(neq):
                assert_almost_equal(ec.so0n[icl, ieq], ref[icl, ieq],
                                    decimal=10)

    def test_calc_dsoln_against_reference(self):
        # Drive calc_dsoln with a non-linear field so the per-FGE gradients
        # differ and the W-1/2 / W-3/4 limiter is actually exercised (the
        # linear-field test leaves it dead, since identical gradients zero the
        # limiter delta).  Compare against an independent reimplementation of
        # the weighting that reuses the verified solve_gradient primitive.
        ec = self._ec()
        nd, neq = self.mesh.ndim, self.mesh.ndim + 2
        mh = self.mesh
        for cell in range(-ec.ngstcell, ec.ncell):
            x = [ec.cecnd[cell, d] if cell >= 0 else mh.clcnd[cell, d]
                 for d in range(nd)]
            for ieq in range(neq):
                val = 1.0 + 0.5 * x[0] + 0.4 * x[0] * x[0] + 0.2 * ieq
                for d in range(1, nd):
                    val += 0.3 * x[d]
                ec.so0c[cell, ieq] = val
                ec.so0n[cell, ieq] = val
                ec.so0t[cell, ieq] = 0.0
                for d in range(nd):
                    g = 0.1 * (ieq + 1) + 0.05 * x[0] * (d + 1)
                    ec.so1c[cell, ieq, d] = g
            ec.cflc[cell] = 0.5
        ec.calc_dsoln()
        ref, active = _calc_dsoln_reference(mh, ec)
        self.assertTrue(active, "limiter delta stayed zero; field too smooth")
        for icl in range(ec.ncell):
            for ieq in range(neq):
                for d in range(nd):
                    assert_almost_equal(ec.so1n[icl, ieq, d],
                                        ref[icl, ieq, d], decimal=10)

    def test_march_bounded(self):
        # Without boundary conditions (phase 5) marching is only well posed on
        # meshes with interior faces; single-cell meshes are all-boundary.
        if self.mesh.ncell < 2:
            self.skipTest("march needs interior faces; BCs land in phase 5")
        ec = self._ec()
        nd = self.mesh.ndim
        ec.init_solution(gamma=self.GAMMA, rho=self.RHO,
                         v=self._vel(nd), p=self.PRES)
        ec.march(steps=3)
        so0n = ec.so0n.ndarray
        self.assertTrue(np.all(np.isfinite(so0n)))
        # Density stays positive and the state stays bounded.
        for icl in range(ec.ncell):
            self.assertGreater(ec.so0n[icl, 0], 0.0)
        self.assertLess(np.abs(so0n).max(), 1e6)


class EulerMarchTriangleTC(_EulerMarchBase, _TriangleMeshBase):
    """EulerCore marching on 3 triangles."""


class EulerMarchQuadTC(_EulerMarchBase, _QuadMeshBase):
    """EulerCore marching on a unit-square quad."""


class EulerMarchMixedTC(_EulerMarchBase, _MixedMeshBase):
    """EulerCore marching on a 2D mixed mesh."""


class EulerMarchTetrahedronTC(_EulerMarchBase, _TetrahedronMeshBase):
    """EulerCore marching on a single tetrahedron."""


class EulerMarchHexahedronTC(_EulerMarchBase, _HexahedronMeshBase):
    """EulerCore marching on a single hexahedron."""


def _build_quad_channel(nx, ny, lx, ly):
    """A structured nx-by-ny quadrilateral grid over [0, lx] x [0, ly]."""
    mh = modmesh.StaticMesh(ndim=2, nnode=(nx + 1) * (ny + 1), nface=0,
                            ncell=nx * ny)
    mh.ndcrd[:, :] = [(i * lx / nx, j * ly / ny)
                      for j in range(ny + 1) for i in range(nx + 1)]
    mh.cltpn.fill(modmesh.StaticMesh.QUADRILATERAL)

    def nid(i, j):
        return j * (nx + 1) + i

    mh.clnds[:, :5] = [
        (4, nid(i, j), nid(i + 1, j), nid(i + 1, j + 1), nid(i, j + 1))
        for j in range(ny) for i in range(nx)]
    mh.build_interior(do_metric=True)
    mh.build_boundary()
    mh.build_ghost()
    return mh


class _EulerBCBase:
    """EulerCore phase 5 boundary conditions (the ghost-cell trim passes)."""

    GAMMA = 1.4

    def _ec(self):
        # A fresh core per test keeps the shared class mesh read-only.
        return modmesh.EulerCore(mesh=self.mesh, time_increment=0.01)

    def _faces(self):
        return [int(f) for f in self.mesh.bndfcs.ndarray[:, 0]]

    def _bnd(self):
        # (ifc, interior cell, ghost cell) for every boundary face.
        return [(ifc, self.mesh.fccls[ifc, 0], self.mesh.fccls[ifc, 1])
                for ifc in self._faces()]

    def _normal(self, ifc):
        nd = self.mesh.ndim
        return np.array([self.mesh.fcnml[ifc, d] for d in range(nd)],
                        dtype="float64")

    def _inlet_value(self, nd):
        # [rho, v(ndim), p, gamma]
        return [1.3] + [2.0, 0.3, -0.1][:nd] + [0.8, self.GAMMA]

    def test_normal_matrix_orthonormal(self):
        # The frame is an orthonormal rotation whose first row is the outward
        # unit normal, so the handlers can rotate in and out by the transpose.
        ec = self._ec()
        nd = self.mesh.ndim
        eye = np.eye(nd)
        for ifc in self._faces():
            mat = np.array(ec.get_normal_matrix(ifc), dtype="float64")
            self.assertEqual((nd, nd), mat.shape)
            assert_almost_equal(mat @ mat.T, eye, decimal=12)
            assert_almost_equal(mat[0], self._normal(ifc), decimal=12)
            assert_almost_equal(np.linalg.det(mat), 1.0, decimal=12)

    def test_nonrefl_do0(self):
        # Non-reflective do0 copies the whole interior state to the ghost.
        ec = self._ec()
        neq = self.mesh.ndim + 2
        for icl in range(ec.ncell):
            for ieq in range(neq):
                ec.so0n[icl, ieq] = 1.0 + 0.1 * icl + 0.37 * ieq
        ec.add_nonrefl(self._faces())
        ec.bc_soln()
        for ifc, icl, jcl in self._bnd():
            for ieq in range(neq):
                assert_almost_equal(ec.so0n[jcl, ieq], ec.so0n[icl, ieq])

    def test_nonrefl_do1(self):
        # Non-reflective do1 zeroes the wall-normal derivative and keeps the
        # tangential part: the ghost gradient is the tangential projection.
        ec = self._ec()
        nd, neq = self.mesh.ndim, self.mesh.ndim + 2
        for icl in range(ec.ncell):
            for ieq in range(neq):
                for d in range(nd):
                    ec.so1c[icl, ieq, d] = \
                        0.2 + 0.1 * ieq - 0.05 * d + 0.03 * icl
        ec.add_nonrefl(self._faces())
        ec.bc_dsoln()
        for ifc, icl, jcl in self._bnd():
            n = self._normal(ifc)
            for ieq in range(neq):
                gi = np.array([ec.so1c[icl, ieq, d] for d in range(nd)],
                              dtype="float64")
                gg = np.array([ec.so1n[jcl, ieq, d] for d in range(nd)],
                              dtype="float64")
                assert_almost_equal(gg, gi - (gi @ n) * n, decimal=12)
                assert_almost_equal(gg @ n, 0.0, decimal=12)

    def test_slipwall_do0(self):
        # Slip-wall do0 copies density and energy and reflects the momentum,
        # so the wall-normal mass flux at the face vanishes.
        ec = self._ec()
        nd, neq = self.mesh.ndim, self.mesh.ndim + 2
        for icl in range(ec.ncell):
            ec.so0n[icl, 0] = 1.1 + 0.1 * icl
            for d in range(nd):
                ec.so0n[icl, 1 + d] = 0.3 + 0.2 * d + 0.05 * icl
            ec.so0n[icl, neq - 1] = 5.0 + 0.1 * icl
        ec.add_slipwall(self._faces())
        ec.bc_soln()
        for ifc, icl, jcl in self._bnd():
            n = self._normal(ifc)
            assert_almost_equal(ec.so0n[jcl, 0], ec.so0n[icl, 0])
            assert_almost_equal(ec.so0n[jcl, neq - 1], ec.so0n[icl, neq - 1])
            momi = np.array([ec.so0n[icl, 1 + d] for d in range(nd)],
                            dtype="float64")
            momg = np.array([ec.so0n[jcl, 1 + d] for d in range(nd)],
                            dtype="float64")
            assert_almost_equal(momg, momi - 2.0 * (momi @ n) * n, decimal=12)
            assert_almost_equal((momi + momg) @ n, 0.0, decimal=12)

    def test_slipwall_do1(self):
        # Slip-wall do1 mirrors the derivatives across the wall: the scalar
        # gradients reflect their normal component and the momentum-gradient
        # tensor is reflected on both indices (R G R, R = I - 2 n n^T).
        ec = self._ec()
        nd, neq = self.mesh.ndim, self.mesh.ndim + 2
        for icl in range(ec.ncell):
            for ieq in range(neq):
                for d in range(nd):
                    ec.so1c[icl, ieq, d] = \
                        0.1 + 0.2 * ieq + 0.13 * d + 0.07 * icl
        ec.add_slipwall(self._faces())
        ec.bc_dsoln()
        eye = np.eye(nd)
        for ifc, icl, jcl in self._bnd():
            n = self._normal(ifc)
            refl = eye - 2.0 * np.outer(n, n)
            for ieq in (0, neq - 1):
                gi = np.array([ec.so1c[icl, ieq, d] for d in range(nd)],
                              dtype="float64")
                gg = np.array([ec.so1n[jcl, ieq, d] for d in range(nd)],
                              dtype="float64")
                assert_almost_equal(gg, refl @ gi, decimal=12)
            gmi = np.array([[ec.so1c[icl, 1 + a, b] for b in range(nd)]
                            for a in range(nd)], dtype="float64")
            gmg = np.array([[ec.so1n[jcl, 1 + a, b] for b in range(nd)]
                            for a in range(nd)], dtype="float64")
            assert_almost_equal(gmg, refl @ gmi @ refl, decimal=12)

    def test_inlet_do0(self):
        # Inlet do0 sets the ghost to the prescribed conserved free stream.
        ec = self._ec()
        nd, neq = self.mesh.ndim, self.mesh.ndim + 2
        val = self._inlet_value(nd)
        ec.add_inlet(self._faces(), value=val)
        ec.bc_soln()
        rho, v = val[0], val[1:1 + nd]
        p, ga = val[1 + nd], val[2 + nd]
        energy = p / (ga - 1.0) + 0.5 * rho * sum(c * c for c in v)
        for ifc, icl, jcl in self._bnd():
            assert_almost_equal(ec.so0n[jcl, 0], rho)
            for d in range(nd):
                assert_almost_equal(ec.so0n[jcl, 1 + d], rho * v[d])
            assert_almost_equal(ec.so0n[jcl, neq - 1], energy)

    def test_inlet_do1(self):
        # Inlet do1 zeroes the ghost gradient.
        ec = self._ec()
        nd, neq = self.mesh.ndim, self.mesh.ndim + 2
        # Pre-seed both interior and ghost so the zeroing is observable.
        for icl in range(ec.ncell):
            for ieq in range(neq):
                for d in range(nd):
                    ec.so1c[icl, ieq, d] = 1.0 + ieq + d
        for ifc, icl, jcl in self._bnd():
            for ieq in range(neq):
                for d in range(nd):
                    ec.so1n[jcl, ieq, d] = 9.0
        ec.add_inlet(self._faces(), value=self._inlet_value(nd))
        ec.bc_dsoln()
        for ifc, icl, jcl in self._bnd():
            for ieq in range(neq):
                for d in range(nd):
                    assert_almost_equal(ec.so1n[jcl, ieq, d], 0.0)

    def test_add_bc_validation(self):
        ec = self._ec()
        nd = self.mesh.ndim
        faces = self._faces()
        # The inlet value must be [rho, v(ndim), p, gamma].
        with self.assertRaises(ValueError):
            ec.add_inlet(faces, value=[1.0] * (nd + 2))
        for bad in ([0.0] + [0.0] * nd + [1.0, 1.4],   # rho <= 0
                    [1.0] + [0.0] * nd + [-1.0, 1.4],  # p < 0
                    [1.0] + [0.0] * nd + [1.0, 1.0]):  # gamma <= 1
            with self.assertRaises(ValueError):
                ec.add_inlet(faces, value=bad)
        # A face index that is not a boundary face is rejected.
        with self.assertRaises(ValueError):
            ec.add_nonrefl([10 ** 7])


class EulerBCTriangleTC(_EulerBCBase, _TriangleMeshBase):
    """EulerCore boundary conditions on 3 triangles."""


class EulerBCQuadTC(_EulerBCBase, _QuadMeshBase):
    """EulerCore boundary conditions on a unit-square quad."""


class EulerBCMixedTC(_EulerBCBase, _MixedMeshBase):
    """EulerCore boundary conditions on a 2D mixed mesh."""


class EulerBCTetrahedronTC(_EulerBCBase, _TetrahedronMeshBase):
    """EulerCore boundary conditions on a single tetrahedron."""


class EulerBCHexahedronTC(_EulerBCBase, _HexahedronMeshBase):
    """EulerCore boundary conditions on a single hexahedron."""


class EulerChannelMarchTC(unittest.TestCase):
    """A walled 2D channel: supersonic inlet, slip walls, and a
    non-reflective outflow -- the first march driven with boundary
    conditions, replacing the all-boundary skip of test_march_bounded."""

    GAMMA = 1.4
    RHO = 1.0
    PRES = 1.0
    VX = 2.0  # supersonic: |v| / sqrt(gamma p / rho) ~ 1.69

    def _classify(self, mh, lx):
        left, right, walls = [], [], []
        for ifc in mh.bndfcs.ndarray[:, 0]:
            ifc = int(ifc)
            cx = mh.fccnd[ifc, 0]
            if abs(cx) < 1e-9:
                left.append(ifc)
            elif abs(cx - lx) < 1e-9:
                right.append(ifc)
            else:
                walls.append(ifc)
        return left, right, walls

    def test_march_channel_bounded(self):
        lx, ly = 4.0, 2.0
        mh = _build_quad_channel(4, 2, lx, ly)
        left, right, walls = self._classify(mh, lx)
        # All three boundary kinds are present.
        self.assertTrue(left)
        self.assertTrue(right)
        self.assertTrue(walls)
        ec = modmesh.EulerCore(mesh=mh, time_increment=0.04)
        ec.init_solution(gamma=self.GAMMA, rho=self.RHO,
                         v=[self.VX, 0.0], p=self.PRES)
        ec.add_inlet(left,
                     value=[self.RHO, self.VX, 0.0, self.PRES, self.GAMMA])
        ec.add_nonrefl(right)
        ec.add_slipwall(walls)
        # Initialize the ghost rows from the initial interior state so the
        # first substep does not read zero-filled ghosts.
        ec.bc_soln()
        ec.bc_dsoln()
        ec.march(steps=5)
        so0n = ec.so0n.ndarray
        self.assertTrue(np.all(np.isfinite(so0n)))
        self.assertLess(np.abs(so0n).max(), 1e6)
        # A uniform stream aligned with the channel is the steady state, so
        # density stays positive and close to the inflow value.
        for icl in range(ec.ncell):
            self.assertGreater(ec.so0n[icl, 0], 0.0)
            assert_almost_equal(ec.so0n[icl, 0], self.RHO, decimal=6)
        # bc_soln runs every substep, so the inlet ghost holds the prescribed
        # free-stream density after marching (fails if the trim pass is not
        # wired into march_substep).
        for ifc in left:
            assert_almost_equal(ec.so0n[mh.fccls[ifc, 1], 0], self.RHO)


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
