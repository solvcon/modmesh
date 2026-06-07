import unittest

import numpy as np
from numpy.testing import assert_almost_equal

import modmesh


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
                            for ifl in range(ge.clnfc)])
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
                mat = np.array(ge.displacement_matrix(ifge))
                self.assertEqual((nd, nd), mat.shape)
                self.assertGreater(
                    abs(np.linalg.det(mat)), 1e-10,
                    f"cell {icl} ifge {ifge}: singular FGE matrix")

    def test_solve_gradient_linear_field(self):
        # For a linear field u(x) = c + g . x the solution delta at each
        # gradient evaluation point is exactly g . idis, so the per-FGE
        # solve must recover g exactly (up to round-off).
        nd = self.mesh.ndim
        grad = np.array([1.5, -2.7, 0.9][:nd])
        for icl in range(self.mesh.ncell):
            ge = self._ge(icl)
            for ifge in range(ge.nfge):
                dst = np.array(ge.displacement_matrix(ifge))
                faces = ge.faces(ifge)
                # Matrix rows are the per-face idis vectors.
                for ivx, ifl in enumerate(faces):
                    for d in range(nd):
                        assert_almost_equal(
                            dst[ivx, d], ge.idis(ifl - 1, d), decimal=12)
                udf = dst @ grad
                got = np.array(ge.solve_gradient(ifge, udf.tolist()))
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
        assert_almost_equal(ec.gamma.ndarray, np.full(total, self.GAMMA))
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
        so1n_view[...] = np.arange(so1n_view.size).reshape(so1n_view.shape)
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


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
