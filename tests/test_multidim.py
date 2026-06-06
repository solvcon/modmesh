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


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
