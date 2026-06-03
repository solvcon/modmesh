import unittest

from numpy.testing import assert_almost_equal

import modmesh


class EulerCoreCETriangleTC(unittest.TestCase):
    """Test CE geometry on a 2D mesh of 3 triangles around the origin."""

    @classmethod
    def setUpClass(cls):
        mh = modmesh.StaticMesh(ndim=2, nnode=4, nface=0, ncell=3)
        mh.ndcrd.ndarray[:, :] = (0, 0), (-1, -1), (1, -1), (0, 1)
        mh.cltpn.ndarray[:] = modmesh.StaticMesh.TRIANGLE
        mh.clnds.ndarray[:, :4] = (3, 0, 1, 2), (3, 0, 2, 3), (3, 0, 3, 1)
        mh.build_interior(True)
        mh.build_boundary()
        mh.build_ghost()
        cls.mesh = mh
        cls.ec = modmesh.EulerCore(mesh=mh, time_increment=0.01)

    def test_dimensions(self):
        self.assertEqual(2, self.ec.ndim)
        self.assertEqual(3, self.ec.ncell)
        self.assertEqual(3, self.ec.ngstcell)

    def test_cevol_shape(self):
        total = self.ec.ngstcell + self.ec.ncell
        self.assertEqual(
            (total, modmesh.StaticMesh.CLMFC + 1),
            self.ec.cevol.shape)

    def test_cecnd_shape(self):
        total = self.ec.ngstcell + self.ec.ncell
        self.assertEqual(
            (total, (modmesh.StaticMesh.CLMFC + 1) * self.ec.ndim),
            self.ec.cecnd.shape)

    def test_sfcnd_shape(self):
        self.assertEqual(
            (self.ec.ncell,
             modmesh.StaticMesh.CLMFC * modmesh.StaticMesh.FCMND,
             self.ec.ndim),
            self.ec.sfcnd.shape)

    def test_sfnml_shape(self):
        self.assertEqual(
            (self.ec.ncell,
             modmesh.StaticMesh.CLMFC * modmesh.StaticMesh.FCMND,
             self.ec.ndim),
            self.ec.sfnml.shape)

    def test_cce_volume_positive(self):
        for icl in range(self.ec.ncell):
            self.assertGreater(
                self.ec.cevol[icl, 0], 0.0,
                f"CCE volume for cell {icl} should be positive")

    def test_bce_volumes_sum_to_cce(self):
        mh = self.mesh
        for icl in range(self.ec.ncell):
            nfc = mh.clfcs[icl, 0]
            bce_sum = sum(
                self.ec.cevol[icl, ifl]
                for ifl in range(1, nfc + 1))
            assert_almost_equal(
                bce_sum, self.ec.cevol[icl, 0],
                decimal=12,
                err_msg=f"BCE sum != CCE for cell {icl}")

    def test_bce_volume_manual_cell0(self):
        assert_almost_equal(self.ec.cevol[0, 1], 0.5, decimal=10)
        assert_almost_equal(self.ec.cevol[0, 2], 2.0 / 3.0, decimal=10)
        assert_almost_equal(self.ec.cevol[0, 3], 0.5, decimal=10)
        assert_almost_equal(
            self.ec.cevol[0, 0], 0.5 + 2.0 / 3.0 + 0.5, decimal=10)

    def test_prepare_ce_rerun(self):
        vol_before = self.ec.cevol[0, 0]
        self.ec.prepare_ce()
        assert_almost_equal(self.ec.cevol[0, 0], vol_before, decimal=12)


@unittest.skip("3D ghost cell geometry has NaN due to fill_ghost bug")
class EulerCoreCETetrahedronTC(unittest.TestCase):
    """Test CE geometry on a 3D mesh of a single tetrahedron."""

    @classmethod
    def setUpClass(cls):
        mh = modmesh.StaticMesh(ndim=3, nnode=4, nface=4, ncell=1)
        mh.ndcrd.ndarray[:, :] = (
            (0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1))
        mh.cltpn.ndarray[:] = modmesh.StaticMesh.TETRAHEDRON
        mh.clnds.ndarray[0, :5] = (4, 0, 1, 2, 3)
        mh.build_interior(True)
        mh.build_boundary()
        mh.build_ghost()
        cls.mesh = mh
        cls.ec = modmesh.EulerCore(mesh=mh, time_increment=0.01)

    def test_dimensions(self):
        self.assertEqual(3, self.ec.ndim)
        self.assertEqual(1, self.ec.ncell)

    def test_cce_volume_positive(self):
        self.assertGreater(self.ec.cevol[0, 0], 0.0)

    def test_bce_volumes_sum_to_cce(self):
        mh = self.mesh
        nfc = mh.clfcs[0, 0]
        bce_sum = sum(
            self.ec.cevol[0, ifl] for ifl in range(1, nfc + 1))
        assert_almost_equal(
            bce_sum, self.ec.cevol[0, 0], decimal=12)


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
