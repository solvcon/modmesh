# Copyright (c) 2021, Yung-Yu Chen <yyc@solvcon.net>
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

import modmesh


class StaticMeshTC(unittest.TestCase):

    def _check_shape(self, mh, ndim, nnode, nface, ncell,
                     nbound, ngstnode, ngstface, ngstcell, nedge):
        self.assertEqual(ndim, mh.ndim)
        self.assertEqual(nnode, mh.nnode)
        self.assertEqual(nface, mh.nface)
        self.assertEqual(ncell, mh.ncell)
        self.assertEqual(nbound, mh.nbound)
        self.assertEqual(ngstnode, mh.ngstnode)
        self.assertEqual(ngstface, mh.ngstface)
        self.assertEqual(ngstcell, mh.ngstcell)

        self.assertEqual(nedge, mh.nedge)

        self.assertEqual((mh.ngstnode + mh.nnode, ndim), mh.ndcrd.shape)
        self.assertEqual((mh.ngstface + mh.nface, ndim), mh.fccnd.shape)
        self.assertEqual((mh.ngstface + mh.nface, ndim), mh.fcnml.shape)
        self.assertEqual((mh.ngstface + mh.nface,), mh.fcara.shape)
        self.assertEqual((mh.ngstcell + mh.ncell, ndim), mh.clcnd.shape)
        self.assertEqual((mh.ngstcell + mh.ncell,), mh.clvol.shape)

        self.assertEqual((mh.ngstface + mh.nface,), mh.fctpn.shape)
        self.assertEqual((mh.ngstcell + mh.ncell,), mh.cltpn.shape)
        self.assertEqual((mh.ngstcell + mh.ncell,), mh.clgrp.shape)

        self.assertEqual((mh.ngstface + mh.nface, mh.FCMND+1), mh.fcnds.shape)
        self.assertEqual((mh.ngstface + mh.nface, mh.FCREL), mh.fccls.shape)
        self.assertEqual((mh.ngstcell + mh.ncell, mh.CLMND+1), mh.clnds.shape)
        self.assertEqual((mh.ngstcell + mh.ncell, mh.CLMFC+1), mh.clfcs.shape)

    def _check_metric_trivial(self, mh):
        self.assertTrue((mh.fccnd.ndarray[:, :] == 0).all())
        self.assertTrue((mh.fcnml.ndarray[:, :] == 0).all())
        self.assertTrue((mh.fcara.ndarray[:] == 0).all())
        self.assertTrue((mh.clcnd.ndarray[:, :] == 0).all())
        self.assertTrue((mh.clvol.ndarray[:] == 0).all())

    def test_construct(self):
        def _test(cls, ndim):
            mh = cls(ndim=ndim, nnode=0)
            self._check_shape(mh, ndim=ndim, nnode=0, nface=0, ncell=0,
                              nbound=0, ngstnode=0, ngstface=0, ngstcell=0,
                              nedge=0)

        _test(modmesh.StaticMesh, ndim=2)
        _test(modmesh.StaticMesh, ndim=3)

    def test_2d_trivial_triangles(self):
        mh = modmesh.StaticMesh(ndim=2, nnode=4, nface=0, ncell=3)
        mh.ndcrd.ndarray[:, :] = (0, 0), (-1, -1), (1, -1), (0, 1)
        mh.cltpn.ndarray[:] = modmesh.StaticMesh.TRIANGLE
        mh.clnds.ndarray[:, :4] = (3, 0, 1, 2), (3, 0, 2, 3), (3, 0, 3, 1)

        self._check_shape(mh, ndim=2, nnode=4, nface=0, ncell=3,
                          nbound=0, ngstnode=0, ngstface=0, ngstcell=0,
                          nedge=0)
        self._check_metric_trivial(mh)

        # Test build interior data.
        mh.build_interior(_do_metric=False, _build_edge=False)
        self._check_shape(mh, ndim=2, nnode=4, nface=6, ncell=3,
                          nbound=0, ngstnode=0, ngstface=0, ngstcell=0,
                          nedge=0)
        self._check_metric_trivial(mh)
        mh.build_interior()  # _do_metric=True, _build_edge=True
        self._check_shape(mh, ndim=2, nnode=4, nface=6, ncell=3,
                          nbound=0, ngstnode=0, ngstface=0, ngstcell=0,
                          nedge=6)
        np.testing.assert_almost_equal(
            mh.fccnd,
            [[-0.5, -0.5], [0.0, -1.0], [0.5, -0.5],
             [0.5, 0.0], [0.0, 0.5], [-0.5, 0.0]])
        np.testing.assert_almost_equal(
            mh.fcnml,
            [[-0.7071068, 0.7071068], [0.0, -1.0], [0.7071068, 0.7071068],
             [0.8944272, 0.4472136], [-1.0, -0.0], [-0.8944272, 0.4472136]])
        np.testing.assert_almost_equal(
            mh.fcara, [1.4142136, 2.0, 1.4142136, 2.236068, 1.0, 2.236068])
        np.testing.assert_almost_equal(
            mh.clcnd, [[0.0, -0.6666667], [0.3333333, 0.0], [-0.3333333, 0.0]])
        np.testing.assert_almost_equal(
            mh.clvol, [1.0, 0.5, 0.5])

        # Build boundary data.
        self.assertEqual(0, mh.nbcs)
        self.assertEqual(0, mh.nbound)
        self.assertEqual((mh.nbound, 3), mh.bndfcs.shape)
        mh.build_boundary()
        self.assertEqual(1, mh.nbcs)
        self.assertEqual(3, mh.nbound)
        self.assertEqual((mh.nbound, 3), mh.bndfcs.shape)
        self.assertEqual(
            [[1, 0, -1], [3, 0, -1], [5, 0, -1]],
            mh.bndfcs.ndarray.tolist()
        )

        # Build ghost data.
        self._check_shape(mh, ndim=2, nnode=4, nface=6, ncell=3,
                          nbound=3, ngstnode=0, ngstface=0, ngstcell=0,
                          nedge=6)
        mh.build_ghost()
        self._check_shape(mh, ndim=2, nnode=4, nface=6, ncell=3,
                          nbound=3, ngstnode=3, ngstface=6, ngstcell=3,
                          nedge=6)

    def test_3d_single_tetrahedron(self):
        mh = modmesh.StaticMesh(ndim=3, nnode=4, nface=4, ncell=1)
        mh.ndcrd.ndarray[:, :] = (0, 0, 0), (0, 1, 0), (-1, 1, 0), (0, 1, 1)
        mh.cltpn.ndarray[:] = modmesh.StaticMesh.TETRAHEDRON
        mh.clnds.ndarray[:, :5] = [(4, 0, 1, 2, 3)]

        self._check_shape(mh, ndim=3, nnode=4, nface=4, ncell=1,
                          nbound=0, ngstnode=0, ngstface=0, ngstcell=0,
                          nedge=0)
        self._check_metric_trivial(mh)

        mh.build_interior(_do_metric=False, _build_edge=False)
        self._check_shape(mh, ndim=3, nnode=4, nface=4, ncell=1,
                          nbound=0, ngstnode=0, ngstface=0, ngstcell=0,
                          nedge=0)
        self._check_metric_trivial(mh)
        mh.build_interior()  # _do_metric=True, _build_edge=True
        self._check_shape(mh, ndim=3, nnode=4, nface=4, ncell=1,
                          nbound=0, ngstnode=0, ngstface=0, ngstcell=0,
                          nedge=6)
        # TODO: I have not verified the numbers.  But the mesh looks OK in
        # viewer.
        np.testing.assert_almost_equal(
            mh.fccnd,
            [[-0.3333333,  0.6666667,  0.       ],  # noqa
             [ 0.       ,  0.6666667,  0.3333333],  # noqa
             [-0.3333333,  0.6666667,  0.3333333],  # noqa
             [-0.3333333,  1.       ,  0.3333333]])  # noqa
        np.testing.assert_almost_equal(
            mh.fcnml,
            [[ 0.       ,  0.       , -1.       ],  # noqa
             [ 1.       ,  0.       ,  0.       ],  # noqa
             [-0.5773503, -0.5773503,  0.5773503],  # noqa
             [ 0.       ,  1.       ,  0.       ]])  # noqa
        np.testing.assert_almost_equal(
            mh.fcara, [0.5      , 0.5      , 0.8660254, 0.5      ])  # noqa
        np.testing.assert_almost_equal(
            mh.clcnd, [[-0.25,  0.75,  0.25]])
        np.testing.assert_almost_equal(
            mh.clvol, [0.1666667])

        # Build boundary data.
        self.assertEqual(0, mh.nbcs)
        self.assertEqual(0, mh.nbound)
        self.assertEqual((mh.nbound, 3), mh.bndfcs.shape)
        mh.build_boundary()
        self.assertEqual(1, mh.nbcs)
        self.assertEqual(4, mh.nbound)
        self.assertEqual((mh.nbound, 3), mh.bndfcs.shape)
        self.assertEqual(
            [[0, 0, -1], [1, 0, -1], [2, 0, -1], [3, 0, -1]],
            mh.bndfcs.ndarray.tolist()
        )

        # Build ghost data.
        self._check_shape(mh, ndim=3, nnode=4, nface=4, ncell=1,
                          nbound=4, ngstnode=0, ngstface=0, ngstcell=0,
                          nedge=6)
        mh.build_ghost()
        self._check_shape(mh, ndim=3, nnode=4, nface=4, ncell=1,
                          nbound=4, ngstnode=4, ngstface=12, ngstcell=4,
                          nedge=6)

    def test_1d_single_line(self):
        mh = modmesh.StaticMesh(ndim=1, nnode=2, nface=0, ncell=1)
        mh.ndcrd.ndarray[:] = [[0], [1]]
        mh.cltpn.ndarray[:] = modmesh.StaticMesh.LINE
        mh.clnds.ndarray[:, :3] = [(2, 0, 1)]

        self._check_shape(mh, ndim=1, nnode=2, nface=0, ncell=1,
                          nbound=0, ngstnode=0, ngstface=0, ngstcell=0,
                          nedge=0)
        self._check_metric_trivial(mh)

        mh.build_interior(_do_metric=False, _build_edge=False)
        self._check_shape(mh, ndim=1, nnode=2, nface=2, ncell=1,
                          nbound=0, ngstnode=0, ngstface=0, ngstcell=0,
                          nedge=0)
        self._check_metric_trivial(mh)

        mh.build_interior()  # _do_metric=True, _build_edge=True
        self._check_shape(mh, ndim=1, nnode=2, nface=2, ncell=1,
                          nbound=0, ngstnode=0, ngstface=0, ngstcell=0,
                          nedge=2)

        # _do_metric do nothing due to dim == 1
        self._check_metric_trivial(mh)
        # TODO: Need to add build_boundary and build_ghost to make sure
        #       Line type behavior.
# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
