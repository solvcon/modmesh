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

import modmesh


class StaticMeshTC(unittest.TestCase):

    def test_construct(self):
        def _test(cls, ndim):

            mh = cls(nnode=0)

            self.assertEqual(ndim, cls.NDIM)

            self.assertEqual(0, mh.nnode)
            self.assertEqual(0, mh.nface)
            self.assertEqual(0, mh.ncell)
            self.assertEqual(0, mh.nbound)
            self.assertEqual(0, mh.ngstnode)
            self.assertEqual(0, mh.ngstface)
            self.assertEqual(0, mh.ngstcell)

            self.assertEqual((mh.nnode, ndim), mh.ndcrd.shape)
            self.assertEqual((mh.nface, ndim), mh.fccnd.shape)
            self.assertEqual((mh.nface, ndim), mh.fcnml.shape)
            self.assertEqual((mh.nface,), mh.fcara.shape)
            self.assertEqual((mh.ncell, ndim), mh.clcnd.shape)
            self.assertEqual((mh.ncell,), mh.clvol.shape)

            self.assertEqual((mh.ncell,), mh.fctpn.shape)
            self.assertEqual((mh.ncell,), mh.cltpn.shape)
            self.assertEqual((mh.ncell,), mh.clgrp.shape)

            self.assertEqual((mh.nface, cls.FCMND), mh.fcnds.shape)
            self.assertEqual((mh.nface, cls.FCMCL), mh.fccls.shape)
            self.assertEqual((mh.ncell, cls.CLMND), mh.clnds.shape)
            self.assertEqual((mh.ncell, cls.CLMFC), mh.clfcs.shape)

        _test(modmesh.StaticMesh2d, 2)
        _test(modmesh.StaticMesh3d, 3)

    def test_2d_trivial_triangles(self):
        mh = modmesh.StaticMesh2d(nnode=4, nface=6, ncell=3, nbound=3)
        mh.ndcrd.ndarray[:, :] = (0, 0), (-1, -1), (1, -1), (0, 1)
        mh.cltpn.ndarray[:] = 3
        mh.clnds.ndarray[:, :4] = (3, 0, 1, 2), (3, 0, 2, 3), (3, 0, 3, 1)
        # FIXME: Need to build interior, boundary, and ghost data.

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
