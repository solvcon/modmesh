# Copyright (c) 2024, Yung-Yu Chen <yyc@solvcon.net>
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

import os

import unittest

import numpy as np

import modmesh as mm


class GmshTB(unittest.TestCase):
    TESTDIR = os.path.abspath(os.path.dirname(__file__))
    DATADIR = os.path.join(TESTDIR, "data")


class GmshTriangleTC(GmshTB):
    def test_gmsh_parsing(self):
        path = os.path.join(self.DATADIR, "gmsh_triangle.msh")

        data = open(path, 'rb').read()
        gmsh = mm.Gmsh(data)
        blk = gmsh.to_block()

        # Check nodes information
        self.assertEqual(blk.nnode, 4)

        # Due to ghost cell and ghost node had been created, the real body
        # had been shifted and start with index 3 (number of ghost)
        ngst = blk.ngstcell
        self.assertEqual(ngst, 3)
        np.testing.assert_almost_equal(blk.ndcrd.ndarray[ngst:, :].tolist(),
                                       [[0.0, 0.0],
                                        [-1.0, -1.0],
                                        [1.0, -1.0],
                                        [0.0, 1.0]])
        # Check cells information
        self.assertEqual(blk.ncell, 3)
        self.assertEqual(blk.cltpn.ndarray[ngst:].tolist(), [4, 4, 4])
        self.assertEqual(blk.clnds.ndarray[ngst:, :4].tolist(),
                         [[3, 0, 1, 2],
                          [3, 0, 2, 3],
                          [3, 0, 3, 1]])


class GmshRectangularTC(GmshTB):
    def setUp(self):
        # Read the Gmsh mesh file.
        path = os.path.join(self.DATADIR, "rectangle.msh")
        with open(path, 'rb') as fobj:
            data = fobj.read()
        # Create the Gmsh object.
        gmsh = mm.Gmsh(data)
        # Convert the Gmsh object to StaticMesh object.
        self.blk = gmsh.to_block()

    def test_shape(self):
        blk = self.blk
        # Test for shape data.
        self.assertEqual(blk.ndim, 2)
        self.assertEqual(blk.nbound, 40)
        self.assertEqual(blk.nnode, 104)
        self.assertEqual(blk.nface, 309)
        self.assertEqual(blk.ncell, 206)
        self.assertEqual(blk.ngstnode, 40)
        self.assertEqual(blk.ngstface, 80)
        self.assertEqual(blk.ngstcell, 40)

    @unittest.expectedFailure
    def test_type(self):
        blk = self.blk
        # TODO: all cell should be triangles including ghost cells.
        np.testing.assert_equal(blk.cltpn, 4)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
