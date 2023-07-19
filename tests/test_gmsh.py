import os

import unittest

import numpy as np

import modmesh


class GmshTC(unittest.TestCase):

    def test_gmsh_parsing(self):
        path = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                            "data/gmsh_triangle.msh")
        gmsh_instance = modmesh.core.Gmsh(path)
        blk = gmsh_instance.toblock()

        # Check nodes information
        self.assertEqual(blk.nnode, 4)

        # Due to ghost cell and ghost node had been created, the real body
        # had been shifted and start with index 3
        np.testing.assert_almost_equal(blk.ndcrd.ndarray[3:, :].tolist(),
                                       [[0.0, 0.0],
                                        [-1.0, -1.0],
                                        [1.0, -1.0],
                                        [0.0, 1.0]])
        # Check cells information
        self.assertEqual(blk.ncell, 3)
        self.assertEqual(blk.cltpn.ndarray[3:].tolist(), [4, 4, 4])
        self.assertEqual(blk.clnds.ndarray[3:, :4].tolist(), [[3, 0, 1, 2],
                                                              [3, 0, 2, 3],
                                                              [3, 0, 3, 1]])
