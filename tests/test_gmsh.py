import os

import unittest

import numpy as np

import modmesh


class GmshTC(unittest.TestCase):

    def test_gmsh_parsing(self):
        path = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                            "data", "gmsh_triangle.msh")

        data = open(path, 'rb').read()
        gmsh_instance = modmesh.core.Gmsh(data)
        blk = gmsh_instance.to_block()

        # Check nodes information
        self.assertEqual(blk.nnode, 4)

        # Due to ghost cell and ghost node had been created, the real body
        # had been shifted and start with index 3
        np.testing.assert_almost_equal(blk.ndcrd[3:, :].ndarray.tolist(),
                                       [[0.0, 0.0],
                                        [-1.0, -1.0],
                                        [1.0, -1.0],
                                        [0.0, 1.0]])
        # Check cells information
        self.assertEqual(blk.ncell, 3)
        self.assertEqual(blk.cltpn[3:].ndarray.tolist(), [4, 4, 4])
        self.assertEqual(blk.clnds[3:, :4].ndarray.tolist(), [[3, 0, 1, 2],
                                                              [3, 0, 2, 3],
                                                              [3, 0, 3, 1]])
