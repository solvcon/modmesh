import os

import unittest

import numpy as np

import modmesh


class Plot3dTC(unittest.TestCase):

    def test_plot3d_parsing(self):
        filepath = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                            "data", "test.p3d")

        plot3d_instance = modmesh.core.Plot3d(filepath)
        blk = plot3d_instance

        # Check nodes information
        self.assertEqual(blk.nnode, 8)

        # Due to ghost cell and ghost node had been created, the real body
        # had been shifted and start with index 3
        np.testing.assert_almost_equal(blk.ndcrd.ndarray[:, :].tolist(),
                                       [[0.0, 0.0, 0.0],
                                        [0.0, 0.0, 1.0],
                                        [0.0, 1.0, 0.0],
                                        [0.0, 1.0, 1.0],
                                        [1.0, 0.0, 0.0],
                                        [1.0, 0.0, 1.0],
                                        [1.0, 1.0, 0.0],
                                        [1.0, 1.0, 1.0]])
        # Check cells information
        self.assertEqual(blk.ncell, 1)