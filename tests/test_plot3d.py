import unittest

import numpy as np

import modmesh


class Plot3dTC(unittest.TestCase):

    def test_plot3d_parsing(self):

        data = """1
2 2 2
0 0 0 0 1 1 1 1
0 0 1 1 0 0 1 1
0 1 0 1 0 1 0 1
"""
        plot3d_instance = modmesh.core.Plot3d(data.encode('utf-8'))
        blk = plot3d_instance.to_block()

        # Check nodes information
        self.assertEqual(blk.nnode, 8)
        # Due to ghost cell and ghost node had been created, the real body
        # had been shifted and start with index 24
        np.testing.assert_almost_equal(blk.ndcrd.ndarray[24:, :].tolist(),
                                       [[0.0, 0.0, 0.0],
                                        [0.0, 0.0, 1.0],
                                        [0.0, 1.0, 0.0],
                                        [0.0, 1.0, 1.0],
                                        [1.0, 0.0, 0.0],
                                        [1.0, 0.0, 1.0],
                                        [1.0, 1.0, 0.0],
                                        [1.0, 1.0, 1.0],
                                        ])
        # Check cells information
        self.assertEqual(blk.ncell, 1)
        self.assertEqual(blk.cltpn.ndarray[6:].tolist(), [5])
        self.assertEqual(blk.clnds.ndarray[6:, :].tolist(),
                         [[8, 0, 2, 6, 4, 1, 3, 7, 5]])
