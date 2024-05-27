# Copyright (c) 2024, Chunhsu Lai <as2266317@gmail.com>
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# - Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
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

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
