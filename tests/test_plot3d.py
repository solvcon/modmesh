import os
import unittest
import numpy as np
import modmesh

class Plot3dTC(unittest.TestCase):
    def test_load_file_success(self):
        filepath = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                "data", "test.p3d")
        plot3d_instance = modmesh.core.Plot3d()  
        try:
            blk=plot3d_instance.load_file(filepath)
            self.assertIsNotNone(blk)
        except Exception as e:
            self.fail("load_file() raised Exception unexpectedly!")
            
    def test_load_file_fail(self):
        filepath = "wrong path/test.p3d"
        plot3d_instance = modmesh.core.Plot3d()  
        with self.assertRaises(RuntimeError):
            blk = plot3d_instance.load_file(filepath)

    def test_plot3d_parsing(self):
        filepath = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                "data", "test.p3d")
       
        plot3d_instance = modmesh.core.Plot3d()  
        blk = plot3d_instance.load_file(filepath) 

        # Check nodes information
        self.assertEqual(blk.nnode, 8)

        # Use assert_allclose for comparing floating point arrays
        np.testing.assert_allclose(blk.ndcrd.ndarray[:, :], 
                                   [[0.0, 0.0, 0.0],
                                    [0.0, 0.0, 1.0],
                                    [0.0, 1.0, 0.0],
                                    [0.0, 1.0, 1.0],
                                    [1.0, 0.0, 0.0],
                                    [1.0, 0.0, 1.0],
                                    [1.0, 1.0, 0.0],
                                    [1.0, 1.0, 1.0]], 
                                   rtol=1e-07, atol=0, equal_nan=True)

        # Check cells information
        self.assertEqual(blk.ncell, 6)
    def _check_shape(self, mh, ndim, nnode, nface, ncell):
        self.assertEqual(ndim, mh.ndim)
        self.assertEqual(nnode, mh.nnode)
        self.assertEqual(nface, mh.nface)
        self.assertEqual(ncell, mh.ncell)

    def test_construct(self):
        def _test(cls, ndim):
            mh = cls(ndim=ndim, nnode=8)
            self._check_shape(mh, ndim=ndim, nnode=8, nface=0, ncell=0)

        _test(modmesh.StaticMesh, ndim=3)

    def test_3d_single_hexahedron(self):
        mh = modmesh.StaticMesh(ndim=3, nnode=8, nface=6, ncell=6)
        mh.ndcrd.ndarray[:, :] = (0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)
        mh.cltpn.ndarray[:] = modmesh.StaticMesh.HEXAHEDRON
        mh.clnds.ndarray[:, :9] = [(8, 0 ,2 ,3 ,1 ,4 ,6 ,7 ,5)]

        self._check_shape(mh, ndim=3, nnode=8, nface=6, ncell=6)

