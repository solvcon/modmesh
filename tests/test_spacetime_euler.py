# Copyright (c) 2018, Yung-Yu Chen <yyc@solvcon.net>
# BSD 3-Clause License, see COPYING

import unittest

import numpy as np

from modmesh import spacetime as libst


class EulerSolverTC(unittest.TestCase):

    @staticmethod
    def _build_solver(resolution):

        # Build grid.
        xcrd = np.arange(resolution+1, dtype='float64') / resolution
        xcrd *= 2 * np.pi
        grid = libst.Grid(xcrd)
        dx = (grid.xmax - grid.xmin) / grid.ncelm

        # Build solver.
        time_stop = 2 * np.pi
        cfl_max = 1.0
        dt_max = dx * cfl_max
        nstep = int(np.ceil(time_stop / dt_max))
        dt = time_stop / nstep
        svr = libst.EulerSolver(grid=grid, time_increment=dt)

        # Initialize.
        svr.set_so0(0, np.zeros(resolution+1, dtype='float64'))
        svr.set_so0(1, np.zeros(resolution+1, dtype='float64'))
        svr.set_so0(2, np.zeros(resolution+1, dtype='float64'))
        svr.set_so1(0, np.zeros(resolution+1, dtype='float64'))
        svr.set_so1(1, np.zeros(resolution+1, dtype='float64'))
        svr.set_so1(2, np.zeros(resolution+1, dtype='float64'))
        svr.setup_march()

        return nstep, xcrd, svr

    def setUp(self):

        self.resolution = 8
        self.nstep, self.xcrd, self.svr = self._build_solver(self.resolution)
        self.cycle = 10

    def test_xctr(self):

        # On even plane.
        self.assertEqual(len(self.svr.xctr()), self.svr.grid.ncelm+1)
        self.assertEqual(self.svr.xctr().tolist(), self.xcrd.tolist())

        # On odd plane.
        self.assertEqual(self.svr.xctr().tolist(), self.xcrd.tolist())

    def test_nvar(self):

        self.assertEqual(3, self.svr.field.nvar)

    def test_array_getter(self):

        v = self.svr.get_so0(0).tolist()
        self.assertEqual(self.svr.grid.ncelm+1, len(v))

        with self.assertRaisesRegex(IndexError, "out of nvar range"):
            self.svr.get_so0(3)
        with self.assertRaisesRegex(IndexError, "out of nvar range"):
            self.svr.get_so0(3, odd_plane=True)

        v = self.svr.get_so1(0).tolist()
        self.assertEqual(self.svr.grid.ncelm+1, len(v))

        with self.assertRaisesRegex(IndexError, "out of nvar range"):
            self.svr.get_so1(3)
        with self.assertRaisesRegex(IndexError, "out of nvar range"):
            self.svr.get_so1(3, odd_plane=True)

        # The odd-plane value is uninitialized before marching.
        self.svr.march_alpha2(steps=1)

        v = self.svr.get_so0(0, odd_plane=True).tolist()
        self.assertEqual(self.svr.grid.ncelm, len(v))

        v = self.svr.get_so1(0, odd_plane=True).tolist()
        self.assertEqual(self.svr.grid.ncelm, len(v))

    def test_initialized(self):

        self.assertEqual(
            self.svr.get_so0(0).tolist(),
            np.zeros(self.resolution + 1, dtype='float64').tolist())
        self.assertEqual(
            self.svr.get_so1(0).tolist(),
            np.zeros(self.resolution + 1, dtype='float64').tolist())

    def test_march_fine_interface(self):

        def _march():

            # first half step.
            self.svr.march_half_so0(odd_plane=False)
            self.svr.treat_boundary_so0()
            self.svr.update_cfl(odd_plane=True)
            self.svr.march_half_so1_alpha2(odd_plane=False)
            self.svr.treat_boundary_so1()
            # second half step.
            self.svr.march_half_so0(odd_plane=True)
            self.svr.update_cfl(odd_plane=False)
            self.svr.march_half_so1_alpha2(odd_plane=True)

        svr2 = self._build_solver(self.resolution)[-1]

        for it in range(self.nstep*self.cycle):
            _march()
            svr2.march_alpha2(steps=1)
            self.assertEqual(self.svr.get_so0(0).tolist(),
                             svr2.get_so0(0).tolist())


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
