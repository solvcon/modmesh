# Copyright (c) 2018, Yung-Yu Chen <yyc@solvcon.net>
# BSD 3-Clause License, see COPYING

import unittest

import numpy as np

from modmesh import spacetime as libst

import math


class LinearScalarSolverTC(unittest.TestCase):

    @staticmethod
    def _build_solver(resolution):

        # Build grid.
        xcrd = np.arange(resolution+1) / resolution
        xcrd *= 2 * np.pi
        grid = libst.Grid(xcrd)
        dx = (grid.xmax - grid.xmin) / grid.ncelm

        # Build solver.
        time_stop = 2*np.pi
        cfl_max = 1.0
        dt_max = dx * cfl_max
        nstep = int(np.ceil(time_stop / dt_max))
        dt = time_stop / nstep
        svr = libst.LinearScalarSolver(grid=grid, time_increment=dt)

        # Initialize.
        svr.set_so0(0, np.sin(xcrd))
        svr.set_so1(0, np.cos(xcrd))
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
        self.assertEqual(self.svr.xctr().tolist(),
                         [e.xctr for e in self.svr.selms(odd_plane=False)])

        # On odd plane.
        self.assertEqual(len(self.svr.xctr(odd_plane=True)),
                         self.svr.grid.ncelm)
        self.assertEqual(self.svr.xctr().tolist(), self.xcrd.tolist())
        self.assertEqual(self.svr.xctr(odd_plane=True).tolist(),
                         [e.xctr for e in self.svr.selms(odd_plane=True)])

    def test_nvar(self):

        self.assertEqual(1, self.svr.nvar)

    def test_array_getter(self):

        v1 = [e.get_so0(0) for e in self.svr.selms(odd_plane=False)]
        v2 = self.svr.get_so0(0).ndarray.tolist()
        self.assertEqual(self.svr.grid.ncelm+1, len(v2))
        self.assertEqual(v1, v2)

        with self.assertRaisesRegex(IndexError, "out of nvar range"):
            self.svr.get_so0(1)
        with self.assertRaisesRegex(IndexError, "out of nvar range"):
            self.svr.get_so0(1, odd_plane=True)

        v1 = [e.get_so1(0) for e in self.svr.selms(odd_plane=False)]
        v2 = self.svr.get_so1(0).ndarray.tolist()
        self.assertEqual(self.svr.grid.ncelm+1, len(v2))
        self.assertEqual(v1, v2)

        with self.assertRaisesRegex(IndexError, "out of nvar range"):
            self.svr.get_so1(1)
        with self.assertRaisesRegex(IndexError, "out of nvar range"):
            self.svr.get_so1(1, odd_plane=True)

        # The odd-plane value is uninitialized before marching.
        self.svr.march_alpha2(steps=1)

        v1 = [e.get_so0(0) for e in self.svr.selms(odd_plane=True)]
        v2 = self.svr.get_so0(0, odd_plane=True).ndarray.tolist()
        self.assertEqual(self.svr.grid.ncelm, len(v2))
        self.assertEqual(v1, v2)

        v1 = [e.get_so1(0) for e in self.svr.selms(odd_plane=True)]
        v2 = self.svr.get_so1(0, odd_plane=True).ndarray.tolist()
        self.assertEqual(self.svr.grid.ncelm, len(v2))
        self.assertEqual(v1, v2)

    def test_initialized(self):

        self.assertEqual(self.svr.get_so0(0).ndarray.tolist(),
                         np.sin(self.xcrd).tolist())
        self.assertEqual(self.svr.get_so1(0).ndarray.tolist(),
                         np.cos(self.xcrd).tolist())

    def test_march(self):

        self.svr.march_alpha2(self.nstep*self.cycle)
        np.testing.assert_allclose(self.svr.get_so0(0), np.sin(self.xcrd),
                                   rtol=0, atol=1.e-14)
        ones = np.ones(self.svr.grid.nselm, dtype='float64')
        np.testing.assert_allclose(self.svr.get_cfl(), ones,
                                   rtol=0, atol=1.e-14)

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
            self.assertEqual(self.svr.get_so0(0).ndarray.tolist(),
                             svr2.get_so0(0).ndarray.tolist())


class LinearScalarGridTestTC(unittest.TestCase):
    """
    Compare linear solver's solution wtih exact solution,
    because solutions of computation field is a vector,
    therefore compared the norm of difference bewteen solver's solution
    and exact solution.

    By comparing norm of difference bewteen solver's solution and exact
    solution we can check if the linear solver is work properly
    or check if the solver's mathematical model is correct.
    """

    @staticmethod
    def _build_solver(resolution):
        grid = libst.Grid(0, 4 * 2 * np.pi, resolution)
        cfl = 0.99
        dx = (grid.xmax - grid.xmin) / grid.ncelm
        dt = dx * cfl
        svr = libst.LinearScalarSolver(grid=grid, time_increment=dt)

        # Initialize
        for e in svr.selms(odd_plane=False):
            if e.xctr < 2 * np.pi or e.xctr > 4 * np.pi:
                v = 0
                dv = 0
            else:
                v = np.sin(e.xctr)
                dv = np.cos(e.xctr)
            e.set_so0(0, v)
            e.set_so1(0, dv)

        svr.setup_march()

        return svr

    @staticmethod
    def _exact_solution(svr, iter_num, gfun):
        v = []
        for e in svr.selms(odd_plane=False):
            x = (
                e.xctr - iter_num * svr.dt
            ) % svr.grid.xmax  # dealt with boundary treatment
            if x < 2 * np.pi or x > 4 * np.pi:
                v.append(0)
            else:
                v.append(gfun(x))
        return np.array(v)

    @unittest.skip("this runs too slow")
    def test_grid_test(self):
        def _norm(vec, ord):
            return sum(np.abs(vec) ** ord) ** (1 / ord)

        grid_num = [320, 640, 1280, 2560, 5120, 10240, 20480,
                    40960, 81920, 163840, 327680]
        dx = []
        err = []
        it_num = 20
        for grid in grid_num:
            svr = self._build_solver(grid)
            svr.march_alpha2(it_num)
            exact_sol = self._exact_solution(svr, it_num, np.sin)
            dx.append(svr.grid.ncelm)
            err.append(_norm(svr.get_so0(0).ndarray - exact_sol, 1))

        def _rate(err, dx_rate):
            return abs(math.log(abs(err)) / math.log(abs(dx_rate)))

        def _check(v, r):
            self.assertAlmostEqual(v, r, places=1)

        _check(1.0, _rate(err[0]/err[1], dx[0]/dx[1]))
        _check(1.0, _rate(err[1]/err[2], dx[1]/dx[2]))
        _check(1.1, _rate(err[2]/err[3], dx[2]/dx[3]))
        _check(1.0, _rate(err[3]/err[4], dx[3]/dx[4]))
        _check(1.0, _rate(err[4]/err[5], dx[4]/dx[5]))
        _check(1.0, _rate(err[5]/err[6], dx[5]/dx[6]))
        _check(0.87, _rate(err[6]/err[7], dx[6]/dx[7]))
        _check(1.0, _rate(err[7]/err[8], dx[7]/dx[8]))
        _check(1.0, _rate(err[8]/err[9], dx[8]/dx[9]))
        _check(1.0, _rate(err[9]/err[10], dx[9]/dx[10]))

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
