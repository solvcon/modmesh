# Copyright (c) 2018, Yung-Yu Chen <yyc@solvcon.net>
# BSD 3-Clause License, see COPYING

import unittest

import numpy as np

from modmesh import spacetime as libst


class SolverTC(unittest.TestCase):

    def setUp(self):
        self.grid10 = libst.Grid(0, 10, 10)
        self.sol10 = libst.Solver(grid=self.grid10, nvar=1,
                                  time_increment=0.2)
        self.sol10.so0.ndarray.fill(-1)
        self.sol10.so1.ndarray.fill(-2)

    def test_str(self):
        self.assertEqual("Solver(grid=Grid(xmin=0, xmax=10, ncelm=10))",
                         str(self.sol10))

        self.assertEqual(
            "SolverElementIterator(celm, on_even_plane, current=0, nelem=10)",
            str(self.sol10.celms(odd_plane=False))
        )

        self.assertEqual(
            "SolverElementIterator(celm, on_odd_plane, current=0, nelem=9)",
            str(self.sol10.celms(odd_plane=True))
        )

        self.assertEqual(
            "SolverElementIterator(selm, on_even_plane, current=0, nelem=11)",
            str(self.sol10.selms(odd_plane=False))
        )

        self.assertEqual(
            "SolverElementIterator(selm, on_odd_plane, current=0, nelem=10)",
            str(self.sol10.selms(odd_plane=True))
        )

    def test_clone(self):
        self.assertEqual(self.grid10, self.sol10.clone().grid)
        self.assertNotEqual(self.grid10, self.sol10.clone(grid=True).grid)

    def test_grid(self):
        self.assertEqual(self.grid10, self.sol10.grid)

    def test_nvar(self):
        self.assertEqual(1, self.sol10.nvar)

    def test_time_increment(self):
        self.assertEqual(0.2, self.sol10.time_increment)
        self.sol10.time_increment = 42
        self.assertEqual(42, self.sol10.time_increment)

    def test_so0_so1(self):
        nx = (self.sol10.grid.ncelm + self.sol10.grid.BOUND_COUNT) * 2 + 1

        # shape
        self.assertEqual((nx, 1), self.sol10.so0.shape)
        self.assertEqual((nx, 1), self.sol10.so1.shape)
        # type
        self.assertEqual(np.float64, self.sol10.so0.ndarray.dtype)
        self.assertEqual(np.float64, self.sol10.so1.ndarray.dtype)
        # content
        self.sol10.so0.ndarray.fill(0)
        self.assertEqual([0.0] * nx, self.sol10.so0.ndarray.flatten().tolist())
        self.sol10.so1.ndarray.fill(1)
        self.assertEqual([1.0] * nx, self.sol10.so1.ndarray.flatten().tolist())

    def test_celm(self):
        with self.assertRaisesRegex(
                IndexError,
                r"Field::celm_at\(ielm=-1, odd_plane=0\): xindex = 1 "
                r"outside the interval \[2, 23\)",
        ):
            self.sol10.celm(-1, odd_plane=False)

        self.assertEqual(
            "Celm(odd, index=-1, x=0, xneg=-0.5, xpos=0.5)",
            str(self.sol10.celm(-1, odd_plane=True)),
        )

        self.assertEqual(
            "Celm(even, index=0, x=0.5, xneg=0, xpos=1)",
            str(self.sol10.celm(0)),
        )

        self.assertEqual(
            "Celm(odd, index=0, x=1, xneg=0.5, xpos=1.5)",
            str(self.sol10.celm(ielm=0, odd_plane=True)),
        )

        self.assertEqual(
            "Celm(even, index=9, x=9.5, xneg=9, xpos=10)",
            str(self.sol10.celm(9, odd_plane=False)),
        )

        self.assertEqual(
            "Celm(odd, index=9, x=10, xneg=9.5, xpos=10.5)",
            str(self.sol10.celm(9, odd_plane=True)),
        )

        with self.assertRaisesRegex(
                IndexError,
                r"Field::celm_at\(ielm=10, odd_plane=0\): xindex = 23 "
                r"outside the interval \[2, 23\)",
        ):
            self.sol10.celm(10)

    def test_selm(self):
        with self.assertRaisesRegex(
                IndexError,
                r"Field::selm_at\(ielm=-1, odd_plane=0\): xindex = 0 "
                r"outside the interval \[1, 24\)",
        ):
            self.sol10.selm(-1, odd_plane=False)

        self.assertEqual(
            "Selm(odd, index=-1, x=-0.5, xneg=-1, xpos=0)",
            str(self.sol10.selm(-1, odd_plane=True)),
        )

        self.assertEqual(
            "Selm(even, index=0, x=0, xneg=-0.5, xpos=0.5)",
            str(self.sol10.selm(0, odd_plane=False)),
        )

        self.assertEqual(
            "Selm(odd, index=0, x=0.5, xneg=0, xpos=1)",
            str(self.sol10.selm(0, odd_plane=True)),
        )

        self.assertEqual(
            "Selm(even, index=10, x=10, xneg=9.5, xpos=10.5)",
            str(self.sol10.selm(10, odd_plane=False)),
        )

        self.assertEqual(
            "Selm(odd, index=10, x=10.5, xneg=10, xpos=11)",
            str(self.sol10.selm(10, odd_plane=True)),
        )

        with self.assertRaisesRegex(
                IndexError,
                r"Field::selm_at\(ielm=11, odd_plane=0\): xindex = 24 "
                r"outside the interval \[1, 24\)",
        ):
            self.sol10.selm(11)

    def test_celms(self):
        sol = self.sol10

        self.assertEqual(list(sol.celms()), list(sol.celms(odd_plane=False)))

        gold = [sol.celm(it, odd_plane=False)
                for it in range(sol.grid.ncelm)]
        self.assertEqual(gold, list(sol.celms(odd_plane=False)))

        gold = [sol.celm(it, odd_plane=True)
                for it in range(sol.grid.ncelm - 1)]
        self.assertEqual(gold, list(sol.celms(odd_plane=True)))

    def test_selms(self):
        sol = self.sol10

        self.assertEqual(list(sol.selms()), list(sol.selms(odd_plane=False)))

        gold = [sol.selm(it, odd_plane=False)
                for it in range(sol.grid.nselm)]
        self.assertEqual(gold, list(sol.selms(odd_plane=False)))

        gold = [sol.selm(it, odd_plane=True)
                for it in range(sol.grid.nselm - 1)]
        self.assertEqual(gold, list(sol.selms(odd_plane=True)))


class PythonCustomSolverTC(unittest.TestCase):

    @staticmethod
    def _build_solver(resolution):
        # Build grid.
        xcrd = np.arange(resolution + 1) / resolution
        xcrd *= 2 * np.pi
        grid = libst.Grid(xcrd)
        dx = (grid.xmax - grid.xmin) / grid.ncelm

        # Build solver.
        time_stop = 2 * np.pi
        cfl_max = 1.0
        dt_max = dx * cfl_max
        nstep = int(np.ceil(time_stop / dt_max))
        dt = time_stop / nstep
        svr = libst.Solver(grid=grid, time_increment=dt, nvar=1)

        # Customize to linear wave solver.
        def xn(se, iv):
            displacement = 0.5 * (se.x + se.xneg) - se.xctr
            return se.dxneg * (se.get_so0(iv) + displacement * se.get_so1(iv))

        svr.kernel.xn_calc = xn

        def xp(se, iv):
            displacement = 0.5 * (se.x + se.xpos) - se.xctr
            return se.dxpos * (se.get_so0(iv) + displacement * se.get_so1(iv))

        svr.kernel.xp_calc = xp

        def tn(se, iv):
            displacement = se.x - se.xctr
            ret = se.get_so0(iv)  # f(u)
            ret += displacement * se.get_so1(iv)  # displacement in x; f_u == 1
            ret += se.qdt * se.get_so1(iv)  # displacement in t
            return se.hdt * ret

        svr.kernel.tn_calc = tn

        def tp(se, iv):
            displacement = se.x - se.xctr
            ret = se.get_so0(iv)  # f(u)
            ret += displacement * se.get_so1(iv)  # displacement in x; f_u == 1
            ret -= se.qdt * se.get_so1(iv)  # displacement in t
            return se.hdt * ret

        svr.kernel.tp_calc = tp

        def so0p(se, iv):
            ret = se.get_so0(iv)
            ret += (se.x - se.xctr) * se.get_so1(iv)  # displacement in x
            ret -= se.hdt * se.get_so1(iv)  # displacement in t
            return ret

        svr.kernel.so0p_calc = so0p

        def cfl(se):
            hdx = min(se.dxneg, se.dxpos)
            se.set_cfl(se.hdt / hdx)

        svr.kernel.cfl_updater = cfl

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
        self.assertEqual(len(self.svr.xctr()), self.svr.grid.ncelm + 1)
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
        self.assertEqual(self.svr.grid.ncelm + 1, len(v2))
        self.assertEqual(v1, v2)

        with self.assertRaisesRegex(IndexError, "out of nvar range"):
            self.svr.get_so0(1)
        with self.assertRaisesRegex(IndexError, "out of nvar range"):
            self.svr.get_so0(1, odd_plane=True)

        v1 = [e.get_so1(0) for e in self.svr.selms(odd_plane=False)]
        v2 = self.svr.get_so1(0).ndarray.tolist()
        self.assertEqual(self.svr.grid.ncelm + 1, len(v2))
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

    def test_se(self):
        self.svr.kernel.reset()
        self.assertIsNotNone(self.svr.kernel.xn_calc)

        se = self.svr.selm(0)
        self.assertEqual(0, se.xn(0))

        def _(se, iv):
            return 1.2

        self.svr.kernel.xn_calc = _
        self.assertEqual(1.2, se.xn(0))

    def test_march(self):
        self.svr.march_alpha2(self.nstep * self.cycle)
        np.testing.assert_allclose(self.svr.get_so0(0), np.sin(self.xcrd),
                                   rtol=0, atol=1.e-14)
        ones = np.ones(self.svr.grid.nselm, dtype='float64')
        np.testing.assert_allclose(self.svr.get_cfl(), ones,
                                   rtol=0, atol=1.e-14)

        self.svr.kernel.reset()
        self.svr.march_alpha2(self.nstep * self.cycle)
        np.testing.assert_array_equal(self.svr.get_so0(0),
                                      np.zeros_like(self.xcrd))

    def test_march_fine_interface(self):
        def _march():
            self.svr.treat_boundary_so0()
            self.svr.treat_boundary_so1()
            self.svr.march_half_so0(odd_plane=False)
            self.svr.update_cfl(odd_plane=True)
            self.svr.march_half_so1_alpha2(odd_plane=False)
            # second half step.
            self.svr.march_half_so0(odd_plane=True)
            self.svr.update_cfl(odd_plane=False)
            self.svr.march_half_so1_alpha2(odd_plane=True)

        svr2 = self._build_solver(self.resolution)[-1]

        for it in range(self.nstep * self.cycle):
            _march()
            svr2.march_alpha2(steps=1)
            self.assertEqual(self.svr.get_so0(0).ndarray.tolist(),
                             svr2.get_so0(0).ndarray.tolist())


class LinearProxy(libst.SolverProxy):

    def _xn_calc(self, se, iv):
        displacement = 0.5 * (se.x + se.xneg) - se.xctr
        return se.dxneg * (se.get_so0(iv) + displacement * se.get_so1(iv))

    def _xp_calc(self, se, iv):
        displacement = 0.5 * (se.x + se.xpos) - se.xctr
        return se.dxpos * (se.get_so0(iv) + displacement * se.get_so1(iv))

    def _tn_calc(self, se, iv):
        displacement = se.x - se.xctr
        ret = se.get_so0(iv)  # f(u)
        ret += displacement * se.get_so1(iv)  # displacement in x; f_u == 1
        ret += se.qdt * se.get_so1(iv)  # displacement in t
        return se.hdt * ret

    def _tp_calc(self, se, iv):
        displacement = se.x - se.xctr
        ret = se.get_so0(iv)  # f(u)
        ret += displacement * se.get_so1(iv)  # displacement in x; f_u == 1
        ret -= se.qdt * se.get_so1(iv)  # displacement in t
        return se.hdt * ret

    def _so0p_calc(self, se, iv):
        ret = se.get_so0(iv)
        ret += (se.x - se.xctr) * se.get_so1(iv)  # displacement in x
        ret -= se.hdt * se.get_so1(iv)  # displacement in t
        return ret

    def _cfl_updater(self, se):
        hdx = min(se.dxneg, se.dxpos)
        se.set_cfl(se.hdt / hdx)


class SolverProxyTC(unittest.TestCase):

    @staticmethod
    def _build_solver(resolution):
        # Build grid.
        xcrd = np.arange(resolution + 1) / resolution
        xcrd *= 2 * np.pi
        grid = libst.Grid(xcrd)
        dx = (grid.xmax - grid.xmin) / grid.ncelm

        # Build solver.
        time_stop = 2 * np.pi
        cfl_max = 1.0
        dt_max = dx * cfl_max
        nstep = int(np.ceil(time_stop / dt_max))
        dt = time_stop / nstep
        svr = LinearProxy(grid=grid, time_increment=dt, nvar=1)

        # Initialize.
        svr.set_so0(0, np.sin(xcrd))
        svr.set_so1(0, np.cos(xcrd))
        svr.setup_march()

        return nstep, xcrd, svr

    def setUp(self):
        self.resolution = 8
        self.nstep, self.xcrd, self.svr = self._build_solver(self.resolution)
        self.cycle = 10

    def test_march(self):
        self.svr.march_alpha2(self.nstep * self.cycle)
        np.testing.assert_allclose(self.svr.get_so0(0), np.sin(self.xcrd),
                                   rtol=0, atol=1.e-14)
        ones = np.ones(self.svr.grid.nselm, dtype='float64')
        np.testing.assert_allclose(self.svr.get_cfl(), ones,
                                   rtol=0, atol=1.e-14)

        self.svr.kernel.reset()
        self.svr.march_alpha2(self.nstep * self.cycle)
        np.testing.assert_array_equal(self.svr.get_so0(0),
                                      np.zeros_like(self.xcrd))

    def test_march_fine_interface(self):
        def _march():
            # first half step.
            self.svr.treat_boundary_so0()
            self.svr.treat_boundary_so1()
            self.svr.march_half_so0(odd_plane=False)
            self.svr.update_cfl(odd_plane=True)
            self.svr.march_half_so1_alpha2(odd_plane=False)
            # second half st
            self.svr.march_half_so0(odd_plane=True)
            self.svr.update_cfl(odd_plane=False)
            self.svr.march_half_so1_alpha2(odd_plane=True)

        svr2 = self._build_solver(self.resolution)[-1]

        for it in range(self.nstep * self.cycle):
            _march()
            svr2.march_alpha2(steps=1)
            self.assertEqual(self.svr.get_so0(0).ndarray.tolist(),
                             svr2.get_so0(0).ndarray.tolist())

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
