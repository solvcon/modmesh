# Copyright (c) 2019, Yung-Yu Chen <yyc@solvcon.net>
# BSD-style license; see COPYING


import unittest
import time

import modmesh


class StopWatchTC(unittest.TestCase):

    def test_singleton(self):

        self.assertIs(modmesh.stop_watch, modmesh.StopWatch.me)

    def test_microsecond_resolution(self):

        sw = modmesh.stop_watch
        self.assertGreater(1.e-6, sw.resolution)

    def test_lap_with_sleep(self):

        sw = modmesh.stop_watch

        # Mark start
        sw.lap()

        time.sleep(0.01)

        elapsed = sw.lap()
        self.assertGreater(elapsed, 0.01)
        # Don't test for the upper bound. CI doesn't like it (to be specific,
        # mac runner of github action).


class WrapperProfilerStatusTC(unittest.TestCase):

    def test_singleton(self):

        self.assertIs(
            modmesh.wrapper_profiler_status,
            modmesh.WrapperProfilerStatus.me)

    def test_default(self):

        self.assertTrue(modmesh.wrapper_profiler_status.enabled)


class TimeRegistryTC(unittest.TestCase):

    _profiler_enabled_default = modmesh.wrapper_profiler_status.enabled

    def setUp(self):

        modmesh.wrapper_profiler_status.enable()

    def tearDown(self):

        if self._profiler_enabled_default:
            modmesh.wrapper_profiler_status.enable()
        else:
            modmesh.wrapper_profiler_status.disable()

    def test_singleton(self):

        self.assertIs(modmesh.time_registry, modmesh.TimeRegistry.me)

    def test_empty_report(self):

        modmesh.time_registry.unset()
        ret = modmesh.time_registry.report()
        self.assertEqual("", ret)

    def test_names(self):

        modmesh.time_registry.unset()
        buf = modmesh.ConcreteBuffer(10)
        buf2 = buf.clone()  # noqa: F841
        self.assertEqual(
            ['ConcreteBuffer.__init__', 'ConcreteBuffer.clone'],
            modmesh.time_registry.names)

    def test_status(self):

        modmesh.wrapper_profiler_status.disable()
        self.assertFalse(modmesh.wrapper_profiler_status.enabled)

        modmesh.time_registry.unset()
        buf = modmesh.ConcreteBuffer(10)
        self.assertEqual([], modmesh.time_registry.names)

        modmesh.wrapper_profiler_status.enable()
        modmesh.time_registry.unset()
        buf = modmesh.ConcreteBuffer(10)  # noqa: F841
        self.assertEqual(
            ['ConcreteBuffer.__init__'],
            modmesh.time_registry.names)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
