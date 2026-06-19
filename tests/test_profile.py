# Copyright (c) 2019, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


import os
import unittest
import time

import solvcon


class StopWatchTC(unittest.TestCase):

    def test_singleton(self):

        self.assertIs(solvcon.stop_watch, solvcon.StopWatch.me)

    def test_microsecond_resolution(self):

        sw = solvcon.stop_watch
        self.assertGreater(1.e-6, sw.resolution)

    @unittest.skipUnless("nt" != os.name,
                         "timing code on windows does not work yet")
    def test_lap_with_sleep(self):

        sw = solvcon.stop_watch

        # Mark start
        sw.lap()

        time.sleep(0.01)

        elapsed = sw.lap()
        self.assertGreater(elapsed, 0.01)
        # Don't test for the upper bound. CI doesn't like it (to be specific,
        # mac runner of github action).
