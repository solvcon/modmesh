# Copyright (c) 2022, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


import unittest

import solvcon
try:
    from solvcon import pilot
except ImportError:
    pilot = None


@unittest.skipUnless(solvcon.HAS_PILOT, "Qt pilot is not built")
class PilotTC(unittest.TestCase):

    def test_import(self):
        self.assertTrue(hasattr(solvcon.pilot, "mgr"))

    @unittest.skip("headless testing is not ready")
    def test_pycon(self):
        self.assertTrue(pilot.mgr.pycon.python_redirect)
        pilot.mgr.pycon.python_redirect = False
        self.assertFalse(pilot.mgr.pycon.python_redirect)


@unittest.skipUnless(solvcon.HAS_PILOT, "Qt pilot is not built")
class SetupProcessTC(unittest.TestCase):

    def test_namespace_includes_pilot(self):
        import builtins
        from solvcon import system, pilot
        system.setup_process([])
        self.assertIs(builtins.solvcon, solvcon)
        self.assertIs(builtins.sc, solvcon)
        self.assertIs(builtins.pilot, pilot)

    def test_broken_pilot_import_warns(self):
        import sys
        from unittest import mock
        from solvcon import system
        # Force "from . import pilot" to raise ImportError: drop the cached
        # attribute so the import re-runs, and poison sys.modules so it
        # fails. setup_process must warn instead of crashing.
        saved = solvcon.pilot
        del solvcon.pilot
        try:
            with mock.patch.dict(sys.modules, {'solvcon.pilot': None}):
                with self.assertWarns(UserWarning):
                    system.setup_process([])
        finally:
            solvcon.pilot = saved

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
