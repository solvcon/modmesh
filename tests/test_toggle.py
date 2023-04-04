# Copyright (c) 2020, Yung-Yu Chen <yyc@solvcon.net>
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# - Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
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


import os
import unittest
import math

import modmesh


class ToggleTC(unittest.TestCase):

    def test_report(self):
        self.assertTrue(
            "Toggle: USE_PYSIDE=" in modmesh.Toggle.instance.report())

    def test_instance(self):
        self.assertTrue(hasattr(modmesh.Toggle.fixed, "use_pyside"))
        self.assertTrue(hasattr(modmesh.Toggle.fixed, "show_axis"))

    def test_clone(self):
        tg = modmesh.Toggle.instance.clone()
        tg.dynamic_clear()
        self.assertEqual(tg.dynamic_keys(), [])

        tg.set_bool("test_bool", True)
        self.assertTrue(tg.get_bool("test_bool"))
        self.assertEqual(tg.dynamic_keys(), ["test_bool"])

        tg1 = tg.clone()
        self.assertEqual(tg.dynamic_keys(), tg1.dynamic_keys())
        self.assertEqual(tg.get_bool("test_bool"), tg1.get_bool("test_bool"))


class ToggleDynamicTC(unittest.TestCase):

    def test_all_types(self):
        tg = modmesh.Toggle.instance.clone()
        tg.dynamic_clear()
        self.assertEqual(tg.dynamic_keys(), [])

        # Add a key of Boolean.
        tg.set_bool("test_bool", True)
        self.assertTrue(tg.get_bool("test_bool"))
        # Make sure the key appears.
        self.assertEqual(tg.dynamic_keys(), ["test_bool"])
        # Test sentinel.
        self.assertEqual(tg.get_bool("test_no_bool"), False)

        # Add a key of int8.
        tg.set_int8("test_int8", 23)
        self.assertEqual(tg.get_int8("test_int8"), 23)
        # Make sure the key appears
        self.assertEqual(sorted(tg.dynamic_keys()),
                         ["test_bool", "test_int8"])
        # Test sentinel.
        self.assertEqual(tg.get_int8("test_no_int8"), 0)

        # Add a key of int16.
        tg.set_int16("test_int16", -46)
        self.assertEqual(tg.get_int16("test_int16"), -46)
        # Make sure the key appears
        self.assertEqual(sorted(tg.dynamic_keys()),
                         ["test_bool", "test_int16", "test_int8"])
        # Test sentinel.
        self.assertEqual(tg.get_int16("test_no_int16"), 0)

        # Add a key of int32.
        tg.set_int32("test_int32", 842)
        self.assertEqual(tg.get_int32("test_int32"), 842)
        # Make sure the key appears
        self.assertEqual(sorted(tg.dynamic_keys()),
                         ["test_bool", "test_int16", "test_int32",
                          "test_int8"])
        # Test sentinel.
        self.assertEqual(tg.get_int32("test_no_int32"), 0)

        # Add a key of int64.
        tg.set_int64("test_int64", -9912)
        self.assertEqual(tg.get_int64("test_int64"), -9912)
        # Make sure the key appears
        self.assertEqual(sorted(tg.dynamic_keys()),
                         ["test_bool", "test_int16", "test_int32",
                          "test_int64", "test_int8"])
        # Test sentinel.
        self.assertEqual(tg.get_int64("test_no_int64"), 0)

        # Clear dynamic keys (and the values).
        tg.dynamic_clear()
        self.assertEqual(tg.dynamic_keys(), [])

        # Add a key of real.
        tg.set_real("test_real", 2.87)
        self.assertEqual(tg.get_real("test_real"), 2.87)
        # Make sure the key appears
        self.assertEqual(sorted(tg.dynamic_keys()), ["test_real"])
        # Test sentinel.
        self.assertTrue(math.isnan(tg.get_real("test_no_real")))

        # Add a key of string.
        tg.set_string("test_string", "a random line")
        self.assertEqual(tg.get_string("test_string"), "a random line")
        # Make sure the key appears
        self.assertEqual(sorted(tg.dynamic_keys()),
                         ["test_real", "test_string"])
        # Test sentinel.
        self.assertEqual(tg.get_string("test_no_string"), "")

        # Clear dynamic keys (and the values) the second time.
        tg.dynamic_clear()
        self.assertEqual(tg.dynamic_keys(), [])

    def test_fatigue(self):
        tg = modmesh.Toggle.instance.clone()
        tg.dynamic_clear()
        self.assertEqual(tg.dynamic_keys(), [])

        # Test sentinel.
        self.assertEqual(tg.get_bool("test_bool"), False)

        # Add a key of Boolean.
        tg.set_bool("test_bool", True)
        self.assertTrue(tg.get_bool("test_bool"))
        # Make sure the key appears.
        self.assertEqual(tg.dynamic_keys(), ["test_bool"])

        # Fatigue test.
        tg.set_bool("test_bool", False)
        self.assertFalse(tg.get_bool("test_bool"))
        tg.set_bool("test_bool", True)
        self.assertTrue(tg.get_bool("test_bool"))
        tg.set_bool("test_bool", False)
        self.assertFalse(tg.get_bool("test_bool"))
        tg.set_bool("test_bool", True)
        self.assertTrue(tg.get_bool("test_bool"))

        tg.dynamic_clear()

    def test_dunder_has_get_set(self):
        tg = modmesh.Toggle.instance.clone()
        tg.dynamic_clear()
        self.assertEqual(tg.dynamic_keys(), [])

        # Raise exception when the requested key is not available (no need to
        # test for all types).
        with self.assertRaisesRegex(
                AttributeError,
                r'Cannt get non-existing key "dunder_nonexist"'
        ):
            tg.dunder_nonexist

        # Need to use set_TYPE() to create the dynamic key-value pair.
        # (Make sure all supported types are tested.)
        tg.set_bool("dunder_bool", True)
        self.assertEqual(tg.dunder_bool, True)
        tg.set_int8("dunder_int8", 12)
        self.assertEqual(tg.dunder_int8, 12)
        tg.set_int16("dunder_int16", -23634)
        self.assertEqual(tg.dunder_int16, -23634)
        tg.set_int32("dunder_int32", 632)
        self.assertEqual(tg.dunder_int32, 632)
        tg.set_int64("dunder_int64", 764)
        self.assertEqual(tg.dunder_int64, 764)
        tg.set_real("dunder_real", -232.1228)
        self.assertEqual(tg.dunder_real, -232.1228)
        tg.set_string("dunder_string", "a line")
        self.assertEqual(tg.dunder_string, "a line")

        # Check for key existence (no need to test for all types).
        self.assertTrue(hasattr(tg, "dunder_int32"))
        self.assertFalse(hasattr(tg, "dunder_nonexist"))

        # Raise exception when the key to be set is not available (no need to
        # test for all types).
        with self.assertRaisesRegex(
                AttributeError,
                r'Cannot set non-existing key "dunder_nonexist_real"; '
                r'use set_TYPE\(\) instead'
        ):
            tg.dunder_nonexist_real = 12.4


class ToggleHierarchicalTC(unittest.TestCase):

    def test_multi_level(self):
        tg = modmesh.Toggle.instance.clone()
        tg.dynamic_clear()
        self.assertEqual(tg.dynamic_keys(), [])

        tg.set_int8("test_int8", 21)
        self.assertEqual(tg.test_int8, 21)
        self.assertEqual(sorted(tg.dynamic_keys()), ["test_int8"])
        tg.add_subkey("level1")
        self.assertIsInstance(tg.level1, modmesh.HierarchicalToggleAccess)
        self.assertEqual(sorted(tg.dynamic_keys()), ["level1", "test_int8"])

        tg.set_real("level1.test_real", 9.42)
        self.assertEqual(tg.level1.test_real, 9.42)
        self.assertEqual(sorted(tg.dynamic_keys()),
                         ["level1", "level1.test_real", "test_int8"])

        # Add second-level subkeys.
        tg.add_subkey("level1.level2")
        self.assertIsInstance(tg.level1.level2,
                              modmesh.HierarchicalToggleAccess)
        tg.add_subkey("level1p")
        tg.level1p.add_subkey("level2p")
        self.assertIsInstance(tg.level1p.level2p,
                              modmesh.HierarchicalToggleAccess)
        tg.level1p.set_bool("test_bool", True)
        self.assertEqual(tg.get_bool('level1p.test_bool'), True)
        tg.set_int32('level1p.level2p.test_int32', -2132)
        self.assertEqual(tg.level1p.level2p.test_int32, -2132)
        self.assertEqual(sorted(tg.dynamic_keys()),
                         ['level1', 'level1.level2', 'level1.test_real',
                          'level1p', 'level1p.level2p',
                          'level1p.level2p.test_int32',
                          'level1p.test_bool', 'test_int8'])


class CommandLineInfoTC(unittest.TestCase):

    def setUp(self):
        self.cmdline = modmesh.ProcessInfo.instance.command_line

    def test_populated(self):
        if "viewer" in modmesh.clinfo.executable_basename:
            self.assertTrue(self.cmdline.populated)
            self.assertNotEqual(len(self.cmdline.populated_argv), 0)
        else:
            self.assertFalse(self.cmdline.populated)


class MetalTC(unittest.TestCase):

    # Github Actions macos-12 does not support GPU yet.
    @unittest.skipUnless(modmesh.METAL_BUILT and "TEST_METAL" in os.environ,
                         "Metal is not built")
    def test_metal_status(self):
        self.assertEqual(True, modmesh.metal_running())

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
