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
import json

import modmesh


class ToggleTC(unittest.TestCase):

    def test_report(self):
        self.assertTrue(
            "Toggle: USE_PYSIDE=" in modmesh.Toggle.instance.report())

    def test_solid_names(self):
        solid = modmesh.Toggle.instance.solid

        # Test names
        golden = ["use_pyside"]
        self.assertEqual(sorted(solid.get_names()), golden)

        # Test key existence
        for n in sorted(solid.get_names()):
            self.assertTrue(hasattr(solid, n))

    def test_fixed_defaults(self):
        fixed = modmesh.Toggle.instance.fixed

        # Hardcoding the property names and default values does not scale, but
        # I have only few properties at the momemnt.  A better way for testing
        # should be implmented in the future.

        # Test names
        golden = ["python_redirect", "show_axis"]
        self.assertEqual(fixed.NAMES, golden)
        self.assertEqual(sorted(fixed.get_names()), golden)

        # Test property defaults
        self.assertEqual(fixed.python_redirect, True)
        self.assertEqual(fixed.show_axis, False)

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

        # Raise exception when the requested key is not available with the
        # dynamic getter (no need to test for all types).
        with self.assertRaisesRegex(
                AttributeError,
                r'Cannot get non-existing key "dunder_nonexist"'
        ):
            tg.dynamic.dunder_nonexist
        # Overall getter has a different message
        with self.assertRaisesRegex(
                AttributeError,
                r'Cannot get by key "dunder_nonexist'
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

        # Raise exception when the key to be set is not available with the
        # dynamic setter (no need to test for all types).
        with self.assertRaisesRegex(
                AttributeError,
                r'Cannot set non-existing key "dunder_nonexist_real"; '
                r'use set_TYPE\(\) instead'
        ):
            tg.dynamic.dunder_nonexist_real = 12.4
        # Overall setter has a different message
        with self.assertRaisesRegex(
                AttributeError,
                r'Cannot set by key "dunder_nonexist_real"'
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

    def test_get_value(self):
        tg = modmesh.Toggle.instance.clone()
        tg.dynamic_clear()

        tg.add_subkey("level1")
        tg.set_int8("level1.test_int8", 21)
        self.assertEqual(tg.level1.test_int8, 21)
        self.assertEqual(tg.get_value('level1.test_int8'), 21)
        self.assertEqual(tg.get_value(key='level1.test_int8'), 21)
        self.assertEqual(tg.get_value('level1.test_int8', 22), 21)
        self.assertEqual(tg.get_value('level1.test_int8', default=22), 21)
        self.assertEqual(tg.get_value(key='level1.test_int8', default=22), 21)

        self.assertEqual(tg.get_value('level1.non_exist', 22), 22)
        self.assertEqual(tg.get_value('level1.non_exist', default=22), 22)
        self.assertEqual(tg.get_value(key='level1.non_exist', default=22), 22)
        with self.assertRaisesRegex(
                AttributeError,
                r'Cannot get non-existing key "level1.non_exist"'
        ):
            self.assertEqual(tg.level1.non_exist, 21)


class ToggleSerializationTC(unittest.TestCase):

    def test_to_json(self):
        tg = modmesh.Toggle.instance.clone()
        tg.dynamic_clear()
        self.assertEqual(tg.dynamic_keys(), [])

        tg.set_bool("kbool", True)
        tg.add_subkey("k1")
        tg.set_real("k1.kreal", -2.12)

        golden = [{'fixed': {'python_redirect': True, 'show_axis': False}},
                  {'dynamic': {'k1': {'kreal': -2.12}, 'kbool': True}}]
        data = tg.to_python()
        self.assertIsInstance(data, list)  # return a list of dict
        self.assertEqual(data, golden)
        # JSON string differs by platform, use back-n-force conversion to test
        self.assertEqual(json.loads(json.dumps(data)), golden)

    def test_solid_to_json(self):
        tg = modmesh.Toggle.instance.clone()
        tg.dynamic_clear()
        self.assertEqual(tg.dynamic_keys(), [])

        golden = {'use_pyside': tg.solid.use_pyside}
        data = tg.to_python(type="solid")
        self.assertIsInstance(data, dict)
        self.assertEqual(data, golden)
        # JSON string differs by platform, use back-n-force conversion to test
        self.assertEqual(json.loads(json.dumps(data)), golden)

    def test_fixed_to_json(self):
        tg = modmesh.Toggle.instance.clone()
        tg.dynamic_clear()
        self.assertEqual(tg.dynamic_keys(), [])

        golden = {'python_redirect': True, 'show_axis': False}
        data = tg.to_python(type="fixed")
        self.assertIsInstance(data, dict)
        self.assertEqual(data, golden)
        # JSON string differs by platform, use back-n-force conversion to test
        self.assertEqual(json.loads(json.dumps(data)), golden)

    def test_dynamic_to_json(self):
        tg = modmesh.Toggle.instance.clone()
        tg.dynamic_clear()
        self.assertEqual(tg.dynamic_keys(), [])

        tg.set_bool("kbool", True)
        tg.add_subkey("k1")
        tg.set_real("k1.kreal", -2.12)

        golden = {'k1': {'kreal': -2.12}, 'kbool': True}
        data = tg.to_python(type="dynamic")
        self.assertIsInstance(data, dict)
        self.assertEqual(data, golden)
        # JSON string differs by platform, use back-n-force conversion to test
        self.assertEqual(json.loads(json.dumps(data)), golden)


class ToggleLoadTC(unittest.TestCase):

    def test_load(self):
        fixture = '''[{"fixed": {"show_axis": false}},
{"dynamic": {"apps": {"euler1d": {"use_sub": false}}}}]'''
        tg = modmesh.toggle.load(
            fixture,
            toggle_instance=modmesh.Toggle.instance.clone())
        self.assertEqual(tg.apps.euler1d.use_sub, False)

        fixture = '''[{"fixed": {"show_axis": false}},
{"dynamic": {"apps": {"euler1d": {"use_sub": true}}}}]'''
        tg = modmesh.toggle.load(
            fixture,
            toggle_instance=modmesh.Toggle.instance.clone())
        self.assertEqual(tg.apps.euler1d.use_sub, True)

    @unittest.skip("the lifecycle issue may cause segfault")
    def test_load_bad_lifecycle(self):
        fixture = '''[{"fixed": {"show_axis": false}},
{"dynamic": {"apps": {"euler1d": {"use_sub": false}}}}]'''
        # TODO: Need to fix this later. It may be wrong lifecycle handling with
        # WrapHierarchicalToggleAccess.
        with self.assertRaisesRegex(
                AttributeError,
                r'Cannot get non-existing key "apps.euler1d"'
        ):
            modmesh.toggle.load(
                fixture,
                toggle_instance=modmesh.Toggle.instance.clone()).apps.euler1d


class CommandLineInfoTC(unittest.TestCase):

    def setUp(self):
        self.cmdline = modmesh.ProcessInfo.instance.command_line

    def test_populated(self):
        if "pilot" in modmesh.clinfo.executable_basename:
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
