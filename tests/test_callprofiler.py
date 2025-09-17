# Copyright (c) 2024, Quentin Tsai <quentin.tsai.tw@gmail.com>
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
import time
import json
import jsonschema
import functools
import modmesh


# TODO
# 1. Implement profile_funcion somewhere else under python modmesh lib
# 2. Add a filed in each node in profiler to distinguish where profiler run
#    like from: C++, from: Python
# 3. Try profile a real app that has both python and C++ codes
#    and add a unit test for this scenario.
def profile_function(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        _ = modmesh.CallProfilerProbe(func.__name__)
        result = func(*args, **kwargs)
        return result
    return wrapper


# duration (seconds)
def busy_loop(duration):
    end_time = time.perf_counter() + duration
    while time.perf_counter() < end_time:
        pass


class CallProfilerTC(unittest.TestCase):

    def test_singleton(self):
        self.assertIs(modmesh.call_profiler, modmesh.CallProfiler.instance)

    def test_reset(self):

        @profile_function
        def foo1():
            busy_loop(0.01)

        foo1()
        modmesh.call_profiler.reset()
        result = modmesh.call_profiler.result()
        self.assertEqual(result, {})

    def test_profiler_result_schema(self):

        @profile_function
        def foo1():
            busy_loop(0.001)

        modmesh.call_profiler.reset()
        foo1()
        result = modmesh.call_profiler.result()

        path = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                            "data", "profiler_python_schema.json")
        with open(path, 'r') as schema_file:
            schema = json.load(schema_file)

        try:
            jsonschema.validate(instance=result, schema=schema)
        except jsonschema.ValidationError as e:
            self.fail(f"JSON data is invalid: {e.message}")

    def test_single_function_profiling(self):

        @profile_function
        def foo():
            busy_loop(0.001)

        modmesh.call_profiler.reset()
        foo()
        root_result = modmesh.call_profiler.result()
        self.assertEqual(len(root_result["children"]), 1)
        self.assertEqual(root_result["children"][0]["name"], "foo")
        foo_result = root_result["children"][0]
        self.assertEqual(len(foo_result["children"]), 0)
        self.assertEqual(foo_result["count"], 1)
        self.assertGreaterEqual(foo_result["total_time"], 1)

    def test_caller_profiling(self):

        @profile_function
        def bar():
            busy_loop(0.005)

        @profile_function
        def foo():
            busy_loop(0.001)
            bar()

        modmesh.call_profiler.reset()
        foo()
        root_result = modmesh.call_profiler.result()
        self.assertEqual(len(root_result["children"]), 1)
        self.assertEqual(root_result["children"][0]["name"], "foo")
        foo_result = root_result["children"][0]
        self.assertEqual(foo_result["count"], 1)
        self.assertGreaterEqual(foo_result["total_time"], 6)

        self.assertEqual(len(foo_result["children"]), 1)
        self.assertEqual(foo_result["children"][0]["name"], "bar")
        bar_result = foo_result["children"][0]
        self.assertEqual(len(bar_result["children"]), 0)
        self.assertEqual(bar_result["count"], 1)
        self.assertGreaterEqual(bar_result["total_time"], 5)

    def test_two_callers_profiling(self):

        @profile_function
        def bar():
            busy_loop(0.005)

        @profile_function
        def foo():
            busy_loop(0.001)
            bar()

        modmesh.call_profiler.reset()
        bar()
        foo()
        root_result = modmesh.call_profiler.result()
        self.assertEqual(len(root_result["children"]), 2)
        search_bar = [i for i in root_result["children"] if i["name"] == "bar"]
        self.assertEqual(len(search_bar), 1)
        search_foo = [i for i in root_result["children"] if i["name"] == "foo"]
        self.assertEqual(len(search_foo), 1)

        bar_result = search_bar[0]
        self.assertEqual(bar_result["count"], 1)
        self.assertGreaterEqual(bar_result["total_time"], 5)

        foo_result = search_foo[0]
        self.assertEqual(foo_result["count"], 1)
        self.assertGreaterEqual(foo_result["total_time"], 6)
        self.assertEqual(len(foo_result["children"]), 1)
        self.assertEqual(foo_result["children"][0]["name"], "bar")

        second_bar_result = foo_result["children"][0]
        self.assertEqual(len(second_bar_result["children"]), 0)
        self.assertEqual(second_bar_result["count"], 1)
        self.assertGreaterEqual(second_bar_result["total_time"], 5)

    def test_get_result_during_profiling(self):

        @profile_function
        def bar():
            busy_loop(0.005)

        @profile_function
        def foo():
            busy_loop(0.001)
            root_result = modmesh.call_profiler.result()
            self.assertEqual(len(root_result["children"]), 1)
            self.assertEqual(root_result["children"][0]["name"], "foo")
            foo_result = root_result["children"][0]
            self.assertEqual(foo_result["count"], 0)
            self.assertGreaterEqual(foo_result["total_time"], 0)
            self.assertIn("current_node", foo_result)
            self.assertTrue(foo_result["current_node"])
            bar()

        modmesh.call_profiler.reset()
        foo()

    def test_zero_callers_serializing(self):
        modmesh.call_profiler.reset()
        sdict = json.loads(modmesh.call_profiler.serialize())

        # There are 3 keys in the serialization_dict
        self.assertEqual(sdict["id_map"], {})
        self.assertEqual(sdict["unique_id"], 0)
        self.assertIn("radix_tree", sdict)

        radix_tree = sdict["radix_tree"]
        self.assertEqual(radix_tree["key"], -1)
        self.assertEqual(radix_tree["name"], "")
        self.assertEqual(radix_tree["call_count"], 0)
        self.assertEqual(radix_tree["total_time"], 0)
        self.assertEqual(radix_tree["children"], [])

    def test_two_callers_serializing(self):

        @profile_function
        def bar():
            busy_loop(0.005)

        @profile_function
        def foo():
            busy_loop(0.001)
            bar()

        modmesh.call_profiler.reset()
        bar()
        foo()
        sdict = json.loads(modmesh.call_profiler.serialize())

        # There are 3 keys in the serialization_dict
        self.assertEqual(sdict["id_map"], {'bar': 0, 'foo': 1})
        self.assertEqual(sdict["unique_id"], 3)
        self.assertIn("radix_tree", sdict)

        radix_tree = sdict["radix_tree"]
        self.assertEqual(radix_tree["key"], -1)
        self.assertEqual(radix_tree["name"], "")
        self.assertEqual(radix_tree["call_count"], 0)
        self.assertEqual(radix_tree["total_time"], 0)
        self.assertIn("children", radix_tree)

        children = radix_tree["children"]
        self.assertEqual(len(children), 2)

        bar_child = children[0]
        self.assertEqual(bar_child["key"], 0)
        self.assertEqual(bar_child["name"], "bar")
        self.assertEqual(bar_child["call_count"], 1)
        self.assertGreaterEqual(bar_child["total_time"], 5e6)
        self.assertEqual(bar_child["children"], [])

        foo_child = children[1]
        self.assertEqual(foo_child["key"], 1)
        self.assertEqual(foo_child["name"], "foo")
        self.assertEqual(foo_child["call_count"], 1)
        self.assertGreaterEqual(foo_child["total_time"], 1e6)
        self.assertEqual(len(foo_child["children"]), 1)

        foo_bar_child = foo_child["children"][0]
        self.assertEqual(foo_bar_child["key"], 0)
        self.assertEqual(foo_bar_child["name"], "bar")
        self.assertEqual(foo_bar_child["call_count"], 1)
        self.assertGreaterEqual(foo_bar_child["total_time"], 5e6)
        self.assertEqual(foo_bar_child["children"], [])

    def test_serialize_during_profiling(self):

        @profile_function
        def bar():
            busy_loop(0.005)

        @profile_function
        def foo():
            bar()
            busy_loop(0.001)
            sdict = json.loads(modmesh.call_profiler.serialize())

            # There are 3 keys in the serialization_dict
            self.assertEqual(sdict["id_map"], {})
            self.assertEqual(sdict["unique_id"], 0)
            self.assertIn("radix_tree", sdict)

            radix_tree = sdict["radix_tree"]
            self.assertEqual(radix_tree["key"], -1)
            self.assertEqual(radix_tree["name"], "")
            self.assertEqual(radix_tree["call_count"], 0)
            self.assertEqual(radix_tree["total_time"], 0)
            self.assertEqual(radix_tree["children"], [])

        modmesh.call_profiler.reset()
        foo()

    def test_status(self):
        modmesh.wrapper_profiler_status.disable()
        self.assertFalse(modmesh.wrapper_profiler_status.enabled)

        modmesh.call_profiler.reset()
        buf = modmesh.ConcreteBuffer(10)
        self.assertEqual({}, modmesh.call_profiler.result())

        modmesh.wrapper_profiler_status.enable()
        modmesh.call_profiler.reset()
        buf = modmesh.ConcreteBuffer(10)  # noqa: F841
        res = modmesh.call_profiler.result().get("children")
        self.assertEqual(['ConcreteBuffer.__init__'], [d["name"] for d in res])

    def test_names(self):
        modmesh.call_profiler.reset()
        buf = modmesh.ConcreteBuffer(10)
        buf2 = buf.clone()  # noqa: F841
        c = modmesh.call_profiler.result().get("children")
        self.assertEqual(
            ['ConcreteBuffer.__init__', 'ConcreteBuffer.clone'],
            [d["name"] for d in c])

    def test_entry(self):
        modmesh.call_profiler.reset()
        buf = modmesh.ConcreteBuffer(10)
        buf2 = buf.clone()  # noqa: F841
        buf2 = buf.clone()  # noqa: F841
        c = modmesh.call_profiler.result().get("children")
        init_result = list(filter(
            lambda d: d["name"] == "ConcreteBuffer.__init__", c
        ))[0]
        clone_result = list(filter(
            lambda d: d["name"] == "ConcreteBuffer.clone", c
        ))[0]

        self.assertEqual(1, init_result["count"])
        self.assertGreater(init_result["total_time"], 0)
        self.assertEqual(2, clone_result["count"])
        self.assertGreater(clone_result["total_time"], 0)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
