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
from jsonschema import validate, ValidationError
from functools import wraps
import modmesh


def profile_function(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        _ = modmesh.CallProfilerProbe(func.__name__)
        result = func(*args, **kwargs)
        return result
    return wrapper


def busy_loop(duration):
    end_time = time.perf_counter() + duration
    while time.perf_counter() < end_time:
        time.sleep(0.00001)


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
            busy_loop(1)

        modmesh.call_profiler.reset()
        foo1()
        result = modmesh.call_profiler.result()

        path = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                            "data", "profiler_python_schema.json")
        with open(path, 'r') as schema_file:
            schema = json.load(schema_file)

        try:
            validate(instance=result, schema=schema)
        except ValidationError as e:
            self.fail(f"JSON data is invalid: {e.message}")

    def test_single_function_profiling(self):

        @profile_function
        def foo():
            busy_loop(0.1)

        modmesh.call_profiler.reset()
        foo()
        root_result = modmesh.call_profiler.result()
        self.assertEqual(len(root_result["children"]), 1)
        self.assertEqual(root_result["children"][0]["name"], "foo")
        foo_result = root_result["children"][0]
        self.assertEqual(len(foo_result["children"]), 0)
        self.assertEqual(foo_result["count"], 1)
        self.assertGreaterEqual(foo_result["total_time"], 100)

    def test_caller_profiling(self):

        @profile_function
        def bar():
            busy_loop(0.5)

        @profile_function
        def foo():
            busy_loop(0.1)
            bar()

        modmesh.call_profiler.reset()
        foo()
        root_result = modmesh.call_profiler.result()
        self.assertEqual(len(root_result["children"]), 1)
        self.assertEqual(root_result["children"][0]["name"], "foo")
        foo_result = root_result["children"][0]
        self.assertEqual(foo_result["count"], 1)
        self.assertGreaterEqual(foo_result["total_time"], 600)

        self.assertEqual(len(foo_result["children"]), 1)
        self.assertEqual(foo_result["children"][0]["name"], "bar")
        bar_result = foo_result["children"][0]
        self.assertEqual(len(bar_result["children"]), 0)
        self.assertEqual(bar_result["count"], 1)
        self.assertGreaterEqual(bar_result["total_time"], 500)

    def test_two_callers_profiling(self):

        @profile_function
        def bar():
            busy_loop(0.5)

        @profile_function
        def foo():
            busy_loop(0.1)
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
        self.assertGreaterEqual(bar_result["total_time"], 500)

        foo_result = search_foo[0]
        self.assertEqual(foo_result["count"], 1)
        self.assertGreaterEqual(foo_result["total_time"], 600)
        self.assertEqual(len(foo_result["children"]), 1)
        self.assertEqual(foo_result["children"][0]["name"], "bar")

        second_bar_result = foo_result["children"][0]
        self.assertEqual(len(second_bar_result["children"]), 0)
        self.assertEqual(second_bar_result["count"], 1)
        self.assertGreaterEqual(second_bar_result["total_time"], 500)

    def test_get_result_during_profiling(self):

        @profile_function
        def bar():
            busy_loop(0.5)

        @profile_function
        def foo():
            busy_loop(0.1)
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
