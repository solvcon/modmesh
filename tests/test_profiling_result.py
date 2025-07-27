# Copyright (c) 2025, Han-Xuan Huang <c1ydehhx@gmail.com>
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

from typing import Any

import numpy
import unittest

import modmesh
from modmesh.profiling import (
    ProfilingResultPrinter,
)
from modmesh.profiling._result import (
    ProfilingColumnData,
    ProfilingTableBuilder,
)


class TestProfilingResultPrinter(unittest.TestCase):
    def run_profile_function(self):
        _ = modmesh.CallProfilerProbe("numpy_arange_100")
        numpy.arange(0, 100, dtype="uint32")

    def setUp(self):
        modmesh.call_profiler.reset()
        self.run_profile_function()
        result: dict[str, Any] = modmesh.call_profiler.result()["children"]

        self.profiling_result_fixture = result

    def test_getitem_valid_key(self) -> None:
        printer = ProfilingResultPrinter(self.profiling_result_fixture)

        expected_name = "numpy_arange_100"
        self.assertEqual(printer["numpy_arange_100"].name, expected_name)

    def test_getitem_absent_key(self) -> None:
        printer = ProfilingResultPrinter(self.profiling_result_fixture)

        with self.assertRaisesRegex(ValueError, ".* is absent."):
            printer["numpy_arange_1000"]

    def test_add_column(self) -> None:
        printer = ProfilingResultPrinter(self.profiling_result_fixture)
        tot: float = printer["numpy_arange_100"].total_time

        printer.add_column("tot_scale_10", lambda r: r.total_time * 10)

        col = list(
            filter(
                lambda cols: cols.column_name == "tot_scale_10",
                printer.column_data,
            )
        )[0]
        self.assertEqual(col.column_value[0], tot * 10)

    def test_default_column(self) -> None:
        printer = ProfilingResultPrinter(self.profiling_result_fixture)

        printer.add_column("tot_scale_10", lambda r: r.total_time * 10)

        col = list(
            filter(
                lambda cols: cols.column_name == "func", printer.column_data
            )
        )[0]
        self.assertEqual(col.column_value[0], "numpy_arange_100")

    def test_null_column(self) -> None:
        ProfilingResultPrinter()


class TestProfilingTableBuilder(unittest.TestCase):
    def setUp(self):
        self.fake_column_data = [
            ProfilingColumnData("func", ["foo", "bar", "foobar"]),
            ProfilingColumnData("runtime", [10, 20, 200]),
        ]
        self.expect_header = (
            f'| {"func".ljust(30, " ")} | {"runtime".ljust(30, " ")} |\n'
        )
        self.expect_horizontal_lines = (
            f'| {"".ljust(30, "-")} | {"".ljust(30, "-")} |\n'
        )
        self.expect_row_data = "".join(
            [
                f'| {"foo".ljust(30, " ")} | {"10".ljust(30, " ")} |\n',
                f'| {"bar".ljust(30, " ")} | {"20".ljust(30, " ")} |\n',
                f'| {"foobar".ljust(30, " ")} | {"200".ljust(30, " ")} |\n',
            ]
        )

    def test_generate_header(self) -> None:
        builder: ProfilingTableBuilder = ProfilingTableBuilder(
            self.fake_column_data
        )

        header: str = builder.generate_header()

        self.assertEqual(self.expect_header, header)

    def test_generate_hr_lines(
        self,
    ) -> None:
        builder: ProfilingTableBuilder = ProfilingTableBuilder(
            self.fake_column_data
        )

        horizontal_line: str = builder.generate_horizontal_lines()

        self.assertEqual(self.expect_horizontal_lines, horizontal_line)

    def test_generate_row(self) -> None:
        builder: ProfilingTableBuilder = ProfilingTableBuilder(
            self.fake_column_data
        )

        result_rows: str = builder.generate_row()

        self.assertEqual(result_rows, self.expect_row_data)

    def test_generate_table_str(self) -> None:
        builder: ProfilingTableBuilder = ProfilingTableBuilder(
            self.fake_column_data
        )

        table_str: str = builder.generate_table_str()

        expect_result = (
            self.expect_header
            + self.expect_horizontal_lines
            + self.expect_row_data
        )
        self.assertEqual(table_str, expect_result)


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
