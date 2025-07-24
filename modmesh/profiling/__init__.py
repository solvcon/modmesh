# Copyright (c) 2025, Han-Xuan Huang <xuan910625.cs13@nycu.edu.tw>
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

from typing import Any, Callable
from dataclasses import dataclass


@dataclass
class ProfilingsFunctionResult:
    name: str
    total_time: float
    count: float
    children: list


@dataclass
class ProfilingColumnData:
    column_name: str
    column_data: list[float | str]


class ProfilingTableBuilder:
    def __init__(
        self,
        column_datas: list[ProfilingColumnData] = [],
        column_width: int = 30,
    ):
        self.column_datas = column_datas
        self.column_width = column_width

        self.row_datas = []

        row_count = len(self.column_datas[0].column_data)
        for i in range(row_count):
            row_data = []

            for j in range(len(self.column_datas)):
                row_data.append(self.column_datas[j].column_data[i])

            self.row_datas.append(row_data)

    def generate_header(self):
        return (
            "| "
            + " | ".join(
                [
                    data.column_name.ljust(self.column_width)
                    for data in self.column_datas
                ]
            )
            + " |\n"
        )

    def generate_horizontal_lines(self):
        return (
            "| "
            + " | ".join(
                ["".ljust(self.column_width, "-") for _ in self.column_datas]
            )
            + " |\n"
        )

    def generate_result_row_by_row(self):
        result = ""

        if len(self.column_datas) == 0:
            return result

        for row_datas in self.row_datas:
            result += (
                "| "
                + " | ".join(
                    [str(data).ljust(self.column_width) for data in row_datas]
                )
                + " |\n"
            )

        return result

    def generate_table_str(self):
        return (
            self.generate_header()
            + self.generate_horizontal_lines()
            + self.generate_result_row_by_row()
        )


class ProfilingResultPrinter:
    def __init__(self, profiling_results: dict[str, Any]):
        self.profiling_results: list[ProfilingsFunctionResult] = [
            ProfilingsFunctionResult(**result) for result in profiling_results
        ]
        self.column_datas: list[ProfilingColumnData] = []

        self.column_datas.append(
            ProfilingColumnData(
                "func",
                [
                    profiling_result.name
                    for profiling_result in self.profiling_results
                ],
            )
        )

    def __getitem__(self, function_name: str) -> ProfilingsFunctionResult:
        result = list(
            filter(lambda x: x.name == function_name, self.profiling_results)
        )

        if len(result) == 0:
            raise ValueError(
                "The result with specific function name is absent."
            )

        return list(result)[0]

    def add_column(
        self,
        column_name: str,
        callable: Callable[[ProfilingsFunctionResult], str | float],
    ):
        self.column_datas.append(
            ProfilingColumnData(
                column_name,
                [
                    callable(profiling_result)
                    for profiling_result in self.profiling_results
                ],
            )
        )

    def print_result(self, column_width=30):
        builder = ProfilingTableBuilder(self.column_datas, column_width)
        print(builder.generate_table_str())

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
