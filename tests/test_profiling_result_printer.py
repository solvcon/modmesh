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

import pytest
import numpy

import modmesh
from modmesh.profiling import (
    ProfilingResultPrinter,
)


class TestProfilingResultPrinter:
    def run_profile_function(self):
        _ = modmesh.CallProfilerProbe("numpy_arange_100")
        numpy.arange(0, 100, dtype="uint32")

    @pytest.fixture
    def profiling_result_fixture(self) -> dict[str, Any]:
        modmesh.call_profiler.reset()

        self.run_profile_function()

        result: dict[str, Any] = modmesh.call_profiler.result()["children"]
        return result

    def test_getitem_valid_key(
        self, profiling_result_fixture: dict[str, Any]
    ) -> None:
        printer: ProfilingResultPrinter = ProfilingResultPrinter(
            profiling_result_fixture
        )

        assert printer["numpy_arange_100"].name == "numpy_arange_100"

    def test_getitem_absent_key(
        self, profiling_result_fixture: dict[str, Any]
    ) -> None:
        printer: ProfilingResultPrinter = ProfilingResultPrinter(
            profiling_result_fixture
        )

        try:
            printer["numpy_arange_1000"]
            assert False
        except ValueError:
            assert True

    def test_add_column(
        self, profiling_result_fixture: dict[str, Any]
    ) -> None:
        printer: ProfilingResultPrinter = ProfilingResultPrinter(
            profiling_result_fixture
        )

        tot: float = printer["numpy_arange_100"].total_time
        printer.add_column("tot_scale_10", lambda r: r.total_time * 10)

        col = list(
            filter(
                lambda cols: cols.column_name == "tot_scale_10",
                printer.column_datas,
            )
        )[0]
        assert col.column_data[0] == tot * 10

    def test_default_column(
        self, profiling_result_fixture: dict[str, Any]
    ) -> None:
        printer: ProfilingResultPrinter = ProfilingResultPrinter(
            profiling_result_fixture
        )

        printer.add_column("tot_scale_10", lambda r: r.total_time * 10)

        col = list(
            filter(
                lambda cols: cols.column_name == "func", printer.column_datas
            )
        )[0]
        assert col.column_data[0] == "numpy_arange_100"

    def test_null_column(self) -> None:
        ProfilingResultPrinter()


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
