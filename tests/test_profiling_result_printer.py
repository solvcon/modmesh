from typing import Any

import pytest
import numpy

import modmesh
from modmesh.profiling import (
    ProfilingColumnData,
    ProfilingTableBuilder,
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

    def test_printer_getitem_with_valid_function_key_should_return_correct_data(
        self, profiling_result_fixture: dict[str, Any]
    ) -> None:
        printer: ProfilingResultPrinter = ProfilingResultPrinter(
            profiling_result_fixture
        )

        assert printer["numpy_arange_100"].name == "numpy_arange_100"

    def test_printer_getitem_with_absent_function_key_should_raise_value_error(
        self, profiling_result_fixture: dict[str, Any]
    ) -> None:
        printer: ProfilingResultPrinter = ProfilingResultPrinter(
            profiling_result_fixture
        )

        try:
            printer["numpy_arange_1000"]
            assert False
        except:
            assert True

    def test_printer_add_column_should_have_correct_column(
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

    def test_printer_constructor_should_have_func_name_column_as_default(
        self, profiling_result_fixture: dict[str, Any]
    ) -> None:
        printer: ProfilingResultPrinter = ProfilingResultPrinter(
            profiling_result_fixture
        )

        tot: float = printer["numpy_arange_100"].total_time
        printer.add_column("tot_scale_10", lambda r: r.total_time * 10)

        col = list(
            filter(
                lambda cols: cols.column_name == "func", printer.column_datas
            )
        )[0]
        assert col.column_data[0] == "numpy_arange_100"


class TestProfilingTableBuilder:
    @pytest.fixture
    def fake_column_datas(self) -> list[ProfilingColumnData]:
        return [
            ProfilingColumnData("func", ["foo", "bar", "foobar"]),
            ProfilingColumnData("runtime", [10, 20, 200]),
        ]

    @pytest.fixture
    def expect_header(self) -> str:
        return "| func                           | runtime                        |\n"

    @pytest.fixture
    def expect_horizontal_lines(self) -> str:
        return "| ------------------------------ | ------------------------------ |\n"

    @pytest.fixture
    def expect_row_data(self) -> str:
        return "".join(
            [
                "| foo                            | 10                             |\n",
                "| bar                            | 20                             |\n",
                "| foobar                         | 200                            |\n",
            ]
        )

    def test_generate_header_should_have_correct_header(
        self, fake_column_datas: list[ProfilingColumnData], expect_header: str
    ) -> None:
        builder: ProfilingTableBuilder = ProfilingTableBuilder(
            fake_column_datas
        )

        header: str = builder.generate_header()

        assert expect_header == header

    def test_generate_horizontal_lines_should_have_correct_horizontal_line(
        self,
        fake_column_datas: list[ProfilingColumnData],
        expect_horizontal_lines: str,
    ) -> None:
        builder: ProfilingTableBuilder = ProfilingTableBuilder(
            fake_column_datas
        )

        horizontal_line: str = builder.generate_horizontal_lines()

        assert expect_horizontal_lines == horizontal_line

    def test_generate_row_data_should_have_correct_row_data(
        self,
        fake_column_datas: list[ProfilingColumnData],
        expect_row_data: str,
    ) -> None:
        builder: ProfilingTableBuilder = ProfilingTableBuilder(
            fake_column_datas
        )

        result_rows: str = builder.generate_result_row_by_row()

        assert result_rows == expect_row_data

    def test_generate_table_str_should_have_correct_table(
        self,
        fake_column_datas: list[ProfilingColumnData],
        expect_header: str,
        expect_horizontal_lines: str,
        expect_row_data: str,
    ) -> None:
        builder: ProfilingTableBuilder = ProfilingTableBuilder(
            fake_column_datas
        )

        table_str: str = builder.generate_table_str()

        assert table_str == (
            expect_header + expect_horizontal_lines + expect_row_data
        )
