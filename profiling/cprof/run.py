# Copyright (c) 2026, modmesh contributors
# BSD-style license; see COPYING

import argparse
import subprocess
from dataclasses import dataclass
from pathlib import Path

from generate_workload import GeneratorConfig


@dataclass(frozen=True)
class CaseResult:
    workload: str
    operations: int
    repeats: int
    workload_seconds: float


@dataclass(frozen=True)
class CaseReport:
    cpp_stat: CaseResult
    gprof_stat: tuple[str, ...]


def parse_result(stdout):
    values = dict(
        word.split("=", 1) for word in stdout.split() if "=" in word
    )
    return CaseResult(
        values["workload"],
        int(values["operations"]),
        int(values["repeats"]),
        float(values["workload_seconds"]),
    )


def format_summary_row(cpp_stat):
    return (
        f"| {cpp_stat.operations} | {cpp_stat.repeats} | "
        f"{cpp_stat.workload_seconds:.6E} |"
    )


def format_gprof_report(report):
    cpp_stat = report.cpp_stat
    gprof_stat = "\n".join(report.gprof_stat)
    return (
        "### gprof top 5: "
        f"operations `{cpp_stat.operations}`, "
        f"repeats `{cpp_stat.repeats}`\n\n"
        "```text\n"
        f"{gprof_stat}\n"
        "```"
    )


def format_workload_report(workload, reports):
    summary = "\n".join(
        format_summary_row(report.cpp_stat) for report in reports
    )
    gprof_reports = "\n\n".join(
        format_gprof_report(report) for report in reports
    )
    return (
        f"## {workload}\n\n"
        "| operations | repeats | workload seconds |\n"
        "| ---------- | ------- | ---------------- |\n"
        f"{summary}\n\n"
        f"{gprof_reports}\n"
    )


class CprofRunner:
    benchmark_cases = GeneratorConfig().benchmark_cases
    excluded_gprof_functions = (
        "_init",
        "_fini",
        "modmesh::CallProfiler::reset(",
    )
    gprof_function_limit = 5
    workloads = (
        "wide_siblings",
        "deep_chain",
        "balanced_tree",
        "hot_name_reuse",
    )

    def __init__(self, executable, gprof, result_dir, working_dir):
        self.executable = str(executable)
        self.gprof = str(gprof)
        self.result_dir = result_dir
        self.working_dir = working_dir

    def run(self):
        self.write_report(self.run_cases())

    def run_command(self, command):
        try:
            completed = subprocess.run(
                command,
                cwd=self.working_dir,
                check=True,
                text=True,
                capture_output=True,
            )
        except subprocess.CalledProcessError as error:
            command_text = " ".join(str(item) for item in command)
            message = f"command failed: {command_text}\n{error.stderr}"
            raise RuntimeError(message) from error

        return completed.stdout

    def run_case(self, workload, operations, repeats):
        gmon_file = self.working_dir / "gmon.out"
        gmon_file.unlink(missing_ok=True)

        stdout = self.run_command([
            self.executable,
            workload,
            str(operations),
            str(repeats),
        ])

        if not gmon_file.is_file():
            raise RuntimeError(f"{gmon_file} was not generated")

        return parse_result(stdout), gmon_file

    def run_gprof(self, gmon_file):
        stdout = self.run_command(
            [self.gprof, "-p", "-b", self.executable, str(gmon_file)]
        )
        gprof_stat = []
        lines = iter(stdout.splitlines())

        for line in lines:
            if line.startswith("  %"):
                gprof_stat.append(line)
                next_line = next(lines, None)
                if next_line is not None:
                    gprof_stat.append(next_line)
                break

        for line in lines:
            if not line.strip():
                continue

            if any(name in line for name in self.excluded_gprof_functions):
                continue

            gprof_stat.append(line)
            if len(gprof_stat) == self.gprof_function_limit + 2:
                break

        return tuple(gprof_stat)

    def run_cases(self):
        reports = []

        for workload in self.workloads:
            workload_reports = []
            for operations, repeats in self.benchmark_cases:
                cpp_stat, gmon_file = self.run_case(
                    workload,
                    operations,
                    repeats,
                )
                gprof_stat = self.run_gprof(gmon_file)
                workload_reports.append(CaseReport(cpp_stat, gprof_stat))
            reports.append(format_workload_report(workload, workload_reports))

        return reports

    def write_report(self, reports):
        self.result_dir.mkdir(parents=True, exist_ok=True)
        report = self.result_dir / "profile_profiler.output"
        content = "# CallProfiler gprof\n\n" + "\n\n".join(reports)
        report.write_text(content + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--gprof", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--working-dir", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    runner = CprofRunner(
        args.executable,
        args.gprof,
        args.result_dir,
        args.working_dir,
    )
    runner.run()


if __name__ == "__main__":
    main()

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
