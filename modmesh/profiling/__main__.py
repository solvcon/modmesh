# Copyright (c) 2025, Kuan-Hsien Lee <khlee870529@gmail.com>
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

import functools
import modmesh
import numpy as np

from ._result import ProfilingResultPrinter


def profile_function(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        _ = modmesh.CallProfilerProbe(func.__name__)
        result = func(*args, **kwargs)
        return result

    return wrapper


def make_container(data):
    if np.isdtype(data.dtype, np.uint8):
        return modmesh.SimpleArrayUint8(array=data)
    elif np.isdtype(data.dtype, np.uint16):
        return modmesh.SimpleArrayUint16(array=data)
    elif np.isdtype(data.dtype, np.uint32):
        return modmesh.SimpleArrayUint32(array=data)
    elif np.isdtype(data.dtype, np.uint64):
        return modmesh.SimpleArrayUint64(array=data)


@profile_function
def profile_sort_np(narr):
    narr.sort()


@profile_function
def profile_sort_sa(sarr):
    sarr.sort()


@profile_function
def profile_argsort_np(narr):
    return narr.argsort()


@profile_function
def profile_argsort_sa(sarr):
    return sarr.argsort()


@profile_function
def profile_take_along_axis_np(narr, indices):
    return np.take_along_axis(narr, indices, -1)


@profile_function
def profile_take_along_axis_sa(sarr, indices):
    return sarr.take_along_axis(indices)


@profile_function
def profile_take_along_axis_simd(sarr, indices):
    return sarr.take_along_axis_simd(indices)


def main():

    N = 100000
    it = 10

    print("## `sort` Ascending Data\n")
    modmesh.call_profiler.reset()
    for _ in range(it):
        test_data = np.arange(0, N, dtype='uint32')
        profile_sort_np(test_data)
        profile_sort_sa(make_container(test_data))

    printer = ProfilingResultPrinter(
        modmesh.call_profiler.result()["children"]
    )
    printer.add_column("per call (ms)", lambda r: r.total_time)

    tot = printer["profile_sort_np"].total_time
    printer.add_column("cmp to np", lambda r: r.total_time / tot)

    printer.print_result(column_width=30)

    print("## `sort` Descending Data\n")
    modmesh.call_profiler.reset()
    for _ in range(it):
        test_data = np.arange(N, 0, -1, dtype='uint32')
        profile_sort_np(test_data)
        profile_sort_sa(make_container(test_data))

    printer = ProfilingResultPrinter(
        modmesh.call_profiler.result()["children"]
    )
    printer.add_column("per call (ms)", lambda r: r.total_time)

    tot = printer["profile_sort_np"].total_time
    printer.add_column("cmp to np", lambda r: r.total_time / tot)

    printer.print_result(column_width=30)

    print("## `sort` Random Data\n")
    modmesh.call_profiler.reset()
    for _ in range(it):
        test_data = np.arange(0, N, dtype='uint32')
        np.random.shuffle(test_data)
        profile_sort_np(test_data)
        profile_sort_sa(make_container(test_data))

    printer = ProfilingResultPrinter(
        modmesh.call_profiler.result()["children"]
    )
    printer.add_column("per call (ms)", lambda r: r.total_time)

    tot = printer["profile_sort_np"].total_time
    printer.add_column("cmp to np", lambda r: r.total_time / tot)

    printer.print_result(column_width=30)

    print("## `argsort` Ascending Data\n")
    modmesh.call_profiler.reset()
    for _ in range(it):
        test_data = np.arange(0, N, dtype='uint32')
        profile_argsort_np(test_data)
        profile_argsort_sa(make_container(test_data))

    printer = ProfilingResultPrinter(
        modmesh.call_profiler.result()["children"]
    )
    printer.add_column("per call (ms)", lambda r: r.total_time)

    tot = printer["profile_argsort_np"].total_time
    printer.add_column("cmp to np", lambda r: r.total_time / tot)

    printer.print_result(column_width=30)

    print("## `argsort` Descending Data\n")
    modmesh.call_profiler.reset()
    for _ in range(it):
        test_data = np.arange(N, 0, -1, dtype='uint32')
        profile_argsort_np(test_data)
        profile_argsort_sa(make_container(test_data))

    printer = ProfilingResultPrinter(
        modmesh.call_profiler.result()["children"]
    )
    printer.add_column("per call (ms)", lambda r: r.total_time)

    tot = printer["profile_argsort_np"].total_time
    printer.add_column("cmp to np", lambda r: r.total_time / tot)

    printer.print_result(column_width=30)

    print("## `argsort` Random Data\n")
    modmesh.call_profiler.reset()
    for _ in range(it):
        test_data = np.arange(0, N, dtype='uint32')
        np.random.shuffle(test_data)
        profile_argsort_np(test_data)
        profile_argsort_sa(make_container(test_data))

    printer = ProfilingResultPrinter(
        modmesh.call_profiler.result()["children"]
    )
    printer.add_column("per call (ms)", lambda r: r.total_time)

    tot = printer["profile_argsort_np"].total_time
    printer.add_column("cmp to np", lambda r: r.total_time / tot)

    printer.print_result(column_width=30)

    print("## `take_along_axis` Ascending Data\n")
    modmesh.call_profiler.reset()
    for _ in range(it):
        test_data = np.arange(0, N, dtype='uint32')
        indices = np.arange(0, N-1, dtype='uint32')
        test_sa = make_container(test_data)
        idx_sa = make_container(indices)
        profile_take_along_axis_np(test_data, indices)
        profile_take_along_axis_sa(test_sa, idx_sa)
        profile_take_along_axis_simd(test_sa, idx_sa)

    printer = ProfilingResultPrinter(
        modmesh.call_profiler.result()["children"]
    )
    printer.add_column("per call (ms)", lambda r: r.total_time)

    tot = printer["profile_take_along_axis_np"].total_time
    printer.add_column("cmp to np", lambda r: r.total_time / tot)

    printer.print_result(column_width=30)

    print("## `take_along_axis` Descending Data\n")
    modmesh.call_profiler.reset()
    for _ in range(it):
        test_data = np.arange(N, 0, -1, dtype='uint32')
        indices = np.arange(N-1, 0, -1, dtype='uint32')
        test_sa = make_container(test_data)
        idx_sa = make_container(indices)
        profile_take_along_axis_np(test_data, indices)
        profile_take_along_axis_sa(test_sa, idx_sa)
        profile_take_along_axis_simd(test_sa, idx_sa)

    printer = ProfilingResultPrinter(
        modmesh.call_profiler.result()["children"]
    )
    printer.add_column("per call (ms)", lambda r: r.total_time)

    tot = printer["profile_take_along_axis_np"].total_time
    printer.add_column("cmp to np", lambda r: r.total_time / tot)

    printer.print_result(column_width=30)

    print("## `take_along_axis` Random Data\n")
    modmesh.call_profiler.reset()
    for _ in range(it):
        test_data = np.arange(0, N, dtype='uint32')
        indices = np.arange(0, N-1, dtype='uint32')
        np.random.shuffle(test_data)
        np.random.shuffle(indices)
        test_sa = make_container(test_data)
        idx_sa = make_container(indices)
        profile_take_along_axis_np(test_data, indices)
        profile_take_along_axis_sa(test_sa, idx_sa)
        profile_take_along_axis_simd(test_sa, idx_sa)

    printer = ProfilingResultPrinter(
        modmesh.call_profiler.result()["children"]
    )
    printer.add_column("per call (ms)", lambda r: r.total_time)

    tot = printer["profile_take_along_axis_np"].total_time
    printer.add_column("cmp to np", lambda r: r.total_time / tot)

    printer.print_result(column_width=30)


if __name__ == '__main__':
    main()

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
