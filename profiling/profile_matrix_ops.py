# Copyright (c) 2026, Chun-Shih Chang <austin20463@gmail.com>
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

import numpy as np
import modmesh


def profile_function(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        _ = modmesh.CallProfilerProbe(func.__name__)
        result = func(*args, **kwargs)
        return result

    return wrapper


def make_container(data):
    if np.issubdtype(data.dtype, np.float32):
        return modmesh.SimpleArrayFloat32(array=data)
    elif np.issubdtype(data.dtype, np.float64):
        return modmesh.SimpleArrayFloat64(array=data)
    raise ValueError(f"Unsupported dtype: {data.dtype}")


@profile_function
def profile_matmul_np(lhs, rhs):
    return np.matmul(lhs, rhs)


@profile_function
def profile_matmul_sa(lhs, rhs):
    return lhs.matmul(rhs)


def make_data(dtype, shape):
    return np.random.rand(*shape).astype(dtype)


def profile_matmul_operation(dtype, shapes, it=10):
    for m in shapes:
        lhs = make_data(dtype, (m, m))
        rhs = make_data(dtype, (m, m))
        lhs_sa = make_container(lhs)
        rhs_sa = make_container(rhs)
        modmesh.call_profiler.reset()
        for _ in range(it):
            profile_matmul_np(lhs, rhs)
            profile_matmul_sa(lhs_sa, rhs_sa)

        res = modmesh.call_profiler.result()["children"]
        out = {}
        for r in res:
            name = r["name"].replace("profile_matmul_", "")
            out[name] = r["total_time"] / r["count"]

        print(f"## 2D x 2D shape: ({m}, {m}) x ({m}, {m}) dtype:"
              f"`{np.dtype(dtype)}`\n")

        def print_row(*cols):
            print(str.format("| {:10s} | {:15s} | {:15s} |", *(cols[0:3])))

        print_row("func", "per call (ms)", "cmp to np")
        print_row("-" * 10, "-" * 15, "-" * 15)
        npbase = out["np"]
        for key in ("np", "sa"):
            value = out[key]
            print_row(f"{key:8s}", f"{value:.3E}", f"{value / npbase:.3f}")
        print()


def main():
    shapes = [4, 16, 64, 256, 1024]

    for dtype in (np.float32, np.float64):
        profile_matmul_operation(dtype, shapes)


if __name__ == "__main__":
    main()

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
