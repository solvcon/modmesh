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
    if np.isdtype(data.dtype, np.uint8):
        return modmesh.SimpleArrayUint8(array=data)
    elif np.isdtype(data.dtype, np.uint16):
        return modmesh.SimpleArrayUint16(array=data)
    elif np.isdtype(data.dtype, np.uint32):
        return modmesh.SimpleArrayUint32(array=data)
    elif np.isdtype(data.dtype, np.uint64):
        return modmesh.SimpleArrayUint64(array=data)
    if np.isdtype(data.dtype, np.int8):
        return modmesh.SimpleArrayInt8(array=data)
    elif np.isdtype(data.dtype, np.int16):
        return modmesh.SimpleArrayInt16(array=data)
    elif np.isdtype(data.dtype, np.int32):
        return modmesh.SimpleArrayInt32(array=data)
    elif np.isdtype(data.dtype, np.int64):
        return modmesh.SimpleArrayInt64(array=data)
    elif np.isdtype(data.dtype, np.float32):
        return modmesh.SimpleArrayFloat32(array=data)
    elif np.isdtype(data.dtype, np.float64):
        return modmesh.SimpleArrayFloat64(array=data)


@profile_function
def profile_add_np(src1, src2):
    return np.add(src1, src2)


@profile_function
def profile_add_sa(src1, src2):
    return src1.add(src2)


@profile_function
def profile_add_simd(src1, src2):
    return src1.add_simd(src2)


@profile_function
def profile_sub_np(src1, src2):
    return np.subtract(src1, src2)


@profile_function
def profile_sub_sa(src1, src2):
    return src1.sub(src2)


@profile_function
def profile_sub_simd(src1, src2):
    return src1.sub_simd(src2)


@profile_function
def profile_mul_np(src1, src2):
    return np.multiply(src1, src2)


@profile_function
def profile_mul_sa(src1, src2):
    return src1.mul(src2)


@profile_function
def profile_mul_simd(src1, src2):
    return src1.mul_simd(src2)


@profile_function
def profile_div_np(src1, src2):
    return np.divide(src1, src2)


@profile_function
def profile_div_sa(src1, src2):
    return src1.div(src2)


@profile_function
def profile_div_simd(src1, src2):
    return src1.div_simd(src2)


def prof_add(src1, src2):
    src1_sa = make_container(src1)
    src2_sa = make_container(src2)
    profile_add_np(src1, src2)
    profile_add_sa(src1_sa, src2_sa)
    profile_add_simd(src1_sa, src2_sa)


def prof_sub(src1, src2):
    src1_sa = make_container(src1)
    src2_sa = make_container(src2)
    profile_sub_np(src1, src2)
    profile_sub_sa(src1_sa, src2_sa)
    profile_sub_simd(src1_sa, src2_sa)


def prof_mul(src1, src2):
    src1_sa = make_container(src1)
    src2_sa = make_container(src2)
    profile_mul_np(src1, src2)
    profile_mul_sa(src1_sa, src2_sa)
    profile_mul_simd(src1_sa, src2_sa)


def prof_div(src1, src2):
    src1_sa = make_container(src1)
    src2_sa = make_container(src2)
    profile_div_np(src1, src2)
    profile_div_sa(src1_sa, src2_sa)
    profile_div_simd(src1_sa, src2_sa)


def make_data(dtype, min_val, max_val, N=2 ** 22):
    if "float" in dtype:
        ret = np.random.rand(N).astype(dtype, copy=False)
        ret = ret * (max_val - min_val) + min_val
        return ret
    else:
        return np.random.randint(min_val, max_val, N, dtype=dtype)


def profile_arithmetic_operation(op, val_range, prof_func, dtype, it=10):
    N = 2 ** 22

    modmesh.call_profiler.reset()
    for _ in range(it):
        src1 = make_data(dtype, val_range[0], val_range[1], N)
        src2 = make_data(dtype, val_range[0], val_range[1], N)
        prof_func(src1, src2)

    res = modmesh.call_profiler.result()["children"]

    print(f"## {op} N = 4M range: $[{val_range[0]}, {val_range[1]}]$ "
          f"type: `{dtype}`\n")
    out = {}
    for r in res:
        name = r["name"].replace(f"profile_{op}_", "")
        time = r["total_time"] / r["count"]
        out[name] = time

    def print_row(*cols):
        print(str.format("| {:10s} | {:15s} | {:15s} |" " {:15s} |",
                         *(cols[0:4])))

    print_row('func', 'per call (ms)', 'cmp to np', 'cmp to sa')
    print_row('-' * 10, '-' * 15, '-' * 15, '-' * 15)
    npbase = out["np"]
    sabase = out["sa"]
    for k, v in out.items():
        print_row(f"{k:8s}", f"{v:.3E}", f"{v/npbase:.3f}", f"{v/sabase:.3f}")

    print()


def profile_operation(op):
    dtypes_range = {
        "uint8": (0, 2 ** 8 - 1),
        "uint16": (0, 2 ** 16 - 1),
        "uint32": (0, 2 ** 32 - 1),
        "uint64": (0, 2 ** 64 - 1),
        "int8": (-(2 ** 7), 2 ** 7 - 1),
        "int16": (-(2 ** 15), 2 ** 15 - 1),
        "int32": (-(2 ** 31), 2 ** 31 - 1),
        "int64": (-(2 ** 63), 2 ** 63 - 1),
        "float32": (-(2.0 ** 32), 2.0 ** 32),
        "float64": (-(2.0 ** 64), 2.0 ** 64),
    }

    op_to_func = {
        "add": prof_add,
        "sub": prof_sub,
        "mul": prof_mul,
        "div": prof_div,
    }

    div_test_dtypes = ["float32", "float64"]
    types_to_test = dtypes_range.keys() if op != "div" else div_test_dtypes

    for dtype in types_to_test:
        profile_arithmetic_operation(op,
                                     dtypes_range[dtype],
                                     op_to_func[op],
                                     dtype)


def main():
    for operation in ["add", "sub", "mul", "div"]:
        profile_operation(operation)


if __name__ == "__main__":
    main()

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
