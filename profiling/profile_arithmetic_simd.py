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


def profile_arithmetic_operation(max_pow, prof_func, title, it=10):
    N = 2 ** 22
    dtype = ["uint8", "uint16", "uint32", "uint32",
             "uint64", "uint64", "uint64", "uint64"][max_pow // 8]
    max_val = 2 ** max_pow

    modmesh.call_profiler.reset()
    for _ in range(it):
        src1 = np.random.randint(1, max_val, N,  dtype=dtype)
        src2 = np.random.randint(1, max_val, N,  dtype=dtype)

        if title == "div":
            src1 = src1.astype("float32")
            src2 = src2.astype("float32")

        prof_func(src1, src2)

    res = modmesh.call_profiler.result()["children"]

    print(f"## {title} N = 4M max: 2^{max_pow} "
          f"type: {dtype if title != 'div' else 'float32'}\n")
    out = {}
    for r in res:
        name = r["name"].replace(f"profile_{title}_", "")
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


def main():
    pow = 6
    it = 6

    for _ in range(it):
        profile_arithmetic_operation(pow, prof_add, "add")
        profile_arithmetic_operation(pow, prof_sub, "sub")
        profile_arithmetic_operation(pow, prof_mul, "mul")
        profile_arithmetic_operation(pow, prof_div, "div")
        pow = pow + 8


if __name__ == "__main__":
    main()

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
