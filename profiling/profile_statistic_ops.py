# Copyright (c) 2025, Chun-Shih Chang <austin20463@gmail.com>
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


def make_container(data, dtype=None):
    if dtype is None:
        dtype = data.dtype
    if np.issubdtype(dtype, np.float64):
        return modmesh.SimpleArrayFloat64(array=data)
    elif np.issubdtype(dtype, np.int32):
        return modmesh.SimpleArrayInt32(array=data)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


@profile_function
def profile_median_sa(src):
    return src.median()


@profile_function
def profile_median_simd_sa(src):
    return src.parallel_median()


@profile_function
def profile_mean_sa(src):
    return src.mean()


@profile_function
def profile_var_sa(src):
    return src.var()


@profile_function
def profile_std_sa(src):
    return src.std()


@profile_function
def profile_average_sa(src, weights=None):
    if weights is not None:
        return src.average(weight=weights)
    else:
        return src.average()


@profile_function
def profile_median_np(src):
    return np.median(src)


@profile_function
def profile_mean_np(src):
    return np.mean(src)


@profile_function
def profile_var_np(src):
    return np.var(src)


@profile_function
def profile_std_np(src):
    return np.std(src)


@profile_function
def profile_average_np(src, weights=None):
    if weights is not None:
        return np.average(src, weights=weights)
    else:
        return np.average(src)


def profile_stat_op(op, prof_func_np, prof_func_sa, dtype, sizes, it=10,
                    axis=None, with_weights=False, prof_func_simd=None,
                    non_contiguous=False):
    axis_str = f" (axis={axis})" if axis is not None else ""
    weight_str = " (with weights)" if with_weights else ""
    non_cont_str = " (non-contiguous)" if non_contiguous else ""
    print(f"\n# {op} (dtype={dtype}){axis_str}{weight_str}{non_cont_str}")
    for N in sizes:
        if axis is not None:
            shape = (N//100, 100) if axis == 1 else (100, N//100)
            if min(shape) <= 0:
                continue
            if np.issubdtype(dtype, np.floating):
                src = np.random.rand(*shape).astype(dtype, copy=False)
            else:
                src = np.random.randint(-1000, 1000, shape, dtype=dtype)
            # Create non-contiguous array using slicing
            if non_contiguous:
                if axis == 1:
                    src = src[:, ::2]  # Take every other column
                else:
                    src = src[::2, :]  # Take every other row
            src_sa = make_container(src, dtype)
        else:
            if np.issubdtype(dtype, np.floating):
                src = np.random.rand(N).astype(dtype, copy=False)
            else:
                src = np.random.randint(-1000, 1000, N, dtype=dtype)
            # Create non-contiguous array using slicing
            if non_contiguous:
                src = src[::2]  # Take every other element
            src_sa = make_container(src, dtype)
        # Create weights if needed
        weights = None
        weights_sa = None
        if with_weights:
            if axis is not None:
                # For axis operations, weights should match the reduced shape
                if axis == 1:
                    weight_shape = (shape[0],)  # Keep first dimension
                else:
                    weight_shape = (shape[1],)  # Keep second dimension
                if np.issubdtype(dtype, np.floating):
                    weights = np.random.rand(*weight_shape).astype(
                        dtype, copy=False)
                else:
                    weights = np.random.randint(
                        1, 1000, weight_shape, dtype=dtype)
                weights_sa = make_container(weights, dtype)
            else:
                # Use actual size of src array (which may be non-contiguous)
                actual_size = src.size
                if np.issubdtype(dtype, np.floating):
                    weights = np.random.rand(actual_size).astype(
                        dtype, copy=False)
                else:
                    weights = np.random.randint(
                        1, 1000, actual_size, dtype=dtype)
                weights_sa = make_container(weights, dtype)
        modmesh.call_profiler.reset()
        for _ in range(it):
            if axis is not None:
                if with_weights:
                    prof_func_np(src, axis=axis, weights=weights)
                    prof_func_sa(src_sa, axis=axis, weights=weights_sa)
                else:
                    prof_func_np(src, axis=axis)
                    prof_func_sa(src_sa, axis=axis)
            else:
                if with_weights:
                    prof_func_np(src, weights=weights)
                    prof_func_sa(src_sa, weights=weights_sa)
                else:
                    prof_func_np(src)
                    prof_func_sa(src_sa)
                    # Add SIMD version if provided and no axis/weights
                    if prof_func_simd is not None:
                        prof_func_simd(src_sa)
        res = modmesh.call_profiler.result()["children"]
        out = {}
        for r in res:
            name = r["name"].replace(f"profile_{op}_", "")
            time_per_call = r["total_time"] / r["count"]
            out[name] = time_per_call
        print(f"## N = {N if axis is None else shape}")

        def print_row(*cols):
            print(str.format("| {:10s} | {:15s} | {:15s} |", *(cols[0:3])))
        print_row('func', 'per call (ms)', 'cmp to np')
        print_row('-'*10, '-'*15, '-'*15)
        if axis is not None:
            npbase = out.get("np_axis", 1e-12)
        else:
            npbase = out.get("np", 1e-12)
        for k, v in out.items():
            print_row(f"{k:8s}", f"{v:.3E}", f"{v/npbase:.3f}")
        print()


@profile_function
def profile_median_sa_axis(src, axis=None):
    return src.median(axis=axis)


@profile_function
def profile_mean_sa_axis(src, axis=None):
    return src.mean(axis=axis)


@profile_function
def profile_var_sa_axis(src, axis=None):
    return src.var(axis=axis)


@profile_function
def profile_std_sa_axis(src, axis=None):
    return src.std(axis=axis)


@profile_function
def profile_average_sa_axis(src, axis=None, weights=None):
    if weights is not None:
        return src.average(axis=axis, weight=weights)
    else:
        return src.average(axis=axis)


@profile_function
def profile_median_np_axis(src, axis=None):
    return np.median(src, axis=axis)


@profile_function
def profile_mean_np_axis(src, axis=None):
    return np.mean(src, axis=axis)


@profile_function
def profile_var_np_axis(src, axis=None):
    return np.var(src, axis=axis)


@profile_function
def profile_std_np_axis(src, axis=None):
    return np.std(src, axis=axis)


@profile_function
def profile_average_np_axis(src, axis=None, weights=None):
    if weights is not None:
        return np.average(src, axis=axis, weights=weights)
    else:
        return np.average(src, axis=axis)


def main():
    sizes = [10**3, 10**4, 10**5, 10**6, 10**7]
    for dtype in [np.float64, np.int32]:
        # Test contiguous arrays
        profile_stat_op("median", profile_median_np, profile_median_sa,
                        dtype, sizes, prof_func_simd=profile_median_simd_sa)
        profile_stat_op("mean", profile_mean_np, profile_mean_sa, dtype,
                        sizes)
        profile_stat_op("var", profile_var_np, profile_var_sa, dtype, sizes)
        profile_stat_op("std", profile_std_np, profile_std_sa, dtype, sizes)
        profile_stat_op("average", profile_average_np, profile_average_sa,
                        dtype, sizes)
        # Test average with weights (no axis)
        profile_stat_op("average", profile_average_np, profile_average_sa,
                        dtype, sizes, with_weights=True)
        # Test non-contiguous arrays (no axis)
        profile_stat_op("median", profile_median_np, profile_median_sa,
                        dtype, sizes, prof_func_simd=profile_median_simd_sa,
                        non_contiguous=True)
        profile_stat_op("mean", profile_mean_np, profile_mean_sa, dtype,
                        sizes, non_contiguous=True)
        profile_stat_op("var", profile_var_np, profile_var_sa, dtype, sizes,
                        non_contiguous=True)
        profile_stat_op("std", profile_std_np, profile_std_sa, dtype, sizes,
                        non_contiguous=True)
        profile_stat_op("average", profile_average_np, profile_average_sa,
                        dtype, sizes, non_contiguous=True)
        # Test average with weights (no axis, non-contiguous)
        profile_stat_op("average", profile_average_np, profile_average_sa,
                        dtype, sizes, with_weights=True, non_contiguous=True)

        for axis in [0, 1]:
            # Test contiguous arrays with axis
            profile_stat_op("median", profile_median_np_axis,
                            profile_median_sa_axis, dtype, sizes, axis=axis)
            profile_stat_op("mean", profile_mean_np_axis,
                            profile_mean_sa_axis, dtype, sizes, axis=axis)
            profile_stat_op("var", profile_var_np_axis, profile_var_sa_axis,
                            dtype, sizes, axis=axis)
            profile_stat_op("std", profile_std_np_axis, profile_std_sa_axis,
                            dtype, sizes, axis=axis)
            profile_stat_op("average", profile_average_np_axis,
                            profile_average_sa_axis, dtype, sizes, axis=axis)

            # Test non-contiguous arrays with axis
            profile_stat_op("median", profile_median_np_axis,
                            profile_median_sa_axis, dtype, sizes, axis=axis,
                            non_contiguous=True)
            profile_stat_op("mean", profile_mean_np_axis,
                            profile_mean_sa_axis, dtype, sizes, axis=axis,
                            non_contiguous=True)
            profile_stat_op("var", profile_var_np_axis, profile_var_sa_axis,
                            dtype, sizes, axis=axis, non_contiguous=True)
            profile_stat_op("std", profile_std_np_axis, profile_std_sa_axis,
                            dtype, sizes, axis=axis, non_contiguous=True)
            profile_stat_op("average", profile_average_np_axis,
                            profile_average_sa_axis, dtype, sizes, axis=axis,
                            non_contiguous=True)


if __name__ == "__main__":
    main()

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
