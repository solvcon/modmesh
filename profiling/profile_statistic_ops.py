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
import matplotlib.pyplot as plt

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
    elif np.issubdtype(dtype, np.float32):
        return modmesh.SimpleArrayFloat32(array=data)
    elif np.issubdtype(dtype, np.int64):
        return modmesh.SimpleArrayInt64(array=data)
    elif np.issubdtype(dtype, np.int32):
        return modmesh.SimpleArrayInt32(array=data)
    elif np.issubdtype(dtype, np.int8):
        return modmesh.SimpleArrayInt8(array=data)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


@profile_function
def profile_median_sa(src):
    return src.median()


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
def profile_average_sa(src):
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
def profile_average_np(src):
    return np.average(src)


def profile_stat_op(op, prof_func_np, prof_func_sa, dtype, sizes, it=10,
                    axis=None, dims=None):
    axis_str = f" (axis={axis})" if axis is not None else ""
    dims_str = f" ({dims}D)" if dims is not None else ""
    print(f"\n# {op} (dtype={dtype}){axis_str}{dims_str}")

    results = []

    for N in sizes:
        if axis is not None:
            shape = (N//100, 100) if axis == 1 else (100, N//100)
            if min(shape) <= 0:
                continue
            if np.issubdtype(dtype, np.floating):
                src = np.random.rand(*shape).astype(dtype, copy=False)
            else:
                if dtype == np.int8:
                    src = np.random.randint(-100, 100, shape, dtype=dtype)
                elif dtype == np.int32:
                    src = np.random.randint(-1000, 1000, shape, dtype=dtype)
                else:
                    src = np.random.randint(-1000, 1000, shape, dtype=dtype)
            src_sa = make_container(src, dtype)
        elif dims == 3:
            cube_size = int(N ** (1/3))
            if cube_size < 2:
                cube_size = 2
            larger_shape = (cube_size * 3, cube_size * 3, cube_size * 3)
            if np.issubdtype(dtype, np.floating):
                src = np.random.rand(*larger_shape).astype(dtype, copy=False)
            else:
                if dtype == np.int8:
                    src = np.random.randint(
                        -100, 100, larger_shape, dtype=dtype)
                elif dtype == np.int32:
                    src = np.random.randint(
                        -1000, 1000, larger_shape, dtype=dtype)
                else:
                    src = np.random.randint(
                        -1000, 1000, larger_shape, dtype=dtype)
            src = src[::3, ::3, ::3]
            shape = src.shape
            src_sa = make_container(src, dtype)
        else:
            if np.issubdtype(dtype, np.floating):
                src = np.random.rand(N * 5).astype(dtype, copy=False)
            else:
                if dtype == np.int8:
                    src = np.random.randint(-100, 100, N, dtype=dtype)
                elif dtype == np.int32:
                    src = np.random.randint(-1000, 1000, N, dtype=dtype)
                else:
                    src = np.random.randint(-1000, 1000, N, dtype=dtype)
            src_sa = make_container(src, dtype)
        modmesh.call_profiler.reset()
        for i in range(it):
            if axis is not None:
                prof_func_np(src, axis=axis)
                prof_func_sa(src_sa, axis=axis)
            else:
                prof_func_np(src)
                prof_func_sa(src_sa)
        res = modmesh.call_profiler.result()["children"]
        out = {}
        for r in res:
            name = r["name"].replace(f"profile_{op}_", "")
            time_per_call = r["total_time"] / r["count"]
            out[name] = time_per_call

        result_row = {
            'operation': op,
            'dtype': str(dtype),
            'size': N if axis is None and dims is None
            else f"{shape[0]}x{shape[1]}" if axis is not None
            else f"{shape[0]}x{shape[1]}x{shape[2]}" if dims == 3 else N,
            'size_numeric': N,
            'axis': axis,
            'dims': dims
        }
        result_row.update(out)
        results.append(result_row)

        if axis is not None:
            print(f"## N = {shape}")
        elif dims == 3:
            print(f"## N = {shape}")
        else:
            print(f"## N = {N}")

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

    return results


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
def profile_average_sa_axis(src, axis=None):
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
def profile_average_np_axis(src, axis=None):
    return np.average(src, axis=axis)


def create_performance_plots(all_results):
    plt.ioff()

    operations = set(r['operation'] for r in all_results)
    dtypes = set(r['dtype'] for r in all_results)

    for op in operations:
        create_1d_performance_plot(all_results, op, dtypes)
        create_axis_performance_plot(all_results, op, dtypes)
        create_3d_performance_plot(all_results, op, dtypes)


def create_1d_performance_plot(all_results, op, dtypes):
    sorted_dtypes = sorted(dtypes)
    n_dtypes = len(sorted_dtypes)

    fig, axes = plt.subplots(n_dtypes, 2, figsize=(15, 4*n_dtypes))
    fig.suptitle(f'1D Performance: {op.title()}', fontsize=16)

    if n_dtypes == 1:
        axes = axes.reshape(1, -1)

    op_data = [r for r in all_results
               if r['operation'] == op
               and r['axis'] is None
               and r.get('dims') is None]

    for i, dtype in enumerate(sorted_dtypes):
        dtype_data = [r for r in op_data if r['dtype'] == dtype]

        sizes = [r['size_numeric'] for r in dtype_data]
        np_times = [r['np'] for r in dtype_data]
        sa_times = [r['sa'] for r in dtype_data]

        ax1 = axes[i, 0]
        ax1.plot(sizes, np_times, 'o-', label='NumPy', color='blue')
        ax1.plot(sizes, sa_times, 's-', label='SimpleArray', color='red')
        ax1.set_xlabel('Array Size')
        ax1.set_ylabel('Time per Call (ms)')
        ax1.set_title(f'{dtype} - Performance')
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2 = axes[i, 1]
        speedups = [r['np'] / r['sa'] for r in dtype_data]
        ax2.plot(sizes, speedups, 'D-', color='green')
        ax2.set_xlabel('Array Size')
        ax2.set_ylabel('Speedup (NumPy / SimpleArray)')
        ax2.set_title(f'{dtype} - Speedup')
        ax2.set_xscale('log')
        ax2.axhline(y=1, color='black', linestyle='--', alpha=0.5)
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'profiling/results/png/performance_1d_{op}.png',
                dpi=300, bbox_inches='tight')
    plt.close()


def create_axis_performance_plot(all_results, op, dtypes):
    sorted_dtypes = sorted(dtypes)
    n_dtypes = len(sorted_dtypes)

    fig, axes = plt.subplots(n_dtypes, 4, figsize=(20, 4*n_dtypes))
    fig.suptitle(f'Axis Performance: {op.title()}', fontsize=16)

    if n_dtypes == 1:
        axes = axes.reshape(1, -1)

    axis0_data = [r for r in all_results
                  if r['operation'] == op and r['axis'] == 0]
    axis1_data = [r for r in all_results
                  if r['operation'] == op and r['axis'] == 1]

    for i, dtype in enumerate(sorted_dtypes):
        dtype_axis0 = [r for r in axis0_data if r['dtype'] == dtype]
        dtype_axis1 = [r for r in axis1_data if r['dtype'] == dtype]

        sizes_0 = [r['size_numeric'] for r in dtype_axis0]
        np_times_0 = [r['np_axis'] for r in dtype_axis0]
        sa_times_0 = [r['sa_axis'] for r in dtype_axis0]

        axes[i, 0].plot(sizes_0, np_times_0, 'o-', label='NumPy',
                        color='blue')
        axes[i, 0].plot(sizes_0, sa_times_0, 's-', label='SimpleArray',
                        color='red')
        axes[i, 0].set_xlabel('Array Size')
        axes[i, 0].set_ylabel('Time per Call (ms)')
        axes[i, 0].set_title(f'{dtype} - Axis 0 Performance')
        axes[i, 0].set_xscale('log')
        axes[i, 0].set_yscale('log')
        axes[i, 0].legend()
        axes[i, 0].grid(True, alpha=0.3)

        speedups_0 = [r['np_axis'] / r['sa_axis'] for r in dtype_axis0]
        axes[i, 2].plot(sizes_0, speedups_0, 'D-', color='green')
        axes[i, 2].set_xlabel('Array Size')
        axes[i, 2].set_ylabel('Speedup (NumPy / SimpleArray)')
        axes[i, 2].set_title(f'{dtype} - Axis 0 Speedup')
        axes[i, 2].set_xscale('log')
        axes[i, 2].axhline(y=1, color='black', linestyle='--', alpha=0.5)
        axes[i, 2].grid(True, alpha=0.3)

        sizes_1 = [r['size_numeric'] for r in dtype_axis1]
        np_times_1 = [r['np_axis'] for r in dtype_axis1]
        sa_times_1 = [r['sa_axis'] for r in dtype_axis1]

        axes[i, 1].plot(sizes_1, np_times_1, 'o-', label='NumPy',
                        color='blue')
        axes[i, 1].plot(sizes_1, sa_times_1, 's-', label='SimpleArray',
                        color='red')
        axes[i, 1].set_xlabel('Array Size')
        axes[i, 1].set_ylabel('Time per Call (ms)')
        axes[i, 1].set_title(f'{dtype} - Axis 1 Performance')
        axes[i, 1].set_xscale('log')
        axes[i, 1].set_yscale('log')
        axes[i, 1].legend()
        axes[i, 1].grid(True, alpha=0.3)

        speedups_1 = [r['np_axis'] / r['sa_axis'] for r in dtype_axis1]
        axes[i, 3].plot(sizes_1, speedups_1, 'D-', color='green')
        axes[i, 3].set_xlabel('Array Size')
        axes[i, 3].set_ylabel('Speedup (NumPy / SimpleArray)')
        axes[i, 3].set_title(f'{dtype} - Axis 1 Speedup')
        axes[i, 3].set_xscale('log')
        axes[i, 3].axhline(y=1, color='black', linestyle='--', alpha=0.5)
        axes[i, 3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'profiling/results/png/performance_axis_{op}.png',
                dpi=300, bbox_inches='tight')
    plt.close()


def create_3d_performance_plot(all_results, op, dtypes):
    sorted_dtypes = sorted(dtypes)
    n_dtypes = len(sorted_dtypes)

    fig, axes = plt.subplots(n_dtypes, 2, figsize=(15, 4*n_dtypes))
    fig.suptitle(f'3D Performance: {op.title()}', fontsize=16)

    if n_dtypes == 1:
        axes = axes.reshape(1, -1)

    op_data = [r for r in all_results
               if r['operation'] == op and r['dims'] == 3]

    for i, dtype in enumerate(sorted_dtypes):
        dtype_data = [r for r in op_data if r['dtype'] == dtype]

        sizes = [r['size_numeric'] for r in dtype_data]
        np_times = [r['np'] for r in dtype_data]
        sa_times = [r['sa'] for r in dtype_data]

        ax1 = axes[i, 0]
        ax1.plot(sizes, np_times, 'o-', label='NumPy', color='blue')
        ax1.plot(sizes, sa_times, 's-', label='SimpleArray', color='red')
        ax1.set_xlabel('Array Size')
        ax1.set_ylabel('Time per Call (ms)')
        ax1.set_title(f'{dtype} - 3D Performance')
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2 = axes[i, 1]
        speedups = [r['np'] / r['sa'] for r in dtype_data]
        ax2.plot(sizes, speedups, 'D-', color='green')
        ax2.set_xlabel('Array Size')
        ax2.set_ylabel('Speedup (NumPy / SimpleArray)')
        ax2.set_title(f'{dtype} - 3D Speedup')
        ax2.set_xscale('log')
        ax2.axhline(y=1, color='black', linestyle='--', alpha=0.5)
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('profiling/results/png/'
                + 'performance_3d_non_contiguous_'
                + f'{op}.png',
                dpi=300, bbox_inches='tight')
    plt.close()


def main():
    print("Starting performance profiling...")
    sizes = [10, 10**3, 10**6]
    dtypes = [np.float64, np.float32, np.int64, np.int32, np.int8]
    all_results = []

    for dtype in dtypes:
        print(f"Testing dtype: {dtype}")
        operations_1d = [
            ("median", profile_median_np, profile_median_sa),
            ("mean", profile_mean_np, profile_mean_sa),
            ("var", profile_var_np, profile_var_sa),
            ("std", profile_std_np, profile_std_sa),
            ("average", profile_average_np, profile_average_sa)
        ]

        for op, prof_func_np, prof_func_sa in operations_1d:
            results = profile_stat_op(
                op, prof_func_np, prof_func_sa, dtype, sizes)
            all_results.extend(results)

        operations_axis = [
            ("median", profile_median_np_axis, profile_median_sa_axis),
            ("mean", profile_mean_np_axis, profile_mean_sa_axis),
            ("var", profile_var_np_axis, profile_var_sa_axis),
            ("std", profile_std_np_axis, profile_std_sa_axis),
            ("average", profile_average_np_axis, profile_average_sa_axis)
        ]

        for axis in [0, 1]:
            for op, prof_func_np, prof_func_sa in operations_axis:
                results = profile_stat_op(
                    op, prof_func_np, prof_func_sa, dtype, sizes, axis=axis)
                all_results.extend(results)

        # 3D without axis tests
        operations_3d = [
            ("median", profile_median_np, profile_median_sa),
            ("mean", profile_mean_np, profile_mean_sa),
            ("var", profile_var_np, profile_var_sa),
            ("std", profile_std_np, profile_std_sa),
            ("average", profile_average_np, profile_average_sa)
        ]

        for op, prof_func_np, prof_func_sa in operations_3d:
            results = profile_stat_op(
                op, prof_func_np, prof_func_sa, dtype, sizes, dims=3)
            all_results.extend(results)

    create_performance_plots(all_results)


if __name__ == "__main__":
    main()

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
