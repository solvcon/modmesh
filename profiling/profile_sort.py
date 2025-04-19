import functools
import modmesh
import numpy as np


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
def profile_take_along_axis_neon(sarr, indices):
    return sarr.take_along_axis_neon(indices)


def print_res(res):
    out = {}
    for r in res:
        name = r["name"]
        time = r["total_time"] / r["count"]
        out[name] = time

    def print_row(*cols):
        print(str.format("| {:30s} | {:15s} | {:15s} |", *(cols[0:3])))

    print_row('func', 'per call (ms)', 'cmp to np')
    print_row('-' * 30, '-' * 15, '-' * 15)
    npkey = [k for k in out.keys() if "np" in k][0]
    npbase = out[npkey]
    for k, v in out.items():
        print_row(f"{k:8s}", f"{v:.3E}", f"{v/npbase:.3f}")

    print()


def main():

    N = 100000
    it = 10

    print("## `sort` Ascending Data\n")
    modmesh.call_profiler.reset()
    for _ in range(it):
        test_data = np.arange(0, N, dtype='uint32')
        profile_sort_np(test_data)
        profile_sort_sa(make_container(test_data))
    print_res(modmesh.call_profiler.result()["children"])

    print("## `sort` Descending Data\n")
    modmesh.call_profiler.reset()
    for _ in range(it):
        test_data = np.arange(N, 0, -1, dtype='uint32')
        profile_sort_np(test_data)
        profile_sort_sa(make_container(test_data))
    print_res(modmesh.call_profiler.result()["children"])

    print("## `sort` Random Data\n")
    modmesh.call_profiler.reset()
    for _ in range(it):
        test_data = np.arange(0, N, dtype='uint32')
        np.random.shuffle(test_data)
        profile_sort_np(test_data)
        profile_sort_sa(make_container(test_data))
    print_res(modmesh.call_profiler.result()["children"])

    print("## `argsort` Ascending Data\n")
    modmesh.call_profiler.reset()
    for _ in range(it):
        test_data = np.arange(0, N, dtype='uint32')
        profile_argsort_np(test_data)
        profile_argsort_sa(make_container(test_data))
    print_res(modmesh.call_profiler.result()["children"])

    print("## `argsort` Descending Data\n")
    modmesh.call_profiler.reset()
    for _ in range(it):
        test_data = np.arange(N, 0, -1, dtype='uint32')
        profile_argsort_np(test_data)
        profile_argsort_sa(make_container(test_data))
    print_res(modmesh.call_profiler.result()["children"])

    print("## `argsort` Random Data\n")
    modmesh.call_profiler.reset()
    for _ in range(it):
        test_data = np.arange(0, N, dtype='uint32')
        np.random.shuffle(test_data)
        profile_argsort_np(test_data)
        profile_argsort_sa(make_container(test_data))
    print_res(modmesh.call_profiler.result()["children"])

    print("## `take_along_axis` Ascending Data\n")
    modmesh.call_profiler.reset()
    for _ in range(it):
        test_data = np.arange(0, N, dtype='uint32')
        indices = np.arange(0, N-1, dtype='uint32')
        test_sa = make_container(test_data)
        idx_sa = make_container(indices)
        profile_take_along_axis_np(test_data, indices)
        profile_take_along_axis_sa(test_sa, idx_sa)
        profile_take_along_axis_neon(test_sa, idx_sa)
    print_res(modmesh.call_profiler.result()["children"])

    print("## `take_along_axis` Descending Data\n")
    modmesh.call_profiler.reset()
    for _ in range(it):
        test_data = np.arange(N, 0, -1, dtype='uint32')
        indices = np.arange(N-1, 0, -1, dtype='uint32')
        test_sa = make_container(test_data)
        idx_sa = make_container(indices)
        profile_take_along_axis_np(test_data, indices)
        profile_take_along_axis_sa(test_sa, idx_sa)
        profile_take_along_axis_neon(test_sa, idx_sa)
    print_res(modmesh.call_profiler.result()["children"])

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
        profile_take_along_axis_neon(test_sa, idx_sa)
    print_res(modmesh.call_profiler.result()["children"])


if __name__ == '__main__':
    main()

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
