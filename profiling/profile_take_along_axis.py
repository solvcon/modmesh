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


@profile_function
def profile_take_along_axis_np(narr, indices):
    return np.take_along_axis(narr, indices, -1)


@profile_function
def profile_take_along_axis_sa(sarr, indices):
    return sarr.take_along_axis(indices)


@profile_function
def profile_take_along_axis_simd(sarr, indices):
    return sarr.take_along_axis_simd(indices)


def profile_take_along_axis(pow, it=10):
    N = 2 ** pow
    ORDER = ["", "K", "M", "G", "T"][pow // 10]
    dtype = ["uint8", "uint16", "uint32", "uint64"][pow // 8]

    modmesh.call_profiler.reset()
    for _ in range(it):
        test_data = np.arange(0, N, dtype=dtype)
        indices = np.arange(0, N-1, dtype=dtype)
        np.random.shuffle(test_data)
        np.random.shuffle(indices)
        test_sa = make_container(test_data)
        idx_sa = make_container(indices)

        profile_take_along_axis_np(test_data, indices)
        profile_take_along_axis_sa(test_sa, idx_sa)
        profile_take_along_axis_simd(test_sa, idx_sa)

    res = modmesh.call_profiler.result()["children"]

    print(f"## N = {2 ** (pow % 10)}{ORDER} type: {dtype}\n")
    out = {}
    for r in res:
        name = r["name"].replace("profile_take_along_axis_", "")
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
    pow = 7
    it = 7

    for _ in range(it):
        profile_take_along_axis(pow)
        pow = pow + 3


if __name__ == "__main__":
    main()

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
