from enum import IntEnum
import functools
import numpy as np
import modmesh


def _profile_function(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        _ = modmesh.CallProfilerProbe(func.__name__)
        result = func(*args, **kwargs)
        return result
    return wrapper


def _make_container(data):
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


class OpList(IntEnum):
    ADD = 0
    SUB = 1
    MUL = 2
    DIV = 3


@_profile_function
def add_np(src1, src2):
    return np.add(src1, src2)


@_profile_function
def add_sa(src1, src2):
    return src1.add(src2)


@_profile_function
def add_simd(src1, src2):
    return src1.add_simd(src2)


@_profile_function
def sub_np(src1, src2):
    return np.subtract(src1, src2)


@_profile_function
def sub_sa(src1, src2):
    return src1.sub(src2)


@_profile_function
def sub_simd(src1, src2):
    return src1.sub_simd(src2)


@_profile_function
def mul_np(src1, src2):
    return np.multiply(src1, src2)


@_profile_function
def mul_sa(src1, src2):
    return src1.mul(src2)


@_profile_function
def mul_simd(src1, src2):
    return src1.mul_simd(src2)


@_profile_function
def div_np(src1, src2):
    return np.divide(src1, src2)


@_profile_function
def div_sa(src1, src2):
    return src1.div(src2)


@_profile_function
def div_simd(src1, src2):
    return src1.div_simd(src2)


@_profile_function
def iadd_np(src1, src2):
    src1 += src2


@_profile_function
def iadd_sa(src1, src2):
    src1.iadd(src2)


@_profile_function
def iadd_simd(src1, src2):
    src1.iadd_simd(src2)


@_profile_function
def isub_np(src1, src2):
    src1 -= src2


@_profile_function
def isub_sa(src1, src2):
    src1.isub(src2)


@_profile_function
def isub_simd(src1, src2):
    src1.isub_simd(src2)


@_profile_function
def imul_np(src1, src2):
    src1 *= src2


@_profile_function
def imul_sa(src1, src2):
    src1.imul(src2)


@_profile_function
def imul_simd(src1, src2):
    src1.imul_simd(src2)


@_profile_function
def idiv_np(src1, src2):
    src1 /= src2


@_profile_function
def idiv_sa(src1, src2):
    src1.idiv(src2)


@_profile_function
def idiv_simd(src1, src2):
    src1.idiv_simd(src2)


def profile_np(lhs, rhs, op, inplace=False, **kwargs):
    if not inplace:
        func = [add_np, sub_np, mul_np, div_np]
    else:
        func = [iadd_np, isub_np, imul_np, idiv_np]

    func[op](lhs, rhs)


def _profile_sa(lhs, rhs, op, inplace=False, **kwargs):
    if not inplace:
        func = [add_sa, sub_sa, mul_sa, div_sa]
    else:
        func = [iadd_sa, isub_sa, imul_sa, idiv_sa]

    func[op](lhs, rhs)


def _profile_sa_simd(lhs, rhs, op, inplace=False, **kwargs):
    if not inplace:
        func = [add_simd, sub_simd, mul_simd, div_simd]
    else:
        func = [iadd_simd, isub_simd, imul_simd, idiv_simd]

    func[op](lhs, rhs)


def profile_sa(lhs, rhs, op, inplace=False, **kwargs):
    _profile_sa(_make_container(lhs), _make_container(rhs), op, inplace)


def profile_sa_simd(lhs, rhs, op, inplace=False, **kwargs):
    _profile_sa_simd(_make_container(lhs), _make_container(rhs), op, inplace)


def get_options():
    return [(k, v.keys()) for k, v in StaticData.opt_dict.items()]


def run(options):
    func = StaticData.opt_dict["function"][options["function"]]
    type_max = StaticData.opt_dict["type"][options["type"]][1]
    lhs_base = (StaticData.lhs * type_max).astype(options["type"])
    rhs_base = (StaticData.rhs * type_max).astype(options["type"])
    operation = StaticData.opt_dict["operation"][options["operation"]]
    inplace = StaticData.opt_dict["In-place"][options["In-place"]]

    for _ in range(10):
        func(lhs_base.copy(), rhs_base.copy(), operation, inplace)


class StaticData:
    N = 2 ** 22
    lhs = np.random.rand(N)
    rhs = np.random.rand(N)
    opt_dict = {
        "function": {
            "profile_np": profile_np,
            "profile_sa": profile_sa,
            "profile_sa_simd": profile_sa_simd,
        },
        "operation": {
            "Addition": OpList.ADD,
            "Subtraction": OpList.SUB,
            "Multiplication": OpList.MUL,
            "Division": OpList.DIV
        },
        "type": {
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
        },
        "In-place": {
            "In-place": True,
            "Out-of-place": False
        }
    }


if __name__ == "__main__":
    pass
