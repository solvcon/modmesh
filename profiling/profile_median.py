# Copyright (c) 2025, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

import json

import numpy
import solvcon


def make_container(data):
    if numpy.isdtype(data.dtype, numpy.uint8):
        return solvcon.SimpleArrayUint8(array=data)
    elif numpy.isdtype(data.dtype, numpy.uint16):
        return solvcon.SimpleArrayUint16(array=data)
    elif numpy.isdtype(data.dtype, numpy.uint32):
        return solvcon.SimpleArrayUint32(array=data)
    elif numpy.isdtype(data.dtype, numpy.uint64):
        return solvcon.SimpleArrayUint64(array=data)


def main():
    solvcon.call_profiler.reset()
    simple_array = make_container(numpy.arange(0, 1e6, dtype="uint8"))
    simple_array.median()

    print(json.dumps(solvcon.call_profiler.result().get("children"), indent=4))


if __name__ == "__main__":
    main()

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
