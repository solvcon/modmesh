# Testing

Tests are driven through `make` from the repository root.  Python tests are the
default and live in `tests/` as `test_*.py`; C++ tests live in `gtests/` as
`test_nopython_*.cpp` and are reserved for behaviour that cannot or should not
be reached from Python.

- `make pytest` -- run the full Python test suite.
- `make pytest PYTEST_OPTS="tests/test_buffer.py::SimpleArrayBasicTC"` --
  forward options verbatim to pytest to run a subset.
- `make run_pilot_pytest` -- Python tests that need the pilot GUI.
- `make gtest` -- build and run the full C++ test suite.
- `make pyprof` -- run the profiling benchmarks (see {doc}`/system/profiling`).

After `make gtest` has built the binary, a single C++ test can be run directly:

```sh
./build/rel<pyvminor>/gtests/run_gtest --gtest_filter=Suite.Test
```

where `<pyvminor>` is the active Python major and minor version, e.g. `314`.

<!-- vim: set ft=markdown ff=unix fenc=utf8 et sw=2 ts=2 sts=2 tw=79: -->
