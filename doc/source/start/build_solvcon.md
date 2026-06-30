# Build solvcon

All workflows are driven through `make` from the repository root.  The Makefile
wraps CMake and sets `PYTHONPATH` so the in-tree `_solvcon` extension is picked
up without installation.

- `make`: build the `_solvcon` Python extension (the default target).
- `make pilot`: build the Qt pilot GUI binary.
- `make clean` / `make cmakeclean`: remove build artifacts.

Release builds (the default) land in `build/rel<pyvminor>` and debug builds in
`build/dbg<pyvminor>`, where `<pyvminor>` is the active Python major and minor
version, e.g. `314`.

## Build Options

Key options can be set in `setup.mk` (which is read by `Makefile`) or as
environment variables:

| Variable             | Default   | Purpose                          |
|:---------------------|:----------|:---------------------------------|
| `CMAKE_BUILD_TYPE`   | `Release` | `Release` or `Debug`             |
| `BUILD_QT`           | `ON`      | build the Qt GUI components      |
| `BUILD_METAL`        | `OFF`     | build Metal GPU support (macOS)  |
| `SOLVCON_PROFILE`    | `OFF`     | enable the runtime profiler      |
| `USE_CLANG_TIDY`     | `OFF`     | run clang-tidy during the build  |

After building, run the tests as described in {doc}`/devguide/testing`.

<!-- vim: set ft=markdown ff=unix fenc=utf8 et sw=2 ts=2 sts=2 tw=79: -->
