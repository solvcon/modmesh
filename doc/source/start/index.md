# Getting Started

solvcon is a hybrid C++/Python project in its early stage of development.
The dependencies should be built from source.  The mmdv scripts under
[`contrib/dependency/{macos26,ubuntu2404}/`](https://github.com/solvcon/solvcon/tree/master/contrib/dependency/)
build the whole chain in user space.  The dependencies are:

- a C++23 compiler (gcc, clang, or MSVC)
- CMake 4.0.1 or newer
- Python 3.12 or newer, with development headers
- NumPy 2.0 or newer
- pybind11 2.12 or newer
- Qt6 and PySide6 for the {doc}`pilot GUI </pilot/index>`
- a Fortran compiler and a BLAS/LAPACK such as OpenBLAS, which NumPy builds
  against and which enables the eigen solvers

Virtual environments (venv, conda) are discouraged and not actively supported.

```{toctree}
:maxdepth: 2

build
quickstart
```

<!-- vim: set ft=markdown ff=unix fenc=utf8 et sw=2 ts=2 sts=2 tw=79: -->
