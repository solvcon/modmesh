# Getting Started

Because solvcon is still in its early stage of development, it needs to be
built from from source, and so are most of its dependencies, except:

- a C++23 compiler (gcc, clang, or MSVC)
- a Fortran compiler for a BLAS/LAPACK
- CMake 4.0.1 or newer

Virtual environments (venv, conda) are discouraged and not actively supported.
Read the documentation for building the dependencies and solvcon itself:

```{toctree}
:maxdepth: 1

build_dep
build_solvcon
```

<!-- vim: set ft=markdown ff=unix fenc=utf8 et sw=2 ts=2 sts=2 tw=79: -->
