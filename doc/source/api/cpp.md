# C++ API

C++ API documentation written in Doxygen format in the C++ source code, bridged
by [breathe](https://www.breathe-doc.org/).

## Core and memory

```{eval-rst}
.. doxygengroup:: group_core
   :project: solvcon
   :content-only:
```

## Mesh

```{eval-rst}
.. doxygengroup:: group_mesh
   :project: solvcon
   :content-only:
```

## Numerics

```{eval-rst}
.. doxygengroup:: group_numerics
   :project: solvcon
   :content-only:
```

## Geometry

```{eval-rst}
.. doxygengroup:: group_geometry
   :project: solvcon
   :content-only:
```

## One-dimensional solvers

```{eval-rst}
.. doxygengroup:: group_onedim
   :project: solvcon
   :content-only:
```

## Multi-dimensional solvers

```{eval-rst}
.. doxygengroup:: group_multidim
   :project: solvcon
   :content-only:
```

## Input and output

```{eval-rst}
.. doxygengroup:: group_inout
   :project: solvcon
   :content-only:
```

## Domain visualizer

The 2/3D visualizer (see {doc}`../pilot/domain`) renders through
[QRhi](https://doc.qt.io/qt-6/qrhi.html), Qt's portable graphics abstraction.

{cpp:class}`solvcon::RDomainWidget` is the Python-facing control object.

```{eval-rst}
.. doxygengroup:: group_domain
   :project: solvcon
   :content-only:
```

## Notes on generating this C++ API document

Doxygen parses the existing ``///`` and ``/**`` comments in `cpp/solvcon/` into
XML; breathe turns that XML into the reference below.

```{important}
Run ``make doxygen`` once before ``make html`` so the Doxygen XML exists.
Until then this directive renders empty (a warning, not an error).
```

<!-- vim: set ft=markdown ff=unix fenc=utf8 et sw=2 ts=2 sts=2 tw=79: -->
