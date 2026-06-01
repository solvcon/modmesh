# C++ API

This page is bridged from Doxygen by ``breathe``.  Doxygen parses the existing
``///`` and ``/**`` comments in `cpp/modmesh/` into XML; breathe turns that XML
into the reference below.

```{important}
Run ``make doxygen`` once before ``make html`` so the Doxygen XML exists.
Until then this directive renders empty (a warning, not an error).
```

Below is an example using `modmesh::ConcreteBuffer`.

```{eval-rst}
.. doxygenclass:: modmesh::ConcreteBuffer
   :project: modmesh
   :members:
```

<!-- vim: set ft=markdown ff=unix fenc=utf8 et sw=2 ts=2 sts=2 tw=79: -->
