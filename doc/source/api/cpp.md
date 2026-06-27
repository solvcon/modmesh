# C++ API

C++ API documentation written in Doxygen format in the C++ source code, bridged
by [breathe](https://www.breathe-doc.org/).

## Array and memory buffer

```{eval-rst}
.. doxygenclass:: solvcon::ConcreteBuffer
   :project: solvcon
   :members:
```

## Pilot domain viewer

The pilot's 3D viewer (see {doc}`../pilot/domain`) renders through
[QRhi](https://doc.qt.io/qt-6/qrhi.html), Qt's portable graphics abstraction.

{cpp:class}`solvcon::RDomainWidget` is the Python-facing control object.

```{eval-rst}
.. doxygenclass:: solvcon::RDomainWidget
   :project: solvcon
   :members:

.. doxygenclass:: solvcon::RScene
   :project: solvcon
   :members:

.. doxygenclass:: solvcon::RCameraController
   :project: solvcon
   :members:

.. doxygenclass:: solvcon::RDrawable
   :project: solvcon
   :members:

.. doxygenclass:: solvcon::RMeshFrame
   :project: solvcon
   :members:

.. doxygenclass:: solvcon::RField
   :project: solvcon
   :members:

.. doxygenclass:: solvcon::RBoundary
   :project: solvcon
   :members:

.. doxygenclass:: solvcon::RAxisGizmo
   :project: solvcon
   :members:

.. doxygenclass:: solvcon::RMaterial
   :project: solvcon
   :members:
```

## Notes on generating this C++ API document

Doxygen parses the existing ``///`` and ``/**`` comments in `cpp/solvcon/` into
XML; breathe turns that XML into the reference below.

```{important}
Run ``make doxygen`` once before ``make html`` so the Doxygen XML exists.
Until then this directive renders empty (a warning, not an error).
```

<!-- vim: set ft=markdown ff=unix fenc=utf8 et sw=2 ts=2 sts=2 tw=79: -->
