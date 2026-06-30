# Python API

The Python API has two layers.  The pure-Python modules under `solvcon/`
provide the high-level interface, and a compiled extension, `_solvcon`,
supplies the performance-critical types through pybind11 wrappers.

## Compiled extension wrappers

{py:mod}`solvcon.core` loads the pybind11 wrappers from the {py:mod}`_solvcon`
extension and re-exports them into the top-level {py:mod}`solvcon` namespace.

## GUI ({py:mod}`solvcon.pilot`)

The Qt-based {doc}`Pilot GUI </pilot/index>` lives in {py:mod}`solvcon.pilot`
and needs Qt6 and PySide6.

```{note}
This section is a placeholder for GUI Python API documentation.
```

## Testing utilities ({py:mod}`solvcon.testing`)

Helpers shared by the Python test suite.

```{eval-rst}
.. automodule:: solvcon.testing
   :members:
   :undoc-members:
   :show-inheritance:
```

<!-- vim: set ft=markdown ff=unix fenc=utf8 et sw=2 ts=2 sts=2 tw=79: -->
