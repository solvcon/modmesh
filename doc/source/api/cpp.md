# C++ API

C++ API documentation written in Doxygen format in the C++ source code, bridged
by [breathe](https://www.breathe-doc.org/).  Each page below renders one Doxygen
group.

```{toctree}
:maxdepth: 1

cpp_core
cpp_numerics
cpp_geometry
cpp_mesh
cpp_onedim
cpp_multidim
cpp_inout
cpp_canvas
cpp_domain
```

```{note}
Run ``make doxygen`` once before ``make html`` so the Doxygen XML exists.
Until then these directives render empty (a warning, not an error). Doxygen
parses the existing ``///`` and ``/**`` comments in `cpp/solvcon/` into XML;
breathe turns that XML into the per-group reference pages.
```


<!-- vim: set ft=markdown ff=unix fenc=utf8 et sw=2 ts=2 sts=2 tw=79: -->
