# modmesh documentation (prototype)

modmesh is a *hybrid C++/Python* numerical library.  This is a minimal, working
prototype of a documentation system for modmesh, built on **Sphinx**.

## Layout

```
doc/
  Makefile            html / doxygen / clean targets
  Doxyfile            Doxygen config (XML only, consumed by breathe)
  requirements.txt    Python build dependencies
  source/
    conf.py           Sphinx configuration (annotated)
    index.md          landing page
    refs.bib          bibliography
    api/python.md     autodoc of modmesh.onedim
    api/cpp.md         breathe of modmesh::ConcreteBuffer
```

## Build

```sh
pip install -r requirements.txt   # Python deps
make doxygen                      # optional: C++ API XML (needs doxygen)
make html                         # -> build/html/index.html
```

`make html` works without `make doxygen`; the C++ API page simply renders
empty (a warning, not an error) until the XML exists.

<!-- vim: set ft=markdown ff=unix fenc=utf8 et sw=2 ts=2 sts=2 tw=79: -->
