# solvcon documentation

solvcon is a *hybrid C++/Python* numerical library.  This directory contains
the Sphinx-based documentation.

## Build

```sh
pip install -r requirements.txt   # Python deps
make doxygen                      # optional: C++ API XML (needs doxygen)
make html                         # -> build/html/index.html
```

`make html` works without `make doxygen`; the C++ API page simply renders
empty (a warning, not an error) until the XML exists.

<!-- vim: set ft=markdown ff=unix fenc=utf8 et sw=2 ts=2 sts=2 tw=79: -->
