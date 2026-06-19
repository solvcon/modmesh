# Configuration file for the Sphinx documentation builder.
#
# Minimal prototype of a hybrid C++/Python documentation toolchain for
# solvcon.  The pieces, and why each is here:
#
#   myst_parser          author pages in Markdown (matches the repo
#                        culture: README.md, STYLE.md, CLAUDE.md)
#   autodoc + napoleon   Python API straight from docstrings
#   breathe              C++ API bridged from Doxygen XML
#   mathjax              CESE / conservation-law equations
#   sphinxcontrib.bibtex academic citations (the CESE literature)
#
# Build with ``make html`` from the doc/ directory.  Run ``make doxygen``
# first if you want the C++ API pages populated.

import os
import sys

# Make the in-tree ``solvcon`` package importable for autodoc.  The repo
# root is two levels up from this file (doc/source/conf.py).
sys.path.insert(0, os.path.abspath("../.."))
sys.path.insert(0, os.path.abspath("../ext"))

# -- Project information ----------------------------------------------------

project = "solvcon"
copyright = "2019-2026, Yung-Yu Chen and solvcon contributors"
author = "Yung-Yu Chen and solvcon contributors"

# -- General configuration --------------------------------------------------

extensions = [
    "myst_parser",            # Markdown authoring
    "sphinx.ext.autodoc",     # pull Python docstrings
    "sphinx.ext.autosummary",  # API summary tables
    "sphinx.ext.napoleon",    # NumPy / Google docstring styles
    "sphinx.ext.viewcode",    # link to highlighted source
    "sphinx.ext.intersphinx",  # cross-link python / numpy docs
    "sphinx.ext.mathjax",     # render LaTeX math
    "breathe",                # C++ via Doxygen XML
    "sphinxcontrib.bibtex",   # citations
    "pstake",                 # PSTricks .tex -> PNG at build time
]

# MyST Markdown extensions: $...$ and $$...$$ math, amsmath
# environments, ::: fences, and definition lists.
myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "colon_fence",
    "deflist",
]

autosummary_generate = True

# The compiled _solvcon extension is a build artifact, absent on a clean
# checkout (and on Read the Docs).  Mock it so autodoc can still import
# the pure-Python layers that sit on top of it.
autodoc_mock_imports = ["_solvcon"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}

# -- Breathe (C++ bridge) ---------------------------------------------------

# Points at the Doxygen XML produced by ``make doxygen`` (see Doxyfile).
breathe_projects = {"solvcon": "../build/doxygen/xml"}
breathe_default_project = "solvcon"

# -- sphinxcontrib.bibtex ---------------------------------------------------

bibtex_bibfiles = ["reference.bib"]

# -- HTML output ------------------------------------------------------------

# pydata-sphinx-theme is the de-facto standard across the scientific
# Python stack (NumPy, SciPy, pandas, matplotlib).  See doc/README.md
# for the sphinx-book-theme alternative tuned for teaching material.
html_theme = "pydata_sphinx_theme"
html_title = "solvcon"
html_static_path = ["_static"]

# -- MathJax configuration --------------------------------------------------

mathjax3_config = {
    "tex": {
        "macros": {
            "defeq": r"\overset{\text{def}}{=}",
            "dif": r"\mathrm{d}",
        },
        "packages": {"[+]": ["cancel"]},
    }
}

numfig = True

# vim: set ft=python ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
