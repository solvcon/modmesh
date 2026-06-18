# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
Multi-dimensional solvers for conservation laws, grouped by equation set;
currently the Euler-equation utilities.
"""

from . import euler

__all__ = [
    'euler',
]

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
