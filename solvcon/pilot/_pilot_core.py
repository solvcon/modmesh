# Copyright (c) 2019, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
Import C++ implementation.
"""

# Use flake8 http://flake8.pycqa.org/en/latest/user/error-codes.html


# Try to import the C++ pilot code but easily give up.
enable = False
try:
    from _solvcon import pilot as _pilot_impl  # noqa: F401

    enable = True
except ImportError:
    pass

# pilot directory symbols organized by source files

# R3DWidget.hpp/.cpp
list_of_r3dwidget = [
    'R3DWidget',
]

# R2DWidget.hpp/.cpp
list_of_r2dwidget = [
    'R2DWidget',
]

# RAxisMark.hpp/.cpp
list_of_raxismark = [
    'RLine',
]

# RManager.hpp/.cpp
list_of_rmanager = [
    'RManager',
    'mgr',
]

# RPythonConsoleDockWidget.hpp/.cpp
list_of_rpythonconsole = [
    'RPythonConsoleDockWidget',
]

# RCameraController.hpp/.cpp
list_of_rcameracontroller = [
    'RCameraController',
]

_from_impl = (  # noqa: F822
    list_of_r3dwidget +
    list_of_r2dwidget +
    list_of_raxismark +
    list_of_rmanager +
    list_of_rpythonconsole +
    list_of_rcameracontroller
)

__all__ = _from_impl + [  # noqa: F822
    'enable',
]


def _load(symbol_list):
    if enable:
        for name in symbol_list:
            globals()[name] = getattr(_pilot_impl, name)
    else:
        for name in symbol_list:
            globals()[name] = None


_load(list_of_r3dwidget)
_load(list_of_r2dwidget)
_load(list_of_raxismark)
_load(list_of_rmanager)
_load(list_of_rpythonconsole)
_load(list_of_rcameracontroller)

del _load

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
