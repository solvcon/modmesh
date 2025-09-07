# Copyright (c) 2019, Yung-Yu Chen <yyc@solvcon.net>
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# - Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
# - Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# - Neither the name of the copyright holder nor the names of its contributors
#   may be used to endorse or promote products derived from this software
#   without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.


"""
Import C++ implementation.
"""

# Use flake8 http://flake8.pycqa.org/en/latest/user/error-codes.html


# Try to import the C++ pilot code but easily give up.
enable = False
try:
    from _modmesh import pilot as _pilot_impl  # noqa: F401

    enable = True
except ImportError:
    pass

# pilot directory symbols organized by source files

# R3DWidget.hpp/.cpp
list_of_r3dwidget = [
    'R3DWidget',
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
_load(list_of_raxismark)
_load(list_of_rmanager)
_load(list_of_rpythonconsole)
_load(list_of_rcameracontroller)

del _load

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
