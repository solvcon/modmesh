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
Graphical-user interface code
"""

# Use flake8 http://flake8.pycqa.org/en/latest/user/error-codes.html


import sys
import importlib

# Try to import the C++ pilot code but easily give up.
enable = False
try:
    from _modmesh import pilot as _vimpl  # noqa: F401
    enable = True
except ImportError:
    pass

if enable:
    from PySide6.QtGui import QAction


_from_impl = [  # noqa: F822
    'R3DWidget',
    'RLine',
    'RPythonConsoleDockWidget',
    'RManager',
    'RCameraController',
    'mgr',
]

__all__ = _from_impl + [  # noqa: F822
    'launch',
]


def _load():
    if enable:
        for name in _from_impl:
            globals()[name] = getattr(_vimpl, name)


_load()
del _load


def populate_menu():
    wm = _vimpl.RManager.instance

    def _addAction(menu, text, tip, funcname):
        act = QAction(text, wm.mainWindow)
        act.setStatusTip(tip)
        if callable(funcname):
            act.triggered.connect(lambda *a: funcname())
        elif funcname:
            modname, funcname = funcname.rsplit('.', maxsplit=1)
            mod = importlib.import_module(modname)
            func = getattr(mod, funcname)
            act.triggered.connect(lambda *a: func())
        menu.addAction(act)

    _addAction(
        menu=wm.fileMenu,
        text="New file (dummy)",
        tip="Create new file",
        funcname=lambda: print("This is only a demo: Create new file!"),
    )

    if sys.platform != 'darwin':
        _addAction(
            menu=wm.fileMenu,
            text="Exit",
            tip="Exit the application",
            funcname=lambda: wm.quit(),
        )

    _addAction(
        menu=wm.oneMenu,
        text="Euler solver",
        tip="One-dimensional shock-tube problem with Euler solver",
        funcname="modmesh.app.euler1d.load_app",
    )

    _addAction(
        menu=wm.meshMenu,
        text="Sample: mesh of a triangle (2D)",
        tip="Create a very simple sample mesh of a triangle",
        funcname="modmesh.gui.sample_mesh.mesh_triangle",
    )

    _addAction(
        menu=wm.meshMenu,
        text="Sample: mesh of a tetrahedron (3D)",
        tip="Create a very simple sample mesh of a tetrahedron",
        funcname="modmesh.gui.sample_mesh.mesh_tetrahedron",
    )

    _addAction(
        menu=wm.meshMenu,
        text="Sample: mesh of \"solvcon\" text in 2D",
        tip="Create a sample mesh drawing a text string of \"solvcon\"",
        funcname="modmesh.gui.sample_mesh.mesh_solvcon_2dtext",
    )

    _addAction(
        menu=wm.meshMenu,
        text="Sample: 2D mesh in a rectangle",
        tip="Triangular mesh in a rectangle",
        funcname="modmesh.gui.sample_mesh.mesh_rectangle",
    )

    _addAction(
        menu=wm.meshMenu,
        text="Sample: 3D mesh of mixed elements",
        tip="Create a very simple sample mesh of mixed elements in 3D",
        funcname="modmesh.gui.sample_mesh.mesh_3dmix",
    )

    _addAction(
        menu=wm.meshMenu,
        text="Sample: NACA 4-digit",
        tip="Draw a NACA 4-digit airfoil",
        funcname="modmesh.gui.naca.runmain",
    )

    _addAction(
        menu=wm.addonMenu,
        text="Load sample_mesh",
        tip="Load sample_mesh",
        funcname="modmesh.app.sample_mesh.load_app",
    )

    _addAction(
        menu=wm.addonMenu,
        text="Load linear_wave",
        tip="Load linear_wave",
        funcname="modmesh.app.linear_wave.load_app",
    )

    _addAction(
        menu=wm.addonMenu,
        text="Load bad_euler1d",
        tip="Load bad_euler1d",
        funcname="modmesh.app.bad_euler1d.load_app",
    )

    _addAction(
        menu=wm.windowMenu,
        text="(empty)",
        tip="(empty)",
        funcname=None,
    )


def launch(name="pilot", size=(1000, 600)):
    """
    The entry point of the pilot GUI application.

    :param name: Main window name.
    :param size: Main window size.
    :return: nothing
    """
    wm = _vimpl.RManager.instance
    wm.setUp()
    wm.windowTitle = name
    wm.resize(w=size[0], h=size[1])
    populate_menu()
    wm.show()
    return wm.exec()

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
