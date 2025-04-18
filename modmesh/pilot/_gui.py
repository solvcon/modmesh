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

from . import _pilot_core as _pcore
from . import airfoil

if _pcore.enable:
    from PySide6.QtGui import QAction
    from . import _mesh
    from . import _euler1d
    from . import _burgers1d
    from . import _svg

__all__ = [  # noqa: F822
    'controller',
    'launch',
]


def launch():
    return controller.launch()


class _Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kw):
        if cls not in cls._instances:
            cls._instances[cls] = super(_Singleton, cls).__call__(*args, **kw)
        return cls._instances[cls]


class _Controller(metaclass=_Singleton):
    def __init__(self):
        # Do not construct any Qt member objects before calling launch(), or
        # Windows may "exited with code -1073740791."
        self._rmgr = None
        self.gmsh_dialog = None
        self.svg_dialog = None
        self.sample_mesh = None
        self.recdom = None
        self.naca4airfoil = None
        self.eulerone = None
        self.burgers = None

    def __getattr__(self, name):
        return None if self._rmgr is None else getattr(self._rmgr, name)

    def launch(self, name="pilot", size=(1000, 600)):
        self._rmgr = _pcore.RManager.instance
        self._rmgr.setUp()
        self._rmgr.windowTitle = name
        self._rmgr.resize(w=size[0], h=size[1])
        self.gmsh_dialog = _mesh.GmshFileDialog(mgr=self._rmgr)
        self.svg_dialog = _svg.SVGFileDialog(mgr=self._rmgr)
        self.sample_mesh = _mesh.SampleMesh(mgr=self._rmgr)
        self.recdom = _mesh.RectangularDomain(mgr=self._rmgr)
        self.naca4airfoil = airfoil.Naca4Airfoil(mgr=self._rmgr)
        self.eulerone = _euler1d.Euler1DApp(mgr=self._rmgr)
        self.burgers = _burgers1d.Burgers1DApp(mgr=self._rmgr)
        self.populate_menu()
        self._rmgr.show()
        return self._rmgr.exec()

    def populate_menu(self):
        wm = self._rmgr

        def _addAction(menu, text, tip, func, checkable=False, checked=False):
            act = QAction(text, wm.mainWindow)
            act.setStatusTip(tip)
            act.setCheckable(checkable)
            if checkable:
                act.setChecked(checked)
            if callable(func):
                act.triggered.connect(lambda *a: func())
            elif func:
                modname, funcname = func.rsplit('.', maxsplit=1)
                mod = importlib.import_module(modname)
                func = getattr(mod, funcname)
                act.triggered.connect(lambda *a: func())
            menu.addAction(act)

        self.gmsh_dialog.populate_menu()
        self.svg_dialog.populate_menu()
        self.sample_mesh.populate_menu()
        self.naca4airfoil.populate_menu()
        self.recdom.populate_menu()
        self.eulerone.populate_menu()
        self.burgers.populate_menu()

        if sys.platform != 'darwin':
            _addAction(
                menu=wm.fileMenu,
                text="Exit",
                tip="Exit the application",
                func=lambda: wm.quit(),
            )

        _addAction(
            menu=wm.addonMenu,
            text="(To be deprecated) load linear_wave",
            tip="Load linear_wave",
            func="modmesh.app.linear_wave.load_app",
        )

        _addAction(
            menu=wm.addonMenu,
            text="(To be deprecated) load bad_euler1d",
            tip="Load bad_euler1d",
            func="modmesh.app.bad_euler1d.load_app",
        )

        _addAction(
            menu=wm.windowMenu,
            text="Console",
            tip="Open / Close Console",
            func=wm.toggleConsole,
            checkable=True,
            checked=True,
        )


controller = _Controller()

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
