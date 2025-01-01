# Copyright (c) 2021, Yung-Yu Chen <yyc@solvcon.net>
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
Show meshes.
"""

import os

from PySide6 import QtCore, QtWidgets, QtGui

from .. import core

__all__ = [  # noqa: F822
    'GmshFileDialog',
]


def _add_menu_item(mainWindow, menu, text, tip, func):
    act = QtGui.QAction(text, mainWindow)
    act.setStatusTip(tip)
    act.triggered.connect(func)
    menu.addAction(act)


class GmshFileDialog(QtCore.QObject):
    def __init__(self, *args, **kw):
        self._mgr = kw.pop('mgr')
        super(GmshFileDialog, self).__init__(*args, **kw)
        self._diag = QtWidgets.QFileDialog()
        self._diag.setFileMode(QtWidgets.QFileDialog.ExistingFile)
        self._diag.setDirectory(self._get_initial_path())
        self._diag.setWindowTitle('Open Gmsh file ...')

    def run(self):
        self._diag.open(self, QtCore.SLOT('on_finished()'))

    def populate_menu(self):
        _add_menu_item(
            mainWindow=self._mgr.mainWindow,
            menu=self._mgr.fileMenu,
            text="Open Gmsh file",
            tip="Open Gmsh file",
            func=self.run,
        )

    @QtCore.Slot()
    def on_finished(self):
        filenames = []
        for path in self._diag.selectedFiles():
            filenames.append(path)
        self._load_gmsh_file(filename=filenames[0])

    @staticmethod
    def _get_initial_path():
        """
        Search for `tests/data/rectangle.msh` and return the directory holding
        it.  If not found, return an empty string.

        :return: The holding directory in absolute path or empty string.
        """
        found = ''
        for dp in ('.', core.__file__):
            dp = os.path.dirname(os.path.abspath(dp))
            dp2 = os.path.dirname(dp)
            while dp != dp2:
                tp = os.path.join(dp, "tests", "data")
                fp = os.path.join(tp, "rectangle.msh")
                if os.path.exists(fp):
                    found = tp
                    break
                dp = dp2
                dp2 = os.path.dirname(dp)
            if found:
                break
        return found

    def _load_gmsh_file(self, filename):
        if not os.path.exists(filename):
            self._pycon.writeToHistory(f"{filename} does not exist\n")
            return

        with open(filename, 'rb') as fobj:
            data = fobj.read()
        self._pycon.writeToHistory(f"gmsh mesh file {filename} is read\n")
        gmsh = core.Gmsh(data)
        mh = gmsh.to_block()
        self._pycon.writeToHistory("StaticMesh object created from gmsh\n")
        # Open a sub window for triangles and quadrilaterals:
        w = self._mgr.add3DWidget()
        w.updateMesh(mh)
        w.showMark()
        self._pycon.writeToHistory(f"nedge: {mh.nedge}\n")

    @property
    def _pycon(self):
        return self._mgr.pycon

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
