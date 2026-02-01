# Copyright (c) 2025, Yung-Yu Chen <yyc@solvcon.net>
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


from . import _pilot_core as _pcore
from PySide6 import QtCore, QtGui


class PilotFeature(QtCore.QObject):
    """
    Base class to house common GUI code for prototyping pilot features.

    :ivar _mgr:
        The modmesh pilot application manager implemented with Qt in C++.
    :vartype mgr: modmesh.pilot.RManager
    """

    def __init__(self, *args, **kw):
        """
        :param mgr:
            The modmesh pilot application manager implemented with Qt in C++.
        :type mgr: modmesh.pilot.RManager
        """
        self._mgr = kw.pop('mgr')
        if not isinstance(self._mgr, _pcore.RManager):
            raise TypeError(
                "'mgr' must be an instance of 'modmesh.pilot.RManager'")
        super(PilotFeature, self).__init__(*args, **kw)

    @property
    def _pycon(self):
        """
        :rtype: modmesh.pilot.RPythonConsoleDockWidget
        """
        return self._mgr.pycon

    @property
    def _mainWindow(self):
        """
        :rtype: PySide6.QtWidget.QMainWindow
        """
        return self._mgr.mainWindow

    def _add_menu_item(self, menu, text, tip, func):
        """
        Add an item to the corresponding menu.

        :param menu: The menu to add the item to.
        :type menu: PySide6.QtWidget.QMenu
        :param text: Menu description string.
        :param tip: Menu tip string.
        :param func: Python callable.

        :return: None
        """
        act = QtGui.QAction(text, self._mainWindow)
        act.setStatusTip(tip)
        act.triggered.connect(func)
        menu.addAction(act)


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
