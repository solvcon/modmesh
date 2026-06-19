# Copyright (c) 2025, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


from . import _pilot_core as _pcore
from PySide6 import QtCore, QtGui


class PilotFeature(QtCore.QObject):
    """
    Base class to house common GUI code for prototyping pilot features.

    :ivar _mgr:
        The solvcon pilot application manager implemented with Qt in C++.
    :vartype mgr: solvcon.pilot.RManager
    """

    def __init__(self, *args, **kw):
        """
        :param mgr:
            The solvcon pilot application manager implemented with Qt in C++.
        :type mgr: solvcon.pilot.RManager
        """
        self._mgr = kw.pop('mgr')
        if not isinstance(self._mgr, _pcore.RManager):
            raise TypeError(
                "'mgr' must be an instance of 'solvcon.pilot.RManager'")
        super(PilotFeature, self).__init__(*args, **kw)

    @property
    def _pycon(self):
        """
        :rtype: solvcon.pilot.RPythonConsoleDockWidget
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
