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
GUI for NACA airfoil shape
"""

from ... import core

from .. import _pilot_core as _pcore
from . import _naca

if _pcore.enable:
    from PySide6 import QtGui


def _add_menu_item(mainWindow, menu, text, tip, func):
    act = QtGui.QAction(text, mainWindow)
    act.setStatusTip(tip)
    act.triggered.connect(func)
    menu.addAction(act)


class Naca4Airfoil(object):
    """
    Provide pilot GUI control for the NACA 4-digit airfoil shape.
    """

    def __init__(self, mgr):
        self._mgr = mgr

    @property
    def _pycon(self):
        return self._mgr.pycon

    def populate_menu(self):
        _add_menu_item(
            mainWindow=self._mgr.mainWindow,
            menu=self._mgr.meshMenu,
            text="Sample: NACA 4-digit",
            tip="Draw a NACA 4-digit airfoil",
            func=self.sample_window,
        )

    def sample_window(self):
        """
        A simple example for drawing a couple of cubic Bezier curves based on
        an airfoil.
        """
        w = core.WorldFp64()
        naca4 = _naca.Naca4(number='0012', open_trailing_edge=False,
                            cosine_spacing=False)
        sampler = _naca.Naca4Sampler(w, naca4)
        sampler.populate_points(npoint=101, fac=5.0, off_x=0.0, off_y=2.0)
        if False:
            sampler.draw_line()
        else:
            sampler.draw_cbc()
        wid = _pcore.mgr.add3DWidget()
        wid.updateWorld(w)
        wid.showMark()

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
