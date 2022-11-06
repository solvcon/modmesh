# Copyright (c) 2022, Yung-Yu Chen <yyc@solvcon.net>
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


import numpy as np

from matplotlib.backends.qt_compat import QtWidgets
from matplotlib.backends.backend_qtagg import (
    FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
from matplotlib.figure import Figure

from .. import view
from .. import spacetime as libst


def load_app():
    cmd = "win, svr = mm.app.bad_euler1d.run(animate=True, interval=10)"
    view.app().pycon.command = cmd


class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self._main = QtWidgets.QWidget()
        self.setCentralWidget(self._main)
        layout = QtWidgets.QVBoxLayout(self._main)

        self.canvas = FigureCanvas(Figure(figsize=(15, 10)))
        # Ideally one would use self.addToolBar here, but it is slightly
        # incompatible between PyQt6 and other bindings, so we just add the
        # toolbar as a plain widget instead.
        layout.addWidget(self.canvas)
        layout.addWidget(NavigationToolbar(self.canvas, self))

        self.svr = None
        self.ax = self.canvas.figure.subplots()
        self.timer = None
        self.line = None

    def set_solver(self, svr, interval):
        """
        :param svr: Solver
        :param interval: milliseconds
        :return: nothing
        """
        self.svr = svr
        x = svr.xctr() / np.pi
        self.line0, = self.ax.plot(x, svr.get_so0(0), '+')
        self.line1, = self.ax.plot(x, svr.get_so0(1), '+')
        self.line2, = self.ax.plot(x, svr.get_so0(2), '+')
        self.timer = self.canvas.new_timer(interval)
        self.timer.add_callback(self._update_canvas)
        self.timer.start()

    def _update_canvas(self):
        self.svr.march_alpha2(1)
        x = self.svr.xctr() / np.pi
        self.line0.set_data(x, self.svr.get_so0(0))
        self.line1.set_data(x, self.svr.get_so0(1))
        self.line2.set_data(x, self.svr.get_so0(2))
        self.line0.figure.canvas.draw()
        self.line1.figure.canvas.draw()
        self.line2.figure.canvas.draw()
        cfl = self.svr.get_cfl()
        print("CFL:", "min", cfl.min(), "max", cfl.max())


def run(animate, interval=10):
    grid = libst.Grid(0, 4 * 2 * np.pi, 4 * 64)

    cfl = 1
    dx = (grid.xmax - grid.xmin) / grid.ncelm
    dt = dx * cfl
    svr = libst.BadEuler1DSolver(grid=grid, time_increment=dt)

    # Initialize.
    d = (grid.ncelm+1) // 2
    u0 = np.zeros(grid.ncelm+1, dtype='float64')
    u0[0:d] = 1.0
    u0[d:] = 0.125
    u1 = np.zeros(grid.ncelm+1, dtype='float64')
    u1[0:d] = 0.0
    u1[d:] = 0.0
    u2 = np.zeros(grid.ncelm+1, dtype='float64')
    u2[0:d] = 2.5
    u2[0:d] = 0.25
    svr.set_so0(0, u0)
    svr.set_so0(1, u1)
    svr.set_so0(2, u2)
    svr.set_so1(0, np.zeros(grid.ncelm+1, dtype='float64'))
    svr.set_so1(1, np.zeros(grid.ncelm+1, dtype='float64'))
    svr.set_so1(2, np.zeros(grid.ncelm+1, dtype='float64'))

    win = ApplicationWindow()
    win.show()
    win.activateWindow()

    svr.setup_march()

    if animate:
        win.set_solver(svr, interval)
    else:
        win.ax.plot(svr.xctr() / np.pi, svr.get_so0(0), '-')
        svr.march_alpha2(50)
        win.ax.plot(svr.xctr() / np.pi, svr.get_so0(0), '+')

    return win, svr

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
