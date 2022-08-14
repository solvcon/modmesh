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
from .. import onedim


def load_app():
    view.app.pytext.code = """
# Need to hold the win object to keep PySide alive.
win, svr = mm.app.euler1d.run(animate=True, interval=10, max_steps=50)
print(mm.apputil.environ)
#win.march_alpha2()
win.start()
#win.stop()
#mm.apputil.stop_code(appenvobj)
#mm.apputil.environ['master'].locals['win'].timer.stop()
""".lstrip()


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

        self.step = 0
        self.max_steps = 0

        self.svr = None
        self.ax = self.canvas.figure.subplots()
        self.timer = None
        self.line = None

    def start(self):
        self.timer.start()

    def stop(self):
        self.timer.stop()

    def march_alpha2(self, steps=1):
        self.svr.march_alpha2(steps=steps)
        cfl = self.svr.cfl
        print("CFL:", "min", cfl.min(), "max", cfl.max())
        self._update_canvas()
        self.step += steps
        if self.max_steps and self.step > self.max_steps:
            self.stop()

    def set_solver(self, svr, interval):
        """
        :param svr: Solver
        :param interval: milliseconds
        :return: nothing
        """
        self.svr = svr
        x = svr.coord[::2]
        self.line0, = self.ax.plot(x, svr.density[::2], 'r+')
        self.line1, = self.ax.plot(x, svr.velocity[::2], 'g+')
        self.line2, = self.ax.plot(x, svr.pressure[::2], 'b+')
        self.timer = self.canvas.new_timer(interval)
        self.timer.add_callback(self.march_alpha2)

    def _update_canvas(self):
        x = self.svr.coord[::2]
        self.line0.set_data(x, self.svr.density[::2])
        self.line1.set_data(x, self.svr.velocity[::2])
        self.line2.set_data(x, self.svr.pressure[::2])
        self.line0.figure.canvas.draw()
        self.line1.figure.canvas.draw()
        self.line2.figure.canvas.draw()


def run(animate, interval=10, max_steps=50):
    dt = 0.05
    dx = 0.1
    ncoord = 201
    svr = onedim.Euler1DSolver(ncoord=ncoord, time_increment=dt)

    garr = np.linspace(-dx * (ncoord // 2), dx * (ncoord // 2), num=ncoord)
    svr.coord[...] = garr

    # Initialize.
    svr.cfl.fill(0)
    svr.gamma.fill(1.4)
    d = ncoord // 2
    # u0
    svr.so0[0:d, 0] = 1.0
    svr.so0[d:, 0] = 0.125
    # u1
    svr.so0[0:d, 1] = 0.0
    svr.so0[d:, 1] = 0.0
    # u2
    svr.so0[0:d, 2] = 2.5
    svr.so0[d:, 2] = 0.25
    # Derivative.
    svr.so1.fill(0)

    win = ApplicationWindow()
    win.max_steps = max_steps
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
