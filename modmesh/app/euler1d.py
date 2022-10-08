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
win = mm.app.euler1d.run(animate=True, interval=10, max_steps=50)
print(mm.apputil.environ)
#win.march_alpha2()
win.start()
#win.stop()
#mm.apputil.stop_code(appenvobj)
#mm.apputil.environ['master'].locals['win'].timer.stop()
""".lstrip()


class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self, svr, max_steps):
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

        self.max_steps = max_steps
        self.step = 0

        self.svr = svr
        self.ax = self.canvas.figure.subplots()
        self.timer = None
        self.line_density = None
        self.line_velocity = None
        self.line_pressure = None

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

    def setup_solver(self, interval):
        """
        :param interval: milliseconds
        :return: nothing
        """
        svr = self.svr
        x = svr.coord[::2]
        self.line_density, = self.ax.plot(x, svr.density[::2], 'r+')
        self.line_velocity, = self.ax.plot(x, svr.velocity[::2], 'g+')
        self.line_pressure, = self.ax.plot(x, svr.pressure[::2], 'b+')
        self.timer = self.canvas.new_timer(interval)
        self.timer.add_callback(self.march_alpha2)

    def _update_canvas(self):
        x = self.svr.coord[::2]
        self.line_density.set_data(x, self.svr.density[::2])
        self.line_velocity.set_data(x, self.svr.velocity[::2])
        self.line_pressure.set_data(x, self.svr.pressure[::2])
        self.line_density.figure.canvas.draw()
        self.line_velocity.figure.canvas.draw()
        self.line_pressure.figure.canvas.draw()


def run(animate, interval=10, max_steps=50):
    svr = onedim.Euler1DSolver(
        xmin=-10, xmax=10, ncoord=201, time_increment=0.05)
    svr.init_sods_problem(
        density0=1.0, pressure0=1.0, density1=0.125, pressure1=0.1,
        xdiaphragm=0.0, gamma=1.4)

    win = ApplicationWindow(svr=svr, max_steps=max_steps)
    win.show()
    win.activateWindow()

    if animate:
        win.setup_solver(interval)
    else:
        win.ax.plot(svr.xctr() / np.pi, svr.get_so0(0), '-')
        svr.march_alpha2(50)
        win.ax.plot(svr.xctr() / np.pi, svr.get_so0(0), '+')

    return win

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
