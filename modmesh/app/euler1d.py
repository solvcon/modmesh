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
from ..onedim import euler1d


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
    def __init__(self, shocktube, max_steps):
        if None is shocktube.gamma:
            raise ValueError("shocktube does not have constant built")
        if None is shocktube.svr:
            raise ValueError("shocktube does not have numerical solver built")

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

        self.shocktube = shocktube
        self.ax = self.canvas.figure.subplots()
        self.timer = None
        self.density_num = None
        self.velocity_num = None
        self.pressure_num = None
        self.density_ana = None
        self.velocity_ana = None
        self.pressure_ana = None

    def start(self):
        self.timer.start()

    def stop(self):
        self.timer.stop()

    def march_alpha2(self, steps=1):
        self.shocktube.svr.march_alpha2(steps=steps)
        self.step += steps
        time_current = self.step * self.shocktube.svr.time_increment
        self.shocktube.build_field(t=time_current)
        cfl = self.shocktube.svr.cfl
        print("CFL:", "min", cfl.min(), "max", cfl.max())
        self._update_canvas()
        if self.max_steps and self.step > self.max_steps:
            self.stop()

    def setup_solver(self, interval):
        """
        :param interval: milliseconds
        :return: nothing
        """
        st = self.shocktube
        st.build_field(t=0)
        svr = self.shocktube.svr
        x = svr.coord[::2]
        self.ax.grid()
        self.density_ana, = self.ax.plot(x, st.density_field, 'r-')
        self.velocity_ana, = self.ax.plot(x, st.velocity_field, 'g-')
        self.pressure_ana, = self.ax.plot(x, st.pressure_field, 'b-')
        self.density_num, = self.ax.plot(x, svr.density[::2], 'rx')
        self.velocity_num, = self.ax.plot(x, svr.velocity[::2], 'gx')
        self.pressure_num, = self.ax.plot(x, svr.pressure[::2], 'bx')
        self.timer = self.canvas.new_timer(interval)
        self.timer.add_callback(self.march_alpha2)

    def _update_canvas(self):
        x = self.shocktube.svr.coord[::2]
        self.density_ana.set_data(x, self.shocktube.density_field)
        self.velocity_ana.set_data(x, self.shocktube.velocity_field)
        self.pressure_ana.set_data(x, self.shocktube.pressure_field)
        self.density_num.set_data(x, self.shocktube.svr.density[::2])
        self.velocity_num.set_data(x, self.shocktube.svr.velocity[::2])
        self.pressure_num.set_data(x, self.shocktube.svr.pressure[::2])
        self.density_ana.figure.canvas.draw()
        self.velocity_ana.figure.canvas.draw()
        self.pressure_ana.figure.canvas.draw()
        self.density_num.figure.canvas.draw()
        self.velocity_num.figure.canvas.draw()
        self.pressure_num.figure.canvas.draw()


def run(animate, interval=10, max_steps=50):
    st = euler1d.ShockTube()
    st.build_constant(gamma=1.4, pressure1=1.0, density1=1.0, pressure5=0.1,
                      density5=0.125)
    st.build_numerical(xmin=-10, xmax=10, ncoord=201, time_increment=0.05)

    win = ApplicationWindow(shocktube=st, max_steps=max_steps)
    win.show()
    win.activateWindow()

    if animate:
        win.setup_solver(interval)
    else:
        win.ax.plot(st.svr.xctr() / np.pi, st.svr.get_so0(0), '-')
        st.svr.march_alpha2(50)
        win.ax.plot(st.svr.xctr() / np.pi, st.svr.get_so0(0), '+')

    return win

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
