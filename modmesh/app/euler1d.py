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


# This is a hack to make this file runs directly from command line.
__package__ = "modmesh.app"

import sys
from dataclasses import dataclass

import numpy as np

import matplotlib
from matplotlib.backends.qt_compat import QtWidgets
from matplotlib.backends.backend_qtagg import (
    FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
from matplotlib.figure import Figure

import modmesh as mm
from .. import view
from ..onedim import euler1d


def load_app():
    view.mgr.pycon.writeToHistory("""
# Use the functions for more examples:
ctrl.start()  # Start the movie
ctrl.step()  # Stepping the solution

# Note: you can enable profiling info by "profiling" option
ctrl = mm.app.euler1d.run(interval=10, max_steps=50, profiling=True)
""")
    cmd = "ctrl = mm.app.euler1d.run(interval=10, max_steps=50)"
    view.mgr.pycon.command = cmd


@dataclass
class QuantityLine:
    ana: matplotlib.lines.Line2D = None
    num: matplotlib.lines.Line2D = None
    name: str = ""
    unit: str = ""

    def update(self, adata, ndata):
        self.ana.set_ydata(adata.copy())
        self.num.set_ydata(ndata.copy())
        self.ana.figure.canvas.draw()
        self.num.figure.canvas.draw()


class Plot:
    def __init__(self, figsize=(15, 10)):
        self.fig = Figure(figsize=figsize)
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.canvas.figure.subplots()
        self.ax.set_ylim(-0.2, 1.2)
        self.ax.grid()

        self.density = QuantityLine(name="density",
                                    unit=r"$\mathrm{kg}/\mathrm{m}^3$")
        self.velocity = QuantityLine(name="velocity",
                                     unit=r"$\mathrm{m}/\mathrm{s}$")
        self.pressure = QuantityLine(name="pressure", unit=r"$\mathrm{Pa}$")

    def build_lines(self, x):
        self.ax.set_xlim(x[0], x[-1])
        lines = []
        for i, (data, color) in enumerate((
                (self.density, 'r'),
                (self.velocity, 'g'),
                (self.pressure, 'b'),
        )):
            # Plot multiple data line with same X axis, it need to plot a
            # data line on main axis first
            if i == 0:
                data.ana, = self.ax.plot(x.copy(), np.zeros_like(x),
                                         f'{color}-',
                                         label=f'{data.name}_ana')
                data.num, = self.ax.plot(x.copy(), np.zeros_like(x),
                                         f'{color}x',
                                         label=f'{data.name}_num')
                self.ax.set_ylabel(f'{data.name} ({data.unit})')
            else:
                ax_new = self.ax.twinx()
                data.ana, = ax_new.plot(x.copy(), np.zeros_like(x),
                                        f'{color}-',
                                        label=f'{data.name}_ana')
                data.num, = ax_new.plot(x.copy(), np.zeros_like(x),
                                        f'{color}x',
                                        label=f'{data.name}_num')
                ax_new.spines.right.set_position(("axes", 1 + (i - 1) * 0.07))
                ax_new.set_ylim(self.ax.get_ylim())
                ax_new.set_ylabel(f'{data.name} ({data.unit})')

            lines.append(data.ana)
            lines.append(data.num)

        self.ax.legend(lines, [line.get_label() for line in lines])
        self.ax.set_xlabel("distance (m)")

    def update_lines(self, shocktube):
        self.density.update(adata=shocktube.density_field,
                            ndata=shocktube.svr.density[::2])
        self.pressure.update(adata=shocktube.pressure_field,
                             ndata=shocktube.svr.pressure[::2])
        self.velocity.update(adata=shocktube.velocity_field,
                             ndata=shocktube.svr.velocity[::2])


class Controller:
    def __init__(self, shocktube, max_steps, use_sub=None, profiling=False):
        if None is shocktube.gamma:
            raise ValueError("shocktube does not have constant built")
        if None is shocktube.svr:
            raise ValueError("shocktube does not have numerical solver built")

        super().__init__()

        self.shocktube = shocktube

        self.max_steps = max_steps
        self.current_step = 0
        self.timer = None

        self.use_sub = use_sub
        if self.use_sub is None:
            self.use_sub = mm.Toggle.solid.use_pyside
        self._main = QtWidgets.QWidget()
        if self.use_sub:
            # FIXME: sub window causes missing QWindow with the following
            # error:
            # RuntimeError:
            # Internal C++ object (PySide6.QtGui.QWindow) already deleted.
            # It is probably because RMainWindow is not recognized by PySide6
            # and matplotlib.  We may consider to use composite for QMainWindow
            # instead of inheritance.
            self._subwin = view.mgr.addSubWindow(self._main)
            self._subwin.resize(400, 300)

        self.plt = Plot(figsize=(15, 10))
        # Ideally one would use self.addToolBar here, but it is slightly
        # incompatible between PyQt6 and other bindings, so we just add the
        # toolbar as a plain widget instead.
        layout = QtWidgets.QVBoxLayout(self._main)
        layout.addWidget(self.plt.canvas)
        layout.addWidget(NavigationToolbar(self.plt.canvas, self._main))

        self.profiling = profiling

    def show(self):
        self._main.show()
        if self.use_sub:
            self._subwin.show()

    def start(self):
        self.timer.start()

    def stop(self):
        self.timer.stop()

    def step(self, steps=1):
        self.march_alpha2(steps=steps)
        if self.max_steps and self.current_step > self.max_steps:
            self.stop()

    def march_alpha2(self, steps=1):
        self.shocktube.svr.march_alpha2(steps=steps)
        self.current_step += steps
        time_current = self.current_step * self.shocktube.svr.time_increment
        self.shocktube.build_field(t=time_current)
        cfl = self.shocktube.svr.cfl
        self.log(f"CFL: min {cfl.min()} max {cfl.max()}")
        if self.profiling:
            self.log(mm.time_registry.report())
        self.plt.update_lines(self.shocktube)

    def setup_solver(self, interval):
        """
        :param interval: milliseconds
        :return: nothing
        """
        self.shocktube.build_field(t=0)
        svr = self.shocktube.svr
        self.plt.build_lines(svr.coord[::2])
        self.timer = self.plt.canvas.new_timer(interval)
        self.timer.add_callback(self.step)

    @staticmethod
    def log(msg):
        sys.stdout.write(msg)
        sys.stdout.write('\n')
        view.mgr.pycon.writeToHistory(msg)
        view.mgr.pycon.writeToHistory('\n')


class ControllerNoViewMgr(Controller):
    def __init__(self, shocktube, max_steps, use_sub=None, profiling=False):
        super().__init__(shocktube,
                         max_steps,
                         use_sub=None,
                         profiling=profiling)

    @staticmethod
    def log(msg):
        sys.stdout.write(msg)
        sys.stdout.write("\n")


def run(interval=10, max_steps=50, no_view_mgr=False, **kw):
    st = euler1d.ShockTube()
    st.build_constant(gamma=1.4, pressure1=1.0, density1=1.0, pressure5=0.1,
                      density5=0.125)
    st.build_numerical(xmin=-10, xmax=10, ncoord=201, time_increment=0.05)

    if no_view_mgr:
        ctrl = ControllerNoViewMgr(shocktube=st, max_steps=max_steps, **kw)
    else:
        ctrl = Controller(shocktube=st, max_steps=max_steps, **kw)
    ctrl.setup_solver(interval)
    ctrl.show()

    return ctrl


if __name__ == "__main__":
    try:
        from PySide6.QtCore import QTimer
        from PySide6.QtWidgets import QApplication

        app = QApplication()

        ctrl = run(interval=10, max_steps=50, no_view_mgr=True, profiling=True)
        ctrl.start()

        # The trick to close the event loop of app automatically
        # The timer will emit a closeAllWindows event after 20 seconds
        # after the app is executed.
        QTimer.singleShot(20000, app.closeAllWindows)

        sys.exit(app.exec())

    except ImportError:
        print("Something wrong when importing PySide6.")
        print("Do you install PySide6?")
        sys.exit(1)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
