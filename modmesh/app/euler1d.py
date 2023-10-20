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
import matplotlib.pyplot
from matplotlib.backends.backend_qtagg import FigureCanvas
from matplotlib.ticker import FormatStrFormatter
from matplotlib.figure import Figure
from PySide6.QtWidgets import (QApplication, QWidget, QScrollArea,
                               QGridLayout, QVBoxLayout, QPushButton,
                               QHBoxLayout, QFileDialog, QLayout, QCheckBox)
from PySide6.QtCore import Qt, QEvent, QMimeData, QTimer
from PySide6.QtGui import QDrag, QPixmap

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
    axis: matplotlib.pyplot.axis = None
    name: str = ""
    unit: str = ""

    def bind_axis(self, axis):
        self.axis = axis

    def update(self, adata, ndata):
        self.ana.set_ydata(adata.copy())
        self.num.set_ydata(ndata.copy())
        self.axis.relim()
        self.axis.autoscale_view()
        self.ana.figure.canvas.draw()
        self.num.figure.canvas.draw()


class PlotManager(QWidget):
    def __init__(self, shocktube):
        super().__init__()
        self.setWindowTitle("Euler 1D")
        # Set minimum window size to prevent the plot from looking
        # distorted due to the window being too small
        self.setMinimumSize(1150, 750)
        self.setAcceptDrops(True)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.main_container = QWidget()
        self.figure_container = None
        self.use_grid_layout = True
        self.main_layout = None
        self.shocktube = shocktube

        self.scroll_area.setWidget(self.main_container)
        self.layout.addWidget(self.scroll_area)

        self.button_layout = QHBoxLayout()

        self.save_button = QPushButton()
        self.save_button.setText("Save")

        self.swtich_button = QPushButton()
        self.swtich_button.setText("Layout swtich")

        self.button_layout.addWidget(self.save_button)
        self.button_layout.addWidget(self.swtich_button)

        self.save_button.clicked.connect(self.save_all)
        self.swtich_button.clicked.connect(self.switch_layout)

        self.layout.addLayout(self.button_layout)

        self.data_lines = {}

        self.density = QuantityLine(name="density",
                                    unit=r"$\mathrm{kg}/\mathrm{m}^3$")
        self.data_lines[self.density.name] = [self.density, True]
        self.velocity = QuantityLine(name="velocity",
                                     unit=r"$\mathrm{m}/\mathrm{s}$")
        self.data_lines[self.velocity.name] = [self.velocity, True]
        self.pressure = QuantityLine(name="pressure", unit=r"$\mathrm{Pa}$")
        self.data_lines[self.pressure.name] = [self.pressure, True]
        self.temperature = QuantityLine(name="temperature",
                                        unit=r"$\mathrm{K}$")
        self.data_lines[self.temperature.name] = [self.temperature, False]
        self.internal_energy = QuantityLine(name="internal_energy",
                                            unit=r"$\mathrm{J}/\mathrm{kg}$")
        self.data_lines[self.internal_energy.name] = [self.internal_energy,
                                                      False]
        self.entropy = QuantityLine(name="entropy",
                                    unit=r"$\mathrm{J}/\mathrm{K}$")
        self.data_lines[self.entropy.name] = [self.entropy, False]

        self.checkbox_select_num = 3

    def save_all(self):
        fig = QPixmap(self.figure_container.size())
        self.figure_container.render(fig)

        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getSaveFileName(None, "Save file", "",
                                                  "All Files (*)",
                                                  options=options)

        if fileName != "":
            fig.save(fileName, "JPG", 100)

    def _update_layout(self):
        while self.main_layout is not None and self.main_layout.count():
            item = self.main_layout.takeAt(0)
            if isinstance(item, QLayout):
                self.delete_layout(item)
            else:
                item.widget().setParent(None)
                item.widget().deleteLater()

        self.main_layout.deleteLater()
        self.main_container.update()

        # Qt objects are managed by Qt internal event loop,
        # hence it will not be deleted immediately when we called
        # deleteLater() and a QWidget only accept one layout at
        # the same time, therefore a object delete callback is
        # needed to set new layout after old layout deleted.
        def del_cb():
            if self.use_grid_layout:
                self.build_lines_grid_layout()
            else:
                self.build_lines_single_plot()

        self.main_layout.destroyed.connect(del_cb)

        self.main_layout.update()
        self.main_container.update()

    def switch_layout(self):
        self.use_grid_layout = not self.use_grid_layout
        self._update_layout()

    def delete_layout(self, layout):
        """Delete all widgets from a layout."""
        for i in reversed(range(layout.count())):
            widget = layout.itemAt(i).widget()
            # Check if the widget is a layout
            if isinstance(widget, QLayout):
                self.delete_layout(widget)
            else:
                widget.setParent(None)
                widget.deleteLater()

        layout.update()
        layout.setParent(None)
        layout.deleteLater()

    def eventFilter(self, obj, event):
        if self.use_grid_layout:
            if event.type() == QEvent.MouseButtonPress:
                self.mousePressEvent(event)
            elif event.type() == QEvent.MouseMove:
                self.mouseMoveEvent(event)

        return super().eventFilter(obj, event)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.use_grid_layout:
            coord = event.windowPos().toPoint()
            self.targetIndex = self.getWindowIndex(coord)
        else:
            self.targetIndex = None

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton and self.targetIndex is not None:
            windowItem = self.figure_layout.itemAt(self.targetIndex)

            drag = QDrag(windowItem)

            pix = windowItem.itemAt(0).widget().grab()

            mimeData = QMimeData()
            mimeData.setImageData(pix)

            drag.setMimeData(mimeData)
            drag.setPixmap(pix)
            drag.setHotSpot(event.pos())
            drag.exec()

    def dragEnterEvent(self, event):
        if event.mimeData().hasImage():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        if not event.source().geometry().contains(event.pos()):
            targetWindowIndex = self.getWindowIndex(event.pos())
            if targetWindowIndex is None:
                return

            i, j = (max(self.targetIndex, targetWindowIndex),
                    min(self.targetIndex, targetWindowIndex))

            p1, p2 = (self.figure_layout.getItemPosition(i),
                      self.figure_layout.getItemPosition(j))

            self.figure_layout.addItem(self.figure_layout.takeAt(i), *p2)
            self.figure_layout.addItem(self.figure_layout.takeAt(j), *p1)

    def getWindowIndex(self, pos):
        for i in range(self.figure_layout.count()):
            if self.figure_layout.itemAt(i).geometry().contains(pos):
                return i

    def update_lines(self):
        if self.use_grid_layout:
            self.density.update(adata=self.shocktube.density_field,
                                ndata=self.shocktube.svr.density[::2])
            self.pressure.update(adata=self.shocktube.pressure_field,
                                 ndata=self.shocktube.svr.pressure[::2])
            self.velocity.update(adata=self.shocktube.velocity_field,
                                 ndata=self.shocktube.svr.velocity[::2])
            self.temperature.update(adata=self.shocktube.temperature_field,
                                    ndata=self.shocktube.svr.temperature[::2])
            self.internal_energy.update(adata=(self.shocktube.
                                               internal_energy_field),
                                        ndata=(self.shocktube.svr.
                                               internal_energy[::2]))
            self.entropy.update(adata=self.shocktube.entropy_field,
                                ndata=self.shocktube.svr.entropy[::2])
        else:
            for name, data_line in self.data_lines.items():
                if data_line[1]:
                    eval(f'(data_line[0].update(adata=self.shocktube.'
                         f'{name}_field, ndata=self.shocktube.svr.'
                         f'{name}[::2]))')

    def build_lines_grid_layout(self):
        self.main_layout = QHBoxLayout(self.main_container)
        self.main_container.setLayout(self.main_layout)

        self.figure_container = QWidget()
        self.figure_layout = QGridLayout()
        self.figure_container.setLayout(self.figure_layout)

        self.main_layout.addWidget(self.figure_container)
        self._update_grid_figure()

    def build_lines_single_plot(self):
        checkbox_layout = QVBoxLayout()
        self._build_checkbox(checkbox_layout)

        self.main_layout = QHBoxLayout(self.main_container)
        self.main_layout.addLayout(checkbox_layout, 1)
        self.main_container.setLayout(self.main_layout)

        self.figure_container = QWidget()
        self.figure_layout = QVBoxLayout()
        self.figure_container.setLayout(self.figure_layout)

        self.main_layout.addWidget(self.figure_container, 4)

        # Legend figure
        legend_fig = Figure()
        legend_canvas = FigureCanvas(legend_fig)
        checkbox_layout.addWidget(legend_canvas)

        lines = self._update_single_figure()

        legend_canvas.figure.legend(lines,
                                    [line.get_label() for line in lines],
                                    loc='center',
                                    frameon=False)

    def _update_grid_figure(self):
        x = self.shocktube.svr.coord[::2]

        for i, (data, color) in enumerate((
                (self.density, 'r'),
                (self.velocity, 'g'),
                (self.pressure, 'b'),
                (self.temperature, 'c'),
                (self.internal_energy, 'k'),
                (self.entropy, 'm')
        )):
            figure = Figure()
            canvas = FigureCanvas(figure)

            axis = figure.add_subplot()
            axis.autoscale(enable=True, axis='y', tight=False)
            data.bind_axis(axis)
            data.ana, = axis.plot(x.copy(), np.zeros_like(x),
                                  f'{color}-',
                                  label=f'{data.name}_ana')
            data.num, = axis.plot(x.copy(), np.zeros_like(x),
                                  f'{color}x',
                                  label=f'{data.name}_num')
            axis.set_ylabel(f'{data.name} ({data.unit})')

            axis.set_xlabel("distance (m)")
            axis.legend()
            axis.grid()
            canvas.installEventFilter(self)

            box = QVBoxLayout()
            box.addWidget(canvas)

            self.figure_layout.addLayout(box, i // 2, i % 2)
            self.figure_layout.setColumnStretch(i % 2, 1)
            self.figure_layout.setRowStretch(i // 2, 1)

        self.update_lines()

    def _update_single_figure(self):
        x = self.shocktube.svr.coord[::2]
        fig = Figure()
        fig.tight_layout()
        canvas = FigureCanvas(fig)
        ax = canvas.figure.subplots()
        ax.autoscale(enable=True, axis='y', tight=False)

        lines = []

        # Matplotlib need to plot y axis on the left hand side first
        # then the reset of axis can be plotted on right hand side
        main_axis_plotted = False

        # Record how many lines had been selected to plot
        select_num = 0

        for data, color in (
            (self.density, 'r'),
            (self.velocity, 'g'),
            (self.pressure, 'b'),
            (self.temperature, 'c'),
            (self.internal_energy, 'k'),
            (self.entropy, 'm')
        ):
            # Plot multiple data line with same X axis, it need to plot a
            # data line on main axis first
            if self.data_lines[data.name][1]:
                if not main_axis_plotted:
                    data.bind_axis(ax)
                    data.ana, = ax.plot(x.copy(), np.zeros_like(x),
                                        f'{color}-',
                                        label=f'{data.name}_ana')
                    data.num, = ax.plot(x.copy(), np.zeros_like(x),
                                        f'{color}x',
                                        label=f'{data.name}_num')
                    ax.set_ylabel(f'{data.name} ({data.unit})')
                    main_axis_plotted = True
                else:
                    ax_new = ax.twinx()
                    data.bind_axis(ax_new)
                    data.ana, = ax_new.plot(x.copy(), np.zeros_like(x),
                                            f'{color}-',
                                            label=f'{data.name}_ana')
                    data.num, = ax_new.plot(x.copy(), np.zeros_like(x),
                                            f'{color}x',
                                            label=f'{data.name}_num')
                    ax_new.spines.right.set_position(("axes",
                                                      (1 + (select_num - 1)
                                                       * 0.2)))
                    ax_new.set_ylabel(f'{data.name} ({data.unit})')
                    ax_new.yaxis.set_major_formatter((FormatStrFormatter
                                                      ('%.2f')))
                select_num += 1
            lines.append(data.ana)
            lines.append(data.num)

        ax.set_xlabel("distance (m)")
        ax.grid()
        canvas.installEventFilter(self)

        # These parameters are the results obtained by my tuning on GUI
        fig.subplots_adjust(left=0.1,
                            right=0.97 - (self.checkbox_select_num - 1) * 0.1,
                            bottom=0.093, top=0.976,
                            wspace=0.2, hspace=0.2)

        prev_fig = self.figure_layout.takeAt(0)

        if prev_fig:
            prev_fig.widget().deleteLater()

        self.update_lines()
        self.figure_layout.addWidget(canvas)

        return lines

    def _build_checkbox(self, layout):
        layout.setSpacing(50)
        for name, data_line in self.data_lines.items():
            check_box = QCheckBox(name)
            if data_line[1]:
                check_box.toggle()
            check_box.clicked.connect(self._checkbox_cb)
            layout.addWidget(check_box)

    def _checkbox_cb(self):
        # Under a single plot layout, that allow 3 lines on the same chart
        # simultaneously to avoid it looking too crowded.
        # I have chosen to use checkboxes for user to select
        # the parameters they want to plot on the chart.
        checkbox = self.sender()
        if self.checkbox_select_num == 3:
            if checkbox.isChecked():
                checkbox.toggle()
            else:
                self.checkbox_select_num -= 1
                self.data_lines[checkbox.text()][1] = False
        elif self.checkbox_select_num == 1:
            if not checkbox.isChecked():
                checkbox.toggle()
            else:
                self.checkbox_select_num += 1
                self.data_lines[checkbox.text()][1] = True
        else:
            if checkbox.isChecked():
                self.checkbox_select_num += 1
                self.data_lines[checkbox.text()][1] = True
            else:
                self.checkbox_select_num -= 1
                self.data_lines[checkbox.text()][1] = False

        self._update_single_figure()


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

        self.use_sub = mm.Toggle.instance.get_value('apps.euler1d.use_sub',
                                                    False)
        if self.use_sub is None:
            self.use_sub = mm.Toggle.instance.solid.use_pyside
        self._main = QWidget()

        if self.use_sub:
            # FIXME: sub window causes missing QWindow with the following
            # error:
            # RuntimeError:
            # Internal C++ object (PySide6.QtGui.QWindow) already deleted.
            # It is probably because RMainWindow is not recognized by PySide6
            # and matplotlib.  We may consider to use composite for QMainWindow
            # instead of inheritance.
            self._subwin = view.mgr.addSubWindow(self._main)
            self._subwin.resize(1150, 750)

        self.plt_mgr = PlotManager(self.shocktube)
        # Ideally one would use self.addToolBar here, but it is slightly
        # incompatible between PyQt6 and other bindings, so we just add the
        # toolbar as a plain widget instead.
        layout = QVBoxLayout(self._main)
        layout.addWidget(self.plt_mgr)

        self.profiling = profiling

    def show(self):
        self._main.show()
        if self.use_sub:
            self._subwin.show()

    def start(self):
        self.timer.start(self.interval)

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
        self.plt_mgr.update_lines()

    def setup_solver(self, interval):
        """
        :param interval: milliseconds
        :return: nothing
        """
        self.shocktube.build_field(t=0)
        self.plt_mgr.shocktube = self.shocktube
        self.plt_mgr.build_lines_grid_layout()
        self.interval = interval
        self.timer = QTimer()
        self.timer.timeout.connect(self.step)

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
