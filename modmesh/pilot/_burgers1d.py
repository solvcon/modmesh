import sys
from dataclasses import dataclass

import numpy as np

import matplotlib
import matplotlib.pyplot
from matplotlib.backends.backend_qtagg import FigureCanvas
from matplotlib.figure import Figure
from matplotlib.pyplot import setp

from PySide6.QtCore import QTimer, Slot, Qt
from PySide6.QtWidgets import QDockWidget

from PUI.state import State
from PUI.PySide6.window import Window

from .. import core as mcore
from ._gui_common import PilotFeature
from ._euler1d import SolverConfig, PlotConfig, PlotArea, ConfigWindow


@dataclass
class QuantityLine(object):
    ana: matplotlib.lines.Line2D = None
    axis: matplotlib.pyplot.axis = None
    y_upper_lim: float = 0.0
    y_bottom_lim: float = 0.0
    name: str = ""
    unit: str = ""

    def update(self, x, y):
        self.ana.set_xdata(x)
        self.ana.set_ydata(y)
        self.ana.figure.canvas.draw()


class BurgersEquation:

    R = 8.31446261815324

    def __init__(self):
        # Initial condition
        self.coord = None
        self.velocity = None

        # Field data of the analytical solution.
        self.coord_field = None
        self.velocity_field = None

    def build_constant(self, x1, x2, x3, velocity12, velocity23):
        self.coord = np.array([x1, x2, x3], dtype='float64')
        self.velocity = np.array([velocity12, velocity23], dtype='float64')

    def build_field(self, t):
        self.coord_field = self.calc_coord_field(t)
        _ = (self.velocity, self.velocity)
        self.velocity_field = np.transpose(_).ravel()

    def calc_coord_field(self, t):
        # calculate shock and expansion wave location
        left = self.coord[1:-1] + self.velocity[0:-1] * t
        right = self.coord[1:-1] + self.velocity[1:] * t
        shock = left > right   # shock wave
        expansion = left <= right   # expansion wave
        avg = (left + right) / 2
        left = shock * avg + expansion * left
        right = shock * avg + expansion * right
        internal = np.transpose((left, right)).ravel()
        return np.hstack((self.coord[0], internal, self.coord[-1]))


class Burgers1DApp(PilotFeature):

    def populate_menu(self):
        self._add_menu_item(
            menu=self._mgr.oneMenu,
            text="Burgers equation",
            tip="One-dimensional Burgers equation problem",
            func=self.run,
        )

    def run(self):
        self.setup_app()
        plotting_area = PlotArea(Window(), self)

        config_window = ConfigWindow(self)
        config_widget = QDockWidget("config")
        config_widget.setWidget(config_window)

        self._mgr.mainWindow.addDockWidget(Qt.LeftDockWidgetArea,
                                           config_widget)
        _subwin = self._mgr.addSubWindow(plotting_area.ui.ui)
        _subwin.showMaximized()

        plotting_area.redraw()
        _subwin.show()

    def setup_app(self):
        solver_config_data = [
            ["x1", -10.0, "The 1st point of x axis"],
            ["x2", 0.0, "The 2nd point of x axis"],
            ["x3", 5.0, "The 3rd point of x axis"],
            ["velocity12", -0.5, "The velocity between point 1 and 2"],
            ["velocity23", 1.0, "The velocity between point 2 and 3"],
            ["time_interval", 0.01, "The amount of time in a time step"],
            ["max_steps", 200, "Maximum time step"],
            ["profiling", False, "Turn on / off solver profiling"],
        ]
        self.solver_config = SolverConfig(solver_config_data)
        self.velocity = QuantityLine(name="velocity",
                                     unit=r"$\mathrm{m}/\mathrm{s}$",
                                     y_upper_lim=1.2,
                                     y_bottom_lim=-1.2)
        plot_config_data = []
        plot_config_data.append([self.velocity.name,
                                 True,
                                 self.velocity.y_upper_lim,
                                 self.velocity.y_bottom_lim])
        self.plot_config = PlotConfig(plot_config_data)
        self.use_grid_layout = False
        self.plot_holder = State()
        self.set_solver_config()
        self.setup_timer()
        self.plot_holder.plot = self.build_single_figure()

    def init_solver(self, x1, x2, x3, velocity12, velocity23):
        self.st = BurgersEquation()
        self.st.build_constant(x1, x2, x3, velocity12, velocity23)
        self.st.build_field(t=0)

    def set_solver_config(self):
        self.init_solver(x1=self.solver_config["x1"]["value"],
                         x2=self.solver_config["x2"]["value"],
                         x3=self.solver_config["x3"]["value"],
                         velocity12=self.solver_config["velocity12"]["value"],
                         velocity23=self.solver_config["velocity23"]["value"])
        self.current_step = 0
        self.time_interval = self.solver_config["time_interval"]["value"]
        self.max_steps = self.solver_config["max_steps"]["value"]
        self.profiling = self.solver_config["profiling"]["value"]

    def setup_timer(self):
        self.timer = QTimer()
        self.timer.timeout.connect(self.timer_timeout)

    def build_grid_figure(self):
        fig = Figure()
        canvas = FigureCanvas(fig)
        ax = canvas.figure.subplots(3, 2)
        fig.tight_layout()
        y_upper_lim_max = 0.0
        y_bottom_lim_min = sys.float_info.max

        data = self.velocity
        x = self.st.coord
        data.axis = ax[0][0]
        data.ana, = data.axis.plot(x,
                                   np.zeros_like(x),
                                   'g-',
                                   label=f'{data.name}_ana')
        data.axis.set_xlabel("location")
        data.axis.set_ylabel(f'{data.name}')
        data.axis.legend()
        data.axis.grid()
        y_upper_lim_max = max(y_upper_lim_max, data.y_upper_lim)
        y_bottom_lim_min = min(y_bottom_lim_min, data.y_bottom_lim)

        setp(ax, ylim=[y_bottom_lim_min, y_upper_lim_max])
        self.update_lines()
        return canvas

    def build_single_figure(self):
        fig = Figure()
        canvas = FigureCanvas(fig)
        ax = canvas.figure.subplots()
        fig.tight_layout()
        y_upper_lim_max = 0.0
        y_bottom_lim_min = sys.float_info.max

        x = self.st.coord
        data = self.velocity
        if self.plot_config[data.name]["line_selection"]:
            data.axis = ax
            data.ana, = ax.plot(x,
                                np.zeros_like(x),
                                'g-',
                                label=f'{data.name}_ana')
            y_upper_lim_max = max(y_upper_lim_max, data.y_upper_lim)
            y_bottom_lim_min = min(y_bottom_lim_min, data.y_bottom_lim)
        ax.set_xlabel("distance")
        ax.grid()
        ax.legend()

        setp(ax, ylim=[y_bottom_lim_min, y_upper_lim_max])
        self.update_lines()

        return canvas

    def step(self, steps=1):
        if self.max_steps and self.current_step > self.max_steps:
            self.stop()
            return

        self.current_step += steps
        self.st.build_field(t=self.current_step * self.time_interval)
        self.update_lines()
        if self.profiling:
            self.log(mcore.time_registry.report())

    def start(self):
        self.timer.start(self.time_interval)

    def set(self):
        self.set_solver_config()
        self.setup_timer()
        self.update_layout()

    def stop(self):
        self.timer.stop()

    def update_layout(self):
        line = self.velocity
        line.y_upper_lim = self.plot_config[line.name]["y_axis_upper_limit"]
        line.y_bottom_lim = self.plot_config[line.name]["y_axis_bottom_limit"]

        if self.use_grid_layout:
            self.plot_holder.plot = self.build_grid_figure()
        else:
            self.plot_holder.plot = self.build_single_figure()

    def single_layout(self):
        self.use_grid_layout = False
        self.plot_holder.plot = self.build_single_figure()

    def grid_layout(self):
        self.use_grid_layout = True
        self.plot_holder.plot = self.build_grid_figure()

    @Slot()
    def timer_timeout(self):
        self.step()

    def log(self, msg):
        if sys.stdout is not None:
            sys.stdout.write(msg)
            sys.stdout.write('\n')
        self._pycon.writeToHistory(msg)
        self._pycon.writeToHistory('\n')

    def update_lines(self):
        if self.use_grid_layout:
            self.velocity.update(self.st.coord_field, self.st.velocity_field)
        else:
            for name, is_selected, *_ in self.plot_config._tbl_content:
                if is_selected:
                    eval(f'(self.{name}.update(x=self.st.coord_field,'
                         f'y=self.st.{name}_field))')


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
