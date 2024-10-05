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
import matplotlib
import matplotlib.pyplot
import numpy as np
import modmesh as mm

from dataclasses import dataclass
from matplotlib.backends.backend_qtagg import FigureCanvas
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
from matplotlib.pyplot import setp
from PySide6.QtCore import QTimer, Slot, Qt
from PySide6.QtWidgets import QDockWidget
from PUI.state import State
from PUI.PySide6.base import PuiInQt, QtInPui
from PUI.PySide6.button import Button
from PUI.PySide6.layout import VBox, Spacer, HBox
from PUI.PySide6.scroll import Scroll
from PUI.PySide6.window import Window
from PUI.PySide6.combobox import ComboBox, ComboBoxItem
from PUI.PySide6.label import Label
from PUI.PySide6.table import Table
from PUI.PySide6.toolbar import ToolBar
from PUI.PySide6.modal import Modal
from ..onedim import euler1d
from .. import view


@dataclass
class QuantityLine:
    """
    A class representing a quantity line with associated data.

    This class is a data structure that holds information about a quantity line
    in a plot. It is designed to be used in conjunction with Matplotlib for
    visualizing analytical and numerical data.

    Class attributes:
        - `ana` (matplotlib.lines.Line2D): Line2D for analytical data.
        - `num` (matplotlib.lines.Line2D): Line2D for numerical data.
        - `axis` (matplotlib.pyplot.axis): Axis for the plot.
        - `name` (str): Name of the quantity.
        - `unit` (str): Unit of measurement.
        - `y_upper_lim` (float): y axis upper limit.
        - `y_bottom_lim` (float): y axis bottom limit.

    Methods:
        - :meth:`update(xdata, adata, ndata)`: Update the line data and
          redraw the plot.
    """
    ana: matplotlib.lines.Line2D = None
    num: matplotlib.lines.Line2D = None
    axis: matplotlib.pyplot.axis = None
    y_upper_lim: float = 0.0
    y_bottom_lim: float = 0.0
    name: str = ""
    unit: str = ""

    def update(self, adata, ndata):
        """
        Update the line data and redraw the plot.

        :param adata: Analytical data.
        :type adata: ndarray

        :param ndata: Numerical data.
        :type ndata: ndarray

        :return: None
        """
        self.ana.set_ydata(adata)
        self.num.set_ydata(ndata)
        self.ana.figure.canvas.draw()
        self.num.figure.canvas.draw()


class _Accessor(object):
    """
    Helper calss to access data within the configuration table using
    multiple dimensions, currenlty only support 2-dimensions table.

    This class allows users to access data by chaining multiple [],
    supporting string-based.

    Attributes:
        :ivar: _data: handle input data that allow user can using
            index or string to access it.
        :ivar: _header (list): list of data's column header name.
        :ivar: _dimIdx (int): using to indicate which dimension is
            currenlty being accessed by __getitem__.
    """
    def __init__(self, data, dimIdx=0, header=None):
        self._data = data
        self._header = header
        self._dimIdx = dimIdx

    def __getitem__(self, key):
        if self._dimIdx == 0:
            for row_idx, row in enumerate(self._data):
                if key == row[0]:
                    return _Accessor(self._data[row_idx],
                                     self._dimIdx+1,
                                     self._header)
        else:
            for idx, ele in enumerate(self._header):
                if key == ele:
                    return self._data[idx]
        raise KeyError(f'Invaild key: {key} not found in table')


class GUIConfig(object):
    """
    Configuration class for the GUI.

    This class provides a configuration interface for the GUI, allowing
    users to set and retrieve parameters related to the simulation.

    Attributes:
        :ivar: state (:class:`State`): The state object holding configuration
            data.
        :ivar: _tbl_content (:class:`State`): The content of the
            configuration table.
        :ivar: _col_header (list): The header for the configuration
            table columns.

    Methods:
        :meth:`data(row, col)`: Get the value at a specific row and column
            in the configuration table.
        :meth:`setData(row, col, value)`: Set the value at a specific row
            and column in the configuration table.
        :meth:`columnHeader(col)`: Get the header for a specific column
            in the configuration table.
        :meth:`editable(row, col)`: Check if a cell in the configuration
            table is editable.
        :meth:`rowCount()`: Get the number of rows in the configuration
            table.
        :meth:`columnCount()`: Get the number of columns in the
            configuration table.
    """

    def __init__(self, input_data, col_headers):
        self.state = State(input_data)
        self._tbl_content = self.state
        self._col_header = col_headers

    def __getitem__(self, key):
        return _Accessor(self._tbl_content, 0, self._col_header)[key]

    def data(self, row, col):
        """
        Get the value at a specific row and column in the configuration table.

        :param row: Row index.
        :type row: int
        :param col: Column index.
        :type col: int
        :return: The value at the specified location in
        the configuration table.
        """
        return self._tbl_content[row][col]

    def setData(self, row, col, value):
        """
        Set the value at a specific row and column in the configuration table.

        :param row: Row index.
        :type row: int
        :param col: Column index.
        :type col: int
        :prarm value: Any
        :return None
        """
        self._tbl_content[row][col] = value

    def columnHeader(self, col):
        """
        Get the specific column header in the configuration table.

        :param col: Column index.
        :type col: int
        :return: The header for the specific column.
        """
        return self._col_header[col]

    def editable(self, row, col):
        """
        Check if the cell is editable.

        :param row: Row index.
        :type row: int
        :param col: Column index.
        :type col: int
        :return: True if the cell is editable, false otherwise.
        """
        if col == 1:
            return True
        return False

    # Delete row header
    rowHeader = None

    def rowCount(self):
        """
        Get the number of rows in the configuration table.

        :return: The number of rows.
        """
        return len(self._tbl_content)

    def columnCount(self):
        """
        Get the number of columns in the configuration table.

        :return: The number of columns.
        """
        return len(self._tbl_content[0])


class SolverConfig(GUIConfig):
    """
    Configuration class for the solver.

    This class provides a configuration interface for the solver, allowing
    users to set and retrieve parameters related to the simulation.

    Attributes:
        - `state` (:class:`State`): The state object holding configuration
          data in the form of [variable_name, value, description].
        - `_tbl_content` (:class:`State`): The content of the
          configuration table.
        - `_col_header` (list): The header for the configuration
          table columns.
    """
    def __init__(self, input_data):
        super().__init__(input_data, ["variable", "value", "description"])


class PlotConfig(GUIConfig):
    """
    Configuration class for the plot.

    This class provides a configuration interface for the plot, allowing
    users to set and retrieve parameters related to the plotting arae.

    Attributes:
        - `state` (:class:`State`): The state object holding configuration
          data in the form of [variable, line_selection, y_axis_upper_limit,
          y_axis_bottom_limit].
        - `_tbl_content` (:class:`State`): The content of the
          configuration table.
        - `_col_header` (list): The header for the configuration
          table columns.
    """
    def __init__(self, input_data):
        super().__init__(input_data, ["variable",
                                      "line_selection",
                                      "y_axis_upper_limit",
                                      "y_axis_bottom_limit"])

    def editable(self, row, col):
        """
        Check if the cell is editable.

        :param row: Row index.
        :type row: int
        :param col: Column index.
        :type col: int
        :return: True if the cell is editable, false otherwise.
        """
        if col >= 1:
            return True
        return False


class Euler1DApp():
    """
    Main application class for the Euler 1D solver.

    This class provides the main application logic for the Euler 1D solver.
    It includes methods for initializing the solver, configuring parameters,
    setting up timers, and building visualization figures.

    Attributes:
        - `solver_config` (:class:`SolverConfig`): Configuration object
          for the solver.
        - `solver_config_data` (list): Solver configuration data
        - `plot_config` (:class:`PlotConfig`): Configuration object
          for the plotting area.
        - `plot_config_data` (list): Plotting area configuration data
        - `data_lines` (dict): Dictionary containing QuantityLine objects and
          their display status.
        - Other QuantityLine (:class:`QuantityLine`) objects such as `density`,
          `velocity`, etc. to save physical variables.
        - `use_grid_layout` (bool): Flag indicating whether to use a
          grid layout.
        - `plot_holder` (:class:`State`): State object for holding plots.

    Methods:
        - :meth:`init_solver(gamma, pressure_left, density_left, ...)`:
          Initialize the shock tube solver and set up the initial conditions.
        - :meth:`set_solver_config()`: Initialize solver configuration
          based on user input.
        - :meth:`setup_timer()`: Set up the Qt timer for data visualization.
        - :meth:`build_grid_figure()`: Build a grid figure for visualization.
        - :meth:`build_single_figure()`: Build a single-figure layout for
          visualization.
        - :meth:`march_alpha2(steps)`: Call the C++ solver to march the
          time step.
        - :meth:`step(steps)`: Callback function for the step button.
        - :meth:`start()`: Start the solver.
        - :meth:`set()`: Set the solver configurations and update the timer.
        - :meth:`stop()`: Stop the solver.
        - :meth:`single_layout()`: Switch plot holder to a single plot layout.
        - :meth:`grid_layout()`: Switch plot holder to a grid plot layout.
        - :meth:`timer_timeout()`: Qt timer timeout callback.
        - :meth:`log(msg)`: Print log messages to the console window and
          standard output.
        - :meth:`update_lines()`: Update all data lines after the solver
          finishes computation.
    """
    def __init__(self):
        self.solver_config_data = [
                ["gamma", 1.4, "The ratio of the specific heats."],
                ["p_left", 1.0, "The pressure of left hand side."],
                ["rho_left", 1.0, "The density of left hand side."],
                ["p_right", 0.1, "The pressure of right hand side."],
                ["rho_right", 0.125, "The density of right hand side."],
                ["xmin", -10, "The most left point of x axis."],
                ["xmax", 10, "The most right point of x axis."],
                ["ncoord", 201, "Number of grid point."],
                ["time_increment", 0.05, "The density of right hand side."],
                ["timer_interval", 10, "Qt timer interval"],
                ["max_steps", 50, "Maximum step"],
                ["profiling", False, "Turn on / off solver profiling"],
                ]
        self.solver_config = SolverConfig(self.solver_config_data)
        self.plot_config_data = []
        self.density = QuantityLine(name="density",
                                    unit=r"$\mathrm{kg}/\mathrm{m}^3$",
                                    y_upper_lim=1.2,
                                    y_bottom_lim=-0.1)
        self.plot_config_data.append([self.density.name,
                                      True,
                                      self.density.y_upper_lim,
                                      self.density.y_bottom_lim])
        self.velocity = QuantityLine(name="velocity",
                                     unit=r"$\mathrm{m}/\mathrm{s}$",
                                     y_upper_lim=1.2,
                                     y_bottom_lim=-0.1)
        self.plot_config_data.append([self.velocity.name,
                                      True,
                                      self.velocity.y_upper_lim,
                                      self.velocity.y_bottom_lim])
        self.pressure = QuantityLine(name="pressure", unit=r"$\mathrm{Pa}$",
                                     y_upper_lim=1.2,
                                     y_bottom_lim=-0.1)
        self.plot_config_data.append([self.pressure.name,
                                      True,
                                      self.pressure.y_upper_lim,
                                      self.pressure.y_bottom_lim])
        self.temperature = QuantityLine(name="temperature",
                                        unit=r"$\mathrm{K}$",
                                        y_upper_lim=0.15,
                                        y_bottom_lim=0.0)
        self.plot_config_data.append([self.temperature.name,
                                      False,
                                      self.temperature.y_upper_lim,
                                      self.temperature.y_bottom_lim])
        self.internal_energy = QuantityLine(name="internal_energy",
                                            unit=r"$\mathrm{J}/\mathrm{kg}$",
                                            y_upper_lim=3.0,
                                            y_bottom_lim=1.5)
        self.plot_config_data.append([self.internal_energy.name,
                                      False,
                                      self.internal_energy.y_upper_lim,
                                      self.internal_energy.y_bottom_lim])
        self.entropy = QuantityLine(name="entropy",
                                    unit=r"$\mathrm{J}/\mathrm{K}$",
                                    y_upper_lim=2.2,
                                    y_bottom_lim=0.9)
        self.plot_config_data.append([self.entropy.name,
                                      False,
                                      self.entropy.y_upper_lim,
                                      self.entropy.y_bottom_lim])
        self.plot_config = PlotConfig(self.plot_config_data)
        self.use_grid_layout = False
        self.plot_holder = State()
        self.set_solver_config()
        self.setup_timer()
        self.plot_holder.plot = self.build_single_figure()

    def init_solver(self, gamma=1.4, pressure_left=1.0, density_left=1.0,
                    pressure_right=0.1, density_right=0.125, xmin=-10,
                    xmax=10, ncoord=201, time_increment=0.05):
        """
        This function is used to initialize shock tube solver and setup
        the initial conition.

        :return: None
        """
        self.st = euler1d.ShockTube()
        self.st.build_constant(gamma, pressure_left, density_left,
                               pressure_right, density_right)
        self.st.build_numerical(xmin, xmax, ncoord, time_increment)
        self.st.build_field(t=0)

    def set_solver_config(self):
        """
        Initializing solver configure by user's input, also reset
        the computational results.

        :return None
        """
        self.init_solver(gamma=self.solver_config["gamma"]["value"],
                         pressure_left=self.solver_config["p_left"]["value"],
                         density_left=self.solver_config["rho_left"]["value"],
                         pressure_right=self.solver_config["p_right"]["value"],
                         density_right=self.solver_config["rho_right"]
                                                         ["value"],
                         xmin=self.solver_config["xmin"]["value"],
                         xmax=self.solver_config["xmax"]["value"],
                         ncoord=self.solver_config["ncoord"]["value"],
                         time_increment=(self.solver_config["time_increment"][
                                                            "value"]))
        self.current_step = 0
        self.interval = self.solver_config["timer_interval"]["value"]
        self.max_steps = self.solver_config["max_steps"]["value"]
        self.profiling = self.solver_config["profiling"]["value"]

    def setup_timer(self):
        """
        Steup the Qt timer for data visualization, timer also driver
        solver marching time step.

        :return: None
        """
        self.timer = QTimer()
        self.timer.timeout.connect(self.timer_timeout)

    def build_grid_figure(self):
        """
        Create a matplotlib figure that includes all data lines, each
        represeting a physical variable. The layout of figure is that
        a grid plot.

        :return: FigureCanvas
        """
        x = self.st.svr.coord[self.st.svr.xindices]
        fig = Figure()
        canvas = FigureCanvas(fig)
        ax = canvas.figure.subplots(3, 2)
        fig.tight_layout()
        y_upper_lim_max = 0.0
        y_bottom_lim_min = sys.float_info.max

        for i, (data, color) in enumerate((
                (self.density, 'r'),
                (self.velocity, 'g'),
                (self.pressure, 'b'),
                (self.temperature, 'c'),
                (self.internal_energy, 'k'),
                (self.entropy, 'm')
        )):
            axis = ax[i // 2][i % 2]
            data.axis = axis
            data.ana, = axis.plot(x, np.zeros_like(x),
                                  f'{color}-',
                                  label=f'{data.name}_ana')
            data.num, = axis.plot(x, np.zeros_like(x),
                                  f'{color}x',
                                  label=f'{data.name}_num')
            axis.set_ylabel(f'{data.name}')

            axis.set_xlabel("distance")
            axis.legend()
            axis.grid()
            y_upper_lim_max = max(y_upper_lim_max, data.y_upper_lim)
            y_bottom_lim_min = min(y_bottom_lim_min, data.y_bottom_lim)

        setp(ax, ylim=[y_bottom_lim_min, y_upper_lim_max])
        self.update_lines()
        return canvas

    def build_single_figure(self):
        """
        Create a matplotlib figure that includes up to 3 data lines, each
        represeting a physical variable. The layout of figure is that
        a single plot contains all data lines.

        :return: FigureCanvas
        """
        x = self.st.svr.coord[self.st.svr.xindices]
        fig = Figure()
        canvas = FigureCanvas(fig)
        ax = canvas.figure.subplots()
        fig.tight_layout()
        y_upper_lim_max = 0.0
        y_bottom_lim_min = sys.float_info.max

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
            if self.plot_config[data.name]["line_selection"]:
                data.axis = ax
                data.ana, = ax.plot(x, np.zeros_like(x),
                                    f'{color}-',
                                    label=f'{data.name}_ana')
                data.num, = ax.plot(x, np.zeros_like(x),
                                    f'{color}x',
                                    label=f'{data.name}_num')
                y_upper_lim_max = max(y_upper_lim_max, data.y_upper_lim)
                y_bottom_lim_min = min(y_bottom_lim_min, data.y_bottom_lim)

        ax.set_xlabel("distance")
        ax.grid()
        ax.legend()

        setp(ax, ylim=[y_bottom_lim_min, y_upper_lim_max])
        self.update_lines()

        return canvas

    def march_alpha2(self, steps=1):
        """
        This function is used to call c++ solver to march the time step, also
        calling python side analytical solution to march the time step.

        :return: None
        """
        if self.max_steps and self.current_step > self.max_steps:
            self.stop()
            return

        self.st.svr.march_alpha2(steps=steps)
        self.current_step += steps
        time_current = self.current_step * self.st.svr.time_increment
        self.st.build_field(t=time_current)
        cfl = self.st.svr.cfl
        self.log(f"CFL: min {cfl.min()} max {cfl.max()}")
        self.update_lines()
        if self.profiling:
            self.log(mm.time_registry.report())

    def step(self, steps=1):
        """
        Callback function of step button.

        :return: None
        """
        self.march_alpha2(steps=steps)

    def start(self):
        """
        This callback function don't care button's checked state,
        therefore the checked state is not used in this function.

        :param checked: button is checked or not
        :return: None
        """
        self.timer.start(self.interval)

    def set(self):
        """
        Callback function of set button that set the solver
        configures and setup the timer.

        :return: None
        """
        self.set_solver_config()
        self.setup_timer()
        self.update_layout()

    def stop(self):
        """
        The stop button callback for stopping Qt timer.
        :return: None
        """
        self.timer.stop()

    def update_layout(self):
        """
        To refresh plotting area layout.

        :return: None
        """
        for line in (
            self.density,
            self.velocity,
            self.pressure,
            self.temperature,
            self.internal_energy,
            self.entropy
        ):
            line.y_upper_lim = self.plot_config[line.name][
                    "y_axis_upper_limit"]
            line.y_bottom_lim = self.plot_config[line.name][
                    "y_axis_bottom_limit"]

        if self.use_grid_layout:
            self.plot_holder.plot = self.build_grid_figure()
        else:
            self.plot_holder.plot = self.build_single_figure()

    def single_layout(self):
        """
        Button action callback that switch plot holder to single plot layout.

        :return: None
        """
        self.use_grid_layout = False
        self.plot_holder.plot = self.build_single_figure()

    def grid_layout(self):
        """
        Button action callback that switch plot holder to grid plot layout.

        :return: None
        """
        self.use_grid_layout = True
        self.plot_holder.plot = self.build_grid_figure()

    @Slot()
    def timer_timeout(self):
        """
        Qt timer timeout callback.

        :return: None
        """
        self.step()

    @staticmethod
    def log(msg):
        """
        Print log in both console window and standard output.

        :return: None
        """
        # stdout can be None under some conditions
        # ref: https://github.com/solvcon/modmesh/issues/334
        if sys.stdout is not None:
            sys.stdout.write(msg)
            sys.stdout.write('\n')
        view.mgr.pycon.writeToHistory(msg)
        view.mgr.pycon.writeToHistory('\n')

    def update_lines(self):
        """
        Updating all data lines after the solver finishes
        its computation each time.

        :return: None
        """
        if self.use_grid_layout:
            _s = self.st.svr.xindices
            self.density.update(adata=self.st.density_field,
                                ndata=self.st.svr.density[_s])
            self.pressure.update(adata=self.st.pressure_field,
                                 ndata=self.st.svr.pressure[_s])
            self.velocity.update(adata=self.st.velocity_field,
                                 ndata=self.st.svr.velocity[_s])
            self.temperature.update(adata=self.st.temperature_field,
                                    ndata=self.st.svr.temperature[_s])
            self.internal_energy.update(adata=(self.st.internal_energy_field),
                                        ndata=(self.st.svr.
                                               internal_energy[_s]))
            self.entropy.update(adata=self.st.entropy_field,
                                ndata=self.st.svr.entropy[_s])
        else:
            for name, is_selected, *_ in self.plot_config.state:
                if is_selected:
                    eval(f'(self.{name}.update(adata=self.st.{name}_field,'
                         f' ndata=self.st.svr.{name}[self.st.svr.xindices]))')


class PlotArea(PuiInQt):
    """
    Class for displaying the plot area in the application.

    This class inherits from `PuiInQt` and is responsible for managing the
    display of the plot area in the application.

    Attributes:
        - `app`: The app want to plot something in plotting area.

    Methods:
        - :meth:`setup()`: Placeholder method for setting up the plot area.
        - :meth:`content()`: Method for defining the content of the plot area,
          including a toolbar and the actual plot.
    """
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app

    def setup(self):
        pass

    def content(self):
        """
        Define the GUI layout of plotting area

        :return: nothing
        """
        with ToolBar():
            QtInPui(NavigationToolbar2QT(self.app.plot_holder.plot, None))
        QtInPui(self.app.plot_holder.plot)


class ConfigWindow(PuiInQt):
    """
    ConfigWindow class for managing solver and plotting area configurations.

    This class inherit from the PuiInQt class and provides a graphical user
    interface for managing solver configurations. It includes options to set
    the solver type, view and edit configuration parameters, and control the
    solver's behavior.

    Attributes:
        - `app`: The app want to plot something in plotting area.
        - `solver_config` (:class:`SolverConfig`): Configuration object
          for the solver.
        - `plot_config` (:class:`PlotConfig`): Configuration object
          for the plotting area.
        - `state` (:class:`State`): The state object holding plot configuration
          window opening state.

    Methods:
        - :meth:`setup()`: Setup method to configure the window.
        - :meth:`content()`: Method to define the content of the window.
        - :meth:`on_open()`: Callback function when plot configure modal
          window is opened.
        - :meth:`on_close()`: Callback function when plot configure modal
          windows is closed.
    """
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.state = State()
        self.state.plot_config_open = False

    def setup(self):
        """
        Assign the configure object from app

        :return: nothing
        """
        self.solver_config = self.app.solver_config
        self.plot_config = self.app.plot_config

    def on_open(self):
        self.state.plot_config_open = True

    def on_close(self):
        self.state.plot_config_open = False

    def content(self):
        """
        Define the GUI layout of the window

        :return: nothing
        """
        with VBox():
            with VBox().layout(weight=4):
                Label("Solver")
                with ComboBox():
                    ComboBoxItem("Euler1D-CESE")
                Label("Configuration")
                with Scroll():
                    Table(self.solver_config)
                Button("Set").click(self.app.set)
                Button("Option").click(self.on_open)
            with VBox().layout(weight=1):
                Spacer()
                Button("Start").click(self.app.start)
                Button("Stop").click(self.app.stop)
                Button("Step").click(self.app.step)
            with (Modal(self.state("plot_config_open"),
                        title="Plot configuration")
                  .open(self.on_open)
                  .close(self.on_close)):
                with VBox():
                    with HBox():
                        Label("Layout selection")
                        Button("Grid").click(lambda: self.app.grid_layout())
                        Button("Single").click((lambda:
                                                self.app.single_layout()))
                    Label("Data line configuration")
                    Table(self.plot_config)
                    Button("Save").click(lambda: self.app.update_layout())


def load_app():
    app = Euler1DApp()
    plotting_area = PlotArea(Window(), app)

    config_window = ConfigWindow(Window(), app)
    config_widget = QDockWidget("config")
    config_widget.setWidget(config_window.ui.ui)

    view.mgr.mainWindow.addDockWidget(Qt.LeftDockWidgetArea, config_widget)
    _subwin = view.mgr.addSubWindow(plotting_area.ui.ui)
    _subwin.showMaximized()

    config_window.redraw()
    plotting_area.redraw()
    _subwin.show()

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
