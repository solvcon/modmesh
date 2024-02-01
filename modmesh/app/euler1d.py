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
from matplotlib.ticker import FormatStrFormatter
from PySide6.QtCore import QTimer, Slot, Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QFileDialog, QDockWidget
from PUI.state import State
from PUI.PySide6.base import PuiInQt, QtInPui
from PUI.PySide6.button import Button
from PUI.PySide6.layout import VBox, Spacer
from PUI.PySide6.scroll import Scroll
from PUI.PySide6.window import Window
from PUI.PySide6.combobox import ComboBox, ComboBoxItem
from PUI.PySide6.label import Label
from PUI.PySide6.table import Table
from PUI.PySide6.toolbar import ToolBar, ToolBarAction
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

    Methods:
        - :meth:`update(xdata, adata, ndata)`: Update the line data and
          redraw the plot.
    """
    ana: matplotlib.lines.Line2D = None
    num: matplotlib.lines.Line2D = None
    axis: matplotlib.pyplot.axis = None
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
        self.axis.relim()
        self.axis.autoscale_view()
        self.ana.figure.canvas.draw()
        self.num.figure.canvas.draw()


class SolverConfig():
    """
    Configuration class for the solver.

    This class provides a configuration interface for the solver, allowing
    users to set and retrieve parameters related to the simulation.

    Attributes:
        - `state` (:class:`State`): The state object holding configuration
          data.
        - `state.data` (list): A list containing configuration parameters
          in the form of [variable_name, value, description].
        - `_tbl_content` (:class:`State`): The content of the
          configuration table.
        - `_col_header` (list): The header for the configuration
          table columns.

    Methods:
        - :meth:`data(row, col)`: Get the value at a specific row and column
          in the configuration table.
        - :meth:`setData(row, col, value)`: Set the value at a specific row
          and column in the configuration table.
        - :meth:`columnHeader(col)`: Get the header for a specific column
          in the configuration table.
        - :meth:`editable(row, col)`: Check if a cell in the configuration
          table is editable.
        - :meth:`rowCount()`: Get the number of rows in the configuration
          table.
        - :meth:`columnCount()`: Get the number of columns in the
          configuration table.
        - :meth:`get_var(key)`: Get the value of a configuration variable
          based on its key.
    """
    def __init__(self):
        self.state = State()
        self.state.data = [
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
        self._tbl_content = self.state("data")
        self._col_header = ["Variable", "Value", "Description"]

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
        return self._tbl_content.value[row][col]

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
        self._tbl_content.value[row][col] = value
        self._tbl_content.emit()

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
        return len(self._tbl_content.value)

    def columnCount(self):
        """
        Get the number of columns in the configuration table.

        :return: The number of columns.
        """
        return len(self._tbl_content.value[0])

    def get_var(self, key):
        """
        Get the value of a configuration variable based on its key.

        :param key: The key of the variable.
        :type key: str
        :return: The value of the specified variable.
        """
        for ele in self.state("data").value:
            if key == ele[0]:
                return ele[1]
        return None


class Euler1DApp():
    """
    Main application class for the Euler 1D solver.

    This class provides the main application logic for the Euler 1D solver.
    It includes methods for initializing the solver, configuring parameters,
    setting up timers, and building visualization figures.

    Attributes:
        - `config` (:class:`SolverConfig`): Configuration object
          for the solver.
        - `data_lines` (dict): Dictionary containing QuantityLine objects and
          their display status.
        - Other QuantityLine objects such as `density`, `velocity`, etc.
        - `use_grid_layout` (bool): Flag indicating whether to use a
          grid layout.
        - `checkbox_select_num` (int): Number of checkboxes selected.
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
        - :meth:`save_file()`: Save the current plot to a file.
        - :meth:`timer_timeout()`: Qt timer timeout callback.
        - :meth:`log(msg)`: Print log messages to the console window and
          standard output.
        - :meth:`update_lines()`: Update all data lines after the solver
          finishes computation.
    """
    def __init__(self):
        self.config = SolverConfig()
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
        self.use_grid_layout = False
        self.checkbox_select_num = 3
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
        self.init_solver(gamma=self.config.get_var("gamma"),
                         pressure_left=self.config.get_var("p_left"),
                         density_left=self.config.get_var("rho_left"),
                         pressure_right=self.config.get_var("p_right"),
                         density_right=self.config.get_var("rho_right"),
                         xmin=self.config.get_var("xmin"),
                         xmax=self.config.get_var("xmax"),
                         ncoord=self.config.get_var("ncoord"),
                         time_increment=self.config.get_var("time_increment"))
        self.current_step = 0
        self.interval = self.config.get_var("timer_interval")
        self.max_steps = self.config.get_var("max_steps")
        self.profiling = self.config.get_var("profiling")

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
        x = self.st.svr.coord[::2]
        fig = Figure()
        canvas = FigureCanvas(fig)
        ax = canvas.figure.subplots(3, 2)
        fig.tight_layout()

        for i, (data, color) in enumerate((
                (self.density, 'r'),
                (self.velocity, 'g'),
                (self.pressure, 'b'),
                (self.temperature, 'c'),
                (self.internal_energy, 'k'),
                (self.entropy, 'm')
        )):
            axis = ax[i // 2][i % 2]
            axis.autoscale(enable=True, axis='y', tight=False)
            data.axis = axis
            data.ana, = axis.plot(x, np.zeros_like(x),
                                  f'{color}-',
                                  label=f'{data.name}_ana')
            data.num, = axis.plot(x, np.zeros_like(x),
                                  f'{color}x',
                                  label=f'{data.name}_num')
            axis.set_ylabel(f'{data.name} ({data.unit})')

            axis.set_xlabel("distance (m)")
            axis.legend()
            axis.grid()

        self.update_lines()
        return canvas

    def build_single_figure(self):
        """
        Create a matplotlib figure that includes up to 3 data lines, each
        represeting a physical variable. The layout of figure is that
        a single plot contains all data lines.

        :return: FigureCanvas
        """
        x = self.st.svr.coord[::2]
        fig = Figure()
        canvas = FigureCanvas(fig)
        ax = canvas.figure.subplots()
        fig.tight_layout()
        ax.autoscale(enable=True, axis='y', tight=False)

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
                    data.axis = ax
                    data.ana, = ax.plot(x, np.zeros_like(x),
                                        f'{color}-',
                                        label=f'{data.name}_ana')
                    data.num, = ax.plot(x, np.zeros_like(x),
                                        f'{color}x',
                                        label=f'{data.name}_num')
                    ax.set_ylabel(f'{data.name} ({data.unit})')
                    main_axis_plotted = True
                else:
                    ax_new = ax.twinx()
                    data.axis = ax_new
                    data.ana, = ax_new.plot(x, np.zeros_like(x),
                                            f'{color}-',
                                            label=f'{data.name}_ana')
                    data.num, = ax_new.plot(x, np.zeros_like(x),
                                            f'{color}x',
                                            label=f'{data.name}_num')
                    ax_new.spines.right.set_position(("axes",
                                                      (1 + (select_num - 1)
                                                       * 0.2)))
                    ax_new.set_ylabel(f'{data.name} ({data.unit})')
                    ax_new.yaxis.set_major_formatter((FormatStrFormatter
                                                      ('%.2f')))
                select_num += 1

        ax.set_xlabel("distance (m)")
        ax.grid()

        # These parameters are the results obtained by my tuning on GUI
        fig.subplots_adjust(left=0.1,
                            right=0.97 - (self.checkbox_select_num - 1) * 0.1,
                            bottom=0.093, top=0.976,
                            wspace=0.2, hspace=0.2)

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
        if self.use_grid_layout:
            self.plot_holder.plot = self.build_grid_figure()
        else:
            self.plot_holder.plot = self.build_single_figure()

    def stop(self):
        """
        The stop button callback for stopping Qt timer.
        :return: None
        """
        self.timer.stop()

    def single_layout(self):
        """
        Toolbar action callback that switch plot holder to single plot layout.

        :return: None
        """
        self.use_grid_layout = False
        self.plot_holder.plot = self.build_single_figure()

    def grid_layout(self):
        """
        Toolbar action callback that switch plot holder to grid plot layout.

        :return: None
        """
        self.use_grid_layout = True
        self.plot_holder.plot = self.build_grid_figure()

    def save_file(self):
        """
        Toolbar action callback, that use pixmap to store plotting area
        and open a dialog that allows user to decide where to store the
        figure file.

        :return: None
        """
        fig = QPixmap(self.plot_holder.plot.size())
        self.plot_holder.plot.render(fig)

        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getSaveFileName(None, "Save file", "",
                                                  "All Files (*)",
                                                  options=options)

        if fileName != "":
            fig.save(fileName, "JPG", 100)

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
            self.density.update(adata=self.st.density_field,
                                ndata=self.st.svr.density[::2])
            self.pressure.update(adata=self.st.pressure_field,
                                 ndata=self.st.svr.pressure[::2])
            self.velocity.update(adata=self.st.velocity_field,
                                 ndata=self.st.svr.velocity[::2])
            self.temperature.update(adata=self.st.temperature_field,
                                    ndata=self.st.svr.temperature[::2])
            self.internal_energy.update(adata=(self.st.internal_energy_field),
                                        ndata=(self.st.svr.
                                               internal_energy[::2]))
            self.entropy.update(adata=self.st.entropy_field,
                                ndata=self.st.svr.entropy[::2])
        else:
            for name, data_line in self.data_lines.items():
                if data_line[1]:
                    eval(f'(data_line[0].update(adata=self.st.{name}_field,'
                         f' ndata=self.st.svr.{name}[::2]))')


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
            ToolBarAction("Save").trigger(self.app.save_file)
            ToolBarAction("SingleLayout").trigger(self.app.single_layout)
            ToolBarAction("GridLayout").trigger(self.app.grid_layout)
        QtInPui(self.app.plot_holder.plot)


class ConfigWindow(PuiInQt):
    """
    ConfigWindow class for managing solver configurations.

    This class inherit from the PuiInQt class and provides a graphical user
    interface for managing solver configurations. It includes options to set
    the solver type, view and edit configuration parameters, and control the
    solver's behavior.

    Attributes:
        - `app`: The app want to plot something in plotting area.
        - `config` (:class:`SolverConfig`): Configuration object
          for the solver.

    Methods:
        - :meth:`setup()`: Setup method to configure the window.
        - :meth:`content()`: Method to define the content of the window.
    """
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app

    def setup(self):
        """
        Assign the configure object from app

        :return: nothing
        """
        self.config = self.app.config

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
                    Table(self.config)
                Button("Set").click(self.app.set)
            with VBox().layout(weight=1):
                Spacer()
                Button("Start").click(self.app.start)
                Button("Stop").click(self.app.stop)
                Button("Step").click(self.app.step)


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
