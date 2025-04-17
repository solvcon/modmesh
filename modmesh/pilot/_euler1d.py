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


import sys
from dataclasses import dataclass

import numpy as np

import matplotlib
import matplotlib.pyplot
from matplotlib.backends.backend_qtagg import FigureCanvas
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
from matplotlib.pyplot import setp

from PySide6.QtCore import QTimer, Slot, Qt, QAbstractTableModel
from PySide6.QtWidgets import (QDockWidget, QLabel, QVBoxLayout, QHBoxLayout,
                               QComboBox, QPushButton, QSpacerItem,
                               QSizePolicy, QDialog, QWidget, QTableView)

from .. import core as mcore
from ..onedim import euler1d
from ._gui_common import PilotFeature


@dataclass
class QuantityLine(object):
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
        - `color` (str): Color of line or symbol.
        - `y_upper_lim` (float): y axis upper limit.
        - `y_bottom_lim` (float): y axis bottom limit.

    Methods:
        - :meth:`update_ana(x, y)`: Update the line data and redraw the plot.
        - :meth:`update_num(x, y)`: Update the line data and redraw the plot.

    """
    ana: matplotlib.lines.Line2D = None
    num: matplotlib.lines.Line2D = None
    axis: matplotlib.pyplot.axis = None
    name: str = ""
    unit: str = ""
    color: str = ""
    y_upper_lim: float = 0.0
    y_bottom_lim: float = 0.0

    def update_ana(self, x=None, y=None):
        """
        Update analytical data and redraw the plot.

        :param x, y (type: ndarray): Analytical data.
        :return: None
        """
        if x is not None:
            self.ana.set_xdata(x)
        if y is not None:
            self.ana.set_ydata(y)
        self.ana.figure.canvas.draw()

    def update_num(self, x=None, y=None):
        """
        Update numerical data and redraw the plot.

        :param x, y (type: ndarray): data.
        :return: None
        """
        if x is not None:
            self.num.set_xdata(x)
        if y is not None:
            self.num.set_ydata(y)
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
        :ivar: _dim_idx (int): using to indicate which dimension is
            currenlty being accessed by __getitem__.
    """

    def __init__(self, data, dim_idx=0, header=None):
        self._data = data
        self._header = header
        self._dim_idx = dim_idx

    def __getitem__(self, key):
        if self._dim_idx == 0:
            for row_idx, row in enumerate(self._data):
                if key == row[0]:
                    return _Accessor(self._data[row_idx],
                                     self._dim_idx + 1,
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
        :ivar: _tbl_content (list[list]): The content of the configuration
            table.
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
        self._tbl_content = input_data
        self._col_header = col_headers

        expected_length = len(self._col_header)
        for i, row in enumerate(self._tbl_content):
            if len(row) != expected_length:
                raise ValueError(
                    f"Row {i} has length {len(row)}, "
                    f"expected {expected_length}"
                )

    def __getitem__(self, key):
        return _Accessor(self._tbl_content, 0, self._col_header)[key]

    def data(self, row, col):
        """
        Get the value at a specific row and column in the configuration table.

        :param row: Row index.
        :type row: int
        :param col: Column index.
        :type col: int
        :return:
            The value at the specified location in the configuration table.
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
        - `_tbl_content` (list[list]): The content of the configuration table.
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
        - `_tbl_content` (list[list]): The content of the configuration table.
        - `_col_header` (list): The header for the configuration
          table columns.
    """

    def __init__(self, input_data):
        super().__init__(input_data, ["variable", "line_selection"])

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


class _1DApp(PilotFeature):
    """
    Main application for 1D solver.

    This class provides the main application logic for the 1D solver.
    It includes methods for initializing the solver, configuring parameters,
    setting up timers, and building visualization figures.

    Attributes:
        - `solver_config` (:class:`SolverConfig`): Configuration object for
          the solver.
        - `plot_data` (list[str, bool]): Data list for the plotting area.
        - `plot_config` (:class:`PlotConfig`): Configuration object for the
          plotting area.
        - `plot` (:class:`State`): State object for the holding plots.
        - `plot_ana` (bool): Flag indicating whether to plot analytical data.
        - `plot_num` (bool): Flag indicating whether to plot numerical data.
        - `use_grid_layout` (bool): Flag indicating whether to use a grid
          layout.

    Methods:
        - :meth:`populate_menu()`: Set menu item for GUI.
        - :meth:`run()`: Create the GUI environment.
        - :meth:`setup_app()`: Create the window for solver.
        - :meth:`init_solver_config()`: Initialize solver configuration data.
        - :meth:`set_plot_data()`: Set the property of st and set list of data.
        - :meth:`set_solver_config()`: Initialize solver configure by user's
          input, also reset the computational results.
        - :meth:`init_solver()`: Initialize the shock tube solver and set up
          the initial conditions.
        - :meth:`setup_timer()`: Set up the Qt timer for data visualization.
        - :meth:`build_single_figure()`: Build a single-figure layout for
          visualization.
        - :meth:`build_grid_figure()`: Build a grid figure for visualization.
        - :meth:`init_plot_data(data, y_limit)`: Initialize analytical and
          numerical data in figure.
        - :meth:`step(steps)`: Callback function for the step button.
        - :meth:`update_step(steps)`: Update data at current step.
        - :meth:`start()`: Start the solver.
        - :meth:`set()`: Set the solver configurations and update the timer.
        - :meth:`update_layout()`: Refresh plotting area layout.
        - :meth:`stop()`: Stop the solver.
        - :meth:`single_layout()`: Switch plot holder to a single plot layout.
        - :meth:`grid_layout()`: Switch plot holder to a grid plot layout.
        - :meth:`timer_timeout()`: Callback function for Qt timer.
        - :meth:`log(msg)`: Print log messages to the console window and
          standard output.
        - :meth:`update_plot()`: Updating plot after the solver finishes its
          computation each time.
    """
    st = None
    solver_config: SolverConfig = None
    plot_data: list[str, bool] = None
    plot_config: PlotConfig = None
    plot: FigureCanvas = None
    plot_ana: bool = False
    plot_num: bool = False
    use_grid_layout: bool = False

    def populate_menu(self):
        """
        Set menu item for GUI.
        """
        raise NotImplementedError(f"{self.__class__.__name__} not implemented"
                                  f"{sys._getframe().f_code.co_name}")

    def run(self):
        """
        Create the GUI environment.
        """
        self.setup_app()

        # A new dock widget for showing config
        config_window = ConfigWindow(self)
        config_widget = QDockWidget("config")
        config_widget.setWidget(config_window)
        self._mgr.mainWindow.addDockWidget(Qt.LeftDockWidgetArea,
                                           config_widget)

        # A new sub-window (`QMdiSubWindow`) for the plot area
        self._subwin = self._mgr.addSubWindow(QWidget())
        self._subwin.setWidget(PlotArea(self))
        self._subwin.showMaximized()
        self._subwin.show()

    def setup_app(self):
        """
        Create the window for solver.
        """
        self.init_solver_config()
        self.set_plot_data()
        self.plot_config = PlotConfig(self.plot_data)
        self.set_solver_config()
        self.setup_timer()
        self.plot = self.build_single_figure()

    def init_solver_config(self):
        """
        Initialize solver configuration data.
        """
        raise NotImplementedError(f"{self.__class__.__name__} not implemented"
                                  f"{sys._getframe().f_code.co_name}")

    def set_plot_data(self):
        """
        Set the property of st and set list of data.
        """
        raise NotImplementedError(f"{self.__class__.__name__} not implemented"
                                  f"{sys._getframe().f_code.co_name}")

    def set_solver_config(self):
        """
        Initialize solver configure by user's input, also reset the
        computational results.
        """
        self.init_solver()
        self.current_step = 0
        self.time_interval = self.solver_config["time_interval"]["value"]
        self.max_steps = self.solver_config["max_steps"]["value"]
        self.profiling = self.solver_config["profiling"]["value"]

    def init_solver(self):
        """
        Initialize solver and set up the initial conition.
        """
        raise NotImplementedError(f"{self.__class__.__name__} not implemented"
                                  f"{sys._getframe().f_code.co_name}")

    def setup_timer(self):
        """
        Set up the Qt timer for data visualization.
        """
        self.timer = QTimer()
        self.timer.timeout.connect(self.timer_timeout)

    def build_single_figure(self):
        """
        Build a single-figure layout for visualization.

        :return: FigureCanvas
        """
        fig = Figure()
        canvas = FigureCanvas(fig)
        ax = canvas.figure.subplots()
        fig.tight_layout()

        y_limit = [sys.float_info.max, 0.0]
        for [name, *_] in self.plot_data:
            if self.plot_config[name]["line_selection"]:
                data = getattr(self, name)
                data.axis = ax
                y_limit = self.init_plot_data(data, y_limit)

        setp(ax, ylim=y_limit)
        self.update_plot()
        return canvas

    def build_grid_figure(self):
        """
        Build a grid figure for visualization.

        :return: FigureCanvas
        """
        fig = Figure()
        canvas = FigureCanvas(fig)
        ax = canvas.figure.subplots(3, 2)
        fig.tight_layout()

        y_limit = [sys.float_info.max, 0.0]
        for i, [name, *_] in enumerate(self.plot_data):
            if self.plot_config[name]["line_selection"]:
                data = getattr(self, name)
                data.axis = ax[i // 2][i % 2]
                y_limit = self.init_plot_data(data, y_limit)

        setp(ax, ylim=y_limit)
        self.update_plot()
        return canvas

    def init_plot_data(self, data, y_limit):
        """
        Initialize analytical and numerical data in figure.

        :param data (type: QuantityLine): property of solver
        :param y_limit (type: list[int, int]): the bottom and top limit of y
        :return: the bottom and top limit of y axis (type: list[int, int])
        """
        if self.plot_ana:
            x = self.st.coord_field
            data.ana, = data.axis.plot(x, np.zeros_like(x),
                                       f"{data.color}-",
                                       label=f'{data.name}_ana')
        if self.plot_num:
            x = self.st.svr.coord[self.st.svr.xindices]
            data.num, = data.axis.plot(x, np.zeros_like(x),
                                       f"{data.color}x",
                                       label=f'{data.name}_num')
        data.axis.set_xlabel("distance")
        data.axis.legend()
        data.axis.grid()

        y_limit[0] = min(y_limit[0], data.y_bottom_lim)
        y_limit[1] = max(y_limit[1], data.y_upper_lim)
        return y_limit

    def step(self, steps=1):
        """
        Callback function for the step button.
        """
        if self.max_steps and self.current_step > self.max_steps:
            self.stop()
            return

        self.update_step(steps)
        self.update_plot()
        if self.profiling:
            self.log(mcore.time_registry.report())

    def update_step(self, steps=1):
        """
        Update data at current step.
        """
        raise NotImplementedError(f"{self.__class__.__name__} not implemented"
                                  f"{sys._getframe().f_code.co_name}")

    def start(self):
        """
        Start the solver.
        """
        self.timer.start(self.time_interval)

    def set(self):
        """
        Set the solver configurations and update the timer.
        """
        self.set_solver_config()
        self.setup_timer()
        self.update_layout()

        # Update PlotArea while click set button
        self._subwin.setWidget(PlotArea(self))

    def update_layout(self):
        """
        Refresh plotting area layout.
        """
        if self.use_grid_layout:
            self.plot = self.build_grid_figure()
        else:
            self.plot = self.build_single_figure()

    def stop(self):
        """
        Stop the solver.
        """
        self.timer.stop()

    def single_layout(self):
        """
        Switch plot holder to a single plot layout.
        """
        self.use_grid_layout = False
        self.plot = self.build_single_figure()

    def grid_layout(self):
        """
        Switch plot holder to a grid plot layout.
        """
        self.use_grid_layout = True
        self.plot = self.build_grid_figure()

    @Slot()
    def timer_timeout(self):
        """
        Callback function for Qt timer timeout.
        """
        self.step()

    def log(self, msg):
        """
        Print log messages to the console window and standard output.
        """
        # stdout can be None under some conditions
        # ref: https://github.com/solvcon/modmesh/issues/334
        if sys.stdout is not None:
            sys.stdout.write(msg)
            sys.stdout.write('\n')
        self._pycon.writeToHistory(msg)
        self._pycon.writeToHistory('\n')

    def update_plot(self):
        """
        Updating plot after the solver finishes its computation each time.
        """
        if self.use_grid_layout:
            for [name, *_] in self.plot_data:
                data = getattr(self, name)
                self.update_plot_data(data)
        else:
            for [name, *_] in self.plot_data:
                if self.plot_config[name]["line_selection"]:
                    data = getattr(self, name)
                    self.update_plot_data(data)

    def update_plot_data(self, data):
        """
        Update analytical and numerical data.

        :param data (type: QuantityLine): property of solver
        """
        if self.plot_ana:
            ana_x = self.st.coord_field
            ana_y = getattr(self.st, data.name + "_field")
            data.update_ana(x=ana_x, y=ana_y)
        if self.plot_num:
            num_y = getattr(self.st.svr, data.name)[self.st.svr.xindices]
            data.update_num(y=num_y)


class Euler1DApp(_1DApp):
    """
    Main application for Euler 1D solver.
    """

    def populate_menu(self):
        """
        Set menu item for GUI.
        """
        self._add_menu_item(
            menu=self._mgr.oneMenu,
            text="Euler solver",
            tip="One-dimensional shock-tube problem with Euler solver",
            func=self.run,
        )

    def init_solver_config(self):
        """
        Set shock tube solver configuration by user' input, and then reset the
        initial condition.
        """
        solver_config_data = [
            ["gamma", 1.4, "The ratio of the specific heats."],
            ["p_left", 1.0, "The pressure of left hand side."],
            ["rho_left", 1.0, "The density of left hand side."],
            ["p_right", 0.1, "The pressure of right hand side."],
            ["rho_right", 0.125, "The density of right hand side."],
            ["xmin", -10, "The most left point of x axis."],
            ["xmax", 10, "The most right point of x axis."],
            ["ncoord", 201, "Number of grid point."],
            ["time_increment", 0.05, "The density of right hand side."],
            ["time_interval", 10, "Qt timer interval"],
            ["max_steps", 50, "Maximum step"],
            ["profiling", False, "Turn on / off solver profiling"],
        ]
        self.solver_config = SolverConfig(solver_config_data)

    def set_plot_data(self):
        """
        Set the analytical and numerical data for plot.
            - density
            - pressure
            - velocity
            - temperature
            - internal_energy
            - entropy
        """
        self.plot_ana = True
        self.plot_num = True
        self.plot_data = []

        density = QuantityLine(name="density",
                               unit=r"$\mathrm{kg}/\mathrm{m}^3$",
                               color='r',
                               y_upper_lim=1.2,
                               y_bottom_lim=-0.1)
        setattr(self, density.name, density)
        self.plot_data.append([self.density.name, True])

        pressure = QuantityLine(name="pressure",
                                unit=r"$\mathrm{Pa}$",
                                color='g',
                                y_upper_lim=1.2,
                                y_bottom_lim=-0.1)
        setattr(self, pressure.name, pressure)
        self.plot_data.append([self.pressure.name, True])

        velocity = QuantityLine(name="velocity",
                                unit=r"$\mathrm{m}/\mathrm{s}$",
                                color='b',
                                y_upper_lim=1.2,
                                y_bottom_lim=-0.1)
        setattr(self, velocity.name, velocity)
        self.plot_data.append([self.velocity.name, True])

        temperature = QuantityLine(name="temperature",
                                   unit=r"$\mathrm{K}$",
                                   color='c',
                                   y_upper_lim=0.15,
                                   y_bottom_lim=0.0)
        setattr(self, temperature.name, temperature)
        self.plot_data.append([self.temperature.name, False])

        internal_energy = QuantityLine(name="internal_energy",
                                       unit=r"$\mathrm{J}/\mathrm{kg}$",
                                       color='k',
                                       y_upper_lim=3.0,
                                       y_bottom_lim=1.5)
        setattr(self, internal_energy.name, internal_energy)
        self.plot_data.append([self.internal_energy.name, False])

        entropy = QuantityLine(name="entropy",
                               unit=r"$\mathrm{J}/\mathrm{K}$",
                               color='m',
                               y_upper_lim=2.2,
                               y_bottom_lim=0.9)
        setattr(self, entropy.name, entropy)
        self.plot_data.append([self.entropy.name, False])

    def init_solver(self):
        """
        Initialize the shock tube solver and set up the initial condition.
        """
        self.st = euler1d.ShockTube()
        _s = self.solver_config
        self.st.build_constant(gamma=_s["gamma"]["value"],
                               pressure1=_s["p_left"]["value"],
                               density1=_s["rho_left"]["value"],
                               pressure5=_s["p_right"]["value"],
                               density5=_s["rho_right"]["value"])
        self.st.build_numerical(xmin=_s["xmin"]["value"],
                                xmax=_s["xmax"]["value"],
                                ncoord=_s["ncoord"]["value"],
                                time_increment=_s["time_increment"]["value"])
        self.st.build_field(t=0)

    def update_step(self, steps):
        """
        Update data at current step.
            - numerical data solved in c++
            - analytical data solved in python
        """
        self.st.svr.march_alpha2(steps=steps)
        self.current_step += steps
        time_current = self.current_step * self.st.svr.time_increment
        self.st.build_field(t=time_current)
        cfl = self.st.svr.cfl
        self.log(f"CFL: min {cfl.min()} max {cfl.max()}")


class PlotArea(QWidget):
    """
    Class for displaying the plot area in the application.

    This class inherits from `QWidget` and is responsible for managing the
    display of the plot area in the application.

    Attributes:
        - `app`: The app want to plot something in plotting area.

    Methods:
        - :meth:`init_ui()`: Define the GUI layout of the window.
    """
    def __init__(self, app, parent=None):
        super().__init__(parent)
        self.app = app
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.addWidget(NavigationToolbar2QT(self.app.plot, None))
        layout.addWidget(self.app.plot)

        self.setLayout(layout)


class ConfigTableModel(QAbstractTableModel):
    """
    ConfigTableModel class for displaying configuration data.

    This class inherits from the QAbstractTableModel class and provides a model
    for displaying configuration data using table. It is used in the QTable
    class to display solver and plotting area configurations.

    Attributes:
        - `config` (:class:`GUIConfig`): Configuration object for the model.

    Methods: (Overridden from QAbstractTableModel)
        - :meth:`rowCount()`: Get the number of rows in the model.
        - :meth:`columnCount()`: Get the number of columns in the model.
        - :meth:`data(index, role)`: Get the data at a specific index in the
          model.
        - :meth:`setData(index, value, role)`: Set the data at a specific index
          in the model.
        - :meth:`flags(index)`: Get the flags for a specific index in the
          model.
        - :meth:`headerData(section, orientation, role)`: Get the header data
          for a specific section in the model.
    """

    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.config = config

    def rowCount(self, parent):
        return self.config.rowCount()

    def columnCount(self, parent):
        return self.config.columnCount()

    def data(self, index, role):
        if role in {Qt.DisplayRole, Qt.EditRole}:
            return self.config.data(index.row(), index.column())
        return None

    def setData(self, index, value, role):
        if role == Qt.EditRole:
            self.config.setData(index.row(), index.column(), value)
            return

    def flags(self, index):
        if self.config.editable(index.row(), index.column()):
            return super().flags(index) | Qt.ItemIsEditable
        return super().flags(index)

    def headerData(self, section, orientation, role):
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            return self.config.columnHeader(section)
        return None


class PlotConfigDialog(QDialog):
    """
    PlotConfigDialog class for managing plot configuration.

    This class inherits from the QDialog class and provides a GUI for managing
    plot configuration. It will pop up when clicking "Option" button. It
    includes options for selecting the layout and setting data line config.
    """
    def __init__(self, app, parent=None):
        super().__init__(parent)
        self.app = app
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Plot Configuration")
        layout = QVBoxLayout()

        layout_selection_label = QLabel("Layout selection")
        layout.addWidget(layout_selection_label)

        hbox = QHBoxLayout()
        grid_button = QPushButton("Grid")
        grid_button.clicked.connect(self.app.grid_layout)
        hbox.addWidget(grid_button)

        single_button = QPushButton("Single")
        single_button.clicked.connect(self.app.single_layout)
        hbox.addWidget(single_button)

        layout.addLayout(hbox)

        data_line_config_label = QLabel("Data line configuration")
        layout.addWidget(data_line_config_label)

        table = QTableView()
        table_model = ConfigTableModel(self.app.plot_config)
        table.setModel(table_model)
        layout.addWidget(table)

        save_button = QPushButton("Save")
        save_button.clicked.connect(self.app.update_layout)
        layout.addWidget(save_button)

        self.setLayout(layout)


class ConfigWindow(QWidget):
    """
    ConfigWindow class for managing solver and plotting area configurations.

    This class inherit from the QWidget class and provides a graphical user
    interface for managing solver configurations. It includes options to set
    the solver type, view and edit configuration parameters, and control the
    solver's behavior.

    Attributes:
        - `app`: The app want to plot something in plotting area.
        - `solver_config` (:class:`SolverConfig`): Configuration object
          for the solver.
        - `plot_config` (:class:`PlotConfig`): Configuration object
          for the plotting area.

    Methods:
        - :meth:`setup()`: Setup method to configure the window.
        - :meth:`on_open()`: Callback function when plot configure modal
          window is opened.
        - :meth:`on_close()`: Callback function when plot configure modal
          windows is closed.
        - :meth:`init_ui()`: Define the GUI layout of the window.
    """

    def __init__(self, app, parent=None):
        super().__init__(parent)
        self.app = app
        self.plot_config_open = False
        self.setup()
        self.init_ui()

    def setup(self):
        """
        Assign the configure object from app

        :return: nothing
        """
        self.solver_config = self.app.solver_config
        self.plot_config = self.app.plot_config

    def on_open(self):
        self.plot_config_open = True
        if not self.plot_config_dialog:
            self.plot_config_dialog = PlotConfigDialog(self.app, self)
        self.plot_config_dialog.exec_()

    def on_close(self):
        self.plot_config_open = False
        if self.plot_config_dialog:
            self.plot_config_dialog.close()

    def init_ui(self):
        """
        Define the GUI layout of the window for this class

        :return: QWidget
        """
        main_layout = QVBoxLayout(self)

        # First VBox, handle operation about solver configuration, weight=4
        vbox1 = QVBoxLayout()
        vbox1.setStretch(0, 4)

        solver_label = QLabel("Solver")
        vbox1.addWidget(solver_label)

        combo_box = QComboBox()
        combo_box.addItem("Euler1D-CESE")
        vbox1.addWidget(combo_box)

        config_label = QLabel("Configuration")
        vbox1.addWidget(config_label)

        table = QTableView()
        table_model = ConfigTableModel(self.solver_config)
        table.setModel(table_model)
        vbox1.addWidget(table)

        set_button = QPushButton("Set")
        set_button.clicked.connect(self.app.set)
        vbox1.addWidget(set_button)

        option_button = QPushButton("Option")
        option_button.clicked.connect(self.on_open)
        vbox1.addWidget(option_button)

        main_layout.addLayout(vbox1)

        # Second VBox, handle operation about calculation, weight=1
        vbox2 = QVBoxLayout()
        vbox2.setStretch(0, 1)

        spacer = QSpacerItem(
            20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        vbox2.addItem(spacer)

        start_button = QPushButton("Start")
        start_button.clicked.connect(self.app.start)
        vbox2.addWidget(start_button)

        stop_button = QPushButton("Stop")
        stop_button.clicked.connect(self.app.stop)
        vbox2.addWidget(stop_button)

        step_button = QPushButton("Step")
        step_button.clicked.connect(lambda: self.app.step())
        vbox2.addWidget(step_button)

        main_layout.addLayout(vbox2)

        # Modal window for plot configuration
        self.plot_config_dialog = PlotConfigDialog(self.app, self)

        self.setLayout(main_layout)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
