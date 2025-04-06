# Copyright (c) 2025, Jie-Yin Lin <geneeee0315@gmail.com>
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

from ._euler1d import QuantityLine, SolverConfig, _1DApp


class BurgersEquation:
    """
    Burgers' equation problem.
    See the Computational Gasdynamics by Laney, Culbert B.
    """

    def __init__(self):
        self.coord = None
        self.velocity = None

        # Field data of the analytical solution.
        self.coord_field = None
        self.velocity_field = None

    def build_constant(self, x1, x2, x3, velocity12, velocity23):
        """
        Set solver initial condition.

        :param xn: the n-th point of x axis.
        :param velocitymn: the velocity between point m and n.
        """
        # Set the given value.
        self.coord = np.array([x1, x2, x3], dtype='float64')
        self.velocity = np.array([velocity12, velocity23], dtype='float64')

    def build_field(self, t):
        """
        Populate the field data using the analytical solution.

        :param t: time
        """
        self.coord_field = self.calc_coord_field(t)
        _ = (self.velocity, self.velocity)
        self.velocity_field = np.transpose(_).ravel()

    def calc_coord_field(self, t):
        """
        Calculate the wavefront location after a period of time.

        :param t: time
        :return: wavefront location (type: ndarray)
        """
        # Calculate the left side and right side wavefront location
        # separated by a jump discontinuity.
        left = self.coord[1:-1] + self.velocity[0:-1] * t
        right = self.coord[1:-1] + self.velocity[1:] * t

        # Check shock wave or expansion wave appears or not.
        shock = left > right
        expansion = left <= right

        # Calculate the wavefront location.
        avg = (left + right) / 2   # shcok wave location
        left = shock * avg + expansion * left
        right = shock * avg + expansion * right
        internal = np.transpose((left, right)).ravel()
        return np.hstack((self.coord[0], internal, self.coord[-1]))


class Burgers1DApp(_1DApp):
    """
    Main application for Burgers' equation 1D solver.
    """

    def populate_menu(self):
        """
        Set menu item for GUI.
        """
        self._add_menu_item(
            menu=self._mgr.oneMenu,
            text="Burgers equation",
            tip="One-dimensional Burgers equation problem",
            func=self.run,
        )

    def init_solver_config(self):
        """
        Set Burgers' equation solver configuration by user' input, and then
        reset the initial condition.
        """
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

    def set_plot_data(self):
        """
        Set the analytical data for plot.
            - velocity
        """
        self.plot_ana = True
        self.plot_data = []

        velocity = QuantityLine(name="velocity",
                                unit=r"$\mathrm{m}/\mathrm{s}$",
                                color='b',
                                y_upper_lim=1.2,
                                y_bottom_lim=-1.2)
        setattr(self, velocity.name, velocity)
        self.plot_data.append([self.velocity.name, True])

    def init_solver(self):
        """
        Initialize Burgers' equation solver and set up the initial condition.
        """
        self.st = BurgersEquation()
        _s = self.solver_config
        self.st.build_constant(x1=_s["x1"]["value"],
                               x2=_s["x2"]["value"],
                               x3=_s["x3"]["value"],
                               velocity12=_s["velocity12"]["value"],
                               velocity23=_s["velocity23"]["value"])
        self.st.build_field(t=0)

    def update_step(self, steps):
        """
        Update data at current step.
            - analytical data solved in python.
        """
        self.current_step += steps
        time_current = self.current_step * self.time_interval
        self.st.build_field(t=time_current)


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
