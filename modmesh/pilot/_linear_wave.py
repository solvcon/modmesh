# Copyright (c) 2025, Ting-Yu Chuang <tychuang.cs10@nycu.edu.tw>
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
from ..onedim.linear_scalar import LinearScalarSolver


class LinearWave:
    """
    """

    def __init__(self):
        self.svr = None
        self.coord_field = None
        self.wave_field = None

    def build_numerical(self, xmin, xmax, ncelm, cfl):

        # Initialize the solver.
        self.svr = LinearScalarSolver(xmin, xmax, ncelm, cfl)

        # Setup the solver for time-marching.
        self.svr.setup_march()

    def build_field(self):
        self.coord_field = self.svr.xctr() / np.pi
        self.wave_field = self.svr.get_so0(0).ndarray


class LinearWave1DApp(_1DApp):
    """
    """

    def populate_menu(self):
        """
        Set menu item for GUI.
        """
        self._add_menu_item(
            menu=self._mgr.oneMenu,
            text="Linear Wave",
            tip="",
            func=self.run,
        )

    def init_solver_config(self):
        """
        """
        solver_config_data = [
            ["xmin", 0, "The most left point of x axis."],
            ["xmax", 4 * 2 * np.pi, "The most right point of x axis."],
            ["ncelm", 4 * 64, ""],
            ["cfl", 1, ""],
            ["time_interval", 10, "Qt timer interval"],
            ["max_steps", 0, "Maximum step"],
            ["profiling", False, "Turn on / off solver profiling"],
        ]
        self.solver_config = SolverConfig(solver_config_data)

    def set_plot_data(self):
        """
        Set the analytical data for plot.
        """
        self.plot_ana = True
        self.plot_data = []

        wave = QuantityLine(name="wave",
                            color='b',
                            y_upper_lim=1.2,
                            y_bottom_lim=-1.2)
        setattr(self, wave.name, wave)
        self.plot_data.append([self.wave.name, True])

    def init_solver(self):
        """
        """
        self.st = LinearWave()
        _s = self.solver_config
        self.st.build_numerical(xmin=_s["xmin"]["value"],
                                xmax=_s["xmax"]["value"],
                                ncelm=_s["ncelm"]["value"],
                                cfl=_s["cfl"]["value"])
        self.st.build_field()

    def update_step(self, steps):
        """
        Update data at current step.
        """
        self.st.svr.march_alpha2(steps=steps)
        self.st.build_field()

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
