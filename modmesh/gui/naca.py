# Copyright (c) 2021, Yung-Yu Chen <yyc@solvcon.net>
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


"""
Show NACA airfoil shape
"""

import numpy as np

from modmesh import core
from modmesh import view


def calc_naca4_points(number, npoint, open_trailing_edge=False,
                      cosine_spacing=False):
    """
    Returns points in [0 1] for the given 4 digit NACA number string.
    Reference from
    https://ntrs.nasa.gov/api/citations/19930091108/downloads/19930091108.pdf
    and https://github.com/dgorissen/naca
    """

    camber = float(number[0]) / 100.0
    pos = float(number[1]) / 10.0
    thick = float(number[2:]) / 100.0

    a0 = +0.2969
    a1 = -0.1260
    a2 = -0.3516
    a3 = +0.2843
    a4 = -0.1015 if open_trailing_edge else -0.1036

    if cosine_spacing:
        xc = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, npoint + 1)))
    else:
        xc = np.linspace(0.0, 1.0, npoint + 1)

    xh = np.sqrt(xc)
    x2 = np.power(xc, 2)
    x3 = np.power(xc, 3)
    x4 = np.power(xc, 4)
    yt = 5.0 * thick * (a0 * xh + a1 * xc + a2 * x2 + a3 * x3 + a4 * x4)

    if pos == 0:
        xu = xc
        yu = yt
        xl = xc
        yl = -yt
    else:
        xc1 = xc[xc <= pos]
        yc1 = camber / np.power(pos, 2) * xc1 * (2 * pos - xc1)
        dyc1_dx = camber / np.power(pos, 2) * (2 * pos - 2 * xc1)

        xc2 = xc[xc > pos]
        yc2 = camber / np.power(1 - pos, 2) * (1 - 2 * pos + xc2) * (1 - xc2)
        dyc2_dx = camber / np.power(1 - pos, 2) * (2 * pos - 2 * xc2)

        yc = np.concatenate([yc1, yc2])
        dyc_dx = np.concatenate([dyc1_dx, dyc2_dx])
        theta = np.arctan(dyc_dx)

        xu = xc - yt * np.sin(theta)
        yu = yc + yt * np.cos(theta)
        xl = xc + yt * np.sin(theta)
        yl = yc - yt * np.cos(theta)

    xr = np.concatenate([xu[::-1], xl[1:]])
    yr = np.concatenate([yu[::-1], yl[1:]])
    ret = np.vstack([xr, yr]).T.copy()
    return ret


def draw_naca4(world, number, npoint, fac, off_x, off_y, **kw):
    crds = calc_naca4_points(number=number, npoint=npoint, **kw)
    crds *= fac  # scaling factor
    crds[:, 0] += off_x  # offset in x
    crds[:, 1] += off_y  # offset in y
    for it in range(crds.shape[0] - 1):
        e = world.add_edge(crds[it, 0], crds[it, 1], 0,
                           crds[it + 1, 0], crds[it + 1, 1], 0)
        print(f"{it}: {e}")
    print("nedge:", world.nedge)
    return crds


def runmain():
    """
    A simple example for drawing a couple of cubic Bezier curves.
    """
    w = core.WorldFp64()
    draw_naca4(w, number='0012', npoint=101, fac=5.0, off_x=0.0, off_y=2.0,
               open_trailing_edge=False, cosine_spacing=True)
    wid = view.mgr.add3DWidget()
    wid.updateWorld(w)
    wid.showMark()

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
