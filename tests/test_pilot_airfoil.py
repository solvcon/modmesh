# Copyright (c) 2024, Yung-Yu Chen <yyc@solvcon.net>
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


import unittest

import modmesh as mm
from modmesh.pilot import airfoil


class Naca4TC(unittest.TestCase):
    def test_npoint(self):
        def _check_size(naca4):
            points = naca4.calc_points(5)
            self.assertEqual((11, 2), points.shape)
            points = naca4.calc_points(11)
            self.assertEqual((23, 2), points.shape)

        _check_size(airfoil.Naca4(number='0012', open_trailing_edge=False,
                                  cosine_spacing=False))
        _check_size(airfoil.Naca4(number='0012', open_trailing_edge=True,
                                  cosine_spacing=False))
        _check_size(airfoil.Naca4(number='0012', open_trailing_edge=True,
                                  cosine_spacing=False))
        _check_size(airfoil.Naca4(number='0012', open_trailing_edge=True,
                                  cosine_spacing=True))


class Naca4SamplerTC(unittest.TestCase):
    def test_construction(self):
        w = mm.WorldFp64()
        naca4 = airfoil.Naca4(number='0012', open_trailing_edge=False,
                              cosine_spacing=False)
        airfoil.Naca4Sampler(w, naca4)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
