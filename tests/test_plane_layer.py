# Copyright (c) 2026, Han-Xuan Huang <c1ydehhx@gmail.com>
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
from modmesh.plot import plane_layer


class PlaneLayerTC(unittest.TestCase):
    def test_add_rect_with_string(self):
        layer = plane_layer.PlaneLayer()
        layer.add_figure("RECT N M1 70 800 180 40")

        ploys = layer.get_polys()
        self.assertEqual(ploys, [[
            [(70.0, 800.0), (250.0, 800.0)],
            [(250.0, 800.0), (250.0, 840.0)],
            [(250.0, 840.0), (70.0, 840.0)],
            [(70.0, 840.0), (70.0, 800.0)],
        ]])

    def test_add_poly_with_string(self):
        layer = plane_layer.PlaneLayer()
        layer.add_figure(
            "PGON N M1 70 720 410 720 410 920 70 920 "
            "70 880 370 880 370 760 70 760"
        )

        ploys = layer.get_polys()
        self.assertEqual(ploys, [[
            [(70.0, 720.0), (410.0, 720.0)],
            [(410.0, 720.0), (410.0, 920.0)],
            [(410.0, 920.0), (70.0, 920.0)],
            [(70.0, 920.0), (70.0, 880.0)],
            [(70.0, 880.0), (370.0, 880.0)],
            [(370.0, 880.0), (370.0, 760.0)],
            [(370.0, 760.0), (70.0, 760.0)],
            [(70.0, 760.0), (70.0, 720.0)],
        ]])

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
