# Copyright (c) 2025, Jenny Yen <jenny35006@gmail.com>
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

from modmesh.testing import TestBase as ModMeshTB
import modmesh.plot.svg as svg


class SvgParserTB(ModMeshTB, unittest.TestCase):

    def assert_allclose(self, *args, **kw):
        if 'rtol' not in kw:
            kw['rtol'] = 1.e-15
        return super().assert_allclose(*args, **kw)


class SvgParserGeneralTC(SvgParserTB):

    def test_single_closed_path(self):
        d_attr = "M10 10 L60 10 L60 60 L10 60 Z"
        fill_attr = "none"
        path_element = svg.EPath(d_attr=d_attr, fill_attr=fill_attr)
        self.assertEqual(len(path_element.get_cmds()), 5)
        self.assertEqual(
            path_element.get_cmds(),
            [('M', [10.0, 10.0]),
             ('L', [60.0, 10.0]),
             ('L', [60.0, 60.0]),
             ('L', [10.0, 60.0]),
             ('Z', [])])

    def test_multiple_closed_paths(self):
        d_attr = "M10 10 L60 10 L60 60 L10 60 Z M100 10 L150 60 L100 60 Z"
        fill_attr = "none"
        path_element = svg.EPath(d_attr=d_attr, fill_attr=fill_attr)
        self.assertEqual(len(path_element.get_cmds()), 9)
        self.assertEqual(
            path_element.get_cmds(),
            [('M', [10.0, 10.0]),
             ('L', [60.0, 10.0]),
             ('L', [60.0, 60.0]),
             ('L', [10.0, 60.0]),
             ('Z', []),
             ('M', [100.0, 10.0]),
             ('L', [150.0, 60.0]),
             ('L', [100.0, 60.0]),
             ('Z', [])])


class SvgMoveToCommandTC(SvgParserTB):
    """
    Test 'M' and 'm' commands in SVG path 'd' attribute.
    See more: https://developer.mozilla.org/en-US/docs/Web/SVG/Reference/Attribute/d#moveto_path_commands
    """  # noqa: E501

    def test_moveto_absolute(self):
        d_attr = "M 10 10 h 10"
        fill_attr = "none"
        path_element = svg.EPath(d_attr=d_attr, fill_attr=fill_attr)
        sp2d = path_element.get_closed_paths()[0]
        cp2d = path_element.get_closed_paths()[1]
        self.assertEqual(sp2d.ndim, 2)
        self.assertEqual(cp2d.ndim, 2)
        self.assertEqual(len(sp2d), 1)
        self.assertEqual(len(cp2d), 0)
        self.assertEqual(list(sp2d.x0), [10.0])
        self.assertEqual(list(sp2d.y0), [10.0])
        self.assertEqual(list(sp2d.x1), [20.0])
        self.assertEqual(list(sp2d.y1), [10.0])

    def test_moveto_relative(self):
        d_attr = "M 10 10 h 10 m 0 10 h 10"
        fill_attr = "none"
        path_element = svg.EPath(d_attr=d_attr, fill_attr=fill_attr)
        sp2d = path_element.get_closed_paths()[0]
        cp2d = path_element.get_closed_paths()[1]
        self.assertEqual(sp2d.ndim, 2)
        self.assertEqual(cp2d.ndim, 2)
        self.assertEqual(len(sp2d), 2)
        self.assertEqual(len(cp2d), 0)
        self.assertEqual(list(sp2d.x0), [10.0, 20.0])
        self.assertEqual(list(sp2d.y0), [10.0, 20.0])
        self.assertEqual(list(sp2d.x1), [20.0, 30.0])
        self.assertEqual(list(sp2d.y1), [10.0, 20.0])

    def test_moveto_absolute_and_relative(self):
        d_attr = "M 10 10 h 10 m  0 10 h 10"
        fill_attr = "none"
        path_element = svg.EPath(d_attr=d_attr, fill_attr=fill_attr)
        sp2d = path_element.get_closed_paths()[0]
        cp2d = path_element.get_closed_paths()[1]
        self.assertEqual(sp2d.ndim, 2)
        self.assertEqual(cp2d.ndim, 2)
        self.assertEqual(len(sp2d), 2)
        self.assertEqual(len(cp2d), 0)
        self.assertEqual(list(sp2d.x0), [10.0, 20.0])
        self.assertEqual(list(sp2d.y0), [10.0, 20.0])
        self.assertEqual(list(sp2d.x1), [20.0, 30.0])
        self.assertEqual(list(sp2d.y1), [10.0, 20.0])

    def test_moveto_implicit_lineto(self):
        d_attr = "M 10 10 20 10 m 0 10 10 0"
        fill_attr = "none"
        path_element = svg.EPath(d_attr=d_attr, fill_attr=fill_attr)
        sp2d = path_element.get_closed_paths()[0]
        cp2d = path_element.get_closed_paths()[1]
        self.assertEqual(sp2d.ndim, 2)
        self.assertEqual(cp2d.ndim, 2)
        self.assertEqual(len(sp2d), 2)
        self.assertEqual(len(cp2d), 0)
        self.assertEqual(list(sp2d.x0), [10.0, 20.0])
        self.assertEqual(list(sp2d.y0), [10.0, 20.0])
        self.assertEqual(list(sp2d.x1), [20.0, 30.0])
        self.assertEqual(list(sp2d.y1), [10.0, 20.0])


class SvgLineToCommandTC(SvgParserTB):
    """
    Test 'L', 'l', 'H', 'h', 'V', and 'v' commands in SVG path 'd' attribute.
    See more: https://developer.mozilla.org/en-US/docs/Web/SVG/Reference/Attribute/d#lineto_path_commands
    """  # noqa: E501

    def test_lineto_absolute(self):
        d_attr = "M 10 10 L 30 20 L 50 30"
        fill_attr = "none"
        path_element = svg.EPath(d_attr=d_attr, fill_attr=fill_attr)
        sp2d = path_element.get_closed_paths()[0]
        cp2d = path_element.get_closed_paths()[1]
        self.assertEqual(sp2d.ndim, 2)
        self.assertEqual(cp2d.ndim, 2)
        self.assertEqual(len(sp2d), 2)
        self.assertEqual(len(cp2d), 0)
        self.assertEqual(list(sp2d.x0), [10.0, 30.0])
        self.assertEqual(list(sp2d.y0), [10.0, 20.0])
        self.assertEqual(list(sp2d.x1), [30.0, 50.0])
        self.assertEqual(list(sp2d.y1), [20.0, 30.0])

    def test_lineto_relative(self):
        d_attr = "M 10 10 l 20 10 l 20 10"
        fill_attr = "none"
        path_element = svg.EPath(d_attr=d_attr, fill_attr=fill_attr)
        sp2d = path_element.get_closed_paths()[0]
        cp2d = path_element.get_closed_paths()[1]
        self.assertEqual(sp2d.ndim, 2)
        self.assertEqual(cp2d.ndim, 2)
        self.assertEqual(len(sp2d), 2)
        self.assertEqual(len(cp2d), 0)
        self.assertEqual(list(sp2d.x0), [10.0, 30.0])
        self.assertEqual(list(sp2d.y0), [10.0, 20.0])
        self.assertEqual(list(sp2d.x1), [30.0, 50.0])
        self.assertEqual(list(sp2d.y1), [20.0, 30.0])

    def test_horizontal_lineto_absolute(self):
        d_attr = "M 10 10 H 30 H 50"
        fill_attr = "none"
        path_element = svg.EPath(d_attr=d_attr, fill_attr=fill_attr)
        sp2d = path_element.get_closed_paths()[0]
        cp2d = path_element.get_closed_paths()[1]
        self.assertEqual(sp2d.ndim, 2)
        self.assertEqual(cp2d.ndim, 2)
        self.assertEqual(len(sp2d), 2)
        self.assertEqual(len(cp2d), 0)
        self.assertEqual(list(sp2d.x0), [10.0, 30.0])
        self.assertEqual(list(sp2d.y0), [10.0, 10.0])
        self.assertEqual(list(sp2d.x1), [30.0, 50.0])
        self.assertEqual(list(sp2d.y1), [10.0, 10.0])

    def test_horizontal_lineto_relative(self):
        d_attr = "M 10 10 h 20 h 20"
        fill_attr = "none"
        path_element = svg.EPath(d_attr=d_attr, fill_attr=fill_attr)
        sp2d = path_element.get_closed_paths()[0]
        cp2d = path_element.get_closed_paths()[1]
        self.assertEqual(sp2d.ndim, 2)
        self.assertEqual(cp2d.ndim, 2)
        self.assertEqual(len(sp2d), 2)
        self.assertEqual(len(cp2d), 0)
        self.assertEqual(list(sp2d.x0), [10.0, 30.0])
        self.assertEqual(list(sp2d.y0), [10.0, 10.0])
        self.assertEqual(list(sp2d.x1), [30.0, 50.0])
        self.assertEqual(list(sp2d.y1), [10.0, 10.0])

    def test_vertical_lineto_absolute(self):
        d_attr = "M 10 10 V 30 V 50"
        fill_attr = "none"
        path_element = svg.EPath(d_attr=d_attr, fill_attr=fill_attr)
        sp2d = path_element.get_closed_paths()[0]
        cp2d = path_element.get_closed_paths()[1]
        self.assertEqual(sp2d.ndim, 2)
        self.assertEqual(cp2d.ndim, 2)
        self.assertEqual(len(sp2d), 2)
        self.assertEqual(len(cp2d), 0)
        self.assertEqual(list(sp2d.x0), [10.0, 10.0])
        self.assertEqual(list(sp2d.y0), [10.0, 30.0])
        self.assertEqual(list(sp2d.x1), [10.0, 10.0])
        self.assertEqual(list(sp2d.y1), [30.0, 50.0])

    def test_vertical_lineto_relative(self):
        d_attr = "M 10 10 v 20 v 20"
        fill_attr = "none"
        path_element = svg.EPath(d_attr=d_attr, fill_attr=fill_attr)
        sp2d = path_element.get_closed_paths()[0]
        cp2d = path_element.get_closed_paths()[1]
        self.assertEqual(sp2d.ndim, 2)
        self.assertEqual(cp2d.ndim, 2)
        self.assertEqual(len(sp2d), 2)
        self.assertEqual(len(cp2d), 0)
        self.assertEqual(list(sp2d.x0), [10.0, 10.0])
        self.assertEqual(list(sp2d.y0), [10.0, 30.0])
        self.assertEqual(list(sp2d.x1), [10.0, 10.0])
        self.assertEqual(list(sp2d.y1), [30.0, 50.0])

    def test_lineto_all_variants(self):
        d_attr = "M 10 10 L 20 10 l 10 0 V 20 v 10 H 20 h -10"
        fill_attr = "none"
        path_element = svg.EPath(d_attr=d_attr, fill_attr=fill_attr)
        sp2d = path_element.get_closed_paths()[0]
        cp2d = path_element.get_closed_paths()[1]
        self.assertEqual(sp2d.ndim, 2)
        self.assertEqual(cp2d.ndim, 2)
        self.assertEqual(len(sp2d), 6)
        self.assertEqual(len(cp2d), 0)
        self.assertEqual(list(sp2d.x0), [10.0, 20.0, 30.0, 30.0, 30.0, 20.0])
        self.assertEqual(list(sp2d.y0), [10.0, 10.0, 10.0, 20.0, 30.0, 30.0])
        self.assertEqual(list(sp2d.x1), [20.0, 30.0, 30.0, 30.0, 20.0, 10.0])
        self.assertEqual(list(sp2d.y1), [10.0, 10.0, 20.0, 30.0, 30.0, 30.0])

    def test_lineto_multiple_coordinates(self):
        d_attr = "M 10 10 L 20 10 30 10 l 10 0 10 0 V 20 30 v 10 10 H 40 30 h -10 -10"  # noqa: E501
        fill_attr = "none"
        path_element = svg.EPath(d_attr=d_attr, fill_attr=fill_attr)
        sp2d = path_element.get_closed_paths()[0]
        cp2d = path_element.get_closed_paths()[1]
        self.assertEqual(sp2d.ndim, 2)
        self.assertEqual(cp2d.ndim, 2)
        self.assertEqual(len(sp2d), 12)
        self.assertEqual(len(cp2d), 0)
        self.assertEqual(list(sp2d.x0), [10.0, 20.0, 30.0, 40.0, 50.0, 50.0,
                                         50.0, 50.0, 50.0, 40.0, 30.0, 20.0])
        self.assertEqual(list(sp2d.y0), [10.0, 10.0, 10.0, 10.0, 10.0, 20.0,
                                         30.0, 40.0, 50.0, 50.0, 50.0, 50.0])
        self.assertEqual(list(sp2d.x1), [20.0, 30.0, 40.0, 50.0, 50.0, 50.0,
                                         50.0, 50.0, 40.0, 30.0, 20.0, 10.0])
        self.assertEqual(list(sp2d.y1), [10.0, 10.0, 10.0, 10.0, 20.0, 30.0,
                                         40.0, 50.0, 50.0, 50.0, 50.0, 50.0])


class SvgCubicBezierCurveCommandTC(SvgParserTB):
    """
    Test 'C', 'c', 'S', and 's' commands in SVG path 'd' attribute.
    See more: https://developer.mozilla.org/en-US/docs/Web/SVG/Reference/Attribute/d#cubic_b%C3%A9zier_curve
    """  # noqa: E501

    def test_absolute(self):
        d_attr = "M 10 90 C 30 90 25 10 50 10"
        fill_attr = "none"
        path_element = svg.EPath(d_attr=d_attr, fill_attr=fill_attr)
        sp2d = path_element.get_closed_paths()[0]
        cp2d = path_element.get_closed_paths()[1]
        self.assertEqual(sp2d.ndim, 2)
        self.assertEqual(cp2d.ndim, 2)
        self.assertEqual(len(sp2d), 0)
        self.assertEqual(len(cp2d), 1)
        self.assertEqual(list(cp2d[0][0]), [10, 90, 0])
        self.assertEqual(list(cp2d[0][1]), [30, 90, 0])
        self.assertEqual(list(cp2d[0][2]), [25, 10, 0])
        self.assertEqual(list(cp2d[0][3]), [50, 10, 0])

    def test_relative(self):
        d_attr = "M 10 90 c 20 0 15 -80 40 -80"
        fill_attr = "none"
        path_element = svg.EPath(d_attr=d_attr, fill_attr=fill_attr)
        sp2d = path_element.get_closed_paths()[0]
        cp2d = path_element.get_closed_paths()[1]
        self.assertEqual(sp2d.ndim, 2)
        self.assertEqual(cp2d.ndim, 2)
        self.assertEqual(len(sp2d), 0)
        self.assertEqual(len(cp2d), 1)
        self.assertEqual(list(cp2d[0][0]), [10, 90, 0])
        self.assertEqual(list(cp2d[0][1]), [30, 90, 0])
        self.assertEqual(list(cp2d[0][2]), [25, 10, 0])
        self.assertEqual(list(cp2d[0][3]), [50, 10, 0])

    def test_smooth_cubic_bezier_absolute(self):
        d_attr = "M 10 90 C 30 90 25 10 50 10 S 70 90 90 90"
        fill_attr = "none"
        path_element = svg.EPath(d_attr=d_attr, fill_attr=fill_attr)
        sp2d = path_element.get_closed_paths()[0]
        cp2d = path_element.get_closed_paths()[1]
        self.assertEqual(sp2d.ndim, 2)
        self.assertEqual(cp2d.ndim, 2)
        self.assertEqual(len(sp2d), 0)
        self.assertEqual(len(cp2d), 2)
        self.assertEqual(list(cp2d[1][0]), [50, 10, 0])
        self.assertEqual(list(cp2d[1][1]), [75, 10, 0])
        self.assertEqual(list(cp2d[1][2]), [70, 90, 0])
        self.assertEqual(list(cp2d[1][3]), [90, 90, 0])

    def test_smooth_cubic_bezier_relative(self):
        d_attr = "M 10 90 C 30 90 25 10 50 10 s 20 80 40 80"
        fill_attr = "none"
        path_element = svg.EPath(d_attr=d_attr, fill_attr=fill_attr)
        sp2d = path_element.get_closed_paths()[0]
        cp2d = path_element.get_closed_paths()[1]
        self.assertEqual(sp2d.ndim, 2)
        self.assertEqual(cp2d.ndim, 2)
        self.assertEqual(len(sp2d), 0)
        self.assertEqual(len(cp2d), 2)
        self.assertEqual(list(cp2d[1][0]), [50, 10, 0])
        self.assertEqual(list(cp2d[1][1]), [75, 10, 0])
        self.assertEqual(list(cp2d[1][2]), [70, 90, 0])
        self.assertEqual(list(cp2d[1][3]), [90, 90, 0])

    def test_cubic_bezier_basic(self):
        d_attr = "M 10 90 C 30 90 25 10 50 10 S 70 90 90 90 c 20 0 15 -80 40 -80 s 20 80 40 80"  # noqa: E501
        fill_attr = "none"
        path_element = svg.EPath(d_attr=d_attr, fill_attr=fill_attr)
        sp2d = path_element.get_closed_paths()[0]
        cp2d = path_element.get_closed_paths()[1]
        self.assertEqual(sp2d.ndim, 2)
        self.assertEqual(cp2d.ndim, 2)
        self.assertEqual(len(sp2d), 0)
        self.assertEqual(len(cp2d), 4)
        self.assertEqual(list(cp2d[0][0]), [10, 90, 0])
        self.assertEqual(list(cp2d[0][1]), [30, 90, 0])
        self.assertEqual(list(cp2d[0][2]), [25, 10, 0])
        self.assertEqual(list(cp2d[0][3]), [50, 10, 0])

        self.assertEqual(list(cp2d[1][0]), [50, 10, 0])
        self.assertEqual(list(cp2d[1][1]), [75, 10, 0])
        self.assertEqual(list(cp2d[1][2]), [70, 90, 0])
        self.assertEqual(list(cp2d[1][3]), [90, 90, 0])

        self.assertEqual(list(cp2d[2][0]), [90, 90, 0])
        self.assertEqual(list(cp2d[2][1]), [110, 90, 0])
        self.assertEqual(list(cp2d[2][2]), [105, 10, 0])
        self.assertEqual(list(cp2d[2][3]), [130, 10, 0])

        self.assertEqual(list(cp2d[3][0]), [130, 10, 0])
        self.assertEqual(list(cp2d[3][1]), [155, 10, 0])
        self.assertEqual(list(cp2d[3][2]), [150, 90, 0])
        self.assertEqual(list(cp2d[3][3]), [170, 90, 0])

    def test_cubic_bezier_implicit_curves(self):
        d_attr = "M 10 90 C 30 90 25 10 50 10 75 10 70 90 90 90 S 105 10 130 10 150 90 170 90 c 20 0 15 -80 40 -80 25 0 20 80 40 80 s 15 -80 40 -80 20 80 40 80"  # noqa: E501
        fill_attr = "none"
        path_element = svg.EPath(d_attr=d_attr, fill_attr=fill_attr)
        sp2d = path_element.get_closed_paths()[0]
        cp2d = path_element.get_closed_paths()[1]
        self.assertEqual(sp2d.ndim, 2)
        self.assertEqual(cp2d.ndim, 2)
        self.assertEqual(len(sp2d), 0)
        self.assertEqual(len(cp2d), 8)

        # two "C" curves
        self.assertEqual(list(cp2d[0][0]), [10, 90, 0])
        self.assertEqual(list(cp2d[0][1]), [30, 90, 0])
        self.assertEqual(list(cp2d[0][2]), [25, 10, 0])
        self.assertEqual(list(cp2d[0][3]), [50, 10, 0])

        self.assertEqual(list(cp2d[1][0]), [50, 10, 0])
        self.assertEqual(list(cp2d[1][1]), [75, 10, 0])
        self.assertEqual(list(cp2d[1][2]), [70, 90, 0])
        self.assertEqual(list(cp2d[1][3]), [90, 90, 0])

        # two "S" curves
        self.assertEqual(list(cp2d[2][0]), [90, 90, 0])
        self.assertEqual(list(cp2d[2][1]), [110, 90, 0])
        self.assertEqual(list(cp2d[2][2]), [105, 10, 0])
        self.assertEqual(list(cp2d[2][3]), [130, 10, 0])

        self.assertEqual(list(cp2d[3][0]), [130, 10, 0])
        self.assertEqual(list(cp2d[3][1]), [155, 10, 0])
        self.assertEqual(list(cp2d[3][2]), [150, 90, 0])
        self.assertEqual(list(cp2d[3][3]), [170, 90, 0])

        # two "c" curves
        self.assertEqual(list(cp2d[4][0]), [170, 90, 0])
        self.assertEqual(list(cp2d[4][1]), [190, 90, 0])
        self.assertEqual(list(cp2d[4][2]), [185, 10, 0])
        self.assertEqual(list(cp2d[4][3]), [210, 10, 0])

        self.assertEqual(list(cp2d[5][0]), [210, 10, 0])
        self.assertEqual(list(cp2d[5][1]), [235, 10, 0])
        self.assertEqual(list(cp2d[5][2]), [230, 90, 0])
        self.assertEqual(list(cp2d[5][3]), [250, 90, 0])

        # two "s" curves
        self.assertEqual(list(cp2d[6][0]), [250, 90, 0])
        self.assertEqual(list(cp2d[6][1]), [270, 90, 0])
        self.assertEqual(list(cp2d[6][2]), [265, 10, 0])
        self.assertEqual(list(cp2d[6][3]), [290, 10, 0])

        self.assertEqual(list(cp2d[7][0]), [290, 10, 0])
        self.assertEqual(list(cp2d[7][1]), [315, 10, 0])
        self.assertEqual(list(cp2d[7][2]), [310, 90, 0])
        self.assertEqual(list(cp2d[7][3]), [330, 90, 0])

    def test_cubic_bezier_point_continuity(self):
        d_attr = "M 10 90 C 30 90 25 10 50 10 S 70 90 90 90 c 20 0 15 -80 40 -80 s 20 80 40 80"  # noqa: E501
        fill_attr = "none"
        path_element = svg.EPath(d_attr=d_attr, fill_attr=fill_attr)
        sp2d = path_element.get_closed_paths()[0]
        cp2d = path_element.get_closed_paths()[1]
        self.assertEqual(sp2d.ndim, 2)
        self.assertEqual(cp2d.ndim, 2)
        self.assertEqual(len(sp2d), 0)
        self.assertEqual(len(cp2d), 4)

        for i in range(len(cp2d) - 1):
            self.assert_allclose(cp2d[i][3][0], cp2d[i + 1][0][0])
            self.assert_allclose(cp2d[i][3][1], cp2d[i + 1][0][1])
            self.assert_allclose(cp2d[i][3][2], cp2d[i + 1][0][2])


class SvgQuadraticBezierCurveCommandTC(SvgParserTB):
    """
    Test 'Q', 'q', 'T', and 't' commands in SVG path 'd' attribute.
    See more: https://developer.mozilla.org/en-US/docs/Web/SVG/Reference/Attribute/d#quadratic_b%C3%A9zier_curve
    """  # noqa: E501

    def test_quadratic_bezier_absolute(self):
        d_attr = "M 10 50 Q 25 25 40 50"
        fill_attr = "none"
        path_element = svg.EPath(d_attr=d_attr, fill_attr=fill_attr)
        sp2d = path_element.get_closed_paths()[0]
        cp2d = path_element.get_closed_paths()[1]
        self.assertEqual(sp2d.ndim, 2)
        self.assertEqual(cp2d.ndim, 2)
        self.assertEqual(len(sp2d), 0)
        self.assertEqual(len(cp2d), 1)
        self.assertEqual(list(cp2d[0][0]), [10, 50, 0])
        self.assertEqual(list(cp2d[0][1]), [25, 25, 0])
        self.assertEqual(list(cp2d[0][2]), [40, 50, 0])
        self.assertEqual(list(cp2d[0][3]), [40, 50, 0])

    def test_quadratic_bezier_relative(self):
        d_attr = "M 10 50 q 15 -25 30 0"
        fill_attr = "none"
        path_element = svg.EPath(d_attr=d_attr, fill_attr=fill_attr)
        sp2d = path_element.get_closed_paths()[0]
        cp2d = path_element.get_closed_paths()[1]
        self.assertEqual(sp2d.ndim, 2)
        self.assertEqual(cp2d.ndim, 2)
        self.assertEqual(len(sp2d), 0)
        self.assertEqual(len(cp2d), 1)
        self.assertEqual(list(cp2d[0][0]), [10, 50, 0])
        self.assertEqual(list(cp2d[0][1]), [25, 25, 0])
        self.assertEqual(list(cp2d[0][2]), [40, 50, 0])
        self.assertEqual(list(cp2d[0][3]), [40, 50, 0])

    def test_smooth_quadratic_bezier_absolute(self):
        d_attr = "M 10 50 Q 25 25 40 50 T 70 50"
        fill_attr = "none"
        path_element = svg.EPath(d_attr=d_attr, fill_attr=fill_attr)
        sp2d = path_element.get_closed_paths()[0]
        cp2d = path_element.get_closed_paths()[1]
        self.assertEqual(sp2d.ndim, 2)
        self.assertEqual(cp2d.ndim, 2)
        self.assertEqual(len(sp2d), 0)
        self.assertEqual(len(cp2d), 2)
        self.assertEqual(list(cp2d[1][0]), [40, 50, 0])
        self.assertEqual(list(cp2d[1][1]), [55, 75, 0])
        self.assertEqual(list(cp2d[1][2]), [70, 50, 0])
        self.assertEqual(list(cp2d[1][3]), [70, 50, 0])

    def test_smooth_quadratic_bezier_relative(self):
        d_attr = "M 10 50 Q 25 25 40 50 t 30 0"
        fill_attr = "none"
        path_element = svg.EPath(d_attr=d_attr, fill_attr=fill_attr)
        sp2d = path_element.get_closed_paths()[0]
        cp2d = path_element.get_closed_paths()[1]
        self.assertEqual(sp2d.ndim, 2)
        self.assertEqual(cp2d.ndim, 2)
        self.assertEqual(len(sp2d), 0)
        self.assertEqual(len(cp2d), 2)
        self.assertEqual(list(cp2d[1][0]), [40, 50, 0])
        self.assertEqual(list(cp2d[1][1]), [55, 75, 0])
        self.assertEqual(list(cp2d[1][2]), [70, 50, 0])
        self.assertEqual(list(cp2d[1][3]), [70, 50, 0])

    def test_quadratic_bezier_basic(self):
        d_attr = "M 10 50 Q 25 25 40 50 q 15 25 30 0 T 100 50 t 30 0"
        fill_attr = "none"
        path_element = svg.EPath(d_attr=d_attr, fill_attr=fill_attr)
        sp2d = path_element.get_closed_paths()[0]
        cp2d = path_element.get_closed_paths()[1]
        self.assertEqual(sp2d.ndim, 2)
        self.assertEqual(cp2d.ndim, 2)
        self.assertEqual(len(sp2d), 0)
        self.assertEqual(len(cp2d), 4)

        self.assertEqual(list(cp2d[0][0]), [10, 50, 0])
        self.assertEqual(list(cp2d[0][1]), [25, 25, 0])
        self.assertEqual(list(cp2d[0][2]), [40, 50, 0])
        self.assertEqual(list(cp2d[0][3]), [40, 50, 0])

        self.assertEqual(list(cp2d[1][0]), [40, 50, 0])
        self.assertEqual(list(cp2d[1][1]), [55, 75, 0])
        self.assertEqual(list(cp2d[1][2]), [70, 50, 0])
        self.assertEqual(list(cp2d[1][3]), [70, 50, 0])

        self.assertEqual(list(cp2d[2][0]), [70, 50, 0])
        self.assertEqual(list(cp2d[2][1]), [85, 25, 0])
        self.assertEqual(list(cp2d[2][2]), [100, 50, 0])
        self.assertEqual(list(cp2d[2][3]), [100, 50, 0])

        self.assertEqual(list(cp2d[3][0]), [100, 50, 0])
        self.assertEqual(list(cp2d[3][1]), [115, 75, 0])
        self.assertEqual(list(cp2d[3][2]), [130, 50, 0])
        self.assertEqual(list(cp2d[3][3]), [130, 50, 0])

    def test_quadratic_bezier_implicit_curves(self):
        d_attr = "M 10 50 Q 25 25 40 50 55 75 70 50 q 15 -25 30 0 15 25 30 0 T 160 50 190 50 t 30 0 30 0"  # noqa: E501
        fill_attr = "none"
        path_element = svg.EPath(d_attr=d_attr, fill_attr=fill_attr)
        sp2d = path_element.get_closed_paths()[0]
        cp2d = path_element.get_closed_paths()[1]
        self.assertEqual(sp2d.ndim, 2)
        self.assertEqual(cp2d.ndim, 2)
        self.assertEqual(len(sp2d), 0)
        self.assertEqual(len(cp2d), 8)

        # two 'Q' curves
        self.assertEqual(list(cp2d[0][0]), [10, 50, 0])
        self.assertEqual(list(cp2d[0][1]), [25, 25, 0])
        self.assertEqual(list(cp2d[0][2]), [40, 50, 0])
        self.assertEqual(list(cp2d[0][3]), [40, 50, 0])

        self.assertEqual(list(cp2d[1][0]), [40, 50, 0])
        self.assertEqual(list(cp2d[1][1]), [55, 75, 0])
        self.assertEqual(list(cp2d[1][2]), [70, 50, 0])
        self.assertEqual(list(cp2d[1][3]), [70, 50, 0])

        # two 'q' curves
        self.assertEqual(list(cp2d[2][0]), [70, 50, 0])
        self.assertEqual(list(cp2d[2][1]), [85, 25, 0])
        self.assertEqual(list(cp2d[2][2]), [100, 50, 0])
        self.assertEqual(list(cp2d[2][3]), [100, 50, 0])

        self.assertEqual(list(cp2d[3][0]), [100, 50, 0])
        self.assertEqual(list(cp2d[3][1]), [115, 75, 0])
        self.assertEqual(list(cp2d[3][2]), [130, 50, 0])
        self.assertEqual(list(cp2d[3][3]), [130, 50, 0])

        # two 'T' curves
        self.assertEqual(list(cp2d[4][0]), [130, 50, 0])
        self.assertEqual(list(cp2d[4][1]), [145, 25, 0])
        self.assertEqual(list(cp2d[4][2]), [160, 50, 0])
        self.assertEqual(list(cp2d[4][3]), [160, 50, 0])

        self.assertEqual(list(cp2d[5][0]), [160, 50, 0])
        self.assertEqual(list(cp2d[5][1]), [175, 75, 0])
        self.assertEqual(list(cp2d[5][2]), [190, 50, 0])
        self.assertEqual(list(cp2d[5][3]), [190, 50, 0])

        # two 't' curves
        self.assertEqual(list(cp2d[6][0]), [190, 50, 0])
        self.assertEqual(list(cp2d[6][1]), [205, 25, 0])
        self.assertEqual(list(cp2d[6][2]), [220, 50, 0])
        self.assertEqual(list(cp2d[6][3]), [220, 50, 0])

        self.assertEqual(list(cp2d[7][0]), [220, 50, 0])
        self.assertEqual(list(cp2d[7][1]), [235, 75, 0])
        self.assertEqual(list(cp2d[7][2]), [250, 50, 0])
        self.assertEqual(list(cp2d[7][3]), [250, 50, 0])

    def test_quadratic_bezier_point_continuity(self):
        d_attr = "M 10 50 Q 25 25 40 50 q 15 25 30 0 T 100 50 t 30 0"
        fill_attr = "none"
        path_element = svg.EPath(d_attr=d_attr, fill_attr=fill_attr)
        sp2d = path_element.get_closed_paths()[0]
        cp2d = path_element.get_closed_paths()[1]
        self.assertEqual(sp2d.ndim, 2)
        self.assertEqual(cp2d.ndim, 2)
        self.assertEqual(len(sp2d), 0)
        self.assertEqual(len(cp2d), 4)

        for i in range(len(cp2d) - 1):
            self.assert_allclose(cp2d[i][2][0], cp2d[i + 1][0][0])
            self.assert_allclose(cp2d[i][2][1], cp2d[i + 1][0][1])
            self.assert_allclose(cp2d[i][2][2], cp2d[i + 1][0][2])


class SvgEllipticalArcCurveCommandTC(SvgParserTB):
    """
    Test 'A' and 'a' commands in SVG path 'd' attribute.
    See more: https://developer.mozilla.org/en-US/docs/Web/SVG/Attribute/d#elliptical_arc_curve
    """  # noqa: E501

    def test_arc_basic(self):
        d_attr = "M 6 10 A 6 4 10 1 0 14 10 A 6 4 10 0 1 20 10"
        fill_attr = "none"
        path_element = svg.EPath(d_attr=d_attr, fill_attr=fill_attr)
        sp2d = path_element.get_closed_paths()[0]
        cp2d = path_element.get_closed_paths()[1]
        self.assertEqual(sp2d.ndim, 2)
        self.assertEqual(cp2d.ndim, 2)
        self.assertEqual(len(sp2d), 39 * 2)
        self.assertEqual(len(cp2d), 0)

        self.assert_allclose(sp2d.x0_at(0), 6)
        self.assert_allclose(sp2d.y0_at(0), 10)
        self.assert_allclose(sp2d.x1_at(77), 20)
        self.assert_allclose(sp2d.y1_at(77), 10)

        self.assert_allclose(sp2d.x1_at(38), 14)
        self.assert_allclose(sp2d.y1_at(38), 10)

        for i in range(len(sp2d) - 1):
            self.assert_allclose(sp2d.x1_at(i), sp2d.x0_at(i + 1))
            self.assert_allclose(sp2d.y1_at(i), sp2d.y0_at(i + 1))

    def test_arc_implicit_curves(self):
        d_attr = "M 6 10 A 6 4 10 1 0 14 10 6 4 10 0 1 20 10"
        fill_attr = "none"
        path_element = svg.EPath(d_attr=d_attr, fill_attr=fill_attr)
        sp2d = path_element.get_closed_paths()[0]
        cp2d = path_element.get_closed_paths()[1]
        self.assertEqual(sp2d.ndim, 2)
        self.assertEqual(cp2d.ndim, 2)
        self.assertEqual(len(sp2d), 39 * 2)
        self.assertEqual(len(cp2d), 0)

        self.assert_allclose(sp2d.x0_at(0), 6)
        self.assert_allclose(sp2d.y0_at(0), 10)
        self.assert_allclose(sp2d.x1_at(77), 20)
        self.assert_allclose(sp2d.y1_at(77), 10)

        self.assert_allclose(sp2d.x1_at(38), 14)
        self.assert_allclose(sp2d.y1_at(38), 10)

        for i in range(len(sp2d) - 1):
            self.assert_allclose(sp2d.x1_at(i), sp2d.x0_at(i + 1))
            self.assert_allclose(sp2d.y1_at(i), sp2d.y0_at(i + 1))

    def test_arc_relative(self):
        d_attr = "M 6 10 a 6 4 10 1 0 8 0 a 6 4 10 0 1 6 0"
        fill_attr = "none"
        path_element = svg.EPath(d_attr=d_attr, fill_attr=fill_attr)
        sp2d = path_element.get_closed_paths()[0]
        cp2d = path_element.get_closed_paths()[1]
        self.assertEqual(sp2d.ndim, 2)
        self.assertEqual(cp2d.ndim, 2)
        self.assertEqual(len(sp2d), 39 * 2)
        self.assertEqual(len(cp2d), 0)

        self.assert_allclose(sp2d.x0_at(0), 6)
        self.assert_allclose(sp2d.y0_at(0), 10)
        self.assert_allclose(sp2d.x1_at(77), 20)
        self.assert_allclose(sp2d.y1_at(77), 10)

        self.assert_allclose(sp2d.x1_at(38), 14)
        self.assert_allclose(sp2d.y1_at(38), 10)

        for i in range(len(sp2d) - 1):
            self.assert_allclose(sp2d.x1_at(i), sp2d.x0_at(i + 1))
            self.assert_allclose(sp2d.y1_at(i), sp2d.y0_at(i + 1))

    def test_arc_mixed_absolute_relative(self):
        d_attr = "M 6 10 A 6 4 10 1 0 14 10 a 6 4 10 0 1 6 0"
        fill_attr = "none"
        path_element = svg.EPath(d_attr=d_attr, fill_attr=fill_attr)
        sp2d = path_element.get_closed_paths()[0]
        cp2d = path_element.get_closed_paths()[1]
        self.assertEqual(sp2d.ndim, 2)
        self.assertEqual(cp2d.ndim, 2)
        self.assertEqual(len(sp2d), 39 * 2)
        self.assertEqual(len(cp2d), 0)

        self.assert_allclose(sp2d.x0_at(0), 6)
        self.assert_allclose(sp2d.y0_at(0), 10)
        self.assert_allclose(sp2d.x1_at(77), 20)
        self.assert_allclose(sp2d.y1_at(77), 10)

        self.assert_allclose(sp2d.x1_at(38), 14)
        self.assert_allclose(sp2d.y1_at(38), 10)

        for i in range(len(sp2d) - 1):
            self.assert_allclose(sp2d.x1_at(i), sp2d.x0_at(i + 1))
            self.assert_allclose(sp2d.y1_at(i), sp2d.y0_at(i + 1))

    def test_arc_point_continuity(self):
        d_attr = "M 0 0 A 10 10 0 0 1 20 0"
        fill_attr = "none"
        path_element = svg.EPath(d_attr=d_attr, fill_attr=fill_attr)
        sp2d = path_element.get_closed_paths()[0]
        cp2d = path_element.get_closed_paths()[1]
        self.assertEqual(sp2d.ndim, 2)
        self.assertEqual(cp2d.ndim, 2)
        self.assertEqual(len(sp2d), 39)
        self.assertEqual(len(cp2d), 0)

        self.assert_allclose(sp2d.x0_at(0), 0, atol=1e-14)
        self.assert_allclose(sp2d.y0_at(0), 0, atol=1e-14)
        self.assert_allclose(sp2d.x1_at(38), 20, atol=1e-14)
        self.assert_allclose(sp2d.y1_at(38), 0, atol=1e-14)

        for i in range(len(sp2d) - 1):
            self.assert_allclose(sp2d.x1_at(i), sp2d.x0_at(i + 1))
            self.assert_allclose(sp2d.y1_at(i), sp2d.y0_at(i + 1))

    def test_arc_circular(self):
        d_attr = "M 100 100 A 50 50 0 1 1 200 100"
        fill_attr = "none"
        path_element = svg.EPath(d_attr=d_attr, fill_attr=fill_attr)
        sp2d = path_element.get_closed_paths()[0]
        cp2d = path_element.get_closed_paths()[1]
        self.assertEqual(sp2d.ndim, 2)
        self.assertEqual(cp2d.ndim, 2)
        self.assertEqual(len(sp2d), 39)
        self.assertEqual(len(cp2d), 0)

        self.assert_allclose(sp2d.x0_at(0), 100)
        self.assert_allclose(sp2d.y0_at(0), 100)
        self.assert_allclose(sp2d.x1_at(38), 200)
        self.assert_allclose(sp2d.y1_at(38), 100)

        center_x = 150
        center_y = 100
        radius = 50

        for i in range(len(sp2d)):
            x0 = sp2d.x0_at(i)
            y0 = sp2d.y0_at(i)
            distance = ((x0 - center_x)**2 + (y0 - center_y)**2)**0.5
            self.assert_allclose(distance, radius, rtol=1e-2)

            x1 = sp2d.x1_at(i)
            y1 = sp2d.y1_at(i)
            distance = ((x1 - center_x)**2 + (y1 - center_y)**2)**0.5
            self.assert_allclose(distance, radius, rtol=1e-2)


class SvgPathCommandTC(SvgParserTB):
    """
    Test 'Z' and 'z' commands in SVG path 'd' attribute.
    See more: https://developer.mozilla.org/en-US/docs/Web/SVG/Attribute/d#clossvg.epath
    """  # noqa: E501

    def test_close_path_uppercase(self):
        d_attr = "M 10 10 L 20 10 l 10 0 Z"
        fill_attr = "none"
        path_element = svg.EPath(d_attr=d_attr, fill_attr=fill_attr)
        sp2d = path_element.get_closed_paths()[0]
        cp2d = path_element.get_closed_paths()[1]
        self.assertEqual(sp2d.ndim, 2)
        self.assertEqual(cp2d.ndim, 2)
        self.assertEqual(len(sp2d), 3)
        self.assertEqual(len(cp2d), 0)
        self.assertEqual(list(sp2d.x0), [10.0, 20.0, 30.0])
        self.assertEqual(list(sp2d.y0), [10.0, 10.0, 10.0])
        self.assertEqual(list(sp2d.x1), [20.0, 30.0, 10.0])
        self.assertEqual(list(sp2d.y1), [10.0, 10.0, 10.0])

    def test_close_path_lowercase(self):
        d_attr = "M 10 10 L 20 10 l 10 0 z"
        fill_attr = "none"
        path_element = svg.EPath(d_attr=d_attr, fill_attr=fill_attr)
        sp2d = path_element.get_closed_paths()[0]
        cp2d = path_element.get_closed_paths()[1]
        self.assertEqual(sp2d.ndim, 2)
        self.assertEqual(cp2d.ndim, 2)
        self.assertEqual(len(sp2d), 3)
        self.assertEqual(len(cp2d), 0)
        self.assertEqual(list(sp2d.x0), [10.0, 20.0, 30.0])
        self.assertEqual(list(sp2d.y0), [10.0, 10.0, 10.0])
        self.assertEqual(list(sp2d.x1), [20.0, 30.0, 10.0])
        self.assertEqual(list(sp2d.y1), [10.0, 10.0, 10.0])


class SvgShapeTC(SvgParserTB):
    """
    Test SVG rectangle shape parsing.
    See more: https://developer.mozilla.org/en-US/docs/Web/SVG/Tutorials/SVG_from_scratch/Basic_shapes
    """  # noqa: E501

    def test_rectangle_shape(self):
        rectangle = svg.ERectangle(x=10, y=10, width=30, height=20,
                                   fill_attr="none")
        self.assertEqual(len(rectangle.spads), 1)
        spad = rectangle.spads[0]
        self.assertEqual(spad.ndim, 2)
        self.assertEqual(len(spad), 4)
        self.assertEqual(list(spad.x0), [10.0, 40.0, 40.0, 10.0])
        self.assertEqual(list(spad.y0), [10.0, 10.0, 30.0, 30.0])
        self.assertEqual(list(spad.x1), [40.0, 40.0, 10.0, 10.0])
        self.assertEqual(list(spad.y1), [10.0, 30.0, 30.0, 10.0])

    def test_circle_shape(self):
        cx, cy, r = 50.0, 50.0, 20.0
        kappa = 0.5522847498

        circle = svg.ECircle(cx=cx, cy=cy, r=r, fill_attr="none")
        self.assertEqual(len(circle.cpads), 1)
        cpad = circle.cpads[0]

        # Check Curve 0: Top-right quadrant (0 to 90 degrees)
        self.assert_allclose(list(cpad.p0_at(0)), [cx + r, cy, 0])
        self.assert_allclose(list(cpad.p1_at(0)), [cx + r, cy + r * kappa, 0])
        self.assert_allclose(list(cpad.p2_at(0)), [cx + r * kappa, cy + r, 0])
        self.assert_allclose(list(cpad.p3_at(0)), [cx, cy + r, 0])

        # Check Curve 1: Top-left quadrant (90 to 180 degrees)
        self.assert_allclose(list(cpad.p0_at(1)), [cx, cy + r, 0])
        self.assert_allclose(list(cpad.p1_at(1)), [cx - r * kappa, cy + r, 0])
        self.assert_allclose(list(cpad.p2_at(1)), [cx - r, cy + r * kappa, 0])
        self.assert_allclose(list(cpad.p3_at(1)), [cx - r, cy, 0])

        # Check Curve 2: Bottom-left quadrant (180 to 270 degrees)
        self.assert_allclose(list(cpad.p0_at(2)), [cx - r, cy, 0])
        self.assert_allclose(list(cpad.p1_at(2)), [cx - r, cy - r * kappa, 0])
        self.assert_allclose(list(cpad.p2_at(2)), [cx - r * kappa, cy - r, 0])
        self.assert_allclose(list(cpad.p3_at(2)), [cx, cy - r, 0])

        # Check Curve 3: Bottom-right quadrant (270 to 360 degrees)
        self.assert_allclose(list(cpad.p0_at(3)), [cx, cy - r, 0])
        self.assert_allclose(list(cpad.p1_at(3)), [cx + r * kappa, cy - r, 0])
        self.assert_allclose(list(cpad.p2_at(3)), [cx + r, cy - r * kappa, 0])
        self.assert_allclose(list(cpad.p3_at(3)), [cx + r, cy, 0])

    def test_ellipse_shape(self):
        cx, cy, rx, ry = 50.0, 50.0, 30.0, 20.0
        kappa = 0.5522847498

        ellipse = svg.EEllipse(cx=cx, cy=cy, rx=rx, ry=ry, fill_attr="none")
        self.assertEqual(len(ellipse.cpads), 1)
        cpad = ellipse.cpads[0]

        # Check Curve 0: Top-right quadrant (0 to 90 degrees)
        self.assert_allclose(list(cpad.p0_at(0)), [cx + rx, cy, 0])
        self.assert_allclose(list(cpad.p1_at(0)),
                             [cx + rx, cy + ry * kappa, 0])
        self.assert_allclose(list(cpad.p2_at(0)),
                             [cx + rx * kappa, cy + ry, 0])
        self.assert_allclose(list(cpad.p3_at(0)), [cx, cy + ry, 0])

        # Check Curve 1: Top-left quadrant (90 to 180 degrees)
        self.assert_allclose(list(cpad.p0_at(1)), [cx, cy + ry, 0])
        self.assert_allclose(list(cpad.p1_at(1)),
                             [cx - rx * kappa, cy + ry, 0])
        self.assert_allclose(list(cpad.p2_at(1)),
                             [cx - rx, cy + ry * kappa, 0])
        self.assert_allclose(list(cpad.p3_at(1)), [cx - rx, cy, 0])

        # Check Curve 2: Bottom-left quadrant (180 to 270 degrees)
        self.assert_allclose(list(cpad.p0_at(2)), [cx - rx, cy, 0])
        self.assert_allclose(list(cpad.p1_at(2)),
                             [cx - rx, cy - ry * kappa, 0])
        self.assert_allclose(list(cpad.p2_at(2)),
                             [cx - rx * kappa, cy - ry, 0])
        self.assert_allclose(list(cpad.p3_at(2)), [cx, cy - ry, 0])

        # Check Curve 3: Bottom-right quadrant (270 to 360 degrees)
        self.assert_allclose(list(cpad.p0_at(3)), [cx, cy - ry, 0])
        self.assert_allclose(list(cpad.p1_at(3)),
                             [cx + rx * kappa, cy - ry, 0])
        self.assert_allclose(list(cpad.p2_at(3)),
                             [cx + rx, cy - ry * kappa, 0])
        self.assert_allclose(list(cpad.p3_at(3)), [cx + rx, cy, 0])

    def test_line_shape(self):
        line = svg.ELine(x1=10, y1=20, x2=30, y2=40, fill_attr="none")
        self.assertEqual(len(line.spads), 1)
        spad = line.spads[0]

        self.assertEqual(spad.ndim, 2)
        self.assertEqual(len(spad), 1)
        self.assertEqual(list(spad.x0), [10.0])
        self.assertEqual(list(spad.y0), [20.0])
        self.assertEqual(list(spad.x1), [30.0])
        self.assertEqual(list(spad.y1), [40.0])

    def test_polyline_shape(self):
        polyline = svg.EPolyline(points=[(10, 10), (20, 20), (30, 10)],
                                 fill_attr="none")
        self.assertEqual(len(polyline.spads), 1)
        spad = polyline.spads[0]

        self.assertEqual(spad.ndim, 2)
        self.assertEqual(len(spad), 2)
        self.assertEqual(list(spad.x0), [10.0, 20.0])
        self.assertEqual(list(spad.y0), [10.0, 20.0])
        self.assertEqual(list(spad.x1), [20.0, 30.0])
        self.assertEqual(list(spad.y1), [20.0, 10.0])

    def test_polygon_shape(self):
        polygon = svg.EPolygon(points=[(10, 10), (20, 20), (30, 10)],
                               fill_attr="none")
        self.assertEqual(len(polygon.spads), 1)
        spad = polygon.spads[0]

        self.assertEqual(spad.ndim, 2)
        self.assertEqual(len(spad), 3)
        self.assertEqual(list(spad.x0), [10.0, 20.0, 30.0])
        self.assertEqual(list(spad.y0), [10.0, 20.0, 10.0])
        self.assertEqual(list(spad.x1), [20.0, 30.0, 10.0])
        self.assertEqual(list(spad.y1), [20.0, 10.0, 10.0])

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
