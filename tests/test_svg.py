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

import modmesh
from modmesh.testing import TestBase as ModMeshTB
from modmesh.plot.svg import EPath


class SvgParserTC(ModMeshTB, unittest.TestCase):

    def assert_allclose(self, *args, **kw):
        if 'rtol' not in kw:
            kw['rtol'] = 1.e-15
        return super().assert_allclose(*args, **kw)

    def test_single_closed_path(self):
        d_attr = """ M10 10 L60 10 L60 60 L10 60 Z """
        fill_attr = "none"
        path_element = EPath(d_attr=d_attr, fill_attr=fill_attr)
        self.assertEqual(len(path_element.get_cmds()), 5)
        self.assertEqual(
            path_element.get_cmds(),
            [('M', [10.0, 10.0]),
             ('L', [60.0, 10.0]),
             ('L', [60.0, 60.0]),
             ('L', [10.0, 60.0]),
             ('Z', [])])

    def test_multiple_closed_paths(self):
        d_attr = """
                M10 10 L60 10 L60 60 L10 60 Z
                M100 10 L150 60 L100 60 Z
                """
        fill_attr = "none"
        path_element = EPath(d_attr=d_attr, fill_attr=fill_attr)
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

    def test_moveto_command(self):
        d_attr = """
                M 10 10 h 10
                m  0 10 h 10
                """
        fill_attr = "none"
        path_element = EPath(d_attr=d_attr, fill_attr=fill_attr)
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

        # Implicit lineTo command
        d_attr = """
                M 10 10 20 10
                m 0 10 10 0
                """
        path_element = EPath(d_attr=d_attr, fill_attr=fill_attr)
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

    def test_lineto_command(self):
        d_attr = """
                M 10 10
                L 20 10
                l 10 0
                V 20
                v 10
                H 20
                h -10
                """
        fill_attr = "none"
        path_element = EPath(d_attr=d_attr, fill_attr=fill_attr)
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

        d_attr = """
                M 10 10
                L 20 10 30 10
                l 10 0 10 0
                V 20 30
                v 10 10
                H 40 30
                h -10 -10
                """
        path_element = EPath(d_attr=d_attr, fill_attr=fill_attr)
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

    def test_cbc_command(self):
        d_attr = """
                M 10 90
                C 30 90 25 10 50 10
                S 70 90 90 90
                c 20 0 15 -80 40 -80
                s 20 80 40 80
                """
        fill_attr = "none"
        path_element = EPath(d_attr=d_attr, fill_attr=fill_attr)
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

        # Test with implicit cubic Bézier curves
        d_attr = """
                M 10 90
                C 30 90 25 10 50 10 75 10 70 90 90 90
                S 105 10 130 10 150 90 170 90
                c 20 0 15 -80 40 -80 25 0 20 80 40 80
                s 15 -80 40 -80 20 80 40 80
                """
        fill_attr = "none"
        path_element = EPath(d_attr=d_attr, fill_attr=fill_attr)
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

    def test_qbc_command(self):
        d_attr = """
                M 10 50
                Q 25 25 40 50
                q 15 25 30 0
                T 100 50
                t 30 0
                """
        fill_attr = "none"
        path_element = EPath(d_attr=d_attr, fill_attr=fill_attr)
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

        # Test with implicit quadratic Bézier curves
        d_attr = """
                M 10 50
                Q 25 25 40 50 55 75 70 50
                q 15 -25 30 0 15 25 30 0
                T 160 50 190 50
                t 30 0 30 0
                """
        path_element = EPath(d_attr=d_attr, fill_attr=fill_attr)
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

    def test_arc_command(self):
        d_attr = """
                M 6 10
                A 6 4 10 1 0 14 10
                A 6 4 10 0 1 20 10
                """
        fill_attr = "none"
        path_element = EPath(d_attr=d_attr, fill_attr=fill_attr)
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

        d_attr = """
                M 6 10
                A 6 4 10 1 0 14 10 6 4 10 0 1 20 10
                """
        path_element = EPath(d_attr=d_attr, fill_attr=fill_attr)
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

    def test_close_path_command(self):
        d_attr = """
                M 10 10
                L 20 10
                l 10 0
                Z
                """
        fill_attr = "none"
        path_element = EPath(d_attr=d_attr, fill_attr=fill_attr)
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

        d_attr = """
                M 10 10
                L 20 10
                l 10 0
                z
                """
        fill_attr = "none"
        path_element = EPath(d_attr=d_attr, fill_attr=fill_attr)
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

class SvgBindingTC(ModMeshTB, unittest.TestCase):

    # TODO: DUMMY test for initialization. Will remove later.
    def test_create_svg_header_basic(self):
        header = modmesh.svg.create_svg_header(100, 200)
        self.assertIsInstance(header, str)


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
