# Copyright (c) 2018, Yung-Yu Chen <yyc@solvcon.net>
# BSD 3-Clause License, see COPYING

# flake8: noqa: E501

import unittest
import logging
import sys
import os

import numpy as np

import modmesh as mm


class DrawBase:

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        if __name__ == '__main__':
            self.main()

    def main(self):
        filename = os.path.splitext(os.path.basename(sys.argv[0]))[0] + '.eps'
        if len(sys.argv) > 1:
            filename = sys.argv[1]

        drawn = self.draw()
        drawn.to_filename(filename)

        if len(sys.argv) > 2:
            filename = sys.argv[2]
            with open(filename, 'w') as fobj:
                fobj.write(str(drawn))
            print("Wrote tex file to", filename)


class DrawTB(unittest.TestCase):

    def __init__(self, *args, **kw):
        self.tex_filename = ''
        self.tex_output_dir = ''
        self.golden = ''
        super().__init__(*args, **kw)

    def setUp(self):
        logger = logging.getLogger()
        logger.level = logging.DEBUG
        stream_handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(stream_handler)

    def _compare(self):
        drawn_tex = str(self.draw())

        # Write a tex file for debugging.
        if self.tex_output_dir and self.tex_filename:
            tex_path = os.path.join(self.tex_output_dir, self.tex_filename)
            with open(tex_path, 'w') as fobj:
                fobj.write(drawn_tex)
            logging.getLogger().info(
                "class {} writes to {}".format(self.__class__.__name__,
                                               tex_path))

        self.assertEqual(self.golden, drawn_tex)


class DrawCeseMarching(DrawBase):
    """
    An example for a command-line class to draw for the CESE method schematics.
    """

    @staticmethod
    def draw():

        grid = mm.spacetime.Grid(0, 6, 4)
        sol = mm.spacetime.Solver(grid=grid, nvar=1, time_increment=6 / 4)

        dx = (grid.xmax - grid.xmin) / grid.ncelm
        dt = dx
        hdt = dt / 2

        c = mm.onedim.draw.PstCanvas(unit='2cm', padding=0.5)
        c.set(linewidth='1pt')
        c.set(linecolor='black')

        # CE grids.
        linewidth = '0.5pt'
        for se in sol.selms(odd_plane=False):
            c.line((se.x, 0), (se.x, dt), linewidth=linewidth)
        c.line((sol.selm(0).x, 0), (sol.selm(grid.ncelm).x, 0),
               linewidth=linewidth)
        c.line((sol.selm(0).x, hdt), (sol.selm(grid.ncelm).x, hdt),
               linewidth=linewidth)
        c.line((sol.selm(0).x, dt), (sol.selm(grid.ncelm).x, dt),
               linewidth=linewidth)
        for se in sol.selms(odd_plane=False):
            c.line((se.x, dt), (se.x, dt * 1.65), arrows='->',
                   linewidth=linewidth, linestyle='dashed')
        c.line((sol.selm(0).x, dt * 1.5), (sol.selm(grid.ncelm).x, dt * 1.5),
               linewidth=linewidth, linestyle='dashed')

        # x-axis.
        sep = 0.05
        c.line((sol.selm(0).x, -hdt * 1.2),
               (sol.selm(grid.ncelm).x, -hdt * 1.2),
               linewidth=linewidth)
        c.uput(sep, 'l', (grid.xmin - sep, -hdt * 1.2), r'$j$')
        c.uput(sep, 'r', (grid.xmax + sep, -hdt * 1.2), r'$x$')
        for se in sol.selms(odd_plane=False):
            c.line((se.x, -hdt * 1.2 - sep), (se.x, -hdt * 1.2 + sep))
            c.uput(sep, 'd', (se.x, -hdt * 1.2 - sep), r'$%d$' % se.index)
        for se in sol.selms(odd_plane=True):
            c.line((se.x, -hdt * 1.2 - sep / 2), (se.x, -hdt * 1.2 + sep / 2))
            c.uput(sep, 'd', (se.x, -hdt * 1.2 - sep),
                   r'$\frac{%d}{2}$' % (se.index * 2 + 1))

        # t-axis.
        x = grid.xmin - dx * 0.6
        c.line((x, 0), (x, dt), linewidth=linewidth)
        c.uput(sep, 'd', (x, 0 - sep), r'$n$')
        c.uput(sep, 'u', (x, dt + sep), r'$t$')
        c.line((x - sep, 0), (x + sep, 0))
        c.uput(sep, 'l', (x - sep, 0), r'$0$')
        c.line((x - sep / 2, hdt), (x + sep / 2, hdt))
        c.uput(sep, 'l', (x - sep, hdt), r'$\frac{1}{2}$')
        c.line((x - sep, dt), (x + sep, dt))
        c.uput(sep, 'l', (x - sep, dt), r'$1$')

        # SE and solution propagation.
        sep = 0.05
        sepx = dx * 0.1
        sept = dt * 0.1
        for se in sol.selms(odd_plane=False):
            c.selm(se, 0,
                   sep=sep, linestyle='dotted', dotsep='1pt', linecolor='red')
            if se.index != grid.ncelm:  # left to right.
                c.line((se.x + sepx, 0 + sept), (se.xpos - sepx, hdt - sept),
                       arrows='->', linecolor='red')
            if se.index != 0:  # right to left.
                c.line((se.x - sepx, 0 + sept), (se.xneg + sepx, hdt - sept),
                       arrows='->', linecolor='red')
        for se in sol.selms(odd_plane=True):
            c.selm(se, hdt,
                   sep=sep, linestyle='dotted', dotsep='1pt', linecolor='blue')
            if se.index != grid.ncelm - 1:  # left to right.
                c.line((se.x + sepx, hdt + sept), (se.xpos - sepx, dt - sept),
                       arrows='->', linecolor='blue')
            if se.index != 0:  # right to left.
                c.line((se.x - sepx, hdt + sept), (se.xneg + sepx, dt - sept),
                       arrows='->', linecolor='blue')
        for se in sol.selms(odd_plane=False):
            c.selm(se, dt,
                   sep=sep, linestyle='dotted', dotsep='1pt',
                   linecolor='orange')

        return c

    @staticmethod
    def draw_cce():

        c = mm.onedim.draw.PstCanvas(unit='2cm', extent=(-2.4, -0.5, 2.4, 1.5))
        c.set(linewidth='1pt')
        c.set(linecolor='black')
        c.frame((-1, 0), (1, 1))
        c.line((0, 0), (0, 1), linestyle='dashed')
        c.frame((-0.95, 0.05), (-0.05, 0.95), linestyle='dotted',
                linecolor='red')
        c.rput('tr', (-0.1, 0.9), r'$\mathrm{CE}_-$')
        c.frame((0.05, 0.05), (0.95, 0.95), linestyle='dotted',
                linecolor='red')
        c.rput('tr', (0.9, 0.9), r'$\mathrm{CE}_+$')
        c.frame((-1.05, -0.05), (1.05, 1.05), linestyle='dotted',
                linecolor='red')
        c.rput('tr', (-1.1, 0.9), r'$\mathrm{CE}$')
        c.dots((0, 1), (-1, 1), (-1, 0), (0, 0), (1, 0), (1, 1), (0, 1),
               dotstyle='*')
        c.uput(0.1, 'u', (0, 1), r'A $(x_j,t^n)$')
        c.uput(0.1, 'ul', (-1, 1), r'B')
        c.uput(0.1, 'dl', (-1, 0),
               r'$(x_{j-\frac{1}{2}},t^{n-\frac{1}{2}})$ C')
        c.uput(0.1, 'd', (0, 0), r'D')
        c.uput(0.1, 'dr', (1, 0), r'E $(x_{j+\frac{1}{2}},t^{n-\frac{1}{2}})$')
        c.uput(0.1, 'ur', (1, 1), r'F')
        c.line((0, 0), (0, 1), linestyle='dashed')

        return c


class DrawCeseMarchingTC(DrawTB, DrawCeseMarching):

    def test(self):
        self.tex_filename = 'cese_marching.tex'
        self.tex_output_dir = ''  # Set to a directory for tex debug output.
        self.golden = r'''\psset{unit=2cm}
\begin{pspicture}(-1.45,-1.45)(7.175,2.975)

\psset{linewidth=1pt}
\psset{linecolor=black}
\psline[linewidth=0.5pt](0,0)(0,1.5)
\psline[linewidth=0.5pt](1.5,0)(1.5,1.5)
\psline[linewidth=0.5pt](3,0)(3,1.5)
\psline[linewidth=0.5pt](4.5,0)(4.5,1.5)
\psline[linewidth=0.5pt](6,0)(6,1.5)
\psline[linewidth=0.5pt](0,0)(6,0)
\psline[linewidth=0.5pt](0,0.75)(6,0.75)
\psline[linewidth=0.5pt](0,1.5)(6,1.5)
\psline[arrows=->,linewidth=0.5pt,linestyle=dashed](0,1.5)(0,2.475)
\psline[arrows=->,linewidth=0.5pt,linestyle=dashed](1.5,1.5)(1.5,2.475)
\psline[arrows=->,linewidth=0.5pt,linestyle=dashed](3,1.5)(3,2.475)
\psline[arrows=->,linewidth=0.5pt,linestyle=dashed](4.5,1.5)(4.5,2.475)
\psline[arrows=->,linewidth=0.5pt,linestyle=dashed](6,1.5)(6,2.475)
\psline[linewidth=0.5pt,linestyle=dashed](0,2.25)(6,2.25)
\psline[linewidth=0.5pt](0,-0.9)(6,-0.9)
\uput{0.05}[l](-0.05,-0.9){$j$}
\uput{0.05}[r](6.05,-0.9){$x$}
\psline(0,-0.95)(0,-0.85)
\uput{0.05}[d](0,-0.95){$0$}
\psline(1.5,-0.95)(1.5,-0.85)
\uput{0.05}[d](1.5,-0.95){$1$}
\psline(3,-0.95)(3,-0.85)
\uput{0.05}[d](3,-0.95){$2$}
\psline(4.5,-0.95)(4.5,-0.85)
\uput{0.05}[d](4.5,-0.95){$3$}
\psline(6,-0.95)(6,-0.85)
\uput{0.05}[d](6,-0.95){$4$}
\psline(0.75,-0.925)(0.75,-0.875)
\uput{0.05}[d](0.75,-0.95){$\frac{1}{2}$}
\psline(2.25,-0.925)(2.25,-0.875)
\uput{0.05}[d](2.25,-0.95){$\frac{3}{2}$}
\psline(3.75,-0.925)(3.75,-0.875)
\uput{0.05}[d](3.75,-0.95){$\frac{5}{2}$}
\psline(5.25,-0.925)(5.25,-0.875)
\uput{0.05}[d](5.25,-0.95){$\frac{7}{2}$}
\psline[linewidth=0.5pt](-0.9,0)(-0.9,1.5)
\uput{0.05}[d](-0.9,-0.05){$n$}
\uput{0.05}[u](-0.9,1.55){$t$}
\psline(-0.95,0)(-0.85,0)
\uput{0.05}[l](-0.95,0){$0$}
\psline(-0.925,0.75)(-0.875,0.75)
\uput{0.05}[l](-0.95,0.75){$\frac{1}{2}$}
\psline(-0.95,1.5)(-0.85,1.5)
\uput{0.05}[l](-0.95,1.5){$1$}
\psdots[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=red](0,0)
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=red](0.0375,0.05625)(0.0375,0.65625)
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=red](-0.0375,0.05625)(-0.0375,0.65625)
\psarc[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=red](0,0.675){0.0375}{0}{180}
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=red](0.0375,-0.05625)(0.0375,-0.65625)
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=red](-0.0375,-0.05625)(-0.0375,-0.65625)
\psarc[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=red](0,-0.675){0.0375}{180}{0}
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=red](0.05625,0.0375)(0.65625,0.0375)
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=red](0.05625,-0.0375)(0.65625,-0.0375)
\psarc[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=red](0.675,0){0.0375}{270}{90}
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=red](-0.05625,0.0375)(-0.65625,0.0375)
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=red](-0.05625,-0.0375)(-0.65625,-0.0375)
\psarc[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=red](-0.675,0){0.0375}{90}{270}
\psline[arrows=->,linecolor=red](0.15,0.15)(0.6,0.6)
\psdots[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=red](1.5,0)
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=red](1.5375,0.05625)(1.5375,0.65625)
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=red](1.4625,0.05625)(1.4625,0.65625)
\psarc[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=red](1.5,0.675){0.0375}{0}{180}
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=red](1.5375,-0.05625)(1.5375,-0.65625)
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=red](1.4625,-0.05625)(1.4625,-0.65625)
\psarc[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=red](1.5,-0.675){0.0375}{180}{0}
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=red](1.55625,0.0375)(2.15625,0.0375)
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=red](1.55625,-0.0375)(2.15625,-0.0375)
\psarc[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=red](2.175,0){0.0375}{270}{90}
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=red](1.44375,0.0375)(0.84375,0.0375)
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=red](1.44375,-0.0375)(0.84375,-0.0375)
\psarc[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=red](0.825,0){0.0375}{90}{270}
\psline[arrows=->,linecolor=red](1.65,0.15)(2.1,0.6)
\psline[arrows=->,linecolor=red](1.35,0.15)(0.9,0.6)
\psdots[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=red](3,0)
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=red](3.0375,0.05625)(3.0375,0.65625)
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=red](2.9625,0.05625)(2.9625,0.65625)
\psarc[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=red](3,0.675){0.0375}{0}{180}
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=red](3.0375,-0.05625)(3.0375,-0.65625)
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=red](2.9625,-0.05625)(2.9625,-0.65625)
\psarc[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=red](3,-0.675){0.0375}{180}{0}
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=red](3.05625,0.0375)(3.65625,0.0375)
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=red](3.05625,-0.0375)(3.65625,-0.0375)
\psarc[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=red](3.675,0){0.0375}{270}{90}
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=red](2.94375,0.0375)(2.34375,0.0375)
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=red](2.94375,-0.0375)(2.34375,-0.0375)
\psarc[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=red](2.325,0){0.0375}{90}{270}
\psline[arrows=->,linecolor=red](3.15,0.15)(3.6,0.6)
\psline[arrows=->,linecolor=red](2.85,0.15)(2.4,0.6)
\psdots[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=red](4.5,0)
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=red](4.5375,0.05625)(4.5375,0.65625)
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=red](4.4625,0.05625)(4.4625,0.65625)
\psarc[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=red](4.5,0.675){0.0375}{0}{180}
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=red](4.5375,-0.05625)(4.5375,-0.65625)
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=red](4.4625,-0.05625)(4.4625,-0.65625)
\psarc[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=red](4.5,-0.675){0.0375}{180}{0}
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=red](4.55625,0.0375)(5.15625,0.0375)
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=red](4.55625,-0.0375)(5.15625,-0.0375)
\psarc[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=red](5.175,0){0.0375}{270}{90}
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=red](4.44375,0.0375)(3.84375,0.0375)
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=red](4.44375,-0.0375)(3.84375,-0.0375)
\psarc[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=red](3.825,0){0.0375}{90}{270}
\psline[arrows=->,linecolor=red](4.65,0.15)(5.1,0.6)
\psline[arrows=->,linecolor=red](4.35,0.15)(3.9,0.6)
\psdots[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=red](6,0)
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=red](6.0375,0.05625)(6.0375,0.65625)
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=red](5.9625,0.05625)(5.9625,0.65625)
\psarc[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=red](6,0.675){0.0375}{0}{180}
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=red](6.0375,-0.05625)(6.0375,-0.65625)
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=red](5.9625,-0.05625)(5.9625,-0.65625)
\psarc[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=red](6,-0.675){0.0375}{180}{0}
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=red](6.05625,0.0375)(6.65625,0.0375)
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=red](6.05625,-0.0375)(6.65625,-0.0375)
\psarc[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=red](6.675,0){0.0375}{270}{90}
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=red](5.94375,0.0375)(5.34375,0.0375)
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=red](5.94375,-0.0375)(5.34375,-0.0375)
\psarc[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=red](5.325,0){0.0375}{90}{270}
\psline[arrows=->,linecolor=red](5.85,0.15)(5.4,0.6)
\psdots[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=blue](0.75,0.75)
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=blue](0.7875,0.80625)(0.7875,1.40625)
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=blue](0.7125,0.80625)(0.7125,1.40625)
\psarc[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=blue](0.75,1.425){0.0375}{0}{180}
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=blue](0.7875,0.69375)(0.7875,0.09375)
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=blue](0.7125,0.69375)(0.7125,0.09375)
\psarc[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=blue](0.75,0.075){0.0375}{180}{0}
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=blue](0.80625,0.7875)(1.40625,0.7875)
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=blue](0.80625,0.7125)(1.40625,0.7125)
\psarc[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=blue](1.425,0.75){0.0375}{270}{90}
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=blue](0.69375,0.7875)(0.09375,0.7875)
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=blue](0.69375,0.7125)(0.09375,0.7125)
\psarc[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=blue](0.075,0.75){0.0375}{90}{270}
\psline[arrows=->,linecolor=blue](0.9,0.9)(1.35,1.35)
\psdots[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=blue](2.25,0.75)
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=blue](2.2875,0.80625)(2.2875,1.40625)
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=blue](2.2125,0.80625)(2.2125,1.40625)
\psarc[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=blue](2.25,1.425){0.0375}{0}{180}
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=blue](2.2875,0.69375)(2.2875,0.09375)
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=blue](2.2125,0.69375)(2.2125,0.09375)
\psarc[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=blue](2.25,0.075){0.0375}{180}{0}
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=blue](2.30625,0.7875)(2.90625,0.7875)
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=blue](2.30625,0.7125)(2.90625,0.7125)
\psarc[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=blue](2.925,0.75){0.0375}{270}{90}
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=blue](2.19375,0.7875)(1.59375,0.7875)
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=blue](2.19375,0.7125)(1.59375,0.7125)
\psarc[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=blue](1.575,0.75){0.0375}{90}{270}
\psline[arrows=->,linecolor=blue](2.4,0.9)(2.85,1.35)
\psline[arrows=->,linecolor=blue](2.1,0.9)(1.65,1.35)
\psdots[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=blue](3.75,0.75)
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=blue](3.7875,0.80625)(3.7875,1.40625)
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=blue](3.7125,0.80625)(3.7125,1.40625)
\psarc[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=blue](3.75,1.425){0.0375}{0}{180}
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=blue](3.7875,0.69375)(3.7875,0.09375)
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=blue](3.7125,0.69375)(3.7125,0.09375)
\psarc[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=blue](3.75,0.075){0.0375}{180}{0}
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=blue](3.80625,0.7875)(4.40625,0.7875)
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=blue](3.80625,0.7125)(4.40625,0.7125)
\psarc[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=blue](4.425,0.75){0.0375}{270}{90}
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=blue](3.69375,0.7875)(3.09375,0.7875)
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=blue](3.69375,0.7125)(3.09375,0.7125)
\psarc[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=blue](3.075,0.75){0.0375}{90}{270}
\psline[arrows=->,linecolor=blue](3.9,0.9)(4.35,1.35)
\psline[arrows=->,linecolor=blue](3.6,0.9)(3.15,1.35)
\psdots[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=blue](5.25,0.75)
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=blue](5.2875,0.80625)(5.2875,1.40625)
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=blue](5.2125,0.80625)(5.2125,1.40625)
\psarc[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=blue](5.25,1.425){0.0375}{0}{180}
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=blue](5.2875,0.69375)(5.2875,0.09375)
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=blue](5.2125,0.69375)(5.2125,0.09375)
\psarc[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=blue](5.25,0.075){0.0375}{180}{0}
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=blue](5.30625,0.7875)(5.90625,0.7875)
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=blue](5.30625,0.7125)(5.90625,0.7125)
\psarc[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=blue](5.925,0.75){0.0375}{270}{90}
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=blue](5.19375,0.7875)(4.59375,0.7875)
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=blue](5.19375,0.7125)(4.59375,0.7125)
\psarc[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=blue](4.575,0.75){0.0375}{90}{270}
\psline[arrows=->,linecolor=blue](5.1,0.9)(4.65,1.35)
\psdots[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=orange](0,1.5)
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=orange](0.0375,1.55625)(0.0375,2.15625)
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=orange](-0.0375,1.55625)(-0.0375,2.15625)
\psarc[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=orange](0,2.175){0.0375}{0}{180}
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=orange](0.0375,1.44375)(0.0375,0.84375)
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=orange](-0.0375,1.44375)(-0.0375,0.84375)
\psarc[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=orange](0,0.825){0.0375}{180}{0}
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=orange](0.05625,1.5375)(0.65625,1.5375)
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=orange](0.05625,1.4625)(0.65625,1.4625)
\psarc[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=orange](0.675,1.5){0.0375}{270}{90}
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=orange](-0.05625,1.5375)(-0.65625,1.5375)
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=orange](-0.05625,1.4625)(-0.65625,1.4625)
\psarc[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=orange](-0.675,1.5){0.0375}{90}{270}
\psdots[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=orange](1.5,1.5)
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=orange](1.5375,1.55625)(1.5375,2.15625)
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=orange](1.4625,1.55625)(1.4625,2.15625)
\psarc[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=orange](1.5,2.175){0.0375}{0}{180}
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=orange](1.5375,1.44375)(1.5375,0.84375)
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=orange](1.4625,1.44375)(1.4625,0.84375)
\psarc[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=orange](1.5,0.825){0.0375}{180}{0}
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=orange](1.55625,1.5375)(2.15625,1.5375)
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=orange](1.55625,1.4625)(2.15625,1.4625)
\psarc[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=orange](2.175,1.5){0.0375}{270}{90}
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=orange](1.44375,1.5375)(0.84375,1.5375)
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=orange](1.44375,1.4625)(0.84375,1.4625)
\psarc[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=orange](0.825,1.5){0.0375}{90}{270}
\psdots[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=orange](3,1.5)
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=orange](3.0375,1.55625)(3.0375,2.15625)
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=orange](2.9625,1.55625)(2.9625,2.15625)
\psarc[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=orange](3,2.175){0.0375}{0}{180}
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=orange](3.0375,1.44375)(3.0375,0.84375)
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=orange](2.9625,1.44375)(2.9625,0.84375)
\psarc[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=orange](3,0.825){0.0375}{180}{0}
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=orange](3.05625,1.5375)(3.65625,1.5375)
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=orange](3.05625,1.4625)(3.65625,1.4625)
\psarc[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=orange](3.675,1.5){0.0375}{270}{90}
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=orange](2.94375,1.5375)(2.34375,1.5375)
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=orange](2.94375,1.4625)(2.34375,1.4625)
\psarc[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=orange](2.325,1.5){0.0375}{90}{270}
\psdots[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=orange](4.5,1.5)
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=orange](4.5375,1.55625)(4.5375,2.15625)
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=orange](4.4625,1.55625)(4.4625,2.15625)
\psarc[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=orange](4.5,2.175){0.0375}{0}{180}
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=orange](4.5375,1.44375)(4.5375,0.84375)
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=orange](4.4625,1.44375)(4.4625,0.84375)
\psarc[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=orange](4.5,0.825){0.0375}{180}{0}
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=orange](4.55625,1.5375)(5.15625,1.5375)
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=orange](4.55625,1.4625)(5.15625,1.4625)
\psarc[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=orange](5.175,1.5){0.0375}{270}{90}
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=orange](4.44375,1.5375)(3.84375,1.5375)
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=orange](4.44375,1.4625)(3.84375,1.4625)
\psarc[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=orange](3.825,1.5){0.0375}{90}{270}
\psdots[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=orange](6,1.5)
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=orange](6.0375,1.55625)(6.0375,2.15625)
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=orange](5.9625,1.55625)(5.9625,2.15625)
\psarc[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=orange](6,2.175){0.0375}{0}{180}
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=orange](6.0375,1.44375)(6.0375,0.84375)
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=orange](5.9625,1.44375)(5.9625,0.84375)
\psarc[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=orange](6,0.825){0.0375}{180}{0}
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=orange](6.05625,1.5375)(6.65625,1.5375)
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=orange](6.05625,1.4625)(6.65625,1.4625)
\psarc[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=orange](6.675,1.5){0.0375}{270}{90}
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=orange](5.94375,1.5375)(5.34375,1.5375)
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=orange](5.94375,1.4625)(5.34375,1.4625)
\psarc[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=orange](5.325,1.5){0.0375}{90}{270}

\end{pspicture}'''
        self._compare()


class DrawNonuniSe(DrawBase):
    """
    An example for a command-line class to draw for non-uniform solution
    element.
    """

    def main(self):

        filename = os.path.splitext(os.path.basename(sys.argv[0]))[0] + '.eps'
        if len(sys.argv) > 1:
            filename = sys.argv[1]

        drawn = self.draw()
        drawn.to_filename(filename)

        if len(sys.argv) > 2:
            filename = sys.argv[2]
            with open(filename, 'w') as fobj:
                fobj.write(str(drawn))
            print("Wrote tex file to", filename)

    @staticmethod
    def draw():

        xloc = np.array([-1, 0, 2, 3.5])
        grid = mm.spacetime.Grid(xloc=xloc)
        sol = mm.spacetime.Solver(grid=grid, nvar=1, time_increment=6 / 4)

        dx = (grid.xmax - grid.xmin) / grid.ncelm
        dt = dx
        hdt = dt / 2

        c = mm.onedim.draw.PstCanvas(unit='2cm', padding=0.5)
        c.set(linewidth='1pt')
        c.set(linecolor='black')

        # CE grids.
        linewidth = '0.5pt'
        for se in sol.selms(odd_plane=False):
            c.line((se.x, 0), (se.x, dt), linewidth=linewidth)
        c.line((sol.selm(0).x, 0), (sol.selm(grid.ncelm).x, 0),
               linewidth=linewidth)
        c.line((sol.selm(0).x, hdt), (sol.selm(grid.ncelm).x, hdt),
               linewidth=linewidth)
        c.line((sol.selm(0).x, dt), (sol.selm(grid.ncelm).x, dt),
               linewidth=linewidth)

        # x-axis.
        sep = 0.05
        c.line((sol.selm(0).x, -hdt * 1.2),
               (sol.selm(grid.ncelm).x, -hdt * 1.2),
               linewidth=linewidth)
        c.uput(sep, 'l', (grid.xmin - sep, -hdt * 1.2), r'$j$')
        c.uput(sep, 'r', (grid.xmax + sep, -hdt * 1.2), r'$x$')
        for se in sol.selms(odd_plane=False):
            c.line((se.x, -hdt * 1.2 - sep), (se.x, -hdt * 1.2 + sep))
            c.uput(sep, 'd', (se.x, -hdt * 1.2 - sep), r'$%d$' % se.index)
        for se in sol.selms(odd_plane=True):
            c.line((se.x, -hdt * 1.2 - sep / 2), (se.x, -hdt * 1.2 + sep / 2))
            c.uput(sep, 'd', (se.x, -hdt * 1.2 - sep),
                   r'$\frac{%d}{2}$' % (se.index * 2 + 1))

        # t-axis.
        x = grid.xmin - dx * 0.6
        c.line((x, 0), (x, dt), linewidth=linewidth)
        c.uput(sep, 'd', (x, 0 - sep), r'$n$')
        c.uput(sep, 'u', (x, dt + sep), r'$t$')
        c.line((x - sep, 0), (x + sep, 0))
        c.uput(sep, 'l', (x - sep, 0), r'$0$')
        c.line((x - sep / 2, hdt), (x + sep / 2, hdt))
        c.uput(sep, 'l', (x - sep, hdt), r'$\frac{1}{2}$')
        c.line((x - sep, dt), (x + sep, dt))
        c.uput(sep, 'l', (x - sep, dt), r'$1$')

        # SE and solution propagation.
        sep = 0.05
        for se in sol.selms(odd_plane=False):
            c.selm(se, 0,
                   sep=sep, linestyle='dotted', dotsep='1pt', linecolor='red')
        for se in sol.selms(odd_plane=True):
            c.selm(se, hdt,
                   sep=sep, linestyle='dotted', dotsep='1pt', linecolor='blue')

        return c

    @staticmethod
    def draw_cce():

        c = mm.onedim.draw.PstCanvas(unit='2cm', extent=(-2.4, -0.5, 2.4, 1.5))
        c.set(linewidth='1pt')
        c.set(linecolor='black')
        c.frame((-1, 0), (1, 1))
        c.line((0, 0), (0, 1), linestyle='dashed')
        c.frame((-0.95, 0.05), (-0.05, 0.95), linestyle='dotted',
                linecolor='red')
        c.rput('tr', (-0.1, 0.9), r'$\mathrm{CE}_-$')
        c.frame((0.05, 0.05), (0.95, 0.95), linestyle='dotted',
                linecolor='red')
        c.rput('tr', (0.9, 0.9), r'$\mathrm{CE}_+$')
        c.frame((-1.05, -0.05), (1.05, 1.05), linestyle='dotted',
                linecolor='red')
        c.rput('tr', (-1.1, 0.9), r'$\mathrm{CE}$')
        c.dots((0, 1), (-1, 1), (-1, 0), (0, 0), (1, 0), (1, 1), (0, 1),
               dotstyle='*')
        c.uput(0.1, 'u', (0, 1), r'A $(x_j,t^n)$')
        c.uput(0.1, 'ul', (-1, 1), r'B')
        c.uput(0.1, 'dl', (-1, 0),
               r'$(x_{j-\frac{1}{2}},t^{n-\frac{1}{2}})$ C')
        c.uput(0.1, 'd', (0, 0), r'D')
        c.uput(0.1, 'dr', (1, 0), r'E $(x_{j+\frac{1}{2}},t^{n-\frac{1}{2}})$')
        c.uput(0.1, 'ur', (1, 1), r'F')
        c.line((0, 0), (0, 1), linestyle='dashed')

        return c


class DrawNonuniSeTC(DrawTB, DrawNonuniSe):

    def test(self):
        self.tex_filename = 'nonuni_se.tex'
        self.tex_output_dir = ''  # Set to a directory for tex debug output.
        self.golden = r'''\psset{unit=2cm}
\begin{pspicture}(-2.45,-1.45)(4.675,2)

\psset{linewidth=1pt}
\psset{linecolor=black}
\psline[linewidth=0.5pt](-1,0)(-1,1.5)
\psline[linewidth=0.5pt](0,0)(0,1.5)
\psline[linewidth=0.5pt](2,0)(2,1.5)
\psline[linewidth=0.5pt](3.5,0)(3.5,1.5)
\psline[linewidth=0.5pt](-1,0)(3.5,0)
\psline[linewidth=0.5pt](-1,0.75)(3.5,0.75)
\psline[linewidth=0.5pt](-1,1.5)(3.5,1.5)
\psline[linewidth=0.5pt](-1,-0.9)(3.5,-0.9)
\uput{0.05}[l](-1.05,-0.9){$j$}
\uput{0.05}[r](3.55,-0.9){$x$}
\psline(-1,-0.95)(-1,-0.85)
\uput{0.05}[d](-1,-0.95){$0$}
\psline(0,-0.95)(0,-0.85)
\uput{0.05}[d](0,-0.95){$1$}
\psline(2,-0.95)(2,-0.85)
\uput{0.05}[d](2,-0.95){$2$}
\psline(3.5,-0.95)(3.5,-0.85)
\uput{0.05}[d](3.5,-0.95){$3$}
\psline(-0.5,-0.925)(-0.5,-0.875)
\uput{0.05}[d](-0.5,-0.95){$\frac{1}{2}$}
\psline(1,-0.925)(1,-0.875)
\uput{0.05}[d](1,-0.95){$\frac{3}{2}$}
\psline(2.75,-0.925)(2.75,-0.875)
\uput{0.05}[d](2.75,-0.95){$\frac{5}{2}$}
\psline[linewidth=0.5pt](-1.9,0)(-1.9,1.5)
\uput{0.05}[d](-1.9,-0.05){$n$}
\uput{0.05}[u](-1.9,1.55){$t$}
\psline(-1.95,0)(-1.85,0)
\uput{0.05}[l](-1.95,0){$0$}
\psline(-1.925,0.75)(-1.875,0.75)
\uput{0.05}[l](-1.95,0.75){$\frac{1}{2}$}
\psline(-1.95,1.5)(-1.85,1.5)
\uput{0.05}[l](-1.95,1.5){$1$}
\psdots[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=red](-1,0)
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=red](-0.9625,0.05625)(-0.9625,0.65625)
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=red](-1.0375,0.05625)(-1.0375,0.65625)
\psarc[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=red](-1,0.675){0.0375}{0}{180}
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=red](-0.9625,-0.05625)(-0.9625,-0.65625)
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=red](-1.0375,-0.05625)(-1.0375,-0.65625)
\psarc[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=red](-1,-0.675){0.0375}{180}{0}
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=red](-0.94375,0.0375)(-0.59375,0.0375)
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=red](-0.94375,-0.0375)(-0.59375,-0.0375)
\psarc[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=red](-0.575,0){0.0375}{270}{90}
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=red](-1.05625,0.0375)(-1.40625,0.0375)
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=red](-1.05625,-0.0375)(-1.40625,-0.0375)
\psarc[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=red](-1.425,0){0.0375}{90}{270}
\psdots[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=red](0.25,0)
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=red](0.0375,0.05625)(0.0375,0.65625)
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=red](-0.0375,0.05625)(-0.0375,0.65625)
\psarc[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=red](0,0.675){0.0375}{0}{180}
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=red](0.0375,-0.05625)(0.0375,-0.65625)
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=red](-0.0375,-0.05625)(-0.0375,-0.65625)
\psarc[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=red](0,-0.675){0.0375}{180}{0}
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=red](0.05625,0.0375)(0.90625,0.0375)
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=red](0.05625,-0.0375)(0.90625,-0.0375)
\psarc[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=red](0.925,0){0.0375}{270}{90}
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=red](-0.05625,0.0375)(-0.40625,0.0375)
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=red](-0.05625,-0.0375)(-0.40625,-0.0375)
\psarc[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=red](-0.425,0){0.0375}{90}{270}
\psdots[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=red](1.875,0)
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=red](2.0375,0.05625)(2.0375,0.65625)
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=red](1.9625,0.05625)(1.9625,0.65625)
\psarc[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=red](2,0.675){0.0375}{0}{180}
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=red](2.0375,-0.05625)(2.0375,-0.65625)
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=red](1.9625,-0.05625)(1.9625,-0.65625)
\psarc[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=red](2,-0.675){0.0375}{180}{0}
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=red](2.05625,0.0375)(2.65625,0.0375)
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=red](2.05625,-0.0375)(2.65625,-0.0375)
\psarc[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=red](2.675,0){0.0375}{270}{90}
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=red](1.94375,0.0375)(1.09375,0.0375)
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=red](1.94375,-0.0375)(1.09375,-0.0375)
\psarc[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=red](1.075,0){0.0375}{90}{270}
\psdots[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=red](3.5,0)
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=red](3.5375,0.05625)(3.5375,0.65625)
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=red](3.4625,0.05625)(3.4625,0.65625)
\psarc[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=red](3.5,0.675){0.0375}{0}{180}
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=red](3.5375,-0.05625)(3.5375,-0.65625)
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=red](3.4625,-0.05625)(3.4625,-0.65625)
\psarc[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=red](3.5,-0.675){0.0375}{180}{0}
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=red](3.55625,0.0375)(4.15625,0.0375)
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=red](3.55625,-0.0375)(4.15625,-0.0375)
\psarc[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=red](4.175,0){0.0375}{270}{90}
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=red](3.44375,0.0375)(2.84375,0.0375)
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=red](3.44375,-0.0375)(2.84375,-0.0375)
\psarc[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=red](2.825,0){0.0375}{90}{270}
\psdots[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=blue](-0.5,0.75)
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=blue](-0.4625,0.80625)(-0.4625,1.40625)
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=blue](-0.5375,0.80625)(-0.5375,1.40625)
\psarc[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=blue](-0.5,1.425){0.0375}{0}{180}
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=blue](-0.4625,0.69375)(-0.4625,0.09375)
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=blue](-0.5375,0.69375)(-0.5375,0.09375)
\psarc[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=blue](-0.5,0.075){0.0375}{180}{0}
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=blue](-0.44375,0.7875)(-0.09375,0.7875)
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=blue](-0.44375,0.7125)(-0.09375,0.7125)
\psarc[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=blue](-0.075,0.75){0.0375}{270}{90}
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=blue](-0.55625,0.7875)(-0.90625,0.7875)
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=blue](-0.55625,0.7125)(-0.90625,0.7125)
\psarc[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=blue](-0.925,0.75){0.0375}{90}{270}
\psdots[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=blue](1,0.75)
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=blue](1.0375,0.80625)(1.0375,1.40625)
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=blue](0.9625,0.80625)(0.9625,1.40625)
\psarc[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=blue](1,1.425){0.0375}{0}{180}
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=blue](1.0375,0.69375)(1.0375,0.09375)
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=blue](0.9625,0.69375)(0.9625,0.09375)
\psarc[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=blue](1,0.075){0.0375}{180}{0}
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=blue](1.05625,0.7875)(1.90625,0.7875)
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=blue](1.05625,0.7125)(1.90625,0.7125)
\psarc[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=blue](1.925,0.75){0.0375}{270}{90}
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=blue](0.94375,0.7875)(0.09375,0.7875)
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=blue](0.94375,0.7125)(0.09375,0.7125)
\psarc[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=blue](0.075,0.75){0.0375}{90}{270}
\psdots[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=blue](2.75,0.75)
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=blue](2.7875,0.80625)(2.7875,1.40625)
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=blue](2.7125,0.80625)(2.7125,1.40625)
\psarc[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=blue](2.75,1.425){0.0375}{0}{180}
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=blue](2.7875,0.69375)(2.7875,0.09375)
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=blue](2.7125,0.69375)(2.7125,0.09375)
\psarc[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=blue](2.75,0.075){0.0375}{180}{0}
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=blue](2.80625,0.7875)(3.40625,0.7875)
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=blue](2.80625,0.7125)(3.40625,0.7125)
\psarc[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=blue](3.425,0.75){0.0375}{270}{90}
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=blue](2.69375,0.7875)(2.09375,0.7875)
\psline[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=blue](2.69375,0.7125)(2.09375,0.7125)
\psarc[linewidth=1pt,linestyle=dotted,dotsep=1pt,linecolor=blue](2.075,0.75){0.0375}{90}{270}

\end{pspicture}'''
        self._compare()


class DrawSeFlux(DrawBase):
    """
    An example for a command-line class to draw for the flux on solution
    element.
    """

    @staticmethod
    def draw():
        xloc = np.array([-1, 0, 2, 3.5])
        grid = mm.spacetime.Grid(xloc=xloc)
        sol = mm.spacetime.Solver(grid=grid, nvar=1, time_increment=1)

        dt = sol.dt
        hdt = dt / 2
        qdt = hdt / 2

        c = mm.onedim.draw.PstCanvas(unit='5cm',
                                     padding=[0.75, 0.25, 0.7, 0.25])
        c.set(linewidth='1pt')
        c.set(linecolor='black')

        ce = sol.celm(ielm=1, odd_plane=False)
        linewidth = '0.5pt'

        # SEs.
        sexn = ce.selm_xn
        c.line((sexn.x, 0), (sexn.x, hdt), linewidth=linewidth,
               linecolor='red')
        c.line((sexn.x, 0), (sexn.xpos, 0), linewidth=linewidth,
               linecolor='red')

        c.dots((sexn.x, 0), dotstyle='o')
        c.uput(0.05, 'dl', (sexn.x, 0), r'$x_{j-\frac{1}{2}}$')
        c.dots((sexn.xctr, 0), dotstyle='triangle')
        c.uput(0.05, 'u', (sexn.xctr, 0), r'$x^s_{j-\frac{1}{2}}$')

        sexp = ce.selm_xp
        c.line((sexp.x, 0), (sexp.x, hdt), linewidth=linewidth,
               linecolor='blue')
        c.line((sexp.xneg, 0), (sexp.x, 0), linewidth=linewidth,
               linecolor='blue')

        c.dots((sexp.x, 0), dotstyle='o')
        c.uput(0.05, 'dr', (sexp.x, 0), r'$x_{j+\frac{1}{2}}$')
        c.dots((sexp.xctr, 0), dotstyle='triangle')
        c.uput(0.05, 'u', (sexp.xctr, 0), r'$x^s_{j+\frac{1}{2}}$')

        c.dots((sexn.xpos, 0), dotstyle='square')
        assert sexn.xpos == sexp.xneg
        c.uput(0.05, 'u', (sexn.xpos, 0),
               r'$x^+_{j-\frac{1}{2}} = x^-_{j+\frac{1}{2}}$')

        setp = ce.selm_tp
        c.line((setp.xneg, hdt), (setp.x, hdt),
               linewidth=linewidth, linecolor='orange')
        c.line((setp.x, hdt), (setp.xpos, hdt),
               linewidth=linewidth, linecolor='orange')

        c.dots((setp.x, hdt), dotstyle='o')
        c.dots((setp.xctr, setp.hdt), dotstyle='triangle')
        c.uput(0.05, 'd', (setp.x, hdt), r'$x_j=x^s_j$')

        # Fluxes.
        vlen = hdt / 6
        sep = 0.015
        # \Delta x^+_{j-\frac{1}{2}}
        c.line(((sexn.x + sexn.xpos) / 2, 0),
               ((sexn.x + sexn.xpos) / 2, -vlen),
               arrows='->', linecolor='red')
        c.uput(0, 'd', ((sexn.x + sexn.xpos) / 2, -vlen),
               r'{\color{red}$(\mathbf{h}^*)^n_{j-\frac{1}{2},+}'
               r'\cdot(0, -\Delta x^+_{j-\frac{1}{2}})$}')
        # \Delta x^-_{j+\frac{1}{2}}
        c.line(((sexp.xneg + sexp.x) / 2, 0),
               ((sexp.xneg + sexp.x) / 2, -vlen),
               arrows='->', linecolor='blue')
        c.uput(0, 'd', ((sexp.xneg + sexp.x) / 2, -vlen),
               r'{\color{blue}$(\mathbf{h}^*)^n_{j+\frac{1}{2},-}'
               r'\cdot(0, -\Delta x^-_{j+\frac{1}{2}})$}')
        # \Delta x_j
        c.line((setp.xctr, hdt + sep), (setp.xctr, hdt + 2 * vlen),
               arrows='->', linecolor='orange')
        c.uput(0, 'u', (setp.xctr, hdt + 2 * vlen),
               r'{\color{orange}$\mathbf{h}^{n+\frac{1}{2}}_{j}'
               r'\cdot(0, \Delta x_j)$}')
        # \Delta t^n_{j-\frac{1}{2}}
        c.line((sexn.x, qdt), (sexn.x - vlen, qdt), arrows='->',
               linecolor='red')
        c.uput(0, 'ul', (sexn.x - vlen, qdt),
               r'{\color{red}$(\mathbf{h}^*)^{n,+}_{j-\frac{1}{2}}'
               r'\cdot(-\frac{\Delta t}{2}, 0)$}')
        # \Delta t^n_{j+\frac{1}{2}}
        c.line((sexp.x, qdt), (sexp.x + vlen, qdt), arrows='->',
               linecolor='blue')
        c.uput(0, 'ur', (sexp.x + vlen, qdt),
               r'{\color{blue}$(\mathbf{h}^*)^{n,+}_{j+\frac{1}{2}}'
               r'\cdot(\frac{\Delta t}{2}, 0)$}')

        return c


class DrawSeFluxTC(DrawTB, DrawSeFlux):

    def test(self):
        self.tex_filename = 'se_flux.tex'
        self.tex_output_dir = ''  # Set to a directory for tex debug output.
        self.golden = r'''\psset{unit=5cm}
\begin{pspicture}(-0.833333,-0.333333)(2.78333,0.916667)

\psset{linewidth=1pt}
\psset{linecolor=black}
\psline[linewidth=0.5pt,linecolor=red](0,0)(0,0.5)
\psline[linewidth=0.5pt,linecolor=red](0,0)(1,0)
\psdots[dotstyle=o](0,0)
\uput{0.05}[dl](0,0){$x_{j-\frac{1}{2}}$}
\psdots[dotstyle=triangle](0.25,0)
\uput{0.05}[u](0.25,0){$x^s_{j-\frac{1}{2}}$}
\psline[linewidth=0.5pt,linecolor=blue](2,0)(2,0.5)
\psline[linewidth=0.5pt,linecolor=blue](1,0)(2,0)
\psdots[dotstyle=o](2,0)
\uput{0.05}[dr](2,0){$x_{j+\frac{1}{2}}$}
\psdots[dotstyle=triangle](1.875,0)
\uput{0.05}[u](1.875,0){$x^s_{j+\frac{1}{2}}$}
\psdots[dotstyle=square](1,0)
\uput{0.05}[u](1,0){$x^+_{j-\frac{1}{2}} = x^-_{j+\frac{1}{2}}$}
\psline[linewidth=0.5pt,linecolor=orange](0,0.5)(1,0.5)
\psline[linewidth=0.5pt,linecolor=orange](1,0.5)(2,0.5)
\psdots[dotstyle=o](1,0.5)
\psdots[dotstyle=triangle](1,0.5)
\uput{0.05}[d](1,0.5){$x_j=x^s_j$}
\psline[arrows=->,linecolor=red](0.5,0)(0.5,-0.0833333)
\uput{0}[d](0.5,-0.0833333){{\color{red}$(\mathbf{h}^*)^n_{j-\frac{1}{2},+}\cdot(0, -\Delta x^+_{j-\frac{1}{2}})$}}
\psline[arrows=->,linecolor=blue](1.5,0)(1.5,-0.0833333)
\uput{0}[d](1.5,-0.0833333){{\color{blue}$(\mathbf{h}^*)^n_{j+\frac{1}{2},-}\cdot(0, -\Delta x^-_{j+\frac{1}{2}})$}}
\psline[arrows=->,linecolor=orange](1,0.515)(1,0.666667)
\uput{0}[u](1,0.666667){{\color{orange}$\mathbf{h}^{n+\frac{1}{2}}_{j}\cdot(0, \Delta x_j)$}}
\psline[arrows=->,linecolor=red](0,0.25)(-0.0833333,0.25)
\uput{0}[ul](-0.0833333,0.25){{\color{red}$(\mathbf{h}^*)^{n,+}_{j-\frac{1}{2}}\cdot(-\frac{\Delta t}{2}, 0)$}}
\psline[arrows=->,linecolor=blue](2,0.25)(2.08333,0.25)
\uput{0}[ur](2.08333,0.25){{\color{blue}$(\mathbf{h}^*)^{n,+}_{j+\frac{1}{2}}\cdot(\frac{\Delta t}{2}, 0)$}}

\end{pspicture}'''
        self._compare()

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
