# Copyright (c) 2026, Yung-Yu Chen <yyc@solvcon.net>
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
Mesh and boundary tagging for the oblique-shock reflection.

A uniform supersonic stream enters from the left over a slip wall whose
bottom turns into a wedge inclined by a fixed angle.  The wedge deflects the
flow, an oblique shock forms at the wedge tip and reflects off the flat top
slip wall, and the flow leaves through the non-reflective outflow on the
right.  This module owns the programmatic mesh builder and the geometric
boundary classifier shared by the unit tests, the pilot GUI, and the
forthcoming Euler-solver driver.
"""

import math

from ... import core

__all__ = [
    'ObliqueShockMesher',
]


class ObliqueShockMesher(object):
    """Generate the mesh for the oblique-shock reflection.

    The rectangular-like domain runs from the lower-left corner ``ll`` to
    the upper-right corner ``ur``.  The bottom edge is flat up to ``x_ramp``
    and then ramps up at ``wedge_angle`` degrees.  The wedge top must stay
    below the domain height (``ur[1] - ll[1]``); otherwise the grid columns
    would invert and the mesh silently self-overlap, so the constructor
    raises :class:`ValueError` instead.
    """

    def __init__(self, nx=48, ny=16, ll=(0.0, 0.0), ur=(3.0, 1.0),
                 x_ramp=1.5, wedge_angle=15.0):
        self.nx = nx
        self.ny = ny
        (self.x0, self.y0), (self.x1, self.y1) = ll, ur
        self.x_ramp = x_ramp
        self.tan_theta = math.tan(math.radians(wedge_angle))
        wedge_top = (self.x1 - x_ramp) * self.tan_theta
        if wedge_top >= self.y1 - self.y0:
            raise ValueError(
                f"wedge top (ur[0] - x_ramp) * tan(wedge_angle) = "
                f"{wedge_top:g} must stay below the domain height "
                f"{self.y1 - self.y0:g}")

    def _node(self, it, jt):
        x = self.x0 + it * (self.x1 - self.x0) / self.nx
        yb = (self.y0 if x <= self.x_ramp
              else self.y0 + (x - self.x_ramp) * self.tan_theta)
        return x, yb + (self.y1 - yb) * jt / self.ny

    def _nid(self, it, jt):
        return jt * (self.nx + 1) + it

    def _box(self, it, jt):
        return (self._nid(it, jt), self._nid(it + 1, jt),
                self._nid(it + 1, jt + 1), self._nid(it, jt + 1))

    def make_mesh(self, cell_type='quad'):
        """Build a :class:`~modmesh.core.StaticMesh` of the selected flavor.

        ``cell_type`` selects the element shape:
        - ``'quad'`` keeps one quadrilateral per grid box,
        - ``'triangle'`` cuts each box along its lower-left-to-upper-right
          diagonal into two triangles (flipping the diagonal at two corners
          so no triangle carries two boundary faces), and
        - ``'unstructured'`` Delaunay-triangulates the same boundary nodes
          plus jittered interior points into an irregular (but deterministic)
          triangulation, refined to the same one-boundary-face-per-cell rule.

        Both triangular flavors keep at most one boundary face per cell,
        which the corner ``'quad'`` cells cannot.  All flavors share the
        boundary layout (nx segments on the bottom and top, ny on the left
        and right) and produce counter-clockwise cells.  The returned mesh
        has ``ndcrd``/``cltpn``/``clnds`` filled and ``build_interior`` /
        ``build_boundary`` / ``build_ghost`` run.
        """
        nx, ny = self.nx, self.ny
        if cell_type in ('quad', 'triangle'):
            nodes = core.PointPadFp64(ndim=2)
            for jt in range(ny + 1):
                for it in range(nx + 1):
                    nodes.append(*self._node(it, jt))
            if cell_type == 'quad':
                tpn = core.StaticMesh.QUADRILATERAL
                cells = [(4,) + self._box(it, jt)
                         for jt in range(ny) for it in range(nx)]
            else:
                tpn = core.StaticMesh.TRIANGLE
                cells = []
                for jt in range(ny):
                    for it in range(nx):
                        ll, lr, ur, ul = self._box(it, jt)
                        # Split along the lower-left-to-upper-right diagonal,
                        # except at the upper-left and lower-right domain
                        # corners, whose corner cell would otherwise carry
                        # the two boundary edges meeting there.
                        if (it, jt) in ((0, ny - 1), (nx - 1, 0)):
                            cells += [(3, ll, lr, ul), (3, lr, ur, ul)]
                        else:
                            cells += [(3, ll, lr, ur), (3, ll, ur, ul)]
        elif cell_type == 'unstructured':
            tpn = core.StaticMesh.TRIANGLE
            nodes = self._jitter_points()
            cells = [(3,) + tri
                     for tri in self._split_double_boundary(nodes)]
        else:
            raise ValueError(f"unknown cell_type '{cell_type}'")
        mh = core.StaticMesh(ndim=2, nnode=len(nodes), nface=0,
                             ncell=len(cells))
        mh.ndcrd[:, :] = nodes.pack_array().ndarray
        mh.cltpn.fill(tpn)
        mh.clnds[:, :len(cells[0])] = cells
        mh.build_interior(do_metric=True)
        mh.build_boundary()
        mh.build_ghost()
        return mh

    def _jitter_points(self):
        """Collect the unstructured point cloud in a :class:`PointPad`.

        The boundary keeps the structured node layout (a counter-clockwise
        outline walk), so classification is identical across flavors.
        Interior nodes are displaced in logical (it, jt) space by an
        RNG-free phase -- deterministic, yet irregular -- bounded by 0.3
        logical units per axis to stay well inside the domain.
        """
        nx, ny = self.nx, self.ny
        pad = core.PointPadFp64(ndim=2)
        for it in range(nx + 1):
            pad.append(*self._node(it, 0))
        for jt in range(1, ny + 1):
            pad.append(*self._node(nx, jt))
        for it in range(nx - 1, -1, -1):
            pad.append(*self._node(it, ny))
        for jt in range(ny - 1, 0, -1):
            pad.append(*self._node(0, jt))
        for jt in range(1, ny):
            for it in range(1, nx):
                pad.append(*self._node(
                    it + 0.3 * math.sin(12.9898 * it + 78.233 * jt),
                    jt + 0.3 * math.cos(26.651 * it + 41.347 * jt)))
        return pad

    @staticmethod
    def _circumcircle(pts, ia, ib, ic):
        """Circumcenter and squared circumradius of triangle (ia, ib, ic)."""
        ax, ay = pts[ia]
        bx, by = pts[ib]
        cx, cy = pts[ic]
        dd = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
        a2, b2, c2 = ax * ax + ay * ay, bx * bx + by * by, cx * cx + cy * cy
        ux = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / dd
        uy = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / dd
        return ux, uy, (ax - ux) ** 2 + (ay - uy) ** 2

    @classmethod
    def _triangulate(cls, pad):
        """Bowyer-Watson Delaunay triangulation of the :class:`PointPad`
        ``pad``; returns CCW index triples.

        Seed a far-away super-triangle, insert one point at a time (carve
        the triangles whose circumcircle strictly contains it, fan the
        cavity rim to the point), then drop the triangles touching a super
        vertex.  On-circle points count as outside, keeping the cavity well
        defined under the cocircular ties of the regular boundary layout.
        Naive O(n^2).

        References:
        - A. Bowyer, "Computing Dirichlet tessellations", The Computer
          Journal 24(2):162-166, 1981.
          https://doi.org/10.1093/comjnl/24.2.162
        - D. F. Watson, "Computing the n-dimensional Delaunay tessellation
          with application to Voronoi polytopes", The Computer Journal
          24(2):167-172, 1981.  https://doi.org/10.1093/comjnl/24.2.167
        - https://en.wikipedia.org/wiki/Bowyer%E2%80%93Watson_algorithm
        """
        npt = len(pad)
        pts = [(pad.x_at(ip), pad.y_at(ip)) for ip in range(npt)]
        xs = [px for px, py in pts]
        ys = [py for px, py in pts]
        xmid = (min(xs) + max(xs)) / 2.0
        ymid = (min(ys) + max(ys)) / 2.0
        span = max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
        pts += [(xmid - 64.0 * span, ymid - 32.0 * span),
                (xmid + 64.0 * span, ymid - 32.0 * span),
                (xmid, ymid + 64.0 * span)]
        # Triangles keyed by CCW vertex triple, valued by circumcircle.
        cc = {(npt, npt + 1, npt + 2):
              cls._circumcircle(pts, npt, npt + 1, npt + 2)}
        for ip in range(npt):
            px, py = pts[ip]
            bad = [tri for tri, (ux, uy, rr) in cc.items()
                   if (px - ux) ** 2 + (py - uy) ** 2 < rr * (1.0 - 1e-12)]
            dedges = set()
            for tri in bad:
                ta, tb, tc = tri
                dedges.update(((ta, tb), (tb, tc), (tc, ta)))
                del cc[tri]
            # The cavity rim is the directed edges whose reverse was not
            # carved; they run CCW, so fanning them to the point keeps the
            # triangles CCW.
            for ea, eb in dedges:
                if (eb, ea) not in dedges:
                    cc[(ea, eb, ip)] = cls._circumcircle(pts, ea, eb, ip)
        tris = []
        for tri in cc:
            if max(tri) >= npt:
                continue
            (ax, ay), (bx, by), (cx, cy) = (pts[iv] for iv in tri)
            if (bx - ax) * (cy - ay) - (cx - ax) * (by - ay) < 0.0:
                tri = (tri[0], tri[2], tri[1])
            tris.append(tri)
        return tris

    @classmethod
    def _split_double_boundary(cls, pad):
        """Triangulate the :class:`PointPad` ``pad``, appending Steiner
        points into it until no cell touches the boundary with more than
        one face; returns the triangles.

        A cell with two single-shared (boundary) edges is a corner ear with
        a single interior neighbour, which the CESE solver dislikes.  A
        Steiner point at the ear's centroid keeps the next Delaunay pass
        from rebuilding it; the boundary nodes never move, so the
        classification is unchanged.  One pass suffices here; the cap only
        bounds pathological input.
        """
        for _ in range(16):
            tris = cls._triangulate(pad)
            shared = {}
            for tri in tris:
                for it in range(3):
                    edge = frozenset((tri[it], tri[(it + 1) % 3]))
                    shared[edge] = shared.get(edge, 0) + 1
            extra = []
            for tri in tris:
                on_boundary = sum(
                    1 for it in range(3)
                    if shared[frozenset((tri[it], tri[(it + 1) % 3]))] == 1)
                if on_boundary >= 2:
                    extra.append((sum(pad.x_at(iv) for iv in tri) / 3.0,
                                  sum(pad.y_at(iv) for iv in tri) / 3.0))
            if not extra:
                return tris
            for xc, yc in extra:
                pad.append(xc, yc)
        raise RuntimeError("boundary-cell refinement did not converge")

    @staticmethod
    def classify_boundary(mh, tol=1e-9):
        """Bucket the boundary faces of ``mh`` by physical role.

        A boundary face has no neighbour cell, i.e. ``fccls(ifc, 1) < 0``.
        Each is classified by its face-centre ``fccnd`` x position and
        outward ``fcnml`` direction: the left edge (``x == xmin``, normal
        pointing in -x) is the supersonic inlet, the right edge
        (``x == xmax``, normal in +x) is the non-reflective outflow, and
        every remaining face -- the deflecting bottom wedge and the
        reflecting top -- is a slip wall.  ``xmin``/``xmax`` come from the
        boundary face centres because ``ndcrd`` also carries extrapolated
        ghost-node coordinates that overshoot the real edges.

        Returns ``(inlet, walls, outflow)`` as sorted face-index lists
        ready to feed ``add_inlet`` / ``add_slipwall`` / ``add_nonrefl``.
        """
        bfaces = [ifc for ifc in range(mh.nface) if mh.fccls[ifc, 1] < 0]
        xcs = [mh.fccnd[ifc, 0] for ifc in bfaces]
        xmin, xmax = min(xcs), max(xcs)
        inlet, walls, outflow = [], [], []
        for ifc in bfaces:
            xc = mh.fccnd[ifc, 0]
            nx = mh.fcnml[ifc, 0]
            if abs(xc - xmin) <= tol and nx < 0.0:
                inlet.append(ifc)
            elif abs(xc - xmax) <= tol and nx > 0.0:
                outflow.append(ifc)
            else:
                walls.append(ifc)
        return sorted(inlet), sorted(walls), sorted(outflow)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
