# Copyright (c) 2025, Yung-Yu Chen <yyc@solvcon.net>
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
Generate interactive HTML/SVG visualizations of polygon boolean operations
and their trapezoidal decompositions.

Usage::

    import modmesh as mm
    from modmesh.plot.polygon_boolean import BooleanVisualizer

    pad = mm.PolygonPadFp64(ndim=2)
    p1 = pad.add_polygon([...])
    p2 = pad.add_polygon([...])

    viz = BooleanVisualizer(pad, p1, p2)
    viz.save("boolean_viz.html")
"""

import modmesh as mm


class BooleanVisualizer:
    """Generate an HTML page with SVG visualizations of polygon boolean
    operations showing input polygons, trapezoidal decompositions, and
    boolean results."""

    # Color palette (dark theme)
    P1_COLOR = "#5b9ef0"
    P2_COLOR = "#f05545"
    RESULT_COLOR = "#3ddc84"
    TRAP_COLORS = ["#6ec6ff", "#ff8a65", "#aed581", "#ce93d8",
                   "#4dd0e1", "#fff176", "#f48fb1", "#90a4ae"]

    def __init__(self, pad, polygon1, polygon2):
        self._pad = pad
        self._p1 = polygon1
        self._p2 = polygon2
        self._compute()

    def _compute(self):
        """Run decompositions and boolean operations."""
        pad = self._pad
        p1, p2 = self._p1, self._p2

        # Decompose both polygons
        self._trap_range1 = pad.decompose_to_trapezoid(p1.polygon_id)
        self._trap_range2 = pad.decompose_to_trapezoid(p2.polygon_id)
        self._trap_pad = pad.decomposed_trapezoids()

        # Boolean operations
        self._union = pad.boolean_union(p1, p2)
        self._intersection = pad.boolean_intersection(p1, p2)
        self._difference = pad.boolean_difference(p1, p2)

    def _get_polygon_points(self, polygon):
        """Extract (x, y) points from a Polygon3d."""
        return [(polygon.get_node(i).x, polygon.get_node(i).y)
                for i in range(polygon.nnode)]

    def _get_trapezoid_points(self, trap_pad, idx):
        """Extract the 4 corners of trapezoid idx as (x,y) tuples.
        Returns in CCW order: p0 (bottom-left), p1 (bottom-right),
        p2 (top-right), p3 (top-left)."""
        return [
            (trap_pad.x0(idx), trap_pad.y0(idx)),
            (trap_pad.x1(idx), trap_pad.y1(idx)),
            (trap_pad.x2(idx), trap_pad.y2(idx)),
            (trap_pad.x3(idx), trap_pad.y3(idx)),
        ]

    def _get_all_bounds(self):
        """Compute bounding box covering all geometry."""
        all_x, all_y = [], []
        for poly in [self._p1, self._p2]:
            for i in range(poly.nnode):
                n = poly.get_node(i)
                all_x.append(n.x)
                all_y.append(n.y)
        margin = 0.5
        return (min(all_x) - margin, min(all_y) - margin,
                max(all_x) + margin, max(all_y) + margin)

    def _world_to_svg(self, x, y, x_min, y_min, x_max, y_max, width, height):
        """Convert world coordinates to SVG coordinates (Y-flipped)."""
        sx = (x - x_min) / (x_max - x_min) * width
        sy = height - (y - y_min) / (y_max - y_min) * height
        return sx, sy

    def _polygon_to_svg_path(self, points, x_min, y_min, x_max, y_max,
                             w, h):
        """Convert polygon points to SVG path string."""
        parts = []
        for i, (px, py) in enumerate(points):
            sx, sy = self._world_to_svg(px, py, x_min, y_min, x_max, y_max,
                                        w, h)
            cmd = "M" if i == 0 else "L"
            parts.append(f"{cmd}{sx:.2f},{sy:.2f}")
        parts.append("Z")
        return " ".join(parts)

    def _render_svg_panel(self, title, polygons_spec, traps_spec,
                          x_min, y_min, x_max, y_max,
                          width=400, height=400):
        """Render one SVG panel.

        polygons_spec: list of (points, stroke_color, fill_color, fill_opacity)
        traps_spec: list of (points, stroke_color, fill_color, fill_opacity)
        """
        pad_left, pad_top = 10, 30
        inner_w, inner_h = width - 2 * pad_left, height - pad_top - pad_left
        svg_w, svg_h = width, height

        lines = []
        lines.append(f'<svg viewBox="0 0 {svg_w} {svg_h}" '
                     f'width="{svg_w}" height="{svg_h}" '
                     f'xmlns="http://www.w3.org/2000/svg">')

        # Title
        lines.append(f'<text x="{svg_w // 2}" y="20" '
                     f'text-anchor="middle" font-size="14" '
                     f'fill="#ddd" font-family="sans-serif" '
                     f'font-weight="600">{title}</text>')

        # Background
        lines.append(f'<rect x="{pad_left}" y="{pad_top}" '
                     f'width="{inner_w}" height="{inner_h}" '
                     f'fill="#1e1e30" rx="4"/>')

        # Grid lines
        def w2s(x, y):
            return self._world_to_svg(x, y, x_min, y_min, x_max, y_max,
                                      inner_w, inner_h)

        # Draw integer grid
        import math
        for gx in range(math.floor(x_min), math.ceil(x_max) + 1):
            sx, _ = w2s(gx, 0)
            lines.append(
                f'<line x1="{pad_left + sx:.1f}" y1="{pad_top}" '
                f'x2="{pad_left + sx:.1f}" y2="{pad_top + inner_h}" '
                f'stroke="#333" stroke-width="0.5"/>')
            lines.append(
                f'<text x="{pad_left + sx:.1f}" '
                f'y="{pad_top + inner_h + 12}" '
                f'text-anchor="middle" font-size="10" '
                f'fill="#777" font-family="monospace">{gx}</text>')
        for gy in range(math.floor(y_min), math.ceil(y_max) + 1):
            _, sy = w2s(0, gy)
            lines.append(
                f'<line x1="{pad_left}" y1="{pad_top + sy:.1f}" '
                f'x2="{pad_left + inner_w}" y2="{pad_top + sy:.1f}" '
                f'stroke="#333" stroke-width="0.5"/>')
            lines.append(
                f'<text x="{pad_left - 3}" y="{pad_top + sy + 3:.1f}" '
                f'text-anchor="end" font-size="10" '
                f'fill="#777" font-family="monospace">{gy}</text>')

        # Trapezoids
        for pts, stroke, fill, opacity in traps_spec:
            path = self._polygon_to_svg_path(pts, x_min, y_min, x_max, y_max,
                                             inner_w, inner_h)
            lines.append(
                f'<path d="{path}" fill="{fill}" fill-opacity="{opacity}" '
                f'stroke="{stroke}" stroke-width="1" '
                f'stroke-dasharray="3,2" '
                f'transform="translate({pad_left},{pad_top})"/>')

        # Polygons
        for pts, stroke, fill, opacity in polygons_spec:
            path = self._polygon_to_svg_path(pts, x_min, y_min, x_max, y_max,
                                             inner_w, inner_h)
            lines.append(
                f'<path d="{path}" fill="{fill}" fill-opacity="{opacity}" '
                f'stroke="{stroke}" stroke-width="2" '
                f'transform="translate({pad_left},{pad_top})"/>')

        lines.append('</svg>')
        return "\n".join(lines)

    def _render_input_panel(self, x_min, y_min, x_max, y_max):
        """Render the input polygons panel."""
        pts1 = self._get_polygon_points(self._p1)
        pts2 = self._get_polygon_points(self._p2)
        polygons = [
            (pts1, self.P1_COLOR, self.P1_COLOR, 0.2),
            (pts2, self.P2_COLOR, self.P2_COLOR, 0.2),
        ]
        return self._render_svg_panel("Input Polygons", polygons, [],
                                      x_min, y_min, x_max, y_max)

    def _render_decomposition_panel(self, polygon, trap_range, label, color,
                                    x_min, y_min, x_max, y_max):
        """Render a trapezoidal decomposition panel for one polygon."""
        pts = self._get_polygon_points(polygon)
        polygons = [(pts, color, "none", 0)]

        begin, end = trap_range
        traps = []
        for i in range(begin, end):
            trap_pts = self._get_trapezoid_points(self._trap_pad, i)
            cidx = (i - begin) % len(self.TRAP_COLORS)
            traps.append(
                (trap_pts, self.TRAP_COLORS[cidx],
                 self.TRAP_COLORS[cidx], 0.25))

        # Add polygon outline on top
        polygons_with_outline = traps
        outline = [(pts, color, "none", 0)]

        return self._render_svg_panel(
            f"{label} Decomposition ({end - begin} trapezoids)",
            outline, polygons_with_outline,
            x_min, y_min, x_max, y_max)

    def _render_boolean_panel(self, result_pad, op_name,
                              x_min, y_min, x_max, y_max):
        """Render a boolean result panel."""
        pts1 = self._get_polygon_points(self._p1)
        pts2 = self._get_polygon_points(self._p2)

        # Faded input polygons
        polygons = [
            (pts1, self.P1_COLOR, "none", 0),
            (pts2, self.P2_COLOR, "none", 0),
        ]

        # Result trapezoids
        traps = []
        for i in range(result_pad.num_polygons):
            rpoly = result_pad.get_polygon(i)
            rpts = self._get_polygon_points(rpoly)
            traps.append(
                (rpts, self.RESULT_COLOR, self.RESULT_COLOR, 0.35))

        # Compute area
        total_area = 0.0
        for i in range(result_pad.num_polygons):
            total_area += abs(
                result_pad.get_polygon(i).compute_signed_area())

        title = (f"{op_name}: {result_pad.num_polygons} trapezoids, "
                 f"area = {total_area:.2f}")
        return self._render_svg_panel(title, polygons, traps,
                                      x_min, y_min, x_max, y_max)

    def _render_combined_decomposition_panel(self, x_min, y_min,
                                             x_max, y_max):
        """Render both decompositions overlaid."""
        pts1 = self._get_polygon_points(self._p1)
        pts2 = self._get_polygon_points(self._p2)

        begin1, end1 = self._trap_range1
        begin2, end2 = self._trap_range2

        traps = []
        for i in range(begin1, end1):
            trap_pts = self._get_trapezoid_points(self._trap_pad, i)
            traps.append((trap_pts, self.P1_COLOR, self.P1_COLOR, 0.15))
        for i in range(begin2, end2):
            trap_pts = self._get_trapezoid_points(self._trap_pad, i)
            traps.append((trap_pts, self.P2_COLOR, self.P2_COLOR, 0.15))

        polygons = [
            (pts1, self.P1_COLOR, "none", 0),
            (pts2, self.P2_COLOR, "none", 0),
        ]

        n1 = end1 - begin1
        n2 = end2 - begin2
        return self._render_svg_panel(
            f"Both Decompositions (P1: {n1}, P2: {n2} trapezoids)",
            polygons, traps,
            x_min, y_min, x_max, y_max)

    def generate_html(self):
        """Generate the full HTML string."""
        x_min, y_min, x_max, y_max = self._get_all_bounds()

        panels = []
        # Row 1: Input + Combined decomposition
        panels.append(("input",
                        self._render_input_panel(
                            x_min, y_min, x_max, y_max)))
        panels.append(("combined_decomp",
                        self._render_combined_decomposition_panel(
                            x_min, y_min, x_max, y_max)))

        # Row 2: Individual decompositions
        panels.append(("decomp1",
                        self._render_decomposition_panel(
                            self._p1, self._trap_range1, "P1",
                            self.P1_COLOR,
                            x_min, y_min, x_max, y_max)))
        panels.append(("decomp2",
                        self._render_decomposition_panel(
                            self._p2, self._trap_range2, "P2",
                            self.P2_COLOR,
                            x_min, y_min, x_max, y_max)))

        # Row 3: Boolean results
        panels.append(("union",
                        self._render_boolean_panel(
                            self._union, "Union",
                            x_min, y_min, x_max, y_max)))
        panels.append(("intersection",
                        self._render_boolean_panel(
                            self._intersection, "Intersection",
                            x_min, y_min, x_max, y_max)))
        panels.append(("difference",
                        self._render_boolean_panel(
                            self._difference, "Difference (P1 - P2)",
                            x_min, y_min, x_max, y_max)))

        # Build legend info
        p1_area = abs(self._p1.compute_signed_area())
        p2_area = abs(self._p2.compute_signed_area())

        html = f"""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Polygon Boolean Visualization - modmesh</title>
<style>
  body {{
    background: #12121f;
    color: #ccc;
    font-family: 'Segoe UI', system-ui, sans-serif;
    margin: 0;
    padding: 1.5rem;
  }}
  h1 {{
    text-align: center;
    color: #eee;
    font-size: 1.5rem;
    margin-bottom: 0.3rem;
  }}
  .subtitle {{
    text-align: center;
    color: #888;
    font-size: 0.9rem;
    margin-bottom: 1.5rem;
  }}
  .legend {{
    display: flex;
    justify-content: center;
    gap: 2rem;
    margin-bottom: 1.5rem;
    font-size: 0.9rem;
  }}
  .legend-item {{
    display: flex;
    align-items: center;
    gap: 0.4rem;
  }}
  .swatch {{
    width: 16px;
    height: 16px;
    border-radius: 3px;
    border: 1px solid #555;
  }}
  .grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
    gap: 1rem;
    max-width: 1300px;
    margin: 0 auto;
  }}
  .panel {{
    background: #1a1a2e;
    border: 1px solid #333;
    border-radius: 8px;
    padding: 0.5rem;
    display: flex;
    justify-content: center;
  }}
  .panel svg {{
    max-width: 100%;
    height: auto;
  }}
  .info-table {{
    max-width: 800px;
    margin: 1.5rem auto;
    border-collapse: collapse;
    font-size: 0.9rem;
  }}
  .info-table th, .info-table td {{
    border: 1px solid #333;
    padding: 0.4rem 0.8rem;
    text-align: left;
  }}
  .info-table th {{
    background: #222;
    color: #aaa;
  }}
  .info-table td {{
    color: #ccc;
  }}
</style>
</head>
<body>

<h1>Polygon Boolean Operations Visualization</h1>
<p class="subtitle">Trapezoidal decomposition + Y-band sweep &mdash; modmesh</p>

<div class="legend">
  <div class="legend-item">
    <div class="swatch" style="background:{self.P1_COLOR}"></div>
    <span>P1 (area = {p1_area:.2f})</span>
  </div>
  <div class="legend-item">
    <div class="swatch" style="background:{self.P2_COLOR}"></div>
    <span>P2 (area = {p2_area:.2f})</span>
  </div>
  <div class="legend-item">
    <div class="swatch" style="background:{self.RESULT_COLOR}"></div>
    <span>Result</span>
  </div>
</div>

<div class="grid">
"""
        for name, svg in panels:
            html += f'  <div class="panel" id="{name}">\n'
            html += f"    {svg}\n"
            html += "  </div>\n"

        # Summary table
        u_area = sum(abs(self._union.get_polygon(i).compute_signed_area())
                     for i in range(self._union.num_polygons))
        i_area = sum(
            abs(self._intersection.get_polygon(i).compute_signed_area())
            for i in range(self._intersection.num_polygons))
        d_area = sum(
            abs(self._difference.get_polygon(i).compute_signed_area())
            for i in range(self._difference.num_polygons))

        html += f"""\
</div>

<table class="info-table">
  <thead>
    <tr>
      <th>Operation</th>
      <th>Result Trapezoids</th>
      <th>Total Area</th>
      <th>Expected (A+B-I)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Union (P1 | P2)</td>
      <td>{self._union.num_polygons}</td>
      <td>{u_area:.4f}</td>
      <td>{p1_area + p2_area - i_area:.4f}</td>
    </tr>
    <tr>
      <td>Intersection (P1 &amp; P2)</td>
      <td>{self._intersection.num_polygons}</td>
      <td>{i_area:.4f}</td>
      <td>&mdash;</td>
    </tr>
    <tr>
      <td>Difference (P1 - P2)</td>
      <td>{self._difference.num_polygons}</td>
      <td>{d_area:.4f}</td>
      <td>{p1_area - i_area:.4f}</td>
    </tr>
  </tbody>
</table>

</body>
</html>
"""
        return html

    def save(self, path):
        """Save the visualization to an HTML file."""
        with open(path, "w") as f:
            f.write(self.generate_html())
        return path

    def _points_to_svg_attr(self, points):
        """Convert list of (x, y) to SVG points attribute string."""
        return " ".join(f"{x:.6f},{y:.6f}" for x, y in points)

    def generate_pilot_svg(self, operation="union"):
        """Generate an SVG file compatible with modmesh pilot's SVG viewer.

        The pilot viewer renders SVG ``<polygon>`` elements as wireframe
        line segments. This method generates an SVG containing:

        - Input polygon outlines (P1, P2)
        - Trapezoidal decomposition of each input polygon
        - Boolean operation result trapezoids

        Args:
            operation: one of "union", "intersection", "difference"

        Returns:
            SVG string
        """
        result_map = {
            "union": self._union,
            "intersection": self._intersection,
            "difference": self._difference,
        }
        result_pad = result_map[operation]

        x_min, y_min, x_max, y_max = self._get_all_bounds()
        # Scale to reasonable SVG coordinates (pilot flips Y internally)
        scale = 100.0
        w = (x_max - x_min) * scale
        h = (y_max - y_min) * scale

        def to_svg(points):
            """Convert world points to SVG coordinates (no Y-flip;
            pilot does its own flip)."""
            return [(
                (x - x_min) * scale,
                (y - y_min) * scale,
            ) for x, y in points]

        lines = []
        lines.append(f'<?xml version="1.0" encoding="UTF-8"?>')
        lines.append(f'<svg xmlns="http://www.w3.org/2000/svg" '
                     f'width="{w:.1f}" height="{h:.1f}" '
                     f'viewBox="0 0 {w:.1f} {h:.1f}">')

        # --- P1 decomposition trapezoids ---
        begin1, end1 = self._trap_range1
        for i in range(begin1, end1):
            pts = to_svg(self._get_trapezoid_points(self._trap_pad, i))
            lines.append(
                f'  <polygon points="{self._points_to_svg_attr(pts)}" '
                f'fill="none" stroke="{self.P1_COLOR}" '
                f'stroke-width="0.5" stroke-dasharray="3,2"/>')

        # --- P2 decomposition trapezoids ---
        begin2, end2 = self._trap_range2
        for i in range(begin2, end2):
            pts = to_svg(self._get_trapezoid_points(self._trap_pad, i))
            lines.append(
                f'  <polygon points="{self._points_to_svg_attr(pts)}" '
                f'fill="none" stroke="{self.P2_COLOR}" '
                f'stroke-width="0.5" stroke-dasharray="3,2"/>')

        # --- P1 outline ---
        pts1 = to_svg(self._get_polygon_points(self._p1))
        lines.append(
            f'  <polygon points="{self._points_to_svg_attr(pts1)}" '
            f'fill="none" stroke="{self.P1_COLOR}" stroke-width="2"/>')

        # --- P2 outline ---
        pts2 = to_svg(self._get_polygon_points(self._p2))
        lines.append(
            f'  <polygon points="{self._points_to_svg_attr(pts2)}" '
            f'fill="none" stroke="{self.P2_COLOR}" stroke-width="2"/>')

        # --- Boolean result trapezoids ---
        for i in range(result_pad.num_polygons):
            rpoly = result_pad.get_polygon(i)
            rpts = to_svg(self._get_polygon_points(rpoly))
            lines.append(
                f'  <polygon points="{self._points_to_svg_attr(rpts)}" '
                f'fill="none" stroke="{self.RESULT_COLOR}" '
                f'stroke-width="1.5"/>')

        lines.append('</svg>')
        return "\n".join(lines)

    def save_svg(self, path, operation="union"):
        """Save a pilot-compatible SVG file.

        Open the saved file in pilot via File > Open SVG file.

        Args:
            path: output file path
            operation: "union", "intersection", or "difference"
        """
        with open(path, "w") as f:
            f.write(self.generate_pilot_svg(operation))
        return path


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
