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


class PlaneLayer:
    def __init__(self):
        self.polys = []

    def add_rectangle(self, x, y, w, h):
        rect = [
            [(x, y), (x+w, y)],
            [(x+w, y), (x+w, y+h)],
            [(x+w, y+h), (x, y+h)],
            [(x, y+h), (x, y)]
        ]

        self.polys.append(rect)

    def add_polygon(self, coords):
        point_count = len(coords) // 2

        poly = []

        for i in range(point_count):
            curr_idx = i * 2
            next_idx = ((i + 1) % point_count) * 2
            poly.append([
                (coords[curr_idx], coords[curr_idx+1]),
                (coords[next_idx], coords[next_idx+1])])

        self.polys.append(poly)

    def add_figure(self, str):
        str_segs = str.split(" ")

        if str_segs[0] == "RECT":
            x = float(str_segs[3])
            y = float(str_segs[4])
            w = float(str_segs[5])
            h = float(str_segs[6])
            self.add_rectangle(x, y, w, h)

        elif str_segs[0] == "PGON":
            poly_coords = list(map(float, str_segs[3:]))
            self.add_polygon(poly_coords)

    def get_polys(self):
        return self.polys

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
