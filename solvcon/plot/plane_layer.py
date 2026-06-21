# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


class PlaneLayer:
    def __init__(self):
        self.polys = []

    def add_rectangle(self, x, y, w, h):
        rect = [
            [(x, y), (x + w, y)],
            [(x + w, y), (x + w, y + h)],
            [(x + w, y + h), (x, y + h)],
            [(x, y + h), (x, y)]
        ]

        self.polys.append(rect)

    def add_polygon(self, coords):
        point_count = len(coords) // 2

        poly = []

        for i in range(point_count):
            curr_idx = i * 2
            next_idx = ((i + 1) % point_count) * 2
            poly.append([
                (coords[curr_idx], coords[curr_idx + 1]),
                (coords[next_idx], coords[next_idx + 1])])

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
