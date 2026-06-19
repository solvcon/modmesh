# Copyright (c) 2023, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

import numpy as np


class TestBase:

    def assert_allclose(self, *args, **kw):
        if 'rtol' not in kw:
            kw['rtol'] = 1.e-12
        return np.testing.assert_allclose(*args, **kw)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
