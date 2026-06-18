# Copyright (c) 2022, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


import unittest

import modmesh


class AppenvTC(unittest.TestCase):

    def setUp(self):
        self.envbak = modmesh.apputil.environ.copy()
        self.envbasenum = len(self.envbak)

    def tearDown(self):
        modmesh.apputil.environ.clear()
        modmesh.apputil.environ = self.envbak.copy()

    def test_anonymous(self):
        def _check(i, basenum):
            env = modmesh.apputil.get_appenv()
            self.assertEqual(f'anonymous{i}', env.name)
            self.assertEqual(env, modmesh.apputil.environ[f'anonymous{i}'])
            self.assertEqual(basenum + i + 1,
                             len(modmesh.apputil.environ))

        _check(0, basenum=self.envbasenum)
        _check(1, basenum=self.envbasenum)
        _check(2, basenum=self.envbasenum)
        _check(3, basenum=self.envbasenum)
        _check(4, basenum=self.envbasenum)
        _check(5, basenum=self.envbasenum)
        _check(6, basenum=self.envbasenum)
        _check(7, basenum=self.envbasenum)
        _check(8, basenum=self.envbasenum)
        _check(9, basenum=self.envbasenum)

        with self.assertRaisesRegex(
                ValueError, r'hit limit of anonymous environments \(10\)'):
            _check(10, basenum=self.envbasenum)

        # Try to reset the environment dictionary.
        modmesh.apputil.environ.clear()
        _check(0, basenum=0)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
