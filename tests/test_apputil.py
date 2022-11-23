# Copyright (c) 2022, Yung-Yu Chen <yyc@solvcon.net>
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
