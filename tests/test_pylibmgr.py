# Copyright (c) 2024, Chunhsu Lai <as2266317@gmail.com>
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# - Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
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

import os

import sys

from modmesh import pylibmgr


class pylibmgrTC(unittest.TestCase):

    def test_pylibmgr_search_library_root(self):
        # This test case assumes that modmesh third-party lib root's name is
        # thirdparty, it is located at modmesh project root: /path/to/modmesh.
        pylibmgr.search_library_root(os.getcwd(), 'thirdparty')
        finder = next(finder for finder in sys.meta_path
                      if isinstance(finder, pylibmgr.ModmeshPathFinder))
        self.assertNotEqual(finder.lib_paths, {})

        # Reset library patch record
        finder.lib_paths = {}

        pylibmgr.search_library_root(os.getcwd(), "Can_not_find")
        self.assertEqual(finder.lib_paths, {})

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
