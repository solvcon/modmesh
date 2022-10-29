# Copyright (c) 2020, Yung-Yu Chen <yyc@solvcon.net>
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


import os
import unittest

import modmesh


class ToggleTC(unittest.TestCase):

    def test_instance(self):
        self.assertTrue(hasattr(modmesh.Toggle.instance, "show_axis"))


@unittest.skipUnless("viewer" in modmesh.clinfo.executable_basename,
                     "not in viewer binary")
class CommandLineInfoTC(unittest.TestCase):

    def setUp(self):
        self.cmdline = modmesh.ProcessInfo.instance.command_line

    def test_populated(self):
        self.assertTrue(self.cmdline.populated)
        self.assertNotEqual(len(self.cmdline.populated_argv), 0)


class MetalTC(unittest.TestCase):

    # Github Actions macos-12 does not support GPU yet.
    @unittest.skipUnless(modmesh.METAL_BUILT and "TEST_METAL" in os.environ,
                         "Metal is not built")
    def test_metal_status(self):
        self.assertEqual(True, modmesh.metal_running())

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
