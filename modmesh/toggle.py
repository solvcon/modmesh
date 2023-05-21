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


"""
High-level control of toggles.
"""


import json

from . import core


def load(data, toggle_instance=None):
    tg = toggle_instance
    if tg is None:
        tg = core.Toggle.instance

    # Parse the input JSON data
    pdata = json.loads(data)
    if len(pdata) != 2:
        raise ValueError("input data must be 2 but get %d" % len(pdata))
    # pfixed = pdata[0].get('fixed', {})
    pdynamic = pdata[1].get('dynamic', {})

    # Get the apps sub-section
    papps = pdynamic.get('apps', {})
    tg.add_subkey('apps')
    ta = tg.apps

    # App euler1d
    ta.add_subkey('euler1d')
    thisapp = ta.euler1d
    thispdata = papps.get('euler1d', {})
    thisapp.set_bool('use_sub', thispdata.get('use_sub', False))

    return tg


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
