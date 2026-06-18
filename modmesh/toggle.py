# Copyright (c) 2022, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


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
