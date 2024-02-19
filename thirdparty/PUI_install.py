#!/usr/bin/env python3

import os
import subprocess

script_root = os.path.dirname(os.path.realpath(__file__))
pui_git = os.path.join(script_root, "PUI", ".git")

if not os.path.exists(pui_git):
    print(f"{pui_git} does not exist, running command...")
    print("git submodule update --init")
    subprocess.run(["git", "submodule", "update", "--init"])
else:
    print(f"{pui_git} exists.")

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
