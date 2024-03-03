# Copyright (c) 2019, Yung-Yu Chen <yyc@solvcon.net>
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
Viewer
"""


# Use flake8 http://flake8.pycqa.org/en/latest/user/error-codes.html

import sys
import os

_from_impl = [  # noqa: F822
    'R3DWidget',
    'RLine',
    'RPythonConsoleDockWidget',
    'RManager',
    'mgr',
]

__all__ = _from_impl + [  # noqa: F822
    'populateApplications',
    'resetApplications',
    'launch',
]

# Try to import the viewer code but easily give up.
enable = False
try:
    from _modmesh import view as _vimpl  # noqa: F401
    enable = True
except ImportError:
    pass


def _load():
    if enable:
        for name in _from_impl:
            globals()[name] = getattr(_vimpl, name)
        # Try to find the PUI in thirdparty, if failed to find PUI
        # modmesh will raise ImportError and terminate itself.
        filename = os.path.join('thirdparty', 'PUI')
        path = os.getcwd()
        try:
            while True:
                if os.path.exists(os.path.join(path, filename)):
                    break
                if path == os.path.dirname(path):
                    path = None
                    break
                else:
                    path = os.path.dirname(path)
            if path is None or not os.path.exists(os.path.join(path,
                                                               filename,
                                                               'PUI')):
                raise ImportError
        except ImportError:
            sys.stderr.write('Can not find PUI in your environment.\n')
            sys.stderr.write('Please run git submodule update --init\n')
            sys.exit(0)

        path = os.path.join(path, filename)
        sys.path.append(path)


_load()
del _load


def populate_applications():
    mw = _vimpl.RManager.instance.mainWindow
    mw.addApplication("sample_mesh")
    mw.addApplication("euler1d")
    mw.addApplication("linear_wave")
    mw.addApplication("bad_euler1d")


def reset_applications():
    mw = _vimpl.RManager.instance.mainWindow
    mw.clearApplications()
    populate_applications()


def launch(name="Modmesh Viewer", size=(1000, 600)):
    app = _vimpl.RManager.instance
    app.setUp()
    wm = app
    wm.windowTitle = name
    wm.resize(w=size[0], h=size[1])
    wm.show()
    return app.exec()

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
