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


_from_impl = [  # noqa: F822
    'R3DWidget',
    'RLine',
    'RPythonConsoleDockWidget',
    'RMainWindow',
    'RApplication',
    'app',
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


_load()
del _load


def populate_applications():
    mw = _vimpl.RApplication.instance.mainWindow
    mw.addApplication("sample_mesh")
    mw.addApplication("euler1d")
    mw.addApplication("linear_wave")
    mw.addApplication("bad_euler1d")


def reset_applications():
    mw = _vimpl.RApplication.instance.mainWindow
    mw.clearApplications()
    populate_applications()


def launch(name="Modmesh Viewer", size=(1000, 600)):
    app = _vimpl.RApplication.instance
    app.setUp()
    wm = app.manager
    wm.windowTitle = name
    wm.resize(w=size[0], h=size[1])
    wm.show()
    return app.exec()

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
