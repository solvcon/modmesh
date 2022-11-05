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

# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.


"""
Tools to run applications
"""


# Use flake8 http://flake8.pycqa.org/en/latest/user/error-codes.html


import importlib


__all__ = [
    'environ',
    'AppEnvironment',
    'run_code',
    'stop_code',
]


# All environment objects of this process.
environ = {}


class AppEnvironment:
    """
    Collects the environment for an application.

    :ivar globals:
        The global namespace of the application.
    :ivar locals:
        The local namespace of the application.
    """
    def __init__(self, name):
        self.globals = {
            # Give the application an alias of the top package.
            'mm': importlib.import_module('modmesh'),
            'appenv': self,
        }
        self.locals = {}
        self.name = name
        # Each run of the application appends a new environment.
        environ[name] = self

    def run_code(self, code):
        exec(code, self.globals, self.locals)


def get_appenv(name=None):
    if None is name:
        i = 0
        name = 'anonymous%d' % i
        while name in environ:
            i += 1
            name = 'anonymous%d' % i
    app = environ.get(name, None)
    if None is app:
        app = AppEnvironment(name)
    return app


get_appenv(name='master')


def run_code(code):
    has_key = False
    for k in reversed(environ):
        has_key = True
        break
    if not has_key:
        raise KeyError("No AppEnviron is available")
    aenv = environ[k]
    aenv.run_code(code)


def stop_code(appenvobj=None):
    if None is appenvobj:
        environ.clear()
    else:
        indices = [i for i, o in enumerate(environ) if o == appenvobj]
        indices = reversed(indices)
        for i in indices:
            del environ[i]

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
