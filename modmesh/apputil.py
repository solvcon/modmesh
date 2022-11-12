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
import contextlib
import traceback
import sys

__all__ = [
    'environ',
    'AppEnvironment',
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

    # default config
    _config = {
        'appEnvName': None,
        'redirectStdOutFile': 'stdout.txt',
        'redirectStdErrFile': 'stderr.txt',
    }

    def __init__(self, config=None):
        self.globals = {
            # Give the application an alias of the top package.
            'mm': importlib.import_module('modmesh'),
            'appenv': self,
        }
        self.locals = {}
        # Each run of the application appends a new environment.
        environ[config['appEnvName']] = self

        # replace the config value; otherwise keep the default value
        for k, v in config.items():
            if v is not None:
                self._config[k] = v

    def run_code(self, code):
        with open(self._config['redirectStdOutFile'], 'w') as f1:
            with contextlib.redirect_stdout(f1):
                with open(self._config['redirectStdErrFile'], 'w') as f2:
                    with contextlib.redirect_stderr(f2):
                        try:
                            exec(code, self.globals, self.locals)
                        except Exception as e:
                            print(("{}: {}".format(
                                type(e).__name__, str(e))), file=sys.stderr)
                            print("traceback:", file=sys.stderr)
                            traceback.print_stack()
                        sys.stdout.flush()
                        sys.stderr.flush()


def get_appenv(name=None):
    if None is name:
        for i in range(10):
            name = f'anonymous{i}'
            if name not in environ:
                break
        else:
            raise ValueError("hit limit of anonymous environments (10)")
    app = environ.get(name, None)
    if None is app:
        app = AppEnvironment(name)
    return app


def get_appenv(name=None):
    app = environ.get(name, None)

    if None is app:
        config = {'name': name}
        new_appenv(config)
    return app


def stop_code(appenvobj=None):
    if None is appenvobj:
        environ.clear()
    else:
        indices = [i for i, o in enumerate(environ) if o == appenvobj]
        indices = reversed(indices)
        for i in indices:
            del environ[i]

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
