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
Runtime system code
"""


import builtins
import sys
import os
import argparse
import traceback

import modmesh
from . import pilot
from . import apputil


__all__ = [
    'setup_process',
    'enter_main',
    'exec_code',
]


class ModmeshArgumentParser(argparse.ArgumentParser):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.exited = False
        self.exited_status = None
        self.exited_message = None

    def exit(self, status=0, message=None):
        self.exited = True
        self.exited_status = status
        self.exited_message = message
        return


def _parse_command_line(argv):
    parser = ModmeshArgumentParser(description="Pilot")
    parser.add_argument('--mode', dest='mode', action='store',
                        default='pilot',
                        choices=['pilot', 'python', 'pytest'],
                        help='mode selection (default = %(default)s)')
    args = parser.parse_args(argv[1:])
    if parser.exited:
        args.exit = (parser.exited_status, parser.exited_message)
    else:
        args.exit = tuple()
    return args


def _run_pilot(argv=None):
    """Run the pilot application."""
    return pilot.launch()


def _run_pytest():
    # Import pytest locally to avoid making it a dependency to the whole
    # modmesh.
    import pytest
    mmpath = os.path.join(os.path.dirname(modmesh.__file__), '..', 'tests')
    mmpath = os.path.abspath(mmpath)
    return pytest.main(['-v', '-x', mmpath])


def setup_process(argv):
    """Set up the runtime environment for the process."""
    # Install the namespace.
    builtins.modmesh = modmesh
    builtins.mm = modmesh


def enter_main(argv):
    args = _parse_command_line(argv)
    ret = 0
    if args.exit:
        ret = args.exit[0]
    elif 'pilot' == args.mode:
        ret = _run_pilot(argv)
    elif 'pytest' == args.mode:
        ret = _run_pytest()
    elif 'python' == args.mode:
        sys.stderr.write('mode "python" should not run in Python main')
        ret = 1
    else:
        sys.stderr.write('mode "{}" is not supported'.format(args.mode))
    return ret


def exec_code(code):
    try:
        apputil.run_code(code)
    except Exception as e:
        sys.stdout.write("code:\n{}\n".format(code))
        sys.stdout.write("{}: {}\n".format(type(e).__name__, str(e)))
        sys.stdout.write("traceback:\n")
        traceback.print_stack()

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
