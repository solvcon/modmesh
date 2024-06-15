# Copyright (c) 2024, Chun-Hsu Lai <as2266317@gmail.com>
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

import sys
import os

lib_path = {}


def register_library():
    """
    Walk through the thridparty library and register all third-party
    library folder path into a dictionary with library name as the key,
    user can select which library need to be imported by library name.

    :return: None
    """
    dirName = os.path.dirname(__file__)
    for item in os.listdir(dirName):
        if os.path.isdir(os.path.join(dirName, item)):
            lib_path[item] = os.path.join(dirName, item)


def load_library(lib_name):
    """
    Append library path into python path by user input library name

    :param lib_name: library name that need to be imported.
    :return: None
    """
    try:
        if lib_name in lib_path:
            sys.path.append(lib_path[lib_name])
        else:
            raise ImportError
    except ImportError:
        sys.stderr.write(f'Can not find {lib_name} in thirdparty library.\n')
        sys.exit(0)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
