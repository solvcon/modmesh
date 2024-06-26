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
import time

lib_path = {}


def search_library_root(curr_path, lib_root_name, timeout=1.0):
    """
    Walk through the thridparty library and register all third-party
    library folder path into a dictionary with library name as the key,
    user can select which library need to be imported by library name.

    :param curr_path: The path to start searching from.
    :type curr_path: str
    :param lib_root_name: The name of the root folder containing libraries.
    :type lib_root_name: str
    :param timeout: Maximum time for root path searching. Default is 1.0 sec.
    :type timeout: float
    :return: None
    """
    # Try to find the library root, if failed to find it
    # modmesh will raise ImportError and remind the user.
    folder_name = os.path.join(lib_root_name)
    _path = curr_path
    start_time = time.time()
    while time.time() - start_time < timeout:
        if os.path.exists(os.path.join(_path, folder_name)):
            break
        if _path == os.path.dirname(_path):
            _path = None
            break
        else:
            _path = os.path.dirname(_path)
    _pf = os.path.join(_path, folder_name) if _path else None
    if _path is None or not os.path.exists(_pf):
        sys.stderr.write(f'Can not find {lib_root_name}.\n')
        return

    _path = os.path.join(_path, folder_name)

    for item in os.listdir(_path):
        if os.path.isdir(os.path.join(_path, item)):
            lib_path[item] = os.path.join(_path, item)


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
        sys.stderr.write(f'Can not find {lib_name} in library root.\n')
        sys.exit(0)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
