#!/bin/bash
#
# Copyright (C) 2022 Yung-Yu Chen <yyc@solvcon.net>.

echo "gcc path: $(which gcc)"
echo "gcc version: $(gcc --version)"
echo "cmake path: $(which cmake)"
echo "cmake version: $(cmake --version)"
echo "python3 path: $(which python3)"
echo "python3 version: $(python3 --version)"
echo "python3-config --prefix: $(python3-config --prefix)"
echo "python3-config --exec-prefix: $(python3-config --exec-prefix)"
echo "python3-config --includes: $(python3-config --includes)"
echo "python3-config --libs: $(python3-config --libs)"
echo "python3-config --cflags: $(python3-config --cflags)"
echo "python3-config --ldflags: $(python3-config --ldflags)"
echo "pip3 path: $(which pip3)"
python3 -c 'import numpy as np; print("np.__version__:", np.__version__, np.get_include())'
echo "pytest path: $(which pytest)"
echo "pytest version: $(pytest --version)"
echo "clang-tidy path: $(which clang-tidy)"
echo "clang-tidy version: $(clang-tidy -version)"
echo "clang-format path: $(which clang-format 2>/dev/null || echo not-installed)"
echo "clang-format version: $(clang-format --version 2>/dev/null || echo not-installed)"
echo "flake8 path: $(which flake8)"
echo "flake8 version: $(flake8 --version)"
echo "black path: $(which black 2>/dev/null || echo not-installed)"
echo "black version: $(black --version 2>/dev/null || echo not-installed)"
