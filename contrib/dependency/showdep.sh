#!/bin/bash
#
# Copyright (C) 2022 Yung-Yu Chen <yyc@solvcon.net>.

echo "gcc path: $(which gcc)"
echo "gcc version: $(gcc --version)"
echo "cmake path: $(which cmake)"
echo "cmake version: $(cmake --version)"
echo "python3 path: $(which python3)"
echo "python3 version: $(python3 --version)"
echo "pip3 path: $(which pip3)"
python3 -c 'import numpy ; print("numpy.__version__:", numpy.__version__)'
echo "pytest path: $(which pytest)"
echo "pytest version: $(pytest --version)"
echo "clang-tidy path: $(which clang-tidy)"
echo "clang-tidy version: $(clang-tidy -version)"
echo "flake8 path: $(which flake8)"
echo "flake8 version: $(flake8 --version)"