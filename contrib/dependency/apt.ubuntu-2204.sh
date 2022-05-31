# Install system tools.
apt-get -qy install sudo curl git
# Install compiler and build tools.
apt-get -qy install build-essential make cmake libc6-dev gcc g++ clang-tidy-14
# Install Python.
apt-get -qy install python3 python3-dev python3-env python3-setuptools python3.10-dev
apt-get -qy install python3-numpy python3-pytest python3-flake8