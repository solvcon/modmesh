FROM ubuntu:22.04@sha256:1b8d8ff4777f36f19bfe73ee4df61e3a0b789caeff29caa019539ec7c9a57f95

WORKDIR /workspace
ENV PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && \
  apt-get install -y curl \
  git \
  build-essential \
  cmake \
  libc6-dev \
  g++ \
  clang-tidy-14 \
  qt6-base-dev
RUN apt-get install -y \
  python3 \
  python3-dev \
  python3-venv \
  python3-setuptools \
  python3-pip \
  python3-numpy \
  python3.10-dev
RUN python3 -m pip install \
  jupyter \
  pytest \
  flake8
RUN apt-get purge && \
  apt-get autoremove && \
  apt-get autoclean && \
  rm -rf /var/lib/apt/lists/*

CMD [ "/bin/bash" ]