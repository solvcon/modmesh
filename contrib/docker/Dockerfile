FROM ubuntu:22.04

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