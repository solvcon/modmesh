#!/usr/bin/env bash

script_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ ! -d "$script_root/PUI" ]; then
    echo "$script_root/PUI does not exist, running commnad..."
    echo "git submodule update --init"
    git submodule update --init
else
    echo "$script_root/PUI exists."
fi
