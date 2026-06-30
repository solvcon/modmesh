# Build Dependencies

To build the dependencies from source and install them into user space rather
than system-wide, use the standalone scdv build scripts in
`contrib/dependency/` described below.

For a complete, self-contained environment, the per-platform scripts
`ubuntu2404/build-scdv-ubuntu24.sh` and `macos26/build-scdv-macos26.sh` build
solvcon's full runtime stack from source -- zlib, OpenSSL, SQLite, CPython,
pybind11, Cython, NumPy, SciPy, Qt, and PySide6 -- into a versioned prefix
under your home directory (by default
`${HOME}/var/scdv/<platform>-py<pyver>-qt<qtver>`).

The build is organized into four sections: `BASE`, `PYTHON`, `NUMPY`, and `QT`
with the corresponding environment variables `SCDVBUILD_BASE`,
`SCDVBUILD_PYTHON`, `SCDVBUILD_NUMPY`, and `SCDVBUILD_QT`. If none is set, the
script builds everything.

The script never runs `apt` or Homebrew itself. Print the prerequisite
commands, review them, and run them yourself:

```sh
cd contrib/dependency/ubuntu2404
./build-scdv-ubuntu24.sh --print-apt   # review, then run the printed commands
./build-scdv-ubuntu24.sh               # build the whole stack into the prefix
```

Useful flags: `--print-prefix` reports the install prefix and exits;
`--no-confirm` skips the pre-build prompt for non-interactive runs; `--skip
PKG` omits a package (repeatable or comma-separated); and
`--write-activate-only` (re)writes just the activation script.

When the build finishes it writes an `activate` script in the prefix.  Source
it to put the freshly built Python and Qt on your `PATH`, and run
`scdv_deactivate` to restore the original environment:

```sh
source ${HOME}/var/scdv/<platform>-py<pyver>-qt<qtver>/activate
```

Once the dependencies are in place, build solvcon as described in
{doc}`build_solvcon`.

<!-- vim: set ft=markdown ff=unix fenc=utf8 et sw=2 ts=2 sts=2 tw=79: -->
