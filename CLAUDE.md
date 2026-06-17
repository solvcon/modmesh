# Agent Instructions

This file provides guidance to AI coding agents working in this repository
(Claude Code and Cursor). `AGENTS.md` at the repo root and
`.cursor/AGENTS.md` are symlinks to this file.

## Project Overview

modmesh is a hybrid C++/Python library for solving conservation laws using the
space-time Conservation Element and Solution Element (CESE) method with
unstructured meshes. The codebase emphasizes:

- High-performance numerical computing through C++ with Python bindings
- Multi-dimensional array operations and contiguous buffer management
- One-dimensional solvers demonstrating the CESE method
- Qt-based GUI (pilot) for spatial data visualization
- Integrated runtime profiler for performance analysis

## Agent Tooling (Claude Code and Cursor)

This repository ships `.claude/` and `.cursor/` directories with permissions,
hooks, and skills tuned to this codebase. General behavioral rules live in
`contrib/prompt/general-rule.md` (not auto-imported). This section indexes
the tools.

### Cursor (`.cursor/`)

- `cli.json` -- project shell/file permissions (translated from
  `.claude/settings.json`; `Shell(...)` replaces Claude's `Bash(...)`).
- `hooks.json` -- `postToolUse` on `Write|StrReplace` and `afterFileEdit` on
  `Write|TabWrite`, wired to the shared `check-source.sh` script.
- `hooks/` -- symlink to `.claude/hooks/`.
- `skills/` -- symlink to `.claude/skills/`.
- `AGENTS.md` -- symlink to `CLAUDE.md` at the repo root.
- `statusline.sh` -- symlink to `.claude/statusline.sh`. Point your
  `~/.cursor/cli-config.json` `statusLine.command` at
  `$PROJECT/.cursor/statusline.sh` (or `.claude/statusline.sh`).

## Claude Code Tooling

### Skills (`.claude/skills/`)

- `cpp-style-review` -- judgment-call C++ review (`m_` prefix, function-body
  placement, `SimpleCollector` preference, pybind11 binding split,
  `const_cast`). Scoped to `git diff`. Invoke after editing files in `cpp/`
  or `gtests/`.
- `python-style-review` -- judgment-call Python review (naming, test intent,
  project conventions). Scoped to `git diff`. Invoke after editing files in
  `modmesh/` or `tests/`.

Skills inherit the caller's model rather than pinning their own. Deterministic
style checks (ASCII, trailing whitespace, modeline, 79-char Python lines) are
owned by hooks, not skills.

### Hooks (`.claude/hooks/`)

- `check-source.sh` -- PostToolUse on `Write|Edit` of source files.  Surfaces
  non-ASCII bytes, trailing whitespace, missing modeline, and Python `>79`
  chars; exits 2 with `path:line -- rule -- fix`.

### Settings (`.claude/settings.json`)

- `permissions.allow` whitelists the safe `make` targets, `cmake`, `pytest`,
  lint/format tools, and read-only git/gh. `make clean` and `make cmakeclean`
  deliberately prompt.
- `permissions.deny` hard-blocks only `sudo` and `rm -rf` of root/home.
  Destructive git operations (force-push, `git reset --hard`, `git clean -fd`)
  are discouraged but not blocked -- use them deliberately and only when asked.
- `hooks` wires the script above.
- `statusLine` runs `.claude/statusline.sh` -- shows model, project, branch
  (with `*` if dirty), and context-window usage.

## Build, Test, Lint, Format

All workflows are driven through `make` from the repo root. The Makefile sets
`PYTHONPATH=$(MODMESH_ROOT)` so the in-tree `_modmesh` extension is picked up
without installation, and works around macOS SIP stripping `DYLD_LIBRARY_PATH`.

**Build**

- `make` -- build the `_modmesh` Python extension (default target).
- `make pilot` -- build the Qt pilot GUI binary.
- `make clean` / `make cmakeclean` -- remove build artifacts.

**Test**

- `make pytest` -- full Python test suite.
- `make pytest PYTEST_OPTS="tests/test_buffer.py::SimpleArrayBasicTC::test_sort"`
  -- run a single test or subset; `PYTEST_OPTS` is forwarded verbatim to
  pytest.
- `make run_pilot_pytest` -- Python tests that require the pilot GUI;
  accepts `PYTEST_OPTS` the same way.
- `make gtest` -- build and run the full C++ test suite.
- `./build/rel<pyvminor>/gtests/run_gtest --gtest_filter=Suite.Test` -- run a
  single gtest after `make gtest` has built the binary
  (`<pyvminor>` is the Python major+minor, e.g. `314`).
- `make pyprof` -- run profiling benchmarks; results land in
  `profiling/results/`.

**Lint** (`make lint` runs all five)

- `make cformat` -- check C++ formatting (read-only; use
  `make FORCE_CLANG_FORMAT=inplace cformat` to fix).
- `make cinclude` -- check `#include` ordering and style.
- `make flake8` -- Python style and 79-char line limit.
- `make checkascii` -- reject non-ASCII bytes in source.
- `make checktws` -- reject trailing whitespace.

**Format**

Automatic formatting is still work in progress. Do not run `make format` or
`make pyformat`.

Any target whose tool (`clang-format`, `flake8`) is missing prints an install
hint and exits 1. `make cformat` also warns when the local `clang-format` major
version differs from the CI pin (`CLANG_FORMAT_CI_VERSION` in the Makefile).

### Build Configuration

Key build variables (set in `setup.mk` or as environment variables):
- `CMAKE_BUILD_TYPE`: `Release` (default) or `Debug`
- `BUILD_QT`: `ON` (default) or `OFF` - build Qt GUI components
- `BUILD_METAL`: `OFF` (default) or `ON` - build Metal GPU support
- `MODMESH_PROFILE`: `OFF` (default) or `ON` - enable profiler
- `USE_CLANG_TIDY`: `OFF` (default) or `ON` - use clang-tidy
- `HIDE_SYMBOL`: `ON` (default) - hide Python wrapper symbols
- `DEBUG_SYMBOL`: `ON` (default) - add debug information

Build paths (`$(pyvminor)` is the active Python major+minor, e.g. `314`):
- Release builds (default): `build/rel<pyvminor>` (e.g., `build/rel314`)
- Debug builds: `build/dbg<pyvminor>` (e.g., `build/dbg314`)

## Architecture

### Hybrid C++/Python Design

modmesh uses a dual-layer hybrid architecture:

1. **C++ Core** (`cpp/modmesh/`): High-performance numerical code
   - Compiled to native libraries with pybind11 bindings
   - Exposed to Python through the `_modmesh` extension module

2. **Python Interface** (`modmesh/`): High-level API and utilities
   - Imports C++ components via `from .core import *`
   - Provides Python-native functionality (plotting, utilities, etc.)

### C++ Component Structure

C++ core lives in `cpp/modmesh/`. Load-bearing pieces:

- `buffer/` -- `ConcreteBuffer`, `SimpleArray`, `BufferExpander`,
  `small_vector`.
- `mesh/` -- `StaticMesh` (unstructured meshes with mixed element types).
- `pilot/` -- Qt GUI (entry point under `cpp/binary/pilot/`; needs Qt6
  and PySide6).
- `python/` -- `module.cpp` is the main pybind11 module.

Other subdirectories cover what their names suggest: `linalg/`
(BLAS/LAPACK wrappers), `inout/` (Gmsh, Plot3D), `onedim/` (1D CESE
solvers), `profiling/` (runtime profiler), `simd/` (NEON/SSE/AVX),
`transform/` (integral transform), `universe/` (3D geometry), `toggle/`
(feature toggle), and per-component `pymod/` subdirs for pybind11
wrappers. `spacetime/` is an old, incorrect CESE implementation kept
for reference only -- do not extend it.

See `cpp/modmesh/` for the current tree.

### Python Package Structure

Python interface in `modmesh/`:

- `core.py`: Main Python API wrapping the C++ extension
- `onedim/`: One-dimensional solver utilities
- `pilot/`: GUI application Python components
- `plot/`: Plotting utilities
- `profiling/`: Profiling result analysis
- `testing.py`: Test utilities
- `toggle.py`: Feature toggle Python interface

### Testing Structure

Python tests are the default. Prefer writing tests in Python (`tests/`); reach
for C++ gtest only when the code cannot or should not be exercised from Python
-- for example, internals with no Python binding, or behavior that must be
verified at the C++ level.

- **Python tests** (`tests/`): pytest-based, files named `test_*.py`. The
  preferred place for tests.
- **C++ tests** (`gtests/`): googletest-based, files named
  `test_nopython_*.cpp`. Use only when Python cannot or should not reach the
  code under test.
- **Profiling benchmarks** (`profiling/`): files named `profile_*.py`.

See "Build, Test, Lint, Format" above for the `make` invocations.

## Code Style

`STYLE.md` is the canonical source. At a glance:

- **Line economy**: Prefer fewer lines for better human readability. Dense
  code within the line-width limits is easier to scan. Do not add unnecessary
  blank lines or spread simple logic across many lines. Always respect the
  linting line-width limits -- never sacrifice them to shorten line count.
- **C++**: 4-space indent, `m_` prefix on member vars, angle-bracket includes,
  C++23, prefer `SimpleCollector` / `small_vector` over STL for fundamentals.
- **Python**: PEP-8, 79-char hard limit, flake8.
- **All source**: UTF-8, Unix LF, ASCII-only, no trailing whitespace, modeline
  at EOF.

How style is enforced in this repo:

- `.claude/hooks/check-source.sh` owns the deterministic checks (ASCII bytes,
  trailing whitespace, modeline at EOF, Python `>79`-char lines).
- The `cpp-style-review` and `python-style-review` skills in
  `.claude/skills/` own the judgment-call rules (`m_` prefix in context,
  function-body placement, container choice, pybind11 binding split, test
  intent). They are scoped to `git diff`.

For the full rule set with examples, see `STYLE.md`.

## Pull Request Guidelines

When opening a pull request, reference the related issue (e.g., "Related to
#725") instead of using closing keywords like "close #725", "closes #725", or
"fixes #725". We do not let PR and commit log comments to mandate the
management.

## Development Workflow

### Build System Notes

- CMake is the primary build system (minimum version 3.27)
- Makefile wraps CMake for convenience
- Python extension built via setuptools with custom CMake integration
- Build output: `_modmesh.cpython-<version>-<platform>.so` in `modmesh/`

### Dependencies

Core dependencies:
- Python 3 with development headers
- pybind11 >= 2.12.0 (for NumPy 2.0 support)
- NumPy
- CMake >= 3.27
- C++23 compiler (gcc, clang, or MSVC)

Optional dependencies:
- Qt6 and PySide6 (for GUI)
- clang-tidy (for linting)
- googletest (auto-fetched by CMake)
- Metal (for macOS GPU support)

Install scripts available in `contrib/dependency/`

### Virtual Environments

**IMPORTANT**: Using Python virtual environments (venv, conda) is **strongly
discouraged** for modmesh development. The project is designed to work with
system Python. Virtual environment bugs are not actively resolved.

Use https://github.com/solvcon/devenv to build dependency from source and
install in user space. Do not install dependency system-wide. Installation of
any dependency requires user review and consent.

### Platform-Specific Notes

**macOS**: System Integrity Protection (SIP) may interfere with
`DYLD_LIBRARY_PATH`. The Makefile sets `PYTHONPATH` as a workaround.

## Profiling System

modmesh includes an integrated runtime profiler:

1. Enable with `MODMESH_PROFILE=ON` during build
2. Use `toggle.py` API to enable/disable profiling regions
3. Run profiling scripts with `make pyprof`
4. Results written to `profiling/results/`

## Qt GUI Development

The pilot application (`cpp/binary/pilot/`) is a standalone Qt6-based viewer:

- Requires `BUILD_QT=ON` (default)
- Uses PySide6 for Python-Qt integration
- Resource files in `resources/pilot/`
- Can be disabled with `BUILD_QT=OFF` for headless builds

## Common Development Patterns

### Adding a New C++ Component

1. Create a directory under `cpp/modmesh/`.
2. Add header files with proper include guards.
3. Update `cpp/modmesh/CMakeLists.txt` to include new sources.
4. Add pybind11 bindings if Python access is needed.
5. Write tests. Prefer Python tests in `tests/`; add a `gtests/` test only
   when the behavior cannot or should not be exercised from Python.

### Adding Python-Only Functionality

1. Add a module to `modmesh/`.
2. Update `modmesh/__init__.py` if needed.
3. Write tests in `tests/`.
4. Update `setup.py` packages list if adding a new package.

### Memory Management

- Use `ConcreteBuffer` for raw memory.
- Use `SimpleArray` for typed multi-dimensional arrays.
- Buffers support both ownership and non-owning views.
- Python and C++ share the same buffer memory (zero-copy).

<!-- vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79: -->
