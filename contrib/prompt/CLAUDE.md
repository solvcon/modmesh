# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with
code in this repository.

## Project Overview

modmesh is a hybrid C++/Python library for solving conservation laws using the
space-time Conservation Element and Solution Element (CESE) method with
unstructured meshes. The codebase emphasizes:

- High-performance numerical computing through C++ with Python bindings
- Multi-dimensional array operations and contiguous buffer management
- One-dimensional solvers demonstrating the CESE method
- Qt-based GUI (pilot) for spatial data visualization
- Integrated runtime profiler for performance analysis

## Build Commands

### Basic Build
```bash
# Build Python extension (default target)
make

# Build with verbose output
make VERBOSE=1

# Build with clang-tidy
make USE_CLANG_TIDY=ON
```

### Testing
```bash
# Run Python tests
make pytest

# Run C++ tests (googletest)
make gtest
```

### Profiling
```bash
# Run profiling benchmarks
make pyprof
```

### GUI Application
```bash
# Build the pilot (Qt-based viewer)
make pilot

# Run pilot tests
make run_pilot_pytest
```

### Linting
```bash
# Run every check
make lint

# Individual checks
make cinclude    # C++ #include uses <...>, not "..."
make flake8      # Python style
make checkascii  # ASCII-only source files
make checktws    # no trailing whitespace

# Auto-fix formatting
make cformat     # C++ formatting (clang-format --dry-run -Werror)
make pyformat    # Python (black)
make format      # C++ and Python (clang-format -i + black)
```

Any target whose tool (`clang-format`, `flake8`, `black`) is missing
prints an install hint and exits 1. `make cformat` also warns when the
local `clang-format` major version differs from the CI pin
(`CLANG_FORMAT_CI_VERSION` in the Makefile).

### Cleanup
```bash
# Clean build artifacts
make clean

# Remove entire build directory
make cmakeclean
```

### Build Configuration

Key build variables (set in `setup.mk` or as environment variables):
- `CMAKE_BUILD_TYPE`: `Release` (default) or `Debug`
- `BUILD_QT`: `ON` (default) or `OFF` - build Qt GUI components
- `BUILD_METAL`: `OFF` (default) or `ON` - build Metal GPU support
- `MODMESH_PROFILE`: `OFF` (default) or `ON` - enable profiler
- `USE_CLANG_TIDY`: `OFF` (default) or `ON` - use clang-tidy
- `HIDE_SYMBOL`: `ON` (default) - hide Python wrapper symbols
- `DEBUG_SYMBOL`: `ON` (default) - add debug information

Build paths:
- Debug builds: `build/dbg<pyversion>` (e.g., `build/dbg313`)
- Release builds: `build/dev<pyversion>` (e.g., `build/dev313`)

## Architecture

### Hybrid C++/Python Design

modmesh uses a dual-layer hybrid architecture:

1. **C++ Core** (`cpp/modmesh/`): High-performance numerical code
   - Compiled to native libraries with pybind11 bindings
   - Exposed to Python through `_modmesh` extension module

2. **Python Interface** (`modmesh/`): High-level API and utilities
   - Imports C++ components via `from .core import *`
   - Provides Python-native functionality (plotting, utilities, etc.)

### C++ Component Structure

Core components in `cpp/modmesh/`:

- `**/pymod/**`: Python wrapper for each component.

- `buffer/**`: Memory management and multi-dimensional arrays
  - `ConcreteBuffer`: Fundamental contiguous memory buffer
  - `SimpleArray`: N-dimensional array wrapper
  - `BufferExpander`: Dynamic buffer resizing
  - `small_vector`: Stack-optimized vector

- `mesh/**`: Unstructured mesh data structures
  - `StaticMesh`: Unstructured meshes with mixed element types

- `linalg/**`: Linear algebra operations (BLAS, LAPACK wrappers)

- `inout/**`: I/O utilities (Gmsh, Plot3D formats)

- `onedim/**`: 1D solvers demonstrating CESE method

- `pilot/**`: Qt-based GUI application for visualization
  - `cpp/binary/pilot/` is the entry point for the standalone executable
  - Requires Qt6 and PySide6

- `profiling/**`: Performance profiler for runtime and memory

- `python/**`: Common infrastructure for Python binding
  - `module.cpp`: Main pybind11 module definition
  - Component-specific wrappers throughout the codebase

- `simd/**`: SIMD optimization helpers
  - Platform-specific (NEON, SSE, AVX) abstractions

- `spacetime/**`: An old, incorrect CESE method implementation only for
  reference

- `toggle/**`: Feature toggle system

- `transform/**`: Integral transform

- `universe/**`: 3-dimensional geometry engine

### Python Package Structure

Python interface in `modmesh/`:

- `core.py`: Main Python API wrapping C++ extension
- `onedim/`: One-dimensional solver utilities
- `pilot/`: GUI application Python components
- `plot/`: Plotting utilities
- `profiling/`: Profiling result analysis
- `testing.py`: Test utilities
- `toggle.py`: Feature toggle Python interface

### Testing Structure

- **Python tests** (`tests/`): pytest-based
  - Named `test_*.py`
  - Run with `make pytest`

- **C++ tests** (`gtests/`): googletest-based
  - Named `test_nopython_*.cpp`
  - Run with `make gtest`

- **Profiling benchmarks** (`profiling/`): Performance tests
  - Named `profile_*.py`
  - Run with `make pyprof`, outputs to `profiling/results/`

## Code Style

For complete description of the code style, use the file `STYLE.rst`.  Important
styles are summarized here.

### C++ Style
- **Indentation**: 4 spaces (no tabs)
- **Naming**:
  - Classes: `CamelCase`
  - Functions/variables: `snake_case`
  - Member variables: `m_snake_case` (prefix with `m_`)
  - Constants: `UPPER_CASE` or `snake_case` (for foreign code interop)
  - Type aliases: `snake_case_t` or `snake_case_type`
- **Format**: Use clang-format (`.clang-format` in repo root)
- **Line length**: No hard limit, prefer < 120 characters
- **Includes**: Use angle brackets (`#include <...>`), not quotes
- **Standard**: C++23

### C++ STL Containers
- Replace `std::vector` with `SimpleCollector` when `value_type` is a
  fundamental type. Use `small_vector` for a small amount of data.
- Do not use STL containers for member data unless it is just in a prototype
  phase (add a `TODO` comment and a follow-up PR/issue).
- For local variables, STL is sometimes OK but discouraged.

### C++ Function Body Placement
- Move non-accessor function bodies to be outside the class declaration when
  the code is not 2-3 times longer than an accessor.
- Keep short accessors inline in the class declaration.
- If a function body is very simple (e.g., a single return or assignment),
  write it as a one-liner (e.g., `double value() const { return m_value; }`).

### C++ pybind11 Binding Style
- Separate constructors and other bindings (methods, properties, etc.) into two
  `(*this)` sections for readability. This can also be addressed in a future PR.

### Python Style
- **Indentation**: 4 spaces (no tabs)
- **Naming**:
  - Classes: `CamelCase`
  - Functions/variables: `snake_case`
  - Constants: `UPPER_CASE`
- **Line length**: 79 characters (PEP-8)
- **Format**: Use flake8

### File Format
- **Encoding**: UTF-8
- **Line endings**: Unix (`\n`)
- **Modelines**: Required at end of files

C++:
```cpp
// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
```

Python:
```python
# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
```

## Pull Request Guidelines

When opening a pull request, reference the related issue (e.g., "Related to
#725") instead of using closing keywords like "close #725", "closes #725",
or "fixes #725". We do not let PR and commit log comments to mandate the
management.

## Development Workflow

### Running Single Tests

Python (using pytest):
```bash
# Run specific test file
pytest tests/test_buffer.py

# Run specific test function
pytest tests/test_buffer.py::test_function_name

# Run with verbose output
pytest -v -s tests/test_buffer.py
```

C++ (using gtest):
```bash
# Build and run all gtests
make gtest

# Run specific test (after building)
./build/dev313/gtests/run_gtest --gtest_filter=TestSuiteName.TestName
```

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

**IMPORTANT**: Using Python virtual environments (venv, conda) is
**strongly discouraged** for modmesh development. The project is designed to
work with system Python. Virtual environment bugs are not actively resolved.

Use https://github.com/solvcon/devenv to build dependency from source and
install in user space. Do not install dependency system-wide. Installation of
any dependency requires user review and consent.

### Platform-Specific Notes

**macOS**: System Integrity Protection (SIP) may interfere with
`DYLD_LIBRARY_PATH`. The Makefile sets `PYTHONPATH` as a workaround.

**Windows**: Portable binaries available from GitHub Actions (see README.rst).

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

1. Create directory under `cpp/modmesh/`
2. Add header files with proper include guards
3. Update `cpp/modmesh/CMakeLists.txt` to include new sources
4. Add pybind11 bindings if Python access needed
5. Write tests in both `gtests/` and `tests/`

### Adding Python-Only Functionality

1. Add module to `modmesh/`
2. Update `modmesh/__init__.py` if needed
3. Write tests in `tests/`
4. Update `setup.py` packages list if adding new package

### Memory Management

- Use `ConcreteBuffer` for raw memory
- Use `SimpleArray` for typed multi-dimensional arrays
- Buffers support both ownership and non-owning views
- Python and C++ share the same buffer memory (zero-copy)

<!-- vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4: -->
