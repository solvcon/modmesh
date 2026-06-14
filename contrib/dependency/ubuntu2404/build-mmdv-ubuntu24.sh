#!/bin/bash
#
# Build modmesh's runtime dependencies from source on Ubuntu 24.04, with no
# dependency on the devenv tool. The per-package build recipes here are inlined
# translations of the scripts under devenv/scripts/build.d/ (zlib, openssl,
# sqlite, python, pybind11, cython, numpy, scipy, qt, pyside6).
#
# 4 sections (BASE, PYTHON, NUMPY, QT) are guarded by the corresponding
# MMDVBUILD_* environment variables.
#
# Before running this script, install the apt prerequisites. The two helper
# functions below print the exact commands; copy and run them yourself (the
# script never invokes apt). LLVM_INSTALL_DIR defaults to /usr/lib/llvm-22 (the
# apt llvm-22-dev path); override if needed.
#
# LLVM 22 is the libclang shiboken parses Qt headers with on Ubuntu 24.04 with
# GCC 16: clang-18 is too old (its parser chokes on libstdc++-16).  Override
# LLVM_INSTALL_DIR for another.  The patchelf apt package is the same binary
# the patchelf PyPI wheel ships; either works, the script installs the wheel
# anyway as a safety net.  libreadline is intentionally not listed: Python is
# built with --with-readline=editline, which uses libedit (libedit-dev).
# libmpdec is not packaged on Ubuntu 24.04; Python is built with
# --with-system-libmpdec=no to use its bundled copy.
#
# Usage:
#   ./build-mmdv-ubuntu24.sh
#       Builds everything (BASE + PYTHON + NUMPY + QT).  The default is "build
#       all" when no MMDVBUILD_* env var is set.  To limit to a single section,
#       set one of MMDVBUILD_BASE/PYTHON/NUMPY/QT=1 and leave the others unset.
#       In addition, MMDVBUILD_ALL=1 works the same as the default invocation.
#       Before any package is built the user is prompted with "Press Enter to
#       start the build, Ctrl-C to abort"; pass --no-confirm to skip.
#   ./build-mmdv-ubuntu24.sh --write-activate-only
#       Write only ${MMDV_BASE}/activate and exit.  Useful for refreshing the
#       activation script for an already-built mmdv without triggering any
#       build section.
#   ./build-mmdv-ubuntu24.sh --skip PKG [--skip PKG ...]
#       Skip a package within whatever sections are otherwise selected.  PKG
#       can be one of: zlib openssl sqlite python pybind11 cython numpy scipy
#       qt pyside6.  The flag accepts a single name, a comma-separated list
#       ("--skip openssl,sqlite"), and may be repeated.  The --skip=PKG form
#       also works.
#   ./build-mmdv-ubuntu24.sh --no-confirm
#       Skip the "Press Enter to start the build" prompt that fires after the
#       startup echo block.  Use in non-interactive runs (CI, scripts).
#   ./build-mmdv-ubuntu24.sh --print-prefix
#       Print MMDV_PREFIX (the path prefix that MMDV_BASE is derived from)
#       to stdout and exit.  Nothing else is printed and no directories are
#       created, so this is safe to capture: PREFIX=$(./script --print-prefix).
#   ./build-mmdv-ubuntu24.sh --print-apt
#       Print the apt prerequisite commands (the BASE/PYTHON/NUMPY section
#       and the QT section) to stdout and exit.  Nothing is built and no
#       directories are created.  The script never runs apt itself; copy the
#       output, review it, and run it.
#
# Overridable variables (search in the script for their defaults and
# descriptions; not repeating here):
#   Package versions:
#     PYTHON_VERSION: CPython release tag.
#     QT_MAJOR_VER: Qt major.minor version.
#     QT_SUB_VER: Qt patch version.
#     PYSIDE_VERSION: Qt for Python (pyside-setup) source release version.
#
#   Build settings:
#     MMDV_NP: Parallel build jobs.
#     MMDV_PREFIX: Path prefix to MMDV_BASE.
#     MMDV_BASE: Base directory holding the install prefix, including Python
#       and Qt version.
#     MMDV_DLDIR: Directory for downloaded tarballs (real dir or symlink,
#       depending on MMDV_SHARED_DLDIR).
#     MMDV_SHARED_DLDIR: Shared cache for downloaded tarballs across modmesh
#       development environments (mmdvs).

set -e
# Without pipefail, exit codes are lost through the `tee` pipe in with_log,
# so a failed build step would otherwise be silently ignored.
set -o pipefail

MMDV_WRITE_ACTIVATE_ONLY=0
MMDV_PRINT_PREFIX_ONLY=0
MMDV_PRINT_APT_ONLY=0
MMDV_NO_CONFIRM=0
MMDV_SKIP_LIST=""
MMDV_KNOWN_PKGS="zlib openssl sqlite python pybind11 cython numpy scipy qt pyside6"

mmdv_add_skip() {
  # Accept "pkg" or "pkg1,pkg2,..."; warn on unknown names but accept.
  local raw=$1 name
  for name in $(echo "${raw}" | tr ',' ' ') ; do
    [ -z "${name}" ] && continue
    case " ${MMDV_KNOWN_PKGS} " in
      *" ${name} "*) ;;
      *) echo "warning: --skip '${name}' is not a known package (known: ${MMDV_KNOWN_PKGS})" >&2 ;;
    esac
    MMDV_SKIP_LIST="${MMDV_SKIP_LIST} ${name}"
  done
}

mmdv_skip_p() {
  case " ${MMDV_SKIP_LIST} " in
    *" $1 "*) return 0 ;;
    *) return 1 ;;
  esac
}

while [ $# -gt 0 ] ; do
  case "$1" in
    --write-activate-only)
      MMDV_WRITE_ACTIVATE_ONLY=1
      ;;
    --print-prefix)
      MMDV_PRINT_PREFIX_ONLY=1
      ;;
    --print-apt)
      MMDV_PRINT_APT_ONLY=1
      ;;
    --no-confirm)
      MMDV_NO_CONFIRM=1
      ;;
    --skip)
      shift
      if [ $# -eq 0 ] ; then
        echo "--skip requires a package name" >&2
        exit 2
      fi
      mmdv_add_skip "$1"
      ;;
    --skip=*)
      mmdv_add_skip "${1#--skip=}"
      ;;
    -h|--help)
      sed -n '2,/^$/p' "$0" | sed 's/^#//' >&2
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
  shift
done

# CPython release tag.
PYTHON_VERSION=${PYTHON_VERSION:-3.14.5}
# Qt major.minor version.
QT_MAJOR_VER=${QT_MAJOR_VER:-6.11}
# Qt patch version.
QT_SUB_VER=${QT_SUB_VER:-1}
# Qt for Python (pyside-setup) source release version.
PYSIDE_VERSION=${PYSIDE_VERSION:-${QT_MAJOR_VER}.${QT_SUB_VER}}

# Default to a full BASE+PYTHON+NUMPY+QT build when the caller has not selected
# a specific section. Setting any one of MMDVBUILD_* to "1" disables this
# default and runs only the explicitly selected sections.
if [ -z "${MMDVBUILD_ALL:-}${MMDVBUILD_BASE:-}${MMDVBUILD_PYTHON:-}${MMDVBUILD_NUMPY:-}${MMDVBUILD_QT:-}" ] ; then
  MMDVBUILD_ALL=1
fi

# Parallel build jobs.
MMDV_NP=${MMDV_NP:-$(nproc)}
# Path prefix to MMDV_BASE.
MMDV_PREFIX=${MMDV_PREFIX:-${HOME}/var/mmdv/ubuntu2404}

# --print-prefix is honored as soon as MMDV_PREFIX is resolved so the caller
# can capture the value without triggering any side effects (mkdir,
# activate-write, build).
if [ "${MMDV_PRINT_PREFIX_ONLY}" = "1" ] ; then
  echo "MMDV_PREFIX=${MMDV_PREFIX}"
  exit 0
fi

# Base directory holding the install prefix, including Python and Qt version.
MMDV_BASE=${MMDV_BASE:-${MMDV_PREFIX}-py${PYTHON_VERSION}-qt${QT_MAJOR_VER}.${QT_SUB_VER}}
# Directory for downloaded tarballs.
MMDV_DLDIR=${MMDV_DLDIR:-${MMDV_BASE}/downloaded}
# Shared cache for downloaded tarballs across modmesh development environments
# (mmdvs).  When non-empty, ${MMDV_DLDIR} is created as a symlink that points
# here.  The "-" (not ":-") form lets the caller explicitly set
# MMDV_SHARED_DLDIR= to opt out and keep a per-mmdv directory instead.
MMDV_SHARED_DLDIR=${MMDV_SHARED_DLDIR-${HOME}/var/mmdv/downloaded}

# Directory for building from source.
MMDV_SRCDIR=${MMDV_BASE}/src
# Install root (user-space).
MMDV_USRDIR=${MMDV_BASE}/usr

# Directories under ${MMDV_BASE} and the activation script are created below,
# after the confirmation prompt (or immediately for the --write-activate-only
# path). We deliberately do not touch the filesystem here so that aborting the
# prompt leaves no trace.

PY=${MMDV_USRDIR}/bin/python3
# Put our prefix on PATH so console scripts (cython, etc.) installed into
# ${MMDV_USRDIR}/bin are found by downstream builders such as meson.
export PATH="${MMDV_USRDIR}/bin:${PATH}"

mmdv_apt_base_cmd() {
  # Print the apt command for the BASE/PYTHON/NUMPY sections. The script
  # never runs apt itself; copy the output, review it, and run it. doxygen
  # is the system half of the documentation C++ API path (see doc/README.md).
  cat <<'EOF'
sudo apt install -y \
  build-essential gcc g++ make cmake ninja-build pkg-config \
  git curl xz-utils gfortran libopenblas-dev \
  libffi-dev libbz2-dev liblzma-dev libgdbm-dev \
  libncurses-dev uuid-dev tk-dev libedit-dev libexpat1-dev \
  doxygen
EOF
}

mmdv_apt_qt_cmd() {
  # Print the apt command for the QT section (Qt + pyside6 build deps).
  # The X11/XCB -dev packages are the full set Qt's xcb platform plugin
  # needs at configure time. Without all of them Qt silently builds without
  # the xcb QPA plugin, leaving only offscreen/minimal, so a real (or xvfb)
  # X display cannot be used. build_qt force-enables FEATURE_xcb so a missing
  # dependency fails the configure loudly instead of dropping the plugin.
  # CI installs the runtime counterparts of these -dev packages in
  # .github/actions/setup_linux/action.yml; keep the two sets in sync.
  cat <<'EOF'
sudo apt install -y \
  llvm-22-dev clang-22 libclang-22-dev patchelf \
  libxkbcommon-dev libxkbcommon-x11-dev libfontconfig1-dev \
  libfreetype-dev libdbus-1-dev libgl1-mesa-dev libglu1-mesa-dev \
  libx11-dev libx11-xcb-dev libxext-dev libxfixes-dev libxi-dev \
  libxrender-dev libxcb1-dev libxcb-glx0-dev libxcb-cursor-dev \
  libxcb-icccm4-dev libxcb-image0-dev libxcb-keysyms1-dev \
  libxcb-randr0-dev libxcb-render0-dev libxcb-render-util0-dev \
  libxcb-shape0-dev libxcb-shm0-dev libxcb-sync-dev libxcb-util-dev \
  libxcb-xfixes0-dev libxcb-xinerama0-dev libxcb-xkb-dev
EOF
}

# --print-apt prints both apt prerequisite commands and exits, before any
# filesystem side effect (mkdir, activate-write, build). The apt helper
# functions must be defined first, so this is honored here rather than
# alongside --print-prefix above.
if [ "${MMDV_PRINT_APT_ONLY}" = "1" ] ; then
  mmdv_apt_base_cmd
  echo
  mmdv_apt_qt_cmd
  exit 0
fi

mmdv_write_activate() {
  # Write ${MMDV_BASE}/activate. The activation script is self-locating (reads
  # its own path at source time) so the mmdv directory can be moved without
  # rewriting the file. Sourcing it prepends our bin to PATH and our lib to
  # LD_LIBRARY_PATH, strips any leaked /var/Qt/ entries from both, and defines
  # mmdv_deactivate to restore the original environment.
  local target=${MMDV_BASE}/activate
  cat > "${target}" <<'ACTIVATE_EOF'
# shellcheck shell=bash
# Activate this MMDV: source this file (do not execute).
#
#   $ source <path-to-this-file>/activate
#   $ mmdv_deactivate   # to restore the original environment
#
# Compatible with bash and zsh.

if [ -n "${MMDV_BASE:-}" ] ; then
  echo "MMDV '${MMDV_BASE##*/}' is already active." \
       "Run mmdv_deactivate first." >&2
  return 1 2>/dev/null || exit 1
fi

# Resolve the directory this file lives in MMDV_BASE.
if [ -n "${BASH_VERSION:-}" ] ; then
  _mmdv_self=${BASH_SOURCE[0]}
elif [ -n "${ZSH_VERSION:-}" ] ; then
  _mmdv_self=${(%):-%x}
else
  _mmdv_self=$0
fi
MMDV_BASE=$(cd "$(dirname "${_mmdv_self}")" && pwd)
unset _mmdv_self
export MMDV_BASE
export MMDV_USRDIR=${MMDV_BASE}/usr

# Snapshot the current environment so mmdv_deactivate can restore it.
# These leading-underscore vars are private to the activation script.
export _MMDV_OLD_PATH=${PATH}
export _MMDV_OLD_LD_LIBRARY_PATH=${LD_LIBRARY_PATH-}
export _MMDV_OLD_CMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH-}
export _MMDV_OLD_PS1=${PS1-}
export _MMDV_OLD_QT_QPA_PLATFORM=${QT_QPA_PLATFORM-}
export _MMDV_HAD_LD_LIBRARY_PATH=${LD_LIBRARY_PATH+1}
export _MMDV_HAD_CMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH+1}
export _MMDV_HAD_QT_QPA_PLATFORM=${QT_QPA_PLATFORM+1}

# A pre-existing Qt (e.g. /home/$USER/var/Qt/6.x) leaking through PATH or
# LD_LIBRARY_PATH will be loaded ahead of this mmdv's freshly built Qt and
# break with "version `Qt_6.x' not found". Strip those.
PATH=$(printf '%s' "${PATH}" | tr ':' '\n' | grep -v '/var/Qt/' | paste -sd: -)
export PATH=${MMDV_USRDIR}/bin:${PATH}

LD_LIBRARY_PATH=$(printf '%s' "${LD_LIBRARY_PATH-}" \
  | tr ':' '\n' | grep -v '/var/Qt/' | paste -sd: -)
if [ -n "${LD_LIBRARY_PATH}" ] ; then
  export LD_LIBRARY_PATH=${MMDV_USRDIR}/lib:${LD_LIBRARY_PATH}
else
  export LD_LIBRARY_PATH=${MMDV_USRDIR}/lib
fi

if [ -n "${CMAKE_PREFIX_PATH-}" ] ; then
  export CMAKE_PREFIX_PATH=${MMDV_USRDIR}:${CMAKE_PREFIX_PATH}
else
  export CMAKE_PREFIX_PATH=${MMDV_USRDIR}
fi

# modmesh's _modmesh module constructs a QApplication during PyInit__modmesh;
# without a display or QT_QPA_PLATFORM Qt aborts in
# QGuiApplicationPrivate::createPlatformIntegration. Default to the offscreen
# platform when nothing is set and no display is available, so plain `import
# modmesh` and `make pytest` work in headless shells.  If a real display
# becomes available later the user can `export QT_QPA_PLATFORM=xcb` (or unset
# it) without re-sourcing.
if [ -z "${QT_QPA_PLATFORM-}" ] \
   && [ -z "${DISPLAY:-}" ] \
   && [ -z "${WAYLAND_DISPLAY:-}" ] ; then
  export QT_QPA_PLATFORM=offscreen
fi

# Mark the prompt so the active mmdv is visible.
if [ -n "${BASH_VERSION:-}" ] || [ -n "${ZSH_VERSION:-}" ] ; then
  PS1="(${MMDV_BASE##*/}) ${PS1-}"
  export PS1
fi

mmdv_deactivate() {
  if [ -z "${MMDV_BASE:-}" ] ; then
    echo "No active MMDV." >&2
    return 1
  fi
  export PATH=${_MMDV_OLD_PATH}
  if [ -n "${_MMDV_HAD_LD_LIBRARY_PATH:-}" ] ; then
    export LD_LIBRARY_PATH=${_MMDV_OLD_LD_LIBRARY_PATH}
  else
    unset LD_LIBRARY_PATH
  fi
  if [ -n "${_MMDV_HAD_CMAKE_PREFIX_PATH:-}" ] ; then
    export CMAKE_PREFIX_PATH=${_MMDV_OLD_CMAKE_PREFIX_PATH}
  else
    unset CMAKE_PREFIX_PATH
  fi
  if [ -n "${_MMDV_HAD_QT_QPA_PLATFORM:-}" ] ; then
    export QT_QPA_PLATFORM=${_MMDV_OLD_QT_QPA_PLATFORM}
  else
    unset QT_QPA_PLATFORM
  fi
  PS1=${_MMDV_OLD_PS1}
  export PS1
  unset _MMDV_OLD_PATH _MMDV_OLD_LD_LIBRARY_PATH \
        _MMDV_OLD_CMAKE_PREFIX_PATH _MMDV_OLD_PS1 \
        _MMDV_OLD_QT_QPA_PLATFORM \
        _MMDV_HAD_LD_LIBRARY_PATH _MMDV_HAD_CMAKE_PREFIX_PATH \
        _MMDV_HAD_QT_QPA_PLATFORM
  unset MMDV_BASE MMDV_USRDIR
  unset -f mmdv_deactivate
}
ACTIVATE_EOF
  # Do not set the execution bit of "${target}"; it should be sourced.
  echo "wrote activation script: ${target}"
}

echo "MMDV_NP=${MMDV_NP}"
echo "MMDV_PREFIX=${MMDV_PREFIX}"
echo "MMDV_BASE=${MMDV_BASE}"
echo "MMDV_DLDIR=${MMDV_DLDIR}"
echo "MMDV_SHARED_DLDIR=${MMDV_SHARED_DLDIR}"
echo "MMDV_SRCDIR=${MMDV_SRCDIR}"
echo "MMDV_USRDIR=${MMDV_USRDIR}"
echo "PYTHON_VERSION=${PYTHON_VERSION}"
if [ -n "${MMDV_SKIP_LIST# }" ] ; then
  echo "MMDV_SKIP_LIST=${MMDV_SKIP_LIST# }"
fi
echo "ready to build"

# --write-activate-only is its own explicit confirmation: create the mmdv
# directory just enough to drop the activate file in, then exit.
if [ "${MMDV_WRITE_ACTIVATE_ONLY}" = "1" ] ; then
  mkdir -p "${MMDV_BASE}"
  mmdv_write_activate
  exit 0
fi

# If MMDV_SHARED_DLDIR is configured, it must exist already.  The script never
# auto-creates it.  Check before any filesystem change (mkdir, symlink, prompt)
# so a typo or unmounted cache fails immediately instead of half-creating the
# per-mmdv tree.
if [ -n "${MMDV_SHARED_DLDIR}" ] && [ ! -d "${MMDV_SHARED_DLDIR}" ] ; then
  echo "MMDV_SHARED_DLDIR=${MMDV_SHARED_DLDIR} does not exist;" \
       "create it (e.g. \`mkdir -p ${MMDV_SHARED_DLDIR}\`)" \
       "or unset MMDV_SHARED_DLDIR." >&2
  exit 1
fi

# Final confirmation before any package is built. Skip with --no-confirm in
# non-interactive contexts (CI, automated runs). No directory or activation
# file is created until the user has confirmed, so aborting at the prompt
# leaves the filesystem untouched.
if [ "${MMDV_NO_CONFIRM}" != "1" ] ; then
  # `test -r /dev/tty` returns true even with no controlling terminal, so try
  # to actually open it; that fails with ENXIO when there is no controlling tty
  # (e.g. under setsid or `< /dev/null` in CI).
  if ! { : </dev/tty ; } 2>/dev/null ; then
    echo "no controlling tty for confirmation prompt; rerun with --no-confirm" >&2
    exit 2
  fi
  read -r -p "Press Enter to start the build, Ctrl-C to abort: " _ </dev/tty
fi

mkdir -p "${MMDV_USRDIR}" "${MMDV_SRCDIR}"
if [ -n "${MMDV_SHARED_DLDIR}" ] ; then
  # MMDV_SHARED_DLDIR is set: make MMDV_DLDIR a symlink into the shared cache.
  # The shared directory's existence was verified above; we do not auto-create
  # it.  Refuse to clobber a non-symlink directory at MMDV_DLDIR to avoid
  # losing data; the caller has to move it (or unset MMDV_SHARED_DLDIR) first.
  if [ -L "${MMDV_DLDIR}" ] ; then
    if [ "$(readlink "${MMDV_DLDIR}")" != "${MMDV_SHARED_DLDIR}" ] ; then
      rm "${MMDV_DLDIR}"
      ln -s "${MMDV_SHARED_DLDIR}" "${MMDV_DLDIR}"
    fi
  elif [ -d "${MMDV_DLDIR}" ] ; then
    echo "MMDV_DLDIR=${MMDV_DLDIR} exists as a real directory but" \
         "MMDV_SHARED_DLDIR is set; move its contents to" \
         "${MMDV_SHARED_DLDIR} and remove the directory," \
         "or unset MMDV_SHARED_DLDIR." >&2
    exit 1
  else
    ln -s "${MMDV_SHARED_DLDIR}" "${MMDV_DLDIR}"
  fi
else
  # No shared cache: MMDV_DLDIR is a real per-mmdv directory.
  mkdir -p "${MMDV_DLDIR}"
fi
mmdv_write_activate

####
# Helpers (translated from devenv/scripts/func.d/build_utils)
####

download_md5() {
  local fn=$1 url=$2 md5hash=${3:-}
  local loc=${MMDV_DLDIR}/${fn}
  local calc=""
  [ -e "${loc}" ] && calc=$(md5sum "${loc}" | cut -d ' ' -f 1)
  if [ ! -e "${loc}" ] || { [ -n "${md5hash}" ] && [ "${md5hash}" != "${calc}" ]; } ; then
    echo "Downloading ${url}"
    curl -fsSL -o "${loc}" "${url}"
    calc=$(md5sum "${loc}" | cut -d ' ' -f 1)
  fi
  if [ -n "${md5hash}" ] && [ "${md5hash}" != "${calc}" ] ; then
    echo "${fn} md5 mismatch: expected ${md5hash} got ${calc} (continuing)"
  fi
}

with_log() {
  local log=$1 ; shift
  echo "run: $*" | tee "${log}"
  { time "$@" ; } 2>&1 | tee -a "${log}"
}

unpack() {
  local fn=$1 destdir=$2
  pushd "${MMDV_SRCDIR}" > /dev/null
  rm -rf "${destdir}"
  tar xf "${MMDV_DLDIR}/${fn}"
  popd > /dev/null
}

####
# Timing: record wall-clock time spent in each build_<pkg> call and print a
# summary table on EXIT. If a build_* aborts the script mid-way (set -e), the
# trap still records that package as FAIL with the partial elapsed time.
####

MMDV_OVERALL_START=$(date +%s)
MMDV_TIMINGS=""
MMDV_CURRENT_PKG=""
MMDV_CURRENT_START=0

mmdv_time() {
  local fn=$1 label=$2 end elapsed status
  MMDV_CURRENT_PKG=${label}
  MMDV_CURRENT_START=$(date +%s)
  "${fn}"
  end=$(date +%s)
  elapsed=$((end - MMDV_CURRENT_START))
  if mmdv_skip_p "${label}" ; then
    status="skipped"
  else
    status="ok"
  fi
  MMDV_TIMINGS+="${label}|${status}|${elapsed}"$'\n'
  MMDV_CURRENT_PKG=""
}

mmdv_fmt_time() {
  local s=$1
  if [ "${s}" -ge 3600 ] ; then
    printf '%dh%02dm%02ds' $((s/3600)) $(((s%3600)/60)) $((s%60))
  elif [ "${s}" -ge 60 ] ; then
    printf '%dm%02ds' $((s/60)) $((s%60))
  else
    printf '%ds' "${s}"
  fi
}

mmdv_print_timings() {
  local exit_code=$?
  set +e
  local end_ts overall part label sec status pkg
  end_ts=$(date +%s)
  if [ -n "${MMDV_CURRENT_PKG}" ] ; then
    part=$((end_ts - MMDV_CURRENT_START))
    if [ "${exit_code}" -eq 0 ] ; then
      label="RUNNING"
    else
      label="FAIL"
    fi
    MMDV_TIMINGS+="${MMDV_CURRENT_PKG}|${label}|${part}"$'\n'
  fi
  overall=$((end_ts - MMDV_OVERALL_START))
  # Print only if at least one package was timed or the script ran a
  # non-trivial time (avoids noise on --help / --write-activate-only).
  if [ -z "${MMDV_TIMINGS}" ] && [ "${overall}" -lt 2 ] ; then
    return 0
  fi
  printf '\n=== build timings ===\n'
  printf '%-10s %-8s %10s\n' "package" "status" "time"
  printf '%-10s %-8s %10s\n' "----------" "--------" "----------"
  printf '%s' "${MMDV_TIMINGS}" | while IFS='|' read -r pkg status sec ; do
    [ -z "${pkg}" ] && continue
    printf '%-10s %-8s %10s\n' "${pkg}" "${status}" "$(mmdv_fmt_time "${sec}")"
  done
  printf '%-10s %-8s %10s\n' "----------" "--------" "----------"
  printf '%-10s %-8s %10s (exit %d)\n' \
    "TOTAL" "-" "$(mmdv_fmt_time "${overall}")" "${exit_code}"
}
trap mmdv_print_timings EXIT

####
# Build functions (translated from devenv/scripts/build.d/<pkg>)
####

build_zlib() {
  mmdv_skip_p zlib && { echo "skip: zlib" ; return 0 ; }
  local ver=1.3.1 full fn
  full=zlib-${ver} ; fn=${full}.tar.gz
  download_md5 "${fn}" \
    "https://github.com/madler/zlib/archive/refs/tags/v${ver}.tar.gz" \
    ddb17dbbf2178807384e57ba0d81e6a1
  unpack "${fn}" "${full}"
  pushd "${MMDV_SRCDIR}/${full}" > /dev/null
    with_log configure.log ./configure --prefix="${MMDV_USRDIR}"
    with_log make.log make -j "${MMDV_NP}"
    with_log install.log make install
  popd > /dev/null
}

build_openssl() {
  mmdv_skip_p openssl && { echo "skip: openssl" ; return 0 ; }
  local ver=1.1.1m full fn
  full=openssl-${ver} ; fn=${full}.tar.gz
  download_md5 "${fn}" \
    "https://www.openssl.org/source/${fn}" \
    8ec70f665c145c3103f6e330f538a9db
  unpack "${fn}" "${full}"
  pushd "${MMDV_SRCDIR}/${full}" > /dev/null
    with_log configure.log ./config \
      --prefix="${MMDV_USRDIR}" \
      --openssldir="${MMDV_USRDIR}/share/ssl"
    with_log make.log make -j "${MMDV_NP}"
    with_log install.log make -j "${MMDV_NP}" install
  popd > /dev/null
}

build_sqlite() {
  mmdv_skip_p sqlite && { echo "skip: sqlite" ; return 0 ; }
  local ver=3360000 full fn
  full=sqlite-autoconf-${ver} ; fn=${full}.tar.gz
  download_md5 "${fn}" \
    "https://www.sqlite.org/2021/${fn}" \
    f5752052fc5b8e1b539af86a3671eac7
  unpack "${fn}" "${full}"
  pushd "${MMDV_SRCDIR}/${full}" > /dev/null
    with_log configure.log ./configure --prefix="${MMDV_USRDIR}"
    with_log make.log make
    with_log install.log make install
  popd > /dev/null
}

build_python() {
  mmdv_skip_p python && { echo "skip: python" ; return 0 ; }
  local ver=${PYTHON_VERSION} full fn
  full=Python-${ver} ; fn=${full}.tar.xz
  # MD5 left empty; checksum varies per release and we don't pin it here.
  download_md5 "${fn}" \
    "https://www.python.org/ftp/python/${ver}/${fn}" \
    ""
  unpack "${fn}" "${full}"
  pushd "${MMDV_SRCDIR}/${full}" > /dev/null
    export CPPFLAGS="-I${MMDV_USRDIR}/include ${CPPFLAGS:-}"
    local ldflags="-Wl,--no-as-needed"
    ldflags="${ldflags} -Wl,-rpath,${MMDV_USRDIR}/lib"
    ldflags="${ldflags} -Wl,-rpath,${MMDV_USRDIR}/lib64"
    ldflags="${ldflags} -L${MMDV_USRDIR}/lib"
    ldflags="${ldflags} -L${MMDV_USRDIR}/lib64"
    export LDFLAGS="${ldflags} ${LDFLAGS:-}"
    with_log configure.log ./configure \
      --prefix="${MMDV_USRDIR}" \
      --enable-shared \
      --enable-ipv6 \
      --enable-optimizations \
      --without-ensurepip \
      --with-system-expat \
      --with-system-libmpdec=no \
      --with-readline=editline \
      --with-lto \
      --with-openssl="${MMDV_USRDIR}" \
      --with-system-ffi \
      --enable-loadable-sqlite-extensions
    with_log profile-opt.log make profile-opt -j "${MMDV_NP}"
    with_log install.log make install -j "${MMDV_NP}"
  popd > /dev/null

  # Bootstrap pip (Python was configured --without-ensurepip). get-pip only
  # installs pip; install setuptools+wheel so subsequent pip installs that
  # disable build isolation (or use setup.py) can find them.
  curl -fsSL https://bootstrap.pypa.io/get-pip.py | "${PY}"
  rm -f "${MMDV_USRDIR}/bin/pip"
  "${PY}" -m pip install -U setuptools wheel
  # Update pip.
  "${PY}" -m pip install -U pip
}

build_pybind11() {
  mmdv_skip_p pybind11 && { echo "skip: pybind11" ; return 0 ; }
  local ver=2.13.6 full fn
  full=pybind11-${ver} ; fn=${full}.tar.gz
  download_md5 "${fn}" \
    "https://github.com/pybind/pybind11/archive/refs/tags/v${ver}.tar.gz" \
    a04dead9c83edae6d84e2e343da7feeb
  unpack "${fn}" "${full}"
  mkdir -p "${MMDV_SRCDIR}/${full}/build"
  pushd "${MMDV_SRCDIR}/${full}/build" > /dev/null
    with_log cmake.log cmake \
      -DPYTHON_EXECUTABLE:FILEPATH="${PY}" \
      -DCMAKE_INSTALL_PREFIX="${MMDV_USRDIR}" \
      -DPYBIND11_TEST=OFF \
      -DCMAKE_BUILD_TYPE=Release \
      ..
    with_log make.log make -j "${MMDV_NP}"
    with_log install.log make install
  popd > /dev/null
  pushd "${MMDV_SRCDIR}/${full}" > /dev/null
    with_log setup.log "${PY}" -m pip install .
  popd > /dev/null
}

build_cython() {
  mmdv_skip_p cython && { echo "skip: cython" ; return 0 ; }
  local ver=3.0.12 full fn
  full=cython-${ver} ; fn=${full}.tar.gz
  download_md5 "${fn}" \
    "https://github.com/cython/cython/archive/refs/tags/${ver}.tar.gz" \
    194658f8ae1ae8804f864d4e147fddf6
  unpack "${fn}" "${full}"
  pushd "${MMDV_SRCDIR}/${full}" > /dev/null
    with_log install.log "${PY}" -m pip install .
  popd > /dev/null
}

build_numpy() {
  mmdv_skip_p numpy && { echo "skip: numpy" ; return 0 ; }
  local ver=2.2.4 full fn
  full=numpy-${ver} ; fn=${full}.tar.gz
  download_md5 "${fn}" \
    "https://github.com/numpy/numpy/releases/download/v${ver}/${fn}" \
    56232f4a69b03dd7a87a55fffc5f2ebc
  unpack "${fn}" "${full}"
  pushd "${MMDV_SRCDIR}/${full}" > /dev/null
    rm -f site.cfg
    # Point numpy at apt's libopenblas-dev. NPY_BLAS_ORDER is honored by the
    # meson build path; site.cfg is kept for older fallback paths.
    export NPY_BLAS_ORDER=openblas
    cat > site.cfg <<EOF
[openblas]
libraries = openblas
library_dirs = /usr/lib/x86_64-linux-gnu:${MMDV_USRDIR}/lib
include_dirs = /usr/include/x86_64-linux-gnu/openblas-pthread:/usr/include/openblas:${MMDV_USRDIR}/include
runtime_library_dirs = /usr/lib/x86_64-linux-gnu
EOF
    with_log dependency.log "${PY}" -m pip install -r requirements/build_requirements.txt
    # GCC 16 trunk on Ubuntu 24.04 rejects the AVX512 `evex512` target
    # attribute that numpy 2.2.x uses, so cap cpu-dispatch at AVX2 instead of
    # MAX.  Pass via --config-settings so the pip-driven meson rebuild honors
    # it (not just an out-of-tree `spin build`).
    with_log install.log "${PY}" -m pip install . --no-build-isolation \
      --config-settings="setup-args=-Dcpu-dispatch=SSE3 SSSE3 SSE41 POPCNT SSE42 AVX F16C FMA3 AVX2"
  popd > /dev/null
  "${PY}" -c "import numpy as np ; np.show_config()"
}

build_scipy() {
  mmdv_skip_p scipy && { echo "skip: scipy" ; return 0 ; }
  local ver=1.15.2 full fn
  full=scipy-${ver} ; fn=${full}.tar.gz
  download_md5 "${fn}" \
    "https://github.com/scipy/scipy/releases/download/v${ver}/${fn}" \
    515fc1544d7617b38fe5a9328538047b
  unpack "${fn}" "${full}"
  pushd "${MMDV_SRCDIR}/${full}" > /dev/null
    # dev.py 1.15.2 breaks on click/rich-click >=8.2 with "Sentinel object is
    # not subscriptable" when loading doit tasks, so bypass dev.py and let pip
    # drive the meson build directly.
    with_log dependency.log "${PY}" -m pip install -r requirements/build.txt
    # Ubuntu's pybind11-dev (2.11) at /usr/include shadows our 2.13.6 and
    # rejects scipy's 3-arg PYBIND11_MODULE. -isystem orders our headers ahead
    # of /usr/include so scipy compiles against the right pybind11.
    local saved_cxxflags=${CXXFLAGS:-}
    export CXXFLAGS="-isystem ${MMDV_USRDIR}/include ${saved_cxxflags}"
    with_log install.log "${PY}" -m pip install . --no-build-isolation
    export CXXFLAGS="${saved_cxxflags}"
  popd > /dev/null
}

build_qt() {
  mmdv_skip_p qt && { echo "skip: qt" ; return 0 ; }
  local major=${QT_MAJOR_VER}
  local sub=${QT_SUB_VER}
  local ver=${major}.${sub}
  local full=qt-${ver}
  local pkgfolder=qt-everywhere-src-${ver}
  local fn=${full}.tar.xz
  local url="https://download.qt.io/official_releases/qt/${major}"
  url="${url}/${ver}/single/qt-everywhere-src-${ver}.tar.xz"
  download_md5 "${fn}" "${url}" 25d4d1dd74c92b978f164e8f20805985
  # A pre-existing Qt install may leak through LD_LIBRARY_PATH or
  # CMAKE_PREFIX_PATH and cause the freshly built Qt 6.11.1 tools (rcc/moc/...)
  # to load the older libQt6Core at runtime and fail with "version `Qt_6.11'
  # not found".  Strip them out.
  unset LD_LIBRARY_PATH CMAKE_PREFIX_PATH QT_ROOT QT_VER
  PATH=$(printf '%s' "${PATH}" | tr ':' '\n' | grep -v '/var/Qt/' | paste -sd:)
  if [ -d "${MMDV_SRCDIR}/${full}" ] ; then
    echo "Qt source already at ${MMDV_SRCDIR}/${full}; skipping extract"
  else
    pushd "${MMDV_SRCDIR}" > /dev/null
      echo "Extracting Qt source (large; takes a few minutes) ..."
      tar xf "${MMDV_DLDIR}/${fn}"
      mv "${pkgfolder}" "${full}"
    popd > /dev/null
  fi

  # Clean previous build dir by default; MMDV_QTNOCLEAN=1 to reuse it.
  if [ "${MMDV_QTNOCLEAN:-}" != "1" ] ; then
    rm -rf "${MMDV_SRCDIR}/${full}/build"
  fi
  mkdir -p "${MMDV_SRCDIR}/${full}/build"
  pushd "${MMDV_SRCDIR}/${full}/build" > /dev/null
    local cfgcmd=(cmake)
    cfgcmd+=("-DCMAKE_INSTALL_PREFIX=${MMDV_USRDIR}")
    cfgcmd+=("-DCMAKE_BUILD_TYPE=Release")
    # Disable Qt modules we do not need (mirrors devenv defaults).
    local m
    for m in qtquicktimeline qtquick3d qtgraphs qt5compat qtactiveqt \
             qtcharts qtcoap qtconnectivity qtdatavis3d qtwebsockets \
             qthttpserver qttools qtdoc qtlottie qtmqtt qtnetworkauth \
             qtopcua qtserialport qtlocation qtpositioning \
             qtquick3dphysics qtremoteobjects qtscxml qtsensors \
             qtserialbus qtspeech qttranslations qtvirtualkeyboard \
             qtwayland qtwebchannel qtwebengine qtwebview \
             qtquickeffectmaker qtgrpc qtmultimedia ; do
      cfgcmd+=("-DBUILD_${m}=OFF")
    done
    # Force the xcb (X11) platform plugin on. Qt otherwise silently drops it
    # when an XCB dev dependency is missing, leaving only the offscreen and
    # minimal QPA plugins and making a real (or xvfb) X display unusable.
    # Force-enabling turns a missing dependency into a configure-time failure
    # here instead of a runtime "Could not find the Qt platform plugin xcb".
    # xcb_xlib is the Xlib/XCB interop the plugin relies on. The required
    # -dev packages are listed in mmdv_apt_qt_cmd above.
    cfgcmd+=("-DFEATURE_xcb=ON")
    cfgcmd+=("-DFEATURE_xcb_xlib=ON")
    cfgcmd+=("-DQT_ALLOW_SYMLINK_IN_PATHS=ON")
    cfgcmd+=("-DCMAKE_PREFIX_PATH=${MMDV_USRDIR}")
    cfgcmd+=("-G" "Ninja")
    cfgcmd+=("..")

    if [ "${DVQT_NOCONFIG:-}" != "1" ] ; then
      with_log configure.log "${cfgcmd[@]}"
    fi
    if [ "${DVQT_NOBUILD:-}" != "1" ] ; then
      with_log build.log cmake --build . --parallel
    fi
    if [ "${DVQT_NOINSTALL:-}" != "1" ] ; then
      with_log install.log cmake --install .
    fi
  popd > /dev/null
}

build_pyside6() {
  mmdv_skip_p pyside6 && { echo "skip: pyside6" ; return 0 ; }
  local ver=${PYSIDE_VERSION} full fn url
  full=pyside-setup-everywhere-src-${ver} ; fn=${full}.tar.xz
  # Official Qt for Python source release, verified against the published
  # .md5 sidecar (download.qt.io also serves a matching .sha256:
  # 6ffd9835bb0dd2c56f061d62f1616bb1707cfc0202b80e3165d6be087f3965e2).
  url="https://download.qt.io/official_releases/QtForPython/pyside6"
  url="${url}/PySide6-${ver}-src/${fn}"
  download_md5 "${fn}" "${url}" a6fe3db5855d3cd09a381d0aca7d7f5e
  unpack "${fn}" "${full}"
  pushd "${MMDV_SRCDIR}/${full}" > /dev/null
    # pyside-setup 6.9.2 has a regression in _get_make: the "make" branch
    # returns a str while others return Path, so a later .is_absolute() call
    # crashes.  Use ninja, which returns a Path.
    with_log install.log "${PY}" setup.py install \
      --qtpaths="${QTPATHS}" \
      --verbose-build \
      --ignore-git \
      --no-qt-tools \
      --enable-numpy-support \
      --parallel="${MMDV_NP}" \
      --make-spec=ninja
  popd > /dev/null
}

####
# Base section
####
if [[ "${MMDVBUILD_ALL:-}" == "1" || "${MMDVBUILD_BASE:-}" == "1" ]] ; then

mmdv_time build_zlib zlib
mmdv_time build_openssl openssl
mmdv_time build_sqlite sqlite

else

echo "Set \${MMDVBUILD_ALL} or \${MMDVBUILD_BASE} to build BASE section"

fi

####
# Python section
####
if [[ "${MMDVBUILD_ALL:-}" == "1" || "${MMDVBUILD_PYTHON:-}" == "1" ]] ; then

echo "Python build uses PGO; expect ~20 min"
mmdv_time build_python python
"${PY}" -m pip install -U flake8 autopep8 black pytest jsonschema certifi
"${PY}" -m pip install -U nose boto paramiko
# Documentation toolchain (Sphinx-based; see doc/README.md). doxygen, the
# system half of the C++ API path, is in mmdv_apt_base_cmd above.
"${PY}" -m pip install -U sphinx myst-parser pydata-sphinx-theme \
  breathe sphinxcontrib-bibtex
mmdv_time build_pybind11 pybind11

else

echo "Set \${MMDVBUILD_ALL} or \${MMDVBUILD_PYTHON} to build PYTHON section"

fi

####
# Numpy section
####
if [[ "${MMDVBUILD_ALL:-}" == "1" || "${MMDVBUILD_NUMPY:-}" == "1" ]] ; then

mmdv_time build_cython cython
# NOFORTRAN mirrors the macOS reference; with apt's gfortran installed, scipy
# below will still pick gfortran up via meson's compiler detection.
NOFORTRAN=1 mmdv_time build_numpy numpy
mmdv_time build_scipy scipy

CERT_PATH=$("${PY}" -m certifi)
export SSL_CERT_FILE=${CERT_PATH}
export REQUESTS_CA_BUNDLE=${CERT_PATH}
"${PY}" -m pip install -U matplotlib

else

echo "Set \${MMDVBUILD_ALL} or \${MMDVBUILD_NUMPY} to build NUMPY section"

fi

####
# Qt section
####
if [[ "${MMDVBUILD_ALL:-}" == "1" || "${MMDVBUILD_QT:-}" == "1" ]] ; then

# LLVM_INSTALL_DIR points at the libclang shiboken should use.  Defaults to the
# apt path for llvm-22, the newest stable and supported by PySide 6.11.1 on
# Ubuntu 24.04 (see the header comment); override if your llvm-22-dev install
# lives elsewhere.
LLVM_INSTALL_DIR=${LLVM_INSTALL_DIR:-/usr/lib/llvm-22}
if [ ! -d "${LLVM_INSTALL_DIR}" ] ; then
  echo "LLVM_INSTALL_DIR='${LLVM_INSTALL_DIR}' does not exist;" \
       "install llvm-22-dev or set LLVM_INSTALL_DIR explicitly" >&2
  exit 1
fi

mmdv_time build_qt qt

QTPATHS=${QTPATHS:-${MMDV_USRDIR}/bin/qtpaths6}
if [ ! -x "${QTPATHS}" ] ; then
  echo "qtpaths6 not found at ${QTPATHS}; check the Qt build"
  exit 1
fi
export LLVM_INSTALL_DIR QTPATHS PYSIDE_BUILD=1

mmdv_time build_pyside6 pyside6

else

echo "Set \${MMDVBUILD_ALL} or \${MMDVBUILD_QT} to build QT section"

fi
