#!/usr/bin/env bash
#
# Automate setup of a solvcon development environment on a fresh macOS VM.
#
# Usage:
#   bash mac26_vmsetup.sh [step]
#
# Without arguments, runs all steps in order. Read the code to learn the steps
# available.
#
# Convenient tools to install manually before running the script:
# * https://macvim.org/
# * https://iterm2.com/
#
# Before or after running the script, create the ssh private key:
# ```
# ssh-keygen
# ```
#
# When running the script, the two steps, xcode-select install dialog and chsh
# password, are interactive. They pop up windows and prompt for password.
#
# After all packages and tools are installed, agent tools will need
# authentication before use:
# * `gh auth login`
# * `claude`
#
# In a VM, automatic detection of available cores may not be accurate. Use a
# setup.mk file in the solvcon repository root with the explicit process
# number:
# ```
# MAKE_PARALLEL ?= -j4
# ```
#
# Also make a symbolic link for Claude Code: `ln -s contrib/prompt/CLAUDE.md`
#
# Additional information.  The following commands speed up key repeat:
# ```
# defaults write -g InitialKeyRepeat -float 10.0 # normal minimum is 15 (225 ms)
# defaults write -g KeyRepeat -float 1.0 # normal minimum is 2 (30 ms)
# ```

set -euo pipefail

MWORK_DIR="${HOME}/mwork"

log() { printf '\n==> %s\n' "$*"; }
warn() { printf '\n!!  %s\n' "$*" >&2; }

require_macos() {
    if [[ "$(uname -s)" != "Darwin" ]]; then
        warn "This script is intended for macOS only."
        exit 1
    fi
}

step_shell() {
    log "Setting login shell to /bin/bash"
    if [[ "${SHELL:-}" == "/bin/bash" ]]; then
        echo "Login shell is already /bin/bash"
    else
        chsh -s /bin/bash
    fi
}

step_workspace() {
    log "Making the solvcon working directory"
    mkdir -p "${MWORK_DIR}"
    mkdir -p "${HOME}/tmp"
}

step_cli_tools() {
    log "Installing Apple Command Line Developer Tools"
    if xcode-select -p >/dev/null 2>&1; then
        echo "Command Line Tools already installed at: $(xcode-select -p)"
    else
        xcode-select --install || true
        echo "A GUI dialog should appear. Complete the installation, then re-run this script."
        echo "Waiting for installation to finish..."
        local waited=0
        until xcode-select -p >/dev/null 2>&1; do
            sleep 10
            waited=$((waited + 10))
            if (( waited >= 1800 )); then
                warn "Timed out waiting for Command Line Tools install."
                exit 1
            fi
        done
    fi
}

step_homebrew() {
    log "Installing Homebrew"
    if command -v brew >/dev/null 2>&1; then
        echo "brew already installed at: $(command -v brew)"
    else
        /bin/bash -c \
            "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    fi

    # Make brew available in this shell (Apple Silicon vs Intel paths).
    if [[ -x /opt/homebrew/bin/brew ]]; then
        eval "$(/opt/homebrew/bin/brew shellenv)"
    elif [[ -x /usr/local/bin/brew ]]; then
        eval "$(/usr/local/bin/brew shellenv)"
    fi
    command -v brew >/dev/null || { warn "brew not on PATH after install"; exit 1; }
}

step_dependency() {
    log "Installing solvcon build/runtime dependencies via Homebrew"
    brew install \
        tree gh cmake clang-format qt pyside \
        python numpy python-matplotlib pybind11 \
        pytest flake8 black

    log "Installing extra Python packages"
    # We do not use venv, so just install without respecting PEP-668. It's OK
    # for a VM for automation, but may not be a good idea for a setup for
    # everyday work.
    pip3 install jsonschema pyyaml --break-system-packages
}

step_claude() {
    log "Installing Claude CLI"
    if command -v claude >/dev/null 2>&1; then
        echo "claude already installed at: $(command -v claude)"
    else
        curl -fsSL https://claude.ai/install.sh | bash
    fi
}

run_all() {
    step_shell
    step_workspace
    step_cli_tools
    step_homebrew
    step_dependency
    step_claude
    log "All steps complete."
}

main() {
    require_macos
    local target="${1:-all}"
    case "${target}" in
        all) run_all ;;
        shell) step_shell ;;
        workspace) step_workspace ;;
        cli-tools) step_cli_tools ;;
        homebrew) step_homebrew ;;
        dependency) step_dependency ;;
        claude) step_claude ;;
        *)
            warn "Unknown step: ${target}"
            echo "Valid: all, shell, workspace, cli-tools, homebrew, dependency, claude"
            exit 2
            ;;
    esac
}

main "$@"

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
