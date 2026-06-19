#!/usr/bin/env python3

# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""
Rename the top-level Python package ``modmesh`` to ``solvcon``.

This renames the Python layer as one mechanical step toward renaming the whole
project from modmesh to solvcon. The repository
(https://github.com/solvcon/modmesh) is renamed by hand, and the C++ layer is
renamed by a separate script; this one touches only:

- the Python package directory ``modmesh`` and every reference to it, which
  become ``solvcon``;
- the conventional import alias ``mm``, which becomes ``sc`` in Python sources;
- the compiled extension module ``_modmesh``, which becomes ``_solvcon`` (the
  pybind11 module, its CMake target, the build directory, and its importers).

The word "modmesh" means several things, so the rewrite is deliberately
conservative. These meanings are not the Python package and are kept as is:

- the C++ namespace ``modmesh`` (``namespace modmesh``, ``modmesh::``) and the
  ``cpp/modmesh`` source tree, including ``#include <modmesh/...>`` paths;
- the ``MODMESH_*`` build macros and identifiers that merely embed the word,
  such as ``setup_modmesh_path``;
- the repository slug ``solvcon/modmesh`` in URLs, until the repository is
  renamed by hand.

Run from anywhere inside the working tree::

    contrib/lint/rename-python-modmesh-solvcon.py            # apply
    contrib/lint/rename-python-modmesh-solvcon.py --dry-run  # preview
"""

import os
import re
import sys
import argparse
import subprocess


# Rename "modmesh" only as a whole identifier token, so that an embedded use
# such as ``_modmesh`` or ``setup_modmesh_path`` is not hit by accident. The
# ``mm`` alias is likewise rewritten only where it stands alone.
PACKAGE_RE = re.compile(r'(?<![A-Za-z0-9_])modmesh(?![A-Za-z0-9_])')
EXTENSION_RE = re.compile(r'(?<![A-Za-z0-9_])_modmesh(?![A-Za-z0-9_])')
ALIAS_RE = re.compile(r'(?<![A-Za-z0-9_])mm(?![A-Za-z0-9_])')

# In C++ the bare word is almost always the kept namespace, so rename the
# package there only inside a quoted Python-module string such as
# ``"modmesh.system"`` or ``'modmesh'``; the back-reference keeps the quote.
CPP_PACKAGE_RE = re.compile(r'''(["'])modmesh((?:\.[A-Za-z_]\w*)*)\1''')

# Patterns whose match must survive the rename. Each is masked with a unique
# sentinel before the rules run and restored afterward. The include directive
# is masked as a whole so the umbrella header ``<modmesh/modmesh.hpp>`` keeps
# both of its tokens, not only the first.
KEEP_PATTERNS = (re.compile(r'<modmesh/[^>\n]*>'),)
KEEP_LITERALS = (
    'solvcon/modmesh',  # Repository slug in a URL.
    'cpp/modmesh',  # C++ source tree path.
    'MODMESH_INCLUDE_DIR}/modmesh',  # Installed C++ header directory.
    'modmesh.hpp',  # C++ umbrella header file name.
    'namespace modmesh',  # C++ namespace declaration.
    'modmesh::',  # C++ namespace qualifier.
)

# Identifiers that the token rules above cannot see because a word character
# sits on the boundary; they are replaced verbatim.
COMPOUND_RENAMES = (
    ('pymod_modmesh', 'pymod_solvcon'),  # Extension build directory.
    ('_modmesh_py', '_solvcon_py'),  # Custom target that copies the .so.
)

# Directories moved with ``git mv`` once their contents are rewritten.
DIRECTORY_RENAMES = (
    ('modmesh', 'solvcon'),
    ('cpp/binary/pymod_modmesh', 'cpp/binary/pymod_solvcon'),
)

# Path prefixes excluded from the content rewrite. The standalone buffer build
# copies the ``cpp/modmesh`` tree into a directory that must keep the
# ``modmesh`` name for its ``#include <modmesh/...>`` directives to resolve,
# and it never names the Python package.
SKIP_PREFIXES = ('thirdparty/', 'contrib/standalone_buffer/')

CPP_SUFFIXES = ('.cpp', '.hpp', '.cc', '.cxx', '.c', '.h', '.hh', '.hxx')


def rewrite(text, relpath):
    """
    Return ``text`` with the Python package, alias, and extension renamed.

    The kept meanings of "modmesh" (see the module docstring) are first masked
    with placeholder sentinels, the rename rules run on what remains, and the
    sentinels are then restored. A NUL byte cannot appear in a text file (such
    a file is detected as binary and skipped), so it is a safe sentinel mark.

    :param relpath: the file path, used only to select the rule set by file
        type, because C++, Python, and other files differ in which form of the
        package name they carry.
    """
    masks = []

    def mask(matched):
        sentinel = '\x00{}\x00'.format(len(masks))
        masks.append((sentinel, matched))
        return sentinel

    for pattern in KEEP_PATTERNS:
        text = pattern.sub(lambda match: mask(match.group(0)), text)
    for literal in KEEP_LITERALS:
        if literal in text:
            text = text.replace(literal, mask(literal))

    for src, dst in COMPOUND_RENAMES:
        text = text.replace(src, dst)
    text = EXTENSION_RE.sub('_solvcon', text)
    if relpath.endswith(CPP_SUFFIXES):
        text = CPP_PACKAGE_RE.sub(r'\g<1>solvcon\g<2>\g<1>', text)
    else:
        text = PACKAGE_RE.sub('solvcon', text)
        if relpath.endswith('.py'):
            text = ALIAS_RE.sub('sc', text)

    for sentinel, literal in masks:
        text = text.replace(sentinel, literal)
    return text


def tracked_files(root):
    """Return the git-tracked files under ``root`` as repo-relative paths."""
    completed = subprocess.run(
        ['git', 'ls-files'], cwd=root, check=True,
        capture_output=True, text=True)
    return [line for line in completed.stdout.splitlines() if line]


def read_text(path):
    """Return the file content as text, or ``None`` if the file is binary."""
    with open(path, 'rb') as stream:
        raw = stream.read()
    if b'\x00' in raw:
        return None
    try:
        return raw.decode('utf-8')
    except UnicodeDecodeError:
        return None


def rewrite_contents(root, skip_relpaths, dry_run):
    """Rewrite the eligible tracked text files; return the count."""
    changed = 0
    for relpath in tracked_files(root):
        if relpath in skip_relpaths or relpath.startswith(SKIP_PREFIXES):
            continue
        path = os.path.join(root, relpath)
        # A symlink is handled through its target in rewrite_symlinks; opening
        # it here would edit the target twice or fail on a directory link.
        if os.path.islink(path) or not os.path.isfile(path):
            continue
        text = read_text(path)
        if text is None:
            continue
        updated = rewrite(text, relpath)
        if updated == text:
            continue
        changed += 1
        print('rewrite {}'.format(relpath))
        if not dry_run:
            with open(path, 'w', encoding='utf-8') as stream:
                stream.write(updated)
    return changed


def rewrite_symlinks(root, dry_run):
    """Repoint tracked symlinks that move with the rename; return the count."""
    changed = 0
    for relpath in tracked_files(root):
        path = os.path.join(root, relpath)
        if not os.path.islink(path):
            continue
        target = os.readlink(path)
        updated = rewrite(target, relpath)
        if updated == target:
            continue
        changed += 1
        print('relink {} -> {}'.format(relpath, updated))
        if not dry_run:
            os.remove(path)
            os.symlink(updated, path)
    return changed


def rename_directories(root, dry_run):
    """Move the package and extension directories with ``git mv``."""
    for src, dst in DIRECTORY_RENAMES:
        if not os.path.isdir(os.path.join(root, src)):
            continue
        print('git mv {} {}'.format(src, dst))
        if not dry_run:
            subprocess.run(['git', 'mv', src, dst], cwd=root, check=True)


def find_root():
    """Return the absolute path of the git working-tree root."""
    completed = subprocess.run(
        ['git', 'rev-parse', '--show-toplevel'], check=True,
        capture_output=True, text=True)
    return completed.stdout.strip()


def main():
    parser = argparse.ArgumentParser(
        description='Rename the Python package modmesh to solvcon.')
    parser.add_argument(
        '--dry-run', action='store_true',
        help='report the changes without modifying the tree')
    args = parser.parse_args()

    root = find_root()
    # Exclude this script so it does not rewrite its own rename rules. Deriving
    # the path from __file__ keeps the exclusion correct if the file is moved.
    self_relpath = os.path.relpath(os.path.realpath(__file__), root)

    changed = rewrite_contents(root, (self_relpath,), args.dry_run)
    changed += rewrite_symlinks(root, args.dry_run)
    rename_directories(root, args.dry_run)
    print('{} {} file(s)'.format(
        'would change' if args.dry_run else 'changed', changed))
    return 0


if __name__ == '__main__':
    sys.exit(main())

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
