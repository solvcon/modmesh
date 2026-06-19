#!/usr/bin/env python3

# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""
Rename the C++ project ``modmesh`` to ``solvcon``.

This renames the C++ layer as one mechanical step toward renaming the whole
project from modmesh to solvcon. The repository
(https://github.com/solvcon/modmesh) is renamed by hand, and the Python layer
was renamed by a separate script (rename-python-modmesh-solvcon.py); this one
touches only the C++ project identifiers:

- the C++ namespace ``modmesh`` (``namespace modmesh``, ``modmesh::``);
- the ``cpp/modmesh`` source tree, including the ``#include <modmesh/...>``
  paths and the umbrella header ``modmesh.hpp``;
- the ``MODMESH_*`` build macros and CMake variables;
- the CMake target ``modmesh_primary`` and the placeholder gtest suite
  ``nopython_modmesh``.

The word "modmesh" means several things, so the rewrite is deliberately
conservative. These meanings are not the C++ project and are kept as is:

- the repository slug ``solvcon/modmesh`` in URLs, until the repository is
  renamed by hand;
- the Python-facing identifiers that merely embed the word, such as
  ``setup_modmesh_path`` and ``is_modmesh_meta_path_finder_registered``, which
  the Python rename left untouched for the same reason; the underscore on the
  word boundary keeps the token rule from hitting them.

The substitutions are length-preserving (both "modmesh"/"solvcon" and
"MODMESH"/"SOLVCON" are seven characters), so formatting does not change, and
the tool is idempotent.

Run from anywhere inside the working tree::

    contrib/lint/rename-cpp-modmesh-solvcon.py            # apply
    contrib/lint/rename-cpp-modmesh-solvcon.py --dry-run  # preview
"""

import os
import re
import sys
import argparse
import subprocess


# Rename "modmesh" only as a whole identifier token, so an embedded use such as
# ``setup_modmesh_path`` is not hit by accident. A path separator, a dot, or an
# angle bracket counts as a boundary, so ``<modmesh/...>``, ``cpp/modmesh``,
# and ``modmesh.hpp`` are all matched.
NAMESPACE_RE = re.compile(r'(?<![A-Za-z0-9_])modmesh(?![A-Za-z0-9_])')

# Every uppercase "MODMESH" is a C++ build macro or CMake variable, so the
# substring is replaced wholesale; no kept meaning spells the word in capitals.
MACRO = ('MODMESH', 'SOLVCON')

# Literals whose match must survive the rename. Each is masked with a unique
# sentinel before the rules run and restored afterward. The slug is masked so
# the token rule does not turn ``solvcon/modmesh`` into ``solvcon/solvcon``.
KEEP_LITERALS = ('solvcon/modmesh',)

# C++ identifiers the token rule cannot see because a word character sits on
# the boundary; they are replaced verbatim.
COMPOUND_RENAMES = (
    ('modmesh_primary', 'solvcon_primary'),  # The primary CMake target.
    ('modmesh_pymod', 'solvcon_pymod'),  # The pybind11 module CMake project.
    ('nopython_modmesh', 'nopython_solvcon'),  # Placeholder gtest suite name.
)

# Files renamed with ``git mv`` once their contents are rewritten. The umbrella
# header and the placeholder gtest carry the project name in their file name.
FILE_RENAMES = (
    ('cpp/modmesh/modmesh.hpp', 'cpp/modmesh/solvcon.hpp'),
    ('gtests/test_nopython_modmesh.cpp', 'gtests/test_nopython_solvcon.cpp'),
)

# The source tree, moved with ``git mv`` after the files inside are renamed.
DIRECTORY_RENAMES = (('cpp/modmesh', 'cpp/solvcon'),)

# Path prefixes excluded from the content rewrite. Third-party sources are
# vendored and never name the C++ project.
SKIP_PREFIXES = ('thirdparty/',)

# Files excluded from the content rewrite. The Python rename script is a frozen
# record of that step; rewriting the C++ literals it documents as kept would
# misstate what it did.
SKIP_RELPATHS = ('contrib/lint/rename-python-modmesh-solvcon.py',)


def rewrite(text):
    """
    Return ``text`` with the C++ project namespace, paths, and macros renamed.

    The kept meanings of "modmesh" (see the module docstring) are first masked
    with placeholder sentinels, the rename rules run on what remains, and the
    sentinels are then restored. A NUL byte cannot appear in a text file (such
    a file is detected as binary and skipped), so it is a safe sentinel mark.
    """
    masks = []

    def mask(matched):
        sentinel = '\x00{}\x00'.format(len(masks))
        masks.append((sentinel, matched))
        return sentinel

    for literal in KEEP_LITERALS:
        if literal in text:
            text = text.replace(literal, mask(literal))

    for src, dst in COMPOUND_RENAMES:
        text = text.replace(src, dst)
    text = text.replace(*MACRO)
    text = NAMESPACE_RE.sub('solvcon', text)

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
        updated = rewrite(text)
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
        updated = rewrite(target)
        if updated == target:
            continue
        changed += 1
        print('relink {} -> {}'.format(relpath, updated))
        if not dry_run:
            os.remove(path)
            os.symlink(updated, path)
    return changed


def rename_paths(root, renames, dry_run):
    """Move the listed tracked paths with ``git mv``."""
    for src, dst in renames:
        if not os.path.exists(os.path.join(root, src)):
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
        description='Rename the C++ project modmesh to solvcon.')
    parser.add_argument(
        '--dry-run', action='store_true',
        help='report the changes without modifying the tree')
    args = parser.parse_args()

    root = find_root()
    # Exclude this script so it does not rewrite its own rename rules. Deriving
    # the path from __file__ keeps the exclusion correct if the file is moved.
    self_relpath = os.path.relpath(os.path.realpath(__file__), root)
    skip_relpaths = SKIP_RELPATHS + (self_relpath,)

    changed = rewrite_contents(root, skip_relpaths, args.dry_run)
    changed += rewrite_symlinks(root, args.dry_run)
    rename_paths(root, FILE_RENAMES, args.dry_run)
    rename_paths(root, DIRECTORY_RENAMES, args.dry_run)
    print('{} {} file(s)'.format(
        'would change' if args.dry_run else 'changed', changed))
    return 0


if __name__ == '__main__':
    sys.exit(main())

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
