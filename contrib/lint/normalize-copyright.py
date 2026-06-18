#!/usr/bin/env python3

# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""
Normalize the copyright header in modmesh source files.

With ``CONTRIBUTORS.md`` holding the contributor list, each per-file
header is reduced to the copyright line plus a pointer to the full
license text in ``COPYING``::

    Copyright (c) <year>, solvcon team <contact@solvcon.net>
    BSD 3-Clause License, see COPYING

The year is kept from the existing header; only the holder is rewritten
to the team. Both comment styles in the tree are handled, the C block
``/* ... */`` and the ``#`` line comment, and every header variant (full
BSD text, an existing ``see COPYING`` reference, or a bare copyright
line) is rewritten while its comment delimiters are kept.

Pass file paths to normalize only those, or none to scan every source
file. ``--check`` reports what would change without writing, so it can
run as a lint step.
"""

import re
import sys
import glob
import argparse
import functools
from typing import NamedTuple


# The fixed content of a normalized header; the year comes from the file.
HOLDER = 'solvcon team <contact@solvcon.net>'
LICENSE_LINE = 'BSD 3-Clause License, see COPYING'
DEFAULT_YEAR = '2026'

# Year span between "(c)" and the first comma. A holder name may contain
# a comma, so the first comma is what ends the year.
YEAR_RE = re.compile(r'Copyright[ \t]*\([cC]\)[ \t]*([0-9][-0-9 ]*?)[ \t]*,')

# A C block comment containing a copyright line, matched without crossing
# the closing "*/" so an earlier unrelated block is skipped.
CBLOCK_RE = re.compile(
    r'(?P<indent>[ \t]*)/\*(?:(?!\*/).)*?'
    r'Copyright[ \t]*\([cC]\)(?:(?!\*/).)*?\*/',
    re.DOTALL,
)

# A "#" copyright line and the consecutive "#" comment lines below it.
HBLOCK_RE = re.compile(
    r'(?P<indent>[ \t]*)#[ \t]*Copyright[ \t]*\([cC]\).*'
    r'(?:\n[ \t]*#.*)*',
)


class CommentStyle(NamedTuple):
    """A comment syntax: how to find a header and how to re-render it."""
    finder: re.Pattern  # matches the whole existing header block
    opener: str         # delimiter line above the body, or "" if none
    prefix: str         # leads every body line
    closer: str         # delimiter line below the body, or "" if none


# A source file uses exactly one of these styles for its header.
COMMENT_STYLES = [
    CommentStyle(CBLOCK_RE, '/*', ' * ', ' */'),  # C and C++
    CommentStyle(HBLOCK_RE, '', '# ', ''),         # Python, shell, CMake
]


def _rewrite(style, match):
    """Render the replacement for one matched header in ``style``.

    The year is read back from the matched text so the file keeps its
    own; only the holder and the license pointer are fixed.
    """
    found = YEAR_RE.search(match[0])
    year = found.group(1) if found else DEFAULT_YEAR
    body = [f"Copyright (c) {year}, {HOLDER}", LICENSE_LINE]
    indent = match.group('indent')
    lines = [indent + style.opener] if style.opener else []
    lines += [indent + style.prefix + content for content in body]
    if style.closer:
        lines.append(indent + style.closer)
    return '\n'.join(lines)


def normalize(text):
    """Return ``text`` with its license header reduced to the short form.

    A file carries one comment style, so rewriting each style's header at
    most once leaves the result stable on a second pass (idempotent).
    """
    for style in COMMENT_STYLES:
        replace = functools.partial(_rewrite, style)
        text = style.finder.sub(replace, text, count=1)
    return text


# Same globs as the other contrib/lint scanners: C++, Python, shell, and
# related source files.
SOURCE_PATTERNS = [
    '**/*.py', '**/*.cpp', '**/*.hpp', '**/*.c', '**/*.h',
    '**/*.cxx', '**/*.hxx', '**/*.sh',
    '**/Makefile', '**/makefile', '**/CMakeLists.txt',
    '.github/workflows/*.yml',
]


def find_source_files(exclude_dirs):
    """Collect source files, skipping the excluded top-level directories."""
    exclude = tuple(d.rstrip('/') + '/' for d in exclude_dirs)
    found = []
    for pattern in SOURCE_PATTERNS:
        for fn in glob.glob(pattern, recursive=True):
            if not fn.startswith(exclude):
                found.append(fn)
    return sorted(set(found))


def process_file(fn, check, verbose):
    """Normalize one file. Return True when it changed or would change."""
    try:
        with open(fn, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
    except OSError as e:
        print(f"Error reading {fn}: {e}", file=sys.stderr)
        return False
    new = normalize(text)
    if new == text:
        if verbose:
            print(f"unchanged {fn}")
        return False
    if not check:
        try:
            with open(fn, 'w', encoding='utf-8') as f:
                f.write(new)
        except OSError as e:
            print(f"Error writing {fn}: {e}", file=sys.stderr)
            return False
    print(f"{'would update' if check else 'updated'} {fn}")
    return True


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Normalize the copyright header in source files.'
    )
    parser.add_argument(
        '--check',
        action='store_true',
        help='report files that would change without writing them'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='also report files that are already up to date'
    )
    parser.add_argument(
        '--exclude-dirs',
        action='store',
        default='build,thirdparty,tmp',
        help='comma-separated directories to skip in the default scan'
    )
    parser.add_argument(
        'files',
        nargs='*',
        help='specific files to normalize (default: all source files)'
    )
    return parser.parse_args()


def main():
    """Normalize the given files, or all source files when none given."""
    args = parse_arguments()
    if args.files:
        files = args.files
    else:
        files = find_source_files(args.exclude_dirs.split(','))
    changed = 0
    for fn in files:
        if process_file(fn, args.check, args.verbose):
            changed += 1
    verb = 'would update' if args.check else 'updated'
    print(f"\n{verb} {changed} of {len(files)} files.")
    # In check mode a pending change is a lint failure.
    return 1 if (args.check and changed) else 0


if __name__ == '__main__':
    sys.exit(main())

# vim: set ff=unix fenc=utf8 et tw=79 sw=4 ts=4 sts=4:
