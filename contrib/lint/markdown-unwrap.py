#!/usr/bin/env python3
# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""
Unwrap hard-wrapped Markdown paragraphs and list items.

The project wraps Markdown prose at 79 columns.  This helper does the
inverse: it joins each hard-wrapped paragraph onto a single continuous
line, and likewise joins each list item (its marker plus the wrapped
continuation lines) onto one line.  The result renders to the same HTML
and is convenient to edit in a tool that re-wraps on demand.

Blocks that must not be reflowed are passed through verbatim: fenced code
blocks (```, ~~~, or :::), HTML comments, ATX headings, thematic breaks,
block quotes, and pipe-table rows.  Blank lines, which separate blocks,
are preserved.

Usage:
    markdown-unwrap.py FILE...      # write the result to stdout
    markdown-unwrap.py -i FILE...   # rewrite the files in place
    markdown-unwrap.py < in > out   # filter stdin to stdout
"""

import re
import sys
import argparse

# A list item marker: a bullet (-, *, +) or an ordered marker (1. or 1)).
LIST_RE = re.compile(r'^(\s*)([-*+]|\d+[.)])(\s+)(\S.*)$')
# A fence opener: three or more backticks, tildes, or colons.
FENCE_RE = re.compile(r'^(\s*)(`{3,}|~{3,}|:{3,})(.*)$')
HEADING_RE = re.compile(r'^\s*#{1,6}(\s|$)')
THEMATIC_RE = re.compile(r'^\s*([-*_])(\s*\1){2,}\s*$')
TABLE_RE = re.compile(r'^\s*\|')
QUOTE_RE = re.compile(r'^\s*>')
BLANK_RE = re.compile(r'^\s*$')


def is_blank(line):
    return BLANK_RE.match(line) is not None


def is_special(line):
    """A line that begins a block which must not be joined into prose."""
    return (
        is_blank(line)
        or FENCE_RE.match(line) is not None
        or HEADING_RE.match(line) is not None
        or THEMATIC_RE.match(line) is not None
        or TABLE_RE.match(line) is not None
        or QUOTE_RE.match(line) is not None
        or LIST_RE.match(line) is not None
        or line.lstrip().startswith('<!--')
    )


def fence_closer(opener):
    """Return a regex matching the closing fence of the given opener."""
    marker = FENCE_RE.match(opener).group(2)
    char, length = marker[0], len(marker)
    return re.compile(r'^\s*' + re.escape(char) + '{' + str(length) +
                      r',}\s*$')


def join_block(block):
    """Collapse wrapped lines into one, keeping the first line's indent."""
    first = block[0].rstrip()
    rest = [line.strip() for line in block[1:]]
    rest = [line for line in rest if line]
    return first + (' ' + ' '.join(rest) if rest else '')


def convert(text):
    """Join wrapped paragraphs and list items in a Markdown string."""
    lines = text.split('\n')
    trailing_newline = bool(lines) and lines[-1] == ''
    if trailing_newline:
        lines = lines[:-1]
    out = []
    i, n = 0, len(lines)
    while i < n:
        line = lines[i]
        if is_blank(line):
            out.append('')
            i += 1
        elif FENCE_RE.match(line):
            closer = fence_closer(line)
            out.append(line.rstrip())
            i += 1
            while i < n and not closer.match(lines[i]):
                out.append(lines[i].rstrip('\r'))
                i += 1
            if i < n:
                out.append(lines[i].rstrip())
                i += 1
        elif line.lstrip().startswith('<!--'):
            out.append(line.rstrip())
            i += 1
            while i < n and '-->' not in lines[i - 1]:
                out.append(lines[i].rstrip())
                i += 1
        elif (HEADING_RE.match(line) or THEMATIC_RE.match(line)):
            out.append(line.rstrip())
            i += 1
        elif TABLE_RE.match(line):
            while i < n and TABLE_RE.match(lines[i]):
                out.append(lines[i].rstrip())
                i += 1
        elif QUOTE_RE.match(line):
            while i < n and QUOTE_RE.match(lines[i]):
                out.append(lines[i].rstrip())
                i += 1
        else:
            # A list item or a plain paragraph: collect the marker (if
            # any) plus the wrapped continuation lines that follow it.
            block = [line]
            i += 1
            while i < n and not is_special(lines[i]):
                block.append(lines[i])
                i += 1
            out.append(join_block(block))
    result = '\n'.join(out)
    return result + '\n' if trailing_newline else result


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Join hard-wrapped Markdown paragraphs and list '
        'items onto continuous single lines.'
    )
    parser.add_argument(
        '-i', '--in-place',
        action='store_true',
        help='rewrite each file in place instead of writing to stdout'
    )
    parser.add_argument(
        'files',
        nargs='*',
        help='Markdown files to convert (default: read from stdin)'
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    if not args.files:
        if args.in_place:
            print('error: --in-place requires file arguments',
                  file=sys.stderr)
            return 2
        sys.stdout.write(convert(sys.stdin.read()))
        return 0
    for path in args.files:
        with open(path, 'r', encoding='utf-8') as fobj:
            converted = convert(fobj.read())
        if args.in_place:
            with open(path, 'w', encoding='utf-8') as fobj:
                fobj.write(converted)
        else:
            sys.stdout.write(converted)
    return 0


if __name__ == '__main__':
    sys.exit(main())

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
