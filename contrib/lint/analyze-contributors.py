#!/usr/bin/env python3

# Copyright (c) 2026, solvcon team <contact@solvcon.net>

"""
Analyze modmesh contributors and maintain CONTRIBUTORS.md.

A contributor belongs in the list when an email appears in a source-file
copyright header or reaches a commit threshold on ``master`` (default 5).
Aliases that share an author name are merged into one entry (``--no-dedup``
disables it), and the display name prefers the copyright-header spelling.

``--update-file FILE`` rewrites the list in place, keeping the file header,
existing names (unless ``--update-name``), and the email order of existing
entries. Otherwise a keep/drop report is printed.
"""

import re
import sys
import glob
import argparse
import subprocess
from collections import Counter, defaultdict


# Only commits merged here count, so unmerged branch work is excluded.
MASTER_REF = 'master'

# Emails ignored in both scans (placeholders, mailing lists, bots).
SKIP_EMAILS = {
    'contact@solvcon.net',
}

# A copyright line: "(c)", a year span, then "Name <email>".
COPYRIGHT_RE = re.compile(
    r'Copyright\s*\([cC]\)\s*[-0-9, ]*\s+([^<]+?)\s*<([^>]+)>'
)

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


def scan_copyright_headers(files):
    """Return {email: set(names)} parsed from copyright headers."""
    headers = defaultdict(set)
    for fn in files:
        try:
            with open(fn, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
        except OSError as e:
            print(f"Error reading {fn}: {e}", file=sys.stderr)
            continue
        for name, email in COPYRIGHT_RE.findall(text):
            email = email.strip()
            if email in SKIP_EMAILS:
                continue
            headers[email].add(name.strip().rstrip('.,'))
    return headers


def git_commit_counts():
    """Return ({email: count}, {email: set(names)}) for commits on master."""
    proc = subprocess.run(
        ['git', 'log', MASTER_REF, '--format=%aE\t%aN'],
        capture_output=True, text=True, check=True,
    )
    counts = Counter()
    names = defaultdict(set)
    for line in proc.stdout.splitlines():
        if '\t' not in line:
            continue
        email, name = line.split('\t', 1)
        email = email.strip()
        if not email or email in SKIP_EMAILS:
            continue
        counts[email] += 1
        names[email].add(name.strip())
    return counts, names


class UnionFind:
    """Minimal disjoint-set used to merge contributor aliases."""

    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def choose_name(header_names, git_names):
    """Pick a display name, preferring a copyright-header real name."""
    if header_names:
        pool = header_names
    else:
        # Favor git names that look like real names (contain a space).
        spaced = {n for n in git_names if ' ' in n}
        pool = spaced or git_names or {'(unknown)'}
    return sorted(pool, key=lambda s: (-len(s), s))[0]


def make_identity(emails, headers, counts, git_names):
    """Collapse a set of emails into a single contributor row."""
    header_names, names = set(), set()
    commits, in_header = 0, False
    for email in emails:
        header_names |= headers.get(email, set())
        names |= git_names.get(email, set())
        commits += counts.get(email, 0)
        in_header = in_header or email in headers
    # Prefer a header email, else the busiest one.
    rep = sorted(emails,
                 key=lambda e: (e in headers, counts.get(e, 0)),
                 reverse=True)[0]
    return {
        'name': choose_name(header_names, names),
        'email': rep,
        'emails': sorted(emails),
        'commits': commits,
        'in_header': in_header,
    }


def build_identities(headers, counts, git_names, dedup, pinned_groups=()):
    """Build contributor rows, merging aliases when ``dedup``.

    ``pinned_groups`` are email sets kept together (an existing entry's
    addresses) so a known address never starts a second entry.
    """
    emails = set(headers) | set(counts)
    for group in pinned_groups:
        emails |= set(group)

    uf = UnionFind()
    for email in emails:
        uf.find(('e', email))
    # Union emails that share an author name into one contributor.
    if dedup:
        for source in (git_names, headers):
            for email, names in source.items():
                for name in names:
                    if name:
                        uf.union(('e', email), ('n', name))
    # Pin emails that already share an existing entry.
    for group in pinned_groups:
        anchor = None
        for email in group:
            if anchor is None:
                anchor = email
            else:
                uf.union(('e', anchor), ('e', email))

    grouped = defaultdict(set)
    for email in emails:
        grouped[uf.find(('e', email))].add(email)
    return [make_identity(group, headers, counts, git_names)
            for group in grouped.values()]


def analyze(threshold, exclude_dirs, dedup=True, pinned_groups=()):
    """Build one decision row per contributor."""
    files = find_source_files(exclude_dirs)
    headers = scan_copyright_headers(files)
    counts, git_names = git_commit_counts()

    rows = build_identities(headers, counts, git_names, dedup, pinned_groups)
    for r in rows:
        r['keep'] = r['in_header'] or r['commits'] >= threshold
    rows.sort(key=lambda r: (not r['keep'], -r['commits'], r['name'].lower()))
    return rows


def reason(row, threshold):
    """Human-readable explanation of the keep/drop decision."""
    parts = []
    if row['in_header']:
        parts.append('copyright header')
    if row['commits'] >= threshold:
        parts.append(f">={threshold} commits")
    return ' + '.join(parts) if parts else f"only {row['commits']} commit(s)"


def print_report(rows, threshold):
    """Print a full keep/drop report to stdout."""
    kept = [r for r in rows if r['keep']]
    dropped = [r for r in rows if not r['keep']]

    print(f"Analyzed {len(rows)} contributor identities "
          f"(threshold: {threshold} commits).\n")

    print(f"KEEP ({len(kept)}):")
    for r in kept:
        print(f"  {r['name']} <{r['email']}>  "
              f"[{r['commits']} commits, {reason(r, threshold)}]")

    print(f"\nDROP ({len(dropped)}):")
    for r in dropped:
        print(f"  {r['name']} <{r['email']}>  "
              f"[{r['commits']} commits, {reason(r, threshold)}]")


def read_header(path):
    """Return the lines of ``path`` before its first ``- `` bullet."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except OSError:
        return ''
    header = []
    for line in lines:
        if line.startswith('- '):
            break
        header.append(line)
    return ''.join(header)


def parse_existing_entries(path):
    """Return [(name, [emails])] for the ``- name <email>...`` lines."""
    entries = []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except OSError:
        return entries
    for line in lines:
        if not line.startswith('- ') or '<' not in line:
            continue
        body = line[2:].rstrip('\n')
        name = body[:body.index('<')].strip()
        emails = [e.strip() for e in re.findall(r'<([^>]+)>', body)]
        entries.append((name, emails))
    return entries


def display_name(row, existing_names, update_name):
    """Resolve the name to write, honoring kept names unless updating."""
    if not update_name:
        for email in row['emails']:
            if email in existing_names:
                return existing_names[email]
    return row['name']


def entry_emails(row, order_index=None):
    """Return an entry's addresses in display order.

    Existing addresses keep their file order (``order_index``); new ones are
    appended sorted, dropping ``.local`` and non-address tokens. A brand-new
    entry leads with the representative.
    """
    order_index = order_index or {}

    def usable(email):
        return '@' in email and not email.endswith('.local')

    existing = sorted((e for e in row['emails'] if e in order_index),
                      key=lambda e: order_index[e])
    fresh = sorted(e for e in row['emails']
                   if e not in order_index and usable(e))
    if existing:
        return existing + fresh
    rep = row['email']
    rest = [e for e in fresh if e != rep]
    return ([rep] if usable(rep) else []) + rest or [rep]


def print_contributors(rows, stream=sys.stdout, existing_names=None,
                       update_name=True, order_index=None):
    """Write the name-sorted ``- name <email>...`` list to ``stream``.

    Existing names are kept unless ``update_name``; an existing entry's email
    order is preserved via ``order_index``.
    """
    existing_names = existing_names or {}
    entries = [(display_name(r, existing_names, update_name),
                entry_emails(r, order_index))
               for r in rows if r['keep']]
    for name, emails in sorted(entries, key=lambda t: t[0]):
        joined = ' '.join(f"<{e}>" for e in emails)
        print(f"- {name} {joined}", file=stream)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Analyze contributors via copyright headers and git '
        'commit counts to decide who belongs in CONTRIBUTORS.md.'
    )
    parser.add_argument(
        '--threshold', type=int, default=5,
        help='minimum commits to keep a contributor (default: 5)'
    )
    parser.add_argument(
        '--exclude-dirs', default='build,thirdparty,tmp',
        help='comma-separated directories to skip when scanning headers'
    )
    parser.add_argument(
        '--update-file', metavar='FILE',
        help='write CONTRIBUTORS.md bullet lines to FILE instead of printing '
        'the full report'
    )
    parser.add_argument(
        '--update-name', action='store_true',
        help='with --update-file, also refresh contributor names from '
        'copyright headers; otherwise existing names are kept'
    )
    parser.add_argument(
        '--no-dedup', dest='dedup', action='store_false',
        help='report every email separately instead of merging aliases '
        'into one contributor (dedup is on by default)'
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    exclude_dirs = [d for d in args.exclude_dirs.split(',') if d]
    if args.update_name and not args.update_file:
        print("warning: --update-name has no effect without --update-file",
              file=sys.stderr)
    if args.update_file:
        entries = parse_existing_entries(args.update_file)
        existing_names = {e: name for name, emails in entries for e in emails}
        order_index = {e: (i, j)
                       for i, (_, emails) in enumerate(entries)
                       for j, e in enumerate(emails)}
        pinned = [emails for _, emails in entries]
        header = read_header(args.update_file)
        rows = analyze(args.threshold, exclude_dirs, args.dedup, pinned)
        with open(args.update_file, 'w', encoding='utf-8') as f:
            f.write(header)
            print_contributors(rows, f, existing_names, args.update_name,
                               order_index)
    else:
        rows = analyze(args.threshold, exclude_dirs, args.dedup)
        print_report(rows, args.threshold)
    return 0


if __name__ == '__main__':
    sys.exit(main())

# vim: set ff=unix fenc=utf8 et tw=79 sw=4 ts=4 sts=4:
