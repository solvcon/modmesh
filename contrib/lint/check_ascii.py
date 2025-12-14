# Copyright (c) 2025, Chun-Shih Chang <austin2046@gmail.com>
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# - Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
# - Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# - Neither the name of the copyright holder nor the names of its contributors
#   may be used to endorse or promote products derived from this software
#   without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""
ASCII-only character and trailing whitespace checker for modmesh source code files.

This script checks that all source files contain only ASCII characters and no trailing whitespace,
as required by the project coding standards.
"""

import sys
import glob
import argparse


def check_ascii_file(filepath):
    """Check if a single file contains only ASCII characters."""
    try:
        with open(filepath, 'rb') as f:
            content = f.read()
            try:
                content.decode('ascii')
                return True
            except UnicodeDecodeError as e:
                print(f"Non-ASCII character found in {filepath}: {e}")
                return False
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return False


def check_no_trailing_whitespace(filepath):
    """Check if a single file contains no trailing whitespace."""
    try:
        no_tws = True
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            for lineno, line in enumerate(lines, start=1):
                if line.rstrip('\n').rstrip('\r').endswith(' ') or line.rstrip('\n').rstrip('\r').endswith('\t'):
                    print(f"Trailing whitespace found in {filepath} at line {lineno}")
                    no_tws = False
            return no_tws
    except Exception as e:
        print(f"Error checking {filepath}: {e}")
        return False


def find_source_files():
    """Find all source code files in the project."""
    patterns = [
        '**/*.py', '**/*.cpp', '**/*.hpp', '**/*.c', '**/*.h',
        '**/*.cxx', '**/*.hxx', '**/*.sh',
        '**/Makefile', '**/makefile', '**/CMakeLists.txt'
    ]

    all_files = []
    for pattern in patterns:
        files = glob.glob(pattern, recursive=True)
        all_files.extend(files)

    return sorted(set(all_files))


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Check that source files contain only ASCII characters or no trailing whitespace'
    )
    parser.add_argument(
        '--check-tws',
        action='store_true',
        help='Check for trailing whitespace instead of non-ASCII characters'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show all checked files'
    )
    parser.add_argument(
        'files',
        nargs='*',
        help='Specific files to check (default: all source files)'
    )
    return parser.parse_args()


def get_files_to_check(args):
    """Determine which files to check based on arguments."""
    if args.files:
        return args.files
    else:
        return find_source_files()


def check_files(files_to_check, check_tws=False, verbose=False):
    """Check all files for ASCII-only characters or no trailing whitespace."""
    failed_files = []
    checked_count = 0

    if check_tws:
        print(f"Checking {len(files_to_check)} source files for no trailing whitespace...")
    else:
        print(f"Checking {len(files_to_check)} source files for ASCII-only "
            f"characters...")

    for filepath in files_to_check:
        if verbose:
            print(f"Checking: {filepath}")

        if check_tws:
            if not check_no_trailing_whitespace(filepath):
                failed_files.append(filepath)
            checked_count += 1
        else:
            if not check_ascii_file(filepath):
                failed_files.append(filepath)
            checked_count += 1

    return failed_files, checked_count


def report_results(failed_files, checked_count, check_tws=False):
    """Report the check results and return exit code."""
    if check_tws:
        if failed_files:
            print(f"\nFAILED: {len(failed_files)} files contain trailing "
                  f"whitespace:")
            for f in failed_files:
                print(f"  - {f}")
            print(f"\nChecked {checked_count} files total.")
            return 1
        else:
            print(f"SUCCESS: All {checked_count} source files contain no "
                  f"trailing whitespace.")
    else:
        if failed_files:
            print(f"\nFAILED: {len(failed_files)} files contain non-ASCII "
                f"characters:")
            for f in failed_files:
                print(f"  - {f}")
            print(f"\nChecked {checked_count} files total.")
            return 1
        else:
            print(f"SUCCESS: All {checked_count} source files contain only "
                f"ASCII characters.")
    return 0


def main():
    args = parse_arguments()
    files_to_check = get_files_to_check(args)

    if not files_to_check:
        print("No files found to check")
        return 0

    failed_files, checked_count = check_files(files_to_check, args.check_tws, args.verbose)
    return report_results(failed_files, checked_count, args.check_tws)


if __name__ == '__main__':
    sys.exit(main())
