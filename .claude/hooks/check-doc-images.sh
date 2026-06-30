#!/bin/bash
# .claude/hooks/check-doc-images.sh
#
# PreToolUse hook for Bash (Claude Code).  Blocks a `git commit` that
# would check in a raster or pre-rendered image blob under doc/.
# Documentation schematics must be authored as .tex PSTricks sources and
# rendered by pstake at build time (see CLAUDE.md / STYLE.md), so an
# image blob in the doc tree means a generated artifact is being
# committed in place of its source.  Exit 2 with the offending paths on
# stderr to block the command.

input=$(cat)

if command -v jq >/dev/null 2>&1; then
    cmd=$(printf '%s' "$input" | jq -r '.tool_input.command // empty')
else
    cmd=$(printf '%s' "$input" \
        | sed -n 's/.*"command"[[:space:]]*:[[:space:]]*"\(.*\)"[^"]*}.*/\1/p' \
        | head -1)
fi

[ -z "$cmd" ] && exit 0

# Only gate the command that checks work in.
case "$cmd" in
    *"git commit"*) ;;
    *) exit 0 ;;
esac

# Image blob extensions that must not live in the documentation source;
# their schematics belong in .tex form rendered by pstake.
ext_re='\.(png|jpg|jpeg|gif|bmp|tif|tiff|webp|ico|svg|eps|pdf)$'

staged=$(git diff --cached --name-only 2>/dev/null \
    | grep -iE '^doc/' \
    | grep -iE "$ext_re")

[ -z "$staged" ] && exit 0

{
    echo "Hook violation: image blob staged for commit under doc/."
    printf '%s\n' "$staged" | sed 's/^/  /'
    echo "Documentation schematics must be .tex PSTricks rendered by pstake,"
    echo "not committed image blobs.  Unstage these (\`git restore --staged\`)"
    echo "and add the .tex source instead (see CLAUDE.md, \"Code Style\")."
} >&2
exit 2

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
