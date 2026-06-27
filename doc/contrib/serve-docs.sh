#!/bin/bash
#
# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING
#
# Serve doc/build/html on a trusted-network address, with teardown tied to
# the process that launched it. Driven by the serve-docs skill; see it for
# the build-serve-teardown flow.
#
# Usage: doc/contrib/serve-docs.sh <bind-ip> [port] [--launcher-pid PID]
#
#   <bind-ip>           Trusted-network address to bind to. Never an
#                       all-interfaces address (0.0.0.0, ::): the server is
#                       an unauthenticated http.server and that would expose
#                       build/html on the local LAN. Derive the address at
#                       run time; do not hardcode it.
#   [port]              Optional. Default is the first free port in 8765-8769.
#   --launcher-pid PID  Optional. Tie teardown to PID instead of the parent
#                       process. Supply this when the parent is short-lived,
#                       for instance an agent that runs each command in a
#                       throwaway shell, so the watchdog tracks the real
#                       session. Defaults to $PPID.

set -u

# True only for a run of decimal digits with at least one non-zero, so empty,
# zero, negative, and non-numeric inputs are all rejected without arithmetic.
is_pos_int() {
    case "$1" in
        ''|*[!0-9]*) return 1 ;;
        *[!0]*) return 0 ;;
        *) return 1 ;;
    esac
}

IP=""
PORT=""
LAUNCHER_PID=""
while [ $# -gt 0 ]; do
    case "$1" in
        --launcher-pid) LAUNCHER_PID="${2:?--launcher-pid needs a PID}"
                        shift 2 ;;
        --launcher-pid=*) LAUNCHER_PID="${1#*=}"; shift ;;
        -*) echo "unknown option: $1" >&2; exit 1 ;;
        *) if [ -z "$IP" ]; then IP="$1"
           elif [ -z "$PORT" ]; then PORT="$1"
           else echo "unexpected argument: $1" >&2; exit 1
           fi
           shift ;;
    esac
done
[ -n "$IP" ] || {
    echo "usage: serve-docs.sh <bind-ip> [port] [--launcher-pid PID]" >&2
    exit 1
}

# Refuse all-interfaces binds; only those expose the server beyond the named
# address. A real IPv6 trusted address still has colons and is allowed.
case "$IP" in
    0.0.0.0|0|::|::0|0:0:0:0:0:0:0:0)
        echo "refusing to bind '$IP' (all interfaces); name a trusted IP" >&2
        exit 1 ;;
esac

LAUNCHER_PID="${LAUNCHER_PID:-$PPID}"
is_pos_int "$LAUNCHER_PID" || {
    echo "launcher PID must be a positive integer" >&2; exit 1
}
if ! kill -0 "$LAUNCHER_PID" 2>/dev/null; then
    echo "launcher PID $LAUNCHER_PID is not alive" >&2
    exit 1
fi

if [ -n "$PORT" ] && { ! is_pos_int "$PORT" || [ "$PORT" -gt 65535 ]; }; then
    echo "port must be an integer in 1-65535" >&2
    exit 1
fi

ROOT=$(git rev-parse --show-toplevel 2>/dev/null) || ROOT=""
[ -n "$ROOT" ] || { echo "not inside a git repository" >&2; exit 1; }
HTML="$ROOT/doc/build/html"
LOG="$ROOT/doc/build/serve-docs.log"
[ -f "$HTML/index.html" ] || {
    echo "no build at $HTML; run 'make html' in doc/ first" >&2
    exit 1
}

# A port is free when nothing answers a connect to it.
port_free() { ! (exec 3<>"/dev/tcp/$IP/$1") 2>/dev/null; }
if [ -n "$PORT" ]; then
    port_free "$PORT" || { echo "port $PORT in use on $IP" >&2; exit 1; }
else
    # A sibling worktree may already hold a port; take the first free one.
    for p in 8765 8766 8767 8768 8769; do
        if port_free "$p"; then PORT=$p; break; fi
    done
    [ -n "$PORT" ] || { echo "no free port in 8765-8769" >&2; exit 1; }
fi

# Serve. Plain nohup (NOT setsid): nohup execs, so $! is the real python PID;
# setsid forks and $! would be the dead parent, breaking the teardown kill.
# --directory pins the served root, so it cannot drift to the current dir.
nohup python3 -m http.server "$PORT" --bind "$IP" --directory "$HTML" \
    > "$LOG" 2>&1 </dev/null &
SERVER_PID=$!
disown

# Watchdog: poll the launcher PID, kill the server when it exits. The PIDs
# are passed as positional args, never interpolated into the code string, so
# a hostile value cannot inject shell commands.
nohup bash -c '
    while kill -0 "$1" 2>/dev/null; do sleep 30; done
    kill "$2" 2>/dev/null
' serve-docs-watchdog "$LAUNCHER_PID" "$SERVER_PID" >/dev/null 2>&1 &
disown

sleep 1
curl -s -o /dev/null -w 'HTTP %{http_code}\n' "http://$IP:$PORT/index.html"
echo "Docs: http://$IP:$PORT/  (use the host DNS name if it has one)"
echo "Server PID $SERVER_PID watches launcher $LAUNCHER_PID."
echo "Log: $LOG"
echo "Stop early with: kill $SERVER_PID"

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
