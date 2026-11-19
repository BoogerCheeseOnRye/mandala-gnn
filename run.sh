#!/usr/bin/env bash
# One tap: start the mandala-gnn demo, open it, and keep it alive.
#
#   ./run.sh            start if needed, open the browser, then stay in the
#                       foreground for as long as the server should live
#   ./run.sh --detach   start and return (safe only inside a long-lived
#                       terminal; the launcher kills a command's process tree
#                       when the command returns)
#   ./run.sh stop       stop the server
#   ./run.sh status     say whether it is running
#   OPEN=0 ./run.sh     start without opening a browser
#
# Why foreground: the launcher runs a .desktop Exec under `proot
# --kill-on-exit`; when that command returns, every process it started is
# killed. A backgrounded server therefore dies the moment this script exits.
# Waiting here ties the server's life to the launch, and closing the launcher
# (or running `run.sh stop`) ends it cleanly.
#
# The model itself stays cold until the page's launch button is pressed;
# /api/state answers in milliseconds either way.
set -euo pipefail

cd "$(dirname "$(readlink -f "$0")")"
PORT="${PORT:-8000}"
HOST="${HOST:-127.0.0.1}"
URL="http://localhost:${PORT}"
LOG="${TMPDIR:-/tmp}/mandala-gnn-serve.log"
PIDFILE="${TMPDIR:-/tmp}/mandala-gnn-serve.pid"
PY=".venv/bin/python"

up() { curl -sf -m 1 "$URL/api/state" >/dev/null 2>&1; }

wait_up() {
  for _ in $(seq 1 160); do
    up && return 0
    sleep 0.25
  done
  return 1
}

pid_of() {
  if [ -f "$PIDFILE" ]; then
    local pid
    pid="$(cat "$PIDFILE" 2>/dev/null || true)"
    if [ -n "${pid:-}" ] && kill -0 "$pid" 2>/dev/null; then
      local cmd
      cmd="$(tr '\0' ' ' <"/proc/${pid}/cmdline" 2>/dev/null || true)"
      case "$cmd" in
        *mandalanet.serve*)
          printf '%s\n' "$pid"
          return 0
          ;;
      esac
    fi
  fi
  return 1
}

open_page() {
  if [ "${OPEN:-1}" = "1" ]; then
    echo "mandala-gnn: opening ${URL}" >>"$LOG"
    if ! xdg-open "$URL" >/dev/null 2>&1; then
      echo "mandala-gnn: could not open a browser; open ${URL}" >&2
      # The browser bridge can be down after a launcher restart. A
      # notification at least leaves the URL on the phone.
      omalauncher notify "Mandala-GNN" "Open ${URL}" >/dev/null 2>&1 || true
    fi
  fi
}

case "${1:-}" in
  stop)
    pid="$(pid_of || true)"
    if [ -n "${pid:-}" ]; then
      kill "$pid" 2>/dev/null || true
      for _ in $(seq 1 20); do kill -0 "$pid" 2>/dev/null || break; sleep 0.1; done
      kill -9 "$pid" 2>/dev/null || true
    else
      pkill -f "[p]ython -m mandalanet.serve --port ${PORT}" 2>/dev/null || true
    fi
    rm -f "$PIDFILE"
    echo "mandala-gnn: stopped"
    exit 0
    ;;
  status)
    if up; then
      pid="$(pid_of || true)"
      echo "mandala-gnn: running at ${URL}${pid:+ (pid ${pid})}"
    else
      echo "mandala-gnn: not running"
    fi
    exit 0
    ;;
  --detach|detach)
    DETACH=1
    ;;
  "")
    DETACH=0
    ;;
  *)
    echo "usage: $0 [--detach|stop|status]" >&2
    exit 2
    ;;
esac

if up; then
  echo "mandala-gnn: already running at ${URL}"
  open_page
  exit 0
fi

if [ ! -x "$PY" ]; then
  echo "mandala-gnn: ${PY} missing; run: python -m venv .venv && .venv/bin/pip install -r requirements.txt" >&2
  exit 1
fi

echo "mandala-gnn: starting on ${URL}" >>"$LOG"
"$PY" -m mandalanet.serve --port "$PORT" --host "$HOST" >>"$LOG" 2>&1 &
SERVER=$!
printf '%s\n' "$SERVER" >"$PIDFILE"

cleanup() {
  trap - INT TERM HUP EXIT
  if kill -0 "$SERVER" 2>/dev/null; then
    kill "$SERVER" 2>/dev/null || true
    wait "$SERVER" 2>/dev/null || true
  fi
  [ "$(cat "$PIDFILE" 2>/dev/null || true)" = "$SERVER" ] && rm -f "$PIDFILE"
  return 0
}
trap cleanup INT TERM HUP EXIT

if ! wait_up; then
  echo "mandala-gnn: server did not start; see ${LOG}" >&2
  tail -n 8 "$LOG" >&2 || true
  exit 1
fi

echo "mandala-gnn: up at ${URL} (model cold until you press launch)"
echo "mandala-gnn: up at ${URL} (pid ${SERVER})" >>"$LOG"
open_page

if [ "$DETACH" = "1" ]; then
  echo "mandala-gnn: detached (pid ${SERVER}); stop it with $0 stop"
  trap - INT TERM HUP EXIT
  exit 0
fi

# Keep the server in the foreground of this launch. Returning would let
# proot --kill-on-exit take it down.
wait "$SERVER"
