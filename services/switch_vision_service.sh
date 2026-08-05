#!/usr/bin/env bash
# Switch between grounding-dino.service and qwen3-vl.service on the MEDIA GPU host.
# They share one 8GB card and aren't meant to run together, so this stops
# whichever one is active before starting the other. Deployed at ~/switch_vision_service.sh
# (source of truth: services/switch_vision_service.sh in the mt4 repo).
#
# sam2.service is not part of the trade: SAM 2.1 small holds ~600MB including
# its CUDA context, so it stays up alongside whichever of the two is running.
set -euo pipefail

DINO_SERVICE="grounding-dino.service"
QWEN_SERVICE="qwen3-vl.service"
DINO_PORT=8765
QWEN_PORT=8766

is_active() {
  systemctl is-active --quiet "$1"
}

wait_for_health() {
  local port="$1" tries=30
  while (( tries > 0 )); do
    curl -sf "http://127.0.0.1:${port}/health" >/dev/null 2>&1 && return 0
    sleep 2
    tries=$((tries - 1))
  done
  return 1
}

# Arrow-key menu: draws to stderr, prints the chosen value (and only that) to
# stdout so callers can do target=$(choose_target "$1"). Falls back to a plain
# text prompt when stdin isn't a real terminal (arrow keys need one).
choose_target() {
  local arg="${1:-}"
  if [[ "$arg" == "dino" || "$arg" == "qwen" ]]; then
    printf '%s' "$arg"
    return
  fi

  if [[ ! -t 0 ]]; then
    local reply
    while true; do
      read -rp "Switch on which service? [dino/qwen]: " reply
      [[ "$reply" == "dino" || "$reply" == "qwen" ]] && { printf '%s' "$reply"; return; }
    done
  fi

  local values=(dino qwen)
  local selected=0
  local n=${#values[@]}
  local key rest i

  draw_options() {
    for i in "${!values[@]}"; do
      if [[ $i -eq $selected ]]; then
        printf '\033[7m> %s\033[0m\n' "${values[$i]}" >&2
      else
        printf '  %s\n' "${values[$i]}" >&2
      fi
    done
  }

  echo "Switch on which service? (Up/Down, Enter to confirm)" >&2
  tput civis >&2 2>/dev/null || true
  draw_options
  while true; do
    IFS= read -rsn1 key
    if [[ $key == $'\x1b' ]]; then
      read -rsn2 -t 0.05 rest 2>/dev/null || rest=""
      case "$rest" in
        '[A') selected=$(( (selected - 1 + n) % n )) ;;
        '[B') selected=$(( (selected + 1) % n )) ;;
      esac
    elif [[ -z $key ]]; then
      break
    fi
    tput cuu "$n" >&2 2>/dev/null || true
    draw_options
  done
  tput cnorm >&2 2>/dev/null || true
  echo "-> ${values[$selected]}" >&2

  printf '%s' "${values[$selected]}"
}

echo "Current status:"
is_active "$DINO_SERVICE" && echo "  [running] grounding-dino  (port $DINO_PORT)" || echo "  [stopped] grounding-dino"
is_active "$QWEN_SERVICE" && echo "  [running] qwen3-vl        (port $QWEN_PORT)" || echo "  [stopped] qwen3-vl"
echo

target=$(choose_target "${1:-}")

if [[ "$target" == "dino" ]]; then
  want="$DINO_SERVICE"; other="$QWEN_SERVICE"; port="$DINO_PORT"
else
  want="$QWEN_SERVICE"; other="$DINO_SERVICE"; port="$QWEN_PORT"
fi

if is_active "$other"; then
  echo "Stopping $other ..."
  systemctl stop "$other"
fi

if is_active "$want"; then
  echo "$want is already running."
else
  echo "Starting $want ..."
  systemctl start "$want"
fi

echo "Waiting for http://127.0.0.1:${port}/health ..."
if wait_for_health "$port"; then
  echo "OK:"
  curl -s "http://127.0.0.1:${port}/health"; echo
else
  echo "Timed out waiting for health check. Check: journalctl -u $want -n 50" >&2
  exit 1
fi
