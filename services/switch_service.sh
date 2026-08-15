#!/usr/bin/env bash
# Switch between the GPU services on the MEDIA host: grounding-dino.service,
# qwen3-vl.service and voice-chat.service. They share one 8GB card and aren't
# meant to run together, so this stops whichever ones are active before starting
# the chosen one. Deployed at ~/switch_service.sh (source of truth:
# services/switch_service.sh in the mt4 repo).
#
# SAM 2.1 is not part of this trade: it runs in-process on the arm host
# (mt4_vision.sam), not as a service on this card.
#
# Was switch_vision_service.sh until voice-chat joined: that one is the rover's
# speech stack rather than vision (source: voice_chat/ in the UGV Rover repo),
# so what this set has in common is the card, not the modality.
set -euo pipefail

DINO_SERVICE="grounding-dino.service"
QWEN_SERVICE="qwen3-vl.service"
VOICE_SERVICE="voice-chat.service"
DINO_PORT=8765
QWEN_PORT=8766
VOICE_PORT=8767

is_active() {
  systemctl is-active --quiet "$1"
}

# Tries, not seconds, and the caller picks: dino and qwen are up in well under a
# minute, but voice-chat loads three models and then compiles and warms the
# decode step before it binds the port -- ~150s measured -- so a 60s ceiling
# would report a false timeout on a perfectly healthy start.
wait_for_health() {
  local port="$1" tries="${2:-30}"
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
  if [[ "$arg" == "dino" || "$arg" == "qwen" || "$arg" == "voice" ]]; then
    printf '%s' "$arg"
    return
  fi

  if [[ ! -t 0 ]]; then
    local reply
    while true; do
      read -rp "Switch on which service? [dino/qwen/voice]: " reply
      [[ "$reply" == "dino" || "$reply" == "qwen" || "$reply" == "voice" ]] && { printf '%s' "$reply"; return; }
    done
  fi

  local values=(dino qwen voice)
  # Open on whatever is already running, so Enter confirms the status quo rather
  # than switching away from it -- the menu is used to check state at least as
  # often as to change it. With nothing running, fall back to qwen.
  local selected=1
  if is_active "$DINO_SERVICE"; then
    selected=0
  elif is_active "$QWEN_SERVICE"; then
    selected=1
  elif is_active "$VOICE_SERVICE"; then
    selected=2
  fi
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
is_active "$DINO_SERVICE"  && echo "  [running] grounding-dino  (port $DINO_PORT)" || echo "  [stopped] grounding-dino"
is_active "$QWEN_SERVICE"  && echo "  [running] qwen3-vl        (port $QWEN_PORT)" || echo "  [stopped] qwen3-vl"
is_active "$VOICE_SERVICE" && echo "  [running] voice-chat      (port $VOICE_PORT)" || echo "  [stopped] voice-chat"
echo

target=$(choose_target "${1:-}")

case "$target" in
  dino)  want="$DINO_SERVICE";  port="$DINO_PORT";  tries=30  ;;
  qwen)  want="$QWEN_SERVICE";  port="$QWEN_PORT";  tries=30  ;;
  voice) want="$VOICE_SERVICE"; port="$VOICE_PORT"; tries=150 ;;
esac

# With three services "the other one" is now plural, so the losers are whatever
# is left once the winner is taken out of the set.
for other in "$DINO_SERVICE" "$QWEN_SERVICE" "$VOICE_SERVICE"; do
  [[ "$other" == "$want" ]] && continue
  if is_active "$other"; then
    echo "Stopping $other ..."
    systemctl stop "$other"
  fi
done

if is_active "$want"; then
  echo "$want is already running."
else
  echo "Starting $want ..."
  systemctl start "$want"
fi

echo "Waiting for http://127.0.0.1:${port}/health ..."
if wait_for_health "$port" "$tries"; then
  echo "OK:"
  curl -s "http://127.0.0.1:${port}/health"; echo
else
  echo "Timed out waiting for health check. Check: journalctl -u $want -n 50" >&2
  exit 1
fi
