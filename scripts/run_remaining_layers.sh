#!/usr/bin/env bash
# Explicit launch only. Preparation never invokes this worker.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
SESSION=jev-layer-remainder
LAYERS=(1 2 3 5 6 7 9 10 11 13 14 15 17 18 19 21 22 23 25 26 27 29 30 31)
if [[ ${1:-} != --worker ]]; then
  if [[ $# -ne 0 ]]; then echo 'Usage: bash scripts/run_remaining_layers.sh' >&2; exit 2; fi
  command -v tmux >/dev/null
  command -v caffeinate >/dev/null
  .venv/bin/python scripts/prepare_remaining_layers.py
  if tmux has-session -t "=$SESSION" 2>/dev/null; then
    dead=$(tmux list-panes -t "$SESSION" -F '#{pane_dead}')
    if [[ -n "$dead" && "$dead" != *0* ]]; then tmux kill-session -t "$SESSION"
    else echo "Session already active: $SESSION" >&2; exit 1; fi
  fi
  printf -v worker_command 'caffeinate -is bash %q --worker' "$ROOT/scripts/run_remaining_layers.sh"
  tmux new-session -d -s "$SESSION" -c "$ROOT" "$worker_command"
  tmux set-option -t "$SESSION" remain-on-exit on
  echo "Started $SESSION. Inspect: tmux attach -t $SESSION. Ctrl-C stops; rerun resumes saved pairs."
  exit 0
fi
mkdir -p runs/layer-remainder
exec > >(tee -a runs/layer-remainder/runner.log) 2>&1
trap 'rc=$?; printf "exit_code=%s finished_at=%s\n" "$rc" "$(date -u +%FT%TZ)" > runs/layer-remainder/launcher-status.txt' EXIT
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_PROGRESS_BARS=1
# Check provenance again inside the worker before loading any weights.
.venv/bin/python scripts/prepare_remaining_layers.py
.venv/bin/python -u scripts/screen_layers.py --model jev --source-plan data/layer-followup/plan.json --layers "${LAYERS[@]}" --out runs/layer-remainder/jev
.venv/bin/python -u scripts/screen_layers.py --model qwen --source-plan data/layer-followup/plan.json --layers "${LAYERS[@]}" --out runs/layer-remainder/qwen
.venv/bin/python -u scripts/combine_layer_screens.py
