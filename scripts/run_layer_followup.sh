#!/usr/bin/env bash
# Starting this launcher is an explicit GPU-run action; preparation does not invoke it.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
SESSION=jev-layer-followup
if [[ ${1:-} != --worker ]]; then
  if [[ $# -ne 0 ]]; then echo 'Usage: bash scripts/run_layer_followup.sh' >&2; exit 2; fi
  command -v tmux >/dev/null
  command -v caffeinate >/dev/null
  if tmux has-session -t "=$SESSION" 2>/dev/null; then
    dead=$(tmux list-panes -t "$SESSION" -F '#{pane_dead}')
    if [[ -n "$dead" && "$dead" != *0* ]]; then tmux kill-session -t "$SESSION"
    else echo "Session already active: $SESSION" >&2; exit 1; fi
  fi
  .venv/bin/python scripts/prepare_layer_followup.py
  printf -v worker_command 'caffeinate -is bash %q --worker' "$ROOT/scripts/run_layer_followup.sh"
  tmux new-session -d -s "$SESSION" -c "$ROOT" "$worker_command"
  tmux set-option -t "$SESSION" remain-on-exit on
  echo "Started $SESSION. Inspect: tmux attach -t $SESSION. Ctrl-C stops; rerun resumes saved pairs."
  exit 0
fi
mkdir -p runs/layer-followup
exec > >(tee -a runs/layer-followup/runner.log) 2>&1
trap 'rc=$?; printf "exit_code=%s finished_at=%s\n" "$rc" "$(date -u +%FT%TZ)" > runs/layer-followup/launcher-status.txt' EXIT
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_PROGRESS_BARS=1
.venv/bin/python -u scripts/screen_layers.py --model jev --source-plan data/layer-followup/plan.json --out runs/layer-followup/jev
.venv/bin/python -u scripts/screen_layers.py --model qwen --source-plan data/layer-followup/plan.json --out runs/layer-followup/qwen
