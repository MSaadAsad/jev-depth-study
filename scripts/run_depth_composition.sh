#!/usr/bin/env bash
# Resumable overnight E1 depth curves plus token-aligned causal patching.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_PROGRESS_BARS=1
SESSION=jev-depth-compose
if [[ ${1:-} != --worker ]]; then
  if [[ $# -ne 0 ]]; then echo "Usage: bash scripts/run_depth_composition.sh" >&2; exit 2; fi
  command -v tmux >/dev/null || { echo 'tmux is required.' >&2; exit 1; }
  if tmux has-session -t "=$SESSION" 2>/dev/null; then
    dead=$(tmux list-panes -t "$SESSION" -F '#{pane_dead}')
    if [[ -n "$dead" && "$dead" != *0* ]]; then
      tmux kill-session -t "$SESSION"
    else
      echo "Session $SESSION is active; inspect with: tmux attach -t $SESSION"; exit 1
    fi
  fi
  # Build and persist the exact aligned pair manifest once, outside the GPU job.
  .venv/bin/python -u scripts/compose_patch.py --prepare-only
  printf -v worker_command 'bash %q --worker' "$ROOT/scripts/run_depth_composition.sh"
  tmux new-session -d -s "$SESSION" -c "$ROOT" "$worker_command"
  tmux set-option -t "$SESSION" remain-on-exit on
  echo "Started $SESSION. Watch: tmux attach -t $SESSION. Stop: Ctrl-C while attached."
  echo "Log: $ROOT/runs/depth-composition/runner.log"
  echo 'It resumes both E1 runs, then runs the six-depth JevK5 counterfactual patch study.'
  exit 0
fi
mkdir -p runs/depth-composition
exec > >(tee -a runs/depth-composition/runner.log) 2>&1
trap 'rc=$?; printf "exit_code=%s finished_at=%s\n" "$rc" "$(date -u +%FT%TZ)" > runs/depth-composition/launcher-status.txt' EXIT
echo "Started full depth plus composition study at $(date -u +%FT%TZ)"
.venv/bin/python -u -m jevprobe.evaluate --data data/e1/test.jsonl \
  --model alibiserikbay/JevK5@v0.3 --out runs/e1-jevk5-v03 --device mps --dtype bfloat16
.venv/bin/python -u -m jevprobe.evaluate --data data/e1/test.jsonl \
  --model Qwen/Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --out runs/e1-qwen35-4b --device mps --dtype bfloat16
echo "Behavior curves complete at $(date -u +%FT%TZ); beginning counterfactual patches."
if command -v caffeinate >/dev/null; then
  caffeinate -i .venv/bin/python -u scripts/compose_patch.py
else
  .venv/bin/python -u scripts/compose_patch.py
fi
echo "Depth and composition study complete at $(date -u +%FT%TZ)."
