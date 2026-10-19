#!/usr/bin/env bash
# Resumable overnight comparison: identical prompts for JevK5 and base Qwen.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
SESSION=jev-context-factorial

if [[ ${1:-} != --worker ]]; then
  if [[ $# -ne 0 ]]; then echo "Usage: bash scripts/run_context_factorial.sh" >&2; exit 2; fi
  command -v tmux >/dev/null || { echo 'tmux is required.' >&2; exit 1; }
  if tmux has-session -t "=$SESSION" 2>/dev/null; then
    dead=$(tmux list-panes -t "$SESSION" -F '#{pane_dead}')
    if [[ -n "$dead" && "$dead" != *0* ]]; then tmux kill-session -t "$SESSION";
    else echo "Session $SESSION is active; inspect with: tmux attach -t $SESSION"; exit 1; fi
  fi
  .venv/bin/python -u scripts/make_context_factorial.py
  if command -v caffeinate >/dev/null; then
    # Hold one assertion through model loading, both models and final analysis.
    # Per-model guards leave an unprotected gap when an idle Mac changes jobs.
    printf -v worker_command 'caffeinate -is bash %q --worker' "$ROOT/scripts/run_context_factorial.sh"
  else
    printf -v worker_command 'bash %q --worker' "$ROOT/scripts/run_context_factorial.sh"
  fi
  tmux new-session -d -s "$SESSION" -c "$ROOT" "$worker_command"
  tmux set-option -t "$SESSION" remain-on-exit on
  echo "Started $SESSION. Watch: tmux attach -t $SESSION"
  echo "Stop: tmux attach -t $SESSION, then Ctrl-C. Resume: rerun this command."
  echo "Log: $ROOT/runs/context-factorial/runner.log"
  exit 0
fi

mkdir -p runs/context-factorial
exec > >(tee -a runs/context-factorial/runner.log) 2>&1
trap 'rc=$?; printf "exit_code=%s finished_at=%s\n" "$rc" "$(date -u +%FT%TZ)" > runs/context-factorial/launcher-status.txt' EXIT
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_PROGRESS_BARS=1
echo "Started paired depth × context evaluation at $(date -u +%FT%TZ)"

.venv/bin/python -u -m jevprobe.evaluate \
  --data data/context-factorial/test.jsonl \
  --model alibiserikbay/JevK5@c4f7fdb3aeab5582336406e78d3bef11bf98833d \
  --out runs/context-factorial/jevk5 --device mps --dtype bfloat16
.venv/bin/python -u -m jevprobe.evaluate \
  --data data/context-factorial/test.jsonl \
  --model Qwen/Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --out runs/context-factorial/qwen35-4b --device mps --dtype bfloat16

.venv/bin/python -u scripts/summarize_context_factorial.py
echo "Paired depth × context evaluation complete at $(date -u +%FT%TZ)."
