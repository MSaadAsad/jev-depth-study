"""Resumable E1 evaluation without activation storage.

Run --depths 1 for the prompt gate, then omit --depths to finish the same run.
Every completed example is fsynced; a torn final line is discarded on resume.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import time

import torch
from huggingface_hub import HfApi

from jevprobe.behavior import write_summary
from jevprobe.data.transitive import load_jsonl
from jevprobe.models import load_model, parse_model_ref
from jevprobe.prompts import render
from jevprobe.readout import forward_score, letter_ids


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def source_digest() -> str:
    root = Path(__file__).parent
    return digest(b"".join(p.relative_to(root).as_posix().encode() + b"\0" + p.read_bytes()
                           for p in sorted(root.rglob("*.py"))))


def resolve_revision(repo: str, revision: str | None) -> str:
    resolved = HfApi().model_info(repo, revision=revision).sha
    if not resolved:
        raise ValueError("evaluation requires a Hub model with an immutable commit revision")
    return resolved


def atomic_json(path: Path, value: dict) -> None:
    tmp = path.with_suffix(".tmp")
    with tmp.open("w") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


@contextmanager
def run_lock(out: Path):
    out.mkdir(parents=True, exist_ok=True)
    with (out / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"another evaluator is writing to {out}") from exc
        yield


def read_completed(path: Path, examples) -> list[dict]:
    """Validate completed records and repair only an unterminated final record."""
    if not path.exists():
        return []
    expected = {e.uid: e for e in examples}
    rows, seen = [], set()
    with path.open("r+b") as f:
        while True:
            start = f.tell()
            raw = f.readline()
            if not raw:
                break
            if not raw.endswith(b"\n"):
                f.truncate(start)
                break
            row = json.loads(raw)
            uid = row["uid"]
            if uid in seen or uid not in expected:
                raise ValueError(f"duplicate or unknown completed uid: {uid}")
            ex = expected[uid]
            if (row["d"], row["answer"], row["split"]) != (ex.d, ex.answer, ex.split):
                raise ValueError(f"completed row disagrees with dataset: {uid}")
            if not math.isfinite(row["score"]):
                raise ValueError(f"non-finite completed score: {uid}")
            seen.add(uid)
            rows.append(row)
    return rows


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True, help="Hub repo[@revision]")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    ap.add_argument("--depths", nargs="+", type=int, help="subset to evaluate now; omitted = all")
    args = ap.parse_args(argv)
    data, out = Path(args.data), Path(args.out)
    examples = load_jsonl(data)
    if not examples or len({e.uid for e in examples}) != len(examples):
        raise ValueError("dataset must be nonempty with unique uids")
    if args.depths and not set(args.depths) <= {e.d for e in examples}:
        raise ValueError("requested depths are not in the dataset")
    repo, rev = parse_model_ref(args.model)
    with run_lock(out):
        manifest_path = out / "manifest.json"
        previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
        if not previous and (out / "meta.jsonl").exists():
            raise ValueError("refusing to adopt scores without a provenance manifest")
        # A resumed run stays pinned even if the requested branch/tag has moved.
        resolved = previous["config"]["revision"] if previous else resolve_revision(repo, rev)
        import jevk5.prompt
        config = {
            "format_version": 1, "model": args.model, "revision": resolved,
            "data_sha256": digest(data.read_bytes()), "n_examples": len(examples),
            "source_sha256": source_digest(), "prompt_sha256": digest(Path(jevk5.prompt.__file__).read_bytes()),
            "device": args.device, "dtype": args.dtype,
            "versions": {p: version(p) for p in ("torch", "transformers", "numpy", "scikit-learn", "jevk5")},
            "readout": "fp32 logit(A=true) - logit(B=false); final normalized decision state",
        }
        if previous and previous["config"] != config:
            changed = [k for k in config if previous["config"].get(k) != config[k]]
            raise ValueError(f"resume provenance mismatch: {', '.join(changed)}; use a new output directory")
        if not previous:
            atomic_json(manifest_path, {"config": config, "created_at": datetime.now(timezone.utc).isoformat(),
                                        "data_path": str(data.resolve())})
        rows = read_completed(out / "meta.jsonl", examples)
        done = {r["uid"] for r in rows}
        pending = [e for e in examples if e.uid not in done and (not args.depths or e.d in args.depths)]
        pending.sort(key=lambda e: e.d)

        def status(state):
            atomic_json(out / "status.json", {"state": state, "completed": len(rows), "total": len(examples),
                                              "updated_at": datetime.now(timezone.utc).isoformat()})

        status("running")
        try:
            if pending:
                if args.device == "mps" and not torch.backends.mps.is_available():
                    raise RuntimeError("Apple GPU unavailable in this process; run with native GPU access")
                print(f"Loading {repo}@{resolved}; {len(rows)} completed, {len(pending)} pending", flush=True)
                model, tok = load_model(repo, args.device, getattr(torch, args.dtype), revision=resolved)
                ids = letter_ids(tok)
                started = time.monotonic()
                with (out / "meta.jsonl").open("a") as meta:
                    for i, ex in enumerate(pending, 1):
                        text = render(ex).text
                        inputs = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(args.device)
                        score = forward_score(model, inputs, ids)
                        row = {"uid": ex.uid, "split": ex.split, "d": ex.d, "answer": ex.answer,
                               "rank_x": ex.rank_x, "rank_y": ex.rank_y,
                               "model": args.model, "score": score, "n_tokens": inputs.shape[1],
                               "prompt_sha256": digest(text.encode())}
                        meta.write(json.dumps(row, allow_nan=False) + "\n")
                        meta.flush()
                        os.fsync(meta.fileno())
                        rows.append(row)
                        if i == 1 or i % 25 == 0 or i == len(pending):
                            status("running")
                            print(f"{len(rows)}/{len(examples)} complete; d={ex.d}; "
                                  f"{(time.monotonic() - started) / i:.2f} s/example", flush=True)
                del model
            write_summary(rows, out)
            status("complete" if len(rows) == len(examples) else "partial")
        except BaseException:
            status("interrupted")
            raise


if __name__ == "__main__":
    main()
