from __future__ import annotations

import csv
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import hydra
from omegaconf import DictConfig

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
csv.field_size_limit(10 ** 9)

from src.prompting.prompter import reconstruction_prompt

MODEL_DIRS = {"gemma4_latest": "gemma4:latest", "qwen3_8b": "qwen3:8b"}


def looks_truncated(text: str) -> bool:
    stripped = text.rstrip()
    return bool(stripped) and stripped[-1].isalnum()


def build_history(phase1_csv: Path, context_turns: int) -> list[dict]:
    history: list[dict] = []
    with open(phase1_csv, newline="", errors="replace") as handle:
        for row in csv.DictReader(handle):
            history.append({"role": "user",
                            "content": f"Define '{row['concept_label']}' in FOL."})
            history.append({"role": "assistant",
                            "content": row.get("fol_block") or "[no FOL block]"})
    if context_turns and len(history) > context_turns:
        history = history[-context_turns:]
    return history


def call(base_url: str, model: str, messages: list[dict], num_predict: int,
         num_ctx: int, timeout: float) -> dict:
    options = {"temperature": 0.0, "num_predict": num_predict}
    if num_ctx > 0:
        options["num_ctx"] = num_ctx
    body = {
        "model": model,
        "messages": [dict(m) for m in messages],
        "stream": False,
        "think": False,
        "options": options,
    }
    request = urllib.request.Request(
        base_url.rstrip("/") + "/api/chat",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    started = time.time()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as handle:
            payload = json.load(handle)
    except urllib.error.HTTPError as exc:
        return {"error": exc.read().decode("utf-8", "replace")[:300],
                "elapsed_s": round(time.time() - started, 1)}
    except (TimeoutError, OSError, ValueError) as exc:
        return {"error": f"{type(exc).__name__}: {exc}"[:300],
                "elapsed_s": round(time.time() - started, 1)}
    message = payload.get("message") or {}
    return {
        "content": message.get("content") or "",
        "done_reason": payload.get("done_reason"),
        "eval_count": payload.get("eval_count"),
        "prompt_eval_count": payload.get("prompt_eval_count"),
        "elapsed_s": round(time.time() - started, 1),
    }


@hydra.main(version_base=None, config_path="../../conf", config_name="analysis")
def main(cfg: DictConfig) -> None:
    root = Path(hydra.utils.get_original_cwd())
    spec = cfg.phase3_untruncate
    source = root / spec.source_dir
    target = root / spec.target_dir
    report_path = root / spec.report_json

    cells = []
    for text_path in sorted(source.glob("seed*/*/*/*/reconstruction.txt")):
        cell = text_path.parent
        seed_dir, ontology, model_dir, strategy = cell.relative_to(source).parts[-4:]
        if model_dir not in MODEL_DIRS:
            continue
        saved = text_path.read_text(encoding="utf-8", errors="replace")
        if spec.only_truncated and not looks_truncated(saved):
            continue
        cells.append((cell, seed_dir, ontology, model_dir, strategy, saved))

    if spec.limit:
        cells = cells[: int(spec.limit)]

    print(f"cells to re-run: {len(cells)}  "
          f"(num_predict={spec.num_predict}, num_ctx={spec.num_ctx})")

    results = []
    for index, (cell, seed_dir, ontology, model_dir, strategy, saved) in enumerate(cells, 1):
        phase1 = cell / "phase1_results.csv"
        if not phase1.exists():
            print(f"  [{index}/{len(cells)}] SKIP {seed_dir}/{ontology}/{model_dir}/{strategy}"
                  f" — no phase1_results.csv")
            continue
        history = build_history(phase1, int(spec.context_turns))
        messages = reconstruction_prompt(history, ontology)
        outcome = call(str(spec.api_base), MODEL_DIRS[model_dir], messages,
                       int(spec.num_predict), int(spec.num_ctx), float(spec.timeout_s))

        label = f"{seed_dir}/{ontology}/{model_dir}/{strategy}"
        if "error" in outcome:
            print(f"  [{index}/{len(cells)}] FAIL {label} {outcome['elapsed_s']}s "
                  f"{outcome['error']}")
            results.append({"cell": label, "error": outcome["error"],
                            "elapsed_s": outcome["elapsed_s"]})
            continue

        new = outcome["content"]
        prefix = 0
        limit = min(len(saved), len(new))
        while prefix < limit and saved[prefix] == new[prefix]:
            prefix += 1
        extends = new.startswith(saved.rstrip())
        record = {
            "cell": label,
            "model": MODEL_DIRS[model_dir],
            "saved_chars": len(saved),
            "new_chars": len(new),
            "common_prefix_chars": prefix,
            "prefix_fraction_of_saved": round(prefix / max(len(saved), 1), 4),
            "new_extends_saved": extends,
            "done_reason": outcome["done_reason"],
            "eval_count": outcome["eval_count"],
            "prompt_eval_count": outcome["prompt_eval_count"],
            "elapsed_s": outcome["elapsed_s"],
        }
        results.append(record)
        print(f"  [{index}/{len(cells)}] {label}  {outcome['elapsed_s']}s  "
              f"{len(saved)}->{len(new)}ch  done={outcome['done_reason']} "
              f"tokens={outcome['eval_count']}  extends={extends} "
              f"prefix={record['prefix_fraction_of_saved']:.2f}")

        if spec.write:
            out_cell = target / seed_dir / ontology / model_dir / strategy
            out_cell.mkdir(parents=True, exist_ok=True)
            (out_cell / "reconstruction.txt").write_text(new, encoding="utf-8")

        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps({
            "num_predict": int(spec.num_predict),
            "num_ctx": int(spec.num_ctx),
            "source_dir": str(spec.source_dir),
            "wrote_outputs": bool(spec.write),
            "complete": False,
            "cells": results,
        }, indent=2))

    stopped = [r for r in results if r.get("done_reason") == "stop"]
    capped = [r for r in results if r.get("done_reason") == "length"]
    extended = [r for r in results if r.get("new_extends_saved")]
    print()
    print(f"stopped naturally : {len(stopped)}/{len(results)}")
    print(f"hit the new cap   : {len(capped)}/{len(results)}")
    print(f"extends saved text: {len(extended)}/{len(results)}")
    if results:
        total = sum(r.get("elapsed_s", 0) for r in results)
        print(f"wall time         : {total/60:.1f} min for {len(results)} cells")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps({
        "num_predict": int(spec.num_predict),
        "num_ctx": int(spec.num_ctx),
        "source_dir": str(spec.source_dir),
        "wrote_outputs": bool(spec.write),
        "complete": True,
        "cells": results,
    }, indent=2))
    print(f"report -> {spec.report_json}")
    if not spec.write:
        print("write=false — no reconstruction was saved. "
              "Re-run with phase3_untruncate.write=true to persist.")


if __name__ == "__main__":
    main()
