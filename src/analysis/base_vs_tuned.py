from __future__ import annotations

import csv
import json
import sys
import time
import urllib.request
from pathlib import Path

import hydra
from omegaconf import DictConfig

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
csv.field_size_limit(10 ** 9)

FEW_SHOT = """Define each concept in first-order logic as a biconditional.

Concept: Bicycle
Definition: forall x: Bicycle(x) <-> (Vehicle(x) and HasTwoWheels(x) and HumanPowered(x))

Concept: Estuary
Definition: forall x: Estuary(x) <-> (WaterBody(x) and PartiallyEnclosed(x) and MixesFreshAndSaltWater(x))

Concept: Thermometer
Definition: forall x: Thermometer(x) <-> (Instrument(x) and MeasuresTemperature(x))

Concept: {label}
Definition:"""

RESULT_COLS = ["ontology", "arm", "family", "llm_model", "concept_iri", "concept_label",
               "prompt_chars", "response_text", "fol_block", "parse_ok", "latency_s"]


def generate_raw(api_base: str, model: str, prompt: str,
                 max_tokens: int, timeout: float) -> str:
    url = api_base.rstrip("/").replace("/v1", "") + "/api/generate"
    body = json.dumps({
        "model": model, "prompt": prompt, "raw": True, "stream": False,
        "options": {"temperature": 0.0, "num_predict": max_tokens},
    }).encode()
    request = urllib.request.Request(url, data=body,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response).get("response", "") or ""


@hydra.main(version_base=None, config_path="../../conf", config_name="analysis")
def main(cfg: DictConfig) -> None:
    from src.analysis.fol_to_owl import classify_definition

    root = Path(hydra.utils.get_original_cwd())
    spec = cfg.base_tuned
    out_path = root / spec.out_csv
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done: set[tuple[str, str, str]] = set()
    if out_path.exists():
        for row in csv.DictReader(open(out_path, newline="", errors="replace")):
            done.add((row["llm_model"], row["ontology"], row["concept_label"]))
        print(f"resume: {len(done)} rows already present")

    new = not out_path.exists()
    handle = open(out_path, "a", newline="")
    writer = csv.DictWriter(handle, fieldnames=RESULT_COLS)
    if new:
        writer.writeheader()

    for ontology in spec.ontologies:
        sample = root / f"data/samples/concepts_{ontology}_seed{spec.seed}.json"
        if not sample.exists():
            print(f"  no sample for {ontology}, skipped")
            continue
        concepts = json.loads(sample.read_text())
        for pair in spec.pairs:
            for arm in ("base", "tuned"):
                model = pair[arm]
                for concept in concepts:
                    label = str(concept.get("label", "")).strip()
                    if not label or (model, ontology, label) in done:
                        continue
                    prompt = FEW_SHOT.format(label=label)
                    start = time.time()
                    try:
                        text = generate_raw(spec.api_base, model, prompt,
                                            int(spec.max_tokens), float(spec.timeout_s))
                    except Exception as exc:
                        text = ""
                        print(f"    call failed {model}/{label[:24]}: {str(exc)[:60]}")
                    elapsed = time.time() - start
                    body = text.split("Concept:")[0].strip()
                    classes, _ = classify_definition(body)
                    writer.writerow({
                        "ontology": ontology, "arm": arm, "family": pair["family"],
                        "llm_model": model, "concept_iri": concept.get("iri", ""),
                        "concept_label": label, "prompt_chars": len(prompt),
                        "response_text": " ".join(text.split())[:4000],
                        "fol_block": " ".join(body.split())[:1200],
                        "parse_ok": int(classes is not None),
                        "latency_s": round(elapsed, 2),
                    })
                    handle.flush()
                print(f"  {ontology:8s} {pair['family']:10s} {arm:5s} {model:24s} done")
    handle.close()
    print(f"written: {out_path.relative_to(root)}")


if __name__ == "__main__":
    main()
