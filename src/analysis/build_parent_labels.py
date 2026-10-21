from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


class MissingSample(FileNotFoundError):
    pass


def _targets(samples_dir: Path, ontology: str, seed: int, builtin: re.Pattern) -> tuple[set[str], int]:
    path = samples_dir / f"concepts_{ontology}_seed{seed}.json"
    if not path.exists():
        raise MissingSample(path)
    concepts = json.loads(path.read_text())
    want: set[str] = set()
    links = 0
    for c in concepts:
        for p in c.get("parents", []):
            if builtin.search(p):
                continue
            want.add(p)
            links += 1
    return want, links


def _from_rdfxml(path: Path, want: set[str], base: str) -> dict[str, str]:
    text = path.read_text(encoding="utf8", errors="replace")
    out: dict[str, str] = {}
    for m in re.finditer(r'<owl:Class\b[^>]*rdf:about="([^"]+)"(.*?)</owl:Class>', text, re.S):
        iri = m.group(1)
        full = iri if iri.startswith("http") else base + iri
        if full not in want:
            continue
        lab = re.search(r"<rdfs:label[^>]*>([^<]+)</rdfs:label>", m.group(2))
        if lab:
            out[full] = lab.group(1).strip()
    return out


def _from_turtle(path: Path, want: set[str], prefix: str) -> dict[str, str]:
    subj_re = re.compile(rf"^<({re.escape(prefix)}[^>]+)>")
    lab_re = re.compile(r'skos:prefLabel\s+(?:"""(.*?)"""|"([^"]*)")')
    out: dict[str, str] = {}
    current: str | None = None
    with path.open(encoding="utf8", errors="replace") as f:
        for line in f:
            if line[:1] == "<":
                m = subj_re.match(line)
                current = m.group(1) if m else None
                continue
            if current in want and "skos:prefLabel" in line:
                m = lab_re.search(line)
                if m:
                    out[current] = (m.group(1) or m.group(2) or "").strip()
    return out


_ABOUT = re.compile(r'<rdf:Description\s+rdf:about="([^"]+)"')
_DIRECT_LABEL = re.compile(r"<skos:prefLabel[^>]*>([^<]+)</skos:prefLabel>")
_XL_REF = re.compile(r'<skosxl:prefLabel\s+rdf:resource="([^"]+)"')
_LITERAL = re.compile(r"<skosxl:literalForm[^>]*>([^<]+)</skosxl:literalForm>")


def _from_skosxl(path: Path, want: set[str], lang: str) -> dict[str, str]:
    marker = f"xl_{lang}_"
    resolved: dict[str, str] = {}
    pending: dict[str, str] = {}
    current: str | None = None
    with path.open(encoding="utf8", errors="replace") as handle:
        for line in handle:
            found = _ABOUT.search(line)
            if found:
                current = found.group(1)
                continue
            if current is None or current not in want:
                continue
            literal = _DIRECT_LABEL.search(line)
            if literal:
                resolved.setdefault(current, literal.group(1).strip())
                continue
            ref = _XL_REF.search(line)
            if ref and marker in ref.group(1):
                pending.setdefault(ref.group(1), current)

    if not pending:
        return resolved

    current = None
    with path.open(encoding="utf8", errors="replace") as handle:
        for line in handle:
            found = _ABOUT.search(line)
            if found:
                current = found.group(1)
                continue
            if current is None or current not in pending:
                continue
            literal = _LITERAL.search(line)
            if literal:
                resolved.setdefault(pending[current], literal.group(1).strip())
    return resolved


def _resolve(spec: DictConfig, path: Path, want: set[str]) -> dict[str, str]:
    fmt = str(spec.format)
    if fmt == "rdfxml":
        return _from_rdfxml(path, want, str(spec.base))
    if fmt == "turtle":
        return _from_turtle(path, want, str(spec.iri_prefix))
    if fmt == "skosxl":
        return _from_skosxl(path, want, str(getattr(spec, "lang", "en")))
    raise ValueError(
        f"unknown format {fmt!r}; expected 'rdfxml', 'turtle' or 'skosxl'")


@hydra.main(version_base=None, config_path="../../conf", config_name="analysis")
def main(cfg: DictConfig) -> None:
    root = Path(hydra.utils.get_original_cwd())
    pl = cfg.parent_labels
    seed = int(pl.seed)
    samples_dir = root / str(pl.samples_dir)
    ontologies_dir = root / str(pl.ontologies_dir)
    builtin = re.compile(str(pl.builtin_pattern))

    missing: list[str] = []
    for ontology, spec in pl.sources.items():
        try:
            want, links = _targets(samples_dir, ontology, seed, builtin)
        except MissingSample as exc:
            print(f"  {ontology:9} NO SAMPLE at {exc.args[0]} -- wrong seed?")
            missing.append(ontology)
            continue
        if not want:
            print(f"  {ontology:9} no non-builtin parents in the sample; nothing to resolve")
            continue
        src = ontologies_dir / str(spec.file)
        if not src.exists():
            print(f"  {ontology:9} SKIPPED: {src} not found")
            continue
        mapping = _resolve(spec, src, want)
        out = samples_dir / str(pl.out_pattern).format(ontology=ontology, seed=seed)
        out.write_text(json.dumps(mapping, indent=1, sort_keys=True))
        print(
            f"  {ontology:9} {len(mapping)}/{len(want)} distinct parent IRIs resolved "
            f"({links} links) -> {out.relative_to(root)}"
        )

    for ontology in pl.skip:
        try:
            want, links = _targets(samples_dir, ontology, seed, builtin)
        except MissingSample:
            print(f"  {ontology:9} skipped: no sample for seed {seed}")
            missing.append(ontology)
            continue
        note = "all parents are builtins" if links == 0 else f"{links} links, no resolver"
        print(f"  {ontology:9} skipped: {note}")

    if len(missing) == len(pl.sources) + len(pl.skip):
        raise SystemExit(
            f"no concept sample found for seed {seed} in {samples_dir} -- nothing was written"
        )


if __name__ == "__main__":
    main()
