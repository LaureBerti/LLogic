"""Reference-definition cascade for judging LLM concept definitions (the reasoner validation / the reasoner validation).

The audit needs an authority to compare an LLM definition against. No single source
covers the five ontologies, so sources are tried in order of authority and the one
actually used is recorded per concept:

  1. the ontology's own definition   -- what the ontology itself intends. Complete for
                                        MeSH (100%) and Book (92%); absent for Anatomy
                                        and CSO. AGROVOC stores dereferenceable URIs
                                        rather than text, so those are rejected here.
  2. the local macOS dictionary      -- New Oxford American via DictionaryServices.
                                        Good for common nouns and, through a head-noun
                                        fallback, for the *genus* of compound anatomical
                                        terms ("Internal_Anal_Sphincter" -> "sphincter").
  3. Merriam-Webster Medical API     -- stub. Would cover Anatomy, the one real gap.
                                        Needs a free non-commercial key in MW_MEDICAL_KEY.
                                        Scraping the website is not an option: it breaches
                                        their terms of use.

Everything here is offline and read-only apart from the (unconfigured) tier 3.

A caveat that must travel with any number produced from this: the models were trained
on dictionary and encyclopedia text, so agreement with a reference is weak evidence of
understanding. The informative signal is DISAGREEMENT -- a model contradicting a source
it has almost certainly seen.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import json
import os
import re
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Tier 2: local macOS dictionary via DictionaryServices
# ---------------------------------------------------------------------------
_kUTF8 = 0x08000100

class _CFRange(ctypes.Structure):
    _fields_ = [("loc", ctypes.c_long), ("len", ctypes.c_long)]

def _load_dict_services():
    try:
        cs = ctypes.cdll.LoadLibrary(ctypes.util.find_library("CoreServices"))
        cf = ctypes.cdll.LoadLibrary(ctypes.util.find_library("CoreFoundation"))
    except (OSError, TypeError):
        return None, None
    cf.CFStringCreateWithCString.restype = ctypes.c_void_p
    cf.CFStringCreateWithCString.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint32]
    cs.DCSCopyTextDefinition.restype = ctypes.c_void_p
    cs.DCSCopyTextDefinition.argtypes = [ctypes.c_void_p, ctypes.c_void_p, _CFRange]
    cf.CFStringGetLength.restype = ctypes.c_long
    cf.CFStringGetLength.argtypes = [ctypes.c_void_p]
    cf.CFStringGetCString.restype = ctypes.c_bool
    cf.CFStringGetCString.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_long, ctypes.c_uint32]
    return cs, cf

_CS, _CF = _load_dict_services()

def dictionary_lookup(word: str) -> str | None:
    """Definition text from the local dictionary, or None."""
    if not word or _CS is None:
        return None
    s = _CF.CFStringCreateWithCString(None, word.encode("utf-8"), _kUTF8)
    if not s:
        return None
    result = _CS.DCSCopyTextDefinition(None, s, _CFRange(0, len(word)))
    if not result:
        return None
    size = _CF.CFStringGetLength(result) * 4 + 8
    buf = ctypes.create_string_buffer(size)
    if not _CF.CFStringGetCString(result, buf, size, _kUTF8):
        return None
    return " ".join(buf.value.decode("utf-8", "replace").split())

def label_variants(label: str):
    """Surface forms to try, most specific first.

    The head-noun fallback (last token) does NOT define the concept; it supplies the
    expected genus for a compound term. Callers must treat a hit from that variant as
    genus-only evidence, which is why the matched variant is reported alongside.
    """
    label = (label or "").strip()
    if not label:
        return
    yield label
    spaced = label.replace("_", " ").strip()
    yield spaced
    yield spaced.lower()
    if spaced.lower().endswith("s") and not spaced.lower().endswith("ss"):
        yield spaced[:-1]
    if "," in spaced:                       # "Head Injuries, Penetrating"
        yield spaced.split(",")[0].strip()
    if " " in spaced:                       # head-noun fallback -- genus only
        yield spaced.split()[-1]

# ---------------------------------------------------------------------------
# Tier 1: the ontology's own definitions
# ---------------------------------------------------------------------------
_URI_DEF = re.compile(r"^https?://")

def load_ontology_definitions(samples_dir: str | Path = "data/samples",
                              seed: int = 42) -> dict[tuple[str, str], str]:
    """Map (ontology, concept_label) -> definition text from the sampled concepts.

    AGROVOC stores dereferenceable URIs in this field rather than definition text;
    those are rejected so they cannot masquerade as a reference definition.
    """
    out: dict[tuple[str, str], str] = {}
    for path in sorted(Path(samples_dir).glob(f"concepts_*_seed{seed}.json")):
        if "INVALID" in path.name:
            continue
        onto = path.name.split("_")[1]
        try:
            concepts = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        for concept in concepts:
            text = (concept.get("definition") or "").strip()
            if text and not _URI_DEF.match(text):
                out[(onto, concept.get("label", "").strip())] = text
    return out

# ---------------------------------------------------------------------------
# Tier 3: Merriam-Webster Medical (stub -- needs a key)
# ---------------------------------------------------------------------------
def merriam_webster_medical(term: str) -> str | None:
    """Official MW Medical API. Returns None unless MW_MEDICAL_KEY is set.

    The key is read from the environment and never stored in the repository.
    Register for a free non-commercial key at dictionaryapi.com; the website itself
    must not be scraped.
    """
    key = os.environ.get("MW_MEDICAL_KEY")
    if not key or not term:
        return None
    import urllib.parse
    import urllib.request

    url = ("https://dictionaryapi.com/api/v3/references/medical/json/"
           f"{urllib.parse.quote(term)}?key={urllib.parse.quote(key)}")
    try:
        with urllib.request.urlopen(url, timeout=15) as response:
            data = json.load(response)
    except Exception:
        return None
    if not isinstance(data, list) or not data or not isinstance(data[0], dict):
        return None            # a list of strings means "no entry, did you mean..."
    shortdef = data[0].get("shortdef") or []
    return "; ".join(shortdef) if shortdef else None

# ---------------------------------------------------------------------------
# The cascade
# ---------------------------------------------------------------------------
GENUS_RE = re.compile(
    r"\b(?:a|an|the)\s+([a-z][a-z \-]{2,40}?)(?=[,.;:]| that| which| with| used| for| in\b| of\b)",
    re.IGNORECASE)

def extract_genus(definition: str) -> str | None:
    """First 'a/an <noun phrase>' in a definition -- its genus, roughly."""
    if not definition:
        return None
    tail = definition
    marker = re.search(r"\bnoun\b", definition, re.IGNORECASE)
    if marker:
        tail = definition[marker.end():]
    match = GENUS_RE.search(tail)
    return " ".join(match.group(1).split()).lower() if match else None

def reference_definition(ontology: str, label: str,
                         onto_defs: dict[tuple[str, str], str] | None = None) -> dict[str, Any]:
    """Best available reference definition, with the source recorded.

    Returns source ∈ {ontology, dictionary, dictionary_headnoun, mw_medical, none}.
    'dictionary_headnoun' flags that only the genus is trustworthy.
    """
    onto_defs = onto_defs if onto_defs is not None else {}
    text = onto_defs.get((ontology, (label or "").strip()))
    if text:
        return {"source": "ontology", "matched": label, "definition": text,
                "genus": extract_genus(text)}

    for i, variant in enumerate(label_variants(label)):
        found = dictionary_lookup(variant)
        if found:
            head_only = (" " in (label or "").replace("_", " ").strip()
                         and variant == (label or "").replace("_", " ").strip().split()[-1])
            return {"source": "dictionary_headnoun" if head_only else "dictionary",
                    "matched": variant, "definition": found, "genus": extract_genus(found)}

    found = merriam_webster_medical((label or "").replace("_", " "))
    if found:
        return {"source": "mw_medical", "matched": label, "definition": found,
                "genus": extract_genus(found)}

    return {"source": "none", "matched": None, "definition": None, "genus": None}
