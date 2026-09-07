#!/usr/bin/env python3
"""
tutorial.py — a 60-second tour of the LLOGIC core: compiling first-order-logic (FOL)
concept definitions into a Z3 satisfiability check.

This is the engine behind Phase 1 of the paper. It shows how a *definition* can be
syntactically fluent yet logically unsatisfiable, and how Z3 catches it.

Run:
    pip install z3-solver==4.13.3.0
    python3 tutorial.py

Optional (LLM section): install Ollama (https://ollama.ai), then `ollama pull llama3.2`.
"""
from __future__ import annotations

# ── Optional LLM backend (guarded — the tutorial runs without it) ──────────────
try:
    from openai import OpenAI
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False

import z3

# ── 1. A tiny FOL -> Z3 compiler (the heart of Phase 1) ────────────────────────
def check_definition(name: str, body: z3.BoolRef) -> str:
    """Return 'unsat' (contradiction), 'sat' (satisfiable), or 'unknown'.

    We assert exists x: body(x) and ask Z3 whether any model exists. A concept whose
    definition is unsatisfiable describes the empty set -- a logical contradiction.
    """
    s = z3.Solver()
    s.set("timeout", 5000)
    s.add(body)
    return str(s.check())

def demo_handwritten() -> None:
    print("=" * 64)
    print(" 1. Hand-written FOL definitions checked by Z3")
    print("=" * 64)

    E = z3.DeclareSort("Entity")
    x = z3.Const("x", E)
    Written = z3.Function("Written", E, z3.BoolSort())
    Published = z3.Function("Published", E, z3.BoolSort())

    # A coherent definition:  Book(x) <-> Written(x) AND Published(x)
    coherent = z3.And(Written(x), Published(x))
    print(f"  Book(x) := Written(x) ∧ Published(x)        -> {check_definition('Book', coherent)}")

    # A contradictory definition:  Book(x) <-> Written(x) AND ¬Written(x)
    contradiction = z3.And(Written(x), z3.Not(Written(x)))
    print(f"  Book(x) := Written(x) ∧ ¬Written(x)         -> {check_definition('Book', contradiction)}")

    print()
    print("  The second definition is 'unsat': no entity can be both Written and")
    print("  not-Written. LLOGIC flags exactly this failure mode at scale.")
    print()

# ── 2. (Optional) ask a local LLM to define a concept, then check it ───────────
def demo_llm(concept: str = "Hardcover book",
             model: str = "llama3.2:latest",
             base_url: str = "http://localhost:11434/v1") -> None:
    print("=" * 64)
    print(" 2. LLM-generated FOL definition  ->  Z3 check")
    print("=" * 64)
    if not _OPENAI_AVAILABLE:
        print("  openai SDK not installed -- run: pip install openai")
        print("  (skipping the LLM section; the core check above is what matters)")
        print()
        return
    try:
        client = OpenAI(api_key="ollama", base_url=base_url)
        prompt = (
            f"Define the concept '{concept}' in first-order logic. "
            f"Give one line of the form:  forall x: {concept.split()[0]}(x) <-> [conditions]. "
            f"Use only predicates, and, or, not, forall, exists. No prose."
        )
        resp = client.chat.completions.create(
            model=model, temperature=0.0, max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )
        fol = (resp.choices[0].message.content or "").strip()
        print(f"  LLM ({model}) produced:\n    {fol[:200]}")
        print("  -> Phase 1 would now compile this to Z3 (see src/parsing/fol_parser.py)")
    except Exception as e:  # noqa: BLE001
        print(f"  Ollama not reachable ({e.__class__.__name__}). Start it with: ollama serve &")
        print("  Then: ollama pull llama3.2")
    print()

# ── 3. A small visual: how the contradiction collapses the model space ─────────
def demo_plot() -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not installed -- skipping plot (pip install matplotlib)")
        return
    import os
    here = os.path.dirname(os.path.abspath(__file__))

    labels = ["Coherent\n(Written ∧ Published)", "Contradiction\n(Written ∧ ¬Written)"]
    models_exist = [1, 0]  # sat -> a model exists; unsat -> none
    colors = ["#2e7d32", "#c62828"]

    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.bar(labels, models_exist, color=colors, width=0.6)
    ax.set_ylim(0, 1.25)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["unsat\n(no model)", "sat\n(model exists)"])
    ax.set_title("Z3 verdict for two FOL concept definitions", fontsize=10)
    for i, v in enumerate(models_exist):
        ax.text(i, v + 0.05, "sat" if v else "unsat", ha="center", fontweight="bold")
    fig.tight_layout()
    out = os.path.join(here, "tutorial_z3_verdicts.png")
    fig.savefig(out, dpi=120)
    print(f"  Plot saved to {out}")
    print()

if __name__ == "__main__":
    demo_handwritten()
    demo_llm()
    demo_plot()
    print("Done. See README.md to run the full Phase 1-3 pipeline over real ontologies.")
