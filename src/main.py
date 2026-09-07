"""JAMER — entry point. Run phases via Hydra config."""
from __future__ import annotations
import sys
from pathlib import Path
# Add project root so `from src.X import Y` works when run as `python src/main.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import hydra
from omegaconf import DictConfig
from dataclasses import dataclass, field

@dataclass
class LLMConfig:
    provider: str = "ollama"
    api_base: str = "http://localhost:11434/v1"
    model: str = "llama3.2:latest"
    temperature: float = 0.0
    max_tokens: int = 2048
    max_tokens_retry: int = 8192
    timeout_retry_s: float = 1800.0
    n_samples: int = 5
    timeout_s: float = 120.0  # HTTP timeout per LLM call; increase for ToT on CPU

@dataclass
class OntologyConfig:
    name: str = "book"
    path: str = "data/ontologies/"

@dataclass
class SamplingConfig:
    n_concepts: int = 50
    n_relations: int = 30
    seed: int = 42
    stratify_depth: bool = True

@dataclass
class PromptingConfig:
    strategy: str = "zero_shot"
    fol_format: str = "strict"

@dataclass
class SolverConfig:
    endpoint: str = "local"
    timeout_s: int = 10

@dataclass
class MetricsConfig:
    embed_model: str = "all-MiniLM-L6-v2"
    ged_method: str = "nx_optimize"

@dataclass
class OutputConfig:
    dir: str = "outputs/results"
    paper_ready: str = "outputs/paper_ready"

@dataclass
class Phase3Config:
    use_phase1_context: bool = True  # set False for no-context control

@dataclass
class JamerConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    ontology: OntologyConfig = field(default_factory=OntologyConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    prompting: PromptingConfig = field(default_factory=PromptingConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    phase3: Phase3Config = field(default_factory=Phase3Config)
    phase: int = 1          # 0=setup, 1=concepts, 2=edges, 3=reconstruct
    phase_2b: bool = False  # set True to run Phase 2b (subClassOf transitivity)

@hydra.main(config_path="../conf", config_name="jamer", version_base=None)
def main(cfg: DictConfig) -> None:
    from pathlib import Path
    import json, time

    out_dir = Path(cfg.output.dir) / cfg.ontology.name / cfg.llm.model.replace(":", "_") / cfg.prompting.strategy
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    phase_label = "2b" if cfg.phase_2b else str(cfg.phase)
    print(f"[JAMER] phase={phase_label}  ontology={cfg.ontology.name}  llm={cfg.llm.model}  strategy={cfg.prompting.strategy}")

    if cfg.phase_2b:
        from src.phase2b_subclassof import run_phase2b
        run_phase2b(cfg, out_dir)
    elif cfg.phase == 0:
        from src.ontology_loader import load_and_sample
        load_and_sample(cfg)
    elif cfg.phase == 1:
        from src.phase1_concepts import run_phase1
        run_phase1(cfg, out_dir)
    elif cfg.phase == 2:
        from src.phase2_edges import run_phase2
        run_phase2(cfg, out_dir)
    elif cfg.phase == 3:
        from src.phase3_reconstruct import run_phase3
        run_phase3(cfg, out_dir)
    else:
        raise ValueError(f"Unknown phase: {cfg.phase}")

    elapsed = time.time() - t0
    print(f"[JAMER] Done in {elapsed:.1f}s ({elapsed/60:.1f} min)")

if __name__ == "__main__":
    main()
