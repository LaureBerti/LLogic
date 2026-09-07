# LLM calls

One row per call to a language model: which concept was asked of which model under which
prompting strategy, and the reply as it was returned.

```
llm_calls_seed42.csv      9,584 calls
llm_calls_seed123.csv     6,346
llm_calls_seed456.csv     2,944
llm_calls_seed789.csv     2,944
llm_calls_seed1011.csv    2,944
                         24,762 total
MANIFEST.json             per-file counts and reply-character totals
```

## Columns

| column | meaning |
|---|---|
| `seed` | which stratified concept sample the call belongs to |
| `ontology` | the source ontology the concept was drawn from |
| `llm_model` | the served model identifier |
| `strategy` | `zero_shot`, `cot`, `tot` or `self_consistency` |
| `concept_iri`, `concept_label` | the concept the model was asked to define |
| `response_text` | **the model's reply, verbatim** |
| `source` | which run the reply came from — see below |

The prompt is not stored per row because it is a deterministic function of
`concept_label` and `strategy`. `src/prompting/prompter.py` reconstructs it exactly, and
the four templates are printed verbatim in the paper's appendix.

## Provenance and completeness

Two runs of the seed-42 grid exist. The primary run stored replies truncated at 300
characters for the study ontologies; a second run stored them in full. Where both cover the
same call the longer reply is used, and `source` records which run it came from:
**3,658 replies are taken from the full-text run.**

**92 replies (0.4% of 24,762) are still truncated at exactly 300 characters** — calls the
full-text run did not cover. They are kept rather than dropped, and are identifiable as the
rows whose `response_text` is exactly 300 characters long. A further 2,392 replies are
shorter than 300 characters because the model genuinely said little; those are complete.

**190 replies are empty.** These are calls where a reasoning model spent its whole token
budget on internal reasoning and emitted no answer. They are kept because removing them
would misrepresent the response rate.

## Regenerating

```bash
python src/analysis/inventory.py   # what result cells exist
```

The calls themselves are produced by the pipeline; see the top-level README.
