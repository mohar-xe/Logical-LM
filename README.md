# Logic-LM (from scratch)

A from-scratch reimplementation of **Logic-LM** (Findings of EMNLP 2023,
[arXiv:2305.12295](https://arxiv.org/abs/2305.12295)), the framework that pairs
LLMs with **deterministic symbolic solvers** for faithful logical reasoning.

```
NL problem ──LLM──▶ logic program (text IR) ──solver──▶ symbolic answer ──map──▶ A/B/C/D/E
                          ▲                                                    │
                          └──── self-refinement (≤3 rounds) ── error msg ──────┘
```

The LLM translates each natural-language problem into a symbolic *logic
program*; a deterministic solver executes it; a self-refinement loop uses the
solver's error messages to revise faulty programs.  When a program cannot be
parsed or executed, a backup strategy (random, or precomputed chain-of-thought
predictions) provides the answer.

This is a clean reimplementation, **not** a fork of
[teacherpeterpan/logic-llm](https://github.com/teacherpeterpan/logic-llm): no
Prover9 binary, no Pyke rule engine, no `exec` of LLM-produced text, no 2023-era
OpenAI API.

## Datasets & solvers

| Dataset | IR sections | Solver (from scratch) | Answer letters |
|---|---|---|---|
| ProntoQA | Predicates / Facts / Rules / Query | forward-chaining rule engine | A / B |
| ProofWriter | Predicates / Facts / Rules / Query | same engine (open-world) | A / B / C |
| FOLIO | Predicates / Premises / Conclusion | FOL parser → Z3 prover | A / B / C |
| LogicalDeduction | Domain / Variables / Constraints / Query | CSP backtracking solver | A…G |
| AR-LSAT | `# Declarations / Constraints / Options` | Z3 DSL compiler | A…E |

## Install

```bash
uv sync                      # creates .venv, installs deps
uv run logiclm --version
```

Requires Python ≥ 3.13.  Core dependency is `z3-solver`; optional LLM backends
need `uv sync --extra llm` (`openai`, `anthropic`, `httpx`).

## Try it offline (no API key)

Bundled mini-datasets ship in `data/samples/` with ready-to-run programs:

```bash
uv run logiclm run --dataset ProntoQA       --split samples --llm mock
uv run logiclm run --dataset ProofWriter    --split samples --llm mock
uv run logiclm run --dataset FOLIO          --split samples --llm mock
uv run logiclm run --dataset LogicalDeduction --split samples --llm mock
uv run logiclm run --dataset AR-LSAT        --split samples --llm mock
```

Each should print `accuracy=1.000` — the bundled programs are the ground truth
for the sample problems.

## Run on real data

```bash
uv sync --extra llm
uv run logiclm download-data                 # fetch + normalize all 5 datasets

# with an OpenAI key
uv run logiclm run --dataset FOLIO --split dev --llm openai --model gpt-4o-mini \
    --backup random

# with a local Ollama server
uv run logiclm run --dataset AR-LSAT --split dev --llm ollama --model qwen3:32b

# self-refinement (paper ships templates for FOLIO and AR-LSAT)
uv run logiclm run --dataset FOLIO --split dev --llm openai --model gpt-4o-mini \
    --max-refine-rounds 3
```

The `run` command composes all stages and prints three numbers:

```
[dataset/split] accuracy=0.612 (125/204)  executable=0.730 (149/204)  exec-acc=0.624 (93/149)
```

* `accuracy` — overall accuracy (fallbacks included)
* `executable` — fraction of programs the solver could execute
* `exec-acc` — accuracy among executable programs (the paper's headline number)

### Subcommands

```bash
logiclm download-data                         # fetch datasets into data/raw/
logiclm generate --dataset FOLIO --llm openai --out outputs/logic_programs
logiclm infer   --dataset FOLIO --programs outputs/logic_programs/...json
logiclm refine  --dataset FOLIO --programs outputs/logic_programs/...json
logiclm evaluate outputs/logic_inference/FOLIO.json
```

### CLI options

| Flag | Meaning |
|---|---|
| `--llm {mock,openai,anthropic,ollama}` | LLM backend (mock runs offline) |
| `--model` | model name (e.g. `gpt-4o-mini`, `claude-sonnet-4-5`, `qwen3:32b`) |
| `--api-key` / `--base-url` | credentials / Ollama URL (defaults to env vars) |
| `--backup {random,LLM}` | fallback on solver failure |
| `--backup-llm-path` | results file for the `LLM` backup strategy |
| `--solver-timeout-ms` | per-solver timeout (default 10000) |
| `--strict-unknown` | treat Z3 `unknown` (hard formula) as an execution error |
| `--max-refine-rounds N` | self-refinement rounds (0 disables) |
| `--seed` | RNG seed for the random backup |
| `--split {dev,test,samples}` | which split to run |

## Architecture

```
logiclm/
├── cli.py            # `logiclm run` + subcommands
├── datasets.py       # fetch + normalize the 5 datasets (network-gated)
├── schema.py         # Example dataclass (id, context, question, options, answer, programs)
├── prompts.py        # few-shot template loading + filling
├── llm/              # LLMClient protocol + openai/anthropic/ollama/mock adapters
├── solvers/
│   ├── datalog.py    # forward-chaining rule engine (ProntoQA / ProofWriter)
│   ├── csp.py        # CSP backtracking (LogicalDeduction)
│   ├── fol/          # FOL parser + Z3 compiler/prover (FOLIO)
│   ├── z3dsl/        # AR-LSAT DSL parser/compiler/solver
│   └── registry.py   # per-dataset dispatch + structured errors
├── pipeline.py       # generate → infer → (refine) → evaluate
├── refine.py         # the self-refinement loop contract
├── backup.py         # random / LLM backup strategies
└── evaluate.py       # accuracy / executable-rate / executable-accuracy
```

## Design notes (things this implementation gets right)

* **FOLIO semantics** — prove the conclusion → `True`; else prove its negation
  → `False`; else `Unknown`, matching Prover9's three-way split.  Each example
  gets a *fresh* uninterpreted sort, which is sound because FOLIO conclusions
  never quantify over unnamed objects.
* **Z3 `unknown` is not `Unknown`** — a solver timeout/hard-formula result must
  never be silently reported as a clean "unknown" verdict (that would fabricate
  the correct-looking answer).  `--strict-unknown` escalates it to an execution
  error for self-refinement.
* **ProofWriter open-world** — the forward-chaining engine uses data-level
  negation: a `False`-valued premise matches only atoms literally present, so
  non-monotonic rules like `Furry($x, False) >>> Nice($x, True)` behave
  correctly, and "neither derivable" maps to answer `C`.
* **No `exec` of LLM output** — every solver compiles the program into in-memory
  objects (AST → Z3 / backtracking / fixpoint), never into executed Python.
  Errors are structured and safe to show an LLM for self-refinement.
* **Unicode-tolerant parsing** — `→ ⟹ ⇒ ->` all parse as implication; `∧ ⋀`,
  `¬ ~`, smart quotes and non-breaking spaces are canonicalized before the
  lexers run.

## Tests

```bash
uv run pytest                       # full offline suite
uv run pytest -m network            # also fetch real datasets (needs network)
```

The suite is 100% offline-capable: solvers are verified against the reference
repo's own golden programs, and the pipeline is exercised end-to-end on the
bundled samples with a deterministic mock LLM.

## Reproducing the paper's results

The framework reproduces the paper's pipeline (program generation, symbolic
inference with random/CoT backup, self-refinement, and the three evaluation
metrics).  Exact numbers differ because the datasets here are drawn from their
current public sources (the reference repo bundles its own preprocessed data,
which is not redistributed).
