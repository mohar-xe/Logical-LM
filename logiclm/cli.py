"""Command-line interface for Logic-LM.

Primary entry point::

    logiclm run --dataset ProntoQA --split samples --llm mock
    logiclm run --dataset FOLIO --split dev --llm openai --model gpt-4o-mini
    logiclm run --dataset AR-LSAT --split dev --llm ollama --model qwen3:32b

Subcommands::

    logiclm download-data           # fetch + normalize all 5 datasets
    logiclm generate                # LLM -> logic programs (stage 1)
    logiclm infer                   # solver -> answers (stage 2)
    logiclm refine                  # self-refinement (stage 3)
    logiclm evaluate                # metrics over a results file
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

from . import __version__, DATASETS
from .datasets import load_dataset, download_all
from .evaluate import compute_metrics
from .llm.factory import build_llm_client
from .pipeline import LogicLMPipeline
from .schema import Example
from .solvers.registry import SolverRegistry


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="logiclm", description="Logic-LM from scratch")
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="full pipeline: generate -> infer -> (refine) -> evaluate")
    _add_common(run)
    run.add_argument("--max-refine-rounds", type=int, default=0,
                     help="self-refinement rounds (0 disables; FOLIO/AR-LSAT only)")
    run.add_argument("--out", default=None, help="where to write results JSON")

    gen = sub.add_parser("generate", help="LLM translates problems into logic programs")
    _add_common(gen)
    gen.add_argument("--out", default="outputs/logic_programs")

    inf = sub.add_parser("infer", help="run the symbolic solver on logic programs")
    _add_common(inf)
    inf.add_argument("--programs", default=None, help="path to generated programs JSON")
    inf.add_argument("--out", default="outputs/logic_inference")

    ref = sub.add_parser("refine", help="self-refine faulty logic programs")
    _add_common(ref)
    ref.add_argument("--programs", default=None)
    ref.add_argument("--max-rounds", type=int, default=3)
    ref.add_argument("--out", default="outputs/logic_programs/refined")

    ev = sub.add_parser("evaluate", help="compute metrics over a results file")
    ev.add_argument("results", help="path to results JSON (list of result dicts)")

    dl = sub.add_parser("download-data", help="fetch + normalize all 5 datasets")
    dl.add_argument("--data-dir", default="data")

    return p


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--dataset", required=True, choices=DATASETS)
    p.add_argument("--split", default="dev", choices=["dev", "test", "samples"])
    p.add_argument("--llm", default="mock", choices=["mock", "openai", "anthropic", "ollama"])
    p.add_argument("--model", default=None)
    p.add_argument("--api-key", default=None)
    p.add_argument("--base-url", default=None)
    p.add_argument("--backup", default="random", choices=["random", "LLM"])
    p.add_argument("--backup-llm-path", default=None)
    p.add_argument("--solver-timeout-ms", type=int, default=10_000)
    p.add_argument("--strict-unknown", action="store_true")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--seed", type=int, default=0)


def _run_cmd(args) -> None:
    random.seed(args.seed)
    examples = load_dataset(args.dataset, split=args.split, data_dir=args.data_dir)
    if not examples:
        print(f"no examples for {args.dataset}/{args.split}; run `logiclm download-data`")
        sys.exit(1)

    llm = build_llm_client(args.llm, model=args.model, api_key=args.api_key,
                           base_url=args.base_url)
    solver = SolverRegistry(timeout_ms=args.solver_timeout_ms,
                            strict_unknown=args.strict_unknown)
    pipe = LogicLMPipeline(
        args.dataset,
        llm=llm,
        solver=solver,
        backup_strategy=args.backup,
        backup_llm_path=args.backup_llm_path,
        seed=args.seed,
        max_refine_rounds=args.max_refine_rounds,
    )
    # Offline mock mode: examples that bundle a ready-to-run program are used
    # as-is instead of asking an empty mock to "generate" one.
    if args.llm == "mock" and all(e.programs for e in examples):
        results = pipe.infer(examples)
    else:
        results = pipe.run(examples)
    if args.out:
        LogicLMPipeline.save(results, args.out)
        print(f"saved {len(results)} results to {args.out}")
    m = compute_metrics(results)
    print(f"[{args.dataset}/{args.split}] {m}")


def _generate_cmd(args) -> None:
    examples = load_dataset(args.dataset, split=args.split, data_dir=args.data_dir)
    llm = build_llm_client(args.llm, model=args.model, api_key=args.api_key,
                           base_url=args.base_url)
    pipe = LogicLMPipeline(args.dataset, llm=llm,
                           solver=SolverRegistry(timeout_ms=args.solver_timeout_ms))
    # mock mode with bundled programs: use them as-is (no network/API key)
    if args.llm == "mock" and all(e.programs for e in examples):
        pass
    else:
        pipe.generate_programs(examples)
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"{args.dataset}_{args.split}_{llm.name()}.json")
    LogicLMPipeline.save([e.to_dict() for e in examples], path)
    print(f"generated {len(examples)} programs -> {path}")


def _infer_cmd(args) -> None:
    if args.programs:
        with open(args.programs, "r", encoding="utf-8") as f:
            examples = [Example.from_dict(d) for d in json.load(f)]
    else:
        examples = load_dataset(args.dataset, split=args.split, data_dir=args.data_dir)
    solver = SolverRegistry(timeout_ms=args.solver_timeout_ms,
                            strict_unknown=args.strict_unknown)
    pipe = LogicLMPipeline(args.dataset, llm=build_llm_client("mock"),
                           solver=solver, backup_strategy=args.backup,
                           backup_llm_path=args.backup_llm_path, seed=args.seed)
    results = pipe.infer(examples)
    LogicLMPipeline.save(results, args.out + f"/{args.dataset}.json")
    m = compute_metrics(results)
    print(f"[{args.dataset}/{args.split}] {m}")


def _refine_cmd(args) -> None:
    from .refine import refine_programs

    if args.programs:
        with open(args.programs, "r", encoding="utf-8") as f:
            examples = [Example.from_dict(d) for d in json.load(f)]
    else:
        examples = load_dataset(args.dataset, split=args.split, data_dir=args.data_dir)
    llm = build_llm_client(args.llm, model=args.model, api_key=args.api_key,
                           base_url=args.base_url)
    solver = SolverRegistry(timeout_ms=args.solver_timeout_ms,
                            strict_unknown=args.strict_unknown)
    report = refine_programs(examples, args.dataset, llm, solver,
                             max_rounds=args.max_rounds)
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"{args.dataset}_{args.split}_refined.json")
    LogicLMPipeline.save([e.to_dict() for e in examples], path)
    print(f"refined {len(report.revised)} programs over {report.rounds_used} round(s) -> {path}")


def _evaluate_cmd(args) -> None:
    with open(args.results, "r", encoding="utf-8") as f:
        results = json.load(f)
    m = compute_metrics(results)
    print(m)


def _download_cmd(args) -> None:
    counts = download_all(data_dir=args.data_dir)
    for name, count in counts.items():
        print(f"  {name}: {count} examples")


def main() -> None:
    args = _parser().parse_args()
    dispatch = {
        "run": _run_cmd,
        "generate": _generate_cmd,
        "infer": _infer_cmd,
        "refine": _refine_cmd,
        "evaluate": _evaluate_cmd,
        "download-data": _download_cmd,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
