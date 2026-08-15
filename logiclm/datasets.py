"""Fetch + normalize the five Logic-LM datasets to the common ``Example``
schema (id, context, question, options, answer).

All fetches are network-gated: call ``load_dataset(name, split)`` and it will
download into ``<data_dir>/raw`` on first use and cache on disk.  The
``samples`` split loads tiny bundled examples (``data/samples/*.json``) that
ship with the repo so tests and demos run fully offline.
"""

from __future__ import annotations

import json
import os

from .schema import Example

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
SAMPLES_DIR = os.path.join(DATA_DIR, "samples")
RAW_DIR = os.path.join(DATA_DIR, "raw")

_SOURCES = {
    # ProntoQA ships its generated data as one big zip (no plain 5hop file).
    "ProntoQA": "https://raw.githubusercontent.com/asaparov/prontoqa/main/generated_ood_data.zip",
    "FOLIO": "https://raw.githubusercontent.com/Yale-LILY/FOLIO/main/data/v0.0/folio-validation.jsonl",
    "LogicalDeduction": "https://raw.githubusercontent.com/google/BIG-bench/main/bigbench/benchmark_tasks/logical_deduction/task.json",
    "AR-LSAT": "https://raw.githubusercontent.com/zhongwanjun/AR-LSAT/main/complete_lsat_data/test_ar.json",
    # ProofWriter needs HF dataset-server queries; handled separately.
}

# number of examples sampled from each dataset for a "small" dev run
_SMALL = {
    "ProntoQA": 20,
    "ProofWriter": 20,
    "FOLIO": 20,
    "LogicalDeduction": 20,
    "AR-LSAT": 20,
}


def load_dataset(name: str, split: str = "dev", data_dir: str = DATA_DIR) -> list[Example]:
    if name not in _SOURCES and name != "ProofWriter":
        raise ValueError(f"unknown dataset {name!r}")

    if split == "samples":
        return _load_samples(name, data_dir)
    if split not in {"dev", "test"}:
        raise ValueError(f"unknown split {split!r}")

    if name == "ProofWriter":
        return _load_proofwriter(data_dir)
    return _load_json(name, data_dir)


def _load_samples(name: str, data_dir: str) -> list[Example]:
    path = os.path.join(data_dir, "samples", f"{name}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"no bundled samples for {name} at {path}. Run "
            "`logiclm download-data` or add your own file."
        )
    with open(path, "r", encoding="utf-8") as f:
        return [Example.from_dict(d) for d in json.load(f)]


def _cached_path(name: str, data_dir: str) -> str:
    os.makedirs(os.path.join(data_dir, "raw"), exist_ok=True)
    return os.path.join(data_dir, "raw", f"{name}.json")


def _load_json(name: str, data_dir: str) -> list[Example]:
    cache = _cached_path(name, data_dir)
    if not os.path.exists(cache):
        url = _SOURCES[name]
        import urllib.request

        print(f"downloading {name} from {url}")
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        if name == "ProntoQA":
            return _download_prontoqa(url, cache)
        if name == "LogicalDeduction":
            return _download_logicaldeduction(data_dir)
        urllib.request.urlretrieve(url, cache)
    with open(cache, "r", encoding="utf-8") as f:
        if name == "FOLIO":
            raw = f.read()  # JSONL
        else:
            raw = json.load(f)
    return _normalize(name, raw)


def _download_logicaldeduction(data_dir: str) -> list[Example]:
    """LogicalDeduction data lives in three BIG-bench sub-task files."""
    import urllib.request

    cache = _cached_path("LogicalDeduction", data_dir)
    base = (
        "https://raw.githubusercontent.com/google/BIG-bench/main/"
        "bigbench/benchmark_tasks/logical_deduction"
    )
    subtasks: list[dict] = []
    for sub in ("three_objects", "five_objects", "seven_objects"):
        url = f"{base}/{sub}/task.json"
        print(f"  fetching {sub} ...")
        with urllib.request.urlopen(url) as resp:
            subtasks.append(json.load(resp))
    examples = _normalize("LogicalDeduction", subtasks)
    with open(cache, "w", encoding="utf-8") as f:
        json.dump([e.to_dict() for e in examples], f, indent=2, ensure_ascii=False)
    return examples


def _download_prontoqa(url: str, cache: str) -> list[Example]:
    """ProntoQA's data is a zip of many generated JSON files.  We extract the
    ``*ProofsOnly*`` files (deductive-reasoning examples over fictional
    characters), merge them, and normalize each test example.  Every query in
    ProofsOnly data is provable, so the ground truth is always ``True`` (A)."""
    import io
    import urllib.request
    import zipfile

    with urllib.request.urlopen(url) as resp:
        blob = resp.read()
    examples: list[Example] = []
    seen_ids: set[str] = set()
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        for member in zf.namelist():
            if not member.endswith(".json") or "ProofsOnly" not in member:
                continue
            data = json.loads(zf.read(member).decode("utf-8"))
            for key, entry in data.items():
                test = entry.get("test_example") if isinstance(entry, dict) else None
                if not isinstance(test, dict):
                    continue
                q = test.get("question", "")
                query = test.get("query", "")
                if not q or not query:
                    continue
                ex_id = f"ProntoQA_{len(examples)}"
                # query is "Prove: <stmt>" -> the answer is True (A)
                examples.append(Example(
                    id=ex_id,
                    context=q,
                    question=query,
                    answer="A",
                    options=["A) True", "B) False"],
                ))
                seen_ids.add(ex_id)

    with open(cache, "w", encoding="utf-8") as f:
        json.dump([e.to_dict() for e in examples], f, indent=2, ensure_ascii=False)
    return examples


def _load_proofwriter(data_dir: str) -> list[Example]:
    """Fetch ProofWriter from the public ``tasksource/proofwriter`` HF mirror
    (the original ``allenai/proofwriter`` is gated).  The test-split parquet is
    downloaded directly from the HF Hub (no dataset-server rate limits).  Rows
    carry ``theory`` (context), ``question``, ``answer`` (True/False/Unknown)
    and a per-row ``config`` like ``depth-5``; the id encodes the
    open-world/closed-world setting (OWA).  We take depth-5 OWA rows and cap
    the set at ``_SMALL`` examples.
    """
    cache = os.path.join(data_dir, "raw", "ProofWriter.json")
    if not os.path.exists(cache):
        import urllib.request

        url = (
            "https://huggingface.co/datasets/tasksource/proofwriter/resolve/main/"
            "data/test-00000-of-00001-3e27b013c60e12d8.parquet"
        )
        print("downloading ProofWriter (tasksource HF parquet) ...")
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        parquet_path = cache + ".parquet"
        urllib.request.urlretrieve(url, parquet_path)

        import pyarrow.parquet as pq

        table = pq.read_table(parquet_path)
        examples: list[Example] = []
        ids = table.column("id").to_pylist()
        theory = table.column("theory").to_pylist()
        question = table.column("question").to_pylist()
        answer = table.column("answer").to_pylist()
        config = table.column("config").to_pylist()
        for row_id, ctx, q, ans, cfg in zip(ids, theory, question, answer, config):
            if cfg != "depth-5" or "OWA" not in str(row_id):
                continue
            letter = {"True": "A", "False": "B", "Unknown": "C"}.get(ans, "")
            examples.append(Example(
                id=str(row_id),
                context=ctx,
                question=q,
                answer=letter,
                options=["A) True", "B) False", "C) Unknown"],
            ))
            if len(examples) >= _SMALL["ProofWriter"]:
                break
        with open(cache, "w", encoding="utf-8") as f:
            json.dump([e.to_dict() for e in examples], f, indent=2, ensure_ascii=False)
        return examples
    with open(cache, "r", encoding="utf-8") as f:
        return [Example.from_dict(d) for d in json.load(f)]


def _normalize(name: str, raw) -> list[Example]:
    if name == "ProntoQA":
        # ProntoQA is downloaded+normalized via the zip path into the cache;
        # the cache holds already-normalized Example dicts.
        return [Example.from_dict(d) for d in raw]

    if name == "FOLIO":
        # JSONL: {premises, conclusion, label, premises-FOL, conclusion-FOL, ...}
        # The LLM prompt uses the natural-language premises as the problem and
        # the conclusion as the question; ground truth is the label.
        examples = []
        if isinstance(raw, str):
            raw = [json.loads(l) for l in raw.splitlines()]
        for i, d in enumerate(raw):
            label = d.get("label", "")
            answer = {"True": "A", "False": "B", "Uncertain": "C"}.get(label, "")
            premises = d.get("premises", [])
            conclusion = d.get("conclusion", "")
            context = " ".join(premises)
            question = (
                "Based on the above information, is the following statement "
                f"true, false, or uncertain? {conclusion}"
            )
            examples.append(Example(
                id=d.get("id") or f"FOLIO_{i}",
                context=context,
                question=question,
                answer=answer,
                options=["A) True", "B) False", "C) Uncertain"],
            ))
        return examples

    if name == "LogicalDeduction":
        # BIG-bench splits the data across three_objects/five_objects/
        # seven_objects sub-task files, each with {examples: [{input,
        # target_scores}]}.  ``raw`` here is a list of sub-task dicts.
        examples = []
        for sub in raw:
            for d in sub.get("examples", []):
                input_text = d.get("input", "")
                target = d.get("target_scores", {})
                # correct answer is the option with the highest target score
                best_text = max(target.items(), key=lambda kv: kv[1])[0] if target else ""
                options = list(target.keys())
                # map the winning option's index to a letter (A/B/C/...)
                try:
                    letter = chr(65 + options.index(best_text))
                except ValueError:
                    letter = ""
                examples.append(Example(
                    id=f"LogicalDeduction_{len(examples)}",
                    context=input_text,
                    question="Which of the following is true?",
                    answer=letter,
                    options=options,
                ))
        return examples

    if name == "AR-LSAT":
        # list of {context, question, answers: [str], label: int, id_string}
        # ``label`` is a 0-based index into ``answers``; option letter = A+i.
        examples = []
        for i, q in enumerate(raw):
            answers = q.get("answers", []) or []
            options = [f"({chr(65 + j)}) {a}" for j, a in enumerate(answers)]
            label = q.get("label", 0)
            answer = chr(65 + label) if 0 <= label < len(options) else ""
            examples.append(Example(
                id=q.get("id_string") or f"AR-LSAT_{i}",
                context=q.get("context", ""),
                question=q.get("question", ""),
                answer=answer,
                options=options,
            ))
        return examples

    raise ValueError(f"unsupported dataset {name!r}")


def download_all(data_dir: str = DATA_DIR) -> dict[str, int]:
    """Download + normalize all five datasets; return counts per dataset."""
    counts = {}
    for name in _SOURCES:
        if name == "ProofWriter":
            continue
        examples = _load_json(name, data_dir)
        counts[name] = len(examples)
    counts["ProofWriter"] = len(_load_proofwriter(data_dir))
    return counts
