"""Command line interface: kyc-graph generate | resolve | assess | run-all | review."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .evaluate import risk_metrics, screening_metrics, typology_metrics
from .model import TYPOLOGIES, Dataset
from .pipeline import analyse, engine_for
from .report import assessment_report, explain_pair, resolution_report, run_all_report
from .review import DECISIONS, DecisionLog, build_queue
from .synth import generate

FLOORS = {"er_f1": 0.95, "screening_precision": 0.9, "screening_recall": 0.9,
          "typology_precision": 0.6, "typology_recall": 0.66, "auc_fused": 0.9}


def _load(data: Path, seed: int) -> Dataset:
    if (data / "dataset.json").exists():
        return Dataset.load(data)
    ds = generate(seed)
    ds.save(data)
    print(f"(no dataset in {data}; generated seed {seed})", file=sys.stderr)
    return ds


def _pipeline(args: argparse.Namespace):
    ds = _load(args.data, args.seed)
    engine = engine_for(ds.seed)
    analysis = analyse(ds, engine.models, engine.champions)
    return ds, engine, analysis


def cmd_generate(args: argparse.Namespace) -> int:
    ds = generate(args.seed)
    ds.save(args.data)
    by_source: dict[str, int] = {}
    for r in [*ds.companies, *ds.persons]:
        by_source[r.source] = by_source.get(r.source, 0) + 1
    print(f"Wrote {args.data / 'dataset.json'} (seed {args.seed})")
    print(f"  company records {len(ds.companies)}, person records {len(ds.persons)}, relations {len(ds.relations)}")
    print(f"  records per source {dict(sorted(by_source.items()))}")
    print(f"  list entries {len(ds.watchlist)}, media items {len(ds.media)}, tender awards {len(ds.tenders)}")
    print("  injected typologies: " + ", ".join(f"{t}={len(ds.truth.typologies[t])}" for t in TYPOLOGIES))
    return 0


def cmd_resolve(args: argparse.Namespace) -> int:
    ds = _load(args.data, args.seed)
    engine = engine_for(ds.seed)
    analysis = analyse(ds, engine.models, engine.champions)
    print(resolution_report(analysis.res, engine.champions))
    if args.explain:
        print()
        print(explain_pair(analysis.res, *args.explain))
    return 0


def cmd_assess(args: argparse.Namespace) -> int:
    _ds, engine, analysis = _pipeline(args)
    if args.company not in analysis.res.entities:
        print(f"unknown entity {args.company}; companies look like VR-000123 or X-A-A00012", file=sys.stderr)
        return 2
    assessments = engine.assess_all(analysis)
    log = DecisionLog(args.data / "decisions.jsonl")
    last = log.latest().get(f"CASE-{args.company}")
    status = f"{last['decision']} by {last['analyst']} at {last['ts']}" if last else ""
    text = assessment_report(assessments[args.company], analysis, engine, status)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
        print(f"Wrote {args.out}")
    else:
        print(text)
    return 0


def cmd_run_all(args: argparse.Namespace) -> int:
    ds = generate(args.seed)
    ds.save(args.data)
    engine = engine_for(args.seed)
    analysis = analyse(ds, engine.models, engine.champions)
    assessments = engine.assess_all(analysis)
    typ, scr = typology_metrics(analysis), screening_metrics(analysis)
    risk = risk_metrics(analysis, assessments)
    print(run_all_report(analysis, engine, typ, scr, risk, assessments))
    failures = [f"ER F1 {k} {m['f1']:.3f}" for k, m in analysis.res.metrics.items() if m["f1"] < FLOORS["er_f1"]]
    failures += [f"screening {m} {scr[m]:.3f}" for m in ("precision", "recall") if scr[m] < FLOORS[f"screening_{m}"]]
    failures += [f"{t} {m} {typ[t][m]:.3f}" for t in TYPOLOGIES for m in ("precision", "recall")
                 if typ[t][m] < FLOORS[f"typology_{m}"]]
    if risk["auc_fused"] < FLOORS["auc_fused"]:
        failures.append(f"AUC {risk['auc_fused']:.3f}")
    print()
    print("Floors: " + ("all met" if not failures else "FAILED: " + "; ".join(failures)))
    return 1 if failures and not args.no_check else 0


def cmd_review(args: argparse.Namespace) -> int:
    log = DecisionLog(args.data / "decisions.jsonl")
    if args.action == "verify":
        ok, msg = log.verify()
        print(f"{len(log.entries())} entries, chain {msg}")
        return 0 if ok else 1
    _ds, engine, analysis = _pipeline(args)
    assessments = engine.assess_all(analysis)
    if args.action == "queue":
        queue = build_queue(assessments, log)
        print(f"{len(queue)} open case(s)")
        for c in queue[: args.limit]:
            print(f"  {c.case_id:<22} {c.kind:<7} risk {c.risk:.2f}  conf {c.confidence:.2f}  {c.name[:38]:<38} {'; '.join(c.reasons)}")
        return 0
    case = f"CASE-{args.subject}"
    a = assessments.get(args.subject)
    if a is None:
        print(f"unknown subject {args.subject}", file=sys.stderr)
        return 2
    entry = log.append(case, args.decision, args.analyst, args.note, a.risk, a.confidence)
    print(f"Logged #{entry['seq']} {case} {args.decision} by {args.analyst} (hash {entry['hash'][:12]})")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kyc-graph", description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"), help="dataset directory (default ./data)")
    parser.add_argument("--seed", type=int, default=7)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("generate", help="generate the synthetic world and sources").set_defaults(func=cmd_generate)
    p = sub.add_parser("resolve", help="entity resolution metrics and model comparison")
    p.add_argument("--explain", nargs=2, metavar=("KEY_A", "KEY_B"), help="explain one record pair")
    p.set_defaults(func=cmd_resolve)
    p = sub.add_parser("assess", help="Markdown risk report for one company")
    p.add_argument("company")
    p.add_argument("--out", type=Path)
    p.set_defaults(func=cmd_assess)
    p = sub.add_parser("run-all", help="full pipeline with metrics against ground truth")
    p.add_argument("--no-check", action="store_true", help="do not fail when a metric is below its floor")
    p.set_defaults(func=cmd_run_all)
    p = sub.add_parser("review", help="review queue and append-only decision log")
    p.add_argument("action", choices=["queue", "decide", "verify"])
    p.add_argument("subject", nargs="?")
    p.add_argument("--decision", choices=DECISIONS, default="escalate")
    p.add_argument("--analyst", default="")
    p.add_argument("--note", default="")
    p.add_argument("--limit", type=int, default=20)
    p.set_defaults(func=cmd_review)
    args = parser.parse_args(argv)
    if args.cmd == "review" and args.action == "decide" and (not args.subject or not args.analyst):
        parser.error("review decide needs SUBJECT and --analyst")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
