# vmas_local_eval.py
# -*- coding: utf-8 -*-
"""
Pure local (no API server) Visual Multi-Agent chain system runner:
Observer -> Reasoner -> Verifier (optional retry)

Updates:
- Pass gt into chain.run_one(..., gt=gt) so Verifier runs in JUDGE mode for benchmark eval.
- Add retry_accept_policy switch (always/never/if_better) to control rollback/overwrite.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import yaml
import jsonlines
from dotenv import load_dotenv
from tqdm import tqdm

from common.logging_utils import (
    setup_run_logging,
    log_run_config,
    log_example_start,
    log_example_result,
    log_example_error,
    log_run_summary,
)
from tools.dataset_utils import (
    ensure_dir,
    set_hf_endpoint,
    prepare_mmbench,
    prepare_mme,
    load_local_dataset,
)
from tools.resume_utils import (
    resolve_resume_dir,
    load_resume_state,
    save_resume_state,
    prepare_examples,
    filter_pending,
    ResumeState,
)
from local_mm_backend import LocalBackendConfig, TransformersQwenVLBackend
from chains.visual_chain import (
    VisualChainSystem,
    ChainConfig,
    score_sample,
    choice_extractor_rule_based,
    choice_extractor_local_llm,
)

# ============================================================
# Config helpers
# ============================================================

def load_config_file(config_path: Optional[str]) -> Dict[str, Any]:
    if not config_path:
        return {}
    p = Path(config_path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        obj = yaml.safe_load(f) or {}
    if not isinstance(obj, dict):
        raise ValueError("Config YAML must be a mapping (dict).")
    obj["_config_path"] = str(p)
    return obj


def deep_get(d: Dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = d
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def apply_override(base: Dict[str, Any], path: str, value: Any) -> None:
    keys = path.split(".")
    cur = base
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value


def merge_cli_into_config(args: argparse.Namespace, cfg: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(cfg)

    # dataset
    if getattr(args, "out_root", None) is not None:
        apply_override(out, "dataset.out_root", args.out_root)
    if getattr(args, "hf_endpoint", None) is not None:
        apply_override(out, "dataset.hf_endpoint", args.hf_endpoint)
    if getattr(args, "dataset_id", None) is not None:
        apply_override(out, "dataset.dataset_id", args.dataset_id)
    if getattr(args, "split", None) is not None:
        apply_override(out, "dataset.split", args.split)
    if getattr(args, "limit", None) is not None:
        apply_override(out, "dataset.limit", args.limit)

    # model
    if getattr(args, "model", None) is not None:
        apply_override(out, "model.name", args.model)
    if getattr(args, "device_map", None) is not None:
        apply_override(out, "model.device_map", args.device_map)
    if getattr(args, "torch_dtype", None) is not None:
        apply_override(out, "model.torch_dtype", args.torch_dtype)
    if getattr(args, "max_image_pixels", None) is not None:
        apply_override(out, "model.max_image_pixels", args.max_image_pixels)
    if getattr(args, "use_fast_processor", None) is not None:
        ufp = parse_boolish(args.use_fast_processor)
        apply_override(out, "model.use_fast_processor", ufp)

    # generation
    if getattr(args, "max_tokens", None) is not None:
        apply_override(out, "generation.max_tokens", args.max_tokens)
    if getattr(args, "temperature", None) is not None:
        apply_override(out, "generation.temperature", args.temperature)
    if getattr(args, "no_retry", None) is not None:
        apply_override(out, "generation.no_retry", bool(args.no_retry))

    # NEW: retry accept policy
    if getattr(args, "retry_accept_policy", None) is not None:
        apply_override(out, "generation.retry_accept_policy", args.retry_accept_policy)

    # mmbench
    if getattr(args, "no_choice_extractor", None) is not None:
        apply_override(out, "mmbench.use_choice_extractor", not bool(args.no_choice_extractor))

    # output
    if getattr(args, "run_name", None) is not None:
        apply_override(out, "output.run_name", args.run_name)
    if getattr(args, "output_dir_prefix", None) is not None:
        apply_override(out, "output.dir_prefix", args.output_dir_prefix)

    return out


def parse_boolish(v: Any) -> Optional[bool]:
    if v is None:
        return None
    s = str(v).strip().lower()
    if s in ("true", "1", "yes", "y", "on"):
        return True
    if s in ("false", "0", "no", "n", "off"):
        return False
    if s in ("none", "null", ""):
        return None
    return None


def generate_output_dir(prefix: str | Path, name: str) -> Path:
    prefix_path = Path(prefix)
    prefix_path.mkdir(parents=True, exist_ok=True)

    base = prefix_path / name
    if base.exists() and not base.is_dir():
        raise FileExistsError(f"Output path exists as file: {base}")

    if not base.exists():
        base.mkdir(parents=True, exist_ok=True)
        return base

    suffix = 2
    while True:
        cand = prefix_path / f"{name}{suffix}"
        if cand.exists() and not cand.is_dir():
            raise FileExistsError(f"Output path exists as file: {cand}")
        if not cand.exists():
            cand.mkdir(parents=True, exist_ok=True)
            return cand
        suffix += 1


# ============================================================
# Eval / Analyze
# ============================================================

def analyze_predictions(pred_jsonl: str | Path) -> Dict[str, Any]:
    from collections import Counter

    pred_jsonl = Path(pred_jsonl)
    total = 0
    incorrect = 0
    by_err = Counter()
    by_verdict = Counter()
    retry_suggest = 0
    choice_map_fail = 0

    with jsonlines.open(pred_jsonl, "r") as r:
        for ex in r:
            total += 1
            if ex.get("choice_mapping_failed"):
                choice_map_fail += 1
            ver = ex.get("verifier", {}) or {}
            by_verdict[ver.get("verdict", "NA")] += 1
            if not ex.get("correct", False):
                incorrect += 1
                by_err[ver.get("error_type", "unknown")] += 1
            if ver.get("should_retry") is True:
                retry_suggest += 1

    summary = {
        "total": total,
        "incorrect": incorrect,
        "incorrect_rate": incorrect / max(total, 1),
        "verifier_verdicts": dict(by_verdict),
        "error_type_among_incorrect": dict(by_err),
        "verifier_suggested_retry": retry_suggest,
        "verifier_suggested_retry_rate": retry_suggest / max(total, 1),
        "choice_mapping_failures": choice_map_fail,
        "choice_mapping_failure_rate": choice_map_fail / max(total, 1),
    }
    return summary


def run_eval(cmd: str, cfg: Dict[str, Any], resume_dir: Optional[Path]) -> Dict[str, Any]:
    run_name = deep_get(cfg, "output.run_name")
    if not run_name:
        model_name = deep_get(cfg, "model.name", "model")
        short = str(model_name).split("/")[-1].replace(":", "_")
        run_name = f"{cmd}_{short}"

    if resume_dir:
        output_dir = resume_dir
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        dir_prefix = deep_get(cfg, "output.dir_prefix", "./output")
        output_dir = generate_output_dir(dir_prefix, run_name)

    logger = setup_run_logging(output_dir, name="vmas")

    resolved_cfg_path = output_dir / "resolved_config.yaml"
    resolved_cfg_path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")

    log_run_config(logger, cfg, deep_get(cfg, "_config_path"))

    ds_out_root = Path(deep_get(cfg, "dataset.out_root", "./datasets_out"))
    hf_endpoint = deep_get(cfg, "dataset.hf_endpoint")
    set_hf_endpoint(hf_endpoint)

    dataset_id = deep_get(cfg, "dataset.dataset_id")
    split = deep_get(cfg, "dataset.split")
    limit = deep_get(cfg, "dataset.limit")

    if cmd == "eval-mmbench":
        dataset_id = dataset_id or "HuggingFaceM4/MMBench_dev"
        split = split or "train"
        data_jsonl, _ = prepare_mmbench(ds_out_root, dataset_id=dataset_id, split=split, limit=limit)
        use_choice_extractor = bool(deep_get(cfg, "mmbench.use_choice_extractor", True))
    elif cmd == "eval-mme":
        dataset_id = dataset_id or "lmms-lab/MME"
        split = split or "test"
        data_jsonl, _ = prepare_mme(ds_out_root, dataset_id=dataset_id, split=split, limit=limit)
        use_choice_extractor = False
    else:
        raise ValueError(f"Unsupported eval cmd: {cmd}")

    logger.info("Dataset prepared: %s", data_jsonl)
    examples = load_local_dataset(data_jsonl)
    prepared = prepare_examples(examples)

    rs: Optional[ResumeState] = None
    if resume_dir:
        rs = load_resume_state(output_dir)
        pending = filter_pending(prepared, rs)
        logger.info("Resume enabled: pending=%d completed=%d invalid=%d",
                    len(pending), rs.completed_count, rs.invalid_count)
    else:
        pending = prepared
        logger.info("Resume disabled: pending=%d", len(pending))

    model_name = deep_get(cfg, "model.name")
    if not model_name:
        raise ValueError("Missing required config: model.name (or pass --model).")

    backend = TransformersQwenVLBackend(
        LocalBackendConfig(
            model_name_or_path=model_name,
            device_map=deep_get(cfg, "model.device_map", "auto"),
            torch_dtype=deep_get(cfg, "model.torch_dtype", "auto"),
            max_image_pixels=deep_get(cfg, "model.max_image_pixels"),
            use_fast_processor=deep_get(cfg, "model.use_fast_processor"),
        )
    )

    gen_max_tokens = int(deep_get(cfg, "generation.max_tokens", 800))
    gen_temperature = float(deep_get(cfg, "generation.temperature", 0.2))
    no_retry = bool(deep_get(cfg, "generation.no_retry", False))

    # NEW: retry accept policy
    retry_accept_policy = deep_get(cfg, "generation.retry_accept_policy", "if_better")
    if retry_accept_policy not in ("always", "never", "if_better"):
        logger.warning("Invalid generation.retry_accept_policy=%s, fallback to if_better", retry_accept_policy)
        retry_accept_policy = "if_better"

    chain = VisualChainSystem(
        backend=backend,
        cfg=ChainConfig(
            max_tokens=gen_max_tokens,
            temperature=gen_temperature,
            retry_on_fail=not no_retry,
            retry_accept_policy=retry_accept_policy,
        ),
    )

    pred_out = output_dir / "predictions.jsonl"
    summary_out = output_dir / "summary.json"

    if not resume_dir and pred_out.exists():
        raise FileExistsError(
            f"{pred_out} already exists. Use --resume or choose a new run_name/output_dir_prefix."
        )

    correct = 0
    total = 0
    choice_map_fail = 0
    error_count = 0

    start_time = datetime.now().isoformat()

    with jsonlines.open(pred_out, "a") as writer:
        with tqdm(total=len(pending), desc=f"Evaluating ({cmd})", unit="ex", ncols=90) as pbar:
            for i, ex in enumerate(pending, 1):
                ex_id = ex.example_id
                data = ex.data

                log_example_start(logger, ex_id, i, len(pending))
                try:
                    img_path = Path(data["image"])
                    question = data["question"]
                    choices = data.get("choices")
                    gt = (data.get("answer") or "").strip()

                    # NEW: pass gt so verifier runs in JUDGE mode
                    obs, cand, ver = chain.run_one(question, img_path, choices=choices, gt=gt)

                    pred_final = cand.final
                    mapped_choice = None
                    if choices and use_choice_extractor:
                        mapped_choice = choice_extractor_rule_based(pred_final, choices)
                        if mapped_choice is None:
                            mapped_choice = choice_extractor_local_llm(
                                backend=backend,
                                question=question,
                                choices=choices,
                                pred_freeform=pred_final,
                                image_path=img_path,
                            )
                        if mapped_choice == "?":
                            choice_map_fail += 1
                        else:
                            pred_final = mapped_choice

                    ok = score_sample(pred_final, gt, is_multiple_choice=choices is not None)

                    total += 1
                    correct += int(ok)

                    rec = {
                        "id": ex_id,
                        "image": str(img_path),
                        "question": question,
                        "choices": choices,
                        "gt": gt,
                        "pred_final": pred_final,
                        "pred_raw": cand.model_dump(),
                        "observation": obs.model_dump(),
                        "verifier": ver.model_dump(),
                        "correct": ok,
                        "choice_mapping_failed": (mapped_choice == "?") if choices else False,
                        "meta": data.get("meta", {}),
                    }
                    writer.write(rec)

                    log_example_result(logger, ex_id, pred_final, gt, ok, ver.verdict)

                    if resume_dir:
                        rs2 = load_resume_state(output_dir)
                        rs2.completed_ids.add(ex_id)
                        if ex_id in rs2.invalid_ids:
                            rs2.invalid_ids.discard(ex_id)
                        save_resume_state(rs2)

                except Exception as e:
                    error_count += 1
                    log_example_error(logger, ex_id, e)
                    if resume_dir:
                        rs2 = load_resume_state(output_dir)
                        rs2.invalid_ids.add(ex_id)
                        save_resume_state(rs2)

                pbar.update(1)

    end_time = datetime.now().isoformat()
    acc = correct / max(total, 1)

    summary = {
        "cmd": cmd,
        "model": model_name,
        "dataset_id": dataset_id,
        "split": split,
        "limit": limit,
        "start_time": start_time,
        "end_time": end_time,
        "output_dir": str(output_dir),
        "predictions_file": str(pred_out),
        "total": total,
        "correct": correct,
        "accuracy": acc,
        "choice_mapping_failures": choice_map_fail,
        "errors": error_count,
        "resume_enabled": bool(resume_dir),
        "retry_accept_policy": retry_accept_policy,
        "retry_on_fail": (not no_retry),
    }

    summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log_run_summary(logger, summary)
    return summary


# ============================================================
# CLI
# ============================================================

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)

    ap.add_argument("--config", default=None, help="Optional YAML config path.")
    ap.add_argument("--hf_endpoint", default=None, help="Optional HF mirror endpoint, e.g. https://hf-mirror.com")
    ap.add_argument("--out_root", default=None, help="Where to place prepared datasets (overrides config dataset.out_root)")
    ap.add_argument("--output_dir_prefix", default=None, help="Output dir prefix (overrides config output.dir_prefix)")
    ap.add_argument("--resume", default=None, help="Resume from an existing output directory (skip completed cases)")

    sub = ap.add_subparsers(dest="cmd", required=True)

    # Prepare
    p1 = sub.add_parser("prepare-mmbench", help="Download/prepare MMBench -> JSONL+images")
    p1.add_argument("--dataset_id", default=None)
    p1.add_argument("--split", default=None)
    p1.add_argument("--limit", type=int, default=None)

    p2 = sub.add_parser("prepare-mme", help="Download/prepare MME -> JSONL+images")
    p2.add_argument("--dataset_id", default=None)
    p2.add_argument("--split", default=None)
    p2.add_argument("--limit", type=int, default=None)

    # Eval MMBench
    e1 = sub.add_parser("eval-mmbench", help="Evaluate on prepared MMBench (local pipeline)")
    e1.add_argument("--dataset_id", default=None)
    e1.add_argument("--split", default=None)
    e1.add_argument("--limit", type=int, default=None)

    e1.add_argument("--run_name", default=None)
    e1.add_argument("--model", default=None)
    e1.add_argument("--device_map", default=None)
    e1.add_argument("--torch_dtype", default=None)
    e1.add_argument("--max_image_pixels", type=int, default=None)
    e1.add_argument("--use_fast_processor", type=str, default=None, help="true/false/none")
    e1.add_argument("--max_tokens", type=int, default=None)
    e1.add_argument("--temperature", type=float, default=None)
    e1.add_argument("--no_retry", action="store_true")
    e1.add_argument("--no_choice_extractor", action="store_true")
    # NEW
    e1.add_argument(
        "--retry_accept_policy",
        default=None,
        choices=["always", "never", "if_better"],
        help="Retry overwrite policy: always / never / if_better",
    )

    # Eval MME
    e2 = sub.add_parser("eval-mme", help="Evaluate on prepared MME (local pipeline)")
    e2.add_argument("--dataset_id", default=None)
    e2.add_argument("--split", default=None)
    e2.add_argument("--limit", type=int, default=None)

    e2.add_argument("--run_name", default=None)
    e2.add_argument("--model", default=None)
    e2.add_argument("--device_map", default=None)
    e2.add_argument("--torch_dtype", default=None)
    e2.add_argument("--max_image_pixels", type=int, default=None)
    e2.add_argument("--use_fast_processor", type=str, default=None, help="true/false/none")
    e2.add_argument("--max_tokens", type=int, default=None)
    e2.add_argument("--temperature", type=float, default=None)
    e2.add_argument("--no_retry", action="store_true")
    # NEW
    e2.add_argument(
        "--retry_accept_policy",
        default=None,
        choices=["always", "never", "if_better"],
        help="Retry overwrite policy: always / never / if_better",
    )

    # Analyze
    a1 = sub.add_parser("analyze", help="Analyze a predictions.jsonl file")
    a1.add_argument("--pred", required=True)

    return ap


def main() -> None:
    load_dotenv()

    ap = build_parser()
    args = ap.parse_args()

    cfg = load_config_file(args.config)
    cfg = merge_cli_into_config(args, cfg)

    resume_target = resolve_resume_dir(args.resume, Path("."))

    if args.cmd == "prepare-mmbench":
        ds_out_root = Path(deep_get(cfg, "dataset.out_root", "./datasets_out"))
        set_hf_endpoint(deep_get(cfg, "dataset.hf_endpoint"))
        dataset_id = deep_get(cfg, "dataset.dataset_id", "HuggingFaceM4/MMBench_dev")
        split = deep_get(cfg, "dataset.split", "train")
        limit = deep_get(cfg, "dataset.limit")
        jp, imgd = prepare_mmbench(ds_out_root, dataset_id=dataset_id, split=split, limit=limit)
        print(f"Prepared: {jp}\nImages: {imgd}")
        return

    if args.cmd == "prepare-mme":
        ds_out_root = Path(deep_get(cfg, "dataset.out_root", "./datasets_out"))
        set_hf_endpoint(deep_get(cfg, "dataset.hf_endpoint"))
        dataset_id = deep_get(cfg, "dataset.dataset_id", "lmms-lab/MME")
        split = deep_get(cfg, "dataset.split", "test")
        limit = deep_get(cfg, "dataset.limit")
        jp, imgd = prepare_mme(ds_out_root, dataset_id=dataset_id, split=split, limit=limit)
        print(f"Prepared: {jp}\nImages: {imgd}")
        return

    if args.cmd == "analyze":
        summary = analyze_predictions(args.pred)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    if args.cmd in ("eval-mmbench", "eval-mme"):
        if resume_target:
            print(f"Resume from: {resume_target}")
        summary = run_eval(args.cmd, cfg, resume_target)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    raise ValueError(f"Unknown cmd: {args.cmd}")


if __name__ == "__main__":
    main()

 # python vmas_local_eval.py --config config/mme_local.yaml eval-mme
 # python vmas_local_eval.py --config config/mmbench_local.yaml eval-mmbench --retry_accept_policy never
 # python vmas_local_eval.py --config config/mme_local.yaml eval-mme --retry_accept_policy never
