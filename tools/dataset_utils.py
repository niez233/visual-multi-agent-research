# tools/dataset_utils.py
from __future__ import annotations
import os
import json
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import jsonlines
from PIL import Image
from tqdm import tqdm

def ensure_dir(p: Union[str, Path]) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p

def safe_str(x: Any) -> str:
    if x is None:
        return ""
    s = str(x)
    return "" if s.lower() in ("nan", "none") else s

def set_hf_endpoint(endpoint: Optional[str]) -> None:
    if endpoint:
        os.environ["HF_ENDPOINT"] = endpoint

def load_hf_dataset(dataset_id: str, split: str):
    from datasets import load_dataset
    return load_dataset(dataset_id, split=split)

def save_any_image(obj: Any, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(obj, Image.Image):
        obj.save(out_path, format="PNG"); return
    if isinstance(obj, dict):
        if obj.get("bytes") is not None:
            out_path.write_bytes(obj["bytes"]); return
        if obj.get("path"):
            Image.open(obj["path"]).convert("RGB").save(out_path, format="PNG"); return
    if isinstance(obj, str) and obj:
        Image.open(obj).convert("RGB").save(out_path, format="PNG"); return
    raise ValueError(f"Unsupported image object type: {type(obj)}")

def infer_answer_from_mmbench_row(row: Dict[str, Any]) -> str:
    if "answer" in row and safe_str(row["answer"]):
        return safe_str(row["answer"]).strip()
    if "label" in row and row["label"] is not None:
        lab = row["label"]
        if isinstance(lab, int):
            mapping = "ABCDEFGH"
            return mapping[lab] if 0 <= lab < len(mapping) else str(lab)
        s = safe_str(lab).strip()
        m = re.search(r"\b([A-H])\b", s.upper())
        return m.group(1) if m else s
    return ""

def prepare_mmbench(
    out_root: Union[str, Path],
    dataset_id: str = "HuggingFaceM4/MMBench_dev",
    split: str = "train",
    limit: Optional[int] = None,
) -> Tuple[Path, Path]:
    out_root = Path(out_root)
    tag = f"{dataset_id.replace('/', '_')}__{split}" + (f"__limit{limit}" if limit else "")
    base = ensure_dir(out_root / "mmbench" / tag)
    images_dir = ensure_dir(base / "images")
    jsonl_path = base / "data.jsonl"
    if jsonl_path.exists() and any(images_dir.iterdir()):
        return jsonl_path, images_dir

    ds = load_hf_dataset(dataset_id, split=split)
    with jsonlines.open(jsonl_path, "w") as w:
        for i, item in enumerate(tqdm(ds, desc=f"prepare mmbench {dataset_id}:{split}")):
            if limit and i >= limit: break
            row = dict(item)
            q = safe_str(row.get("question"))
            hint = safe_str(row.get("hint"))
            if hint:
                q = f"{q}\n\nHint:\n{hint}"

            choices: Dict[str, str] = {}
            for key in list("ABCDEFGH"):
                vs = safe_str(row.get(key))
                if vs:
                    choices[key] = vs

            ans = infer_answer_from_mmbench_row(row)
            ex_id = safe_str(row.get("index")) or safe_str(row.get("id")) or str(i)

            img_path = images_dir / f"{ex_id}.png"
            save_any_image(row.get("image"), img_path)

            w.write({
                "id": f"mmbench:{ex_id}",
                "image": str(img_path),
                "question": q,
                "choices": choices if choices else None,
                "answer": ans,
                "meta": {"source": "MMBench", "dataset_id": dataset_id, "split": split},
            })
    return jsonl_path, images_dir

def prepare_mme(
    out_root: Union[str, Path],
    dataset_id: str = "lmms-lab/MME",
    split: str = "test",
    limit: Optional[int] = None,
) -> Tuple[Path, Path]:
    out_root = Path(out_root)
    tag = f"{dataset_id.replace('/', '_')}__{split}" + (f"__limit{limit}" if limit else "")
    base = ensure_dir(out_root / "mme" / tag)
    images_dir = ensure_dir(base / "images")
    jsonl_path = base / "data.jsonl"
    if jsonl_path.exists() and any(images_dir.iterdir()):
        return jsonl_path, images_dir

    ds = load_hf_dataset(dataset_id, split=split)
    with jsonlines.open(jsonl_path, "w") as w:
        for i, item in enumerate(tqdm(ds, desc=f"prepare mme {dataset_id}:{split}")):
            if limit and i >= limit: break
            row = dict(item)
            q = safe_str(row.get("question")) or safe_str(row.get("text")) or safe_str(row.get("prompt"))
            ans = safe_str(row.get("answer")) or safe_str(row.get("label")) or safe_str(row.get("gt_answer"))
            img_obj = row.get("image") or row.get("img") or row.get("image_path") or row.get("img_path")
            ex_id = safe_str(row.get("id")) or safe_str(row.get("index")) or str(i)

            img_path = images_dir / f"{ex_id}.png"
            save_any_image(img_obj, img_path)

            w.write({
                "id": f"mme:{ex_id}",
                "image": str(img_path),
                "question": q,
                "choices": None,
                "answer": ans,
                "meta": {"source": "MME", "dataset_id": dataset_id, "split": split},
            })
    return jsonl_path, images_dir

def load_local_dataset(jsonl_path: Union[str, Path]) -> list[dict]:
    jsonl_path = Path(jsonl_path)
    out = []
    with jsonlines.open(jsonl_path, "r") as r:
        for ex in r:
            out.append(ex)
    return out
