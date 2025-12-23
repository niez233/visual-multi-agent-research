# tools/resume_utils.py
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

@dataclass(frozen=True)
class PreparedExample:
    example_id: str
    data: dict
    absolute_index: int  # 1-based index for stable logging

@dataclass
class ResumeState:
    output_dir: Path
    completed_ids: set[str]
    invalid_ids: set[str]

    @property
    def completed_count(self) -> int: return len(self.completed_ids)
    @property
    def invalid_count(self) -> int: return len(self.invalid_ids)

def resolve_resume_dir(resume_arg: Optional[str], output_dir: Path) -> Optional[Path]:
    if resume_arg:
        return Path(resume_arg)
    # 也可以允许从 output_dir 自动判断（这里先简单点）
    return None

def load_resume_state(output_dir: Path) -> ResumeState:
    state_file = output_dir / "resume_state.json"
    if not state_file.exists():
        return ResumeState(output_dir=output_dir, completed_ids=set(), invalid_ids=set())
    obj = json.loads(state_file.read_text(encoding="utf-8"))
    return ResumeState(
        output_dir=output_dir,
        completed_ids=set(obj.get("completed_ids", [])),
        invalid_ids=set(obj.get("invalid_ids", [])),
    )

def save_resume_state(state: ResumeState) -> None:
    state_file = state.output_dir / "resume_state.json"
    payload = {
        "completed_ids": sorted(state.completed_ids),
        "invalid_ids": sorted(state.invalid_ids),
    }
    state_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

def prepare_examples(examples: Iterable[dict]) -> list[PreparedExample]:
    out: list[PreparedExample] = []
    for idx, ex in enumerate(examples, 1):
        ex_id = str(ex.get("id") or idx)
        out.append(PreparedExample(example_id=ex_id, data=ex, absolute_index=idx))
    return out

def filter_pending(prepared: list[PreparedExample], resume: ResumeState) -> list[PreparedExample]:
    pending = [ex for ex in prepared if ex.example_id not in resume.completed_ids]
    # invalid_ids 可以选择强制重跑（这里默认重跑 invalid）
    return pending
