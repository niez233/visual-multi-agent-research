# chains/visual_chain.py
# -*- coding: utf-8 -*-
"""
Observer -> Reasoner -> Verifier pipeline (pure local).

Updates:
1) Verifier supports two modes:
   - JUDGE mode (when gt is provided): verdict = PASS iff candidate.final matches gt.
   - GROUNDING mode (when gt is None): keep previous "supported by image" verifier behavior.
2) Add retry_accept_policy to control rollback / overwrite behavior:
   - "always": always accept round2 if retried (old behavior)
   - "never": never accept round2 (run it optionally but keep round1)
   - "if_better": accept round2 only if it's better than round1 (PASS > FAIL, higher confidence, fewer issues)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from pydantic import BaseModel, Field, ValidationError

from local_mm_backend import TransformersQwenVLBackend


# ============================================================
# 1) Schemas (RELAXED)
# ============================================================

class ObjectEvidence(BaseModel):
    label: str
    attributes: Union[List[str], Dict[str, Any]] = Field(default_factory=list)
    count: Optional[int] = None
    bbox_xyxy: Optional[Union[List[float], List[List[float]]]] = None
    relations: List[str] = Field(default_factory=list)
    # normalized storage for multi-bbox case
    bboxes_xyxy: Optional[List[List[float]]] = None


class Observation(BaseModel):
    objects: List[ObjectEvidence] = Field(default_factory=list)
    # keep as list[str], but we will sanitize in normalize_observation
    ocr_text_blocks: List[str] = Field(default_factory=list)
    scene_summary: str
    uncertainties: List[str] = Field(default_factory=list)
    focus_suggestions: List[str] = Field(default_factory=list)


class CandidateAnswer(BaseModel):
    final: str
    rationale_brief: str
    # IMPORTANT: relax cited typing to avoid schema-validation crashes like [[1]] / "[1]" / weird lists.
    # We'll normalize it downstream into Dict[str, List[int]].
    cited: Dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(ge=0.0, le=1.0)


ErrorType = Literal[
    "perception_grounding",
    "reasoning",
    "answer_mapping",
    "propagation",
    "communication_compression",
    "unknown",
]


class VerificationResult(BaseModel):
    verdict: Literal["PASS", "FAIL"]
    error_type: ErrorType = "unknown"
    issues: List[str] = Field(default_factory=list)
    fix_hint: str = ""
    should_retry: bool = False


# ============================================================
# 2) Utils
# ============================================================

def normalize_text(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s\-\.\+]", "", s)
    return s


def extract_choice_letter(s: str) -> Optional[str]:
    ss = (s or "").strip().upper()
    m = re.search(r"\b([A-H])\b", ss)
    if m:
        return m.group(1)
    m = re.search(r"ANSWER[:\s]*\(?([A-H])\)?", ss)
    if m:
        return m.group(1)
    return None


def score_sample(pred_final: str, gt: str, is_multiple_choice: bool) -> bool:
    gt = (gt or "").strip()
    pred_final = (pred_final or "").strip()
    if is_multiple_choice:
        p = extract_choice_letter(pred_final) or pred_final.strip().upper()
        g = gt.strip().upper()
        return p == g
    return normalize_text(pred_final) == normalize_text(gt)


def safe_str(x: Any) -> str:
    if x is None:
        return ""
    s = str(x)
    if s.lower() in ("nan", "none"):
        return ""
    return s


# ---------- JSON extraction / repair helpers ----------

def best_effort_json_extract(text: str) -> Optional[dict]:
    """
    Attempt to extract a JSON object (dict) from model output.
    - tries direct json.loads
    - strips code fences
    - brace matching for the first complete {...} block
    """
    t = (text or "").strip()
    # 1) direct
    try:
        obj = json.loads(t)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # 2) strip code fences
    t2 = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
    t2 = re.sub(r"\s*```$", "", t2).strip()
    try:
        obj = json.loads(t2)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # 3) brace matching
    start = t2.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(t2)):
        ch = t2[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                block = t2[start : i + 1]
                try:
                    obj = json.loads(block)
                    if isinstance(obj, dict):
                        return obj
                except Exception:
                    return None
    return None


def normalize_observation(obs: Observation) -> Observation:
    """
    Normalize relaxed fields into more stable representation:
    - attributes dict -> list[str] (key:value strings); values can be list/dict -> json.dumps
    - bbox many -> store into bboxes_xyxy and set bbox_xyxy to first bbox
    - sanitize ocr_text_blocks items to string
    """
    new_objs: List[ObjectEvidence] = []
    for o in obs.objects:
        d = o.model_dump()
        attrs = d.get("attributes", [])
        if isinstance(attrs, dict):
            attrs_list: List[str] = []
            for k, v in attrs.items():
                if isinstance(v, (dict, list)):
                    attrs_list.append(f"{k}:{json.dumps(v, ensure_ascii=False)}")
                else:
                    attrs_list.append(f"{k}:{safe_str(v)}")
            d["attributes"] = attrs_list
        elif isinstance(attrs, list):
            d["attributes"] = [safe_str(x) for x in attrs if safe_str(x)]
        else:
            d["attributes"] = []

        bbox = d.get("bbox_xyxy", None)
        if isinstance(bbox, list) and bbox and isinstance(bbox[0], list):
            d["bboxes_xyxy"] = bbox
            d["bbox_xyxy"] = bbox[0] if bbox[0] else None

        new_objs.append(ObjectEvidence.model_validate(d))

    # sanitize OCR blocks
    ocr_blocks: List[str] = []
    for x in (obs.ocr_text_blocks or []):
        sx = safe_str(x).strip()
        if sx:
            ocr_blocks.append(sx)

    return Observation(
        objects=new_objs,
        ocr_text_blocks=ocr_blocks,
        scene_summary=obs.scene_summary,
        uncertainties=obs.uncertainties,
        focus_suggestions=obs.focus_suggestions,
    )


def _extract_ints_from_string(s: str) -> List[int]:
    s = (s or "").strip()
    if not s:
        return []
    nums = re.findall(r"-?\d+", s)
    out: List[int] = []
    for n in nums:
        try:
            out.append(int(n))
        except Exception:
            pass
    return out


def _to_int_list(v: Any) -> List[int]:
    """
    Convert v into list[int] best-effort, robust to common LLM glitches.
    """
    if v is None:
        return []

    if isinstance(v, bool):
        return []

    if isinstance(v, int):
        return [v]

    if isinstance(v, float):
        return [int(v)] if v.is_integer() else []

    if isinstance(v, str):
        s = v.strip()
        if s.isdigit() or (s.startswith("-") and s[1:].isdigit()):
            try:
                return [int(s)]
            except Exception:
                return []
        return _extract_ints_from_string(s)

    if isinstance(v, list):
        out: List[int] = []
        for x in v:
            out.extend(_to_int_list(x))
        seen = set()
        uniq: List[int] = []
        for x in out:
            if x not in seen:
                uniq.append(x)
                seen.add(x)
        return uniq

    if isinstance(v, dict):
        out: List[int] = []
        for _, vv in v.items():
            out.extend(_to_int_list(vv))
        seen = set()
        uniq: List[int] = []
        for x in out:
            if x not in seen:
                uniq.append(x)
                seen.add(x)
        return uniq

    return []


def normalize_candidate(c: CandidateAnswer) -> CandidateAnswer:
    """
    Normalize CandidateAnswer.cited into Dict[str, List[int]]-like structure.
    Also filter keys to expected ones to reduce downstream noise.
    """
    d = c.model_dump()
    cited = d.get("cited") or {}
    allowed_keys = {"objects", "ocr_text_blocks", "uncertainties"}

    cited2: Dict[str, List[int]] = {}
    if isinstance(cited, dict):
        for k, v in cited.items():
            if k in allowed_keys:
                cited2[k] = _to_int_list(v)

    d["cited"] = cited2
    return CandidateAnswer.model_validate(d)


# ============================================================
# 3) Structured JSON caller
# ============================================================

JSON_RULES = (
    "Return ONLY ONE valid JSON object that matches the schema.\n"
    "- Do NOT include Markdown or code fences (```).\n"
    "- Do NOT include $defs, schema, or any explanations.\n"
    "- Output must be a JSON object with ONLY the required/allowed fields.\n"
)

def call_structured_json(
    backend: TransformersQwenVLBackend,
    schema: type[BaseModel],
    system_prompt: str,
    user_prompt: str,
    image_path: Optional[Union[str, Path]] = None,
    max_tokens: int = 900,
    temperature: float = 0.2,
    repair_rounds: int = 4,
) -> BaseModel:
    schema_fields = schema.model_json_schema()

    def build_messages(sp: str, up: str) -> List[Dict[str, Any]]:
        return [
            {
                "role": "system",
                "content": (
                    f"{sp}\n\n{JSON_RULES}\n\nJSON schema:\n"
                    f"{json.dumps(schema_fields, ensure_ascii=False)}"
                ),
            },
            {"role": "user", "content": up},
        ]

    messages = build_messages(system_prompt, user_prompt)
    raw = backend.generate_text(
        messages,
        image_path=image_path,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    obj = best_effort_json_extract(raw)
    if obj is not None:
        try:
            return schema.model_validate(obj)
        except ValidationError:
            pass

    last_raw = raw
    for _ in range(repair_rounds):
        repair_prompt = (
            "Your output is invalid or does not match the schema.\n"
            "Fix it and return ONLY the corrected JSON object.\n"
            "Rules:\n"
            "- No ``` fences\n"
            "- No $defs or schema\n"
            "- Must conform exactly\n\n"
            f"Bad output:\n{last_raw}\n"
        )
        messages = build_messages(system_prompt, repair_prompt)
        raw2 = backend.generate_text(
            messages,
            image_path=image_path,
            max_tokens=max_tokens,
            temperature=0.0,
        )
        obj2 = best_effort_json_extract(raw2)
        if obj2 is None:
            last_raw = raw2
            continue
        try:
            return schema.model_validate(obj2)
        except ValidationError:
            last_raw = raw2
            continue

    raise RuntimeError(
        f"Failed to get valid JSON for schema={schema.__name__}. Last output:\n{last_raw}"
    )


# ============================================================
# 4) Chain prompts
# ============================================================

OBSERVER_SYS = """You are Observer/Grounder.
Task: look at the image and produce structured, factual visual evidence ONLY.
Rules:
- Do NOT guess. If uncertain, add to uncertainties.
- Prefer concrete, checkable claims: counts, relative positions, exact text.
- If you mention an object, put it into objects.
- attributes MUST be either:
  - a list of strings, e.g. ["white", "metal", "upside-down"]
  - OR a dict of short properties, values can be string/list/dict
- bbox_xyxy can be:
  - one box [x1,y1,x2,y2]
  - OR a list of boxes [[x1,y1,x2,y2], ...]
- OCR: include all readable text blocks.
"""

REASONER_SYS = """You are Reasoner.
Task: answer the question using ONLY the provided Observation (objects/ocr/summary).
Rules:
- If multiple-choice, output a single choice letter as final.
- Your answer MUST be supported by cited evidence indices.

- cited MUST be a JSON dict with ONLY these optional keys:
  "objects", "ocr_text_blocks", "uncertainties".
- Each cited value MUST be a list of integer indices into the corresponding list in Observation.
  Example: {"objects":[0,2], "ocr_text_blocks":[1]}
- Do NOT put sentences as keys. Do NOT use single numbers (must be a list).

- If evidence is insufficient, say so in final, keep confidence low, and cite uncertainties.
"""

# Grounding verifier (no gt)
VERIFIER_SYS_GROUNDING = """You are Verifier.
Task: verify whether CandidateAnswer is supported by the IMAGE (not just the text from previous agents).
You must:
1) Check grounding: cited evidence matches what is in the image.
2) Check logic: the conclusion follows from evidence (briefly).
3) If FAIL: assign error_type and give minimal fix_hint and whether to retry.

Error types:
- perception_grounding
- reasoning
- answer_mapping
- propagation
- communication_compression
"""

# Judge verifier (with gt) — for benchmark/eval
VERIFIER_SYS_JUDGE = """You are Verifier/Judge.
You will be given: question, choices(optional), ground-truth answer (gt),
observation, candidate answer, and the image.

Task:
1) Decide if candidate.final matches gt after normalization:
   - For multiple-choice: compare option letter (A-H).
   - For yes/no: case-insensitive compare "yes"/"no" (also accept "y"/"n").
2) If matches gt: verdict=PASS.
3) If NOT matches gt: verdict=FAIL, and diagnose the PRIMARY cause:
   - perception_grounding: observation evidence is wrong/missing vs image (OCR/object mistake)
   - reasoning: observation contains enough correct evidence but candidate reasoning/conclusion is wrong
   - answer_mapping: output format or mapping mistake (wrong letter, polarity flip, extra text)
   - propagation: evidence indices/cited broken or info lost between agents
   - communication_compression: schema/format constraints caused loss/truncation
   - unknown: cannot decide
4) When FAIL, provide minimal issues + fix_hint, and set should_retry:
   - should_retry=true if the error seems fixable by re-observing or re-reasoning.
   - otherwise false.

Keep issues concise and concrete.
"""


# ============================================================
# 5) Chain system
# ============================================================

RetryAcceptPolicy = Literal["always", "never", "if_better"]

@dataclass
class ChainConfig:
    max_tokens: int = 800
    temperature: float = 0.2
    retry_on_fail: bool = True
    # NEW: rollback / overwrite switch
    retry_accept_policy: RetryAcceptPolicy = "if_better"


def _is_better(
    ver_a: VerificationResult,
    cand_a: CandidateAnswer,
    ver_b: VerificationResult,
    cand_b: CandidateAnswer,
) -> bool:
    """
    Decide if (b) is better than (a).
    Priority:
      1) PASS beats FAIL
      2) higher confidence
      3) fewer issues
    """
    if ver_b.verdict != ver_a.verdict:
        return ver_b.verdict == "PASS"
    if cand_b.confidence != cand_a.confidence:
        return cand_b.confidence > cand_a.confidence
    return len(ver_b.issues) < len(ver_a.issues)


class VisualChainSystem:
    def __init__(self, backend: TransformersQwenVLBackend, cfg: ChainConfig):
        self.backend = backend
        self.cfg = cfg

    def observe(
        self,
        question: str,
        image_path: Union[str, Path],
        extra_focus: str = "",
    ) -> Observation:
        q = question if not extra_focus else f"{question}\n\n[Extra focus]\n{extra_focus}"
        obs = call_structured_json(
            backend=self.backend,
            schema=Observation,
            system_prompt=OBSERVER_SYS,
            user_prompt=q,
            image_path=image_path,
            max_tokens=self.cfg.max_tokens,
            temperature=self.cfg.temperature,
        )
        return normalize_observation(obs)

    def reason(
        self,
        question: str,
        obs: Observation,
        choices: Optional[Dict[str, str]] = None,
    ) -> CandidateAnswer:
        payload = {"question": question, "choices": choices, "observation": obs.model_dump()}
        cand = call_structured_json(
            backend=self.backend,
            schema=CandidateAnswer,
            system_prompt=REASONER_SYS,
            user_prompt=json.dumps(payload, ensure_ascii=False),
            image_path=None,
            max_tokens=self.cfg.max_tokens,
            temperature=self.cfg.temperature,
        )
        return normalize_candidate(cand)

    def verify(
        self,
        question: str,
        image_path: Union[str, Path],
        obs: Observation,
        cand: CandidateAnswer,
        choices: Optional[Dict[str, str]] = None,
        gt: Optional[str] = None,  # NEW
    ) -> VerificationResult:
        payload = {
            "question": question,
            "choices": choices,
            "gt": gt,  # NEW (None allowed)
            "observation": obs.model_dump(),
            "candidate": cand.model_dump(),
        }
        sys_prompt = VERIFIER_SYS_JUDGE if (gt is not None and str(gt).strip() != "") else VERIFIER_SYS_GROUNDING
        return call_structured_json(
            backend=self.backend,
            schema=VerificationResult,
            system_prompt=sys_prompt,
            user_prompt=json.dumps(payload, ensure_ascii=False),
            image_path=image_path,
            max_tokens=self.cfg.max_tokens,
            temperature=0.0,
        )

    def run_one(
        self,
        question: str,
        image_path: Union[str, Path],
        choices: Optional[Dict[str, str]] = None,
        gt: Optional[str] = None,  # NEW
    ) -> Tuple[Observation, CandidateAnswer, VerificationResult]:
        # Round 1
        obs1 = self.observe(question, image_path)
        cand1 = self.reason(question, obs1, choices=choices)
        ver1 = self.verify(question, image_path, obs1, cand1, choices=choices, gt=gt)

        # Retry gate
        if not (self.cfg.retry_on_fail and ver1.verdict == "FAIL" and ver1.should_retry):
            return obs1, cand1, ver1

        # Round 2 (fix_hint is passed to Observer)
        focus = ver1.fix_hint or "Re-check the most uncertain parts carefully."
        obs2 = self.observe(question, image_path, extra_focus=focus)
        cand2 = self.reason(question, obs2, choices=choices)
        ver2 = self.verify(question, image_path, obs2, cand2, choices=choices, gt=gt)

        # Accept policy (rollback switch)
        if self.cfg.retry_accept_policy == "always":
            return obs2, cand2, ver2
        if self.cfg.retry_accept_policy == "never":
            return obs1, cand1, ver1

        # if_better
        return (obs2, cand2, ver2) if _is_better(ver1, cand1, ver2, cand2) else (obs1, cand1, ver1)


# ============================================================
# 6) Choice mapping for MMBench (local)
# ============================================================

def choice_extractor_rule_based(pred: str, choices: Dict[str, str]) -> Optional[str]:
    letter = extract_choice_letter(pred)
    if letter and letter in choices:
        return letter

    p = normalize_text(pred)
    best = None
    best_score = 0.0
    for k, v in choices.items():
        vv = normalize_text(v)
        p_tokens = set(p.split())
        v_tokens = set(vv.split())
        if not v_tokens:
            continue
        score = len(p_tokens & v_tokens) / (len(v_tokens) + 1e-9)
        if score > best_score:
            best_score = score
            best = k
    if best is not None and best_score >= 0.6:
        return best
    return None


def choice_extractor_local_llm(
    backend: TransformersQwenVLBackend,
    question: str,
    choices: Dict[str, str],
    pred_freeform: str,
    image_path: Optional[Union[str, Path]] = None,
) -> str:
    sys_prompt = (
        "You are a strict choice extractor.\n"
        "Given the question, options, and a prediction, output ONLY one capital letter among the option keys.\n"
        "No other text."
    )
    payload = {"question": question, "choices": choices, "prediction": pred_freeform}
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]
    out = backend.generate_text(messages, image_path=image_path, max_tokens=64, temperature=0.0)
    letter = extract_choice_letter(out) or out.strip().upper()
    if letter not in choices:
        rb = choice_extractor_rule_based(out, choices)
        if rb:
            return rb
        if len(letter) >= 1 and letter[0] in choices:
            return letter[0]
        return "?"
    return letter
