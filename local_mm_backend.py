# local_mm_backend.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from PIL import Image


@dataclass
class LocalBackendConfig:
    model_name_or_path: str
    device_map: str = "auto"
    torch_dtype: str = "auto"
    max_image_pixels: Optional[int] = None  # downscale for speed/memory if needed
    use_fast_processor: Optional[bool] = None  # None=default; False forces slow when supported


class TransformersQwenVLBackend:
    """
    Pure local multimodal backend using HuggingFace Transformers.
    Works best with Qwen2.5-VL, but generally compatible with VL chat models that support:
      processor.apply_chat_template(messages, add_generation_prompt=True)
    and accept images in processor(..., images=[PIL,...]).

    messages format:
      [{"role":"system","content":"..."},
       {"role":"user","content":"..."}]
    or multimodal list content in user message (we normalize internally).
    """

    def __init__(self, cfg: LocalBackendConfig):
        import torch
        from transformers import AutoProcessor

        self.cfg = cfg
        self.torch = torch

        # Load model
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration  # type: ignore
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                cfg.model_name_or_path,
                device_map=cfg.device_map,
                torch_dtype=cfg.torch_dtype,  # may warn deprecated in some versions; ok
            )
        except Exception:
            from transformers import AutoModelForCausalLM
            self.model = AutoModelForCausalLM.from_pretrained(
                cfg.model_name_or_path,
                device_map=cfg.device_map,
                torch_dtype=cfg.torch_dtype,
            )

        # Load processor
        proc_kwargs = {}
        if cfg.use_fast_processor is not None:
            proc_kwargs["use_fast"] = cfg.use_fast_processor
        self.processor = AutoProcessor.from_pretrained(cfg.model_name_or_path, **proc_kwargs)

    def _maybe_downscale(self, img: Image.Image) -> Image.Image:
        if not self.cfg.max_image_pixels:
            return img
        w, h = img.size
        if w * h <= self.cfg.max_image_pixels:
            return img
        scale = math.sqrt(self.cfg.max_image_pixels / float(w * h))
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        return img.resize((nw, nh))

    def generate_text(
        self,
        messages: List[Dict[str, Any]],
        image_path: Optional[Union[str, Path]] = None,
        max_tokens: int = 800,
        temperature: float = 0.2,
    ) -> str:
        # Normalize messages into multimodal content parts
        mm_messages: List[Dict[str, Any]] = []
        for m in messages:
            role = m["role"]
            content = m.get("content", "")
            if isinstance(content, list):
                mm_messages.append({"role": role, "content": content})
            else:
                mm_messages.append({"role": role, "content": [{"type": "text", "text": str(content)}]})

        # Attach image to last user message if provided
        if image_path is not None:
            img = Image.open(image_path).convert("RGB")
            img = self._maybe_downscale(img)
            if not mm_messages or mm_messages[-1]["role"] != "user":
                mm_messages.append({"role": "user", "content": []})
            mm_messages[-1]["content"].append({"type": "image", "image": img})

        # Chat template -> prompt text
        text = self.processor.apply_chat_template(mm_messages, tokenize=False, add_generation_prompt=True)

        # Collect images
        images: List[Image.Image] = []
        for m in mm_messages:
            for part in m.get("content", []):
                if isinstance(part, dict) and part.get("type") == "image":
                    images.append(part["image"])

        inputs = self.processor(
            text=[text],
            images=images if images else None,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        do_sample = temperature > 1e-6
        gen_kwargs = dict(
            max_new_tokens=max_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
        )
        gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

        with self.torch.inference_mode():
            out = self.model.generate(**inputs, **gen_kwargs)

        # Decode generated part
        try:
            input_len = inputs["input_ids"].shape[-1]
            gen_ids = out[0][input_len:]
        except Exception:
            gen_ids = out[0]

        decoded = self.processor.decode(gen_ids, skip_special_tokens=True)
        return decoded.strip()
