# local_mm_backend.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import inspect
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from PIL import Image


@dataclass
class LocalBackendConfig:
    model_name_or_path: str
    device_map: str = "auto"

    # 兼容你现有 YAML：仍然用 torch_dtype
    # 如果你后续想改成 dtype，也可以在外层把值映射到这里
    torch_dtype: str = "auto"

    max_image_pixels: Optional[int] = None  # downscale for speed/memory if needed
    use_fast_processor: Optional[bool] = None  # None=default; False forces slow when supported


def _supports_kwarg(fn, name: str) -> bool:
    try:
        return name in inspect.signature(fn).parameters
    except Exception:
        return False


def _build_dtype_kwargs(fn, dtype_value: Any) -> Dict[str, Any]:
    """
    transformers 不同版本里，from_pretrained 的 dtype 参数名可能是 dtype 或 torch_dtype
    用签名探测自动适配。
    """
    try:
        sig = inspect.signature(fn)
        params = sig.parameters
        if "dtype" in params:
            return {"dtype": dtype_value}
        if "torch_dtype" in params:
            return {"torch_dtype": dtype_value}
    except Exception:
        pass
    # 探测失败默认用 torch_dtype（更常见）
    return {"torch_dtype": dtype_value}


def _maybe_add_trust_remote_code(fn, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    某些 AutoModel / from_pretrained 支持 trust_remote_code，某些不支持；
    这里按签名探测后再加，避免“unexpected kwarg”。
    """
    if _supports_kwarg(fn, "trust_remote_code"):
        kwargs["trust_remote_code"] = True
    return kwargs


class TransformersQwenVLBackend:
    """
    Pure local multimodal backend using HuggingFace Transformers.

    - 自动识别 Qwen3-VL / Qwen2.5-VL，并尽量使用正确的专用类
    - transformers==4.57 环境下不依赖 AutoModelForConditionalGeneration（你环境里没有导出）
    - 兜底优先使用 AutoModelForVision2Seq（更贴合多模态条件生成），再降级到 Seq2SeqLM，最后才考虑 CausalLM

    messages format:
      [{"role":"system","content":"..."},
       {"role":"user","content":"..."}]
    or multimodal list content in user message (we normalize internally).
    """

    def __init__(self, cfg: LocalBackendConfig):
        import torch
        from transformers import AutoConfig, AutoProcessor

        self.cfg = cfg
        self.torch = torch

        # 1) 读 config 判定 model_type
        config = AutoConfig.from_pretrained(cfg.model_name_or_path)
        model_type = getattr(config, "model_type", None)

        # 2) dtype 值（你的 yaml 里还是 torch_dtype）
        dtype_value = cfg.torch_dtype

        self.model = None
        last_err: Optional[BaseException] = None

        # 3) 优先：按 model_type 走专用模型类（如果你的 transformers 4.57 构建里包含这些类）
        if model_type == "qwen3_vl":
            try:
                from transformers import Qwen3VLForConditionalGeneration  # type: ignore

                fn = Qwen3VLForConditionalGeneration.from_pretrained
                kwargs = _build_dtype_kwargs(fn, dtype_value)
                kwargs = _maybe_add_trust_remote_code(fn, kwargs)
                self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                    cfg.model_name_or_path,
                    device_map=cfg.device_map,
                    **kwargs,
                )
            except Exception as e:
                last_err = e

        elif model_type == "qwen2_5_vl":
            try:
                from transformers import Qwen2_5_VLForConditionalGeneration  # type: ignore

                fn = Qwen2_5_VLForConditionalGeneration.from_pretrained
                kwargs = _build_dtype_kwargs(fn, dtype_value)
                kwargs = _maybe_add_trust_remote_code(fn, kwargs)
                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    cfg.model_name_or_path,
                    device_map=cfg.device_map,
                    **kwargs,
                )
            except Exception as e:
                last_err = e

        # 4) 兜底：AutoModelForVision2Seq（更适合 VL/条件生成）
        if self.model is None:
            auto_loaded = False

            try:
                from transformers import AutoModelForVision2Seq  # type: ignore

                fn = AutoModelForVision2Seq.from_pretrained
                kwargs = _build_dtype_kwargs(fn, dtype_value)
                kwargs = _maybe_add_trust_remote_code(fn, kwargs)
                self.model = AutoModelForVision2Seq.from_pretrained(
                    cfg.model_name_or_path,
                    device_map=cfg.device_map,
                    **kwargs,
                )
                auto_loaded = True
            except Exception as e:
                last_err = e

            # 5) 再兜底：AutoModelForSeq2SeqLM
            if not auto_loaded:
                try:
                    from transformers import AutoModelForSeq2SeqLM  # type: ignore

                    fn = AutoModelForSeq2SeqLM.from_pretrained
                    kwargs = _build_dtype_kwargs(fn, dtype_value)
                    kwargs = _maybe_add_trust_remote_code(fn, kwargs)
                    self.model = AutoModelForSeq2SeqLM.from_pretrained(
                        cfg.model_name_or_path,
                        device_map=cfg.device_map,
                        **kwargs,
                    )
                    auto_loaded = True
                except Exception as e:
                    last_err = e

            # 6) 最后兜底：AutoModelForCausalLM（不推荐，但给非 VL 模型留口子）
            if not auto_loaded:
                try:
                    from transformers import AutoModelForCausalLM  # type: ignore

                    fn = AutoModelForCausalLM.from_pretrained
                    kwargs = _build_dtype_kwargs(fn, dtype_value)
                    kwargs = _maybe_add_trust_remote_code(fn, kwargs)
                    self.model = AutoModelForCausalLM.from_pretrained(
                        cfg.model_name_or_path,
                        device_map=cfg.device_map,
                        **kwargs,
                    )
                    auto_loaded = True
                except Exception as e:
                    last_err = e

            if not auto_loaded or self.model is None:
                # 给出明确可操作提示：你的 transformers 4.57 包里可能缺少 Qwen3-VL 相关实现
                if model_type == "qwen3_vl":
                    raise RuntimeError(
                        "无法在当前 transformers==4.57 环境中加载 Qwen3-VL。\n"
                        "常见原因：你安装的 4.57 发行包未包含 Qwen3VLForConditionalGeneration，且 AutoModelForVision2Seq 也无法匹配该配置。\n"
                        "建议：安装 transformers 最新源码版（或升级到包含 Qwen3-VL 支持的版本）。\n"
                        f"原始错误：{repr(last_err)}"
                    ) from last_err

                raise RuntimeError(
                    f"无法加载模型：model_type={model_type!r}, path={cfg.model_name_or_path!r}\n"
                    f"原始错误：{repr(last_err)}"
                ) from last_err

        self.model.eval()

        # 7) processor
        proc_kwargs: Dict[str, Any] = {}
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

        # move tensors to model device (keep non-tensors as-is)
        device = self.model.device
        for k, v in list(inputs.items()):
            if hasattr(v, "to"):
                inputs[k] = v.to(device)

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
