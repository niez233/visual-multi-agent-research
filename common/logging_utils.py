# common/logging_utils.py
from __future__ import annotations
import logging, re
from pathlib import Path
from datetime import datetime

class APIKeyMaskingFilter(logging.Filter):
    API_KEY_PATTERNS = [
        (re.compile(r"(sk-[a-zA-Z0-9_-]{20,})"), r"sk-***MASKED***"),
        (re.compile(r"(api_key['\"]?\s*[:=]\s*['\"]?)([a-zA-Z0-9_-]{20,})(['\"]?)", re.I), r"\1***MASKED***\3"),
        (re.compile(r"(API[_ ]?Key[:\s]*)['\"]?([a-zA-Z0-9_-]{20,})['\"]?", re.I), r"\1***MASKED***"),
    ]

    def filter(self, record: logging.LogRecord) -> bool:
        msg = str(record.getMessage())
        for pat, rep in self.API_KEY_PATTERNS:
            msg = pat.sub(rep, msg)
        record.msg = msg
        record.args = ()
        return True

def setup_run_logging(output_dir: Path, name: str = "vmas") -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
    fh = logging.FileHandler(output_dir / "run.log", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)

    flt = APIKeyMaskingFilter()
    fh.addFilter(flt)
    sh.addFilter(flt)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger

def log_run_config(logger: logging.Logger, config: dict, config_path: str | None = None) -> None:
    logger.info("=" * 60)
    logger.info("Run Config")
    if config_path:
        logger.info("Config file: %s", config_path)
    for k, v in config.items():
        logger.info("%s: %s", k, v)
    logger.info("=" * 60)

def log_example_start(logger: logging.Logger, example_id: str, idx: int, total: int) -> None:
    logger.info(">> [%d/%d] Start example=%s", idx, total, example_id)

def log_example_result(logger: logging.Logger, example_id: str, pred: str, gt: str, ok: bool, verdict: str | None) -> None:
    logger.info("<< example=%s ok=%s verdict=%s pred=%s gt=%s", example_id, ok, verdict, pred, gt)

def log_example_error(logger: logging.Logger, example_id: str, exc: Exception) -> None:
    logger.warning("!! example=%s error=%s", example_id, repr(exc), exc_info=True)

def log_run_summary(logger: logging.Logger, summary: dict) -> None:
    logger.info("=" * 60)
    logger.info("Summary")
    for k, v in summary.items():
        logger.info("%s: %s", k, v)
    logger.info("=" * 60)
