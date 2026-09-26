"""
Assignment 11 — Audit Log starter (TODO).

Records every interaction for forensics. Never blocks by itself —
other layers catch attacks; this layer makes them reviewable.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def default_audit_log_path() -> str:
    """Always resolve to <repo>/outputs/… (safe when cwd is src/)."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "audit_log.json")


class AuditLogPlugin:
    """Framework-agnostic audit logger (wire into ADK callbacks or your pipeline)."""

    def __init__(self):
        self.name = "audit_log"
        self.logs: list[dict] = []
        self._open: dict[str, float] = {}

    @staticmethod
    def _safe_text(text: str) -> str:
        """Redact secrets/PII before writing them to a durable audit file."""
        try:
            from guardrails.output_guardrails import content_filter
            return content_filter(text or "")["redacted"]
        except Exception:
            # Logging must never break the request path; use a conservative
            # fallback if optional guardrail dependencies are unavailable.
            import re
            return re.sub(r"(?i)(?:password|api\s*key)\s*[:=]\s*\S+", "[REDACTED]", text or "")

    def record_input(self, *, user_id: str, text: str, request_id: str | None = None):
        """TODO: store input + start timestamp keyed by request_id/user_id."""
        import time
        key = request_id or user_id or "anonymous"
        self._open[key] = time.perf_counter()
        self.logs.append({
            "event": "input", "request_id": request_id,
            "user_id": user_id or "anonymous", "input": self._safe_text(text),
            "timestamp": utc_now_iso(),
        })
        return key

    def record_output(
        self,
        *,
        user_id: str,
        text: str,
        blocked: bool = False,
        layer: str | None = None,
        request_id: str | None = None,
    ):
        """Store output, decision and latency as a separate audit event."""
        import time
        key = request_id or user_id or "anonymous"
        started = self._open.pop(key, None)
        latency_ms = round((time.perf_counter() - started) * 1000, 3) if started else None
        self.logs.append({
            "event": "output", "request_id": request_id,
            "user_id": user_id or "anonymous", "output": self._safe_text(text),
            "blocked": bool(blocked), "layer": layer,
            "latency_ms": latency_ms, "timestamp": utc_now_iso(),
        })
        return self.logs[-1]

    def export_json(self, filepath: str | None = None):
        """Write logs to disk (JSON array) under repo-root ``outputs/`` by default."""
        path = Path(filepath or default_audit_log_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.logs, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(path)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
