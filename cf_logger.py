"""
cf_logger.py
============
Sends pipeline run and step data to the GCP exec-logger Cloud Function.
Mirrors the local StepLogger so every step is also persisted remotely.

Usage in run_pipeline.py:
    from cf_logger import CFLogger
    cf = CFLogger(run_id="run-20260228-001")
    cf.create_run(program="run_pipeline.py", cmd="...", options={...})
    cf.log_step(step_num=1, input_text="...", output_text="...", file_path="/path/to/file.png")

Set CF_ENABLED=false (env var) or pass enabled=False to disable without code changes.
"""

import os
import uuid
import datetime
import requests


FUNC_URL = os.environ.get(
    "CF_FUNC_URL",
    "https://exec-logger-7jstnd322q-uc.a.run.app"
)
CF_TIMEOUT = int(os.environ.get("CF_TIMEOUT", "10"))  # seconds


class CFLogger:
    """
    Thin wrapper around the exec-logger Cloud Function.
    All methods are safe to call even if the CF is unreachable —
    errors are printed as warnings and never raise.
    """

    def __init__(self, run_id: str = None, enabled: bool = True):
        env_flag = os.environ.get("CF_ENABLED", "true").lower()
        self.enabled = enabled and (env_flag not in ("false", "0", "no"))
        self.run_id = run_id or f"run-{datetime.datetime.utcnow().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"

        if self.enabled:
            print(f"  [CFLogger] enabled → run_id: {self.run_id}  url: {FUNC_URL}")
        else:
            print("  [CFLogger] disabled (set CF_ENABLED=true to enable)")

    # ------------------------------------------------------------------
    # Create run
    # ------------------------------------------------------------------
    def create_run(self, program: str = "run_pipeline.py",
                   cmd: str = "", options: dict = None, status: str = "started"):
        """POST /run — register this execution with the CF."""
        if not self.enabled:
            return
        payload = {
            "runId":   self.run_id,
            "program": program,
            "cmd":     cmd,
            "options": options or {},
            "status":  status,
        }
        self._post_json("/run", payload)

    # ------------------------------------------------------------------
    # Log a step (text + optional file attachment)
    # ------------------------------------------------------------------
    def log_step(self, step_num: int, input_text: str = "", output_text: str = "",
                 status: str = "ok", file_path: str = None, file_mime: str = None):
        """
        POST /step — log one pipeline step.
        If file_path is provided, attaches the file as a multipart upload.
        """
        if not self.enabled:
            return

        data = {
            "runId":   self.run_id,
            "stepNum": step_num,
            "input":   str(input_text)[:2000],   # truncate very long strings
            "output":  str(output_text)[:2000],
            "status":  status,
        }

        if file_path and os.path.isfile(file_path):
            fname = os.path.basename(file_path)
            mime  = file_mime or _guess_mime(file_path)
            try:
                with open(file_path, "rb") as fh:
                    self._post_multipart("/step", data, fname, fh, mime)
            except Exception as e:
                print(f"  [CFLogger] WARNING: could not open attachment {file_path}: {e}")
                self._post_json("/step", data)
        else:
            self._post_json("/step", data)

    # ------------------------------------------------------------------
    # Mark run complete
    # ------------------------------------------------------------------
    def complete_run(self, status: str = "complete"):
        """POST /run with status=complete to mark the run finished."""
        if not self.enabled:
            return
        payload = {
            "runId":  self.run_id,
            "status": status,
        }
        self._post_json("/run", payload)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _post_json(self, path: str, payload: dict):
        try:
            r = requests.post(
                f"{FUNC_URL}{path}", json=payload, timeout=CF_TIMEOUT
            )
            if r.status_code >= 400:
                print(f"  [CFLogger] WARNING: {path} returned {r.status_code}: {r.text[:200]}")
            else:
                print(f"  [CFLogger] ✓ {path} → {r.status_code}")
        except Exception as e:
            print(f"  [CFLogger] WARNING: {path} failed: {e}")

    def _post_multipart(self, path: str, data: dict, fname: str, fh, mime: str):
        try:
            r = requests.post(
                f"{FUNC_URL}{path}",
                data=data,
                files={"file": (fname, fh, mime)},
                timeout=CF_TIMEOUT,
            )
            if r.status_code >= 400:
                print(f"  [CFLogger] WARNING: {path} (multipart) returned {r.status_code}: {r.text[:200]}")
            else:
                print(f"  [CFLogger] ✓ {path} (+ {fname}) → {r.status_code}")
        except Exception as e:
            print(f"  [CFLogger] WARNING: {path} multipart failed: {e}")


def _guess_mime(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    return {
        ".png":  "image/png",
        ".jpg":  "image/jpeg",
        ".jpeg": "image/jpeg",
        ".csv":  "text/csv",
        ".tsv":  "text/tab-separated-values",
        ".json": "application/json",
        ".npz":  "application/octet-stream",
        ".npy":  "application/octet-stream",
        ".pkl":  "application/octet-stream",
        ".txt":  "text/plain",
    }.get(ext, "application/octet-stream")
