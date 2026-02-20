from __future__ import annotations

import os
import shlex
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any


def _truthy(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _tail(text: str, max_chars: int = 600) -> str:
    t = (text or "").strip()
    if len(t) <= max_chars:
        return t
    return t[-max_chars:]


def backup_database_to_hpc(*, trigger: str = "scrape", ingested_count: int = 0) -> dict[str, Any]:
    """Create pg_dump backup and copy to HPC via scp.

    The backup file is created in a temp local directory and removed by default
    after successful transfer.
    """
    started = time.perf_counter()

    enabled = _truthy(os.environ.get("ENABLE_POST_SCRAPE_HPC_BACKUP"), default=False)
    if not enabled:
        return {"attempted": False, "ok": False, "reason": "disabled"}

    if ingested_count <= 0 and not _truthy(os.environ.get("BACKUP_ON_EMPTY_SCRAPE"), default=False):
        return {"attempted": False, "ok": False, "reason": "no_new_ingest"}

    database_url = (os.environ.get("DATABASE_URL") or "").strip()
    hpc_target = (os.environ.get("DB_BACKUP_HPC_TARGET") or "").strip()
    if not database_url:
        return {"attempted": False, "ok": False, "reason": "missing_database_url"}
    if not hpc_target:
        return {"attempted": False, "ok": False, "reason": "missing_hpc_target"}

    tmp_dir = Path((os.environ.get("DB_BACKUP_TMP_DIR") or "/tmp/slack-rag-agent-backups").strip())
    tmp_dir.mkdir(parents=True, exist_ok=True)

    prefix = (os.environ.get("DB_BACKUP_FILENAME_PREFIX") or "ragdb").strip() or "ragdb"
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{ts}.dump"
    local_path = tmp_dir / filename

    pg_dump_bin = (os.environ.get("DB_BACKUP_PG_DUMP_BIN") or "pg_dump").strip() or "pg_dump"
    scp_bin = (os.environ.get("DB_BACKUP_SCP_BIN") or "scp").strip() or "scp"
    ssh_opts = shlex.split((os.environ.get("DB_BACKUP_SSH_OPTS") or "").strip())

    dump_cmd = [pg_dump_bin, database_url, "-Fc", "-f", str(local_path)]
    dump = subprocess.run(dump_cmd, capture_output=True, text=True)
    if dump.returncode != 0:
        return {
            "attempted": True,
            "ok": False,
            "stage": "pg_dump",
            "error": _tail(dump.stderr or dump.stdout),
            "duration_ms": round((time.perf_counter() - started) * 1000, 2),
        }

    file_size = int(local_path.stat().st_size) if local_path.exists() else 0

    destination = hpc_target.rstrip("/") + "/" + filename
    scp_cmd = [scp_bin, *ssh_opts, str(local_path), destination]
    scp = subprocess.run(scp_cmd, capture_output=True, text=True)
    if scp.returncode != 0:
        return {
            "attempted": True,
            "ok": False,
            "stage": "scp",
            "error": _tail(scp.stderr or scp.stdout),
            "local_path": str(local_path),
            "remote_path": destination,
            "bytes": file_size,
            "duration_ms": round((time.perf_counter() - started) * 1000, 2),
        }

    keep_local = _truthy(os.environ.get("DB_BACKUP_KEEP_LOCAL"), default=False)
    if not keep_local:
        try:
            local_path.unlink(missing_ok=True)
        except Exception:
            pass

    return {
        "attempted": True,
        "ok": True,
        "trigger": trigger,
        "remote_path": destination,
        "bytes": file_size,
        "local_retained": bool(keep_local),
        "duration_ms": round((time.perf_counter() - started) * 1000, 2),
    }
