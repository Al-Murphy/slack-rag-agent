#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys

from dotenv import load_dotenv

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from database.backup import backup_database_to_hpc


def main() -> None:
    load_dotenv()
    result = backup_database_to_hpc(trigger="manual", ingested_count=1)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
