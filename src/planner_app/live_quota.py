"""Persistent local Converse quota for controlled application testing."""

import fcntl
import os
from pathlib import Path


class LiveQuotaExceeded(RuntimeError):
    """No more Bedrock Converse calls are authorized."""


class LocallyCappedBedrock:
    """Reserve a persistent quota slot before each Converse attempt."""

    def __init__(self, client, quota_path, max_calls=12):
        if type(max_calls) is not int or max_calls < 1:
            raise ValueError("max_calls must be a positive integer.")

        self.client = client
        self.quota_path = Path(quota_path)
        self.max_calls = max_calls

    def _reserve_call(self):
        self.quota_path.parent.mkdir(parents=True, exist_ok=True)

        with self.quota_path.open("a+", encoding="utf-8") as file:
            fcntl.flock(file.fileno(), fcntl.LOCK_EX)
            file.seek(0)
            content = file.read().strip()

            if content and not content.isdigit():
                raise ValueError("Invalid persisted Bedrock quota.")

            used = int(content) if content else 0

            if used >= self.max_calls:
                raise LiveQuotaExceeded(
                    "Local Bedrock Converse quota exhausted."
                )

            file.seek(0)
            file.truncate()
            file.write(str(used + 1))
            file.flush()
            os.fsync(file.fileno())

    def converse(self, **kwargs):
        self._reserve_call()
        return self.client.converse(**kwargs)
