"""Offline tests for persistent Bedrock Converse limits."""

import pytest

from planner_app.live_quota import (
    LiveQuotaExceeded,
    LocallyCappedBedrock,
)


class FakeBedrock:
    def __init__(self):
        self.calls = []

    def converse(self, **kwargs):
        self.calls.append(kwargs)
        return {"usage": {"inputTokens": 10}}


def test_quota_does_not_call_bedrock_on_initialization(tmp_path):
    fake = FakeBedrock()
    quota_path = tmp_path / "quota"

    LocallyCappedBedrock(fake, quota_path)

    assert fake.calls == []
    assert not quota_path.exists()


def test_quota_blocks_after_limit(tmp_path):
    fake = FakeBedrock()
    client = LocallyCappedBedrock(
        fake, tmp_path / "quota", max_calls=2
    )

    client.converse(modelId="test")
    client.converse(modelId="test")

    with pytest.raises(LiveQuotaExceeded):
        client.converse(modelId="test")

    assert len(fake.calls) == 2


def test_quota_persists_across_client_instances(tmp_path):
    fake = FakeBedrock()
    path = tmp_path / "quota"

    first = LocallyCappedBedrock(fake, path, max_calls=1)
    first.converse(modelId="test")

    second = LocallyCappedBedrock(fake, path, max_calls=1)

    with pytest.raises(LiveQuotaExceeded):
        second.converse(modelId="test")

    assert path.read_text(encoding="utf-8") == "1"
    assert len(fake.calls) == 1


def test_corrupted_quota_fails_closed(tmp_path):
    fake = FakeBedrock()
    path = tmp_path / "quota"
    path.write_text("unknown", encoding="utf-8")

    client = LocallyCappedBedrock(fake, path)

    with pytest.raises(ValueError, match="Invalid"):
        client.converse(modelId="test")

    assert fake.calls == []


def test_failed_converse_still_consumes_quota(tmp_path):
    class FailingBedrock:
        def converse(self, **kwargs):
            raise RuntimeError("Simulated provider failure")

    path = tmp_path / "quota"
    client = LocallyCappedBedrock(
        FailingBedrock(), path, max_calls=1
    )

    with pytest.raises(RuntimeError, match="provider failure"):
        client.converse(modelId="test")

    assert path.read_text(encoding="utf-8") == "1"

    with pytest.raises(LiveQuotaExceeded):
        client.converse(modelId="test")
