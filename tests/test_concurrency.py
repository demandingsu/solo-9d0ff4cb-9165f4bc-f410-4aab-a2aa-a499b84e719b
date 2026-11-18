"""并发测试：多线程同时提交分片/变更，验证写事务串行化、幂等与序号约束。"""
from __future__ import annotations

import os
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.models import ChangeRequest, OpenBatchRequest, OutletItem, PublishRequest, ShardSubmitRequest


@pytest.fixture()
def svc(monkeypatch):
    tmpdir = tempfile.mkdtemp(prefix="discharge-conc-")
    db_path = os.path.join(tmpdir, "conc.db")
    monkeypatch.setenv("DB_PATH", db_path)

    from app import database
    from app import service

    monkeypatch.setattr(database, "DB_PATH", db_path)
    database.init_db(db_path)
    return service


def _open_and_publish_v1(svc):
    bid = svc.open_batch(OpenBatchRequest(total_shards=1, source_cutoff_seq=0))["batch_id"]
    svc.submit_shard(
        bid,
        ShardSubmitRequest(shard_no=1, items=[
            OutletItem(outlet_no="O1", district="A", active=True),
        ]),
    )
    svc.publish_batch(bid, PublishRequest(catch_up_seq=0))
    return bid


def test_concurrent_shard_distinct_numbers(svc):
    bid = svc.open_batch(OpenBatchRequest(total_shards=50, source_cutoff_seq=0))["batch_id"]

    def submit(i):
        return svc.submit_shard(
            bid,
            ShardSubmitRequest(shard_no=i, items=[
                OutletItem(outlet_no=f"O{i}", district="A", active=True),
            ]),
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(submit, range(1, 51)))
    assert all(r["total_shards"] == 50 for r in results)

    status = svc.current_batch_status()
    assert status["missing_shards"] == [] and len(status["received_shards"]) == 50


def test_concurrent_duplicate_shard_retry(svc):
    bid = svc.open_batch(OpenBatchRequest(total_shards=1, source_cutoff_seq=0))["batch_id"]
    req = ShardSubmitRequest(shard_no=1, items=[
        OutletItem(outlet_no="O1", district="A", active=True),
    ])

    outcomes = []

    def submit(_):
        try:
            outcomes.append(svc.submit_shard(bid, req))
        except Exception as exc:  # noqa: BLE001
            outcomes.append(exc)

    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(submit, range(6)))

    # 恰好一次生效，其余全部幂等命中
    fresh = [o for o in outcomes if isinstance(o, dict) and o.get("idempotent") is False]
    idem = [o for o in outcomes if isinstance(o, dict) and o.get("idempotent") is True]
    assert len(fresh) == 1 and len(idem) == 5
    assert svc.current_batch_status()["received_shards"] == [1]


def test_concurrent_changes_respect_sequence(svc):
    _open_and_publish_v1(svc)

    def make_req(seq):
        return ChangeRequest(seq=seq, change_type="deactivate", outlet_no="O1")

    # 打乱顺序并发提交 1..40：事务串行化，成功的必然是从 1 开始的连续前缀，其余全部 SEQ_GAP
    errors, oks = [], []

    def submit(seq):
        try:
            oks.append(svc.submit_change(make_req(seq)))
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    with ThreadPoolExecutor(max_workers=10) as pool:
        list(pool.map(submit, range(1, 41)))

    ok_seqs = sorted(o["seq"] for o in oks)
    assert ok_seqs == list(range(1, len(ok_seqs) + 1))  # 无缺号、无重复
    assert all(e.code == "SEQ_GAP" for e in errors)

    # 顺序补齐剩余序号后无缺号
    for seq in range(len(ok_seqs) + 1, 41):
        r = svc.submit_change(make_req(seq))
        assert r["idempotent"] is False
    assert svc.list_changes(limit=1000, offset=0)["missing_seqs"] == []


def test_concurrent_open_batch_single_winner(svc):
    req = OpenBatchRequest(total_shards=1, source_cutoff_seq=0)
    errors, oks = [], []

    def open_(_):
        try:
            oks.append(svc.open_batch(req))
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(open_, range(8)))

    assert len(oks) == 1
    assert len(errors) == 7 and all(e.code == "OPEN_BATCH_EXISTS" for e in errors)


def _publish_cutoff2_with_stale_block(svc):
    """v1 发布后制造旧断点 seq=1 与暂挂 seq=2，返回待发布的 cutoff=2 批次。"""
    _open_and_publish_v1(svc)
    svc.submit_change(ChangeRequest(seq=1, change_type="move", outlet_no="X", district="Z"))
    svc.submit_change(ChangeRequest(seq=2, change_type="add", outlet_no="Y",
                                     district="B", active=True))
    bid = svc.open_batch(OpenBatchRequest(total_shards=1, source_cutoff_seq=2))["batch_id"]
    svc.submit_shard(bid, ShardSubmitRequest(shard_no=1, items=[
        OutletItem(outlet_no="O1", district="A", active=True),
        OutletItem(outlet_no="X", district="Z", active=True),
        OutletItem(outlet_no="Y", district="B", active=True),
    ]))
    return bid


def test_concurrent_publish_same_batch_single_winner(svc):
    bid = _publish_cutoff2_with_stale_block(svc)

    outcomes = []

    def do_publish(_):
        try:
            outcomes.append(svc.publish_batch(bid, PublishRequest(catch_up_seq=2)))
        except Exception as exc:  # noqa: BLE001
            outcomes.append(exc)

    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(do_publish, range(6)))

    oks = [o for o in outcomes if isinstance(o, dict)]
    errs = [o for o in outcomes if isinstance(o, Exception)]
    assert len(oks) == 1 and len(errs) == 5
    assert all(e.code == "NO_OPEN_BATCH" for e in errs)
    assert svc.health() == {"status": "ok", "version": 2, "last_served_seq": 2}
    changes = {c["seq"]: c for c in svc.list_changes(limit=1000, offset=0)["items"]}
    assert changes[1]["applied"] is True and changes[1]["blocked"] is False
    assert changes[2]["applied"] is True and changes[2]["blocked"] is False
    assert svc.stats()["total"] == 3


def test_change_arriving_during_publish_is_not_lost(svc, monkeypatch):
    """发布事务进行中到达的变更：被写事务串行化，发布提交后套用到新版本，不丢不越号。"""
    bid = _publish_cutoff2_with_stale_block(svc)
    in_publish = threading.Event()
    real = svc._reconcile_snapshot_covered
    result = {}

    def slow_reconcile(conn, version, cutoff):
        in_publish.set()
        # 让并发变更先进入 BEGIN IMMEDIATE 等待，验证其被发布事务串行化
        threading.Event().wait(0.3)
        return real(conn, version, cutoff)

    monkeypatch.setattr(svc, "_reconcile_snapshot_covered", slow_reconcile)

    def do_publish():
        result["publish"] = svc.publish_batch(bid, PublishRequest(catch_up_seq=2))

    def do_change():
        in_publish.wait(timeout=5)
        result["change"] = svc.submit_change(
            ChangeRequest(seq=3, change_type="deactivate", outlet_no="X")
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        f1 = pool.submit(do_publish)
        f2 = pool.submit(do_change)
        f1.result(timeout=30)
        f2.result(timeout=30)

    assert result["publish"]["version"] == 2
    assert result["change"]["applied"] is True and result["change"]["blocked"] is False
    assert svc.health()["last_served_seq"] == 2
    outlets = {o["outlet_no"]: o for o in svc.list_outlets(None, False, 100, 0)["items"]}
    assert outlets["X"]["active"] is False
    # 快照覆盖的旧断点与并发变更一起生效
    changes = {c["seq"]: c for c in svc.list_changes(limit=1000, offset=0)["items"]}
    assert changes[1]["applied"] is True and changes[1]["blocked"] is False
    assert changes[3]["applied"] is True


def test_publish_failure_leaves_no_partial_state(svc, monkeypatch):
    bid = _publish_cutoff2_with_stale_block(svc)
    real = svc._reconcile_snapshot_covered

    def boom(conn, version, cutoff):
        real(conn, version, cutoff)
        raise RuntimeError("simulated write failure after reconcile")

    monkeypatch.setattr(svc, "_reconcile_snapshot_covered", boom)
    with pytest.raises(RuntimeError):
        svc.publish_batch(bid, PublishRequest(catch_up_seq=2))

    # 断点、现役清单、水位、批次状态一起保持旧态
    assert svc.health() == {"status": "ok", "version": 1, "last_served_seq": 0}
    assert svc.stats()["total"] == 1
    changes = {c["seq"]: c for c in svc.list_changes(limit=1000, offset=0)["items"]}
    assert changes[1]["blocked"] is True and changes[1]["applied"] is False
    assert changes[2]["blocked"] is False and changes[2]["applied"] is False
    assert svc.current_batch_status()["batch_id"] == bid

    # 同一批次重试发布成功，状态一起推进
    monkeypatch.setattr(svc, "_reconcile_snapshot_covered", real)
    out = svc.publish_batch(bid, PublishRequest(catch_up_seq=2))
    assert out["version"] == 2
    assert svc.health() == {"status": "ok", "version": 2, "last_served_seq": 2}
    changes = {c["seq"]: c for c in svc.list_changes(limit=1000, offset=0)["items"]}
    assert changes[1]["applied"] is True and changes[1]["blocked"] is False
