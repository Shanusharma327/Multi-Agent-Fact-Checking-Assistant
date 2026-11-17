"""Tests for the single-instance run lock.

Two benchmark processes once ran concurrently in this project: they wrote
the same cache and together rate-limited the search host into refusing
every request, with no symptom beyond retrieval quietly ceasing to
progress. The lock exists so that cannot recur, and these tests pin both
halves of the contract -- a live run blocks a second, and a dead run does
not block anything.
"""

from __future__ import annotations

import json
import os

import pytest

from fact_checker.locking import LockHeld, RunLock


class TestAcquire:
    def test_a_lock_can_be_taken(self, tmp_path):
        lock = RunLock(tmp_path / "scope", name="t").acquire()
        assert lock.path.exists()
        lock.release()

    def test_it_records_who_holds_it(self, tmp_path):
        lock = RunLock(tmp_path / "scope", name="t").acquire()
        payload = json.loads(lock.path.read_text(encoding="utf-8"))
        assert payload["pid"] == os.getpid()
        assert payload["started"]
        lock.release()

    def test_a_second_acquire_is_refused(self, tmp_path):
        first = RunLock(tmp_path / "scope", name="t").acquire()
        with pytest.raises(LockHeld, match="Another run already holds"):
            RunLock(tmp_path / "scope", name="t").acquire()
        first.release()

    def test_the_refusal_explains_the_consequence(self, tmp_path):
        """A bare 'locked' message invites someone to delete the file."""
        first = RunLock(tmp_path / "scope", name="t").acquire()
        with pytest.raises(LockHeld, match="corrupt each other"):
            RunLock(tmp_path / "scope", name="t").acquire()
        first.release()

    def test_different_scopes_do_not_block_each_other(self, tmp_path):
        a = RunLock(tmp_path / "one", name="t").acquire()
        b = RunLock(tmp_path / "two", name="t").acquire()
        assert a.path != b.path
        a.release()
        b.release()

    def test_different_names_in_one_scope_do_not_block(self, tmp_path):
        """Retrieval and the conditions stage lock separately."""
        a = RunLock(tmp_path / "scope", name="retrieve").acquire()
        b = RunLock(tmp_path / "scope", name="conditions").acquire()
        a.release()
        b.release()


class TestStaleLocks:
    def test_a_dead_owner_does_not_block(self, tmp_path):
        lock = RunLock(tmp_path / "scope", name="t")
        lock.path.parent.mkdir(parents=True, exist_ok=True)
        lock.path.write_text(
            json.dumps({"pid": 999_999, "started": "2020-01-01", "command": "x"}),
            encoding="utf-8",
        )
        taken = RunLock(tmp_path / "scope", name="t").acquire()
        assert json.loads(taken.path.read_text(encoding="utf-8"))["pid"] == os.getpid()
        taken.release()

    def test_an_unreadable_lock_does_not_block(self, tmp_path):
        lock = RunLock(tmp_path / "scope", name="t")
        lock.path.parent.mkdir(parents=True, exist_ok=True)
        lock.path.write_text("not json at all", encoding="utf-8")
        taken = RunLock(tmp_path / "scope", name="t").acquire()
        taken.release()

    def test_force_overrides_a_live_lock(self, tmp_path):
        first = RunLock(tmp_path / "scope", name="t").acquire()
        second = RunLock(tmp_path / "scope", name="t", force=True).acquire()
        second.release()
        first.release()


class TestRelease:
    def test_release_removes_the_file(self, tmp_path):
        lock = RunLock(tmp_path / "scope", name="t").acquire()
        path = lock.path
        lock.release()
        assert not path.exists()

    def test_releasing_twice_is_safe(self, tmp_path):
        lock = RunLock(tmp_path / "scope", name="t").acquire()
        lock.release()
        lock.release()

    def test_a_lock_never_releases_another_process_s_claim(self, tmp_path):
        """Otherwise a crashing run would unlock the run that replaced it."""
        lock = RunLock(tmp_path / "scope", name="t").acquire()
        lock.path.write_text(
            json.dumps({"pid": 999_999, "started": "x", "command": "y"}),
            encoding="utf-8",
        )
        lock.release()
        assert lock.path.exists()

    def test_releasing_without_acquiring_is_safe(self, tmp_path):
        RunLock(tmp_path / "scope", name="t").release()


class TestContextManager:
    def test_the_lock_is_held_inside_and_freed_after(self, tmp_path):
        with RunLock(tmp_path / "scope", name="t") as lock:
            assert lock.path.exists()
            with pytest.raises(LockHeld):
                RunLock(tmp_path / "scope", name="t").acquire()
        assert not lock.path.exists()

    def test_an_exception_still_frees_the_lock(self, tmp_path):
        lock = RunLock(tmp_path / "scope", name="t")
        with pytest.raises(ValueError):
            with lock:
                raise ValueError("boom")
        assert not lock.path.exists()
