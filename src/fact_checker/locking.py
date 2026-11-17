"""Single-instance locking, so two runs cannot write the same files.

Two copies of the benchmark ran concurrently once in this project. Each
wrote the same passage cache and would have written the same per-condition
result files, and together they hammered the search endpoint hard enough
to rate-limit it into total failure. Neither process noticed; the only
symptom was that retrieval stopped making progress.

The lock makes that failure impossible rather than unlikely. It is a
file in the working tree holding the owning process id and start time,
so a stale lock from a crashed run is detectable and clearable instead
of blocking the next run forever.

Deliberately not an OS-level advisory lock: those are inconsistent across
platforms and invisible to a user trying to work out why a run refuses
to start. A readable file that says which process holds it, since when,
and how to clear it, is worth more here than strict correctness against
an adversary -- the thing being defended against is a mistake, not an
attacker.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType
from typing import Optional, Type


class LockHeld(RuntimeError):
    """Raised when another process already holds the lock."""


def _process_alive(pid: int) -> bool:
    """Whether ``pid`` names a running process.

    A lock whose owner has died is stale, and refusing to start because
    of one would be worse than the race it guards against.
    """
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # It exists and belongs to someone else.
        return True
    except OSError:
        return False
    return True


class RunLock:
    """A single-instance lock held for the duration of a run.

    Usage::

        with RunLock(Path("results/av250"), name="tier1"):
            ...

    Args:
        scope: Directory the run writes to. The lock lives beside it, so
            two runs targeting different outputs do not block each other.
        name: Distinguishes locks within one scope.
        force: Take the lock even if a live process holds it. For
            recovering from a lock whose owner is wedged rather than dead.
    """

    def __init__(self, scope: Path, *, name: str = "run", force: bool = False) -> None:
        self.scope = Path(scope)
        self.name = name
        self.force = force
        self.path = self.scope.parent / f".{self.scope.name}.{name}.lock"
        self._taken = False

    def _read(self) -> Optional[dict]:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def acquire(self) -> "RunLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        existing = self._read() if self.path.exists() else None

        if existing and not self.force:
            pid = int(existing.get("pid", 0))
            if _process_alive(pid):
                raise LockHeld(
                    f"Another run already holds {self.path.name}.\n"
                    f"  pid     {pid}\n"
                    f"  started {existing.get('started', 'unknown')}\n"
                    f"  command {existing.get('command', 'unknown')}\n\n"
                    "Two runs writing the same outputs corrupt each other's "
                    "results and rate-limit the search endpoint. Wait for it "
                    "to finish, stop it, or pass --force if you are certain "
                    "it is wedged."
                )
            print(
                f"  clearing a stale lock from pid {pid} "
                f"(started {existing.get('started', 'unknown')}; not running)"
            )

        self.path.write_text(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "started": datetime.now(timezone.utc)
                    .replace(microsecond=0)
                    .isoformat(),
                    "command": " ".join(os.sys.argv[:4]),
                    "scope": str(self.scope),
                },
                indent=1,
            ),
            encoding="utf-8",
        )
        self._taken = True
        return self

    def release(self) -> None:
        """Release the lock, but never another process's lock."""
        if not self._taken:
            return
        existing = self._read()
        if existing and int(existing.get("pid", 0)) == os.getpid():
            self.path.unlink(missing_ok=True)
        self._taken = False

    def __enter__(self) -> "RunLock":
        return self.acquire()

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> None:
        self.release()


__all__ = ["LockHeld", "RunLock"]
