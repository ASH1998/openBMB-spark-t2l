"""Budget guard for paid compute (Modal).

Every paid run reserves money here first. Reservations are refused until the
user has confirmed the cap (config/budget.json: confirmed_by_user), and when a
reservation would exceed the stage cap, the project cap, or the single-run cap.

Usage:
    from minicpm_t2l.budget import Budget
    b = Budget()
    rid = b.reserve("gate0", 5.0, run="gate0-base-eval")
    try:
        ...  # paid work
        b.finish(rid, "completed", actual_usd=3.2)
    except BaseException as e:
        b.finish(rid, "failed", error_type=type(e).__name__)
        raise

Spend for a stage counts each reservation's actual_usd once known, else its
reserved_usd (released reservations count zero).
"""

from __future__ import annotations

import json
import math
import os
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class BudgetError(RuntimeError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Budget:
    def __init__(self, root: Path | str = ROOT):
        self.root = Path(root)
        self.policy_path = self.root / "config" / "budget.json"
        self.ledger_path = self.root / "budget" / "ledger.json"
        self.lock_path = self.root / "budget" / ".ledger.lock"

    # -- storage -----------------------------------------------------------
    def policy(self) -> dict:
        return json.loads(self.policy_path.read_text(encoding="utf-8"))

    def _read(self) -> dict:
        if not self.ledger_path.exists():
            return {"schema_version": 1, "reservations": []}
        return json.loads(self.ledger_path.read_text(encoding="utf-8"))

    def _write(self, ledger: dict) -> None:
        tmp = self.ledger_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(ledger, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, self.ledger_path)

    @contextmanager
    def _lock(self, timeout: float = 30.0):
        deadline = time.monotonic() + timeout
        while True:
            try:
                fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                break
            except FileExistsError:
                if time.monotonic() > deadline:
                    raise BudgetError(f"ledger lock held: {self.lock_path} (delete it if no run is active)")
                time.sleep(0.1)
        try:
            yield
        finally:
            os.close(fd)
            os.remove(self.lock_path)

    # -- accounting --------------------------------------------------------
    @staticmethod
    def _cost(r: dict) -> float:
        if r["status"] == "released":
            return 0.0
        if r.get("actual_usd") is not None:
            return float(r["actual_usd"])
        return float(r["reserved_usd"])

    def spent(self, stage: str | None = None) -> float:
        rs = self._read()["reservations"]
        return sum(self._cost(r) for r in rs if stage is None or r["stage"] == stage)

    def summary(self) -> dict:
        p = self.policy()
        return {
            "confirmed": bool(p.get("confirmed_by_user")),
            "project": {"cap": p["project_cap_usd"], "spent_or_held": round(self.spent(), 2)},
            "stages": {s: {"cap": c, "spent_or_held": round(self.spent(s), 2)} for s, c in p["stage_caps_usd"].items()},
        }

    # -- operations --------------------------------------------------------
    def reserve(self, stage: str, usd: float, *, run: str = "", note: str = "") -> str:
        p = self.policy()
        if not p.get("confirmed_by_user"):
            raise BudgetError("budget cap not confirmed by the user; no paid compute allowed")
        if type(usd) not in (int, float) or not math.isfinite(usd) or usd <= 0:
            raise BudgetError("reservation must be a positive finite amount")
        if usd > p["max_single_reservation_usd"]:
            raise BudgetError(f"reservation {usd} exceeds single-run cap {p['max_single_reservation_usd']}")
        caps = p["stage_caps_usd"]
        if stage not in caps:
            raise BudgetError(f"unknown stage {stage!r}; known: {sorted(caps)}")
        with self._lock():
            ledger = self._read()
            rs = ledger["reservations"]
            stage_spent = sum(self._cost(r) for r in rs if r["stage"] == stage)
            total = sum(self._cost(r) for r in rs)
            if stage_spent + usd > caps[stage] + 1e-9:
                raise BudgetError(f"stage {stage!r} cap {caps[stage]} would be exceeded ({stage_spent:.2f} used)")
            if total + usd > p["project_cap_usd"] + 1e-9:
                raise BudgetError(f"project cap {p['project_cap_usd']} would be exceeded ({total:.2f} used)")
            rid = uuid.uuid4().hex
            rs.append({
                "id": rid, "stage": stage, "run": run, "note": note,
                "reserved_usd": float(usd), "actual_usd": None,
                "status": "reserved", "started_at": _now(), "finished_at": None, "details": {},
            })
            self._write(ledger)
        return rid

    def finish(self, rid: str, status: str, *, actual_usd: float | None = None, **details) -> None:
        if status not in {"completed", "failed", "released"}:
            raise BudgetError(f"bad status {status!r}")
        with self._lock():
            ledger = self._read()
            for r in ledger["reservations"]:
                if r["id"] == rid:
                    if r["status"] != "reserved":
                        raise BudgetError(f"reservation {rid} already {r['status']}")
                    r.update(status=status, finished_at=_now())
                    if actual_usd is not None:
                        r["actual_usd"] = float(actual_usd)
                    r["details"].update(details)
                    self._write(ledger)
                    return
        raise BudgetError(f"no reservation {rid}")


if __name__ == "__main__":
    print(json.dumps(Budget().summary(), indent=2))
