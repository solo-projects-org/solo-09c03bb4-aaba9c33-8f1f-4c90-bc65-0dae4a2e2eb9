"""Blind-review rounds: open, submit judgments, query (database layer).

Atomicity rules:
  * opening a round inserts the round and all of its entries in one
    ``BEGIN IMMEDIATE`` transaction — a failure leaves no half round;
  * a judgment batch is validated in full and applied in one transaction —
    a wrong-round entry or a conflicting verdict rejects the whole batch.

Rounds snapshot their entries (sample, spectrum, peptide, score, label) at
open time, so a candidate revision change or a process restart never affects
submissions for rounds that are still collecting answers.
"""

from __future__ import annotations

import random
import sqlite3
from pathlib import Path

from . import db, review, service
from .compute import SpectrumKey, decimal_text
from .review import JudgmentRow


class SubmitResult:
    def __init__(
        self,
        batch_id: int,
        received: int,
        unique_entries: int,
        applied: int,
        judged: int,
        total: int,
        completed: bool,
    ) -> None:
        self.batch_id = batch_id
        self.received = received
        self.unique_entries = unique_entries
        self.applied = applied
        self.judged = judged
        self.total = total
        self.completed = completed

    @property
    def message(self) -> str:
        text = (
            f"本批接收 {self.received} 行（去重后 {self.unique_entries} 条），"
            f"新录入 {self.applied} 条，同判重送 "
            f"{self.unique_entries - self.applied} 条已忽略。"
            f"当前进度 {self.judged}/{self.total}。"
        )
        if self.completed:
            text += " 本轮答卷已收齐，靶标/诱饵标签与统计已解锁。"
        return text


def _current_candidate_source(conn: sqlite3.Connection) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT s.* FROM current_sources c JOIN sources s ON s.id = c.source_id "
        "WHERE c.kind = 'candidates'"
    ).fetchone()


def _used_keys(conn: sqlite3.Connection, candidate_source_id: int) -> set[SpectrumKey]:
    rows = conn.execute(
        "SELECT e.sample, e.spectrum FROM review_entries e "
        "JOIN review_rounds r ON r.id = e.round_id "
        "WHERE r.candidate_source_id = ?",
        (candidate_source_id,),
    ).fetchall()
    # Entries from every round of this candidate revision count, including
    # rounds that have not collected all answers yet.
    return {SpectrumKey(row["sample"], row["spectrum"]) for row in rows}


def open_round(db_path: str | Path, rng: random.Random | None = None) -> int:
    """Open a blind-review round from the current candidate list.

    Returns the new round id.  Raises ``ServiceError`` (leaving no data
    behind) when no candidate list is present or no target+decoy pair can
    be formed from the remaining spectra.
    """
    rng = rng or random.SystemRandom()
    with db.write_tx(db_path) as conn:
        source = _current_candidate_source(conn)
        if source is None:
            raise service.ServiceError(
                "尚未上传现行候选清单，无法开轮；请先在工作台上传候选。"
            )
        source_id = int(source["id"])
        candidates = service._load_candidates(conn, source_id)
        plan = review.plan_round(candidates, _used_keys(conn, source_id), rng)
        if plan.pair_count < 1:
            raise service.ServiceError(
                "无法开轮：现行候选修订"
                f"（修订号 {source['revision']}）中剩余可抽的唯一最高分"
                f"靶标 {plan.pool_targets} 条、诱饵 {plan.pool_decoys} 条，"
                "凑不出一对靶标+诱饵。"
            )
        numbers = review.random_entry_numbers(len(plan.chosen), rng)
        now = db.now_iso()
        cur = conn.execute(
            "INSERT INTO review_rounds (candidate_source_id, pair_count, "
            "status, created_at) VALUES (?, ?, 'open', ?)",
            (source_id, plan.pair_count, now),
        )
        round_id = int(cur.lastrowid)
        conn.executemany(
            "INSERT INTO review_entries (round_id, entry_no, sample, "
            "spectrum, peptide, label, score) VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    round_id,
                    entry_no,
                    candidate.key.sample,
                    candidate.key.spectrum,
                    candidate.peptide,
                    candidate.label,
                    decimal_text(candidate.score),
                )
                for entry_no, candidate in zip(numbers, plan.chosen)
            ],
        )
        return round_id


def _validate_batch(
    rows: list[JudgmentRow],
    entries_by_no: dict[int, sqlite3.Row],
    recorded: dict[int, str],
    round_id: int,
) -> dict[int, str]:
    """Return {entry_no: verdict} or raise — nothing is written on error."""
    seen: dict[int, str] = {}
    for row in rows:
        if row.entry_no not in entries_by_no:
            raise service.ServiceError(
                f"条目号 {row.entry_no} 不属于第 {round_id} 轮（错轮条目），"
                "整批拒收。"
            )
        previous = seen.get(row.entry_no)
        if previous is not None and previous != row.verdict:
            raise service.ServiceError(
                f"条目号 {row.entry_no} 在本批内出现互相冲突的判定，整批拒收。"
            )
        seen[row.entry_no] = row.verdict
    for entry_no, verdict in seen.items():
        entry_id = int(entries_by_no[entry_no]["id"])
        existing = recorded.get(entry_id)
        if existing is not None and existing != verdict:
            raise service.ServiceError(
                f"条目号 {entry_no} 与已记录判定冲突"
                f"（已记录：{review.VERDICT_LABELS[existing]}，"
                f"本批：{review.VERDICT_LABELS[verdict]}），整批拒收。"
            )
    return seen


def submit_judgments(
    db_path: str | Path, round_id: int, data: bytes, filename: str
) -> SubmitResult:
    """Apply one judgment batch to a round, atomically.

    Re-sent identical judgments count once; a wrong-round entry number or a
    verdict conflicting with a recorded one rejects the entire batch.
    """
    rows = review.parse_judgments(data, filename)
    with db.write_tx(db_path) as conn:
        round_row = conn.execute(
            "SELECT * FROM review_rounds WHERE id = ?", (round_id,)
        ).fetchone()
        if round_row is None:
            raise service.NotFoundError(f"盲审轮次 {round_id} 不存在")
        entries = conn.execute(
            "SELECT * FROM review_entries WHERE round_id = ?", (round_id,)
        ).fetchall()
        entries_by_no = {int(row["entry_no"]): row for row in entries}
        recorded = {
            int(row["entry_id"]): row["verdict"]
            for row in conn.execute(
                "SELECT j.entry_id, j.verdict FROM review_judgments j "
                "JOIN review_entries e ON e.id = j.entry_id "
                "WHERE e.round_id = ?",
                (round_id,),
            )
        }
        seen = _validate_batch(rows, entries_by_no, recorded, round_id)

        now = db.now_iso()
        cur = conn.execute(
            "INSERT INTO review_batches (round_id, received, applied, "
            "created_at) VALUES (?, ?, 0, ?)",
            (round_id, len(rows), now),
        )
        batch_id = int(cur.lastrowid)
        applied = 0
        for entry_no, verdict in seen.items():
            entry_id = int(entries_by_no[entry_no]["id"])
            if entry_id in recorded:
                continue  # identical re-send: counts once
            conn.execute(
                "INSERT INTO review_judgments (entry_id, verdict, batch_id, "
                "created_at) VALUES (?, ?, ?, ?)",
                (entry_id, verdict, batch_id, now),
            )
            applied += 1
        conn.execute(
            "UPDATE review_batches SET applied = ? WHERE id = ?",
            (applied, batch_id),
        )

        total = len(entries)
        judged = conn.execute(
            "SELECT COUNT(*) AS n FROM review_judgments j "
            "JOIN review_entries e ON e.id = j.entry_id WHERE e.round_id = ?",
            (round_id,),
        ).fetchone()["n"]
        completed = False
        if judged == total:
            conn.execute(
                "UPDATE review_rounds SET status = 'completed', "
                "completed_at = ? WHERE id = ? AND status = 'open'",
                (now, round_id),
            )
            completed = True
        return SubmitResult(
            batch_id=batch_id,
            received=len(rows),
            unique_entries=len(seen),
            applied=applied,
            judged=judged,
            total=total,
            completed=completed,
        )


def _progress(conn: sqlite3.Connection, round_id: int) -> tuple[int, int]:
    row = conn.execute(
        "SELECT "
        "(SELECT COUNT(*) FROM review_entries WHERE round_id = ?) AS total, "
        "(SELECT COUNT(*) FROM review_judgments j "
        " JOIN review_entries e ON e.id = j.entry_id "
        " WHERE e.round_id = ?) AS judged",
        (round_id, round_id),
    ).fetchone()
    return int(row["judged"]), int(row["total"])


def list_rounds(conn: sqlite3.Connection) -> list[dict[str, object]]:
    rounds = conn.execute(
        "SELECT r.*, s.revision AS candidate_revision FROM review_rounds r "
        "JOIN sources s ON s.id = r.candidate_source_id "
        "ORDER BY r.id DESC"
    ).fetchall()
    items: list[dict[str, object]] = []
    for row in rounds:
        judged, total = _progress(conn, int(row["id"]))
        items.append(
            {
                "id": row["id"],
                "candidate_revision": row["candidate_revision"],
                "pair_count": row["pair_count"],
                "status": row["status"],
                "created_at": row["created_at"],
                "completed_at": row["completed_at"],
                "judged": judged,
                "total": total,
            }
        )
    return items


def current_candidate_info(conn: sqlite3.Connection) -> dict[str, object] | None:
    source = _current_candidate_source(conn)
    if source is None:
        return None
    return {
        "revision": source["revision"],
        "filename": source["filename"],
        "row_count": source["row_count"],
    }


def get_round_detail(
    conn: sqlite3.Connection, round_id: int
) -> dict[str, object] | None:
    """Round detail; target/decoy labels are included only after completion."""
    row = conn.execute(
        "SELECT r.*, s.revision AS candidate_revision FROM review_rounds r "
        "JOIN sources s ON s.id = r.candidate_source_id WHERE r.id = ?",
        (round_id,),
    ).fetchone()
    if row is None:
        return None
    completed = row["status"] == "completed"
    judged, total = _progress(conn, round_id)
    entry_rows = conn.execute(
        "SELECT e.*, j.verdict FROM review_entries e "
        "LEFT JOIN review_judgments j ON j.entry_id = e.id "
        "WHERE e.round_id = ? ORDER BY e.entry_no",
        (round_id,),
    ).fetchall()
    entries = []
    for entry in entry_rows:
        item: dict[str, object] = {
            "entry_no": entry["entry_no"],
            "sample": entry["sample"],
            "spectrum": entry["spectrum"],
            "peptide": entry["peptide"],
            "score": entry["score"],
            "answered": entry["verdict"] is not None,
        }
        if completed:
            item["label"] = entry["label"]
            item["verdict"] = entry["verdict"]
        entries.append(item)

    summary = None
    if completed:
        stats = conn.execute(
            "SELECT e.label AS label, j.verdict AS verdict, COUNT(*) AS n "
            "FROM review_entries e "
            "JOIN review_judgments j ON j.entry_id = e.id "
            "WHERE e.round_id = ? GROUP BY e.label, j.verdict",
            (round_id,),
        ).fetchall()
        counts = {(s["label"], s["verdict"]): s["n"] for s in stats}
        summary = {
            "target_pass": counts.get(("target", review.PASS), 0),
            "target_total": counts.get(("target", review.PASS), 0)
            + counts.get(("target", review.FAIL), 0),
            "decoy_pass": counts.get(("decoy", review.PASS), 0),
            "decoy_total": counts.get(("decoy", review.PASS), 0)
            + counts.get(("decoy", review.FAIL), 0),
        }

    return {
        "id": row["id"],
        "candidate_revision": row["candidate_revision"],
        "pair_count": row["pair_count"],
        "status": row["status"],
        "created_at": row["created_at"],
        "completed_at": row["completed_at"],
        "judged": judged,
        "total": total,
        "completed": completed,
        "entries": entries,
        "summary": summary,
    }


def sheet_rows(
    db_path: str | Path, round_id: int, rng: random.Random | None = None
) -> list[sqlite3.Row]:
    """Blind sheet rows in random order; labels are never selected."""
    rng = rng or random.SystemRandom()
    with db.read_conn(db_path) as conn:
        round_row = conn.execute(
            "SELECT id FROM review_rounds WHERE id = ?", (round_id,)
        ).fetchone()
        if round_row is None:
            raise service.NotFoundError(f"盲审轮次 {round_id} 不存在")
        rows = conn.execute(
            "SELECT entry_no, sample, spectrum, peptide, score "
            "FROM review_entries WHERE round_id = ?",
            (round_id,),
        ).fetchall()
    return [row for row in review.shuffled_rows(rows, rng)]
