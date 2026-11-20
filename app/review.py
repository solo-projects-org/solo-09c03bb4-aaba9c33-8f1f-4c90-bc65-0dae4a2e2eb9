"""Pure blind-review logic: top-hit selection, sampling and judgment parsing.

This module has no database or web dependencies so the sampling semantics can
be tested in isolation.  A "round" draws at most ``MAX_PAIRS_PER_ROUND`` target
top hits and the same number of decoy top hits from the current candidate
list; spectra already drawn in any round of the same candidate revision are
ineligible.
"""

from __future__ import annotations

import csv
import io
import random
from dataclasses import dataclass
from typing import Iterable, Sequence

from .compute import DECOY, TARGET, Candidate, SpectrumKey
from .csvio import ParseError, _cell, _decode, _map_header

MAX_PAIRS_PER_ROUND = 10

# Entry numbers are drawn from a large space (not 1..2n) so that a judgment
# CSV accidentally uploaded to the wrong round is reliably detected: its
# entry numbers will not belong to that round's set.
ENTRY_NO_SPACE = 1_000_000

PASS = "pass"
FAIL = "fail"

JUDGMENT_ALIASES = {
    "entry": {"entry", "entry_no", "entryno", "no", "编号", "条目", "条目号"},
    "verdict": {"verdict", "judgment", "decision", "result", "判定", "结果", "结论"},
}

PASS_ALIASES = {"pass", "p", "yes", "y", "true", "通过", "合格"}
FAIL_ALIASES = {"fail", "f", "no", "n", "false", "不通过", "不合格"}

VERDICT_LABELS = {PASS: "通过", FAIL: "不通过"}


def unique_top_hits(candidates: Iterable[Candidate]) -> list[Candidate]:
    """Keep each spectrum's unique top-scoring candidate; drop score ties."""
    by_key: dict[SpectrumKey, list[Candidate]] = {}
    for candidate in candidates:
        by_key.setdefault(candidate.key, []).append(candidate)
    tops: list[Candidate] = []
    for group in by_key.values():
        top_score = max(candidate.score for candidate in group)
        winners = [c for c in group if c.score == top_score]
        if len(winners) == 1:
            tops.append(winners[0])
    return tops


@dataclass(frozen=True)
class RoundPlan:
    """Sampling plan for one round.

    ``chosen`` is empty when no target+decoy pair can be formed; the pool
    sizes are still reported so the operator sees why opening failed.
    """

    chosen: tuple[Candidate, ...]
    pool_targets: int
    pool_decoys: int

    @property
    def pair_count(self) -> int:
        return len(self.chosen) // 2


def plan_round(
    candidates: Iterable[Candidate],
    used_keys: set[SpectrumKey],
    rng: random.Random,
    max_pairs: int = MAX_PAIRS_PER_ROUND,
) -> RoundPlan:
    """Pick up to ``max_pairs`` targets and the same number of decoys.

    Only unique top hits whose spectrum was never drawn before (in any round
    of the same candidate revision, including still-open rounds) are
    eligible.  The pair count is limited by the smaller remaining class.
    """
    tops = unique_top_hits(candidates)
    targets = [
        c for c in tops if c.label == TARGET and c.key not in used_keys
    ]
    decoys = [c for c in tops if c.label == DECOY and c.key not in used_keys]
    pair_count = min(len(targets), len(decoys), max_pairs)
    if pair_count < 1:
        return RoundPlan((), len(targets), len(decoys))
    chosen = rng.sample(targets, pair_count) + rng.sample(decoys, pair_count)
    return RoundPlan(tuple(chosen), len(targets), len(decoys))


def random_entry_numbers(count: int, rng: random.Random) -> list[int]:
    """Distinct random entry numbers for one round (order arbitrary)."""
    return rng.sample(range(1, ENTRY_NO_SPACE), count)


def shuffled_rows(rows: Sequence[object], rng: random.Random) -> list[object]:
    """Return the rows in random order (for the downloadable blind sheet)."""
    ordered = list(rows)
    rng.shuffle(ordered)
    return ordered


@dataclass(frozen=True)
class JudgmentRow:
    entry_no: int
    verdict: str  # PASS or FAIL


def _normalise_verdict(value: str, filename: str, line_no: int) -> str:
    text = value.strip().lower().replace(" ", "")
    if text in PASS_ALIASES:
        return PASS
    if text in FAIL_ALIASES:
        return FAIL
    raise ParseError(
        f"{filename} 第 {line_no} 行：判定只能为 通过/不通过（或 pass/fail），"
        f"实际为 {value!r}"
    )


def parse_judgments(data: bytes, filename: str = "judgments.csv") -> list[JudgmentRow]:
    """Parse a judgment batch: columns 条目号 + 判定, one row per entry."""
    text = _decode(data)
    reader = csv.reader(io.StringIO(text))
    try:
        header = next(reader)
    except StopIteration:
        raise ParseError(f"{filename}：文件为空")
    mapping = _map_header(
        header, JUDGMENT_ALIASES, ("entry", "verdict"), filename
    )

    rows: list[JudgmentRow] = []
    for line_no, row in enumerate(reader, start=2):
        if not row or all(not cell.strip() for cell in row):
            continue
        entry_text = _cell(row, mapping["entry"])
        verdict_text = _cell(row, mapping["verdict"])
        try:
            entry_no = int(entry_text)
        except ValueError:
            raise ParseError(
                f"{filename} 第 {line_no} 行：条目号不是整数（{entry_text!r}）"
            )
        if entry_no < 1:
            raise ParseError(
                f"{filename} 第 {line_no} 行：条目号必须为正整数（{entry_text!r}）"
            )
        rows.append(
            JudgmentRow(
                entry_no=entry_no,
                verdict=_normalise_verdict(verdict_text, filename, line_no),
            )
        )
    if not rows:
        raise ParseError(f"{filename}：没有任何判定行")
    return rows
