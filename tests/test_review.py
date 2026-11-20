import csv
import io
import random
import threading

import pytest
from fastapi.testclient import TestClient

from app import db, review, review_service, service
from app.compute import Candidate, SpectrumKey
from app.csvio import ParseError
from app.main import create_app
from decimal import Decimal


CAND_HEADER = "sample,spectrum,peptide,target_decoy,score\n"


def candidates_csv(*rows):
    buf = io.StringIO()
    buf.write(CAND_HEADER)
    for sample, spectrum, peptide, label, score in rows:
        buf.write(f"{sample},{spectrum},{peptide},{label},{score}\n")
    return buf.getvalue().encode("utf-8")


def pool_candidates(n_targets, n_decoys, sample="S1"):
    """n_targets target top hits + n_decoys decoy top hits, one per spectrum."""
    rows = []
    for i in range(1, n_targets + 1):
        rows.append((sample, f"T{i}", f"PEP_T{i}", "target", str(1000 - i)))
    for i in range(1, n_decoys + 1):
        rows.append((sample, f"D{i}", f"PEP_D{i}", "decoy", str(2000 - i)))
    return candidates_csv(*rows)


def judgments_csv(*rows):
    buf = io.StringIO()
    buf.write("条目号,判定\n")
    for entry_no, verdict in rows:
        buf.write(f"{entry_no},{verdict}\n")
    return buf.getvalue().encode("utf-8")


@pytest.fixture()
def harness(tmp_path):
    db_path = tmp_path / "release.db"
    app = create_app(db_path, start_worker=False)
    with TestClient(app) as client:
        client.db_path = db_path
        yield client


def upload_candidates(client, revision, payload):
    return client.post(
        "/upload/candidates",
        data={"revision": str(revision)},
        files={"file": ("candidates.csv", payload, "text/csv")},
        follow_redirects=False,
    )


def query(client, sql, args=()):
    conn = db.connect(client.db_path)
    try:
        return conn.execute(sql, args).fetchall()
    finally:
        conn.close()


def round_entries(client, round_id):
    return query(
        client,
        "SELECT * FROM review_entries WHERE round_id = ? ORDER BY entry_no",
        (round_id,),
    )


def judged_count(client, round_id):
    row = query(
        client,
        "SELECT COUNT(*) AS n FROM review_judgments j "
        "JOIN review_entries e ON e.id = j.entry_id WHERE e.round_id = ?",
        (round_id,),
    )[0]
    return row["n"]


def open_round_http(client):
    return client.post("/blind/open", follow_redirects=False)


def submit_http(client, round_id, payload, filename="judgments.csv"):
    return client.post(
        f"/blind/{round_id}/judgments",
        files={"file": (filename, payload, "text/csv")},
        follow_redirects=False,
    )


def cand(spectrum, label, score="10", sample="S1", peptide="P"):
    return Candidate(
        key=SpectrumKey(sample, spectrum),
        peptide=peptide,
        label=label,
        score=Decimal(score),
    )


# ---------------------------------------------------------------------------
# Pure logic
# ---------------------------------------------------------------------------


def test_unique_top_hits_skips_ties():
    candidates = [
        cand("A", "target", "10"),
        cand("B", "target", "9"),
        Candidate(
            key=SpectrumKey("S1", "B"), peptide="Q", label="decoy",
            score=Decimal("9"),
        ),
        cand("C", "decoy", "5"),
    ]
    tops = review.unique_top_hits(candidates)
    assert {c.key.spectrum for c in tops} == {"A", "C"}


def test_plan_round_pairs_up_to_ten_and_excludes_used():
    rng = random.Random(7)
    candidates = [cand(f"T{i}", "target", str(100 - i)) for i in range(1, 15)]
    candidates += [cand(f"D{i}", "decoy", str(200 - i)) for i in range(1, 15)]
    plan = review.plan_round(candidates, set(), rng)
    assert plan.pair_count == 10
    assert sum(1 for c in plan.chosen if c.label == "target") == 10
    assert sum(1 for c in plan.chosen if c.label == "decoy") == 10

    # Fewer decoys remaining -> pair count follows the smaller class.
    used = {SpectrumKey("S1", f"D{i}") for i in range(1, 12)}
    plan = review.plan_round(candidates, used, rng)
    assert plan.pair_count == 3
    assert plan.pool_decoys == 3
    assert plan.pool_targets == 14
    assert all(c.key not in used for c in plan.chosen)


def test_plan_round_needs_a_pair():
    rng = random.Random(1)
    only_targets = [cand(f"T{i}", "target") for i in range(1, 6)]
    plan = review.plan_round(only_targets, set(), rng)
    assert plan.pair_count == 0
    assert plan.chosen == ()
    plan = review.plan_round([], set(), rng)
    assert plan.pair_count == 0


def test_random_entry_numbers_are_unique_and_in_space():
    rng = random.Random(3)
    numbers = review.random_entry_numbers(40, rng)
    assert len(set(numbers)) == 40
    assert all(1 <= n < review.ENTRY_NO_SPACE for n in numbers)


def test_parse_judgments_accepts_chinese_and_english():
    payload = "条目号,判定\n5,通过\n6,不通过\n".encode("utf-8")
    rows = review.parse_judgments(payload)
    assert rows == [
        review.JudgmentRow(5, "pass"),
        review.JudgmentRow(6, "fail"),
    ]
    payload = b"entry,verdict\n1,pass\n2,FAIL\n"
    rows = review.parse_judgments(payload)
    assert [r.verdict for r in rows] == ["pass", "fail"]


def test_parse_judgments_rejects_bad_input():
    with pytest.raises(ParseError):
        review.parse_judgments(b"")
    with pytest.raises(ParseError):
        review.parse_judgments("条目号,判定\n".encode("utf-8"))
    with pytest.raises(ParseError):
        review.parse_judgments("条目号,判定\nabc,通过\n".encode("utf-8"))
    with pytest.raises(ParseError):
        review.parse_judgments("条目号,判定\n1,也许\n".encode("utf-8"))
    with pytest.raises(ParseError):
        review.parse_judgments("条目号\n1\n".encode("utf-8"))


# ---------------------------------------------------------------------------
# Opening rounds
# ---------------------------------------------------------------------------


def test_open_round_happy_path(harness):
    client = harness
    upload_candidates(client, 1, pool_candidates(12, 12))
    response = open_round_http(client)
    assert response.status_code == 303

    rounds = query(client, "SELECT * FROM review_rounds")
    assert len(rounds) == 1
    round_id = rounds[0]["id"]
    assert rounds[0]["pair_count"] == 10
    assert rounds[0]["status"] == "open"

    entries = round_entries(client, round_id)
    assert len(entries) == 20
    labels = [e["label"] for e in entries]
    assert labels.count("target") == 10
    assert labels.count("decoy") == 10
    numbers = [e["entry_no"] for e in entries]
    assert len(set(numbers)) == 20

    sheet = client.get(f"/blind/{round_id}/sheet.csv")
    assert sheet.status_code == 200
    rows = list(csv.reader(io.StringIO(sheet.content.decode("utf-8-sig"))))
    assert rows[0] == ["条目号", "样本号", "谱图号", "肽段", "分数"]
    assert len(rows) == 21
    assert {int(r[0]) for r in rows[1:]} == set(numbers)
    # The blind sheet must not leak labels anywhere.
    assert "靶标" not in sheet.text and "decoy" not in sheet.text.lower()


def test_open_round_without_candidates_fails(harness):
    client = harness
    response = open_round_http(client)
    assert response.status_code == 400
    assert "无法开轮" in response.text
    assert query(client, "SELECT COUNT(*) AS n FROM review_rounds")[0]["n"] == 0


def test_open_round_without_pair_leaves_no_half_round(harness):
    client = harness
    upload_candidates(client, 1, pool_candidates(10, 0))  # targets only
    response = open_round_http(client)
    assert response.status_code == 400
    assert "无法开轮" in response.text
    assert "凑不出一对" in response.text
    assert query(client, "SELECT COUNT(*) AS n FROM review_rounds")[0]["n"] == 0
    assert query(client, "SELECT COUNT(*) AS n FROM review_entries")[0]["n"] == 0


def test_tied_spectrum_is_never_sampled(harness):
    client = harness
    rows = [
        ("S1", "TIE", "PEP_A", "target", "500"),
        ("S1", "TIE", "PEP_B", "decoy", "500.0"),  # tie at the top score
    ]
    payload = candidates_csv(*rows) + pool_candidates(3, 3).split(b"\n", 1)[1]
    upload_candidates(client, 1, payload)
    response = open_round_http(client)
    assert response.status_code == 303
    entries = round_entries(client, 1)
    assert len(entries) == 6
    assert all(e["spectrum"] != "TIE" for e in entries)


def test_only_unique_top_label_enters_pool(harness):
    client = harness
    # Spectrum X: decoy outranks target -> decoy pool (with the decoy
    # peptide).  Spectrum Y: target on top -> target pool.  Exactly one pair.
    payload = candidates_csv(
        ("S1", "X", "PEP_DX", "decoy", "100"),
        ("S1", "X", "PEP_TX", "target", "50"),
        ("S1", "Y", "PEP_TY", "target", "90"),
    )
    upload_candidates(client, 1, payload)
    assert open_round_http(client).status_code == 303
    entries = round_entries(client, 1)
    assert len(entries) == 2
    by_spectrum = {e["spectrum"]: e for e in entries}
    assert by_spectrum["X"]["label"] == "decoy"
    assert by_spectrum["X"]["peptide"] == "PEP_DX"
    assert by_spectrum["Y"]["label"] == "target"


def test_second_round_excludes_spectra_from_open_round(harness):
    client = harness
    upload_candidates(client, 1, pool_candidates(10, 10))
    assert open_round_http(client).status_code == 303
    # Round 1 is still collecting answers, but its spectra already count.
    response = open_round_http(client)
    assert response.status_code == 400
    assert "无法开轮" in response.text
    assert query(client, "SELECT COUNT(*) AS n FROM review_rounds")[0]["n"] == 1


def test_rounds_on_same_revision_are_disjoint(harness):
    client = harness
    upload_candidates(client, 1, pool_candidates(15, 15))
    assert open_round_http(client).status_code == 303
    assert open_round_http(client).status_code == 303
    keys1 = {(e["sample"], e["spectrum"]) for e in round_entries(client, 1)}
    keys2 = {(e["sample"], e["spectrum"]) for e in round_entries(client, 2)}
    assert keys1.isdisjoint(keys2)
    assert len(keys1) == 20
    assert len(keys2) == 10  # only 5+5 remained for the second round


def test_new_candidate_revision_starts_fresh_pool(harness):
    client = harness
    upload_candidates(client, 1, pool_candidates(10, 10))
    assert open_round_http(client).status_code == 303
    upload_candidates(client, 2, pool_candidates(10, 10))  # same spectra, new revision
    response = open_round_http(client)
    assert response.status_code == 303
    assert query(client, "SELECT COUNT(*) AS n FROM review_rounds")[0]["n"] == 2


def test_concurrent_open_leaves_consistent_state(tmp_path):
    db_path = tmp_path / "concurrent.db"
    app = create_app(db_path, start_worker=False)
    with TestClient(app):
        pass
    service.upload_source(
        db_path, "candidates", "1", pool_candidates(10, 10), "c.csv"
    )

    barrier = threading.Barrier(2)
    results, errors = [], []

    def worker():
        try:
            barrier.wait(timeout=10)
            results.append(review_service.open_round(db_path))
        except service.ServiceError as exc:
            errors.append(str(exc))

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
        assert not thread.is_alive()

    # Exactly one round could be formed from a 10+10 pool; the loser gets a
    # clean "cannot open" error and no half round remains.
    assert len(results) == 1
    assert len(errors) == 1
    assert "无法开轮" in errors[0]
    conn = db.connect(db_path)
    try:
        rounds = conn.execute("SELECT * FROM review_rounds").fetchall()
        assert len(rounds) == 1
        entries = conn.execute("SELECT * FROM review_entries").fetchall()
        assert len(entries) == 20
        assert len({(e["sample"], e["spectrum"]) for e in entries}) == 20
    finally:
        conn.close()


def test_concurrent_open_with_larger_pool_both_succeed_disjoint(tmp_path):
    db_path = tmp_path / "concurrent2.db"
    app = create_app(db_path, start_worker=False)
    with TestClient(app):
        pass
    service.upload_source(
        db_path, "candidates", "1", pool_candidates(30, 30), "c.csv"
    )

    barrier = threading.Barrier(3)
    results, errors = [], []

    def worker():
        try:
            barrier.wait(timeout=10)
            results.append(review_service.open_round(db_path))
        except service.ServiceError as exc:
            errors.append(str(exc))

    threads = [threading.Thread(target=worker) for _ in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
        assert not thread.is_alive()

    assert not errors
    assert len(results) == 3
    conn = db.connect(db_path)
    try:
        entries = conn.execute("SELECT * FROM review_entries").fetchall()
        assert len(entries) == 60
        keys = [(e["sample"], e["spectrum"]) for e in entries]
        assert len(set(keys)) == 60  # no spectrum drawn twice
        numbers = [e["entry_no"] for e in entries]
        # entry numbers unique within each round
        for round_id in results:
            round_numbers = [
                e["entry_no"] for e in entries if e["round_id"] == round_id
            ]
            assert len(set(round_numbers)) == 20
        assert len(numbers) == 60
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Submitting judgments
# ---------------------------------------------------------------------------


def open_round_with_pool(client, targets=10, decoys=10, revision=1):
    upload_candidates(client, revision, pool_candidates(targets, decoys))
    assert open_round_http(client).status_code == 303
    return query(client, "SELECT MAX(id) AS m FROM review_rounds")[0]["m"]


def entry_map(client, round_id):
    return {e["entry_no"]: e for e in round_entries(client, round_id)}


def test_submit_batches_until_complete_then_labels_revealed(harness):
    client = harness
    round_id = open_round_with_pool(client)
    entries = entry_map(client, round_id)
    targets = [n for n, e in entries.items() if e["label"] == "target"]
    decoys = [n for n, e in entries.items() if e["label"] == "decoy"]

    # Before completion: labels hidden on the page.
    page = client.get(f"/blind/{round_id}")
    assert page.status_code == 200
    assert ">靶标<" not in page.text and ">诱饵<" not in page.text
    assert "标签保密中" in page.text

    # Batch 1: all targets pass, two decoys (wrongly) pass as well.
    batch1 = [(n, "通过") for n in targets] + [(n, "通过") for n in decoys[:2]]
    response = submit_http(client, round_id, judgments_csv(*batch1))
    assert response.status_code == 303
    assert judged_count(client, round_id) == 12

    # Batch 2: the remaining decoys fail.
    batch2 = [(n, "不通过") for n in decoys[2:]]
    response = submit_http(client, round_id, judgments_csv(*batch2))
    assert response.status_code == 303
    assert judged_count(client, round_id) == 20

    round_row = query(client, "SELECT * FROM review_rounds WHERE id = ?", (round_id,))[0]
    assert round_row["status"] == "completed"
    assert round_row["completed_at"] is not None

    page = client.get(f"/blind/{round_id}")
    assert ">靶标<" in page.text and ">诱饵<" in page.text
    assert "2/10" in page.text  # 诱饵通过 2/10
    assert "10/10" in page.text  # 靶标通过 10/10


def test_same_judgment_resent_counts_once(harness):
    client = harness
    round_id = open_round_with_pool(client)
    entries = entry_map(client, round_id)
    batch = [(n, "通过") for n in list(entries)[:5]]
    payload = judgments_csv(*batch)
    assert submit_http(client, round_id, payload).status_code == 303
    assert judged_count(client, round_id) == 5

    # Resend the identical batch: accepted, but nothing new is recorded.
    response = submit_http(client, round_id, payload)
    assert response.status_code == 303
    assert judged_count(client, round_id) == 5
    batch_row = query(
        client, "SELECT * FROM review_batches ORDER BY id DESC LIMIT 1"
    )[0]
    assert batch_row["received"] == 5
    assert batch_row["applied"] == 0

    # Duplicated rows inside one batch also count once.
    doubled = judgments_csv(*(batch + batch))
    assert submit_http(client, round_id, doubled).status_code == 303
    assert judged_count(client, round_id) == 5


def test_conflicting_judgment_rejects_whole_batch(harness):
    client = harness
    round_id = open_round_with_pool(client)
    entries = list(entry_map(client, round_id))
    first, second, third = entries[0], entries[1], entries[2]
    assert submit_http(client, round_id, judgments_csv((first, "通过"))).status_code == 303

    # 'first' now conflicts; the valid rows in the same batch must not land.
    response = submit_http(
        client,
        round_id,
        judgments_csv((first, "不通过"), (second, "通过"), (third, "通过")),
    )
    assert response.status_code == 400
    assert "冲突" in response.text
    assert judged_count(client, round_id) == 1
    assert query(client, "SELECT COUNT(*) AS n FROM review_batches")[0]["n"] == 1

    # Conflict inside the batch itself.
    response = submit_http(
        client,
        round_id,
        judgments_csv((second, "通过"), (second, "不通过")),
    )
    assert response.status_code == 400
    assert judged_count(client, round_id) == 1


def test_wrong_round_entry_rejects_whole_batch(harness):
    client = harness
    upload_candidates(client, 1, pool_candidates(25, 25))
    assert open_round_http(client).status_code == 303
    assert open_round_http(client).status_code == 303
    entries1 = entry_map(client, 1)
    entries2 = entry_map(client, 2)
    foreign = next(n for n in entries1 if n not in entries2)

    good = next(iter(entries2))
    response = submit_http(
        client, 2, judgments_csv((good, "通过"), (foreign, "通过"))
    )
    assert response.status_code == 400
    assert "错轮条目" in response.text
    # The whole batch is rejected: even the valid entry was not recorded.
    assert judged_count(client, 2) == 0
    assert query(client, "SELECT COUNT(*) AS n FROM review_batches")[0]["n"] == 0


def test_invalid_judgment_file_leaves_no_trace(harness):
    client = harness
    round_id = open_round_with_pool(client)
    response = submit_http(client, round_id, b"entry,verdict\n1,maybe\n")
    assert response.status_code == 400
    assert judged_count(client, round_id) == 0
    assert query(client, "SELECT COUNT(*) AS n FROM review_batches")[0]["n"] == 0


def test_unknown_round_returns_404(harness):
    client = harness
    assert client.get("/blind/99").status_code == 404
    assert client.get("/blind/99/sheet.csv").status_code == 404
    assert submit_http(client, 99, judgments_csv((1, "通过"))).status_code == 404


def test_revision_change_keeps_original_sampling_for_submissions(harness):
    client = harness
    round_id = open_round_with_pool(client, revision=1)
    entries = entry_map(client, round_id)

    # Candidate list moves to revision 2; round 1 must still accept batches
    # against its original sampling.
    upload_candidates(client, 2, pool_candidates(10, 10))
    numbers = list(entries)
    half = len(numbers) // 2
    assert submit_http(
        client, round_id, judgments_csv(*[(n, "通过") for n in numbers[:half]])
    ).status_code == 303
    assert judged_count(client, round_id) == half
    assert submit_http(
        client, round_id, judgments_csv(*[(n, "不通过") for n in numbers[half:]])
    ).status_code == 303
    assert judged_count(client, round_id) == len(numbers)
    round_row = query(client, "SELECT * FROM review_rounds WHERE id = ?", (round_id,))[0]
    assert round_row["status"] == "completed"

    # Entries still match the revision-1 sampling.
    stored = {(e["sample"], e["spectrum"], e["peptide"]) for e in entries.values()}
    assert len(stored) == 20


def test_restart_still_accepts_submissions(tmp_path):
    db_path = tmp_path / "restart.db"
    app = create_app(db_path, start_worker=False)
    with TestClient(app) as client:
        client.db_path = db_path
        round_id = open_round_with_pool(client)
        entries = entry_map(client, round_id)
        numbers = sorted(entries)
        assert submit_http(
            client, round_id, judgments_csv(*[(n, "通过") for n in numbers[:10]])
        ).status_code == 303

    # Simulate a process restart: a brand-new app on the same database file.
    app2 = create_app(db_path, start_worker=False)
    with TestClient(app2) as client2:
        client2.db_path = db_path
        response = submit_http(
            client2, round_id, judgments_csv(*[(n, "不通过") for n in numbers[10:]])
        )
        assert response.status_code == 303
        assert judged_count(client2, round_id) == 20
        page = client2.get(f"/blind/{round_id}")
        assert "已收齐" in page.text


def test_blind_list_page_shows_rounds(harness):
    client = harness
    page = client.get("/blind")
    assert page.status_code == 200
    assert "尚无盲审轮次" in page.text
    round_id = open_round_with_pool(client)
    page = client.get("/blind")
    assert f"#{round_id}" in page.text
    assert "0/20" in page.text
    assert "进行中" in page.text
