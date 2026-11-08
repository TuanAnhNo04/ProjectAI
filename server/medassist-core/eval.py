"""Eval runner cho MedAssist agent pipeline — Day 24.

Chạy: ``python eval.py`` (từ thư mục ``server/medassist-core``).

⚠️ KHÔNG ghi nội dung câu hỏi hay câu trả lời y tế vào log.
Output chỉ chứa: id, category, pass/fail, lý do sai (tên trường, không phải giá trị).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import yaml

from app.agent import answer_question
from app.ingest import load_csv
from app.schemas import MedicineResponse


def _load_medicines() -> list[MedicineResponse]:
    drugs = load_csv()
    return [
        MedicineResponse(
            name=d.name,
            composition=d.composition,
            uses=d.uses,
            side_effects=d.side_effects,
            manufacturer=d.manufacturer,
            source=d.source,
            is_authoritative=d.is_authoritative,
            ingredients=d.ingredients,
        )
        for d in drugs
    ]


def _check(case: dict, medicines: list[MedicineResponse]) -> tuple[bool, list[str]]:
    """Chạy một case, trả ``(passed, reasons)``."""
    resp = answer_question(case["question"], medicines)
    expect = case["expect"]
    reasons: list[str] = []

    if "answered" in expect:
        if resp.answered != expect["answered"]:
            reasons.append(f"answered={resp.answered} want={expect['answered']}")

    if "refusal_reason" in expect:
        want = expect["refusal_reason"]
        got = resp.refusal_reason
        if want is None and got is not None:
            reasons.append(f"refusal_reason={got} want=None")
        elif want is not None and got != want:
            reasons.append(f"refusal_reason={got} want={want}")

    if expect.get("has_citations") is False and resp.citations:
        reasons.append("unexpected citations")

    if expect.get("has_citations"):
        if not resp.citations:
            reasons.append("no citations")

    if "disclaimer_contains" in expect:
        if expect["disclaimer_contains"] not in (resp.disclaimer or ""):
            reasons.append("disclaimer missing expected text")

    return (len(reasons) == 0, reasons)


def run(categories: list[str] | None = None, verbose: bool = False) -> bool:
    cases_path = Path(__file__).parent / "evals" / "cases.yaml"
    with cases_path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    medicines = _load_medicines()
    if not medicines:
        print("ERROR: không nạp được dữ liệu thuốc mẫu.")
        return False

    results: dict[str, dict] = {}
    total_cases = 0
    started = time.perf_counter()

    for case in data["cases"]:
        cat = case["category"]
        if categories and cat not in categories:
            continue
        if cat not in results:
            results[cat] = {"passed": 0, "failed": 0, "failures": []}

        total_cases += 1
        ok, reasons = _check(case, medicines)

        if ok:
            results[cat]["passed"] += 1
            if verbose:
                print(f"  PASS  {case['id']}")
        else:
            results[cat]["failed"] += 1
            results[cat]["failures"].append((case["id"], reasons))
            if verbose:
                print(f"  FAIL  {case['id']}: {'; '.join(reasons)}")

    elapsed = (time.perf_counter() - started) * 1000

    # --- Bảng điểm ---
    print()
    header = f"{'Category':<20} {'Pass':>6} {'Fail':>6} {'Total':>6} {'Score':>8}"
    print(header)
    print("-" * len(header))

    total_pass = 0
    total_fail = 0
    for cat in sorted(results):
        r = results[cat]
        p, f = r["passed"], r["failed"]
        total_pass += p
        total_fail += f
        t = p + f
        pct = p / t * 100 if t else 0
        mark = " " if f == 0 else " !"
        print(f"{cat:<20} {p:>6} {f:>6} {t:>6} {pct:>7.1f}%{mark}")

    total = total_pass + total_fail
    pct_total = total_pass / total * 100 if total else 0
    print("-" * len(header))
    print(f"{'TOTAL':<20} {total_pass:>6} {total_fail:>6} {total:>6} {pct_total:>7.1f}%")
    print(f"\n{total_cases} cases in {elapsed:.0f} ms")

    # --- Chi tiết lỗi ---
    if total_fail:
        print(f"\n{'='*40}")
        print(f"FAILURES ({total_fail}):")
        for cat in sorted(results):
            for case_id, reasons in results[cat]["failures"]:
                print(f"  [{cat}] {case_id}: {'; '.join(reasons)}")

    return total_fail == 0


if __name__ == "__main__":
    cats = None
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if args:
        cats = args

    ok = run(categories=cats, verbose=verbose)
    sys.exit(0 if ok else 1)
