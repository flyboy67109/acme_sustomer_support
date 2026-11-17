"""
assert_check.py -- Step 3 of the five A's: ASSERT

Grade the matcher's suggestions against the hidden answer key, on tickets the
learner never trained on (the "test" tickets). Like a unit test, but instead of
pass/fail on one case we get a report card over many.

Questions the report answers:
  * How often is the top suggestion right?          (accuracy)
  * Is the right order at least in the top 3?       (what the rep would see)
  * When there is NO matching order, does it say so? (abstain)
  * How often is it CONFIDENTLY WRONG?              (the dangerous kind)
  * If we trust the machine above some confidence, how much work does it save
    and how often is it right?                      (where to set thresholds)

Run:  python assert_check.py
"""
from collections import defaultdict

from acme_common import (AUTO_THRESHOLD, NONE, load_answer_key,
                         load_system_data, read_csv, split_ticket_ids)

REPORTS = [
    ("BASELINE: hand-set weights", "suggestions_baseline.csv"),
    ("LEARNED: weights found in Adjust", "suggestions_learned.csv"),
]


def load_suggestions(filename):
    try:
        rows = read_csv(filename)
    except SystemExit:
        return None
    by_ticket = defaultdict(list)
    for row in sorted(rows, key=lambda r: int(r["rank"])):
        by_ticket[row["ticket_id"]].append((row["order_id"], float(row["probability"])))
    return by_ticket


def pct(part, whole):
    return f"{part / whole:6.1%}" if whole else "   n/a"


def grade(label, suggestions, key, test_ids, messages):
    results = []
    for ticket_id in sorted(test_ids):
        truth = key[ticket_id]["true_order_id"] or NONE
        ranked = suggestions[ticket_id]
        top, p = ranked[0]
        results.append({
            "id": ticket_id,
            "difficulty": key[ticket_id]["difficulty"],
            "truth": truth, "top": top, "p": p,
            "right": top == truth,
            "in_top3": truth in [order_id for order_id, _ in ranked],
        })
    n = len(results)

    print(f"\n{'=' * 70}\n{label}   ({n} held-out tickets)\n{'=' * 70}")
    for r in results:
        # Dangerous = a specific order suggested with high confidence that is
        # not the true one. Choosing NONE ("can't tell, ask for info") is safe.
        r["dangerous"] = r["top"] != NONE and not r["right"] and r["p"] >= AUTO_THRESHOLD
        r["gave_up"] = r["top"] == NONE and r["truth"] != NONE
    right = sum(r["right"] for r in results)
    top3 = sum(r["in_top3"] for r in results)
    gave_up = sum(r["gave_up"] for r in results)
    confident_wrong = [r for r in results if r["dangerous"]]
    print(f"Top suggestion is right:            {pct(right, n)}   ({right}/{n})")
    print(f"Right answer within the top 3:      {pct(top3, n)}   ({top3}/{n})")
    print(f"Said 'ask for info' (safe, no help): {pct(gave_up, n)}   ({gave_up}/{n})")
    print(f"Confidently wrong (>= {AUTO_THRESHOLD:.0%}):        "
          f"{pct(len(confident_wrong), n)}   ({len(confident_wrong)}/{n})   <- the dangerous kind")

    print("\nBy how hard the ticket was designed to be:")
    print("  level         right          said 'ask for info'   confidently wrong   avg confidence")
    for level in ("easy", "medium", "hard", "unmatchable"):
        rows = [r for r in results if r["difficulty"] == level]
        if rows:
            ok = sum(r["right"] for r in rows)
            gave = sum(r["top"] == NONE for r in rows)
            bad = sum(r["dangerous"] for r in rows)
            avg_p = sum(r["p"] for r in rows) / len(rows)
            print(f"  {level:<12} {ok:>3}/{len(rows):<3} {pct(ok, len(rows))}"
                  f"      {gave:>3}               {bad:>3}              {avg_p:6.1%}")

    print("\nIf the machine only acts when it is at least this confident:")
    print("  threshold   tickets it handles   of those, correct")
    for threshold in (0.5, 0.7, 0.9, 0.95):
        handled = [r for r in results if r["top"] != NONE and r["p"] >= threshold]
        good = sum(r["right"] for r in handled)
        print(f"     {threshold:.2f}         {pct(len(handled), n)}           {pct(good, len(handled))}")

    if confident_wrong:
        print(f"\nConfidently wrong tickets (look at these first):")
        for r in confident_wrong[:5]:
            print(f"  {r['id']}: suggested {r['top']} at {r['p']:.0%}, truth was {r['truth']}")
            print(f"      \"{messages[r['id']][:100]}...\"")
    return right / n


def main():
    data = load_system_data()
    key = load_answer_key()
    _, test_ids = split_ticket_ids(data["tickets"])
    messages = {t["ticket_id"]: t["message"] for t in data["tickets"]}

    scores = []
    for label, filename in REPORTS:
        suggestions = load_suggestions(filename)
        if suggestions is None:
            print(f"\n(No {filename} yet. It appears once that step has run.)")
            continue
        scores.append((label, grade(label, suggestions, key, test_ids, messages)))

    if len(scores) == 2:
        print(f"\nSummary: top-suggestion accuracy {scores[0][1]:.1%} -> {scores[1][1]:.1%}")


if __name__ == "__main__":
    main()
