"""
act_match.py -- Step 2 of the five A's: ACT

Run the matcher. For every ticket it ranks candidate orders, gives each a
probability, and writes the top 3 with plain-language reasons.

The WEIGHTS below are our first guess at how much each clue matters, chosen by
intuition ("a PO match is worth a lot, a date fit is worth a little"). In the
Adjust step the computer will learn better numbers from the tickets.

Run:  python act_match.py
"""
import csv

from acme_common import (DATA_DIR, build_cases, explain, load_system_data,
                         probabilities)

HAND_WEIGHTS = {
    "po_match": 5.0,
    "order_id_match": 6.0,
    "product_match": 1.5,
    "buyer_name_match": 2.0,
    "company_match": 1.5,
    "email_domain_match": 1.5,
    "date_fit": 1.0,
    "none_bias": 2.0,
    "unmatched_ref": 1.0,
}


def write_suggestions(cases, weights, filename):
    """Save the top 3 choices per ticket: what Assert will grade."""
    path = DATA_DIR / filename
    fields = ["ticket_id", "rank", "order_id", "probability", "reasons"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for case in cases:
            ranked = probabilities(case, weights)
            for rank, (order_id, p) in enumerate(ranked[:3], start=1):
                writer.writerow({
                    "ticket_id": case.ticket["ticket_id"],
                    "rank": rank,
                    "order_id": order_id,
                    "probability": round(p, 4),
                    "reasons": explain(case, order_id, weights),
                })
    return path


def run_act(weights, filename):
    cases = build_cases(load_system_data())
    path = write_suggestions(cases, weights, filename)
    return cases, path


def show_ticket(case, weights, ticket_id="T-0001"):
    if case.ticket["ticket_id"] != ticket_id:
        return
    print(f"\n{ticket_id}: {case.ticket['message']}")
    print(f"  Sender: {case.ticket['submitter_name']} <{case.ticket['submitter_email']}>")
    print(f"  Candidate orders considered: {len(case.candidates) - 1}")
    for order_id, p in probabilities(case, weights)[:3]:
        print(f"    {order_id:<10} {p:6.1%}   {explain(case, order_id, weights)}")


if __name__ == "__main__":
    cases, path = run_act(HAND_WEIGHTS, "suggestions_baseline.csv")
    print(f"Matched {len(cases)} tickets with the hand-set weights.")
    print(f"Saved the top 3 choices per ticket to {path}")
    for case in cases:
        show_ticket(case, HAND_WEIGHTS)
    print("\nNext: python assert_check.py   (grade these against the answer key)")
